package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/smhanov/llmhub"

	"github.com/smhanov/agentloop"
	"github.com/smhanov/agentloop/search"
)

// runResult is the unified result of one run across the three agents:
// the core trace plus the preset-specific extras that are not reachable
// from the core answer alone (the bash exit status; search sources are
// derived from the trace at print time).
type runResult struct {
	kind       string
	core       *agentloop.Answer
	exitStatus string
	submission string
}

// newRunResult builds a runResult from a core answer. The bash exit
// status is only available in non-streaming runs (the preset attaches it
// to its own Ask result, not to stream events), so streaming callers get
// a result without it.
func newRunResult(kind string, core *agentloop.Answer) *runResult {
	if core == nil {
		return nil
	}
	return &runResult{kind: kind, core: core}
}

// sources returns the search agent's deduped sources, extracted from the
// tool trace (a pure function, so it works for streamed runs too).
func (res *runResult) sources() []search.Source {
	if res.kind != agentKindSearch {
		return nil
	}
	return search.Sources(res.core)
}

// exitCodeFor maps a run error to the documented exit codes: 2 on a
// tripped budget, 1 otherwise.
func exitCodeFor(err error) int {
	if errors.Is(err, agentloop.ErrBudgetExhausted) {
		return 2
	}
	return 1
}

// statusString names an AnswerStatus for text and JSON output.
func statusString(s agentloop.AnswerStatus) string {
	switch s {
	case agentloop.StatusCompleted:
		return "completed"
	case agentloop.StatusBudgetExhausted:
		return "budget_exhausted"
	case agentloop.StatusStopped:
		return "stopped"
	}
	return fmt.Sprintf("unknown(%d)", int(s))
}

// printSummary writes the post-answer summary lines: the bash exit
// status and submission, the numbered search sources, and the usage cost
// when the provider reported one.
func printSummary(stdout io.Writer, res *runResult) {
	if res.kind == agentKindBash {
		if res.exitStatus != "" {
			fmt.Fprintf(stdout, "exit status: %s\n", res.exitStatus)
		}
		if submission := strings.TrimSpace(res.submission); submission != "" {
			fmt.Fprintf(stdout, "submission: %s\n", submission)
		}
	}
	if sources := res.sources(); len(sources) > 0 {
		fmt.Fprintln(stdout, "sources:")
		for i, s := range sources {
			fmt.Fprintf(stdout, "  [%d] %s %s\n", i+1, strings.TrimSpace(s.Title), s.URL)
		}
	}
	printCost(stdout, res.core.Usage, "")
}

// printCost writes the usage line when the provider reported a cost.
func printCost(w io.Writer, u llmhub.UsageMetadata, label string) {
	if u.Cost > 0 {
		fmt.Fprintf(w, "%susage: %d prompt tokens, %d completion tokens, cost $%.6f\n",
			label, u.PromptTokens, u.CompletionTokens, u.Cost)
	}
}

// printTrace writes the -v event trace to stderr: tool calls with
// durations, step and tool-call counts, exit status, and timing.
func printTrace(stderr io.Writer, res *runResult) {
	core := res.core
	for _, tr := range core.Tools {
		if tr.Err != nil {
			fmt.Fprintf(stderr, "trace: tool %s failed after %v: %v\n", tr.Name, tr.Duration, tr.Err)
			continue
		}
		fmt.Fprintf(stderr, "trace: tool %s ok (%v)\n", tr.Name, tr.Duration)
	}
	line := fmt.Sprintf("trace: steps=%d tool calls=%d status=%s", core.Steps, core.ToolCalls, statusString(core.Status))
	if res.exitStatus != "" {
		line += " exit_status=" + res.exitStatus
	}
	if core.Timing != nil {
		line += fmt.Sprintf(" ttft=%v total=%v", core.Timing.FirstToken, core.Timing.Total)
	}
	fmt.Fprintln(stderr, line)
}

// streamPrinter consumes an AskStream event channel, printing answer
// deltas to stdout, reasoning deltas dimmed to stderr, and inline tool
// notes to stdout as they arrive.
type streamPrinter struct {
	stdout   io.Writer
	stderr   io.Writer
	verbose  bool
	lineOpen bool // last stdout delta had no trailing newline
}

// consume drains events until the channel closes and returns the final
// answer (present on Done and on budget exhaustion) plus the run error.
func (p *streamPrinter) consume(events <-chan agentloop.Event) (*agentloop.Answer, error) {
	var (
		final  *agentloop.Answer
		runErr error
	)
	for ev := range events {
		switch ev.Kind {
		case agentloop.EventAnswerDelta:
			fmt.Fprint(p.stdout, ev.Delta)
			p.lineOpen = !strings.HasSuffix(ev.Delta, "\n")
		case agentloop.EventReasoningDelta:
			// Reasoning is dimmed on stderr so the two channels stay
			// separable.
			fmt.Fprintf(p.stderr, "\x1b[2m%s\x1b[22m", ev.Delta)
		case agentloop.EventAfterToolCall:
			p.breakLine()
			name := ""
			if ev.Call != nil {
				name = ev.Call.Name
			}
			if ev.Err != nil {
				fmt.Fprintf(p.stdout, "[tool] %s failed: %v (%v)\n", name, ev.Err, ev.Duration)
			} else {
				fmt.Fprintf(p.stdout, "[tool] %s (%v)\n", name, ev.Duration)
			}
			if p.verbose {
				fmt.Fprintf(p.stderr, "trace: tool %s (%v)\n", name, ev.Duration)
			}
		case agentloop.EventAfterGenerate:
			if p.verbose {
				fmt.Fprintf(p.stderr, "trace: step %d\n", ev.Step)
			}
		case agentloop.EventDone:
			final = ev.Answer
		case agentloop.EventError:
			runErr = ev.Err
			if ev.Answer != nil {
				final = ev.Answer // partial answer alongside the error
			}
			if p.verbose {
				fmt.Fprintf(p.stderr, "trace: error: %v\n", ev.Err)
			}
		}
	}
	return final, runErr
}

// breakLine closes a delta-only line so following notes start clean.
func (p *streamPrinter) breakLine() {
	if p.lineOpen {
		fmt.Fprintln(p.stdout)
		p.lineOpen = false
	}
}

// endLine terminates the final streamed line, if one is open.
func (p *streamPrinter) endLine() { p.breakLine() }

// JSON output shapes. The library answer types are plain structs without
// JSON tags, so the CLI marshals this view instead of changing library
// types.
type usageJSON struct {
	PromptTokens        int     `json:"prompt_tokens"`
	CompletionTokens    int     `json:"completion_tokens"`
	TotalTokens         int     `json:"total_tokens"`
	CacheReadTokens     int     `json:"cache_read_tokens"`
	CacheCreationTokens int     `json:"cache_creation_tokens"`
	ReasoningTokens     int     `json:"reasoning_tokens"`
	Cost                float64 `json:"cost"`
}

type timingJSON struct {
	ConnectedMS  float64 `json:"connected_ms"`
	FirstByteMS  float64 `json:"first_byte_ms"`
	FirstTokenMS float64 `json:"first_token_ms"`
	TotalMS      float64 `json:"total_ms"`
	Chunks       int     `json:"chunks"`
}

type sourceJSON struct {
	Title     string `json:"title"`
	URL       string `json:"url"`
	Snippet   string `json:"snippet,omitempty"`
	Date      string `json:"date,omitempty"`
	Publisher string `json:"publisher,omitempty"`
	Fetched   bool   `json:"fetched"`
}

type toolJSON struct {
	Name       string  `json:"name"`
	ID         string  `json:"id,omitempty"`
	Args       string  `json:"args,omitempty"`
	Output     string  `json:"output,omitempty"`
	Error      string  `json:"error,omitempty"`
	DurationMS float64 `json:"duration_ms"`
}

type answerJSON struct {
	Agent      string       `json:"agent"`
	Text       string       `json:"text"`
	Reasoning  string       `json:"reasoning,omitempty"`
	Steps      int          `json:"steps"`
	ToolCalls  int          `json:"tool_calls"`
	Status     string       `json:"status"`
	ExitStatus string       `json:"exit_status,omitempty"`
	Submission string       `json:"submission,omitempty"`
	Sources    []sourceJSON `json:"sources,omitempty"`
	Usage      usageJSON    `json:"usage"`
	Timing     *timingJSON  `json:"timing,omitempty"`
	Tools      []toolJSON   `json:"tools,omitempty"`
}

func ms(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }

// buildAnswerJSON renders one run result as an indented JSON document
// ending in a newline.
func buildAnswerJSON(res *runResult) ([]byte, error) {
	core := res.core
	out := answerJSON{
		Agent:      res.kind,
		Text:       core.Text,
		Reasoning:  core.ReasoningText,
		Steps:      core.Steps,
		ToolCalls:  core.ToolCalls,
		Status:     statusString(core.Status),
		ExitStatus: res.exitStatus,
		Submission: res.submission,
		Usage: usageJSON{
			PromptTokens:        core.Usage.PromptTokens,
			CompletionTokens:    core.Usage.CompletionTokens,
			TotalTokens:         core.Usage.TotalTokens,
			CacheReadTokens:     core.Usage.CacheReadTokens,
			CacheCreationTokens: core.Usage.CacheCreationTokens,
			ReasoningTokens:     core.Usage.ReasoningTokens,
			Cost:                core.Usage.Cost,
		},
	}
	if core.Timing != nil {
		out.Timing = &timingJSON{
			ConnectedMS:  ms(core.Timing.Connected),
			FirstByteMS:  ms(core.Timing.FirstByte),
			FirstTokenMS: ms(core.Timing.FirstToken),
			TotalMS:      ms(core.Timing.Total),
			Chunks:       core.Timing.Chunks,
		}
	}
	for _, s := range res.sources() {
		out.Sources = append(out.Sources, sourceJSON{
			Title: s.Title, URL: s.URL, Snippet: s.Snippet,
			Date: s.Date, Publisher: s.Publisher, Fetched: s.Fetched,
		})
	}
	for _, tr := range core.Tools {
		tool := toolJSON{
			Name: tr.Name, ID: tr.ID,
			Args:       strings.TrimSpace(string(tr.Args)),
			Output:     tr.Output,
			DurationMS: ms(tr.Duration),
		}
		if tr.Err != nil {
			tool.Error = tr.Err.Error()
		}
		out.Tools = append(out.Tools, tool)
	}
	b, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(b, '\n'), nil
}

// printJSON writes the JSON document for one run, or a short note to
// stderr when marshaling fails.
func printJSON(stdout io.Writer, res *runResult) {
	b, err := buildAnswerJSON(res)
	if err != nil {
		fmt.Fprintf(stdout, "// json encode failed: %v\n", err)
		return
	}
	stdout.Write(b)
}

// confirmingApprover returns a bash approval gate that prompts on stderr
// and reads the answer from stdin: y/yes approves, anything else (or end
// of input) vetoes the command.
func confirmingApprover(stdin io.Reader, stderr io.Writer) func(ctx context.Context, command string) error {
	r := newLineReader(stdin)
	return func(_ context.Context, command string) error {
		fmt.Fprintf(stderr, "run command? [y/N] %s\n", command)
		line, err := r()
		if err != nil && line == "" {
			return errors.New("end of input before confirmation")
		}
		switch strings.ToLower(strings.TrimSpace(line)) {
		case "y", "yes":
			return nil
		default:
			return errors.New("declined by user")
		}
	}
}
