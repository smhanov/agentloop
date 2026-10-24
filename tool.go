package agentloop

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/smhanov/llmhub"
)

// Handler executes a tool invocation. args is the raw JSON arguments string
// from the model (llmhub ToolCallContent.Arguments). The return value is
// inserted verbatim as the tool result message content; a returned error is
// delivered to the model as an {"error": ...} observation and the loop
// continues, except for context errors (cancellation or deadline), which
// abort the run.
type Handler func(ctx context.Context, args json.RawMessage) (string, error)

// Tool is one model-callable tool: its llmhub definition plus the Go
// handler that executes it.
type Tool struct {
	// Def is the schema shown to the model: name, description, and JSON
	// Schema parameters.
	Def llmhub.Tool
	// Run executes the tool.
	Run Handler

	// ConcurrencySafe reports whether concurrent executions of this tool
	// are safe. Safe tools are eligible for bounded parallel batching and
	// unsafe tools always execute in strict call order, never overlapping
	// with anything. The zero value is false (not safe); set it to true for
	// pure, read-only tools and leave it false for stateful tools (shell
	// sessions, transactions). Tool dispatch is sequential in this version,
	// so the flag is advisory.
	ConcurrencySafe bool
}

// Observation payloads the loop itself delivers to the model.
const (
	// skippedResult is the tool result message content for calls vetoed by
	// a BeforeToolCall hook via ErrSkipToolCall.
	skippedResult = "skipped"
	// warningObservation replaces the output of the tool call that reaches
	// the duplicate-call limit.
	warningObservation = `{"warning": "repeated identical call; change approach"}`
	// stopObservation is delivered for one repeat too many; the run then
	// stops gracefully with StatusStopped.
	stopObservation = `{"warning": "repeated identical call; stopping"}`
	// budgetObservation is delivered for calls denied because the
	// MaxToolCalls budget was spent mid-step (several calls in one
	// response).
	budgetObservation = `{"error": "tool call budget exhausted"}`
)

// errorJSON renders a failure as the in-band error observation the model
// sees; the loop itself continues.
func errorJSON(err error) string {
	b, _ := json.Marshal(map[string]string{"error": err.Error()})
	return string(b)
}

// unknownToolJSON renders the observation for a call naming an
// unregistered tool, including a Levenshtein suggestion when one is close
// enough.
func unknownToolJSON(suggestion string) string {
	msg := "unknown tool"
	if suggestion != "" {
		msg = fmt.Sprintf("unknown tool; did you mean %q?", suggestion)
	}
	b, _ := json.Marshal(map[string]string{"error": msg})
	return string(b)
}

// levenshtein returns the edit distance between a and b (insertions,
// deletions, substitutions).
func levenshtein(a, b string) int {
	ar, br := []rune(a), []rune(b)
	prev := make([]int, len(br)+1)
	curr := make([]int, len(br)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(ar); i++ {
		curr[0] = i
		for j := 1; j <= len(br); j++ {
			cost := 1
			if ar[i-1] == br[j-1] {
				cost = 0
			}
			curr[j] = min(curr[j-1]+1, prev[j]+1, prev[j-1]+cost)
		}
		prev, curr = curr, prev
	}
	return prev[len(br)]
}

// suggestName returns the registered name closest to name when it is within
// edit distance 2 (Hermes' hallucinated-name repair), or "" when nothing is
// close enough. Ties go to the first candidate with the smallest distance.
func suggestName(name string, registered []string) string {
	best := ""
	bestDist := 3
	for _, candidate := range registered {
		if d := levenshtein(name, candidate); d < bestDist {
			best, bestDist = candidate, d
		}
	}
	return best
}

// lookup finds the registered tool with the given definition name.
func (r *run) lookup(name string) (Tool, bool) {
	for _, t := range r.cfg.tools {
		if t.Def.Name == name {
			return t, true
		}
	}
	return Tool{}, false
}

// registeredNames lists the tool names known to the run, in registration
// order.
func (r *run) registeredNames() []string {
	names := make([]string, 0, len(r.cfg.tools))
	for _, t := range r.cfg.tools {
		names = append(names, t.Def.Name)
	}
	return names
}

// dispatch resolves one model-requested tool call: it fires BeforeToolCall
// (veto/rewrite), applies duplicate-call detection and the tool budget,
// executes the handler, records the trace entry, appends the tool result
// message, and fires AfterToolCall. It reports stop=true when the
// duplicate-call limit was exceeded and the run should finish gracefully.
func (r *run) dispatch(ctx context.Context, step int, call *llmhub.ToolCallContent) (bool, error) {
	hooks := r.cfg.hooks

	// Hook: veto or argument rewrite.
	ev := &Event{Kind: EventBeforeToolCall, Step: step, Call: call, Args: json.RawMessage(call.Arguments)}
	if err := fire(ctx, hooks.BeforeToolCall, ev); err != nil {
		if errors.Is(err, ErrSkipToolCall) {
			r.appendToolResult(call, skippedResult)
			return false, nil
		}
		return false, fmt.Errorf("agentloop: BeforeToolCall hook: %w", err)
	}
	args := ev.Args
	if len(args) == 0 {
		args = json.RawMessage(call.Arguments)
	}

	// Loop detection: identical consecutive (name, args) calls warn at the
	// limit and stop the run on the next repeat. Any differing call resets
	// the run.
	warn := false
	if r.cfg.dupCallLimit > 0 {
		key := call.Name + "\x00" + string(args)
		if key == r.dupKey {
			r.dupCount++
		} else {
			r.dupKey, r.dupCount = key, 1
		}
		switch {
		case r.dupCount > r.cfg.dupCallLimit:
			r.appendToolResult(call, stopObservation)
			return true, nil
		case r.dupCount == r.cfg.dupCallLimit:
			warn = true
		}
	}

	// Budget: one response may carry more calls than the budget has left.
	if r.executed >= r.cfg.effectiveMaxToolCalls() {
		r.appendToolResult(call, budgetObservation)
		return false, nil
	}

	tool, found := r.lookup(call.Name)
	if !found {
		observation := unknownToolJSON(suggestName(call.Name, r.registeredNames()))
		r.executed++
		r.trace = append(r.trace, ToolResult{Name: call.Name, ID: call.ID, Args: args, Output: observation})
		r.appendToolResult(call, observation)
		after := &Event{Kind: EventAfterToolCall, Step: step, Call: call, Args: args, Result: observation}
		if err := fire(ctx, hooks.AfterToolCall, after); err != nil {
			return false, fmt.Errorf("agentloop: AfterToolCall hook: %w", err)
		}
		return false, nil
	}

	// Execute.
	start := time.Now()
	output, handlerErr := tool.Run(ctx, args)
	duration := time.Since(start)

	observation := output
	switch {
	case handlerErr != nil:
		if errors.Is(handlerErr, context.Canceled) || errors.Is(handlerErr, context.DeadlineExceeded) {
			// Context errors abort the run; every other handler error is
			// repaired in-band.
			return false, handlerErr
		}
		observation = errorJSON(handlerErr)
	case warn:
		observation = warningObservation
	}

	r.executed++
	r.trace = append(r.trace, ToolResult{Name: call.Name, ID: call.ID, Args: args, Output: observation, Err: handlerErr, Duration: duration})
	r.appendToolResult(call, observation)

	after := &Event{Kind: EventAfterToolCall, Step: step, Call: call, Args: args, Result: observation, Err: handlerErr, Duration: duration}
	if err := fire(ctx, hooks.AfterToolCall, after); err != nil {
		return false, fmt.Errorf("agentloop: AfterToolCall hook: %w", err)
	}
	return false, nil
}

// appendToolResult appends the tool result message for call, keeping the
// transcript answerable for every tool call the model made.
func (r *run) appendToolResult(call *llmhub.ToolCallContent, observation string) {
	r.transcript = append(r.transcript, llmhub.NewToolResultMessage(call.ID, call.Name, llmhub.Text(observation)))
}
