package agentloop

import (
	"time"

	"github.com/smhanov/llmhub"
)

const (
	// defaultMaxToolCalls is the per-Ask tool-execution budget.
	defaultMaxToolCalls = 5
	// defaultDupCallLimit is how many consecutive identical tool calls
	// with identical output are tolerated before the loop intervenes.
	defaultDupCallLimit = 3
	// defaultParallelTools bounds concurrent parallel-eligible executions.
	defaultParallelTools = 3
	// defaultStreamWatchdog aborts streams that receive no chunk for this
	// long.
	defaultStreamWatchdog = 90 * time.Second
)

// config holds the resolved settings for an Agent and, layered on top of a
// copy, the per-Ask overrides. It is unexported; Option mutates it.
type config struct {
	systemPrompt    string
	tools           []Tool
	maxToolCalls    int
	maxSteps        int
	dupCallLimit    int
	parallelTools   int
	streamWatchdog  time.Duration
	toolChoice      llmhub.ToolChoice
	generateOptions []llmhub.Option
	hooks           Hooks
	compactor       Compactor
	settings        map[string]any
	// skillDirs and pinnedNames record construction-time skill intent
	// (WithSkills / WithPinnedSkills). resolveSkills consumes them once in
	// New, so the catalog and pinned bodies are frozen at construction;
	// the fields stay untouched by per-call option application.
	skillDirs   []string
	pinnedNames []string
}

// effectiveMaxToolCalls returns the tool-execution budget for a run.
func (c *config) effectiveMaxToolCalls() int {
	if c.maxToolCalls > 0 {
		return c.maxToolCalls
	}
	return defaultMaxToolCalls
}

// effectiveMaxSteps returns the inference ceiling for a run, defaulting to
// twice the tool-call budget plus two (room for the first and the forced
// final inference).
func (c *config) effectiveMaxSteps() int {
	if c.maxSteps > 0 {
		return c.maxSteps
	}
	return 2*c.effectiveMaxToolCalls() + 2
}

// effectiveParallelTools returns the concurrency bound for parallel-eligible
// tool executions.
func (c *config) effectiveParallelTools() int {
	if c.parallelTools >= 1 {
		return c.parallelTools
	}
	return defaultParallelTools
}

// Option configures an Agent at construction time or a single Ask at call
// time. Options apply in order, so per-call options (applied after the
// constructor's) override the Agent defaults.
type Option func(*config)

// WithSystemPrompt prepends a system message to every Ask. The prompt is
// byte-stable for the lifetime of an Agent; dynamic facts belong in the
// query text.
func WithSystemPrompt(p string) Option {
	return func(c *config) { c.systemPrompt = p }
}

// WithTools replaces the registered tool set. Keep the set small: tool
// schemas are sent to the model on every inference.
func WithTools(t ...Tool) Option {
	return func(c *config) { c.tools = append([]Tool(nil), t...) }
}

// WithMaxToolCalls caps tool executions per Ask; the default is 5 and the
// minimum is 1. Values below 1 are ignored, keeping the previous setting.
// Once the budget is spent, the loop forces a final answer with
// llmhub.NoToolChoice.
func WithMaxToolCalls(n int) Option {
	return func(c *config) {
		if n >= 1 {
			c.maxToolCalls = n
		}
	}
}

// WithMaxSteps caps model inferences per Ask; values below 1 are ignored.
// When unset the ceiling defaults to 2*MaxToolCalls+2. Exceeding it fails
// the run with ErrBudgetExhausted.
func WithMaxSteps(n int) Option {
	return func(c *config) {
		if n >= 1 {
			c.maxSteps = n
		}
	}
}

// WithDupCallLimit sets how many consecutive identical tool calls with
// identical output (same name and arguments, unchanged result) are
// tolerated; that many repeats deliver the tool's real output with a
// warning appended, and one more identical call stops the run gracefully
// with StatusStopped before it executes. A differing call or a changed
// output resets the counter. 0 disables detection entirely; the default is
// 3.
func WithDupCallLimit(n int) Option {
	return func(c *config) {
		if n >= 0 {
			c.dupCallLimit = n
		}
	}
}

// WithParallelTools bounds concurrent execution of parallel-eligible tool
// calls (Sequential == false) per step; the default is 3 and 1 forces
// strictly sequential execution. Values below 1 are ignored, keeping the
// previous setting.
func WithParallelTools(n int) Option {
	return func(c *config) {
		if n >= 1 {
			c.parallelTools = n
		}
	}
}

// WithStreamWatchdog aborts AskStream when no stream chunk arrives for d —
// the Hermes stale-stream pattern; the default is 90s and 0 disables the
// watchdog. Negative durations are ignored.
func WithStreamWatchdog(d time.Duration) Option {
	return func(c *config) {
		if d >= 0 {
			c.streamWatchdog = d
		}
	}
}

// WithToolChoice sets the default tool-choice policy sent with every
// Generate call; the default is llmhub.AutoToolChoice. The loop forces
// llmhub.NoToolChoice itself when the tool budget is exhausted.
func WithToolChoice(choice llmhub.ToolChoice) Option {
	return func(c *config) { c.toolChoice = choice }
}

// WithGenerateOptions forwards llmhub options to every Generate call (for
// example temperature or a prompt-cache key). Repeated uses append; per-call
// options are appended after the Agent's, so later llmhub options win.
func WithGenerateOptions(opts ...llmhub.Option) Option {
	return func(c *config) { c.generateOptions = append(c.generateOptions, opts...) }
}

// WithHooks installs hook callbacks, the library's observability and
// extension surface. Unset hook fields are no-ops.
func WithHooks(h Hooks) Option {
	return func(c *config) { c.hooks = h }
}

// WithCompactor installs a Compactor that bounds context growth. It runs
// after the RebuildContext hook before every LLM call and may replace the
// view wholesale; replacements are ephemeral, exactly like the hook's (see
// the Compactor interface for the contract, and NewTokenCompactor for the
// shipped reference implementation). Unset means no compaction.
func WithCompactor(c Compactor) Option {
	return func(cfg *config) { cfg.compactor = c }
}

// WithSetting attaches a named value to the option list on behalf of the
// preset that defines it. The core itself ignores settings; a preset reads
// them back with Settings to decode its own options before forwarding the
// option list to New. Later options with the same key win.
func WithSetting(key string, value any) Option {
	return func(c *config) {
		if c.settings == nil {
			c.settings = make(map[string]any)
		}
		c.settings[key] = value
	}
}

// Settings applies opts in order to a throwaway configuration and returns
// the values attached with WithSetting, keyed by name (nil when none).
// The returned map is a snapshot; mutating it affects no Agent.
func Settings(opts ...Option) map[string]any {
	var c config
	applyOptions(&c, opts)
	return c.settings
}
