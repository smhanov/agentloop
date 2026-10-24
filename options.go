package agentloop

import (
	"github.com/smhanov/llmhub"
)

const (
	// defaultMaxToolCalls is the per-Ask tool-execution budget.
	defaultMaxToolCalls = 5
	// defaultDupCallLimit is how many consecutive identical tool calls are
	// tolerated before the loop intervenes.
	defaultDupCallLimit = 3
)

// config holds the resolved settings for an Agent and, layered on top of a
// copy, the per-Ask overrides. It is unexported; Option mutates it.
type config struct {
	systemPrompt    string
	tools           []Tool
	maxToolCalls    int
	maxSteps        int
	dupCallLimit    int
	toolChoice      llmhub.ToolChoice
	generateOptions []llmhub.Option
	hooks           Hooks
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

// WithDupCallLimit sets how many consecutive identical tool calls (same
// name and arguments) are tolerated; that many repeats deliver a warning
// observation, and one more stops the run gracefully with StatusStopped.
// 0 disables detection entirely; the default is 3.
func WithDupCallLimit(n int) Option {
	return func(c *config) {
		if n >= 0 {
			c.dupCallLimit = n
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
