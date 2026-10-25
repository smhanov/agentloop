// Package agentloop implements a general-purpose agentic loop: send prompts
// to an LLM through llmhub, let the model invoke tools, execute them, feed
// the results back, and repeat until the model produces a final answer.
// Budgets bound every run; hooks observe every step.
package agentloop

import (
	"context"
	"fmt"

	"github.com/smhanov/llmhub"
)

// Agent runs the agentic loop over an llmhub client. An Agent is immutable
// after construction and safe for concurrent use; per-run state lives in
// locals.
type Agent struct {
	llm *llmhub.Client
	cfg config
}

// New creates a general-purpose agent over the given llmhub client with no
// tools and no system prompt; configure it with options. It returns an
// error wrapping ErrNoLLMClient when llm is nil.
func New(llm *llmhub.Client, opts ...Option) (*Agent, error) {
	if llm == nil {
		return nil, fmt.Errorf("agentloop: %w", ErrNoLLMClient)
	}
	cfg := config{
		maxToolCalls:   defaultMaxToolCalls,
		dupCallLimit:   defaultDupCallLimit,
		parallelTools:  defaultParallelTools,
		streamWatchdog: defaultStreamWatchdog,
		toolChoice:     llmhub.AutoToolChoice(),
	}
	applyOptions(&cfg, opts)
	if err := resolveSkills(&cfg); err != nil {
		return nil, err
	}
	return &Agent{llm: llm, cfg: cfg}, nil
}

// MustNew is like New but panics on error (for examples and tests).
func MustNew(llm *llmhub.Client, opts ...Option) *Agent {
	agent, err := New(llm, opts...)
	if err != nil {
		panic(err)
	}
	return agent
}

// Tools returns the Agent's registered tools in registration order — the
// construction-time set, with skills resolved. The result is a copy;
// mutating it affects no Agent. Presets use it to re-issue one merged
// WithTools for an Ask (the core's WithTools replaces the set, it never
// merges).
func (a *Agent) Tools() []Tool {
	out := make([]Tool, len(a.cfg.tools))
	copy(out, a.cfg.tools)
	return out
}

// Ask runs the loop to completion for one user query and returns the full
// trace. opts override the Agent's construction options for this call only;
// they are applied after the constructor options, so later options win.
func (a *Agent) Ask(ctx context.Context, query string, opts ...Option) (*Answer, error) {
	cfg := a.cfg
	applyOptions(&cfg, opts)
	return a.ask(ctx, query, cfg, nil)
}

// AskStream is Ask with streaming, mirroring llmhub's streaming semantics:
// underlying llmhub Stream chunks are forwarded as they arrive — reasoning
// deltas (EventReasoningDelta) and content deltas (EventAnswerDelta) from
// EVERY step, each tagged with Step. Deltas from steps that end in tool
// calls are interim model commentary — streamed for live UIs but excluded
// from the final Answer.Text, which is the final step's text. Every delta
// event carries an llmhub Timing snapshot; Done carries the complete set.
// Intermediate tool/step events are also delivered so CLI/UI consumers see
// progress. The channel closes after Done or Error; consumers must drain it
// until close. A run that receives no stream chunk for the
// WithStreamWatchdog window aborts with ErrStreamStale.
func (a *Agent) AskStream(ctx context.Context, query string, opts ...Option) (<-chan Event, error) {
	cfg := a.cfg
	applyOptions(&cfg, opts)
	return a.askStream(ctx, query, cfg, nil)
}

// ask runs one query with an optional conversation prefix.
func (a *Agent) ask(ctx context.Context, query string, cfg config, prefix []*llmhub.Message) (*Answer, error) {
	if query == "" {
		return nil, fmt.Errorf("agentloop: %w: query is empty", ErrInvalidInput)
	}
	run := newRun(cfg)
	run.prefix = prefix
	return run.execute(ctx, a.llm, query)
}

// applyOptions applies opts in order, skipping nil entries.
func applyOptions(cfg *config, opts []Option) {
	for _, opt := range opts {
		if opt != nil {
			opt(cfg)
		}
	}
}
