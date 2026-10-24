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
		maxToolCalls: defaultMaxToolCalls,
		dupCallLimit: defaultDupCallLimit,
		toolChoice:   llmhub.AutoToolChoice(),
	}
	applyOptions(&cfg, opts)
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

// Ask runs the loop to completion for one user query and returns the full
// trace. opts override the Agent's construction options for this call only;
// they are applied after the constructor options, so later options win.
func (a *Agent) Ask(ctx context.Context, query string, opts ...Option) (*Answer, error) {
	if query == "" {
		return nil, fmt.Errorf("agentloop: %w: query is empty", ErrInvalidInput)
	}
	cfg := a.cfg
	applyOptions(&cfg, opts)
	return newRun(cfg).execute(ctx, a.llm, query)
}

// applyOptions applies opts in order, skipping nil entries.
func applyOptions(cfg *config, opts []Option) {
	for _, opt := range opts {
		if opt != nil {
			opt(cfg)
		}
	}
}
