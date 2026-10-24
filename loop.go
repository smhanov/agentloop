package agentloop

import (
	"context"
	"fmt"
	"strings"

	"github.com/smhanov/llmhub"
)

// run holds the mutable state of a single Ask. A run is used by one
// goroutine only.
type run struct {
	ctx        context.Context
	cfg        config
	transcript []*llmhub.Message
	steps      int // model inferences so far
	executed   int // tool calls counted against the budget (Answer.ToolCalls)
	usage      llmhub.UsageMetadata
	timing     *llmhub.TimingMetrics
	reasoning  strings.Builder
	lastText   string // the most recent step's assistant text
	trace      []ToolResult
	dupKey     string // (name, args) key of the current consecutive run
	dupCount   int    // length of the current consecutive identical-call run
}

// newRun initializes per-run state from a resolved config.
func newRun(cfg config) *run {
	return &run{cfg: cfg}
}

// execute runs the loop: build the transcript, then repeat Generate and
// sequential tool dispatch until the model produces a final answer or a
// budget stops the run.
func (r *run) execute(ctx context.Context, llm *llmhub.Client, query string) (*Answer, error) {
	r.ctx = ctx
	hooks := r.cfg.hooks

	// Seed the transcript and announce the run.
	if r.cfg.systemPrompt != "" {
		r.transcript = append(r.transcript, llmhub.NewSystemMessage(llmhub.Text(r.cfg.systemPrompt)))
	}
	r.transcript = append(r.transcript, llmhub.NewUserMessage(llmhub.Text(query)))
	if err := fire(ctx, hooks.OnStart, &Event{Kind: EventStart, Query: query, Messages: r.transcript}); err != nil {
		return r.fail(fmt.Errorf("agentloop: OnStart hook: %w", err))
	}

	for {
		// Hard ceiling on inferences; every other limit funnels into it.
		maxSteps := r.cfg.effectiveMaxSteps()
		if r.steps >= maxSteps {
			err := fmt.Errorf("agentloop: %w: reached the %d-step ceiling", ErrBudgetExhausted, maxSteps)
			ans := r.answer(StatusBudgetExhausted)
			r.reportError(hooks, ans, err)
			return ans, err
		}
		step := r.steps + 1

		// Hook: steering seam and graceful stop.
		ev := &Event{Kind: EventBeforeGenerate, Step: step, Messages: r.transcript}
		if err := fire(ctx, hooks.BeforeGenerate, ev); err != nil {
			return r.fail(fmt.Errorf("agentloop: BeforeGenerate hook: %w", err))
		}
		r.transcript = ev.Messages
		if ev.Skip {
			return r.finish(StatusStopped)
		}

		// Query the model. Once the tool budget is spent, force a final
		// answer by disabling tools for this call.
		forceFinal := r.executed >= r.cfg.effectiveMaxToolCalls()
		resp, err := llm.Generate(ctx, r.transcript, r.generateOptions(forceFinal)...)
		if err != nil {
			if ctxErr := ctx.Err(); ctxErr != nil {
				r.reportError(hooks, nil, ctxErr)
				return nil, ctxErr
			}
			wrapped := fmt.Errorf("agentloop: generate: %w", err)
			r.reportError(hooks, nil, wrapped)
			return nil, wrapped
		}

		// Append the assistant message verbatim (tool-call parts preserved)
		// and accumulate the step's telemetry.
		r.transcript = append(r.transcript, llmhub.NewAssistantMessage(resp.Content...))
		r.steps++
		r.accumulate(resp)

		// Hook: observation only in this version — Finish is reported
		// as-is, flips are not consulted.
		ev = &Event{Kind: EventAfterGenerate, Step: r.steps, Messages: r.transcript, Response: resp, Finish: len(resp.ToolCalls()) == 0}
		if err := fire(ctx, hooks.AfterGenerate, ev); err != nil {
			return r.fail(fmt.Errorf("agentloop: AfterGenerate hook: %w", err))
		}
		r.transcript = ev.Messages

		// No tool calls: the step's text is the final answer.
		calls := resp.ToolCalls()
		if len(calls) == 0 {
			return r.finish(StatusCompleted)
		}

		// Budget spent: execute none and finish with the step's text — the
		// loop can never spin forever.
		if forceFinal {
			return r.finish(StatusCompleted)
		}

		// Dispatch the step's tool calls sequentially.
		for _, call := range calls {
			stop, err := r.dispatch(ctx, step, call)
			if err != nil {
				return r.fail(err)
			}
			if stop {
				return r.finish(StatusStopped)
			}
		}
	}
}

// generateOptions builds the per-call llmhub options: the configured
// defaults plus the tool-choice policy, with NoToolChoice appended when the
// budget forces a final answer. The slice is copied so concurrent runs
// never share backing arrays.
func (r *run) generateOptions(forceFinal bool) []llmhub.Option {
	opts := make([]llmhub.Option, 0, len(r.cfg.generateOptions)+2)
	opts = append(opts, r.cfg.generateOptions...)
	choice := r.cfg.toolChoice
	if choice.Mode == "" {
		choice = llmhub.AutoToolChoice()
	}
	opts = append(opts, llmhub.WithToolChoice(choice))
	if forceFinal {
		opts = append(opts, llmhub.WithToolChoice(llmhub.NoToolChoice()))
	}
	return opts
}

// accumulate folds one step's usage, reasoning, text, and timing into the
// run totals.
func (r *run) accumulate(resp *llmhub.Response) {
	u := resp.Usage
	r.usage.PromptTokens += u.PromptTokens
	r.usage.CompletionTokens += u.CompletionTokens
	r.usage.TotalTokens += u.TotalTokens
	r.usage.CacheReadTokens += u.CacheReadTokens
	r.usage.CacheCreationTokens += u.CacheCreationTokens
	r.usage.ReasoningTokens += u.ReasoningTokens
	r.usage.Cost += u.Cost
	r.reasoning.WriteString(resp.ReasoningText())
	r.lastText = resp.Text()

	if t := resp.Timing; t != nil {
		if r.timing == nil {
			clone := *t
			r.timing = &clone
		} else {
			r.timing.Total += t.Total
			r.timing.Chunks += t.Chunks
		}
	}
}

// finish assembles the Answer and reports it via OnDone. An OnDone error is
// honored like any hook error but the completed answer still comes back.
func (r *run) finish(status AnswerStatus) (*Answer, error) {
	ans := r.answer(status)
	if err := fire(r.ctx, r.cfg.hooks.OnDone, &Event{Kind: EventDone, Timing: ans.Timing, Answer: ans}); err != nil {
		return ans, fmt.Errorf("agentloop: OnDone hook: %w", err)
	}
	return ans, nil
}

// fail reports a terminal failure via OnError and returns it; the error is
// still returned from execute.
func (r *run) fail(err error) (*Answer, error) {
	r.reportError(r.cfg.hooks, nil, err)
	return nil, err
}

// reportError reports a terminal failure via OnError; the error is still
// returned from execute.
func (r *run) reportError(hooks Hooks, ans *Answer, err error) {
	_ = fire(r.ctx, hooks.OnError, &Event{Kind: EventError, Err: err, Answer: ans})
}

// answer assembles the Answer for the current run state.
func (r *run) answer(status AnswerStatus) *Answer {
	return &Answer{
		Text:          r.lastText,
		ReasoningText: r.reasoning.String(),
		Steps:         r.steps,
		ToolCalls:     r.executed,
		Status:        status,
		Usage:         r.usage,
		Timing:        r.timing,
		Tools:         r.trace,
		Messages:      r.transcript,
	}
}

// fire invokes a hook with the run's context; a nil hook is a no-op.
func fire(ctx context.Context, h func(context.Context, *Event) error, ev *Event) error {
	if h == nil {
		return nil
	}
	return h(ctx, ev)
}
