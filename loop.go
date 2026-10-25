package agentloop

import (
	"context"
	"fmt"
	"strings"

	"github.com/smhanov/llmhub"
)

// forcedFinalNotice is the budget-exhaustion instruction appended to the
// transcript exactly once, immediately before the first forced-final
// inference. Until it arrives, the transcript's last entry is a tool result
// and the model has no signal that no further tools will be available —
// providers honor a fresh user-role instruction far better than
// tool-choice alone, so without it models keep narrating their interim
// plan (their next searches) instead of answering from what they found.
// The wording is domain-neutral and instructs synthesis of the gathered
// evidence, plain statements of anything undetermined, and no further plan
// narration.
const forcedFinalNotice = "The tool budget for this run is exhausted: no further tool calls will be executed. " +
	"Answer the user's original question now, using only the evidence already gathered in this conversation. " +
	"Synthesize your findings into a complete final answer, and state plainly anything that could not be determined. " +
	"Do not narrate further research plans, do not say what you would check next, and do not ask permission — " +
	"produce the final answer the user will read."

// run holds the mutable state of a single Ask or AskStream. A run is used
// by one goroutine (plus short-lived dispatch workers that only touch
// their own result slots).
type run struct {
	ctx        context.Context
	cfg        config
	streaming  bool              // AskStream mode: Stream + delta forwarding
	out        chan<- Event      // event sink in streaming mode; nil in ask mode
	prefix     []*llmhub.Message // conversation history prepended before the query
	prefixLen  int               // transcript messages before this run's own
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
	notice     bool   // the forced-final notice is already in the transcript
}

// newRun initializes per-run state from a resolved config.
func newRun(cfg config) *run {
	return &run{cfg: cfg}
}

// execute runs the loop: build the transcript, then repeat inference and
// tool dispatch until the model produces a final answer, a hook stops the
// run, or a budget trips.
func (r *run) execute(ctx context.Context, llm *llmhub.Client, query string) (*Answer, error) {
	r.ctx = ctx
	hooks := r.cfg.hooks

	// Seed the transcript. Plain Asks seed the configured system message
	// and keep it in Answer.Messages (prefixLen 0); conversation runs
	// receive the byte-stable system message inside the prefix, and
	// Answer.Messages carries only this run's own messages.
	r.prefixLen = 0
	if r.prefix == nil {
		if r.cfg.systemPrompt != "" {
			r.transcript = append(r.transcript, llmhub.NewSystemMessage(llmhub.Text(r.cfg.systemPrompt)))
		}
	} else {
		r.transcript = append(r.transcript, r.prefix...)
		r.prefixLen = len(r.prefix)
	}
	r.transcript = append(r.transcript, llmhub.NewUserMessage(llmhub.Text(query)))

	if err := fire(ctx, hooks.OnStart, &Event{Kind: EventStart, Query: query, Messages: r.transcript}); err != nil {
		return r.fail(fmt.Errorf("agentloop: OnStart hook: %w", err))
	}
	r.emit(&Event{Kind: EventStart, Query: query, Messages: r.transcript})

	for {
		// Hard ceiling on inferences; every other limit funnels into it.
		maxSteps := r.cfg.effectiveMaxSteps()
		if r.steps >= maxSteps {
			err := fmt.Errorf("agentloop: %w: reached the %d-step ceiling", ErrBudgetExhausted, maxSteps)
			ans := r.answer(StatusBudgetExhausted)
			r.reportError(hooks, ans, err)
			r.emit(&Event{Kind: EventError, Err: err, Answer: ans})
			return ans, err
		}
		step := r.steps + 1

		// Hook: steering seam and graceful stop. Appends persist in the
		// transcript.
		ev := &Event{Kind: EventBeforeGenerate, Step: step, Messages: r.transcript}
		if err := fire(ctx, hooks.BeforeGenerate, ev); err != nil {
			return r.fail(fmt.Errorf("agentloop: BeforeGenerate hook: %w", err))
		}
		r.transcript = ev.Messages
		r.emit(&Event{Kind: EventBeforeGenerate, Step: step, Messages: r.transcript})
		if ev.Skip {
			return r.finish(StatusStopped)
		}

		// Once the tool budget is spent, force a final answer by disabling
		// tools for this call.
		forceFinal := r.executed >= r.cfg.effectiveMaxToolCalls()

		// On the first forced step, tell the model the situation changed:
		// the budget is spent and it must answer from the evidence already
		// gathered. The notice rides a user message appended AFTER the last
		// tool result and BEFORE the forced inference — the transcript
		// stays append-only and honestly records the instruction. A vetoed
		// text-only forced step re-enters this path; the guard keeps the
		// notice single.
		if forceFinal && !r.notice {
			r.notice = true
			r.transcript = append(r.transcript, llmhub.NewUserMessage(llmhub.Text(forcedFinalNotice)))
		}

		// Hook: observe or replace the exact context the model sees. The
		// transcript itself is never mutated by this hook.
		context := r.transcript
		if hooks.RebuildContext != nil {
			ev := &Event{Kind: EventRebuildContext, Step: step, Messages: r.transcript, Context: r.transcript}
			if err := fire(ctx, hooks.RebuildContext, ev); err != nil {
				return r.fail(fmt.Errorf("agentloop: RebuildContext hook: %w", err))
			}
			if ev.Context != nil {
				context = ev.Context
			}
			r.emit(&Event{Kind: EventRebuildContext, Step: step, Messages: r.transcript, Context: context})
		}

		resp, err := r.infer(ctx, llm, context, r.generateOptions(forceFinal))
		if err != nil {
			if ctxErr := ctx.Err(); ctxErr != nil {
				// Context errors come back unwrapped.
				r.reportError(hooks, nil, ctxErr)
				r.emit(&Event{Kind: EventError, Err: ctxErr})
				return nil, ctxErr
			}
			return r.fail(err)
		}

		// Append the assistant message verbatim (tool-call parts preserved)
		// and accumulate the step's telemetry.
		r.transcript = append(r.transcript, llmhub.NewAssistantMessage(resp.Content...))
		r.steps++
		r.accumulate(resp)

		// Hook: finish veto. Finish arrives pre-set (no tool calls means
		// done); the hook may flip it either way — false continues the
		// loop with whatever the hook appended, true stops without
		// executing the tools.
		ev = &Event{Kind: EventAfterGenerate, Step: r.steps, Messages: r.transcript, Response: resp, Finish: len(resp.ToolCalls()) == 0}
		if err := fire(ctx, hooks.AfterGenerate, ev); err != nil {
			return r.fail(fmt.Errorf("agentloop: AfterGenerate hook: %w", err))
		}
		r.transcript = ev.Messages
		r.emit(&Event{Kind: EventAfterGenerate, Step: r.steps, Messages: r.transcript, Response: resp, Finish: ev.Finish})

		if ev.Finish {
			return r.finish(StatusCompleted)
		}

		calls := resp.ToolCalls()
		if len(calls) == 0 {
			// Vetoed text-only response: the hook appended observation
			// messages; continue the loop with them.
			continue
		}

		// Budget spent: execute none and finish with the step's text — the
		// loop can never spin forever, not even under a vetoing hook.
		if forceFinal {
			return r.finish(StatusCompleted)
		}

		stopped, err := r.dispatchAll(ctx, step, calls)
		if err != nil {
			return r.fail(err)
		}
		if stopped {
			return r.finish(StatusStopped)
		}
	}
}

// infer performs one model inference for the current step: Generate in ask
// mode, Stream with delta forwarding in streaming mode. Returned errors are
// already contextualized; context cancellation is detected by the caller.
func (r *run) infer(ctx context.Context, llm *llmhub.Client, context []*llmhub.Message, opts []llmhub.Option) (*llmhub.Response, error) {
	if !r.streaming {
		resp, err := llm.Generate(ctx, context, opts...)
		if err != nil {
			return nil, fmt.Errorf("agentloop: generate: %w", err)
		}
		return resp, nil
	}
	return r.streamInfer(ctx, llm, context, opts)
}

// generateOptions builds the per-call llmhub options: the registered
// tools' schemas come first (so caller-supplied llmhub options can still
// override them), then the configured defaults, plus the tool-choice
// policy, with NoToolChoice appended when the budget forces a final
// answer. Tool schemas and tool choice are only sent when tools are
// registered — several OpenAI-compatible servers (vLLM included) reject
// tool_choice without tools. The slice is copied so concurrent runs never
// share backing arrays.
func (r *run) generateOptions(forceFinal bool) []llmhub.Option {
	opts := make([]llmhub.Option, 0, len(r.cfg.tools)+len(r.cfg.generateOptions)+2)
	if len(r.cfg.tools) == 0 {
		return append(opts, r.cfg.generateOptions...)
	}
	defs := make([]llmhub.Tool, len(r.cfg.tools))
	for i, tool := range r.cfg.tools {
		defs[i] = tool.Def
	}
	opts = append(opts, llmhub.WithTools(defs...))
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
	r.usage = addUsage(r.usage, resp.Usage)
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
	ev := &Event{Kind: EventDone, Timing: ans.Timing, Answer: ans}
	if err := fire(r.ctx, r.cfg.hooks.OnDone, ev); err != nil {
		err = fmt.Errorf("agentloop: OnDone hook: %w", err)
		r.emit(&Event{Kind: EventError, Err: err, Answer: ans})
		return ans, err
	}
	r.emit(ev)
	return ans, nil
}

// fail reports a terminal failure via OnError and the event stream and
// returns it; the error is still returned from execute.
func (r *run) fail(err error) (*Answer, error) {
	r.reportError(r.cfg.hooks, nil, err)
	r.emit(&Event{Kind: EventError, Err: err})
	return nil, err
}

// reportError reports a terminal failure via OnError; the error is still
// returned from execute.
func (r *run) reportError(hooks Hooks, ans *Answer, err error) {
	_ = fire(r.ctx, hooks.OnError, &Event{Kind: EventError, Err: err, Answer: ans})
}

// answer assembles the Answer for the current run state. Messages carries
// this run's own messages: the system message and everything after it, but
// not the conversation prefix.
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
		Messages:      r.transcript[r.prefixLen:],
	}
}

// emit delivers an event to an AskStream consumer; it is a no-op in ask
// mode. A shallow copy is sent so later transcript appends cannot extend
// the consumer's view of the event.
func (r *run) emit(ev *Event) {
	if r.out == nil {
		return
	}
	r.out <- *ev
}

// fire invokes a hook with the run's context; a nil hook is a no-op.
func fire(ctx context.Context, h func(context.Context, *Event) error, ev *Event) error {
	if h == nil {
		return nil
	}
	return h(ctx, ev)
}
