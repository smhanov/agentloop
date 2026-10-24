package agentloop

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/smhanov/llmhub"
)

// eventBuffer is the capacity of the AskStream event channel; consumers
// that fall behind apply backpressure to the run.
const eventBuffer = 64

// askStream is AskStream's engine: it runs the shared loop in streaming
// mode, emitting loop events (deltas included) to a channel that closes
// after Done or Error.
func (a *Agent) askStream(ctx context.Context, query string, cfg config, prefix []*llmhub.Message) (<-chan Event, error) {
	if query == "" {
		return nil, fmt.Errorf("agentloop: %w: query is empty", ErrInvalidInput)
	}
	events := make(chan Event, eventBuffer)
	run := newRun(cfg)
	run.prefix = prefix
	run.streaming = true
	run.out = events
	go func() {
		defer close(events)
		_, _ = run.execute(ctx, a.llm, query)
	}()
	return events, nil
}

// streamInfer performs one model inference over llmhub's Stream channel:
// it forwards reasoning and content deltas as Step-tagged events, merges
// streamed tool-call fragments, accumulates usage and timing, and
// synthesizes the equivalent Response so the rest of the loop is shared
// with Ask. The watchdog (when enabled) aborts runs that receive no chunk
// for its window.
func (r *run) streamInfer(ctx context.Context, llm *llmhub.Client, context []*llmhub.Message, opts []llmhub.Option) (*llmhub.Response, error) {
	ch, err := llm.Stream(ctx, context, opts...)
	if err != nil {
		return nil, fmt.Errorf("agentloop: stream: %w", err)
	}
	if r.cfg.streamWatchdog > 0 {
		ch = watchStream(ctx, ch, r.cfg.streamWatchdog)
	}

	var (
		text      strings.Builder
		reasoning strings.Builder
		calls     []*llmhub.ToolCallContent
		usage     llmhub.UsageMetadata
		timing    *llmhub.TimingMetrics
	)
	step := r.steps + 1
	// abandon drains the rest of the stream in the background so the
	// provider goroutine can finish when this loop stops reading early.
	abandon := func() { go drainStream(ch) }
	for chunk := range ch {
		if chunk.Err != nil {
			abandon()
			return nil, chunk.Err
		}
		if chunk.ReasoningDelta != "" {
			reasoning.WriteString(chunk.ReasoningDelta)
			ev := &Event{Kind: EventReasoningDelta, Step: step, Delta: chunk.ReasoningDelta, Timing: chunk.Timing}
			if err := fire(ctx, r.cfg.hooks.OnDelta, ev); err != nil {
				abandon()
				return nil, fmt.Errorf("agentloop: OnDelta hook: %w", err)
			}
			r.emit(ev)
		}
		if chunk.Delta != "" {
			text.WriteString(chunk.Delta)
			ev := &Event{Kind: EventAnswerDelta, Step: step, Delta: chunk.Delta, Timing: chunk.Timing}
			if err := fire(ctx, r.cfg.hooks.OnDelta, ev); err != nil {
				abandon()
				return nil, fmt.Errorf("agentloop: OnDelta hook: %w", err)
			}
			r.emit(ev)
		}
		for _, frag := range chunk.ToolCalls {
			calls = mergeToolCallFragment(calls, frag)
		}
		if chunk.Usage != nil {
			usage = addUsage(usage, *chunk.Usage)
		}
		if chunk.Timing != nil {
			timing = chunk.Timing // last snapshot; the terminal chunk is complete
		}
		if chunk.Done {
			break
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	var content []llmhub.ContentPart
	if reasoning.Len() > 0 {
		content = append(content, llmhub.Reasoning(reasoning.String()))
	}
	if text.Len() > 0 {
		content = append(content, llmhub.Text(text.String()))
	}
	for _, call := range calls {
		content = append(content, call)
	}
	return &llmhub.Response{Content: content, Usage: usage, Timing: timing}, nil
}

// watchStream forwards stream chunks, aborting with ErrStreamStale when no
// chunk arrives within d. Once tripped it keeps draining the source in the
// background so the provider goroutine never blocks. Context cancellation
// surfaces as a chunk carrying ctx.Err().
func watchStream(ctx context.Context, src <-chan llmhub.StreamChunk, d time.Duration) <-chan llmhub.StreamChunk {
	out := make(chan llmhub.StreamChunk)
	go func() {
		defer close(out)
		timer := time.NewTimer(d)
		defer timer.Stop()
		for {
			select {
			case chunk, ok := <-src:
				if !ok {
					return
				}
				out <- chunk
				if chunk.Done || chunk.Err != nil {
					return
				}
				timer.Reset(d)
			case <-timer.C:
				out <- llmhub.StreamChunk{Err: fmt.Errorf("%w: no chunk received for %v", ErrStreamStale, d)}
				go drainStream(src)
				return
			case <-ctx.Done():
				out <- llmhub.StreamChunk{Err: ctx.Err()}
				go drainStream(src)
				return
			}
		}
	}()
	return out
}

// drainStream consumes a stream to its end so its producer can finish.
func drainStream(src <-chan llmhub.StreamChunk) {
	for chunk := range src {
		if chunk.Done || chunk.Err != nil {
			return
		}
	}
}

// mergeToolCallFragment merges one streamed tool-call delta into calls,
// keyed by ToolCallContent.Index the way providers emit parallel argument
// fragments.
func mergeToolCallFragment(calls []*llmhub.ToolCallContent, frag *llmhub.ToolCallContent) []*llmhub.ToolCallContent {
	for _, existing := range calls {
		if existing.Index == frag.Index {
			if frag.ID != "" {
				existing.ID = frag.ID
			}
			if frag.Name != "" {
				existing.Name = frag.Name
			}
			existing.Arguments += frag.Arguments
			return calls
		}
	}
	clone := *frag
	return append(calls, &clone)
}

// addUsage sums two usage records (per step, per stream, and per turn).
func addUsage(a, b llmhub.UsageMetadata) llmhub.UsageMetadata {
	a.PromptTokens += b.PromptTokens
	a.CompletionTokens += b.CompletionTokens
	a.TotalTokens += b.TotalTokens
	a.CacheReadTokens += b.CacheReadTokens
	a.CacheCreationTokens += b.CacheCreationTokens
	a.ReasoningTokens += b.ReasoningTokens
	a.Cost += b.Cost
	return a
}
