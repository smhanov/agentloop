package agentloop

import (
	"context"
	"encoding/json"
	"time"

	"github.com/smhanov/llmhub"
)

// EventKind identifies which loop point an Event was fired from.
type EventKind int

// The loop points. Every event carries its kind in Event.Kind.
const (
	EventStart EventKind = iota
	EventBeforeGenerate
	EventRebuildContext
	EventAfterGenerate
	EventAnswerDelta
	EventReasoningDelta
	EventBeforeToolCall
	EventAfterToolCall
	EventDone
	EventError
	EventCompact
)

// Hooks is the extension surface of the loop: optional callbacks fired
// synchronously at each step. Unset fields are no-ops. A hook returning an
// error aborts the run and Ask returns a wrapped error naming the hook,
// except ErrSkipToolCall from BeforeToolCall, which merely skips that tool
// call. Hooks must be fast and non-blocking (BeforeToolCall may block for
// approval gates). The *Event is valid only during the hook call; copy
// anything retained.
type Hooks struct {
	// OnStart fires when a run begins.
	OnStart func(ctx context.Context, ev *Event) error
	// BeforeGenerate fires before each LLM call; it may APPEND to
	// ev.Messages (the appends are permanent transcript entries) and may
	// set ev.Skip to stop gracefully before querying.
	BeforeGenerate func(ctx context.Context, ev *Event) error
	// RebuildContext fires after BeforeGenerate, before each LLM call:
	// ev.Messages is the full transcript (read-only) and ev.Context is
	// pre-seeded to it — the hook may observe or replace ev.Context
	// wholesale. Replacements are ephemeral: they affect only this call,
	// while the transcript, Answer.Messages, and Conversation history stay
	// untouched. Hooks must preserve the trailing assistant tool-call
	// message and its tool results so the context stays answerable.
	RebuildContext func(ctx context.Context, ev *Event) error
	// AfterGenerate fires after each LLM response (the assistant message is
	// already appended); it may APPEND to ev.Messages. Finish arrives
	// pre-set to "no tool calls"; the hook may flip it either way — false
	// continues the loop with the appended messages, true stops without
	// executing the tools. The forced-final step after the tool budget is
	// spent always finishes, veto or not.
	AfterGenerate func(ctx context.Context, ev *Event) error
	// BeforeToolCall fires before each tool call; it may rewrite ev.Args,
	// and returning ErrSkipToolCall vetoes the execution.
	BeforeToolCall func(ctx context.Context, ev *Event) error
	// AfterToolCall fires when a tool execution completes; ev.Result,
	// ev.Err, and ev.Duration describe the execution.
	AfterToolCall func(ctx context.Context, ev *Event) error
	// OnDelta receives streaming answer/reasoning deltas (EventAnswerDelta
	// / EventReasoningDelta distinguishes them) in stream order, in
	// addition to the AskStream channel.
	OnDelta func(ctx context.Context, ev *Event) error // OnDone fires once when the run finishes; ev.Answer is fully
	// populated.
	OnDone func(ctx context.Context, ev *Event) error
	// OnError fires once on terminal failure; ev.Err carries the error,
	// which is still returned from Ask.
	OnError func(ctx context.Context, ev *Event) error
}

// Event is the single event type passed to every hook (plain struct, not
// interface). Which fields are populated depends on Kind.
type Event struct {
	Kind EventKind
	// Step is the 1-based model inference the event belongs to.
	Step int
	// Query is the user query (Start).
	Query string
	// Messages is the transcript so far. For BeforeGenerate and
	// AfterGenerate the list is appendable; appends become permanent
	// transcript entries.
	Messages []*llmhub.Message
	// Context is the LLM-visible view for this step: pre-seeded to the
	// transcript and replaceable by the RebuildContext hook and the
	// Compactor.
	Context []*llmhub.Message
	// Usage is the previous step's provider usage (RebuildContext and
	// Compact); zero on a run's first step.
	Usage llmhub.UsageMetadata
	// Skip requests a graceful stop (BeforeGenerate).
	Skip bool
	// Finish reports — and may be flipped to change — whether the response
	// ends the run: true when it carried no tool calls (AfterGenerate).
	Finish bool
	// Response is the model response (AfterGenerate).
	Response *llmhub.Response
	// Call is the model-requested tool call (tool events).
	Call *llmhub.ToolCallContent
	// Args holds the raw call arguments; BeforeToolCall may rewrite them.
	Args json.RawMessage
	// Result is the payload delivered to the model (AfterToolCall).
	Result string
	// Err is the error associated with the event (AfterToolCall handler
	// errors, Error).
	Err error
	// Duration is the tool execution's wall time (AfterToolCall).
	Duration time.Duration
	// Delta is the incremental streamed text (reserved for streaming).
	Delta string
	// Timing carries llmhub latency telemetry snapshots (reserved for
	// streaming and Done).
	Timing *llmhub.TimingMetrics
	// Answer is the fully populated result (Done).
	Answer *Answer
}
