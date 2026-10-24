package agentloop

import (
	"encoding/json"
	"time"

	"github.com/smhanov/llmhub"
)

// AnswerStatus reports how an Ask finished.
type AnswerStatus int

const (
	// StatusCompleted means the model produced a final answer — including
	// the forced answer delivered under llmhub.NoToolChoice after the tool
	// budget ran out.
	StatusCompleted AnswerStatus = iota
	// StatusBudgetExhausted means the MaxSteps ceiling was reached first;
	// Ask also returns an error wrapping ErrBudgetExhausted.
	StatusBudgetExhausted
	// StatusStopped means the run stopped gracefully: a BeforeGenerate hook
	// set Skip, or the duplicate-call limit detected a loop.
	StatusStopped
)

// ToolResult records one tool execution in the trace.
type ToolResult struct {
	// Name is the tool name the model requested.
	Name string
	// ID is the provider call ID.
	ID string
	// Args are the arguments as executed (post-rewrite by BeforeToolCall).
	Args json.RawMessage
	// Output is the payload delivered to the model: the handler's return
	// value, or the {"error": ...} / warning observation that replaced it.
	Output string
	// Err is the handler error, if any.
	Err error
	// Duration is the handler's wall time.
	Duration time.Duration
}

// Answer is the full result of one Ask: the final text plus the complete
// trace that produced it, making every run inspectable.
type Answer struct {
	// Text is the final step's assistant text (empty on graceful stops
	// before the model answered).
	Text string
	// ReasoningText concatenates the reasoning blocks across steps.
	ReasoningText string
	// Steps counts model inferences (one step = one inference plus its tool
	// executions).
	Steps int
	// ToolCalls counts executed tool invocations (skips not counted).
	ToolCalls int
	// Status reports how the run finished.
	Status AnswerStatus
	// Usage sums token and cost accounting across steps.
	Usage llmhub.UsageMetadata
	// Timing aggregates llmhub latency telemetry: Total and Chunks are
	// summed across steps; Connected, FirstByte, and FirstToken come from
	// the first step. It is nil only when the provider bypassed llmhub's
	// Client wrapper.
	Timing *llmhub.TimingMetrics
	// Tools records every execution in order — the trace.
	Tools []ToolResult
	// Messages is the full transcript of the run: system (if configured),
	// user, then assistant/tool messages as they were appended.
	Messages []*llmhub.Message
}
