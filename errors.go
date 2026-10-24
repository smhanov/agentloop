package agentloop

import "errors"

// Sentinel errors reported by the loop. Every one is reachable from the
// error returned by Agent.Ask via errors.Is.
var (
	// ErrInvalidInput reports invalid arguments, such as an empty query.
	ErrInvalidInput = errors.New("agentloop: invalid input")
	// ErrBudgetExhausted reports that the MaxSteps ceiling was reached
	// before the model produced a final answer.
	ErrBudgetExhausted = errors.New("agentloop: budget exhausted")
	// ErrSkipToolCall is returned from a BeforeToolCall hook to veto the
	// execution of a single tool call without aborting the run; the call's
	// result message says "skipped".
	ErrSkipToolCall = errors.New("agentloop: skip tool call")
	// ErrNoLLMClient reports that New was called with a nil llmhub client.
	ErrNoLLMClient = errors.New("agentloop: no LLM client")
	// ErrConversationBusy reports an Ask or AskStream started on a
	// Conversation while another turn is still in flight.
	ErrConversationBusy = errors.New("agentloop: conversation busy")
	// ErrStreamStale reports a streaming run aborted because no chunk
	// arrived within the WithStreamWatchdog window.
	ErrStreamStale = errors.New("agentloop: stream stale")
	// ErrTooManyPinnedSkills reports a WithPinnedSkills call naming more
	// than two skills; at most two bodies may be inlined into the system
	// prompt.
	ErrTooManyPinnedSkills = errors.New("agentloop: too many pinned skills")
)
