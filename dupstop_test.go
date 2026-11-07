package agentloop

import (
	"context"
	"testing"

	"github.com/smhanov/llmhub"
)

// TestAskDupCallStopClosesTrailingCalls pins the invariant that every tool
// call in an assistant message is answered by a tool result before the turn
// is remembered. When duplicate detection stops a run in the middle of a
// batch, the trailing calls of that same assistant message are never
// planned, so nothing would close them — and providers reject a transcript
// whose calls have no results.
func TestAskDupCallStopClosesTrailingCalls(t *testing.T) {
	same := func(id string) *llmhub.ToolCallContent { return llmhub.ToolCall(id, "echo", `{"n":1}`) }
	resetScript(t,
		callTurn(same("c1")),
		callTurn(same("c2")),
		callTurn(same("c3")),
		// The stop trips on c4; c5 rides the same assistant message but is
		// never planned, because resolution stops at the first stop plan.
		callTurn(same("c4"), same("c5")),
		textTurn("unreachable"),
	)
	ag := newFakeAgent(t, WithTools(echoTool()))

	ans, err := ag.Ask(context.Background(), "Loop away.")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ans.Status != StatusStopped {
		t.Fatalf("Status = %v, want StatusStopped", ans.Status)
	}
	assertToolCallsAnswered(t, ans.Messages)
}

// TestConversationDupCallStopKeepsHistorySendable runs the same stop inside
// a Conversation and checks the exact view the next turn sends: the history
// is appended to, never repaired afterwards, so the defect reaches the
// provider on the following request.
func TestConversationDupCallStopKeepsHistorySendable(t *testing.T) {
	same := func(id string) *llmhub.ToolCallContent { return llmhub.ToolCall(id, "echo", `{"n":1}`) }
	resetScript(t,
		callTurn(same("c1")),
		callTurn(same("c2")),
		callTurn(same("c3")),
		callTurn(same("c4"), same("c5")),
		textTurn("unreachable"),
	)
	ag := newFakeAgent(t, WithTools(echoTool()))
	conv := ag.Conversation()

	if _, err := conv.Ask(context.Background(), "Loop away."); err != nil {
		t.Fatalf("first turn: %v", err)
	}
	resetScript(t, textTurn("second turn"))
	if _, err := conv.Ask(context.Background(), "next"); err != nil {
		t.Fatalf("second turn: %v", err)
	}
	reqs := fakeRequests()
	if len(reqs) == 0 {
		t.Fatal("no provider request captured")
	}
	assertToolCallsAnswered(t, reqs[len(reqs)-1].prompt)
}

// assertToolCallsAnswered walks a transcript and fails when an assistant
// tool call never receives its tool result — the shape providers reject
// (Anthropic requires a tool_result per tool_use, OpenAI a tool message per
// call ID).
func assertToolCallsAnswered(t *testing.T, msgs []*llmhub.Message) {
	t.Helper()
	for i := 0; i < len(msgs); i++ {
		m := msgs[i]
		if m.Role != llmhub.RoleAssistant {
			continue
		}
		for _, part := range m.Content {
			call, ok := part.(*llmhub.ToolCallContent)
			if !ok {
				continue
			}
			if i+1 >= len(msgs) || msgs[i+1].Role != llmhub.RoleTool {
				t.Fatalf("tool call %q (%s) is unanswered", call.Name, call.ID)
			}
			if got := msgs[i+1].Meta["tool_call_id"]; got != call.ID {
				t.Fatalf("tool result answers call %q, want %q", got, call.ID)
			}
			i++
		}
	}
}
