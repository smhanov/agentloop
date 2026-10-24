package agentloop

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/smhanov/llmhub"
)

func TestFinishVetoContinuesWithObservation(t *testing.T) {
	resetScript(t, textTurn("first"), textTurn("second"))
	stepped := 0
	ag := newFakeAgent(t, WithHooks(Hooks{
		AfterGenerate: func(ctx context.Context, ev *Event) error {
			stepped++
			if stepped == 1 {
				// Text-action protocol: append an observation message and
				// keep the loop running.
				ev.Messages = append(ev.Messages, llmhub.NewUserMessage(llmhub.Text("observation: continue")))
				ev.Finish = false
			}
			return nil
		},
	}))

	ans, err := ag.Ask(context.Background(), "Keep going.")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ans.Status != StatusCompleted || ans.Steps != 2 || ans.Text != "second" {
		t.Fatalf("got Status=%v Steps=%d Text=%q", ans.Status, ans.Steps, ans.Text)
	}
	wantRoles := []llmhub.Role{llmhub.RoleUser, llmhub.RoleAssistant, llmhub.RoleUser, llmhub.RoleAssistant}
	if got := rolesOf(ans.Messages); !reflect.DeepEqual(got, wantRoles) {
		t.Errorf("roles = %v, want %v", got, wantRoles)
	}
	if got := textOf(ans.Messages[2]); got != "observation: continue" {
		t.Errorf("observation message = %q", got)
	}
	// The second inference saw the appended observation.
	reqs := fakeRequests()
	if len(reqs) != 2 {
		t.Fatalf("requests = %d, want 2", len(reqs))
	}
	if got := textOf(reqs[1].prompt[2]); got != "observation: continue" {
		t.Errorf("model saw %q", got)
	}
}

func TestFinishVetoTrueSkipsTools(t *testing.T) {
	resetScript(t,
		callTurn(llmhub.ToolCall("c1", "echo", `{}`)),
		textTurn("never reached"),
	)
	ran := false
	tool := echoTool()
	tool.Run = func(ctx context.Context, args json.RawMessage) (string, error) {
		ran = true
		return "", nil
	}
	ag := newFakeAgent(t, WithTools(tool), WithHooks(Hooks{
		AfterGenerate: func(ctx context.Context, ev *Event) error {
			ev.Finish = true // veto: stop without executing the tools
			return nil
		},
	}))

	ans, err := ag.Ask(context.Background(), "Veto the tools.")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ran {
		t.Error("tools executed despite a Finish=true veto")
	}
	if ans.ToolCalls != 0 || len(ans.Tools) != 0 || ans.Steps != 1 || ans.Status != StatusCompleted {
		t.Errorf("got ToolCalls=%d Tools=%d Steps=%d Status=%v", ans.ToolCalls, len(ans.Tools), ans.Steps, ans.Status)
	}
	// The unanswered assistant tool-call message is still in the transcript.
	if got := len(callsOf(ans.Messages[1])); got != 1 {
		t.Errorf("transcript tool calls = %d, want 1", got)
	}
}

func TestFinishVetoIgnoredOnForcedFinal(t *testing.T) {
	// The forced-final path is the loop's termination guarantee: even a
	// vetoing hook cannot extend a run whose tool budget is spent.
	resetScript(t,
		callTurn(llmhub.ToolCall("c1", "echo", `{}`)),
		callTextTurn("giving up", llmhub.ToolCall("c2", "echo", `{}`)),
	)
	ag := newFakeAgent(t, WithTools(echoTool()), WithMaxToolCalls(1), WithHooks(Hooks{
		AfterGenerate: func(ctx context.Context, ev *Event) error {
			ev.Finish = false // try to keep the loop running
			return nil
		},
	}))

	ans, err := ag.Ask(context.Background(), "One call only.")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ans.Status != StatusCompleted || ans.Steps != 2 || ans.ToolCalls != 1 {
		t.Fatalf("got Status=%v Steps=%d ToolCalls=%d", ans.Status, ans.Steps, ans.ToolCalls)
	}
	if ans.Text != "giving up" {
		t.Errorf("Text = %q", ans.Text)
	}
	for _, tr := range ans.Tools {
		if tr.ID == "c2" {
			t.Errorf("forced step executed call c2: %+v", tr)
		}
	}
}

func TestRebuildContextObserve(t *testing.T) {
	resetScript(t,
		callTurn(llmhub.ToolCall("c1", "echo", `{}`)),
		textTurn("done"),
	)
	var seenContexts, seenMessages [][]*llmhub.Message
	ag := newFakeAgent(t, WithTools(echoTool()), WithHooks(Hooks{
		RebuildContext: func(ctx context.Context, ev *Event) error {
			if ev.Kind != EventRebuildContext {
				t.Errorf("kind = %v", ev.Kind)
			}
			// Pre-seeded: the context starts as the full transcript.
			if !equalMessages(ev.Context, ev.Messages) {
				t.Errorf("step %d: context not pre-seeded with the transcript", ev.Step)
			}
			seenContexts = append(seenContexts, snapshotMessages(ev.Context))
			seenMessages = append(seenMessages, snapshotMessages(ev.Messages))
			return nil
		},
	}))

	if _, err := ag.Ask(context.Background(), "Observe."); err != nil {
		t.Fatalf("Ask: %v", err)
	}
	reqs := fakeRequests()
	if len(reqs) != 2 || len(seenContexts) != 2 {
		t.Fatalf("requests=%d contexts=%d, want 2/2", len(reqs), len(seenContexts))
	}
	for i := range reqs {
		if !equalMessages(seenContexts[i], reqs[i].prompt) {
			t.Errorf("step %d: observed context != sent prompt", i+1)
		}
		if !equalMessages(seenMessages[i], reqs[i].prompt) {
			t.Errorf("step %d: transcript != sent prompt (legacy behavior)", i+1)
		}
	}
}

func TestRebuildContextReplace(t *testing.T) {
	script := []fakeTurn{
		callTurn(llmhub.ToolCall("c1", "echo", `{}`)),
		textTurn("done"),
	}
	compress := func(ctx context.Context, ev *Event) error {
		if len(ev.Messages) <= 2 {
			return nil
		}
		// Drop the oldest user message but always keep the trailing
		// assistant tool-call message and its result.
		projection := []*llmhub.Message{ev.Messages[0]}
		projection = append(projection, ev.Messages[len(ev.Messages)-2:]...)
		ev.Context = projection
		return nil
	}

	resetScript(t, script...)
	plain := newFakeAgent(t, WithTools(echoTool()), WithSystemPrompt("sys"))
	plainAns, err := plain.Ask(context.Background(), "Compress me.")
	if err != nil {
		t.Fatalf("plain Ask: %v", err)
	}

	resetScript(t, script...)
	hooked := newFakeAgent(t, WithTools(echoTool()), WithSystemPrompt("sys"), WithHooks(Hooks{RebuildContext: compress}))
	hookedAns, err := hooked.Ask(context.Background(), "Compress me.")
	if err != nil {
		t.Fatalf("hooked Ask: %v", err)
	}

	// The replacement is ephemeral: the transcript is byte-identical to an
	// un-hooked run.
	if !equalMessages(hookedAns.Messages, plainAns.Messages) {
		t.Errorf("transcript changed under RebuildContext:\nhooked: %v\nplain: %v",
			rolesOf(hookedAns.Messages), rolesOf(plainAns.Messages))
	}

	// The model saw the projection: step 2 drops the original user message
	// but keeps the trailing tool-call/result pair.
	reqs := fakeRequests()
	if len(reqs) != 2 {
		t.Fatalf("requests = %d, want 2", len(reqs))
	}
	step2 := reqs[1].prompt
	wantRoles := []llmhub.Role{llmhub.RoleSystem, llmhub.RoleAssistant, llmhub.RoleTool}
	if got := rolesOf(step2); !reflect.DeepEqual(got, wantRoles) {
		t.Fatalf("step 2 sent roles = %v, want %v", got, wantRoles)
	}
	if len(callsOf(step2[1])) != 1 || step2[2].Meta["tool_call_id"] != "c1" {
		t.Error("trailing assistant tool-call/result pair did not survive the projection")
	}
}

func TestSteeringAppendsReachTranscriptAndContext(t *testing.T) {
	resetScript(t,
		callTurn(llmhub.ToolCall("c1", "echo", `{}`)),
		textTurn("done"),
	)
	ag := newFakeAgent(t, WithTools(echoTool()), WithHooks(Hooks{
		BeforeGenerate: func(ctx context.Context, ev *Event) error {
			if ev.Step == 2 {
				ev.Messages = append(ev.Messages, llmhub.NewUserMessage(llmhub.Text("steer: be brief")))
			}
			return nil
		},
	}))

	ans, err := ag.Ask(context.Background(), "Steer me.")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	// The append is a permanent transcript entry.
	found := false
	for _, m := range ans.Messages {
		if textOf(m) == "steer: be brief" {
			found = true
		}
	}
	if !found {
		t.Error("steering message missing from the transcript")
	}
	// And the model received it (default context == transcript): it is the
	// last message of the step-2 prompt.
	reqs := fakeRequests()
	if got := textOf(reqs[1].prompt[len(reqs[1].prompt)-1]); got != "steer: be brief" {
		t.Errorf("model saw %q as the last prompt message", got)
	}
}

func TestRebuildContextDeterministicAcrossTurns(t *testing.T) {
	compress := func(ctx context.Context, ev *Event) error {
		if len(ev.Messages) <= 3 {
			return nil
		}
		projection := []*llmhub.Message{ev.Messages[0]}
		projection = append(projection, ev.Messages[len(ev.Messages)-2:]...)
		ev.Context = projection
		return nil
	}
	runOnce := func(t *testing.T) [][]*llmhub.Message {
		t.Helper()
		resetScript(t,
			callTurn(llmhub.ToolCall("c1", "echo", `{}`)),
			textTurn("one"),
			textTurn("two"),
		)
		ag := newFakeAgent(t, WithTools(echoTool()), WithSystemPrompt("sys"))
		conv := ag.Conversation(WithHooks(Hooks{RebuildContext: compress}))
		if _, err := conv.Ask(context.Background(), "q1"); err != nil {
			t.Fatalf("turn 1: %v", err)
		}
		if _, err := conv.Ask(context.Background(), "q2"); err != nil {
			t.Fatalf("turn 2: %v", err)
		}
		var contexts [][]*llmhub.Message
		for _, req := range fakeRequests() {
			contexts = append(contexts, req.prompt)
		}
		return contexts
	}

	first := runOnce(t)
	second := runOnce(t)
	if len(first) != len(second) {
		t.Fatalf("context counts differ: %d vs %d", len(first), len(second))
	}
	for i := range first {
		if !equalMessages(first[i], second[i]) {
			t.Errorf("step %d: projection not deterministic", i+1)
		}
	}
}

func TestHookErrorNamesRebuildContext(t *testing.T) {
	errBoom := errors.New("rebuild exploded")
	resetScript(t, textTurn("never fetched"))
	ag := newFakeAgent(t, WithHooks(Hooks{
		RebuildContext: func(ctx context.Context, ev *Event) error { return errBoom },
	}))
	_, err := ag.Ask(context.Background(), "q")
	if !errors.Is(err, errBoom) {
		t.Fatalf("error = %v, want errBoom", err)
	}
	if !strings.Contains(err.Error(), "RebuildContext") {
		t.Errorf("error %q does not name the hook", err)
	}
}
