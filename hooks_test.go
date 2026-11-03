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

// TestAfterToolCallRewritesResult pins the rewrite seam end to end: the
// transcript, the trace, and the emitted event all carry the payload the
// hook wrote, and the handler's original output never reaches the model.
func TestAfterToolCallRewritesResult(t *testing.T) {
	resetScript(t,
		callTurn(llmhub.ToolCall("c1", "echo", `{"secret":"hunter2"}`)),
		textTurn("done"),
	)
	ag := newFakeAgent(t, WithTools(echoTool()), WithHooks(Hooks{
		AfterToolCall: func(ctx context.Context, ev *Event) error {
			ev.Result = strings.ReplaceAll(ev.Result, "hunter2", "[redacted]")
			return nil
		},
	}))

	events, err := ag.AskStream(context.Background(), "Redact.")
	if err != nil {
		t.Fatalf("AskStream: %v", err)
	}
	evs := collectEvents(t, events)
	var emitted []string
	for _, ev := range evs {
		if ev.Kind == EventAfterToolCall {
			emitted = append(emitted, ev.Result)
		}
	}
	done := evs[len(evs)-1]
	if done.Kind != EventDone || done.Answer == nil {
		t.Fatalf("final event = %v, want Done with an answer", done.Kind)
	}
	ans := done.Answer
	if ans.Status != StatusCompleted || len(ans.Tools) != 1 {
		t.Fatalf("got Status=%v Tools=%d", ans.Status, len(ans.Tools))
	}

	want := `echo:{"secret":"[redacted]"}`
	if ans.Tools[0].Output != want {
		t.Errorf("trace output = %q, want %q", ans.Tools[0].Output, want)
	}
	if len(emitted) != 1 || emitted[0] != want {
		t.Errorf("emitted results = %q, want [%q]", emitted, want)
	}

	// The transcript carries the rewrite, never the handler output.
	toolMsgs := 0
	for _, m := range ans.Messages {
		if m.Role != llmhub.RoleTool {
			continue
		}
		toolMsgs++
		if got := textOf(m); got != want {
			t.Errorf("transcript tool message = %q, want %q", got, want)
		}
		if strings.Contains(textOf(m), "hunter2") {
			t.Errorf("original handler output leaked: %q", textOf(m))
		}
	}
	if toolMsgs != 1 {
		t.Fatalf("transcript tool messages = %d, want 1", toolMsgs)
	}

	// And the model received it: the step-2 prompt ends with that result.
	reqs := fakeRequests()
	if len(reqs) != 2 {
		t.Fatalf("requests = %d, want 2", len(reqs))
	}
	if got := textOf(reqs[1].prompt[len(reqs[1].prompt)-1]); got != want {
		t.Errorf("model saw %q, want %q", got, want)
	}
}

// TestAfterToolCallRewritesErrorObservation checks the error path: the
// default {"error": ...} rendering is what the hook sees, a rewrite replaces
// what the model receives, and ToolResult.Err still records the failure.
func TestAfterToolCallRewritesErrorObservation(t *testing.T) {
	errBoom := errors.New("secret value leaked")
	tool := echoTool()
	tool.Run = func(ctx context.Context, args json.RawMessage) (string, error) {
		return "", errBoom
	}
	resetScript(t,
		callTurn(llmhub.ToolCall("c1", "echo", `{}`)),
		textTurn("done"),
	)
	const scrubbed = `{"error":"[redacted]"}`
	var sawDefault string
	ag := newFakeAgent(t, WithTools(tool), WithHooks(Hooks{
		AfterToolCall: func(ctx context.Context, ev *Event) error {
			if ev.Err != nil {
				sawDefault = ev.Result
				ev.Result = scrubbed
			}
			return nil
		},
	}))

	ans, err := ag.Ask(context.Background(), "Fail.")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if want := errorJSON(errBoom); sawDefault != want {
		t.Errorf("hook saw %q, want the default error rendering %q", sawDefault, want)
	}
	if len(ans.Tools) != 1 {
		t.Fatalf("tools = %d, want 1", len(ans.Tools))
	}
	if !errors.Is(ans.Tools[0].Err, errBoom) {
		t.Errorf("trace Err = %v, want errBoom", ans.Tools[0].Err)
	}
	if ans.Tools[0].Output != scrubbed {
		t.Errorf("trace output = %q, want %q", ans.Tools[0].Output, scrubbed)
	}
	toolMsgs := 0
	for _, m := range ans.Messages {
		if m.Role != llmhub.RoleTool {
			continue
		}
		toolMsgs++
		if got := textOf(m); got != scrubbed {
			t.Errorf("transcript tool message = %q, want %q", got, scrubbed)
		}
	}
	if toolMsgs != 1 {
		t.Fatalf("transcript tool messages = %d, want 1", toolMsgs)
	}
}

// TestAfterToolCallRewriteAndDupDetection pins the interaction between the
// rewrite seam and loop detection: accounting judges the payload the model
// sees, not the handler's raw output.
func TestAfterToolCallRewriteAndDupDetection(t *testing.T) {
	const args = `{"n":1}`

	t.Run("identical rewrites extend the streak", func(t *testing.T) {
		resetScript(t,
			callTurn(llmhub.ToolCall("c1", "echo", args)),
			callTurn(llmhub.ToolCall("c2", "echo", args)),
			callTurn(llmhub.ToolCall("c3", "echo", args)),
			textTurn("never reached"),
		)
		ag := newFakeAgent(t, WithTools(echoTool()), WithDupCallLimit(2), WithHooks(Hooks{
			AfterToolCall: func(ctx context.Context, ev *Event) error {
				ev.Result = "masked"
				return nil
			},
		}))

		ans, err := ag.Ask(context.Background(), "Repeat.")
		if err != nil {
			t.Fatalf("Ask: %v", err)
		}
		if ans.Status != StatusStopped {
			t.Fatalf("Status = %v, want StatusStopped", ans.Status)
		}
		if len(ans.Tools) != 2 {
			t.Fatalf("tools = %d, want 2 executed (the third never runs)", len(ans.Tools))
		}
		if ans.Tools[0].Output != "masked" {
			t.Errorf("first output = %q, want the rewrite", ans.Tools[0].Output)
		}
		wantWarned := dupWarningObservation("masked")
		if ans.Tools[1].Output != wantWarned {
			t.Errorf("second output = %q, want the warning appended to the rewrite", ans.Tools[1].Output)
		}
		var toolMsgs []string
		for _, m := range ans.Messages {
			if m.Role == llmhub.RoleTool {
				toolMsgs = append(toolMsgs, textOf(m))
			}
		}
		if len(toolMsgs) != 3 || toolMsgs[1] != wantWarned || toolMsgs[2] != stopObservation {
			t.Errorf("transcript tool messages = %q, want masked, warned, stop", toolMsgs)
		}
	})

	t.Run("changing rewrites reset the streak", func(t *testing.T) {
		resetScript(t,
			callTurn(llmhub.ToolCall("c1", "echo", args)),
			callTurn(llmhub.ToolCall("c2", "echo", args)),
			callTurn(llmhub.ToolCall("c3", "echo", args)),
			textTurn("done"),
		)
		rewrites := []string{"redacted-a", "redacted-b", "redacted-c"}
		n := 0
		ag := newFakeAgent(t, WithTools(echoTool()), WithDupCallLimit(2), WithHooks(Hooks{
			AfterToolCall: func(ctx context.Context, ev *Event) error {
				ev.Result = rewrites[n]
				n++
				return nil
			},
		}))

		ans, err := ag.Ask(context.Background(), "Alternate rewrites.")
		if err != nil {
			t.Fatalf("Ask: %v", err)
		}
		if ans.Status != StatusCompleted || len(ans.Tools) != 3 {
			t.Fatalf("got Status=%v Tools=%d, want completed with 3 calls", ans.Status, len(ans.Tools))
		}
		for i, want := range rewrites {
			if ans.Tools[i].Output != want {
				t.Errorf("trace %d = %q, want %q", i, ans.Tools[i].Output, want)
			}
		}
		for _, m := range ans.Messages {
			if m.Role == llmhub.RoleTool && strings.Contains(textOf(m), dupWarning) {
				t.Errorf("unexpected duplicate warning for a changing rewrite: %q", textOf(m))
			}
		}
	})
}

// TestAfterToolCallRewriteParallelBatch runs two gated tools concurrently,
// releases them in reverse call order, and verifies each rewrite stays with
// its own call while results still append in call order.
func TestAfterToolCallRewriteParallelBatch(t *testing.T) {
	log := &callLog{}
	onstart := make(chan string, 2)
	gates := map[string]chan struct{}{"t1": make(chan struct{}), "t2": make(chan struct{})}
	t1 := gateTool(log, "t1", gates["t1"], onstart)
	t2 := gateTool(log, "t2", gates["t2"], onstart)
	resetScript(t,
		callTurn(
			llmhub.ToolCall("c1", "t1", `{}`),
			llmhub.ToolCall("c2", "t2", `{}`),
		),
		textTurn("ok"),
	)
	rewritten := make(chan string, 2) // rewritten results in completion order
	ag := newFakeAgent(t, WithTools(t1, t2), WithParallelTools(2), WithHooks(Hooks{
		AfterToolCall: func(ctx context.Context, ev *Event) error {
			ev.Result = ev.Result + "@" + ev.Call.ID
			rewritten <- ev.Result
			return nil
		},
	}))

	done := make(chan error, 1)
	var ans *Answer
	go func() {
		a, err := ag.Ask(context.Background(), "Fan out.")
		ans = a
		done <- err
	}()

	// Both handlers started before either gate opens: the batch is parallel.
	starts := map[string]bool{<-onstart: true, <-onstart: true}
	if !starts["t1"] || !starts["t2"] {
		t.Fatalf("starts = %v, want both t1 and t2", starts)
	}
	close(gates["t2"])
	if got := <-rewritten; got != "done:t2@c2" {
		t.Fatalf("first completion = %q, want done:t2@c2", got)
	}
	close(gates["t1"])
	if got := <-rewritten; got != "done:t1@c1" {
		t.Fatalf("second completion = %q, want done:t1@c1", got)
	}
	if err := <-done; err != nil {
		t.Fatalf("Ask: %v", err)
	}

	if len(ans.Tools) != 2 {
		t.Fatalf("tools = %d, want 2", len(ans.Tools))
	}
	want := []string{"done:t1@c1", "done:t2@c2"}
	for i, w := range want {
		if ans.Tools[i].Output != w {
			t.Errorf("trace %d = %q, want %q", i, ans.Tools[i].Output, w)
		}
	}
	var toolMsgs []string
	for _, m := range ans.Messages {
		if m.Role == llmhub.RoleTool {
			toolMsgs = append(toolMsgs, textOf(m))
		}
	}
	if len(toolMsgs) != 2 || toolMsgs[0] != want[0] || toolMsgs[1] != want[1] {
		t.Errorf("transcript tool messages = %q, want call order %q", toolMsgs, want)
	}
}

// TestAfterToolCallRewriteUnknownTool covers the unknown-name synthetic
// observation: the rewrite replaces what the model sees and what the trace
// records.
func TestAfterToolCallRewriteUnknownTool(t *testing.T) {
	resetScript(t,
		callTurn(llmhub.ToolCall("c1", "nope", `{}`)),
		textTurn("done"),
	)
	const scrubbed = `{"error":"[scrubbed]"}`
	var sawSynthetic string
	ag := newFakeAgent(t, WithHooks(Hooks{
		AfterToolCall: func(ctx context.Context, ev *Event) error {
			sawSynthetic = ev.Result
			ev.Result = scrubbed
			return nil
		},
	}))

	ans, err := ag.Ask(context.Background(), "Unknown.")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if !strings.Contains(sawSynthetic, "unknown tool") {
		t.Errorf("hook saw %q, want the synthetic unknown-tool observation", sawSynthetic)
	}
	if len(ans.Tools) != 1 || ans.Tools[0].Output != scrubbed {
		t.Fatalf("trace = %+v, want one rewritten entry", ans.Tools)
	}
	if ans.Tools[0].Err != nil {
		t.Errorf("trace Err = %v, want nil", ans.Tools[0].Err)
	}
	var toolMsgs []string
	for _, m := range ans.Messages {
		if m.Role == llmhub.RoleTool {
			toolMsgs = append(toolMsgs, textOf(m))
		}
	}
	if len(toolMsgs) != 1 || toolMsgs[0] != scrubbed {
		t.Errorf("transcript tool messages = %q, want [%q]", toolMsgs, scrubbed)
	}
}
