package agentloop

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/smhanov/llmhub"
)

func TestNewNilClient(t *testing.T) {
	_, err := New(nil)
	if !errors.Is(err, ErrNoLLMClient) {
		t.Fatalf("New(nil) error = %v, want ErrNoLLMClient", err)
	}
}

func TestMustNewPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("MustNew(nil) did not panic")
		}
	}()
	MustNew(nil)
}

func TestAskEmptyQuery(t *testing.T) {
	ag := newFakeAgent(t)
	if _, err := ag.Ask(context.Background(), ""); !errors.Is(err, ErrInvalidInput) {
		t.Fatalf("Ask(\"\") error = %v, want ErrInvalidInput", err)
	}
}

func TestConfigDefaultsAndOverrides(t *testing.T) {
	ag := newFakeAgent(t)
	if got := ag.cfg.effectiveMaxToolCalls(); got != 5 {
		t.Errorf("default maxToolCalls = %d, want 5", got)
	}
	if got := ag.cfg.effectiveMaxSteps(); got != 12 {
		t.Errorf("default maxSteps = %d, want 12", got)
	}
	if ag.cfg.dupCallLimit != 3 {
		t.Errorf("default dupCallLimit = %d, want 3", ag.cfg.dupCallLimit)
	}
	if ag.cfg.toolChoice.Mode != llmhub.ToolChoiceAuto {
		t.Errorf("default toolChoice = %v, want auto", ag.cfg.toolChoice.Mode)
	}
	if ag.cfg.parallelTools != 3 {
		t.Errorf("default parallelTools = %d, want 3", ag.cfg.parallelTools)
	}
	if ag.cfg.streamWatchdog != 90*time.Second {
		t.Errorf("default streamWatchdog = %v, want 90s", ag.cfg.streamWatchdog)
	}

	// Explicit settings apply; MaxSteps defaults off the tool budget.
	ag = newFakeAgent(t, WithMaxToolCalls(2), WithDupCallLimit(0))
	if got := ag.cfg.effectiveMaxSteps(); got != 6 {
		t.Errorf("maxSteps after WithMaxToolCalls(2) = %d, want 6", got)
	}
	if got := ag.cfg.effectiveMaxToolCalls(); got != 2 {
		t.Errorf("maxToolCalls = %d, want 2", got)
	}
	if ag.cfg.dupCallLimit != 0 {
		t.Errorf("dupCallLimit = %d, want 0 (disabled)", ag.cfg.dupCallLimit)
	}

	// Values below the minimum are ignored.
	ag = newFakeAgent(t, WithMaxToolCalls(0), WithMaxSteps(-1), WithDupCallLimit(-2), WithParallelTools(0))
	if got := ag.cfg.effectiveMaxToolCalls(); got != 5 {
		t.Errorf("maxToolCalls = %d, want default 5", got)
	}
	if got := ag.cfg.effectiveMaxSteps(); got != 12 {
		t.Errorf("maxSteps = %d, want default 12", got)
	}
	if ag.cfg.dupCallLimit != 3 {
		t.Errorf("dupCallLimit = %d, want default 3", ag.cfg.dupCallLimit)
	}
	if got := ag.cfg.effectiveParallelTools(); got != 3 {
		t.Errorf("parallelTools = %d, want default 3", got)
	}

	// The watchdog default can be disabled or shortened explicitly.
	ag = newFakeAgent(t, WithStreamWatchdog(0))
	if ag.cfg.streamWatchdog != 0 {
		t.Errorf("streamWatchdog = %v, want 0 (disabled)", ag.cfg.streamWatchdog)
	}
}

func TestAskNoTools(t *testing.T) {
	resetScript(t, textTurn("The answer is forty-two."))
	ag := newFakeAgent(t, WithSystemPrompt("Be brief."))

	ans, err := ag.Ask(context.Background(), "What is the meaning of life?")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ans.Text != "The answer is forty-two." {
		t.Errorf("Text = %q", ans.Text)
	}
	if ans.Steps != 1 || ans.ToolCalls != 0 || ans.Status != StatusCompleted {
		t.Errorf("got Steps=%d ToolCalls=%d Status=%v", ans.Steps, ans.ToolCalls, ans.Status)
	}
	if ans.Timing == nil {
		t.Error("Timing = nil, want client telemetry")
	}

	// Transcript: system, user, assistant.
	wantRoles := []llmhub.Role{llmhub.RoleSystem, llmhub.RoleUser, llmhub.RoleAssistant}
	if !reflect.DeepEqual(rolesOf(ans.Messages), wantRoles) {
		t.Errorf("roles = %v, want %v", rolesOf(ans.Messages), wantRoles)
	}
	if got := textOf(ans.Messages[0]); got != "Be brief." {
		t.Errorf("system message = %q", got)
	}

	// Exactly one inference, seeded with the system prompt and the query.
	reqs := fakeRequests()
	if len(reqs) != 1 {
		t.Fatalf("got %d provider requests, want 1", len(reqs))
	}
	if len(reqs[0].prompt) != 2 {
		t.Fatalf("prompt has %d messages, want 2", len(reqs[0].prompt))
	}
	if got := textOf(reqs[0].prompt[0]); got != "Be brief." {
		t.Errorf("prompt system = %q", got)
	}
	if got := textOf(reqs[0].prompt[1]); got != "What is the meaning of life?" {
		t.Errorf("prompt user = %q", got)
	}
	if reqs[0].config.ToolChoice == nil || reqs[0].config.ToolChoice.Mode != llmhub.ToolChoiceAuto {
		t.Errorf("tool choice = %+v, want auto", reqs[0].config.ToolChoice)
	}
}

func TestAskSingleToolCall(t *testing.T) {
	resetScript(t,
		callTurn(llmhub.ToolCall("call-1", "echo", `{"text":"hi"}`)),
		textTurn("All done."),
	)
	ag := newFakeAgent(t, WithTools(echoTool()))

	ans, err := ag.Ask(context.Background(), "Say hi using the tool.")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ans.Steps != 2 || ans.ToolCalls != 1 || ans.Status != StatusCompleted {
		t.Fatalf("got Steps=%d ToolCalls=%d Status=%v", ans.Steps, ans.ToolCalls, ans.Status)
	}
	if ans.Text != "All done." {
		t.Errorf("Text = %q", ans.Text)
	}

	// Transcript order: user, assistant(tool call), tool result, assistant.
	wantRoles := []llmhub.Role{llmhub.RoleUser, llmhub.RoleAssistant, llmhub.RoleTool, llmhub.RoleAssistant}
	if !reflect.DeepEqual(rolesOf(ans.Messages), wantRoles) {
		t.Fatalf("roles = %v, want %v", rolesOf(ans.Messages), wantRoles)
	}
	calls := callsOf(ans.Messages[1])
	if len(calls) != 1 || calls[0].ID != "call-1" || calls[0].Name != "echo" || calls[0].Arguments != `{"text":"hi"}` {
		t.Fatalf("assistant tool call = %+v", calls)
	}
	result := ans.Messages[2]
	if got := textOf(result); got != `echo:{"text":"hi"}` {
		t.Errorf("tool result = %q", got)
	}
	if result.Meta["tool_call_id"] != "call-1" || result.Meta["name"] != "echo" {
		t.Errorf("tool result meta = %v", result.Meta)
	}

	// The trace records the execution.
	if len(ans.Tools) != 1 {
		t.Fatalf("Tools = %+v", ans.Tools)
	}
	tr := ans.Tools[0]
	if tr.Name != "echo" || tr.ID != "call-1" || tr.Err != nil || tr.Output != `echo:{"text":"hi"}` {
		t.Errorf("ToolResult = %+v", tr)
	}
	if string(tr.Args) != `{"text":"hi"}` {
		t.Errorf("Args = %s", tr.Args)
	}

	// The second inference saw the assistant call and its result.
	reqs := fakeRequests()
	if len(reqs) != 2 {
		t.Fatalf("got %d requests, want 2", len(reqs))
	}
	wantRoles = []llmhub.Role{llmhub.RoleUser, llmhub.RoleAssistant, llmhub.RoleTool}
	if got := rolesOf(reqs[1].prompt); !reflect.DeepEqual(got, wantRoles) {
		t.Errorf("second prompt roles = %v, want %v", got, wantRoles)
	}
}

func TestAskSequentialToolSteps(t *testing.T) {
	resetScript(t,
		callTurn(llmhub.ToolCall("c1", "echo", `{"n":1}`)),
		callTurn(llmhub.ToolCall("c2", "echo", `{"n":2}`)),
		textTurn("finished"),
	)
	ag := newFakeAgent(t, WithTools(echoTool()))

	ans, err := ag.Ask(context.Background(), "Do two things.")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ans.Steps != 3 || ans.ToolCalls != 2 || ans.Status != StatusCompleted {
		t.Fatalf("got Steps=%d ToolCalls=%d Status=%v", ans.Steps, ans.ToolCalls, ans.Status)
	}
	// Result messages appear in call order.
	if got := textOf(ans.Messages[2]); got != `echo:{"n":1}` {
		t.Errorf("first result = %q", got)
	}
	if got := textOf(ans.Messages[4]); got != `echo:{"n":2}` {
		t.Errorf("second result = %q", got)
	}
}

func TestAskMultipleCallsOneStepInOrder(t *testing.T) {
	resetScript(t,
		callTurn(
			llmhub.ToolCall("c1", "echo", `{"n":1}`),
			llmhub.ToolCall("c2", "echo", `{"n":2}`),
		),
		textTurn("ok"),
	)
	ag := newFakeAgent(t, WithTools(echoTool()))

	ans, err := ag.Ask(context.Background(), "Two calls at once.")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ans.Steps != 2 || ans.ToolCalls != 2 || ans.Status != StatusCompleted {
		t.Fatalf("got Steps=%d ToolCalls=%d Status=%v", ans.Steps, ans.ToolCalls, ans.Status)
	}
	// Both results are appended, in original call order.
	if got := ans.Messages[2].Meta["tool_call_id"]; got != "c1" {
		t.Errorf("first result call id = %q, want c1", got)
	}
	if got := textOf(ans.Messages[2]); got != `echo:{"n":1}` {
		t.Errorf("first result = %q", got)
	}
	if got := ans.Messages[3].Meta["tool_call_id"]; got != "c2" {
		t.Errorf("second result call id = %q, want c2", got)
	}
	if got := textOf(ans.Messages[3]); got != `echo:{"n":2}` {
		t.Errorf("second result = %q", got)
	}
}

func TestAskHandlerErrorContinues(t *testing.T) {
	boom := Tool{
		Def: llmhub.NewTool("boom", "Always fails", map[string]any{"type": "object"}),
		Run: func(ctx context.Context, args json.RawMessage) (string, error) {
			return "", errors.New("disk on fire")
		},
	}
	resetScript(t,
		callTurn(llmhub.ToolCall("c1", "boom", `{}`)),
		textTurn("recovered"),
	)
	ag := newFakeAgent(t, WithTools(boom))

	ans, err := ag.Ask(context.Background(), "Try the tool.")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ans.Status != StatusCompleted || ans.ToolCalls != 1 {
		t.Fatalf("got Status=%v ToolCalls=%d", ans.Status, ans.ToolCalls)
	}

	// The handler error became an in-band observation and the run finished.
	if got := textOf(ans.Messages[2]); got != `{"error":"disk on fire"}` {
		t.Errorf("observation = %q", got)
	}
	reqs := fakeRequests()
	if len(reqs) != 2 {
		t.Fatalf("requests = %d, want 2", len(reqs))
	}
	if got := textOf(reqs[1].prompt[2]); got != `{"error":"disk on fire"}` {
		t.Errorf("model saw %q", got)
	}
	if len(ans.Tools) != 1 || ans.Tools[0].Err == nil || ans.Tools[0].Output != `{"error":"disk on fire"}` {
		t.Errorf("trace = %+v", ans.Tools)
	}
	if ans.Tools[0].Err.Error() != "disk on fire" {
		t.Errorf("trace error = %v", ans.Tools[0].Err)
	}
}

func TestAskUnknownTool(t *testing.T) {
	cases := []struct {
		name   string
		called string
		want   string
	}{
		{"near miss suggests", "clck", `{"error":"unknown tool; did you mean \"clock\"?"}`},
		{"distant name has no suggestion", "widget", `{"error":"unknown tool"}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			resetScript(t,
				callTurn(llmhub.ToolCall("c1", tc.called, `{}`)),
				textTurn("adapted"),
			)
			ag := newFakeAgent(t, WithTools(clockTool()))

			ans, err := ag.Ask(context.Background(), "Use the tool.")
			if err != nil {
				t.Fatalf("Ask: %v", err)
			}
			if ans.Status != StatusCompleted {
				t.Fatalf("Status = %v", ans.Status)
			}
			if got := textOf(ans.Messages[2]); got != tc.want {
				t.Errorf("observation = %q, want %q", got, tc.want)
			}
			// Unknown names count against the tool budget.
			if ans.ToolCalls != 1 {
				t.Errorf("ToolCalls = %d, want 1", ans.ToolCalls)
			}
			if len(ans.Tools) != 1 || ans.Tools[0].Name != tc.called {
				t.Errorf("trace = %+v", ans.Tools)
			}
		})
	}
}

func TestAskDupCallWarningThenStop(t *testing.T) {
	same := func(id string) *llmhub.ToolCallContent { return llmhub.ToolCall(id, "echo", `{"n":1}`) }
	resetScript(t,
		callTurn(same("c1")),
		callTurn(same("c2")),
		callTurn(same("c3")),
		callTurn(same("c4")),
		textTurn("unreachable"),
	)
	var doneStatus AnswerStatus
	ag := newFakeAgent(t, WithTools(echoTool()), WithHooks(Hooks{
		OnDone: func(ctx context.Context, ev *Event) error {
			doneStatus = ev.Answer.Status
			return nil
		},
	}))

	ans, err := ag.Ask(context.Background(), "Loop away.")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ans.Status != StatusStopped || doneStatus != StatusStopped {
		t.Fatalf("got Status=%v OnDone=%v, want StatusStopped", ans.Status, doneStatus)
	}
	if ans.ToolCalls != 3 || ans.Steps != 4 {
		t.Errorf("got ToolCalls=%d Steps=%d, want 3/4", ans.ToolCalls, ans.Steps)
	}

	// Transcript: user, then (assistant, tool) three times with the warning
	// on the third result, then the fourth assistant call and its stop note.
	if got := textOf(ans.Messages[6]); got != warningObservation {
		t.Errorf("third observation = %q, want warning", got)
	}
	if got := textOf(ans.Messages[8]); got != stopObservation {
		t.Errorf("fourth observation = %q, want stop", got)
	}
	for _, idx := range []int{2, 4} {
		if got := textOf(ans.Messages[idx]); got != `echo:{"n":1}` {
			t.Errorf("result %d = %q", idx, got)
		}
	}
}

func TestAskDupCallCounterResets(t *testing.T) {
	resetScript(t,
		callTurn(llmhub.ToolCall("c1", "echo", `{"n":1}`)),
		callTurn(llmhub.ToolCall("c2", "echo", `{"n":2}`)),
		callTurn(llmhub.ToolCall("c3", "echo", `{"n":1}`)),
		callTurn(llmhub.ToolCall("c4", "echo", `{"n":1}`)),
		textTurn("ok"),
	)
	ag := newFakeAgent(t, WithTools(echoTool()))

	ans, err := ag.Ask(context.Background(), "Alternate calls.")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	// Two consecutive identical calls never reach the limit of three.
	if ans.Status != StatusCompleted || ans.ToolCalls != 4 || ans.Steps != 5 {
		t.Fatalf("got Status=%v ToolCalls=%d Steps=%d", ans.Status, ans.ToolCalls, ans.Steps)
	}
	for i, m := range ans.Messages {
		if strings.Contains(textOf(m), `"warning"`) {
			t.Errorf("message %d carries a warning: %q", i, textOf(m))
		}
	}
}

func TestAskBudgetForcesNoToolChoice(t *testing.T) {
	resetScript(t,
		callTurn(llmhub.ToolCall("c1", "echo", `{}`)),
		textTurn("forced answer"),
	)
	ag := newFakeAgent(t, WithTools(echoTool()), WithMaxToolCalls(1))

	ans, err := ag.Ask(context.Background(), "One call only.")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ans.Status != StatusCompleted || ans.Steps != 2 || ans.ToolCalls != 1 {
		t.Fatalf("got Status=%v Steps=%d ToolCalls=%d", ans.Status, ans.Steps, ans.ToolCalls)
	}
	if ans.Text != "forced answer" {
		t.Errorf("Text = %q", ans.Text)
	}

	reqs := fakeRequests()
	if reqs[0].config.ToolChoice == nil || reqs[0].config.ToolChoice.Mode != llmhub.ToolChoiceAuto {
		t.Errorf("first call tool choice = %+v, want auto", reqs[0].config.ToolChoice)
	}
	if reqs[1].config.ToolChoice == nil || reqs[1].config.ToolChoice.Mode != llmhub.ToolChoiceNone {
		t.Errorf("second call tool choice = %+v, want none", reqs[1].config.ToolChoice)
	}
}

func TestAskForcedFinalSkipsToolCalls(t *testing.T) {
	resetScript(t,
		callTurn(llmhub.ToolCall("c1", "echo", `{}`)),
		callTextTurn("giving up", llmhub.ToolCall("c2", "echo", `{}`)),
	)
	ag := newFakeAgent(t, WithTools(echoTool()), WithMaxToolCalls(1))

	ans, err := ag.Ask(context.Background(), "One call only.")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ans.Status != StatusCompleted || ans.ToolCalls != 1 || ans.Steps != 2 {
		t.Fatalf("got Status=%v ToolCalls=%d Steps=%d", ans.Status, ans.ToolCalls, ans.Steps)
	}
	if ans.Text != "giving up" {
		t.Errorf("Text = %q", ans.Text)
	}
	// The forced step's tool call was not executed: no result for c2.
	if got := len(ans.Messages); got != 4 {
		t.Fatalf("messages = %d, want 4", got)
	}
	for _, tr := range ans.Tools {
		if tr.ID == "c2" {
			t.Errorf("forced step executed call c2: %+v", tr)
		}
	}
}

func TestAskMaxStepsExhausted(t *testing.T) {
	cases := []struct {
		name         string
		maxToolCalls int
		maxSteps     int
		wantSteps    int
		wantCalls    int
	}{
		{"step ceiling below tool budget", 100, 3, 3, 3},
		{"tool budget consumed first", 1, 1, 1, 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			resetScript(t,
				callTurn(llmhub.ToolCall("c1", "echo", `{}`)),
				callTurn(llmhub.ToolCall("c2", "echo", `{}`)),
				callTurn(llmhub.ToolCall("c3", "echo", `{}`)),
				callTurn(llmhub.ToolCall("c4", "echo", `{}`)),
			)
			var reported error
			ag := newFakeAgent(t,
				WithTools(echoTool()),
				WithMaxToolCalls(tc.maxToolCalls),
				WithMaxSteps(tc.maxSteps),
				WithHooks(Hooks{
					OnError: func(ctx context.Context, ev *Event) error {
						reported = ev.Err
						return nil
					},
				}))

			ans, err := ag.Ask(context.Background(), "Never finish.")
			if !errors.Is(err, ErrBudgetExhausted) {
				t.Fatalf("error = %v, want ErrBudgetExhausted", err)
			}
			if !errors.Is(reported, ErrBudgetExhausted) {
				t.Errorf("OnError event err = %v", reported)
			}
			if ans == nil || ans.Status != StatusBudgetExhausted {
				t.Fatalf("status = %+v, want StatusBudgetExhausted", ans)
			}
			if ans.Steps != tc.wantSteps || ans.ToolCalls != tc.wantCalls {
				t.Errorf("Steps=%d ToolCalls=%d, want %d/%d", ans.Steps, ans.ToolCalls, tc.wantSteps, tc.wantCalls)
			}
			// The partial trace keeps every executed step's messages.
			if got, want := len(ans.Messages), 1+2*tc.wantSteps; got != want {
				t.Errorf("messages = %d, want %d", got, want)
			}
		})
	}
}

func TestAskUsageSummation(t *testing.T) {
	first := llmhub.UsageMetadata{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15, CacheReadTokens: 1, ReasoningTokens: 2, Cost: 0.01}
	second := llmhub.UsageMetadata{PromptTokens: 20, CompletionTokens: 7, TotalTokens: 27, CacheCreationTokens: 3, Cost: 0.02}
	turn1 := callTurn(llmhub.ToolCall("c1", "echo", `{}`))
	turn1.usage = first
	turn2 := textTurn("done")
	turn2.usage = second
	turn2.reasoning = "thinking"
	resetScript(t, turn1, turn2)
	ag := newFakeAgent(t, WithTools(echoTool()))

	ans, err := ag.Ask(context.Background(), "Sum usage.")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	want := llmhub.UsageMetadata{PromptTokens: 30, CompletionTokens: 12, TotalTokens: 42, CacheReadTokens: 1, CacheCreationTokens: 3, ReasoningTokens: 2, Cost: 0.03}
	if ans.Usage != want {
		t.Errorf("Usage = %+v, want %+v", ans.Usage, want)
	}
	if ans.ReasoningText != "thinking" {
		t.Errorf("ReasoningText = %q", ans.ReasoningText)
	}
	if ans.Timing == nil {
		t.Error("Timing = nil, want client telemetry")
	}
}

func TestAskSkipGracefulStop(t *testing.T) {
	resetScript(t, textTurn("never fetched"))
	var done *Answer
	ag := newFakeAgent(t, WithHooks(Hooks{
		BeforeGenerate: func(ctx context.Context, ev *Event) error {
			ev.Skip = true
			return nil
		},
		OnDone: func(ctx context.Context, ev *Event) error {
			done = ev.Answer
			return nil
		},
	}))

	ans, err := ag.Ask(context.Background(), "Stop early.")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ans.Status != StatusStopped {
		t.Fatalf("Status = %v, want StatusStopped", ans.Status)
	}
	if ans.Steps != 0 || ans.ToolCalls != 0 || ans.Text != "" {
		t.Errorf("got Steps=%d ToolCalls=%d Text=%q", ans.Steps, ans.ToolCalls, ans.Text)
	}
	if done == nil || done.Status != StatusStopped {
		t.Errorf("OnDone answer = %+v", done)
	}
	if reqs := fakeRequests(); len(reqs) != 0 {
		t.Errorf("provider called %d times, want 0", len(reqs))
	}
}

func TestAskPerCallOverrides(t *testing.T) {
	resetScript(t, textTurn("ok"), textTurn("ok again"))
	ag := newFakeAgent(t,
		WithSystemPrompt("base prompt"),
		WithGenerateOptions(llmhub.WithModel("model-default")))

	if _, err := ag.Ask(context.Background(), "q",
		WithSystemPrompt("override prompt"),
		WithGenerateOptions(llmhub.WithTemperature(0.25))); err != nil {
		t.Fatalf("Ask: %v", err)
	}
	reqs := fakeRequests()
	if got := textOf(reqs[0].prompt[0]); got != "override prompt" {
		t.Errorf("overridden system = %q", got)
	}
	if reqs[0].config.Temperature == nil || *reqs[0].config.Temperature != 0.25 {
		t.Errorf("temperature = %v, want 0.25", reqs[0].config.Temperature)
	}
	if reqs[0].config.Model != "model-default" {
		t.Errorf("agent default lost: model = %q", reqs[0].config.Model)
	}

	// The overrides do not leak into the next Ask.
	if _, err := ag.Ask(context.Background(), "q2"); err != nil {
		t.Fatalf("second Ask: %v", err)
	}
	reqs = fakeRequests()
	if got := textOf(reqs[1].prompt[0]); got != "base prompt" {
		t.Errorf("second Ask system = %q, want base prompt", got)
	}
	if reqs[1].config.Temperature != nil {
		t.Errorf("temperature leaked: %v", *reqs[1].config.Temperature)
	}
}

func TestAskCtxCancelMidTool(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	canceling := Tool{
		Def: llmhub.NewTool("canceler", "Cancels the run", map[string]any{"type": "object"}),
		Run: func(ctx context.Context, args json.RawMessage) (string, error) {
			cancel()
			return "", ctx.Err()
		},
	}
	resetScript(t,
		callTurn(llmhub.ToolCall("c1", "canceler", `{}`)),
		textTurn("unreachable"),
	)
	ag := newFakeAgent(t, WithTools(canceling))

	ans, err := ag.Ask(ctx, "Cancel me.")
	if err == nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want context.Canceled", err)
	}
	if ans != nil {
		t.Errorf("Answer = %+v, want nil", ans)
	}
	if reqs := fakeRequests(); len(reqs) != 1 {
		t.Errorf("requests = %d, want 1", len(reqs))
	}
}

func TestOnStartEvent(t *testing.T) {
	resetScript(t, textTurn("hi"))
	var queries []string
	ag := newFakeAgent(t, WithSystemPrompt("sys"), WithHooks(Hooks{
		OnStart: func(ctx context.Context, ev *Event) error {
			if ev.Kind != EventStart {
				t.Errorf("kind = %v, want EventStart", ev.Kind)
			}
			queries = append(queries, ev.Query)
			return nil
		},
	}))

	if _, err := ag.Ask(context.Background(), "hello"); err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if !reflect.DeepEqual(queries, []string{"hello"}) {
		t.Errorf("OnStart queries = %v", queries)
	}
}

func TestAfterGenerateFinishVetoStops(t *testing.T) {
	resetScript(t,
		callTurn(llmhub.ToolCall("c1", "echo", `{}`)),
		textTurn("never reached"),
	)
	var finishes []bool
	ran := false
	tool := echoTool()
	tool.Run = func(ctx context.Context, args json.RawMessage) (string, error) {
		ran = true
		return "", nil
	}
	ag := newFakeAgent(t, WithTools(tool), WithHooks(Hooks{
		AfterGenerate: func(ctx context.Context, ev *Event) error {
			finishes = append(finishes, ev.Finish)
			ev.Finish = true // veto: stop without executing the tools
			return nil
		},
	}))

	ans, err := ag.Ask(context.Background(), "Veto the tools.")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	// Finish arrived pre-set to false (tool calls present); the flip stopped
	// the run before execution.
	if !reflect.DeepEqual(finishes, []bool{false}) {
		t.Errorf("finishes = %v", finishes)
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

func TestBeforeToolCallRewritesArgs(t *testing.T) {
	resetScript(t,
		callTurn(llmhub.ToolCall("c1", "echo", `{"text":"original"}`)),
		textTurn("done"),
	)
	ag := newFakeAgent(t, WithTools(echoTool()), WithHooks(Hooks{
		BeforeToolCall: func(ctx context.Context, ev *Event) error {
			ev.Args = json.RawMessage(`{"text":"rewritten"}`)
			return nil
		},
	}))

	ans, err := ag.Ask(context.Background(), "Rewrite me.")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if got := textOf(ans.Messages[2]); got != `echo:{"text":"rewritten"}` {
		t.Errorf("observation = %q", got)
	}
	if string(ans.Tools[0].Args) != `{"text":"rewritten"}` {
		t.Errorf("trace args = %s", ans.Tools[0].Args)
	}
	// The model saw the rewritten arguments.
	reqs := fakeRequests()
	if got := textOf(reqs[1].prompt[2]); got != `echo:{"text":"rewritten"}` {
		t.Errorf("model saw %q", got)
	}
}

func TestSkipToolCallVeto(t *testing.T) {
	resetScript(t,
		callTurn(llmhub.ToolCall("c1", "echo", `{}`)),
		textTurn("done"),
	)
	ran := false
	tool := echoTool()
	tool.Run = func(ctx context.Context, args json.RawMessage) (string, error) {
		ran = true
		return "should not run", nil
	}
	ag := newFakeAgent(t, WithTools(tool), WithHooks(Hooks{
		BeforeToolCall: func(ctx context.Context, ev *Event) error {
			return ErrSkipToolCall
		},
	}))

	ans, err := ag.Ask(context.Background(), "Veto the tool.")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ran {
		t.Error("handler ran despite ErrSkipToolCall")
	}
	if ans.ToolCalls != 0 || len(ans.Tools) != 0 {
		t.Errorf("ToolCalls=%d Tools=%+v, want no executions", ans.ToolCalls, ans.Tools)
	}
	if got := textOf(ans.Messages[2]); got != skippedResult {
		t.Errorf("result message = %q, want %q", got, skippedResult)
	}
	if ans.Status != StatusCompleted {
		t.Errorf("Status = %v", ans.Status)
	}
}

func TestHookErrorAborts(t *testing.T) {
	errBoom := errors.New("hook exploded")
	resetScript(t, textTurn("never fetched"))
	var reported error
	ag := newFakeAgent(t, WithHooks(Hooks{
		BeforeGenerate: func(ctx context.Context, ev *Event) error { return errBoom },
		OnError:        func(ctx context.Context, ev *Event) error { reported = ev.Err; return nil },
	}))

	ans, err := ag.Ask(context.Background(), "q")
	if !errors.Is(err, errBoom) {
		t.Fatalf("error = %v, want errBoom", err)
	}
	if !strings.Contains(err.Error(), "BeforeGenerate") {
		t.Errorf("error %q does not name the hook", err)
	}
	if !errors.Is(reported, errBoom) {
		t.Errorf("OnError err = %v", reported)
	}
	if ans != nil {
		t.Errorf("Answer = %+v, want nil", ans)
	}
	if reqs := fakeRequests(); len(reqs) != 0 {
		t.Errorf("provider called %d times, want 0", len(reqs))
	}
}
