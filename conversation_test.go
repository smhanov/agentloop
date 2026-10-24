package agentloop

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/smhanov/llmhub"
)

func TestConversationHistoryAccumulation(t *testing.T) {
	resetScript(t,
		callTurn(llmhub.ToolCall("c1", "echo", `{"n":1}`)),
		textTurn("first answer"),
		textTurn("second answer"),
	)
	ag := newFakeAgent(t, WithTools(echoTool()), WithSystemPrompt("Be helpful."))
	conv := ag.Conversation()

	a1, err := conv.Ask(context.Background(), "question one")
	if err != nil {
		t.Fatalf("turn 1: %v", err)
	}
	if a1.Text != "first answer" || a1.Steps != 2 {
		t.Fatalf("turn 1 = (Text %q, Steps %d)", a1.Text, a1.Steps)
	}
	a2, err := conv.Ask(context.Background(), "question two")
	if err != nil {
		t.Fatalf("turn 2: %v", err)
	}
	if a2.Text != "second answer" {
		t.Fatalf("turn 2 Text = %q", a2.Text)
	}

	// Turn 2's prompt = system + turn 1's full run + the new user message.
	reqs := fakeRequests()
	if len(reqs) != 3 {
		t.Fatalf("requests = %d, want 3", len(reqs))
	}
	got := rolesOf(reqs[2].prompt)
	want := []llmhub.Role{
		llmhub.RoleSystem, llmhub.RoleUser, llmhub.RoleAssistant, llmhub.RoleTool, llmhub.RoleAssistant, llmhub.RoleUser,
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("turn 2 roles = %v, want %v", got, want)
	}
	if systems := countRoles(got, llmhub.RoleSystem); systems != 1 {
		t.Errorf("system messages = %d, want exactly 1", systems)
	}
	// Turn 2 sees turn 1's tool results.
	if got := textOf(reqs[2].prompt[3]); got != `echo:{"n":1}` {
		t.Errorf("turn 1 tool result in turn 2 context = %q", got)
	}
	// The conversation history starts with the system message and grows
	// turn by turn.
	wantHistory := []llmhub.Role{
		llmhub.RoleSystem, llmhub.RoleUser, llmhub.RoleAssistant, llmhub.RoleTool, llmhub.RoleAssistant, llmhub.RoleUser, llmhub.RoleAssistant,
	}
	if got := rolesOf(conv.Messages()); !reflect.DeepEqual(got, wantHistory) {
		t.Errorf("history roles = %v, want %v", got, wantHistory)
	}
	// Each Answer carries only its own run (no system, no prior turns).
	if got, want := len(a2.Messages), 2; got != want {
		t.Errorf("turn 2 Answer.Messages = %d, want %d", got, want)
	}
}

func countRoles(roles []llmhub.Role, want llmhub.Role) int {
	n := 0
	for _, r := range roles {
		if r == want {
			n++
		}
	}
	return n
}

func TestConversationBudgetsResetPerAsk(t *testing.T) {
	resetScript(t,
		callTurn(llmhub.ToolCall("c1", "echo", `{}`)),
		textTurn("one"),
		callTurn(llmhub.ToolCall("c2", "echo", `{}`)),
		textTurn("two"),
	)
	conv := newFakeAgent(t, WithTools(echoTool()), WithMaxToolCalls(1)).Conversation()

	a1, err := conv.Ask(context.Background(), "q1")
	if err != nil {
		t.Fatalf("turn 1: %v", err)
	}
	a2, err := conv.Ask(context.Background(), "q2")
	if err != nil {
		t.Fatalf("turn 2: %v", err)
	}
	if a1.ToolCalls != 1 || a2.ToolCalls != 1 {
		t.Errorf("ToolCalls = %d/%d, want 1/1 (budgets reset per turn)", a1.ToolCalls, a2.ToolCalls)
	}
	// Turn 2's first inference ran with the normal tool choice: proof the
	// budget was not carried over from turn 1.
	reqs := fakeRequests()
	if reqs[2].config.ToolChoice == nil || reqs[2].config.ToolChoice.Mode != llmhub.ToolChoiceAuto {
		t.Errorf("turn 2 first call tool choice = %+v, want auto", reqs[2].config.ToolChoice)
	}
}

func TestConversationTotalUsageAndReset(t *testing.T) {
	first := llmhub.UsageMetadata{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15, Cost: 0.01}
	second := llmhub.UsageMetadata{PromptTokens: 20, CompletionTokens: 7, TotalTokens: 27, Cost: 0.02}
	third := llmhub.UsageMetadata{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2, Cost: 0.001}
	turn1 := textTurn("one")
	turn1.usage = first
	turn2 := textTurn("two")
	turn2.usage = second
	turn3 := textTurn("three")
	turn3.usage = third
	resetScript(t, turn1, turn2, turn3)

	conv := newFakeAgent(t, WithSystemPrompt("sys")).Conversation()
	if _, err := conv.Ask(context.Background(), "q1"); err != nil {
		t.Fatalf("turn 1: %v", err)
	}
	if _, err := conv.Ask(context.Background(), "q2"); err != nil {
		t.Fatalf("turn 2: %v", err)
	}
	want := addUsage(first, second)
	if got := conv.TotalUsage(); got != want {
		t.Errorf("TotalUsage = %+v, want %+v", got, want)
	}

	conv.Reset()
	if got := conv.TotalUsage(); got != (llmhub.UsageMetadata{}) {
		t.Errorf("TotalUsage after Reset = %+v, want zero", got)
	}
	// The system message is kept: the history is exactly it.
	gotRoles := rolesOf(conv.Messages())
	if len(gotRoles) != 1 || gotRoles[0] != llmhub.RoleSystem {
		t.Errorf("history after Reset = %v, want [system]", gotRoles)
	}
	// Turn 3 is seeded with exactly the system message and its query.
	if _, err := conv.Ask(context.Background(), "q3"); err != nil {
		t.Fatalf("turn 3: %v", err)
	}
	reqs := fakeRequests()
	if len(reqs[2].prompt) != 2 {
		t.Fatalf("turn 3 prompt = %d messages, want 2", len(reqs[2].prompt))
	}
	if reqs[2].prompt[0].Role != llmhub.RoleSystem || textOf(reqs[2].prompt[0]) != "sys" {
		t.Errorf("turn 3 system = %v %q", reqs[2].prompt[0].Role, textOf(reqs[2].prompt[0]))
	}
	if got := conv.TotalUsage(); got != third {
		t.Errorf("TotalUsage after reset turn = %+v, want %+v", got, third)
	}
}

func TestConversationBusy(t *testing.T) {
	release := make(chan struct{})
	onstart := make(chan string, 1)
	blocker := gateTool(&callLog{}, "blocker", release, onstart)
	resetScript(t,
		callTurn(llmhub.ToolCall("c1", "blocker", `{}`)),
		textTurn("done"),
		textTurn("later"),
	)
	conv := newFakeAgent(t, WithTools(blocker)).Conversation()

	type result struct {
		ans *Answer
		err error
	}
	res1 := make(chan result, 1)
	go func() {
		a, e := conv.Ask(context.Background(), "q1")
		res1 <- result{a, e}
	}()
	<-onstart // turn 1 is in flight, holding the conversation

	if _, err := conv.Ask(context.Background(), "q2"); !errors.Is(err, ErrConversationBusy) {
		t.Errorf("overlapping Ask error = %v, want ErrConversationBusy", err)
	}
	if _, err := conv.AskStream(context.Background(), "q3"); !errors.Is(err, ErrConversationBusy) {
		t.Errorf("overlapping AskStream error = %v, want ErrConversationBusy", err)
	}

	close(release)
	r := <-res1
	if r.err != nil {
		t.Fatalf("turn 1: %v", r.err)
	}
	// Once the turn completes, the conversation is usable again.
	a3, err := conv.Ask(context.Background(), "q3")
	if err != nil {
		t.Fatalf("turn after busy: %v", err)
	}
	if a3.Text != "later" {
		t.Errorf("post-busy turn Text = %q", a3.Text)
	}
}

func TestConversationSystemFrozenAndNoOptionLeak(t *testing.T) {
	resetScript(t, textTurn("one"), textTurn("two"), textTurn("three"))
	var convStarts int
	ag := newFakeAgent(t,
		WithSystemPrompt("base"),
		WithGenerateOptions(llmhub.WithModel("base-model")))
	conv := ag.Conversation(WithHooks(Hooks{
		OnStart: func(ctx context.Context, ev *Event) error {
			convStarts++
			return nil
		},
	}))

	if _, err := conv.Ask(context.Background(), "q1"); err != nil {
		t.Fatalf("turn 1: %v", err)
	}
	// Turn-level options merge over the conversation defaults, but the
	// system prompt stays frozen.
	if _, err := conv.Ask(context.Background(), "q2",
		WithSystemPrompt("hijack"),
		WithGenerateOptions(llmhub.WithModel("turn-model"))); err != nil {
		t.Fatalf("turn 2: %v", err)
	}

	reqs := fakeRequests()
	if textOf(reqs[0].prompt[0]) != "base" || textOf(reqs[1].prompt[0]) != "base" {
		t.Errorf("system prompts = %q / %q, want frozen base",
			textOf(reqs[0].prompt[0]), textOf(reqs[1].prompt[0]))
	}
	if reqs[0].config.Model != "base-model" || reqs[1].config.Model != "turn-model" {
		t.Errorf("models = %q / %q, want base-model then turn-model",
			reqs[0].config.Model, reqs[1].config.Model)
	}
	if convStarts != 2 {
		t.Errorf("conversation hooks fired %d times, want 2", convStarts)
	}

	// The conversation's hooks do not leak into plain Agent calls.
	if _, err := ag.Ask(context.Background(), "plain"); err != nil {
		t.Fatalf("plain Ask: %v", err)
	}
	if convStarts != 2 {
		t.Errorf("conversation hooks fired %d times after plain Ask, want 2", convStarts)
	}
}

func TestConversationPrefixStability(t *testing.T) {
	resetScript(t,
		callTurn(llmhub.ToolCall("c1", "echo", `{}`)),
		textTurn("one"),
		textTurn("two"),
	)
	conv := newFakeAgent(t, WithTools(echoTool()), WithSystemPrompt("sys")).Conversation()
	if _, err := conv.Ask(context.Background(), "q1"); err != nil {
		t.Fatalf("turn 1: %v", err)
	}
	before := conv.Messages()
	if len(before) != 5 {
		t.Fatalf("turn 1 transcript = %d messages, want 5", len(before))
	}

	if _, err := conv.Ask(context.Background(), "q2"); err != nil {
		t.Fatalf("turn 2: %v", err)
	}
	after := conv.Messages()

	// The prefix is intact and identical, entry by entry.
	if len(after) <= len(before) {
		t.Fatalf("history shrank: %d messages", len(after))
	}
	for i := range before {
		if before[i] != after[i] {
			t.Errorf("history entry %d was replaced, want the same message", i)
		}
	}

	// Turn 2's prompt reproduces the turn-1 transcript verbatim.
	reqs := fakeRequests()
	if !equalMessages(reqs[2].prompt[:len(before)], before) {
		t.Error("turn 2 prompt does not reproduce the turn 1 transcript verbatim")
	}
}

func TestConversationBudgetExhaustedTurnStillRemembers(t *testing.T) {
	resetScript(t,
		callTurn(llmhub.ToolCall("c1", "echo", `{}`)),
		callTurn(llmhub.ToolCall("c2", "echo", `{}`)),
		callTurn(llmhub.ToolCall("c3", "echo", `{}`)),
	)
	conv := newFakeAgent(t, WithTools(echoTool()), WithMaxSteps(1)).Conversation()
	_, err := conv.Ask(context.Background(), "q1")
	if !errors.Is(err, ErrBudgetExhausted) {
		t.Fatalf("error = %v, want ErrBudgetExhausted", err)
	}
	// The partial turn (its own user + assistant + tool result) joins the
	// history so a follow-up can recover. This conversation has no system
	// prompt, so the history is exactly the run's messages.
	roles := rolesOf(conv.Messages())
	want := []llmhub.Role{llmhub.RoleUser, llmhub.RoleAssistant, llmhub.RoleTool}
	if !reflect.DeepEqual(roles, want) {
		t.Errorf("history roles = %v, want %v", roles, want)
	}
}
