package agentloop

import (
	"context"
	"reflect"
	"strings"
	"testing"

	"github.com/smhanov/llmhub"
)

// flightPrefix is a prior conversation carrying a system message, a user
// turn, an assistant tool call, and its result — the shape a caller loads
// from external storage.
func flightPrefix() []*llmhub.Message {
	return []*llmhub.Message{
		llmhub.NewSystemMessage(llmhub.Text("You are a historian.")),
		llmhub.NewUserMessage(llmhub.Text("When was the first powered flight?")),
		llmhub.NewMessage(llmhub.RoleAssistant,
			llmhub.Text("Checking the date."),
			llmhub.ToolCall("call-1", "clock", `{}`)),
		llmhub.NewToolResultMessage("call-1", "clock", llmhub.Text("1903-12-17")),
	}
}

// TestWithPrefixSeedsTranscript: the provider request and every hook's
// ev.Messages begin with the prefix, and Answer.Messages is the full
// transcript, prefix included.
func TestWithPrefixSeedsTranscript(t *testing.T) {
	resetScript(t, textTurn("The Wright brothers."))
	prefix := flightPrefix()
	query := llmhub.NewUserMessage(llmhub.Text("Who flew it?"))
	want := append(append([]*llmhub.Message(nil), prefix...), query)

	var start, before, after, rebuilt []*llmhub.Message
	ag := newFakeAgent(t, WithHooks(Hooks{
		OnStart: func(ctx context.Context, ev *Event) error {
			start = snapshotMessages(ev.Messages)
			return nil
		},
		BeforeGenerate: func(ctx context.Context, ev *Event) error {
			before = snapshotMessages(ev.Messages)
			return nil
		},
		AfterGenerate: func(ctx context.Context, ev *Event) error {
			after = snapshotMessages(ev.Messages)
			return nil
		},
		RebuildContext: func(ctx context.Context, ev *Event) error {
			rebuilt = snapshotMessages(ev.Messages)
			return nil
		},
	}))

	ans, err := ag.Ask(context.Background(), "Who flew it?", WithPrefix(prefix...))
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}

	reqs := fakeRequests()
	if len(reqs) != 1 {
		t.Fatalf("requests = %d, want 1", len(reqs))
	}
	if !equalMessages(reqs[0].prompt, want) {
		t.Error("provider request does not begin with the prefix followed by the query")
	}
	if !equalMessages(start, want) {
		t.Error("OnStart messages do not carry the prefix and query")
	}
	if !equalMessages(before, want) {
		t.Error("BeforeGenerate messages do not carry the prefix and query")
	}
	if !equalMessages(rebuilt, want) {
		t.Error("RebuildContext messages do not carry the prefix and query")
	}
	if len(after) != len(want)+1 || !equalMessages(after[:len(want)], want) {
		t.Error("AfterGenerate messages do not carry the full transcript")
	}

	// Answer.Messages is the FULL transcript: prefix + query + response.
	if len(ans.Messages) != len(want)+1 {
		t.Fatalf("Answer.Messages = %d messages, want the full %d-message transcript", len(ans.Messages), len(want)+1)
	}
	if !equalMessages(ans.Messages[:len(want)], want) {
		t.Error("Answer.Messages does not include the prefix at its head")
	}
	if ans.Messages[len(want)].Role != llmhub.RoleAssistant {
		t.Errorf("final message role = %v, want assistant", ans.Messages[len(want)].Role)
	}
}

// TestWithPrefixRoundTrip is the external-store journey: persist the first
// answer's full transcript and pass it back as the next turn's prefix.
func TestWithPrefixRoundTrip(t *testing.T) {
	resetScript(t, textTurn("first answer"), textTurn("second answer"))
	ag := newFakeAgent(t, WithSystemPrompt("You keep exact history."))

	first, err := ag.Ask(context.Background(), "question one")
	if err != nil {
		t.Fatalf("first Ask: %v", err)
	}
	stored := append([]*llmhub.Message(nil), first.Messages...)

	second, err := ag.Ask(context.Background(), "question two", WithPrefix(stored...))
	if err != nil {
		t.Fatalf("second Ask: %v", err)
	}

	reqs := fakeRequests()
	if len(reqs) != 2 {
		t.Fatalf("requests = %d, want 2", len(reqs))
	}
	want := append(append([]*llmhub.Message(nil), stored...), llmhub.NewUserMessage(llmhub.Text("question two")))
	if !equalMessages(reqs[1].prompt, want) {
		t.Error("round-tripped request head does not equal the stored transcript")
	}

	// The second answer's full transcript starts with everything stored and
	// appends the new exchange.
	if len(second.Messages) != len(stored)+2 {
		t.Fatalf("second Answer.Messages = %d messages, want %d", len(second.Messages), len(stored)+2)
	}
	if !equalMessages(second.Messages[:len(stored)], stored) {
		t.Error("second Answer.Messages lost the round-tripped prefix")
	}
}

// TestWithPrefixDoesNotMutateCaller: an empty-arguments tool call in the
// prefix is normalized on the copy; the caller's message stays untouched.
func TestWithPrefixDoesNotMutateCaller(t *testing.T) {
	resetScript(t, textTurn("done"))
	call := llmhub.ToolCall("call-1", "clock", "")
	prefix := []*llmhub.Message{
		llmhub.NewAssistantMessage(call),
		llmhub.NewToolResultMessage("call-1", "clock", llmhub.Text("ok")),
	}
	ag := newFakeAgent(t)

	ans, err := ag.Ask(context.Background(), "continue", WithPrefix(prefix...))
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}

	if call.Arguments != "" {
		t.Errorf("caller tool-call arguments mutated to %q, want empty", call.Arguments)
	}
	if prefix[0].Content[0].(*llmhub.ToolCallContent).Arguments != "" {
		t.Error("caller message content was mutated")
	}
	if ans.Messages[0] == prefix[0] {
		t.Error("transcript shares the caller's message object, want a copy")
	}
	if got := ans.Messages[0].Content[0].(*llmhub.ToolCallContent).Arguments; got != "{}" {
		t.Errorf("transcript tool-call arguments = %q, want {}", got)
	}
	reqs := fakeRequests()
	if got := reqs[0].prompt[0].Content[0].(*llmhub.ToolCallContent).Arguments; got != "{}" {
		t.Errorf("request tool-call arguments = %q, want {}", got)
	}
}

// TestWithPrefixNoSystemDuplication: a prefix owns the opening messages, so
// the configured system prompt is not prepended; a per-call prefix replaces
// a construction-time prefix.
func TestWithPrefixNoSystemDuplication(t *testing.T) {
	resetScript(t, textTurn("one"), textTurn("two"))
	basePrefix := []*llmhub.Message{
		llmhub.NewSystemMessage(llmhub.Text("prefix system")),
		llmhub.NewUserMessage(llmhub.Text("earlier question")),
	}
	ag := newFakeAgent(t, WithSystemPrompt("agent system"), WithPrefix(basePrefix...))

	if _, err := ag.Ask(context.Background(), "next"); err != nil {
		t.Fatalf("first Ask: %v", err)
	}
	reqs := fakeRequests()
	if n := countRoles(rolesOf(reqs[0].prompt), llmhub.RoleSystem); n != 1 {
		t.Errorf("system messages = %d, want exactly the prefix's one", n)
	}
	if got := textOf(reqs[0].prompt[0]); got != "prefix system" {
		t.Errorf("leading message = %q, want the prefix's system message", got)
	}

	override := []*llmhub.Message{llmhub.NewUserMessage(llmhub.Text("replacement history"))}
	if _, err := ag.Ask(context.Background(), "again", WithPrefix(override...)); err != nil {
		t.Fatalf("second Ask: %v", err)
	}
	reqs = fakeRequests()
	want := append(append([]*llmhub.Message(nil), override...), llmhub.NewUserMessage(llmhub.Text("again")))
	if !equalMessages(reqs[1].prompt, want) {
		t.Error("per-call prefix did not replace the construction prefix")
	}
}

// TestConversationIgnoresWithPrefix: a conversation turn's own history is
// the prefix; a per-turn WithPrefix changes nothing about the request.
func TestConversationIgnoresWithPrefix(t *testing.T) {
	resetScript(t, textTurn("answer"))
	prefix := []*llmhub.Message{
		llmhub.NewSystemMessage(llmhub.Text("injected system")),
		llmhub.NewUserMessage(llmhub.Text("injected question")),
	}
	conv := newFakeAgent(t, WithSystemPrompt("conversation system")).Conversation()

	ans, err := conv.Ask(context.Background(), "real question", WithPrefix(prefix...))
	if err != nil {
		t.Fatalf("conversation Ask: %v", err)
	}

	reqs := fakeRequests()
	if len(reqs) != 1 {
		t.Fatalf("requests = %d, want 1", len(reqs))
	}
	if got, want := rolesOf(reqs[0].prompt), []llmhub.Role{llmhub.RoleSystem, llmhub.RoleUser}; !reflect.DeepEqual(got, want) {
		t.Errorf("request roles = %v, want %v", got, want)
	}
	if got := textOf(reqs[0].prompt[0]); got != "conversation system" {
		t.Errorf("system message = %q, want the conversation's own", got)
	}
	if got := textOf(reqs[0].prompt[1]); got != "real question" {
		t.Errorf("user message = %q, want the turn's query", got)
	}
	// Conversation semantics are unchanged: Answer.Messages is just this
	// turn's own exchange.
	if len(ans.Messages) != 2 {
		t.Errorf("Answer.Messages = %d messages, want this turn's 2", len(ans.Messages))
	}
}

// TestWithPrefixStream: AskStream seeds the transcript exactly like Ask.
func TestWithPrefixStream(t *testing.T) {
	resetScript(t, streamTurn(fakeChunk{delta: "answer"}, fakeChunk{done: true}))
	prefix := []*llmhub.Message{
		llmhub.NewSystemMessage(llmhub.Text("stream system")),
		llmhub.NewUserMessage(llmhub.Text("earlier question")),
	}
	ag := newFakeAgent(t)

	events, err := ag.AskStream(context.Background(), "stream question", WithPrefix(prefix...))
	if err != nil {
		t.Fatalf("AskStream: %v", err)
	}
	evs := collectEvents(t, events)

	want := append(append([]*llmhub.Message(nil), prefix...), llmhub.NewUserMessage(llmhub.Text("stream question")))
	reqs := fakeRequests()
	if len(reqs) != 1 || !equalMessages(reqs[0].prompt, want) {
		t.Fatal("streaming request does not begin with the prefix followed by the query")
	}
	var ans *Answer
	for _, ev := range evs {
		if ev.Kind == EventDone {
			ans = ev.Answer
		}
	}
	if ans == nil {
		t.Fatalf("no Done event in %v", kindsOf(evs))
	}
	if len(ans.Messages) != len(want)+1 || !equalMessages(ans.Messages[:len(want)], want) {
		t.Error("streaming Answer.Messages is not the full transcript")
	}
}

// TestWithPrefixCompactorKeepsFirstMessage: the compactor's projection
// keeps the transcript's first message verbatim — with a prefix, that is
// the prefix's opening message.
func TestWithPrefixCompactorKeepsFirstMessage(t *testing.T) {
	const window = int64(20000) // threshold: window - 16000 = 4000
	resetScript(t,
		fakeTurn{
			calls: []*llmhub.ToolCallContent{llmhub.ToolCall("c1", "echo", `{}`)},
			usage: llmhub.UsageMetadata{PromptTokens: 4000},
		},
		textTurn("done"),
	)
	prefix := []*llmhub.Message{
		llmhub.NewSystemMessage(llmhub.Text("prefix system")),
		llmhub.NewUserMessage(llmhub.Text("earlier history")),
	}
	for i := 0; i < 10; i++ {
		prefix = append(prefix,
			llmhub.NewAssistantMessage(llmhub.Text(strings.Repeat("h", 4000))),
			llmhub.NewUserMessage(llmhub.Text(strings.Repeat("u", 4000))))
	}
	ag := newFakeAgent(t,
		WithTools(echoTool()),
		WithCompactor(NewTokenCompactor(window, 3000, 1)))

	ans, err := ag.Ask(context.Background(), "continue", WithPrefix(prefix...))
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}

	reqs := fakeRequests()
	if len(reqs) != 2 {
		t.Fatalf("requests = %d, want 2", len(reqs))
	}
	// The trip actually shortened the second view, and its head is the
	// prefix's first message, untouched.
	if len(reqs[1].prompt) >= len(prefix)+3 {
		t.Errorf("second view = %d messages, want a compacted projection of the %d-message transcript",
			len(reqs[1].prompt), len(prefix)+3)
	}
	if got := textOf(reqs[1].prompt[0]); got != "prefix system" {
		t.Errorf("projected first message = %q, want the prefix's system message", got)
	}
	// Compaction is ephemeral: the answer still carries the full transcript.
	if len(ans.Messages) != len(prefix)+4 || !equalMessages(ans.Messages[:len(prefix)], prefix) {
		t.Error("Answer.Messages is not the full transcript with the prefix at its head")
	}
}
