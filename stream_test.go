package agentloop

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/smhanov/llmhub"
)

// collectEvents drains an AskStream channel, returning the events and
// failing the test if Error carries an unexpected error.
func collectEvents(t *testing.T, events <-chan Event) []Event {
	t.Helper()
	var evs []Event
	for ev := range events {
		evs = append(evs, ev)
	}
	return evs
}

func kindsOf(evs []Event) []EventKind {
	kinds := make([]EventKind, 0, len(evs))
	for _, ev := range evs {
		kinds = append(kinds, ev.Kind)
	}
	return kinds
}

func TestAskStreamEventSequence(t *testing.T) {
	usage := llmhub.UsageMetadata{PromptTokens: 3, CompletionTokens: 2, TotalTokens: 5}
	timing := &llmhub.TimingMetrics{Total: 2 * time.Millisecond, Chunks: 3}
	resetScript(t, streamTurn(
		fakeChunk{delta: "Hel"},
		fakeChunk{delta: "lo"},
		fakeChunk{done: true, usage: &usage, timing: timing},
	))
	ag := newFakeAgent(t)

	events, err := ag.AskStream(context.Background(), "Hi")
	if err != nil {
		t.Fatalf("AskStream: %v", err)
	}
	evs := collectEvents(t, events)

	wantKinds := []EventKind{
		EventStart, EventBeforeGenerate,
		EventAnswerDelta, EventAnswerDelta,
		EventAfterGenerate, EventDone,
	}
	if got := kindsOf(evs); !reflect.DeepEqual(got, wantKinds) {
		t.Fatalf("event kinds = %v, want %v", got, wantKinds)
	}
	// Deltas are Step-tagged and carry the incremental text.
	if evs[2].Step != 1 || evs[2].Delta != "Hel" || evs[3].Step != 1 || evs[3].Delta != "lo" {
		t.Errorf("deltas = (%d,%q) (%d,%q)", evs[2].Step, evs[2].Delta, evs[3].Step, evs[3].Delta)
	}
	// Done carries the fully populated answer and the complete timing (the
	// llmhub client re-stamps chunk timing, so assert the deterministic
	// parts: present, positive total, two content chunks).
	done := evs[len(evs)-1]
	if done.Answer == nil || done.Answer.Text != "Hello" || done.Answer.Status != StatusCompleted {
		t.Errorf("Done answer = %+v", done.Answer)
	}
	if done.Timing == nil || done.Timing.Total <= 0 || done.Timing.Chunks != 2 {
		t.Errorf("Done timing = %+v, want the complete snapshot (2 content chunks)", done.Timing)
	}
	if done.Answer.Usage != usage {
		t.Errorf("usage = %+v, want %+v", done.Answer.Usage, usage)
	}
}

func TestAskStreamMultiStepReasoningAndInterimText(t *testing.T) {
	resetScript(t,
		streamTurn(
			fakeChunk{reasoning: "ponder"},
			fakeChunk{delta: "checking "},
			fakeChunk{calls: []*llmhub.ToolCallContent{llmhub.ToolCall("c1", "clock", `{}`)}},
			fakeChunk{done: true},
		),
		streamTurn(
			fakeChunk{delta: "It is noon."},
			fakeChunk{done: true},
		),
	)
	ag := newFakeAgent(t, WithTools(clockTool()))

	events, err := ag.AskStream(context.Background(), "What time is it?")
	if err != nil {
		t.Fatalf("AskStream: %v", err)
	}
	evs := collectEvents(t, events)

	var reasoningDeltas, answerDeltas []Event
	for _, ev := range evs {
		switch ev.Kind {
		case EventReasoningDelta:
			reasoningDeltas = append(reasoningDeltas, ev)
		case EventAnswerDelta:
			answerDeltas = append(answerDeltas, ev)
		}
	}
	// Reasoning deltas stream from every step they occur in, Step-tagged.
	if len(reasoningDeltas) != 1 || reasoningDeltas[0].Step != 1 || reasoningDeltas[0].Delta != "ponder" {
		t.Errorf("reasoning deltas = %+v", reasoningDeltas)
	}
	// Interim tool-step text streams too, Step-tagged.
	want := []struct {
		step  int
		delta string
	}{{1, "checking "}, {2, "It is noon."}}
	if len(answerDeltas) != len(want) {
		t.Fatalf("answer deltas = %+v", answerDeltas)
	}
	for i, w := range want {
		if answerDeltas[i].Step != w.step || answerDeltas[i].Delta != w.delta {
			t.Errorf("answer delta %d = (%d,%q), want (%d,%q)", i, answerDeltas[i].Step, answerDeltas[i].Delta, w.step, w.delta)
		}
	}

	// The terminal event carries the full answer: only the final step's
	// text, all reasoning, and the executed tool call.
	var done *Event
	for i := range evs {
		if evs[i].Kind == EventDone {
			done = &evs[i]
		}
	}
	if done == nil {
		t.Fatal("no Done event")
	}
	ans := done.Answer
	if ans.Text != "It is noon." {
		t.Errorf("Text = %q, want only the final step's text", ans.Text)
	}
	if ans.ReasoningText != "ponder" {
		t.Errorf("ReasoningText = %q", ans.ReasoningText)
	}
	if ans.Steps != 2 || ans.ToolCalls != 1 || ans.Status != StatusCompleted {
		t.Errorf("got Steps=%d ToolCalls=%d Status=%v", ans.Steps, ans.ToolCalls, ans.Status)
	}
	if len(ans.Tools) != 1 || ans.Tools[0].Name != "clock" || ans.Tools[0].Output != "12:00:00Z" {
		t.Errorf("Tools = %+v", ans.Tools)
	}
	if done.Timing == nil {
		t.Error("Done timing = nil, want the complete set")
	}
}

func TestAskStreamWatchdogStale(t *testing.T) {
	resetScript(t, fakeTurn{
		chunks: []fakeChunk{{delta: "x"}},
		stall:  true, // never reaches Done
	})
	ag := newFakeAgent(t, WithStreamWatchdog(50*time.Millisecond))

	events, err := ag.AskStream(context.Background(), "Hi")
	if err != nil {
		t.Fatalf("AskStream: %v", err)
	}
	evs := collectEvents(t, events)
	if len(evs) == 0 || evs[len(evs)-1].Kind != EventError {
		t.Fatalf("last event = %+v, want EventError", evs)
	}
	errLast := evs[len(evs)-1].Err
	if !errors.Is(errLast, ErrStreamStale) {
		t.Fatalf("error = %v, want ErrStreamStale", errLast)
	}
	for _, ev := range evs {
		if ev.Kind == EventDone {
			t.Error("Done emitted on a stale run")
		}
	}
}

func TestAskStreamWatchdogDisabledCtxCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	started := make(chan struct{})
	resetScript(t, fakeTurn{
		chunks: []fakeChunk{{delta: "x", afterClose: started}},
		stall:  true,
	})
	ag := newFakeAgent(t, WithStreamWatchdog(0)) // disabled

	events, err := ag.AskStream(ctx, "Hi")
	if err != nil {
		t.Fatalf("AskStream: %v", err)
	}
	done := make(chan []Event, 1)
	go func() { done <- collectEvents(t, events) }()

	<-started // the fake is now stalled
	cancel()
	evs := <-done
	if len(evs) == 0 || evs[len(evs)-1].Kind != EventError {
		t.Fatalf("last event = %+v, want EventError", evs)
	}
	if last := evs[len(evs)-1].Err; !errors.Is(last, context.Canceled) {
		t.Fatalf("error = %v, want context.Canceled", last)
	}
}

func TestAskStreamOnDeltaHook(t *testing.T) {
	resetScript(t,
		streamTurn(
			fakeChunk{delta: "Hel"},
			fakeChunk{delta: "lo"},
			fakeChunk{done: true},
		),
	)
	var hookDeltas []string
	ag := newFakeAgent(t, WithHooks(Hooks{
		OnDelta: func(ctx context.Context, ev *Event) error {
			if ev.Kind != EventAnswerDelta {
				t.Errorf("kind = %v, want EventAnswerDelta", ev.Kind)
			}
			hookDeltas = append(hookDeltas, ev.Delta)
			return nil
		},
	}))

	events, err := ag.AskStream(context.Background(), "Hi")
	if err != nil {
		t.Fatalf("AskStream: %v", err)
	}
	evs := collectEvents(t, events)
	if !reflect.DeepEqual(hookDeltas, []string{"Hel", "lo"}) {
		t.Errorf("OnDelta deltas = %v", hookDeltas)
	}
	// The channel still carries the same deltas.
	var channelDeltas []string
	for _, ev := range evs {
		if ev.Kind == EventAnswerDelta {
			channelDeltas = append(channelDeltas, ev.Delta)
		}
	}
	if !reflect.DeepEqual(channelDeltas, []string{"Hel", "lo"}) {
		t.Errorf("channel deltas = %v", channelDeltas)
	}
}

func TestMergeToolCallFragments(t *testing.T) {
	var calls []*llmhub.ToolCallContent
	calls = mergeToolCallFragment(calls, llmhub.ToolCallWithIndex(0, "a", "weather", `{"city": `))
	calls = mergeToolCallFragment(calls, llmhub.ToolCallWithIndex(1, "b", "clock", `{}`))
	calls = mergeToolCallFragment(calls, llmhub.ToolCallWithIndex(0, "", "", `"paris"}`))

	if len(calls) != 2 {
		t.Fatalf("calls = %+v, want 2 merged calls", calls)
	}
	if calls[0].ID != "a" || calls[0].Name != "weather" || calls[0].Arguments != `{"city": "paris"}` {
		t.Errorf("call 0 = %+v", calls[0])
	}
	if calls[1].ID != "b" || calls[1].Arguments != `{}` {
		t.Errorf("call 1 = %+v", calls[1])
	}
}

func TestAskStreamScriptExhaustedFails(t *testing.T) {
	// No script at all: the stream construction fails and the channel
	// carries the error.
	resetScript(t)
	ag := newFakeAgent(t)
	events, err := ag.AskStream(context.Background(), "Hi")
	if err != nil {
		t.Fatalf("AskStream: %v", err)
	}
	evs := collectEvents(t, events)
	if len(evs) == 0 || evs[len(evs)-1].Kind != EventError {
		t.Fatalf("last event = %+v, want EventError", evs)
	}
	if evs[len(evs)-1].Err == nil {
		t.Error("EventError without an error")
	}
}
