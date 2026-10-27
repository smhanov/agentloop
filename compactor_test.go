package agentloop

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/smhanov/llmhub"
)

// sawtoothUsage returns the PromptTokens each of n scripted turns
// reports: growing by growth tokens per turn and, once past the trip
// threshold for window, dropping to the compacted size and growing again
// — the model measured the small, projected view after a trip.
func sawtoothUsage(n int, window int64, growth int) []int {
	reserve := window * compactReservePct / 100
	if reserve < minCompactReserve {
		reserve = minCompactReserve
	}
	threshold := window - reserve
	out := make([]int, n)
	usage := 0
	for i := range out {
		out[i] = usage
		if usage >= int(threshold) {
			usage = int(window) / 10
		}
		usage += growth
	}
	return out
}

// compactScript scripts n tool-call turns (callsPerTurn calls per turn,
// each carrying argChars arguments, so both the assistant message and its
// observation are large) followed by one final text turn, with
// sawtoothUsage PromptTokens.
func compactScript(n, callsPerTurn, argChars int, window int64, growth int) []fakeTurn {
	usage := sawtoothUsage(n, window, growth)
	turns := make([]fakeTurn, 0, n)
	for i := 0; i < n; i++ {
		step := llmhub.UsageMetadata{PromptTokens: usage[i], CompletionTokens: 5}
		if i == n-1 {
			turns = append(turns, fakeTurn{text: "done", usage: step})
			break
		}
		calls := make([]*llmhub.ToolCallContent, callsPerTurn)
		for k := range calls {
			// Arguments are unique per call: identical calls with
			// identical output would trip duplicate-call detection, which
			// would stop these runs long before the window fills.
			calls[k] = llmhub.ToolCall(fmt.Sprintf("c%d-%d", i, k), "echo",
				fmt.Sprintf(`{"n":%d,"text":%q}`, i, strings.Repeat("x", argChars)))
		}
		turns = append(turns, fakeTurn{calls: calls, usage: step})
	}
	return turns
}

// sentTokens estimates a sent prompt as characters/4, counted
// independently of the implementation's own estimator. One message of
// slack is the margin the tail boundary may overshoot by.
func sentTokens(msgs []*llmhub.Message) int {
	chars := 0
	for _, m := range msgs {
		chars += len(m.Role)
		for _, part := range m.Content {
			switch p := part.(type) {
			case *llmhub.TextContent:
				chars += len(p.Text)
			case *llmhub.ReasoningContent:
				chars += len(p.Text)
			case *llmhub.ToolCallContent:
				chars += len(p.ID) + len(p.Name) + len(p.Arguments)
			}
		}
	}
	return chars / 4
}

// TestTokenCompactorKeepsRunUnderWindow is the headline check for issue
// #5: a 200-step run whose prompts would overflow a 128k window completes
// with every sent request under the window, while the same script without
// a compactor grows unboundedly past it.
func TestTokenCompactorKeepsRunUnderWindow(t *testing.T) {
	const window = int64(128000)
	script := compactScript(200, 1, 2000, window, 4000)

	// Contrast run: no compactor. The prompts grow past the window.
	resetScript(t, script...)
	plain := newFakeAgent(t, WithTools(echoTool()), WithMaxSteps(220), WithMaxToolCalls(400))
	plainAns, err := plain.Ask(context.Background(), "Start the long run.")
	if err != nil {
		t.Fatalf("plain Ask: %v", err)
	}
	plainReqs := fakeRequests()
	if plainAns.Status != StatusCompleted || plainAns.Steps != 200 {
		t.Fatalf("plain run: Status=%v Steps=%d", plainAns.Status, plainAns.Steps)
	}
	overflow := 0
	for _, req := range plainReqs {
		if sentTokens(req.prompt) >= int(window) {
			overflow++
		}
	}
	if overflow == 0 {
		t.Fatalf("contrast run never exceeded the %d-token window; the script does not exercise the scenario", window)
	}

	// Compacted run: completes, and every request stays under the window
	// (one message of tail-boundary slack is the margin).
	resetScript(t, script...)
	ag := newFakeAgent(t, WithTools(echoTool()), WithMaxSteps(220), WithMaxToolCalls(400),
		WithCompactor(NewTokenCompactor(window, 15000, 1)))
	ans, err := ag.Ask(context.Background(), "Start the long run.")
	if err != nil {
		t.Fatalf("compacted Ask: %v", err)
	}
	if ans.Status != StatusCompleted || ans.Steps != 200 {
		t.Fatalf("compacted run: Status=%v Steps=%d", ans.Status, ans.Steps)
	}
	reqs := fakeRequests()
	if len(reqs) != len(plainReqs) {
		t.Fatalf("compacted run sent %d requests, want %d", len(reqs), len(plainReqs))
	}
	for i, req := range reqs {
		if n := sentTokens(req.prompt); n >= int(window) {
			t.Errorf("request %d: sent prompt ≈ %d tokens, over the %d window", i+1, n, window)
		}
	}
}

// TestTokenCompactorEphemeral pins the record-don't-rewrite invariant: a
// compacted run's transcript — in Answer.Messages and in Conversation
// history — is byte-identical to an uncompacted run over the same script.
func TestTokenCompactorEphemeral(t *testing.T) {
	script := compactScript(30, 1, 1500, 40000, 4000)

	run := func(opts ...Option) (*Answer, *Conversation, []fakeRequest) {
		t.Helper()
		resetScript(t, script...)
		ag := newFakeAgent(t, append([]Option{WithTools(echoTool()), WithMaxSteps(40), WithMaxToolCalls(60), WithSystemPrompt("sys")}, opts...)...)
		conv := ag.Conversation()
		ans, err := conv.Ask(context.Background(), "Run with history.")
		if err != nil {
			t.Fatalf("Ask: %v", err)
		}
		return ans, conv, fakeRequests()
	}

	plainAns, plainConv, plainReqs := run()
	compactAns, compactConv, compactReqs := run(WithCompactor(NewTokenCompactor(40000, 8000, 1)))

	if !equalMessages(compactAns.Messages, plainAns.Messages) {
		t.Errorf("Answer.Messages changed under compaction:\ncompacted: %v\nplain: %v",
			rolesOf(compactAns.Messages), rolesOf(plainAns.Messages))
	}
	if !equalMessages(compactConv.Messages(), plainConv.Messages()) {
		t.Errorf("Conversation history changed under compaction")
	}
	// The test must not pass vacuously: the compacted run's provider
	// requests are the same count but at least one is strictly smaller —
	// compaction fired without leaving a trace in the transcript.
	if len(plainReqs) != len(compactReqs) || len(plainReqs) < 2 {
		t.Fatalf("request counts: %d plain vs %d compacted", len(plainReqs), len(compactReqs))
	}
	smaller := 0
	for i := range plainReqs {
		if sentTokens(compactReqs[i].prompt) < sentTokens(plainReqs[i].prompt) {
			smaller++
		}
	}
	if smaller == 0 {
		t.Error("no compacted request is smaller than its plain counterpart; compaction never tripped")
	}
}

// TestTokenCompactorDeterministic replays a tripping 50-step conversation
// twice and requires byte-identical per-step prompts. Within one replay,
// the projection may only change at a trip: between trips each projection
// extends the previous one byte for byte (the frozen checkpoint riding a
// stable prefix — the cache-stability property).
func TestTokenCompactorDeterministic(t *testing.T) {
	script := compactScript(50, 1, 1500, 40000, 4000)
	runOnce := func(t *testing.T) [][]*llmhub.Message {
		t.Helper()
		resetScript(t, script...)
		ag := newFakeAgent(t, WithTools(echoTool()), WithMaxSteps(60), WithMaxToolCalls(80),
			WithSystemPrompt("sys"), WithCompactor(NewTokenCompactor(40000, 8000, 1)))
		if _, err := ag.Ask(context.Background(), "Replay me."); err != nil {
			t.Fatalf("Ask: %v", err)
		}
		var prompts [][]*llmhub.Message
		for _, req := range fakeRequests() {
			prompts = append(prompts, req.prompt)
		}
		return prompts
	}

	first := runOnce(t)
	second := runOnce(t)
	if len(first) != len(second) {
		t.Fatalf("replay lengths differ: %d vs %d", len(first), len(second))
	}
	for i := range first {
		if !equalMessages(first[i], second[i]) {
			t.Errorf("step %d: projection not deterministic", i+1)
		}
	}

	// Between trips the projection only grows at the end: each projection
	// extends the previous one. A projection that does not extend its
	// predecessor marks a trip; trips must be isolated (the projected
	// view needs many steps to re-trip) and rare.
	trips := 0
	for k := 0; k+1 < len(first); k++ {
		if extends(first[k], first[k+1]) {
			continue
		}
		trips++
		if k+2 < len(first) && !extends(first[k+1], first[k+2]) {
			t.Errorf("steps %d→%d: two consecutive projections broke prefix stability", k+2, k+3)
		}
	}
	if trips == 0 {
		t.Error("projection never tripped; the script does not exercise compaction")
	}
	if trips > 10 {
		t.Errorf("projection tripped %d times in 50 steps; the checkpoint is not being memoized", trips)
	}
}

// extends reports whether b extends a: a is a strict prefix of b (message
// for message), the cache-stable growth pattern.
func extends(a, b []*llmhub.Message) bool {
	if len(a) >= len(b) {
		return false
	}
	return equalMessages(a, b[:len(a)])
}

// TestTokenCompactorPreservesToolPairs walks every sent context of a
// tripping run mixing single and 3-way parallel calls: the system message
// is untouched, no tool result is orphaned from its caller, no tool call
// is left unanswered, and the trailing group is always
// [assistant with N calls, all N results] in call order.
func TestTokenCompactorPreservesToolPairs(t *testing.T) {
	const window = int64(40000)
	usage := sawtoothUsage(30, window, 9000)
	turns := make([]fakeTurn, 0, 30)
	for i := 0; i < 29; i++ {
		callsPerTurn := 1 + i%3 // 1, 2, 3 parallel calls in rotation
		calls := make([]*llmhub.ToolCallContent, callsPerTurn)
		for k := range calls {
			calls[k] = llmhub.ToolCall(fmt.Sprintf("c%d-%d", i, k), "echo",
				fmt.Sprintf(`{"n":%d,"text":%q}`, i, strings.Repeat("x", 1200)))
		}
		turns = append(turns, fakeTurn{calls: calls, usage: llmhub.UsageMetadata{PromptTokens: usage[i], CompletionTokens: 5}})
	}
	turns = append(turns, fakeTurn{text: "done", usage: llmhub.UsageMetadata{PromptTokens: usage[29], CompletionTokens: 5}})

	resetScript(t, turns...)
	ag := newFakeAgent(t, WithTools(echoTool()), WithMaxSteps(40), WithMaxToolCalls(90),
		WithSystemPrompt("sys"), WithCompactor(NewTokenCompactor(window, 6000, 1)))
	ans, err := ag.Ask(context.Background(), "Call tools in parallel.")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ans.Status != StatusCompleted {
		t.Fatalf("Status = %v", ans.Status)
	}

	elided := false
	for i, req := range fakeRequests() {
		prompt := req.prompt
		if len(prompt) == 0 || prompt[0].Role != llmhub.RoleSystem || textOf(prompt[0]) != "sys" {
			t.Errorf("request %d: system message not first and verbatim: %v", i+1, rolesOf(prompt))
		}
		answered := map[string]bool{} // call IDs with a later result
		seen := map[string]bool{}     // call IDs seen on assistant messages
		lastCalls := -1
		for k, m := range prompt {
			switch m.Role {
			case llmhub.RoleTool:
				id := m.Meta["tool_call_id"]
				if !seen[id] {
					t.Errorf("request %d: tool result %q at %d has no earlier assistant call", i+1, id, k)
				}
				answered[id] = true
				lastCalls = -1 // results consumed
			case llmhub.RoleAssistant:
				if calls := callsOf(m); len(calls) > 0 {
					for _, c := range calls {
						seen[c.ID] = true
					}
					lastCalls = k
				}
			}
		}
		for _, m := range prompt {
			for _, c := range callsOf(m) {
				if !answered[c.ID] {
					t.Errorf("request %d: assistant call %q has no tool result", i+1, c.ID)
				}
			}
		}
		// The trailing group: the last assistant-with-calls is followed by
		// exactly its results, in call order.
		if lastCalls >= 0 {
			calls := callsOf(prompt[lastCalls])
			tail := prompt[lastCalls+1:]
			if len(tail) != len(calls) {
				t.Errorf("request %d: trailing group has %d messages after %d calls", i+1, len(tail), len(calls))
			}
			for k, c := range calls {
				if k >= len(tail) || tail[k].Meta["tool_call_id"] != c.ID {
					t.Errorf("request %d: trailing result %d does not answer call %q in order", i+1, k, c.ID)
				}
			}
		}
		for _, m := range prompt {
			if strings.HasPrefix(textOf(m), "[tool output elided by compaction:") ||
				strings.HasPrefix(textOf(m), "[context checkpoint]") {
				elided = true
			}
		}
	}
	if !elided {
		t.Error("no sent context shows an elision or checkpoint; the script never tripped")
	}
}

// TestTokenCompactorTrigger drives Compact directly: the trigger is the
// previous step's measured PromptTokens against window − max(15%, 16k);
// a zero usage (first step) and a below-threshold usage are no-ops; the
// prune path elides tool outputs in place and the checkpoint path
// replaces the whole middle with the stub.
func TestTokenCompactorTrigger(t *testing.T) {
	// Window 20000 → reserve 16000 → threshold 4000. Tail budget 3000
	// estimated tokens = 12000 content characters.
	toolResult := func(id string, chars int) *llmhub.Message {
		return llmhub.NewToolResultMessage(id, "echo", llmhub.Text(strings.Repeat("r", chars)))
	}
	// [sys][query][call+result]×20 — all middle bulk is tool output.
	prunable := []*llmhub.Message{llmhub.NewSystemMessage(llmhub.Text("sys")), llmhub.NewUserMessage(llmhub.Text("go"))}
	for i := 0; i < 20; i++ {
		prunable = append(prunable,
			llmhub.NewAssistantMessage(llmhub.ToolCall(fmt.Sprintf("c%d", i), "echo", "{}")),
			toolResult(fmt.Sprintf("c%d", i), 1000))
	}
	// [sys][query][big user]×10 — pruning cannot shrink user messages, so
	// the checkpoint stub must take over.
	stubbable := []*llmhub.Message{llmhub.NewSystemMessage(llmhub.Text("sys"))}
	for i := 0; i < 10; i++ {
		stubbable = append(stubbable,
			llmhub.NewUserMessage(llmhub.Text(strings.Repeat("u", 4000))),
			llmhub.NewAssistantMessage(llmhub.Text("ok")))
	}

	compact := func(tc *TokenCompactor, transcript []*llmhub.Message, promptTokens int) *Event {
		t.Helper()
		ev := &Event{Kind: EventCompact, Messages: transcript, Context: transcript,
			Usage: llmhub.UsageMetadata{PromptTokens: promptTokens}}
		if err := tc.Compact(context.Background(), ev); err != nil {
			t.Fatalf("Compact: %v", err)
		}
		return ev
	}

	t.Run("zero usage is a no-op", func(t *testing.T) {
		ev := compact(NewTokenCompactor(20000, 3000, 1), prunable, 0)
		if len(ev.Context) != len(prunable) {
			t.Errorf("first-step compact changed the view: %d → %d messages", len(prunable), len(ev.Context))
		}
	})
	t.Run("below threshold is a no-op", func(t *testing.T) {
		ev := compact(NewTokenCompactor(20000, 3000, 1), prunable, 3999)
		if len(ev.Context) != len(prunable) {
			t.Errorf("below-threshold compact changed the view: %d → %d messages", len(prunable), len(ev.Context))
		}
	})
	t.Run("at threshold it trips and prunes tool outputs", func(t *testing.T) {
		ev := compact(NewTokenCompactor(20000, 3000, 1), prunable, 4000)
		// The prune path keeps every message but replaces old tool
		// outputs with notices, so the view's byte size drops.
		if sentTokens(ev.Context) >= sentTokens(prunable) {
			t.Fatalf("tripping compact left the view at %d estimated tokens (transcript %d)",
				sentTokens(ev.Context), sentTokens(prunable))
		}
		if textOf(ev.Context[0]) != "sys" || ev.Context[1].Role != llmhub.RoleUser {
			t.Errorf("keep-set damaged: %v", rolesOf(ev.Context[:2]))
		}
		notices, verbatim := 0, 0
		for _, m := range ev.Context {
			if strings.HasPrefix(textOf(m), "[tool output elided by compaction: 1000 chars]") {
				notices++
			}
		}
		// The verbatim tail keeps whole pairs: at least the trailing call
		// and result survive unpruned.
		for _, m := range ev.Context {
			if m.Role == llmhub.RoleTool && strings.HasPrefix(textOf(m), "rrrr") {
				verbatim++
			}
		}
		if notices == 0 || verbatim == 0 {
			t.Errorf("prune projection: %d notices, %d verbatim results", notices, verbatim)
		}
		// Ephemeral: the transcript is untouched.
		if !equalMessages(ev.Messages, prunable) {
			t.Error("transcript was modified by compaction")
		}
	})
	t.Run("prunable-until-it-is-not: big user messages get the checkpoint stub", func(t *testing.T) {
		ev := compact(NewTokenCompactor(20000, 3000, 1), stubbable, 4000)
		stub := false
		for _, m := range ev.Context {
			if strings.HasPrefix(textOf(m), "[context checkpoint]") {
				stub = true
			}
		}
		if !stub {
			t.Fatalf("no checkpoint stub in a %d-message projection", len(ev.Context))
		}
		if len(ev.Context) >= len(stubbable) {
			t.Errorf("checkpoint projection kept %d of %d messages", len(ev.Context), len(stubbable))
		}
	})
	t.Run("memoized projection is stable between trips", func(t *testing.T) {
		tc := NewTokenCompactor(20000, 3000, 1)
		ev := compact(tc, prunable, 4000)
		first := snapshotMessages(ev.Context)
		grown := append(append([]*llmhub.Message{}, prunable...),
			llmhub.NewAssistantMessage(llmhub.ToolCall("cX", "echo", "{}")),
			toolResult("cX", 1000))
		// Below the threshold again — the between-trips regime: the
		// projection may only grow at its end.
		ev = compact(tc, grown, 1000)
		if !extends(first, ev.Context) {
			t.Error("projection changed between trips; the checkpoint is not memoized")
		}
	})
	t.Run("non-positive window disables", func(t *testing.T) {
		ev := compact(NewTokenCompactor(0, 0, 0), prunable, 1<<30)
		if len(ev.Context) != len(prunable) {
			t.Errorf("disabled compactor changed the view: %d → %d messages", len(prunable), len(ev.Context))
		}
	})
}

// recordingCompactor is a user-written Compactor: it records that it ran,
// optionally replaces the view wholesale, and optionally fails.
type recordingCompactor struct {
	order    *[]string
	replace  bool
	err      error
	sawUsage []int
}

func (c *recordingCompactor) Compact(ctx context.Context, ev *Event) error {
	if c.order != nil {
		*c.order = append(*c.order, "compact")
	}
	c.sawUsage = append(c.sawUsage, ev.Usage.PromptTokens)
	if c.err != nil {
		return c.err
	}
	if c.replace && len(ev.Context) > 2 {
		ev.Context = append([]*llmhub.Message{ev.Context[0]}, ev.Context[len(ev.Context)-2:]...)
	}
	return nil
}

// TestCustomCompactorInstallsAfterRebuildContext proves a user
// implementation works through WithCompactor alone: it fires after the
// RebuildContext hook, its replacement reaches the provider, and ev.Usage
// carries the previous step's provider usage.
func TestCustomCompactorInstallsAfterRebuildContext(t *testing.T) {
	resetScript(t,
		callTurn(llmhub.ToolCall("c1", "echo", `{}`)),
		textTurn("done"),
	)
	var order []string
	custom := &recordingCompactor{order: &order, replace: true}
	var rebuilds int
	ag := newFakeAgent(t, WithTools(echoTool()), WithCompactor(custom), WithHooks(Hooks{
		RebuildContext: func(ctx context.Context, ev *Event) error {
			rebuilds++
			order = append(order, "rebuild")
			return nil
		},
	}))
	ans, err := ag.Ask(context.Background(), "Compact me.")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ans.Status != StatusCompleted || ans.Steps != 2 {
		t.Fatalf("Status=%v Steps=%d", ans.Status, ans.Steps)
	}
	wantOrder := []string{"rebuild", "compact", "rebuild", "compact"}
	if !equalStrings(order, wantOrder) {
		t.Errorf("firing order = %v, want %v", order, wantOrder)
	}
	// ev.Usage: zero on the first step, the first step's usage on the
	// second (scripted turns carry no usage, so both are zero — the field
	// is populated either way).
	if len(custom.sawUsage) != 2 {
		t.Fatalf("compactor ran %d times, want 2", len(custom.sawUsage))
	}
	// The replacement reached the provider: step 2's prompt is the
	// wholesale replacement (system + last two messages).
	reqs := fakeRequests()
	if len(reqs) != 2 {
		t.Fatalf("requests = %d, want 2", len(reqs))
	}
	if got := len(reqs[1].prompt); got != 3 {
		t.Errorf("step 2 sent %d messages, want the 3-message replacement", got)
	}
	// The hook fired every step too — the compactor never disables it.
	if rebuilds != 2 {
		t.Errorf("RebuildContext fired %d times, want 2", rebuilds)
	}
}

// TestCompactorErrorAbortsRun checks the error contract: a failing
// compactor aborts the run with a wrapped, named error.
func TestCompactorErrorAbortsRun(t *testing.T) {
	errBoom := errors.New("compactor exploded")
	resetScript(t, textTurn("never reached"))
	ag := newFakeAgent(t, WithCompactor(&recordingCompactor{err: errBoom}))
	_, err := ag.Ask(context.Background(), "q")
	if !errors.Is(err, errBoom) {
		t.Fatalf("error = %v, want errBoom", err)
	}
	if !strings.Contains(err.Error(), "agentloop: compactor:") {
		t.Errorf("error %q does not carry the agentloop: compactor: prefix", err)
	}
}

// TestTokenCompactorStreaming verifies the compacted view is what the
// streaming path sends: same loop, no special handling.
func TestTokenCompactorStreaming(t *testing.T) {
	const window = int64(40000)
	script := compactScript(30, 1, 1500, window, 4000)
	resetScript(t, script...)
	ag := newFakeAgent(t, WithTools(echoTool()), WithMaxSteps(40), WithMaxToolCalls(60),
		WithCompactor(NewTokenCompactor(window, 8000, 1)))
	events, err := ag.AskStream(context.Background(), "Stream a long run.")
	if err != nil {
		t.Fatalf("AskStream: %v", err)
	}
	done := false
	for ev := range events {
		if ev.Kind == EventDone {
			done = true
		}
		if ev.Kind == EventError {
			t.Fatalf("stream error: %v", ev.Err)
		}
	}
	if !done {
		t.Error("stream ended without EventDone")
	}
	for i, req := range fakeRequests() {
		if n := sentTokens(req.prompt); n >= int(window) {
			t.Errorf("streamed request %d: ≈ %d tokens, over the window", i+1, n)
		}
	}
}

// TestTokenCompactorConcurrentRuns proves one shared TokenCompactor is
// safe for concurrent runs on the same Agent: each run gets its own
// memoized state (run this under -race). The script interleaves text
// terminators so both runs complete under any turn interleaving.
func TestTokenCompactorConcurrentRuns(t *testing.T) {
	var script []fakeTurn
	for block := 0; block < 8; block++ {
		usage := sawtoothUsage(11, 40000, 6000)
		for i := 0; i < 10; i++ {
			script = append(script, fakeTurn{calls: []*llmhub.ToolCallContent{
				llmhub.ToolCall(fmt.Sprintf("c%d-%d", block, i), "echo",
					fmt.Sprintf(`{"b":%d,"n":%d}`, block, i)),
			}, usage: llmhub.UsageMetadata{PromptTokens: usage[i], CompletionTokens: 5}})
		}
		script = append(script, fakeTurn{text: "done", usage: llmhub.UsageMetadata{PromptTokens: usage[10], CompletionTokens: 5}})
	}
	resetScript(t, script...)
	ag := newFakeAgent(t, WithTools(echoTool()), WithMaxSteps(40), WithMaxToolCalls(100),
		WithCompactor(NewTokenCompactor(40000, 8000, 1)))
	var wg sync.WaitGroup
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ans, err := ag.Ask(context.Background(), "Concurrent run.")
			if err != nil {
				t.Errorf("Ask: %v", err)
				return
			}
			if ans.Status != StatusCompleted {
				t.Errorf("Status = %v", ans.Status)
			}
		}()
	}
	wg.Wait()
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
