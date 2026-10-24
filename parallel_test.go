package agentloop

import (
	"context"
	"reflect"
	"sort"
	"testing"

	"github.com/smhanov/llmhub"
)

// parallelScript builds three gated parallel-eligible tools (t1, t2, t3)
// with their gates and a shared start signal channel.
func parallelScript(log *callLog, onstart chan string) (t1, t2, t3 Tool, gates []chan struct{}) {
	gates = []chan struct{}{make(chan struct{}), make(chan struct{}), make(chan struct{})}
	t1 = gateTool(log, "t1", gates[0], onstart)
	t2 = gateTool(log, "t2", gates[1], onstart)
	t3 = gateTool(log, "t3", gates[2], onstart)
	return t1, t2, t3, gates
}

func TestParallelToolsCompleteOrderVsCallOrder(t *testing.T) {
	log := &callLog{}
	onstart := make(chan string, 3)
	afterEvents := make(chan string, 3) // AfterToolCall results, in fire order
	t1, t2, t3, gates := parallelScript(log, onstart)
	resetScript(t,
		callTurn(
			llmhub.ToolCall("c1", "t1", `{}`),
			llmhub.ToolCall("c2", "t2", `{}`),
			llmhub.ToolCall("c3", "t3", `{}`),
		),
		textTurn("ok"),
	)
	ag := newFakeAgent(t, WithTools(t1, t2, t3), WithHooks(Hooks{
		AfterToolCall: func(ctx context.Context, ev *Event) error {
			afterEvents <- ev.Result
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

	// All three run concurrently before any gate is released (the order in
	// which they start is not deterministic — that is the point).
	var starts []string
	for range []int{0, 1, 2} {
		starts = append(starts, <-onstart)
	}
	sort.Strings(starts)
	if !reflect.DeepEqual(starts, []string{"t1", "t2", "t3"}) {
		t.Fatalf("starts = %v, want all three", starts)
	}

	// Release out of call order, waiting for each completion before the
	// next release so the completion order is deterministic.
	close(gates[2])
	if got := <-afterEvents; got != "done:t3" {
		t.Fatalf("first completion = %q, want done:t3", got)
	}
	close(gates[0])
	if got := <-afterEvents; got != "done:t1" {
		t.Fatalf("second completion = %q, want done:t1", got)
	}
	close(gates[1])
	if got := <-afterEvents; got != "done:t2" {
		t.Fatalf("third completion = %q, want done:t2", got)
	}
	if err := <-done; err != nil {
		t.Fatalf("Ask: %v", err)
	}
	var names []string
	for _, tr := range ans.Tools {
		names = append(names, tr.Name)
	}
	if !reflect.DeepEqual(names, []string{"t1", "t2", "t3"}) {
		t.Errorf("trace order = %v, want call order", names)
	}
	reqs := fakeRequests()
	for i, wantID := range []string{"c1", "c2", "c3"} {
		got := reqs[1].prompt[2+i]
		if got.Meta["tool_call_id"] != wantID || textOf(got) != "done:t"+wantID[1:] {
			t.Errorf("result message %d = (%s, %q), want call order", i, got.Meta["tool_call_id"], textOf(got))
		}
	}
}

func TestParallelToolsOneIsSequential(t *testing.T) {
	log := &callLog{}
	onstart := make(chan string, 3)
	t1, t2, t3, gates := parallelScript(log, onstart)
	resetScript(t,
		callTurn(
			llmhub.ToolCall("c1", "t1", `{}`),
			llmhub.ToolCall("c2", "t2", `{}`),
			llmhub.ToolCall("c3", "t3", `{}`),
		),
		textTurn("ok"),
	)
	ag := newFakeAgent(t, WithTools(t1, t2, t3), WithParallelTools(1))

	done := make(chan error, 1)
	go func() {
		_, err := ag.Ask(context.Background(), "One at a time.")
		done <- err
	}()

	// With the bound at 1, calls start strictly in call order and each one
	// must be released before the next starts.
	for i, want := range []string{"t1", "t2", "t3"} {
		if got := <-onstart; got != want {
			t.Fatalf("start %d = %q, want %q", i, got, want)
		}
		close(gates[i])
	}
	if err := <-done; err != nil {
		t.Fatalf("Ask: %v", err)
	}
	want := []string{
		"start:t1", "end:t1",
		"start:t2", "end:t2",
		"start:t3", "end:t3",
	}
	if got := log.snapshot(); !reflect.DeepEqual(got, want) {
		t.Errorf("log = %v, want strictly sequential %v", got, want)
	}
}

func TestParallelToolsOverlap(t *testing.T) {
	log := &callLog{}
	onstart := make(chan string, 2)
	t1, t2, _, gates := parallelScript(log, onstart)
	resetScript(t,
		callTurn(
			llmhub.ToolCall("c1", "t1", `{}`),
			llmhub.ToolCall("c2", "t2", `{}`),
		),
		textTurn("ok"),
	)
	ag := newFakeAgent(t, WithTools(t1, t2))

	done := make(chan error, 1)
	go func() {
		_, err := ag.Ask(context.Background(), "Overlap.")
		done <- err
	}()

	// t2 starts while t1 is still blocked: proof of concurrency (start
	// order between the two is not deterministic).
	starts := map[string]bool{<-onstart: true, <-onstart: true}
	if !starts["t1"] || !starts["t2"] {
		t.Fatalf("starts = %v, want both t1 and t2", starts)
	}
	gotLog := log.snapshot()
	sort.Strings(gotLog)
	if !reflect.DeepEqual(gotLog, []string{"start:t1", "start:t2"}) {
		t.Fatalf("log = %v, want overlapping starts", gotLog)
	}
	close(gates[0])
	close(gates[1])
	if err := <-done; err != nil {
		t.Fatalf("Ask: %v", err)
	}
}

func TestSequentialToolNeverOverlapsAnything(t *testing.T) {
	log := &callLog{}
	onstart := make(chan string, 3)
	gates := map[string]chan struct{}{
		"p1": make(chan struct{}),
		"s1": make(chan struct{}),
		"p2": make(chan struct{}),
	}
	p1 := gateTool(log, "p1", gates["p1"], onstart)
	p2 := gateTool(log, "p2", gates["p2"], onstart)
	s1 := gateTool(log, "s1", gates["s1"], onstart)
	s1.Sequential = true

	resetScript(t,
		callTurn(
			llmhub.ToolCall("c1", "p1", `{}`),
			llmhub.ToolCall("c2", "s1", `{}`),
			llmhub.ToolCall("c3", "p2", `{}`),
		),
		textTurn("ok"),
	)
	ag := newFakeAgent(t, WithTools(p1, s1, p2))

	done := make(chan error, 1)
	go func() {
		_, err := ag.Ask(context.Background(), "Ordered.")
		done <- err
	}()

	// p1 runs alone in its batch; s1 must not start until p1 ends.
	if got := <-onstart; got != "p1" {
		t.Fatalf("first start = %q", got)
	}
	if got := log.snapshot(); !reflect.DeepEqual(got, []string{"start:p1"}) {
		t.Fatalf("log before release = %v, want only p1 started", got)
	}
	close(gates["p1"])
	if got := <-onstart; got != "s1" {
		t.Fatalf("second start = %q, want s1", got)
	}
	if got := log.snapshot(); !reflect.DeepEqual(got, []string{"start:p1", "end:p1", "start:s1"}) {
		t.Fatalf("log = %v, want s1 strictly after p1", got)
	}
	close(gates["s1"])
	if got := <-onstart; got != "p2" {
		t.Fatalf("third start = %q, want p2", got)
	}
	if got := log.snapshot(); !reflect.DeepEqual(got, []string{"start:p1", "end:p1", "start:s1", "end:s1", "start:p2"}) {
		t.Fatalf("log = %v, want p2 strictly after s1", got)
	}
	close(gates["p2"])
	if err := <-done; err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if got, want := log.snapshot(), []string{
		"start:p1", "end:p1",
		"start:s1", "end:s1",
		"start:p2", "end:p2",
	}; !reflect.DeepEqual(got, want) {
		t.Errorf("log = %v, want %v", got, want)
	}
}
