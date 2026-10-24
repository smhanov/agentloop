package agentloop

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/smhanov/llmhub"
)

// fakeTurn is one scripted provider response. The text/reasoning/calls
// fields serve both Generate (directly) and Stream (synthesized into
// chunks); an explicit chunks script overrides synthesis in Stream.
type fakeTurn struct {
	text      string
	reasoning string
	calls     []*llmhub.ToolCallContent
	usage     llmhub.UsageMetadata
	err       error
	chunks    []fakeChunk // explicit streaming script (optional)
	stall     bool        // streaming: hang without Done after the chunks
}

// fakeChunk is one scripted StreamChunk.
type fakeChunk struct {
	delta      string
	reasoning  string
	calls      []*llmhub.ToolCallContent // fragments, merged by Index
	usage      *llmhub.UsageMetadata
	timing     *llmhub.TimingMetrics
	done       bool
	err        error
	afterClose chan struct{} // closed right after this chunk is delivered
}

// textTurn scripts a text-only response.
func textTurn(text string) fakeTurn { return fakeTurn{text: text} }

// callTurn scripts a response carrying tool calls.
func callTurn(calls ...*llmhub.ToolCallContent) fakeTurn { return fakeTurn{calls: calls} }

// callTextTurn scripts a response with both text and tool calls.
func callTextTurn(text string, calls ...*llmhub.ToolCallContent) fakeTurn {
	return fakeTurn{text: text, calls: calls}
}

// streamTurn scripts a response delivered as explicit stream chunks.
func streamTurn(chunks ...fakeChunk) fakeTurn { return fakeTurn{chunks: chunks} }

// fakeRequest records one provider call: the exact context and merged
// config the model saw.
type fakeRequest struct {
	prompt []*llmhub.Message
	config llmhub.Config
}

// fakeProvider replays fakeState.script and records every request.
type fakeProvider struct{}

func (fakeProvider) Name() string { return "fake" }

func (fakeProvider) Generate(ctx context.Context, prompt []*llmhub.Message, opts ...llmhub.Option) (*llmhub.Response, error) {
	turn, err := fakeState.next(ctx, prompt, opts...)
	if err != nil {
		return nil, err
	}
	if turn.err != nil {
		return nil, turn.err
	}
	resp := &llmhub.Response{ID: fmt.Sprintf("fake-gen-%d", fakeState.count()), Usage: turn.usage}
	if turn.reasoning != "" {
		resp.Content = append(resp.Content, llmhub.Reasoning(turn.reasoning))
	}
	if turn.text != "" {
		resp.Content = append(resp.Content, llmhub.Text(turn.text))
	}
	for _, call := range turn.calls {
		resp.Content = append(resp.Content, call)
	}
	return resp, nil
}

func (fakeProvider) Stream(ctx context.Context, prompt []*llmhub.Message, opts ...llmhub.Option) (<-chan llmhub.StreamChunk, error) {
	turn, err := fakeState.next(ctx, prompt, opts...)
	if err != nil {
		return nil, err
	}
	if turn.err != nil {
		return nil, turn.err
	}
	chunks := turn.chunks
	if chunks == nil {
		chunks = synthesizeChunks(turn)
	}
	ch := make(chan llmhub.StreamChunk)
	go func() {
		defer close(ch)
		for _, c := range chunks {
			if c.err != nil {
				ch <- llmhub.StreamChunk{Err: c.err}
				return
			}
			wire := llmhub.StreamChunk{Delta: c.delta, ReasoningDelta: c.reasoning, Usage: c.usage, Timing: c.timing, Done: c.done}
			for _, call := range c.calls {
				wire.ToolCalls = append(wire.ToolCalls, call)
			}
			select {
			case ch <- wire:
			case <-ctx.Done():
				return
			}
			if c.afterClose != nil {
				close(c.afterClose)
			}
		}
		if turn.stall {
			// Never reach Done: the watchdog (or ctx) must end this.
			<-ctx.Done()
		}
	}()
	return ch, nil
}

// synthesizeChunks builds a simple streaming script from a plain turn: one
// chunk per reasoning/text part, one fragment chunk per tool call, then a
// terminal Done chunk carrying the usage and a complete timing snapshot.
func synthesizeChunks(turn fakeTurn) []fakeChunk {
	var chunks []fakeChunk
	if turn.reasoning != "" {
		chunks = append(chunks, fakeChunk{reasoning: turn.reasoning})
	}
	if turn.text != "" {
		chunks = append(chunks, fakeChunk{delta: turn.text})
	}
	for i, call := range turn.calls {
		frag := *call
		frag.Index = i
		chunks = append(chunks, fakeChunk{calls: []*llmhub.ToolCallContent{&frag}})
	}
	timing := &llmhub.TimingMetrics{Chunks: len(chunks) + 1, Total: 2 * time.Millisecond}
	return append(chunks, fakeChunk{done: true, usage: &turn.usage, timing: timing})
}

// fakeHarness is the scripted-provider state shared by the test binary.
type fakeHarness struct {
	mu       sync.Mutex
	script   []fakeTurn
	requests []fakeRequest
	index    int
}

var fakeState fakeHarness

func init() {
	llmhub.MustRegisterProvider("fake", func(apiKey string, opts ...llmhub.Option) (llmhub.Provider, error) {
		return fakeProvider{}, nil
	})
}

// next pops the next scripted turn and records the request.
func (h *fakeHarness) next(ctx context.Context, prompt []*llmhub.Message, opts ...llmhub.Option) (fakeTurn, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return fakeTurn{}, err
	}
	h.requests = append(h.requests, fakeRequest{
		prompt: snapshotMessages(prompt),
		config: llmhub.NewConfig(opts...),
	})
	if h.index >= len(h.script) {
		return fakeTurn{}, fmt.Errorf("fake provider: script exhausted at request %d", h.index+1)
	}
	turn := h.script[h.index]
	h.index++
	return turn, nil
}

// count returns the number of requests served so far.
func (h *fakeHarness) count() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.index
}

// resetScript installs a fresh response script and clears captured requests.
func resetScript(t *testing.T, turns ...fakeTurn) {
	t.Helper()
	fakeState.mu.Lock()
	defer fakeState.mu.Unlock()
	fakeState.script = turns
	fakeState.requests = nil
	fakeState.index = 0
}

// fakeRequests returns the provider calls captured so far.
func fakeRequests() []fakeRequest {
	fakeState.mu.Lock()
	defer fakeState.mu.Unlock()
	return append([]fakeRequest(nil), fakeState.requests...)
}

// newFakeAgent returns an Agent over the scripted fake provider.
func newFakeAgent(t *testing.T, opts ...Option) *Agent {
	t.Helper()
	client, err := llmhub.New("fake", "test-key")
	if err != nil {
		t.Fatalf("llmhub.New(fake): %v", err)
	}
	return MustNew(client, opts...)
}

// drainEvents collects every event until the channel closes and returns
// them, failing the test if it stays open beyond the deadline.
func drainEvents(t *testing.T, events <-chan Event) []Event {
	t.Helper()
	var collected []Event
	for ev := range events {
		collected = append(collected, ev)
	}
	return collected
}

// snapshotMessages copies a message list so later transcript appends cannot
// mutate captured requests.
func snapshotMessages(msgs []*llmhub.Message) []*llmhub.Message {
	out := make([]*llmhub.Message, 0, len(msgs))
	for _, m := range msgs {
		if m == nil {
			continue
		}
		clone := &llmhub.Message{Role: m.Role, Content: append([]llmhub.ContentPart(nil), m.Content...), Meta: m.Meta}
		out = append(out, clone)
	}
	return out
}

// textOf joins the text parts of a message.
func textOf(m *llmhub.Message) string {
	var b strings.Builder
	for _, part := range m.Content {
		if tc, ok := part.(*llmhub.TextContent); ok {
			b.WriteString(tc.Text)
		}
	}
	return b.String()
}

// callsOf lists the tool-call parts of a message.
func callsOf(m *llmhub.Message) []*llmhub.ToolCallContent {
	var calls []*llmhub.ToolCallContent
	for _, part := range m.Content {
		if call, ok := part.(*llmhub.ToolCallContent); ok {
			calls = append(calls, call)
		}
	}
	return calls
}

// rolesOf lists the roles of a message list.
func rolesOf(msgs []*llmhub.Message) []llmhub.Role {
	roles := make([]llmhub.Role, 0, len(msgs))
	for _, m := range msgs {
		roles = append(roles, m.Role)
	}
	return roles
}

// equalMessages reports whether two message lists are deep-equal (roles,
// metadata, and content parts).
func equalMessages(a, b []*llmhub.Message) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].Role != b[i].Role || !reflect.DeepEqual(a[i].Meta, b[i].Meta) {
			return false
		}
		if len(a[i].Content) != len(b[i].Content) {
			return false
		}
		for j := range a[i].Content {
			if !reflect.DeepEqual(a[i].Content[j], b[i].Content[j]) {
				return false
			}
		}
	}
	return true
}

// echoTool echoes its raw arguments back as its output.
func echoTool() Tool {
	return Tool{
		Def: llmhub.NewTool("echo", "Echo the arguments back", map[string]any{
			"type":       "object",
			"properties": map[string]any{"text": map[string]any{"type": "string"}},
		}),
		Run: func(ctx context.Context, args json.RawMessage) (string, error) {
			return "echo:" + string(args), nil
		},
	}
}

// clockTool returns a deterministic clock tool (no real time).
func clockTool() Tool {
	return Tool{
		Def: llmhub.NewTool("clock", "Current time", map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		}),
		Run: func(ctx context.Context, args json.RawMessage) (string, error) {
			return "12:00:00Z", nil
		},
	}
}

// callLog records ordered events from tool handlers.
type callLog struct {
	mu     sync.Mutex
	events []string
}

func (l *callLog) add(event string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.events = append(l.events, event)
}

func (l *callLog) snapshot() []string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]string(nil), l.events...)
}

// gateTool returns a parallel-eligible tool whose handler logs start/end
// events and blocks until gate is closed. When onstart is non-nil it
// receives the tool name as soon as the handler starts, so tests can
// synchronize without sleeping.
func gateTool(log *callLog, name string, gate chan struct{}, onstart chan<- string) Tool {
	return Tool{
		Def: llmhub.NewTool(name, "Gated tool", map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		}),
		Run: func(ctx context.Context, args json.RawMessage) (string, error) {
			log.add("start:" + name)
			if onstart != nil {
				onstart <- name
			}
			<-gate
			log.add("end:" + name)
			return "done:" + name, nil
		},
	}
}
