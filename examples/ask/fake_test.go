package main

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/smhanov/llmhub"
)

// fakeTurn scripts one provider response. The text/reasoning/calls
// fields serve Generate directly and are synthesized into chunks for
// Stream.
type fakeTurn struct {
	text      string
	reasoning string
	calls     []*llmhub.ToolCallContent
	usage     *llmhub.UsageMetadata
	err       error
}

func textTurn(text string) fakeTurn { return fakeTurn{text: text} }

func callTurn(calls ...*llmhub.ToolCallContent) fakeTurn { return fakeTurn{calls: calls} }

// streamTurn scripts a response delivered as reasoning + text deltas.
func streamTurn(reasoning, text string) fakeTurn {
	return fakeTurn{reasoning: reasoning, text: text}
}

// usageTurn scripts a text response carrying provider-reported usage.
func usageTurn(text string, u llmhub.UsageMetadata) fakeTurn {
	return fakeTurn{text: text, usage: &u}
}

// fakeRequest records one provider call: the exact message list the
// model saw.
type fakeRequest struct {
	prompt []*llmhub.Message
}

// fakeProvider replays fakeState.script and records every request.
type fakeProvider struct{}

func (fakeProvider) Name() string { return "fake" }

func (fakeProvider) Generate(ctx context.Context, prompt []*llmhub.Message, opts ...llmhub.Option) (*llmhub.Response, error) {
	turn, err := fakeState.next(ctx, prompt)
	if err != nil {
		return nil, err
	}
	if turn.err != nil {
		return nil, turn.err
	}
	resp := &llmhub.Response{}
	if turn.reasoning != "" {
		resp.Content = append(resp.Content, llmhub.Reasoning(turn.reasoning))
	}
	if turn.text != "" {
		resp.Content = append(resp.Content, llmhub.Text(turn.text))
	}
	for _, call := range turn.calls {
		resp.Content = append(resp.Content, call)
	}
	if turn.usage != nil {
		resp.Usage = *turn.usage
	}
	return resp, nil
}

func (fakeProvider) Stream(ctx context.Context, prompt []*llmhub.Message, opts ...llmhub.Option) (<-chan llmhub.StreamChunk, error) {
	turn, err := fakeState.next(ctx, prompt)
	if err != nil {
		return nil, err
	}
	if turn.err != nil {
		return nil, turn.err
	}
	ch := make(chan llmhub.StreamChunk, 4)
	go func() {
		defer close(ch)
		if turn.reasoning != "" {
			ch <- llmhub.StreamChunk{ReasoningDelta: turn.reasoning}
		}
		if turn.text != "" {
			ch <- llmhub.StreamChunk{Delta: turn.text}
		}
		for _, call := range turn.calls {
			ch <- llmhub.StreamChunk{ToolCalls: []*llmhub.ToolCallContent{call}}
		}
		ch <- llmhub.StreamChunk{
			Done:   true,
			Usage:  turn.usage,
			Timing: &llmhub.TimingMetrics{FirstToken: time.Millisecond, Total: 2 * time.Millisecond, Chunks: 3},
		}
	}()
	return ch, nil
}

// fakeState is the scripted-provider state shared by the test binary.
var fakeState fakeHarness

type fakeHarness struct {
	mu       sync.Mutex
	script   []fakeTurn
	requests []fakeRequest
}

func init() {
	llmhub.MustRegisterProvider("fake", func(string, ...llmhub.Option) (llmhub.Provider, error) {
		return fakeProvider{}, nil
	})
}

// next pops the next scripted turn and records the request.
func (h *fakeHarness) next(ctx context.Context, prompt []*llmhub.Message) (fakeTurn, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return fakeTurn{}, err
	}
	h.requests = append(h.requests, fakeRequest{prompt: snapshotMessages(prompt)})
	if len(h.script) == 0 {
		return fakeTurn{}, fmt.Errorf("fake provider: script exhausted at request %d", len(h.requests))
	}
	turn := h.script[0]
	h.script = h.script[1:]
	return turn, nil
}

// resetFake installs a fresh response script and clears captured
// requests.
func resetFake(t *testing.T, turns ...fakeTurn) {
	t.Helper()
	fakeState.mu.Lock()
	defer fakeState.mu.Unlock()
	fakeState.script = turns
	fakeState.requests = nil
}

// fakeRequests returns the provider calls captured so far.
func fakeRequests() []fakeRequest {
	fakeState.mu.Lock()
	defer fakeState.mu.Unlock()
	return append([]fakeRequest(nil), fakeState.requests...)
}

// snapshotMessages copies a message list so later transcript appends
// cannot mutate captured requests.
func snapshotMessages(msgs []*llmhub.Message) []*llmhub.Message {
	out := make([]*llmhub.Message, 0, len(msgs))
	for _, m := range msgs {
		if m == nil {
			continue
		}
		clone := &llmhub.Message{Role: m.Role, Content: append([]llmhub.ContentPart(nil), m.Content...)}
		out = append(out, clone)
	}
	return out
}

// rolesOf lists the roles of a message list.
func rolesOf(msgs []*llmhub.Message) []llmhub.Role {
	roles := make([]llmhub.Role, 0, len(msgs))
	for _, m := range msgs {
		roles = append(roles, m.Role)
	}
	return roles
}

// textOf joins the text parts of a message.
func textOf(m *llmhub.Message) string {
	var b []byte
	for _, part := range m.Content {
		if tc, ok := part.(*llmhub.TextContent); ok {
			b = append(b, tc.Text...)
		}
	}
	return string(b)
}
