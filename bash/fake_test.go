package bash

import (
	"context"
	"fmt"
	"sync"
	"testing"

	"github.com/smhanov/llmhub"
)

// fakeTurn is one scripted provider response for Generate. usage feeds the
// cost-limit tests; calls simulate a misbehaving provider that answers
// with native tool calls instead of the text protocol.
type fakeTurn struct {
	text  string
	usage llmhub.UsageMetadata
	calls []*llmhub.ToolCallContent
}

// textTurn scripts a plain text response.
func textTurn(text string) fakeTurn { return fakeTurn{text: text} }

// cmdTurn scripts a protocol-conformant response: a THOUGHT plus one
// mswea_bash_command fence carrying command.
func cmdTurn(thought, command string) fakeTurn {
	return textTurn(thought + "\n" + fence + "mswea_bash_command\n" + command + "\n" + fence)
}

// bashTurn scripts a response using the plain bash fence alias.
func bashTurn(command string) fakeTurn {
	return textTurn(fence + "bash\n" + command + "\n" + fence)
}

// submitTurn scripts the finishing command: the marker on the first output
// line and submission on the second.
func submitTurn(thought, submission string) fakeTurn {
	return cmdTurn(thought, "echo '"+defaultSubmitMarker+"'\necho '"+submission+"'")
}

// callTurn scripts a native tool-call response (no text).
func callTurn(calls ...*llmhub.ToolCallContent) fakeTurn {
	return fakeTurn{calls: calls}
}

// fakeRequest records the exact prompt and merged config the model saw.
type fakeRequest struct {
	prompt []*llmhub.Message
	config llmhub.Config
}

// fakeHarness replays fakeState.script and records every request.
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

// fakeProvider replays fakeState.script; only Generate is exercised by the
// preset tests, but Stream is required by the llmhub Provider interface.
type fakeProvider struct{}

func (fakeProvider) Name() string { return "fake" }

func (fakeProvider) Generate(ctx context.Context, prompt []*llmhub.Message, opts ...llmhub.Option) (*llmhub.Response, error) {
	turn, err := fakeState.next(ctx, prompt, opts...)
	if err != nil {
		return nil, err
	}
	resp := &llmhub.Response{ID: fmt.Sprintf("fake-gen-%d", fakeState.count()), Usage: turn.usage}
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
	ch := make(chan llmhub.StreamChunk, 2)
	if turn.text != "" {
		ch <- llmhub.StreamChunk{Delta: turn.text}
	}
	for _, call := range turn.calls {
		ch <- llmhub.StreamChunk{ToolCalls: []*llmhub.ToolCallContent{call}}
	}
	ch <- llmhub.StreamChunk{Done: true}
	close(ch)
	return ch, nil
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

// newFakeAgent returns a bash agent over the scripted fake provider.
func newFakeAgent(t *testing.T, opts ...Option) *Agent {
	t.Helper()
	client, err := llmhub.New("fake", "test-key")
	if err != nil {
		t.Fatalf("llmhub.New(fake): %v", err)
	}
	agent, err := New(client, opts...)
	if err != nil {
		t.Fatalf("bash.New: %v", err)
	}
	return agent
}

// snapshotMessages copies a message list so later transcript appends
// cannot mutate captured requests.
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
	var out string
	for _, part := range m.Content {
		if tc, ok := part.(*llmhub.TextContent); ok {
			out += tc.Text
		}
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
