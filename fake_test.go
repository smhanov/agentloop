package agentloop

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/smhanov/llmhub"
)

// fakeTurn is one scripted provider response.
type fakeTurn struct {
	text      string
	reasoning string
	calls     []*llmhub.ToolCallContent
	usage     llmhub.UsageMetadata
	err       error
}

// textTurn scripts a text-only response.
func textTurn(text string) fakeTurn { return fakeTurn{text: text} }

// callTurn scripts a response carrying tool calls.
func callTurn(calls ...*llmhub.ToolCallContent) fakeTurn { return fakeTurn{calls: calls} }

// callTextTurn scripts a response with both text and tool calls.
func callTextTurn(text string, calls ...*llmhub.ToolCallContent) fakeTurn {
	return fakeTurn{text: text, calls: calls}
}

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
	fakeState.mu.Lock()
	defer fakeState.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	fakeState.requests = append(fakeState.requests, fakeRequest{
		prompt: snapshotMessages(prompt),
		config: llmhub.NewConfig(opts...),
	})
	if fakeState.index >= len(fakeState.script) {
		return nil, fmt.Errorf("fake provider: script exhausted at request %d", fakeState.index+1)
	}
	turn := fakeState.script[fakeState.index]
	fakeState.index++
	if turn.err != nil {
		return nil, turn.err
	}
	resp := &llmhub.Response{ID: fmt.Sprintf("fake-%d", fakeState.index), Usage: turn.usage}
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
	return nil, llmhub.ErrNotImplemented
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
		ConcurrencySafe: true,
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
		ConcurrencySafe: true,
	}
}
