package search

import (
	"context"
	"fmt"
	"sync"
	"testing"

	"github.com/smhanov/llmhub"

	"github.com/smhanov/agentloop"
)

// fakeTurn is one scripted provider response for Generate.
type fakeTurn struct {
	text  string
	calls []*llmhub.ToolCallContent
}

// textTurn scripts a text-only response.
func textTurn(text string) fakeTurn { return fakeTurn{text: text} }

// callTurn scripts a response carrying tool calls.
func callTurn(calls ...*llmhub.ToolCallContent) fakeTurn { return fakeTurn{calls: calls} }

// searchCall scripts a web_search call.
func searchCall(id, args string) *llmhub.ToolCallContent {
	return llmhub.ToolCall(id, searchToolName, args)
}

// fetchCall scripts a fetch_page call.
func fetchCall(id, args string) *llmhub.ToolCallContent {
	return llmhub.ToolCall(id, fetchToolName, args)
}

// fakeHarness replays the script of the current test and counts requests.
type fakeHarness struct {
	mu     sync.Mutex
	script []fakeTurn
	index  int
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
	turn, err := fakeState.next(ctx)
	if err != nil {
		return nil, err
	}
	resp := &llmhub.Response{ID: fmt.Sprintf("fake-gen-%d", fakeState.count())}
	if turn.text != "" {
		resp.Content = append(resp.Content, llmhub.Text(turn.text))
	}
	for _, call := range turn.calls {
		resp.Content = append(resp.Content, call)
	}
	return resp, nil
}

func (fakeProvider) Stream(ctx context.Context, prompt []*llmhub.Message, opts ...llmhub.Option) (<-chan llmhub.StreamChunk, error) {
	turn, err := fakeState.next(ctx)
	if err != nil {
		return nil, err
	}
	ch := make(chan llmhub.StreamChunk, 1+len(turn.calls))
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

// next pops the next scripted turn.
func (h *fakeHarness) next(ctx context.Context) (fakeTurn, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return fakeTurn{}, err
	}
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

// resetScript installs a fresh response script.
func resetScript(t *testing.T, turns ...fakeTurn) {
	t.Helper()
	fakeState.mu.Lock()
	defer fakeState.mu.Unlock()
	fakeState.script = turns
	fakeState.index = 0
}

// newFakeAgent returns a search agent over the scripted fake provider.
func newFakeAgent(t *testing.T, opts ...Option) *Agent {
	t.Helper()
	client, err := llmhub.New("fake", "test-key")
	if err != nil {
		t.Fatalf("llmhub.New(fake): %v", err)
	}
	agent, err := New(client, opts...)
	if err != nil {
		t.Fatalf("search.New: %v", err)
	}
	return agent
}

// scriptedSearcher answers Search calls from a fixed queue and records the
// requests it saw.
type scriptedSearcher struct {
	mu        sync.Mutex
	responses []searchResponse
	requests  []SearchRequest
}

// searchResponse is one queued Search outcome.
type searchResponse struct {
	sources []Source
	related []string
	err     error
}

func (s *scriptedSearcher) Search(ctx context.Context, req SearchRequest) ([]Source, []string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.requests = append(s.requests, req)
	if len(s.responses) == 0 {
		return nil, nil, fmt.Errorf("scriptedSearcher: script exhausted")
	}
	resp := s.responses[0]
	s.responses = s.responses[1:]
	return resp.sources, resp.related, resp.err
}

// seen returns the recorded requests.
func (s *scriptedSearcher) seen() []SearchRequest {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]SearchRequest(nil), s.requests...)
}

// scriptedFetcher answers Fetch calls from a fixed queue and records the
// requests it saw.
type scriptedFetcher struct {
	mu        sync.Mutex
	responses []fetchResponse
	requests  []FetchRequest
}

// fetchResponse is one queued Fetch outcome.
type fetchResponse struct {
	page *Page
	err  error
}

func (f *scriptedFetcher) Fetch(ctx context.Context, req FetchRequest) (*Page, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.requests = append(f.requests, req)
	if len(f.responses) == 0 {
		return nil, fmt.Errorf("scriptedFetcher: script exhausted")
	}
	resp := f.responses[0]
	f.responses = f.responses[1:]
	return resp.page, resp.err
}

// seen returns the recorded requests.
func (f *scriptedFetcher) seen() []FetchRequest {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]FetchRequest(nil), f.requests...)
}

// coreTrace runs the agent on the current script and returns the core
// answer for trace inspection.
func coreTrace(t *testing.T, agent *Agent, query string) *agentloop.Answer {
	t.Helper()
	ans, err := agent.Ask(context.Background(), query)
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	return ans.Answer
}
