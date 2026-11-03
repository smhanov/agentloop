package agentloop

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/smhanov/llmhub"
	_ "github.com/smhanov/llmhub/providers/openai"
)

func TestLevenshtein(t *testing.T) {
	cases := []struct {
		a, b string
		want int
	}{
		{"", "", 0},
		{"", "abc", 3},
		{"abc", "abc", 0},
		{"kitten", "sitting", 3},
		{"flaw", "lawn", 2},
		{"clck", "clock", 1},
		{"clockx", "clock", 1},
	}
	for _, tc := range cases {
		if got := levenshtein(tc.a, tc.b); got != tc.want {
			t.Errorf("levenshtein(%q, %q) = %d, want %d", tc.a, tc.b, got, tc.want)
		}
	}
}

func TestSuggestName(t *testing.T) {
	registered := []string{"clock", "echo", "weather"}
	cases := []struct {
		name string
		want string
	}{
		{"clock", "clock"},    // exact match
		{"clck", "clock"},     // one deletion
		{"clockx", "clock"},   // one insertion
		{"cxyck", "clock"},    // two substitutions
		{"echoo", "echo"},     // closest of the set
		{"weathr", "weather"}, // one deletion
		{"zzzzzz", ""},        // too far
		{"", ""},              // empty name
	}
	for _, tc := range cases {
		if got := suggestName(tc.name, registered); got != tc.want {
			t.Errorf("suggestName(%q) = %q, want %q", tc.name, got, tc.want)
		}
	}
}

func TestUnknownToolObservation(t *testing.T) {
	if got, want := unknownToolJSON(""), `{"error":"unknown tool"}`; got != want {
		t.Errorf("got %q, want %q", got, want)
	}
	if got, want := unknownToolJSON("clock"), `{"error":"unknown tool; did you mean \"clock\"?"}`; got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestEmptyToolCallArgumentsRoundTrip(t *testing.T) {
	var (
		mu          sync.Mutex
		bodies      []string
		handlerArgs []string
		calls       int
	)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request body: %v", err)
			return
		}
		mu.Lock()
		bodies = append(bodies, string(body))
		calls++
		n := calls
		mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		if n == 1 {
			// A no-argument tool call as some providers send it: the
			// function object carries no arguments field at all.
			io.WriteString(w, `{"id":"gen-1","choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call-1","type":"function","function":{"name":"lookup"}}]}}]}`)
			return
		}
		io.WriteString(w, `{"id":"gen-2","choices":[{"message":{"role":"assistant","content":"done"}}]}`)
	}))
	defer srv.Close()

	client, err := llmhub.New("openai", "test-key", llmhub.WithBaseURL(srv.URL), llmhub.WithModel("m"))
	if err != nil {
		t.Fatalf("llmhub.New(openai): %v", err)
	}

	lookup := Tool{
		Def: llmhub.NewTool("lookup", "Look something up", map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		}),
		Run: func(ctx context.Context, args json.RawMessage) (string, error) {
			var parsed map[string]any
			if err := json.Unmarshal(args, &parsed); err != nil {
				return "", err
			}
			mu.Lock()
			handlerArgs = append(handlerArgs, string(args))
			mu.Unlock()
			return "found", nil
		},
	}

	ans, err := MustNew(client, WithTools(lookup)).Ask(context.Background(), "look it up")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ans.Status != StatusCompleted || ans.Text != "done" {
		t.Fatalf("status/text = %v/%q, want completed/%q", ans.Status, ans.Text, "done")
	}

	mu.Lock()
	gotBodies := append([]string(nil), bodies...)
	gotHandlerArgs := append([]string(nil), handlerArgs...)
	mu.Unlock()

	// The handler sees valid JSON, not the omitted (empty) arguments.
	if len(gotHandlerArgs) != 1 || gotHandlerArgs[0] != "{}" {
		t.Errorf("handler args = %q, want [\"{}\"]", gotHandlerArgs)
	}
	if len(ans.Tools) != 1 {
		t.Fatalf("trace = %+v, want one entry", ans.Tools)
	}
	if got := string(ans.Tools[0].Args); got != "{}" {
		t.Errorf("trace args = %q, want %q", got, "{}")
	}
	if ans.Tools[0].Err != nil || ans.Tools[0].Output != "found" {
		t.Errorf("trace result = (%q, %v), want (%q, nil)", ans.Tools[0].Output, ans.Tools[0].Err, "found")
	}

	// The transcript's assistant tool call carries the normalized arguments.
	var transcriptArgs []string
	for _, msg := range ans.Messages {
		if msg.Role != llmhub.RoleAssistant {
			continue
		}
		for _, call := range callsOf(msg) {
			transcriptArgs = append(transcriptArgs, call.Arguments)
		}
	}
	if len(transcriptArgs) != 1 || transcriptArgs[0] != "{}" {
		t.Errorf("transcript tool-call args = %q, want [\"{}\"]", transcriptArgs)
	}

	// The next request must serialize the arguments field; strict
	// OpenAI-compatible servers treat it as required.
	if len(gotBodies) != 2 {
		t.Fatalf("requests = %d, want 2", len(gotBodies))
	}
	if !strings.Contains(gotBodies[1], `"arguments"`) {
		t.Errorf("second request omits the arguments field: %s", gotBodies[1])
	}
}
