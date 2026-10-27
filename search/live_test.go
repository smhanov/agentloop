//go:build live

// Live smoke tests: a real model via the z.ai
// coding endpoint plus real Serper search. They never run in the default
// suite or CI — the `live` build tag excludes them — and they do not
// count against the default suite's time budget. Run them explicitly:
//
//	ZAI_API_KEY=… SERPER_API_KEY=… go test -tags live ./search -v -timeout 10m
//
// Both credential environment variables must be set; otherwise every test
// skips with a clear message. Assertions are structural only (tool-call
// counts, source shapes, non-empty text) — never exact prose — and each
// test logs the full answer and source list for triage.
package search

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/smhanov/llmhub"

	// The zai provider registers itself on import; this is the default
	// live configuration.
	_ "github.com/smhanov/llmhub/providers/zai"
)

// liveTimeout bounds each live test's context. Real model+search runs
// take tens of seconds; the budget stays far below `go test -timeout`.
const liveTimeout = 120 * time.Second

// liveLLM builds the live provider client from the environment: the z.ai
// coding endpoint with the default model (§12.11).
func liveLLM(t *testing.T) *llmhub.Client {
	t.Helper()
	llm, err := llmhub.New("zai", os.Getenv("ZAI_API_KEY"),
		llmhub.WithBaseURL("https://api.z.ai/api/coding/paas/v4"),
		llmhub.WithModel("glm-5.3-flash"))
	if err != nil {
		t.Fatalf("build live llmhub client: %v", err)
	}
	return llm
}

// liveAgent builds the search agent wired to real Serper via the key
// environment variable.
func liveAgent(t *testing.T) *Agent {
	t.Helper()
	a, err := New(liveLLM(t), WithSerperKey(os.Getenv("SERPER_API_KEY")))
	if err != nil {
		t.Fatalf("build search agent: %v", err)
	}
	return a
}

// skipUnlessLive skips the test unless both credential variables are set.
func skipUnlessLive(t *testing.T) {
	t.Helper()
	for _, key := range []string{"ZAI_API_KEY", "SERPER_API_KEY"} {
		if os.Getenv(key) == "" {
			t.Skipf("%s not set; live tests need both ZAI_API_KEY and SERPER_API_KEY (see testdata-live/README.md)", key)
		}
	}
}

// liveContext returns the test's context with the generous live timeout.
func liveContext(t *testing.T) (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), liveTimeout)
}

// countOKTools counts successful executions of one tool in a trace.
func countOKTools(t *testing.T, ans *Answer, name string) int {
	t.Helper()
	n := 0
	for _, tr := range ans.Tools {
		if tr.Name == name && tr.Err == nil {
			n++
		}
	}
	return n
}

// logSources prints the extracted source list for triage.
func logSources(t *testing.T, ans *Answer) {
	t.Helper()
	for i, s := range ans.Sources {
		t.Logf("source [%d] fetched=%v %s %s", i+1, s.Fetched, s.Title, s.URL)
	}
}

// TestLiveCurrentFacts asks one current-facts question that cannot be
// answered from training data: it must exercise at least one web_search,
// yield non-empty Sources, and produce a non-empty answer.
func TestLiveCurrentFacts(t *testing.T) {
	skipUnlessLive(t)
	ctx, cancel := liveContext(t)
	defer cancel()

	ans, err := liveAgent(t).Ask(ctx, "What Go version was released most recently, and when?")
	if err != nil {
		t.Fatalf("ask: %v", err)
	}
	t.Logf("answer (%d steps, %d tool calls):\n%s", ans.Steps, ans.ToolCalls, ans.Text)
	logSources(t, ans)

	if n := countOKTools(t, ans, searchToolName); n < 1 {
		t.Errorf("web_search calls = %d, want ≥ 1 (answer: %q)", n, ans.Text)
	}
	if len(ans.Sources) == 0 {
		t.Error("Sources is empty, want at least one entry")
	}
	if strings.TrimSpace(ans.Text) == "" {
		t.Error("answer text is empty")
	}
}

// TestLiveFetchChain asks the fetch-dependent question: the model must
// chain web_search → fetch_page, and Sources must mark the fetched page
// (its URL matches one of the returned sources).
func TestLiveFetchChain(t *testing.T) {
	skipUnlessLive(t)
	ctx, cancel := liveContext(t)
	defer cancel()

	ans, err := liveAgent(t).Ask(ctx, "Find the most recent Go blog post about loop variables, fetch the full article, and summarize the exact semantics change in 3 sentences.")
	if err != nil {
		t.Fatalf("ask: %v", err)
	}
	t.Logf("answer (%d steps, %d tool calls):\n%s", ans.Steps, ans.ToolCalls, ans.Text)
	logSources(t, ans)

	if n := countOKTools(t, ans, fetchToolName); n < 1 {
		t.Errorf("fetch_page calls = %d, want ≥ 1 (answer: %q)", n, ans.Text)
	}
	fetched := false
	for _, s := range ans.Sources {
		if s.Fetched {
			fetched = true
			break
		}
	}
	if !fetched {
		t.Error("no Source marked Fetched, want the fetch_page URL to match a returned source")
	}
	if strings.TrimSpace(ans.Text) == "" {
		t.Error("answer text is empty")
	}
}

// TestLiveZeroToolControl verifies AutoToolChoice on a real model with a
// question answerable entirely from static knowledge: the loop must
// finish in a single step with zero tool calls. The local corpus carries
// the domain-bearing counterpart of this control; the committed file
// stays generic.
func TestLiveZeroToolControl(t *testing.T) {
	skipUnlessLive(t)
	ctx, cancel := liveContext(t)
	defer cancel()

	ans, err := liveAgent(t).Ask(ctx, "Without searching the web: what is the largest planet in the solar system? Answer in one sentence.")
	if err != nil {
		t.Fatalf("ask: %v", err)
	}
	t.Logf("answer (%d steps, %d tool calls):\n%s", ans.Steps, ans.ToolCalls, ans.Text)

	if ans.Steps != 1 {
		t.Errorf("Steps = %d, want 1 (zero-tool control)", ans.Steps)
	}
	if ans.ToolCalls != 0 {
		t.Errorf("ToolCalls = %d, want 0 (zero-tool control)", ans.ToolCalls)
	}
	if strings.TrimSpace(ans.Text) == "" {
		t.Error("answer text is empty")
	}
}
