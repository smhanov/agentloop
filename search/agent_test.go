package search

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/smhanov/agentloop"
	"github.com/smhanov/llmhub"
)

// firstSearch and the following fixtures are generic, domain-neutral
// sources used across the preset tests.
const (
	urlA = "https://a.example/1"
	urlB = "https://b.example/2"
	urlC = "https://c.example/3"
	urlD = "https://d.example/4"
)

func TestSourcesFromTrace(t *testing.T) {
	ans := &agentloop.Answer{Tools: []agentloop.ToolResult{
		{Name: searchToolName, Output: `{"results":[
			{"title":"Alpha","link":"` + urlA + `","snippet":"first"},
			{"title":"Beta","link":"` + urlB + `","snippet":"second","date":"2 days ago"}]}`},
		{Name: fetchToolName, Args: json.RawMessage(`{"url":"` + urlA + `"}`),
			Output: `{"url":"` + urlA + `","title":"Alpha","text":"full"}`},
		{Name: searchToolName, Output: `{"results":[
			{"title":"Beta","link":"` + urlB + `","snippet":"second"},
			{"title":"Gamma","link":"` + urlC + `","snippet":"third"}]}`},
		{Name: fetchToolName, Args: json.RawMessage(`{"url":"` + urlC + `"}`),
			Output: `{"error":"fetch failed: 404"}`,
			Err:    fmt.Errorf("fetch failed: 404")},
		{Name: searchToolName, Output: `{"error":"search failed: 429"}`,
			Err: fmt.Errorf("search failed: 429")},
	}}

	sources := Sources(ans)
	if len(sources) != 3 {
		t.Fatalf("got %d sources, want 3 deduped: %+v", len(sources), sources)
	}
	wantURLs := []string{urlA, urlB, urlC} // first-seen order
	for i, want := range wantURLs {
		if sources[i].URL != want {
			t.Errorf("sources[%d].URL = %q, want %q", i, sources[i].URL, want)
		}
	}
	if !sources[0].Fetched {
		t.Errorf("sources[0] fetched, got %+v", sources[0])
	}
	if sources[1].Fetched || sources[2].Fetched {
		t.Errorf("unfetched sources marked Fetched: %+v", sources)
	}
	if sources[0].Title != "Alpha" || sources[1].Date != "2 days ago" {
		t.Errorf("payload fields lost: %+v", sources)
	}

	if got := Sources(nil); got != nil {
		t.Errorf("Sources(nil) = %v, want nil", got)
	}
	if got := Sources(&agentloop.Answer{}); got != nil {
		t.Errorf("Sources(empty) = %v, want nil", got)
	}
}

func TestSourcesFetchOnlyVisibility(t *testing.T) {
	ans := &agentloop.Answer{Tools: []agentloop.ToolResult{
		// A fetch-only page read before any search still appends last.
		{Name: fetchToolName, Args: json.RawMessage(`{"url":"` + urlD + `"}`),
			Output: `{"url":"` + urlD + `","title":"Delta","text":"early text"}`},
		{Name: searchToolName, Output: `{"results":[
			{"title":"Alpha","link":"` + urlA + `","snippet":"first"},
			{"title":"Beta","link":"` + urlB + `","snippet":"second"}]}`},
		{Name: fetchToolName, Args: json.RawMessage(`{"url":"` + urlC + `"}`),
			Output: `{"url":"` + urlC + `","title":"Gamma","text":"full text"}`},
		// Fetching an already-seen URL marks it without duplicating.
		{Name: fetchToolName, Args: json.RawMessage(`{"url":"` + urlA + `"}`),
			Output: `{"url":"` + urlA + `","title":"Alpha","text":"full"}`},
		// Failed and malformed fetches add nothing.
		{Name: fetchToolName, Args: json.RawMessage(`{"url":"https://gone.example/x"}`),
			Output: `{"error":"fetch failed: 404"}`,
			Err:    fmt.Errorf("fetch failed: 404")},
		{Name: fetchToolName, Args: json.RawMessage(`{"url":"https://raw.example/y"}`),
			Output: `not json`},
	}}

	sources := Sources(ans)
	if len(sources) != 4 {
		t.Fatalf("got %d sources, want 4: %+v", len(sources), sources)
	}
	// Search-derived sources keep first-seen order; fetch-only pages append
	// after them in fetch order.
	for i, want := range []string{urlA, urlB, urlD, urlC} {
		if sources[i].URL != want {
			t.Errorf("sources[%d].URL = %q, want %q", i, sources[i].URL, want)
		}
	}
	// Fetch-only entries carry the payload's title and Fetched, and none of
	// the search-only fields.
	for i, wantTitle := range []string{"Delta", "Gamma"} {
		s := sources[2+i]
		if !s.Fetched {
			t.Errorf("sources[%d] fetched page not marked: %+v", 2+i, s)
		}
		if s.Title != wantTitle {
			t.Errorf("sources[%d].Title = %q, want %q", 2+i, s.Title, wantTitle)
		}
		if s.Snippet != "" || s.Date != "" || s.Publisher != "" {
			t.Errorf("sources[%d] carries search-only fields: %+v", 2+i, s)
		}
	}
	if !sources[0].Fetched {
		t.Errorf("already-seen URL re-fetched but not marked Fetched: %+v", sources[0])
	}
	if sources[1].Fetched {
		t.Errorf("unfetched source marked Fetched: %+v", sources[1])
	}
}

func TestPresetSearchFetchAnswerFlow(t *testing.T) {
	s := &scriptedSearcher{responses: []searchResponse{
		{sources: []Source{
			{Title: "Alpha", URL: urlA, Snippet: "first"},
			{Title: "Beta", URL: urlB, Snippet: "second"},
		}},
		{sources: []Source{
			{Title: "Beta", URL: urlB, Snippet: "second"},
			{Title: "Gamma", URL: urlC, Snippet: "third"},
		}, related: []string{"retry differently"}},
	}}
	f := &scriptedFetcher{responses: []fetchResponse{
		{page: &Page{URL: urlA, Title: "Alpha", Text: "full page text"}},
	}}
	resetScript(t,
		callTurn(searchCall("c1", `{"q":"first"}`)),
		callTurn(searchCall("c2", `{"q":"second"}`)),
		callTurn(fetchCall("c3", `{"url":"`+urlA+`"}`)),
		textTurn("Grounded answer [1]."),
	)
	agent := newFakeAgent(t, WithSearcher(s), WithFetcher(f))

	ans, err := agent.Ask(context.Background(), "question needing sources")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ans.Text != "Grounded answer [1]." {
		t.Errorf("Text = %q", ans.Text)
	}
	if ans.Answer == nil || ans.Answer.Text != ans.Text {
		t.Fatalf("embedded core answer missing or detached")
	}
	if ans.ToolCalls != 3 || ans.Steps != 4 {
		t.Errorf("ToolCalls=%d Steps=%d, want 3/4", ans.ToolCalls, ans.Steps)
	}
	if len(ans.Sources) != 3 {
		t.Fatalf("Sources = %+v, want 3", ans.Sources)
	}
	for i, want := range []string{urlA, urlB, urlC} {
		if ans.Sources[i].URL != want {
			t.Errorf("Sources[%d].URL = %q, want %q (first-seen order)", i, ans.Sources[i].URL, want)
		}
	}
	if !ans.Sources[0].Fetched || ans.Sources[1].Fetched || ans.Sources[2].Fetched {
		t.Errorf("Fetched marking wrong: %+v", ans.Sources)
	}
	if ans.Sources[2].Title != "Gamma" {
		t.Errorf("Sources[2] = %+v", ans.Sources[2])
	}
	if reqs := s.seen(); len(reqs) != 2 {
		t.Errorf("searcher requests = %d, want 2", len(reqs))
	}
	if reqs := f.seen(); len(reqs) != 1 || reqs[0].URL != urlA {
		t.Errorf("fetcher requests = %+v", reqs)
	}
}

func TestNumClamping(t *testing.T) {
	s := &scriptedSearcher{responses: []searchResponse{{}, {}, {}}}
	resetScript(t,
		callTurn(
			searchCall("c1", `{"q":"no num"}`),
			searchCall("c2", `{"q":"too many","num":25}`),
			searchCall("c3", `{"q":"few","num":3}`),
		),
		textTurn("done"),
	)
	agent := newFakeAgent(t, WithSearcher(s))
	if _, err := agent.Ask(context.Background(), "q"); err != nil {
		t.Fatalf("Ask: %v", err)
	}
	var nums []int
	for _, req := range s.seen() {
		nums = append(nums, req.Num)
	}
	if len(nums) != 3 {
		t.Fatalf("got %d requests, want 3", len(nums))
	}
	// Parallel dispatch means recording order varies; compare as a multiset.
	sort.Ints(nums)
	for i, want := range []int{3, 10, 10} {
		if nums[i] != want {
			t.Errorf("sorted num[%d] = %d, want %v", i, nums[i], want)
		}
	}

	s5 := &scriptedSearcher{responses: []searchResponse{{}}}
	resetScript(t, callTurn(searchCall("c1", `{"q":"a"}`)), textTurn("done"))
	agent5 := newFakeAgent(t, WithSearcher(s5), WithMaxResults(5))
	if _, err := agent5.Ask(context.Background(), "q"); err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if reqs := s5.seen(); len(reqs) != 1 || reqs[0].Num != 5 {
		t.Errorf("WithMaxResults(5): requests = %+v, want Num 5", reqs)
	}
}

func TestFetchTruncation(t *testing.T) {
	marker := "<5000 characters elided>"
	long := strings.Repeat("a", 25000)

	f := &scriptedFetcher{responses: []fetchResponse{
		{page: &Page{URL: urlA, Title: "Long", Text: long}},
	}}
	resetScript(t, callTurn(fetchCall("c1", `{"url":"`+urlA+`"}`)), textTurn("done"))
	agent := newFakeAgent(t, WithFetcher(f))
	ans, err := agent.Ask(context.Background(), "read the page")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	var payload fetchPayload
	if err := json.Unmarshal([]byte(ans.Tools[0].Output), &payload); err != nil {
		t.Fatalf("decode fetch payload: %v", err)
	}
	if payload.URL != urlA || payload.Title != "Long" {
		t.Errorf("payload envelope = %+v", payload)
	}
	if got := payload.Text; got != strings.Repeat("a", 10000)+marker+strings.Repeat("a", 10000) {
		t.Errorf("truncated text has wrong shape: len=%d", len(got))
	}
	if !strings.Contains(payload.Text, marker) {
		t.Errorf("elision marker missing")
	}

	// A small limit splits head and tail around the marker.
	f7 := &scriptedFetcher{responses: []fetchResponse{
		{page: &Page{URL: urlA, Text: "0123456789"}},
	}}
	resetScript(t, callTurn(fetchCall("c1", `{"url":"`+urlA+`"}`)), textTurn("done"))
	agent7 := newFakeAgent(t, WithFetcher(f7), WithFetchLimit(7))
	ans7, err := agent7.Ask(context.Background(), "read the page")
	if err != nil {
		t.Fatalf("Ask small limit: %v", err)
	}
	var p7 fetchPayload
	if err := json.Unmarshal([]byte(ans7.Tools[0].Output), &p7); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if want := "012<4 characters elided>789"; p7.Text != want {
		t.Errorf("small-limit text = %q, want %q", p7.Text, want)
	}

	// Zero means unlimited.
	f0 := &scriptedFetcher{responses: []fetchResponse{
		{page: &Page{URL: urlA, Text: long}},
	}}
	resetScript(t, callTurn(fetchCall("c1", `{"url":"`+urlA+`"}`)), textTurn("done"))
	agent0 := newFakeAgent(t, WithFetcher(f0), WithFetchLimit(0))
	ans0, err := agent0.Ask(context.Background(), "read the page")
	if err != nil {
		t.Fatalf("Ask unlimited: %v", err)
	}
	var p0 fetchPayload
	if err := json.Unmarshal([]byte(ans0.Tools[0].Output), &p0); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if p0.Text != long {
		t.Errorf("unlimited text len = %d, want %d", len(p0.Text), len(long))
	}
}

func TestInvalidToolArgs(t *testing.T) {
	f := &scriptedFetcher{responses: []fetchResponse{{page: &Page{}}}}
	s := &scriptedSearcher{responses: []searchResponse{{}}}
	resetScript(t,
		callTurn(
			searchCall("c1", `not json`),
			searchCall("c2", `{"q":"","type":"news"}`),
			searchCall("c3", `{"q":"x","type":"videos"}`),
			fetchCall("c4", `not json`),
			fetchCall("c5", `{"url":"ftp://example.com/f"}`),
			fetchCall("c6", `{"url":"example.com/page"}`),
			fetchCall("c7", `{}`),
		),
		textTurn("done"),
	)
	agent := newFakeAgent(t, WithSearcher(s), WithFetcher(f))
	ans, err := agent.Ask(context.Background(), "q")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if len(ans.Tools) != 7 {
		t.Fatalf("got %d tool results, want 7", len(ans.Tools))
	}
	for i, tr := range ans.Tools {
		if want := `{"error":"invalid arguments"}`; tr.Output != want {
			t.Errorf("Tools[%d].Output = %q, want %q", i, tr.Output, want)
		}
	}
	if ans.Status != agentloop.StatusCompleted {
		t.Errorf("Status = %v, want completed (malformed args never abort the run)", ans.Status)
	}
}

func TestUnconfiguredKeyGraceful(t *testing.T) {
	t.Setenv("SERPER_API_KEY", "")
	resetScript(t,
		callTurn(searchCall("c1", `{"q":"anything"}`)),
		callTurn(fetchCall("c2", `{"url":"`+urlA+`"}`)),
		textTurn("I cannot verify that without search."),
	)
	agent := newFakeAgent(t) // no key, no injected backends
	ans, err := agent.Ask(context.Background(), "q")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if len(ans.Tools) != 2 {
		t.Fatalf("got %d tool results, want 2", len(ans.Tools))
	}
	if want := `{"error":"web search is not configured"}`; ans.Tools[0].Output != want {
		t.Errorf("search observation = %q, want %q", ans.Tools[0].Output, want)
	}
	if want := `{"error":"page fetching is not configured"}`; ans.Tools[1].Output != want {
		t.Errorf("fetch observation = %q, want %q", ans.Tools[1].Output, want)
	}
	if ans.Status != agentloop.StatusCompleted || ans.Text == "" {
		t.Errorf("agent did not complete ungrounded: status=%v text=%q", ans.Status, ans.Text)
	}
}

func TestWithNoFetchDropsTool(t *testing.T) {
	t.Setenv("SERPER_API_KEY", "")
	s := &scriptedSearcher{responses: []searchResponse{
		{sources: []Source{{Title: "T", URL: urlA, Snippet: "S"}}},
	}}
	resetScript(t,
		callTurn(fetchCall("c1", `{"url":"`+urlA+`"}`)),
		callTurn(searchCall("c2", `{"q":"x"}`)),
		textTurn("done"),
	)
	agent := newFakeAgent(t, WithSearcher(s), WithNoFetch())
	ans, err := agent.Ask(context.Background(), "q")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if len(ans.Tools) != 2 {
		t.Fatalf("got %d tool results, want 2", len(ans.Tools))
	}
	if want := `{"error":"unknown tool"}`; ans.Tools[0].Output != want {
		t.Errorf("fetch_page observation = %q, want %q", ans.Tools[0].Output, want)
	}
	if !strings.Contains(ans.Tools[1].Output, `"results"`) {
		t.Errorf("web_search observation = %q, want a result payload", ans.Tools[1].Output)
	}
}

func TestSerper429ModelObservation(t *testing.T) {
	stubSleeper(t)
	var attempts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"message":"rate limited"}`))
	}))
	defer srv.Close()

	resetScript(t,
		callTurn(searchCall("c1", `{"q":"x"}`)),
		textTurn("Answering without results."),
	)
	agent := newFakeAgent(t,
		WithSerperBaseURL(srv.URL), WithSerperKey("k"), WithSerperRetries(1))
	ans, err := agent.Ask(context.Background(), "q")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if want := `{"error":"search failed: 429"}`; ans.Tools[0].Output != want {
		t.Errorf("model observation = %q, want %q", ans.Tools[0].Output, want)
	}
	if got := attempts.Load(); got != 2 {
		t.Errorf("attempts = %d, want 2 (one retry, then exhausted)", got)
	}
	if ans.Status != agentloop.StatusCompleted {
		t.Errorf("Status = %v, want completed", ans.Status)
	}
}

func TestFetchFailureObservation(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"message":"Page not found.","statusCode":404}`))
	}))
	defer srv.Close()

	resetScript(t,
		callTurn(fetchCall("c1", `{"url":"`+urlA+`"}`)),
		textTurn("That page is gone; trying another."),
	)
	agent := newFakeAgent(t, WithSerperBaseURL(srv.URL), WithSerperKey("k"))
	ans, err := agent.Ask(context.Background(), "q")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if want := `{"error":"fetch failed: 404"}`; ans.Tools[0].Output != want {
		t.Errorf("model observation = %q, want %q", ans.Tools[0].Output, want)
	}
}

func TestPresetKeyAndHeaders(t *testing.T) {
	var rec recorder
	srv := httptest.NewServer(rec.handle(http.StatusOK, `{"organic":[]}`))
	defer srv.Close()

	resetScript(t, callTurn(searchCall("c1", `{"q":"g"}`)), textTurn("done"))
	agent := newFakeAgent(t, WithSerperBaseURL(srv.URL), WithSerperKey("secret"))
	if _, err := agent.Ask(context.Background(), "q"); err != nil {
		t.Fatalf("Ask: %v", err)
	}
	reqs := rec.snapshot()
	if len(reqs) != 1 {
		t.Fatalf("got %d requests, want 1", len(reqs))
	}
	if reqs[0].APIKey != "secret" {
		t.Errorf("X-API-KEY = %q, want WithSerperKey override", reqs[0].APIKey)
	}

	// The environment variable is the fallback when no option is given.
	t.Setenv("SERPER_API_KEY", "env-key")
	var rec2 recorder
	srv2 := httptest.NewServer(rec2.handle(http.StatusOK, `{"organic":[]}`))
	defer srv2.Close()

	resetScript(t, callTurn(searchCall("c1", `{"q":"g"}`)), textTurn("done"))
	agent2 := newFakeAgent(t, WithSerperBaseURL(srv2.URL))
	if _, err := agent2.Ask(context.Background(), "q"); err != nil {
		t.Fatalf("Ask env: %v", err)
	}
	if reqs := rec2.snapshot(); len(reqs) != 1 || reqs[0].APIKey != "env-key" {
		t.Errorf("requests = %+v, want X-API-KEY from SERPER_API_KEY", reqs)
	}
}

func TestPresetNewsRouting(t *testing.T) {
	var rec recorder
	srv := httptest.NewServer(rec.handle(http.StatusOK, `{"news":[]}`))
	defer srv.Close()

	resetScript(t, callTurn(searchCall("c1", `{"q":"breaking","type":"news"}`)), textTurn("done"))
	agent := newFakeAgent(t, WithSerperBaseURL(srv.URL), WithSerperKey("k"))
	ans, err := agent.Ask(context.Background(), "q")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	reqs := rec.snapshot()
	if len(reqs) != 1 || reqs[0].Path != "/news" {
		t.Fatalf("paths = %+v, want one POST /news", reqs)
	}
	// The news payload keeps date and publisher for the model and Sources.
	if !strings.Contains(ans.Tools[0].Output, `"results":[]`) {
		t.Errorf("news payload = %q, want trimmed results payload", ans.Tools[0].Output)
	}
}

func TestCustomBackendsNeedNoKey(t *testing.T) {
	t.Setenv("SERPER_API_KEY", "")
	s := &scriptedSearcher{responses: []searchResponse{
		{sources: []Source{{Title: "T", URL: urlA, Snippet: "S"}}},
	}}
	f := &scriptedFetcher{responses: []fetchResponse{
		{page: &Page{URL: urlA, Title: "T", Text: "body"}},
	}}
	resetScript(t,
		callTurn(searchCall("c1", `{"q":"x"}`)),
		callTurn(fetchCall("c2", `{"url":"`+urlA+`"}`)),
		textTurn("done"),
	)
	agent := newFakeAgent(t, WithSearcher(s), WithFetcher(f))
	ans, err := agent.Ask(context.Background(), "q")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	for i, tr := range ans.Tools {
		if strings.Contains(tr.Output, `"error"`) {
			t.Errorf("Tools[%d].Output = %q, want success via injected backend", i, tr.Output)
		}
	}
}

func TestDefaultPromptAndToolsRegistered(t *testing.T) {
	t.Setenv("SERPER_API_KEY", "")
	resetScript(t, textTurn("ok"))
	agent := newFakeAgent(t)
	ans, err := agent.Ask(context.Background(), "q")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	// The system prompt is the §7.1 default: research assistant framing.
	if len(ans.Messages) == 0 || ans.Messages[0].Role != llmhub.RoleSystem {
		t.Fatalf("first message = %+v, want system", ans.Messages)
	}
	var sys string
	for _, part := range ans.Messages[0].Content {
		if tc, ok := part.(*llmhub.TextContent); ok {
			sys += tc.Text
		}
	}
	if !strings.HasPrefix(sys, "You are a research assistant with access to two tools backed by Google:") {
		t.Errorf("system prompt = %q", sys)
	}
	if !strings.Contains(sys, "`web_search`") || !strings.Contains(sys, "`fetch_page`") {
		t.Errorf("system prompt missing tool names: %q", sys)
	}
}
