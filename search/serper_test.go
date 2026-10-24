package search

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// capturedRequest is one recorded inbound request.
type capturedRequest struct {
	Method string
	Path   string
	APIKey string
	CT     string
	Body   map[string]any
}

// recorder collects requests and replies with a fixed status and body.
type recorder struct {
	mu       sync.Mutex
	requests []capturedRequest
}

// handle returns a handler that records the request and replies with
// status and response.
func (rec *recorder) handle(status int, response string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &body)
		rec.mu.Lock()
		rec.requests = append(rec.requests, capturedRequest{
			Method: r.Method,
			Path:   r.URL.Path,
			APIKey: r.Header.Get("X-API-KEY"),
			CT:     r.Header.Get("Content-Type"),
			Body:   body,
		})
		rec.mu.Unlock()
		w.WriteHeader(status)
		_, _ = io.WriteString(w, response)
	}
}

// snapshot returns the recorded requests.
func (rec *recorder) snapshot() []capturedRequest {
	rec.mu.Lock()
	defer rec.mu.Unlock()
	return append([]capturedRequest(nil), rec.requests...)
}

// stubSleeper replaces the retry sleeper with an instant no-op that
// records the requested backoffs, keeping tests fast and deterministic.
func stubSleeper(t *testing.T) *[]time.Duration {
	t.Helper()
	sleeps := &[]time.Duration{}
	prev := sleeper
	sleeper = func(ctx context.Context, d time.Duration) error {
		*sleeps = append(*sleeps, d)
		return ctx.Err()
	}
	t.Cleanup(func() { sleeper = prev })
	return sleeps
}

func TestSerperRoutingAndHeaders(t *testing.T) {
	var rec recorder
	srv := httptest.NewServer(rec.handle(http.StatusOK,
		`{"organic":[{"title":"T","link":"https://a.example/1","snippet":"S"}]}`))
	defer srv.Close()

	c := newSerperClient(srv.URL, "k", nil, 0)
	sources, _, err := c.Search(context.Background(), SearchRequest{Query: "g", Num: 3})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(sources) != 1 || sources[0].URL != "https://a.example/1" {
		t.Fatalf("sources = %+v, want one result", sources)
	}
	reqs := rec.snapshot()
	if len(reqs) != 1 {
		t.Fatalf("got %d requests, want 1", len(reqs))
	}
	req := reqs[0]
	if req.Method != http.MethodPost || req.Path != "/search" {
		t.Errorf("method/path = %s %s, want POST /search", req.Method, req.Path)
	}
	if req.APIKey != "k" {
		t.Errorf("X-API-KEY = %q, want %q", req.APIKey, "k")
	}
	if req.CT != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", req.CT)
	}
	if req.Body["q"] != "g" || req.Body["num"] != float64(3) {
		t.Errorf("body = %v, want {q: g, num: 3}", req.Body)
	}
}

func TestSerperNewsRoutingByType(t *testing.T) {
	var rec recorder
	srv := httptest.NewServer(rec.handle(http.StatusOK, `{"news":[]}`))
	defer srv.Close()

	c := newSerperClient(srv.URL, "k", nil, 0)
	if _, _, err := c.Search(context.Background(), SearchRequest{Query: "g", Type: SearchNews}); err != nil {
		t.Fatalf("Search news: %v", err)
	}
	reqs := rec.snapshot()
	if len(reqs) != 1 || reqs[0].Path != "/news" {
		t.Fatalf("paths = %v, want [/news]", reqs)
	}
	if _, _, err := c.Search(context.Background(), SearchRequest{Query: "g"}); err != nil {
		t.Fatalf("Search general: %v", err)
	}
	if reqs = rec.snapshot(); len(reqs) != 2 || reqs[1].Path != "/search" {
		t.Fatalf("paths = %v, want second request on /search", reqs)
	}
}

func TestSerperKnowledgeGraphPrepended(t *testing.T) {
	var rec recorder
	srv := httptest.NewServer(rec.handle(http.StatusOK, `{
		"knowledgeGraph": {"title": "KG", "description": "KG desc", "websiteLink": "https://kg.example"},
		"organic": [
			{"title": "T1", "link": "https://a.example/1", "snippet": "S1"},
			{"title": "T2", "link": "https://a.example/2", "snippet": "S2", "date": "2 days ago"},
			{"title": "T3", "link": "https://a.example/3", "snippet": "S3"}
		]}`))
	defer srv.Close()

	c := newSerperClient(srv.URL, "k", nil, 0)
	sources, _, err := c.Search(context.Background(), SearchRequest{Query: "g", Num: 2})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(sources) != 3 {
		t.Fatalf("got %d sources, want 3 (KG prepended + 2 capped organic)", len(sources))
	}
	kg := sources[0]
	if kg.Title != "KG" || kg.Snippet != "KG desc" || kg.URL != "https://kg.example" {
		t.Errorf("knowledge graph source = %+v", kg)
	}
	if sources[1].Date != "" {
		t.Errorf("organic date present, want dropped: %+v", sources[1])
	}
	if sources[2].Date != "2 days ago" {
		t.Errorf("organic date = %q, want %q", sources[2].Date, "2 days ago")
	}
}

func TestSerperNewsMapping(t *testing.T) {
	var rec recorder
	srv := httptest.NewServer(rec.handle(http.StatusOK, `{
		"news": [
			{"title": "N1", "link": "https://n.example/1", "snippet": "S1",
			 "date": "1 hour ago", "source": "The Paper"},
			{"title": "N2", "link": "https://n.example/2", "snippet": "S2",
			 "date": "3 hours ago", "source": "Other Outlet"}
		]}`))
	defer srv.Close()

	c := newSerperClient(srv.URL, "k", nil, 0)
	sources, _, err := c.Search(context.Background(), SearchRequest{Query: "g", Type: SearchNews})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(sources) != 2 {
		t.Fatalf("got %d sources, want 2", len(sources))
	}
	for i, want := range []struct{ date, publisher string }{
		{"1 hour ago", "The Paper"},
		{"3 hours ago", "Other Outlet"},
	} {
		if sources[i].Date != want.date || sources[i].Publisher != want.publisher {
			t.Errorf("sources[%d] = %+v, want date %q publisher %q", i, sources[i], want.date, want.publisher)
		}
	}
}

func TestSerperRelatedCapAndPeopleAlsoAskDropped(t *testing.T) {
	var rec recorder
	srv := httptest.NewServer(rec.handle(http.StatusOK, `{
		"organic": [{"title": "T", "link": "https://a.example/1", "snippet": "S"}],
		"peopleAlsoAsk": [
			{"question": "Q1?", "snippet": "low signal"},
			{"question": "Q2?", "snippet": "no answers"}
		],
		"relatedSearches": [
			{"query": "r1"}, {"query": "r2"}, {"query": "r3"},
			{"query": "r4"}, {"query": "r5"}, {"query": "r6"}, {"query": "r7"}
		]}`))
	defer srv.Close()

	c := newSerperClient(srv.URL, "k", nil, 0)
	sources, related, err := c.Search(context.Background(), SearchRequest{Query: "g"})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(sources) != 1 {
		t.Fatalf("got %d sources, want 1 (peopleAlsoAsk dropped)", len(sources))
	}
	if len(related) != 5 {
		t.Fatalf("related = %v, want capped at 5", related)
	}
	for i, r := range related {
		if want := fmt.Sprintf("r%d", i+1); r != want {
			t.Errorf("related[%d] = %q, want %q (first-seen order)", i, r, want)
		}
	}
}

func TestSerperEmptyResults(t *testing.T) {
	var rec recorder
	srv := httptest.NewServer(rec.handle(http.StatusOK, `{}`))
	defer srv.Close()

	c := newSerperClient(srv.URL, "k", nil, 0)
	sources, related, err := c.Search(context.Background(), SearchRequest{Query: "g"})
	if err != nil {
		t.Fatalf("Search on empty response: %v", err)
	}
	if len(sources) != 0 || len(related) != 0 {
		t.Fatalf("sources=%v related=%v, want both empty", sources, related)
	}
}

func TestSerperScrapeRequestAndCredits(t *testing.T) {
	var rec recorder
	srv := httptest.NewServer(rec.handle(http.StatusOK, `{
		"text": "plain text",
		"markdown": "# heading",
		"metadata": {"title": "Page title", "description": "desc"},
		"credits": 2}`))
	defer srv.Close()

	c := newSerperClient(srv.URL, "k", nil, 0)
	page, err := c.Fetch(context.Background(), FetchRequest{URL: "https://a.example/1", Markdown: true})
	if err != nil {
		t.Fatalf("Fetch: %v", err)
	}
	reqs := rec.snapshot()
	if len(reqs) != 1 {
		t.Fatalf("got %d requests, want 1", len(reqs))
	}
	req := reqs[0]
	if req.Method != http.MethodPost || req.Path != "/scrape" {
		t.Errorf("method/path = %s %s, want POST /scrape", req.Method, req.Path)
	}
	if req.Body["url"] != "https://a.example/1" {
		t.Errorf("body url = %v", req.Body["url"])
	}
	if req.Body["includeMarkdown"] != true {
		t.Errorf("body includeMarkdown = %v, want true", req.Body["includeMarkdown"])
	}
	if page.Title != "Page title" || page.Description != "desc" {
		t.Errorf("page metadata = %q/%q", page.Title, page.Description)
	}
	if page.Text != "plain text" || page.Markdown != "# heading" {
		t.Errorf("page content = %q/%q", page.Text, page.Markdown)
	}
	if page.Credits != 2 {
		t.Errorf("credits = %d, want 2", page.Credits)
	}

	// Without the markdown option the flag must be absent.
	if _, err := c.Fetch(context.Background(), FetchRequest{URL: "https://a.example/1"}); err != nil {
		t.Fatalf("Fetch plain: %v", err)
	}
	if reqs = rec.snapshot(); len(reqs) != 2 {
		t.Fatalf("got %d requests, want 2", len(reqs))
	}
	if _, ok := reqs[1].Body["includeMarkdown"]; ok {
		t.Errorf("includeMarkdown sent without markdown option: %v", reqs[1].Body)
	}
}

func TestSerperTargetNotFoundPropagates(t *testing.T) {
	var rec recorder
	srv := httptest.NewServer(rec.handle(http.StatusNotFound,
		`{"message":"Page not found.","statusCode":404}`))
	defer srv.Close()

	c := newSerperClient(srv.URL, "k", nil, 0)
	_, _, err := c.Search(context.Background(), SearchRequest{Query: "g"})
	assertSerperError(t, err, http.StatusNotFound, "Page not found.")
	_, err = c.Fetch(context.Background(), FetchRequest{URL: "https://a.example/missing"})
	assertSerperError(t, err, http.StatusNotFound, "Page not found.")
	if reqs := rec.snapshot(); len(reqs) != 2 {
		t.Fatalf("got %d requests, want 2 (404 must not be retried)", len(reqs))
	}
}

func TestSerperRetryOn429ThenSuccess(t *testing.T) {
	sleeps := stubSleeper(t)
	var (
		mu       sync.Mutex
		attempts int
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		attempts++
		n := attempts
		mu.Unlock()
		if n == 1 {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = io.WriteString(w, `{"message":"rate limited"}`)
			return
		}
		_, _ = io.WriteString(w, `{"organic":[{"title":"T","link":"https://a.example/1","snippet":"S"}]}`)
	}))
	defer srv.Close()

	c := newSerperClient(srv.URL, "k", nil, 1)
	sources, _, err := c.Search(context.Background(), SearchRequest{Query: "g"})
	if err != nil {
		t.Fatalf("Search after retry: %v", err)
	}
	if len(sources) != 1 {
		t.Fatalf("sources = %+v", sources)
	}
	mu.Lock()
	n := attempts
	mu.Unlock()
	if n != 2 {
		t.Fatalf("attempts = %d, want 2 (one retry)", n)
	}
	if len(*sleeps) != 1 || (*sleeps)[0] <= 0 {
		t.Fatalf("sleeps = %v, want one positive backoff", *sleeps)
	}
}

func TestSerperRetryExhaustedOn429(t *testing.T) {
	sleeps := stubSleeper(t)
	var rec recorder
	srv := httptest.NewServer(rec.handle(http.StatusTooManyRequests, `{"message":"rate limited"}`))
	defer srv.Close()

	c := newSerperClient(srv.URL, "k", nil, 1)
	_, _, err := c.Search(context.Background(), SearchRequest{Query: "g"})
	assertSerperError(t, err, http.StatusTooManyRequests, "rate limited")
	if reqs := rec.snapshot(); len(reqs) != 2 {
		t.Fatalf("got %d requests, want 2 (1 retry, then exhausted)", len(reqs))
	}
	if len(*sleeps) != 1 {
		t.Fatalf("sleeps = %v, want 1", *sleeps)
	}
}

func TestSerperRetryCapAtMax(t *testing.T) {
	stubSleeper(t)
	var rec recorder
	srv := httptest.NewServer(rec.handle(http.StatusTooManyRequests, `{}`))
	defer srv.Close()

	c := newSerperClient(srv.URL, "k", nil, 99)
	_, _, err := c.Search(context.Background(), SearchRequest{Query: "g"})
	assertSerperError(t, err, http.StatusTooManyRequests, "")
	if reqs := rec.snapshot(); len(reqs) != maxRetries+1 {
		t.Fatalf("got %d requests, want %d (retries capped)", len(reqs), maxRetries+1)
	}
}

func TestSerperNon200NotRetried(t *testing.T) {
	stubSleeper(t)
	var rec recorder
	srv := httptest.NewServer(rec.handle(http.StatusInternalServerError, `{"message":"boom"}`))
	defer srv.Close()

	c := newSerperClient(srv.URL, "k", nil, 3)
	_, _, err := c.Search(context.Background(), SearchRequest{Query: "g"})
	assertSerperError(t, err, http.StatusInternalServerError, "boom")
	if reqs := rec.snapshot(); len(reqs) != 1 {
		t.Fatalf("got %d requests, want 1 (only 429 is retried)", len(reqs))
	}
}

func TestSerperTimeout(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-release
	}))
	defer srv.Close()
	defer close(release)

	c := newSerperClient(srv.URL, "k", &http.Client{Timeout: 25 * time.Millisecond}, 0)
	_, _, err := c.Search(context.Background(), SearchRequest{Query: "g"})
	assertSerperError(t, err, 0, "Client.Timeout exceeded")
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("err = %v, want DeadlineExceeded in chain", err)
	}
}

// assertSerperError checks that err is a *SerperError with the given
// status and a body containing excerpt (skipped when excerpt is "").
func assertSerperError(t *testing.T, err error, status int, excerpt string) {
	t.Helper()
	if err == nil {
		t.Fatalf("err = nil, want *SerperError")
	}
	var se *SerperError
	if !errors.As(err, &se) {
		t.Fatalf("err = %T (%v), want *SerperError", err, err)
	}
	if se.Status != status {
		t.Fatalf("status = %d, want %d (err: %v)", se.Status, status, err)
	}
	if excerpt != "" && !strings.Contains(se.Body, excerpt) {
		t.Fatalf("body = %q, want it to contain %q", se.Body, excerpt)
	}
}
