// Package search provides the search preset: an agentloop.Agent wired with
// web_search and fetch_page tools backed by the Serper API, plus source
// extraction from the tool trace. The preset is domain-neutral; callers
// layer domain behavior through query text or a custom system prompt.
package search

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"strings"
	"time"
)

// SearchType selects which Serper endpoint a search hits.
type SearchType int

const (
	// SearchGeneral queries the /search endpoint (the default zero value).
	SearchGeneral SearchType = iota
	// SearchNews queries the /news endpoint for time-sensitive results.
	SearchNews
)

// SearchRequest describes one web search.
type SearchRequest struct {
	// Query is the search query text.
	Query string
	// Num is the maximum number of results; 0 means the backend default.
	Num int
	// Type selects the endpoint: SearchGeneral or SearchNews.
	Type SearchType
}

// Searcher retrieves web search results. serperClient implements it;
// WithSearcher injects alternatives (caching wrappers, other backends,
// test doubles).
type Searcher interface {
	// Search returns the result sources and up to five related queries.
	Search(ctx context.Context, req SearchRequest) ([]Source, []string, error)
}

// FetchRequest describes one page fetch.
type FetchRequest struct {
	// URL is the absolute http(s) URL of the page.
	URL string
	// Markdown requests markdown in addition to plain text.
	Markdown bool
}

// Page is the readable content of one fetched page.
type Page struct {
	// URL is the fetched URL.
	URL string
	// Title is the page title (Serper metadata.title).
	Title string
	// Description is the page description (metadata.description).
	Description string
	// Text is the extracted plain text (boilerplate stripped).
	Text string
	// Markdown is the extracted markdown, when requested.
	Markdown string
	// Credits is the number of API credits Serper charged.
	Credits int
}

// Fetcher retrieves the readable content of a web page. serperClient
// implements it; WithFetcher injects alternatives.
type Fetcher interface {
	// Fetch returns the page content.
	Fetch(ctx context.Context, req FetchRequest) (*Page, error)
}

// Source is one search result as returned to the model and as extracted
// from a run's trace by Sources.
type Source struct {
	// Title is the result title.
	Title string
	// URL is the result link.
	URL string
	// Snippet is the result excerpt.
	Snippet string
	// Date is Serper's relative date, when provided (always set for news
	// results).
	Date string
	// Publisher is the news source name, when Serper provides it.
	Publisher string
	// Fetched reports whether the model fetched this page's full content
	// during the run (set only by Sources, never by the backend).
	Fetched bool
}

// SerperError reports a failed Serper API call: a non-200 HTTP status (or,
// for transport failures, Status 0) plus a body excerpt.
type SerperError struct {
	// Status is the HTTP status code, or 0 for transport failures.
	Status int
	// Body is an excerpt of the response body (the transport error text
	// when Status is 0).
	Body string

	err error // underlying transport error, if any
}

// Error implements error with the status and body excerpt.
func (e *SerperError) Error() string {
	if e.Status == 0 {
		return fmt.Sprintf("serper: request failed: %s", e.Body)
	}
	return fmt.Sprintf("serper: status %d: %s", e.Status, e.Body)
}

// Unwrap exposes the underlying transport error, when there is one, so
// errors.Is(err, context.DeadlineExceeded) and friends keep working.
func (e *SerperError) Unwrap() error { return e.err }

// serper API constants and retry tuning. The backoff is deliberately small;
// tests replace sleeper entirely.
const (
	defaultBaseURL     = "https://google.serper.dev"
	defaultHTTPTimeout = 10 * time.Second
	defaultRetries     = 1
	maxRetries         = 3
	retryBaseDelay     = 250 * time.Millisecond
	retryMaxDelay      = 2 * time.Second
	// errorBodyLimit caps how much of an error response body is kept.
	errorBodyLimit = 4096
	// relatedLimit is the number of related queries handed to the model.
	relatedLimit = 5
)

// serperClient implements Searcher and Fetcher over one http.Client. It is
// immutable after construction and safe for concurrent use.
type serperClient struct {
	baseURL string
	apiKey  string
	http    *http.Client
	retries int // retries after the initial attempt; 0 disables
}

// newSerperClient builds the shared client for search, news, and scrape.
func newSerperClient(baseURL, apiKey string, hc *http.Client, retries int) *serperClient {
	if hc == nil {
		hc = &http.Client{Timeout: defaultHTTPTimeout}
	}
	if retries < 0 {
		retries = 0
	}
	if retries > maxRetries {
		retries = maxRetries
	}
	return &serperClient{baseURL: strings.TrimRight(baseURL, "/"), apiKey: apiKey, http: hc, retries: retries}
}

// sleeper pauses for d or until ctx is done, returning ctx's error on
// cancellation. It is a variable so tests can make backoff instantaneous
// and observe the requested durations.
var sleeper = func(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// retryBackoff computes the jittered exponential backoff before the given
// zero-based retry: half to full of base*2^attempt, capped at
// retryMaxDelay.
func retryBackoff(attempt int) time.Duration {
	d := retryBaseDelay << attempt
	if d <= 0 || d > retryMaxDelay {
		d = retryMaxDelay
	}
	return d/2 + time.Duration(rand.Int63n(int64(d/2)+1))
}

// post issues one POST to path with a JSON body, retrying transparently on
// 429 up to c.retries times with jittered backoff. Any other failure —
// transport error or non-200 after retries — comes back as *SerperError.
func (c *serperClient) post(ctx context.Context, path string, body any) ([]byte, error) {
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("serper: encode request: %w", err)
	}
	var lastErr error
	for attempt := 0; attempt <= c.retries; attempt++ {
		if attempt > 0 {
			if err := sleeper(ctx, retryBackoff(attempt-1)); err != nil {
				return nil, err
			}
		}
		data, err := c.attempt(ctx, path, payload)
		if err == nil {
			return data, nil
		}
		lastErr = err
		var se *SerperError
		if !errors.As(err, &se) || se.Status != http.StatusTooManyRequests {
			return nil, err
		}
	}
	return nil, lastErr
}

// attempt performs one HTTP round trip and maps every failure to
// *SerperError.
func (c *serperClient) attempt(ctx context.Context, path string, payload []byte) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, bytes.NewReader(payload))
	if err != nil {
		return nil, &SerperError{Body: err.Error(), err: err}
	}
	req.Header.Set("X-API-KEY", c.apiKey)
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, &SerperError{Body: err.Error(), err: err}
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		excerpt, _ := io.ReadAll(io.LimitReader(resp.Body, errorBodyLimit))
		return nil, &SerperError{Status: resp.StatusCode, Body: strings.TrimSpace(string(excerpt))}
	}
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, &SerperError{Status: resp.StatusCode, Body: err.Error(), err: err}
	}
	return data, nil
}

// serperSearchResponse is the subset of Serper's search/news response the
// preset maps; everything else is dropped.
type serperSearchResponse struct {
	Organic []struct {
		Title   string `json:"title"`
		Link    string `json:"link"`
		Snippet string `json:"snippet"`
		Date    string `json:"date"`
	} `json:"organic"`
	News []struct {
		Title   string `json:"title"`
		Link    string `json:"link"`
		Snippet string `json:"snippet"`
		Date    string `json:"date"`
		Source  string `json:"source"`
	} `json:"news"`
	KnowledgeGraph struct {
		Title       string `json:"title"`
		Description string `json:"description"`
		WebsiteLink string `json:"websiteLink"`
	} `json:"knowledgeGraph"`
	RelatedSearches []struct {
		Query string `json:"query"`
	} `json:"relatedSearches"`
}

// serperScrapeResponse is the subset of Serper's scrape response the
// preset maps.
type serperScrapeResponse struct {
	Text     string `json:"text"`
	Markdown string `json:"markdown"`
	Metadata struct {
		Title       string `json:"title"`
		Description string `json:"description"`
	} `json:"metadata"`
	Credits int `json:"credits"`
}

// Search queries /search or /news (by req.Type) and maps the response per
// the preset's trimming rules: a knowledgeGraph pseudo-result prepended,
// organic or news results capped at req.Num, relatedSearches capped at
// five, peopleAlsoAsk dropped. An empty result set is valid, not an error.
func (c *serperClient) Search(ctx context.Context, req SearchRequest) ([]Source, []string, error) {
	path := "/search"
	if req.Type == SearchNews {
		path = "/news"
	}
	body := map[string]any{"q": req.Query}
	if req.Num > 0 {
		body["num"] = req.Num
	}
	data, err := c.post(ctx, path, body)
	if err != nil {
		return nil, nil, err
	}
	var raw serperSearchResponse
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, nil, fmt.Errorf("serper: decode search response: %w", err)
	}

	var sources []Source
	if kg := raw.KnowledgeGraph; kg.Title != "" || kg.Description != "" {
		sources = append(sources, Source{Title: kg.Title, URL: kg.WebsiteLink, Snippet: kg.Description})
	}
	added := 0
	add := func(s Source) {
		if req.Num > 0 && added >= req.Num {
			return
		}
		sources = append(sources, s)
		added++
	}
	for _, o := range raw.Organic {
		add(Source{Title: o.Title, URL: o.Link, Snippet: o.Snippet, Date: o.Date})
	}
	for _, n := range raw.News {
		add(Source{Title: n.Title, URL: n.Link, Snippet: n.Snippet, Date: n.Date, Publisher: n.Source})
	}
	related := make([]string, 0, relatedLimit)
	for _, r := range raw.RelatedSearches {
		if len(related) >= relatedLimit {
			break
		}
		if r.Query != "" {
			related = append(related, r.Query)
		}
	}
	return sources, related, nil
}

// Fetch posts to /scrape and maps text, markdown (when requested),
// metadata title/description, and the credits charged.
func (c *serperClient) Fetch(ctx context.Context, req FetchRequest) (*Page, error) {
	body := map[string]any{"url": req.URL}
	if req.Markdown {
		body["includeMarkdown"] = true
	}
	data, err := c.post(ctx, "/scrape", body)
	if err != nil {
		return nil, err
	}
	var raw serperScrapeResponse
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("serper: decode scrape response: %w", err)
	}
	return &Page{
		URL:         req.URL,
		Title:       raw.Metadata.Title,
		Description: raw.Metadata.Description,
		Text:        raw.Text,
		Markdown:    raw.Markdown,
		Credits:     raw.Credits,
	}, nil
}
