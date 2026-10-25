package search

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strings"

	"github.com/smhanov/llmhub"

	"github.com/smhanov/agentloop"
)

// Tool and budget defaults for the preset.
const (
	// searchToolName and fetchToolName are the model-visible tool names.
	searchToolName = "web_search"
	fetchToolName  = "fetch_page"
	// defaultMaxToolCalls raises the core budget: effective research
	// typically alternates search, fetch, then answer.
	defaultMaxToolCalls = 8
	// defaultMaxResults is the Serper "num" value; Serper caps at 10.
	defaultMaxResults = 10
	// defaultFetchLimit is the per-fetch page-text character budget:
	// 10000 head + elision marker + 10000 tail.
	defaultFetchLimit = 20000
)

// Option composes core and preset options in a single variadic list.
type Option = agentloop.Option

// Setting keys this preset decodes from the option list via
// agentloop.Settings. They are unexported implementation details; callers
// use the With* constructors below.
const (
	settingSerperKey  = "search.serperKey"
	settingBaseURL    = "search.baseURL"
	settingHTTPClient = "search.httpClient"
	settingSearcher   = "search.searcher"
	settingFetcher    = "search.fetcher"
	settingRetries    = "search.retries"
	settingMaxResults = "search.maxResults"
	settingFetchLimit = "search.fetchLimit"
	settingNoFetch    = "search.noFetch"
)

// WithSerperKey sets the Serper API key, overriding SERPER_API_KEY. With
// neither set, the tools return graceful "not configured" results and the
// agent answers ungrounded.
func WithSerperKey(key string) Option {
	return agentloop.WithSetting(settingSerperKey, key)
}

// WithSerperBaseURL overrides the Serper API base URL; the default is
// https://google.serper.dev. It is a test hook.
func WithSerperBaseURL(url string) Option {
	return agentloop.WithSetting(settingBaseURL, url)
}

// WithSerperHTTPClient overrides the http.Client used for every Serper
// call; the default has a 10s timeout.
func WithSerperHTTPClient(c *http.Client) Option {
	return agentloop.WithSetting(settingHTTPClient, c)
}

// WithSearcher replaces the search backend. The preset depends only on the
// interface, so caching wrappers, other providers, and test doubles drop
// in; a custom Searcher needs no Serper key.
func WithSearcher(s Searcher) Option {
	return agentloop.WithSetting(settingSearcher, s)
}

// WithFetcher replaces the fetch backend.
func WithFetcher(f Fetcher) Option {
	return agentloop.WithSetting(settingFetcher, f)
}

// WithSerperRetries sets how many times a 429 is retried transparently
// with jittered exponential backoff before the model sees a failure; the
// default is 1 and the maximum is 3. 0 disables retries; negative values
// keep the default.
func WithSerperRetries(n int) Option {
	return agentloop.WithSetting(settingRetries, n)
}

// WithMaxResults sets the Serper "num" value; the default is 10. Values
// outside Serper's 1–10 range are clamped.
func WithMaxResults(n int) Option {
	return agentloop.WithSetting(settingMaxResults, n)
}

// WithFetchLimit caps the page text returned per fetch; the default is
// 20000 chars applied as head+tail truncation (first half, elision marker,
// last half). 0 or less means unlimited.
func WithFetchLimit(n int) Option {
	return agentloop.WithSetting(settingFetchLimit, n)
}

// WithNoFetch drops the fetch_page tool, making a search-only agent.
func WithNoFetch() Option {
	return agentloop.WithSetting(settingNoFetch, true)
}

// presetSettings is the resolved preset configuration.
type presetSettings struct {
	serperKey  string
	baseURL    string
	httpClient *http.Client
	searcher   Searcher
	fetcher    Fetcher
	retries    int
	maxResults int
	fetchLimit int
	noFetch    bool
}

// Handler errors delivered to the model as in-band {"error": ...}
// observations; the run itself continues.
var (
	errInvalidArguments    = errors.New("invalid arguments")
	errSearchNotConfigured = errors.New("web search is not configured")
	errFetchNotConfigured  = errors.New("page fetching is not configured")
)

// New returns a search agent: the core loop plus the web_search and
// fetch_page tools and the default research-assistant prompt. Serper key
// resolution is WithSerperKey, then the SERPER_API_KEY environment
// variable, then none — with no key the tools return graceful "not
// configured" results and the agent still works ungrounded. Core options
// passed in opts (budgets, hooks, skills, …) apply after the preset
// defaults, so later options win.
func New(llm *llmhub.Client, opts ...Option) (*Agent, error) {
	st := agentloop.Settings(opts...)
	cfg := presetSettings{
		baseURL:    defaultBaseURL,
		retries:    defaultRetries,
		maxResults: defaultMaxResults,
		fetchLimit: defaultFetchLimit,
	}
	if v, ok := st[settingSerperKey].(string); ok {
		cfg.serperKey = v
	}
	if v, ok := st[settingBaseURL].(string); ok && v != "" {
		cfg.baseURL = v
	}
	if v, ok := st[settingHTTPClient].(*http.Client); ok {
		cfg.httpClient = v
	}
	if v, ok := st[settingSearcher].(Searcher); ok {
		cfg.searcher = v
	}
	if v, ok := st[settingFetcher].(Fetcher); ok {
		cfg.fetcher = v
	}
	if v, ok := st[settingRetries].(int); ok {
		cfg.retries = clampRetries(v, cfg.retries)
	}
	if v, ok := st[settingMaxResults].(int); ok {
		cfg.maxResults = clampMaxResults(v)
	}
	if v, ok := st[settingFetchLimit].(int); ok {
		cfg.fetchLimit = v
	}
	if v, ok := st[settingNoFetch].(bool); ok {
		cfg.noFetch = v
	}

	// Key resolution: explicit option, then environment, then none. A
	// custom Searcher/Fetcher needs no key, so only the built-in Serper
	// client path depends on one.
	key := cfg.serperKey
	if key == "" {
		key = os.Getenv("SERPER_API_KEY")
	}
	searcherCustom, fetcherCustom := cfg.searcher != nil, cfg.fetcher != nil
	if !searcherCustom || !fetcherCustom {
		client := newSerperClient(cfg.baseURL, key, cfg.httpClient, cfg.retries)
		if !searcherCustom {
			cfg.searcher = client
		}
		if !fetcherCustom {
			cfg.fetcher = client
		}
	}
	p := &preset{
		searcher:    cfg.searcher,
		fetcher:     cfg.fetcher,
		maxResults:  cfg.maxResults,
		fetchLimit:  cfg.fetchLimit,
		searchReady: searcherCustom || key != "",
		fetchReady:  fetcherCustom || key != "",
	}

	tools := []agentloop.Tool{{
		Def: webSearchDef(),
		Run: p.handleSearch,
	}}
	if !cfg.noFetch {
		tools = append(tools, agentloop.Tool{
			Def: fetchPageDef(),
			Run: p.handleFetch,
		})
	}

	coreOpts := append([]Option{
		agentloop.WithSystemPrompt(defaultSystemPrompt),
		agentloop.WithTools(tools...),
		agentloop.WithMaxToolCalls(defaultMaxToolCalls),
	}, opts...)
	core, err := agentloop.New(llm, coreOpts...)
	if err != nil {
		return nil, err
	}
	return &Agent{agent: core}, nil
}

// clampRetries clamps a WithSerperRetries value: 0 disables, negative
// keeps def, and anything above the maximum is capped.
func clampRetries(n, def int) int {
	switch {
	case n == 0:
		return 0
	case n < 0:
		return def
	case n > maxRetries:
		return maxRetries
	default:
		return n
	}
}

// clampMaxResults clamps a WithMaxResults value into Serper's 1–10 range.
func clampMaxResults(n int) int {
	if n < 1 {
		return 1
	}
	if n > 10 {
		return 10
	}
	return n
}

// preset holds the resolved backends the tool handlers close over. It is
// immutable after construction and safe for concurrent use.
type preset struct {
	searcher    Searcher
	fetcher     Fetcher
	maxResults  int
	fetchLimit  int
	searchReady bool
	fetchReady  bool
}

// searchPayload is the trimmed, token-friendly result payload handed to
// the model and parsed back out of the trace by Sources.
type searchPayload struct {
	Results []resultPayload `json:"results"`
	Related []string        `json:"related,omitempty"`
}

// resultPayload is one entry of searchPayload.Results. Publisher is set
// for news results; Date only when the backend provided one.
type resultPayload struct {
	Title     string `json:"title"`
	Link      string `json:"link"`
	Snippet   string `json:"snippet"`
	Date      string `json:"date,omitempty"`
	Publisher string `json:"publisher,omitempty"`
}

// fetchPayload is the compact fetch result handed to the model.
type fetchPayload struct {
	URL   string `json:"url"`
	Title string `json:"title"`
	Text  string `json:"text"`
}

// handleSearch decodes {"q","num","type"}, calls the Searcher (type "news"
// routes to the news endpoint), and returns the trimmed payload.
func (p *preset) handleSearch(ctx context.Context, args json.RawMessage) (string, error) {
	if !p.searchReady {
		return "", errSearchNotConfigured
	}
	var a struct {
		Q    string `json:"q"`
		Num  int    `json:"num"`
		Type string `json:"type"`
	}
	if err := json.Unmarshal(args, &a); err != nil || strings.TrimSpace(a.Q) == "" {
		return "", errInvalidArguments
	}
	req := SearchRequest{Query: a.Q, Num: p.maxResults}
	switch a.Type {
	case "", "search":
	case "news":
		req.Type = SearchNews
	default:
		return "", errInvalidArguments
	}
	if a.Num >= 1 {
		req.Num = min(a.Num, p.maxResults)
	}
	sources, related, err := p.searcher.Search(ctx, req)
	if err != nil {
		return "", backendFailure("search", err)
	}
	payload := searchPayload{Results: make([]resultPayload, 0, len(sources)), Related: related}
	for _, s := range sources {
		payload.Results = append(payload.Results, resultPayload{
			Title: s.Title, Link: s.URL, Snippet: s.Snippet, Date: s.Date, Publisher: s.Publisher,
		})
	}
	b, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("search: encode results: %w", err)
	}
	return string(b), nil
}

// handleFetch validates the absolute http(s) URL, calls the Fetcher,
// truncates the text head+tail at the configured limit, and returns
// compact JSON. Target-site failures surface as
// {"error": "fetch failed: <status>"} so the model can try another result.
func (p *preset) handleFetch(ctx context.Context, args json.RawMessage) (string, error) {
	if !p.fetchReady {
		return "", errFetchNotConfigured
	}
	var a struct {
		URL      string `json:"url"`
		Markdown bool   `json:"markdown"`
	}
	if err := json.Unmarshal(args, &a); err != nil {
		return "", errInvalidArguments
	}
	if !validPageURL(a.URL) {
		return "", errInvalidArguments
	}
	page, err := p.fetcher.Fetch(ctx, FetchRequest{URL: a.URL, Markdown: a.Markdown})
	if err != nil {
		return "", backendFailure("fetch", err)
	}
	text := page.Text
	if a.Markdown && page.Markdown != "" {
		text = page.Markdown
	}
	b, err := json.Marshal(fetchPayload{
		URL:   a.URL,
		Title: page.Title,
		Text:  truncateHeadTail(text, p.fetchLimit),
	})
	if err != nil {
		return "", fmt.Errorf("fetch: encode page: %w", err)
	}
	return string(b), nil
}

// validPageURL reports whether s is an absolute http(s) URL.
func validPageURL(s string) bool {
	u, err := url.Parse(s)
	return err == nil && u.Host != "" && (u.Scheme == "http" || u.Scheme == "https")
}

// backendFailure converts a Serper failure into the model-facing message
// shape "search failed: <status>" / "fetch failed: <status>", wrapping the
// original error so callers can still inspect it.
func backendFailure(action string, err error) error {
	var se *SerperError
	if errors.As(err, &se) && se.Status != 0 {
		return fmt.Errorf("%s failed: %d", action, se.Status)
	}
	return fmt.Errorf("%s failed: %w", action, err)
}

// truncateHeadTail limits s to limit runes using head+tail truncation: the
// first half, an elision marker counting the middle, and the last half —
// never naive head-only, which loses conclusions. limit <= 0 (or text
// already within the limit) returns s unchanged.
func truncateHeadTail(s string, limit int) string {
	if limit <= 0 {
		return s
	}
	r := []rune(s)
	if len(r) <= limit {
		return s
	}
	half := limit / 2
	if half == 0 {
		return string(r[:limit])
	}
	elided := len(r) - 2*half
	return string(r[:half]) + fmt.Sprintf("<%d characters elided>", elided) + string(r[len(r)-half:])
}

// Agent is the search preset: an immutable wrapper around the core
// agentloop.Agent adding source extraction. Safe for concurrent use.
type Agent struct {
	agent *agentloop.Agent
}

// Ask runs the loop to completion for one query and returns the trace with
// Sources extracted. opts override the construction options for this call.
func (a *Agent) Ask(ctx context.Context, query string, opts ...Option) (*Answer, error) {
	core, err := a.agent.Ask(ctx, query, opts...)
	if err != nil {
		return nil, err
	}
	return &Answer{Answer: core, Sources: Sources(core)}, nil
}

// AskStream is Ask with streaming; events flow exactly as the core
// delivers them. Use Sources on the EventDone answer to extract sources
// from a streaming run.
func (a *Agent) AskStream(ctx context.Context, query string, opts ...Option) (<-chan agentloop.Event, error) {
	return a.agent.AskStream(ctx, query, opts...)
}

// Agent returns the wrapped core agent — the escape hatch to the full core
// API, e.g. for conversations.
func (a *Agent) Agent() *agentloop.Agent { return a.agent }

// Answer is a search run's result: the core trace plus the sources the
// model actually used, deduped by URL in first-seen order (citation [n]
// aligns with Sources[n-1]).
type Answer struct {
	*agentloop.Answer
	// Sources lists the web_search results from the trace, marking entries
	// the model also fetched in full, plus fetch-only pages appended last.
	Sources []Source
}

// Sources extracts sources from any core Answer's tool trace: every
// web_search result, deduped by URL in first-seen order, plus any page the
// model read in full through fetch_page whose URL no search returned —
// appended after the search-derived sources in fetch order, with Fetched
// set and no Snippet (the fetch payload carries none). Entries whose URL
// the model also passed to a successful fetch_page are marked Fetched,
// whenever the fetch happened relative to the search. Failed fetches
// (error observations) and unreadable payloads add nothing. It is a pure
// function over the trace — usable on search.Answer and directly on
// EventDone payloads from streaming runs.
func Sources(ans *agentloop.Answer) []Source {
	if ans == nil {
		return nil
	}
	var out []Source
	seen := make(map[string]bool)
	// Pass 1: search-derived sources, first-seen dedup.
	for _, tr := range ans.Tools {
		if tr.Name != searchToolName {
			continue
		}
		var payload searchPayload
		if json.Unmarshal([]byte(tr.Output), &payload) != nil {
			continue // error observation, not a result payload
		}
		for _, r := range payload.Results {
			if r.Link != "" {
				if seen[r.Link] {
					continue
				}
				seen[r.Link] = true
			}
			out = append(out, Source{
				Title: r.Title, URL: r.Link, Snippet: r.Snippet, Date: r.Date, Publisher: r.Publisher,
			})
		}
	}
	// Pass 2: fetch_page results, after every search-derived source is
	// known. A successful fetch of an already-present URL marks it Fetched;
	// the rest append as fetch-only sources in fetch order, so pages the
	// model actually read are never invisible to consumers.
	for _, tr := range ans.Tools {
		if tr.Name != fetchToolName {
			continue
		}
		var args struct {
			URL string `json:"url"`
		}
		if json.Unmarshal(tr.Args, &args) != nil || args.URL == "" || tr.Err != nil {
			continue
		}
		var obs struct {
			Error string `json:"error"`
		}
		if json.Unmarshal([]byte(tr.Output), &obs) == nil && obs.Error != "" {
			continue // the fetch failed; the page was never read
		}
		var payload fetchPayload
		if json.Unmarshal([]byte(tr.Output), &payload) != nil {
			continue // unreadable output; the page's identity is unknown
		}
		if seen[args.URL] {
			for i := range out {
				if out[i].URL == args.URL {
					out[i].Fetched = true
				}
			}
			continue
		}
		seen[args.URL] = true
		out = append(out, Source{Title: payload.Title, URL: args.URL, Fetched: true})
	}
	return out
}
