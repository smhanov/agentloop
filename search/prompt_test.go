package search

import (
	"reflect"
	"testing"
)

// The declarations and prompt below are locked to the specification
// verbatim; a drift here changes what every model sees.

func TestSystemPromptVerbatim(t *testing.T) {
	want := "You are a research assistant with access to two tools backed by Google: " +
		"`web_search` (Serper Google search) and `fetch_page` (full page text). " +
		"Decide for each request whether search is needed. When snippets are " +
		"insufficient to answer, call `fetch_page` on the most promising result " +
		"URL and read the full content — but fetch sparingly; snippets answer " +
		"most questions. When you search, cite the sources you actually used " +
		"inline as `[n]`, where `n` is the 1-based index of the source in the " +
		"order they were first returned to you. If the search results do not " +
		"answer the question, say so plainly rather than guessing. Provide the " +
		"final answer only when it is grounded in the search results or clearly " +
		"within your own knowledge; never fabricate URLs, statistics, or dates."
	if defaultSystemPrompt != want {
		t.Errorf("system prompt drifted:\n got: %s\nwant: %s", defaultSystemPrompt, want)
	}
}

func TestToolDeclarationsVerbatim(t *testing.T) {
	searchDef := webSearchDef()
	if searchDef.Name != "web_search" {
		t.Errorf("name = %q", searchDef.Name)
	}
	wantSearchDesc := "Search Google via Serper for current information. Use this whenever the " +
		"answer depends on facts that may be recent, volatile, or outside your " +
		"training data. Use type=\"news\" for time-sensitive current events; the " +
		"default general search otherwise. Formulate concise keyword queries; " +
		"call again with a revised query if the results are insufficient."
	if searchDef.Description != wantSearchDesc {
		t.Errorf("web_search description drifted:\n got: %s\nwant: %s", searchDef.Description, wantSearchDesc)
	}
	wantSearchParams := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"q": map[string]any{"type": "string", "description": "the search query"},
			"num": map[string]any{"type": "integer", "minimum": 1, "maximum": 10,
				"description": "number of results (default 10)"},
			"type": map[string]any{"type": "string", "enum": []string{"search", "news"},
				"description": "\"search\" (default) or \"news\" for recent events"},
		},
		"required": []string{"q"},
	}
	if !reflect.DeepEqual(searchDef.Parameters, wantSearchParams) {
		t.Errorf("web_search parameters drifted:\n got: %#v\nwant: %#v", searchDef.Parameters, wantSearchParams)
	}

	fetchDef := fetchPageDef()
	if fetchDef.Name != "fetch_page" {
		t.Errorf("name = %q", fetchDef.Name)
	}
	wantFetchDesc := "Fetch the readable full text of a web page. Use when a search snippet is " +
		"insufficient and you need the details behind a specific result URL. " +
		"Prefer at most one or two fetches per question."
	if fetchDef.Description != wantFetchDesc {
		t.Errorf("fetch_page description drifted:\n got: %s\nwant: %s", fetchDef.Description, wantFetchDesc)
	}
	wantFetchParams := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"url": map[string]any{"type": "string",
				"description": "absolute http(s) URL of the page"},
			"markdown": map[string]any{"type": "boolean",
				"description": "return markdown instead of plain text (default false)"},
		},
		"required": []string{"url"},
	}
	if !reflect.DeepEqual(fetchDef.Parameters, wantFetchParams) {
		t.Errorf("fetch_page parameters drifted:\n got: %#v\nwant: %#v", fetchDef.Parameters, wantFetchParams)
	}
}
