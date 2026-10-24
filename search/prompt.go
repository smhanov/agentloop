package search

import "github.com/smhanov/llmhub"

// defaultSystemPrompt is the search preset's default system prompt. It is
// domain-neutral: domain conventions travel with the query text or a
// caller-supplied WithSystemPrompt, never in this default.
const defaultSystemPrompt = "You are a research assistant with access to two tools backed by Google: " +
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

// webSearchDef declares the web_search tool for the model.
func webSearchDef() llmhub.Tool {
	return llmhub.NewTool("web_search",
		"Search Google via Serper for current information. Use this whenever the "+
			"answer depends on facts that may be recent, volatile, or outside your "+
			"training data. Use type=\"news\" for time-sensitive current events; the "+
			"default general search otherwise. Formulate concise keyword queries; "+
			"call again with a revised query if the results are insufficient.",
		map[string]any{
			"type": "object",
			"properties": map[string]any{
				"q": map[string]any{"type": "string", "description": "the search query"},
				"num": map[string]any{"type": "integer", "minimum": 1, "maximum": 10,
					"description": "number of results (default 10)"},
				"type": map[string]any{"type": "string", "enum": []string{"search", "news"},
					"description": "\"search\" (default) or \"news\" for recent events"},
			},
			"required": []string{"q"},
		})
}

// fetchPageDef declares the fetch_page tool for the model.
func fetchPageDef() llmhub.Tool {
	return llmhub.NewTool("fetch_page",
		"Fetch the readable full text of a web page. Use when a search snippet is "+
			"insufficient and you need the details behind a specific result URL. "+
			"Prefer at most one or two fetches per question.",
		map[string]any{
			"type": "object",
			"properties": map[string]any{
				"url": map[string]any{"type": "string",
					"description": "absolute http(s) URL of the page"},
				"markdown": map[string]any{"type": "boolean",
					"description": "return markdown instead of plain text (default false)"},
			},
			"required": []string{"url"},
		})
}
