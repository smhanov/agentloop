# agentloop

<div align="center">

**Everything you need for building an agent.**

[![CI](https://github.com/smhanov/agentloop/actions/workflows/ci.yml/badge.svg)](https://github.com/smhanov/agentloop/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/smhanov/agentloop.svg)](https://pkg.go.dev/github.com/smhanov/agentloop)
[![Go Report Card](https://goreportcard.com/badge/github.com/smhanov/agentloop)](https://goreportcard.com/report/github.com/smhanov/agentloop)
[![Go](https://img.shields.io/badge/go-1.21%2B-00ADD8?logo=go&logoColor=white)](https://go.dev)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

agentloop is a Go library
that takes care of everything you don't want to think about right now —
the loop every agent needs (send a prompt to the model, execute the tool
calls it requests, feed the results back, repeat until it produces a
final answer), plus the budgets, hooks, skills, and tracing around it —
so you can just build your agent. Every step of the loop is observable
and configurable.

It uses its sister project
[`github.com/smhanov/llmhub`](https://github.com/smhanov/llmhub) so you
can plug into any LLM — OpenAI, Anthropic, Gemini, Ollama, xAI,
OpenRouter, Z.AI, or your own provider — without ever talking to a
vendor API directly.

Two pre-configured agents are included as reference examples, each
assembled entirely from the public core:

- a **search agent** that uses [serper](https://serper.dev) web search
  to answer multi-step queries, with inline citations
- a **coding agent**, based on
  [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent), that
  solves tasks by executing shell commands in a workspace

**Going deeper.** If you want the full picture of how agents work —
the tool-calling loop, budgets, streaming, skills, and how to build
them all from first principles — read the author's book,
[*Building AI Agents from Scratch*](https://stevehanov.ca/books/building-ai-agents-from-scratch.pdf).
It is the in-depth companion to this library: agentloop is the
distillation of the techniques it teaches, with the scaffolding
already written for you.

## Why agentloop?

- **The loop, done once and well.** llmhub gives you provider-agnostic LLM
  access including tool calling, but you still have to dispatch calls,
  append result messages, budget iterations, and aggregate usage.
  agentloop is that loop.
- **Hooks at every step.** Logging, tracing, streaming UIs, approval
  gates, cost caps, steering, context compression, and entire custom
  agents are all expressed with one `Hooks` struct — the presets
  themselves are built this way, which is the proof the surface is enough.
- **Safe by default.** Every run is bounded: tool-call budgets, inference
  ceilings, duplicate-call loop detection, stream watchdogs, and your
  `context.Context`. The loop can never spin forever.
- **Skills, not prompt stuffing.** Repeatable procedures live in skill
  folders, advertised in a tiny catalog and loaded on demand through a
  jailed read tool — with progressive disclosure from one catalog line to
  full bodies to linked files to executable scripts.
- **Transparent traces.** Every `Ask` returns the final text plus the
  complete trace: every tool execution with output and duration, the full
  message transcript, and token/cost usage.
- **Plain Go, tiny footprint.** Two small pure-Go dependencies, one
  static binary: agents compile with `CGO_ENABLED=0 go build` to any
  `GOOS`/`GOARCH` and run in about 7 MB of RAM — no Python environment,
  no Node sidecar, no container to babysit. Embed an agent in a CLI, a
  cron job, an existing Go service, or a Raspberry Pi; between model
  calls the loop itself is nearly free.

## Installation

```bash
go get github.com/smhanov/agentloop
```

Go 1.21 or newer. The library depends only on llmhub and the pure-Go
[`gopkg.in/yaml.v3`](https://pkg.go.dev/gopkg.in/yaml.v3) frontmatter
parser; the search preset also wants a
[serper.dev](https://serper.dev) API key at runtime (never at build time).

The full API reference — every exported type, option, and hook — lives
on [pkg.go.dev](https://pkg.go.dev/github.com/smhanov/agentloop).

## Quick Start

The search agent answers current-events questions grounded in Serper
results, citing the sources it used:

```go
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/smhanov/llmhub"
	_ "github.com/smhanov/llmhub/providers/openai"

	"github.com/smhanov/agentloop/search"
)

func main() {
	llm, _ := llmhub.New("openai", os.Getenv("OPENAI_API_KEY"),
		llmhub.WithModel("gpt-4o-mini"))

	agent, _ := search.New(llm,
		search.WithSerperKey(os.Getenv("SERPER_API_KEY")))

	ans, err := agent.Ask(context.Background(),
		"What Go version was released most recently, and when?")
	if err != nil {
		panic(err)
	}

	fmt.Println(ans.Text)
	for i, s := range ans.Sources {
		fmt.Printf("[%d] %s — %s\n", i+1, s.Title, s.URL)
	}
	fmt.Printf("%d tool calls, $%.4f\n", ans.ToolCalls, ans.Usage.Cost)
}
```

`ans.Text` carries inline `[n]` citations that line up with `ans.Sources`
(see [The search agent](#the-search-agent)).

## Command line

The repository ships `examples/ask`, a general-purpose runner for all
three agents (search, bash, and the bare core). Three commands cover the
common cases — using the Z.AI coding endpoint from the local key file as
the example provider:

```bash
# Search agent (the default): grounded answers with sources
export ZAI_API_KEY=… SERPER_API_KEY=…
go run ./examples/ask \
  -provider zai -model glm-5.3-flash \
  -base-url https://api.z.ai/api/coding/paas/v4 \
  -serper-key "$SERPER_API_KEY" \
  -prompt "What Go version was released most recently, and when?" -stream
```

```bash
# Bash coding agent: solves a task by executing commands in a workspace
go run ./examples/ask -provider zai -model glm-5.3-flash \
  -base-url https://api.z.ai/api/coding/paas/v4 \
  -agent bash -cwd /tmp/workspace -step-limit 20 -cost-limit 1.5 \
  -prompt "Add a README.md describing this project." -stream -v
```

```bash
# Chat mode: one conversation, one prompt per stdin line (blank line exits)
go run ./examples/ask -provider openai -model gpt-4o-mini -chat -cache-key demo
```

The main flags (run `go run ./examples/ask -h` for the full list):

| Flag | Description |
| --- | --- |
| `-provider` | llmhub provider name (required); the key comes from `-api-key` or the provider's standard environment variable (`ZAI_API_KEY`, `OPENAI_API_KEY`, …) |
| `-model` | model identifier |
| `-base-url` | provider base URL override |
| `-agent` | `search` (default), `bash`, or `plain` (the bare core loop, no tools) |
| `-prompt` / `-prompt-file` | prompt or task text, inline or from a file |
| `-chat` | interactive multi-turn mode over one `Conversation` |
| `-cache-key` | llmhub prompt-cache key — recommended with `-chat` |
| `-serper-key` | Serper key (or `SERPER_API_KEY`); search agent only |
| `-system` | replace the system prompt |
| `-skills` | comma-separated skill directories to load |
| `-cwd`, `-env` | bash agent: working directory and extra `KEY=VALUE` entries (repeatable) |
| `-step-limit`, `-cost-limit` | bash agent limits |
| `-confirm` | bash agent: prompt before every command (default: unattended) |
| `-tool-protocol` | bash agent: native tool calling instead of the text protocol |
| `-stream` | stream answer deltas as they arrive; reasoning dimmed on stderr; tool activity inline |
| `-v` | print the event trace: tool calls with durations, steps, timing |
| `-json` | print the final answer as JSON (text, sources or submission, usage, trace) |
| `-max-tool-calls`, `-max-steps` | budget overrides |

Exit codes: `0` success (any bash exit status counts as success), `1`
error, `2` budget exhausted.

## Example output

A real transcript of the first command above (answer on stdout, `-v`
trace on stderr):

```text
$ go run ./examples/ask -provider zai -model glm-5.3-flash \
    -base-url https://api.z.ai/api/coding/paas/v4 \
    -serper-key "$SERPER_API_KEY" \
    -prompt "What Go version was released most recently, and when?" -v

The most recently released version of Go is **Go 1.27.3**, released on
**October 19, 2026** [1].

Some additional context from the official release history:

- **Go 1.27.0** (the current major release) came out on **August 19,
  2026**, followed by minor revisions 1.27.1 (Sept 1, 2026), 1.27.2
  (Oct 8, 2026), and 1.27.3 (Oct 19, 2026) [1]. The latest patch, 1.27.3,
  was primarily a security fix for the `crypto/x509` package [1].
- The previous major release, **Go 1.26.0**, was released on **February
  10, 2026** [1][2], and its branch also received an update the same day
  (Go 1.26.10, released October 19, 2026) [1].
- Both Go 1.27 and Go 1.26 are the currently supported versions; the next
  major release, Go 1.28, is expected in February 2027 [1][3].

Sources:
1. [Release History – The Go Programming Language (go.dev)](https://go.dev/doc/devel/release)
2. [Go 1.26 Introduces Two Language Changes – Phoronix](https://www.phoronix.com/news/Go-1.26-Released)
3. [Next Release Notes Draft (tip.golang.org)](https://tip.golang.org/doc/next)
sources:
  [1] A new pkg.go.dev API for Go https://opensource.googleblog.com/2026/06/a-new-pkggodev-api-for-go.html
  [2] Should You Try pkg.go.dev API? A Practical AI Launch Review https://kingy.ai/news/pkg-go-dev-api-guide/
  [3] Why Go is an Ideal Language for AI-Assisted Software Engineering https://developers.googleblog.com/why-go-is-an-ideal-language-for-ai-assisted-software-engineering/
  [4] Retro Rail Go! Dev Removes AI Assets & Rebrands …
  … 16 more sources
trace: tool web_search ok (1.163450694s)
trace: tool web_search ok (1.372012303s)
trace: tool fetch_page ok (355.807307ms)
trace: steps=3 tool calls=3 status=completed ttft=0s total=26.823753714s
```

The model formatted its own `Sources:` list from the search results it
read (numbered by the order *it* received them); the `sources:` block is
the CLI's print-out of `search.Sources` from the tool trace — deduped by
URL across all searches in first-seen order. With a single `web_search`
call the two numberings are identical.

## Where to go next

- [API reference](https://pkg.go.dev/github.com/smhanov/agentloop) —
  godoc for every exported symbol
- [Building AI Agents from Scratch](https://stevehanov.ca/books/building-ai-agents-from-scratch.pdf)
  — the author's book: an in-depth, from-first-principles guide to
  writing agents, and the ideas behind this library
- [Building an agent](#building-an-agent) — the core API and every option
- [Writing tools](#writing-tools) — plain and stateful tools, error handling
- [Hooks](#hooks) — the full hook and event reference
- [Budgets and loop safety](#budgets-and-loop-safety)
- [Streaming](#streaming) — `AskStream`, deltas, watchdog
- [Conversations](#conversations) — multi-turn sessions and prompt caching
- [Skills](#skills) — folder layout, progressive disclosure, authoring
- [The search agent](#the-search-agent) — complete usage guide
- [The bash agent](#the-bash-agent) — mini-swe-agent-compatible coding agent
- [Writing your own agent](#writing-your-own-agent) — recipes
- [Testing your agent](#testing-your-agent) — scripted fake providers
- [Troubleshooting](#troubleshooting)

---

## Building an agent

The core package is `github.com/smhanov/agentloop`. An `Agent` wraps an
llmhub client plus a configuration; it is immutable after construction and
safe for concurrent use.

```go
ag := agentloop.MustNew(llm,
    agentloop.WithSystemPrompt("You keep exact time."),
    agentloop.WithTools(clock),
)

ans, err := ag.Ask(ctx, "What time is it in UTC right now?")
```

`New` returns an error instead of panicking (wrapping `ErrNoLLMClient`
for a nil client); `MustNew` is the same for examples and tests. `Ask`
runs the loop to completion for one query and returns `*Answer` — text
plus the full trace. `AskStream` is the streaming form (see
[Streaming](#streaming)). `Tools` returns the registered tool set, which
presets use to merge their tools with yours. Every option works both at
construction and per call (`ag.Ask(ctx, q, agentloop.WithMaxToolCalls(2))`);
per-call options apply after the constructor's, so the later one wins.

| Option | Default | Effect |
| --- | --- | --- |
| `WithSystemPrompt(p)` | none | Prepends a system message to every `Ask`. Byte-stable for the agent's lifetime — dynamic facts belong in the query text. |
| `WithTools(t ...Tool)` | none | Replaces the registered tool set. Keep it small: schemas are sent to the model on every inference. |
| `WithMaxToolCalls(n)` | 5 (min 1) | Tool executions allowed per `Ask`. Once spent, the loop forces a final answer (see [Budgets](#budgets-and-loop-safety)). |
| `WithMaxSteps(n)` | `2*MaxToolCalls+2` | Hard ceiling on model inferences per `Ask`; exceeding it fails with `ErrBudgetExhausted`. |
| `WithDupCallLimit(n)` | 3 | Consecutive identical tool calls with identical output tolerated before loop detection intervenes; any differing call **or output** resets the streak. 0 disables. |
| `WithParallelTools(n)` | 3 | Max concurrent parallel-eligible executions per step; 1 forces strictly sequential. |
| `WithStreamWatchdog(d)` | 90s | Aborts `AskStream` when no chunk arrives for `d` (`ErrStreamStale`); 0 disables. |
| `WithHooks(h)` | none | Hook callbacks — the extension surface ([Hooks](#hooks)). |
| `WithSkills(dirs ...)` | none | Enables skills: catalog in the prompt + `skill_read` tool. Construction-time only. |
| `WithPinnedSkills(names ...)` | none | Inlines up to two skill bodies into the system prompt. Construction-time only. |
| `WithToolChoice(c)` | `llmhub.AutoToolChoice()` | Default tool-choice policy; the loop forces `llmhub.NoToolChoice()` itself when the tool budget is spent. |
| `WithGenerateOptions(opts ...)` | none | llmhub options forwarded to every LLM call (temperature, cache key, …). Repeated uses append. |
| `WithSetting(k, v)` / `Settings(opts...)` | — | Preset-author channel: attach named values to an option list and read them back; the core itself ignores settings. |

## Writing tools

A tool is an llmhub definition plus a Go handler. The handler receives the
model's raw JSON arguments and returns the payload inserted verbatim as
the tool result:

```go
clock := agentloop.Tool{
    Def: llmhub.NewTool("clock", "Current time in RFC 3339", map[string]any{
        "type": "object", "properties": map[string]any{},
    }),
    Run: func(ctx context.Context, args json.RawMessage) (string, error) {
        return time.Now().UTC().Format(time.RFC3339), nil
    },
}
```

The `Sequential` flag controls execution ordering. The zero value means
the tool is eligible for parallel batching (bounded by `WithParallelTools`);
set it for stateful tools that must never overlap:

```go
session := &shellSession{dir: "/tmp/workspace"} // your type: holds cwd, env, history

sh := agentloop.Tool{
    Def: llmhub.NewTool("shell", "Run one command in the workspace", map[string]any{
        "type": "object",
        "properties": map[string]any{
            "command": map[string]any{"type": "string"},
        },
        "required": []string{"command"},
    }),
    Run: func(ctx context.Context, args json.RawMessage) (string, error) {
        return session.Run(ctx, string(args)) // reads and mutates session state
    },
    Sequential: true, // stateful: strict call order, never overlaps anything
}
```

Error handling is in-band repair: a handler error does **not** abort the
run — the model receives `{"error": "<message>"}` as the tool result and
adapts (only context cancellation and deadline errors abort). A call
naming an unregistered tool gets `{"error": "unknown tool; did you mean
\"<name>\"?"}` when a registered name is within edit distance 2, else
`{"error": "unknown tool"}` — both count against the tool budget. When
one response requests several calls, parallel-eligible tools execute
concurrently while `Sequential` tools run strictly in call order; tool
results are always appended in original call order, and `AfterToolCall`
fires as each execution completes.

Set `Repetitive: true` for polling or status tools. It exempts the tool
from duplicate-call loop detection: identical calls are expected, and a
changing observation is the progress signal. The tool still counts
against `WithMaxToolCalls` and `WithMaxSteps` like any other.

## Hooks

Hooks are the extension surface — enough to build logging, tracing,
streaming UIs, caching, cost control, and entire agents (both presets are
built on them). `Hooks` is a struct of optional function fields; unset
fields cost nothing. All hooks run synchronously in loop order; a
returning error aborts the run and `Ask` returns a wrapped error naming
the hook — except `ErrSkipToolCall` from `BeforeToolCall`, which merely
skips that one tool call. The `*Event` is valid only during the hook
call; copy anything you retain.

| Hook | Fires | May do | Event fields of interest |
| --- | --- | --- | --- |
| `OnStart` | when a run begins | — | `Query`, `Messages` |
| `BeforeGenerate` | before each LLM call | **Append** to `Messages` (appends become permanent transcript entries — the steering seam); set `Skip` for a graceful stop | `Step`, `Messages`, `Skip` |
| `RebuildContext` | after `BeforeGenerate`, before the call | Observe or **replace** `Context` — the exact message list the model is about to see (see [recipes](#writing-your-own-agent)) | `Step`, `Messages` (read-only transcript), `Context` |
| `AfterGenerate` | after each LLM response | **Append** to `Messages`; flip `Finish` (the finish veto) | `Step`, `Messages`, `Response`, `Finish` |
| `BeforeToolCall` | before each tool call | Rewrite `Args`; return `ErrSkipToolCall` to veto; **may block** (approval gates) | `Step`, `Call`, `Args` |
| `AfterToolCall` | when a tool execution completes | — (fires in completion order) | `Step`, `Call`, `Args`, `Result`, `Err`, `Duration` |
| `OnDelta` | streaming answer/reasoning deltas | — | `Kind`, `Step`, `Delta`, `Timing` |
| `OnDone` | once, when the run finishes | — | `Answer` (fully populated), `Timing` |
| `OnError` | once, on terminal failure | — | `Err` (also returned from `Ask`), `Answer` |

```go
hooks := agentloop.Hooks{
    OnStart: func(ctx context.Context, ev *agentloop.Event) error {
        fmt.Println("run:", ev.Query)
        return nil
    },
    AfterToolCall: func(ctx context.Context, ev *agentloop.Event) error {
        fmt.Printf("%s -> %.80s (%v)\n", ev.Call.Name, ev.Result, ev.Duration)
        return nil
    },
}
ag := agentloop.MustNew(llm, agentloop.WithTools(clock), agentloop.WithHooks(hooks))
```

**The finish veto.** `AfterGenerate` receives `Finish` pre-set to "the
response carried no tool calls". The hook may flip it either way:
`false` on a text-only response — after appending observation messages —
keeps the loop running (this is how text-action protocols work; the bash
preset is the working example), and `true` on a tool-call response stops
immediately without executing the tools.

`Event` is one plain struct for all hook kinds; which fields are
populated depends on `Kind` (`EventStart`, `EventBeforeGenerate`,
`EventRebuildContext`, `EventAfterGenerate`, `EventAnswerDelta`,
`EventReasoningDelta`, `EventBeforeToolCall`, `EventAfterToolCall`,
`EventDone`, `EventError`):

| Field | Meaning |
| --- | --- |
| `Kind` | which loop point fired |
| `Step` | 1-based model inference the event belongs to |
| `Query` | the user query (`EventStart`) |
| `Messages` | the transcript so far; appendable in `BeforeGenerate`/`AfterGenerate`, read-only elsewhere |
| `Context` | the LLM-visible view for this step; pre-seeded to the transcript, replaceable by `RebuildContext` only |
| `Skip` | request a graceful stop (`BeforeGenerate`) |
| `Finish` | whether the response ends the run; flippable (`AfterGenerate`) |
| `Response` | the model response (`AfterGenerate`) |
| `Call` | the requested tool call (tool events) |
| `Args` | raw call arguments; rewritable by `BeforeToolCall` |
| `Result` | the payload delivered to the model (`AfterToolCall`) |
| `Err` | associated error (`AfterToolCall` handler errors, `EventError`) |
| `Duration` | the tool execution's wall time (`AfterToolCall`) |
| `Delta` | incremental streamed text (`OnDelta`) |
| `Timing` | llmhub latency telemetry snapshot (delta events; `EventDone` carries the complete set) |
| `Answer` | the fully populated result (`EventDone`, partial on `EventError`) |

## Budgets and loop safety

Multiple limit dimensions always apply at once, so no run can run away:

- **Tool budget** — `WithMaxToolCalls` (default 5; the search preset
  raises it to 8). When the budget is spent, the loop forces a final
  answer by sending the next inference with `llmhub.NoToolChoice()`. It
  also appends a one-time user message to the transcript telling the
  model the budget is exhausted and it must answer the original question
  now — synthesize the evidence already gathered, state plainly anything
  that could not be determined, and do not narrate further plans. The
  notice lands after the last tool result, appears in
  `Answer.Messages`, and is idempotent if forced steps repeat. Without
  it, models tend to spend the forced step narrating their next search
  instead of answering. If the model still returns tool calls on that
  step, none are executed and the step's text finishes the run — the
  loop can never spin forever, not even under a vetoing hook.
- **Step ceiling** — `WithMaxSteps` (default `2*MaxToolCalls+2`). A hard
  ceiling on model inferences; exceeding it returns `ErrBudgetExhausted`
  with `Answer.Status = StatusBudgetExhausted` (partial trace included).
  All graceful limits elsewhere funnel into this safety net.
- **Loop detection** — `WithDupCallLimit` (default 3, 0 disables). A
  repeat counts only when both the call (same name and arguments) **and**
  its observation are identical; changing output is progress, so legitimate
  polling does not trip it. That many repeats deliver the tool's real
  output with a warning JSON object appended — `{"warning": "repeated
  identical call; change approach", "repeated": "<excerpt>"}`; the real
  output is never replaced. One more identical call stops the run
  gracefully with `StatusStopped` before it executes. Any differing call
  or differing output resets the counter. Polling-style tools can opt out
  entirely with `Repetitive: true`.
- **Your context** — `ctx` deadlines and cancellation are honored
  everywhere (LLM calls, tools, HTTP, command execution) and returned
  unwrapped. This is the hard wall-clock stop; for *graceful* wall-time
  limits see the [bash agent](#the-bash-agent).

How runs end — `Answer.Status`:

| Status | Meaning |
| --- | --- |
| `StatusCompleted` | the model produced a final answer (including the forced answer after the tool budget ran out) |
| `StatusStopped` | graceful stop: a `BeforeGenerate` hook set `Skip`, or duplicate-call detection fired (identical calls with identical output) |
| `StatusBudgetExhausted` | the `MaxSteps` ceiling was reached; `Ask` also returns an error wrapping `ErrBudgetExhausted` |

## Streaming

`AskStream(ctx, query)` mirrors llmhub's streaming semantics: underlying
`Stream` chunks are forwarded as they arrive as `EventReasoningDelta` and
`EventAnswerDelta` events from **every** step, each tagged with `Step`.
Deltas from steps that end in tool calls are interim model commentary —
streamed for live UIs but excluded from the final `Answer.Text`, which is
the final step's text. Buffer per step if you want only the final answer,
or render step by step. Intermediate tool and step events are delivered
too, so CLI/UI consumers see progress. The channel closes after
`EventDone` or `EventError`; consumers must drain it until close.

```go
events, err := agent.AskStream(ctx, "What Go version was released most recently, and when?")
if err != nil {
    log.Fatal(err)
}
for ev := range events {
    switch ev.Kind {
    case agentloop.EventAnswerDelta:
        fmt.Print(ev.Delta)
    case agentloop.EventDone:
        for i, s := range search.Sources(ev.Answer) {
            fmt.Printf("[%d] %s — %s\n", i+1, s.Title, s.URL)
        }
    case agentloop.EventError:
        log.Print(ev.Err)
    }
}
```

- **Watchdog.** A run that receives no chunk for the
  `WithStreamWatchdog` window (default 90s) aborts with `ErrStreamStale`
  — a hung provider connection can no longer hang your process. `0`
  disables the watchdog.
- **Timing.** Every delta event carries an llmhub `Timing` snapshot;
  `EventDone` carries the complete set, and `Answer.Timing` aggregates it
  (`Total`/`Chunks` summed across steps; `Connected`/`FirstByte`/
  `FirstToken` from the first step). Timing is stamped by llmhub's
  client wrapper, which re-stamps each chunk as it is delivered — so
  provider-side stamps never reach consumers, only client-measured
  values do. `Answer.Timing` is nil only when the provider bypassed
  llmhub's `Client`.

The same `Hooks` work in streaming mode (`OnDelta` observes every delta
in stream order in addition to the channel). `AskStream` is implemented
entirely on top of the hook surface — nothing in the core special-cases
streaming.

## Conversations

`Ask` is one-shot. `Conversation` chains asks over a shared history so
follow-up turns see prior user messages, tool calls, results, and
answers:

```go
conv := agent.Conversation(
    agentloop.WithSystemPrompt("You are a research assistant."),
    agentloop.WithGenerateOptions(llmhub.WithCacheKey("session-17")),
)

first, err := conv.Ask(ctx, "What Go version was released most recently?")
second, err := conv.Ask(ctx, "And when is the next one expected?")
```

- **Append-only prefix.** The loop never edits, reorders, or rewrites
  prior messages, and the system message is frozen for the conversation's
  lifetime — this preserves provider prompt caching across turns. Long
  sessions are ideal cache candidates: pass
  `WithGenerateOptions(llmhub.WithCacheKey("session-id"))`.
- **Per-turn options.** Options passed to `Conversation` set the defaults
  (including the frozen system message); options passed to a turn's
  `Ask`/`AskStream` apply only to that turn. A per-turn
  `WithSystemPrompt` is ignored in favor of the frozen one.
- **Concurrency.** A `Conversation` is not safe for concurrent use: an
  `Ask`/`AskStream` while another turn is in flight fails fast with
  `ErrConversationBusy`.
- **Budgets are per turn.** `MaxToolCalls`, `MaxSteps`, and duplicate
  detection reset each turn; `TotalUsage()` aggregates across turns.
  `Messages()` returns the full transcript (read-only); `Answer.Messages`
  carries only that turn's own messages. `Reset()` drops the history and
  usage totals but keeps the system message.
- **Presets.** Build one from a preset's escape hatch:
  `searchAgent.Agent().Conversation(...)` — and keep using
  `search.Sources(ans)` on each turn's answer. Note the conversation runs
  the bare core loop without the preset protocol; typed preset
  conversations are deliberately deferred. (For the bash agent this means
  a conversation via `Agent()` has no text protocol, limits, or exit
  statuses — use one-shot `Ask` there.)
- **Compaction is a projection.** When history grows too long, don't
  rewrite it — derive a shorter LLM view with a `RebuildContext` hook
  (see [recipes](#writing-your-own-agent)); the conversation log stays
  complete.

## Skills

Skills are repeatable *procedures* — folders of instructions the model
discovers through a tiny catalog and loads on demand. Skills prepare,
tools execute: a skill is not a native tool and not a second agent, and
procedures never bloat the system prompt.

```
my-skill/
  SKILL.md        # required: YAML frontmatter (name, description) + markdown body
  references/     # optional deeper docs, loaded individually (L2)
  assets/         # optional templates/data (L2)
  scripts/        # optional executable helpers (L3 — run via an execution tool)
```

Enable them with `agentloop.WithSkills("path/to/skills")` (any agent,
preset, or conversation). At construction the loader scans the
directories (first occurrence of a name wins), appends a byte-stable
`## Skills` catalog to the system prompt, and registers one extra tool:

- **L0 Advertise** — one `- name: description` line per skill in the
  catalog (~100 tokens per skill, every call).
- **L1 Load** — the model calls `skill_read{"name": …}` and receives the
  full `SKILL.md` body as a tool result (never a prompt injection, which
  would bust the prefix cache).
- **L2 Resources** — `skill_read{"name": …, "file": …}` returns one file
  under `references/` or `assets/`. Linked files are listed in the body
  result so the model never needs to list directories. The path is
  jailed: absolute paths, `~`, `..`, and symlink escapes are rejected.
  `offset`/`limit` parameters are **refused by design** — models read
  page 1 and skip the rest, so instructional reads are never paginated.
- **L3 Scripts** — no dedicated tool: scripts run through whatever
  execution tool the agent already has (the bash agent's `bash` tool);
  only their output enters context. A skill that ships `scripts/` gains a
  footer in its `skill_read` body result (and in the pinned-skill
  rendering) with the skill's absolute base directory, the relative-path
  semantics, and a run-don't-read listing of its `scripts/` entries;
  script-less skills render exactly as before.

**Authoring a skill.** The frontmatter takes `name` (must equal the
directory name; lowercase letters/digits joined by single hyphens, 1–64
characters) and `description` (1–1024 characters — the router: write it
imperatively, "Use when…"; near-miss negatives belong in the body).
The body is markdown instructions below the frontmatter:

```markdown
---
name: release-notes
description: Use when the user asks for a changelog or release summary from a set of commits or a milestone.
---
Read the listed commits, group them as features / fixes / breaking
changes, and lead with the user-visible impact. See
references/style-guide.md for tone rules.
```

Pin the one or two procedures every run needs with
`agentloop.WithPinnedSkills("name")` — their bodies are inlined into the
system prompt at construction (at most two; more fails with
`ErrTooManyPinnedSkills`).

**Loading is fail-closed.** A folder that fails validation (bad
frontmatter, name/directory mismatch, oversized description, empty body)
is omitted and reported as a `SkillIssue` by `agentloop.LoadSkills(dirs...)`
— never an error, never a crash (only an unreadable scan root fails
construction). As a prompt-injection defense, NUL bytes reject the skill,
while Unicode Tags (U+E0001–U+E007F) and bidi overrides are stripped
before validation — when stripping changed anything the skill loads and
an advisory `SkillIssue` ("unicode steganography stripped") is recorded.

Skill loads are ordinary tool calls: they appear in `Answer.Tools`, in
traces, and in `AfterToolCall` hooks like any other execution.

## The search agent

`github.com/smhanov/agentloop/search` wires the core loop with two tools
and a domain-neutral research-assistant prompt: the model searches when
the answer may be recent or outside its training data, and fetches a
result's full page when snippets are insufficient.

- `web_search({"q", "num"?, "type"?})` — Serper Google search; `"type":
  "news"` routes to the news endpoint for time-sensitive events (news
  results carry `date` and `publisher`). Results return trimmed to
  title/link/snippet, plus up to five related queries for reformulation.
- `fetch_page({"url", "markdown"?})` — Serper's scrape endpoint returns
  readable page text (markdown optional). Text is truncated head+tail at
  20000 characters (`WithFetchLimit`, 0 = unlimited) so conclusions
  survive; a missing page yields `{"error": "fetch failed: 404"}` and the
  model tries another result.

**Sources.** `search.Answer` embeds the core answer plus `Sources`: every
`web_search` result from the trace, deduped by URL in first-seen order,
with `Fetched: true` marking pages the model read in full; a page fetched
by a URL no search returned is appended after them, also marked `Fetched`.
The default
prompt instructs the model to cite inline as `[n]` by first-seen order, so
`[n]` in the text aligns with `Sources[n-1]` — exactly when one search
returned the sources; across several searches the global dedupe order can
differ from the model's per-search numbering. `search.Sources(ans)` is a
pure function over any core answer, so streaming consumers can call it on
the `EventDone` payload. A user prompt may override citation style —
`Sources` is populated regardless, since it is built from the trace, not
text markers.

Searches and fetches both count against the tool budget, which the preset
raises to **8** because effective research alternates search → fetch →
answer.

**All preset options** (compose freely with core options in one list):

| Option | Default | Effect |
| --- | --- | --- |
| `WithSerperKey(key)` | `SERPER_API_KEY` env | Overrides the key source. |
| `WithSerperBaseURL(url)` | `https://google.serper.dev` | Test hook. |
| `WithSerperHTTPClient(c)` | 10s-timeout client | Custom `*http.Client` for all Serper calls. |
| `WithSearcher(s)` | built-in Serper client | Replaces the search backend (see below). |
| `WithFetcher(f)` | built-in Serper client | Replaces the fetch backend. |
| `WithSerperRetries(n)` | 1 (max 3) | Transparent 429 retries with jittered backoff; 0 disables; the model sees a failure only after retries are exhausted. |
| `WithMaxResults(n)` | 10 | Serper `num`, clamped to 1–10. |
| `WithFetchLimit(n)` | 20000 | Page-text chars per fetch, head+tail truncated; ≤0 unlimited. |
| `WithNoFetch()` | — | Drops `fetch_page`, making a search-only agent. |

**Graceful without a key.** Key resolution is `WithSerperKey`, then the
`SERPER_API_KEY` environment variable, then none — with no key the tools
return `{"error": "web search is not configured"}` / `{"error": "page
fetching is not configured"}` results and the agent still answers,
ungrounded. Nothing fails at construction.

**Custom backends.** The preset depends only on two tiny interfaces, so
caching wrappers, other APIs, or test doubles drop in without a Serper
key:

```go
type Searcher interface {
    Search(ctx context.Context, req SearchRequest) ([]Source, []string, error)
}

type Fetcher interface {
    Fetch(ctx context.Context, req FetchRequest) (*Page, error)
}
```

`Source` carries `Title`, `URL`, `Snippet`, `Date`, `Publisher`, and
`Fetched`; `Page` carries `URL`, `Title`, `Description`, `Text`,
`Markdown`, and `Credits`. Errors surface to the model as
`{"error": "search failed: <status>"}` / `{"error": "fetch failed:
<status>"}` while the run continues; the underlying `*SerperError`
(status + body excerpt) is visible in `ToolResult.Err` for callers.

## The bash agent

`github.com/smhanov/agentloop/bash` is a coding agent that solves tasks
by executing shell commands in a workspace until it reports completion —
a reimplementation of
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)'s behavior
on the unmodified agentloop core: same templates, same limits, same exit
statuses, same submission mechanism. Built with
[`WithSkills`](#skills), loaded scripts run through its `bash` tool (L3):
the `skill_read` result discloses the skill's base directory and
`scripts/` entries, and only the script's output enters context.

```go
llm, _ := llmhub.New("zai", os.Getenv("ZAI_API_KEY"),
    llmhub.WithBaseURL("https://api.z.ai/api/coding/paas/v4"),
    llmhub.WithModel("glm-5.3"))

agent, _ := bash.New(llm,
    bash.WithEnvironment(bash.NewLocalEnvironment("/tmp/workspace", nil, 60*time.Second)),
    bash.WithStepLimit(20),
    bash.WithCostLimit(1.5))

ans, err := agent.Ask(ctx, "Add a README.md describing this project.")
// ans.ExitStatus == bash.ExitSubmitted; ans.Submission holds the output
// ans.Messages is the full trajectory; ans.Tools the command trace
```

**Action protocols.** The default is mini-swe-agent's *text protocol*:
the model responds with plain text containing exactly one fenced bash
block (a `THOUGHT` section plus ```` ```mswea_bash_command ````; ````
```bash ```` is accepted as an alias). The preset parses the block,
executes the command, and appends the formatted observation as a user
message — using the core's finish veto to keep the loop going without
native tool calls. Responses with zero or multiple blocks get mini's
format-error guidance instead. `bash.WithToolProtocol()` switches to a
native `bash` tool (`{"command": string}`) for models with reliable tool
calling: same observations, limits, approval gate, and exit statuses;
the run simply ends when the model stops calling tools (the format-error
machinery does not apply, so `ExitStatus` is then empty).

**Environments.** Commands execute behind one interface, so containers
and remote backends are caller-implemented:

```go
type Environment interface {
    Execute(ctx context.Context, command string) (ExecResult, error)
}
```

`bash.NewLocalEnvironment(cwd, env, timeout)` runs each command in a
fresh `bash -c` subprocess (sh fallback): fixed working directory, extra
environment merged over the inherited one, per-command timeout (30s
default). Children start in their own process group and a timeout kills
the whole group with SIGKILL, so no orphans survive; partial output is
preserved and delivered with `ReturnCode: -1` plus `ExceptionInfo`.
Non-zero exit codes are results, not errors — the model reads the output
and iterates. Observations use mini's format: `<returncode>` and
`<output>` (or a head/tail/elision warning above `WithOutputLimit`,
default 10000 characters).

**Limits map to exit statuses** — every limit stops the run *gracefully*
and produces a `*bash.Answer` (a result, not a Go error; errors are
reserved for infrastructure). `Ask` returns the core trace plus:

| `ExitStatus` | Meaning |
| --- | --- |
| `"Submitted"` | a command exited 0 with the submit marker (`WithSubmitMarker`, default `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`) as its first output line; `Submission` holds the remaining output |
| `"LimitsExceeded"` | `WithStepLimit` (default 0 = unlimited) or `WithCostLimit` (cumulative provider-reported cost) tripped |
| `"TimeExceeded"` | `WithWallTimeLimit` (monotonic clock) tripped |
| `"RepeatedFormatError"` | `WithMaxConsecutiveFormatErrors` (default 3) malformed responses in a row; any clean step resets the counter |
| `""` (empty) | tool protocol only: the model stopped calling tools — the loop's natural end |

**Approvals.** `WithApprover(fn)` consults the gate before every command.
A returned error vetoes — the model sees
`{"error": "command not approved: …"}` and may adjust while the run
continues. `nil` (the default) runs unattended, mini's *yolo* mode; a
gate that blocks on human input implements *confirm* mode (the CLI's
`-confirm` flag is exactly this). Blocking in the approval gate is
sanctioned; `BeforeToolCall` may block for the same reason.

**mini-swe-agent compatibility:**

| mini-swe-agent | agentloop implementation |
| --- | --- |
| `run(task)` message loop | core `Ask` loop; continuation via the `AfterGenerate` finish veto |
| `step_limit` | `WithStepLimit` enforced in a `BeforeGenerate` hook → graceful stop, `ExitStatus="LimitsExceeded"` |
| `cost_limit` (stop after exceeding) | same hook, cumulative `Usage.Cost` across steps |
| `wall_time_limit_seconds` | same hook, monotonic clock → `"TimeExceeded"` |
| `max_consecutive_format_errors` (3) | text-protocol counter → `"RepeatedFormatError"` |
| `Submitted` + submission capture | submit-marker detection on command output |
| trajectory JSON | native: `bash.Answer` serializes messages, trace, exit status, and usage — save it from an `OnDone` hook or with `-json` |
| `handle_uncaught_exception` | deliberate deviation: infrastructure failures return Go errors; only agent-level outcomes are results |

**All preset options:**

| Option | Default | Effect |
| --- | --- | --- |
| `WithEnvironment(env)` | `NewLocalEnvironment("", nil, 0)` | Command execution backend. |
| `WithStepLimit(n)` | 0 (unlimited) | Max model steps per run. The core's hard `MaxSteps` is raised to `max(StepLimit, 2*MaxToolCalls+2)` as the safety net. |
| `WithCostLimit(usd)` | 0 (unlimited) | Graceful stop once cumulative reported cost reaches `usd`. |
| `WithWallTimeLimit(d)` | 0 (unlimited) | Graceful wall-clock stop. |
| `WithMaxConsecutiveFormatErrors(n)` | 3 | Malformed-response stop; 0 = unlimited. |
| `WithSubmitMarker(s)` | mini's marker | The output line that ends a run as `Submitted`. |
| `WithOutputLimit(n)` | 10000 | Observation truncation threshold in characters. |
| `WithTemplates(system, task, observation)` | ported mini templates | Override templates (Go `text/template`); empty strings keep defaults. The system template renders once at construction; task and observation overrides also work per `Ask`. |
| `WithApprover(fn)` | nil (yolo) | Approval gate before every command. |
| `WithToolProtocol()` | text protocol | Native tool-calling transport. |
| `WithHooks(h)` | — | **Composes** with the protocol's internal hooks (protocol first, then yours). Prefer it over `agentloop.WithHooks`, which replaces the whole hook set and would silently disable the protocol. |

**Streaming note.** `AskStream` forwards the core events exactly, but
`ExitStatus` and `Submission` are attached only to `bash.Ask`'s result —
they are not on stream events. Streaming consumers that need them should
call `Ask` (or use the CLI's `-json`, which reads it from `Ask`).

## Writing your own agent

Both presets are assembled from the public core with no special access —
which means your custom agent has exactly the same powers. The recipes:

**Custom tools + hooks.** Register domain tools with `WithTools` and
observe or gate them with `WithHooks`. A complete agent can be one tool
and two hooks.

**Text-action protocols via the finish veto.** If your model or task
prefers a wire format instead of native tool calls, implement it in an
`AfterGenerate` hook: parse the response, execute the action, append the
observation as a user message, and set `ev.Finish = false` to keep the
loop going. Malformed responses append corrective guidance instead —
also with `Finish = false`, so the model retries in-band. The bash
preset's text protocol is the working example of the whole pattern
(including the consecutive-error stop).

```go
hooks := agentloop.Hooks{
    AfterGenerate: func(ctx context.Context, ev *agentloop.Event) error {
        action, ok := parseAction(ev.Response.Text()) // your format
        if !ok {
            ev.Messages = append(ev.Messages,
                llmhub.NewUserMessage(llmhub.Text("Respond with exactly one action block.")))
            ev.Finish = false // guidance appended; the model retries
            return nil
        }
        ev.Messages = append(ev.Messages,
            llmhub.NewUserMessage(llmhub.Text(runAndObserve(ctx, action))))
        ev.Finish = false // one more inference: the model reacts to the output
        return nil
    },
}
ag := agentloop.MustNew(llm, agentloop.WithSystemPrompt(protocolPrompt), agentloop.WithHooks(hooks))
```

**Steering via `BeforeGenerate`.** The hook's `ev.Messages` appends are
permanent transcript entries. Expose a thread-safe queue in your process;
each `BeforeGenerate` drains whatever arrived into the transcript, so
other goroutines steer a running agent:

```go
var inbox steeringQueue // your thread-safe queue of user messages

hooks := agentloop.Hooks{
    BeforeGenerate: func(ctx context.Context, ev *agentloop.Event) error {
        for _, msg := range inbox.Take() {
            ev.Messages = append(ev.Messages, llmhub.NewUserMessage(llmhub.Text(msg)))
        }
        return nil
    },
}
```

**Cost caps via `ev.Skip`.** Accumulate `ev.Response.Usage.Cost` in an
`AfterGenerate` hook and stop gracefully in `BeforeGenerate` once a cap
is reached — the run finishes with `Answer.Status = StatusStopped` and
whatever context existed at that point:

```go
var spent float64

hooks := agentloop.Hooks{
    AfterGenerate: func(ctx context.Context, ev *agentloop.Event) error {
        if ev.Response != nil {
            spent += ev.Response.Usage.Cost
        }
        return nil
    },
    BeforeGenerate: func(ctx context.Context, ev *agentloop.Event) error {
        if spent >= 1.00 {
            ev.Skip = true // graceful stop before the next paid call
        }
        return nil
    },
}
```

**Context rebuild and compression via `RebuildContext`.** Long runs and
conversations outgrow the context window. `RebuildContext` fires before
every LLM call with `ev.Context` pre-seeded to the full transcript; the
hook may observe it (log, audit, measure) or replace it wholesale —
compress older steps into a summary, drop stale tool outputs, enforce a
token budget, redact secrets:

```go
const keep = 6 // never compress the most recent messages

hooks := agentloop.Hooks{
    RebuildContext: func(ctx context.Context, ev *agentloop.Event) error {
        if len(ev.Context) <= keep {
            return nil
        }
        summary := summarize(ev.Context[:len(ev.Context)-keep]) // your summarizer
        ev.Context = append([]*llmhub.Message{
            llmhub.NewSystemMessage(llmhub.Text("Earlier steps, summarized: " + summary)),
        }, ev.Context[len(ev.Context)-keep:]...)
        return nil
    },
}
```

Three invariants make this safe: replacements are **ephemeral** (only
this call is affected; the transcript, `Answer.Messages`, and
conversation history are untouched); the projection must be
**deterministic** (same transcript prefix → same projected prefix, or
prompt caching busts); and the trailing assistant tool-call message with
its tool-result messages — the in-flight pair — must survive verbatim,
because providers reject contexts with unanswered tool calls. Compression
that eats in-flight state breaks every tool protocol.

## Testing your agent

Register a scripted provider with llmhub — no network, no keys, fully
deterministic — and run the loop against it. agentloop's own test suite
is built this way; the shape:

```go
type scriptedTurn struct {
    text  string
    calls []*llmhub.ToolCallContent
}

type scriptedProvider struct {
    mu    sync.Mutex
    turns []scriptedTurn
}

func (p *scriptedProvider) Name() string { return "scripted" }

func (p *scriptedProvider) Generate(ctx context.Context, prompt []*llmhub.Message, opts ...llmhub.Option) (*llmhub.Response, error) {
    p.mu.Lock()
    defer p.mu.Unlock()
    if len(p.turns) == 0 {
        return nil, errors.New("scripted: script exhausted")
    }
    turn := p.turns[0]
    p.turns = p.turns[1:]
    resp := &llmhub.Response{}
    if turn.text != "" {
        resp.Content = append(resp.Content, llmhub.Text(turn.text))
    }
    for _, call := range turn.calls {
        resp.Content = append(resp.Content, call)
    }
    return resp, nil
}

func (p *scriptedProvider) Stream(ctx context.Context, prompt []*llmhub.Message, opts ...llmhub.Option) (<-chan llmhub.StreamChunk, error) {
    resp, err := p.Generate(ctx, prompt, opts...)
    ch := make(chan llmhub.StreamChunk, 1)
    if err == nil {
        ch <- llmhub.StreamChunk{Delta: resp.Text(), Done: true}
    }
    close(ch)
    return ch, err
}
```

The factory registration is the seam: the provider constructor returns
whatever state you want the tests to reach:

```go
var current *scriptedProvider

func init() {
    llmhub.MustRegisterProvider("scripted", func(apiKey string, opts ...llmhub.Option) (llmhub.Provider, error) {
        return current, nil
    })
}

func scriptAgent(t *testing.T, turns ...scriptedTurn) *agentloop.Agent {
    t.Helper()
    current = &scriptedProvider{turns: turns}
    llm, err := llmhub.New("scripted", "test-key")
    if err != nil {
        t.Fatalf("llmhub.New: %v", err)
    }
    return agentloop.MustNew(llm, agentloop.WithTools(clock))
}

func TestAgentCallsToolThenAnswers(t *testing.T) {
    ag := scriptAgent(t,
        scriptedTurn{calls: []*llmhub.ToolCallContent{
            {ID: "call_1", Name: "clock", Arguments: `{}`},
        }},
        scriptedTurn{text: "It is 12:00:00Z."},
    )

    ans, err := ag.Ask(context.Background(), "What time is it?")
    if err != nil {
        t.Fatal(err)
    }
    if ans.ToolCalls != 1 || ans.Steps != 2 {
        t.Fatalf("trace: %+v", ans)
    }
}
```

Because every request records the exact prompt and config, you can assert
on the context the model saw — conversation prefix stability, budget
forcing, hook rewrites — the way agentloop's own tests do.

## Troubleshooting

- **`ErrBudgetExhausted` / `status=budget_exhausted`.** The step ceiling
  was reached before a final answer. Raise `WithMaxToolCalls` /
  `WithMaxSteps`, check that your tool outputs give the model what it
  needs to finish (huge outputs waste steps — truncate in the handler),
  and rely on `WithDupCallLimit` if the model repeats the same call with
  unchanged output. The CLI's exit code 2 means exactly this.
- **`ErrStreamStale`.** No stream chunk arrived within the watchdog
  window — usually a hung provider connection. Retry, or tune
  `WithStreamWatchdog` (`0` disables, not recommended for unattended
  use).
- **`StatusStopped` without an obvious cause.** Either a hook set
  `ev.Skip` (your cost cap or the bash limits fire this way) or
  duplicate-call detection fired after the same call kept returning the
  same output. In the latter case the last tool result carries the real
  output with `"repeated identical call; change approach"` appended, and
  the run stops on the next identical call with a
  `"repeated identical call; stopping"` observation. The trace in
  `Answer.Tools` shows which; for a deliberately polled tool, set
  `Repetitive: true`.
- **`RepeatedFormatError` (bash agent).** The model cannot reliably
  follow the text-action protocol. Try `bash.WithToolProtocol()` (or the
  CLI's `-tool-protocol`) for models with solid tool calling, or lower
  the temperature. A cleaner model usually fixes it outright.
- **`ErrConversationBusy`.** Two asks overlapped on one `Conversation`.
  Serialize turns, or give concurrent workers separate conversations.
- **`{"error": "unknown tool; did you mean …"}` in a tool result.** The
  model called a name you never registered (or a per-call `WithTools`
  replaced your construction-time set — the core's `WithTools` replaces,
  it never merges). Check for per-call option surprises.
- **`web search is not configured`.** The search agent ran without a
  Serper key: pass `search.WithSerperKey` or export `SERPER_API_KEY`.
- **The `AskStream` channel never closes.** It closes after
  `EventDone`/`EventError` — make sure you drain until `range` ends,
  including on error paths.

---

## Testing

```bash
go test ./...
```

The suite is fully offline and deterministic (scripted providers, fake
searchers, temp-dir environments). `go test -race ./...` is also part of
CI, along with `gofmt -l`, `go vet ./...`, and two policy tests: an
exported-API budget (the library keeps a deliberately small surface —
the test fails when exported symbols grow past a fixed ceiling) and a
repo hygiene scan (the repository stays domain-neutral).

## Versioning and stability

Releases follow [semantic versioning](https://semver.org) and the module
path is stable: `go get -u github.com/smhanov/agentloop` always moves
you forward safely. While the project is pre-v1, the minor version is
the breaking-change signal; after v1, breaking changes only ever cross a
major version.

Two project rules keep upgrades boring:

- **A deliberately small API surface.** The exported API is under a
  fixed budget enforced by a test in CI — new symbols must earn their
  place, so the surface you compile against stays lean (see
  [Testing](#testing)).
- **A supported Go version window.** The minimum is Go 1.21; each new
  Go release is adopted once CI proves the suite against it.

## Contributing

Issues and PRs are welcome. This project uses a **rebase-only workflow**
for merging into `main`: PRs are rebased onto `main` and merged with a
fast-forward rebase — never merge commits or squash. Keep your branches
up to date with `git rebase main` and keep commits small and clean, as
they land in `main` individually.

## Comparison

- **[llmhub](https://github.com/smhanov/llmhub)** is the layer below:
  provider-agnostic LLM access, tool calling, streaming, caching, usage
  telemetry. agentloop adds the loop around it. Use llmhub when you want
  to write the loop yourself; use agentloop when you want the loop done.
- **[laconic](https://github.com/smhanov/laconic)** is a research-agent
  framework with fixed pipeline stages (planner/synthesizer/finalizer)
  and state compression for small-context models. agentloop is a
  lower-level, more general building block: one transparent tool-calling
  loop with hooks at every step, from which specific agents are assembled
  by configuration. Use laconic when you want a research pipeline; use
  agentloop when you want to build your own agent.
- **[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)** is a
  specific, minimal coding agent design. agentloop's bash preset
  reproduces its protocol, templates, limits, and exit statuses on the
  unmodified core — use it as-is, or as the reference for building your
  own protocol on the same loop.

## License

MIT License © 2026 agentloop contributors
