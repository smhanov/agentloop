# agentloop — Agentic Loop Library Specification

**Module:** `github.com/smhanov/agentloop`
**Language:** Go 1.21+ (matches llmhub)
**Repo:** public (`github.com/smhanov/agentloop`), MIT license
**Status:** Draft — v1 design

## 1. Overview

`agentloop` is a Go library providing a **general-purpose agentic loop**:
send prompts to an LLM, let the model invoke tools, execute them, feed
results back, and repeat until the model produces a final answer. Every step
of the loop is observable and configurable through **hooks**.

The library ships with two built-in presets assembled entirely from the core
loop's public surface — no preset-specific hacks in the core:

- **Search agent** — grounds answers in Serper web-search results.
- **Bash coding agent** — solves tasks by executing shell commands in a
  workspace; protocol-compatible with
  [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) (§8).

All LLM access goes through
[`github.com/smhanov/llmhub`](https://github.com/smhanov/llmhub); agentloop
never talks to an LLM vendor directly.

Design decisions throughout this spec are cross-checked against
`docs/references.md` — the *Agent Harness Writing Reference* distilled from
studies of OpenClaw, GoClaw (+NLB), PicoClaw, ZeroClaw, Hermes, gptme,
mini-SWE-agent, and the DeepSeek harness (dsh). §18 maps its guidance to
the choices made here.

```
            ┌──────────────────────── core loop ────────────────────────┐
query ──▶   │  [Start]──▶[Generate]──▶ tool calls? ──no──▶ [Done] Answer │
            │                 │ yes                                  │
            │                 ▼                                      │
            │          [Run tools]──▶ append results ──▶ (loop)       │
            └──────────────────────────────────────────────────────────┘
 hooks:      OnStart  Before/AfterGenerate  Before/AfterToolCall  OnDelta  OnDone/OnError
```

### Why this package exists

llmhub provides provider-agnostic LLM access including tool calling, but the
caller must still write the loop: dispatch tool calls, append result
messages, budget iterations, aggregate usage. `agentloop` is that loop,
done once and well.

**Generality rule (enforced by the two presets):** anything a preset needs
must be expressible via the core's public surface — `Tool`, `Hooks`,
`Options`. If a preset cannot be built that way, the core gains the missing
*general* mechanism (e.g. the AfterGenerate finish-veto added for the bash
preset, §5.3), never a preset-specific branch. The search preset and the
bash preset together prove the loop is general: one is a single-tool
RAG-style agent, the other is a long-running command-execution agent with a
text-action protocol, limits, and exit statuses.

Positioning vs [laconic](https://github.com/smhanov/laconic): laconic is a
research-agent framework with fixed pipeline stages
(planner/synthesizer/finalizer) and state compression for small-context
models. `agentloop` is a lower-level, more general building block: a
single transparent tool-calling loop with hooks at every step, from which
specific agents are assembled by configuration. Use laconic when you want a
research pipeline; use agentloop when you want to build your own agent.

### Goals

- General-purpose agentic loop: any tools, any llmhub provider.
- Configurable hooks at every step, sufficient for logging, tracing,
  streaming UIs, caching layers, cost control, and custom agent presets.
- First-class **multi-turn conversations** (`Conversation`, §5.5).
- **Multiple parallel tool calls** per step, bounded and order-preserving
  (§5.2).
- At least two built-in presets (search, bash coding) to enforce and
  demonstrate generality.
- **Skills** with progressive disclosure (L0–L3 invariant,
  `docs/references.md` §7): procedures live in skill folders, not native
  tools and not the system prompt (§5.8).
- One-call API for the common case (ask a question, get an answer).
- Idiomatic Go throughout (see §3).
- Deterministic, offline-testable behavior.

### Non-goals (v0)

- Self-hosted page fetching (own HTTP client + HTML-to-text extraction) —
  fetching goes through Serper's scrape endpoint (§9.4); a custom `Fetcher`
  (§9.1) can replace it.
- Search backends other than Serper (extension point provided, §9).
- Durable/persistent sessions (disk-backed transcripts, per-user session
  stores) — the in-memory `Conversation` is the v0 scope.
- Orchestration of multiple agents, planner/synthesizer pipelines (laconic's
  territory), or autonomous long-running daemons.
- Skills roll-out phases 3–4 (`docs/references.md` §7): agent authoring
  nudges, BM25 routing, skill hubs/marketplaces, quarantine + provenance
  lockfiles, per-skill subprocess isolation. v0 ships phases 0–2
  (catalog + filesystem + reader tool).
- SWE-bench batch harnessing/submission, trajectory browser UI.
- Container environments beyond local (docker/podman/singularity) — the
  `Environment` interface (§8.2) allows them; only `LocalEnvironment` ships
  in v0.

## 2. Dependencies

| Dependency | Purpose |
| --- | --- |
| `github.com/smhanov/llmhub` | All LLM access: providers, tool calling, streaming, usage/cost telemetry. |
| Serper.dev HTTP API | Web search **and** page scraping (`/search`, `/scrape`) for the search preset. Accessed with `net/http`; no SDK. |
| `os/exec` + `syscall` | Local command execution for the bash preset. |

Stdlib otherwise. Importing the package requires no API keys; only the
presets' backends do.

## 3. Go Conventions (idiomatic Go, enforced)

- `context.Context` is the first parameter of every function that does I/O;
  the loop, tools, and hooks all honor cancellation and deadlines.
- Functional options for both construction (`New`) and per-call overrides
  (`Ask`), matching llmhub's `Generate(ctx, prompt, opts...)` idiom.
- Errors are wrapped with `%w`; sentinel errors (`errors.Is`-able) for
  expected conditions; the library never panics; no `log` calls in library
  code (hooks are the observability surface).
- Types are safe for concurrent use after construction; per-run state lives
  in locals, not fields.
- Small exported surface: every exported symbol has a doc comment;
  `gofmt`/`go vet` clean; table-driven tests; no `time.Sleep` in tests.
- No generics unless they pay for themselves (v0 uses none).
- Zero-config defaults; every default overridable.

## 4. Package Layout

Three packages: the root holds the general-purpose core; each preset is its
own subpackage. Preset packages alias the core option type
(`type Option = agentloop.Option`) so core and preset options compose in a
single variadic list.

```
agentloop/          github.com/smhanov/agentloop — the core loop
  agent.go          Agent type, New, Ask, AskStream — the core loop
  loop.go           step execution, budgets, forced-final logic
  conversation.go   Conversation: multi-turn sessions over Agent
  tool.go           Tool type, Handler, dispatch, unknown-tool handling
  hooks.go          Hooks struct, Event types and kinds
  options.go        functional options (constructor + per-call)
  answer.go         Answer, ToolResult trace types
  skills.go         Skill loading/validation, catalog rendering, skill_read tool (§5.8)
  errors.go         sentinel errors
  json.go           ExtractJSONObject and balanced-JSON helpers (§14.8)
  search/           github.com/smhanov/agentloop/search — search preset (§6)
    agent.go        New, Agent, Answer, Source, Sources
    serper.go       Serper client, Searcher/Fetcher interfaces, response mapping
    prompt.go       default system prompt, tool declarations
  bash/             github.com/smhanov/agentloop/bash — coding preset (§8)
    agent.go        New, Agent, Answer, protocols, limits, templates,
                    action parser, observation formatter
    environment.go  Environment interface, ExecResult, NewLocalEnvironment
  *_test.go         offline tests (generic prompts only — see §15)
  examples/ask/     example CLI (§13), runs the local test corpus
  README.md         public documentation (structure per §15)
  LICENSE           MIT
```

## 5. Core API

### 5.1 Construction

```go
package agentloop

// New creates a general-purpose agent over the given llmhub client with no
// tools and no system prompt; configure it with options.
func New(llm *llmhub.Client, opts ...Option) (*Agent, error)

// MustNew is like New but panics on error (for examples and tests).
func MustNew(llm *llmhub.Client, opts ...Option) *Agent
```

Options (constructor and per-call unless noted):

```go
WithSystemPrompt(p string) Option      // prepended as the system message; byte-stable for the session (§5.5)
WithTools(t ...Tool) Option            // register tools (replaces set); keep the set small — schemas cost tokens every call
WithMaxToolCalls(n int) Option         // tool executions allowed per Ask; default 5, min 1 (search preset: 8)
WithMaxSteps(n int) Option             // hard ceiling on model inferences per Ask; default 2*MaxToolCalls+2
WithParallelTools(n int) Option        // max concurrent parallel-eligible (Sequential == false) executions per step; default 3; 1 = sequential
WithDupCallLimit(n int) Option         // consecutive identical calls with identical output before graceful stop; default 3; 0 disables
WithStreamWatchdog(d time.Duration) Option // abort AskStream when no chunk arrives for d; default 90s; 0 disables
WithHooks(h Hooks) Option              // hook callbacks (see §5.3)
WithSkills(dirs ...string) Option      // enable skills: catalog in prompt + skill_read tool (see §5.8)
WithToolChoice(c llmhub.ToolChoice) Option // default llmhub.AutoToolChoice()
WithGenerateOptions(opts ...llmhub.Option) Option // forwarded to every Generate/Stream (e.g. temperature, cache key)
```

Vocabulary (adopted harness-wide convention): a **Turn** handles one user
message (one `Ask`); a **Step** is one model inference plus its tool
executions. `Answer.Steps` counts inferences, not turns. Wall-time caps at
the core level come from the caller's `ctx` deadline (hard stop); graceful
wall-time limits are preset-level (§8.1 `WithWallTimeLimit`).

### 5.2 Tools

```go
// Handler executes a tool invocation. args is the raw JSON arguments string
// from the model (llmhub ToolCallContent.Arguments). The return value is
// inserted verbatim as the tool result message content.
type Handler func(ctx context.Context, args json.RawMessage) (string, error)

```go
type Tool struct {
    Def llmhub.Tool // Name, Description, Parameters (JSON Schema)
    Run Handler

    // Sequential forces strict call-order execution that never overlaps
    // with anything (the inverse of dsh's isConcurrencySafe, chosen so the
    // Go zero value means the common default). Tools with the zero value
    // are eligible for parallel batching bounded by WithParallelTools; set
    // Sequential to true for stateful tools (shell sessions,
    // transactions).
    Sequential bool

    // Repetitive exempts the tool from duplicate-call loop detection. Set
    // it for polling/status tools: identical calls are expected, and a
    // changing observation is the progress signal. Budgets still apply.
    Repetitive bool
}
```
```

- Tool results are appended with `llmhub.NewToolResultMessage(call.ID,
  call.Name, llmhub.Text(output))` in call order, after the assistant message
  is appended verbatim (tool-call parts preserved).
- A handler error becomes a tool result of `{"error": "<err>"}` — the model
  sees the failure and may adapt; the loop does not abort (except ctx).
  This is deliberate in-band repair: parse/format failures go back to the
  model as tool results, never hard-fail the run.
- A call naming an unregistered tool yields
  `{"error": "unknown tool; did you mean \"<name>\"?"}` when a registered
  name is within edit distance 2 (Levenshtein — Hermes' hallucinated-name
  repair), else `{"error": "unknown tool"}`; counts against `MaxToolCalls`.
- **Parallel tool calls** are supported end-to-end: llmhub reassembles
  OpenAI-style parallel argument fragments (`ToolCallContent.Index`), the
  loop dispatches parallel-eligible (`Sequential: false`) executions
  concurrently bounded by `WithParallelTools` (default 3; `1` = sequential)
  while `Sequential: true` tools execute strictly in order, never
  overlapping with anything, `AfterToolCall` fires as each execution
  completes, and tool result messages are appended in **original call
  order** regardless of completion order.
- **Loop detection** (ZeroClaw, output-aware): `WithDupCallLimit(n)`
  (default 3, 0 disables) — N consecutive identical tool calls (same name
  and arguments) **with identical output** trigger the tool's real output
  with a `{"warning": "repeated identical call; change approach",
  "repeated": "<excerpt>"}` object appended (the output is never
  replaced); the next identical call stops the run gracefully with
  `Status = StatusStopped` before it executes. A differing call or a
  changed output resets the streak, so legitimate polling is not a loop.
  `Tool.Repetitive` opts a polling/status tool out entirely. Required for
  unattended operation; the stop is pre-execution, so calls and steps
  never exceed their ceilings.

### 5.3 Hooks

Hooks are the extension surface — enough to build logging, tracing,
streaming UIs, caching layers, cost caps, and entire agent presets (the
bash preset is one, §8). `Hooks` is a struct of optional function fields;
unset fields are no-ops (zero cost).

```go
type Hooks struct {
    OnStart        func(ctx context.Context, ev *Event) error // run begins; ev.Query, ev.Messages
    BeforeGenerate func(ctx context.Context, ev *Event) error // each LLM call; may APPEND to ev.Messages (steering/injection seam — appends are permanent, part of the transcript); may set ev.Skip to stop gracefully before querying
    RebuildContext func(ctx context.Context, ev *Event) error // fires after BeforeGenerate, before the LLM call: OBSERVE or REPLACE ev.Context — the exact message list the model sees this step (see below); the transcript (ev.Messages) is never mutated
    AfterGenerate  func(ctx context.Context, ev *Event) error // each LLM response; may APPEND to ev.Messages; may flip ev.Finish (see below)
    BeforeToolCall func(ctx context.Context, ev *Event) error // each tool call; may REWRITE ev.Args; return ErrSkipToolCall to veto execution; MAY BLOCK (approval gates, §8.1)
    AfterToolCall  func(ctx context.Context, ev *Event) error // tool finished; ev.Result, ev.Err, ev.Duration
    OnDelta        func(ctx context.Context, ev *Event) error // streaming answer/reasoning deltas (Kind distinguishes)
    OnDone         func(ctx context.Context, ev *Event) error // final; ev.Answer fully populated
    OnError        func(ctx context.Context, ev *Event) error // final failure; ev.Err; the error is still returned from Ask
}
```

Semantics:

- All hooks run synchronously in loop order (see diagram §1). Deltas are
  emitted in stream order.
- `AfterToolCall` fires in **completion order** (useful for progress UIs);
  tool result messages still append in call order (§5.2).
- **Finish veto (AfterGenerate):** `ev.Finish` arrives pre-set to
  `len(resp.ToolCalls()) == 0`. The hook may flip it either way:
  - `false` on a text-only response **after appending observation
    messages** keeps the loop running — this is how text-action protocols
    (§8.3) continue without native tool calls;
  - `true` on a tool-call response stops immediately without executing
    the tools.
- **Context observe/rebuild (RebuildContext):** fires before every LLM
  call with `ev.Messages` = the full append-only transcript and
  `ev.Context` = the message list the model is about to see (pre-seeded
  with the transcript). The hook may *observe* (`ev.Context` as sent —
  log it, audit it, measure it) or *replace* `ev.Context` wholesale —
  compress older steps into a summary, drop stale tool outputs, enforce a
  token budget, redact secrets. This is the dsh `deriveMessages()`
  pattern (`docs/references.md` §2.5, §8): the transcript is the
  append-only source of truth; the LLM context is a per-step derived
  projection. Rules:
  - Replacements are **ephemeral** — they affect only this call; the
    transcript, `Answer.Messages`, and Conversation history are unchanged.
  - Ordering: `BeforeGenerate` (transcript appends persist) →
    `RebuildContext` (projection) → `Generate(ctx, ev.Context, …)`.
  - **Invariant hooks must preserve:** the trailing assistant
    tool-call message and its tool-result messages of the most recent
    step must survive verbatim — providers reject contexts with
    unanswered tool calls, and tool-call protocols break without them
    (`docs/references.md` §13.6: compression must not eat in-flight
    state).
  - A **deterministic** projection preserves prompt caching (same
    transcript prefix → same projected prefix); ad-hoc rewriting of the
    prefix mid-run busts the cache (`docs/references.md` §5).
  - Unset (the default) means `ev.Context` == the transcript: exact
    legacy behavior.
- A hook returning an error aborts the run; `Ask` returns a wrapped error
  identifying the hook, except `ErrSkipToolCall` from `BeforeToolCall`,
  which merely skips that tool (its result message says "skipped").
- Hooks must be fast and non-blocking, with one exception: `BeforeToolCall`
  may block for **approval gates** (human confirmation before destructive
  tools — the bash preset's approver, §8.1, relies on this). Long work
  otherwise belongs in a goroutine fed from the hook.
- The `*Event` is valid only during the hook call; copy anything retained.

One event type, llmhub-chunk style (plain struct, not interface):

```go
type EventKind int
const (
    EventStart EventKind = iota
    EventBeforeGenerate
    EventRebuildContext
    EventAfterGenerate
    EventAnswerDelta
    EventReasoningDelta
    EventBeforeToolCall
    EventAfterToolCall
    EventDone
    EventError
)

type Event struct {
    Kind     EventKind
    Step     int                     // 1-based model inference (see vocabulary, §5.1)
    Query    string                  // Start
    Messages []*llmhub.Message       // Start, BeforeGenerate, AfterGenerate (appendable); RebuildContext: full transcript, read-only
    Context  []*llmhub.Message       // RebuildContext: the LLM-visible view this step; pre-seeded with the transcript; hook may replace
    Skip     bool                    // BeforeGenerate: request graceful stop
    Finish   bool                    // AfterGenerate: default = no tool calls; hook may flip
    Response *llmhub.Response       // AfterGenerate
    Call     *llmhub.ToolCallContent // tool events
    Args     json.RawMessage         // BeforeToolCall (rewritable), AfterToolCall
    Result   string                  // AfterToolCall
    Err      error                   // AfterToolCall, Error
    Duration time.Duration           // AfterToolCall
    Delta    string                  // Answer/ReasoningDelta
    Timing   *llmhub.TimingMetrics   // delta/done events: snapshot; Done carries the complete set (llmhub semantics)
    Answer   *Answer                 // Done
}
```

### 5.4 Asking

```go
// Ask runs the loop to completion.
func (a *Agent) Ask(ctx context.Context, query string, opts ...Option) (*Answer, error)

// AskStream is Ask with streaming, mirroring llmhub's streaming semantics:
// underlying llmhub Stream chunks are forwarded as they arrive — reasoning
// deltas (EventReasoningDelta, from llmhub StreamChunk.ReasoningDelta) and
// content deltas (EventAnswerDelta) from EVERY step, each tagged with Step.
// Deltas from steps that end in tool calls are interim model commentary —
// streamed for live UIs but excluded from the final Answer.Text, which is
// the final step's text; consumers wanting only the final answer buffer
// until Done (or render step by step). Every delta/done event carries a
// llmhub Timing snapshot; Done carries the complete set. Intermediate
// tool/step events are also delivered so CLI/UI consumers see progress.
// The channel closes after Done/Error. Implemented entirely on top of
// Hooks.
func (a *Agent) AskStream(ctx context.Context, query string, opts ...Option) (<-chan Event, error)
```

Loop algorithm:

1. `messages := [system?, user(query)]`; fire `OnStart`.
2. Fire `BeforeGenerate`; if `ev.Skip`, stop gracefully and synthesize the
   answer from current context (used for cost caps, "stop early" rules).
3. Fire `RebuildContext` (if set) with `ev.Context` pre-seeded to the
   transcript; `Generate` is called with `ev.Context` — unset hook means
   the transcript itself is the context (§5.3). (Stream in streaming
   mode, deltas forwarded to `OnDelta`.)
4. Append the assistant message verbatim; fire `AfterGenerate` with
   `ev.Finish = len(resp.ToolCalls()) == 0` and `ev.Messages` appendable.
5. If `ev.Finish` is true: the step's text is the final answer → fire
   `OnDone`, return. (A hook may have flipped it false after appending
   observation messages — the loop then continues with them.)
6. Else fire `BeforeToolCall` per call (veto via `ErrSkipToolCall`, rewrite
   via `ev.Args`), execute concurrently (bounded 3), fire `AfterToolCall`
   per call, append tool result messages in order, loop to 2.
7. Budget: when `MaxToolCalls` is reached, subsequent `Generate` calls use
   `WithToolChoice(llmhub.NoToolChoice())` to force a final answer, and one
   user-role notice is appended once (after the last tool result, before
   the forced inference) instructing the model to synthesize the gathered
   evidence now and state plainly anything undetermined; if the
   model still returns tool calls, execute none and finish with
   `resp.Text()` — the loop can never spin forever. `MaxSteps` exceeded →
   `Status = StatusBudgetExhausted` returned as `ErrBudgetExhausted`
   (with partial trace in `Answer.Messages` where practical). Multiple
   limit dimensions always apply: tool calls + steps + dup-call detection
   (§5.2) + the caller's ctx deadline (`docs/references.md` §2).
8. Accumulate `resp.Usage` per step into `Answer.Usage`.
9. Context cancellation aborts promptly; ctx errors returned unwrapped.

### 5.5 Multi-turn conversations

`Ask` is one-shot; `Conversation` chains Asks over a shared history —
follow-up questions see prior turns, tool results, and answers.

```go
// Conversation runs a sequence of Asks over a shared message history.
// Obtain from Agent.Conversation. Not safe for concurrent use: an Ask
// while another is in flight fails fast with ErrConversationBusy.
type Conversation struct { /* … */ }

// Conversation starts a multi-turn session. The opts set the system
// message (prepended once) and the default per-turn options; opts passed
// to Ask apply only to that turn.
func (a *Agent) Conversation(opts ...Option) *Conversation

func (c *Conversation) Ask(ctx context.Context, query string, opts ...Option) (*Answer, error)
func (c *Conversation) AskStream(ctx context.Context, query string, opts ...Option) (<-chan Event, error)

func (c *Conversation) Messages() []*llmhub.Message     // history so far; read-only, do not mutate
func (c *Conversation) TotalUsage() llmhub.UsageMetadata // summed across all turns
func (c *Conversation) Reset()                           // drop history; system message is kept
```

Semantics:

- Each `Ask` runs the full loop (§5.4) with
  `messages = [system?] + history + user(query)`; when the run completes,
  its messages (user, assistant, tool steps) are appended to the history.
  One `Ask` = one **Turn** in the adopted vocabulary (§5.1).
- **Prefix-stability invariant** (`docs/references.md` §5 — the sacred rule):
  history is **append-only**. The loop never edits, reorders, or rewrites
  prior messages within a run or across turns; the system message is fixed
  for the conversation's lifetime; dynamic facts (time, cwd, context
  updates) belong in the newest user message — never the system prompt.
  This preserves provider prompt caching across turns. **History compaction
  is the sanctioned mutation, done as a projection**: a `RebuildContext`
  hook derives a shorter LLM view (e.g. older turns summarized) while the
  Conversation log stays complete — no need to start a fresh
  Conversation (§5.3).
- `Answer.Messages` contains only that run's messages; the full transcript
  is `Conversation.Messages()`. Budgets (`MaxToolCalls`, `MaxSteps`,
  dup-call detection) apply per `Ask`, not per conversation;
  `TotalUsage()` aggregates for cost display across turns.
- Prompt caching: long sessions are ideal llmhub cache candidates — pass
  `WithGenerateOptions(llmhub.WithCacheKey("session-id"))` (per llmhub's
  multi-turn guidance) so repeated prefixes are served from cache.
- Hooks observe the full message list each step (`BeforeGenerate`
  `ev.Messages` includes history), so custom context injection keeps
  working. `BeforeGenerate` message-append is the **steering seam**
  (`docs/references.md` §2): a thread-safe queue drained by the hook
  implements steer/inject; ctx cancellation implements cancel.
- Presets: get a conversation via the escape hatch —
  `searchAgent.Agent().Conversation(...)` or
  `bashAgent.Agent().Conversation(...)`; `search.Sources()` still works on
  each turn's `Answer`. Typed preset conversations are deferred (§17).

### 5.6 Answer

```go
type ToolResult struct {
    Name     string          // tool name
    ID       string          // provider call ID
    Args     json.RawMessage // arguments as executed (post-rewrite)
    Output   string          // result payload
    Err      error           // handler error, if any
    Duration time.Duration
}

type Answer struct {
    Text      string               // final assistant text (the final step's text)
    ReasoningText string           // concatenated reasoning blocks across steps (llmhub ReasoningContent)
    Steps     int                  // model inferences (one step = one inference + its tool executions)
    ToolCalls int                  // executed tool invocations (skips not counted)
    Status    AnswerStatus         // StatusCompleted | StatusBudgetExhausted | StatusStopped (graceful: hook skip or loop detection)
    Usage     llmhub.UsageMetadata // summed across steps (tokens, cost)
    Timing    *llmhub.TimingMetrics // aggregated llmhub telemetry: Total/Chunks summed across steps; Connected/FirstByte/FirstToken from the first step; nil only when the provider bypassed llmhub's Client wrapper
    Tools     []ToolResult         // every execution, in order — the trace
    Messages  []*llmhub.Message    // full transcript
}
```

The trace makes every agent inspectable and enables preset-specific
extraction (§6, §8) without core knowing about any domain.

### 5.7 Minimal generic example (no domain)

```go
clock := agentloop.Tool{
    Def: llmhub.NewTool("clock", "Current time in RFC 3339", map[string]any{
        "type": "object", "properties": map[string]any{},
    }),
    Run: func(ctx context.Context, args json.RawMessage) (string, error) {
        return time.Now().UTC().Format(time.RFC3339), nil
    },
}

llm, _ := llmhub.New("openai", os.Getenv("OPENAI_API_KEY"),
    llmhub.WithModel("gpt-4o-mini"))
ag := agentloop.MustNew(llm,
    agentloop.WithSystemPrompt("You keep exact time."),
    agentloop.WithTools(clock))

ans, err := ag.Ask(ctx, "What time is it in UTC right now?")
```

### 5.8 Skills (progressive disclosure)

Skills are repeatable *procedures* — folders of instructions the model can
discover and load on demand (`docs/references.md` §7). **Skills prepare,
tools execute**: skills are not a second agent, not native tools, and not
playbooks stuffed into the system prompt. A skill folder contains:

```
my-skill/
  SKILL.md        # required: YAML frontmatter (name, description) + markdown body
  references/     # optional deeper docs, loaded individually (L2)
  assets/         # optional templates/data (L2)
  scripts/        # optional executable helpers (L3 — run via an execution tool)
```

**The L0–L3 invariant** (violating it means you don't have a skill system):

| Level | What | Cost |
| --- | --- | --- |
| L0 Advertise | `- name: description` line per skill in the system prompt catalog | ~100 tokens/skill, every call |
| L1 Load | full `SKILL.md` body, only when the model decides it matches | <5k tokens, on demand |
| L2 Resources | one referenced file on demand | on demand |
| L3 Scripts | executed via an execution tool (bash preset); only the *output* enters context — never script source | output only |

```go
// Skill is one validated skill folder.
type Skill struct {
    Name        string   // == directory name; ^[a-z0-9]+(-[a-z0-9]+)*$, 1-64 chars
    Description string   // 1-1024 chars, cap enforced; write imperatively: "Use when…"
    Body        string   // SKILL.md markdown below the frontmatter
    Dir         string   // resolved absolute path
    LinkedFiles []string // relative paths under references/ and assets/ (L2), precomputed so the model never needs to ls
    ScriptFiles []string // relative paths under scripts/ (L3), advertised in skill_read for execution; never readable through it
}

// SkillIssue reports a skipped (fail-closed) skill directory and why.
// Invalid skills are logged to the caller via issues and OMITTED — never
// an error, never a crash (fail-closed loading).
type SkillIssue struct {
    Dir    string
    Name   string // skill folder name; set for every rejected skill folder
    Reason string // e.g. "bad frontmatter" (invalid YAML), "name mismatch", "empty body", "unicode steganography stripped"
}

// LoadSkills scans dirs in precedence order (first occurrence of a name
// wins; later duplicates are reported as issues, never merged). Validation:
// YAML frontmatter (yaml.v3); name regex + equality with dirname; description
// 1-1024 post-fold; body non-empty; NUL-byte check; strip Unicode Tags
// (U+E0001–U+E007F) and bidi overrides (steganography defense).
func LoadSkills(dirs ...string) ([]Skill, []SkillIssue, error)

WithSkills(dirs ...string) Option     // load + enable on an Agent/Conversation
WithPinnedSkills(names ...string) Option // bodies of ≤2 named skills inlined into the system prompt at construction (the sanctioned always-on exception)
```

Mechanics:

- **Frontmatter.** `SKILL.md` frontmatter is parsed as real YAML with
  `gopkg.in/yaml.v3`: folded (`>`/`>-`/`>+`) and literal (`|`/`|-`) block
  scalars, quoted multi-line values, comments, nested maps (`metadata:`),
  and block/flow sequences all load, and unknown keys are ignored. `name`
  and `description` must be strings — a bool, number, sequence, or map is
  `"bad frontmatter"`, as is a non-map document or invalid YAML (the
  stable reason string now means genuinely malformed YAML). A UTF-8 BOM
  and CRLF/CR endings are normalized, the block is capped at 64 KB before
  parsing, and the description cap is measured on the post-fold string.
  Every rejected skill folder carries its folder name in
  `SkillIssue.Name`, so headless consumers can tell which skills dropped.
- **Catalog (L0):** `WithSkills` appends a `## Skills` section to the
  system prompt — one `- name: description` line per skill, byte-stable
  for the lifetime of the Agent/Conversation (frozen at construction;
  filesystem changes affect the *next* construction, never an in-flight
  run — the prefix-stability invariant, §5.5). Descriptions are the
  router: imperative "Use when…" phrasing; near-miss negatives ("Don't
  use for…") belong in the body, not the description.
- **Loading (L1/L2):** when skills are enabled the framework registers a
  `skill_read` tool: `{"name": string, "file": string (optional)}` → the
  skill body, or the named file under `references/`/`assets/`. Bodies and
  files arrive as **tool results, never prompt injection** (skill loading
  via prompt injection busts the prefix cache mid-run — `docs/references.md`
  §5/§13.1). `file` is path-jailed: absolute paths, `~`, `..`, and symlink
  escapes rejected (EvalSymlinks + Rel); `offset`/`limit` parameters are
  **refused by design** — models read page 1 and skip the rest; NEVER
  paginate instructional reads. Linked files are listed in the tool result
  so the model doesn't `ls`.
- **Scripts (L3):** no dedicated execution tool. Skill scripts are run by
  whatever execution tool the agent already has (the bash preset's `bash`
  tool); only command output enters context. A skill that ships
  `scripts/` gains a footer in its `skill_read` body result and in the
  pinned-skill rendering, disclosing the absolute base directory, the
  relative-path semantics, and a run-don't-read listing of its `scripts/`
  entries, so bodies can reference them while the scripts stay unreadable
  through `skill_read` (script-less skills render exactly as before).
  Agents without an execution tool (e.g. the search preset) simply never
  run scripts — their skills are procedures for *how to search/fetch*, not
  shell scripts.
- **Trust:** every loaded skill's instructions become authoritative for
  the model. v0 performs load-time sanitization only (NUL, Unicode Tags,
  bidi); install-time scanning, provenance hashes, and quarantine are
  phases 3–4 (§1 non-goals). `allowed-tools` ceilings for skills arrive
  with phase 4.
- Skills work with any Agent, both presets, and Conversations; skill loads
  are ordinary tool calls, so they appear in `Answer.Tools`, traces, and
  `AfterToolCall` hooks like any other execution.

## 6. Search Preset — `agentloop/search`

The first preconfigured agent: an `agentloop.Agent` wired with two tools —
`web_search` backed by Serper and `fetch_page` backed by Serper's scrape
endpoint (§9.4) — plus the default research-assistant prompt (§7) and
source extraction. The model searches when it needs current information and
fetches a result's full page when snippets are insufficient.

```go
package search // import "github.com/smhanov/agentloop/search"

type Option = agentloop.Option // core and preset options compose together

// New returns a search agent: Agent + web_search + fetch_page tools
// + default prompt. Serper key resolution: WithSerperKey →
// SERPER_API_KEY env → none (the tools return graceful "not configured"
// error results; the agent still works ungrounded).
func New(llm *llmhub.Client, opts ...Option) (*Agent, error)

type Agent struct { /* wraps *agentloop.Agent; immutable */ }

func (a *Agent) Ask(ctx context.Context, query string, opts ...Option) (*Answer, error)
func (a *Agent) AskStream(ctx context.Context, query string, opts ...Option) (<-chan agentloop.Event, error)
func (a *Agent) Agent() *agentloop.Agent // escape hatch: full core API

type Answer struct {
    *agentloop.Answer        // embedded core trace answer
    Sources []Source         // deduped by URL, first-seen order, indexed from 1
}

type Source struct {
    Title     string
    URL       string
    Snippet   string
    Date      string // when Serper provides it (always set for news results)
    Publisher string // news source name, when Serper provides it
    Fetched   bool   // true when the model fetched this page's full content
}

// Sources extracts sources from any core Answer's web_search tool results,
// marking entries whose URL was also passed to fetch_page. Pure function
// over the trace — used by Answer and by streaming consumers on EventDone.
func Sources(ans *agentloop.Answer) []Source
```

Preset options (in addition to all core options):

```go
WithSerperKey(key string) Option             // override env var
WithSerperBaseURL(url string) Option         // default "https://google.serper.dev"; test hook
WithSerperHTTPClient(c *http.Client) Option  // default 10s-timeout client
WithSearcher(s Searcher) Option              // replace search backend (§9)
WithFetcher(f Fetcher) Option                // replace fetch backend (§9)
WithSerperRetries(n int) Option              // transparent 429 retries with jittered backoff; default 1, max 3
WithMaxResults(n int) Option                 // Serper "num"; default 10, clamped 1–10
WithFetchLimit(n int) Option                 // max chars of page text per fetch; default 20000; 0 = unlimited
WithNoFetch() Option                         // drop the fetch_page tool (search-only agent)
```

Budget note: searches and fetches both count against `MaxToolCalls`; the
preset raises the default to **8** because effective research typically
alternates search → fetch → answer. A fetch of a URL that 404s (target-site
errors propagate from Serper, §9.4) returns
`{"error": "fetch failed: <status>"}` so the model can try another result.

Skills compose: `WithSkills` adds the catalog to the prompt and the
`skill_read` tool alongside `web_search`/`fetch_page` (§5.8) — e.g. a
"sec-filings" skill teaching the model how to search for and read
regulatory filings.

**Domain neutrality:** the preset's default prompt contains no
domain-specific guidance. Domain conventions (as-of dates on figures,
primary-source preferences, JSON response schemas, citation-style
overrides) are supplied by the caller — in the query text or via
`WithSystemPrompt`. The test corpus in §14 demonstrates this pattern. A
user prompt may override citation style (e.g. "do not include reference
citations") — `Answer.Sources` is populated regardless, since it is
built from tool results, not text markers.

### Minimal usage

```go
import (
    "github.com/smhanov/llmhub"
    "github.com/smhanov/agentloop"
    "github.com/smhanov/agentloop/search"
    _ "github.com/smhanov/llmhub/providers/openai"
)

llm, _ := llmhub.New("openai", os.Getenv("OPENAI_API_KEY"),
    llmhub.WithModel("gpt-4o-mini"))

agent, err := search.New(llm,
    search.WithSerperKey(os.Getenv("SERPER_API_KEY")))
if err != nil { panic(err) }

ans, err := agent.Ask(ctx, "What Go version was released most recently, and when?")
if err != nil { panic(err) }

fmt.Println(ans.Text)
for i, s := range ans.Sources {
    fmt.Printf("[%d] %s — %s\n", i+1, s.Title, s.URL)
}
fmt.Printf("%.4f USD, %d searches\n", ans.Usage.Cost, ans.ToolCalls)
```

## 7. Search Preset Internals

### 7.1 Default system prompt

> You are a research assistant with access to two tools backed by Google:
> `web_search` (Serper Google search) and `fetch_page` (full page text).
> Decide for each request whether search is needed. When snippets are
> insufficient to answer, call `fetch_page` on the most promising result URL
> and read the full content — but fetch sparingly; snippets answer most
> questions. When you search, cite the sources you actually used inline as
> `[n]`, where `n` is the 1-based index of the source in the order they were
> first returned to you. If the search results do not answer the question,
> say so plainly rather than guessing. Provide the final answer only when it
> is grounded in the search results or clearly within your own knowledge;
> never fabricate URLs, statistics, or dates.

`Sources()` returns first-seen, URL-deduped sources in the same order (with
`Fetched` marking pages the model read in full), so `[n]` in the text aligns
with `Sources[n-1]`.

### 7.2 Tool declarations

Declared via `llmhub.NewTool` and registered with `WithTools` on every call:

```go
llmhub.NewTool("web_search",
  "Search Google via Serper for current information. Use this whenever the "+
  "answer depends on facts that may be recent, volatile, or outside your "+
  "training data. Use type=\"news\" for time-sensitive current events; the "+
  "default general search otherwise. Formulate concise keyword queries; "+
  "call again with a revised query if the results are insufficient.",
  map[string]any{
    "type": "object",
    "properties": map[string]any{
      "q":    map[string]any{"type": "string", "description": "the search query"},
      "num":  map[string]any{"type": "integer", "minimum": 1, "maximum": 10,
                             "description": "number of results (default 10)"},
      "type": map[string]any{"type": "string", "enum": []string{"search", "news"},
                             "description": "\"search\" (default) or \"news\" for recent events"},
    },
    "required": []string{"q"},
  })

llmhub.NewTool("fetch_page",
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
```

The `web_search` handler decodes `{"q", "num", "type"}`, calls the
`Searcher` (`type:"news"` routes to Serper's `/news` endpoint, §9.3), and
returns the trimmed payload. The `fetch_page` handler validates the URL
(absolute `http`/`https` only), calls the `Fetcher`, truncates the text to
`WithFetchLimit` (default 20000 chars) using **head+tail truncation** —
first 10000 + `"<N characters elided>"` + last 10000 — never naive
head-only (`docs/references.md` §13.5: head-only loses the ends of
commands, paths, and conclusions), and returns compact JSON:
`{"url": "...", "title": "...", "text": "..."}`. Malformed arguments (either
tool) produce a `{"error": "invalid arguments"}` tool result (counts
against budget). Both count against `MaxToolCalls`.

## 8. Bash Coding Preset — `agentloop/bash` (mini-swe-agent compatible)

The second built-in preset — a coding agent that solves tasks by executing
shell commands in a workspace until it reports completion. It behaves the
same way as [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)
(the Princeton/Stanford "100-line agent"): same protocol, same templates,
same limits and exit statuses, same submission mechanism — reimplemented on
the unmodified agentloop core using only `Tool`, `Hooks`, and `Options`.
This preset exists to **enforce generality**: the core loop must express a
long-running command-execution agent just as naturally as a one-shot search
agent.

### 8.1 API

```go
package bash // import "github.com/smhanov/agentloop/bash"

type Option = agentloop.Option // core and preset options compose together

// New returns a coding agent that executes bash commands in an
// Environment until the task is submitted or a limit is hit.
// Default protocol is mini-swe-agent's text-action protocol (§8.3).
func New(llm *llmhub.Client, opts ...Option) (*Agent, error)

type Agent struct { /* wraps *agentloop.Agent; immutable */ }

func (b *Agent) Ask(ctx context.Context, task string, opts ...Option) (*Answer, error)
func (b *Agent) AskStream(ctx context.Context, task string, opts ...Option) (<-chan agentloop.Event, error)
func (b *Agent) Agent() *agentloop.Agent // escape hatch: full core API

type Answer struct {
    *agentloop.Answer
    ExitStatus string // "Submitted", "LimitsExceeded", "TimeExceeded", "RepeatedFormatError"
    Submission string // final output captured by the submit marker, when Submitted
}
```

Preset options (in addition to all core options):

```go
WithEnvironment(env Environment) Option       // default LocalEnvironment("", 30s)
WithStepLimit(n int) Option                   // max LLM steps; 0 = unlimited (default 0, like mini's default config)
WithCostLimit(usd float64) Option             // graceful stop when cumulative Usage.Cost exceeds; 0 = unlimited
WithWallTimeLimit(d time.Duration) Option     // graceful stop on wall-clock time; 0 = unlimited
WithMaxConsecutiveFormatErrors(n int) Option  // text protocol; default 3, 0 = unlimited
WithSubmitMarker(s string) Option             // default "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
WithOutputLimit(n int) Option                 // observation truncation threshold; default 10000 chars
WithTemplates(system, task, observation string) Option // override default templates (§8.5)
WithToolProtocol() Option                     // native tool calling instead of the text protocol
WithApprover(fn func(ctx context.Context, command string) error) Option
                                              // approval gate: called before every command
                                              // execution; a returned error vetoes with a
                                              // {"error": "command not approved: …"}
                                              // observation. nil (default) runs unattended
                                              // (mini's "yolo"); a human prompt implements
                                              // mini's "confirm" mode — blocking in
                                              // BeforeToolCall/hook is sanctioned (§5.3)
```

`Ask` returns a `*Answer` (not an error) for every exit status — limits
and format failures are *results* in mini-swe-agent, not errors. Only
infrastructure failures (LLM transport, ctx) return errors.

Skills compose (§5.8): `WithSkills` works on the bash agent, where L3
scripts are runnable — a loaded skill's `scripts/` helpers execute through
the `bash` tool with only their output entering context.

### 8.2 Environment abstraction

```go
// Environment executes shell commands for the bash preset.
type Environment interface {
    Execute(ctx context.Context, command string) (ExecResult, error)
}

type ExecResult struct {
    Output        string // combined stdout+stderr
    ReturnCode    int
    ExceptionInfo string // non-empty when execution itself failed (e.g. timeout)
}

// NewLocalEnvironment returns an Environment running commands on the local
// machine. Only LocalEnvironment ships — container environments (Docker,
// podman, …) are caller-implemented via this interface; the README
// documents a Docker example pattern (§17 decisions).
func NewLocalEnvironment(cwd string, env []string, timeout time.Duration) Environment
```

`LocalEnvironment` mirrors mini-swe-agent's `LocalEnvironment`:

- `shell=True` equivalent: runs `bash -c <command>` (fallback `sh`).
- Working directory (`cwd`, default `os.Getwd`), per-environment extra env
  vars merged over `os.Environ()`, per-command timeout (default 30s).
- **Process-group kill on timeout**: the child is started with
  `Setpgid` (new session); on timeout the whole group gets
  `SIGKILL` (`syscall.Kill(-pgid, SIGKILL)`) so no orphans remain —
  exactly mini's `_run` semantics. Output collected so far is preserved in
  `ExceptionInfo`+`Output`.
- Non-zero return codes are **not** errors: they flow to the model as
  observations (the agent iterates on failing commands, like mini).

### 8.3 Action protocols

**Text protocol (default) — mini-swe-agent wire compatibility.** The model
responds with plain text containing exactly one fenced bash block:

````
THOUGHT: reasoning about the next step…

```mswea_bash_command
sed -i 's/old/new/g' main.go
```
````

Implemented as an `AfterGenerate` hook:

1. Parse `resp.Text()` for fenced blocks tagged `mswea_bash_command`
   (`` ```bash `` accepted as alias). Exactly one block → the command.
2. Execute via `Environment.Execute`; format the observation (§8.4); append
   it as a **user message**; set `ev.Finish = false` (the finish-veto from
   §5.3 — this is what keeps the loop going without native tool calls).
3. Zero blocks or more than one → **format error**: append mini's
   `format_error_template` guidance as a user message, still `Finish =
   false`; increment the consecutive counter (reset on any clean step);
   at `MaxConsecutiveFormatErrors` stop with
   `ExitStatus = "RepeatedFormatError"`.
4. Submit detection (also used by the tool protocol): if the command's
   output's first line, stripped, equals the submit marker and
   `ReturnCode == 0`, stop with `ExitStatus = "Submitted"` and
   `Submission` = the remaining output lines (mini's `_check_finished`).

**Tool protocol (`WithToolProtocol`).** The same behavior expressed with a
native `bash` tool (`{"command": string}`) — for providers/models with
reliable tool calling. The loop's natural termination (model stops calling
tools) replaces the format-error machinery; submit-marker detection and all
limits stay identical. The handler and the hook share `formatObservation`
and submit detection; only the transport differs.

### 8.4 Observation formatting (mini-swe-agent `observation_template`)

Every execution result is rendered for the model with identical semantics
to mini's default template:

```
<exception>…</exception>        (only when ExceptionInfo non-empty)
<returncode>N</returncode>
<output>…</output>              (when len(Output) < 10000)
```

When output is ≥ 10000 chars, instead of `<output>`:

```
<warning>
The output of your last command was too long. … (mini's guidance: use
head/tail/sed, more selective grep, redirect to a file)
</warning>
<output_head>first 5000 chars</output_head>
<elided_chars>N characters elided</elided_chars>
<output_tail>last 5000 chars</output_tail>
```

### 8.5 Default templates

Ported from mini-swe-agent's `config/default.yaml`, rendered with Go
`text/template` instead of Jinja2 (same variables where applicable):

- **System template:** "You are a helpful assistant that can interact with
  a computer… exactly ONE bash code block with ONE command… include a
  THOUGHT section…", with the `<format_example>` block.
- **Instance/task template:** "Please solve this issue: {{.Task}}", the
  recommended workflow (analyze → reproduce → fix → verify → edge cases →
  submit), the persistence rules (every action runs in a new subshell),
  `<system_information>` filled from `runtime.GOOS`/`GOARCH`/hostname, the
  example response, and the useful-command examples (heredoc file creation,
  sed edits, `nl -ba | sed -n` viewing).
- **Observation template:** §8.4.
- **Format-error template:** mini's two variants — the finish_reason
  truncation hint ("respond more concisely…") and the "found N actions,
  exactly one expected" guidance.

macOS `sed -i ''` note and the `TRIPLEBACKTICKSBASH` escape hatch are
included for compatibility (§17).

### 8.6 Limits and exit statuses (mapping table)

| mini-swe-agent | agentloop implementation |
| --- | --- |
| `run(task)` message loop | core `Ask` loop; continuation via AfterGenerate finish-veto |
| `query()` `step_limit` check | `WithStepLimit` enforced in BeforeGenerate hook → `ev.Skip` graceful stop, `ExitStatus="LimitsExceeded"` |
| `cost_limit` (stop after exceeding) | same hook, cumulative `Usage.Cost` across steps |
| `wall_time_limit_seconds` | same hook, monotonic clock |
| `max_consecutive_format_errors` (3) | text-protocol hook counter → `"RepeatedFormatError"` |
| `TimeExceeded` exit | `"TimeExceeded"` |
| `Submitted` + submission (env `_check_finished`) | submit-marker detection (§8.3.4) |
| `handle_uncaught_exception` → exit status | infrastructure errors still return Go errors (only deviation; transport failures are not agent results) |
| trajectory JSON (`mini-swe-agent-1.1`) | **native format** (§17 decisions): `bash.Answer` JSON serialization — messages, trace, exit status, usage; no mini-format adapter (deferred indefinitely) |
| `output_path` trajectory save | caller-side via `OnDone` hook (example in README) |

Note: preset limits stop **gracefully** (`ev.Skip`/finish with a result);
the core's `MaxSteps`/`ErrBudgetExhausted` remains the hard safety net and
is set by the preset to `max(StepLimit, 2*MaxToolCalls+2)` automatically.

### 8.7 Minimal usage

```go
import (
    "github.com/smhanov/llmhub"
    "github.com/smhanov/agentloop/bash"
)

llm, _ := llmhub.New("zai", os.Getenv("ZAI_API_KEY"),
    llmhub.WithBaseURL("https://api.z.ai/api/coding/paas/v4"),
    llmhub.WithModel("glm-5.3"))

agent, _ := bash.New(llm,
    bash.WithEnvironment(bash.NewLocalEnvironment(
        "/tmp/workspace", nil, 60*time.Second)),
    bash.WithStepLimit(20),
    bash.WithCostLimit(1.5))

ans, err := agent.Ask(ctx, "Add a README.md describing this project.")
// ans.ExitStatus == "Submitted"; ans.Submission holds the final output
// ans.Messages is the full trajectory; ans.Tools the command trace
```

## 9. Serper Integration: Search & Fetch

### 9.1 Interfaces

```go
type SearchRequest struct {
    Query string
    Num   int
    Type  SearchType // SearchGeneral (default) | SearchNews
}

type Searcher interface {
    Search(ctx context.Context, req SearchRequest) ([]Source, []string, error)
    // (results, relatedQueries, err)
}

type FetchRequest struct {
    URL      string
    Markdown bool // request markdown instead of/in addition to plain text
}

type Page struct {
    URL         string
    Title       string // metadata.title
    Description string // metadata.description / og:description
    Text        string // extracted plain text
    Markdown    string // extracted markdown, when requested
    Credits     int    // credits charged by Serper
}

// Fetcher retrieves the readable content of a web page.
type Fetcher interface {
    Fetch(ctx context.Context, req FetchRequest) (*Page, error)
}
```

`serperClient` implements both. `WithSearcher` / `WithFetcher` inject
alternatives (caching wrappers, different backends, test doubles). The
preset depends only on the interfaces.

### 9.2 Search: HTTP details

- `POST {baseURL}/search` (general) or `POST {baseURL}/news`
  (`Type: SearchNews`), JSON body `{"q": "...", "num": N}`.
- **Rate limits (429): retried transparently** with jittered exponential
  backoff inside the client — `WithSerperRetries(n)` (default 1, max 3;
  mirroring llmhub's `WithRetryOnStatus` pattern). Rate limiting is
  transport business, not agent business: the model sees a failure only
  after retries are exhausted.
- Headers: `X-API-KEY: <key>`, `Content-Type: application/json`.
- Timeout from the option-injected `http.Client` (default 10s); request
  carries `ctx`.
- Non-200 (after retries) → `*SerperError` (status + body excerpt) in the
  trace; the model receives `{"error": "search failed: <status>"}` and may
  retry with a different query (counts against budget).

### 9.3 Response mapping

Trimmed, token-friendly JSON handed to the model:

```json
{
  "results": [
    {"title": "...", "link": "https://...", "snippet": "...", "date": "..."}
  ],
  "related": ["up to 5 related queries, no link"]
}
```

| Serper field | Mapped to | Notes |
| --- | --- | --- |
| `organic[]` | `results[]` | capped at `num`; `date` only when present |
| `news[]` (on `/news`) | `results[]` | `title`, `link`, `snippet`, `date` always set; `source` → `Source.Publisher` |
| `knowledgeGraph.title/description` | prepended pseudo-result | `websiteLink` when present |
| `peopleAlsoAsk[]` | dropped | low signal without answers |
| `relatedSearches[]` | `related` (≤5) | lets the model reformulate |
| everything else | dropped | |

Empty result set is valid, non-error: `{"results": []}`.

### 9.4 Fetch: Serper `/scrape` endpoint

Verified against the live API (same host and key as `/search`):

- `POST {baseURL}/scrape`, JSON body `{"url": "https://…"}`; add
  `"includeMarkdown": true` to also receive markdown.
- Response fields: `text` (extracted plain text — headlines, paragraphs,
  boilerplate stripped), `markdown` (only when requested), `metadata`
  (`title`, `description`, `og:*`), `credits` (scrapes cost **2 credits**,
  searches 1 — relevant for API-plan budgeting, not `Usage.Cost`).
- **Target-site errors propagate**: a nonexistent page returns
  `{"message":"Page not found.","statusCode":404}` from Serper. The client
  maps non-200s (both transport and target-echoed) to `*SerperError`; the
  model receives `{"error": "fetch failed: <status>"}` and may try another
  result (counts against budget).
- Same option-injected `http.Client` (default 10s) and `ctx` as search.

## 10. Error Handling

| Situation | Behavior |
| --- | --- |
| nil llm client at `New` | error from constructor (no panic; `MustNew` for examples) |
| tool handler error | tool result `{"error": ...}`; loop continues |
| unknown tool name | tool result `{"error": "unknown tool"}`; counts against budget |
| search without Serper key | tool result `{"error": "web search is not configured"}`; agent answers ungrounded |
| fetch of a 404/unreachable page | tool result `{"error": "fetch failed: <status>"}`; model may try another result (§9.4) |
| fetch of a non-http(s)/malformed URL | tool result `{"error": "invalid arguments"}` |
| hook error (non-sentinel) | run aborts; wrapped error naming the hook |
| `ErrSkipToolCall` from BeforeToolCall | tool skipped, not executed, not counted |
| bash limit hit (step/cost/wall/format) | `bash.Answer` with ExitStatus — not an error |
| command rejected by approver | observation `{"error": "command not approved: …"}`; model may adjust — not an error |
| `Ask` on a busy Conversation | `ErrConversationBusy` |
| command non-zero return code | observation to the model — not an error |
| environment failure (timeout, spawn error) | observation via `ExceptionInfo` — not an error |
| llmhub Generate error | propagate immediately, wrapped |
| budget/step ceiling | `ErrBudgetExhausted` / `EventError`; `Answer.Status = StatusBudgetExhausted` |
| repeated identical tool calls with identical output (dup limit) | real output with warning appended, then graceful stop with `Status = StatusStopped` on the next identical call |
| stale stream (no chunk for `WithStreamWatchdog`) | run aborted with `ErrStreamStale` (Hermes 90s watchdog pattern) |
| ctx canceled | `ctx.Err()` unwrapped |

Sentinels: `ErrInvalidInput`, `ErrBudgetExhausted`, `ErrSkipToolCall`,
`ErrNoLLMClient`, `ErrConversationBusy`, `ErrStreamStale`.

## 11. Idiomatic Go Checklist (release gate)

- [ ] `context.Context` honored end-to-end (tools, hooks, HTTP, exec, LLM calls)
- [ ] No panics exported; `MustNew` clearly documented for examples/tests
- [ ] All sentinels reachable via `errors.Is`; wraps use `%w`
- [ ] No library-side logging; hooks are the only observability surface
- [ ] Concurrency-safe types; `-race` clean tests
- [ ] Doc comments on 100% of exported symbols; `gofmt`/`go vet` clean
- [ ] Table-driven tests; deterministic (fake clocks/searchers/environments/providers)
- [ ] Exported-API budget enforced in CI, gptme `arewetiny`-style: a test
      (or CI step) fails when the count of exported symbols exceeds a fixed
      ceiling — the architecture's "small surface" promise, made testable
      (`docs/references.md` §12)

## 12. Testing Strategy

All offline tests run without network (`go test ./...`).

1. **Core loop:** register a scripted fake provider with
   `llmhub.MustRegisterProvider("fake", …)` replaying canned responses;
   fake tools (clock/echo). Cases: no-tool answer (1 step); single tool
   call; two sequential turns; parallel calls in one turn (order preserved);
   handler error propagation; unknown tool; arg rewrite in `BeforeToolCall`;
   veto via `ErrSkipToolCall`; `ev.Skip` graceful stop; **finish veto**
   (AfterGenerate flips `Finish` to false after appending a user message —
   loop continues; flips to true on tool-call response — stops without
   executing); budget exhaustion → forced final via `NoToolChoice`;
   `MaxSteps` ceiling; usage summation; ctx cancellation mid-tool;
   hook-error abort; per-call options override; **parallel tools** —
   randomized completion delays prove result messages append in call order
   while `AfterToolCall` fires in completion order, `Sequential=true`
   tools execute strictly in order amid parallel-eligible ones, and
   `WithParallelTools(1)` executes sequentially; **loop detection** —
   dup-call warning appended to the real output at the limit, graceful
   `StatusStopped` on the next identical call with identical output,
   counter resets on any differing call or changed output,
   `Repetitive: true` exempts a polling tool, and parallel calls are
   accounted in call order; **stream watchdog** —
   stalled fake stream aborts with `ErrStreamStale`, nil when disabled;
   **completion status** — StatusCompleted/BudgetExhausted/Stopped across
   the graceful-stop paths; **conversations** —
   history accumulates across Asks (turn 2 sees turn 1's tool results),
   system message prepended exactly once, budgets reset per Ask,
   `TotalUsage()` aggregates, `Reset()` keeps the system message,
   `ErrConversationBusy` on overlapping Asks, per-turn option overrides
   don't leak into the next turn, **prefix stability** — the loop never
   mutates prior history; **context rebuild** — a RebuildContext hook that
   compresses older turns sees the full transcript in `ev.Messages`,
   replaces `ev.Context`, and the Conversation log is unchanged; the fake
   provider captures the exact per-step context sent.
2. **AskStream (llmhub-parity streaming):** event sequence
   `Start → BeforeGenerate → AfterGenerate → [BeforeToolCall → AfterToolCall]* → AnswerDelta* → Done`;
   channel closes after Done/Error; reasoning deltas forwarded from all
   turns (Step-tagged); interim content deltas from tool-call steps stream
   as `EventAnswerDelta` but are excluded from final `Answer.Text`;
   `Event.Timing` snapshots on delta events and the complete set on Done;
   `Answer.ReasoningText`/`Answer.Timing` populated.
3. **Serper clients (search + news + scrape):** `httptest.Server` — happy
   paths, knowledgeGraph merge, `/news` mapping (Publisher/Date),
   `type` routing, empty results, markdown option and `credits` capture on
   scrape, target-site 404 propagation mapped to `*SerperError`,
   **429 → jittered retry then `*SerperError` when exhausted** (retry
   count honored), non-200, timeout, header/body assertions for all
   endpoints.
4. **Search preset:** fake provider + fake `Searcher`/`Fetcher` — source
   dedup and first-seen order across multiple searches; `Sources()` from
   trace incl. `Fetched` marking when a `fetch_page` URL matches a source;
   unconfigured key graceful path; `num` clamping; fetch truncation at
   `WithFetchLimit` with elision marker; search→fetch→answer trajectory;
   `WithNoFetch()` drops the tool.
5. **Bash preset — parser/observations:** fenced-block parser (one block,
   none, multiple, ```bash alias, nested fences) — **conformance-tested
   against mini-swe-agent's MIT parser test corpus** (ported, §17
   decisions), including the `TRIPLEBACKTICKSBASH` escape hatch;
   observation formatting incl. the ≥10000-char head/tail/elided path;
   format-error template selection; consecutive-format-error counter and
   reset.
6. **Bash preset — environment:** `LocalEnvironment` in a temp dir —
   stdout/stderr merge, return codes, env merge, cwd, timeout kills the
   whole process group (spawn a child that outlives a killed parent;
   assert no orphan), partial output preserved on timeout.
7. **Bash preset — end to end:** fake provider scripting a mini-style
   trajectory (command → observation → command → submit marker) against a
   real temp-dir `LocalEnvironment`; assert `ExitStatus="Submitted"`,
   `Submission`, message shapes (user-message observations), cost/step
   limits firing with graceful `bash.Answer`s; tool-protocol variant of the
   same scenario; approver veto produces an observation and the run
   continues. **P0 blocking tier** (fail the build first, run before all
   else — GoClaw NLB pattern): the loop can never exceed `MaxSteps` or
   `MaxToolCalls` even with a malicious/malformed provider script,
   identical-call-and-identical-output loops always terminate (warn at
   the limit, stop pre-execution on the next repeat), and no tool
   result exceeds its byte cap.
8. **JSON helpers:** fenced/prefixed/nested JSON extraction cases.
9. **Skills:** validation table (bad frontmatter, name/dirname mismatch,
   description caps, empty body, NUL, Unicode Tags/bidi stripped — each
   fail-closed to a `SkillIssue`, never an error); precedence
   (first-dir-wins, duplicate reported); catalog byte-stability across
   conversation turns; `skill_read` path jail (abs/`~`/`..`/symlink
   escapes rejected); pagination params refused; body delivered as tool
   result (not a message mutation — assert prefix unchanged); LinkedFiles
   present; `WithPinnedSkills` cap of 2 enforced; end-to-end: fake
   provider loads a skill (L1) and follows it.
10. **Live smoke (opt-in, build tag `live`):** real Serper + one provider,
   runs a rotation of the §14 prompt suite from the gitignored
   `testdata-live/` corpus; asserts `len(Sources) > 0` on search-requiring
   categories and at least one exercised `fetch_page` whose URL matches a
   returned source. Nothing financial is committed. (The bash preset is
   exercised by hand via the CLI, §13 — no live gating.)
11. **Local credentials (not committed):** keys for live testing live in
    `~/api_keys.md` — a `ZAI_API_KEY` for the Z.AI GLM coding endpoint and
    a `SERPER_API_KEY` for Serper. Export them before running the live
    suite or the CLI; the file stays outside the repo and no key material
    is ever committed (§15). Default live configuration:

    ```go
    llm, _ := llmhub.New("zai", os.Getenv("ZAI_API_KEY"),
        llmhub.WithBaseURL("https://api.z.ai/api/coding/paas/v4"),
        llmhub.WithModel("glm-5.3")) // or "glm-5.3-flash"
    ```

    The `live` test runner falls back to these defaults when
    `AGENTLOOP_LLM_PROVIDER` is unset.

## 13. Example CLI — `examples/ask`

A general-purpose runner (no domain terminology), also the primary tool for
exercising the §14 test corpus locally.

```
go run ./examples/ask \
  -provider openai -model gpt-4o-mini \
  -serper-key "$SERPER_API_KEY" \
  -prompt-file testdata-live/prompts/why-moving-nvda.md \
  -stream -v
```

With the local keys from `~/api_keys.md` (§12.11) — no other provider
accounts needed:

```bash
export ZAI_API_KEY=… SERPER_API_KEY=…   # values from ~/api_keys.md
go run ./examples/ask \
  -provider zai -model glm-5.3 \
  -base-url https://api.z.ai/api/coding/paas/v4 \
  -serper-key "$SERPER_API_KEY" \
  -prompt "What Go version was released most recently, and when?" \
  -stream
```

Bash preset (coding agent, mini-swe-agent style):

```bash
go run ./examples/ask \
  -provider zai -model glm-5.3 \
  -base-url https://api.z.ai/api/coding/paas/v4 \
  -agent bash -cwd /tmp/workspace \
  -step-limit 20 -cost-limit 1.5 \
  -prompt "Add a README.md describing this project." \
  -stream -v
```

| Flag | Description |
| --- | --- |
| `-provider` | llmhub provider (required); env-keyed like llmhub's CLI example |
| `-model` | model identifier |
| `-api-key` | LLM API key (or provider env vars) |
| `-base-url` | provider base URL override |
| `-agent` | `search` (default; web_search + Serper), `bash` (coding agent, §8), or `plain` (no tools — the core loop bare) |
| `-chat` | interactive multi-turn mode: after each answer, read the next prompt from stdin (blank line or Ctrl-D exits); one `Conversation`, shared history |
| `-cache-key` | llmhub prompt-cache key — recommended with `-chat` (§5.5) |
| `-serper-key` | Serper key (or `SERPER_API_KEY`); search agent only |
| `-prompt` | prompt/task text |
| `-prompt-file` | read prompt from file (how the local corpus is fed) |
| `-system` | replace system prompt |
| `-cwd` | working directory for the bash agent (default `.`) |
| `-env` | extra env `KEY=VALUE` for the bash agent (repeatable) |
| `-skills` | comma-separated skill directories to load (§5.8); works with every agent |
| `-step-limit`, `-cost-limit` | bash agent limits |
| `-confirm` | bash agent: prompt before each command (mini's confirm mode; default off = yolo) |
| `-tool-protocol` | bash agent: native tool calling instead of text protocol |
| `-stream` | stream live to stdout: answer deltas as they arrive; reasoning/thinking deltas dimmed on stderr (llmhub `ReasoningDelta` passthrough); tool activity inline |
| `-v` | print event trace: turns, tool calls, durations, exit status, TTFT/total timing |
| `-json` | emit the final `search.Answer`/`bash.Answer`/`agentloop.Answer` as JSON (text, reasoning, timing, sources or submission, usage, trace) |
| `-max-tool-calls`, `-max-steps` | budget overrides |

Exit codes: `0` success (any ExitStatus), `1` error, `2` budget exhausted.
Prints total cost when the provider reports usage.

## 14. Test Prompt Suite — Stock Q&A (local only, not committed)

agentloop is domain-neutral (§6), but its primary acceptance corpus is
stock Q&A, derived from the production prompts in `~/eh-trade` (`go/ai.go`:
`PrepareOptionsNews`, `GetStockOpinion`, `GetPortfolioOpinion`,
`GetCompareOpinion`, `GetSwingOverview`, `InferCompanyInfoWithAgent`). Stock
prompts stress everything a search agent must do: time-sensitive facts,
structured JSON output, multi-entity queries, and citation-style overrides.

**This corpus never enters the public repo.** Per §15, the shipped
documentation and source — including tests and testdata — must contain no
financial or stock terminology. The corpus lives in this spec and in a
gitignored `testdata-live/prompts/` directory consumed by `examples/ask
-prompt-file` (§13) and the `live` test runner. Committed offline tests use
generic prompts with the same loop shapes. Each entry lists the eh-trade
prompt it mirrors, the verbatim query, and pass criteria. **[live]** =
requires real search; **[offline]** = loop-behavior fixture whose committed
equivalent is genericized.

### 14.1 Why is it moving? (mirrors `GetStockOpinion`)

> Why is NVIDIA stock up this week? Explain in 2-3 short sentences, in simple
> language, what is driving the move. Do not use jargon.

Data-bearing variant, matching the eh-trade prompt that includes indicators:

> You are a concise investment assistant. Explain the story behind TSLA's
> current trend in 2-3 short sentences. Search for recent news first.
> Data: price $402.11, SMA20 $391.40, SMA50 $372.85, SMA200 $331.02,
> RSI14 68.3, MACD 4.21, ADX 27.5, support [372, 351], resistance [411, 425].

**Pass:** ≥1 `web_search` call; `Sources` non-empty; text ≤ 4 sentences;
names a concrete catalyst drawn from results.

### 14.2 Options-activity commentary (mirrors `PrepareOptionsNews`)

> Here is a table of today's Canadian option volumes:
> `[{"symbol":"RY","volume":12450},{"symbol":"BAM","volume":8800},{"symbol":"ENB","volume":6100}]`
> Research recent news about these companies and write 3-5 bullet points
> explaining any unusual trends for an audience unfamiliar with markets, at a
> high-school level. One ticker per bullet, ticker in parentheses after the
> company name, one relevant emoji per bullet. Do not include reference
> citations like [1], [2].

**Pass:** ≥1 search; ≤1 bullet per ticker; no `[n]` markers in text while
`Sources` is populated; no jargon (spot-check "RSI", "IV", "delta").

### 14.3 Portfolio opinion (mirrors `GetPortfolioOpinion`)

> Analyze this portfolio and give 1-2 concise sentences on diversification,
> risk, and growth potential. Search for recent news about these companies
> first. Portfolio: AAPL 25%, MSFT 25%, NVDA 20%, XOM 15%, ENB.TO 15%.

**Pass:** ≥1 search; no hallucinated tickers; ≤ 2 sentences.

### 14.4 Head-to-head comparison (mirrors `GetCompareOpinion`)

> Compare AMD and Intel head-to-head over a 1-year horizon. Search for recent
> results and news on both. Return valid JSON only:
> {"title": string (≤6 words), "summary": string (≤60 words),
> "verdict": string, "bullets": string[] (exactly 3),
> "uncertainty": string, "what_would_change": string}.
> The verdict must state "technically stronger", "fundamentally stronger",
> or "mixed/inconclusive".

**Pass:** ≥1 search; parses via `ExtractJSONObject`; verdict from the three
bases; exactly 3 bullets.

### 14.5 Swing-trade overview (mirrors `GetSwingOverview`)

> You are an expert swing trader. Analyze this position and give your opinion
> on entry points, take profit, and stop loss levels, for someone already in
> the position and someone considering entering. Be concise; use headings and
> brief paragraphs. Position: {"ticker":"TSLA","quantity":100,
> "averageCost":380.50,"currentPrice":402.11,"RSI14":68.3,
> "SMA20":391.40,"SMA50":372.85,"SMA200":331.02,"ATR14":9.2}.

**Pass (live):** ≥1 search; stop < current < target. **[offline]** variant:
no search required — verifies the zero-tool-call path with a data-bearing
prompt.

### 14.6 Company identification & sector (mirrors `InferCompanyInfoWithAgent`)

> Research the publicly traded company with ticker "BCE.TO" (Toronto Stock
> Exchange). Find: 1) the official legal name, 2) a 1-2 sentence description
> of the business, 3) the most appropriate GICS sector from: Communication
> Services, Consumer Discretionary, Consumer Staples, Energy, Financials,
> Health Care, Industrials, Information Technology, Materials, Real Estate,
> Utilities, Miscellaneous. Respond ONLY with JSON:
> {"legal_name": "...", "description": "...", "sector": "..."}.

**Pass (live):** ≥1 search; `legal_name` is the corporate name; sector from
the list (ETF variant "XIU.TO" → "Miscellaneous").

### 14.7 Current-facts market questions (new; core agentic-search cases)

- **[live]** "What were Apple's most recent quarterly revenue and EPS, how
  did they compare to analyst expectations, and what reporting period do
  they cover?"
- **[live]** "When do the big Canadian banks report earnings next?"
- **[live]** "What is Enbridge's (ENB.TO) current dividend yield and when is
  its next ex-dividend date?"
- **[live]** "What are analysts' price targets for Shopify after its most
  recent earnings?"
- **[live]** "What happened in the US and Canadian markets today and why?"
- **[live]** Fetch-dependent: "Find the most recent Go blog post about loop
  variables, fetch the full article, and summarize the exact semantics
  change in 3 sentences." — must chain `web_search` → `fetch_page`;
  `Sources` marks the fetched entry `Fetched: true`.
- **[offline]** "What does a P/E ratio mean?" — **zero** tool calls
  (`ToolCalls == 0`, `Steps == 1`).

**Pass:** search-requiring prompts yield ≥1 search, reputable sources, and
figures with an as-of date or reporting period (the query text instructs
this itself — domain guidance travels with the query, never the agent).

### 14.8 Suite mechanics

- Corpus lives in gitignored `testdata-live/prompts/` (a committed
  `testdata-live/README.md` explains the convention) with a manifest of
  category, mirror source, live/offline, and pass criteria. Run via
  `examples/ask -prompt-file` (§13) and the `-tags live` runner.
- Committed offline tests replicate each category's *loop shape* with
  generic queries — e.g. 14.4's committed equivalent compares two
  programming languages with the same JSON verdict schema; 14.6's resolves
  an organization's official name and country from a website domain.
- Structured-output categories (14.4, 14.6) double as tests of the public
  **JSON-tolerant extraction helper** `ExtractJSONObject(text) (string,
  error)` — ported from eh-trade's `extractJSONObject`/
  `findBalancedJSONObjects` (handles markdown fences, prose prefixes,
  balanced braces inside strings). Structured answers are a common
  search-agent need, independent of domain.
- eh-trade's `<think>` stripping (qwen/ollama models) is **not** needed:
  llmhub surfaces reasoning as `ReasoningContent`, never inline in `Text()`.

## 15. Repository, License & Documentation

- Public git repository at `github.com/smhanov/agentloop`; MIT `LICENSE`,
  matching llmhub and laconic. Package structure: `agentloop` (core loop),
  `agentloop/search`, `agentloop/bash` (§4) — the README reference
  documents all three.
- **No financial, stock, or trading terminology anywhere in the shipped
  repo** — README, doc comments, identifiers, tests, testdata, examples, or
  commit-facing docs. The package is presented as a general-purpose agentic
  loop library with two presets; the stock corpus is local-only (§14).
- `README.md` is the public face and must serve two readers without
  compromise — **a 5-minute tutorial up top, the complete reference
  below** (mirrors llmhub's README style):
  1. **Tutorial (top, in order):** title + one-paragraph pitch ("a small
     Go library for building LLM agents with tools, hooks, skills, and
     ready-made presets"); Why?; `go get` install; Quick Start — the §6
     search agent in ~10 lines with `Sources` printed; the CLI in three
     commands (search, bash agent, `-chat`); "where to go next" links into
     the reference sections.
  2. **Complete reference (below):** core API — building an agent
     (`New`, options table); writing tools (§5.7 clock example, plus a
     stateful `Sequential: true` example); the hooks reference table
     with every hook point and `Event` field; budgets & loop safety;
     streaming (llmhub parity, watchdog); conversations (multi-turn,
     prompt-cache guidance); skills (folder layout, L0–L3 explanation,
     authoring guidance: "Use when…" descriptions, body template);
     presets — **the search agent as a complete usage guide** (both tools,
     `Sources`, `search.Answer`, all options) and **the bash agent**
     (mini-swe-agent compatibility table, environments, limits, exit
     statuses, approvals); **writing your own agent** — recipes: custom
     tools + hooks, text-action protocols via the finish veto (§5.3),
     steering via `BeforeGenerate`, cost caps via `ev.Skip`, **context
     rebuilding/compression via `RebuildContext`** (observe or rebuild the
     exact context the model sees each step); custom
     `Searcher`/`Fetcher`/`Environment`; testing your agent with a fake
     provider; troubleshooting (budget exhaustion, stale streams,
     format-error loops).
  3. **Back matter:** Testing (`go test ./...`), Contributing
     (rebase-only workflow, as in llmhub), comparison vs
     laconic/mini-swe-agent, License.
- A laconic-style "Example output" section shows one real transcript of a
  **generic** query (e.g. a recent release or current-event question) with
  its tool calls and sources.
- Dependencies: stdlib + `github.com/smhanov/llmhub` only. No vendor SDKs.
- CI on push: `gofmt -l`, `go vet`, `go test ./...` (GitHub Actions).

## 16. Milestones

| # | Deliverable | Done when |
| --- | --- | --- |
| M1 | Core loop (`Agent`, `Tool`, `Answer`) + tests | §12.1 core cases green |
| M2 | Hooks (incl. finish veto) + `AskStream` with llmhub-parity streaming (reasoning deltas, timing passthrough) + `Conversation` + parallel tools + tests | §12.1–12.2 green |
| M3 | search pkg: Serper clients (search/news/scrape, 429 retry) + preset with `web_search` (+news type) + `fetch_page` + `Sources()` + tests | §12.3–12.4 green |
| M4 | bash pkg: `LocalEnvironment`, text protocol (mini parser corpus ported), limits, exit statuses + tests | §12.5–12.7 green; scripted mini-style trajectory passes |
| M5 | Bash tool protocol variant + tests | same trajectory green via native tool calls |
| M6 | Skills: `LoadSkills` validation/jail, catalog, `skill_read`, pinning + tests | §12.9 green |
| M7 | `examples/ask` CLI (search + bash + plain) | runs §13 flags incl. `-agent bash` and `-skills`; `-prompt-file` works against local corpus |
| M8 | README + LICENSE + repo hygiene | README matches the §15 tutorial+reference contract; repo scan finds no financial terminology |
| M9 | Local corpus pass | §14 [live] prompts run green via CLI + `-tags live` |

## 17. Decisions

All former open questions are resolved (Oct 2026):

| # | Question | Decision |
| --- | --- | --- |
| 1 | Preset code layout | **Subpackages now**: `agentloop/search`, `agentloop/bash`; each aliases `type Option = agentloop.Option`. |
| 2 | Tool middleware | **Deferred** — hooks suffice until result caching is actually built. |
| 3 | Extra Serper endpoints | **`type` parameter on `web_search`** limited to `"search" \| "news"`; no separate tools; scholar/images skipped. |
| 4 | Serper 429 handling | **Transparent client-side retry** with jittered backoff (`WithSerperRetries`, default 1, max 3); surface to the model only after retries are exhausted. |
| 5 | Docker environment | **Interface only**: ship `bash.Environment` + `NewLocalEnvironment`; container environments are caller-implemented; README documents a Docker example pattern. |
| 6 | Trajectory format | **Native JSON** (`bash.Answer` serialization); no mini-swe-agent-1.1 adapter. |
| 7 | Parser conformance | **Port mini-swe-agent's MIT parser test corpus** (nested fences, `TRIPLEBACKTICKSBASH`). |
| 8 | Typed preset conversations | **Deferred**; use `Agent().Conversation()` + `search.Sources(ans)`. |
| 9 | Skills phase 3+ (BM25, authoring, quarantine) | **Deferred** per the roll-out ladder (`docs/references.md` §7); revisit past ~40 skills or on untrusted skills. |

Also decided: the package is domain-neutral — no domain guidance ships in
any default prompt and no domain-specific option exists; callers layer
domain behavior via query text or `WithSystemPrompt` (§6, §15). Named
`agentloop` (general-purpose loop; presets are configurations, not the
identity). Multi-turn ships first-class as `Conversation` (§5.5); parallel
tool calls are supported per step, bounded by `WithParallelTools`
(default 3) with order-preserving result messages.

## 18. Appendix: harness-research alignment

Mapping from `docs/references.md` (the distilled harness-writing reference)
to this spec's choices:

| Reference guidance | This spec |
| --- | --- |
| §2 vocabulary: Turn = one user message, Step = inference + tool execs | adopted (§5.1); `Answer.Steps`, `WithMaxSteps`, `Event.Step` |
| §2 loop shape 3: reusable loop fn + parallel per-iteration execution (PicoClaw `ToolLoop`) | the core loop (§5.4) — one implementation shared by presets and conversations |
| §2 multiple limit dimensions (steps + cost + wall-time) | `MaxToolCalls` + `MaxSteps` + dup-call limit + ctx deadline; bash preset adds graceful cost/wall-time (§8.1) |
| §2 termination: completion-reason enum (dsh) | `Answer.Status` (§5.6); `bash.Answer.ExitStatus` keeps mini's strings |
| §2 loop detection (identical calls and output) | `WithDupCallLimit` + `Tool.Repetitive` (§5.2) |
| §2 steering/inbox seam (steer/inject/cancel) | `BeforeGenerate` message-append + ctx cancel (§5.5) |
| §2 loop shape 5: event-log projection (dsh `deriveMessages`) | append-only transcript + `RebuildContext` projection hook (§5.3) — the LLM context is derived, never mutated |
| §4 overflow → compress + retry; compression is the sanctioned prefix mutation | `RebuildContext` hook: deterministic compression of older steps, trailing tool-call/result pair preserved (§5.3) |
| §3 ToolDefinition/ToolSchema split; `isConcurrencySafe` batching | `Tool{Def, Run, Sequential}` + `Repetitive` (§5.2) |
| §3 guarded execution / approval gates | `WithApprover` on the bash preset; blocking `BeforeToolCall` sanctioned (§5.3, §8.1) |
| §3 malformed-call repair in-band (Levenshtein names, error results) | §5.2 — tool errors and unknown-name suggestions are tool results, never hard fails |
| §3 one-tool counter-case (mini) | bash preset ships exactly one `bash` tool (§8) |
| §4 head+tail truncation with elided-chars notice | bash observations (§8.4, identical to mini); `fetch_page` (§7.2) |
| §5 prompt-cache discipline: never mutate the prefix; dynamic facts in latest user message | Conversation prefix-stability invariant (§5.5); `WithCacheKey` guidance |
| §6 stale-stream watchdog (90s) | `WithStreamWatchdog` (§5.1), `ErrStreamStale` |
| §8 append-only log → projected context | `Conversation` history is append-only; `Answer.Messages` is the per-run projection; durable event-log state is a non-goal (§1) |
| §10 autonomy levels / destructive-action hygiene | `WithApprover` confirm/yolo modes (§8.1); trash-over-rm guidance belongs to callers' approvers |
| §12 layered test tiers; blocking P0 | P0 invariant tier in §12.7 |
| §12 `arewetiny` CI metric | exported-API budget in CI (§11) |
| §7 skills (L0–L3) | shipped: L0–L2 with `LoadSkills`/`skill_read`/catalog + pinning (§5.8); L3 via the bash preset's execution tool; phases 3–4 (authoring, hubs, quarantine) are non-goals |
| §13.9 formats degrade over iterations | format-error counters (bash §8.3); dup-call limit (core) |

Deliberate departures, with reasons: no internal exception-driven control
flow (Go hooks + finish-veto express the same exits without panics; the
bash preset keeps mini's *semantics*, not its Python control flow); no
built-in failover/retry ladder (llmhub owns HTTP retries and 429 backoff —
one retry policy per process, not per harness layer); no self-improvement
loops (rare even in the reference set; rollback guardrails mandatory if
ever added).
