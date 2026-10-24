# Agent Harness Writing Reference (for LLM agents)

> Distilled from the harness-study notes in Steve's vault (`AI Agents/` folder): OpenClaw, GoClaw (+ NLB fork), PicoClaw, ZeroClaw, Hermes, gptme, mini-SWE-agent, DeepSeek harness (dsh), plus a skill-system playbook and a 7-framework comparison. Use ONLY this knowledge; do not invent beyond it.

## Index (find your topic without loading the rest)

- **§1 Architecture choice** — priorities, footprint numbers, self-evolution
- **§2 The agent loop** — turn/step structure, caps, termination, exception-driven control flow, steering
- **§3 Tool design** — definition vs wire schema, guarded execution, parallelism, one-bash-tool philosophy
- **§4 Context management** — truncation, compression, budgets, overflow recovery
- **§5 System prompt & prompt-cache discipline** — stable tiers, cache boundary, section ordering
- **§6 Error handling & failover** — retry classification, response-repair, empty/truncated handling
- **§7 Skills & progressive disclosure** — L0–L3 invariant, validation, trust gates, roll-out phases
- **§8 Memory & durable state** — event-log projection, memory tiers, knowledge graphs
- **§9 Subagents & multi-agent** — isolation, delegation, spawn patterns
- **§10 Security & approvals** — sandboxing, autonomy levels, secret scrubbing, prompt-injection defense
- **§11 Streaming**
- **§12 Evaluation & testing**
- **§13 Master list of gotchas & failure modes**
- **§14 Per-harness "copy / avoid" table**
- **§15 Source notes**

---

## §1 Architecture choice

**Pick the architecture whose #1 priority matches your deployment.** The divergence between harnesses is intentional, not accident:

| Priority | Champion |
|---|---|
| Reliability, "never give up, never fabricate" | Hermes (Nous Research) |
| Engineering discipline + ecosystem | OpenClaw (strict TS, TypeBox→Swift codegen, enforced plugin boundary) |
| Keep-it-tiny (enforced in CI via `arewetiny` metric) | gptme |
| Ubiquity on $10 hardware (<10MB RAM target, RISC-V/MIPS/ARM) | PicoClaw |
| Trust (state never duplicated, signed intents) | ZeroClaw |
| Zero-dependency single binary | GoClaw original |
| Enterprise multi-tenancy | GoClaw NLB (PostgreSQL+pgvector, RBAC) |
| Minimalism anti-revolution (~300 lines, one tool) | mini-SWE-agent |

**Footprints actually observed:** Hermes ~50K LoC Python, 1–2s boot, 200–500MB; OpenClaw 400K+ TS, ~6s, ~1.5GB; gptme Python, ~0.8s, ~100MB; PicoClaw <1s, 10–20MB; ZeroClaw Rust 126K LoC, <10ms, 26MiB binary; GoClaw orig 17K Go, <100ms, 20–50MB; GoClaw NLB 474K Go, ~1s, 50–200MB.

**Footprint is a real decision, not cosmetic:** GoClaw's rewrite was motivated by measuring OpenClaw at 1.5GB RAM / 6s startup. PicoClaw runs the same mission on a $10 LicheeRV-Nano.

**Known complexity debt in EVERY harness — plan for it:** Hermes has a 4,000-line `run_conversation` god-function with an implicit retry-flag state machine (exhaustive error recovery traded for readability); OpenClaw has a 3,800-line `run.ts`; ZeroClaw has a 40+-field Agent builder struct and 7,740-line agent.rs (builder pattern can MASK insufficient decomposition). Don't strive to avoid long loop files; do understand the trade.

**Self-improvement loops are rare:** only Hermes (memory-review + skill-creation nudges after turns) and GoClaw NLB (metrics → suggestion → auto-adapt with apply/validate/rollback guardrails) have them. If you build one, the rollback guardrail is mandatory.

**Convergence points everyone reached independently:** layered/stable prompt tiers for caching (→§5), append-only or human-readable durable state (→§8), guarded/sandboxed tool execution with approval gates (→§3, §10), progressive skill disclosure (→§7), and no contextual auto-guidance without pattern triggers (→§7 Lessons).

---

## §2 The agent loop

*(Relates to §9 subagents sharing this loop, §13 caps/detection gotchas, §6 failover around the loop.)*

**Vocabulary everyone converged on:** Turn = handling of one user message (0+ Steps); Step = one model inference + its tool executions. Adopt this.

**Loop shapes observed (pick consciously):**
1. *Monolithic function* (Hermes `run_conversation` ~3,900 lines; OpenClaw `run.ts` ~3,800 lines; GoClaw orig a 271-line goroutine think-act loop). Working but god-function risk — see §1.
2. *Formal pipeline stages* (GoClaw NLB: 8 stages context→history→prompt→think→act→observe→memory→summarize; Stage interface `Execute(ctx,*RunState)` returning `Continue/BreakLoop/AbortRun` signals). **The cleanest formal loop across the family — best for testability.**
3. *Reusable loop function* (PicoClaw: `ToolLoop` shared by main agent AND subagents: build tool defs → LLM → no tool calls = done → normalize calls → execute ALL this iteration's calls in parallel → append results → iterate).
4. *Exception-driven control flow* (mini-SWE-agent): `InterruptAgentFlow` exceptions (`Submitted`, `LimitsExceeded`, `UserInterruption`, `FormatError`) carry messages to append, so deep branches exit cheaply to the top loop. Copy this — it prevents error-handling spaghetti at every nesting level.
5. *Event-log projection* (dsh): the loop never mutates LLM context directly; history is *derived* (`deriveMessages()`) from an append-only session log. → §8.

**Iteration caps observed:** Hermes 90/turn, GoClaw orig 25/user message, PicoClaw ~10 per ToolLoop, ZeroClaw max 10 iters. Always have caps, and *multiple* limit dimensions: steps + cost + wall-time (mini-SWE-agent uses all three) ; token/cost budgets (ZeroClaw).

**Termination signals:** completion reason enum (dsh: `completed|max-tokens|error|aborted|blocked`). mini-SWE-agent: agent echoes a marker line `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` with rc==0 — a completion protocol that needs NO special API surface.

**Loop detection:** ZeroClaw checks repeated identical calls + no-progress + max iters. Required if the harness runs unattended.

**Steering / inbox injection (the standard HITL seam):** dsh exposes `followup(msg)` (queues next turn + wakes), `steer(msg)` (injects at next step boundary + wakes), `inject(msg)` (adds context silently, no wake), `cancel()` (abort in-flight). OpenClaw serializes runs per session with queue modes (steer default / followup / collect / interrupt). PicoClaw has a dedicated `pkg/steering`. Build the seam before you need it.

**Sniff test from the field:** GoClaw orig deliberately ships NO retry/failover machinery (→§6 trade-off), while OpenClaw treats failover as day-one (→§6).

---

## §3 Tool design

*(Cross-refs: tool schema cost → prompt tiers §5; skill vs tool vs MCP split → §7; safety gates → §10.)*

**ToolDefinition vs ToolSchema (dsh's key split):** ToolSchema = wire format for the LLM (name, description, JSON Schema). ToolDefinition adds:
- `output`: **lossless canonical JSON** + pure `render()` projecting JSON → model ContentBlock[] — decouples backend data from LLM formatting.
- `isConcurrencySafe(args)` predicate → registry batches parallel-safe calls; others run in strict order.
- `presentCall/presentResult`: pure projections for the UI.
- `execute(args, execContext)` with `exec.signal` cancellation/timeout support.
Use a `defineTool()` typed DSL — never hand-write JSON Schema.

**Guarded tool pipeline (dsh 7 stages — the reference design):**
1. `tool/call` logged durably (+ presentCall UI card);
2. `tools/pre-execute` waterfall (approval hooks/flags);
3. monotonic guards — allow/deny only, strictly ordered, cannot be bypassed;
4. `ctx.approval` gate — halts until user answers or timeout;
5. `tools/execute` around-waterfall (timeout, metrics, retry);
6. body `execute()` returning lossless JSON, fs write-intent events on mutation;
7. `tools/post-execute` waterfall (rewrite errors, attach additionalContext) then `finalizeContent` enforces byte caps + truncation notices.

**Tool protocols:** support multiple call formats where model compatibility matters — ZeroClaw dispatches Native JSON | XML | PromptGuided, with a separate parser crate, malformed-envelope retries (max 2), and think-tag stripping. Hermes repairs hallucinated tool names via Levenshtein and injects invalid-JSON errors as role:'tool' results for self-correction (→§6).

**Tool cost is real:** Hermes' ~50 tools cost 20–30K tokens of schemas ALONE per call. This is why narrow-waist + skills progressive disclosure (→§7) exists. Bar for adding a native tool must be high.

**The one-tool counter-case (mini-SWE-agent):** a single bash tool + native function calling matches complex frameworks on SWE-bench. Its claim: custom ACI tools are liabilities for modern models. Decide rich-tools vs one-bash FOR YOUR models, not by fashion.

**Tool availability & discovery:** Hermes: thread-safe singleton registry, self-registration at import, AST-based discovery of tools/*.py, `check_fn` availability probes TTL-cached 30s. gptme: ToolSpec instances auto-discovered by scanning modules — zero registration boilerplate.

**Token-cost matrix (where to put capability):**
| Layer | Cost | Use for |
|---|---|---|
| system prompt / AGENTS.md | every turn, forever | identity, hard safety, house rules |
| native tools | every API call | the narrow waist: read/write/bash/search |
| MCP | all schemas usually up front | remote APIs/OAuth/live data (agents tolerate ~2–3 fat MCP servers before tool-choice degrades) |
| skills | ~100 tok metadata + body on load | repeatable procedures (→§7) |
| memory | injected snapshot | user facts, NOT playbooks |

---

## §4 Context management

*(Cross-refs: prefix-mutation rules → §5; overflow retries → §6; event-log projection → §8.)*

**Overflow handling (Hermes reference implementation):** classify API errors by failover reason; on context overflow → compress + retry up to 3x; compression via a cheap auxiliary model producing structured summaries, with a defensive `SUMMARY_PREFIX`; auth profile rotation on rate limits is adjacent machinery (→§6).

**Truncation patterns:**
- mini-SWE-agent observations: head+tail truncation keeping 5000 head/tail chars beyond a 10K budget, with `elided_chars` warning — not naive head-only.
- dsh: `finalizeContent` byte caps + explicit truncation notices; ZeroClaw: UTF-8 boundary prompt truncation when capped.
- gptme: dedupe repeated tool outputs and summarize near token limits.

**Compression must not break in-flight state:** never prune the most recent `skill_view` of a skill in use (Go skill playbook). Cache discipline interacts here — compression is the ONLY sanctioned mutation of the prompt prefix (→§5).

**Projects history from a log, not memory:** dsh `deriveMessages()` projects the LLM context from the append-only event log → zero hallucinated history; the invariant "model-visible means logged" bans side-band prompt injection.

**Beyond dedupe: pruning as a pipeline stage** (GoClaw NLB Prune stage between think and act).

---

## §5 System prompt & prompt-cache discipline

*(Cross-refs: compression as the only sanctioned prefix mutation → §4; skill catalog placement → §7 top-tier rules; session-pinned caches → §8.)*

**THE sacred rule (all harnesses converge): never mutate the prompt prefix mid-conversation.** Hermes treats prompt caching as a cost constraint that is "sacred"; gptme has an explicit cache boundary; dsh maximizes KV-cache hits; mini uses one fixed template pair; PicoClaw tags each prompt part with `PromptSourceID` + stability metadata so stable parts cache and ephemeral never do.

**Three-tier prompt structure (Hermes):** STABLE (identity, tools guidance, skills index) / CONTEXT (cwd-dependent AGENTS.md etc.) / VOLATILE (MEMORY.md, USER.md, date-only). Built once per session, persisted to DB, byte-stable. Dynamic facts get injected into the LATEST USER MESSAGE, not the system prompt.

**dsh prompt assembly:** ordered sections by increasing `order` int (-100 harness identity, 0 persona, 99 code-mode collapse, 100–199 tool guidance, 200+ skills catalog). Split static sections (byte-identical per step) from dynamic context — and dynamic values (cwd, git branch, editor files) are projected as context updates logged as user-role events ONLY when the value changes.

**Prompt-cache boundary pattern (OpenClaw):** stable content above the boundary, volatile (messaging/voice/heartbeats/runtime) below; provider plugins can inject stable-prefix/dynamic-suffix around the boundary. Deterministic ordering everywhere — cache integrity depends on it.

**Do's and don'ts:**
- Skill catalog rendered byte-stable into the STABLE tier; loaded skill bodies arrive as TOOL RESULTS, never prompt injection (prompt injection busts the prefix cache mid-session) — optional exception: pin ≤2 always-on skills max.
- Do NOT rebuild the system prompt after a skill load — that's a cache nuke.
- Session-pinned catalog: first call freezes catalog bytes; rebuild only on mtime/size manifest change; fsnotify-driven reloads affect NEXT session, never in-flight.
- Host-side BM25 retrieval into prompts is cache-hostile — if used, retrieve once per user turn and freeze.
- Store the built prompt (SQLite in Hermes) — rebuild only per session, and only when sources change.

**Top-of-prompt imperatives (ZeroClaw, order matters):** anti-narration ("NEVER narrate tool usage") FIRST, then tool-honesty ("NEVER fabricate results"), then tooling/execution-bias/safety sections.

---

## §6 Error handling & failover

*(Cross-refs: overflow→compression specifics → §4; malformed tool-call repair → §3; dead-stream detection → §11; caps that bound retries → §2.)*

**Classify API errors, then act (Hermes FailoverReason):** rate_limit / context_overflow / overloaded / billing etc. Response: credential pool rotation on 429, fallback provider chain, jittered 2–60s backoff, compression+retry (≤3) on overflow. OpenClaw equivalents: auth profile rotation with cooldowns, model idle watchdog, fallback thinking levels.

**Streaming is not free reliability:** Hermes runs streaming-first with 90s stale-stream detection — detect a dead stream, don't hang on it.

**Repair malformed LLM responses in-band (all three agree):** return format/parse errors to the model in-conversation (Hermes: role:'tool' error results; mini: FormatError self-correction template, with DISTINCT guidance for `finish_reason=='length'` vs malformed; with a max-consecutive-format-errors counter) instead of hard-failing.

**Empty / truncated outputs (Hermes):** text truncated → continuation retries (≤3); truncated tool calls → re-run the API call; thinking-only response → prefill; empty → nudge → prefill → retry → fallback '(empty)' placeholder.

**Minimalism as an explicit trade:** GoClaw orig ships zero retry machinery — acceptability depends on the deployment surface (CLI vs unattended daemon). Decide consciously; don't just inherit whatever you copied.

**Fail-fast config:** mini-SWE-agent's Jinja2 `StrictUndefined` templates surface config typos at load time. gptme skill validation is fail-closed: skip the invalid file, never abort the agent.

---

## §7 Skills & progressive disclosure

*(Cross-refs: token-cost matrix → §3; catalog must sit in the stable prompt tier → §5; trust/quarantine detail → §10; routing evals → §12; discovery pitfalls → §13 items 4/12.)*

**Definition:** a skill = folder with required `SKILL.md` (YAML frontmatter `name`,`description` + markdown body) + optional `scripts/`, `references/`, `assets/`. **Skills are NOT a second agent. Skills prepare, tools execute.** Never encode house procedures as native tools or stash playbooks in memory (→§3 matrix).

**THE invariant — progress-discloser levels (violating this = you don't have a skill system):**
- **L0 Advertise:** ~100 tokens/skill (name+description) in prompt every turn.
- **L1 Load:** full SKILL.md (<5k tokens, <500 lines) only when the model decides it matches.
- **L2 Resources:** individual referenced file on demand.
- **L3 Scripts:** run via bash/exec; ONLY the output enters context — never paste script source.

**Discovery trade-offs:** default = catalog in system prompt + slash commands. Search-only skills_list fails — models don't reliably search. If the catalog is noisy, demote to names-only, NEVER hide names. Conditional visibility (`requires_tools`, `fallback_for_tools`, platforms) filters at catalog render, not by deleting files.

**Precedence stack (first name wins, never merge, log collisions):** project → `.agents/` → user dirs → bundled. Bundled delivery: seed-to-disk + manifest of hashes so upgrades refresh only unmodified files (Hermes pattern).

**Description = the router:** imperative "Use when…", include user intent + paraphrases; near-miss negatives ("Don't use for:") go in the BODY, not the description. Spec cap 1024 chars; operational target 80–160. After a triggering miss, patch the DESCRIPTION first — trigger failures are almost always description failures.

**Body template models actually follow:** Title + one-paragraph what/doesn't/deps → When to Use (+ Don't use for) → Prerequisites (exact binaries/env/keys) → Procedure (numbered steps, each with completion check) → Pitfalls → Verification. Author by running the agent WITHOUT the skill, noting flails, then writing the smallest SKILL.md that would have prevented it. Watch trajectories: never loads → fix description; loads but fails → fix procedure. Split fat bodies into one-level `references/`. No router/index skills — the catalog already routes.

**Validation (fail closed):** frontmatter structure, name `^[a-z0-9]+(-[a-z0-9]+)*$` 1–64 chars == dirname, description 1–1024, body non-empty; invalid → log + omit, never crash. Path jail: reject abs paths/`~`/`..`/symlink escapes (EvalSymlinks+Rel); refuse offset/limit on skill-read tools (models read page 1 and skip the rest — NEVER paginate instructional tools); return `linked_files` so the model doesn't `ls`.

**Model-dependence warning:** weaker models miss relevant skills even with "MUST load" language — description quality + evals (→§12) matter more than prompt shouting.

**Security gates (skills inherit shell/FS/credential powers):** provenance lockfile (source URL + sha256); scan on install (NUL check, strip Unicode Tags U+E0001–U+E007F + bidi overrides, secret regex, `curl|bash` detection); quarantine dir for hub installs until scan+confirm; untrusted skills = instructions only, scripts require graduated trust; `allowed-tools` is a CEILING not a floor; NO live remote instruction fetching ("follow this URL" = runtime supply chain); loaded-skill instructions become authoritative, so a "don't ask again" in a skill can cascade. Scan stats: Snyk ToxicSkills n=3,984 → 13.4% critical; Liu et al. n=42,447 → 26.1% ≥1 vulnerability, scripts 2.12× more vulnerable.

**Roll-out phases (don't start at 4):** 0 fake it (hardcoded catalog + reader) → 1 filesystem (scan/validate/snapshot/jail, 5–20 bundled skills) → 2 UX (slash commands, linked_files) → 3 authoring (`skill_manage`, validator, agent-authored-after-hard-tasks nudge) → 4 scale/security (BM25, hub, quarantine). Don't build first: vector search over bodies, marketplace, auto-synthesis/RL, per-skill subprocess isolation, MCP-delivered skills, router subagent.

---

## §8 Memory & durable state

*(Cross-refs: what belongs where → §3 token-cost matrix and §7 "not in memory"; session-pinned caches → §5; log-as-eval-data → §12.)*

**Append-only event log → projected context (dsh, the gold standard):** every state change is an event (`session/create`, `turn/start`, `user/message`, `assistant/chunk`, `tool/call`, `tool/result`, `step/end`, `turn/end`, `agent/error`); the loop derives LLM messages from it via `deriveMessages()`. Benefits: zero hallucinated history, deterministic resume/fork/replay, full auditability. OpenClaw similarly keeps session transcripts in JSONL but its session state in SQLite.

**Human-readable file state (GoClaw orig):** `~/.goclaw` with hot-reloaded json5 config (fsnotify), JSONL sessions, markdown memory, SQLite knowledge graph — zero-dep and debuggable. PicoClaw/Hermes also persist sessions (Hermes: SQLite, plus trajectory JSONL per turn; mini: `.traj.json` every step with a `finally`).

**Memory tiers (GoClaw NLB):** L0 Working (conversation) → L1 Episodic (session summaries, consolidation workers) → L2 Semantic (knowledge graph, pgvector); async consolidation via event bus + worker pool with dedup/retry.

**Knowledge-extraction memory (Cortex, GoClaw orig):** hybrid deterministic + LLM extraction of entities/relations/facts into SQLite; prompt-injected; skips trivial exchanges (<100 chars, "ok"/"thanks"). GoClaw memory search is BM25 over markdown — no vectors — and is sufficient at small scale.

**Multi-backend memory w/ decay (ZeroClaw):** markdown/SQLite/embeddings/vector-merge/response-cache + decay; auto-save memory after each loop iteration.

**What does NOT belong in memory:** playbooks/procedures (→ skills, §7), tool schemas (→ tools), long bodies (progressive disclosure). Cap entry lists; consolidation beats accumulation.

---

## §9 Subagents & multi-agent

*(Cross-refs: the one shared ToolLoop → §2; isolated workspaces = the isolation boundary → §10; announce-loop guard → §13 item 18.)*

- **Isolation is the default:** isolated subagents get their OWN workspace, tools, session (OpenClaw `sessions_spawn` with contextMode `isolated|fork`; PicoClaw `spawn/spawn_status` with own workspace+session; ZeroClaw `spawn_subagent`).
- **Scope hierarchy:** child contexts inherit parent services without leaking modifications back (dsh contexts; OpenClaw per-agent agentDir/session stores).
- **Reusable loop:** PicoClaw's `ToolLoop` is literally shared by main agent and subagents — one loop implementation, no behavioral divergence (→§2).
- **Push-based completion + announce-loop guard:** OpenClaw is the reference (push completion callback, guard so agents don't loop announcing).
- **Delegation modes (GoClaw NLB):** auto capability routing / explicit / manual approve, with `agent_links`; plus shared task boards and BatchQueue aggregation.
- **Hermes forms:** `delegate_task` subagents with depth/concurrency caps; kanban board (SQLite, dispatcher + worker lifecycle tools).
- **P2P agent messaging** exists (ZeroClaw channels) but is fringe; dsh `ask_agent`-style tools are the lightweight path.

---

## §10 Security & approvals

*(Cross-refs: guard pipeline stages → §3; `allowed-tools` ceiling on skills → §7; counters/caps → §2; scrub-before-persist pairs with §8.)*

**Autonomy levels (ZeroClaw):** Full / ReadOnly / Default-ask-before-destructive — an `autonomy_level` field + `approval_manager` are cheap and saved us elsewhere.

**SecurityPolicy shape:** `{allowed_tools, excluded_tools, allowed_commands, risk_profile}` (ZeroClaw); per-mode tool allowlists keyed to sandbox mode (OpenClaw); `isConcurrencySafe` to separate parallelizable tools from locked-order tools (dsh).

**Sandbox backends:** Docker/SSH/OpenShell (OpenClaw); `ctx.fs` seam lets local FS / Docker / SSH providers be swapped without touching tools (dsh Capability Seam — a sandbox swap redirects every dependent tool instantly).

**Secret hygiene:** scrub credentials on tool output before persisting (ZeroClaw `scrub_credentials`); Hermes' gateway scrubs secrets; dsh records tool calls durably with the scrub step as part of the pipeline (→§3 stage 7).

**Credential/key storage:** AES-256-GCM at rest (GoClaw NLB).

**Prompt-injection defense GoClaw NLB ships:** monitoring layer; five security layers total — RBAC, per-connection rate limit, injection monitoring, shell deny patterns, SSRF + path-traversal prevention. OpenClaw: DM pairing-code default, device identity + nonce signing (v3 signature binds platform+deviceFamily), TLS + optional pinning, trusted-proxy mode.

**Destructive-action hygiene (ZeroClaw prompt rules):** no exfiltration/destructive ops; prefer trash over `rm`. mini-SWE-agent's default modes: confirm/yolo/human.

**State integrity:** ZeroClaw's no-duplicate-state rule (SSOT with gate-enforced auto-revert) is itself a security property — duplicated state gets back-stroked.

**Malformed-envelope retries (max 2), counters everywhere** — every gate should have a counter and a cap so an attack or bug can't loop forever (→§2).

---

## §11 Streaming

*(Cross-refs: stale-stream watchdog → §6; deltas logged as events → §8; channel rate-limits → §2 steering context.)*

- Stream-min deliberation: OpenClaw agent spawn model emits from a stream; drafts to channels are rate-limited (min 80 chars / 500ms) so you don't spam a channel per token.
- Stale-stream watchdog: 90s (Hermes) — a stream that stops sending is a failure, detect it.
- Streamed deltas are events in the log (dsh `assistant/chunk` events into a BlockAssembler, finalized to one `assistant/message`).
- `StreamDelta::Text|Status` distinction (ZeroClaw): separate agent-status chatter from content.

---

## §12 Evaluation & testing

*(Cross-refs: skill-routing harness → §7; trajectories as eval substrate → §8; trajectory data → §13 item 4 diagnosis loop.)*

- **Skill-routing eval harness (playbook):** ~20 queries (10 should-fire, 10 near-miss), 3 runs each, pass ≥0.5 fire rate; instrument the dispatcher, not the prose; ALSO eval compliance — did the agent follow loaded instructions.
- **Activation logging = eval data:** log `{skill, path, sha256, bytes, session_id}` on every skill activation.
- **Trajectory persistence:** every step `.traj.json` (mini) / trajectory JSONL per turn (Hermes) — your post-hoc eval substrate.
- **Layered test tiers (GoClaw NLB):** P0 tenant-isolation (BLOCKING) → P1 API → P2 scenarios; put the invariants you fear most in the blocking tier.
- **Keep-it-tiny enforcement:** gptme's `arewetiny` metric in CI — encode your architecture's #1 promise as a testable metric.
- **Batch harness pattern (mini-SWE-agent):** thread pool of workers, each with own Docker container, image mapping, lock-guarded resumable dump file.
- **Fail-fast validation everywhere** (StrictUndefined templates, fail-closed loaders) — config errors are eval-visible bugs.

---

## §13 Master list of gotchas & failure modes

1. **Prompt-cache busting** — any mid-session mutation to the prompt prefix (skill loading as prompt injection, re-rendered system prompt, hot-reloaded config pushed in-flight) nukes the cache. Solutions: §5 rules.
2. **Tool schema cost** — rich tool sets cost 20–30K tokens/call; too many tools degrades tool-choice. Narrow waist + skills + MCP fat-server cap (~2–3). §3.
3. **God functions** — loops grow to 4K lines; builder pattern can hide bad decomposition. Accept with eyes open; pipeline stages + retries extracted help. §1/§2.
4. **Weak models miss skills** even with MUST language — invest in descriptions + evals, not louder prompts. §7.
5. **Naive truncation destroys observations** — head-only truncation loses ends of commands/paths. Head+tail keep with `elided_chars` warning. §4.
6. **Compression eating in-flight state** — never prune the most recent skill_view of a skill in use; keep the compression model cheap, summaries structured. §4.
7. **Truncated thinking-only or empty model outputs** happen at scale; have the nudge→prefill→retry→fallback ladder. §6.
8. **Streams that die silently** — stale-stream watchdog. §11/§6.
9. **Formats degrade over ITers** — malformed call retries, format-error counters, distinct fixes for length-vs-malformed. §3/§6.
10. **State drift between agent steps** in shell environments — fresh subshell/killpg per action, or long-lived exec container with `start_new_session=True`. §14.
11. **Skill sprawl** — near-duplicate skills; patch umbrella skills in place, never spawn variants. §7.
12. **Search-only skill discovery fails**; models skip optional catalogs; models skip pagination. §7.
13. **Config reloads racing sessions** — fsnotify reloads apply next session, never in-flight; snapshot caches pinned to session. §5/§8.
14. **Duplicated state rot** (caches/copies drifting from SSOT) — resolve-rule + auto-revert gates. §10.
15. **No budget = runaway loops** — step/cost/time caps + loop detection. §2.
16. ** Keys/secrets leaking into persistence** — scrub before persist; AES at rest. §10.
17. **Boot-time regression from laziness of consumers** — lazy loading + late-binding to keep startup light. §1.
18. **Announce loops** in multi-agent (each completion message re-triggers); push-based completion + guard. §9.
19. **Unicode steganography in shared skills** — strip Tags U+E0001–E007F + bidi overrides. §7.
20. **Rating agents by feature count instead of deployment priority** — pick by §1 priority table.

## §14 Per-harness "copy / avoid"

| Harness | Copy | Avoid |
|---|---|---|
| OpenClaw | enforced SDK path rules, lanes/queue steering modes, failover machinery, determinist prompts | path-level "SDK" as real decoupling (deep coupling persists); 1.5GB/6s footprint as aspirational; WS-only protocol |
| GoClaw orig | zero-dep static binary, human-readable state files, Cortex KG memory, 271-line loop | shipping zero failover in unattended daemons; no test coverage |
| GoClaw NLB | 8-stage pipeline w/ explicit loop signals, 3-tier memory + async consolidation, delegation modes, P0 blocking tests | 474K LOC growth-speed; license; PostgreSQL-Required for full features |
| PicoClaw | ToolLoop as reusable fn, parallel per-iteration tool execution, stability-tagged prompt parts, 10–20MB footprint | （only 95% AI-reviwed code — human review is cited but scarce) |
| ZeroClaw | SSOT/no-duplicate-state with auto-revert, allowed/excluded policy, tool-model-flex parsers w/ retry, per-skill trust tiers, hooks + Ed25519 signed intent, SOP engine | 126K-line transitional crate before v0.8; multi-backend optionality until much later |
| Hermes | prompt cache discipline + three-tier prompt, FailoverReason classification, repair ladders (nudge/prefill/retry), singleton thread-safe tool registry, seed-to-disk skills + manifest, profiles | 3,900-line run_conversation; 50+ AIAgent args; gateway as 16K-line monolith |
| gptme | ToolSpec auto-discovery (no boiler), Lessons system (pattern-triggered injection incl. negative patterns), `arewetiny` CI metric, hooks enum, JSONL logs | keep-it-tiny VS feature creep tension — the CI metric IS the control |
| mini-SWE-agent | exception-driven control flow, one bash tool + unified exec env signature, completion marker protocol, head+tail truncation, StrictUndefined, finally- trajectories, batch Docker worker pools | when unconstrained generalization needed; relies on modern models' tool fidelity |
| dsh | everything-a-plugin w/ reversible disposals, 7-stage guarded tool pipeline, ToolDefinition output/render split, isConcurrencySafe batching, append-only event log + deriveMessages, 2-tier skill disclosure, inbox steering verbs, | complexity of CORDIS-CAPE microkernel for small teams — start simpler (→§7 phases) |

## §15 Source notes

All content derives from the following vault notes (Claude `AI Agents/` folder, Oct–Jun 2026 study notebooks):
- `OpenClaw Architecture.md`, `GoClaw Architecture.md`, `PicoClaw Architecture.md`, `ZeroClaw Architecture.md`
- `Hermes Agent Architecture.md`, `gptme Architecture.md`, `mini-SWE-agent Architecture.md`
- `DeepSeek Harness Architecture - Everything is a Plugin.md`
- `Adding a Skill System to a Go Agent.md`
- `Agent Strengths Weaknesses and Creator Priorities.md`
