//go:build live

// Live skills-pack validation: agentloop must load and serve a real,
// foreign skills pack end to end — the reporter's scenario before they
// build their app on the library. The pack is never referenced from the
// repository: this test reads AGENTLOOP_LIVE_SKILLS_ROOT, a colon-separated
// list of skills-root directories (each root's children are skill folders),
// and skips with instructions when the variable is unset. Nothing
// pack-specific is asserted or logged: assertions are structural
// (counts, agentloop's own stable issue-reason and refusal strings), and
// failure messages carry only counts, booleans, and indices — never skill
// names, paths, or content. Run it:
//
//	AGENTLOOP_LIVE_SKILLS_ROOT=/path/to/skills[/another/root] \
//		go test -tags live . -run TestLiveSkillsPack -v
//
// Checks: the whole pack loads with zero SkillIssues and per-skill
// invariants hold (name = folder basename, description within the cap,
// every linked/script file resolvable under the skill directory); every
// supplied root loads as a root; the catalog renders complete, sorted, and
// byte-identical across two constructions without leaking skill paths;
// skill_read serves bodies with the script-disclosure footer and jailed
// linked-file reads; and a scripted-model bash-preset run loads a skill
// through skill_read and submits — no network, no real model, and no pack
// script executed. The file is excluded from the default suite by the
// `live` build tag.
package agentloop_test

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"
	"unicode/utf8"

	"github.com/smhanov/llmhub"

	"github.com/smhanov/agentloop"
	"github.com/smhanov/agentloop/bash"
)

// liveRootsEnv is the only channel through which the pack's location
// enters the test.
const liveRootsEnv = "AGENTLOOP_LIVE_SKILLS_ROOT"

// liveSubmitMarker is this test's own bash-preset submit marker.
const liveSubmitMarker = "AGENTLOOP_LIVE_PACK_SUBMIT"

// liveRoots returns the colon-separated skills roots from the environment,
// skipping with instructions when it is unset.
func liveRoots(t *testing.T) []string {
	t.Helper()
	raw := os.Getenv(liveRootsEnv)
	if strings.TrimSpace(raw) == "" {
		t.Skipf("%s not set; point it at one or more colon-separated skills-root directories (each root's children are skill folders), e.g. %s=/path/to/skills go test -tags live . -run TestLiveSkillsPack -v",
			liveRootsEnv, liveRootsEnv)
	}
	var roots []string
	for _, part := range strings.Split(raw, ":") {
		if part = strings.TrimSpace(part); part != "" {
			roots = append(roots, part)
		}
	}
	if len(roots) == 0 {
		t.Fatalf("%s carries no usable paths", liveRootsEnv)
	}
	return roots
}

// livePack loads the environment's roots. Load errors are fatal; issues are
// returned for the caller to judge (the zero-issue gate lives in the load
// subtest).
func livePack(t *testing.T) ([]agentloop.Skill, []agentloop.SkillIssue) {
	t.Helper()
	skills, issues, err := agentloop.LoadSkills(liveRoots(t)...)
	if err != nil {
		t.Fatalf("LoadSkills: %v", err)
	}
	return skills, issues
}

// issueHistogram counts issues by reason — agentloop's own stable strings,
// safe to log; the affected skills' identities stay unlogged.
func issueHistogram(issues []agentloop.SkillIssue) map[string]int {
	histo := make(map[string]int)
	for _, issue := range issues {
		histo[issue.Reason]++
	}
	return histo
}

// liveTop returns rel's first path segment.
func liveTop(rel string) string {
	if i := strings.IndexAny(rel, `/\`); i >= 0 {
		return rel[:i]
	}
	return rel
}

// liveEscapes reports whether rel, a path relative to a skill directory,
// points outside it.
func liveEscapes(rel string) bool {
	return filepath.IsAbs(rel) || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

// liveCatalog renders the catalog body the loader's documented rendering
// must produce: one "- name: description" line per skill, sorted by name.
func liveCatalog(skills []agentloop.Skill) string {
	sorted := append([]agentloop.Skill(nil), skills...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].Name < sorted[j].Name })
	var b strings.Builder
	for _, s := range sorted {
		fmt.Fprintf(&b, "- %s: %s\n", s.Name, s.Description)
	}
	return b.String()
}

// liveSkillReadTool builds an agent over the given roots and returns its
// registered skill_read tool.
func liveSkillReadTool(t *testing.T, roots []string) agentloop.Tool {
	t.Helper()
	agent, err := agentloop.New(llmhub.Wrap(&liveProvider{}), agentloop.WithSkills(roots...))
	if err != nil {
		t.Fatalf("construct agent: %v", err)
	}
	for _, tool := range agent.Tools() {
		if tool.Def.Name == "skill_read" {
			return tool
		}
	}
	t.Fatal("skill_read tool not registered")
	return agentloop.Tool{}
}

// liveRunJSON marshals a string-map argument object for a scripted call.
func liveRunJSON(t *testing.T, args map[string]string) string {
	t.Helper()
	raw, err := json.Marshal(args)
	if err != nil {
		t.Fatalf("marshal tool arguments: %v", err)
	}
	return string(raw)
}

// TestLiveSkillsPack validates a real skills pack end to end. See the file
// header for the invocation and the confidentiality discipline.
func TestLiveSkillsPack(t *testing.T) {
	ctx := context.Background()

	t.Run("load", func(t *testing.T) {
		skills, issues := livePack(t)
		if len(skills) == 0 {
			t.Fatal("loaded 0 skills")
		}
		if n := len(issues); n > 0 {
			t.Errorf("loaded %d skills with %d issues: %v", len(skills), n, issueHistogram(issues))
		}
		for i := range skills {
			s := &skills[i]
			if !filepath.IsAbs(s.Dir) {
				t.Fatalf("skill #%d: Dir not absolute", i)
			}
			if base := filepath.Base(s.Dir); s.Name != base {
				t.Fatalf("skill #%d: name != folder basename", i)
			}
			if n := utf8.RuneCountInString(s.Description); n < 1 || n > 1024 {
				t.Fatalf("skill #%d: description %d runes, want 1..1024", i, n)
			}
			if strings.TrimSpace(s.Body) == "" {
				t.Fatalf("skill #%d: empty body", i)
			}
			for j, f := range s.LinkedFiles {
				if top := liveTop(f); top != "references" && top != "assets" {
					t.Fatalf("skill #%d: linked file #%d not under references/ or assets/", i, j)
				}
				if _, err := os.Stat(filepath.Join(s.Dir, filepath.FromSlash(f))); err != nil {
					t.Fatalf("skill #%d: linked file #%d missing on disk", i, j)
				}
			}
			for j, f := range s.ScriptFiles {
				if top := liveTop(f); top != "scripts" {
					t.Fatalf("skill #%d: script file #%d not under scripts/", i, j)
				}
				real, err := filepath.EvalSymlinks(filepath.Join(s.Dir, filepath.FromSlash(f)))
				if err != nil {
					t.Fatalf("skill #%d: script file #%d does not resolve", i, j)
				}
				if rel, err := filepath.Rel(s.Dir, real); err != nil || liveEscapes(rel) {
					t.Fatalf("skill #%d: script file #%d resolves outside the skill directory", i, j)
				}
			}
		}
	})

	t.Run("roots", func(t *testing.T) {
		// Each supplied root must load as a root (its children are skill
		// folders): at least one skill, zero issues, no shape misfire.
		for i, root := range liveRoots(t) {
			skills, issues, err := agentloop.LoadSkills(root)
			if err != nil {
				t.Fatalf("LoadSkills(root #%d): %v", i, err)
			}
			if len(skills) == 0 {
				t.Errorf("root #%d loaded 0 skills", i)
			}
			if n := len(issues); n > 0 {
				t.Errorf("root #%d loaded %d skills with %d issues: %v",
					i, len(skills), n, issueHistogram(issues))
			}
		}
	})

	t.Run("catalog", func(t *testing.T) {
		skills, _ := livePack(t)
		if len(skills) == 0 {
			t.Fatal("loaded 0 skills")
		}
		roots := liveRoots(t)

		// Two independent constructions over one scripted provider: the
		// catalog is frozen per construction, so byte-identical system
		// prompts across the two Asks show rendering determinism.
		prov := &liveProvider{script: []liveTurn{
			{text: "one"}, {text: "two"},
		}}
		client := llmhub.Wrap(prov)
		first, err := agentloop.New(client, agentloop.WithSkills(roots...))
		if err != nil {
			t.Fatalf("construct first agent: %v", err)
		}
		second, err := agentloop.New(client, agentloop.WithSkills(roots...))
		if err != nil {
			t.Fatalf("construct second agent: %v", err)
		}
		if _, err := first.Ask(ctx, "catalog probe one"); err != nil {
			t.Fatalf("first ask: %v", err)
		}
		if _, err := second.Ask(ctx, "catalog probe two"); err != nil {
			t.Fatalf("second ask: %v", err)
		}
		reqs := prov.captured()
		if len(reqs) != 2 {
			t.Fatalf("provider saw %d requests, want 2", len(reqs))
		}
		sys := liveSystemText(reqs[0].prompt)
		if !strings.Contains(sys, "## Skills") {
			t.Error("system prompt missing the ## Skills catalog section")
		}
		if want := liveCatalog(skills); !strings.Contains(sys, want) {
			t.Errorf("catalog body mismatch: %d skills rendered into a %d-byte system prompt",
				len(skills), len(sys))
		}
		for i := range skills {
			if strings.Contains(sys, skills[i].Dir) {
				t.Errorf("catalog leaks skill #%d's base path", i)
			}
		}
		if sys2 := liveSystemText(reqs[1].prompt); sys != sys2 {
			t.Error("catalog not byte-identical across two constructions")
		}
	})

	t.Run("skill_read", func(t *testing.T) {
		roots := liveRoots(t)
		skills, _ := livePack(t)
		var scriptSkill, linkedSkill *agentloop.Skill
		for i := range skills {
			if scriptSkill == nil && len(skills[i].ScriptFiles) > 0 {
				scriptSkill = &skills[i]
			}
			if linkedSkill == nil && len(skills[i].LinkedFiles) > 0 {
				linkedSkill = &skills[i]
			}
		}

		t.Run("script disclosure", func(t *testing.T) {
			if scriptSkill == nil {
				t.Skip("no script-backed skill in this pack; disclosure untested")
			}
			tool := liveSkillReadTool(t, roots)
			out, err := tool.Run(ctx, json.RawMessage(liveRunJSON(t, map[string]string{"name": scriptSkill.Name})))
			if err != nil {
				t.Fatalf("body read failed: %v", err)
			}
			if !strings.HasPrefix(out, scriptSkill.Body) {
				t.Errorf("body read does not start with the skill body (result %d bytes)", len(out))
			}
			if want := "base directory: " + scriptSkill.Dir + " (relative paths in this skill resolve from here)"; !strings.Contains(out, want) {
				t.Errorf("body read missing the base-directory line (result %d bytes)", len(out))
			}
			missing := 0
			for _, f := range scriptSkill.ScriptFiles {
				if !strings.Contains(out, "- "+f) {
					missing++
				}
			}
			if missing > 0 {
				t.Errorf("body read missing %d of %d disclosed script paths", missing, len(scriptSkill.ScriptFiles))
			}
		})

		t.Run("linked file read", func(t *testing.T) {
			if linkedSkill == nil {
				t.Skip("no skill with linked files; linked-file reads untested")
			}
			tool := liveSkillReadTool(t, roots)
			rel := linkedSkill.LinkedFiles[0]
			out, err := tool.Run(ctx, json.RawMessage(liveRunJSON(t, map[string]string{
				"name": linkedSkill.Name, "file": rel,
			})))
			if err != nil {
				t.Fatalf("linked-file read failed: %v", err)
			}
			want, err := os.ReadFile(filepath.Join(linkedSkill.Dir, filepath.FromSlash(rel)))
			if err != nil {
				t.Fatalf("reading the linked file from disk for comparison failed")
			}
			if out != string(want) {
				t.Errorf("linked-file read = %d bytes, want the %d bytes on disk", len(out), len(want))
			}
		})

		t.Run("jail", func(t *testing.T) {
			if scriptSkill == nil {
				t.Skip("no script-backed skill in this pack; jail untested")
			}
			tool := liveSkillReadTool(t, roots)
			out, err := tool.Run(ctx, json.RawMessage(liveRunJSON(t, map[string]string{
				"name": scriptSkill.Name, "file": scriptSkill.ScriptFiles[0],
			})))
			if err == nil {
				t.Errorf("a scripts/ read through skill_read was not refused (got %d bytes)", len(out))
			} else if !strings.Contains(err.Error(), "only files under references/ or assets/") {
				t.Errorf("scripts/ read refused with an unexpected reason: %v", err)
			}
		})
	})

	t.Run("bash preset usability", func(t *testing.T) {
		// The reporter's scenario: a bash-preset agent built over the pack
		// with WithSkills, driven by a scripted model (no network, no real
		// LLM, no pack script executed) — load the first cataloged skill
		// with skill_read, then submit.
		skills, _ := livePack(t)
		if len(skills) == 0 {
			t.Fatal("loaded 0 skills")
		}
		first := &skills[0]
		roots := liveRoots(t)

		prov := &liveProvider{script: []liveTurn{
			{calls: []*llmhub.ToolCallContent{
				llmhub.ToolCall("live-1", "skill_read", liveRunJSON(t, map[string]string{"name": first.Name})),
			}},
			{calls: []*llmhub.ToolCallContent{
				llmhub.ToolCall("live-2", "bash", liveRunJSON(t, map[string]string{
					"command": "echo '" + liveSubmitMarker + "'",
				})),
			}},
		}}
		agent, err := bash.New(llmhub.Wrap(prov),
			bash.WithEnvironment(bash.NewLocalEnvironment(t.TempDir(), nil, 0)),
			bash.WithToolProtocol(),
			bash.WithSubmitMarker(liveSubmitMarker),
			agentloop.WithSkills(roots...),
		)
		if err != nil {
			t.Fatalf("construct bash preset agent: %v", err)
		}
		ans, err := agent.Ask(ctx, "Load the first skill from the catalog with skill_read, then submit.")
		if err != nil {
			t.Fatalf("Ask: %v", err)
		}

		if ans.ExitStatus != bash.ExitSubmitted {
			t.Errorf("ExitStatus = %q, want %q", ans.ExitStatus, bash.ExitSubmitted)
		}
		if len(ans.Submission) != 0 {
			t.Errorf("Submission = %d bytes, want empty", len(ans.Submission))
		}
		var reads, runs int
		for _, tr := range ans.Tools {
			switch tr.Name {
			case "skill_read":
				reads++
				if tr.Err != nil {
					t.Error("skill_read execution errored")
				} else if tr.Output == "" {
					t.Error("skill_read produced an empty result")
				}
			case "bash":
				runs++
			}
		}
		if reads != 1 {
			t.Errorf("skill_read executions = %d, want 1", reads)
		}
		if runs != 1 {
			t.Errorf("bash executions = %d, want 1 (only the submit echo; no pack script ran)", runs)
		}

		// What the model saw: the frozen catalog in the system prompt, then
		// the skill_read call and its result in the second request.
		reqs := prov.captured()
		if len(reqs) != 2 {
			t.Fatalf("provider saw %d requests, want 2", len(reqs))
		}
		sys := liveSystemText(reqs[0].prompt)
		if !strings.Contains(sys, "## Skills") {
			t.Error("preset system prompt missing the skills catalog")
		}
		for i := range skills {
			if strings.Contains(sys, skills[i].Dir) {
				t.Errorf("system prompt leaks skill #%d's base path", i)
			}
		}
		var toolMsgs, callMsgs int
		for _, m := range reqs[1].prompt {
			switch m.Role {
			case llmhub.RoleTool:
				toolMsgs++
			case llmhub.RoleAssistant:
				if len(liveCalls(m)) > 0 {
					callMsgs++
				}
			}
		}
		if toolMsgs != 1 {
			t.Errorf("tool-result messages in the second request = %d, want 1 (the skill_read result reached the model)", toolMsgs)
		}
		if callMsgs != 1 {
			t.Errorf("assistant tool-call messages in the second request = %d, want 1 (the skill_read call)", callMsgs)
		}
	})
}

// liveTurn is one scripted provider response: final text, native tool
// calls, or both.
type liveTurn struct {
	text  string
	calls []*llmhub.ToolCallContent
}

// liveRequest records one provider call's prompt (snapshotted, so
// later transcript appends cannot mutate it).
type liveRequest struct {
	prompt []*llmhub.Message
}

// liveProvider replays a fixed script of turns — a fake LLM, so the
// live tests never touch a network or a real model.
type liveProvider struct {
	mu       sync.Mutex
	script   []liveTurn
	requests []liveRequest
	index    int
}

// Name implements llmhub.Provider.
func (p *liveProvider) Name() string { return "scripted" }

// Generate implements llmhub.Provider.
func (p *liveProvider) Generate(ctx context.Context, prompt []*llmhub.Message, opts ...llmhub.Option) (*llmhub.Response, error) {
	turn, err := p.next(ctx, prompt)
	if err != nil {
		return nil, err
	}
	resp := &llmhub.Response{ID: fmt.Sprintf("scripted-%d", p.count())}
	if turn.text != "" {
		resp.Content = append(resp.Content, llmhub.Text(turn.text))
	}
	for _, call := range turn.calls {
		resp.Content = append(resp.Content, call)
	}
	return resp, nil
}

// Stream implements llmhub.Provider (unused by these tests; synthesized
// from the same script).
func (p *liveProvider) Stream(ctx context.Context, prompt []*llmhub.Message, opts ...llmhub.Option) (<-chan llmhub.StreamChunk, error) {
	turn, err := p.next(ctx, prompt)
	if err != nil {
		return nil, err
	}
	ch := make(chan llmhub.StreamChunk, 2+len(turn.calls))
	defer close(ch)
	if turn.text != "" {
		ch <- llmhub.StreamChunk{Delta: turn.text}
	}
	for _, call := range turn.calls {
		ch <- llmhub.StreamChunk{ToolCalls: []*llmhub.ToolCallContent{call}}
	}
	ch <- llmhub.StreamChunk{Done: true}
	return ch, nil
}

// next pops the next scripted turn and records the request.
func (p *liveProvider) next(ctx context.Context, prompt []*llmhub.Message) (liveTurn, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return liveTurn{}, err
	}
	p.requests = append(p.requests, liveRequest{prompt: liveSnapshotPrompt(prompt)})
	if p.index >= len(p.script) {
		return liveTurn{}, fmt.Errorf("scripted provider: script exhausted at request %d", p.index+1)
	}
	turn := p.script[p.index]
	p.index++
	return turn, nil
}

// count returns the number of requests served so far.
func (p *liveProvider) count() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.index
}

// captured returns the provider calls recorded so far.
func (p *liveProvider) captured() []liveRequest {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]liveRequest(nil), p.requests...)
}

// liveSnapshotPrompt copies a message list so later transcript appends
// cannot mutate captured requests.
func liveSnapshotPrompt(msgs []*llmhub.Message) []*llmhub.Message {
	out := make([]*llmhub.Message, 0, len(msgs))
	for _, m := range msgs {
		if m == nil {
			continue
		}
		out = append(out, &llmhub.Message{
			Role:    m.Role,
			Content: append([]llmhub.ContentPart(nil), m.Content...),
			Meta:    m.Meta,
		})
	}
	return out
}

// liveSystemText returns the first system message's text ("" when absent).
func liveSystemText(msgs []*llmhub.Message) string {
	for _, m := range msgs {
		if m != nil && m.Role == llmhub.RoleSystem {
			return liveText(m)
		}
	}
	return ""
}

// liveText joins the text parts of a message.
func liveText(m *llmhub.Message) string {
	var b strings.Builder
	for _, part := range m.Content {
		if tc, ok := part.(*llmhub.TextContent); ok {
			b.WriteString(tc.Text)
		}
	}
	return b.String()
}

// liveCalls lists the tool-call parts of a message.
func liveCalls(m *llmhub.Message) []*llmhub.ToolCallContent {
	var calls []*llmhub.ToolCallContent
	for _, part := range m.Content {
		if call, ok := part.(*llmhub.ToolCallContent); ok {
			calls = append(calls, call)
		}
	}
	return calls
}
