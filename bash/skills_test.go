package bash

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/smhanov/llmhub"

	"github.com/smhanov/agentloop"
)

// skillTree materializes files (relative paths under a fresh temp root)
// and returns the root, ready to hand to agentloop.WithSkills.
func skillTree(t *testing.T, files map[string]string) string {
	t.Helper()
	root := t.TempDir()
	for rel, content := range files {
		path := filepath.Join(root, rel)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

// skillMD renders a valid SKILL.md with the given frontmatter and body.
func skillMD(name, description, body string) string {
	return "---\nname: " + name + "\ndescription: " + description + "\n---\n\n" + body + "\n"
}

// fixtureSkillDir builds the one skill the composition tests use: its body
// instructs writing greeting.txt and running the bundled script, which the
// scripted model then "follows".
func fixtureSkillDir(t *testing.T) string {
	t.Helper()
	return skillTree(t, map[string]string{
		"greet-skill/SKILL.md": skillMD("greet-skill",
			"Use when the task asks for a greeting.",
			"Write a file named greeting.txt containing exactly: hello from the skill\n\n"+
				"To verify scripting, run the skill helper `scripts/hello.py` with Python "+
				"from the skill's base directory; it writes script-ran.txt."),
		"greet-skill/scripts/hello.py": "from pathlib import Path\n\n" +
			"print(\"hello from the script\")\n" +
			"Path(__file__).resolve().parent.parent.joinpath(\"script-ran.txt\").write_text(\"script ran\\n\")\n",
	})
}

// shellQuote single-quotes s for the bash tool's command line.
func shellQuote(s string) string {
	return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
}

// pythonInterpreter returns an available Python command for the script E2E
// test, skipping when the host has none.
func pythonInterpreter(t *testing.T) string {
	t.Helper()
	for _, name := range []string{"python3", "python"} {
		if _, err := exec.LookPath(name); err == nil {
			return name
		}
	}
	t.Skip("no python3/python on PATH: the skill script E2E needs a Python interpreter")
	return ""
}

// skillReadCall scripts a native skill_read tool call.
func skillReadCall(id, name string) *llmhub.ToolCallContent {
	raw, err := json.Marshal(map[string]string{"name": name})
	if err != nil {
		panic(err) // static test data; never fails
	}
	return &llmhub.ToolCallContent{ID: id, Name: "skill_read", Arguments: string(raw)}
}

// advertisedTools lists the tool names the model saw on one request.
func advertisedTools(req fakeRequest) []string {
	names := make([]string, 0, len(req.config.Tools))
	for _, tool := range req.config.Tools {
		names = append(names, tool.Name)
	}
	return names
}

// hasTool reports whether names contains want.
func hasTool(names []string, want string) bool {
	for _, n := range names {
		if n == want {
			return true
		}
	}
	return false
}

// systemOf returns the first request's system message text.
func systemOf(t *testing.T, req fakeRequest) string {
	t.Helper()
	if len(req.prompt) == 0 || req.prompt[0].Role != llmhub.RoleSystem {
		t.Fatalf("first prompt has no leading system message: %v", rolesOf(req.prompt))
	}
	return textOf(req.prompt[0])
}

// TestSkillsComposeWithBothProtocols is the composition matrix: with
// agentloop.WithSkills the construction-time system prompt keeps the
// ## Skills catalog and skill_read is advertised to the model in BOTH
// protocols; in tool mode the bash tool is advertised alongside it.
// Without WithSkills none of that appears.
func TestSkillsComposeWithBothProtocols(t *testing.T) {
	cases := []struct {
		name         string
		toolProtocol bool
		skills       bool
		wantCatalog  bool
	}{
		{"text+skills", false, true, true},
		{"tool+skills", true, true, true},
		{"text-skills", false, false, false},
		{"tool-skills", true, false, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			resetScript(t, submitTurn("THOUGHT: done", "done"))
			opts := []Option{WithEnvironment(NewLocalEnvironment(t.TempDir(), nil, 0))}
			if tc.toolProtocol {
				opts = append(opts, WithToolProtocol())
			}
			if tc.skills {
				opts = append(opts, agentloop.WithSkills(fixtureSkillDir(t)))
			}
			agent := newFakeAgent(t, opts...)

			if _, err := agent.Ask(context.Background(), "say hi"); err != nil {
				t.Fatalf("Ask: %v", err)
			}
			reqs := fakeRequests()
			if len(reqs) == 0 {
				t.Fatal("no provider requests captured")
			}

			// The construction-time system prompt survives the per-Ask
			// wiring: the catalog is there iff skills are on, and the task
			// text stays in the first user message.
			sys := systemOf(t, reqs[0])
			if got := strings.Contains(sys, "## Skills"); got != tc.wantCatalog {
				t.Errorf("system prompt has catalog = %v, want %v:\n%.200s", got, tc.wantCatalog, sys)
			}
			if tc.wantCatalog {
				for _, want := range []string{"greet-skill", "Use when the task asks for a greeting."} {
					if !strings.Contains(sys, want) {
						t.Errorf("catalog missing %q", want)
					}
				}
			}
			if strings.Contains(sys, "say hi") {
				t.Error("task text leaked into the system prompt")
			}
			if got := textOf(reqs[0].prompt[1]); !strings.Contains(got, "say hi") {
				t.Errorf("task text missing from the first user message: %.120q", got)
			}

			// The advertised tool list: skill_read iff skills are on, and
			// the bash tool only in tool-protocol mode.
			tools := advertisedTools(reqs[0])
			if got := hasTool(tools, "skill_read"); got != tc.skills {
				t.Errorf("skill_read advertised = %v, want %v (advertised: %v)", got, tc.skills, tools)
			}
			if got := hasTool(tools, bashToolName); got != tc.toolProtocol {
				t.Errorf("bash advertised = %v, want %v (advertised: %v)", got, tc.toolProtocol, tools)
			}
		})
	}
}

// TestSkillReadExecutesInBothProtocols is the E2E: a scripted model reads
// the fixture skill with skill_read and follows its instructions with a
// bash command, in both transports. The skill body arrives as a tool
// result (never prompt injection) and the run submits.
func TestSkillReadExecutesInBothProtocols(t *testing.T) {
	const (
		body    = "Write a file named greeting.txt containing exactly: hello from the skill"
		command = "printf 'hello from the skill' > greeting.txt"
		endMark = "greeting written"
	)

	for _, tc := range []struct {
		name   string
		tool   bool
		script []fakeTurn
	}{
		{
			name: "text",
			// The native skill_read call carries no fenced command, so the
			// text protocol appends its format-error guidance while the
			// loop still executes the registered skill_read tool.
			script: []fakeTurn{
				callTurn(skillReadCall("s1", "greet-skill")),
				cmdTurn("THOUGHT: follow the skill", command),
				submitTurn("THOUGHT: done", endMark),
			},
		},
		{
			name: "tool",
			tool: true,
			script: []fakeTurn{
				callTurn(skillReadCall("s1", "greet-skill")),
				toolCallTurn("c1", command),
				toolSubmitTurn(endMark),
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			resetScript(t, tc.script...)
			opts := []Option{
				WithEnvironment(NewLocalEnvironment(dir, nil, 0)),
				agentloop.WithSkills(fixtureSkillDir(t)),
			}
			if tc.tool {
				opts = append(opts, WithToolProtocol())
			}
			ans, err := newFakeAgent(t, opts...).Ask(context.Background(), "greet the world")
			if err != nil {
				t.Fatalf("Ask: %v", err)
			}
			if ans.ExitStatus != ExitSubmitted || ans.Submission != endMark+"\n" {
				t.Errorf("ExitStatus = %q, Submission = %q; want Submitted and %q",
					ans.ExitStatus, ans.Submission, endMark+"\n")
			}

			// The skill body was delivered as a tool result.
			var reads []string
			for _, tr := range ans.Tools {
				if tr.Name == "skill_read" {
					reads = append(reads, tr.Output)
				}
			}
			if len(reads) != 1 || !strings.Contains(reads[0], body) {
				t.Fatalf("skill_read trace = %v, want one result carrying the body", reads)
			}
			var results []string
			for _, m := range ans.Messages {
				if m.Role == llmhub.RoleTool {
					results = append(results, textOf(m))
				}
			}
			if len(results) == 0 || !strings.Contains(strings.Join(results, "\n"), body) {
				t.Errorf("no tool result message carries the skill body: %v", results)
			}

			// The model followed the skill: the file exists with the
			// prescribed content.
			data, err := os.ReadFile(filepath.Join(dir, "greeting.txt"))
			if err != nil {
				t.Fatalf("the skill's instruction was not followed: %v", err)
			}
			if string(data) != "hello from the skill" {
				t.Errorf("greeting.txt = %q, want %q", data, "hello from the skill")
			}
		})
	}
}

// TestSkillScriptDisclosureExecutesInBothProtocols is the issue-#2 E2E:
// the model learns the skill's absolute base directory and script names
// from the skill_read result (never from the L0 catalog), then runs the
// body-referenced script through the bash tool — both as an absolute path
// and as a cd-prefixed relative path, in both transports. Only the script
// output enters context; the source stays unreadable through skill_read.
func TestSkillScriptDisclosureExecutesInBothProtocols(t *testing.T) {
	const (
		scriptOutput = "hello from the script"
		endMark      = "script ran"
	)
	python := pythonInterpreter(t)

	transports := []struct {
		name string
		tool bool
	}{
		{name: "text"},
		{name: "tool", tool: true},
	}
	forms := []struct {
		name    string
		command func(skillDir string) string
	}{
		{
			name: "absolute",
			command: func(skillDir string) string {
				return python + " " + shellQuote(filepath.Join(skillDir, "scripts", "hello.py"))
			},
		},
		{
			name: "cd",
			command: func(skillDir string) string {
				return "cd " + shellQuote(skillDir) + " && " + python + " scripts/hello.py"
			},
		},
	}

	for _, transport := range transports {
		for _, form := range forms {
			t.Run(transport.name+"/"+form.name, func(t *testing.T) {
				skillRoot := fixtureSkillDir(t)
				skillDir, err := filepath.EvalSymlinks(filepath.Join(skillRoot, "greet-skill"))
				if err != nil {
					t.Fatal(err)
				}
				dir := t.TempDir()
				command := form.command(skillDir)

				var script []fakeTurn
				if transport.tool {
					script = []fakeTurn{
						callTurn(skillReadCall("s1", "greet-skill")),
						toolCallTurn("c1", command),
						toolSubmitTurn(endMark),
					}
				} else {
					script = []fakeTurn{
						callTurn(skillReadCall("s1", "greet-skill")),
						cmdTurn("THOUGHT: run the disclosed script", command),
						submitTurn("THOUGHT: done", endMark),
					}
				}
				resetScript(t, script...)

				opts := []Option{
					WithEnvironment(NewLocalEnvironment(dir, nil, 0)),
					agentloop.WithSkills(skillRoot),
				}
				if transport.tool {
					opts = append(opts, WithToolProtocol())
				}
				ans, err := newFakeAgent(t, opts...).Ask(context.Background(), "run the skill script")
				if err != nil {
					t.Fatalf("Ask: %v", err)
				}
				if ans.ExitStatus != ExitSubmitted || ans.Submission != endMark+"\n" {
					t.Errorf("ExitStatus = %q, Submission = %q; want Submitted and %q",
						ans.ExitStatus, ans.Submission, endMark+"\n")
				}

				// The L0 catalog stays path-free: disclosure happens only
				// when a skill is actually loaded.
				reqs := fakeRequests()
				if len(reqs) == 0 {
					t.Fatal("no provider requests captured")
				}
				if sys := systemOf(t, reqs[0]); strings.Contains(sys, skillDir) ||
					strings.Contains(sys, "scripts/hello.py") {
					t.Errorf("catalog/system prompt leaked the skill path:\n%s", sys)
				}

				// The load result — not the catalog — disclosed the base
				// directory and the script path, and it never carried the
				// script source.
				var reads []string
				for _, tr := range ans.Tools {
					if tr.Name == "skill_read" {
						reads = append(reads, tr.Output)
					}
				}
				if len(reads) != 1 {
					t.Fatalf("skill_read trace = %v, want one result", reads)
				}
				read := reads[0]
				if want := "base directory: " + skillDir; !strings.Contains(read, want) {
					t.Errorf("skill_read result missing %q:\n%s", want, read)
				}
				if !strings.Contains(read, "- scripts/hello.py") {
					t.Errorf("skill_read result missing the script listing:\n%s", read)
				}
				if strings.Contains(read, "Path(__file__)") {
					t.Errorf("script source leaked through skill_read:\n%s", read)
				}

				// The script ran: its output is the only part of it in the
				// transcript...
				observations := strings.Join(observationTexts(ans.Messages), "\n")
				if !strings.Contains(observations, scriptOutput) {
					t.Errorf("script output missing from the observations:\n%s", observations)
				}
				if strings.Contains(observations, "Path(__file__)") {
					t.Errorf("script source leaked into the observations:\n%s", observations)
				}

				// ...and its side effect landed next to the skill.
				data, err := os.ReadFile(filepath.Join(skillDir, "script-ran.txt"))
				if err != nil {
					t.Fatalf("the skill's script did not run: %v", err)
				}
				if string(data) != "script ran\n" {
					t.Errorf("script-ran.txt = %q, want %q", data, "script ran\n")
				}
			})
		}
	}
}

// echoTool is a caller-registered stand-in tool for the composition tests.
func echoTool() agentloop.Tool {
	return agentloop.Tool{
		Def: llmhub.NewTool("echo", "Echo the arguments back", map[string]any{
			"type":       "object",
			"properties": map[string]any{"text": map[string]any{"type": "string"}},
		}),
		Run: func(ctx context.Context, args json.RawMessage) (string, error) {
			return "echo:" + string(args), nil
		},
	}
}

// TestCallerOptionsCompose pins the merge semantics around the preset: a
// construction-time caller system prompt overrides the rendered template
// (and still gets the skills catalog appended), and a construction-time
// caller tool survives into tool mode alongside skill_read and bash.
func TestCallerOptionsCompose(t *testing.T) {
	resetScript(t, toolSubmitTurn("done"))
	agent := newFakeAgent(t,
		WithEnvironment(NewLocalEnvironment(t.TempDir(), nil, 0)),
		WithToolProtocol(),
		agentloop.WithSkills(fixtureSkillDir(t)),
		agentloop.WithSystemPrompt("House style: be terse."),
		agentloop.WithTools(echoTool()),
	)

	if _, err := agent.Ask(context.Background(), "task"); err != nil {
		t.Fatalf("Ask: %v", err)
	}
	reqs := fakeRequests()
	sys := systemOf(t, reqs[0])
	if !strings.HasPrefix(sys, "House style: be terse.") {
		t.Errorf("caller system prompt did not override the preset template: %.80q", sys)
	}
	if !strings.Contains(sys, "## Skills") {
		t.Error("catalog missing from the overridden prompt")
	}
	tools := advertisedTools(reqs[0])
	for _, want := range []string{"echo", "skill_read", bashToolName} {
		if !hasTool(tools, want) {
			t.Errorf("tool %q missing from the advertised set %v", want, tools)
		}
	}
}

// TestSystemPromptByteStable pins AC-style prompt stability: the system
// prompt is rendered once at construction and is byte-identical across
// every step of one Ask and across separate Asks of the same agent.
func TestSystemPromptByteStable(t *testing.T) {
	resetScript(t,
		cmdTurn("THOUGHT: one", "echo one"),
		cmdTurn("THOUGHT: two", "echo two"),
		submitTurn("THOUGHT: done", "done"),
		submitTurn("THOUGHT: done", "done"), // the second Ask
	)
	agent := newFakeAgent(t,
		WithEnvironment(NewLocalEnvironment(t.TempDir(), nil, 0)),
		agentloop.WithSkills(fixtureSkillDir(t)),
	)

	for i := 1; i <= 2; i++ {
		if _, err := agent.Ask(context.Background(), "task"); err != nil {
			t.Fatalf("Ask %d: %v", i, err)
		}
	}
	reqs := fakeRequests()
	if len(reqs) != 4 { // 3 steps in Ask one, 1 in Ask two
		t.Fatalf("captured %d requests, want 4", len(reqs))
	}
	first := systemOf(t, reqs[0])
	for i, req := range reqs[1:] {
		if got := systemOf(t, req); got != first {
			t.Errorf("request %d system prompt drifted:\nfirst:  %q\nnow:    %q", i+1, first, got)
		}
	}
	if !strings.Contains(first, "## Skills") {
		t.Error("stability probe lost the skills catalog")
	}
}
