package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/smhanov/llmhub"
)

// runCLI executes run with in-memory streams.
func runCLI(t *testing.T, args []string, stdin string) (code int, stdout string, stderr string) {
	t.Helper()
	var out, errBuf strings.Builder
	code = run(args, strings.NewReader(stdin), &out, &errBuf)
	return code, out.String(), errBuf.String()
}

// newSerperStub starts a fake Serper API returning one organic result
// and records the API key header it received.
func newSerperStub(t *testing.T) (url string, gotKey *string) {
	t.Helper()
	key := ""
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		key = r.Header.Get("X-API-KEY")
		w.Write([]byte(`{"organic":[{"title":"Go 1.22 released","link":"https://go.dev/blog/go1.22","snippet":"The newest Go release"}]}`))
	}))
	t.Cleanup(srv.Close)
	return srv.URL, &key
}

// searchScript returns the two scripted turns of a search run: one
// web_search call, then the grounded final answer.
func searchScript() []fakeTurn {
	return []fakeTurn{
		callTurn(llmhub.ToolCall("call-1", "web_search", `{"q":"latest Go release"}`)),
		usageTurn("Go 1.22 is the most recent release [1].", llmhub.UsageMetadata{
			PromptTokens: 120, CompletionTokens: 40, TotalTokens: 160, Cost: 0.0042,
		}),
	}
}

// decodedAnswer mirrors the CLI's JSON answer shape for assertions.
type decodedAnswer struct {
	Agent      string `json:"agent"`
	Text       string `json:"text"`
	Status     string `json:"status"`
	Steps      int    `json:"steps"`
	ToolCalls  int    `json:"tool_calls"`
	ExitStatus string `json:"exit_status"`
	Submission string `json:"submission"`
	Sources    []struct {
		Title string `json:"title"`
		URL   string `json:"url"`
	} `json:"sources"`
	Usage struct {
		PromptTokens     int     `json:"prompt_tokens"`
		CompletionTokens int     `json:"completion_tokens"`
		TotalTokens      int     `json:"total_tokens"`
		Cost             float64 `json:"cost"`
	} `json:"usage"`
	Timing *struct {
		TotalMS float64 `json:"total_ms"`
	} `json:"timing"`
	Tools []struct {
		Name       string  `json:"name"`
		DurationMS float64 `json:"duration_ms"`
	} `json:"tools"`
}

func decodeJSON(t *testing.T, data string) decodedAnswer {
	t.Helper()
	var ans decodedAnswer
	if err := json.Unmarshal([]byte(data), &ans); err != nil {
		t.Fatalf("decode JSON output: %v\n%s", err, data)
	}
	return ans
}

// AC1: -agent search -json with a fake provider and a fake Serper prints
// a JSON object with text, sources, usage, steps, and the tool trace.
func TestSearchJSONOutput(t *testing.T) {
	resetFake(t, searchScript()...)
	baseURL, gotKey := newSerperStub(t)
	args := []string{
		"-provider", "fake",
		"-agent", "search",
		"-serper-key", "serper-test-key",
		"-serper-base-url", baseURL,
		"-json",
		"-prompt", "Which Go version was released most recently?",
	}
	code, out, stderr := runCLI(t, args, "")
	if code != 0 {
		t.Fatalf("exit code = %d, stderr: %s", code, stderr)
	}
	ans := decodeJSON(t, out)
	if ans.Agent != "search" {
		t.Errorf("agent = %q, want search", ans.Agent)
	}
	if !strings.Contains(ans.Text, "Go 1.22") {
		t.Errorf("text = %q, want it to mention Go 1.22", ans.Text)
	}
	if ans.Status != "completed" {
		t.Errorf("status = %q, want completed", ans.Status)
	}
	if ans.Steps != 2 || ans.ToolCalls != 1 {
		t.Errorf("steps = %d, tool calls = %d, want 2 and 1", ans.Steps, ans.ToolCalls)
	}
	if len(ans.Sources) != 1 || ans.Sources[0].URL != "https://go.dev/blog/go1.22" {
		t.Fatalf("sources = %+v, want one result pointing at the Go blog", ans.Sources)
	}
	if ans.Usage.TotalTokens != 160 || ans.Usage.Cost == 0 {
		t.Errorf("usage = %+v, want scripted totals and a cost", ans.Usage)
	}
	if ans.Timing == nil {
		t.Error("timing missing from JSON output")
	}
	if len(ans.Tools) != 1 || ans.Tools[0].Name != "web_search" {
		t.Errorf("tools = %+v, want one web_search entry", ans.Tools)
	}
	if *gotKey != "serper-test-key" {
		t.Errorf("Serper API key header = %q, want serper-test-key", *gotKey)
	}
}

// AC2: -chat over piped stdin runs one conversation; the second request
// contains turn one's user message and answer.
func TestChatSharesHistory(t *testing.T) {
	resetFake(t, textTurn("answer one"), textTurn("answer two"))
	args := []string{"-provider", "fake", "-agent", "plain", "-chat"}
	code, out, stderr := runCLI(t, args, "first question\nsecond question\n")
	if code != 0 {
		t.Fatalf("exit code = %d, stderr: %s", code, stderr)
	}
	if !strings.Contains(out, "answer one") || !strings.Contains(out, "answer two") {
		t.Errorf("stdout = %q, want both turn answers", out)
	}
	reqs := fakeRequests()
	if len(reqs) != 2 {
		t.Fatalf("provider requests = %d, want 2", len(reqs))
	}
	msgs := reqs[1].prompt
	wantRoles := []llmhub.Role{llmhub.RoleUser, llmhub.RoleAssistant, llmhub.RoleUser}
	if got := rolesOf(msgs); len(got) != len(wantRoles) {
		t.Fatalf("second request roles = %v, want %v", got, wantRoles)
	}
	if textOf(msgs[0]) != "first question" || textOf(msgs[1]) != "answer one" {
		t.Errorf("second request lost turn one history: %q, %q", textOf(msgs[0]), textOf(msgs[1]))
	}
	if textOf(msgs[2]) != "second question" {
		t.Errorf("second request query = %q, want second question", textOf(msgs[2]))
	}
}

// A blank stdin line ends the chat session cleanly.
func TestChatBlankLineEndsSession(t *testing.T) {
	resetFake(t, textTurn("only answer"))
	args := []string{"-provider", "fake", "-agent", "plain", "-chat"}
	code, out, stderr := runCLI(t, args, "only turn\n\nnever asked\n")
	if code != 0 {
		t.Fatalf("exit code = %d, stderr: %s", code, stderr)
	}
	if !strings.Contains(out, "only answer") || strings.Contains(out, "never asked") {
		t.Errorf("stdout = %q, want the first answer only", out)
	}
	if reqs := fakeRequests(); len(reqs) != 1 {
		t.Fatalf("provider requests = %d, want 1 (session ended at the blank line)", len(reqs))
	}
}

// With -chat -json each turn prints one JSON object.
func TestChatJSONPerTurn(t *testing.T) {
	resetFake(t, textTurn("answer one"), textTurn("answer two"))
	args := []string{"-provider", "fake", "-agent", "plain", "-chat", "-json"}
	code, out, stderr := runCLI(t, args, "one\ntwo\n")
	if code != 0 {
		t.Fatalf("exit code = %d, stderr: %s", code, stderr)
	}
	dec := json.NewDecoder(strings.NewReader(out))
	want := []string{"answer one", "answer two"}
	for i, w := range want {
		var ans decodedAnswer
		if err := dec.Decode(&ans); err != nil {
			t.Fatalf("decode turn %d JSON: %v\n%s", i+1, err, out)
		}
		if ans.Text != w {
			t.Errorf("turn %d text = %q, want %q", i+1, ans.Text, w)
		}
	}
}

// AC3: exit codes — 0 success, 2 budget exhausted, 1 construction and
// input errors.
func TestExitCodes(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		resetFake(t, textTurn("done"))
		code, _, stderr := runCLI(t, []string{"-provider", "fake", "-prompt", "hello"}, "")
		if code != 0 {
			t.Fatalf("exit code = %d, stderr: %s", code, stderr)
		}
	})
	t.Run("budget exhausted", func(t *testing.T) {
		resetFake(t, callTurn(llmhub.ToolCall("c1", "no_such_tool", `{}`)))
		code, _, stderr := runCLI(t, []string{"-provider", "fake", "-max-steps", "1", "-prompt", "loop"}, "")
		if code != 2 {
			t.Fatalf("exit code = %d, want 2, stderr: %s", code, stderr)
		}
	})
	t.Run("unknown provider", func(t *testing.T) {
		code, _, stderr := runCLI(t, []string{"-provider", "no-such-provider", "-prompt", "x"}, "")
		if code != 1 {
			t.Fatalf("exit code = %d, want 1", code)
		}
		if !strings.Contains(stderr, "no-such-provider") {
			t.Errorf("stderr = %q, want the provider name", stderr)
		}
	})
	t.Run("missing prompt", func(t *testing.T) {
		code, _, _ := runCLI(t, []string{"-provider", "fake"}, "")
		if code != 1 {
			t.Fatalf("exit code = %d, want 1", code)
		}
	})
	t.Run("prompt and prompt-file", func(t *testing.T) {
		code, _, _ := runCLI(t, []string{"-provider", "fake", "-prompt", "a", "-prompt-file", "b"}, "")
		if code != 1 {
			t.Fatalf("exit code = %d, want 1", code)
		}
	})
	t.Run("unreadable prompt file", func(t *testing.T) {
		code, _, _ := runCLI(t, []string{"-provider", "fake", "-prompt-file", filepath.Join(t.TempDir(), "absent.md")}, "")
		if code != 1 {
			t.Fatalf("exit code = %d, want 1", code)
		}
	})
	t.Run("missing skills directory", func(t *testing.T) {
		code, _, _ := runCLI(t, []string{"-provider", "fake", "-skills", filepath.Join(t.TempDir(), "absent"), "-prompt", "x"}, "")
		if code != 1 {
			t.Fatalf("exit code = %d, want 1", code)
		}
	})
	t.Run("bad env pair", func(t *testing.T) {
		code, _, _ := runCLI(t, []string{"-provider", "fake", "-agent", "bash", "-env", "NO_EQUALS", "-prompt", "x"}, "")
		if code != 1 {
			t.Fatalf("exit code = %d, want 1", code)
		}
	})
}

// AC4: -stream puts answer deltas on stdout and reasoning deltas dimmed
// on stderr; -v adds a trace with step numbers and durations.
func TestStreamSplitsReasoningFromAnswer(t *testing.T) {
	resetFake(t, streamTurn("pondering the question", "final answer"))
	args := []string{"-provider", "fake", "-agent", "plain", "-stream", "-v", "-prompt", "hi"}
	code, out, stderr := runCLI(t, args, "")
	if code != 0 {
		t.Fatalf("exit code = %d, stderr: %s", code, stderr)
	}
	if out != "final answer\n" {
		t.Errorf("stdout = %q, want the answer deltas only", out)
	}
	if !strings.Contains(stderr, "\x1b[2mpondering the question\x1b[22m") {
		t.Errorf("stderr = %q, want the dimmed reasoning deltas", stderr)
	}
	if strings.Contains(out, "pondering") {
		t.Error("reasoning leaked to stdout")
	}
	if strings.Contains(stderr, "final answer") {
		t.Error("answer leaked to stderr")
	}
	if !strings.Contains(stderr, "trace: step 1\n") {
		t.Errorf("stderr = %q, want a step number in the trace", stderr)
	}
	if !strings.Contains(stderr, "trace: steps=1") || !strings.Contains(stderr, "total=") || !strings.Contains(stderr, "ttft=") {
		t.Errorf("stderr = %q, want step count and timing durations", stderr)
	}
}

// AC5: -prompt-file reads the prompt from disk.
func TestPromptFile(t *testing.T) {
	resetFake(t, textTurn("ok"))
	path := filepath.Join(t.TempDir(), "prompt.md")
	prompt := "Which Go version was released most recently?"
	if err := os.WriteFile(path, []byte(prompt), 0o644); err != nil {
		t.Fatal(err)
	}
	code, out, stderr := runCLI(t, []string{"-provider", "fake", "-prompt-file", path}, "")
	if code != 0 {
		t.Fatalf("exit code = %d, stderr: %s", code, stderr)
	}
	reqs := fakeRequests()
	if len(reqs) != 1 {
		t.Fatalf("provider requests = %d, want 1", len(reqs))
	}
	msgs := reqs[0].prompt
	if len(msgs) == 0 || textOf(msgs[len(msgs)-1]) != prompt {
		got := "<empty>"
		if len(msgs) > 0 {
			got = textOf(msgs[len(msgs)-1])
		}
		t.Errorf("prompt sent = %q, want the file contents %q", got, prompt)
	}
	if !strings.Contains(out, "ok") {
		t.Errorf("stdout = %q, want the answer", out)
	}
}

// -agent bash prints the exit status and submission, and every exit
// status counts as success.
func TestBashExitStatusSummary(t *testing.T) {
	script := textTurn("```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\necho All done\n```")
	t.Run("summary", func(t *testing.T) {
		resetFake(t, script)
		code, out, stderr := runCLI(t, []string{"-provider", "fake", "-agent", "bash", "-cwd", t.TempDir(), "-prompt", "wrap up"}, "")
		if code != 0 {
			t.Fatalf("exit code = %d, stderr: %s", code, stderr)
		}
		if !strings.Contains(out, "exit status: Submitted") {
			t.Errorf("stdout = %q, want the exit status line", out)
		}
		if !strings.Contains(out, "submission: All done") {
			t.Errorf("stdout = %q, want the submission", out)
		}
	})
	t.Run("json", func(t *testing.T) {
		resetFake(t, script)
		code, out, stderr := runCLI(t, []string{"-provider", "fake", "-agent", "bash", "-cwd", t.TempDir(), "-json", "-prompt", "wrap up"}, "")
		if code != 0 {
			t.Fatalf("exit code = %d, stderr: %s", code, stderr)
		}
		ans := decodeJSON(t, out)
		if ans.ExitStatus != "Submitted" || ans.Agent != "bash" {
			t.Errorf("exit status = %q, agent = %q, want Submitted/bash", ans.ExitStatus, ans.Agent)
		}
		if strings.TrimSpace(ans.Submission) != "All done" {
			t.Errorf("submission = %q, want All done", ans.Submission)
		}
		if ans.Status != "completed" {
			t.Errorf("status = %q, want completed", ans.Status)
		}
	})
	t.Run("confirm approves", func(t *testing.T) {
		resetFake(t, script)
		code, out, stderr := runCLI(t, []string{"-provider", "fake", "-agent", "bash", "-cwd", t.TempDir(), "-confirm", "-prompt", "wrap up"}, "y\n")
		if code != 0 {
			t.Fatalf("exit code = %d, stderr: %s", code, stderr)
		}
		if !strings.Contains(stderr, "run command?") {
			t.Errorf("stderr = %q, want the approval prompt", stderr)
		}
		if !strings.Contains(out, "exit status: Submitted") {
			t.Errorf("stdout = %q, want the submit", out)
		}
	})
	t.Run("confirm declines", func(t *testing.T) {
		resetFake(t, script)
		code, _, stderr := runCLI(t, []string{"-provider", "fake", "-agent", "bash", "-cwd", t.TempDir(), "-confirm", "-max-steps", "1", "-prompt", "wrap up"}, "n\n")
		if code != 2 {
			t.Fatalf("exit code = %d, want 2 (declined command keeps the run going into the budget), stderr: %s", code, stderr)
		}
	})
}

// -system replaces the system prompt.
func TestSystemPromptOverride(t *testing.T) {
	resetFake(t, textTurn("ok"))
	code, _, stderr := runCLI(t, []string{"-provider", "fake", "-system", "be terse", "-prompt", "hi"}, "")
	if code != 0 {
		t.Fatalf("exit code = %d, stderr: %s", code, stderr)
	}
	msgs := fakeRequests()[0].prompt
	if len(msgs) == 0 || msgs[0].Role != llmhub.RoleSystem || textOf(msgs[0]) != "be terse" {
		t.Errorf("first message role = %s, text = %q; want a system message saying 'be terse'", msgs[0].Role, textOf(msgs[0]))
	}
}

// The -v trace covers tool durations, step counts, and timing in
// non-streaming runs too.
func TestVerboseTraceNonStream(t *testing.T) {
	resetFake(t, searchScript()...)
	baseURL, _ := newSerperStub(t)
	args := []string{
		"-provider", "fake",
		"-agent", "search",
		"-serper-key", "k",
		"-serper-base-url", baseURL,
		"-v",
		"-prompt", "latest Go release",
	}
	code, out, stderr := runCLI(t, args, "")
	if code != 0 {
		t.Fatalf("exit code = %d, stderr: %s", code, stderr)
	}
	if !strings.Contains(stderr, "trace: tool web_search ok (") {
		t.Errorf("stderr = %q, want the tool line with its duration", stderr)
	}
	if !strings.Contains(stderr, "trace: steps=2") || !strings.Contains(stderr, "ttft=") || !strings.Contains(stderr, "total=") {
		t.Errorf("stderr = %q, want steps and timing", stderr)
	}
	if !strings.Contains(out, "sources:") {
		t.Errorf("stdout = %q, want the numbered source list", out)
	}
}

// The total cost line is printed when the provider reports usage: per
// run in one-shot mode and as a session total in chat mode.
func TestCostLineReported(t *testing.T) {
	usage := llmhub.UsageMetadata{PromptTokens: 120, CompletionTokens: 40, TotalTokens: 160, Cost: 0.0042}
	t.Run("one-shot", func(t *testing.T) {
		resetFake(t, usageTurn("done", usage))
		code, out, stderr := runCLI(t, []string{"-provider", "fake", "-prompt", "x"}, "")
		if code != 0 {
			t.Fatalf("exit code = %d, stderr: %s", code, stderr)
		}
		if !strings.Contains(out, "usage: 120 prompt tokens, 40 completion tokens, cost $0.004200") {
			t.Errorf("stdout = %q, want the usage line", out)
		}
	})
	t.Run("chat total", func(t *testing.T) {
		resetFake(t, usageTurn("one", usage), usageTurn("two", usage))
		code, out, stderr := runCLI(t, []string{"-provider", "fake", "-chat"}, "a\nb\n")
		if code != 0 {
			t.Fatalf("exit code = %d, stderr: %s", code, stderr)
		}
		if !strings.Contains(out, "total usage: 240 prompt tokens, 80 completion tokens, cost $0.008400") {
			t.Errorf("stdout = %q, want the summed usage line", out)
		}
	})
}

// envVarForProvider mirrors llmhub's example CLI conventions.
func TestEnvVarForProvider(t *testing.T) {
	cases := map[string]string{
		"zai":        "ZAI_API_KEY",
		"openai":     "OPENAI_API_KEY",
		"anthropic":  "ANTHROPIC_API_KEY",
		"gemini":     "GEMINI_API_KEY",
		"xai":        "XAI_API_KEY",
		"openrouter": "OPENROUTER_API_KEY",
		"some-new":   "SOME_NEW_API_KEY",
	}
	for provider, want := range cases {
		if got := envVarForProvider(provider); got != want {
			t.Errorf("envVarForProvider(%q) = %q, want %q", provider, got, want)
		}
	}
}

// -h renders the full flag surface and exits 0.
func TestHelpRendersAllFlags(t *testing.T) {
	code, _, stderr := runCLI(t, []string{"-h"}, "")
	if code != 0 {
		t.Fatalf("exit code = %d, want 0", code)
	}
	for _, flag := range []string{
		"-provider", "-model", "-api-key", "-base-url", "-agent", "-chat",
		"-cache-key", "-serper-key", "-prompt", "-prompt-file", "-system",
		"-cwd", "-env", "-skills", "-step-limit", "-cost-limit", "-confirm",
		"-tool-protocol", "-stream", "-v", "-json", "-max-tool-calls", "-max-steps",
	} {
		if !strings.Contains(stderr, flag) {
			t.Errorf("usage is missing %s", flag)
		}
	}
	if !strings.Contains(stderr, "Exit codes") {
		t.Error("usage is missing the exit-code documentation")
	}
}
