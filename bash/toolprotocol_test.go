package bash

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/smhanov/llmhub"

	"github.com/smhanov/agentloop"
)

// toolCallTurn scripts a native bash tool call for the tool protocol: the
// command JSON-encoded as {"command": …} arguments of a call named bash.
func toolCallTurn(id, command string) fakeTurn {
	args, err := json.Marshal(bashToolArgs{Command: command})
	if err != nil {
		panic(err) // static test data; never fails
	}
	return callTurn(llmhub.ToolCall(id, bashToolName, string(args)))
}

// toolSubmitTurn scripts the finishing tool call: the marker on the first
// output line and the submission on the second.
func toolSubmitTurn(submission string) fakeTurn {
	return toolCallTurn("submit", "echo '"+defaultSubmitMarker+"'\necho '"+submission+"'")
}

// observationTexts extracts the observation payloads in order from either
// protocol's transcript: text-protocol observations ride user messages,
// tool-protocol ones ride tool-result messages — both carry the §8.4
// <returncode> block no other message contains.
func observationTexts(msgs []*llmhub.Message) []string {
	var out []string
	for _, m := range msgs {
		if text := textOf(m); strings.Contains(text, "<returncode>") {
			out = append(out, text)
		}
	}
	return out
}

// TestToolProtocolTrajectoryMatchesText (§12.7 variant): the same scripted
// trajectory as the text-protocol E2E — command, observation, command,
// submit marker — expressed as native tool calls. The runs must produce
// byte-equal observations and the same ExitStatus/Submission.
func TestToolProtocolTrajectoryMatchesText(t *testing.T) {
	createCmd, catCmd := "printf 'step-one' > step-one.txt", "cat step-one.txt"
	submission := "Task completed successfully"

	// Text protocol baseline: the bash E2E trajectory (command, observation,
	// command, submit marker).
	resetScript(t,
		cmdTurn("THOUGHT: create a file", createCmd),
		cmdTurn("THOUGHT: verify it", catCmd),
		submitTurn("THOUGHT: done", submission),
	)
	textAgent := newFakeAgent(t, WithEnvironment(NewLocalEnvironment(t.TempDir(), nil, 0)))
	textAns, err := textAgent.Ask(context.Background(), "Create and verify step-one.txt.")
	if err != nil {
		t.Fatalf("text Ask: %v", err)
	}

	// Tool protocol: same trajectory, native bash tool calls.
	dir := t.TempDir()
	resetScript(t,
		toolCallTurn("c1", createCmd),
		toolCallTurn("c2", catCmd),
		toolSubmitTurn(submission),
	)
	toolAgent := newFakeAgent(t,
		WithEnvironment(NewLocalEnvironment(dir, nil, 0)),
		WithToolProtocol(),
	)
	toolAns, err := toolAgent.Ask(context.Background(), "Create and verify step-one.txt.")
	if err != nil {
		t.Fatalf("tool Ask: %v", err)
	}

	if toolAns.ExitStatus != textAns.ExitStatus || toolAns.ExitStatus != ExitSubmitted {
		t.Errorf("ExitStatus: tool %q vs text %q; want %q",
			toolAns.ExitStatus, textAns.ExitStatus, ExitSubmitted)
	}
	if toolAns.Submission != textAns.Submission {
		t.Errorf("Submission: tool %q vs text %q", toolAns.Submission, textAns.Submission)
	}
	if toolAns.Submission != submission+"\n" {
		t.Errorf("Submission = %q, want %q", toolAns.Submission, submission+"\n")
	}

	// Byte-equal observations across the transports.
	textObs, toolObs := observationTexts(textAns.Messages), observationTexts(toolAns.Messages)
	if len(textObs) != 3 || len(toolObs) != 3 {
		t.Fatalf("observations: text %d, tool %d; want 3 each", len(textObs), len(toolObs))
	}
	for i := range textObs {
		if toolObs[i] != textObs[i] {
			t.Errorf("observation %d differs across protocols:\ntext: %q\ntool: %q",
				i, textObs[i], toolObs[i])
		}
	}

	// Transport shapes: every bash step is an assistant tool-call message
	// answered by a tool result (never a user-message observation).
	for i, m := range toolAns.Messages {
		switch m.Role {
		case llmhub.RoleTool:
			if toolAns.Messages[i-1].Role != llmhub.RoleAssistant {
				t.Errorf("tool result at %d not preceded by an assistant message", i)
			}
			if got := textOf(m); !strings.Contains(got, "<returncode>") {
				t.Errorf("tool result at %d is not a formatted observation: %q", i, got)
			}
		case llmhub.RoleUser:
			if i != 1 { // the task message
				t.Errorf("unexpected user message at %d: %q", i, textOf(m))
			}
		}
	}
	if len(toolAns.Tools) != 3 {
		t.Fatalf("trace has %d entries, want 3", len(toolAns.Tools))
	}
	for i, tr := range toolAns.Tools {
		if tr.Name != bashToolName || tr.Err != nil {
			t.Errorf("trace[%d] = %s (err %v); want a clean bash execution", i, tr.Name, tr.Err)
		}
	}

	// The commands really ran in the tool run's environment.
	data, err := os.ReadFile(filepath.Join(dir, "step-one.txt"))
	if err != nil {
		t.Fatalf("commands did not run in the environment cwd: %v", err)
	}
	if string(data) != "step-one" {
		t.Errorf("file content = %q, want %q", data, "step-one")
	}
}

// TestProtocolLimitsAndApprover (§12.7): the step/token/wall limits and the
// approval gate fire identically in both protocols — same exit statuses,
// same step counts, same veto observation, and a vetoed command is never
// executed. The first turn is billed in every script so the token-limit
// case can trip.
func TestProtocolLimitsAndApprover(t *testing.T) {
	billed := func(turn fakeTurn) fakeTurn {
		turn.usage = llmhub.UsageMetadata{TotalTokens: 120000}
		return turn
	}

	cases := []struct {
		name      string
		opts      []Option
		wantExit  string
		wantSteps int
	}{
		{"step-limit", []Option{WithStepLimit(1)}, ExitLimitsExceeded, 1},
		{"token-limit", []Option{WithTokenLimit(100000)}, ExitLimitsExceeded, 1},
		// The 1ns wall-time limit trips before the first inference, so no
		// step completes in either protocol.
		{"wall-time", []Option{WithWallTimeLimit(time.Nanosecond)}, ExitTimeExceeded, 0},
		{"approver", []Option{WithApprover(func(ctx context.Context, command string) error {
			if strings.Contains(command, "vetoed") {
				return fmt.Errorf("not allowed")
			}
			return nil
		})}, ExitSubmitted, 3},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			scripts := map[string][]fakeTurn{
				"text": {
					billed(cmdTurn("THOUGHT: try", "touch vetoed.txt")),
					cmdTurn("THOUGHT: go on", "echo two"),
					submitTurn("THOUGHT: done", "done"),
				},
				"tool": {
					billed(toolCallTurn("c1", "touch vetoed.txt")),
					toolCallTurn("c2", "echo two"),
					toolSubmitTurn("done"),
				},
			}

			type result struct {
				exit    string
				steps   int
				veto    string // the not-approved observation, when present
				fileRan bool   // the vetoed command executed anyway
			}
			got := map[string]result{}
			for _, proto := range []struct {
				name string
				tool bool
			}{{"text", false}, {"tool", true}} {
				dir := t.TempDir()
				resetScript(t, scripts[proto.name]...)
				opts := append([]Option{WithEnvironment(NewLocalEnvironment(dir, nil, 0))}, tc.opts...)
				if proto.tool {
					opts = append(opts, WithToolProtocol())
				}
				ans, err := newFakeAgent(t, opts...).Ask(context.Background(), "task")
				if err != nil {
					t.Fatalf("%s Ask: %v", proto.name, err)
				}
				r := result{exit: ans.ExitStatus, steps: ans.Steps}
				for _, m := range ans.Messages {
					if text := textOf(m); strings.Contains(text, "command not approved") {
						r.veto = text
					}
				}
				_, statErr := os.Stat(filepath.Join(dir, "vetoed.txt"))
				r.fileRan = statErr == nil
				got[proto.name] = r
			}

			if got["text"] != got["tool"] {
				t.Fatalf("protocols diverge:\ntext: %+v\ntool: %+v", got["text"], got["tool"])
			}
			r := got["tool"]
			if r.exit != tc.wantExit {
				t.Errorf("ExitStatus = %q, want %q", r.exit, tc.wantExit)
			}
			if r.steps != tc.wantSteps {
				t.Errorf("Steps = %d, want %d", r.steps, tc.wantSteps)
			}
			if tc.name == "approver" {
				if r.veto != `{"error":"command not approved: not allowed"}` {
					t.Errorf("veto observation = %q, want the JSON error shape", r.veto)
				}
				if r.fileRan {
					t.Error("the vetoed command was executed anyway")
				}
			}
		})
	}
}

// TestToolProtocolSubmitViaToolCall (§12.7): the submit marker on a tool
// call's output stops the run with Submitted before the next inference —
// the fake script would otherwise be exhausted and Ask would error.
func TestToolProtocolSubmitViaToolCall(t *testing.T) {
	resetScript(t,
		toolCallTurn("c1", "echo working"),
		toolSubmitTurn("all done"),
	)
	agent := newFakeAgent(t,
		WithEnvironment(NewLocalEnvironment(t.TempDir(), nil, 0)),
		WithToolProtocol(),
	)

	ans, err := agent.Ask(context.Background(), "task")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ans.ExitStatus != ExitSubmitted {
		t.Errorf("ExitStatus = %q, want %q", ans.ExitStatus, ExitSubmitted)
	}
	if ans.Submission != "all done\n" {
		t.Errorf("Submission = %q, want %q", ans.Submission, "all done\n")
	}
	if ans.Steps != 2 {
		t.Errorf("Steps = %d, want 2 (no inference after the submit step)", ans.Steps)
	}
	if ans.Status != agentloop.StatusStopped {
		t.Errorf("Status = %v, want StatusStopped (graceful stop before the next inference)",
			ans.Status)
	}
}

// TestToolProtocolTextFinishCompleted: a plain-text response (no tool call)
// ends a tool-protocol run through the loop's natural termination —
// StatusCompleted with no preset-level exit status.
func TestToolProtocolTextFinishCompleted(t *testing.T) {
	resetScript(t,
		toolCallTurn("c1", "echo hello"),
		textTurn("The task is done; here is the summary."),
	)
	agent := newFakeAgent(t,
		WithEnvironment(NewLocalEnvironment(t.TempDir(), nil, 0)),
		WithToolProtocol(),
	)

	ans, err := agent.Ask(context.Background(), "task")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ans.Status != agentloop.StatusCompleted {
		t.Errorf("Status = %v, want StatusCompleted", ans.Status)
	}
	if ans.ExitStatus != "" {
		t.Errorf("ExitStatus = %q, want empty (no preset-level outcome)", ans.ExitStatus)
	}
	if ans.Steps != 2 || ans.ToolCalls != 1 {
		t.Errorf("Steps = %d, ToolCalls = %d; want 2 and 1", ans.Steps, ans.ToolCalls)
	}
	if ans.Text != "The task is done; here is the summary." {
		t.Errorf("Text = %q, want the final text response", ans.Text)
	}
}

// TestToolProtocolForcedFinalNotice: a tool-protocol run that spends the
// core's default five-command budget reaches the forced-final step. The
// core appends its budget-exhaustion instruction once as a user message
// after the last tool result (inert for the protocol: observations ride
// tool results, and the parser only ever reads assistant text), and the
// forced step's text ends the run through the natural path — StatusCompleted
// with no preset exit status.
func TestToolProtocolForcedFinalNotice(t *testing.T) {
	resetScript(t,
		toolCallTurn("c1", "echo one"),
		toolCallTurn("c2", "echo two"),
		toolCallTurn("c3", "echo three"),
		toolCallTurn("c4", "echo four"),
		toolCallTurn("c5", "echo five"),
		textTurn("All five commands ran; final summary."),
	)
	agent := newFakeAgent(t,
		WithEnvironment(NewLocalEnvironment(t.TempDir(), nil, 0)),
		WithToolProtocol(),
	)

	ans, err := agent.Ask(context.Background(), "task")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ans.Status != agentloop.StatusCompleted || ans.ExitStatus != "" {
		t.Fatalf("Status = %v ExitStatus = %q; want natural Completed termination", ans.Status, ans.ExitStatus)
	}
	if ans.Steps != 6 || ans.ToolCalls != 5 {
		t.Fatalf("Steps = %d ToolCalls = %d; want 6/5", ans.Steps, ans.ToolCalls)
	}

	// Exactly one user-role notice, after the last tool result and before
	// the forced final response.
	notices := 0
	for i, m := range ans.Messages {
		if m.Role == llmhub.RoleUser && strings.Contains(textOf(m), "tool budget for this run is exhausted") {
			notices++
			if ans.Messages[i-1].Role != llmhub.RoleTool {
				t.Errorf("notice at %d not preceded by the last tool result", i)
			}
			if i != len(ans.Messages)-2 {
				t.Errorf("notice at %d does not sit right before the final response", i)
			}
		}
	}
	if notices != 1 {
		t.Errorf("transcript carries %d budget-exhaustion notices, want 1", notices)
	}
}

// TestToolProtocolSwitchesTransport: with WithToolProtocol a text-protocol
// response is not parsed at all — no format error, no execution; the run
// completes the moment the model stops calling tools. (The default remains
// the text protocol, proven by the malicious tool-call tests.)
func TestToolProtocolSwitchesTransport(t *testing.T) {
	resetScript(t, submitTurn("THOUGHT: done", "done"))
	agent := newFakeAgent(t,
		WithEnvironment(NewLocalEnvironment(t.TempDir(), nil, 0)),
		WithToolProtocol(),
	)

	ans, err := agent.Ask(context.Background(), "task")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ans.ExitStatus != "" || ans.Status != agentloop.StatusCompleted {
		t.Errorf("ExitStatus = %q, Status = %v; want empty/Completed — the fenced block must not parse",
			ans.ExitStatus, ans.Status)
	}
	if ans.ToolCalls != 0 || ans.Steps != 1 {
		t.Errorf("ToolCalls = %d, Steps = %d; want 0 and 1", ans.ToolCalls, ans.Steps)
	}
}

// TestToolProtocolBadArgsRecover: malformed tool arguments yield the core's
// in-band {"error": …} observation (the tool protocol's analog of the
// format-error guidance) and the run continues to submission.
func TestToolProtocolBadArgsRecover(t *testing.T) {
	resetScript(t,
		callTurn(llmhub.ToolCall("bad", bashToolName, "{not json")),
		toolSubmitTurn("recovered"),
	)
	agent := newFakeAgent(t,
		WithEnvironment(NewLocalEnvironment(t.TempDir(), nil, 0)),
		WithToolProtocol(),
	)

	ans, err := agent.Ask(context.Background(), "task")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ans.ExitStatus != ExitSubmitted || ans.Submission != "recovered\n" {
		t.Errorf("ExitStatus = %q, Submission = %q; want Submitted and %q",
			ans.ExitStatus, ans.Submission, "recovered\n")
	}
	var errs []string
	for _, m := range ans.Messages {
		if m.Role == llmhub.RoleTool {
			if text := textOf(m); strings.Contains(text, `"error"`) {
				errs = append(errs, text)
			}
		}
	}
	if len(errs) != 1 || !strings.Contains(errs[0], "decode tool arguments") {
		t.Errorf("bad-args observation = %v, want one in-band decode error", errs)
	}
}
