package bash

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/smhanov/llmhub"

	"github.com/smhanov/agentloop"
)

// TestE2ESubmittedTrajectory scripts a mini-style trajectory against a
// real temp-dir LocalEnvironment: command, observation, command, submit.
// It asserts the exit status, the captured submission, the message shapes
// (observations are user messages with <returncode>/<output>), and that
// the commands really ran in the environment's working directory.
func TestE2ESubmittedTrajectory(t *testing.T) {
	dir := t.TempDir()
	resetScript(t,
		cmdTurn("THOUGHT: create a file", "printf 'step-one' > step-one.txt"),
		cmdTurn("THOUGHT: verify it", "cat step-one.txt"),
		submitTurn("THOUGHT: done", "Task completed successfully"),
	)
	agent := newFakeAgent(t, WithEnvironment(NewLocalEnvironment(dir, nil, 0)))

	ans, err := agent.Ask(context.Background(), "Create and verify step-one.txt.")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ans.ExitStatus != ExitSubmitted {
		t.Errorf("ExitStatus = %q, want %q", ans.ExitStatus, ExitSubmitted)
	}
	if ans.Submission != "Task completed successfully\n" {
		t.Errorf("Submission = %q, want %q", ans.Submission, "Task completed successfully\n")
	}
	if ans.Steps != 3 {
		t.Errorf("Steps = %d, want 3", ans.Steps)
	}

	// Message shapes: system, task, alternating assistant/user observations,
	// and the submit step's own observation (appended before the run ends).
	wantRoles := []llmhub.Role{
		llmhub.RoleSystem, llmhub.RoleUser,
		llmhub.RoleAssistant, llmhub.RoleUser,
		llmhub.RoleAssistant, llmhub.RoleUser,
		llmhub.RoleAssistant, llmhub.RoleUser,
	}
	if got := rolesOf(ans.Messages); fmt.Sprint(got) != fmt.Sprint(wantRoles) {
		t.Fatalf("roles = %v, want %v", got, wantRoles)
	}
	if got := textOf(ans.Messages[0]); !strings.Contains(got, "exactly ONE bash code block") {
		t.Errorf("system message missing the protocol rules: %q", got[:80]+"...")
	}
	if got := textOf(ans.Messages[1]); !strings.Contains(got, "Create and verify step-one.txt.") {
		t.Error("task message missing the task text")
	}
	// Observations are user messages in the §8.4 shape.
	obs := textOf(ans.Messages[3])
	for _, want := range []string{"<returncode>0</returncode>", "<output>\n\n</output>"} {
		if !strings.Contains(obs, want) {
			t.Errorf("first observation missing %q: %q", want, obs)
		}
	}
	obs = textOf(ans.Messages[5])
	if !strings.Contains(obs, "<returncode>0</returncode>") ||
		!strings.Contains(obs, "<output>\nstep-one\n</output>") {
		t.Errorf("second observation shape wrong: %q", obs)
	}

	// The first command really ran in the environment's cwd.
	data, err := os.ReadFile(filepath.Join(dir, "step-one.txt"))
	if err != nil {
		t.Fatalf("command did not run in the environment cwd: %v", err)
	}
	if string(data) != "step-one" {
		t.Errorf("file content = %q, want %q", data, "step-one")
	}
}

// TestSubmissionCarriesMarkerReport: the report the task template asks
// for on the lines after the submit marker (what you did, what you found,
// open questions) lands verbatim in Answer.Submission.
func TestSubmissionCarriesMarkerReport(t *testing.T) {
	resetScript(t,
		cmdTurn("THOUGHT: look at the directory first", "ls"),
		textTurn("THOUGHT: findings verified, submitting\n"+
			fence+"mswea_bash_command\n"+
			"echo '"+defaultSubmitMarker+"'\n"+
			"echo 'What I did: listed the directory and counted the entries.'\n"+
			"echo 'What I found: three files, all readable.'\n"+
			"echo 'Open questions: none.'\n"+
			fence),
	)
	agent := newFakeAgent(t, WithEnvironment(NewLocalEnvironment(t.TempDir(), nil, 0)))

	ans, err := agent.Ask(context.Background(), "Report the directory contents.")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ans.ExitStatus != ExitSubmitted {
		t.Errorf("ExitStatus = %q, want %q", ans.ExitStatus, ExitSubmitted)
	}
	want := "What I did: listed the directory and counted the entries.\n" +
		"What I found: three files, all readable.\n" +
		"Open questions: none.\n"
	if ans.Submission != want {
		t.Errorf("Submission = %q, want %q", ans.Submission, want)
	}
}

// TestFormatErrorThenRecovery: a response with zero blocks appends the
// format-error guidance (general variant, found-0 count) and the run
// continues to a clean submit.
func TestFormatErrorThenRecovery(t *testing.T) {
	resetScript(t,
		textTurn("I will just say things without any action."),
		submitTurn("THOUGHT: recovered", "done"),
	)
	agent := newFakeAgent(t, WithEnvironment(NewLocalEnvironment(t.TempDir(), nil, 0)))

	ans, err := agent.Ask(context.Background(), "task")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ans.ExitStatus != ExitSubmitted {
		t.Errorf("ExitStatus = %q, want %q", ans.ExitStatus, ExitSubmitted)
	}
	var guidance []string
	for _, m := range ans.Messages {
		if text := textOf(m); strings.Contains(text, "EXACTLY ONE action") {
			guidance = append(guidance, text)
		}
	}
	if len(guidance) != 1 {
		t.Fatalf("got %d format-error messages, want 1", len(guidance))
	}
	for _, want := range []string{"found 0 actions", "TRIPLEBACKTICKSBASH", "echo " + defaultSubmitMarker} {
		if !strings.Contains(guidance[0], want) {
			t.Errorf("format-error guidance missing %q", want)
		}
	}
}

// TestRepeatedFormatErrorsStop: three consecutive malformed responses stop
// the run with RepeatedFormatError (mini's max_consecutive_format_errors).
func TestRepeatedFormatErrorsStop(t *testing.T) {
	resetScript(t, textTurn("garbage"), textTurn("garbage"), textTurn("garbage"))
	agent := newFakeAgent(t, WithEnvironment(NewLocalEnvironment(t.TempDir(), nil, 0)))

	ans, err := agent.Ask(context.Background(), "task")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ans.ExitStatus != ExitRepeatedFormatError {
		t.Errorf("ExitStatus = %q, want %q", ans.ExitStatus, ExitRepeatedFormatError)
	}
	if ans.Steps != 3 {
		t.Errorf("Steps = %d, want 3 (stopped at the third consecutive error)", ans.Steps)
	}
}

// TestFormatErrorCounterResetsOnCleanStep: error, clean step, error, submit
// never reaches two consecutive errors, so the run submits instead of
// stopping (mirrors mini's counter-reset test).
func TestFormatErrorCounterResetsOnCleanStep(t *testing.T) {
	resetScript(t,
		textTurn("garbage"),
		cmdTurn("THOUGHT: clean", "echo hello"),
		textTurn("garbage again"),
		submitTurn("THOUGHT: done", "ok"),
	)
	agent := newFakeAgent(t,
		WithEnvironment(NewLocalEnvironment(t.TempDir(), nil, 0)),
		WithMaxConsecutiveFormatErrors(2),
	)

	ans, err := agent.Ask(context.Background(), "task")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ans.ExitStatus != ExitSubmitted {
		t.Errorf("ExitStatus = %q, want %q", ans.ExitStatus, ExitSubmitted)
	}
	if ans.Submission != "ok\n" {
		t.Errorf("Submission = %q, want %q", ans.Submission, "ok\n")
	}
}

// TestStepLimitGracefulStop: one scripted command and a step limit of one
// stops before the second inference with LimitsExceeded.
func TestStepLimitGracefulStop(t *testing.T) {
	resetScript(t,
		cmdTurn("THOUGHT: one", "echo one"),
		cmdTurn("THOUGHT: two", "echo two"),
	)
	agent := newFakeAgent(t,
		WithEnvironment(NewLocalEnvironment(t.TempDir(), nil, 0)),
		WithStepLimit(1),
	)

	ans, err := agent.Ask(context.Background(), "task")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ans.ExitStatus != ExitLimitsExceeded {
		t.Errorf("ExitStatus = %q, want %q", ans.ExitStatus, ExitLimitsExceeded)
	}
	if ans.Steps != 1 {
		t.Errorf("Steps = %d, want 1", ans.Steps)
	}
}

// TestTokenLimitGracefulStop: cumulative counted tokens reaching the limit
// stop before the next inference with LimitsExceeded — nil error, core
// StatusStopped, and Steps counts only the completed inferences.
func TestTokenLimitGracefulStop(t *testing.T) {
	billed := cmdTurn("THOUGHT: one", "echo one")
	billed.usage = llmhub.UsageMetadata{TotalTokens: 120000}
	resetScript(t, billed, cmdTurn("THOUGHT: two", "echo two"))
	agent := newFakeAgent(t,
		WithEnvironment(NewLocalEnvironment(t.TempDir(), nil, 0)),
		WithTokenLimit(100000),
	)

	ans, err := agent.Ask(context.Background(), "task")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ans.ExitStatus != ExitLimitsExceeded {
		t.Errorf("ExitStatus = %q, want %q", ans.ExitStatus, ExitLimitsExceeded)
	}
	if ans.Status != agentloop.StatusStopped {
		t.Errorf("Status = %v, want StatusStopped", ans.Status)
	}
	if ans.Steps != 1 {
		t.Errorf("Steps = %d, want 1", ans.Steps)
	}
}

// TestTokenLimitFallbackCounting: a provider that reports no TotalTokens
// but populated Prompt/Completion counts is still budgeted — the sum is
// billed either way.
func TestTokenLimitFallbackCounting(t *testing.T) {
	billed := cmdTurn("THOUGHT: one", "echo one")
	billed.usage = llmhub.UsageMetadata{PromptTokens: 60000, CompletionTokens: 60000}
	resetScript(t, billed, cmdTurn("THOUGHT: two", "echo two"))
	agent := newFakeAgent(t,
		WithEnvironment(NewLocalEnvironment(t.TempDir(), nil, 0)),
		WithTokenLimit(100000),
	)

	ans, err := agent.Ask(context.Background(), "task")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ans.ExitStatus != ExitLimitsExceeded {
		t.Errorf("ExitStatus = %q, want %q", ans.ExitStatus, ExitLimitsExceeded)
	}
	if ans.Steps != 1 {
		t.Errorf("Steps = %d, want 1", ans.Steps)
	}
}

// TestTokenLimitCountsFormatErrorTurns: a malformed response is billed
// like any other inference, so it counts toward the token budget too.
func TestTokenLimitCountsFormatErrorTurns(t *testing.T) {
	garbage := textTurn("no action block in this text")
	garbage.usage = llmhub.UsageMetadata{TotalTokens: 120000}
	resetScript(t, garbage, cmdTurn("THOUGHT: two", "echo two"))
	agent := newFakeAgent(t,
		WithEnvironment(NewLocalEnvironment(t.TempDir(), nil, 0)),
		WithTokenLimit(100000),
	)

	ans, err := agent.Ask(context.Background(), "task")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ans.ExitStatus != ExitLimitsExceeded {
		t.Errorf("ExitStatus = %q, want %q", ans.ExitStatus, ExitLimitsExceeded)
	}
	if ans.Steps != 1 {
		t.Errorf("Steps = %d, want 1", ans.Steps)
	}
}

// TestTokenLimitUnlimitedDefault: the default 0 (and a negative value,
// which resolution clamps to 0) never trips, no matter how large the
// reported usage.
func TestTokenLimitUnlimitedDefault(t *testing.T) {
	for _, limit := range []int64{0, -5} {
		billed := cmdTurn("THOUGHT: one", "echo one")
		billed.usage = llmhub.UsageMetadata{TotalTokens: 1 << 30}
		resetScript(t, billed, submitTurn("THOUGHT: done", "done"))
		agent := newFakeAgent(t,
			WithEnvironment(NewLocalEnvironment(t.TempDir(), nil, 0)),
			WithTokenLimit(limit),
		)

		ans, err := agent.Ask(context.Background(), "task")
		if err != nil {
			t.Fatalf("Ask(limit %d): %v", limit, err)
		}
		if ans.ExitStatus != ExitSubmitted {
			t.Errorf("limit %d: ExitStatus = %q, want %q (0/negative means unlimited)",
				limit, ans.ExitStatus, ExitSubmitted)
		}
	}
}

// TestTokenLimitResetsPerAsk: token accounting is per-run like the other
// limits. The first Ask ends submitted with a cumulative total past the
// limit; a second Ask starts from zero — had the counter leaked, its very
// first limit check would stop it with LimitsExceeded and 0 steps.
func TestTokenLimitResetsPerAsk(t *testing.T) {
	billed := func(turn fakeTurn) fakeTurn {
		turn.usage = llmhub.UsageMetadata{TotalTokens: 100000}
		return turn
	}
	resetScript(t,
		billed(cmdTurn("THOUGHT: work", "echo one")),
		billed(submitTurn("THOUGHT: done", "done")),
		billed(cmdTurn("THOUGHT: work again", "echo two")),
		billed(submitTurn("THOUGHT: done again", "done")),
	)
	agent := newFakeAgent(t,
		WithEnvironment(NewLocalEnvironment(t.TempDir(), nil, 0)),
		WithTokenLimit(150000),
	)

	for i := 1; i <= 2; i++ {
		ans, err := agent.Ask(context.Background(), "task")
		if err != nil {
			t.Fatalf("Ask %d: %v", i, err)
		}
		if ans.ExitStatus != ExitSubmitted {
			t.Errorf("Ask %d: ExitStatus = %q, want %q (the counter must reset per Ask)",
				i, ans.ExitStatus, ExitSubmitted)
		}
		if ans.Steps != 2 {
			t.Errorf("Ask %d: Steps = %d, want 2", i, ans.Steps)
		}
	}
}

// TestTokenLimitCountsStreamedRuns: the streaming path counts identically —
// usage arrives via the terminal chunk and the limit trips before the next
// inference. One provider request only: a counting miss would request the
// second turn, exhaust the script, and surface an EventError.
func TestTokenLimitCountsStreamedRuns(t *testing.T) {
	billed := cmdTurn("THOUGHT: one", "echo one")
	billed.usage = llmhub.UsageMetadata{TotalTokens: 120000}
	resetScript(t, billed, cmdTurn("THOUGHT: two", "echo two"))
	agent := newFakeAgent(t,
		WithEnvironment(NewLocalEnvironment(t.TempDir(), nil, 0)),
		WithTokenLimit(100000),
	)

	events, err := agent.AskStream(context.Background(), "task")
	if err != nil {
		t.Fatalf("AskStream: %v", err)
	}
	var done *agentloop.Answer
	for ev := range events {
		switch ev.Kind {
		case agentloop.EventDone:
			done = ev.Answer
		case agentloop.EventError:
			t.Errorf("stream error: %v", ev.Err)
		}
	}
	if done == nil {
		t.Fatal("no Done event")
	}
	if done.Status != agentloop.StatusStopped || done.Steps != 1 {
		t.Errorf("Status = %v, Steps = %d; want StatusStopped and 1", done.Status, done.Steps)
	}
	if n := len(fakeRequests()); n != 1 {
		t.Errorf("provider requests = %d, want 1 (stopped before the second inference)", n)
	}
}

// TestWallTimeLimitGracefulStop: a nanosecond wall-time limit always trips
// by the second limit check, yielding TimeExceeded (not LimitsExceeded).
func TestWallTimeLimitGracefulStop(t *testing.T) {
	resetScript(t, cmdTurn("THOUGHT: one", "echo one"), cmdTurn("THOUGHT: two", "echo two"))
	agent := newFakeAgent(t,
		WithEnvironment(NewLocalEnvironment(t.TempDir(), nil, 0)),
		WithWallTimeLimit(time.Nanosecond),
	)

	ans, err := agent.Ask(context.Background(), "task")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ans.ExitStatus != ExitTimeExceeded {
		t.Errorf("ExitStatus = %q, want %q", ans.ExitStatus, ExitTimeExceeded)
	}
}

// TestApproverVetoContinues: a vetoed command never executes, the model
// sees the not-approved observation, and the run continues to submission.
func TestApproverVetoContinues(t *testing.T) {
	dir := t.TempDir()
	resetScript(t,
		cmdTurn("THOUGHT: try", "touch forbidden.txt"),
		submitTurn("THOUGHT: done", "finished"),
	)
	agent := newFakeAgent(t,
		WithEnvironment(NewLocalEnvironment(dir, nil, 0)),
		WithApprover(func(ctx context.Context, command string) error {
			if strings.Contains(command, "forbidden") {
				return errors.New("not allowed")
			}
			return nil
		}),
	)

	ans, err := agent.Ask(context.Background(), "task")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ans.ExitStatus != ExitSubmitted {
		t.Errorf("ExitStatus = %q, want %q (the run must continue past a veto)",
			ans.ExitStatus, ExitSubmitted)
	}
	if _, statErr := os.Stat(filepath.Join(dir, "forbidden.txt")); !os.IsNotExist(statErr) {
		t.Error("vetoed command was executed anyway")
	}
	var veto string
	for _, m := range ans.Messages {
		if text := textOf(m); strings.Contains(text, "command not approved") {
			veto = text
		}
	}
	if veto != `{"error":"command not approved: not allowed"}` {
		t.Errorf("veto observation = %q, want the JSON error shape", veto)
	}
}

// TestCustomSubmitMarker: a custom marker ends the run and replaces the
// default in the rendered task prompt.
func TestCustomSubmitMarker(t *testing.T) {
	resetScript(t, cmdTurn("THOUGHT: done", "echo 'ALL_DONE'\necho 'finished'"))
	agent := newFakeAgent(t,
		WithEnvironment(NewLocalEnvironment(t.TempDir(), nil, 0)),
		WithSubmitMarker("ALL_DONE"),
	)

	ans, err := agent.Ask(context.Background(), "task")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ans.ExitStatus != ExitSubmitted || ans.Submission != "finished\n" {
		t.Errorf("ExitStatus = %q, Submission = %q; want Submitted and %q",
			ans.ExitStatus, ans.Submission, "finished\n")
	}
	taskMsg := textOf(fakeRequests()[0].prompt[1])
	if !strings.Contains(taskMsg, "echo ALL_DONE") {
		t.Errorf("task prompt does not mention the custom marker: %q", taskMsg)
	}
}

// TestTimeoutObservationFeedsBack: a command that outlives the environment
// timeout produces an exception observation and the agent recovers on the
// next step (mirrors mini's timeout tests).
func TestTimeoutObservationFeedsBack(t *testing.T) {
	resetScript(t,
		cmdTurn("THOUGHT: slow", "echo 999; sleep 30"),
		submitTurn("THOUGHT: recovered", "recovered"),
	)
	agent := newFakeAgent(t,
		WithEnvironment(NewLocalEnvironment(t.TempDir(), nil, 300*time.Millisecond)),
	)

	start := time.Now()
	ans, err := agent.Ask(context.Background(), "task")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Fatalf("run took %s; the timeout kill did not bound the command", elapsed)
	}
	if ans.ExitStatus != ExitSubmitted || ans.Submission != "recovered\n" {
		t.Errorf("ExitStatus = %q, Submission = %q; want Submitted and %q",
			ans.ExitStatus, ans.Submission, "recovered\n")
	}
	var timedOut []string
	for _, m := range ans.Messages {
		if text := textOf(m); strings.Contains(text, "timed out") {
			timedOut = append(timedOut, text)
		}
	}
	if len(timedOut) != 1 {
		t.Fatalf("got %d timeout observations, want 1", len(timedOut))
	}
	if !strings.Contains(timedOut[0], "<returncode>-1</returncode>") ||
		!strings.Contains(timedOut[0], "999") {
		t.Errorf("timeout observation lost the partial output: %q", timedOut[0])
	}
}

// TestObservationsNeverExceedByteCap (P0): with a small output limit, a
// command producing far more output than the limit yields bounded
// observations — never the raw blob.
func TestObservationsNeverExceedByteCap(t *testing.T) {
	resetScript(t,
		cmdTurn("THOUGHT: big", "head -c 50000 /dev/zero | tr '\\0' 'x'"),
		submitTurn("THOUGHT: done", "done"),
	)
	agent := newFakeAgent(t,
		WithEnvironment(NewLocalEnvironment(t.TempDir(), nil, 0)),
		WithOutputLimit(500),
	)

	ans, err := agent.Ask(context.Background(), "task")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ans.ExitStatus != ExitSubmitted {
		t.Errorf("ExitStatus = %q, want %q", ans.ExitStatus, ExitSubmitted)
	}
	var bigObs string
	for i, m := range ans.Messages {
		text := textOf(m)
		if !strings.Contains(text, "<returncode>") {
			continue // not an observation
		}
		if len(text) > 2500 {
			t.Errorf("observation %d is %d bytes, over the cap", i, len(text))
		}
		if strings.Contains(text, "xxxx") {
			bigObs = text
		}
	}
	if bigObs == "" {
		t.Fatal("the large-output observation is missing")
	}
	if !strings.Contains(bigObs, "characters elided") ||
		!strings.Contains(bigObs, "<elided_chars>49500 characters elided</elided_chars>") {
		t.Errorf("large observation took the elision path incompletely: %.160q", bigObs)
	}
}

// TestMaliciousToolCallScriptBounded (P0): a provider that ignores the
// text protocol and keeps emitting native tool calls must terminate.
// With the default format-error ceiling the preset stops it at three
// steps, well before the core's hard ceiling.
func TestMaliciousToolCallScriptBounded(t *testing.T) {
	script := make([]fakeTurn, 0, 8)
	for i := 0; i < 8; i++ {
		script = append(script, callTurn(llmhub.ToolCall(fmt.Sprintf("call-%d", i), "bash", "{}")))
	}
	resetScript(t, script...)
	agent := newFakeAgent(t, WithEnvironment(NewLocalEnvironment(t.TempDir(), nil, 0)))

	ans, err := agent.Ask(context.Background(), "task")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if ans.ExitStatus != ExitRepeatedFormatError {
		t.Errorf("ExitStatus = %q, want %q", ans.ExitStatus, ExitRepeatedFormatError)
	}
	if ans.Steps != 3 {
		t.Errorf("Steps = %d, want 3", ans.Steps)
	}
}

// TestMaliciousToolCallScriptCoreNet (P0): with the format-error ceiling
// and every other preset/core guard disabled, the core MaxSteps safety net
// still ends the run with ErrBudgetExhausted — no provider script can keep
// the loop going.
func TestMaliciousToolCallScriptCoreNet(t *testing.T) {
	script := make([]fakeTurn, 0, 30)
	for i := 0; i < 30; i++ {
		script = append(script, callTurn(llmhub.ToolCall(fmt.Sprintf("call-%d", i), "bash", "{}")))
	}
	resetScript(t, script...)
	agent := newFakeAgent(t,
		WithEnvironment(NewLocalEnvironment(t.TempDir(), nil, 0)),
		WithMaxConsecutiveFormatErrors(0), // unlimited format errors
		agentloop.WithDupCallLimit(0),     // no loop detection
		agentloop.WithMaxToolCalls(100),   // no forced-final rescue
		agentloop.WithMaxSteps(6),         // the last guard standing
	)

	_, err := agent.Ask(context.Background(), "task")
	if !errors.Is(err, agentloop.ErrBudgetExhausted) {
		t.Fatalf("err = %v, want ErrBudgetExhausted", err)
	}
}

// TestStreamedTrajectory: the same wiring works through AskStream; the
// event channel closes after Done without an Error event.
func TestStreamedTrajectory(t *testing.T) {
	resetScript(t,
		cmdTurn("THOUGHT: work", "echo hello"),
		submitTurn("THOUGHT: done", "done"),
	)
	agent := newFakeAgent(t, WithEnvironment(NewLocalEnvironment(t.TempDir(), nil, 0)))

	events, err := agent.AskStream(context.Background(), "task")
	if err != nil {
		t.Fatalf("AskStream: %v", err)
	}
	sawDone, sawError := false, false
	for ev := range events {
		switch ev.Kind {
		case agentloop.EventDone:
			sawDone = true
		case agentloop.EventError:
			sawError = true
			t.Errorf("stream error: %v", ev.Err)
		}
	}
	if !sawDone || sawError {
		t.Errorf("sawDone = %v, sawError = %v; want a clean Done", sawDone, sawError)
	}
}

// TestPerCallOptionOverride: a per-Ask environment override wins over the
// construction default and receives the commands.
func TestPerCallOptionOverride(t *testing.T) {
	dir := t.TempDir()
	resetScript(t,
		cmdTurn("THOUGHT: where", "pwd"),
		submitTurn("THOUGHT: done", "done"),
	)
	agent := newFakeAgent(t) // default environment

	ans, err := agent.Ask(context.Background(), "task", WithEnvironment(NewLocalEnvironment(dir, nil, 0)))
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if !strings.Contains(textOf(ans.Messages[3]), dir) {
		t.Errorf("per-call environment ignored; observation = %q", textOf(ans.Messages[3]))
	}
}

// TestWithHooksCompose: caller hooks observe events after the protocol
// hooks ran; a caller AfterGenerate sees the appended observation.
func TestWithHooksCompose(t *testing.T) {
	resetScript(t, submitTurn("THOUGHT: done", "done"))
	var observedSteps int
	var sawUserAppend bool
	agent := newFakeAgent(t,
		WithEnvironment(NewLocalEnvironment(t.TempDir(), nil, 0)),
		WithHooks(agentloop.Hooks{
			AfterGenerate: func(ctx context.Context, ev *agentloop.Event) error {
				observedSteps = ev.Step
				for _, m := range ev.Messages {
					if m.Role == llmhub.RoleUser && strings.Contains(textOf(m), "<returncode>") {
						sawUserAppend = true
					}
				}
				return nil
			},
		}),
	)
	if _, err := agent.Ask(context.Background(), "task"); err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if observedSteps != 1 || !sawUserAppend {
		t.Errorf("composed hook saw step %d with user append %v; want step 1 and true",
			observedSteps, sawUserAppend)
	}
}

// TestNewNilClient: construction with no LLM client fails with the core
// sentinel, not a panic.
func TestNewNilClient(t *testing.T) {
	if _, err := New(nil); !errors.Is(err, agentloop.ErrNoLLMClient) {
		t.Fatalf("err = %v, want ErrNoLLMClient", err)
	}
}
