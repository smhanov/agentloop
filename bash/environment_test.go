package bash

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

// TestLocalEnvironmentOutputAndCodes covers stdout/stderr merging and exit
// codes: combined output arrives in order, non-zero codes are results and
// never errors.
func TestLocalEnvironmentOutputAndCodes(t *testing.T) {
	env := NewLocalEnvironment(t.TempDir(), nil, 0)

	res, err := env.Execute(context.Background(), "echo out; echo err 1>&2")
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if res.Output != "out\nerr\n" {
		t.Errorf("combined output = %q, want %q", res.Output, "out\nerr\n")
	}
	if res.ReturnCode != 0 || res.ExceptionInfo != "" {
		t.Errorf("rc = %d, exception = %q, want 0 and empty", res.ReturnCode, res.ExceptionInfo)
	}

	res, err = env.Execute(context.Background(), "exit 3")
	if err != nil {
		t.Fatalf("Execute exit 3: %v", err)
	}
	if res.ReturnCode != 3 {
		t.Errorf("rc = %d, want 3 (non-zero codes are results, not errors)", res.ReturnCode)
	}
}

// TestLocalEnvironmentEnvAndCwd covers extra env merged over the inherited
// environment (later entries win) and the fixed working directory.
func TestLocalEnvironmentEnvAndCwd(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("AGENTLOOP_BASH_TEST", "outer")
	env := NewLocalEnvironment(dir, []string{
		"AGENTLOOP_BASH_TEST=inner",
		"AGENTLOOP_BASH_EXTRA=extra",
	}, 0)

	res, err := env.Execute(context.Background(), "echo $AGENTLOOP_BASH_TEST $AGENTLOOP_BASH_EXTRA")
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if strings.TrimSpace(res.Output) != "inner extra" {
		t.Errorf("env merge = %q, want %q", res.Output, "inner extra")
	}

	res, err = env.Execute(context.Background(), "pwd")
	if err != nil {
		t.Fatalf("Execute pwd: %v", err)
	}
	want, evalErr := filepath.EvalSymlinks(dir)
	if evalErr != nil {
		t.Fatalf("EvalSymlinks: %v", evalErr)
	}
	got, evalErr := filepath.EvalSymlinks(strings.TrimSpace(res.Output))
	if evalErr != nil {
		t.Fatalf("EvalSymlinks(output): %v", evalErr)
	}
	if got != want {
		t.Errorf("pwd = %q, want %q", got, want)
	}
}

// TestLocalEnvironmentTimeoutKillsProcessGroup pins the mini _run
// semantics: a timeout kills the whole process group (no orphan `sleep`),
// preserves the partial output, and reports the exception without an
// error.
func TestLocalEnvironmentTimeoutKillsProcessGroup(t *testing.T) {
	dir := t.TempDir()
	pgidFile := filepath.Join(dir, "pgid")
	env := NewLocalEnvironment(dir, nil, 300*time.Millisecond)

	start := time.Now()
	res, err := env.Execute(context.Background(),
		"echo $$ > pgid; echo partial; sleep 30 & wait")
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Fatalf("timeout took %s, kill did not fire", elapsed)
	}
	if res.ReturnCode != -1 {
		t.Errorf("rc = %d, want -1 on timeout", res.ReturnCode)
	}
	if !strings.Contains(res.ExceptionInfo, "timed out") {
		t.Errorf("ExceptionInfo = %q, want a timeout note", res.ExceptionInfo)
	}
	if res.Output != "partial\n" {
		t.Errorf("partial output = %q, want %q", res.Output, "partial\n")
	}

	data, err := os.ReadFile(pgidFile)
	if err != nil {
		t.Fatalf("child never recorded its pgid: %v", err)
	}
	pgid, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil {
		t.Fatalf("bad pgid %q: %v", data, err)
	}
	assertNoGroupMembers(t, pgid)
}

// assertNoGroupMembers polls until the process group has no live members
// (or fails after a short bounded deadline). SIGKILL delivery to the last
// group member can lag the kill(2) call by a scheduling instant, so an
// immediate check would be the flaky one.
func assertNoGroupMembers(t *testing.T, pgid int) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for {
		err := syscall.Kill(-pgid, 0)
		if errors.Is(err, syscall.ESRCH) {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("process group %d still has live members after the timeout kill", pgid)
		}
		// Bounded backoff between checks; the first check almost always
		// already sees ESRCH.
		<-time.After(5 * time.Millisecond)
	}
}

// TestLocalEnvironmentSpawnError: an unusable working directory is an
// environment failure delivered through ExceptionInfo, not an error.
func TestLocalEnvironmentSpawnError(t *testing.T) {
	env := NewLocalEnvironment(filepath.Join(t.TempDir(), "missing"), nil, 0)
	res, err := env.Execute(context.Background(), "echo hi")
	if err != nil {
		t.Fatalf("Execute returned an error for a spawn failure: %v", err)
	}
	if res.ReturnCode != -1 || res.ExceptionInfo == "" {
		t.Errorf("rc = %d, exception = %q, want -1 and a reason", res.ReturnCode, res.ExceptionInfo)
	}
}

// TestLocalEnvironmentContextCancel: the caller's cancellation surfaces as
// an error, not as a result.
func TestLocalEnvironmentContextCancel(t *testing.T) {
	env := NewLocalEnvironment(t.TempDir(), nil, 0)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := env.Execute(ctx, "echo hi"); err == nil {
		t.Fatal("Execute with a canceled context should fail")
	}
}

// TestMergeEnv: base order preserved, duplicates overwritten in place,
// new keys appended.
func TestMergeEnv(t *testing.T) {
	got := mergeEnv(
		[]string{"A=1", "B=2", "C=3"},
		[]string{"B=over", "D=new"},
	)
	want := []string{"A=1", "B=over", "C=3", "D=new"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Errorf("mergeEnv = %v, want %v", got, want)
	}
	if got := mergeEnv([]string{"A=1"}, nil); len(got) != 1 || got[0] != "A=1" {
		t.Errorf("mergeEnv with no extras = %v, want the base copy", got)
	}
}

// TestLocalEnvironmentShellResolvesOnce: the shell resolution succeeds on
// any sane test machine (bash or sh) and caches the absolute path.
func TestLocalEnvironmentShellResolvesOnce(t *testing.T) {
	env := NewLocalEnvironment(t.TempDir(), nil, 0).(*LocalEnvironment)
	env.once.Do(env.resolveShell)
	if env.shell == "" {
		t.Fatalf("no shell found: %v", env.shellErr)
	}
	if !filepath.IsAbs(env.shell) {
		t.Errorf("shell = %q, want an absolute path", env.shell)
	}
	if _, err := os.Stat(env.shell); err != nil {
		t.Errorf("resolved shell missing: %v", err)
	}
	if _, err := exec.LookPath("bash"); err != nil && !strings.HasSuffix(env.shell, "/sh") {
		t.Errorf("fallback to sh expected when bash is missing, got %q", env.shell)
	}
}
