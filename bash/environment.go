package bash

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"sync"
	"syscall"
	"time"
)

// defaultCommandTimeout is the per-command timeout used when
// NewLocalEnvironment is given a non-positive timeout.
const defaultCommandTimeout = 30 * time.Second

// waitDelay bounds how long os/exec waits for output pipes to close after
// the timeout kill, so grandchildren holding the pipes cannot stall Execute.
const waitDelay = 100 * time.Millisecond

// Environment executes shell commands for the bash preset. Only
// LocalEnvironment ships in this package; container and remote backends
// (Docker, podman, SSH, …) implement this interface in caller code.
type Environment interface {
	// Execute runs command in a shell and returns its outcome. Non-zero
	// exit codes are results, not errors — the agent iterates on failing
	// commands. Execution failures of the environment itself (timeout,
	// spawn error) are also results: they arrive as ReturnCode -1 with a
	// non-empty ExceptionInfo. The error return is reserved for the
	// caller's context being canceled or otherwise aborting the run.
	Execute(ctx context.Context, command string) (ExecResult, error)
}

// ExecResult is the outcome of one command execution.
type ExecResult struct {
	// Output is the combined stdout and stderr of the command, in arrival
	// order. On a timeout it holds whatever the command produced before it
	// was killed.
	Output string
	// ReturnCode is the command's exit status. It is -1 when the command
	// never produced one (timeout or spawn failure).
	ReturnCode int
	// ExceptionInfo is non-empty when the execution itself failed — for
	// example the command timed out or could not be started. The command's
	// own failure (non-zero exit) does not set it.
	ExceptionInfo string
}

// NewLocalEnvironment returns an Environment that runs commands on the
// local machine with bash (sh as the fallback). An empty cwd means the
// process's current working directory; env entries are merged over the
// inherited environment; a non-positive timeout means the 30s default.
func NewLocalEnvironment(cwd string, env []string, timeout time.Duration) Environment {
	if timeout <= 0 {
		timeout = defaultCommandTimeout
	}
	return &LocalEnvironment{
		cwd:      cwd,
		extraEnv: append([]string(nil), env...),
		timeout:  timeout,
	}
}

// LocalEnvironment runs every command in a fresh shell subprocess,
// mirroring mini-swe-agent's LocalEnvironment: bash -c with a sh fallback,
// a fixed working directory, extra environment variables merged over the
// inherited ones, and a per-command timeout. Children start in their own
// process group; a timeout kills the whole group so no orphans survive.
// The zero value is usable and resolves cwd and timeout per Execute.
type LocalEnvironment struct {
	cwd      string
	extraEnv []string
	timeout  time.Duration

	once     sync.Once
	shell    string // absolute path of the resolved shell
	shellErr error  // set when no shell was found
}

// Execute implements Environment.
func (e *LocalEnvironment) Execute(ctx context.Context, command string) (ExecResult, error) {
	e.once.Do(e.resolveShell)
	if e.shell == "" {
		return ExecResult{
			ReturnCode:    -1,
			ExceptionInfo: fmt.Sprintf("no shell found: %v", e.shellErr),
		}, nil
	}
	dir := e.cwd
	if dir == "" {
		wd, err := os.Getwd()
		if err != nil {
			return ExecResult{
				ReturnCode:    -1,
				ExceptionInfo: fmt.Sprintf("resolve working directory: %v", err),
			}, nil
		}
		dir = wd
	}

	// The run context carries the per-command timeout on top of the
	// caller's context; firing it triggers cmd.Cancel below.
	runCtx, cancel := context.WithTimeout(ctx, e.timeout)
	defer cancel()

	cmd := exec.CommandContext(runCtx, e.shell, "-c", command)
	cmd.Dir = dir
	cmd.Env = mergeEnv(os.Environ(), e.extraEnv)
	// New process group: the child becomes a group leader, so the whole
	// tree can be killed with one signal to -pid.
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	var out bytes.Buffer
	// The same writer for both pipes: os/exec serializes writes to it.
	cmd.Stdout = &out
	cmd.Stderr = &out
	cmd.Cancel = func() error {
		if cmd.Process == nil {
			return nil
		}
		return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	}
	cmd.WaitDelay = waitDelay

	err := cmd.Run()
	if ctx.Err() != nil {
		// The caller canceled; surface the cancellation to the loop.
		return ExecResult{}, ctx.Err()
	}
	if runCtx.Err() != nil { // deadline exceeded
		return ExecResult{
			Output:        out.String(),
			ReturnCode:    -1,
			ExceptionInfo: fmt.Sprintf("command timed out after %s and its process group was killed", e.timeout),
		}, nil
	}
	if err != nil {
		if _, ok := err.(*exec.ExitError); !ok {
			// The command never ran (start failure), which is an
			// environment failure, not a command result.
			return ExecResult{
				ReturnCode:    -1,
				ExceptionInfo: fmt.Sprintf("start command: %v", err),
			}, nil
		}
		// Non-zero exit or signal: a result for the model to iterate on.
	}
	rc := 0
	if cmd.ProcessState != nil {
		rc = cmd.ProcessState.ExitCode() // -1 when the command was signaled
	}
	return ExecResult{Output: out.String(), ReturnCode: rc}, nil
}

// resolveShell finds the shell once: bash preferred, sh as the fallback.
func (e *LocalEnvironment) resolveShell() {
	for _, name := range []string{"bash", "sh"} {
		if path, err := exec.LookPath(name); err == nil {
			e.shell = path
			return
		} else {
			e.shellErr = err
		}
	}
}

// mergeEnv merges extra over base, preserving base's order and appending
// new keys; later entries win on duplicates.
func mergeEnv(base, extra []string) []string {
	if len(extra) == 0 {
		return append([]string(nil), base...)
	}
	merged := make([]string, 0, len(base)+len(extra))
	index := make(map[string]int, len(base)+len(extra))
	for _, kv := range base {
		key, _, _ := strings.Cut(kv, "=")
		if at, ok := index[key]; ok {
			merged[at] = kv
			continue
		}
		index[key] = len(merged)
		merged = append(merged, kv)
	}
	for _, kv := range extra {
		key, _, _ := strings.Cut(kv, "=")
		if at, ok := index[key]; ok {
			merged[at] = kv
			continue
		}
		index[key] = len(merged)
		merged = append(merged, kv)
	}
	return merged
}
