// Package bash implements the mini-swe-agent-compatible coding preset: a
// coding agent that solves tasks by executing shell commands in an
// Environment until it reports completion. The templates, limits, exit
// statuses, and submission mechanism follow mini-swe-agent; the loop itself
// is the unmodified agentloop core. Two transports drive it: the default
// text-action protocol (an AfterGenerate finish veto keeps the run going
// without native tool calls) and WithToolProtocol's native `bash` tool —
// same observations, limits, and exit statuses, different wire format.
package bash

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"text/template"
	"time"

	"github.com/smhanov/llmhub"

	"github.com/smhanov/agentloop"
)

// Exit statuses reported in Answer.ExitStatus. The strings mirror
// mini-swe-agent's exit statuses verbatim for compatibility.
const (
	// ExitSubmitted means the model ran the submit command: the command's
	// first output line equaled the submit marker and the command exited 0.
	ExitSubmitted = "Submitted"
	// ExitLimitsExceeded means the step limit or the cost limit tripped.
	ExitLimitsExceeded = "LimitsExceeded"
	// ExitTimeExceeded means the wall-time limit tripped.
	ExitTimeExceeded = "TimeExceeded"
	// ExitRepeatedFormatError means the model produced malformed responses
	// max_consecutive_format_errors times in a row.
	ExitRepeatedFormatError = "RepeatedFormatError"
)

// Preset defaults, mirroring mini-swe-agent's default config.
const (
	// defaultSubmitMarker is the first output line that ends a run.
	defaultSubmitMarker = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
	// defaultOutputLimit is the observation truncation threshold in runes.
	defaultOutputLimit = 10000
	// defaultMaxConsecutiveFormatErrors stops runs stuck in malformed
	// responses; 0 would mean unlimited.
	defaultMaxConsecutiveFormatErrors = 3
	// coreDefaultMaxToolCalls mirrors the core loop's default tool budget.
	// The text protocol registers no native tools, so this is the value the
	// hard step ceiling (2*MaxToolCalls+2) derives from; in tool mode it is
	// also the default bash-execution budget.
	coreDefaultMaxToolCalls = 5
)

// Option composes core and preset options in a single variadic list.
type Option = agentloop.Option

// Setting keys this preset decodes from the option list via
// agentloop.Settings. They are unexported implementation details; callers
// use the With* constructors below.
const (
	settingEnvironment  = "bash.environment"
	settingStepLimit    = "bash.stepLimit"
	settingCostLimit    = "bash.costLimit"
	settingWallTime     = "bash.wallTimeLimit"
	settingMaxFormatErr = "bash.maxConsecutiveFormatErrors"
	settingSubmitMarker = "bash.submitMarker"
	settingOutputLimit  = "bash.outputLimit"
	settingTemplates    = "bash.templates"
	settingApprover     = "bash.approver"
	settingToolProtocol = "bash.toolProtocol"
	settingHooks        = "bash.hooks"
)

// WithEnvironment replaces the command execution backend; the default is
// NewLocalEnvironment("", nil, 0) — the current directory, the inherited
// environment, and a 30s per-command timeout. Container environments
// implement the Environment interface and drop in here.
func WithEnvironment(env Environment) Option {
	return agentloop.WithSetting(settingEnvironment, env)
}

// WithStepLimit caps the number of model steps (inferences) per run; the
// default 0 means unlimited, like mini's default config. Values below 0
// count as 0. The graceful stop yields ExitStatus ExitLimitsExceeded; the
// core loop's MaxSteps is additionally raised to max(StepLimit,
// 2*MaxToolCalls+2) as the hard safety net.
func WithStepLimit(n int) Option {
	return agentloop.WithSetting(settingStepLimit, n)
}

// WithCostLimit stops the run gracefully once the cumulative cost reported
// by the provider reaches usd (mini's cost_limit semantics: stop after
// exceeding); the default 0 means unlimited. Negative values count as 0.
func WithCostLimit(usd float64) Option {
	return agentloop.WithSetting(settingCostLimit, usd)
}

// WithWallTimeLimit stops the run gracefully once wall-clock time (a
// monotonic clock) reaches d; the default 0 means unlimited. Tripping it
// yields ExitStatus ExitTimeExceeded.
func WithWallTimeLimit(d time.Duration) Option {
	return agentloop.WithSetting(settingWallTime, d)
}

// WithMaxConsecutiveFormatErrors sets how many consecutive malformed
// responses stop the run with ExitStatus ExitRepeatedFormatError. The
// default is 3; 0 means unlimited; negative values keep the default. Any
// cleanly parsed step resets the counter.
func WithMaxConsecutiveFormatErrors(n int) Option {
	return agentloop.WithSetting(settingMaxFormatErr, n)
}

// WithSubmitMarker overrides the output line that signals task completion;
// the default is mini's COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT. A run ends
// with ExitStatus ExitSubmitted when a command exits 0 and prints the
// marker as its first output line.
func WithSubmitMarker(s string) Option {
	return agentloop.WithSetting(settingSubmitMarker, s)
}

// WithOutputLimit sets the observation truncation threshold in runes; the
// default is 10000, rendered as the first and last limit/2 runes plus an
// elision count. Values below 1 keep the default.
func WithOutputLimit(n int) Option {
	return agentloop.WithSetting(settingOutputLimit, n)
}

// WithTemplates overrides the ported mini-swe-agent templates (rendered
// with text/template): the system prompt, the first user message (task),
// and the per-command observation layout. Empty strings keep the
// corresponding default. Task templates receive Task, SubmitMarker, GOOS,
// GOARCH, Hostname, and IsDarwin; observation templates receive Exception,
// ReturnCode, Output, TooLong, Head, Tail, and Elided.
func WithTemplates(system, task, observation string) Option {
	return agentloop.WithSetting(settingTemplates, templateOverrides{
		system: system, task: task, observation: observation,
	})
}

// WithApprover installs an approval gate consulted before every command
// execution. A returned error vetoes the command, which the model sees as
// an {"error": "command not approved: …"} observation while the run
// continues; nil (the default) runs unattended — mini's yolo mode. The
// gate may block for human confirmation, mirroring mini's confirm mode.
func WithApprover(fn func(ctx context.Context, command string) error) Option {
	return agentloop.WithSetting(settingApprover, fn)
}

// WithToolProtocol selects the native tool-calling transport instead of the
// default text protocol: the preset registers a `bash` tool (schema
// {"command": string}) and the model drives the run with tool calls. The
// observation formatting, submit detection, limits, approval gate, and exit
// statuses are identical to the text protocol's; the run ends naturally
// when the model stops calling tools, or with ExitSubmitted once a
// command's output announces the submit marker. The format-error machinery
// is a text-protocol concept and does not run in tool mode. The preset owns
// the registered tool set in this mode: core WithTools options passed
// alongside are overridden.
func WithToolProtocol() Option {
	return agentloop.WithSetting(settingToolProtocol, true)
}

// WithHooks installs additional hook callbacks that compose with the
// protocol's internal hooks: for the two hooks the protocol owns
// (BeforeGenerate for limits, AfterGenerate for the selected transport) the
// protocol hooks run first and the caller's hooks observe the event
// afterwards; all other hook fields pass through untouched. Prefer this
// over agentloop.WithHooks, which replaces the whole hook set and would
// silently disable the protocol.
func WithHooks(h agentloop.Hooks) Option {
	return agentloop.WithSetting(settingHooks, h)
}

// templateOverrides carries the WithTemplates values; empty fields keep
// the ported defaults.
type templateOverrides struct {
	system      string
	task        string
	observation string
}

// approverFn is the approval-gate signature.
type approverFn = func(ctx context.Context, command string) error

// presetConfig is the fully resolved preset configuration for one run.
type presetConfig struct {
	environment         Environment
	stepLimit           int
	costLimit           float64
	wallTimeLimit       time.Duration
	maxConsecFormatErrs int
	submitMarker        string
	outputLimit         int
	systemTemplate      string
	taskTemplate        string
	observationTemplate string
	formatErrorTemplate string
	approver            approverFn
	toolProtocol        bool
}

// resolvePreset decodes the preset settings from opts in order (later
// options win) over the defaults.
func resolvePreset(opts []Option) presetConfig {
	st := agentloop.Settings(opts...)
	cfg := presetConfig{
		environment:         NewLocalEnvironment("", nil, 0),
		maxConsecFormatErrs: defaultMaxConsecutiveFormatErrors,
		submitMarker:        defaultSubmitMarker,
		outputLimit:         defaultOutputLimit,
		systemTemplate:      defaultSystemTemplate,
		taskTemplate:        defaultTaskTemplate,
		observationTemplate: defaultObservationTemplate,
		formatErrorTemplate: defaultFormatErrorTemplate,
	}
	if v, ok := st[settingEnvironment].(Environment); ok && v != nil {
		cfg.environment = v
	}
	if v, ok := st[settingStepLimit].(int); ok {
		cfg.stepLimit = max(v, 0)
	}
	if v, ok := st[settingCostLimit].(float64); ok {
		cfg.costLimit = max(v, 0)
	}
	if v, ok := st[settingWallTime].(time.Duration); ok {
		cfg.wallTimeLimit = max(v, 0)
	}
	if v, ok := st[settingMaxFormatErr].(int); ok {
		switch {
		case v == 0:
			cfg.maxConsecFormatErrs = 0 // unlimited
		case v > 0:
			cfg.maxConsecFormatErrs = v
		}
	}
	if v, ok := st[settingSubmitMarker].(string); ok && v != "" {
		cfg.submitMarker = v
	}
	if v, ok := st[settingOutputLimit].(int); ok && v >= 1 {
		cfg.outputLimit = v
	}
	if v, ok := st[settingTemplates].(templateOverrides); ok {
		if v.system != "" {
			cfg.systemTemplate = v.system
		}
		if v.task != "" {
			cfg.taskTemplate = v.task
		}
		if v.observation != "" {
			cfg.observationTemplate = v.observation
		}
	}
	if v, ok := st[settingApprover].(approverFn); ok {
		cfg.approver = v
	}
	if v, ok := st[settingToolProtocol].(bool); ok {
		cfg.toolProtocol = v
	}
	return cfg
}

// hardStepCeiling maps the preset's step limit onto the core loop's hard
// inference ceiling: max(StepLimit, 2*MaxToolCalls+2) — room for the
// protocol's inferences plus the forced-final margin. 0 keeps the core
// default.
func hardStepCeiling(stepLimit int) int {
	if stepLimit <= 0 {
		return 0
	}
	floor := 2*coreDefaultMaxToolCalls + 2
	return max(stepLimit, floor)
}

// Agent is the bash coding preset: an immutable wrapper around the core
// agentloop.Agent wiring the selected action protocol (text or native
// tool), the limits, and the approval gate. Safe for concurrent use.
type Agent struct {
	agent *agentloop.Agent
	opts  []Option // construction options, re-applied per Ask for overrides
	sys   systemData
}

// New returns a coding agent that executes bash commands in an Environment
// until the task is submitted or a limit is hit. The default protocol is
// mini-swe-agent's text-action protocol. Core options passed in opts
// (budgets, skills, …) apply after the preset defaults, so later options
// win; the preset itself owns the system prompt and the hard step ceiling.
func New(llm *llmhub.Client, opts ...Option) (*Agent, error) {
	cfg := resolvePreset(opts)
	sys := newSystemData()
	systemPrompt, err := renderSystem(cfg.systemTemplate, sys)
	if err != nil {
		return nil, err
	}
	core, err := agentloop.New(llm, append([]Option{
		agentloop.WithSystemPrompt(systemPrompt),
	}, opts...)...)
	if err != nil {
		return nil, err
	}
	return &Agent{agent: core, opts: append([]Option(nil), opts...), sys: sys}, nil
}

// Ask runs the agent on one task and returns the full trajectory. It
// returns a *Answer for every agent-level outcome — submitted, limits, or
// repeated format errors are results, not errors (mini's semantics); Go
// errors are reserved for infrastructure failures (transport, hooks,
// context). opts override the construction options for this call.
func (a *Agent) Ask(ctx context.Context, task string, opts ...Option) (*Answer, error) {
	coreAns, st, err := a.run(ctx, task, opts...)
	if err != nil {
		return nil, err
	}
	return &Answer{Answer: coreAns, ExitStatus: st.exitStatus, Submission: st.submission}, nil
}

// AskStream is Ask with streaming; events flow exactly as the core
// delivers them, with the protocol hooks active. The streamed EventDone
// carries the core answer only — ExitStatus and Submission are not
// attached to events, so streaming consumers needing them should use Ask
// (or inspect the trajectory in EventDone.Answer.Messages).
func (a *Agent) AskStream(ctx context.Context, task string, opts ...Option) (<-chan agentloop.Event, error) {
	events, err := a.runStream(ctx, task, opts...)
	return events, err
}

// Agent returns the wrapped core agent — the escape hatch to the full core
// API. Note that the text-action protocol lives in the preset's Ask and
// AskStream; a core Agent (including conversations built from this escape
// hatch) runs without the protocol, limits, and exit statuses.
func (a *Agent) Agent() *agentloop.Agent { return a.agent }

// Answer is a bash run's result: the core trace plus mini-swe-agent's exit
// status and submission. ExitStatus is empty only when the run ended
// through the core loop without a preset-level outcome — the tool
// protocol's natural termination (the model stopped calling tools), which
// the text protocol never produces.
type Answer struct {
	*agentloop.Answer
	// ExitStatus reports how the run ended: ExitSubmitted,
	// ExitLimitsExceeded, ExitTimeExceeded, or ExitRepeatedFormatError.
	ExitStatus string
	// Submission holds the output that followed the submit marker's line
	// when ExitStatus is ExitSubmitted.
	Submission string
}

// run executes one task: resolve the effective configuration, render the
// task prompt, wire the protocol hooks, and run the core loop.
func (a *Agent) run(ctx context.Context, task string, opts ...Option) (*agentloop.Answer, *runState, error) {
	merged := append(append([]Option(nil), a.opts...), opts...)
	extras, st, err := a.prepare(merged, task)
	if err != nil {
		return nil, nil, err
	}
	coreAns, err := a.agent.Ask(ctx, extras.query, extras.opts...)
	if err != nil {
		return nil, nil, err
	}
	return coreAns, st, nil
}

// runStream is run for streaming; it shares all wiring and differs only in
// the core entry point.
func (a *Agent) runStream(ctx context.Context, task string, opts ...Option) (<-chan agentloop.Event, error) {
	merged := append(append([]Option(nil), a.opts...), opts...)
	extras, _, err := a.prepare(merged, task)
	if err != nil {
		return nil, err
	}
	return a.agent.AskStream(ctx, extras.query, extras.opts...)
}

// runExtras is what prepare hands to the core call: the rendered query and
// the options appended after the merged list.
type runExtras struct {
	query string
	opts  []Option
}

// prepare resolves the effective preset config from the merged option
// list, renders the task prompt, and builds the protocol hook set.
func (a *Agent) prepare(merged []Option, task string) (runExtras, *runState, error) {
	cfg := resolvePreset(merged)
	st := &runState{start: time.Now()}
	p, err := newProtocol(cfg, st)
	if err != nil {
		return runExtras{}, nil, err
	}
	query, err := renderTask(cfg.taskTemplate, taskData{
		Task:         task,
		SubmitMarker: cfg.submitMarker,
		GOOS:         a.sys.GOOS,
		GOARCH:       a.sys.GOARCH,
		Hostname:     a.sys.Hostname,
		IsDarwin:     a.sys.IsDarwin,
	})
	if err != nil {
		return runExtras{}, nil, err
	}
	hooks := agentloop.Hooks{
		BeforeGenerate: p.beforeGenerate,
		AfterGenerate:  p.afterGenerate,
	}
	coreOpts := append([]Option(nil), merged...)
	if cfg.toolProtocol {
		// Tool protocol: the native bash tool replaces the text-protocol
		// AfterGenerate hook — only the limits' step/cost counter stays.
		// Termination is the loop's natural one (the model stops calling
		// tools), and the preset owns the registered tool set.
		hooks.AfterGenerate = p.countStep
		coreOpts = append(coreOpts, agentloop.WithTools(agentloop.Tool{
			Def:        bashToolDef(cfg.submitMarker),
			Run:        p.handleToolCall,
			Sequential: true, // one stateful shell: strict call order, no overlap
		}))
	}
	if user, ok := agentloop.Settings(merged...)[settingHooks].(agentloop.Hooks); ok {
		hooks = composeHooks(hooks, user)
	}
	systemPrompt, err := renderSystem(cfg.systemTemplate, a.sys)
	if err != nil {
		return runExtras{}, nil, err
	}
	coreOpts = append(coreOpts,
		agentloop.WithSystemPrompt(systemPrompt),
		agentloop.WithHooks(hooks),
	)
	if ceiling := hardStepCeiling(cfg.stepLimit); ceiling > 0 {
		coreOpts = append(coreOpts, agentloop.WithMaxSteps(ceiling))
	}
	return runExtras{query: query, opts: coreOpts}, st, nil
}

// runState is the per-run protocol state; it lives in locals of one Ask.
type runState struct {
	start      time.Time
	steps      int     // model inferences observed so far
	cost       float64 // cumulative provider-reported cost
	fmtErrs    int     // consecutive malformed responses
	exitStatus string
	submission string
}

// protocol bundles what the hooks need for one run.
type protocol struct {
	cfg     presetConfig
	st      *runState
	obsTmpl *template.Template
	fmtTmpl *template.Template
}

// newProtocol parses the observation and format-error templates for one
// run; configuration errors surface here instead of mid-run.
func newProtocol(cfg presetConfig, st *runState) (*protocol, error) {
	obsTmpl, err := parseTemplate("observation", cfg.observationTemplate)
	if err != nil {
		return nil, fmt.Errorf("bash: %w", err)
	}
	fmtTmpl, err := parseTemplate("format-error", cfg.formatErrorTemplate)
	if err != nil {
		return nil, fmt.Errorf("bash: %w", err)
	}
	return &protocol{cfg: cfg, st: st, obsTmpl: obsTmpl, fmtTmpl: fmtTmpl}, nil
}

// beforeGenerate enforces the graceful limits before each inference:
// step count, cumulative cost, and wall-clock time. Each maps to
// mini-swe-agent's exit statuses; the wall-time limit specifically yields
// ExitTimeExceeded. A tool call that hit the submit marker stops here too —
// the earliest point after the execution (the tool protocol's submission
// stop; the text protocol ends inside AfterGenerate instead).
func (p *protocol) beforeGenerate(ctx context.Context, ev *agentloop.Event) error {
	cfg, st := p.cfg, p.st
	switch {
	case st.exitStatus == ExitSubmitted:
		ev.Skip = true
	case cfg.stepLimit > 0 && st.steps >= cfg.stepLimit,
		cfg.costLimit > 0 && st.cost >= cfg.costLimit:
		st.exitStatus = ExitLimitsExceeded
		ev.Skip = true
	case cfg.wallTimeLimit > 0 && time.Since(st.start) >= cfg.wallTimeLimit:
		st.exitStatus = ExitTimeExceeded
		ev.Skip = true
	}
	return nil
}

// approvalError is the model-facing shape of a vetoed command.
type approvalError struct {
	Error string `json:"error"`
}

// afterGenerate implements the text-action protocol: parse exactly one
// fenced command from the response, execute it through the shared path
// (runCommand), append the formatted observation as a user message, and
// veto the finish so the loop continues. Zero or multiple blocks append
// format-error guidance and count toward the consecutive-error stop. Submit
// detection wraps every execution: the marker on the first output line of a
// successful command ends the run.
func (p *protocol) afterGenerate(ctx context.Context, ev *agentloop.Event) error {
	st := p.st
	st.steps++
	if resp := ev.Response; resp != nil {
		st.cost += resp.Usage.Cost
	}
	commands := findCommands(ev.Response.Text())
	if len(commands) != 1 {
		return p.formatError(ev, len(commands))
	}
	st.fmtErrs = 0 // a cleanly parsed step resets the counter

	out, err := p.runCommand(ctx, commands[0])
	if err != nil {
		return err
	}
	ev.Messages = append(ev.Messages, llmhub.NewUserMessage(llmhub.Text(out.observation)))
	if out.submitted {
		st.exitStatus = ExitSubmitted
		st.submission = out.submission
		return nil // leave Finish true: the run ends here
	}
	ev.Finish = false
	return nil
}

// commandOutcome is the shared execution path's result for one model
// command: the model-facing observation (the §8.4 shape) and the
// submit-marker outcome (§8.3.4). An approval-gate veto yields the JSON
// not-approved observation with submitted=false; the run continues either
// way.
type commandOutcome struct {
	observation string
	submission  string
	submitted   bool
}

// runCommand executes one model-chosen command through the path both
// protocols share: the approval gate (a veto becomes the JSON not-approved
// observation), Environment.Execute, the observation formatter, and
// submit-marker detection. Only infrastructure failures and caller
// cancellation return an error — every agent-level outcome is expressed as
// the observation instead.
func (p *protocol) runCommand(ctx context.Context, command string) (commandOutcome, error) {
	if p.cfg.approver != nil {
		if err := p.cfg.approver(ctx, command); err != nil {
			payload, marshalErr := json.Marshal(approvalError{
				Error: fmt.Sprintf("command not approved: %v", err),
			})
			if marshalErr != nil {
				return commandOutcome{}, fmt.Errorf("bash: encode approval error: %w", marshalErr)
			}
			return commandOutcome{observation: string(payload)}, nil
		}
	}
	res, err := p.cfg.environment.Execute(ctx, command)
	if err != nil {
		if ctx.Err() != nil {
			return commandOutcome{}, ctx.Err() // the caller aborted; surface the cancellation
		}
		return commandOutcome{}, fmt.Errorf("bash: execute: %w", err)
	}
	observation, err := formatObservation(res, p.cfg.outputLimit, p.obsTmpl)
	if err != nil {
		return commandOutcome{}, err
	}
	out := commandOutcome{observation: observation}
	if submission, ok := detectSubmission(res.Output, p.cfg.submitMarker, res.ReturnCode); ok {
		out.submission, out.submitted = submission, true
	}
	return out, nil
}

// formatError appends the format-error guidance for a response with count
// (not exactly one) action blocks, increments the consecutive counter, and
// stops the run with ExitRepeatedFormatError once the configured maximum
// is reached.
func (p *protocol) formatError(ev *agentloop.Event, count int) error {
	p.st.fmtErrs++
	msg, err := renderText(p.fmtTmpl, formatErrorData{
		Error:        fmt.Sprintf("Expected exactly 1 action, found %d.", count),
		Count:        count,
		SubmitMarker: p.cfg.submitMarker,
	})
	if err != nil {
		return err
	}
	ev.Messages = append(ev.Messages, llmhub.NewUserMessage(llmhub.Text(msg)))
	if p.cfg.maxConsecFormatErrs > 0 && p.st.fmtErrs >= p.cfg.maxConsecFormatErrs {
		p.st.exitStatus = ExitRepeatedFormatError
		ev.Finish = true // stop the run
		return nil
	}
	ev.Finish = false // a malformed response never executes its (native) tool calls
	return nil
}

// composeHooks merges the protocol's hooks with caller hooks: for the two
// hooks the protocol owns, the protocol hook runs first and the caller's
// observes the same event afterwards (an error from either aborts); every
// other field passes through from the caller's set.
func composeHooks(protocolHooks agentloop.Hooks, user agentloop.Hooks) agentloop.Hooks {
	out := user
	pb, ub := protocolHooks.BeforeGenerate, user.BeforeGenerate
	out.BeforeGenerate = func(ctx context.Context, ev *agentloop.Event) error {
		if err := pb(ctx, ev); err != nil {
			return err
		}
		if ub != nil {
			return ub(ctx, ev)
		}
		return nil
	}
	pa, ua := protocolHooks.AfterGenerate, user.AfterGenerate
	out.AfterGenerate = func(ctx context.Context, ev *agentloop.Event) error {
		if err := pa(ctx, ev); err != nil {
			return err
		}
		if ua != nil {
			return ua(ctx, ev)
		}
		return nil
	}
	return out
}

// detectSubmission reports whether a command output announces completion:
// the first output line, stripped, equals the submit marker and the
// command exited 0 (mini's _check_finished). The submission is the
// remaining output, verbatim.
func detectSubmission(output, marker string, returnCode int) (submission string, ok bool) {
	if returnCode != 0 {
		return "", false
	}
	trimmed := strings.TrimLeft(output, " \t\n\r\v\f")
	first, rest, hasRest := strings.Cut(trimmed, "\n")
	if !hasRest {
		rest = ""
	}
	if strings.TrimSpace(first) != marker {
		return "", false
	}
	return rest, true
}
