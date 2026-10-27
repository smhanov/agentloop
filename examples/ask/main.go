// Command ask is a general-purpose runner for the three agentloop agents:
//
//	plain  the bare core loop (no tools beyond -skills)
//	search web_search + fetch_page backed by Serper (the default)
//	bash   a mini-swe-agent-style coding agent
//
// Prompts arrive via -prompt or -prompt-file; output is plain text,
// live streaming (-stream), or a JSON answer object (-json). It is also
// the primary tool for exercising the local test-prompt corpus.
//
// Provider credentials are read from -api-key or, failing that, the
// provider's standard environment variable (for example ZAI_API_KEY for
// the zai provider, OPENAI_API_KEY for openai). Keys are never read from
// anywhere else and never logged.
//
// The default live configuration, matching the local key file:
//
//	export ZAI_API_KEY=… SERPER_API_KEY=…
//	go run ./examples/ask \
//	  -provider zai -model glm-5.3-flash \
//	  -base-url https://api.z.ai/api/coding/paas/v4 \
//	  -serper-key "$SERPER_API_KEY" -prompt "…" -stream
//
// Exit codes: 0 success (any bash exit status counts as success),
// 1 error, 2 budget exhausted.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/smhanov/llmhub"

	"github.com/smhanov/agentloop"
	"github.com/smhanov/agentloop/bash"
	"github.com/smhanov/agentloop/search"

	// Providers register themselves on import; the same set as llmhub's
	// own CLI example.
	_ "github.com/smhanov/llmhub/providers/anthropic"
	_ "github.com/smhanov/llmhub/providers/gemini"
	_ "github.com/smhanov/llmhub/providers/ollama"
	_ "github.com/smhanov/llmhub/providers/openai"
	_ "github.com/smhanov/llmhub/providers/openrouter"
	_ "github.com/smhanov/llmhub/providers/xai"
	_ "github.com/smhanov/llmhub/providers/zai"
)

// Agent kinds accepted by -agent.
const (
	agentKindSearch = "search"
	agentKindBash   = "bash"
	agentKindPlain  = "plain"
)

// appConfig is the parsed flag set.
type appConfig struct {
	provider      string
	model         string
	apiKey        string
	baseURL       string
	agent         string
	chat          bool
	cacheKey      string
	serperKey     string
	serperBaseURL string
	prompt        string
	promptFile    string
	system        string
	cwd           string
	envPairs      repeatFlags
	skills        string
	stepLimit     int
	tokenLimit    int64
	confirm       bool
	toolProtocol  bool
	stream        bool
	verbose       bool
	asJSON        bool
	maxToolCalls  int
	maxSteps      int

	promptText string // resolved from -prompt or -prompt-file
}

// usageText precedes the flag list in -h output.
const usageText = `ask — run an agentloop agent from the command line

Agents:
  -agent search  web_search and fetch_page tools backed by Serper (default)
  -agent bash    coding agent executing shell commands (mini-swe-agent style)
  -agent plain   the bare core loop, no tools

Examples:
  export ZAI_API_KEY=… SERPER_API_KEY=…
  go run ./examples/ask \
    -provider zai -model glm-5.3-flash \
    -base-url https://api.z.ai/api/coding/paas/v4 \
    -serper-key "$SERPER_API_KEY" \
    -prompt "What Go version was released most recently, and when?" -stream

  go run ./examples/ask -provider zai -model glm-5.3-flash \
    -base-url https://api.z.ai/api/coding/paas/v4 \
    -agent bash -cwd /tmp/workspace -step-limit 20 -token-limit 2000000 \
    -prompt "Add a README.md describing this project." -stream -v

Chat mode (-chat) reads one prompt per stdin line, sharing one
conversation; a blank line or end of input ends the session. With -json
it prints one JSON object per turn.

With -agent bash, -system overrides the preset's system template, and the
exit-status summary is printed in non-streaming runs. Any bash exit
status (submitted, limits, format errors) is a successful run.

Exit codes: 0 success (any bash exit status), 1 error, 2 budget exhausted.

Flags:
`

// repeatFlags collects repeated string flags (used for -env).
type repeatFlags []string

func (f *repeatFlags) String() string { return strings.Join(*f, ", ") }

func (f *repeatFlags) Set(v string) error {
	*f = append(*f, v)
	return nil
}

// newFlagSet builds the flag set bound to cfg, writing usage to stderr.
func newFlagSet(cfg *appConfig, stderr io.Writer) *flag.FlagSet {
	fs := flag.NewFlagSet("ask", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() {
		fmt.Fprint(stderr, usageText)
		fs.PrintDefaults()
	}

	fs.StringVar(&cfg.provider, "provider", "", "llmhub provider name (required); the API key comes from -api-key or the provider's standard environment variable (ZAI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, XAI_API_KEY, OPENROUTER_API_KEY)")
	fs.StringVar(&cfg.model, "model", "", "model identifier")
	fs.StringVar(&cfg.apiKey, "api-key", "", "LLM API key (overrides the provider's environment variable)")
	fs.StringVar(&cfg.baseURL, "base-url", "", "provider base URL override (for zai coding accounts: https://api.z.ai/api/coding/paas/v4)")
	fs.StringVar(&cfg.agent, "agent", agentKindSearch, "agent preset: search (web search via Serper), bash (coding agent), or plain (no tools)")
	fs.BoolVar(&cfg.chat, "chat", false, "multi-turn mode: read one prompt per stdin line until a blank line or end of input; one shared conversation")
	fs.StringVar(&cfg.cacheKey, "cache-key", "", "llmhub prompt-cache key, recommended with -chat")
	fs.StringVar(&cfg.serperKey, "serper-key", "", "Serper API key for the search agent (or SERPER_API_KEY environment variable)")
	fs.StringVar(&cfg.serperBaseURL, "serper-base-url", "", "override the Serper API base URL (test hook)")
	fs.StringVar(&cfg.prompt, "prompt", "", "prompt/task text")
	fs.StringVar(&cfg.promptFile, "prompt-file", "", "read the prompt from a file")
	fs.StringVar(&cfg.system, "system", "", "replace the system prompt (for bash this overrides the preset's system template)")
	fs.StringVar(&cfg.cwd, "cwd", ".", "working directory for the bash agent")
	fs.Var(&cfg.envPairs, "env", "extra KEY=VALUE environment entry for the bash agent (repeatable)")
	fs.StringVar(&cfg.skills, "skills", "", "comma-separated skill directories to load (works with every agent)")
	fs.IntVar(&cfg.stepLimit, "step-limit", 0, "bash agent: maximum model steps per run (0 means unlimited)")
	fs.Int64Var(&cfg.tokenLimit, "token-limit", 0, "bash agent: stop once cumulative counted tokens reach this value (0 means unlimited)")
	fs.BoolVar(&cfg.confirm, "confirm", false, "bash agent: ask before every command (reads y/n from stdin); default is unattended")
	fs.BoolVar(&cfg.toolProtocol, "tool-protocol", false, "bash agent: native tool calling instead of the default text-action protocol")
	fs.BoolVar(&cfg.stream, "stream", false, "stream answer deltas to stdout as they arrive; reasoning deltas dimmed on stderr; tool activity inline (the bash exit-status summary is only printed without -stream)")
	fs.BoolVar(&cfg.verbose, "v", false, "print the event trace to stderr: steps, tool calls with durations, exit status, timing")
	fs.BoolVar(&cfg.asJSON, "json", false, "print the final answer as JSON (text, reasoning, status, usage, timing, trace, sources or submission); with -chat, one object per turn")
	fs.IntVar(&cfg.maxToolCalls, "max-tool-calls", 0, "budget override: tool executions per run (0 keeps the preset default)")
	fs.IntVar(&cfg.maxSteps, "max-steps", 0, "budget override: model inferences per run (0 keeps the default)")
	return fs
}

// parseConfig parses args into a validated appConfig. A flag.ErrHelp
// error means usage was printed.
func parseConfig(args []string, stderr io.Writer) (*appConfig, error) {
	cfg := &appConfig{}
	if err := newFlagSet(cfg, stderr).Parse(args); err != nil {
		return nil, err
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	return cfg, nil
}

// validate checks flag combinations and resolves the prompt text.
func (cfg *appConfig) validate() error {
	if cfg.provider == "" {
		return errors.New("-provider is required")
	}
	switch cfg.agent {
	case agentKindSearch, agentKindBash, agentKindPlain:
	default:
		return fmt.Errorf("unknown -agent %q (want search, bash, or plain)", cfg.agent)
	}
	if cfg.prompt != "" && cfg.promptFile != "" {
		return errors.New("-prompt and -prompt-file are mutually exclusive")
	}
	if !cfg.chat && cfg.prompt == "" && cfg.promptFile == "" {
		return errors.New("one of -prompt or -prompt-file is required")
	}
	for _, pair := range cfg.envPairs {
		if _, _, ok := strings.Cut(pair, "="); !ok {
			return fmt.Errorf("-env %q is not KEY=VALUE", pair)
		}
	}
	if cfg.promptFile != "" {
		data, err := os.ReadFile(cfg.promptFile)
		if err != nil {
			return fmt.Errorf("read prompt file: %w", err)
		}
		cfg.promptText = string(data)
	} else {
		cfg.promptText = cfg.prompt
	}
	return nil
}

// run is the CLI entry point: it returns the process exit code.
func run(args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	cfg, err := parseConfig(args, stderr)
	switch {
	case errors.Is(err, flag.ErrHelp):
		return 0
	case err != nil:
		fmt.Fprintf(stderr, "ask: %v\nrun with -h for usage\n", err)
		return 1
	}

	ctx := context.Background()
	if cfg.chat {
		return cfg.runChat(ctx, stdin, stdout, stderr)
	}
	return cfg.runOnce(ctx, stdin, stdout, stderr)
}

// newClient builds the llmhub client: the API key comes from -api-key or
// the provider's standard environment variable.
func (cfg *appConfig) newClient() (*llmhub.Client, error) {
	key := cfg.apiKey
	if key == "" {
		key = os.Getenv(envVarForProvider(cfg.provider))
	}
	var opts []llmhub.Option
	if cfg.model != "" {
		opts = append(opts, llmhub.WithModel(cfg.model))
	}
	if cfg.baseURL != "" {
		opts = append(opts, llmhub.WithBaseURL(cfg.baseURL))
	}
	return llmhub.New(cfg.provider, key, opts...)
}

// providerEnvKeys maps provider names to their standard key environment
// variables, mirroring llmhub's example CLI.
var providerEnvKeys = map[string]string{
	"openai":     "OPENAI_API_KEY",
	"anthropic":  "ANTHROPIC_API_KEY",
	"gemini":     "GEMINI_API_KEY",
	"xai":        "XAI_API_KEY",
	"openrouter": "OPENROUTER_API_KEY",
	"zai":        "ZAI_API_KEY",
}

// envVarForProvider returns the environment variable holding the API key
// for a provider: the documented name when there is one, otherwise the
// upper-snake provider name plus "_API_KEY".
func envVarForProvider(provider string) string {
	if v, ok := providerEnvKeys[provider]; ok {
		return v
	}
	var b strings.Builder
	for _, r := range strings.ToUpper(provider) {
		if (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
		} else {
			b.WriteByte('_')
		}
	}
	b.WriteString("_API_KEY")
	return b.String()
}

// coreOptions builds the options shared by every agent kind. Order
// matters: presets put their defaults first, so later options here
// override them (-system replaces a preset's system prompt).
func (cfg *appConfig) coreOptions() []agentloop.Option {
	var opts []agentloop.Option
	if cfg.cacheKey != "" {
		opts = append(opts, agentloop.WithGenerateOptions(llmhub.WithCacheKey(cfg.cacheKey)))
	}
	if cfg.maxToolCalls > 0 {
		opts = append(opts, agentloop.WithMaxToolCalls(cfg.maxToolCalls))
	}
	if cfg.maxSteps > 0 {
		opts = append(opts, agentloop.WithMaxSteps(cfg.maxSteps))
	}
	if cfg.skills != "" {
		var dirs []string
		for _, d := range strings.Split(cfg.skills, ",") {
			if d = strings.TrimSpace(d); d != "" {
				dirs = append(dirs, d)
			}
		}
		if len(dirs) > 0 {
			opts = append(opts, agentloop.WithSkills(dirs...))
		}
	}
	if cfg.system != "" {
		opts = append(opts, agentloop.WithSystemPrompt(cfg.system))
	}
	return opts
}

// agentHandle unifies the three agent kinds: ask returns the enriched
// answer (bash exit status, search sources are derived at print time),
// stream is the raw core event channel, and core is the escape hatch the
// chat loop builds its Conversation from.
type agentHandle struct {
	kind   string
	core   *agentloop.Agent
	ask    func(ctx context.Context, query string) (*runResult, error)
	stream func(ctx context.Context, query string) (<-chan agentloop.Event, error)
}

// newAgent assembles the selected agent over the client.
func (cfg *appConfig) newAgent(stdin io.Reader, stderr io.Writer) (*agentHandle, error) {
	client, err := cfg.newClient()
	if err != nil {
		return nil, err
	}
	coreOpts := cfg.coreOptions()
	switch cfg.agent {
	case agentKindPlain:
		a, err := agentloop.New(client, coreOpts...)
		if err != nil {
			return nil, err
		}
		return &agentHandle{
			kind: agentKindPlain,
			core: a,
			ask: func(ctx context.Context, query string) (*runResult, error) {
				ans, err := a.Ask(ctx, query)
				if ans == nil {
					return nil, err
				}
				return &runResult{kind: agentKindPlain, core: ans}, err
			},
			stream: func(ctx context.Context, query string) (<-chan agentloop.Event, error) {
				return a.AskStream(ctx, query)
			},
		}, nil
	case agentKindSearch:
		var opts []agentloop.Option
		if cfg.serperKey != "" {
			opts = append(opts, search.WithSerperKey(cfg.serperKey))
		}
		if cfg.serperBaseURL != "" {
			opts = append(opts, search.WithSerperBaseURL(cfg.serperBaseURL))
		}
		opts = append(opts, coreOpts...)
		a, err := search.New(client, opts...)
		if err != nil {
			return nil, err
		}
		return &agentHandle{
			kind: agentKindSearch,
			core: a.Agent(),
			ask: func(ctx context.Context, query string) (*runResult, error) {
				ans, err := a.Ask(ctx, query)
				if ans == nil {
					return nil, err
				}
				return &runResult{kind: agentKindSearch, core: ans.Answer}, err
			},
			stream: func(ctx context.Context, query string) (<-chan agentloop.Event, error) {
				return a.AskStream(ctx, query)
			},
		}, nil
	case agentKindBash:
		opts := []agentloop.Option{
			bash.WithEnvironment(bash.NewLocalEnvironment(cfg.cwd, []string(cfg.envPairs), 0)),
		}
		if cfg.stepLimit > 0 {
			opts = append(opts, bash.WithStepLimit(cfg.stepLimit))
		}
		if cfg.tokenLimit > 0 {
			opts = append(opts, bash.WithTokenLimit(cfg.tokenLimit))
		}
		if cfg.toolProtocol {
			opts = append(opts, bash.WithToolProtocol())
		}
		if cfg.confirm {
			opts = append(opts, bash.WithApprover(confirmingApprover(stdin, stderr)))
		}
		opts = append(opts, coreOpts...)
		a, err := bash.New(client, opts...)
		if err != nil {
			return nil, err
		}
		return &agentHandle{
			kind: agentKindBash,
			core: a.Agent(),
			ask: func(ctx context.Context, query string) (*runResult, error) {
				ans, err := a.Ask(ctx, query)
				if ans == nil {
					return nil, err
				}
				return &runResult{
					kind:       agentKindBash,
					core:       ans.Answer,
					exitStatus: ans.ExitStatus,
					submission: ans.Submission,
				}, err
			},
			stream: func(ctx context.Context, query string) (<-chan agentloop.Event, error) {
				return a.AskStream(ctx, query)
			},
		}, nil
	}
	return nil, fmt.Errorf("unknown -agent %q", cfg.agent)
}

// runOnce executes one prompt and prints the answer. It returns the
// process exit code.
func (cfg *appConfig) runOnce(ctx context.Context, stdin io.Reader, stdout, stderr io.Writer) int {
	handle, err := cfg.newAgent(stdin, stderr)
	if err != nil {
		fmt.Fprintf(stderr, "ask: %v\n", err)
		return 1
	}

	var (
		res  *runResult
		rerr error
	)
	if cfg.stream {
		events, err := handle.stream(ctx, cfg.promptText)
		if err != nil {
			fmt.Fprintf(stderr, "ask: %v\n", err)
			return 1
		}
		p := &streamPrinter{stdout: stdout, stderr: stderr, verbose: cfg.verbose}
		core, runErr := p.consume(events)
		p.endLine()
		res, rerr = newRunResult(handle.kind, core), runErr
	} else {
		res, rerr = handle.ask(ctx, cfg.promptText)
	}

	if rerr != nil {
		if cfg.asJSON && res != nil && res.core != nil {
			printJSON(stdout, res)
		}
		if cfg.verbose && res != nil && res.core != nil {
			printTrace(stderr, res)
		}
		fmt.Fprintf(stderr, "ask: %v\n", rerr)
		return exitCodeFor(rerr)
	}
	if res == nil || res.core == nil {
		fmt.Fprintln(stderr, "ask: agent returned no answer")
		return 1
	}
	if cfg.verbose {
		printTrace(stderr, res)
	}
	if cfg.asJSON {
		printJSON(stdout, res)
		return 0
	}
	if !cfg.stream {
		fmt.Fprintln(stdout, strings.TrimRight(res.core.Text, "\n"))
	}
	printSummary(stdout, res)
	return 0
}

func main() {
	os.Exit(run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr))
}
