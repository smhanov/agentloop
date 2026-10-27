// Code mirroring the README's Go snippets: every code block in README.md
// is reproduced here inside a compile-only function (or, for the scripted
// provider pattern, a working test), so a documentation change that stops
// compiling fails `go test` like any other code change. Snippet bodies
// match the README verbatim; the enclosing functions, blank-identifier
// discards, and the scaffolding types at the bottom exist only to satisfy
// the compiler.
package agentloop_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/smhanov/llmhub"
	_ "github.com/smhanov/llmhub/providers/openai"

	"github.com/smhanov/agentloop"
	"github.com/smhanov/agentloop/bash"
	"github.com/smhanov/agentloop/search"
)

// README: Quick Start.
func snippetQuickStart() {
	llm, _ := llmhub.New("openai", os.Getenv("OPENAI_API_KEY"),
		llmhub.WithModel("gpt-4o-mini"))

	agent, _ := search.New(llm,
		search.WithSerperKey(os.Getenv("SERPER_API_KEY")))

	ctx := context.Background()
	ans, err := agent.Ask(ctx,
		"What Go version was released most recently, and when?")
	if err != nil {
		panic(err)
	}

	fmt.Println(ans.Text)
	for i, s := range ans.Sources {
		fmt.Printf("[%d] %s — %s\n", i+1, s.Title, s.URL)
	}
	fmt.Printf("%d tool calls, $%.4f\n", ans.ToolCalls, ans.Usage.Cost)
}

// README: Building an agent.
func snippetBuildingAnAgent(llm *llmhub.Client) {
	ag := agentloop.MustNew(llm,
		agentloop.WithSystemPrompt("You keep exact time."),
		agentloop.WithTools(clock),
	)

	ctx := context.Background()
	ans, err := ag.Ask(ctx, "What time is it in UTC right now?")
	_, _ = ans, err
}

// README: Writing tools (the clock).
func snippetClockTool() agentloop.Tool {
	clock := agentloop.Tool{
		Def: llmhub.NewTool("clock", "Current time in RFC 3339", map[string]any{
			"type": "object", "properties": map[string]any{},
		}),
		Run: func(ctx context.Context, args json.RawMessage) (string, error) {
			return time.Now().UTC().Format(time.RFC3339), nil
		},
	}
	return clock
}

// README: Writing tools (stateful, Sequential).
func snippetStatefulTool() agentloop.Tool {
	session := &shellSession{dir: "/tmp/workspace"} // your type: holds cwd, env, history

	sh := agentloop.Tool{
		Def: llmhub.NewTool("shell", "Run one command in the workspace", map[string]any{
			"type": "object",
			"properties": map[string]any{
				"command": map[string]any{"type": "string"},
			},
			"required": []string{"command"},
		}),
		Run: func(ctx context.Context, args json.RawMessage) (string, error) {
			return session.Run(ctx, string(args)) // reads and mutates session state
		},
		Sequential: true, // stateful: strict call order, never overlaps anything
	}
	return sh
}

// README: Hooks.
func snippetHooks(llm *llmhub.Client) {
	hooks := agentloop.Hooks{
		OnStart: func(ctx context.Context, ev *agentloop.Event) error {
			fmt.Println("run:", ev.Query)
			return nil
		},
		AfterToolCall: func(ctx context.Context, ev *agentloop.Event) error {
			fmt.Printf("%s -> %.80s (%v)\n", ev.Call.Name, ev.Result, ev.Duration)
			return nil
		},
	}
	ag := agentloop.MustNew(llm, agentloop.WithTools(clock), agentloop.WithHooks(hooks))
	_ = ag
}

// README: Streaming.
func snippetStreaming(agent *search.Agent) {
	ctx := context.Background()
	events, err := agent.AskStream(ctx, "What Go version was released most recently, and when?")
	if err != nil {
		log.Fatal(err)
	}
	for ev := range events {
		switch ev.Kind {
		case agentloop.EventAnswerDelta:
			fmt.Print(ev.Delta)
		case agentloop.EventDone:
			for i, s := range search.Sources(ev.Answer) {
				fmt.Printf("[%d] %s — %s\n", i+1, s.Title, s.URL)
			}
		case agentloop.EventError:
			log.Print(ev.Err)
		}
	}
}

// README: Conversations.
func snippetConversations(agent *agentloop.Agent) {
	conv := agent.Conversation(
		agentloop.WithSystemPrompt("You are a research assistant."),
		agentloop.WithGenerateOptions(llmhub.WithCacheKey("session-17")),
	)

	ctx := context.Background()
	first, err := conv.Ask(ctx, "What Go version was released most recently?")
	_, _ = first, err
	second, err := conv.Ask(ctx, "And when is the next one expected?")
	_, _ = second, err
}

// README: The search agent (custom backends — proves the interface shapes).
type cachingSearcher struct{}

func (cachingSearcher) Search(ctx context.Context, req search.SearchRequest) ([]search.Source, []string, error) {
	return nil, nil, errors.New("not implemented")
}

type cachingFetcher struct{}

func (cachingFetcher) Fetch(ctx context.Context, req search.FetchRequest) (*search.Page, error) {
	return nil, errors.New("not implemented")
}

var (
	_ search.Searcher = cachingSearcher{}
	_ search.Fetcher  = cachingFetcher{}
)

func snippetCustomBackends(llm *llmhub.Client) (*search.Agent, error) {
	return search.New(llm,
		search.WithSearcher(cachingSearcher{}),
		search.WithFetcher(cachingFetcher{}),
	)
}

// README: The bash agent.
func snippetBashAgent() {
	ctx := context.Background()
	llm, _ := llmhub.New("zai", os.Getenv("ZAI_API_KEY"),
		llmhub.WithBaseURL("https://api.z.ai/api/coding/paas/v4"),
		llmhub.WithModel("glm-5.3"))

	agent, _ := bash.New(llm,
		bash.WithEnvironment(bash.NewLocalEnvironment("/tmp/workspace", nil, 60*time.Second)),
		bash.WithStepLimit(20),
		bash.WithTokenLimit(2000000))

	ans, err := agent.Ask(ctx, "Add a README.md describing this project.")
	_, _ = ans, err
}

// README: Specializing the bash agent (coding-agent task-template override).
func snippetSpecializeBashAgent(llm *llmhub.Client) {
	const sweTask = "Please solve this issue: {{.Task}}\n\n" +
		"You can execute bash commands and edit files to implement the necessary changes.\n" +
		"Work step by step: analyze the codebase, reproduce the problem, fix it,\n" +
		"and verify with the project's tests before submitting.\n\n" +
		"Finish by issuing this command alone: `echo {{.SubmitMarker}}`.\n" +
		"On the lines after it, summarize the change.\n"

	agent, _ := bash.New(llm,
		bash.WithTemplates("", sweTask, ""), // system and observation keep the defaults
	)
	_ = agent
}

// README: Writing your own agent (text-action protocol via the finish veto).
func snippetFinishVeto(llm *llmhub.Client) {
	hooks := agentloop.Hooks{
		AfterGenerate: func(ctx context.Context, ev *agentloop.Event) error {
			action, ok := parseAction(ev.Response.Text()) // your format
			if !ok {
				ev.Messages = append(ev.Messages,
					llmhub.NewUserMessage(llmhub.Text("Respond with exactly one action block.")))
				ev.Finish = false // guidance appended; the model retries
				return nil
			}
			ev.Messages = append(ev.Messages,
				llmhub.NewUserMessage(llmhub.Text(runAndObserve(ctx, action))))
			ev.Finish = false // one more inference: the model reacts to the output
			return nil
		},
	}
	ag := agentloop.MustNew(llm, agentloop.WithSystemPrompt(protocolPrompt), agentloop.WithHooks(hooks))
	_ = ag
}

// README: Writing your own agent (steering via BeforeGenerate).
func snippetSteering(llm *llmhub.Client) {
	var inbox steeringQueue // your thread-safe queue of user messages

	hooks := agentloop.Hooks{
		BeforeGenerate: func(ctx context.Context, ev *agentloop.Event) error {
			for _, msg := range inbox.Take() {
				ev.Messages = append(ev.Messages, llmhub.NewUserMessage(llmhub.Text(msg)))
			}
			return nil
		},
	}
	ag := agentloop.MustNew(llm, agentloop.WithHooks(hooks))
	_ = ag
}

// README: Writing your own agent (cost caps via ev.Skip).
func snippetCostCap(llm *llmhub.Client) {
	var spent float64

	hooks := agentloop.Hooks{
		AfterGenerate: func(ctx context.Context, ev *agentloop.Event) error {
			if ev.Response != nil {
				spent += ev.Response.Usage.Cost
			}
			return nil
		},
		BeforeGenerate: func(ctx context.Context, ev *agentloop.Event) error {
			if spent >= 1.00 {
				ev.Skip = true // graceful stop before the next paid call
			}
			return nil
		},
	}
	ag := agentloop.MustNew(llm, agentloop.WithHooks(hooks))
	_ = ag
}

// README: Writing your own agent (context compaction via WithCompactor).
func snippetCompactor(llm *llmhub.Client) {
	ag := agentloop.MustNew(llm,
		agentloop.WithSystemPrompt("You are a research assistant."),
		agentloop.WithTools(clock),
		agentloop.WithCompactor(agentloop.NewTokenCompactor(128000, 15000, 1)),
	)
	_ = ag
}

// README: Writing your own agent (writing your own Compactor). The
// memoization guidance in the prose applies to real summarizers; this
// skeleton shows the trigger, the wholesale replacement, and the tail.
type summarizer struct {
	llm *llmhub.Client
}

func (s *summarizer) Compact(ctx context.Context, ev *agentloop.Event) error {
	if ev.Usage.PromptTokens < 100000 {
		return nil // under the trip point: the no-op is the common case
	}
	summary, err := s.llm.Generate(ctx, []*llmhub.Message{
		llmhub.NewUserMessage(llmhub.Text(
			"Summarize the conversation so far in a few sentences: " +
				messageText(ev.Messages))), // your prompt
	})
	if err != nil {
		return err
	}
	ev.Context = append([]*llmhub.Message{
		ev.Messages[0], // the system message, verbatim
		llmhub.NewUserMessage(llmhub.Text("Earlier steps, summarized: " + summary.Text())),
	}, ev.Messages[len(ev.Messages)-20:]...) // the recent tail, verbatim
	return nil
}

func snippetCustomCompactor(llm *llmhub.Client) {
	ag := agentloop.MustNew(llm, agentloop.WithCompactor(&summarizer{llm: llm}))
	_ = ag
}

// README: Testing your agent — the scripted provider, exercised for real.
type scriptedTurn struct {
	text  string
	calls []*llmhub.ToolCallContent
}

type scriptedProvider struct {
	mu    sync.Mutex
	turns []scriptedTurn
}

func (p *scriptedProvider) Name() string { return "scripted" }

func (p *scriptedProvider) Generate(ctx context.Context, prompt []*llmhub.Message, opts ...llmhub.Option) (*llmhub.Response, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.turns) == 0 {
		return nil, errors.New("scripted: script exhausted")
	}
	turn := p.turns[0]
	p.turns = p.turns[1:]
	resp := &llmhub.Response{}
	if turn.text != "" {
		resp.Content = append(resp.Content, llmhub.Text(turn.text))
	}
	for _, call := range turn.calls {
		resp.Content = append(resp.Content, call)
	}
	return resp, nil
}

func (p *scriptedProvider) Stream(ctx context.Context, prompt []*llmhub.Message, opts ...llmhub.Option) (<-chan llmhub.StreamChunk, error) {
	resp, err := p.Generate(ctx, prompt, opts...)
	ch := make(chan llmhub.StreamChunk, 1)
	if err == nil {
		ch <- llmhub.StreamChunk{Delta: resp.Text(), Done: true}
	}
	close(ch)
	return ch, err
}

var current *scriptedProvider

func init() {
	llmhub.MustRegisterProvider("scripted", func(apiKey string, opts ...llmhub.Option) (llmhub.Provider, error) {
		return current, nil
	})
}

func scriptAgent(t *testing.T, turns ...scriptedTurn) *agentloop.Agent {
	t.Helper()
	current = &scriptedProvider{turns: turns}
	llm, err := llmhub.New("scripted", "test-key")
	if err != nil {
		t.Fatalf("llmhub.New: %v", err)
	}
	return agentloop.MustNew(llm, agentloop.WithTools(clock))
}

func TestAgentCallsToolThenAnswers(t *testing.T) {
	ag := scriptAgent(t,
		scriptedTurn{calls: []*llmhub.ToolCallContent{
			{ID: "call_1", Name: "clock", Arguments: `{}`},
		}},
		scriptedTurn{text: "It is 12:00:00Z."},
	)

	ans, err := ag.Ask(context.Background(), "What time is it?")
	if err != nil {
		t.Fatal(err)
	}
	if ans.ToolCalls != 1 || ans.Steps != 2 {
		t.Fatalf("trace: %+v", ans)
	}
}

// --- Scaffolding the README snippets refer to ("your …" types). -------

// clock is the README's clock tool (snippetClockTool shows its literal).
var clock = agentloop.Tool{
	Def: llmhub.NewTool("clock", "Current time in RFC 3339", map[string]any{
		"type": "object", "properties": map[string]any{},
	}),
	Run: func(ctx context.Context, args json.RawMessage) (string, error) {
		return time.Now().UTC().Format(time.RFC3339), nil
	},
}

// shellSession is the stateful tool's backing type.
type shellSession struct{ dir string }

func (s *shellSession) Run(ctx context.Context, command string) (string, error) {
	return "ran in " + s.dir + ": " + command, nil
}

// protocolPrompt, parseAction, and runAndObserve back the finish-veto
// recipe.
const protocolPrompt = "Respond with exactly one action block."

func parseAction(text string) (string, bool) {
	return text, text != ""
}

func runAndObserve(ctx context.Context, action string) string {
	return "observed: " + action
}

// steeringQueue backs the steering recipe.
type steeringQueue struct {
	mu   sync.Mutex
	msgs []string
}

func (q *steeringQueue) Add(msg string) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.msgs = append(q.msgs, msg)
}

func (q *steeringQueue) Take() []string {
	q.mu.Lock()
	defer q.mu.Unlock()
	msgs := q.msgs
	q.msgs = nil
	return msgs
}

// messageText backs the custom-Compactor recipe's prompt text.
func messageText(msgs []*llmhub.Message) string {
	var b strings.Builder
	for _, m := range msgs {
		for _, part := range m.Content {
			if tc, ok := part.(*llmhub.TextContent); ok {
				b.WriteString(tc.Text)
			}
		}
	}
	return b.String()
}
