package agentloop

import (
	"context"
	"fmt"
	"sync"

	"github.com/smhanov/llmhub"
)

// Conversation runs a sequence of Asks over a shared message history:
// follow-up turns see prior turns' user messages, tool calls, results, and
// answers. The history starts with the system message (when configured) and
// each turn's own messages append to it, so the transcript is append-only —
// the loop never edits, reorders, or rewrites prior messages — which
// preserves provider prompt caching across turns. Answer.Messages carries
// only that run's messages (the conversation prefix excluded). A
// Conversation is not safe for concurrent use: an Ask or AskStream started
// while another turn is in flight fails fast with ErrConversationBusy.
type Conversation struct {
	agent   *Agent
	base    config // conversation defaults; the system prompt is frozen here
	history []*llmhub.Message
	usage   llmhub.UsageMetadata
	mu      sync.Mutex
}

// Conversation starts a multi-turn session. The opts set the system message
// (prepended exactly once and frozen for the session's lifetime) and the
// default per-turn options; opts passed to Ask or AskStream apply only to
// that turn, and a per-turn WithSystemPrompt is ignored in favor of the
// frozen one.
func (a *Agent) Conversation(opts ...Option) *Conversation {
	cfg := a.cfg
	applyOptions(&cfg, opts)
	conv := &Conversation{agent: a, base: cfg}
	if cfg.systemPrompt != "" {
		conv.history = []*llmhub.Message{llmhub.NewSystemMessage(llmhub.Text(cfg.systemPrompt))}
	}
	return conv
}

// Ask runs one turn of the conversation to completion. Budgets (tool calls,
// steps, duplicate detection) apply per turn. When the run produces an
// Answer — even a budget-exhausted partial one — its messages join the
// history; aborted runs (transport, hook, or context errors) leave the
// history untouched.
func (c *Conversation) Ask(ctx context.Context, query string, opts ...Option) (*Answer, error) {
	if !c.mu.TryLock() {
		return nil, fmt.Errorf("agentloop: %w", ErrConversationBusy)
	}
	defer c.mu.Unlock()

	ans, err := c.agent.ask(ctx, query, c.turnConfig(opts), c.history)
	if ans != nil {
		c.remember(ans)
	}
	return ans, err
}

// AskStream is Conversation.Ask with streaming (see Agent.AskStream): all
// loop events are forwarded on the returned channel, which closes after
// Done or Error. The turn's messages join the history when the run
// finishes, just like Ask. The channel must be drained until close.
func (c *Conversation) AskStream(ctx context.Context, query string, opts ...Option) (<-chan Event, error) {
	if !c.mu.TryLock() {
		return nil, fmt.Errorf("agentloop: %w", ErrConversationBusy)
	}
	src, err := c.agent.askStream(ctx, query, c.turnConfig(opts), c.history)
	if err != nil {
		c.mu.Unlock()
		return nil, err
	}
	out := make(chan Event, eventBuffer)
	go func() {
		defer close(out)
		defer c.mu.Unlock()
		for ev := range src {
			if ev.Answer != nil && (ev.Kind == EventDone || ev.Kind == EventError) {
				c.remember(ev.Answer)
			}
			out <- ev
		}
	}()
	return out, nil
}

// Messages returns the full transcript so far: the system message (when
// configured) plus one entry per run message across turns. The caller must
// not mutate the returned messages.
func (c *Conversation) Messages() []*llmhub.Message {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]*llmhub.Message(nil), c.history...)
}

// TotalUsage returns the usage summed across all turns since construction
// or the last Reset.
func (c *Conversation) TotalUsage() llmhub.UsageMetadata {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.usage
}

// Reset drops the turn history and usage totals. The system message is
// kept and still opens the next turn.
func (c *Conversation) Reset() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.history = nil
	if c.base.systemPrompt != "" {
		c.history = []*llmhub.Message{llmhub.NewSystemMessage(llmhub.Text(c.base.systemPrompt))}
	}
	c.usage = llmhub.UsageMetadata{}
}

// remember appends a finished run's own messages to the history and folds
// the turn's usage into the total. Run messages exclude the conversation
// prefix, so the transcript only ever grows.
func (c *Conversation) remember(ans *Answer) {
	c.history = append(c.history, ans.Messages...)
	c.usage = addUsage(c.usage, ans.Usage)
}

// turnConfig merges the frozen conversation defaults with per-turn
// overrides, keeping the construction-time system prompt.
func (c *Conversation) turnConfig(opts []Option) config {
	cfg := c.base
	applyOptions(&cfg, opts)
	cfg.systemPrompt = c.base.systemPrompt
	return cfg
}
