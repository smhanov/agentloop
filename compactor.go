package agentloop

import (
	"context"
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/smhanov/llmhub"
)

// Compactor bounds context growth. Compact runs before each model call,
// after BeforeGenerate appends and after the RebuildContext hook;
// ev.Context is pre-seeded to the current LLM view and ev.Usage carries
// the previous step's provider usage (zero on the first step). A compactor
// may replace ev.Context. Implementations must be deterministic (same
// input view and state → same output, or prompt caching busts), must
// preserve the trailing assistant tool-call message and all of its tool
// results, and must treat the replacement as ephemeral (the transcript,
// Answer.Messages, and Conversation history stay complete).
type Compactor interface {
	Compact(ctx context.Context, ev *Event) error
}

// runCompactor is implemented by compactors that keep memoized per-run
// state (the TokenCompactor's frozen checkpoint). The loop asks for a
// fresh copy at the start of each run so concurrent runs never share one;
// compactors without per-run state are used as installed.
type runCompactor interface {
	forRun() Compactor
}

const (
	// defaultCompactTail is the default verbatim-tail budget in estimated
	// tokens (opencode's 15k tail).
	defaultCompactTail = 15000
	// minCompactReserve is the minimum headroom the compactor keeps under
	// the window; the trigger is window − max(15% of window, this floor) —
	// the opencode/Codex convergence.
	minCompactReserve = 16000
	// compactReservePct is the percentage of the window reserved on top of
	// the floor.
	compactReservePct = 15
	// maxCheckpointLines bounds how many elided user messages' first lines
	// the checkpoint stub lists; the rest are counted. A stub that grows
	// with the transcript would defeat the point of compacting it.
	maxCheckpointLines = 20
	// maxFirstLineChars caps one stub first line, cut on a UTF-8 boundary.
	maxFirstLineChars = 120
)

// compactState is the TokenCompactor's memoized projection state. Between
// trips the projection is a pure function of (transcript, state): the
// checkpoint and the tail boundary stay frozen while the transcript only
// appends, so the projected prefix is byte-stable step-over-step and the
// prompt-cache breakpoint survives.
type compactState struct {
	tripped    bool            // a trip has happened; keep projecting
	tailStart  int             // transcript index where the verbatim tail begins
	checkpoint *llmhub.Message // frozen checkpoint stub; nil = prune-only projection
}

// TokenCompactor is the shipped reference Compactor: a deterministic,
// token-triggered projection that keeps long runs inside the model's
// context window without any LLM calls. Construct it with
// NewTokenCompactor and install it with WithCompactor. Its memoized
// checkpoint state is per run — the loop clones it at the start of each
// run — so a single TokenCompactor may be shared by any number of
// concurrent runs and Agents. (A custom Compactor that delegates to an
// embedded TokenCompactor bypasses that cloning: its runs would share one
// memo.)
type TokenCompactor struct {
	windowTokens int64 // the model's usable context window
	tailTokens   int64 // verbatim-tail budget in estimated tokens; ≤0 = default
	keepFirst    int   // messages kept after the system message; ≤0 = default
	st           compactState
}

// NewTokenCompactor returns the reference Compactor configured for a
// model. windowTokens is the model's usable context window; the compactor
// trips when the previous step's measured PromptTokens reach window −
// max(15% of window, 16000) and never before (a zero usage — a run's
// first step — is a no-op). tailTokens is the verbatim-recent-tail budget
// in estimated tokens (characters/4; 0 or less selects 15000). keepFirst
// is how many messages after the system message stay verbatim at the head
// of every projection (0 or less selects 1 — the task message). A
// non-positive windowTokens disables the compactor: it never trips.
//
// The character/4 estimate is deliberately crude and is used only to size
// the tail and to decide between pruning and a checkpoint stub; the
// trigger itself is the exact provider-reported number.
func NewTokenCompactor(windowTokens, tailTokens int64, keepFirst int) *TokenCompactor {
	return &TokenCompactor{windowTokens: windowTokens, tailTokens: tailTokens, keepFirst: keepFirst}
}

// forRun returns a copy with fresh state, implementing the runCompactor
// seam: the memoized checkpoint is per run, so concurrent runs never share
// one.
func (t *TokenCompactor) forRun() Compactor {
	clone := *t
	clone.st = compactState{}
	return &clone
}

// Compact implements Compactor. Until the trigger trips it leaves
// ev.Context untouched — the no-op is the common case and costs nothing.
// Once tripped it replaces ev.Context with the projection and keeps
// projecting, with the same frozen checkpoint and tail boundary: the
// transcript only appends, so the projection grows at its end and the
// prefix stays byte-stable. When the measured size of the view trips
// again, the state is recomputed — the elided range grows, the tail
// budget stays, and the checkpoint is refrozen (one cache bust per trip).
func (t *TokenCompactor) Compact(ctx context.Context, ev *Event) error {
	if t.windowTokens <= 0 {
		return nil // disabled: never trips
	}
	if ev.Usage.PromptTokens > 0 && int64(ev.Usage.PromptTokens) >= t.threshold() {
		t.st = t.trip(ev.Messages)
	}
	if t.st.tripped {
		ev.Context = t.project(ev.Messages, t.st)
	}
	return nil
}

// threshold returns the PromptTokens count that trips the compactor: the
// window minus the reserve — the larger of 15% of the window and 16k
// tokens.
func (t *TokenCompactor) threshold() int64 {
	reserve := t.windowTokens * compactReservePct / 100
	if reserve < minCompactReserve {
		reserve = minCompactReserve
	}
	return t.windowTokens - reserve
}

// tailBudget returns the verbatim-tail budget in estimated tokens.
func (t *TokenCompactor) tailBudget() int64 {
	if t.tailTokens > 0 {
		return t.tailTokens
	}
	return defaultCompactTail
}

// keepEnd returns the number of transcript messages kept verbatim at the
// head of every projection: the system message (when present) plus
// keepFirst messages, extended forward when the cut would land inside a
// trailing tool-call group — an assistant tool-call message never travels
// without all of its results.
func (t *TokenCompactor) keepEnd(transcript []*llmhub.Message) int {
	keep := t.keepFirst
	if keep < 1 {
		keep = 1
	}
	if len(transcript) > 0 && transcript[0].Role == llmhub.RoleSystem {
		keep++ // the system message stays at the head, verbatim, always
	}
	if keep > len(transcript) {
		keep = len(transcript)
	}
	for keep < len(transcript) && transcript[keep].Role == llmhub.RoleTool {
		keep++
	}
	return keep
}

// tailBoundary returns the transcript index where the verbatim tail
// starts: roughly the tail budget's worth of estimated tokens walked back
// from the end, with the boundary moved back so it never lands inside a
// tool-call group (an orphaned tool result — its caller elided — would be
// rejected by providers). The trailing assistant tool-call message plus
// all of its results, parallel calls included, always travel together.
func (t *TokenCompactor) tailBoundary(transcript []*llmhub.Message, keepEnd int) int {
	end := len(transcript)
	// Start the walk after the full trailing group so it is always inside
	// the tail.
	i := end
	for i > keepEnd && transcript[i-1].Role == llmhub.RoleTool {
		i--
	}
	acc := estimateTokens(transcript[i:end])
	for i > keepEnd && acc < t.tailBudget() {
		i--
		acc += estimateTokens(transcript[i : i+1])
	}
	// The walk may have landed on a tool result of an earlier group: move
	// back to the assistant message that issued the results.
	for i > keepEnd && transcript[i].Role == llmhub.RoleTool {
		i--
	}
	return i
}

// trip computes the projection state for a transcript that just crossed
// the threshold: a tail boundary at the tail budget, then a prune-only
// projection when that fits under the threshold and a checkpoint stub
// when it does not. Pure function of the transcript.
func (t *TokenCompactor) trip(transcript []*llmhub.Message) compactState {
	keepEnd := t.keepEnd(transcript)
	tailStart := t.tailBoundary(transcript, keepEnd)
	st := compactState{tripped: true, tailStart: tailStart}
	if tailStart <= keepEnd {
		return st // nothing elided; the full view is the best we can do
	}
	if estimateTokens(t.project(transcript, st)) > t.threshold() {
		st.checkpoint = t.checkpointStub(transcript, keepEnd, tailStart)
	}
	return st
}

// project returns the LLM view for transcript under st: the keep-set
// verbatim, the elided middle — tool outputs replaced by short notices,
// or the whole middle replaced by the frozen checkpoint stub — and the
// verbatim tail. Pure function of (transcript, st). The transcript is
// never modified: the projection shares its message pointers except for
// the synthesized elision notices and the stub.
func (t *TokenCompactor) project(transcript []*llmhub.Message, st compactState) []*llmhub.Message {
	keepEnd := t.keepEnd(transcript)
	tailStart := st.tailStart
	if tailStart < keepEnd {
		tailStart = keepEnd
	}
	if tailStart >= len(transcript) {
		return transcript // nothing elided; the view is the transcript
	}
	view := make([]*llmhub.Message, 0, len(transcript)-(tailStart-keepEnd)+1)
	view = append(view, transcript[:keepEnd]...)
	if st.checkpoint != nil {
		view = append(view, st.checkpoint)
	} else {
		for _, m := range transcript[keepEnd:tailStart] {
			if m.Role == llmhub.RoleTool {
				view = append(view, elidedToolResult(m))
				continue
			}
			view = append(view, m)
		}
	}
	return append(view, transcript[tailStart:]...)
}

// checkpointStub builds the frozen user message that replaces the elided
// middle when pruning alone does not fit: role counts, the size of the
// elided range, and the first line of each elided user message (bounded —
// see maxCheckpointLines). The text is a pure function of the transcript
// range with no timestamps, so replays are byte-identical and the frozen
// stub keeps the projected prefix cache-stable between trips.
func (t *TokenCompactor) checkpointStub(transcript []*llmhub.Message, keepEnd, tailStart int) *llmhub.Message {
	elided := transcript[keepEnd:tailStart]
	var (
		assistant, system, tool, user int
		firstLines                    []string
	)
	for _, m := range elided {
		switch m.Role {
		case llmhub.RoleAssistant:
			assistant++
		case llmhub.RoleSystem:
			system++
		case llmhub.RoleTool:
			tool++
		case llmhub.RoleUser:
			user++
			if line := firstLine(m); line != "" && len(firstLines) < maxCheckpointLines {
				firstLines = append(firstLines, line)
			}
		}
	}
	var b strings.Builder
	fmt.Fprintf(&b, "[context checkpoint] The %d messages before this point were elided from this view to fit the context window", len(elided))
	var counts []string
	if assistant > 0 {
		counts = append(counts, fmt.Sprintf("%d assistant", assistant))
	}
	if system > 0 {
		counts = append(counts, fmt.Sprintf("%d system", system))
	}
	if tool > 0 {
		counts = append(counts, fmt.Sprintf("%d tool results", tool))
	}
	if user > 0 {
		counts = append(counts, fmt.Sprintf("%d user", user))
	}
	if len(counts) > 0 {
		b.WriteString(" (")
		b.WriteString(strings.Join(counts, ", "))
		b.WriteString(")")
	}
	b.WriteString(". Their content remains in the full transcript.")
	if len(firstLines) > 0 {
		b.WriteString(" The elided user messages began with:")
		for _, line := range firstLines {
			b.WriteString("\n- ")
			b.WriteString(line)
		}
		if more := user - len(firstLines); more > 0 {
			fmt.Fprintf(&b, "\n(and %d more user messages elided)", more)
		}
	}
	return llmhub.NewUserMessage(llmhub.Text(b.String()))
}

// elidedToolResult returns a copy of a tool result message whose content
// is replaced by a short elision notice. Role and metadata are preserved
// so the surrounding tool-call protocol stays provider-valid; the
// original message is never modified.
func elidedToolResult(m *llmhub.Message) *llmhub.Message {
	notice := fmt.Sprintf("[tool output elided by compaction: %d chars]", contentChars(m))
	return &llmhub.Message{Role: m.Role, Content: []llmhub.ContentPart{llmhub.Text(notice)}, Meta: m.Meta}
}

// estimateTokens estimates the token count of a message list as content
// characters divided by four. The compactor's trigger is the
// provider-reported PromptTokens of the previous step (exact); this
// estimate only sizes the tail budget and decides between pruning and a
// checkpoint stub.
func estimateTokens(msgs []*llmhub.Message) int64 {
	var chars int64
	for _, m := range msgs {
		chars += int64(len(m.Role)) + int64(contentChars(m))
	}
	return chars / 4
}

// contentChars returns the character count of a message's content parts.
func contentChars(m *llmhub.Message) int {
	n := 0
	for _, part := range m.Content {
		switch p := part.(type) {
		case *llmhub.TextContent:
			n += len(p.Text)
		case *llmhub.ReasoningContent:
			n += len(p.Text)
		case *llmhub.ToolCallContent:
			n += len(p.ID) + len(p.Name) + len(p.Arguments)
		case *llmhub.ImageContent:
			n += len(p.URL)
		}
	}
	return n
}

// firstLine returns the first line of a message's text, capped on a UTF-8
// boundary (same shape as the duplicate-call warning's excerpt cap).
func firstLine(m *llmhub.Message) string {
	text := messageText(m)
	if i := strings.IndexByte(text, '\n'); i >= 0 {
		text = text[:i]
	}
	if len(text) > maxFirstLineChars {
		cut := maxFirstLineChars
		for cut > 0 && !utf8.RuneStart(text[cut]) {
			cut--
		}
		text = text[:cut]
	}
	return text
}

// messageText returns the concatenated text parts of a message.
func messageText(m *llmhub.Message) string {
	var b strings.Builder
	for _, part := range m.Content {
		if tc, ok := part.(*llmhub.TextContent); ok {
			b.WriteString(tc.Text)
		}
	}
	return b.String()
}
