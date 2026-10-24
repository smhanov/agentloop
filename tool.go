package agentloop

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/smhanov/llmhub"
)

// Handler executes a tool invocation. args is the raw JSON arguments string
// from the model (llmhub ToolCallContent.Arguments). The return value is
// inserted verbatim as the tool result message content; a returned error is
// delivered to the model as an {"error": ...} observation and the loop
// continues, except for context errors (cancellation or deadline), which
// abort the run.
type Handler func(ctx context.Context, args json.RawMessage) (string, error)

// Tool is one model-callable tool: its llmhub definition plus the Go
// handler that executes it.
type Tool struct {
	// Def is the schema shown to the model: name, description, and JSON
	// Schema parameters.
	Def llmhub.Tool
	// Run executes the tool.
	Run Handler

	// Sequential forces strict call-order execution that never overlaps
	// with anything (the inverse of dsh's isConcurrencySafe, chosen so the
	// Go zero value means the common default). Tools with the zero value
	// are eligible for parallel batching bounded by WithParallelTools; set
	// Sequential to true for stateful tools (shell sessions,
	// transactions).
	Sequential bool
}

// Observation payloads the loop itself delivers to the model.
const (
	// skippedResult is the tool result message content for calls vetoed by
	// a BeforeToolCall hook via ErrSkipToolCall.
	skippedResult = "skipped"
	// warningObservation replaces the output of the tool call that reaches
	// the duplicate-call limit.
	warningObservation = `{"warning": "repeated identical call; change approach"}`
	// stopObservation is delivered for one repeat too many; the run then
	// stops gracefully with StatusStopped.
	stopObservation = `{"warning": "repeated identical call; stopping"}`
	// budgetObservation is delivered for calls denied because the
	// MaxToolCalls budget was spent.
	budgetObservation = `{"error": "tool call budget exhausted"}`
)

// errorJSON renders a failure as the in-band error observation the model
// sees; the loop itself continues.
func errorJSON(err error) string {
	b, _ := json.Marshal(map[string]string{"error": err.Error()})
	return string(b)
}

// unknownToolJSON renders the observation for a call naming an
// unregistered tool, including a Levenshtein suggestion when one is close
// enough.
func unknownToolJSON(suggestion string) string {
	msg := "unknown tool"
	if suggestion != "" {
		msg = fmt.Sprintf("unknown tool; did you mean %q?", suggestion)
	}
	b, _ := json.Marshal(map[string]string{"error": msg})
	return string(b)
}

// levenshtein returns the edit distance between a and b (insertions,
// deletions, substitutions).
func levenshtein(a, b string) int {
	ar, br := []rune(a), []rune(b)
	prev := make([]int, len(br)+1)
	curr := make([]int, len(br)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(ar); i++ {
		curr[0] = i
		for j := 1; j <= len(br); j++ {
			cost := 1
			if ar[i-1] == br[j-1] {
				cost = 0
			}
			curr[j] = min(curr[j-1]+1, prev[j]+1, prev[j-1]+cost)
		}
		prev, curr = curr, prev
	}
	return prev[len(br)]
}

// suggestName returns the registered name closest to name when it is within
// edit distance 2 (Hermes' hallucinated-name repair), or "" when nothing is
// close enough. Ties go to the first candidate with the smallest distance.
func suggestName(name string, registered []string) string {
	best := ""
	bestDist := 3
	for _, candidate := range registered {
		if d := levenshtein(name, candidate); d < bestDist {
			best, bestDist = candidate, d
		}
	}
	return best
}

// lookup finds the registered tool with the given definition name.
func (r *run) lookup(name string) (Tool, bool) {
	for _, t := range r.cfg.tools {
		if t.Def.Name == name {
			return t, true
		}
	}
	return Tool{}, false
}

// registeredNames lists the tool names known to the run, in registration
// order.
func (r *run) registeredNames() []string {
	names := make([]string, 0, len(r.cfg.tools))
	for _, t := range r.cfg.tools {
		names = append(names, t.Def.Name)
	}
	return names
}

// planMode classifies how a planned tool call will be processed.
type planMode int

const (
	planExecuteParallel   planMode = iota // batch concurrently, bounded
	planExecuteSequential                 // run alone, strict call order
	planResolved                          // observation pre-decided, no execution
	planStop                              // duplicate limit exceeded: stop run
)

// callPlan is the phase-A resolution of one model-requested tool call.
type callPlan struct {
	mode   planMode
	call   *llmhub.ToolCallContent
	args   json.RawMessage
	tool   Tool
	warn   bool        // deliver the duplicate-call warning instead of the output
	result *stepResult // pre-filled for planResolved and planStop
}

// stepResult is the outcome of processing one planned call: the observation
// delivered to the model and, when the call did real work, its trace entry.
type stepResult struct {
	observation string
	trace       *ToolResult
}

// dispatchAll resolves and executes one step's tool calls. Resolution
// (hooks, duplicate detection, budget, unknown names) happens first, in
// call order; execution then batches parallel-eligible calls concurrently
// (bounded by WithParallelTools) while Sequential tools run strictly in
// call order and never overlap anything. Tool result messages always
// append in original call order, while AfterToolCall fires in completion
// order. It reports stopped=true when the duplicate-call limit was
// exceeded and the run should finish gracefully.
func (r *run) dispatchAll(ctx context.Context, step int, calls []*llmhub.ToolCallContent) (stopped bool, err error) {
	// Phase A: resolve every call in call order.
	plans := make([]callPlan, 0, len(calls))
	for _, call := range calls {
		plan, err := r.planCall(ctx, step, call)
		if err != nil {
			return false, err
		}
		plans = append(plans, plan)
		if plan.mode == planStop {
			stopped = true
			break // one repeat too many: trailing calls are ignored
		}
	}

	// Phase B: execute. results[k] is written only by the goroutine that
	// processes plan k and read only after all of them synchronize, so the
	// distinct slots need no mutex.
	results := make([]*stepResult, len(plans))
	var (
		mu       sync.Mutex
		firstErr error
	)
	wctx, cancel := context.WithCancel(ctx)
	defer cancel()
	fail := func(err error) {
		mu.Lock()
		defer mu.Unlock()
		if firstErr == nil {
			firstErr = err
			cancel() // stop sibling executions that honor ctx
		}
	}

	for i := 0; i < len(plans); {
		switch plan := plans[i]; plan.mode {
		case planStop, planResolved:
			results[i] = plan.result
			if plan.result.trace != nil {
				// Unknown-name resolution: observers see it like an
				// execution.
				ev := &Event{Kind: EventAfterToolCall, Step: step, Call: plan.call, Args: plan.args, Result: plan.result.observation}
				if err := fire(ctx, r.cfg.hooks.AfterToolCall, ev); err != nil {
					return false, fmt.Errorf("agentloop: AfterToolCall hook: %w", err)
				}
				r.emit(ev)
			}
			i++
		case planExecuteSequential:
			res, err := r.executePlan(wctx, step, plan)
			if err != nil {
				return false, err
			}
			results[i] = res
			i++
		case planExecuteParallel:
			// Batch the maximal run of consecutive parallel plans.
			j := i
			for j < len(plans) && plans[j].mode == planExecuteParallel {
				j++
			}
			bound := r.cfg.effectiveParallelTools()
			if bound <= 1 || j == i+1 {
				for k := i; k < j; k++ {
					res, err := r.executePlan(wctx, step, plans[k])
					if err != nil {
						return false, err
					}
					results[k] = res
				}
				i = j
				break
			}
			var wg sync.WaitGroup
			sem := make(chan struct{}, bound)
			for k := i; k < j; k++ {
				wg.Add(1)
				go func(k int) {
					defer wg.Done()
					select {
					case sem <- struct{}{}:
						defer func() { <-sem }()
					case <-wctx.Done():
						return
					}
					if wctx.Err() != nil {
						return
					}
					res, err := r.executePlan(wctx, step, plans[k])
					if err != nil {
						fail(err)
						return
					}
					results[k] = res
				}(k)
			}
			wg.Wait()
			if firstErr != nil {
				return false, firstErr
			}
			if err := wctx.Err(); err != nil {
				// Outer cancellation left part of the batch unexecuted.
				return false, err
			}
			i = j
		}
	}

	// Append results in original call order regardless of completion order.
	for k, plan := range plans {
		res := results[k]
		if res == nil {
			continue // aborted before execution; the run is failing anyway
		}
		r.appendToolResult(plan.call, res.observation)
		if res.trace != nil {
			r.trace = append(r.trace, *res.trace)
		}
	}
	return stopped, nil
}

// planCall resolves one model-requested tool call: it fires BeforeToolCall
// (veto/rewrite), applies duplicate-call detection and the tool budget, and
// classifies the call. Budget accounting (r.executed) is incremented here,
// in call order, so several calls in one response cannot overspend.
func (r *run) planCall(ctx context.Context, step int, call *llmhub.ToolCallContent) (callPlan, error) {
	ev := &Event{Kind: EventBeforeToolCall, Step: step, Call: call, Args: json.RawMessage(call.Arguments)}
	if err := fire(ctx, r.cfg.hooks.BeforeToolCall, ev); err != nil {
		if errors.Is(err, ErrSkipToolCall) {
			return callPlan{mode: planResolved, call: call, result: &stepResult{observation: skippedResult}}, nil
		}
		return callPlan{}, fmt.Errorf("agentloop: BeforeToolCall hook: %w", err)
	}
	plan := callPlan{call: call}
	plan.args = ev.Args
	if len(plan.args) == 0 {
		plan.args = json.RawMessage(call.Arguments)
	}
	r.emit(&Event{Kind: EventBeforeToolCall, Step: step, Call: call, Args: plan.args})

	// Loop detection: identical consecutive (name, args) calls warn at the
	// limit and stop the run on the next repeat. Any differing call resets
	// the run.
	if r.cfg.dupCallLimit > 0 {
		key := call.Name + "\x00" + string(plan.args)
		if key == r.dupKey {
			r.dupCount++
		} else {
			r.dupKey, r.dupCount = key, 1
		}
		switch {
		case r.dupCount > r.cfg.dupCallLimit:
			plan.mode = planStop
			plan.result = &stepResult{observation: stopObservation}
			return plan, nil
		case r.dupCount == r.cfg.dupCallLimit:
			plan.warn = true
		}
	}

	// Budget: one response may carry more calls than the budget has left.
	if r.executed >= r.cfg.effectiveMaxToolCalls() {
		plan.mode = planResolved
		plan.result = &stepResult{observation: budgetObservation}
		return plan, nil
	}

	tool, found := r.lookup(call.Name)
	if !found {
		observation := unknownToolJSON(suggestName(call.Name, r.registeredNames()))
		r.executed++ // unknown names count against the budget
		plan.mode = planResolved
		plan.result = &stepResult{
			observation: observation,
			trace:       &ToolResult{Name: call.Name, ID: call.ID, Args: plan.args, Output: observation},
		}
		return plan, nil
	}
	r.executed++
	plan.tool = tool
	if tool.Sequential {
		plan.mode = planExecuteSequential
	} else {
		plan.mode = planExecuteParallel
	}
	return plan, nil
}

// executePlan runs one planned call's handler, renders the observation
// (in-band error or duplicate warning as needed), records the trace entry,
// and fires AfterToolCall in completion order. Returned errors abort the
// run: handler context errors and AfterToolCall hook errors.
func (r *run) executePlan(ctx context.Context, step int, plan callPlan) (*stepResult, error) {
	start := time.Now()
	output, handlerErr := plan.tool.Run(ctx, plan.args)
	duration := time.Since(start)

	observation := output
	switch {
	case handlerErr != nil:
		if errors.Is(handlerErr, context.Canceled) || errors.Is(handlerErr, context.DeadlineExceeded) {
			// Context errors abort the run; every other handler error is
			// repaired in-band.
			return nil, handlerErr
		}
		observation = errorJSON(handlerErr)
	case plan.warn:
		observation = warningObservation
	}

	res := &stepResult{
		observation: observation,
		trace:       &ToolResult{Name: plan.call.Name, ID: plan.call.ID, Args: plan.args, Output: observation, Err: handlerErr, Duration: duration},
	}
	after := &Event{Kind: EventAfterToolCall, Step: step, Call: plan.call, Args: plan.args, Result: observation, Err: handlerErr, Duration: duration}
	if err := fire(ctx, r.cfg.hooks.AfterToolCall, after); err != nil {
		return res, fmt.Errorf("agentloop: AfterToolCall hook: %w", err)
	}
	r.emit(after)
	return res, nil
}

// appendToolResult appends the tool result message for call, keeping the
// transcript answerable for every tool call the model made.
func (r *run) appendToolResult(call *llmhub.ToolCallContent, observation string) {
	r.transcript = append(r.transcript, llmhub.NewToolResultMessage(call.ID, call.Name, llmhub.Text(observation)))
}
