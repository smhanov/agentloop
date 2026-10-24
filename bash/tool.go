package bash

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/smhanov/llmhub"

	"github.com/smhanov/agentloop"
)

// bashToolName is the native tool's registered name in tool-protocol mode.
const bashToolName = "bash"

// bashToolArgs is the bash tool's argument shape: {"command": string}.
type bashToolArgs struct {
	Command string `json:"command"`
}

// bashToolDef declares the bash tool (the tool protocol's transport): one
// shell command per call, executed in the preset's Environment with the
// same observation formatting and submit detection as the text protocol.
// The submit marker is baked into the description so the model knows the
// completion convention even though the tool outlives the task prompt.
func bashToolDef(submitMarker string) llmhub.Tool {
	return llmhub.NewTool(bashToolName,
		"Run one bash command in the task's working directory and receive its "+
			"combined output with the exit status. Non-zero exit codes are "+
			"observations, not failures: read the output and adjust. Directory and "+
			"environment-variable changes do not persist between calls — prefix "+
			"them onto the command (cd dir && …) when needed. To finish the task, "+
			"end a successful command's output with "+submitMarker+" as the first "+
			"line; the run then stops with your submission.",
		map[string]any{
			"type": "object",
			"properties": map[string]any{
				"command": map[string]any{
					"type": "string",
					"description": "the bash command to run, as it would be " +
						"passed to bash -c",
				},
			},
			"required": []string{"command"},
		})
}

// handleToolCall is the bash tool's handler — the tool protocol's half of
// the shared execution path (runCommand: approval gate, Environment
// execution, observation formatting, submit detection). The observation is
// delivered as the tool result. Argument decode failures are returned
// rather than executed: the core delivers them in-band as {"error": …}
// observations and the loop continues, playing the role the text protocol's
// format-error guidance plays. A command whose output announces the submit
// marker ends the run before the next inference (beforeGenerate).
func (p *protocol) handleToolCall(ctx context.Context, args json.RawMessage) (string, error) {
	var req bashToolArgs
	if err := json.Unmarshal(args, &req); err != nil {
		return "", fmt.Errorf("bash: decode tool arguments: %w", err)
	}
	if strings.TrimSpace(req.Command) == "" {
		return "", fmt.Errorf("bash: empty command")
	}
	out, err := p.runCommand(ctx, req.Command)
	if err != nil {
		return "", err
	}
	if out.submitted {
		p.st.exitStatus = ExitSubmitted
		p.st.submission = out.submission
	}
	return out.observation, nil
}

// countStep is the tool protocol's AfterGenerate hook: it only keeps the
// limits' counters current (model inferences and provider-reported cost).
// Finish arrives pre-set (a response without tool calls ends the run) and
// is left untouched — the loop's natural termination replaces the text
// protocol's finish veto, and no message is appended here (the bash tool's
// result already carries the observation).
func (p *protocol) countStep(ctx context.Context, ev *agentloop.Event) error {
	p.st.steps++
	if resp := ev.Response; resp != nil {
		p.st.cost += resp.Usage.Cost
	}
	return nil
}
