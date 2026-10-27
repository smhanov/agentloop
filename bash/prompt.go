package bash

import (
	"fmt"
	"os"
	"runtime"
	"strings"
	"text/template"
)

// fence is the markdown code-fence marker used in the templates and the
// action parser. Declared as a constant because Go raw string literals
// cannot contain backticks.
const fence = "```"

// defaultSystemTemplate is the system prompt: exactly one bash code block
// per response with a THOUGHT section. The wire contract (ONE block, ONE
// command, THOUGHT, format example, rejection rule) is ported verbatim
// from mini-swe-agent's config/default.yaml (system_template); only the
// role sentence is generalized to a domain-neutral assistant. The
// mswea_bash_command fence name stays: it is a wire identifier, not
// mission text — renaming it would churn parsers for no generality gain.
const defaultSystemTemplate = `You are a general-purpose assistant that can interact with a computer.

Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).
Include a THOUGHT section before your command where you explain your reasoning process.
Format your response as shown in <format_example>.

<format_example>
Your reasoning and analysis here. Explain why you want to perform the action.

` + fence + `mswea_bash_command
your_command_here
` + fence + `
</format_example>

Failure to follow these rules will cause your response to be rejected.
`

// defaultTaskTemplate is the first user message: the task plus the
// recommended workflow, the operational cautions, the persistence rules,
// system information, an example response, and useful command examples.
// The structure descends from mini's instance_template, but the copy is
// the general-purpose mission (T018): neutral framing, a
// restate→gather→verify→record→submit workflow with an explicit
// verification step, read-only investigation examples instead of file
// editing recipes, and the report-after-marker line whose output lands in
// Answer.Submission. Patterns per docs/harness-research.md §5 (gptme
// ending discipline, OpenHands cautions and structured finish report,
// terminal-bench per-step analysis); the macOS note is conditional on the
// platform.
const defaultTaskTemplate = `Complete this task: {{.Task}}

You can execute bash commands to gather information, run tools, and inspect or modify the environment. You run unattended: every response must end with the next command or the submit marker. Never end with an offer to elaborate or a request for confirmation; there is no one to answer.

## Recommended Workflow

Work step by step so that each command's output informs the next decision, and scale the effort to the task.

1. Restate the objective in a sentence or two, and decide what to establish first.
2. Gather information one command at a time, reading each result before choosing the next command.
3. Verify before you report: cross-check every important claim against a second, independent command or source.
4. Record what you learn in a notes file as you go, so the final report rests on written evidence.
5. Submit your changes and finish your work by issuing the following command: ` + "`echo {{.SubmitMarker}}`" + `.
   Do not combine it with any other command. <important>After this command, you cannot continue working on this task.</important>
   On the lines after the marker, include a summary of what you did, what you found, and any open questions.

## Important Rules

1. Every response must contain exactly one action
2. The action must be enclosed in triple backticks
3. Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
   However, you can prefix any action with ` + "`MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...`" + ` or write/load environment variables from files

## Operational cautions

- If the task asks a question, answer it. Do not modify the environment to change the answer.
- Avoid commands that wait for interactive input: prefer ` + "`--no-pager`" + ` and batch flags over pagers and prompts, and poll long-running work with short checks instead of blocking.
- Kill processes by their exact PID. A broad name or pattern can match unrelated work, including your own session.
- Keep each command's output under the truncation limit: pipe through ` + "`| head`" + ` or ` + "`| tail`" + `, use targeted flags, or redirect to a file and view it in slices.

<system_information>
OS: {{.GOOS}}; Architecture: {{.GOARCH}}; Hostname: {{.Hostname}}
</system_information>
{{if .IsDarwin}}
You are on MacOS. Some commands behave differently than on Linux (BSD userland); when output surprises you, check the command's documentation.
{{end}}
## Formatting your response

Here is an example of a correct response:

<example_response>
THOUGHT: I need to see what is in the current directory and which tools are available before planning the first real step.

` + fence + `mswea_bash_command
ls -la
` + fence + `
</example_response>

## Useful command examples

### Explore files and text

` + fence + `mswea_bash_command
# Find files by name, first 20 matches only
find . -name '*.conf' -type f | head -20

# Search inside files, with line numbers, bounded
grep -rn "ERROR" logs/ | head -40

# Show the start or the end of a file
head -50 filename.txt
tail -50 filename.txt
` + fence + `

### Inspect the system without paging

` + fence + `mswea_bash_command
# History and pending modifications, without an interactive pager
git --no-pager log --oneline -10
git --no-pager diff --stat

# What is running, and which tools are installed
ps aux | head -20
command -v python3 && python3 --version
` + fence + `

### Any other command you want to run

` + fence + `mswea_bash_command
anything
` + fence

// defaultObservationTemplate renders one command result for the model
// (mini's observation_template): the exception block only when the
// execution itself failed, the return code, and either the full output or
// the too-long warning with head/tail excerpts. The formatter below
// decides which fields are populated; the template only lays them out.
const defaultObservationTemplate = `{{if .Exception}}<exception>{{.Exception}}</exception>
{{end}}<returncode>{{.ReturnCode}}</returncode>
{{if .TooLong}}<warning>
The output of your last command was too long.
Please try a different command that produces less output.
If you're looking at a file you can try use head, tail or sed to view a smaller number of lines selectively.
If you're using grep or find and it produced too much output, you can use a more selective search pattern.
If you really need to see something from the full command's output, you can redirect output to a file and then search in that file.
</warning>
<output_head>
{{.Head}}
</output_head>
<elided_chars>{{.Elided}} characters elided</elided_chars>
<output_tail>
{{.Tail}}
</output_tail>{{else}}<output>
{{.Output}}
</output>{{end}}`

// defaultFormatErrorTemplate is appended as guidance when the model's
// response does not contain exactly one action block (mini's
// format_error_template). It has two variants: a conciseness hint when the
// response was truncated (finish_reason length/tool_calls), and the
// general "exactly one action" guidance — which includes the
// TRIPLEBACKTICKSBASH escape hatch for commands that must embed fences.
const defaultFormatErrorTemplate = `{{if .FinishReason}}Your previous response reached the output token limit (finish_reason={{.FinishReason}}) before you produced a complete action, so it was cut off. Respond more concisely and provide exactly one action in the required format. If you need to think more, do so briefly.{{else}}Format error:

<error>
{{.Error}}
</error>

Here is general guidance on how to format your response:

Please always provide EXACTLY ONE action in triple backticks, found {{.Count}} actions.
If you want to end the task, please issue the following command: ` + "`echo {{.SubmitMarker}}`" + `
without any other command.
Else, please format your response exactly as follows:

<response_example>
Here are some thoughts about why you want to perform the action.

` + fence + `mswea_bash_command
<action>
` + fence + `
</response_example>

Note: In rare cases, if you need to reference a similar format in your command, you might have
to proceed in two steps, first writing TRIPLEBACKTICKSBASH, then replacing them with ` + fence + `mswea_bash_command.{{end}}`

// systemData carries the runtime facts available to the system and task
// templates (mini exposes platform.* values; the Go port uses the runtime
// package and the hostname).
type systemData struct {
	GOOS     string
	GOARCH   string
	Hostname string
	IsDarwin bool
}

// taskData is the data for the task (instance) template: the task text,
// the submit marker, and the system facts.
type taskData struct {
	Task         string
	SubmitMarker string
	GOOS         string
	GOARCH       string
	Hostname     string
	IsDarwin     bool
}

// observationData is the data for the observation template. Exactly one of
// Output (within the limit) or the TooLong excerpt set (Head, Tail,
// Elided) is populated; Exception is non-empty only when the execution
// itself failed.
type observationData struct {
	Exception  string
	ReturnCode int
	Output     string
	TooLong    bool
	Head       string
	Tail       string
	Elided     int
}

// formatErrorData is the data for the format-error template. FinishReason
// selects the truncation variant (mini uses "length" or "tool_calls");
// Count is the number of action blocks found (0 when none).
type formatErrorData struct {
	FinishReason string
	Error        string
	Count        int
	SubmitMarker string
}

// newSystemData captures the runtime facts for template rendering.
func newSystemData() systemData {
	host, err := os.Hostname()
	if err != nil {
		host = ""
	}
	return systemData{
		GOOS:     runtime.GOOS,
		GOARCH:   runtime.GOARCH,
		Hostname: host,
		IsDarwin: runtime.GOOS == "darwin",
	}
}

// parseTemplate parses name with strict missing-key handling so template
// typos fail fast instead of rendering "<no value>".
func parseTemplate(name, text string) (*template.Template, error) {
	return template.New(name).Option("missingkey=error").Parse(text)
}

// renderText executes tmpl over data and returns the rendered string.
func renderText(tmpl *template.Template, data any) (string, error) {
	var b strings.Builder
	if err := tmpl.Execute(&b, data); err != nil {
		return "", fmt.Errorf("bash: render %s: %w", tmpl.Name(), err)
	}
	return b.String(), nil
}

// renderSystem parses and renders the system template over the runtime
// facts (the system prompt is static per run; the task arrives later).
func renderSystem(text string, sys systemData) (string, error) {
	tmpl, err := parseTemplate("system", text)
	if err != nil {
		return "", fmt.Errorf("bash: %w", err)
	}
	return renderText(tmpl, sys)
}

// renderTask parses and renders the task (instance) template: the first
// user message with the task text, the submit marker, and the runtime
// facts.
func renderTask(text string, data taskData) (string, error) {
	tmpl, err := parseTemplate("task", text)
	if err != nil {
		return "", fmt.Errorf("bash: %w", err)
	}
	return renderText(tmpl, data)
}

// formatObservation renders one execution result with the observation
// template: exception block (only on execution failure), return code, and
// the full output while it is shorter than limit runes. At limit or more,
// the warning block plus the first and last limit/2 runes and an elision
// count replace the raw output — never the full blob.
func formatObservation(res ExecResult, limit int, tmpl *template.Template) (string, error) {
	data := observationData{Exception: res.ExceptionInfo, ReturnCode: res.ReturnCode}
	runes := []rune(res.Output)
	if limit > 0 && len(runes) >= limit {
		half := limit / 2
		data.TooLong = true
		data.Head = string(runes[:half])
		data.Tail = string(runes[len(runes)-half:])
		data.Elided = len(runes) - 2*half
	} else {
		data.Output = res.Output
	}
	return renderText(tmpl, data)
}

// commandTags are the fence tags accepted as bash actions: mini-swe-agent's
// mswea_bash_command first, with plain bash as the alias.
var commandTags = map[string]bool{"mswea_bash_command": true, "bash": true}

// findCommands extracts the fenced bash commands from a model response.
// A command is the body of a fence tagged mswea_bash_command (or bash):
// opening line, body lines, closing fence. The protocol requires exactly
// one; callers treat any other count as a format error. The body is
// whitespace-trimmed like mini-swe-agent's parse_action.
//
// Nested-fence tolerance: fence lines inside the body (as in a heredoc
// writing markdown) pair up and do not close the command early; the body
// ends at the first bare closing fence at nesting depth zero. This is a
// deliberate superset of mini's non-greedy regex, which stops at the first
// inner fence — the TRIPLEBACKTICKSBASH escape hatch covers that case for
// mini-conformant models and keeps working here.
func findCommands(text string) []string {
	lines := strings.Split(text, "\n")
	var commands []string
	for i := 0; i < len(lines); i++ {
		if _, ok := openingTag(lines[i]); !ok {
			continue
		}
		body, closeAt, found := scanBody(lines, i+1)
		if !found {
			// Unterminated block: no closing fence, like mini's regex
			// which requires one. Skip and keep scanning after the opener.
			continue
		}
		commands = append(commands, strings.TrimSpace(strings.Join(body, "\n")))
		i = closeAt
	}
	return commands
}

// openingTag reports whether line opens a command fence and returns the
// tag (mswea_bash_command or bash). Leading and trailing whitespace is
// ignored, matching mini's line-tolerant matching.
func openingTag(line string) (string, bool) {
	t := strings.TrimSpace(line)
	if !strings.HasPrefix(t, fence) {
		return "", false
	}
	tag := strings.TrimSpace(strings.TrimPrefix(t, fence))
	return tag, commandTags[tag]
}

// scanBody collects body lines starting at start until the command's
// closing fence — the first bare fence line at nesting depth zero. Fence
// lines inside the body (inner blocks) increase and decrease the depth
// instead of closing. found is false when the text ends without the
// closing fence.
func scanBody(lines []string, start int) (body []string, closeAt int, found bool) {
	depth := 0
	for i := start; i < len(lines); i++ {
		t := strings.TrimSpace(lines[i])
		switch {
		case t == fence:
			if depth == 0 {
				return body, i, true
			}
			depth--
		case strings.HasPrefix(t, fence):
			// An inner fenced block opened inside the command body.
			depth++
		}
		body = append(body, lines[i])
	}
	return nil, 0, false
}
