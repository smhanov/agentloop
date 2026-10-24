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
// per response with a THOUGHT section, ported from mini-swe-agent's
// config/default.yaml (system_template).
const defaultSystemTemplate = `You are a helpful assistant that can interact with a computer.

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
// recommended workflow, the persistence rules, system information, an
// example response, and useful command examples — ported from mini's
// instance_template. The macOS sed note is conditional on the platform.
const defaultTaskTemplate = `Please solve this issue: {{.Task}}

You can execute bash commands and edit files to implement the necessary changes.

## Recommended Workflow

This workflow should be done step-by-step so that you can iterate on your changes and any possible problems.

1. Analyze the codebase by finding and reading relevant files
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Test edge cases to ensure your fix is robust
6. Submit your changes and finish your work by issuing the following command: ` + "`echo {{.SubmitMarker}}`" + `.
   Do not combine it with any other command. <important>After this command, you cannot continue working on this task.</important>

## Important Rules

1. Every response must contain exactly one action
2. The action must be enclosed in triple backticks
3. Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
   However, you can prefix any action with ` + "`MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...`" + ` or write/load environment variables from files

<system_information>
OS: {{.GOOS}}; Architecture: {{.GOARCH}}; Hostname: {{.Hostname}}
</system_information>

## Formatting your response

Here is an example of a correct response:

<example_response>
THOUGHT: I need to understand the structure of the repository first. Let me check what files are in the current directory to get a better understanding of the codebase.

` + fence + `mswea_bash_command
ls -la
` + fence + `
</example_response>

## Useful command examples

### Create a new file:

` + fence + `mswea_bash_command
cat <<'EOF' > newfile.txt
hello = "world"
print(hello)
EOF
` + fence + `

### Edit files with sed:
{{if .IsDarwin}}
<important>
You are on MacOS. For all the below examples, you need to use ` + "`sed -i ''`" + ` instead of ` + "`sed -i`" + `.
</important>
{{end}}
` + fence + `mswea_bash_command
# Replace all occurrences
sed -i 's/old_string/new_string/g' filename.txt

# Replace only first occurrence
sed -i 's/old_string/new_string/' filename.txt

# Replace first occurrence on line 1
sed -i '1s/old_string/new_string/' filename.txt

# Replace all occurrences in lines 1-10
sed -i '1,10s/old_string/new_string/g' filename.txt
` + fence + `

### View file content:

` + fence + `mswea_bash_command
# View specific lines with numbers
nl -ba filename.txt | sed -n '10,20p'
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
