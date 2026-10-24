package bash

import (
	"reflect"
	"testing"
)

// The fenced-action test corpus is ported from mini-swe-agent's
// MIT-licensed parser tests (fetched 2026-10-24):
//
//	https://github.com/SWE-agent/mini-swe-agent/blob/v1.17.5/tests/agents/test_default.py
//	(test_parse_action_success, test_parse_action_failures)
//	https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/models/litellm_textbased_model.py
//	(the v2 action regex: ```mswea_bash_command\s*\n(.*?)\n``` with
//	re.DOTALL, findall, and an exactly-one requirement)
//
// The nested-fence cases are a documented delta: mini's non-greedy regex
// ends the command at the first inner fence, while findCommands pairs
// inner fences so the full command survives (spec: nested-fence
// tolerance); the TRIPLEBACKTICKSBASH escape hatch remains valid for
// models trained on mini's guidance.
func TestFindCommandsCorpus(t *testing.T) {
	tests := []struct {
		name string
		text string
		want []string
	}{
		{
			// v1 corpus: plain valid block.
			name: "single bash block",
			text: "```bash\necho 'test'\n```",
			want: []string{"echo 'test'"},
		},
		{
			// v1 corpus: another command shape.
			name: "single bash block ls",
			text: "```bash\nls -la\n```",
			want: []string{"ls -la"},
		},
		{
			// v1 corpus: prose around the block.
			name: "prose around block",
			text: "Some text\n```bash\necho 'hello'\n```\nMore text",
			want: []string{"echo 'hello'"},
		},
		{
			// v1 corpus: no block at all.
			name: "zero blocks",
			text: "No code blocks here",
			want: nil,
		},
		{
			// v1 corpus: two blocks is a format error upstream.
			name: "two blocks",
			text: "```bash\necho 'first'\n```\n```bash\necho 'second'\n```",
			want: []string{"echo 'first'", "echo 'second'"},
		},
		{
			// v1 corpus: an untagged fence is not an action.
			name: "untagged fence",
			text: "```\nls -la\n```",
			want: nil,
		},
		{
			// v2 tag: mswea_bash_command is the primary fence tag.
			name: "mswea_bash_command tag",
			text: "THOUGHT: look around\n```mswea_bash_command\nls\n```",
			want: []string{"ls"},
		},
		{
			// v2 regex allows whitespace after the tag and before the body.
			name: "tag with trailing spaces and blank line",
			text: "```mswea_bash_command  \n\nls -la\n```",
			want: []string{"ls -la"},
		},
		{
			// Tolerance: an untagged fence in the reasoning does not hide
			// the tagged action.
			name: "untagged fence in thought",
			text: "THOUGHT: for example\n```python\nprint('hi')\n```\n\n```mswea_bash_command\npwd\n```",
			want: []string{"pwd"},
		},
		{
			// Delta vs mini's regex: a fence pair inside the body (heredoc
			// writing markdown) does not close the command early.
			name: "nested fence inside body",
			text: "```mswea_bash_command\ncat <<'EOF' > readme.md\n# Example\n\n" + fence + "bash\necho hi\n" + fence + "\nEOF\n" + fence + "",
			want: []string{"cat <<'EOF' > readme.md\n# Example\n\n" + fence + "bash\necho hi\n" + fence + "\nEOF"},
		},
		{
			// TRIPLEBACKTICKSBASH escape hatch: the placeholder passes
			// through verbatim and does not disturb the parse.
			name: "triplebackticksbash placeholder",
			text: "```mswea_bash_command\nprintf 'TRIPLEBACKTICKSBASH mswea_bash_command\\n' > snippet.txt\n```",
			want: []string{"printf 'TRIPLEBACKTICKSBASH mswea_bash_command\\n' > snippet.txt"},
		},
		{
			// Regex parity: an unterminated block is no action.
			name: "unterminated block",
			text: "```bash\necho 'never closed'",
			want: nil,
		},
		{
			// Line-tolerant: indented fences still parse.
			name: "indented block",
			text: "  ```bash\n  echo indented\n  ```",
			want: []string{"echo indented"},
		},
		{
			// Empty body is still exactly one (empty) action, matching
			// mini's regex capture and strip.
			name: "empty body",
			text: "```bash\n```",
			want: []string{""},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := findCommands(tt.text)
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("findCommands(%q) = %q, want %q", tt.text, got, tt.want)
			}
		})
	}
}

// TestOpeningTagRejectsCoversFenceLikeLines pins the tag matching so a
// fence for another language never opens a command block.
func TestOpeningTagRejectsNonCommandFences(t *testing.T) {
	for _, line := range []string{"```python", "```sh", "```mswea", "text", "```bashx", "```mswea_bash_commandy"} {
		if _, ok := openingTag(line); ok {
			t.Errorf("openingTag(%q) accepted a non-command fence", line)
		}
	}
	for _, line := range []string{"```bash", "```mswea_bash_command", "```bash  ", "  ```mswea_bash_command"} {
		if _, ok := openingTag(line); !ok {
			t.Errorf("openingTag(%q) rejected a command fence", line)
		}
	}
}

// TestFindCommandsIgnoresInnerCloseOnlyFence: a body containing only a
// stray inner closing fence keeps scanning to the real close at depth zero
// only when pairs balance; an unbalanced stray keeps the block open.
func TestFindCommandsUnbalancedInnerFence(t *testing.T) {
	text := "```bash\necho hi\n" + fence + "inner\nmore"
	if got := findCommands(text); len(got) != 0 {
		t.Fatalf("findCommands = %q, want none", got)
	}
}
