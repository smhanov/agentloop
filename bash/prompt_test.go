package bash

import (
	"strings"
	"testing"
)

// renderObservation is a test helper: parse the template and format.
func renderObservation(t *testing.T, text string, res ExecResult, limit int) string {
	t.Helper()
	tmpl, err := parseTemplate("observation", text)
	if err != nil {
		t.Fatalf("parse observation template: %v", err)
	}
	out, err := formatObservation(res, limit, tmpl)
	if err != nil {
		t.Fatalf("formatObservation: %v", err)
	}
	return out
}

// TestObservationShortOutput pins the §8.4 shape for output below the
// threshold: exception block only when present, return code, full output.
func TestObservationShortOutput(t *testing.T) {
	got := renderObservation(t, defaultObservationTemplate,
		ExecResult{Output: "hello\nworld\n", ReturnCode: 0}, defaultOutputLimit)
	want := "<returncode>0</returncode>\n<output>\nhello\nworld\n\n</output>"
	if got != want {
		t.Fatalf("observation = %q, want %q", got, want)
	}
}

// TestObservationException covers the execution-failure path: the
// exception block precedes the return code and only appears when non-empty.
func TestObservationException(t *testing.T) {
	got := renderObservation(t, defaultObservationTemplate,
		ExecResult{Output: "partial", ReturnCode: -1, ExceptionInfo: "command timed out after 1s"}, defaultOutputLimit)
	want := "<exception>command timed out after 1s</exception>\n<returncode>-1</returncode>\n<output>\npartial\n</output>"
	if got != want {
		t.Fatalf("observation = %q, want %q", got, want)
	}
}

// TestObservationTruncationBoundary pins the threshold semantics: output
// shorter than the limit renders in full; at the limit the head/tail
// elision path starts (spec: <output> when len < 10000).
func TestObservationTruncationBoundary(t *testing.T) {
	below := renderObservation(t, defaultObservationTemplate,
		ExecResult{Output: strings.Repeat("x", defaultOutputLimit-1), ReturnCode: 0}, defaultOutputLimit)
	if !strings.Contains(below, "<output>\n") || strings.Contains(below, "characters elided") {
		t.Fatalf("output one below the limit should render untruncated, got %q", below[:120]+"...")
	}
	atLimit := renderObservation(t, defaultObservationTemplate,
		ExecResult{Output: strings.Repeat("x", defaultOutputLimit), ReturnCode: 0}, defaultOutputLimit)
	if !strings.Contains(atLimit, "<elided_chars>0 characters elided</elided_chars>") {
		t.Fatal("output at the limit should take the elision path with 0 elided chars")
	}
}

// TestObservationTruncationHeadTail pins the ≥limit path exactly: first
// limit/2 runes, an elision count, last limit/2 runes, and never the
// middle of the output.
func TestObservationTruncationHeadTail(t *testing.T) {
	output := strings.Repeat("H", 5000) + strings.Repeat("M", 2345) + strings.Repeat("T", 5000)
	got := renderObservation(t, defaultObservationTemplate,
		ExecResult{Output: output, ReturnCode: 0}, defaultOutputLimit)
	if !strings.Contains(got, "<output_head>\n"+strings.Repeat("H", 5000)+"\n</output_head>") {
		t.Error("head excerpt missing or wrong")
	}
	if !strings.Contains(got, "<output_tail>\n"+strings.Repeat("T", 5000)+"\n</output_tail>") {
		t.Error("tail excerpt missing or wrong")
	}
	if !strings.Contains(got, "<elided_chars>2345 characters elided</elided_chars>") {
		t.Errorf("elision count missing or wrong: %q", got[:200]+"...")
	}
	if strings.Contains(got, strings.Repeat("M", 100)) {
		t.Error("elided middle leaked into the observation")
	}
	if strings.Contains(got, "<output>\n") {
		t.Error("full <output> block must not appear on the elision path")
	}
}

// TestObservationCustomLimitDerivesHalves: the head/tail split derives
// from the configured limit, not a hardcoded 5000.
func TestObservationCustomLimitDerivesHalves(t *testing.T) {
	output := strings.Repeat("A", 50) + strings.Repeat("B", 50) + strings.Repeat("C", 50)
	got := renderObservation(t, defaultObservationTemplate,
		ExecResult{Output: output, ReturnCode: 0}, 100)
	if !strings.Contains(got, "<output_head>\n"+strings.Repeat("A", 50)+"\n</output_head>") ||
		!strings.Contains(got, "<output_tail>\n"+strings.Repeat("C", 50)+"\n</output_tail>") ||
		!strings.Contains(got, "<elided_chars>50 characters elided</elided_chars>") {
		t.Fatalf("custom-limit excerpts wrong: %q", got)
	}
}

// TestFormatErrorTemplateVariants covers both ported variants: the
// truncation hint when a finish reason is provided, and the general
// guidance (with the found-N count and the TRIPLEBACKTICKSBASH escape
// hatch) otherwise.
func TestFormatErrorTemplateVariants(t *testing.T) {
	tmpl, err := parseTemplate("format-error", defaultFormatErrorTemplate)
	if err != nil {
		t.Fatalf("parse format-error template: %v", err)
	}
	truncated, err := renderText(tmpl, formatErrorData{
		FinishReason: "length", Error: "Expected exactly 1 action, found 0.", Count: 0, SubmitMarker: defaultSubmitMarker,
	})
	if err != nil {
		t.Fatalf("render truncated variant: %v", err)
	}
	if !strings.Contains(truncated, "finish_reason=length") || !strings.Contains(truncated, "Respond more concisely") {
		t.Fatalf("truncation variant wrong: %q", truncated)
	}
	if strings.Contains(truncated, "TRIPLEBACKTICKSBASH") {
		t.Error("truncation variant should be the short conciseness hint")
	}

	general, err := renderText(tmpl, formatErrorData{
		Error: "Expected exactly 1 action, found 2.", Count: 2, SubmitMarker: defaultSubmitMarker,
	})
	if err != nil {
		t.Fatalf("render general variant: %v", err)
	}
	for _, want := range []string{
		"found 2 actions",
		"EXACTLY ONE action",
		"echo " + defaultSubmitMarker,
		"TRIPLEBACKTICKSBASH",
	} {
		if !strings.Contains(general, want) {
			t.Errorf("general variant missing %q", want)
		}
	}
}

// TestDefaultTemplatesAreFinanciallyNeutral: the ported templates carry
// only generic coding-agent content (repo hygiene rule).
func TestDefaultTemplatesAreFinanciallyNeutral(t *testing.T) {
	for name, text := range map[string]string{
		"system":      defaultSystemTemplate,
		"task":        defaultTaskTemplate,
		"observation": defaultObservationTemplate,
		"format":      defaultFormatErrorTemplate,
	} {
		for _, word := range []string{"stock", "portfolio", "trade", "trading", "invest"} {
			if strings.Contains(strings.ToLower(text), word) {
				t.Errorf("%s template contains domain term %q", name, word)
			}
		}
	}
}

// TestRenderSystemAndTask pins the ported prompt shapes: the system
// template shows the format example; the task template carries the task,
// the submit marker, and the system information block.
func TestRenderSystemAndTask(t *testing.T) {
	sys := newSystemData()
	system, err := renderSystem(defaultSystemTemplate, sys)
	if err != nil {
		t.Fatalf("render system: %v", err)
	}
	for _, want := range []string{"exactly ONE bash code block", fence + "mswea_bash_command", "<format_example>"} {
		if !strings.Contains(system, want) {
			t.Errorf("system prompt missing %q", want)
		}
	}

	task, err := renderTask(defaultTaskTemplate, taskData{
		Task: "Fix the bug in widget.go", SubmitMarker: defaultSubmitMarker, GOOS: sys.GOOS, GOARCH: sys.GOARCH, Hostname: sys.Hostname, IsDarwin: sys.IsDarwin,
	})
	if err != nil {
		t.Fatalf("render task: %v", err)
	}
	for _, want := range []string{
		"Please solve this issue: Fix the bug in widget.go",
		"echo " + defaultSubmitMarker,
		"OS: " + sys.GOOS,
		"Architecture: " + sys.GOARCH,
		"<system_information>",
	} {
		if !strings.Contains(task, want) {
			t.Errorf("task prompt missing %q", want)
		}
	}
	if sys.IsDarwin && !strings.Contains(task, "sed -i ''") {
		t.Error("macOS sed note missing on darwin")
	}
}
