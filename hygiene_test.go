package agentloop

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// agentloop is a domain-neutral library, and the public repository keeps
// it that way (specs.md §15): no financial, stock, or trading terminology
// in any committed file — README, doc comments, identifiers, tests,
// testdata, or examples. This test walks every git-tracked file and fails
// when a forbidden term appears outside the allowlist below. Like the
// exported-API budget, it turns a documentation promise into a gate.
//
// The scan is substring-based and case-insensitive, so it deliberately
// over-matches ("stock" also finds "stockholm"); the allowlist exists for
// such legitimate false positives, not for real findings.

// forbiddenTerms are the financial/trading terms that must not appear in
// committed files.
var forbiddenTerms = []string{
	"stock",
	"ticker",
	"portfolio",
	"dividend",
	"earnings",
	"swing trade",
	"options trading",
	"trading",
}

// allowedHits is the allowlist: known, legitimate occurrences of forbidden
// terms, keyed by file path (a term is allowed for the whole file, which
// keeps the table readable). Every entry documents why it is fine.
var allowedHits = map[string]string{
	// The budget/hygiene tests' own term lists: mentioning a forbidden
	// word is the only way to forbid it. bash/prompt_test.go carries the
	// same guard for the ported mini-swe-agent templates.
	"apibudget_test.go":   "term list of the guard tests themselves",
	"hygiene_test.go":     "term list of the guard tests themselves",
	"bash/prompt_test.go": "template-neutrality check's own term list",
	// The internal design specification documents the local-only test
	// corpus (its §14), which by the same repo rule never enters the
	// published documentation; the file is the one place the corpus may be
	// written down and is maintained out-of-band of the shipped surface.
	"specs.md": "internal design doc; its local-corpus section",
}

// trackedFiles returns the repository's git-tracked files (with a plain
// filesystem-walk fallback when git is unavailable, e.g. a source tarball
// without .git; the fallback skips the same untracked paths).
func trackedFiles(t *testing.T) []string {
	t.Helper()
	cmd := exec.Command("git", "ls-files", "-z")
	if out, err := cmd.Output(); err == nil {
		var files []string
		for _, f := range strings.Split(string(out), "\x00") {
			if f != "" {
				files = append(files, f)
			}
		}
		return files
	}
	t.Log("git unavailable; falling back to a directory walk")
	var files []string
	err := filepath.WalkDir(".", func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			switch d.Name() {
			case ".git", ".plans", "testdata-live":
				return filepath.SkipDir
			}
			return nil
		}
		files = append(files, path)
		return nil
	})
	if err != nil {
		t.Fatalf("walk repository: %v", err)
	}
	return files
}

// TestNoFinancialTerminology scans every tracked file for the forbidden
// terms and fails on any finding outside the documented allowlist.
func TestNoFinancialTerminology(t *testing.T) {
	files := trackedFiles(t)
	if len(files) == 0 {
		t.Fatal("no tracked files found; run from the repository root")
	}
	findings := 0
	for _, file := range files {
		data, err := os.ReadFile(file)
		if err != nil {
			t.Fatalf("read %s: %v", file, err)
		}
		lower := strings.ToLower(string(data))
		for _, term := range forbiddenTerms {
			if !strings.Contains(lower, term) {
				continue
			}
			if reason, ok := allowedHits[file]; ok {
				t.Logf("allowed: %s contains %q (%s)", file, term, reason)
				continue
			}
			findings++
			t.Errorf("%s contains forbidden term %q; the repository stays domain-neutral", file, term)
		}
	}
	if findings == 0 {
		t.Logf("scanned %d tracked files: no findings outside the %d-entry allowlist",
			len(files), len(allowedHits))
	}
}
