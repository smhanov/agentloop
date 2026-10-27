package search

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/smhanov/agentloop"
)

// skillTree materializes files (relative paths under a fresh temp root)
// and returns the root, ready to hand to agentloop.WithSkills.
func skillTree(t *testing.T, files map[string]string) string {
	t.Helper()
	root := t.TempDir()
	for rel, content := range files {
		path := filepath.Join(root, rel)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

// skillMD renders a valid SKILL.md with the given frontmatter and body.
func skillMD(name, description, body string) string {
	return "---\nname: " + name + "\ndescription: " + description + "\n---\n\n" + body + "\n"
}

// TestSkillIssuesPassThrough pins that the preset forwards WithSkills to
// the core unchanged: a mis-shaped skills root never fails construction,
// and its stable SkillIssue is observable through the wrapped core agent
// exactly as on a directly built one. The preset has no WithSkills of its
// own — core options compose here like anywhere else.
func TestSkillIssuesPassThrough(t *testing.T) {
	folder := filepath.Join(skillTree(t, map[string]string{
		"misplaced/SKILL.md": skillMD("misplaced", "Use when misplaced.", "Body."),
	}), "misplaced")
	agent := newFakeAgent(t, agentloop.WithSkills(folder))

	issues := agent.Agent().SkillIssues()
	if len(issues) != 1 {
		t.Fatalf("issues = %+v, want exactly 1", issues)
	}
	if want := "skill folder, not a skills root (pass its parent directory)"; issues[0].Reason != want {
		t.Fatalf("Reason = %q, want %q", issues[0].Reason, want)
	}
	if issues[0].Name != "misplaced" || issues[0].Dir != folder {
		t.Fatalf("issue = %+v, want the root-shape issue for misplaced", issues[0])
	}
}
