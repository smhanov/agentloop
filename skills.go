package agentloop

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/smhanov/llmhub"
	"gopkg.in/yaml.v3"
)

// Skill-file conventions and validation limits (specs.md §5.8).
const (
	// skillFileName is the required marker file inside a skill folder.
	skillFileName = "SKILL.md"
	// skillReadToolName is the tool registered whenever skills are enabled.
	skillReadToolName = "skill_read"
	// maxSkillNameLen caps skill names; the regex already enforces >= 1.
	maxSkillNameLen = 64
	// maxDescriptionRunes caps catalog descriptions.
	maxDescriptionRunes = 1024
	// maxPinnedSkills bounds WithPinnedSkills: at most two bodies may be
	// inlined into the system prompt.
	maxPinnedSkills = 2
	// maxFrontmatterBytes caps the YAML frontmatter block before parsing,
	// a cheap guard against pathological alias or nesting expansion in a
	// hand-authored pack.
	maxFrontmatterBytes = 64 << 10
)

// skillNameRe matches a valid skill name: lowercase alphanumerics joined
// by single hyphens, 1-64 characters (length re-checked separately).
var skillNameRe = regexp.MustCompile(`^[a-z0-9]+(-[a-z0-9]+)*$`)

// SkillIssue reasons. They are stable strings so callers and tests can
// match on them.
const (
	issueBadFrontmatter   = "bad frontmatter"
	issueMissingSkillMD   = "missing SKILL.md"
	issueUnreadableMD     = "unreadable SKILL.md"
	issueNameMismatch     = "name mismatch"
	issueInvalidName      = "invalid name"
	issueEmptyDescription = "empty description"
	issueLongDescription  = "description too long"
	issueEmptyBody        = "empty body"
	issueNULByte          = "NUL byte"
	issueStegoStripped    = "unicode steganography stripped"
	issueDuplicate        = "duplicate skill"
	issueNotSkillsRoot    = "skill folder, not a skills root (pass its parent directory)"
	issueUnreadableDir    = "unreadable directory"
)

// Skill is one validated skill folder: a repeatable procedure the model
// discovers through the catalog (L0) and loads on demand (L1/L2).
type Skill struct {
	// Name is the skill identifier. It must equal the directory name and
	// match ^[a-z0-9]+(-[a-z0-9]+)*$ with 1-64 characters.
	Name string
	// Description is the catalog line — the router. Write it
	// imperatively ("Use when…"); the spec caps it at 1024 characters.
	Description string
	// Body is the SKILL.md markdown below the frontmatter, with
	// invisible characters stripped (steganography defense).
	Body string
	// Dir is the resolved absolute path of the skill folder.
	Dir string
	// LinkedFiles lists the skill's resource files: relative slash paths
	// under references/ and assets/, precomputed (symlinks resolved,
	// escapes dropped, sorted) so the model never needs to list the
	// directory. Scripts under scripts/ are deliberately not listed:
	// they run via an execution tool and only their output enters
	// context (L3).
	LinkedFiles []string
	// ScriptFiles lists the skill's executable helpers: relative slash
	// paths under scripts/, precomputed like LinkedFiles (symlinks
	// resolved, escapes dropped, sorted) so the body can reference them.
	// They are advertised for execution, never for reading: run them
	// through the agent's execution tool from Dir and only their output
	// enters context (L3).
	ScriptFiles []string
}

// SkillIssue reports a fail-closed outcome of skill loading: either a
// skipped skill directory and why it was rejected, or an advisory about a
// loaded skill whose content required sanitization. Invalid skills are
// omitted from the result — never an error, never a crash.
type SkillIssue struct {
	// Dir is the skill directory the issue refers to (the scan root
	// itself when the directory could not be scanned).
	Dir string
	// Name is the skill folder name, populated for every rejection of a
	// specific skill folder so a dropped skill stays identifiable. It is
	// empty only for issues about the scan root itself.
	Name string
	// Reason is a short stable string, e.g. "bad frontmatter",
	// "name mismatch", "empty body", "duplicate skill", or
	// "unicode steganography stripped".
	Reason string
}

// LoadSkills scans the given skills roots — directories whose immediate
// children are skill folders — in precedence order and returns the valid
// skills plus the issues encountered on the way. Within and across roots
// the first occurrence of a name wins; later duplicates are reported as
// issues and never merged. A skill folder itself (a directory containing
// SKILL.md) is not a root: passed as an argument it loads nothing and is
// reported as exactly one SkillIssue with the reason "skill folder, not
// a skills root (pass its parent directory)" — pass the parent
// directory instead.
//
// Loading is fail-closed: a skill folder that fails any check (frontmatter
// structure, name regex and directory-name equality, description length,
// non-empty body, NUL-byte check) is reported as a SkillIssue and omitted.
// The error return is reserved for systemic failures such as a
// permission-denied scan root; a root that is merely missing or not a
// directory yields an issue, not an error. Unicode Tags (U+E0001-U+E007F)
// and bidi overrides (U+202A-U+202E, U+2066-U+2069) are stripped from
// name, description, and body; when stripping changed anything an advisory
// SkillIssue ("unicode steganography stripped") is recorded and the
// sanitized skill proceeds through validation.
func LoadSkills(dirs ...string) ([]Skill, []SkillIssue, error) {
	var (
		skills []Skill
		issues []SkillIssue
		seen   = make(map[string]bool)
	)
	for _, root := range dirs {
		absRoot, err := filepath.Abs(root)
		if err != nil {
			return skills, issues, fmt.Errorf("agentloop: load skills: resolve %q: %w", root, err)
		}
		entries, err := os.ReadDir(absRoot)
		if err != nil {
			if errors.Is(err, fs.ErrPermission) {
				// A scan root we are not allowed to look at is a
				// systemic failure, not a bad skill folder.
				return skills, issues, fmt.Errorf("agentloop: load skills: scan %q: %w", root, err)
			}
			issues = append(issues, SkillIssue{Dir: root, Reason: issueUnreadableDir})
			continue
		}
		// Root-shape guard: a directory that itself contains SKILL.md is
		// a skill folder, not a skills root — the documented convention
		// puts SKILL.md one level down, in each child. Without this guard
		// the mis-shaped argument degrades into misleading per-child
		// "missing SKILL.md" issues, or into nothing at all for a bare
		// folder. Report it once, distinctly, and move on to the next
		// root. A legitimate root never contains a top-level SKILL.md
		// file, so this cannot misfire; the IsDir check keeps a
		// pathologically named SKILL.md directory a child-scan matter.
		if info, err := os.Stat(filepath.Join(absRoot, skillFileName)); err == nil && !info.IsDir() {
			issues = append(issues, SkillIssue{
				Dir:    absRoot,
				Name:   filepath.Base(absRoot),
				Reason: issueNotSkillsRoot,
			})
			continue
		}
		names := make([]string, 0, len(entries))
		byName := make(map[string]fs.DirEntry, len(entries))
		for _, entry := range entries {
			names = append(names, entry.Name())
			byName[entry.Name()] = entry
		}
		sort.Strings(names) // deterministic scan order
		for _, name := range names {
			path := filepath.Join(absRoot, name)
			info, err := os.Stat(path) // follows symlinks to directories
			if err != nil {
				issues = append(issues, SkillIssue{Dir: path, Name: filepath.Base(path), Reason: issueUnreadableDir})
				continue
			}
			if !info.IsDir() {
				continue // an ordinary file is not a skill attempt
			}
			resolved, err := filepath.EvalSymlinks(path)
			if err != nil {
				issues = append(issues, SkillIssue{Dir: path, Name: filepath.Base(path), Reason: issueUnreadableDir})
				continue
			}
			skill, skillIssues, ok := loadOneSkill(resolved)
			issues = append(issues, skillIssues...)
			if !ok {
				continue
			}
			if seen[skill.Name] {
				issues = append(issues, SkillIssue{Dir: resolved, Name: skill.Name, Reason: issueDuplicate})
				continue
			}
			seen[skill.Name] = true
			skills = append(skills, skill)
		}
	}
	return skills, issues, nil
}

// loadOneSkill validates one skill directory. It returns the skill with
// ok=true, or ok=false with the accumulated issues explaining the
// rejection (which may include a sanitization advisory recorded before a
// later validation failure).
func loadOneSkill(dir string) (Skill, []SkillIssue, bool) {
	var issues []SkillIssue
	fail := func(reason string) (Skill, []SkillIssue, bool) {
		return Skill{}, append(issues, SkillIssue{Dir: dir, Name: filepath.Base(dir), Reason: reason}), false
	}

	raw, err := os.ReadFile(filepath.Join(dir, skillFileName))
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return fail(issueMissingSkillMD)
		}
		return fail(issueUnreadableMD)
	}
	if strings.IndexByte(string(raw), 0) >= 0 {
		// NUL bytes are a payload/binary marker: reject outright rather
		// than strip.
		return fail(issueNULByte)
	}
	fields, bodyRaw, ok := parseFrontmatter(string(raw))
	if !ok {
		return fail(issueBadFrontmatter)
	}

	// Steganography defense: strip invisible characters, then validate the
	// sanitized values. When stripping changed anything, record an
	// advisory and continue with the clean content.
	name, changed1 := stripHiddenChars(string(fields.Name))
	description, changed2 := stripHiddenChars(string(fields.Description))
	description = strings.TrimSpace(description)
	strippedBody, changed3 := stripHiddenChars(bodyRaw)
	if changed1 || changed2 || changed3 {
		issues = append(issues, SkillIssue{Dir: dir, Name: name, Reason: issueStegoStripped})
	}
	body := strings.TrimSpace(strippedBody)

	if !skillNameRe.MatchString(name) || len(name) > maxSkillNameLen {
		return fail(issueInvalidName)
	}
	if name != filepath.Base(dir) {
		return fail(issueNameMismatch)
	}
	switch n := utf8.RuneCountInString(description); {
	case n == 0:
		return fail(issueEmptyDescription)
	case n > maxDescriptionRunes:
		return fail(issueLongDescription)
	}
	if body == "" {
		return fail(issueEmptyBody)
	}
	linked, err := linkedFiles(dir)
	if err != nil {
		return fail(issueUnreadableDir)
	}
	scripts, err := scriptFiles(dir)
	if err != nil {
		return fail(issueUnreadableDir)
	}
	return Skill{
		Name:        name,
		Description: description,
		Body:        body,
		Dir:         dir,
		LinkedFiles: linked,
		ScriptFiles: scripts,
	}, issues, true
}

// skillFrontmatter is the subset of SKILL.md YAML frontmatter the loader
// consumes: name and description. Unknown keys — including nested maps
// such as metadata — are ignored by the non-strict decode.
type skillFrontmatter struct {
	Name        yamlString `yaml:"name"`
	Description yamlString `yaml:"description"`
}

// yamlString is a frontmatter scalar that must be a string or null.
// yaml.v3 decodes non-string scalars (bool, int) into a Go string field by
// taking their literal text; for the fields that drive validation that
// coercion is a frontmatter error, so the tag is checked explicitly.
type yamlString string

// UnmarshalYAML implements yaml.Unmarshaler.
func (s *yamlString) UnmarshalYAML(node *yaml.Node) error {
	if node.Kind == yaml.ScalarNode && node.Tag == "!!null" {
		*s = ""
		return nil
	}
	if node.Kind != yaml.ScalarNode || node.Tag != "!!str" {
		return fmt.Errorf("frontmatter: expected a string, got %s", node.Tag)
	}
	*s = yamlString(node.Value)
	return nil
}

// parseFrontmatter splits a SKILL.md into its frontmatter fields and the
// markdown body below the closing delimiter. The block between the
// opening and closing "---" lines is parsed as real YAML
// (gopkg.in/yaml.v3), as the Agent Skills spec requires: folded and
// literal block scalars, quoted multi-line values, comments, nested maps,
// block and flow sequences, and unknown keys all load; only name and
// description are consumed. UTF-8 BOMs are dropped and CRLF/CR line
// endings normalized first. Anything that is not a YAML mapping of string
// name/description values — missing or unclosed delimiters, invalid YAML,
// a block over maxFrontmatterBytes, or a non-string field — reports
// ok=false.
func parseFrontmatter(content string) (fields skillFrontmatter, body string, ok bool) {
	content = strings.TrimPrefix(content, "\ufeff")
	content = strings.ReplaceAll(content, "\r\n", "\n")
	content = strings.ReplaceAll(content, "\r", "\n")
	lines := strings.Split(content, "\n")
	if len(lines) == 0 || lines[0] != "---" {
		return skillFrontmatter{}, "", false
	}
	end := 1
	for ; end < len(lines) && lines[end] != "---"; end++ {
	}
	if end == len(lines) {
		return skillFrontmatter{}, "", false
	}
	block := strings.Join(lines[1:end], "\n")
	if len(block) > maxFrontmatterBytes {
		return skillFrontmatter{}, "", false
	}
	if err := yaml.Unmarshal([]byte(block), &fields); err != nil {
		return skillFrontmatter{}, "", false
	}
	return fields, strings.Join(lines[end+1:], "\n"), true
}

// stripHiddenChars removes the invisible characters used in steganographic
// prompt injection: Unicode Tags (U+E0001-U+E007F) and bidi overrides and
// isolates (U+202A-U+202E, U+2066-U+2069). It reports whether anything
// was removed.
func stripHiddenChars(s string) (string, bool) {
	changed := false
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		if isHiddenChar(r) {
			changed = true
			continue
		}
		b.WriteRune(r)
	}
	return b.String(), changed
}

// isHiddenChar reports whether r is one of the stripped invisible
// characters (Unicode Tags or bidi overrides/isolates).
func isHiddenChar(r rune) bool {
	switch {
	case r >= 0xE0001 && r <= 0xE007F: // Unicode Tags
		return true
	case r >= 0x202A && r <= 0x202E: // bidi embeddings and overrides
		return true
	case r >= 0x2066 && r <= 0x2069: // bidi isolates
		return true
	}
	return false
}

// linkedFiles lists the skill's resource files: every regular file under
// references/ and assets/, as slash paths relative to the skill
// directory. Symlinks are resolved; files whose real location escapes the
// skill directory are dropped, as are symlinked directories. The result
// is sorted and deduplicated. Missing resource directories are fine (they
// are optional); other walk errors are systemic and returned.
func linkedFiles(dir string) ([]string, error) {
	return skillFiles(dir, "references", "assets")
}

// scriptFiles lists the skill's executable helpers: every regular file
// under scripts/, with the same resolution rules as linkedFiles. Missing
// scripts/ is fine; other walk errors are systemic and returned, so an
// unreadable scripts tree fails the skill rather than silently hiding
// runnable helpers.
func scriptFiles(dir string) ([]string, error) {
	return skillFiles(dir, "scripts")
}

// skillFiles lists every regular file under the given optional
// subdirectories of dir, as slash paths relative to dir. Symlinks are
// resolved; files whose real location escapes dir are dropped, as are
// symlinked directories. The result is sorted and deduplicated across all
// listed subdirectories.
func skillFiles(dir string, subs ...string) ([]string, error) {
	var files []string
	seen := make(map[string]bool)
	for _, sub := range subs {
		root := filepath.Join(dir, sub)
		err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				if errors.Is(err, fs.ErrNotExist) {
					return fs.SkipAll // optional directory is absent
				}
				return err
			}
			if d.IsDir() {
				return nil
			}
			real, err := filepath.EvalSymlinks(path)
			if err != nil {
				return nil // unreadable entry: drop it
			}
			if info, err := os.Stat(real); err != nil || info.IsDir() {
				return nil // broken link or symlinked directory: skip
			}
			rel, err := filepath.Rel(dir, real)
			if err != nil || escapesDir(rel) || seen[rel] {
				return nil
			}
			seen[rel] = true
			files = append(files, filepath.ToSlash(rel))
			return nil
		})
		if err != nil && !errors.Is(err, fs.ErrNotExist) {
			return nil, err
		}
	}
	sort.Strings(files)
	return files, nil
}

// escapesDir reports whether rel, a path relative to a skill directory,
// points outside it.
func escapesDir(rel string) bool {
	return filepath.IsAbs(rel) || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

// WithSkills enables skills: it loads the skill folders found in the
// given skills roots — directories whose immediate children are skill
// folders — in precedence order (the first occurrence of a name wins).
// Passing a skill folder itself instead of a root is fail-closed like any
// other invalid input: it is reported as a SkillIssue by LoadSkills, and
// construction proceeds without it. This is a construction-time option:
// New consumes the directories once, appends the byte-stable
// "## Skills" catalog section to the system prompt, and registers the
// skill_read tool; passing it to Ask or AskStream has no effect. A scan
// root that cannot be read at all (for example permission-denied) fails
// construction.
func WithSkills(dirs ...string) Option {
	return func(c *config) { c.skillDirs = append(c.skillDirs, dirs...) }
}

// WithPinnedSkills inlines the full bodies of the named skills into the
// system prompt at construction — the sanctioned always-on exception to
// the load-on-demand discipline, for the couple of procedures every run
// needs. At most two names are allowed: New fails with an error wrapping
// ErrTooManyPinnedSkills when more are given, and with an error wrapping
// ErrInvalidInput when a name is not among the loaded skills (load them
// with WithSkills in the same construction). Like WithSkills this is a
// construction-time option; per-call use has no effect.
func WithPinnedSkills(names ...string) Option {
	return func(c *config) { c.pinnedNames = append(c.pinnedNames, names...) }
}

// resolveSkills performs the construction-time skill wiring: loading,
// catalog and pinned-body rendering, and skill_read registration. New
// calls it once, after all options have applied, so option order never
// matters and the rendered prompt is frozen for the Agent's lifetime.
func resolveSkills(c *config) error {
	if len(c.skillDirs) == 0 && len(c.pinnedNames) == 0 {
		return nil
	}
	skills, _, err := LoadSkills(c.skillDirs...)
	if err != nil {
		return err
	}
	if len(c.pinnedNames) > maxPinnedSkills {
		return fmt.Errorf("agentloop: %w: %d pinned skill names (at most %d)",
			ErrTooManyPinnedSkills, len(c.pinnedNames), maxPinnedSkills)
	}
	byName := make(map[string]*Skill, len(skills))
	for i := range skills {
		byName[skills[i].Name] = &skills[i]
	}
	pinned := make([]Skill, 0, len(c.pinnedNames))
	pinnedSeen := make(map[string]bool, len(c.pinnedNames))
	for _, name := range c.pinnedNames {
		if pinnedSeen[name] {
			continue
		}
		skill, ok := byName[name]
		if !ok {
			return fmt.Errorf("agentloop: %w: pinned skill %q is not among the loaded skills",
				ErrInvalidInput, name)
		}
		pinnedSeen[name] = true
		pinned = append(pinned, *skill)
	}
	if section := renderSkillCatalog(skills); section != "" {
		c.systemPrompt = joinPromptSection(c.systemPrompt, section)
	}
	if section := renderPinnedSkills(pinned); section != "" {
		c.systemPrompt = joinPromptSection(c.systemPrompt, section)
	}
	if len(skills) > 0 {
		c.tools = append(c.tools, newSkillReadTool(skills))
	}
	return nil
}

// joinPromptSection appends a rendered section to the system prompt,
// separated by a blank line.
func joinPromptSection(prompt, section string) string {
	if prompt == "" {
		return section
	}
	return prompt + "\n\n" + section
}

// renderSkillCatalog renders the L0 catalog section: one
// "- name: description" line per skill, sorted by name, so the bytes are
// deterministic and frozen at construction.
func renderSkillCatalog(skills []Skill) string {
	if len(skills) == 0 {
		return ""
	}
	sorted := append([]Skill(nil), skills...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].Name < sorted[j].Name })
	var b strings.Builder
	b.WriteString("## Skills\n\nProcedures you can load on demand. When one matches the task, read its full instructions with the skill_read tool before acting on it.\n")
	for _, s := range sorted {
		fmt.Fprintf(&b, "- %s: %s\n", s.Name, s.Description)
	}
	return b.String()
}

// renderPinnedSkills renders the pinned-skill section: each named skill's
// full body inlined in construction order.
func renderPinnedSkills(pinned []Skill) string {
	if len(pinned) == 0 {
		return ""
	}
	var b strings.Builder
	for _, s := range pinned {
		fmt.Fprintf(&b, "## Skill: %s (pinned)\n\n%s\n", s.Name, s.Body)
		if len(s.ScriptFiles) > 0 {
			fmt.Fprintf(&b, "\n%s\n", skillFooter(&s))
		}
	}
	return b.String()
}

// newSkillReadTool builds the skill_read tool over a frozen skill set.
// The lookup map is read-only, so the tool is safe for concurrent runs.
func newSkillReadTool(skills []Skill) Tool {
	byName := make(map[string]*Skill, len(skills))
	for i := range skills {
		byName[skills[i].Name] = &skills[i]
	}
	return Tool{
		Def: llmhub.NewTool(skillReadToolName,
			"Read a skill's full instructions, or one of its linked files. "+
				"Pass the skill name from the catalog; pass file only to read a linked file. "+
				"Reads are never paginated: the whole body or file is returned at once.",
			map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name": map[string]any{
						"type":        "string",
						"description": "skill name from the catalog",
					},
					"file": map[string]any{
						"type":        "string",
						"description": "optional linked file path, relative under references/ or assets/",
					},
				},
				"required": []string{"name"},
			}),
		Run: func(ctx context.Context, args json.RawMessage) (string, error) {
			return readSkill(byName, args)
		},
	}
}

// readSkill executes one skill_read call: the skill body (with the linked
// files listed) or one jailed linked-file read. Returned errors reach the
// model as in-band {"error": ...} observations and the loop continues.
func readSkill(byName map[string]*Skill, args json.RawMessage) (string, error) {
	var req struct {
		Name   string          `json:"name"`
		File   string          `json:"file"`
		Offset json.RawMessage `json:"offset"`
		Limit  json.RawMessage `json:"limit"`
	}
	if err := json.Unmarshal(args, &req); err != nil {
		return "", fmt.Errorf("skill_read: decode arguments: %w", err)
	}
	for _, param := range []struct {
		key string
		raw json.RawMessage
	}{{"offset", req.Offset}, {"limit", req.Limit}} {
		if jsonPresent(param.raw) {
			return "", fmt.Errorf("skill_read: %q is not supported: instructional reads are never paginated; the full text is returned at once", param.key)
		}
	}
	if req.Name == "" {
		return "", errors.New(`skill_read: "name" is required`)
	}
	skill, ok := byName[req.Name]
	if !ok {
		return "", fmt.Errorf("skill_read: unknown skill %q", req.Name)
	}
	if strings.TrimSpace(req.File) == "" {
		return skillBodyWithFiles(skill), nil
	}
	return readSkillFile(skill, req.File)
}

// jsonPresent reports whether a raw JSON argument slot carries a value
// (as opposed to being absent or explicitly null).
func jsonPresent(raw json.RawMessage) bool {
	s := strings.TrimSpace(string(raw))
	return s != "" && s != "null"
}

// skillBodyWithFiles renders the tool result for a body read: the full
// SKILL.md body, the linked-files list, and — for a skill that ships
// scripts — the disclosure footer, so the model never has to list the
// directory and can resolve skill-relative paths the body mentions.
func skillBodyWithFiles(s *Skill) string {
	var b strings.Builder
	b.WriteString(s.Body)
	if len(s.LinkedFiles) > 0 {
		b.WriteString("\n\nlinked files (pass as skill_read's file argument):\n- ")
		b.WriteString(strings.Join(s.LinkedFiles, "\n- "))
	}
	// Script-less skills keep their previous rendering byte-for-byte; the
	// footer exists to make body-referenced scripts resolvable, and there
	// is nothing to resolve without scripts/.
	if len(s.ScriptFiles) > 0 {
		b.WriteString("\n\n")
		b.WriteString(skillFooter(s))
	}
	return b.String()
}

// skillFooter renders the disclosure appended to a loaded skill body that
// ships scripts: the absolute base directory and the relative-path
// semantics, plus a run-don't-read listing of its scripts/ entries.
// Scripts stay unreadable through skill_read (L3): the listing is metadata
// for the execution tool, and only the script's output enters context.
func skillFooter(s *Skill) string {
	var b strings.Builder
	fmt.Fprintf(&b, "base directory: %s (relative paths in this skill resolve from here)", s.Dir)
	if len(s.ScriptFiles) > 0 {
		b.WriteString("\nscripts (execute via your execution tool from the base directory; do not read):\n- ")
		b.WriteString(strings.Join(s.ScriptFiles, "\n- "))
	}
	return b.String()
}

// readSkillFile reads one linked file under the skill's references/ or
// assets/ tree. The path is jailed: absolute paths, home paths, ".."
// elements, and symlink escapes are rejected (EvalSymlinks + Rel under
// the resolved skill directory), and only references/ and assets/ are
// readable — scripts (L3) run via an execution tool, never through here.
func readSkillFile(s *Skill, file string) (string, error) {
	if strings.HasPrefix(file, "~") {
		return "", errors.New(`skill_read: home paths are not allowed; pass a linked file path under references/ or assets/`)
	}
	if filepath.IsAbs(file) {
		return "", errors.New(`skill_read: absolute paths are not allowed; pass a linked file path under references/ or assets/`)
	}
	clean := filepath.Clean(file)
	if escapesDir(clean) {
		return "", errors.New(`skill_read: ".." escapes the skill directory`)
	}
	top := clean
	if i := strings.IndexAny(top, `/\`); i >= 0 {
		top = top[:i]
	}
	if top != "references" && top != "assets" {
		return "", errors.New(`skill_read: only files under references/ or assets/ can be read`)
	}
	real, err := filepath.EvalSymlinks(filepath.Join(s.Dir, clean))
	if err != nil {
		return "", fmt.Errorf("skill_read: %s: %w", file, err)
	}
	rel, err := filepath.Rel(s.Dir, real)
	if err != nil || escapesDir(rel) {
		return "", fmt.Errorf("skill_read: %s: path escapes the skill directory", file)
	}
	data, err := os.ReadFile(real)
	if err != nil {
		return "", fmt.Errorf("skill_read: %s: %w", file, err)
	}
	return string(data), nil
}
