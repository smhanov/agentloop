package agentloop

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/smhanov/llmhub"
	"gopkg.in/yaml.v3"
)

// Skill-file conventions and validation limits.
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
	// Dir is the skill folder: the resolved absolute path for skills loaded
	// from disk with LoadSkills/WithSkills, or the slash path relative to
	// the filesystem root for skills loaded from an fs.FS with
	// LoadSkillsFS/WithSkillsFS (for example "skills/my-skill").
	Dir string
	// LinkedFiles lists the skill's resource files: relative slash paths
	// under references/ and assets/, precomputed and sorted so the model
	// never needs to list the directory. For disk roots symlinks are
	// resolved and escapes dropped; for fs.FS sources paths are lexical and
	// symlink-typed entries are skipped. Scripts under scripts/ are
	// deliberately not listed:
	// they run via an execution tool and only their output enters
	// context (L3).
	LinkedFiles []string
	// ScriptFiles lists the skill's executable helpers: relative slash
	// paths under scripts/, precomputed like LinkedFiles and sorted so the
	// body can reference them. They are advertised for execution, never for
	// reading: run them through the agent's execution tool from Dir and only
	// their output enters context (L3). Executing scripts that live inside
	// an fs.FS (an embed.FS, for example) is the caller's concern — extract
	// or mount them first.
	ScriptFiles []string
	// backend is the filesystem the skill was loaded from; nil means the
	// OS. It stays unexported so hand-built Skill values keep working and
	// callers cannot inject arbitrary backends into skill_read.
	backend skillBackend
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

// skillBackend abstracts the filesystem operations the skill loader
// performs, so LoadSkills and LoadSkillsFS share one validation path. Paths
// are in the backend's own namespace: native OS paths for osBackend, slash
// paths rooted at the fs.FS for fsBackend.
type skillBackend interface {
	readDir(dir string) ([]fs.DirEntry, error)
	statFile(path string) (fs.FileInfo, error)
	readFile(path string) ([]byte, error)
	listFiles(dir string, subs []string) ([]string, error)
	base(path string) string
}

// osPathSemantics is the optional capability of backends that live in the
// OS path namespace: roots are made absolute before scanning and directory
// symlinks are resolved. An fs.FS namespace has no absolute paths and
// io/fs offers no symlink resolution, so fsBackend does not implement it.
type osPathSemantics interface {
	absRoot(root string) (string, error)
	resolveDir(path string) (string, error)
}

// osBackend loads skills from directories on disk, with the symlink
// resolution and jail the disk loader has always used.
type osBackend struct{}

func (osBackend) absRoot(root string) (string, error)    { return filepath.Abs(root) }
func (osBackend) resolveDir(path string) (string, error) { return filepath.EvalSymlinks(path) }
func (osBackend) readDir(dir string) ([]fs.DirEntry, error) {
	return os.ReadDir(dir)
}
func (osBackend) statFile(path string) (fs.FileInfo, error) { return os.Stat(path) }
func (osBackend) readFile(path string) ([]byte, error)      { return os.ReadFile(path) }
func (osBackend) base(path string) string                   { return filepath.Base(path) }

// listFiles is the disk walk behind linkedFiles and scriptFiles: every
// regular file under the given optional subdirectories of dir, as slash
// paths relative to dir. Symlinks are resolved; files whose real location
// escapes dir are dropped, as are symlinked directories. The result is
// sorted and deduplicated across all listed subdirectories.
func (osBackend) listFiles(dir string, subs []string) ([]string, error) {
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

// fsBackend loads skills from any fs.FS: an embed.FS, os.DirFS, a
// fstest.MapFS, or a custom filesystem. Paths are slash paths rooted at the
// filesystem; io/fs offers no symlink resolution, so the jail is lexical
// and symlink-typed entries are skipped.
type fsBackend struct{ fsys fs.FS }

func (b fsBackend) readDir(dir string) ([]fs.DirEntry, error) {
	return fs.ReadDir(b.fsys, dir)
}
func (b fsBackend) statFile(path string) (fs.FileInfo, error) { return fs.Stat(b.fsys, path) }
func (b fsBackend) readFile(path string) ([]byte, error)      { return fs.ReadFile(b.fsys, path) }
func (b fsBackend) base(p string) string                      { return path.Base(p) }

// listFiles walks the fs.FS counterparts of osBackend.listFiles. Missing
// optional directories are fine. io/fs has no ReadLink, so an entry that
// reports itself as a symlink cannot be resolved and is skipped rather
// than followed; an entry that reports as a regular file is one.
func (b fsBackend) listFiles(dir string, subs []string) ([]string, error) {
	var files []string
	seen := make(map[string]bool)
	for _, sub := range subs {
		root := path.Join(dir, sub)
		err := fs.WalkDir(b.fsys, root, func(p string, d fs.DirEntry, err error) error {
			if err != nil {
				if errors.Is(err, fs.ErrNotExist) {
					return fs.SkipAll // optional directory is absent
				}
				return err
			}
			if d.IsDir() {
				return nil
			}
			if d.Type()&fs.ModeSymlink != 0 {
				return nil // no ReadLink in io/fs: skip, never follow
			}
			info, err := fs.Stat(b.fsys, p)
			if err != nil || info.IsDir() {
				return nil // unreadable entry or directory entry: skip
			}
			rel, ok := slashRel(dir, p)
			if !ok || escapesSlashDir(rel) || seen[rel] {
				return nil
			}
			seen[rel] = true
			files = append(files, rel)
			return nil
		})
		if err != nil && !errors.Is(err, fs.ErrNotExist) {
			return nil, err
		}
	}
	sort.Strings(files)
	return files, nil
}

// slashRel returns p relative to dir for slash paths, and whether p is
// inside dir. The special directory "." makes p its own relative path.
func slashRel(dir, p string) (string, bool) {
	dir = path.Clean(dir)
	if dir == "." {
		return p, true
	}
	prefix := dir + "/"
	if !strings.HasPrefix(p, prefix) {
		return "", false
	}
	return p[len(prefix):], true
}

// escapesSlashDir reports whether rel, a slash path relative to a skill
// directory, points outside it.
func escapesSlashDir(rel string) bool {
	return strings.HasPrefix(rel, "/") || rel == ".." || strings.HasPrefix(rel, "../")
}

// skillJoin joins path elements in the backend's namespace.
func skillJoin(backend skillBackend, elems ...string) string {
	if _, ok := backend.(fsBackend); ok {
		return path.Join(elems...)
	}
	return filepath.Join(elems...)
}

// skillSource is one backend plus the roots to scan from it. WithSkills and
// WithSkillsFS append sources in option order, which is also precedence
// order.
type skillSource struct {
	backend skillBackend
	roots   []string
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
//
// LoadSkillsFS loads the same packs from an fs.FS. WithSkills and
// WithSkillsFS compose in option order; when both are configured, the
// first occurrence of a name across the ordered sources wins.
func LoadSkills(dirs ...string) ([]Skill, []SkillIssue, error) {
	return loadSkills([]skillSource{{backend: osBackend{}, roots: dirs}})
}

// LoadSkillsFS is LoadSkills for any fs.FS — an embed.FS, os.DirFS, a
// fstest.MapFS, or a custom filesystem — so a skill pack can ship inside
// the binary without a writable temporary directory. It applies the same
// validation, fail-closed issues, catalog fields, pinning, and skill_read
// behavior as LoadSkills.
//
// Roots are slash paths relative to the filesystem root, not OS paths:
// fs.ValidPath governs them, so a leading slash or a ".." element is
// invalid (pass "." to scan the filesystem root). A root that is
// missing or not a directory yields an issue, not an error, exactly like
// the disk loader. There is no ReadLink in io/fs, so the loader has no
// symlink resolution to perform: the linked-file jail is lexical
// (fs.ValidPath plus a references/ or assets/ prefix), and entries that
// report themselves as symlinks are skipped during scans rather than
// followed. Entry types follow the underlying filesystem, so an os.DirFS
// tree behaves differently from the same tree passed to WithSkills (which
// resolves symlinks and drops escaping targets). Use WithSkills for disk
// roots that rely on symlinks.
//
// Skill.Dir is the fs-relative slash path of the skill folder. Scripts
// under a skill's scripts/ directory are listed for the execution tool but
// are not readable through skill_read; executing embedded scripts is the
// caller's concern (extract or mount them first).
func LoadSkillsFS(fsys fs.FS, roots ...string) ([]Skill, []SkillIssue, error) {
	if fsys == nil {
		return nil, nil, fmt.Errorf("agentloop: load skills: %w: nil filesystem", ErrInvalidInput)
	}
	return loadSkills([]skillSource{{backend: fsBackend{fsys: fsys}, roots: roots}})
}

// loadSkills is the shared loader behind LoadSkills and LoadSkillsFS. Each
// source is scanned in order; the seen map spans all sources, so the first
// occurrence of a skill name wins across backends.
func loadSkills(sources []skillSource) ([]Skill, []SkillIssue, error) {
	var (
		skills []Skill
		issues []SkillIssue
		seen   = make(map[string]bool)
	)
	for _, source := range sources {
		backend := source.backend
		for _, root := range source.roots {
			rootPath := root
			if ops, ok := backend.(osPathSemantics); ok {
				absRoot, err := ops.absRoot(root)
				if err != nil {
					return skills, issues, fmt.Errorf("agentloop: load skills: resolve %q: %w", root, err)
				}
				rootPath = absRoot
			}
			entries, err := backend.readDir(rootPath)
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
			if info, err := backend.statFile(skillJoin(backend, rootPath, skillFileName)); err == nil && !info.IsDir() {
				issues = append(issues, SkillIssue{
					Dir:    rootPath,
					Name:   backend.base(rootPath),
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
				childPath := skillJoin(backend, rootPath, name)
				if _, ok := backend.(fsBackend); ok {
					if entry := byName[name]; entry != nil && entry.Type()&fs.ModeSymlink != 0 {
						// io/fs cannot resolve symlinks; skip a
						// symlink-typed child rather than follow it.
						continue
					}
				}
				info, err := backend.statFile(childPath) // OS stat follows symlinks to directories
				if err != nil {
					issues = append(issues, SkillIssue{Dir: childPath, Name: backend.base(childPath), Reason: issueUnreadableDir})
					continue
				}
				if !info.IsDir() {
					continue // an ordinary file is not a skill attempt
				}
				resolved := childPath
				if ops, ok := backend.(osPathSemantics); ok {
					resolved, err = ops.resolveDir(childPath)
					if err != nil {
						issues = append(issues, SkillIssue{Dir: childPath, Name: backend.base(childPath), Reason: issueUnreadableDir})
						continue
					}
				}
				skill, skillIssues, ok := loadOneSkill(backend, resolved)
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
	}
	return skills, issues, nil
}

// loadOneSkill validates one skill directory. It returns the skill with
// ok=true, or ok=false with the accumulated issues explaining the
// rejection (which may include a sanitization advisory recorded before a
// later validation failure).
func loadOneSkill(backend skillBackend, dir string) (Skill, []SkillIssue, bool) {
	var issues []SkillIssue
	fail := func(reason string) (Skill, []SkillIssue, bool) {
		return Skill{}, append(issues, SkillIssue{Dir: dir, Name: backend.base(dir), Reason: reason}), false
	}

	raw, err := backend.readFile(skillJoin(backend, dir, skillFileName))
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
	if name != backend.base(dir) {
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
	linked, err := linkedFiles(backend, dir)
	if err != nil {
		return fail(issueUnreadableDir)
	}
	scripts, err := scriptFiles(backend, dir)
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
		backend:     backend,
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
// directory. For disk roots symlinks are resolved and files whose real
// location escapes the skill directory are dropped, as are symlinked
// directories; an fs.FS has no symlinks, so its paths are lexical. The
// result is sorted and deduplicated. Missing resource directories are fine
// (they are optional); other walk errors are systemic and returned.
func linkedFiles(backend skillBackend, dir string) ([]string, error) {
	return backend.listFiles(dir, []string{"references", "assets"})
}

// scriptFiles lists the skill's executable helpers: every regular file
// under scripts/, with the same resolution rules as linkedFiles. Missing
// scripts/ is fine; other walk errors are systemic and returned, so an
// unreadable scripts tree fails the skill rather than silently hiding
// runnable helpers.
func scriptFiles(backend skillBackend, dir string) ([]string, error) {
	return backend.listFiles(dir, []string{"scripts"})
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
// construction proceeds without it. The recorded issues stay observable
// afterwards through Agent.SkillIssues (see the README's Skills section,
// "Diagnosing dropped skills"). This is a construction-time option:
// New consumes the directories once, appends the byte-stable
// "## Skills" catalog section to the system prompt, and registers the
// skill_read tool; passing it to Ask or AskStream has no effect. A scan
// root that cannot be read at all (for example permission-denied) fails
// construction.
//
// WithSkills composes with WithSkillsFS in option order: sources are
// scanned in the order their options were applied, and the first
// occurrence of a skill name across all of them wins.
func WithSkills(dirs ...string) Option {
	return func(c *config) {
		if len(dirs) == 0 {
			return
		}
		c.skillSources = append(c.skillSources, skillSource{
			backend: osBackend{},
			roots:   append([]string(nil), dirs...),
		})
	}
}

// WithSkillsFS is WithSkills for any fs.FS — an embed.FS, os.DirFS, a
// fstest.MapFS, or a custom filesystem — so a skill pack can ship inside
// the binary. It loads the skill folders found in the given roots in
// precedence order and otherwise behaves exactly like WithSkills: the
// same validation, fail-closed SkillIssues on Agent.SkillIssues, catalog
// in the system prompt, and skill_read tool.
//
// Roots are slash paths relative to the filesystem root, not OS paths:
// fs.ValidPath governs them (no leading slash, no "." or ".." elements;
// pass "." to scan the filesystem root). There are no symlinks in an
// fs.FS (io/fs has no ReadLink), so linked-file reads are jailed lexically
// rather than with EvalSymlinks and entries that report themselves as
// symlinks are skipped during scans. An os.DirFS tree therefore behaves
// differently from the same tree passed to WithSkills; use WithSkills for
// disk roots that rely on symlinks. Skill.Dir is the fs-relative slash
// path of the skill folder. Executing
// scripts embedded in the filesystem is the caller's concern: skill_read
// lists them but never reads them, so extract or mount them before
// running them through an execution tool. A nil filesystem or an empty
// root list is a no-op.
//
// WithSkillsFS composes with WithSkills in option order: sources are
// scanned in the order their options were applied, and the first
// occurrence of a skill name across all of them wins. Like WithSkills this
// is a construction-time option; per-call use has no effect.
func WithSkillsFS(fsys fs.FS, roots ...string) Option {
	return func(c *config) {
		if fsys == nil || len(roots) == 0 {
			return
		}
		c.skillSources = append(c.skillSources, skillSource{
			backend: fsBackend{fsys: fsys},
			roots:   append([]string(nil), roots...),
		})
	}
}

// WithPinnedSkills inlines the full bodies of the named skills into the
// system prompt at construction — the sanctioned always-on exception to
// the load-on-demand discipline, for the couple of procedures every run
// needs. At most two names are allowed: New fails with an error wrapping
// ErrTooManyPinnedSkills when more are given, and with an error wrapping
// ErrInvalidInput when a name is not among the loaded skills (load them
// with WithSkills or WithSkillsFS in the same construction). Like WithSkills this is a
// construction-time option; per-call use has no effect.
func WithPinnedSkills(names ...string) Option {
	return func(c *config) { c.pinnedNames = append(c.pinnedNames, names...) }
}

// WithRequireSkills makes construction fail when the resolved skill
// catalog is empty: New returns an error wrapping ErrNoSkillsLoaded when
// WithSkills/WithSkillsFS and WithPinnedSkills together load zero skills — no roots
// configured at all, roots that exist but contain no valid skill folder,
// or roots whose skills all failed validation. It is the boot-time check
// for a consumer that expects skills, for example when the roots come
// from an environment variable that may be unset; without it an empty
// catalog is legal and the agent runs skillless.
//
// This is a construction-time option: New resolves skills once, so
// passing it to Ask or AskStream has no effect. Agent.Skills reports the
// loaded catalog whether or not the requirement is set.
func WithRequireSkills() Option {
	return func(c *config) { c.requireSkills = true }
}

// resolveSkills performs the construction-time skill wiring: loading,
// catalog and pinned-body rendering, and skill_read registration. It
// returns the loaded skills and the issues LoadSkills recorded, for
// observability through Agent.Skills and Agent.SkillIssues. New calls it
// once, after all options have applied, so option order never matters and
// the rendered prompt is frozen for the Agent's lifetime.
func resolveSkills(c *config) ([]Skill, []SkillIssue, error) {
	if len(c.skillSources) == 0 && len(c.pinnedNames) == 0 && !c.requireSkills {
		return nil, nil, nil
	}
	skills, issues, err := loadSkills(c.skillSources)
	if err != nil {
		return nil, nil, err
	}
	if c.requireSkills && len(skills) == 0 {
		return nil, nil, fmt.Errorf("%w: %d configured root(s) yielded no valid skills",
			ErrNoSkillsLoaded, skillRootCount(c.skillSources))
	}
	if len(c.pinnedNames) > maxPinnedSkills {
		return nil, nil, fmt.Errorf("agentloop: %w: %d pinned skill names (at most %d)",
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
			return nil, nil, fmt.Errorf("agentloop: %w: pinned skill %q is not among the loaded skills",
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
	return skills, issues, nil
}

// skillRootCount returns the number of configured scan roots across all
// sources, for the boot-check error prose.
func skillRootCount(sources []skillSource) int {
	n := 0
	for _, source := range sources {
		n += len(source.roots)
	}
	return n
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
// ships scripts: the base directory (absolute for disk roots, fs-relative
// for embedded packs) and the relative-path semantics, plus a
// run-don't-read listing of its scripts/ entries.
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
// assets/ tree. Disk-loaded skills use the OS jail below (EvalSymlinks +
// Rel); skills loaded from an fs.FS use the lexical jail. Either way,
// absolute paths, home paths, ".." elements, and paths outside
// references/ or assets/ are rejected — scripts (L3) run via an execution
// tool, never through here.
func readSkillFile(s *Skill, file string) (string, error) {
	if backend, ok := s.backend.(fsBackend); ok {
		return readSkillFileFS(s, backend, file)
	}
	return readSkillFileOS(s, file)
}

// readSkillFileOS is the disk jail: absolute paths, home paths, ".."
// elements, and symlink escapes are rejected (EvalSymlinks + Rel under the
// resolved skill directory), and only references/ and assets/ are
// readable.
func readSkillFileOS(s *Skill, file string) (string, error) {
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

// readSkillFileFS is the lexical jail for fs.FS sources. The path must be
// a valid fs path (slash-separated, no leading slash, no "." or ".."
// elements) under references/ or assets/; an fs.FS has no symlinks, so a
// separate EvalSymlinks step has nothing to resolve.
func readSkillFileFS(s *Skill, backend fsBackend, file string) (string, error) {
	if strings.HasPrefix(file, "~") {
		return "", errors.New(`skill_read: home paths are not allowed; pass a linked file path under references/ or assets/`)
	}
	if strings.HasPrefix(file, "/") {
		return "", errors.New(`skill_read: absolute paths are not allowed; pass a linked file path under references/ or assets/`)
	}
	clean := path.Clean(file)
	if escapesSlashDir(clean) {
		return "", errors.New(`skill_read: ".." escapes the skill directory`)
	}
	if !fs.ValidPath(clean) {
		return "", fmt.Errorf("skill_read: %s: not a valid filesystem path", file)
	}
	top := clean
	if i := strings.IndexByte(top, '/'); i >= 0 {
		top = top[:i]
	}
	if top != "references" && top != "assets" {
		return "", errors.New(`skill_read: only files under references/ or assets/ can be read`)
	}
	data, err := backend.readFile(path.Join(s.Dir, clean))
	if err != nil {
		return "", fmt.Errorf("skill_read: %s: %w", file, err)
	}
	return string(data), nil
}
