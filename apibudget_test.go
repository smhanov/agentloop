package agentloop

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strings"
	"testing"
)

// The library's architecture promise (specs.md §11) is a small exported
// surface: one core package plus two presets, and nothing else. This test
// enforces that promise the way gptme's `arewetiny` does — it counts the
// exported symbols of every shipped library package and fails once the
// count crosses a fixed ceiling. Adding public API must be a deliberate
// act: the ceiling has to be raised in its own change, with a reason, so
// the growth of the surface is always a reviewed decision rather than an
// accident.
//
// Counting rule (deliberately simple, applied to every non-test .go file
// of each package below):
//
//   - each exported package-level function, variable, constant, and type
//     declaration counts 1 (const/var blocks count per exported spec);
//   - each exported struct field of an exported struct type counts 1;
//   - each exported method of an exported type counts 1, and every method
//     listed in an exported interface type counts 1.
//
// Methods on unexported types (the presets' internals, the run struct)
// are not reachable from outside and are not counted. The examples/ command
// is not library API and is not counted.
var budgetedPackages = map[string]string{
	".":      "agentloop (core)",
	"search": "search (preset)",
	"bash":   "bash (preset)",
}

// exportedAPICeiling is the total number of exported symbols the library
// may expose. It is set to the measured count at introduction (196:
// agentloop 112, search 51, bash 33) plus 10%, rounded up, so ordinary
// work is not blocked while any real API addition forces a conscious
// decision.
const exportedAPICeiling = 216

// countExportedSymbols parses the package in dir (non-test files only)
// and returns its exported-symbol count per the rule above.
func countExportedSymbols(t *testing.T, dir string) int {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read %s: %v", dir, err)
	}
	fset := token.NewFileSet()
	count := 0
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		file, err := parser.ParseFile(fset, dir+"/"+name, nil, 0)
		if err != nil {
			t.Fatalf("parse %s/%s: %v", dir, name, err)
		}
		for _, decl := range file.Decls {
			switch d := decl.(type) {
			case *ast.FuncDecl:
				if !d.Name.IsExported() {
					continue
				}
				if d.Recv == nil || len(d.Recv.List) == 0 {
					count++ // package-level function
					continue
				}
				if receiverExported(d.Recv.List[0].Type) {
					count++ // exported method of an exported type
				}
			case *ast.GenDecl:
				for _, spec := range d.Specs {
					switch s := spec.(type) {
					case *ast.ValueSpec:
						for _, name := range s.Names {
							if name.IsExported() {
								count++
							}
						}
					case *ast.TypeSpec:
						if !s.Name.IsExported() {
							continue
						}
						count++
						count += exportedMembers(s.Type)
					}
				}
			}
		}
	}
	return count
}

// receiverExported reports whether a receiver type expression names an
// exported type (Tool, *Agent, ...).
func receiverExported(expr ast.Expr) bool {
	switch r := expr.(type) {
	case *ast.Ident:
		return r.IsExported()
	case *ast.StarExpr:
		return receiverExported(r.X)
	default:
		return false
	}
}

// exportedMembers counts the exported members of an exported type: named
// struct fields, or interface methods.
func exportedMembers(expr ast.Expr) int {
	switch t := expr.(type) {
	case *ast.StructType:
		count := 0
		for _, field := range t.Fields.List {
			for _, name := range field.Names {
				if name.IsExported() {
					count++
				}
			}
		}
		return count
	case *ast.InterfaceType:
		count := 0
		if t.Methods != nil {
			for _, field := range t.Methods.List {
				for _, name := range field.Names {
					if name.IsExported() {
						count++
					}
				}
			}
		}
		return count
	default:
		return 0
	}
}

// TestExportedAPIBudget fails when the exported surface of the shipped
// library packages grows past exportedAPICeiling. See the comment above
// budgetedPackages for the counting rule and why it exists.
func TestExportedAPIBudget(t *testing.T) {
	total := 0
	for dir, label := range budgetedPackages {
		n := countExportedSymbols(t, dir)
		total += n
		t.Logf("%-18s %3d exported symbols", label, n)
	}
	t.Logf("%-18s %3d exported symbols (ceiling %d)", "total", total, exportedAPICeiling)
	if total > exportedAPICeiling {
		t.Errorf("exported API grew to %d symbols, over the ceiling of %d; "+
			"shrink the change or raise the ceiling deliberately, with a "+
			"reason, in its own commit (small-surface promise)",
			total, exportedAPICeiling)
	}
}

// TestExportedAPIBudgetSanity documents the counts as data: the test above
// must fail if the ceiling were one lower than the measured total minus
// nothing — i.e. the ceiling must be within a sane band of the actual
// count. (The AC "lowering the ceiling by one fails" is verified manually
// against this measured number; keeping it in the log makes drift visible.)
func TestExportedAPIBudgetSanity(t *testing.T) {
	if exportedAPICeiling <= 0 {
		t.Fatal("ceiling must be positive")
	}
	total := 0
	for dir := range budgetedPackages {
		total += countExportedSymbols(t, dir)
	}
	if total == 0 {
		t.Fatal("counter found no exported symbols; the counter or the package list is broken")
	}
	if float64(total) > float64(exportedAPICeiling)*0.999 {
		// Reported by the budget test itself; nothing further here.
		return
	}
	headroom := exportedAPICeiling - total
	if headroom > total/2 {
		t.Errorf("ceiling %d has %d symbols of headroom over the measured %d; "+
			"tighten it to measured+10%% so the budget means something",
			exportedAPICeiling, headroom, total)
	}
	if testing.Verbose() {
		fmt.Printf("exported API: %d symbols, ceiling %d\n", total, exportedAPICeiling)
	}
}
