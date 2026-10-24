package agentloop

import "testing"

func TestLevenshtein(t *testing.T) {
	cases := []struct {
		a, b string
		want int
	}{
		{"", "", 0},
		{"", "abc", 3},
		{"abc", "abc", 0},
		{"kitten", "sitting", 3},
		{"flaw", "lawn", 2},
		{"clck", "clock", 1},
		{"clockx", "clock", 1},
	}
	for _, tc := range cases {
		if got := levenshtein(tc.a, tc.b); got != tc.want {
			t.Errorf("levenshtein(%q, %q) = %d, want %d", tc.a, tc.b, got, tc.want)
		}
	}
}

func TestSuggestName(t *testing.T) {
	registered := []string{"clock", "echo", "weather"}
	cases := []struct {
		name string
		want string
	}{
		{"clock", "clock"},    // exact match
		{"clck", "clock"},     // one deletion
		{"clockx", "clock"},   // one insertion
		{"cxyck", "clock"},    // two substitutions
		{"echoo", "echo"},     // closest of the set
		{"weathr", "weather"}, // one deletion
		{"zzzzzz", ""},        // too far
		{"", ""},              // empty name
	}
	for _, tc := range cases {
		if got := suggestName(tc.name, registered); got != tc.want {
			t.Errorf("suggestName(%q) = %q, want %q", tc.name, got, tc.want)
		}
	}
}

func TestUnknownToolObservation(t *testing.T) {
	if got, want := unknownToolJSON(""), `{"error":"unknown tool"}`; got != want {
		t.Errorf("got %q, want %q", got, want)
	}
	if got, want := unknownToolJSON("clock"), `{"error":"unknown tool; did you mean \"clock\"?"}`; got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}
