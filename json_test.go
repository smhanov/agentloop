package agentloop

import (
	"encoding/json"
	"errors"
	"reflect"
	"testing"
)

func TestExtractJSONObject(t *testing.T) {
	cases := []struct {
		name    string
		text    string
		want    string
		wantErr bool
	}{
		{"plain", `{"a":1}`, `{"a":1}`, false},
		{"plain with whitespace", "{\n  \"a\": 1\n}", "{\n  \"a\": 1\n}", false},
		{"fenced json block", "```json\n{\"a\": 1}\n```", `{"a": 1}`, false},
		{"fenced without language", "```\n{\"a\": 1}\n```", `{"a": 1}`, false},
		{"fenced with prose", "Here you go:\n```json\n{\"ok\": true}\n```\nDone.", `{"ok": true}`, false},
		{"prose prefix and suffix", `The answer is {"a": 1} as requested.`, `{"a": 1}`, false},
		{"nested braces in strings", `{"code": "if (a) { return b; }", "n": 2}`, `{"code": "if (a) { return b; }", "n": 2}`, false},
		{"escaped quote with brace in string", `{"s": "a \"} brace"}`, `{"s": "a \"} brace"}`, false},
		{"first of several objects", `first {"a":1} then {"b":2}`, `{"a":1}`, false},
		{"array then object", `[1, 2] and {"a":1}`, `{"a":1}`, false},
		{"no json", "no objects here", "", true},
		{"unbalanced", `{"a": 1`, "", true},
		{"empty", "", "", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ExtractJSONObject(tc.text)
			if tc.wantErr {
				if !errors.Is(err, ErrNoJSON) {
					t.Fatalf("error = %v, want ErrNoJSON", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Errorf("got %q, want %q", got, tc.want)
			}
		})
	}
}

func TestExtractJSONObjectUnmarshals(t *testing.T) {
	text := "Sure!\n```json\n" + `{"name": "loop", "count": 3}` + "\n```"
	var v struct {
		Name  string `json:"name"`
		Count int    `json:"count"`
	}
	blob, err := ExtractJSONObject(text)
	if err != nil {
		t.Fatalf("ExtractJSONObject: %v", err)
	}
	if err := json.Unmarshal([]byte(blob), &v); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if v.Name != "loop" || v.Count != 3 {
		t.Errorf("decoded %+v", v)
	}
}

func TestFindBalancedJSONObjects(t *testing.T) {
	got := findBalancedJSONObjects(`a {"x": {"y": 1}} b {"z": 2} {oops`)
	want := []string{`{"x": {"y": 1}}`, `{"z": 2}`}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %q, want %q", got, want)
	}
}
