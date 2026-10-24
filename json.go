package agentloop

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// ErrNoJSON reports that no balanced JSON object could be extracted from
// model text. It is returned (possibly wrapped) by ExtractJSONObject.
var ErrNoJSON = errors.New("agentloop: no JSON object found")

// ExtractJSONObject returns the first JSON object embedded in text,
// tolerating the ways models wrap structured output: markdown fences
// (```json … ```), prose before or after the object, and balanced braces
// inside string literals. When several objects are present the first valid
// one wins. The returned substring is valid JSON; the caller still
// unmarshals it into its target type. When nothing extractable is present
// the returned error wraps ErrNoJSON.
func ExtractJSONObject(text string) (string, error) {
	var candidates []string
	for _, block := range fencedBlocks(text) {
		candidates = append(candidates, findBalancedJSONObjects(block)...)
	}
	candidates = append(candidates, findBalancedJSONObjects(text)...)
	for _, candidate := range candidates {
		if json.Valid([]byte(candidate)) {
			return candidate, nil
		}
	}
	if len(candidates) > 0 {
		return "", fmt.Errorf("%w: %q is not valid JSON", ErrNoJSON, candidates[0])
	}
	return "", ErrNoJSON
}

// fencedBlocks returns the contents of markdown fenced code blocks, with or
// without an info string such as "json". An unterminated fence extends to
// the end of the text.
func fencedBlocks(text string) []string {
	var blocks []string
	for i := 0; i < len(text); {
		start := strings.Index(text[i:], "```")
		if start < 0 {
			return blocks
		}
		start += i
		lineEnd := strings.IndexByte(text[start:], '\n')
		if lineEnd < 0 {
			// No newline after the opening fence: no block can follow.
			return blocks
		}
		contentStart := start + lineEnd + 1
		end := strings.Index(text[contentStart:], "```")
		if end < 0 {
			blocks = append(blocks, text[contentStart:])
			return blocks
		}
		end += contentStart
		blocks = append(blocks, text[contentStart:end])
		i = end + 3
	}
	return blocks
}

// findBalancedJSONObjects extracts all top-level balanced { ... } substrings
// from text, taking JSON string literals and backslash escapes into account
// so braces inside strings never break the scan.
func findBalancedJSONObjects(s string) []string {
	var objects []string
	inString := false
	escaped := false
	depth := 0
	startIdx := -1

	for i := 0; i < len(s); i++ {
		b := s[i]
		if inString {
			if escaped {
				escaped = false
			} else if b == '\\' {
				escaped = true
			} else if b == '"' {
				inString = false
			}
			continue
		}

		switch b {
		case '"':
			inString = true
		case '{':
			if depth == 0 {
				startIdx = i
			}
			depth++
		case '}':
			if depth > 0 {
				depth--
				if depth == 0 && startIdx != -1 {
					objects = append(objects, s[startIdx:i+1])
					startIdx = -1
				}
			}
		}
	}
	return objects
}
