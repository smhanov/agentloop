package agentloop

import "testing"

func TestWithSettingAndSettings(t *testing.T) {
	opts := []Option{
		WithSetting("preset.key", "v1"),
		WithMaxToolCalls(3),
		WithSetting("preset.other", 42),
		WithSetting("preset.key", "v2"), // later wins
	}
	got := Settings(opts...)
	if got["preset.key"] != "v2" || got["preset.other"] != 42 {
		t.Errorf("Settings = %v, want {preset.key: v2, preset.other: 42}", got)
	}
	if Settings() != nil {
		t.Error("Settings() with no options = non-nil, want nil")
	}

	// Settings must not leak between calls: the throwaway config is not
	// retained anywhere.
	if again := Settings(opts...); again["preset.key"] != "v2" {
		t.Errorf("repeated Settings = %v", again)
	}
}
