package agent

import (
	"os"
	"path/filepath"
	"testing"
)

func TestZotHomePrefersExplicitHomeOverXDGStateHome(t *testing.T) {
	t.Setenv("ZOT_HOME", filepath.Join("explicit", "zot"))
	t.Setenv("XDG_STATE_HOME", filepath.Join("xdg", "state"))

	if got, want := ZotHome(), filepath.Join("explicit", "zot"); got != want {
		t.Fatalf("ZotHome() = %q, want %q", got, want)
	}
}

func TestZotHomeUsesXDGStateHome(t *testing.T) {
	t.Setenv("ZOT_HOME", "")
	t.Setenv("XDG_STATE_HOME", filepath.Join("xdg", "state"))

	if got, want := ZotHome(), filepath.Join("xdg", "state", "zot"); got != want {
		t.Fatalf("ZotHome() = %q, want %q", got, want)
	}
}

func TestZotHomeDefaultsToExecutableDirectory(t *testing.T) {
	t.Setenv("ZOT_HOME", "")
	t.Setenv("XDG_STATE_HOME", "")

	exe, err := os.Executable()
	if err != nil {
		t.Skipf("os.Executable() unavailable: %v", err)
	}
	want := filepath.Dir(exe)

	if got := ZotHome(); got != want {
		t.Fatalf("ZotHome() = %q, want executable dir %q", got, want)
	}
}
