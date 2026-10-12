package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/patriceckhart/zot/packages/continuous"
	"github.com/patriceckhart/zot/packages/provider"
	"github.com/patriceckhart/zot/packages/provider/auth"
)

func TestContinuousInferenceScopesCredentialsAndEndpoint(t *testing.T) {
	t.Setenv("ZOT_HOME", t.TempDir())
	t.Setenv("ANTHROPIC_API_KEY", "synthetic-other")
	t.Setenv("ANTHROPIC_OAUTH_TOKEN", "")
	host := Resolved{Provider: "openai", Model: "gpt-4o-mini", Credential: "synthetic-host", AuthMethod: "apikey", BaseURL: "http://127.0.0.1:1", InsecureTLS: true}
	other, err := continuousInference(context.Background(), host, continuous.AgentConfig{Provider: "anthropic", Model: "claude-sonnet-4-5"})
	if err != nil {
		t.Fatal(err)
	}
	if other.Credential != "synthetic-other" || other.BaseURL == host.BaseURL || other.InsecureTLS {
		t.Fatal("host credential or endpoint settings leaked to another provider")
	}
	same, err := continuousInference(context.Background(), host, continuous.AgentConfig{Model: "gpt-4o"})
	if err != nil {
		t.Fatal(err)
	}
	if same.Credential != host.Credential || same.BaseURL != host.BaseURL || !same.InsecureTLS {
		t.Fatal("same-provider override lost host inference settings")
	}
	t.Setenv("ANTHROPIC_API_KEY", "")
	if _, err := continuousInference(context.Background(), host, continuous.AgentConfig{Provider: "anthropic", Model: "claude-sonnet-4-5"}); err == nil {
		t.Fatal("missing credentials accepted")
	}
	t.Setenv("DEEPSEEK_API_KEY", "")
	if _, err := continuousInference(context.Background(), host, continuous.AgentConfig{Provider: "deepseek", Model: "deepseek-v4-pro"}); err == nil {
		t.Fatal("catalog endpoint allowed a remote provider without credentials")
	}
	if _, err := continuousInference(context.Background(), host, continuous.AgentConfig{Provider: "unknown"}); err == nil {
		t.Fatal("unknown provider accepted")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := continuousInference(ctx, host, continuous.AgentConfig{}); err != context.Canceled {
		t.Fatalf("cancellation: %v", err)
	}
}

func TestContinuousAvailableModelsDoesNotExecuteCredentialCommands(t *testing.T) {
	home := t.TempDir()
	t.Setenv("ZOT_HOME", home)
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	t.Setenv("ZOT_AGENT_API_KEY_COMMAND_HELPER", "1")
	t.Setenv("DEEPSEEK_API_KEY", "")
	marker := filepath.Join(home, "command-ran")
	credentials := auth.Credentials{DeepSeek: auth.ProviderCreds{APIKeyCommand: &auth.APIKeyCommand{
		Program: os.Args[0], Args: []string{"-test.run=^TestAgentAPIKeyCommandHelperProcess$", "--", marker},
	}}}
	encoded, err := json.Marshal(credentials)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(AuthPath(), encoded, 0600); err != nil {
		t.Fatal(err)
	}
	previous := provider.ModelsForProvider("ollama")
	provider.SetManagedModelsForProvider("ollama", []provider.Model{{Provider: "ollama", ID: "routing-local"}})
	t.Cleanup(func() { provider.SetManagedModelsForProvider("ollama", previous) })
	models, err := continuousAvailableModels(context.Background(), Resolved{Provider: "openai"})
	if err != nil {
		t.Fatal(err)
	}
	found := map[string]bool{}
	for _, m := range models {
		found[m.Provider] = true
	}
	if !found["openai"] || !found["deepseek"] || !found["ollama"] {
		t.Fatal("host, local, or command-backed provider omitted")
	}
	if _, err := os.Stat(marker); !os.IsNotExist(err) {
		t.Fatal("credential command ran during model discovery")
	}
}

func TestContinuousInferenceDoesNotDowngradeFailedCredentialCommand(t *testing.T) {
	home := t.TempDir()
	t.Setenv("ZOT_HOME", home)
	const name = "routing-command"
	configs := provider.CustomProviders()
	previous, existed := configs[name]
	configs[name] = provider.CustomProviderConfig{API: "openai", BaseURL: "http://127.0.0.1:1"}
	t.Cleanup(func() {
		if existed {
			configs[name] = previous
		} else {
			delete(configs, name)
		}
	})
	creds := auth.Credentials{AdditionalAPIKeyCreds: map[string]auth.ProviderCreds{name: {
		APIKeyCommand: &auth.APIKeyCommand{Program: filepath.Join(home, "missing-command")},
	}}}
	encoded, err := json.Marshal(creds)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(AuthPath(), encoded, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := continuousInference(context.Background(), Resolved{Provider: "openai", Model: "gpt-4o-mini", Credential: "synthetic"}, continuous.AgentConfig{Provider: name, Model: "custom-model"}); err == nil {
		t.Fatal("failed credential command silently downgraded to keyless access")
	}
}

func TestContinuousCompactionUsesSelectedModel(t *testing.T) {
	host := Resolved{Provider: "openai", Model: "gpt-4o-mini"}
	config := continuous.AgentConfig{Provider: "anthropic", Model: "claude-sonnet-4-5"}
	policy := continuousCompactionFor(host)(config)
	model, err := provider.FindModel(config.Provider, config.Model)
	if err != nil {
		t.Fatal(err)
	}
	if policy.ContextWindow != model.ContextWindow || policy.ReserveTokens == 0 {
		t.Fatalf("incorrect model policy: %+v", policy)
	}
}

// TestContinuousInferenceDerivesCustomDefaultAfterRefresh checks that a
// provider-only AgentConfig for a discovery-enabled custom provider
// derives its default model after refreshing the catalog, so the global
// (anthropic) default id is never routed to a custom endpoint.
func TestContinuousInferenceDerivesCustomDefaultAfterRefresh(t *testing.T) {
	t.Setenv("ZOT_HOME", t.TempDir())
	t.Setenv("AAA_API_KEY", "")
	var requests atomic.Int32
	server := newModelsServer(t, &requests, "", "aaa-first", "aaa-second")
	loadTestModelsJSON(t, fmt.Sprintf(`{"providers": {"aaa": {"baseUrl": %q, "api": "openai", "discover": true}}}`, server.URL+"/v1"))
	host := Resolved{Provider: "openai", Model: "gpt-4o-mini", Credential: "synthetic", AuthMethod: "apikey"}

	other, err := continuousInference(context.Background(), host, continuous.AgentConfig{Provider: "aaa"})
	if err != nil {
		t.Fatal(err)
	}
	if other.Provider != "aaa" {
		t.Fatalf("provider = %q, want aaa", other.Provider)
	}
	if other.Model != "aaa-first" {
		t.Fatalf("model = %q, want aaa-first derived after the refresh", other.Model)
	}
	if other.Credential != "" {
		t.Fatalf("credential = %q, want empty (host credentials must not leak)", other.Credential)
	}
}

// TestContinuousInferenceDiscoveryFailureErrors mirrors resolve's
// refusal path: when continuousInference must derive the default model
// for a discovery-enabled provider and the refresh yields nothing, the
// routing fails instead of sending the global (anthropic) default id to
// the custom endpoint.
func TestContinuousInferenceDiscoveryFailureErrors(t *testing.T) {
	t.Setenv("ZOT_HOME", t.TempDir())
	t.Setenv("AAA_API_KEY", "")
	loadTestModelsJSON(t, `{"providers": {
		"aaa": {"baseUrl": "http://127.0.0.1:1/v1", "api": "openai", "discover": true}
	}}`)
	host := Resolved{Provider: "openai", Model: "gpt-4o-mini", Credential: "synthetic", AuthMethod: "apikey"}

	_, err := continuousInference(context.Background(), host, continuous.AgentConfig{Provider: "aaa"})
	if err == nil || !strings.Contains(err.Error(), "discover:true") {
		t.Fatalf("err = %v, want the no-discovered-models error", err)
	}
}
