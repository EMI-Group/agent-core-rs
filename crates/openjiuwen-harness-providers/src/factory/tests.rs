//! Unit tests for the provider registry and the harness-context builder.
//!
//! The registry is process-global, so every test registers its providers under a fresh
//! `uuid`-derived name instead of resetting the registry; the tests therefore stay independent
//! under the parallel test harness.

// Tests assert by panicking on failure, which is the intent of a test.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_harness_protocol::{
    CheckpointReason, CheckpointSaveReceipt, HarnessCapability, HarnessCard, HarnessCheckpoint,
    HarnessCheckpointSink, HarnessError, HarnessProtocol, HarnessResult, HostCapability,
    JsonObject, JsonValue, McpServerConfig, McpTransport, ModelChoice, ModelEndpoint, ResumePolicy,
};
// Only the native-feature tests construct a real harness.
#[cfg(feature = "native")]
use openjiuwen_harness_protocol::HarnessState;

#[cfg(feature = "native")]
use super::create_harness;
use super::{
    available_providers, build_harness_context, describe_provider, provider_descriptors,
    provider_model_config, register_harness_provider, resolve_provider,
    unregister_harness_provider, HarnessContextOptions, ProviderFactory, PROVIDER_NAMES,
};
use crate::provider::{provider_names, ProviderSpec};

/// Returns a registry name no other test (or earlier run) can collide with.
fn unique_name(prefix: &str) -> String {
    format!("factory-test-{prefix}-{}", uuid::Uuid::new_v4())
}

/// Fake provider: an explicit card plus the model-hook switch.
struct FakeSpec {
    /// Card reported by the spec.
    card: HarnessCard,
    /// Whether [`ProviderSpec::configures_models`] reports `true`.
    models: bool,
}

impl FakeSpec {
    /// Builds a model-configuring fake reporting `card_name`.
    fn new(card_name: &str) -> Self {
        Self {
            card: HarnessCard::new(card_name, "0.0.1"),
            models: true,
        }
    }

    /// Declares `capabilities` on the card.
    fn with_capabilities(mut self, capabilities: &[HarnessCapability]) -> Self {
        self.card.capabilities = capabilities.iter().copied().collect();
        self
    }

    /// Turns the model-configuration hook off.
    fn without_models(mut self) -> Self {
        self.models = false;
        self
    }
}

impl ProviderSpec for FakeSpec {
    fn card(&self) -> &HarnessCard {
        &self.card
    }

    fn create(&self, _config: JsonObject) -> HarnessResult<Box<dyn HarnessProtocol>> {
        Err(HarnessError::validation(
            "the fake provider cannot create a harness",
        ))
    }

    fn configures_models(&self) -> bool {
        self.models
    }

    fn model_config(
        &self,
        choice: &ModelChoice,
        fallback: Option<&ModelChoice>,
    ) -> HarnessResult<JsonObject> {
        let mut fragment = JsonObject::new();
        if let Some(model) = &choice.model {
            fragment.insert("model".to_owned(), JsonValue::String(model.clone()));
        }
        if let Some(fallback) = fallback {
            fragment.insert(
                "fallback".to_owned(),
                JsonValue::String(fallback.model.clone().unwrap_or_default()),
            );
        }
        Ok(fragment)
    }
}

/// Wraps `spec` into a factory that always succeeds.
fn fake_factory(spec: Arc<dyn ProviderSpec>) -> ProviderFactory {
    Arc::new(move || Ok(Arc::clone(&spec)))
}

/// Registers `spec` under a fresh unique name and returns that name.
fn register_fake(spec: Arc<dyn ProviderSpec>) -> String {
    let name = unique_name("provider");
    register_harness_provider(&name, fake_factory(spec))
        .expect("a unique non-built-in name is registrable");
    name
}

/// Returns the position of `name` in [`available_providers`], if it is listed.
fn position(name: &str) -> Option<usize> {
    available_providers()
        .iter()
        .position(|candidate| candidate == name)
}

/// Checkpoint sink that refuses every write; the tests only need the trait object.
struct FakeSink;

#[async_trait]
impl HarnessCheckpointSink for FakeSink {
    async fn save(
        &self,
        _checkpoint: &HarnessCheckpoint,
        _reason: CheckpointReason,
        _expected_storage_revision: Option<&str>,
    ) -> HarnessResult<CheckpointSaveReceipt> {
        Err(HarnessError::unsupported_capability(
            "the fake sink does not persist checkpoints",
        ))
    }
}

/// One stdio MCP server configuration.
fn mcp_server(name: &str) -> McpServerConfig {
    McpServerConfig {
        name: name.to_owned(),
        transport: McpTransport::Stdio,
        command: vec!["mcp-server".to_owned()],
        url: None,
        instance: None,
        env: BTreeMap::new(),
        headers: BTreeMap::new(),
    }
}

#[test]
fn empty_and_builtin_names_are_rejected() {
    for name in ["", provider_names::NATIVE, provider_names::DSH] {
        let error = register_harness_provider(name, fake_factory(Arc::new(FakeSpec::new("x"))))
            .expect_err("empty and built-in names are not registrable");
        assert!(
            matches!(error, Error::Validation(_)),
            "unexpected error for {name}: {error}"
        );
    }
    let error = register_harness_provider("", fake_factory(Arc::new(FakeSpec::new("x"))))
        .expect_err("the empty name is not registrable");
    assert_eq!(
        error.to_string(),
        "validation: cannot register harness provider '': empty or built-in name"
    );
}

#[test]
fn registered_names_follow_the_builtins_in_insertion_order() {
    let first = register_fake(Arc::new(FakeSpec::new("first-harness")));
    let second = register_fake(Arc::new(FakeSpec::new("second-harness")));

    let first_position = position(&first).expect("the first name is listed");
    let second_position = position(&second).expect("the second name is listed");
    assert!(
        first_position >= PROVIDER_NAMES.len(),
        "registered providers follow the built-ins: {first_position}"
    );
    assert!(
        first_position < second_position,
        "insertion order is preserved: {first_position} then {second_position}"
    );
    for name in PROVIDER_NAMES {
        assert!(position(name).is_some(), "{name} stays listed");
    }
}

#[test]
fn re_registration_replaces_in_place() {
    let name = unique_name("replace");
    register_harness_provider(&name, fake_factory(Arc::new(FakeSpec::new("before"))))
        .expect("the first registration succeeds");
    let before = position(&name).expect("the name is listed");

    register_harness_provider(&name, fake_factory(Arc::new(FakeSpec::new("after"))))
        .expect("re-registering a non-built-in name replaces it");
    let after = position(&name).expect("the name is still listed");

    assert_eq!(before, after, "the insertion position is kept");
    assert_eq!(
        available_providers()
            .iter()
            .filter(|candidate| candidate.as_str() == name)
            .count(),
        1,
        "re-registration does not duplicate the name"
    );
    let descriptor = describe_provider(&name).expect("the replacement is resolvable");
    assert_eq!(descriptor.card_name, "after");
}

#[test]
fn unregistering_unknown_names_is_ignored() {
    unregister_harness_provider(&unique_name("missing"));
    unregister_harness_provider("");
    unregister_harness_provider(provider_names::NATIVE);
    assert!(
        position(provider_names::NATIVE).is_some(),
        "the built-in table is not registry state"
    );

    let name = register_fake(Arc::new(FakeSpec::new("gone")));
    assert!(position(&name).is_some());
    unregister_harness_provider(&name);
    assert!(position(&name).is_none());
    unregister_harness_provider(&name);
    assert!(position(&name).is_none(), "unregistering twice is harmless");
}

#[test]
fn unknown_names_report_the_available_providers() {
    let name = unique_name("unknown");
    let error = resolve_provider(&name)
        .err()
        .expect("the name is not registered");
    assert!(matches!(error, Error::NotFound(_)));
    let message = error.to_string();
    assert!(
        message.contains(&format!(
            "unknown harness provider '{name}'; expected one of "
        )),
        "unexpected message: {message}"
    );
    assert!(
        message.contains("native, native_v2, claudecode, codex, dsh"),
        "the message lists the built-ins in order: {message}"
    );
}

#[test]
fn a_failing_factory_reports_instantiation_failure() {
    let name = unique_name("failing");
    register_harness_provider(
        &name,
        Arc::new(|| -> Result<Arc<dyn ProviderSpec>> { Err(Error::unsupported("no runtime")) }),
    )
    .expect("the name is registrable");

    let error = resolve_provider(&name).err().expect("the factory fails");
    assert!(matches!(error, Error::Validation(_)));
    assert_eq!(
        error.to_string(),
        format!(
            "validation: harness provider '{name}' could not be instantiated: \
             unsupported: no runtime"
        )
    );
    assert!(
        describe_provider(&name).is_err(),
        "describing a failing provider reports the same error"
    );
}

#[test]
fn builtin_descriptors_are_always_listed() {
    let descriptors = provider_descriptors();
    let names: Vec<&str> = descriptors
        .iter()
        .take(PROVIDER_NAMES.len())
        .map(|descriptor| descriptor.name.as_str())
        .collect();
    assert_eq!(
        names, PROVIDER_NAMES,
        "declared providers come first, in order"
    );
    for descriptor in &descriptors[..PROVIDER_NAMES.len()] {
        assert!(descriptor.builtin, "{} is declared", descriptor.name);
        assert!(
            descriptor.requires.is_none() || !descriptor.implemented,
            "{}: only a deferred provider names a requirement",
            descriptor.name
        );
    }
}

#[test]
fn describe_provider_answers_declared_names_statically() {
    let native = describe_provider(provider_names::NATIVE).expect("native is declared");
    assert_eq!(native.name, "native");
    assert_eq!(native.card_name, openjiuwen_harness::protocol::HARNESS_NAME);
    assert_eq!(
        native.implementation_version,
        openjiuwen_harness::protocol::PROVIDER_VERSION
    );
    assert!(native.builtin && native.implemented && native.configures_models);
    assert!(native.requires.is_none());

    let native_v2 = describe_provider(provider_names::NATIVE_V2).expect("native_v2 is declared");
    assert_eq!(native_v2.card_name, "native_v2");
    assert_eq!(native_v2.implementation_version, "0.2.0");
    assert!(native_v2.builtin);
    assert!(!native_v2.implemented);
    assert!(!native_v2.configures_models);
    assert_eq!(
        native_v2.requires.as_deref(),
        Some("openjiuwen-agent-teams")
    );
}

#[test]
fn describe_provider_reports_unknown_names_as_not_found() {
    let name = unique_name("unknown-descriptor");
    let error = describe_provider(&name).expect_err("the name is unknown");
    assert!(matches!(error, Error::NotFound(_)));
}

#[test]
fn describe_provider_derives_registered_descriptors() {
    let spec = Arc::new(FakeSpec::new("fake-harness").with_capabilities(&[]));
    let name = register_fake(spec);
    let descriptor = describe_provider(&name).expect("the registered provider is described");
    assert_eq!(descriptor.name, name);
    assert_eq!(descriptor.card_name, "fake-harness");
    assert_eq!(descriptor.implementation_version, "0.0.1");
    assert!(!descriptor.builtin);
    assert!(descriptor.implemented);
    assert!(descriptor.configures_models);
    assert!(descriptor.requires.is_none());

    let listed = provider_descriptors();
    assert!(
        listed.iter().any(|candidate| candidate.name == name),
        "the registered provider is listed"
    );
}

#[cfg(feature = "native")]
#[test]
fn create_harness_returns_an_unstarted_native_harness() {
    let mut config = JsonObject::new();
    config.insert("model".to_owned(), JsonValue::String("scripted".to_owned()));
    let harness =
        create_harness(provider_names::NATIVE, config).expect("the native provider creates");
    assert_eq!(harness.state(), HarnessState::Terminated);
    assert_eq!(
        harness.card().name,
        openjiuwen_harness::protocol::HARNESS_NAME
    );
}

#[cfg(feature = "native")]
#[test]
fn create_harness_rejects_an_invalid_native_config() {
    let mut config = JsonObject::new();
    config.insert("zeta".to_owned(), JsonValue::from(1));
    let error = create_harness(provider_names::NATIVE, config)
        .err()
        .expect("unknown native keys are rejected");
    assert!(matches!(error, Error::Validation(_)));
    assert_eq!(
        error.to_string(),
        "validation: unknown native harness configuration fields: zeta"
    );
}

#[test]
fn context_defaults_follow_the_provider_card() {
    let name = register_fake(Arc::new(FakeSpec::new("fake-harness")));
    let context =
        build_harness_context(&name, HarnessContextOptions::new("session-1", "be helpful"))
            .expect("the registered provider resolves");

    assert_eq!(context.agent_name, "fake-harness");
    assert_eq!(context.agent_id, "fake-harness");
    assert_eq!(context.host_session_id, "session-1");
    assert_eq!(context.system_prompt, "be helpful");
    assert_eq!(context.resume_policy, ResumePolicy::New);
    assert!(context.host_capabilities.is_empty());
    assert_eq!(context.mcp_servers.len(), 0);
    assert!(context.checkpoint.is_none());
    assert!(context.checkpoint_sink.is_none());
    assert!(context.interactions.is_none());
    assert!(context.tools.is_none());
    assert!(context.hooks.is_none());
    assert!(context.telemetry.is_none());
    assert!(context.metadata.is_empty());
}

#[test]
fn context_keeps_explicit_identity_and_an_empty_prompt() {
    let name = register_fake(Arc::new(FakeSpec::new("fake-harness")));
    let mut options = HarnessContextOptions::new("session-2", "");
    options.agent_name = Some("planner".to_owned());
    options.agent_id = Some("agent-7".to_owned());
    let context = build_harness_context(&name, options).expect("the provider resolves");
    assert_eq!(context.agent_name, "planner");
    assert_eq!(context.agent_id, "agent-7");
    assert_eq!(context.system_prompt, "");
}

#[test]
fn a_checkpoint_sink_injects_the_host_capability() {
    let name = register_fake(Arc::new(FakeSpec::new("fake-harness")));
    let mut options = HarnessContextOptions::new("session-3", "prompt");
    options.checkpoint_sink = Some(Arc::new(FakeSink));
    let context = build_harness_context(&name, options).expect("the provider resolves");
    assert!(context.checkpoint_sink.is_some());
    assert!(
        context
            .host_capabilities
            .contains(&HostCapability::CheckpointSink),
        "the sink is advertised to the harness"
    );
}

#[test]
fn non_native_providers_receive_the_mcp_servers() {
    let name = register_fake(Arc::new(FakeSpec::new("fake-harness")));
    let mut options = HarnessContextOptions::new("session-4", "prompt");
    options.mcp_servers = vec![mcp_server("filesystem")];
    let context = build_harness_context(&name, options).expect("the provider resolves");
    assert_eq!(context.mcp_servers.len(), 1);
    assert_eq!(context.mcp_servers[0].name, "filesystem");
    assert!(
        context
            .host_capabilities
            .contains(&HostCapability::McpServers),
        "the servers are advertised to the harness"
    );
}

#[cfg(feature = "native")]
#[test]
fn native_drops_the_mcp_servers() {
    let mut options = HarnessContextOptions::new("session-5", "prompt");
    options.mcp_servers = vec![mcp_server("filesystem")];
    let context = build_harness_context(provider_names::NATIVE, options).expect("native resolves");
    assert_eq!(
        context.mcp_servers.len(),
        0,
        "the native harness owns its tools"
    );
    assert!(!context
        .host_capabilities
        .contains(&HostCapability::McpServers));
    assert_eq!(
        context.agent_name,
        openjiuwen_harness::protocol::HARNESS_NAME
    );
}

#[test]
fn a_provider_without_the_model_hook_is_rejected() {
    let name = register_fake(Arc::new(FakeSpec::new("fake-harness").without_models()));
    let choice = ModelChoice {
        model: Some("gpt-5".to_owned()),
        effort: None,
        endpoint: None,
    };
    let error = provider_model_config(&name, &choice, None)
        .expect_err("a provider without the hook cannot configure models");
    assert!(matches!(error, Error::Validation(_)));
    assert_eq!(
        error.to_string(),
        format!("validation: harness provider '{name}' does not configure models")
    );
}

#[test]
fn model_configuration_gates_on_the_card_capabilities() {
    let plain = register_fake(Arc::new(FakeSpec::new("fake-harness")));
    let choice = ModelChoice {
        model: Some("gpt-5".to_owned()),
        effort: None,
        endpoint: Some(ModelEndpoint {
            api_base: Some("https://example.invalid/v1".to_owned()),
            api_key: None,
            provider: None,
        }),
    };
    let error = provider_model_config(&plain, &choice, None)
        .expect_err("endpoint models need the capability");
    assert!(matches!(error, Error::Validation(_)));
    assert_eq!(
        error.to_string(),
        format!("validation: harness provider '{plain}' does not take endpoint models")
    );

    let builtin_choice = ModelChoice {
        model: Some("gpt-5".to_owned()),
        effort: None,
        endpoint: None,
    };
    let fallback = ModelChoice {
        model: Some("gpt-5-mini".to_owned()),
        effort: None,
        endpoint: None,
    };
    let error = provider_model_config(&plain, &builtin_choice, Some(&fallback))
        .expect_err("a fallback model needs the capability");
    assert_eq!(
        error.to_string(),
        format!("validation: harness provider '{plain}' has no authentication fallback")
    );
}

#[test]
fn model_configuration_forwards_to_a_capable_provider() {
    let capable = register_fake(Arc::new(FakeSpec::new("fake-harness").with_capabilities(
        &[
            HarnessCapability::AuthFallback,
            HarnessCapability::EndpointModels,
        ],
    )));
    let choice = ModelChoice {
        model: Some("gpt-5".to_owned()),
        effort: None,
        endpoint: Some(ModelEndpoint {
            api_base: Some("https://example.invalid/v1".to_owned()),
            api_key: None,
            provider: None,
        }),
    };
    let fallback = ModelChoice {
        model: Some("gpt-5-mini".to_owned()),
        effort: None,
        endpoint: None,
    };
    let fragment = provider_model_config(&capable, &choice, Some(&fallback))
        .expect("the capabilities are declared");
    assert_eq!(
        fragment.get("model"),
        Some(&JsonValue::String("gpt-5".to_owned()))
    );
    assert_eq!(
        fragment.get("fallback"),
        Some(&JsonValue::String("gpt-5-mini".to_owned()))
    );
}

#[cfg(feature = "native")]
#[test]
fn native_model_configuration_rejects_endpoints_and_fallbacks() {
    let choice = ModelChoice {
        model: Some("gpt-5".to_owned()),
        effort: None,
        endpoint: Some(ModelEndpoint {
            api_base: Some("https://example.invalid/v1".to_owned()),
            api_key: None,
            provider: None,
        }),
    };
    let error = provider_model_config(provider_names::NATIVE, &choice, None)
        .expect_err("native has no endpoint models");
    assert_eq!(
        error.to_string(),
        "validation: harness provider 'native' does not take endpoint models"
    );

    let builtin_choice = ModelChoice {
        model: Some("gpt-5".to_owned()),
        effort: None,
        endpoint: None,
    };
    let fallback = ModelChoice {
        model: Some("gpt-5-mini".to_owned()),
        effort: None,
        endpoint: None,
    };
    let error = provider_model_config(provider_names::NATIVE, &builtin_choice, Some(&fallback))
        .expect_err("native has no auth fallback");
    assert_eq!(
        error.to_string(),
        "validation: harness provider 'native' has no authentication fallback"
    );

    let fragment = provider_model_config(provider_names::NATIVE, &builtin_choice, None)
        .expect("native maps a built-in model");
    assert_eq!(
        fragment.get("model"),
        Some(&JsonValue::String("gpt-5".to_owned()))
    );
}

#[test]
fn context_options_debug_hides_secrets() {
    let mut options = HarnessContextOptions::new("session-6", "prompt");
    options
        .env
        .insert("API_KEY".to_owned(), "sk-secret".to_owned());
    options
        .metadata
        .insert("token".to_owned(), JsonValue::String("t-secret".to_owned()));
    let rendered = format!("{options:?}");
    assert!(rendered.contains("session-6"), "{rendered}");
    assert!(rendered.contains("<1 variables>"), "{rendered}");
    assert!(rendered.contains("<1 keys>"), "{rendered}");
    assert!(!rendered.contains("sk-secret"), "{rendered}");
    assert!(!rendered.contains("t-secret"), "{rendered}");
    assert!(!rendered.contains("API_KEY"), "{rendered}");
}
