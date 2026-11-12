#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Offline integration suite for the provider registry, harness creation and context building.
//!
//! Port of the Python counterpart `tests/unit_tests/harness_providers/test_factory.py` (v0.1.20).
//!
//! The registry is process-global (Python's module dict), so every test registers its providers
//! under a fresh `uuid`-derived name instead of resetting the map: the suite therefore stays
//! deterministic under the parallel test harness. Built-in names are read from static metadata, so
//! nothing here depends on a provider's live runtime beyond the native create path, which only
//! *constructs* an unstarted harness.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::Error;
use openjiuwen_harness_protocol::{
    CheckpointReason, CheckpointSaveReceipt, HarnessCard, HarnessCheckpoint, HarnessCheckpointSink,
    HarnessError, HarnessProtocol, HarnessProvider, HarnessResult, HostCapability, JsonObject,
    JsonValue, McpServerConfig, McpTransport, ModelChoice, ResumePolicy,
};
#[cfg(feature = "native")]
use openjiuwen_harness_providers::create_harness;
// Only the feature-gated suites build an endpoint model choice.
#[cfg(feature = "native")]
use openjiuwen_harness_protocol::HarnessState;
#[cfg(any(feature = "native", feature = "vendor"))]
use openjiuwen_harness_protocol::ModelEndpoint;
use openjiuwen_harness_providers::{
    available_providers, build_harness_context, describe_provider, plain_provider,
    provider_descriptors, provider_model_config, register_harness_provider, resolve_provider,
    unregister_harness_provider, HarnessContextOptions, ProviderDescriptor, ProviderFactory,
    ProviderSpec, HARNESS_PROVIDER_ENTRY_POINT_GROUP, PROVIDER_NAMES,
};
// Only the feature-gated suites build JSON literals and create mappings.
#[cfg(any(feature = "native", feature = "vendor"))]
use serde_json::json;

/// Converts a JSON literal into a provider create mapping (`JsonObject` is a `BTreeMap`, so
/// `json!({...})` does not coerce).
#[cfg(any(feature = "native", feature = "vendor"))]
fn object(value: serde_json::Value) -> JsonObject {
    openjiuwen_harness_protocol::json_object_from_value(value).unwrap()
}

/// Returns a registry name no other test (or earlier run) can collide with.
///
/// Python's `test_factory.py` monkeypatches the module dict; the Rust registry is process-global,
/// so tests use unique names instead of a reset hook.
fn unique_name(tag: &str) -> String {
    format!("registry-test-{}-{}", tag, uuid::Uuid::new_v4())
}

/// A provider spec double: an explicit card plus the model-hook switch.
struct FakeSpec {
    /// Card reported by the spec.
    card: HarnessCard,
    /// Whether [`ProviderSpec::configures_models`] reports `true`.
    models: bool,
}

impl FakeSpec {
    /// Builds a model-configuring fake reporting `card_name`/`0.0.1`.
    fn new(card_name: &str) -> Self {
        Self {
            card: HarnessCard::new(card_name, "0.0.1"),
            models: true,
        }
    }
}

impl ProviderSpec for FakeSpec {
    fn card(&self) -> &HarnessCard {
        &self.card
    }

    fn create(&self, _config: JsonObject) -> HarnessResult<Box<dyn HarnessProtocol>> {
        Err(HarnessError::protocol("not created"))
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

/// A plain [`HarnessProvider`] without a model-configuration hook (the `plain_provider` path).
#[derive(Debug)]
struct FakeProvider {
    /// Card reported by the provider.
    card: HarnessCard,
}

impl FakeProvider {
    /// Builds a provider reporting `card_name`.
    fn new(card_name: &str) -> Self {
        Self {
            card: HarnessCard::new(card_name, "0.0.1"),
        }
    }
}

impl HarnessProvider for FakeProvider {
    fn card(&self) -> &HarnessCard {
        &self.card
    }

    fn create(&self, _config: JsonObject) -> HarnessResult<Box<dyn HarnessProtocol>> {
        Err(HarnessError::protocol("not created"))
    }
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

/// Wraps `spec` into a factory that always succeeds.
fn fake_factory(spec: Arc<dyn ProviderSpec>) -> ProviderFactory {
    Arc::new(move || Ok(Arc::clone(&spec)))
}

/// Registers `spec` under a fresh unique name and returns that name.
fn register_fake(spec: Arc<dyn ProviderSpec>, tag: &str) -> String {
    let name = unique_name(tag);
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

/// One stdio MCP server configuration.
fn mcp_server(name: &str) -> McpServerConfig {
    McpServerConfig {
        name: name.to_owned(),
        transport: McpTransport::Stdio,
        command: vec!["server".to_owned()],
        url: None,
        instance: None,
        env: BTreeMap::new(),
        headers: BTreeMap::new(),
    }
}

/// Builds a built-in (non-endpoint) model choice.
fn choice(model: Option<&str>, effort: Option<&str>) -> ModelChoice {
    ModelChoice {
        model: model.map(str::to_owned),
        effort: effort.map(str::to_owned),
        endpoint: None,
    }
}

/// Asserts `error` is an [`Error::Validation`] whose payload is exactly `expected`.
fn assert_validation(error: &Error, expected: &str) {
    assert!(
        matches!(error, Error::Validation(_)),
        "unexpected error: {error}"
    );
    assert_eq!(error.message().as_ref(), expected);
}

// Python: `test_factory.py` module constants (`HARNESS_PROVIDER_ENTRY_POINT_GROUP`,
// `PROVIDER_NAMES`).
#[test]
fn constants_match_the_python_module() {
    assert_eq!(
        HARNESS_PROVIDER_ENTRY_POINT_GROUP,
        "openjiuwen.harness_providers"
    );
    assert_eq!(
        PROVIDER_NAMES,
        ["native", "native_v2", "claudecode", "codex", "dsh"]
    );
}

#[cfg(feature = "native")]
#[test]
fn native_resolves_to_the_harness_card() {
    let spec = resolve_provider("native").expect("native is implemented with the native feature");
    assert_eq!(spec.card().name, "openjiuwen-native");
    assert_eq!(
        spec.card().implementation_version,
        openjiuwen_harness::protocol::PROVIDER_VERSION
    );
    assert!(spec.configures_models());
}

#[cfg(not(feature = "native"))]
#[test]
fn native_reports_the_missing_feature() {
    let error = resolve_provider("native")
        .err()
        .expect("native is compiled out without the native feature");
    assert!(matches!(error, Error::Unsupported(_)), "{error}");
    assert!(error.to_string().contains("unsupported"), "{error}");
}

#[test]
fn native_v2_reports_the_sibling_crate() {
    let error = resolve_provider("native_v2")
        .err()
        .expect("native_v2 is hosted by another crate");
    assert!(matches!(error, Error::Unsupported(_)), "{error}");
    assert!(
        error.to_string().contains("openjiuwen-agent-teams"),
        "unexpected message: {error}"
    );
}

#[cfg(feature = "vendor")]
#[test]
fn vendor_names_resolve_when_enabled() {
    for (name, card_name) in [
        ("claudecode", "claude-code"),
        ("codex", "codex"),
        ("dsh", "deepseek-harness"),
    ] {
        let spec = resolve_provider(name).expect("the vendor feature registers the stub");
        assert_eq!(spec.card().name, card_name, "for {name}");
    }
}

#[cfg(not(feature = "vendor"))]
#[test]
fn vendor_names_report_the_deferral() {
    for name in ["claudecode", "codex", "dsh"] {
        let error = resolve_provider(name)
            .err()
            .expect("the vendor feature is disabled");
        assert!(matches!(error, Error::Unsupported(_)), "{error}");
        assert!(
            error.to_string().contains("deferred"),
            "unexpected message for {name}: {error}"
        );
    }
}

// Python: `test_factory.py::test_unknown_provider_lists_available_providers`.
#[test]
fn unknown_names_are_not_found_and_list_the_providers() {
    let error = resolve_provider("nope")
        .err()
        .expect("nope is not a provider name");
    assert!(matches!(error, Error::NotFound(_)), "{error}");
    let message = error.to_string();
    assert!(
        message.starts_with(
            "not_found: unknown harness provider 'nope'; expected one of native, native_v2, \
             claudecode, codex, dsh"
        ),
        "unexpected message: {message}"
    );
    for name in PROVIDER_NAMES {
        assert!(message.contains(name), "{name} is missing from: {message}");
    }
}

// Python: `test_factory.py` registration/resolution round trip.
#[test]
fn registration_appends_after_the_builtins_and_resolves() {
    let name = register_fake(Arc::new(FakeSpec::new("registered-harness")), "registered");
    let names = available_providers();
    let builtins: Vec<String> = names.iter().take(PROVIDER_NAMES.len()).cloned().collect();
    assert_eq!(builtins, PROVIDER_NAMES.to_vec());
    assert!(
        position(&name).is_some_and(|index| index >= PROVIDER_NAMES.len()),
        "{name} must follow the built-ins in {names:?}"
    );
    let resolved = resolve_provider(&name).expect("a registered provider resolves");
    assert_eq!(resolved.card().name, "registered-harness");

    unregister_harness_provider(&name);
    assert!(
        position(&name).is_none(),
        "{name} must be gone after unregistering"
    );
    // Unregistering twice, or an unknown name, is a no-op.
    unregister_harness_provider(&name);
    unregister_harness_provider(&unique_name("never-registered"));
}

#[test]
fn re_registration_replaces_in_place_without_duplicating() {
    let first = register_fake(Arc::new(FakeSpec::new("first-harness")), "replace-a");
    let second = register_fake(Arc::new(FakeSpec::new("second-harness")), "replace-b");
    let first_before = position(&first).expect("the first name is listed");
    assert!(
        first_before < position(&second).expect("the second name is listed"),
        "insertion order is preserved"
    );

    register_harness_provider(
        &first,
        fake_factory(Arc::new(FakeSpec::new("first-replaced"))),
    )
    .expect("re-registering a non-built-in name replaces it");
    let first_after = position(&first).expect("the name is still listed");
    assert_eq!(first_before, first_after, "the insertion position is kept");
    assert!(
        first_after < position(&second).expect("the second name stays listed"),
        "the replacement does not move the name behind its successor"
    );
    assert_eq!(
        available_providers()
            .iter()
            .filter(|candidate| candidate.as_str() == first)
            .count(),
        1,
        "re-registration does not duplicate the name"
    );
    assert_eq!(
        describe_provider(&first)
            .expect("the replacement is resolvable")
            .card_name,
        "first-replaced"
    );

    unregister_harness_provider(&first);
    unregister_harness_provider(&second);
}

// Python: `test_factory.py` rejects empty and built-in registration keys.
#[test]
fn empty_and_builtin_names_cannot_be_registered() {
    let error = register_harness_provider("", fake_factory(Arc::new(FakeSpec::new("x"))))
        .expect_err("the empty name is not registrable");
    assert_validation(
        &error,
        "cannot register harness provider '': empty or built-in name",
    );

    let error = register_harness_provider("native", fake_factory(Arc::new(FakeSpec::new("x"))))
        .expect_err("a built-in name is not registrable");
    assert_validation(
        &error,
        "cannot register harness provider 'native': empty or built-in name",
    );

    for name in PROVIDER_NAMES {
        let error = register_harness_provider(name, fake_factory(Arc::new(FakeSpec::new("x"))))
            .expect_err("every declared name is reserved");
        assert!(matches!(error, Error::Validation(_)), "{error}");
    }
}

// Python: a failing provider factory surfaces as a `ValueError` naming the provider.
#[test]
fn a_failing_factory_reports_a_validation_error() {
    let name = unique_name("failing");
    let failing: ProviderFactory = Arc::new(|| Err(Error::validation("boom")));
    register_harness_provider(&name, failing).expect("a unique non-built-in name is registrable");

    let error = resolve_provider(&name)
        .err()
        .expect("the factory always fails");
    assert!(matches!(error, Error::Validation(_)), "{error}");
    assert_eq!(
        error.message().as_ref(),
        format!("harness provider '{name}' could not be instantiated: validation: boom")
    );
    assert_eq!(
        error.to_string(),
        format!(
            "validation: harness provider '{name}' could not be instantiated: validation: boom"
        )
    );

    unregister_harness_provider(&name);
}

// Python: `test_factory.py` provider listing (`list_providers`/descriptor metadata).
#[test]
fn builtin_descriptors_are_declared_in_python_order() {
    let descriptors = provider_descriptors();
    let builtins: Vec<&ProviderDescriptor> =
        descriptors.iter().filter(|entry| entry.builtin).collect();
    assert_eq!(builtins.len(), 5, "exactly five built-ins are declared");
    for (index, descriptor) in builtins.iter().enumerate() {
        assert_eq!(descriptor.name, PROVIDER_NAMES[index]);
        assert_eq!(
            *descriptor, &descriptors[index],
            "the built-ins are listed first"
        );
    }

    let native = builtins[0];
    assert_eq!(native.card_name, "openjiuwen-native");
    assert_eq!(
        native.implementation_version,
        openjiuwen_harness::protocol::PROVIDER_VERSION
    );
    assert!(native.builtin && native.implemented && native.configures_models);
    assert_eq!(native.requires, None);
}

#[test]
fn deferred_descriptors_name_their_prerequisite() {
    let descriptors = provider_descriptors();

    let native_v2 = describe_provider("native_v2").expect("native_v2 is declared");
    assert_eq!(native_v2.card_name, "native_v2");
    assert_eq!(native_v2.implementation_version, "0.2.0");
    assert!(native_v2.builtin && !native_v2.implemented && !native_v2.configures_models);
    assert_eq!(
        native_v2.requires.as_deref(),
        Some("openjiuwen-agent-teams")
    );

    let claudecode = describe_provider("claudecode").expect("claudecode is declared");
    assert_eq!(claudecode.card_name, "claude-code");
    assert_eq!(claudecode.implementation_version, "0.1.0");
    assert!(claudecode.builtin && !claudecode.implemented && claudecode.configures_models);
    assert!(
        claudecode
            .requires
            .as_deref()
            .is_some_and(|text| text.contains("Claude Agent SDK")),
        "unexpected requirement: {:?}",
        claudecode.requires
    );

    let codex = describe_provider("codex").expect("codex is declared");
    assert_eq!(codex.card_name, "codex");
    assert_eq!(codex.implementation_version, "0.1.0");
    assert!(codex.builtin && !codex.implemented && codex.configures_models);
    assert!(
        codex
            .requires
            .as_deref()
            .is_some_and(|text| text.contains("openai-codex")),
        "unexpected requirement: {:?}",
        codex.requires
    );

    let dsh = describe_provider("dsh").expect("dsh is declared");
    assert_eq!(dsh.card_name, "deepseek-harness");
    assert_eq!(dsh.implementation_version, "0.3.0");
    assert!(dsh.builtin && !dsh.implemented && dsh.configures_models);
    assert!(
        dsh.requires
            .as_deref()
            .is_some_and(|text| text.contains("deepseek-harness-sdk")),
        "unexpected requirement: {:?}",
        dsh.requires
    );

    // `describe_provider` reports the same static entry the listing does.
    assert_eq!(descriptor_named(&descriptors, "claudecode"), claudecode);
    assert_eq!(descriptor_named(&descriptors, "native_v2"), native_v2);
}

/// Returns the descriptor `name` from an existing listing.
fn descriptor_named(descriptors: &[ProviderDescriptor], name: &str) -> ProviderDescriptor {
    descriptors
        .iter()
        .find(|entry| entry.name == name)
        .cloned()
        .unwrap_or_else(|| panic!("{name} is listed"))
}

#[test]
fn describe_provider_rejects_unknown_names() {
    let error = describe_provider(&unique_name("unknown-descriptor"))
        .expect_err("an unregistered name is not described");
    assert!(matches!(error, Error::NotFound(_)), "{error}");
}

#[test]
fn registered_descriptors_report_the_card_and_no_requirement() {
    let name = register_fake(
        Arc::new(FakeSpec::new("fake-descriptor-harness")),
        "descriptor",
    );
    let expected = ProviderDescriptor {
        name: name.clone(),
        card_name: "fake-descriptor-harness".to_owned(),
        implementation_version: "0.0.1".to_owned(),
        builtin: false,
        implemented: true,
        configures_models: true,
        requires: None,
    };
    assert_eq!(
        describe_provider(&name).expect("the registered provider is described"),
        expected
    );
    assert_eq!(
        descriptor_named(&provider_descriptors(), &name),
        expected,
        "the registered provider appears in the listing"
    );

    unregister_harness_provider(&name);
}

#[cfg(feature = "native")]
#[test]
fn native_model_configuration_maps_models_and_efforts() {
    let fragment = provider_model_config("native", &choice(Some("gpt-4o"), Some("medium")), None)
        .expect("native maps a built-in model");
    assert_eq!(
        fragment.get("model"),
        Some(&JsonValue::String("gpt-4o".to_owned()))
    );
    assert_eq!(fragment.get("temperature"), Some(&json!(0.6)));

    for (effort, temperature) in [("low", 0.2), ("medium", 0.6), ("high", 1.0)] {
        let fragment = provider_model_config("native", &choice(None, Some(effort)), None)
            .expect("native maps a recognized effort");
        assert_eq!(
            fragment.get("temperature"),
            Some(&json!(temperature)),
            "for {effort}"
        );
    }
}

#[cfg(feature = "native")]
#[test]
fn native_model_configuration_rejects_invalid_choices() {
    let error = provider_model_config("native", &choice(Some(" "), None), None)
        .expect_err("a blank model is invalid");
    assert_validation(
        &error,
        "native harness string options must be non-empty strings",
    );

    let endpoint = ModelChoice {
        model: Some("gpt-4o".to_owned()),
        effort: None,
        endpoint: Some(ModelEndpoint {
            api_base: Some("https://example.invalid/v1".to_owned()),
            api_key: None,
            provider: None,
        }),
    };
    let error = provider_model_config("native", &endpoint, None)
        .expect_err("native has no endpoint models");
    assert!(matches!(error, Error::Validation(_)), "{error}");
    assert_eq!(
        error.to_string(),
        "validation: harness provider 'native' does not take endpoint models"
    );

    let error = provider_model_config(
        "native",
        &choice(Some("gpt-4o"), None),
        Some(&choice(Some("gpt-4o-mini"), None)),
    )
    .expect_err("native has no authentication fallback");
    assert!(matches!(error, Error::Validation(_)), "{error}");
    assert_eq!(
        error.to_string(),
        "validation: harness provider 'native' has no authentication fallback"
    );

    let error = provider_model_config("native", &choice(None, Some("extreme")), None)
        .expect_err("the native runtime has no such effort");
    assert_validation(&error, "unsupported reasoning effort 'extreme'");
}

#[test]
fn a_plain_provider_does_not_configure_models() {
    let name = unique_name("plain");
    register_harness_provider(
        &name,
        fake_factory(plain_provider(FakeProvider::new("plain-harness"))),
    )
    .expect("a unique non-built-in name is registrable");
    let error = provider_model_config(&name, &choice(Some("gpt-4o"), None), None)
        .expect_err("a plain provider has no model hook");
    assert_validation(
        &error,
        &format!("harness provider '{name}' does not configure models"),
    );
    unregister_harness_provider(&name);
}

#[cfg(feature = "vendor")]
#[test]
fn claude_code_fragments_are_nested_and_drop_the_provider_label() {
    let endpoint_choice = ModelChoice {
        model: Some("claude-sonnet-4".to_owned()),
        effort: Some("high".to_owned()),
        endpoint: Some(ModelEndpoint {
            api_base: Some("https://example.invalid/v1".to_owned()),
            api_key: Some("sk-test".to_owned()),
            provider: Some("openai".to_owned()),
        }),
    };
    let fragment = provider_model_config("claudecode", &endpoint_choice, None)
        .expect("claudecode maps models, efforts and endpoints");
    let nested = fragment
        .get("model")
        .and_then(JsonValue::as_object)
        .expect("the fragment nests the model configuration");
    assert_eq!(nested.get("model"), Some(&json!("claude-sonnet-4")));
    assert_eq!(nested.get("effort"), Some(&json!("high")));
    assert_eq!(
        nested.get("api_base"),
        Some(&json!("https://example.invalid/v1"))
    );
    assert_eq!(nested.get("api_key"), Some(&json!("sk-test")));
    assert!(
        !nested.contains_key("provider"),
        "claudecode has no provider label: {nested:?}"
    );
    assert!(!fragment.contains_key("fallback_model"));

    let fragment = provider_model_config(
        "claudecode",
        &choice(Some("claude-sonnet-4"), None),
        Some(&choice(Some("claude-haiku-4"), None)),
    )
    .expect("claudecode supports an authentication fallback");
    let fallback = fragment
        .get("fallback_model")
        .and_then(JsonValue::as_object)
        .expect("the fallback is nested the same way");
    assert_eq!(fallback.get("model"), Some(&json!("claude-haiku-4")));
}

#[cfg(feature = "vendor")]
#[test]
fn codex_fragments_keep_the_endpoint_provider_label() {
    let endpoint_choice = ModelChoice {
        model: Some("gpt-5".to_owned()),
        effort: None,
        endpoint: Some(ModelEndpoint {
            api_base: Some("https://example.invalid/v1".to_owned()),
            api_key: Some("sk-test".to_owned()),
            provider: Some("openai".to_owned()),
        }),
    };
    let fragment = provider_model_config("codex", &endpoint_choice, None)
        .expect("codex maps an endpoint model");
    let nested = fragment
        .get("model")
        .and_then(JsonValue::as_object)
        .expect("the fragment nests the model configuration");
    assert_eq!(nested.get("model"), Some(&json!("gpt-5")));
    assert_eq!(nested.get("provider"), Some(&json!("openai")));
    assert_eq!(
        nested.get("api_base"),
        Some(&json!("https://example.invalid/v1"))
    );
    assert_eq!(nested.get("api_key"), Some(&json!("sk-test")));
}

#[cfg(feature = "vendor")]
#[test]
fn dsh_fragments_are_flat_and_reject_effort_and_fallback() {
    let endpoint_choice = ModelChoice {
        model: Some("deepseek-chat".to_owned()),
        effort: None,
        endpoint: Some(ModelEndpoint {
            api_base: Some("https://example.invalid/v1".to_owned()),
            api_key: Some("sk-test".to_owned()),
            provider: None,
        }),
    };
    let fragment = provider_model_config("dsh", &endpoint_choice, None)
        .expect("dsh maps a flat model configuration");
    assert_eq!(fragment.len(), 3, "unexpected keys: {fragment:?}");
    assert_eq!(fragment.get("model"), Some(&json!("deepseek-chat")));
    assert_eq!(
        fragment.get("base_url"),
        Some(&json!("https://example.invalid/v1"))
    );
    assert_eq!(fragment.get("api_key"), Some(&json!("sk-test")));

    let error = provider_model_config("dsh", &choice(Some("deepseek-chat"), Some("high")), None)
        .expect_err("dsh has no reasoning effort setting");
    assert_validation(&error, "DSH has no reasoning effort setting");

    let error = provider_model_config(
        "dsh",
        &choice(Some("deepseek-chat"), None),
        Some(&choice(Some("deepseek-reasoner"), None)),
    )
    .expect_err("dsh has no authentication fallback");
    assert!(matches!(error, Error::Validation(_)), "{error}");
    assert_eq!(
        error.to_string(),
        "validation: harness provider 'dsh' has no authentication fallback"
    );
}

#[cfg(feature = "native")]
#[test]
fn create_harness_returns_an_unstarted_native_harness() {
    let harness = create_harness("native", object(json!({"model": "scripted"})))
        .expect("a minimal native configuration creates a harness");
    assert_eq!(harness.state(), HarnessState::Terminated);
    assert_eq!(harness.card().name, "openjiuwen-native");
}

#[cfg(feature = "native")]
#[test]
fn create_harness_rejects_invalid_configuration_and_unknown_providers() {
    let error = create_harness("native", object(json!({"zeta": 1})))
        .err()
        .expect("an unknown native configuration key is invalid");
    assert!(matches!(error, Error::Validation(_)), "{error}");

    let error = create_harness("nope", JsonObject::new())
        .err()
        .expect("an unknown provider cannot create a harness");
    assert!(matches!(error, Error::NotFound(_)), "{error}");
}

// Python: `test_factory.py::test_build_harness_context` defaults.
#[test]
fn context_defaults_come_from_the_provider_card() {
    let name = register_fake(
        Arc::new(FakeSpec::new("context-harness")),
        "context-defaults",
    );
    let context =
        build_harness_context(&name, HarnessContextOptions::new("session-1", "sys prompt"))
            .expect("the provider resolves");
    assert_eq!(context.agent_name, "context-harness");
    assert_eq!(context.agent_id, "context-harness");
    assert_eq!(context.host_session_id, "session-1");
    assert_eq!(context.system_prompt, "sys prompt");
    assert!(context.env.is_empty());
    assert!(context.host_capabilities.is_empty());
    assert!(context.metadata.is_empty());

    unregister_harness_provider(&name);
}

#[test]
fn explicit_identity_cwd_env_and_resume_policy_are_copied() {
    let name = register_fake(
        Arc::new(FakeSpec::new("context-explicit-harness")),
        "context-explicit",
    );
    let mut options = HarnessContextOptions::new("session-2", "prompt");
    options.agent_name = Some("my-agent".to_owned());
    options.agent_id = Some("my-agent-id".to_owned());
    options.cwd = Some("/work".to_owned());
    options.env.insert("MODE".to_owned(), "test".to_owned());
    options.resume_policy = ResumePolicy::RequireResume;

    let context = build_harness_context(&name, options).expect("the provider resolves");
    assert_eq!(context.agent_name, "my-agent");
    assert_eq!(context.agent_id, "my-agent-id");
    assert_eq!(context.cwd.as_deref(), Some("/work"));
    assert_eq!(context.env.get("MODE").map(String::as_str), Some("test"));
    assert_eq!(context.resume_policy, ResumePolicy::RequireResume);

    unregister_harness_provider(&name);
}

#[test]
fn a_checkpoint_sink_adds_the_host_capability() {
    let name = register_fake(
        Arc::new(FakeSpec::new("context-sink-harness")),
        "context-sink",
    );
    let mut options = HarnessContextOptions::new("session-3", "prompt");
    options.checkpoint_sink = Some(Arc::new(FakeSink));

    let context = build_harness_context(&name, options).expect("the provider resolves");
    assert!(context.checkpoint_sink.is_some());
    assert!(
        context
            .host_capabilities
            .contains(&HostCapability::CheckpointSink),
        "unexpected capabilities: {:?}",
        context.host_capabilities
    );

    unregister_harness_provider(&name);
}

#[test]
fn non_native_providers_receive_mcp_servers() {
    let name = register_fake(
        Arc::new(FakeSpec::new("context-mcp-harness")),
        "context-mcp",
    );
    let mut options = HarnessContextOptions::new("session-4", "prompt");
    options.mcp_servers.push(mcp_server("search"));

    let context = build_harness_context(&name, options).expect("the provider resolves");
    assert_eq!(context.mcp_servers.len(), 1);
    assert_eq!(context.mcp_servers[0].name, "search");
    assert_eq!(context.mcp_servers[0].transport, McpTransport::Stdio);
    assert!(
        context
            .host_capabilities
            .contains(&HostCapability::McpServers),
        "unexpected capabilities: {:?}",
        context.host_capabilities
    );

    unregister_harness_provider(&name);
}

#[cfg(feature = "native")]
#[test]
fn native_drops_mcp_servers_and_the_capability() {
    let mut options = HarnessContextOptions::new("session-5", "prompt");
    options.mcp_servers.push(mcp_server("search"));

    let context = build_harness_context("native", options).expect("native resolves");
    assert_eq!(context.mcp_servers.len(), 0, "native drops MCP servers");
    assert!(
        !context
            .host_capabilities
            .contains(&HostCapability::McpServers),
        "unexpected capabilities: {:?}",
        context.host_capabilities
    );
    assert_eq!(context.agent_name, "openjiuwen-native");
    assert_eq!(context.agent_id, "openjiuwen-native");
}
