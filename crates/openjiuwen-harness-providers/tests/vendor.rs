//! Offline integration suite for the vendor providers (`claudecode`, `codex`, `dsh`).
//!
//! Python counterparts live under `tests/unit_tests/harness_providers/<provider>/test_config.py`
//! (configuration decoding, defaults and cross-field rules) and `test_provider.py` (identity,
//! cards, model fragments and the `create` path). The Python tree keeps the DSH data
//! cases in `dsh/test_harness.py`, but the split is the same.
//!
//! Every case is a pure data transform — no network, no vendor SDK, no external CLI. The `create`
//! cases resolve a fake executable written by [`support::FakeCli`] (or a nonexistent explicit
//! path); resolution never starts the process, which the per-provider suites cover.

#![cfg(feature = "vendor")]
#![allow(clippy::unwrap_used, clippy::expect_used, missing_docs)]

mod support;

use std::collections::BTreeSet;

use openjiuwen_common::error::Error;
use openjiuwen_common::security::SecretString;
use openjiuwen_harness_protocol::{
    HarnessCapability, HarnessErrorKind, HarnessProvider, HarnessState, HostCapability, JsonObject,
    ModelChoice, ModelEndpoint,
};
use openjiuwen_harness_providers::providers::{claudecode, codex, dsh};
use openjiuwen_harness_providers::{
    create_harness, describe_provider, provider_model_config, SkillSource,
};
use serde_json::{json, Value};

use support::FakeCli;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Converts a JSON literal into a provider mapping (`JsonObject` is a `BTreeMap`).
// The temporary from `json!` is forwarded into a decoder, so taking the value by value keeps the
// call sites readable (same helper as the unit tests beside the provider modules).
#[allow(clippy::needless_pass_by_value)]
fn object(value: Value) -> JsonObject {
    openjiuwen_harness_protocol::json_object_from_value(value).unwrap()
}

/// Model choice served by an external endpoint, with an effort level and a provider label.
fn endpoint_choice() -> ModelChoice {
    ModelChoice {
        model: Some("m".to_owned()),
        effort: Some("high".to_owned()),
        endpoint: Some(ModelEndpoint {
            api_base: Some("https://x/v1".to_owned()),
            api_key: Some("k".to_owned()),
            provider: Some("openai".to_owned()),
        }),
    }
}

/// Model choice served by an external endpoint without an effort level (DSH refuses effort).
fn flat_choice() -> ModelChoice {
    ModelChoice {
        model: Some("m".to_owned()),
        effort: None,
        endpoint: Some(ModelEndpoint {
            api_base: Some("https://x/v1".to_owned()),
            api_key: Some("k".to_owned()),
            provider: Some("openai".to_owned()),
        }),
    }
}

/// Model choice with every field unset.
fn empty_choice() -> ModelChoice {
    ModelChoice {
        model: None,
        effort: None,
        endpoint: None,
    }
}

/// Returns the `Validation` payload of an error.
fn validation_message(error: Error) -> String {
    match error {
        Error::Validation(message) => message,
        other => panic!("expected a validation error, got {other}"),
    }
}

/// Asserts that `error` is `Error::Validation` carrying exactly `expected`.
fn assert_validation(error: Error, expected: &str) {
    assert_eq!(validation_message(error), expected);
}

/// Asserts that `error` is `Error::Validation` mentioning `needle`.
fn assert_validation_mentions(error: Error, needle: &str) {
    let message = validation_message(error);
    assert!(
        message.contains(needle),
        "expected the message to mention {needle:?}, got {message:?}"
    );
}

// ---------------------------------------------------------------------------
// Cards and identity constants
// Ports of tests/unit_tests/harness_providers/{claudecode,codex,dsh}/test_provider.py
// ---------------------------------------------------------------------------

#[test]
fn claudecode_identity_matches_python() {
    assert_eq!(claudecode::PROVIDER_NAME, "claudecode");
    assert_eq!(claudecode::CARD_NAME, "claude-code");
    assert_eq!(claudecode::IMPLEMENTATION_VERSION, "0.1.0");

    let card = claudecode::card();
    assert_eq!(card.name, "claude-code");
    assert_eq!(card.implementation_version, "0.1.0");
    assert_eq!(card.protocol_version, "1.0");
    assert_eq!(
        card.compatible_protocol_versions,
        BTreeSet::from(["1.0".to_owned()])
    );
    assert!(card.required_host_capabilities.is_empty());
    assert_eq!(claudecode::provider().card(), &card);
}

#[test]
fn claudecode_card_declares_python_capabilities() {
    let card = claudecode::card();
    assert_eq!(
        card.capabilities,
        BTreeSet::from([
            HarnessCapability::Steer,
            HarnessCapability::GracefulAbort,
            HarnessCapability::PersistentSession,
            HarnessCapability::Checkpoint,
            HarnessCapability::McpTools,
            HarnessCapability::ModelSelection,
            HarnessCapability::ModelDiscovery,
            HarnessCapability::EndpointModels,
            HarnessCapability::AuthFallback,
        ])
    );
    // Steering plus a graceful abort is the whole interrupt surface: the CLI has no harder stop,
    // no pause/resume and no native-tool gateway.
    for capability in [
        HarnessCapability::ForceAbort,
        HarnessCapability::PauseResume,
        HarnessCapability::NativeTools,
    ] {
        assert!(
            !card.capabilities.contains(&capability),
            "claudecode must not declare {capability:?}"
        );
    }
    assert_eq!(
        card.optional_host_capabilities,
        BTreeSet::from([
            HostCapability::ToolApproval,
            HostCapability::UserInput,
            HostCapability::CheckpointSink,
            HostCapability::McpServers,
            HostCapability::ProviderInteraction,
            HostCapability::ModelRequestObservation,
        ])
    );
    assert_eq!(
        card.fork_modes,
        BTreeSet::from(["before".to_owned(), "full".to_owned()])
    );
}

#[test]
fn codex_identity_matches_python() {
    assert_eq!(codex::PROVIDER_NAME, "codex");
    assert_eq!(codex::CARD_NAME, "codex");
    assert_eq!(codex::IMPLEMENTATION_VERSION, "0.1.0");

    let card = codex::card();
    assert_eq!(card.name, "codex");
    assert_eq!(card.implementation_version, "0.1.0");
    assert_eq!(card.protocol_version, "1.0");
    assert_eq!(
        card.compatible_protocol_versions,
        BTreeSet::from(["1.0".to_owned()])
    );
    assert!(card.required_host_capabilities.is_empty());
    assert_eq!(codex::provider().card(), &card);
}

#[test]
fn codex_card_declares_steering_abort_checkpoints_and_mcp_tools() {
    let card = codex::card();
    assert_eq!(
        card.capabilities,
        BTreeSet::from([
            HarnessCapability::Steer,
            HarnessCapability::GracefulAbort,
            HarnessCapability::PersistentSession,
            HarnessCapability::Checkpoint,
            HarnessCapability::McpTools,
            HarnessCapability::EndpointModels,
            HarnessCapability::AuthFallback,
        ])
    );
    // The app-server transport has no harder stop than `turn/interrupt`, no pause and no
    // `HarnessModelControl` seam, so these must stay undeclared.
    for capability in [
        HarnessCapability::ForceAbort,
        HarnessCapability::PauseResume,
        HarnessCapability::ModelSelection,
        HarnessCapability::ModelDiscovery,
        HarnessCapability::NativeTools,
    ] {
        assert!(
            !card.capabilities.contains(&capability),
            "codex must not declare {capability:?}"
        );
    }
    assert_eq!(
        card.optional_host_capabilities,
        BTreeSet::from([
            HostCapability::ToolApproval,
            HostCapability::UserInput,
            HostCapability::CheckpointSink,
            HostCapability::McpServers,
            HostCapability::NativeToolGateway,
            HostCapability::DynamicToolCall,
            HostCapability::ProviderInteraction,
            HostCapability::ModelRequestObservation,
        ])
    );
    assert_eq!(
        card.fork_modes,
        BTreeSet::from(["before".to_owned(), "full".to_owned()])
    );
}

#[test]
fn dsh_identity_matches_python() {
    assert_eq!(dsh::PROVIDER_NAME, "dsh");
    assert_eq!(dsh::CARD_NAME, "deepseek-harness");
    assert_eq!(dsh::IMPLEMENTATION_VERSION, "0.3.0");

    let card = dsh::card();
    assert_eq!(card.name, "deepseek-harness");
    assert_eq!(card.implementation_version, "0.3.0");
    assert_eq!(card.protocol_version, "1.0");
    assert_eq!(
        card.compatible_protocol_versions,
        BTreeSet::from(["1.0".to_owned()])
    );
    assert!(card.required_host_capabilities.is_empty());
    assert!(card.fork_modes.is_empty());
    assert_eq!(dsh::provider().card(), &card);
}

#[test]
fn dsh_card_declares_a_persistent_session_mcp_tools_and_endpoint_models() {
    let card = dsh::card();
    assert_eq!(
        card.capabilities,
        BTreeSet::from([
            HarnessCapability::PersistentSession,
            HarnessCapability::McpTools,
            HarnessCapability::EndpointModels,
        ])
    );
    // The pinned DSH protocol offers no steering, no interrupt, no pause and no portable
    // checkpoint, so none of them may be declared.
    for capability in [
        HarnessCapability::Steer,
        HarnessCapability::GracefulAbort,
        HarnessCapability::ForceAbort,
        HarnessCapability::PauseResume,
        HarnessCapability::Checkpoint,
    ] {
        assert!(
            !card.capabilities.contains(&capability),
            "dsh must not declare {capability:?}"
        );
    }
    assert_eq!(
        card.optional_host_capabilities,
        BTreeSet::from([HostCapability::McpServers])
    );
}

#[test]
fn descriptors_report_the_deferred_vendor_providers() {
    for (name, card_name, version) in [
        (
            claudecode::PROVIDER_NAME,
            claudecode::CARD_NAME,
            claudecode::IMPLEMENTATION_VERSION,
        ),
        (
            codex::PROVIDER_NAME,
            codex::CARD_NAME,
            codex::IMPLEMENTATION_VERSION,
        ),
        (
            dsh::PROVIDER_NAME,
            dsh::CARD_NAME,
            dsh::IMPLEMENTATION_VERSION,
        ),
    ] {
        let descriptor = describe_provider(name).expect("a declared provider");
        assert_eq!(descriptor.name, name);
        assert_eq!(descriptor.card_name, card_name);
        assert_eq!(descriptor.implementation_version, version);
        assert!(descriptor.builtin, "{name} stays a built-in name");
        assert!(!descriptor.implemented, "{name} is deferred");
        assert!(descriptor.configures_models);
        assert!(descriptor.requires.is_some(), "{name} needs an SDK");
    }
}

// ---------------------------------------------------------------------------
// Configuration defaults
// Ports of tests/unit_tests/harness_providers/{claudecode,codex,dsh}/test_config.py
// ---------------------------------------------------------------------------

#[test]
fn claudecode_defaults_match_python() {
    let config =
        claudecode::ClaudeCodeHarnessConfig::from_json(&JsonObject::new()).expect("defaults");
    assert_eq!(config, claudecode::ClaudeCodeHarnessConfig::default());

    assert_eq!(
        config.permission_mode,
        claudecode::ClaudePermissionMode::BypassPermissions
    );
    assert_eq!(
        config.system_prompt_mode,
        claudecode::ClaudeSystemPromptMode::Append
    );
    assert!(config.inherit_process_env);
    assert!(config.include_partial_messages);
    assert_eq!(config.event_buffer_capacity, 1024);
    assert!((config.request_observation_wait_s - 5.0).abs() < f64::EPSILON);
    assert!((config.lifecycle_ack_timeout_s - 10.0).abs() < f64::EPSILON);
    assert_eq!(config.skills, Vec::<SkillSource>::new());
    assert!(config.env.is_empty());
    assert!(config.settings_env.is_empty());
    assert_eq!(config.add_dirs, Vec::<std::path::PathBuf>::new());
    assert_eq!(config.session_id, None);
    assert_eq!(config.model, None);
    assert_eq!(config.fallback_model, None);
    assert_eq!(config.max_turns, None);
    assert_eq!(config.max_buffer_size, None);
    assert_eq!(claudecode::DEFAULT_CLAUDE_MAX_BUFFER_SIZE, 33_554_432);
}

#[test]
fn codex_defaults_match_python() {
    let config = codex::CodexHarnessConfig::from_json(&JsonObject::new()).expect("defaults");
    assert_eq!(config, codex::CodexHarnessConfig::default());

    assert_eq!(
        config.system_prompt_mode,
        codex::CodexSystemPromptMode::Append
    );
    assert_eq!(
        config.thread_config,
        object(json!({"model_reasoning_summary": "detailed"}))
    );
    assert!((config.turn_idle_timeout_s - 180.0).abs() < f64::EPSILON);
    assert_eq!(config.turn_idle_retries, 1);
    assert_eq!(config.max_will_retry_count, 5);
    assert_eq!(config.mcp_startup_timeout_s, 120);
    assert!(config.mcp_required);
    assert_eq!(config.client_name, "openjiuwen_harness");
    assert_eq!(config.client_title, "OpenJiuwen Harness");
    assert!(config.experimental_raw_events);
    assert_eq!(config.event_buffer_capacity, 1024);
    assert!((config.request_observation_wait_s - 5.0).abs() < f64::EPSILON);
    assert!(!config.bypass_approvals_and_sandbox);
    assert_eq!(config.skills, Vec::<SkillSource>::new());
    assert!(config.env.is_empty());
    assert_eq!(config.model, None);
    assert_eq!(config.mcp_default_tools_approval_mode, None);
}

#[test]
fn dsh_defaults_match_python() {
    let config = dsh::DshHarnessConfig::from_json(&JsonObject::new()).expect("defaults");
    assert_eq!(config, dsh::DshHarnessConfig::default());

    assert_eq!(config.provider, "deepseek-official");
    assert_eq!(config.model, "deepseek-v4-flash");
    assert_eq!(config.system_prompt_mode, dsh::DshSystemPromptMode::Replace);
    assert!((config.shutdown_timeout_seconds - 1.0).abs() < f64::EPSILON);
    assert_eq!(config.event_buffer_capacity, 1024);
    assert_eq!(config.patches, Vec::<String>::new());
    assert!(config.env.is_empty());
    assert_eq!(config.launch_args_override, None);
    assert_eq!(config.max_tokens, None);
    assert_eq!(config.api_key, None);
}

// ---------------------------------------------------------------------------
// Unknown configuration keys
// Ports of tests/unit_tests/harness_providers/{claudecode,codex,dsh}/test_config.py
// ---------------------------------------------------------------------------

#[test]
fn claudecode_rejects_unknown_configuration_fields() {
    let error =
        claudecode::ClaudeCodeHarnessConfig::from_json(&object(json!({"zeta": 1, "alpha": true})))
            .unwrap_err();
    assert_validation(
        error,
        "unknown Claude Code configuration fields: alpha, zeta",
    );
}

#[test]
fn claudecode_rejects_unknown_model_config_fields() {
    let error = claudecode::ClaudeCodeHarnessConfig::from_json(&object(
        json!({"model": {"zeta": 1, "alpha": true}}),
    ))
    .unwrap_err();
    assert_validation(error, "unknown Claude model config fields: alpha, zeta");
}

#[test]
fn codex_rejects_unknown_configuration_fields() {
    let error = codex::CodexHarnessConfig::from_json(&object(json!({"zeta": 1, "alpha": true})))
        .unwrap_err();
    assert_validation(error, "unknown Codex configuration fields: alpha, zeta");
}

#[test]
fn codex_rejects_unknown_model_config_fields() {
    let error =
        codex::CodexHarnessConfig::from_json(&object(json!({"model": {"zeta": 1, "alpha": true}})))
            .unwrap_err();
    assert_validation(error, "unknown Codex model config fields: alpha, zeta");
}

#[test]
fn dsh_rejects_unknown_configuration_fields() {
    let error =
        dsh::DshHarnessConfig::from_json(&object(json!({"zeta": 1, "alpha": true}))).unwrap_err();
    assert_validation(error, "unknown DSH configuration fields: alpha, zeta");
}

// ---------------------------------------------------------------------------
// Cross-field and range rules
// ---------------------------------------------------------------------------

/// Port of `test_config.py::test_rejects_cross_field_violations` (DSH).
#[test]
fn dsh_rejects_invalid_cross_field_values() {
    for (value, expected) in [
        (
            json!({"system_prompt_mode": "append", "system_prompt_env_var": "X"}),
            "append mode cannot use system_prompt_env_var",
        ),
        (
            json!({"system_prompt_mode": "bogus"}),
            "system_prompt_mode must be 'append' or 'replace'",
        ),
        (
            json!({"provider": ""}),
            "DSH provider and model must not be empty",
        ),
        (json!({"max_tokens": 0}), "DSH max_tokens must be positive"),
        (
            json!({"event_buffer_capacity": 0}),
            "DSH event_buffer_capacity must be positive",
        ),
        (
            json!({"request_timeout_seconds": 0.0}),
            "DSH request_timeout_seconds must be positive when provided",
        ),
        (json!({"dsh_home": " "}), "DSH dsh_home must not be empty"),
        (
            json!({"launch_args_override": []}),
            "DSH launch_args_override must contain non-empty strings",
        ),
        (
            json!({"launch_args_override": [""]}),
            "DSH launch_args_override must contain non-empty strings",
        ),
    ] {
        let error = dsh::DshHarnessConfig::from_json(&object(value)).unwrap_err();
        assert_validation(error, expected);
    }
}

#[test]
fn dsh_rejects_a_non_array_launch_args_override() {
    let error = dsh::DshHarnessConfig::from_json(&object(json!({"launch_args_override": "x"})))
        .unwrap_err();
    assert_validation_mentions(error, "launch_args_override");
}

/// Port of `test_config.py::test_rejects_cross_field_violations` (Claude Code).
#[test]
fn claudecode_rejects_invalid_values() {
    for (value, expected) in [
        (
            json!({"permission_mode": "bogus"}),
            "Claude permission_mode must be one of default, acceptEdits, plan, \
             bypassPermissions, dontAsk, auto",
        ),
        (
            json!({"system_prompt_mode": "bogus"}),
            "Claude system_prompt_mode must be 'append' or 'replace'",
        ),
        (
            json!({"max_turns": 0}),
            "Claude max_turns must be a positive integer when provided",
        ),
        (
            json!({"event_buffer_capacity": 0}),
            "Claude event_buffer_capacity must be positive",
        ),
        (
            json!({"session_id": ""}),
            "Claude session_id must not be empty",
        ),
    ] {
        let error = claudecode::ClaudeCodeHarnessConfig::from_json(&object(value)).unwrap_err();
        assert_validation(error, expected);
    }
}

#[test]
fn claudecode_rejects_a_boolean_max_turns() {
    let error = claudecode::ClaudeCodeHarnessConfig::from_json(&object(json!({"max_turns": true})))
        .unwrap_err();
    assert_validation_mentions(error, "max_turns");
}

#[test]
fn claudecode_accepts_a_valid_permission_mode() {
    let config = claudecode::ClaudeCodeHarnessConfig::from_json(&object(
        json!({"permission_mode": "acceptEdits"}),
    ))
    .expect("a valid permission mode");
    assert_eq!(
        config.permission_mode,
        claudecode::ClaudePermissionMode::AcceptEdits
    );
}

/// Port of `test_config.py::test_rejects_cross_field_violations` (Codex).
#[test]
fn codex_rejects_invalid_values() {
    for (value, expected) in [
        (
            json!({"system_prompt_mode": "bogus"}),
            "system_prompt_mode must be 'append' or 'replace'",
        ),
        (
            json!({"turn_idle_timeout_s": 0.0}),
            "Codex turn_idle_timeout_s must be greater than zero",
        ),
        (
            json!({"event_buffer_capacity": 0}),
            "Codex mcp_startup_timeout_s and event_buffer_capacity must be positive",
        ),
        (
            json!({"mcp_default_tools_approval_mode": "bogus"}),
            "Codex mcp_default_tools_approval_mode must be one of auto, prompt, writes, approve",
        ),
        (
            json!({"thread_config": "x"}),
            "Codex thread_config must be an object",
        ),
    ] {
        let error = codex::CodexHarnessConfig::from_json(&object(value)).unwrap_err();
        assert_validation(error, expected);
    }
}

// ---------------------------------------------------------------------------
// Model fragments
// Ports of tests/unit_tests/harness_providers/test_model_config.py
// ---------------------------------------------------------------------------

#[test]
fn claudecode_model_config_nests_the_model_without_the_endpoint_provider() {
    let choice = endpoint_choice();
    let fragment = claudecode::model_config(&choice, None).expect("a model fragment");
    assert_eq!(
        fragment,
        object(json!({
            "model": {
                "model": "m",
                "effort": "high",
                "api_base": "https://x/v1",
                "api_key": "k",
            },
        }))
    );
    // Claude Code speaks the Anthropic protocol only: the endpoint label is dropped everywhere.
    assert!(!serde_json::to_string(&fragment)
        .unwrap()
        .contains("provider"));
}

#[test]
fn claudecode_model_config_mirrors_the_fallback_shape() {
    let choice = endpoint_choice();
    let fragment = claudecode::model_config(&choice, Some(&choice)).expect("a model fragment");
    let inner = fragment.get("model").expect("the model entry");
    assert_eq!(fragment.get("fallback_model"), Some(inner));
    assert_eq!(
        inner,
        &json!({
            "model": "m",
            "effort": "high",
            "api_base": "https://x/v1",
            "api_key": "k",
        })
    );
}

#[test]
fn claudecode_model_config_of_an_empty_choice_wraps_an_empty_object() {
    let fragment = claudecode::model_config(&empty_choice(), None).expect("a model fragment");
    assert_eq!(fragment, object(json!({"model": {}})));
}

#[test]
fn codex_model_config_keeps_the_endpoint_provider() {
    let choice = endpoint_choice();
    let fragment = codex::model_config(&choice, None).expect("a model fragment");
    assert_eq!(
        fragment,
        object(json!({
            "model": {
                "model": "m",
                "effort": "high",
                "provider": "openai",
                "api_base": "https://x/v1",
                "api_key": "k",
            },
        }))
    );
}

#[test]
fn codex_model_config_mirrors_the_fallback_shape() {
    let choice = ModelChoice {
        model: Some("gpt-5-mini".to_owned()),
        effort: None,
        endpoint: None,
    };
    let fragment = codex::model_config(&endpoint_choice(), Some(&choice)).expect("a fragment");
    assert_eq!(
        fragment.get("fallback_model"),
        Some(&json!({"model": "gpt-5-mini"}))
    );
}

#[test]
fn dsh_model_config_is_flat() {
    let fragment = dsh::model_config(&flat_choice(), None).expect("a model fragment");
    assert_eq!(
        fragment,
        object(json!({
            "model": "m",
            "base_url": "https://x/v1",
            "api_key": "k",
        }))
    );
    assert!(!fragment.contains_key("model_config"));
}

#[test]
fn dsh_model_config_rejects_effort_and_fallback() {
    let effort = ModelChoice {
        effort: Some("high".to_owned()),
        ..empty_choice()
    };
    let error = dsh::model_config(&effort, None).expect_err("DSH has no effort setting");
    assert_eq!(error.kind(), HarnessErrorKind::Validation);
    assert_eq!(error.message(), "DSH has no reasoning effort setting");

    let fallback = ModelChoice {
        model: Some("deepseek-v4-lite".to_owned()),
        ..empty_choice()
    };
    let error = dsh::model_config(&empty_choice(), Some(&fallback))
        .expect_err("DSH has no authentication fallback");
    assert_eq!(error.kind(), HarnessErrorKind::UnsupportedCapability);
    assert_eq!(error.message(), "DSH has no authentication fallback");
}

#[test]
fn dsh_model_config_of_an_empty_choice_is_empty() {
    let fragment = dsh::model_config(&empty_choice(), None).expect("a model fragment");
    assert!(fragment.is_empty());
}

#[test]
fn factory_model_config_matches_the_module_fragments() {
    let choice = endpoint_choice();
    let flat = flat_choice();
    for (name, selected, module_fragment) in [
        (
            claudecode::PROVIDER_NAME,
            &choice,
            claudecode::model_config(&choice, None).unwrap(),
        ),
        (
            codex::PROVIDER_NAME,
            &choice,
            codex::model_config(&choice, None).unwrap(),
        ),
        (
            dsh::PROVIDER_NAME,
            &flat,
            dsh::model_config(&flat, None).unwrap(),
        ),
    ] {
        let fragment = provider_model_config(name, selected, None).expect("a model fragment");
        assert_eq!(fragment, module_fragment, "fragment of {name}");
    }

    let codex_fragment =
        provider_model_config(codex::PROVIDER_NAME, &choice, None).expect("a model fragment");
    assert_eq!(
        codex_fragment
            .get("model")
            .and_then(|model| model.get("provider")),
        Some(&json!("openai"))
    );
    let claude_fragment =
        provider_model_config(claudecode::PROVIDER_NAME, &choice, None).expect("a fragment");
    assert_eq!(
        claude_fragment
            .get("model")
            .and_then(|model| model.get("provider")),
        None
    );
}

#[test]
fn factory_model_config_enforces_the_dsh_fallback_gate() {
    let choice = endpoint_choice();
    // The factory rejects the fallback from the card before the module maps it, so the caller sees
    // the registry wording instead of the module's `UnsupportedCapability`.
    let error = provider_model_config(dsh::PROVIDER_NAME, &choice, Some(&choice)).unwrap_err();
    assert_validation(
        error,
        "harness provider 'dsh' has no authentication fallback",
    );
}

// ---------------------------------------------------------------------------
// The create path
// Ports of tests/unit_tests/harness_providers/{claudecode,codex,dsh}/test_provider.py::test_create
// ---------------------------------------------------------------------------

#[test]
fn create_reports_a_missing_binary_as_a_protocol_error() {
    // Each provider pins its own executable key, so a nonexistent explicit path always fails the
    // same way: a typed `Protocol` error naming the binary, never a missing-SDK claim.
    let cases = [
        (
            "claude",
            claudecode::provider().create(object(json!({"cli_path": "/nonexistent/claude"}))),
        ),
        (
            "codex",
            codex::provider().create(object(json!({"codex_bin": "/nonexistent/codex"}))),
        ),
        (
            "dsh",
            dsh::provider().create(object(json!({"dsh_bin": "/nonexistent/dsh"}))),
        ),
    ];
    for (binary, result) in cases {
        let error = result
            .err()
            .unwrap_or_else(|| panic!("a missing {binary} must not create a harness"));
        assert_ne!(
            error.kind(),
            HarnessErrorKind::UnsupportedCapability,
            "{binary}: {}",
            error.message()
        );
        assert_eq!(
            error.kind(),
            HarnessErrorKind::Protocol,
            "{binary}: {}",
            error.message()
        );
        assert!(
            error.message().contains(binary),
            "the error names {binary}: {}",
            error.message()
        );
    }
}

#[test]
fn create_returns_an_unstarted_harness_for_every_vendor_provider() {
    // The fake executable is only resolved: `create` never spawns it.
    let cli = FakeCli::script("# a fake executable that is never started\n");
    let home = tempfile::tempdir().expect("a temporary DSH home");
    let binary = cli.path().to_string_lossy().into_owned();
    let cases = [
        (claudecode::PROVIDER_NAME, json!({"cli_path": &binary})),
        (codex::PROVIDER_NAME, json!({"codex_bin": &binary})),
        (
            dsh::PROVIDER_NAME,
            json!({"dsh_bin": &binary, "dsh_home": home.path().to_string_lossy()}),
        ),
    ];
    for (name, config) in cases {
        let harness = create_harness(name, object(config))
            .unwrap_or_else(|error| panic!("{name} creates an unstarted harness: {error}"));
        assert_eq!(harness.state(), HarnessState::Terminated, "{name}");
        assert_eq!(harness.provider_session_id(), None, "{name}");
    }
    assert!(cli.lines().is_empty(), "create never starts the binary");
}

#[test]
fn create_surfaces_configuration_errors_before_the_binary_check() {
    let error = claudecode::provider()
        .create(object(json!({"permission_mode": "bogus"})))
        .err()
        .expect("an invalid configuration wins over the binary check");
    assert_eq!(error.kind(), HarnessErrorKind::Validation);
    assert_eq!(
        error.message(),
        "Claude permission_mode must be one of default, acceptEdits, plan, bypassPermissions, \
         dontAsk, auto"
    );
}

// ---------------------------------------------------------------------------
// Serde round-trip and secret redaction
// Ports of tests/unit_tests/harness_providers/{claudecode,codex,dsh}/test_config.py
// ---------------------------------------------------------------------------

#[test]
fn claudecode_config_round_trips_and_redacts_secrets() {
    let config = claudecode::ClaudeCodeHarnessConfig::from_json(&object(json!({
        "skills": [{"dir": "/tmp/skill", "mode": "auto_list", "enabled_skills": ["a"]}],
        "env": {"CLAUDE_TOKEN": "env-secret"},
        "settings_env": {"SETTINGS_TOKEN": "env-secret"},
        "cwd": "/tmp/work",
        "add_dirs": ["/tmp/extra"],
        "model": {"model": "sonnet", "api_key": "super-secret", "effort": "high"},
        "fallback_model": {"model": "haiku"},
        "session_id": "session-1",
        "permission_mode": "acceptEdits",
        "system_prompt_mode": "replace",
        "max_turns": 3,
        "settings": {"sandbox": {"enabled": true}},
    })))
    .expect("a valid configuration");
    assert_eq!(config.skills.len(), 1);
    assert_eq!(
        config.model.as_ref().map(|model| model.effort.as_deref()),
        Some(Some("high"))
    );

    let value = serde_json::to_value(&config).expect("the configuration serializes");
    let restored: claudecode::ClaudeCodeHarnessConfig =
        serde_json::from_value(value).expect("the configuration round-trips");
    assert_eq!(restored, config);

    let rendered = format!("{config:?}");
    assert!(!rendered.contains("super-secret"));
    assert!(!rendered.contains("env-secret"));
    assert!(rendered.contains("<redacted>"));
    assert_eq!(
        config.model.as_ref().map(|model| model.api_key.as_ref()),
        Some(Some(&SecretString::new("super-secret")))
    );
}

#[test]
fn codex_config_round_trips_and_redacts_secrets() {
    let config = codex::CodexHarnessConfig::from_json(&object(json!({
        "skills": [{"dir": "/tmp/skill", "mode": "auto_list", "enabled_skills": ["a"]}],
        "env": {"CODEX_TOKEN": "env-secret"},
        "model": {
            "model": "gpt-5",
            "provider": "openai",
            "api_base": "https://x/v1",
            "api_key": "super-secret",
            "effort": "high",
        },
        "fallback_model": {"model": "gpt-5-mini"},
        "thread_config": {"model_reasoning_summary": "concise", "sandbox_mode": "workspace-write"},
        "config_overrides": ["model_reasoning_effort=\"high\""],
        "mcp_env_passthrough": ["PATH"],
        "mcp_default_tools_approval_mode": "writes",
        "turn_idle_retries": 3,
        "client_name": "test-client",
    })))
    .expect("a valid configuration");
    assert_eq!(config.skills.len(), 1);
    assert_eq!(
        config.mcp_default_tools_approval_mode,
        Some(codex::CodexApprovalMode::Writes)
    );

    let value = serde_json::to_value(&config).expect("the configuration serializes");
    let restored: codex::CodexHarnessConfig =
        serde_json::from_value(value).expect("the configuration round-trips");
    assert_eq!(restored, config);

    let rendered = format!("{config:?}");
    assert!(!rendered.contains("super-secret"));
    assert!(!rendered.contains("env-secret"));
    assert!(rendered.contains("<redacted>"));
}

#[test]
fn dsh_config_round_trips_and_redacts_secrets() {
    let config = dsh::DshHarnessConfig::from_json(&object(json!({
        "provider": "custom",
        "model": "deepseek-v4-pro",
        "reasoning_effort": "high",
        "max_tokens": 512,
        "skills": [{"dir": "/tmp/skill", "mode": "auto_list", "enabled_skills": ["a"]}],
        "cwd": "/tmp/work",
        "runtime_cwd": "/tmp/runtime",
        "dsh_bin": "/usr/bin/dsh",
        "dsh_home": "/tmp/dsh-home",
        "profile": "prod",
        "patches": ["plugin-a"],
        "env": {"DSH_TOKEN": "env-secret"},
        "launch_args_override": ["--profile", "prod"],
        "initialize_timeout_seconds": 12.5,
        "request_timeout_seconds": 30.0,
        "shutdown_timeout_seconds": 2.0,
        "base_url": "https://x/v1",
        "api_key": "super-secret",
        "system_prompt_env_var": "DSH_SYSTEM_PROMPT",
        "event_buffer_capacity": 64,
    })))
    .expect("a valid configuration");
    assert_eq!(config.skills.len(), 1);
    assert_eq!(config.launch_args_override.as_ref().map(Vec::len), Some(2));

    let value = serde_json::to_value(&config).expect("the configuration serializes");
    let restored: dsh::DshHarnessConfig =
        serde_json::from_value(value).expect("the configuration round-trips");
    assert_eq!(restored, config);

    let rendered = format!("{config:?}");
    assert!(!rendered.contains("super-secret"));
    assert!(!rendered.contains("env-secret"));
    assert!(rendered.contains("<redacted>"));
    assert_eq!(
        config.api_key.as_ref(),
        Some(&SecretString::new("super-secret"))
    );
}
