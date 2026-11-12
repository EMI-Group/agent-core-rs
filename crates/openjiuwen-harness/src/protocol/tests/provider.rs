//! Provider creation and model-choice mapping.

use openjiuwen_harness_protocol::ModelConfigProvider;
use openjiuwen_harness_protocol::{HarnessCapability, HarnessProvider, ModelChoice, ModelEndpoint};

use super::*;
use crate::permission::PermissionsSection;
use crate::protocol::{native_card, NativeHarnessProvider, PROVIDER_VERSION};

/// Builds a full create mapping, with a temporary workspace root.
fn create_config(workspace: &std::path::Path) -> JsonObject {
    serde_json::from_value(serde_json::json!({
        "model": "gpt-4o",
        "provider": "openai",
        "api_key": "sk-test",
        "api_base": "https://example.invalid/v1",
        "temperature": 0.5,
        "workspace": workspace.display().to_string(),
        "enable_task_loop": true,
        "max_iterations": 3,
        "system_prompt": "be brief",
        "permissions": PermissionsSection::default(),
        "language": "en",
    }))
    .expect("the create mapping is a JSON object")
}

/// Builds a built-in (non-endpoint) model choice.
fn choice(model: Option<&str>, effort: Option<&str>) -> ModelChoice {
    ModelChoice {
        model: model.map(str::to_owned),
        effort: effort.map(str::to_owned),
        endpoint: None,
    }
}

#[test]
fn provider_card_declares_the_documented_capabilities() {
    let provider = NativeHarnessProvider::new();
    let card = provider.card();

    assert_eq!(card.name, HARNESS_NAME);
    assert_eq!(card.implementation_version, PROVIDER_VERSION);
    let declared: Vec<&str> = card
        .capabilities
        .iter()
        .map(|capability| capability.as_str())
        .collect();
    assert_eq!(
        declared,
        vec![
            "steer",
            "graceful_abort",
            "force_abort",
            "pause_resume",
            "persistent_session",
            "checkpoint",
            "native_tools",
            "mcp_tools",
            "model_selection",
            "model_discovery",
        ]
    );
    assert_eq!(
        card.capabilities,
        native_card(HARNESS_NAME, PROVIDER_VERSION).capabilities
    );
    assert_eq!(card.required_host_capabilities.len(), 0);
    let optional: Vec<&str> = card
        .optional_host_capabilities
        .iter()
        .map(|capability| capability.as_str())
        .collect();
    assert_eq!(
        optional,
        vec![
            "checkpoint_sink",
            "tool_approval",
            "user_input",
            "telemetry"
        ]
    );
    assert_eq!(card.fork_modes.len(), 0);
    assert_eq!(
        card.protocol_version,
        openjiuwen_harness_protocol::PROTOCOL_VERSION
    );
}

#[tokio::test]
async fn provider_create_returns_an_unstarted_harness_with_the_mapped_model() {
    let workspace = tempfile::tempdir().expect("temporary workspace");
    let provider = NativeHarnessProvider::new();
    let harness = provider
        .create(create_config(workspace.path()))
        .expect("the create mapping is valid");

    assert_eq!(harness.card().name, HARNESS_NAME);
    assert_eq!(harness.state(), HarnessState::Terminated);
    assert_eq!(harness.provider_session_id(), None);
    assert_eq!(harness.event_buffer_config().capacity, 1024);
    assert_eq!(
        harness.event_buffer_config().overflow,
        openjiuwen_harness_protocol::EventOverflowPolicy::CoalesceOrBlock
    );

    // `create` hands back a boxed `HarnessProtocol`, which exposes no agent
    // handle (`agent().config()` is unreachable from here) and not the
    // `HarnessModelControl` seam. The observable proof that the mapping was
    // applied is that the lazily resolved model — named, provider, base URL and
    // credential all come from the mapping — initializes, so `start` succeeds
    // without any model handle being supplied. The per-key mapping itself is
    // asserted by `the_create_mapping_is_validated_and_folded` in `provider.rs`.
    harness
        .start(context())
        .await
        .expect("the created harness resolves its model and starts");
    assert_eq!(harness.state(), HarnessState::Idle);
    assert_eq!(harness.provider_session_id(), Some("host-1"));
    let checkpoint = harness
        .export_checkpoint()
        .await
        .unwrap()
        .expect("start publishes a checkpoint");
    assert_eq!(checkpoint.provider, HARNESS_NAME);
    assert_eq!(checkpoint.agent_id, "agent-1");
}

#[tokio::test]
async fn provider_create_rejects_invalid_configuration() {
    let provider = NativeHarnessProvider::new();
    let cases = [
        (
            serde_json::json!({"zeta": 1, "alpha": true}),
            "unknown native harness configuration fields: alpha, zeta",
        ),
        (
            serde_json::json!({"model": " "}),
            "native harness string options must be non-empty strings",
        ),
        (
            serde_json::json!({"enable_task_loop": "yes"}),
            "enable_task_loop must be a boolean",
        ),
        (
            serde_json::json!({"max_iterations": 0}),
            "max_iterations must be a positive integer",
        ),
        (
            serde_json::json!({"temperature": 3.0}),
            "temperature must be a number between 0 and 2",
        ),
    ];
    for (config, expected) in cases {
        let mapping: JsonObject =
            serde_json::from_value(config.clone()).expect("the case is a JSON object");
        let error = provider
            .create(mapping)
            .err()
            .unwrap_or_else(|| panic!("{config} must be rejected"));
        assert_eq!(error.kind(), HarnessErrorKind::Validation, "for {config}");
        assert_eq!(error.message(), expected, "for {config}");
    }
}

#[test]
fn model_config_maps_choices_and_gates_capabilities() {
    let provider = NativeHarnessProvider::new();

    let fragment = provider
        .model_config(&choice(Some("gpt-4o"), Some("high")), None)
        .expect("the choice is supported");
    assert_eq!(fragment.get("model"), Some(&JsonValue::from("gpt-4o")));
    assert_eq!(fragment.get("temperature"), Some(&JsonValue::from(1.0_f64)));
    let empty = provider
        .model_config(&choice(None, None), None)
        .expect("an empty choice is supported");
    assert_eq!(empty.len(), 0);
    for (effort, expected) in [("low", 0.2_f64), ("medium", 0.6), ("high", 1.0)] {
        let fragment = provider
            .model_config(&choice(None, Some(effort)), None)
            .expect("a recognized effort is supported");
        assert_eq!(
            fragment.get("temperature"),
            Some(&JsonValue::from(expected)),
            "for {effort}"
        );
    }

    let endpoint = ModelChoice {
        model: None,
        effort: None,
        endpoint: Some(ModelEndpoint {
            api_base: Some("https://example.invalid/v1".to_owned()),
            api_key: None,
            provider: Some("openai".to_owned()),
        }),
    };
    let error = provider
        .model_config(&endpoint, None)
        .expect_err("endpoint models are not supported");
    assert_eq!(error.kind(), HarnessErrorKind::UnsupportedCapability);
    assert_eq!(
        error.message(),
        "openjiuwen-native does not take endpoint models"
    );

    let error = provider
        .model_config(&choice(None, None), Some(&choice(None, None)))
        .expect_err("authentication fallbacks are not supported");
    assert_eq!(error.kind(), HarnessErrorKind::UnsupportedCapability);
    assert_eq!(
        error.message(),
        "openjiuwen-native has no authentication fallback"
    );

    let error = provider
        .model_config(&choice(None, Some("extreme")), None)
        .expect_err("unknown efforts are rejected");
    assert_eq!(error.kind(), HarnessErrorKind::Validation);
    assert_eq!(error.message(), "unsupported reasoning effort 'extreme'");
}

#[test]
fn native_card_matches_the_provider_card() {
    let card = native_card(HARNESS_NAME, PROVIDER_VERSION);
    assert!(card.supports(HarnessCapability::Steer));
    assert!(card.supports(HarnessCapability::ModelDiscovery));
    assert!(!card.supports(HarnessCapability::Hooks));
    assert!(!card.supports(HarnessCapability::EndpointModels));
    assert!(!card.supports(HarnessCapability::AuthFallback));
}
