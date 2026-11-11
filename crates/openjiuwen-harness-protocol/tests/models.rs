//! Round A model-parity suite: harness card and capability negotiation, model selection and
//! discovery, delivery input/receipts, and the shared JSON helpers.
//!
//! Where a Python counterpart exists it is named in a `// Python:` comment and lives in
//! `tests/unit_tests/harness_protocol/` of the Python SDK (v0.1.20).

// Tests assert by panicking, which is the intent of a test suite.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::collections::BTreeSet;

use serde_json::json;

use openjiuwen_harness_protocol::*;

/// A self-consistent card that requires the two host capabilities used by the negotiation tests.
fn base_card() -> HarnessCard {
    HarnessCard {
        capabilities: BTreeSet::from([HarnessCapability::Steer]),
        required_host_capabilities: BTreeSet::from([
            HostCapability::McpServers,
            HostCapability::ToolApproval,
        ]),
        ..HarnessCard::new("acme-harness", "1.2.3")
    }
}

/// A fully populated catalog entry.
fn model_option() -> ModelOption {
    ModelOption {
        model_id: "sonnet".to_owned(),
        display_name: "Claude Sonnet".to_owned(),
        description: "balanced frontier model".to_owned(),
        efforts: vec!["low".to_owned(), "high".to_owned()],
        default_effort: Some("high".to_owned()),
        is_default: true,
        extensions: json_object_from_value(json!({"context_window": 200_000}))
            .expect("extensions are an object"),
    }
}

#[test]
fn fork_modes_default_empty() {
    // Python: tests/unit_tests/harness_protocol/test_models_fork_modes.py::test_fork_modes_default_empty
    let card = HarnessCard::new("x", "1");
    assert!(card.fork_modes.is_empty());
}

#[test]
fn card_reports_capabilities_and_has_value_semantics() {
    // Python: tests/unit_tests/harness_protocol/test_protocol.py::test_card_is_immutable_and_reports_capabilities
    assert_eq!(PROTOCOL_VERSION, "1.0");

    let card = base_card();
    assert_eq!(card.protocol_version, "1.0");
    assert!(card.supports(HarnessCapability::Steer));
    assert!(!card.supports(HarnessCapability::PauseResume));
    card.validate().expect("self-consistent card");

    let snapshot = card.clone();
    let renamed = HarnessCard {
        name: "other-harness".to_owned(),
        ..card
    };
    assert_eq!(snapshot.name, "acme-harness");
    assert_eq!(renamed.name, "other-harness");
    assert_ne!(snapshot, renamed);
}

#[test]
fn validate_host_checks_version_then_required_capabilities() {
    let card = base_card();
    let satisfied = BTreeSet::from([HostCapability::McpServers, HostCapability::ToolApproval]);
    card.validate_host(PROTOCOL_VERSION, &satisfied)
        .expect("host satisfies the card");

    let error = card
        .validate_host("99.0", &satisfied)
        .expect_err("unsupported protocol version");
    assert_eq!(error.kind(), HarnessErrorKind::Protocol);
    assert_eq!(
        error.message(),
        "host protocol '99.0' is not supported by 'acme-harness'"
    );
    assert!(error.to_string().contains("is not supported"));

    let error = card
        .validate_host(PROTOCOL_VERSION, &BTreeSet::new())
        .expect_err("host misses the required capabilities");
    assert_eq!(error.kind(), HarnessErrorKind::Protocol);
    assert_eq!(
        error.message(),
        "host is missing required capabilities: mcp_servers, tool_approval"
    );

    // Optional capabilities are never validated: extra host capabilities are accepted.
    let extras = BTreeSet::from([
        HostCapability::McpServers,
        HostCapability::ToolApproval,
        HostCapability::Telemetry,
    ]);
    card.validate_host(PROTOCOL_VERSION, &extras)
        .expect("extra host capabilities are allowed");
}

#[test]
fn card_self_consistency_is_validated() {
    for malformed in [
        HarnessCard {
            name: String::new(),
            ..base_card()
        },
        HarnessCard {
            implementation_version: String::new(),
            ..base_card()
        },
        HarnessCard {
            protocol_version: String::new(),
            ..base_card()
        },
    ] {
        let error = malformed.validate().expect_err("empty identity or version");
        assert_eq!(error.kind(), HarnessErrorKind::Validation);
        assert_eq!(
            error.message(),
            "harness card identity and versions must not be empty"
        );
    }

    let incompatible = HarnessCard {
        protocol_version: "2.0".to_owned(),
        ..base_card()
    };
    assert_eq!(
        incompatible
            .validate()
            .expect_err("protocol_version is not compatible")
            .message(),
        "card protocol_version must be included in compatible_protocol_versions"
    );

    let overlapping = HarnessCard {
        required_host_capabilities: BTreeSet::from([HostCapability::ToolApproval]),
        optional_host_capabilities: BTreeSet::from([HostCapability::ToolApproval]),
        ..base_card()
    };
    assert_eq!(
        overlapping
            .validate()
            .expect_err("a capability is both required and optional")
            .message(),
        "host capabilities cannot be both required and optional: tool_approval"
    );
}

#[test]
fn card_decode_applies_defaults_and_tolerates_unknown_keys() {
    let card: HarnessCard =
        from_str_validated(r#"{"name": "n", "implementation_version": "1"}"#).expect("valid card");
    assert_eq!(card.protocol_version, "1.0");
    assert_eq!(
        card.compatible_protocol_versions,
        default_compatible_protocol_versions()
    );
    assert_eq!(
        card.compatible_protocol_versions,
        BTreeSet::from(["1.0".to_owned()])
    );
    assert!(card.capabilities.is_empty());
    assert!(card.required_host_capabilities.is_empty());
    assert!(card.optional_host_capabilities.is_empty());
    assert!(card.fork_modes.is_empty());

    let error = from_str_validated::<HarnessCard>(r#"{"implementation_version": "1"}"#)
        .expect_err("name is required");
    assert_eq!(error.kind(), HarnessErrorKind::Protocol);

    let tolerant: HarnessCard = from_str_validated(
        r#"{"name": "n", "implementation_version": "1", "future_field": {"x": 1}}"#,
    )
    .expect("unknown card fields are tolerated");
    assert_eq!(tolerant.name, "n");

    let option: ModelOption = from_str_validated(r#"{"model_id": "sonnet", "future_field": true}"#)
        .expect("unknown option fields are tolerated");
    assert_eq!(option.model_id, "sonnet");
    assert_eq!(option.efforts, Vec::<String>::new());
}

#[test]
fn model_selection_requires_a_non_blank_model_or_effort() {
    // Python: tests/unit_tests/harness_protocol/test_protocol.py::test_model_selection_and_option_values_are_validated_and_frozen
    let effort_only = ModelSelection {
        model: None,
        effort: Some("low".to_owned()),
    };
    effort_only
        .validate()
        .expect("an effort-only selection is valid");
    assert!(effort_only.model.is_none());

    let both = ModelSelection {
        model: Some("opus".to_owned()),
        effort: Some("high".to_owned()),
    };
    both.validate()
        .expect("a model-and-effort selection is valid");

    let neither = ModelSelection {
        model: None,
        effort: None,
    };
    assert_eq!(
        neither
            .validate()
            .expect_err("neither model nor effort")
            .message(),
        "model selection must set a model, an effort, or both"
    );

    for model in [" ".to_owned(), String::new()] {
        let selection = ModelSelection {
            model: Some(model),
            effort: None,
        };
        assert_eq!(
            selection.validate().expect_err("blank model").message(),
            "model selection model must be a non-empty string when provided"
        );
    }

    for effort in [" ".to_owned(), String::new()] {
        let selection = ModelSelection {
            model: Some("opus".to_owned()),
            effort: Some(effort),
        };
        assert_eq!(
            selection.validate().expect_err("blank effort").message(),
            "model selection effort must be a non-empty string when provided"
        );
    }
}

#[test]
fn model_option_validates_and_round_trips_catalog_data() {
    // Python: tests/unit_tests/harness_protocol/test_protocol.py::test_model_selection_and_option_values_are_validated_and_frozen
    let option = model_option();
    option.validate().expect("valid option");
    assert_eq!(option.efforts, ["low", "high"]);
    assert_eq!(option.extensions["context_window"], json!(200_000));

    let encoded = serde_json::to_value(&option).expect("serializes");
    assert_eq!(encoded["efforts"], json!(["low", "high"]));
    assert_eq!(encoded["default_effort"], json!("high"));
    assert_eq!(encoded["extensions"], json!({"context_window": 200_000}));
    let decoded: ModelOption = from_str_validated(&encoded.to_string()).expect("round trip");
    assert_eq!(decoded, option);

    let empty_id = ModelOption {
        model_id: String::new(),
        ..option.clone()
    };
    assert_eq!(
        empty_id.validate().expect_err("empty model id").message(),
        "model option model_id must not be empty"
    );

    let blank_effort = ModelOption {
        efforts: vec!["low".to_owned(), String::new()],
        default_effort: None,
        ..option.clone()
    };
    assert_eq!(
        blank_effort.validate().expect_err("blank effort").message(),
        "model option efforts must be non-empty strings"
    );

    let stray_default = ModelOption {
        default_effort: Some("max".to_owned()),
        ..option
    };
    assert_eq!(
        stray_default
            .validate()
            .expect_err("default_effort is not one of efforts")
            .message(),
        "model option default_effort must be one of its efforts"
    );
}

#[test]
fn model_option_defaults_match_python() {
    let minimal: ModelOption =
        from_str_validated(r#"{"model_id": "sonnet"}"#).expect("valid option");
    assert_eq!(minimal.model_id, "sonnet");
    assert_eq!(minimal.display_name, "");
    assert_eq!(minimal.description, "");
    assert_eq!(minimal.efforts, Vec::<String>::new());
    assert_eq!(minimal.default_effort, None);
    assert!(!minimal.is_default);
    assert!(minimal.extensions.is_empty());
    minimal.validate().expect("defaults are valid");
}

#[test]
fn model_choice_reports_builtin_and_validates_its_endpoint() {
    let builtin = ModelChoice {
        model: Some("opus".to_owned()),
        effort: None,
        endpoint: None,
    };
    assert!(builtin.is_builtin());
    builtin.validate().expect("valid built-in choice");

    let hosted = ModelChoice {
        model: Some("opus".to_owned()),
        effort: None,
        endpoint: Some(ModelEndpoint {
            api_base: Some("https://api.example.test/v1".to_owned()),
            api_key: None,
            provider: Some("openai".to_owned()),
        }),
    };
    assert!(!hosted.is_builtin());
    hosted.validate().expect("valid hosted choice");

    let neither = ModelChoice {
        model: None,
        effort: None,
        endpoint: None,
    };
    assert_eq!(
        neither
            .validate()
            .expect_err("neither model nor effort")
            .message(),
        "model choice must set a model, an effort, or both"
    );

    for model in [" ".to_owned(), String::new()] {
        let choice = ModelChoice {
            model: Some(model),
            effort: None,
            endpoint: None,
        };
        assert_eq!(
            choice.validate().expect_err("blank model").message(),
            "model choice model must be a non-empty string when provided"
        );
    }

    let empty_endpoint = ModelChoice {
        model: Some("opus".to_owned()),
        effort: None,
        endpoint: Some(ModelEndpoint {
            api_base: None,
            api_key: None,
            provider: None,
        }),
    };
    assert_eq!(
        empty_endpoint
            .validate()
            .expect_err("nested endpoint is validated")
            .message(),
        "model endpoint must set an api_base, a provider, or both"
    );
}

#[test]
fn model_endpoint_redacts_api_key_and_requires_a_target() {
    let endpoint = ModelEndpoint {
        api_base: Some("https://api.example.test/v1".to_owned()),
        api_key: Some("sk-secret-key-value".to_owned()),
        provider: None,
    };
    let rendered = format!("{endpoint:?}");
    assert!(rendered.contains(r#"Some("<redacted>")"#), "{rendered}");
    assert!(!rendered.contains("sk-secret-key-value"), "{rendered}");
    endpoint.validate().expect("valid endpoint");

    let provider_only = ModelEndpoint {
        api_base: None,
        api_key: None,
        provider: Some("anthropic".to_owned()),
    };
    provider_only
        .validate()
        .expect("a provider alone identifies the endpoint");

    let untargeted = ModelEndpoint {
        api_base: None,
        api_key: Some("sk-key".to_owned()),
        provider: None,
    };
    assert_eq!(
        untargeted
            .validate()
            .expect_err("no api_base and no provider")
            .message(),
        "model endpoint must set an api_base, a provider, or both"
    );

    for api_key in [" ".to_owned(), String::new()] {
        let blank_key = ModelEndpoint {
            api_base: Some("https://api.example.test/v1".to_owned()),
            api_key: Some(api_key),
            provider: None,
        };
        assert_eq!(
            blank_key.validate().expect_err("blank api_key").message(),
            "model endpoint api_key must be a non-empty string when provided"
        );
    }
}

#[test]
fn delivery_modes_inputs_and_receipts_match_python() {
    assert_eq!(DeliveryMode::default(), DeliveryMode::Auto);
    assert_eq!(AbortMode::default(), AbortMode::Graceful);

    for (mode, wire) in [
        (DeliveryMode::Auto, "auto"),
        (DeliveryMode::Steer, "steer"),
        (DeliveryMode::FollowUp, "follow_up"),
    ] {
        assert_eq!(serde_json::to_value(mode).expect("serializes"), json!(wire));
    }
    for (mode, wire) in [
        (AbortMode::Graceful, "graceful"),
        (AbortMode::Force, "force"),
    ] {
        assert_eq!(serde_json::to_value(mode).expect("serializes"), json!(wire));
    }

    let input = HarnessInput::text("hi");
    assert_eq!(input.content, JsonValue::String("hi".to_owned()));
    assert!(input.metadata.is_empty());

    let receipt = SendReceipt {
        message_id: "m1".to_owned(),
        turn_id: "t1".to_owned(),
        accepted_mode: DeliveryMode::Steer,
    };
    receipt.validate().expect("valid receipt");

    let missing_message = SendReceipt {
        message_id: String::new(),
        ..receipt.clone()
    };
    assert_eq!(
        missing_message
            .validate()
            .expect_err("empty message id")
            .message(),
        "send receipt message_id must not be empty"
    );

    let missing_turn = SendReceipt {
        message_id: "m1".to_owned(),
        turn_id: String::new(),
        ..receipt
    };
    assert_eq!(
        missing_turn
            .validate()
            .expect_err("empty turn id")
            .message(),
        "send receipt turn_id must not be empty"
    );
}

#[test]
fn json_helpers_round_trip_and_validate_decodes() {
    // Python: tests/unit_tests/harness_protocol/test_protocol.py::test_protocol_json_values_are_validated_and_frozen
    let object: JsonObject =
        json_object_from_value(json!({"b": [1, 2], "a": {"nested": true}})).expect("object");
    assert_eq!(
        object.keys().map(String::as_str).collect::<Vec<_>>(),
        ["a", "b"]
    );

    let value = json_object_to_value(&object);
    assert_eq!(value, json!({"a": {"nested": true}, "b": [1, 2]}));
    assert_eq!(json_object_from_value(value), Some(object));

    assert!(json_object_from_value(JsonValue::Null).is_none());
    assert!(json_object_from_value(json!([1, 2])).is_none());
    assert!(json_object_from_value(json!("text")).is_none());
    assert!(empty_json_object().is_empty());

    let error =
        from_value_validated::<HarnessCard>(json!({"name": "", "implementation_version": "1"}))
            .expect_err("invalid card");
    assert_eq!(
        error.message(),
        "harness card identity and versions must not be empty"
    );

    let card: HarnessCard =
        from_value_validated(json!({"name": "acme", "implementation_version": "1.0.0"}))
            .expect("valid card");
    assert_eq!(card.name, "acme");

    let error = from_str_validated::<ModelOption>("{not json}").expect_err("invalid JSON");
    assert_eq!(error.kind(), HarnessErrorKind::Protocol);
}
