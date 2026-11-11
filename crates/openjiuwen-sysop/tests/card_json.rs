//! Ports `tests/unit_tests/core/sys_operation/test_sys_operation.py` (card/config JSON parity).
//!
//! Covers the registry-safe surface of a sys-operation service: `SysOperationCard` field
//! names and defaults, mode parsing, validation (199002), tool-id generation, and the
//! serialized shape of `LocalWorkConfig` and the sandbox gateway configuration.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use openjiuwen_sysop::card::{generate_tool_id, SysOperationCard};
use openjiuwen_sysop::config::{EnvInheritMode, LocalWorkConfig};
use openjiuwen_sysop::constants::{DEFAULT_MAX_OUTPUT_BYTES, DEFAULT_TIMEOUT_SECS};
use openjiuwen_sysop::mode::{OperationKind, OperationMode};
use openjiuwen_sysop::sandbox::config::{
    ContainerScope, OnStop, SandboxGatewayConfig, SandboxIsolationConfig, SandboxLauncherConfig,
};
use serde_json::json;

#[test]
fn card_json_uses_python_field_names() {
    let card = SysOperationCard::new("sys_op")
        .with_name("ops")
        .with_mode(OperationMode::Local);
    let json = serde_json::to_value(&card).unwrap();

    assert_eq!(json["id"], "sys_op", "card id must serialize as `id`");
    assert_eq!(json["name"], "ops", "card name must serialize as `name`");
    assert_eq!(
        json["mode"], "local",
        "mode must serialize as the lowercase Python name"
    );
    assert!(
        json.get("work_config")
            .is_some_and(serde_json::Value::is_null),
        "an unset work_config must be present as JSON null, got {json}"
    );
    assert!(
        json.get("gateway_config")
            .is_some_and(serde_json::Value::is_null),
        "an unset gateway_config must be present as JSON null, got {json}"
    );
    assert!(
        json.get("description").is_none(),
        "an absent description must be omitted, got {json}"
    );

    let described = card.with_description("system operations");
    let json = serde_json::to_value(&described).unwrap();
    assert_eq!(
        json["description"], "system operations",
        "a set description must be serialized"
    );
}

#[test]
fn card_json_round_trips() {
    let card = SysOperationCard::new("sys_op")
        .with_name("ops")
        .with_description("system operations")
        .with_work_config(LocalWorkConfig::unrestricted("/tmp/ws"))
        .with_gateway_config(SandboxGatewayConfig {
            launcher_config: Some(SandboxLauncherConfig::default()),
            ..SandboxGatewayConfig::default()
        });
    let json = serde_json::to_value(&card).unwrap();
    let restored: SysOperationCard = serde_json::from_value(json.clone()).unwrap();
    assert_eq!(restored, card, "card must round-trip through JSON: {json}");
}

#[test]
fn card_mode_deserialization_is_case_insensitive() {
    let local: SysOperationCard =
        serde_json::from_value(json!({"id": "a", "mode": "LOCAL"})).unwrap();
    assert_eq!(
        local.mode,
        OperationMode::Local,
        "'LOCAL' must parse as the local mode"
    );

    let sandbox: SysOperationCard =
        serde_json::from_value(json!({"id": "a", "mode": "Sandbox"})).unwrap();
    assert_eq!(
        sandbox.mode,
        OperationMode::Sandbox,
        "'Sandbox' must parse as the sandbox mode"
    );

    let error = serde_json::from_value::<SysOperationCard>(json!({"id": "a", "mode": "remote"}))
        .unwrap_err();
    assert!(
        error.to_string().contains("invalid operation mode"),
        "an unknown mode must be a serde error, got: {error}"
    );
}

#[test]
fn default_card_id_is_generated_uuid_hex() {
    let default_card = SysOperationCard::default();
    assert_eq!(
        default_card.id.len(),
        32,
        "default card id must be a 32-char uuid hex, got {:?}",
        default_card.id
    );
    assert!(
        default_card
            .id
            .chars()
            .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
        "default card id must be lowercase hex, got {:?}",
        default_card.id
    );
    assert_ne!(
        default_card.id,
        SysOperationCard::default().id,
        "two default cards must not share an id"
    );

    let card = SysOperationCard::new("x");
    assert_eq!(card.id, "x", "new() must keep the supplied id");
    assert_eq!(
        card.mode,
        OperationMode::Local,
        "new() must default to local mode"
    );
}

#[test]
fn validate_rejects_blank_id() {
    for card in [
        SysOperationCard::new(""),
        SysOperationCard::new("   "),
        SysOperationCard::new("\t\n"),
    ] {
        let error = card.validate().unwrap_err();
        assert!(
            error
                .message()
                .contains("sys operation card param error, reason: card id must not be empty"),
            "unexpected message for id {:?}: {}",
            card.id,
            error.message()
        );
    }
}

#[test]
fn validate_rejects_sandbox_without_gateway_config() {
    let card = SysOperationCard::new("sys_op").with_mode(OperationMode::Sandbox);
    let error = card.validate().unwrap_err();
    assert!(
        error
            .message()
            .contains("gateway_config is required when mode is 'sandbox'"),
        "unexpected message: {}",
        error.message()
    );
}

#[test]
fn validate_rejects_sandbox_without_launcher_config() {
    let card = SysOperationCard::new("sys_op")
        .with_mode(OperationMode::Sandbox)
        .with_gateway_config(SandboxGatewayConfig::default());
    let error = card.validate().unwrap_err();
    assert!(
        error
            .message()
            .contains("launcher_config is required when mode is 'sandbox'"),
        "unexpected message: {}",
        error.message()
    );
}

#[test]
fn validate_accepts_local_and_configured_sandbox_cards() {
    let local = SysOperationCard::new("sys_op");
    assert!(
        local.validate().is_ok(),
        "a local card must validate: {:?}",
        local.validate()
    );

    let sandbox = SysOperationCard::new("sys_op")
        .with_mode(OperationMode::Sandbox)
        .with_gateway_config(SandboxGatewayConfig {
            launcher_config: Some(SandboxLauncherConfig::default()),
            ..SandboxGatewayConfig::default()
        });
    assert!(
        sandbox.validate().is_ok(),
        "a sandbox card with a launcher config must validate: {:?}",
        sandbox.validate()
    );
}

#[test]
fn tool_id_shape_matches_python() {
    let card = SysOperationCard::new("sys_op");
    assert_eq!(
        card.tool_id(OperationKind::Fs, "read_file"),
        "sys_op.fs.read_file",
        "fs tool ids are `<card>.fs.<method>`"
    );
    assert_eq!(
        card.tool_id(OperationKind::Shell, "execute_cmd"),
        "sys_op.shell.execute_cmd",
        "shell tool ids are `<card>.shell.<method>`"
    );
    assert_eq!(
        card.tool_id(OperationKind::Code, "execute_code"),
        "sys_op.code.execute_code",
        "code tool ids are `<card>.code.<method>`"
    );
    assert_eq!(
        generate_tool_id("c", OperationKind::Code, "execute_code"),
        "c.code.execute_code",
        "generate_tool_id must match the card method"
    );
}

#[test]
fn local_work_config_defaults_match_python() {
    let config = LocalWorkConfig::default();
    let allowlist = config
        .shell_allowlist
        .as_ref()
        .expect("the default allowlist must be present, not `None`");
    assert_eq!(
        allowlist.len(),
        30,
        "Python's default allowlist has 30 entries"
    );
    assert_eq!(
        allowlist.first().map(String::as_str),
        Some("echo"),
        "the default allowlist starts with echo"
    );
    assert_eq!(
        allowlist.last().map(String::as_str),
        Some("ping"),
        "the default allowlist ends with ping"
    );
    assert!(
        !config.restrict_to_sandbox,
        "restrict_to_sandbox defaults to false (Python parity)"
    );
    assert_eq!(
        config.max_output_bytes, DEFAULT_MAX_OUTPUT_BYTES,
        "max_output_bytes must use the crate default"
    );
    assert_eq!(
        config.default_timeout_secs, DEFAULT_TIMEOUT_SECS,
        "default_timeout_secs must be 300"
    );
    assert_eq!(config.default_timeout_secs, 300, "Python timeout is 300s");
    assert_eq!(
        config.env_policy.mode,
        EnvInheritMode::All,
        "the default environment policy inherits everything"
    );
    assert_eq!(
        config.max_file_size_bytes, None,
        "max_file_size_bytes is unlimited by default (Python parity)"
    );
    assert_eq!(
        config.sandbox_root, None,
        "sandbox_root is unset by default (Python parity)"
    );
}

#[test]
fn null_shell_allowlist_means_allow_all() {
    let unset: LocalWorkConfig = serde_json::from_value(json!({"shell_allowlist": null})).unwrap();
    assert_eq!(
        unset.shell_allowlist, None,
        "a JSON null allowlist must deserialize to None (allow every command)"
    );

    let empty: LocalWorkConfig = serde_json::from_value(json!({"shell_allowlist": []})).unwrap();
    assert_eq!(
        empty.shell_allowlist,
        Some(Vec::new()),
        "an empty JSON allowlist must stay an empty list"
    );

    let explicit: LocalWorkConfig =
        serde_json::from_value(json!({"shell_allowlist": ["git", "ls"]})).unwrap();
    assert_eq!(
        explicit.shell_allowlist,
        Some(vec!["git".to_string(), "ls".to_string()]),
        "an explicit allowlist must keep its entries"
    );
}

#[test]
fn sandbox_gateway_config_defaults_match_python() {
    let config = SandboxGatewayConfig::default();
    assert_eq!(
        config.timeout_seconds, 30,
        "the gateway timeout defaults to 30 seconds"
    );
    assert!(
        config.launcher_config.is_none(),
        "launcher_config is unset by default and required only for sandbox cards"
    );
    assert!(
        !config.supports_conditional_write,
        "supports_conditional_write defaults to false"
    );
    assert!(
        config.auth_headers.is_empty(),
        "auth_headers must start empty, got {:?}",
        config.auth_headers
    );
    assert!(
        config.auth_query_params.is_empty(),
        "auth_query_params must start empty, got {:?}",
        config.auth_query_params
    );
    assert_eq!(
        config.isolation.container_scope,
        ContainerScope::Session,
        "the isolation container scope defaults to session"
    );

    let launcher = SandboxLauncherConfig::default();
    assert_eq!(
        launcher.on_stop,
        OnStop::Delete,
        "on_stop defaults to delete"
    );
    assert_eq!(
        launcher.launcher_type, "pre_deploy",
        "launcher_type default"
    );
    assert_eq!(launcher.sandbox_type, "aio", "sandbox_type default");
    assert_eq!(
        launcher.idle_ttl_seconds, None,
        "idle_ttl_seconds is unset by default"
    );
    assert!(
        launcher.extra_params.is_empty(),
        "extra_params must start empty, got {:?}",
        launcher.extra_params
    );
}

#[test]
fn sandbox_config_json_uses_snake_case_field_names() {
    let gateway = serde_json::to_value(SandboxGatewayConfig::default()).unwrap();
    for key in [
        "isolation",
        "launcher_config",
        "timeout_seconds",
        "auth_headers",
        "auth_query_params",
        "supports_conditional_write",
    ] {
        assert!(
            gateway.get(key).is_some(),
            "gateway JSON is missing `{key}`: {gateway}"
        );
    }

    let launcher = serde_json::to_value(SandboxLauncherConfig::default()).unwrap();
    for key in [
        "launcher_type",
        "gateway_url",
        "sandbox_type",
        "on_stop",
        "idle_ttl_seconds",
        "extra_params",
        "base_url",
    ] {
        assert!(
            launcher.get(key).is_some(),
            "launcher JSON is missing `{key}`: {launcher}"
        );
    }
    assert_eq!(
        launcher["on_stop"], "delete",
        "OnStop must serialize snake_case"
    );

    let isolation = serde_json::to_value(SandboxIsolationConfig {
        custom_id: Some("tenant".to_string()),
        container_scope: ContainerScope::Custom,
        prefix: Some("agent1_".to_string()),
    })
    .unwrap();
    for key in ["custom_id", "container_scope", "prefix"] {
        assert!(
            isolation.get(key).is_some(),
            "isolation JSON is missing `{key}`: {isolation}"
        );
    }
    assert_eq!(
        isolation["container_scope"], "custom",
        "ContainerScope must serialize lowercase"
    );
    assert_eq!(
        isolation["prefix"], "agent1_",
        "the isolation prefix must round-trip verbatim"
    );
}

#[test]
fn unknown_fields_are_tolerated() {
    let card: SysOperationCard = serde_json::from_value(json!({
        "id": "sys_op",
        "name": "ops",
        "mode": "local",
        "future_field": 1,
        "nested": {"anything": [1, 2, 3]},
    }))
    .unwrap();
    assert_eq!(card.id, "sys_op", "known card fields must survive");
    assert_eq!(card.name, "ops", "known card fields must survive");
    assert_eq!(card.mode, OperationMode::Local);

    let config: LocalWorkConfig = serde_json::from_value(json!({
        "shell_allowlist": ["git"],
        "future_limit": 512,
    }))
    .unwrap();
    assert_eq!(
        config.shell_allowlist,
        Some(vec!["git".to_string()]),
        "known config fields must survive an unknown sibling"
    );
    assert_eq!(
        config.max_output_bytes, DEFAULT_MAX_OUTPUT_BYTES,
        "omitted fields keep their defaults"
    );

    let gateway: SandboxGatewayConfig =
        serde_json::from_value(json!({"timeout_seconds": 5, "future_flag": true})).unwrap();
    assert_eq!(
        gateway.timeout_seconds, 5,
        "known gateway fields must survive an unknown sibling"
    );
    assert!(
        gateway.launcher_config.is_none(),
        "omitted gateway fields keep their defaults"
    );
}
