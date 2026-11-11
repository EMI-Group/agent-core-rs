//! Ports `tests/unit_tests/core/sys_operation/sandbox/` (public-API failure paths without a
//! remote provider).
//!
//! Every case stays on a path that never installs a provider in the process-global registry:
//! either the card is rejected during validation (code 199002) or the launcher carries no
//! usable `base_url`, so dispatch reports the missing provider as `ErrorKind::Unsupported`.
//! No network call and no subprocess is ever made.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(feature = "sandbox")]

use openjiuwen_sysop::card::SysOperationCard;
use openjiuwen_sysop::config::LocalWorkConfig;
use openjiuwen_sysop::mode::OperationMode;
use openjiuwen_sysop::sandbox::config::{
    generate_isolation_key_template, isolation_key_template_from_config, resolve_isolation_key,
    ContainerScope, SandboxGatewayConfig, SandboxIsolationConfig, SandboxLauncherConfig,
};
use openjiuwen_sysop::sandbox::operation::{
    sandbox_operations, sandbox_operations_for_session, unsupported_sandbox,
};
use openjiuwen_sysop::SysOperation;

/// A launcher that names `sandbox_type` but never carries a `base_url`, so nothing is
/// installed in the provider registry.
fn launcher(sandbox_type: &str) -> SandboxLauncherConfig {
    SandboxLauncherConfig {
        sandbox_type: sandbox_type.to_string(),
        ..SandboxLauncherConfig::default()
    }
}

/// A sandbox gateway configuration with a launcher and no `base_url`.
fn gateway_for(sandbox_type: &str) -> SandboxGatewayConfig {
    SandboxGatewayConfig {
        launcher_config: Some(launcher(sandbox_type)),
        ..SandboxGatewayConfig::default()
    }
}

/// A sandbox-mode card, with or without a gateway configuration.
fn sandbox_card(id: &str, gateway: Option<SandboxGatewayConfig>) -> SysOperationCard {
    let card = SysOperationCard::new(id).with_mode(OperationMode::Sandbox);
    match gateway {
        Some(gateway) => card.with_gateway_config(gateway),
        None => card,
    }
}

/// The message of the error `SysOperation::new` reports for `card`.
fn construction_error(card: SysOperationCard) -> String {
    SysOperation::new(card).unwrap_err().message().into_owned()
}

#[test]
fn sandbox_card_without_gateway_config_is_rejected() {
    let message = construction_error(sandbox_card("sbx", None));
    assert!(
        message.contains(
            "sys operation card param error, reason: gateway_config is required when mode is 'sandbox'"
        ),
        "unexpected message: {message}"
    );
}

#[test]
fn sandbox_card_without_launcher_config_is_rejected() {
    // `SandboxGatewayConfig::default()` has `launcher_config: None`.
    let card = sandbox_card("sbx", Some(SandboxGatewayConfig::default()));
    let message = construction_error(card);
    assert!(
        message.contains(
            "sys operation card param error, reason: launcher_config is required when mode is 'sandbox'"
        ),
        "unexpected message: {message}"
    );
}

#[test]
fn launcher_with_empty_type_is_rejected() {
    let gateway = SandboxGatewayConfig {
        launcher_config: Some(SandboxLauncherConfig {
            launcher_type: String::new(),
            ..launcher("aio")
        }),
        ..SandboxGatewayConfig::default()
    };
    let message = construction_error(sandbox_card("sbx", Some(gateway)));
    assert!(
        message.contains("launcher_type must not be empty"),
        "unexpected message: {message}"
    );

    let gateway = SandboxGatewayConfig {
        launcher_config: Some(SandboxLauncherConfig {
            sandbox_type: String::new(),
            ..launcher("aio")
        }),
        ..SandboxGatewayConfig::default()
    };
    let message = construction_error(sandbox_card("sbx", Some(gateway)));
    assert!(
        message.contains("sandbox_type must not be empty"),
        "unexpected message: {message}"
    );
}

#[test]
fn construction_without_a_registered_provider_is_unsupported() {
    // No `base_url`, so `register_builtin_providers` installs nothing and dispatch reports
    // the missing provider for the filesystem kind first.
    let message = construction_error(sandbox_card("sbx", Some(gateway_for("no-such-provider"))));
    assert!(
        message.contains("Sandbox type 'no-such-provider' does not support operation 'fs'"),
        "unexpected message: {message}"
    );
}

#[test]
fn blank_base_url_does_not_install_a_provider() {
    let gateway = SandboxGatewayConfig {
        launcher_config: Some(SandboxLauncherConfig {
            base_url: Some("   ".to_string()),
            ..launcher("no-such-provider")
        }),
        ..SandboxGatewayConfig::default()
    };
    let message = construction_error(sandbox_card("sbx", Some(gateway)));
    assert!(
        message.contains("Sandbox type 'no-such-provider' does not support operation 'fs'"),
        "a blank base_url must not register a provider, got: {message}"
    );
    assert!(
        !message.to_lowercase().contains("gateway")
            && !message.to_lowercase().contains("transport"),
        "a blank base_url must fail before any transport error, got: {message}"
    );
}

#[test]
fn sandbox_operations_reports_a_missing_provider() {
    let card = sandbox_card("sbx", Some(gateway_for("no-such-provider")));
    let error = sandbox_operations(&card).unwrap_err();
    assert!(
        error
            .message()
            .contains("Sandbox type 'no-such-provider' does not support operation 'fs'"),
        "unexpected message: {}",
        error.message()
    );
}

#[test]
fn sandbox_operations_for_session_also_reports_missing_provider() {
    let card = sandbox_card("sbx", Some(gateway_for("no-such-provider")));
    let error = sandbox_operations_for_session(&card, Some("s1")).unwrap_err();
    assert!(
        error
            .message()
            .contains("Sandbox type 'no-such-provider' does not support operation 'fs'"),
        "unexpected message: {}",
        error.message()
    );
}

#[test]
fn empty_card_id_is_rejected_before_provider_dispatch() {
    let card = sandbox_card("  ", Some(gateway_for("aio")));
    let message = construction_error(card);
    assert!(
        message.contains("card id must not be empty"),
        "unexpected message: {message}"
    );
}

#[test]
fn local_mode_has_no_isolation_key_template() {
    let workspace = tempfile::tempdir().unwrap();
    let config = LocalWorkConfig::unrestricted(workspace.path().to_string_lossy().into_owned());
    let operation = SysOperation::local(config).unwrap();
    assert_eq!(operation.mode(), OperationMode::Local);
    assert!(
        operation.isolation_key_template().is_none(),
        "local mode has no isolation-key template"
    );
    assert!(
        operation.fs().is_some(),
        "local mode provides a filesystem backend"
    );
}

#[test]
fn isolation_key_template_shape() {
    let session = generate_isolation_key_template(
        Some("agent1_"),
        ContainerScope::Session,
        None,
        "pre_deploy",
        "aio",
    )
    .unwrap();
    assert_eq!(session, "session_pre_deploy_aio_agent1_{session_id}");

    let system = generate_isolation_key_template(
        Some("agent1_"),
        ContainerScope::System,
        None,
        "pre_deploy",
        "aio",
    )
    .unwrap();
    assert_eq!(system, "system_pre_deploy_aio_agent1_system");

    let custom = generate_isolation_key_template(
        Some("agent1_"),
        ContainerScope::Custom,
        Some("alice"),
        "pre_deploy",
        "aio",
    )
    .unwrap();
    assert_eq!(custom, "custom_pre_deploy_aio_agent1_alice");
}

#[test]
fn custom_scope_requires_a_custom_id() {
    let error =
        generate_isolation_key_template(None, ContainerScope::Custom, None, "l", "s").unwrap_err();
    assert!(
        error
            .message()
            .contains("custom_id is required when container_scope is 'custom'"),
        "unexpected message: {}",
        error.message()
    );

    let error = generate_isolation_key_template(None, ContainerScope::Custom, Some("  "), "l", "s")
        .unwrap_err();
    assert!(
        error
            .message()
            .contains("custom_id is required when container_scope is 'custom'"),
        "unexpected message: {}",
        error.message()
    );
}

#[test]
fn isolation_key_placeholder_is_resolved_for_the_default_session() {
    assert_eq!(
        resolve_isolation_key("x_{session_id}", None),
        "x_default_session"
    );
    assert_eq!(resolve_isolation_key("x_{session_id}", Some("s1")), "x_s1");
    assert_eq!(
        resolve_isolation_key("a_{session_id}_{session_id}", Some("s1")),
        "a_s1_s1"
    );
}

#[test]
fn isolation_key_template_from_config_uses_launcher_and_isolation_fields() {
    let gateway = SandboxGatewayConfig {
        isolation: SandboxIsolationConfig {
            prefix: Some("p_".to_string()),
            container_scope: ContainerScope::Session,
            custom_id: None,
        },
        launcher_config: Some(SandboxLauncherConfig {
            launcher_type: "pre_deploy".to_string(),
            sandbox_type: "aio".to_string(),
            ..SandboxLauncherConfig::default()
        }),
        ..SandboxGatewayConfig::default()
    };
    assert_eq!(
        isolation_key_template_from_config(&gateway).unwrap(),
        "session_pre_deploy_aio_p_{session_id}"
    );

    let error = isolation_key_template_from_config(&SandboxGatewayConfig::default()).unwrap_err();
    assert!(
        error
            .message()
            .contains("launcher_config is required when mode is 'sandbox'"),
        "unexpected message: {}",
        error.message()
    );
}

#[test]
fn unsupported_sandbox_error_names_the_operation() {
    let error = unsupported_sandbox("read_file");
    assert!(
        error
            .message()
            .contains("sandbox operation 'read_file' is not available"),
        "unexpected message: {}",
        error.message()
    );
}
