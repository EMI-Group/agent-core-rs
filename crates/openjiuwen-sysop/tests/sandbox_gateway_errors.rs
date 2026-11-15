//! The HTTP sandbox gateway distinguishes a capability gap (`199012`) from a real gateway
//! failure (`199007`).
//!
//! Every case runs against a **local** `SandboxProviderRegistry` built by
//! [`openjiuwen_sysop::sandbox::http::install`], so nothing is registered in
//! `sandbox::provider::global()`. The `base_url` points at a closed local port and
//! `HttpSandboxGateway::invoke` never opens a socket, so the whole file is offline and
//! deterministic.
//!
//! The end-to-end proof is possible without a mock server precisely because the gateway
//! reports its deferred protocol as `ErrorKind::Unsupported`: that is the error kind this
//! change maps to the new code.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(feature = "sandbox")]

use futures::StreamExt;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_sysop::code::types::CodeExecParams;
use openjiuwen_sysop::fs::types::ReadFileParams;
use openjiuwen_sysop::result::OpResult;
use openjiuwen_sysop::sandbox::config::{SandboxEndpoint, SandboxGatewayConfig};
use openjiuwen_sysop::sandbox::http::install;
use openjiuwen_sysop::sandbox::provider::SandboxProviderRegistry;
use openjiuwen_sysop::shell::types::ShellExecParams;
use openjiuwen_sysop::status::SysOpStatus;

/// The sandbox type every case registers under.
const SANDBOX_TYPE: &str = "aio_http_error_mock";

/// A loopback endpoint on a port nothing listens on: no case may open a socket.
fn endpoint() -> SandboxEndpoint {
    SandboxEndpoint {
        base_url: "http://127.0.0.1:1".to_string(),
        sandbox_id: Some("sandbox-1".to_string()),
        isolation_key: Some("session_pre_deploy_aio_http_error_mock_s1".to_string()),
    }
}

/// A registry with the HTTP gateway providers installed under [`SANDBOX_TYPE`].
fn registry() -> SandboxProviderRegistry {
    let registry = SandboxProviderRegistry::new();
    install(
        &registry,
        SANDBOX_TYPE,
        &endpoint(),
        &SandboxGatewayConfig::default(),
    )
    .unwrap();
    registry
}

/// Asserts the shape of an `Unsupported` failure: the new code, the Python gateway
/// envelope, and the underlying reason.
fn assert_capability_gap<T: std::fmt::Debug>(result: &OpResult<T>, method: &str) {
    assert_eq!(
        result.code, 199_012,
        "expected the Rust-only Unsupported code"
    );
    assert_eq!(result.status(), Some(SysOpStatus::Unsupported));
    assert_eq!(
        SysOpStatus::from_code(result.code),
        Some(SysOpStatus::Unsupported)
    );
    assert!(result.data().is_none(), "a failure carries no payload");
    let prefix = format!("sandbox gateway error, operation: {method}, reason: ");
    assert!(
        result.message.starts_with(&prefix),
        "message must keep the Python envelope, got: {}",
        result.message
    );
    // The reason names the real gap, not just "unsupported".
    assert!(
        result
            .message
            .contains("sandbox gateway protocol is not implemented"),
        "the underlying Unsupported message must survive, got: {}",
        result.message
    );
    assert!(
        result.message.contains("method: ") && result.message.contains(method),
        "the reason names the provider method, got: {}",
        result.message
    );
    // `199012` is not `199007`.
    assert_ne!(result.code, SysOpStatus::SandboxGatewayError.code());
}

#[tokio::test]
async fn filesystem_calls_report_a_capability_gap_not_a_gateway_failure() {
    let fs = registry()
        .create_fs(SANDBOX_TYPE, &endpoint(), &SandboxGatewayConfig::default())
        .unwrap();
    let result = fs.read_file(ReadFileParams::text("x")).await;
    assert_capability_gap(&result, "read_file");

    let error = result.into_result().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error
        .message()
        .contains("sandbox gateway error, operation: read_file"));
}

#[tokio::test]
async fn shell_calls_report_a_capability_gap() {
    let shell = registry()
        .create_shell(SANDBOX_TYPE, &endpoint(), &SandboxGatewayConfig::default())
        .unwrap();
    let result = shell.execute_cmd(ShellExecParams::new("echo hi")).await;
    assert_capability_gap(&result, "execute_cmd");
    assert_eq!(
        result.into_optional_result().unwrap_err().kind(),
        ErrorKind::Unsupported
    );
}

#[tokio::test]
async fn code_calls_report_a_capability_gap() {
    let code = registry()
        .create_code(SANDBOX_TYPE, &endpoint(), &SandboxGatewayConfig::default())
        .unwrap();
    let result = code.execute_code(CodeExecParams::new("print(1)")).await;
    assert_capability_gap(&result, "execute_code");
}

#[tokio::test]
async fn streaming_calls_yield_exactly_one_199012_chunk() {
    let shell = registry()
        .create_shell(SANDBOX_TYPE, &endpoint(), &SandboxGatewayConfig::default())
        .unwrap();
    let mut chunks = shell.execute_cmd_stream(ShellExecParams::new("echo hi"));
    let first = chunks.next().await.unwrap();
    assert_capability_gap(&first, "execute_cmd_stream");
    assert!(chunks.next().await.is_none(), "exactly one error chunk");

    let fs = registry()
        .create_fs(SANDBOX_TYPE, &endpoint(), &SandboxGatewayConfig::default())
        .unwrap();
    let mut chunks = fs.read_file_stream(ReadFileParams::text("x"));
    let first = chunks.next().await.unwrap();
    assert_capability_gap(&first, "read_file_stream");
    assert_eq!(
        first.into_result().unwrap_err().kind(),
        ErrorKind::Unsupported
    );
    assert!(chunks.next().await.is_none());
}

#[tokio::test]
async fn the_capability_gap_code_round_trips_as_a_status() {
    let fs = registry()
        .create_fs(SANDBOX_TYPE, &endpoint(), &SandboxGatewayConfig::default())
        .unwrap();
    let result = fs.read_file(ReadFileParams::text("x")).await;
    assert_eq!(result.status(), Some(SysOpStatus::Unsupported));
    assert_eq!(SysOpStatus::Unsupported.code(), 199_012);
    assert_eq!(SysOpStatus::Unsupported.name(), "unsupported");
    // The wire shape only differs in `code`: the message text stays Python's envelope.
    let json = serde_json::to_value(&result).unwrap();
    assert_eq!(json["code"], 199_012);
    assert!(json.get("data").is_none());
    assert!(json["message"]
        .as_str()
        .unwrap()
        .starts_with("sandbox gateway error, operation: read_file, reason: "));
}
