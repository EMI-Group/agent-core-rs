//! Unit tests of the `aio` remote sandbox provider.
//!
//! Python counterpart: `tests/unit_tests/extensions/sys_operation/sandbox/test_aio_*.py`,
//! which only asserts registration; the behavior there is integration-gated. These
//! tests therefore pin the wire contract instead — the kernel gateway envelope, the
//! `params` names of the Python provider signatures, and the mapping of `data` back
//! into the kernel result structs — against a loopback `wiremock` server.
//!
//! This root suite covers the envelope, authentication, invoke path, error
//! mapping, conditional-write capability, and registration; the per-operation wire
//! contracts live in `tests/fs.rs`, `tests/shell.rs`, and `tests/code.rs`.
// A test module owns its panics: a failed `unwrap`/`expect` is an assertion.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use serde_json::{json, Value};
use tempfile::tempdir;
use wiremock::matchers::{body_partial_json, header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

use openjiuwen_common::error::ErrorKind;
use openjiuwen_core::sys_operation::code::types::{CodeChunk, CodeExecParams, CodeOutput};
use openjiuwen_core::sys_operation::fs::types::{
    DownloadFileData, DownloadFileParams, FileContent, FileSystemItem, LineRange, ListDirsParams,
    ListFilesData, ListFilesParams, ReadChunkData, ReadFileData, ReadFileParams, ReadMode,
    SearchFilesData, SearchFilesParams, SortBy, TransferChunkData, UploadFileData,
    UploadFileParams, WriteFileData, WriteFileParams,
};
use openjiuwen_core::sys_operation::operation::{
    CodeOperation, FsOperation, OpStream, ShellOperation,
};
use openjiuwen_core::sys_operation::result::OpResult;
use openjiuwen_core::sys_operation::sandbox::config::{
    SandboxEndpoint, SandboxGatewayConfig, SandboxLauncherConfig,
};
use openjiuwen_core::sys_operation::shell::types::{
    ShellArgvParams, ShellBackgroundOutput, ShellChunk, ShellExecParams, ShellOutput, ShellType,
    StreamKind,
};

use super::*;

mod code;
mod fs;
mod shell;

// ─── fixtures ───────────────────────────────────────────────────────────────

/// An endpoint pointing at the loopback server, with a resolved isolation key.
fn endpoint(server: &MockServer) -> SandboxEndpoint {
    SandboxEndpoint {
        base_url: server.uri(),
        sandbox_id: None,
        isolation_key: Some("session_pre_deploy_aio_s1".to_string()),
    }
}

/// A gateway configuration with the kernel defaults.
fn gateway_config() -> SandboxGatewayConfig {
    SandboxGatewayConfig::default()
}

fn client(server: &MockServer) -> Arc<AioClient> {
    Arc::new(AioClient::new(&endpoint(server), &gateway_config()).unwrap())
}

fn fs(server: &MockServer) -> AioFs {
    AioFs {
        client: client(server),
    }
}

fn fs_arc(server: &MockServer) -> Arc<AioFs> {
    Arc::new(fs(server))
}

fn shell(server: &MockServer) -> AioShell {
    AioShell {
        client: client(server),
    }
}

fn shell_arc(server: &MockServer) -> Arc<AioShell> {
    Arc::new(shell(server))
}

fn code(server: &MockServer) -> AioCode {
    AioCode {
        client: client(server),
    }
}

fn code_arc(server: &MockServer) -> Arc<AioCode> {
    Arc::new(code(server))
}

/// Answers every envelope with `body`.
async fn respond(server: &MockServer, body: Value) {
    Mock::given(method("POST"))
        .and(path(DEFAULT_INVOKE_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(server)
        .await;
}

/// Answers a successful envelope carrying `data`.
async fn respond_data(server: &MockServer, data: Value) {
    respond(server, json!({"code": 0, "message": "OK", "data": data})).await;
}

/// The decoded envelopes the server received, in arrival order.
async fn envelopes(server: &MockServer) -> Vec<Value> {
    server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .map(|request| request.body_json::<Value>().unwrap())
        .collect()
}

/// The last envelope the server received.
async fn last_envelope(server: &MockServer) -> Value {
    envelopes(server).await.pop().unwrap()
}

/// The remote paths a listing returned, in order.
fn names(items: &[FileSystemItem]) -> Vec<&str> {
    items.iter().map(|item| item.name.as_str()).collect()
}

/// Collects a buffered operation stream.
async fn collect<T>(stream: OpStream<T>) -> Vec<OpResult<T>> {
    stream.collect().await
}

// ─── envelope shape ─────────────────────────────────────────────────────────

#[tokio::test]
async fn every_call_posts_the_gateway_envelope() {
    let server = MockServer::start().await;
    respond_data(&server, json!({"path": "/work/a.txt", "content": "hi"})).await;

    let result = fs(&server)
        .read_file(ReadFileParams::text("/work/a.txt"))
        .await;
    assert_eq!(result.code, 0);

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method.as_str(), "POST");
    assert_eq!(requests[0].url.path(), DEFAULT_INVOKE_PATH);
    let envelope = envelopes(&server).await.remove(0);
    assert_eq!(envelope["op_type"], "fs");
    assert_eq!(envelope["method"], "read_file");
    assert_eq!(envelope["isolation_key"], "session_pre_deploy_aio_s1");
    assert_eq!(envelope["params"]["path"], "/work/a.txt");
    assert_eq!(envelope["params"]["mode"], "text");
}

#[tokio::test]
async fn a_missing_isolation_key_is_sent_as_null() {
    let server = MockServer::start().await;
    respond_data(&server, json!({"content": "hi"})).await;
    let client = Arc::new(
        AioClient::new(
            &SandboxEndpoint {
                base_url: server.uri(),
                ..SandboxEndpoint::default()
            },
            &gateway_config(),
        )
        .unwrap(),
    );
    let fs = AioFs { client };
    let result = fs.read_file(ReadFileParams::text("/work/a.txt")).await;
    assert_eq!(result.code, 0);
    assert!(last_envelope(&server).await["isolation_key"].is_null());
}

#[tokio::test]
async fn the_invoke_path_can_be_overridden_by_the_launcher_configuration() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/custom/invoke"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"code": 0, "data": {}})))
        .mount(&server)
        .await;
    let config = SandboxGatewayConfig {
        launcher_config: Some(SandboxLauncherConfig {
            base_url: None,
            extra_params: BTreeMap::from([(INVOKE_PATH_PARAM.to_string(), json!("custom/invoke"))]),
            ..SandboxLauncherConfig::default()
        }),
        ..SandboxGatewayConfig::default()
    };
    let fs = AioFs {
        client: Arc::new(AioClient::new(&endpoint(&server), &config).unwrap()),
    };

    let result = fs.read_file(ReadFileParams::text("/work/a.txt")).await;

    assert_eq!(result.code, 0);
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests[0].url.path(), "/custom/invoke");
}

#[tokio::test]
async fn auth_headers_and_query_parameters_reach_the_gateway() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(DEFAULT_INVOKE_PATH))
        .and(header("x-api-key", "k-1"))
        .and(query_param("token", "s1"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"code": 0, "data": {}})))
        .mount(&server)
        .await;
    let mut config = SandboxGatewayConfig::default();
    config
        .auth_headers
        .insert("x-api-key".to_string(), "k-1".to_string());
    config
        .auth_query_params
        .insert("token".to_string(), "s1".to_string());
    let fs = AioFs {
        client: Arc::new(AioClient::new(&endpoint(&server), &config).unwrap()),
    };

    let result = fs.read_file(ReadFileParams::text("/work/a.txt")).await;

    assert_eq!(result.code, 0);
}

// ─── gateway failures ───────────────────────────────────────────────────────

#[tokio::test]
async fn a_gateway_failure_becomes_the_sandbox_gateway_status() {
    let server = MockServer::start().await;
    respond(
        &server,
        json!({"code": -1, "message": "container vanished"}),
    )
    .await;

    let result = fs(&server)
        .read_file(ReadFileParams::text("/work/a.txt"))
        .await;

    assert_eq!(result.code, 199_007);
    assert_eq!(
        result.message,
        "sandbox gateway error, operation: read_file, \
         reason: aio gateway code -1: container vanished"
    );
    assert!(result.data.is_none());
}

#[tokio::test]
async fn a_gateway_rejection_names_the_calling_operation() {
    let server = MockServer::start().await;
    respond(
        &server,
        json!({"code": -1, "message": "methods unavailable"}),
    )
    .await;

    let result = shell(&server)
        .execute_cmd_background(ShellExecParams::new("sleep 1"))
        .await;

    assert_eq!(result.code, 199_007);
    assert_eq!(
        result.message,
        "sandbox gateway error, operation: execute_cmd_background, \
         reason: aio gateway code -1: methods unavailable"
    );
    assert!(result.data.is_none());
}

#[tokio::test]
async fn a_non_json_gateway_answer_is_a_transport_failure() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(DEFAULT_INVOKE_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_string("<html></html>"))
        .mount(&server)
        .await;

    let result = fs(&server)
        .read_file(ReadFileParams::text("/work/a.txt"))
        .await;

    assert_eq!(result.code, 199_007);
    assert!(result.message.contains("could not decode"), "{result:?}");
}

#[tokio::test]
async fn an_unreachable_gateway_is_a_transport_failure() {
    // Bind then drop a listener so the port is (almost certainly) closed.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    let client = Arc::new(
        AioClient::new(
            &SandboxEndpoint {
                base_url: format!("http://{address}"),
                ..SandboxEndpoint::default()
            },
            &gateway_config(),
        )
        .unwrap(),
    );
    let fs = AioFs { client };

    let result = fs.read_file(ReadFileParams::text("/work/a.txt")).await;

    assert_eq!(result.code, 199_007);
    assert!(result.message.contains("request failed"), "{result:?}");
}

#[tokio::test]
async fn a_gateway_that_exceeds_the_configured_timeout_fails_without_panicking() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(DEFAULT_INVOKE_PATH))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"code": 0, "data": {}}))
                .set_delay(Duration::from_secs(5)),
        )
        .mount(&server)
        .await;
    let config = SandboxGatewayConfig {
        timeout_seconds: 1,
        ..SandboxGatewayConfig::default()
    };
    let fs = AioFs {
        client: Arc::new(AioClient::new(&endpoint(&server), &config).unwrap()),
    };

    let result = fs.read_file(ReadFileParams::text("/work/a.txt")).await;

    assert_eq!(result.code, 199_007);
    assert!(result.message.contains("request failed"), "{result:?}");
}

// ─── conditional writes ─────────────────────────────────────────────────────

#[tokio::test]
async fn conditional_writes_are_always_supported_by_the_provider() {
    let server = MockServer::start().await;
    for advertised in [true, false] {
        let config = SandboxGatewayConfig {
            supports_conditional_write: advertised,
            ..SandboxGatewayConfig::default()
        };
        let fs = AioFs {
            client: Arc::new(AioClient::new(&endpoint(&server), &config).unwrap()),
        };
        assert!(
            fs.supports_conditional_write(),
            "the provider evaluates the precondition itself, \
             so the gateway flag {advertised} must not matter"
        );
    }
}

#[tokio::test]
async fn the_provider_set_factories_advertise_conditional_writes() {
    let endpoint = SandboxEndpoint {
        base_url: "http://127.0.0.1:1".to_string(),
        ..SandboxEndpoint::default()
    };
    for advertised in [true, false] {
        let config = SandboxGatewayConfig {
            supports_conditional_write: advertised,
            ..SandboxGatewayConfig::default()
        };
        let factory = provider_set().fs.expect("a filesystem factory");
        assert!(factory(&endpoint, &config)
            .unwrap()
            .supports_conditional_write());
    }
}

// ─── registration ───────────────────────────────────────────────────────────

#[test]
fn the_provider_set_registers_three_factories() {
    let set = provider_set();
    assert!(set.fs.is_some());
    assert!(set.shell.is_some());
    assert!(set.code.is_some());

    let endpoint = SandboxEndpoint {
        base_url: "http://127.0.0.1:1".to_string(),
        ..SandboxEndpoint::default()
    };
    let config = SandboxGatewayConfig::default();
    let fs = set.fs.clone().unwrap();
    let shell = set.shell.clone().unwrap();
    let code = set.code.clone().unwrap();
    assert!(fs(&endpoint, &config).is_ok());
    assert!(shell(&endpoint, &config).is_ok());
    assert!(code(&endpoint, &config).is_ok());
}

#[test]
fn a_missing_base_url_is_a_configuration_error() {
    let error = AioClient::new(&SandboxEndpoint::default(), &gateway_config())
        .err()
        .unwrap();
    assert_eq!(error.kind(), ErrorKind::Config);
    assert_eq!(error.message(), "aio provider requires endpoint.base_url");

    let set = provider_set();
    let factory = set.fs.clone().unwrap();
    let error = factory(&SandboxEndpoint::default(), &gateway_config())
        .err()
        .unwrap();
    assert_eq!(error.kind(), ErrorKind::Config);
    assert_eq!(error.message(), "aio provider requires endpoint.base_url");
}
