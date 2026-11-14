//! Crate-boundary integration suite for the three real remote sandbox providers.
//!
//! Every assertion goes through the public `openjiuwen_extensions::sys_operation`
//! facade plus the kernel traits behind it, so a regression in a registration
//! path, a factory signature, a wire shape, or an error mapping fails here
//! first. The in-module suites under `src/sys_operation/remote/` cover the same
//! providers at a lower level; this binary proves the crate boundary and the
//! registration surface a host actually calls.
//!
//! Python parity: `openjiuwen/extensions/sys_operation/sandbox/providers/`
//! (`aio.py`, `jiuwenbox.py`, `yuanrong.py`) and
//! `tests/unit_tests/extensions/sys_operation/sandbox/`, which only assert
//! registration; the transports are integration-gated there, so these tests pin
//! the wire contract against a loopback `wiremock` server instead.
//!
//! Everything is offline and deterministic: loopback `wiremock` only, a fresh
//! registry and a fresh `MockServer` per test, one isolation key per test so the
//! process-global container caches of `jiuwenbox`/`yuanrong` cannot couple two
//! tests, and every await bounded by [`BOUND`].
#![allow(clippy::unwrap_used, clippy::expect_used)]
use base64::Engine as _;
use futures::StreamExt;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_core::sys_operation::code::types::{CodeExecParams, Language};
use openjiuwen_core::sys_operation::fs::types::{
    FileContent, ListFilesParams, ReadChunkData, ReadFileData, ReadFileParams, ReadMode,
    SearchFilesParams, WriteFileData, WriteFileParams,
};
use openjiuwen_core::sys_operation::operation::{
    CodeOperation, FsOperation, OpStream, ShellOperation,
};
use openjiuwen_core::sys_operation::result::OpResult;
use openjiuwen_core::sys_operation::sandbox::config::SandboxLauncherConfig;
use openjiuwen_core::sys_operation::sandbox::provider::SandboxProviderRegistry;
use openjiuwen_core::sys_operation::shell::types::{
    ShellChunk, ShellExecParams, ShellOutput, ShellType, StreamKind,
};
use openjiuwen_extensions::sys_operation::{
    is_deferred, register_deferred, register_remote, register_remote_providers, registered_types,
    remote_provider_set, SandboxEndpoint, SandboxGatewayConfig, DEFERRED_SANDBOX_TYPES,
    REMOTE_SANDBOX_TYPES,
};
use serde_json::{json, Value};
use std::net::TcpListener;
use std::sync::Arc;
use std::time::Duration;
use wiremock::matchers::{body_partial_json, method, path, query_param};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};
/// Upper bound on every await in this suite; a provider that hangs fails here.
const BOUND: Duration = Duration::from_secs(10);
/// The bearer token the `jiuwenbox` tests configure on the card.
///
/// It is passed through `auth_headers` rather than `JIUWENBOX_API_TOKEN` so the
/// tests stay independent of the process environment (the transport prefers a
/// configured `Authorization` header over the environment token).
const JIUWENBOX_TOKEN: &str = "it-jiuwenbox-token";
// ─── fixtures ───────────────────────────────────────────────────────────────
/// A registry holding the three remote providers, never the process-global one.
fn registry() -> SandboxProviderRegistry {
    let registry = SandboxProviderRegistry::new();
    register_remote_providers(&registry).unwrap();
    registry
}
/// An endpoint pointing at the loopback server.
fn endpoint(server: &MockServer, sandbox_id: Option<&str>, isolation_key: &str) -> SandboxEndpoint {
    endpoint_at(&server.uri(), sandbox_id, isolation_key)
}
/// An endpoint for an arbitrary base URL (used by the unreachable-host test).
fn endpoint_at(base_url: &str, sandbox_id: Option<&str>, isolation_key: &str) -> SandboxEndpoint {
    SandboxEndpoint {
        base_url: base_url.to_string(),
        sandbox_id: sandbox_id.map(str::to_string),
        isolation_key: Some(isolation_key.to_string()),
    }
}
/// A launcher configuration carrying `extra_params`, with the server as base URL.
fn launcher(server: &MockServer, extra_params: Value) -> SandboxLauncherConfig {
    let extra_params = match extra_params {
        Value::Object(map) => map.into_iter().collect(),
        _ => std::collections::BTreeMap::default(),
    };
    SandboxLauncherConfig {
        base_url: Some(server.uri()),
        extra_params,
        ..SandboxLauncherConfig::default()
    }
}
/// A gateway configuration with a short timeout and `extra_params`.
fn gateway_config(server: &MockServer, extra_params: Value) -> SandboxGatewayConfig {
    SandboxGatewayConfig {
        timeout_seconds: 5,
        launcher_config: Some(launcher(server, extra_params)),
        ..SandboxGatewayConfig::default()
    }
}
/// The same configuration plus the card-level auth headers and query parameters.
fn with_auth(mut config: SandboxGatewayConfig) -> SandboxGatewayConfig {
    config
        .auth_headers
        .insert("x-api-key".to_string(), "secret".to_string());
    config
        .auth_query_params
        .insert("token".to_string(), "t0k".to_string());
    config
}
/// A `jiuwenbox` configuration: the card's bearer token plus `extra_params`.
fn jiuwenbox_config(server: &MockServer, extra_params: Value) -> SandboxGatewayConfig {
    let mut config = gateway_config(server, extra_params);
    config.auth_headers.insert(
        "Authorization".to_string(),
        format!("Bearer {JIUWENBOX_TOKEN}"),
    );
    config
}
/// Resolves the filesystem operation of `sandbox_type` from `registry`.
fn fs_operation(
    registry: &SandboxProviderRegistry,
    sandbox_type: &str,
    endpoint: &SandboxEndpoint,
    config: &SandboxGatewayConfig,
) -> Arc<dyn FsOperation> {
    registry.create_fs(sandbox_type, endpoint, config).unwrap()
}
/// Resolves the shell operation of `sandbox_type` from `registry`.
fn shell_operation(
    registry: &SandboxProviderRegistry,
    sandbox_type: &str,
    endpoint: &SandboxEndpoint,
    config: &SandboxGatewayConfig,
) -> Arc<dyn ShellOperation> {
    registry
        .create_shell(sandbox_type, endpoint, config)
        .unwrap()
}
/// Resolves the code operation of `sandbox_type` from `registry`.
fn code_operation(
    registry: &SandboxProviderRegistry,
    sandbox_type: &str,
    endpoint: &SandboxEndpoint,
    config: &SandboxGatewayConfig,
) -> Arc<dyn CodeOperation> {
    registry
        .create_code(sandbox_type, endpoint, config)
        .unwrap()
}
// ─── request inspection helpers ─────────────────────────────────────────────
/// Every request the mock server received, in arrival order.
async fn requests(server: &MockServer) -> Vec<Request> {
    server
        .received_requests()
        .await
        .expect("the mock server must be running to report its requests")
}
/// One request header as text.
fn header(request: &Request, name: &str) -> String {
    request
        .headers
        .get(name)
        .unwrap_or_else(|| panic!("missing request header '{name}'"))
        .to_str()
        .expect("the header must be ASCII")
        .to_string()
}
/// Every query pair of a request, decoded.
fn query(request: &Request) -> Vec<(String, String)> {
    request
        .url
        .query_pairs()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect()
}
/// Whether the request carries the query pair.
fn has_query(request: &Request, key: &str, value: &str) -> bool {
    query(request).iter().any(|(k, v)| k == key && v == value)
}
/// A request body decoded as JSON.
fn body_json(request: &Request) -> Value {
    request.body_json().unwrap_or_else(|error| {
        panic!(
            "request body of {} is not JSON: {error}",
            request.url.path()
        )
    })
}
/// Collects a buffered operation stream, failing when it does not finish.
async fn collect<T>(stream: OpStream<T>) -> Vec<OpResult<T>> {
    tokio::time::timeout(BOUND, stream.collect())
        .await
        .expect("the operation stream must finish")
}
/// Mounts one AIO envelope answer.
async fn mount_aio_envelope(server: &MockServer, body: Value) {
    Mock::given(method("POST"))
        .and(path("/api/v1/invoke"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(server)
        .await;
}
/// Mounts a successful AIO envelope carrying `data`.
async fn mount_aio_data(server: &MockServer, data: Value) {
    mount_aio_envelope(server, json!({"code": 0, "message": "OK", "data": data})).await;
}
/// Lowercase hex, the `yuanrong` `file.write`/`file.read` encoding.
fn hex(text: &str) -> String {
    use std::fmt::Write as _;
    text.bytes().fold(String::new(), |mut out, byte| {
        write!(out, "{byte:02x}").expect("writing into a String cannot fail");
        out
    })
}
/// Standard padded base64, the `yuanrong` management-plane `data` encoding.
fn base64_text(text: &str) -> String {
    base64::engine::general_purpose::STANDARD.encode(text.as_bytes())
}
/// Mounts the `yuanrong` environment creation, answering `data` base64-encoded.
async fn mount_yuanrong_create(server: &MockServer, environment_id: &str, runtime: &str) {
    let payload = json!({"environmentId": environment_id, "runtime": runtime}).to_string();
    Mock::given(method("POST"))
        .and(path("/api/environment/v1/environments"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({"data": base64_text(&payload)})),
        )
        .mount(server)
        .await;
}
/// Mounts one `yuanrong` `/direct/{id}/invoke` answer.
async fn mount_yuanrong_invoke(
    server: &MockServer,
    environment_id: &str,
    status: u16,
    body: Value,
) {
    Mock::given(method("POST"))
        .and(path(format!("/direct/{environment_id}/invoke")))
        .respond_with(ResponseTemplate::new(status).set_body_json(body))
        .mount(server)
        .await;
}
// ─── registration ───────────────────────────────────────────────────────────
// Python: `SandboxRegistry.provider("<type>", "fs" | "shell" | "code")` at import
// time of `extensions/sys_operation/sandbox/providers/__init__.py`.
#[test]
fn register_remote_providers_installs_every_operation_of_the_three_types() {
    let registry = SandboxProviderRegistry::new();
    register_remote_providers(&registry).unwrap();
    for sandbox_type in REMOTE_SANDBOX_TYPES {
        assert!(registry.has_fs(sandbox_type), "{sandbox_type} needs fs");
        assert!(
            registry.has_shell(sandbox_type),
            "{sandbox_type} needs shell"
        );
        assert!(registry.has_code(sandbox_type), "{sandbox_type} needs code");
    }
    assert_eq!(registered_types(&registry), REMOTE_SANDBOX_TYPES);
}
// Python: the provider package registers exactly `aio`, `jiuwenbox`, `yuanrong`.
#[test]
fn the_sandbox_type_constants_match_the_python_provider_package() {
    assert_eq!(REMOTE_SANDBOX_TYPES, ["aio", "jiuwenbox", "yuanrong"]);
    assert_eq!(DEFERRED_SANDBOX_TYPES, [] as [&str; 0]);
    for sandbox_type in REMOTE_SANDBOX_TYPES {
        assert!(
            !is_deferred(sandbox_type),
            "{sandbox_type} is a real provider"
        );
        assert!(remote_provider_set(sandbox_type).is_ok());
    }
    for deferred in ["local", "browser", ""] {
        assert!(is_deferred(deferred), "{deferred} has no remote provider");
    }
}
// Python: the deferred registration path, kept as a compatibility no-op.
#[tokio::test]
async fn register_deferred_registers_nothing_and_stays_called_afterwards() {
    let registry = SandboxProviderRegistry::new();
    register_deferred(&registry).await.unwrap();
    assert_eq!(registered_types(&registry), Vec::<String>::new());
    // Calling it after the real registration must not fail either.
    register_remote_providers(&registry).unwrap();
    register_deferred(&registry).await.unwrap();
    assert_eq!(registered_types(&registry), REMOTE_SANDBOX_TYPES);
}
// Python: an unknown provider name raises, naming the supported types.
#[test]
fn an_unknown_sandbox_type_is_unsupported_and_names_the_three_types() {
    let registry = SandboxProviderRegistry::new();
    let error = register_remote(&registry, "not-a-sandbox").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    for sandbox_type in REMOTE_SANDBOX_TYPES {
        assert!(
            error.message().contains(sandbox_type),
            "the message must name {sandbox_type}: {}",
            error.message()
        );
    }
    assert_eq!(registered_types(&registry), Vec::<String>::new());
    let error = remote_provider_set("nope").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error
        .message()
        .contains("supported types: aio, jiuwenbox, yuanrong"));
}
// Python: `SandboxRegistry.register_provider` raises on a duplicate key.
#[test]
fn a_duplicate_registration_is_a_sys_op_error() {
    let registry = registry();
    let error = register_remote(&registry, "aio").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::SysOp);
    assert!(error.message().contains("aio"));
}
// Python: `create_provider` of an unregistered type raises `NotImplementedError`.
#[test]
fn an_unregistered_sandbox_type_cannot_create_operations() {
    let registry = registry();
    let endpoint = SandboxEndpoint::default();
    let config = SandboxGatewayConfig::default();
    let error = registry
        .create_fs("unknown", &endpoint, &config)
        .err()
        .unwrap();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(registry
        .create_shell("unknown", &endpoint, &config)
        .is_err());
    assert!(registry.create_code("unknown", &endpoint, &config).is_err());
}
// ─── aio: kernel gateway envelope ───────────────────────────────────────────
// Python: `AIOFSProvider.read_file` (params dispatched as keyword arguments).
#[tokio::test]
async fn aio_read_file_posts_the_gateway_envelope_with_the_card_auth() {
    let server = MockServer::start().await;
    mount_aio_data(
        &server,
        json!({"path": "/workspace/a.txt", "content": "hello", "mode": "text"}),
    )
    .await;
    let endpoint = endpoint(&server, None, "session_aio_read");
    let config = with_auth(gateway_config(&server, json!({})));
    let fs = fs_operation(&registry(), "aio", &endpoint, &config);
    let result = tokio::time::timeout(
        BOUND,
        fs.read_file(ReadFileParams::text("/workspace/a.txt")),
    )
    .await
    .unwrap();
    assert_eq!(result.code, 0, "{}", result.message);
    let data: ReadFileData = result.data.unwrap();
    assert_eq!(data.path, "/workspace/a.txt");
    assert_eq!(data.content, FileContent::Text("hello".to_string()));
    assert_eq!(data.mode, ReadMode::Text);
    let requests = requests(&server).await;
    assert_eq!(requests.len(), 1);
    let request = &requests[0];
    assert_eq!(request.method.as_str(), "POST");
    assert_eq!(request.url.path(), "/api/v1/invoke");
    assert_eq!(header(request, "x-api-key"), "secret");
    assert!(has_query(request, "token", "t0k"));
    let body = body_json(request);
    assert_eq!(body["op_type"], "fs");
    assert_eq!(body["method"], "read_file");
    assert_eq!(body["params"]["path"], "/workspace/a.txt");
    assert_eq!(body["params"]["mode"], "text");
    assert_eq!(body["isolation_key"], "session_aio_read");
}
// Python: `AIOFSProvider.write_file` (newline flags and encoding are wire fields).
#[tokio::test]
async fn aio_write_file_sends_the_content_and_the_newline_flags() {
    let server = MockServer::start().await;
    mount_aio_data(
        &server,
        json!({"path": "/workspace/a.txt", "size": 5, "mode": "text"}),
    )
    .await;
    let endpoint = endpoint(&server, None, "session_aio_write");
    let config = gateway_config(&server, json!({}));
    let fs = fs_operation(&registry(), "aio", &endpoint, &config);
    let mut params = WriteFileParams::raw("/workspace/a.txt", "hello");
    params.append_newline = true;
    params.append = true;
    let result = tokio::time::timeout(BOUND, fs.write_file(params))
        .await
        .unwrap();
    assert_eq!(result.code, 0, "{}", result.message);
    let data: WriteFileData = result.data.unwrap();
    assert_eq!(data.path, "/workspace/a.txt");
    assert_eq!(data.size, 5);
    assert_eq!(data.mode, ReadMode::Text);
    let body = body_json(&requests(&server).await[0]);
    assert_eq!(body["op_type"], "fs");
    assert_eq!(body["method"], "write_file");
    assert_eq!(
        body["params"],
        json!({
            "path": "/workspace/a.txt",
            "content": "hello",
            "mode": "text",
            "prepend_newline": false,
            "append_newline": true,
            "append": true,
            "create_if_not_exist": true,
            "permissions": "644",
            "encoding": "utf-8",
        })
    );
}
// Python: `AIOFSProvider.list_files` (directories are dropped host-side).
#[tokio::test]
async fn aio_list_files_sends_the_listing_parameters() {
    let server = MockServer::start().await;
    mount_aio_data(
        &server,
        json!({"list_items": [
            {"name": "a.txt", "path": "/workspace/a.txt", "size": 3,
             "modified_time": "2024-01-01 00:00:00", "is_directory": false, "type": ".txt"},
            {"name": "sub", "path": "/workspace/sub", "is_directory": true},
        ]}),
    )
    .await;
    let endpoint = endpoint(&server, None, "session_aio_list");
    let config = gateway_config(&server, json!({}));
    let fs = fs_operation(&registry(), "aio", &endpoint, &config);
    let result = tokio::time::timeout(BOUND, fs.list_files(ListFilesParams::new("/workspace")))
        .await
        .unwrap();
    assert_eq!(result.code, 0, "{}", result.message);
    let data = result.data.unwrap();
    assert_eq!(data.total_count, 1);
    assert_eq!(data.list_items[0].path, "/workspace/a.txt");
    assert_eq!(data.list_items[0].file_type, Some(".txt".to_string()));
    assert_eq!(data.root_path, "/workspace");
    let body = body_json(&requests(&server).await[0]);
    assert_eq!(body["method"], "list_files");
    assert_eq!(
        body["params"],
        json!({
            "path": "/workspace",
            "recursive": false,
            "max_depth": null,
            "sort_by": "name",
            "sort_descending": false,
            "file_types": null,
        })
    );
}
// Python: `AIOFSProvider.search_files` (exclude patterns are a wire field).
#[tokio::test]
async fn aio_search_files_sends_the_pattern_and_exclusions() {
    let server = MockServer::start().await;
    mount_aio_data(
        &server,
        json!({"matching_files": [
            {"name": "a.txt", "path": "/workspace/a.txt", "size": 3, "is_directory": false},
        ]}),
    )
    .await;
    let endpoint = endpoint(&server, None, "session_aio_search");
    let config = gateway_config(&server, json!({}));
    let fs = fs_operation(&registry(), "aio", &endpoint, &config);
    let params = SearchFilesParams::new("/workspace", "*.txt").with_excludes(["target"]);
    let result = tokio::time::timeout(BOUND, fs.search_files(params))
        .await
        .unwrap();
    assert_eq!(result.code, 0, "{}", result.message);
    let data = result.data.unwrap();
    assert_eq!(data.total_matches, 1);
    assert_eq!(data.search_pattern, "*.txt");
    assert_eq!(data.matching_files[0].name, "a.txt");
    let body = body_json(&requests(&server).await[0]);
    assert_eq!(body["method"], "search_files");
    assert_eq!(
        body["params"],
        json!({"path": "/workspace", "pattern": "*.txt", "exclude_patterns": ["target"]})
    );
}
// Python: `AIOShellProvider.execute_cmd`.
#[tokio::test]
async fn aio_execute_cmd_sends_the_command_and_its_working_directory() {
    let server = MockServer::start().await;
    mount_aio_data(
        &server,
        json!({"command": "echo hi", "cwd": "/workspace", "stdout": "hi\n",
               "stderr": "", "exit_code": 0, "duration_ms": 3}),
    )
    .await;
    let endpoint = endpoint(&server, None, "session_aio_shell");
    let config = gateway_config(&server, json!({}));
    let shell = shell_operation(&registry(), "aio", &endpoint, &config);
    let params = ShellExecParams::new("echo hi")
        .with_cwd("/workspace")
        .with_timeout_secs(7)
        .with_shell_type(ShellType::Bash);
    let result = tokio::time::timeout(BOUND, shell.execute_cmd(params))
        .await
        .unwrap();
    assert_eq!(result.code, 0, "{}", result.message);
    let output: ShellOutput = result.data.unwrap();
    assert_eq!(output.stdout, "hi\n");
    assert_eq!(output.exit_code, 0);
    assert_eq!(output.cwd, "/workspace");
    let body = body_json(&requests(&server).await[0]);
    assert_eq!(body["op_type"], "shell");
    assert_eq!(body["method"], "execute_cmd");
    assert_eq!(body["params"]["command"], "echo hi");
    assert_eq!(body["params"]["cwd"], "/workspace");
    assert_eq!(body["params"]["timeout"], 7);
    assert_eq!(body["params"]["shell_type"], "bash");
}
// Python: `AIOShellProvider.execute_cmd` chunked on the host (no wire streaming).
#[tokio::test]
async fn aio_execute_cmd_stream_ends_with_a_terminal_chunk() {
    let server = MockServer::start().await;
    mount_aio_data(
        &server,
        json!({"stdout": "one\ntwo\n", "stderr": "", "exit_code": 0}),
    )
    .await;
    let endpoint = endpoint(&server, None, "session_aio_shell_stream");
    let config = gateway_config(&server, json!({}));
    let shell = shell_operation(&registry(), "aio", &endpoint, &config);
    let chunks = collect(shell.execute_cmd_stream(ShellExecParams::new("echo one"))).await;
    assert_eq!(chunks.len(), 3, "two stdout lines plus the terminal frame");
    assert_eq!(chunks[0].data.as_ref().unwrap().text, "one\n");
    assert_eq!(
        chunks[0].data.as_ref().unwrap().stream,
        Some(StreamKind::Stdout)
    );
    assert_eq!(chunks[1].data.as_ref().unwrap().text, "two\n");
    let terminal: &ShellChunk = chunks[2].data.as_ref().unwrap();
    // `ShellChunk` carries no `is_last_chunk`: the kernel's terminal frame is the one
    // with no stream and an exit code, and it is the last item the stream yields.
    assert_eq!(terminal.stream, None);
    assert_eq!(terminal.exit_code, Some(0));
    assert_eq!(terminal.text, "");
}
// Python: `AIOCodeProvider.execute_code`.
#[tokio::test]
async fn aio_execute_code_sends_the_language_and_the_snippet() {
    let server = MockServer::start().await;
    mount_aio_data(
        &server,
        json!({"exit_code": 0, "stdout": "1\n", "stderr": ""}),
    )
    .await;
    let endpoint = endpoint(&server, None, "session_aio_code");
    let config = gateway_config(&server, json!({}));
    let code = code_operation(&registry(), "aio", &endpoint, &config);
    let params = CodeExecParams::new("print(1)").with_timeout_secs(9);
    let result = tokio::time::timeout(BOUND, code.execute_code(params))
        .await
        .unwrap();
    assert_eq!(result.code, 0, "{}", result.message);
    let output = result.data.unwrap();
    assert_eq!(output.stdout, "1\n");
    assert_eq!(output.exit_code, 0);
    assert_eq!(output.language, "python");
    let body = body_json(&requests(&server).await[0]);
    assert_eq!(body["op_type"], "code");
    assert_eq!(body["method"], "execute_code");
    assert_eq!(body["params"]["code"], "print(1)");
    assert_eq!(body["params"]["language"], "python");
    assert_eq!(body["params"]["timeout"], 9);
    assert_eq!(body["params"]["options"]["force_file"], false);
}
// Python: a rejected envelope is a provider failure, not a panic.
#[tokio::test]
async fn aio_gateway_rejection_maps_to_the_sandbox_gateway_status() {
    let server = MockServer::start().await;
    mount_aio_envelope(
        &server,
        json!({"code": -1, "message": "boom", "data": null}),
    )
    .await;
    let endpoint = endpoint(&server, None, "session_aio_rejected");
    let config = gateway_config(&server, json!({}));
    let fs = fs_operation(&registry(), "aio", &endpoint, &config);
    let result = tokio::time::timeout(
        BOUND,
        fs.read_file(ReadFileParams::text("/workspace/a.txt")),
    )
    .await
    .unwrap();
    assert_eq!(result.code, 199_007);
    assert!(
        result
            .message
            .contains("sandbox gateway error, operation: read_file"),
        "unexpected message: {}",
        result.message
    );
    assert!(result.message.contains("aio gateway code -1: boom"));
}
// Python: a non-2xx gateway answer is a failure with the status and detail.
#[tokio::test]
async fn aio_http_error_maps_to_the_sandbox_gateway_status() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/invoke"))
        .respond_with(ResponseTemplate::new(500).set_body_json(json!({"message": "exploded"})))
        .mount(&server)
        .await;
    let endpoint = endpoint(&server, None, "session_aio_500");
    let config = gateway_config(&server, json!({}));
    let fs = fs_operation(&registry(), "aio", &endpoint, &config);
    let result = tokio::time::timeout(
        BOUND,
        fs.read_file(ReadFileParams::text("/workspace/a.txt")),
    )
    .await
    .unwrap();
    assert_eq!(result.code, 199_007);
    assert!(
        result.message.contains("HTTP 500 for read_file: exploded"),
        "unexpected message: {}",
        result.message
    );
}
// Python: an unreachable container is a provider failure, never a panic.
#[tokio::test]
async fn aio_an_unreachable_host_fails_with_the_gateway_status() {
    // Bind then drop a listener so the port is (almost certainly) closed.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    let endpoint = endpoint_at(
        &format!("http://{address}"),
        None,
        "session_aio_unreachable",
    );
    let config = SandboxGatewayConfig {
        timeout_seconds: 2,
        ..SandboxGatewayConfig::default()
    };
    let fs = fs_operation(&registry(), "aio", &endpoint, &config);
    let result = tokio::time::timeout(
        BOUND,
        fs.read_file(ReadFileParams::text("/workspace/a.txt")),
    )
    .await
    .unwrap();
    assert_eq!(result.code, 199_007);
    assert!(
        result.message.contains("request failed"),
        "unexpected message: {}",
        result.message
    );
}
// Python: `launcher_config.extra_params["invoke_path"]` overrides the gateway path.
#[tokio::test]
async fn aio_honours_the_invoke_path_override() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/custom/invoke"))
        .respond_with(ResponseTemplate::new(200).set_body_json(
            json!({"code": 0, "message": "OK", "data": {"content": "hello", "mode": "text"}}),
        ))
        .mount(&server)
        .await;
    let endpoint = endpoint(&server, None, "session_aio_invoke_path");
    let config = gateway_config(&server, json!({"invoke_path": "custom/invoke"}));
    let fs = fs_operation(&registry(), "aio", &endpoint, &config);
    let result = tokio::time::timeout(
        BOUND,
        fs.read_file(ReadFileParams::text("/workspace/a.txt")),
    )
    .await
    .unwrap();
    assert_eq!(result.code, 0, "{}", result.message);
    let requests = requests(&server).await;
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].url.path(), "/custom/invoke");
}
// Python: the gateway timeout surfaces as a provider failure.
#[tokio::test]
async fn aio_a_gateway_timeout_fails_with_the_gateway_status() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/invoke"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_secs(3))
                .set_body_json(json!({"code": 0, "message": "OK", "data": {}})),
        )
        .mount(&server)
        .await;
    let endpoint = endpoint(&server, None, "session_aio_timeout");
    let mut config = gateway_config(&server, json!({}));
    config.timeout_seconds = 1;
    let fs = fs_operation(&registry(), "aio", &endpoint, &config);
    let result = tokio::time::timeout(
        BOUND,
        fs.read_file(ReadFileParams::text("/workspace/a.txt")),
    )
    .await
    .unwrap();
    assert_eq!(result.code, 199_007);
    assert!(
        result.message.contains("request failed"),
        "unexpected message: {}",
        result.message
    );
}
// ─── jiuwenbox: native REST API ─────────────────────────────────────────────
// Python: `JiuwenBoxFSProvider.read_file` after `_get_sandbox_id` created the sandbox.
#[tokio::test]
async fn jiuwenbox_creates_a_sandbox_then_reads_through_it() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/sandboxes"))
        .and(body_partial_json(
            json!({"policy": {"mode": "read-only"}, "sandbox_runtime": "conch"}),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"id": "sbx-read"})))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/v1/sandboxes/sbx-read/download"))
        .and(query_param("sandbox_path", "/workspace/a.txt"))
        .respond_with(ResponseTemplate::new(200).set_body_string("hello"))
        .mount(&server)
        .await;
    let endpoint = endpoint(&server, None, "session_jiuwenbox_create");
    let config = jiuwenbox_config(
        &server,
        json!({"policy": {"mode": "read-only"}, "sandbox_runtime": "conch"}),
    );
    let fs = fs_operation(&registry(), "jiuwenbox", &endpoint, &config);
    let result = tokio::time::timeout(
        BOUND,
        fs.read_file(ReadFileParams::text("/workspace/a.txt")),
    )
    .await
    .unwrap();
    assert_eq!(result.code, 0, "{}", result.message);
    let data: ReadFileData = result.data.unwrap();
    assert_eq!(data.path, "/workspace/a.txt");
    assert_eq!(data.content, FileContent::Text("hello".to_string()));
    let requests = requests(&server).await;
    assert_eq!(requests.len(), 2, "one create then one download");
    assert_eq!(requests[0].method.as_str(), "POST");
    assert_eq!(requests[0].url.path(), "/api/v1/sandboxes");
    assert_eq!(
        header(&requests[0], "authorization"),
        format!("Bearer {JIUWENBOX_TOKEN}")
    );
    assert_eq!(
        body_json(&requests[0]),
        json!({"policy": {"mode": "read-only"}, "sandbox_runtime": "conch"})
    );
    assert_eq!(
        requests[1].url.path(),
        "/api/v1/sandboxes/sbx-read/download"
    );
    assert!(has_query(&requests[1], "sandbox_path", "/workspace/a.txt"));
}
// Python: `launcher_config.extra_params["sandbox_id"]` pins a pre-provisioned container.
#[tokio::test]
async fn jiuwenbox_a_pinned_sandbox_id_skips_creation() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/sandboxes/sbx-pinned/download"))
        .and(query_param("sandbox_path", "/workspace/a.txt"))
        .respond_with(ResponseTemplate::new(200).set_body_string("pinned"))
        .mount(&server)
        .await;
    let endpoint = endpoint(&server, None, "session_jiuwenbox_pinned");
    let config = jiuwenbox_config(&server, json!({"sandbox_id": "sbx-pinned"}));
    let fs = fs_operation(&registry(), "jiuwenbox", &endpoint, &config);
    let result = tokio::time::timeout(
        BOUND,
        fs.read_file(ReadFileParams::text("/workspace/a.txt")),
    )
    .await
    .unwrap();
    assert_eq!(result.code, 0, "{}", result.message);
    assert_eq!(
        result.data.unwrap().content,
        FileContent::Text("pinned".to_string())
    );
    let requests = requests(&server).await;
    assert_eq!(requests.len(), 1, "a pinned id must not create a sandbox");
    assert_eq!(
        requests[0].url.path(),
        "/api/v1/sandboxes/sbx-pinned/download"
    );
}
// Python: `JiuwenBoxFSProvider.write_file` uploads the bytes as multipart form data.
#[tokio::test]
async fn jiuwenbox_write_file_uploads_a_multipart_part() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/sandboxes/sbx-pinned/upload"))
        .and(query_param("sandbox_path", "/workspace/a.txt"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;
    let endpoint = endpoint(&server, None, "session_jiuwenbox_upload");
    let config = jiuwenbox_config(&server, json!({"sandbox_id": "sbx-pinned"}));
    let fs = fs_operation(&registry(), "jiuwenbox", &endpoint, &config);
    let params = WriteFileParams::raw("/workspace/a.txt", "hello");
    let result = tokio::time::timeout(BOUND, fs.write_file(params))
        .await
        .unwrap();
    assert_eq!(result.code, 0, "{}", result.message);
    let data = result.data.unwrap();
    assert_eq!(data.path, "/workspace/a.txt");
    assert_eq!(data.size, 5);
    assert_eq!(data.mode, ReadMode::Text);
    let requests = requests(&server).await;
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method.as_str(), "POST");
    assert_eq!(
        requests[0].url.path(),
        "/api/v1/sandboxes/sbx-pinned/upload"
    );
    assert!(
        header(&requests[0], "content-type").starts_with("multipart/form-data"),
        "the upload must be a multipart form"
    );
    let body = String::from_utf8_lossy(&requests[0].body);
    assert!(
        body.contains("name=\"file\""),
        "missing the file part: {body}"
    );
    assert!(
        body.contains("filename=\"a.txt\""),
        "missing the file name: {body}"
    );
    assert!(body.contains("hello"), "missing the payload: {body}");
}
// Python: `JiuwenBoxFSProvider.list_files`.
#[tokio::test]
async fn jiuwenbox_list_files_queries_the_listing_endpoint() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/sandboxes/sbx-pinned/files"))
        .and(query_param("sandbox_path", "/workspace"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"items": [
            {"name": "a.txt", "path": "/workspace/a.txt", "size": 3,
             "modified_time": "2024-01-01 00:00:00", "type": ".txt"},
        ]})))
        .mount(&server)
        .await;
    let endpoint = endpoint(&server, None, "session_jiuwenbox_list");
    let config = jiuwenbox_config(&server, json!({"sandbox_id": "sbx-pinned"}));
    let fs = fs_operation(&registry(), "jiuwenbox", &endpoint, &config);
    let result = tokio::time::timeout(BOUND, fs.list_files(ListFilesParams::new("/workspace")))
        .await
        .unwrap();
    assert_eq!(result.code, 0, "{}", result.message);
    let data = result.data.unwrap();
    assert_eq!(data.total_count, 1);
    assert_eq!(data.list_items[0].path, "/workspace/a.txt");
    assert_eq!(data.list_items[0].size, 3);
    assert_eq!(data.list_items[0].file_type, Some(".txt".to_string()));
    let requests = requests(&server).await;
    assert_eq!(requests.len(), 1);
    let request = &requests[0];
    assert!(has_query(request, "sandbox_path", "/workspace"));
    assert!(has_query(request, "recursive", "false"));
    assert!(has_query(request, "include_files", "true"));
    assert!(has_query(request, "include_dirs", "false"));
}
// Python: `JiuwenBoxFSProvider.search_files`.
#[tokio::test]
async fn jiuwenbox_search_files_queries_the_search_endpoint() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/sandboxes/sbx-pinned/search"))
        .and(query_param("pattern", "*.txt"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"items": [
            {"name": "b.txt", "path": "/workspace/b.txt", "size": 1, "type": ".txt"},
        ]})))
        .mount(&server)
        .await;
    let endpoint = endpoint(&server, None, "session_jiuwenbox_search");
    let config = jiuwenbox_config(&server, json!({"sandbox_id": "sbx-pinned"}));
    let fs = fs_operation(&registry(), "jiuwenbox", &endpoint, &config);
    let params = SearchFilesParams::new("/workspace", "*.txt");
    let result = tokio::time::timeout(BOUND, fs.search_files(params))
        .await
        .unwrap();
    assert_eq!(result.code, 0, "{}", result.message);
    let data = result.data.unwrap();
    assert_eq!(data.total_matches, 1);
    assert_eq!(data.matching_files[0].name, "b.txt");
    let requests = requests(&server).await;
    let request = &requests[0];
    assert_eq!(request.url.path(), "/api/v1/sandboxes/sbx-pinned/search");
    assert!(has_query(request, "sandbox_path", "/workspace"));
    assert!(has_query(request, "pattern", "*.txt"));
}
// Python: `JiuwenBoxShellProvider.execute_cmd` sends an argv array, not a command line.
#[tokio::test]
async fn jiuwenbox_execute_cmd_sends_an_argv_array_with_workdir_and_timeout() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/sandboxes/sbx-pinned/exec"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"exit_code": 0, "stdout": "hi\n", "stderr": ""})),
        )
        .mount(&server)
        .await;
    let endpoint = endpoint(&server, None, "session_jiuwenbox_shell");
    let config = jiuwenbox_config(&server, json!({"sandbox_id": "sbx-pinned"}));
    let shell = shell_operation(&registry(), "jiuwenbox", &endpoint, &config);
    let params = ShellExecParams::new("echo hi")
        .with_cwd("/workspace")
        .with_timeout_secs(7);
    let result = tokio::time::timeout(BOUND, shell.execute_cmd(params))
        .await
        .unwrap();
    assert_eq!(result.code, 0, "{}", result.message);
    let output: ShellOutput = result.data.unwrap();
    assert_eq!(output.stdout, "hi\n");
    assert_eq!(output.exit_code, 0);
    assert_eq!(output.cwd, "/workspace");
    assert_eq!(
        body_json(&requests(&server).await[0]),
        json!({
            "command": ["bash", "-lc", "echo hi"],
            "workdir": "/workspace",
            "timeout_seconds": 7,
        })
    );
}
// Python: `JiuwenBoxShellProvider.execute_cmd` chunked on the host.
#[tokio::test]
async fn jiuwenbox_execute_cmd_stream_ends_with_a_terminal_chunk() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/sandboxes/sbx-pinned/exec"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"exit_code": 0, "stdout": "one\ntwo\n", "stderr": ""})),
        )
        .mount(&server)
        .await;
    let endpoint = endpoint(&server, None, "session_jiuwenbox_shell_stream");
    let config = jiuwenbox_config(&server, json!({"sandbox_id": "sbx-pinned"}));
    let shell = shell_operation(&registry(), "jiuwenbox", &endpoint, &config);
    let chunks = collect(shell.execute_cmd_stream(ShellExecParams::new("echo one"))).await;
    assert_eq!(chunks.len(), 3, "two stdout lines plus the terminal frame");
    assert_eq!(chunks[1].data.as_ref().unwrap().text, "two\n");
    let terminal: &ShellChunk = chunks[2].data.as_ref().unwrap();
    assert_eq!(terminal.stream, None);
    assert_eq!(terminal.exit_code, Some(0));
}
// Python: `JiuwenBoxCodeProvider.execute_code` runs the inline interpreter form.
#[tokio::test]
async fn jiuwenbox_execute_code_runs_the_inline_interpreter() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/sandboxes/sbx-pinned/exec"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"exit_code": 0, "stdout": "1\n", "stderr": ""})),
        )
        .mount(&server)
        .await;
    let endpoint = endpoint(&server, None, "session_jiuwenbox_code");
    let config = jiuwenbox_config(&server, json!({"sandbox_id": "sbx-pinned"}));
    let code = code_operation(&registry(), "jiuwenbox", &endpoint, &config);
    let result = tokio::time::timeout(BOUND, code.execute_code(CodeExecParams::new("print(1)")))
        .await
        .unwrap();
    assert_eq!(result.code, 0, "{}", result.message);
    let output = result.data.unwrap();
    assert_eq!(output.stdout, "1\n");
    assert_eq!(output.exit_code, 0);
    assert_eq!(output.language, "python");
    let body = body_json(&requests(&server).await[0]);
    assert_eq!(body["command"], json!(["python3", "-u", "-c", "print(1)"]));
    assert_eq!(body["workdir"], "/tmp");
    assert_eq!(body["timeout_seconds"], 300);
    assert_eq!(body["env"]["PYTHONUTF8"], "1");
}
// ─── yuanrong: ADX environment API ──────────────────────────────────────────
// Python: `YuanRongFSProvider.read_file` after `_ensure_instance` created the environment.
#[tokio::test]
async fn yuanrong_creates_an_environment_with_the_launcher_knobs_then_reads() {
    let server = MockServer::start().await;
    mount_yuanrong_create(&server, "env-1", "docker").await;
    mount_yuanrong_invoke(&server, "env-1", 200, json!({"data": hex("hello")})).await;
    let endpoint = endpoint(&server, None, "session_yuanrong_create");
    let extra = json!({
        "cpu": 2,
        "memory": 512,
        "idleTimeoutSeconds": 300,
        "rootfs": "image:latest",
        "executor": "docker",
    });
    let config = gateway_config(&server, extra);
    let fs = fs_operation(&registry(), "yuanrong", &endpoint, &config);
    let result = tokio::time::timeout(
        BOUND,
        fs.read_file(ReadFileParams::text("/workspace/a.txt")),
    )
    .await
    .unwrap();
    assert_eq!(result.code, 0, "{}", result.message);
    let data: ReadFileData = result.data.unwrap();
    assert_eq!(data.path, "/workspace/a.txt");
    assert_eq!(data.content, FileContent::Text("hello".to_string()));
    let requests = requests(&server).await;
    assert_eq!(requests.len(), 2, "one create then one invoke");
    assert_eq!(requests[0].url.path(), "/api/environment/v1/environments");
    assert_eq!(header(&requests[0], "accept"), "application/json");
    assert_eq!(
        body_json(&requests[0]),
        json!({
            "rootfs": "image:latest",
            "cpu": 2,
            "memory": 512,
            "idleTimeoutSeconds": 300,
            "runtime": "docker",
        })
    );
    assert_eq!(requests[1].url.path(), "/direct/env-1/invoke");
    assert_eq!(
        body_json(&requests[1]),
        json!({"action": "file.read", "args": {"path": "/workspace/a.txt", "binary": true}})
    );
}
// Python: `YuanRongFSProvider.read_file` accepts a base64 `content_encoding`.
#[tokio::test]
async fn yuanrong_read_file_decodes_a_base64_payload() {
    let server = MockServer::start().await;
    mount_yuanrong_invoke(
        &server,
        "env-b64",
        200,
        json!({"data": base64_text("hello"), "content_encoding": "base64"}),
    )
    .await;
    // The endpoint's sandbox id pins the environment, so no create is issued.
    let endpoint = endpoint(&server, Some("env-b64"), "session_yuanrong_b64");
    let config = gateway_config(&server, json!({}));
    let fs = fs_operation(&registry(), "yuanrong", &endpoint, &config);
    let result = tokio::time::timeout(
        BOUND,
        fs.read_file(ReadFileParams::text("/workspace/a.txt")),
    )
    .await
    .unwrap();
    assert_eq!(result.code, 0, "{}", result.message);
    assert_eq!(
        result.data.unwrap().content,
        FileContent::Text("hello".to_string())
    );
    let requests = requests(&server).await;
    assert_eq!(
        requests.len(),
        1,
        "a pinned environment must not be created"
    );
}
// Python: `YuanRongFSProvider.write_file` (this build always sends binary hex).
#[tokio::test]
async fn yuanrong_write_file_sends_hexadecimal_binary_data() {
    let server = MockServer::start().await;
    mount_yuanrong_invoke(&server, "env-w", 200, json!({})).await;
    let endpoint = endpoint(&server, Some("env-w"), "session_yuanrong_write");
    let config = gateway_config(&server, json!({}));
    let fs = fs_operation(&registry(), "yuanrong", &endpoint, &config);
    let params = WriteFileParams::raw("/workspace/a.txt", "hello");
    let result = tokio::time::timeout(BOUND, fs.write_file(params))
        .await
        .unwrap();
    assert_eq!(result.code, 0, "{}", result.message);
    let data = result.data.unwrap();
    assert_eq!(data.path, "/workspace/a.txt");
    assert_eq!(data.size, 5);
    assert_eq!(
        body_json(&requests(&server).await[0]),
        json!({
            "action": "file.write",
            "args": {
                "path": "/workspace/a.txt",
                "data": hex("hello"),
                "binary": true,
                "append": false,
            },
        })
    );
}
// Python: `YuanRongFSProvider.list_files` (depth `1` when not recursive).
#[tokio::test]
async fn yuanrong_list_files_sends_the_depth_and_drops_directories() {
    let server = MockServer::start().await;
    mount_yuanrong_invoke(
        &server,
        "env-list",
        200,
        json!({"entries": [
            {"name": "a.txt", "path": "/workspace/a.txt", "size": 3, "type": ".txt"},
            {"name": "sub", "path": "/workspace/sub", "type": "dir"},
        ]}),
    )
    .await;
    let endpoint = endpoint(&server, Some("env-list"), "session_yuanrong_list");
    let config = gateway_config(&server, json!({}));
    let fs = fs_operation(&registry(), "yuanrong", &endpoint, &config);
    let result = tokio::time::timeout(BOUND, fs.list_files(ListFilesParams::new("/workspace")))
        .await
        .unwrap();
    assert_eq!(result.code, 0, "{}", result.message);
    let data = result.data.unwrap();
    assert_eq!(data.total_count, 1, "directories are dropped");
    assert_eq!(data.list_items[0].name, "a.txt");
    assert_eq!(
        body_json(&requests(&server).await[0]),
        json!({"action": "file.list", "args": {"path": "/workspace", "depth": 1}})
    );
}
// Python: `YuanRongFSProvider.search_files`.
#[tokio::test]
async fn yuanrong_search_files_sends_the_pattern() {
    let server = MockServer::start().await;
    mount_yuanrong_invoke(
        &server,
        "env-search",
        200,
        json!({"entries": [
            {"name": "b.txt", "path": "/workspace/b.txt", "size": 1, "type": ".txt"},
        ]}),
    )
    .await;
    let endpoint = endpoint(&server, Some("env-search"), "session_yuanrong_search");
    let config = gateway_config(&server, json!({}));
    let fs = fs_operation(&registry(), "yuanrong", &endpoint, &config);
    let params = SearchFilesParams::new("/workspace", "*.txt");
    let result = tokio::time::timeout(BOUND, fs.search_files(params))
        .await
        .unwrap();
    assert_eq!(result.code, 0, "{}", result.message);
    assert_eq!(result.data.unwrap().total_matches, 1);
    assert_eq!(
        body_json(&requests(&server).await[0]),
        json!({"action": "file.search", "args": {"path": "/workspace", "pattern": "*.txt"}})
    );
}
// Python: `YuanRongShellProvider.execute_cmd` sends `process.exec` with `cmd`.
#[tokio::test]
async fn yuanrong_execute_cmd_sends_process_exec_with_cmd() {
    let server = MockServer::start().await;
    mount_yuanrong_invoke(
        &server,
        "env-exec",
        200,
        json!({"exit_code": 0, "stdout": "hi\n", "stderr": ""}),
    )
    .await;
    let endpoint = endpoint(&server, Some("env-exec"), "session_yuanrong_shell");
    let config = gateway_config(&server, json!({}));
    let shell = shell_operation(&registry(), "yuanrong", &endpoint, &config);
    let params = ShellExecParams::new("echo hi").with_cwd("/workspace");
    let result = tokio::time::timeout(BOUND, shell.execute_cmd(params))
        .await
        .unwrap();
    assert_eq!(result.code, 0, "{}", result.message);
    let output: ShellOutput = result.data.unwrap();
    assert_eq!(output.stdout, "hi\n");
    assert_eq!(output.exit_code, 0);
    assert_eq!(output.cwd, "/workspace");
    assert_eq!(
        body_json(&requests(&server).await[0]),
        json!({
            "action": "process.exec",
            "args": {"cmd": "echo hi", "cwd": "/workspace", "timeout": 300},
        })
    );
}
// Python: an ADX `file.search` that hit the server result limit is a failure.
#[tokio::test]
async fn yuanrong_a_search_limit_failure_maps_to_the_gateway_status() {
    let server = MockServer::start().await;
    mount_yuanrong_invoke(
        &server,
        "env-limit",
        429,
        json!({"error": "file search max_results exceeded"}),
    )
    .await;
    let endpoint = endpoint(&server, Some("env-limit"), "session_yuanrong_limit");
    let config = gateway_config(&server, json!({}));
    let fs = fs_operation(&registry(), "yuanrong", &endpoint, &config);
    let params = SearchFilesParams::new("/workspace", "*");
    let result = tokio::time::timeout(BOUND, fs.search_files(params))
        .await
        .unwrap();
    assert_eq!(result.code, 199_007);
    assert!(
        result.message.contains("file search max_results exceeded"),
        "unexpected message: {}",
        result.message
    );
}
// Python: a lost environment is recreated and the action retried.
#[tokio::test]
async fn yuanrong_a_gone_environment_is_recreated_then_succeeds() {
    let server = MockServer::start().await;
    let old = json!({"environmentId": "env-old", "runtime": "supervisor"}).to_string();
    Mock::given(method("POST"))
        .and(path("/api/environment/v1/environments"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data": base64_text(&old)})))
        .with_priority(1)
        .up_to_n_times(1)
        .mount(&server)
        .await;
    mount_yuanrong_create(&server, "env-new", "supervisor").await;
    mount_yuanrong_invoke(
        &server,
        "env-old",
        404,
        json!({"message": "environment not found"}),
    )
    .await;
    mount_yuanrong_invoke(&server, "env-new", 200, json!({"data": hex("hello")})).await;
    let endpoint = endpoint(&server, None, "session_yuanrong_recreate");
    let config = gateway_config(&server, json!({}));
    let fs = fs_operation(&registry(), "yuanrong", &endpoint, &config);
    let result = tokio::time::timeout(
        BOUND,
        fs.read_file(ReadFileParams::text("/workspace/a.txt")),
    )
    .await
    .unwrap();
    assert_eq!(result.code, 0, "{}", result.message);
    assert_eq!(
        result.data.unwrap().content,
        FileContent::Text("hello".to_string())
    );
    let requests = requests(&server).await;
    let creates: Vec<&Request> = requests
        .iter()
        .filter(|request| request.url.path() == "/api/environment/v1/environments")
        .collect();
    assert_eq!(creates.len(), 2, "the stale environment must be recreated");
    let invokes: Vec<&Request> = requests
        .iter()
        .filter(|request| request.url.path().ends_with("/invoke"))
        .collect();
    assert_eq!(invokes.len(), 2, "the lost invoke must be retried");
    assert_eq!(invokes[0].url.path(), "/direct/env-old/invoke");
    assert_eq!(invokes[1].url.path(), "/direct/env-new/invoke");
}
// Python: a non-200 ADX answer is a provider failure, not a panic.
#[tokio::test]
async fn yuanrong_a_server_error_maps_to_the_gateway_status() {
    let server = MockServer::start().await;
    mount_yuanrong_invoke(
        &server,
        "env-boom",
        500,
        json!({"message": "environment exploded"}),
    )
    .await;
    let endpoint = endpoint(&server, Some("env-boom"), "session_yuanrong_500");
    let config = gateway_config(&server, json!({}));
    let fs = fs_operation(&registry(), "yuanrong", &endpoint, &config);
    let result = tokio::time::timeout(
        BOUND,
        fs.read_file(ReadFileParams::text("/workspace/a.txt")),
    )
    .await
    .unwrap();
    assert_eq!(result.code, 199_007);
    assert!(
        result
            .message
            .contains("HTTP 500 for file.read: environment exploded"),
        "unexpected message: {}",
        result.message
    );
}
// ─── cross-provider end-to-end ──────────────────────────────────────────────
/// Mounts the provider-specific answer of one successful `read_file`.
async fn mount_read(
    server: &MockServer,
    sandbox_type: &str,
    remote_path: &str,
    content: &str,
) -> (SandboxEndpoint, SandboxGatewayConfig) {
    let isolation = format!("cross_read_{sandbox_type}");
    match sandbox_type {
        "aio" => {
            mount_aio_data(
                server,
                json!({"path": remote_path, "content": content, "mode": "text"}),
            )
            .await;
            (
                endpoint(server, None, &isolation),
                gateway_config(server, json!({})),
            )
        }
        "jiuwenbox" => {
            Mock::given(method("GET"))
                .and(path("/api/v1/sandboxes/sbx-cross/download"))
                .and(query_param("sandbox_path", remote_path))
                .respond_with(ResponseTemplate::new(200).set_body_string(content))
                .mount(server)
                .await;
            (
                endpoint(server, None, &isolation),
                jiuwenbox_config(server, json!({"sandbox_id": "sbx-cross"})),
            )
        }
        "yuanrong" => {
            mount_yuanrong_invoke(server, "env-cross", 200, json!({"data": hex(content)})).await;
            (
                endpoint(server, Some("env-cross"), &isolation),
                gateway_config(server, json!({})),
            )
        }
        other => panic!("unknown sandbox type '{other}'"),
    }
}
/// Mounts the provider-specific answer of one successful `execute_cmd`.
async fn mount_shell(
    server: &MockServer,
    sandbox_type: &str,
    stdout: &str,
) -> (SandboxEndpoint, SandboxGatewayConfig) {
    let isolation = format!("cross_shell_{sandbox_type}");
    match sandbox_type {
        "aio" => {
            mount_aio_data(
                server,
                json!({"stdout": stdout, "stderr": "", "exit_code": 0}),
            )
            .await;
            (
                endpoint(server, None, &isolation),
                gateway_config(server, json!({})),
            )
        }
        "jiuwenbox" => {
            Mock::given(method("POST"))
                .and(path("/api/v1/sandboxes/sbx-cross/exec"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_json(json!({"exit_code": 0, "stdout": stdout, "stderr": ""})),
                )
                .mount(server)
                .await;
            (
                endpoint(server, None, &isolation),
                jiuwenbox_config(server, json!({"sandbox_id": "sbx-cross"})),
            )
        }
        "yuanrong" => {
            mount_yuanrong_invoke(
                server,
                "env-cross",
                200,
                json!({"exit_code": 0, "stdout": stdout, "stderr": ""}),
            )
            .await;
            (
                endpoint(server, Some("env-cross"), &isolation),
                gateway_config(server, json!({})),
            )
        }
        other => panic!("unknown sandbox type '{other}'"),
    }
}
// Python: every provider implements `BaseFsOperation.read_file` and
// `BaseShellOperation.execute_cmd` with the same result shape.
#[tokio::test]
async fn every_provider_maps_a_read_and_a_shell_run_into_op_result_data() {
    for sandbox_type in REMOTE_SANDBOX_TYPES {
        // One server per operation: the AIO envelope answers `fs`, `shell`, and `code`
        // calls on the same path, so a second mock on it would shadow the first.
        let read_server = MockServer::start().await;
        let shell_server = MockServer::start().await;
        let sandbox_type: &str = sandbox_type;
        let registry = registry();
        let (endpoint, config) =
            mount_read(&read_server, sandbox_type, "/workspace/a.txt", "hello").await;
        let fs = fs_operation(&registry, sandbox_type, &endpoint, &config);
        let result = tokio::time::timeout(
            BOUND,
            fs.read_file(ReadFileParams::text("/workspace/a.txt")),
        )
        .await
        .unwrap();
        assert_eq!(result.code, 0, "{sandbox_type}: {}", result.message);
        let data: ReadFileData = result.data.unwrap();
        assert_eq!(data.path, "/workspace/a.txt", "{sandbox_type}");
        assert_eq!(
            data.content,
            FileContent::Text("hello".to_string()),
            "{sandbox_type}"
        );
        assert_eq!(data.mode, ReadMode::Text, "{sandbox_type}");

        let (endpoint, config) = mount_shell(&shell_server, sandbox_type, "hi\n").await;
        let shell = shell_operation(&registry, sandbox_type, &endpoint, &config);
        let params = ShellExecParams::new("echo hi").with_cwd("/workspace");
        let result = tokio::time::timeout(BOUND, shell.execute_cmd(params))
            .await
            .unwrap();
        assert_eq!(result.code, 0, "{sandbox_type}: {}", result.message);
        let output: ShellOutput = result.data.unwrap();
        assert_eq!(output.stdout, "hi\n", "{sandbox_type}");
        assert_eq!(output.exit_code, 0, "{sandbox_type}");
        assert_eq!(output.cwd, "/workspace", "{sandbox_type}");
    }
}
// Python: `read_file_stream` yields one chunk per line and a final chunk flagged
// `is_last_chunk`; nothing follows it.
#[tokio::test]
async fn every_provider_streams_a_read_ending_in_a_terminal_chunk() {
    for sandbox_type in REMOTE_SANDBOX_TYPES {
        let server = MockServer::start().await;
        let sandbox_type: &str = sandbox_type;
        let registry = registry();
        let (endpoint, config) =
            mount_read(&server, sandbox_type, "/workspace/a.txt", "one\ntwo\n").await;
        let fs = fs_operation(&registry, sandbox_type, &endpoint, &config);
        let chunks = collect(fs.read_file_stream(ReadFileParams::text("/workspace/a.txt"))).await;
        assert_eq!(chunks.len(), 2, "{sandbox_type}: one chunk per line");
        let last = chunks.len() - 1;
        for (index, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.code, 0, "{sandbox_type}: {}", chunk.message);
            let data: &ReadChunkData = chunk.data.as_ref().unwrap();
            assert_eq!(data.path, "/workspace/a.txt", "{sandbox_type}");
            assert_eq!(data.chunk_index, u64::try_from(index).unwrap());
            assert_eq!(
                data.is_last_chunk,
                index == last,
                "{sandbox_type}: only the final chunk is terminal"
            );
        }
        assert_eq!(
            chunks[last].data.as_ref().unwrap().chunk,
            FileContent::Text("two\n".to_string()),
            "{sandbox_type}"
        );
    }
}
// Python: `execute_cmd_stream` yields the output lines then one terminal frame.
#[tokio::test]
async fn every_provider_streams_shell_output_ending_in_a_terminal_chunk() {
    for sandbox_type in REMOTE_SANDBOX_TYPES {
        let server = MockServer::start().await;
        let sandbox_type: &str = sandbox_type;
        let registry = registry();
        let (endpoint, config) = mount_shell(&server, sandbox_type, "one\ntwo\n").await;
        let shell = shell_operation(&registry, sandbox_type, &endpoint, &config);
        let chunks = collect(shell.execute_cmd_stream(ShellExecParams::new("echo one"))).await;
        assert_eq!(
            chunks.len(),
            3,
            "{sandbox_type}: two stdout lines plus the terminal frame"
        );
        assert_eq!(
            chunks[0].data.as_ref().unwrap().text,
            "one\n",
            "{sandbox_type}"
        );
        assert_eq!(
            chunks[1].data.as_ref().unwrap().text,
            "two\n",
            "{sandbox_type}"
        );
        // `ShellChunk` has no `is_last_chunk` field: the terminal frame is the one with
        // no stream and an exit code, and the stream yields nothing after it.
        let terminal: &ShellChunk = chunks[2].data.as_ref().unwrap();
        assert_eq!(terminal.stream, None, "{sandbox_type}");
        assert_eq!(terminal.exit_code, Some(0), "{sandbox_type}");
        assert_eq!(terminal.text, "", "{sandbox_type}");
    }
}
// `Language` is part of the public code surface this suite drives.
#[test]
fn the_code_surface_accepts_both_languages() {
    assert_eq!(Language::Python.as_str(), "python");
    assert_eq!(Language::Javascript.as_str(), "javascript");
    let _ = Language::parse("python");
}
// ─── module suites ──────────────────────────────────────────────────────────
// The teardown and conditional-write cases live in sibling files so this one stays
// within its size budget; both reuse the fixtures above through `use super::*`. A test
// target's crate root is a file, so the paths are explicit (as `tests/redis.rs` does).
#[path = "sandbox_providers/conditional_write.rs"]
mod conditional_write;
#[path = "sandbox_providers/teardown.rs"]
mod teardown;
