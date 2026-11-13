//! Unit tests of the `aio` remote sandbox provider.
//!
//! Python counterpart: `tests/unit_tests/extensions/sys_operation/sandbox/test_aio_*.py`,
//! which only asserts registration; the behavior there is integration-gated. These
//! tests therefore pin the wire contract instead — the kernel gateway envelope, the
//! `params` names of the Python provider signatures, and the mapping of `data` back
//! into the kernel result structs — against a loopback `wiremock` server.
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

// ─── fs parameters ──────────────────────────────────────────────────────────

#[tokio::test]
async fn read_file_sends_the_python_signature_parameters() {
    let server = MockServer::start().await;
    respond_data(&server, json!({"content": "hi"})).await;
    let mut params = ReadFileParams::text("/work/a.txt");
    params.line_range = Some(LineRange::new(2, 4));

    let result = fs(&server).read_file(params).await;

    assert_eq!(result.code, 0);
    assert_eq!(
        last_envelope(&server).await["params"],
        json!({
            "path": "/work/a.txt",
            "mode": "text",
            "head": null,
            "tail": null,
            "line_range": [2, 4],
            "encoding": "utf-8",
            "chunk_size": 0,
        })
    );
}

#[tokio::test]
async fn read_file_validates_the_line_selection_before_calling_the_gateway() {
    let server = MockServer::start().await;

    let result = fs(&server)
        .read_file(ReadFileParams::bytes("/work/a.bin").with_head(2))
        .await;

    assert_eq!(result.code, 199_003);
    assert!(result.message.contains("only supported in text mode"));
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn write_file_sends_base64_bytes_and_the_newline_flags() {
    let server = MockServer::start().await;
    respond_data(
        &server,
        json!({"path": "/work/a.bin", "size": 3, "mode": "bytes"}),
    )
    .await;

    let result = fs(&server)
        .write_file(WriteFileParams::new(
            "/work/a.bin",
            FileContent::Bytes(vec![1, 2, 250]),
        ))
        .await;

    assert_eq!(result.code, 0);
    let data: WriteFileData = result.data.unwrap();
    assert_eq!(data.path, "/work/a.bin");
    assert_eq!(data.size, 3);
    assert_eq!(data.mode, ReadMode::Bytes);
    assert_eq!(
        last_envelope(&server).await["params"],
        json!({
            "path": "/work/a.bin",
            "content": "AQL6",
            "mode": "bytes",
            "prepend_newline": true,
            "append_newline": false,
            "append": false,
            "create_if_not_exist": true,
            "permissions": "644",
            "encoding": "base64",
        })
    );
}

#[tokio::test]
async fn write_file_sends_text_content_and_the_request_encoding() {
    let server = MockServer::start().await;
    respond_data(&server, json!({})).await;
    let mut params = WriteFileParams::raw("/work/a.txt", "hello");
    params.append_newline = true;
    params.append = true;
    params.create_if_not_exist = false;

    let result = fs(&server).write_file(params).await;

    assert_eq!(result.code, 0);
    let envelope = last_envelope(&server).await;
    assert_eq!(envelope["params"]["content"], "hello");
    assert_eq!(envelope["params"]["encoding"], "utf-8");
    assert_eq!(envelope["params"]["append_newline"], true);
    assert_eq!(envelope["params"]["append"], true);
    assert_eq!(envelope["params"]["create_if_not_exist"], false);
    assert_eq!(envelope["params"]["prepend_newline"], false);
}

#[tokio::test]
async fn conditional_writes_are_rejected_before_calling_the_gateway() {
    let server = MockServer::start().await;
    let fs = fs(&server);

    let digest = fs
        .write_file(WriteFileParams::raw("/work/a.txt", "hi").with_expected_sha256("ab"))
        .await;
    assert_eq!(digest.code, 199_003);
    assert_eq!(
        digest.message,
        "file system operation execution error, execution: write_file, \
         reason: expected_content_sha256 is not supported by the aio provider \
         (remote sandbox transports have no compare-and-swap)"
    );

    let mut absent = WriteFileParams::raw("/work/a.txt", "hi");
    absent.create_if_absent = true;
    let absent = fs.write_file(absent).await;
    assert_eq!(absent.code, 199_003);
    assert!(absent
        .message
        .contains("create_if_absent is not supported by the aio provider"));
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn listing_sends_the_sort_fields_and_the_type_filter() {
    let server = MockServer::start().await;
    respond_data(&server, json!({"total_count": 0, "list_items": []})).await;
    let mut params = ListFilesParams::new("/work").recursive(Some(2));
    params.sort_by = SortBy::ModifiedTime;
    params.sort_descending = true;
    params.file_types = Some(vec![".txt".to_string()]);

    let result = fs(&server).list_files(params).await;

    assert_eq!(result.code, 0);
    assert_eq!(
        last_envelope(&server).await["params"],
        json!({
            "path": "/work",
            "recursive": true,
            "max_depth": 2,
            "sort_by": "modified_time",
            "sort_descending": true,
            "file_types": [".txt"],
        })
    );
}

#[tokio::test]
async fn listing_directories_omits_the_type_filter() {
    let server = MockServer::start().await;
    respond_data(&server, json!({"list_items": []})).await;

    let result = fs(&server)
        .list_directories(ListDirsParams::new("/work"))
        .await;

    assert_eq!(result.code, 0);
    let envelope = last_envelope(&server).await;
    assert_eq!(envelope["method"], "list_directories");
    assert_eq!(
        envelope["params"],
        json!({
            "path": "/work",
            "recursive": false,
            "max_depth": null,
            "sort_by": "name",
            "sort_descending": false,
        })
    );
    assert!(envelope["params"].get("file_types").is_none());
}

#[tokio::test]
async fn searching_sends_the_pattern_and_the_exclusions() {
    let server = MockServer::start().await;
    respond_data(&server, json!({"matching_files": []})).await;

    let result = fs(&server)
        .search_files(SearchFilesParams::new("/work", "*.rs").with_excludes(["target/*"]))
        .await;

    assert_eq!(result.code, 0);
    assert_eq!(
        last_envelope(&server).await["params"],
        json!({
            "path": "/work",
            "pattern": "*.rs",
            "exclude_patterns": ["target/*"],
        })
    );
}

// ─── fs data mapping ────────────────────────────────────────────────────────

#[tokio::test]
async fn read_file_maps_text_content() {
    let server = MockServer::start().await;
    respond_data(
        &server,
        json!({"path": "/work/a.txt", "content": "one\ntwo\n", "mode": "text"}),
    )
    .await;

    let result = fs(&server)
        .read_file(ReadFileParams::text("/work/a.txt"))
        .await;

    assert_eq!(result.code, 0);
    let data: ReadFileData = result.data.unwrap();
    assert_eq!(data.path, "/work/a.txt");
    assert_eq!(data.content, FileContent::Text("one\ntwo\n".to_string()));
    assert_eq!(data.mode, ReadMode::Text);
    assert_eq!(data.encoding, None);
    assert!(!data.truncated);
    assert_eq!(data.total_bytes, 8);
}

#[tokio::test]
async fn read_file_maps_a_byte_array_and_falls_back_to_the_request_mode() {
    let server = MockServer::start().await;
    respond_data(&server, json!({"content": [1, 2, 255]})).await;

    let result = fs(&server)
        .read_file(ReadFileParams::bytes("/work/a.bin"))
        .await;

    assert_eq!(result.code, 0);
    let data: ReadFileData = result.data.unwrap();
    assert_eq!(data.content, FileContent::Bytes(vec![1, 2, 255]));
    assert_eq!(data.mode, ReadMode::Bytes);
    assert_eq!(data.path, "/work/a.bin", "the request path is the fallback");
    assert_eq!(data.total_bytes, 3);
}

#[tokio::test]
async fn a_response_without_data_is_tolerated() {
    let server = MockServer::start().await;
    respond(&server, json!({"code": 0})).await;

    let result = fs(&server)
        .read_file(ReadFileParams::text("/work/a.txt"))
        .await;

    assert_eq!(result.code, 0, "a missing code defaults to success");
    let data: ReadFileData = result.data.unwrap();
    assert_eq!(data.content, FileContent::Text(String::new()));
    assert_eq!(data.total_bytes, 0);
}

#[tokio::test]
async fn write_file_falls_back_to_the_payload_size_and_the_request_mode() {
    let server = MockServer::start().await;
    respond_data(&server, json!({})).await;

    let result = fs(&server)
        .write_file(WriteFileParams::raw("/work/a.txt", "abc"))
        .await;

    assert_eq!(result.code, 0);
    let data: WriteFileData = result.data.unwrap();
    assert_eq!(data.path, "/work/a.txt");
    assert_eq!(data.size, 3, "the text payload length");
    assert_eq!(data.mode, ReadMode::Text);
}

#[tokio::test]
async fn list_files_maps_entries_filters_directories_and_sorts() {
    let server = MockServer::start().await;
    respond_data(
        &server,
        json!({
            "total_count": 4,
            "root_path": "/work",
            "recursive": false,
            "max_depth": null,
            "list_items": [
                {"name": "b.txt", "path": "/work/b.txt", "size": 2,
                 "modified_time": "2024-01-02 00:00:00"},
                {"name": "a.rs", "path": "/work/a.rs", "size": 1},
                {"name": "dir", "path": "/work/dir", "is_directory": true},
                {"name": "c.md", "path": "/work/c.md", "size": 3, "type": ".md",
                 "modified_time": "2024-01-01 00:00:00"},
            ],
        }),
    )
    .await;

    let result = fs(&server).list_files(ListFilesParams::new("/work")).await;

    assert_eq!(result.code, 0);
    let data: ListFilesData = result.data.unwrap();
    assert_eq!(data.total_count, 3, "the directory is dropped");
    assert_eq!(data.root_path, "/work");
    assert!(!data.recursive);
    assert_eq!(data.max_depth, None);
    assert_eq!(names(&data.list_items), vec!["a.rs", "b.txt", "c.md"]);
    assert_eq!(
        data.list_items[0].file_type,
        Some(".rs".to_string()),
        "the extension falls back from the name"
    );
    assert_eq!(
        data.list_items[0].modified_time, "0",
        "a missing modification time falls back to '0'"
    );
    assert!(!data.list_items[0].is_directory);
}

#[tokio::test]
async fn list_files_accepts_the_items_shape_and_the_type_filter() {
    let server = MockServer::start().await;
    respond_data(
        &server,
        json!({"items": [
            {"name": "b.txt", "path": "/work/b.txt", "size": 2},
            {"name": "a.rs", "path": "/work/a.rs", "size": 1},
            {"name": "dir", "path": "/work/dir", "is_directory": true},
        ]}),
    )
    .await;
    let mut params = ListFilesParams::new("/work");
    params.file_types = Some(vec![".txt".to_string()]);

    let result = fs(&server).list_files(params).await;

    assert_eq!(result.code, 0);
    let data: ListFilesData = result.data.unwrap();
    assert_eq!(data.total_count, 1);
    assert_eq!(names(&data.list_items), vec!["b.txt"]);
}

#[tokio::test]
async fn list_files_sorts_descending_on_request() {
    let server = MockServer::start().await;
    respond_data(
        &server,
        json!({"list_items": [
            {"name": "a.txt", "size": 1},
            {"name": "b.txt", "size": 3},
            {"name": "c.txt", "size": 2},
        ]}),
    )
    .await;
    let mut params = ListFilesParams::new("/work");
    params.sort_by = SortBy::Size;
    params.sort_descending = true;

    let result = fs(&server).list_files(params).await;

    let data: ListFilesData = result.data.unwrap();
    assert_eq!(names(&data.list_items), vec!["b.txt", "c.txt", "a.txt"]);
}

#[tokio::test]
async fn list_directories_keeps_only_directories_with_a_zero_size() {
    let server = MockServer::start().await;
    respond_data(
        &server,
        json!({"list_items": [
            {"name": "b", "path": "/work/b", "is_directory": true, "size": 10, "type": ".x"},
            {"name": "a", "path": "/work/a", "is_directory": true, "size": 12},
            {"name": "f.txt", "path": "/work/f.txt", "size": 4},
        ]}),
    )
    .await;

    let result = fs(&server)
        .list_directories(ListDirsParams::new("/work"))
        .await;

    assert_eq!(result.code, 0);
    let data: ListFilesData = result.data.unwrap();
    assert_eq!(data.total_count, 2);
    assert_eq!(names(&data.list_items), vec!["a", "b"]);
    assert_eq!(data.list_items[0].size, 0);
    assert_eq!(data.list_items[0].file_type, None);
    assert!(data.list_items.iter().all(|item| item.is_directory));
}

#[tokio::test]
async fn search_files_maps_and_sorts_the_matches() {
    let server = MockServer::start().await;
    respond_data(
        &server,
        json!({"matching_files": [
            {"name": "z.rs", "path": "/work/z.rs", "size": 9},
            {"name": "a.rs", "path": "/work/a.rs", "size": 1},
        ]}),
    )
    .await;

    let result = fs(&server)
        .search_files(SearchFilesParams::new("/work", "*.rs").with_excludes(["target/*"]))
        .await;

    assert_eq!(result.code, 0);
    let data: SearchFilesData = result.data.unwrap();
    assert_eq!(data.total_matches, 2);
    assert_eq!(names(&data.matching_files), vec!["a.rs", "z.rs"]);
    assert_eq!(data.search_path, "/work");
    assert_eq!(data.search_pattern, "*.rs");
    assert_eq!(data.exclude_patterns, Some(vec!["target/*".to_string()]));
}

// ─── transfers ──────────────────────────────────────────────────────────────

/// Mounts a missing-path answer for the upload probe and a write answer.
async fn mock_upload(server: &MockServer, probe: Value, written: Value) {
    Mock::given(method("POST"))
        .and(path(DEFAULT_INVOKE_PATH))
        .and(body_partial_json(json!({"method": "read_file"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(probe))
        .mount(server)
        .await;
    Mock::given(method("POST"))
        .and(path(DEFAULT_INVOKE_PATH))
        .and(body_partial_json(json!({"method": "write_file"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(written))
        .mount(server)
        .await;
}

#[tokio::test]
async fn upload_reads_the_host_file_and_sends_base64_bytes() {
    let dir = tempdir().unwrap();
    let local = dir.path().join("payload.bin");
    tokio::fs::write(&local, [1_u8, 2, 250]).await.unwrap();
    let server = MockServer::start().await;
    mock_upload(
        &server,
        json!({"code": -1, "message": "No such file or directory"}),
        json!({"code": 0, "data": {"size": 3}}),
    )
    .await;

    let result = fs(&server)
        .upload_file(UploadFileParams {
            local_path: local.to_str().unwrap().to_string(),
            target_path: "/work/payload.bin".to_string(),
            ..UploadFileParams::default()
        })
        .await;

    assert_eq!(result.code, 0);
    let data: UploadFileData = result.data.unwrap();
    assert_eq!(data.local_path, local.to_str().unwrap());
    assert_eq!(data.target_path, "/work/payload.bin");
    assert_eq!(data.size, 3);

    let envelopes = envelopes(&server).await;
    assert_eq!(envelopes.len(), 2, "one probe then one write");
    assert_eq!(envelopes[0]["method"], "read_file");
    assert_eq!(envelopes[0]["params"]["mode"], "bytes");
    assert_eq!(envelopes[1]["method"], "write_file");
    assert_eq!(envelopes[1]["params"]["content"], "AQL6");
    assert_eq!(envelopes[1]["params"]["encoding"], "base64");
    assert_eq!(envelopes[1]["params"]["mode"], "bytes");
    assert_eq!(envelopes[1]["params"]["prepend_newline"], false);
    assert_eq!(envelopes[1]["params"]["append_newline"], false);
    assert_eq!(envelopes[1]["params"]["append"], false);
    assert_eq!(envelopes[1]["params"]["create_if_not_exist"], true);
}

#[tokio::test]
async fn upload_skips_the_probe_when_overwriting() {
    let dir = tempdir().unwrap();
    let local = dir.path().join("payload.bin");
    tokio::fs::write(&local, b"hi").await.unwrap();
    let server = MockServer::start().await;
    respond_data(&server, json!({"size": 2})).await;

    let result = fs(&server)
        .upload_file(UploadFileParams {
            local_path: local.to_str().unwrap().to_string(),
            target_path: "/work/payload.bin".to_string(),
            overwrite: true,
            ..UploadFileParams::default()
        })
        .await;

    assert_eq!(result.code, 0);
    let envelopes = envelopes(&server).await;
    assert_eq!(envelopes.len(), 1);
    assert_eq!(envelopes[0]["method"], "write_file");
}

#[tokio::test]
async fn upload_refuses_an_existing_destination() {
    let dir = tempdir().unwrap();
    let local = dir.path().join("payload.bin");
    tokio::fs::write(&local, b"hi").await.unwrap();
    let server = MockServer::start().await;
    mock_upload(
        &server,
        json!({"code": 0, "data": {"content": [1]}}),
        json!({"code": 0, "data": {}}),
    )
    .await;

    let result = fs(&server)
        .upload_file(UploadFileParams {
            local_path: local.to_str().unwrap().to_string(),
            target_path: "/work/payload.bin".to_string(),
            ..UploadFileParams::default()
        })
        .await;

    assert_eq!(result.code, 199_003);
    assert_eq!(
        result.message,
        "file system operation execution error, execution: upload_file, \
         reason: File already exists: /work/payload.bin"
    );
    assert_eq!(envelopes(&server).await.len(), 1, "only the probe ran");
}

#[tokio::test]
async fn upload_reports_a_missing_host_file_without_calling_the_gateway() {
    let server = MockServer::start().await;

    let result = fs(&server)
        .upload_file(UploadFileParams {
            local_path: "/definitely/not/here.bin".to_string(),
            target_path: "/work/payload.bin".to_string(),
            ..UploadFileParams::default()
        })
        .await;

    assert_eq!(result.code, 199_003);
    assert!(
        result.message.contains("failed to read host file"),
        "{result:?}"
    );
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn download_writes_the_local_file_and_creates_the_parents() {
    let dir = tempdir().unwrap();
    let local = dir.path().join("nested/dir/out.bin");
    let server = MockServer::start().await;
    respond_data(&server, json!({"content": [104, 105], "mode": "bytes"})).await;

    let result = fs(&server)
        .download_file(DownloadFileParams {
            source_path: "/work/out.bin".to_string(),
            local_path: local.to_str().unwrap().to_string(),
            ..DownloadFileParams::default()
        })
        .await;

    assert_eq!(result.code, 0);
    let data: DownloadFileData = result.data.unwrap();
    assert_eq!(data.source_path, "/work/out.bin");
    assert_eq!(data.local_path, local.to_str().unwrap());
    assert_eq!(data.size, 2);
    assert_eq!(tokio::fs::read(&local).await.unwrap(), b"hi");
    assert_eq!(
        last_envelope(&server).await["params"]["mode"],
        "bytes",
        "a download reads the remote file in bytes mode"
    );
}

#[tokio::test]
async fn download_decodes_encoded_content_and_honours_overwrite() {
    let dir = tempdir().unwrap();
    let local = dir.path().join("out.txt");
    tokio::fs::write(&local, b"old").await.unwrap();
    let server = MockServer::start().await;
    respond_data(
        &server,
        json!({"content": "6869", "content_encoding": "hex"}),
    )
    .await;
    let params = DownloadFileParams {
        source_path: "/work/out.txt".to_string(),
        local_path: local.to_str().unwrap().to_string(),
        ..DownloadFileParams::default()
    };

    let refused = fs(&server).download_file(params.clone()).await;
    assert_eq!(refused.code, 199_003);
    assert!(refused
        .message
        .contains(&format!("File already exists: {}", local.display())));
    assert_eq!(tokio::fs::read(&local).await.unwrap(), b"old");

    let overwritten = fs(&server)
        .download_file(DownloadFileParams {
            overwrite: true,
            ..params
        })
        .await;
    assert_eq!(overwritten.code, 0);
    assert_eq!(tokio::fs::read(&local).await.unwrap(), b"hi");
}

#[tokio::test]
async fn download_decodes_base64_content() {
    let dir = tempdir().unwrap();
    let local = dir.path().join("out.txt");
    let server = MockServer::start().await;
    respond_data(
        &server,
        json!({"content": "aGVsbG8=", "content_encoding": "base64"}),
    )
    .await;

    let result = fs(&server)
        .download_file(DownloadFileParams {
            source_path: "/work/out.txt".to_string(),
            local_path: local.to_str().unwrap().to_string(),
            ..DownloadFileParams::default()
        })
        .await;

    assert_eq!(result.code, 0);
    assert_eq!(tokio::fs::read(&local).await.unwrap(), b"hello");
}

#[tokio::test]
async fn the_streaming_transfers_emit_exactly_one_terminal_chunk() {
    let dir = tempdir().unwrap();
    let local = dir.path().join("payload.bin");
    tokio::fs::write(&local, b"abc").await.unwrap();
    let server = MockServer::start().await;
    mock_upload(
        &server,
        json!({"code": -1, "message": "not found"}),
        json!({"code": 0, "data": {"size": 7}}),
    )
    .await;

    let chunks: Vec<OpResult<TransferChunkData>> =
        collect(fs_arc(&server).upload_file_stream(UploadFileParams {
            local_path: local.to_str().unwrap().to_string(),
            target_path: "/work/payload.bin".to_string(),
            ..UploadFileParams::default()
        }))
        .await;

    assert_eq!(chunks.len(), 1);
    let data = chunks[0].data.as_ref().unwrap();
    assert_eq!(data.source_path, local.to_str().unwrap());
    assert_eq!(data.target_path, "/work/payload.bin");
    assert_eq!(data.chunk_size, 3, "the real size of the uploaded bytes");
    assert_eq!(data.chunk_index, 0);
    assert!(data.is_last_chunk);
}

#[tokio::test]
async fn a_failing_transfer_stream_emits_one_terminal_error_item() {
    let server = MockServer::start().await;
    respond_data(&server, json!({"content": []})).await;

    let chunks: Vec<OpResult<TransferChunkData>> =
        collect(fs_arc(&server).download_file_stream(DownloadFileParams {
            source_path: "/work/missing.bin".to_string(),
            local_path: "/definitely/not/here.bin".to_string(),
            ..DownloadFileParams::default()
        }))
        .await;

    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].code, 199_003);
    let data = chunks[0].data.as_ref().unwrap();
    assert!(data.is_last_chunk);
    assert_eq!(data.chunk_size, 0);
    assert_eq!(data.source_path, "/work/missing.bin");
    assert_eq!(data.target_path, "/definitely/not/here.bin");
}

// ─── shell ──────────────────────────────────────────────────────────────────

#[tokio::test]
async fn execute_cmd_sends_the_python_signature_parameters() {
    let server = MockServer::start().await;
    respond_data(&server, json!({"stdout": "hi\n", "exit_code": 0})).await;
    let mut params = ShellExecParams::new("echo hi")
        .with_timeout_secs(7)
        .with_cwd("/work")
        .with_shell_type(ShellType::Bash);
    params
        .environment
        .insert("A".to_string(), Some("1".to_string()));
    params.environment.insert("B".to_string(), None);

    let result = shell(&server).execute_cmd(params).await;

    assert_eq!(result.code, 0);
    let envelope = last_envelope(&server).await;
    assert_eq!(envelope["op_type"], "shell");
    assert_eq!(envelope["method"], "execute_cmd");
    assert_eq!(
        envelope["params"],
        json!({
            "command": "echo hi",
            "cwd": "/work",
            "timeout": 7,
            "environment": {"A": "1", "B": null},
            "shell_type": "bash",
        })
    );
}

#[tokio::test]
async fn execute_cmd_defaults_the_timeout_and_drops_a_dot_cwd() {
    let server = MockServer::start().await;
    respond_data(&server, json!({"exit_code": 0})).await;

    let result = shell(&server)
        .execute_cmd(ShellExecParams::new("echo hi").with_cwd("."))
        .await;

    assert_eq!(result.code, 0);
    let envelope = last_envelope(&server).await;
    assert_eq!(envelope["params"]["timeout"], 300);
    assert!(envelope["params"]["cwd"].is_null());
    assert_eq!(envelope["params"]["shell_type"], "auto");
}

#[tokio::test]
async fn an_empty_command_is_rejected_before_calling_the_gateway() {
    let server = MockServer::start().await;

    let result = shell(&server)
        .execute_cmd(ShellExecParams::new("   "))
        .await;

    assert_eq!(result.code, 199_004);
    assert_eq!(
        result.message,
        "shell operation execution error, execution: execute_cmd, \
         reason: command can not be empty"
    );
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn shell_output_prefers_stdout_and_falls_back_to_output() {
    let server = MockServer::start().await;
    respond_data(
        &server,
        json!({
            "command": "make all",
            "cwd": "/work",
            "stdout": "one\n",
            "stderr": "warn\n",
            "exit_code": 1,
            "duration_ms": 12,
        }),
    )
    .await;

    let result = shell(&server)
        .execute_cmd(ShellExecParams::new("make all"))
        .await;

    assert_eq!(result.code, 0, "a non-zero exit code is still a success");
    let data: ShellOutput = result.data.unwrap();
    assert_eq!(data.command, "make all");
    assert_eq!(data.cwd, "/work");
    assert_eq!(data.exit_code, 1);
    assert_eq!(data.stdout, "one\n");
    assert_eq!(data.stderr, "warn\n");
    assert_eq!(data.duration_ms, 12);
    assert!(!data.timed_out);
    assert!(!data.truncated);
    assert!(!data.cancelled);

    let server = MockServer::start().await;
    respond_data(&server, json!({"output": "merged\n", "exit_code": 2})).await;
    let result = shell(&server)
        .execute_cmd(ShellExecParams::new("make all").with_cwd("/work"))
        .await;
    let data: ShellOutput = result.data.unwrap();
    assert_eq!(data.stdout, "merged\n", "a merged output field is stdout");
    assert_eq!(data.stderr, "");
    assert_eq!(
        data.command, "make all",
        "the request command is the fallback"
    );
    assert_eq!(data.exit_code, 2);
}

#[tokio::test]
async fn a_shell_timeout_maps_to_199004_with_output() {
    let server = MockServer::start().await;
    respond_data(&server, json!({"stdout": "partial\n", "exit_code": 124})).await;

    let result = shell(&server)
        .execute_cmd(ShellExecParams::new("sleep 99").with_timeout_secs(5))
        .await;

    assert_eq!(result.code, 199_004);
    assert_eq!(result.message, "execution timeout after 5 seconds");
    let data: ShellOutput = result.data.unwrap();
    assert!(data.timed_out);
    assert_eq!(data.exit_code, 124);
    assert_eq!(data.stdout, "partial\n");
}

#[tokio::test]
async fn execute_cmd_background_sends_its_own_envelope() {
    let server = MockServer::start().await;
    respond_data(
        &server,
        json!({"command": "sleep 1", "cwd": "/work", "pid": 4242}),
    )
    .await;

    let result = shell(&server)
        .execute_cmd_background(ShellExecParams::new("sleep 1").with_cwd("/work"))
        .await;

    assert_eq!(result.code, 0);
    let data: ShellBackgroundOutput = result.data.unwrap();
    assert_eq!(data.command, "sleep 1");
    assert_eq!(data.cwd, "/work");
    assert_eq!(data.pid, Some(4242));
    assert_eq!(data.error, None);

    let envelope = last_envelope(&server).await;
    assert_eq!(envelope["method"], "execute_cmd_background");
    assert_eq!(
        envelope["params"],
        json!({
            "command": "sleep 1",
            "cwd": "/work",
            "environment": {},
            "shell_type": "auto",
        })
    );
}

#[tokio::test]
async fn a_background_launch_error_is_reported_inside_the_output() {
    let server = MockServer::start().await;
    respond_data(
        &server,
        json!({
            "command": "nope",
            "cwd": ".",
            "pid": null,
            "error": "process exited early with code 2: command not found",
        }),
    )
    .await;

    let result = shell(&server)
        .execute_cmd_background(ShellExecParams::new("nope"))
        .await;

    assert_eq!(result.code, 0);
    let data: ShellBackgroundOutput = result.data.unwrap();
    assert_eq!(data.pid, None);
    assert_eq!(
        data.error.as_deref(),
        Some("process exited early with code 2: command not found")
    );
}

#[tokio::test]
async fn execute_argv_quotes_the_argument_vector_into_a_command_line() {
    let server = MockServer::start().await;
    respond_data(&server, json!({"exit_code": 0})).await;

    let result = shell(&server)
        .execute_argv(ShellArgvParams::new(["printf", "%s\\n", "hello world"]))
        .await;

    assert_eq!(result.code, 0);
    let data: ShellOutput = result.data.unwrap();
    assert_eq!(data.command, "printf '%s\\n' 'hello world'");
    let envelope = last_envelope(&server).await;
    assert_eq!(envelope["method"], "execute_cmd");
    assert_eq!(
        envelope["params"]["command"],
        "printf '%s\\n' 'hello world'"
    );
}

#[tokio::test]
async fn an_empty_argv_is_rejected_before_calling_the_gateway() {
    let server = MockServer::start().await;

    let result = shell(&server)
        .execute_argv(ShellArgvParams::default())
        .await;

    assert_eq!(result.code, 199_004);
    assert!(result.message.contains("argv can not be empty"));
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn the_shell_stream_chunks_stdout_then_stderr_and_terminates() {
    let server = MockServer::start().await;
    respond_data(
        &server,
        json!({"stdout": "a\nb\n", "stderr": "e\n", "exit_code": 3}),
    )
    .await;

    let items: Vec<OpResult<ShellChunk>> =
        collect(shell_arc(&server).execute_cmd_stream(ShellExecParams::new("echo a"))).await;

    assert_eq!(items.len(), 4);
    assert!(items.iter().all(|item| item.code == 0));
    let chunks: Vec<&ShellChunk> = items
        .iter()
        .map(|item| item.data.as_ref().unwrap())
        .collect();
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| chunk.text.as_str())
            .collect::<Vec<_>>(),
        vec!["a\n", "b\n", "e\n", ""]
    );
    assert_eq!(
        chunks.iter().map(|chunk| chunk.stream).collect::<Vec<_>>(),
        vec![
            Some(StreamKind::Stdout),
            Some(StreamKind::Stdout),
            Some(StreamKind::Stderr),
            None,
        ]
    );
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| chunk.chunk_index)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
    assert_eq!(chunks[3].exit_code, Some(3));
}

#[tokio::test]
async fn an_empty_shell_stream_still_terminates() {
    let server = MockServer::start().await;
    respond_data(&server, json!({"exit_code": 0})).await;

    let items: Vec<OpResult<ShellChunk>> =
        collect(shell_arc(&server).execute_cmd_stream(ShellExecParams::new("true"))).await;

    assert_eq!(items.len(), 1);
    assert_eq!(items[0].code, 0);
    let chunk = items[0].data.as_ref().unwrap();
    assert_eq!(chunk.chunk_index, 0);
    assert_eq!(chunk.exit_code, Some(0));
    assert!(chunk.stream.is_none());
}

#[tokio::test]
async fn a_failing_shell_stream_emits_one_terminal_error_item() {
    let server = MockServer::start().await;
    respond(&server, json!({"code": -1, "message": "nope"})).await;

    let items: Vec<OpResult<ShellChunk>> =
        collect(shell_arc(&server).execute_cmd_stream(ShellExecParams::new("echo a"))).await;

    assert_eq!(items.len(), 1);
    assert_eq!(items[0].code, 199_007, "gateway failures report 199007");
    assert!(
        items[0].message.contains("aio gateway code -1: nope"),
        "{}",
        items[0].message
    );
    let chunk = items[0].data.as_ref().unwrap();
    assert_eq!(chunk.chunk_index, 0);
    assert_eq!(chunk.exit_code, Some(-1));
}

// ─── code ───────────────────────────────────────────────────────────────────

#[tokio::test]
async fn execute_code_sends_the_python_signature_parameters() {
    let server = MockServer::start().await;
    respond_data(&server, json!({"exit_code": 0})).await;

    let result = code(&server)
        .execute_code(
            CodeExecParams::new("print(1)")
                .with_timeout_secs(9)
                .with_cwd("/work")
                .with_force_file(true),
        )
        .await;

    assert_eq!(result.code, 0);
    let envelope = last_envelope(&server).await;
    assert_eq!(envelope["op_type"], "code");
    assert_eq!(envelope["method"], "execute_code");
    assert_eq!(
        envelope["params"],
        json!({
            "code": "print(1)",
            "language": "python",
            "timeout": 9,
            "environment": {},
            "cwd": "/work",
            "options": {"force_file": true},
        })
    );
}

#[tokio::test]
async fn an_empty_code_snippet_maps_to_199005_with_the_output() {
    let server = MockServer::start().await;

    let result = code(&server)
        .execute_code(CodeExecParams::javascript("   "))
        .await;

    assert_eq!(result.code, 199_005);
    assert_eq!(result.message, "code can not be empty");
    let data: CodeOutput = result.data.unwrap();
    assert_eq!(data.code_content, "   ");
    assert_eq!(data.language, "javascript");
    assert_eq!(data.exit_code, -1);
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn code_output_maps_the_payload_and_falls_back_to_the_request() {
    let server = MockServer::start().await;
    respond_data(
        &server,
        json!({
            "code_content": "print(1)",
            "language": "python",
            "exit_code": 0,
            "stdout": "1\n",
            "duration_ms": 21,
        }),
    )
    .await;

    let result = code(&server)
        .execute_code(CodeExecParams::new("print(1)"))
        .await;

    assert_eq!(result.code, 0);
    let data: CodeOutput = result.data.unwrap();
    assert_eq!(data.code_content, "print(1)");
    assert_eq!(data.language, "python");
    assert_eq!(data.exit_code, 0);
    assert_eq!(data.stdout, "1\n");
    assert_eq!(data.stderr, "");
    assert_eq!(data.duration_ms, 21);
    assert!(!data.timed_out);
    assert!(!data.truncated);

    let server = MockServer::start().await;
    respond_data(&server, json!({})).await;
    let result = code(&server)
        .execute_code(CodeExecParams::javascript("console.log(1)"))
        .await;
    let data: CodeOutput = result.data.unwrap();
    assert_eq!(data.code_content, "console.log(1)");
    assert_eq!(data.language, "javascript");
    assert_eq!(data.exit_code, -1);
}

#[tokio::test]
async fn a_code_gateway_failure_maps_to_199007() {
    let server = MockServer::start().await;
    respond(&server, json!({"code": -1, "message": "no interpreter"})).await;

    let result = code(&server)
        .execute_code(CodeExecParams::new("print(1)"))
        .await;

    assert_eq!(result.code, 199_007);
    assert_eq!(
        result.message,
        "sandbox gateway error, operation: execute_code, \
         reason: aio gateway code -1: no interpreter"
    );
    assert!(result.data.is_none());
}

#[tokio::test]
async fn a_code_timeout_is_detected_from_the_exit_code_or_stderr() {
    let server = MockServer::start().await;
    respond_data(&server, json!({"stdout": "", "exit_code": 124})).await;

    let result = code(&server)
        .execute_code(CodeExecParams::new("import time").with_timeout_secs(2))
        .await;

    assert_eq!(result.code, 199_005);
    assert_eq!(result.message, "execution timeout after 2 seconds");
    let data: CodeOutput = result.data.unwrap();
    assert!(data.timed_out);
    assert_eq!(data.exit_code, 124);

    let server = MockServer::start().await;
    respond_data(
        &server,
        json!({"exit_code": 1, "stderr": "Command Timed Out\n"}),
    )
    .await;
    let result = code(&server)
        .execute_code(CodeExecParams::new("import time").with_timeout_secs(3))
        .await;
    assert_eq!(result.code, 199_005);
    assert_eq!(result.message, "execution timeout after 3 seconds");
    let data: CodeOutput = result.data.unwrap();
    assert!(data.timed_out);
    assert_eq!(data.exit_code, 1, "the reported status is kept");
}

#[tokio::test]
async fn the_code_stream_chunks_the_output_and_terminates() {
    let server = MockServer::start().await;
    respond_data(
        &server,
        json!({"stdout": "one\n", "stderr": "bad\n", "exit_code": 4}),
    )
    .await;

    let items: Vec<OpResult<CodeChunk>> =
        collect(code_arc(&server).execute_code_stream(CodeExecParams::new("print(1)"))).await;

    assert_eq!(items.len(), 3);
    let chunks: Vec<&CodeChunk> = items
        .iter()
        .map(|item| item.data.as_ref().unwrap())
        .collect();
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| chunk.text.as_str())
            .collect::<Vec<_>>(),
        vec!["one\n", "bad\n", ""]
    );
    assert_eq!(chunks[0].stream, Some(StreamKind::Stdout));
    assert_eq!(chunks[1].stream, Some(StreamKind::Stderr));
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| chunk.chunk_index)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(chunks[2].exit_code, Some(4));
}

#[tokio::test]
async fn an_empty_code_stream_still_terminates() {
    let server = MockServer::start().await;
    respond_data(&server, json!({"exit_code": 0})).await;

    let items: Vec<OpResult<CodeChunk>> =
        collect(code_arc(&server).execute_code_stream(CodeExecParams::new("pass"))).await;

    assert_eq!(items.len(), 1);
    assert_eq!(items[0].data.as_ref().unwrap().chunk_index, 0);
    assert_eq!(items[0].data.as_ref().unwrap().exit_code, Some(0));
}

#[tokio::test]
async fn a_failing_code_stream_emits_one_terminal_error_item() {
    let server = MockServer::start().await;
    respond(&server, json!({"code": -1, "message": "nope"})).await;

    let items: Vec<OpResult<CodeChunk>> =
        collect(code_arc(&server).execute_code_stream(CodeExecParams::new("pass"))).await;

    assert_eq!(items.len(), 1);
    assert_eq!(items[0].code, 199_007);
    let chunk = items[0].data.as_ref().unwrap();
    assert_eq!(chunk.chunk_index, 0);
    assert_eq!(chunk.exit_code, Some(-1));
}

// ─── streamed reads ─────────────────────────────────────────────────────────

#[tokio::test]
async fn a_text_read_stream_emits_one_chunk_per_line() {
    let server = MockServer::start().await;
    respond_data(&server, json!({"content": "one\ntwo\nthree"})).await;

    let items: Vec<OpResult<ReadChunkData>> =
        collect(fs_arc(&server).read_file_stream(ReadFileParams::text("/work/a.txt"))).await;

    assert_eq!(items.len(), 3);
    assert!(items.iter().all(|item| item.code == 0));
    let chunks: Vec<&ReadChunkData> = items
        .iter()
        .map(|item| item.data.as_ref().unwrap())
        .collect();
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| match &chunk.chunk {
                FileContent::Text(text) => text.as_str(),
                FileContent::Bytes(_) => panic!("expected text chunks"),
            })
            .collect::<Vec<_>>(),
        vec!["one\n", "two\n", "three"]
    );
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| chunk.is_last_chunk)
            .collect::<Vec<_>>(),
        vec![false, false, true]
    );
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| chunk.chunk_index)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(chunks[0].path, "/work/a.txt");
    assert_eq!(chunks[0].mode, ReadMode::Text);
}

#[tokio::test]
async fn a_byte_read_stream_honors_the_chunk_size() {
    let server = MockServer::start().await;
    respond_data(
        &server,
        json!({"content": [1, 2, 3, 4, 5], "mode": "bytes"}),
    )
    .await;
    let mut params = ReadFileParams::bytes("/work/a.bin");
    params.chunk_size = 2;

    let items: Vec<OpResult<ReadChunkData>> =
        collect(fs_arc(&server).read_file_stream(params)).await;

    assert_eq!(items.len(), 3);
    let chunks: Vec<&ReadChunkData> = items
        .iter()
        .map(|item| item.data.as_ref().unwrap())
        .collect();
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| chunk.chunk_size)
            .collect::<Vec<_>>(),
        vec![2, 2, 1]
    );
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| chunk.chunk_index)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert!(chunks[2].is_last_chunk);
}

#[tokio::test]
async fn an_empty_read_stream_still_terminates() {
    let server = MockServer::start().await;
    respond_data(&server, json!({"content": ""})).await;

    let items: Vec<OpResult<ReadChunkData>> =
        collect(fs_arc(&server).read_file_stream(ReadFileParams::text("/work/empty.txt"))).await;

    assert_eq!(items.len(), 1);
    let chunk = items[0].data.as_ref().unwrap();
    assert_eq!(chunk.chunk_index, 0);
    assert_eq!(chunk.chunk_size, 0);
    assert!(chunk.is_last_chunk);
    assert_eq!(chunk.path, "/work/empty.txt");
}

#[tokio::test]
async fn a_failing_read_stream_emits_one_terminal_error_item() {
    let server = MockServer::start().await;
    respond(&server, json!({"code": -1, "message": "boom"})).await;

    let items: Vec<OpResult<ReadChunkData>> =
        collect(fs_arc(&server).read_file_stream(ReadFileParams::text("/work/a.txt"))).await;

    assert_eq!(items.len(), 1);
    assert_eq!(items[0].code, 199_007);
    assert!(items[0].message.contains("aio gateway code -1: boom"));
    let chunk = items[0].data.as_ref().unwrap();
    assert!(chunk.is_last_chunk);
    assert_eq!(chunk.chunk_size, 0);
    assert_eq!(chunk.chunk_index, 0);
}

// ─── conditional writes ─────────────────────────────────────────────────────

#[tokio::test]
async fn the_conditional_write_flag_comes_from_the_gateway_configuration() {
    let server = MockServer::start().await;
    let config = SandboxGatewayConfig {
        supports_conditional_write: true,
        ..SandboxGatewayConfig::default()
    };
    let fs = AioFs {
        client: Arc::new(AioClient::new(&endpoint(&server), &config).unwrap()),
    };
    assert!(fs.supports_conditional_write());

    let plain = AioFs {
        client: Arc::new(AioClient::new(&endpoint(&server), &gateway_config()).unwrap()),
    };
    assert!(!plain.supports_conditional_write());
}

#[tokio::test]
async fn the_provider_set_forwards_the_conditional_write_flag() {
    let config = SandboxGatewayConfig {
        supports_conditional_write: true,
        ..SandboxGatewayConfig::default()
    };
    let endpoint = SandboxEndpoint {
        base_url: "http://127.0.0.1:1".to_string(),
        ..SandboxEndpoint::default()
    };
    let factory = provider_set().fs.expect("a filesystem factory");
    assert!(factory(&endpoint, &config)
        .unwrap()
        .supports_conditional_write());

    let factory = provider_set().fs.expect("a filesystem factory");
    assert!(!factory(&endpoint, &gateway_config())
        .unwrap()
        .supports_conditional_write());
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
