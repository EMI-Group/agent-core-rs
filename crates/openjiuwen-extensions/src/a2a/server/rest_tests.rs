//! Unit tests for the REST (`HTTP+JSON`) binding and the transport-aware
//! configuration.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use super::tests::{card, stored_task, EchoExecutor, FailingExecutor};
use super::*;
use crate::a2a::jsonrpc::error_code;
use crate::a2a::types::{AgentInterface, TaskState};
use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use openjiuwen_common::error::ErrorKind;
use serde_json::{json, Value};
use std::sync::Arc;
use tower::ServiceExt;

/// An interface with `binding` and `url`.
fn interface(url: &str, binding: &str) -> AgentInterface {
    AgentInterface {
        url: String::from(url),
        protocol_binding: String::from(binding),
        protocol_version: String::from("1.0"),
        tenant: None,
    }
}

/// A server in `transport` mode over a fresh in-memory store.
fn server_in(transport: A2aTransport) -> A2aServer {
    A2aServer::new(
        A2aServerConfig::new(card()).with_transport(transport),
        Arc::new(EchoExecutor),
    )
    .unwrap()
}

/// A server in `transport` mode over `store`.
fn server_with_store(transport: A2aTransport, store: &Arc<InMemoryTaskStore>) -> A2aServer {
    let config = A2aServerConfig::new(card()).with_transport(transport);
    A2aServer::with_store(
        config,
        Arc::new(EchoExecutor),
        Arc::clone(store) as Arc<dyn TaskStore>,
    )
    .unwrap()
}

/// Issues one request through the router and returns the status, content type,
/// and body.
async fn call(
    server: &A2aServer,
    method: Method,
    uri: &str,
    body: Option<&str>,
) -> (StatusCode, String, Vec<u8>) {
    let builder = Request::builder().method(method).uri(uri);
    let request = match body {
        Some(body) => builder
            .header("content-type", "application/json")
            .body(Body::from(body.to_owned()))
            .unwrap(),
        None => builder.body(Body::empty()).unwrap(),
    };
    let response = server.router().oneshot(request).await.unwrap();
    let status = response.status();
    let content_type = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_owned();
    let bytes = http_body_util::BodyExt::collect(response.into_body())
        .await
        .unwrap()
        .to_bytes();
    (status, content_type, bytes.to_vec())
}

/// Decodes a JSON response body.
fn body_json(bytes: &[u8]) -> Value {
    serde_json::from_slice(bytes).unwrap()
}

/// Splits an SSE body into its bare JSON frames.
fn sse_frames(bytes: &[u8]) -> Vec<Value> {
    String::from_utf8(bytes.to_vec())
        .unwrap()
        .split("\n\n")
        .filter(|frame| !frame.trim().is_empty())
        .map(|frame| serde_json::from_str::<Value>(frame.strip_prefix("data: ").unwrap()).unwrap())
        .collect()
}

#[test]
fn config_defaults_keep_the_jsonrpc_transport() {
    let config = A2aServerConfig::default();
    assert_eq!(config.transport, A2aTransport::JsonRpc);
    assert_eq!(config.rest_path, "/a2a/rest");
    assert_eq!(config.rest_prefix, "/v1");
    assert!(config.rest_interface_url.is_none());
    assert_eq!(config.rest_base_path(), "/a2a/rest/v1");
    assert!(config.transport.mounts_json_rpc());
    assert!(!config.transport.mounts_rest());
}

#[test]
fn config_deserializes_an_empty_object_to_the_defaults() {
    let config: A2aServerConfig = serde_json::from_str("{}").unwrap();
    assert_eq!(config, A2aServerConfig::default());

    // Unknown fields are tolerated, and the transport accepts its aliases.
    let config: A2aServerConfig = serde_json::from_str(
        r#"{"unknown": true, "transport": "rest", "rest_prefix": "", "host": "0.0.0.0", "port": 0}"#,
    )
    .unwrap();
    assert_eq!(config.transport, A2aTransport::Rest);
    assert_eq!(config.rest_base_path(), "/a2a/rest");
    assert_eq!(config.host, "0.0.0.0");
    assert_eq!(config.port, 0);
}

#[test]
fn config_round_trips_through_json() {
    let config = A2aServerConfig::new(card())
        .with_transport(A2aTransport::Both)
        .with_rest_path("/rest/")
        .with_rest_prefix("v2")
        .with_rest_interface_url("http://example.test/rest")
        .with_interface_url("http://example.test/rpc");
    let encoded = serde_json::to_string(&config).unwrap();
    let decoded: A2aServerConfig = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, config);
}

#[test]
fn rest_base_path_normalizes_slashes() {
    assert_eq!(A2aServerConfig::default().rest_base_path(), "/a2a/rest/v1");
    let config = A2aServerConfig::default()
        .with_rest_path("/a2a/rest/")
        .with_rest_prefix("/v1/");
    assert_eq!(config.rest_base_path(), "/a2a/rest/v1");
    let config = A2aServerConfig::default().with_rest_prefix("");
    assert_eq!(config.rest_base_path(), "/a2a/rest");
    let config = A2aServerConfig::default().with_rest_prefix("v2");
    assert_eq!(config.rest_base_path(), "/a2a/rest/v2");
    let config = A2aServerConfig::default()
        .with_rest_path("/")
        .with_rest_prefix("/v1");
    assert_eq!(config.rest_base_path(), "/v1");
}

#[test]
fn transport_serde_uses_the_wire_spellings_and_aliases() {
    assert_eq!(
        serde_json::to_string(&A2aTransport::JsonRpc).unwrap(),
        r#""JSONRPC""#
    );
    assert_eq!(
        serde_json::to_string(&A2aTransport::Rest).unwrap(),
        r#""HTTP+JSON""#
    );
    assert_eq!(
        serde_json::to_string(&A2aTransport::Both).unwrap(),
        r#""BOTH""#
    );
    let cases = [
        (r#""JSONRPC""#, A2aTransport::JsonRpc),
        (r#""json_rpc""#, A2aTransport::JsonRpc),
        (r#""jsonrpc""#, A2aTransport::JsonRpc),
        (r#""HTTP+JSON""#, A2aTransport::Rest),
        (r#""http+json""#, A2aTransport::Rest),
        (r#""http_json""#, A2aTransport::Rest),
        (r#""rest""#, A2aTransport::Rest),
        (r#""BOTH""#, A2aTransport::Both),
        (r#""both""#, A2aTransport::Both),
    ];
    for (raw, expected) in cases {
        assert_eq!(serde_json::from_str::<A2aTransport>(raw).unwrap(), expected);
    }
}

#[test]
fn transport_builders_select_the_mounted_bindings() {
    assert!(A2aTransport::JsonRpc.mounts_json_rpc());
    assert!(!A2aTransport::JsonRpc.mounts_rest());
    assert!(!A2aTransport::Rest.mounts_json_rpc());
    assert!(A2aTransport::Rest.mounts_rest());
    assert!(A2aTransport::Both.mounts_json_rpc());
    assert!(A2aTransport::Both.mounts_rest());

    let config = A2aServerConfig::default()
        .with_transport(A2aTransport::Both)
        .with_rest_interface_url("http://rest.test/v1");
    assert_eq!(config.transport, A2aTransport::Both);
    assert_eq!(
        config.rest_interface_url.as_deref(),
        Some("http://rest.test/v1")
    );
}

#[test]
fn jsonrpc_mode_serves_the_configured_card_unchanged() {
    let mut configured = card();
    configured.supported_interfaces = vec![
        interface("http://orig.test/rest", "HTTP+JSON"),
        interface("http://orig.test/rpc", "JSONRPC"),
    ];
    let server = A2aServer::new(
        A2aServerConfig::new(configured.clone()),
        Arc::new(EchoExecutor),
    )
    .unwrap();
    assert_eq!(server.agent_card(), &configured);
}

#[test]
fn rest_mode_advertises_exactly_one_rest_interface_first() {
    let mut configured = card();
    configured.supported_interfaces = vec![
        interface("http://orig.test/rpc", "JSONRPC"),
        interface("http://orig.test/rest", "HTTP+JSON"),
        interface("http://custom.test/", "CUSTOM"),
    ];
    let server = server_in(A2aTransport::Rest);
    // The config-derived card (with no configured interfaces) still advertises
    // the mounted binding.
    assert_eq!(server.agent_card().supported_interfaces.len(), 1);
    assert_eq!(
        server.agent_card().supported_interfaces[0].protocol_binding,
        "HTTP+JSON"
    );

    let server = A2aServer::new(
        A2aServerConfig::new(configured).with_transport(A2aTransport::Rest),
        Arc::new(EchoExecutor),
    )
    .unwrap();
    let interfaces = &server.agent_card().supported_interfaces;
    assert_eq!(interfaces.len(), 2, "unexpected interfaces: {interfaces:?}");
    assert_eq!(interfaces[0].protocol_binding, "HTTP+JSON");
    assert_eq!(interfaces[0].url, "http://127.0.0.1:8000/a2a/rest/v1");
    assert_eq!(interfaces[1].protocol_binding, "CUSTOM");
}

#[test]
fn both_mode_advertises_one_interface_per_binding() {
    let mut configured = card();
    configured.supported_interfaces = vec![
        interface("http://orig.test/rest-a", "HTTP+JSON"),
        interface("http://orig.test/rpc", "JSONRPC"),
        interface("http://orig.test/rest-b", "http+json"),
        interface("http://custom.test/", "CUSTOM"),
    ];
    let server = A2aServer::new(
        A2aServerConfig::new(configured).with_transport(A2aTransport::Both),
        Arc::new(EchoExecutor),
    )
    .unwrap();
    let interfaces = &server.agent_card().supported_interfaces;
    assert_eq!(interfaces.len(), 3, "unexpected interfaces: {interfaces:?}");
    assert_eq!(interfaces[0].protocol_binding, "HTTP+JSON");
    assert_eq!(interfaces[0].url, "http://127.0.0.1:8000/a2a/rest/v1");
    assert_eq!(interfaces[1].protocol_binding, "JSONRPC");
    assert_eq!(interfaces[1].url, "http://127.0.0.1:8000/a2a/jsonrpc/");
    assert_eq!(interfaces[2].protocol_binding, "CUSTOM");
}

#[test]
fn card_derivation_honours_the_configured_interface_urls() {
    let config = A2aServerConfig::new(card())
        .with_transport(A2aTransport::Both)
        .with_bind("0.0.0.0", 1234)
        .with_interface_url("https://rpc.example.test/a2a")
        .with_rest_interface_url("https://rest.example.test/a2a");
    let server = A2aServer::new(config, Arc::new(EchoExecutor)).unwrap();
    let interfaces = &server.agent_card().supported_interfaces;
    assert_eq!(interfaces[0].url, "https://rest.example.test/a2a");
    assert_eq!(interfaces[0].protocol_version, "1.0");
    assert_eq!(interfaces[1].url, "https://rpc.example.test/a2a");
    assert_eq!(interfaces[1].protocol_version, "1.0");
}

#[test]
fn construction_rejects_relative_mount_paths() {
    for config in [
        A2aServerConfig::default().with_rpc_path("rpc"),
        A2aServerConfig::default().with_rest_path("rest"),
    ] {
        let error = A2aServer::new(config, Arc::new(EchoExecutor)).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
    }
    let config = A2aServerConfig {
        card_path: String::from("card.json"),
        ..A2aServerConfig::default()
    };
    let error = A2aServer::new(config, Arc::new(EchoExecutor)).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Config);
}

#[test]
fn construction_rejects_a_colliding_rest_base() {
    let config = A2aServerConfig::default()
        .with_transport(A2aTransport::Both)
        .with_rpc_path("/a2a/rest/v1");
    let error = A2aServer::new(config, Arc::new(EchoExecutor)).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(error.to_string().contains("collides"));

    let config = A2aServerConfig::default()
        .with_transport(A2aTransport::Rest)
        .with_rest_path("/.well-known/agent-card.json")
        .with_rest_prefix("");
    let error = A2aServer::new(config, Arc::new(EchoExecutor)).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Config);
}

#[tokio::test]
async fn rest_card_route_serves_the_derived_card() {
    let server = server_in(A2aTransport::Both);
    let (status, content_type, bytes) = call(&server, Method::GET, "/a2a/rest/v1/card", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(content_type, "application/json");
    let card = body_json(&bytes);
    assert_eq!(card["name"], json!("test-agent"));
    let bindings: Vec<&str> = card["supportedInterfaces"]
        .as_array()
        .unwrap()
        .iter()
        .map(|interface| interface["protocolBinding"].as_str().unwrap())
        .collect();
    assert_eq!(bindings, vec!["HTTP+JSON", "JSONRPC"]);
}

#[tokio::test]
async fn rest_send_answers_with_the_stored_task() {
    let server = server_in(A2aTransport::Both);
    let body = json!({ "message": { "role": "ROLE_USER", "parts": [{ "text": "hi" }] } });
    let (status, content_type, bytes) = call(
        &server,
        Method::POST,
        "/a2a/rest/v1/message:send",
        Some(&body.to_string()),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(content_type, "application/json");
    let task = body_json(&bytes);
    assert_eq!(task["status"]["state"], json!("TASK_STATE_COMPLETED"));
    assert_eq!(task["history"][0]["parts"][0]["text"], json!("hi"));
    assert_eq!(task["artifacts"].as_array().unwrap().len(), 2);
    let id = task["id"].as_str().unwrap();
    assert_eq!(
        server.store().get(id).unwrap().status.state,
        TaskState::Completed
    );
}

#[tokio::test]
async fn rest_plural_send_alias_dispatches() {
    let server = server_in(A2aTransport::Both);
    let (status, _, bytes) = call(
        &server,
        Method::POST,
        "/a2a/rest/v1/messages:send",
        Some(&json!({ "query": "hello" }).to_string()),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        body_json(&bytes)["history"][0]["parts"][0]["text"],
        json!("hello")
    );
}

#[tokio::test]
async fn rest_stream_emits_bare_value_frames() {
    let server = server_in(A2aTransport::Both);
    let (status, content_type, bytes) = call(
        &server,
        Method::POST,
        "/a2a/rest/v1/message:stream",
        Some(&json!({ "query": "hi" }).to_string()),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(content_type, "text/event-stream");
    let frames = sse_frames(&bytes);
    assert_eq!(frames.len(), 6, "unexpected frames: {frames:?}");
    for frame in &frames {
        // REST frames carry the event itself, not a JSON-RPC envelope.
        assert!(frame.get("jsonrpc").is_none());
        assert!(frame.get("result").is_none());
    }
    assert_eq!(frames[0]["status"]["state"], json!("TASK_STATE_SUBMITTED"));
    assert_eq!(frames[1]["status"]["state"], json!("TASK_STATE_WORKING"));
    assert_eq!(frames[5]["status"]["state"], json!("TASK_STATE_COMPLETED"));
}

#[tokio::test]
async fn rest_stream_validates_before_opening_the_stream() {
    let server = server_in(A2aTransport::Both);
    let (status, content_type, bytes) = call(
        &server,
        Method::POST,
        "/a2a/rest/v1/message:stream",
        Some("[1, 2]"),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(content_type, "application/json");
    let envelope = body_json(&bytes);
    assert_eq!(envelope["error"]["code"], json!(error_code::INVALID_PARAMS));
    assert_eq!(envelope["error"]["status"], json!("INVALID_ARGUMENT"));
}

#[tokio::test]
async fn rest_executor_failure_is_a_500_envelope() {
    let store = Arc::new(InMemoryTaskStore::new());
    let server = A2aServer::with_store(
        A2aServerConfig::new(card()).with_transport(A2aTransport::Both),
        Arc::new(FailingExecutor),
        Arc::clone(&store) as Arc<dyn TaskStore>,
    )
    .unwrap();
    let (status, _, bytes) = call(
        &server,
        Method::POST,
        "/a2a/rest/v1/message:send",
        Some(&json!({ "query": "hi" }).to_string()),
    )
    .await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    let envelope = body_json(&bytes);
    assert_eq!(envelope["error"]["code"], json!(error_code::INTERNAL));
    assert_eq!(envelope["error"]["status"], json!("INTERNAL"));
    // The task is still stored, in FAILED, as on the JSON-RPC binding.
    assert_eq!(store.all()[0].status.state, TaskState::Failed);
}

#[tokio::test]
async fn rest_invalid_json_is_a_parse_envelope() {
    let server = server_in(A2aTransport::Both);
    let (status, _, bytes) = call(
        &server,
        Method::POST,
        "/a2a/rest/v1/message:send",
        Some("{not json"),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let envelope = body_json(&bytes);
    assert_eq!(envelope["error"]["code"], json!(error_code::PARSE));
    assert_eq!(envelope["error"]["status"], json!("INVALID_ARGUMENT"));
}

#[tokio::test]
async fn rest_get_task_reports_not_found() {
    let server = server_in(A2aTransport::Both);
    let (status, content_type, bytes) =
        call(&server, Method::GET, "/a2a/rest/v1/tasks/missing", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(content_type, "application/json");
    let envelope = body_json(&bytes);
    assert_eq!(envelope["error"]["code"], json!(error_code::TASK_NOT_FOUND));
    assert_eq!(envelope["error"]["status"], json!("NOT_FOUND"));
}

#[tokio::test]
async fn rest_get_task_returns_a_stored_task() {
    let store = Arc::new(InMemoryTaskStore::new());
    store.put(stored_task("t1", TaskState::Working));
    let server = server_with_store(A2aTransport::Both, &store);
    let (status, _, bytes) = call(&server, Method::GET, "/a2a/rest/v1/tasks/t1", None).await;
    assert_eq!(status, StatusCode::OK);
    let task = body_json(&bytes);
    assert_eq!(task["id"], json!("t1"));
    assert_eq!(task["status"]["state"], json!("TASK_STATE_WORKING"));
}

#[tokio::test]
async fn rest_cancel_accepts_both_spellings() {
    let store = Arc::new(InMemoryTaskStore::new());
    store.put(stored_task("t1", TaskState::Working));
    store.put(stored_task("t2", TaskState::Working));
    let server = server_with_store(A2aTransport::Both, &store);

    let (status, _, bytes) =
        call(&server, Method::POST, "/a2a/rest/v1/tasks/t1/cancel", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        body_json(&bytes)["status"]["state"],
        json!("TASK_STATE_CANCELED")
    );
    assert_eq!(store.get("t1").unwrap().status.state, TaskState::Canceled);

    let (status, _, bytes) =
        call(&server, Method::POST, "/a2a/rest/v1/tasks/t2:cancel", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        body_json(&bytes)["status"]["state"],
        json!("TASK_STATE_CANCELED")
    );

    // Cancelling a terminal task is a conflict envelope.
    let (status, _, bytes) =
        call(&server, Method::POST, "/a2a/rest/v1/tasks/t1:cancel", None).await;
    assert_eq!(status, StatusCode::CONFLICT);
    let envelope = body_json(&bytes);
    assert_eq!(
        envelope["error"]["code"],
        json!(error_code::TASK_NOT_CANCELABLE)
    );
    assert_eq!(envelope["error"]["status"], json!("FAILED_PRECONDITION"));
}

#[tokio::test]
async fn rest_subscribe_answers_with_a_snapshot_and_status() {
    let store = Arc::new(InMemoryTaskStore::new());
    store.put(stored_task("t1", TaskState::Working));
    let server = server_with_store(A2aTransport::Both, &store);

    for (method, uri) in [
        (Method::GET, "/a2a/rest/v1/tasks/t1/subscribe"),
        (Method::POST, "/a2a/rest/v1/tasks/t1:subscribe"),
    ] {
        let (status, content_type, bytes) = call(&server, method, uri, None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(content_type, "text/event-stream");
        let frames = sse_frames(&bytes);
        assert_eq!(frames.len(), 2, "unexpected frames: {frames:?}");
        assert_eq!(frames[0]["id"], json!("t1"));
        assert_eq!(frames[1]["status"]["state"], json!("TASK_STATE_WORKING"));
    }
}

#[tokio::test]
async fn rest_unknown_action_is_not_found() {
    let store = Arc::new(InMemoryTaskStore::new());
    store.put(stored_task("t1", TaskState::Working));
    let server = server_with_store(A2aTransport::Both, &store);
    let (status, _, bytes) =
        call(&server, Method::GET, "/a2a/rest/v1/tasks/t1:explode", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(
        body_json(&bytes)["error"]["code"],
        json!(error_code::TASK_NOT_FOUND)
    );
}

#[tokio::test]
async fn rest_method_mismatch_is_a_json_405() {
    let server = server_in(A2aTransport::Both);

    // A known REST path with an unsupported method.
    let (status, content_type, bytes) =
        call(&server, Method::DELETE, "/a2a/rest/v1/message:send", None).await;
    assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(content_type, "application/json");
    let envelope = body_json(&bytes);
    assert_eq!(
        envelope["error"]["code"],
        json!(error_code::METHOD_NOT_FOUND)
    );
    assert_eq!(envelope["error"]["status"], json!("METHOD_NOT_ALLOWED"));

    // A GET where the route only allows POST.
    let (status, _, bytes) = call(&server, Method::GET, "/a2a/rest/v1/tasks/t1/cancel", None).await;
    assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(
        body_json(&bytes)["error"]["status"],
        json!("METHOD_NOT_ALLOWED")
    );

    // A POST to the task lookup route.
    let (status, _, _) = call(&server, Method::POST, "/a2a/rest/v1/tasks/t1", None).await;
    assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED);
}

#[tokio::test]
async fn rest_only_mode_does_not_mount_the_jsonrpc_route() {
    let server = server_in(A2aTransport::Rest);
    let body = json!({ "jsonrpc": "2.0", "id": 1, "method": "SendMessage" }).to_string();
    let (status, _, _) = call(&server, Method::POST, "/a2a/jsonrpc/", Some(&body)).await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    // The well-known card and the REST routes stay mounted.
    let (status, _, _) = call(&server, Method::GET, "/.well-known/agent-card.json", None).await;
    assert_eq!(status, StatusCode::OK);
    let (status, _, _) = call(&server, Method::GET, "/a2a/rest/v1/card", None).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn jsonrpc_only_mode_does_not_mount_the_rest_routes() {
    let server = server_in(A2aTransport::JsonRpc);
    let (status, _, _) = call(&server, Method::GET, "/a2a/rest/v1/card", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, _, _) = call(&server, Method::GET, "/.well-known/agent-card.json", None).await;
    assert_eq!(status, StatusCode::OK);
}
