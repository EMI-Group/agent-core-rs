//! Crate-boundary coverage for the A2A REST (`HTTP+JSON`) binding, through
//! public paths only.
//!
//! Python counterpart: the `a2a-sdk` REST app mounted by
//! `tests/unit_tests/extensions/a2a/test_a2a_server.py` at `rest_url`
//! (`/a2a/rest`), which serves `/v1/message:send`, `/v1/tasks/{id}`, and the
//! SSE routes next to the JSON-RPC binding.
#![cfg(feature = "a2a")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use async_trait::async_trait;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use openjiuwen_common::error::Result;
use serde_json::{json, Value};
use tower::ServiceExt;

use openjiuwen_extensions::a2a::jsonrpc::error_code;
use openjiuwen_extensions::a2a::server::{
    A2aExecutionContext, A2aExecutor, A2aServer, A2aServerConfig, A2aTransport, EventSink,
    DEFAULT_CARD_PATH, DEFAULT_REST_PATH, DEFAULT_REST_PREFIX, DEFAULT_RPC_PATH,
};
use openjiuwen_extensions::a2a::types::{
    AgentCapabilities, AgentCard, Artifact, Part, Task, TaskState, TaskStatus,
};

/// The default REST route base (`DEFAULT_REST_PATH` + `DEFAULT_REST_PREFIX`).
const REST_BASE: &str = "/a2a/rest/v1";

/// An executor that publishes one complete artifact, then completes.
struct FakeExecutor;

#[async_trait]
impl A2aExecutor for FakeExecutor {
    async fn execute(&self, _ctx: A2aExecutionContext, sink: &EventSink) -> Result<()> {
        sink.artifact(
            Artifact {
                artifact_id: String::from("a1"),
                parts: vec![Part::text("hello")],
                ..Artifact::default()
            },
            true,
        );
        Ok(())
    }
}

/// A streaming agent card.
fn card() -> AgentCard {
    AgentCard {
        name: String::from("test-agent"),
        capabilities: AgentCapabilities {
            streaming: true,
            ..AgentCapabilities::default()
        },
        default_input_modes: vec![String::from("text/plain")],
        ..AgentCard::default()
    }
}

/// A server over [`card`] and [`FakeExecutor`], mounting `transport`.
fn server_in(transport: A2aTransport) -> A2aServer {
    A2aServer::new(
        A2aServerConfig::new(card()).with_transport(transport),
        Arc::new(FakeExecutor),
    )
    .expect("the card advertises a supported transport")
}

/// The `SendMessage` params object both bindings accept, carrying `text`.
fn send_params(text: &str) -> Value {
    json!({
        "message": { "messageId": "m1", "role": "ROLE_USER", "parts": [{ "text": text }] }
    })
}

/// Seeds a still-`WORKING` task, as a live streaming turn leaves it.
fn seed_working(server: &A2aServer, id: &str, context_id: &str) {
    server.store().put(Task {
        id: String::from(id),
        context_id: String::from(context_id),
        status: TaskStatus {
            state: TaskState::Working,
            ..TaskStatus::default()
        },
        ..Task::default()
    });
}

/// Issues one request through the router and returns the status, content type,
/// and raw body.
async fn call(
    server: &A2aServer,
    method: &str,
    uri: &str,
    body: Option<&str>,
) -> (StatusCode, String, Vec<u8>) {
    let builder = Request::builder().method(method).uri(uri);
    let request = match body {
        Some(body) => builder
            .header("content-type", "application/json")
            .body(Body::from(body.to_owned()))
            .expect("the request builds"),
        None => builder.body(Body::empty()).expect("the request builds"),
    };
    let response = server
        .router()
        .oneshot(request)
        .await
        .expect("the router answers");
    let status = response.status();
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_owned();
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("the body is readable")
        .to_bytes();
    (status, content_type, bytes.to_vec())
}

/// Decodes a JSON response body.
fn body_json(bytes: &[u8]) -> Value {
    serde_json::from_slice(bytes).expect("the body is JSON")
}

/// Splits an SSE body into its frames, each decoded as a bare JSON value.
fn sse_frames(bytes: &[u8]) -> Vec<Value> {
    std::str::from_utf8(bytes)
        .expect("the SSE body is UTF-8")
        .split("\n\n")
        .filter(|frame| !frame.trim().is_empty())
        .map(|frame| {
            let data = frame
                .strip_prefix("data: ")
                .expect("every frame carries a data line");
            serde_json::from_str(data).expect("every frame is a JSON value")
        })
        .collect()
}

/// The `status.state` of a task or status update.
fn state_of(value: &Value) -> &str {
    value["status"]["state"]
        .as_str()
        .expect("the payload carries a status state")
}

/// The `supportedInterfaces[*].protocolBinding` values of a card.
fn bindings_of(card: &Value) -> Vec<String> {
    card["supportedInterfaces"]
        .as_array()
        .expect("the card lists its interfaces")
        .iter()
        .map(|interface| {
            interface["protocolBinding"]
                .as_str()
                .expect("an interface carries its binding")
                .to_owned()
        })
        .collect()
}

// `test_a2a_server.py::test_agent_card` over the REST card route
#[tokio::test]
async fn card_routes_agree_and_advertise_both_bindings() {
    let server = server_in(A2aTransport::Both);

    let (status, content_type, well_known) = call(&server, "GET", DEFAULT_CARD_PATH, None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(content_type, "application/json");
    let (status, _, rest) = call(&server, "GET", &format!("{REST_BASE}/card"), None).await;
    assert_eq!(status, StatusCode::OK);

    let well_known = body_json(&well_known);
    let rest = body_json(&rest);
    assert_eq!(rest, well_known, "both card routes serve the same card");
    assert_eq!(well_known["name"], "test-agent");
    assert_eq!(
        bindings_of(&well_known),
        vec![String::from("HTTP+JSON"), String::from("JSONRPC")],
        "one interface per mounted binding"
    );
}

// `test_a2a_server.py` REST-only mode: the JSON-RPC route is not mounted
#[tokio::test]
async fn rest_only_mode_advertises_only_the_rest_interface() {
    let server = server_in(A2aTransport::Rest);

    let (status, _, bytes) = call(&server, "GET", DEFAULT_CARD_PATH, None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        bindings_of(&body_json(&bytes)),
        vec![String::from("HTTP+JSON")]
    );

    let body = json!({ "jsonrpc": "2.0", "id": 1, "method": "SendMessage" }).to_string();
    let (status, _, _) = call(&server, "POST", DEFAULT_RPC_PATH, Some(&body)).await;
    assert_eq!(
        status,
        StatusCode::NOT_FOUND,
        "no JSON-RPC route is mounted"
    );
}

// `test_a2a_server.py::test_message_send` over the REST binding
#[tokio::test]
async fn rest_send_answers_with_the_bare_task() {
    let server = server_in(A2aTransport::Both);
    let body = send_params("hi").to_string();

    let (status, content_type, bytes) = call(
        &server,
        "POST",
        &format!("{REST_BASE}/message:send"),
        Some(&body),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(content_type, "application/json");

    let task = body_json(&bytes);
    assert!(
        task.get("jsonrpc").is_none(),
        "REST answers the task itself"
    );
    assert!(task.get("result").is_none(), "REST answers the task itself");
    assert_eq!(state_of(&task), "TASK_STATE_COMPLETED");
    assert_eq!(task["history"][0]["messageId"], "m1");
    assert_eq!(task["history"][0]["role"], "ROLE_USER");
    assert_eq!(task["history"][0]["parts"][0]["text"], "hi");
    assert_eq!(task["artifacts"][0]["artifactId"], "a1");
    assert_eq!(task["artifacts"][0]["parts"][0]["text"], "hello");

    let id = task["id"].as_str().expect("the task carries its id");
    assert_eq!(
        server
            .store()
            .get(id)
            .expect("the task is stored")
            .status
            .state,
        TaskState::Completed
    );
}

// `test_a2a_server.py` v0.3 plural alias, on the REST binding
#[tokio::test]
async fn rest_plural_send_alias_dispatches_like_the_singular_route() {
    let server = server_in(A2aTransport::Both);
    let body = send_params("hi").to_string();

    let (status, _, bytes) = call(
        &server,
        "POST",
        &format!("{REST_BASE}/messages:send"),
        Some(&body),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let task = body_json(&bytes);
    assert!(
        task.get("jsonrpc").is_none(),
        "REST answers the task itself"
    );
    assert_eq!(state_of(&task), "TASK_STATE_COMPLETED");
    assert_eq!(task["history"][0]["parts"][0]["text"], "hi");
    assert_eq!(task["artifacts"][0]["artifactId"], "a1");
}

// `test_a2a_server.py::test_message_stream` over the REST binding
#[tokio::test]
async fn rest_stream_emits_raw_frames_in_the_documented_order() {
    let server = server_in(A2aTransport::Both);
    let body = send_params("hi").to_string();

    let (status, content_type, bytes) = call(
        &server,
        "POST",
        &format!("{REST_BASE}/message:stream"),
        Some(&body),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(content_type, "text/event-stream");

    let frames = sse_frames(&bytes);
    // Task snapshot → status update → artifact chunk → terminal status.
    assert_eq!(frames.len(), 4, "the documented frame order: {frames:?}");
    for frame in &frames {
        assert!(frame.get("jsonrpc").is_none(), "bare frames: {frame}");
        assert!(frame.get("result").is_none(), "bare frames: {frame}");
    }

    assert_eq!(state_of(&frames[0]), "TASK_STATE_SUBMITTED");
    assert_eq!(frames[0]["history"][0]["role"], "ROLE_USER");
    assert_eq!(frames[0]["history"][0]["parts"][0]["text"], "hi");
    assert_eq!(frames[0]["artifacts"], json!([]));

    assert_eq!(state_of(&frames[1]), "TASK_STATE_WORKING");

    assert_eq!(frames[2]["artifact"]["artifactId"], "a1");
    assert_eq!(frames[2]["artifact"]["parts"][0]["text"], "hello");
    assert_eq!(frames[2]["lastChunk"], json!(true));

    assert_eq!(state_of(&frames[3]), "TASK_STATE_COMPLETED");
}

// `test_a2a_server.py::test_get_task` and `::test_get_task_not_found` over REST
#[tokio::test]
async fn rest_get_task_round_trips_and_reports_unknown_ids() {
    let server = server_in(A2aTransport::Both);
    let body = send_params("hi").to_string();
    let (status, _, bytes) = call(
        &server,
        "POST",
        &format!("{REST_BASE}/message:send"),
        Some(&body),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let task = body_json(&bytes);
    let id = task["id"].as_str().expect("the task carries its id");

    let (status, content_type, bytes) =
        call(&server, "GET", &format!("{REST_BASE}/tasks/{id}"), None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(content_type, "application/json");
    let fetched = body_json(&bytes);
    assert_eq!(fetched["id"], json!(id));
    assert_eq!(state_of(&fetched), "TASK_STATE_COMPLETED");

    let (status, _, bytes) =
        call(&server, "GET", &format!("{REST_BASE}/tasks/missing"), None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let envelope = body_json(&bytes);
    assert_eq!(envelope["error"]["code"], json!(error_code::TASK_NOT_FOUND));
    assert_eq!(envelope["error"]["code"], json!(-32001));
    assert_eq!(envelope["error"]["status"], "NOT_FOUND");
    assert!(
        !envelope["error"]["message"]
            .as_str()
            .expect("the envelope carries a message")
            .is_empty(),
        "the envelope explains the failure: {envelope}"
    );
}

// `test_a2a_server.py::test_cancel_task` over the REST sub-path spelling
#[tokio::test]
async fn rest_cancel_sub_path_cancels_a_working_task() {
    let server = server_in(A2aTransport::Both);
    seed_working(&server, "task-cancel", "ctx-cancel");

    let (status, _, bytes) = call(
        &server,
        "POST",
        &format!("{REST_BASE}/tasks/task-cancel/cancel"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let task = body_json(&bytes);
    assert_eq!(state_of(&task), "TASK_STATE_CANCELED");
    assert_eq!(task["id"], "task-cancel");
    assert_eq!(
        server
            .store()
            .get("task-cancel")
            .expect("the cancelled task is stored")
            .status
            .state,
        TaskState::Canceled
    );
}

// `test_a2a_server.py::test_cancel_task` over the REST colon spelling
#[tokio::test]
async fn rest_cancel_colon_spelling_cancels_a_working_task() {
    let server = server_in(A2aTransport::Both);
    seed_working(&server, "task-cancel", "ctx-cancel");

    let (status, _, bytes) = call(
        &server,
        "POST",
        &format!("{REST_BASE}/tasks/task-cancel:cancel"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(state_of(&body_json(&bytes)), "TASK_STATE_CANCELED");
    assert_eq!(
        server
            .store()
            .get("task-cancel")
            .expect("the cancelled task is stored")
            .status
            .state,
        TaskState::Canceled
    );
}

// `test_a2a_server.py::test_cancel_task` follow-up: unknown and terminal tasks
#[tokio::test]
async fn rest_cancel_reports_unknown_and_terminal_tasks() {
    let server = server_in(A2aTransport::Both);

    let (status, _, bytes) = call(
        &server,
        "POST",
        &format!("{REST_BASE}/tasks/missing/cancel"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let envelope = body_json(&bytes);
    assert_eq!(envelope["error"]["code"], json!(error_code::TASK_NOT_FOUND));
    assert_eq!(envelope["error"]["status"], "NOT_FOUND");

    // A non-streaming send always reaches a terminal status.
    let body = send_params("hi").to_string();
    let (status, _, bytes) = call(
        &server,
        "POST",
        &format!("{REST_BASE}/message:send"),
        Some(&body),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let id = body_json(&bytes)["id"]
        .as_str()
        .expect("the task carries its id")
        .to_owned();

    let (status, _, bytes) = call(
        &server,
        "POST",
        &format!("{REST_BASE}/tasks/{id}/cancel"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT);
    let envelope = body_json(&bytes);
    assert_eq!(
        envelope["error"]["code"],
        json!(error_code::TASK_NOT_CANCELABLE)
    );
    assert_eq!(envelope["error"]["code"], json!(-32002));
    assert_eq!(envelope["error"]["status"], "FAILED_PRECONDITION");
}

// `test_a2a_server.py` subscribe: the snapshot, then the current status
#[tokio::test]
async fn rest_subscribe_streams_the_snapshot_then_the_status() {
    let server = server_in(A2aTransport::Both);
    seed_working(&server, "t1", "ctx-1");

    for (method, uri) in [
        ("GET", format!("{REST_BASE}/tasks/t1/subscribe")),
        ("POST", format!("{REST_BASE}/tasks/t1/subscribe")),
        ("GET", format!("{REST_BASE}/tasks/t1:subscribe")),
        ("POST", format!("{REST_BASE}/tasks/t1:subscribe")),
    ] {
        let (status, content_type, bytes) = call(&server, method, &uri, None).await;
        assert_eq!(status, StatusCode::OK, "`{method} {uri}`");
        assert_eq!(content_type, "text/event-stream", "`{method} {uri}`");

        let frames = sse_frames(&bytes);
        assert_eq!(frames.len(), 2, "`{method} {uri}` frames: {frames:?}");
        assert_eq!(frames[0]["id"], "t1", "`{method} {uri}`");
        assert_eq!(frames[0]["contextId"], "ctx-1", "`{method} {uri}`");
        assert_eq!(
            state_of(&frames[0]),
            "TASK_STATE_WORKING",
            "`{method} {uri}`"
        );

        assert_eq!(frames[1]["taskId"], "t1", "`{method} {uri}`");
        assert_eq!(frames[1]["contextId"], "ctx-1", "`{method} {uri}`");
        assert_eq!(
            state_of(&frames[1]),
            "TASK_STATE_WORKING",
            "`{method} {uri}`"
        );
    }
}

// `test_a2a_server.py` subscribe of an unknown task: an error, not a stream
#[tokio::test]
async fn rest_subscribe_unknown_task_is_a_json_404() {
    let server = server_in(A2aTransport::Both);

    let (status, content_type, bytes) = call(
        &server,
        "GET",
        &format!("{REST_BASE}/tasks/missing/subscribe"),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(content_type, "application/json");
    let envelope = body_json(&bytes);
    assert_eq!(envelope["error"]["code"], json!(error_code::TASK_NOT_FOUND));
    assert_eq!(envelope["error"]["status"], "NOT_FOUND");
}

// `test_a2a_server.py` REST parse failure envelope
#[tokio::test]
async fn rest_send_with_a_malformed_body_is_a_parse_error() {
    let server = server_in(A2aTransport::Both);

    let (status, content_type, bytes) = call(
        &server,
        "POST",
        &format!("{REST_BASE}/message:send"),
        Some("not-json"),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(content_type, "application/json");
    let envelope = body_json(&bytes);
    assert_eq!(envelope["error"]["code"], json!(error_code::PARSE));
    assert_eq!(envelope["error"]["code"], json!(-32700));
    assert_eq!(envelope["error"]["status"], "INVALID_ARGUMENT");
    assert!(
        !envelope["error"]["message"]
            .as_str()
            .expect("the envelope carries a message")
            .is_empty(),
        "the envelope explains the failure: {envelope}"
    );
}

// `test_a2a_server.py` REST 405 envelope for a known path, wrong method
#[tokio::test]
async fn rest_method_mismatches_answer_a_json_405() {
    let server = server_in(A2aTransport::Both);
    seed_working(&server, "t1", "ctx-1");

    for (method, uri) in [
        ("GET", format!("{REST_BASE}/message:send")),
        ("DELETE", format!("{REST_BASE}/tasks/t1")),
    ] {
        let (status, content_type, bytes) = call(&server, method, &uri, None).await;
        assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED, "`{method} {uri}`");
        assert_eq!(content_type, "application/json", "`{method} {uri}`");
        let envelope = body_json(&bytes);
        assert_eq!(
            envelope["error"]["code"],
            json!(error_code::METHOD_NOT_FOUND),
            "`{method} {uri}`"
        );
        assert_eq!(envelope["error"]["code"], json!(-32601), "`{method} {uri}`");
        assert_eq!(
            envelope["error"]["status"], "METHOD_NOT_ALLOWED",
            "`{method} {uri}`"
        );
    }
}

// `test_a2a_server.py` configuration: the REST binding is opt-in on the wire
#[tokio::test]
async fn config_serde_keeps_the_jsonrpc_transport_default() {
    let config: A2aServerConfig = serde_json::from_str("{}").expect("an empty object is a config");
    assert_eq!(config.transport, A2aTransport::JsonRpc);
    assert_eq!(config.rest_path, DEFAULT_REST_PATH);
    assert_eq!(config.rest_prefix, DEFAULT_REST_PREFIX);
    assert_eq!(config.rest_base_path(), REST_BASE);
    assert!(config.transport.mounts_json_rpc());
    assert!(!config.transport.mounts_rest());

    for raw in [r#"{"transport":"HTTP+JSON"}"#, r#"{"transport":"rest"}"#] {
        let config: A2aServerConfig =
            serde_json::from_str(raw).expect("the transport alias parses");
        assert_eq!(config.transport, A2aTransport::Rest, "`{raw}`");
    }

    let config = A2aServerConfig::new(card())
        .with_transport(A2aTransport::Both)
        .with_rest_path("/rest/")
        .with_rest_prefix("v2")
        .with_rest_interface_url("http://example.test/rest")
        .with_interface_url("http://example.test/rpc");
    let encoded = serde_json::to_string(&config).expect("the config serializes");
    let decoded: A2aServerConfig =
        serde_json::from_str(&encoded).expect("the config deserializes again");
    assert_eq!(decoded, config);

    // A default-configured server mounts the JSON-RPC binding only.
    let server = A2aServer::new(A2aServerConfig::new(card()), Arc::new(FakeExecutor))
        .expect("the card advertises a supported transport");
    let body = json!({
        "jsonrpc": "2.0", "id": 1, "method": "SendMessage", "params": send_params("hi")
    })
    .to_string();
    let (status, _, bytes) = call(&server, "POST", DEFAULT_RPC_PATH, Some(&body)).await;
    assert_eq!(status, StatusCode::OK);
    let response = body_json(&bytes);
    assert_eq!(response["jsonrpc"], "2.0");
    assert_eq!(state_of(&response["result"]), "TASK_STATE_COMPLETED");

    let (status, _, _) = call(
        &server,
        "POST",
        &format!("{REST_BASE}/message:send"),
        Some(&body),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "no REST route is mounted");
}

// `test_a2a_server.py` both bindings over one executor and one store
#[tokio::test]
async fn both_mode_keeps_the_jsonrpc_envelope_unchanged() {
    let server = server_in(A2aTransport::Both);

    let body = json!({
        "jsonrpc": "2.0", "id": 7, "method": "SendMessage", "params": send_params("hi")
    })
    .to_string();
    let (status, content_type, bytes) = call(&server, "POST", DEFAULT_RPC_PATH, Some(&body)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(content_type, "application/json");
    let response = body_json(&bytes);
    assert!(
        response.get("error").is_none(),
        "the call succeeded: {response}"
    );
    assert_eq!(response["jsonrpc"], "2.0");
    assert_eq!(response["id"], json!(7));
    assert_eq!(state_of(&response["result"]), "TASK_STATE_COMPLETED");
    assert_eq!(response["result"]["artifacts"][0]["artifactId"], "a1");

    // The REST send path answers the task itself on the same server.
    let body = send_params("hi").to_string();
    let (status, _, bytes) = call(
        &server,
        "POST",
        &format!("{REST_BASE}/message:send"),
        Some(&body),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let task = body_json(&bytes);
    assert!(
        task.get("jsonrpc").is_none(),
        "REST answers the task itself"
    );
    assert_eq!(state_of(&task), "TASK_STATE_COMPLETED");
}
