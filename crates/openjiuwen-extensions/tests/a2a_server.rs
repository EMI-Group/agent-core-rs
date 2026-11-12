//! Crate-boundary integration coverage for the A2A server.
//!
//! Ports `tests/unit_tests/extensions/a2a/test_a2a_server.py`: a real
//! [`A2aServer`] is driven through its axum router with `tower`'s in-process
//! `oneshot`, asserting the client-visible JSON-RPC envelopes, agent card, and
//! SSE frames. Only the public `openjiuwen_extensions::a2a::` surface is used.
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

use openjiuwen_extensions::a2a::jsonrpc::{error_code, JsonRpcResponse};
use openjiuwen_extensions::a2a::server::{
    A2aExecutionContext, A2aExecutor, A2aServer, A2aServerConfig, EventSink, DEFAULT_CARD_PATH,
    DEFAULT_RPC_PATH,
};
use openjiuwen_extensions::a2a::types::{
    AgentCapabilities, AgentCard, Artifact, Message, Part, Role, Task, TaskState, TaskStatus,
};

/// The card every server in this suite publishes.
fn card() -> AgentCard {
    AgentCard {
        name: String::from("test-agent"),
        description: String::from("d"),
        capabilities: AgentCapabilities {
            streaming: true,
            push_notifications: false,
        },
        default_input_modes: vec![String::from("text/plain")],
        default_output_modes: vec![String::from("text/plain")],
        ..AgentCard::default()
    }
}

/// An executor that publishes one artifact chunk, then completes.
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

/// A server over [`card`] and [`FakeExecutor`].
fn server() -> A2aServer {
    A2aServer::new(A2aServerConfig::new(card()), Arc::new(FakeExecutor))
        .expect("the card advertises a JSON-RPC transport")
}

/// The `SendMessage` params carrying `text` as a user message.
fn send_params(text: &str) -> Value {
    json!({
        "message": {
            "messageId": "m1",
            "role": "ROLE_USER",
            "parts": [{ "text": text }]
        }
    })
}

/// POSTs `body` to the JSON-RPC route and returns the raw response.
async fn post_response(server: &A2aServer, body: &Value) -> axum::response::Response {
    server
        .router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(DEFAULT_RPC_PATH)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("the request builds"),
        )
        .await
        .expect("the router answers")
}

/// POSTs `body` to the JSON-RPC route and decodes the JSON-RPC envelope.
async fn post(server: &A2aServer, body: &Value) -> JsonRpcResponse {
    let response = post_response(server, body).await;
    assert_eq!(response.status(), StatusCode::OK, "JSON-RPC answers 200");
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("the body is readable")
        .to_bytes();
    serde_json::from_slice(&bytes).expect("the body is a JSON-RPC response")
}

/// GETs `uri` from the router and returns the status and decoded body.
async fn get(server: &A2aServer, uri: &str) -> (StatusCode, Value) {
    let response = server
        .router()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri(uri)
                .body(Body::empty())
                .expect("the request builds"),
        )
        .await
        .expect("the router answers");
    let status = response.status();
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("the body is readable")
        .to_bytes();
    let body = serde_json::from_slice(&bytes).expect("the body is JSON");
    (status, body)
}

/// Decodes every SSE frame of `response` as a JSON-RPC envelope.
async fn sse_frames(response: axum::response::Response) -> Vec<JsonRpcResponse> {
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream"),
        "a streaming call answers with an event stream"
    );
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("the body is readable")
        .to_bytes();
    let text = std::str::from_utf8(&bytes).expect("the SSE body is UTF-8");
    text.split("\n\n")
        .filter(|frame| !frame.is_empty())
        .map(|frame| {
            let data = frame
                .strip_prefix("data: ")
                .expect("every frame carries a data line");
            serde_json::from_str::<JsonRpcResponse>(data)
                .expect("every frame is a JSON-RPC response")
        })
        .collect()
}

/// The `status.state` of a successful frame.
fn state_of(frame: &JsonRpcResponse) -> &str {
    frame.result.as_ref().expect("the frame carries a result")["status"]["state"]
        .as_str()
        .expect("the frame carries a status state")
}

// ports `test_a2a_server.py::test_card_route` (card served at the well-known path)
#[tokio::test]
async fn card_route_serves_the_agent_card() {
    let server = server();

    let (status, card) = get(&server, DEFAULT_CARD_PATH).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(card["name"], "test-agent");
    assert_eq!(card["capabilities"]["streaming"], json!(true));
    assert!(
        card["defaultInputModes"]
            .as_array()
            .expect("the modes are a list")
            .contains(&json!("text/plain")),
        "the default input modes are published: {card}"
    );
}

// ports `test_a2a_server.py::test_message_send` and the v0.3 alias dispatch
#[tokio::test]
async fn send_message_and_the_v0_3_alias_both_dispatch() {
    let server = server();

    for method in ["message/send", "SendMessage"] {
        let response = post(
            &server,
            &json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": method,
                "params": send_params("hi")
            }),
        )
        .await;

        assert!(
            response.error.is_none(),
            "`{method}` succeeded: {response:?}"
        );
        let result = response.result.expect("the call returns a Task");
        assert!(
            result["id"].as_str().is_some(),
            "`{method}` returns a Task with an id: {result}"
        );
        assert!(
            result["contextId"].as_str().is_some(),
            "`{method}` returns a Task with a context: {result}"
        );
        assert_eq!(
            result["status"]["state"], "TASK_STATE_COMPLETED",
            "`{method}` completed the task"
        );
        let artifact_ids: Vec<&Value> = result["artifacts"]
            .as_array()
            .expect("the task carries artifacts")
            .iter()
            .map(|artifact| &artifact["artifactId"])
            .collect();
        assert_eq!(artifact_ids, vec![&json!("a1")], "`{method}`");
    }
}

// ports `test_a2a_server.py::test_get_task_not_found`
#[tokio::test]
async fn get_task_for_an_unknown_id_reports_task_not_found() {
    let server = server();

    let response = post(
        &server,
        &json!({ "jsonrpc": "2.0", "id": 1, "method": "tasks/get", "params": { "id": "nope" } }),
    )
    .await;

    assert!(response.result.is_none());
    let error = response.error.expect("an unknown task is an error");
    assert_eq!(error.code, error_code::TASK_NOT_FOUND);
    assert_eq!(error.code, -32001);
}

// ports `test_a2a_server.py::test_cancel_task` and the not-cancelable follow-up
#[tokio::test]
async fn cancel_task_returns_canceled_then_reports_not_cancelable() {
    let server = server();
    // A synchronous `SendMessage` always publishes a terminal status, so the
    // only task a cancel can legally act on is seeded through the public
    // `TaskStore` SPI: one still `WORKING`, as a live streaming turn has it.
    server.store().put(Task {
        id: String::from("task-cancel"),
        context_id: String::from("ctx-cancel"),
        status: TaskStatus {
            state: TaskState::Working,
            ..TaskStatus::default()
        },
        history: vec![Message {
            message_id: String::from("m-cancel"),
            role: Role::User,
            parts: vec![Part::text("hi")],
            ..Message::default()
        }],
        ..Task::default()
    });

    let response = post(
        &server,
        &json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tasks/cancel",
            "params": { "id": "task-cancel" }
        }),
    )
    .await;

    assert!(
        response.error.is_none(),
        "the cancel succeeds: {response:?}"
    );
    let result = response.result.expect("the cancel returns the task");
    assert_eq!(result["status"]["state"], "TASK_STATE_CANCELED");
    assert_eq!(result["id"], "task-cancel");
    assert_eq!(
        server
            .store()
            .get("task-cancel")
            .expect("the cancelled task is stored")
            .status
            .state,
        TaskState::Canceled
    );

    // Cancelling the now-terminal task again is refused.
    let again = post(
        &server,
        &json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tasks/cancel",
            "params": { "id": "task-cancel" }
        }),
    )
    .await;
    let error = again.error.expect("a terminal task is not cancelable");
    assert_eq!(error.code, error_code::TASK_NOT_CANCELABLE);
    assert_eq!(error.code, -32002);
}

// ports `test_a2a_server.py::test_message_stream` (Task → status → terminal)
#[tokio::test]
async fn message_stream_emits_the_python_frame_order() {
    let server = server();

    let response = post_response(
        &server,
        &json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "message/stream",
            "params": send_params("hi")
        }),
    )
    .await;
    let frames = sse_frames(response).await;

    assert_eq!(
        frames.len(),
        4,
        "submitted, working, one artifact chunk, completed: {frames:?}"
    );

    // The initial Task snapshot must come before any status update.
    assert_eq!(state_of(&frames[0]), "TASK_STATE_SUBMITTED");
    let snapshot = frames[0].result.as_ref().expect("the snapshot is a result");
    assert!(snapshot["id"].as_str().is_some(), "the snapshot is a Task");
    assert_eq!(snapshot["history"][0]["role"], "ROLE_USER");
    assert_eq!(snapshot["history"][0]["parts"][0]["text"], "hi");
    assert_eq!(snapshot["artifacts"], json!([]));

    assert_eq!(state_of(&frames[1]), "TASK_STATE_WORKING");

    let chunk = frames[2]
        .result
        .as_ref()
        .expect("the artifact chunk is a result");
    assert_eq!(chunk["artifact"]["artifactId"], "a1");
    assert_eq!(chunk["artifact"]["parts"][0]["text"], "hello");
    assert_eq!(chunk["lastChunk"], json!(true));

    assert_eq!(state_of(&frames[3]), "TASK_STATE_COMPLETED");
    for frame in &frames {
        assert_eq!(frame.id, json!(1), "every frame answers the request id");
    }
}
