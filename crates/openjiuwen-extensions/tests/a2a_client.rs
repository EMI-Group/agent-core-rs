//! Crate-boundary integration suite for the A2A client (`feature = "a2a"`).
//!
//! Everything here is driven through the public `openjiuwen_extensions::a2a::*`
//! surface, so it doubles as a guard on the exported client shape. The peer is a
//! `wiremock` server and each case is asserted from both ends: the folded
//! `transform::AgentResult` the caller receives and the JSON-RPC envelope the
//! peer actually received. Python counterparts:
//! `tests/unit_tests/extensions/a2a/test_a2a_client.py` and
//! `tests/unit_tests/extensions/a2a/test_a2a_transformer.py`.
#![cfg(feature = "a2a")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use futures::StreamExt;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_extensions::a2a::client::{
    A2aClient, A2aClientConfig, A2aRemoteClient, RemoteClientConfig,
};
use openjiuwen_extensions::a2a::transform::{AgentResult, AgentStatus};
use openjiuwen_extensions::a2a::types::{
    Artifact, Part, Task, TaskArtifactUpdateEvent, TaskState, TaskStatus, TaskStatusUpdateEvent,
};
use serde_json::{json, Map, Value};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// The JSON-RPC path every mock in this suite serves.
const RPC_PATH: &str = "/a2a/jsonrpc/";

/// The JSON-RPC endpoint of `server`.
fn rpc_url(server: &MockServer) -> String {
    format!("{}{RPC_PATH}", server.uri())
}

/// A client posting to the JSON-RPC endpoint of `server`.
fn client(server: &MockServer) -> A2aClient {
    A2aClient::new(A2aClientConfig::new(rpc_url(server))).expect("the client builds")
}

/// Serializes `frames` as one SSE `data:` frame each.
fn sse_body(frames: &[Value]) -> String {
    let mut body = String::new();
    for frame in frames {
        body.push_str("data: ");
        body.push_str(&frame.to_string());
        body.push_str("\n\n");
    }
    body
}

/// A 200 JSON-RPC response envelope carrying `result`.
fn result_envelope(result: &impl serde::Serialize) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": 1,
        "result": serde_json::to_value(result).expect("the wire type serializes"),
    })
}

/// The single request `server` received, decoded as JSON.
async fn received_body(server: &MockServer) -> Value {
    let requests = server
        .received_requests()
        .await
        .expect("the mock records requests");
    assert_eq!(requests.len(), 1, "exactly one request was sent");
    serde_json::from_slice(&requests[0].body).expect("the request body is JSON")
}

#[tokio::test]
async fn send_folds_a_task_into_an_agent_result_and_posts_send_message() {
    // ports `A2AClient.invoke` (`_merge_agent_results` + `_with_session_id`).
    let server = MockServer::start().await;
    let task = Task {
        id: String::from("task-1"),
        context_id: String::from("ctx-1"),
        status: TaskStatus {
            state: TaskState::Completed,
            ..TaskStatus::default()
        },
        artifacts: vec![Artifact {
            artifact_id: String::from("a1"),
            parts: vec![Part::text("done")],
            ..Artifact::default()
        }],
        metadata: Map::from_iter([(String::from("k"), json!("v"))]),
        ..Task::default()
    };
    Mock::given(method("POST"))
        .and(path(RPC_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(result_envelope(&task)))
        .mount(&server)
        .await;

    let result = client(&server)
        .invoke(json!({ "query": "hi", "conversation_id": "sess-1" }))
        .await
        .expect("the send succeeds");

    assert_eq!(result.status, AgentStatus::Completed);
    assert_eq!(result.task_id.as_deref(), Some("task-1"));
    // The request session id wins over the task's own `contextId`.
    assert_eq!(result.session_id.as_deref(), Some("sess-1"));
    assert_eq!(result.artifacts.len(), 1);
    assert_eq!(result.artifacts[0].artifact_id.as_deref(), Some("a1"));
    assert_eq!(result.artifacts[0].parts[0].text.as_deref(), Some("done"));
    assert_eq!(result.metadata.get("k"), Some(&json!("v")));

    let body = received_body(&server).await;
    assert_eq!(body["jsonrpc"], "2.0");
    assert_eq!(body["method"], "SendMessage");
    assert_eq!(body["params"]["message"]["role"], "ROLE_USER");
    assert_eq!(body["params"]["message"]["parts"][0]["text"], "hi");
    assert_eq!(body["params"]["message"]["contextId"], "sess-1");
    assert!(
        body["params"]["message"]["messageId"]
            .as_str()
            .is_some_and(|id| !id.is_empty()),
        "the client generates a message id: {body}"
    );
}

#[tokio::test]
async fn stream_yields_one_result_per_sse_frame() {
    // ports `A2AClient.stream` yielding one `from_a2a_response` per SSE event.
    let server = MockServer::start().await;
    let status = |state: TaskState| TaskStatusUpdateEvent {
        task_id: String::from("task-1"),
        context_id: String::from("ctx-1"),
        status: TaskStatus {
            state,
            ..TaskStatus::default()
        },
        ..TaskStatusUpdateEvent::default()
    };
    let chunk = TaskArtifactUpdateEvent {
        task_id: String::from("task-1"),
        context_id: String::from("ctx-1"),
        artifact: Artifact {
            artifact_id: String::from("a1"),
            parts: vec![Part::text("chunk")],
            ..Artifact::default()
        },
        ..TaskArtifactUpdateEvent::default()
    };
    let body = sse_body(&[
        result_envelope(&status(TaskState::Working)),
        result_envelope(&chunk),
        result_envelope(&status(TaskState::Completed)),
    ]);
    Mock::given(method("POST"))
        .and(path(RPC_PATH))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .mount(&server)
        .await;

    let frames: Vec<AgentResult> = client(&server)
        .stream(json!({ "query": "go", "sessionId": "s-1" }))
        .map(|frame| frame.expect("every frame decodes"))
        .collect()
        .await;

    assert_eq!(frames.len(), 3, "one result per SSE frame");
    assert_eq!(frames[0].status, AgentStatus::Working);
    assert_eq!(frames[0].artifacts, Vec::new());
    assert_eq!(frames[1].artifacts.len(), 1);
    assert_eq!(frames[1].artifacts[0].artifact_id.as_deref(), Some("a1"));
    assert_eq!(
        frames[1].artifacts[0].parts[0].text.as_deref(),
        Some("chunk")
    );
    assert_eq!(frames[2].status, AgentStatus::Completed);
    for frame in &frames {
        assert_eq!(frame.session_id.as_deref(), Some("s-1"));
        assert_eq!(frame.task_id.as_deref(), Some("task-1"));
    }

    let body = received_body(&server).await;
    assert_eq!(body["method"], "SendStreamingMessage");
}

#[tokio::test]
async fn cancel_task_sends_cancel_task_and_maps_canceled() {
    let server = MockServer::start().await;
    let task = Task {
        id: String::from("task-9"),
        context_id: String::from("ctx-9"),
        status: TaskStatus {
            state: TaskState::Canceled,
            ..TaskStatus::default()
        },
        ..Task::default()
    };
    Mock::given(method("POST"))
        .and(path(RPC_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(result_envelope(&task)))
        .mount(&server)
        .await;

    let result = client(&server)
        .cancel_task("task-9", Some("tenant-7"))
        .await
        .expect("the cancel succeeds");

    assert_eq!(result.status, AgentStatus::Canceled);
    assert_eq!(result.task_id.as_deref(), Some("task-9"));

    let body = received_body(&server).await;
    assert_eq!(body["method"], "CancelTask");
    assert_eq!(body["params"]["id"], "task-9");
    assert_eq!(body["params"]["tenant"], "tenant-7");
}

#[tokio::test]
async fn json_rpc_error_becomes_a_typed_error() {
    // `-32001` is `TASK_NOT_FOUND`; the peer's message must survive.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(RPC_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "jsonrpc": "2.0",
            "id": 1,
            "error": { "code": -32001, "message": "Task not found" }
        })))
        .mount(&server)
        .await;

    let error = client(&server)
        .invoke(json!({ "query": "hi" }))
        .await
        .expect_err("the JSON-RPC error is reported");

    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(
        error.to_string().contains("Task not found"),
        "the peer message survives: {error}"
    );
}

#[tokio::test]
async fn remote_client_rejects_a_config_without_a_card() {
    // Python's `RemoteClientConfig` requires the card in `kwargs`.
    let error = A2aRemoteClient::new(RemoteClientConfig {
        url: String::from("http://127.0.0.1:9"),
        ..RemoteClientConfig::default()
    })
    .expect_err("the card is required");

    assert_eq!(error.kind(), ErrorKind::Validation);
}

/// Serves the SSE `frames` on a real loopback socket, sleeping `gap` before
/// every frame and (optionally) `stall` after the last one.
///
/// `wiremock` delivers a response body in one piece, so pacing needs an `axum`
/// server that streams the frames itself.
async fn serve_paced_stream(
    frames: Vec<String>,
    gap: std::time::Duration,
    stall: Option<std::time::Duration>,
) -> String {
    use axum::body::Body;
    use axum::http::{header, StatusCode};
    use axum::routing::post;
    use axum::Router;

    let app = Router::new().route(
        RPC_PATH,
        post(move || async move {
            let body = Body::from_stream(async_stream::stream! {
                for frame in frames {
                    tokio::time::sleep(gap).await;
                    yield Ok::<_, std::io::Error>(bytes::Bytes::from(frame));
                }
                if let Some(stall) = stall {
                    tokio::time::sleep(stall).await;
                }
            });
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/event-stream")],
                body,
            )
        }),
    );
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .expect("the stream server binds a loopback port");
    let addr = listener.local_addr().expect("the listener has an address");
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    format!("http://{addr}{RPC_PATH}")
}

/// One `data:` SSE frame carrying a completed task status.
fn working_frame(task_id: &str) -> String {
    let envelope = result_envelope(&TaskStatusUpdateEvent {
        task_id: task_id.to_string(),
        context_id: String::from("ctx-1"),
        status: TaskStatus {
            state: TaskState::Working,
            ..TaskStatus::default()
        },
        ..TaskStatusUpdateEvent::default()
    });
    format!("data: {envelope}\n\n")
}

/// A healthy stream may run far longer than the request budget: `timeout` is an
/// idle deadline between chunks, not a total budget, so pacing a stream slower
/// than the budget must not cut it off.
#[tokio::test]
async fn a_slow_stream_outlives_the_request_budget() {
    let budget = std::time::Duration::from_millis(500);
    let frames = vec![
        working_frame("task-1"),
        working_frame("task-1"),
        working_frame("task-1"),
    ];
    let url = serve_paced_stream(frames, std::time::Duration::from_millis(200), None).await;
    let client =
        A2aClient::new(A2aClientConfig::new(url).with_timeout(budget)).expect("the client builds");

    let started = std::time::Instant::now();
    let results: Vec<Result<AgentResult, _>> = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        client.stream(json!({ "query": "go" })).collect(),
    )
    .await
    .expect("the stream finished within the test bound");
    let elapsed = started.elapsed();

    assert_eq!(results.len(), 3, "every paced frame arrives");
    for result in &results {
        let result = result.as_ref().expect("a healthy stream yields no error");
        assert_eq!(result.status, AgentStatus::Working);
    }
    assert!(
        elapsed > budget,
        "the stream outlived the {budget:?} budget: {elapsed:?}"
    );
}

/// A peer that answers the headers and then goes quiet is a timeout, so retry
/// policies see `Error::Timeout` instead of a transport error or a hang.
#[tokio::test]
async fn a_stalled_stream_times_out_with_the_typed_error() {
    let budget = std::time::Duration::from_millis(200);
    let url = serve_paced_stream(
        vec![working_frame("task-1")],
        std::time::Duration::from_millis(0),
        Some(std::time::Duration::from_secs(30)),
    )
    .await;
    let client =
        A2aClient::new(A2aClientConfig::new(url).with_timeout(budget)).expect("the client builds");

    let started = std::time::Instant::now();
    let results: Vec<Result<AgentResult, _>> = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        client.stream(json!({ "query": "go" })).collect(),
    )
    .await
    .expect("the stalled stream ended instead of hanging");
    let elapsed = started.elapsed();

    assert_eq!(results.len(), 2, "one frame, then the failure");
    assert!(results[0].is_ok(), "the frame that arrived is delivered");
    let error = results[1].as_ref().expect_err("the idle peer is an error");
    assert_eq!(error.kind(), ErrorKind::Timeout);
    assert!(error.to_string().contains("timed out"), "message {error}");
    assert!(
        elapsed >= budget,
        "the idle deadline was {budget:?}, elapsed {elapsed:?}"
    );
}
