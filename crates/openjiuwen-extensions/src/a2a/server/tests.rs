//! Unit tests for the server: routing, JSON-RPC dispatch, the Python event
//! order, the store, and the start/stop lifecycle.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use super::*;
use crate::a2a::jsonrpc::error_code;
use crate::a2a::types::{AgentInterface, Artifact, Part, Task, TaskState, TaskStatus};
use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use openjiuwen_common::error::{Error, ErrorKind, Result};
use serde_json::{json, Map, Value};
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

use super::event_sink::{EventStream, EVENT_BUFFER_CAPACITY};

/// Compile-time check: every public item stays reachable at
/// `crate::a2a::server::` after the module split into `server/`.
#[allow(unused_imports)]
mod public_paths {
    use crate::a2a::server::{
        A2aExecutionContext, A2aExecutor, A2aServer, A2aServerAdapter, A2aServerConfig,
        A2aTransport, EventSink, InMemoryTaskStore, RetentionPolicy, ServerHandle, TaskStore,
        DEFAULT_CARD_PATH, DEFAULT_HOST, DEFAULT_MAX_TASKS, DEFAULT_PORT, DEFAULT_PROTOCOL_VERSION,
        DEFAULT_REST_PATH, DEFAULT_REST_PREFIX, DEFAULT_RPC_PATH, DEFAULT_TERMINAL_TASK_TTL_MS,
    };
}

/// An executor that publishes an open artifact and a terminal one.
pub(super) struct EchoExecutor;

#[async_trait::async_trait]
impl A2aExecutor for EchoExecutor {
    async fn execute(&self, _ctx: A2aExecutionContext, sink: &EventSink) -> Result<()> {
        sink.artifact(
            Artifact {
                parts: vec![Part::text("chunk")],
                ..Artifact::default()
            },
            false,
        );
        sink.artifact(
            Artifact {
                artifact_id: String::from("a2"),
                parts: vec![Part::text("done")],
                ..Artifact::default()
            },
            true,
        );
        Ok(())
    }
}

/// An executor that fails.
pub(super) struct FailingExecutor;

#[async_trait::async_trait]
impl A2aExecutor for FailingExecutor {
    async fn execute(&self, _ctx: A2aExecutionContext, _sink: &EventSink) -> Result<()> {
        Err(Error::validation("boom"))
    }
}

pub(super) fn card() -> types::AgentCard {
    types::AgentCard {
        name: String::from("test-agent"),
        description: String::from("does things"),
        version: String::from("1.0"),
        capabilities: types::AgentCapabilities {
            streaming: true,
            push_notifications: false,
        },
        ..types::AgentCard::default()
    }
}

fn server(executor: Arc<dyn A2aExecutor>) -> A2aServer {
    A2aServer::new(A2aServerConfig::new(card()), executor).unwrap()
}

/// A stored task in `state`, for cancel/resubscribe tests.
pub(super) fn stored_task(id: &str, state: TaskState) -> Task {
    Task {
        id: String::from(id),
        context_id: String::from("ctx-1"),
        status: TaskStatus {
            state,
            message: None,
            timestamp: None,
        },
        artifacts: Vec::new(),
        history: Vec::new(),
        metadata: Map::new(),
    }
}

/// Drains a sink stream without blocking.
fn drain(stream: &mut EventStream) -> Vec<Value> {
    std::iter::from_fn(|| stream.try_recv().ok()).collect()
}

/// POSTs one JSON-RPC request through the router and returns the envelope.
async fn rpc(server: &A2aServer, method_name: &str, params: Value) -> Value {
    let body = json!({
        "jsonrpc": "2.0",
        "id": 7,
        "method": method_name,
        "params": params,
    })
    .to_string();
    let request = Request::builder()
        .method(Method::POST)
        .uri("/a2a/jsonrpc/")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();
    let response = server.router().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = http_body_util::BodyExt::collect(response.into_body())
        .await
        .unwrap()
        .to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

/// The `result` of a successful envelope, panicking on an error envelope.
fn result(envelope: &Value) -> &Value {
    assert!(
        envelope.get("error").is_none(),
        "unexpected error: {envelope}"
    );
    &envelope["result"]
}

#[test]
fn in_memory_task_store_put_get_and_all() {
    let store = InMemoryTaskStore::new();
    assert!(store.get("missing").is_none());
    assert_eq!(store.all().len(), 0);

    store.put(stored_task("t2", TaskState::Working));
    store.put(stored_task("t1", TaskState::Submitted));
    assert_eq!(store.get("t1").unwrap().status.state, TaskState::Submitted);
    // `all` is sorted by id.
    let ids: Vec<String> = store.all().into_iter().map(|task| task.id).collect();
    assert_eq!(ids, vec![String::from("t1"), String::from("t2")]);

    // `put` replaces by id.
    store.put(stored_task("t1", TaskState::Completed));
    assert_eq!(store.get("t1").unwrap().status.state, TaskState::Completed);
    assert_eq!(store.all().len(), 2);
}

#[test]
fn event_sink_publishes_the_python_event_order() {
    // ports `tests/unit_tests/extensions/a2a/test_a2a_server.py` (event order).
    let store = Arc::new(InMemoryTaskStore::new());
    let (sink, mut receiver) = EventSink::new(
        String::from("t1"),
        String::from("c1"),
        Arc::clone(&store) as Arc<dyn TaskStore>,
    );
    sink.task(stored_task("t1", TaskState::Submitted));
    sink.working(Some(agent_message("working", "t1", "c1")));
    sink.artifact(
        Artifact {
            artifact_id: String::from("open"),
            parts: vec![Part::text("one")],
            ..Artifact::default()
        },
        false,
    );
    sink.artifact(
        Artifact {
            artifact_id: String::from("final"),
            parts: vec![Part::text("two")],
            ..Artifact::default()
        },
        true,
    );
    sink.complete();

    let events = drain(&mut receiver);
    assert_eq!(events.len(), 6, "unexpected events: {events:?}");
    // 1. the Task snapshot.
    assert_eq!(events[0]["id"], json!("t1"));
    assert_eq!(events[0]["status"]["state"], json!("TASK_STATE_SUBMITTED"));
    // 2. WORKING.
    assert_eq!(events[1]["status"]["state"], json!("TASK_STATE_WORKING"));
    assert_eq!(events[1]["taskId"], json!("t1"));
    assert_eq!(events[1]["contextId"], json!("c1"));
    // 3. the open artifact chunk.
    assert_eq!(events[2]["artifact"]["artifactId"], json!("open"));
    assert_eq!(events[2]["lastChunk"], json!(false));
    assert_eq!(events[2]["append"], json!(false));
    // 4. the terminal artifact chunk.
    assert_eq!(events[3]["artifact"]["artifactId"], json!("final"));
    assert_eq!(events[3]["lastChunk"], json!(true));
    // 5. the open artifact is closed without replacing its parts.
    assert_eq!(events[4]["artifact"]["artifactId"], json!("open"));
    assert_eq!(events[4]["artifact"]["parts"], json!([]));
    assert_eq!(events[4]["append"], json!(true));
    assert_eq!(events[4]["lastChunk"], json!(true));
    // 6. the terminal status.
    assert_eq!(events[5]["status"]["state"], json!("TASK_STATE_COMPLETED"));

    let stored = store.get("t1").unwrap();
    assert_eq!(stored.status.state, TaskState::Completed);
    assert_eq!(stored.artifacts.len(), 2);
    let open = &stored.artifacts[0];
    assert_eq!(open.artifact_id, "open");
    assert_eq!(open.parts, vec![Part::text("one")]);
    assert_eq!(stored.artifacts[1].artifact_id, "final");
}

#[test]
fn terminal_events_are_idempotent() {
    let store = Arc::new(InMemoryTaskStore::new());
    let (sink, mut receiver) = EventSink::new(
        String::from("t1"),
        String::from("c1"),
        Arc::clone(&store) as Arc<dyn TaskStore>,
    );
    sink.task(stored_task("t1", TaskState::Working));
    sink.complete();
    assert!(sink.is_terminal());
    let after_first = drain(&mut receiver).len();

    // A second terminal call (any of them) publishes nothing.
    sink.complete();
    sink.fail("late");
    sink.canceled();
    sink.require_input(None);
    assert_eq!(drain(&mut receiver).len(), 0);
    assert_eq!(store.get("t1").unwrap().status.state, TaskState::Completed);
    assert!(after_first > 0);
}

#[test]
fn fail_publishes_the_failure_text() {
    let store = Arc::new(InMemoryTaskStore::new());
    let (sink, mut receiver) = EventSink::new(
        String::from("t1"),
        String::from("c1"),
        Arc::clone(&store) as Arc<dyn TaskStore>,
    );
    sink.task(stored_task("t1", TaskState::Working));
    sink.fail("executor exploded");

    let events = drain(&mut receiver);
    let last = events.last().unwrap();
    assert_eq!(last["status"]["state"], json!("TASK_STATE_FAILED"));
    assert_eq!(
        last["status"]["message"]["parts"][0]["text"],
        json!("executor exploded")
    );
    assert_eq!(last["status"]["message"]["role"], json!("ROLE_AGENT"));
    assert_eq!(last["status"]["message"]["taskId"], json!("t1"));

    let stored = store.get("t1").unwrap();
    assert_eq!(stored.status.state, TaskState::Failed);
    let message = stored.status.message.unwrap();
    assert_eq!(message.parts, vec![Part::text("executor exploded")]);
}

#[test]
fn config_defaults_match_python() {
    let config = A2aServerConfig::default();
    assert_eq!(config.rpc_path, "/a2a/jsonrpc/");
    assert_eq!(config.card_path, "/.well-known/agent-card.json");
    assert_eq!(config.protocol_version, "1.0");
    assert_eq!(config.host, "127.0.0.1");
    assert_eq!(config.port, 8000);
    assert_eq!(config.max_tasks, DEFAULT_MAX_TASKS);
    assert_eq!(config.terminal_task_ttl_ms, DEFAULT_TERMINAL_TASK_TTL_MS);
    assert_eq!(config.retention_policy(), RetentionPolicy::default());
    assert!(config.interface_url.is_none());

    let config = A2aServerConfig::new(card())
        .with_interface_url("http://127.0.0.1:1/a2a/jsonrpc/")
        .with_bind("0.0.0.0", 0)
        .with_rpc_path("/rpc/");
    assert_eq!(config.agent_card.name, "test-agent");
    assert_eq!(
        config.interface_url.as_deref(),
        Some("http://127.0.0.1:1/a2a/jsonrpc/")
    );
    assert_eq!(config.host, "0.0.0.0");
    assert_eq!(config.port, 0);
    assert_eq!(config.rpc_path, "/rpc/");

    // `0` disables both bounds; explicit values build the matching policy.
    let config = config.with_max_tasks(5).with_terminal_task_ttl_ms(2_000);
    assert_eq!(
        config.retention_policy(),
        RetentionPolicy::new(5, Some(Duration::from_millis(2_000)))
    );
    let config = config.with_max_tasks(0).with_terminal_task_ttl_ms(0);
    assert_eq!(config.retention_policy(), RetentionPolicy::new(0, None));
}

#[test]
fn new_rejects_a_grpc_interface() {
    let mut card = card();
    card.supported_interfaces = vec![AgentInterface {
        url: String::from("http://127.0.0.1:1"),
        protocol_binding: String::from("gRPC"),
        protocol_version: String::from("1.0"),
        tenant: None,
    }];
    let error = A2aServer::new(A2aServerConfig::new(card), Arc::new(EchoExecutor)).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert_eq!(
        error.to_string(),
        "unsupported: gRPC transport is not supported"
    );
}

#[tokio::test]
async fn card_route_serves_the_agent_card() {
    let server = server(Arc::new(EchoExecutor));
    let request = Request::builder()
        .uri("/.well-known/agent-card.json")
        .body(Body::empty())
        .unwrap();
    let response = server.router().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");
    let bytes = http_body_util::BodyExt::collect(response.into_body())
        .await
        .unwrap()
        .to_bytes();
    let card: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(card["name"], json!("test-agent"));
    assert_eq!(card["capabilities"]["streaming"], json!(true));
}

#[tokio::test]
async fn send_message_completes_the_task() {
    let server = server(Arc::new(EchoExecutor));
    let envelope = rpc(
        &server,
        "message/send",
        json!({ "message": { "role": "ROLE_USER", "parts": [{ "text": "hi" }] } }),
    )
    .await;
    let task = result(&envelope);
    assert_eq!(task["status"]["state"], json!("TASK_STATE_COMPLETED"));
    assert_eq!(task["id"].as_str().unwrap().len(), 32);
    assert_eq!(task["history"][0]["parts"][0]["text"], json!("hi"));
    assert_eq!(task["artifacts"].as_array().unwrap().len(), 2);
    // The stored task is the one the result reports.
    let stored = server.store().get(task["id"].as_str().unwrap()).unwrap();
    assert_eq!(stored.status.state, TaskState::Completed);
}

#[tokio::test]
async fn send_message_with_the_v0_3_alias_dispatches() {
    let server = server(Arc::new(EchoExecutor));
    let envelope = rpc(&server, "message/send", json!({ "query": "hello" })).await;
    let task = result(&envelope);
    assert_eq!(task["status"]["state"], json!("TASK_STATE_COMPLETED"));
    assert_eq!(task["history"][0]["parts"][0]["text"], json!("hello"));
    assert_eq!(task["history"][0]["role"], json!("ROLE_USER"));
}

#[tokio::test]
async fn unknown_method_reports_method_not_found() {
    let server = server(Arc::new(EchoExecutor));
    let envelope = rpc(&server, "message/teleport", json!({})).await;
    assert_eq!(
        envelope["error"]["code"],
        json!(error_code::METHOD_NOT_FOUND)
    );
    assert!(envelope["error"]["message"]
        .as_str()
        .unwrap()
        .contains("message/teleport"));
}

#[tokio::test]
async fn get_task_reports_an_unknown_id() {
    let server = server(Arc::new(EchoExecutor));
    let envelope = rpc(&server, "tasks/get", json!({ "id": "nope" })).await;
    assert_eq!(envelope["error"]["code"], json!(error_code::TASK_NOT_FOUND));
    // A missing id is a params problem, not a lookup problem.
    let envelope = rpc(&server, "GetTask", json!({})).await;
    assert_eq!(envelope["error"]["code"], json!(error_code::INVALID_PARAMS));
}

#[tokio::test]
async fn get_task_returns_a_stored_task() {
    let store = Arc::new(InMemoryTaskStore::new());
    store.put(stored_task("t1", TaskState::Working));
    let server = A2aServer::with_store(
        A2aServerConfig::new(card()),
        Arc::new(EchoExecutor),
        Arc::clone(&store) as Arc<dyn TaskStore>,
    )
    .unwrap();
    let envelope = rpc(&server, "tasks/get", json!({ "taskId": "t1" })).await;
    assert_eq!(result(&envelope)["id"], json!("t1"));
    assert_eq!(
        result(&envelope)["status"]["state"],
        json!("TASK_STATE_WORKING")
    );
}

#[tokio::test]
async fn cancel_task_publishes_canceled() {
    let store = Arc::new(InMemoryTaskStore::new());
    store.put(stored_task("t1", TaskState::Working));
    let server = A2aServer::with_store(
        A2aServerConfig::new(card()),
        Arc::new(EchoExecutor),
        Arc::clone(&store) as Arc<dyn TaskStore>,
    )
    .unwrap();

    let envelope = rpc(&server, "tasks/cancel", json!({ "id": "t1" })).await;
    assert_eq!(
        result(&envelope)["status"]["state"],
        json!("TASK_STATE_CANCELED")
    );
    assert_eq!(store.get("t1").unwrap().status.state, TaskState::Canceled);

    // Cancelling a terminal task is refused.
    let envelope = rpc(&server, "CancelTask", json!({ "id": "t1" })).await;
    assert_eq!(
        envelope["error"]["code"],
        json!(error_code::TASK_NOT_CANCELABLE)
    );
}

#[tokio::test]
async fn streaming_message_emits_the_python_frame_order() {
    let server = server(Arc::new(EchoExecutor));
    let body = json!({
        "jsonrpc": "2.0",
        "id": 11,
        "method": "message/stream",
        "params": { "message": { "parts": [{ "text": "hi" }] } },
    })
    .to_string();
    let request = Request::builder()
        .method(Method::POST)
        .uri("/a2a/jsonrpc/")
        .body(Body::from(body))
        .unwrap();
    let response = server.router().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers()[header::CONTENT_TYPE],
        "text/event-stream"
    );
    assert_eq!(response.headers()[header::CACHE_CONTROL], "no-cache");

    let bytes = http_body_util::BodyExt::collect(response.into_body())
        .await
        .unwrap()
        .to_bytes();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let frames: Vec<Value> = text
        .split("\n\n")
        .filter(|frame| !frame.trim().is_empty())
        .map(|frame| {
            let data = frame.strip_prefix("data: ").unwrap();
            serde_json::from_str::<Value>(data).unwrap()
        })
        .collect();

    assert_eq!(frames.len(), 6, "unexpected frames: {frames:?}");
    for frame in &frames {
        assert_eq!(frame["jsonrpc"], json!("2.0"));
        assert_eq!(frame["id"], json!(11));
    }
    let open_id = frames[2]["result"]["artifact"]["artifactId"]
        .as_str()
        .unwrap()
        .to_owned();
    assert_eq!(open_id.len(), 32);
    // 1. Task snapshot.
    assert_eq!(
        frames[0]["result"]["status"]["state"],
        json!("TASK_STATE_SUBMITTED")
    );
    // 2. WORKING.
    assert_eq!(
        frames[1]["result"]["status"]["state"],
        json!("TASK_STATE_WORKING")
    );
    // 3./4. the executor's artifacts.
    assert_eq!(frames[2]["result"]["lastChunk"], json!(false));
    assert_eq!(frames[3]["result"]["artifact"]["artifactId"], json!("a2"));
    assert_eq!(frames[3]["result"]["lastChunk"], json!(true));
    // 5. the open artifact is closed.
    assert_eq!(
        frames[4]["result"]["artifact"]["artifactId"],
        json!(open_id)
    );
    assert_eq!(frames[4]["result"]["append"], json!(true));
    assert_eq!(frames[4]["result"]["artifact"]["parts"], json!([]));
    // 6. COMPLETED.
    assert_eq!(
        frames[5]["result"]["status"]["state"],
        json!("TASK_STATE_COMPLETED")
    );
}

#[tokio::test]
async fn executor_error_publishes_failed_and_a_jsonrpc_error() {
    let store = Arc::new(InMemoryTaskStore::new());
    let server = A2aServer::with_store(
        A2aServerConfig::new(card()),
        Arc::new(FailingExecutor),
        Arc::clone(&store) as Arc<dyn TaskStore>,
    )
    .unwrap();

    let envelope = rpc(&server, "SendMessage", json!({ "query": "hi" })).await;
    assert_eq!(envelope["error"]["code"], json!(error_code::INTERNAL));
    assert!(envelope["error"]["message"]
        .as_str()
        .unwrap()
        .contains("boom"));

    // The task is still stored, in FAILED, with the failure text.
    let tasks = store.all();
    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks[0].status.state, TaskState::Failed);
    let message = tasks[0].status.message.clone().unwrap();
    assert_eq!(message.parts, vec![Part::text("validation: boom")]);
}

#[tokio::test]
async fn subscribe_to_task_answers_with_a_snapshot() {
    let store = Arc::new(InMemoryTaskStore::new());
    store.put(stored_task("t1", TaskState::Working));
    let server = A2aServer::with_store(
        A2aServerConfig::new(card()),
        Arc::new(EchoExecutor),
        Arc::clone(&store) as Arc<dyn TaskStore>,
    )
    .unwrap();

    let body = json!({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tasks/resubscribe",
        "params": { "id": "t1" },
    })
    .to_string();
    let request = Request::builder()
        .method(Method::POST)
        .uri("/a2a/jsonrpc/")
        .body(Body::from(body))
        .unwrap();
    let response = server.router().oneshot(request).await.unwrap();
    assert_eq!(
        response.headers()[header::CONTENT_TYPE],
        "text/event-stream"
    );
    let bytes = http_body_util::BodyExt::collect(response.into_body())
        .await
        .unwrap()
        .to_bytes();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let frames: Vec<Value> = text
        .split("\n\n")
        .filter(|frame| !frame.trim().is_empty())
        .map(|frame| serde_json::from_str::<Value>(frame.strip_prefix("data: ").unwrap()).unwrap())
        .collect();
    assert_eq!(frames.len(), 2, "unexpected frames: {frames:?}");
    assert_eq!(frames[0]["result"]["id"], json!("t1"));
    assert_eq!(
        frames[1]["result"]["status"]["state"],
        json!("TASK_STATE_WORKING")
    );

    // An unknown id yields one error frame and ends.
    let body = json!({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "SubscribeToTask",
        "params": { "id": "missing" },
    })
    .to_string();
    let request = Request::builder()
        .method(Method::POST)
        .uri("/a2a/jsonrpc/")
        .body(Body::from(body))
        .unwrap();
    let response = server.router().oneshot(request).await.unwrap();
    let bytes = http_body_util::BodyExt::collect(response.into_body())
        .await
        .unwrap()
        .to_bytes();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let frame: Value = serde_json::from_str(text.trim().strip_prefix("data: ").unwrap()).unwrap();
    assert_eq!(frame["error"]["code"], json!(error_code::TASK_NOT_FOUND));
}

#[tokio::test]
async fn adapter_starts_and_stops_on_an_ephemeral_port() {
    let config = A2aServerConfig::new(card()).with_bind("127.0.0.1", 0);
    let server = A2aServer::new(config, Arc::new(EchoExecutor)).unwrap();
    let mut adapter = A2aServerAdapter::new(server);
    assert!(!adapter.active());
    assert!(adapter.local_addr().is_none());

    adapter.start().await.unwrap();
    assert!(adapter.active());
    let address = adapter.local_addr().unwrap();
    assert_eq!(address.ip().to_string(), "127.0.0.1");
    assert_ne!(address.port(), 0);

    // Starting twice keeps the first server.
    adapter.start().await.unwrap();
    assert_eq!(adapter.local_addr(), Some(address));

    adapter.stop().await.unwrap();
    assert!(!adapter.active());
    assert!(adapter.local_addr().is_none());
    // Stopping twice is a no-op.
    adapter.stop().await.unwrap();
}

#[test]
fn task_store_evicts_the_oldest_terminal_task_first() {
    let store = InMemoryTaskStore::with_policy(RetentionPolicy::new(3, None));
    store.put(stored_task("run", TaskState::Working));
    store.put(stored_task("old", TaskState::Completed));
    std::thread::sleep(Duration::from_millis(2));
    store.put(stored_task("new", TaskState::Completed));
    std::thread::sleep(Duration::from_millis(2));
    // The fourth task pushes the store over its cap: only the oldest terminal
    // task may go.
    store.put(stored_task("newest", TaskState::Completed));

    let ids: Vec<String> = store.all().into_iter().map(|task| task.id).collect();
    assert_eq!(ids.len(), 3, "unexpected tasks: {ids:?}");
    assert!(
        ids.contains(&String::from("run")),
        "running task was evicted"
    );
    assert!(
        ids.contains(&String::from("new")),
        "newer terminal task was evicted"
    );
    assert!(ids.contains(&String::from("newest")));
    assert!(
        !ids.contains(&String::from("old")),
        "the oldest terminal task must be evicted first: {ids:?}"
    );
}

#[test]
fn task_store_caps_retained_tasks() {
    let store = InMemoryTaskStore::with_policy(RetentionPolicy::new(4, None));
    for index in 0..20 {
        store.put(stored_task(&format!("t{index:02}"), TaskState::Completed));
    }
    assert!(
        store.all().len() <= 4,
        "the store exceeded its cap: {} tasks",
        store.all().len()
    );
}

#[test]
fn task_store_never_evicts_a_running_or_awaited_task() {
    let store = InMemoryTaskStore::with_policy(RetentionPolicy::new(1, None));
    store.put(stored_task("run", TaskState::Working));
    store.put(stored_task("wait", TaskState::InputRequired));
    for index in 0..10 {
        store.put(stored_task(&format!("done{index}"), TaskState::Completed));
    }

    let ids: Vec<String> = store.all().into_iter().map(|task| task.id).collect();
    assert!(
        ids.contains(&String::from("run")),
        "running task was evicted"
    );
    assert!(
        ids.contains(&String::from("wait")),
        "awaited task was evicted"
    );
    assert!(
        ids.iter().all(|id| !id.starts_with("done")),
        "terminal tasks must be the only eviction candidates: {ids:?}"
    );
}

#[test]
fn task_store_keeps_a_completed_task_readable_until_the_ttl_expires() {
    let ttl = Duration::from_millis(30);
    let store = InMemoryTaskStore::with_policy(RetentionPolicy::new(100, Some(ttl)));
    store.put(stored_task("done", TaskState::Completed));
    assert!(
        store.get("done").is_some(),
        "a completed task must stay readable within its TTL"
    );

    std::thread::sleep(Duration::from_millis(60));
    assert!(
        store.get("done").is_none(),
        "an expired terminal task must be pruned"
    );

    // Awaited and running tasks are not subject to the terminal TTL.
    store.put(stored_task("wait", TaskState::InputRequired));
    store.put(stored_task("run", TaskState::Working));
    std::thread::sleep(Duration::from_millis(60));
    assert!(store.get("wait").is_some(), "awaited task expired");
    assert!(store.get("run").is_some(), "running task expired");
}

#[tokio::test]
async fn server_config_bounds_the_default_task_store() {
    let config = A2aServerConfig::new(card()).with_max_tasks(3);
    let server = A2aServer::new(config, Arc::new(EchoExecutor)).unwrap();
    for index in 0..12 {
        let envelope = rpc(
            &server,
            "SendMessage",
            json!({ "query": format!("request-{index}") }),
        )
        .await;
        assert!(
            envelope.get("error").is_none(),
            "unexpected error: {envelope}"
        );
    }
    assert!(
        server.store().all().len() <= 3,
        "the configured cap must bound the server's store"
    );
}

#[tokio::test]
async fn a_subscriber_that_never_reads_is_dropped_without_blocking_the_publisher() {
    let store = Arc::new(InMemoryTaskStore::new());
    let (sink, mut stream) = EventSink::new(
        String::from("t1"),
        String::from("c1"),
        Arc::clone(&store) as Arc<dyn TaskStore>,
    );
    sink.task(stored_task("t1", TaskState::Working));
    for index in 0..(EVENT_BUFFER_CAPACITY + 25) {
        sink.artifact(
            Artifact {
                artifact_id: format!("a{index}"),
                parts: vec![Part::text("chunk")],
                ..Artifact::default()
            },
            true,
        );
    }

    // The publisher ran to completion without a reader: it was never blocked,
    // and the buffer stayed bounded.
    assert!(
        sink.subscriber_overflowed(),
        "the overflow must be surfaced"
    );
    let buffered = drain(&mut stream);
    assert!(
        buffered.len() <= EVENT_BUFFER_CAPACITY,
        "the buffer grew past the channel capacity: {} events",
        buffered.len()
    );
    assert!(
        stream.overflowed(),
        "the subscriber must observe the terminal overflow state"
    );

    // The task itself keeps running and the store stays coherent.
    sink.complete();
    assert_eq!(store.get("t1").unwrap().status.state, TaskState::Completed);
}

#[tokio::test]
async fn a_healthy_subscriber_receives_every_event_in_order() {
    let store = Arc::new(InMemoryTaskStore::new());
    let (sink, mut stream) = EventSink::new(
        String::from("t1"),
        String::from("c1"),
        Arc::clone(&store) as Arc<dyn TaskStore>,
    );
    let total = EVENT_BUFFER_CAPACITY * 2;
    let producer = tokio::spawn(async move {
        sink.task(stored_task("t1", TaskState::Working));
        for index in 0..total {
            sink.artifact(
                Artifact {
                    artifact_id: format!("a{index:04}"),
                    parts: vec![Part::text(format!("{index}"))],
                    ..Artifact::default()
                },
                true,
            );
            tokio::task::yield_now().await;
        }
        sink.complete();
    });

    let mut ids = Vec::new();
    tokio::time::timeout(Duration::from_secs(10), async {
        while let Some(value) = stream.recv().await {
            if let Some(id) = value
                .get("artifact")
                .and_then(|artifact| artifact.get("artifactId"))
                .and_then(Value::as_str)
            {
                ids.push(id.to_owned());
            }
        }
    })
    .await
    .expect("a healthy subscriber must not stall");
    producer.await.expect("the producer must finish");

    assert!(
        !stream.overflowed(),
        "a healthy subscriber must not overflow"
    );
    assert_eq!(ids.len(), total, "every event must be delivered");
    for (index, id) in ids.iter().enumerate() {
        assert_eq!(id, &format!("a{index:04}"), "events arrived out of order");
    }
}
