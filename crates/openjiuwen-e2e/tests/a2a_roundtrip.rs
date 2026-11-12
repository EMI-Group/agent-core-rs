// `missing_docs` is allowed before the `cfg` gate: when the `a2a` feature is off
// the whole crate (including the `//!` docs) is stripped, and a later `allow`
// would be stripped with it, leaving the denied `missing_docs` lint to fire on
// the empty crate root.
#![allow(missing_docs)]
#![cfg(feature = "a2a")]
#![allow(clippy::unwrap_used, clippy::expect_used)]
//! A2A protocol round trip: a real `A2aClient` talks to a real `A2aServer` over
//! a loopback socket bound to an ephemeral port.
//!
//! Five seams are exercised end to end:
//!
//! 1. **Card fetch** — the agent card is served over HTTP, decoded back into an
//!    [`AgentCard`], and checked for the advertised name, streaming capability,
//!    and input modes.
//! 2. **`SendMessage`** — a non-streaming turn folds the stored task into an
//!    [`AgentResult`]; the server-side [`TaskStore`] is inspected directly to
//!    prove the same task, artifact, and request history were persisted.
//! 3. **Streaming frames** — `SendStreamingMessage` yields the deterministic
//!    sequence SUBMITTED snapshot → WORKING prelude → artifact chunk →
//!    COMPLETED terminal, all carrying the same task id.
//! 4. **Cancel** — a turn paused in `INPUT_REQUIRED` is cancelled and the stored
//!    task records CANCELED.
//! 5. **Validation** — a non-object payload is rejected by the client before any
//!    HTTP request is made.
//!
//! Everything is offline and deterministic: one loopback socket, no credentials,
//! no sleeps, and no assertion on generated task/message ids beyond the values
//! the server returned. Every network await is bounded by [`support::within`].
mod support;

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen::core::common::error::{Error, Result};
use openjiuwen::extensions::a2a::client::{A2aClient, A2aClientConfig};
use openjiuwen::extensions::a2a::server::{
    A2aExecutionContext, A2aExecutor, A2aServer, A2aServerConfig, EventSink, ServerHandle,
    TaskStore, DEFAULT_CARD_PATH, DEFAULT_RPC_PATH,
};
use openjiuwen::extensions::a2a::transform::{AgentResult, AgentStatus};
use openjiuwen::extensions::a2a::types::{
    AgentCapabilities, AgentCard, AgentInterface, Artifact, Part, TaskState,
};
use serde_json::json;
use tokio_util::sync::CancellationToken;

/// The card the server advertises.
///
/// Exactly one JSON-RPC interface and no gRPC one: `A2aServer::new` rejects a
/// card that advertises a gRPC transport, and both round-trip scenarios post to
/// the JSON-RPC mount path.
fn card() -> AgentCard {
    AgentCard {
        name: "e2e-a2a-agent".to_owned(),
        description: "A2A round-trip fixture".to_owned(),
        version: "1.0".to_owned(),
        supported_interfaces: vec![AgentInterface {
            url: format!("http://127.0.0.1:0{DEFAULT_RPC_PATH}"),
            protocol_binding: "JSONRPC".to_owned(),
            ..AgentInterface::default()
        }],
        capabilities: AgentCapabilities {
            streaming: true,
            push_notifications: false,
        },
        default_input_modes: vec!["text/plain".to_owned()],
        default_output_modes: vec!["text/plain".to_owned()],
        ..AgentCard::default()
    }
}

/// A running server plus the pieces a scenario asserts against.
struct Started {
    handle: ServerHandle,
    store: Arc<dyn TaskStore>,
    base_url: String,
    rpc_url: String,
}

impl Started {
    /// Stops the server and waits for its serving task to finish.
    async fn shutdown(self) {
        self.handle.shutdown();
        self.handle.join().await.expect("the server stops cleanly");
    }
}

/// Builds a server for `executor` on an ephemeral loopback port and starts it.
async fn start(executor: Arc<dyn A2aExecutor>) -> Started {
    let config = A2aServerConfig::new(card()).with_bind("127.0.0.1", 0);
    let server = A2aServer::new(config, executor).expect("the card advertises only JSON-RPC");
    // Capture the store first: `serve_in_background` consumes the server.
    let store = server.store();
    let handle = support::within(
        "A2A server bind",
        server.serve_in_background(CancellationToken::new()),
    )
    .await
    .expect("the server binds an ephemeral loopback port");
    let base_url = format!("http://{}", handle.local_addr());
    let rpc_url = format!("{base_url}{DEFAULT_RPC_PATH}");
    Started {
        handle,
        store,
        base_url,
        rpc_url,
    }
}

/// A client posting to `rpc_url` with a short per-request budget.
fn client(rpc_url: &str) -> A2aClient {
    A2aClient::new(A2aClientConfig::new(rpc_url).with_timeout(Duration::from_secs(2)))
        .expect("the client builds for a non-empty url")
}

/// Publishes exactly one complete artifact and leaves the sink open.
///
/// The server's own COMPLETED is the terminal status the scenarios assert, so
/// the executor deliberately does not publish a terminal status itself.
struct RoundTripExecutor;

#[async_trait]
impl A2aExecutor for RoundTripExecutor {
    async fn execute(&self, _ctx: A2aExecutionContext, sink: &EventSink) -> Result<()> {
        sink.artifact(
            Artifact {
                artifact_id: "a1".to_owned(),
                parts: vec![Part::text("hello")],
                ..Artifact::default()
            },
            true,
        );
        Ok(())
    }
}

/// Leaves the turn in `INPUT_REQUIRED` and returns.
///
/// The server auto-completes any turn whose sink was left open, so
/// `INPUT_REQUIRED` is the only durable non-terminal state reachable through a
/// real `SendMessage`. The server does not treat `TaskState::InputRequired` as
/// terminal, which is what lets `CancelTask` act on it.
struct PausingExecutor;

#[async_trait]
impl A2aExecutor for PausingExecutor {
    async fn execute(&self, _ctx: A2aExecutionContext, sink: &EventSink) -> Result<()> {
        sink.require_input(None);
        Ok(())
    }
}

#[tokio::test]
async fn card_and_send_round_trip_over_loopback() {
    let started = start(Arc::new(RoundTripExecutor)).await;

    let response = support::within(
        "A2A card GET",
        reqwest::get(format!("{}{DEFAULT_CARD_PATH}", started.base_url)),
    )
    .await
    .expect("the card route answers");
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let fetched: AgentCard = support::within("A2A card body", response.json())
        .await
        .expect("the card body decodes");
    assert_eq!(fetched.name, "e2e-a2a-agent");
    assert!(
        fetched.capabilities.streaming,
        "the card advertises streaming"
    );
    assert!(
        fetched
            .default_input_modes
            .iter()
            .any(|mode| mode == "text/plain"),
        "the card advertises the text/plain input mode: {fetched:?}"
    );

    let caller = client(&started.rpc_url);
    let result = support::within("A2A SendMessage", caller.invoke(json!({"query": "hello"})))
        .await
        .expect("the send succeeds");
    assert_eq!(result.status, AgentStatus::Completed);
    let task_id = result
        .task_id
        .clone()
        .expect("the reply carries the task id");
    assert_eq!(result.artifacts.len(), 1);
    assert_eq!(result.artifacts[0].artifact_id.as_deref(), Some("a1"));
    assert_eq!(result.artifacts[0].parts[0].text.as_deref(), Some("hello"));

    assert_eq!(started.store.all().len(), 1, "one task was stored");
    let stored = started
        .store
        .get(&task_id)
        .expect("the returned task id is stored");
    assert_eq!(stored.id, task_id);
    assert_eq!(stored.status.state, TaskState::Completed);
    assert_eq!(stored.artifacts.len(), 1);
    assert_eq!(stored.artifacts[0].artifact_id, "a1");
    assert_eq!(stored.artifacts[0].parts[0].text.as_deref(), Some("hello"));
    assert_eq!(
        stored.history[0].parts[0].text.as_deref(),
        Some("hello"),
        "the request message is the task history's first entry"
    );

    caller.stop().await;
    started.shutdown().await;
}

#[tokio::test]
async fn streaming_send_yields_the_deterministic_frame_sequence() {
    let started = start(Arc::new(RoundTripExecutor)).await;

    let raw: Vec<Result<AgentResult>> = support::collect_bounded(
        client(&started.rpc_url).stream(json!({"query": "hello"})),
        4,
    )
    .await;
    let results: Vec<AgentResult> = raw
        .into_iter()
        .map(|frame| frame.expect("the stream yields a decoded frame"))
        .collect();
    assert_eq!(results.len(), 4, "the turn publishes exactly four frames");

    let statuses: Vec<AgentStatus> = results.iter().map(|frame| frame.status).collect();
    assert_eq!(
        statuses,
        vec![
            AgentStatus::Submitted,
            AgentStatus::Working,
            AgentStatus::Working,
            AgentStatus::Completed,
        ],
        "SUBMITTED snapshot, WORKING prelude, artifact chunk, COMPLETED terminal"
    );

    let task_id = results[0]
        .task_id
        .clone()
        .expect("every frame carries the task id");
    assert!(
        results
            .iter()
            .all(|frame| frame.task_id.as_deref() == Some(task_id.as_str())),
        "status updates and artifacts carry the same task id: {results:?}"
    );

    let artifact_frame = &results[2];
    assert_eq!(artifact_frame.artifacts.len(), 1);
    assert_eq!(
        artifact_frame.artifacts[0].artifact_id.as_deref(),
        Some("a1")
    );
    assert_eq!(
        artifact_frame.artifacts[0].parts[0].text.as_deref(),
        Some("hello")
    );

    let terminal = results.last().expect("there is a terminal frame");
    assert_eq!(terminal.status, AgentStatus::Completed);
    assert_eq!(terminal.task_id.as_deref(), Some(task_id.as_str()));

    started.shutdown().await;
}

#[tokio::test]
async fn cancel_task_marks_the_task_canceled() {
    let started = start(Arc::new(PausingExecutor)).await;
    let caller = client(&started.rpc_url);

    let paused = support::within(
        "A2A SendMessage (pause)",
        caller.invoke(json!({"query": "please wait"})),
    )
    .await
    .expect("the pausing send succeeds");
    assert_eq!(paused.status, AgentStatus::InputRequired);
    let task_id = paused
        .task_id
        .clone()
        .expect("the reply carries the task id");

    let canceled = support::within("A2A CancelTask", caller.cancel_task(&task_id, None))
        .await
        .expect("the paused task is cancellable");
    assert_eq!(canceled.status, AgentStatus::Canceled);
    assert_eq!(canceled.task_id.as_deref(), Some(task_id.as_str()));

    let stored = started
        .store
        .get(&task_id)
        .expect("the task is still stored");
    assert_eq!(stored.status.state, TaskState::Canceled);

    caller.stop().await;
    started.shutdown().await;
}

#[tokio::test]
async fn non_object_inputs_are_rejected() {
    // No server: the payload is rejected while the request is being built.
    let caller = client("http://127.0.0.1:1/a2a/jsonrpc/");
    let error = caller
        .invoke(json!("plain string"))
        .await
        .expect_err("a non-object payload is rejected before any HTTP call");
    assert!(matches!(error, Error::Validation(_)), "got {error:?}");
    caller.stop().await;
}
