//! Behaviour suite for the HTTP remote-agent client of the spawn seam.
//!
//! Drives `openjiuwen_runner::spawn::remote` end to end against a loopback
//! `wiremock` server: the invoke/stream/cancel wire, the error mapping, the
//! deferred protocol variants, and the `RemoteSpawner` adapter.
//! Ports `tests/unit_tests/core/runner/drunner/test_remote_agent.py`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::Duration;

use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_agent::output::{AgentChunk, ResultType};
use openjiuwen_common::error::{ErrorKind, Result};
use openjiuwen_runner::spawn::remote::{
    create_remote_client, RemoteAgent, RemoteAgentConfig, RemoteClient, RemoteProtocol,
    RemoteSpawner,
};
use openjiuwen_runner::spawn::{AgentSpawner, SpawnMessage, SpawnMessageKind, SpawnSpec};
use serde_json::{json, Value};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

/// Deadline for collecting a stream: only proves termination, never ordering.
const BOUND: Duration = Duration::from_secs(5);

/// A config for `agent_id` served by `server`, with the default paths.
fn config_for(server: &MockServer, agent_id: &str) -> RemoteAgentConfig {
    RemoteAgentConfig::new(agent_id, server.uri())
}

/// `frame` as one NDJSON line, without the trailing newline.
fn frame_line(frame: &SpawnMessage) -> String {
    frame.to_line().unwrap().trim_end().to_owned()
}

/// A `stream_chunk` line carrying one `llm_output` frame.
fn chunk_line(index: i64, text: &str) -> String {
    frame_line(&SpawnMessage::new(
        SpawnMessageKind::StreamChunk,
        serde_json::to_value(AgentChunk::llm_output(index, text)).unwrap(),
    ))
}

/// The JSON body the tested client sent.
fn body_json(request: &Request) -> Value {
    serde_json::from_slice(&request.body).expect("a remote-agent request body is JSON")
}

/// The first value of header `name` in `request`.
fn header_value(request: &Request, name: &str) -> Option<String> {
    request
        .headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
}

/// The requests `server` recorded whose path `keep` accepts, in arrival order.
///
/// `wiremock` hands its servers out of a process-wide pool, so a recording is
/// not guaranteed to hold only this test's traffic: dropping a `MockServer`
/// returns its listener to the pool (its recordings are reset for the next
/// owner), and a request the previous owner left in flight can still be served —
/// and therefore recorded — after that reset. Every assertion about what this
/// client sent filters to the paths this test actually owns and never counts
/// foreign entries.
async fn recorded(server: &MockServer, keep: impl Fn(&str) -> bool) -> Vec<Request> {
    server
        .received_requests()
        .await
        .expect("a mock server records its requests")
        .into_iter()
        .filter(|request| keep(request.url.path()))
        .collect()
}

/// Collects a whole stream under [`BOUND`], asserting it ends on its own.
async fn collect_stream(stream: BoxStream<'static, Result<AgentChunk>>) -> Vec<Result<AgentChunk>> {
    tokio::time::timeout(BOUND, stream.collect::<Vec<_>>())
        .await
        .expect("a remote stream must end on its own")
}

/// Mounts a 200 `invoke` answer for `agent_id`.
async fn mount_invoke(server: &MockServer, agent_id: &str, answer: &str) {
    Mock::given(method("POST"))
        .and(path(format!("/agents/{agent_id}/invoke")))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "output": answer,
            "result_type": "answer",
        })))
        .mount(server)
        .await;
}

/// Mounts a 200 NDJSON stream body for `agent_id`.
async fn mount_stream_body(server: &MockServer, agent_id: &str, body: String) {
    Mock::given(method("POST"))
        .and(path(format!("/agents/{agent_id}/stream")))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/x-ndjson")
                .set_body_string(body),
        )
        .mount(server)
        .await;
}

#[tokio::test]
async fn invoke_returns_the_remote_output_and_sends_session_and_inputs() {
    let server = MockServer::start().await;
    mount_invoke(&server, "echo", "hi").await;

    let plain = RemoteAgent::new(config_for(&server, "echo"));
    let output = plain.invoke(json!({"query": "hello"}), None).await.unwrap();
    assert_eq!(output.output, "hi");
    assert_eq!(output.result_type, ResultType::Answer);

    let bound = RemoteAgent::new(config_for(&server, "echo").with_session_id("session-7"));
    bound.invoke(json!({"query": "again"}), None).await.unwrap();

    let received = recorded(&server, |path| path == "/agents/echo/invoke").await;
    assert_eq!(received.len(), 2);
    assert_eq!(received[0].method.as_str(), "POST");
    assert_eq!(received[0].url.path(), "/agents/echo/invoke");
    assert_eq!(
        body_json(&received[0]),
        json!({"session_id": null, "inputs": {"query": "hello"}}),
        "an unbound agent sends a null session_id"
    );
    assert_eq!(
        body_json(&received[1]),
        json!({"session_id": "session-7", "inputs": {"query": "again"}})
    );
}

#[tokio::test]
async fn invoke_maps_an_unmounted_route_to_not_found() {
    let server = MockServer::start().await;
    let agent = RemoteAgent::new(config_for(&server, "ghost"));

    let error = agent
        .invoke(json!({"query": "hi"}), None)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(
        error.to_string().contains("ghost"),
        "unexpected message: {error}"
    );
}

#[tokio::test]
async fn invoke_maps_a_server_error_to_transport_with_the_status() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/agents/echo/invoke"))
        .respond_with(ResponseTemplate::new(500).set_body_string("upstream exploded"))
        .mount(&server)
        .await;

    let agent = RemoteAgent::new(config_for(&server, "echo"));
    let error = agent
        .invoke(json!({"query": "hi"}), None)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Transport);
    assert!(
        error.to_string().contains("500"),
        "the status must be reported: {error}"
    );
    assert!(
        error.to_string().contains("upstream exploded"),
        "the body snippet must be reported: {error}"
    );
}

#[tokio::test]
async fn invoke_maps_an_expired_deadline_to_timeout() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/agents/echo/invoke"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_secs(2))
                .set_body_json(json!({"output": "late", "result_type": "answer"})),
        )
        .mount(&server)
        .await;

    let agent =
        RemoteAgent::new(config_for(&server, "echo").with_timeout(Duration::from_millis(200)));
    let error = agent
        .invoke(json!({"query": "hi"}), None)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Timeout);
}

#[tokio::test]
async fn stream_yields_each_chunk_frame_and_ends_on_done() {
    let server = MockServer::start().await;
    let body = [
        ": keep-alive".to_owned(),
        chunk_line(0, "he"),
        String::new(),
        chunk_line(1, "llo"),
        frame_line(&SpawnMessage::new(SpawnMessageKind::Done, Value::Null)),
    ]
    .join("\n");
    mount_stream_body(&server, "echo", format!("{body}\n")).await;

    let agent = RemoteAgent::new(config_for(&server, "echo"));
    let frames = collect_stream(RemoteClient::stream(&agent, json!({"query": "hi"}), None)).await;
    assert_eq!(frames.len(), 2, "noise and `done` yield no chunk item");
    let chunks: Vec<AgentChunk> = frames.into_iter().map(|frame| frame.unwrap()).collect();
    assert_eq!(chunks[0].kind, "llm_output");
    assert_eq!(chunks[0].index, 0);
    assert_eq!(chunks[0].payload, json!({"content": "he"}));
    assert_eq!(chunks[1].kind, "llm_output");
    assert_eq!(chunks[1].index, 1);
    assert_eq!(chunks[1].payload, json!({"content": "llo"}));
}

#[tokio::test]
async fn stream_error_frame_is_the_only_item_then_ends() {
    let server = MockServer::start().await;
    let line = frame_line(&SpawnMessage::new(
        SpawnMessageKind::Error,
        json!({"message": "remote blew up"}),
    ));
    mount_stream_body(&server, "echo", format!("{line}\n")).await;

    let agent = RemoteAgent::new(config_for(&server, "echo"));
    let frames = collect_stream(RemoteClient::stream(&agent, json!({"query": "hi"}), None)).await;
    assert_eq!(frames.len(), 1);
    let error = frames[0].as_ref().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Transport);
    assert!(
        error.to_string().contains("remote blew up"),
        "unexpected message: {error}"
    );
}

#[tokio::test]
async fn stream_non_success_status_is_a_single_error_frame() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/agents/echo/stream"))
        .respond_with(ResponseTemplate::new(503).set_body_string("no capacity"))
        .mount(&server)
        .await;

    let agent = RemoteAgent::new(config_for(&server, "echo"));
    let frames = collect_stream(RemoteClient::stream(&agent, json!({"query": "hi"}), None)).await;
    assert_eq!(frames.len(), 1);
    let error = frames[0].as_ref().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Transport);
    assert!(
        error.to_string().contains("503"),
        "the status must be reported: {error}"
    );
}

#[tokio::test]
async fn deferred_protocols_are_typed_unsupported_and_http_works() {
    let server = MockServer::start().await;
    mount_invoke(&server, "echo", "hi").await;
    let config = config_for(&server, "echo");

    let mq = create_remote_client(RemoteProtocol::Mq, config.clone())
        .err()
        .expect("the MQ transport is deferred");
    assert_eq!(mq.kind(), ErrorKind::Unsupported);
    assert!(
        mq.to_string().contains("Pulsar") && mq.to_string().contains("ZeroMQ"),
        "the MQ deferral must name its transport: {mq}"
    );

    let a2a = create_remote_client(RemoteProtocol::A2a, config.clone())
        .err()
        .expect("the A2A transport is deferred");
    assert_eq!(a2a.kind(), ErrorKind::Unsupported);
    assert!(
        a2a.to_string().contains("A2A"),
        "the A2A deferral must name its transport: {a2a}"
    );

    let http = create_remote_client(RemoteProtocol::Http, config).unwrap();
    assert!(!http.is_started());
    let output = http.invoke(json!({"query": "hi"}), None).await.unwrap();
    assert_eq!(output.output, "hi");
    assert!(http.is_started());
}

#[tokio::test]
async fn remote_spawner_builds_a_working_spawned_agent() {
    let server = MockServer::start().await;
    mount_invoke(&server, "echo", "spawned").await;
    mount_stream_body(&server, "echo", format!("{}\n", chunk_line(0, "one"))).await;

    let spawner = RemoteSpawner::new(server.uri());
    assert_eq!(spawner.base_url(), server.uri());
    assert_eq!(spawner.config_for("other").agent_id, "other");

    let handle = spawner
        .spawn(SpawnSpec::class_agent("pkg.agents", "echo").with_session_id("sess-9"))
        .await
        .unwrap();
    assert_eq!(handle.id(), "echo");
    assert_eq!(handle.pid(), None, "a remote agent has no OS process");

    let output = handle.send(json!({"query": "hi"})).await.unwrap();
    assert_eq!(output.output, "spawned");
    let bodies = recorded(&server, |path| path == "/agents/echo/invoke").await;
    assert_eq!(bodies.len(), 1, "only the invocation reached the server");
    assert_eq!(
        body_json(&bodies[0]),
        json!({"session_id": "sess-9", "inputs": {"query": "hi"}}),
        "the spawn spec's session reaches the remote agent"
    );

    let frames = collect_stream(handle.stream(json!({"query": "hi"}))).await;
    assert_eq!(frames.len(), 1);
    assert_eq!(
        frames[0].as_ref().unwrap().payload,
        json!({"content": "one"})
    );

    assert!(handle.shutdown(Duration::from_secs(1)).await.unwrap());
    assert!(!handle.is_alive());
}

#[tokio::test]
async fn invoke_auto_starts_and_stop_is_idempotent() {
    let server = MockServer::start().await;
    mount_invoke(&server, "echo", "hi").await;

    let agent = RemoteAgent::new(config_for(&server, "echo"));
    assert!(!agent.is_started());
    assert!(agent.is_stopped());

    agent.invoke(json!({"query": "hi"}), None).await.unwrap();
    assert!(agent.is_started(), "an invocation starts the client");
    assert!(!agent.is_stopped());

    agent.stop().await.unwrap();
    assert!(!agent.is_started());
    agent.stop().await.unwrap();
    assert!(agent.is_stopped());
}

#[tokio::test]
async fn cancel_task_posts_the_task_id_and_needs_a_cancel_path() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/agents/echo/cancel"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;

    let agent = RemoteAgent::new(config_for(&server, "echo"));
    agent.cancel_task("task-1").await.unwrap();
    let received = recorded(&server, |path| path == "/agents/echo/cancel").await;
    assert_eq!(received.len(), 1);
    assert_eq!(received[0].url.path(), "/agents/echo/cancel");
    assert_eq!(body_json(&received[0]), json!({"task_id": "task-1"}));

    let disabled = RemoteAgent::new(config_for(&server, "echo").with_cancel_path(None));
    let error = disabled.cancel_task("task-2").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(
        error.to_string().contains("cancel_path"),
        "unexpected message: {error}"
    );
}

#[tokio::test]
async fn configured_headers_reach_the_server() {
    let server = MockServer::start().await;
    mount_invoke(&server, "echo", "hi").await;

    let agent = RemoteAgent::new(
        config_for(&server, "echo")
            .with_header("x-api-key", "secret")
            .with_header("x-trace-id", "trace-1"),
    );
    agent.invoke(json!({"query": "hi"}), None).await.unwrap();

    let received = recorded(&server, |path| path == "/agents/echo/invoke").await;
    assert_eq!(received.len(), 1, "only the invocation reached the server");
    assert_eq!(
        header_value(&received[0], "x-api-key").as_deref(),
        Some("secret")
    );
    assert_eq!(
        header_value(&received[0], "x-trace-id").as_deref(),
        Some("trace-1")
    );
}

#[tokio::test]
async fn custom_paths_and_a_trailing_slash_base_url_hit_the_expected_routes() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v2/agents/echo/run"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"output": "custom", "result_type": "answer"})),
        )
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v2/agents/echo/events"))
        .respond_with(
            ResponseTemplate::new(200).set_body_string(format!("{}\n", chunk_line(0, "e"))),
        )
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v2/agents/echo/stop"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;

    let config = RemoteAgentConfig::new("echo", format!("{}/", server.uri()))
        .with_paths("/v2/agents/{agent_id}/run", "/v2/agents/{agent_id}/events")
        .with_cancel_path(Some("/v2/agents/{agent_id}/stop".to_owned()));
    let agent = RemoteAgent::new(config);

    assert_eq!(
        agent
            .invoke(json!({"query": "hi"}), None)
            .await
            .unwrap()
            .output,
        "custom"
    );
    let frames = collect_stream(RemoteClient::stream(&agent, json!({"query": "hi"}), None)).await;
    assert_eq!(frames.len(), 1);
    agent.cancel_task("task-1").await.unwrap();

    // The custom prefix is what this test owns, so keep only requests under it:
    // the mock server's pool can hand a listener to us that still receives a
    // request from a test that borrowed it before us (see [`recorded`]).
    let paths: Vec<String> = recorded(&server, |path| path.starts_with("/v2/"))
        .await
        .iter()
        .map(|request| request.url.path().to_owned())
        .collect();
    assert_eq!(
        paths,
        vec![
            "/v2/agents/echo/run".to_owned(),
            "/v2/agents/echo/events".to_owned(),
            "/v2/agents/echo/stop".to_owned(),
        ]
    );
}

#[tokio::test]
async fn an_unreachable_endpoint_is_a_transport_error_not_a_panic() {
    // A deterministic unreachable endpoint: port 1 is privileged, so nothing in
    // this process can listen there and `wiremock` never hands it out of its
    // server pool. A bind-then-release ephemeral port would be racy — the freed
    // port can be re-bound by another test's mock server before this connect,
    // which turns the expected transport failure into a `404`.
    let agent = RemoteAgent::new(
        RemoteAgentConfig::new("echo", "http://127.0.0.1:1")
            .with_timeout(Duration::from_millis(500)),
    );

    let error = agent
        .invoke(json!({"query": "hi"}), None)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Transport);

    let frames = collect_stream(RemoteClient::stream(&agent, json!({"query": "hi"}), None)).await;
    assert_eq!(frames.len(), 1, "a connect failure ends the stream");
    assert_eq!(frames[0].as_ref().unwrap_err().kind(), ErrorKind::Transport);
}
