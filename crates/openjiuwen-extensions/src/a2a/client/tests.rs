#![allow(clippy::unwrap_used, clippy::expect_used)]

use super::*;
use crate::a2a::transform::AgentStatus;
use crate::a2a::types::AgentInterface;
use openjiuwen_common::error::ErrorKind;
use serde_json::json;
use wiremock::matchers::{header, method as http_method};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// One card interface entry.
fn interface(url: &str, binding: &str) -> AgentInterface {
    AgentInterface {
        url: String::from(url),
        protocol_binding: String::from(binding),
        protocol_version: String::from("1.0"),
        tenant: None,
    }
}

/// A card advertising `interfaces`.
fn card_with(interfaces: Vec<AgentInterface>) -> AgentCard {
    AgentCard {
        supported_interfaces: interfaces,
        ..AgentCard::default()
    }
}

/// No SSE frames, for the "nothing was emitted yet" assertions.
fn no_frames() -> Vec<String> {
    Vec::new()
}

/// A remote configuration carrying `card` and `url`.
fn remote_config(card: AgentCard, url: &str) -> RemoteClientConfig {
    RemoteClientConfig {
        card: Some(card),
        url: String::from(url),
        ..RemoteClientConfig::default()
    }
}

#[test]
fn client_requires_a_non_empty_rpc_url() {
    // ports `A2AClient.__init__` raising when the SDK client cannot be built.
    for config in [
        A2aClientConfig::default(),
        A2aClientConfig::new(""),
        A2aClientConfig::new("   "),
    ] {
        let error = A2aClient::new(config).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
        assert_eq!(
            error.to_string(),
            "config: A2A client requires a non-empty rpc_url"
        );
    }
}

#[test]
fn config_builders_set_the_timeout_and_headers() {
    let config = A2aClientConfig::new("http://agent.test/a2a/jsonrpc/")
        .with_timeout(Duration::from_secs(5))
        .with_header("x-api-key", "secret")
        .with_header("x-api-key", "rotated");
    assert_eq!(config.rpc_url, "http://agent.test/a2a/jsonrpc/");
    assert_eq!(config.timeout, Some(Duration::from_secs(5)));
    assert_eq!(config.headers.len(), 1);
    assert_eq!(
        config.headers.get("x-api-key").map(String::as_str),
        Some("rotated")
    );
    assert_eq!(config.effective_timeout(), Duration::from_secs(5));
    assert_eq!(
        A2aClientConfig::default().effective_timeout(),
        Duration::from_secs(60)
    );
}

#[test]
fn remote_client_rejects_a_foreign_protocol() {
    // ports `_resolve_a2a_card` rejecting a non-A2A protocol.
    let config = RemoteClientConfig {
        protocol: String::from("gRPC"),
        card: Some(card_with(Vec::new())),
        ..RemoteClientConfig::default()
    };
    let error = A2aRemoteClient::new(config).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(
        error.to_string(),
        "validation: unsupported protocol 'gRPC', expected A2A"
    );
}

#[test]
fn remote_client_requires_an_agent_card() {
    let error = A2aRemoteClient::new(RemoteClientConfig::default()).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(
        error.to_string(),
        "validation: A2A remote client requires an agent card"
    );
}

#[test]
fn interface_urls_are_derived_like_python() {
    // ports `_convert_to_a2a_card`'s `interface_url` derivation.
    for (base, expected) in [
        ("http://h:1", "http://h:1/a2a/jsonrpc/"),
        ("http://h:1/", "http://h:1/a2a/jsonrpc/"),
        ("http://h:1/a2a/jsonrpc", "http://h:1/a2a/jsonrpc/"),
        ("http://h:1/a2a/jsonrpc/", "http://h:1/a2a/jsonrpc/"),
        ("http://h:1//", "http://h:1/a2a/jsonrpc/"),
        ("http://h:1/gateway", "http://h:1/gateway/a2a/jsonrpc/"),
    ] {
        assert_eq!(derive_interface_url(base), expected, "base `{base}`");
    }
}

#[test]
fn a_jsonrpc_interface_wins_over_the_config_url() {
    let card = card_with(vec![interface("https://peer.test:9/rpc", "JSONRPC")]);
    let client = A2aRemoteClient::new(remote_config(card, "http://ignored:1")).unwrap();
    assert_eq!(client.rpc_url(), "https://peer.test:9/rpc");
    assert!(!client.is_started());
    assert_eq!(client.config().url, "http://ignored:1");
    assert_eq!(
        client.card().supported_interfaces,
        vec![interface("https://peer.test:9/rpc", "JSONRPC")]
    );
}

#[test]
fn non_jsonrpc_and_empty_interfaces_are_skipped() {
    let card = card_with(vec![
        interface("https://peer.test/rest", "HTTP+JSON"),
        interface("", "JSONRPC"),
        interface("https://peer.test:9/rpc", "jsonrpc"),
    ]);
    let client = A2aRemoteClient::new(remote_config(card, "http://ignored:1")).unwrap();
    assert_eq!(client.rpc_url(), "https://peer.test:9/rpc");

    let derived = A2aRemoteClient::new(remote_config(
        card_with(vec![interface("https://peer.test/rest", "HTTP+JSON")]),
        "https://peer.test",
    ))
    .unwrap();
    assert_eq!(derived.rpc_url(), "https://peer.test/a2a/jsonrpc/");
}

#[test]
fn remote_client_requires_a_url_without_an_interface() {
    let error = A2aRemoteClient::new(remote_config(card_with(Vec::new()), "")).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(
        error.to_string(),
        "validation: A2A remote client requires a url"
    );
}

#[test]
fn sse_frame_split_across_chunks_is_reassembled() {
    let mut parser = SseFramer::new();
    assert_eq!(parser.push(br#"data: {"jsonrpc":"2.0","id":"#), no_frames());
    assert_eq!(
        parser.push(br#"1,"result":{"status":{"state":"TASK_STATE_WORKING"}}}"#),
        no_frames()
    );
    let frames = parser.push(b"\n\n");
    assert_eq!(
        frames,
        vec![String::from(
            r#"{"jsonrpc":"2.0","id":1,"result":{"status":{"state":"TASK_STATE_WORKING"}}}"#
        )]
    );
    assert!(parser.finish().is_none());
}

#[test]
fn sse_crlf_delimiters_are_accepted() {
    let mut parser = SseFramer::new();
    let frames = parser.push(b"data: first\r\n\r\ndata: second\r\n\r\n");
    assert_eq!(frames, vec![String::from("first"), String::from("second")]);
}

#[test]
fn sse_two_frames_in_one_chunk_are_both_emitted() {
    let mut parser = SseFramer::new();
    let frames = parser.push(b"data: one\n\ndata: two\n\n");
    assert_eq!(frames, vec![String::from("one"), String::from("two")]);
}

#[test]
fn sse_comments_and_control_fields_are_ignored() {
    let mut parser = SseFramer::new();
    // Keep-alive heartbeats, an event name, an id, and a retry hint alone
    // never produce a frame.
    assert_eq!(parser.push(b":keep-alive\n\n"), no_frames());
    assert_eq!(
        parser.push(b"event: ping\nid: 7\nretry: 100\n\n"),
        no_frames()
    );
    let frames = parser.push(b": keep-alive\nevent: message\ndata: payload\n\n");
    assert_eq!(frames, vec![String::from("payload")]);
}

#[test]
fn sse_multiline_data_is_joined_with_newlines() {
    let mut parser = SseFramer::new();
    let frames = parser.push(b"data: line one\ndata: line two\n\n");
    assert_eq!(frames, vec![String::from("line one\nline two")]);
}

#[test]
fn sse_leading_space_after_the_colon_is_stripped_once() {
    let mut parser = SseFramer::new();
    let frames = parser.push(b"data:  two spaces\n\ndata:no space\n\n");
    assert_eq!(
        frames,
        vec![String::from(" two spaces"), String::from("no space")]
    );
}

#[test]
fn sse_cr_at_a_chunk_boundary_is_joined_with_lf() {
    let mut parser = SseFramer::new();
    assert_eq!(parser.push(b"data: split\r"), no_frames());
    let frames = parser.push(b"\n\r\n");
    assert_eq!(frames, vec![String::from("split")]);
}

#[test]
fn sse_trailing_frame_without_a_blank_line_is_flushed() {
    let mut parser = SseFramer::new();
    assert_eq!(parser.push(b"data: tail"), no_frames());
    assert_eq!(parser.finish().as_deref(), Some("tail"));
    assert_eq!(parser.finish(), None);
    // A trailing frame that never carried a `data:` line is discarded.
    let mut empty = SseFramer::new();
    assert_eq!(empty.push(b"event: ping"), no_frames());
    assert_eq!(empty.finish(), None);
}

#[test]
fn sse_error_frame_becomes_the_typed_error() {
    // ports the JSON-RPC error branch of `A2AClient.send_message`.
    let mut parser = SseFramer::new();
    let frames = parser.push(
        br#"data: {"jsonrpc":"2.0","id":2,"error":{"code":-32001,"message":"no such task"}}"#,
    );
    assert_eq!(frames, no_frames());
    let frames = parser.push(b"\n\n");
    assert_eq!(frames.len(), 1);
    let error = decode_frame(&frames[0], "stream").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert_eq!(
        error.to_string(),
        "not_found: json-rpc error -32001: no such task"
    );
}

#[test]
fn sse_malformed_payload_is_a_transport_error() {
    let error = decode_frame("not json", "stream").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Transport);
    assert!(error
        .to_string()
        .starts_with("transport: A2A stream frame is not a valid JSON-RPC response: "));

    // A valid response without a result is a transport error as well.
    let mut parser = SseFramer::new();
    let frames = parser.push(br#"data: {"jsonrpc":"2.0","id":3}"#);
    assert_eq!(frames, no_frames());
    let frames = parser.push(b"\n\n");
    let error = decode_frame(&frames[0], "stream").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Transport);
    assert_eq!(
        error.to_string(),
        "transport: A2A stream response is missing result"
    );
}

#[tokio::test]
async fn invoke_folds_a_task_and_overrides_the_session() {
    // ports `A2AClient.invoke` folding `_merge_agent_results` and stamping
    // `_with_session_id`.
    let server = MockServer::start().await;
    Mock::given(http_method("POST"))
        .and(header("content-type", "application/json"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "id": "task-1",
                "contextId": "ctx-from-task",
                "status": {
                    "state": "TASK_STATE_COMPLETED",
                    "message": { "messageId": "m2", "role": "ROLE_AGENT", "parts": [] }
                },
                "artifacts": [{
                    "artifactId": "a1",
                    "parts": [{ "text": "done" }],
                    "metadata": { "chunks": 1 }
                }],
                "metadata": { "trace": "t-9" }
            }
        })))
        .mount(&server)
        .await;

    let client = A2aClient::new(A2aClientConfig::new(server.uri())).unwrap();
    let result = client
        .invoke(json!({ "query": "hello", "conversation_id": "conv-7" }))
        .await
        .unwrap();

    assert_eq!(result.task_id.as_deref(), Some("task-1"));
    // `conversation_id` wins over the task's own `contextId`.
    assert_eq!(result.session_id.as_deref(), Some("conv-7"));
    assert_eq!(result.status, AgentStatus::Completed);
    assert_eq!(result.artifacts.len(), 1);
    assert_eq!(result.artifacts[0].artifact_id.as_deref(), Some("a1"));
    assert_eq!(result.artifacts[0].parts[0].text.as_deref(), Some("done"));
    assert_eq!(result.metadata.get("trace"), Some(&json!("t-9")));
}

#[tokio::test]
async fn invoke_maps_a_json_rpc_error_body() {
    let server = MockServer::start().await;
    Mock::given(http_method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "jsonrpc": "2.0",
            "id": 1,
            "error": { "code": -32001, "message": "no such task" }
        })))
        .mount(&server)
        .await;

    let client = A2aClient::new(A2aClientConfig::new(server.uri())).unwrap();
    let error = client
        .invoke(json!({ "query": "hello" }))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert_eq!(
        error.to_string(),
        "not_found: json-rpc error -32001: no such task"
    );
}

#[tokio::test]
async fn invoke_reports_a_non_2xx_status() {
    let server = MockServer::start().await;
    Mock::given(http_method("POST"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&server)
        .await;

    let client = A2aClient::new(A2aClientConfig::new(server.uri())).unwrap();
    let error = client
        .invoke(json!({ "query": "hello" }))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Transport);
    assert_eq!(
        error.to_string(),
        "transport: A2A send failed with status 503"
    );
}

#[tokio::test]
async fn stream_yields_one_result_per_frame_in_order() {
    // ports `A2AClient.stream` yielding one `from_a2a_response` per event,
    // each stamped with the request session id.
    let server = MockServer::start().await;
    let body = concat!(
            "data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"taskId\":\"t1\",\"contextId\":\"c1\",\"status\":{\"state\":\"TASK_STATE_WORKING\"}}}\n\n",
            "data: {\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"taskId\":\"t1\",\"contextId\":\"c1\",\"artifact\":{\"artifactId\":\"a1\",\"parts\":[{\"text\":\"chunk\"}]},\"lastChunk\":true}}\n\n",
            "data: {\"jsonrpc\":\"2.0\",\"id\":3,\"result\":{\"taskId\":\"t1\",\"contextId\":\"c1\",\"status\":{\"state\":\"TASK_STATE_COMPLETED\"}}}\n\n",
        );
    Mock::given(http_method("POST"))
        .and(header("accept", "text/event-stream"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .mount(&server)
        .await;

    let client = A2aClient::new(A2aClientConfig::new(server.uri())).unwrap();
    let frames: Vec<Result<AgentResult>> = client
        .stream(json!({ "query": "go", "sessionId": "s-1" }))
        .collect()
        .await;
    assert_eq!(frames.len(), 3);
    let results: Vec<AgentResult> = frames
        .into_iter()
        .map(|frame| frame.expect("every frame decodes"))
        .collect();

    assert_eq!(results[0].status, AgentStatus::Working);
    assert_eq!(results[0].artifacts, Vec::new());
    assert_eq!(results[1].status, AgentStatus::Working);
    assert_eq!(results[1].artifacts.len(), 1);
    assert_eq!(
        results[1].artifacts[0].parts[0].text.as_deref(),
        Some("chunk")
    );
    assert_eq!(results[2].status, AgentStatus::Completed);
    for result in &results {
        assert_eq!(result.session_id.as_deref(), Some("s-1"));
        assert_eq!(result.task_id.as_deref(), Some("t1"));
    }
}

#[tokio::test]
async fn stream_stops_on_a_json_rpc_error_frame() {
    let server = MockServer::start().await;
    let body = concat!(
            "data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"taskId\":\"t1\",\"status\":{\"state\":\"TASK_STATE_WORKING\"}}}\n\n",
            "data: {\"jsonrpc\":\"2.0\",\"id\":2,\"error\":{\"code\":-32001,\"message\":\"gone\"}}\n\n",
        );
    Mock::given(http_method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .mount(&server)
        .await;

    let client = A2aClient::new(A2aClientConfig::new(server.uri())).unwrap();
    let frames: Vec<Result<AgentResult>> = client.stream(json!({ "query": "go" })).collect().await;
    assert_eq!(frames.len(), 2);
    assert!(frames[0].is_ok());
    let error = frames[1].as_ref().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
}

#[tokio::test]
async fn start_and_stop_toggle_the_started_flag() {
    // ports `A2ARemoteClient.start` / `stop` / `is_started`.
    let card = card_with(vec![interface("http://agent.test/a2a/jsonrpc/", "JSONRPC")]);
    let client = A2aRemoteClient::new(remote_config(card, "")).unwrap();
    assert!(!client.is_started());
    client.start().await.unwrap();
    assert!(client.is_started());
    // Starting twice is idempotent.
    client.start().await.unwrap();
    assert!(client.is_started());
    client.stop().await;
    assert!(!client.is_started());
    // Stopping an already stopped client is a no-op.
    client.stop().await;
    assert!(!client.is_started());
}
