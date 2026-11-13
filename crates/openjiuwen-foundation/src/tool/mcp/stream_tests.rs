//! Tests for the server-initiated `GET` event stream of the streamable-HTTP
//! transport, plus the pure helpers it is built from.
//!
//! Every server is a loopback `wiremock` instance: the initialize handshake and
//! the `GET` channel are mocked, no test reaches the network, spawns a process,
//! or waits without a bound.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use std::time::Duration;

use serde_json::{json, Value};
use wiremock::matchers::{body_partial_json, header, method};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

use super::client::{McpClient, McpNotification, McpServerConfig, RmcpClient};
use super::http::{
    server_stream_headers, stream_reconnect_delay, SseParser, DEFAULT_STREAM_RETRY,
    NOTIFICATION_BUFFER_BOUND,
};

/// Upper bound on every wait for a spawned stream task.
const BOUND: Duration = Duration::from_secs(5);
/// Poll interval while waiting for a spawned stream task.
const POLL: Duration = Duration::from_millis(20);
/// The SSE frame the mocked `GET` channel carries.
const NOTIFICATION_FRAME: &str = "event: message\ndata: {\"jsonrpc\":\"2.0\",\"method\":\
                                  \"notifications/tools/list_changed\",\"params\":{\"reason\":\
                                  \"test\"}}\n\n";

/// Answers the `initialize` request like an MCP server, echoing the request id.
struct InitializeResponder;

impl Respond for InitializeResponder {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let id = serde_json::from_slice::<Value>(&request.body)
            .ok()
            .and_then(|body| body.get("id").cloned())
            .unwrap_or(Value::Null);
        ResponseTemplate::new(200)
            .insert_header("Mcp-Session-Id", "sess-1")
            .insert_header("Content-Type", "application/json")
            .set_body_json(json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "serverInfo": {"name": "mock", "version": "1.0"}
                }
            }))
    }
}

/// Mounts the handshake mocks every streamable-HTTP test needs.
async fn mount_handshake(server: &MockServer) {
    Mock::given(method("POST"))
        .and(body_partial_json(json!({"method": "initialize"})))
        .respond_with(InitializeResponder)
        .mount(server)
        .await;
    Mock::given(method("POST"))
        .and(body_partial_json(
            json!({"method": "notifications/initialized"}),
        ))
        .respond_with(ResponseTemplate::new(202))
        .mount(server)
        .await;
}

/// Builds a streamable-HTTP client pointed at `server`.
fn client_for(server: &MockServer) -> RmcpClient {
    let config = McpServerConfig::http("srv", server.uri()).with_server_id("srv-id");
    RmcpClient::new(&config).unwrap()
}

/// Polls `take_notifications` until a notification arrives or [`BOUND`] elapses.
async fn wait_for_notifications(client: &RmcpClient) -> Vec<McpNotification> {
    let deadline = tokio::time::Instant::now() + BOUND;
    loop {
        let notifications = client.take_notifications().await.unwrap();
        if !notifications.is_empty() || tokio::time::Instant::now() >= deadline {
            return notifications;
        }
        tokio::time::sleep(POLL).await;
    }
}

/// Polls the recorded requests until `predicate` holds or [`BOUND`] elapses.
async fn wait_for_request<F>(server: &MockServer, predicate: F) -> Vec<Request>
where
    F: Fn(&[Request]) -> bool,
{
    let deadline = tokio::time::Instant::now() + BOUND;
    loop {
        let requests = server.received_requests().await.unwrap();
        if predicate(&requests) || tokio::time::Instant::now() >= deadline {
            return requests;
        }
        tokio::time::sleep(POLL).await;
    }
}

/// Returns the number of `GET` requests `server` recorded.
fn get_request_count(requests: &[Request]) -> usize {
    requests
        .iter()
        .filter(|request| request.method.as_str() == "GET")
        .count()
}

// --- end-to-end: a supported GET channel ------------------------------------

#[tokio::test]
async fn server_stream_buffers_notifications_and_reports_support() {
    let server = MockServer::start().await;
    mount_handshake(&server).await;
    Mock::given(method("GET"))
        .and(header("accept", "text/event-stream"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("Content-Type", "text/event-stream")
                .set_body_string(NOTIFICATION_FRAME),
        )
        .mount(&server)
        .await;
    let client = client_for(&server);

    client.connect().await.unwrap();
    let notifications = wait_for_notifications(&client).await;

    assert!(!notifications.is_empty(), "no notification was buffered");
    assert_eq!(notifications[0].method, "notifications/tools/list_changed");
    assert_eq!(notifications[0].params["reason"], json!("test"));
    assert!(client.supports_server_stream());
    client.disconnect().await.unwrap();
}

// --- end-to-end: a refused GET channel --------------------------------------

#[tokio::test]
async fn refused_server_stream_is_reported_as_unsupported_and_never_retried() {
    let server = MockServer::start().await;
    mount_handshake(&server).await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(405))
        .mount(&server)
        .await;
    let client = client_for(&server);

    client.connect().await.unwrap();
    let requests = wait_for_request(&server, |requests| get_request_count(requests) >= 1).await;
    assert_eq!(
        get_request_count(&requests),
        1,
        "the GET channel was not opened"
    );
    // Give the spawned loop time to record the refusal and (wrongly) retry.
    tokio::time::sleep(Duration::from_millis(300)).await;
    let requests = server.received_requests().await.unwrap();

    assert!(!client.supports_server_stream());
    assert_eq!(
        client.take_notifications().await.unwrap(),
        Vec::<McpNotification>::new()
    );
    assert_eq!(
        get_request_count(&requests),
        1,
        "a refused channel was retried"
    );
    client.disconnect().await.unwrap();
}

#[tokio::test]
async fn take_notifications_is_empty_without_a_session() {
    let session = client_for(&MockServer::start().await);

    assert!(!session.supports_server_stream());
    assert_eq!(
        session.take_notifications().await.unwrap(),
        Vec::<McpNotification>::new()
    );
}

// --- pure helpers -----------------------------------------------------------

#[test]
fn server_stream_headers_advertise_the_session_and_the_event_stream() {
    let headers = server_stream_headers("sess-1", None);

    let names: Vec<&str> = headers.iter().map(|(name, _)| *name).collect();
    assert_eq!(
        names,
        vec![
            "Accept",
            "Mcp-Session-Id",
            "MCP-Protocol-Version",
            "Cache-Control"
        ]
    );
    assert_eq!(headers[0].1, "text/event-stream");
    assert_eq!(headers[1].1, "sess-1");
    assert!(!headers[2].1.is_empty(), "no protocol version was sent");
    assert_eq!(headers[3].1, "no-store");
}

#[test]
fn server_stream_headers_replay_the_last_event_id() {
    let headers = server_stream_headers("sess-1", Some("event-7"));

    let last = headers.last().unwrap();
    assert_eq!(last.0, "Last-Event-ID");
    assert_eq!(last.1, "event-7");
}

#[test]
fn stream_reconnect_delay_defaults_to_one_second() {
    assert_eq!(stream_reconnect_delay(1, None), DEFAULT_STREAM_RETRY);
    assert_eq!(stream_reconnect_delay(1, None), Duration::from_secs(1));
}

#[test]
fn stream_reconnect_delay_prefers_a_short_server_hint() {
    assert_eq!(
        stream_reconnect_delay(1, Some(Duration::from_millis(250))),
        Duration::from_millis(250)
    );
}

#[test]
fn stream_reconnect_delay_caps_a_long_server_hint() {
    // Attempt 6 is where the shared policy itself saturates at 120 s.
    assert_eq!(
        stream_reconnect_delay(6, Some(Duration::from_secs(600))),
        Duration::from_secs(120)
    );
}

#[test]
fn stream_reconnect_delay_stays_within_the_shared_policy() {
    assert_eq!(
        stream_reconnect_delay(1, Some(Duration::from_secs(600))),
        Duration::from_secs(5)
    );
}

#[test]
fn sse_parser_captures_id_and_retry() {
    let mut parser = SseParser::default();

    let frames = parser.push(b"id: 42\nretry: 2500\ndata: keep\n\n");

    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].id.as_deref(), Some("42"));
    assert_eq!(frames[0].retry, Some(2500));
    assert_eq!(frames[0].data, "keep");
    assert_eq!(frames[0].event, None);
}

#[test]
fn sse_parser_ignores_an_unparsable_retry() {
    let mut parser = SseParser::default();

    let frames = parser.push(b"retry: soon\ndata: keep\n\n");

    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].retry, None);
    assert_eq!(frames[0].id, None);
    assert_eq!(frames[0].data, "keep");
}

#[test]
fn sse_parser_does_not_change_event_and_data_frames() {
    let mut parser = SseParser::default();

    let frames = parser.push(b"event: message\ndata: one\ndata: two\n\n");

    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].event.as_deref(), Some("message"));
    assert_eq!(frames[0].data, "one\ntwo");
    assert_eq!(frames[0].id, None);
    assert_eq!(frames[0].retry, None);
}

// --- notification buffer ----------------------------------------------------

#[test]
fn notification_buffer_drops_the_oldest_entry_when_full() {
    use super::http::ServerStreamState;

    let state = ServerStreamState::default();
    for index in 0..(NOTIFICATION_BUFFER_BOUND + 2) {
        state.push_notification(McpNotification {
            method: format!("notification-{index}"),
            params: Value::Null,
        });
    }

    let notifications = state.take_notifications();
    assert_eq!(notifications.len(), NOTIFICATION_BUFFER_BOUND);
    assert_eq!(notifications[0].method, "notification-2");
    assert_eq!(
        notifications[NOTIFICATION_BUFFER_BOUND - 1].method,
        format!("notification-{}", NOTIFICATION_BUFFER_BOUND + 1)
    );
    assert!(
        state.take_notifications().is_empty(),
        "the buffer was not drained"
    );
}
