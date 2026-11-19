//! Tests for the server-initiated `GET` event stream of the streamable-HTTP
//! transport, the resource-subscription requests of the `rmcp` client, and the
//! legacy SSE transport's `Last-Event-ID` replay, plus the pure helpers they are
//! built from.
//!
//! Every server is a loopback `wiremock` instance: the initialize handshake, the
//! `GET` channel, and the JSON-RPC requests are mocked, no test reaches the
//! network, spawns a process, or waits without a bound.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use openjiuwen_common::error::ErrorKind;
use rmcp::model::{
    ClientJsonRpcMessage, ClientNotification, Extensions, InitializedNotification,
    InitializedNotificationMethod,
};
use rmcp::transport::Transport;
use serde_json::{json, Value};
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

use super::client::{McpClient, McpClientType, McpNotification, McpServerConfig, RmcpClient};
use super::http::sse::{sse_resume_error, sse_stream_headers, SseStop, SseTransport};
use super::http::{
    server_stream_headers, stream_reconnect_delay, HttpTransportError, SseParser,
    StreamableHttpTransport, DEFAULT_STREAM_RETRY, NOTIFICATION_BUFFER_BOUND,
};

/// Upper bound on every wait for a spawned stream task.
const BOUND: Duration = Duration::from_secs(5);
/// Poll interval while waiting for a spawned stream task.
const POLL: Duration = Duration::from_millis(20);
/// The SSE frame the mocked `GET` channel carries.
const NOTIFICATION_FRAME: &str = "event: message\ndata: {\"jsonrpc\":\"2.0\",\"method\":\
                                  \"notifications/tools/list_changed\",\"params\":{\"reason\":\
                                  \"test\"}}\n\n";
/// The resource the subscription tests subscribe to.
const RESOURCE_URI: &str = "file:///tmp/report.md";

/// Answers the `initialize` request like an MCP server, echoing the request id.
///
/// `capabilities` is what the server advertises in its `initialize` result.
struct InitializeResponder {
    capabilities: Value,
}

impl Default for InitializeResponder {
    fn default() -> Self {
        Self {
            capabilities: json!({}),
        }
    }
}

impl Respond for InitializeResponder {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        ResponseTemplate::new(200)
            .insert_header("Mcp-Session-Id", "sess-1")
            .insert_header("Content-Type", "application/json")
            .set_body_json(json!({
                "jsonrpc": "2.0",
                "id": request_id(request),
                "result": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": self.capabilities,
                    "serverInfo": {"name": "mock", "version": "1.0"}
                }
            }))
    }
}

/// Answers a JSON-RPC request with an empty `result`, echoing the request id.
struct EmptyResultResponder;

impl Respond for EmptyResultResponder {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        ResponseTemplate::new(200)
            .insert_header("Content-Type", "application/json")
            .set_body_json(json!({
                "jsonrpc": "2.0",
                "id": request_id(request),
                "result": {}
            }))
    }
}

/// Returns the `id` of the JSON-RPC `request`.
fn request_id(request: &Request) -> Value {
    serde_json::from_slice::<Value>(&request.body)
        .ok()
        .and_then(|body| body.get("id").cloned())
        .unwrap_or(Value::Null)
}

/// Mounts the handshake mocks every streamable-HTTP test needs.
async fn mount_handshake(server: &MockServer) {
    mount_handshake_with(server, json!({})).await;
}

/// Mounts the handshake mocks with `capabilities` as the advertised capabilities.
async fn mount_handshake_with(server: &MockServer, capabilities: Value) {
    Mock::given(method("POST"))
        .and(body_partial_json(json!({"method": "initialize"})))
        .respond_with(InitializeResponder { capabilities })
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

// --- resource subscriptions (rmcp client) -----------------------------------

/// Mounts the handshake advertising `resources.subscribe` and the matching
/// `resources/subscribe` / `resources/unsubscribe` responders.
async fn mount_subscription_handshake(server: &MockServer) {
    mount_handshake_with(server, json!({"resources": {"subscribe": true}})).await;
    for method_name in ["resources/subscribe", "resources/unsubscribe"] {
        Mock::given(method("POST"))
            .and(body_partial_json(json!({"method": method_name})))
            .respond_with(EmptyResultResponder)
            .mount(server)
            .await;
    }
}

/// Returns the recorded POST requests whose JSON-RPC body carries `wanted`.
fn posts_with_method<'a>(requests: &'a [Request], wanted: &str) -> Vec<&'a Request> {
    requests
        .iter()
        .filter(|request| {
            request.method.as_str() == "POST"
                && serde_json::from_slice::<Value>(&request.body)
                    .ok()
                    .and_then(|body| {
                        body.get("method")
                            .and_then(Value::as_str)
                            .map(|method_name| method_name == wanted)
                    })
                    .unwrap_or(false)
        })
        .collect()
}

#[tokio::test]
async fn subscribe_resource_sends_the_negotiated_request_and_records_the_capability() {
    let server = MockServer::start().await;
    mount_subscription_handshake(&server).await;
    let client = client_for(&server);

    client.subscribe_resource(RESOURCE_URI).await.unwrap();

    assert!(client.supports_resource_subscriptions());
    let requests = server.received_requests().await.unwrap();
    let subscribe = posts_with_method(&requests, "resources/subscribe");
    assert_eq!(subscribe.len(), 1, "resources/subscribe was not sent");
    let body: Value = serde_json::from_slice(&subscribe[0].body).unwrap();
    // `rmcp` adds its own `_meta.progressToken`, so only the URI is asserted.
    assert_eq!(body["params"]["uri"], json!(RESOURCE_URI));
    assert!(
        posts_with_method(&requests, "resources/unsubscribe").is_empty(),
        "unsubscribe was sent by subscribe"
    );
    client.disconnect().await.unwrap();
}

#[tokio::test]
async fn unsubscribe_resource_sends_the_negotiated_request() {
    let server = MockServer::start().await;
    mount_subscription_handshake(&server).await;
    let client = client_for(&server);

    client.unsubscribe_resource(RESOURCE_URI).await.unwrap();

    let requests = server.received_requests().await.unwrap();
    let unsubscribe = posts_with_method(&requests, "resources/unsubscribe");
    assert_eq!(unsubscribe.len(), 1, "resources/unsubscribe was not sent");
    let body: Value = serde_json::from_slice(&unsubscribe[0].body).unwrap();
    assert_eq!(body["params"]["uri"], json!(RESOURCE_URI));
    client.disconnect().await.unwrap();
}

#[tokio::test]
async fn a_subscription_without_the_negotiated_capability_is_a_typed_error() {
    let server = MockServer::start().await;
    mount_handshake(&server).await;
    let client = client_for(&server);

    let error = client
        .subscribe_resource(RESOURCE_URI)
        .await
        .expect_err("the server advertised no resources.subscribe");

    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.message().contains("resources/subscribe"), "{error}");
    assert!(!client.supports_resource_subscriptions());
    let requests = server.received_requests().await.unwrap();
    assert!(
        posts_with_method(&requests, "resources/subscribe").is_empty(),
        "a request the server never advertised was sent"
    );
    client.disconnect().await.unwrap();
}

#[tokio::test]
async fn a_blank_subscription_uri_fails_before_the_session_is_opened() {
    let server = MockServer::start().await;
    mount_subscription_handshake(&server).await;
    let client = client_for(&server);

    let error = client
        .subscribe_resource("   ")
        .await
        .expect_err("a blank uri must not reach the server");

    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(!client.is_connected().await, "the client connected anyway");
    let requests = server.received_requests().await.unwrap();
    assert!(requests.is_empty(), "a blank uri made a request");
}

// --- legacy SSE replay ------------------------------------------------------

/// Answers the legacy SSE `GET` stream with one `endpoint` frame.
///
/// The frame carries `event_id` when one is scripted; the call counter lets a
/// test tell the reconnects apart.
struct LegacySseResponder {
    calls: Arc<AtomicUsize>,
    event_id: Option<&'static str>,
}

impl Respond for LegacySseResponder {
    fn respond(&self, _request: &Request) -> ResponseTemplate {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let id = match self.event_id {
            Some(id) => format!("id: {id}\n"),
            None => String::new(),
        };
        ResponseTemplate::new(200)
            .insert_header("Content-Type", "text/event-stream")
            .set_body_string(format!("event: endpoint\ndata: /messages\n{id}\n"))
    }
}

/// Mounts the legacy SSE `GET` stream and the POST endpoint it announces.
async fn mount_legacy_sse(server: &MockServer, event_id: Option<&'static str>) -> Arc<AtomicUsize> {
    let calls = Arc::new(AtomicUsize::new(0));
    Mock::given(method("GET"))
        .respond_with(LegacySseResponder {
            calls: calls.clone(),
            event_id,
        })
        .mount(server)
        .await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(202))
        .mount(server)
        .await;
    calls
}

/// Builds the legacy SSE configuration for `server`.
fn sse_config(server: &MockServer) -> McpServerConfig {
    McpServerConfig {
        client_type: McpClientType::Sse,
        ..McpServerConfig::http("sse", server.uri())
    }
}

/// Builds a notification, enough to trigger the legacy transport's first send.
fn trigger_message() -> ClientJsonRpcMessage {
    ClientJsonRpcMessage::notification(ClientNotification::InitializedNotification(
        InitializedNotification {
            method: InitializedNotificationMethod,
            extensions: Extensions::default(),
        },
    ))
}

/// Returns the `Last-Event-ID` header of `request`, when the client sent one.
fn last_event_id(request: &Request) -> Option<String> {
    request
        .headers
        .get("Last-Event-ID")
        .and_then(|value| value.to_str().ok())
        .map(ToOwned::to_owned)
}

/// Returns the recorded `GET` requests, oldest first.
fn get_requests(requests: &[Request]) -> Vec<&Request> {
    requests
        .iter()
        .filter(|request| request.method.as_str() == "GET")
        .collect()
}

#[tokio::test]
async fn legacy_sse_replays_the_last_event_id_on_reconnect() {
    let server = MockServer::start().await;
    mount_legacy_sse(&server, Some("7")).await;
    let mut transport = SseTransport::new(&sse_config(&server)).unwrap();

    transport.send(trigger_message()).await.unwrap();
    let requests = wait_for_request(&server, |requests| get_request_count(requests) >= 2).await;

    let gets = get_requests(&requests);
    assert!(
        gets.len() >= 2,
        "the GET stream was not reopened: {} request(s)",
        gets.len()
    );
    assert!(
        last_event_id(gets[0]).is_none(),
        "the first GET replayed an event id it never saw"
    );
    assert_eq!(last_event_id(gets[1]).as_deref(), Some("7"));
    transport.close().await.unwrap();
}

#[tokio::test]
async fn legacy_sse_sends_no_last_event_id_when_it_never_accepted_one() {
    let server = MockServer::start().await;
    mount_legacy_sse(&server, None).await;
    let mut transport = SseTransport::new(&sse_config(&server)).unwrap();

    transport.send(trigger_message()).await.unwrap();
    let requests = wait_for_request(&server, |requests| get_request_count(requests) >= 2).await;

    let gets = get_requests(&requests);
    assert!(
        gets.len() >= 2,
        "the GET stream was not reopened: {} request(s)",
        gets.len()
    );
    for request in gets {
        assert!(
            last_event_id(request).is_none(),
            "an id the client never saw was replayed"
        );
    }
    transport.close().await.unwrap();
}

#[tokio::test]
async fn a_rejected_legacy_sse_stream_fails_the_send_with_the_status() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(405))
        .mount(&server)
        .await;
    let mut transport = SseTransport::new(&sse_config(&server)).unwrap();

    let error = transport
        .send(trigger_message())
        .await
        .expect_err("refused");

    assert!(error.to_string().contains("405"), "{error}");
    transport.close().await.unwrap();
}

#[test]
fn sse_stream_headers_advertise_the_event_stream_and_replay_an_id() {
    assert_eq!(
        sse_stream_headers(None),
        vec![("Accept", "text/event-stream".to_owned())]
    );
    assert_eq!(
        sse_stream_headers(Some("7")),
        vec![
            ("Accept", "text/event-stream".to_owned()),
            ("Last-Event-ID", "7".to_owned()),
        ]
    );
}

#[test]
fn sse_resume_error_names_the_last_event_id_and_the_budget() {
    let exhausted = sse_resume_error(SseStop::Exhausted(3), Some("7")).to_string();
    let without_id = sse_resume_error(SseStop::Exhausted(2), None).to_string();
    let rejected = sse_resume_error(SseStop::Rejected(405), None).to_string();

    assert!(exhausted.contains("`7`"), "{exhausted}");
    assert!(exhausted.contains("3 attempt"), "{exhausted}");
    assert!(!without_id.contains("event id"), "{without_id}");
    assert!(rejected.contains("405"), "{rejected}");
}

// --- request deadlines ------------------------------------------------------
/// A deadline short enough to keep the timeout tests fast.
const SHORT_TIMEOUT: Duration = Duration::from_millis(150);

#[tokio::test]
async fn streamable_http_post_times_out_instead_of_waiting_forever() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(30)))
        .mount(&server)
        .await;
    let config = McpServerConfig::http("srv", server.uri())
        .with_server_id("srv-id")
        .with_request_timeout(SHORT_TIMEOUT);
    let mut transport = StreamableHttpTransport::new(&config).unwrap();

    let started = tokio::time::Instant::now();
    let error = tokio::time::timeout(BOUND, transport.send(trigger_message()))
        .await
        .expect("the send hung past the deadline")
        .expect_err("a stalled server must not answer");
    assert!(matches!(error, HttpTransportError::Timeout(_)), "{error:?}");
    assert!(error.to_string().contains("timed out"), "{error}");
    assert!(started.elapsed() < BOUND, "the deadline was not applied");
    transport.close().await.unwrap();
}

#[tokio::test]
async fn streamable_http_close_times_out_instead_of_waiting_forever() {
    let server = MockServer::start().await;
    // The notification is accepted and hands out a session id; the DELETE stalls.
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(202).insert_header("Mcp-Session-Id", "sess-1"))
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .respond_with(ResponseTemplate::new(204).set_delay(Duration::from_secs(30)))
        .mount(&server)
        .await;
    let config = McpServerConfig::http("srv", server.uri())
        .with_server_id("srv-id")
        .with_request_timeout(SHORT_TIMEOUT);
    let mut transport = StreamableHttpTransport::new(&config).unwrap();
    let _ = transport.send(trigger_message()).await;

    tokio::time::timeout(BOUND, transport.close())
        .await
        .expect("close hung past the deadline")
        .expect("close ignores a failed DELETE");
}

#[tokio::test]
async fn legacy_sse_post_times_out_instead_of_waiting_forever() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(LegacySseResponder {
            calls: Arc::new(AtomicUsize::new(0)),
            event_id: None,
        })
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(202).set_delay(Duration::from_secs(30)))
        .mount(&server)
        .await;
    let config = sse_config(&server).with_request_timeout(SHORT_TIMEOUT);
    let mut transport = SseTransport::new(&config).unwrap();

    let error = tokio::time::timeout(BOUND, transport.send(trigger_message()))
        .await
        .expect("the POST hung past the deadline")
        .expect_err("a stalled server must not answer");
    assert!(matches!(error, HttpTransportError::Timeout(_)), "{error:?}");
    transport.close().await.unwrap();
}

#[tokio::test]
async fn list_tools_reports_a_typed_timeout_on_a_stalled_server() {
    let server = MockServer::start().await;
    mount_handshake(&server).await;
    Mock::given(method("POST"))
        .and(body_partial_json(json!({"method": "tools/list"})))
        .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(30)))
        .mount(&server)
        .await;
    let config = McpServerConfig::http("srv", server.uri())
        .with_server_id("srv-id")
        .with_request_timeout(SHORT_TIMEOUT);
    let client = RmcpClient::new(&config).unwrap();

    let error = tokio::time::timeout(BOUND, client.list_tools())
        .await
        .expect("list_tools hung past the deadline")
        .expect_err("a stalled server must not answer");
    assert_eq!(error.kind(), ErrorKind::Timeout, "{error}");
    assert!(error.to_string().contains("timed out"), "{error}");
}
