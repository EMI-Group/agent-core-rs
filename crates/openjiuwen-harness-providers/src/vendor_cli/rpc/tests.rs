//! Unit tests for the JSON-RPC client.
//!
//! The tests use an in-memory [`RpcTransport`](super::RpcTransport) that plays the peer, so they
//! are offline and deterministic: no subprocess, no network, no wall-clock ordering.

#![allow(clippy::expect_used, clippy::unwrap_used)] // tests assert by panicking

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_harness_protocol::{HarnessError, HarnessErrorKind, HarnessResult, JsonValue};
use serde_json::json;
use tokio::sync::mpsc;

use super::{request_id_value, JsonRpcClient, RpcTransport};

/// Label every test client reports.
const LABEL: &str = "Memory";

/// Id the fixture injects to release a reader that is blocked waiting for input.
const PUMP_ID: u64 = 9_999_999;

/// Bounded await budget for every client interaction.
const BOUND: Duration = Duration::from_secs(10);

/// One message the fake peer sends.
enum Inbound {
    /// A decodable message.
    Message(JsonValue),
    /// An unreadable line, reported as a transport error.
    Error(String),
    /// End of stream.
    End,
}

/// In-memory stand-in for a vendor CLI: lines the client writes are observed, and messages the
/// test pushes are read back.
struct MemoryTransport {
    /// Lines the client wrote, oldest first.
    written: mpsc::UnboundedSender<String>,
    /// Messages the fake peer sends, oldest first.
    incoming: mpsc::UnboundedReceiver<Inbound>,
    /// Diagnostic tail reported to the client.
    stderr: String,
}

#[async_trait]
impl RpcTransport for MemoryTransport {
    async fn write_line(&mut self, line: &str) -> HarnessResult<()> {
        let _ = self.written.send(line.to_owned());
        Ok(())
    }

    async fn read_json(&mut self) -> HarnessResult<Option<JsonValue>> {
        match self.incoming.recv().await {
            Some(Inbound::Message(value)) => Ok(Some(value)),
            Some(Inbound::Error(message)) => Err(HarnessError::protocol(message)),
            Some(Inbound::End) | None => Ok(None),
        }
    }

    fn stderr_tail(&self) -> String {
        self.stderr.clone()
    }
}

/// One connected client plus both ends of its transport.
struct Fixture {
    /// The client under test.
    client: Arc<JsonRpcClient>,
    /// Lines the client wrote.
    written: mpsc::UnboundedReceiver<String>,
    /// Messages the fake peer sends.
    server: mpsc::UnboundedSender<Inbound>,
}

impl Fixture {
    /// Creates a connected fixture with an empty stderr tail.
    fn new() -> Self {
        Self::with_stderr("")
    }

    /// Creates a connected fixture reporting `stderr` as the peer's diagnostic tail.
    fn with_stderr(stderr: &str) -> Self {
        let (written, written_rx) = mpsc::unbounded_channel();
        let (server, incoming) = mpsc::unbounded_channel();
        let transport = MemoryTransport {
            written,
            incoming,
            stderr: stderr.to_owned(),
        };
        Self {
            client: JsonRpcClient::spawn(Box::new(transport), LABEL),
            written: written_rx,
            server,
        }
    }

    /// Returns a fresh handle to the client.
    fn handle(&self) -> Arc<JsonRpcClient> {
        Arc::clone(&self.client)
    }

    /// Pushes one message the peer sends.
    fn send(&self, value: JsonValue) {
        let _ = self.server.send(Inbound::Message(value));
    }

    /// Pushes one unreadable line.
    fn send_error(&self, message: &str) {
        let _ = self.server.send(Inbound::Error(message.to_owned()));
    }

    /// Ends the stream.
    fn end(&self) {
        let _ = self.server.send(Inbound::End);
    }

    /// Waits for the next line the client wrote, pumping the reader so a queued write is
    /// flushed even while the reader is parked waiting for input.
    async fn next_written(&mut self) -> JsonValue {
        let pump = self.server.clone();
        let task = tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_millis(5)).await;
                // An unknown response id: the client logs it and ignores it.
                let keepalive = json!({"id": PUMP_ID, "result": null});
                if pump.send(Inbound::Message(keepalive)).is_err() {
                    return;
                }
            }
        });
        let waited = tokio::time::timeout(BOUND, self.written.recv()).await;
        task.abort();
        let line = waited
            .expect("the client must write its line")
            .expect("a written line");
        serde_json::from_str(&line).expect("a JSON line")
    }
}

/// Awaits `next_incoming` with a bounded budget.
async fn incoming(client: &Arc<JsonRpcClient>) -> Option<openjiuwen_harness_protocol::JsonObject> {
    tokio::time::timeout(BOUND, client.next_incoming())
        .await
        .expect("the peer message must arrive")
}

#[test]
fn request_ids_are_numbers_counting_from_one() {
    assert_eq!(request_id_value(1), json!(1));
    assert_eq!(request_id_value(42), json!(42));
}

#[tokio::test]
async fn requests_are_correlated_out_of_order() {
    let mut fixture = Fixture::new();
    let first = {
        let client = fixture.handle();
        tokio::spawn(async move { client.request("first", json!({"n": 1})).await })
    };
    let first_line = fixture.next_written().await;
    assert_eq!(first_line["id"], json!(1));
    assert_eq!(first_line["method"], json!("first"));
    assert_eq!(first_line["params"], json!({"n": 1}));

    let second = {
        let client = fixture.handle();
        tokio::spawn(async move { client.request("second", json!({"n": 2})).await })
    };
    let second_line = fixture.next_written().await;
    assert_eq!(second_line["id"], json!(2));
    assert_eq!(second_line["method"], json!("second"));

    // The second request is answered before the first one.
    fixture.send(json!({"id": 2, "result": {"who": "second"}}));
    fixture.send(json!({"id": 1, "result": {"who": "first"}}));
    assert_eq!(
        second.await.expect("no panic").expect("the second answer"),
        json!({"who": "second"})
    );
    assert_eq!(
        first.await.expect("no panic").expect("the first answer"),
        json!({"who": "first"})
    );
}

#[tokio::test]
async fn notifications_arrive_in_order_through_next_incoming() {
    let fixture = Fixture::new();
    fixture.send(json!({"method": "first", "params": {"n": 1}}));
    fixture.send(json!({"id": 7, "method": "ask", "params": {"n": 2}}));

    let notification = incoming(&fixture.client).await.expect("the notification");
    assert_eq!(notification.get("method"), Some(&json!("first")));
    assert_eq!(notification.get("id"), None);
    assert_eq!(notification.get("params"), Some(&json!({"n": 1})));

    let request = incoming(&fixture.client).await.expect("the server request");
    assert_eq!(request.get("method"), Some(&json!("ask")));
    assert_eq!(request.get("id"), Some(&json!(7)));
}

#[tokio::test]
async fn a_server_request_is_answered_with_the_id_echoed_verbatim() {
    let mut fixture = Fixture::new();
    fixture.send(json!({"id": "peer-1", "method": "workspace/apply", "params": {"path": "a"}}));
    let request = incoming(&fixture.client).await.expect("the server request");
    assert_eq!(request.get("id"), Some(&json!("peer-1")));

    fixture
        .client
        .respond(json!("peer-1"), json!({"ok": true}))
        .await
        .expect("the answer is queued");
    assert_eq!(
        fixture.next_written().await,
        json!({"id": "peer-1", "result": {"ok": true}})
    );
}

#[tokio::test]
async fn a_server_request_can_be_declined_with_an_error() {
    let mut fixture = Fixture::new();
    fixture.send(json!({"id": 5, "method": "workspace/apply"}));
    let request = incoming(&fixture.client).await.expect("the server request");
    assert_eq!(request.get("id"), Some(&json!(5)));

    fixture
        .client
        .respond_error(json!(5), -32601, "nope")
        .await
        .expect("the answer is queued");
    assert_eq!(
        fixture.next_written().await,
        json!({"id": 5, "error": {"code": -32601, "message": "nope"}})
    );
}

#[tokio::test]
async fn a_peer_error_carries_its_message_and_code() {
    let fixture = Fixture::new();
    fixture.send(json!({"id": 1, "error": {"code": -32601, "message": "method not found"}}));
    let error = fixture
        .client
        .request("missing", json!({}))
        .await
        .expect_err("the peer failed");
    assert_eq!(error.kind(), HarnessErrorKind::Protocol);
    assert_eq!(
        error.message(),
        "Memory JSON-RPC error for 'missing': method not found (code -32601)"
    );
}

#[tokio::test]
async fn a_missing_answer_times_out() {
    let fixture = Fixture::new();
    let error = fixture
        .client
        .request_with_timeout("slow", json!({}), Duration::from_millis(50))
        .await
        .expect_err("no answer arrives");
    assert_eq!(error.kind(), HarnessErrorKind::Protocol);
    assert_eq!(error.message(), "Memory did not answer 'slow' within 0.05s");
    assert!(
        !fixture.client.is_closed(),
        "a timeout keeps the client usable"
    );
}

#[tokio::test]
async fn end_of_stream_fails_pending_requests_and_ends_the_stream() {
    let mut fixture = Fixture::with_stderr("fatal: pipe closed");
    let request = {
        let client = fixture.handle();
        tokio::spawn(async move { client.request("ping", json!({})).await })
    };
    let _line = fixture.next_written().await;
    fixture.end();

    let error = request
        .await
        .expect("no panic")
        .expect_err("the stream ended");
    assert_eq!(error.kind(), HarnessErrorKind::Protocol);
    assert!(
        error
            .message()
            .starts_with("the Memory CLI exited before answering 'ping'"),
        "unexpected message: {}",
        error.message()
    );
    assert!(error.message().contains("fatal: pipe closed"));
    assert!(fixture.client.is_closed());
    assert!(incoming(&fixture.client).await.is_none());
    let late = fixture
        .client
        .request("late", json!({}))
        .await
        .expect_err("the client is closed");
    assert_eq!(late.message(), "the Memory CLI closed its JSON-RPC stream");
}

#[tokio::test]
async fn a_malformed_line_is_skipped_and_the_client_stays_usable() {
    let fixture = Fixture::new();
    fixture.send_error("invalid Memory CLI output: not json");
    fixture.send(json!({"id": 1, "result": "ok"}));
    let value = fixture
        .client
        .request("ping", json!({}))
        .await
        .expect("the client survives");
    assert_eq!(value, json!("ok"));
    assert!(!fixture.client.is_closed());
}

#[tokio::test]
async fn notifications_keep_their_order_before_requests() {
    let mut fixture = Fixture::new();
    fixture
        .client
        .notify("telemetry", json!({"n": 1}))
        .await
        .expect("the notification is queued");
    fixture.send(json!({"id": 1, "result": null}));
    fixture
        .client
        .request("ping", json!({}))
        .await
        .expect("the request is answered");

    let notification = fixture.next_written().await;
    assert_eq!(
        notification,
        json!({"method": "telemetry", "params": {"n": 1}})
    );
    assert_eq!(notification.get("id"), None);
    let request = fixture.next_written().await;
    assert_eq!(request["id"], json!(1));
    assert_eq!(request["method"], json!("ping"));
}

#[tokio::test]
async fn close_is_idempotent_and_rejects_later_writes() {
    let fixture = Fixture::new();
    assert!(!fixture.client.is_closed());
    fixture.client.close().await;
    assert!(fixture.client.is_closed());
    fixture.client.close().await;

    let error = fixture
        .client
        .request("late", json!({}))
        .await
        .expect_err("the client is closed");
    assert_eq!(error.message(), "the Memory CLI closed its JSON-RPC stream");
    assert_eq!(
        fixture
            .client
            .notify("late", json!({}))
            .await
            .expect_err("the client is closed")
            .message(),
        "the Memory CLI closed its JSON-RPC stream"
    );
    assert_eq!(fixture.client.label(), LABEL);
    assert!(incoming(&fixture.client).await.is_none());
}

#[tokio::test]
async fn the_stderr_tail_is_read_from_the_idle_transport() {
    let fixture = Fixture::with_stderr("warning: deprecated flag");
    assert_eq!(fixture.client.stderr_tail(), "warning: deprecated flag");
}
