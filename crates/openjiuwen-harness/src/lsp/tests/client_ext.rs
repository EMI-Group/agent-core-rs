//! Client tests for the implementation, call-hierarchy, pull-diagnostic, notification and
//! liveness surface.
//!
//! Everything runs against the in-process fake server: exact method names and parameter
//! shapes, capability capture, handler dispatch order and panic isolation, server death and
//! explicit shutdown.
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use serde_json::{json, Value};

use openjiuwen_common::error::Error;

use crate::lsp::client::LspClient;
use crate::lsp::protocol::{client_capabilities, write_message, MessageReader, Position, Range};
use crate::lsp::transport::ChannelTransport;
use crate::lsp::wire::{
    parse_diagnostics_event, CallHierarchyItem, DiagnosticCode, DiagnosticReportKind,
    DiagnosticsEvent,
};

use super::fake_server::{self, FakeServer};

/// The root URI every test initializes against.
const ROOT: &str = "file:///workspace";
/// The document URI every test targets.
const DOC: &str = "file:///workspace/src/lib.rs";

/// Builds a client connected to a fresh fake server.
fn connected(timeout: Duration) -> (LspClient, FakeServer) {
    let (transport, peer) = ChannelTransport::new(64 * 1024);
    let client = LspClient::new(Box::new(transport), timeout);
    let server = fake_server::spawn(peer);
    (client, server)
}

/// Returns the last message the fake server received for `method`.
fn request_with_method(server: &FakeServer, method: &str) -> Value {
    server
        .messages()
        .into_iter()
        .rev()
        .find(|message| message.get("method").and_then(Value::as_str) == Some(method))
        .unwrap_or_else(|| panic!("the client must send '{method}'"))
}

/// Polls a condition for at most one second.
async fn eventually(mut condition: impl FnMut() -> bool) -> bool {
    for _ in 0..200 {
        if condition() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    false
}

#[tokio::test]
async fn initialize_captures_implementation_call_hierarchy_and_diagnostics() {
    let (client, _server) = connected(Duration::from_secs(2));
    let advertised = client_capabilities();
    let capabilities = client.initialize(ROOT, advertised.clone()).await.unwrap();
    assert!(capabilities.implementation);
    assert!(capabilities.call_hierarchy);
    assert!(capabilities.diagnostics);
    assert!(capabilities.supports("textDocument/implementation"));
    assert!(capabilities.supports("textDocument/prepareCallHierarchy"));
    assert!(capabilities.supports("callHierarchy/incomingCalls"));
    assert!(capabilities.supports("callHierarchy/outgoingCalls"));
    assert!(capabilities.supports("textDocument/diagnostic"));

    assert_eq!(
        advertised["textDocument"]["implementation"],
        json!({ "dynamicRegistration": false })
    );
    assert_eq!(
        advertised["textDocument"]["callHierarchy"],
        json!({ "dynamicRegistration": false })
    );
    assert_eq!(
        advertised["textDocument"]["diagnostic"],
        json!({
            "dynamicRegistration": false,
            "relatedDocumentSupport": true,
            "versionSupport": true,
        })
    );
    client.shutdown().await;
}

#[tokio::test]
async fn implementation_normalises_location_links_and_sends_exact_params() {
    let (client, server) = connected(Duration::from_secs(2));
    client.initialize(ROOT, json!({})).await.unwrap();

    let locations = client
        .implementation(DOC, Position::new(3, 7))
        .await
        .unwrap();
    assert_eq!(locations.len(), 1);
    assert_eq!(locations[0].uri, DOC);
    assert_eq!(locations[0].range.start, Position::new(5, 0));

    let request = request_with_method(&server, "textDocument/implementation");
    assert_eq!(
        request["params"],
        json!({
            "textDocument": { "uri": DOC },
            "position": { "line": 3, "character": 7 },
        })
    );
    client.shutdown().await;
}

#[tokio::test]
async fn prepare_call_hierarchy_flattens_flat_and_nested_items() {
    let (client, server) = connected(Duration::from_secs(2));
    client.initialize(ROOT, json!({})).await.unwrap();

    let items = client
        .prepare_call_hierarchy(DOC, Position::new(1, 4))
        .await
        .unwrap();
    assert_eq!(items.len(), 2);
    assert_eq!(items[0].name, "caller");
    assert_eq!(items[0].uri, "file:///workspace/src/caller.rs");
    assert_eq!(items[0].detail.as_deref(), Some("fn caller()"));
    assert_eq!(items[0].selection_range.start, Position::new(1, 3));
    assert_eq!(items[0].data, None);
    assert_eq!(items[1].name, "callee");
    assert_eq!(items[1].uri, "file:///workspace/src/callee.rs");
    assert_eq!(items[1].data, Some(json!({ "id": 7 })));

    let request = request_with_method(&server, "textDocument/prepareCallHierarchy");
    assert_eq!(
        request["params"],
        json!({
            "textDocument": { "uri": DOC },
            "position": { "line": 1, "character": 4 },
        })
    );
    client.shutdown().await;
}

#[tokio::test]
async fn incoming_calls_send_a_flat_item_payload() {
    let (client, server) = connected(Duration::from_secs(2));
    client.initialize(ROOT, json!({})).await.unwrap();

    let item = CallHierarchyItem {
        name: "target".to_owned(),
        kind: 12,
        detail: Some("fn target()".to_owned()),
        uri: "file:///workspace/src/target.rs".to_owned(),
        range: Range {
            start: Position::new(1, 0),
            end: Position::new(2, 0),
        },
        selection_range: Range {
            start: Position::new(1, 3),
            end: Position::new(1, 9),
        },
        tags: vec![1],
        data: Some(json!({ "server": "fake" })),
    };
    let calls = client.incoming_calls(&item).await.unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].from.name, "caller");
    assert_eq!(calls[0].from.uri, "file:///workspace/src/caller.rs");
    assert_eq!(
        calls[0].from_ranges,
        vec![Range {
            start: Position::new(2, 4),
            end: Position::new(2, 6),
        }]
    );

    let request = request_with_method(&server, "callHierarchy/incomingCalls");
    let payload = &request["params"]["item"];
    assert_eq!(payload["name"], json!("target"));
    assert_eq!(payload["kind"], json!(12));
    assert_eq!(payload["uri"], json!("file:///workspace/src/target.rs"));
    assert_eq!(
        payload["selectionRange"]["start"],
        json!({ "line": 1, "character": 3 })
    );
    assert_eq!(payload["data"], json!({ "server": "fake" }));
    assert!(
        payload.get("textDocument").is_none(),
        "the normalised item is sent flat, not nested"
    );
    client.shutdown().await;
}

#[tokio::test]
async fn outgoing_calls_send_the_item_and_parse_from_ranges() {
    let (client, server) = connected(Duration::from_secs(2));
    client.initialize(ROOT, json!({})).await.unwrap();

    let item = CallHierarchyItem {
        name: "main".to_owned(),
        kind: 12,
        detail: None,
        uri: DOC.to_owned(),
        range: Range::default(),
        selection_range: Range::default(),
        tags: Vec::new(),
        data: None,
    };
    let calls = client.outgoing_calls(&item).await.unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].to.name, "callee");
    assert_eq!(calls[0].to.uri, "file:///workspace/src/callee.rs");
    assert_eq!(calls[0].from_ranges.len(), 1);

    let request = request_with_method(&server, "callHierarchy/outgoingCalls");
    let payload = &request["params"]["item"];
    assert_eq!(payload["name"], json!("main"));
    assert_eq!(payload["uri"], json!(DOC));
    assert_eq!(
        payload["range"]["start"],
        json!({ "line": 0, "character": 0 })
    );
    assert!(
        payload.get("data").is_none(),
        "absent `data` is not sent back"
    );
    client.shutdown().await;
}

#[tokio::test]
async fn document_diagnostic_passes_previous_result_ids() {
    let (client, server) = connected(Duration::from_secs(2));
    client.initialize(ROOT, json!({})).await.unwrap();

    let first = client.document_diagnostic(DOC, None).await.unwrap();
    assert_eq!(first.kind, DiagnosticReportKind::Full);
    assert_eq!(first.result_id.as_deref(), Some("r1"));
    assert_eq!(first.items.len(), 1);
    assert_eq!(first.items[0].message, "cannot find name");
    assert_eq!(first.items[0].severity, Some(1));
    assert_eq!(first.items[0].code, Some(DiagnosticCode::Number(2304)));
    assert_eq!(first.items[0].source.as_deref(), Some("fake"));
    let request = request_with_method(&server, "textDocument/diagnostic");
    assert_eq!(
        request["params"],
        json!({ "textDocument": { "uri": DOC } }),
        "no previousResultId is sent without a cached report"
    );

    let unchanged = client.document_diagnostic(DOC, Some("r1")).await.unwrap();
    assert_eq!(unchanged.kind, DiagnosticReportKind::Unchanged);
    assert_eq!(unchanged.result_id.as_deref(), Some("r1"));
    assert_eq!(unchanged.items, Vec::new());
    let request = request_with_method(&server, "textDocument/diagnostic");
    assert_eq!(
        request["params"],
        json!({ "textDocument": { "uri": DOC }, "previousResultId": "r1" })
    );

    let refreshed = client.document_diagnostic(DOC, Some("r0")).await.unwrap();
    assert_eq!(refreshed.kind, DiagnosticReportKind::Full);
    assert_eq!(refreshed.result_id.as_deref(), Some("r2"));
    assert_eq!(refreshed.items.len(), 1);
    assert_eq!(refreshed.items[0].message, "stale report replaced");
    client.shutdown().await;
}

#[tokio::test]
async fn publish_diagnostics_reaches_handlers_in_registration_order() {
    let (client, _server) = connected(Duration::from_secs(2));
    let order = Arc::new(Mutex::new(Vec::new()));
    let received = Arc::new(Mutex::new(Vec::new()));

    let log = Arc::clone(&order);
    let captured = Arc::clone(&received);
    client.add_notification_handler(
        "textDocument/publishDiagnostics",
        Arc::new(move |params: &Value| {
            log.lock().push("first");
            captured.lock().push(params.clone());
        }),
    );
    let log = Arc::clone(&order);
    client.add_notification_handler(
        "textDocument/publishDiagnostics",
        Arc::new(move |_: &Value| log.lock().push("second")),
    );
    // Registered before `initialize` on purpose: the seam must be live at any time.
    client.initialize(ROOT, json!({})).await.unwrap();

    client
        .did_open(DOC, "rust", "fn main() {}\n")
        .await
        .unwrap();
    assert!(
        eventually(|| order.lock().len() == 2).await,
        "both handlers must run for one notification"
    );
    assert_eq!(*order.lock(), vec!["first", "second"]);

    let events: Vec<DiagnosticsEvent> = received
        .lock()
        .iter()
        .filter_map(parse_diagnostics_event)
        .collect();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].uri, DOC);
    assert_eq!(events[0].version, Some(1));
    assert_eq!(events[0].diagnostics.len(), 1);
    assert_eq!(events[0].diagnostics[0].message, "fake warning");
    assert_eq!(events[0].diagnostics[0].severity, Some(2));

    // A change whose text contains `clean` resolves the diagnostics.
    client.did_change(DOC, "clean file\n").await.unwrap();
    assert!(
        eventually(|| order.lock().len() == 4).await,
        "the empty-diagnostics notification must be dispatched too"
    );
    let events: Vec<DiagnosticsEvent> = received
        .lock()
        .iter()
        .filter_map(parse_diagnostics_event)
        .collect();
    assert_eq!(events.len(), 2);
    assert_eq!(events[1].version, Some(2));
    assert_eq!(events[1].diagnostics, Vec::new());
    client.shutdown().await;
}

#[tokio::test]
async fn unregistered_notifications_stay_ignored() {
    let (client, _server) = connected(Duration::from_secs(2));
    let seen = Arc::new(Mutex::new(0_usize));
    let counter = Arc::clone(&seen);
    client.add_notification_handler(
        "custom/neverSent",
        Arc::new(move |_: &Value| {
            *counter.lock() += 1;
        }),
    );
    client.initialize(ROOT, json!({})).await.unwrap();
    // `didOpen` publishes diagnostics, but nothing listens for that method, so the
    // `custom/neverSent` handler must not fire and the notification stays ignored.
    client
        .did_open(DOC, "rust", "fn main() {}\n")
        .await
        .unwrap();
    assert!(client
        .hover(DOC, Position::new(0, 0))
        .await
        .unwrap()
        .is_some());
    assert_eq!(*seen.lock(), 0);
    // The client survives and keeps answering requests after handling notifications.
    assert!(client.is_alive());
    client.shutdown().await;
}

#[tokio::test]
async fn a_panicking_handler_does_not_kill_the_io_task() {
    let (client, _server) = connected(Duration::from_secs(2));
    client.add_notification_handler(
        "textDocument/publishDiagnostics",
        Arc::new(|_: &Value| panic!("handler blew up")),
    );
    let seen = Arc::new(Mutex::new(0_usize));
    let counter = Arc::clone(&seen);
    client.add_notification_handler(
        "textDocument/publishDiagnostics",
        Arc::new(move |_: &Value| *counter.lock() += 1),
    );
    client.initialize(ROOT, json!({})).await.unwrap();
    client
        .did_open(DOC, "rust", "fn main() {}\n")
        .await
        .unwrap();
    assert!(
        eventually(|| *seen.lock() == 1).await,
        "the handler after the panicking one must still run"
    );
    assert!(
        client.is_alive(),
        "a panicking handler must not close the client"
    );
    assert!(client.hover(DOC, Position::new(0, 0)).await.is_ok());
    client.shutdown().await;
}

#[tokio::test]
async fn a_dead_server_fails_pending_requests_and_marks_the_client_closed() {
    let (transport, peer) = ChannelTransport::new(64 * 1024);
    let client = LspClient::new(Box::new(transport), Duration::from_secs(2));
    tokio::spawn(async move {
        let (read, mut write) = tokio::io::split(peer);
        let mut reader = MessageReader::new(read);
        while let Ok(Some(message)) = reader.next().await {
            let Some(id) = message.get("id").cloned() else {
                continue;
            };
            if message.get("method").and_then(Value::as_str) != Some("initialize") {
                // Leave the request unanswered and drop the stream: the server "crashed".
                return;
            }
            let reply = json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": { "capabilities": {} },
            });
            if write_message(&mut write, &reply).await.is_err() {
                return;
            }
        }
    });
    client.initialize(ROOT, json!({})).await.unwrap();
    assert!(client.is_alive());
    assert_eq!(client.close_reason(), None);
    let mut closed = client.closed();

    let error = client.hover(DOC, Position::new(0, 0)).await.unwrap_err();
    assert!(matches!(error, Error::Transport(_)), "{error}");
    assert!(
        error.to_string().contains("closed the connection"),
        "{error}"
    );

    let changed = tokio::time::timeout(Duration::from_secs(2), closed.changed()).await;
    assert!(changed.is_ok(), "the closed watch must flip");
    assert!(*closed.borrow());
    assert!(!client.is_alive());
    let reason = client.close_reason().unwrap_or_default();
    assert!(reason.contains("closed the connection"), "{reason}");
}

#[tokio::test]
async fn shutdown_marks_the_client_closed_with_a_shutdown_reason() {
    let (client, server) = connected(Duration::from_secs(2));
    client.initialize(ROOT, json!({})).await.unwrap();
    assert!(client.is_alive());
    assert_eq!(client.close_reason(), None);

    client.shutdown().await;
    assert!(!client.is_alive());
    let reason = client.close_reason().unwrap_or_default();
    assert!(reason.contains("shutdown"), "{reason}");
    assert!(*client.closed().borrow());
    assert!(server.position_of("shutdown").is_some());
}
