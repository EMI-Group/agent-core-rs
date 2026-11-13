//! Client tests against the in-process fake server.
//!
//! Covers initialize capability capture, server-initiated request answers, typed navigation
//! methods, notification ordering and versions, unknown-method mapping, per-request timeouts,
//! malformed frames, clean EOF, and shutdown.
use std::time::Duration;

use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use openjiuwen_common::error::Error;

use crate::lsp::client::LspClient;
use crate::lsp::protocol::{client_capabilities, Position};
use crate::lsp::transport::ChannelTransport;

use super::fake_server::{self, FakeServer};

/// The root URI every test initializes against.
const ROOT: &str = "file:///workspace";

/// Builds a client connected to a fresh fake server.
fn connected(timeout: Duration) -> (LspClient, FakeServer) {
    let (transport, peer) = ChannelTransport::new(64 * 1024);
    let client = LspClient::new(Box::new(transport), timeout);
    let server = fake_server::spawn(peer);
    (client, server)
}

#[tokio::test]
async fn initialize_captures_capabilities_and_answers_server_requests() {
    let (client, server) = connected(Duration::from_secs(2));
    let capabilities = client
        .initialize(ROOT, client_capabilities())
        .await
        .unwrap();
    assert!(client.is_initialized());
    assert!(capabilities.hover);
    assert!(capabilities.definition);
    assert!(capabilities.references);
    assert!(capabilities.completion);
    assert!(capabilities.document_symbol);
    assert!(capabilities.workspace_symbol);
    assert!(!capabilities.diagnostics);
    assert_eq!(capabilities.raw["hoverProvider"], json!(true));
    assert!(capabilities.supports("textDocument/hover"));
    assert!(capabilities.supports("unknown/method"));

    let messages = server.messages();
    // The fake sent `workspace/configuration` and waited; the client must have answered `[]`.
    let answer = messages
        .iter()
        .find(|message| message.get("id") == Some(&json!(9001)))
        .expect("client must answer the server's workspace/configuration request");
    assert_eq!(answer["result"], json!([]));
    assert_eq!(
        client.capabilities().map(|caps| caps.hover),
        Some(true),
        "capabilities are cached on the client"
    );
    client.shutdown().await;
}

#[tokio::test]
async fn typed_navigation_methods_normalise_results() {
    let (client, _server) = connected(Duration::from_secs(2));
    client.initialize(ROOT, json!({})).await.unwrap();
    let uri = "file:///workspace/src/lib.rs";
    let position = Position::new(1, 4);

    let hover = client.hover(uri, position).await.unwrap().unwrap();
    assert!(hover.contents.contains("fake hover"));
    assert_eq!(hover.range.unwrap().start.line, 1);

    let definitions = client.definition(uri, position).await.unwrap();
    assert_eq!(definitions.len(), 1);
    assert_eq!(definitions[0].uri, uri);
    assert_eq!(definitions[0].range.start, Position::new(2, 4));

    let references = client.references(uri, position, true).await.unwrap();
    assert_eq!(references.len(), 2);
    assert_eq!(references[1].range.start, Position::new(3, 4));

    let completion = client.completion(uri, position).await.unwrap();
    assert_eq!(completion.len(), 2);
    assert_eq!(completion[0].label, "alpha");
    assert_eq!(completion[0].kind, Some(3));
    assert_eq!(completion[0].detail.as_deref(), Some("fake detail"));
    assert_eq!(completion[1].label, "beta");

    let symbols = client.document_symbols(uri).await.unwrap();
    assert_eq!(symbols.len(), 1);
    assert_eq!(symbols[0].name, "main");
    assert_eq!(symbols[0].detail.as_deref(), Some("fn main()"));
    assert_eq!(symbols[0].children.len(), 1);
    assert_eq!(symbols[0].children[0].name, "helper");

    let workspace = client.workspace_symbols("main").await.unwrap();
    assert_eq!(workspace.len(), 1);
    assert_eq!(workspace[0].container_name.as_deref(), Some("crate"));
    client.shutdown().await;
}

#[tokio::test]
async fn did_open_is_sent_before_the_request_it_prepares() {
    let (client, server) = connected(Duration::from_secs(2));
    client.initialize(ROOT, json!({})).await.unwrap();
    let uri = "file:///workspace/src/lib.rs";
    client
        .did_open(uri, "rust", "fn main() {}\n")
        .await
        .unwrap();
    client.hover(uri, Position::new(0, 3)).await.unwrap();

    let opened = server
        .position_of("textDocument/didOpen")
        .expect("didOpen must be sent");
    let hover = server
        .position_of("textDocument/hover")
        .expect("hover must be sent");
    assert!(opened < hover, "didOpen must precede the hover request");

    let open = server
        .messages()
        .into_iter()
        .find(|message| message.get("method") == Some(&json!("textDocument/didOpen")))
        .unwrap();
    assert_eq!(open["params"]["textDocument"]["languageId"], json!("rust"));
    assert_eq!(open["params"]["textDocument"]["version"], json!(1));
    client.shutdown().await;
}

#[tokio::test]
async fn did_open_twice_degrades_to_did_change_and_bumps_the_version() {
    let (client, server) = connected(Duration::from_secs(2));
    client.initialize(ROOT, json!({})).await.unwrap();
    let uri = "file:///workspace/src/lib.rs";
    client.did_open(uri, "rust", "one\n").await.unwrap();
    client.did_change(uri, "two\n").await.unwrap();
    client.did_open(uri, "rust", "three\n").await.unwrap();
    assert!(
        server.wait_for_count("textDocument/didChange", 2).await,
        "both changes must reach the server"
    );

    let changes: Vec<serde_json::Value> = server
        .messages()
        .into_iter()
        .filter(|message| message.get("method") == Some(&json!("textDocument/didChange")))
        .collect();
    assert_eq!(changes.len(), 2);
    assert_eq!(changes[0]["params"]["textDocument"]["version"], json!(2));
    assert_eq!(changes[1]["params"]["textDocument"]["version"], json!(3));
    assert_eq!(
        changes[1]["params"]["contentChanges"][0]["text"],
        json!("three\n")
    );
    client.shutdown().await;
}

#[tokio::test]
async fn unknown_methods_map_to_a_clear_error() {
    let (client, _server) = connected(Duration::from_secs(2));
    client.initialize(ROOT, json!({})).await.unwrap();
    let error = client
        .request("textDocument/unknown", json!({}))
        .await
        .unwrap_err();
    assert!(error.to_string().contains("-32601"), "{error}");
    assert!(error.to_string().contains("method not found"), "{error}");
    client.shutdown().await;
}

#[tokio::test]
async fn a_slow_response_times_out() {
    let (client, _server) = connected(Duration::from_millis(100));
    client.initialize(ROOT, json!({})).await.unwrap();
    let error = client
        .definition("file:///workspace/src/slow.rs", Position::new(0, 0))
        .await
        .unwrap_err();
    assert!(matches!(error, Error::Timeout(_)), "{error}");
    assert!(error.to_string().contains("timed out"), "{error}");
    client.shutdown().await;
}

#[tokio::test]
async fn a_malformed_frame_fails_the_pending_request() {
    let (transport, mut peer) = ChannelTransport::new(256);
    let client = LspClient::new(Box::new(transport), Duration::from_secs(2));
    peer.write_all(b"Content-Length: nope\r\n\r\n")
        .await
        .unwrap();
    let error = client.request("initialize", json!({})).await.unwrap_err();
    assert!(error.to_string().contains("Content-Length"), "{error}");
}

#[tokio::test]
async fn eof_fails_the_pending_request() {
    let (transport, mut peer) = ChannelTransport::new(256);
    let client = LspClient::new(Box::new(transport), Duration::from_secs(2));
    tokio::spawn(async move {
        let mut buffer = [0_u8; 256];
        let _ = peer.read(&mut buffer).await;
        drop(peer);
    });
    let error = client.request("initialize", json!({})).await.unwrap_err();
    assert!(
        error.to_string().contains("closed the connection"),
        "{error}"
    );
}

#[tokio::test]
async fn dropping_the_client_closes_the_transport() {
    let (transport, mut peer) = ChannelTransport::new(256);
    let client = LspClient::new(Box::new(transport), Duration::from_secs(2));
    drop(client);
    let mut buffer = [0_u8; 8];
    let read = tokio::time::timeout(Duration::from_secs(2), peer.read(&mut buffer))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(read, 0, "dropping the client must close the peer stream");
}

#[tokio::test]
async fn shutdown_closes_the_client() {
    let (client, server) = connected(Duration::from_secs(2));
    client.initialize(ROOT, json!({})).await.unwrap();
    client.shutdown().await;
    assert!(
        server.position_of("shutdown").is_some(),
        "shutdown request must be sent"
    );
    assert!(
        server.wait_for("exit").await,
        "exit notification must be sent"
    );
    server.join().await;
    let error = client.request("textDocument/hover", json!({})).await;
    assert!(error.is_err(), "a shut-down client must reject requests");
}
