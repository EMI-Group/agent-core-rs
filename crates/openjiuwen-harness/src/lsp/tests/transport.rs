//! Transport seam tests: the in-memory channel and the process transport.
//!
//! The process test uses `/bin/cat` as an echo server when it exists, so it never depends on
//! a real language server being installed (and skips on platforms without `/bin/cat`).
use std::path::Path;
use std::time::Duration;

use serde_json::json;
use tokio::io::AsyncWriteExt;

use crate::lsp::config::LspServerConfig;
use crate::lsp::protocol::{write_message, MessageReader};
use crate::lsp::transport::{ChannelTransport, LspTransport, StdioTransport};

#[tokio::test]
async fn channel_transport_round_trips_a_message() {
    let (mut transport, peer) = ChannelTransport::new(4096);
    let peer_task = tokio::spawn(async move {
        let (read, mut write) = tokio::io::split(peer);
        let mut reader = MessageReader::new(read);
        let message = reader.next().await.unwrap().unwrap();
        write_message(&mut write, &message).await.unwrap();
    });
    transport
        .send(json!({"jsonrpc": "2.0", "method": "ping", "params": {}}))
        .await
        .unwrap();
    let echo = transport.recv().await.unwrap().unwrap();
    assert_eq!(echo["method"], "ping");
    peer_task.await.unwrap();
}

#[tokio::test]
async fn channel_transport_reports_clean_eof() {
    let (mut transport, peer) = ChannelTransport::new(64);
    drop(peer);
    assert!(transport.recv().await.unwrap().is_none());
}

#[tokio::test]
async fn channel_transport_close_shuts_down_the_writer() {
    let (mut transport, peer) = ChannelTransport::new(64);
    drop(peer);
    transport.close().await.unwrap();
}

#[tokio::test]
async fn stdio_transport_echoes_through_cat() {
    let Some(cat) = ["/bin/cat", "/usr/bin/cat"]
        .into_iter()
        .find(|candidate| Path::new(candidate).exists())
    else {
        return;
    };
    let mut transport = StdioTransport::spawn(&LspServerConfig::new(cat)).unwrap();
    transport
        .send(json!({"jsonrpc": "2.0", "method": "ping", "params": {}}))
        .await
        .unwrap();
    let echo = tokio::time::timeout(Duration::from_secs(5), transport.recv())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(echo["method"], "ping");
    transport.close().await.unwrap();
}

#[tokio::test]
async fn stdio_transport_reports_a_missing_command() {
    let config = LspServerConfig::new("openjiuwen-definitely-not-a-real-lsp-binary");
    let error = match StdioTransport::spawn(&config) {
        Ok(_) => panic!("spawning a missing binary should fail"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("failed to start LSP server"),
        "{error}"
    );
}

#[tokio::test]
async fn stdio_transport_rejects_an_empty_command() {
    let error = match StdioTransport::spawn(&LspServerConfig::default()) {
        Ok(_) => panic!("an empty command should fail"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("empty"), "{error}");
}

#[tokio::test]
async fn stdio_transport_uses_the_configured_environment_and_root() {
    let Some(cat) = ["/bin/cat", "/usr/bin/cat"]
        .into_iter()
        .find(|candidate| Path::new(candidate).exists())
    else {
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let config = LspServerConfig::new(cat)
        .with_root(dir.path())
        .with_env("OPENJIUWEN_LSP_FAKE", "1");
    let mut transport = StdioTransport::spawn(&config).unwrap();
    assert!(transport.process_id().is_some());
    transport
        .send(json!({"jsonrpc": "2.0", "method": "ping"}))
        .await
        .unwrap();
    let echo = tokio::time::timeout(Duration::from_secs(5), transport.recv())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(echo["method"], "ping");
    transport.close().await.unwrap();
}

#[tokio::test]
async fn channel_transport_reports_a_malformed_frame() {
    let (mut transport, mut peer) = ChannelTransport::new(256);
    peer.write_all(b"Content-Length: nope\r\n\r\n")
        .await
        .unwrap();
    let error = transport.recv().await.unwrap_err();
    assert!(error.to_string().contains("Content-Length"), "{error}");
}
