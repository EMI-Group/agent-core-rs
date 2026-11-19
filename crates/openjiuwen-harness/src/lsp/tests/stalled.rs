//! Regression tests for the bounded teardown of a stalled language server.
//!
//! A server that stops reading its stdin used to park requests forever: the enqueue into the
//! outgoing queue happened *before* the request deadline started, the transport write was
//! unbounded, and `Drop`'s `try_send(Close)` could not reach a task blocked on a full pipe, so
//! the child leaked too. These tests pin the fix: the enqueue is bounded, a stalled write tears
//! the connection down, and the transport (and with it a child process) is closed and reaped.
use std::time::Duration;

use serde_json::json;
use tokio::io::AsyncReadExt;

use openjiuwen_common::error::Error;

use crate::lsp::client::{LspClient, OUTGOING_CAPACITY};
use crate::lsp::config::LspServerConfig;
use crate::lsp::transport::{ChannelTransport, LspTransport, StdioTransport};

/// The deadline every stalled-server test uses; small enough to keep the suite fast.
const DEADLINE: Duration = Duration::from_millis(150);

/// How long a test may wait before it declares a hang.
const BOUND: Duration = Duration::from_secs(10);

#[tokio::test]
async fn a_full_outgoing_queue_times_out_and_closes_the_transport() {
    // A tiny duplex buffer plus a peer that never reads: the I/O task's first write blocks
    // inside `write_all`, the queue fills up, and the requests that cannot be handed over must
    // fail with a typed timeout instead of parking on `send`.
    let (transport, mut peer) = ChannelTransport::new(8);
    let client = LspClient::new(Box::new(transport), DEADLINE);

    let requests = (0..(OUTGOING_CAPACITY + 8)).map(|index| {
        client.request(
            "textDocument/hover",
            json!({ "index": index, "padding": "x".repeat(64) }),
        )
    });
    let results = tokio::time::timeout(BOUND, futures::future::join_all(requests))
        .await
        .expect("no request may park on a stalled transport");

    let mut enqueue_overruns = 0_usize;
    for result in results {
        match result {
            Err(Error::Timeout(message)) if message.contains("could not be handed") => {
                enqueue_overruns += 1;
            }
            // An answer that never arrives is the same stall seen from the other side.
            Ok(_) | Err(Error::Timeout(_)) => {}
            Err(other) => panic!("expected typed timeouts only, got {other}"),
        }
    }
    assert!(
        enqueue_overruns > 0,
        "a full outgoing queue must surface as an enqueue timeout"
    );
    assert!(!client.is_alive(), "the stalled client must be closed");
    let reason = client.close_reason().unwrap_or_default();
    assert!(
        reason.contains("could not be handed to the transport"),
        "the close reason must explain the stall: {reason}"
    );
    assert_drained(&mut peer).await;
}

#[tokio::test]
async fn dropping_a_client_with_a_full_queue_still_closes_the_transport() {
    // A long request deadline keeps the awaits pending, so the queue really is full when the
    // client is dropped: `try_send(Close)` fails and the teardown token must reach the blocked
    // I/O task instead.
    let (transport, mut peer) = ChannelTransport::new(8);
    let client = LspClient::new(Box::new(transport), Duration::from_secs(30));
    let requests = (0..(OUTGOING_CAPACITY + 4)).map(|index| {
        client.request(
            "textDocument/hover",
            json!({ "index": index, "padding": "x".repeat(64) }),
        )
    });
    let _ = tokio::time::timeout(
        Duration::from_millis(300),
        futures::future::join_all(requests),
    )
    .await;
    assert!(
        client.is_alive(),
        "the stalled client is still up before the drop"
    );
    drop(client);
    assert_drained(&mut peer).await;
}

/// Reads `peer` until end-of-stream, so a closed transport is proven even when the peer has
/// buffered bytes left from the partial frame.
async fn assert_drained(peer: &mut tokio::io::DuplexStream) {
    let drained = tokio::time::timeout(BOUND, async {
        let mut buffer = [0_u8; 64];
        while let Ok(read) = peer.read(&mut buffer).await {
            if read == 0 {
                break;
            }
        }
    })
    .await;
    assert!(
        drained.is_ok(),
        "the transport must be closed by the client"
    );
}

#[tokio::test]
async fn a_write_that_never_completes_times_out_and_fails_the_pending_request() {
    // The transport accepts the frame but never finishes the write, which is the shape of a
    // blocked `write_all` on a full child stdin pipe.
    let (transport, _peer) = ChannelTransport::new(8);
    let client = LspClient::new(Box::new(StalledTransport::new(transport)), DEADLINE);
    let error = tokio::time::timeout(
        BOUND,
        client.request("textDocument/hover", json!({ "query": "x" })),
    )
    .await
    .expect("the request must not park")
    .unwrap_err();
    assert!(matches!(error, Error::Timeout(_)), "{error}");
    assert!(
        error.to_string().contains("timed out"),
        "the write overrun must be a typed timeout: {error}"
    );
    let closed = tokio::time::timeout(BOUND, async {
        while client.is_alive() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
    assert!(
        closed.is_ok(),
        "the client must stop after the write overrun"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn a_child_server_that_never_reads_is_killed_and_reaped() {
    // `sh -c 'sleep 60'` never reads stdin, so a frame above the pipe buffer blocks the write.
    // The client must time out and the child must be killed and reaped, not leaked.
    let config = LspServerConfig {
        command: "sh".to_owned(),
        args: vec!["-c".to_owned(), "sleep 60".to_owned()],
        ..LspServerConfig::default()
    };
    let transport = StdioTransport::spawn(&config).expect("the test shell must start");
    let pid = transport.process_id().expect("a live child has a pid");
    let client = LspClient::new(Box::new(transport), Duration::from_millis(200));

    let mut timed_out = false;
    for index in 0..(2 * 64) {
        let request = tokio::time::timeout(
            BOUND,
            client.request(
                "textDocument/hover",
                json!({ "index": index, "padding": "x".repeat(64 * 1024) }),
            ),
        )
        .await
        .expect("a request against a deaf server must not park");
        if matches!(request, Err(Error::Timeout(_))) {
            timed_out = true;
            break;
        }
    }
    assert!(timed_out, "a deaf child server must time out the request");

    let exited = tokio::time::timeout(BOUND, async {
        loop {
            #[cfg(target_os = "linux")]
            let gone = !std::path::Path::new(&format!("/proc/{pid}")).exists();
            #[cfg(not(target_os = "linux"))]
            let gone = true;
            if gone && !client.is_alive() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await;
    assert!(
        exited.is_ok(),
        "the child server must be reaped, not leaked"
    );
}

/// A transport whose `send` never completes, wrapping a real channel transport.
struct StalledTransport {
    inner: ChannelTransport,
}

impl StalledTransport {
    fn new(inner: ChannelTransport) -> Self {
        Self { inner }
    }
}

#[async_trait::async_trait]
impl LspTransport for StalledTransport {
    async fn send(&mut self, _message: serde_json::Value) -> openjiuwen_common::error::Result<()> {
        // A write to a full pipe: never completes.
        std::future::pending::<()>().await;
        Ok(())
    }

    async fn recv(&mut self) -> openjiuwen_common::error::Result<Option<serde_json::Value>> {
        self.inner.recv().await
    }

    async fn close(&mut self) -> openjiuwen_common::error::Result<()> {
        self.inner.close().await
    }
}
