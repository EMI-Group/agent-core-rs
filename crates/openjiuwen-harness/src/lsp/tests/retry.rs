//! Supervisor tests: lazy start, one automatic crash retry, deterministic back-off, budget
//! exhaustion and final shutdown.
//!
//! Every test owns a scripted in-process server behind [`ChannelTransport`]; the crash mode
//! drops the stream before answering the first hover so the client reports a dead server. The
//! clock is [`InstantTimer`], so no test waits for a real back-off.
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};
use tokio::io::{AsyncWriteExt, DuplexStream};

use openjiuwen_common::error::{Error, Result};

use crate::lsp::config::LspServerConfig;
use crate::lsp::instance::{
    BackoffTimer, InstantTimer, RestartPolicy, ServerFactory, ServerState, ServerSupervisor,
};
use crate::lsp::protocol::{write_message, MessageReader, Position};
use crate::lsp::transport::{ChannelTransport, LspTransport};

/// Document URI every hover targets.
const DOC: &str = "file:///workspace/src/lib.rs";

/// How the scripted server answers `textDocument/hover`.
#[derive(Clone, Copy, Debug)]
enum HoverMode {
    /// Drop the stream without answering, simulating a crash mid-request.
    Crash,
    /// Answer with a hover result.
    Answer,
    /// Answer with a JSON-RPC error, leaving the client alive.
    Error,
}

/// Reads framed requests from `stream` until the peer goes away.
async fn scripted(stream: DuplexStream, hover: HoverMode) {
    let (read, mut write) = tokio::io::split(stream);
    let mut reader = MessageReader::new(read);
    loop {
        let message = match reader.next().await {
            Ok(Some(message)) => message,
            Ok(None) | Err(_) => break,
        };
        let method = message
            .get("method")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let Some(id) = message.get("id").cloned() else {
            if method == "exit" {
                let _ = write.shutdown().await;
                break;
            }
            continue;
        };
        let response = match method.as_str() {
            "initialize" => json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": { "capabilities": { "hoverProvider": true } },
            }),
            "textDocument/hover" => match hover {
                HoverMode::Crash => break,
                HoverMode::Answer => json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": { "contents": { "kind": "markdown", "value": "second server" } },
                }),
                HoverMode::Error => json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": { "code": -32601, "message": "method not found" },
                }),
            },
            _ => json!({ "jsonrpc": "2.0", "id": id, "result": null }),
        };
        if write_message(&mut write, &response).await.is_err() {
            break;
        }
    }
}

/// Builds a factory that counts starts and scripts each started server by its index.
fn factory(starts: Arc<AtomicUsize>, mode_for: fn(usize) -> HoverMode) -> ServerFactory {
    Arc::new(move |_config: &LspServerConfig| {
        let index = starts.fetch_add(1, Ordering::SeqCst);
        let (transport, peer) = ChannelTransport::new(64 * 1024);
        tokio::spawn(scripted(peer, mode_for(index)));
        Ok(Box::new(transport) as Box<dyn LspTransport>)
    })
}

/// The server definition every test supervises.
fn config() -> LspServerConfig {
    LspServerConfig::new("fake")
        .with_name("fake")
        .with_language_id("rust")
}

/// Builds a supervisor rooted at a fixed fake workspace.
fn supervisor(factory: ServerFactory, timer: Arc<InstantTimer>) -> ServerSupervisor {
    ServerSupervisor::new(config(), factory)
        .with_timer(timer)
        .with_workspace_root(PathBuf::from("/workspace"))
}

#[tokio::test]
async fn run_restarts_after_a_crash_and_retries_the_operation_once() {
    let starts = Arc::new(AtomicUsize::new(0));
    let timer = Arc::new(InstantTimer::new());
    let factory = factory(Arc::clone(&starts), |index| {
        if index == 0 {
            HoverMode::Crash
        } else {
            HoverMode::Answer
        }
    });
    let supervisor = supervisor(factory, Arc::clone(&timer));

    let hover = supervisor
        .run(|client| async move { client.hover(DOC, Position::new(0, 0)).await })
        .await
        .unwrap()
        .expect("the retried hover returns a result");
    assert_eq!(hover.contents, "second server");
    assert_eq!(starts.load(Ordering::SeqCst), 2, "one restart, one retry");
    assert_eq!(
        timer.delays(),
        vec![Duration::from_millis(500)],
        "the first back-off is the policy base delay"
    );

    let status = supervisor.status();
    assert_eq!(status.state, ServerState::Running);
    assert!(status.healthy);
    assert_eq!(status.attempts, 0, "a successful restart resets the budget");
    assert!(status.last_error.is_none());
    assert_eq!(status.key.name, "fake");
    assert_eq!(status.key.root, PathBuf::from("/workspace"));
    assert_eq!(status.language_id, "rust");
    supervisor.shutdown().await;
}

#[tokio::test]
async fn live_server_errors_are_propagated_without_a_restart() {
    let starts = Arc::new(AtomicUsize::new(0));
    let timer = Arc::new(InstantTimer::new());
    let factory = factory(Arc::clone(&starts), |_| HoverMode::Error);
    let supervisor = supervisor(factory, Arc::clone(&timer));

    let error = supervisor
        .run(|client| async move { client.hover(DOC, Position::new(0, 0)).await })
        .await
        .unwrap_err();
    assert!(
        error.message().contains("method not found"),
        "unexpected error: {error}"
    );
    assert_eq!(starts.load(Ordering::SeqCst), 1);
    assert!(timer.delays().is_empty(), "no back-off for a live server");
    assert!(supervisor.status().healthy);
    supervisor.shutdown().await;
}

#[tokio::test]
async fn start_failures_exhaust_the_budget_with_a_typed_error() {
    let invocations = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&invocations);
    let factory: ServerFactory = Arc::new(move |_config: &LspServerConfig| {
        counter.fetch_add(1, Ordering::SeqCst);
        Err(Error::config("boom"))
    });
    let supervisor = ServerSupervisor::new(config(), factory).with_restart_policy(RestartPolicy {
        max_attempts: 1,
        ..RestartPolicy::default()
    });

    let error = supervisor
        .run(|client| async move { client.hover(DOC, Position::new(0, 0)).await })
        .await
        .unwrap_err();
    let message = error.message().into_owned();
    assert!(
        message.contains("exceeded max crash recovery attempts (1)"),
        "unexpected error: {message}"
    );
    assert!(message.contains("boom"), "the last error is reported");

    let status = supervisor.status();
    assert_eq!(status.state, ServerState::Failed);
    assert_eq!(status.attempts, 1);
    assert!(status.last_error.unwrap().contains("boom"));
    assert!(!status.healthy);

    // The budget is sticky: a later call fails without touching the factory again.
    let again = supervisor
        .client()
        .await
        .err()
        .expect("the budget is exhausted");
    assert!(again
        .message()
        .contains("exceeded max crash recovery attempts"));
    assert_eq!(invocations.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn restart_is_forbidden_after_shutdown() {
    let starts = Arc::new(AtomicUsize::new(0));
    let timer = Arc::new(InstantTimer::new());
    let factory = factory(Arc::clone(&starts), |_| HoverMode::Answer);
    let supervisor = supervisor(factory, timer);

    let first = supervisor.client().await.unwrap();
    assert_eq!(starts.load(Ordering::SeqCst), 1);
    supervisor.restart().await.unwrap();
    assert_eq!(
        starts.load(Ordering::SeqCst),
        2,
        "restart starts a new server"
    );
    let second = supervisor.client().await.unwrap();
    assert!(!Arc::ptr_eq(&first, &second));

    supervisor.shutdown().await;
    assert_eq!(supervisor.status().state, ServerState::Stopped);
    let error = supervisor
        .run(|client| async move { client.hover(DOC, Position::new(0, 0)).await })
        .await
        .unwrap_err();
    assert!(
        error.message().contains("has been shut down"),
        "unexpected error: {error}"
    );
    let restart = supervisor.restart().await.unwrap_err();
    assert!(restart.message().contains("has been shut down"));
    let client = supervisor
        .client()
        .await
        .err()
        .expect("the supervisor is shut down");
    assert!(client.message().contains("has been shut down"));
    assert_eq!(
        starts.load(Ordering::SeqCst),
        2,
        "a shut-down supervisor never starts another process"
    );
}

#[test]
fn restart_policy_backoff_doubles_and_is_capped() {
    let policy = RestartPolicy::default();
    assert_eq!(policy.max_attempts, 3);
    assert_eq!(policy.base_delay, Duration::from_millis(500));
    assert_eq!(policy.max_delay, Duration::from_secs(8));
    assert_eq!(policy.delay_for(1), Duration::from_millis(500));
    assert_eq!(policy.delay_for(2), Duration::from_secs(1));
    assert_eq!(policy.delay_for(3), Duration::from_secs(2));
    assert_eq!(policy.delay_for(4), Duration::from_secs(4));
    assert_eq!(
        policy.delay_for(20),
        Duration::from_secs(8),
        "the delay is capped by max_delay"
    );
    assert_eq!(
        policy.delay_for(0),
        Duration::from_millis(500),
        "attempt 0 is treated as the first attempt"
    );

    let capped = RestartPolicy {
        max_attempts: 5,
        base_delay: Duration::from_secs(1),
        max_delay: Duration::from_secs(2),
    };
    assert_eq!(capped.delay_for(1), Duration::from_secs(1));
    assert_eq!(capped.delay_for(2), Duration::from_secs(2));
    assert_eq!(capped.delay_for(3), Duration::from_secs(2));
}

#[tokio::test]
async fn instant_timer_is_a_deterministic_clock() {
    let timer = InstantTimer::new();
    timer.sleep(Duration::from_millis(500)).await;
    timer.sleep(Duration::from_secs(1)).await;
    assert_eq!(
        timer.delays(),
        vec![Duration::from_millis(500), Duration::from_secs(1)]
    );
}

/// Compiles the public supervisor surface with its documented types.
#[tokio::test]
async fn supervisor_exposes_its_config_and_key() -> Result<()> {
    let starts = Arc::new(AtomicUsize::new(0));
    let factory = factory(starts, |_| HoverMode::Answer);
    let supervisor = supervisor(factory, Arc::new(InstantTimer::new()));
    assert_eq!(supervisor.config().command, "fake");
    assert_eq!(supervisor.key().name, "fake");
    assert_eq!(supervisor.attempts(), 0);
    assert_eq!(supervisor.status().state, ServerState::Stopped);
    supervisor.shutdown().await;
    Ok(())
}
