//! Interruption and bounded-output tests for the streamed shell run.
//!
//! Python parity: `execute_cmd_stream` in `local/shell_operation.py` forwards the child's
//! output while a timeout or a cancellation kills its process tree. Two failure modes are
//! guarded here:
//!
//! 1. A child that closes its stdout/stderr pipes but keeps running must still be killed by
//!    the timeout (and must still be cancellable). The driver keeps `child.wait()` in the
//!    same `select!` as the deadline, so the deadline does not become unreachable once both
//!    readers report EOF.
//! 2. A child that emits one enormous line must cost the per-stream budget, not the line:
//!    the stream is read in fixed-size chunks under backpressure and stops at
//!    `max_output_bytes` with the truncation marker.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::StreamExt;
use openjiuwen_sysop::config::LocalWorkConfig;
use openjiuwen_sysop::operation::ShellOperation;
use openjiuwen_sysop::shell::types::{ShellChunk, ShellExecParams};
use openjiuwen_sysop::shell::LocalShellOperation;
use tokio_util::sync::CancellationToken;

/// The Python status code of a shell failure (timeout, cancellation).
const SHELL_ERROR: i32 = 199_004;
/// Upper bound on every wait in this file; a regression would otherwise hang the suite.
const BOUND: Duration = Duration::from_secs(10);
/// Per-stream capture budget used by the bounded-output test.
const OUTPUT_BUDGET: u64 = 4_096;
/// The marker appended when a captured stream hits its budget.
const TRUNCATION_MARKER: &str = "\n[output truncated]";

/// Returns `true` when `sh` is available; otherwise prints a skip notice.
fn require_sh() -> bool {
    if which::which("sh").is_ok() {
        return true;
    }
    eprintln!("skipping streamed-interrupt test: `sh` is not on PATH");
    false
}

/// A command that closes both of its own output pipes and then keeps running.
///
/// `sleep 30` outlives the test by far, so only the deadline (or a cancellation) can end
/// the run; both pipes reach EOF immediately, which is the state that used to bypass the
/// timeout.
const CLOSED_PIPE_COMMAND: &str = "exec >/dev/null 2>&1; sleep 30";

/// A fresh temporary workspace.
fn workspace() -> tempfile::TempDir {
    tempfile::tempdir().unwrap()
}

/// The unrestricted local backend rooted at `dir` (no allowlist, no size ceiling).
fn backend(dir: &tempfile::TempDir) -> LocalShellOperation {
    LocalShellOperation::new(LocalWorkConfig::unrestricted(
        dir.path().to_string_lossy().into_owned(),
    ))
    .expect("the unrestricted local configuration is valid")
}

/// Drains `stream` under [`BOUND`], failing the test instead of hanging it.
async fn drain(
    stream: openjiuwen_sysop::operation::OpStream<ShellChunk>,
) -> Vec<openjiuwen_sysop::result::OpResult<ShellChunk>> {
    tokio::time::timeout(BOUND, stream.collect::<Vec<_>>())
        .await
        .expect("the streamed run must end within the test's bound")
}

#[tokio::test]
async fn timeout_reaches_a_child_that_closed_its_pipes() {
    if !require_sh() {
        return;
    }
    let dir = workspace();
    let op = Arc::new(backend(&dir));
    let params = ShellExecParams::new(CLOSED_PIPE_COMMAND).with_timeout_secs(1);

    let started = Instant::now();
    let items = drain(op.execute_cmd_stream(params)).await;
    let elapsed = started.elapsed();

    let failure = items.last().expect("a terminal item");
    assert_eq!(failure.code, SHELL_ERROR, "{}", failure.message);
    assert!(
        failure.message.contains("timeout"),
        "the run must report a timeout, got: {}",
        failure.message
    );
    assert!(
        elapsed < BOUND,
        "the deadline must survive the pipes reaching EOF, took {elapsed:?}"
    );
    // No stray data events: the child redirected its output before the first read.
    for item in &items {
        if let Some(chunk) = item.data.as_ref() {
            assert!(
                chunk.text.trim().is_empty(),
                "unexpected output chunk: {:?}",
                chunk.text
            );
        }
    }
}

#[tokio::test]
async fn cancellation_reaches_a_child_that_closed_its_pipes() {
    if !require_sh() {
        return;
    }
    let dir = workspace();
    let token = CancellationToken::new();
    let op = Arc::new(backend(&dir).with_cancellation(token.clone()));

    let started = Instant::now();
    let handle = {
        let op = Arc::clone(&op);
        tokio::spawn(async move {
            drain(op.execute_cmd_stream(ShellExecParams::new(CLOSED_PIPE_COMMAND))).await
        })
    };
    tokio::time::sleep(Duration::from_millis(200)).await;
    token.cancel();
    let items = handle.await.unwrap();

    let failure = items.last().expect("a terminal item");
    assert_eq!(failure.code, SHELL_ERROR, "{}", failure.message);
    assert!(
        failure.message.contains("cancelled"),
        "the run must report a cancellation, got: {}",
        failure.message
    );
    assert!(
        started.elapsed() < BOUND,
        "a cancelled run must not wait for the child to exit, took {:?}",
        started.elapsed()
    );
}

#[tokio::test]
async fn a_single_huge_line_is_captured_within_the_output_budget() {
    if !require_sh() || which::which("head").is_err() || which::which("tr").is_err() {
        eprintln!("skipping bounded-output test: `head`/`tr` are missing");
        return;
    }
    let dir = workspace();
    let config = LocalWorkConfig {
        max_output_bytes: OUTPUT_BUDGET,
        ..LocalWorkConfig::unrestricted(dir.path().to_string_lossy().into_owned())
    };
    let op = Arc::new(LocalShellOperation::new(config).unwrap());

    // One line of 20 000 bytes and no trailing newline: a line-oriented reader that buffered
    // whole lines would hold all of it regardless of `max_output_bytes`.
    let params = ShellExecParams::new("head -c 20000 /dev/zero | tr '\\0' a");
    let items = drain(op.execute_cmd_stream(params)).await;

    let text: String = items
        .iter()
        .filter_map(|item| item.data.as_ref())
        .map(|chunk| chunk.text.as_str())
        .collect();
    assert!(
        text.ends_with(TRUNCATION_MARKER),
        "the capture must end with the truncation marker, got {} bytes",
        text.len()
    );
    let kept = text.len() - TRUNCATION_MARKER.len();
    assert_eq!(
        u64::try_from(kept).unwrap(),
        OUTPUT_BUDGET,
        "the retained output must fill the budget exactly"
    );
    for item in &items {
        if let Some(chunk) = item.data.as_ref() {
            assert!(
                chunk.text.len() <= 4096,
                "a streamed chunk must stay bounded, got {} bytes",
                chunk.text.len()
            );
        }
    }
    let terminal = items.last().unwrap().data.as_ref().unwrap();
    assert_eq!(terminal.exit_code, Some(0), "the command itself succeeded");
}
