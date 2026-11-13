//! PTY-backed streaming tests for the local shell backend.
//!
//! `LocalWorkConfig::pty_streaming` attaches a *streaming* command to a pseudo-terminal so
//! the child line-buffers its own output, the way Python's Linux `stdbuf -oL -eL` wrapper
//! does. Every chunk of a PTY stream is `stdout` (a terminal merges stderr), chunk text
//! carries no carriage returns, and the terminal chunk still reports the exit code.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use openjiuwen_sysop::config::LocalWorkConfig;
use openjiuwen_sysop::operation::ShellOperation;
use openjiuwen_sysop::shell::local_shell;
use openjiuwen_sysop::shell::types::{ShellChunk, ShellExecParams, StreamKind};

/// Upper bound on one streamed run.
const BOUND: Duration = Duration::from_secs(10);
/// A command that writes four lines with a short pause between them.
const FOUR_LINES: &str = "for i in 1 2 3 4; do echo line$i; sleep 0.05; done";

/// Returns `true` when `sh` is available; otherwise prints a skip notice.
fn require_sh() -> bool {
    if which::which("sh").is_ok() {
        return true;
    }
    eprintln!("skipping PTY streaming test: `sh` is not on PATH");
    false
}

/// The unrestricted local configuration rooted at `dir`, with PTY streaming toggled.
fn config(dir: &tempfile::TempDir, pty_streaming: bool) -> LocalWorkConfig {
    let mut config = LocalWorkConfig::unrestricted(dir.path().to_string_lossy().into_owned());
    config.pty_streaming = pty_streaming;
    config
}

/// Collects one streamed run, failing the test rather than hanging on a stuck child.
async fn collect(backend: Arc<dyn ShellOperation>, command: &str) -> Vec<ShellChunk> {
    let events = backend.execute_cmd_stream(ShellExecParams::new(command));
    let items = tokio::time::timeout(BOUND, events.collect::<Vec<_>>())
        .await
        .expect("the streamed run finished within the bound");
    for item in &items {
        assert_eq!(item.code, 0, "chunk failure: {}", item.message);
    }
    items
        .into_iter()
        .filter_map(|item| item.data)
        .collect::<Vec<_>>()
}

#[tokio::test]
async fn pty_streaming_emits_line_chunks_without_carriage_returns() {
    if !require_sh() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let backend = local_shell(config(&dir, true)).unwrap();
    let chunks = collect(backend, FOUR_LINES).await;

    let data: Vec<&ShellChunk> = chunks
        .iter()
        .filter(|chunk| chunk.exit_code.is_none())
        .collect();
    assert!(
        data.len() >= 2,
        "expected line-buffered chunks, got {}: {:?}",
        data.len(),
        data.iter()
            .map(|chunk| chunk.text.as_str())
            .collect::<Vec<_>>()
    );
    // A terminal has one stream: stderr is merged into stdout.
    assert!(
        data.iter()
            .all(|chunk| chunk.stream == Some(StreamKind::Stdout)),
        "PTY chunks must all be stdout"
    );
    let text: String = data.iter().map(|chunk| chunk.text.as_str()).collect();
    assert!(!text.contains('\r'), "carriage returns leaked: {text:?}");
    for line in ["line1", "line2", "line3", "line4"] {
        assert!(text.contains(line), "missing {line} in {text:?}");
    }
    let terminal = chunks.last().expect("a terminal chunk");
    assert_eq!(terminal.exit_code, Some(0));
    assert_eq!(
        terminal.chunk_index,
        u64::try_from(chunks.len() - 1).unwrap()
    );
}

#[tokio::test]
async fn pty_streaming_attaches_a_terminal_and_pipes_do_not() {
    if !require_sh() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let probe = "[ -t 1 ] && echo tty || echo pipe";
    let pty = collect(local_shell(config(&dir, true)).unwrap(), probe).await;
    let pty_text: String = pty
        .iter()
        .filter(|chunk| chunk.exit_code.is_none())
        .map(|chunk| chunk.text.as_str())
        .collect();
    assert_eq!(
        pty_text.trim(),
        "tty",
        "the streaming run must be attached to a terminal, got {pty_text:?}"
    );

    let pipes = collect(local_shell(config(&dir, false)).unwrap(), probe).await;
    let pipe_text: String = pipes
        .iter()
        .filter(|chunk| chunk.exit_code.is_none())
        .map(|chunk| chunk.text.as_str())
        .collect();
    assert_eq!(
        pipe_text.trim(),
        "pipe",
        "with PTY streaming off the run must use pipes, got {pipe_text:?}"
    );
}

#[tokio::test]
async fn pty_streaming_reports_a_non_zero_exit_code() {
    if !require_sh() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let backend = local_shell(config(&dir, true)).unwrap();
    let chunks = collect(backend, "echo failing; exit 3").await;
    let terminal = chunks.last().expect("a terminal chunk");
    assert_eq!(terminal.exit_code, Some(3));
    let text: String = chunks
        .iter()
        .filter(|chunk| chunk.exit_code.is_none())
        .map(|chunk| chunk.text.as_str())
        .collect();
    assert!(text.contains("failing"), "{text:?}");
}

#[tokio::test]
async fn pipe_streaming_still_works_when_pty_streaming_is_off() {
    if !require_sh() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let backend = local_shell(config(&dir, false)).unwrap();
    let chunks = collect(backend, FOUR_LINES).await;
    let text: String = chunks
        .iter()
        .filter(|chunk| chunk.exit_code.is_none())
        .map(|chunk| chunk.text.as_str())
        .collect();
    for line in ["line1", "line2", "line3", "line4"] {
        assert!(text.contains(line), "missing {line} in {text:?}");
    }
    assert_eq!(chunks.last().expect("a terminal chunk").exit_code, Some(0));
}

#[tokio::test]
async fn captured_runs_keep_separate_streams_even_with_pty_streaming_on() {
    if !require_sh() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let backend = local_shell(config(&dir, true)).unwrap();
    let result = backend
        .execute_cmd(ShellExecParams::new("printf out; printf err >&2"))
        .await;
    assert_eq!(result.code, 0, "{}", result.message);
    let output = result.data.expect("captured run");
    // A PTY would merge the two; captured runs stay on pipes.
    assert_eq!(output.stdout, "out");
    assert_eq!(output.stderr, "err");
    assert_eq!(output.exit_code, 0);
}
