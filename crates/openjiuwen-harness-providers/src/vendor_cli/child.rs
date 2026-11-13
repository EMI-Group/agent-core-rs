//! One spawned vendor CLI process with NDJSON stdin/stdout and a bounded stderr tail.
//!
//! Vendor CLIs are long-lived child processes: the harness writes one JSON request per line to
//! stdin, reads one JSON message per line from stdout, and treats stderr as diagnostic output that
//! must never block the transport. [`ChildProcess`] owns that plumbing:
//!
//! - stdin/stdout/stderr are piped, and the child is killed when the handle is dropped, so a
//!   harness that fails mid-turn cannot leave an orphan process behind;
//! - stderr is drained by a background task into a bounded tail (the last
//!   [`STDERR_TAIL_LINES`] lines, at most [`STDERR_TAIL_CHARS`] characters) that is appended to
//!   transport errors — a vendor CLI usually explains itself on stderr;
//! - every read has a hard per-line byte ceiling, so a runaway CLI cannot exhaust memory;
//! - waits are bounded, so a wedged child cannot hang a stop.

use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use openjiuwen_harness_protocol::{HarnessError, HarnessResult, JsonValue};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStderr, ChildStdin, ChildStdout};
use tokio::task::JoinHandle;

use super::jsonl::{encode_line, snippet};
use super::lock;

/// Per-line stdout ceiling applied when [`ChildProcessConfig::max_line_bytes`] is zero (32 MiB).
pub const DEFAULT_MAX_LINE_BYTES: usize = 32 * 1024 * 1024;

/// Number of stderr lines retained for diagnostics.
pub const STDERR_TAIL_LINES: usize = 40;

/// Number of stderr characters retained for diagnostics.
pub const STDERR_TAIL_CHARS: usize = 8000;

/// Upper bound on how long [`ChildProcess::kill`] and [`ChildProcess::wait`] block.
pub const EXIT_TIMEOUT: Duration = Duration::from_secs(5);

/// Characters of an invalid output line quoted in the error message.
const OUTPUT_SNIPPET_CHARS: usize = 200;

/// How to spawn one vendor CLI process.
pub struct ChildProcessConfig<'a> {
    /// Executable to run, already resolved (see [`super::binary::resolve_binary`]).
    pub program: &'a Path,
    /// Command-line arguments passed to the executable.
    pub args: &'a [String],
    /// Working directory, when the harness pins one.
    pub cwd: Option<&'a Path>,
    /// Variables added on top of the inherited environment (or on top of an empty one).
    pub env: &'a BTreeMap<String, String>,
    /// Whether the child inherits the current process environment before `env` is applied.
    pub inherit_env: bool,
    /// Human-readable harness label used in messages, for example `Claude Code`.
    pub label: &'a str,
    /// Per-line stdout ceiling in bytes; `0` selects [`DEFAULT_MAX_LINE_BYTES`].
    pub max_line_bytes: usize,
}

/// A spawned vendor CLI process speaking newline-delimited JSON.
pub struct ChildProcess {
    /// The child handle; `kill_on_drop` guarantees it is reaped when the struct is dropped.
    child: Child,
    /// stdin of the child; [`None`] once it has been closed or the pipe broke.
    stdin: Option<ChildStdin>,
    /// Line-buffered stdout of the child.
    stdout: BufReader<ChildStdout>,
    /// Retained stderr lines, shared with the drain task.
    stderr: Arc<Mutex<StderrTail>>,
    /// The stderr drain task, so it can be joined or aborted.
    stderr_task: Option<JoinHandle<()>>,
    /// Label used in error messages.
    label: String,
    /// Executable path used in error messages.
    program: PathBuf,
    /// Effective per-line stdout ceiling.
    max_line_bytes: usize,
}

impl ChildProcess {
    /// Spawns the configured process with piped stdin, stdout and stderr.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::Protocol`] with `failed to start {label} CLI '{program}': {error}`
    /// when the process cannot be spawned.
    // Async keeps the constructor uniform with the transport calls.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn spawn(config: ChildProcessConfig<'_>) -> HarnessResult<Self> {
        let mut command = tokio::process::Command::new(config.program);
        command.args(config.args);
        if let Some(cwd) = config.cwd {
            command.current_dir(cwd);
        }
        if !config.inherit_env {
            command.env_clear();
        }
        command.envs(config.env);
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let mut child = command.spawn().map_err(|error| {
            HarnessError::protocol(format!(
                "failed to start {} CLI '{}': {error}",
                config.label,
                config.program.display()
            ))
        })?;
        let stdin = child.stdin.take();
        let (Some(stdout), Some(stderr)) = (child.stdout.take(), child.stderr.take()) else {
            // `Stdio::piped` guarantees both pipes; losing one means the spawn is unusable.
            let _ = child.start_kill();
            return Err(HarnessError::protocol(format!(
                "failed to start {} CLI '{}': its output pipes are unavailable",
                config.label,
                config.program.display()
            )));
        };
        let tail = Arc::new(Mutex::new(StderrTail::default()));
        let stderr_task = tokio::spawn(drain_stderr(stderr, Arc::clone(&tail)));
        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            stderr: tail,
            stderr_task: Some(stderr_task),
            label: config.label.to_owned(),
            program: config.program.to_path_buf(),
            max_line_bytes: if config.max_line_bytes == 0 {
                DEFAULT_MAX_LINE_BYTES
            } else {
                config.max_line_bytes
            },
        })
    }

    /// Writes `value` to the child as one compact JSON line.
    ///
    /// # Errors
    ///
    /// Propagates [`write_line`](Self::write_line).
    pub async fn write_json(&mut self, value: &JsonValue) -> HarnessResult<()> {
        self.write_line(&encode_line(value)).await
    }

    /// Writes `line` followed by a newline and flushes stdin.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::Protocol`] with
    /// `the {label} CLI exited before the message was written`, plus the stderr tail, when stdin
    /// was already closed or the write failed.
    pub async fn write_line(&mut self, line: &str) -> HarnessResult<()> {
        let Some(stdin) = self.stdin.as_mut() else {
            return Err(self.exited_error());
        };
        let mut payload = String::with_capacity(line.len() + 1);
        payload.push_str(line);
        payload.push('\n');
        let written = async {
            stdin.write_all(payload.as_bytes()).await?;
            stdin.flush().await
        }
        .await;
        match written {
            Ok(()) => Ok(()),
            Err(error) => {
                self.stdin = None;
                tracing::debug!(%error, label = %self.label, "failed to write to a vendor CLI");
                Err(self.exited_error())
            }
        }
    }

    /// Reads the next JSON message from stdout.
    ///
    /// Blank lines are skipped. Returns `Ok(None)` once stdout reaches end of file, which is how a
    /// vendor CLI reports that it exited.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::Protocol`] when a line exceeds the configured
    /// `max_line_bytes`, when a line is not valid JSON
    /// (`invalid {label} CLI output: {snippet}`, plus the stderr tail), or when the read itself
    /// fails. Invalid UTF-8 is decoded lossily so the diagnostic still reports the line.
    pub async fn read_json(&mut self) -> HarnessResult<Option<JsonValue>> {
        loop {
            let mut buffer = Vec::new();
            let read = match self.stdout.read_until(b'\n', &mut buffer).await {
                Ok(read) => read,
                Err(error) => {
                    return Err(self.with_tail(format!(
                        "failed to read from the {} CLI: {error}",
                        self.label
                    )))
                }
            };
            if read == 0 {
                return Ok(None);
            }
            if read > self.max_line_bytes {
                return Err(HarnessError::protocol(format!(
                    "a {} CLI output line exceeded {} bytes",
                    self.label, self.max_line_bytes
                )));
            }
            let text = String::from_utf8_lossy(&buffer);
            let trimmed = text.trim();
            if trimmed.is_empty() {
                continue;
            }
            return match serde_json::from_str(trimmed) {
                Ok(value) => Ok(Some(value)),
                Err(error) => {
                    tracing::debug!(%error, label = %self.label, "invalid vendor CLI output line");
                    Err(self.with_tail(format!(
                        "invalid {} CLI output: {}",
                        self.label,
                        snippet(trimmed, OUTPUT_SNIPPET_CHARS)
                    )))
                }
            };
        }
    }

    /// Returns the retained stderr tail, trimmed, with one line per element.
    #[must_use]
    pub fn stderr_tail(&self) -> String {
        lock(&self.stderr).rendered().trim().to_owned()
    }

    /// Closes stdin, which signals `EOF` to a CLI that reads until end of file.
    pub async fn close_stdin(&mut self) {
        if let Some(mut stdin) = self.stdin.take() {
            // Flushing before the drop surfaces a broken pipe here instead of losing it silently.
            if let Err(error) = stdin.flush().await {
                tracing::debug!(%error, label = %self.label, "failed to flush vendor CLI stdin");
            }
        }
    }

    /// Sends `SIGKILL` and waits at most [`EXIT_TIMEOUT`] for the process to be reaped.
    pub async fn kill(&mut self) -> Option<i32> {
        if let Err(error) = self.child.start_kill() {
            tracing::debug!(%error, label = %self.label, program = %self.program.display(), "failed to kill a vendor CLI");
        }
        let code = self.wait_bounded().await;
        self.stdin = None;
        code
    }

    /// Waits at most [`EXIT_TIMEOUT`] for a natural exit and returns its status code.
    ///
    /// Returns [`None`] when the process did not exit within the window (it is not killed) or when
    /// it was terminated by a signal.
    pub async fn wait(&mut self) -> Option<i32> {
        self.wait_bounded().await
    }

    /// Returns the operating-system process id, when the child is still running.
    #[must_use]
    pub fn pid(&self) -> Option<u32> {
        self.child.id()
    }

    /// Waits for the child with the shared timeout and tears the stderr task down.
    async fn wait_bounded(&mut self) -> Option<i32> {
        let waited = tokio::time::timeout(EXIT_TIMEOUT, self.child.wait()).await;
        match waited {
            Ok(Ok(status)) => {
                self.finish_stderr_task().await;
                status.code()
            }
            Ok(Err(error)) => {
                tracing::debug!(%error, label = %self.label, "failed to wait for a vendor CLI");
                self.finish_stderr_task().await;
                None
            }
            Err(_) => {
                tracing::debug!(label = %self.label, "vendor CLI did not exit within the timeout");
                if let Some(task) = self.stderr_task.take() {
                    task.abort();
                }
                None
            }
        }
    }

    /// Joins the stderr drain task (it ends by itself once the child closes stderr).
    async fn finish_stderr_task(&mut self) {
        if let Some(task) = self.stderr_task.take() {
            let _ = task.await;
        }
    }

    /// Builds the "the CLI exited before the message was written" error, with the stderr tail.
    fn exited_error(&self) -> HarnessError {
        self.with_tail(format!(
            "the {} CLI exited before the message was written",
            self.label
        ))
    }

    /// Appends the retained stderr tail to `message` when the child produced any output.
    fn with_tail(&self, message: String) -> HarnessError {
        let tail = self.stderr_tail();
        if tail.is_empty() {
            HarnessError::protocol(message)
        } else {
            HarnessError::protocol(format!("{message}\n{tail}"))
        }
    }
}

/// Bounded, order-preserving tail of the child's stderr.
#[derive(Default)]
struct StderrTail {
    /// Retained lines, oldest first.
    lines: VecDeque<String>,
    /// Number of characters currently retained.
    chars: usize,
}

impl StderrTail {
    /// Appends `line`, dropping the oldest lines once a cap is exceeded.
    fn push(&mut self, line: &str) {
        let text: String = line.chars().take(STDERR_TAIL_CHARS).collect();
        self.chars += text.chars().count();
        self.lines.push_back(text);
        while self.lines.len() > STDERR_TAIL_LINES {
            if let Some(dropped) = self.lines.pop_front() {
                self.chars -= dropped.chars().count();
            }
        }
        while self.chars > STDERR_TAIL_CHARS && self.lines.len() > 1 {
            if let Some(dropped) = self.lines.pop_front() {
                self.chars -= dropped.chars().count();
            }
        }
    }

    /// Renders the retained lines, oldest first, separated by newlines.
    fn rendered(&self) -> String {
        self.lines
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Drains the child's stderr into `tail` until end of file.
async fn drain_stderr(stderr: ChildStderr, tail: Arc<Mutex<StderrTail>>) {
    let mut lines = BufReader::new(stderr).lines();
    loop {
        match lines.next_line().await {
            Ok(Some(line)) => lock(&tail).push(&line),
            Ok(None) => break,
            Err(error) => {
                tracing::debug!(%error, "failed to read vendor CLI stderr");
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use super::{StderrTail, STDERR_TAIL_CHARS, STDERR_TAIL_LINES};

    #[test]
    fn tail_keeps_the_last_lines() {
        let mut tail = StderrTail::default();
        for index in 0..(STDERR_TAIL_LINES + 5) {
            tail.push(&format!("line {index}"));
        }
        assert_eq!(tail.lines.len(), STDERR_TAIL_LINES);
        assert_eq!(tail.lines.front().map(String::as_str), Some("line 5"),);
        assert_eq!(tail.lines.back().map(String::as_str), Some("line 44"),);
    }

    #[test]
    fn tail_keeps_the_last_characters() {
        let mut tail = StderrTail::default();
        let line = "x".repeat(STDERR_TAIL_CHARS / 2 + 100);
        for index in 0..4 {
            tail.push(&format!("{line}{index}"));
        }
        assert!(tail.chars <= STDERR_TAIL_CHARS);
        assert_eq!(tail.lines.len(), 1);
    }

    #[test]
    fn tail_truncates_a_single_huge_line() {
        let mut tail = StderrTail::default();
        tail.push(&"y".repeat(STDERR_TAIL_CHARS * 3));
        assert_eq!(tail.chars, STDERR_TAIL_CHARS);
        assert_eq!(tail.rendered().chars().count(), STDERR_TAIL_CHARS);
    }
}
