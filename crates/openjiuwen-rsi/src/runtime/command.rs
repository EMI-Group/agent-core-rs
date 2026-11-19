//! The typed process-command seam: a [`CommandSpec`] in, a [`CommandOutcome`] out.
//!
//! Ports the subprocess half of the Python RSI infrastructure
//! (`asyncio.create_subprocess_exec` in
//! `harness_rsi/auto_harness/infra/git_operations.py` and
//! `worktree_manager.py`) to a reusable seam. Exit codes are data, never
//! errors: only a missing executable ([`Error::config`]) or an I/O failure
//! while spawning/reading ([`Error::tool`]) is returned as an error, and a
//! timeout is a [`CommandOutcome`] with `timed_out` set.
//!
//! [`ProcessCommandRunner`] is the real implementation over
//! `tokio::process::Command`; [`FakeCommandRunner`] is a scripted test double so
//! downstream layers (git, CI gates, pytest) can be tested without spawning
//! processes.
//!
//! Every real run is bounded: [`DEFAULT_COMMAND_TIMEOUT`] preempts a command
//! that never finishes, and [`MAX_OUTPUT_BYTES`] caps what is retained from each
//! stream while the surplus is drained. Both are overridable (and the deadline
//! can be removed) through the [`ProcessCommandRunner`] builders; truncation is
//! reported on [`CommandOutcome::truncated`], never by dropping the run.
//!
//! Decoding choice: both streams are decoded as UTF-8 lossily and exactly one
//! trailing newline is stripped (including the `\r\n` pair), matching the
//! Python helpers' `rstrip()` on command output without hiding internal blank
//! lines.
use std::collections::BTreeMap;
use std::io;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

use openjiuwen_common::error::{Error, Result};

/// Default ceiling on the bytes retained from one command stream.
///
/// The remainder is still drained from the pipe (so a child never blocks on a
/// full pipe) but is not kept, so a runaway command's memory cost is bounded no
/// matter how much it writes. Override it with
/// [`ProcessCommandRunner::with_max_output_bytes`].
pub const MAX_OUTPUT_BYTES: usize = 10 * 1024 * 1024;

/// Default fallback deadline applied to specs that carry no timeout.
///
/// Long enough for a real test suite or git operation, short enough that a hung
/// command cannot outlive the session budget. Override it with
/// [`ProcessCommandRunner::with_default_timeout`], or opt out of the bound with
/// [`ProcessCommandRunner::without_default_timeout`].
pub const DEFAULT_COMMAND_TIMEOUT: Duration = Duration::from_secs(600);

/// A description of one external command.
///
/// Build it with [`CommandSpec::new`] and the chainable setters; every field is
/// public so callers can also construct it literally.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CommandSpec {
    /// Executable name or path.
    pub program: String,
    /// Arguments, passed verbatim (no shell interpretation).
    pub args: Vec<String>,
    /// Working directory; `None` inherits the parent process directory.
    pub cwd: Option<PathBuf>,
    /// Extra environment variables layered over the parent environment.
    pub env: BTreeMap<String, String>,
    /// Environment variables removed from the inherited environment.
    pub env_remove: Vec<String>,
    /// Per-command deadline; `None` falls back to the runner default.
    pub timeout: Option<Duration>,
    /// Fold standard error into [`CommandOutcome::stdout`] (`stderr=STDOUT`).
    pub merge_stderr: bool,
    /// Text written to the child's standard input; `None` uses a null stdin.
    pub stdin: Option<String>,
}

impl CommandSpec {
    /// Creates a spec for `program` with no arguments.
    #[must_use]
    pub fn new(program: impl Into<String>) -> Self {
        Self {
            program: program.into(),
            ..Self::default()
        }
    }

    /// Appends one argument.
    #[must_use]
    pub fn arg(mut self, arg: impl Into<String>) -> Self {
        self.args.push(arg.into());
        self
    }

    /// Appends every argument of `args`, in order.
    #[must_use]
    pub fn args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.args.extend(args.into_iter().map(Into::into));
        self
    }

    /// Sets the working directory.
    #[must_use]
    pub fn cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }

    /// Adds one environment variable.
    #[must_use]
    pub fn env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.insert(key.into(), value.into());
        self
    }

    /// Removes one inherited environment variable.
    #[must_use]
    pub fn env_remove(mut self, key: impl Into<String>) -> Self {
        self.env_remove.push(key.into());
        self
    }

    /// Sets the per-command deadline.
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Folds standard error into standard output for this command.
    #[must_use]
    pub fn merge_stderr(mut self) -> Self {
        self.merge_stderr = true;
        self
    }

    /// Sets the text written to the child's standard input.
    #[must_use]
    pub fn stdin(mut self, stdin: impl Into<String>) -> Self {
        self.stdin = Some(stdin.into());
        self
    }

    /// Sets the deadline only when the spec does not already have one.
    #[must_use]
    pub fn with_default_timeout(mut self, timeout: Duration) -> Self {
        if self.timeout.is_none() {
            self.timeout = Some(timeout);
        }
        self
    }

    /// A one-line rendering of the command for logs and test matching.
    ///
    /// The program and each argument are single-quoted only when they contain
    /// characters that need it, so ordinary commands read as
    /// `git status --porcelain`.
    #[must_use]
    pub fn command_line(&self) -> String {
        let mut parts = Vec::with_capacity(self.args.len() + 1);
        parts.push(shell_quote(&self.program));
        parts.extend(self.args.iter().map(|arg| shell_quote(arg)));
        parts.join(" ")
    }
}

/// Quotes one command-line word only when it is not already safe.
fn shell_quote(value: &str) -> String {
    let safe = !value.is_empty()
        && value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || "-_./:=@%+,^".contains(ch));
    if safe {
        value.to_owned()
    } else {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
}

/// The result of running one command.
///
/// `code` is `None` when the child was signalled or killed instead of exiting
/// normally. `stderr` is always empty when the command was run with
/// [`CommandSpec::merge_stderr`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandOutcome {
    /// Exit code, or `None` when the child was signalled/killed.
    pub code: Option<i32>,
    /// Standard output, decoded lossily; the combined stream when merged.
    pub stdout: String,
    /// Standard error, decoded lossily; empty when merged.
    pub stderr: String,
    /// Whether the command hit its deadline and was killed.
    pub timed_out: bool,
    /// Whether the command was cancelled through a [`CancellationToken`].
    pub cancelled: bool,
    /// Whether either retained stream was cut off at the runner's output ceiling.
    ///
    /// Set when the child wrote more than [`MAX_OUTPUT_BYTES`] (or the runner's
    /// configured ceiling) on a stream; the retained text is the head of that
    /// stream and the surplus was drained and discarded.
    #[serde(default)]
    pub truncated: bool,
    /// Wall-clock duration of the run.
    pub duration: Duration,
}

impl CommandOutcome {
    /// Builds a successful outcome (`code == Some(0)`) carrying `stdout`.
    #[must_use]
    pub fn success_with(stdout: impl Into<String>) -> Self {
        Self {
            code: Some(0),
            stdout: stdout.into(),
            stderr: String::new(),
            timed_out: false,
            cancelled: false,
            truncated: false,
            duration: Duration::ZERO,
        }
    }

    /// Builds a failed outcome with `code` and `output` as standard output.
    #[must_use]
    pub fn failure_with(code: i32, output: impl Into<String>) -> Self {
        Self {
            code: Some(code),
            stdout: output.into(),
            stderr: String::new(),
            timed_out: false,
            cancelled: false,
            truncated: false,
            duration: Duration::ZERO,
        }
    }

    /// Builds a timed-out outcome that ran for `duration`.
    #[must_use]
    pub fn timed_out_after(duration: Duration) -> Self {
        Self {
            code: None,
            stdout: String::new(),
            stderr: String::new(),
            timed_out: true,
            cancelled: false,
            truncated: false,
            duration,
        }
    }

    /// Builds a cancelled outcome that ran for `duration`.
    #[must_use]
    pub fn cancelled_after(duration: Duration) -> Self {
        Self {
            code: None,
            stdout: String::new(),
            stderr: String::new(),
            timed_out: false,
            cancelled: true,
            truncated: false,
            duration,
        }
    }

    /// `true` only for a clean exit: code `0`, not timed out, not cancelled.
    #[must_use]
    pub fn success(&self) -> bool {
        self.code == Some(0) && !self.timed_out && !self.cancelled
    }

    /// The primary output text.
    ///
    /// When the command was merged this is the combined stream; otherwise it is
    /// standard output, falling back to standard error when stdout is empty (a
    /// command that only reports an error). Use [`CommandOutcome::stdout`] and
    /// [`CommandOutcome::stderr`] directly when both streams are needed.
    #[must_use]
    pub fn combined_output(&self) -> &str {
        if self.stdout.is_empty() && !self.stderr.is_empty() {
            &self.stderr
        } else {
            &self.stdout
        }
    }

    /// Returns at most `limit` characters from the end of the output.
    ///
    /// The tail is kept because command failures usually report at the end.
    /// `…[truncated]` is appended only when characters were actually dropped;
    /// the cut is by characters, so multi-byte text is never split.
    #[must_use]
    pub fn truncated_output(&self, limit: usize) -> String {
        let text = self.combined_output();
        let total = text.chars().count();
        if total <= limit {
            return text.to_owned();
        }
        let tail: String = text.chars().skip(total - limit).collect();
        format!("{tail}…[truncated]")
    }

    /// Turns a non-successful outcome into an [`Error::Tool`].
    ///
    /// `label` names the operation (for example `"git status"`); the message
    /// carries the exit code and the tail of the output so a caller can log it
    /// without dumping megabytes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::tool`] when [`CommandOutcome::success`] is false.
    pub fn require_success(&self, label: &str) -> Result<()> {
        if self.success() {
            return Ok(());
        }
        let code = self
            .code
            .map_or_else(|| "none".to_owned(), |code| code.to_string());
        let state = if self.timed_out {
            "timed out"
        } else if self.cancelled {
            "was cancelled"
        } else {
            "failed"
        };
        Err(Error::tool(format!(
            "{label} {state} (exit code {code}): {}",
            self.truncated_output(500)
        )))
    }
}

/// Decodes captured bytes lossily and strips one trailing newline.
///
/// A `\r\n` pair counts as one newline, so Windows output matches Unix output.
/// Internal newlines are preserved.
#[must_use]
pub fn decode_output(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes).into_owned();
    let text = text.strip_suffix('\n').unwrap_or(&text);
    let text = text.strip_suffix('\r').unwrap_or(text);
    text.to_owned()
}

/// Runs external commands.
///
/// Implementations are `Send + Sync` and are shared behind an
/// `Arc<dyn CommandRunner>`, so git and CI layers never depend on a concrete
/// process runner.
#[async_trait]
pub trait CommandRunner: Send + Sync {
    /// Short implementation name used in diagnostics (`"process"`, `"fake"`).
    fn name(&self) -> &str;

    /// Runs `spec` to completion, capturing its output.
    ///
    /// A non-zero exit is data. A timeout is reported through
    /// [`CommandOutcome::timed_out`] rather than an error.
    ///
    /// # Errors
    ///
    /// Returns [`Error::config`] when the executable does not exist and
    /// [`Error::tool`] when spawning, writing stdin, or reading output fails.
    async fn run(&self, spec: CommandSpec) -> Result<CommandOutcome>;

    /// Runs `spec`, aborting with [`Error::cancelled`] when `token` fires.
    ///
    /// The default implementation polls cancellation first and drops the
    /// in-flight [`CommandRunner::run`] future, so a real child spawned with
    /// `kill_on_drop` is killed; a scripted implementation simply stops waiting.
    ///
    /// # Errors
    ///
    /// Returns [`Error::cancelled`] when `token` is or becomes cancelled, and
    /// otherwise propagates the errors of [`CommandRunner::run`].
    async fn run_cancellable(
        &self,
        spec: CommandSpec,
        token: &CancellationToken,
    ) -> Result<CommandOutcome> {
        if token.is_cancelled() {
            return Err(Error::cancelled());
        }
        tokio::select! {
            biased;
            () = token.cancelled() => Err(Error::cancelled()),
            outcome = self.run(spec) => outcome,
        }
    }
}

/// The real [`CommandRunner`], spawning processes through `tokio::process`.
///
/// Children are spawned with `kill_on_drop`, a null stdin by default, and piped
/// output, so dropping the future (cancellation) or hitting the deadline kills
/// the process instead of leaking it.
///
/// Every run is bounded twice: [`ProcessCommandRunner::default_timeout`]
/// ([`DEFAULT_COMMAND_TIMEOUT`] unless overridden) preempts a command that never
/// finishes, and [`ProcessCommandRunner::max_output_bytes`]
/// ([`MAX_OUTPUT_BYTES`] unless overridden) caps what is retained from each
/// stream. The surplus is drained, so a chatty command still exits normally and
/// reports [`CommandOutcome::truncated`] instead of exhausting memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessCommandRunner {
    default_timeout: Option<Duration>,
    max_output_bytes: usize,
}

impl Default for ProcessCommandRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl ProcessCommandRunner {
    /// Creates a runner with the documented defaults: [`DEFAULT_COMMAND_TIMEOUT`]
    /// and [`MAX_OUTPUT_BYTES`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            default_timeout: Some(DEFAULT_COMMAND_TIMEOUT),
            max_output_bytes: MAX_OUTPUT_BYTES,
        }
    }

    /// Creates a runner without a fallback deadline.
    ///
    /// This is the explicit opt-out of [`DEFAULT_COMMAND_TIMEOUT`]; the output
    /// ceiling still applies.
    #[must_use]
    pub fn without_default_timeout() -> Self {
        Self {
            default_timeout: None,
            max_output_bytes: MAX_OUTPUT_BYTES,
        }
    }

    /// Sets the fallback deadline applied to specs without their own.
    #[must_use]
    pub fn with_default_timeout(mut self, timeout: Duration) -> Self {
        self.default_timeout = Some(timeout);
        self
    }

    /// Removes the fallback deadline (the explicit opt-out of
    /// [`DEFAULT_COMMAND_TIMEOUT`]).
    #[must_use]
    pub fn clear_default_timeout(mut self) -> Self {
        self.default_timeout = None;
        self
    }

    /// Sets how many bytes are retained from each stream.
    ///
    /// `usize::MAX` restores the pre-ceiling behaviour and is not recommended:
    /// a runaway command can then exhaust memory.
    #[must_use]
    pub fn with_max_output_bytes(mut self, max_output_bytes: usize) -> Self {
        self.max_output_bytes = max_output_bytes;
        self
    }

    /// Returns how many bytes are retained from each stream.
    #[must_use]
    pub fn max_output_bytes(&self) -> usize {
        self.max_output_bytes
    }

    /// Returns the fallback deadline, if any.
    #[must_use]
    pub fn default_timeout(&self) -> Option<Duration> {
        self.default_timeout
    }
}

#[async_trait]
impl CommandRunner for ProcessCommandRunner {
    #[allow(clippy::unnecessary_literal_bound)] // the trait contract is `&str`
    fn name(&self) -> &str {
        "process"
    }

    async fn run(&self, spec: CommandSpec) -> Result<CommandOutcome> {
        let started = Instant::now();
        let mut command = Command::new(&spec.program);
        command.args(&spec.args);
        if let Some(cwd) = &spec.cwd {
            command.current_dir(cwd);
        }
        for (key, value) in &spec.env {
            command.env(key, value);
        }
        for key in &spec.env_remove {
            command.env_remove(key);
        }
        command
            .stdin(if spec.stdin.is_some() {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let mut child = command
            .spawn()
            .map_err(|error| spawn_error(&spec.program, &error))?;

        if let Some(input) = spec.stdin.as_ref() {
            if let Some(mut stdin) = child.stdin.take() {
                // A child that exits before reading (for example `head`) closes
                // the pipe; that is not an error of the command.
                if let Err(error) = stdin.write_all(input.as_bytes()).await {
                    if error.kind() != io::ErrorKind::BrokenPipe {
                        return Err(Error::tool(format!(
                            "failed to write stdin to `{}`: {error}",
                            spec.program
                        )));
                    }
                }
            }
        }

        let timeout = spec.timeout.or(self.default_timeout);
        let max_output_bytes = self.max_output_bytes;
        let (stdout, stderr, code) = if let Some(limit) = timeout {
            if let Ok(result) =
                tokio::time::timeout(limit, collect_output(&mut child, max_output_bytes)).await
            {
                let (stdout, stderr, status) =
                    result.map_err(|error| read_error(&spec.program, &error))?;
                (stdout, stderr, status.code())
            } else {
                if let Err(error) = child.start_kill() {
                    tracing::warn!(
                        program = %spec.program,
                        %error,
                        "failed to kill a timed-out command"
                    );
                }
                if let Err(error) = child.wait().await {
                    tracing::warn!(
                        program = %spec.program,
                        %error,
                        "failed to reap a timed-out command"
                    );
                }
                return Ok(CommandOutcome {
                    code: None,
                    stdout: String::new(),
                    stderr: String::new(),
                    timed_out: true,
                    cancelled: false,
                    truncated: false,
                    duration: started.elapsed(),
                });
            }
        } else {
            let (stdout, stderr, status) = collect_output(&mut child, max_output_bytes)
                .await
                .map_err(|error| read_error(&spec.program, &error))?;
            (stdout, stderr, status.code())
        };

        let (stdout, stderr, truncated) = if spec.merge_stderr {
            let (combined, truncated) = merge_captured(stdout, &stderr, max_output_bytes);
            (decode_output(&combined), String::new(), truncated)
        } else {
            (
                decode_output(&stdout.bytes),
                decode_output(&stderr.bytes),
                stdout.truncated || stderr.truncated,
            )
        };

        Ok(CommandOutcome {
            code,
            stdout,
            stderr,
            timed_out: false,
            cancelled: false,
            truncated,
            duration: started.elapsed(),
        })
    }
}

/// Maps a spawn failure to the typed crate error.
fn spawn_error(program: &str, error: &io::Error) -> Error {
    if error.kind() == io::ErrorKind::NotFound {
        Error::config(format!("command not found: `{program}`"))
    } else {
        Error::tool(format!("failed to spawn `{program}`: {error}"))
    }
}

/// Maps an output-read failure to the typed crate error.
fn read_error(program: &str, error: &io::Error) -> Error {
    Error::tool(format!("failed to read output of `{program}`: {error}"))
}

/// One captured stream: at most the ceiling, plus whether anything was dropped.
struct Captured {
    bytes: Vec<u8>,
    truncated: bool,
}

impl Captured {
    fn empty() -> Self {
        Self {
            bytes: Vec::new(),
            truncated: false,
        }
    }
}

/// Reads one optional stream to end, retaining at most `limit` bytes.
///
/// The cap is enforced while reading: once `limit` bytes are held, the read
/// loop keeps draining the pipe (so a chatty child never blocks on a full pipe
/// and exits normally) but discards the surplus, and the pre-allocation is
/// bounded so the buffer never grows past the ceiling.
async fn read_stream<R>(stream: Option<R>, limit: usize) -> io::Result<Captured>
where
    R: AsyncRead + Unpin,
{
    let Some(mut stream) = stream else {
        return Ok(Captured::empty());
    };
    let mut captured = Captured {
        bytes: Vec::with_capacity(limit.min(64 * 1024)),
        truncated: false,
    };
    // Heap-allocated so the read future stays small (the buffer is reused).
    let mut chunk = vec![0u8; 8192];
    loop {
        let read = stream.read(&mut chunk).await?;
        if read == 0 {
            return Ok(captured);
        }
        let room = limit.saturating_sub(captured.bytes.len());
        let kept = room.min(read);
        captured.bytes.extend_from_slice(&chunk[..kept]);
        if kept < read {
            captured.truncated = true;
        }
    }
}

/// Combines the two captured streams, keeping at most `limit` bytes.
///
/// Output is appended in capture order (stdout first, then stderr) and the
/// retained prefix never exceeds the ceiling, so `merge_stderr` cannot double
/// the memory cost.
fn merge_captured(stdout: Captured, stderr: &Captured, limit: usize) -> (Vec<u8>, bool) {
    let mut truncated = stdout.truncated || stderr.truncated;
    let mut combined = stdout.bytes;
    let room = limit.saturating_sub(combined.len());
    if stderr.bytes.len() > room {
        truncated = true;
    }
    combined.extend_from_slice(&stderr.bytes[..stderr.bytes.len().min(room)]);
    (combined, truncated)
}

/// Reads both pipes and waits for the child, avoiding pipe-buffer deadlocks.
async fn collect_output(
    child: &mut tokio::process::Child,
    max_output_bytes: usize,
) -> io::Result<(Captured, Captured, std::process::ExitStatus)> {
    let (stdout, stderr, status) = tokio::join!(
        read_stream(child.stdout.take(), max_output_bytes),
        read_stream(child.stderr.take(), max_output_bytes),
        child.wait(),
    );
    Ok((stdout?, stderr?, status?))
}

/// One scripted reply for [`FakeCommandRunner`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FakeReply {
    /// Program to match; `"*"` matches any program.
    pub program: String,
    /// Fragments that must all appear as substrings of the joined arguments.
    pub args_contain: Vec<String>,
    /// Outcome returned when the reply matches.
    pub outcome: CommandOutcome,
}

impl FakeReply {
    /// Creates a reply for `program` (or `"*"`) with no argument constraints.
    #[must_use]
    pub fn outcome(program: impl Into<String>, outcome: CommandOutcome) -> Self {
        Self {
            program: program.into(),
            args_contain: Vec::new(),
            outcome,
        }
    }

    /// Requires every fragment of `args` to appear in the joined arguments.
    #[must_use]
    pub fn with_args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.args_contain = args.into_iter().map(Into::into).collect();
        self
    }

    /// Whether this reply is the one scripted for `spec`.
    fn matches(&self, spec: &CommandSpec) -> bool {
        let program_matches = self.program == "*" || self.program == spec.program;
        if !program_matches {
            return false;
        }
        let joined = spec.args.join(" ");
        self.args_contain
            .iter()
            .all(|fragment| joined.contains(fragment.as_str()))
    }
}

/// Mutable state of a [`FakeCommandRunner`].
#[derive(Debug, Default)]
struct FakeState {
    replies: Vec<FakeReply>,
    fallback: Option<CommandOutcome>,
    calls: Vec<CommandSpec>,
}

/// A scripted [`CommandRunner`] for tests.
///
/// Replies are matched in insertion order: the first reply whose program
/// matches and whose `args_contain` fragments are all substrings of the joined
/// arguments wins. When nothing matches, the fallback outcome is returned; with
/// no fallback the run fails with [`Error::tool`]. Every call is recorded so a
/// test can assert the exact command line, working directory, and environment.
///
/// This is a test double, not a sandbox: it records and returns data but never
/// spawns a process.
#[derive(Debug, Default)]
pub struct FakeCommandRunner {
    state: Mutex<FakeState>,
}

impl FakeCommandRunner {
    /// Creates a runner with no replies and no fallback.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Mutex::new(FakeState::default()),
        }
    }

    /// Appends a scripted reply, taking precedence over later replies.
    #[must_use]
    pub fn reply(mut self, reply: FakeReply) -> Self {
        self.state_mut().replies.push(reply);
        self
    }

    /// Sets the outcome returned when no reply matches.
    #[must_use]
    pub fn fallback(mut self, outcome: CommandOutcome) -> Self {
        self.state_mut().fallback = Some(outcome);
        self
    }

    /// Returns a copy of every recorded call, in call order.
    #[must_use]
    pub fn calls(&self) -> Vec<CommandSpec> {
        self.state().calls.clone()
    }

    /// Returns how many commands were run.
    #[must_use]
    pub fn call_count(&self) -> usize {
        self.state().calls.len()
    }

    /// Returns the recorded command lines, in call order.
    #[must_use]
    pub fn command_lines(&self) -> Vec<String> {
        self.state()
            .calls
            .iter()
            .map(CommandSpec::command_line)
            .collect()
    }

    /// Locks the state, recovering from a poisoned lock.
    fn state(&self) -> MutexGuard<'_, FakeState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Mutable state accessor used by the builders.
    fn state_mut(&mut self) -> MutexGuard<'_, FakeState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[async_trait]
impl CommandRunner for FakeCommandRunner {
    #[allow(clippy::unnecessary_literal_bound)] // the trait contract is `&str`
    fn name(&self) -> &str {
        "fake"
    }

    async fn run(&self, spec: CommandSpec) -> Result<CommandOutcome> {
        let outcome = {
            // The lock is released before the first await point.
            let mut state = self.state();
            state.calls.push(spec.clone());
            state
                .replies
                .iter()
                .find(|reply| reply.matches(&spec))
                .map(|reply| reply.outcome.clone())
                .or_else(|| state.fallback.clone())
        };
        outcome.ok_or_else(|| {
            Error::tool(format!(
                "fake command runner has no scripted reply for `{}`",
                spec.command_line()
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_line_quotes_only_when_needed() {
        let spec = CommandSpec::new("git")
            .args(["commit", "-m", "hello world"])
            .arg("it's");
        assert_eq!(
            spec.command_line(),
            "git commit -m 'hello world' 'it'\\''s'"
        );
        assert_eq!(CommandSpec::new("echo").command_line(), "echo");
    }

    #[test]
    fn decode_strips_one_newline_and_is_lossy() {
        assert_eq!(decode_output(b"hello\n"), "hello");
        assert_eq!(decode_output(b"hello\r\n"), "hello");
        assert_eq!(decode_output(b"a\n\n"), "a\n");
        assert_eq!(decode_output(&[0xff, b'a']), "\u{fffd}a");
    }

    #[test]
    fn truncation_keeps_a_character_safe_tail() {
        let outcome = CommandOutcome::success_with("abcd😀ef");
        assert_eq!(outcome.truncated_output(100), "abcd😀ef");
        assert_eq!(outcome.truncated_output(3), "😀ef…[truncated]");
        assert_eq!(outcome.truncated_output(0), "…[truncated]");
    }

    #[test]
    fn require_success_names_the_operation_and_code() {
        let error = CommandOutcome::failure_with(2, "lint exploded")
            .require_success("lint")
            .unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Tool);
        let message = error.to_string();
        assert!(message.contains("lint"));
        assert!(message.contains("exit code 2"));
        assert!(message.contains("lint exploded"));
        assert!(CommandOutcome::success_with("ok")
            .require_success("lint")
            .is_ok());
    }

    #[test]
    fn default_timeout_only_fills_the_gap() {
        let fallback = Duration::from_secs(30);
        assert_eq!(
            CommandSpec::new("x").with_default_timeout(fallback).timeout,
            Some(fallback)
        );
        let explicit = Duration::from_secs(5);
        assert_eq!(
            CommandSpec::new("x")
                .timeout(explicit)
                .with_default_timeout(fallback)
                .timeout,
            Some(explicit)
        );
    }
}
