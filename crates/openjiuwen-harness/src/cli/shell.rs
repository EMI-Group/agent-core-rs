//! The `!cmd` shell-passthrough command of the REPL.
//!
//! A REPL line starting with `!` runs the rest of the line as a shell command in
//! the CLI workspace, bypassing the agent. Python does this with
//! `asyncio.create_subprocess_shell`, prints stdout verbatim and stderr in red,
//! and never looks at the exit code; this port keeps that behaviour in the
//! renderer and adds the exit status to [`ShellOutput`] so the REPL can print a
//! failure line (a documented addition, not a parity change to the command).
//!
//! The runner is a trait so the REPL and its tests never depend on a real
//! process: [`ProcessShellRunner`] is the `sh -c` implementation, and tests
//! substitute a scripted runner behind `Arc<dyn ShellRunner>`.
//!
//! # Bounded by construction
//!
//! Python's passthrough is unbounded in three ways, and a REPL cannot afford any
//! of them: `!yes` buffers output until the process dies, `!sleep 9999` wedges
//! the prompt, and Ctrl-C only reaches the `sh` process, not the pipeline it
//! started. [`ProcessShellRunner`] therefore:
//!
//! - **owns the process group** — the child is spawned with
//!   `process_group(0)`, so a kill reaches every descendant through
//!   [`terminate_group`] (what `os.killpg` does in Python's shell backend), with
//!   [`Child::start_kill`] as the fallback on platforms without POSIX groups;
//! - **bounds the retained output** — each stream keeps at most
//!   [`ProcessShellRunner::max_output_bytes`] bytes and appends
//!   [`OUTPUT_TRUNCATION_NOTICE`]. The pipe is still drained to end-of-stream, so
//!   the child never blocks on a full buffer;
//! - **bounds the runtime** — the command is killed after
//!   [`ProcessShellRunner::timeout`] (Python has no timeout; ten minutes is the
//!   documented deviation), reported as [`ShellOutput::timed_out`];
//! - **is interruptible** — the command races its cancellation token, so a Ctrl-C
//!   in the REPL kills the group and returns instead of waiting for the command,
//!   reported as [`ShellOutput::cancelled`]. The direct child is always reaped
//!   before the call returns, so an interrupted command leaves no zombie and no
//!   descendant behind.
//!
//! Failure mapping: spawn and wait failures carry no domain meaning, so they are
//! wrapped with [`Error::external`], which keeps the original IO error reachable
//! as the [`std::error::Error::source`]. A command that runs but exits non-zero is
//! not an error here — it is data in [`ShellOutput::code`], as are a timeout and an
//! interrupt.

use std::path::Path;
use std::process::ExitStatus;
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_sysop::shell::registry::terminate_group;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::{Child, Command};
use tokio_util::sync::CancellationToken;

/// Bytes read from a pipe per `read` call while capturing.
const READ_CHUNK_SIZE: usize = 16 * 1024;

/// Default bound on the retained bytes of *each* captured stream.
pub const DEFAULT_MAX_OUTPUT_BYTES: usize = 256 * 1024;

/// Default runtime bound of one `!cmd` command.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(600);

/// Appended to a captured stream that hit [`DEFAULT_MAX_OUTPUT_BYTES`].
pub const OUTPUT_TRUNCATION_NOTICE: &str = "\n[output truncated]";

/// How long the readers may take to finish after the child has been reaped.
const READER_GRACE: Duration = Duration::from_secs(5);

/// Result of one shell command.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ShellOutput {
    /// The command line that was run, without the leading `!`.
    pub command: String,
    /// Captured standard output, decoded lossily as UTF-8.
    pub stdout: String,
    /// Captured standard error, decoded lossily as UTF-8.
    pub stderr: String,
    /// Exit code, or `None` when the process was terminated by a signal.
    pub code: Option<i32>,
    /// Whether the command exceeded the runner's timeout and was killed.
    pub timed_out: bool,
    /// Whether the command was killed by the caller's cancellation token.
    pub cancelled: bool,
}

impl ShellOutput {
    /// Returns `true` when the command exited with code 0.
    ///
    /// A signal termination (`code == None`), a timeout and an interrupt are
    /// never a success.
    #[must_use]
    pub fn success(&self) -> bool {
        self.code == Some(0) && !self.timed_out && !self.cancelled
    }

    /// Returns the failure status line, or `None` when the command succeeded.
    ///
    /// The line is `Interrupted` for a cancelled command, `Timed out` for one the
    /// runner killed, `Exit code: {n}` for a normal exit and `Terminated by signal`
    /// when the process did not report a code.
    #[must_use]
    pub fn status_line(&self) -> Option<String> {
        if self.success() {
            return None;
        }
        if self.cancelled {
            return Some("Interrupted".to_string());
        }
        if self.timed_out {
            return Some("Timed out".to_string());
        }
        Some(match self.code {
            Some(code) => format!("Exit code: {code}"),
            None => "Terminated by signal".to_string(),
        })
    }

    /// Returns `true` when the command produced no output and succeeded.
    ///
    /// The REPL uses this to skip the trailing blank line after a command that
    /// printed nothing, matching Python's silent `!cmd` output.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.stdout.is_empty() && self.stderr.is_empty() && self.success()
    }
}

/// Runs one command line, capturing stdout/stderr.
///
/// Implementations are `Send + Sync` so the REPL can hold one behind
/// `Arc<dyn ShellRunner>` and share it across turns.
#[async_trait]
pub trait ShellRunner: Send + Sync {
    /// Runs `command` with `cwd` as its working directory, interruptible through
    /// `cancel`.
    ///
    /// An implementation must return once `cancel` fires, with
    /// [`ShellOutput::cancelled`] set, and must have terminated the command (and
    /// the process group it leads) by then.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the shell cannot be spawned or its status
    /// cannot be collected. A non-zero exit status, a timeout and an interrupt are
    /// reported in [`ShellOutput`], not as an error.
    async fn run_cancellable(
        &self,
        command: &str,
        cwd: &Path,
        cancel: &CancellationToken,
    ) -> Result<ShellOutput>;

    /// Runs `command` without an interrupt source.
    ///
    /// # Errors
    ///
    /// Returns whatever [`ShellRunner::run_cancellable`] reports.
    async fn run(&self, command: &str, cwd: &Path) -> Result<ShellOutput> {
        self.run_cancellable(command, cwd, &CancellationToken::new())
            .await
    }
}

/// Default shell program of the host platform.
fn default_shell_program() -> &'static str {
    if cfg!(windows) {
        "cmd"
    } else {
        "sh"
    }
}

/// The flag that makes the shell run one command string.
fn shell_command_flag() -> &'static str {
    if cfg!(windows) {
        "/C"
    } else {
        "-c"
    }
}

/// [`ShellRunner`] backed by a real `sh -c` process.
///
/// The command inherits the CLI workspace as its working directory — that is a
/// convenience, not a sandbox: the shell can reach anything the CLI process can.
/// The runtime, the retained output and the interruption behaviour are bounded;
/// see the module docs.
#[derive(Debug, Clone)]
pub struct ProcessShellRunner {
    program: String,
    timeout: Duration,
    max_output_bytes: usize,
}

impl Default for ProcessShellRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl ProcessShellRunner {
    /// Creates a runner using the platform shell (`sh`, or `cmd` on Windows).
    #[must_use]
    pub fn new() -> Self {
        Self::with_program(default_shell_program())
    }

    /// Creates a runner using `program` as the shell binary.
    #[must_use]
    pub fn with_program(program: impl Into<String>) -> Self {
        Self {
            program: program.into(),
            timeout: DEFAULT_TIMEOUT,
            max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
        }
    }

    /// Overrides the runtime bound of one command.
    ///
    /// A zero duration is clamped to one millisecond, so a command always gets a
    /// chance to start.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout.max(Duration::from_millis(1));
        self
    }

    /// Overrides the retained bytes of each captured stream.
    ///
    /// The retained text is capped at exactly this many bytes (before the
    /// truncation notice); more output is drained from the pipe and discarded.
    #[must_use]
    pub fn with_max_output_bytes(mut self, max_output_bytes: usize) -> Self {
        self.max_output_bytes = max_output_bytes.max(1);
        self
    }

    /// Returns the shell program this runner spawns.
    #[must_use]
    pub fn program(&self) -> &str {
        &self.program
    }

    /// Returns the runtime bound of one command.
    #[must_use]
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Returns the retained-bytes bound of each captured stream.
    #[must_use]
    pub fn max_output_bytes(&self) -> usize {
        self.max_output_bytes
    }
}

#[async_trait]
impl ShellRunner for ProcessShellRunner {
    async fn run_cancellable(
        &self,
        command: &str,
        cwd: &Path,
        cancel: &CancellationToken,
    ) -> Result<ShellOutput> {
        let mut child = self.spawn(command, cwd)?;
        let mut group = ProcessGroupGuard::new(child.id());
        let limit = self.max_output_bytes;
        let stdout = child
            .stdout
            .take()
            .map(|pipe| tokio::spawn(drain(pipe, limit)));
        let stderr = child
            .stderr
            .take()
            .map(|pipe| tokio::spawn(drain(pipe, limit)));

        let outcome = tokio::select! {
            status = child.wait() => RunEnd::Exited(status),
            () = tokio::time::sleep(self.timeout) => RunEnd::TimedOut,
            () = cancel.cancelled() => RunEnd::Cancelled,
        };
        let (status, timed_out, cancelled) = match outcome {
            RunEnd::Exited(status) => {
                // The child is gone, so its process group id must not be signalled again: a
                // recycled pid could belong to an unrelated process.
                group.disarm();
                (status.ok(), false, false)
            }
            RunEnd::TimedOut | RunEnd::Cancelled => {
                group.terminate();
                let _ = child.start_kill();
                let status = child.wait().await.ok();
                group.disarm();
                (
                    status,
                    matches!(outcome, RunEnd::TimedOut),
                    matches!(outcome, RunEnd::Cancelled),
                )
            }
        };

        let (stdout, stdout_truncated) = collect(stdout).await;
        let (stderr, stderr_truncated) = collect(stderr).await;
        Ok(ShellOutput {
            command: command.to_string(),
            stdout: capped(stdout, stdout_truncated),
            stderr: capped(stderr, stderr_truncated),
            code: status.as_ref().and_then(ExitStatus::code),
            timed_out,
            cancelled,
        })
    }
}

impl ProcessShellRunner {
    /// Spawns the shell for `command` with piped output and its own process group.
    fn spawn(&self, command: &str, cwd: &Path) -> Result<Child> {
        let mut process = Command::new(&self.program);
        process
            .arg(shell_command_flag())
            .arg(command)
            .current_dir(cwd)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);
        #[cfg(unix)]
        process.process_group(0);
        process.spawn().map_err(Error::external)
    }
}

/// How one command ended.
#[derive(Debug)]
enum RunEnd {
    /// The child exited on its own.
    Exited(std::io::Result<ExitStatus>),
    /// The runtime bound elapsed.
    TimedOut,
    /// The caller's cancellation token fired.
    Cancelled,
}

/// The process group of one child, terminated at most once and never after the reap.
struct ProcessGroupGuard {
    /// Group id (= the child pid) while the child is unreaped.
    pid: Option<u32>,
    /// Set by the first termination so a second path is a no-op.
    terminated: bool,
}

impl ProcessGroupGuard {
    /// Guards the group of `pid`.
    fn new(pid: Option<u32>) -> Self {
        Self {
            pid,
            terminated: false,
        }
    }

    /// Kills the group, best effort, once.
    fn terminate(&mut self) {
        if self.terminated {
            return;
        }
        self.terminated = true;
        if let Some(pid) = self.pid {
            let delivered = terminate_group(pid);
            tracing::debug!(pid, delivered, "terminated the `!cmd` process group");
        }
    }

    /// Forgets the group after the child has been reaped.
    fn disarm(&mut self) {
        self.pid = None;
    }
}

impl Drop for ProcessGroupGuard {
    fn drop(&mut self) {
        // An aborted future (the REPL task itself cancelled) must not leave the pipeline
        // running: `kill_on_drop` only reaches the direct child.
        self.terminate();
    }
}

/// Reads `pipe` to end-of-stream, retaining at most `limit` bytes.
///
/// The pipe is always drained so the child never blocks on a full buffer; the excess is
/// counted and dropped.
async fn drain<R>(mut pipe: R, limit: usize) -> (String, bool)
where
    R: AsyncRead + Unpin,
{
    let mut retained: Vec<u8> = Vec::new();
    let mut truncated = false;
    let mut buffer = vec![0_u8; READ_CHUNK_SIZE];
    loop {
        match pipe.read(&mut buffer).await {
            Ok(0) => break,
            Ok(read) => {
                let room = limit.saturating_sub(retained.len());
                if read <= room {
                    retained.extend_from_slice(&buffer[..read]);
                } else {
                    retained.extend_from_slice(&buffer[..room]);
                    truncated = true;
                }
            }
            Err(error) => {
                tracing::debug!(%error, "stopped reading a `!cmd` output pipe");
                break;
            }
        }
    }
    (String::from_utf8_lossy(&retained).into_owned(), truncated)
}

/// Waits for one reader task, giving up after [`READER_GRACE`].
async fn collect(handle: Option<tokio::task::JoinHandle<(String, bool)>>) -> (String, bool) {
    let Some(handle) = handle else {
        return (String::new(), false);
    };
    match tokio::time::timeout(READER_GRACE, handle).await {
        Ok(Ok(collected)) => collected,
        Ok(Err(error)) => {
            tracing::warn!(%error, "a `!cmd` output reader task failed");
            (String::new(), false)
        }
        Err(_) => {
            tracing::warn!("a `!cmd` output pipe did not close after the command ended");
            (String::new(), false)
        }
    }
}

/// Appends the truncation notice to a stream that hit the retained-bytes bound.
fn capped(mut text: String, truncated: bool) -> String {
    if truncated {
        text.push_str(OUTPUT_TRUNCATION_NOTICE);
    }
    text
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use openjiuwen_common::error::ErrorKind;

    use super::*;

    /// Upper bound of every real-process assertion, so a hang fails instead of blocking.
    const BOUND: Duration = Duration::from_secs(20);

    /// A runner that returns a canned output; proves the trait is object-safe and
    /// lets the REPL suites run without spawning a process.
    struct FakeRunner {
        output: ShellOutput,
    }

    #[async_trait]
    impl ShellRunner for FakeRunner {
        async fn run_cancellable(
            &self,
            _command: &str,
            _cwd: &Path,
            _cancel: &CancellationToken,
        ) -> Result<ShellOutput> {
            Ok(self.output.clone())
        }
    }

    fn output(stdout: &str, stderr: &str, code: Option<i32>) -> ShellOutput {
        ShellOutput {
            command: "cmd".to_string(),
            stdout: stdout.to_string(),
            stderr: stderr.to_string(),
            code,
            ..ShellOutput::default()
        }
    }

    #[tokio::test]
    async fn a_fake_runner_is_usable_behind_a_trait_object() {
        let runner: Arc<dyn ShellRunner> = Arc::new(FakeRunner {
            output: output("hi\n", "", Some(0)),
        });
        let result = runner.run("printf hi", Path::new(".")).await.unwrap();
        assert_eq!(result.command, "cmd");
        assert_eq!(result.stdout, "hi\n");
        assert!(result.success());
    }

    #[test]
    fn success_status_line_and_emptiness_form_a_truth_table() {
        let ok = output("", "", Some(0));
        assert!(ok.success());
        assert_eq!(ok.status_line(), None);
        assert!(ok.is_empty());

        let silent_failure = output("", "", Some(3));
        assert!(!silent_failure.success());
        assert_eq!(
            silent_failure.status_line().as_deref(),
            Some("Exit code: 3")
        );
        assert!(!silent_failure.is_empty());

        let signal = output("", "", None);
        assert!(!signal.success());
        assert_eq!(
            signal.status_line().as_deref(),
            Some("Terminated by signal")
        );

        let printed = output("hello", "", Some(0));
        assert!(printed.success());
        assert!(!printed.is_empty());

        let errored = output("", "oops", Some(0));
        assert!(errored.success());
        assert!(!errored.is_empty());

        let default = ShellOutput::default();
        assert!(!default.success());
        assert!(!default.is_empty());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_real_shell_captures_stdout_and_a_zero_exit_code() {
        let temp = tempfile::tempdir().unwrap();
        let result = ProcessShellRunner::new()
            .run("printf hello", temp.path())
            .await
            .unwrap();
        assert_eq!(result.command, "printf hello");
        assert_eq!(result.stdout, "hello");
        assert_eq!(result.stderr, "");
        assert_eq!(result.code, Some(0));
        assert!(result.success());
        assert_eq!(result.status_line(), None);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_real_shell_reports_a_non_zero_exit_code() {
        let temp = tempfile::tempdir().unwrap();
        let result = ProcessShellRunner::default()
            .run("exit 3", temp.path())
            .await
            .unwrap();
        assert_eq!(result.code, Some(3));
        assert!(!result.success());
        assert_eq!(result.status_line().as_deref(), Some("Exit code: 3"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_real_shell_captures_stderr_separately() {
        let temp = tempfile::tempdir().unwrap();
        let result = ProcessShellRunner::new()
            .run("printf oops >&2", temp.path())
            .await
            .unwrap();
        assert_eq!(result.stdout, "");
        assert_eq!(result.stderr, "oops");
        assert_eq!(result.code, Some(0));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_real_shell_runs_in_the_given_working_directory() {
        let temp = tempfile::tempdir().unwrap();
        let expected = std::fs::canonicalize(temp.path()).unwrap();
        // `-P` prints the physical path, which is what `canonicalize` returns even
        // when an ancestor directory is a symlink (as on macOS).
        let result = ProcessShellRunner::new()
            .run("pwd -P", temp.path())
            .await
            .unwrap();
        assert_eq!(result.stdout.trim_end(), expected.to_string_lossy());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn an_output_flood_keeps_exactly_the_cap_plus_a_notice() {
        let temp = tempfile::tempdir().unwrap();
        let runner = ProcessShellRunner::new().with_max_output_bytes(64);
        assert_eq!(runner.max_output_bytes(), 64);
        let result = tokio::time::timeout(
            BOUND,
            runner.run("head -c 200000 /dev/zero | tr '\\0' x", temp.path()),
        )
        .await
        .expect("a flooding command must finish")
        .unwrap();
        assert_eq!(result.stdout.len(), 64 + OUTPUT_TRUNCATION_NOTICE.len());
        assert!(
            result.stdout.ends_with(OUTPUT_TRUNCATION_NOTICE),
            "the retained output must end with the notice: {:?}",
            &result.stdout[result.stdout.len().saturating_sub(40)..]
        );
        assert_eq!(result.code, Some(0));
        assert!(!result.timed_out && !result.cancelled);
        assert!(result.success());
        assert!(result.status_line().is_none());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_long_running_command_is_interrupted_through_the_token() {
        let temp = tempfile::tempdir().unwrap();
        let pid_file = temp.path().join("pid");
        let cancel = CancellationToken::new();
        let canceller = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(150)).await;
            canceller.cancel();
        });
        let runner = ProcessShellRunner::new();
        let started = std::time::Instant::now();
        let result = tokio::time::timeout(
            BOUND,
            runner.run_cancellable(
                &format!("echo $ > {}; exec sleep 30", pid_file.display()),
                temp.path(),
                &cancel,
            ),
        )
        .await
        .expect("an interrupted command must return")
        .unwrap();
        assert!(
            started.elapsed() < BOUND,
            "the interrupt must return promptly, not after the command"
        );
        assert!(result.cancelled, "the output must report the interrupt");
        assert!(!result.timed_out);
        assert!(!result.success());
        assert_eq!(result.status_line().as_deref(), Some("Interrupted"));

        let pid = std::fs::read_to_string(&pid_file)
            .expect("the command must have written its pid")
            .trim()
            .to_owned();
        let gone = tokio::time::timeout(BOUND, async {
            loop {
                if !process_exists(&pid) {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await;
        assert!(
            gone.is_ok(),
            "the interrupted child must be gone (pid {pid})"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_command_past_its_runtime_bound_times_out() {
        let temp = tempfile::tempdir().unwrap();
        let runner = ProcessShellRunner::new().with_timeout(Duration::from_millis(100));
        assert_eq!(runner.timeout(), Duration::from_millis(100));
        let result = tokio::time::timeout(BOUND, runner.run("sleep 30", temp.path()))
            .await
            .expect("a timed-out command must return")
            .unwrap();
        assert!(result.timed_out, "the output must report the timeout");
        assert!(!result.cancelled);
        assert!(!result.success());
        assert_eq!(result.status_line().as_deref(), Some("Timed out"));
    }

    #[tokio::test]
    async fn bounds_are_clamped_so_a_command_always_gets_a_chance_to_run() {
        let runner = ProcessShellRunner::new()
            .with_timeout(Duration::ZERO)
            .with_max_output_bytes(0);
        assert_eq!(runner.timeout(), Duration::from_millis(1));
        assert_eq!(runner.max_output_bytes(), 1);
    }

    #[test]
    fn timeout_and_interrupt_are_reported_as_failures() {
        let timed_out = ShellOutput {
            code: None,
            timed_out: true,
            ..ShellOutput::default()
        };
        assert!(!timed_out.success());
        assert_eq!(timed_out.status_line().as_deref(), Some("Timed out"));

        let cancelled = ShellOutput {
            code: None,
            cancelled: true,
            ..ShellOutput::default()
        };
        assert!(!cancelled.success());
        assert_eq!(cancelled.status_line().as_deref(), Some("Interrupted"));
        assert!(!cancelled.is_empty());
    }

    /// Reports whether a process with `pid` still exists (and is not a zombie).
    #[cfg(unix)]
    fn process_exists(pid: &str) -> bool {
        let Ok(pid) = pid.parse::<i32>() else {
            return false;
        };
        #[cfg(target_os = "linux")]
        {
            match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
                Ok(stat) => {
                    // Field 3 is the state; a zombie (`Z`) has been reaped by nobody, which is
                    // exactly what we assert does not happen.
                    stat.split_whitespace().nth(2) != Some("Z")
                }
                Err(_) => false,
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            // No portable probe: `kill -0` through the shell keeps the assertion meaningful.
            std::process::Command::new("kill")
                .arg("-0")
                .arg(pid.to_string())
                .stderr(std::process::Stdio::null())
                .status()
                .map(|status| status.success())
                .unwrap_or(false)
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_missing_shell_program_is_a_typed_error() {
        let temp = tempfile::tempdir().unwrap();
        let runner = ProcessShellRunner::with_program("no-such-shell-for-tests");
        assert_eq!(runner.program(), "no-such-shell-for-tests");
        let error = runner.run("printf hello", temp.path()).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::External);
    }
}
