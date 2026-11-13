//! The command seam every external side effect goes through.
//!
//! Python calls `subprocess`/`shutil.which` directly from the Docker environment
//! and the benchmark adapters, which is why its pipeline is only testable with a
//! real Docker daemon. The Rust port funnels every invocation through
//! [`CommandRunner`]:
//!
//! - [`ProcessCommandRunner`] — the real runner: spawns through
//!   `tokio::process`, captures both streams, enforces the timeout and maps a
//!   missing binary to [`Error::Unsupported`].
//! - [`FakeCommandRunner`] — a deterministic, scripted runner used by the
//!   pipeline's tests and available to consumers that need to exercise the
//!   pipeline without Docker or git.
//!
//! A missing program is deliberately *not* an I/O error: the runner returns
//! `Error::unsupported("<program> is not available: ...")`, so the pipeline can
//! report a typed deferred/absent-tool failure instead of a generic failure.
//! A non-zero exit is a normal [`ExecResult`]; only a spawn failure, a timeout,
//! or cancellation is an error.
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use tokio_util::sync::CancellationToken;

use super::models::{CommandSpec, ExecResult};

/// The seam the pipeline, the Docker environment, and the git preparer run
/// commands through.
///
/// Implementations must be safe to share across tasks (`Send + Sync`) and
/// `Debug` (the pipeline and the Docker environment derive `Debug`); the
/// pipeline holds one runner behind an [`Arc`](std::sync::Arc) for its whole
/// lifetime.
#[async_trait]
pub trait CommandRunner: Send + Sync + std::fmt::Debug {
    /// Runs `spec` and captures its output.
    ///
    /// Returns `Ok` for any completed process, including a non-zero exit code
    /// and a timeout (reported through [`ExecResult::timed_out`]). Returns
    /// [`Error::Cancelled`] when `cancel` is triggered, and
    /// [`Error::Unsupported`] when the program cannot be found.
    async fn run(&self, spec: &CommandSpec, cancel: &CancellationToken) -> Result<ExecResult>;

    /// Resolves `program` on `PATH`, returning `None` when it is absent.
    async fn which(&self, program: &str) -> Result<Option<PathBuf>>;
}

/// Returns [`Error::Cancelled`] when `cancel` has been triggered.
///
/// The pipeline, the git preparer, and the fake runner share this check so
/// cancellation is observed at every step boundary, not only inside a running
/// child process.
pub(crate) fn check_cancel(cancel: &CancellationToken) -> Result<()> {
    if cancel.is_cancelled() {
        Err(Error::cancelled())
    } else {
        Ok(())
    }
}

/// The real runner: one child process per [`CommandSpec`].
///
/// The child is spawned with `kill_on_drop`, so dropping the waiting future —
/// because the timeout elapsed or the token was cancelled — kills it instead of
/// leaking a process.
#[derive(Debug, Clone, Copy, Default)]
pub struct ProcessCommandRunner;

impl ProcessCommandRunner {
    /// Creates a runner that spawns real processes.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl CommandRunner for ProcessCommandRunner {
    async fn run(&self, spec: &CommandSpec, cancel: &CancellationToken) -> Result<ExecResult> {
        if spec.program.trim().is_empty() {
            return Err(Error::validation("command program must not be empty"));
        }
        if cancel.is_cancelled() {
            return Err(Error::cancelled());
        }
        let mut command = tokio::process::Command::new(&spec.program);
        command
            .args(&spec.args)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);
        if let Some(cwd) = &spec.cwd {
            command.current_dir(cwd);
        }
        for (key, value) in &spec.env {
            command.env(key, value);
        }
        let wait = command.output();
        tokio::pin!(wait);
        let deadline = async {
            match spec.timeout() {
                Some(limit) => tokio::time::sleep(limit).await,
                None => std::future::pending::<()>().await,
            }
        };
        tokio::pin!(deadline);
        tokio::select! {
            output = &mut wait => {
                let output = output.map_err(|error| spawn_error(&spec.program, &error))?;
                Ok(ExecResult {
                    stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
                    stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
                    returncode: output.status.code().unwrap_or(-1),
                    timed_out: false,
                })
            }
            () = &mut deadline => {
                let message = format!("`{}` timed out", spec.command_line());
                Ok(ExecResult::timeout(message))
            }
            () = cancel.cancelled() => Err(Error::cancelled()),
        }
    }

    async fn which(&self, program: &str) -> Result<Option<PathBuf>> {
        Ok(find_program(program))
    }
}

/// Maps a spawn failure: a missing binary becomes [`Error::Unsupported`], every
/// other spawn failure becomes [`Error::sys_op`].
fn spawn_error(program: &str, error: &std::io::Error) -> Error {
    if error.kind() == std::io::ErrorKind::NotFound {
        Error::unsupported(format!("{program} is not available: {error}"))
    } else {
        Error::sys_op(format!("failed to spawn {program}: {error}"))
    }
}

/// Resolves `program` against `PATH` without spawning a shell.
///
/// A program containing a path separator is checked as a path; otherwise every
/// `PATH` entry is tried in order. On Unix a candidate must be a file with at
/// least one execute bit set.
#[must_use]
pub fn find_program(program: &str) -> Option<PathBuf> {
    if program.trim().is_empty() {
        return None;
    }
    let candidate = Path::new(program);
    if candidate.is_absolute() || candidate.components().count() > 1 {
        return is_executable(candidate).then(|| candidate.to_path_buf());
    }
    let paths = std::env::var_os("PATH")?;
    std::env::split_paths(&paths)
        .map(|dir| dir.join(program))
        .find(|path| is_executable(path))
}

/// `true` when `path` is a file (and, on Unix, is executable).
fn is_executable(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        path.metadata()
            .is_ok_and(|metadata| metadata.permissions().mode() & 0o111 != 0)
    }
    #[cfg(not(unix))]
    {
        true
    }
}

/// Locks `mutex`, recovering the guard if a previous holder panicked.
///
/// The fake runner only guards short bookkeeping sections, so a poisoned lock
/// carries no invariant worth propagating; recovering keeps a failed test from
/// cascading into unrelated suites.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// A deterministic, scripted [`CommandRunner`] for tests and offline consumers.
///
/// Results are queued per `(program, first-argument prefix)` and handed out
/// first-in-first-out, so a test controls both the sequence of results and the
/// exact command lines the pipeline produced:
///
/// ```
/// use openjiuwen_agent_evolving::evaluator_pipeline::{
///     ExecResult, FakeCommandRunner,
/// };
///
/// let runner = FakeCommandRunner::new();
/// runner.push_result("docker", "run", ExecResult::ok("container-id\n"));
/// runner.push_result("docker", "rm", ExecResult::ok(""));
/// ```
///
/// Every invocation is recorded in order (see
/// [`invocations`](FakeCommandRunner::invocations)). An unscripted command
/// returns [`Error::validation`] unless the runner was built with
/// [`record_only`](FakeCommandRunner::record_only), in which case it succeeds
/// with empty output.
#[derive(Debug, Default)]
pub struct FakeCommandRunner {
    scripted: Mutex<BTreeMap<(String, String), VecDeque<ExecResult>>>,
    missing: Mutex<BTreeSet<String>>,
    invocations: Mutex<Vec<CommandSpec>>,
    record_only: bool,
}

impl FakeCommandRunner {
    /// Creates a runner with no scripted results.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Queues `result` for a command whose program is `program` and whose first
    /// argument starts with `arg_prefix`.
    ///
    /// An empty prefix matches any first argument, including a command with no
    /// arguments.
    pub fn push_result(
        &self,
        program: impl Into<String>,
        arg_prefix: impl Into<String>,
        result: ExecResult,
    ) {
        let key = (program.into(), arg_prefix.into());
        lock(&self.scripted)
            .entry(key)
            .or_default()
            .push_back(result);
    }

    /// Drops every queued result for `program`.
    pub fn clear_results(&self, program: &str) {
        lock(&self.scripted).retain(|(queued, _), _| queued != program);
    }

    /// Marks `program` as absent: [`run`](CommandRunner::run) returns
    /// [`Error::Unsupported`] and [`which`](CommandRunner::which) returns `None`.
    #[must_use]
    pub fn with_program_missing(self, program: impl Into<String>) -> Self {
        lock(&self.missing).insert(program.into());
        self
    }

    /// Makes every unscripted command succeed with empty output while still
    /// being recorded.
    #[must_use]
    pub fn record_only(mut self) -> Self {
        self.record_only = true;
        self
    }

    /// A snapshot of every command this runner has been asked to run, in order.
    ///
    /// A snapshot (rather than a borrowed slice) lets the runner stay behind an
    /// `Arc` and outlive the assertion.
    #[must_use]
    pub fn invocations(&self) -> Vec<CommandSpec> {
        lock(&self.invocations).clone()
    }

    /// The recorded commands rendered with [`CommandSpec::command_line`].
    #[must_use]
    pub fn command_lines(&self) -> Vec<String> {
        lock(&self.invocations)
            .iter()
            .map(CommandSpec::command_line)
            .collect()
    }

    /// Removes every recorded invocation.
    pub fn clear_invocations(&self) {
        lock(&self.invocations).clear();
    }

    /// `true` when `program` was declared absent.
    #[must_use]
    pub fn is_program_missing(&self, program: &str) -> bool {
        lock(&self.missing).contains(program)
    }
}

#[async_trait]
impl CommandRunner for FakeCommandRunner {
    async fn run(&self, spec: &CommandSpec, cancel: &CancellationToken) -> Result<ExecResult> {
        lock(&self.invocations).push(spec.clone());
        if cancel.is_cancelled() {
            return Err(Error::cancelled());
        }
        if self.is_program_missing(&spec.program) {
            return Err(Error::unsupported(format!(
                "{} is not available: declared missing by the fake runner",
                spec.program
            )));
        }
        let first_arg = spec.args.first().cloned().unwrap_or_default();
        let scripted = {
            let mut results = lock(&self.scripted);
            let key = results
                .keys()
                .find(|(program, prefix)| {
                    program == &spec.program && first_arg.starts_with(prefix.as_str())
                })
                .cloned();
            key.and_then(|key| results.get_mut(&key).and_then(VecDeque::pop_front))
        };
        if let Some(result) = scripted {
            return Ok(result);
        }
        if self.record_only {
            return Ok(ExecResult::ok(""));
        }
        Err(Error::validation(format!(
            "FakeCommandRunner has no scripted result for `{}`",
            spec.command_line()
        )))
    }

    async fn which(&self, program: &str) -> Result<Option<PathBuf>> {
        if self.is_program_missing(program) {
            Ok(None)
        } else {
            Ok(Some(PathBuf::from(format!("/fake/bin/{program}"))))
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn spec(program: &str, args: &[&str]) -> CommandSpec {
        CommandSpec::new(program).args(args.iter().copied())
    }

    #[tokio::test]
    async fn fake_runner_hands_out_scripted_results_in_order() {
        let runner = FakeCommandRunner::new();
        runner.push_result("docker", "run", ExecResult::ok("first\n"));
        runner.push_result("docker", "run", ExecResult::ok("second\n"));
        let cancel = CancellationToken::new();
        let first = runner
            .run(&spec("docker", &["run", "-d"]), &cancel)
            .await
            .unwrap();
        let second = runner
            .run(&spec("docker", &["run", "-d"]), &cancel)
            .await
            .unwrap();
        assert_eq!(first.stdout, "first\n");
        assert_eq!(second.stdout, "second\n");
        assert_eq!(runner.invocations().len(), 2);
        assert_eq!(runner.command_lines()[0], "docker run -d");
        let error = runner
            .run(&spec("docker", &["run", "-d"]), &cancel)
            .await
            .unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
    }

    #[tokio::test]
    async fn fake_runner_reports_absent_and_unscripted_programs() {
        let missing = FakeCommandRunner::new().with_program_missing("docker");
        let cancel = CancellationToken::new();
        assert!(missing.which("docker").await.unwrap().is_none());
        assert!(missing.which("git").await.unwrap().is_some());
        let error = missing
            .run(&spec("docker", &["run"]), &cancel)
            .await
            .unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Unsupported
        );
        assert!(error.to_string().contains("docker"));

        let recorded = FakeCommandRunner::new().record_only();
        let result = recorded
            .run(&spec("git", &["rev-parse"]), &cancel)
            .await
            .unwrap();
        assert!(result.success());
        assert_eq!(result.stdout, "");
    }

    #[tokio::test]
    async fn fake_runner_honours_cancellation() {
        let runner = FakeCommandRunner::new();
        let cancel = CancellationToken::new();
        cancel.cancel();
        let error = runner
            .run(&spec("docker", &["run"]), &cancel)
            .await
            .unwrap_err();
        assert!(error.is_cancelled());
        assert_eq!(runner.invocations().len(), 1);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn process_runner_captures_output_and_exit_codes() {
        let runner = ProcessCommandRunner::new();
        let cancel = CancellationToken::new();
        let ok = runner
            .run(&spec("sh", &["-c", "printf hello"]), &cancel)
            .await
            .unwrap();
        assert_eq!(ok.stdout, "hello");
        assert!(ok.success());
        let failed = runner
            .run(&spec("sh", &["-c", "echo problem >&2; exit 3"]), &cancel)
            .await
            .unwrap();
        assert_eq!(failed.returncode, 3);
        assert_eq!(failed.stderr.trim(), "problem");
        assert!(!failed.success());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn process_runner_reports_absent_programs_as_unsupported() {
        let runner = ProcessCommandRunner::new();
        let cancel = CancellationToken::new();
        let error = runner
            .run(
                &spec("openjiuwen-definitely-missing-binary", &["--version"]),
                &cancel,
            )
            .await
            .unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Unsupported
        );
        assert!(error.to_string().contains("not available"));
        assert!(runner
            .which("openjiuwen-definitely-missing-binary")
            .await
            .unwrap()
            .is_none());
        assert!(runner.which("sh").await.unwrap().is_some());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn process_runner_kills_a_timed_out_command() {
        let runner = ProcessCommandRunner::new();
        let cancel = CancellationToken::new();
        let result = runner
            .run(
                &spec("sh", &["-c", "sleep 5"]).with_timeout(std::time::Duration::from_millis(50)),
                &cancel,
            )
            .await
            .unwrap();
        assert!(result.timed_out);
        assert_eq!(result.returncode, -1);
        assert!(!result.success());
    }
}
