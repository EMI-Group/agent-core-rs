//! Local shell backend: commands, argv, streaming, and background execution.
//!
//! Python parity: `LocalShellOperation`
//! (`openjiuwen/core/sys_operation/local/shell_operation.py`). Every call passes the same
//! gates before a child process exists — empty-command check, first-token allowlist plus
//! optional regex deny list, working-directory resolution through [`PathPolicy`], sandbox
//! containment of the absolute path literals a command mentions, and interpreter
//! resolution — and every failure is an [`crate::result::OpResult`] value carrying Python's
//! status code `199004`, never a panic. [`execute_argv`](ShellOperation::execute_argv)
//! bypasses the shell entirely: its arguments are data, so no quoting, globbing, or
//! redirection is interpreted and the shell-text deny list is not applied.
//!
//! ```
//! use openjiuwen_sysop::config::LocalWorkConfig;
//! use openjiuwen_sysop::shell::LocalShellOperation;
//!
//! let backend = LocalShellOperation::new(LocalWorkConfig::default()).unwrap();
//! assert!(backend.filter().allows_command("echo hi"));
//! assert!(!backend.filter().allows_command("ssh host"));
//! ```
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::{Child, Command};
use tokio_util::sync::CancellationToken;

use openjiuwen_common::error::Result;

use crate::config::LocalWorkConfig;
use crate::constants;
use crate::env::{self, ExecutionEnvironment};
use crate::operation::{OpStream, ShellOperation};
use crate::result::{shell_failure, OpResult};
use crate::shell::allowlist::CommandFilter;
use crate::shell::plan;
use crate::shell::registry::{Registration, ShellProcessRegistry};
use crate::shell::types::{
    ShellArgvParams, ShellBackgroundOutput, ShellChunk, ShellExecParams, ShellOutput,
};
use crate::status::{self, SysOpStatus};
use crate::util::format_path;
use crate::workspace::{self, PathPolicy};

/// Marker appended to backgrounded stderr that exceeded the retention budget.
const STDERR_TRUNCATION_MARKER: &str = "\n[stderr truncated]";
/// Window used to detect a backgrounded command that exits immediately.
///
/// Python's `grace` defaults to 3 s; reporting a pid once the child survived this short
/// window keeps `execute_cmd_background` fast while still catching launch failures.
const EARLY_EXIT_GRACE: Duration = Duration::from_millis(150);
/// Poll interval used while waiting out [`EARLY_EXIT_GRACE`].
const EARLY_EXIT_POLL: Duration = Duration::from_millis(10);
/// How long the early-exit path waits for the stderr reader to finish.
const STDERR_JOIN_GRACE: Duration = Duration::from_millis(100);

/// The local shell backend: one configuration, one path policy, one command filter.
pub struct LocalShellOperation {
    config: LocalWorkConfig,
    policy: PathPolicy,
    filter: CommandFilter,
    cancellation: Option<CancellationToken>,
}

impl LocalShellOperation {
    /// Builds the backend; validates `config` and compiles its dangerous patterns.
    ///
    /// # Errors
    ///
    /// Returns `ErrorKind::Validation` when `config` is invalid, for example when a
    /// `dangerous_patterns` entry is not a valid regular expression.
    pub fn new(config: LocalWorkConfig) -> Result<Self> {
        config.validate()?;
        let filter = CommandFilter::new(&config)?;
        let policy = PathPolicy::new(&config);
        Ok(Self {
            config,
            policy,
            filter,
            cancellation: None,
        })
    }

    /// Returns this backend with an extra per-instance cancellation token.
    ///
    /// It is combined with the per-session token of [`ShellProcessRegistry::global`], so
    /// either one interrupts a running command.
    #[must_use]
    pub fn with_cancellation(mut self, token: CancellationToken) -> Self {
        self.cancellation = Some(token);
        self
    }

    /// The configuration this backend was built from.
    #[must_use]
    pub fn config(&self) -> &LocalWorkConfig {
        &self.config
    }

    /// The path policy applied to working directories and path literals.
    #[must_use]
    pub fn policy(&self) -> &PathPolicy {
        &self.policy
    }

    /// The command allowlist/deny-list filter applied before spawning.
    #[must_use]
    pub fn filter(&self) -> &CommandFilter {
        &self.filter
    }

    /// Resolves the working directory, enforcing the sandbox roots (Python `_resolve_cwd`).
    async fn resolve_cwd(&self, cwd: Option<&str>) -> std::result::Result<PathBuf, Rejection> {
        let path = match cwd {
            None => self.policy.workspace().to_path_buf(),
            // Python maps a sandbox violation to its `Access denied: ...` text.
            Some(requested) => self
                .policy
                .resolve_existing(requested)
                .await
                .map_err(|error| Rejection::new(error.message().to_string()))?,
        };
        let display = format_path(&path);
        if !tokio::fs::metadata(&path)
            .await
            .is_ok_and(|meta| meta.is_dir())
        {
            let reason = format!("working directory does not exist: {display}");
            return Err(Rejection::with_cwd(reason, display));
        }
        Ok(path)
    }

    /// Applies the gates shared by `execute_cmd`, `execute_cmd_stream`, and
    /// `execute_cmd_background` (Python `_check_command_safety`, `_check_allowlist`,
    /// `_check_shell_sandbox`, `_build_shell_command`) and resolves the spawn specification.
    async fn prepare(&self, params: &ShellExecParams) -> std::result::Result<Prepared, Rejection> {
        let ShellExecParams {
            command,
            shell_type,
            environment,
            ..
        } = params;
        if command.trim().is_empty() {
            return Err(Rejection::new("command can not be empty"));
        }
        if let Err(reason) = self.filter.check(command) {
            return Err(Rejection::new(reason));
        }
        let cwd_path = self.resolve_cwd(params.cwd.as_deref()).await?;
        let cwd_display = format_path(&cwd_path);
        if self.policy.is_restricted() {
            for literal in plan::extract_absolute_paths(command) {
                if plan::is_ignored_absolute_path(&literal) {
                    continue;
                }
                let normalized = workspace::canonicalize_containing(Path::new(&literal)).await;
                if let Err(error) = self.policy.ensure_contained(&literal, &normalized) {
                    let reason = error.message().to_string();
                    return Err(Rejection::with_cwd(reason, cwd_display));
                }
            }
        }
        let plan = plan::resolve_plan(*shell_type, command)
            .map_err(|reason| Rejection::with_cwd(reason, cwd_display))?;
        let environment = resolve_environment(&self.config, environment);
        Ok(Prepared {
            program: plan.program,
            args: plan.args,
            cwd: cwd_path,
            environment,
        })
    }

    /// Builds the cancellation tokens for a run plus the guard that deregisters the
    /// session token when the run ends.
    fn cancellation_tokens(
        &self,
        session_id: Option<&str>,
    ) -> (Vec<CancellationToken>, Option<Registration>) {
        let mut tokens = Vec::new();
        if let Some(token) = &self.cancellation {
            tokens.push(token.clone());
        }
        let registration = match session_id {
            Some(id) if !id.trim().is_empty() => {
                let token = CancellationToken::new();
                tokens.push(token.clone());
                Some(ShellProcessRegistry::global().register(id, token))
            }
            _ => None,
        };
        (tokens, registration)
    }

    /// Spawns a prepared command (no shell involved) and captures its output.
    async fn run_prepared(
        &self,
        execution: &str,
        command: &str,
        prepared: Prepared,
        session_id: Option<&str>,
        timeout_secs: u64,
    ) -> OpResult<ShellOutput> {
        let cwd = format_path(&prepared.cwd);
        let mut child_command = Command::new(&prepared.program);
        child_command
            .args(&prepared.args)
            .current_dir(&prepared.cwd);
        process::apply_environment(&mut child_command, &prepared.environment);
        let child = match process::spawn_child(child_command) {
            Ok(child) => child,
            Err(error) => {
                tracing::warn!(execution, %error, "failed to start shell command");
                let reason = format!("failed to start command: {error}");
                return command_failure(execution, &reason, command, &cwd);
            }
        };
        let (tokens, _registration) = self.cancellation_tokens(session_id);
        let max_output_bytes = self.config.max_output_bytes;
        let options = process::RunOptions {
            timeout_secs,
            max_output_bytes,
            cancellation: tokens,
        };
        let outcome = process::run_captured(child, options).await;
        let timed_out = outcome.timed_out;
        let cancelled = outcome.cancelled;
        let output = ShellOutput {
            command: command.to_string(),
            cwd,
            exit_code: outcome.exit_code,
            truncated: outcome.stdout.truncated || outcome.stderr.truncated,
            stdout: outcome.stdout.text,
            stderr: outcome.stderr.text,
            timed_out,
            cancelled,
            duration_ms: outcome.duration_ms,
        };
        if timed_out {
            tracing::warn!(execution, timeout_secs, "shell command timed out");
            let reason = format!("execution timeout after {timeout_secs} seconds");
            return command_failure_with_data(execution, &reason, output);
        }
        if cancelled {
            tracing::warn!(execution, "shell command cancelled");
            return command_failure_with_data(execution, "execution cancelled by user", output);
        }
        OpResult::success(output)
    }
}

#[async_trait]
impl ShellOperation for LocalShellOperation {
    async fn execute_cmd(&self, params: ShellExecParams) -> OpResult<ShellOutput> {
        let timeout = params.effective_timeout_secs(self.config.default_timeout_secs);
        let prepared = match self.prepare(&params).await {
            Ok(prepared) => prepared,
            Err(rejection) => {
                tracing::warn!(reason = %rejection.reason, "shell command rejected");
                let command = &params.command;
                return command_failure("execute_cmd", &rejection.reason, command, &rejection.cwd);
            }
        };
        let session = params.session_id.as_deref();
        self.run_prepared("execute_cmd", &params.command, prepared, session, timeout)
            .await
    }

    fn execute_cmd_stream(self: Arc<Self>, params: ShellExecParams) -> OpStream<ShellChunk> {
        Box::pin(async_stream::stream! {
            let timeout = params.effective_timeout_secs(self.config.default_timeout_secs);
            let prepared = match self.prepare(&params).await {
                Ok(prepared) => prepared,
                Err(rejection) => {
                    yield stream_failure("execute_cmd_stream", &rejection.reason, 0);
                    return;
                }
            };
            let mut command = Command::new(&prepared.program);
            command.args(&prepared.args).current_dir(&prepared.cwd);
            process::apply_environment(&mut command, &prepared.environment);
            let child = match process::spawn_child(command) {
                Ok(child) => child,
                Err(error) => {
                    let reason = format!("failed to start command: {error}");
                    yield stream_failure("execute_cmd_stream", &reason, 0);
                    return;
                }
            };
            let (tokens, _registration) = self.cancellation_tokens(params.session_id.as_deref());
            let max_output_bytes = self.config.max_output_bytes;
            let options = process::RunOptions { timeout_secs: timeout, max_output_bytes, cancellation: tokens };
            let mut events = process::stream_child(child, options);
            let mut index = 0_u64;
            while let Some(event) = futures::StreamExt::next(&mut events).await {
                let item = match event {
                    process::EngineEvent::Data { kind, text } => OpResult {
                        code: SysOpStatus::Success.code(),
                        message: format!("Get {} stream successfully", kind.as_str()),
                        data: Some(ShellChunk::new(text, kind, index)),
                    },
                    process::EngineEvent::Exit { code } => OpResult {
                        code: SysOpStatus::Success.code(),
                        message: "Command executed successfully".to_string(),
                        data: Some(ShellChunk::terminal(index, code)),
                    },
                    process::EngineEvent::TimedOut { .. } => stream_failure(
                        "execute_cmd_stream",
                        &format!("execution timeout after {timeout} seconds"),
                        index,
                    ),
                    process::EngineEvent::Cancelled { .. } => stream_failure(
                        "execute_cmd_stream",
                        "execution cancelled by user",
                        index,
                    ),
                };
                index += 1;
                yield item;
            }
        })
    }

    async fn execute_cmd_background(
        &self,
        params: ShellExecParams,
    ) -> OpResult<ShellBackgroundOutput> {
        let prepared = match self.prepare(&params).await {
            Ok(prepared) => prepared,
            Err(rejection) => {
                tracing::warn!(reason = %rejection.reason, "background command rejected");
                let (command, cwd) = (&params.command, rejection.cwd.as_str());
                return background_failure(command, cwd, None, None, &rejection.reason);
            }
        };
        let command = params.command;
        let cwd = format_path(&prepared.cwd);
        let mut child = match spawn_background(&prepared) {
            Ok(child) => child,
            Err(error) => {
                tracing::warn!(%error, "failed to start background command");
                let reason = format!("background command failed: {error}");
                return background_failure(&command, &cwd, None, Some(&reason), &reason);
            }
        };
        let pid = child.id();
        let sink = Arc::new(Mutex::new(StderrSink::default()));
        let limit = constants::DEFAULT_STDERR_RETENTION_BYTES;
        let stderr_task = child
            .stderr
            .take()
            .map(|pipe| tokio::spawn(drain_stderr(pipe, limit, Arc::clone(&sink))));
        let exit = match wait_grace(&mut child).await {
            Ok(exit) => exit,
            Err(error) => {
                let _ = child.start_kill();
                let reason = format!("background command failed: {error}");
                return background_failure(&command, &cwd, pid, Some(&reason), &reason);
            }
        };
        match exit {
            Some(status) if !status.success() => {
                let early = background_early_exit(pid, status, stderr_task, &sink).await;
                let reason = format!("background command failed: {early}");
                return background_failure(&command, &cwd, pid, Some(&early), &reason);
            }
            Some(_) => {}
            // Keep draining stderr and reap the child so no zombie is left behind.
            None => {
                tokio::spawn(async move {
                    let _ = child.wait().await;
                    if let Some(task) = stderr_task {
                        let _ = task.await;
                    }
                });
            }
        }
        OpResult::success(ShellBackgroundOutput {
            command,
            cwd,
            pid,
            error: None,
        })
    }

    async fn execute_argv(&self, params: ShellArgvParams) -> OpResult<ShellOutput> {
        let command = params.argv.join(" ");
        let Some(program) = params.argv.first() else {
            return command_failure("execute_argv", "argv must not be empty", &command, "");
        };
        // Only the program is policy-checked: the arguments are data, never shell text.
        if !self.filter.allows_program(program) {
            let reason = "command not allowed by allowlist";
            return command_failure("execute_argv", reason, &command, "");
        }
        let cwd = match self.resolve_cwd(params.cwd.as_deref()).await {
            Ok(path) => path,
            Err(rejection) => {
                let (reason, known) = (&rejection.reason, rejection.cwd.as_str());
                return command_failure("execute_argv", reason, &command, known);
            }
        };
        let timeout = params.effective_timeout_secs(self.config.default_timeout_secs);
        let prepared = Prepared {
            program: program.clone(),
            args: params.argv.get(1..).unwrap_or_default().to_vec(),
            cwd,
            environment: resolve_environment(&self.config, &params.environment),
        };
        let session = params.session_id.as_deref();
        self.run_prepared("execute_argv", &command, prepared, session, timeout)
            .await
    }
}

/// A command that passed the policy gates, with interpreter, cwd, and environment resolved.
struct Prepared {
    program: String,
    args: Vec<String>,
    cwd: PathBuf,
    environment: Vec<(String, String)>,
}

/// A rejected command: Python's reason text plus the working directory when it was known.
struct Rejection {
    reason: String,
    cwd: String,
}

impl Rejection {
    /// A rejection raised before the working directory is known.
    fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
            cwd: String::new(),
        }
    }
    /// A rejection raised after the working directory was resolved.
    fn with_cwd(reason: impl Into<String>, cwd: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
            cwd: cwd.into(),
        }
    }
}

/// Resolves the child environment for `environment` under the configured policy.
fn resolve_environment(
    config: &LocalWorkConfig,
    environment: &BTreeMap<String, Option<String>>,
) -> Vec<(String, String)> {
    let overlay = ExecutionEnvironment::default();
    env::resolve_environment(&config.env_policy, &overlay, Some(environment))
}

/// Builds a shell failure whose payload carries the command, cwd, and `exit_code = -1`.
fn command_failure(
    execution: &str,
    reason: &str,
    command: &str,
    cwd: &str,
) -> OpResult<ShellOutput> {
    let output = ShellOutput {
        command: command.into(),
        cwd: cwd.into(),
        ..ShellOutput::default()
    };
    let mut failure = shell_failure::<ShellOutput>(execution, reason);
    failure.data = Some(output);
    failure
}

/// Builds a shell failure that retains a partially filled [`ShellOutput`].
fn command_failure_with_data(
    execution: &str,
    reason: &str,
    output: ShellOutput,
) -> OpResult<ShellOutput> {
    let mut failure = shell_failure::<ShellOutput>(execution, reason);
    failure.data = Some(output);
    failure
}

/// Builds a background failure carrying the command, cwd, pid, and early-exit message.
fn background_failure(
    command: &str,
    cwd: &str,
    pid: Option<u32>,
    error: Option<&str>,
    reason: &str,
) -> OpResult<ShellBackgroundOutput> {
    let mut failure = shell_failure::<ShellBackgroundOutput>("execute_cmd_background", reason);
    let data = ShellBackgroundOutput {
        command: command.into(),
        cwd: cwd.into(),
        pid,
        error: error.map(str::to_string),
    };
    failure.data = Some(data);
    failure
}

/// Builds the single error chunk a streamed shell call yields when a gate rejects the run.
fn stream_failure(execution: &str, reason: &str, chunk_index: u64) -> OpResult<ShellChunk> {
    let data = Some(ShellChunk::terminal(chunk_index, -1));
    let code = SysOpStatus::ShellExecutionError.code();
    OpResult {
        code,
        message: status::shell_execution_message(execution, reason),
        data,
    }
}

/// Spawns a detached background child: no stdin, discarded stdout, piped stderr.
fn spawn_background(prepared: &Prepared) -> std::io::Result<Child> {
    let mut command = Command::new(&prepared.program);
    command
        .args(&prepared.args)
        .current_dir(&prepared.cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(false);
    #[cfg(unix)]
    command.process_group(0);
    process::apply_environment(&mut command, &prepared.environment);
    command.spawn()
}

/// Polls `child` for an early exit during [`EARLY_EXIT_GRACE`].
async fn wait_grace(child: &mut Child) -> std::io::Result<Option<ExitStatus>> {
    let deadline = Instant::now() + EARLY_EXIT_GRACE;
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(Some(status));
        }
        if Instant::now() >= deadline {
            return Ok(None);
        }
        tokio::time::sleep(EARLY_EXIT_POLL).await;
    }
}

/// Builds Python's early-exit message (`process exited early with code <n>: <stderr>`).
async fn background_early_exit(
    pid: Option<u32>,
    status: ExitStatus,
    stderr_task: Option<tokio::task::JoinHandle<String>>,
    sink: &Arc<Mutex<StderrSink>>,
) -> String {
    let code = status.code().unwrap_or(-1);
    let stderr = collect_stderr(stderr_task, sink).await;
    let trimmed = stderr.trim();
    let message = if trimmed.is_empty() {
        format!("process exited early with code {code}")
    } else {
        format!("process exited early with code {code}: {trimmed}")
    };
    tracing::warn!(pid, reason = %message, "background command exited early");
    message
}

/// Retained stderr of a backgrounded command, shared with its reader task.
#[derive(Debug, Default)]
struct StderrSink {
    bytes: Vec<u8>,
    truncated: bool,
}

/// Locks the stderr sink, recovering from a poisoned lock.
fn lock_sink(sink: &Mutex<StderrSink>) -> std::sync::MutexGuard<'_, StderrSink> {
    sink.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Drains `pipe` into `sink`, retaining at most `limit` bytes, and returns the final text.
async fn drain_stderr<R>(mut pipe: R, limit: u64, sink: Arc<Mutex<StderrSink>>) -> String
where
    R: AsyncRead + Unpin,
{
    let limit = usize::try_from(limit).unwrap_or(usize::MAX);
    let mut buffer = vec![0_u8; constants::DEFAULT_TEXT_CHUNK_SIZE];
    loop {
        let read = match pipe.read(&mut buffer).await {
            // EOF or a read failure ends the capture and keeps what was retained.
            Ok(0) | Err(_) => break,
            Ok(read) => read,
        };
        let mut state = lock_sink(&sink);
        let take = limit.saturating_sub(state.bytes.len()).min(read);
        if take > 0 {
            state.bytes.extend_from_slice(&buffer[..take]);
        }
        state.truncated |= take < read;
    }
    let mut text = String::from_utf8_lossy(&lock_sink(&sink).bytes).into_owned();
    if lock_sink(&sink).truncated {
        text.push_str(STDERR_TRUNCATION_MARKER);
    }
    text
}

/// Awaits the stderr reader, falling back to the shared buffer when it is still busy.
async fn collect_stderr(
    task: Option<tokio::task::JoinHandle<String>>,
    sink: &Arc<Mutex<StderrSink>>,
) -> String {
    let Some(mut task) = task else {
        return String::new();
    };
    if let Ok(Ok(text)) = tokio::time::timeout(STDERR_JOIN_GRACE, &mut task).await {
        return text;
    }
    task.abort();
    let state = lock_sink(sink);
    let mut text = String::from_utf8_lossy(&state.bytes).into_owned();
    if state.truncated {
        text.push_str(STDERR_TRUNCATION_MARKER);
    }
    text
}

/// Shared child-process engine used by the shell backend and (later) the code backend.
///
/// It owns the three behaviors every command or snippet run needs: a hardened spawn (no
/// stdin, piped output, own process group), bounded output capture, and timeout or
/// cancellation handling that kills the child exactly once and always reaps it.
pub(crate) mod process {
    use std::future::Future;
    use std::pin::Pin;
    use std::process::{ExitStatus, Stdio};
    use std::time::{Duration, Instant};

    use futures::stream::BoxStream;
    use tokio::io::{AsyncRead, AsyncReadExt};
    use tokio::process::{Child, Command};
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;

    use crate::constants;
    use crate::shell::types::StreamKind;

    /// Bytes read from a pipe per `read` call while capturing.
    const READ_CHUNK_SIZE: usize = 65_536;
    /// How long to wait for a pipe reader once the child has been reaped.
    const READER_GRACE: Duration = Duration::from_secs(5);
    /// Capacity of the streamed-event channel.
    const EVENT_CHANNEL_CAPACITY: usize = 64;

    /// A boxed, sendable future used for the deadline and cancellation branches.
    type EngineFuture = Pin<Box<dyn Future<Output = ()> + Send>>;

    /// Text captured from one child stream.
    #[derive(Debug, Default, Clone, PartialEq, Eq)]
    pub(crate) struct CapturedText {
        pub(crate) text: String,
        pub(crate) truncated: bool,
    }

    /// Limits and cancellation tokens applied to one child run.
    #[derive(Debug, Clone, Default)]
    pub(crate) struct RunOptions {
        pub(crate) timeout_secs: u64,
        pub(crate) max_output_bytes: u64,
        pub(crate) cancellation: Vec<CancellationToken>,
    }

    /// The outcome of a captured child run (`exit_code` is `-1` when killed by a signal).
    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub(crate) struct RunOutcome {
        pub(crate) exit_code: i32,
        pub(crate) stdout: CapturedText,
        pub(crate) stderr: CapturedText,
        pub(crate) timed_out: bool,
        pub(crate) cancelled: bool,
        pub(crate) duration_ms: u64,
    }

    /// One event of a streamed child run.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) enum EngineEvent {
        /// Output arrived on a pipe.
        Data { kind: StreamKind, text: String },
        /// The child exited; `code` is its exit status.
        Exit { code: i32 },
        /// The timeout expired and the child was killed.
        TimedOut { code: i32 },
        /// A cancellation token fired and the child was killed.
        Cancelled { code: i32 },
    }

    /// Spawns `command` with the hardened child configuration: stdin closed, both output
    /// pipes captured, killed on drop, and (on Unix) in its own process group so the
    /// descendants of a shell die with it.
    pub(crate) fn spawn_child(mut command: Command) -> std::io::Result<Child> {
        command
            .kill_on_drop(true)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        command.process_group(0);
        command.spawn()
    }

    /// Replaces the child environment with `pairs` (already policy-scoped by the caller).
    pub(crate) fn apply_environment(command: &mut Command, pairs: &[(String, String)]) {
        let envs = pairs
            .iter()
            .map(|(name, value)| (name.as_str(), value.as_str()));
        command.env_clear().envs(envs);
    }

    /// Runs `child` to completion, capturing bounded output.
    ///
    /// Output is drained concurrently with the wait, so a chatty child cannot deadlock on
    /// a full pipe. A timeout or a cancellation token kills the child, keeps the partial
    /// output, and sets the matching flag.
    pub(crate) async fn run_captured(mut child: Child, options: RunOptions) -> RunOutcome {
        let started = Instant::now();
        let max_output_bytes = options.max_output_bytes;
        let stdout_task = tokio::spawn(read_bounded(child.stdout.take(), max_output_bytes));
        let stderr_task = tokio::spawn(read_bounded(child.stderr.take(), max_output_bytes));
        let deadline = deadline_future(options.timeout_secs);
        let cancel = cancellation_future(&options.cancellation);
        let mut timed_out = false;
        let mut cancelled = false;
        // The branch handlers must not touch `child`: the wait future borrows it.
        tokio::select! {
            _ = child.wait() => {}
            () = deadline => timed_out = true,
            () = cancel => cancelled = true,
        }
        let _ = child.start_kill();
        let exit_code = exit_code_of(child.wait().await);
        let stdout = join_reader(stdout_task).await;
        let stderr = join_reader(stderr_task).await;
        let duration_ms = elapsed_ms(started);
        RunOutcome {
            exit_code,
            stdout,
            stderr,
            timed_out,
            cancelled,
            duration_ms,
        }
    }

    /// Runs `child`, yielding output events until it exits or the run is interrupted.
    ///
    /// The stream always ends with exactly one terminal event: [`EngineEvent::Exit`],
    /// [`EngineEvent::TimedOut`], or [`EngineEvent::Cancelled`]. A reader failure ends the
    /// affected pipe instead of panicking, and a per-stream byte budget emits
    /// [`constants::OUTPUT_TRUNCATION_MARKER`] once before discarding further bytes.
    pub(crate) fn stream_child(
        mut child: Child,
        options: RunOptions,
    ) -> BoxStream<'static, EngineEvent> {
        Box::pin(async_stream::stream! {
            let max_output_bytes = options.max_output_bytes;
            let (sender, mut receiver) = mpsc::channel(EVENT_CHANNEL_CAPACITY);
            if let Some(pipe) = child.stdout.take() {
                tokio::spawn(stream_reader(pipe, StreamKind::Stdout, max_output_bytes, sender.clone()));
            }
            if let Some(pipe) = child.stderr.take() {
                tokio::spawn(stream_reader(pipe, StreamKind::Stderr, max_output_bytes, sender.clone()));
            }
            // Only the reader tasks may keep the channel open.
            drop(sender);
            let mut deadline = deadline_future(options.timeout_secs);
            let mut cancel = cancellation_future(&options.cancellation);
            let mut interrupted = None;
            loop {
                let mut event: Option<EngineEvent> = None;
                let mut finished = false;
                // `yield` is not allowed inside a branch, so the event is buffered.
                tokio::select! {
                    incoming = receiver.recv() => match incoming {
                        Some(item) => event = Some(item),
                        None => finished = true,
                    },
                    () = &mut deadline => {
                        interrupted = Some(EngineEvent::TimedOut { code: -1 });
                        finished = true;
                    }
                    () = &mut cancel => {
                        interrupted = Some(EngineEvent::Cancelled { code: -1 });
                        finished = true;
                    }
                }
                if let Some(item) = event {
                    yield item;
                }
                if finished {
                    break;
                }
            }
            if interrupted.is_some() {
                let _ = child.start_kill();
            }
            let code = exit_code_of(child.wait().await);
            match interrupted {
                Some(event) => yield event,
                None => yield EngineEvent::Exit { code },
            }
        })
    }

    /// Reads `pipe`, sending bounded UTF-8 chunks to `sender`.
    async fn stream_reader<R>(
        mut pipe: R,
        kind: StreamKind,
        max_output_bytes: u64,
        sender: mpsc::Sender<EngineEvent>,
    ) where
        R: AsyncRead + Unpin,
    {
        let limit = usize::try_from(max_output_bytes).unwrap_or(usize::MAX);
        let mut produced = 0_usize;
        let mut marked = false;
        let mut decoder = Utf8Decoder::default();
        let mut buffer = vec![0_u8; constants::DEFAULT_TEXT_CHUNK_SIZE];
        loop {
            let read = match pipe.read(&mut buffer).await {
                // EOF and read failures both mean "no more data"; dropping the sender
                // signals the end of this pipe to the generator.
                Ok(0) | Err(_) => break,
                Ok(read) => read,
            };
            if marked {
                // Past the budget: keep draining so the child never blocks on a full pipe.
                continue;
            }
            let take = limit.saturating_sub(produced).min(read);
            if take > 0 {
                produced += take;
                let text = decoder.decode(&buffer[..take]);
                if !text.is_empty() && sender.send(EngineEvent::Data { kind, text }).await.is_err()
                {
                    // The consumer went away: stop reading.
                    return;
                }
            }
            if take < read {
                marked = true;
                let mut tail = decoder.flush();
                tail.push_str(constants::OUTPUT_TRUNCATION_MARKER);
                if sender
                    .send(EngineEvent::Data { kind, text: tail })
                    .await
                    .is_err()
                {
                    return;
                }
            }
        }
        let tail = decoder.flush();
        if !marked && !tail.is_empty() {
            let _ = sender.send(EngineEvent::Data { kind, text: tail }).await;
        }
    }

    /// Incremental UTF-8 decoder: a character split across reads is reassembled and
    /// invalid sequences decode to `U+FFFD` (Python `errors="replace"`).
    #[derive(Debug, Default, Clone)]
    struct Utf8Decoder {
        /// Bytes of an incomplete trailing sequence carried to the next call.
        pending: Vec<u8>,
    }

    impl Utf8Decoder {
        /// Decodes as much of `bytes` as possible, keeping a partial trailing sequence.
        fn decode(&mut self, bytes: &[u8]) -> String {
            let mut combined = std::mem::take(&mut self.pending);
            combined.extend_from_slice(bytes);
            let mut text = String::with_capacity(combined.len());
            let mut rest = combined.as_slice();
            loop {
                match std::str::from_utf8(rest) {
                    Ok(valid) => {
                        text.push_str(valid);
                        break;
                    }
                    Err(error) => {
                        let valid_up_to = error.valid_up_to();
                        if let Ok(valid) = std::str::from_utf8(&rest[..valid_up_to]) {
                            text.push_str(valid);
                        }
                        let Some(length) = error.error_len() else {
                            self.pending = rest[valid_up_to..].to_vec();
                            break;
                        };
                        text.push('\u{FFFD}');
                        rest = &rest[valid_up_to + length..];
                    }
                }
            }
            text
        }

        /// Decodes the carried partial sequence, replacing undecodable bytes.
        fn flush(&mut self) -> String {
            let pending = std::mem::take(&mut self.pending);
            if pending.is_empty() {
                return String::new();
            }
            String::from_utf8_lossy(&pending).into_owned()
        }
    }

    /// A future that completes when `timeout_secs` elapses (`0` = never).
    fn deadline_future(timeout_secs: u64) -> EngineFuture {
        if timeout_secs == 0 {
            return Box::pin(std::future::pending());
        }
        let duration = Duration::from_secs(timeout_secs);
        Box::pin(async move { tokio::time::sleep(duration).await })
    }

    /// A future that completes when any of `tokens` is cancelled (`pending` when empty).
    fn cancellation_future(tokens: &[CancellationToken]) -> EngineFuture {
        let futures: Vec<EngineFuture> = tokens
            .iter()
            .map(|token| {
                let token = token.clone();
                Box::pin(async move { token.cancelled_owned().await }) as EngineFuture
            })
            .collect();
        if futures.is_empty() {
            return Box::pin(std::future::pending());
        }
        Box::pin(async move {
            let _ = futures::future::select_all(futures).await;
        })
    }

    /// Reads `pipe` to EOF, retaining at most `max_output_bytes` bytes.
    async fn read_bounded<R>(pipe: Option<R>, max_output_bytes: u64) -> CapturedText
    where
        R: AsyncRead + Unpin,
    {
        let Some(mut pipe) = pipe else {
            return CapturedText::default();
        };
        let limit = usize::try_from(max_output_bytes).unwrap_or(usize::MAX);
        let mut buffer = vec![0_u8; READ_CHUNK_SIZE];
        let mut kept: Vec<u8> = Vec::new();
        let mut truncated = false;
        loop {
            match pipe.read(&mut buffer).await {
                // EOF and I/O errors both end the capture; whatever was read is kept.
                Ok(0) | Err(_) => break,
                Ok(read) => {
                    let take = limit.saturating_sub(kept.len()).min(read);
                    kept.extend_from_slice(&buffer[..take]);
                    truncated |= take < read;
                }
            }
        }
        let mut text = String::from_utf8_lossy(&kept).into_owned();
        if truncated {
            text.push_str(constants::OUTPUT_TRUNCATION_MARKER);
        }
        CapturedText { text, truncated }
    }

    /// Awaits a pipe reader, giving up after [`READER_GRACE`].
    async fn join_reader(mut task: tokio::task::JoinHandle<CapturedText>) -> CapturedText {
        if let Ok(Ok(captured)) = tokio::time::timeout(READER_GRACE, &mut task).await {
            return captured;
        }
        task.abort();
        CapturedText::default()
    }

    /// Wall-clock milliseconds since `started`, saturating at `u64::MAX`.
    fn elapsed_ms(started: Instant) -> u64 {
        u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    /// The exit code of `status`, or `-1` when the child died from a signal.
    fn exit_code_of(status: std::io::Result<ExitStatus>) -> i32 {
        status.ok().and_then(|status| status.code()).unwrap_or(-1)
    }

    #[cfg(test)]
    #[allow(clippy::unwrap_used, clippy::expect_used)]
    mod tests {
        use super::*;

        #[test]
        fn incremental_decoder_handles_split_characters() {
            let text = "hello 中中 end";
            let bytes = text.as_bytes();
            for split in 0..=bytes.len() {
                let mut decoder = Utf8Decoder::default();
                let mut out = decoder.decode(&bytes[..split]);
                out.push_str(&decoder.decode(&bytes[split..]));
                out.push_str(&decoder.flush());
                assert_eq!(out, text, "split at byte {split}");
            }
            let mut decoder = Utf8Decoder::default();
            assert_eq!(decoder.decode(b""), "");
            assert_eq!(decoder.flush(), "");
        }

        #[test]
        fn decoder_replaces_invalid_and_incomplete_bytes() {
            let mut decoder = Utf8Decoder::default();
            assert_eq!(decoder.decode(&[0xFF, 0x41]), "\u{FFFD}A");
            let mut decoder = Utf8Decoder::default();
            // An incomplete three-byte sequence is carried, then flushed lossily.
            assert_eq!(decoder.decode(&[0xE4, 0xB8]), "");
            assert_eq!(decoder.flush(), "\u{FFFD}");
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::shell::types::StreamKind;
    use futures::StreamExt;

    /// `true` when a POSIX `sh` exists; the tests that need one skip otherwise.
    fn has_sh() -> bool {
        which::which("sh").is_ok()
    }

    /// An unrestricted backend rooted at `dir` (Python's `LocalWorkConfig()` defaults).
    fn new_backend(dir: &tempfile::TempDir) -> LocalShellOperation {
        LocalShellOperation::new(LocalWorkConfig::unrestricted(format_path(dir.path()))).unwrap()
    }

    /// A restricted backend whose only sandbox root is `dir`.
    fn restricted(dir: &tempfile::TempDir) -> LocalShellOperation {
        let config = LocalWorkConfig::default()
            .with_workspace(format_path(dir.path()))
            .with_sandbox_roots([format_path(dir.path())]);
        LocalShellOperation::new(config).unwrap()
    }

    #[tokio::test]
    async fn empty_command_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let result = new_backend(&dir)
            .execute_cmd(ShellExecParams::new("  "))
            .await;
        assert_eq!(result.code, 199_004);
        assert!(
            result.message.contains("command can not be empty"),
            "{}",
            result.message
        );
        assert_eq!(result.data.unwrap().exit_code, -1);
    }

    #[tokio::test]
    async fn allowlist_rejects_unlisted_command() {
        let dir = tempfile::tempdir().unwrap();
        let config = LocalWorkConfig::unrestricted(format_path(dir.path()));
        let backend = LocalShellOperation::new(config.with_shell_allowlist(["echo"])).unwrap();
        let result = backend.execute_cmd(ShellExecParams::new("uname -a")).await;
        assert_eq!(result.code, 199_004);
        assert!(result.message.contains("not allowed"), "{}", result.message);
    }

    #[tokio::test]
    async fn dangerous_pattern_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = LocalWorkConfig::unrestricted(format_path(dir.path()));
        config.dangerous_patterns = Some(vec![r"rm\s+-rf".to_string()]);
        let backend = LocalShellOperation::new(config).unwrap();
        let result = backend
            .execute_cmd(ShellExecParams::new("rm -rf /tmp/x"))
            .await;
        assert_eq!(result.code, 199_004);
        assert!(
            result.message.contains("command rejected for safety"),
            "{}",
            result.message
        );
        assert!(result.message.contains(r"rm\s+-rf"), "{}", result.message);
    }

    #[tokio::test]
    async fn captures_stdout_stderr_and_exit_code() {
        if !has_sh() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let params = ShellExecParams::new("printf out; printf err >&2; exit 3");
        let result = new_backend(&dir).execute_cmd(params).await;
        assert_eq!(result.code, 0, "{}", result.message);
        let output = result.data.unwrap();
        assert_eq!((output.exit_code, output.stdout.as_str()), (3, "out"));
        assert_eq!(output.stderr, "err");
        assert!(!output.truncated && !output.timed_out && !output.cancelled);
        assert_eq!(
            std::fs::canonicalize(&output.cwd).unwrap(),
            std::fs::canonicalize(dir.path()).unwrap()
        );
    }

    #[tokio::test]
    async fn output_is_bounded_and_marked() {
        if !has_sh() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let mut config = LocalWorkConfig::unrestricted(format_path(dir.path()));
        config.max_output_bytes = 8;
        let backend = LocalShellOperation::new(config).unwrap();
        let output = backend
            .execute_cmd(ShellExecParams::new("printf abcdefghijklmnop"))
            .await
            .data
            .unwrap();
        assert!(output.truncated);
        assert!(output.stdout.starts_with("abcdefgh"), "{}", output.stdout);
        assert!(output.stdout.ends_with(constants::OUTPUT_TRUNCATION_MARKER));
    }

    #[tokio::test]
    async fn timeout_kills_child_and_keeps_partial_output() {
        if !has_sh() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let params = ShellExecParams::new("printf before; sleep 5").with_timeout_secs(1);
        let result = new_backend(&dir).execute_cmd(params).await;
        assert_eq!(result.code, 199_004);
        assert!(result.message.contains("timeout"), "{}", result.message);
        let output = result.data.unwrap();
        assert!(output.timed_out);
        assert_eq!(output.exit_code, -1);
        assert!(output.stdout.starts_with("before"), "{}", output.stdout);
    }

    #[tokio::test]
    async fn environment_overlay_and_scrub() {
        if !has_sh() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let params = || {
            let mut params = ShellExecParams::new("printf %s \"$FOO\"");
            params
                .environment
                .insert("FOO".to_string(), Some("bar".to_string()));
            params
        };
        let output = new_backend(&dir).execute_cmd(params()).await.data.unwrap();
        assert_eq!(output.stdout, "bar");
        let mut config = LocalWorkConfig::unrestricted(format_path(dir.path()));
        config.env_policy.scrub = vec!["FOO".to_string()];
        let scrubbed = LocalShellOperation::new(config).unwrap();
        assert_eq!(
            scrubbed.execute_cmd(params()).await.data.unwrap().stdout,
            ""
        );
    }

    #[tokio::test]
    async fn cwd_resolves_against_workspace() {
        if !has_sh() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        let expected = std::fs::canonicalize(dir.path().join("sub")).unwrap();
        let params = ShellExecParams::new("pwd").with_cwd("sub");
        let output = new_backend(&dir).execute_cmd(params).await.data.unwrap();
        assert_eq!(Path::new(output.stdout.trim()), expected);
    }

    #[tokio::test]
    async fn restricted_policy_rejects_outside_cwd() {
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let params = ShellExecParams::new("pwd").with_cwd(format_path(outside.path()));
        let result = restricted(&dir).execute_cmd(params).await;
        assert_eq!(result.code, 199_004);
        assert!(
            result.message.contains("Access denied"),
            "{}",
            result.message
        );
    }

    #[tokio::test]
    async fn restricted_policy_rejects_absolute_path_literal() {
        let dir = tempfile::tempdir().unwrap();
        let result = restricted(&dir)
            .execute_cmd(ShellExecParams::new("cat /etc/hostname"))
            .await;
        assert_eq!(result.code, 199_004);
        assert!(
            result.message.contains("outside sandbox"),
            "{}",
            result.message
        );
    }

    #[tokio::test]
    async fn execute_argv_never_uses_a_shell() {
        if which::which("echo").is_err() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let backend = new_backend(&dir);
        let output = backend
            .execute_argv(ShellArgvParams::new(["echo", "$HOME"]))
            .await;
        assert_eq!(output.code, 0, "{}", output.message);
        assert_eq!(output.data.unwrap().stdout.trim(), "$HOME");
        let empty = backend.execute_argv(ShellArgvParams::default()).await;
        assert_eq!(empty.code, 199_004);
        assert!(
            empty.message.contains("argv must not be empty"),
            "{}",
            empty.message
        );
        assert_eq!(empty.data.unwrap().exit_code, -1);
    }

    #[tokio::test]
    async fn background_reports_early_exit() {
        if !has_sh() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let params = ShellExecParams::new("printf boom >&2; exit 7");
        let result = new_backend(&dir).execute_cmd_background(params).await;
        assert_eq!(result.code, 199_004);
        assert!(
            result.message.contains("background command failed"),
            "{}",
            result.message
        );
        let output = result.data.unwrap();
        assert_eq!(
            output.error.as_deref(),
            Some("process exited early with code 7: boom")
        );
        assert!(output.pid.is_some());
    }

    #[tokio::test]
    async fn background_returns_pid_for_running_command() {
        if !has_sh() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let result = new_backend(&dir)
            .execute_cmd_background(ShellExecParams::new("sleep 2"))
            .await;
        assert_eq!(result.code, 0, "{}", result.message);
        let output = result.data.unwrap();
        assert!(output.pid.is_some() && output.error.is_none());
    }

    #[tokio::test]
    async fn stream_yields_chunks_and_terminal_chunk() {
        if !has_sh() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let backend = Arc::new(new_backend(&dir));
        let params = ShellExecParams::new("echo one; echo two");
        let items: Vec<_> = backend.execute_cmd_stream(params).collect().await;
        assert!(items.iter().all(|item| item.code == 0));
        let stdout: String = items
            .iter()
            .filter_map(|item| item.data.as_ref())
            .filter(|data| data.stream == Some(StreamKind::Stdout))
            .map(|data| data.text.as_str())
            .collect();
        assert!(stdout.contains("one") && stdout.contains("two"), "{stdout}");
        let last = items.last().unwrap().data.as_ref().unwrap();
        assert_eq!(last.exit_code, Some(0));
        assert_eq!(last.chunk_index, u64::try_from(items.len() - 1).unwrap());
    }

    #[tokio::test]
    async fn stream_empty_command_yields_single_error_chunk() {
        let dir = tempfile::tempdir().unwrap();
        let backend = Arc::new(new_backend(&dir));
        let items: Vec<_> = backend
            .execute_cmd_stream(ShellExecParams::new(" "))
            .collect()
            .await;
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].code, 199_004);
        assert!(
            items[0].message.contains("command can not be empty"),
            "{}",
            items[0].message
        );
        let data = items[0].data.as_ref().unwrap();
        assert_eq!((data.chunk_index, data.exit_code), (0, Some(-1)));
    }

    #[tokio::test]
    async fn session_cancellation_kills_running_command() {
        if !has_sh() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let backend = Arc::new(new_backend(&dir));
        let params = ShellExecParams::new("sleep 5").with_session("cancel-test");
        let started = Instant::now();
        let handle = tokio::spawn(async move { backend.execute_cmd(params).await });
        let registry = ShellProcessRegistry::global();
        for _ in 0..200 {
            if registry.is_registered("cancel-test") {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(registry.cancel_session("cancel-test"), 1);
        let result = handle.await.unwrap();
        assert_eq!(result.code, 199_004);
        assert!(result.message.contains("cancelled"), "{}", result.message);
        assert!(result.data.unwrap().cancelled);
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "{:?}",
            started.elapsed()
        );
    }

    #[tokio::test]
    async fn instance_cancellation_token_is_honoured() {
        if !has_sh() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let token = CancellationToken::new();
        let config = LocalWorkConfig::unrestricted(format_path(dir.path()));
        let backend = LocalShellOperation::new(config)
            .unwrap()
            .with_cancellation(token.clone());
        assert!(backend.config().shell_allowlist.is_none() && !backend.policy().is_restricted());
        let started = Instant::now();
        let handle =
            tokio::spawn(async move { backend.execute_cmd(ShellExecParams::new("sleep 5")).await });
        tokio::time::sleep(Duration::from_millis(150)).await;
        token.cancel();
        assert!(handle.await.unwrap().data.unwrap().cancelled);
        assert!(started.elapsed() < Duration::from_secs(3));
    }
}
