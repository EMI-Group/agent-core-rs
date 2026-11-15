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
#[cfg(unix)]
use crate::shell::pty;
use crate::shell::registry::{Registration, ShellProcessRegistry};
use crate::shell::types::{
    ShellArgvParams, ShellBackgroundOutput, ShellChunk, ShellExecParams, ShellOutput,
};
use crate::status::{self, SysOpStatus};
use crate::util::format_path;
use crate::workspace::{self, PathPolicy};

// The shared child-process engine, re-exported so `crate::shell::local::process` keeps
// resolving for the code backend.
pub(crate) use crate::shell::process;

/// Window used to detect a backgrounded command that exits immediately.
///
/// Python's `grace` defaults to 3 s; reporting a pid once the child survived this short
/// window keeps `execute_cmd_background` fast while still catching launch failures.
const EARLY_EXIT_GRACE: Duration = Duration::from_millis(150);
/// Poll interval used while waiting out [`EARLY_EXIT_GRACE`].
const EARLY_EXIT_POLL: Duration = Duration::from_millis(10);

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

    /// Whether a spawn with `wrapping` may prepend the line-buffering wrapper.
    ///
    /// Both the call site (never for [`Wrapping::Plain`]) and the configuration
    /// ([`LocalWorkConfig::line_buffered`]) must agree.
    fn line_buffered(&self, wrapping: Wrapping) -> bool {
        wrapping.is_line_buffered() && self.config.line_buffered
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
        let cwd_display = format_path(&path);
        match tokio::fs::metadata(&path).await {
            Ok(metadata) if metadata.is_dir() => {}
            Ok(_) => {
                let reason = format!("working directory does not exist: {cwd_display}");
                return Err(Rejection::with_cwd(reason, cwd_display));
            }
            Err(error) => {
                // Python's `Path.is_dir()` is also `False` here, so the message stays the
                // same; the underlying reason is logged so a permission problem on the
                // workspace root is diagnosable.
                tracing::debug!(
                    path = %cwd_display,
                    %error,
                    "the working directory could not be inspected; reporting it as missing"
                );
                let reason = format!("working directory does not exist: {cwd_display}");
                return Err(Rejection::with_cwd(reason, cwd_display));
            }
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
        wrapping: Wrapping,
        session_id: Option<&str>,
        timeout_secs: u64,
    ) -> OpResult<ShellOutput> {
        let cwd = format_path(&prepared.cwd);
        let line_buffered = self.line_buffered(wrapping);
        let child = match process::spawn_child(prepared.command(line_buffered)) {
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

    /// Spawns a streaming run and returns the engine's event stream.
    ///
    /// The PTY path is taken only when [`LocalWorkConfig::pty_streaming`] is set and the
    /// host can provide a terminal; a failed allocation falls back to pipes without
    /// failing the command, and non-Unix hosts always use pipes. Both paths share the
    /// engine, so bounded capture, timeout, and cancellation behave identically.
    fn stream_engine(
        &self,
        prepared: &Prepared,
        wrapping: Wrapping,
        options: process::RunOptions,
    ) -> std::result::Result<futures::stream::BoxStream<'static, process::EngineEvent>, String>
    {
        let argv = prepared.argv(self.line_buffered(wrapping));
        #[cfg(unix)]
        if self.config.pty_streaming {
            if let Some(pty_child) = pty::try_spawn(prepared.command_for(&argv)) {
                return Ok(pty::stream_pty_child(
                    pty_child.child,
                    pty_child.master,
                    options,
                ));
            }
            tracing::debug!("pseudo-terminal unavailable; streaming through pipes");
        }
        let child = process::spawn_child(prepared.command_for(&argv))
            .map_err(|error| format!("failed to start command: {error}"))?;
        Ok(process::stream_child(child, options))
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
        self.run_prepared(
            "execute_cmd",
            &params.command,
            prepared,
            Wrapping::LineBuffered,
            session,
            timeout,
        )
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
            let (tokens, _registration) = self.cancellation_tokens(params.session_id.as_deref());
            let max_output_bytes = self.config.max_output_bytes;
            let options = process::RunOptions { timeout_secs: timeout, max_output_bytes, cancellation: tokens };
            let mut events = match self.stream_engine(&prepared, Wrapping::LineBuffered, options) {
                Ok(events) => events,
                Err(reason) => {
                    yield stream_failure("execute_cmd_stream", &reason, 0);
                    return;
                }
            };
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
        let mut child =
            match spawn_background(&prepared, self.line_buffered(Wrapping::LineBuffered)) {
                Ok(child) => child,
                Err(error) => {
                    tracing::warn!(%error, "failed to start background command");
                    let reason = format!("background command failed: {error}");
                    return background_failure(&command, &cwd, None, Some(&reason), &reason);
                }
            };
        let pid = child.id();
        // A background child has no run loop of its own, so the registry kills its process
        // group when the session is cancelled (Python's `killpg` teardown).
        let registration = match (params.session_id.as_deref(), pid) {
            (Some(session), Some(group)) if !session.trim().is_empty() => {
                Some(ShellProcessRegistry::global().register_group(session, group))
            }
            _ => None,
        };
        let sink = Arc::new(Mutex::new(process::StderrSink::default()));
        let limit = constants::DEFAULT_STDERR_RETENTION_BYTES;
        let stderr_task = child
            .stderr
            .take()
            .map(|pipe| tokio::spawn(process::drain_stderr(pipe, limit, Arc::clone(&sink))));
        let exit = match wait_grace(&mut child).await {
            Ok(exit) => exit,
            Err(error) => {
                if let Some(group) = pid {
                    process::kill_group(group);
                }
                // Best-effort teardown of a background run that could not be started
                // properly: the failure of the teardown itself is reported, not dropped.
                if let Err(kill) = child.start_kill() {
                    tracing::warn!(pid, %kill, "failed to kill the background child");
                }
                let reason = format!("background command failed: {error}");
                return background_failure(&command, &cwd, pid, Some(&reason), &reason);
            }
        };
        match exit {
            Some(status) if !status.success() => {
                let early = process::background_early_exit(pid, status, stderr_task, &sink).await;
                let reason = format!("background command failed: {early}");
                return background_failure(&command, &cwd, pid, Some(&early), &reason);
            }
            Some(_) => {}
            // Keep draining stderr and reap the child so no zombie is left behind.
            None => {
                tokio::spawn(async move {
                    if let Err(error) = child.wait().await {
                        // The child stays unreaped, so the OS keeps its zombie until the
                        // process exits: that must not be silent.
                        tracing::warn!(pid, %error, "failed to reap the backgrounded child");
                    }
                    // Deregistering only after the reap keeps the group addressable for
                    // as long as the child could still be killed.
                    drop(registration);
                    if let Some(task) = stderr_task {
                        if let Err(error) = task.await {
                            tracing::debug!(
                                pid,
                                %error,
                                "the background stderr reader task failed"
                            );
                        }
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
        self.run_prepared(
            "execute_argv",
            &command,
            prepared,
            Wrapping::Plain,
            session,
            timeout,
        )
        .await
    }
}

/// Whether a spawn may prepend the line-buffering wrapper.
///
/// `execute_cmd`, `execute_cmd_stream`, and `execute_cmd_background` use
/// [`Wrapping::LineBuffered`] (Python's `stdbuf` wrapper); `execute_argv` must not, because
/// it never involves a shell and its arguments are data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Wrapping {
    /// Wrap the plan in the line-buffering wrapper when the host provides one.
    LineBuffered,
    /// Launch the plan exactly as resolved.
    Plain,
}

impl Wrapping {
    /// Whether the line-buffering wrapper is allowed for this spawn.
    fn is_line_buffered(self) -> bool {
        self == Self::LineBuffered
    }
}

/// A command that passed the policy gates, with interpreter, cwd, and environment resolved.
struct Prepared {
    program: String,
    args: Vec<String>,
    cwd: PathBuf,
    environment: Vec<(String, String)>,
}

impl Prepared {
    /// The argv to launch, wrapped for line buffering when `line_buffered` is set.
    fn argv(&self, line_buffered: bool) -> Vec<String> {
        let mut argv = Vec::with_capacity(self.args.len() + 1);
        argv.push(self.program.clone());
        argv.extend_from_slice(&self.args);
        process::line_buffered_argv(argv, line_buffered)
    }

    /// A configured (not yet spawned) command, wrapped for line buffering on request.
    fn command(&self, line_buffered: bool) -> Command {
        self.command_for(&self.argv(line_buffered))
    }

    /// A configured (not yet spawned) command for an already-resolved `argv`.
    fn command_for(&self, argv: &[String]) -> Command {
        let program = argv.first().unwrap_or(&self.program);
        let mut command = Command::new(program);
        command
            .args(argv.get(1..).unwrap_or_default())
            .current_dir(&self.cwd);
        process::apply_environment(&mut command, &self.environment);
        command
    }
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
fn spawn_background(prepared: &Prepared, line_buffered: bool) -> std::io::Result<Child> {
    let mut command = prepared.command(line_buffered);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(false);
    #[cfg(unix)]
    command.process_group(0);
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
