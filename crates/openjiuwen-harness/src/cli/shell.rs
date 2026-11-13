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
//! Failure mapping: spawn and wait failures carry no domain meaning, so they are
//! wrapped with [`Error::external`], which keeps the original IO error reachable
//! as the [`std::error::Error::source`]. A command that runs but exits non-zero is
//! not an error here — it is data in [`ShellOutput::code`].

use std::path::Path;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};

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
}

impl ShellOutput {
    /// Returns `true` when the command exited with code 0.
    ///
    /// A signal termination (`code == None`) is never a success.
    #[must_use]
    pub fn success(&self) -> bool {
        self.code == Some(0)
    }

    /// Returns the failure status line, or `None` when the command succeeded.
    ///
    /// The line is `Exit code: {n}` for a normal exit and `Terminated by signal`
    /// when the process did not report a code.
    #[must_use]
    pub fn status_line(&self) -> Option<String> {
        if self.success() {
            return None;
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
    /// Runs `command` with `cwd` as its working directory.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the shell cannot be spawned or its status
    /// cannot be collected. A non-zero exit status is reported in
    /// [`ShellOutput::code`], not as an error.
    async fn run(&self, command: &str, cwd: &Path) -> Result<ShellOutput>;
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
/// No timeout is applied, matching Python's `create_subprocess_shell`; the command
/// runs until it exits or the REPL is killed. The child is killed when the future
/// is dropped mid-run (`kill_on_drop`), so a cancelled turn does not leak it.
#[derive(Debug, Clone)]
pub struct ProcessShellRunner {
    program: String,
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
        }
    }

    /// Returns the shell program this runner spawns.
    #[must_use]
    pub fn program(&self) -> &str {
        &self.program
    }
}

#[async_trait]
impl ShellRunner for ProcessShellRunner {
    async fn run(&self, command: &str, cwd: &Path) -> Result<ShellOutput> {
        let output = tokio::process::Command::new(&self.program)
            .arg(shell_command_flag())
            .arg(command)
            .current_dir(cwd)
            .kill_on_drop(true)
            .output()
            .await
            .map_err(Error::external)?;
        Ok(ShellOutput {
            command: command.to_string(),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            code: output.status.code(),
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;

    use openjiuwen_common::error::ErrorKind;

    use super::*;

    /// A runner that returns a canned output; proves the trait is object-safe and
    /// lets the REPL suites run without spawning a process.
    struct FakeRunner {
        output: ShellOutput,
    }

    #[async_trait]
    impl ShellRunner for FakeRunner {
        async fn run(&self, _command: &str, _cwd: &Path) -> Result<ShellOutput> {
            Ok(self.output.clone())
        }
    }

    fn output(stdout: &str, stderr: &str, code: Option<i32>) -> ShellOutput {
        ShellOutput {
            command: "cmd".to_string(),
            stdout: stdout.to_string(),
            stderr: stderr.to_string(),
            code,
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
    async fn a_missing_shell_program_is_a_typed_error() {
        let temp = tempfile::tempdir().unwrap();
        let runner = ProcessShellRunner::with_program("no-such-shell-for-tests");
        assert_eq!(runner.program(), "no-such-shell-for-tests");
        let error = runner.run("printf hello", temp.path()).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::External);
    }
}
