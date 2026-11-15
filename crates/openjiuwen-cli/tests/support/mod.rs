//! Shared process-boundary helpers for the `openjiuwen` binary test suites.
//!
//! Every suite in this package spawns the compiled binary as a real child
//! process, so the rules that keep such a test hermetic live here once instead
//! of being re-stated per case:
//!
//! - the child runs with a **temporary working directory** and
//!   `OPENJIUWEN_HOME` pointed at a **temporary home**, so `dotenvy` (which walks
//!   the working directory's ancestors for `.env`) and every CLI-owned file
//!   (`settings.json`, `permissions.yaml`, `mcp.json`, session records) stay
//!   inside the test;
//! - every ambient `OPENJIUWEN_*` / `OPENAI_*`, `API_KEY`, `API_BASE`,
//!   `MODEL_PROVIDER`, `MODEL_NAME`, `NO_COLOR` and `RUST_LOG` variable is
//!   removed from the child's environment, so a developer's shell cannot change
//!   a verdict;
//! - stdin is `/dev/null` (a `run` without a prompt sees EOF instead of hanging
//!   on the test harness's terminal);
//! - every invocation is bounded: the child is killed and the test fails with
//!   the command line when it does not finish within [`CHILD_WAIT`], so a
//!   regression can never hang `cargo test`;
//! - the child is waited for **asynchronously** on the caller's runtime, which
//!   matters for the scripted-`run` suite: its loopback mock model server is a
//!   task on that same runtime and would never be polled by a blocking wait.
//!
//! The binary is located through `env!("CARGO_BIN_EXE_openjiuwen")`, which Cargo
//! defines only for the tests of the package that owns the binary — this one.
#![allow(dead_code)] // each suite uses a different subset of these helpers.
#![allow(clippy::missing_panics_doc, clippy::must_use_candidate)]

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::Duration;

use openjiuwen_test_support::TempWorkspace;
use serde_json::Value;

/// Longest one child invocation may run before the suite fails.
///
/// A `run` against the loopback mock server takes well under a second; the bound
/// only has to be generous enough for a loaded CI machine and small enough that a
/// hang is reported instead of blocking `cargo test` forever.
pub(crate) const CHILD_WAIT: Duration = Duration::from_secs(120);

/// Ambient variables removed from every child, matched exactly.
///
/// `NO_COLOR` and `RUST_LOG` are included because they change the bytes the
/// suites assert on: a colour switch would add escape sequences to `run` output
/// and a log filter would add noise to stderr.
const AMBIENT_EXACT: &[&str] = &[
    "API_KEY",
    "API_BASE",
    "MODEL_PROVIDER",
    "MODEL_NAME",
    "NO_COLOR",
    "RUST_LOG",
];

/// Ambient variable prefixes removed from every child.
///
/// `OPENJIUWEN_*` is the CLI's own override namespace (home, workspace, model,
/// token budgets, log settings) and `OPENAI_*` is the generic SDK fallback.
const AMBIENT_PREFIXES: &[&str] = &["OPENJIUWEN_", "OPENAI_"];

/// Returns whether `name` is ambient state a child must not inherit.
fn is_ambient(name: &str) -> bool {
    AMBIENT_EXACT.contains(&name)
        || AMBIENT_PREFIXES
            .iter()
            .any(|prefix| name.starts_with(prefix))
}

/// One isolated environment for the `openjiuwen` binary: temp cwd, home and workspace.
///
/// The tree is a [`TempWorkspace`] root plus a `cwd/` directory:
///
/// ```text
/// <root>/
///   home/       OPENJIUWEN_HOME
///   workspace/  the `-w` workspace
///   logs/
///   cwd/        the child's working directory
/// ```
pub(crate) struct Sandbox {
    /// Owns the temporary root and removes it when the sandbox is dropped.
    temp: TempWorkspace,
}

impl Sandbox {
    /// Creates a fresh sandbox; panics only when the temp tree cannot be made.
    pub(crate) fn new() -> Self {
        let temp = TempWorkspace::new().expect("a temporary workspace root");
        temp.create_dir("cwd")
            .expect("a temporary working directory");
        Self { temp }
    }

    /// The `-w` workspace directory (`<root>/workspace`).
    pub(crate) fn workspace(&self) -> PathBuf {
        self.temp.workspace()
    }

    /// The runtime-state root handed to the child as `OPENJIUWEN_HOME`.
    pub(crate) fn home(&self) -> PathBuf {
        self.temp.home()
    }

    /// The child's working directory (`<root>/cwd`).
    pub(crate) fn cwd(&self) -> PathBuf {
        self.temp.path("cwd")
    }

    /// The workspace flag pair `["-w", "<workspace>"]`, ready to prepend.
    pub(crate) fn workspace_flag(&self) -> [String; 2] {
        ["-w".to_owned(), self.workspace().display().to_string()]
    }

    /// Resolves `rel` under the sandbox root without touching the filesystem.
    pub(crate) fn path(&self, rel: impl AsRef<Path>) -> PathBuf {
        self.temp.path(rel)
    }

    /// Resolves `rel` under `<workspace>/auto_harness` without touching the filesystem.
    pub(crate) fn auto_harness_path(&self, rel: impl AsRef<Path>) -> PathBuf {
        self.workspace().join("auto_harness").join(rel)
    }

    /// Writes `contents` to `rel` under the sandbox root, creating parents.
    pub(crate) fn write(&self, rel: impl AsRef<Path>, contents: impl AsRef<[u8]>) -> PathBuf {
        self.temp
            .write(&rel, contents)
            .unwrap_or_else(|err| panic!("writing test file {}: {err}", rel.as_ref().display()))
    }

    /// Writes `contents` to `rel` under `<workspace>/auto_harness`, creating parents.
    pub(crate) fn write_auto_harness(
        &self,
        rel: impl AsRef<Path>,
        contents: impl AsRef<[u8]>,
    ) -> PathBuf {
        let rel = rel.as_ref();
        let path = self.auto_harness_path(rel);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap_or_else(|err| {
                panic!("creating {}: {err}", parent.display());
            });
        }
        std::fs::write(&path, contents)
            .unwrap_or_else(|err| panic!("writing {}: {err}", path.display()));
        path
    }

    /// Reads `rel` under the sandbox root as UTF-8 text.
    pub(crate) fn read(&self, rel: impl AsRef<Path>) -> String {
        self.temp
            .read(&rel)
            .unwrap_or_else(|err| panic!("reading test file {}: {err}", rel.as_ref().display()))
    }

    /// Runs the binary on the caller's runtime, bounded by [`CHILD_WAIT`].
    ///
    /// Use this from `#[tokio::test]` suites that drive the loopback mock model
    /// server: the server is a task on the same runtime, so the wait must yield.
    pub(crate) async fn run(&self, args: &[impl AsRef<OsStr>]) -> Output {
        let mut command = tokio::process::Command::from(self.command(args));
        command
            .kill_on_drop(true)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let child = command
            .spawn()
            .unwrap_or_else(|err| panic!("spawning `openjiuwen {}`: {err}", joined(args)));
        match tokio::time::timeout(CHILD_WAIT, child.wait_with_output()).await {
            Ok(output) => output
                .unwrap_or_else(|err| panic!("waiting for `openjiuwen {}`: {err}", joined(args))),
            // Dropping the wait future above drops the child, and
            // `kill_on_drop` reaps it.
            Err(elapsed) => panic!(
                "`openjiuwen {}` did not finish within {CHILD_WAIT:?}: {elapsed}",
                joined(args)
            ),
        }
    }

    /// Runs the binary to completion on a throwaway runtime (plain `#[test]` suites).
    pub(crate) fn run_blocking(&self, args: &[impl AsRef<OsStr>]) -> Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("a throwaway tokio runtime")
            .block_on(self.run(args))
    }

    /// Runs the binary with extra `key=path` environment overrides, bounded by [`CHILD_WAIT`].
    ///
    /// The overrides sit on top of the isolated environment described in the
    /// module docs, so the ambient-variable scrubbing still applies. This is how
    /// a suite points `OPENJIUWEN_HOME` at a directory that does not exist (the
    /// empty-watch-plan case) without giving up hermeticity.
    pub(crate) async fn run_with_env(
        &self,
        args: &[impl AsRef<OsStr>],
        env: &[(&str, &Path)],
    ) -> Output {
        let mut command = tokio::process::Command::from(self.command(args));
        for (key, value) in env {
            command.env(key, value);
        }
        command
            .kill_on_drop(true)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let child = command
            .spawn()
            .unwrap_or_else(|err| panic!("spawning `openjiuwen {}`: {err}", joined(args)));
        match tokio::time::timeout(CHILD_WAIT, child.wait_with_output()).await {
            Ok(output) => output
                .unwrap_or_else(|err| panic!("waiting for `openjiuwen {}`: {err}", joined(args))),
            Err(elapsed) => panic!(
                "`openjiuwen {}` did not finish within {CHILD_WAIT:?}: {elapsed}",
                joined(args)
            ),
        }
    }

    /// Runs the binary with `input` piped on stdin, bounded by [`CHILD_WAIT`].
    ///
    /// [`Sandbox::run`] closes stdin, so a `chat` invocation driving the REPL
    /// would see EOF before its first turn. This variant keeps the same isolated
    /// cwd/home, scrubbed environment, `kill_on_drop` and piped stdout/stderr,
    /// but writes `input` to the child and then drops the handle, so the child
    /// reads exactly those lines and then end of input. The wait is bounded and
    /// reported exactly like [`Sandbox::run`].
    pub(crate) async fn run_with_stdin(&self, args: &[impl AsRef<OsStr>], input: &str) -> Output {
        use tokio::io::AsyncWriteExt;

        let mut command = tokio::process::Command::from(self.command(args));
        command
            .kill_on_drop(true)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = command
            .spawn()
            .unwrap_or_else(|err| panic!("spawning `openjiuwen {}`: {err}", joined(args)));
        let mut stdin = child.stdin.take();
        if let Some(stdin) = stdin.as_mut() {
            stdin
                .write_all(input.as_bytes())
                .await
                .unwrap_or_else(|err| {
                    panic!("writing stdin of `openjiuwen {}`: {err}", joined(args))
                });
        }
        // Closing the pipe is the child's end-of-input signal.
        drop(stdin);
        match tokio::time::timeout(CHILD_WAIT, child.wait_with_output()).await {
            Ok(output) => output
                .unwrap_or_else(|err| panic!("waiting for `openjiuwen {}`: {err}", joined(args))),
            Err(elapsed) => panic!(
                "`openjiuwen {}` did not finish within {CHILD_WAIT:?}: {elapsed}",
                joined(args)
            ),
        }
    }

    /// Spawns the binary with piped streams for a suite that drives it live.
    ///
    /// The child inherits the same isolated cwd/home, scrubbed environment and
    /// `/dev/null` stdin as [`Sandbox::run`], plus `kill_on_drop`, so a failed
    /// assertion cannot leak a long-running watcher into the rest of the test
    /// run. The caller owns stdout/stderr and must bound its reads itself (see
    /// `tests/watch.rs` for the readiness/exit pattern).
    pub(crate) fn spawn(&self, args: &[impl AsRef<OsStr>]) -> tokio::process::Child {
        let mut command = tokio::process::Command::from(self.command(args));
        command
            .kill_on_drop(true)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        command
            .spawn()
            .unwrap_or_else(|err| panic!("spawning `openjiuwen {}`: {err}", joined(args)))
    }

    /// Builds the child command: isolated cwd/home, ambient state removed, stdin closed.
    fn command(&self, args: &[impl AsRef<OsStr>]) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_openjiuwen"));
        for (name, _) in std::env::vars_os() {
            if name.to_str().is_some_and(is_ambient) {
                command.env_remove(&name);
            }
        }
        command
            .env("OPENJIUWEN_HOME", self.home())
            .current_dir(self.cwd())
            .stdin(Stdio::null())
            .args(args);
        command
    }
}

/// Renders a command line for a failure message.
fn joined(args: &[impl AsRef<OsStr>]) -> String {
    args.iter()
        .map(|arg| arg.as_ref().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Returns the child's exit code, panicking when it was killed by a signal.
pub(crate) fn exit_code(output: &Output) -> i32 {
    output
        .status
        .code()
        .expect("the child should exit normally, not be killed by a signal")
}

/// Renders a child's stdout as lossy UTF-8.
pub(crate) fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// Renders a child's stderr as lossy UTF-8.
pub(crate) fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// Renders both streams and the exit code for a failure message.
pub(crate) fn describe(output: &Output) -> String {
    format!(
        "exit {}, stdout {:?}, stderr {:?}",
        exit_code(output),
        stdout(output),
        stderr(output)
    )
}

/// Asserts exit `0` and returns stdout.
pub(crate) fn assert_success(output: &Output) -> String {
    assert_eq!(
        exit_code(output),
        0,
        "expected a successful exit: {}",
        describe(output)
    );
    stdout(output)
}

/// Asserts the exact `expected` exit code and returns stdout.
pub(crate) fn assert_exit(output: &Output, expected: i32) -> String {
    assert_eq!(
        exit_code(output),
        expected,
        "expected exit {expected}: {}",
        describe(output)
    );
    stdout(output)
}

/// Asserts `text` carries no ANSI escape sequence.
pub(crate) fn assert_no_ansi(text: &str) {
    assert!(
        !text.contains('\u{1b}'),
        "output must not contain ANSI escapes: {text:?}"
    );
}

/// Returns whether some line of `text` starts with `word` (a clap help row).
pub(crate) fn has_row(text: &str, word: &str) -> bool {
    text.lines()
        .any(|line| line.split_whitespace().next() == Some(word))
}

/// Parses `text` (trimmed) as one JSON document.
pub(crate) fn parse_json(text: &str) -> Value {
    serde_json::from_str(text.trim())
        .unwrap_or_else(|err| panic!("expected one JSON document ({err}): {text:?}"))
}

/// Returns the non-empty lines of `text`.
pub(crate) fn non_empty_lines(text: &str) -> Vec<&str> {
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .collect()
}
