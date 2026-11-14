//! Process-level suite of the binary's own `run`/`chat` additions.
//!
//! The harness owns `run` and `chat`; this suite pins only the two flags the
//! binary adds around them:
//!
//! 1. **`run --worktree`** — the workspace's repository gets a worktree at
//!    `<workspace>/.worktrees/<slug>`, the announcement goes to stderr, and the
//!    prompt runs through the real loopback model server with the worktree as
//!    its workspace.
//! 2. **No repository** — `--worktree` outside a git repository is a typed
//!    `config` error (exit `1`, empty stdout) before any model is built.
//! 3. **`chat --watch`** — the watcher starts on stderr, the REPL sees EOF on a
//!    closed stdin and exits `0`, and stdout carries no ANSI escapes.
//!
//! The git cases skip when `git --version` cannot run, so the suite stays
//! offline and portable.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::path::Path;
use std::process::Output;

use openjiuwen_test_support::{MockLlmServer, ScriptedResponse, TEST_API_KEY};
use support::{assert_exit, assert_no_ansi, assert_success, parse_json, stderr, Sandbox};

/// The answer the scripted model returns for every request.
const ANSWER: &str = "hello from the worktree";

/// Starts a mock model server whose reusable fallback answers [`ANSWER`].
async fn scripted_server() -> MockLlmServer {
    let server = MockLlmServer::start().await;
    server.set_fallback(ScriptedResponse::text(ANSWER));
    server
}

/// Builds the model-connection and workspace flags of one invocation.
fn connection_args(sandbox: &Sandbox, server: &MockLlmServer) -> Vec<String> {
    vec![
        "--provider".to_owned(),
        "openai".to_owned(),
        "--api-base".to_owned(),
        server.base_url().to_owned(),
        "--api-key".to_owned(),
        TEST_API_KEY.to_owned(),
        "-m".to_owned(),
        "mock-model".to_owned(),
        "-w".to_owned(),
        sandbox.workspace().display().to_string(),
    ]
}

/// Appends `tail` to `args`.
fn with(mut args: Vec<String>, tail: &[&str]) -> Vec<String> {
    args.extend(tail.iter().map(|arg| (*arg).to_owned()));
    args
}

/// Whether a working `git` executable is on `PATH`.
fn git_available() -> bool {
    std::process::Command::new("git")
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success())
}

/// Runs one git command in `cwd`, panicking with git's stderr on failure.
fn git(cwd: &Path, args: &[&str]) -> Output {
    let output = std::process::Command::new("git")
        .args(args)
        .current_dir(cwd)
        .output()
        .unwrap_or_else(|err| panic!("running `git {args:?}`: {err}"));
    assert!(
        output.status.success(),
        "`git {args:?}` failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

/// Turns the sandbox workspace into a git repository with one commit.
///
/// The identity is passed per command with `-c`, so the suite never touches
/// (or depends on) a developer's global git configuration.
fn init_repository(workspace: &Path) {
    git(workspace, &["init", "-q"]);
    git(
        workspace,
        &[
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "user.name=Test",
            "commit",
            "--allow-empty",
            "-q",
            "-m",
            "init",
        ],
    );
}

#[tokio::test]
async fn run_worktree_creates_the_worktree_and_runs_the_prompt() {
    if !git_available() {
        eprintln!("skipping: git is not available");
        return;
    }
    let sandbox = Sandbox::new();
    init_repository(&sandbox.workspace());
    let server = scripted_server().await;

    let args = with(
        connection_args(&sandbox, &server),
        &["run", "--worktree", "demo", "hello", "-f", "json"],
    );
    let output = sandbox.run(&args).await;
    let text = assert_success(&output);
    assert_no_ansi(&text);

    let value = parse_json(&text);
    assert_eq!(value["answer"], ANSWER);
    assert_eq!(value["result_type"], "answer");
    assert!(
        value["session_id"]
            .as_str()
            .is_some_and(|id| id.starts_with("cli-")),
        "the result carries the session id: {value}"
    );

    let err = stderr(&output);
    let announcement = err
        .lines()
        .find(|line| line.starts_with("worktree: demo -> "))
        .unwrap_or_else(|| panic!("the worktree is announced on stderr: {err:?}"));
    assert!(
        announcement.contains("(branch worktree-demo)"),
        "the announcement names the branch: {announcement:?}"
    );

    let worktree = sandbox.workspace().join(".worktrees").join("demo");
    assert!(
        worktree.is_dir(),
        "the worktree lives at {}",
        worktree.display()
    );
    assert!(
        server.request_count() >= 1,
        "the run reached the mock model server"
    );
    server.shutdown().await;
}

#[tokio::test]
async fn run_worktree_outside_a_repository_is_a_config_error() {
    if !git_available() {
        eprintln!("skipping: git is not available");
        return;
    }
    let sandbox = Sandbox::new();
    let args = with(
        sandbox.workspace_flag().to_vec(),
        &["run", "--worktree", "demo", "hello"],
    );
    let output = sandbox.run(&args).await;

    let stdout = assert_exit(&output, 1);
    assert!(
        stdout.is_empty(),
        "a rejected worktree writes no stdout: {stdout:?}"
    );
    let err = stderr(&output);
    assert!(
        err.contains("error: config:"),
        "a non-repository is a config error: {err:?}"
    );
    assert!(
        err.contains("not inside a git repository"),
        "the diagnostic names the repository: {err:?}"
    );
    assert!(
        !sandbox.workspace().join(".worktrees").exists(),
        "nothing is created when the repository cannot be found"
    );
}

#[tokio::test]
async fn chat_watch_starts_the_watcher_and_exits_on_eof() {
    let sandbox = Sandbox::new();
    sandbox.write("workspace/OPENJIUWEN.md", "# workspace memory\n");
    let server = scripted_server().await;

    let args = with(connection_args(&sandbox, &server), &["chat", "--watch"]);
    let output = sandbox.run(&args).await;
    let text = assert_success(&output);
    assert_no_ansi(&text);

    let err = stderr(&output);
    assert!(
        err.contains("watching ") && err.contains("target(s)"),
        "the watcher announces itself on stderr: {err:?}"
    );
    server.shutdown().await;
}
