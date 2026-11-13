//! Process-group lifecycle tests for the local shell backend.
//!
//! Python parity: `_kill_process_tree` in `local/utils.py` and `os.killpg` in
//! `shell_process_registry.py`. A child spawned by the engine leads its own process group,
//! so a timeout, a cancellation, or a session teardown must kill the whole tree — the
//! direct child *and* the grandchildren a shell script left behind — before reaping it.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};

use openjiuwen_sysop::config::LocalWorkConfig;
use openjiuwen_sysop::operation::ShellOperation;
use openjiuwen_sysop::shell::registry::ShellProcessRegistry;
use openjiuwen_sysop::shell::types::{ShellArgvParams, ShellExecParams};
use openjiuwen_sysop::shell::{local_shell, LocalShellOperation};
use tokio_util::sync::CancellationToken;

/// The Python status code of a shell failure (timeout, cancellation).
const SHELL_ERROR: i32 = 199_004;
/// Upper bound on every wait in this file.
const BOUND: Duration = Duration::from_secs(5);
/// Poll interval while waiting for a pid file or a process to disappear.
const POLL: Duration = Duration::from_millis(20);
/// Environment probe: how many `stdbuf` wrapper variables the child inherited.
const WRAPPER_PROBE: &str = "env | grep -c -E '_STDBUF_|libstdbuf'";

/// Returns `true` when `sh` is available; otherwise prints a skip notice.
fn require_sh() -> bool {
    if which::which("sh").is_ok() {
        return true;
    }
    eprintln!("skipping process-group test: `sh` is not on PATH");
    false
}

/// A fresh temporary workspace.
fn workspace() -> tempfile::TempDir {
    tempfile::tempdir().unwrap()
}

/// The unrestricted local configuration rooted at `dir`.
fn config(dir: &tempfile::TempDir) -> LocalWorkConfig {
    LocalWorkConfig::unrestricted(dir.path().to_string_lossy().into_owned())
}

/// A command that leaves a `sleep` grandchild behind and records its pid in `pid_file`.
fn grandchild_command(pid_file: &Path) -> String {
    format!("sleep 30 & echo $! > '{}'; sleep 30", pid_file.display())
}

/// Waits up to [`BOUND`] for `path` to contain a pid.
async fn wait_for_pid(path: &Path) -> Option<u32> {
    let deadline = Instant::now() + BOUND;
    while Instant::now() < deadline {
        if let Ok(text) = std::fs::read_to_string(path) {
            if let Ok(pid) = text.trim().parse::<u32>() {
                return Some(pid);
            }
        }
        tokio::time::sleep(POLL).await;
    }
    None
}

/// `true` while `pid` is a live (non-zombie) process.
#[cfg(target_os = "linux")]
fn process_alive(pid: u32) -> bool {
    match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
        Ok(stat) => {
            // The state follows the command name, which may itself contain spaces.
            match stat
                .rsplit(')')
                .next()
                .and_then(|rest| rest.split_whitespace().next())
            {
                // `Z` is a dead process waiting to be reaped: it cannot run code anymore.
                Some(state) => state != "Z",
                None => false,
            }
        }
        Err(_) => false,
    }
}

/// `true` while `pid` accepts signal 0 (the non-Linux fallback).
#[cfg(not(target_os = "linux"))]
fn process_alive(pid: u32) -> bool {
    Command::new("sh")
        .args(["-c", &format!("kill -0 {pid} 2>/dev/null")])
        .status()
        .is_ok_and(|status| status.success())
}

/// Waits up to [`BOUND`] for `pid` to disappear.
async fn wait_until_gone(pid: u32) -> bool {
    let deadline = Instant::now() + BOUND;
    while Instant::now() < deadline {
        if !process_alive(pid) {
            return true;
        }
        tokio::time::sleep(POLL).await;
    }
    !process_alive(pid)
}

#[tokio::test]
async fn timeout_kills_the_whole_process_group() {
    if !require_sh() {
        return;
    }
    let dir = workspace();
    let pid_file = dir.path().join("grandchild.pid");
    let op = local_shell(config(&dir)).unwrap();
    let params = ShellExecParams::new(grandchild_command(&pid_file)).with_timeout_secs(1);
    let started = Instant::now();
    let result = op.execute_cmd(params).await;

    assert_eq!(result.code, SHELL_ERROR, "{}", result.message);
    assert!(result.message.contains("timeout"), "{}", result.message);
    let output = result.data.expect("a timed-out run keeps its output");
    assert!(output.timed_out);
    assert_eq!(output.exit_code, -1);
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "the run was not bounded: {:?}",
        started.elapsed()
    );

    let grandchild = wait_for_pid(&pid_file)
        .await
        .expect("the command recorded the grandchild pid");
    assert!(
        wait_until_gone(grandchild).await,
        "grandchild {grandchild} survived the process-group kill"
    );
}

#[tokio::test]
async fn cancellation_kills_the_whole_process_group() {
    if !require_sh() {
        return;
    }
    let dir = workspace();
    let pid_file = dir.path().join("cancel.pid");
    let token = CancellationToken::new();
    let backend = Arc::new(
        LocalShellOperation::new(config(&dir))
            .unwrap()
            .with_cancellation(token.clone()),
    );
    let params = ShellExecParams::new(grandchild_command(&pid_file));
    let started = Instant::now();
    let handle = tokio::spawn(async move { backend.execute_cmd(params).await });

    let grandchild = wait_for_pid(&pid_file)
        .await
        .expect("the command recorded the grandchild pid");
    token.cancel();
    let result = handle.await.unwrap();

    assert_eq!(result.code, SHELL_ERROR, "{}", result.message);
    assert!(result.message.contains("cancelled"), "{}", result.message);
    assert!(result.data.expect("cancelled run").cancelled);
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "{:?}",
        started.elapsed()
    );
    assert!(
        wait_until_gone(grandchild).await,
        "grandchild {grandchild} survived the cancellation kill"
    );
}

#[tokio::test]
async fn session_cancellation_kills_the_whole_process_group() {
    if !require_sh() {
        return;
    }
    let dir = workspace();
    let pid_file = dir.path().join("session.pid");
    let op = local_shell(config(&dir)).unwrap();
    let session = format!("group-cancel-{}", std::process::id());
    let params = ShellExecParams::new(grandchild_command(&pid_file)).with_session(session.clone());
    let handle = tokio::spawn(async move { op.execute_cmd(params).await });

    let grandchild = wait_for_pid(&pid_file)
        .await
        .expect("the command recorded the grandchild pid");
    let registry = ShellProcessRegistry::global();
    let deadline = Instant::now() + BOUND;
    while Instant::now() < deadline && !registry.is_registered(&session) {
        tokio::time::sleep(POLL).await;
    }
    assert_eq!(registry.cancel_session(&session), 1);
    let result = handle.await.unwrap();

    assert!(result.data.expect("cancelled run").cancelled);
    assert!(
        wait_until_gone(grandchild).await,
        "grandchild {grandchild} survived the session kill"
    );
}

#[tokio::test]
async fn background_command_is_registered_and_stoppable_as_a_group() {
    if !require_sh() {
        return;
    }
    let dir = workspace();
    let pid_file = dir.path().join("background.pid");
    let op = local_shell(config(&dir)).unwrap();
    let session = format!("group-background-{}", std::process::id());
    let params = ShellExecParams::new(grandchild_command(&pid_file)).with_session(session.clone());
    let result = op.execute_cmd_background(params).await;

    assert_eq!(result.code, 0, "{}", result.message);
    let background = result.data.expect("a started background command");
    let group = background
        .pid
        .expect("a started background command has a pid");
    let registry = ShellProcessRegistry::global();
    assert!(
        registry.is_registered(&session),
        "the background command must be registered for its session"
    );

    let grandchild = wait_for_pid(&pid_file)
        .await
        .expect("the command recorded the grandchild pid");
    assert_eq!(registry.cancel_session(&session), 1);
    assert!(
        wait_until_gone(group).await,
        "the background child {group} survived the group kill"
    );
    assert!(
        wait_until_gone(grandchild).await,
        "grandchild {grandchild} survived the background group kill"
    );
}

/// `true` when running a command through `stdbuf` really exports the wrapper marker.
fn stdbuf_is_effective() -> bool {
    if which::which("stdbuf").is_err() {
        return false;
    }
    Command::new("stdbuf")
        .args(["-oL", "-eL", "sh", "-c", WRAPPER_PROBE])
        .output()
        .is_ok_and(|output| String::from_utf8_lossy(&output.stdout).trim() != "0")
}

#[tokio::test]
async fn execute_argv_is_never_wrapped() {
    if !require_sh() {
        return;
    }
    let dir = workspace();
    let op = local_shell(config(&dir)).unwrap();
    let probe = ShellExecParams::new(WRAPPER_PROBE);

    // `execute_cmd` goes through the shell plan, so the wrapper applies to it.
    let command = op.execute_cmd(probe).await;
    assert_eq!(command.code, 0, "{}", command.message);
    let command_marker = command.data.expect("captured run").stdout;

    // `execute_argv` execs the program directly: no shell, no wrapper.
    let argv = op
        .execute_argv(ShellArgvParams::new(["sh", "-c", WRAPPER_PROBE]))
        .await;
    assert_eq!(argv.code, 0, "{}", argv.message);
    assert_eq!(
        argv.data.expect("captured run").stdout.trim(),
        "0",
        "execute_argv must not be wrapped in the line-buffering wrapper"
    );

    if stdbuf_is_effective() && std::env::var_os("LD_PRELOAD").is_none() {
        assert_ne!(
            command_marker.trim(),
            "0",
            "execute_cmd should be wrapped when stdbuf is available"
        );
    }
}
