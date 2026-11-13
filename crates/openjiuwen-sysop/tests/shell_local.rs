//! Ports `tests/unit_tests/core/sys_operation/local/test_shell_operation.py`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::StreamExt;
use openjiuwen_sysop::config::{EnvPolicy, LocalWorkConfig};
use openjiuwen_sysop::constants;
use openjiuwen_sysop::operation::ShellOperation;
use openjiuwen_sysop::shell::local_shell;
use openjiuwen_sysop::shell::types::{ShellArgvParams, ShellExecParams, StreamKind};

/// The Python status code of every shell failure.
const SHELL_ERROR: i32 = 199_004;

/// Returns `true` when `sh` is available; otherwise prints a skip notice.
fn require_sh() -> bool {
    if which::which("sh").is_ok() {
        return true;
    }
    eprintln!("skipping shell test: `sh` is not on PATH");
    false
}

/// Returns `true` when `name` resolves on `PATH`; otherwise prints a skip notice.
fn require_tool(name: &str) -> bool {
    if which::which(name).is_ok() {
        return true;
    }
    eprintln!("skipping shell test: `{name}` is not on PATH");
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

#[tokio::test]
async fn empty_command_is_rejected() {
    let dir = workspace();
    let op = local_shell(config(&dir)).unwrap();
    for command in ["", "   "] {
        let result = op.execute_cmd(ShellExecParams::new(command)).await;
        assert_eq!(result.code, SHELL_ERROR, "command {command:?}");
        assert!(
            result.message.contains("command can not be empty"),
            "message for {command:?}: {}",
            result.message
        );
    }
}

#[tokio::test]
async fn allowlist_accepts_an_allowed_command() {
    if !require_sh() {
        return;
    }
    let dir = workspace();
    let op = local_shell(config(&dir).with_shell_allowlist(["echo"])).unwrap();
    let result = op.execute_cmd(ShellExecParams::new("echo hello")).await;
    assert_eq!(result.code, 0, "{}", result.message);
    let data = result.data.expect("success result carries output");
    assert_eq!(data.stdout, "hello\n");
    assert_eq!(data.stderr, "");
    assert_eq!(data.exit_code, 0);
}

#[tokio::test]
async fn allowlist_rejects_an_unlisted_command() {
    let dir = workspace();
    let op = local_shell(config(&dir).with_shell_allowlist(["echo"])).unwrap();
    let result = op.execute_cmd(ShellExecParams::new("ls -la")).await;
    assert_eq!(result.code, SHELL_ERROR, "{}", result.message);
    assert!(
        result.message.contains("command not allowed by allowlist"),
        "{}",
        result.message
    );
}

#[tokio::test]
async fn deny_list_rejects_a_dangerous_command() {
    let dir = workspace();
    let op = local_shell(config(&dir).with_default_dangerous_patterns()).unwrap();
    let result = op
        .execute_cmd(ShellExecParams::new("rm -rf /nonexistent-path"))
        .await;
    assert_eq!(result.code, SHELL_ERROR, "{}", result.message);
    assert!(
        result.message.contains("command rejected for safety"),
        "{}",
        result.message
    );
    // Rejected before spawning: the failure payload never ran a child.
    assert_eq!(
        result.data.expect("failure keeps output").exit_code,
        -1,
        "a rejected command must not have run"
    );
}

#[tokio::test]
async fn non_zero_exit_is_a_success_result() {
    if !require_sh() {
        return;
    }
    let dir = workspace();
    let op = local_shell(config(&dir)).unwrap();
    let result = op.execute_cmd(ShellExecParams::new("exit 7")).await;
    // Python returns `BaseResult(code=0)`; the exit status lives in the payload.
    assert_eq!(result.code, 0, "{}", result.message);
    let data = result.data.expect("success result carries output");
    assert_eq!(data.exit_code, 7);
    assert!(!data.timed_out && !data.cancelled);
}

#[tokio::test]
async fn stdout_and_stderr_are_captured_separately() {
    if !require_sh() {
        return;
    }
    let dir = workspace();
    let op = local_shell(config(&dir)).unwrap();
    let result = op
        .execute_cmd(ShellExecParams::new("echo out; echo err >&2"))
        .await;
    assert_eq!(result.code, 0, "{}", result.message);
    let data = result.data.expect("success result carries output");
    assert_eq!(data.stdout, "out\n");
    assert_eq!(data.stderr, "err\n");
}

#[tokio::test]
async fn output_capture_is_truncated() {
    if !require_sh() {
        return;
    }
    let dir = workspace();
    let mut work = config(&dir);
    work.max_output_bytes = 8;
    let op = local_shell(work).unwrap();
    let result = op
        .execute_cmd(ShellExecParams::new("printf 'abcdefghijklmnop'"))
        .await;
    assert_eq!(result.code, 0, "{}", result.message);
    let data = result.data.expect("success result carries output");
    assert!(data.truncated, "stdout={:?}", data.stdout);
    assert!(
        data.stdout.contains(constants::OUTPUT_TRUNCATION_MARKER),
        "stdout={:?}",
        data.stdout
    );
    assert!(
        data.stdout.starts_with("abcdefgh"),
        "stdout={:?}",
        data.stdout
    );
    assert_eq!(data.exit_code, 0);
}

#[tokio::test]
async fn timeout_kills_the_process_and_returns_199004() {
    if !require_sh() {
        return;
    }
    let dir = workspace();
    let op = local_shell(config(&dir)).unwrap();
    let result = op
        .execute_cmd(ShellExecParams::new("sleep 5").with_timeout_secs(1))
        .await;
    assert_eq!(result.code, SHELL_ERROR, "{}", result.message);
    assert!(result.message.contains("timeout"), "{}", result.message);
    // A timeout keeps the partial output in the failure payload (documented deviation
    // from Python, which returns no data for `cmd` failures).
    let data = result.data.expect("timeout failure keeps partial output");
    assert!(data.timed_out, "timed_out flag not set");
    assert!(!data.cancelled);
    assert_eq!(data.exit_code, -1, "a killed child has no exit status");
    assert_eq!(data.command, "sleep 5");
}

#[tokio::test]
async fn timeout_is_bounded() {
    if !require_sh() {
        return;
    }
    let dir = workspace();
    let op = local_shell(config(&dir)).unwrap();
    let started = Instant::now();
    let result = op
        .execute_cmd(ShellExecParams::new("sleep 5").with_timeout_secs(1))
        .await;
    let elapsed = started.elapsed();
    assert_eq!(result.code, SHELL_ERROR, "{}", result.message);
    assert!(
        elapsed < Duration::from_secs(4),
        "timeout took {elapsed:?}, expected about 1 s"
    );
}

#[tokio::test]
async fn cwd_resolves_against_the_workspace_root() {
    if !require_sh() {
        return;
    }
    let dir = workspace();
    std::fs::create_dir(dir.path().join("sub")).unwrap();
    let op = local_shell(config(&dir)).unwrap();
    let result = op
        .execute_cmd(ShellExecParams::new("pwd").with_cwd("sub"))
        .await;
    assert_eq!(result.code, 0, "{}", result.message);
    let data = result.data.expect("success result carries output");
    assert_eq!(data.exit_code, 0);
    let printed = data.stdout.trim();
    assert!(printed.ends_with("sub"), "pwd printed {printed:?}");
    assert_eq!(data.cwd, printed, "reported cwd must match the child cwd");
}

#[tokio::test]
async fn environment_overlay_is_applied() {
    if !require_sh() {
        return;
    }
    let dir = workspace();
    let op = local_shell(config(&dir)).unwrap();
    let mut params = ShellExecParams::new("printenv GREETING");
    params
        .environment
        .insert("GREETING".to_string(), Some("hello".to_string()));
    let result = op.execute_cmd(params).await;
    assert_eq!(result.code, 0, "{}", result.message);
    let data = result.data.expect("success result carries output");
    assert_eq!(data.stdout, "hello\n");
    assert_eq!(data.exit_code, 0);
}

#[tokio::test]
async fn environment_policy_scrub_wins_over_an_overlay() {
    if !require_sh() || !require_tool("printenv") {
        return;
    }
    let dir = workspace();
    let work = LocalWorkConfig {
        env_policy: EnvPolicy {
            scrub: vec!["GREETING".to_string()],
            ..EnvPolicy::minimal()
        },
        ..config(&dir)
    };
    let op = local_shell(work).unwrap();
    let mut params = ShellExecParams::new("printenv GREETING");
    params
        .environment
        .insert("GREETING".to_string(), Some("hello".to_string()));
    let result = op.execute_cmd(params).await;
    // `printenv` exits 1 for a missing variable; that is still a success envelope.
    assert_eq!(result.code, 0, "{}", result.message);
    let data = result.data.expect("success result carries output");
    assert_eq!(
        data.stdout, "",
        "a scrubbed name must not be re-added by an explicit value"
    );
    assert_eq!(data.exit_code, 1);
}

#[tokio::test]
async fn execute_argv_runs_without_a_shell() {
    if !require_tool("echo") {
        return;
    }
    let dir = workspace();
    let op = local_shell(config(&dir)).unwrap();
    let result = op
        .execute_argv(ShellArgvParams::new(["echo", "a b", "$HOME"]))
        .await;
    assert_eq!(result.code, 0, "{}", result.message);
    let data = result.data.expect("success result carries output");
    // No word splitting and no variable expansion: arguments are data.
    assert_eq!(data.stdout, "a b $HOME\n");
    assert_eq!(data.exit_code, 0);
}

#[tokio::test]
async fn execute_argv_allowlist_applies_to_the_program() {
    let dir = workspace();
    let op = local_shell(config(&dir).with_shell_allowlist(["echo"])).unwrap();
    let result = op
        .execute_argv(ShellArgvParams::new(["printf", "hi"]))
        .await;
    assert_eq!(result.code, SHELL_ERROR, "{}", result.message);
    assert!(
        result.message.contains("command not allowed by allowlist"),
        "{}",
        result.message
    );
}

#[tokio::test]
async fn stream_ends_with_a_terminal_chunk() {
    if !require_sh() {
        return;
    }
    let dir = workspace();
    let op: Arc<dyn ShellOperation> = local_shell(config(&dir)).unwrap();
    let chunks: Vec<_> = op
        .execute_cmd_stream(ShellExecParams::new("echo hi"))
        .collect()
        .await;
    assert!(
        !chunks.is_empty(),
        "a stream yields at least the terminal chunk"
    );
    assert!(
        chunks.iter().all(|chunk| chunk.code == 0),
        "{:?}",
        chunks
            .iter()
            .map(|chunk| &chunk.message)
            .collect::<Vec<_>>()
    );
    let terminal_index = chunks.len() - 1;
    let terminal = chunks[terminal_index]
        .data
        .as_ref()
        .expect("terminal chunk carries data");
    assert_eq!(
        terminal.stream, None,
        "the terminal chunk has no stream kind"
    );
    assert_eq!(terminal.exit_code, Some(0));
    assert_eq!(terminal.chunk_index, terminal_index as u64);

    let text: String = chunks[..terminal_index]
        .iter()
        .filter_map(|chunk| chunk.data.as_ref())
        .filter(|data| data.stream == Some(StreamKind::Stdout))
        .map(|data| data.text.as_str())
        .collect();
    assert!(text.contains("hi"), "streamed stdout was {text:?}");
}

#[tokio::test]
async fn background_returns_a_pid() {
    if !require_sh() {
        return;
    }
    let dir = workspace();
    let op = local_shell(config(&dir)).unwrap();
    let result = op
        .execute_cmd_background(ShellExecParams::new("sleep 1"))
        .await;
    assert_eq!(result.code, 0, "{}", result.message);
    let data = result.data.expect("success result carries output");
    assert!(data.pid.is_some(), "a started process reports its pid");
    assert_eq!(
        data.error, None,
        "no early-exit error for a running command"
    );
}

/// Environment probe: how many `stdbuf` wrapper variables the child inherited.
const WRAPPER_PROBE: &str = "env | grep -c -E '_STDBUF_|libstdbuf'";

/// `true` when a command run through `stdbuf` really exports the wrapper marker.
fn stdbuf_is_effective() -> bool {
    if which::which("stdbuf").is_err() {
        return false;
    }
    std::process::Command::new("stdbuf")
        .args(["-oL", "-eL", "sh", "-c", WRAPPER_PROBE])
        .output()
        .is_ok_and(|output| String::from_utf8_lossy(&output.stdout).trim() != "0")
}

#[tokio::test]
async fn line_buffering_wraps_by_default_and_can_be_disabled() {
    if !require_sh() {
        return;
    }
    let dir = workspace();
    let default_config = config(&dir);
    assert!(
        default_config.line_buffered,
        "line buffering defaults to on"
    );
    let wrapped = local_shell(default_config)
        .unwrap()
        .execute_cmd(ShellExecParams::new(WRAPPER_PROBE))
        .await;
    assert_eq!(wrapped.code, 0, "{}", wrapped.message);
    let wrapped_marker = wrapped.data.expect("captured run").stdout;

    let mut plain_config = config(&dir);
    plain_config.line_buffered = false;
    let plain = local_shell(plain_config)
        .unwrap()
        .execute_cmd(ShellExecParams::new(WRAPPER_PROBE))
        .await;
    assert_eq!(plain.code, 0, "{}", plain.message);
    assert_eq!(
        plain.data.expect("captured run").stdout.trim(),
        "0",
        "line_buffered = false must spawn the plan unwrapped"
    );

    if stdbuf_is_effective() && std::env::var_os("LD_PRELOAD").is_none() {
        assert_ne!(
            wrapped_marker.trim(),
            "0",
            "the default configuration should be wrapped in stdbuf"
        );
    }
}
