//! Offline integration suite for the process command seam.
//!
//! Python parity: the `asyncio.create_subprocess_exec` helpers in
//! `openjiuwen/rsi/harness_rsi/auto_harness/infra/{git_operations,
//! worktree_manager}.py` (v0.1.20) capture `stdout` with
//! `stderr=STDOUT`, decode lossily, and `rstrip()` the result. These tests pin
//! the Rust contract: exit codes are data, a timeout is an outcome (never an
//! error), a missing executable is a configuration error, and cancellation
//! kills the child.
//!
//! Everything runs offline: the real-runner cases use `sh`/`sleep` from the
//! host (gated to Unix) or a deliberately absent program name; the scripted
//! cases use [`FakeCommandRunner`].
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::{Duration, Instant};

use openjiuwen_common::error::ErrorKind;
use openjiuwen_rsi::runtime::command::{
    decode_output, CommandOutcome, CommandRunner, CommandSpec, FakeCommandRunner, FakeReply,
    ProcessCommandRunner,
};
use tokio_util::sync::CancellationToken;

#[cfg(unix)]
#[tokio::test]
async fn process_runner_captures_success() {
    let runner = ProcessCommandRunner::new();
    let spec = CommandSpec::new("sh").arg("-c").arg("printf 'hello\\n'");
    let outcome = runner.run(spec).await.unwrap();
    assert_eq!(outcome.code, Some(0));
    assert_eq!(outcome.stdout, "hello");
    assert_eq!(outcome.stderr, "");
    assert!(outcome.success());
    assert!(!outcome.timed_out);
    assert!(!outcome.cancelled);
    assert!(outcome.duration > Duration::ZERO);
    assert_eq!(runner.name(), "process");
    assert_eq!(runner.default_timeout(), None);
}

#[cfg(unix)]
#[tokio::test]
async fn non_zero_exit_is_data_not_an_error() {
    let outcome = ProcessCommandRunner::new()
        .run(
            CommandSpec::new("sh")
                .arg("-c")
                .arg("printf boom >&2; exit 3"),
        )
        .await
        .unwrap();
    assert_eq!(outcome.code, Some(3));
    assert_eq!(outcome.stdout, "");
    assert_eq!(outcome.stderr, "boom");
    assert!(!outcome.success());
    assert_eq!(outcome.combined_output(), "boom");
    let error = outcome.require_success("probe").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Tool);
    assert!(error.to_string().contains("exit code 3"));
}

#[cfg(unix)]
#[tokio::test]
async fn merge_stderr_folds_streams_into_stdout() {
    let outcome = ProcessCommandRunner::new()
        .run(
            CommandSpec::new("sh")
                .arg("-c")
                .arg("printf out; printf err >&2")
                .merge_stderr(),
        )
        .await
        .unwrap();
    assert_eq!(outcome.stdout, "outerr");
    assert_eq!(outcome.stderr, "");
    assert_eq!(outcome.combined_output(), "outerr");
    assert!(outcome.success());
}

#[cfg(unix)]
#[tokio::test]
async fn stdin_is_written_to_the_child() {
    let outcome = ProcessCommandRunner::new()
        .run(CommandSpec::new("sh").arg("-c").arg("cat").stdin("hello"))
        .await
        .unwrap();
    assert_eq!(outcome.stdout, "hello");
    assert!(outcome.success());
}

#[cfg(unix)]
#[tokio::test]
async fn env_and_cwd_apply_to_the_child() {
    let dir = tempfile::tempdir().unwrap();
    let outcome = ProcessCommandRunner::new()
        .run(
            CommandSpec::new("sh")
                .arg("-c")
                .arg("printf '%s@%s' \"$GREETING\" \"$(basename \"$PWD\")\"")
                .env("GREETING", "bonjour")
                .cwd(dir.path()),
        )
        .await
        .unwrap();
    assert!(outcome.stdout.starts_with("bonjour@"), "{}", outcome.stdout);
    assert!(outcome.success());
}

#[tokio::test]
async fn missing_program_is_a_configuration_error() {
    let error = ProcessCommandRunner::new()
        .run(CommandSpec::new(
            "openjiuwen-rsi-definitely-missing-program",
        ))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(error.to_string().contains("command not found"), "{}", error);
    assert!(error
        .to_string()
        .contains("openjiuwen-rsi-definitely-missing-program"));
}

#[cfg(unix)]
#[tokio::test]
async fn timeout_kills_the_child_and_is_reported_as_data() {
    let started = Instant::now();
    let outcome = ProcessCommandRunner::new()
        .run(
            CommandSpec::new("sh")
                .arg("-c")
                .arg("sleep 5")
                .timeout(Duration::from_millis(100)),
        )
        .await
        .unwrap();
    assert!(outcome.timed_out);
    assert!(!outcome.success());
    assert_eq!(outcome.code, None);
    assert!(
        started.elapsed() < Duration::from_secs(4),
        "the child was not killed promptly: {:?}",
        started.elapsed()
    );
}

#[cfg(unix)]
#[tokio::test]
async fn runner_default_timeout_applies_when_the_spec_has_none() {
    let started = Instant::now();
    let runner = ProcessCommandRunner::new().with_default_timeout(Duration::from_millis(100));
    assert_eq!(runner.default_timeout(), Some(Duration::from_millis(100)));
    let outcome = runner
        .run(CommandSpec::new("sh").arg("-c").arg("sleep 5"))
        .await
        .unwrap();
    assert!(outcome.timed_out);
    assert!(started.elapsed() < Duration::from_secs(4));
}

#[cfg(unix)]
#[tokio::test]
async fn pre_cancelled_token_stops_before_spawning() {
    let token = CancellationToken::new();
    token.cancel();
    let error = ProcessCommandRunner::new()
        .run_cancellable(CommandSpec::new("sh").arg("-c").arg("sleep 5"), &token)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Cancelled);
    assert!(error.is_cancelled());
}

#[cfg(unix)]
#[tokio::test]
async fn in_flight_cancellation_kills_the_child() {
    let runner = ProcessCommandRunner::new();
    let token = CancellationToken::new();
    let canceller = token.clone();
    let canceller_task = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(50)).await;
        canceller.cancel();
    });
    let started = Instant::now();
    let error = runner
        .run_cancellable(CommandSpec::new("sh").arg("-c").arg("sleep 5"), &token)
        .await
        .unwrap_err();
    canceller_task.await.unwrap();
    assert_eq!(error.kind(), ErrorKind::Cancelled);
    assert!(started.elapsed() < Duration::from_secs(4));
}

#[tokio::test]
async fn fake_runner_matches_program_and_argument_fragments() {
    let runner = FakeCommandRunner::new()
        .reply(
            FakeReply::outcome("*", CommandOutcome::success_with("any-status"))
                .with_args(["status", "--porcelain"]),
        )
        .reply(
            FakeReply::outcome("git", CommandOutcome::success_with("pushed")).with_args(["push"]),
        )
        .fallback(CommandOutcome::failure_with(1, "fallback"));

    let status = runner
        .run(CommandSpec::new("git").args(["status", "--porcelain", "--untracked-files=all"]))
        .await
        .unwrap();
    assert_eq!(status.stdout, "any-status");

    let pushed = runner
        .run(CommandSpec::new("git").arg("push"))
        .await
        .unwrap();
    assert_eq!(pushed.stdout, "pushed");

    let fallback = runner
        .run(CommandSpec::new("git").arg("fetch"))
        .await
        .unwrap();
    assert_eq!(fallback.code, Some(1));

    assert_eq!(runner.call_count(), 3);
    assert_eq!(
        runner.command_lines(),
        vec![
            "git status --porcelain --untracked-files=all",
            "git push",
            "git fetch",
        ]
    );
    assert_eq!(runner.name(), "fake");
}

#[tokio::test]
async fn fake_runner_prefers_the_first_matching_reply() {
    let runner = FakeCommandRunner::new()
        .reply(
            FakeReply::outcome("git", CommandOutcome::success_with("first")).with_args(["fetch"]),
        )
        .reply(
            FakeReply::outcome("git", CommandOutcome::success_with("second")).with_args(["fetch"]),
        );
    let outcome = runner
        .run(CommandSpec::new("git").arg("fetch"))
        .await
        .unwrap();
    assert_eq!(outcome.stdout, "first");
}

#[tokio::test]
async fn fake_runner_records_specs_and_errors_when_unmatched() {
    let runner = FakeCommandRunner::new();
    let spec = CommandSpec::new("git")
        .args(["commit", "-m", "hello world"])
        .cwd("/repo")
        .env("FOO", "bar");
    let error = runner.run(spec.clone()).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Tool);
    assert!(error.to_string().contains("no scripted reply"), "{}", error);
    assert!(error.to_string().contains("'hello world'"));

    let calls = runner.calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0], spec);
}

#[test]
fn command_line_quotes_only_when_needed() {
    let simple = CommandSpec::new("git").args(["diff", "--stat"]);
    assert_eq!(simple.command_line(), "git diff --stat");

    let spaced = CommandSpec::new("git")
        .args(["commit", "-m", "hello world"])
        .arg("it's");
    assert_eq!(
        spaced.command_line(),
        "git commit -m 'hello world' 'it'\\''s'"
    );

    assert_eq!(CommandSpec::new("program").command_line(), "program");
}

#[test]
fn with_default_timeout_only_fills_an_unset_timeout() {
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

#[test]
fn decode_strips_a_single_newline_and_is_lossy() {
    assert_eq!(decode_output(b"hello\n"), "hello");
    assert_eq!(decode_output(b"hello\r\n"), "hello");
    assert_eq!(decode_output(b"a\n\n"), "a\n");
    assert_eq!(decode_output(b"no-newline"), "no-newline");
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
fn outcome_helpers_and_serde_round_trip() {
    let outcome = CommandOutcome {
        code: Some(4),
        stdout: "out".to_owned(),
        stderr: "err".to_owned(),
        timed_out: false,
        cancelled: false,
        duration: Duration::from_millis(12),
    };
    let json = serde_json::to_string(&outcome).unwrap();
    let restored: CommandOutcome = serde_json::from_str(&json).unwrap();
    assert_eq!(restored, outcome);

    assert_eq!(CommandOutcome::success_with("x").code, Some(0));
    assert!(!CommandOutcome::failure_with(2, "x").success());
    assert!(CommandOutcome::timed_out_after(Duration::from_secs(1)).timed_out);
    assert!(CommandOutcome::cancelled_after(Duration::from_secs(1)).cancelled);
    // stdout is empty, so the error text is what combined_output exposes.
    let stderr_only = CommandOutcome {
        stderr: "only-error".to_owned(),
        ..CommandOutcome::success_with("")
    };
    assert_eq!(stderr_only.combined_output(), "only-error");
}
