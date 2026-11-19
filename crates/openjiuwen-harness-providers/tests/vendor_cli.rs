//! Subprocess-transport tests for the shared vendor CLI plumbing.
//!
//! Every case spawns a real POSIX shell script written by [`support::FakeCli`], so the tests cover
//! the actual pipe/flush/EOF/signal path the vendor harnesses depend on. They are offline,
//! deterministic and bounded: no network, no vendor CLI, and every await that could block is
//! wrapped in a [`BOUND`] timeout.

#![cfg(unix)]
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests assert by panicking

mod support;

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

use openjiuwen_harness_protocol::{HarnessErrorKind, HarnessResult, JsonValue};
use openjiuwen_harness_providers::vendor_cli::{
    resolve_binary, BinarySpec, ChildProcess, ChildProcessConfig,
};
use serde_json::json;
use support::FakeCli;

/// Bounded await budget for every child interaction.
const BOUND: Duration = Duration::from_secs(10);

/// Spawns `fake` with no arguments and no working directory.
async fn spawn(fake: &FakeCli, env: &BTreeMap<String, String>) -> ChildProcess {
    ChildProcess::spawn(ChildProcessConfig {
        program: fake.path(),
        args: &[],
        cwd: None,
        env,
        inherit_env: true,
        label: "Fake",
        max_line_bytes: 0,
    })
    .await
    .expect("the fake CLI must spawn")
}

/// Reads the next line with a bounded await.
async fn read(child: &mut ChildProcess) -> HarnessResult<Option<JsonValue>> {
    tokio::time::timeout(BOUND, child.read_json())
        .await
        .expect("the fake CLI must answer or exit")
}

#[tokio::test]
async fn ndjson_round_trips_and_the_transcript_records_the_request() {
    let fake = FakeCli::replies(&[r#"{"type":"reply","text":"pong"}"#]);
    let env = fake.env();
    let mut child = spawn(&fake, &env).await;
    assert!(child.pid().is_some());

    tokio::time::timeout(BOUND, child.write_json(&json!({"type": "ping", "id": 1})))
        .await
        .expect("the write must not block")
        .expect("the write must succeed");
    let reply = read(&mut child)
        .await
        .expect("valid JSON")
        .expect("a reply");
    assert_eq!(reply, json!({"type": "reply", "text": "pong"}));
    assert_eq!(fake.lines(), vec![r#"{"type":"ping","id":1}"#.to_owned()]);

    tokio::time::timeout(BOUND, child.close_stdin())
        .await
        .expect("closing stdin must not block");
    assert!(read(&mut child).await.expect("clean EOF").is_none());
}

#[tokio::test]
async fn replies_are_emitted_in_order() {
    let fake = FakeCli::replies(&[r#"{"n":1}"#, r#"{"n":2}"#]);
    let env = fake.env();
    let mut child = spawn(&fake, &env).await;
    for request in [json!({"go": 1}), json!({"go": 2})] {
        child.write_json(&request).await.expect("write the request");
        let reply = read(&mut child)
            .await
            .expect("valid JSON")
            .expect("a reply");
        assert_eq!(reply["n"], request["go"]);
    }
    assert_eq!(fake.lines().len(), 2);
    child.close_stdin().await;
    assert!(read(&mut child).await.expect("clean EOF").is_none());
}

#[tokio::test]
async fn read_json_returns_none_when_the_cli_exits() {
    let fake = FakeCli::script("exit 0");
    let env = fake.env();
    let mut child = spawn(&fake, &env).await;
    assert!(read(&mut child).await.expect("clean EOF").is_none());
    assert_eq!(child.wait().await, Some(0));
}

#[tokio::test]
async fn stderr_is_captured_in_the_tail() {
    let fake =
        FakeCli::script("echo 'fatal: bad flag' >&2\nwhile IFS= read -r line; do :; done\nexit 7");
    let env = fake.env();
    let mut child = spawn(&fake, &env).await;
    child.close_stdin().await;
    assert_eq!(child.wait().await, Some(7));
    let tail = child.stderr_tail();
    assert_eq!(tail, "fatal: bad flag");
}

#[tokio::test]
async fn a_broken_pipe_reports_the_stderr_tail() {
    let fake = FakeCli::script("echo 'gone: shutting down' >&2\nexit 3");
    let env = fake.env();
    let mut child = spawn(&fake, &env).await;
    assert_eq!(child.wait().await, Some(3));

    let error = tokio::time::timeout(BOUND, child.write_json(&json!({"late": true})))
        .await
        .expect("the write must not block")
        .expect_err("a dead CLI rejects the write");
    assert_eq!(error.kind(), HarnessErrorKind::Protocol);
    assert!(
        error
            .message()
            .starts_with("the Fake CLI exited before the message was written"),
        "unexpected message: {}",
        error.message()
    );
    assert!(error.message().contains("gone: shutting down"));
}

#[tokio::test]
async fn malformed_output_is_reported_with_the_offending_line() {
    let fake = FakeCli::replies(&["this is not json"]);
    let env = fake.env();
    let mut child = spawn(&fake, &env).await;
    child
        .write_json(&json!({"type": "ping"}))
        .await
        .expect("write");
    let error = read(&mut child).await.expect_err("invalid JSON");
    assert_eq!(error.kind(), HarnessErrorKind::Protocol);
    assert_eq!(error.message(), "invalid Fake CLI output: this is not json");
}

#[tokio::test]
async fn an_over_long_output_line_is_rejected() {
    let fake = FakeCli::script("printf 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\\n'");
    let env = fake.env();
    let mut child = ChildProcess::spawn(ChildProcessConfig {
        program: fake.path(),
        args: &[],
        cwd: None,
        env: &env,
        inherit_env: true,
        label: "Fake",
        max_line_bytes: 8,
    })
    .await
    .expect("the fake CLI must spawn");
    let error = read(&mut child)
        .await
        .expect_err("the line exceeds the ceiling");
    assert_eq!(error.message(), "a Fake CLI output line exceeded 8 bytes");
}

#[tokio::test]
async fn an_over_long_line_is_truncated_and_its_surplus_drained() {
    const CAP: usize = 1024 * 1024;
    // 20 MiB without a newline, then a newline and a well-formed line.
    let fake = FakeCli::script(
        "head -c 20971520 /dev/zero | tr '\\0' 'a'\nprintf '\\n'\necho '{\"ok\":true}'",
    );
    let env = fake.env();
    let mut child = ChildProcess::spawn(ChildProcessConfig {
        program: fake.path(),
        args: &[],
        cwd: None,
        env: &env,
        inherit_env: true,
        label: "Fake",
        max_line_bytes: CAP,
    })
    .await
    .expect("the fake CLI must spawn");

    let started = std::time::Instant::now();
    let error = read(&mut child)
        .await
        .expect_err("the line exceeds the ceiling");
    assert_eq!(error.kind(), HarnessErrorKind::Protocol);
    assert_eq!(
        error.message(),
        "a Fake CLI output line exceeded 1048576 bytes"
    );
    assert!(
        started.elapsed() < BOUND,
        "the read must stop retaining at the cap instead of buffering 20 MiB"
    );

    // The surplus was drained without being retained, so the next line is intact and the child
    // never blocked on a pipe the harness stopped reading.
    assert_eq!(
        read(&mut child).await.expect("valid JSON"),
        Some(json!({"ok": true}))
    );
    child.close_stdin().await;
    assert_eq!(child.wait().await, Some(0));
}

#[tokio::test]
async fn a_newline_free_line_is_capped_at_end_of_file() {
    let fake = FakeCli::script("head -c 1048576 /dev/zero | tr '\\0' 'a'");
    let env = fake.env();
    let mut child = ChildProcess::spawn(ChildProcessConfig {
        program: fake.path(),
        args: &[],
        cwd: None,
        env: &env,
        inherit_env: true,
        label: "Fake",
        max_line_bytes: 64 * 1024,
    })
    .await
    .expect("the fake CLI must spawn");

    let error = read(&mut child)
        .await
        .expect_err("the line exceeds the ceiling before end of file");
    assert_eq!(
        error.message(),
        "a Fake CLI output line exceeded 65536 bytes"
    );
    assert!(read(&mut child).await.expect("clean EOF").is_none());
    assert_eq!(child.wait().await, Some(0));
}

#[tokio::test]
async fn kill_reaps_the_process() {
    let fake = FakeCli::script("while :; do sleep 1; done");
    let env = fake.env();
    let mut child = spawn(&fake, &env).await;
    assert!(child.pid().is_some());

    let code = tokio::time::timeout(BOUND, child.kill())
        .await
        .expect("kill must not block");
    // A SIGKILLed process has no exit code; a shell that traps it may still report one.
    assert!(code != Some(0), "unexpected exit code: {code:?}");
    assert!(child.pid().is_none(), "the process must be reaped");
}

#[tokio::test]
async fn arguments_and_working_directory_reach_the_child() {
    let fake = FakeCli::script("printf '{\"args\":\"%s\",\"cwd\":\"%s\"}\\n' \"$*\" \"$(pwd)\"");
    let env = fake.env();
    let args = vec!["--flag".to_owned(), "value".to_owned()];
    let expected_cwd = fake
        .path()
        .parent()
        .expect("the script lives in the temp directory");
    let mut child = ChildProcess::spawn(ChildProcessConfig {
        program: fake.path(),
        args: &args,
        cwd: Some(expected_cwd),
        env: &env,
        inherit_env: true,
        label: "Fake",
        max_line_bytes: 0,
    })
    .await
    .expect("the fake CLI must spawn");
    let output = read(&mut child).await.expect("valid JSON").expect("a line");
    assert_eq!(output["args"], json!("--flag value"));
    let reported_cwd = output["cwd"].as_str().expect("a cwd string");
    assert_eq!(
        std::fs::canonicalize(reported_cwd).expect("the reported cwd exists"),
        std::fs::canonicalize(expected_cwd).expect("the temp directory exists")
    );
}

#[tokio::test]
async fn a_cleared_environment_keeps_the_supplied_variables() {
    let fake = FakeCli::replies(&[r#"{"type":"ready"}"#]);
    let env = fake.env();
    let mut child = ChildProcess::spawn(ChildProcessConfig {
        program: fake.path(),
        args: &[],
        cwd: None,
        env: &env,
        inherit_env: false,
        label: "Fake",
        max_line_bytes: 0,
    })
    .await
    .expect("the fake CLI must spawn");
    // The script records through `$FAKE_CLI_LOG`; a reply therefore proves the variable arrived.
    child
        .write_json(&json!({"type": "ping"}))
        .await
        .expect("write");
    assert_eq!(
        read(&mut child).await.expect("valid JSON"),
        Some(json!({"type": "ready"}))
    );
    assert_eq!(fake.lines(), vec![r#"{"type":"ping"}"#.to_owned()]);
    child.close_stdin().await;
}

#[tokio::test]
async fn the_fake_cli_resolves_through_an_explicit_path() {
    let fake = FakeCli::script("exit 0");
    let resolved = resolve_binary(&BinarySpec {
        name: "fake-cli",
        explicit: Some(fake.path()),
        label: "Fake",
        install_hint: "write the fake CLI",
    })
    .expect("the script is executable");
    assert_eq!(resolved, fake.path().to_path_buf());
    assert_eq!(
        resolve_binary(&BinarySpec {
            name: "fake-cli",
            explicit: Some(Path::new("/nonexistent/fake-cli")),
            label: "Fake",
            install_hint: "write the fake CLI",
        })
        .expect_err("a missing explicit path is rejected")
        .message(),
        "Fake CLI binary '/nonexistent/fake-cli' is not executable: write the fake CLI"
    );
}
