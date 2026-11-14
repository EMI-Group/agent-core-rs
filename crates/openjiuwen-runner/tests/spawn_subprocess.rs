//! End-to-end suite for the subprocess spawn backend
//! ([`openjiuwen_runner::spawn::subprocess`]), driven over real stdio pipes.
//!
//! Hermetic and offline: the "child agent" is this very test binary re-executed
//! with the libtest filter of the `#[ignore]`d [`fake_child`] test, which speaks
//! the NDJSON protocol on stdin/stdout. Its behaviour is selected by the
//! `FAKE_CHILD_MODE` environment variable, so one binary covers echo, streaming,
//! crash, silence, malformed framing, oversized framing, health failure, and
//! shutdown acknowledgement without a second fixture or any new dependency.
//!
//! Ports the shape of Python `tests/unit_tests/core/runner/spawn/`
//! (`test_child_process.py`, `test_process_manager.py`): request/reply, streaming,
//! liveness, graceful-to-force escalation, and the kill-on-drop guarantee.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::{BufRead, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use openjiuwen_agent::output::{AgentOutput, ResultType};
use openjiuwen_common::error::{ErrorKind, Result};
use openjiuwen_runner::spawn::{
    AgentSpawner, SpawnSpec, SpawnedAgent, SubprocessConfig, SubprocessSpawner,
};
use serde_json::{json, Value};

/// Upper bound on any await in this suite, so a broken exchange fails the test
/// instead of stalling the harness.
const BOUND: Duration = Duration::from_secs(5);

/// How often a poll-based assertion re-probes its condition.
const POLL_INTERVAL: Duration = Duration::from_millis(20);

// --- fixtures ---------------------------------------------------------------

/// The libtest arguments that turn this test binary into one protocol child.
///
/// `--nocapture` is required: without it libtest redirects the test thread's
/// stdout into its capture buffer and the protocol frames never reach the pipe.
fn child_args() -> [&'static str; 5] {
    [
        "fake_child",
        "--exact",
        "--ignored",
        "--test-threads=1",
        "--nocapture",
    ]
}

/// The launch configuration for a child in `mode`.
///
/// The background health task is disabled (`Duration::ZERO` interval) so no
/// probe races the tests; explicit [`SpawnedAgent::health`] calls still work, and
/// they are bounded tightly so a silent child fails fast.
fn child_config(mode: &str) -> SubprocessConfig {
    let program = std::env::current_exe()
        .expect("the test binary path is available")
        .to_string_lossy()
        .into_owned();
    SubprocessConfig::new(program)
        .with_args(child_args())
        .with_env("FAKE_CHILD_MODE", mode)
        .with_request_timeout(BOUND)
        .with_health_check(Duration::ZERO, Duration::from_millis(300), 3)
}

/// A spawner that re-executes this test binary as the protocol child in `mode`.
fn spawner(mode: &str) -> SubprocessSpawner {
    SubprocessSpawner::new(child_config(mode))
}

/// The spawn spec every test hands over; the fake child ignores its contents.
fn spec() -> SpawnSpec {
    SpawnSpec::class_agent("tests.spawn_subprocess", "FakeAgent")
}

/// Sends one invocation under [`BOUND`], so a hung exchange fails the test.
///
/// Generic and `?Sized` so it serves both the concrete `SpawnedProcess` handle
/// and the `Arc<dyn SpawnedAgent>` the trait seam returns.
async fn send_bounded<A: SpawnedAgent + ?Sized>(agent: &A, inputs: Value) -> Result<AgentOutput> {
    tokio::time::timeout(BOUND, agent.send(inputs))
        .await
        .expect("send must not hang")
}

/// Polls `probe` every [`POLL_INTERVAL`] until it reports `true` or `deadline`
/// expires, returning the last observation.
async fn wait_until(deadline: Duration, mut probe: impl FnMut() -> bool) -> bool {
    let started = tokio::time::Instant::now();
    loop {
        if probe() {
            return true;
        }
        if started.elapsed() >= deadline {
            return false;
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

// --- the hermetic child -----------------------------------------------------

/// Writes one protocol frame (one NDJSON line) and flushes it.
fn emit_frame(out: &mut impl Write, counter: &mut u64, kind: &str, payload: &Value) {
    *counter = counter.saturating_add(1);
    let pid = std::process::id();
    let sequence = *counter;
    let frame = json!({
        "kind": kind,
        "payload": payload,
        "timestamp": 1_700_000_000.0,
        "message_id": format!("child-{pid}-{sequence}"),
    });
    let _ = writeln!(out, "{frame}");
    let _ = out.flush();
}

/// The text an `echo`/`echo_env` child replies with for `payload`.
fn child_output_text(mode: &str, payload: &Value) -> String {
    if mode == "echo_env" {
        // Both variables prove the merged environment reached the child: one is
        // set by the config and the spec (the spec must win), the other only by
        // the config.
        let shared = std::env::var("FAKE_SHARED").unwrap_or_default();
        let config_only = std::env::var("FAKE_CONFIG_ONLY").unwrap_or_default();
        return format!("{shared}/{config_only}");
    }
    payload["inputs"]["query"]
        .as_str()
        .map_or_else(|| payload["inputs"].to_string(), str::to_owned)
}

/// Replies to one invocation, according to `mode`.
fn child_handle_invoke(out: &mut impl Write, counter: &mut u64, mode: &str, payload: &Value) {
    match mode {
        // Never answers, so the parent's request deadline is what ends the
        // exchange.
        "silent" => {}
        "bad_frame" => {
            // Valid JSON that is not a `SpawnMessage`: the parent must reject it
            // rather than mistake it for the child's stdout noise.
            let line = json!({
                "kind": "teleport",
                "payload": null,
                "timestamp": 1.0,
                "message_id": "not-a-frame",
            });
            let _ = writeln!(out, "{line}");
            let _ = out.flush();
        }
        "oversize" => {
            // One line far longer than the small `max_frame_bytes` the oversize
            // test configures.
            let line = json!({
                "kind": "done",
                "payload": "x".repeat(2_048),
                "timestamp": 1.0,
                "message_id": "oversized",
            });
            let _ = writeln!(out, "{line}");
            let _ = out.flush();
        }
        "stream" => {
            for (index, text) in ["one", "two", "three"].iter().enumerate() {
                emit_frame(
                    out,
                    counter,
                    "stream_chunk",
                    &json!({"type": "content", "index": index, "payload": {"content": text}}),
                );
            }
            emit_frame(out, counter, "done", &Value::Null);
        }
        _ => {
            let text = child_output_text(mode, payload);
            emit_frame(
                out,
                counter,
                "output",
                &json!({"output": text, "result_type": "answer"}),
            );
        }
    }
}

/// Applies one frame the parent wrote to the child's stdin; returns `true` when
/// the child must exit after it.
fn child_handle_frame(
    out: &mut impl Write,
    counter: &mut u64,
    mode: &str,
    kind: &str,
    payload: &Value,
) -> bool {
    match kind {
        "health_check" => {
            if !matches!(mode, "silent" | "unresponsive_shutdown") {
                let healthy = mode != "health_fail";
                emit_frame(
                    out,
                    counter,
                    "health_check_response",
                    &json!({"healthy": healthy}),
                );
            }
            false
        }
        "shutdown" => {
            if matches!(mode, "silent" | "unresponsive_shutdown") {
                return false;
            }
            emit_frame(out, counter, "shutdown_ack", &Value::Null);
            true
        }
        "input" => {
            child_handle_invoke(out, counter, mode, payload);
            false
        }
        _ => false,
    }
}

/// The protocol child: an `#[ignore]`d test that this suite re-executes.
///
/// Modes (`FAKE_CHILD_MODE`): `echo`, `echo_env`, `stream`, `crash`, `silent`,
/// `unresponsive_shutdown`, `bad_frame`, `oversize`, `health_fail`,
/// `shutdown_ack`. Anything else behaves like `echo`.
#[test]
#[ignore = "spawned as a stdio protocol child by the subprocess tests"]
fn fake_child() {
    let mode = std::env::var("FAKE_CHILD_MODE").unwrap_or_default();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    // libtest prints `running 1 test` and then `test fake_child ... ` without a
    // trailing newline before this body runs; closing that line keeps every
    // protocol frame on a line of its own (the parent skips the noise line).
    let _ = writeln!(out);
    let _ = out.flush();

    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    // The bootstrap frame (`{"spawn_spec": ...}`) arrives before any invocation.
    if !matches!(lines.next(), Some(Ok(_))) {
        return;
    }
    if mode == "crash" {
        // Die without replying: the parent's `send` must fail, not hang.
        std::process::exit(1);
    }

    let mut counter = 0_u64;
    for line in lines {
        let Ok(line) = line else { break };
        let Ok(frame) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        let kind = frame["kind"].as_str().unwrap_or_default();
        let payload = frame.get("payload").cloned().unwrap_or(Value::Null);
        if child_handle_frame(&mut out, &mut counter, &mode, kind, &payload) {
            return;
        }
    }

    // The parent closed stdin. A mode that ignores `shutdown` must stay alive so
    // the parent's graceful deadline expires and escalates to a kill.
    if matches!(mode.as_str(), "silent" | "unresponsive_shutdown") {
        loop {
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

// --- request/reply ----------------------------------------------------------

#[tokio::test]
async fn spawn_and_send_round_trips_through_the_subprocess_backend() {
    let process = spawner("echo")
        .spawn_process(spec())
        .await
        .expect("the fake child launches");

    assert!(
        !process.id().is_empty(),
        "the handle needs a stable id: {:?}",
        process.id()
    );
    let pid = process.pid().expect("the OS reported a process id");
    assert_ne!(pid, 0, "the reported pid is nonsense");
    assert!(process.is_alive(), "a freshly spawned child is alive");

    let output = send_bounded(process.as_ref(), json!({"query": "hello-spawn"}))
        .await
        .expect("the echo child answers");

    assert_eq!(output.output, "hello-spawn");
    assert_eq!(output.result_type, ResultType::Answer);
    assert!(!output.is_interrupt());
    assert!(process.is_alive(), "a reply does not end the child");
}

#[tokio::test]
async fn the_agent_spawner_seam_returns_a_working_trait_object() {
    let spawned: Arc<dyn SpawnedAgent> = spawner("echo")
        .spawn(spec())
        .await
        .expect("the trait seam launches the child");

    assert_ne!(spawned.id(), "", "the trait object needs a stable id");
    assert!(
        spawned.pid().is_some(),
        "the trait object reports the child's pid"
    );
    assert!(spawned.is_alive());

    let output = send_bounded(spawned.as_ref(), json!({"query": "through-the-seam"}))
        .await
        .expect("the echo child answers through the seam");
    assert_eq!(output.output, "through-the-seam");
    assert_eq!(output.result_type, ResultType::Answer);
}

#[tokio::test]
async fn spec_env_overrides_win_over_the_config_env() {
    let spawner = SubprocessSpawner::new(
        child_config("echo_env")
            .with_env("FAKE_SHARED", "config")
            .with_env("FAKE_CONFIG_ONLY", "cfg"),
    );
    let process = spawner
        .spawn_process(spec().with_env("FAKE_SHARED", "spec"))
        .await
        .expect("the fake child launches");

    let output = send_bounded(process.as_ref(), json!({"query": "env"}))
        .await
        .expect("the echo_env child answers");
    assert_eq!(
        output.output, "spec/cfg",
        "the child must see the spec's value for the shared key and the config's own key"
    );
}

// --- streaming --------------------------------------------------------------

#[tokio::test]
async fn stream_yields_the_child_chunks_in_order() {
    let process = spawner("stream")
        .spawn_process(spec())
        .await
        .expect("the fake child launches");

    let frames = tokio::time::timeout(
        BOUND,
        process
            .stream(json!({"query": "stream"}))
            .collect::<Vec<_>>(),
    )
    .await
    .expect("the stream must not hang");

    let expected = ["one", "two", "three"];
    assert_eq!(frames.len(), expected.len(), "one chunk per child frame");
    for (position, (frame, text)) in frames.iter().zip(expected).enumerate() {
        let chunk = frame.as_ref().expect("every child frame is valid");
        assert_eq!(chunk.kind, "content");
        assert_eq!(
            chunk.index,
            i64::try_from(position).expect("a small index"),
            "chunks arrive in child order"
        );
        assert_eq!(chunk.payload, json!({"content": text}));
    }
    assert!(process.is_alive(), "the child survives its own stream");
}

// --- failure paths ----------------------------------------------------------

#[tokio::test]
async fn a_child_that_dies_mid_request_is_a_transport_error() {
    let process = spawner("crash")
        .spawn_process(spec())
        .await
        .expect("the fake child launches");

    let error = send_bounded(process.as_ref(), json!({"query": "boom"}))
        .await
        .expect_err("a child that exited cannot answer");
    assert_eq!(
        error.kind(),
        ErrorKind::Transport,
        "unexpected error: {error}"
    );
    assert!(
        wait_until(BOUND, || !process.is_alive()).await,
        "the handle reports the exited child"
    );
}

#[tokio::test]
async fn a_valid_json_non_frame_is_a_validation_error() {
    let process = spawner("bad_frame")
        .spawn_process(spec())
        .await
        .expect("the fake child launches");

    let error = send_bounded(process.as_ref(), json!({"query": "bad"}))
        .await
        .expect_err("valid JSON that is not a frame breaks the protocol");
    assert_eq!(
        error.kind(),
        ErrorKind::Validation,
        "unexpected error: {error}"
    );
    assert!(
        error.to_string().contains("invalid spawn message line"),
        "unexpected message: {error}"
    );
}

#[tokio::test]
async fn an_oversized_frame_is_a_validation_error() {
    let spawner = SubprocessSpawner::new(child_config("oversize").with_max_frame_bytes(512));
    let process = spawner
        .spawn_process(spec())
        .await
        .expect("the fake child launches");

    let error = send_bounded(process.as_ref(), json!({"query": "big"}))
        .await
        .expect_err("a line over the frame cap is rejected");
    assert_eq!(
        error.kind(),
        ErrorKind::Validation,
        "unexpected error: {error}"
    );
    assert!(
        error.to_string().contains("512 byte cap"),
        "unexpected message: {error}"
    );
}

#[tokio::test]
async fn a_silent_child_trips_the_request_deadline() {
    let spawner = SubprocessSpawner::new(
        child_config("silent").with_request_timeout(Duration::from_millis(300)),
    );
    let process = spawner
        .spawn_process(spec())
        .await
        .expect("the fake child launches");

    let error = send_bounded(process.as_ref(), json!({"query": "no answer"}))
        .await
        .expect_err("a child that never replies must trip the deadline");
    assert_eq!(
        error.kind(),
        ErrorKind::Timeout,
        "unexpected error: {error}"
    );
    assert!(
        error.to_string().contains("subprocess request timed out"),
        "unexpected message: {error}"
    );
}

// --- liveness and health ----------------------------------------------------

#[tokio::test]
async fn health_reports_liveness_and_failure() {
    // A cooperative child answers the probe and stays healthy.
    let healthy = spawner("echo")
        .spawn_process(spec())
        .await
        .expect("the fake child launches");
    assert!(healthy.is_healthy(), "a fresh child starts healthy");
    let reported = tokio::time::timeout(BOUND, healthy.health())
        .await
        .expect("the probe is bounded")
        .expect("a cooperative child answers the probe");
    assert!(reported, "the echo child reports itself healthy");
    assert!(healthy.is_healthy(), "a healthy probe is recorded");

    // A child that answers `false` is observably unhealthy.
    let failing = spawner("health_fail")
        .spawn_process(spec())
        .await
        .expect("the fake child launches");
    let reported = tokio::time::timeout(BOUND, failing.health())
        .await
        .expect("the probe is bounded")
        .expect("the probe itself succeeds");
    assert!(!reported, "the health_fail child reports itself unhealthy");
    assert!(
        !failing.is_healthy(),
        "the failure is recorded on the handle"
    );

    // A child that never answers trips the probe deadline and is recorded
    // unhealthy.
    let silent = spawner("silent")
        .spawn_process(spec())
        .await
        .expect("the fake child launches");
    let error = tokio::time::timeout(BOUND, silent.health())
        .await
        .expect("the probe deadline is far below the suite bound")
        .expect_err("a silent child cannot answer the probe");
    assert_eq!(
        error.kind(),
        ErrorKind::Timeout,
        "unexpected error: {error}"
    );
    assert!(!silent.is_healthy(), "an unanswered probe is unhealthy");
}

// --- runtime re-tuning of the background health interval --------------------

#[tokio::test]
async fn the_background_health_interval_is_re_tunable_on_a_live_child() {
    // A configured cadence starts the background loop; the handle reports it.
    let process = SubprocessSpawner::new(child_config("echo").with_health_check(
        Duration::from_millis(250),
        Duration::from_millis(300),
        3,
    ))
    .spawn_process(spec())
    .await
    .expect("the fake child launches");
    assert_eq!(process.health_interval(), Duration::from_millis(250));

    // Zero stops background probing, and explicit probes still answer.
    tokio::time::timeout(BOUND, process.set_health_interval(Duration::ZERO))
        .await
        .expect("a re-tune must not hang")
        .expect("a live child accepts a re-tune");
    assert_eq!(process.health_interval(), Duration::ZERO);
    assert!(
        tokio::time::timeout(BOUND, process.health())
            .await
            .expect("the probe is bounded")
            .expect("the child answers a probe without a background loop"),
        "an explicit probe is independent of the background cadence"
    );

    // And back: the restarted loop follows the value the handle holds, and the
    // child itself was never restarted or put out of sync.
    tokio::time::timeout(
        BOUND,
        process.set_health_interval(Duration::from_millis(100)),
    )
    .await
    .expect("a re-tune must not hang")
    .expect("a live child accepts a re-tune");
    assert_eq!(process.health_interval(), Duration::from_millis(100));
    assert!(process.is_alive(), "a re-tune never restarts the child");
    assert!(
        process.pid().is_some(),
        "the same OS process is still there"
    );
    let output = send_bounded(process.as_ref(), json!({"query": "after-re-tune"}))
        .await
        .expect("a re-tuned handle is still in sync");
    assert_eq!(output.output, "after-re-tune");

    // Repeating a re-tune is idempotent, and the shutdown still works.
    for _ in 0..3 {
        tokio::time::timeout(BOUND, process.set_health_interval(Duration::ZERO))
            .await
            .expect("a re-tune must not hang")
            .expect("a live child accepts a re-tune");
        assert_eq!(process.health_interval(), Duration::ZERO);
    }
    assert!(process.is_alive());
    assert!(
        tokio::time::timeout(BOUND, process.shutdown(Duration::from_secs(2)))
            .await
            .expect("shutdown must not hang")
            .expect("the fake child acknowledges the shutdown")
    );

    // A stopped handle supervises nothing, so it refuses rather than starting a
    // loop on a dead child.
    let error = process
        .set_health_interval(Duration::from_millis(100))
        .await
        .expect_err("a stopped child cannot be re-tuned");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("has stopped"), "{error}");
}

// --- shutdown and ownership -------------------------------------------------

#[tokio::test]
async fn an_acknowledged_shutdown_exits_gracefully() {
    let process = spawner("shutdown_ack")
        .spawn_process(spec())
        .await
        .expect("the fake child launches");
    assert!(process.is_alive(), "the child is live before the shutdown");

    let graceful = tokio::time::timeout(BOUND, process.shutdown(Duration::from_secs(2)))
        .await
        .expect("shutdown must not hang")
        .expect("a cooperative child shuts down without an error");
    assert!(graceful, "the child acknowledged the shutdown and exited");
    assert!(
        wait_until(BOUND, || !process.is_alive()).await,
        "the child is gone after the graceful shutdown"
    );
}

#[tokio::test]
async fn an_unresponsive_child_is_killed_and_reports_false() {
    let process = spawner("unresponsive_shutdown")
        .spawn_process(spec())
        .await
        .expect("the fake child launches");
    // Prove the child is fully functional first: it ignores `shutdown`, not
    // invocations.
    let output = send_bounded(process.as_ref(), json!({"query": "still here"}))
        .await
        .expect("an unresponsive_shutdown child still answers invocations");
    assert_eq!(output.output, "still here");

    let graceful = tokio::time::timeout(BOUND, process.shutdown(Duration::from_millis(500)))
        .await
        .expect("shutdown must not hang")
        .expect("escalating to a kill is not an error");
    assert!(
        !graceful,
        "a child that never acknowledges had to be killed"
    );
    assert!(
        wait_until(BOUND, || !process.is_alive()).await,
        "the killed child is reaped"
    );
}

#[tokio::test]
async fn dropping_the_handle_kills_the_child() {
    let process = spawner("echo")
        .spawn_process(spec())
        .await
        .expect("the fake child launches");
    let pid = process.pid().expect("the OS reported a process id");
    let proc_entry = format!("/proc/{pid}");
    assert!(
        Path::new(&proc_entry).exists(),
        "the child is running before the handle is dropped"
    );

    drop(process);

    // No child outlives its handle: the pid must disappear (which needs a proc
    // filesystem to observe; elsewhere the assertion is vacuously true).
    let gone = wait_until(BOUND, || !Path::new(&proc_entry).exists()).await;
    assert!(gone, "child {pid} outlived its handle");
}
