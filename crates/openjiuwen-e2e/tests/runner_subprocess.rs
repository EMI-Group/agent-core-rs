//! The runner's real subprocess spawn seam, driven over stdio pipes.
//!
//! This scenario answers the question the per-crate runner suite cannot answer
//! alone: does `Runner::set_spawner` + `Runner::spawn_agent` + `send` really
//! reach a *separate operating-system process* over the NDJSON protocol, and
//! does that process shut down cleanly?
//!
//! The "agent child" is this very test binary re-executed. That keeps the
//! scenario hermetic — no fixture crate, no second binary, no Python, no
//! network — while still exercising the real `tokio::process` transport, the
//! real framing helpers, and the real process lifecycle. The child entry point
//! is deliberately a **normal** `#[test]` rather than an `#[ignore]`d one: it
//! asserts nothing and returns immediately unless `E2E_FAKE_AGENT_CHILD` is set,
//! so `cargo test` runs it as a passing no-op and the parent — which sets that
//! variable in the child's environment — gets the blocking protocol loop.
//!
//! The child is launched with
//! `["fake_agent_child", "--exact", "--test-threads=1", "--nocapture"]`.
//! `--nocapture` is mandatory: without it libtest redirects the test thread's
//! stdout into its capture buffer and the protocol frames never reach the pipe.
//! libtest still prints its own non-JSON lines (`running 1 test`,
//! `test fake_agent_child ...`, `test result: ok.`); the parent's frame reader
//! skips anything that is not JSON, so that noise is harmless.
//!
//! `E2E_FAKE_AGENT_MODE` selects the child's behaviour: `echo` (default),
//! `error_frame` (an `error` frame instead of an answer) and `bad_frame` (JSON
//! that is not a protocol frame). Every await is bounded by
//! [`support::within`], and `SubprocessConfig::with_request_timeout(BOUND)`
//! makes a hung child fail its exchange instead of stalling the suite.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod support;

use std::io::{BufRead, Write};
use std::sync::Arc;
use std::time::Duration;

use openjiuwen::core::common::error::ErrorKind;
use openjiuwen::core::runner::spawn::{
    AgentSpawner, SpawnMessage, SpawnMessageKind, SpawnSpec, SpawnedAgent, SubprocessConfig,
    SubprocessSpawner,
};
use openjiuwen::core::runner::Runner;
use openjiuwen::core::single_agent::{AgentOutput, ResultType};
use serde_json::{json, Value};

use support::{within, BOUND};

/// How often a poll-based assertion re-probes its condition.
const POLL_INTERVAL: Duration = Duration::from_millis(20);

/// Set on the spawned child only: the presence of this variable is what turns
/// the no-op [`fake_agent_child`] test into the blocking protocol loop.
const CHILD_ENV: &str = "E2E_FAKE_AGENT_CHILD";

/// Selects the child's behaviour (see the module docs).
const MODE_ENV: &str = "E2E_FAKE_AGENT_MODE";

// --- the hermetic child -----------------------------------------------------

/// The protocol child: a normal test that is inert unless it is re-executed.
///
/// The parent passes [`CHILD_ENV`], its own executable path and the libtest
/// filter `fake_agent_child`, so this body owns the child's stdin/stdout for the
/// lifetime of the exchange. Running the suite normally (the variable unset)
/// leaves this a silent pass.
#[test]
fn fake_agent_child() {
    if std::env::var(CHILD_ENV).is_err() {
        return;
    }
    let mode = std::env::var(MODE_ENV).unwrap_or_default();
    child_loop(&mode);
}

/// Writes one protocol frame (one NDJSON line) and flushes it, so the parent
/// never waits on a buffered reply.
fn emit_frame(out: &mut impl Write, kind: SpawnMessageKind, payload: Value) {
    let frame = SpawnMessage::new(kind, payload);
    let line = frame.to_line().expect("a fake child frame is serializable");
    let _ = out.write_all(line.as_bytes());
    let _ = out.flush();
}

/// The text the `echo` child replies with: the invocation's `query`, or the
/// whole `inputs` value when there is no string `query`.
fn child_echo(payload: &Value) -> String {
    let inputs = &payload["inputs"];
    inputs["query"]
        .as_str()
        .map_or_else(|| inputs.to_string(), str::to_owned)
}

/// Answers one invocation according to `mode`.
///
/// `bad_frame` writes JSON that is *not* a [`SpawnMessage`] on purpose: the
/// parent must reject it as a validation failure instead of mistaking it for
/// libtest noise and waiting forever.
fn child_reply(out: &mut impl Write, mode: &str, payload: &Value) {
    match mode {
        "error_frame" => emit_frame(
            out,
            SpawnMessageKind::Error,
            json!({"message": "fake child failed on purpose"}),
        ),
        "bad_frame" => {
            let _ = writeln!(
                out,
                r#"{{"kind":"teleport","payload":null,"timestamp":0.0,"message_id":"not-a-frame"}}"#
            );
            let _ = out.flush();
        }
        _ => {
            let output = AgentOutput {
                output: child_echo(payload),
                result_type: ResultType::Answer,
                ..AgentOutput::default()
            };
            let payload = serde_json::to_value(output)
                .expect("an AgentOutput with a string and a result type is serializable");
            emit_frame(out, SpawnMessageKind::Output, payload);
        }
    }
}

/// The blocking child loop: read one frame per line, answer it, and leave on
/// `shutdown`.
fn child_loop(mode: &str) {
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    // libtest prints `running 1 test` and then `test fake_agent_child ... `
    // without a trailing newline before this body runs; closing that line keeps
    // every protocol frame on a line of its own.
    let _ = writeln!(out);
    let _ = out.flush();

    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    // The bootstrap frame (`{"spawn_spec": ...}`) arrives before any invocation.
    if !matches!(lines.next(), Some(Ok(_))) {
        return;
    }

    for line in lines {
        let Ok(line) = line else { break };
        // libtest noise is not a frame; the parent writes only frames.
        let Ok(frame) = SpawnMessage::from_line(&line) else {
            continue;
        };
        match frame.kind {
            // The bootstrap frame carries no inputs and needs no reply.
            SpawnMessageKind::Input if frame.payload.get("spawn_spec").is_some() => {}
            SpawnMessageKind::Input => child_reply(&mut out, mode, &frame.payload),
            SpawnMessageKind::HealthCheck => {
                emit_frame(
                    &mut out,
                    SpawnMessageKind::HealthCheckResponse,
                    json!({"healthy": true}),
                );
            }
            SpawnMessageKind::Shutdown => {
                emit_frame(&mut out, SpawnMessageKind::ShutdownAck, Value::Null);
                return;
            }
            other => {
                // Nothing else is expected from the parent; ignoring it keeps
                // the child's contract explicit rather than silently dropping.
                let _ = other;
            }
        }
    }
}

// --- parent-side fixtures ---------------------------------------------------

/// The launch configuration for a fake child in `mode`.
///
/// The background health task is disabled (`Duration::ZERO` interval) so no
/// probe races an exchange; explicit `health()` calls still work, and they are
/// bounded tightly enough to fail fast.
fn child_config(mode: &str) -> SubprocessConfig {
    let program = std::env::current_exe()
        .expect("the test binary path is available")
        .to_string_lossy()
        .into_owned();
    SubprocessConfig::new(program)
        // `--nocapture` is mandatory: libtest would otherwise capture the
        // child's stdout and the frames would never reach the pipe.
        .with_args([
            "fake_agent_child",
            "--exact",
            "--test-threads=1",
            "--nocapture",
        ])
        .with_env(CHILD_ENV, "1")
        .with_env(MODE_ENV, mode)
        .with_request_timeout(BOUND)
        .with_shutdown_timeout(BOUND)
        .with_health_check(Duration::ZERO, Duration::from_secs(5), 3)
}

/// The spawner every test drives. Kept concrete so the health scenario can use
/// the richer `SpawnedProcess` handle; the runner-facing tests coerce it to the
/// [`AgentSpawner`] trait object themselves.
fn spawner(mode: &str) -> SubprocessSpawner {
    SubprocessSpawner::new(child_config(mode))
}

/// The spawn spec the runner hands over; the fake child ignores its contents.
fn spec() -> SpawnSpec {
    SpawnSpec::class_agent("tests.runner_subprocess", "FakeAgent")
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

/// Shuts a child down and asserts it left cleanly, so no scenario leaves a live
/// process behind.
async fn stop(agent: &dyn SpawnedAgent) {
    let graceful = within("spawned.shutdown", agent.shutdown(BOUND))
        .await
        .expect("a cooperative child shuts down without an error");
    assert!(graceful, "the fake child acknowledged the shutdown");
}

// --- scenarios --------------------------------------------------------------

#[tokio::test]
async fn the_runner_spawns_a_real_subprocess_agent() {
    let runner = Runner::new();
    runner.set_spawner(Arc::new(spawner("echo")));

    let spawned = within("runner.spawn_agent", runner.spawn_agent(spec()))
        .await
        .expect("the subprocess spawner launches the fake child");

    let pid = spawned.pid().expect("the OS reported a process id");
    assert_ne!(pid, 0, "the reported pid is nonsense");
    assert!(spawned.is_alive(), "a freshly spawned child is alive");
    assert!(
        !spawned.id().is_empty(),
        "the handle needs a stable id: {:?}",
        spawned.id()
    );

    let output = within("spawned.send", spawned.send(json!({"query": "hi"})))
        .await
        .expect("the echo child answers the invocation");

    // The child serialized a real `AgentOutput`; the parent deserialized it.
    assert_eq!(output.output, "hi", "the child echoed the query");
    assert_eq!(output.result_type, ResultType::Answer);
    assert!(!output.is_interrupt());
    assert!(spawned.is_alive(), "a reply does not end the child");

    stop(spawned.as_ref()).await;
}

#[tokio::test]
async fn the_spawner_seam_returns_a_working_trait_object() {
    // Bind the trait object explicitly: the seam must hand back usable
    // `Arc<dyn SpawnedAgent>`, not only a concrete handle.
    let agent: Arc<dyn SpawnedAgent> = within("spawner.spawn", spawner("echo").spawn(spec()))
        .await
        .expect("the trait seam launches the child");

    let first = within("first send", agent.send(json!({"query": "first"})))
        .await
        .expect("the first invocation round-trips");
    let second = within("second send", agent.send(json!({"query": "second"})))
        .await
        .expect("a second invocation round-trips on the same child");

    // Each reply answers its own request, so the exchange is correlated even
    // though the protocol has no request id.
    assert_eq!(first.output, "first");
    assert_eq!(second.output, "second");
    assert_ne!(
        first.output, second.output,
        "the child echoes the payload it received, so the replies differ"
    );
    assert_eq!(first.result_type, ResultType::Answer);
    assert_eq!(second.result_type, ResultType::Answer);
    assert!(agent.pid().is_some(), "the seam reports the child's pid");

    stop(agent.as_ref()).await;
}

#[tokio::test]
async fn health_probe_and_graceful_shutdown_round_trip() {
    // This drives the very backend `Runner::set_spawner` installs; the concrete
    // `SpawnedProcess` handle adds the recorded health state and the pid, which
    // the `SpawnedAgent` trait does not expose.
    let spawned = within(
        "spawner.spawn_process",
        spawner("echo").spawn_process(spec()),
    )
    .await
    .expect("the subprocess spawner launches the fake child");

    let healthy = within("spawned.health", spawned.health())
        .await
        .expect("the health probe itself succeeds");
    assert!(healthy, "the fake child reports itself healthy");
    assert!(
        spawned.is_healthy(),
        "a successful probe is recorded on the handle"
    );

    let graceful = within("spawned.shutdown", spawned.shutdown(BOUND))
        .await
        .expect("a cooperative child shuts down without an error");
    assert!(graceful, "the child acknowledged the shutdown and exited");
    assert!(
        wait_until(BOUND, || !spawned.is_alive()).await,
        "the child is gone after the graceful shutdown"
    );
}

#[tokio::test]
async fn a_broken_child_surfaces_a_typed_error_instead_of_hanging() {
    // An `error` frame is a transport failure carrying the child's message.
    let failing = within("spawner.spawn", spawner("error_frame").spawn(spec()))
        .await
        .expect("the broken child still launches");
    let error = within("failing send", failing.send(json!({"query": "boom"})))
        .await
        .expect_err("an error frame is not an answer");
    assert_eq!(
        error.kind(),
        ErrorKind::Transport,
        "unexpected error: {error}"
    );
    assert!(
        error.to_string().contains("fake child failed on purpose"),
        "the child's message reaches the caller: {error}"
    );

    // JSON that is not a protocol frame is rejected, not read as stdout noise.
    let garbled = within("spawner.spawn", spawner("bad_frame").spawn(spec()))
        .await
        .expect("the garbled child still launches");
    let error = within("garbled send", garbled.send(json!({"query": "boom"})))
        .await
        .expect_err("a non-frame JSON object is not an answer");
    assert_eq!(
        error.kind(),
        ErrorKind::Validation,
        "unexpected error: {error}"
    );

    // Both handles are dropped here; the backend kills its child on drop.
}
