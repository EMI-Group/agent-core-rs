//! Behaviour suite for the spawn seam's health, snapshot, and cancellation
//! accessors, and for the backends that serve them.
//!
//! Every handle assertion goes through `Arc<dyn SpawnedAgent>` — never the
//! concrete `SpawnedProcess`/`InProcessAgent`/`RemoteAgent` type — because the
//! seam's promise is that a host can probe, snapshot, and cancel a child without
//! knowing which backend launched it.
//!
//! Hermetic and offline. The subprocess and external-command cases re-execute
//! this very test binary with the libtest filter of the `#[ignore]`d
//! [`fake_child`] test (see the doc comment there for the `--nocapture` and
//! leading-newline requirements); the in-process cases host a local handler; the
//! remote case talks to a loopback `wiremock` server.
//!
//! Ports the shape of Python `tests/unit_tests/core/runner/spawn/`
//! (`test_child_process.py`, `test_process_manager.py`) plus the health surface
//! of `openjiuwen/core/runner/spawn/process_manager.py`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_agent::output::{AgentChunk, AgentOutput};
use openjiuwen_common::error::{ErrorKind, Result};
use openjiuwen_runner::spawn::{
    AgentSpawner, ExternalCommandSpawner, InProcessHandler, InProcessSpawner, RemoteSpawner,
    SpawnBackend, SpawnHealth, SpawnSpec, SpawnedAgent, SubprocessConfig, SubprocessSpawner,
};
use serde_json::{json, Value};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

/// Upper bound on any await in this suite, so a broken exchange fails the test
/// instead of stalling the harness.
const BOUND: Duration = Duration::from_secs(10);

/// How often a poll-based assertion re-probes its condition.
const POLL_INTERVAL: Duration = Duration::from_millis(20);

// --- shared helpers ---------------------------------------------------------

/// The spawn spec every test hands over; the fake child ignores its contents and
/// the remote backend reads the agent id (`echo`) out of it.
fn spec() -> SpawnSpec {
    SpawnSpec::class_agent("tests.spawn_seam", "echo")
}

/// The requests `server` recorded whose path `keep` accepts, in arrival order.
///
/// `wiremock` hands its servers out of a process-wide pool, so a recording can
/// hold a request another test left in flight when its server was returned to
/// the pool and reset for us. These assertions are about what this test sent, so
/// they filter to the paths it owns instead of counting the whole recording.
async fn recorded(server: &MockServer, keep: impl Fn(&str) -> bool) -> Vec<Request> {
    server
        .received_requests()
        .await
        .expect("the mock server recorded its requests")
        .into_iter()
        .filter(|request| keep(request.url.path()))
        .collect()
}

/// Runs one health probe under [`BOUND`].
async fn health_bounded(agent: &Arc<dyn SpawnedAgent>) -> Result<bool> {
    tokio::time::timeout(BOUND, agent.health())
        .await
        .expect("a health probe must not hang")
}

/// Runs one invocation under [`BOUND`].
async fn send_bounded(agent: &Arc<dyn SpawnedAgent>, inputs: Value) -> Result<AgentOutput> {
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
/// spontaneous probe races the assertions; explicit
/// [`SpawnedAgent::health`] calls still work, bounded tightly so a silent child
/// fails fast.
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

/// Writes one protocol frame (one NDJSON line) and flushes it.
fn emit_frame(out: &mut impl Write, counter: &mut u64, kind: &str, payload: &Value) {
    *counter = counter.saturating_add(1);
    let pid = std::process::id();
    let sequence = *counter;
    let frame = json!({
        "kind": kind,
        "payload": payload,
        "timestamp": 1.0,
        "message_id": format!("child-{pid}-{sequence}"),
    });
    let _ = writeln!(out, "{frame}");
    let _ = out.flush();
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
            // `health_fail` is the mode that answers a probe with
            // `healthy: false`; every other mode reports itself healthy.
            emit_frame(
                out,
                counter,
                "health_check_response",
                &json!({"healthy": mode != "health_fail"}),
            );
            false
        }
        "shutdown" => {
            emit_frame(out, counter, "shutdown_ack", &Value::Null);
            true
        }
        "input" => {
            let text = payload["inputs"]["query"]
                .as_str()
                .map_or_else(|| payload["inputs"].to_string(), str::to_owned);
            emit_frame(
                out,
                counter,
                "output",
                &json!({"output": text, "result_type": "answer"}),
            );
            false
        }
        _ => false,
    }
}

/// The protocol child: an `#[ignore]`d test that this suite re-executes.
///
/// Modes (`FAKE_CHILD_MODE`): `echo` (default) answers invocations and health
/// probes and acknowledges a shutdown; `health_fail` answers invocations and
/// acknowledges a shutdown but reports `healthy: false` from every probe.
#[test]
#[ignore = "spawned as a stdio protocol child by the spawn-seam tests"]
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
}

// --- subprocess -------------------------------------------------------------

#[tokio::test]
async fn the_subprocess_handle_reports_identity_health_and_uncancellable_tasks() {
    let spawner = SubprocessSpawner::new(child_config("echo"));
    let child: Arc<dyn SpawnedAgent> = spawner
        .spawn(spec())
        .await
        .expect("the fake child launches through the seam");

    assert_eq!(child.backend(), SpawnBackend::Subprocess);
    assert!(!child.id().is_empty(), "the handle needs a stable id");
    let pid = child.pid().expect("the OS reported a process id");
    assert_ne!(pid, 0, "the reported pid is nonsense");
    assert!(child.is_alive(), "a freshly spawned child is alive");
    assert!(child.is_healthy(), "a fresh child starts healthy");

    assert!(
        health_bounded(&child)
            .await
            .expect("a cooperative child answers the probe"),
        "the echo child reports itself healthy"
    );

    // The snapshot is composed from the live accessors, not fabricated.
    let snapshot = child.health_snapshot();
    assert_eq!(snapshot.backend, SpawnBackend::Subprocess);
    assert_eq!(snapshot.alive, child.is_alive());
    assert_eq!(snapshot.healthy, child.is_healthy());
    assert_eq!(snapshot.pid, child.pid());
    assert_eq!(snapshot.last_error, None);

    let output = send_bounded(&child, json!({"query": "through-the-seam"}))
        .await
        .expect("the echo child answers through the seam");
    assert_eq!(output.output, "through-the-seam");

    // The NDJSON protocol carries no request correlation id, so a child handle
    // owns no cancellable task and says so instead of pretending.
    let cancel = child
        .cancel_task("t-1")
        .await
        .expect_err("the subprocess backend cannot cancel a task");
    assert_eq!(cancel.kind(), ErrorKind::Unsupported);
    assert!(cancel.to_string().contains("subprocess"), "{cancel}");
    assert!(
        cancel.to_string().contains("cancellable task handle"),
        "{cancel}"
    );

    let graceful = tokio::time::timeout(BOUND, child.shutdown(Duration::from_secs(2)))
        .await
        .expect("shutdown must not hang")
        .expect("a cooperative child shuts down without an error");
    assert!(graceful, "the child acknowledged the shutdown and exited");
    assert!(
        wait_until(BOUND, || !child.is_alive()).await,
        "the child is gone after the graceful shutdown"
    );
    assert!(
        !child.health_snapshot().alive,
        "the snapshot follows the reaped child"
    );
}

#[tokio::test]
async fn a_child_that_reports_itself_unhealthy_is_observable_through_the_seam() {
    let spawner = SubprocessSpawner::new(child_config("health_fail"));
    let child: Arc<dyn SpawnedAgent> = spawner
        .spawn(spec())
        .await
        .expect("the fake child launches through the seam");

    assert!(
        child.is_healthy(),
        "no probe has run yet, so the child starts healthy"
    );

    assert!(
        !health_bounded(&child)
            .await
            .expect("the probe itself succeeds"),
        "the health_fail child reports itself unhealthy"
    );
    assert!(
        !child.is_healthy(),
        "the failed probe is recorded on the handle"
    );

    let snapshot = child.health_snapshot();
    assert!(!snapshot.healthy, "the snapshot carries the failed probe");
    assert!(snapshot.alive, "an unhealthy child is still running");
    assert!(
        child.is_alive(),
        "liveness stays true: healthy is the probe answer"
    );

    child
        .shutdown(Duration::from_secs(2))
        .await
        .expect("an unhealthy child can still be shut down");
}

// --- external command -------------------------------------------------------

#[tokio::test]
async fn the_external_command_backend_launches_a_resolved_program() {
    let program = std::env::current_exe()
        .expect("the test binary path is available")
        .to_string_lossy()
        .into_owned();
    let spawner = ExternalCommandSpawner::new(program.clone())
        .with_args(child_args().map(str::to_owned).to_vec())
        .with_env("FAKE_CHILD_MODE", "echo")
        .with_request_timeout(BOUND)
        .with_health_check(Duration::ZERO, Duration::from_millis(300), 3);

    assert_eq!(spawner.backend(), SpawnBackend::ExternalCommand);
    assert_eq!(spawner.command(), program, "the command is kept as written");

    let child: Arc<dyn SpawnedAgent> = spawner
        .spawn(spec())
        .await
        .expect("the resolved program launches through the seam");

    // The child is supervised by the subprocess protocol, but it is launched by
    // the external-command backend, so the handle, the spawner, and the runner's
    // monitor detail all name that backend.
    assert_eq!(child.backend(), SpawnBackend::ExternalCommand);
    assert_eq!(
        child.health_snapshot().backend,
        SpawnBackend::ExternalCommand
    );
    assert!(
        child.pid().is_some(),
        "the resolved program is a real child"
    );
    assert!(child.is_alive());

    let output = send_bounded(&child, json!({"query": "via-external-command"}))
        .await
        .expect("the external command speaks the protocol");
    assert_eq!(output.output, "via-external-command");

    child
        .shutdown(Duration::from_secs(2))
        .await
        .expect("the external command shuts down");
}

#[tokio::test]
async fn a_missing_external_command_is_not_found_and_launches_nothing() {
    let directory = tempfile::tempdir().expect("a temp directory");
    let spawner =
        ExternalCommandSpawner::new("openjiuwen-not-installed").with_search_path(directory.path());

    let error = spawner
        .spawn(spec())
        .await
        .err()
        .expect("a missing command cannot be launched");
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(
        error.to_string().contains("openjiuwen-not-installed"),
        "the message must name the command: {error}"
    );
    assert_eq!(
        std::fs::read_dir(directory.path())
            .expect("the search directory is readable")
            .count(),
        0,
        "resolution happens before anything is launched"
    );
}

// --- in-process -------------------------------------------------------------

/// A hosted handler that records the calls it served.
struct EchoHandler {
    /// Whether the handler still serves calls.
    alive: AtomicBool,
    /// How many times it was stopped.
    shutdowns: AtomicUsize,
    /// The inputs of every invocation it answered.
    seen: Mutex<Vec<Value>>,
}

impl EchoHandler {
    /// An alive handler with no calls recorded.
    fn new() -> Arc<Self> {
        Arc::new(Self {
            alive: AtomicBool::new(true),
            shutdowns: AtomicUsize::new(0),
            seen: Mutex::new(Vec::new()),
        })
    }

    /// The invocations the handler answered.
    fn seen(&self) -> Vec<Value> {
        self.seen.lock().unwrap().clone()
    }

    /// How many times the handler was stopped.
    fn shutdowns(&self) -> usize {
        self.shutdowns.load(Ordering::SeqCst)
    }
}

#[async_trait]
#[allow(clippy::unused_async_trait_impl)]
impl InProcessHandler for EchoHandler {
    fn is_alive(&self) -> bool {
        self.alive.load(Ordering::SeqCst)
    }

    async fn send(&self, inputs: Value) -> Result<AgentOutput> {
        self.seen.lock().unwrap().push(inputs.clone());
        Ok(AgentOutput::answer(format!("echo: {inputs}")))
    }

    fn stream(&self, inputs: Value) -> BoxStream<'static, Result<AgentChunk>> {
        Box::pin(futures::stream::iter(vec![
            Ok(AgentChunk::llm_output(0, "first")),
            Ok(AgentChunk::llm_output(1, inputs.to_string())),
        ]))
    }

    async fn shutdown(&self) -> Result<()> {
        self.shutdowns.fetch_add(1, Ordering::SeqCst);
        self.alive.store(false, Ordering::SeqCst);
        Ok(())
    }
}

#[tokio::test]
async fn the_in_process_backend_serves_calls_through_the_seam() {
    let handler = EchoHandler::new();
    let spawner = InProcessSpawner::from_handler(Arc::clone(&handler) as Arc<dyn InProcessHandler>);
    let child: Arc<dyn SpawnedAgent> = spawner
        .spawn(spec())
        .await
        .expect("an in-process handler always spawns");

    assert_eq!(child.backend(), SpawnBackend::InProcess);
    assert_eq!(child.pid(), None, "an in-process handler has no OS process");
    assert!(child.is_alive());
    assert!(child.is_healthy());
    assert!(
        health_bounded(&child)
            .await
            .expect("the in-process handle answers its own probe"),
        "an open handle reports itself healthy"
    );

    let re_tune = child
        .set_health_interval(Duration::from_secs(1))
        .await
        .expect_err("an in-process handle has no background health loop");
    assert_eq!(re_tune.kind(), ErrorKind::Unsupported);
    assert!(
        re_tune.to_string().contains("in_process"),
        "the message names the backend: {re_tune}"
    );

    let output = send_bounded(&child, json!({"query": "in-process"}))
        .await
        .expect("the handler answers through the seam");
    assert_eq!(output.output, "echo: {\"query\":\"in-process\"}");
    assert_eq!(handler.seen(), vec![json!({"query": "in-process"})]);

    let frames = tokio::time::timeout(
        BOUND,
        child.stream(json!({"query": "stream"})).collect::<Vec<_>>(),
    )
    .await
    .expect("an in-process stream must end on its own");
    assert_eq!(frames.len(), 2, "one chunk item per handler frame");
    assert_eq!(frames[0].as_ref().unwrap().index, 0);

    assert!(
        tokio::time::timeout(BOUND, child.shutdown(Duration::from_secs(1)))
            .await
            .expect("shutdown must not hang")
            .expect("stopping the handler is not an error"),
        "a stopped handler reports a graceful exit"
    );
    assert!(!child.is_alive(), "the handle is dead after the shutdown");
    assert!(!child.is_healthy());
    assert!(
        !health_bounded(&child)
            .await
            .expect("a stopped handle still answers the probe"),
        "a stopped handle reports itself unhealthy"
    );

    // Stopping twice, and killing an already-stopped handle, are both `Ok`.
    assert!(child
        .shutdown(Duration::from_secs(1))
        .await
        .expect("a second shutdown is idempotent"),);
    assert!(
        child.force_kill().await.is_ok(),
        "killing a stopped in-process handle is not an error"
    );
    assert_eq!(handler.shutdowns(), 3, "two shutdowns and one force_kill");
    assert!(!child.is_alive());
}

#[tokio::test]
async fn the_in_process_factory_runs_once_per_spawn_with_the_spec() {
    let sessions: Arc<Mutex<Vec<Option<String>>>> = Arc::new(Mutex::new(Vec::new()));
    let seen_by_factory = Arc::clone(&sessions);
    let spawner = InProcessSpawner::new(Arc::new(move |spec: &SpawnSpec| {
        seen_by_factory
            .lock()
            .unwrap()
            .push(spec.session_id.clone());
        Ok(EchoHandler::new() as Arc<dyn InProcessHandler>)
    }));

    let first: Arc<dyn SpawnedAgent> = spawner
        .spawn(SpawnSpec::class_agent("tests.spawn_seam", "echo").with_session_id("s-1"))
        .await
        .expect("the factory builds the first handler");
    let second: Arc<dyn SpawnedAgent> = spawner
        .spawn(SpawnSpec::class_agent("tests.spawn_seam", "echo").with_session_id("s-2"))
        .await
        .expect("the factory builds the second handler");

    assert_eq!(
        sessions.lock().unwrap().clone(),
        vec![Some("s-1".to_owned()), Some("s-2".to_owned())],
        "the factory sees each spawn's own spec"
    );
    assert_ne!(first.id(), second.id(), "each spawn gets its own handle");
    assert_eq!(first.backend(), SpawnBackend::InProcess);
    assert_eq!(second.backend(), SpawnBackend::InProcess);
    assert_eq!(first.pid(), None);
    assert_eq!(second.pid(), None);
    assert!(first.is_alive() && second.is_alive());
}

// --- remote -----------------------------------------------------------------

/// Whether `path` is one of the remote backend's routes for the `echo` spec, i.e.
/// a request this test's own client could have sent.
fn remote_routes(path: &str) -> bool {
    matches!(
        path,
        "/agents/echo/invoke" | "/agents/echo/stream" | "/agents/echo/cancel"
    )
}

#[tokio::test]
async fn the_remote_backend_reports_health_and_cancels_through_the_seam() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/agents/echo/invoke"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "output": "remote-hi",
            "result_type": "answer",
        })))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/agents/echo/cancel"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;

    let spawner = RemoteSpawner::new(server.uri());
    assert_eq!(spawner.backend(), SpawnBackend::RemoteHttp);
    let child: Arc<dyn SpawnedAgent> = spawner
        .spawn(spec())
        .await
        .expect("a remote handle is built and started at spawn");
    assert_eq!(child.backend(), SpawnBackend::RemoteHttp);
    assert_eq!(child.id(), "echo");
    assert_eq!(child.pid(), None, "a remote agent has no OS process");
    assert!(
        child.is_alive() && child.is_healthy(),
        "a spawned remote handle is started, so it reports itself live"
    );
    assert!(
        health_bounded(&child)
            .await
            .expect("the remote handle answers its own probe"),
        "a spawned remote handle is healthy before any request"
    );
    assert!(
        recorded(&server, remote_routes).await.is_empty(),
        "the spawn itself makes no request"
    );

    let output = send_bounded(&child, json!({"query": "hi"}))
        .await
        .expect("the mount answers the invocation");
    assert_eq!(output.output, "remote-hi");
    assert!(
        health_bounded(&child)
            .await
            .expect("the started client answers the probe"),
        "the client stays started across a request"
    );

    // The client ships with a cancel path configured, so cancellation works
    // through the seam without any extra wiring.
    child
        .cancel_task("t-1")
        .await
        .expect("the configured cancel path answers");
    let received = recorded(&server, remote_routes).await;
    assert_eq!(received.len(), 2, "one invoke, one cancel");
    assert_eq!(received[1].url.path(), "/agents/echo/cancel");
    assert_eq!(
        serde_json::from_slice::<Value>(&received[1].body).expect("the cancel body is JSON"),
        json!({"task_id": "t-1"})
    );

    // No background health loop runs behind an HTTP handle, so a re-tune
    // reports the seam instead of pretending to change a cadence.
    let re_tune = child
        .set_health_interval(Duration::from_secs(1))
        .await
        .expect_err("a remote handle has no background health loop");
    assert_eq!(re_tune.kind(), ErrorKind::Unsupported);
    assert!(
        re_tune.to_string().contains("remote_http"),
        "the message names the backend: {re_tune}"
    );
    assert!(
        re_tune.to_string().contains("background health loop"),
        "the message names the missing capability: {re_tune}"
    );

    let snapshot = child.health_snapshot();
    assert_eq!(snapshot.backend, SpawnBackend::RemoteHttp);
    assert_eq!(snapshot.pid, None);
    assert!(snapshot.alive);
    assert!(snapshot.healthy);
    assert_eq!(snapshot.last_error, None);

    assert!(child
        .shutdown(Duration::from_secs(1))
        .await
        .expect("a remote handle shuts its client down"));
    assert!(!child.is_alive());
}

#[tokio::test]
async fn a_spawned_remote_handle_is_started_at_spawn_and_validates_its_url() {
    let server = MockServer::start().await;

    // Readiness polled through the seam is true right after the spawn, without
    // any request having been made: the client is built and started by `spawn`.
    let child: Arc<dyn SpawnedAgent> = RemoteSpawner::new(server.uri())
        .spawn(spec())
        .await
        .expect("a remote spawn starts its client");
    assert!(child.is_alive());
    assert!(child.is_healthy());
    assert!(tokio::time::timeout(BOUND, child.health())
        .await
        .expect("the probe must not hang")
        .expect("the started handle answers its own probe"));
    let snapshot = child.health_snapshot();
    assert!(snapshot.alive && snapshot.healthy);
    assert_eq!(snapshot.backend, SpawnBackend::RemoteHttp);
    assert!(
        recorded(&server, remote_routes).await.is_empty(),
        "no request is made just to become ready"
    );

    // A base URL that is not an absolute URL fails at spawn, not at first invoke.
    let error = RemoteSpawner::new("not a url")
        .spawn(spec())
        .await
        .err()
        .expect("an unusable base url cannot be spawned");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("base_url"), "{error}");
}

// --- seam defaults ----------------------------------------------------------

/// A handle that implements only the required seam methods, so every defaulted
/// accessor is exercised.
struct MinimalChild {
    /// Stable handle id.
    id: String,
    /// Whether the handle reports itself alive.
    alive: AtomicBool,
}

impl MinimalChild {
    /// An alive handle under `id`.
    fn new(id: &str) -> Self {
        Self {
            id: id.to_owned(),
            alive: AtomicBool::new(true),
        }
    }
}

#[async_trait]
#[allow(clippy::unused_async_trait_impl)]
impl SpawnedAgent for MinimalChild {
    fn id(&self) -> &str {
        &self.id
    }

    fn is_alive(&self) -> bool {
        self.alive.load(Ordering::SeqCst)
    }

    fn pid(&self) -> Option<u32> {
        None
    }

    async fn send(&self, inputs: Value) -> Result<AgentOutput> {
        Ok(AgentOutput::answer(format!("minimal: {inputs}")))
    }

    fn stream(&self, _inputs: Value) -> BoxStream<'static, Result<AgentChunk>> {
        Box::pin(futures::stream::empty())
    }

    async fn shutdown(&self, _timeout: Duration) -> Result<bool> {
        self.alive.store(false, Ordering::SeqCst);
        Ok(true)
    }

    async fn force_kill(&self) -> Result<()> {
        self.alive.store(false, Ordering::SeqCst);
        Ok(())
    }
}

#[tokio::test]
async fn the_defaulted_seam_methods_report_the_missing_capability() {
    let child: Arc<dyn SpawnedAgent> = Arc::new(MinimalChild::new("minimal-1"));

    assert_eq!(
        child.backend(),
        SpawnBackend::Unknown,
        "a handle that does not name its backend is not misreported"
    );
    assert!(child.is_alive());
    assert!(child.is_healthy(), "liveness is the default health answer");

    let probe = child
        .health()
        .await
        .expect_err("a default handle has no probe");
    assert_eq!(probe.kind(), ErrorKind::Unsupported);
    assert!(
        probe.to_string().contains("unknown"),
        "the message names the backend: {probe}"
    );
    assert!(
        probe.to_string().contains("no probe"),
        "the message points at the fallbacks: {probe}"
    );

    let re_tune = child
        .set_health_interval(Duration::from_secs(1))
        .await
        .expect_err("a default handle runs no background health loop");
    assert_eq!(re_tune.kind(), ErrorKind::Unsupported);
    assert!(
        re_tune.to_string().contains("unknown"),
        "the message names the backend: {re_tune}"
    );
    assert!(
        re_tune.to_string().contains("background health loop"),
        "the message explains why: {re_tune}"
    );

    let cancel = child
        .cancel_task("t-1")
        .await
        .expect_err("a default handle owns no task");
    assert_eq!(cancel.kind(), ErrorKind::Unsupported);
    assert!(
        cancel.to_string().contains("unknown"),
        "the message names the backend: {cancel}"
    );
    assert!(
        cancel.to_string().contains("cancellable task handle"),
        "the message explains why: {cancel}"
    );

    assert_eq!(
        child.health_snapshot(),
        SpawnHealth {
            backend: SpawnBackend::Unknown,
            alive: true,
            healthy: true,
            pid: None,
            last_error: None,
        },
        "the default snapshot composes the accessors without fabricating a failure"
    );

    child
        .shutdown(Duration::from_secs(1))
        .await
        .expect("the required shutdown method still works");
    assert!(!child.is_alive());
    assert!(
        !child.is_healthy(),
        "the default health answer follows liveness"
    );
    assert!(!child.health_snapshot().alive);
}

#[test]
fn spawn_health_round_trips_through_snake_case_json() {
    let health = SpawnHealth {
        backend: SpawnBackend::Subprocess,
        alive: true,
        healthy: false,
        pid: Some(4242),
        last_error: Some("probe failed".to_owned()),
    };
    let encoded = serde_json::to_value(&health).expect("a snapshot is JSON-safe");
    assert_eq!(
        encoded,
        json!({
            "backend": "subprocess",
            "alive": true,
            "healthy": false,
            "pid": 4242,
            "last_error": "probe failed",
        })
    );

    let restored: SpawnHealth = serde_json::from_value(encoded).expect("the snapshot round-trips");
    assert_eq!(restored, health);

    // Unknown fields are tolerated and missing ones default.
    let partial: SpawnHealth = serde_json::from_str(r#"{"healthy": true, "future_field": 7}"#)
        .expect("unknown fields are tolerated");
    assert_eq!(
        partial,
        SpawnHealth {
            backend: SpawnBackend::Unknown,
            alive: false,
            healthy: true,
            pid: None,
            last_error: None,
        }
    );
}
