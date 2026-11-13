//! Behaviour suite for the runner's spawn facade: single-spawner installation,
//! registry routing, backend introspection, and the monitor observation of every
//! spawned child.
//!
//! The handles here are the ones the real backends hand out (an in-process
//! handler) plus local fakes that count their calls, so routing decisions are
//! observable as call counts rather than inferred from ids. Offline and
//! deterministic: no child process, no network, no sleeps for ordering.
//!
//! Ports the facade surface of Python `openjiuwen/core/runner/runner.py`
//! (`set_spawner`/`spawn_agent`) and the routing rules this crate documents on
//! `SpawnerRegistry`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::{stream::BoxStream, StreamExt};
use openjiuwen_agent::output::{AgentChunk, AgentOutput};
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_runner::{
    AgentSpawner, InProcessHandler, InProcessSpawner, RunKind, RunPhase, Runner, SpawnBackend,
    SpawnSpec, SpawnedAgent, SpawnerRegistry,
};
use serde_json::{json, Value};

// --- fixtures ---------------------------------------------------------------

/// A child handle a fake spawner hands out.
struct FakeChild {
    /// Stable handle id, unique per spawn.
    id: String,
    /// Backend the handle reports.
    backend: SpawnBackend,
}

#[async_trait]
#[allow(clippy::unused_async_trait_impl)]
impl SpawnedAgent for FakeChild {
    fn id(&self) -> &str {
        &self.id
    }

    fn is_alive(&self) -> bool {
        true
    }

    fn pid(&self) -> Option<u32> {
        Some(4242)
    }

    async fn send(&self, _inputs: Value) -> Result<AgentOutput> {
        Ok(AgentOutput::answer("fake answer"))
    }

    fn stream(&self, _inputs: Value) -> BoxStream<'static, Result<AgentChunk>> {
        Box::pin(futures::stream::empty())
    }

    async fn shutdown(&self, _timeout: Duration) -> Result<bool> {
        Ok(true)
    }

    async fn force_kill(&self) -> Result<()> {
        Ok(())
    }

    fn backend(&self) -> SpawnBackend {
        self.backend
    }
}

/// A spawner that counts its spawns and hands out a fresh [`FakeChild`].
struct FakeSpawner {
    /// Backend this spawner names.
    backend: SpawnBackend,
    /// Id prefix of the children it hands out.
    id_prefix: &'static str,
    /// How many times `spawn` was called.
    calls: Arc<AtomicUsize>,
}

impl FakeSpawner {
    /// A spawner naming `backend` with its own call counter.
    fn new(backend: SpawnBackend, id_prefix: &'static str) -> Arc<Self> {
        Arc::new(Self {
            backend,
            id_prefix,
            calls: Arc::new(AtomicUsize::new(0)),
        })
    }

    /// How many spawns it served.
    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl AgentSpawner for FakeSpawner {
    async fn spawn(&self, _spec: SpawnSpec) -> Result<Arc<dyn SpawnedAgent>> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(Arc::new(FakeChild {
            id: format!("{}-{call}", self.id_prefix),
            backend: self.backend,
        }))
    }

    fn backend(&self) -> SpawnBackend {
        self.backend
    }
}

/// A spawner whose every spawn fails.
struct FailingSpawner;

#[async_trait]
#[allow(clippy::unused_async_trait_impl)]
impl AgentSpawner for FailingSpawner {
    async fn spawn(&self, _spec: SpawnSpec) -> Result<Arc<dyn SpawnedAgent>> {
        Err(Error::transport("launch failed"))
    }
}

/// A spawner that never reports its backend, so a registry key is the only label.
struct UnlabelledSpawner {
    /// Id prefix of the children it hands out.
    id_prefix: &'static str,
}

#[async_trait]
#[allow(clippy::unused_async_trait_impl)]
impl AgentSpawner for UnlabelledSpawner {
    async fn spawn(&self, _spec: SpawnSpec) -> Result<Arc<dyn SpawnedAgent>> {
        Ok(Arc::new(FakeChild {
            id: format!("{}-0", self.id_prefix),
            backend: SpawnBackend::Unknown,
        }))
    }
}

/// A spawner handing out children whose liveness and failures the test drives.
struct ScriptedSpawner {
    /// Backend this spawner names.
    backend: SpawnBackend,
    /// Liveness flag every child reports and the test flips.
    alive: Arc<AtomicBool>,
    /// Whether `send`/`stream` answer with a transport error.
    fails: bool,
    /// How many children it handed out, so their ids are unique.
    spawned: Arc<AtomicUsize>,
}

impl ScriptedSpawner {
    /// A spawner naming `backend`, sharing `alive`, failing calls when `fails`.
    fn new(backend: SpawnBackend, alive: Arc<AtomicBool>, fails: bool) -> Arc<Self> {
        Arc::new(Self {
            backend,
            alive,
            fails,
            spawned: Arc::new(AtomicUsize::new(0)),
        })
    }
}

#[async_trait]
impl AgentSpawner for ScriptedSpawner {
    async fn spawn(&self, _spec: SpawnSpec) -> Result<Arc<dyn SpawnedAgent>> {
        let call = self.spawned.fetch_add(1, Ordering::SeqCst);
        Ok(Arc::new(ScriptedChild {
            id: format!("scripted-{call}"),
            backend: self.backend,
            alive: Arc::clone(&self.alive),
            fails: self.fails,
        }))
    }

    fn backend(&self) -> SpawnBackend {
        self.backend
    }
}

/// A child handle whose liveness, probe and failures the test controls.
struct ScriptedChild {
    /// Stable handle id, unique per spawn.
    id: String,
    /// Backend the handle reports.
    backend: SpawnBackend,
    /// Liveness the handle reports; the test flips it.
    alive: Arc<AtomicBool>,
    /// Whether `send`/`stream` answer with a transport error.
    fails: bool,
}

#[async_trait]
impl SpawnedAgent for ScriptedChild {
    fn id(&self) -> &str {
        &self.id
    }

    fn is_alive(&self) -> bool {
        self.alive.load(Ordering::SeqCst)
    }

    fn pid(&self) -> Option<u32> {
        None
    }

    async fn send(&self, _inputs: Value) -> Result<AgentOutput> {
        if self.fails {
            Err(Error::transport("the child did not answer"))
        } else {
            Ok(AgentOutput::answer("scripted"))
        }
    }

    fn stream(&self, _inputs: Value) -> BoxStream<'static, Result<AgentChunk>> {
        if self.fails {
            let frame: Result<AgentChunk> = Err(Error::transport("the child did not answer"));
            Box::pin(futures::stream::iter([frame]))
        } else {
            Box::pin(futures::stream::empty())
        }
    }

    async fn shutdown(&self, _timeout: Duration) -> Result<bool> {
        self.alive.store(false, Ordering::SeqCst);
        Ok(true)
    }

    async fn force_kill(&self) -> Result<()> {
        self.alive.store(false, Ordering::SeqCst);
        Ok(())
    }

    fn backend(&self) -> SpawnBackend {
        self.backend
    }

    async fn health(&self) -> Result<bool> {
        Ok(self.alive.load(Ordering::SeqCst))
    }
}

/// The smallest hosted handler the in-process backend accepts.
struct EchoHandler {
    /// Whether the handler still serves calls.
    alive: AtomicBool,
}

impl EchoHandler {
    /// An alive handler.
    fn new() -> Arc<Self> {
        Arc::new(Self {
            alive: AtomicBool::new(true),
        })
    }
}

#[async_trait]
#[allow(clippy::unused_async_trait_impl)]
impl InProcessHandler for EchoHandler {
    fn is_alive(&self) -> bool {
        self.alive.load(Ordering::SeqCst)
    }

    async fn send(&self, inputs: Value) -> Result<AgentOutput> {
        Ok(AgentOutput::answer(format!("echo: {inputs}")))
    }

    fn stream(&self, _inputs: Value) -> BoxStream<'static, Result<AgentChunk>> {
        Box::pin(futures::stream::empty())
    }

    async fn shutdown(&self) -> Result<()> {
        self.alive.store(false, Ordering::SeqCst);
        Ok(())
    }
}

/// `spawner` as the trait object the runner takes.
fn as_spawner<S: AgentSpawner + 'static>(spawner: Arc<S>) -> Arc<dyn AgentSpawner> {
    spawner
}

/// A real in-process backend, so routing can be observed end to end.
fn in_process() -> InProcessSpawner {
    InProcessSpawner::from_handler(EchoHandler::new() as Arc<dyn InProcessHandler>)
}

/// A registry hosting an in-process backend beside a counted subprocess fake.
fn registry(subprocess: &Arc<FakeSpawner>) -> Arc<SpawnerRegistry> {
    let registry = Arc::new(SpawnerRegistry::new());
    registry.register_in_process(in_process());
    registry.register(SpawnBackend::Subprocess, as_spawner(Arc::clone(subprocess)));
    registry
}

// --- introspection ----------------------------------------------------------

#[test]
fn spawn_backends_reports_the_registry_sorted_and_the_single_spawner_apart() {
    let runner = Runner::new();
    assert_eq!(
        runner.spawn_backend(),
        SpawnBackend::Unknown,
        "the default UnsupportedSpawner names no backend"
    );
    assert_eq!(runner.spawn_backends(), vec![SpawnBackend::Unknown]);
    assert!(runner.spawner_registry().is_none());

    let in_process = FakeSpawner::new(SpawnBackend::InProcess, "in");
    runner.set_spawner(as_spawner(Arc::clone(&in_process)));
    assert_eq!(runner.spawn_backend(), SpawnBackend::InProcess);
    assert_eq!(
        runner.spawner().backend(),
        SpawnBackend::InProcess,
        "spawner() reflects the installation"
    );
    assert_eq!(runner.spawn_backends(), vec![SpawnBackend::InProcess]);

    let subprocess = FakeSpawner::new(SpawnBackend::Subprocess, "sub");
    runner.set_spawner_registry(registry(&subprocess));
    assert_eq!(
        runner.spawn_backends(),
        vec![SpawnBackend::Subprocess, SpawnBackend::InProcess],
        "the registry's backends are listed sorted"
    );
    assert_eq!(
        runner.spawn_backend(),
        SpawnBackend::InProcess,
        "the single spawner stays installed beside the registry"
    );
    assert!(runner.spawner_registry().is_some());

    // The last installation wins: `set_spawner` drops the registry.
    runner.set_spawner(as_spawner(FakeSpawner::new(
        SpawnBackend::ExternalCommand,
        "cli",
    )));
    assert!(runner.spawner_registry().is_none());
    assert_eq!(runner.spawn_backends(), vec![SpawnBackend::ExternalCommand]);
}

// --- routing ----------------------------------------------------------------

#[tokio::test]
async fn a_spec_backend_routes_to_the_in_process_backend() {
    let subprocess = FakeSpawner::new(SpawnBackend::Subprocess, "sub");
    let runner = Runner::new();
    runner.set_spawner(as_spawner(FakeSpawner::new(
        SpawnBackend::RemoteHttp,
        "http",
    )));
    runner.set_spawner_registry(registry(&subprocess));

    let child = runner
        .spawn_agent(SpawnSpec::default().with_backend(SpawnBackend::InProcess))
        .await
        .expect("the registry hosts an in-process backend");

    assert_eq!(child.backend(), SpawnBackend::InProcess);
    assert!(
        child.id().starts_with("in-process-"),
        "the real in-process backend answered: {}",
        child.id()
    );
    assert_eq!(
        subprocess.calls(),
        0,
        "the sibling backend was not consulted"
    );
    let output = child.send(json!({"query": "routed"})).await.unwrap();
    assert_eq!(output.output, "echo: {\"query\":\"routed\"}");
    assert!(child.shutdown(Duration::from_secs(1)).await.unwrap());
}

#[tokio::test]
async fn a_runner_without_a_registry_never_consults_one() {
    let subprocess = FakeSpawner::new(SpawnBackend::Subprocess, "sub");
    let with_registry = Runner::new();
    with_registry.set_spawner_registry(registry(&subprocess));

    let single = FakeSpawner::new(SpawnBackend::InProcess, "single");
    let without_registry = Runner::new();
    without_registry.set_spawner(as_spawner(Arc::clone(&single)));

    // The spec's hint belongs to the registry's routing; a runner with only a
    // single spawner sends everything there.
    let child = without_registry
        .spawn_agent(SpawnSpec::default().with_backend(SpawnBackend::InProcess))
        .await
        .expect("the single spawner answers");
    assert_eq!(single.calls(), 1);
    assert_eq!(child.id(), "single-0");
    assert_eq!(child.backend(), SpawnBackend::InProcess);

    assert!(
        without_registry.spawner_registry().is_none(),
        "runner state is per instance"
    );
    assert!(with_registry.spawner_registry().is_some());
    assert_eq!(
        subprocess.calls(),
        0,
        "the other runner's registry was never asked to spawn"
    );
}

#[tokio::test]
async fn a_contradicting_spec_backend_is_a_config_error() {
    let single = FakeSpawner::new(SpawnBackend::RemoteHttp, "http");
    let runner = Runner::new();
    runner.set_spawner(as_spawner(Arc::clone(&single)));

    let error = runner
        .spawn_agent(SpawnSpec::default().with_backend(SpawnBackend::Subprocess))
        .await
        .err()
        .expect("the single spawner cannot serve a subprocess hint");
    assert_eq!(error.kind(), ErrorKind::Config);
    let message = error.to_string();
    assert!(
        message.contains("subprocess"),
        "the request is named: {message}"
    );
    assert!(
        message.contains("remote_http"),
        "the installation is named: {message}"
    );
    assert!(
        message.contains("Runner::set_spawner_registry"),
        "the registry remedy is named: {message}"
    );
    assert!(
        message.contains("Runner::spawn_agent_with"),
        "the addressed remedy is named: {message}"
    );
    assert_eq!(single.calls(), 0, "nothing was launched");
    assert!(
        runner.active_spawns().is_empty(),
        "a refused spec admits no run"
    );
}

#[tokio::test]
async fn a_matching_spec_backend_spawns_through_the_single_spawner() {
    let single = FakeSpawner::new(SpawnBackend::RemoteHttp, "http");
    let runner = Runner::new();
    runner.set_spawner(as_spawner(Arc::clone(&single)));

    let child = runner
        .spawn_agent(SpawnSpec::default().with_backend(SpawnBackend::RemoteHttp))
        .await
        .expect("the hint agrees with the installed spawner");
    assert_eq!(single.calls(), 1);
    assert_eq!(child.id(), "http-0");
    assert_eq!(
        runner.active_spawns()[0].detail["backend"],
        json!("remote_http"),
        "the run is labelled with the agreed backend"
    );
}

#[tokio::test]
async fn an_unlabelled_single_spawner_keeps_the_spec_hint() {
    let runner = Runner::new();
    runner.set_spawner(as_spawner(Arc::new(UnlabelledSpawner {
        id_prefix: "unlabelled",
    })));

    let child = runner
        .spawn_agent(SpawnSpec::default().with_backend(SpawnBackend::InProcess))
        .await
        .expect("an unlabelled spawner cannot contradict the hint");
    assert_eq!(child.id(), "unlabelled-0");
    assert_eq!(
        child.backend(),
        SpawnBackend::Unknown,
        "the handle itself names no backend"
    );
    assert_eq!(
        runner.active_spawns()[0].detail["backend"],
        json!("in_process"),
        "the hint labels the run"
    );
}

#[tokio::test]
async fn a_registry_registered_unlabelled_spawner_is_labelled_with_its_key() {
    let runner = Runner::new();
    let registry = Arc::new(SpawnerRegistry::new());
    registry.register(
        SpawnBackend::RemoteHttp,
        as_spawner(Arc::new(UnlabelledSpawner {
            id_prefix: "unlabelled",
        })),
    );
    runner.set_spawner_registry(registry);

    let child = runner
        .spawn_agent(SpawnSpec::default())
        .await
        .expect("the sole registered spawner answers");
    assert_eq!(
        child.backend(),
        SpawnBackend::Unknown,
        "the spawner's self-report is silent"
    );
    assert_eq!(
        runner.active_spawns()[0].detail["backend"],
        json!("remote_http"),
        "the registry key labels the run, not the silent self-report"
    );
}

// --- liveness ---------------------------------------------------------------

#[tokio::test]
async fn a_probe_of_a_dead_child_finishes_the_spawn() {
    let alive = Arc::new(AtomicBool::new(true));
    let runner = Runner::new();
    runner.set_spawner(as_spawner(ScriptedSpawner::new(
        SpawnBackend::InProcess,
        Arc::clone(&alive),
        false,
    )));

    let child = runner
        .spawn_agent(SpawnSpec::default())
        .await
        .expect("the scripted spawner answers");
    let run_id = child.id().to_owned();

    assert!(child.health().await.unwrap(), "a live probe reports health");
    assert_eq!(
        runner.monitor().snapshot(&run_id).unwrap().phase,
        RunPhase::Running,
        "a live probe only observes"
    );

    alive.store(false, Ordering::SeqCst);
    assert!(
        !child.health().await.unwrap(),
        "the probe result is returned unchanged"
    );
    let status = runner.monitor().snapshot(&run_id).unwrap();
    assert_eq!(status.phase, RunPhase::Failed);
    assert_eq!(status.error.as_deref(), Some("spawned child exited"));
    assert_eq!(
        status.detail,
        json!({"reason": "child_exited", "backend": "in_process"})
    );
    assert!(
        runner.active_spawns().is_empty(),
        "a dead child is no longer active"
    );
}

#[tokio::test]
async fn a_failed_call_to_a_live_child_leaves_the_spawn_active() {
    let alive = Arc::new(AtomicBool::new(true));
    let runner = Runner::new();
    runner.set_spawner(as_spawner(ScriptedSpawner::new(
        SpawnBackend::InProcess,
        Arc::clone(&alive),
        true,
    )));

    let child = runner
        .spawn_agent(SpawnSpec::default())
        .await
        .expect("the scripted spawner answers");
    let error = child
        .send(json!({"query": "hi"}))
        .await
        .expect_err("the scripted child fails the call");
    assert_eq!(error.kind(), ErrorKind::Transport);
    assert_eq!(
        runner.active_spawns().len(),
        1,
        "a failure against a live child is not terminal"
    );
    assert_eq!(
        runner.monitor().snapshot(child.id()).unwrap().phase,
        RunPhase::Running
    );
}

#[tokio::test]
async fn a_failed_call_to_a_dead_child_finishes_the_spawn() {
    let alive = Arc::new(AtomicBool::new(true));
    let runner = Runner::new();
    runner.set_spawner(as_spawner(ScriptedSpawner::new(
        SpawnBackend::InProcess,
        Arc::clone(&alive),
        true,
    )));

    let child = runner
        .spawn_agent(SpawnSpec::default())
        .await
        .expect("the scripted spawner answers");
    let run_id = child.id().to_owned();

    alive.store(false, Ordering::SeqCst);
    let error = child
        .send(json!({"query": "hi"}))
        .await
        .expect_err("the scripted child fails the call");
    assert_eq!(error.kind(), ErrorKind::Transport);
    let status = runner.monitor().snapshot(&run_id).unwrap();
    assert_eq!(status.phase, RunPhase::Failed);
    assert_eq!(status.error.as_deref(), Some("spawned child exited"));
    assert!(
        runner.active_spawns().is_empty(),
        "the exited child left the active list"
    );
}

#[tokio::test]
async fn a_stream_error_from_a_dead_child_finishes_the_spawn() {
    let alive = Arc::new(AtomicBool::new(true));
    let runner = Runner::new();
    runner.set_spawner(as_spawner(ScriptedSpawner::new(
        SpawnBackend::InProcess,
        Arc::clone(&alive),
        true,
    )));

    let child = runner
        .spawn_agent(SpawnSpec::default())
        .await
        .expect("the scripted spawner answers");
    let run_id = child.id().to_owned();

    alive.store(false, Ordering::SeqCst);
    let mut frames = child.stream(json!({}));
    let frame = StreamExt::next(&mut frames)
        .await
        .expect("the scripted child yields one error frame");
    assert!(frame.is_err());
    drop(frames);
    let status = runner.monitor().snapshot(&run_id).unwrap();
    assert_eq!(status.phase, RunPhase::Failed);
    assert!(
        runner.active_spawns().is_empty(),
        "an exited child leaves the active list"
    );
}

#[tokio::test]
async fn a_stream_error_from_a_live_child_leaves_the_spawn_active() {
    let alive = Arc::new(AtomicBool::new(true));
    let runner = Runner::new();
    runner.set_spawner(as_spawner(ScriptedSpawner::new(
        SpawnBackend::InProcess,
        Arc::clone(&alive),
        true,
    )));

    let child = runner
        .spawn_agent(SpawnSpec::default())
        .await
        .expect("the scripted spawner answers");
    let mut frames = child.stream(json!({}));
    assert!(StreamExt::next(&mut frames)
        .await
        .expect("one frame")
        .is_err());
    drop(frames);
    assert_eq!(runner.active_spawns().len(), 1);
}

// --- observability ----------------------------------------------------------

#[tokio::test]
async fn a_successful_spawn_is_monitored_and_its_shutdown_clears_it() {
    let subprocess = FakeSpawner::new(SpawnBackend::Subprocess, "sub");
    let runner = Runner::new();
    runner.set_spawner_registry(registry(&subprocess));

    let child = runner
        .spawn_agent(
            SpawnSpec::class_agent("tests.spawn_routing", "echo")
                .with_backend(SpawnBackend::InProcess)
                .with_session_id("s-1"),
        )
        .await
        .expect("the in-process backend spawns");

    let active = runner.active_spawns();
    assert_eq!(active.len(), 1, "a successful spawn is admitted");
    assert_eq!(active[0].run_id, child.id());
    assert_eq!(active[0].kind, RunKind::Spawn);
    assert_eq!(active[0].session_id.as_deref(), Some("s-1"));
    assert_eq!(
        active[0].detail["backend"],
        json!("in_process"),
        "the detail names the resolved backend"
    );

    assert!(child.shutdown(Duration::from_secs(1)).await.unwrap());
    assert!(
        runner.active_spawns().is_empty(),
        "a shut-down spawn is no longer live"
    );
    let finished = runner
        .monitor()
        .snapshot(child.id())
        .expect("the finished run stays readable");
    assert_eq!(
        finished.detail,
        json!({"graceful": true}),
        "the terminal outcome records the graceful exit"
    );
}

#[tokio::test]
async fn dropping_the_spawn_handle_clears_the_active_spawn() {
    let subprocess = FakeSpawner::new(SpawnBackend::Subprocess, "sub");
    let runner = Runner::new();
    runner.set_spawner_registry(registry(&subprocess));

    let child = runner
        .spawn_agent(SpawnSpec::default().with_backend(SpawnBackend::InProcess))
        .await
        .expect("the in-process backend spawns");
    let run_id = child.id().to_owned();
    assert_eq!(runner.active_spawns().len(), 1);

    drop(child);
    assert!(
        runner.active_spawns().is_empty(),
        "dropping the handle finishes the run"
    );
    let finished = runner
        .monitor()
        .snapshot(&run_id)
        .expect("the dropped run stays readable");
    assert_eq!(finished.detail, json!({"reason": "handle_dropped"}));
}

#[tokio::test]
async fn a_failed_spawn_adds_nothing_to_the_active_spawns() {
    let runner = Runner::new();
    runner.set_spawner(Arc::new(FailingSpawner));

    let error = runner
        .spawn_agent(SpawnSpec::default())
        .await
        .err()
        .expect("the spawner fails");
    assert_eq!(error.kind(), ErrorKind::Transport);
    assert_eq!(runner.active_spawns().len(), 0);
    assert_eq!(runner.monitor().list_active().len(), 0);
}

// --- typed failures ---------------------------------------------------------

#[tokio::test]
async fn spawning_with_a_deferred_backend_is_unsupported_and_names_pulsar() {
    let subprocess = FakeSpawner::new(SpawnBackend::Subprocess, "sub");
    let runner = Runner::new();
    runner.set_spawner_registry(registry(&subprocess));

    let error = runner
        .spawn_agent_with(SpawnBackend::RemoteMq, SpawnSpec::default())
        .await
        .err()
        .expect("the MQ transport is deferred");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(
        error.to_string().contains("Pulsar"),
        "the deferral names its transport: {error}"
    );
    assert!(
        runner.active_spawns().is_empty(),
        "a deferred backend spawns nothing"
    );
    assert_eq!(subprocess.calls(), 0, "no backend was consulted");
}

#[tokio::test]
async fn spawning_without_a_registry_is_a_config_error() {
    let runner = Runner::new();
    assert!(runner.spawner_registry().is_none());

    let error = runner
        .spawn_agent_with(SpawnBackend::InProcess, SpawnSpec::default())
        .await
        .err()
        .expect("a registry is required");
    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(
        error.to_string().contains("Runner::set_spawner_registry"),
        "the message names the fix: {error}"
    );
    assert_eq!(runner.active_spawns().len(), 0);
}
