//! The runner as a *host*: several spawn backends side by side, the roster of the
//! children it is currently keeping alive, their health, and a background driver
//! sharing the runner's async context with a hosted spawn.
//!
//! This scenario answers what the per-crate runner suites cannot answer from the
//! outside: does a `SpawnerRegistry` installed on a *local* `Runner::new()` really
//! route a `SpawnSpec` to the backend it names, does `Runner::active_spawns`
//! report exactly the children that are still hosted, does a hosted spawn report
//! health through the seam, and can the runner's async context drive an unrelated
//! background loop (memory dreaming) without starving the spawn?
//!
//! Offline and deterministic throughout: the only backend that really launches
//! anything is the in-process one, whose handler is a local echo double; the
//! external-command backend is given a search directory nothing lives in, so its
//! failure is the typed binary-resolution error rather than a child process. No
//! `Runner::global()`, no shared checkpointer, and every await is bounded by
//! [`support::within`].
//!
//! **Known gap** — the runner exposes no dream (or generic background-driver)
//! API: `openjiuwen-runner` has no `dream`/driver entry point at all, and
//! `DreamOrchestrator` explicitly documents that "a caller that wants background
//! consolidation owns the loop". The last scenario therefore composes the two
//! halves at the test level — a `tokio::spawn`ed `DreamOrchestrator::tick` joined
//! under the suite's deadline while the runner hosts a spawn — instead of driving
//! a runner-owned driver. Closing that gap means adding such an API to the runner
//! crate, which this suite must not do.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod support;

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::stream::BoxStream;
use serde_json::{json, Value};

use openjiuwen::core::common::error::{ErrorKind, Result};
use openjiuwen::core::common::utils::id::now_utc;
use openjiuwen::core::memory::dreaming::{
    DreamConfig, DreamJob, DreamKind, DreamOrchestrator, DreamOrchestratorConfig, DreamPlanner,
    DreamRepository, DreamState, DreamTrigger, InMemoryDreamRepository, NoopConsolidator,
};
use openjiuwen::core::memory::MemoryScope;
use openjiuwen::core::runner::{
    DeferredSpawner, ExternalCommandSpawner, InProcessHandler, InProcessSpawner, RunKind, RunPhase,
    Runner, SpawnBackend, SpawnSpec, SpawnerRegistry,
};
use openjiuwen::core::single_agent::{AgentChunk, AgentOutput};

use support::{within, BOUND};

/// The command the external-command backend is configured with. Nothing is ever
/// created under that name, so every spawn through it ends at resolution.
const MISSING_COMMAND: &str = "openjiuwen-e2e-missing-cli";

// --- fixtures ---------------------------------------------------------------

/// The invocation log one spawner's handlers share.
type Seen = Arc<Mutex<Vec<Value>>>;

/// A fresh, empty invocation log.
fn seen() -> Seen {
    Arc::new(Mutex::new(Vec::new()))
}

/// The hosted agent double: reports liveness, echoes its inputs, records them.
struct EchoHandler {
    /// Whether this handler still serves calls.
    running: AtomicBool,
    /// Every invocation this handler answered, in order.
    seen: Seen,
}

impl EchoHandler {
    /// An alive handler sharing `seen` with its siblings.
    fn new(seen: Seen) -> Arc<Self> {
        Arc::new(Self {
            running: AtomicBool::new(true),
            seen,
        })
    }

    /// Whether the handler is still up.
    fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}

#[async_trait]
#[allow(clippy::unused_async_trait_impl)]
impl InProcessHandler for EchoHandler {
    fn is_alive(&self) -> bool {
        self.is_running()
    }

    async fn send(&self, inputs: Value) -> Result<AgentOutput> {
        self.seen
            .lock()
            .expect("the invocation log lock is not poisoned")
            .push(inputs.clone());
        Ok(AgentOutput::answer(format!("echo: {inputs}")))
    }

    fn stream(&self, _inputs: Value) -> BoxStream<'static, Result<AgentChunk>> {
        Box::pin(futures::stream::empty())
    }

    async fn shutdown(&self) -> Result<()> {
        self.running.store(false, Ordering::SeqCst);
        Ok(())
    }
}

/// An in-process spawner building one independent handler per spawn (the factory
/// form, so stopping one child leaves its siblings alone), all logging into
/// `seen`.
fn in_process(seen: Seen) -> InProcessSpawner {
    InProcessSpawner::new(Arc::new(move |_spec: &SpawnSpec| {
        Ok(EchoHandler::new(Arc::clone(&seen)) as Arc<dyn InProcessHandler>)
    }))
}

/// The external-command backend, pointed at a search directory nothing lives in.
fn missing_command(search: &Path) -> ExternalCommandSpawner {
    ExternalCommandSpawner::new(MISSING_COMMAND).with_search_path(search)
}

/// A registry hosting the in-process backend, the unresolvable external command,
/// and the deferred MQ transport.
fn registry(seen: Seen, search: &Path) -> Arc<SpawnerRegistry> {
    let registry = SpawnerRegistry::new();
    registry.register_in_process(in_process(seen));
    registry.register_external_command(missing_command(search));
    registry.register(
        SpawnBackend::RemoteMq,
        Arc::new(DeferredSpawner::for_backend(SpawnBackend::RemoteMq)),
    );
    Arc::new(registry)
}

/// A spec addressed at `backend`.
fn spec(backend: SpawnBackend) -> SpawnSpec {
    SpawnSpec::class_agent("tests.runner_hosting", "HostedAgent").with_backend(backend)
}

/// The run ids of `statuses`, sorted so the assertion never depends on order.
fn run_ids(statuses: &[openjiuwen::core::runner::RunStatus]) -> Vec<String> {
    let mut ids: Vec<String> = statuses.iter().map(|run| run.run_id.clone()).collect();
    ids.sort_unstable();
    ids
}

// --- routing ----------------------------------------------------------------

#[test]
fn the_registry_routes_by_spec_hint_default_and_sole_spawner() {
    let search = tempfile::tempdir().expect("a temp search directory");
    let registry = registry(seen(), search.path());
    assert_eq!(
        registry.backends(),
        vec![
            SpawnBackend::ExternalCommand,
            SpawnBackend::InProcess,
            SpawnBackend::RemoteMq,
        ],
        "backends() reports the registered backends in enum order"
    );

    // A spec hint wins and resolves to the very spawner registered under that key,
    // not merely to a spawner that names the same backend.
    let (backend, routed) = registry.route(&spec(SpawnBackend::InProcess));
    assert_eq!(backend, SpawnBackend::InProcess);
    assert!(Arc::ptr_eq(
        &routed,
        &registry
            .get(SpawnBackend::InProcess)
            .expect("the in-process backend is registered")
    ));
    assert_eq!(
        registry
            .spawner_for(&spec(SpawnBackend::ExternalCommand))
            .backend(),
        SpawnBackend::ExternalCommand
    );

    // A registered deferred transport routes by name to its deferral.
    let (backend, deferred) = registry.route(&spec(SpawnBackend::RemoteMq));
    assert_eq!(backend, SpawnBackend::RemoteMq);
    assert_eq!(deferred.backend(), SpawnBackend::RemoteMq);
    assert_eq!(deferred.describe()["deferred"], json!(true));

    // Two usable backends and no default: the route is ambiguous, never a guess.
    let (backend, unrouted) = registry.route(&SpawnSpec::default());
    assert_eq!(backend, SpawnBackend::Unknown);
    assert_eq!(
        unrouted.describe()["reason"],
        json!("ambiguous spawn backend")
    );

    // An explicit default answers the unhinted spec; a spec hint still beats it.
    assert_eq!(registry.default_backend(), None);
    registry
        .set_default(SpawnBackend::InProcess)
        .expect("a registered backend can be the default");
    assert_eq!(registry.default_backend(), Some(SpawnBackend::InProcess));
    assert_eq!(
        registry.route(&SpawnSpec::default()).0,
        SpawnBackend::InProcess
    );
    assert_eq!(
        registry.route(&spec(SpawnBackend::ExternalCommand)).0,
        SpawnBackend::ExternalCommand,
        "the spec hint still outranks the default"
    );

    // A default the registry cannot honour is refused and changes nothing.
    let error = registry
        .set_default(SpawnBackend::Subprocess)
        .expect_err("subprocess is not registered");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("subprocess"), "{error}");
    assert_eq!(registry.default_backend(), Some(SpawnBackend::InProcess));

    // Removing a backend keeps the routing set consistent.
    registry.unregister(SpawnBackend::ExternalCommand);
    assert_eq!(
        registry.backends(),
        vec![SpawnBackend::InProcess, SpawnBackend::RemoteMq]
    );
    assert_eq!(
        registry.route(&SpawnSpec::default()).0,
        SpawnBackend::InProcess
    );

    // An empty registry names the routing problem instead of picking a backend.
    let empty = SpawnerRegistry::new();
    let (backend, unrouted) = empty.route(&SpawnSpec::default());
    assert_eq!(backend, SpawnBackend::Unknown);
    assert_eq!(
        unrouted.describe()["reason"],
        json!("no default spawn backend")
    );

    // The sole registered spawner is the route when nothing else is available.
    let sole = SpawnerRegistry::new();
    sole.register_in_process(in_process(seen()));
    assert_eq!(sole.route(&SpawnSpec::default()).0, SpawnBackend::InProcess);
}

#[tokio::test]
async fn the_runner_hosts_the_registry_and_spawns_through_the_routed_backend() {
    let search = tempfile::tempdir().expect("a temp search directory");
    let log = seen();
    let runner = Runner::new();
    runner.set_spawner_registry(registry(Arc::clone(&log), search.path()));

    assert!(runner.spawner_registry().is_some());
    assert_eq!(
        runner.spawn_backends(),
        vec![
            SpawnBackend::ExternalCommand,
            SpawnBackend::InProcess,
            SpawnBackend::RemoteMq,
        ]
    );
    assert_eq!(
        runner.spawn_backend(),
        SpawnBackend::Unknown,
        "the registry leaves the default UnsupportedSpawner installed"
    );

    // The hinted spec takes the in-process route end to end.
    let child = within(
        "runner.spawn_agent",
        runner.spawn_agent(spec(SpawnBackend::InProcess).with_session_id("hosting-route")),
    )
    .await
    .expect("the routed backend launches the hosted agent");
    let output = within("child.send", child.send(json!({"query": "routed"})))
        .await
        .expect("the hosted agent answers");
    assert_eq!(output.output, "echo: {\"query\":\"routed\"}");
    assert_eq!(
        log.lock()
            .expect("the invocation log lock is not poisoned")
            .len(),
        1,
        "the routed handler saw the invocation"
    );
    let active = runner.active_spawns();
    assert_eq!(active.len(), 1);
    assert_eq!(active[0].detail["backend"], json!("in_process"));
    assert!(within("child.shutdown", child.shutdown(BOUND))
        .await
        .expect("the hosted agent stops"));
    assert_eq!(
        runner.active_spawns().len(),
        0,
        "a stopped spawn leaves the roster"
    );

    // A deferred transport is a typed Unsupported naming the missing transport.
    let error = runner
        .spawn_agent(spec(SpawnBackend::RemoteMq))
        .await
        .err()
        .expect("the MQ transport is deferred");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.to_string().contains("Pulsar"), "{error}");

    // A backend the registry does not host is refused by name, never rerouted.
    let error = runner
        .spawn_agent_with(SpawnBackend::Subprocess, SpawnSpec::default())
        .await
        .err()
        .expect("no subprocess spawner is registered");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(error.to_string().contains("register_subprocess"), "{error}");
    assert_eq!(
        runner.active_spawns().len(),
        0,
        "a failed spawn creates no active entry"
    );
}

// --- the hosted roster ------------------------------------------------------

#[tokio::test]
async fn active_spawns_track_each_hosted_child_until_its_own_handle_ends() {
    let search = tempfile::tempdir().expect("a temp search directory");
    let runner = Runner::new();
    runner.set_spawner_registry(registry(seen(), search.path()));

    let first = within(
        "first spawn",
        runner.spawn_agent(spec(SpawnBackend::InProcess).with_session_id("session-a")),
    )
    .await
    .expect("the first hosted child launches");
    let first_id = first.id().to_owned();
    let active = runner.active_spawns();
    assert_eq!(active.len(), 1, "one spawn is one active run");
    assert_eq!(active[0].run_id, first_id, "the run is the child's own id");
    assert_eq!(active[0].kind, RunKind::Spawn);
    assert_eq!(active[0].phase, RunPhase::Running);
    assert_eq!(active[0].session_id.as_deref(), Some("session-a"));
    assert_eq!(
        active[0].detail,
        json!({"backend": "in_process", "pid": null}),
        "an in-process child has a backend and no OS process"
    );

    let second = within(
        "second spawn",
        runner.spawn_agent(spec(SpawnBackend::InProcess).with_session_id("session-b")),
    )
    .await
    .expect("the second hosted child launches");
    let second_id = second.id().to_owned();
    assert_ne!(first_id, second_id, "every spawn mints its own handle id");
    assert_eq!(
        run_ids(&runner.active_spawns()),
        {
            let mut expected = vec![first_id.clone(), second_id.clone()];
            expected.sort_unstable();
            expected
        },
        "a second spawn adds a second entry and leaves the first alone"
    );

    // Stopping one handle clears only its own entry.
    assert!(within("second.shutdown", second.shutdown(BOUND))
        .await
        .expect("the second hosted child stops"));
    let remaining = runner.active_spawns();
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0].run_id, first_id, "the sibling is untouched");
    let finished = runner
        .monitor()
        .snapshot(&second_id)
        .expect("the stopped run is retained by the monitor");
    assert_eq!(finished.phase, RunPhase::Completed);
    assert_eq!(finished.detail, json!({"graceful": true}));
    assert!(
        finished.finished_at.is_some(),
        "a completed run records its end"
    );

    // Dropping the last handle ends its run without touching anything else.
    drop(first);
    assert_eq!(
        runner.active_spawns().len(),
        0,
        "dropping the handle ends its run"
    );
    let dropped = runner
        .monitor()
        .snapshot(&first_id)
        .expect("the dropped run is retained by the monitor");
    assert_eq!(dropped.phase, RunPhase::Completed);
    assert_eq!(dropped.detail, json!({"reason": "handle_dropped"}));
}

// --- health -----------------------------------------------------------------

#[tokio::test]
async fn a_hosted_spawn_reports_its_health_and_a_missing_command_fails_typed() {
    let search = tempfile::tempdir().expect("a temp search directory");
    let runner = Runner::new();
    runner.set_spawner_registry(registry(seen(), search.path()));

    let child = within("spawn", runner.spawn_agent(spec(SpawnBackend::InProcess)))
        .await
        .expect("the hosted child launches");
    let healthy = within("child.health", child.health())
        .await
        .expect("the probe itself succeeds");
    assert!(healthy, "a live in-process agent is healthy");
    assert!(child.is_healthy());
    assert!(child.is_alive());
    let snapshot = child.health_snapshot();
    assert_eq!(snapshot.backend, SpawnBackend::InProcess);
    assert!(snapshot.alive && snapshot.healthy);
    assert_eq!(snapshot.pid, None, "an in-process agent has no OS process");
    assert!(snapshot.last_error.is_none());
    let active = runner.active_spawns();
    assert_eq!(active.len(), 1);
    assert_eq!(active[0].phase, RunPhase::Running);
    assert_eq!(
        active[0].detail["healthy"],
        json!(true),
        "the probe is recorded on the run"
    );

    // A spec whose external command cannot resolve fails with the typed
    // binary-resolution error before anything is launched, and creates no run.
    let error = within(
        "external spawn",
        runner.spawn_agent(spec(SpawnBackend::ExternalCommand)),
    )
    .await
    .err()
    .expect("a missing program cannot be launched");
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(error.to_string().contains(MISSING_COMMAND), "{error}");
    assert_eq!(
        std::fs::read_dir(search.path())
            .expect("the search directory is readable")
            .count(),
        0,
        "resolution happens before a child is started"
    );
    assert_eq!(
        runner.active_spawns().len(),
        1,
        "a failed spawn adds no active entry"
    );

    assert!(within("child.shutdown", child.shutdown(BOUND))
        .await
        .expect("the hosted child stops"));
    assert!(!child.is_alive());
    assert!(
        !child.health_snapshot().healthy,
        "a stopped agent is not healthy"
    );
    assert_eq!(
        runner.active_spawns().len(),
        0,
        "the stopped child leaves the roster"
    );
}

// --- a background driver beside the spawn -----------------------------------

#[tokio::test]
async fn a_runner_hosts_a_dream_driver_beside_a_live_spawn() {
    // The dream half: an in-memory queue, the documented no-op consolidator (it
    // skips instead of failing), and the default orchestrator schedule. Nothing
    // here talks to a model or a store.
    let repository: Arc<dyn DreamRepository> = Arc::new(InMemoryDreamRepository::new());
    let planner = Arc::new(
        DreamPlanner::new(
            Arc::clone(&repository),
            Arc::new(NoopConsolidator),
            DreamConfig::default(),
        )
        .expect("the default dream config is valid"),
    );
    let orchestrator = Arc::new(
        DreamOrchestrator::new(
            planner,
            Arc::clone(&repository),
            DreamOrchestratorConfig::default(),
        )
        .expect("the default orchestrator config is valid"),
    );
    let job = DreamJob::new(
        MemoryScope::new("alice", "workspace"),
        DreamKind::Consolidation,
        DreamTrigger::Manual,
        now_utc(),
    )
    .with_id("e2e-dream-tick");
    within("repository.enqueue", repository.enqueue(job))
        .await
        .expect("the dream job is enqueued");

    // The hosting half: a local runner with one live in-process spawn.
    let search = tempfile::tempdir().expect("a temp search directory");
    let runner = Runner::new();
    runner.set_spawner_registry(registry(seen(), search.path()));
    let child = within(
        "spawn",
        runner.spawn_agent(spec(SpawnBackend::InProcess).with_session_id("hosting-dream")),
    )
    .await
    .expect("the hosted child launches");
    let child_id = child.id().to_owned();
    assert!(child.is_alive(), "the spawn is live before the driver runs");

    // No runner-owned dream driver exists, so the driver is composed here: the
    // tick runs on its own task and is joined under the suite's deadline.
    let tick_now = now_utc();
    let driver = {
        let orchestrator = Arc::clone(&orchestrator);
        tokio::spawn(async move { orchestrator.tick(tick_now).await })
    };
    let summary = within("dream tick", driver)
        .await
        .expect("the driver task joins without panicking")
        .expect("the tick succeeds");

    assert_eq!(summary.scanned, 1);
    assert_eq!(
        summary.skipped, 1,
        "the default consolidator skips its job instead of failing it"
    );
    assert_eq!(summary.outcomes.len(), 1);
    assert_eq!(summary.outcomes[0].job_id, "e2e-dream-tick");
    assert_eq!(summary.outcomes[0].state, DreamState::Skipped);
    assert!(
        summary.outcomes[0]
            .notes
            .iter()
            .any(|note| note.contains("no consolidator configured")),
        "unexpected notes: {:?}",
        summary.outcomes[0].notes
    );

    let stored = within("repository.get", repository.get("e2e-dream-tick"))
        .await
        .expect("the queue is readable")
        .expect("the tick left the job stored");
    assert_eq!(
        stored.state,
        DreamState::Skipped,
        "the tick persisted the terminal state"
    );
    let status = orchestrator.status();
    assert_eq!(status.ticks, 1);
    assert_eq!(status.last_tick_at, Some(tick_now));
    assert!(status.last_error.is_none());

    // The hosted spawn survived the driver: the runner's context was not starved.
    assert!(
        child.is_alive(),
        "the hosted child is still alive after the tick"
    );
    assert!(
        within("child.health", child.health())
            .await
            .expect("the probe itself succeeds"),
        "the hosted child is still healthy"
    );
    let active = runner.active_spawns();
    assert_eq!(active.len(), 1);
    assert_eq!(active[0].run_id, child_id);

    assert!(within("child.shutdown", child.shutdown(BOUND))
        .await
        .expect("the hosted child stops"));
    assert_eq!(
        runner.active_spawns().len(),
        0,
        "the hosted driver outlives no spawn"
    );
}
