//! The runner as a *host*: several spawn backends side by side, the roster of the
//! children it is currently keeping alive, their health, and the memory-dreaming
//! service the runner drives for itself.
//!
//! This scenario answers what the per-crate runner suites cannot answer from the
//! outside: does a `SpawnerRegistry` installed on a *local* `Runner::new()` really
//! route a `SpawnSpec` to the backend it names, does `Runner::active_spawns`
//! report exactly the children that are still hosted, does a hosted spawn report
//! health through the seam, and does the runner-owned dreaming driver
//! ([`Runner::set_dreaming_service`] / `start_dreaming` / `dreaming_status` /
//! `stop_dreaming`, i.e. [`DreamService`]) run a queued consolidation job beside a
//! live spawn without starving it?
//!
//! Offline and deterministic throughout: the only backend that really launches
//! anything is the in-process one, whose handler is a local echo double; the
//! external-command backend is given a search directory nothing lives in, so its
//! failure is the typed binary-resolution error rather than a child process. No
//! `Runner::global()`, no shared checkpointer, and every await is bounded by
//! [`support::within`].
//!
//! Dreaming has two documented drivers and both are exercised here: the runner's
//! own loop (a single-sweep schedule plus a bounded `eventually` poll over
//! `Runner::dreaming_status`, never a sleep, and no task spawned by this scenario)
//! and the host's own timer ([`DreamService::tick`] with an injected instant,
//! which leaves the loop untouched). An empty slot is a typed absence rather than
//! a silent no-op: the lifecycle calls fail with `ErrorKind::Config` naming
//! `Runner::set_dreaming_service` until a service is installed.
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
    DreamRepository, DreamRunSummary, DreamState, DreamTrigger, InMemoryDreamRepository,
    NoopConsolidator,
};
use openjiuwen::core::memory::MemoryScope;
use openjiuwen::core::runner::{
    DeferredSpawner, DreamService, ExternalCommandSpawner, InProcessHandler, InProcessSpawner,
    RunKind, RunPhase, Runner, SpawnBackend, SpawnSpec, SpawnerRegistry,
};
use openjiuwen::core::single_agent::{AgentChunk, AgentOutput};
use openjiuwen_test_support::eventually;

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
    // An abandoned handle is not a run that finished: silently reporting
    // success for work that never completed is the bug, so a dropped run is
    // `Cancelled` and its detail says why.
    assert_eq!(dropped.phase, RunPhase::Cancelled);
    assert_eq!(dropped.detail, json!({"reason": "handle_dropped"}));
    assert!(
        dropped.finished_at.is_some(),
        "a cancelled run still records its end"
    );
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

// --- the runner-owned dreaming driver ---------------------------------------
//
// The runner hosts a `DreamService` and decides when its loop runs; the loop's
// schedule is the only thing that may tick it in the background, and a host with
// its own timer calls `DreamService::tick` instead. Both shapes are covered below,
// plus the typed absence of a runner with no service installed.

/// The id of the one job every dreaming scenario here queues.
const JOB_ID: &str = "e2e-dream-tick";

/// The schedule the hosted loop runs under: the first sweep is immediate and the
/// next one is a minute out, so exactly one sweep happens inside a test and the
/// summary it left behind can be read back without racing a second (empty) sweep.
fn single_sweep() -> DreamOrchestratorConfig {
    DreamOrchestratorConfig::default()
        .with_initial_delay_ms(0)
        .with_poll_interval_ms(60_000)
        .with_max_interval_ms(60_000)
}

/// A manual consolidation job for [`JOB_ID`], stamped with the process clock.
fn dream_job() -> DreamJob {
    DreamJob::new(
        MemoryScope::new("alice", "workspace"),
        DreamKind::Consolidation,
        DreamTrigger::Manual,
        now_utc(),
    )
    .with_id(JOB_ID)
}

/// The queue behind `service`, shared with the test so jobs can be queued and read
/// back through the same repository the loop sweeps.
fn queue(service: &Arc<DreamService>) -> Arc<dyn DreamRepository> {
    Arc::clone(service.orchestrator().planner().repository())
}

/// Polls the runner's hosted dreaming status until it reports `ticks` orchestrator
/// ticks. A bounded poll, never a sleep used for ordering.
async fn wait_for_ticks(runner: &Runner, ticks: u64) {
    eventually(BOUND, || async {
        runner
            .dreaming_status()
            .is_ok_and(|status| status.orchestrator.ticks >= ticks)
    })
    .await
    .expect("the hosted dreaming loop reaches the requested tick count");
}

/// Asserts the single outcome `summary` recorded for [`JOB_ID`]: the documented
/// `NoopConsolidator` skip, notes included.
fn assert_noop_skip(summary: &DreamRunSummary) {
    assert_eq!(summary.scanned, 1);
    assert_eq!(
        summary.skipped, 1,
        "the default consolidator skips its job instead of failing it"
    );
    assert_eq!(summary.outcomes.len(), 1);
    assert_eq!(summary.outcomes[0].job_id, JOB_ID);
    assert_eq!(summary.outcomes[0].state, DreamState::Skipped);
    assert!(
        summary.outcomes[0]
            .notes
            .iter()
            .any(|note| note.contains("no consolidator configured")),
        "unexpected notes: {:?}",
        summary.outcomes[0].notes
    );
}

#[tokio::test]
async fn a_runner_drives_a_dreaming_service_beside_a_live_spawn() {
    // The dream half: the runner's own service over an in-memory queue, the
    // documented no-op consolidator, and a schedule whose first sweep is immediate.
    let service = DreamService::with_in_memory(
        Arc::new(NoopConsolidator),
        DreamConfig::default(),
        single_sweep(),
    )
    .expect("the single-sweep schedule is valid");
    let repository = queue(&service);
    within("repository.enqueue", repository.enqueue(dream_job()))
        .await
        .expect("the dream job is enqueued");
    assert!(!service.is_running(), "a fresh service owns no loop");

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
    assert!(child.is_alive(), "the spawn is live before the loop starts");

    // The runner owns the loop from here on: this scenario spawns no task.
    assert!(
        runner
            .set_dreaming_service(Some(Arc::clone(&service)))
            .is_none(),
        "the slot starts empty"
    );
    let started = runner.start_dreaming().await.expect("the service starts");
    assert!(started.running, "the loop is live after start");
    assert!(started.last_stop_error.is_none());
    wait_for_ticks(&runner, 1).await;

    // The sweep consumed exactly the queued job, and the no-op consolidator made
    // that a documented skip rather than a failure.
    let summary = runner
        .dreaming_status()
        .expect("the service is installed")
        .orchestrator
        .last_summary
        .expect("the sweep left its summary");
    assert_noop_skip(&summary);
    let stored = within("repository.get", repository.get(JOB_ID))
        .await
        .expect("the queue is readable")
        .expect("the sweep left the job stored");
    assert_eq!(
        stored.state,
        DreamState::Skipped,
        "the sweep persisted the terminal state"
    );
    assert_eq!(stored.attempts, 1, "the job was claimed exactly once");
    assert!(
        stored.finished_at.is_some(),
        "a skip is terminal, so it stamps the end"
    );

    // The hosted spawn survived the loop: the runner's context was not starved.
    assert!(
        child.is_alive(),
        "the hosted child is alive beside the loop"
    );
    assert!(
        within("child.health", child.health())
            .await
            .expect("the probe itself succeeds"),
        "the hosted child is still healthy"
    );
    let active = runner.active_spawns();
    assert_eq!(active.len(), 1, "the dream loop is not an active spawn");
    assert_eq!(active[0].run_id, child_id);

    // Stopping joins the loop and leaves the service installed for a later start.
    let stopped = runner.stop_dreaming().await.expect("the service stops");
    assert!(!stopped.running, "the loop is joined");
    assert!(
        stopped.last_stop_error.is_none(),
        "cancellation is the documented shutdown path"
    );
    assert!(!service.is_running());
    assert_eq!(
        stopped.orchestrator.ticks, 1,
        "the joined loop cannot tick a second time"
    );
    let again = runner.stop_dreaming().await.expect("a second stop is Ok");
    assert!(!again.running, "a second stop is idempotent");

    assert!(within("child.shutdown", child.shutdown(BOUND))
        .await
        .expect("the hosted child stops"));
    assert_eq!(
        runner.active_spawns().len(),
        0,
        "the stopped spawn leaves the roster"
    );
}

#[tokio::test]
async fn the_dreaming_lifecycle_is_a_typed_absence_until_a_service_is_installed() {
    let runner = Runner::new();
    let errors = [
        runner.start_dreaming().await.expect_err("nothing to start"),
        runner.stop_dreaming().await.expect_err("nothing to stop"),
        runner.dreaming_status().expect_err("nothing to report"),
    ];
    for error in errors {
        assert_eq!(error.kind(), ErrorKind::Config, "got {error:?}");
        assert!(
            error.to_string().contains("Runner::set_dreaming_service"),
            "`{error}` should name the fix"
        );
    }
    assert!(runner.dreaming_service().is_none());

    // An installed service makes the very same calls work; the empty queue means
    // every tick it takes is a no-op sweep.
    let service = DreamService::with_in_memory(
        Arc::new(NoopConsolidator),
        DreamConfig::default(),
        single_sweep(),
    )
    .expect("the single-sweep schedule is valid");
    assert!(runner
        .set_dreaming_service(Some(Arc::clone(&service)))
        .is_none());
    assert!(runner
        .dreaming_service()
        .is_some_and(|installed| Arc::ptr_eq(&installed, &service)));
    let started = runner.start_dreaming().await.expect("the service starts");
    assert!(started.running);
    assert!(
        runner
            .dreaming_status()
            .expect("the service is installed")
            .running
    );
    let stopped = runner.stop_dreaming().await.expect("the service stops");
    assert!(!stopped.running);

    // Clearing the slot hands the service back and restores the typed absence.
    let previous = runner
        .set_dreaming_service(None)
        .expect("the installed service is returned");
    assert!(Arc::ptr_eq(&previous, &service));
    assert!(runner.dreaming_service().is_none());
    let error = runner
        .dreaming_status()
        .expect_err("an empty slot is a typed absence");
    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(
        error.to_string().contains("Runner::set_dreaming_service"),
        "`{error}` should name the fix"
    );
}

#[tokio::test]
async fn a_host_driven_tick_through_the_hosted_service_needs_no_loop() {
    // The explicit composition: the host builds the orchestrator — its own queue,
    // planner and schedule — and wraps it, the shape an application with a shared
    // repository uses.
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
        DreamOrchestrator::new(planner, Arc::clone(&repository), single_sweep())
            .expect("the single-sweep schedule is valid"),
    );
    let service = DreamService::new(orchestrator);
    within("repository.enqueue", repository.enqueue(dream_job()))
        .await
        .expect("the dream job is enqueued");

    let runner = Runner::new();
    runner.set_dreaming_service(Some(Arc::clone(&service)));
    let hosted = runner.dreaming_service().expect("the service is installed");
    assert!(
        !hosted.is_running(),
        "a host-driven tick spawns no loop task"
    );

    // The only clock-free assertion in this file: the injected instant is the
    // instant the orchestrator records, and nothing reads the process clock.
    let tick_now = now_utc();
    let summary = within("service.tick", hosted.tick(tick_now))
        .await
        .expect("the tick succeeds");
    assert_noop_skip(&summary);
    let stored = within("repository.get", repository.get(JOB_ID))
        .await
        .expect("the queue is readable")
        .expect("the tick left the job stored");
    assert_eq!(
        stored.state,
        DreamState::Skipped,
        "the tick persisted the terminal state"
    );

    let status = runner.dreaming_status().expect("the service is installed");
    assert!(!status.running, "ticking does not start the loop");
    assert_eq!(status.orchestrator.ticks, 1);
    assert_eq!(status.orchestrator.last_tick_at, Some(tick_now));
    assert!(status.orchestrator.last_error.is_none());
    assert!(status.last_stop_error.is_none());
    assert!(!service.is_running());
}
