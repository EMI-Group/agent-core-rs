//! The [`Runner`] facade: configuration, lifecycle, and the six execution
//! entry points.
//!
//! Ports Python `openjiuwen/core/runner/runner.py` (`_RunnerImpl` plus the
//! classmethod facade and `GLOBAL_RUNNER`) with Python's two traps removed: an
//! instance is always explicit ([`Runner::new`]) and only [`Runner::global`]
//! touches the documented process singleton.
//!
//! The facade owns the collaborators — the [`ResourceMgr`] registries, the
//! [`CallbackFramework`], the [`MessageQueue`], the optional [`Checkpointer`],
//! and the two observation seams every run reports into ([`RunMonitor`] and
//! [`InteractionHub`]) — and hands them to the free functions in [`crate::run`];
//! it never reaches into a lower layer itself.
//!
//! The team runtime pool is the one collaborator the facade keeps behind a lock:
//! [`Runner::start`] and [`Runner::stop`] rebuild it from the live configuration,
//! so a [`Runner::set_config`] takes effect on restart and the pool a host
//! addresses through [`Runner::start_agent_team`] is always the pool of the
//! current configuration.
//!
//! Runs do **not** require [`Runner::start`]: the queue is host plumbing, not a
//! precondition for `run_agent`/`run_workflow`/`run_agent_team` (Python's global
//! runner is lazily started for the same reason).
//!
//! The spawn seam is installed, not constructed: [`Runner::set_spawner`] installs
//! one backend and [`Runner::set_spawner_registry`] hosts several with routing
//! between them. Every successful spawn is wrapped in
//! [`crate::spawn::MonitoredSpawnedAgent`], so a host observes the child as a
//! [`crate::monitor::RunKind::Spawn`] status on [`Runner::monitor`] until it is
//! shut down, killed, or dropped. The spawn accessors live in the crate's
//! `runner_spawn` module, beside this file.

use crate::callback::{bind_guardrails, CallbackFramework, CallbackId};
use crate::config::RunnerConfig;
use crate::error;
use crate::hitl::{DeliverResult, InteractPayload, InteractionHub, RunTarget};
use crate::monitor::{RunMonitor, RunStatus};
use crate::mq::{self, MessageQueue};
use crate::pool::{ActiveTeamInfo, PoolKey, PoolRequest, PoolRun, TeamPool};
use crate::resources::ResourceMgr;
use crate::run::{
    invoke_agent, invoke_team, invoke_workflow, stream_agent, stream_team, stream_workflow,
    AgentRef, RunOptions, RunServices, TeamRef, WorkflowRef,
};
use crate::spawn::{AgentSpawner, SpawnerRegistry, UnsupportedSpawner};
use futures::stream::BoxStream;
use openjiuwen_agent::output::{AgentChunk, AgentOutput};
use openjiuwen_common::error::Result;
use openjiuwen_security::{GuardrailContext, GuardrailOutcome, GuardrailPipeline};
use openjiuwen_session::interaction::InteractionResponse;
use openjiuwen_session::Checkpointer;
use openjiuwen_workflow::{WorkflowChunk, WorkflowOutput};
use parking_lot::RwLock;
use serde_json::Value;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

/// `tracing` target every runner facade log uses.
const LOG_TARGET: &str = "openjiuwen_runner::runner";

/// The process-global runner slot (`GLOBAL_RUNNER` in Python).
///
/// This is the crate's one documented piece of global mutable state: the slot is
/// filled on first use by [`Runner::global`] and replaced wholesale by
/// [`Runner::reset_global`]. Everything else is per-instance.
static GLOBAL: OnceLock<RwLock<Option<Arc<Runner>>>> = OnceLock::new();

/// The process-level orchestrator: configuration, resources, callbacks, queue,
/// checkpointer, guardrails, the single spawn backend and its optional
/// multi-backend registry, the run monitor, the human-in-the-loop hub, and the
/// team runtime pool.
///
/// Create an isolated instance with [`Runner::new`] / [`Runner::with_config`];
/// [`Runner::global`] is the Python-compatible singleton for host processes.
///
/// ```
/// use openjiuwen_runner::Runner;
/// use serde_json::json;
///
/// # async fn example() -> openjiuwen_common::error::Result<()> {
/// let runner = Runner::new();
/// runner.start().await?;
/// assert!(runner.is_started());
/// runner.stop().await?;
/// # Ok(())
/// # }
/// ```
// `runner_id` mirrors Python's public attribute name.
#[allow(clippy::struct_field_names)]
pub struct Runner {
    /// The instance identity, fixed at construction (`RunnerConfig::runner_id`).
    runner_id: Box<str>,
    /// The live configuration tree; replaced by [`Runner::set_config`].
    config: RwLock<RunnerConfig>,
    /// The four typed resource registries (shared with hosts).
    resources: Arc<ResourceMgr>,
    /// The callback framework (shared with hosts).
    callbacks: Arc<CallbackFramework>,
    /// The live message queue; rebuilt from the config by [`Runner::start`].
    queue: RwLock<Arc<dyn MessageQueue>>,
    /// The checkpointer built by [`Runner::start`]; `None` before `start`.
    checkpointer: RwLock<Option<Arc<dyn Checkpointer>>>,
    /// Whether [`Runner::start`] completed and [`Runner::stop`] has not.
    started: AtomicBool,
    /// Serializes `start`/`stop` so lifecycle transitions are idempotent.
    lifecycle: tokio::sync::Mutex<()>,
    /// The guardrail pipeline bound to the callback framework, when set.
    guardrails: RwLock<Option<Arc<GuardrailPipeline>>>,
    /// The callback ids [`Runner::set_guardrails`] registered (for rebinding).
    guardrail_ids: RwLock<Vec<CallbackId>>,
    /// The installed single spawn backend ([`UnsupportedSpawner`] by default).
    pub(crate) spawner: RwLock<Arc<dyn AgentSpawner>>,
    /// The installed multi-backend spawn registry; routing takes precedence over
    /// [`Runner::spawner`] while one is installed.
    pub(crate) spawner_registry: RwLock<Option<Arc<SpawnerRegistry>>>,
    /// The run status monitor every execution path mirrors into.
    monitor: Arc<RunMonitor>,
    /// The human-in-the-loop hub every execution path registers on.
    hitl: Arc<InteractionHub>,
    /// The team runtime pool; rebuilt from the live configuration by
    /// [`Runner::start`] and [`Runner::stop`].
    team_pool: RwLock<TeamPool>,
}

impl std::fmt::Debug for Runner {
    /// Prints identity, lifecycle state, and counts — never configuration
    /// secrets, resource contents, or pooled payloads.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Runner")
            .field("runner_id", &self.runner_id())
            .field("started", &self.is_started())
            .field("tools", &self.resources.tool_count())
            .field("agents", &self.resources.agent_count())
            .field("workflows", &self.resources.workflow_count())
            .field("teams", &self.resources.team_count())
            .field("callbacks", &self.callbacks.len())
            .field("checkpointer", &self.checkpointer().is_some())
            .field("guardrails", &self.guardrails().is_some())
            .field("pooled_teams", &self.team_pool().len())
            .field("active_runs", &self.monitor().list_active().len())
            .finish_non_exhaustive()
    }
}

impl Runner {
    /// Builds a runner with [`RunnerConfig::default`] and installs no global.
    ///
    /// The queue is created but **not started**; call [`Runner::start`] when a
    /// host needs it.
    #[must_use]
    pub fn new() -> Arc<Self> {
        Self::with_config(RunnerConfig::default())
    }

    /// Builds an isolated runner from `config`.
    ///
    /// The queue is created from `config.message_queue` in the stopped state;
    /// a configuration naming a deferred transport (`pulsar` / `zeromq`) yields
    /// the typed [`crate::mq::DeferredMessageQueue`] placeholder instead of an
    /// in-memory queue, so [`Runner::message_queue`] never misreports the
    /// backend (its operations are `Error::Unsupported`).
    ///
    /// The checkpointer is not built until [`Runner::start`]. The monitor, the
    /// human-in-the-loop hub, and the team pool are built here (from
    /// `config.team_pool`) and are **open**: pooled team runs and plain runs work
    /// before [`Runner::start`]. No spawn registry is installed: the
    /// [`UnsupportedSpawner`] default routes every spawn until a host installs a
    /// backend with [`Runner::set_spawner`] or [`Runner::set_spawner_registry`].
    ///
    /// `config.distributed_mode` is accepted and preserved, but a distributed
    /// runner cannot start: [`Runner::start`] refuses it until a distributed
    /// transport lands.
    #[must_use]
    pub fn with_config(config: RunnerConfig) -> Arc<Self> {
        let queue: Arc<dyn MessageQueue> = mq::build_message_queue(&config.message_queue);
        let monitor = Arc::new(RunMonitor::new());
        // The hub mirrors its transitions into the same monitor the runs report
        // into, so a host renders one consistent view of both seams.
        let hitl = Arc::new(InteractionHub::new().with_monitor(Arc::clone(&monitor)));
        let team_pool = TeamPool::from_runner_config(&config);
        Arc::new(Self {
            runner_id: config.runner_id.clone().into_boxed_str(),
            config: RwLock::new(config),
            resources: ResourceMgr::new(),
            callbacks: CallbackFramework::new(),
            queue: RwLock::new(queue),
            checkpointer: RwLock::new(None),
            started: AtomicBool::new(false),
            lifecycle: tokio::sync::Mutex::new(()),
            guardrails: RwLock::new(None),
            guardrail_ids: RwLock::new(Vec::new()),
            spawner: RwLock::new(Arc::new(UnsupportedSpawner)),
            spawner_registry: RwLock::new(None),
            monitor,
            hitl,
            team_pool: RwLock::new(team_pool),
        })
    }

    /// Returns the process-global runner, creating it on first use with
    /// [`RunnerConfig::default`].
    ///
    /// This is the Python `GLOBAL_RUNNER` parity entry point. Tests use
    /// [`Runner::new`]; [`Runner::reset_global`] restores a clean singleton.
    pub fn global() -> Arc<Self> {
        let slot = GLOBAL.get_or_init(|| RwLock::new(None));
        if let Some(runner) = slot.read().as_ref() {
            return Arc::clone(runner);
        }
        let mut slot = slot.write();
        Arc::clone(slot.get_or_insert_with(|| Runner::with_config(RunnerConfig::default())))
    }

    /// Installs a freshly built runner in the process-global slot and returns it.
    ///
    /// The previous singleton (if any) is dropped from the slot; callers holding
    /// an `Arc` to it are unaffected. Test-only: production hosts call
    /// [`Runner::global`] once.
    pub fn reset_global() -> Arc<Self> {
        let runner = Runner::with_config(RunnerConfig::default());
        *GLOBAL.get_or_init(|| RwLock::new(None)).write() = Some(Arc::clone(&runner));
        runner
    }

    /// The instance identity (`RunnerConfig::runner_id`, `"global"` by default).
    ///
    /// The id is fixed at construction: [`Runner::set_config`] keeps it even if
    /// the replacement tree names another one, so the instance never changes
    /// identity halfway through its life.
    #[must_use]
    pub fn runner_id(&self) -> &str {
        &self.runner_id
    }

    /// Returns a clone of the live configuration tree.
    #[must_use]
    pub fn config(&self) -> RunnerConfig {
        self.config.read().clone()
    }

    /// Replaces the configuration tree; it takes effect on the next
    /// [`Runner::start`].
    ///
    /// The instance id is preserved (see [`Runner::runner_id`]); a running
    /// queue, checkpointer, or guardrail binding is left untouched until the
    /// runner is stopped and started again.
    pub fn set_config(&self, mut config: RunnerConfig) {
        if config.runner_id != self.runner_id() {
            tracing::debug!(
                target: LOG_TARGET,
                runner_id = %self.runner_id,
                requested = %config.runner_id,
                "ignoring runner_id in set_config; the instance id is fixed"
            );
            self.runner_id().clone_into(&mut config.runner_id);
        }
        *self.config.write() = config;
        tracing::debug!(target: LOG_TARGET, runner_id = %self.runner_id, "runner config updated");
    }

    /// The resource registries (tools, agents, workflows, teams).
    #[must_use]
    pub fn resource_mgr(&self) -> &Arc<ResourceMgr> {
        &self.resources
    }

    /// The callback framework hosts register handlers on.
    #[must_use]
    pub fn callbacks(&self) -> &Arc<CallbackFramework> {
        &self.callbacks
    }

    /// The live message queue.
    ///
    /// Before [`Runner::start`] this is the configured backend in the stopped
    /// state (operations fail with `Error::Config`); after `start` it is the
    /// started queue; after `stop` it is that queue in the stopped state.
    ///
    /// When the configuration names a deferred transport (`pulsar` / `zeromq`)
    /// this is [`crate::mq::DeferredMessageQueue`] — a placeholder that reports
    /// the configured kind and answers `Error::Unsupported` naming the missing
    /// transport from every operation. The backend is **never** silently
    /// downgraded to the in-memory queue.
    #[must_use]
    pub fn message_queue(&self) -> Arc<dyn MessageQueue> {
        Arc::clone(&self.queue.read())
    }

    /// The checkpointer built by [`Runner::start`], when one is configured.
    #[must_use]
    pub fn checkpointer(&self) -> Option<Arc<dyn Checkpointer>> {
        self.checkpointer.read().clone()
    }

    /// The run status monitor every execution path mirrors into.
    ///
    /// The instance is fixed at construction and shared by every run path, the
    /// human-in-the-loop hub, and the team pool, so a host that subscribes here
    /// observes the transitions of every run this runner executes.
    #[must_use]
    pub fn monitor(&self) -> &Arc<RunMonitor> {
        &self.monitor
    }

    /// The human-in-the-loop hub every execution path registers on.
    ///
    /// The instance is fixed at construction and mirrors its transitions into
    /// [`Runner::monitor`]. Use it to list the live runs, pause/resume/cancel a
    /// registered run, and answer the interaction requests a run published; the
    /// facade shortcuts over it are [`Runner::interact_agent_team`] and
    /// [`Runner::answer_interaction`].
    #[must_use]
    pub fn hitl(&self) -> &Arc<InteractionHub> {
        &self.hitl
    }

    /// A cheap clone of the live team runtime pool.
    ///
    /// The pool is `Arc`-backed, so the clone shares the entries, the
    /// concurrency permits, and the reaper with the runner. [`Runner::start`],
    /// [`Runner::stop`], and a restart after [`Runner::set_config`] replace the
    /// pool; a clone taken before a restart keeps addressing the retired pool.
    #[must_use]
    pub fn team_pool(&self) -> TeamPool {
        self.team_pool.read().clone()
    }

    /// Whether [`Runner::start`] completed and [`Runner::stop`] has not.
    #[must_use]
    pub fn is_started(&self) -> bool {
        self.started.load(Ordering::Acquire)
    }

    /// Starts the runner: builds and starts the queue from the live
    /// configuration, builds the configured checkpointer, and rebuilds the team
    /// pool from the live configuration.
    ///
    /// Idempotent: a second call while started is a logged no-op. Starting after
    /// [`Runner::stop`] rebuilds the queue, the checkpointer, and the pool from
    /// the current configuration, so [`Runner::set_config`] takes effect here.
    ///
    /// The spawn seam is not part of the lifecycle: the installed spawner and
    /// registry survive `start`/`stop` and keep routing spawns.
    ///
    /// Rebuilding the pool **retires** the previous one: its live pooled team runs
    /// are stopped (bounded by `TeamPoolConfig::stop_timeout`) and a fresh open
    /// pool takes its place. Pooled runs do not require `start`, so a run admitted
    /// before a `start` is stopped by it — a restart is a hard boundary for pooled
    /// team runs.
    ///
    /// # Errors
    ///
    /// `Error::Unsupported` when the live configuration selects
    /// `distributed_mode` — the distributed transports are deferred, so a
    /// distributed runner refuses to start rather than running in-process under a
    /// distributed name (see [`RunnerConfig::distributed_mode`]) — or when
    /// `create_message_queue` rejects the configured kind (`Error::Unsupported`
    /// for the deferred broker backends) or the checkpointer cannot be built
    /// (`Error::Validation` / `Error::Unsupported`). The runner stays stopped on
    /// any error.
    pub async fn start(&self) -> Result<()> {
        let _lifecycle = self.lifecycle.lock().await;
        if self.is_started() {
            tracing::debug!(target: LOG_TARGET, runner_id = %self.runner_id, "runner already started");
            return Ok(());
        }
        let config = self.config();
        if config.distributed_mode {
            tracing::warn!(
                target: LOG_TARGET,
                runner_id = %self.runner_id,
                "refusing to start a distributed runner; the transports are deferred"
            );
            return Err(error::unsupported(
                "distributed runner mode (`drunner`)",
                "the distributed transports are deferred (Pulsar/ZeroMQ reply-topic \
                 routing, `RemoteProtocol::Mq`/`A2a`, cross-process attach); run \
                 in-process or enable distributed execution once a transport lands",
            ));
        }
        let queue = match mq::create_message_queue(&config.message_queue).await {
            Ok(queue) => queue,
            Err(error) => {
                // A deferred transport is a hard error here, but the placeholder
                // still replaces the slot so `message_queue()` keeps reporting
                // the configured kind after a `set_config`.
                *self.queue.write() = mq::build_message_queue(&config.message_queue);
                return Err(error);
            }
        };
        queue.start().await?;
        let checkpointer = match config.checkpointer.as_ref() {
            Some(checkpointer_config) => {
                Some(openjiuwen_session::create_checkpointer(checkpointer_config).await?)
            }
            None => None,
        };
        self.rebuild_team_pool(&config).await;
        *self.queue.write() = queue;
        *self.checkpointer.write() = checkpointer;
        self.started.store(true, Ordering::Release);
        tracing::info!(target: LOG_TARGET, runner_id = %self.runner_id, "runner started");
        Ok(())
    }

    /// Stops the runner: stops every pooled team run, stops the queue (bounded by
    /// [`RunnerConfig::shutdown_timeout`]), drops the checkpointer, and releases
    /// every registry entry (Python `Runner.stop`).
    ///
    /// The pool is swept **before** the started check, so a runner that never
    /// started (but admitted pooled runs, which do not require `start`) does not
    /// leak them. A fresh open pool built from the live configuration replaces the
    /// swept one, so a later [`Runner::start_agent_team`] — or a restart — can
    /// admit runs again.
    ///
    /// Idempotent; stopping a runner that was never started is `Ok(())`. A queue
    /// that does not stop within the grace period is logged and left to its own
    /// shutdown rather than failing the whole stop; a pool sweep failure is
    /// reported through `tracing` only, so the queue's result stays the return
    /// value.
    ///
    /// # Errors
    ///
    /// Returns the queue's error when it fails to stop.
    pub async fn stop(&self) -> Result<()> {
        let _lifecycle = self.lifecycle.lock().await;
        self.sweep_team_pool().await;
        if !self.started.swap(false, Ordering::AcqRel) {
            tracing::debug!(target: LOG_TARGET, runner_id = %self.runner_id, "runner already stopped");
            return Ok(());
        }
        let queue = Arc::clone(&self.queue.read());
        let timeout = self.config().shutdown_timeout();
        let stopped = if timeout.is_zero() {
            queue.stop().await
        } else if let Ok(result) = tokio::time::timeout(timeout, queue.stop()).await {
            result
        } else {
            tracing::warn!(
                target: LOG_TARGET,
                runner_id = %self.runner_id,
                timeout_ms = timeout.as_millis(),
                "message queue did not stop within the shutdown grace period"
            );
            Ok(())
        };
        *self.checkpointer.write() = None;
        self.resources.release();
        tracing::info!(target: LOG_TARGET, runner_id = %self.runner_id, "runner stopped");
        stopped
    }

    /// Releases the state stored for `session_id` in the runner's checkpointer.
    ///
    /// Python `Runner.release`: a no-op when no checkpointer is configured or
    /// the runner has not been started.
    ///
    /// # Errors
    ///
    /// Returns the checkpointer's error when the release fails.
    pub async fn release(&self, session_id: &str) -> Result<()> {
        if let Some(checkpointer) = self.checkpointer() {
            checkpointer.release(session_id).await
        } else {
            tracing::debug!(
                target: LOG_TARGET,
                runner_id = %self.runner_id,
                session_id,
                "release without a checkpointer is a no-op"
            );
            Ok(())
        }
    }

    // ---------------------------------------------------------- execution --

    /// Runs an agent (registry id or live instance) and awaits its output.
    ///
    /// Delegates to [`crate::run::invoke_agent`]; see [`crate::run`] for session
    /// resolution, callbacks, cancellation, and teardown. Does not require
    /// [`Runner::start`].
    ///
    /// # Errors
    ///
    /// Returns `Error::NotFound` for an unregistered id, the abort error of a
    /// blocking callback, `Error::Cancelled`, or `Error::Timeout`.
    pub async fn run_agent(
        &self,
        agent: impl Into<AgentRef>,
        inputs: Value,
        options: RunOptions,
    ) -> Result<AgentOutput> {
        invoke_agent(self.services(), agent.into(), inputs, options).await
    }

    /// Runs an agent and yields its frames as they arrive.
    ///
    /// Delegates to [`crate::run::stream_agent`].
    pub fn run_agent_streaming(
        &self,
        agent: impl Into<AgentRef>,
        inputs: Value,
        options: RunOptions,
    ) -> BoxStream<'static, Result<AgentChunk>> {
        stream_agent(self.services(), agent.into(), inputs, options)
    }

    /// Runs a workflow (registry id or live instance) and awaits its output.
    ///
    /// Delegates to [`crate::run::invoke_workflow`].
    ///
    /// # Errors
    ///
    /// Returns `Error::NotFound` for an unregistered id, or the workflow
    /// engine's error.
    pub async fn run_workflow(
        &self,
        workflow: impl Into<WorkflowRef>,
        inputs: Value,
        options: RunOptions,
    ) -> Result<WorkflowOutput> {
        invoke_workflow(self.services(), workflow.into(), inputs, options).await
    }

    /// Runs a workflow and yields its frames as they arrive.
    ///
    /// Delegates to [`crate::run::stream_workflow`].
    pub fn run_workflow_streaming(
        &self,
        workflow: impl Into<WorkflowRef>,
        inputs: Value,
        options: RunOptions,
    ) -> BoxStream<'static, Result<WorkflowChunk>> {
        stream_workflow(self.services(), workflow.into(), inputs, options)
    }

    /// Runs an agent team (registry id or live instance) and awaits its result.
    ///
    /// Delegates to [`crate::run::invoke_team`]; the result value is the team's
    /// own wire shape.
    ///
    /// # Errors
    ///
    /// Returns `Error::NotFound` for an unregistered id, `Error::Cancelled` when
    /// the run's cancellation token fires, `Error::Timeout` when the run deadline
    /// expires, or the team's error.
    pub async fn run_agent_team(
        &self,
        team: impl Into<TeamRef>,
        inputs: Value,
        options: RunOptions,
    ) -> Result<Value> {
        invoke_team(self.services(), team.into(), inputs, options).await
    }

    /// Runs an agent team and yields its frames as they arrive.
    ///
    /// Delegates to [`crate::run::stream_team`].
    pub fn run_agent_team_streaming(
        &self,
        team: impl Into<TeamRef>,
        inputs: Value,
        options: RunOptions,
    ) -> BoxStream<'static, Result<Value>> {
        stream_team(self.services(), team.into(), inputs, options)
    }

    // ------------------------------------------------------------- team pool --

    /// Starts a pooled team run and hands back its handle.
    ///
    /// The pool-backed counterpart of [`Runner::run_agent_team`], which stays the
    /// direct (non-pooled) entry point: this call admits the run through
    /// [`crate::pool::TeamPool::start`], which keys it by
    /// `(team name, session id)`, bounds how many runs are in flight, owns the
    /// monitor and hub registration for it, and returns before the worker's first
    /// poll. The handle can be awaited ([`PoolRun::wait`]), subscribed to, paused,
    /// or cancelled; [`Runner::list_active_teams`] lists the live ones.
    ///
    /// The session id is derived by the pool from `inputs["conversation_id"]`,
    /// then [`RunOptions::session_id`], then
    /// [`RunnerConfig::default_session_id`](crate::config::RunnerConfig), and is
    /// written back into the options the run sees.
    ///
    /// # Errors
    ///
    /// `Error::Validation` when the key already has a live run
    /// ([`crate::pool::PoolError::Busy`]), `Error::Config` when the pool was
    /// closed or the call is made outside a tokio runtime, and
    /// `Error::Cancelled` when the run was cancelled while queued.
    pub async fn start_agent_team(
        &self,
        team: impl Into<TeamRef>,
        inputs: Value,
        options: RunOptions,
    ) -> Result<PoolRun> {
        let request = PoolRequest::new(team.into(), inputs).with_options(options);
        self.team_pool().start(self.services(), request).await
    }

    /// Lists the pooled team runs that are not finished, ordered by
    /// `(team_name, session_id)`.
    #[must_use]
    pub fn list_active_teams(&self) -> Vec<ActiveTeamInfo> {
        self.team_pool().list_active()
    }

    /// Stops the pooled team run bound to `(team_name, session_id)`.
    ///
    /// Cancels the run, finalises it as cancelled, removes it from the pool, and
    /// awaits its worker for `TeamPoolConfig::stop_timeout`. An unknown key is
    /// `Ok(false)` rather than an error, so a host can call it unconditionally.
    ///
    /// # Errors
    ///
    /// None today (every internal failure is logged); see
    /// [`crate::pool::TeamPool::stop`].
    pub async fn stop_agent_team(&self, team_name: &str, session_id: &str) -> Result<bool> {
        let key = PoolKey::new(team_name, session_id);
        self.team_pool().stop(&key).await
    }

    /// Pauses the pooled team run bound to `(team_name, session_id)`.
    ///
    /// A coordination signal, not a scheduler: it closes the run's interaction
    /// gate and reports the run as paused, but does not suspend the team loop
    /// (Python's live `InteractGate` wake-up is deferred).
    ///
    /// Kept `async` so the facade's team controls share one shape and a future
    /// gate-aware pause can await without an API change.
    ///
    /// # Errors
    ///
    /// `Error::NotFound` when no live run is pooled under the key.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)] // deliberate: see the doc comment
    pub async fn pause_agent_team(&self, team_name: &str, session_id: &str) -> Result<()> {
        self.team_pool().pause(&PoolKey::new(team_name, session_id))
    }

    /// Resumes a paused pooled team run bound to `(team_name, session_id)`.
    ///
    /// A coordination signal; see [`Runner::pause_agent_team`].
    ///
    /// # Errors
    ///
    /// `Error::NotFound` when no live run is pooled under the key.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)] // deliberate: see `pause_agent_team`
    pub async fn resume_agent_team(&self, team_name: &str, session_id: &str) -> Result<()> {
        self.team_pool()
            .resume(&PoolKey::new(team_name, session_id))
    }

    /// Delivers an inbound human payload to the run addressed by
    /// `(team_name, session_id)`.
    ///
    /// Delegates to [`InteractionHub::deliver`] with a target that sets both
    /// fields, so the payload reaches the one run carrying that pair and never
    /// broadcasts. A pooled run buffers the payload for the host to replay (the
    /// pool's inbound channel); a direct run receives it on its own inbox.
    #[must_use]
    pub fn interact_agent_team(
        &self,
        team_name: &str,
        session_id: &str,
        payload: InteractPayload,
    ) -> DeliverResult {
        let target = RunTarget {
            team_name: Some(team_name.to_owned()),
            session_id: Some(session_id.to_owned()),
        };
        self.hitl.deliver(target, payload)
    }

    /// Answers the interaction request published by a live run.
    ///
    /// Delegates to [`InteractionHub::answer`], which routes the response to the
    /// run's session by request id.
    ///
    /// # Errors
    ///
    /// `Error::NotFound` when no active run has that request id pending.
    pub async fn answer_interaction(&self, response: InteractionResponse) -> Result<()> {
        self.hitl.answer(response).await
    }

    /// The monitor snapshot of the live (or retained finished) team run addressed
    /// by `(team_name, session_id)`, when this runner knows it.
    ///
    /// Correlation is best effort and ordered: a retained pool entry for the key
    /// first (a pooled run, live or recently finished), then a live hub
    /// registration carrying that team name and session (a direct
    /// [`Runner::run_agent_team`] invocation while it is in flight). `None` means
    /// neither seam knows the pair — never an error, so a dashboard can poll it.
    #[must_use]
    pub fn agent_team_status(&self, team_name: &str, session_id: &str) -> Option<RunStatus> {
        let key = PoolKey::new(team_name, session_id);
        if let Some(info) = self.team_pool().get(&key) {
            if let Ok(status) = self.monitor.snapshot(&info.run_id) {
                return Some(status);
            }
        }
        let target = RunTarget {
            team_name: Some(team_name.to_owned()),
            session_id: Some(session_id.to_owned()),
        };
        let run_id = self
            .hitl
            .list_active()
            .into_iter()
            .find(|info| target.matches(info))
            .map(|info| info.run_id)?;
        self.monitor.snapshot(&run_id).ok()
    }

    // ----------------------------------------------------------- security --

    /// Installs a guardrail pipeline and binds it to the callback events.
    ///
    /// Returns the four [`CallbackId`]s [`bind_guardrails`] registered (model
    /// and tool input/output stages, priority 100, namespace `guardrail`). A
    /// previous pipeline's bindings are unregistered first, so calling this
    /// twice replaces the binding instead of double-scanning.
    pub fn set_guardrails(&self, pipeline: Arc<GuardrailPipeline>) -> Vec<CallbackId> {
        let previous = std::mem::take(&mut *self.guardrail_ids.write());
        for id in previous {
            self.callbacks.unregister(id);
        }
        *self.guardrails.write() = Some(Arc::clone(&pipeline));
        let ids = bind_guardrails(&self.callbacks, pipeline);
        self.guardrail_ids.write().clone_from(&ids);
        ids
    }

    /// The installed guardrail pipeline, when one was set.
    #[must_use]
    pub fn guardrails(&self) -> Option<Arc<GuardrailPipeline>> {
        self.guardrails.read().clone()
    }

    /// Scans `ctx` with the installed pipeline and returns its aggregated
    /// outcome.
    ///
    /// The outcome is returned **as-is**: callers decide whether to continue by
    /// calling [`GuardrailOutcome::enforce`] (a blocked outcome yields
    /// `Error::Security`). This is the user-input scan `bind_guardrails` cannot
    /// subscribe to (it has no preset event).
    ///
    /// # Errors
    ///
    /// Returns `Error::Config` when no pipeline is installed.
    pub async fn scan(&self, ctx: GuardrailContext) -> Result<GuardrailOutcome> {
        let pipeline = self
            .guardrails()
            .ok_or_else(|| error::not_started("guardrail pipeline"))?;
        Ok(pipeline.run(&ctx).await)
    }

    // ----------------------------------------------------------- internals --

    /// Builds the per-run execution context handed to the [`crate::run`] free
    /// functions.
    ///
    /// The two observation seams are always attached, so every run the facade
    /// starts — direct or streamed, agent, workflow, or team — is listed by
    /// [`Runner::monitor`] and registered on [`Runner::hitl`].
    fn services(&self) -> RunServices {
        let mut services = RunServices::new(
            Arc::clone(&self.resources),
            Arc::clone(&self.callbacks),
            self.config(),
            self.checkpointer(),
        )
        .with_monitor(Arc::clone(&self.monitor))
        .with_hitl(Arc::clone(&self.hitl));
        if let Some(pipeline) = self.guardrails() {
            services = services.with_guardrails(pipeline);
        }
        services
    }

    /// Replaces the pool with one built from `config` and stops the previous one.
    ///
    /// The previous pool is swapped out before it is swept, so the write lock is
    /// never held across the sweep's `.await`.
    async fn rebuild_team_pool(&self, config: &RunnerConfig) {
        let previous = {
            let mut slot = self.team_pool.write();
            std::mem::replace(&mut *slot, TeamPool::from_runner_config(config))
        };
        sweep_pool(&previous, self.runner_id()).await;
        tracing::debug!(
            target: LOG_TARGET,
            runner_id = %self.runner_id,
            capacity = self.team_pool().capacity(),
            "team pool rebuilt from the live configuration"
        );
    }

    /// Stops every pooled team run and installs a fresh open pool.
    ///
    /// The logs are the only report: [`Runner::stop`]'s return value stays the
    /// queue's, exactly as before the pool existed.
    async fn sweep_team_pool(&self) {
        let previous = self.team_pool();
        sweep_pool(&previous, self.runner_id()).await;
        *self.team_pool.write() = TeamPool::from_runner_config(&self.config());
    }
}

/// Stops every run of `pool` (awaited outside any lock) and logs the sweep.
async fn sweep_pool(pool: &TeamPool, runner_id: &str) {
    let shutdown = pool.stop_all().await;
    let stopped = shutdown.stopped;
    if let Some(error) = shutdown.first_error {
        tracing::warn!(
            target: LOG_TARGET,
            runner_id,
            %error,
            "stopping a pooled team run failed"
        );
    }
    if stopped > 0 {
        tracing::debug!(
            target: LOG_TARGET,
            runner_id,
            stopped,
            "stopped pooled team runs"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_uses_documented_defaults() {
        let runner = Runner::new();
        assert_eq!(runner.runner_id(), "global");
        assert!(!runner.is_started());
        assert!(runner.guardrails().is_none());
        assert!(runner.checkpointer().is_none());
        assert!(runner.resource_mgr().is_empty());
        assert!(runner.callbacks().is_empty());
        // The observation seams and the pool exist from construction, open and
        // empty, so runs do not require `start`.
        assert_eq!(runner.monitor().list_active(), Vec::new());
        assert!(runner.hitl().is_empty());
        let Some(hub_monitor) = runner.hitl().monitor() else {
            panic!("the hub mirrors its transitions into a monitor");
        };
        assert!(Arc::ptr_eq(&hub_monitor, runner.monitor()));
        assert_eq!(runner.list_active_teams(), Vec::new());
        assert_eq!(runner.team_pool().capacity(), 8);
        assert!(!runner.team_pool().is_closed());
        assert!(runner.agent_team_status("crew", "s-1").is_none());
        let debug = format!("{runner:?}");
        assert!(debug.starts_with("Runner {"));
        assert!(debug.contains("runner_id: \"global\""));
        assert!(debug.contains("started: false"));
        assert!(debug.contains("guardrails: false"));
        assert!(debug.contains("pooled_teams: 0"));
        assert!(debug.contains("active_runs: 0"));
    }

    #[test]
    fn runner_id_is_fixed_at_construction() {
        let runner = Runner::with_config(RunnerConfig {
            runner_id: "custom".to_owned(),
            ..RunnerConfig::default()
        });
        assert_eq!(runner.runner_id(), "custom");
        runner.set_config(RunnerConfig {
            runner_id: "other".to_owned(),
            ..RunnerConfig::default()
        });
        assert_eq!(runner.runner_id(), "custom");
        assert_eq!(runner.config().runner_id, "custom");
    }
}
