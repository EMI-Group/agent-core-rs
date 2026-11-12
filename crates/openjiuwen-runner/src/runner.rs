//! The [`Runner`] facade: configuration, lifecycle, and the six execution
//! entry points.
//!
//! Ports Python `openjiuwen/core/runner/runner.py` (`_RunnerImpl` plus the
//! classmethod facade and `GLOBAL_RUNNER`) with Python's two traps removed: an
//! instance is always explicit ([`Runner::new`]) and only [`Runner::global`]
//! touches the documented process singleton.
//!
//! The facade owns the four collaborators — the [`ResourceMgr`] registries, the
//! [`CallbackFramework`], the [`MessageQueue`], and the optional
//! [`Checkpointer`] — and hands them to the free functions in [`crate::run`];
//! it never reaches into a lower layer itself.
//!
//! Runs do **not** require [`Runner::start`]: the queue is host plumbing, not a
//! precondition for `run_agent`/`run_workflow`/`run_agent_team` (Python's global
//! runner is lazily started for the same reason).

use crate::callback::{bind_guardrails, CallbackFramework, CallbackId};
use crate::config::RunnerConfig;
use crate::error;
use crate::mq::{self, InMemoryMessageQueue, MessageQueue};
use crate::resources::ResourceMgr;
use crate::run::{
    invoke_agent, invoke_team, invoke_workflow, stream_agent, stream_team, stream_workflow,
    AgentRef, RunOptions, RunServices, TeamRef, WorkflowRef,
};
use crate::spawn::{AgentSpawner, SpawnSpec, SpawnedAgent, UnsupportedSpawner};
use futures::stream::BoxStream;
use openjiuwen_agent::output::{AgentChunk, AgentOutput};
use openjiuwen_common::error::Result;
use openjiuwen_security::{GuardrailContext, GuardrailOutcome, GuardrailPipeline};
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
/// checkpointer, guardrails, and the spawn seam.
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
    /// The installed subprocess backend ([`UnsupportedSpawner`] by default).
    spawner: RwLock<Arc<dyn AgentSpawner>>,
}

impl std::fmt::Debug for Runner {
    /// Prints identity, lifecycle state, and counts — never configuration
    /// secrets or resource contents.
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
    /// the checkpointer is not built until [`Runner::start`].
    #[must_use]
    pub fn with_config(config: RunnerConfig) -> Arc<Self> {
        let queue: Arc<dyn MessageQueue> = InMemoryMessageQueue::new(config.message_queue.clone());
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
    #[must_use]
    pub fn message_queue(&self) -> Arc<dyn MessageQueue> {
        Arc::clone(&self.queue.read())
    }

    /// The checkpointer built by [`Runner::start`], when one is configured.
    #[must_use]
    pub fn checkpointer(&self) -> Option<Arc<dyn Checkpointer>> {
        self.checkpointer.read().clone()
    }

    /// Whether [`Runner::start`] completed and [`Runner::stop`] has not.
    #[must_use]
    pub fn is_started(&self) -> bool {
        self.started.load(Ordering::Acquire)
    }

    /// Starts the runner: builds and starts the queue from the live
    /// configuration and builds the configured checkpointer.
    ///
    /// Idempotent: a second call while started is a logged no-op. Starting after
    /// [`Runner::stop`] rebuilds the queue from the current configuration.
    ///
    /// # Errors
    ///
    /// Returns the backend's error when `create_message_queue` rejects the
    /// configured kind (`Error::Unsupported` for the deferred broker backends)
    /// or the checkpointer cannot be built (`Error::Validation` /
    /// `Error::Unsupported`). The runner stays stopped on any error.
    pub async fn start(&self) -> Result<()> {
        let _lifecycle = self.lifecycle.lock().await;
        if self.is_started() {
            tracing::debug!(target: LOG_TARGET, runner_id = %self.runner_id, "runner already started");
            return Ok(());
        }
        let config = self.config();
        let queue = mq::create_message_queue(&config.message_queue).await?;
        queue.start().await?;
        let checkpointer = match config.checkpointer.as_ref() {
            Some(checkpointer_config) => {
                Some(openjiuwen_session::create_checkpointer(checkpointer_config).await?)
            }
            None => None,
        };
        *self.queue.write() = queue;
        *self.checkpointer.write() = checkpointer;
        self.started.store(true, Ordering::Release);
        tracing::info!(target: LOG_TARGET, runner_id = %self.runner_id, "runner started");
        Ok(())
    }

    /// Stops the runner: stops the queue (bounded by
    /// [`RunnerConfig::shutdown_timeout`]), drops the checkpointer, and releases
    /// every registry entry (Python `Runner.stop`).
    ///
    /// Idempotent; stopping a runner that was never started is `Ok(())`. A queue
    /// that does not stop within the grace period is logged and left to its own
    /// shutdown rather than failing the whole stop.
    ///
    /// # Errors
    ///
    /// Returns the queue's error when it fails to stop.
    pub async fn stop(&self) -> Result<()> {
        let _lifecycle = self.lifecycle.lock().await;
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
    /// Returns `Error::NotFound` for an unregistered id, or the team's error.
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

    // -------------------------------------------------------------- spawn --

    /// Installs the subprocess backend [`Runner::spawn_agent`] delegates to.
    pub fn set_spawner(&self, spawner: Arc<dyn AgentSpawner>) {
        *self.spawner.write() = spawner;
    }

    /// Asks the installed spawner to launch the agent described by `spec`.
    ///
    /// The default [`UnsupportedSpawner`] answers `Error::Unsupported` naming
    /// the seam.
    ///
    /// # Errors
    ///
    /// Returns the spawner's error (by default `Error::Unsupported`).
    pub async fn spawn_agent(&self, spec: SpawnSpec) -> Result<Arc<dyn SpawnedAgent>> {
        let spawner = Arc::clone(&self.spawner.read());
        spawner.spawn(spec).await
    }

    // ----------------------------------------------------------- internals --

    /// Builds the per-run execution context handed to the [`crate::run`] free
    /// functions.
    fn services(&self) -> RunServices {
        let mut services = RunServices::new(
            Arc::clone(&self.resources),
            Arc::clone(&self.callbacks),
            self.config(),
            self.checkpointer(),
        );
        if let Some(pipeline) = self.guardrails() {
            services = services.with_guardrails(pipeline);
        }
        services
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
        let debug = format!("{runner:?}");
        assert!(debug.starts_with("Runner {"));
        assert!(debug.contains("runner_id: \"global\""));
        assert!(debug.contains("started: false"));
        assert!(debug.contains("guardrails: false"));
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
