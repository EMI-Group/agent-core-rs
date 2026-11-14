//! The controller loop: event queue, intent routing, task dispatch, and outputs.
//!
//! Ports `openjiuwen/core/controller/base.py` together with
//! `modules/event_queue.py` and `modules/event_handler.py` as a single-turn,
//! in-process loop:
//!
//! 1. queue the inbound [`Event`] on the bounded FIFO
//!    [`event_queue_size`](ControllerConfig::event_queue_size) queue and drain
//!    it,
//! 2. recognize intents (the [`EventHandler`] seam, the [`IntentRecognizer`],
//!    or the built-in default intent),
//! 3. plan tasks with the [`Planner`], routing their task type through the
//!    optional [`TaskRouter`],
//! 4. dispatch the tasks to the registered [`TaskExecutor`]s, and
//! 5. aggregate the runs into one [`ControllerOutput`].
//!
//! Task state lives in the shared [`TaskManager`] and survives between turns in
//! the session under [`TASKS_STATE_KEY`], so a later controller instance
//! restores it with [`BaseController::load_state`]. One `invoke` processes
//! exactly the events queued when it started and dispatches its batch
//! synchronously; [`BaseController::start`] additionally runs the
//! [`TaskSchedulerLoop`] in the background, so work added straight to the
//! manager (by a reasoner, a host, or a restored session) runs without another
//! turn.
//!
//! The controller is also the kernel's seam for the surrounding controller
//! stack: [`BaseController::pause_task`] / [`BaseController::cancel_task`] and
//! the [`TaskControl`] impl behind the reasoner, the additive
//! [`BaseController::reasoner`] / [`BaseController::apply_task_ops`] path, and
//! the group routing of [`BaseController::send_to_agent`] /
//! [`BaseController::publish`].
//!
//! A host that only uses the public controller API reaches those seams without
//! crate internals: [`BaseController::task_manager`] is the shared store,
//! [`BaseController::scheduler_state`] / [`BaseController::wake_scheduler`] /
//! [`BaseController::running_tasks`] are the background scheduler's view, and
//! [`BaseController::with_group`] / [`BaseController::group_router`] wire and
//! read the group router. The per-conversation
//! [`TaskQueue`](super::queue::TaskQueue) is legacy-scoped: this controller
//! dispatches each turn synchronously and tracks no per-conversation turn, so
//! the queue lives only on the legacy wrapper.
//!
//! The turn pipeline lives in `turn` (event processing, intent routing, task
//! dispatch, and aggregation) and the task persistence in `state`; this module
//! holds the controller surface, its shared state, and the plain wiring.
mod state;
mod turn;

#[cfg(test)]
mod fixtures;
#[cfg(test)]
mod tests;

use super::config::ControllerConfig;
use super::executors::{
    SessionProvider, TaskContext, TaskExecutor, TaskExecutorRegistry, TaskRouter, TaskScheduler,
    TaskSchedulerLoop,
};
use super::group::{require_group, GroupMethod, GroupRouter};
use super::intent::{IntentRecognizer, Planner, RuleIntentRecognizer};
use super::reasoner::{Reasoner, TaskControl, TaskOp, TaskOpOutcome};
use super::schema::{ControllerOutput, ControllerOutputChunk, Event, Intent, TaskStatus};
use super::task_manager::TaskManager;
use crate::card::AgentCard;
use async_trait::async_trait;
use futures::stream::BoxStream;
use openjiuwen_common::error::Result;
use openjiuwen_session::Session;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, warn};

/// Key the controller task state is persisted under.
pub const TASKS_STATE_KEY: &str = "__controller_tasks__";

/// Host-facing state of a controller's background task scheduler.
///
/// The controller keeps two facts a host would otherwise have to combine: the
/// running flag ([`BaseController::is_running`]) and the loop handle
/// ([`BaseController::scheduler_loop`]). This state derives both into one value,
/// so a host can tell "not started" from "live" from "winding down" without
/// guessing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskSchedulerState {
    /// The controller is not marked running: it was never started, or it was
    /// stopped.
    Stopped,
    /// The controller is marked running and a live [`TaskSchedulerLoop`] backs
    /// it.
    Running,
    /// The controller is marked running but no live loop backs it: the loop was
    /// requested outside a `tokio` runtime, or it is winding down.
    Stopping,
}

impl TaskSchedulerState {
    /// Returns the wire name: `"stopped"`, `"running"`, or `"stopping"`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stopped => "stopped",
            Self::Running => "running",
            Self::Stopping => "stopping",
        }
    }
}

impl std::fmt::Display for TaskSchedulerState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// The controller surface: an event in, a controller output out.
#[async_trait]
pub trait Controller: Send + Sync {
    /// Returns the controller's card.
    fn card(&self) -> &AgentCard;

    /// Returns a snapshot of the controller's configuration.
    fn config(&self) -> ControllerConfig;

    /// Processes one turn and returns its aggregated output.
    ///
    /// # Errors
    ///
    /// Returns a validation error when `inputs` is not a controller input
    /// shape, and propagates recognizer, handler, and router failures.
    async fn invoke(
        &self,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> Result<ControllerOutput>;

    /// Streams one turn as controller output chunks.
    ///
    /// The first frame is always the `processing` sentinel; the last frame has
    /// [`last_chunk`](ControllerOutputChunk::last_chunk) set. A turn that fails
    /// sends the error as one `Err` item after the `processing` frame.
    fn stream(
        &self,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> BoxStream<'static, Result<ControllerOutputChunk>>;
}

/// What an [`EventHandler`] decided about one event.
#[derive(Debug, Clone)]
pub enum EventHandlerOutcome {
    /// Use these intents instead of running the recognizer.
    Intents(Vec<Intent>),
    /// Short-circuit the turn with this output.
    Output(Box<ControllerOutput>),
    /// Ignore the event; the turn continues with the next queued event.
    Ignored,
}

/// Replaces/augments the built-in intent recognizer for inbound events.
///
/// A handler sees every drained event, including the task signal kinds
/// (`task_interaction`, `task_completion`, `task_failed`) the built-in path
/// handles without intent recognition.
#[async_trait]
pub trait EventHandler: Send + Sync {
    /// Returns the handler's stable name.
    fn name(&self) -> &str;

    /// Decides what happens to `event`.
    ///
    /// # Errors
    ///
    /// Returns an error when the handler cannot inspect the event at all; the
    /// turn then fails instead of falling back to the recognizer.
    async fn handle(&self, event: &Event, ctx: &TaskContext<'_>) -> Result<EventHandlerOutcome>;
}

/// Per-turn routing state, shared by every event of one invocation.
struct TurnState {
    /// Query the router sees: the first non-empty event text of the turn.
    pub(super) query: String,
    /// Whether the router has already been consulted this turn.
    pub(super) resolved: bool,
    /// Router decision, meaningful once `resolved`.
    pub(super) task_type: Option<String>,
}

/// Controller internals behind one `Arc`, so `stream` can spawn a `'static`
/// task.
pub(super) struct Inner {
    /// Identity of the controller.
    pub(super) card: AgentCard,
    /// Behaviour knobs, shared with the background scheduler.
    pub(super) config: Arc<RwLock<ControllerConfig>>,
    /// Recognizer used when no handler is set.
    pub(super) recognizer: RwLock<Arc<dyn IntentRecognizer>>,
    /// Optional event handler.
    pub(super) handler: RwLock<Option<Arc<dyn EventHandler>>>,
    /// Optional task-type router.
    pub(super) router: RwLock<Option<Arc<dyn TaskRouter>>>,
    /// Intent-to-task planner (owns the id sequence).
    pub(super) planner: RwLock<Planner>,
    /// Executors keyed by task type, shared with the background scheduler.
    pub(super) registry: Arc<RwLock<TaskExecutorRegistry>>,
    /// Bounded FIFO event queue.
    pub(super) events: RwLock<VecDeque<Event>>,
    /// Tasks persisted across turns and shared with the background scheduler.
    pub(super) tasks: TaskManager,
    /// Sessions this controller is currently bound to, shared with the
    /// background scheduler's session lookup.
    pub(super) sessions: Arc<RwLock<BTreeMap<String, Arc<dyn Session>>>>,
    /// Whether the controller loop is marked running.
    pub(super) running: RwLock<bool>,
    /// The lazily started background task scheduler, once [`BaseController::start`]
    /// ran inside a `tokio` runtime.
    pub(super) scheduler_loop: RwLock<Option<Arc<TaskSchedulerLoop>>>,
    /// The group this controller's agent belongs to, when injected.
    pub(super) group: RwLock<Option<Arc<dyn GroupRouter>>>,
}

/// Event-queue + intent recognizer + planner + task scheduler, wired into one
/// turn.
///
/// The controller is a cheap handle: cloning it shares the same state, which is
/// what [`Controller::stream`] relies on to run the turn on a spawned task.
#[derive(Clone)]
pub struct BaseController {
    /// Shared controller state.
    pub(super) inner: Arc<Inner>,
}

impl BaseController {
    /// Builds a controller over `card` and `config` with the default
    /// [`RuleIntentRecognizer`].
    #[must_use]
    pub fn new(card: AgentCard, config: ControllerConfig) -> Self {
        Self::with_recognizer(card, config, Arc::new(RuleIntentRecognizer::new()))
    }

    /// Builds a controller with a custom intent recognizer.
    #[must_use]
    pub fn with_recognizer(
        card: AgentCard,
        config: ControllerConfig,
        recognizer: Arc<dyn IntentRecognizer>,
    ) -> Self {
        let planner = planner_for(&config, 1);
        let tasks = TaskManager::with_config(config.clone());
        Self {
            inner: Arc::new(Inner {
                card,
                config: Arc::new(RwLock::new(config)),
                recognizer: RwLock::new(recognizer),
                handler: RwLock::new(None),
                router: RwLock::new(None),
                planner: RwLock::new(planner),
                registry: Arc::new(RwLock::new(TaskExecutorRegistry::new())),
                events: RwLock::new(VecDeque::new()),
                tasks,
                sessions: Arc::new(RwLock::new(BTreeMap::new())),
                running: RwLock::new(false),
                scheduler_loop: RwLock::new(None),
                group: RwLock::new(None),
            }),
        }
    }

    /// Returns the controller's card.
    #[must_use]
    pub fn card(&self) -> &AgentCard {
        &self.inner.card
    }

    /// Returns a snapshot of the configuration.
    #[must_use]
    pub fn config(&self) -> ControllerConfig {
        self.inner.config.read().clone()
    }

    /// Replaces the configuration and re-derives the planner defaults from it.
    ///
    /// The planner's id sequence is preserved, so task ids stay unique across a
    /// reconfiguration.
    pub fn set_config(&self, config: ControllerConfig) {
        let next_seq = self.inner.planner.read().next_seq();
        *self.inner.planner.write() = planner_for(&config, next_seq);
        self.inner.tasks.set_config(config.clone());
        *self.inner.config.write() = config;
    }

    /// Queues `event`; returns `false` when the bounded queue is full and the
    /// event was dropped. A capacity of `0` means unlimited.
    #[must_use]
    pub fn push_event(&self, event: Event) -> bool {
        let capacity = self.inner.config.read().event_queue_size;
        let mut queue = self.inner.events.write();
        if capacity > 0 && queue.len() >= capacity {
            return false;
        }
        queue.push_back(event);
        true
    }

    /// Drains the queue in FIFO order.
    #[must_use]
    pub fn drain_events(&self) -> Vec<Event> {
        let mut queue = self.inner.events.write();
        queue.drain(..).collect()
    }

    /// Returns the number of queued events.
    #[must_use]
    pub fn pending_events(&self) -> usize {
        self.inner.events.read().len()
    }

    /// Replaces the intent recognizer.
    pub fn set_intent_recognizer(&self, recognizer: Arc<dyn IntentRecognizer>) {
        *self.inner.recognizer.write() = recognizer;
    }

    /// Returns the configured intent recognizer.
    #[must_use]
    pub fn intent_recognizer(&self) -> Arc<dyn IntentRecognizer> {
        Arc::clone(&self.inner.recognizer.read())
    }

    /// Installs an event handler, which then replaces the built-in recognition
    /// for every drained event.
    pub fn set_event_handler(&self, handler: Arc<dyn EventHandler>) {
        *self.inner.handler.write() = Some(handler);
    }

    /// Removes the event handler.
    pub fn clear_event_handler(&self) {
        *self.inner.handler.write() = None;
    }

    /// Installs the task-type router.
    pub fn set_router(&self, router: Arc<dyn TaskRouter>) {
        *self.inner.router.write() = Some(router);
    }

    /// Removes the task-type router.
    pub fn clear_router(&self) {
        *self.inner.router.write() = None;
    }

    /// Replaces the planner, including its id sequence.
    pub fn set_planner(&self, planner: Planner) {
        *self.inner.planner.write() = planner;
    }

    /// Registers `executor` under `task_type`.
    pub fn add_executor(&self, task_type: &str, executor: Arc<dyn TaskExecutor>) -> &Self {
        self.inner.registry.write().insert(task_type, executor);
        self
    }

    /// Registers `executor` under its own
    /// [`task_type`](TaskExecutor::task_type).
    pub fn add_default_executor(&self, executor: Arc<dyn TaskExecutor>) -> &Self {
        self.inner.registry.write().add(executor);
        self
    }

    /// Removes the executor registered for `task_type`.
    #[must_use]
    pub fn remove_executor(&self, task_type: &str) -> Option<Arc<dyn TaskExecutor>> {
        self.inner.registry.write().remove(task_type)
    }

    /// Returns the executor registered for `task_type`.
    #[must_use]
    pub fn executor(&self, task_type: &str) -> Option<Arc<dyn TaskExecutor>> {
        self.inner.registry.read().get(task_type)
    }

    /// Registered task types, in sorted order.
    #[must_use]
    pub fn task_types(&self) -> Vec<String> {
        self.inner.registry.read().task_types()
    }

    /// Returns a snapshot of the executor registry.
    #[must_use]
    pub fn registry(&self) -> TaskExecutorRegistry {
        self.inner.registry.read().clone()
    }

    /// Builds a scheduler over the current registry, the configured task budget,
    /// and the configured per-task timeout.
    #[must_use]
    pub fn scheduler(&self) -> TaskScheduler {
        let config = self.config();
        TaskScheduler::new(self.registry(), config.max_tasks)
            .with_timeout(config.task_timeout_ms.map(Duration::from_millis))
    }

    /// Returns a handle to the task store this controller and its background
    /// scheduler share.
    ///
    /// The handle is cheap to clone and synchronous, so a host can inspect,
    /// insert, or update tasks without going through a turn.
    #[must_use]
    pub fn task_manager(&self) -> TaskManager {
        self.inner.tasks.clone()
    }

    /// Marks the controller loop running and starts the background task
    /// scheduler.
    ///
    /// The scheduler is built lazily on the first call and started once: a
    /// second `start` keeps the running loop instead of spawning another. A
    /// stopped loop is replaced by a fresh one, because a
    /// [`TaskSchedulerLoop`] is single-use.
    ///
    /// Outside a `tokio` runtime only the running flag flips (the scheduler is
    /// a spawned task); the call never panics. The loop shares this
    /// controller's task store, executor registry, configuration, and bound
    /// sessions, so a task inserted with
    /// [`task_manager`](Self::task_manager) runs without another turn.
    pub fn start(&self) {
        *self.inner.running.write() = true;
        if tokio::runtime::Handle::try_current().is_err() {
            debug!(
                controller_id = self.card().id(),
                "no tokio runtime: the controller runs without a background task scheduler"
            );
            return;
        }
        let mut slot = self.inner.scheduler_loop.write();
        if slot
            .as_ref()
            .is_some_and(|scheduler| scheduler.is_running())
        {
            return;
        }
        let scheduler = self.build_scheduler_loop();
        scheduler.start();
        *slot = Some(scheduler);
    }

    /// Marks the controller loop stopped and asks the background scheduler to
    /// stop.
    ///
    /// The call is synchronous: inside a `tokio` runtime it spawns the
    /// scheduler's [`stop`](TaskSchedulerLoop::stop) and returns, so a caller
    /// that needs the in-flight runs finished uses
    /// [`shutdown`](Self::shutdown) instead. The handle is cleared immediately,
    /// so a later [`start`](Self::start) builds a fresh loop. Outside a runtime
    /// there is nothing to await; a loop started by an earlier runtime is left
    /// to that runtime's teardown and a warning is logged.
    pub fn stop(&self) {
        *self.inner.running.write() = false;
        let scheduler = self.inner.scheduler_loop.write().take();
        let Some(scheduler) = scheduler else {
            return;
        };
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move { scheduler.stop().await });
        } else {
            warn!(
                controller_id = self.card().id(),
                "no tokio runtime: the background task scheduler is left to its runtime"
            );
        }
    }

    /// Marks the controller loop stopped and awaits the background scheduler's
    /// stop, including every in-flight run.
    ///
    /// Unlike [`stop`](Self::stop) this is async, so a caller that needs the
    /// runs settled (a test, or a host shutting down) awaits it. Calling it on a
    /// controller that never started a scheduler is a no-op.
    pub async fn shutdown(&self) {
        *self.inner.running.write() = false;
        let scheduler = self.inner.scheduler_loop.write().take();
        if let Some(scheduler) = scheduler {
            scheduler.stop().await;
        }
    }

    /// Returns the background scheduler, once [`start`](Self::start) ran inside
    /// a `tokio` runtime.
    #[must_use]
    pub fn scheduler_loop(&self) -> Option<Arc<TaskSchedulerLoop>> {
        self.inner.scheduler_loop.read().clone()
    }

    /// Returns the host-facing state of the background scheduler.
    ///
    /// Derived from [`is_running`](Self::is_running) (the controller's running
    /// flag) and [`scheduler_loop`](Self::scheduler_loop) (the loop handle), so a
    /// caller does not have to combine them. [`is_running`](Self::is_running)
    /// keeps its own meaning: a loop started outside a `tokio` runtime leaves the
    /// flag set without a live loop, which this reports as
    /// [`TaskSchedulerState::Stopping`].
    #[must_use]
    pub fn scheduler_state(&self) -> TaskSchedulerState {
        if !self.is_running() {
            return TaskSchedulerState::Stopped;
        }
        match self.scheduler_loop() {
            Some(scheduler) if scheduler.is_running() => TaskSchedulerState::Running,
            _ => TaskSchedulerState::Stopping,
        }
    }

    /// Wakes the background scheduler for an immediate scan, returning whether a
    /// running loop was there to wake.
    ///
    /// Returns `false` when no loop is running (never started, already stopped,
    /// or started outside a `tokio` runtime), so a caller that gates its own
    /// follow-up work on the answer never assumes a scan happened.
    #[must_use]
    pub fn wake_scheduler(&self) -> bool {
        let Some(scheduler) = self.scheduler_loop() else {
            return false;
        };
        if !scheduler.is_running() {
            return false;
        }
        scheduler.wake();
        true
    }

    /// Ids of the tasks the background scheduler is currently running, in sorted
    /// order.
    ///
    /// This is the live run view the controller owns: an empty list means no loop
    /// is started, or nothing is in flight. It deliberately does not cover the
    /// per-conversation [`TaskQueue`](super::queue::TaskQueue), which is
    /// legacy-scoped — [`BaseController`] dispatches each turn synchronously and
    /// tracks no per-conversation turn, so that queue lives only on
    /// [`IntentDetectionController`](super::legacy::IntentDetectionController).
    #[must_use]
    pub fn running_tasks(&self) -> Vec<String> {
        self.scheduler_loop()
            .map(|scheduler| scheduler.in_flight())
            .unwrap_or_default()
    }

    /// Pauses the task `task_id`, returning whether it was paused.
    ///
    /// A started scheduler pauses a running task through the executor hooks and
    /// cancels its run (see [`TaskSchedulerLoop::pause_task`]). Without one the
    /// call is manager-only: an unknown id returns `false`, a
    /// [`TaskStatus::Submitted`] or [`TaskStatus::Working`] task becomes
    /// [`TaskStatus::Paused`] and returns `true`, and every other status
    /// returns `false`.
    pub async fn pause_task(&self, task_id: &str) -> bool {
        if let Some(scheduler) = self.scheduler_loop() {
            return scheduler.pause_task(task_id).await;
        }
        let Some(task) = self.inner.tasks.task(task_id) else {
            return false;
        };
        if !matches!(task.status, TaskStatus::Submitted | TaskStatus::Working) {
            return false;
        }
        self.inner
            .tasks
            .update_task_status(task_id, TaskStatus::Paused, false, false, None);
        true
    }

    /// Cancels the task `task_id`, returning whether it was cancelled.
    ///
    /// A started scheduler drives the two-phase cancel of
    /// [`TaskSchedulerLoop::cancel_task`]. Without one the call is manager-only:
    /// an unknown id returns `false`, a [`TaskStatus::Submitted`] task becomes
    /// [`TaskStatus::Canceled`] and returns `true`, and a task that already
    /// reached a terminal status returns `true` without changing (idempotent).
    /// A [`TaskStatus::Working`] task needs the scheduler to be stopped, so it
    /// returns `false` here.
    pub async fn cancel_task(&self, task_id: &str) -> bool {
        if let Some(scheduler) = self.scheduler_loop() {
            return scheduler.cancel_task(task_id).await;
        }
        let Some(task) = self.inner.tasks.task(task_id) else {
            return false;
        };
        if task.status.is_terminal() {
            return true;
        }
        if task.status != TaskStatus::Submitted {
            return false;
        }
        self.inner
            .tasks
            .update_task_status(task_id, TaskStatus::Canceled, false, false, None);
        true
    }

    /// Whether the controller loop is marked running.
    #[must_use]
    pub fn is_running(&self) -> bool {
        *self.inner.running.read()
    }

    /// Binds `session`; returns `true` when it was not bound before.
    ///
    /// An already-bound session keeps its original handle.
    ///
    // `async` for parity with the rest of the session-facing surface.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn bind_session(&self, session: Arc<dyn Session>) -> bool {
        let id = session.id().to_owned();
        let mut sessions = self.inner.sessions.write();
        if sessions.contains_key(&id) {
            return false;
        }
        sessions.insert(id, session);
        true
    }

    /// Unbinds and returns the session registered under `session_id`.
    // `async` for parity with the rest of the session-facing surface.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn unbind_session(&self, session_id: &str) -> Option<Arc<dyn Session>> {
        self.inner.sessions.write().remove(session_id)
    }

    /// Ids of the currently bound sessions, in sorted order.
    #[must_use]
    pub fn bound_sessions(&self) -> Vec<String> {
        self.inner.sessions.read().keys().cloned().collect()
    }

    /// Builds the reasoner over the current configuration, with this controller
    /// installed as its [`TaskControl`].
    ///
    /// The reasoner is the additive seam hosts use to turn intents into task
    /// operations; the built-in turn keeps its own control-intent handling.
    #[must_use]
    pub fn reasoner(&self) -> Reasoner {
        let control: Arc<dyn TaskControl> = Arc::new(self.clone());
        Reasoner::new(self.config(), Some(control))
    }

    /// Applies `ops` to this controller's task store and wakes the background
    /// scheduler.
    ///
    /// The planner is held for the batch (task ids stay unique across callers)
    /// and restored afterwards, so the caller sees the outcomes of every op;
    /// `session_id` scopes the minted ids, and `session` carries the
    /// clarification state of an [`TaskOp::Unknown`].
    pub async fn apply_task_ops(
        &self,
        ops: &[TaskOp],
        session_id: &str,
        session: Option<&Arc<dyn Session>>,
    ) -> Vec<TaskOpOutcome> {
        let reasoner = self.reasoner();
        let manager = self.task_manager();
        // `Reasoner::apply_task_ops` needs `&mut Planner` across awaits, so the
        // planner is moved out of its lock for the batch and written back
        // afterwards instead of holding a lock across an `await`.
        let mut planner = std::mem::take(&mut *self.inner.planner.write());
        let outcomes = reasoner
            .apply_task_ops(ops, &manager, &mut planner, session_id, session)
            .await;
        *self.inner.planner.write() = planner;
        if let Some(scheduler) = self.scheduler_loop() {
            scheduler.wake();
        }
        outcomes
    }

    /// Injects the group this controller's agent belongs to.
    ///
    /// Python calls this from `BaseGroup.add_agent`; [`send_to_agent`](
    /// Self::send_to_agent) and [`publish`](Self::publish) then route through
    /// the group's controller.
    pub fn set_group(&self, group: Arc<dyn GroupRouter>) {
        *self.inner.group.write() = Some(group);
    }

    /// Removes the group reference, so routing fails with the typed
    /// "not part of a group" error again.
    pub fn clear_group(&self) {
        *self.inner.group.write() = None;
    }

    /// Returns the group router this controller routes through, if any.
    #[must_use]
    pub fn group(&self) -> Option<Arc<dyn GroupRouter>> {
        self.inner.group.read().clone()
    }

    /// Returns the group router this controller routes through, if any.
    ///
    /// This is [`group`](Self::group) under the facade's name; both read the same
    /// injected router.
    #[must_use]
    pub fn group_router(&self) -> Option<Arc<dyn GroupRouter>> {
        self.group()
    }

    /// Injects `router` as the group this controller's agent belongs to and
    /// returns the controller, for builder-style wiring.
    ///
    /// The router is stored exactly as [`set_group`](Self::set_group) stores it,
    /// so clones made before the call keep the previous wiring and clones made
    /// after share the new one.
    #[must_use]
    pub fn with_group(self, router: Arc<dyn GroupRouter>) -> Self {
        self.set_group(router);
        self
    }

    /// Sends `event` to `agent_id` through the group's controller.
    ///
    /// # Errors
    ///
    /// Returns the typed configuration error of
    /// [`group::send_without_group`](super::group::send_without_group) when no
    /// group is installed, and whatever the group reports for an unknown target
    /// or a failing routed turn.
    pub async fn send_to_agent(
        &self,
        agent_id: &str,
        event: &Event,
        session: Option<Arc<dyn Session>>,
    ) -> Result<Value> {
        let group = self.group();
        let router = require_group(
            group.as_ref(),
            self.group_owner(),
            GroupMethod::SendToAgent(agent_id),
        )?;
        router.send_to_agent(agent_id, event, session).await
    }

    /// Publishes `event` to every subscriber through the group's controller.
    ///
    /// # Errors
    ///
    /// Returns the typed configuration error of
    /// [`group::publish_without_group`](super::group::publish_without_group) when
    /// no group is installed, and whatever the group reports for a publish it
    /// cannot dispatch.
    pub async fn publish(
        &self,
        event: &Event,
        session: Option<Arc<dyn Session>>,
    ) -> Result<Vec<Value>> {
        let group = self.group();
        let router = require_group(group.as_ref(), self.group_owner(), GroupMethod::Publish)?;
        router.publish(event, session).await
    }

    /// Name the group errors are attributed to: the card name, or its id when
    /// the name is empty.
    fn group_owner(&self) -> &str {
        let card = &self.inner.card;
        if card.name().is_empty() {
            card.id()
        } else {
            card.name()
        }
    }

    /// Builds the background scheduler over the shared state.
    fn build_scheduler_loop(&self) -> Arc<TaskSchedulerLoop> {
        let sessions = Arc::clone(&self.inner.sessions);
        let provider: SessionProvider =
            Arc::new(move |session_id: &str| sessions.read().get(session_id).cloned());
        Arc::new(TaskSchedulerLoop::new(
            self.inner.tasks.clone(),
            Arc::clone(&self.inner.registry),
            Arc::clone(&self.inner.config),
            self.inner.card.id().to_owned(),
            provider,
        ))
    }
}

#[async_trait]
impl TaskControl for BaseController {
    async fn pause(&self, task_id: &str) -> bool {
        self.pause_task(task_id).await
    }

    async fn cancel(&self, task_id: &str) -> bool {
        self.cancel_task(task_id).await
    }

    /// Flips a [`TaskStatus::Paused`] task back to [`TaskStatus::Submitted`]
    /// and wakes the scheduler.
    #[allow(clippy::unused_async_trait_impl)]
    async fn resume(&self, task_id: &str) -> bool {
        let Some(task) = self.inner.tasks.task(task_id) else {
            return false;
        };
        if task.status != TaskStatus::Paused {
            return false;
        }
        self.inner
            .tasks
            .update_task_status(task_id, TaskStatus::Submitted, false, false, None);
        if let Some(scheduler) = self.scheduler_loop() {
            scheduler.wake();
        }
        true
    }
}

/// Builds the planner defaults of `config`, keeping `next_seq`.
fn planner_for(config: &ControllerConfig, next_seq: u64) -> Planner {
    let mut planner = Planner::new()
        .with_default_task_type(config.default_task_type.clone())
        .with_max_tasks(config.max_tasks);
    planner.set_next_seq(next_seq.max(1));
    planner
}

#[cfg(test)]
mod accessor_tests {
    use super::fixtures::{controller_with_executor, test_session, StubExecutor};
    use super::*;

    #[test]
    fn state_key_matches_python() {
        assert_eq!(TASKS_STATE_KEY, "__controller_tasks__");
    }

    #[test]
    fn scheduler_state_names_are_stable() {
        assert_eq!(TaskSchedulerState::Stopped.as_str(), "stopped");
        assert_eq!(TaskSchedulerState::Running.to_string(), "running");
        assert_eq!(TaskSchedulerState::Stopping.as_str(), "stopping");
        assert_eq!(
            serde_json::to_value(TaskSchedulerState::Running).unwrap(),
            serde_json::json!("running")
        );
    }

    #[tokio::test]
    async fn scheduler_state_and_wake_follow_the_loop_lifecycle() {
        let (controller, _calls) = controller_with_executor();
        assert_eq!(controller.scheduler_state(), TaskSchedulerState::Stopped);
        assert!(!controller.wake_scheduler(), "no loop to wake");
        assert_eq!(controller.running_tasks(), Vec::<String>::new());

        controller.start();
        assert_eq!(controller.scheduler_state(), TaskSchedulerState::Running);
        assert!(controller.wake_scheduler(), "the started loop is woken");
        assert_eq!(
            controller.running_tasks(),
            Vec::<String>::new(),
            "nothing is in flight"
        );
        controller.shutdown().await;
        assert_eq!(controller.scheduler_state(), TaskSchedulerState::Stopped);
        assert!(!controller.wake_scheduler());

        controller.start();
        assert_eq!(controller.scheduler_state(), TaskSchedulerState::Running);
        controller.stop();
        assert_eq!(controller.scheduler_state(), TaskSchedulerState::Stopped);
        assert!(!controller.wake_scheduler());
    }

    #[test]
    fn scheduler_state_without_a_runtime_is_stopping() {
        let (controller, _calls) = controller_with_executor();
        controller.start();
        assert!(controller.is_running(), "the flag flips outside a runtime");
        assert_eq!(controller.scheduler_state(), TaskSchedulerState::Stopping);
        assert!(
            !controller.wake_scheduler(),
            "no spawned loop exists to wake"
        );
        controller.stop();
        assert_eq!(controller.scheduler_state(), TaskSchedulerState::Stopped);
    }

    #[tokio::test]
    async fn lifecycle_and_event_queue() {
        let config = ControllerConfig {
            event_queue_size: 2,
            ..ControllerConfig::new()
        };
        let controller = BaseController::new(AgentCard::new("ctrl"), config);
        assert!(!controller.is_running());
        controller.start();
        assert!(controller.is_running());
        controller.stop();
        assert!(!controller.is_running());

        let session = test_session();
        assert!(controller.bind_session(Arc::clone(&session)).await);
        assert!(!controller.bind_session(Arc::clone(&session)).await);
        assert_eq!(controller.bound_sessions(), vec!["s1".to_owned()]);
        assert!(controller.unbind_session("s1").await.is_some());
        assert!(controller.unbind_session("s1").await.is_none());

        let (executor, _) = StubExecutor::new("agent", "done");
        controller.add_executor("agent", Arc::new(executor));
        assert_eq!(controller.task_types(), vec!["agent".to_owned()]);
        assert!(controller.executor("agent").is_some());
        assert_eq!(controller.registry().len(), 1);
        assert!(controller.remove_executor("agent").is_some());
        assert!(
            controller.task_types().is_empty(),
            "registry should be empty"
        );

        assert!(controller.push_event(Event::user_input("a")));
        assert!(controller.push_event(Event::user_input("b")));
        assert!(!controller.push_event(Event::user_input("c")));
        assert_eq!(controller.pending_events(), 2);
        let drained = controller.drain_events();
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].text().as_deref(), Some("a"));
        assert_eq!(drained[1].text().as_deref(), Some("b"));
        assert_eq!(controller.pending_events(), 0);
    }
}
