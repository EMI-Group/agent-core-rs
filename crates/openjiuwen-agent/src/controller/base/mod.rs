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
//! Task state survives between turns in the session under [`TASKS_STATE_KEY`],
//! so a later controller instance restores it with
//! [`BaseController::load_state`]. Unlike Python's background scheduler this
//! port never spawns a scheduling loop: one `invoke` processes exactly the
//! events queued when it started, and the `stream` seam wraps that turn in a
//! channel.
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
    TaskContext, TaskExecutor, TaskExecutorRegistry, TaskRouter, TaskScheduler,
};
use super::intent::{IntentRecognizer, Planner, RuleIntentRecognizer};
use super::schema::{ControllerOutput, ControllerOutputChunk, Event, Intent, Task};
use crate::card::AgentCard;
use async_trait::async_trait;
use futures::stream::BoxStream;
use openjiuwen_common::error::Result;
use openjiuwen_session::Session;
use parking_lot::RwLock;
use serde_json::Value;
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;

/// Key the controller task state is persisted under.
pub const TASKS_STATE_KEY: &str = "__controller_tasks__";

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
    /// Behaviour knobs.
    pub(super) config: RwLock<ControllerConfig>,
    /// Recognizer used when no handler is set.
    pub(super) recognizer: RwLock<Arc<dyn IntentRecognizer>>,
    /// Optional event handler.
    pub(super) handler: RwLock<Option<Arc<dyn EventHandler>>>,
    /// Optional task-type router.
    pub(super) router: RwLock<Option<Arc<dyn TaskRouter>>>,
    /// Intent-to-task planner (owns the id sequence).
    pub(super) planner: RwLock<Planner>,
    /// Executors keyed by task type.
    pub(super) registry: RwLock<TaskExecutorRegistry>,
    /// Bounded FIFO event queue.
    pub(super) events: RwLock<VecDeque<Event>>,
    /// Tasks persisted across turns, keyed by task id.
    pub(super) tasks: RwLock<BTreeMap<String, Task>>,
    /// Sessions this controller is currently bound to, keyed by session id.
    pub(super) sessions: RwLock<BTreeMap<String, Arc<dyn Session>>>,
    /// Whether the controller loop is marked running.
    pub(super) running: RwLock<bool>,
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
        Self {
            inner: Arc::new(Inner {
                card,
                config: RwLock::new(config),
                recognizer: RwLock::new(recognizer),
                handler: RwLock::new(None),
                router: RwLock::new(None),
                planner: RwLock::new(planner),
                registry: RwLock::new(TaskExecutorRegistry::new()),
                events: RwLock::new(VecDeque::new()),
                tasks: RwLock::new(BTreeMap::new()),
                sessions: RwLock::new(BTreeMap::new()),
                running: RwLock::new(false),
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

    /// Marks the controller loop running.
    pub fn start(&self) {
        *self.inner.running.write() = true;
    }

    /// Marks the controller loop stopped.
    pub fn stop(&self) {
        *self.inner.running.write() = false;
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
    use super::fixtures::{test_session, StubExecutor};
    use super::*;

    #[test]
    fn state_key_matches_python() {
        assert_eq!(TASKS_STATE_KEY, "__controller_tasks__");
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
