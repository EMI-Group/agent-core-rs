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
use super::config::ControllerConfig;
use super::executors::{
    TaskContext, TaskExecutor, TaskExecutorRegistry, TaskRouter, TaskRun, TaskScheduler,
};
use super::intent::{IntentRecognizer, Planner, RuleIntentRecognizer};
use super::schema::{
    frames_text, ControllerOutput, ControllerOutputChunk, ControllerOutputPayload, DataFrame,
    Event, EventType, Intent, IntentType, OutputKind, Task, TaskStatus,
};
use crate::card::AgentCard;
use crate::run::resolve_session;
use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_session::Session;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, warn};

/// Key the controller task state is persisted under.
pub const TASKS_STATE_KEY: &str = "__controller_tasks__";

/// Clarification text used when an unknown-task intent carries no prompt.
const CLARIFICATION: &str = "Please clarify your request.";
/// Reply of a paused task control intent.
const PAUSED_REPLY: &str = "paused";
/// Reply of a cancelled task control intent.
const CANCELED_REPLY: &str = "canceled";

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
    /// Returns [`Error::validation`] when `inputs` is not a controller input
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
    query: String,
    /// Whether the router has already been consulted this turn.
    resolved: bool,
    /// Router decision, meaningful once `resolved`.
    task_type: Option<String>,
}

impl TurnState {
    /// Builds the routing state of a turn over `events`.
    fn new(events: &[Event]) -> Self {
        Self {
            query: first_query(events),
            resolved: false,
            task_type: None,
        }
    }
}

/// Controller internals behind one `Arc`, so `stream` can spawn a `'static`
/// task.
struct Inner {
    /// Identity of the controller.
    card: AgentCard,
    /// Behaviour knobs.
    config: RwLock<ControllerConfig>,
    /// Recognizer used when no handler is set.
    recognizer: RwLock<Arc<dyn IntentRecognizer>>,
    /// Optional event handler.
    handler: RwLock<Option<Arc<dyn EventHandler>>>,
    /// Optional task-type router.
    router: RwLock<Option<Arc<dyn TaskRouter>>>,
    /// Intent-to-task planner (owns the id sequence).
    planner: RwLock<Planner>,
    /// Executors keyed by task type.
    registry: RwLock<TaskExecutorRegistry>,
    /// Bounded FIFO event queue.
    events: RwLock<VecDeque<Event>>,
    /// Tasks persisted across turns, keyed by task id.
    tasks: RwLock<BTreeMap<String, Task>>,
    /// Sessions this controller is currently bound to, keyed by session id.
    sessions: RwLock<BTreeMap<String, Arc<dyn Session>>>,
    /// Whether the controller loop is marked running.
    running: RwLock<bool>,
}

/// The persisted controller state.
#[derive(Debug, Default, Serialize, Deserialize)]
struct TasksState {
    /// Tasks in id order.
    #[serde(default)]
    tasks: Vec<Task>,
    /// Next planner sequence number.
    #[serde(default)]
    next_seq: u64,
}

/// Event-queue + intent recognizer + planner + task scheduler, wired into one
/// turn.
///
/// The controller is a cheap handle: cloning it shares the same state, which is
/// what [`Controller::stream`] relies on to run the turn on a spawned task.
#[derive(Clone)]
pub struct BaseController {
    /// Shared controller state.
    inner: Arc<Inner>,
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

    /// Persists the task map and the planner sequence on `session`.
    ///
    /// Does nothing when
    /// [`enable_task_persistence`](ControllerConfig::enable_task_persistence)
    /// is off.
    ///
    /// # Errors
    ///
    /// Propagates the session's state-write error and returns
    /// [`Error::validation`] when the task map cannot be serialized.
    pub async fn save_state(&self, session: &dyn Session) -> Result<()> {
        if !self.config().enable_task_persistence {
            return Ok(());
        }
        let state = TasksState {
            tasks: self.inner.tasks.read().values().cloned().collect(),
            next_seq: self.inner.planner.read().next_seq(),
        };
        let value = serde_json::to_value(&state)
            .map_err(|error| Error::validation(format!("cannot serialize task state: {error}")))?;
        session
            .update_state(json!({ TASKS_STATE_KEY: value }))
            .await
    }

    /// Restores the task map and the planner sequence from `session`.
    ///
    /// The read is tolerant: a missing or malformed state clears the in-memory
    /// task map and restarts the id sequence at one instead of failing.
    ///
    /// # Errors
    ///
    /// Propagates the session's state-read error.
    pub async fn load_state(&self, session: &dyn Session) -> Result<()> {
        let state = session.get_state(TASKS_STATE_KEY).await?;
        let parsed = match state {
            None | Some(Value::Null) => None,
            Some(value) => match serde_json::from_value::<TasksState>(value) {
                Ok(state) => Some(state),
                Err(error) => {
                    warn!(%error, "ignoring an unreadable controller task state");
                    None
                }
            },
        };
        let (tasks, next_seq) = parsed.map_or_else(
            || (Vec::new(), 1),
            |state| (state.tasks, state.next_seq.max(1)),
        );
        let map = tasks
            .into_iter()
            .map(|task| (task.task_id.clone(), task))
            .collect();
        *self.inner.tasks.write() = map;
        self.inner.planner.write().set_next_seq(next_seq);
        Ok(())
    }

    /// Returns the persisted tasks, sorted by task id.
    // `async` because the surface is shared with `save_state`/`load_state`.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    #[must_use]
    pub async fn tasks(&self) -> Vec<Task> {
        self.inner.tasks.read().values().cloned().collect()
    }

    /// Runs one turn: queue the event, recognize, plan, dispatch, aggregate.
    async fn invoke_turn(
        &self,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> Result<ControllerOutput> {
        let event = Event::from_user_inputs(&inputs)?;
        let run = resolve_session(
            session,
            self.card().id(),
            Some(self.card().name()),
            None,
            None,
        )
        .await?;
        let session = Arc::clone(run.session());
        self.start();
        self.bind_session(Arc::clone(&session)).await;
        if self.config().enable_task_persistence {
            self.load_state(session.as_ref()).await?;
        }
        if !self.push_event(event) {
            warn!(
                controller_id = self.card().id(),
                "controller event queue is full; dropping the inbound event"
            );
        }
        let events = self.drain_events();
        let mut turn = TurnState::new(&events);
        let mut output: Option<ControllerOutput> = None;
        for event in &events {
            if let Some(produced) = self.process_event(event, &session, &mut turn).await? {
                output = Some(produced);
            }
        }
        let output = output.unwrap_or_else(|| self.fallback_output());
        self.save_state(session.as_ref()).await?;
        self.unbind_session(session.id()).await;
        run.finish().await;
        Ok(output)
    }

    /// Streams one turn: a `processing` frame, the turn frames, and one final
    /// frame.
    fn stream_turn(
        &self,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> BoxStream<'static, Result<ControllerOutputChunk>> {
        let controller = self.clone();
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Result<ControllerOutputChunk>>();
        tokio::spawn(async move {
            let _ = tx.send(Ok(ControllerOutputChunk::new(
                0,
                ControllerOutputPayload::processing(),
            )));
            match controller.invoke_turn(inputs, session).await {
                Ok(output) => {
                    let mut index = 1_i64;
                    for frame in &output.frames {
                        let mut frame = frame.clone();
                        frame.index = index;
                        frame.last_chunk = false;
                        index += 1;
                        if tx.send(Ok(frame)).is_err() {
                            return;
                        }
                    }
                    let kind = match output.status {
                        TaskStatus::InputRequired => OutputKind::TaskInteraction,
                        TaskStatus::Failed => OutputKind::TaskFailed,
                        _ => OutputKind::TaskCompletion,
                    };
                    let last = ControllerOutputChunk::last(
                        index,
                        ControllerOutputPayload::text(kind, output.output),
                    );
                    let _ = tx.send(Ok(last));
                }
                Err(error) => {
                    let _ = tx.send(Err(error));
                }
            }
        });
        futures::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        })
        .boxed()
    }

    /// Produces the turn output of one drained event.
    ///
    /// `Ok(None)` means the event contributed nothing (an ignored handler
    /// decision or an input that planned no task), so an earlier event's output
    /// stays the turn output.
    async fn process_event(
        &self,
        event: &Event,
        session: &Arc<dyn Session>,
        turn: &mut TurnState,
    ) -> Result<Option<ControllerOutput>> {
        let handler = self.inner.handler.read().clone();
        if let Some(handler) = handler {
            let ctx = TaskContext::new(Some(Arc::clone(session)), self.card().id());
            return match handler.handle(event, &ctx).await? {
                EventHandlerOutcome::Output(output) => Ok(Some(*output)),
                EventHandlerOutcome::Ignored => Ok(None),
                EventHandlerOutcome::Intents(intents) => {
                    self.handle_intents(event, intents, session, turn).await
                }
            };
        }
        if matches!(event.event_type, EventType::Input | EventType::FollowUp) {
            let intents = if self.config().enable_intent_recognition {
                let recognizer = self.inner.recognizer.read().clone();
                recognizer.recognize(event).await?
            } else {
                vec![Intent::default_for(event.clone())]
            };
            return self.handle_intents(event, intents, session, turn).await;
        }
        Ok(self.signal_output(event))
    }

    /// Routes recognized intents: clarification, task control, or new tasks.
    async fn handle_intents(
        &self,
        event: &Event,
        intents: Vec<Intent>,
        session: &Arc<dyn Session>,
        turn: &mut TurnState,
    ) -> Result<Option<ControllerOutput>> {
        let config = self.config();
        let mut accepted: Vec<Intent> = intents
            .into_iter()
            .filter(|intent| config.accepts(intent.intent_type))
            .collect();
        if config.max_intents > 0 {
            accepted.truncate(config.max_intents);
        }
        if accepted.is_empty() && matches!(event.event_type, EventType::Input | EventType::FollowUp)
        {
            accepted.push(Intent::default_for(event.clone()));
        }
        if accepted.is_empty() {
            return Ok(None);
        }
        let mut planned: Vec<Intent> = Vec::new();
        let mut dispatch_queue: Vec<Task> = Vec::new();
        let mut control: Option<ControllerOutput> = None;
        for intent in accepted {
            match intent.intent_type {
                IntentType::UnknownTask => return Ok(Some(clarification_output(&intent))),
                IntentType::PauseTask | IntentType::CancelTask => {
                    let status = if intent.intent_type == IntentType::PauseTask {
                        TaskStatus::Paused
                    } else {
                        TaskStatus::Canceled
                    };
                    control = Some(match self.set_task_status(&intent, status) {
                        Some(output) => output,
                        None => not_found_output(&intent),
                    });
                }
                IntentType::ResumeTask => match self.resume_task(&intent) {
                    Some(task) => dispatch_queue.push(task),
                    None => control = Some(not_found_output(&intent)),
                },
                _ => planned.push(intent),
            }
        }
        if !planned.is_empty() {
            let task_type = self.route(turn, &planned).await?;
            dispatch_queue.extend(self.plan_tasks(&planned, task_type.as_deref(), session.id()));
        }
        if dispatch_queue.is_empty() {
            return Ok(control);
        }
        Ok(Some(self.dispatch_runs(dispatch_queue, session).await))
    }

    /// Asks the router once per turn for the planned task type.
    async fn route(&self, turn: &mut TurnState, intents: &[Intent]) -> Result<Option<String>> {
        if turn.resolved {
            return Ok(turn.task_type.clone());
        }
        let router = self.inner.router.read().clone();
        let has_choice = self.inner.registry.read().len() >= 2;
        let route = match router {
            Some(router) if has_choice => router
                .select(&turn.query, intents)
                .await?
                .filter(|task_type| !task_type.is_empty()),
            _ => None,
        };
        turn.resolved = true;
        turn.task_type.clone_from(&route);
        Ok(route)
    }

    /// Plans one task per valid, distinct intent.
    fn plan_tasks(
        &self,
        intents: &[Intent],
        task_type: Option<&str>,
        session_id: &str,
    ) -> Vec<Task> {
        self.inner
            .planner
            .write()
            .plan_with_task_type(session_id, intents, task_type)
    }

    /// Moves the persisted task targeted by a control intent to `status`.
    fn set_task_status(&self, intent: &Intent, status: TaskStatus) -> Option<ControllerOutput> {
        let id = intent.target_task_id.as_deref()?;
        let mut tasks = self.inner.tasks.write();
        tasks.get_mut(id)?.status = status;
        let reply = match status {
            TaskStatus::Canceled => CANCELED_REPLY,
            _ => PAUSED_REPLY,
        };
        Some(
            ControllerOutput::new(format!("task {id} {reply}"), status)
                .with_task_id(id)
                .with_data(json!({"task_id": id, "status": status.as_str()})),
        )
    }

    /// Returns the persisted task targeted by a resume intent, re-queued.
    fn resume_task(&self, intent: &Intent) -> Option<Task> {
        let id = intent.target_task_id.as_deref()?;
        let mut tasks = self.inner.tasks.write();
        let task = tasks.get_mut(id)?;
        task.status = TaskStatus::Submitted;
        Some(task.clone())
    }

    /// Dispatches `tasks` and aggregates the runs into one output.
    async fn dispatch_runs(
        &self,
        tasks: Vec<Task>,
        session: &Arc<dyn Session>,
    ) -> ControllerOutput {
        let ctx = TaskContext::new(Some(Arc::clone(session)), self.card().id());
        let runs = self.scheduler().dispatch(tasks, &ctx).await;
        let config = self.config();
        let mut frames = Vec::new();
        let mut index = 0_i64;
        let mut texts = Vec::new();
        let mut entries = Vec::new();
        for run in &runs {
            frames.extend(run.frames(index));
            index += 1;
            if run.status() == TaskStatus::Completed {
                let text = run.text();
                if !text.is_empty() {
                    texts.push(text);
                }
            }
            entries.push(json!({
                "task_id": run.task.task_id,
                "status": run.status().as_str(),
                "output": run.text(),
            }));
        }
        {
            let mut stored = self.inner.tasks.write();
            for run in &runs {
                stored.insert(run.task.task_id.clone(), run.task.clone());
            }
        }
        let status = aggregate_status(&runs);
        let mut text = texts.join("\n\n");
        if text.is_empty() {
            text = config.default_response.text.clone().unwrap_or_default();
        }
        if !config.suppress_completion_signal {
            let kind = match status {
                TaskStatus::InputRequired => OutputKind::TaskInteraction,
                TaskStatus::Failed => OutputKind::TaskFailed,
                _ => OutputKind::TaskCompletion,
            };
            frames.push(ControllerOutputChunk::last(
                index,
                ControllerOutputPayload::text(kind, text.clone()),
            ));
        }
        let mut output = ControllerOutput::new(text, status)
            .with_data(json!({"tasks": entries}))
            .with_frames(frames);
        if let Some(run) = runs.last() {
            output = output.with_task_id(run.task.task_id.clone());
        }
        output
    }

    /// Builds the output of a task signal event and records it on the task.
    fn signal_output(&self, event: &Event) -> Option<ControllerOutput> {
        let task_id = event.task_id.clone().unwrap_or_default();
        let (status, text, kind, data) = match event.event_type {
            EventType::TaskCompletion => {
                let text = event
                    .text()
                    .filter(|text| !text.is_empty())
                    .unwrap_or_else(|| format!("task {task_id} completed"));
                (
                    TaskStatus::Completed,
                    text,
                    OutputKind::TaskCompletion,
                    event.task_result.clone(),
                )
            }
            EventType::TaskFailed => {
                let text = event
                    .error_message
                    .clone()
                    .filter(|text| !text.is_empty())
                    .or_else(|| event.text().filter(|text| !text.is_empty()))
                    .unwrap_or_else(|| format!("task {task_id} failed"));
                (TaskStatus::Failed, text, OutputKind::TaskFailed, Vec::new())
            }
            EventType::TaskInteraction => {
                let text = frames_text(&event.interaction)
                    .filter(|text| !text.is_empty())
                    .unwrap_or_else(|| format!("task {task_id} requires input"));
                (
                    TaskStatus::InputRequired,
                    text,
                    OutputKind::TaskInteraction,
                    event.interaction.clone(),
                )
            }
            EventType::Input | EventType::FollowUp => return None,
        };
        let frames = if data.is_empty() {
            vec![DataFrame::text(text.clone())]
        } else {
            data
        };
        let chunk = ControllerOutputChunk::last(0, ControllerOutputPayload::new(kind, frames));
        if !task_id.is_empty() {
            self.record_signal(&task_id, status, event, &text, &chunk);
        }
        let mut output = ControllerOutput::new(text, status).with_frames(vec![chunk]);
        if !task_id.is_empty() {
            output = output.with_task_id(task_id);
        }
        Some(output)
    }

    /// Folds a task signal event into the persisted task it names.
    fn record_signal(
        &self,
        task_id: &str,
        status: TaskStatus,
        event: &Event,
        text: &str,
        chunk: &ControllerOutputChunk,
    ) {
        let mut tasks = self.inner.tasks.write();
        let Some(task) = tasks.get_mut(task_id) else {
            debug!(
                task_id,
                "controller signal names a task that is not persisted"
            );
            return;
        };
        task.status = status;
        if status == TaskStatus::Failed {
            task.error_message = event
                .error_message
                .clone()
                .or_else(|| Some(text.to_owned()));
        }
        if status == TaskStatus::InputRequired && task.input_required_fields.is_none() {
            task.input_required_fields = Some(json!({"fields": []}));
        }
        task.outputs.push(chunk.clone());
    }

    /// Output of a turn that produced nothing: the configured default response.
    fn fallback_output(&self) -> ControllerOutput {
        let text = self.config().default_response.text.unwrap_or_default();
        ControllerOutput::new(text, TaskStatus::Completed)
    }
}

#[async_trait]
impl Controller for BaseController {
    fn card(&self) -> &AgentCard {
        BaseController::card(self)
    }

    fn config(&self) -> ControllerConfig {
        BaseController::config(self)
    }

    async fn invoke(
        &self,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> Result<ControllerOutput> {
        self.invoke_turn(inputs, session).await
    }

    fn stream(
        &self,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> BoxStream<'static, Result<ControllerOutputChunk>> {
        self.stream_turn(inputs, session)
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

/// First non-empty event text of a turn, or `""`.
fn first_query(events: &[Event]) -> String {
    events
        .iter()
        .find_map(|event| event.text().filter(|text| !text.is_empty()))
        .unwrap_or_default()
}

/// Status of a dispatched batch: interaction first, then completion, then
/// failure; an empty batch counts as completed.
fn aggregate_status(runs: &[TaskRun]) -> TaskStatus {
    if runs
        .iter()
        .any(|run| run.status() == TaskStatus::InputRequired)
    {
        TaskStatus::InputRequired
    } else if runs.iter().any(|run| run.status() == TaskStatus::Completed) {
        TaskStatus::Completed
    } else if runs.iter().any(|run| run.status() == TaskStatus::Failed) {
        TaskStatus::Failed
    } else {
        TaskStatus::Completed
    }
}

/// Builds the clarification output of an unknown-task intent.
fn clarification_output(intent: &Intent) -> ControllerOutput {
    let prompt = intent
        .clarification_prompt
        .clone()
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| CLARIFICATION.to_owned());
    ControllerOutput::new(prompt.clone(), TaskStatus::InputRequired)
        .with_data(json!({"clarification_prompt": prompt}))
}

/// Builds the failed output of a control intent whose target is unknown.
fn not_found_output(intent: &Intent) -> ControllerOutput {
    let id = intent.target_task_id.clone().unwrap_or_default();
    warn!(
        intent = intent.label(),
        task_id = %id,
        "controller intent targets an unknown task"
    );
    ControllerOutput::new(format!("task {id} not found"), TaskStatus::Failed)
        .with_data(json!({"task_id": id}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controller::config::DefaultResponse;
    use crate::controller::executors::{ExecutorInfo, LLMController};
    use crate::controller::intent::IntentRule;
    use crate::controller::schema::TaskResult;
    use futures::StreamExt;
    use openjiuwen_session::AgentSession;
    use openjiuwen_test_support::ScriptedModel;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Description reported by [`StubExecutor`].
    const STUB_DESCRIPTION: &str = "stub executor";
    /// Name reported by [`StubHandler`] and [`CountingRecognizer`].
    const STUB_NAME: &str = "stub";

    /// Executor that records its calls and answers with a fixed text.
    struct StubExecutor {
        task_type: &'static str,
        text: &'static str,
        calls: Arc<AtomicUsize>,
    }

    impl StubExecutor {
        fn new(task_type: &'static str, text: &'static str) -> (Self, Arc<AtomicUsize>) {
            let calls = Arc::new(AtomicUsize::new(0));
            (
                Self {
                    task_type,
                    text,
                    calls: Arc::clone(&calls),
                },
                calls,
            )
        }
    }

    #[async_trait]
    impl TaskExecutor for StubExecutor {
        fn task_type(&self) -> &str {
            self.task_type
        }

        fn description(&self) -> &str {
            STUB_DESCRIPTION
        }

        async fn execute(&self, _task: &Task, _ctx: &TaskContext<'_>) -> Result<TaskResult> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(TaskResult::text(self.text))
        }
    }

    /// Handler that answers every event with one fixed outcome.
    struct StubHandler {
        outcome: EventHandlerOutcome,
    }

    #[async_trait]
    impl EventHandler for StubHandler {
        fn name(&self) -> &str {
            STUB_NAME
        }

        async fn handle(
            &self,
            _event: &Event,
            _ctx: &TaskContext<'_>,
        ) -> Result<EventHandlerOutcome> {
            Ok(self.outcome.clone())
        }
    }

    /// Recognizer that answers the first call with one intent and every later
    /// call with two distinct intents.
    struct CountingRecognizer {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl IntentRecognizer for CountingRecognizer {
        fn name(&self) -> &str {
            STUB_NAME
        }

        async fn recognize(&self, event: &Event) -> Result<Vec<Intent>> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            let count = if call == 0 { 1 } else { 2 };
            Ok((0..count)
                .map(|index| {
                    Intent::create_task(event.clone(), format!("task {call}-{index}"))
                        .with_name(format!("t{call}{index}"))
                })
                .collect())
        }
    }

    fn test_session() -> Arc<dyn Session> {
        Arc::new(AgentSession::new("s1"))
    }

    fn controller_with_executor() -> (BaseController, Arc<AtomicUsize>) {
        let controller = BaseController::new(AgentCard::new("ctrl"), ControllerConfig::new());
        let (executor, calls) = StubExecutor::new("agent", "done");
        controller.add_executor("agent", Arc::new(executor));
        (controller, calls)
    }

    /// Invokes `controller` with an empty query after queueing one control event
    /// that names `task_id`.
    async fn control_turn(
        controller: &BaseController,
        session: &Arc<dyn Session>,
        task_id: &str,
        text: &str,
    ) -> ControllerOutput {
        let mut event = Event::user_input(text);
        event.task_id = Some(task_id.to_owned());
        assert!(controller.push_event(event));
        Controller::invoke(controller, json!({"query": ""}), Some(Arc::clone(session)))
            .await
            .unwrap()
    }

    #[test]
    fn state_key_matches_python() {
        assert_eq!(TASKS_STATE_KEY, "__controller_tasks__");
    }

    #[tokio::test]
    async fn router_selects_between_two_executors() {
        let (controller, agent_calls) = controller_with_executor();
        let (workflow, workflow_calls) = StubExecutor::new("workflow", "workflow reply");
        controller.add_executor("workflow", Arc::new(workflow));
        let model = Arc::new(ScriptedModel::builder().respond_text("workflow").build());
        controller.set_router(Arc::new(LLMController::new(
            model.clone(),
            vec![
                ExecutorInfo::new("agent", "agent", "runs an agent"),
                ExecutorInfo::new("workflow", "workflow", "runs a workflow"),
            ],
        )));
        let output =
            Controller::invoke(&controller, json!({"query": "do it"}), Some(test_session()))
                .await
                .unwrap();
        assert_eq!(output.output, "workflow reply");
        assert_eq!(output.status, TaskStatus::Completed);
        assert_eq!(model.call_count(), 1);
        assert_eq!(agent_calls.load(Ordering::SeqCst), 0);
        assert_eq!(workflow_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn single_executor_skips_the_router() {
        let (controller, calls) = controller_with_executor();
        let model = Arc::new(ScriptedModel::builder().build());
        controller.set_router(Arc::new(LLMController::new(
            model.clone(),
            vec![ExecutorInfo::new("agent", "agent", "runs an agent")],
        )));
        let output =
            Controller::invoke(&controller, json!({"query": "do it"}), Some(test_session()))
                .await
                .unwrap();
        assert_eq!(output.output, "done");
        assert_eq!(model.call_count(), 0);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn unregistered_default_task_type_uses_the_default_response() {
        let config = ControllerConfig::new()
            .with_default_task_type("missing")
            .with_default_response(DefaultResponse::text("fallback"));
        let controller = BaseController::new(AgentCard::new("ctrl"), config);
        let session = test_session();
        let output = Controller::invoke(
            &controller,
            json!({"query": "do it"}),
            Some(Arc::clone(&session)),
        )
        .await
        .unwrap();
        assert_eq!(output.output, "fallback");
        assert_eq!(output.status, TaskStatus::Failed);
        assert_eq!(
            output.data.as_ref().unwrap()["tasks"][0]["status"],
            json!("failed")
        );
        let tasks = controller.tasks().await;
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].status, TaskStatus::Failed);
        let state = session.get_state(TASKS_STATE_KEY).await.unwrap().unwrap();
        assert_eq!(state["tasks"][0]["status"], json!("failed"));
    }

    #[tokio::test]
    async fn state_persists_across_turns_and_instances() {
        let controller = BaseController::with_recognizer(
            AgentCard::new("ctrl"),
            ControllerConfig::new(),
            Arc::new(CountingRecognizer {
                calls: AtomicUsize::new(0),
            }),
        );
        controller.add_executor("agent", Arc::new(StubExecutor::new("agent", "done").0));
        let session = test_session();
        let first = Controller::invoke(
            &controller,
            json!({"query": "one"}),
            Some(Arc::clone(&session)),
        )
        .await
        .unwrap();
        assert_eq!(controller.tasks().await.len(), 1);
        let second = Controller::invoke(
            &controller,
            json!({"query": "two"}),
            Some(Arc::clone(&session)),
        )
        .await
        .unwrap();
        assert_ne!(first.task_id, second.task_id);
        let tasks = controller.tasks().await;
        assert_eq!(tasks.len(), 3);
        let ids: std::collections::BTreeSet<_> =
            tasks.iter().map(|task| task.task_id.clone()).collect();
        assert_eq!(ids.len(), 3);
        let restored = BaseController::new(AgentCard::new("ctrl"), ControllerConfig::new());
        restored.load_state(session.as_ref()).await.unwrap();
        assert_eq!(restored.tasks().await.len(), 3);
    }

    #[tokio::test]
    async fn stream_starts_with_processing_and_ends_with_last_chunk() {
        let (controller, _) = controller_with_executor();
        let mut stream =
            Controller::stream(&controller, json!({"query": "go"}), Some(test_session()));
        let mut chunks = Vec::new();
        while let Some(item) = stream.next().await {
            chunks.push(item.unwrap());
        }
        assert!(chunks.len() >= 2);
        assert_eq!(
            chunks[0].payload.as_ref().unwrap().kind,
            OutputKind::Processing
        );
        let last = chunks.last().unwrap();
        assert!(last.last_chunk);
        assert_eq!(
            last.payload.as_ref().unwrap().kind,
            OutputKind::TaskCompletion
        );
        assert_eq!(chunks.iter().filter(|chunk| chunk.last_chunk).count(), 1);
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

    #[tokio::test]
    async fn event_handler_intents_override_the_recognizer() {
        let (controller, _) = controller_with_executor();
        let intent =
            Intent::create_task(Event::user_input("handled"), "handled").with_name("from_handler");
        controller.set_event_handler(Arc::new(StubHandler {
            outcome: EventHandlerOutcome::Intents(vec![intent]),
        }));
        let output =
            Controller::invoke(&controller, json!({"query": "do it"}), Some(test_session()))
                .await
                .unwrap();
        assert_eq!(output.output, "done");
        let tasks = controller.tasks().await;
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].description.as_deref(), Some("handled"));
        assert_eq!(tasks[0].metadata["intent_label"], json!("from_handler"));
    }

    #[tokio::test]
    async fn event_handler_output_short_circuits_the_turn() {
        let (controller, calls) = controller_with_executor();
        controller.set_event_handler(Arc::new(StubHandler {
            outcome: EventHandlerOutcome::Output(Box::new(ControllerOutput::new(
                "short",
                TaskStatus::Completed,
            ))),
        }));
        let output =
            Controller::invoke(&controller, json!({"query": "do it"}), Some(test_session()))
                .await
                .unwrap();
        assert_eq!(output.output, "short");
        assert_eq!(controller.tasks().await.len(), 0);
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn event_handler_ignored_falls_back() {
        let config =
            ControllerConfig::new().with_default_response(DefaultResponse::text("fallback"));
        let controller = BaseController::new(AgentCard::new("ctrl"), config);
        controller.set_event_handler(Arc::new(StubHandler {
            outcome: EventHandlerOutcome::Ignored,
        }));
        let output =
            Controller::invoke(&controller, json!({"query": "do it"}), Some(test_session()))
                .await
                .unwrap();
        assert_eq!(output.output, "fallback");
        assert_eq!(controller.tasks().await.len(), 0);
        controller.clear_event_handler();
    }

    #[tokio::test]
    async fn task_interaction_event_marks_the_task_input_required() {
        let (controller, _) = controller_with_executor();
        let session = test_session();
        Controller::invoke(
            &controller,
            json!({"query": "build it"}),
            Some(Arc::clone(&session)),
        )
        .await
        .unwrap();
        let task_id = controller.tasks().await[0].task_id.clone();
        assert!(controller.push_event(Event::task_interaction(
            task_id.clone(),
            vec![DataFrame::text("need input")],
        )));
        let output = Controller::invoke(
            &controller,
            json!({"query": ""}),
            Some(Arc::clone(&session)),
        )
        .await
        .unwrap();
        assert_eq!(output.status, TaskStatus::InputRequired);
        assert_eq!(output.output, "need input");
        assert_eq!(output.task_id.as_deref(), Some(task_id.as_str()));
        assert_eq!(
            controller.tasks().await[0].status,
            TaskStatus::InputRequired
        );
        let state = session.get_state(TASKS_STATE_KEY).await.unwrap().unwrap();
        assert_eq!(state["tasks"][0]["status"], json!("input-required"));
    }

    #[tokio::test]
    async fn pause_and_cancel_flip_the_persisted_task_status() {
        let recognizer = Arc::new(RuleIntentRecognizer::with_rules(vec![
            IntentRule::new("pause", IntentType::PauseTask, ["pause"]),
            IntentRule::new("cancel", IntentType::CancelTask, ["cancel"]),
        ]));
        let controller = BaseController::with_recognizer(
            AgentCard::new("ctrl"),
            ControllerConfig::new(),
            recognizer,
        );
        controller.add_executor("agent", Arc::new(StubExecutor::new("agent", "done").0));
        let session = test_session();
        Controller::invoke(
            &controller,
            json!({"query": "build it"}),
            Some(Arc::clone(&session)),
        )
        .await
        .unwrap();
        let task_id = controller.tasks().await[0].task_id.clone();
        let paused = control_turn(&controller, &session, &task_id, "pause it").await;
        assert_eq!(paused.status, TaskStatus::Paused);
        assert_eq!(controller.tasks().await[0].status, TaskStatus::Paused);
        let canceled = control_turn(&controller, &session, &task_id, "cancel it").await;
        assert_eq!(canceled.status, TaskStatus::Canceled);
        assert_eq!(controller.tasks().await[0].status, TaskStatus::Canceled);
        let missing = control_turn(&controller, &session, "nope", "pause it").await;
        assert_eq!(missing.status, TaskStatus::Failed);
        assert!(missing.output.contains("not found"));
    }

    #[tokio::test]
    async fn max_intents_keeps_only_the_first() {
        let controller = BaseController::with_recognizer(
            AgentCard::new("ctrl"),
            ControllerConfig::new().with_max_intents(1),
            Arc::new(CountingRecognizer {
                calls: AtomicUsize::new(1),
            }),
        );
        controller.add_executor("agent", Arc::new(StubExecutor::new("agent", "done").0));
        Controller::invoke(&controller, json!({"query": "both"}), Some(test_session()))
            .await
            .unwrap();
        assert_eq!(controller.tasks().await.len(), 1);
    }
}
