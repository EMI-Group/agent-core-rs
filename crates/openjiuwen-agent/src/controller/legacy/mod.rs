//! The legacy controller lineage: value objects, the LLM intent detector, and the
//! string-routed `IntentDetectionController`.
//!
//! Ports `openjiuwen/core/controller/legacy/` — the controller generation that
//! predates the event-queue pipeline. Two generations live side by side in this
//! crate and must not be mixed:
//!
//! - the **new pipeline** ([`crate::controller::base`], [`crate::controller::schema`],
//!   [`crate::controller::intent`], [`crate::controller::executors`]) queues
//!   [`Event`]s, recognizes [`Intent`]s, plans [`Task`]s,
//!   dispatches them, and aggregates a [`ControllerOutput`];
//! - the **legacy lineage** (this module) keeps Python's own value objects
//!   ([`LegacyTask`], [`LegacyEvent`], …) and its LLM [`IntentDetector`] plus
//!   stub [`LegacyPlanner`], which Python's `LLMController` and
//!   `WorkflowController` subclass.
//!
//! The new pipeline supersedes this lineage: [`IntentDetectionController`] is a
//! thin adapter that exposes Python's entry points (`intent_detection`,
//! `add_executor`, `set_router`, `set_event_handler`) on top of a
//! [`BaseController`], so a legacy caller keeps its call shape while the turn
//! itself runs through the new loop.
//!
//! # Never mix the two type families
//!
//! [`LegacyTask`] is not [`crate::controller::schema::Task`] and [`LegacyEvent`]
//! is not [`crate::controller::schema::Event`]; their field sets, statuses, and
//! wire names differ. Convert explicitly at the boundary instead of serializing
//! one into the other, and keep the legacy types out of executor/planner
//! signatures, which speak the new pipeline's types.
//!
//! # Per-conversation preemption
//!
//! [`IntentDetectionController`] keeps at most one in-flight turn per
//! conversation, the way Python's class does:
//!
//! - [`Controller::invoke`] resolves the conversation from `conversation_id` or
//!   `conversation` on the inputs, then from the session id, then falls back to
//!   `"default"`, and stops whatever turn is already running for it before
//!   spawning its own;
//! - a running turn is tracked twice: in a [`TaskQueue`] entry (the join handle
//!   its cancellation awaits, the task snapshot, the cancellation token, and the
//!   `started_at` offset) and in the handler map that holds the cloneable
//!   `AbortHandle` preemption aborts;
//! - a preempted turn stops at its next await point, and the `invoke` that owned
//!   it returns [`Error::Cancelled`] (a turn that panics reports
//!   [`Error::session`] naming the conversation);
//! - [`Controller::stream`] preempts through the handler map only and stays
//!   untracked, because the returned [`BoxStream`] owns its turn and there is no
//!   `JoinHandle` for a host to abort while the caller is still reading it;
//! - [`IntentDetectionController::interrupt`] stops a conversation without
//!   starting a turn, and `task_queue`/`has_running`/`running_conversations`
//!   expose the registry to a host.
//!
//! Preemption is per conversation: a turn running for another id is left alone.
//! It is also passive without a runtime — `invoke` then runs its turn inline and
//! untracked — and an aborted turn never finishes its own session and state
//! bookkeeping, because it is dropped at the await point it was aborted at.
//!
//! # Deferred Python behaviour
//!
//! - `IntentDetector` category-to-`Task` generation (Python's
//!   `_generate_tasks_from_intent`): the detector reports the category name and
//!   the controller's recognizer/planner build the new pipeline's `Intent` and
//!   `Task`, so workflow-target selection by category is not ported.
//! - `IntentDetectionConfig`'s chat history, `category_info`, `example_content`,
//!   and `enable_input` knobs: [`IntentDetector`] takes a category list and a
//!   prompt; conversation history comes from the context engine of the caller.
//! - `ReasonerConfig`/`PlannerConfig`/`ProactiveIdentifierConfig`/
//!   `ReflectorConfig`: the proactive identifier and reflector are not ported,
//!   and the planner has no configuration beyond its task type.
//! - `BaseController.group` routing (`set_group`, `send_to_agent`, `publish`)
//!   and the `Message*` backward-compatibility aliases of `event.py`.
//! - `reasoner/agent_reasoner.py` is an empty, unreferenced Python artifact and
//!   is intentionally not ported.
pub mod reasoner;
pub mod schema;
use super::base::{BaseController, Controller, EventHandler};
use super::config::ControllerConfig;
use super::executors::{TaskExecutor, TaskRouter};
use super::intent::IntentRecognizer;
use super::queue::TaskQueue;
use super::schema::{ControllerOutput, ControllerOutputChunk, Event, Intent, Task};
use crate::card::AgentCard;
use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::FutureExt;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_session::Session;
use parking_lot::Mutex;
pub use reasoner::{IntentDetector, LegacyPlanner, INTENT_DETECTION_PROMPT};
pub use schema::{
    DependencyType, EventContent, EventContext, EventPriority, EventSource, LegacyEvent,
    LegacyEventType, LegacyTask, LegacyTaskResult, LegacyTaskStatus, SourceType, TaskDependency,
    TaskInput,
};
use serde_json::Value;
use std::collections::BTreeMap;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::time::Duration;
use tokio::task::{AbortHandle, Id};
use tokio_util::sync::CancellationToken;
use tracing::debug;

/// Conversation id a turn belongs to when neither the inputs nor the session
/// name one.
const DEFAULT_CONVERSATION_ID: &str = "default";

/// Resolves the conversation a turn belongs to.
///
/// `inputs["conversation_id"]` and `inputs["conversation"]` win (the first
/// non-empty string, Python's key order), then the session id, then
/// [`DEFAULT_CONVERSATION_ID`].
fn resolve_conversation_id(inputs: &Value, session: Option<&dyn Session>) -> String {
    for key in ["conversation_id", "conversation"] {
        if let Some(id) = inputs.get(key).and_then(Value::as_str) {
            if !id.is_empty() {
                return id.to_owned();
            }
        }
    }
    if let Some(id) = session.map(Session::id) {
        if !id.is_empty() {
            return id.to_owned();
        }
    }
    DEFAULT_CONVERSATION_ID.to_owned()
}

/// Builds the task snapshot the queue tracks for one conversation turn.
///
/// The snapshot is what a host reads back from [`TaskQueue::find`]; the turn's
/// real tasks are the planner's, so the id derives from the conversation and the
/// description is the `query` the caller sent.
fn tracked_task(conversation_id: &str, task_type: &str, inputs: &Value) -> Task {
    let mut task = Task::new(
        conversation_id,
        format!("{conversation_id}_turn"),
        task_type,
    );
    if let Some(query) = inputs.get("query").and_then(Value::as_str) {
        task = task.with_description(query);
    }
    task
}

/// Legacy intent-detection controller (Python's `IntentDetectionController`).
///
/// The controller keeps one in-flight turn per conversation: a newer `invoke`
/// (or [`Controller::stream`], or [`Self::interrupt`]) stops the turn already
/// running for that conversation before it proceeds, and the stopped turn
/// reports [`Error::Cancelled`]. The turn itself still runs through the
/// single-turn [`BaseController`] loop; preemption is layered on top of it, and
/// the public entry points keep the Python call shape.
///
/// The handle is cheap to clone and shares the inner controller, the turn
/// registry, and the handler map.
#[derive(Clone)]
pub struct IntentDetectionController {
    /// The new-pipeline controller that runs the turns.
    controller: BaseController,
    /// In-flight turn of every conversation, shared by every clone.
    queue: TaskQueue,
    /// Abort handle of the spawned turn of every conversation.
    ///
    /// `JoinHandle` is not `Clone` (its join is owned by [`Self::queue`]), so the
    /// map keeps the cloneable abort capability tokio hands out for exactly this
    /// purpose.
    handlers: Arc<Mutex<BTreeMap<String, AbortHandle>>>,
}
impl IntentDetectionController {
    /// Builds a controller over `card` and `config` with the default rule
    /// recognizer.
    #[must_use]
    pub fn new(card: AgentCard, config: ControllerConfig) -> Self {
        Self {
            controller: BaseController::new(card, config),
            queue: TaskQueue::new(),
            handlers: Arc::new(Mutex::new(BTreeMap::new())),
        }
    }
    /// Builds a controller with a custom intent recognizer.
    ///
    /// This is where an [`IntentDetector`]-backed recognizer is plugged in: the
    /// detector classifies text, the recognizer turns the class into intents.
    #[must_use]
    pub fn with_recognizer(
        card: AgentCard,
        config: ControllerConfig,
        recognizer: Arc<dyn IntentRecognizer>,
    ) -> Self {
        Self {
            controller: BaseController::with_recognizer(card, config, recognizer),
            queue: TaskQueue::new(),
            handlers: Arc::new(Mutex::new(BTreeMap::new())),
        }
    }
    /// Returns the inner controller, for the seams this wrapper does not
    /// forward (state persistence, event queue inspection, session binding).
    #[must_use]
    pub fn controller(&self) -> &BaseController {
        &self.controller
    }
    /// Registers `executor` under `task_type`.
    pub fn add_executor(&self, task_type: &str, executor: Arc<dyn TaskExecutor>) -> &Self {
        self.controller.add_executor(task_type, executor);
        self
    }
    /// Installs the task-type router consulted when several executors exist.
    pub fn set_router(&self, router: Arc<dyn TaskRouter>) -> &Self {
        self.controller.set_router(router);
        self
    }
    /// Installs the event handler that replaces the built-in recognizer path.
    pub fn set_event_handler(&self, handler: Arc<dyn EventHandler>) -> &Self {
        self.controller.set_event_handler(handler);
        self
    }
    /// Returns a clone of the per-conversation registry of in-flight turns.
    ///
    /// The clone shares the registry with the controller, so a host can watch it
    /// (or drive [`TaskQueue::cancel_running`]) while the controller keeps
    /// turning.
    #[must_use]
    pub fn task_queue(&self) -> TaskQueue {
        self.queue.clone()
    }
    /// Returns whether a turn is tracked for `conversation_id`.
    ///
    /// A tracked turn may already be finishing; its entry disappears once the
    /// `invoke` that owns it returns.
    #[must_use]
    pub fn has_running(&self, conversation_id: &str) -> bool {
        self.queue.has_running(conversation_id)
    }
    /// Returns the conversation ids with a tracked turn, in sorted order.
    #[must_use]
    pub fn running_conversations(&self) -> Vec<String> {
        self.queue.running_conversations()
    }
    /// Stops the in-flight turn of `conversation_id` without starting a new one.
    ///
    /// Returns `false` when the conversation has no live turn (nothing tracked,
    /// or the tracked handle already finished) and `true` when one was stopped.
    /// The stopped turn's `invoke` returns [`Error::Cancelled`], and the
    /// conversation is left idle, ready for its next turn.
    pub async fn interrupt(&self, conversation_id: &str) -> bool {
        self.preempt(conversation_id).await
    }
    /// Runs the configured recognizer over one event.
    ///
    /// Python's `intent_detection` hook is the recognition step of the loop;
    /// overriding it is how a subclass changes routing. Here the hook is a
    /// method on the controller value, and the loop runs the same recognizer.
    ///
    /// # Errors
    ///
    /// Propagates the recognizer's own failure (a recognizer that cannot
    /// inspect the event at all).
    pub async fn intent_detection(&self, event: &Event) -> Result<Vec<Intent>> {
        self.controller.intent_recognizer().recognize(event).await
    }
    /// Stops the in-flight turn of `conversation_id` and forgets both of its
    /// registrations.
    ///
    /// The tracked handle is aborted first (fire-and-forget, exactly like
    /// Python's `old_handler.cancel()`), then the queue entry is cancelled and
    /// awaited so the conversation is idle by the time this returns.
    /// `cancel_running` deliberately leaves its entry in place for its owner, so
    /// the entry is unregistered here too: the owner is the turn being stopped.
    async fn preempt(&self, conversation_id: &str) -> bool {
        let aborted = self.abort_turn(conversation_id);
        let cancelled = self.queue.cancel_running(conversation_id).await;
        self.queue.unregister(conversation_id);
        aborted || cancelled
    }
    /// Stops the in-flight turn of `conversation_id` without awaiting anything.
    ///
    /// `stream` cannot await while it builds its [`BoxStream`], so it aborts the
    /// handler (the abort is what actually stops the turn), cancels the queue's
    /// token, and drops the entry instead of joining it.
    fn preempt_now(&self, conversation_id: &str) {
        self.abort_turn(conversation_id);
        if let Some(running) = self.queue.find(conversation_id) {
            running.token().cancel();
        }
        self.queue.unregister(conversation_id);
    }
    /// Removes and aborts the tracked turn of `conversation_id`; returns whether
    /// a live turn was aborted.
    fn abort_turn(&self, conversation_id: &str) -> bool {
        self.handlers
            .lock()
            .remove(conversation_id)
            .is_some_and(|handle| {
                if handle.is_finished() {
                    return false;
                }
                debug!(
                    conversation_id,
                    "IntentDetectionController: aborting the in-flight turn"
                );
                handle.abort();
                true
            })
    }
    /// Forgets this turn, unless a newer turn of the conversation already
    /// replaced it.
    ///
    /// The handler entry is dropped only while it still holds `turn_id`, and the
    /// queue entry only while it still starts at `started_at`: a preempting turn
    /// registers its own entries before the preempted turn unwinds, and those
    /// must survive.
    fn forget_turn(&self, conversation_id: &str, turn_id: Id, started_at: Duration) {
        {
            let mut handlers = self.handlers.lock();
            if handlers
                .get(conversation_id)
                .is_some_and(|handle| handle.id() == turn_id)
            {
                handlers.remove(conversation_id);
            }
        }
        if self
            .queue
            .find(conversation_id)
            .is_some_and(|running| running.started_at() == started_at)
        {
            self.queue.unregister(conversation_id);
        }
    }
}
#[async_trait]
impl Controller for IntentDetectionController {
    fn card(&self) -> &AgentCard {
        self.controller.card()
    }
    fn config(&self) -> ControllerConfig {
        self.controller.config()
    }
    /// Runs one turn, preempting the conversation's in-flight turn first.
    ///
    /// The conversation is resolved from the inputs, then the session, then
    /// `"default"`; its running turn (if any) is stopped, this turn is spawned
    /// and tracked, and its result is returned once the turn hands it over.
    /// Without a tokio runtime the turn runs inline and untracked.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Cancelled`] when this turn was preempted by a newer turn
    /// of the same conversation (or by [`IntentDetectionController::interrupt`])
    /// before it produced an output, [`Error::session`] naming the conversation
    /// when the spawned turn panicked, and otherwise whatever the inner turn
    /// fails with.
    async fn invoke(
        &self,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> Result<ControllerOutput> {
        let conversation_id = resolve_conversation_id(&inputs, session.as_deref());
        self.preempt(&conversation_id).await;
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            debug!(
                conversation_id,
                "IntentDetectionController: no tokio runtime, running the turn inline"
            );
            return Controller::invoke(&self.controller, inputs, session).await;
        };
        let task_type = self.config().default_task_type;
        let task = tracked_task(&conversation_id, &task_type, &inputs);
        let controller = self.controller.clone();
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let panicking = conversation_id.clone();
        let handle = runtime.spawn(async move {
            // A turn that panics reports a session error instead of vanishing:
            // nothing else can observe the failure once the join handle belongs
            // to the task queue.
            let turn = AssertUnwindSafe(Controller::invoke(&controller, inputs, session));
            let result = match turn.catch_unwind().await {
                Ok(result) => result,
                Err(_) => Err(Error::session(format!(
                    "controller turn for conversation '{panicking}' panicked"
                ))),
            };
            // The receiver is gone once this turn was preempted; the result is
            // then dropped on purpose.
            let _ = sender.send(result);
        });
        let turn_id = handle.id();
        let abort = handle.abort_handle();
        let started_at = self
            .queue
            .register(
                conversation_id.clone(),
                task,
                handle,
                CancellationToken::new(),
                task_type,
            )
            .started_at();
        self.handlers.lock().insert(conversation_id.clone(), abort);
        let outcome = receiver.await;
        self.forget_turn(&conversation_id, turn_id, started_at);
        // A turn that ends without handing its result over was aborted before it
        // could send: it was preempted.
        outcome.unwrap_or_else(|_| {
            debug!(conversation_id, "IntentDetectionController: turn preempted");
            Err(Error::Cancelled)
        })
    }
    /// Streams one turn, preempting the conversation's in-flight turn first.
    ///
    /// The preemption is synchronous (abort plus a forgotten queue entry) and
    /// the streaming turn itself stays untracked: the returned [`BoxStream`]
    /// owns the turn, so there is no join handle a later turn could abort.
    fn stream(
        &self,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> BoxStream<'static, Result<ControllerOutputChunk>> {
        let conversation_id = resolve_conversation_id(&inputs, session.as_deref());
        self.preempt_now(&conversation_id);
        Controller::stream(&self.controller, inputs, session)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::controller::executors::TaskContext;
    use crate::controller::schema::{OutputKind, TaskResult, TaskStatus};
    use futures::StreamExt;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration as StdDuration;
    use tokio::task::JoinHandle;
    use tokio::time::timeout;
    /// Task type the test executor is registered under.
    const AGENT_TASK_TYPE: &str = "agent";
    /// Upper bound for the awaits of a test, so a stuck task fails instead of
    /// hanging the suite.
    const BOUND: StdDuration = StdDuration::from_secs(5);
    /// How long the slow executor keeps its turn busy.
    const SLOW: StdDuration = StdDuration::from_millis(100);
    /// Executor that records its calls and answers with a fixed text.
    struct StubExecutor {
        /// Task type it handles.
        task_type: &'static str,
        /// Text it answers with.
        text: &'static str,
        /// Call counter shared with the test.
        calls: Arc<AtomicUsize>,
    }
    impl StubExecutor {
        /// Builds the executor together with its call counter.
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
        async fn execute(&self, _task: &Task, _ctx: &TaskContext<'_>) -> Result<TaskResult> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(TaskResult::text(self.text))
        }
    }
    /// Executor that stays busy for [`SLOW`] before answering, so a turn can be
    /// preempted while it runs.
    struct SlowExecutor {
        /// Call counter shared with the test.
        calls: Arc<AtomicUsize>,
    }
    impl SlowExecutor {
        /// Builds the executor together with its call counter.
        fn new() -> (Self, Arc<AtomicUsize>) {
            let calls = Arc::new(AtomicUsize::new(0));
            (
                Self {
                    calls: Arc::clone(&calls),
                },
                calls,
            )
        }
    }
    #[async_trait]
    impl TaskExecutor for SlowExecutor {
        fn task_type(&self) -> &str {
            AGENT_TASK_TYPE
        }
        async fn execute(&self, _task: &Task, _ctx: &TaskContext<'_>) -> Result<TaskResult> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(SLOW).await;
            Ok(TaskResult::text("slow"))
        }
    }
    /// Recognizer that always yields the same two intents.
    struct StubRecognizer;
    #[async_trait]
    impl IntentRecognizer for StubRecognizer {
        fn name(&self) -> &'static str {
            "stub"
        }
        async fn recognize(&self, event: &Event) -> Result<Vec<Intent>> {
            Ok(vec![
                Intent::create_task(event.clone(), "first").with_name("first"),
                Intent::create_task(event.clone(), "second").with_name("second"),
            ])
        }
    }
    /// Recognizer whose every turn fails.
    struct FailingRecognizer;
    #[async_trait]
    impl IntentRecognizer for FailingRecognizer {
        fn name(&self) -> &'static str {
            "failing"
        }
        async fn recognize(&self, _event: &Event) -> Result<Vec<Intent>> {
            Err(Error::validation("recognition failed"))
        }
    }
    /// Router with no opinion.
    struct StubRouter;
    #[async_trait]
    impl TaskRouter for StubRouter {
        fn name(&self) -> &'static str {
            "stub"
        }
        async fn select(&self, _query: &str, _intents: &[Intent]) -> Result<Option<String>> {
            Ok(None)
        }
    }
    /// Event handler that ignores every event.
    struct StubHandler;
    #[async_trait]
    impl EventHandler for StubHandler {
        fn name(&self) -> &'static str {
            "stub"
        }
        async fn handle(
            &self,
            _event: &Event,
            _ctx: &TaskContext<'_>,
        ) -> Result<super::super::base::EventHandlerOutcome> {
            Ok(super::super::base::EventHandlerOutcome::Ignored)
        }
    }
    /// Builds a controller with one registered `agent` executor.
    fn controller_with_executor() -> (IntentDetectionController, Arc<AtomicUsize>) {
        let controller =
            IntentDetectionController::new(AgentCard::new("ctrl"), ControllerConfig::new());
        let (executor, calls) = StubExecutor::new(AGENT_TASK_TYPE, "done");
        controller.add_executor(AGENT_TASK_TYPE, Arc::new(executor));
        (controller, calls)
    }
    /// Builds a controller with one slow `agent` executor.
    fn slow_controller() -> (IntentDetectionController, Arc<AtomicUsize>) {
        let controller =
            IntentDetectionController::new(AgentCard::new("ctrl"), ControllerConfig::new());
        let (executor, calls) = SlowExecutor::new();
        controller.add_executor(AGENT_TASK_TYPE, Arc::new(executor));
        (controller, calls)
    }
    /// Starts a turn on a spawned task, so a test can preempt it.
    fn spawn_turn(
        controller: &IntentDetectionController,
        query: &str,
        conversation_id: &str,
    ) -> JoinHandle<Result<ControllerOutput>> {
        let controller = controller.clone();
        let inputs = json!({"query": query, "conversation_id": conversation_id});
        tokio::spawn(async move { Controller::invoke(&controller, inputs, None).await })
    }
    /// Waits until `check` holds, polling every millisecond up to [`BOUND`].
    async fn eventually(check: impl Fn() -> bool, what: &str) {
        let waited = timeout(BOUND, async {
            while !check() {
                tokio::time::sleep(StdDuration::from_millis(1)).await;
            }
        })
        .await;
        assert!(waited.is_ok(), "timed out waiting for {what}");
    }
    #[tokio::test]
    async fn invoke_delegates_to_the_inner_controller() {
        let (controller, calls) = controller_with_executor();
        let output = Controller::invoke(&controller, json!({"query": "do it"}), None)
            .await
            .unwrap();
        assert_eq!(output.output, "done");
        assert_eq!(output.status, TaskStatus::Completed);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(controller.card().name(), "ctrl");
        assert_eq!(controller.config(), ControllerConfig::new());
        assert_eq!(
            controller.controller().task_types(),
            vec![AGENT_TASK_TYPE.to_owned()]
        );
        assert!(
            controller.task_queue().is_empty(),
            "a finished turn leaves no entry behind"
        );
        assert!(controller.handlers.lock().is_empty());
    }
    #[tokio::test]
    async fn stream_yields_a_processing_frame_and_a_final_chunk() {
        let (controller, calls) = controller_with_executor();
        let items: Vec<_> = Controller::stream(&controller, json!({"query": "do it"}), None)
            .collect()
            .await;
        let frames: Vec<_> = items.into_iter().map(Result::unwrap).collect();
        assert!(frames.len() >= 2, "processing frame plus the turn frames");
        assert_eq!(
            frames[0].payload.as_ref().map(|payload| payload.kind),
            Some(OutputKind::Processing)
        );
        let last = frames.last().expect("a final frame");
        assert!(last.last_chunk);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(
            controller.task_queue().is_empty(),
            "streaming turns stay untracked"
        );
    }
    #[tokio::test]
    async fn intent_detection_runs_the_configured_recognizer() {
        let controller = IntentDetectionController::with_recognizer(
            AgentCard::new("ctrl"),
            ControllerConfig::new(),
            Arc::new(StubRecognizer),
        );
        let intents = controller
            .intent_detection(&Event::user_input("hello"))
            .await
            .unwrap();
        assert_eq!(intents.len(), 2);
        assert_eq!(intents[0].label(), "first");
        assert_eq!(intents[1].label(), "second");
    }
    #[test]
    fn builders_chain_and_return_self() {
        let controller =
            IntentDetectionController::new(AgentCard::new("ctrl"), ControllerConfig::new());
        let (executor, _calls) = StubExecutor::new(AGENT_TASK_TYPE, "done");
        let _chained: &IntentDetectionController =
            controller.add_executor(AGENT_TASK_TYPE, Arc::new(executor));
        let _chained: &IntentDetectionController = controller.set_router(Arc::new(StubRouter));
        let _chained: &IntentDetectionController =
            controller.set_event_handler(Arc::new(StubHandler));
        assert_eq!(
            controller.controller().task_types(),
            vec![AGENT_TASK_TYPE.to_owned()]
        );
    }
    #[test]
    fn legacy_value_objects_are_reachable_from_the_module_root() {
        let event = LegacyEvent::user_input("hi");
        let planner = LegacyPlanner::new();
        let task = planner.plan(&event);
        assert_eq!(task.task_type, LegacyTask::UNDEFINED_TASK_TYPE);
        assert_eq!(task.status, LegacyTaskStatus::Pending);
        assert_eq!(LegacyEventType::UserInput.as_str(), "user_input");
        assert_eq!(EventPriority::High.value(), 3);
        assert_eq!(SourceType::default(), SourceType::User);
        assert_eq!(
            TaskDependency::new("t0").dependency_type,
            DependencyType::Sequential
        );
        assert_eq!(TaskInput::default().target_id, "");
        assert_eq!(
            LegacyTaskResult::failed("boom").error.as_deref(),
            Some("boom")
        );
        assert_ne!(INTENT_DETECTION_PROMPT, "");
        assert_eq!(IntentDetector::DEFAULT_CLASS, "unknown");
        assert_eq!(EventContent::text("q").get_query(), "q");
        assert_eq!(EventContext::default().task_id, None);
        assert_eq!(EventSource::user("c1", None).source_type, SourceType::User);
        assert_eq!(LegacyTask::new("workflow").level, 0);
    }
    #[test]
    fn conversation_ids_come_from_the_inputs_then_the_session() {
        assert_eq!(
            resolve_conversation_id(&json!({"conversation_id": "c1"}), None),
            "c1"
        );
        assert_eq!(
            resolve_conversation_id(&json!({"conversation": "c2"}), None),
            "c2"
        );
        assert_eq!(
            resolve_conversation_id(
                &json!({"conversation_id": "c1", "conversation": "c2"}),
                None
            ),
            "c1",
            "conversation_id wins over conversation"
        );
        assert_eq!(
            resolve_conversation_id(&json!({"conversation_id": ""}), None),
            DEFAULT_CONVERSATION_ID,
            "an empty id falls back"
        );
        assert_eq!(
            resolve_conversation_id(&json!({"query": "hi"}), None),
            DEFAULT_CONVERSATION_ID
        );
        assert_eq!(
            resolve_conversation_id(&json!("plain input"), None),
            DEFAULT_CONVERSATION_ID
        );
    }
    #[test]
    fn invoke_runs_inline_without_a_tokio_runtime() {
        let (controller, calls) = controller_with_executor();
        let output = futures::executor::block_on(Controller::invoke(
            &controller,
            json!({"query": "do it", "conversation_id": "c1"}),
            None,
        ))
        .unwrap();
        assert_eq!(output.output, "done");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(
            controller.task_queue().is_empty(),
            "a turn without a runtime cannot be tracked"
        );
        assert!(controller.handlers.lock().is_empty());
    }
    #[tokio::test]
    async fn invoke_preempts_the_in_flight_turn_of_the_same_conversation() {
        let (controller, calls) = slow_controller();
        let first = spawn_turn(&controller, "one", "c1");
        eventually(
            || controller.has_running("c1"),
            "the first turn to register",
        )
        .await;
        eventually(
            || calls.load(Ordering::SeqCst) >= 1,
            "the first turn to reach its executor",
        )
        .await;
        let second = Controller::invoke(
            &controller,
            json!({"query": "two", "conversation_id": "c1"}),
            None,
        )
        .await
        .unwrap();
        assert_eq!(second.output, "slow");
        assert_eq!(second.status, TaskStatus::Completed);
        let first = timeout(BOUND, first)
            .await
            .expect("the preempted turn finishes")
            .expect("the spawned turn joins");
        let error = first.expect_err("the preempted turn is cancelled");
        assert!(error.is_cancelled(), "unexpected error: {error}");
        assert!(!controller.has_running("c1"));
        assert!(controller.task_queue().is_empty());
        assert!(
            controller.handlers.lock().is_empty(),
            "both trackers are cleaned up"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }
    #[tokio::test]
    async fn a_turn_for_another_conversation_does_not_preempt() {
        let (controller, calls) = slow_controller();
        let first = spawn_turn(&controller, "one", "c1");
        eventually(
            || controller.has_running("c1"),
            "the first turn to register",
        )
        .await;
        let second = spawn_turn(&controller, "two", "c2");
        eventually(
            || controller.has_running("c2"),
            "the second turn to register",
        )
        .await;
        assert_eq!(
            controller.running_conversations(),
            vec!["c1".to_owned(), "c2".to_owned()]
        );
        let (first, second) = timeout(BOUND, async {
            (
                first.await.expect("the first turn joins"),
                second.await.expect("the second turn joins"),
            )
        })
        .await
        .expect("both turns finish");
        assert_eq!(first.expect("the first turn completes").output, "slow");
        assert_eq!(second.expect("the second turn completes").output, "slow");
        assert!(!controller.has_running("c1"));
        assert!(!controller.has_running("c2"));
        assert_eq!(controller.running_conversations(), Vec::<String>::new());
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }
    #[tokio::test]
    async fn interrupt_stops_a_running_turn_and_reports_idleness() {
        let (controller, _calls) = slow_controller();
        assert!(
            !controller.interrupt("c1").await,
            "an idle conversation has nothing to stop"
        );
        assert_eq!(controller.running_conversations(), Vec::<String>::new());
        let first = spawn_turn(&controller, "one", "c1");
        eventually(|| controller.has_running("c1"), "the turn to register").await;
        assert!(controller.interrupt("c1").await);
        let outcome = timeout(BOUND, first)
            .await
            .expect("the interrupted turn finishes")
            .expect("the spawned turn joins");
        assert!(outcome
            .expect_err("the interrupted turn is cancelled")
            .is_cancelled());
        assert!(!controller.has_running("c1"));
        assert!(controller.task_queue().is_empty());
        assert!(controller.handlers.lock().is_empty());
        assert!(
            !controller.interrupt("c1").await,
            "the conversation is idle afterwards"
        );
    }
    #[tokio::test]
    async fn a_failed_turn_leaves_no_tracker_behind() {
        let controller = IntentDetectionController::with_recognizer(
            AgentCard::new("ctrl"),
            ControllerConfig::new(),
            Arc::new(FailingRecognizer),
        );
        let error = Controller::invoke(
            &controller,
            json!({"query": "boom", "conversation_id": "c1"}),
            None,
        )
        .await
        .expect_err("the turn fails");
        assert!(matches!(error, Error::Validation(_)), "unexpected: {error}");
        assert!(!controller.has_running("c1"));
        assert!(controller.task_queue().is_empty());
        assert!(
            controller.handlers.lock().is_empty(),
            "the handler map is cleaned up after a failed turn"
        );
        assert!(!controller.interrupt("c1").await);
    }
    #[tokio::test]
    async fn stream_preempts_an_in_flight_invoke_and_stays_untracked() {
        let (controller, calls) = slow_controller();
        let first = spawn_turn(&controller, "one", "c1");
        eventually(|| controller.has_running("c1"), "the turn to register").await;
        eventually(
            || calls.load(Ordering::SeqCst) >= 1,
            "the turn to reach its executor",
        )
        .await;
        let items: Vec<_> = Controller::stream(
            &controller,
            json!({"query": "two", "conversation_id": "c1"}),
            None,
        )
        .collect()
        .await;
        let frames: Vec<_> = items.into_iter().map(Result::unwrap).collect();
        assert!(frames.len() >= 2, "processing frame plus the turn frames");
        assert_eq!(
            frames[0].payload.as_ref().map(|payload| payload.kind),
            Some(OutputKind::Processing)
        );
        assert!(frames.last().expect("a final frame").last_chunk);
        let outcome = timeout(BOUND, first)
            .await
            .expect("the preempted turn finishes")
            .expect("the spawned turn joins");
        assert!(outcome
            .expect_err("the preempted turn is cancelled")
            .is_cancelled());
        assert!(!controller.has_running("c1"));
        assert!(controller.task_queue().is_empty());
        assert!(controller.handlers.lock().is_empty());
    }
}
