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
//! # Group routing
//!
//! [`IntentDetectionController::set_group`],
//! [`clear_group`](IntentDetectionController::clear_group), and
//! [`group`](IntentDetectionController::group) inject, drop, and inspect the
//! group the controller's agent belongs to;
//! [`send_to_agent`](IntentDetectionController::send_to_agent) and
//! [`publish`](IntentDetectionController::publish) route through it. All five
//! delegate to [`BaseController`], so the wrapper owns no routing of its own and
//! a direct controller call behaves identically.
//!
//! Without a group, sending and publishing fail with the typed configuration
//! error of [`super::group::send_without_group`]/
//! [`super::group::publish_without_group`], where Python raises `RuntimeError`.
//! The owner name those messages interpolate is the inner controller's card name
//! (its id when the name is empty), because the group sees the wrapped
//! controller — not the legacy wrapper — as its peer.
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
//! - the `Message*` backward-compatibility aliases of `event.py`.
//! - `reasoner/agent_reasoner.py` is an empty, unreferenced Python artifact and
//!   is intentionally not ported.
pub mod reasoner;
pub mod schema;
#[cfg(test)]
mod tests;
use super::base::{BaseController, Controller, EventHandler};
use super::config::ControllerConfig;
use super::executors::{TaskExecutor, TaskRouter};
use super::group::GroupRouter;
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
    /// Injects the group this controller's agent belongs to.
    ///
    /// Python's `BaseGroup.add_agent` calls this automatically; a host owning
    /// its own group injects it explicitly. The reference is stored by the inner
    /// [`BaseController`], so the wrapper keeps no group state of its own and
    /// [`send_to_agent`](Self::send_to_agent)/[`publish`](Self::publish) route
    /// through the same group a direct controller call would.
    pub fn set_group(&self, group: Arc<dyn GroupRouter>) -> &Self {
        self.controller.set_group(group);
        self
    }
    /// Removes the group reference, so routing fails again the way it does
    /// before a group is injected.
    pub fn clear_group(&self) {
        self.controller.clear_group();
    }
    /// Returns the group router this controller's agent routes through, if any.
    #[must_use]
    pub fn group(&self) -> Option<Arc<dyn GroupRouter>> {
        self.controller.group()
    }
    /// Sends `event` to `agent_id` through the group's controller.
    ///
    /// The event, the session, and the target travel unchanged to the group, so
    /// which handoff target an event names stays the caller's choice.
    ///
    /// # Errors
    ///
    /// Returns [`Error::config`] when no group is installed; see
    /// [`BaseController::send_to_agent`] for the message, which names the inner
    /// controller's card name as the owner. Otherwise propagates whatever the
    /// group reports for an unknown target or a failing routed turn.
    pub async fn send_to_agent(
        &self,
        agent_id: &str,
        event: &Event,
        session: Option<Arc<dyn Session>>,
    ) -> Result<Value> {
        self.controller
            .send_to_agent(agent_id, event, session)
            .await
    }
    /// Publishes `event` to every subscriber through the group's controller.
    ///
    /// Dispatch (and the subscription key it matches on) belongs to the group,
    /// so this wrapper only forwards the event and the session.
    ///
    /// # Errors
    ///
    /// Returns [`Error::config`] when no group is installed; see
    /// [`BaseController::publish`] for the message, which names the inner
    /// controller's card name as the owner. Otherwise propagates whatever the
    /// group reports for a publish it cannot dispatch.
    pub async fn publish(
        &self,
        event: &Event,
        session: Option<Arc<dyn Session>>,
    ) -> Result<Vec<Value>> {
        self.controller.publish(event, session).await
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
