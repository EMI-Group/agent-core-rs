//! The legacy controller lineage: value objects, the LLM intent detector, and the
//! string-routed `IntentDetectionController`.
//!
//! Ports `openjiuwen/core/controller/legacy/` — the controller generation that
//! predates the event-queue pipeline. Two generations live side by side in this
//! crate and must not be mixed:
//!
//! - the **new pipeline** ([`crate::controller::base`], [`crate::controller::schema`],
//!   [`crate::controller::intent`], [`crate::controller::executors`]) queues
//!   [`Event`]s, recognizes [`Intent`]s, plans [`Task`](crate::controller::schema::Task)s,
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
pub mod reasoner;
pub mod schema;
use super::base::{BaseController, Controller, EventHandler};
use super::config::ControllerConfig;
use super::executors::{TaskExecutor, TaskRouter};
use super::intent::IntentRecognizer;
use super::schema::{ControllerOutput, ControllerOutputChunk, Event, Intent};
use crate::card::AgentCard;
use async_trait::async_trait;
use futures::stream::BoxStream;
use openjiuwen_common::error::Result;
use openjiuwen_session::Session;
pub use reasoner::{IntentDetector, LegacyPlanner, INTENT_DETECTION_PROMPT};
pub use schema::{
    DependencyType, EventContent, EventContext, EventPriority, EventSource, LegacyEvent,
    LegacyEventType, LegacyTask, LegacyTaskResult, LegacyTaskStatus, SourceType, TaskDependency,
    TaskInput,
};
use serde_json::Value;
use std::sync::Arc;
/// Legacy intent-detection controller (Python's `IntentDetectionController`).
///
/// Python's class owns the per-conversation in-flight task queue and cancels the
/// previous turn before handling a new message. This port keeps the public
/// entry points — recognition, executor registration, routing, and the event
/// handler hook — and delegates the turn itself to the single-turn
/// [`BaseController`] loop, which does not keep a background scheduling task.
///
/// The handle is cheap to clone and shares the inner controller's state.
#[derive(Clone)]
pub struct IntentDetectionController {
    /// The new-pipeline controller that runs the turns.
    controller: BaseController,
}
impl IntentDetectionController {
    /// Builds a controller over `card` and `config` with the default rule
    /// recognizer.
    #[must_use]
    pub fn new(card: AgentCard, config: ControllerConfig) -> Self {
        Self {
            controller: BaseController::new(card, config),
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
}
#[async_trait]
impl Controller for IntentDetectionController {
    fn card(&self) -> &AgentCard {
        self.controller.card()
    }
    fn config(&self) -> ControllerConfig {
        self.controller.config()
    }
    async fn invoke(
        &self,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> Result<ControllerOutput> {
        Controller::invoke(&self.controller, inputs, session).await
    }
    fn stream(
        &self,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> BoxStream<'static, Result<ControllerOutputChunk>> {
        Controller::stream(&self.controller, inputs, session)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::controller::executors::TaskContext;
    use crate::controller::schema::{OutputKind, Task, TaskResult, TaskStatus};
    use futures::StreamExt;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    /// Task type the test executor is registered under.
    const AGENT_TASK_TYPE: &str = "agent";
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
}
