use super::*;
use crate::controller::executors::TaskContext;
use crate::controller::schema::{OutputKind, TaskResult, TaskStatus};
use futures::StreamExt;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_session::sessions::AgentSession;
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
/// Group router that records what reaches it, together with the session id
/// it was handed.
#[derive(Default)]
struct RecordingGroup {
    /// `(agent_id, event, session id)` of every point-to-point send.
    sent: Mutex<Vec<(String, Event, Option<String>)>>,
    /// `(event, session id)` of every publish.
    published: Mutex<Vec<(Event, Option<String>)>>,
}
#[async_trait]
impl GroupRouter for RecordingGroup {
    async fn send_to_agent(
        &self,
        agent_id: &str,
        event: &Event,
        session: Option<Arc<dyn Session>>,
    ) -> Result<Value> {
        self.sent.lock().push((
            agent_id.to_owned(),
            event.clone(),
            session.as_deref().map(|session| session.id().to_owned()),
        ));
        Ok(json!({"delivered": agent_id}))
    }
    async fn publish(
        &self,
        event: &Event,
        session: Option<Arc<dyn Session>>,
    ) -> Result<Vec<Value>> {
        self.published.lock().push((
            event.clone(),
            session.as_deref().map(|session| session.id().to_owned()),
        ));
        Ok(vec![json!({"subscriber": 1}), json!({"subscriber": 2})])
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
    let _chained: &IntentDetectionController = controller.set_event_handler(Arc::new(StubHandler));
    assert_eq!(
        controller.controller().task_types(),
        vec![AGENT_TASK_TYPE.to_owned()]
    );
}
#[tokio::test]
async fn group_routing_forwards_the_event_and_session_unchanged() {
    let controller =
        IntentDetectionController::new(AgentCard::new("ctrl"), ControllerConfig::new());
    assert!(
        controller.group().is_none(),
        "no group is injected by default"
    );
    let recorder = Arc::new(RecordingGroup::default());
    let group: Arc<dyn GroupRouter> = Arc::clone(&recorder) as Arc<dyn GroupRouter>;
    let _chained: &IntentDetectionController = controller.set_group(group);
    assert!(controller.group().is_some());

    let event = Event::user_input("hand off");
    let session: Option<Arc<dyn Session>> = Some(Arc::new(AgentSession::new("s1")));
    let value = controller
        .send_to_agent("a1", &event, session.clone())
        .await
        .expect("a group is installed");
    assert_eq!(value, json!({"delivered": "a1"}));
    let values = controller
        .publish(&event, session)
        .await
        .expect("a group is installed");
    assert_eq!(values.len(), 2);

    {
        let sent = recorder.sent.lock();
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].0, "a1");
        assert_eq!(sent[0].1, event, "the same event reaches the group");
        assert_eq!(sent[0].2.as_deref(), Some("s1"));
    }
    let published = recorder.published.lock();
    assert_eq!(published.len(), 1);
    assert_eq!(published[0].0, event);
    assert_eq!(published[0].1.as_deref(), Some("s1"));
}
#[tokio::test]
async fn routing_without_a_group_reports_the_inner_controllers_name() {
    let controller =
        IntentDetectionController::new(AgentCard::new("legacy"), ControllerConfig::new());
    let event = Event::user_input("hi");
    let send = controller
        .send_to_agent("a1", &event, None)
        .await
        .expect_err("no group is installed");
    assert_eq!(send.kind(), ErrorKind::Config);
    assert_eq!(
        send.message(),
        "legacy: Cannot send_to_agent('a1'). Agent is not part of a group with a controller."
    );
    let publish = controller
        .publish(&event, None)
        .await
        .expect_err("no group is installed");
    assert_eq!(publish.kind(), ErrorKind::Config);
    assert_eq!(
        publish.message(),
        "legacy: Cannot publish(). Agent is not part of a group with a controller."
    );
}
#[tokio::test]
async fn clear_group_makes_a_later_send_fail_again() {
    let controller =
        IntentDetectionController::new(AgentCard::new("ctrl"), ControllerConfig::new());
    let recorder = Arc::new(RecordingGroup::default());
    controller.set_group(Arc::clone(&recorder) as Arc<dyn GroupRouter>);
    let event = Event::user_input("hi");
    controller
        .send_to_agent("a1", &event, None)
        .await
        .expect("a group is installed");
    controller.clear_group();
    assert!(controller.group().is_none());
    let error = controller
        .send_to_agent("a1", &event, None)
        .await
        .expect_err("the group reference is gone");
    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(error.message().contains("Cannot send_to_agent('a1')"));
    assert_eq!(recorder.sent.lock().len(), 1, "nothing else was forwarded");
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
