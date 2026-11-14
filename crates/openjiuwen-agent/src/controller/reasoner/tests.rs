//! Unit tests for the task-operation reasoner.
//!
//! Every test is offline and deterministic: the control seam is a recording
//! double, the task store is a real [`TaskManager`], and the session is an
//! [`AgentSession`]; the one stream read is bounded.
use super::*;

use futures::StreamExt;
use openjiuwen_session::{AgentSession, StreamEvent};
use parking_lot::Mutex;
use serde_json::Value;
use std::time::Duration;
/// Bound on the one stream read of the suite.
const BOUND: Duration = Duration::from_millis(500);
/// Control double recording every call and answering `answer`.
struct StubControl {
    calls: Mutex<Vec<(&'static str, String)>>,
    answer: bool,
}
impl StubControl {
    fn new(answer: bool) -> Arc<Self> {
        Arc::new(Self {
            calls: Mutex::new(Vec::new()),
            answer,
        })
    }
    fn calls(&self) -> Vec<(&'static str, String)> {
        self.calls.lock().clone()
    }
    fn record(&self, method: &'static str, task_id: &str) -> bool {
        self.calls.lock().push((method, task_id.to_owned()));
        self.answer
    }
}
#[async_trait]
impl TaskControl for StubControl {
    async fn pause(&self, task_id: &str) -> bool {
        self.record("pause", task_id)
    }
    async fn cancel(&self, task_id: &str) -> bool {
        self.record("cancel", task_id)
    }
    async fn resume(&self, task_id: &str) -> bool {
        self.record("resume", task_id)
    }
}
/// A reasoner over the default configuration.
fn reasoner(control: Option<Arc<dyn TaskControl>>) -> Reasoner {
    Reasoner::new(ControllerConfig::new(), control)
}
/// A reasoner over the default configuration with `control` installed.
fn with_control(control: &Arc<StubControl>) -> Reasoner {
    reasoner(Some(Arc::clone(control) as Arc<dyn TaskControl>))
}
/// A manager holding one submitted dependency task.
fn seeded_manager() -> TaskManager {
    let manager = TaskManager::new();
    manager
        .add_task(
            Task::submitted("s1", "dep", "agent", "write the report")
                .with_input(Event::user_input("write the report"))
                .with_context_id("ctx_dep")
                .with_metadata("intent_type", json!("create_task")),
        )
        .expect("the seeded task is accepted");
    manager
}
/// A session handle.
fn session() -> Arc<dyn Session> {
    Arc::new(AgentSession::new("s1"))
}
/// A create-task intent over `"do it"`.
fn create_intent() -> Intent {
    Intent::create_task(Event::user_input("do it"), "do it")
}
#[test]
fn config_and_control_round_trip() {
    let control = StubControl::new(true);
    let slot = Arc::clone(&control) as Arc<dyn TaskControl>;
    let reasoner = Reasoner::new(ControllerConfig::new(), Some(slot));
    assert_eq!(reasoner.config(), ControllerConfig::new());
    assert!(reasoner.control().is_some());
    let changed = ControllerConfig::new()
        .with_default_task_type("workflow")
        .with_max_tasks(2);
    reasoner.set_config(changed.clone());
    assert_eq!(reasoner.config(), changed);
    reasoner.set_control(None);
    assert!(reasoner.control().is_none());
    assert_eq!(Reasoner::default().config(), ControllerConfig::default());
}
#[test]
fn plan_ops_maps_every_intent_type() {
    let reasoner = reasoner(None);
    let source = create_intent();
    let create = reasoner.plan_ops(&source);
    assert_eq!(create.len(), 1);
    assert_eq!(create[0].kind(), "create");
    assert_eq!(create[0].task_id(), None);
    match &create[0] {
        TaskOp::Create { intent, task_type } => {
            assert_eq!(intent, &source);
            assert_eq!(task_type, "agent");
        }
        other => panic!("unexpected op: {other:?}"),
    }
    let configured = Intent::create_task(Event::user_input("x"), "x").with_task_type("workflow");
    match &reasoner.plan_ops(&configured)[0] {
        TaskOp::Create { task_type, .. } => assert_eq!(task_type, "workflow"),
        other => panic!("unexpected op: {other:?}"),
    }
    let pause =
        Intent::new(IntentType::PauseTask, Event::user_input("stop")).with_target_task_id("t1");
    assert_eq!(
        reasoner.plan_ops(&pause),
        vec![TaskOp::Pause {
            task_id: "t1".to_owned()
        }]
    );
    let resume =
        Intent::new(IntentType::ResumeTask, Event::user_input("go")).with_target_task_id("t1");
    assert_eq!(
        reasoner.plan_ops(&resume),
        vec![TaskOp::Resume {
            task_id: "t1".to_owned()
        }]
    );
    let cancel =
        Intent::new(IntentType::CancelTask, Event::user_input("drop")).with_target_task_id("t1");
    assert_eq!(
        reasoner.plan_ops(&cancel),
        vec![TaskOp::Cancel {
            task_id: "t1".to_owned()
        }]
    );
    let mut continuation = Intent::new(IntentType::ContinueTask, Event::user_input("and then"));
    continuation.depend_task_id = vec!["dep".to_owned(), "  ".to_owned()];
    continuation.metadata.insert("k".to_owned(), json!(1));
    continuation.target_task_description = Some("continue it".to_owned());
    let ops = reasoner.plan_ops(&continuation);
    assert_eq!(ops.len(), 1);
    assert_eq!(ops[0].kind(), "continue");
    assert_eq!(ops[0].task_id(), Some("dep"));
    match &ops[0] {
        TaskOp::Continue {
            depend_task_id,
            metadata,
            description,
            ..
        } => {
            assert_eq!(depend_task_id, &["dep".to_owned()]);
            assert_eq!(metadata.get("k"), Some(&json!(1)));
            assert_eq!(description.as_deref(), Some("continue it"));
        }
        other => panic!("unexpected op: {other:?}"),
    }
    let mut supplement = Intent::new(IntentType::SupplementTask, Event::user_input("here"));
    supplement.target_task_id = Some("t1".to_owned());
    supplement.supplementary_info = Some("  the numbers  ".to_owned());
    assert_eq!(
        reasoner.plan_ops(&supplement),
        vec![TaskOp::Supplement {
            task_id: "t1".to_owned(),
            info: "the numbers".to_owned()
        }]
    );
    let mut modify = Intent::new(IntentType::ModifyTask, Event::user_input("change"));
    modify.target_task_id = Some("t1".to_owned());
    modify.modification_details = Some("use euros".to_owned());
    assert_eq!(
        reasoner.plan_ops(&modify),
        vec![TaskOp::Modify {
            task_id: "t1".to_owned(),
            details: "use euros".to_owned()
        }]
    );
}
#[test]
fn switch_task_cancels_then_creates() {
    let reasoner = reasoner(None);
    let switch = Intent::new(IntentType::SwitchTask, Event::user_input("do this instead"))
        .with_target_task_id("t1")
        .with_description("do this instead")
        .with_name("switch");
    let ops = reasoner.plan_ops(&switch);
    assert_eq!(ops.len(), 2);
    assert_eq!(ops[0].kind(), "cancel");
    assert_eq!(ops[0].task_id(), Some("t1"));
    assert_eq!(ops[1].kind(), "create");
    match &ops[1] {
        TaskOp::Create { intent, task_type } => {
            assert_eq!(intent.intent_type, IntentType::CreateTask);
            assert_eq!(intent.description(), Some("do this instead"));
            assert_eq!(intent.label(), "switch");
            assert_eq!(task_type, "agent");
        }
        other => panic!("unexpected op: {other:?}"),
    }
    let mut untargeted = Intent::new(IntentType::SwitchTask, Event::user_input("switch to that"));
    untargeted.target_task_id = Some("   ".to_owned());
    let ops = reasoner.plan_ops(&untargeted);
    assert_eq!(ops.len(), 1, "{ops:?}");
    assert_eq!(ops[0].kind(), "create");
    match &ops[0] {
        TaskOp::Create { intent, .. } => {
            assert_eq!(intent.description(), Some("switch to that"));
        }
        other => panic!("unexpected op: {other:?}"),
    }
}
#[test]
fn missing_targets_and_payloads_become_unknown() {
    let reasoner = reasoner(None);
    for intent_type in [
        IntentType::PauseTask,
        IntentType::ResumeTask,
        IntentType::CancelTask,
    ] {
        let intent = Intent::new(intent_type, Event::user_input("hmm"));
        let ops = reasoner.plan_ops(&intent);
        assert_eq!(ops, vec![unknown_op(&intent)], "{intent_type:?}");
        assert_eq!(ops[0].kind(), "unknown");
        assert_eq!(ops[0].task_id(), None);
    }
    let blank_target =
        Intent::new(IntentType::PauseTask, Event::user_input("hmm")).with_target_task_id("   ");
    assert_eq!(reasoner.plan_ops(&blank_target)[0].kind(), "unknown");
    let mut supplement = Intent::new(IntentType::SupplementTask, Event::user_input("hmm"));
    supplement.target_task_id = Some("t1".to_owned());
    supplement.supplementary_info = Some("  ".to_owned());
    assert_eq!(reasoner.plan_ops(&supplement)[0].kind(), "unknown");
    let mut modify = Intent::new(IntentType::ModifyTask, Event::user_input("hmm"));
    modify.target_task_id = Some("t1".to_owned());
    assert_eq!(reasoner.plan_ops(&modify)[0].kind(), "unknown");
    let continuation = Intent::new(IntentType::ContinueTask, Event::user_input("and then"));
    assert_eq!(reasoner.plan_ops(&continuation)[0].kind(), "unknown");
    let unknown = Intent::unknown(Event::user_input("??"), "which report?");
    assert_eq!(
        reasoner.plan_ops(&unknown),
        vec![TaskOp::Unknown {
            prompt: "which report?".to_owned()
        }]
    );
    let bare = Intent::new(IntentType::UnknownTask, Event::user_input("??"));
    assert_eq!(
        reasoner.plan_ops(&bare),
        vec![TaskOp::Unknown {
            prompt: DEFAULT_CLARIFICATION.to_owned()
        }]
    );
}
#[test]
fn confidence_gate_degrades_to_one_unknown() {
    let reasoner = reasoner(None);
    let below = create_intent().with_confidence(0.69);
    let ops = reasoner.plan_ops(&below);
    assert_eq!(ops.len(), 1, "{ops:?}");
    assert_eq!(ops[0].kind(), "unknown");
    let at_threshold = create_intent().with_confidence(0.7);
    assert_eq!(reasoner.plan_ops(&at_threshold)[0].kind(), "create");
    let above = create_intent().with_confidence(0.95);
    assert_eq!(reasoner.plan_ops(&above)[0].kind(), "create");
    let nan = create_intent().with_confidence(f32::NAN);
    let ops = reasoner.plan_ops(&nan);
    assert_eq!(ops.len(), 1);
    assert_eq!(ops[0].kind(), "unknown");
    let switch = Intent::new(IntentType::SwitchTask, Event::user_input("switch"))
        .with_target_task_id("t1")
        .with_description("switch")
        .with_confidence(0.1);
    assert_eq!(reasoner.plan_ops(&switch), vec![unknown_op(&switch)]);
    let strict = Reasoner::new(
        ControllerConfig::new().with_intent_type_list(vec![IntentType::CreateTask]),
        None,
    );
    assert_eq!(strict.plan_ops(&below)[0].kind(), "unknown");
}
#[tokio::test]
async fn reason_records_the_clarification() {
    let reasoner = reasoner(None);
    let session = session();
    let mut events = session.subscribe();
    let intent = Intent::unknown(Event::user_input("??"), "which report?").with_confidence(0.2);
    let ops = reasoner.reason(&intent, Some(&session)).await.unwrap();
    assert_eq!(
        ops,
        vec![TaskOp::Unknown {
            prompt: "which report?".to_owned()
        }]
    );
    let state = session
        .get_state(CLARIFICATION_STATE_KEY)
        .await
        .unwrap()
        .expect("the clarification is recorded");
    assert_eq!(state["prompt"], json!("which report?"));
    assert_eq!(state["event_id"], json!(intent.event.event_id));
    let confidence = state["confidence"].as_f64().unwrap();
    assert!((confidence - 0.2).abs() < 1e-6);
    let item = match tokio::time::timeout(BOUND, events.next()).await {
        Ok(Some(Ok(StreamEvent::Item(item)))) => item,
        other => panic!("expected one frame, got {other:?}"),
    };
    assert_eq!(item.item_type, "controller_output");
    assert_eq!(item.payload["type"], json!("task_interaction"));
    assert_eq!(item.payload["data"][0]["text"], json!("which report?"));
}
#[tokio::test]
async fn reason_without_a_session_still_plans() {
    let reasoner = reasoner(None);
    let intent = Intent::new(IntentType::UnknownTask, Event::user_input("??"));
    let ops = reasoner.reason(&intent, None).await.unwrap();
    assert_eq!(
        ops,
        vec![TaskOp::Unknown {
            prompt: DEFAULT_CLARIFICATION.to_owned()
        }]
    );
    let session = session();
    let ops = reasoner
        .reason(&create_intent(), Some(&session))
        .await
        .unwrap();
    assert_eq!(ops[0].kind(), "create");
    assert!(
        session
            .get_state(CLARIFICATION_STATE_KEY)
            .await
            .unwrap()
            .is_none(),
        "a created task records no clarification"
    );
}
#[tokio::test]
async fn create_stores_a_submitted_task() {
    let reasoner = reasoner(None);
    let manager = TaskManager::new();
    let mut planner = Planner::new();
    let mut intent = create_intent();
    intent.target_task_description = Some("write the report".to_owned());
    intent.metadata.insert("k".to_owned(), json!(7));
    let ops = reasoner.plan_ops(&intent);
    let outcomes = reasoner
        .apply_task_ops(&ops, &manager, &mut planner, "s1", None)
        .await;
    assert_eq!(outcomes.len(), 1);
    let outcome = &outcomes[0];
    assert_eq!(outcome.kind, "create");
    assert!(outcome.applied, "{outcome:?}");
    assert_eq!(outcome.status, Some(TaskStatus::Submitted));
    let task_id = outcome.task_id.clone().expect("the outcome names the task");
    assert_eq!(outcome.message, format!("task {task_id} created"));
    assert!(task_id.starts_with("s1_task_"), "{task_id}");
    let task = manager.task(&task_id).expect("the task is stored");
    assert_eq!(task.status, TaskStatus::Submitted);
    assert_eq!(task.priority, 1);
    assert_eq!(task.task_type, "agent");
    assert_eq!(task.session_id, "s1");
    assert_eq!(task.context_id.as_deref(), Some(&*format!("s1_{task_id}")));
    assert_eq!(task.description.as_deref(), Some("write the report"));
    assert_eq!(task.inputs, vec![intent.event.clone()]);
    assert_eq!(task.metadata.get("k"), Some(&json!(7)));
    assert_eq!(task.metadata.get("intent_type"), None);
    let second = reasoner
        .apply_task_ops(&ops, &manager, &mut planner, "s1", None)
        .await;
    assert!(second[0].applied, "{second:?}");
    let second_id = second[0].task_id.clone().unwrap();
    assert_ne!(second_id, task_id);
    assert_eq!(manager.len(), 2);
}
#[tokio::test]
async fn create_uses_the_default_priority_and_scope() {
    let reasoner = Reasoner::new(
        ControllerConfig::new().with_default_task_type("workflow"),
        None,
    );
    let manager = TaskManager::new();
    let mut planner = Planner::new();
    let mut config = ControllerConfig::new();
    config.default_task_priority = 3;
    config.default_task_type = "workflow".to_owned();
    reasoner.set_config(config);
    let session = session();
    let outcomes = reasoner
        .apply_task_ops(
            &reasoner.plan_ops(&create_intent()),
            &manager,
            &mut planner,
            "   ",
            Some(&session),
        )
        .await;
    assert!(outcomes[0].applied, "{outcomes:?}");
    let task = manager
        .task(outcomes[0].task_id.as_deref().unwrap())
        .expect("the task is stored");
    assert_eq!(task.priority, 3);
    assert_eq!(task.task_type, "workflow");
    assert_eq!(task.session_id, "s1");
    assert_eq!(task.context_id.as_deref(), Some("s1_s1_task_1"));
}
#[tokio::test]
async fn pause_and_cancel_forward_to_the_control() {
    let control = StubControl::new(true);
    let reasoner = with_control(&control);
    let manager = seeded_manager();
    let mut planner = Planner::new();
    let mut ops = Vec::new();
    ops.extend(reasoner.plan_ops(
        &Intent::new(IntentType::PauseTask, Event::user_input("stop")).with_target_task_id("dep"),
    ));
    ops.extend(reasoner.plan_ops(
        &Intent::new(IntentType::CancelTask, Event::user_input("drop")).with_target_task_id("dep"),
    ));
    let outcomes = reasoner
        .apply_task_ops(&ops, &manager, &mut planner, "s1", None)
        .await;
    assert_eq!(outcomes.len(), 2);
    assert_eq!(outcomes[0].kind, "pause");
    assert!(outcomes[0].applied, "{outcomes:?}");
    assert_eq!(outcomes[0].status, Some(TaskStatus::Paused));
    assert_eq!(outcomes[0].message, "task dep paused");
    assert_eq!(outcomes[1].kind, "cancel");
    assert!(outcomes[1].applied, "{outcomes:?}");
    assert_eq!(outcomes[1].status, Some(TaskStatus::Canceled));
    assert_eq!(outcomes[1].message, "task dep canceled");
    assert_eq!(
        control.calls(),
        vec![("pause", "dep".to_owned()), ("cancel", "dep".to_owned())]
    );
}
#[tokio::test]
async fn control_ops_report_a_missing_seam_or_task() {
    let reasoner = reasoner(None);
    let manager = seeded_manager();
    let mut planner = Planner::new();
    let mut ops = Vec::new();
    ops.extend(reasoner.plan_ops(
        &Intent::new(IntentType::PauseTask, Event::user_input("stop")).with_target_task_id("dep"),
    ));
    ops.extend(reasoner.plan_ops(
        &Intent::new(IntentType::CancelTask, Event::user_input("drop")).with_target_task_id("nope"),
    ));
    let outcomes = reasoner
        .apply_task_ops(&ops, &manager, &mut planner, "s1", None)
        .await;
    assert!(!outcomes[0].applied);
    assert_eq!(outcomes[0].message, "no task control installed");
    assert_eq!(outcomes[0].status, None);
    assert!(!outcomes[1].applied);
    assert_eq!(outcomes[1].message, "task nope not found");
    assert_eq!(manager.task("nope"), None);
    let control = StubControl::new(false);
    let refusing = with_control(&control);
    let outcomes = refusing
        .apply_task_ops(&ops[..1], &manager, &mut planner, "s1", None)
        .await;
    assert!(!outcomes[0].applied, "{outcomes:?}");
    assert_eq!(outcomes[0].message, "task dep pause refused");
    assert_eq!(outcomes[0].status, None);
    assert_eq!(control.calls(), vec![("pause", "dep".to_owned())]);
}
#[tokio::test]
async fn resume_requires_a_paused_task() {
    let control = StubControl::new(true);
    let reasoner = with_control(&control);
    let manager = seeded_manager();
    let mut planner = Planner::new();
    let resume = reasoner.plan_ops(
        &Intent::new(IntentType::ResumeTask, Event::user_input("go")).with_target_task_id("dep"),
    );
    let refused = reasoner
        .apply_task_ops(&resume, &manager, &mut planner, "s1", None)
        .await;
    assert!(!refused[0].applied, "{refused:?}");
    assert_eq!(refused[0].message, "task dep is not paused");
    assert_eq!(refused[0].status, Some(TaskStatus::Submitted));
    assert!(control.calls().is_empty(), "the control is not consulted");
    manager.update_task_status("dep", TaskStatus::Paused, false, false, None);
    let resumed = reasoner
        .apply_task_ops(&resume, &manager, &mut planner, "s1", None)
        .await;
    assert!(resumed[0].applied, "{resumed:?}");
    assert_eq!(resumed[0].status, Some(TaskStatus::Submitted));
    assert_eq!(resumed[0].message, "task dep resumed");
    assert_eq!(control.calls(), vec![("resume", "dep".to_owned())]);
    let missing = reasoner
        .apply_task_ops(
            &reasoner.plan_ops(
                &Intent::new(IntentType::ResumeTask, Event::user_input("go"))
                    .with_target_task_id("gone"),
            ),
            &manager,
            &mut planner,
            "s1",
            None,
        )
        .await;
    assert!(!missing[0].applied);
    assert_eq!(missing[0].message, "task gone not found");
}
#[tokio::test]
async fn continue_inherits_the_dependency() {
    let reasoner = reasoner(None);
    let manager = seeded_manager();
    let mut planner = Planner::new();
    let mut continuation = Intent::new(
        IntentType::ContinueTask,
        Event::user_input("now shorten it"),
    );
    continuation.depend_task_id = vec!["dep".to_owned()];
    continuation.metadata.insert("k".to_owned(), json!("v"));
    let ops = reasoner.plan_ops(&continuation);
    let outcomes = reasoner
        .apply_task_ops(&ops, &manager, &mut planner, "s1", None)
        .await;
    let outcome = &outcomes[0];
    assert!(outcome.applied, "{outcome:?}");
    assert_eq!(outcome.status, Some(TaskStatus::Submitted));
    let new_id = outcome.task_id.clone().unwrap();
    assert_eq!(outcome.message, format!("task {new_id} continued from dep"));
    let task = manager.task(&new_id).expect("the continued task is stored");
    assert_eq!(task.parent_task_id.as_deref(), Some("dep"));
    assert_eq!(task.context_id.as_deref(), Some("ctx_dep"));
    assert_eq!(task.task_type, "agent");
    assert_eq!(task.description.as_deref(), Some("write the report"));
    assert_eq!(task.metadata.get("k"), Some(&json!("v")));
    assert_eq!(task.inputs.len(), 2, "{:?}", task.inputs);
    let dependency_inputs = manager
        .task("dep")
        .expect("the dependency is stored")
        .inputs;
    assert_eq!(task.inputs[0], dependency_inputs[0]);
    assert_eq!(task.inputs[1], continuation.event);
    let missing = reasoner
        .apply_task_ops(
            &reasoner.plan_ops(&{
                let mut intent = Intent::new(IntentType::ContinueTask, Event::user_input("more"));
                intent.depend_task_id = vec!["gone".to_owned()];
                intent
            }),
            &manager,
            &mut planner,
            "s1",
            None,
        )
        .await;
    assert!(!missing[0].applied);
    assert_eq!(missing[0].message, "task gone not found");
    assert_eq!(missing[0].task_id.as_deref(), Some("gone"));
}
#[tokio::test]
async fn supplement_pauses_and_reenqueues() {
    let control = StubControl::new(true);
    let reasoner = with_control(&control);
    let manager = seeded_manager();
    let mut planner = Planner::new();
    let mut intent = Intent::new(IntentType::SupplementTask, Event::user_input("here"));
    intent.target_task_id = Some("dep".to_owned());
    intent.supplementary_info = Some("the numbers".to_owned());
    manager.update_task_status("dep", TaskStatus::InputRequired, false, false, None);
    let mut stored = manager.task("dep").unwrap();
    stored.input_required_fields = Some(json!({"fields": []}));
    let _ = manager.update_task(stored);
    let outcomes = reasoner
        .apply_task_ops(
            &reasoner.plan_ops(&intent),
            &manager,
            &mut planner,
            "s1",
            None,
        )
        .await;
    assert!(outcomes[0].applied, "{outcomes:?}");
    assert_eq!(outcomes[0].status, Some(TaskStatus::Submitted));
    assert_eq!(outcomes[0].message, "task dep supplemented");
    assert_eq!(control.calls(), vec![("pause", "dep".to_owned())]);
    let task = manager.task("dep").unwrap();
    assert_eq!(
        task.description.as_deref(),
        Some(&*format!("write the report{SUPPLEMENT_MARKER}the numbers"))
    );
    assert_eq!(task.status, TaskStatus::Submitted);
    assert_eq!(manager.len(), 1, "no new task is created");
}
#[tokio::test]
async fn modify_replaces_the_task() {
    let control = StubControl::new(true);
    let reasoner = with_control(&control);
    let manager = seeded_manager();
    let mut planner = Planner::new();
    let mut intent = Intent::new(IntentType::ModifyTask, Event::user_input("change"));
    intent.target_task_id = Some("dep".to_owned());
    intent.modification_details = Some("use euros".to_owned());
    let outcomes = reasoner
        .apply_task_ops(
            &reasoner.plan_ops(&intent),
            &manager,
            &mut planner,
            "s1",
            None,
        )
        .await;
    let outcome = &outcomes[0];
    assert!(outcome.applied, "{outcome:?}");
    assert_eq!(outcome.status, Some(TaskStatus::Submitted));
    let new_id = outcome.task_id.clone().unwrap();
    assert_ne!(new_id, "dep");
    assert_eq!(outcome.message, format!("task dep modified into {new_id}"));
    assert_eq!(control.calls(), vec![("cancel", "dep".to_owned())]);
    let task = manager.task(&new_id).expect("the replacement is stored");
    assert_eq!(task.parent_task_id.as_deref(), Some("dep"));
    assert_eq!(task.context_id.as_deref(), Some("ctx_dep"));
    assert_eq!(task.status, TaskStatus::Submitted);
    assert_eq!(
        task.description.as_deref(),
        Some(&*format!("write the report{MODIFY_MARKER}use euros"))
    );
    assert_eq!(
        task.inputs,
        manager.task("dep").expect("the original is stored").inputs
    );
    assert!(manager.task("dep").is_some(), "the original is untouched");
    let unmodified = reasoner
        .apply_task_ops(
            &reasoner.plan_ops(&{
                let mut intent = Intent::new(IntentType::ModifyTask, Event::user_input("change"));
                intent.target_task_id = Some("gone".to_owned());
                intent.modification_details = Some("x".to_owned());
                intent
            }),
            &manager,
            &mut planner,
            "s1",
            None,
        )
        .await;
    assert!(!unmodified[0].applied);
    assert_eq!(unmodified[0].message, "task gone not found");
}
#[tokio::test]
async fn unknown_mutates_nothing() {
    let reasoner = reasoner(None);
    let manager = seeded_manager();
    let mut planner = Planner::new();
    let intent = Intent::unknown(Event::user_input("??"), "which one?");
    let outcomes = reasoner
        .apply_task_ops(
            &reasoner.plan_ops(&intent),
            &manager,
            &mut planner,
            "s1",
            None,
        )
        .await;
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].kind, "unknown");
    assert!(!outcomes[0].applied);
    assert_eq!(outcomes[0].task_id, None);
    assert_eq!(outcomes[0].status, None);
    assert_eq!(outcomes[0].message, "which one?");
    assert_eq!(outcomes[0].clarification.as_deref(), Some("which one?"));
    assert_eq!(manager.len(), 1);
    assert_eq!(planner.next_seq(), 1);
    assert_eq!(
        manager.task("dep").map(|task| task.status),
        Some(TaskStatus::Submitted)
    );
}
#[tokio::test]
async fn a_duplicate_id_becomes_a_failed_outcome() {
    let reasoner = reasoner(None);
    let manager = TaskManager::new();
    manager
        .add_task(Task::submitted("s1", "s1_task_1", "agent", "occupier"))
        .unwrap();
    let mut planner = Planner::new();
    let outcomes = reasoner
        .apply_task_ops(
            &reasoner.plan_ops(&create_intent()),
            &manager,
            &mut planner,
            "s1",
            None,
        )
        .await;
    assert_eq!(outcomes.len(), 1);
    assert!(!outcomes[0].applied);
    assert_eq!(outcomes[0].status, None);
    assert_eq!(outcomes[0].task_id.as_deref(), Some("s1_task_1"));
    assert!(
        outcomes[0].message.contains("s1_task_1 already exists"),
        "{}",
        outcomes[0].message
    );
    assert_eq!(manager.len(), 1);
}
#[tokio::test]
async fn a_batch_keeps_going_after_a_failure() {
    let control = StubControl::new(true);
    let reasoner = with_control(&control);
    let manager = seeded_manager();
    let mut planner = Planner::new();
    let mut ops = reasoner.plan_ops(
        &Intent::new(IntentType::CancelTask, Event::user_input("drop")).with_target_task_id("gone"),
    );
    ops.extend(reasoner.plan_ops(&create_intent()));
    let outcomes = reasoner
        .apply_task_ops(&ops, &manager, &mut planner, "s1", None)
        .await;
    assert_eq!(outcomes.len(), 2);
    assert!(!outcomes[0].applied);
    assert!(outcomes[1].applied, "{outcomes:?}");
    assert_eq!(manager.len(), 2);
}
#[test]
fn op_helpers_report_kinds_and_ids() {
    let value: Value = json!({});
    assert_eq!(value, json!({}));
    let create = TaskOp::Create {
        intent: create_intent(),
        task_type: "agent".to_owned(),
    };
    assert_eq!(create.kind(), "create");
    assert_eq!(create.task_id(), None);
    let supplement = TaskOp::Supplement {
        task_id: "t1".to_owned(),
        info: "x".to_owned(),
    };
    assert_eq!(supplement.kind(), "supplement");
    assert_eq!(supplement.task_id(), Some("t1"));
    let modify = TaskOp::Modify {
        task_id: "t2".to_owned(),
        details: "y".to_owned(),
    };
    assert_eq!(modify.kind(), "modify");
    assert_eq!(modify.task_id(), Some("t2"));
    let continuation = TaskOp::Continue {
        depend_task_id: vec!["t3".to_owned()],
        event: Event::user_input("more"),
        metadata: BTreeMap::new(),
        description: None,
    };
    assert_eq!(continuation.kind(), "continue");
    assert_eq!(continuation.task_id(), Some("t3"));
    let empty_continuation = TaskOp::Continue {
        depend_task_id: Vec::new(),
        event: Event::user_input("more"),
        metadata: BTreeMap::new(),
        description: None,
    };
    assert_eq!(empty_continuation.task_id(), None);
    assert_eq!(
        unknown_op(&Intent::new(
            IntentType::UnknownTask,
            Event::user_input("?")
        ))
        .kind(),
        "unknown"
    );
}
