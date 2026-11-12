//! Tests for the task-loop controller: hook order, executor outcomes, timeouts,
//! abort, follow-ups, steering, pause/resume, persistence and waiting.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use parking_lot::Mutex;
use serde_json::{json, Value};

use openjiuwen_common::error::{Error, Result};

use super::{RoundExecutor, RoundId, StateSink, TaskLoopConfig, TaskLoopController};
use crate::schema::{
    ActiveInteractionRound, DeepAgentState, InteractionPhase, RoundOutcome, RoundWorkItem,
};
use crate::task_loop::coordinator::LoopCoordinator;
use crate::task_loop::handler::TaskLoopEventHandler;
use crate::task_loop::queues::{LoopQueues, QueuedInput};

/// Executor fake: records calls, can delay, can fail and can decide next work.
#[derive(Default)]
struct FakeExecutor {
    calls: Mutex<Vec<RoundWorkItem>>,
    delay: Option<Duration>,
    failure: Option<String>,
    next_work: Mutex<Option<RoundWorkItem>>,
    repeat_next_work: Mutex<Option<RoundWorkItem>>,
    tokens: AtomicU64,
    last_result: Mutex<Option<Value>>,
}

impl FakeExecutor {
    fn with_delay(delay: Duration) -> Self {
        Self {
            delay: Some(delay),
            ..Self::default()
        }
    }

    fn with_failure(message: &str) -> Self {
        Self {
            failure: Some(message.to_string()),
            ..Self::default()
        }
    }

    fn with_next_work(work: RoundWorkItem) -> Self {
        Self {
            next_work: Mutex::new(Some(work)),
            ..Self::default()
        }
    }

    fn with_repeating_next_work(work: RoundWorkItem) -> Self {
        Self {
            repeat_next_work: Mutex::new(Some(work)),
            ..Self::default()
        }
    }

    fn with_usage(tokens: u64, last_result: Value) -> Self {
        Self {
            tokens: AtomicU64::new(tokens),
            last_result: Mutex::new(Some(last_result)),
            ..Self::default()
        }
    }

    fn call_count(&self) -> usize {
        self.calls.lock().len()
    }

    fn calls(&self) -> Vec<RoundWorkItem> {
        self.calls.lock().clone()
    }
}

#[async_trait]
impl RoundExecutor for FakeExecutor {
    async fn run_round(&self, work: RoundWorkItem) -> Result<RoundOutcome> {
        self.calls.lock().push(work);
        if let Some(delay) = self.delay {
            tokio::time::sleep(delay).await;
        }
        if let Some(message) = &self.failure {
            return Err(Error::llm(message.clone()));
        }
        let next = {
            let one_shot = self.next_work.lock().take();
            match one_shot {
                Some(work) => Some(work),
                None => self.repeat_next_work.lock().clone(),
            }
        };
        Ok(RoundOutcome::answer(next))
    }

    fn last_round_result(&self) -> Option<Value> {
        self.last_result.lock().clone()
    }

    fn token_usage(&self) -> u64 {
        self.tokens.load(Ordering::SeqCst)
    }
}

/// Handler fake: records every hook call and can fail on one named hook.
#[derive(Default)]
struct RecordingHandler {
    log: Mutex<Vec<String>>,
    fail_on: Mutex<Option<String>>,
    interactions: Mutex<Vec<QueuedInput>>,
    round_ids: Mutex<Vec<RoundId>>,
}

impl RecordingHandler {
    fn new() -> Self {
        Self::default()
    }

    fn failing_on(hook: &str) -> Self {
        Self {
            fail_on: Mutex::new(Some(hook.to_string())),
            ..Self::default()
        }
    }

    fn log(&self) -> Vec<String> {
        self.log.lock().clone()
    }

    fn interactions(&self) -> Vec<QueuedInput> {
        self.interactions.lock().clone()
    }

    fn round_ids(&self) -> Vec<RoundId> {
        self.round_ids.lock().clone()
    }

    fn record(&self, hook: &str, detail: impl Into<String>) -> Result<()> {
        let should_fail = self.fail_on.lock().as_deref() == Some(hook);
        self.log.lock().push(format!("{hook}:{}", detail.into()));
        if should_fail {
            Err(Error::validation(format!("{hook} failed")))
        } else {
            Ok(())
        }
    }
}

#[async_trait]
impl TaskLoopEventHandler for Arc<RecordingHandler> {
    async fn prepare_round(
        &self,
        round_id: RoundId,
        _round: &ActiveInteractionRound,
    ) -> Result<()> {
        self.round_ids.lock().push(round_id);
        self.record("prepare_round", round_id.to_string())
    }

    async fn handle_input(&self, round_id: RoundId, _round: &ActiveInteractionRound) -> Result<()> {
        self.record("handle_input", round_id.to_string())
    }

    async fn handle_interaction(&self, _round_id: RoundId, input: QueuedInput) -> Result<()> {
        self.interactions.lock().push(input.clone());
        self.record("handle_interaction", input.content().to_string())
    }

    async fn handle_task_completion(
        &self,
        _round_id: RoundId,
        _round: &ActiveInteractionRound,
        _outcome: &RoundOutcome,
    ) -> Result<()> {
        self.record("handle_task_completion", _round_id.to_string())
    }

    async fn handle_task_failed(
        &self,
        round_id: RoundId,
        _round: &ActiveInteractionRound,
        outcome: &RoundOutcome,
    ) -> Result<()> {
        self.record(
            "handle_task_failed",
            outcome.error_code.clone().unwrap_or_default(),
        )
    }

    async fn on_abort(&self, round_id: RoundId, _round: &ActiveInteractionRound) -> Result<()> {
        self.record("on_abort", round_id.to_string())
    }

    async fn after_round(
        &self,
        round_id: RoundId,
        _round: &ActiveInteractionRound,
        _outcome: &RoundOutcome,
    ) -> Result<()> {
        self.record("after_round", round_id.to_string())
    }
}

fn query_inputs(query: &str) -> BTreeMap<String, Value> {
    let mut inputs = BTreeMap::new();
    inputs.insert("query".to_string(), json!(query));
    inputs
}

fn direct_work(query: &str) -> RoundWorkItem {
    RoundWorkItem::user(None, query_inputs(query), false, true)
}

fn controller(
    handler: &Arc<RecordingHandler>,
    queues: LoopQueues,
    config: TaskLoopConfig,
) -> TaskLoopController {
    TaskLoopController::new(
        Box::new(Arc::clone(handler)),
        LoopCoordinator::new(Vec::new()),
        queues,
        config,
    )
}

/// Builds a counting sink that snapshots every persisted state.
fn snapshot_sink(snapshots: &Arc<Mutex<Vec<DeepAgentState>>>) -> StateSink {
    let snapshots = Arc::clone(snapshots);
    Arc::new(move |state: &DeepAgentState| {
        snapshots.lock().push(state.clone());
        Ok(())
    })
}

#[tokio::test]
async fn single_round_runs_hooks_in_order() {
    let handler = Arc::new(RecordingHandler::new());
    let mut ctrl = controller(&handler, LoopQueues::default(), TaskLoopConfig::default());
    let executor = FakeExecutor::default();

    let round_id = ctrl.submit_round(query_inputs("hello"), false, Some("task-1".to_string()));
    assert_eq!(round_id, 1);
    assert_eq!(ctrl.pending_rounds(), 1);

    let report = ctrl.run_until_idle(&executor).await;
    assert_eq!(report.rounds, 1);
    assert!(report.stop_reason.is_none());
    assert!(!report.aborted);
    assert!(!report.paused);
    assert!(!report.is_error());
    assert_eq!(
        handler.log(),
        vec![
            "prepare_round:1",
            "handle_input:1",
            "handle_task_completion:1",
            "after_round:1",
        ]
    );
    assert_eq!(executor.call_count(), 1);
    let calls = executor.calls();
    assert_eq!(calls[0].query(), Some(&json!("hello")));
    assert!(!calls[0].is_follow_up);
    assert_eq!(ctrl.iteration(), 1);
    assert_eq!(ctrl.phase(), InteractionPhase::Idle);
    assert_eq!(ctrl.pending_rounds(), 0);
    assert_eq!(ctrl.state().iteration, 1);
    assert!(ctrl.state().stop_condition_state.is_some());
}

#[tokio::test]
async fn submitted_round_ids_reach_the_hooks() {
    let handler = Arc::new(RecordingHandler::new());
    let mut ctrl = controller(&handler, LoopQueues::default(), TaskLoopConfig::default());
    let executor = FakeExecutor::default();

    let first = ctrl.submit_round(query_inputs("one"), false, None);
    let second = ctrl.submit_round(query_inputs("two"), false, None);
    assert_eq!((first, second), (1, 2));

    let report = ctrl.run_until_idle(&executor).await;
    assert_eq!(report.rounds, 2);
    assert_eq!(handler.round_ids(), vec![1, 2]);
    assert!(handler.log().contains(&"prepare_round:1".to_string()));
    assert!(handler.log().contains(&"prepare_round:2".to_string()));
}

#[tokio::test]
async fn run_round_allocates_a_fresh_id_and_skips_the_queue() {
    let handler = Arc::new(RecordingHandler::new());
    let mut ctrl = controller(&handler, LoopQueues::default(), TaskLoopConfig::default());
    let executor = FakeExecutor::default();

    let outcome = ctrl.run_round(&executor, direct_work("direct")).await;
    assert!(!outcome.is_error());
    assert_eq!(handler.round_ids(), vec![1]);
    assert_eq!(executor.call_count(), 1);
    assert_eq!(ctrl.pending_rounds(), 0);
}

#[tokio::test]
async fn prepare_round_hook_error_becomes_handler_error() {
    let handler = Arc::new(RecordingHandler::failing_on("prepare_round"));
    let mut ctrl = controller(&handler, LoopQueues::default(), TaskLoopConfig::default());
    let executor = FakeExecutor::default();
    ctrl.submit_round(query_inputs("hello"), false, None);

    let report = ctrl.run_until_idle(&executor).await;
    assert_eq!(report.rounds, 1);
    assert_eq!(report.stop_reason.as_deref(), Some("handler_error"));
    let outcome = report.last_outcome.as_ref().expect("outcome");
    assert_eq!(outcome.error_code.as_deref(), Some("handler_error"));
    assert_eq!(executor.call_count(), 0);
    assert_eq!(
        handler.log(),
        vec![
            "prepare_round:1",
            "handle_task_failed:handler_error",
            "after_round:1",
        ]
    );
}

#[tokio::test]
async fn handle_input_hook_error_skips_execution() {
    let handler = Arc::new(RecordingHandler::failing_on("handle_input"));
    let mut ctrl = controller(&handler, LoopQueues::default(), TaskLoopConfig::default());
    let executor = FakeExecutor::default();
    ctrl.submit_round(query_inputs("hello"), false, None);

    let report = ctrl.run_until_idle(&executor).await;
    let outcome = report.last_outcome.as_ref().expect("outcome");
    assert_eq!(outcome.error_code.as_deref(), Some("handler_error"));
    assert_eq!(executor.call_count(), 0);
    assert_eq!(
        handler.log(),
        vec![
            "prepare_round:1",
            "handle_input:1",
            "handle_task_failed:handler_error",
            "after_round:1",
        ]
    );
}

#[tokio::test]
async fn after_round_hook_error_is_logged_only() {
    let handler = Arc::new(RecordingHandler::failing_on("after_round"));
    let mut ctrl = controller(&handler, LoopQueues::default(), TaskLoopConfig::default());
    let executor = FakeExecutor::default();
    ctrl.submit_round(query_inputs("hello"), false, None);

    let report = ctrl.run_until_idle(&executor).await;
    assert_eq!(report.rounds, 1);
    assert!(!report.is_error());
    assert!(report.stop_reason.is_none());
    assert_eq!(executor.call_count(), 1);
    assert!(handler.log().contains(&"after_round:1".to_string()));
}

#[tokio::test]
async fn terminal_hook_error_turns_success_into_handler_error() {
    let handler = Arc::new(RecordingHandler::failing_on("handle_task_completion"));
    let mut ctrl = controller(&handler, LoopQueues::default(), TaskLoopConfig::default());
    let executor = FakeExecutor::default();
    ctrl.submit_round(query_inputs("hello"), false, None);

    let report = ctrl.run_until_idle(&executor).await;
    let outcome = report.last_outcome.as_ref().expect("outcome");
    assert_eq!(outcome.error_code.as_deref(), Some("handler_error"));
    assert_eq!(report.stop_reason.as_deref(), Some("handler_error"));
    assert!(handler.log().contains(&"after_round:1".to_string()));
}

#[tokio::test]
async fn executor_error_becomes_round_execution_error() {
    let handler = Arc::new(RecordingHandler::new());
    let mut ctrl = controller(&handler, LoopQueues::default(), TaskLoopConfig::default());
    let executor = FakeExecutor::with_failure("llm down");
    ctrl.submit_round(query_inputs("hello"), false, None);

    let report = ctrl.run_until_idle(&executor).await;
    assert_eq!(report.rounds, 1);
    assert!(report.is_error());
    assert_eq!(report.stop_reason.as_deref(), Some("round_execution_error"));
    let outcome = report.last_outcome.as_ref().expect("outcome");
    assert_eq!(outcome.error_code.as_deref(), Some("round_execution_error"));
    assert!(handler
        .log()
        .contains(&"handle_task_failed:round_execution_error".to_string()));
}

#[tokio::test]
async fn completion_timeout_latches_abort_and_stops_the_loop() {
    let handler = Arc::new(RecordingHandler::new());
    let config = TaskLoopConfig::default().with_completion_timeout(Duration::from_millis(10));
    let mut ctrl = controller(&handler, LoopQueues::default(), config);
    let executor = FakeExecutor::with_delay(Duration::from_millis(200));
    ctrl.submit_round(query_inputs("slow"), false, None);

    let report = ctrl.run_until_idle(&executor).await;
    assert_eq!(report.rounds, 1);
    assert_eq!(report.stop_reason.as_deref(), Some("completion_timeout"));
    assert!(ctrl.is_aborted());
    assert_eq!(ctrl.phase(), InteractionPhase::Terminated);
    let outcome = report.last_outcome.as_ref().expect("outcome");
    assert_eq!(outcome.error_code.as_deref(), Some("completion_timeout"));
    assert!(handler
        .log()
        .contains(&"handle_task_failed:completion_timeout".to_string()));
}

#[tokio::test]
async fn zero_completion_timeout_disables_the_timeout() {
    let handler = Arc::new(RecordingHandler::new());
    let config = TaskLoopConfig::default().with_completion_timeout(Duration::ZERO);
    let mut ctrl = controller(&handler, LoopQueues::default(), config);
    let executor = FakeExecutor::with_delay(Duration::from_millis(20));
    ctrl.submit_round(query_inputs("slow"), false, None);

    let report = ctrl.run_until_idle(&executor).await;
    assert_eq!(report.rounds, 1);
    assert!(!report.is_error());
    assert!(!ctrl.is_aborted());
}

#[tokio::test]
async fn abort_before_the_round_skips_execution() {
    let handler = Arc::new(RecordingHandler::new());
    let mut ctrl = controller(&handler, LoopQueues::default(), TaskLoopConfig::default());
    let executor = FakeExecutor::default();
    ctrl.submit_round(query_inputs("hello"), false, None);
    ctrl.request_abort();
    assert!(ctrl.is_aborted());

    let report = ctrl.run_until_idle(&executor).await;
    assert_eq!(report.rounds, 0);
    assert!(report.aborted);
    assert_eq!(report.stop_reason.as_deref(), Some("Aborted"));
    assert_eq!(executor.call_count(), 0);
    assert_eq!(ctrl.pending_rounds(), 1);
    assert_eq!(ctrl.phase(), InteractionPhase::Terminated);
    assert!(ctrl.pause().is_err());
}

#[tokio::test]
async fn aborted_round_reports_abort_outcome_and_fires_on_abort() {
    let handler = Arc::new(RecordingHandler::new());
    let mut ctrl = controller(&handler, LoopQueues::default(), TaskLoopConfig::default());
    let executor = FakeExecutor::default();
    ctrl.request_abort();

    let outcome = ctrl.run_round(&executor, direct_work("hello")).await;
    assert_eq!(outcome.error_code.as_deref(), Some("aborted"));
    assert_eq!(executor.call_count(), 0);
    assert!(handler.log().contains(&"on_abort:1".to_string()));
    assert_eq!(ctrl.phase(), InteractionPhase::Terminated);
}

#[tokio::test]
async fn follow_ups_run_one_per_round() {
    let handler = Arc::new(RecordingHandler::new());
    let queues = LoopQueues::default();
    queues.push_follow_up("f1");
    queues.push_follow_up("f2");
    let mut ctrl = controller(&handler, queues.clone(), TaskLoopConfig::default());
    let executor = FakeExecutor::default();
    ctrl.submit_round(query_inputs("first"), false, None);

    let report = ctrl.run_until_idle(&executor).await;
    assert_eq!(report.rounds, 3);
    let calls = executor.calls();
    assert_eq!(calls.len(), 3);
    assert!(!calls[0].is_follow_up);
    assert!(calls[1].is_follow_up);
    assert_eq!(calls[1].query(), Some(&json!("f1")));
    assert_eq!(calls[2].query(), Some(&json!("f2")));
    assert!(queues.is_empty());
    assert_eq!(handler.round_ids(), vec![1, 2, 3]);
}

#[tokio::test]
async fn state_sink_runs_after_every_round_with_remaining_follow_ups() {
    let handler = Arc::new(RecordingHandler::new());
    let queues = LoopQueues::default();
    queues.push_follow_up("f1");
    queues.push_follow_up("f2");
    let snapshots = Arc::new(Mutex::new(Vec::new()));
    let mut ctrl = controller(&handler, queues, TaskLoopConfig::default())
        .with_state_sink(snapshot_sink(&snapshots));
    let executor = FakeExecutor::default();
    ctrl.submit_round(query_inputs("first"), false, None);

    let report = ctrl.run_until_idle(&executor).await;
    assert_eq!(report.rounds, 3);
    let snapshots = snapshots.lock().clone();
    assert_eq!(snapshots.len(), 3);
    assert_eq!(snapshots[0].iteration, 1);
    assert_eq!(snapshots[0].pending_follow_ups, vec!["f2".to_string()]);
    assert!(snapshots[0].stop_condition_state.is_some());
    assert!(snapshots[1].pending_follow_ups.is_empty());
    assert_eq!(snapshots[2].iteration, 3);
}

#[tokio::test]
async fn persist_state_false_keeps_state_in_memory_only() {
    let handler = Arc::new(RecordingHandler::new());
    let snapshots = Arc::new(Mutex::new(Vec::new()));
    let config = TaskLoopConfig::default().with_persist_state(false);
    let mut ctrl = controller(&handler, LoopQueues::default(), config)
        .with_state_sink(snapshot_sink(&snapshots));
    let executor = FakeExecutor::default();
    ctrl.submit_round(query_inputs("hello"), false, None);

    ctrl.run_until_idle(&executor).await;
    assert!(snapshots.lock().is_empty());
    assert_eq!(ctrl.state().iteration, 1);
}

#[tokio::test]
async fn next_work_is_enqueued_and_outer_round_cap_stops_the_loop() {
    let handler = Arc::new(RecordingHandler::new());
    let config = TaskLoopConfig::default().with_max_outer_rounds(2);
    let mut ctrl = controller(&handler, LoopQueues::default(), config);
    let executor = FakeExecutor::with_repeating_next_work(RoundWorkItem::user(
        None,
        query_inputs("again"),
        false,
        true,
    ));
    ctrl.submit_round(query_inputs("first"), false, None);

    let report = ctrl.run_until_idle(&executor).await;
    assert_eq!(report.rounds, 2);
    assert_eq!(report.stop_reason.as_deref(), Some("MaxOuterRounds"));
    assert_eq!(executor.call_count(), 2);
    assert!(ctrl.pending_rounds() >= 1);
}

#[tokio::test]
async fn next_work_follow_up_returns_to_the_queue() {
    let handler = Arc::new(RecordingHandler::new());
    let queues = LoopQueues::default();
    let mut ctrl = controller(&handler, queues.clone(), TaskLoopConfig::default());
    let next = RoundWorkItem::user(None, query_inputs("again"), true, true);
    let executor = FakeExecutor::with_next_work(next);
    ctrl.submit_round(query_inputs("first"), false, None);

    let report = ctrl.run_until_idle(&executor).await;
    assert_eq!(report.rounds, 2);
    let calls = executor.calls();
    assert!(calls[1].is_follow_up);
    assert_eq!(calls[1].query(), Some(&json!("again")));
    assert!(queues.is_empty());
}

#[tokio::test]
async fn steering_is_dispatched_once_while_the_queue_keeps_its_copy() {
    let handler = Arc::new(RecordingHandler::new());
    let queues = LoopQueues::default();
    let mut ctrl = controller(&handler, queues.clone(), TaskLoopConfig::default());
    let executor = FakeExecutor::default();
    ctrl.push_steer("steer me");
    ctrl.submit_round(query_inputs("first"), false, None);

    let report = ctrl.run_until_idle(&executor).await;
    assert_eq!(report.rounds, 1);
    assert_eq!(handler.interactions().len(), 1);
    assert_eq!(handler.interactions()[0].content(), "steer me");
    assert_eq!(queues.steering_len(), 1);
    assert!(handler
        .log()
        .contains(&"handle_interaction:steer me".to_string()));

    ctrl.submit_round(query_inputs("second"), false, None);
    ctrl.run_until_idle(&executor).await;
    assert_eq!(handler.interactions().len(), 1);
    assert_eq!(queues.steering_len(), 1);
}

#[tokio::test]
async fn pause_stops_between_rounds_and_resume_continues() {
    let handler = Arc::new(RecordingHandler::new());
    let mut ctrl = controller(&handler, LoopQueues::default(), TaskLoopConfig::default());
    let executor = FakeExecutor::default();
    ctrl.submit_round(query_inputs("hello"), false, None);

    ctrl.pause().expect("pause");
    assert!(ctrl.is_paused());
    let report = ctrl.run_until_idle(&executor).await;
    assert!(report.paused);
    assert_eq!(report.rounds, 0);
    assert_eq!(report.stop_reason.as_deref(), Some("paused"));
    assert!(!report.aborted);
    assert_eq!(ctrl.pending_rounds(), 1);
    assert_eq!(executor.call_count(), 0);

    ctrl.resume().expect("resume");
    assert!(!ctrl.is_paused());
    let report = ctrl.run_until_idle(&executor).await;
    assert_eq!(report.rounds, 1);
    assert!(!report.paused);
}

#[tokio::test]
async fn pause_and_resume_reject_a_terminated_controller() {
    let handler = Arc::new(RecordingHandler::new());
    let mut ctrl = controller(&handler, LoopQueues::default(), TaskLoopConfig::default());
    let executor = FakeExecutor::default();
    ctrl.submit_round(query_inputs("hello"), false, None);

    ctrl.cancel();
    assert!(ctrl.is_cancelled());
    assert_eq!(ctrl.phase(), InteractionPhase::Terminated);
    assert!(ctrl.pause().is_err());
    assert!(ctrl.resume().is_err());

    let report = ctrl.run_until_idle(&executor).await;
    assert!(report.aborted);
    assert_eq!(report.rounds, 0);
    assert_eq!(report.stop_reason.as_deref(), Some("Aborted"));
    assert_eq!(executor.call_count(), 0);
}

#[tokio::test]
async fn wait_round_completion_reports_no_active_round_then_last_outcome() {
    let handler = Arc::new(RecordingHandler::new());
    let mut ctrl = controller(&handler, LoopQueues::default(), TaskLoopConfig::default());
    let executor = FakeExecutor::default();

    let idle = ctrl.wait_round_completion(None).await;
    assert_eq!(idle.error_code.as_deref(), Some("no_active_round"));
    assert_eq!(idle.error_message.as_deref(), Some("no active round"));

    ctrl.submit_round(query_inputs("hello"), false, None);
    let report = ctrl.run_until_idle(&executor).await;
    let last = ctrl.wait_round_completion(None).await;
    assert_eq!(Some(last), report.last_outcome);
}

#[tokio::test]
async fn wait_round_completion_resolves_on_notification() {
    let handler = Arc::new(RecordingHandler::new());
    let mut ctrl = controller(&handler, LoopQueues::default(), TaskLoopConfig::default());
    let executor = FakeExecutor::default();
    ctrl.submit_round(query_inputs("hello"), false, None);
    ctrl.run_until_idle(&executor).await;

    // Simulate an in-flight round whose outcome is already recorded.
    *ctrl.phase.lock() = InteractionPhase::Running;
    let expected = ctrl.last_outcome_snapshot().expect("recorded outcome");
    let notify = Arc::clone(&ctrl.completion);
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(5)).await;
        notify.notify_one();
    });

    let outcome = ctrl.wait_round_completion(None).await;
    assert_eq!(outcome, expected);
}

#[tokio::test]
async fn wait_round_completion_timeout_cancels_the_loop() {
    let handler = Arc::new(RecordingHandler::new());
    let ctrl = controller(&handler, LoopQueues::default(), TaskLoopConfig::default());
    *ctrl.phase.lock() = InteractionPhase::Running;

    let outcome = ctrl
        .wait_round_completion(Some(Duration::from_millis(10)))
        .await;
    assert_eq!(outcome.error_code.as_deref(), Some("completion_timeout"));
    assert!(ctrl.is_cancelled());
    assert_eq!(ctrl.phase(), InteractionPhase::Terminated);
}

#[tokio::test]
async fn resume_continuation_stamps_continuation_rounds() {
    let handler = Arc::new(RecordingHandler::new());
    let config = TaskLoopConfig::default().with_resume_continuation(true);
    let mut ctrl = controller(&handler, LoopQueues::default(), config);
    let executor = FakeExecutor::default();
    ctrl.submit_round(query_inputs("first"), false, None);
    ctrl.submit_round(query_inputs("second"), false, None);

    let report = ctrl.run_until_idle(&executor).await;
    assert_eq!(report.rounds, 2);
    let calls = executor.calls();
    assert!(calls[0].context.get("resume_continuation").is_none());
    assert_eq!(
        calls[1].context.get("resume_continuation"),
        Some(&json!(true))
    );
}

#[tokio::test]
async fn set_state_is_replaced_by_loop_bookkeeping() {
    let handler = Arc::new(RecordingHandler::new());
    let mut seeded = DeepAgentState::default();
    seeded.iteration = 7;
    seeded.pending_follow_ups = vec!["queued".to_string()];
    let mut ctrl = controller(&handler, LoopQueues::default(), TaskLoopConfig::default())
        .with_state(seeded.clone());
    assert_eq!(ctrl.state(), seeded);

    let executor = FakeExecutor::default();
    ctrl.submit_round(query_inputs("hello"), false, None);
    ctrl.run_until_idle(&executor).await;
    assert_eq!(ctrl.state().iteration, 1);
    assert!(ctrl.state().pending_follow_ups.is_empty());
}

#[tokio::test]
async fn persist_propagates_sink_errors_but_the_loop_continues() {
    let handler = Arc::new(RecordingHandler::new());
    let sink: StateSink = Arc::new(|_state: &DeepAgentState| Err(Error::validation("sink down")));
    let mut ctrl = controller(&handler, LoopQueues::default(), TaskLoopConfig::default())
        .with_state_sink(sink);
    let executor = FakeExecutor::default();
    ctrl.submit_round(query_inputs("hello"), false, None);

    let report = ctrl.run_until_idle(&executor).await;
    assert_eq!(report.rounds, 1);
    assert!(!report.is_error());
    assert!(ctrl.persist().is_err());
}

#[tokio::test]
async fn queue_helpers_share_the_controller_queues() {
    let handler = Arc::new(RecordingHandler::new());
    let ctrl = controller(&handler, LoopQueues::default(), TaskLoopConfig::default());
    assert!(!ctrl.has_follow_up());

    ctrl.push_follow_up("queued");
    assert!(ctrl.has_follow_up());
    assert_eq!(ctrl.queues().follow_up_len(), 1);
    let drained = ctrl.drain_follow_up();
    assert_eq!(drained.len(), 1);
    assert_eq!(drained[0].content(), "queued");
    assert!(!ctrl.has_follow_up());
    assert_eq!(ctrl.queues().follow_up_len(), 0);
}

#[tokio::test]
async fn token_usage_is_converted_into_deltas() {
    let handler = Arc::new(RecordingHandler::new());
    let mut ctrl = controller(&handler, LoopQueues::default(), TaskLoopConfig::default());
    let executor = FakeExecutor::with_usage(120, json!({"result_type": "answer", "output": "hi"}));
    ctrl.submit_round(query_inputs("first"), false, None);
    ctrl.submit_round(query_inputs("second"), false, None);

    let report = ctrl.run_until_idle(&executor).await;
    assert_eq!(report.rounds, 2);
    let state = ctrl.state();
    let stop_state = state.stop_condition_state.expect("stop condition state");
    assert_eq!(stop_state["iteration"], json!(2));
    // The executor reports cumulative usage, so the second round adds no delta.
    assert_eq!(stop_state["token_usage"], json!(120));
}

#[tokio::test]
async fn debug_renders_loop_machinery() {
    let handler = Arc::new(RecordingHandler::new());
    let ctrl = controller(&handler, LoopQueues::default(), TaskLoopConfig::default());
    let rendered = format!("{ctrl:?}");
    assert!(rendered.contains("TaskLoopController"));
    assert!(rendered.contains("phase"));
    assert!(rendered.contains("pending_rounds"));
}
