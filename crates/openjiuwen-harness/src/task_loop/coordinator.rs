//! Loop budget accounting and the stop-condition evaluator chain.
//!
//! Ports `openjiuwen/harness/task_loop/loop_coordinator.py`. The coordinator is
//! the single budget authority of the outer loop: it owns the iteration and
//! token counters, the elapsed-time reading, the last round result, the abort
//! latch and the ordered evaluator chain.
//!
//! `should_continue` is the one entry point the task-loop driver calls before
//! each round; the chain is OR-combined and the *first* verdict wins, so
//! `stop_reason` matches the Python evaluator class name that fired.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde_json::{json, Map, Value};
use tracing::debug;

use crate::schema::{
    NoProgressAnswer, StopConditionEvaluator, StopConditionState, StopEvaluationContext,
    StopVerdict, TaskLoopNoProgressGuardConfig,
};

/// Default cap on outer task-loop rounds (`DEFAULT_MAX_OUTER_ROUNDS`).
pub const DEFAULT_MAX_OUTER_ROUNDS: usize = 50;

/// Stop reason recorded when the outer-round budget is exhausted.
const MAX_OUTER_ROUNDS_REASON: &str = "MaxOuterRounds";

/// Stop reason recorded when an abort was requested.
const ABORTED_REASON: &str = "Aborted";

/// Builds the task-loop default evaluator chain from the no-progress guard config.
///
/// Python parity: the `DeepAgent` task loop installs exactly the
/// `NoProgressAnswerEvaluator` guard when `task_loop_no_progress_guard.enabled`
/// and no evaluator otherwise.
#[must_use]
pub fn default_evaluators(
    guard: &TaskLoopNoProgressGuardConfig,
) -> Vec<Box<dyn StopConditionEvaluator>> {
    if guard.enabled {
        vec![Box::new(NoProgressAnswer::new(
            guard.max_consecutive_empty_answers,
            guard.min_answer_chars,
        ))]
    } else {
        Vec::new()
    }
}

/// Budget accounting plus the stop-condition evaluator chain of the outer loop.
pub struct LoopCoordinator {
    evaluators: Vec<Box<dyn StopConditionEvaluator>>,
    max_outer_rounds: usize,
    iteration: usize,
    token_usage: u64,
    elapsed_seconds: f64,
    last_result: Option<Value>,
    stop_reason: Option<String>,
    abort: Arc<AtomicBool>,
}

impl fmt::Debug for LoopCoordinator {
    /// Prints evaluator *names* because the boxed evaluators are not `Debug`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LoopCoordinator")
            .field("evaluators", &self.evaluator_names())
            .field("max_outer_rounds", &self.max_outer_rounds)
            .field("iteration", &self.iteration)
            .field("token_usage", &self.token_usage)
            .field("elapsed_seconds", &self.elapsed_seconds)
            .field("last_result", &self.last_result)
            .field("stop_reason", &self.stop_reason)
            .field("abort", &self.is_aborted())
            .finish()
    }
}

impl LoopCoordinator {
    /// Builds a coordinator over an explicit evaluator chain.
    #[must_use]
    pub fn new(evaluators: Vec<Box<dyn StopConditionEvaluator>>) -> Self {
        Self {
            evaluators,
            max_outer_rounds: DEFAULT_MAX_OUTER_ROUNDS,
            iteration: 0,
            token_usage: 0,
            elapsed_seconds: 0.0,
            last_result: None,
            stop_reason: None,
            abort: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Builds a coordinator with the task-loop default chain for `guard`.
    #[must_use]
    pub fn with_defaults(guard: &TaskLoopNoProgressGuardConfig) -> Self {
        Self::new(default_evaluators(guard))
    }

    /// Returns this coordinator with the outer-round cap replaced.
    #[must_use]
    pub fn with_max_outer_rounds(mut self, max_outer_rounds: usize) -> Self {
        self.set_max_outer_rounds(max_outer_rounds);
        self
    }

    /// Replaces the outer-round cap.
    pub fn set_max_outer_rounds(&mut self, max_outer_rounds: usize) {
        self.max_outer_rounds = max_outer_rounds;
    }

    /// The outer-round cap (`MaxOuterRounds` stop condition).
    #[must_use]
    pub fn max_outer_rounds(&self) -> usize {
        self.max_outer_rounds
    }

    /// Decides whether another round may start, refreshing `context` first.
    ///
    /// Abort wins before any evaluation; the outer-round budget is checked
    /// next, and only then the evaluator chain runs. Returns `false` once the
    /// loop must stop and records the reason in [`Self::stop_reason`].
    pub async fn should_continue(&mut self, context: &mut StopEvaluationContext) -> bool {
        if self.is_aborted() {
            self.stop_reason = Some(ABORTED_REASON.to_string());
            debug!("task-loop stopping: abort requested");
            return false;
        }
        self.refresh(context);
        if self.iteration >= self.max_outer_rounds {
            self.stop_reason = Some(MAX_OUTER_ROUNDS_REASON.to_string());
            debug!(
                iteration = self.iteration,
                max_outer_rounds = self.max_outer_rounds,
                "task-loop stopping: outer-round budget exhausted"
            );
            return false;
        }
        self.evaluate(context).await.is_none()
    }

    /// Runs the evaluator chain and returns the first verdict.
    ///
    /// The caller's context is refreshed from the coordinator counters first, so
    /// `extra` supplied by the caller survives while loop bookkeeping stays
    /// authoritative. Evaluators are stateful: call either this or
    /// [`Self::should_continue`], never both, for one loop pass.
    pub async fn evaluate(&mut self, context: &mut StopEvaluationContext) -> Option<StopVerdict> {
        self.refresh(context);
        let mut verdict = None;
        for evaluator in &mut self.evaluators {
            verdict = evaluator.evaluate(context).await;
            if verdict.is_some() {
                break;
            }
        }
        if let Some(fired) = &verdict {
            self.stop_reason = Some(fired.reason().to_string());
            debug!(stop_reason = fired.reason(), "task-loop evaluator fired");
        }
        verdict
    }

    /// The stop reason recorded by the last stop decision.
    #[must_use]
    pub fn stop_reason(&self) -> Option<&str> {
        self.stop_reason.as_deref()
    }

    /// Latches the abort flag observed at the top of [`Self::should_continue`].
    pub fn request_abort(&self) {
        self.abort.store(true, Ordering::SeqCst);
    }

    /// Clears a previously latched abort flag.
    pub fn clear_abort(&self) {
        self.abort.store(false, Ordering::SeqCst);
    }

    /// Whether an abort has been requested.
    #[must_use]
    pub fn is_aborted(&self) -> bool {
        self.abort.load(Ordering::SeqCst)
    }

    /// Number of completed outer-loop rounds.
    #[must_use]
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Cumulative token usage across all rounds.
    #[must_use]
    pub fn token_usage(&self) -> u64 {
        self.token_usage
    }

    /// Wall-clock seconds reported by the driver for the last observed round.
    #[must_use]
    pub fn elapsed_seconds(&self) -> f64 {
        self.elapsed_seconds
    }

    /// Result payload of the most recent round, when one is available.
    #[must_use]
    pub fn last_result(&self) -> Option<&Value> {
        self.last_result.as_ref()
    }

    /// Adds one completed round to the iteration counter.
    pub fn increment_iteration(&mut self) {
        self.iteration = self.iteration.saturating_add(1);
    }

    /// Adds a token-usage delta to the cumulative counter.
    pub fn add_token_usage(&mut self, delta: u64) {
        self.token_usage = self.token_usage.saturating_add(delta);
    }

    /// Replaces the last round result fed to the evaluators.
    pub fn set_last_result(&mut self, last_result: Option<Value>) {
        self.last_result = last_result;
    }

    /// Records one completed round in a single call.
    ///
    /// Python parity: the driver's per-round bookkeeping (`iteration += 1`,
    /// `token_usage += delta`, `elapsed`, `last_result`) before evaluating.
    pub fn observe_round(
        &mut self,
        iteration: usize,
        token_usage_delta: u64,
        elapsed_seconds: f64,
        last_result: Option<Value>,
    ) {
        self.iteration = iteration;
        self.add_token_usage(token_usage_delta);
        self.elapsed_seconds = elapsed_seconds;
        self.last_result = last_result;
    }

    /// Resets the counters, the stop reason, the abort latch and the evaluators.
    pub fn reset(&mut self) {
        self.iteration = 0;
        self.token_usage = 0;
        self.elapsed_seconds = 0.0;
        self.last_result = None;
        self.stop_reason = None;
        self.clear_abort();
        for evaluator in &mut self.evaluators {
            evaluator.reset();
        }
    }

    /// Exports the aggregate stop-condition state for persistence.
    ///
    /// Per-evaluator snapshots are keyed by evaluator name, mirroring the
    /// Python `StopConditionState` payload.
    #[must_use]
    pub fn state(&self) -> StopConditionState {
        let mut evaluator_states = Map::new();
        for evaluator in &self.evaluators {
            if let Some(snapshot) = evaluator.get_state() {
                evaluator_states.insert(evaluator.name().to_string(), snapshot);
            }
        }
        let payload = json!({
            "iteration": self.iteration,
            "token_usage": self.token_usage,
            "stop_reason": self.stop_reason,
            "evaluator_states": Value::Object(evaluator_states),
        });
        StopConditionState::restore(&payload)
    }

    /// Restores counters, stop reason and per-evaluator snapshots from `state`.
    ///
    /// Evaluators whose name is absent from the snapshot keep their current
    /// internal state, matching Python's tolerant `load_state`.
    pub fn restore(&mut self, state: &StopConditionState) {
        let snapshot = state.get_state();
        let Some(data) = snapshot.as_object() else {
            return;
        };
        self.iteration = data
            .get("iteration")
            .and_then(Value::as_u64)
            .map_or(0, |value| usize::try_from(value).unwrap_or(usize::MAX));
        self.token_usage = data.get("token_usage").and_then(Value::as_u64).unwrap_or(0);
        self.stop_reason = data
            .get("stop_reason")
            .and_then(Value::as_str)
            .map(str::to_string);
        let Some(states) = data.get("evaluator_states").and_then(Value::as_object) else {
            return;
        };
        for evaluator in &mut self.evaluators {
            if let Some(snapshot) = states.get(evaluator.name()) {
                evaluator.load_state(snapshot);
            }
        }
    }

    /// Names of the installed evaluators, in chain order.
    #[must_use]
    pub fn evaluator_names(&self) -> Vec<&str> {
        self.evaluators
            .iter()
            .map(|evaluator| evaluator.name())
            .collect()
    }

    /// Copies the coordinator counters into the caller's evaluation context.
    ///
    /// [`StopEvaluationContext`] only exposes consuming `with_*` builders, so the
    /// refresh replaces the value in place while preserving `extra`.
    fn refresh(&self, context: &mut StopEvaluationContext) {
        let current = std::mem::take(context);
        *context = current
            .with_iteration(self.iteration)
            .with_token_usage(self.token_usage)
            .with_elapsed_seconds(self.elapsed_seconds)
            .with_last_result(self.last_result.clone());
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::{default_evaluators, LoopCoordinator, DEFAULT_MAX_OUTER_ROUNDS};
    use crate::schema::{
        CustomPredicate, NoProgressAnswer, StopConditionState, StopEvaluationContext,
        TaskLoopNoProgressGuardConfig,
    };
    use serde_json::{json, Value};
    use std::collections::BTreeMap;

    fn short_answer() -> Value {
        json!({"result_type": "answer", "output": "ok"})
    }

    fn long_answer() -> Value {
        json!({"result_type": "answer", "output": "an answer that is long enough to count"})
    }

    #[test]
    fn default_evaluators_follow_guard_flag() {
        let enabled = TaskLoopNoProgressGuardConfig::default();
        let evaluators = default_evaluators(&enabled);
        let names: Vec<&str> = evaluators
            .iter()
            .map(|evaluator| evaluator.name())
            .collect();
        assert_eq!(names, vec!["NoProgressAnswerEvaluator"]);

        let disabled = TaskLoopNoProgressGuardConfig {
            enabled: false,
            ..TaskLoopNoProgressGuardConfig::default()
        };
        assert!(default_evaluators(&disabled).is_empty());
    }

    #[tokio::test]
    async fn should_continue_is_true_without_verdict() {
        let mut coordinator = LoopCoordinator::new(Vec::new());
        let mut context = StopEvaluationContext::default();
        assert!(coordinator.should_continue(&mut context).await);
        assert!(coordinator.stop_reason().is_none());
    }

    #[tokio::test]
    async fn evaluator_chain_is_first_stop_wins() {
        let first = Box::new(CustomPredicate::new("FirstEvaluator", |_ctx| false));
        let second = Box::new(CustomPredicate::new("SecondEvaluator", |_ctx| true));
        let third = Box::new(CustomPredicate::new("ThirdEvaluator", |_ctx| true));
        let mut coordinator = LoopCoordinator::new(vec![first, second, third]);
        assert_eq!(
            coordinator.evaluator_names(),
            vec!["FirstEvaluator", "SecondEvaluator", "ThirdEvaluator"]
        );
        let mut context = StopEvaluationContext::default();
        assert!(!coordinator.should_continue(&mut context).await);
        assert_eq!(coordinator.stop_reason(), Some("SecondEvaluator"));

        let verdict = coordinator.evaluate(&mut context).await.expect("verdict");
        assert_eq!(verdict.reason(), "SecondEvaluator");
    }

    #[tokio::test]
    async fn max_outer_rounds_stop_reason() {
        let mut coordinator = LoopCoordinator::new(Vec::new()).with_max_outer_rounds(2);
        assert_eq!(coordinator.max_outer_rounds(), 2);
        let mut context = StopEvaluationContext::default();
        assert!(coordinator.should_continue(&mut context).await);
        coordinator.increment_iteration();
        assert!(coordinator.should_continue(&mut context).await);
        coordinator.increment_iteration();
        assert!(!coordinator.should_continue(&mut context).await);
        assert_eq!(coordinator.stop_reason(), Some("MaxOuterRounds"));

        coordinator.set_max_outer_rounds(DEFAULT_MAX_OUTER_ROUNDS);
        assert_eq!(coordinator.max_outer_rounds(), DEFAULT_MAX_OUTER_ROUNDS);
    }

    #[tokio::test]
    async fn abort_wins_without_evaluating() {
        let mut coordinator = LoopCoordinator::new(vec![Box::new(CustomPredicate::new(
            "CountingEvaluator",
            |_ctx| false,
        ))]);
        coordinator.request_abort();
        assert!(coordinator.is_aborted());
        let mut context = StopEvaluationContext::default();
        assert!(!coordinator.should_continue(&mut context).await);
        assert_eq!(coordinator.stop_reason(), Some("Aborted"));

        coordinator.clear_abort();
        assert!(!coordinator.is_aborted());
        assert!(coordinator.should_continue(&mut context).await);
        // A cleared abort does not clear the previously recorded reason.
        assert_eq!(coordinator.stop_reason(), Some("Aborted"));
    }

    #[tokio::test]
    async fn should_continue_refreshes_context_and_keeps_extra() {
        let mut coordinator = LoopCoordinator::new(Vec::new());
        coordinator.observe_round(3, 120, 1.5, Some(short_answer()));
        let mut extra = BTreeMap::new();
        extra.insert("origin".to_string(), json!("host"));
        let mut context = StopEvaluationContext::default().with_extra(extra);
        assert!(coordinator.should_continue(&mut context).await);
        assert_eq!(context.iteration(), 3);
        assert_eq!(context.token_usage(), 120);
        assert_eq!(context.elapsed_seconds(), 1.5);
        assert_eq!(context.last_result(), Some(&short_answer()));
        assert_eq!(context.extra().get("origin"), Some(&json!("host")));
        assert_eq!(coordinator.iteration(), 3);
        assert_eq!(coordinator.token_usage(), 120);
        assert_eq!(coordinator.last_result(), Some(&short_answer()));
    }

    #[tokio::test]
    async fn reset_clears_counters_reason_and_evaluator_state() {
        let guard = TaskLoopNoProgressGuardConfig {
            enabled: true,
            max_consecutive_empty_answers: 2,
            min_answer_chars: 20,
        };
        let mut coordinator = LoopCoordinator::with_defaults(&guard);
        coordinator.observe_round(1, 7, 0.25, Some(short_answer()));
        coordinator.add_token_usage(3);
        coordinator.set_last_result(Some(short_answer()));
        coordinator.request_abort();
        let mut context = StopEvaluationContext::default();
        // Abort wins at the top of `should_continue`, so the loop stops.
        assert!(!coordinator.should_continue(&mut context).await);

        coordinator.reset();
        assert_eq!(coordinator.iteration(), 0);
        assert_eq!(coordinator.token_usage(), 0);
        assert_eq!(coordinator.elapsed_seconds(), 0.0);
        assert!(coordinator.last_result().is_none());
        assert!(coordinator.stop_reason().is_none());
        assert!(!coordinator.is_aborted());

        let state = coordinator.state().get_state();
        assert_eq!(state["iteration"], json!(0));
        assert_eq!(state["token_usage"], json!(0));
        assert!(state["stop_reason"].is_null());
    }

    #[tokio::test]
    async fn no_progress_guard_triggers_after_streak() {
        let guard = TaskLoopNoProgressGuardConfig {
            enabled: true,
            max_consecutive_empty_answers: 2,
            min_answer_chars: 20,
        };
        let mut coordinator = LoopCoordinator::with_defaults(&guard);
        let mut context = StopEvaluationContext::default();
        coordinator.observe_round(1, 0, 0.0, Some(short_answer()));
        assert!(coordinator.should_continue(&mut context).await);
        coordinator.observe_round(2, 0, 0.0, Some(short_answer()));
        assert!(!coordinator.should_continue(&mut context).await);
        assert_eq!(coordinator.stop_reason(), Some("NoProgressAnswerEvaluator"));

        // A long answer resets the streak.
        coordinator.reset();
        coordinator.observe_round(1, 0, 0.0, Some(short_answer()));
        assert!(coordinator.should_continue(&mut context).await);
        coordinator.observe_round(2, 0, 0.0, Some(long_answer()));
        assert!(coordinator.should_continue(&mut context).await);
    }

    #[tokio::test]
    async fn state_round_trip_preserves_no_progress_streak() {
        let guard = TaskLoopNoProgressGuardConfig {
            enabled: true,
            max_consecutive_empty_answers: 3,
            min_answer_chars: 20,
        };
        let mut coordinator = LoopCoordinator::with_defaults(&guard);
        let mut context = StopEvaluationContext::default();
        coordinator.observe_round(1, 42, 2.0, Some(short_answer()));
        assert!(coordinator.should_continue(&mut context).await);
        let snapshot = coordinator.state();
        let encoded = snapshot.get_state();
        assert_eq!(encoded["iteration"], json!(1));
        assert_eq!(encoded["token_usage"], json!(42));

        let mut restored = LoopCoordinator::with_defaults(&guard);
        restored.restore(&StopConditionState::restore(&encoded));
        assert_eq!(restored.iteration(), 1);
        assert_eq!(restored.token_usage(), 42);
        assert_eq!(restored.state(), snapshot);

        // `last_result` is not part of `StopConditionState`, so the caller
        // re-supplies it before the refreshed context reaches the evaluators.
        restored.set_last_result(Some(short_answer()));

        // The restored streak (1) advances to 2 on the next short answer ...
        assert!(restored.should_continue(&mut context).await);
        restored.observe_round(2, 0, 0.0, Some(short_answer()));
        // ... and to 3 on the following one, firing the guard.
        assert!(!restored.should_continue(&mut context).await);
        assert_eq!(restored.stop_reason(), Some("NoProgressAnswerEvaluator"));
    }

    #[tokio::test]
    async fn state_omits_evaluators_without_snapshots() {
        let mut coordinator = LoopCoordinator::new(vec![Box::new(CustomPredicate::new(
            "StatelessEvaluator",
            |_ctx| false,
        ))]);
        let snapshot = coordinator.state().get_state();
        assert_eq!(snapshot["evaluator_states"], json!({}));
        let mut context = StopEvaluationContext::default();
        assert!(coordinator.should_continue(&mut context).await);
    }

    #[tokio::test]
    async fn restore_ignores_malformed_state() {
        let mut coordinator = LoopCoordinator::new(vec![Box::new(NoProgressAnswer::new(1, 5))]);
        coordinator.observe_round(4, 9, 0.5, None);
        coordinator.restore(&StopConditionState::restore(&json!("not-an-object")));
        // Tolerant: a malformed payload degrades to the default snapshot.
        assert_eq!(coordinator.iteration(), 0);
        assert_eq!(coordinator.token_usage(), 0);
        assert!(coordinator.stop_reason().is_none());
    }

    #[tokio::test]
    async fn debug_prints_evaluator_names() {
        let coordinator = LoopCoordinator::new(vec![Box::new(CustomPredicate::new(
            "NamedEvaluator",
            |_ctx| false,
        ))]);
        let rendered = format!("{coordinator:?}");
        assert!(rendered.contains("NamedEvaluator"));
        assert!(rendered.contains(&DEFAULT_MAX_OUTER_ROUNDS.to_string()));
    }
}
