//! Stop-condition schemas and evaluators for the `DeepAgent` task loop.
//!
//! Ports `openjiuwen/harness/schema/stop_condition.py`: the runtime evaluation
//! context, the strategy trait implemented by every stop condition, the
//! built-in evaluators, and the persisted stop-condition state.
//! A verdict's `reason` is the evaluator name, so logged and persisted
//! `stop_reason` values stay Python-compatible (`MaxRoundsEvaluator`,
//! `NoProgressAnswerEvaluator`, …).

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt;

/// Runtime context passed to each [`StopConditionEvaluator`].
///
/// Decoupled from the agent callback context so evaluators only depend on the
/// data they actually inspect.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct StopEvaluationContext {
    /// Number of completed outer-loop rounds.
    iteration: usize,
    /// Cumulative token usage across all rounds.
    token_usage: u64,
    /// Wall-clock seconds since the loop started.
    elapsed_seconds: f64,
    /// Result payload of the most recent round, when one is available.
    last_result: Option<Value>,
    /// Arbitrary extra data for custom evaluators.
    extra: BTreeMap<String, Value>,
}

impl StopEvaluationContext {
    /// Return a copy of this context with `iteration` replaced.
    #[must_use]
    pub fn with_iteration(mut self, iteration: usize) -> Self {
        self.iteration = iteration;
        self
    }

    /// Return a copy of this context with `token_usage` replaced.
    #[must_use]
    pub fn with_token_usage(mut self, token_usage: u64) -> Self {
        self.token_usage = token_usage;
        self
    }

    /// Return a copy of this context with `elapsed_seconds` replaced.
    #[must_use]
    pub fn with_elapsed_seconds(mut self, elapsed_seconds: f64) -> Self {
        self.elapsed_seconds = elapsed_seconds;
        self
    }

    /// Return a copy of this context with `last_result` replaced.
    #[must_use]
    pub fn with_last_result(mut self, last_result: Option<Value>) -> Self {
        self.last_result = last_result;
        self
    }

    /// Return a copy of this context with `extra` replaced.
    #[must_use]
    pub fn with_extra(mut self, extra: BTreeMap<String, Value>) -> Self {
        self.extra = extra;
        self
    }

    /// Return the number of completed outer-loop rounds.
    #[must_use]
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Return the cumulative token usage.
    #[must_use]
    pub fn token_usage(&self) -> u64 {
        self.token_usage
    }

    /// Return the wall-clock seconds since the loop started.
    #[must_use]
    pub fn elapsed_seconds(&self) -> f64 {
        self.elapsed_seconds
    }

    /// Return the result payload of the most recent round.
    #[must_use]
    pub fn last_result(&self) -> Option<&Value> {
        self.last_result.as_ref()
    }

    /// Return the arbitrary extra data for custom evaluators.
    #[must_use]
    pub fn extra(&self) -> &BTreeMap<String, Value> {
        &self.extra
    }
}

/// The verdict of a stop-condition evaluator that decided to stop the loop.
///
/// `reason` is the evaluator name (Python class name) so the persisted
/// `stop_reason` stays compatible.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct StopVerdict {
    /// Evaluator name that produced the verdict.
    reason: String,
    /// Optional human-readable explanation of the observed value.
    detail: Option<String>,
}

impl StopVerdict {
    /// Build a verdict with the given reason and no detail.
    #[must_use]
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
            detail: None,
        }
    }

    /// Attach a human-readable detail to the verdict.
    #[must_use]
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }

    /// Return the evaluator name that produced the verdict.
    #[must_use]
    pub fn reason(&self) -> &str {
        &self.reason
    }

    /// Return the optional explanation of the observed value.
    #[must_use]
    pub fn detail(&self) -> Option<&str> {
        self.detail.as_deref()
    }
}

/// Strategy interface for a single stop condition.
///
/// Implement [`evaluate`](StopConditionEvaluator::evaluate) to return a
/// [`StopVerdict`] when the outer task loop should terminate. The Python
/// `reset`/`get_state`/`load_state` hooks are kept so evaluator state can
/// survive a persisted loop cycle.
#[async_trait]
pub trait StopConditionEvaluator: Send + Sync {
    /// Evaluator name used as the stop reason.
    fn name(&self) -> &str;

    /// Evaluate the condition against the current loop context.
    ///
    /// Returns `Some(verdict)` when the loop should stop, `None` otherwise.
    async fn evaluate(&mut self, ctx: &StopEvaluationContext) -> Option<StopVerdict>;

    /// Reset internal state for a new invoke cycle.
    fn reset(&mut self) {}

    /// Export a JSON-safe state snapshot, or `None` when there is no state.
    fn get_state(&self) -> Option<Value> {
        None
    }

    /// Restore state from a previously exported snapshot.
    fn load_state(&mut self, _state: &Value) {}
}

/// Stop after a fixed number of outer-loop rounds.
///
/// Python `MaxRoundsEvaluator`: triggers when `ctx.iteration >= max_rounds`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaxRounds {
    /// Maximum number of completed rounds before stopping.
    max_rounds: usize,
}

impl MaxRounds {
    /// Build an evaluator that stops after `max_rounds` completed rounds.
    #[must_use]
    pub fn new(max_rounds: usize) -> Self {
        Self { max_rounds }
    }
}

#[async_trait]
impl StopConditionEvaluator for MaxRounds {
    fn name(&self) -> &'static str {
        "MaxRoundsEvaluator"
    }

    async fn evaluate(&mut self, ctx: &StopEvaluationContext) -> Option<StopVerdict> {
        if ctx.iteration < self.max_rounds {
            return None;
        }
        Some(StopVerdict::new(self.name()).with_detail(format!(
            "iteration {} >= max_rounds {}",
            ctx.iteration, self.max_rounds
        )))
    }
}

/// Stop when cumulative token usage reaches a budget.
///
/// Python `TokenBudgetEvaluator`: triggers when `ctx.token_usage >= max_tokens`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenBudget {
    /// Token budget across all rounds.
    max_tokens: u64,
}

impl TokenBudget {
    /// Build an evaluator with the given token budget.
    #[must_use]
    pub fn new(max_tokens: u64) -> Self {
        Self { max_tokens }
    }
}

#[async_trait]
impl StopConditionEvaluator for TokenBudget {
    fn name(&self) -> &'static str {
        "TokenBudgetEvaluator"
    }

    async fn evaluate(&mut self, ctx: &StopEvaluationContext) -> Option<StopVerdict> {
        if ctx.token_usage < self.max_tokens {
            return None;
        }
        Some(StopVerdict::new(self.name()).with_detail(format!(
            "token_usage {} >= max_tokens {}",
            ctx.token_usage, self.max_tokens
        )))
    }
}

/// Stop when the wall-clock elapsed time reaches a limit.
///
/// Python `TimeoutEvaluator`: triggers when
/// `ctx.elapsed_seconds >= timeout_seconds`.
#[derive(Debug, Clone, PartialEq)]
pub struct Timeout {
    /// Elapsed-time limit in seconds.
    timeout_seconds: f64,
}

impl Timeout {
    /// Build an evaluator with the given elapsed-time limit in seconds.
    #[must_use]
    pub fn new(timeout_seconds: f64) -> Self {
        Self { timeout_seconds }
    }
}

#[async_trait]
impl StopConditionEvaluator for Timeout {
    fn name(&self) -> &'static str {
        "TimeoutEvaluator"
    }

    async fn evaluate(&mut self, ctx: &StopEvaluationContext) -> Option<StopVerdict> {
        if ctx.elapsed_seconds < self.timeout_seconds {
            return None;
        }
        Some(StopVerdict::new(self.name()).with_detail(format!(
            "elapsed_seconds {} >= timeout_seconds {}",
            ctx.elapsed_seconds, self.timeout_seconds
        )))
    }
}

/// Stop after repeated no-tool answers that carry little or no content.
///
/// Python `NoProgressAnswerEvaluator`: counts consecutive rounds whose
/// `result_type` is `answer` and whose trimmed `output` is shorter than
/// `min_answer_chars`; any other round resets the counter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoProgressAnswer {
    /// Consecutive short answers required to trigger.
    max_consecutive_empty_answers: usize,
    /// Answers shorter than this many characters count as no progress.
    min_answer_chars: usize,
    /// Current consecutive short-answer streak.
    consecutive_empty_answers: usize,
}

impl Default for NoProgressAnswer {
    /// Python production defaults: 3 consecutive answers and 20 characters.
    fn default() -> Self {
        Self {
            max_consecutive_empty_answers: 3,
            min_answer_chars: 20,
            consecutive_empty_answers: 0,
        }
    }
}

impl NoProgressAnswer {
    /// Build an evaluator with an explicit threshold and minimum length.
    ///
    /// The consecutive-answer threshold is clamped to at least `1`, matching
    /// the Python constructor.
    #[must_use]
    pub fn new(max_consecutive_empty_answers: usize, min_answer_chars: usize) -> Self {
        Self {
            max_consecutive_empty_answers: max_consecutive_empty_answers.max(1),
            min_answer_chars,
            consecutive_empty_answers: 0,
        }
    }
}

#[async_trait]
impl StopConditionEvaluator for NoProgressAnswer {
    fn name(&self) -> &'static str {
        "NoProgressAnswerEvaluator"
    }

    async fn evaluate(&mut self, ctx: &StopEvaluationContext) -> Option<StopVerdict> {
        // Missing or non-string fields degrade to an empty type and zero
        // length, as Python's `str(...)`/`len(...)` calls do for absent values.
        let result = ctx.last_result.as_ref().and_then(Value::as_object);
        let result_type = result
            .and_then(|result| result.get("result_type"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_lowercase();
        let output_len = result
            .and_then(|result| result.get("output"))
            .and_then(Value::as_str)
            .map_or(0, |output| output.trim().chars().count());
        if result_type == "answer" && output_len < self.min_answer_chars {
            self.consecutive_empty_answers += 1;
        } else {
            self.consecutive_empty_answers = 0;
        }
        if self.consecutive_empty_answers < self.max_consecutive_empty_answers {
            return None;
        }
        Some(StopVerdict::new(self.name()).with_detail(format!(
            "consecutive_empty_answers {} >= max_consecutive_empty_answers {} \
             (output_len {}, min_answer_chars {})",
            self.consecutive_empty_answers,
            self.max_consecutive_empty_answers,
            output_len,
            self.min_answer_chars
        )))
    }

    fn reset(&mut self) {
        self.consecutive_empty_answers = 0;
    }

    fn get_state(&self) -> Option<Value> {
        Some(serde_json::json!({
            "consecutive_empty_answers": self.consecutive_empty_answers,
            "max_consecutive_empty_answers": self.max_consecutive_empty_answers,
            "min_answer_chars": self.min_answer_chars,
        }))
    }

    fn load_state(&mut self, state: &Value) {
        let Some(data) = state.as_object() else {
            return;
        };
        self.consecutive_empty_answers = data
            .get("consecutive_empty_answers")
            .and_then(Value::as_u64)
            .map_or(0, |count| usize::try_from(count).unwrap_or(usize::MAX));
        if let Some(max_consecutive) = data
            .get("max_consecutive_empty_answers")
            .and_then(Value::as_u64)
        {
            self.max_consecutive_empty_answers = usize::try_from(max_consecutive)
                .unwrap_or(usize::MAX)
                .max(1);
        }
        if let Some(min_chars) = data.get("min_answer_chars").and_then(Value::as_u64) {
            self.min_answer_chars = usize::try_from(min_chars).unwrap_or(usize::MAX);
        }
    }
}

/// Stop once a completion promise has been confirmed enough times.
///
/// Python `CompletionPromiseEvaluator`: the evaluator does not parse model
/// output itself; the task-completion rail detects the promise and calls
/// [`notify_fulfilled`](Self::notify_fulfilled) or
/// [`notify_absent`](Self::notify_absent). Confirmations must be consecutive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletionPromise {
    /// Expected promise string.
    promise: String,
    /// Consecutive confirmations required before stopping.
    required_confirmations: usize,
    /// Whether the required confirmations have been observed.
    fulfilled: bool,
    /// Text matched by the latest successful confirmation.
    matched_text: String,
    /// Current consecutive confirmation streak.
    confirmation_count: usize,
}

impl CompletionPromise {
    /// Build an evaluator for `promise` requiring `required_confirmations`
    /// consecutive confirmations (clamped to at least `1`).
    #[must_use]
    pub fn new(promise: impl Into<String>, required_confirmations: usize) -> Self {
        Self {
            promise: promise.into(),
            required_confirmations: required_confirmations.max(1),
            fulfilled: false,
            matched_text: String::new(),
            confirmation_count: 0,
        }
    }

    /// Return the expected promise string.
    #[must_use]
    pub fn promise(&self) -> &str {
        &self.promise
    }

    /// Record one confirmation of the promise.
    ///
    /// Increments the consecutive confirmation count, stores the matched text
    /// and marks the promise fulfilled once the required count is reached.
    pub fn notify_fulfilled(&mut self, matched_text: &str) {
        self.confirmation_count += 1;
        self.matched_text = matched_text.to_string();
        self.fulfilled = self.confirmation_count >= self.required_confirmations;
    }

    /// Record that the expected promise was absent.
    ///
    /// Confirmations are consecutive, so this resets the streak and clears the
    /// fulfilled flag and matched text.
    pub fn notify_absent(&mut self) {
        self.confirmation_count = 0;
        self.fulfilled = false;
        self.matched_text.clear();
    }
}

#[async_trait]
impl StopConditionEvaluator for CompletionPromise {
    fn name(&self) -> &'static str {
        "CompletionPromiseEvaluator"
    }

    async fn evaluate(&mut self, _ctx: &StopEvaluationContext) -> Option<StopVerdict> {
        if !self.fulfilled {
            return None;
        }
        Some(StopVerdict::new(self.name()).with_detail(format!(
            "completion promise {:?} confirmed {} time(s)",
            self.promise, self.confirmation_count
        )))
    }

    fn reset(&mut self) {
        self.fulfilled = false;
        self.matched_text.clear();
        self.confirmation_count = 0;
    }

    fn get_state(&self) -> Option<Value> {
        Some(serde_json::json!({
            "fulfilled": self.fulfilled,
            "matched_text": self.matched_text,
            "required_confirmations": self.required_confirmations,
            "confirmation_count": self.confirmation_count,
        }))
    }

    fn load_state(&mut self, state: &Value) {
        let Some(data) = state.as_object() else {
            return;
        };
        let persisted_fulfilled = data
            .get("fulfilled")
            .and_then(Value::as_bool)
            .unwrap_or(self.fulfilled);
        if let Some(matched_text) = data.get("matched_text").and_then(Value::as_str) {
            self.matched_text = matched_text.to_string();
        }
        if let Some(required) = data.get("required_confirmations").and_then(Value::as_u64) {
            self.required_confirmations = usize::try_from(required).unwrap_or(usize::MAX).max(1);
        }
        if let Some(count) = data.get("confirmation_count").and_then(Value::as_u64) {
            self.confirmation_count = usize::try_from(count).unwrap_or(usize::MAX);
        }
        // A persisted `true` is never downgraded to false.
        self.fulfilled =
            persisted_fulfilled || self.confirmation_count >= self.required_confirmations;
    }
}

/// Stop based on a user-supplied predicate.
///
/// Python `CustomPredicateEvaluator`: the predicate receives the full
/// [`StopEvaluationContext`], so it can inspect iteration count, token usage
/// and anything stored in `extra`. The evaluator keeps the caller-supplied
/// name, which becomes the verdict reason.
pub struct CustomPredicate {
    /// Name reported by [`StopConditionEvaluator::name`].
    name: String,
    /// Predicate deciding whether the loop should stop.
    predicate: Box<dyn Fn(&StopEvaluationContext) -> bool + Send + Sync>,
}

impl CustomPredicate {
    /// Build an evaluator from a name and a stop predicate.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        predicate: impl Fn(&StopEvaluationContext) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            predicate: Box::new(predicate),
        }
    }
}

impl fmt::Debug for CustomPredicate {
    /// Print the evaluator name without exposing the opaque predicate.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CustomPredicate")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl StopConditionEvaluator for CustomPredicate {
    fn name(&self) -> &str {
        &self.name
    }

    async fn evaluate(&mut self, ctx: &StopEvaluationContext) -> Option<StopVerdict> {
        if (self.predicate)(ctx) {
            Some(StopVerdict::new(self.name()))
        } else {
            None
        }
    }
}

/// Persisted aggregate state of the stop-condition subsystem.
///
/// Carries the last iteration/token counters, the winning stop reason and the
/// per-evaluator state snapshots keyed by evaluator name.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct StopConditionState {
    /// Number of completed outer-loop rounds.
    iteration: usize,
    /// Cumulative token usage across all rounds.
    token_usage: u64,
    /// Winning stop reason, if the loop has already stopped.
    stop_reason: Option<String>,
    /// Per-evaluator state snapshots keyed by evaluator name.
    evaluator_states: BTreeMap<String, Value>,
}

impl StopConditionState {
    /// Export the state as a JSON value.
    ///
    /// Serialization of this plain data type cannot fail; a failure degrades to
    /// [`Value::Null`] rather than panicking.
    #[must_use]
    pub fn get_state(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }

    /// Restore state from a previously exported JSON value.
    ///
    /// Unknown or malformed payloads degrade to the default state.
    #[must_use]
    pub fn restore(value: &Value) -> Self {
        serde_json::from_value(value.clone()).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::{
        CompletionPromise, CustomPredicate, MaxRounds, NoProgressAnswer, StopConditionEvaluator,
        StopConditionState, StopEvaluationContext, StopVerdict, Timeout, TokenBudget,
    };
    use serde_json::{json, Value};
    use std::collections::BTreeMap;

    fn short_answer() -> Value {
        json!({"result_type": "answer", "output": "hi"})
    }

    fn long_answer() -> Value {
        json!({
            "result_type": "answer",
            "output": "this answer is definitely long enough",
        })
    }

    fn empty_ctx() -> StopEvaluationContext {
        StopEvaluationContext::default()
    }

    #[tokio::test]
    async fn max_rounds_triggers_inclusively() {
        let mut evaluator = MaxRounds::new(3);

        let ctx = empty_ctx().with_iteration(2);
        assert!(evaluator.evaluate(&ctx).await.is_none());
        let verdict = evaluator
            .evaluate(&empty_ctx().with_iteration(3))
            .await
            .expect("iteration == max_rounds must trigger");
        assert_eq!(verdict.reason(), "MaxRoundsEvaluator");
        assert!(verdict.detail().is_some());
        let ctx = empty_ctx().with_iteration(4);
        assert!(evaluator.evaluate(&ctx).await.is_some());
    }

    #[tokio::test]
    async fn token_budget_triggers_inclusively() {
        let mut evaluator = TokenBudget::new(100);

        let ctx = empty_ctx().with_token_usage(99);
        assert!(evaluator.evaluate(&ctx).await.is_none());
        let verdict = evaluator
            .evaluate(&empty_ctx().with_token_usage(100))
            .await
            .expect("token_usage == max_tokens must trigger");
        assert_eq!(verdict.reason(), "TokenBudgetEvaluator");
        assert!(verdict.detail().unwrap().contains("100"));
    }

    #[tokio::test]
    async fn timeout_triggers_inclusively_with_fractional_bounds() {
        let mut evaluator = Timeout::new(1.5);

        let ctx = empty_ctx().with_elapsed_seconds(1.499);
        assert!(evaluator.evaluate(&ctx).await.is_none());
        let verdict = evaluator
            .evaluate(&empty_ctx().with_elapsed_seconds(1.5))
            .await
            .expect("elapsed == timeout must trigger");
        assert_eq!(verdict.reason(), "TimeoutEvaluator");
        let ctx = empty_ctx().with_elapsed_seconds(9.0);
        assert!(evaluator.evaluate(&ctx).await.is_some());
    }

    #[tokio::test]
    async fn no_progress_counts_short_answers_and_resets_on_long_answer() {
        let mut evaluator = NoProgressAnswer::new(3, 20);

        for expected in 1..=2 {
            let verdict = evaluator
                .evaluate(&empty_ctx().with_last_result(Some(short_answer())))
                .await;
            assert!(verdict.is_none(), "streak {expected} must not trigger");
            assert_eq!(evaluator.consecutive_empty_answers, expected);
        }
        assert_eq!(evaluator.consecutive_empty_answers, 2);

        let ctx = empty_ctx().with_last_result(Some(long_answer()));
        assert!(evaluator.evaluate(&ctx).await.is_none());
        assert_eq!(evaluator.consecutive_empty_answers, 0);
    }

    #[tokio::test]
    async fn no_progress_resets_on_non_answer_and_missing_result() {
        let mut evaluator = NoProgressAnswer::new(2, 20);
        let _ = evaluator
            .evaluate(&empty_ctx().with_last_result(Some(short_answer())))
            .await;
        assert_eq!(evaluator.consecutive_empty_answers, 1);

        let tool_result = Some(json!({"result_type": "tool_call", "output": "x"}));
        let ctx = empty_ctx().with_last_result(tool_result);
        assert!(evaluator.evaluate(&ctx).await.is_none());
        assert_eq!(evaluator.consecutive_empty_answers, 0);

        let _ = evaluator
            .evaluate(&empty_ctx().with_last_result(Some(short_answer())))
            .await;
        assert_eq!(evaluator.consecutive_empty_answers, 1);
        let ctx = empty_ctx().with_last_result(None);
        assert!(evaluator.evaluate(&ctx).await.is_none());
        assert_eq!(evaluator.consecutive_empty_answers, 0);
    }

    #[tokio::test]
    async fn no_progress_triggers_exactly_at_threshold() {
        let mut evaluator = NoProgressAnswer::new(2, 5);
        let ctx = empty_ctx().with_last_result(Some(short_answer()));

        assert!(evaluator.evaluate(&ctx).await.is_none());
        assert_eq!(evaluator.consecutive_empty_answers, 1);
        let verdict = evaluator
            .evaluate(&ctx)
            .await
            .expect("second short answer must trigger");
        assert_eq!(verdict.reason(), "NoProgressAnswerEvaluator");
    }

    #[tokio::test]
    async fn no_progress_defaults_and_clamping() {
        let evaluator = NoProgressAnswer::default();
        assert_eq!(evaluator.consecutive_empty_answers, 0);
        assert_eq!(evaluator.max_consecutive_empty_answers, 3);
        assert_eq!(evaluator.min_answer_chars, 20);

        let clamped = NoProgressAnswer::new(0, 20);
        assert_eq!(clamped.max_consecutive_empty_answers, 1);

        let mut triggering = NoProgressAnswer::new(0, 20);
        let verdict = triggering
            .evaluate(&empty_ctx().with_last_result(Some(short_answer())))
            .await;
        assert!(verdict.is_some(), "clamped threshold of 1 triggers at once");
    }

    #[tokio::test]
    async fn no_progress_state_round_trip() {
        let mut evaluator = NoProgressAnswer::new(2, 20);
        let ctx = empty_ctx().with_last_result(Some(short_answer()));
        assert!(evaluator.evaluate(&ctx).await.is_none());
        let state = evaluator.get_state().expect("state is exported");
        assert_eq!(state["consecutive_empty_answers"], json!(1));
        assert_eq!(state["max_consecutive_empty_answers"], json!(2));
        assert_eq!(state["min_answer_chars"], json!(20));

        let mut restored = NoProgressAnswer::default();
        restored.load_state(&state);
        assert_eq!(restored.consecutive_empty_answers, 1);
        assert_eq!(restored.max_consecutive_empty_answers, 2);
        assert_eq!(restored.min_answer_chars, 20);
        let verdict = restored
            .evaluate(&ctx)
            .await
            .expect("restored streak must trigger at the loaded threshold");
        assert_eq!(verdict.reason(), "NoProgressAnswerEvaluator");

        restored.load_state(&json!({"consecutive_empty_answers": "junk"}));
        assert_eq!(restored.consecutive_empty_answers, 0);
        restored.load_state(&json!("not an object"));
        assert_eq!(restored.consecutive_empty_answers, 0);
    }

    #[tokio::test]
    async fn completion_promise_requires_consecutive_confirmations() {
        let mut evaluator = CompletionPromise::new("DONE", 2);

        evaluator.notify_fulfilled("done");
        assert!(evaluator.evaluate(&empty_ctx()).await.is_none());
        assert_eq!(evaluator.confirmation_count, 1);
        assert_eq!(evaluator.matched_text, "done");

        evaluator.notify_fulfilled("still done");
        let verdict = evaluator
            .evaluate(&empty_ctx())
            .await
            .expect("two confirmations with required 2 must trigger");
        assert_eq!(verdict.reason(), "CompletionPromiseEvaluator");
        assert!(verdict.detail().unwrap().contains("DONE"));
    }

    #[tokio::test]
    async fn completion_promise_absent_resets_streak() {
        let mut evaluator = CompletionPromise::new("DONE", 2);

        evaluator.notify_fulfilled("done");
        evaluator.notify_absent();
        assert_eq!(evaluator.confirmation_count, 0);
        assert!(!evaluator.fulfilled);
        assert_eq!(evaluator.matched_text, "");

        evaluator.notify_fulfilled("done");
        assert!(
            evaluator.evaluate(&empty_ctx()).await.is_none(),
            "the streak restarted, one confirmation is not enough"
        );
        evaluator.notify_fulfilled("done");
        assert!(evaluator.evaluate(&empty_ctx()).await.is_some());

        evaluator.reset();
        assert!(!evaluator.fulfilled);
        assert_eq!(evaluator.matched_text, "");
        assert_eq!(evaluator.confirmation_count, 0);
        assert!(evaluator.evaluate(&empty_ctx()).await.is_none());
    }

    #[tokio::test]
    async fn completion_promise_state_keys_and_restore() {
        let mut evaluator = CompletionPromise::new("DONE", 2);
        evaluator.notify_fulfilled("done");
        evaluator.notify_fulfilled("done");

        let state = evaluator.get_state().expect("state is exported");
        let object = state.as_object().expect("state is a JSON object");
        assert_eq!(object.len(), 4);
        assert_eq!(object["fulfilled"], json!(true));
        assert_eq!(object["matched_text"], json!("done"));
        assert_eq!(object["required_confirmations"], json!(2));
        assert_eq!(object["confirmation_count"], json!(2));

        let mut fresh = CompletionPromise::new("DONE", 2);
        fresh.load_state(&json!({
            "fulfilled": false,
            "matched_text": "",
            "required_confirmations": 2,
            "confirmation_count": 0,
        }));
        assert!(!fresh.fulfilled);

        // A persisted `true` is OR-merged and never downgraded.
        fresh.load_state(&state);
        assert!(fresh.fulfilled);
        assert_eq!(fresh.matched_text, "done");
        assert_eq!(fresh.required_confirmations, 2);
        assert_eq!(fresh.confirmation_count, 2);

        fresh.load_state(&json!({"fulfilled": true, "confirmation_count": 0}));
        assert!(fresh.fulfilled);
        assert_eq!(fresh.confirmation_count, 0);
        assert_eq!(
            fresh.required_confirmations, 2,
            "missing keys keep the current required confirmations"
        );
    }

    #[tokio::test]
    async fn verdict_reasons_use_python_class_names() {
        assert_eq!(MaxRounds::new(1).name(), "MaxRoundsEvaluator");
        assert_eq!(TokenBudget::new(1).name(), "TokenBudgetEvaluator");
        assert_eq!(Timeout::new(1.0).name(), "TimeoutEvaluator");
        assert_eq!(
            NoProgressAnswer::default().name(),
            "NoProgressAnswerEvaluator"
        );
        assert_eq!(
            CompletionPromise::new("DONE", 1).name(),
            "CompletionPromiseEvaluator"
        );

        let ctx = empty_ctx()
            .with_iteration(1)
            .with_token_usage(1)
            .with_elapsed_seconds(1.0)
            .with_last_result(Some(short_answer()));
        let mut max_rounds = MaxRounds::new(1);
        assert_eq!(
            max_rounds.evaluate(&ctx).await.unwrap().reason(),
            "MaxRoundsEvaluator"
        );
        let mut budget = TokenBudget::new(1);
        assert_eq!(
            budget.evaluate(&ctx).await.unwrap().reason(),
            "TokenBudgetEvaluator"
        );
        let mut timeout = Timeout::new(1.0);
        assert_eq!(
            timeout.evaluate(&ctx).await.unwrap().reason(),
            "TimeoutEvaluator"
        );
        let mut no_progress = NoProgressAnswer::new(1, 20);
        assert_eq!(
            no_progress.evaluate(&ctx).await.unwrap().reason(),
            "NoProgressAnswerEvaluator"
        );

        let plain = StopVerdict::new("MaxRoundsEvaluator");
        assert_eq!(plain.reason(), "MaxRoundsEvaluator");
        assert!(plain.detail().is_none());
        assert_eq!(
            plain.with_detail("iteration 3").detail(),
            Some("iteration 3")
        );
    }

    #[tokio::test]
    async fn custom_predicate_triggers_on_its_predicate() {
        let mut evaluator = CustomPredicate::new("TokenEighty", |ctx: &StopEvaluationContext| {
            ctx.token_usage >= 80
        });
        assert_eq!(evaluator.name(), "TokenEighty");

        let ctx = empty_ctx().with_token_usage(79);
        assert!(evaluator.evaluate(&ctx).await.is_none());
        let verdict = evaluator
            .evaluate(&empty_ctx().with_token_usage(80))
            .await
            .expect("predicate is satisfied");
        assert_eq!(verdict.reason(), "TokenEighty");
        assert!(verdict.detail().is_none());
        assert!(evaluator.get_state().is_none());

        let debug = format!("{evaluator:?}");
        assert!(debug.contains("CustomPredicate"));
        assert!(debug.contains("TokenEighty"));
    }

    #[tokio::test]
    async fn evaluator_trait_is_object_safe() {
        let mut evaluators: Vec<Box<dyn StopConditionEvaluator>> = vec![
            Box::new(MaxRounds::new(1)),
            Box::new(TokenBudget::new(1)),
            Box::new(Timeout::new(1.0)),
        ];
        let ctx = empty_ctx()
            .with_iteration(1)
            .with_token_usage(1)
            .with_elapsed_seconds(1.0);
        for evaluator in &mut evaluators {
            assert!(evaluator.evaluate(&ctx).await.is_some());
        }
    }

    #[test]
    fn stop_condition_state_round_trip_and_junk_restore() {
        let mut evaluator_states = BTreeMap::new();
        evaluator_states.insert("MaxRoundsEvaluator".to_string(), json!({"hit": true}));
        let state = StopConditionState {
            iteration: 4,
            token_usage: 512,
            stop_reason: Some("MaxRoundsEvaluator".to_string()),
            evaluator_states,
        };

        let restored = StopConditionState::restore(&state.get_state());
        assert_eq!(restored, state);

        assert_eq!(
            StopConditionState::restore(&json!("junk")),
            StopConditionState::default()
        );
        assert_eq!(
            StopConditionState::restore(&json!([1, 2, 3])),
            StopConditionState::default()
        );
        assert_eq!(
            StopConditionState::restore(&Value::Null),
            StopConditionState::default()
        );

        let decoded: StopConditionState = serde_json::from_value(json!({
            "iteration": 2,
            "unknown": "ignored",
        }))
        .expect("unknown fields are tolerated");
        assert_eq!(decoded.iteration, 2);
        assert_eq!(decoded.token_usage, 0);
        assert!(decoded.stop_reason.is_none());
        assert!(decoded.evaluator_states.is_empty());
    }

    #[test]
    fn evaluation_context_builders_and_defaults() {
        let default = empty_ctx();
        assert_eq!(default.iteration(), 0);
        assert_eq!(default.token_usage(), 0);
        assert!(default.elapsed_seconds().abs() < f64::EPSILON);
        assert!(default.last_result().is_none());
        assert!(default.extra().is_empty());

        let mut extra = BTreeMap::new();
        extra.insert("goal".to_string(), json!("ship it"));
        let ctx = default
            .with_iteration(7)
            .with_token_usage(42)
            .with_elapsed_seconds(2.5)
            .with_last_result(Some(short_answer()))
            .with_extra(extra);
        assert_eq!(ctx.iteration(), 7);
        assert_eq!(ctx.token_usage(), 42);
        assert!((ctx.elapsed_seconds() - 2.5).abs() < f64::EPSILON);
        assert_eq!(ctx.last_result(), Some(&short_answer()));
        assert_eq!(ctx.extra().get("goal"), Some(&json!("ship it")));

        let encoded = serde_json::to_value(&ctx).expect("context serializes");
        assert_eq!(
            serde_json::from_value::<StopEvaluationContext>(encoded).expect("context deserializes"),
            ctx
        );
        let empty: StopEvaluationContext =
            serde_json::from_value(json!({})).expect("all fields default");
        assert_eq!(empty, empty_ctx());
    }
}
