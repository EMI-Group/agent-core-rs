//! The rollout runtime: agent seam, reward composition and executors
//! (Python `runtime/`).
//!
//! Python's `RuntimeExecutor` runs an agent through a trajectory span processor
//! and projects the trajectory onto `Rollout` records; `ParallelRuntimeExecutor`
//! runs N worker loops over a [`TaskQueue`]. Both are ported here, with the
//! agent replaced by a seam:
//!
//! - [`RolloutSource`] produces the turns of one task (the "run the agent" step);
//! - [`RewardComposer`] scores them afterwards (the reward-function step).
//!
//! Errors are never swallowed. Python logs an executor failure and returns an
//! empty, zero-reward message; this port does the same but also records the
//! failure in [`RuntimeOutcome::metadata`], because `RolloutMessage` itself has
//! no metadata field — Python's is likewise silent about the failure, which is
//! exactly the silent degradation this crate avoids.
//!
//! No wall clock is read: Python fills `start_time`/`end_time` with ISO stamps,
//! here both stay `None` so replaying the same tasks yields byte-identical
//! records.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

use openjiuwen_common::error::{Error, Result};

use crate::rl::{RlTask, Rollout, RolloutMessage};

use super::queue::TaskQueue;
use super::{finite_reward, prompt_ground_truth, render_response_text, rollout_turns, stable_id};

/// How long a worker sleeps before it looks for more work.
const POLL_INTERVAL: Duration = Duration::from_millis(2);

/// The agent seam: turns one task into the turns of its rollout.
///
/// Python's executor builds an agent per task through `agent_factory` and
/// collects its trajectory through an OTLP span processor. Neither the agent
/// runtime nor an exporter belongs in the offline runtime, so the seam is the
/// whole contract: given a task, return its turns.
#[async_trait]
pub trait RolloutSource: Send + Sync {
    /// A stable name used in diagnostics.
    fn name(&self) -> &'static str {
        "rollout-source"
    }

    /// Rolls `task` out into its turns, in order.
    async fn rollout(&self, task: &RlTask) -> Result<Vec<Rollout>>;
}

/// Scores a rollout after it was produced.
///
/// Python's `reward_fn` takes the whole `RolloutMessage` and returns
/// `{"reward_list": [...], "global_reward": x}`; composition in place keeps the
/// message intact (turns included) while the reward fields are replaced.
pub trait RewardComposer: Send + Sync {
    /// A stable name used in diagnostics and weighted composition errors.
    fn name(&self) -> &str;

    /// Fills `rollouts.reward_list` and `rollouts.global_reward`.
    ///
    /// # Errors
    ///
    /// A composer may fail on a malformed rollout; the executor records the
    /// failure and leaves the message with no reward instead of inventing one.
    fn compose(&self, task: &RlTask, rollouts: &mut RolloutMessage) -> Result<()>;
}

/// The expected answer of a task, when its sample carries one.
fn task_expected(task: &RlTask) -> String {
    if let Value::Object(map) = &task.task_sample {
        for key in ["expected", "answer", "label", "ground_truth"] {
            if let Some(Value::String(text)) = map.get(key) {
                if !text.trim().is_empty() {
                    return text.clone();
                }
            }
        }
        if let Some(Value::Object(metadata)) = map.get("metadata") {
            if let Some(Value::String(text)) = metadata.get("ground_truth") {
                return text.clone();
            }
        }
    }
    String::new()
}

/// Rewards a rollout by exact match against the task's expected answer.
///
/// The expected answer comes from the first turn's `ground_truth` (where Python
/// stores it) and falls back to the task sample. Every turn is scored
/// independently (`1.0` on a match, `0.0` otherwise) and the global reward is
/// the **last** turn's score, which is what "the final answer is right" means.
/// Without an expected answer nothing can match, so every reward is `0.0`; that
/// is a deterministic judgement, not a silent fallback.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExactMatchComposer;

impl ExactMatchComposer {
    /// Creates the composer.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl RewardComposer for ExactMatchComposer {
    fn name(&self) -> &'static str {
        "exact_match"
    }

    fn compose(&self, task: &RlTask, rollouts: &mut RolloutMessage) -> Result<()> {
        let turns = rollout_turns(rollouts)?;
        let mut expected = turns
            .first()
            .map(|turn| prompt_ground_truth(&turn.input_prompt))
            .unwrap_or_default();
        if expected.is_empty() {
            expected = task_expected(task);
        }
        let rewards: Vec<f64> = turns
            .iter()
            .map(|turn| {
                if expected.is_empty() {
                    0.0
                } else if render_response_text(&turn.output_response).trim() == expected.trim() {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        rollouts.global_reward = rewards.last().copied().unwrap_or(0.0);
        rollouts.reward_list = rewards;
        Ok(())
    }
}

/// Combines two or more composers with weights.
///
/// Every composer runs against a clone of the message and contributes its
/// `global_reward`; the result is the weighted mean (weights normalised by their
/// sum, so `0.7/0.3` and `7.0/3.0` agree). Per-turn rewards are combined the same
/// way when every composer produced one reward per turn; otherwise the composed
/// global reward becomes the single per-turn entry, so `reward_list` never
/// contradicts `global_reward`.
pub struct WeightedRewardComposer {
    composers: Vec<(Arc<dyn RewardComposer>, f64)>,
}

impl fmt::Debug for WeightedRewardComposer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WeightedRewardComposer")
            .field(
                "composers",
                &self
                    .composers
                    .iter()
                    .map(|(composer, weight)| (composer.name().to_owned(), *weight))
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl WeightedRewardComposer {
    /// Creates a weighted composer.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when fewer than two composers are supplied, a
    /// weight is negative or non-finite, or the weights sum to zero.
    pub fn new(composers: Vec<(Arc<dyn RewardComposer>, f64)>) -> Result<Self> {
        if composers.len() < 2 {
            return Err(Error::validation(format!(
                "WeightedRewardComposer needs at least two composers, got {}",
                composers.len()
            )));
        }
        let mut total = 0.0;
        for (composer, weight) in &composers {
            if !weight.is_finite() || *weight < 0.0 {
                return Err(Error::validation(format!(
                    "WeightedRewardComposer weight for {:?} must be finite and non-negative, got {weight}",
                    composer.name()
                )));
            }
            total += weight;
        }
        if total <= 0.0 {
            return Err(Error::validation(
                "WeightedRewardComposer weights must not all be zero",
            ));
        }
        Ok(Self { composers })
    }

    /// The composed rewards and their weights, in registration order.
    #[must_use]
    pub fn composers(&self) -> Vec<(&str, f64)> {
        self.composers
            .iter()
            .map(|(composer, weight)| (composer.name(), *weight))
            .collect()
    }
}

impl RewardComposer for WeightedRewardComposer {
    fn name(&self) -> &'static str {
        "weighted"
    }

    fn compose(&self, task: &RlTask, rollouts: &mut RolloutMessage) -> Result<()> {
        let total: f64 = self.composers.iter().map(|(_, weight)| weight).sum();
        let mut global = 0.0;
        let mut lists: Vec<Vec<f64>> = Vec::with_capacity(self.composers.len());
        for (composer, weight) in &self.composers {
            let mut candidate = rollouts.clone();
            composer.compose(task, &mut candidate)?;
            global += weight * finite_reward(&candidate);
            lists.push(candidate.reward_list);
        }
        let global = if global.is_finite() {
            global / total
        } else {
            0.0
        };

        let reward_list = if lists.first().is_some_and(|first| {
            !first.is_empty() && lists.iter().all(|list| list.len() == first.len())
        }) {
            (0..lists[0].len())
                .map(|index| {
                    let weighted: f64 = lists
                        .iter()
                        .zip(&self.composers)
                        .map(|(list, (_, weight))| weight * list[index])
                        .sum();
                    if weighted.is_finite() {
                        weighted / total
                    } else {
                        0.0
                    }
                })
                .collect()
        } else {
            vec![global]
        };

        rollouts.global_reward = global;
        rollouts.reward_list = reward_list;
        Ok(())
    }
}

/// The result of executing one task.
///
/// `message` is always a valid [`RolloutMessage`]; `metadata` carries the
/// diagnostics Python only logs. Keys: `"rollout_source"`, plus `"error"` when
/// the source failed and `"reward_error"` when composition failed.
#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeOutcome {
    /// The rollout message, valid even when the task failed.
    pub message: RolloutMessage,
    /// Diagnostics of the execution.
    pub metadata: BTreeMap<String, Value>,
}

impl RuntimeOutcome {
    /// Whether the rollout source failed.
    #[must_use]
    pub fn is_error(&self) -> bool {
        self.metadata.contains_key("error")
    }

    /// Whether reward composition failed.
    #[must_use]
    pub fn is_reward_error(&self) -> bool {
        self.metadata.contains_key("reward_error")
    }

    /// The recorded failure message, when there was one.
    #[must_use]
    pub fn error_message(&self) -> Option<&str> {
        self.metadata.get("error").and_then(Value::as_str)
    }
}

/// Executes one rollout task: run the source, then compose the reward.
pub struct RuntimeExecutor {
    source: Arc<dyn RolloutSource>,
    composer: Option<Arc<dyn RewardComposer>>,
}

impl fmt::Debug for RuntimeExecutor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RuntimeExecutor")
            .field("source", &self.source.name())
            .field("composer", &self.composer.as_ref().map(|c| c.name()))
            .finish()
    }
}

impl RuntimeExecutor {
    /// Creates an executor over `source` without reward composition.
    #[must_use]
    pub fn new(source: Arc<dyn RolloutSource>) -> Self {
        Self {
            source,
            composer: None,
        }
    }

    /// Adds reward composition.
    #[must_use]
    pub fn with_composer(mut self, composer: Arc<dyn RewardComposer>) -> Self {
        self.composer = Some(composer);
        self
    }

    /// The rollout source.
    #[must_use]
    pub fn source(&self) -> &Arc<dyn RolloutSource> {
        &self.source
    }

    /// The reward composer, when one is configured.
    #[must_use]
    pub fn composer(&self) -> Option<&Arc<dyn RewardComposer>> {
        self.composer.as_ref()
    }

    /// Executes `task`, capturing every failure in the outcome.
    ///
    /// A failed source yields an empty message with `global_reward = 0.0` and an
    /// `"error"` entry in the metadata. A failed composer keeps the turns but
    /// clears the rewards and records `"reward_error"`, mirroring Python's
    /// `_apply_reward` failure path — with the difference that the reason is
    /// kept instead of only logged.
    pub async fn execute(&self, task: &RlTask) -> RuntimeOutcome {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "rollout_source".to_owned(),
            Value::String(self.source.name().to_owned()),
        );
        let mut message = RolloutMessage {
            task_id: task.task_id.clone(),
            origin_task_id: task.origin_task_id.clone(),
            rollout_id: stable_id("rollout", &[task.task_id.as_str()]),
            start_time: None,
            end_time: None,
            rollout_info: json!([]),
            reward_list: Vec::new(),
            global_reward: 0.0,
            turn_count: 0,
            round_num: task.round_num,
        };

        match self.source.rollout(task).await {
            Ok(turns) => {
                message.turn_count = turns.len() as u64;
                match serde_json::to_value(&turns) {
                    Ok(value) => message.rollout_info = value,
                    Err(error) => {
                        metadata.insert("error".to_owned(), Value::String(error.to_string()));
                        return RuntimeOutcome { message, metadata };
                    }
                }
                if let Some(composer) = &self.composer {
                    metadata.insert(
                        "reward_composer".to_owned(),
                        Value::String(composer.name().to_owned()),
                    );
                    if let Err(error) = composer.compose(task, &mut message) {
                        metadata
                            .insert("reward_error".to_owned(), Value::String(error.to_string()));
                        message.reward_list.clear();
                        message.global_reward = 0.0;
                    }
                }
            }
            Err(error) => {
                metadata.insert("error".to_owned(), Value::String(error.to_string()));
                message.rollout_info = json!([]);
                message.reward_list.clear();
                message.global_reward = 0.0;
                message.turn_count = 0;
            }
        }
        RuntimeOutcome { message, metadata }
    }

    /// Executes `task` and returns just the message.
    pub async fn execute_message(&self, task: &RlTask) -> RolloutMessage {
        self.execute(task).await.message
    }
}

/// Runs N worker loops over one [`TaskQueue`].
///
/// Each worker pulls a task, executes it and buffers the result. Workers keep
/// polling the queue until [`Self::stop`] is called (Python's semantics, where
/// the executor outlives many rounds); [`Self::wait_for_finish`] is the bounded
/// replacement for Python's unbounded `while True` poll.
pub struct ParallelRuntimeExecutor {
    queue: Arc<TaskQueue>,
    num_workers: usize,
    executor: Arc<RuntimeExecutor>,
    running: Arc<AtomicBool>,
    workers: Mutex<Vec<JoinHandle<()>>>,
}

impl fmt::Debug for ParallelRuntimeExecutor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ParallelRuntimeExecutor")
            .field("num_workers", &self.num_workers)
            .field("executor", &self.executor)
            .field("running", &self.running.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl ParallelRuntimeExecutor {
    /// Creates an executor with `num_workers` worker loops (at least one).
    #[must_use]
    pub fn new(queue: Arc<TaskQueue>, num_workers: usize, executor: Arc<RuntimeExecutor>) -> Self {
        Self {
            queue,
            num_workers: num_workers.max(1),
            executor,
            running: Arc::new(AtomicBool::new(false)),
            workers: Mutex::new(Vec::new()),
        }
    }

    /// The queue the workers pull from.
    #[must_use]
    pub fn queue(&self) -> &Arc<TaskQueue> {
        &self.queue
    }

    /// The number of worker loops.
    #[must_use]
    pub fn num_workers(&self) -> usize {
        self.num_workers
    }

    /// Whether the workers are running.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Launches the worker loops.
    ///
    /// Calling this while the workers run is a no-op (Python logs a warning).
    ///
    /// # Errors
    ///
    /// Never; the method returns `Result` so callers can propagate a future
    /// runtime-setup failure without changing the signature.
    pub async fn start(&self) -> Result<()> {
        let mut workers = self.workers.lock().await;
        if self.running.load(Ordering::SeqCst) {
            return Ok(());
        }
        self.running.store(true, Ordering::SeqCst);
        for _ in 0..self.num_workers {
            workers.push(tokio::spawn(worker_loop(
                Arc::clone(&self.queue),
                Arc::clone(&self.executor),
                Arc::clone(&self.running),
            )));
        }
        Ok(())
    }

    /// Stops the workers and waits for them to finish the task in flight.
    ///
    /// # Errors
    ///
    /// Never; see [`Self::start`].
    pub async fn stop(&self) -> Result<()> {
        self.running.store(false, Ordering::SeqCst);
        let handles: Vec<JoinHandle<()>> = {
            let mut workers = self.workers.lock().await;
            std::mem::take(&mut *workers)
        };
        for handle in handles {
            // A panicking worker is a bug in this module, not a task failure:
            // the join error is surfaced through the queue still holding the
            // in-processing task, so nothing is silently skipped.
            let _ = handle.await;
        }
        Ok(())
    }

    /// Waits until the queue is drained, bounded by `timeout`.
    ///
    /// # Errors
    ///
    /// [`Error::timeout`] when the queue still holds queued or in-processing
    /// tasks after `timeout`. Python polls forever; a hung rollout must not hang
    /// the trainer, so the caller decides the bound.
    pub async fn wait_for_finish(&self, timeout: Duration) -> Result<()> {
        let wait = async {
            loop {
                if self.queue.is_finished().await {
                    return;
                }
                tokio::time::sleep(POLL_INTERVAL).await;
            }
        };
        tokio::time::timeout(timeout, wait).await.map_err(|_| {
            Error::timeout(format!(
                "ParallelRuntimeExecutor::wait_for_finish exceeded {timeout:?}; \
                 the queue still holds unfinished rollout tasks"
            ))
        })
    }
}

/// One worker loop: pull, execute, buffer.
async fn worker_loop(
    queue: Arc<TaskQueue>,
    executor: Arc<RuntimeExecutor>,
    running: Arc<AtomicBool>,
) {
    while running.load(Ordering::SeqCst) {
        match queue.get_task().await {
            Some(task) => {
                let outcome = executor.execute(&task).await;
                queue.add_rollout(outcome.message).await;
            }
            None => tokio::time::sleep(POLL_INTERVAL).await,
        }
    }
}

/// A scripted [`RolloutSource`] over a fixed task id → turns table.
///
/// Useful in tests and examples: the map lookup is exact, and an absent task id
/// is a validation error rather than an empty rollout, so a typo cannot look
/// like a legitimate failed rollout.
#[derive(Debug, Clone, Default)]
pub struct ScriptedRolloutSource {
    turns: BTreeMap<String, Vec<Rollout>>,
}

impl ScriptedRolloutSource {
    /// Creates an empty scripted source.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Scripts the turns of `task_id`.
    #[must_use]
    pub fn with_turns(mut self, task_id: impl Into<String>, turns: Vec<Rollout>) -> Self {
        self.turns.insert(task_id.into(), turns);
        self
    }
}

#[async_trait]
impl RolloutSource for ScriptedRolloutSource {
    fn name(&self) -> &'static str {
        "scripted"
    }

    async fn rollout(&self, task: &RlTask) -> Result<Vec<Rollout>> {
        self.turns.get(&task.task_id).cloned().ok_or_else(|| {
            Error::validation(format!("no scripted turns for task {}", task.task_id))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task(id: &str, expected: &str) -> RlTask {
        RlTask {
            task_id: id.to_owned(),
            origin_task_id: Some("origin".to_owned()),
            task_sample: json!({"query": "q", "expected": expected}),
            round_num: 0,
        }
    }

    fn turn(prompt: &str, response: &str) -> Rollout {
        Rollout {
            turn_id: 0,
            input_prompt: json!({"message": [{"role": "user", "content": prompt}]}),
            output_response: json!({"content": response}),
            llm_config: json!({}),
            input_prompt_ids: vec![1],
            output_response_ids: vec![2],
        }
    }

    fn message(turns: &[(&str, &str)]) -> RolloutMessage {
        RolloutMessage {
            task_id: "t-1".to_owned(),
            origin_task_id: Some("origin".to_owned()),
            rollout_id: "r-1".to_owned(),
            start_time: None,
            end_time: None,
            rollout_info: serde_json::to_value(
                turns
                    .iter()
                    .map(|(prompt, response)| turn(prompt, response))
                    .collect::<Vec<_>>(),
            )
            .expect("serialize"),
            reward_list: Vec::new(),
            global_reward: 0.0,
            turn_count: turns.len() as u64,
            round_num: 0,
        }
    }

    struct FixedComposer {
        name: &'static str,
        reward: f64,
        fail: bool,
    }

    impl RewardComposer for FixedComposer {
        fn name(&self) -> &str {
            self.name
        }

        fn compose(&self, _task: &RlTask, rollouts: &mut RolloutMessage) -> Result<()> {
            if self.fail {
                return Err(Error::validation("composer exploded"));
            }
            rollouts.global_reward = self.reward;
            rollouts.reward_list = vec![self.reward];
            Ok(())
        }
    }

    #[test]
    fn exact_match_scores_every_turn_and_keeps_the_last() {
        let composer = ExactMatchComposer::new();
        assert_eq!(composer.name(), "exact_match");

        let mut msg = message(&[("q", "wrong"), ("q", "42")]);
        composer
            .compose(&task("t-1", "42"), &mut msg)
            .expect("compose");
        assert_eq!(msg.reward_list, vec![0.0, 1.0]);
        assert_eq!(msg.global_reward, 1.0);

        // The first turn's ground truth wins over the task sample.
        let mut msg = message(&[("q", "42")]);
        msg.rollout_info[0]["input_prompt"]["ground_truth"] = json!("nope");
        composer
            .compose(&task("t-1", "42"), &mut msg)
            .expect("compose");
        assert_eq!(msg.global_reward, 0.0);

        // No expectation anywhere: nothing matches.
        let mut msg = message(&[("q", "")]);
        composer
            .compose(&task("t-1", ""), &mut msg)
            .expect("compose");
        assert_eq!(msg.reward_list, vec![0.0]);
    }

    #[test]
    fn weighted_composition_normalises_the_weights() {
        let composer = WeightedRewardComposer::new(vec![
            (
                Arc::new(FixedComposer {
                    name: "high",
                    reward: 1.0,
                    fail: false,
                }),
                0.75,
            ),
            (
                Arc::new(FixedComposer {
                    name: "low",
                    reward: 0.0,
                    fail: false,
                }),
                0.25,
            ),
        ])
        .expect("valid");
        assert_eq!(composer.name(), "weighted");
        assert_eq!(composer.composers(), vec![("high", 0.75), ("low", 0.25)]);

        let mut msg = message(&[("q", "a")]);
        composer
            .compose(&task("t-1", "a"), &mut msg)
            .expect("compose");
        assert!((msg.global_reward - 0.75).abs() < 1e-12);
        assert_eq!(msg.reward_list.len(), 1);
        assert!((msg.reward_list[0] - 0.75).abs() < 1e-12);

        // Different per-turn widths fall back to a single entry.
        let mut msg = message(&[("q", "a"), ("q", "b")]);
        composer
            .compose(&task("t-1", "a"), &mut msg)
            .expect("compose");
        assert_eq!(msg.reward_list, vec![msg.global_reward]);
    }

    #[test]
    fn weighted_composition_validates_its_inputs() {
        let single = WeightedRewardComposer::new(vec![(
            Arc::new(FixedComposer {
                name: "one",
                reward: 1.0,
                fail: false,
            }),
            1.0,
        )]);
        assert!(single
            .expect_err("single")
            .to_string()
            .contains("at least two"));

        let negative = WeightedRewardComposer::new(vec![
            (
                Arc::new(FixedComposer {
                    name: "a",
                    reward: 1.0,
                    fail: false,
                }),
                -1.0,
            ),
            (
                Arc::new(FixedComposer {
                    name: "b",
                    reward: 1.0,
                    fail: false,
                }),
                1.0,
            ),
        ]);
        assert!(negative
            .expect_err("negative")
            .to_string()
            .contains("finite and non-negative"));

        let zero = WeightedRewardComposer::new(vec![
            (
                Arc::new(FixedComposer {
                    name: "a",
                    reward: 1.0,
                    fail: false,
                }),
                0.0,
            ),
            (
                Arc::new(FixedComposer {
                    name: "b",
                    reward: 1.0,
                    fail: false,
                }),
                0.0,
            ),
        ]);
        assert!(zero
            .expect_err("zero")
            .to_string()
            .contains("must not all be zero"));
    }

    #[tokio::test]
    async fn the_executor_records_turns_and_composes_rewards() {
        let source =
            Arc::new(ScriptedRolloutSource::new().with_turns("t-1", vec![turn("q", "42")]));
        let executor =
            RuntimeExecutor::new(source).with_composer(Arc::new(ExactMatchComposer::new()));
        let outcome = executor.execute(&task("t-1", "42")).await;
        assert!(!outcome.is_error());
        assert!(!outcome.is_reward_error());
        assert_eq!(outcome.message.turn_count, 1);
        assert_eq!(outcome.message.global_reward, 1.0);
        assert_eq!(outcome.message.reward_list, vec![1.0]);
        assert_eq!(outcome.message.origin_task_id.as_deref(), Some("origin"));
        assert_eq!(outcome.message.round_num, 0);
        assert_eq!(outcome.message.start_time, None);
        assert_eq!(outcome.metadata["rollout_source"], json!("scripted"));
        assert_eq!(outcome.metadata["reward_composer"], json!("exact_match"));
        assert_eq!(
            outcome.message.rollout_info[0]["output_response_ids"],
            json!([2])
        );

        let message = executor.execute_message(&task("t-1", "42")).await;
        assert_eq!(message.global_reward, 1.0);
    }

    #[tokio::test]
    async fn a_source_failure_yields_a_valid_empty_message() {
        let executor = RuntimeExecutor::new(Arc::new(ScriptedRolloutSource::new()));
        let outcome = executor.execute(&task("missing", "42")).await;
        assert!(outcome.is_error());
        assert!(outcome
            .error_message()
            .expect("message")
            .contains("no scripted turns"));
        assert_eq!(outcome.message.global_reward, 0.0);
        assert_eq!(outcome.message.rollout_info, json!([]));
        assert_eq!(outcome.message.turn_count, 0);
        assert!(
            outcome.message.reward_list.is_empty(),
            "a failed source clears rewards"
        );
    }

    #[tokio::test]
    async fn a_composer_failure_keeps_the_turns_and_clears_the_reward() {
        let source =
            Arc::new(ScriptedRolloutSource::new().with_turns("t-1", vec![turn("q", "42")]));
        let executor = RuntimeExecutor::new(source).with_composer(Arc::new(FixedComposer {
            name: "broken",
            reward: 1.0,
            fail: true,
        }));
        let outcome = executor.execute(&task("t-1", "42")).await;
        assert!(outcome.is_reward_error());
        assert_eq!(
            outcome.metadata["reward_error"],
            json!("validation: composer exploded")
        );
        assert_eq!(outcome.message.global_reward, 0.0);
        assert!(
            outcome.message.reward_list.is_empty(),
            "a failed source clears rewards"
        );
        assert_eq!(outcome.message.turn_count, 1);
    }

    #[tokio::test]
    async fn parallel_workers_drain_the_queue() {
        let queue = Arc::new(TaskQueue::new());
        let source = Arc::new(
            ScriptedRolloutSource::new()
                .with_turns("t-1", vec![turn("q", "42")])
                .with_turns("t-2", vec![turn("q", "7")])
                .with_turns("t-3", vec![turn("q", "42")]),
        );
        let executor = Arc::new(
            RuntimeExecutor::new(source).with_composer(Arc::new(ExactMatchComposer::new())),
        );
        let parallel = ParallelRuntimeExecutor::new(Arc::clone(&queue), 2, executor);
        assert_eq!(parallel.num_workers(), 2);
        assert_eq!(
            ParallelRuntimeExecutor::new(
                Arc::clone(&queue),
                0,
                Arc::new(RuntimeExecutor::new(Arc::new(ScriptedRolloutSource::new()))),
            )
            .num_workers(),
            1
        );
        assert!(!parallel.is_running());

        // Rollouts are keyed by rollout id, which derives from the task id, so
        // every queued task needs a distinct id.
        for (id, expected) in [("t-1", "42"), ("t-2", "nope"), ("t-3", "42")] {
            queue.queue_task(task(id, expected)).await;
        }
        parallel.start().await.expect("start");
        assert!(parallel.is_running());
        // Starting twice is a no-op.
        parallel.start().await.expect("start again");

        parallel
            .wait_for_finish(Duration::from_secs(5))
            .await
            .expect("finish");
        let rollouts = queue.get_rollouts().await;
        assert_eq!(rollouts.len(), 3);
        let total: f64 = rollouts.values().map(|r| r.global_reward).sum();
        assert!((total - 2.0).abs() < 1e-12);
        assert!(queue.is_finished().await);

        parallel.stop().await.expect("stop");
        assert!(!parallel.is_running());
    }

    #[tokio::test]
    async fn wait_for_finish_times_out_on_stuck_queues() {
        let queue = Arc::new(TaskQueue::new());
        queue.queue_task(task("t-1", "42")).await;
        let parallel = ParallelRuntimeExecutor::new(
            Arc::clone(&queue),
            1,
            Arc::new(RuntimeExecutor::new(Arc::new(ScriptedRolloutSource::new()))),
        );
        // Nothing was started, so the queued task cannot finish.
        let error = parallel
            .wait_for_finish(Duration::from_millis(20))
            .await
            .expect_err("timeout");
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Timeout);
        assert!(error.to_string().contains("wait_for_finish"));

        queue.clear().await;
        parallel
            .wait_for_finish(Duration::from_millis(20))
            .await
            .expect("finished queue");
    }

    #[test]
    fn debug_output_names_the_seams() {
        let executor = RuntimeExecutor::new(Arc::new(ScriptedRolloutSource::new()))
            .with_composer(Arc::new(ExactMatchComposer::new()));
        let debug = format!("{executor:?}");
        assert!(debug.contains("scripted"));
        assert!(debug.contains("exact_match"));

        let composer = WeightedRewardComposer::new(vec![
            (
                Arc::new(FixedComposer {
                    name: "a",
                    reward: 1.0,
                    fail: false,
                }),
                1.0,
            ),
            (
                Arc::new(FixedComposer {
                    name: "b",
                    reward: 0.0,
                    fail: false,
                }),
                1.0,
            ),
        ])
        .expect("valid");
        assert!(format!("{composer:?}").contains("\"a\""));

        let parallel = ParallelRuntimeExecutor::new(
            Arc::new(TaskQueue::new()),
            3,
            Arc::new(RuntimeExecutor::new(Arc::new(ScriptedRolloutSource::new()))),
        );
        assert!(format!("{parallel:?}").contains("num_workers: 3"));
    }
}
