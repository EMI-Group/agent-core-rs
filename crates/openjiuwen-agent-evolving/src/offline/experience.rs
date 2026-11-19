//! The experience-collection hook (Python's rollout → trajectory → experience path).
//!
//! Python's offline runtime has an optional experience sink plugged into the
//! rollout loop: each finished rollout becomes a trajectory, the trajectory is
//! mined for lessons, and the lessons land in the experience archive the agent
//! reads on its next run. The Rust port keeps that shape behind the
//! [`ExperienceCollector`] seam so a trainer can run with or without a sink.
//!
//! [`TrajectoryExperienceCollector`] is the default implementation:
//!
//! 1. decode the rollout's turns and render each to text;
//! 2. build one [`Trajectory`] with a user/model step pair per turn, a tool-call
//!    step for every tool the response names, the rollout reward as the score
//!    and the task id as the case id;
//! 3. classify the outcome (`Success` when `global_reward >= 0.5`, else
//!    `Failure`) so the extractor builds either a tool-ordering lesson or an
//!    "avoid" lesson;
//! 4. mine the trajectory with [`ExperienceExtractor::from_trajectories`] and add
//!    every record to a shared [`ExperienceArchive`].
//!
//! The extractor's own rules still apply: a failure always yields an "avoid"
//! record, while a success yields a tool-ordering record only when the rollout
//! names at least two distinct tools.
//!
//! Identifiers are `stable_id("trajectory", [rollout_id])` and the archive
//! deduplicates by content, so collecting the same rollout twice merges instead
//! of duplicating.

use std::sync::{Arc, Mutex};

use serde_json::Value;

use crate::experience::{
    ExperienceArchive, ExperienceExtractor, ExperienceRecord, ExperienceStats,
};
use crate::rl::{Rollout, RolloutMessage};
use crate::trajectory::{StepKind, Trajectory, TrajectoryOutcome, TrajectoryStep};
use openjiuwen_common::error::{Error, Result};

use super::stable_id;

/// The reward at or above which a rollout is recorded as a success.
const SUCCESS_REWARD_THRESHOLD: f64 = 0.5;

/// Receives finished rollouts so they can be mined for experience.
///
/// A collector is deliberately synchronous: it only touches memory (the archive
/// lock) and never performs I/O, so calling it from a rollout worker cannot
/// block the async runtime.
pub trait ExperienceCollector: Send + Sync {
    /// Collects one finished rollout.
    ///
    /// # Errors
    ///
    /// Whatever the implementation cannot accept — a malformed
    /// `rollout_info`, or a record the archive refuses.
    fn collect(&self, task: &crate::rl::RlTask, rollout: &RolloutMessage) -> Result<()>;
}

/// Turns rollouts into trajectory lessons and files them in an experience archive.
///
/// The archive is behind a `Mutex` so the collector can be shared by several
/// rollout workers through an `Arc`; the lock is held only for the duration of
/// one `add`, never across an `.await` (the trait is synchronous).
pub struct TrajectoryExperienceCollector {
    extractor: ExperienceExtractor,
    archive: Arc<Mutex<ExperienceArchive>>,
    skill_name: String,
}

impl std::fmt::Debug for TrajectoryExperienceCollector {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TrajectoryExperienceCollector")
            .field("skill_name", &self.skill_name)
            .field("archive_records", &lock(&self.archive).records().len())
            .finish_non_exhaustive()
    }
}

impl Default for TrajectoryExperienceCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl TrajectoryExperienceCollector {
    /// Creates a collector with a fresh, empty archive and the default skill name.
    #[must_use]
    pub fn new() -> Self {
        Self {
            extractor: ExperienceExtractor::new(),
            archive: Arc::new(Mutex::new(ExperienceArchive::new())),
            skill_name: "offline-rl".to_owned(),
        }
    }

    /// Sets the skill name the extracted records are filed under.
    #[must_use]
    pub fn with_skill_name(mut self, skill_name: impl Into<String>) -> Self {
        self.skill_name = skill_name.into();
        self
    }

    /// Sets the extractor's confidence floor.
    #[must_use]
    pub fn with_min_confidence(mut self, min_confidence: f64) -> Self {
        self.extractor = std::mem::take(&mut self.extractor).with_min_confidence(min_confidence);
        self
    }

    /// Shares an existing archive instead of the private one.
    #[must_use]
    pub fn with_archive(mut self, archive: Arc<Mutex<ExperienceArchive>>) -> Self {
        self.archive = archive;
        self
    }

    /// The shared archive handle.
    #[must_use]
    pub fn archive(&self) -> &Arc<Mutex<ExperienceArchive>> {
        &self.archive
    }

    /// The skill name records are filed under.
    #[must_use]
    pub fn skill_name(&self) -> &str {
        &self.skill_name
    }

    /// The records collected so far, in archive order.
    #[must_use]
    pub fn records(&self) -> Vec<ExperienceRecord> {
        lock(&self.archive).records().to_vec()
    }

    /// The archive statistics.
    #[must_use]
    pub fn stats(&self) -> ExperienceStats {
        lock(&self.archive).stats()
    }

    /// How many records the archive holds.
    #[must_use]
    pub fn record_count(&self) -> usize {
        lock(&self.archive).len()
    }

    /// Builds the trajectory of one rollout without filing it.
    ///
    /// This is the testable half of [`ExperienceCollector::collect`]: it is pure
    /// and returns the trajectory so callers can inspect the steps.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] for a malformed `rollout_info`, for a non-finite
    /// global reward, or when the rollout has no turns to record.
    pub fn build_trajectory(
        &self,
        task: &crate::rl::RlTask,
        rollout: &RolloutMessage,
    ) -> Result<Trajectory> {
        let turns = super::rollout_turns(rollout)?;
        if turns.is_empty() {
            return Err(Error::validation(format!(
                "rollout {} has no turns to record as a trajectory",
                rollout.rollout_id
            )));
        }
        let mut trajectory = Trajectory::new(stable_id("trajectory", &[&rollout.rollout_id]))?
            .with_case(task.task_id.clone())
            .with_outcome(if rollout.global_reward >= SUCCESS_REWARD_THRESHOLD {
                TrajectoryOutcome::Success
            } else {
                TrajectoryOutcome::Failure
            })
            .with_score(rollout.global_reward)?
            .with_metadata("rollout_id", Value::from(rollout.rollout_id.clone()))
            .with_metadata("skill_name", Value::from(self.skill_name.clone()));
        if let Some(origin) = &rollout.origin_task_id {
            trajectory = trajectory.with_metadata("origin_task_id", Value::from(origin.clone()));
        }
        for (index, turn) in turns.iter().enumerate() {
            let reward = rollout.reward_list.get(index).copied().unwrap_or(0.0);
            trajectory.push(
                TrajectoryStep::new(StepKind::Message, prompt_text(turn))
                    .with_role("user")
                    .with_metadata("turn_id", Value::from(index)),
            );
            trajectory.push(
                TrajectoryStep::new(StepKind::Model, response_text(turn))
                    .with_role("assistant")
                    .with_metadata("turn_id", Value::from(index))
                    .with_metadata("reward", Value::from(reward)),
            );
            for tool in tool_names(&turn.output_response) {
                trajectory.push(
                    TrajectoryStep::new(StepKind::ToolCall, format!("call {tool}"))
                        .with_tool_name(tool)
                        .with_metadata("turn_id", Value::from(index)),
                );
            }
        }
        Ok(trajectory)
    }
}

impl ExperienceCollector for TrajectoryExperienceCollector {
    fn collect(&self, task: &crate::rl::RlTask, rollout: &RolloutMessage) -> Result<()> {
        let trajectory = self.build_trajectory(task, rollout)?;
        let mut records = self.extractor.from_trajectories(&[trajectory]);
        // The extractor's floor applies to every source, including trajectory
        // lessons, so a strict collector files nothing below it.
        records.retain(|record| record.confidence >= self.extractor.min_confidence);
        for record in &mut records {
            if record.skill_name.trim().is_empty() {
                record.skill_name.clone_from(&self.skill_name);
            }
        }
        let mut archive = lock(&self.archive);
        for record in records {
            archive.add(record)?;
        }
        Ok(())
    }
}

/// Names of the tools one turn asked for, in document order.
///
/// A response payload is either plain text (no tools), an object with
/// `tool_name`/`name`/`tool_calls`, or a message envelope whose
/// `message`/`messages` list holds such objects. Neither the extraction nor the
/// recursion is open-ended: the payload is an already-parsed JSON value, so its
/// depth is bounded by the parser's recursion limit.
fn tool_names(response: &Value) -> Vec<String> {
    fn collect(value: &Value, names: &mut Vec<String>) {
        match value {
            Value::Object(map) => {
                if let Some(name) = map.get("tool_name").and_then(Value::as_str) {
                    names.push(name.to_owned());
                }
                if !map.contains_key("tool_calls") {
                    if let Some(name) = map.get("name").and_then(Value::as_str) {
                        names.push(name.to_owned());
                    }
                }
                if let Some(Value::Array(calls)) = map.get("tool_calls") {
                    for call in calls {
                        let name = call
                            .get("function")
                            .and_then(|function| function.get("name"))
                            .or_else(|| call.get("name"))
                            .and_then(Value::as_str);
                        if let Some(name) = name {
                            names.push(name.to_owned());
                        }
                    }
                }
                for key in ["message", "messages"] {
                    if let Some(inner) = map.get(key) {
                        collect(inner, names);
                    }
                }
            }
            Value::Array(items) => {
                for item in items {
                    collect(item, names);
                }
            }
            _ => {}
        }
    }
    let mut names = Vec::new();
    collect(response, &mut names);
    names.retain(|name| !name.trim().is_empty());
    names
}

fn prompt_text(turn: &Rollout) -> String {
    super::render_prompt_text(&turn.input_prompt)
}

fn response_text(turn: &Rollout) -> String {
    super::render_response_text(&turn.output_response)
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    use crate::rl::RlTask;

    fn task() -> RlTask {
        RlTask {
            task_id: "task-1-0-0".to_owned(),
            origin_task_id: Some("origin-1-0".to_owned()),
            task_sample: json!({"query": "fix the build"}),
            round_num: 0,
        }
    }

    fn rollout(reward: f64) -> RolloutMessage {
        RolloutMessage {
            task_id: "task-1-0-0".to_owned(),
            origin_task_id: Some("origin-1-0".to_owned()),
            rollout_id: "r-1".to_owned(),
            start_time: None,
            end_time: None,
            rollout_info: json!([
                {
                    "turn_id": 0,
                    "input_prompt": {"message": [{"role": "user", "content": "fix the build"}]},
                    "output_response": "cargo build",
                },
                {
                    "turn_id": 1,
                    "input_prompt": {"message": [{"role": "user", "content": "still failing"}]},
                    "output_response": "run cargo test",
                },
            ]),
            reward_list: vec![0.0, reward],
            global_reward: reward,
            turn_count: 2,
            round_num: 0,
        }
    }

    fn tool_rollout(reward: f64) -> RolloutMessage {
        RolloutMessage {
            rollout_info: json!([
                {
                    "input_prompt": {"message": [{"role": "user", "content": "patch the bug"}]},
                    "output_response": {"tool_calls": [{"function": {"name": "read_file"}}]},
                },
                {
                    "input_prompt": {"message": [{"role": "user", "content": "write it"}]},
                    "output_response": {"tool_name": "write_file"},
                },
            ]),
            reward_list: vec![1.0, reward],
            ..rollout(reward)
        }
    }

    #[test]
    fn a_rollout_becomes_a_trajectory_with_one_step_pair_per_turn() {
        let collector = TrajectoryExperienceCollector::new();
        let trajectory = collector
            .build_trajectory(&task(), &rollout(1.0))
            .expect("trajectory");
        assert_eq!(trajectory.steps().len(), 4);
        assert_eq!(trajectory.outcome, TrajectoryOutcome::Success);
        assert_eq!(trajectory.score, Some(1.0));
        assert_eq!(trajectory.case_id.as_deref(), Some("task-1-0-0"));
        assert_eq!(trajectory.steps()[0].content, "fix the build");
        assert_eq!(trajectory.steps()[1].content, "cargo build");
        assert_eq!(trajectory.steps()[3].metadata["reward"], json!(1.0));
        assert_eq!(trajectory.steps()[3].role.as_deref(), Some("assistant"));
        assert_eq!(trajectory.metadata["origin_task_id"], json!("origin-1-0"));
        assert_eq!(trajectory.metadata["skill_name"], json!("offline-rl"));
        // The identifier is derived, not random.
        assert_eq!(trajectory.trajectory_id, stable_id("trajectory", &["r-1"]));
        assert!(trajectory.trajectory_id.starts_with("trajectory-"));
    }

    #[test]
    fn tool_calls_become_tool_steps() {
        let collector = TrajectoryExperienceCollector::new();
        let trajectory = collector
            .build_trajectory(&task(), &tool_rollout(1.0))
            .expect("trajectory");
        let tools: Vec<&str> = trajectory
            .steps()
            .iter()
            .filter(|step| matches!(step.kind, StepKind::ToolCall))
            .filter_map(|step| step.tool_name.as_deref())
            .collect();
        assert_eq!(tools, vec!["read_file", "write_file"]);
        assert_eq!(trajectory.tool_names(), vec!["read_file", "write_file"]);
    }

    #[test]
    fn the_outcome_follows_the_reward_threshold() {
        let collector = TrajectoryExperienceCollector::new();
        for (reward, expected) in [
            (1.0, TrajectoryOutcome::Success),
            (0.5, TrajectoryOutcome::Success),
            (0.49, TrajectoryOutcome::Failure),
            (0.0, TrajectoryOutcome::Failure),
        ] {
            let trajectory = collector
                .build_trajectory(&task(), &rollout(reward))
                .expect("trajectory");
            assert_eq!(trajectory.outcome, expected, "reward {reward}");
        }
    }

    #[test]
    fn malformed_and_empty_rollouts_are_rejected() {
        let collector = TrajectoryExperienceCollector::new();
        let mut bad = rollout(1.0);
        bad.rollout_info = json!({"not": "a list"});
        let error = collector
            .build_trajectory(&task(), &bad)
            .expect_err("bad info");
        assert!(error.to_string().contains("malformed rollout_info"));

        let mut empty = rollout(1.0);
        empty.rollout_info = json!([]);
        let error = collector
            .build_trajectory(&task(), &empty)
            .expect_err("empty");
        assert!(error.to_string().contains("no turns"));

        let mut nan = rollout(1.0);
        nan.global_reward = f64::NAN;
        assert!(collector.build_trajectory(&task(), &nan).is_err());
    }

    #[test]
    fn successful_tool_order_and_failures_become_lessons() {
        let collector = TrajectoryExperienceCollector::new().with_skill_name("repo-repair");
        assert_eq!(collector.record_count(), 0);

        // A successful rollout naming two tools yields the ordering lesson.
        collector
            .collect(&task(), &tool_rollout(1.0))
            .expect("collect success");
        let records = collector.records();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].skill_name, "repo-repair");
        assert_eq!(records[0].source, "trajectory");
        assert!(records[0].content.contains("call tools in order"));
        assert!(records[0].content.contains("read_file -> write_file"));

        // The same rollout again merges instead of duplicating.
        collector
            .collect(&task(), &tool_rollout(1.0))
            .expect("collect again");
        assert_eq!(collector.record_count(), 1);
        assert_eq!(collector.stats().count, 1);

        // A failing rollout yields an "avoid" lesson.
        collector
            .collect(&task(), &rollout(0.0))
            .expect("collect failure");
        let records = collector.records();
        assert_eq!(records.len(), 2);
        assert!(records
            .iter()
            .any(|record| record.content.starts_with("avoid:")));
    }

    #[test]
    fn a_shared_archive_is_reused_and_errors_surface() {
        let archive = Arc::new(Mutex::new(ExperienceArchive::new()));
        let collector = TrajectoryExperienceCollector::new().with_archive(Arc::clone(&archive));
        collector.collect(&task(), &rollout(0.0)).expect("collect");
        assert_eq!(lock(&archive).len(), 1);
        assert_eq!(collector.record_count(), 1);
        assert_eq!(collector.skill_name(), "offline-rl");

        let mut bad = rollout(1.0);
        bad.rollout_info = json!("nope");
        let error = collector.collect(&task(), &bad).expect_err("bad rollout");
        assert!(error.to_string().contains("malformed rollout_info"));

        // The confidence floor of the extractor is honoured.
        let strict = TrajectoryExperienceCollector::new().with_min_confidence(1.0);
        strict.collect(&task(), &rollout(0.0)).expect("collect");
        assert_eq!(strict.record_count(), 0);

        let debug = format!("{collector:?}");
        assert!(debug.contains("offline-rl"));
        assert!(debug.contains("archive_records"));
    }
}
