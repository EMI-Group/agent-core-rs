//! Task, iteration, session, and stop-condition records of an auto-harness session.
//!
//! Ported from Python `auto_harness.schema` (`OptimizationTask`, `TaskStatus`,
//! `StageResult`, `Gap`) plus the Rust-side [`StopConditions`] policy that decides
//! when a session ends. Every record type round-trips through JSON/YAML with
//! Python field names preserved (`snake_case`, defaulted collections, tolerant of
//! unknown fields).
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::evolution::CandidateId;
use crate::harness_rsi::auto_harness::budget::Budget;
use crate::harness_rsi::auto_harness::config::AutoHarnessConfig;

/// Lifecycle state of an [`OptimizationTask`] (Python `TaskStatus`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    /// Created but not started.
    #[default]
    Pending,
    /// Currently executing.
    Running,
    /// Finished successfully.
    Success,
    /// Finished with a failure.
    Failed,
    /// Exceeded its deadline.
    Timeout,
    /// Changes were rolled back.
    Reverted,
}

/// One optimization task handed to the pipeline.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct OptimizationTask {
    /// Short task topic.
    pub topic: String,
    /// Detailed description of what should change.
    pub description: String,
    /// Files the task is expected to touch.
    pub files: Vec<String>,
    /// Upstream issue reference, when the task came from a tracker.
    pub issue_ref: Option<String>,
    /// Expected user-visible effect of the change.
    pub expected_effect: String,
    /// Python pipeline name that should execute the task.
    pub pipeline_name: String,
    /// Current status.
    pub status: TaskStatus,
}

/// Outcome of one pipeline stage (Python uses the strings `"success"`/`"failed"`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageStatus {
    /// The stage completed successfully (Python default).
    #[default]
    Success,
    /// The stage failed.
    Failed,
}

/// Result of running one pipeline stage.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct StageResult {
    /// Stage outcome.
    pub status: StageStatus,
    /// Named artifacts produced by the stage.
    pub artifacts: BTreeMap<String, Value>,
    /// Human-readable progress messages.
    pub messages: Vec<String>,
    /// Numeric metrics reported by the stage.
    pub metrics: BTreeMap<String, f64>,
    /// Failure description; empty on success.
    pub error: String,
}

/// A capability gap against a competing implementation.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct Gap {
    /// Stable gap identifier.
    pub id: String,
    /// What is missing.
    pub description: String,
    /// How much closing the gap matters.
    pub impact: f64,
    /// How feasible closing the gap is.
    pub feasibility: f64,
}

impl Gap {
    /// Priority of the gap: `impact * feasibility`.
    #[must_use]
    pub fn priority(&self) -> f64 {
        self.impact * self.feasibility
    }
}

/// Reason an auto-harness session stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// The configured iteration budget was reached.
    MaxIterations,
    /// The best score reached `target_score`.
    TargetScore,
    /// The last `patience` iterations improved too little.
    Converged,
    /// The wall-clock or cost budget is exhausted.
    BudgetExhausted,
    /// The caller cancelled the session.
    Cancelled,
    /// The session failed and cannot continue.
    Failed,
}

/// One completed optimization iteration.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct IterationRecord {
    /// Iteration number, 1-based; the first completed iteration is 1.
    pub iteration: u32,
    /// Candidates that were used as mutation parents.
    pub parents: Vec<CandidateId>,
    /// Number of candidates the mutator proposed.
    pub proposed: usize,
    /// Number of candidates that were evaluated.
    pub evaluated: usize,
    /// Number of evaluated candidates that produced a valid report.
    pub valid: usize,
    /// Best candidate of this iteration.
    pub best_candidate: Option<CandidateId>,
    /// Best score of this iteration; `None` when nothing was graded.
    pub best_score: Option<f64>,
    /// Incumbent best score before this iteration.
    pub previous_best_score: Option<f64>,
    /// Improvement of the incumbent best score in this iteration (0 when none).
    pub improvement: f64,
    /// Whether this iteration's best candidate replaced the incumbent.
    pub accepted: bool,
    /// Wall-clock duration of the iteration.
    pub duration_ms: u64,
    /// Stop reason decided after this iteration, when one applies.
    pub stop: Option<StopReason>,
}

/// Final report of one auto-harness session.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SessionSummary {
    /// Session identifier.
    pub session_id: String,
    /// Number of completed iterations.
    pub iterations: u32,
    /// Best candidate found by the session.
    pub best_candidate: Option<CandidateId>,
    /// Best score found by the session.
    pub best_score: Option<f64>,
    /// Total number of evaluated candidates.
    pub evaluated: u64,
    /// Number of accepted improvements.
    pub accepted: u64,
    /// Why the session stopped; `None` when it is still running.
    pub stop_reason: Option<StopReason>,
    /// Total cost spent, in US dollars.
    pub spent_usd: f64,
}

/// Configuration-derived policy deciding when a session stops.
///
/// Build it with [`StopConditions::from_config`]; the derived [`Default`] has
/// `max_iterations == 0` and therefore stops immediately.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct StopConditions {
    /// Hard cap on completed iterations.
    pub max_iterations: u32,
    /// Optional score that ends the session once reached.
    pub target_score: Option<f64>,
    /// Improvement below this value counts as "flat".
    pub min_improvement: f64,
    /// Number of consecutive flat iterations before declaring convergence.
    pub patience: usize,
}

impl StopConditions {
    /// Build the policy from an [`AutoHarnessConfig`].
    #[must_use]
    pub fn from_config(config: &AutoHarnessConfig) -> Self {
        Self {
            max_iterations: config.max_iterations,
            target_score: config.target_score,
            min_improvement: config.min_improvement,
            patience: config.patience,
        }
    }

    /// Decide whether the session should stop after `iteration` completed
    /// iterations.
    ///
    /// `iteration` is the count of **completed** iterations, so callers pass the
    /// number just appended to `history` (the first completed iteration is 1) and
    /// the session stops once `iteration >= max_iterations`. Cancellation is
    /// caller-supplied and deliberately not evaluated here.
    ///
    /// The checks run in a deterministic order: budget, then max iterations, then
    /// target score, then convergence. A session with no history and an
    /// unexhausted budget continues.
    #[must_use]
    pub fn evaluate(
        &self,
        history: &[IterationRecord],
        iteration: u32,
        budget: &Budget,
    ) -> StopDecision {
        if let Some(reason) = budget.exhausted() {
            return StopDecision::halt(
                reason,
                format!("budget exhausted after {iteration} completed iteration(s)"),
            );
        }
        if iteration >= self.max_iterations {
            return StopDecision::halt(
                StopReason::MaxIterations,
                format!("reached max_iterations ({})", self.max_iterations),
            );
        }
        let best = best_score(history);
        if let (Some(target), Some(best)) = (self.target_score, best) {
            if best >= target {
                return StopDecision::halt(
                    StopReason::TargetScore,
                    format!("best score {best} reached target {target}"),
                );
            }
        }
        if self.patience > 0 && history.len() >= self.patience {
            let window = &history[history.len() - self.patience..];
            if window
                .iter()
                .all(|record| record.improvement < self.min_improvement)
            {
                return StopDecision::halt(
                    StopReason::Converged,
                    format!(
                        "no improvement of at least {} in the last {} iteration(s)",
                        self.min_improvement, self.patience
                    ),
                );
            }
        }
        StopDecision::go()
    }
}

/// Outcome of [`StopConditions::evaluate`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct StopDecision {
    /// Whether the session should stop.
    pub stop: bool,
    /// Why it should stop; always `Some` when `stop` is true.
    pub reason: Option<StopReason>,
    /// Human-readable justification, used in logs and events.
    pub detail: String,
}

impl StopDecision {
    /// A decision to keep running.
    #[must_use]
    pub fn go() -> Self {
        Self {
            stop: false,
            reason: None,
            detail: String::from("continue"),
        }
    }

    /// A decision to stop for `reason`.
    #[must_use]
    pub fn halt(reason: StopReason, detail: impl Into<String>) -> Self {
        Self {
            stop: true,
            reason: Some(reason),
            detail: detail.into(),
        }
    }
}

/// Highest `best_score` across `history`, ignoring iterations without a score.
fn best_score(history: &[IterationRecord]) -> Option<f64> {
    history
        .iter()
        .filter_map(|record| record.best_score)
        .reduce(f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    use crate::harness_rsi::auto_harness::budget::{Budget, BudgetLimits};

    fn candidate_id(id: &str) -> CandidateId {
        serde_json::from_value(Value::String(id.to_owned())).unwrap()
    }

    fn healthy_budget() -> Budget {
        Budget::start(BudgetLimits {
            wall_clock: Duration::from_secs(3_600),
            cost_limit_usd: 10.0,
            task_timeout: Duration::from_secs(60),
        })
    }

    fn record(iteration: u32, best: f64, improvement: f64) -> IterationRecord {
        IterationRecord {
            iteration,
            best_score: Some(best),
            previous_best_score: Some(best - improvement),
            improvement,
            accepted: improvement > 0.0,
            ..IterationRecord::default()
        }
    }

    #[test]
    fn task_status_serde_names() {
        let yaml = serde_yaml_ng::to_string(&TaskStatus::Reverted).unwrap();
        assert_eq!(yaml.trim(), "reverted");
        assert_eq!(TaskStatus::default(), TaskStatus::Pending);
        assert_eq!(StageStatus::default(), StageStatus::Success);
        assert_eq!(OptimizationTask::default().status, TaskStatus::Pending);
    }

    #[test]
    fn stage_result_round_trips() {
        let mut artifacts = BTreeMap::new();
        artifacts.insert("diff".to_owned(), Value::String("+1".to_owned()));
        let mut metrics = BTreeMap::new();
        metrics.insert("duration_ms".to_owned(), 12.5);
        let result = StageResult {
            status: StageStatus::Failed,
            artifacts,
            messages: vec!["boom".to_owned()],
            metrics,
            error: "failed".to_owned(),
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: StageResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, result);
    }

    #[test]
    fn optimization_task_tolerates_unknown_fields() {
        let task: OptimizationTask = serde_json::from_value(serde_json::json!({
            "topic": "perf",
            "future_field": 1,
        }))
        .unwrap();
        assert_eq!(task.topic, "perf");
        assert_eq!(task.status, TaskStatus::Pending);
        assert_eq!(task.files, Vec::<String>::new());
    }

    #[test]
    fn gap_priority_is_impact_times_feasibility() {
        let gap = Gap {
            id: "g1".to_owned(),
            description: "missing tool".to_owned(),
            impact: 0.5,
            feasibility: 0.4,
        };
        assert!((gap.priority() - 0.2).abs() < f64::EPSILON);
        assert_eq!(Gap::default().priority(), 0.0);
    }

    #[test]
    fn iteration_record_round_trips_through_json() {
        let record = IterationRecord {
            iteration: 3,
            parents: vec![candidate_id("c0"), candidate_id("c1")],
            proposed: 2,
            evaluated: 2,
            valid: 1,
            best_candidate: Some(candidate_id("c0/g1/v0")),
            best_score: Some(0.75),
            previous_best_score: Some(0.5),
            improvement: 0.25,
            accepted: true,
            duration_ms: 42,
            stop: Some(StopReason::Converged),
        };
        let json = serde_json::to_string(&record).unwrap();
        let parsed: IterationRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, record);
        assert!(json.contains("\"stop\":\"converged\""));
    }

    #[test]
    fn session_summary_round_trips_and_defaults_collections() {
        let summary = SessionSummary {
            session_id: "s1".to_owned(),
            iterations: 4,
            best_candidate: Some(candidate_id("c9")),
            best_score: Some(0.9),
            evaluated: 12,
            accepted: 2,
            stop_reason: Some(StopReason::BudgetExhausted),
            spent_usd: 1.5,
        };
        let json = serde_json::to_string(&summary).unwrap();
        let parsed: SessionSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, summary);

        let empty: SessionSummary = serde_json::from_str("{}").unwrap();
        assert_eq!(empty, SessionSummary::default());
        assert!(empty.stop_reason.is_none());
    }

    #[test]
    fn stop_reason_serde_names() {
        let names = [
            (StopReason::MaxIterations, "max_iterations"),
            (StopReason::TargetScore, "target_score"),
            (StopReason::Converged, "converged"),
            (StopReason::BudgetExhausted, "budget_exhausted"),
            (StopReason::Cancelled, "cancelled"),
            (StopReason::Failed, "failed"),
        ];
        for (reason, name) in names {
            let value = serde_json::to_value(reason).unwrap();
            assert_eq!(value, Value::String(name.to_owned()));
        }
    }

    #[test]
    fn from_config_maps_fields() {
        let config = AutoHarnessConfig {
            max_iterations: 7,
            target_score: Some(0.8),
            min_improvement: 0.1,
            patience: 3,
            ..AutoHarnessConfig::default()
        };
        let conditions = StopConditions::from_config(&config);
        assert_eq!(conditions.max_iterations, 7);
        assert_eq!(conditions.target_score, Some(0.8));
        assert_eq!(conditions.min_improvement, 0.1);
        assert_eq!(conditions.patience, 3);
    }

    #[test]
    fn budget_beats_max_iterations_beats_target_beats_convergence() {
        let exhausted = Budget::start(BudgetLimits {
            wall_clock: Duration::from_secs(60),
            cost_limit_usd: 0.0,
            task_timeout: Duration::from_secs(1),
        });
        let conditions = StopConditions {
            max_iterations: 1,
            target_score: Some(0.5),
            min_improvement: 0.05,
            patience: 1,
        };
        let history = vec![record(1, 0.9, 0.0)];
        let decision = conditions.evaluate(&history, 1, &exhausted);
        assert_eq!(decision.reason, Some(StopReason::BudgetExhausted));

        let decision = conditions.evaluate(&history, 1, &healthy_budget());
        assert_eq!(decision.reason, Some(StopReason::MaxIterations));

        let conditions = StopConditions {
            max_iterations: 10,
            ..conditions
        };
        let decision = conditions.evaluate(&history, 1, &healthy_budget());
        assert_eq!(decision.reason, Some(StopReason::TargetScore));

        let conditions = StopConditions {
            target_score: None,
            ..conditions
        };
        let decision = conditions.evaluate(&history, 1, &healthy_budget());
        assert_eq!(decision.reason, Some(StopReason::Converged));
        assert!(decision.stop);
    }

    #[test]
    fn improving_history_continues() {
        let conditions = StopConditions {
            max_iterations: 10,
            target_score: None,
            min_improvement: 0.05,
            patience: 2,
        };
        let history = vec![record(1, 0.1, 0.1), record(2, 0.4, 0.3)];
        let decision = conditions.evaluate(&history, 2, &healthy_budget());
        assert!(!decision.stop);
        assert_eq!(decision.reason, None);
        assert_eq!(decision, StopDecision::go());
    }

    #[test]
    fn convergence_requires_a_full_patience_window() {
        let conditions = StopConditions {
            max_iterations: 100,
            target_score: None,
            min_improvement: 0.05,
            patience: 2,
        };
        let partial = vec![record(1, 0.1, 0.0)];
        assert!(!conditions.evaluate(&partial, 1, &healthy_budget()).stop);

        let flat = vec![record(1, 0.1, 0.01), record(2, 0.11, 0.01)];
        let decision = conditions.evaluate(&flat, 2, &healthy_budget());
        assert_eq!(decision.reason, Some(StopReason::Converged));

        let recovered = vec![
            record(1, 0.1, 0.0),
            record(2, 0.4, 0.3),
            record(3, 0.41, 0.01),
        ];
        assert!(!conditions.evaluate(&recovered, 3, &healthy_budget()).stop);
    }

    #[test]
    fn target_uses_best_score_from_history() {
        let conditions = StopConditions {
            max_iterations: 10,
            target_score: Some(0.8),
            min_improvement: 0.05,
            patience: 5,
        };
        let history = vec![record(1, 0.9, 0.0), record(2, 0.2, 0.0)];
        let decision = conditions.evaluate(&history, 2, &healthy_budget());
        assert_eq!(decision.reason, Some(StopReason::TargetScore));
    }

    #[test]
    fn stop_decision_constructors() {
        let go = StopDecision::go();
        assert!(!go.stop);
        assert_eq!(go.reason, None);
        let halt = StopDecision::halt(StopReason::Failed, "bad state");
        assert!(halt.stop);
        assert_eq!(halt.reason, Some(StopReason::Failed));
        assert_eq!(halt.detail, "bad state");
    }

    #[test]
    fn stop_conditions_round_trip() {
        let conditions = StopConditions {
            max_iterations: 5,
            target_score: Some(0.7),
            min_improvement: 0.02,
            patience: 1,
        };
        let json = serde_json::to_string(&conditions).unwrap();
        let parsed: StopConditions = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, conditions);
        let defaulted: StopConditions = serde_json::from_str("{}").unwrap();
        assert_eq!(defaulted, StopConditions::default());
    }
}
