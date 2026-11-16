//! The prompt-search (RLAF-P) domain layer.
//!
//! Ports the data/reward/memory half of
//! `openjiuwen/dev_tools/tune/optimizer/prompt_search/`: the task and execution
//! vocabulary, the composite reward model, convergence detection, the drift
//! judge, the prompt memory, the history compressor, and the JSONL run log. The
//! optimization loop itself (policy, environment) lives above this node; this
//! module is the vocabulary and the deterministic mechanics it builds on.
//!
//! - [`PromptTaskSpec`]/[`PromptTaskCase`] describe what is optimized for, with
//!   held-out cases for overfitting detection.
//! - [`PromptCandidate`]/[`Execution`]/[`CaseResult`] describe one run, and
//!   [`Evaluation`]/[`IterationRecord`] one round of comparisons.
//! - [`CompositeReward`] turns executions into [`RewardBreakdown`]s: a weighted
//!   mean of [`RewardComponent`]s, plus a min-correctness gate, an overfitting
//!   penalty, and a drift penalty.
//! - [`ConvergenceDetector`] decides when further iterations cannot help.
//! - [`DriftJudge`] protects the original objective; [`LexicalDriftJudge`] is the
//!   deterministic default and [`ModelDriftJudge`] the optional model judge.
//! - [`PromptMemory`] persists outcomes and retrieves similar prior runs, with
//!   [`JsonlPromptMemory`] as the dependency-free default.
//! - [`OptimizationHistory`] and [`HistoryCompressor`] turn iterations into a
//!   budgeted textual "policy update".
//! - [`RunLog`] records one [`RunLogRecord`] per optimization round, with
//!   [`JsonlRunLog`] appending to a file, [`InMemoryRunLog`] keeping the records
//!   for tests, and [`NullRunLog`] discarding; [`read_run_log`] and
//!   [`read_run_log_limited`] read a log back tolerantly.
//!
//! Everything here is deterministic and offline: identifiers come from
//! [`candidate_id`] (FNV-1a) instead of `uuid4`, timestamps are caller-supplied,
//! and no function reads the clock, the environment, or a random number
//! generator. Python's async components become synchronous where they do no
//! I/O; only the drift judge keeps an asynchronous seam because it may call a
//! model.
//!
//! This node supplies the vocabulary — models, rewards, convergence, drift,
//! memory, history, and the run-log sink — that the bounded optimization loop in
//! `crate::tune::optimizer::search` drives.
//! Still unported from Python (see the crate `CONTEXT.md`): `policy.py` (LLM
//! multi-candidate generation), `environment.py` (LLM/callable environments),
//! and `service.py`, because candidate generation here is the deterministic
//! mutation bank plus one model rewrite per round and candidates execute
//! through the caller's `CaseEvaluator`.
//! Python's LLM-backed history buckets, embedding-backed memory, and LLM
//! correctness judge are likewise replaced by deterministic equivalents, and a
//! model answer never fails a run: an unusable answer degrades to a documented
//! deterministic value.
//!
//! # Example
//!
//! ```
//! use std::collections::BTreeMap;
//! use openjiuwen_dev_tools::tune::{
//!     CaseResult, CompositeReward, Execution, PromptCandidate, PromptTaskSpec,
//! };
//!
//! let mut execution = Execution::new(PromptCandidate::new("Answer briefly.", "seed"));
//! execution.latency_ms = 1000.0;
//! execution.total_tokens = 500;
//! execution.push_result(CaseResult::new("c1", "42").with_score(1.0));
//!
//! let task = PromptTaskSpec::new("answer arithmetic questions");
//! let breakdowns = CompositeReward::default()
//!     .evaluate(&[execution], &task, &BTreeMap::new());
//! assert!(breakdowns[0].score > 0.0);
//! assert!(!breakdowns[0].gated);
//! ```

mod convergence;
mod drift;
mod history;
mod memory;
mod models;
mod reward;
mod run_log;

pub use run_log::{
    read_run_log, read_run_log_limited, InMemoryRunLog, JsonlRunLog, NullRunLog, RunLog,
    RunLogRecord,
};

pub use convergence::{population_variance, ConvergenceDetector, ConvergenceState};
pub use drift::{DriftJudge, LexicalDriftJudge, ModelDriftJudge, NullDriftJudge};
pub use history::{HistoryCompressor, HistoryEntry, OptimizationHistory};
pub use memory::{JsonlPromptMemory, NullPromptMemory, PromptMemory};
pub use models::{
    candidate_id, CaseResult, Evaluation, Execution, IterationRecord, PromptCandidate,
    PromptRecord, PromptTaskCase, PromptTaskSpec, RewardBreakdown,
};
pub use reward::{
    clamp01, default_weights, min_max_normalize, CompletenessReward, CompositeReward,
    CorrectnessReward, CostReward, CustomReward, LatencyReward, RewardComponent,
    StructuredValidationReward, TokenUsageReward, OVERFIT_MARGIN, OVERFIT_PENALTY,
};

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_test_support::ScriptedModel;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    #[test]
    fn re_exports_compose_into_one_round() {
        let task = PromptTaskSpec::new("write a haiku")
            .with_cases(vec![
                PromptTaskCase::from_text("autumn", "leaf").unwrap(),
                PromptTaskCase::from_text("winter", "snow")
                    .unwrap()
                    .with_hidden(true),
            ])
            .with_constraints(vec!["three lines".to_owned()]);
        let characteristics = task.characteristics();
        assert!(characteristics.contains("constraints: three lines"));

        let mut best = Execution::new(PromptCandidate::new("write a haiku", "seed"));
        best.push_result(CaseResult::new("v", "leaf").with_score(0.9));
        best.push_result(
            CaseResult::new("h", "snow")
                .with_score(0.8)
                .with_hidden(true),
        );
        let mut worst = Execution::new(PromptCandidate::new("write an essay", "").with_id("w"));
        worst.push_result(CaseResult::new("v", "").with_score(0.0));

        let reward = CompositeReward::new(
            vec![Box::new(CorrectnessReward::new())],
            default_weights()
                .into_iter()
                .filter(|(name, _)| name == "correctness")
                .collect(),
        );
        let drift = BTreeMap::from([("w".to_owned(), 0.8)]);
        let breakdowns = reward.evaluate(&[best.clone(), worst], &task, &drift);
        assert!(breakdowns[0].score > breakdowns[1].score);
        assert_eq!(breakdowns[0].notes.len(), 0);
        assert!(breakdowns[1].score < 1e-12);

        let mut record = IterationRecord::new(1);
        record.push(best.candidate.clone(), breakdowns[0].score);
        assert_eq!(
            record.best_candidate().unwrap().candidate_id,
            best.candidate.candidate_id
        );

        let mut history = OptimizationHistory::default();
        history.add(
            HistoryEntry::new(1, breakdowns[0].score, best.candidate.prompt.clone())
                .with_observations(vec!["kept the form".to_owned()]),
        );
        assert!(HistoryCompressor::new()
            .compress(&history)
            .contains("Best so far"));
        assert_eq!(history.best_prompt(), "write a haiku");

        let detector = ConvergenceDetector::new(0.01, 2, Some(0.5)).unwrap();
        assert_eq!(
            detector.check(&[0.9]).map(|state| state.converged),
            Some(true)
        );

        let memory = NullPromptMemory::new();
        memory
            .store(PromptRecord::new("r", "p", 1.0, "objective"))
            .unwrap();
        assert_eq!(memory.pending(0.0).len(), 0);

        assert_eq!(clamp01(2.0), 1.0);
        assert_eq!(min_max_normalize(&[1.0, 2.0]), vec![0.0, 1.0]);
        assert!((OVERFIT_MARGIN - 0.25).abs() < 1e-12);
        assert!((OVERFIT_PENALTY - 0.5).abs() < 1e-12);
        assert_eq!(population_variance(&[1.0]), 0.0);
        assert_eq!(candidate_id("prompt", 0).len(), 8);
    }

    #[tokio::test]
    async fn judges_are_reachable_from_the_module_root() {
        let candidate = PromptCandidate::new("write a haiku", "");
        assert_eq!(
            NullDriftJudge::new()
                .deviation("write a haiku", &candidate)
                .await,
            0.0
        );
        let lexical = LexicalDriftJudge::new();
        assert_eq!(lexical.deviation("write a haiku", &candidate).await, 0.0);

        let model = Arc::new(ScriptedModel::text("```json\n{\"deviation\": 0.1}\n```"));
        let judge = ModelDriftJudge::new(model);
        assert!((judge.deviation("write a haiku", &candidate).await - 0.1).abs() < 1e-12);
    }

    #[test]
    fn memory_and_validation_error_types_are_exposed() {
        use openjiuwen_common::error::ErrorKind;

        let workspace = openjiuwen_test_support::TempWorkspace::new().unwrap();
        let memory = JsonlPromptMemory::new(workspace.path("memory")).unwrap();
        assert_eq!(memory.records().len(), 0);

        let error = ConvergenceDetector::new(0.0, 3, None).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
    }
}
