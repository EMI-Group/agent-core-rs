//! Prompt optimisation: constants, the shared result, and the four strategies.
//!
//! Ports the orchestration surface of `openjiuwen/dev_tools/tune/` (`base.py`'s
//! `TuneConstant`, `optimizer/`, `trainer/`). The Python optimizers mutate an
//! `LLMCall` parameter in place and report through callbacks; here every
//! optimizer is a value with a bearer-style builder, returns an
//! [`OptimizationResult`], and scores candidates through the
//! [`CaseEvaluator`](crate::tune::evaluator::CaseEvaluator) seam, so the same
//! code runs offline against the deterministic evaluator or against a model.
//!
//! - [`InstructionOptimizer`] — beam search over instruction rewrites.
//! - [`ExampleOptimizer`] — greedy few-shot example selection.
//! - [`JointOptimizer`] — alternating instruction and example ascent.
//! - [`PromptSearchOptimizer`] — Python's bounded, multi-round prompt search
//!   (RLAF-P): elitist rounds over the deterministic mutation bank plus one
//!   model rewrite, scored by the composite reward with convergence, drift, and
//!   memory.
//! - [`Trainer`] — optimisation on a train split, early stop on a validation
//!   split.
//!
//! All optimizers share [`OptimizerConfig`], the tuning constants below, and one
//! deterministic tie-break: higher score, then the earlier candidate, then the
//! lexicographically smaller prompt. They are `async` because they may call a
//! model; each one also has a fully offline path.
//! [`PromptSearchConfig`] carries the prompt-search budget and stopping rules
//! (rounds, candidates per round, convergence threshold/window/target, drift
//! penalty, correctness gate, policy temperature) with the Python bounds
//! validated.
//! A trainer can also report its evaluation as an [`EvaluationSummary`], which
//! pairs the mean score and per-case results with the tool calls behind them.

mod config;
mod example;
mod instruction;
mod joint;
mod parameter_search;
mod search;
mod search_config;
mod suggestion;
mod trainer;

use std::collections::BTreeMap;

use openjiuwen_common::error::Result;
use serde_json::Value;

pub use config::OptimizerConfig;
pub use example::{format_prompt, render_examples, ExampleOptimizer};
pub use instruction::InstructionOptimizer;
pub use joint::JointOptimizer;
pub use parameter_search::{
    candidate_space, generate_candidates, ParameterSearchOutcome, ParameterSearcher,
    DEFAULT_CANDIDATES_SAMPLE_NUM,
};
pub use search::PromptSearchOptimizer;
pub use search_config::{
    PromptSearchConfig, DEFAULT_CONVERGENCE_THRESHOLD, DEFAULT_CONVERGENCE_WINDOW,
    DEFAULT_DRIFT_PENALTY, DEFAULT_MIN_CORRECTNESS, DEFAULT_POLICY_TEMPERATURE,
    DEFAULT_SEARCH_CANDIDATE_PROMPTS, DEFAULT_SEARCH_MAX_ITERATIONS,
};
pub use suggestion::placeholders;
pub use trainer::{EvaluationSummary, Trainer};

use crate::tune::base::Case;
use crate::tune::IterationRecord;

/// Python `TuneConstant.DEFAULT_EXAMPLE_NUM`.
pub const DEFAULT_EXAMPLE_NUM: usize = 1;
/// Python `TuneConstant.DEFAULT_ITERATION_NUM`.
pub const DEFAULT_ITERATION_NUM: usize = 3;
/// Python `TuneConstant.DEFAULT_PARALLEL_NUM`.
pub const DEFAULT_PARALLEL_NUM: usize = 1;
/// Python `TuneConstant.DEFAULT_EARLY_STOP_SCORE`.
pub const DEFAULT_EARLY_STOP_SCORE: f64 = 1.0;
/// Python `TuneConstant.MIN_ITERATION_NUM`.
pub const MIN_ITERATION_NUM: usize = 1;
/// Python `TuneConstant.MAX_ITERATION_NUM`.
pub const MAX_ITERATION_NUM: usize = 20;
/// Python `TuneConstant.MIN_PARALLEL_NUM`.
pub const MIN_PARALLEL_NUM: usize = 1;
/// Python `TuneConstant.MAX_PARALLEL_NUM`.
pub const MAX_PARALLEL_NUM: usize = 20;
/// Python `TuneConstant.MIN_EXAMPLE_NUM`.
pub const MIN_EXAMPLE_NUM: usize = 0;
/// Python `TuneConstant.MAX_EXAMPLE_NUM`.
pub const MAX_EXAMPLE_NUM: usize = 20;
/// Python `TuneConstant.DEFAULT_MAX_SAMPLED_EXAMPLE_NUM`.
pub const DEFAULT_MAX_SAMPLED_EXAMPLE_NUM: usize = 10;
/// Python `TuneConstant.DEFAULT_MAX_NUM_SAMPLE_ERROR_CASES`.
pub const DEFAULT_MAX_NUM_SAMPLE_ERROR_CASES: usize = 10;

/// The result every optimizer returns: the winning prompt, its score, and the
/// prompt-search bookkeeping around it.
///
/// Python's `OptimizationResult` also carries a success flag, per-iteration
/// records, the convergence verdict, token usage, and a run id; the port fills in
/// what a ported optimizer can report honestly. `iterations` stays the number of
/// rounds (epochs, selection steps) that actually ran, so an early stop shows up
/// as a smaller number. `converged`/`convergence_reason` hold the prompt-search
/// convergence verdict (`convergence_reason` is `None` when the run used its
/// whole budget), `token_usage` is empty until the evaluator reports usage,
/// `run_id` is caller-supplied (never a clock or a uuid, so a run is
/// reproducible), and `records` keeps the per-round candidates with their
/// rewards.
///
/// Only the three original fields are mandatory: every prompt-search field is
/// serde-defaulted, so JSON written before they existed still deserializes.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct OptimizationResult {
    /// The prompt that scored best.
    pub best_prompt: String,
    /// The best score achieved, in `0.0..=1.0`.
    pub score: f64,
    /// How many optimisation iterations were run.
    pub iterations: usize,
    /// Whether the search stopped on a convergence rule rather than its budget.
    pub converged: bool,
    /// The convergence reason (`None` when the run used its whole budget).
    pub convergence_reason: Option<String>,
    /// Accumulated token usage per candidate; empty when the evaluator reports none.
    pub token_usage: BTreeMap<String, Value>,
    /// Deterministic run identifier (the caller supplies the seed; never a clock/uuid).
    pub run_id: String,
    /// Per-round records: the candidates tried and their rewards.
    pub records: Vec<IterationRecord>,
}

impl OptimizationResult {
    /// Records the convergence verdict and its human-readable reason.
    #[must_use]
    pub fn with_convergence(mut self, converged: bool, reason: impl Into<String>) -> Self {
        self.converged = converged;
        self.convergence_reason = Some(reason.into());
        self
    }

    /// Replaces the accumulated token usage.
    #[must_use]
    pub fn with_token_usage(mut self, token_usage: BTreeMap<String, Value>) -> Self {
        self.token_usage = token_usage;
        self
    }

    /// Replaces the deterministic run identifier.
    #[must_use]
    pub fn with_run_id(mut self, run_id: impl Into<String>) -> Self {
        self.run_id = run_id.into();
        self
    }

    /// Replaces the per-round records.
    #[must_use]
    pub fn with_records(mut self, records: Vec<IterationRecord>) -> Self {
        self.records = records;
        self
    }

    /// Every distinct non-empty candidate prompt tried across [`Self::records`],
    /// best-first.
    ///
    /// A prompt repeated over several rounds keeps its best score, and ties keep
    /// first-appearance order, so the report is deterministic. Mirrors Python's
    /// `OptimizationResult.candidate_prompts`.
    #[must_use]
    pub fn candidate_prompts(&self) -> Vec<String> {
        let mut seen: Vec<(String, f64)> = Vec::new();
        for record in &self.records {
            for (index, candidate) in record.candidates.iter().enumerate() {
                let prompt = candidate.prompt();
                if prompt.is_empty() {
                    continue;
                }
                let score = record.scores.get(index).copied().unwrap_or(0.0);
                match seen.iter_mut().find(|(known, _)| known.as_str() == prompt) {
                    Some((_, best)) => {
                        if score > *best {
                            *best = score;
                        }
                    }
                    None => seen.push((prompt.to_owned(), score)),
                }
            }
        }
        seen.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        seen.into_iter().map(|(prompt, _)| prompt).collect()
    }
}

/// Optimizes `prompt` against `cases` with the default instruction optimizer.
///
/// Ports Python's module-level convenience function `optimize_prompt` by
/// delegating to [`InstructionOptimizer::optimize`] with the default
/// configuration.
///
/// # Errors
///
/// The same validation and evaluation errors as
/// [`InstructionOptimizer::optimize`].
pub async fn optimize_prompt(prompt: &str, cases: &[Case]) -> Result<OptimizationResult> {
    InstructionOptimizer::new().optimize(prompt, cases).await
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::Value;

    use super::*;
    use crate::tune::PromptCandidate;

    /// A minimal valid case with the given identifier.
    fn case(id: &str) -> Case {
        let mut inputs = BTreeMap::new();
        inputs.insert("question".to_owned(), Value::String(id.to_owned()));
        let mut label = BTreeMap::new();
        label.insert("answer".to_owned(), Value::String(id.to_owned()));
        Case::new(inputs, label).unwrap().with_case_id(id)
    }

    #[test]
    fn constants_match_the_python_tune_constant() {
        assert_eq!(DEFAULT_EXAMPLE_NUM, 1);
        assert_eq!(DEFAULT_ITERATION_NUM, 3);
        assert_eq!(DEFAULT_PARALLEL_NUM, 1);
        assert!((DEFAULT_EARLY_STOP_SCORE - 1.0).abs() < f64::EPSILON);
        assert_eq!(MIN_ITERATION_NUM, 1);
        assert_eq!(MAX_ITERATION_NUM, 20);
        assert_eq!(MIN_PARALLEL_NUM, 1);
        assert_eq!(MAX_PARALLEL_NUM, 20);
        assert_eq!(MIN_EXAMPLE_NUM, 0);
        assert_eq!(MAX_EXAMPLE_NUM, 20);
        assert_eq!(DEFAULT_MAX_SAMPLED_EXAMPLE_NUM, 10);
        assert_eq!(DEFAULT_MAX_NUM_SAMPLE_ERROR_CASES, 10);
        // Read the thresholds through locals so these stay plain value checks.
        for (min, default, max) in [
            (MIN_ITERATION_NUM, DEFAULT_ITERATION_NUM, MAX_ITERATION_NUM),
            (MIN_PARALLEL_NUM, DEFAULT_PARALLEL_NUM, MAX_PARALLEL_NUM),
            (MIN_EXAMPLE_NUM, DEFAULT_EXAMPLE_NUM, MAX_EXAMPLE_NUM),
        ] {
            assert!(min <= default, "{min} <= {default}");
            assert!(default <= max, "{default} <= {max}");
        }
    }

    #[test]
    fn prompt_search_defaults_are_reachable_from_the_module_root() {
        assert_eq!(DEFAULT_SEARCH_MAX_ITERATIONS, 6);
        assert_eq!(DEFAULT_SEARCH_CANDIDATE_PROMPTS, 5);
        assert!((DEFAULT_CONVERGENCE_THRESHOLD - 0.01).abs() < f64::EPSILON);
        assert_eq!(DEFAULT_CONVERGENCE_WINDOW, 3);
        assert!((DEFAULT_DRIFT_PENALTY - 0.5).abs() < f64::EPSILON);
        assert!((DEFAULT_MIN_CORRECTNESS - 0.5).abs() < f64::EPSILON);
        assert!((DEFAULT_POLICY_TEMPERATURE - 0.9).abs() < f64::EPSILON);
        assert!(PromptSearchConfig::new().validate().is_ok());
    }

    #[tokio::test]
    async fn optimize_prompt_delegates_to_the_instruction_optimizer() {
        let cases = [case("a"), case("b")];
        let via_helper = optimize_prompt("Answer the question.", &cases)
            .await
            .unwrap();
        let direct = InstructionOptimizer::new()
            .optimize("Answer the question.", &cases)
            .await
            .unwrap();
        assert_eq!(via_helper, direct);
        assert!(via_helper.score > 0.0);
    }

    #[tokio::test]
    async fn optimize_prompt_propagates_validation() {
        let error = optimize_prompt("", &[case("a")]).await.unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
    }

    #[test]
    fn optimization_result_round_trips_through_json() {
        let result = OptimizationResult {
            best_prompt: "You are helpful.".to_owned(),
            score: 0.85,
            iterations: DEFAULT_ITERATION_NUM,
            ..OptimizationResult::default()
        };
        let value = serde_json::to_value(&result).unwrap();
        let object = value.as_object().unwrap();
        for key in ["best_prompt", "score", "iterations"] {
            assert!(object.contains_key(key), "missing field {key}");
        }
        let restored: OptimizationResult = serde_json::from_value(value).unwrap();
        assert_eq!(restored, result);
    }

    #[test]
    fn legacy_json_deserializes_with_the_prompt_search_defaults() {
        let result: OptimizationResult =
            serde_json::from_str(r#"{"best_prompt": "p", "score": 0.5, "iterations": 1}"#).unwrap();
        assert_eq!(result.best_prompt, "p");
        assert!((result.score - 0.5).abs() < f64::EPSILON);
        assert_eq!(result.iterations, 1);
        assert!(!result.converged);
        assert!(result.convergence_reason.is_none());
        assert!(result.token_usage.is_empty());
        assert!(result.run_id.is_empty());
        assert!(result.records.is_empty());
    }

    #[test]
    fn prompt_search_fields_round_trip() {
        let mut token_usage = BTreeMap::new();
        token_usage.insert("prompt_tokens".to_owned(), Value::from(120));
        let mut record = IterationRecord::new(1);
        record.push(PromptCandidate::new("Alpha prompt", "seed"), 0.4);
        record.push(PromptCandidate::new("Beta prompt", "mutation"), 0.6);

        let result = OptimizationResult {
            best_prompt: "Beta prompt".to_owned(),
            score: 0.6,
            iterations: 1,
            ..OptimizationResult::default()
        }
        .with_convergence(true, "target reward 0.600 reached")
        .with_token_usage(token_usage)
        .with_run_id("run-1")
        .with_records(vec![record]);
        assert!(result.converged);
        assert_eq!(
            result.convergence_reason.as_deref(),
            Some("target reward 0.600 reached")
        );
        assert_eq!(result.token_usage["prompt_tokens"], Value::from(120));
        assert_eq!(result.run_id, "run-1");
        assert_eq!(result.records.len(), 1);

        let value = serde_json::to_value(&result).unwrap();
        let object = value.as_object().unwrap();
        for key in [
            "best_prompt",
            "score",
            "iterations",
            "converged",
            "convergence_reason",
            "token_usage",
            "run_id",
            "records",
        ] {
            assert!(object.contains_key(key), "missing field {key}");
        }
        let restored: OptimizationResult = serde_json::from_value(value).unwrap();
        assert_eq!(restored, result);
    }

    #[test]
    fn candidate_prompts_dedupe_keep_the_best_score_and_skip_empty() {
        assert!(OptimizationResult::default().candidate_prompts().is_empty());

        let mut first = IterationRecord::new(1);
        first.push(PromptCandidate::new("draft", "seed"), 0.2);
        first.push(PromptCandidate::new("", "blank"), 0.9);
        let mut second = IterationRecord::new(2);
        second.push(PromptCandidate::new("polished", "mutation"), 0.8);
        second.push(PromptCandidate::new("draft", "mutation"), 0.6);

        let result = OptimizationResult::default().with_records(vec![first, second]);
        assert_eq!(
            result.candidate_prompts(),
            vec!["polished".to_owned(), "draft".to_owned()]
        );

        // Ties keep first-appearance order.
        let mut tied = IterationRecord::new(1);
        tied.push(PromptCandidate::new("second", "a"), 0.5);
        tied.push(PromptCandidate::new("first", "b"), 0.5);
        let result = OptimizationResult::default().with_records(vec![tied]);
        assert_eq!(
            result.candidate_prompts(),
            vec!["second".to_owned(), "first".to_owned()]
        );
    }

    #[test]
    fn placeholders_is_reachable_from_the_module_root() {
        assert_eq!(placeholders("Answer {{question}}.").len(), 1);
    }
}
