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
//! - [`PromptSearchOptimizer`] — one breadth-first sweep over every candidate.
//! - [`Trainer`] — optimisation on a train split, early stop on a validation
//!   split.
//!
//! All optimizers share [`OptimizerConfig`], the tuning constants below, and one
//! deterministic tie-break: higher score, then the earlier candidate, then the
//! lexicographically smaller prompt. They are `async` because they may call a
//! model; each one also has a fully offline path.
//! A trainer can also report its evaluation as an [`EvaluationSummary`], which
//! pairs the mean score and per-case results with the tool calls behind them.

mod config;
mod example;
mod instruction;
mod joint;
mod parameter_search;
mod search;
mod suggestion;
mod trainer;

use openjiuwen_common::error::Result;

pub use config::OptimizerConfig;
pub use example::{format_prompt, render_examples, ExampleOptimizer};
pub use instruction::InstructionOptimizer;
pub use joint::JointOptimizer;
pub use parameter_search::{
    candidate_space, generate_candidates, ParameterSearchOutcome, ParameterSearcher,
    DEFAULT_CANDIDATES_SAMPLE_NUM,
};
pub use search::PromptSearchOptimizer;
pub use suggestion::placeholders;
pub use trainer::{EvaluationSummary, Trainer};

use crate::tune::base::Case;

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

/// The result every optimizer returns: the winning prompt and its score.
///
/// A reduced form of Python's `OptimizationResult` (which also carries
/// per-iteration records and token usage) that keeps the fields a ported
/// optimizer can fill in honestly. `iterations` counts the rounds (epochs,
/// selection steps) that actually ran, so an early stop shows up as a smaller
/// number.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct OptimizationResult {
    /// The prompt that scored best.
    pub best_prompt: String,
    /// The best score achieved, in `0.0..=1.0`.
    pub score: f64,
    /// How many optimisation iterations were run.
    pub iterations: usize,
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
    fn placeholders_is_reachable_from_the_module_root() {
        assert_eq!(placeholders("Answer {{question}}.").len(), 1);
    }
}
