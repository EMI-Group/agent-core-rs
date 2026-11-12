//! Tuning constants and the deferred optimizer/trainer entry points.
//!
//! Ports `TuneConstant` and the public optimizer surface of
//! `openjiuwen/dev_tools/tune/` (`optimizer/`, `trainer/`). The Python
//! instruction, example, joint, and prompt-search optimizers, and the trainer,
//! depend on Python-only optimisation internals (prompt-diffing agents, sampler
//! services, regression pipelines). Here they keep their API shape and constants
//! but every entry point returns [`Error::unsupported`] naming the deferred
//! pipeline and pointing at `docs/PARITY.md`, so callers and future ports share
//! one stable signature set.

// The deferred entry points keep the `async` signatures of the Python
// `async def optimize` coroutines even though they resolve immediately.
#![allow(clippy::unused_async_trait_impl)]

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

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
/// optimizer can honestly fill in.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizationResult {
    /// The prompt that scored best.
    pub best_prompt: String,
    /// The best score achieved, normally in `0.0..=1.0`.
    pub score: f64,
    /// How many optimisation iterations were run.
    pub iterations: usize,
}

/// The deferred instruction optimizer.
///
/// Python: `openjiuwen.dev_tools.tune.optimizer.InstructionOptimizer`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InstructionOptimizer;

impl InstructionOptimizer {
    /// Creates the optimizer; no configuration is available before the port.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Rewrites the instructions of `prompt` against `cases`.
    ///
    /// # Errors
    ///
    /// Always [`Error::unsupported`]: the Python pipeline is not ported.
    pub async fn optimize(&self, prompt: &str, cases: &[Case]) -> Result<OptimizationResult> {
        let _ = (prompt, cases);
        Err(Error::unsupported(
            "InstructionOptimizer is not ported: the Python instruction-optimization pipeline \
             depends on Python-only optimizer internals; see docs/PARITY.md",
        ))
    }
}

/// The deferred few-shot example optimizer.
///
/// Python: `openjiuwen.dev_tools.tune.optimizer.ExampleOptimizer`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExampleOptimizer;

impl ExampleOptimizer {
    /// Creates the optimizer; no configuration is available before the port.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Selects up to `num_examples` examples for `prompt` from `cases`.
    ///
    /// # Errors
    ///
    /// Always [`Error::unsupported`]: the Python pipeline is not ported.
    pub async fn optimize(
        &self,
        prompt: &str,
        cases: &[Case],
        num_examples: usize,
    ) -> Result<OptimizationResult> {
        let _ = (prompt, cases, num_examples);
        Err(Error::unsupported(
            "ExampleOptimizer is not ported: the Python few-shot example-selection pipeline \
             depends on Python-only optimizer internals; see docs/PARITY.md",
        ))
    }
}

/// The deferred joint instruction-and-example optimizer.
///
/// Python: `openjiuwen.dev_tools.tune.optimizer.JointOptimizer`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct JointOptimizer;

impl JointOptimizer {
    /// Creates the optimizer; no configuration is available before the port.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Optimizes instructions and examples together for `prompt` against `cases`.
    ///
    /// # Errors
    ///
    /// Always [`Error::unsupported`]: the Python pipeline is not ported.
    pub async fn optimize(&self, prompt: &str, cases: &[Case]) -> Result<OptimizationResult> {
        let _ = (prompt, cases);
        Err(Error::unsupported(
            "JointOptimizer is not ported: the Python joint instruction-and-example pipeline \
             depends on Python-only optimizer internals; see docs/PARITY.md",
        ))
    }
}

/// The deferred tuning trainer.
///
/// Python: `openjiuwen.dev_tools.tune.trainer.Trainer`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Trainer;

impl Trainer {
    /// Creates the trainer; no configuration is available before the port.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Trains `prompt` on `train` and reports the result measured on `val`.
    ///
    /// # Errors
    ///
    /// Always [`Error::unsupported`]: the Python trainer is not ported.
    pub fn train(&self, prompt: &str, train: &[Case], val: &[Case]) -> Result<OptimizationResult> {
        let _ = (prompt, train, val);
        Err(Error::unsupported(
            "Trainer is not ported: the Python trainer depends on Python-only optimizer and \
             evaluator internals; see docs/PARITY.md",
        ))
    }
}

/// The deferred prompt-search optimizer.
///
/// Python: `openjiuwen.dev_tools.tune.optimizer.prompt_search.PromptSearchOptimizer`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PromptSearchOptimizer;

impl PromptSearchOptimizer {
    /// Creates the optimizer; no configuration is available before the port.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Searches for a better prompt for `cases` starting from `prompt`.
    ///
    /// # Errors
    ///
    /// Always [`Error::unsupported`]: the Python prompt-search pipeline is not
    /// ported.
    pub async fn optimize(&self, prompt: &str, cases: &[Case]) -> Result<OptimizationResult> {
        let _ = (prompt, cases);
        Err(Error::unsupported(
            "PromptSearchOptimizer is not ported: the Python prompt-search pipeline depends on \
             Python-only optimizer internals; see docs/PARITY.md",
        ))
    }
}

/// Optimizes `prompt` against `cases` with the default optimizer.
///
/// Ports Python's module-level convenience function `optimize_prompt` by
/// delegating to [`InstructionOptimizer::optimize`].
///
/// # Errors
///
/// Always [`Error::unsupported`]: the instruction optimizer is not ported.
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

    /// Asserts that `error` is an unsupported-pipeline error mentioning the
    /// deferred pipeline and the parity document.
    fn assert_deferred(error: Error, pipeline: &str) {
        match &error {
            Error::Unsupported(message) => {
                assert!(
                    message.contains(pipeline),
                    "message should name {pipeline}: {message}"
                );
                assert!(
                    message.contains("docs/PARITY.md"),
                    "message should point at docs/PARITY.md: {message}"
                );
            }
            other => panic!("expected Error::Unsupported, got {other:?}"),
        }
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
    async fn instruction_optimizer_is_deferred() {
        let cases = [case("a"), case("b")];
        let error = InstructionOptimizer::new()
            .optimize("prompt", &cases)
            .await
            .unwrap_err();
        assert_deferred(error, "InstructionOptimizer");
    }

    #[tokio::test]
    async fn example_optimizer_is_deferred() {
        let cases = [case("a")];
        let error = ExampleOptimizer::new()
            .optimize("prompt", &cases, DEFAULT_EXAMPLE_NUM)
            .await
            .unwrap_err();
        assert_deferred(error, "ExampleOptimizer");
    }

    #[tokio::test]
    async fn joint_optimizer_is_deferred() {
        let cases = [case("a")];
        let error = JointOptimizer::new()
            .optimize("prompt", &cases)
            .await
            .unwrap_err();
        assert_deferred(error, "JointOptimizer");
    }

    #[test]
    fn trainer_is_deferred() {
        let train = [case("a")];
        let val = [case("b")];
        let error = Trainer::new().train("prompt", &train, &val).unwrap_err();
        assert_deferred(error, "Trainer");
    }

    #[tokio::test]
    async fn prompt_search_optimizer_is_deferred() {
        let cases = [case("a")];
        let error = PromptSearchOptimizer::new()
            .optimize("prompt", &cases)
            .await
            .unwrap_err();
        assert_deferred(error, "PromptSearchOptimizer");
    }

    #[tokio::test]
    async fn optimize_prompt_delegates_to_the_instruction_optimizer() {
        let cases = [case("a")];
        let error = optimize_prompt("prompt", &cases).await.unwrap_err();
        let direct = InstructionOptimizer::new()
            .optimize("prompt", &cases)
            .await
            .unwrap_err();
        assert_eq!(error.to_string(), direct.to_string());
        assert_deferred(error, "InstructionOptimizer");
    }

    #[tokio::test]
    async fn stubs_reject_empty_inputs_identically() {
        // Empty case lists are accepted by the stubs: validation belongs to the
        // (deferred) implementation, so the deferred error still comes back.
        let error = InstructionOptimizer::new()
            .optimize("", &[])
            .await
            .unwrap_err();
        assert_deferred(error, "InstructionOptimizer");
    }

    /// Asserts that `value` equals its type's `Default` value.
    fn assert_default_constructible<T: Default + PartialEq + std::fmt::Debug>(value: T) {
        assert_eq!(T::default(), value);
    }

    #[test]
    fn stub_structs_are_default_constructible() {
        assert_default_constructible(InstructionOptimizer::new());
        assert_default_constructible(ExampleOptimizer::new());
        assert_default_constructible(JointOptimizer::new());
        assert_default_constructible(Trainer::new());
        assert_default_constructible(PromptSearchOptimizer::new());
        // The stubs carry no state, so their debug output is just the type name.
        assert_eq!(
            format!("{:?}", InstructionOptimizer::new()),
            "InstructionOptimizer"
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
}
