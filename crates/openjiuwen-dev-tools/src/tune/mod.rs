//! Prompt tuning: cases, evaluators, optimizers, and a training loop.
//!
//! Ports `openjiuwen/dev_tools/tune/` onto three cooperating layers:
//!
//! - **Data** — [`Case`] is one tuning example (the model [`inputs`](Case::inputs),
//!   the expected [`label`](Case::label), optional [`tools`](Case::tools), and a
//!   [`case_id`](Case::case_id)); [`EvaluatedCase`] adds the [`answer`](EvaluatedCase::answer),
//!   [`score`](EvaluatedCase::score), and [`reason`](EvaluatedCase::reason) an
//!   evaluator produced; [`CaseLoader`] is an in-memory collection with
//!   deterministic [`shuffle`](CaseLoader::shuffle) and [`split`](CaseLoader::split).
//! - **Evaluation** — [`CaseEvaluator`] scores a candidate prompt over cases.
//!   [`DeterministicEvaluator`] does so offline from placeholder preservation,
//!   instruction structure, case-term coverage, and hygiene;
//!   [`ModelEvaluator`] runs the prompt through a [`Model`] and scores the answer
//!   with a [`CaseScorer`] (default [`LabelScorer`]); [`ScriptedPromptRunner`]
//!   replays canned answers for tests.
//! - **Search** — [`InstructionOptimizer`] (beam search over rewrites),
//!   [`ExampleOptimizer`] (greedy few-shot selection), [`JointOptimizer`]
//!   (alternating both), [`PromptSearchOptimizer`] (one breadth-first sweep),
//!   and [`Trainer`] (train/validation with early stop) share
//!   [`OptimizerConfig`] and the tuning constants ([`DEFAULT_ITERATION_NUM`],
//!   [`MAX_PARALLEL_NUM`], …). A [`Trainer`] can also report its evaluation as
//!   an [`EvaluationSummary`], which adds the per-case tool usage to the mean
//!   score and results.
//!
//! A case may declare tools, and the tool layer makes them runnable and
//! observable: [`ToolStub`] is a named in-process fixture, [`ToolStubs`] the table
//! a runner looks them up in, [`ToolTrace`] the ordered [`ToolCallRecord`]s of one
//! run, and [`RunOutcome`] an answer map bundled with that trace — so an
//! evaluator can execute the tools a case declares and report what was used.
//!
//! Every optimizer runs offline with no arguments: the default evaluator is the
//! deterministic one and the default candidate source is the deterministic
//! mutation bank. Attaching a [`Model`] adds model rewrites without making the
//! run depend on it — a failed or unparsable answer falls back to the bank.
//!
//! [`Model`]: openjiuwen_foundation::llm::Model
//!
//! # Example
//!
//! ```
//! # use std::collections::BTreeMap;
//! # use openjiuwen_dev_tools::tune::{Case, CaseLoader, InstructionOptimizer};
//! # use serde_json::Value;
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() -> Result<(), openjiuwen_common::error::Error> {
//! let mut inputs = BTreeMap::new();
//! inputs.insert("question".to_owned(), Value::String("2 + 2?".to_owned()));
//! let mut label = BTreeMap::new();
//! label.insert("answer".to_owned(), Value::String("4".to_owned()));
//!
//! let case = Case::new(inputs, label)?;
//! let loader = CaseLoader::new([case]);
//! let (train, test) = loader.split(0.5)?;
//! assert_eq!(train.size() + test.size(), 1);
//!
//! let result = InstructionOptimizer::new()
//!     .optimize("Answer the question.", loader.cases())
//!     .await?;
//! assert!(result.score > 0.0);
//! # Ok(())
//! # }
//! ```

mod base;
mod evaluator;
mod loader;
mod optimizer;
mod prompt_search;
mod tool;

pub use base::{Case, EvaluatedCase};
pub use evaluator::{
    render_case_inputs, tool_usage, CaseEvaluator, CaseScorer, DeterministicEvaluator,
    DeterministicScore, LabelScorer, ModelEvaluator, ModelPromptRunner, PromptCall, PromptRunner,
    RunnerEvaluator, ScriptedPromptRunner, ToolUsageScorer, DEFAULT_MAX_TOOL_ROUNDS,
};
pub use loader::CaseLoader;
pub use optimizer::{
    format_prompt, optimize_prompt, placeholders, render_examples, EvaluationSummary,
    ExampleOptimizer, InstructionOptimizer, JointOptimizer, OptimizationResult, OptimizerConfig,
    PromptSearchOptimizer, Trainer,
};
pub use optimizer::{
    DEFAULT_EARLY_STOP_SCORE, DEFAULT_EXAMPLE_NUM, DEFAULT_ITERATION_NUM, DEFAULT_PARALLEL_NUM,
    MAX_EXAMPLE_NUM, MAX_ITERATION_NUM, MAX_PARALLEL_NUM, MIN_EXAMPLE_NUM, MIN_ITERATION_NUM,
    MIN_PARALLEL_NUM,
};
pub use prompt_search::{
    candidate_id, clamp01, default_weights, min_max_normalize, population_variance, CaseResult,
    CompletenessReward, CompositeReward, ConvergenceDetector, ConvergenceState, CorrectnessReward,
    CostReward, CustomReward, DriftJudge, Evaluation, Execution, HistoryCompressor, HistoryEntry,
    IterationRecord, JsonlPromptMemory, LatencyReward, LexicalDriftJudge, ModelDriftJudge,
    NullDriftJudge, NullPromptMemory, OptimizationHistory, PromptCandidate, PromptMemory,
    PromptRecord, PromptTaskCase, PromptTaskSpec, RewardBreakdown, RewardComponent,
    StructuredValidationReward, TokenUsageReward, OVERFIT_MARGIN, OVERFIT_PENALTY,
};
pub use tool::{RunOutcome, ToolCallRecord, ToolHandler, ToolStub, ToolStubs, ToolTrace};

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

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
    fn re_exports_compose_into_a_loader_workflow() {
        let loader = CaseLoader::new([case("a"), case("b")]);
        assert_eq!(loader.size(), 2);

        let shuffled = loader.shuffle(DEFAULT_PARALLEL_NUM as u64);
        assert_eq!(shuffled.size(), 2);

        let (train, test) = shuffled.split(0.5).unwrap();
        assert_eq!(train.size(), 1);
        assert_eq!(test.size(), 1);

        let evaluated = EvaluatedCase::from(test.into_cases().remove(0));
        assert!(evaluated.validate().is_ok());
        assert!(evaluated.score.abs() < f64::EPSILON);
    }

    #[test]
    fn re_exported_constants_are_usable_from_the_module_root() {
        assert_eq!(MIN_ITERATION_NUM, 1);
        assert_eq!(MAX_ITERATION_NUM, 20);
        assert!((DEFAULT_EARLY_STOP_SCORE - 1.0).abs() < f64::EPSILON);
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
    async fn re_exported_pipeline_entry_points_do_real_work() {
        let cases = [case("a"), case("b")];
        let result = optimize_prompt("Answer the question.", &cases)
            .await
            .unwrap();
        assert!(result.score > 0.0);
        assert!(result.iterations >= 1);
        assert_ne!(result.best_prompt, "");
    }

    #[tokio::test]
    async fn evaluator_seam_scores_a_scripted_runner() {
        let runner = Arc::new(
            ScriptedPromptRunner::new()
                .with_text_answer("a", "a")
                .with_text_answer("b", "wrong"),
        );
        let evaluator = RunnerEvaluator::new(runner);
        let (mean, evaluated) = evaluator
            .score_cases("prompt", &[case("a"), case("b")])
            .await
            .unwrap();
        assert_eq!(evaluated.len(), 2);
        assert!((mean - 0.5).abs() < 1e-9, "one of two matched: {mean}");
    }

    #[test]
    fn deterministic_evaluator_is_reachable_from_the_module_root() {
        let evaluator = DeterministicEvaluator::new("Answer {{question}}.");
        let parts = evaluator.score_parts("Answer {{question}}.", &case("a"));
        assert!((parts.placeholder - 1.0).abs() < 1e-9);
        assert_eq!(evaluator.seed_placeholders().len(), 1);
    }

    #[test]
    fn label_scorer_is_reachable_from_the_module_root() {
        let mut answer = BTreeMap::new();
        answer.insert("answer".to_owned(), Value::String("a".to_owned()));
        let (score, _) = LabelScorer::new().score(&case("a"), &answer);
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn config_and_placeholders_are_reachable_from_the_module_root() {
        assert!(OptimizerConfig::default().validate().is_ok());
        assert_eq!(placeholders("{{a}} {{b}}").len(), 2);
        assert_eq!(render_case_inputs(&case("a")), "[question]: a");
    }
}
