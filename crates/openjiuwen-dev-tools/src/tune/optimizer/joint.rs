//! [`JointOptimizer`]: alternating instruction and example optimisation.
//!
//! Python's `JointOptimizer` randomly alternates between the instruction
//! optimizer's textual-gradient update and the example optimizer's selection
//! (`random.choice([True, False])` per round) and keeps whichever parameter was
//! updated last.
//!
//! The port replaces the coin flip with coordinate ascent that can be measured:
//!
//! 1. Measure three starting points with the shared evaluator: the seed, the
//!    result of an instruction-only search, and the result of an example-only
//!    selection. Keeping those as baselines is what makes the joint result never
//!    worse than either optimizer run alone.
//! 2. Per round: run one instruction round from the current best prompt, then
//!    select examples for that rewritten prompt, and score both.
//! 3. Keep the best prompt seen under the deterministic tie-break (higher score,
//!    then the lexicographically smaller prompt), and stop at `early_stop_score`
//!    or after `iterations` rounds.
//!
//! `num_examples == 0` disables the example coordinate, leaving the instruction
//! rounds plus the example-free baseline.

use std::fmt;
use std::sync::Arc;

use openjiuwen_common::error::Result;
use openjiuwen_foundation::llm::Model;

use crate::tune::base::Case;
use crate::tune::evaluator::CaseEvaluator;
use crate::tune::optimizer::config::OptimizerConfig;
use crate::tune::optimizer::example::ExampleOptimizer;
use crate::tune::optimizer::instruction::InstructionOptimizer;
use crate::tune::optimizer::suggestion::{
    default_evaluator, describe_model, ensure_placeholders, is_better, validate_inputs,
};
use crate::tune::optimizer::OptimizationResult;

/// Optimizes instructions and few-shot examples together.
///
/// Python: `openjiuwen.dev_tools.tune.optimizer.JointOptimizer`.
pub struct JointOptimizer {
    /// The search budget, shared with both coordinates.
    config: OptimizerConfig,
    /// The model consulted by both coordinates, when one is attached.
    model: Option<Arc<dyn Model>>,
    /// The evaluator, or the deterministic default when none was injected.
    evaluator: Option<Arc<dyn CaseEvaluator>>,
}

impl Default for JointOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl JointOptimizer {
    /// Creates an optimizer with the default configuration, no model, and the
    /// deterministic evaluator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: OptimizerConfig::default(),
            model: None,
            evaluator: None,
        }
    }

    /// Replaces the search budget.
    #[must_use]
    pub fn with_config(mut self, config: OptimizerConfig) -> Self {
        self.config = config;
        self
    }

    /// Attaches a model that both coordinates may consult.
    #[must_use]
    pub fn with_model(mut self, model: Arc<dyn Model>) -> Self {
        self.model = Some(model);
        self
    }

    /// Replaces the evaluator.
    #[must_use]
    pub fn with_evaluator(mut self, evaluator: Arc<dyn CaseEvaluator>) -> Self {
        self.evaluator = Some(evaluator);
        self
    }

    /// The search budget.
    #[must_use]
    pub fn config(&self) -> &OptimizerConfig {
        &self.config
    }

    /// Whether a model is attached.
    #[must_use]
    pub fn has_model(&self) -> bool {
        self.model.is_some()
    }

    /// The evaluator in use for `prompt`.
    fn evaluator_for(&self, prompt: &str) -> Arc<dyn CaseEvaluator> {
        self.evaluator
            .clone()
            .unwrap_or_else(|| default_evaluator(prompt))
    }

    /// A one-round instruction optimizer sharing this optimizer's evaluator.
    fn instruction_for(
        &self,
        evaluator: &Arc<dyn CaseEvaluator>,
        rounds: usize,
    ) -> InstructionOptimizer {
        let config = self
            .config
            .with_iterations(rounds)
            .with_num_examples(self.config.num_examples);
        let optimizer = InstructionOptimizer::new()
            .with_config(config)
            .with_evaluator(Arc::clone(evaluator));
        match &self.model {
            Some(model) => optimizer.with_model(Arc::clone(model)),
            None => optimizer,
        }
    }

    /// An example optimizer sharing this optimizer's evaluator.
    fn examples_for(&self, evaluator: &Arc<dyn CaseEvaluator>) -> ExampleOptimizer {
        let optimizer = ExampleOptimizer::new()
            .with_config(self.config)
            .with_evaluator(Arc::clone(evaluator));
        match &self.model {
            Some(model) => optimizer.with_model(Arc::clone(model)),
            None => optimizer,
        }
    }

    /// Optimizes instructions and examples for `prompt` against `cases`.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::validation`] for a blank prompt, an
    /// empty case list, or an out-of-range configuration; otherwise the
    /// evaluator's own error.
    pub async fn optimize(&self, prompt: &str, cases: &[Case]) -> Result<OptimizationResult> {
        validate_inputs(prompt, cases)?;
        self.config.validate()?;

        let evaluator = self.evaluator_for(prompt);
        let seed = ensure_placeholders(prompt, prompt.trim());
        let (mut best_score, _) = evaluator.score_cases(&seed, cases).await?;
        let mut best_prompt = seed.clone();
        let mut iterations = 0;

        // The two single-coordinate baselines make the joint result dominant.
        let instruction_alone = self
            .instruction_for(&evaluator, self.config.iterations)
            .optimize(&seed, cases)
            .await?;
        consider(
            &mut best_prompt,
            &mut best_score,
            instruction_alone.best_prompt,
            instruction_alone.score,
        );
        if self.config.num_examples > 0 {
            let example_alone = self
                .examples_for(&evaluator)
                .optimize(&seed, cases, self.config.num_examples)
                .await?;
            consider(
                &mut best_prompt,
                &mut best_score,
                example_alone.best_prompt,
                example_alone.score,
            );
        }

        while iterations < self.config.iterations {
            if best_score + 1e-12 >= self.config.early_stop_score {
                break;
            }
            let instruction = self
                .instruction_for(&evaluator, 1)
                .optimize(&best_prompt, cases)
                .await?;
            iterations += 1;

            let instruction_prompt = instruction.best_prompt;
            consider(
                &mut best_prompt,
                &mut best_score,
                instruction_prompt.clone(),
                instruction.score,
            );

            if self.config.num_examples > 0 {
                let with_examples = self
                    .examples_for(&evaluator)
                    .optimize(&instruction_prompt, cases, self.config.num_examples)
                    .await?;
                consider(
                    &mut best_prompt,
                    &mut best_score,
                    with_examples.best_prompt,
                    with_examples.score,
                );
            }
        }

        Ok(OptimizationResult {
            best_prompt,
            score: best_score,
            iterations,
            ..OptimizationResult::default()
        })
    }
}

/// Keeps `(candidate, score)` when it beats the current best.
fn consider(best_prompt: &mut String, best_score: &mut f64, candidate: String, score: f64) {
    if is_better(score, &candidate, *best_score, best_prompt) {
        *best_score = score;
        *best_prompt = candidate;
    }
}

impl fmt::Debug for JointOptimizer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("JointOptimizer")
            .field("config", &self.config)
            .field("model", &describe_model(self.model.as_ref()))
            .field("custom_evaluator", &self.evaluator.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use openjiuwen_test_support::ScriptedModel;
    use serde_json::Value;

    use super::*;
    use crate::tune::base::EvaluatedCase;
    use crate::tune::evaluator::DeterministicEvaluator;

    fn case(id: &str, question: &str, answer: &str) -> Case {
        let mut inputs = BTreeMap::new();
        inputs.insert("question".to_owned(), Value::String(question.to_owned()));
        let mut label = BTreeMap::new();
        label.insert("answer".to_owned(), Value::String(answer.to_owned()));
        Case::new(inputs, label).unwrap().with_case_id(id)
    }

    fn cases() -> Vec<Case> {
        vec![
            case("c1", "capital of France", "Paris"),
            case("c2", "2 + 2", "four"),
            case("c3", "tallest mountain", "Everest"),
        ]
    }

    const SEED: &str = "Answer the question.";

    #[tokio::test]
    async fn is_at_least_as_good_as_either_coordinate_alone() {
        let cases = cases();
        let config = OptimizerConfig::default();
        let joint = JointOptimizer::new().optimize(SEED, &cases).await.unwrap();
        let instruction = InstructionOptimizer::new()
            .optimize(SEED, &cases)
            .await
            .unwrap();
        let example = ExampleOptimizer::new()
            .optimize(SEED, &cases, config.num_examples)
            .await
            .unwrap();
        assert!(
            joint.score + 1e-12 >= instruction.score,
            "joint {} >= instruction {}",
            joint.score,
            instruction.score
        );
        assert!(
            joint.score + 1e-12 >= example.score,
            "joint {} >= example {}",
            joint.score,
            example.score
        );
    }

    #[tokio::test]
    async fn improves_on_the_seed() {
        let cases = cases();
        let baseline = DeterministicEvaluator::new(SEED)
            .score_cases(SEED, &cases)
            .await
            .unwrap()
            .0;
        let result = JointOptimizer::new().optimize(SEED, &cases).await.unwrap();
        assert!(result.score > baseline, "{} > {baseline}", result.score);
        assert!(result.iterations <= OptimizerConfig::default().iterations);
    }

    #[tokio::test]
    async fn both_coordinates_contribute() {
        let cases = cases();
        let config = OptimizerConfig::default().with_num_examples(2);
        let result = JointOptimizer::new()
            .with_config(config)
            .optimize(SEED, &cases)
            .await
            .unwrap();
        assert!(
            result.best_prompt.contains("example "),
            "the winning prompt uses examples: {}",
            result.best_prompt
        );
        assert!(
            result.best_prompt.len() > SEED.len(),
            "the winning prompt was rewritten: {}",
            result.best_prompt
        );
    }

    #[tokio::test]
    async fn zero_examples_leaves_the_example_coordinate_out() {
        let cases = cases();
        let config = OptimizerConfig::default().with_num_examples(0);
        let result = JointOptimizer::new()
            .with_config(config)
            .optimize(SEED, &cases)
            .await
            .unwrap();
        assert!(
            !result.best_prompt.contains("example "),
            "{}",
            result.best_prompt
        );
        let instruction = InstructionOptimizer::new()
            .with_config(config)
            .optimize(SEED, &cases)
            .await
            .unwrap();
        assert!(result.score + 1e-12 >= instruction.score);
    }

    #[tokio::test]
    async fn repeat_runs_are_identical() {
        let optimizer = JointOptimizer::new();
        let first = optimizer.optimize(SEED, &cases()).await.unwrap();
        let second = optimizer.optimize(SEED, &cases()).await.unwrap();
        assert_eq!(first, second);
    }

    #[tokio::test]
    async fn early_stop_at_the_seed_runs_zero_rounds() {
        let result = JointOptimizer::new()
            .with_config(OptimizerConfig::default().with_early_stop_score(0.0))
            .optimize(SEED, &cases())
            .await
            .unwrap();
        assert_eq!(result.iterations, 0);
        let (seed_score, _) = DeterministicEvaluator::new(SEED)
            .score_cases(SEED, &cases())
            .await
            .unwrap();
        assert!((result.score - seed_score).abs() < 1e-9);
    }

    #[tokio::test]
    async fn honors_the_iteration_cap() {
        let result = JointOptimizer::new()
            .with_config(
                OptimizerConfig::default()
                    .with_iterations(2)
                    .with_early_stop_score(1.0),
            )
            .optimize(SEED, &cases())
            .await
            .unwrap();
        assert_eq!(result.iterations, 2);
    }

    #[tokio::test]
    async fn rejects_invalid_inputs() {
        let error = JointOptimizer::new()
            .optimize(" ", &cases())
            .await
            .unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );

        let error = JointOptimizer::new().optimize(SEED, &[]).await.unwrap_err();
        assert!(error.to_string().contains("cases"));

        let error = JointOptimizer::new()
            .with_config(OptimizerConfig::default().with_num_examples(MAX_EXAMPLE_NUM_LOCAL + 1))
            .optimize(SEED, &cases())
            .await
            .unwrap_err();
        assert!(error.to_string().contains("num_examples"));
    }

    #[tokio::test]
    async fn keeps_placeholders_across_both_coordinates() {
        let seed = "Answer the question about {{topic}}.";
        let result = JointOptimizer::new()
            .with_config(OptimizerConfig::default().with_num_examples(1))
            .optimize(seed, &cases())
            .await
            .unwrap();
        assert!(
            result.best_prompt.contains("{{topic}}"),
            "{}",
            result.best_prompt
        );
    }

    #[tokio::test]
    async fn model_rewrites_reach_both_coordinates() {
        let model = ScriptedModel::text(
            "<PROMPT_OPTIMIZED>Answer the question step by step and verify the result.</PROMPT_OPTIMIZED>",
        );
        let result = JointOptimizer::new()
            .with_model(Arc::new(model.clone()))
            .optimize(SEED, &cases())
            .await
            .unwrap();
        assert!(model.call_count() > 0, "the model was consulted");
        assert!(result.score > 0.0);
    }

    #[tokio::test]
    async fn an_injected_evaluator_is_shared_with_the_coordinates() {
        /// Rewards prompts that mention `Paris`.
        struct Paris;

        #[async_trait::async_trait]
        impl CaseEvaluator for Paris {
            async fn evaluate(&self, prompt: &str, case: &Case) -> Result<EvaluatedCase> {
                let score = if prompt.contains("Paris") { 1.0 } else { 0.0 };
                EvaluatedCase::new(case.clone(), BTreeMap::new(), score, "paris")
            }
        }

        let result = JointOptimizer::new()
            .with_evaluator(Arc::new(Paris))
            .optimize(SEED, &cases())
            .await
            .unwrap();
        // Both coordinates score 1.0 through the injected evaluator, so the seed's
        // example baseline already clears the early-stop threshold.
        assert!((result.score - 1.0).abs() < 1e-9);
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn debug_reports_the_model_choice() {
        assert!(format!("{:?}", JointOptimizer::new()).contains("model: \"none\""));
    }

    /// The example bound, re-read here so the test names the limit it exercises.
    const MAX_EXAMPLE_NUM_LOCAL: usize = crate::tune::optimizer::MAX_EXAMPLE_NUM;
}
