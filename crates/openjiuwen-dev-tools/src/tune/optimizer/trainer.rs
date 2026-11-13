//! [`Trainer`]: optimise on a train split, select on a validation split.
//!
//! Ports the role of `openjiuwen/dev_tools/tune/trainer/base.py`. Python's
//! trainer drives an agent through `Progress`/`Callbacks` hooks with no scoring
//! of its own; the port keeps the two-split discipline and makes the selection
//! explicit:
//!
//! 1. Score the seed on the validation split — the baseline.
//! 2. Stop immediately when the baseline already reaches `early_stop_score`
//!    (the default is `1.0`, so an already-perfect prompt is never touched).
//! 3. Per epoch: run one instruction round on the train split, score the
//!    candidate on the validation split, and keep it when it beats the best
//!    validation score (ties go to the lexicographically smaller prompt).
//! 4. Stop at `early_stop_score` or after `iterations` epochs.
//!
//! The reported `score` is measured on the validation split, never on the train
//! split, so a candidate that overfits the train cases cannot win. An empty
//! validation split falls back to the train split (documented deviation from
//! Python, which requires both); an empty train split is a validation error.
//! `train` is `async` here even though Python's is a synchronous generator loop,
//! because the underlying optimizers may call a model.

use std::fmt;
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::Model;

use crate::tune::base::{Case, EvaluatedCase};
use crate::tune::evaluator::CaseEvaluator;
use crate::tune::optimizer::config::OptimizerConfig;
use crate::tune::optimizer::instruction::InstructionOptimizer;
use crate::tune::optimizer::suggestion::{
    default_evaluator, describe_model, ensure_placeholders, is_better,
};
use crate::tune::optimizer::OptimizationResult;

/// Trains a prompt on one case split and reports the validation result.
///
/// Python: `openjiuwen.dev_tools.tune.trainer.Trainer`.
pub struct Trainer {
    /// The epoch budget and the instruction budget inside each epoch.
    config: OptimizerConfig,
    /// The model consulted by the instruction optimiser, when one is attached.
    model: Option<Arc<dyn Model>>,
    /// The evaluator, or the deterministic default when none was injected.
    evaluator: Option<Arc<dyn CaseEvaluator>>,
}

impl Default for Trainer {
    fn default() -> Self {
        Self::new()
    }
}

impl Trainer {
    /// Creates a trainer with the default configuration, no model, and the
    /// deterministic evaluator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: OptimizerConfig::default(),
            model: None,
            evaluator: None,
        }
    }

    /// Replaces the epoch budget.
    #[must_use]
    pub fn with_config(mut self, config: OptimizerConfig) -> Self {
        self.config = config;
        self
    }

    /// Attaches a model that proposes instruction rewrites.
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

    /// The epoch budget.
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

    /// Scores `prompt` over `cases`: the mean score and the per-case results.
    ///
    /// # Errors
    ///
    /// The evaluator's own error.
    pub async fn evaluate(
        &self,
        prompt: &str,
        cases: &[Case],
    ) -> Result<(f64, Vec<EvaluatedCase>)> {
        self.evaluator_for(prompt).score_cases(prompt, cases).await
    }

    /// Trains `prompt` on `train` and selects on `val`.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when the prompt is blank, the train split is empty,
    /// the validation split is empty *and* the train split is empty, or the
    /// configuration is out of range; otherwise the evaluator's own error.
    pub async fn train(
        &self,
        prompt: &str,
        train: &[Case],
        val: &[Case],
    ) -> Result<OptimizationResult> {
        if prompt.trim().is_empty() {
            return Err(Error::validation("prompt must not be blank"));
        }
        if train.is_empty() {
            return Err(Error::validation(
                "train split must not be empty: there is nothing to optimize on",
            ));
        }
        self.config.validate()?;

        let evaluator = self.evaluator_for(prompt);
        let seed = ensure_placeholders(prompt, prompt.trim());
        let val = if val.is_empty() { train } else { val };
        let (mut best_score, _) = evaluator.score_cases(&seed, val).await?;
        let mut best_prompt = seed;
        let mut iterations = 0;

        while iterations < self.config.iterations {
            if best_score + 1e-12 >= self.config.early_stop_score {
                break;
            }
            let instruction = self.instruction_for(&evaluator);
            let candidate = instruction.optimize(&best_prompt, train).await?;
            let (val_score, _) = evaluator.score_cases(&candidate.best_prompt, val).await?;
            iterations += 1;
            if is_better(val_score, &candidate.best_prompt, best_score, &best_prompt) {
                best_score = val_score;
                best_prompt = candidate.best_prompt;
            }
        }

        Ok(OptimizationResult {
            best_prompt,
            score: best_score,
            iterations,
        })
    }

    /// A one-round instruction optimizer sharing this trainer's evaluator.
    fn instruction_for(&self, evaluator: &Arc<dyn CaseEvaluator>) -> InstructionOptimizer {
        let optimizer = InstructionOptimizer::new()
            .with_config(self.config.with_iterations(1))
            .with_evaluator(Arc::clone(evaluator));
        match &self.model {
            Some(model) => optimizer.with_model(Arc::clone(model)),
            None => optimizer,
        }
    }
}

impl fmt::Debug for Trainer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Trainer")
            .field("config", &self.config)
            .field("model", &describe_model(self.model.as_ref()))
            .field("custom_evaluator", &self.evaluator.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use async_trait::async_trait;
    use openjiuwen_test_support::ScriptedModel;
    use serde_json::Value;

    use super::*;
    use crate::tune::evaluator::DeterministicEvaluator;

    fn case(id: &str, question: &str, answer: &str) -> Case {
        let mut inputs = BTreeMap::new();
        inputs.insert("question".to_owned(), Value::String(question.to_owned()));
        let mut label = BTreeMap::new();
        label.insert("answer".to_owned(), Value::String(answer.to_owned()));
        Case::new(inputs, label).unwrap().with_case_id(id)
    }

    fn train() -> Vec<Case> {
        vec![
            case("t1", "capital of France", "Paris"),
            case("t2", "2 + 2", "four"),
        ]
    }

    fn val() -> Vec<Case> {
        vec![case("v1", "capital of Italy", "Rome")]
    }

    const SEED: &str = "Answer the question.";

    #[tokio::test]
    async fn baseline_early_stop_skips_optimization() {
        let result = Trainer::new()
            .with_config(OptimizerConfig::default().with_early_stop_score(0.0))
            .train(SEED, &train(), &val())
            .await
            .unwrap();
        assert_eq!(result.iterations, 0);
        assert_eq!(result.best_prompt, SEED);

        let (baseline, _) = Trainer::new().evaluate(SEED, &val()).await.unwrap();
        assert!((result.score - baseline).abs() < 1e-9);
    }

    #[tokio::test]
    async fn improves_or_keeps_the_validation_score() {
        let (baseline, _) = Trainer::new().evaluate(SEED, &val()).await.unwrap();
        let result = Trainer::new().train(SEED, &train(), &val()).await.unwrap();
        assert!(
            result.score + 1e-12 >= baseline,
            "{} >= {baseline}",
            result.score
        );
        assert!(result.iterations >= 1);
    }

    #[tokio::test]
    async fn epochs_are_bounded_by_the_configuration() {
        let result = Trainer::new()
            .with_config(
                OptimizerConfig::default()
                    .with_iterations(2)
                    .with_early_stop_score(1.0),
            )
            .train(SEED, &train(), &val())
            .await
            .unwrap();
        assert_eq!(result.iterations, 2);
    }

    #[tokio::test]
    async fn an_empty_validation_split_falls_back_to_train() {
        let result = Trainer::new().train(SEED, &train(), &[]).await.unwrap();
        assert!(result.iterations >= 1);
        assert!(result.score > 0.0);
        // The score is now measured on the train split.
        let (train_score, _) = Trainer::new()
            .evaluate(&result.best_prompt, &train())
            .await
            .unwrap();
        assert!((result.score - train_score).abs() < 1e-9);
    }

    #[tokio::test]
    async fn an_empty_train_split_is_a_validation_error() {
        let error = Trainer::new().train(SEED, &[], &val()).await.unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        assert!(error.to_string().contains("train"));

        let error = Trainer::new()
            .train("  ", &train(), &val())
            .await
            .unwrap_err();
        assert!(error.to_string().contains("prompt"));
    }

    #[tokio::test]
    async fn rejects_an_invalid_configuration() {
        let error = Trainer::new()
            .with_config(OptimizerConfig::default().with_iterations(0))
            .train(SEED, &train(), &val())
            .await
            .unwrap_err();
        assert!(error.to_string().contains("iterations"));
    }

    #[tokio::test]
    async fn evaluate_reports_the_mean_and_every_case() {
        let (mean, cases) = Trainer::new().evaluate(SEED, &train()).await.unwrap();
        assert_eq!(cases.len(), 2);
        let recomputed: f64 = cases.iter().map(|case| case.score).sum::<f64>() / 2.0;
        assert!((mean - recomputed).abs() < 1e-9);

        let (empty_mean, empty) = Trainer::new().evaluate(SEED, &[]).await.unwrap();
        assert!(empty_mean.abs() < 1e-9);
        assert_eq!(empty.len(), 0);
    }

    #[tokio::test]
    async fn repeat_runs_are_identical() {
        let trainer = Trainer::new();
        let first = trainer.train(SEED, &train(), &val()).await.unwrap();
        let second = trainer.train(SEED, &train(), &val()).await.unwrap();
        assert_eq!(first, second);
    }

    #[tokio::test]
    async fn keeps_the_seed_placeholders() {
        let seed = "Answer the question about {{topic}}.";
        let result = Trainer::new()
            .with_config(OptimizerConfig::default().with_early_stop_score(1.0))
            .train(seed, &train(), &val())
            .await
            .unwrap();
        assert!(
            result.best_prompt.contains("{{topic}}"),
            "{}",
            result.best_prompt
        );
    }

    #[tokio::test]
    async fn model_rewrites_are_used_during_training() {
        let model = ScriptedModel::text(
            "<PROMPT_OPTIMIZED>Answer the question step by step and verify the result.</PROMPT_OPTIMIZED>",
        );
        let result = Trainer::new()
            .with_model(Arc::new(model.clone()))
            .train(SEED, &train(), &val())
            .await
            .unwrap();
        assert!(model.call_count() > 0);
        assert!(result.score > 0.0);
    }

    #[tokio::test]
    async fn an_injected_evaluator_drives_the_selection() {
        /// Rewards prompts mentioning `Rome`, the validation answer.
        struct Rome;

        #[async_trait]
        impl CaseEvaluator for Rome {
            async fn evaluate(&self, prompt: &str, case: &Case) -> Result<EvaluatedCase> {
                let score = if prompt.contains("Rome") { 1.0 } else { 0.0 };
                EvaluatedCase::new(case.clone(), BTreeMap::new(), score, "rome")
            }
        }

        let result = Trainer::new()
            .with_evaluator(Arc::new(Rome))
            .train(SEED, &train(), &val())
            .await
            .unwrap();
        // Every candidate scores 0.0 on validation, so the seed stays.
        assert_eq!(result.best_prompt, SEED);
        assert!(result.score.abs() < 1e-9);
        assert_eq!(result.iterations, 3);
    }

    #[test]
    fn debug_reports_the_model_choice() {
        assert!(format!("{:?}", Trainer::new()).contains("model: \"none\""));
    }

    #[tokio::test]
    async fn the_default_evaluator_matches_the_deterministic_one() {
        let (score, _) = Trainer::new().evaluate(SEED, &train()).await.unwrap();
        let expected = DeterministicEvaluator::new(SEED)
            .score_cases(SEED, &train())
            .await
            .unwrap()
            .0;
        assert!((score - expected).abs() < 1e-9);
    }
}
