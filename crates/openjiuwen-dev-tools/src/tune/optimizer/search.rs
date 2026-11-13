//! [`PromptSearchOptimizer`]: one breadth-first sweep over every candidate.
//!
//! Python's `optimizer/prompt_search/` package runs a stateful service (policy,
//! memory, reward, drift, convergence) around an LLM that proposes prompts. The
//! port keeps the observable contract — start from a prompt, evaluate a pool of
//! candidates, return the best — as a stateless, single-depth search:
//!
//! 1. Build the pool from the seed, every deterministic mutation, and one model
//!    rewrite when a model is attached.
//! 2. Repair missing `{{placeholder}}`s and drop duplicates while preserving
//!    order (the seed always stays first).
//! 3. Score every candidate and return the best under the deterministic
//!    tie-break (higher score, then the lexicographically smaller prompt).
//!
//! Because the seed is in the pool, the result is never worse than the input.
//! `iterations` is `1`: the search is a single sweep, not a multi-round beam.

use std::fmt;
use std::sync::Arc;

use openjiuwen_common::error::Result;
use openjiuwen_foundation::llm::Model;

use crate::tune::base::Case;
use crate::tune::evaluator::CaseEvaluator;
use crate::tune::optimizer::config::OptimizerConfig;
use crate::tune::optimizer::suggestion::{
    default_evaluator, describe_model, ensure_placeholders, normalized_candidates,
    score_candidates, select_best_index, validate_inputs, worst_cases, Mutator,
};
use crate::tune::optimizer::OptimizationResult;

/// Searches a single pool of prompt candidates and returns the best one.
///
/// Python: `openjiuwen.dev_tools.tune.optimizer.prompt_search.PromptSearchOptimizer`.
pub struct PromptSearchOptimizer {
    /// The search budget.
    config: OptimizerConfig,
    /// The model consulted for one extra candidate, when one is attached.
    model: Option<Arc<dyn Model>>,
    /// The evaluator, or the deterministic default when none was injected.
    evaluator: Option<Arc<dyn CaseEvaluator>>,
}

impl Default for PromptSearchOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl PromptSearchOptimizer {
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

    /// Attaches a model that proposes one extra candidate.
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

    /// Searches for a better prompt than `prompt` for `cases`.
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
        let mutator = Mutator::new(self.model.clone());
        let seed = ensure_placeholders(prompt, prompt.trim());

        let (_, seed_cases) = evaluator.score_cases(&seed, cases).await?;
        // The seed leads the pool, so the result is never worse than the input.
        let mut proposals = vec![seed.clone()];
        proposals.extend(Mutator::deterministic(&seed));
        if let Some(rewrite) = mutator
            .model_rewrite(&seed, &worst_cases(&seed_cases, 5))
            .await
        {
            proposals.push(rewrite);
        }
        let pool = normalized_candidates(&seed, proposals);

        let scores = score_candidates(&*evaluator, &pool, cases, self.config.num_parallel).await?;
        let best = select_best_index(&scores, &pool).unwrap_or(0);

        Ok(OptimizationResult {
            best_prompt: pool.get(best).cloned().unwrap_or_else(|| seed.clone()),
            score: scores.get(best).copied().unwrap_or(0.0),
            iterations: 1,
        })
    }
}

impl fmt::Debug for PromptSearchOptimizer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PromptSearchOptimizer")
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
        ]
    }

    const SEED: &str = "Answer the question.";

    #[tokio::test]
    async fn returns_the_best_candidate_in_one_sweep() {
        let cases = cases();
        let baseline = DeterministicEvaluator::new(SEED)
            .score_cases(SEED, &cases)
            .await
            .unwrap()
            .0;
        let result = PromptSearchOptimizer::new()
            .optimize(SEED, &cases)
            .await
            .unwrap();
        assert_eq!(result.iterations, 1);
        assert!(result.score > baseline, "{} > {baseline}", result.score);
        assert_ne!(result.best_prompt, SEED);
    }

    #[tokio::test]
    async fn the_pool_is_deduplicated() {
        // Every mutation produces a distinct prompt, so the best cannot be a
        // duplicate; a repeated run yields the same winner.
        let optimizer = PromptSearchOptimizer::new();
        let first = optimizer.optimize(SEED, &cases()).await.unwrap();
        let second = optimizer.optimize(SEED, &cases()).await.unwrap();
        assert_eq!(first, second);
    }

    #[tokio::test]
    async fn keeps_placeholders_and_the_seed_contract() {
        let seed = "Answer about {{topic}}.";
        let result = PromptSearchOptimizer::new()
            .with_evaluator(Arc::new(DeterministicEvaluator::new(seed)))
            .optimize(seed, &cases())
            .await
            .unwrap();
        assert!(
            result.best_prompt.contains("{{topic}}"),
            "{}",
            result.best_prompt
        );
        assert!(result.best_prompt.starts_with("Answer about {{topic}}."));
    }

    #[tokio::test]
    async fn rejects_invalid_inputs() {
        let error = PromptSearchOptimizer::new()
            .optimize("  ", &cases())
            .await
            .unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );

        let error = PromptSearchOptimizer::new()
            .optimize(SEED, &[])
            .await
            .unwrap_err();
        assert!(error.to_string().contains("cases"));

        let error = PromptSearchOptimizer::new()
            .with_config(OptimizerConfig::default().with_beam_width(0))
            .optimize(SEED, &cases())
            .await
            .unwrap_err();
        assert!(error.to_string().contains("beam_width"));
    }

    #[tokio::test]
    async fn a_model_rewrite_joins_the_pool() {
        /// Rewards the marker only the model answer carries.
        struct Marker;

        #[async_trait]
        impl CaseEvaluator for Marker {
            async fn evaluate(&self, prompt: &str, case: &Case) -> Result<EvaluatedCase> {
                let score = if prompt.contains("MAGIC") { 1.0 } else { 0.0 };
                EvaluatedCase::new(case.clone(), BTreeMap::new(), score, "marker")
            }
        }

        let model = ScriptedModel::text("<PROMPT_OPTIMIZED>MAGIC answer.</PROMPT_OPTIMIZED>");
        let result = PromptSearchOptimizer::new()
            .with_model(Arc::new(model.clone()))
            .with_evaluator(Arc::new(Marker))
            .optimize(SEED, &cases())
            .await
            .unwrap();
        assert_eq!(model.call_count(), 1);
        assert_eq!(result.best_prompt, "MAGIC answer.");
        assert!((result.score - 1.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn an_unusable_model_answer_keeps_the_bank() {
        let model = ScriptedModel::text("plain text, no tags");
        let result = PromptSearchOptimizer::new()
            .with_model(Arc::new(model.clone()))
            .optimize(SEED, &cases())
            .await
            .unwrap();
        assert_eq!(model.call_count(), 1);
        assert!(result.best_prompt.starts_with(SEED));
        assert_ne!(result.best_prompt, SEED);
    }

    #[tokio::test]
    async fn the_seed_wins_when_nothing_improves() {
        /// Scores the seed highest, so no rewrite may win.
        struct SeedOnly;

        #[async_trait]
        impl CaseEvaluator for SeedOnly {
            async fn evaluate(&self, prompt: &str, case: &Case) -> Result<EvaluatedCase> {
                let score = if prompt == "Answer the question." {
                    1.0
                } else {
                    0.2
                };
                EvaluatedCase::new(case.clone(), BTreeMap::new(), score, "seed")
            }
        }

        let result = PromptSearchOptimizer::new()
            .with_evaluator(Arc::new(SeedOnly))
            .optimize(SEED, &cases())
            .await
            .unwrap();
        assert_eq!(result.best_prompt, SEED);
        assert!((result.score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn debug_reports_the_model_choice() {
        assert!(format!("{:?}", PromptSearchOptimizer::new()).contains("model: \"none\""));
    }
}
