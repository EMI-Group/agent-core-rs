//! [`InstructionOptimizer`]: beam search over instruction rewrites.
//!
//! Python's `InstructionOptimizer` wraps an `LLMCall` parameter, asks a model
//! for a textual gradient over the failing cases, then asks again for a rewritten
//! prompt — one model call per phase per round, no scoring of the result.
//!
//! The port keeps the goal (a better instruction prompt, placeholders intact,
//! failing cases repaired) and makes the search measurable:
//!
//! 1. Score the seed with the evaluator (default:
//!    [`DeterministicEvaluator`](crate::tune::evaluator::DeterministicEvaluator),
//!    seeded with the input prompt).
//! 2. Per round: expand every beam member with the deterministic mutation bank
//!    and, when a model is attached, the best member with one model rewrite;
//!    repair missing `{{placeholder}}`s; drop candidates already seen.
//! 3. Score every new candidate over all cases and keep the best `beam_width`.
//! 4. Stop at `early_stop_score` (checked before a round, so an already-good
//!    seed runs zero rounds) or after `iterations` rounds.
//!
//! Ties are resolved deterministically — higher score, then the earlier
//! candidate, then the lexicographically smaller prompt — so a repeated run
//! returns the same prompt, score, and iteration count.

use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use openjiuwen_common::error::Result;
use openjiuwen_foundation::llm::Model;

use crate::tune::base::Case;
use crate::tune::evaluator::CaseEvaluator;
use crate::tune::optimizer::config::OptimizerConfig;
use crate::tune::optimizer::suggestion::{
    default_evaluator, describe_model, ensure_placeholders, is_better, normalized_candidates,
    score_candidates, select_best_index, validate_inputs, worst_cases, Mutator,
};
use crate::tune::optimizer::OptimizationResult;

/// Rewrites the instructions of a prompt against a case set.
///
/// Python: `openjiuwen.dev_tools.tune.optimizer.InstructionOptimizer`.
pub struct InstructionOptimizer {
    /// The search budget.
    config: OptimizerConfig,
    /// The model consulted for rewrites, when one is attached.
    model: Option<Arc<dyn Model>>,
    /// The evaluator, or the deterministic default when none was injected.
    evaluator: Option<Arc<dyn CaseEvaluator>>,
}

impl Default for InstructionOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl InstructionOptimizer {
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

    /// Attaches a model that proposes additional rewrites.
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

    /// The evaluator in use for `prompt`: the injected one or the deterministic
    /// default seeded with `prompt`.
    fn evaluator_for(&self, prompt: &str) -> Arc<dyn CaseEvaluator> {
        self.evaluator
            .clone()
            .unwrap_or_else(|| default_evaluator(prompt))
    }

    /// Rewrites `prompt` against `cases` and returns the best candidate.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::validation`] when the prompt is blank,
    /// the case list is empty, or the configuration is out of range; otherwise
    /// the evaluator's own error.
    pub async fn optimize(&self, prompt: &str, cases: &[Case]) -> Result<OptimizationResult> {
        validate_inputs(prompt, cases)?;
        self.config.validate()?;

        let evaluator = self.evaluator_for(prompt);
        let mutator = Mutator::new(self.model.clone());
        let seed = ensure_placeholders(prompt, prompt.trim());
        let (mut best_score, mut best_cases) = evaluator.score_cases(&seed, cases).await?;
        let mut best_prompt = seed.clone();
        let mut beam = vec![seed.clone()];
        let mut visited: BTreeSet<String> = BTreeSet::from([seed]);
        let mut iterations = 0;

        while iterations < self.config.iterations {
            if best_score + 1e-12 >= self.config.early_stop_score {
                break;
            }

            let mut proposals: Vec<String> = Vec::new();
            for member in &beam {
                proposals.extend(Mutator::deterministic(member));
            }
            if let Some(rewrite) = mutator
                .model_rewrite(&best_prompt, &worst_cases(&best_cases, 5))
                .await
            {
                proposals.push(rewrite);
            }

            let candidates: Vec<String> = normalized_candidates(prompt, proposals)
                .into_iter()
                .filter(|candidate| !visited.contains(candidate))
                .collect();
            if candidates.is_empty() {
                break;
            }
            for candidate in &candidates {
                visited.insert(candidate.clone());
            }

            let scores =
                score_candidates(&*evaluator, &candidates, cases, self.config.num_parallel).await?;
            iterations += 1;

            if let Some(index) = select_best_index(&scores, &candidates) {
                let (score, case_results) =
                    evaluator.score_cases(&candidates[index], cases).await?;
                if is_better(score, &candidates[index], best_score, &best_prompt) {
                    best_score = score;
                    best_prompt = candidates[index].clone();
                    best_cases = case_results;
                }
            }

            beam = ranked_beam(&candidates, &scores, self.config.beam_width);
        }

        Ok(OptimizationResult {
            best_prompt,
            score: best_score,
            iterations,
        })
    }
}

impl fmt::Debug for InstructionOptimizer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("InstructionOptimizer")
            .field("config", &self.config)
            .field("model", &describe_model(self.model.as_ref()))
            .field("custom_evaluator", &self.evaluator.is_some())
            .finish()
    }
}

/// The best `width` candidates ordered by score, then prompt, then position.
fn ranked_beam(candidates: &[String], scores: &[f64], width: usize) -> Vec<String> {
    let mut order: Vec<usize> = (0..candidates.len().min(scores.len())).collect();
    order.sort_by(|left, right| {
        scores[*right]
            .partial_cmp(&scores[*left])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| candidates[*left].cmp(&candidates[*right]))
            .then_with(|| left.cmp(right))
    });
    order
        .into_iter()
        .take(width.max(1))
        .map(|index| candidates[index].clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use async_trait::async_trait;
    use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};
    use serde_json::Value;

    use super::*;
    use crate::tune::base::EvaluatedCase;
    use crate::tune::evaluator::DeterministicEvaluator;

    /// A case whose answer is the input, so coverage depends on the text.
    fn case(id: &str, question: &str, answer: &str) -> Case {
        let mut inputs = BTreeMap::new();
        inputs.insert("question".to_owned(), Value::String(question.to_owned()));
        let mut label = BTreeMap::new();
        label.insert("answer".to_owned(), Value::String(answer.to_owned()));
        Case::new(inputs, label).unwrap().with_case_id(id)
    }

    /// Two cases with short labels so the seed covers few terms.
    fn cases() -> Vec<Case> {
        vec![
            case("c1", "capital of France", "Paris"),
            case("c2", "2 + 2", "four"),
        ]
    }

    /// The seed scores low on structure so the mutation bank can help.
    const SEED: &str = "Answer the question.";

    #[tokio::test]
    async fn improves_the_seed_with_the_mutation_bank() {
        let cases = cases();
        let baseline = DeterministicEvaluator::new(SEED)
            .score_cases(SEED, &cases)
            .await
            .unwrap()
            .0;
        let result = InstructionOptimizer::new()
            .optimize(SEED, &cases)
            .await
            .unwrap();
        assert!(result.score > baseline, "{} > {baseline}", result.score);
        assert_ne!(result.best_prompt, SEED);
        assert!(
            result.best_prompt.starts_with(SEED),
            "rewrites extend the seed: {}",
            result.best_prompt
        );
        assert_eq!(result.iterations, OptimizerConfig::default().iterations);
        assert!((0.0..=1.0).contains(&result.score));
    }

    #[tokio::test]
    async fn repeat_runs_are_identical() {
        let cases = cases();
        let optimizer =
            InstructionOptimizer::new().with_config(OptimizerConfig::default().with_beam_width(2));
        let first = optimizer.optimize(SEED, &cases).await.unwrap();
        let second = optimizer.optimize(SEED, &cases).await.unwrap();
        assert_eq!(first, second);
    }

    #[tokio::test]
    async fn rejects_blank_prompts_and_empty_cases() {
        let error = InstructionOptimizer::new()
            .optimize("   ", &cases())
            .await
            .unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        assert!(error.to_string().contains("prompt"));

        let error = InstructionOptimizer::new()
            .optimize(SEED, &[])
            .await
            .unwrap_err();
        assert!(error.to_string().contains("cases"));
    }

    #[tokio::test]
    async fn rejects_an_invalid_configuration() {
        let optimizer = InstructionOptimizer::new()
            .with_config(OptimizerConfig::default().with_early_stop_score(2.0));
        let error = optimizer.optimize(SEED, &cases()).await.unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        assert!(error.to_string().contains("early_stop_score"));
    }

    #[tokio::test]
    async fn early_stop_at_the_seed_runs_zero_rounds() {
        let optimizer = InstructionOptimizer::new()
            .with_config(OptimizerConfig::default().with_early_stop_score(0.0));
        let result = optimizer.optimize(SEED, &cases()).await.unwrap();
        assert_eq!(result.iterations, 0);
        assert_eq!(result.best_prompt, SEED);
        assert!(result.score > 0.0);
    }

    #[tokio::test]
    async fn early_stop_after_reaching_the_threshold_runs_one_round() {
        /// Scores `0.9` once the step-by-step mutation is present, else `0.1`.
        struct Cue;

        #[async_trait]
        impl CaseEvaluator for Cue {
            async fn evaluate(&self, prompt: &str, case: &Case) -> Result<EvaluatedCase> {
                let score = if prompt.contains("Think step by step") {
                    0.9
                } else {
                    0.1
                };
                EvaluatedCase::new(case.clone(), BTreeMap::new(), score, "cue")
            }
        }

        // The first round finds the cue and clears the threshold; later rounds stop.
        let optimizer = InstructionOptimizer::new()
            .with_config(
                OptimizerConfig::default()
                    .with_iterations(5)
                    .with_early_stop_score(0.5),
            )
            .with_evaluator(Arc::new(Cue));
        let result = optimizer.optimize(SEED, &cases()).await.unwrap();
        assert_eq!(result.iterations, 1);
        assert!((result.score - 0.9).abs() < 1e-9);
    }

    #[tokio::test]
    async fn honors_the_iteration_and_beam_caps() {
        let cases = cases();
        let result = InstructionOptimizer::new()
            .with_config(
                OptimizerConfig::default()
                    .with_iterations(2)
                    .with_beam_width(1)
                    .with_early_stop_score(1.0),
            )
            .optimize(SEED, &cases)
            .await
            .unwrap();
        assert_eq!(result.iterations, 2);
        let wider = InstructionOptimizer::new()
            .with_config(
                OptimizerConfig::default()
                    .with_iterations(2)
                    .with_beam_width(3)
                    .with_early_stop_score(1.0),
            )
            .optimize(SEED, &cases)
            .await
            .unwrap();
        assert!(
            wider.score + 1e-12 >= result.score,
            "a wider beam never loses: {} >= {}",
            wider.score,
            result.score
        );
    }

    #[tokio::test]
    async fn keeps_the_seed_placeholders() {
        let seed = "Answer the question about {{topic}}.";
        let result = InstructionOptimizer::new()
            .with_config(OptimizerConfig::default().with_early_stop_score(1.0))
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
    async fn uses_a_model_rewrite_and_repairs_its_placeholders() {
        let model = ScriptedModel::text(
            "<PROMPT_OPTIMIZED>Answer about {{topic}} step by step in the requested output \
             format, and never invent facts.</PROMPT_OPTIMIZED>",
        );
        let optimizer = InstructionOptimizer::new().with_model(Arc::new(model.clone()));
        let result = optimizer
            .optimize("Answer about {{topic}}.", &cases())
            .await
            .unwrap();
        assert!(model.call_count() >= 1, "the model was consulted");
        assert!(
            result.best_prompt.contains("{{topic}}"),
            "{}",
            result.best_prompt
        );
        let call = model.last_call().expect("recorded call");
        assert!(call.messages[0].text().contains("PROMPT_OPTIMIZED"));
        let user = call.messages[1].text();
        assert!(user.contains("<prompt_base>"), "{user}");
        assert!(user.contains("{{topic}}"), "{user}");
    }

    #[tokio::test]
    async fn falls_back_to_the_bank_when_the_model_answer_is_unusable() {
        for turn in [
            ScriptedTurn::text("no tags at all"),
            ScriptedTurn::fail("provider down"),
            ScriptedTurn::text("<PROMPT_OPTIMIZED>   </PROMPT_OPTIMIZED>"),
        ] {
            let model = ScriptedModel::new([turn]);
            let optimizer = InstructionOptimizer::new().with_model(Arc::new(model.clone()));
            let result = optimizer.optimize(SEED, &cases()).await.unwrap();
            assert!(result.score > 0.0);
            assert_eq!(
                model.call_count(),
                result.iterations,
                "one model rewrite is attempted per round"
            );
        }
    }

    #[tokio::test]
    async fn injected_evaluator_drives_the_search() {
        /// Scores a fixed `1.0`, so the seed is already optimal.
        struct Perfect;

        #[async_trait]
        impl CaseEvaluator for Perfect {
            async fn evaluate(&self, _prompt: &str, case: &Case) -> Result<EvaluatedCase> {
                EvaluatedCase::new(case.clone(), BTreeMap::new(), 1.0, "perfect")
            }
        }

        let result = InstructionOptimizer::new()
            .with_evaluator(Arc::new(Perfect))
            .optimize(SEED, &cases())
            .await
            .unwrap();
        assert_eq!(result.iterations, 0);
        assert_eq!(result.best_prompt, SEED);
        assert!((result.score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn ranked_beam_orders_by_score_then_prompt() {
        let candidates = vec!["b".to_owned(), "a".to_owned(), "c".to_owned()];
        assert_eq!(
            ranked_beam(&candidates, &[0.1, 0.1, 0.9], 2),
            vec!["c".to_owned(), "a".to_owned()]
        );
        assert_eq!(
            ranked_beam(&candidates, &[0.5, 0.5, 0.5], 1),
            vec!["a".to_owned()]
        );
    }

    #[test]
    fn debug_reports_the_model_and_evaluator_choices() {
        let optimizer = InstructionOptimizer::new();
        assert!(format!("{optimizer:?}").contains("model: \"none\""));

        let optimizer = optimizer.with_model(Arc::new(ScriptedModel::text("x")));
        let debug = format!("{optimizer:?}");
        assert!(debug.contains("scripted"), "{debug}");
        assert!(debug.contains("custom_evaluator: false"), "{debug}");
    }
}
