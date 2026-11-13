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
//!
//! [`Trainer::evaluate_traced`] is the observable form of
//! [`Trainer::evaluate`]: it returns the same mean score and per-case results
//! plus an [`EvaluationSummary`] carrying the tool calls behind them, so a
//! report can show which tools a candidate prompt actually used.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::Model;
use serde::{Deserialize, Serialize};

use crate::tune::base::{Case, EvaluatedCase};
use crate::tune::evaluator::{tool_usage, CaseEvaluator};
use crate::tune::optimizer::config::OptimizerConfig;
use crate::tune::optimizer::instruction::InstructionOptimizer;
use crate::tune::optimizer::suggestion::{
    default_evaluator, describe_model, ensure_placeholders, is_better,
};
use crate::tune::optimizer::OptimizationResult;

/// The traced outcome of scoring one prompt over a case set.
///
/// The score and results are exactly what [`Trainer::evaluate`] reports; the
/// extra [`tool_usage`](Self::tool_usage) map says which tools the evaluated
/// answers needed, summed over every case in the run.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct EvaluationSummary {
    /// The mean score over `results` (`0.0` for no cases).
    pub mean_score: f64,
    /// The per-case results, in case order.
    pub results: Vec<EvaluatedCase>,
    /// Tool name -> number of calls made while producing those results.
    pub tool_usage: BTreeMap<String, usize>,
}

impl EvaluationSummary {
    /// Whether no case was evaluated.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// The tool names that were called, sorted.
    #[must_use]
    pub fn tool_names(&self) -> Vec<&str> {
        self.tool_usage.keys().map(String::as_str).collect()
    }
}

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

    /// Scores `prompt` over `cases` and reports the tool calls behind each
    /// result.
    ///
    /// The mean score and the per-case results are the same as
    /// [`evaluate`](Self::evaluate)'s; the returned [`EvaluationSummary`] adds
    /// the per-tool call counts of the whole run. The traces themselves are not
    /// exposed, because an [`EvaluatedCase`] keeps its wire shape. An empty
    /// `cases` slice yields a `0.0` mean, no results, and no usage.
    ///
    /// # Errors
    ///
    /// The evaluator's own error.
    pub async fn evaluate_traced(&self, prompt: &str, cases: &[Case]) -> Result<EvaluationSummary> {
        let (mean_score, traced) = self
            .evaluator_for(prompt)
            .score_cases_traced(prompt, cases)
            .await?;
        let tool_usage = tool_usage(&traced);
        Ok(EvaluationSummary {
            mean_score,
            results: traced.into_iter().map(|(evaluated, _)| evaluated).collect(),
            tool_usage,
        })
    }

    /// Trains `prompt` on `train` and selects on `val`.
    ///
    /// Each train case keeps its declared tools, which the instruction
    /// optimizer renders into its rewrite prompt.
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
    use openjiuwen_foundation::tool::ToolInfo;
    use openjiuwen_test_support::{MockCompletion, ScriptedModel};
    use serde_json::{json, Value};

    use super::*;
    use crate::tune::evaluator::{DeterministicEvaluator, ModelPromptRunner, RunnerEvaluator};
    use crate::tune::tool::{ToolCallRecord, ToolStub, ToolStubs, ToolTrace};

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

    /// A tool-aware trainer: the model calls `lookup`, then answers `7`.
    fn tool_trainer() -> Trainer {
        let model = ScriptedModel::builder()
            .model_name("test-model")
            .completion(MockCompletion::tool_call("lookup", json!({"q": "7"})))
            .respond_text("7")
            .build();
        let stubs = ToolStubs::new().with_stub(ToolStub::constant("lookup", json!("7")).unwrap());
        let runner = Arc::new(ModelPromptRunner::new(Arc::new(model)).with_stubs(stubs));
        Trainer::new().with_evaluator(Arc::new(RunnerEvaluator::new(runner)))
    }

    /// The one case declaring the tool [`tool_trainer`] calls.
    fn tool_case() -> Case {
        Case::from_text("what is 7?", "7")
            .unwrap()
            .with_tools(vec![ToolInfo::new(
                "lookup",
                "Look something up",
                json!({"type": "object"}),
            )])
    }

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

    #[tokio::test]
    async fn evaluate_traced_reports_tool_usage() {
        let summary = tool_trainer()
            .evaluate_traced("Answer with the tool.", &[tool_case()])
            .await
            .unwrap();
        assert!(summary.mean_score > 0.0);
        assert_eq!(summary.results.len(), 1);
        assert_eq!(
            summary.tool_usage,
            BTreeMap::from([("lookup".to_owned(), 1)])
        );
        assert_eq!(summary.tool_names(), vec!["lookup"]);
        assert!(!summary.is_empty());
    }

    #[tokio::test]
    async fn evaluate_traced_matches_evaluate() {
        let cases = [tool_case()];
        let (mean, results) = tool_trainer()
            .evaluate("Answer with the tool.", &cases)
            .await
            .unwrap();
        let summary = tool_trainer()
            .evaluate_traced("Answer with the tool.", &cases)
            .await
            .unwrap();
        assert!((summary.mean_score - mean).abs() < 1e-12);
        assert_eq!(summary.results, results);
    }

    #[tokio::test]
    async fn evaluate_traced_on_no_cases_is_empty() {
        let summary = tool_trainer()
            .evaluate_traced("Answer with the tool.", &[])
            .await
            .unwrap();
        assert!(summary.is_empty());
        assert!(summary.mean_score.abs() < 1e-12);
        assert!(summary.tool_usage.is_empty());
        assert_eq!(summary.tool_names(), Vec::<&str>::new());
    }

    #[tokio::test]
    async fn evaluate_traced_without_tools_reports_no_usage() {
        let summary = Trainer::new()
            .evaluate_traced(SEED, &train())
            .await
            .unwrap();
        assert!(summary.tool_usage.is_empty());
        assert_eq!(summary.tool_names(), Vec::<&str>::new());
        assert_eq!(summary.results.len(), 2);
        assert!(summary.mean_score > 0.0);
    }

    #[tokio::test]
    async fn evaluate_traced_honors_an_injected_evaluator() {
        /// Scores every case `0.25` and reports one `search` call.
        struct Searcher;

        #[async_trait]
        impl CaseEvaluator for Searcher {
            async fn evaluate(&self, _prompt: &str, case: &Case) -> Result<EvaluatedCase> {
                EvaluatedCase::new(case.clone(), BTreeMap::new(), 0.25, "searched")
            }

            async fn evaluate_traced(
                &self,
                prompt: &str,
                case: &Case,
            ) -> Result<(EvaluatedCase, ToolTrace)> {
                let evaluated = self.evaluate(prompt, case).await?;
                let trace =
                    ToolTrace::new([ToolCallRecord::success("search", json!({}), json!("ok"))]);
                Ok((evaluated, trace))
            }
        }

        let summary = Trainer::new()
            .with_evaluator(Arc::new(Searcher))
            .evaluate_traced(SEED, &[case("s1", "capital of Spain", "Madrid")])
            .await
            .unwrap();
        assert!((summary.mean_score - 0.25).abs() < 1e-12);
        assert_eq!(summary.tool_usage.get("search"), Some(&1));
        assert_eq!(summary.tool_names(), vec!["search"]);
    }

    #[test]
    fn evaluation_summary_round_trips_through_json() {
        let evaluated =
            EvaluatedCase::new(case("c1", "q", "a"), BTreeMap::new(), 0.5, "half").unwrap();
        let summary = EvaluationSummary {
            mean_score: 0.5,
            results: vec![evaluated],
            tool_usage: BTreeMap::from([("search".to_owned(), 2)]),
        };

        let value = serde_json::to_value(&summary).unwrap();
        let object = value.as_object().unwrap();
        for key in ["mean_score", "results", "tool_usage"] {
            assert!(object.contains_key(key), "missing field {key}");
        }
        let restored: EvaluationSummary = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(restored, summary);

        let mut unknown = value;
        unknown["extra"] = json!(true);
        let restored: EvaluationSummary = serde_json::from_value(unknown).unwrap();
        assert_eq!(restored, summary);

        let partial: EvaluationSummary =
            serde_json::from_str(r#"{"mean_score": 0.0, "results": []}"#).unwrap();
        assert!(partial.is_empty());
        assert!(partial.tool_usage.is_empty());
        let defaulted: EvaluationSummary = serde_json::from_str("{}").unwrap();
        assert_eq!(defaulted, EvaluationSummary::default());
    }
}
