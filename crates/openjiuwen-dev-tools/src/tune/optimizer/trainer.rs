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
//! split, so a candidate that overfits the train cases cannot win, and the
//! optimizer is only ever handed the train cases, so a held-out case is never
//! fitted against. [`Trainer::train`] takes the two bare case lists (an empty
//! validation split falls back to the train split, a documented deviation from
//! Python, which requires both); [`Trainer::train_split`] takes
//! [`SplitDataset`]s instead, so the partitions carry their
//! [`CaseSplit`] tag and the result can report both scores and the split the
//! winner was chosen on.
//! `train` is `async` here even though Python's is a synchronous generator loop,
//! because the underlying optimizers may call a model.
//!
//! [`Trainer::evaluate_traced`] is the observable form of
//! [`Trainer::evaluate`]: it returns the same mean score and per-case results
//! plus an [`EvaluationSummary`] carrying the tool calls behind them and the
//! [`RunTelemetry`] summed over the evaluated cases, so a report can show which
//! tools a candidate prompt actually used and what the run cost.
//! [`Trainer::evaluate_split`] is the same evaluation over a tagged
//! [`SplitDataset`], so the summary says which partition the scores came from.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::Model;
use serde::{Deserialize, Serialize};

use crate::tune::base::{Case, EvaluatedCase};
use crate::tune::dataset::{CaseSplit, SplitDataset};
use crate::tune::evaluator::{tool_usage, CaseEvaluator};
use crate::tune::optimizer::config::OptimizerConfig;
use crate::tune::optimizer::instruction::InstructionOptimizer;
use crate::tune::optimizer::suggestion::{
    default_evaluator, describe_model, ensure_placeholders, is_better,
};
use crate::tune::optimizer::OptimizationResult;
use crate::tune::tool::{RunTelemetry, ToolTrace};

/// The traced outcome of scoring one prompt over a case set.
///
/// The score and results are exactly what [`Trainer::evaluate`] reports; the
/// extra [`tool_usage`](Self::tool_usage) map says which tools the evaluated
/// answers needed, summed over every case in the run,
/// [`telemetry`](Self::telemetry) says what the run cost, and
/// [`split`](Self::split) says which partition the cases came from.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct EvaluationSummary {
    /// The mean score over `results` (`0.0` for no cases).
    pub mean_score: f64,
    /// The per-case results, in case order.
    pub results: Vec<EvaluatedCase>,
    /// Tool name -> number of calls made while producing those results.
    pub tool_usage: BTreeMap<String, usize>,
    /// What the evaluated cases cost, summed over the run.
    ///
    /// Absent fields mean *not reported* rather than zero, so an evaluator that
    /// measures nothing leaves this at [`RunTelemetry::default`].
    pub telemetry: RunTelemetry,
    /// The partition the scores were measured on.
    ///
    /// [`CaseSplit::Unspecified`] — the serde default, so a summary written
    /// before the tag existed still loads — for an untagged evaluation;
    /// [`Trainer::evaluate_split`] keeps the tag of the dataset it scored.
    pub split: CaseSplit,
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

    /// Scores `prompt` over `cases` and reports the tool calls and telemetry
    /// behind each result.
    ///
    /// The mean score and the per-case results are the same as
    /// [`evaluate`](Self::evaluate)'s; the returned [`EvaluationSummary`] adds
    /// the per-tool call counts of the whole run and the [`RunTelemetry`] of
    /// every evaluated case, summed in case order. The traces themselves are not
    /// exposed, because an [`EvaluatedCase`] keeps its wire shape. An empty
    /// `cases` slice yields a `0.0` mean, no results, no usage, and absent
    /// telemetry. The summary stays [`CaseSplit::Unspecified`] because a bare
    /// case list carries no partition tag; use
    /// [`evaluate_split`](Self::evaluate_split) to keep one.
    ///
    /// # Errors
    ///
    /// The evaluator's own error.
    pub async fn evaluate_traced(&self, prompt: &str, cases: &[Case]) -> Result<EvaluationSummary> {
        self.summary_for(prompt, cases).await
    }

    /// Scores `prompt` over a tagged [`SplitDataset`] and keeps its tag.
    ///
    /// The same traced evaluation as [`evaluate_traced`](Self::evaluate_traced)
    /// — the mean score, the per-case results, the per-tool call counts, and the
    /// summed [`RunTelemetry`] are identical for the same cases — with
    /// [`EvaluationSummary::split`] set from
    /// [`dataset.split`](SplitDataset::split), so a report can say whether the
    /// numbers were measured on the training or the validation partition.
    ///
    /// # Errors
    ///
    /// The evaluator's own error.
    pub async fn evaluate_split(
        &self,
        prompt: &str,
        dataset: &SplitDataset,
    ) -> Result<EvaluationSummary> {
        let mut summary = self.summary_for(prompt, dataset.cases()).await?;
        summary.split = dataset.split;
        Ok(summary)
    }

    /// The explicit traced name of [`evaluate_split`](Self::evaluate_split).
    ///
    /// Identical contract: the traced evaluation of `dataset.cases()` with
    /// [`EvaluationSummary::split`] carried from the dataset, so
    /// tool usage and telemetry are reported exactly as
    /// [`evaluate_traced`](Self::evaluate_traced) reports them.
    ///
    /// # Errors
    ///
    /// The evaluator's own error.
    pub async fn evaluate_split_traced(
        &self,
        prompt: &str,
        dataset: &SplitDataset,
    ) -> Result<EvaluationSummary> {
        self.evaluate_split(prompt, dataset).await
    }

    /// The one implementation behind [`evaluate_traced`](Self::evaluate_traced)
    /// and [`evaluate_split`](Self::evaluate_split).
    ///
    /// Returns the tagged-as-[`CaseSplit::Unspecified`] summary; the caller
    /// retags it when it knows the partition.
    async fn summary_for(&self, prompt: &str, cases: &[Case]) -> Result<EvaluationSummary> {
        let (mean_score, runs) = self.evaluator_for(prompt).score_runs(prompt, cases).await?;
        let traced: Vec<(EvaluatedCase, ToolTrace)> = runs
            .iter()
            .map(|run| (run.evaluated.clone(), run.trace.clone()))
            .collect();
        let tool_usage = tool_usage(&traced);
        let mut telemetry = RunTelemetry::default();
        for run in &runs {
            telemetry.merge(run.telemetry);
        }
        Ok(EvaluationSummary {
            mean_score,
            results: runs.into_iter().map(|run| run.evaluated).collect(),
            tool_usage,
            telemetry,
            split: CaseSplit::Unspecified,
        })
    }

    /// Trains `prompt` on `train` and selects on `val`.
    ///
    /// Each train case keeps its declared tools, which the instruction
    /// optimizer renders into its rewrite prompt. The two lists are untagged, so
    /// the result carries no `train_score`: when `val` is non-empty it is
    /// reported as [`val_score`](OptimizationResult::val_score) with
    /// [`selected_on`](OptimizationResult::selected_on) left `None`, and when
    /// `val` is empty the fallback means the single score is really a training
    /// score, so both stay `None`. Use [`train_split`](Self::train_split) to tag
    /// the partitions and report both scores.
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

        // An empty validation split falls back to the train split (documented
        // deviation from Python, which requires both), so nothing was held out.
        let scored_on = if val.is_empty() { train } else { val };
        let fit = self.fit(prompt, train, scored_on).await?;
        let result = OptimizationResult {
            best_prompt: fit.best_prompt,
            score: fit.best_score,
            iterations: fit.iterations,
            ..OptimizationResult::default()
        };
        if val.is_empty() {
            Ok(result)
        } else {
            Ok(result.with_val_score(fit.best_score))
        }
    }

    /// Trains `prompt` on a tagged train split and selects on a tagged
    /// validation split.
    ///
    /// Exactly [`train`](Self::train)'s discipline — baseline, early stop, one
    /// instruction round per epoch on the train cases only, selection on the
    /// validation cases with the shared tie-break, and the same epoch cap — with
    /// two differences:
    ///
    /// - the partitions must carry their tags (`train.split ==
    ///   [CaseSplit::Train]`, `val.split == [CaseSplit::Val]`), because a
    ///   mis-tagged dataset is almost always a caller mistake, and
    /// - an empty validation split is a validation error instead of a fallback:
    ///   a tagged run promises a held-out score.
    ///
    /// The returned [`OptimizationResult`] reports the selected validation score
    /// as [`score`](OptimizationResult::score), the same value in
    /// [`val_score`](OptimizationResult::val_score), the winning prompt's
    /// in-sample mean over the train cases in
    /// [`train_score`](OptimizationResult::train_score) (measured once at the
    /// end), and [`CaseSplit::Val`] in
    /// [`selected_on`](OptimizationResult::selected_on).
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when the prompt is blank, the configuration is out
    /// of range, `train` is not tagged [`CaseSplit::Train`], `val` is not tagged
    /// [`CaseSplit::Val`], or either split is empty; otherwise the evaluator's
    /// own error.
    pub async fn train_split(
        &self,
        prompt: &str,
        train: &SplitDataset,
        val: &SplitDataset,
    ) -> Result<OptimizationResult> {
        if prompt.trim().is_empty() {
            return Err(Error::validation("prompt must not be blank"));
        }
        self.config.validate()?;
        require_tag(train, CaseSplit::Train, "train")?;
        require_tag(val, CaseSplit::Val, "val")?;
        if train.is_empty() {
            return Err(Error::validation(
                "train split must not be empty: there is nothing to optimize on",
            ));
        }
        if val.is_empty() {
            return Err(Error::validation(
                "val split must not be empty: there is nothing to select on",
            ));
        }

        let fit = self.fit(prompt, train.cases(), val.cases()).await?;
        // The in-sample score of the winner, over the train cases only, measured
        // once at the end with the evaluator the epochs used.
        let (train_score, _) = self
            .evaluator_for(prompt)
            .score_cases(&fit.best_prompt, train.cases())
            .await?;
        Ok(OptimizationResult {
            best_prompt: fit.best_prompt,
            score: fit.best_score,
            iterations: fit.iterations,
            ..OptimizationResult::default()
        }
        .with_train_score(train_score)
        .with_val_score(fit.best_score)
        .with_selected_on(CaseSplit::Val))
    }

    /// The seed → selection → epoch loop shared by [`train`](Self::train) and
    /// [`train_split`](Self::train_split).
    ///
    /// `train_cases` is the only case list ever handed to the instruction
    /// optimizer, so a held-out case can never be fitted against; `val_cases` is
    /// used solely to score the seed and each finished candidate. Callers
    /// validate their inputs first.
    async fn fit(&self, prompt: &str, train_cases: &[Case], val_cases: &[Case]) -> Result<Fit> {
        let evaluator = self.evaluator_for(prompt);
        let seed = ensure_placeholders(prompt, prompt.trim());
        let (mut best_score, _) = evaluator.score_cases(&seed, val_cases).await?;
        let mut best_prompt = seed;
        let mut iterations = 0;

        while iterations < self.config.iterations {
            if best_score + 1e-12 >= self.config.early_stop_score {
                break;
            }
            let instruction = self.instruction_for(&evaluator);
            let candidate = instruction.optimize(&best_prompt, train_cases).await?;
            let (val_score, _) = evaluator
                .score_cases(&candidate.best_prompt, val_cases)
                .await?;
            iterations += 1;
            if is_better(val_score, &candidate.best_prompt, best_score, &best_prompt) {
                best_score = val_score;
                best_prompt = candidate.best_prompt;
            }
        }

        Ok(Fit {
            best_prompt,
            best_score,
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

/// The outcome of the shared epoch loop: the winning prompt, the selection score
/// it won with, and how many epochs actually ran.
struct Fit {
    /// The prompt that scored best on the selection split.
    best_prompt: String,
    /// The selection-split score of [`Fit::best_prompt`].
    best_score: f64,
    /// The number of epochs that ran (an early stop reports fewer than the cap).
    iterations: usize,
}

/// Rejects a dataset whose tag is not `expected`, naming the argument and the
/// tag it should have carried.
fn require_tag(dataset: &SplitDataset, expected: CaseSplit, name: &str) -> Result<()> {
    if dataset.split == expected {
        return Ok(());
    }
    Err(Error::validation(format!(
        "{name} split must be tagged `{expected}`, got `{}`",
        dataset.split
    )))
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
    use crate::tune::evaluator::{
        DeterministicEvaluator, EvaluatedRun, ModelPromptRunner, RunnerEvaluator,
    };
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
        // The score is now measured on the train split, so nothing is held out:
        // no held-out score and no selection tag are reported.
        assert!(result.val_score.is_none());
        assert!(result.selected_on.is_none());
        assert!(result.train_score.is_none());
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
        assert!(summary.telemetry.is_empty());
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

            async fn evaluate_run(&self, prompt: &str, case: &Case) -> Result<EvaluatedRun> {
                let evaluated = self.evaluate(prompt, case).await?;
                let trace =
                    ToolTrace::new([ToolCallRecord::success("search", json!({}), json!("ok"))]);
                Ok(EvaluatedRun::new(evaluated, trace, RunTelemetry::default()))
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
        assert!(summary.telemetry.is_empty());
    }

    /// Scores every case `0.4`, calls `search` once, and reports 10 ms and 3+4
    /// tokens per case.
    struct Measured;

    #[async_trait]
    impl CaseEvaluator for Measured {
        async fn evaluate(&self, _prompt: &str, case: &Case) -> Result<EvaluatedCase> {
            EvaluatedCase::new(case.clone(), BTreeMap::new(), 0.4, "measured")
        }

        async fn evaluate_run(&self, prompt: &str, case: &Case) -> Result<EvaluatedRun> {
            let evaluated = self.evaluate(prompt, case).await?;
            let trace = ToolTrace::new([ToolCallRecord::success("search", json!({}), json!("ok"))]);
            let telemetry = RunTelemetry::default()
                .with_latency_ms(10.0)
                .with_prompt_tokens(3)
                .with_completion_tokens(4);
            Ok(EvaluatedRun::new(evaluated, trace, telemetry))
        }
    }

    #[tokio::test]
    async fn evaluate_traced_aggregates_the_run_telemetry() {
        let cases = train();
        let trainer = Trainer::new().with_evaluator(Arc::new(Measured));
        let (mean, results) = trainer.evaluate(SEED, &cases).await.unwrap();
        let summary = trainer.evaluate_traced(SEED, &cases).await.unwrap();

        // Score, results, and tool usage are exactly `evaluate`'s.
        assert!((summary.mean_score - mean).abs() < 1e-12);
        assert_eq!(summary.results, results);
        assert_eq!(
            summary.tool_usage,
            BTreeMap::from([("search".to_owned(), 2)])
        );
        // Telemetry is summed in case order.
        assert_eq!(summary.telemetry.latency_ms, Some(20.0));
        assert_eq!(summary.telemetry.prompt_tokens, Some(6));
        assert_eq!(summary.telemetry.completion_tokens, Some(8));
        assert_eq!(summary.telemetry.total_tokens(), Some(14));
    }

    #[tokio::test]
    async fn evaluation_summary_without_measurements_has_absent_telemetry() {
        let summary = Trainer::new()
            .evaluate_traced(SEED, &train())
            .await
            .unwrap();
        assert!(summary.telemetry.is_empty());
        assert!(!summary.is_empty());
    }

    #[test]
    fn evaluation_summary_round_trips_through_json() {
        let evaluated =
            EvaluatedCase::new(case("c1", "q", "a"), BTreeMap::new(), 0.5, "half").unwrap();
        let summary = EvaluationSummary {
            mean_score: 0.5,
            results: vec![evaluated],
            tool_usage: BTreeMap::from([("search".to_owned(), 2)]),
            telemetry: RunTelemetry::default()
                .with_latency_ms(1.5)
                .with_prompt_tokens(2)
                .with_completion_tokens(3),
            split: CaseSplit::Val,
        };

        let value = serde_json::to_value(&summary).unwrap();
        let object = value.as_object().unwrap();
        for key in ["mean_score", "results", "tool_usage", "telemetry", "split"] {
            assert!(object.contains_key(key), "missing field {key}");
        }
        assert_eq!(value["telemetry"]["latency_ms"], json!(1.5));
        assert_eq!(value["telemetry"]["prompt_tokens"], json!(2));
        assert_eq!(value["split"], json!("val"));
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
        // An older payload without the key deserializes with absent telemetry and
        // no partition tag.
        assert!(partial.telemetry.is_empty());
        assert_eq!(partial.split, CaseSplit::Unspecified);
        let defaulted: EvaluationSummary = serde_json::from_str("{}").unwrap();
        assert_eq!(defaulted, EvaluationSummary::default());
        assert_eq!(defaulted.split, CaseSplit::Unspecified);
    }

    /// The train split of [`train`] tagged for a trainer.
    fn train_split() -> SplitDataset {
        SplitDataset::new(CaseSplit::Train, train())
    }

    /// The validation split of [`val`] tagged for a trainer.
    fn val_split() -> SplitDataset {
        SplitDataset::new(CaseSplit::Val, val())
    }

    #[tokio::test]
    async fn evaluate_split_tags_the_summary_and_matches_evaluate_traced() {
        let dataset = val_split();
        let summary = Trainer::new().evaluate_split(SEED, &dataset).await.unwrap();
        let untagged = Trainer::new()
            .evaluate_traced(SEED, dataset.cases())
            .await
            .unwrap();

        assert_eq!(summary.split, CaseSplit::Val);
        assert_eq!(summary.mean_score, untagged.mean_score);
        assert_eq!(summary.results, untagged.results);
        assert_eq!(summary.tool_usage, untagged.tool_usage);
        assert_eq!(summary.telemetry, untagged.telemetry);
        // The untagged sibling keeps reporting no partition.
        assert_eq!(untagged.split, CaseSplit::Unspecified);
    }

    #[tokio::test]
    async fn evaluate_split_keeps_the_train_tag_and_the_traced_telemetry() {
        let dataset = SplitDataset::new(CaseSplit::Train, train());
        let summary = Trainer::new()
            .with_evaluator(Arc::new(Measured))
            .evaluate_split(SEED, &dataset)
            .await
            .unwrap();
        assert_eq!(summary.split, CaseSplit::Train);
        assert_eq!(summary.results.len(), 2);
        assert_eq!(
            summary.tool_usage,
            BTreeMap::from([("search".to_owned(), 2)])
        );
        assert_eq!(summary.telemetry.latency_ms, Some(20.0));
        assert_eq!(summary.telemetry.total_tokens(), Some(14));

        let empty = Trainer::new()
            .evaluate_split(SEED, &SplitDataset::new(CaseSplit::Val, Vec::new()))
            .await
            .unwrap();
        assert!(empty.is_empty());
        assert_eq!(empty.split, CaseSplit::Val);
    }

    #[tokio::test]
    async fn evaluate_split_traced_is_the_same_tagged_contract() {
        let dataset = val_split();
        let explicit = Trainer::new()
            .evaluate_split_traced(SEED, &dataset)
            .await
            .unwrap();
        let short = Trainer::new().evaluate_split(SEED, &dataset).await.unwrap();
        assert_eq!(explicit, short);
        assert_eq!(explicit.split, CaseSplit::Val);
        assert!(!explicit.is_empty());
    }

    #[tokio::test]
    async fn train_split_reports_both_scores_and_the_selection_split() {
        let train = train_split();
        let val = val_split();
        let result = Trainer::new()
            .train_split(SEED, &train, &val)
            .await
            .unwrap();

        assert_eq!(result.selected_on, Some(CaseSplit::Val));
        assert_eq!(result.val_score, Some(result.score));
        let train_score = result
            .train_score
            .expect("the in-sample score must be reported");
        assert!((0.0..=1.0).contains(&train_score), "{train_score}");

        // The in-sample score is the winner measured over the train cases only.
        let (expected, _) = Trainer::new()
            .evaluate(&result.best_prompt, train.cases())
            .await
            .unwrap();
        assert!((train_score - expected).abs() < 1e-9);

        // The selection discipline is exactly `train`'s.
        let untagged = Trainer::new()
            .train(SEED, train.cases(), val.cases())
            .await
            .unwrap();
        assert_eq!(result.best_prompt, untagged.best_prompt);
        assert_eq!(result.score, untagged.score);
        assert_eq!(result.iterations, untagged.iterations);
        assert!(result.iterations >= 1);
    }

    #[tokio::test]
    async fn train_split_early_stop_skips_optimization() {
        let train = train_split();
        let val = val_split();
        let result = Trainer::new()
            .with_config(OptimizerConfig::default().with_early_stop_score(0.0))
            .train_split(SEED, &train, &val)
            .await
            .unwrap();
        assert_eq!(result.iterations, 0);
        assert_eq!(result.best_prompt, SEED);
        assert_eq!(result.selected_on, Some(CaseSplit::Val));
        assert_eq!(result.val_score, Some(result.score));
        // The seed is still measured in-sample, once.
        assert!(result.train_score.is_some());
    }

    #[tokio::test]
    async fn train_split_epochs_are_bounded_by_the_configuration() {
        let result = Trainer::new()
            .with_config(
                OptimizerConfig::default()
                    .with_iterations(2)
                    .with_early_stop_score(1.0),
            )
            .train_split(SEED, &train_split(), &val_split())
            .await
            .unwrap();
        assert_eq!(result.iterations, 2);
        assert_eq!(result.selected_on, Some(CaseSplit::Val));
    }

    #[tokio::test]
    async fn train_split_rejects_mis_tagged_splits() {
        for (train, val, expected) in [
            (
                SplitDataset::new(CaseSplit::Train, train()),
                SplitDataset::new(CaseSplit::Train, val()),
                "val",
            ),
            (
                SplitDataset::new(CaseSplit::Unspecified, train()),
                SplitDataset::new(CaseSplit::Val, val()),
                "train",
            ),
            (
                SplitDataset::new(CaseSplit::Val, train()),
                SplitDataset::new(CaseSplit::Train, val()),
                "train",
            ),
            (
                SplitDataset::new(CaseSplit::Train, train()),
                SplitDataset::new(CaseSplit::Unspecified, val()),
                "val",
            ),
        ] {
            let error = Trainer::new()
                .train_split(SEED, &train, &val)
                .await
                .unwrap_err();
            assert_eq!(
                error.kind(),
                openjiuwen_common::error::ErrorKind::Validation
            );
            let message = error.to_string();
            assert!(message.contains("tagged"), "{message}");
            assert!(message.contains(expected), "{message}");
        }
    }

    #[tokio::test]
    async fn train_split_rejects_empty_splits() {
        let empty_train = SplitDataset::new(CaseSplit::Train, Vec::new());
        let error = Trainer::new()
            .train_split(SEED, &empty_train, &val_split())
            .await
            .unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        assert!(error.to_string().contains("train"));

        // A tagged run promises a held-out score, so an empty validation split is
        // an error rather than the untagged fallback to train.
        let empty_val = SplitDataset::new(CaseSplit::Val, Vec::new());
        let error = Trainer::new()
            .train_split(SEED, &train_split(), &empty_val)
            .await
            .unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        assert!(error.to_string().contains("val"));

        let error = Trainer::new()
            .train_split("  ", &train_split(), &val_split())
            .await
            .unwrap_err();
        assert!(error.to_string().contains("prompt"));

        let error = Trainer::new()
            .with_config(OptimizerConfig::default().with_iterations(0))
            .train_split(SEED, &train_split(), &val_split())
            .await
            .unwrap_err();
        assert!(error.to_string().contains("iterations"));
    }

    #[tokio::test]
    async fn train_reports_the_val_score_without_a_selection_tag() {
        let result = Trainer::new().train(SEED, &train(), &val()).await.unwrap();
        assert_eq!(result.val_score, Some(result.score));
        assert!(result.selected_on.is_none());
        // The untagged caller gets no in-sample score, because the only split it
        // supplied is the held-out one.
        assert!(result.train_score.is_none());
    }

    /// Records the case ids of every `score_cases` call, then delegates.
    struct Recorder {
        /// The evaluator whose scores the trainer actually uses.
        inner: Arc<dyn CaseEvaluator>,
        /// The case-id list of each `score_cases` call, in call order.
        calls: Arc<std::sync::Mutex<Vec<Vec<String>>>>,
    }

    #[async_trait]
    impl CaseEvaluator for Recorder {
        async fn evaluate(&self, prompt: &str, case: &Case) -> Result<EvaluatedCase> {
            self.inner.evaluate(prompt, case).await
        }

        async fn score_cases(
            &self,
            prompt: &str,
            cases: &[Case],
        ) -> Result<(f64, Vec<EvaluatedCase>)> {
            let ids: Vec<String> = cases.iter().map(|case| case.case_id.clone()).collect();
            self.calls
                .lock()
                .expect("the recording lock is not poisoned")
                .push(ids);
            self.inner.score_cases(prompt, cases).await
        }
    }

    #[tokio::test]
    async fn candidate_search_never_sees_validation_cases() {
        let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
        let trainer = Trainer::new().with_evaluator(Arc::new(Recorder {
            inner: Arc::new(DeterministicEvaluator::new(SEED)),
            calls: Arc::clone(&calls),
        }));
        let train = train_split();
        let val = val_split();
        trainer.train_split(SEED, &train, &val).await.unwrap();

        let calls = calls
            .lock()
            .expect("the recording lock is not poisoned")
            .clone();
        assert!(!calls.is_empty(), "the trainer must score something");
        let train_ids = ["t1", "t2"];
        let val_ids = ["v1"];
        for ids in &calls {
            let all_train = ids.iter().all(|id| train_ids.contains(&id.as_str()));
            let all_val = ids.iter().all(|id| val_ids.contains(&id.as_str()));
            assert!(
                all_train || all_val,
                "a scored case set mixed the splits: {ids:?}"
            );
        }
        assert!(
            calls
                .iter()
                .any(|ids| !ids.is_empty() && ids.iter().all(|id| train_ids.contains(&id.as_str()))),
            "at least one call must score only the train split: {calls:?}"
        );
        assert!(
            calls
                .iter()
                .any(|ids| !ids.is_empty() && ids.iter().all(|id| val_ids.contains(&id.as_str()))),
            "at least one call must score only the val split: {calls:?}"
        );
    }

    /// A validation case whose inputs and label carry a unique marker token.
    fn poisoned_val() -> Vec<Case> {
        let mut inputs = BTreeMap::new();
        inputs.insert(
            "question".to_owned(),
            Value::String("does VALONLYTOKEN appear?".to_owned()),
        );
        let mut label = BTreeMap::new();
        label.insert(
            "answer".to_owned(),
            Value::String("VALONLYTOKEN".to_owned()),
        );
        vec![Case::new(inputs, label).unwrap().with_case_id("v1")]
    }

    #[tokio::test]
    async fn train_split_never_learns_a_validation_token() {
        const TOKEN: &str = "VALONLYTOKEN";
        let train = train_split();
        let val = SplitDataset::new(CaseSplit::Val, poisoned_val());
        let model =
            ScriptedModel::text("<PROMPT_OPTIMIZED>Answer step by step.</PROMPT_OPTIMIZED>");
        let result = Trainer::new()
            .with_model(Arc::new(model.clone()))
            .train_split(SEED, &train, &val)
            .await
            .unwrap();

        assert!(result.iterations >= 1);
        assert!(
            !result.best_prompt.contains(TOKEN),
            "the winning prompt learned a validation token: {}",
            result.best_prompt
        );
        for record in &result.records {
            for candidate in &record.candidates {
                assert!(
                    !candidate.prompt().contains(TOKEN),
                    "a candidate prompt learned a validation token: {}",
                    candidate.prompt()
                );
            }
        }
        // Nothing the model was asked to rewrite carried the validation case.
        assert!(model.call_count() >= 1);
        for call in model.calls() {
            for message in &call.messages {
                assert!(
                    !message.text().contains(TOKEN),
                    "a rewrite request carried the validation case: {}",
                    message.text()
                );
            }
        }
    }
}
