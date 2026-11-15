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
mod tests;
