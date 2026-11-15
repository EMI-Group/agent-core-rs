//! The evaluator seam: compose a runner and a scorer into `EvaluatedCase`s.
//!
//! Every evaluation has a traced form: [`CaseEvaluator::evaluate_traced`] also
//! returns the [`ToolTrace`] behind the answer, and
//! [`CaseEvaluator::score_cases_traced`] keeps the per-case traces.
//! [`tool_usage`] aggregates those traces into per-tool call counts.
//!
//! [`CaseEvaluator::evaluate_run`] is the same evaluation with what the run cost
//! attached: an [`EvaluatedRun`] carries the [`EvaluatedCase`], the trace, and
//! the [`RunTelemetry`] the runner measured. [`CaseEvaluator::score_runs`] keeps
//! those runs per case, and `evaluate_traced`/`score_cases_traced` remain the
//! telemetry-free views of the same calls.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::llm::Model;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::tune::base::{Case, EvaluatedCase};
use crate::tune::tool::{RunTelemetry, ToolStubs, ToolTrace};

use super::runner::ModelPromptRunner;
use super::scorer::LabelScorer;
use super::{CaseScorer, PromptRunner};

/// One evaluation together with everything the run reported.
///
/// The [`EvaluatedCase`] is the score and answer, the [`ToolTrace`] the calls
/// behind that answer, and the [`RunTelemetry`] what the run cost. Telemetry is
/// absent when the runner measured nothing, so a deterministic or scripted
/// evaluator reports the default (empty) telemetry without pretending it
/// measured zero.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct EvaluatedRun {
    /// The scored case.
    pub evaluated: EvaluatedCase,
    /// The tool calls executed while producing the answer.
    pub trace: ToolTrace,
    /// What the run cost, when the runner measured it.
    pub telemetry: RunTelemetry,
}

impl EvaluatedRun {
    /// Composes a scored case with its trace and telemetry.
    #[must_use]
    pub fn new(evaluated: EvaluatedCase, trace: ToolTrace, telemetry: RunTelemetry) -> Self {
        Self {
            evaluated,
            trace,
            telemetry,
        }
    }

    /// The score of the evaluated case, in `0.0..=1.0`.
    #[must_use]
    pub fn score(&self) -> f64 {
        self.evaluated.score
    }

    /// The answer the scorer compared against the case label.
    #[must_use]
    pub fn answer(&self) -> &BTreeMap<String, Value> {
        &self.evaluated.answer
    }

    /// The tools the run called, in call order and with repetitions.
    #[must_use]
    pub fn used_tools(&self) -> Vec<String> {
        self.trace
            .names()
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>()
    }

    /// Whether the run reported no telemetry at all.
    #[must_use]
    pub fn is_empty_telemetry(&self) -> bool {
        self.telemetry.is_empty()
    }
}

impl Default for EvaluatedRun {
    /// An empty run: a placeholder case nothing evaluated, an empty trace, and
    /// absent telemetry.
    ///
    /// Deserialization needs it for a payload that omits fields; the
    /// placeholder case carries no inputs, no label, and a blank id.
    fn default() -> Self {
        Self {
            evaluated: EvaluatedCase {
                case: Case {
                    inputs: BTreeMap::new(),
                    label: BTreeMap::new(),
                    tools: None,
                    case_id: String::new(),
                },
                answer: BTreeMap::new(),
                score: 0.0,
                reason: String::new(),
            },
            trace: ToolTrace::default(),
            telemetry: RunTelemetry::default(),
        }
    }
}

/// Evaluates one candidate prompt against one case.
#[async_trait]
pub trait CaseEvaluator: Send + Sync {
    /// Evaluates `prompt` on `case`.
    ///
    /// # Errors
    ///
    /// Propagates the runner's error and any score that cannot be represented
    /// as an [`EvaluatedCase`].
    async fn evaluate(&self, prompt: &str, case: &Case) -> Result<EvaluatedCase>;

    /// Evaluates `prompt` on `case` and reports the trace and telemetry behind
    /// the answer.
    ///
    /// The default evaluates normally and reports an empty trace and absent
    /// telemetry; an evaluator over a tool-aware or measuring runner overrides
    /// it.
    ///
    /// # Errors
    ///
    /// Whatever [`evaluate`](Self::evaluate) returns.
    async fn evaluate_run(&self, prompt: &str, case: &Case) -> Result<EvaluatedRun> {
        let evaluated = self.evaluate(prompt, case).await?;
        Ok(EvaluatedRun::new(
            evaluated,
            ToolTrace::default(),
            RunTelemetry::default(),
        ))
    }

    /// Evaluates `prompt` on `case` and reports the tool calls behind the
    /// answer.
    ///
    /// The default takes the traced-and-measured run from
    /// [`evaluate_run`](Self::evaluate_run) and drops the telemetry, so an
    /// evaluator that overrides only `evaluate_run` is honoured here too.
    ///
    /// # Errors
    ///
    /// Whatever [`evaluate_run`](Self::evaluate_run) returns.
    async fn evaluate_traced(
        &self,
        prompt: &str,
        case: &Case,
    ) -> Result<(EvaluatedCase, ToolTrace)> {
        let run = self.evaluate_run(prompt, case).await?;
        Ok((run.evaluated, run.trace))
    }

    /// Evaluates `prompt` on every case and returns the mean score plus the
    /// per-case results.
    ///
    /// An empty `cases` slice yields `(0.0, vec![])`.
    ///
    /// # Errors
    ///
    /// Propagates the first [`evaluate`](Self::evaluate) failure.
    async fn score_cases(&self, prompt: &str, cases: &[Case]) -> Result<(f64, Vec<EvaluatedCase>)> {
        if cases.is_empty() {
            return Ok((0.0, Vec::new()));
        }
        let mut evaluated = Vec::with_capacity(cases.len());
        for case in cases {
            evaluated.push(self.evaluate(prompt, case).await?);
        }
        let total: f64 = evaluated.iter().map(|case| case.score).sum();
        Ok((total / evaluated.len() as f64, evaluated))
    }

    /// Like [`score_cases`](Self::score_cases) but keeps each case's tool trace.
    ///
    /// An empty `cases` slice yields `(0.0, vec![])`.
    ///
    /// # Errors
    ///
    /// Propagates the first [`evaluate_traced`](Self::evaluate_traced) failure.
    async fn score_cases_traced(
        &self,
        prompt: &str,
        cases: &[Case],
    ) -> Result<(f64, Vec<(EvaluatedCase, ToolTrace)>)> {
        if cases.is_empty() {
            return Ok((0.0, Vec::new()));
        }
        let mut evaluated = Vec::with_capacity(cases.len());
        for case in cases {
            evaluated.push(self.evaluate_traced(prompt, case).await?);
        }
        let total: f64 = evaluated.iter().map(|(case, _)| case.score).sum();
        Ok((total / evaluated.len() as f64, evaluated))
    }

    /// Like [`score_cases`](Self::score_cases) but keeps each case's trace and
    /// telemetry.
    ///
    /// An empty `cases` slice yields `(0.0, vec![])`.
    ///
    /// # Errors
    ///
    /// Propagates the first [`evaluate_run`](Self::evaluate_run) failure.
    async fn score_runs(&self, prompt: &str, cases: &[Case]) -> Result<(f64, Vec<EvaluatedRun>)> {
        if cases.is_empty() {
            return Ok((0.0, Vec::new()));
        }
        let mut runs = Vec::with_capacity(cases.len());
        for case in cases {
            runs.push(self.evaluate_run(prompt, case).await?);
        }
        let total: f64 = runs.iter().map(EvaluatedRun::score).sum();
        Ok((total / runs.len() as f64, runs))
    }
}

/// Merges the tool-usage counts of every traced result.
///
/// Each trace's [`usage`](ToolTrace::usage) counts by tool name and the counts
/// are summed, so the result answers "how often did the agent call each tool
/// over this run".
#[must_use]
pub fn tool_usage(results: &[(EvaluatedCase, ToolTrace)]) -> BTreeMap<String, usize> {
    let mut usage = BTreeMap::new();
    for (_, trace) in results {
        for (name, count) in trace.usage() {
            *usage.entry(name).or_insert(0) += count;
        }
    }
    usage
}

/// A [`CaseEvaluator`] composing any [`PromptRunner`] with any [`CaseScorer`].
pub struct RunnerEvaluator {
    /// Produces the answer map for a candidate prompt.
    runner: Arc<dyn PromptRunner>,
    /// Turns the answer map into a score.
    scorer: Arc<dyn CaseScorer>,
}

impl RunnerEvaluator {
    /// Creates an evaluator over `runner` using the default [`LabelScorer`].
    #[must_use]
    pub fn new(runner: Arc<dyn PromptRunner>) -> Self {
        Self {
            runner,
            scorer: Arc::new(LabelScorer),
        }
    }

    /// Replaces the scorer.
    #[must_use]
    pub fn with_scorer(mut self, scorer: Arc<dyn CaseScorer>) -> Self {
        self.scorer = scorer;
        self
    }

    /// The runner in use.
    #[must_use]
    pub fn runner(&self) -> &Arc<dyn PromptRunner> {
        &self.runner
    }

    /// The scorer in use.
    #[must_use]
    pub fn scorer(&self) -> &Arc<dyn CaseScorer> {
        &self.scorer
    }
}

impl fmt::Debug for RunnerEvaluator {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RunnerEvaluator")
            .field("runner", &"custom runner")
            .field("scorer", &"custom scorer")
            .finish()
    }
}

#[async_trait]
impl CaseEvaluator for RunnerEvaluator {
    async fn evaluate(&self, prompt: &str, case: &Case) -> Result<EvaluatedCase> {
        self.evaluate_run(prompt, case)
            .await
            .map(|run| run.evaluated)
    }

    async fn evaluate_run(&self, prompt: &str, case: &Case) -> Result<EvaluatedRun> {
        let outcome = self.runner.run_traced(prompt, case).await?;
        let (score, reason) = self
            .scorer
            .score_with_trace(case, &outcome.answer, &outcome.trace);
        let evaluated =
            EvaluatedCase::new(case.clone(), outcome.answer, sanitize_score(score), reason)?;
        Ok(EvaluatedRun::new(
            evaluated,
            outcome.trace,
            outcome.telemetry,
        ))
    }
}

/// Clamps a score into `0.0..=1.0`; a non-finite score becomes `0.0`.
fn sanitize_score(score: f64) -> f64 {
    if score.is_finite() {
        score.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// A [`CaseEvaluator`] that runs candidate prompts through a [`Model`].
///
/// This is the model-backed preset of [`RunnerEvaluator`]: a
/// [`ModelPromptRunner`] plus a [`CaseScorer`] that defaults to
/// [`LabelScorer`].
pub struct ModelEvaluator {
    /// The composed runner-backed evaluator.
    inner: RunnerEvaluator,
}

impl ModelEvaluator {
    /// Creates an evaluator over `model` using the default [`LabelScorer`].
    #[must_use]
    pub fn new(model: Arc<dyn Model>) -> Self {
        Self {
            inner: RunnerEvaluator::new(Arc::new(ModelPromptRunner::new(model))),
        }
    }

    /// Creates an evaluator over `model` that executes the case's tools through
    /// `stubs`.
    #[must_use]
    pub fn with_stubs(model: Arc<dyn Model>, stubs: ToolStubs) -> Self {
        Self {
            inner: RunnerEvaluator::new(Arc::new(ModelPromptRunner::new(model).with_stubs(stubs))),
        }
    }

    /// Replaces the scorer.
    #[must_use]
    pub fn with_scorer(mut self, scorer: Arc<dyn CaseScorer>) -> Self {
        self.inner = self.inner.with_scorer(scorer);
        self
    }

    /// The composed evaluator.
    #[must_use]
    pub fn inner(&self) -> &RunnerEvaluator {
        &self.inner
    }
}

impl fmt::Debug for ModelEvaluator {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModelEvaluator")
            .field("inner", &self.inner)
            .finish()
    }
}

#[async_trait]
impl CaseEvaluator for ModelEvaluator {
    async fn evaluate(&self, prompt: &str, case: &Case) -> Result<EvaluatedCase> {
        self.inner.evaluate(prompt, case).await
    }

    async fn evaluate_run(&self, prompt: &str, case: &Case) -> Result<EvaluatedRun> {
        self.inner.evaluate_run(prompt, case).await
    }
}
