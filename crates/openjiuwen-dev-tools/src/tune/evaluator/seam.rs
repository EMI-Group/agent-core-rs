//! The evaluator seam: compose a runner and a scorer into `EvaluatedCase`s.
//!
//! Every evaluation has a traced form: [`CaseEvaluator::evaluate_traced`] also
//! returns the [`ToolTrace`] behind the answer, and
//! [`CaseEvaluator::score_cases_traced`] keeps the per-case traces.
//! [`tool_usage`] aggregates those traces into per-tool call counts.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::llm::Model;

use crate::tune::base::{Case, EvaluatedCase};
use crate::tune::tool::{ToolStubs, ToolTrace};

use super::runner::ModelPromptRunner;
use super::scorer::LabelScorer;
use super::{CaseScorer, PromptRunner};

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

    /// Evaluates `prompt` on `case` and reports the tool calls behind the
    /// answer.
    ///
    /// The default evaluates normally and reports an empty trace; an evaluator
    /// over a tool-aware runner overrides it.
    ///
    /// # Errors
    ///
    /// Whatever [`evaluate`](Self::evaluate) returns.
    async fn evaluate_traced(
        &self,
        prompt: &str,
        case: &Case,
    ) -> Result<(EvaluatedCase, ToolTrace)> {
        Ok((self.evaluate(prompt, case).await?, ToolTrace::default()))
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
        self.evaluate_traced(prompt, case)
            .await
            .map(|(evaluated, _)| evaluated)
    }

    async fn evaluate_traced(
        &self,
        prompt: &str,
        case: &Case,
    ) -> Result<(EvaluatedCase, ToolTrace)> {
        let outcome = self.runner.run_traced(prompt, case).await?;
        let (score, reason) = self
            .scorer
            .score_with_trace(case, &outcome.answer, &outcome.trace);
        let evaluated =
            EvaluatedCase::new(case.clone(), outcome.answer, sanitize_score(score), reason)?;
        Ok((evaluated, outcome.trace))
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

    async fn evaluate_traced(
        &self,
        prompt: &str,
        case: &Case,
    ) -> Result<(EvaluatedCase, ToolTrace)> {
        self.inner.evaluate_traced(prompt, case).await
    }
}
