//! The evaluator seam: compose a runner and a scorer into `EvaluatedCase`s.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::llm::Model;

use crate::tune::base::{Case, EvaluatedCase};

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
        let answer = self.runner.run(prompt, case).await?;
        let (score, reason) = self.scorer.score(case, &answer);
        EvaluatedCase::new(case.clone(), answer, sanitize_score(score), reason)
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
}
