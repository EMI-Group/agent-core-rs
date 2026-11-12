//! Builder for [`AutoHarnessOrchestrator`].
//!
//! The builder is the only way to construct an orchestrator: it validates the
//! [`AutoHarnessConfig`], defaults every optional collaborator, and derives the
//! budget limits. Keeping it in its own module keeps the session loop readable.
//!
//! Defaults (all overridable):
//!
//! | collaborator | default |
//! |---|---|
//! | mutator | [`DeterministicMutator`] (one marked variant per parent) |
//! | selector | [`ElitistSelection`] (elite of one) |
//! | checkpoint store | [`InMemoryCheckpointStore`] (process memory) |
//! | cases | empty (the runner decides what an empty case set means) |
//! | seed body | [`DEFAULT_SEED_BODY`] |
//! | session id | a fresh `uuid_v4` |
//! | mutation count | `1` proposal per iteration |
//!
//! ## Evaluator defaulting
//!
//! Supplying `with_evaluator` wins over `with_runner`. Supplying only a runner
//! wraps it in a [`SessionIterationEvaluator`], which delegates to
//! [`HarnessEvaluator`] once per call and stamps the orchestrator's session id
//! and the iteration currently running. The shared [`Evaluator`] protocol only
//! receives the candidate and the cases (no iteration), so this is how the
//! harness seam keeps seeing `session_id` and `iteration` for tracing and
//! idempotency. Building without either a runner or an evaluator is a
//! configuration error: the session would have nothing to grade with.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::id::uuid_v4;

use crate::evolution::{
    Candidate, Case, DeterministicMutator, ElitistSelection, EvaluationReport, Evaluator,
    MutationStrategy, SelectionStrategy,
};

use super::budget::BudgetLimits;
use super::checkpoint::{CheckpointStore, InMemoryCheckpointStore};
use super::config::AutoHarnessConfig;
use super::orchestrator::{AutoHarnessOrchestrator, OrchestratorParts, DEFAULT_SEED_BODY};
use super::runner::{HarnessEvaluator, HarnessRunner};

/// Upper bound applied to a configured budget in seconds.
///
/// [`AutoHarnessConfig::validate`] already rejects non-positive and non-finite
/// values; the cap only keeps an absurdly large number from overflowing
/// [`Duration`], which would panic in `Duration::from_secs_f64`.
const MAX_BUDGET_SECS: f64 = 1e15;

/// Builds an [`AutoHarnessOrchestrator`].
///
/// Obtain one with [`AutoHarnessOrchestrator::builder`]; every `with_*` method
/// takes `self` and returns `self`, so calls chain into a single [`Self::build`].
pub struct AutoHarnessOrchestratorBuilder {
    /// Configuration to validate and hand to the orchestrator.
    config: AutoHarnessConfig,
    /// Harness runner; wrapped in an evaluator when no evaluator is supplied.
    runner: Option<Arc<dyn HarnessRunner>>,
    /// Explicit mutator.
    mutator: Option<Arc<dyn MutationStrategy>>,
    /// Explicit selection strategy.
    selector: Option<Arc<dyn SelectionStrategy>>,
    /// Explicit evaluator; wins over `runner`.
    evaluator: Option<Arc<dyn Evaluator>>,
    /// Explicit checkpoint store.
    checkpoint_store: Option<Arc<dyn CheckpointStore>>,
    /// Cases every candidate is graded against.
    cases: Vec<Case>,
    /// Explicit seed body.
    seed_body: Option<String>,
    /// Explicit session id.
    session_id: Option<String>,
    /// Number of proposals per iteration.
    mutation_count: usize,
}

impl AutoHarnessOrchestratorBuilder {
    /// Creates a builder for `config` with every collaborator at its default.
    pub(crate) fn new(config: AutoHarnessConfig) -> Self {
        Self {
            config,
            runner: None,
            mutator: None,
            selector: None,
            evaluator: None,
            checkpoint_store: None,
            cases: Vec::new(),
            seed_body: None,
            session_id: None,
            mutation_count: 1,
        }
    }

    /// Sets the harness runner the default evaluator wraps.
    #[must_use]
    pub fn with_runner(mut self, runner: Arc<dyn HarnessRunner>) -> Self {
        self.runner = Some(runner);
        self
    }

    /// Sets the mutation strategy (default [`DeterministicMutator`]).
    #[must_use]
    pub fn with_mutator(mut self, mutator: Arc<dyn MutationStrategy>) -> Self {
        self.mutator = Some(mutator);
        self
    }

    /// Sets the selection strategy (default [`ElitistSelection`]).
    #[must_use]
    pub fn with_selector(mut self, selector: Arc<dyn SelectionStrategy>) -> Self {
        self.selector = Some(selector);
        self
    }

    /// Sets the evaluator, taking precedence over any configured runner.
    #[must_use]
    pub fn with_evaluator(mut self, evaluator: Arc<dyn Evaluator>) -> Self {
        self.evaluator = Some(evaluator);
        self
    }

    /// Sets the checkpoint backend (default [`InMemoryCheckpointStore`]).
    #[must_use]
    pub fn with_checkpoint_store(mut self, store: Arc<dyn CheckpointStore>) -> Self {
        self.checkpoint_store = Some(store);
        self
    }

    /// Sets the cases every candidate is graded against (empty by default).
    #[must_use]
    pub fn with_cases(mut self, cases: Vec<Case>) -> Self {
        self.cases = cases;
        self
    }

    /// Sets the body of the seed candidate (default [`DEFAULT_SEED_BODY`]).
    #[must_use]
    pub fn with_seed_body(mut self, body: impl Into<String>) -> Self {
        self.seed_body = Some(body.into());
        self
    }

    /// Sets the session id (default a fresh `uuid_v4`).
    ///
    /// The id is the checkpoint key, so a caller that wants to resume across
    /// processes must pass the same id to both builders.
    #[must_use]
    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Sets how many proposals each iteration requests (default `1`).
    #[must_use]
    pub fn with_mutation_count(mut self, count: usize) -> Self {
        self.mutation_count = count;
        self
    }

    /// Validates the configuration and builds the orchestrator.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when the configuration is invalid (the
    /// underlying [`AutoHarnessConfig::validate`] error is attached) or when
    /// neither a runner nor an evaluator was supplied, and
    /// [`Error::Validation`] when `mutation_count` is zero.
    pub fn build(self) -> Result<AutoHarnessOrchestrator> {
        self.config.validate().map_err(|err| {
            Error::config(format!(
                "invalid auto-harness config for session builder: {err}"
            ))
        })?;
        if self.mutation_count == 0 {
            return Err(Error::validation(
                "auto-harness mutation_count must be greater than zero",
            ));
        }

        let session_id = self.session_id.unwrap_or_else(uuid_v4);
        let iteration_clock = Arc::new(AtomicU32::new(0));
        let evaluator = match (self.evaluator, self.runner) {
            (Some(evaluator), _) => evaluator,
            (None, Some(runner)) => Arc::new(SessionIterationEvaluator::new(
                runner,
                session_id.clone(),
                Arc::clone(&iteration_clock),
            )) as Arc<dyn Evaluator>,
            (None, None) => {
                return Err(Error::config(
                    "auto-harness orchestrator needs a HarnessRunner or an Evaluator to grade candidates",
                ))
            }
        };
        let mutator = self
            .mutator
            .unwrap_or_else(|| Arc::new(DeterministicMutator::default()));
        let selector = self
            .selector
            .unwrap_or_else(|| Arc::new(ElitistSelection::default()));
        let checkpoint_store = self
            .checkpoint_store
            .unwrap_or_else(|| Arc::new(InMemoryCheckpointStore::default()));
        let limits = BudgetLimits {
            wall_clock: seconds_to_duration(self.config.session_budget_secs),
            cost_limit_usd: self.config.cost_limit_usd,
            task_timeout: seconds_to_duration(self.config.task_timeout_secs),
        };

        Ok(AutoHarnessOrchestrator::assemble(OrchestratorParts {
            config: self.config,
            session_id,
            seed_body: self
                .seed_body
                .unwrap_or_else(|| DEFAULT_SEED_BODY.to_owned()),
            mutation_count: self.mutation_count,
            cases: self.cases,
            mutator,
            selector,
            evaluator,
            checkpoint_store,
            limits,
            iteration_clock,
        }))
    }
}

/// Converts a configured number of seconds into a [`Duration`], defensively.
///
/// [`AutoHarnessConfig::validate`] rejects non-positive and non-finite values, and
/// the cap keeps the conversion inside [`Duration`]'s range, so this never panics.
fn seconds_to_duration(seconds: f64) -> Duration {
    if seconds.is_finite() && seconds > 0.0 {
        Duration::from_secs_f64(seconds.min(MAX_BUDGET_SECS))
    } else {
        Duration::ZERO
    }
}

/// Adapts a [`HarnessRunner`] to the [`Evaluator`] protocol, stamping the
/// orchestrator's session id and the iteration currently running.
///
/// The shared protocol carries only the candidate and the cases, so the iteration
/// is read from a counter the orchestrator advances at the start of every
/// iteration. Delegating to [`HarnessEvaluator`] keeps the request construction in
/// exactly one place.
struct SessionIterationEvaluator {
    /// The wrapped runner.
    runner: Arc<dyn HarnessRunner>,
    /// Session id stamped on every request.
    session_id: String,
    /// Iteration currently running, advanced by the orchestrator.
    iteration: Arc<AtomicU32>,
}

impl SessionIterationEvaluator {
    /// Wraps `runner`, stamping `session_id` and reading `iteration`.
    fn new(runner: Arc<dyn HarnessRunner>, session_id: String, iteration: Arc<AtomicU32>) -> Self {
        Self {
            runner,
            session_id,
            iteration,
        }
    }
}

#[async_trait]
impl Evaluator for SessionIterationEvaluator {
    fn name(&self) -> &str {
        HarnessEvaluator::NAME
    }

    async fn evaluate(&self, candidate: &Candidate, cases: &[Case]) -> Result<EvaluationReport> {
        let iteration = self.iteration.load(Ordering::Relaxed);
        HarnessEvaluator::new(Arc::clone(&self.runner))
            .with_session_id(self.session_id.clone())
            .with_iteration(iteration)
            .evaluate(candidate, cases)
            .await
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Mutex;

    use async_trait::async_trait;
    use serde_json::Value;

    use super::*;
    use crate::evolution::{CandidateStatus, CaseResult, MutationRequest};

    /// An evaluator that scores every candidate with a fixed value.
    struct FixedEvaluator {
        /// Score reported for every candidate.
        score: f64,
    }

    #[async_trait]
    impl Evaluator for FixedEvaluator {
        fn name(&self) -> &'static str {
            "fixed"
        }

        async fn evaluate(
            &self,
            _candidate: &Candidate,
            _cases: &[Case],
        ) -> Result<EvaluationReport> {
            Ok(EvaluationReport::from_results(vec![CaseResult {
                case_id: "fixed".to_owned(),
                passed: true,
                score: Some(self.score),
                weight: 1.0,
                error: None,
                details: Value::Null,
            }]))
        }
    }

    /// A mutator that records the requested proposal count.
    struct RecordingMutator {
        /// Counts of every request, in order.
        counts: Mutex<Vec<usize>>,
    }

    #[async_trait]
    impl MutationStrategy for RecordingMutator {
        fn name(&self) -> &'static str {
            "recording"
        }

        async fn propose(&self, request: MutationRequest) -> Result<Vec<Candidate>> {
            self.counts.lock().expect("counts lock").push(request.count);
            Ok(Vec::new())
        }
    }

    fn config() -> AutoHarnessConfig {
        AutoHarnessConfig {
            data_dir: PathBuf::from("rsi-builder-tests"),
            ..AutoHarnessConfig::default()
        }
    }

    #[tokio::test]
    async fn defaults_are_offline_and_deterministic() {
        let orchestrator = AutoHarnessOrchestrator::builder(config())
            .with_evaluator(Arc::new(FixedEvaluator { score: 0.5 }))
            .build()
            .expect("orchestrator builds");

        assert!(!orchestrator.session_id().is_empty(), "a uuid is generated");
        assert_eq!(orchestrator.state().session_id, orchestrator.session_id());
        assert_eq!(orchestrator.state().candidates.len(), 0);
        assert!(!orchestrator.is_running());

        let record = orchestrator.run_iteration().await.expect("iteration");
        assert_eq!(record.proposed, 1);
        assert_eq!(record.evaluated, 1);
        let state = orchestrator.state();
        assert_eq!(state.candidates[0].body, DEFAULT_SEED_BODY);
        assert_eq!(
            state.candidates[1].status,
            CandidateStatus::Selected,
            "the only graded candidate becomes the elite"
        );
        assert_eq!(state.best_score, Some(0.5));
    }

    #[test]
    fn explicit_session_is_kept() {
        let orchestrator = AutoHarnessOrchestrator::builder(config())
            .with_evaluator(Arc::new(FixedEvaluator { score: 0.1 }))
            .with_session_id("sess-42")
            .build()
            .expect("orchestrator builds");
        assert_eq!(orchestrator.session_id(), "sess-42");
        assert_eq!(orchestrator.state().session_id, "sess-42");
        assert_eq!(
            orchestrator.config().data_dir,
            PathBuf::from("rsi-builder-tests")
        );
    }

    #[test]
    fn budget_limits_come_from_the_config() {
        let config = AutoHarnessConfig {
            data_dir: PathBuf::from("rsi-builder-tests"),
            session_budget_secs: 12.0,
            cost_limit_usd: 2.5,
            task_timeout_secs: 6.0,
            ..AutoHarnessConfig::default()
        };
        let orchestrator = AutoHarnessOrchestrator::builder(config)
            .with_evaluator(Arc::new(FixedEvaluator { score: 0.0 }))
            .build()
            .expect("orchestrator builds");
        let limits = orchestrator.limits();
        assert_eq!(limits.wall_clock, Duration::from_secs(12));
        assert!((limits.cost_limit_usd - 2.5).abs() < f64::EPSILON);
        assert_eq!(limits.task_timeout, Duration::from_secs(6));
    }

    #[test]
    fn seconds_conversion_is_defensive() {
        assert_eq!(seconds_to_duration(f64::NAN), Duration::ZERO);
        assert_eq!(seconds_to_duration(-1.0), Duration::ZERO);
        assert_eq!(seconds_to_duration(0.0), Duration::ZERO);
        assert_eq!(seconds_to_duration(1.5), Duration::from_secs_f64(1.5));
        assert_eq!(
            seconds_to_duration(f64::MAX),
            Duration::from_secs_f64(MAX_BUDGET_SECS)
        );
    }

    #[test]
    fn cases_and_mutation_count_are_forwarded() {
        let cases = vec![Case {
            id: "case-1".to_owned(),
            ..Case::default()
        }];
        let orchestrator = AutoHarnessOrchestrator::builder(config())
            .with_evaluator(Arc::new(FixedEvaluator { score: 0.2 }))
            .with_mutation_count(3)
            .with_cases(cases.clone())
            .build()
            .expect("orchestrator builds");
        assert_eq!(orchestrator.cases(), cases.as_slice());
        assert_eq!(orchestrator.mutation_count(), 3);

        let default = AutoHarnessOrchestrator::builder(self::config())
            .with_evaluator(Arc::new(FixedEvaluator { score: 0.0 }))
            .build()
            .expect("orchestrator builds");
        assert_eq!(default.mutation_count(), 1);
        assert_eq!(default.cases(), []);
    }

    #[tokio::test]
    async fn mutation_count_reaches_the_strategy() {
        let mutator = Arc::new(RecordingMutator {
            counts: Mutex::new(Vec::new()),
        });
        let orchestrator = AutoHarnessOrchestrator::builder(config())
            .with_evaluator(Arc::new(FixedEvaluator { score: 0.0 }))
            .with_mutator(mutator.clone())
            .with_mutation_count(4)
            .build()
            .expect("orchestrator builds");

        orchestrator.run_iteration().await.expect("iteration");
        assert_eq!(*mutator.counts.lock().expect("counts lock"), vec![4]);
    }
}
