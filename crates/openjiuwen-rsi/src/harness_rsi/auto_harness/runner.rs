//! The harness seam: one run of one candidate through `openjiuwen-harness`.
//!
//! Python's auto-harness constructs `DeepAgent` pipelines directly
//! (`create_deep_agent`, `DeepAgent`, rails) inside its stages. Rust inverts
//! that dependency: the orchestrator never builds a harness, it hands a
//! [`HarnessRunRequest`] to a [`HarnessRunner`] and receives a
//! [`HarnessRunOutcome`].
//!
//! This trait is **the only seam to `openjiuwen-harness`**. Callers wire a real
//! `DeepAgent` run (or a CLI-driven pipeline) behind it; tests and offline
//! consumers plug in a scripted fake. Because the seam is inverted, this crate
//! stays testable without a model, and the harness crate carries no RSI
//! knowledge.
//!
//! [`HarnessEvaluator`] adapts a runner to the shared
//! [`Evaluator`] protocol, so the orchestrator can
//! score candidates without knowing whether the report came from a real
//! harness, a fake, or a replay.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::Result;

use crate::evolution::{Candidate, Case, EvaluationReport, Evaluator};

/// Everything one harness run needs: which candidate to run and how to grade it.
///
/// `session_id` and `iteration` are carried for tracing and idempotency; they
/// are never interpreted by the runner beyond logging. `metadata` is an open
/// bag for caller annotations (pipeline name, worktree path, …).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct HarnessRunRequest {
    /// Auto-harness session that issued the run; empty when the evaluator is
    /// used outside a session.
    pub session_id: String,
    /// 1-based iteration that proposed the candidate; `0` when outside a session.
    pub iteration: u32,
    /// The candidate configuration or program to run.
    pub candidate: Candidate,
    /// The cases the run is graded against.
    pub cases: Vec<Case>,
    /// Caller annotations, forwarded verbatim to the runner.
    pub metadata: BTreeMap<String, Value>,
}

/// What one harness run produced.
///
/// The [`EvaluationReport`] is the graded outcome: its per-case results are what
/// the orchestrator aggregates, and its `cost_usd` is the cost accumulated by the
/// run, so the caller can charge the session budget without a second channel.
/// `artifacts` and `messages` are diagnostic only.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct HarnessRunOutcome {
    /// Graded report of the run.
    pub report: EvaluationReport,
    /// Named artifacts the run produced (patch, transcript, logs, …).
    pub artifacts: BTreeMap<String, Value>,
    /// Human-readable progress or summary messages.
    pub messages: Vec<String>,
}

/// Executes one candidate through the harness and returns its graded report.
///
/// This is the only seam to `openjiuwen-harness`: implementations construct a
/// `DeepAgent` (or drive a pipeline) and grade the resulting behaviour, while
/// this crate only sees a [`HarnessRunRequest`] and a [`HarnessRunOutcome`].
/// A fake implementation is all an offline test needs.
#[async_trait]
pub trait HarnessRunner: Send + Sync {
    /// Returns the runner's human-readable name (used in logs and artifacts).
    fn name(&self) -> &str;

    /// Runs `request` once and returns its graded outcome.
    ///
    /// # Errors
    ///
    /// Returns an error when the run cannot be executed at all (harness build
    /// failure, timeout, infrastructure problem). The orchestrator turns such an
    /// error into an invalid candidate and continues the session, so a broken
    /// candidate never aborts it.
    async fn run(&self, request: HarnessRunRequest) -> Result<HarnessRunOutcome>;
}

/// Adapts a [`HarnessRunner`] to the shared [`Evaluator`] protocol.
///
/// `evaluate` performs exactly one run per candidate and returns the runner's
/// report unchanged — including `cost_usd`, which is the accumulated cost the
/// orchestrator charges against the session budget.
///
/// The adapter is deliberately session-agnostic: [`HarnessEvaluator::new`]
/// leaves `session_id` empty and `iteration` at `0`, and a caller that wants the
/// orchestrator's session and iteration stamped on every request sets them with
/// [`HarnessEvaluator::with_session_id`] and [`HarnessEvaluator::with_iteration`].
/// The request `metadata` is left empty; callers needing annotations call the
/// runner directly.
pub struct HarnessEvaluator {
    /// The wrapped runner.
    runner: Arc<dyn HarnessRunner>,
    /// Session id stamped on every request; empty by default.
    session_id: String,
    /// Iteration stamped on every request; `0` by default.
    iteration: u32,
}

impl HarnessEvaluator {
    /// Returns the evaluator name reported to the orchestrator and logs.
    pub const NAME: &'static str = "harness";

    /// Wraps `runner` with an empty session id and iteration `0`.
    #[must_use]
    pub fn new(runner: Arc<dyn HarnessRunner>) -> Self {
        Self {
            runner,
            session_id: String::new(),
            iteration: 0,
        }
    }

    /// Returns the wrapped runner.
    #[must_use]
    pub fn runner(&self) -> &Arc<dyn HarnessRunner> {
        &self.runner
    }

    /// Sets the session id stamped on every request.
    #[must_use]
    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = session_id.into();
        self
    }

    /// Sets the iteration stamped on every request.
    #[must_use]
    pub const fn with_iteration(mut self, iteration: u32) -> Self {
        self.iteration = iteration;
        self
    }

    /// Returns the session id stamped on every request (empty when unset).
    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Returns the iteration stamped on every request (`0` when unset).
    #[must_use]
    pub const fn iteration(&self) -> u32 {
        self.iteration
    }

    /// Builds the request for one evaluation.
    fn request(&self, candidate: &Candidate, cases: &[Case]) -> HarnessRunRequest {
        HarnessRunRequest {
            session_id: self.session_id.clone(),
            iteration: self.iteration,
            candidate: candidate.clone(),
            cases: cases.to_vec(),
            metadata: BTreeMap::new(),
        }
    }
}

#[async_trait]
impl Evaluator for HarnessEvaluator {
    fn name(&self) -> &str {
        Self::NAME
    }

    async fn evaluate(&self, candidate: &Candidate, cases: &[Case]) -> Result<EvaluationReport> {
        let outcome = self.runner.run(self.request(candidate, cases)).await?;
        Ok(outcome.report)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Mutex;

    use serde_json::json;

    use super::*;
    use crate::evolution::CaseResult;
    use openjiuwen_common::error::{Error, ErrorKind};

    /// A runner that answers with queued outcomes and records every request.
    struct ScriptedRunner {
        name: &'static str,
        calls: Mutex<Vec<HarnessRunRequest>>,
        outcomes: Mutex<VecDeque<Result<HarnessRunOutcome>>>,
    }

    impl ScriptedRunner {
        fn new(outcomes: Vec<Result<HarnessRunOutcome>>) -> Self {
            Self {
                name: "scripted",
                calls: Mutex::new(Vec::new()),
                outcomes: Mutex::new(outcomes.into()),
            }
        }

        fn calls(&self) -> Vec<HarnessRunRequest> {
            self.calls.lock().expect("calls lock").clone()
        }
    }

    #[async_trait]
    impl HarnessRunner for ScriptedRunner {
        fn name(&self) -> &str {
            self.name
        }

        async fn run(&self, request: HarnessRunRequest) -> Result<HarnessRunOutcome> {
            self.calls.lock().expect("calls lock").push(request);
            self.outcomes
                .lock()
                .expect("outcomes lock")
                .pop_front()
                .unwrap_or_else(|| {
                    Err(Error::external(std::io::Error::other(
                        "scripted runner exhausted",
                    )))
                })
        }
    }

    fn case(id: &str) -> Case {
        Case {
            id: id.to_owned(),
            input: json!({"problem": id}),
            ..Case::default()
        }
    }

    fn report(score: f64, cost: f64) -> EvaluationReport {
        let mut report = EvaluationReport::from_results(vec![CaseResult {
            case_id: "case-1".to_owned(),
            passed: true,
            score: Some(score),
            weight: 1.0,
            error: None,
            details: Value::Null,
        }]);
        report.cost_usd = cost;
        report
    }

    fn outcome(score: f64, cost: f64) -> HarnessRunOutcome {
        HarnessRunOutcome {
            report: report(score, cost),
            artifacts: BTreeMap::from([("patch".to_owned(), json!("--- a\n+++ b"))]),
            messages: vec!["run finished".to_owned()],
        }
    }

    #[tokio::test]
    async fn report_passes_through_unchanged() {
        let runner = Arc::new(ScriptedRunner::new(vec![Ok(outcome(0.625, 0.125))]));
        let evaluator = HarnessEvaluator::new(runner);
        let candidate = Candidate::new("seed", "model: demo");

        let evaluated = evaluator
            .evaluate(&candidate, &[case("case-1")])
            .await
            .expect("evaluation succeeds");

        assert_eq!(evaluated, report(0.625, 0.125));
        assert_eq!(evaluated.score(), Some(0.625));
        assert!((evaluated.cost_usd - 0.125).abs() < 1e-12);
        assert_eq!(evaluator.name(), "harness");
        assert_eq!(HarnessEvaluator::NAME, "harness");
    }

    #[test]
    fn new_defaults_session_and_iteration() {
        let runner: Arc<dyn HarnessRunner> = Arc::new(ScriptedRunner::new(Vec::new()));
        let evaluator = HarnessEvaluator::new(runner.clone());

        assert_eq!(evaluator.session_id(), "");
        assert_eq!(evaluator.iteration(), 0);
        assert!(Arc::ptr_eq(evaluator.runner(), &runner));

        let configured = HarnessEvaluator::new(runner)
            .with_session_id("sess-7")
            .with_iteration(3);
        assert_eq!(configured.session_id(), "sess-7");
        assert_eq!(configured.iteration(), 3);
    }

    #[tokio::test]
    async fn request_carries_candidate_cases_session_and_iteration() {
        let runner = Arc::new(ScriptedRunner::new(vec![
            Ok(outcome(0.5, 0.0)),
            Ok(outcome(0.5, 0.0)),
        ]));
        let evaluator = HarnessEvaluator::new(runner.clone())
            .with_session_id("sess-7")
            .with_iteration(3);
        let candidate = Candidate::new("seed/g3/v0", "model: demo");
        let cases = vec![case("case-1"), case("case-2")];

        evaluator
            .evaluate(&candidate, &cases)
            .await
            .expect("evaluation succeeds");

        let calls = runner.calls();
        assert_eq!(calls.len(), 1);
        let request = &calls[0];
        assert_eq!(request.session_id, "sess-7");
        assert_eq!(request.iteration, 3);
        assert_eq!(request.candidate, candidate);
        assert_eq!(request.cases, cases);
        assert!(request.metadata.is_empty());
    }

    #[tokio::test]
    async fn default_request_uses_an_empty_session_and_iteration_zero() {
        let runner = Arc::new(ScriptedRunner::new(vec![Ok(outcome(0.5, 0.0))]));
        let evaluator = HarnessEvaluator::new(runner.clone());

        evaluator
            .evaluate(&Candidate::new("seed", "body"), &[])
            .await
            .expect("evaluation succeeds");

        let calls = runner.calls();
        assert_eq!(calls[0].session_id, "");
        assert_eq!(calls[0].iteration, 0);
        assert!(calls[0].cases.is_empty(), "no cases requested");
    }

    #[tokio::test]
    async fn runner_errors_propagate_unchanged() {
        let runner = Arc::new(ScriptedRunner::new(vec![Err(Error::llm("boom"))]));
        let evaluator = HarnessEvaluator::new(runner);

        let error = evaluator
            .evaluate(&Candidate::new("seed", "body"), &[case("case-1")])
            .await
            .expect_err("runner failure surfaces");

        assert_eq!(error.kind(), ErrorKind::Llm);
        assert!(error.to_string().contains("boom"));
    }

    #[tokio::test]
    async fn one_evaluation_per_candidate() {
        let runner = Arc::new(ScriptedRunner::new(vec![
            Ok(outcome(0.25, 0.0)),
            Ok(outcome(0.75, 0.0)),
        ]));
        let evaluator = HarnessEvaluator::new(runner.clone());

        let first = evaluator
            .evaluate(&Candidate::new("a", "body"), &[])
            .await
            .expect("first evaluation succeeds");
        let second = evaluator
            .evaluate(&Candidate::new("b", "body"), &[])
            .await
            .expect("second evaluation succeeds");

        assert_eq!(first.score(), Some(0.25));
        assert_eq!(second.score(), Some(0.75));
        assert_eq!(runner.calls().len(), 2);
    }

    #[test]
    fn run_types_round_trip_through_json() {
        let request = HarnessRunRequest {
            session_id: "sess-1".to_owned(),
            iteration: 2,
            candidate: Candidate::new("seed", "model: demo"),
            cases: vec![case("case-1")],
            metadata: BTreeMap::from([("pipeline".to_owned(), json!("meta"))]),
        };
        let encoded = serde_json::to_string(&request).expect("request serializes");
        let decoded: HarnessRunRequest = serde_json::from_str(&encoded).expect("request parses");
        assert_eq!(decoded, request);

        let run = outcome(0.5, 0.25);
        let encoded = serde_json::to_string(&run).expect("outcome serializes");
        let decoded: HarnessRunOutcome = serde_json::from_str(&encoded).expect("outcome parses");
        assert_eq!(decoded, run);
    }
}
