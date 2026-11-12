//! Offline integration tests for the auto-harness orchestrator.
//!
//! They drive the public API exactly like a caller would: a scripted
//! [`HarnessRunner`] stands in for the `DeepAgent` a production session would
//! run, so every assertion is deterministic and no test touches the network, a
//! model, git, or the clock.
//!
//! Behaviours mirror Python `AutoHarnessOrchestrator`
//! (`openjiuwen/rsi/harness_rsi/auto_harness/orchestrator.py`, v0.1.20): one
//! iteration proposes candidates, grades them, accepts a strictly better
//! incumbent, checkpoints, and then evaluates the stop conditions in the order
//! budget → max iterations → target score → convergence.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::Value;

use openjiuwen_rsi::{
    AutoHarnessConfig, AutoHarnessEvent, AutoHarnessOrchestrator, CandidateId, CandidateStatus,
    Case, CaseResult, DeterministicMutator, Error, ErrorKind, EvaluationReport, HarnessRunOutcome,
    HarnessRunRequest, HarnessRunner, InMemoryCheckpointStore, Result, StopReason,
};

/// A runner that answers with queued outcomes and records every request.
///
/// It is the whole harness seam: an outcome is either a graded
/// [`EvaluationReport`] or an error, which the orchestrator must turn into an
/// invalid candidate instead of aborting the session.
struct ScriptedRunner {
    /// Outcomes handed out in call order.
    outcomes: Mutex<VecDeque<Result<HarnessRunOutcome>>>,
    /// Every request the orchestrator issued, in call order.
    calls: Mutex<Vec<HarnessRunRequest>>,
}

impl ScriptedRunner {
    /// Wraps the queued outcomes.
    fn new(outcomes: Vec<Result<HarnessRunOutcome>>) -> Arc<Self> {
        Arc::new(Self {
            outcomes: Mutex::new(outcomes.into()),
            calls: Mutex::new(Vec::new()),
        })
    }

    /// Returns the recorded requests.
    fn requests(&self) -> Vec<HarnessRunRequest> {
        self.calls.lock().expect("calls lock").clone()
    }
}

#[async_trait]
impl HarnessRunner for ScriptedRunner {
    // The trait ties the returned lifetime to `&self`; the fake's name is a
    // literal, so the elision is redundant here.
    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "scripted"
    }

    async fn run(&self, request: HarnessRunRequest) -> Result<HarnessRunOutcome> {
        self.calls.lock().expect("calls lock").push(request);
        self.outcomes
            .lock()
            .expect("outcomes lock")
            .pop_front()
            .expect("scripted outcome")
    }
}

/// Builds a one-case report with `score` and stamps `cost` on it.
fn report(score: f64, cost: f64) -> EvaluationReport {
    let mut report = EvaluationReport::from_results(vec![CaseResult {
        case_id: "case".into(),
        passed: true,
        score: Some(score),
        weight: 1.0,
        error: None,
        details: Value::Null,
    }]);
    report.cost_usd = cost;
    report
}

/// A successful run graded `score` and charged `cost` US dollars.
fn outcome(score: f64, cost: f64) -> HarnessRunOutcome {
    HarnessRunOutcome {
        report: report(score, cost),
        artifacts: std::collections::BTreeMap::new(),
        messages: vec![String::from("scripted run")],
    }
}

/// A run that could not be executed at all.
fn error_outcome() -> Result<HarnessRunOutcome> {
    Err(Error::external(std::io::Error::other("boom")))
}

/// A valid configuration rooted at `data_dir`; no filesystem IO happens.
fn config(data_dir: &str) -> AutoHarnessConfig {
    AutoHarnessConfig {
        data_dir: PathBuf::from(data_dir),
        ..AutoHarnessConfig::default()
    }
}

/// The single case every candidate is graded against.
fn case() -> Case {
    Case {
        id: "case".into(),
        name: Some(String::from("orchestrator case")),
        ..Case::default()
    }
}

/// Builds a session whose mutator honours `mutation_count`.
///
/// [`DeterministicMutator`] derives `variants_per_parent` proposals per parent
/// (one by default) and the orchestrator truncates them to the requested count,
/// so a count of `n` needs `n` variants to actually produce `n` candidates.
fn build_session(
    config: AutoHarnessConfig,
    outcomes: Vec<Result<HarnessRunOutcome>>,
    mutation_count: usize,
) -> (Arc<AutoHarnessOrchestrator>, Arc<ScriptedRunner>) {
    let runner = ScriptedRunner::new(outcomes);
    let orchestrator = AutoHarnessOrchestrator::builder(config)
        .with_runner(runner.clone())
        .with_mutator(Arc::new(
            DeterministicMutator::default().with_variants_per_parent(mutation_count),
        ))
        .with_checkpoint_store(Arc::new(InMemoryCheckpointStore::new()))
        .with_cases(vec![case()])
        .with_seed_body("# test seed\n")
        .with_session_id("rsi-orchestrator-tests")
        .with_mutation_count(mutation_count)
        .build()
        .expect("orchestrator builds");
    (Arc::new(orchestrator), runner)
}

/// Two iterations: the first scored candidate is accepted, then a strictly
/// better score replaces the incumbent while the losers stay `Evaluated`.
#[tokio::test]
async fn two_iterations_improve_and_accept() {
    let (orchestrator, runner) = build_session(
        config("rsi-orchestrator-tests"),
        vec![
            Ok(outcome(0.3, 0.0)),
            Ok(outcome(0.1, 0.0)),
            Ok(outcome(0.9, 0.0)),
            Ok(outcome(0.2, 0.0)),
        ],
        2,
    );

    let first = orchestrator.run_iteration().await.expect("first iteration");
    assert_eq!(first.iteration, 1);
    assert_eq!(first.proposed, 2);
    assert_eq!(first.evaluated, 2);
    assert_eq!(first.valid, 2);
    assert_eq!(first.previous_best_score, None);
    assert!(first.accepted, "the first scored candidate is accepted");
    assert!(
        first.improvement.abs() < 1e-9,
        "with no incumbent there is no improvement to report"
    );
    let first_best = first.best_score.expect("the iteration graded a candidate");
    assert!((first_best - 0.3).abs() < 1e-9);
    assert_eq!(first.stop, None, "nothing stops after one flat iteration");
    // `duration_ms` is the only timing value; assert it exists as a `u64` and
    // never compare it against a wall clock.
    let _: u64 = first.duration_ms;

    let second = orchestrator
        .run_iteration()
        .await
        .expect("second iteration");
    assert_eq!(second.iteration, 2);
    assert_eq!(second.proposed, 2);
    assert_eq!(second.evaluated, 2);
    assert!(second.accepted, "0.9 strictly beats the 0.3 incumbent");
    let previous = second
        .previous_best_score
        .expect("the first iteration set an incumbent");
    assert!((previous - 0.3).abs() < 1e-9);
    assert!((second.improvement - 0.6).abs() < 1e-9);
    let second_best = second.best_score.expect("the iteration graded a candidate");
    assert!((second_best - 0.9).abs() < 1e-9);

    let state = orchestrator.state();
    assert_eq!(state.history.len(), 2, "one record per completed iteration");
    assert_eq!(state.iteration, 2);
    let incumbent = state.best_score.expect("the session has an incumbent");
    assert!((incumbent - 0.9).abs() < 1e-9);
    assert_eq!(
        state.best_candidate.as_ref().map(CandidateId::as_str),
        Some("seed/g1/v0/g2/v0"),
        "the second iteration mutated the first winner"
    );
    let winner = state
        .candidates
        .iter()
        .find(|candidate| Some(&candidate.id) == state.best_candidate.as_ref())
        .expect("the incumbent is one of the candidates");
    assert_eq!(winner.status, CandidateStatus::Selected);
    assert!(state
        .candidates
        .iter()
        .any(|candidate| candidate.status == CandidateStatus::Selected));
    assert!(state
        .candidates
        .iter()
        .any(|candidate| candidate.status == CandidateStatus::Evaluated));

    // Builder wiring reaches the harness: one run per proposal, stamped with the
    // session id, iteration, candidate, and the configured cases.
    let requests = runner.requests();
    assert_eq!(requests.len(), 4);
    assert!(requests
        .iter()
        .all(|request| request.session_id == "rsi-orchestrator-tests"));
    assert_eq!(requests[0].iteration, 1);
    assert_eq!(requests[2].iteration, 2);
    assert_eq!(requests[0].cases, vec![case()]);
    assert!(requests[0].candidate.body.starts_with("# test seed"));
    assert!(requests[0].candidate.body.contains("rsi variant g1-0"));
}

/// Two flat scores trip the default patience of two: the session converges.
#[tokio::test]
async fn flat_scores_converge_after_patience() {
    let (orchestrator, _) = build_session(
        config("rsi-orchestrator-tests"),
        vec![Ok(outcome(0.5, 0.0)), Ok(outcome(0.5, 0.0))],
        1,
    );

    let first = orchestrator.run_iteration().await.expect("first iteration");
    assert_eq!(first.stop, None, "patience needs a second flat iteration");

    let second = orchestrator
        .run_iteration()
        .await
        .expect("second iteration");
    assert!(
        (second.improvement).abs() < 1e-9,
        "the best score did not move"
    );
    assert!(
        !second.accepted,
        "an equal score does not replace the incumbent"
    );
    assert_eq!(second.stop, Some(StopReason::Converged));

    let state = orchestrator.state();
    assert_eq!(state.stop_reason, Some(StopReason::Converged));
    assert!(!state.running);
    assert!(!orchestrator.is_running());
}

/// Reaching `target_score` stops the session after the iteration that hit it.
#[tokio::test]
async fn reaching_the_target_score_stops_the_session() {
    let mut config = config("rsi-orchestrator-tests");
    config.target_score = Some(0.8);
    let (orchestrator, _) = build_session(config, vec![Ok(outcome(0.9, 0.0))], 1);

    let record = orchestrator.run_iteration().await.expect("iteration");
    assert!(record.accepted);
    let best = record.best_score.expect("the iteration graded a candidate");
    assert!((best - 0.9).abs() < 1e-9);
    assert_eq!(record.stop, Some(StopReason::TargetScore));

    let state = orchestrator.state();
    assert_eq!(state.stop_reason, Some(StopReason::TargetScore));
    assert!(!orchestrator.is_running());
}

/// Spending the configured cost limit stops the session with `BudgetExhausted`.
#[tokio::test]
async fn cost_limit_exhaustion_stops_the_session() {
    let mut config = config("rsi-orchestrator-tests");
    config.cost_limit_usd = 1.0;
    config.session_budget_secs = 900_000.0;
    let (orchestrator, _) = build_session(config, vec![Ok(outcome(0.5, 1.0))], 1);

    let record = orchestrator.run_iteration().await.expect("iteration");
    assert_eq!(record.stop, Some(StopReason::BudgetExhausted));

    let state = orchestrator.state();
    assert_eq!(state.stop_reason, Some(StopReason::BudgetExhausted));
    assert!((state.spent_usd - 1.0).abs() < 1e-9);
    assert!(!orchestrator.is_running());
}

/// `stop()` is terminal: the next iteration reports `Cancelled`.
#[tokio::test]
async fn stop_then_run_iteration_returns_cancelled() {
    let (orchestrator, runner) = build_session(config("rsi-orchestrator-tests"), Vec::new(), 1);

    orchestrator.stop();
    assert!(orchestrator.is_cancelled());
    assert!(!orchestrator.is_running());
    assert_eq!(
        orchestrator.state().stop_reason,
        Some(StopReason::Cancelled)
    );

    let error = orchestrator
        .run_iteration()
        .await
        .expect_err("a stopped session refuses iterations");
    assert_eq!(error.kind(), ErrorKind::Cancelled);
    assert!(
        runner.requests().is_empty(),
        "nothing was ever sent to the harness"
    );
}

/// The event stream opens with `session_started`, reports every iteration, and
/// ends with the Python `harness_session_finished` chunk.
#[tokio::test]
async fn stream_orders_events_and_ends_with_finished() {
    let mut config = config("rsi-orchestrator-tests");
    config.target_score = Some(0.8);
    let (orchestrator, _) = build_session(config, vec![Ok(outcome(0.9, 0.0))], 1);

    let events: Vec<AutoHarnessEvent> = orchestrator
        .stream()
        .map(|item| item.expect("stream items are Ok"))
        .collect()
        .await;

    let kinds: Vec<&str> = events.iter().map(AutoHarnessEvent::kind).collect();
    assert_eq!(
        kinds,
        vec![
            "session_started",
            "iteration_started",
            "candidates_evaluated",
            "iteration_finished",
            "checkpoint_saved",
            "harness_session_finished",
        ]
    );
    assert_eq!(events[0].kind(), "session_started");
    assert_eq!(events[1].kind(), "iteration_started");
    let finished = kinds
        .iter()
        .position(|kind| *kind == "iteration_finished")
        .expect("the iteration reports its record");
    assert!(
        finished > 1,
        "iteration_started precedes iteration_finished"
    );
    assert!(
        finished < kinds.len() - 1,
        "the terminal event comes after iteration_finished"
    );
    let last = events.last().expect("the stream has a terminal event");
    assert_eq!(last.kind(), "harness_session_finished");

    let value = serde_json::to_value(last).expect("the terminal event serializes");
    assert_eq!(value["type"], serde_json::json!("harness_session_finished"));
    match last {
        AutoHarnessEvent::SessionFinished { summary } => {
            assert_eq!(summary.stop_reason, Some(StopReason::TargetScore));
            assert_eq!(summary.iterations, 1);
        }
        other => panic!("expected the terminal event, got `{}`", other.kind()),
    }
    assert!(orchestrator.state().is_finished());
}

/// `run()` is the event-free single-call entry point and returns the summary.
#[tokio::test]
async fn run_returns_a_session_summary() {
    let mut config = config("rsi-orchestrator-tests");
    config.max_iterations = 3;
    let (orchestrator, runner) = build_session(
        config,
        vec![
            Ok(outcome(0.2, 0.0)),
            Ok(outcome(0.3, 0.0)),
            Ok(outcome(0.4, 0.0)),
        ],
        1,
    );

    let summary = orchestrator.run().await.expect("run completes the session");
    assert!(summary.iterations >= 1);
    assert_eq!(summary.iterations, 3);
    assert_eq!(summary.stop_reason, Some(StopReason::MaxIterations));
    assert_eq!(summary.evaluated, 3);
    assert_eq!(summary.accepted, 3);
    assert_eq!(runner.requests().len(), 3);
    assert!(
        orchestrator.run().await.is_err(),
        "a finished session cannot run again"
    );
}

/// A failing run marks its candidate `Invalid`; the session keeps going.
#[tokio::test]
async fn evaluation_error_marks_the_candidate_invalid() {
    let (orchestrator, _) =
        build_session(config("rsi-orchestrator-tests"), vec![error_outcome()], 1);

    let record = orchestrator
        .run_iteration()
        .await
        .expect("iteration survives");
    assert_eq!(record.proposed, 1);
    assert_eq!(record.evaluated, 0, "a failed run is not evaluated");
    assert_eq!(record.valid, 0);
    assert_eq!(record.best_score, None);
    assert_eq!(record.best_candidate, None);
    assert!(!record.accepted);
    assert_eq!(record.stop, None, "one invalid candidate is not fatal");

    let state = orchestrator.state();
    let invalid = state
        .candidates
        .iter()
        .find(|candidate| candidate.status == CandidateStatus::Invalid)
        .expect("the failed candidate is recorded as invalid");
    let message = invalid
        .metadata
        .get("error")
        .expect("the failure reason is recorded");
    assert!(
        message.as_str().is_some_and(|text| text.contains("boom")),
        "the recorded error carries the runner failure"
    );
    assert!(
        orchestrator.is_running(),
        "max_iterations and patience are both still open"
    );
}

/// After an invalid candidate the session recovers on the next iteration.
#[tokio::test]
async fn session_continues_after_an_invalid_candidate() {
    let mut config = config("rsi-orchestrator-tests");
    // A long patience keeps convergence out of the way, so recovery is the only
    // thing the second iteration demonstrates.
    config.patience = 5;
    let (orchestrator, _) = build_session(config, vec![error_outcome(), Ok(outcome(0.4, 0.0))], 1);

    let failed = orchestrator.run_iteration().await.expect("first iteration");
    assert_eq!(failed.evaluated, 0);

    let recovered = orchestrator
        .run_iteration()
        .await
        .expect("the session recovers");
    assert_eq!(recovered.iteration, 2);
    assert_eq!(recovered.evaluated, 1);
    assert_eq!(recovered.valid, 1);
    let best = recovered.best_score.expect("the recovered run is graded");
    assert!((best - 0.4).abs() < 1e-9);
    assert!(
        recovered.accepted,
        "the first graded candidate sets the incumbent"
    );
    assert_eq!(recovered.stop, None);

    let state = orchestrator.state();
    assert_eq!(state.history.len(), 2);
    assert!(
        state
            .candidates
            .iter()
            .any(|candidate| candidate.status == CandidateStatus::Invalid),
        "the invalid candidate is kept for the record"
    );
    assert!(orchestrator.is_running());
}
