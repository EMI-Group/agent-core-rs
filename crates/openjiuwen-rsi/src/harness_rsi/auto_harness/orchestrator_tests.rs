//! Unit tests for [`AutoHarnessOrchestrator`](super::AutoHarnessOrchestrator).
//!
//! A child module of `orchestrator`, kept in its own file so the session loop
//! stays within the crate's file-size budget; `use super::*` still reaches every
//! private helper.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Mutex;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::Value;

use super::*;
use crate::evolution::{CaseResult, EvaluationReport};
use crate::harness_rsi::auto_harness::checkpoint::{InMemoryCheckpointStore, KvCheckpointStore};
use crate::harness_rsi::auto_harness::runner::{
    HarnessRunOutcome, HarnessRunRequest, HarnessRunner,
};
use crate::harness_rsi::auto_harness::task::StopReason;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_core::foundation::store::kv::{InMemoryKvStore, KvStore};

/// A runner that answers with queued outcomes and records every request.
struct ScriptedRunner {
    name: &'static str,
    calls: Mutex<Vec<HarnessRunRequest>>,
    outcomes: Mutex<VecDeque<Result<HarnessRunOutcome>>>,
}

impl ScriptedRunner {
    fn new(outcomes: Vec<Result<HarnessRunOutcome>>) -> Arc<Self> {
        Arc::new(Self {
            name: "scripted",
            calls: Mutex::new(Vec::new()),
            outcomes: Mutex::new(outcomes.into()),
        })
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
        artifacts: BTreeMap::new(),
        messages: vec![format!("scored {score}")],
    }
}

fn config() -> AutoHarnessConfig {
    AutoHarnessConfig {
        data_dir: PathBuf::from("rsi-auto-harness-tests"),
        cost_limit_usd: 100.0,
        session_budget_secs: 3_600.0,
        ..AutoHarnessConfig::default()
    }
}

fn orchestrator_with(
    config: AutoHarnessConfig,
    outcomes: Vec<Result<HarnessRunOutcome>>,
) -> (Arc<AutoHarnessOrchestrator>, Arc<ScriptedRunner>) {
    let runner = ScriptedRunner::new(outcomes);
    let orchestrator = AutoHarnessOrchestrator::builder(config)
        .with_runner(runner.clone())
        .with_session_id("sess-1")
        .build()
        .expect("orchestrator builds");
    (Arc::new(orchestrator), runner)
}

#[test]
fn builder_requires_a_runner_or_evaluator() {
    let error = AutoHarnessOrchestrator::builder(config())
        .build()
        .expect_err("a session without a runner or evaluator is rejected");
    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(error.to_string().contains("HarnessRunner or an Evaluator"));
}

#[test]
fn builder_rejects_an_invalid_config() {
    let error = AutoHarnessOrchestrator::builder(AutoHarnessConfig::default())
        .with_runner(ScriptedRunner::new(Vec::new()))
        .build()
        .expect_err("the default config has no data_dir");
    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(error.to_string().contains("data_dir"));
}

#[tokio::test]
async fn builder_rejects_a_zero_mutation_count() {
    let error = AutoHarnessOrchestrator::builder(config())
        .with_runner(ScriptedRunner::new(Vec::new()))
        .with_mutation_count(0)
        .build()
        .expect_err("zero proposals is rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[tokio::test]
async fn two_iterations_improve_and_accept() {
    let (orchestrator, runner) = orchestrator_with(
        config(),
        vec![Ok(outcome(0.4, 0.25)), Ok(outcome(0.7, 0.25))],
    );

    let first = orchestrator.run_iteration().await.expect("first iteration");
    assert_eq!(first.iteration, 1);
    assert_eq!(first.parents, vec![CandidateId::new("seed")]);
    assert_eq!(first.proposed, 1);
    assert_eq!(first.evaluated, 1);
    assert_eq!(first.valid, 1);
    assert_eq!(first.best_candidate, Some(CandidateId::new("seed/g1/v0")));
    assert_eq!(first.previous_best_score, None);
    assert!((first.best_score.unwrap() - 0.4).abs() < 1e-12);
    assert!(
        (first.improvement - 0.0).abs() < 1e-12,
        "the first accepted best has no incumbent to improve on"
    );
    assert!(first.accepted);
    assert_eq!(first.stop, None);

    let state = orchestrator.state();
    assert_eq!(state.candidates.len(), 2);
    assert_eq!(state.candidates[0].id, CandidateId::new("seed"));
    assert_eq!(state.best_candidate, Some(CandidateId::new("seed/g1/v0")));
    assert!((state.spent_usd - 0.25).abs() < 1e-12);
    assert!(orchestrator.is_running());

    let second = orchestrator
        .run_iteration()
        .await
        .expect("second iteration");
    assert_eq!(second.iteration, 2);
    assert_eq!(
        second.parents,
        vec![CandidateId::new("seed/g1/v0")],
        "the elite becomes the next parent"
    );
    assert_eq!(
        second.best_candidate,
        Some(CandidateId::new("seed/g1/v0/g2/v0"))
    );
    assert!((second.previous_best_score.unwrap() - 0.4).abs() < 1e-12);
    assert!((second.improvement - 0.3).abs() < 1e-12);
    assert!(second.accepted);

    let state = orchestrator.state();
    assert_eq!(state.candidates.len(), 3);
    let winner = state
        .candidates
        .iter()
        .find(|candidate| candidate.id == CandidateId::new("seed/g1/v0/g2/v0"))
        .expect("winner recorded");
    assert_eq!(winner.status, CandidateStatus::Selected);
    assert!((state.best_score.unwrap() - 0.7).abs() < 1e-12);
    assert!((state.spent_usd - 0.5).abs() < 1e-12);

    let calls = runner.calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].session_id, "sess-1");
    assert_eq!(calls[0].iteration, 1);
    assert_eq!(calls[1].iteration, 2);
}

#[tokio::test]
async fn seed_candidate_is_created_from_the_configured_body() {
    let runner = ScriptedRunner::new(vec![Ok(outcome(0.5, 0.0))]);
    let orchestrator = Arc::new(
        AutoHarnessOrchestrator::builder(config())
            .with_runner(runner)
            .with_session_id("sess-1")
            .with_seed_body("model: custom\n")
            .build()
            .expect("orchestrator builds"),
    );

    orchestrator.run_iteration().await.expect("iteration");
    let state = orchestrator.state();
    assert_eq!(state.candidates[0].id, CandidateId::new(SEED_CANDIDATE_ID));
    assert_eq!(state.candidates[0].body, "model: custom\n");
    assert_eq!(state.candidates[0].status, CandidateStatus::Pending);
}

#[tokio::test]
async fn default_seed_body_is_documented_and_pending() {
    let (orchestrator, _) = orchestrator_with(config(), vec![Ok(outcome(0.5, 0.0))]);
    orchestrator.run_iteration().await.expect("iteration");
    let state = orchestrator.state();
    assert_eq!(state.candidates[0].body, DEFAULT_SEED_BODY);
    assert_ne!(DEFAULT_SEED_BODY, "");
}

#[tokio::test]
async fn target_score_stops_the_session() {
    let mut config = config();
    config.target_score = Some(0.5);
    let (orchestrator, _) =
        orchestrator_with(config, vec![Ok(outcome(0.8, 0.0)), Ok(outcome(0.9, 0.0))]);

    let record = orchestrator.run_iteration().await.expect("iteration");
    assert_eq!(record.stop, Some(StopReason::TargetScore));
    let state = orchestrator.state();
    assert_eq!(state.stop_reason, Some(StopReason::TargetScore));
    assert!(!state.running);
    assert!(!orchestrator.is_running());

    let error = orchestrator
        .run_iteration()
        .await
        .expect_err("a finished session refuses more iterations");
    assert_eq!(error.kind(), ErrorKind::Cancelled);
}

#[tokio::test]
async fn max_iterations_stops_the_session() {
    let mut config = config();
    config.max_iterations = 2;
    let (orchestrator, _) =
        orchestrator_with(config, vec![Ok(outcome(0.1, 0.0)), Ok(outcome(0.5, 0.0))]);

    let first = orchestrator.run_iteration().await.expect("first iteration");
    assert_eq!(first.stop, None);
    let second = orchestrator
        .run_iteration()
        .await
        .expect("second iteration");
    assert_eq!(second.stop, Some(StopReason::MaxIterations));
    assert_eq!(
        orchestrator.state().stop_reason,
        Some(StopReason::MaxIterations)
    );
    assert_eq!(
        orchestrator
            .run_iteration()
            .await
            .expect_err("stopped")
            .kind(),
        ErrorKind::Cancelled
    );
}

#[tokio::test]
async fn budget_exhaustion_stops_the_session() {
    let mut config = config();
    config.cost_limit_usd = 0.25;
    let (orchestrator, _) = orchestrator_with(config, vec![Ok(outcome(0.5, 1.0))]);

    let record = orchestrator.run_iteration().await.expect("iteration");
    assert_eq!(record.stop, Some(StopReason::BudgetExhausted));
    assert_eq!(
        orchestrator.state().stop_reason,
        Some(StopReason::BudgetExhausted)
    );
    assert!((orchestrator.state().spent_usd - 1.0).abs() < 1e-12);
}

#[tokio::test]
async fn convergence_stops_after_flat_iterations() {
    let mut config = config();
    config.patience = 2;
    let (orchestrator, _) =
        orchestrator_with(config, vec![Ok(outcome(0.5, 0.0)), Ok(outcome(0.5, 0.0))]);

    assert_eq!(
        orchestrator
            .run_iteration()
            .await
            .expect("first iteration")
            .stop,
        None
    );
    let second = orchestrator
        .run_iteration()
        .await
        .expect("second iteration");
    assert_eq!(second.stop, Some(StopReason::Converged));
    assert!((second.improvement - 0.0).abs() < 1e-12);
    assert!(!second.accepted, "an equal score is not an improvement");
}

#[tokio::test]
async fn stop_cancels_the_session() {
    let (orchestrator, _) = orchestrator_with(config(), vec![Ok(outcome(0.5, 0.0))]);
    orchestrator.start().expect("session starts");
    assert!(orchestrator.is_running());
    orchestrator.stop();
    assert!(orchestrator.is_cancelled());
    let state = orchestrator.state();
    assert_eq!(state.stop_reason, Some(StopReason::Cancelled));
    assert!(!state.running);

    let error = orchestrator
        .run_iteration()
        .await
        .expect_err("a cancelled session refuses iterations");
    assert_eq!(error.kind(), ErrorKind::Cancelled);
}

#[tokio::test]
async fn failing_evaluator_marks_the_candidate_invalid_and_continues() {
    let (orchestrator, runner) = orchestrator_with(
        config(),
        vec![Err(Error::llm("harness exploded")), Ok(outcome(0.9, 0.0))],
    );

    let first = orchestrator
        .run_iteration()
        .await
        .expect("iteration survives");
    assert_eq!(first.proposed, 1);
    assert_eq!(first.evaluated, 0);
    assert_eq!(first.valid, 0);
    assert_eq!(first.best_candidate, None);
    assert_eq!(first.improvement, 0.0);
    assert!(!first.accepted);
    assert_eq!(first.stop, None);

    let state = orchestrator.state();
    let invalid = state
        .candidates
        .iter()
        .find(|candidate| candidate.id == CandidateId::new("seed/g1/v0"))
        .expect("failed candidate is recorded");
    assert_eq!(invalid.status, CandidateStatus::Invalid);
    assert!(invalid.score().is_none());
    assert!(invalid.metadata["error"]
        .as_str()
        .expect("error message")
        .contains("harness exploded"));

    let second = orchestrator
        .run_iteration()
        .await
        .expect("iteration continues");
    assert!(second.accepted);
    assert!((second.best_score.unwrap() - 0.9).abs() < 1e-12);
    assert_eq!(runner.calls().len(), 2);
}

#[tokio::test]
async fn duplicate_proposals_are_not_evaluated_twice() {
    let runner = ScriptedRunner::new(Vec::new());
    let orchestrator = Arc::new(
        AutoHarnessOrchestrator::builder(config())
            .with_runner(runner.clone())
            .with_session_id("sess-1")
            .with_mutator(Arc::new(RepeatingMutator))
            .build()
            .expect("orchestrator builds"),
    );

    let record = orchestrator.run_iteration().await.expect("iteration");
    assert_eq!(record.proposed, 2, "both proposals are counted");
    assert_eq!(record.evaluated, 0, "the seed is never re-evaluated");
    assert_eq!(runner.calls().len(), 0);
    assert_eq!(orchestrator.state().candidates.len(), 1);
}

#[tokio::test]
async fn checkpoint_is_persisted_after_each_iteration() {
    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    let runner = ScriptedRunner::new(vec![Ok(outcome(0.6, 0.0))]);
    let orchestrator = Arc::new(
        AutoHarnessOrchestrator::builder(config())
            .with_runner(runner)
            .with_session_id("sess-1")
            .with_checkpoint_store(Arc::clone(&store))
            .build()
            .expect("orchestrator builds"),
    );

    orchestrator.run_iteration().await.expect("iteration");
    let persisted = store
        .load("sess-1")
        .await
        .expect("checkpoint readable")
        .expect("checkpoint written");
    assert_eq!(persisted.session_id, "sess-1");
    assert_eq!(persisted.iteration, 1);
    assert_eq!(persisted.candidates.len(), 2);
    assert_eq!(persisted.history.len(), 1);
    assert_eq!(
        persisted.best_candidate,
        Some(CandidateId::new("seed/g1/v0"))
    );
}

#[tokio::test]
async fn kv_checkpoint_store_persists_json_under_the_prefix() {
    let kv = Arc::new(InMemoryKvStore::default());
    let store = Arc::new(KvCheckpointStore::new(kv.clone()));
    let runner = ScriptedRunner::new(vec![Ok(outcome(0.6, 0.0))]);
    let orchestrator = Arc::new(
        AutoHarnessOrchestrator::builder(config())
            .with_runner(runner)
            .with_session_id("sess-kv")
            .with_checkpoint_store(store.clone())
            .build()
            .expect("orchestrator builds"),
    );

    orchestrator.run_iteration().await.expect("iteration");
    let key = store.key("sess-kv");
    let stored = kv.get(&key).await.expect("kv read").expect("kv value");
    let text = stored.as_str().expect("checkpoint is JSON text");
    assert!(text.contains("\"iteration\":1"));
    let loaded = store
        .load("sess-kv")
        .await
        .expect("checkpoint readable")
        .expect("checkpoint written");
    assert_eq!(loaded.iteration, 1);
}

#[tokio::test]
async fn resume_continues_at_the_persisted_iteration() {
    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    let first_runner = ScriptedRunner::new(vec![Ok(outcome(0.6, 0.25))]);
    let first = Arc::new(
        AutoHarnessOrchestrator::builder(config())
            .with_runner(first_runner)
            .with_session_id("sess-resume")
            .with_checkpoint_store(Arc::clone(&store))
            .build()
            .expect("orchestrator builds"),
    );
    first.run_iteration().await.expect("first iteration");

    let second_runner = ScriptedRunner::new(vec![Ok(outcome(0.9, 0.25))]);
    let second = Arc::new(
        AutoHarnessOrchestrator::builder(config())
            .with_runner(second_runner.clone())
            .with_session_id("sess-resume")
            .with_checkpoint_store(Arc::clone(&store))
            .build()
            .expect("orchestrator builds"),
    );

    let restored = second
        .resume()
        .await
        .expect("resume succeeds")
        .expect("checkpoint exists");
    assert_eq!(restored.iteration, 1);
    assert_eq!(restored.candidates.len(), 2);
    assert!(second.is_running());
    assert!((second.state().spent_usd - 0.25).abs() < 1e-12);

    let record = second.run_iteration().await.expect("resumed iteration");
    assert_eq!(record.iteration, 2);
    assert_eq!(record.parents, vec![CandidateId::new("seed/g1/v0")]);
    assert_eq!(
        record
            .previous_best_score
            .map(|score| (score * 1e12).round() / 1e12),
        Some(0.6)
    );
    assert!((second.state().spent_usd - 0.5).abs() < 1e-12);
}

#[tokio::test]
async fn resume_without_a_checkpoint_leaves_the_session_untouched() {
    let (orchestrator, _) = orchestrator_with(config(), Vec::new());
    assert_eq!(orchestrator.resume().await.expect("resume succeeds"), None);
    assert!(!orchestrator.is_running());
    assert_eq!(orchestrator.state().iteration, 0);
}

#[tokio::test]
async fn stream_emits_events_in_order() {
    let mut config = config();
    config.max_iterations = 2;
    let (orchestrator, _) =
        orchestrator_with(config, vec![Ok(outcome(0.2, 0.0)), Ok(outcome(0.6, 0.0))]);

    let results: Vec<Result<AutoHarnessEvent>> = orchestrator
        .stream()
        .collect::<Vec<Result<AutoHarnessEvent>>>()
        .await;
    let events = results
        .into_iter()
        .collect::<Result<Vec<_>>>()
        .expect("stream completes");

    let kinds: Vec<&str> = events.iter().map(AutoHarnessEvent::kind).collect();
    assert_eq!(
        kinds,
        [
            "session_started",
            "iteration_started",
            "candidates_evaluated",
            "iteration_finished",
            "checkpoint_saved",
            "iteration_started",
            "candidates_evaluated",
            "iteration_finished",
            "checkpoint_saved",
            "harness_session_finished",
        ]
    );
    assert!(matches!(
        events.first(),
        Some(AutoHarnessEvent::SessionStarted {
            max_iterations: 2,
            ..
        })
    ));
    let finished = events
        .iter()
        .filter(|event| matches!(event, AutoHarnessEvent::SessionFinished { .. }))
        .count();
    assert_eq!(finished, 1, "exactly one terminal event");
    let encoded =
        serde_json::to_value(events.last().expect("last event")).expect("event serializes");
    assert_eq!(
        encoded["type"],
        serde_json::json!("harness_session_finished")
    );
    assert_eq!(encoded["summary"]["iterations"], serde_json::json!(2));
    assert!(!orchestrator.is_running());
}

#[tokio::test]
async fn stream_after_finish_is_cancelled() {
    let (orchestrator, _) = orchestrator_with(config(), vec![Ok(outcome(0.5, 0.0))]);
    orchestrator.run_iteration().await.expect("iteration");
    orchestrator.finish().await.expect("finish");

    let results = orchestrator
        .stream()
        .collect::<Vec<Result<AutoHarnessEvent>>>()
        .await;
    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0].as_ref().expect_err("no events").kind(),
        ErrorKind::Cancelled
    );
}

#[tokio::test]
async fn stream_reports_cancellation() {
    let (orchestrator, _) = orchestrator_with(config(), vec![Ok(outcome(0.5, 0.0))]);
    orchestrator.start().expect("session starts");
    let mut stream = orchestrator.stream();
    let first = stream.next().await.expect("first event").expect("started");
    assert_eq!(first.kind(), "session_started");
    orchestrator.stop();
    let next = stream
        .next()
        .await
        .expect("second event")
        .expect("cancelled");
    assert_eq!(next.kind(), "session_cancelled");
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn finish_summarizes_the_session() {
    let (orchestrator, _) = orchestrator_with(
        config(),
        vec![
            Ok(outcome(0.3, 0.1)),
            Ok(outcome(0.6, 0.2)),
            Ok(outcome(0.1, 0.0)),
        ],
    );
    orchestrator.run_iteration().await.expect("first iteration");
    orchestrator
        .run_iteration()
        .await
        .expect("second iteration");
    orchestrator.run_iteration().await.expect("third iteration");

    let summary = orchestrator.finish().await.expect("finish succeeds");
    assert_eq!(summary.session_id, "sess-1");
    assert_eq!(summary.iterations, 3);
    assert_eq!(summary.evaluated, 3);
    assert_eq!(summary.accepted, 2, "the last iteration regressed");
    assert_eq!(
        summary.best_candidate,
        Some(CandidateId::new("seed/g1/v0/g2/v0"))
    );
    assert!((summary.best_score.unwrap() - 0.6).abs() < 1e-12);
    assert!((summary.spent_usd - 0.3).abs() < 1e-12);
    assert_eq!(summary.stop_reason, Some(StopReason::Failed));
    assert!(!orchestrator.is_running());

    let again = orchestrator.finish().await.expect("finish is idempotent");
    assert_eq!(again, summary);
}

#[tokio::test]
async fn run_drives_the_session_to_the_iteration_cap() {
    let mut config = config();
    config.max_iterations = 3;
    let (orchestrator, runner) = orchestrator_with(
        config,
        vec![
            Ok(outcome(0.1, 0.0)),
            Ok(outcome(0.2, 0.0)),
            Ok(outcome(0.3, 0.0)),
        ],
    );

    let summary = orchestrator.run().await.expect("run completes");
    assert_eq!(summary.iterations, 3);
    assert_eq!(summary.stop_reason, Some(StopReason::MaxIterations));
    assert_eq!(summary.accepted, 3);
    assert_eq!(runner.calls().len(), 3);

    let error = orchestrator
        .run()
        .await
        .expect_err("a finished session cannot run again");
    assert_eq!(error.kind(), ErrorKind::Cancelled);
}

#[test]
fn best_of_breaks_ties_by_candidate_id() {
    let mut low = Candidate::new("b-low", "body");
    low.status = CandidateStatus::Evaluated;
    low.evaluation = Some(report(0.25, 0.0));
    let mut tie_b = Candidate::new("b", "body");
    tie_b.status = CandidateStatus::Evaluated;
    tie_b.evaluation = Some(report(0.75, 0.0));
    let mut tie_a = Candidate::new("a", "body");
    tie_a.status = CandidateStatus::Evaluated;
    tie_a.evaluation = Some(report(0.75, 0.0));
    let ungraded = Candidate::new("c", "body");
    let candidates = vec![low, tie_b, tie_a, ungraded];
    let ids: Vec<CandidateId> = candidates.iter().map(|c| c.id.clone()).collect();

    let best = best_of(&candidates, &ids).expect("a scored candidate wins");
    assert_eq!(best.0, CandidateId::new("a"));
    assert!((best.1 - 0.75).abs() < 1e-12);
    assert_eq!(best_of(&candidates, &[]), None);
}

/// A mutator that always proposes the seed id again.
struct RepeatingMutator;

#[async_trait]
impl MutationStrategy for RepeatingMutator {
    fn name(&self) -> &'static str {
        "repeating"
    }

    async fn propose(&self, request: MutationRequest) -> Result<Vec<Candidate>> {
        let _ = request;
        Ok(vec![
            Candidate::new(SEED_CANDIDATE_ID, "duplicate"),
            Candidate::new(SEED_CANDIDATE_ID, "duplicate again"),
        ])
    }
}
