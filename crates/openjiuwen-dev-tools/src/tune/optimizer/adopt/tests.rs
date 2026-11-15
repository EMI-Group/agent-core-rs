//! Cross-module suite: the decision layer, the memory, and the service.
//!
//! The per-file suites cover their own units; this one exercises the seams
//! between them — a measured candidate reaching the policy, the verdict reaching
//! the archive, and the archive reaching the injected [`PromptMemory`] — plus the
//! JSON contract of [`AdoptOutcome`].

use std::collections::BTreeMap;
use std::sync::Arc;

use openjiuwen_common::error::Result;
use serde_json::json;

use crate::tune::prompt_search::{
    ConvergenceDetector, DriftJudge, PromptCandidate, PromptMemory, PromptTaskSpec,
};

use super::support::{evaluated, FixedDriftJudge, RecordingMemory};
use super::*;

/// An adoption task with a base prompt.
fn task() -> PromptTaskSpec {
    PromptTaskSpec::new("answer arithmetic questions").with_base_prompt("Answer.")
}

/// A candidate with a stable identifier.
fn candidate(prompt: &str) -> PromptCandidate {
    PromptCandidate::new(prompt, "seed").with_id("cand-1")
}

#[tokio::test]
async fn the_default_service_adopts_a_first_candidate_and_archives_it() {
    let store = Arc::new(RecordingMemory::new());
    let service = AdoptService::default().with_memory(store.clone());
    let outcome = service
        .decide(
            &task(),
            &candidate("Solve {{question}}."),
            0.6,
            &[1.0, 0.5],
            &[],
            10.0,
        )
        .await
        .unwrap();
    assert!(outcome.adopted());
    assert_eq!(outcome.decision.reason(), &AdoptReason::FirstCandidate);
    assert!((outcome.drift - 0.0).abs() < f64::EPSILON);
    assert!((outcome.correctness - 0.75).abs() < 1e-12);
    assert!(outcome.convergence.is_none());
    let record = outcome.record.as_ref().unwrap();
    assert_eq!(record.record_id, "adopt-cand-1");
    assert_eq!(record.prompt, "Solve {{question}}.");
    assert_eq!(record.objective, "answer arithmetic questions");
    assert_eq!(record.created_at, 10.0);
    assert!(record.applied);
    assert_eq!(record.applied_at, Some(10.0));
    assert!(record.baseline_reward.is_none());
    assert_eq!(record.metrics["reward"], json!(0.6));
    assert_eq!(record.metrics["correctness"], json!(0.75));
    assert_eq!(record.metrics["drift"], json!(0.0));
    assert_eq!(record.metrics["gain"], json!(0.0));
    assert_eq!(record.task_characteristics, task().characteristics());
    assert_eq!(record.metadata["candidate_id"], json!("cand-1"));
    assert_eq!(record.metadata["rationale"], json!("seed"));
    // Archived exactly once, marked applied.
    assert_eq!(store.stored().len(), 1);
    assert!(store.stored()[0].applied);
    assert_eq!(service.memory().adopted().len(), 1);
}

#[tokio::test]
async fn an_improving_candidate_gets_a_baseline_from_previous_best() {
    let store = Arc::new(RecordingMemory::new());
    store
        .store(
            crate::tune::prompt_search::PromptRecord::new(
                "old",
                "Answer.",
                0.4,
                "answer arithmetic questions",
            )
            .with_created_at(1.0),
        )
        .unwrap();
    let service = AdoptService::default().with_memory(store.clone());
    let outcome = service
        .decide(&task(), &candidate("Solve."), 0.8, &[1.0], &[0.4], 20.0)
        .await
        .unwrap();
    assert!(outcome.adopted());
    assert_eq!(
        outcome.decision.reason(),
        &AdoptReason::Improved { gain: 0.4 }
    );
    let record = outcome.record.as_ref().unwrap();
    assert_eq!(record.baseline_reward, Some(0.4));
    assert!((record.gain() - 0.4).abs() < 1e-12);
    assert_eq!(store.stored().len(), 2);
    // The second adopted record now supplies the baseline for the next check.
    assert_eq!(service.memory().previous_best(&task().objective), Some(0.8));
}

#[tokio::test]
async fn a_rejected_candidate_persists_nothing() {
    let store = Arc::new(RecordingMemory::new());
    let service = AdoptService::default().with_memory(store.clone());
    let outcome = service
        .decide(&task(), &candidate("Solve."), 0.5, &[1.0], &[0.9], 5.0)
        .await
        .unwrap();
    assert!(!outcome.adopted());
    assert!(matches!(
        outcome.decision.reason(),
        AdoptReason::NoImprovement { .. }
    ));
    assert!(outcome.record.is_none());
    assert_eq!(store.stored().len(), 0);
    assert_eq!(service.memory().adopted().len(), 0);
    assert_eq!(service.memory().rejected().len(), 0);
}

#[tokio::test]
async fn the_injected_drift_judge_gates_the_decision() {
    let drifting = AdoptService::default().with_drift_judge(Arc::new(FixedDriftJudge::new(0.9)));
    let outcome = drifting
        .decide(&task(), &candidate("Solve."), 1.0, &[1.0], &[0.0], 1.0)
        .await
        .unwrap();
    assert!(!outcome.adopted());
    assert_eq!(
        outcome.decision.reason(),
        &AdoptReason::Drift { deviation: 0.9 }
    );
    assert!((outcome.drift - 0.9).abs() < 1e-12);
    assert!(outcome.record.is_none());

    // The same judge passes with a permissive policy.
    let permissive =
        AdoptService::new(AdoptPolicy::new().with_config(AdoptConfig::new().with_max_drift(0.95)))
            .with_drift_judge(Arc::new(FixedDriftJudge::new(0.9)));
    let outcome = permissive
        .decide(&task(), &candidate("Solve."), 1.0, &[1.0], &[0.0], 1.0)
        .await
        .unwrap();
    assert!(outcome.adopted());
    assert!((outcome.drift - 0.9).abs() < 1e-12);
}

#[tokio::test]
async fn a_max_drift_of_one_skips_the_judge() {
    let service =
        AdoptService::new(AdoptPolicy::new().with_config(AdoptConfig::new().with_max_drift(1.0)))
            .with_drift_judge(Arc::new(FixedDriftJudge::new(0.9)));
    let outcome = service
        .decide(&task(), &candidate("Solve."), 1.0, &[1.0], &[0.0], 1.0)
        .await
        .unwrap();
    assert!(outcome.adopted());
    // The judge would have said 0.9; it was never asked.
    assert_eq!(outcome.drift, 0.0);
    assert_eq!(
        outcome.decision.reason(),
        &AdoptReason::Improved { gain: 1.0 }
    );
}

#[tokio::test]
async fn correctness_is_the_mean_of_the_case_scores() {
    let service = AdoptService::default();
    // Two of three cases pass: 0.667 clears the 0.5 floor.
    let outcome = service
        .decide(&task(), &candidate("p"), 0.5, &[1.0, 1.0, 0.0], &[], 1.0)
        .await
        .unwrap();
    assert!(outcome.adopted());
    assert!((outcome.correctness - 2.0 / 3.0).abs() < 1e-12);
    // A poor average is rejected before the improvement is considered.
    let outcome = service
        .decide(&task(), &candidate("p"), 1.0, &[0.1, 0.2], &[0.0], 1.0)
        .await
        .unwrap();
    assert!(!outcome.adopted());
    assert!(matches!(
        outcome.decision.reason(),
        AdoptReason::InsufficientCorrectness { .. }
    ));
    // No scores at all counts as zero correctness.
    let outcome = service
        .decide(&task(), &candidate("p"), 1.0, &[], &[0.0], 1.0)
        .await
        .unwrap();
    assert_eq!(outcome.correctness, 0.0);
    assert_eq!(
        outcome.decision.reason(),
        &AdoptReason::InsufficientCorrectness { score: 0.0 }
    );
}

#[tokio::test]
async fn convergence_is_reported_only_when_a_detector_is_configured() {
    let detector = ConvergenceDetector::new(0.01, 2, Some(0.9)).unwrap();
    let service = AdoptService::default().with_convergence(detector.clone());
    let outcome = service
        .decide(&task(), &candidate("p"), 1.0, &[1.0], &[0.5, 0.95], 1.0)
        .await
        .unwrap();
    let convergence = outcome.convergence.as_ref().unwrap();
    assert!(convergence.converged);
    assert!(convergence.reason.contains("target reward"));
    // Convergence never changes the verdict.
    assert!(outcome.adopted());
    assert_eq!(service.convergence(), Some(&detector));

    let plain = AdoptService::default();
    let outcome = plain
        .decide(&task(), &candidate("p"), 1.0, &[1.0], &[0.5, 0.95], 1.0)
        .await
        .unwrap();
    assert!(outcome.convergence.is_none());
    assert!(plain.convergence().is_none());
}

#[tokio::test]
async fn the_characteristics_override_reaches_the_record_and_metadata_follows_the_task() {
    let mut metadata = BTreeMap::new();
    metadata.insert("team".to_owned(), json!("search"));
    let task = task().with_metadata(metadata);
    let service = AdoptService::default().with_characteristics("custom characteristics");
    assert_eq!(service.characteristics(), Some("custom characteristics"));
    let outcome = service
        .decide(&task, &candidate("p"), 0.5, &[1.0], &[], 1.0)
        .await
        .unwrap();
    let record = outcome.record.as_ref().unwrap();
    assert_eq!(record.task_characteristics, "custom characteristics");
    assert_eq!(record.metadata["team"], json!("search"));
    assert!(record.metadata.contains_key("candidate_id"));
}

#[tokio::test]
async fn a_failing_store_never_fails_the_decision() {
    let store = Arc::new(RecordingMemory::failing());
    let service = AdoptService::default().with_memory(store.clone());
    let outcome = service
        .decide(&task(), &candidate("p"), 0.5, &[1.0], &[], 3.0)
        .await
        .unwrap();
    assert!(outcome.adopted());
    let record = outcome.record.as_ref().unwrap();
    assert!(record.applied);
    assert_eq!(service.memory().adopted().len(), 1);
    assert_eq!(store.stored().len(), 0);
}

#[tokio::test]
async fn repeated_decisions_are_deterministic_and_share_the_record_id() {
    let service = AdoptService::default();
    let first = service
        .decide(&task(), &candidate("p"), 0.5, &[1.0], &[], 1.0)
        .await
        .unwrap();
    let second = AdoptService::default()
        .decide(&task(), &candidate("p"), 0.5, &[1.0], &[], 1.0)
        .await
        .unwrap();
    assert_eq!(first, second);
    assert_eq!(
        first.record.as_ref().unwrap().record_id,
        "adopt-cand-1".to_owned()
    );
}

#[tokio::test]
async fn service_readers_expose_the_injected_seams() {
    let service = AdoptService::default();
    assert_eq!(service.policy(), &AdoptPolicy::new());
    assert_eq!(service.config(), &AdoptConfig::default());
    assert!(service.characteristics().is_none());
    assert_eq!(service.memory().adopted().len(), 0);
    assert!(format!("{service:?}").contains("AdoptService"));
    // The default judge measures no drift.
    let deviation = service
        .drift_judge()
        .deviation("objective", &candidate("p"))
        .await;
    assert!(deviation.abs() < f64::EPSILON);
}

#[tokio::test]
async fn the_whole_family_composes_over_one_run() {
    // A failing evaluation supplies the bad cases, the gradient renders them, and
    // the service archives the adopted prompt.
    let cases = [
        evaluated("c1", 0.0, "wrong result"),
        evaluated("c2", 1.0, "right"),
        evaluated("c3", 0.5, "partly right"),
    ];
    let bad = select_bad_cases(&cases, DEFAULT_BAD_CASES_SAMPLE_NUM);
    assert_eq!(bad.len(), 1);
    let good = select_good_cases(&cases, DEFAULT_BAD_CASES_SAMPLE_NUM);
    assert_eq!(good.len(), 1);
    let gradient = DeterministicGradientPolicy::new()
        .gradient(&task(), &candidate("p"), &bad)
        .await
        .unwrap();
    assert!(gradient.contains("feedback: wrong result"), "{gradient}");

    let store = Arc::new(RecordingMemory::new());
    let service = AdoptService::default().with_memory(store.clone());
    let scores: Vec<f64> = cases.iter().map(|case| case.score).collect();
    let outcome = service
        .decide(&task(), &candidate("p"), 0.5, &scores, &[0.2], 4.0)
        .await
        .unwrap();
    assert!(outcome.adopted());
    assert_eq!(store.stored().len(), 1);
}

#[test]
fn outcome_round_trips_through_json() {
    let adopted = AdoptOutcome {
        decision: AdoptDecision::new(true, AdoptReason::Improved { gain: 0.25 }),
        drift: 0.1,
        correctness: 0.75,
        convergence: ConvergenceDetector::new(0.01, 2, Some(0.5))
            .unwrap()
            .check(&[0.4, 0.6]),
        record: Some(
            crate::tune::prompt_search::PromptRecord::new("r1", "prompt", 0.6, "objective")
                .with_applied(true, Some(2.0)),
        ),
    };
    assert!(adopted.adopted());
    let value = serde_json::to_value(&adopted).unwrap();
    for key in ["decision", "drift", "correctness", "convergence", "record"] {
        assert!(value.get(key).is_some(), "missing field {key}");
    }
    assert_eq!(value["decision"]["adopted"], json!(true));
    assert_eq!(value["decision"]["reason"]["kind"], json!("improved"));
    let restored: AdoptOutcome = serde_json::from_value(value).unwrap();
    assert_eq!(restored, adopted);

    let rejected = AdoptOutcome {
        decision: AdoptDecision::new(false, AdoptReason::Drift { deviation: 0.8 }),
        drift: 0.8,
        correctness: 1.0,
        convergence: None,
        record: None,
    };
    let restored: AdoptOutcome =
        serde_json::from_value(serde_json::to_value(&rejected).unwrap()).unwrap();
    assert_eq!(restored, rejected);
    // Missing fields default rather than failing.
    let defaulted: AdoptOutcome = serde_json::from_str("{}").unwrap();
    assert_eq!(defaulted, AdoptOutcome::default());
    assert!(!defaulted.adopted());
}

/// A store that is both a `PromptMemory` and records the objectives it was asked
/// about, to prove the drift judge sees the task objective.
struct ObjectiveRecorder {
    /// The objectives the judge was asked to measure against.
    seen: std::sync::Mutex<Vec<String>>,
}

#[async_trait::async_trait]
impl DriftJudge for ObjectiveRecorder {
    async fn deviation(&self, objective: &str, _candidate: &PromptCandidate) -> f64 {
        self.seen
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(objective.to_owned());
        0.0
    }
}

#[tokio::test]
async fn the_drift_judge_measures_against_the_task_objective() {
    let judge = Arc::new(ObjectiveRecorder {
        seen: std::sync::Mutex::new(Vec::new()),
    });
    let service = AdoptService::default().with_drift_judge(judge.clone());
    service
        .decide(&task(), &candidate("p"), 0.5, &[1.0], &[], 1.0)
        .await
        .unwrap();
    let seen = judge
        .seen
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    assert_eq!(seen, vec!["answer arithmetic questions".to_owned()]);
}

/// Compile-time guard: the seams keep the service usable across tasks and
/// threads, which an `async` caller needs.
fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn the_service_and_its_results_are_send_and_sync() {
    assert_send_sync::<AdoptService>();
    assert_send_sync::<AdoptMemory>();
    assert_send_sync::<AdoptOutcome>();
    assert_send_sync::<AdoptDecision>();
    assert_send_sync::<AdoptConfig>();
    assert_send_sync::<Result<AdoptOutcome>>();
}
