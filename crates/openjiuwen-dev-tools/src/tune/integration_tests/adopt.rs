//! Scenario 5: the adoption decision flow over a persistent prompt memory.
//!
//! The first candidate becomes the baseline, a worse one is rejected without
//! being archived, a better one is adopted as an improvement, and the same
//! inputs decide identically. The good/bad case split is covered too.

use std::sync::Arc;

use openjiuwen_test_support::TempWorkspace;

use super::support::*;
use crate::tune::*;

#[tokio::test]
async fn adoption_archives_improvements_and_skips_regressions() {
    let workspace = TempWorkspace::new().expect("a temporary workspace");
    let store =
        Arc::new(JsonlPromptMemory::new(workspace.path("memory-a")).expect("a prompt memory"));
    let memory: Arc<dyn PromptMemory> = store.clone();
    let service = adopt_service(memory);
    let task = PromptTaskSpec::new("Answer the question accurately.")
        .with_base_prompt("Answer the question.");

    // Nothing comparable yet: the first candidate becomes the baseline.
    let first = PromptCandidate::new("Answer the question carefully.", "round one");
    let outcome = service
        .decide(&task, &first, 0.6, &[1.0, 0.5], &[], NOW)
        .await
        .expect("a deterministic decision");
    assert!(outcome.adopted());
    assert_eq!(outcome.decision.reason(), &AdoptReason::FirstCandidate);
    assert_close(outcome.correctness, 0.75);
    let archived = outcome
        .record
        .clone()
        .expect("an adopted candidate is archived");
    assert!(archived.applied, "the archived prompt is marked applied");
    assert_eq!(archived.applied_at, Some(NOW));
    assert_eq!(archived.prompt, first.prompt());
    assert_eq!(store.records().len(), 1);
    assert!(
        store.pending(0.0).is_empty(),
        "an applied prompt is not pending"
    );

    // A lower reward than the recorded best is rejected and archived nowhere.
    let worse = PromptCandidate::new("Answer the question briefly.", "round two");
    let rejected = service
        .decide(&task, &worse, 0.5, &[0.5, 0.5], &[0.6], NOW)
        .await
        .expect("a deterministic decision");
    assert!(!rejected.adopted());
    assert!(matches!(
        rejected.decision.reason(),
        AdoptReason::NoImprovement { .. }
    ));
    assert!(rejected.record.is_none(), "a regression is not archived");
    assert_eq!(store.records().len(), 1, "the store is unchanged");

    // A reward above the recorded best wins by the measured gain.
    let better = PromptCandidate::new(
        "Answer the question with one reasoning step.",
        "round three",
    );
    let improved = service
        .decide(&task, &better, 0.9, &[1.0, 1.0], &[0.6], NOW)
        .await
        .expect("a deterministic decision");
    assert!(improved.adopted());
    match improved.decision.reason() {
        AdoptReason::Improved { gain } => assert_close(*gain, 0.3),
        other => panic!("expected an improvement, got {other:?}"),
    }
    assert_eq!(store.records().len(), 2);
    assert_eq!(store.records()[1].prompt, better.prompt());

    // The same three inputs on a fresh service and store decide identically.
    let replay_store =
        Arc::new(JsonlPromptMemory::new(workspace.path("memory-b")).expect("a prompt memory"));
    let replay_memory: Arc<dyn PromptMemory> = replay_store.clone();
    let replay = adopt_service(replay_memory);
    let replay_first = replay
        .decide(&task, &first, 0.6, &[1.0, 0.5], &[], NOW)
        .await
        .expect("a deterministic decision");
    let replay_worse = replay
        .decide(&task, &worse, 0.5, &[0.5, 0.5], &[0.6], NOW)
        .await
        .expect("a deterministic decision");
    let replay_better = replay
        .decide(&task, &better, 0.9, &[1.0, 1.0], &[0.6], NOW)
        .await
        .expect("a deterministic decision");
    assert_eq!(replay_first, outcome);
    assert_eq!(replay_worse, rejected);
    assert_eq!(replay_better, improved);
    assert_eq!(replay_store.records(), store.records());
}

#[test]
fn case_selection_partitions_the_mixed_evaluated_cases() {
    let mixed = vec![
        evaluated("d", 0.0),
        evaluated("a", 1.0),
        evaluated("c", 0.5),
        evaluated("b", 0.0),
    ];

    let bad = select_bad_cases(&mixed, DEFAULT_BAD_CASES_SAMPLE_NUM);
    assert_eq!(
        evaluated_ids(&bad),
        ["b", "d"],
        "exactly the failed cases, ordered by case id"
    );
    let good = select_good_cases(&mixed, DEFAULT_BAD_CASES_SAMPLE_NUM);
    assert_eq!(
        evaluated_ids(&good),
        ["a"],
        "exactly the fully scored cases"
    );
    assert_eq!(
        select_bad_cases(&mixed, DEFAULT_BAD_CASES_SAMPLE_NUM),
        bad,
        "a repeated call partitions identically"
    );

    // The split reads only the exact scores, so the input order cannot matter.
    let reordered = vec![
        evaluated("b", 0.0),
        evaluated("a", 1.0),
        evaluated("d", 0.0),
        evaluated("c", 0.5),
    ];
    assert_eq!(
        select_bad_cases(&reordered, DEFAULT_BAD_CASES_SAMPLE_NUM),
        bad
    );
    assert_eq!(
        select_good_cases(&reordered, DEFAULT_BAD_CASES_SAMPLE_NUM),
        good
    );
    // A partially scored case is evidence of neither failure nor success.
    assert!(evaluated_ids(&bad)
        .iter()
        .chain(evaluated_ids(&good).iter())
        .all(|id| id != "c"));
    // The limit truncates the ordered selection.
    assert_eq!(evaluated_ids(&select_bad_cases(&mixed, 1)), ["b"]);
    assert!(
        select_bad_cases(&mixed, 0).is_empty(),
        "a zero limit selects nothing"
    );
}
