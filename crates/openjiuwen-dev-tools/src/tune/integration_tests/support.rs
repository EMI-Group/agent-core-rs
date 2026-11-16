//! Shared fixtures for the integration suite.
//!
//! Case builders, a score comparison, and the two handles the suites share (an
//! in-memory run log and an adoption service over a prompt memory). Everything
//! here is deterministic: ids, seeds, and timestamps are literals.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use serde_json::json;

use crate::tune::{
    AdoptPolicy, AdoptService, Case, EvaluatedCase, InMemoryRunLog, PromptMemory, RunLog,
    RunLogRecord,
};

/// The timestamp every scenario stamps its records with (no clock is read).
pub(super) const NOW: f64 = 1_700_000_000.0;

/// The seed prompt the trainer scenarios optimize from.
pub(super) const SEED_PROMPT: &str = "Answer the question: {{question}}";

/// A token that appears only in the held-out cases of the isolation scenario.
pub(super) const VAL_ONLY: &str = "VALONLYTOKEN";

/// A case whose inputs and label carry `question`/`answer`.
pub(super) fn case(id: &str, question: &str, answer: &str) -> Case {
    Case::new(
        BTreeMap::from([("question".to_owned(), json!(question))]),
        BTreeMap::from([("answer".to_owned(), json!(answer))]),
    )
    .expect("a case with non-empty inputs and label")
    .with_case_id(id)
}

/// Nine cases with distinct questions and labels.
pub(super) fn nine_cases() -> Vec<Case> {
    (1..=9)
        .map(|index| {
            case(
                &format!("c{index}"),
                &format!("question {index}"),
                &format!("answer {index}"),
            )
        })
        .collect()
}

/// Nine cases in three three-case strata, keyed by the label `alpha`/`beta`/`gamma`.
pub(super) fn stratified_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for stratum in ["alpha", "beta", "gamma"] {
        for index in 0..3 {
            cases.push(case(
                &format!("{stratum}-{index}"),
                &format!("question {stratum} {index}"),
                stratum,
            ));
        }
    }
    cases
}

/// The case ids of `cases`, in order.
pub(super) fn ids(cases: &[Case]) -> Vec<String> {
    cases.iter().map(|entry| entry.case_id.clone()).collect()
}

/// The case ids of `cases`, in order.
pub(super) fn evaluated_ids(cases: &[EvaluatedCase]) -> Vec<String> {
    cases
        .iter()
        .map(|entry| entry.case_id().to_owned())
        .collect()
}

/// An evaluated case with the given id and score.
pub(super) fn evaluated(id: &str, score: f64) -> EvaluatedCase {
    EvaluatedCase::new(
        case(id, "question", "answer"),
        BTreeMap::new(),
        score,
        "scripted score",
    )
    .expect("a score inside 0.0..=1.0")
}

/// Whether any input or label value of `entry` mentions `token`.
pub(super) fn mentions(entry: &Case, token: &str) -> bool {
    entry
        .inputs
        .values()
        .chain(entry.label.values())
        .any(|value| value.to_string().contains(token))
}

/// Asserts that two scores agree to the crate's comparison tolerance.
pub(super) fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "expected {expected}, got {actual}"
    );
}

/// A shared in-memory run log plus the handle an optimizer takes.
pub(super) fn shared_log() -> (Arc<Mutex<InMemoryRunLog>>, Arc<Mutex<dyn RunLog>>) {
    let concrete = Arc::new(Mutex::new(InMemoryRunLog::new()));
    let handle: Arc<Mutex<dyn RunLog>> = concrete.clone();
    (concrete, handle)
}

/// The records an in-memory run log holds, in write order.
pub(super) fn recorded(log: &Arc<Mutex<InMemoryRunLog>>) -> Vec<RunLogRecord> {
    log.lock()
        .expect("the in-memory run log is not poisoned")
        .records()
        .to_vec()
}

/// A default adoption service over `memory`.
pub(super) fn adopt_service(memory: Arc<dyn PromptMemory>) -> AdoptService {
    AdoptService::new(AdoptPolicy::new()).with_memory(memory)
}
