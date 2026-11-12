//! Integration tests for the shared evolution protocol.
//!
//! These cover the behaviours Python encodes in `tests/unit_tests/rsi/`
//! (`test_evaluator.py` aggregation, the `ElitistSelection` ordering, and the
//! deterministic mutation output) through the crate's public API only.
//! Everything here is offline and deterministic: no model, network, git, or
//! clock-dependent value is involved.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use openjiuwen_rsi::{
    rank_score, ranked, AutoHarnessConfig, AutoHarnessOrchestrator, Candidate, CandidateId,
    CandidateStatus, Case, CaseResult, DeterministicMutator, ElitistSelection, ErrorKind,
    EvaluationReport, Evaluator, MutationRequest, MutationStrategy, Result, SelectionStrategy,
};

/// Builds a per-case result; `passed` mirrors "no hard failure", as the
/// aggregator intends.
fn result(id: &str, score: Option<f64>, weight: f64, error: Option<&str>) -> CaseResult {
    CaseResult {
        case_id: id.to_owned(),
        passed: error.is_none(),
        score,
        weight,
        error: error.map(str::to_owned),
        details: Value::Null,
    }
}

/// Builds an evaluated candidate whose report holds a single graded case.
fn scored_candidate(id: &str, score: f64) -> Candidate {
    let mut candidate = Candidate::new(id, "body");
    candidate.status = CandidateStatus::Evaluated;
    candidate.evaluation = Some(EvaluationReport::from_results(vec![result(
        "case",
        Some(score),
        1.0,
        None,
    )]));
    candidate
}

/// Builds a candidate nothing has graded yet.
fn unscored_candidate(id: &str) -> Candidate {
    Candidate::new(id, "body")
}

/// Builds a mutation request by hand (no orchestrator needed).
fn mutation_request(parents: Vec<Candidate>, generation: u32, count: usize) -> MutationRequest {
    MutationRequest {
        parents,
        generation,
        count,
        metadata: BTreeMap::new(),
    }
}

/// Compares optional scores with the suite's float tolerance.
fn approx(actual: Option<f64>, expected: f64) -> bool {
    actual.is_some_and(|value| (value - expected).abs() < 1e-9)
}

/// Compares plain floats with the suite's float tolerance.
fn approx_f64(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() < 1e-9
}

/// Renders selected ids as owned strings for readable assertions.
fn selected_ids(ids: Vec<CandidateId>) -> Vec<String> {
    ids.into_iter().map(|id| id.as_str().to_owned()).collect()
}

/// `total`/`passed` count every result, but an ungraded case never enters the
/// average (Python `test_evaluator.py`): 1.0 and 0.0 average to 0.5, not 0.33.
#[test]
fn from_results_counts_all_results_but_averages_only_graded() {
    let report = EvaluationReport::from_results(vec![
        result("a", Some(1.0), 1.0, None),
        result("b", None, 1.0, None),
        result("c", Some(0.0), 1.0, None),
    ]);

    assert_eq!(report.total, 3);
    assert_eq!(report.passed, 3);
    assert_eq!(report.graded, 2);
    assert!(approx(report.average_score, 0.5));
    assert!(approx(report.weighted_score, 0.5));
    assert!(approx(report.score(), 0.5));
}

/// Case weights drive the weighted score while the plain average stays
/// unweighted, and `score()` prefers the weighted value.
#[test]
fn weighted_score_uses_case_weights_and_is_preferred() {
    let report = EvaluationReport::from_results(vec![
        result("a", Some(1.0), 3.0, None),
        result("b", Some(0.0), 1.0, None),
    ]);

    assert_eq!(report.total, 2);
    assert_eq!(report.graded, 2);
    assert!(approx(report.average_score, 0.5));
    assert!(approx(report.weighted_score, 0.75));
    assert!(approx(report.score(), 0.75));
}

/// A zero-weight result is excluded from the weighted score but still counted
/// as a result; with every weight zero the weighted score disappears and
/// `score()` falls back to the average. An empty report is valid but unscored.
#[test]
fn zero_weights_are_excluded_and_empty_reports_are_valid() {
    let report = EvaluationReport::from_results(vec![
        result("a", Some(1.0), 1.0, None),
        result("b", Some(0.0), 0.0, None),
    ]);
    assert_eq!(report.total, 2);
    assert_eq!(report.passed, 2);
    assert_eq!(report.graded, 2);
    assert!(approx(report.weighted_score, 1.0));
    assert!(approx(report.average_score, 0.5));

    let all_zero = EvaluationReport::from_results(vec![
        result("a", Some(1.0), 0.0, None),
        result("b", Some(0.0), 0.0, None),
    ]);
    assert!(all_zero.weighted_score.is_none());
    assert!(approx(all_zero.average_score, 0.5));
    assert!(approx(all_zero.score(), 0.5));

    let empty = EvaluationReport::from_results(Vec::new());
    assert_eq!(empty.total, 0);
    assert_eq!(empty.passed, 0);
    assert_eq!(empty.graded, 0);
    assert!(empty.average_score.is_none());
    assert!(empty.weighted_score.is_none());
    assert!(empty.score().is_none());
    assert!(empty.is_valid(), "nothing failed in an empty report");
}

/// `is_valid()` demands both a score and the absence of an error on every
/// result, and an empty report trivially satisfies it.
#[test]
fn is_valid_requires_graded_error_free_results() {
    let errored = EvaluationReport::from_results(vec![
        result("a", Some(1.0), 1.0, None),
        result("b", Some(0.25), 1.0, Some("grader crashed")),
    ]);
    assert!(
        !errored.is_valid(),
        "an errored result invalidates the report"
    );

    let ungraded = EvaluationReport::from_results(vec![result("a", None, 1.0, None)]);
    assert!(
        !ungraded.is_valid(),
        "an ungraded result invalidates the report"
    );

    let clean = EvaluationReport::from_results(vec![
        result("a", Some(1.0), 1.0, None),
        result("b", Some(0.0), 1.0, None),
    ]);
    assert!(clean.is_valid());
}

/// Elitist selection orders by score descending and breaks ties by candidate id
/// ascending, so the same input always yields the same elite.
#[test]
fn elitist_selection_orders_by_score_then_id() {
    let selection = ElitistSelection::new(3);
    assert_eq!(selection.name(), "elitist");
    assert_eq!(selection.elite_size(), 3);
    assert_eq!(ElitistSelection::default().elite_size(), 1);

    let candidates = vec![
        scored_candidate("low", 0.1),
        scored_candidate("high", 0.9),
        scored_candidate("mid", 0.5),
    ];
    assert_eq!(
        selected_ids(selection.select(&candidates, 3)),
        ["high", "mid", "low"]
    );

    let tied = vec![
        scored_candidate("b", 0.5),
        scored_candidate("a", 0.5),
        scored_candidate("c", 0.5),
    ];
    assert_eq!(
        selected_ids(selection.select(&tied, 3)),
        ["a", "b", "c"],
        "ties break by id ascending"
    );
}

/// Unscored candidates rank last, the result is capped by both `count` and
/// `elite_size`, duplicate ids are selected once, and `elite_size == 0` selects
/// nothing.
#[test]
fn elitist_selection_pushes_unscored_last_and_respects_caps() {
    let selection = ElitistSelection::new(3);
    let candidates = vec![
        unscored_candidate("none"),
        scored_candidate("some", 0.0),
        scored_candidate("high", 0.9),
    ];
    assert_eq!(
        selected_ids(selection.select(&candidates, 3)),
        ["high", "some", "none"]
    );

    assert_eq!(selection.select(&candidates, 1).len(), 1, "capped by count");
    assert_eq!(
        ElitistSelection::default().select(&candidates, 10).len(),
        1,
        "capped by elite_size"
    );
    assert!(
        ElitistSelection::new(0).select(&candidates, 10).is_empty(),
        "a zero elite selects nothing"
    );

    let duplicates = vec![
        scored_candidate("same", 0.9),
        scored_candidate("same", 0.8),
        scored_candidate("other", 0.1),
    ];
    assert_eq!(
        selected_ids(selection.select(&duplicates, 3)),
        ["same", "other"]
    );
}

/// `rank_score` normalises ranks and returns 0.5 when there is no spread;
/// `ranked` sorts ascending with unscored candidates last, feeding the `PUCT`
/// rank prior.
#[test]
fn rank_helpers_normalise_and_sort_ascending() {
    assert!(approx_f64(rank_score(0, 0), 0.5));
    assert!(approx_f64(rank_score(7, 1), 0.5));
    assert!(approx_f64(rank_score(0, 3), 0.0));
    assert!(approx_f64(rank_score(1, 3), 0.5));
    assert!(approx_f64(rank_score(2, 3), 1.0));

    let candidates = vec![
        scored_candidate("c", 0.5),
        unscored_candidate("none"),
        scored_candidate("a", 0.1),
        scored_candidate("b", 0.9),
    ];
    let order: Vec<&str> = ranked(&candidates)
        .iter()
        .map(|candidate| candidate.id.as_str())
        .collect();
    assert_eq!(order, ["a", "c", "b", "none"]);
}

/// The default mutator appends a comment marker and derives ids from the parent
/// id, generation, and variant; identical requests produce identical output.
#[tokio::test]
async fn deterministic_mutator_appends_markers_and_is_stable() {
    let mutator = DeterministicMutator::new();
    assert_eq!(mutator.name(), "deterministic");

    let request = mutation_request(vec![Candidate::new("seed", "model: demo")], 1, 2);
    let first = mutator
        .propose(request.clone())
        .await
        .expect("mutation succeeds");
    let second = mutator.propose(request).await.expect("mutation succeeds");

    assert_eq!(first, second, "mutation is deterministic across runs");
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].id.as_str(), "seed/g1/v0");
    assert_eq!(
        first[0].parent.as_ref().map(CandidateId::as_str),
        Some("seed")
    );
    assert_eq!(first[0].generation, 1);
    assert_eq!(first[0].body, "model: demo\n# rsi variant g1-0");
    assert_eq!(first[0].status, CandidateStatus::Pending);
}

/// Multiple variants are emitted parent-major, `count` truncates the output,
/// and a custom comment prefix is applied verbatim.
#[tokio::test]
async fn deterministic_mutator_respects_variants_count_and_prefix() {
    let mutator = DeterministicMutator::new().with_variants_per_parent(2);
    let proposals = mutator
        .propose(mutation_request(
            vec![Candidate::new("a", "body-a"), Candidate::new("b", "body-b")],
            3,
            4,
        ))
        .await
        .expect("mutation succeeds");
    let ids: Vec<&str> = proposals
        .iter()
        .map(|candidate| candidate.id.as_str())
        .collect();
    assert_eq!(ids, ["a/g3/v0", "a/g3/v1", "b/g3/v0", "b/g3/v1"]);
    assert_eq!(proposals[0].body, "body-a\n# rsi variant g3-0");
    assert_eq!(proposals[1].body, "body-a\n# rsi variant g3-1");
    assert_eq!(proposals[2].body, "body-b\n# rsi variant g3-0");

    let capped = mutator
        .propose(mutation_request(
            vec![Candidate::new("a", "body-a"), Candidate::new("b", "body-b")],
            1,
            3,
        ))
        .await
        .expect("mutation succeeds");
    let capped_ids: Vec<&str> = capped
        .iter()
        .map(|candidate| candidate.id.as_str())
        .collect();
    assert_eq!(capped_ids, ["a/g1/v0", "a/g1/v1", "b/g1/v0"]);

    let prefixed = DeterministicMutator::new()
        .with_comment_prefix("// ")
        .propose(mutation_request(
            vec![Candidate::new("p", "fn main() {}")],
            2,
            1,
        ))
        .await
        .expect("mutation succeeds");
    assert_eq!(prefixed[0].body, "fn main() {}\n// rsi variant g2-0");
}

/// Unusable mutation requests are validation errors, never silent no-ops.
#[tokio::test]
async fn deterministic_mutator_rejects_zero_count_and_zero_variants() {
    let parents = vec![Candidate::new("a", "x")];

    let zero_count = DeterministicMutator::new()
        .propose(mutation_request(parents.clone(), 1, 0))
        .await
        .expect_err("zero count is rejected");
    assert_eq!(zero_count.kind(), ErrorKind::Validation);

    let zero_variants = DeterministicMutator::new()
        .with_variants_per_parent(0)
        .propose(mutation_request(parents, 1, 1))
        .await
        .expect_err("zero variants is rejected");
    assert_eq!(zero_variants.kind(), ErrorKind::Validation);
}

/// A mutator that repeats a proposal id must not burn the budget: the
/// orchestrator counts both proposals but evaluates the unique id only once.
struct DuplicateMutator;

#[async_trait]
impl MutationStrategy for DuplicateMutator {
    fn name(&self) -> &'static str {
        "duplicate"
    }

    async fn propose(&self, _request: MutationRequest) -> Result<Vec<Candidate>> {
        Ok(vec![
            Candidate::new("dup/g1/v0", "a"),
            Candidate::new("dup/g1/v0", "b"),
        ])
    }
}

/// Grades every candidate with a single 0.5 case.
struct HalfScoreEvaluator;

#[async_trait]
impl Evaluator for HalfScoreEvaluator {
    fn name(&self) -> &'static str {
        "half-score"
    }

    async fn evaluate(&self, _candidate: &Candidate, _cases: &[Case]) -> Result<EvaluationReport> {
        Ok(EvaluationReport::from_results(vec![result(
            "case",
            Some(0.5),
            1.0,
            None,
        )]))
    }
}

/// Duplicate proposal ids count as proposed but are never evaluated, and only
/// one candidate with the shared id reaches the session state.
#[tokio::test]
async fn orchestrator_skips_duplicate_proposals() {
    let config = AutoHarnessConfig {
        data_dir: PathBuf::from("rsi-evolution-tests"),
        ..AutoHarnessConfig::default()
    };
    let orchestrator = AutoHarnessOrchestrator::builder(config)
        .with_mutator(Arc::new(DuplicateMutator))
        .with_evaluator(Arc::new(HalfScoreEvaluator))
        .with_mutation_count(2)
        .build()
        .expect("orchestrator builds");

    let record = orchestrator
        .run_iteration()
        .await
        .expect("one iteration succeeds");

    assert_eq!(record.proposed, 2);
    assert_eq!(record.evaluated, 1);
    assert_eq!(record.valid, 1);

    let state = orchestrator.state();
    let duplicates: Vec<&Candidate> = state
        .candidates
        .iter()
        .filter(|candidate| candidate.id.as_str() == "dup/g1/v0")
        .collect();
    assert_eq!(duplicates.len(), 1, "the duplicate id is stored once");
    // The single survivor wins the iteration, so the orchestrator promotes it
    // from `Evaluated` to `Selected`; either way it carries a graded report.
    assert!(matches!(
        duplicates[0].status,
        CandidateStatus::Evaluated | CandidateStatus::Selected
    ));
    assert!(approx(duplicates[0].score(), 0.5));
}
