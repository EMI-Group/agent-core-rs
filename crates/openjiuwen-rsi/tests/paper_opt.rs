//! Offline integration suite for the paper-optimisation core.
//!
//! Python parity: `openjiuwen/rsi/artifact_rsi/paper_opt/auto_research/`
//! `modules/paper_scoring/{schemas,aggregation}.py` and
//! `tree_provider/{schemas,orchestrator}.py` (v0.1.20). Every expected score below
//! is hand-computed from the documented formulas, so no test depends on a golden
//! value produced by the implementation.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use async_trait::async_trait;
use openjiuwen_rsi::artifact_rsi::paper_opt::{
    adoption_gate, aggregate, compact_scoresheet, dispersion, flags_dispersion, round_half_even,
    AdoptionOutcome, ExperimentScores, ExperimentWeights, GeneralScores, GeneralWeights,
    PaperCandidate, PaperChange, PaperNodeType, PaperOptConfig, PaperOptimizer,
    PaperProposalRequest, PaperProposer, PaperRunOutcome, PaperScorer, PaperStatus, PaperTaskState,
    PaperTree, PaperWeights, RubricDispersion, RubricSample, ScoreBreakdown, TemplatePaperProposer,
    DEFAULT_MAD_THRESHOLD, DEFAULT_RANGE_THRESHOLD, PIPELINE_COMPLETE,
};
use openjiuwen_rsi::{Error, ErrorKind, Result};
use serde_json::Value;
use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};
/// Builds a sample from the four general and six experiment dimension scores.
fn sample(general: [f64; 4], experiment: [f64; 6]) -> RubricSample {
    RubricSample {
        general: GeneralScores {
            soundness: general[0],
            clarity: general[1],
            contribution: general[2],
            overall: general[3],
        },
        experiment: ExperimentScores {
            question_alignment: experiment[0],
            design_and_controls: experiment[1],
            measurement_and_statistics: experiment[2],
            reporting_and_reproducibility: experiment[3],
            claim_evidence_alignment: experiment[4],
            overall_experimental_rigor: experiment[5],
        },
    }
}
fn breakdown(composite: f64) -> ScoreBreakdown {
    ScoreBreakdown {
        general: composite,
        experiment: composite,
        composite,
        sample_count: 1,
    }
}
fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-9,
        "expected {expected}, got {actual}"
    );
}
// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------
/// Hand computation for the two samples below:
///
/// general means: soundness 7, clarity 7, contribution 8, overall 7
///   -> 0.30*7 + 0.20*7 + 0.25*8 + 0.25*7 = 2.1 + 1.4 + 2.0 + 1.75 = 7.25
/// experiment means: 8, 7, 7, 6, 6, 8
///   -> 0.20*8 + 0.20*7 + 0.20*7 + 0.15*6 + 0.15*6 + 0.10*8 = 7.0
/// composite = 0.5*7.25 + 0.5*7.0 = 7.125
#[test]
fn aggregation_is_hand_computable() {
    let samples = [
        sample([8.0, 6.0, 7.0, 8.0], [7.0, 6.0, 8.0, 5.0, 7.0, 9.0]),
        sample([6.0, 8.0, 9.0, 6.0], [9.0, 8.0, 6.0, 7.0, 5.0, 7.0]),
    ];
    let scores = aggregate(&samples).expect("aggregate");
    assert_close(scores.general, 7.25);
    assert_close(scores.experiment, 7.0);
    assert_close(scores.composite, 7.125);
    assert_eq!(scores.sample_count, 2);
    assert!(scores.is_improvement_over(&breakdown(7.0)));
    assert!(!scores.is_improvement_over(&scores));
}
#[test]
fn aggregation_rejects_empty_samples_and_bad_weights() {
    let error = aggregate(&[]).expect_err("empty samples");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let weights = PaperWeights {
        general: GeneralWeights {
            soundness: 0.5,
            ..GeneralWeights::default()
        },
        ..PaperWeights::default()
    };
    let error = openjiuwen_rsi::artifact_rsi::paper_opt::aggregate_with_weights(
        &[RubricSample::uniform(7.0)],
        &weights,
    )
    .expect_err("bad weights");
    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(error.to_string().contains("general"), "{error}");
}
#[test]
fn weight_sets_must_sum_to_one_and_name_the_offender() {
    let bad_general = PaperWeights {
        general: GeneralWeights {
            clarity: 0.10,
            ..GeneralWeights::default()
        },
        ..PaperWeights::default()
    };
    let error = bad_general.validate().expect_err("general sum");
    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(error.to_string().contains("general"), "{error}");

    let bad_experiment = PaperWeights {
        experiment: ExperimentWeights {
            overall_experimental_rigor: 0.05,
            ..ExperimentWeights::default()
        },
        ..PaperWeights::default()
    };
    let error = bad_experiment.validate().expect_err("experiment sum");
    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(error.to_string().contains("experiment"), "{error}");

    let bad_composite = PaperWeights {
        composite_general: 0.75,
        ..PaperWeights::default()
    };
    let error = bad_composite.validate().expect_err("composite sum");
    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(error.to_string().contains("composite"), "{error}");

    assert!(GeneralWeights::default().validate().is_ok());
    assert!(ExperimentWeights::default().validate().is_ok());
}
#[test]
fn compact_scoresheet_rounds_half_to_even() {
    // 7.25 / 7.0 / 7.125 are exact; 1/3 exercises the four-decimal rounding.
    let sheet = compact_scoresheet(&ScoreBreakdown {
        general: 7.25,
        experiment: 7.0,
        composite: 1.0 / 3.0,
        sample_count: 3,
    });
    assert_eq!(sheet["general"], 7.25);
    assert_eq!(sheet["experiment"], 7.0);
    assert_eq!(sheet["composite"], 0.3333);
    assert_eq!(sheet["sample_count"], 3.0);

    assert_eq!(round_half_even(2.5, 0), 2.0);
    assert_eq!(round_half_even(3.5, 0), 4.0);
    assert_eq!(round_half_even(0.125, 2), 0.12);
    assert_eq!(round_half_even(0.375, 2), 0.38);
    // The scaled product is exactly 1234.5; the even neighbour is 1234.
    assert_eq!(round_half_even(0.12345, 4), 0.1234);
}
#[test]
fn dispersion_flags_range_mad_and_neither() {
    // overalls 5, 8: range 3, mean 6.5, mad 1.5 -> both
    let both = [
        sample([5.0, 5.0, 5.0, 5.0], [1.0; 6]),
        sample([8.0, 8.0, 8.0, 8.0], [1.0; 6]),
    ];
    let spread = dispersion(&both);
    assert_close(spread.score_range, 3.0);
    assert_close(spread.mad, 1.5);
    assert!(flags_dispersion(
        &both,
        DEFAULT_RANGE_THRESHOLD,
        DEFAULT_MAD_THRESHOLD
    ));

    // overalls 1, 4, 4, 4, 4, 4, 4: range 3, mad (2.571+6*0.429)/7 ~= 0.734 -> range only
    let mut range_only = Vec::new();
    for overall in [1.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0] {
        range_only.push(sample([overall; 4], [4.0; 6]));
    }
    let spread = dispersion(&range_only);
    assert_close(spread.score_range, 3.0);
    assert!(spread.mad < 1.0, "mad {}", spread.mad);
    assert!(flags_dispersion(
        &range_only,
        DEFAULT_RANGE_THRESHOLD,
        DEFAULT_MAD_THRESHOLD
    ));

    // overalls 4, 4, 4, 6, 6, 6: range 2, mean 5, mad 1.0 -> mad only
    let mad_only: Vec<RubricSample> = [4.0, 4.0, 4.0, 6.0, 6.0, 6.0]
        .into_iter()
        .map(|overall| sample([overall; 4], [overall; 6]))
        .collect();
    let spread = dispersion(&mad_only);
    assert_close(spread.score_range, 2.0);
    assert_close(spread.mad, 1.0);
    assert!(flags_dispersion(
        &mad_only,
        DEFAULT_RANGE_THRESHOLD,
        DEFAULT_MAD_THRESHOLD
    ));

    // overalls 5, 5, 6: range 1, mean 5.333..., mad 4/9 -> neither
    let tight: Vec<RubricSample> = [5.0, 5.0, 6.0]
        .into_iter()
        .map(|overall| sample([overall; 4], [overall; 6]))
        .collect();
    assert!(!flags_dispersion(
        &tight,
        DEFAULT_RANGE_THRESHOLD,
        DEFAULT_MAD_THRESHOLD
    ));
    assert_eq!(dispersion(&[]), RubricDispersion::default());
}
// ---------------------------------------------------------------------------
// Adoption gate
// ---------------------------------------------------------------------------
#[test]
fn adoption_gate_follows_pythons_check_order() {
    let candidate = breakdown(9.0);
    assert_eq!(
        adoption_gate(None, &candidate, true, None, "complete"),
        AdoptionOutcome::Adopted
    );
    assert_eq!(
        adoption_gate(None, &candidate, true, None, "failed"),
        AdoptionOutcome::Pruned {
            failure_class: "pipeline_failed".to_owned()
        }
    );
    assert_eq!(
        adoption_gate(None, &candidate, false, None, "complete"),
        AdoptionOutcome::Pruned {
            failure_class: "paper_generation".to_owned()
        }
    );
    assert_eq!(
        adoption_gate(None, &candidate, true, Some("judge down"), "complete"),
        AdoptionOutcome::Pruned {
            failure_class: "scoring_error".to_owned()
        }
    );
    // A pipeline failure outranks the missing artifact, which outranks the
    // scoring error.
    assert_eq!(
        adoption_gate(None, &candidate, false, Some("judge down"), "blocked"),
        AdoptionOutcome::Pruned {
            failure_class: "pipeline_blocked".to_owned()
        }
    );
}
#[test]
fn adoption_gate_requires_strict_improvement() {
    let parent = breakdown(7.0);
    assert_eq!(
        adoption_gate(
            Some(&parent),
            &breakdown(7.5),
            true,
            None,
            PIPELINE_COMPLETE
        ),
        AdoptionOutcome::Adopted
    );
    assert_eq!(
        adoption_gate(
            Some(&parent),
            &breakdown(7.0),
            true,
            None,
            PIPELINE_COMPLETE
        ),
        AdoptionOutcome::Rejected {
            failure_class: "rejected_by_score".to_owned()
        }
    );
    assert_eq!(
        adoption_gate(
            Some(&parent),
            &breakdown(6.5),
            true,
            None,
            PIPELINE_COMPLETE
        ),
        AdoptionOutcome::Rejected {
            failure_class: "rejected_by_score".to_owned()
        }
    );
    let outcome = adoption_gate(None, &parent, true, None, PIPELINE_COMPLETE);
    assert!(outcome.is_adopted());
    assert!(outcome.failure_class().is_none());
    assert_eq!(
        adoption_gate(Some(&parent), &parent, true, None, PIPELINE_COMPLETE).failure_class(),
        Some("rejected_by_score")
    );
}
// ---------------------------------------------------------------------------
// Tree
// ---------------------------------------------------------------------------
#[test]
fn tree_tracks_ids_iterations_paths_and_pruned_nodes() {
    let mut tree = PaperTree::new();
    assert!(tree.is_empty());
    let root = tree.with_root("seed");
    assert_eq!(root, "n000");
    let first = tree
        .add_candidate(Some(&root), "first", Vec::new())
        .expect("add");
    let second = tree
        .add_candidate(Some(&first), "second", Vec::new())
        .expect("add");
    assert_eq!((first.as_str(), second.as_str()), ("n001", "n002"));
    assert_eq!(tree.node(&root).expect("root").iteration, 0);
    assert_eq!(tree.node(&second).expect("second").iteration, 2);
    assert_eq!(tree.len(), 3);
    assert_eq!(tree.next_iteration(), 3);

    let error = tree
        .add_candidate(Some("n999"), "orphan", Vec::new())
        .expect_err("unknown parent");
    assert_eq!(error.kind(), ErrorKind::Validation);

    // Equal scores: the earliest node wins; a later adoption moves the frontier.
    tree.record_score(&first, breakdown(7.0)).expect("score");
    tree.record_score(&second, breakdown(7.0)).expect("score");
    assert_eq!(tree.best().expect("best").node_id, first);
    tree.set_outcome(&first, true, PaperNodeType::Reporting, None)
        .expect("adopt");
    assert_eq!(tree.frontier().expect("frontier").node_id, first);
    tree.set_outcome(
        &second,
        false,
        PaperNodeType::Pruned,
        Some("rejected_by_score".to_owned()),
    )
    .expect("reject");
    assert_eq!(tree.frontier().expect("frontier").node_id, first);
    assert_eq!(tree.pruned_count(), 1);
    assert!(tree.pruned().any(|node| node.node_id == second));

    let path: Vec<&str> = tree
        .path_to(&second)
        .expect("path")
        .into_iter()
        .map(|node| node.node_id.as_str())
        .collect();
    assert_eq!(path, vec![root.as_str(), first.as_str(), second.as_str()]);
    assert_eq!(
        tree.path_to("n999").expect_err("unknown node").kind(),
        ErrorKind::Validation
    );
}
// ---------------------------------------------------------------------------
// Scripted collaborators
// ---------------------------------------------------------------------------
/// Pops one scripted proposal round per call.
struct QueueProposer {
    rounds: Mutex<VecDeque<Vec<PaperCandidate>>>,
    calls: Mutex<u32>,
}
impl QueueProposer {
    fn new(rounds: Vec<Vec<PaperCandidate>>) -> Self {
        Self {
            rounds: Mutex::new(rounds.into()),
            calls: Mutex::new(0),
        }
    }
    fn calls(&self) -> u32 {
        *self.calls.lock().expect("calls lock")
    }
}
#[async_trait]
impl PaperProposer for QueueProposer {
    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "queue"
    }
    async fn propose(&self, _request: PaperProposalRequest) -> Result<Vec<PaperCandidate>> {
        *self.calls.lock().expect("calls lock") += 1;
        Ok(self
            .rounds
            .lock()
            .expect("rounds lock")
            .pop_front()
            .unwrap_or_default())
    }
}
/// Scores a candidate from its `score` metadata; a `scoring_error` flag fails.
struct MetadataScorer;
#[async_trait]
impl PaperScorer for MetadataScorer {
    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "metadata"
    }
    async fn score(&self, candidate: &PaperCandidate) -> Result<ScoreBreakdown> {
        if candidate
            .metadata
            .get("scoring_error")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            return Err(Error::validation("scripted scorer failure"));
        }
        let score = candidate
            .metadata
            .get("score")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        Ok(breakdown(score))
    }
}
fn candidate(id: &str, score: f64) -> PaperCandidate {
    PaperCandidate {
        candidate_id: id.to_owned(),
        parent_id: None,
        summary: format!("candidate {id}"),
        body: format!("body of {id}"),
        changes: vec![PaperChange {
            group: "paper".to_owned(),
            operation: "revise".to_owned(),
            function: "reporting".to_owned(),
            target: "method".to_owned(),
            summary: "revised the method".to_owned(),
        }],
        samples: vec![RubricSample::uniform(score)],
        required_artifact_present: true,
        pipeline_status: PIPELINE_COMPLETE.to_owned(),
        metadata: BTreeMap::from([("score".to_owned(), Value::from(score))]),
    }
}
fn scripted(rounds: Vec<Vec<PaperCandidate>>) -> Arc<QueueProposer> {
    Arc::new(QueueProposer::new(rounds))
}
fn optimizer(
    max_iterations: u32,
    rounds: Vec<Vec<PaperCandidate>>,
) -> (PaperOptimizer, Arc<QueueProposer>) {
    let proposer = scripted(rounds);
    let config = PaperOptConfig {
        max_iterations,
        beam_width: 1,
        ..PaperOptConfig::default()
    };
    let optimizer = PaperOptimizer::new(config)
        .with_proposer(proposer.clone())
        .with_scorer(Arc::new(MetadataScorer))
        .with_task_id("task-1")
        .with_seed_body("seed body");
    (optimizer, proposer)
}
// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------
#[tokio::test]
async fn adopts_a_strictly_better_candidate() {
    let (mut optimizer, proposer) = optimizer(
        2,
        vec![vec![candidate("a1", 7.0)], vec![candidate("a2", 8.5)]],
    );
    let outcome = optimizer.run().await.expect("run");
    assert_eq!(outcome.state.status, PaperStatus::Completed);
    assert_eq!(proposer.calls(), 2);
    assert_eq!(outcome.adopted.len(), 2);
    assert_eq!(outcome.rejected.len(), 0);
    assert_eq!(outcome.state.score, Some(8.5));
    assert_eq!(outcome.state.baseline, Some(7.0));
    assert_eq!(outcome.state.best_node_id.as_deref(), Some("n002"));
    assert_eq!(outcome.state.frontier_node_id.as_deref(), Some("n002"));
    assert_eq!(outcome.state.node_count, 2);
    assert_eq!(outcome.state.attempts_since_last_adoption, 0);
    assert_eq!(outcome.state.last_reason, "max_iterations");
    let best = outcome.tree.best().expect("best node");
    assert_eq!(best.node_id, "n002");
    assert_eq!(best.node_type, PaperNodeType::Reporting);
    assert!(best.adopted);
    assert!((best.score.expect("score").composite - 8.5).abs() < f64::EPSILON);
}
#[tokio::test]
async fn rejects_a_regressing_candidate() {
    let (mut optimizer, _proposer) = optimizer(
        2,
        vec![vec![candidate("a1", 7.0)], vec![candidate("a2", 6.0)]],
    );
    let outcome = optimizer.run().await.expect("run");
    assert_eq!(outcome.state.status, PaperStatus::Completed);
    assert_eq!(outcome.adopted.len(), 1);
    assert_eq!(outcome.rejected.len(), 1);
    assert_eq!(outcome.state.score, Some(7.0));
    assert_eq!(outcome.state.frontier_node_id.as_deref(), Some("n001"));
    assert_eq!(outcome.state.attempts_since_last_adoption, 1);
    assert_eq!(outcome.state.last_reason, "rejected_by_score");
    let rejected = outcome
        .tree
        .node(&outcome.rejected[0])
        .expect("rejected node");
    assert_eq!(rejected.node_type, PaperNodeType::Pruned);
    assert!(!rejected.adopted);
    assert_eq!(rejected.failure_class.as_deref(), Some("rejected_by_score"));
    assert_eq!(outcome.tree.pruned_count(), 1);
}
#[tokio::test]
async fn stops_at_the_iteration_budget() {
    let rounds: Vec<Vec<PaperCandidate>> = (1..=5)
        .map(|round| vec![candidate(&format!("a{round}"), 6.0 + f64::from(round))])
        .collect();
    let (mut optimizer, proposer) = optimizer(3, rounds);
    let outcome = optimizer.run().await.expect("run");
    assert_eq!(proposer.calls(), 3);
    assert_eq!(outcome.state.status, PaperStatus::Completed);
    assert_eq!(outcome.state.node_count, 3);
    assert_eq!(outcome.tree.len(), 4);
    assert_eq!(outcome.state.score, Some(9.0));
}
#[tokio::test]
async fn stops_when_the_target_composite_is_reached() {
    let proposer = scripted(vec![
        vec![candidate("a1", 7.0)],
        vec![candidate("a2", 8.0)],
        vec![candidate("a3", 9.0)],
    ]);
    let config = PaperOptConfig {
        max_iterations: 5,
        beam_width: 1,
        min_composite: Some(7.5),
        ..PaperOptConfig::default()
    };
    let mut optimizer = PaperOptimizer::new(config)
        .with_proposer(proposer.clone())
        .with_scorer(Arc::new(MetadataScorer));
    let outcome = optimizer.run().await.expect("run");
    assert_eq!(outcome.state.status, PaperStatus::Completed);
    assert_eq!(outcome.state.score, Some(8.0));
    assert_eq!(outcome.state.last_reason, "min_composite_reached");
    assert_eq!(proposer.calls(), 2);
    assert_eq!(outcome.adopted.len(), 2);
}
#[tokio::test]
async fn stop_marks_the_run_cancelled() {
    let (mut optimizer, proposer) = optimizer(3, vec![vec![candidate("a1", 7.0)]]);
    optimizer.stop();
    let outcome = optimizer.run().await.expect("run");
    assert_eq!(outcome.state.status, PaperStatus::Cancelled);
    assert_eq!(outcome.state.last_reason, "cancelled");
    assert_eq!(proposer.calls(), 0);
    assert!(outcome.tree.is_empty());
    assert_eq!(outcome.adopted.len(), 0);
    assert_eq!(outcome.rejected.len(), 0);
}
#[tokio::test]
async fn scoring_errors_prune_without_aborting_the_run() {
    let mut broken = candidate("broken", 9.0);
    broken
        .metadata
        .insert("scoring_error".to_owned(), Value::Bool(true));
    let (mut optimizer, _proposer) = optimizer(2, vec![vec![broken], vec![candidate("a2", 6.0)]]);
    let outcome = optimizer.run().await.expect("run");
    assert_eq!(outcome.state.status, PaperStatus::Completed);
    assert_eq!(outcome.rejected.len(), 1);
    assert_eq!(outcome.adopted.len(), 1);
    let pruned = outcome.tree.node(&outcome.rejected[0]).expect("pruned");
    assert_eq!(pruned.failure_class.as_deref(), Some("scoring_error"));
    // The unadopted root is a valid frontier for the next round, so the second
    // candidate becomes the baseline.
    assert_eq!(outcome.state.baseline, Some(6.0));
    assert_eq!(outcome.state.score, Some(6.0));
}
#[tokio::test]
async fn pipeline_failures_and_missing_artifacts_are_pruned() {
    let mut failed = candidate("failed", 9.0);
    failed.pipeline_status = "failed".to_owned();
    let mut no_artifact = candidate("no-artifact", 9.5);
    no_artifact.required_artifact_present = false;
    let config = PaperOptConfig {
        max_iterations: 1,
        beam_width: 2,
        ..PaperOptConfig::default()
    };
    let proposer = scripted(vec![vec![failed, no_artifact]]);
    let mut optimizer = PaperOptimizer::new(config)
        .with_proposer(proposer)
        .with_scorer(Arc::new(MetadataScorer));
    let outcome = optimizer.run().await.expect("run");
    // One node per iteration: the round winner is the 9.5 candidate, pruned for
    // its missing artifact (the pipeline failure loses the ranking on score).
    assert_eq!(outcome.rejected.len(), 1);
    assert_eq!(outcome.adopted.len(), 0);
    assert_eq!(outcome.state.last_reason, "paper_generation");
    assert_eq!(outcome.state.score, None);
    assert_eq!(outcome.state.best_node_id, None);
}
#[tokio::test]
async fn ids_are_deterministic_across_runs() {
    let rounds = || {
        vec![
            vec![candidate("a1", 7.0)],
            vec![candidate("a2", 6.0)],
            vec![candidate("a3", 8.0)],
        ]
    };
    let run = |rounds: Vec<Vec<PaperCandidate>>| {
        let proposer = scripted(rounds);
        let config = PaperOptConfig {
            max_iterations: 3,
            beam_width: 1,
            ..PaperOptConfig::default()
        };
        PaperOptimizer::new(config)
            .with_proposer(proposer)
            .with_scorer(Arc::new(MetadataScorer))
            .with_task_id("fixed-task")
            .with_seed_body("seed body")
    };
    let mut first = run(rounds());
    let mut second = run(rounds());
    let first_outcome = first.run().await.expect("first run");
    let second_outcome = second.run().await.expect("second run");
    let ids = |outcome: &PaperRunOutcome| {
        outcome
            .tree
            .nodes()
            .iter()
            .map(|node| (node.node_id.clone(), node.iteration, node.adopted))
            .collect::<Vec<_>>()
    };
    assert_eq!(ids(&first_outcome), ids(&second_outcome));
    assert_eq!(first_outcome.adopted, second_outcome.adopted);
    assert_eq!(first_outcome.rejected, second_outcome.rejected);
    assert_eq!(first_outcome.state, second_outcome.state);
}
#[tokio::test]
async fn default_template_proposer_and_rubric_scorer_end_to_end() {
    let config = PaperOptConfig {
        max_iterations: 2,
        beam_width: 2,
        ..PaperOptConfig::default()
    };
    let mut optimizer = PaperOptimizer::new(config).with_task_id("template-task");
    let outcome = optimizer.run().await.expect("run");
    assert_eq!(outcome.state.status, PaperStatus::Completed);
    // Round 1: two equal 7.0 candidates, the first is adopted. Round 2: two
    // equal 7.0 candidates tie the frontier, so the second round is rejected.
    assert_eq!(outcome.adopted.len(), 1);
    assert_eq!(outcome.rejected.len(), 1);
    assert_eq!(outcome.state.baseline, Some(7.0));
    assert_eq!(outcome.state.score, Some(7.0));
    assert_eq!(outcome.state.attempts_since_last_adoption, 1);
    assert_eq!(outcome.state.last_reason, "rejected_by_score");
    assert_eq!(outcome.tree.pruned_count(), 1);
    // Ids are the sequential tree ids; candidate ids never leak into the tree.
    assert_eq!(outcome.adopted[0], "n001");
    assert_eq!(outcome.rejected[0], "n002");
}
#[tokio::test]
async fn empty_proposals_complete_the_run() {
    let (mut optimizer, _proposer) = optimizer(5, Vec::new());
    let outcome = optimizer.run().await.expect("run");
    assert_eq!(outcome.state.status, PaperStatus::Completed);
    assert_eq!(outcome.state.last_reason, "no_candidates");
    assert_eq!(outcome.state.node_count, 0);
    assert_eq!(outcome.adopted.len(), 0);
}
#[tokio::test]
async fn run_iteration_reports_none_once_stopped() {
    let (mut optimizer, _proposer) = optimizer(1, vec![vec![candidate("a1", 7.0)]]);
    let node = optimizer.run_iteration().await.expect("iteration");
    assert_eq!(node.as_deref(), Some("n001"));
    assert_eq!(optimizer.state().status, PaperStatus::Completed);
    assert_eq!(optimizer.run_iteration().await.expect("stopped"), None);
}
#[test]
fn task_state_round_trips_through_json() {
    let state = PaperTaskState {
        task_id: "task-1".to_owned(),
        status: PaperStatus::Cancelled,
        max_iterations: 3,
        artifact_path: "paper/main.pdf".to_owned(),
        best_node_id: Some("n001".to_owned()),
        score: Some(7.25),
        baseline: Some(7.0),
        frontier_node_id: Some("n001".to_owned()),
        node_count: 1,
        attempts_since_last_adoption: 0,
        last_reason: "cancelled".to_owned(),
        error_code: Some("validation".to_owned()),
        error_message: Some("boom".to_owned()),
    };
    let encoded = serde_json::to_value(&state).expect("serialize");
    let decoded: PaperTaskState = serde_json::from_value(encoded).expect("deserialize");
    assert_eq!(decoded, state);
}
#[tokio::test]
async fn template_proposer_is_deterministic_across_optimizers() {
    let mut first = PaperOptimizer::new(PaperOptConfig {
        max_iterations: 2,
        beam_width: 2,
        ..PaperOptConfig::default()
    })
    .with_proposer(Arc::new(TemplatePaperProposer::new()))
    .with_task_id("fixed");
    let mut second = PaperOptimizer::new(PaperOptConfig {
        max_iterations: 2,
        beam_width: 2,
        ..PaperOptConfig::default()
    })
    .with_proposer(Arc::new(TemplatePaperProposer::new()))
    .with_task_id("fixed");
    let first_outcome = first.run().await.expect("first");
    let second_outcome = second.run().await.expect("second");
    assert_eq!(first_outcome.tree, second_outcome.tree);
    assert_eq!(first_outcome.state, second_outcome.state);
}
