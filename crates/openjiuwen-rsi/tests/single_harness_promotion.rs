//! Integration tests for the single-harness promotion mathematics and state.
//!
//! These pin the deterministic rules Python encodes in
//! `openjiuwen/rsi/harness_rsi/single_harness/promotion_policy.py` (batch
//! filter, target improvement, the two promotion gates) plus the resumable
//! state layout (`state.yaml`). Everything here is offline: no evaluator, no
//! model, no clock.

// The emptiness assertions read better than `assert_eq!(value, empty_value)`.
#![allow(clippy::assert_is_empty)]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use serde_json::json;

use openjiuwen_rsi::harness_rsi::single_harness::{
    attempt_key, batch_filter, batch_key, case_scores, full_evaluation_decision,
    is_batch_improvement, mean_improvement, stage_one_gate, state_path, targets_improved,
    threshold_saturated, CandidateGateRecord, CaseScores, JournalEntry, PromotionPolicyMode,
    SingleHarnessConfig, SingleHarnessState, SingleHarnessStatus, UsageState, EPSILON,
    REASON_BATCH_GATE_PASSED, REASON_CONTENT_UNCHANGED, REASON_ENGINEERING_FAILED,
    REASON_PROTECTED_CASE_REGRESSED, REASON_SELECTED, REASON_SELECTED_SCORE_REGRESSED,
    REASON_TARGET_REPLAY_FAILED, REASON_THRESHOLD_SATURATED,
};
use openjiuwen_rsi::{CaseResult, ErrorKind, EvaluationReport, StopReason};

/// Builds a score map from literal pairs.
fn scores(entries: &[(&str, f64)]) -> CaseScores {
    entries
        .iter()
        .map(|(case_id, score)| ((*case_id).to_owned(), *score))
        .collect()
}

/// Builds an owned id list from literals.
fn ids(entries: &[&str]) -> Vec<String> {
    entries.iter().map(|entry| (*entry).to_owned()).collect()
}

/// Compares floats with the promotion tolerance.
fn approx(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() < 1e-9
}

/// Builds a report with the given `(case_id, score)` pairs; `None` marks an
/// ungraded case.
fn report(entries: &[(&str, Option<f64>, Option<&str>)]) -> EvaluationReport {
    EvaluationReport::from_results(
        entries
            .iter()
            .map(|(case_id, score, error)| CaseResult {
                case_id: (*case_id).to_owned(),
                passed: true,
                score: *score,
                weight: 1.0,
                error: error.map(str::to_owned),
                details: json!(null),
            })
            .collect(),
    )
}

#[test]
fn batch_filter_excludes_the_exact_margin_and_keeps_epsilon_above() {
    let filtered = batch_filter(
        &scores(&[
            ("exact", 0.5),
            ("epsilon-above", 0.499_999_9),
            ("epsilon-below", 0.499_999_999_5),
            ("satisfied", 0.96),
        ]),
        0.5,
    );
    assert_eq!(filtered, ids(&["epsilon-above"]));
}

#[test]
fn batch_filter_is_ascending_and_handles_an_empty_map() {
    assert_eq!(
        batch_filter(&scores(&[("b", 0.0), ("a", 0.1)]), 0.05),
        ids(&["a", "b"])
    );
    assert!(batch_filter(&CaseScores::new(), 0.05).is_empty());
}

#[test]
fn is_batch_improvement_requires_strictly_more_than_epsilon() {
    assert!(is_batch_improvement(0.5, 0.6));
    assert!(!is_batch_improvement(0.5, 0.5));
    assert!(!is_batch_improvement(0.5, 0.5 + EPSILON));
    assert!(!is_batch_improvement(0.6, 0.5));
}

#[test]
fn targets_improved_is_strict_and_follows_missing_entries() {
    let before = scores(&[("a", 0.5), ("b", 0.5), ("c", 0.5)]);
    let after = scores(&[("a", 0.6), ("b", 0.5 + EPSILON), ("d", 0.7)]);
    assert_eq!(targets_improved(&before, &after), ids(&["a", "d"]));

    let equal = scores(&[("a", 0.5)]);
    assert!(targets_improved(&equal, &equal).is_empty());
    assert!(targets_improved(&CaseScores::new(), &CaseScores::new()).is_empty());
}

#[test]
fn mean_improvement_averages_only_the_improved_targets() {
    let before = scores(&[("a", 0.5), ("b", 0.5), ("c", 0.5)]);
    let after = scores(&[("a", 0.6), ("b", 0.7), ("c", 0.5)]);
    assert!(approx(mean_improvement(&before, &after), 0.15));

    // A case without a baseline contributes its whole score as the delta.
    let fresh = scores(&[("new", 0.4)]);
    assert!(approx(mean_improvement(&CaseScores::new(), &fresh), 0.4));

    // No improved target at all: the helper never divides by zero.
    assert!(approx(mean_improvement(&before, &before), 0.0));
    assert!(approx(
        mean_improvement(&CaseScores::new(), &CaseScores::new()),
        0.0
    ));
}

#[test]
fn threshold_saturated_needs_a_non_empty_fully_satisfied_batch() {
    assert!(threshold_saturated(&scores(&[
        ("a", 1.0),
        ("b", 0.999_999_999)
    ])));
    assert!(!threshold_saturated(&scores(&[("a", 0.999)])));
    assert!(!threshold_saturated(&CaseScores::new()));
}

#[test]
fn stage_one_gate_reports_every_reason_literal() {
    let before = scores(&[("a", 0.5)]);
    let better = scores(&[("a", 0.7)]);

    let unchanged = stage_one_gate(&before, &better, false, None);
    assert!(!unchanged.passed);
    assert!(unchanged.improved.is_empty());
    assert_eq!(unchanged.reason, REASON_CONTENT_UNCHANGED);

    let failed = stage_one_gate(&before, &better, true, Some("runner crashed"));
    assert!(!failed.passed);
    assert_eq!(
        failed.reason,
        format!("{REASON_ENGINEERING_FAILED}: runner crashed")
    );

    let saturated = stage_one_gate(&scores(&[("a", 1.0)]), &better, true, None);
    assert!(!saturated.passed);
    assert_eq!(saturated.reason, REASON_THRESHOLD_SATURATED);

    let flat = stage_one_gate(&before, &before, true, None);
    assert!(!flat.passed);
    assert!(flat.improved.is_empty());
    assert_eq!(flat.reason, REASON_TARGET_REPLAY_FAILED);

    let passed = stage_one_gate(&before, &better, true, None);
    assert!(passed.passed);
    assert_eq!(passed.improved, ids(&["a"]));
    assert_eq!(passed.reason, REASON_BATCH_GATE_PASSED);
}

#[test]
fn full_evaluation_promotes_a_held_target_with_retention() {
    let decision = full_evaluation_decision(
        &scores(&[("a", 0.5), ("b", 0.5)]),
        &scores(&[("a", 0.7), ("b", 0.5)]),
        &ids(&["a"]),
        &[],
        0.05,
        0.05,
    );
    assert!(decision.promoted);
    assert_eq!(decision.reason, REASON_SELECTED);
    assert!(decision.failed_targets.is_empty());
    assert!(decision.regressed_others.is_empty());
    assert!(approx(decision.mean_improvement, 0.1));
}

#[test]
fn full_evaluation_rejects_a_target_that_did_not_hold_its_gain() {
    let decision = full_evaluation_decision(
        &scores(&[("a", 0.5), ("b", 0.5)]),
        &scores(&[("a", 0.5), ("b", 0.5)]),
        &ids(&["a"]),
        &[],
        0.05,
        0.05,
    );
    assert!(!decision.promoted);
    assert_eq!(decision.reason, REASON_TARGET_REPLAY_FAILED);
    assert_eq!(decision.failed_targets, ids(&["a"]));

    // A target that vanished from the full evaluation also fails replay.
    let vanished = full_evaluation_decision(
        &scores(&[("a", 0.5)]),
        &scores(&[("b", 0.5)]),
        &ids(&["a"]),
        &[],
        0.05,
        0.05,
    );
    assert!(!vanished.promoted);
    assert_eq!(vanished.reason, REASON_TARGET_REPLAY_FAILED);
}

#[test]
fn full_evaluation_rejects_an_empty_target_set() {
    let decision = full_evaluation_decision(
        &scores(&[("a", 0.5)]),
        &scores(&[("a", 0.9)]),
        &[],
        &[],
        0.05,
        0.05,
    );
    assert!(!decision.promoted);
    assert_eq!(decision.reason, REASON_TARGET_REPLAY_FAILED);
}

#[test]
fn full_evaluation_rejects_a_protected_case_regression_first() {
    let decision = full_evaluation_decision(
        &scores(&[("a", 0.5), ("p", 0.9)]),
        &scores(&[("a", 0.8), ("p", 0.7)]),
        &ids(&["a"]),
        &ids(&["p"]),
        0.05,
        0.05,
    );
    assert!(!decision.promoted);
    assert_eq!(decision.reason, REASON_PROTECTED_CASE_REGRESSED);
    assert!(decision.failed_targets.is_empty());

    // A protected case that disappears counts as a total loss.
    let vanished = full_evaluation_decision(
        &scores(&[("a", 0.5), ("p", 0.9)]),
        &scores(&[("a", 0.8)]),
        &ids(&["a"]),
        &ids(&["p"]),
        0.05,
        0.05,
    );
    assert!(!vanished.promoted);
    assert_eq!(vanished.reason, REASON_PROTECTED_CASE_REGRESSED);

    // A protected case with no baseline cannot regress.
    let fresh = full_evaluation_decision(
        &scores(&[("a", 0.5)]),
        &scores(&[("a", 0.8), ("p", 0.1)]),
        &ids(&["a"]),
        &ids(&["p"]),
        0.05,
        0.05,
    );
    assert!(fresh.promoted);
}

#[test]
fn full_evaluation_rejects_an_aggregate_regression() {
    let decision = full_evaluation_decision(
        &scores(&[("a", 0.5), ("b", 0.5)]),
        &scores(&[("a", 0.6), ("b", 0.2)]),
        &ids(&["a"]),
        &[],
        0.05,
        0.05,
    );
    assert!(!decision.promoted);
    assert_eq!(decision.reason, REASON_SELECTED_SCORE_REGRESSED);
    assert_eq!(decision.regressed_others, ids(&["b"]));
    assert!(approx(decision.mean_improvement, -0.1));
}

#[test]
fn full_evaluation_rejects_a_retention_mean_on_the_boundary() {
    // The retained mean is exactly -max_regression (both values are exactly
    // representable); the rule is "strictly greater than -max_regression", so
    // the candidate is rejected.
    let decision = full_evaluation_decision(
        &scores(&[("a", 0.5), ("b", 0.5)]),
        &scores(&[("a", 0.9), ("b", 0.25)]),
        &ids(&["a"]),
        &[],
        0.05,
        0.25,
    );
    assert!(!decision.promoted);
    assert_eq!(decision.reason, REASON_SELECTED_SCORE_REGRESSED);
}

#[test]
fn full_evaluation_rejects_a_positive_but_negligible_gain() {
    // The target holds its margin while a non-target loses the same amount, so
    // the overall mean gain is inside EPSILON: retention is fine, but rule (d)
    // still rejects the promotion.
    let decision = full_evaluation_decision(
        &scores(&[("a", 0.5), ("b", 0.5)]),
        &scores(&[("a", 0.5 + 1.5 * EPSILON), ("b", 0.5 - 1.5 * EPSILON)]),
        &ids(&["a"]),
        &[],
        0.05,
        0.05,
    );
    assert!(!decision.promoted);
    assert_eq!(decision.reason, REASON_SELECTED_SCORE_REGRESSED);
}

#[test]
fn case_scores_skips_ungraded_and_errored_results() {
    let scored = case_scores(&report(&[
        ("a", Some(0.9), None),
        ("b", None, None),
        ("c", Some(0.4), Some("grader crashed")),
    ]));
    assert_eq!(scored, scores(&[("a", 0.9)]));
}

#[test]
fn fingerprint_is_stable_and_covers_dataset_signature_and_mode() {
    let config = SingleHarnessConfig::default();
    let dataset = ids(&["a", "b"]);
    let fingerprint = config.fingerprint(&dataset);
    assert_eq!(fingerprint, config.fingerprint(&dataset));
    assert_eq!(fingerprint.len(), 16);
    assert!(fingerprint
        .chars()
        .all(|ch| ch.is_ascii_hexdigit() && !ch.is_ascii_uppercase()));

    assert_ne!(fingerprint, config.fingerprint(&ids(&["b", "a"])));
    assert_ne!(fingerprint, config.fingerprint(&ids(&["a"])));

    let mut other_signature = config.clone();
    other_signature.evaluation_protocol_signature = "v2".to_owned();
    assert_ne!(fingerprint, other_signature.fingerprint(&dataset));

    let mut other_mode = config;
    other_mode.mode = PromotionPolicyMode::LegacyEpochV1;
    assert_ne!(fingerprint, other_mode.fingerprint(&dataset));
}

#[test]
fn config_defaults_and_validation_match_python() {
    let config = SingleHarnessConfig::default();
    assert_eq!(config.mode, PromotionPolicyMode::ContinuousTwoStageV2);
    assert_eq!(config.max_epochs, 5);
    assert_eq!(config.max_iterations, 10);
    assert_eq!(config.batch_size, 5);
    assert!(approx(config.min_improvement, 0.05));
    assert!(approx(config.max_regression, 0.05));
    assert_eq!(config.patience, 2);
    assert_eq!(config.max_attempts_per_batch, 2);
    assert_eq!(config.max_trials, 8);
    assert!(config.promote_on_best_found);
    assert_eq!(config.evaluation_protocol_signature, "v1");
    assert!(config.validate().is_ok());

    let mut no_batch = config.clone();
    no_batch.batch_size = 0;
    assert_eq!(no_batch.validate().unwrap_err().kind(), ErrorKind::Config);

    let mut no_iterations = config.clone();
    no_iterations.max_iterations = 0;
    assert_eq!(
        no_iterations.validate().unwrap_err().kind(),
        ErrorKind::Config
    );

    let mut bad_margin = config.clone();
    bad_margin.min_improvement = 1.5;
    assert_eq!(bad_margin.validate().unwrap_err().kind(), ErrorKind::Config);

    let mut nan_margin = config;
    nan_margin.max_regression = f64::NAN;
    assert_eq!(nan_margin.validate().unwrap_err().kind(), ErrorKind::Config);
}

#[test]
fn config_tolerates_unknown_and_missing_yaml_keys() {
    let config: SingleHarnessConfig =
        serde_yaml_ng::from_str("max_epochs: 3\nbatch_size: 2\nfuture_key:\n  nested: true\n")
            .expect("config parses");
    assert_eq!(config.max_epochs, 3);
    assert_eq!(config.batch_size, 2);
    assert_eq!(config.max_iterations, 10);
    assert_eq!(config.mode, PromotionPolicyMode::ContinuousTwoStageV2);
}

#[test]
fn batch_and_attempt_keys_are_padded_and_stable() {
    assert_eq!(batch_key(1, 0), "epoch_001:batch_000");
    assert_eq!(batch_key(12, 7), "epoch_012:batch_007");
    assert_eq!(
        attempt_key("epoch_001:batch_000", 3),
        "epoch_001:batch_000:attempt_003"
    );
}

#[test]
fn state_records_completed_batches_idempotently() {
    let mut state = SingleHarnessState::new(
        "task",
        "fingerprint",
        ids(&["a"]),
        3,
        PromotionPolicyMode::ContinuousTwoStageV2,
    );
    assert_eq!(state.version, 7);
    assert_eq!(state.status, SingleHarnessStatus::Running);
    assert!(!state.is_batch_completed("epoch_001:batch_000"));
    state.mark_batch_completed("epoch_001:batch_000");
    state.mark_batch_completed("epoch_001:batch_000");
    assert_eq!(state.completed_batches.len(), 1);
    assert!(state.is_batch_completed("epoch_001:batch_000"));
}

#[test]
fn state_yaml_round_trip_preserves_every_field() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    runtime.block_on(async {
        let mut state = SingleHarnessState::new(
            "task-1",
            "fp",
            ids(&["a", "b"]),
            4,
            PromotionPolicyMode::LegacyEpochV1,
        );
        state.status = SingleHarnessStatus::Completed;
        state.best_harness_refs_path = Some("refs.yaml".to_owned());
        state.best_score = Some(0.8);
        state.best_body = "model: demo".to_owned();
        state.baseline_score = Some(0.5);
        state.mark_batch_completed("epoch_001:batch_000");
        state.candidate_gates.push(CandidateGateRecord {
            batch_key: "epoch_001:batch_000".to_owned(),
            candidate_id: "seed/g1/v0".to_owned(),
            passed: true,
            reason: REASON_BATCH_GATE_PASSED.to_owned(),
            round: 1,
        });
        state.optimization_journal.push(JournalEntry {
            epoch: 1,
            batch_index: 0,
            reason: REASON_SELECTED.to_owned(),
            score: Some(0.8),
            promoted: true,
        });
        state.historical_peak_score = Some(0.9);
        state.continuous_attempts = 1;
        state.continuous_trials = 3;
        state.evaluation_protocol_signature = "v1".to_owned();
        state.stop_reason = Some(StopReason::TargetScore);
        state.usage = UsageState {
            calls: 7,
            cost_usd: 0.25,
        };
        state
            .metadata
            .insert("note".to_owned(), json!({"nested": [1, 2]}));

        let dir = tempfile::tempdir().expect("temp dir");
        let path = state_path(dir.path());
        assert!(path.ends_with("single_harness_state.yaml"));
        state.save(&path).await.expect("state saves");

        let loaded = SingleHarnessState::load(&path).await.expect("state loads");
        assert_eq!(loaded, state);
    });
}

#[test]
fn state_load_reports_a_missing_file_as_a_session_error() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    runtime.block_on(async {
        let dir = tempfile::tempdir().expect("temp dir");
        let err = SingleHarnessState::load(dir.path().join("absent.yaml"))
            .await
            .expect_err("missing file");
        assert_eq!(err.kind(), ErrorKind::Session);
    });
}

#[test]
fn resume_from_accepts_a_matching_state_and_clears_the_stop() {
    let mut previous = SingleHarnessState::new(
        "task",
        "fp",
        ids(&["a"]),
        3,
        PromotionPolicyMode::ContinuousTwoStageV2,
    );
    previous.status = SingleHarnessStatus::Completed;
    previous.stop_reason = Some(StopReason::MaxIterations);
    previous.best_score = Some(0.7);
    previous.mark_batch_completed("epoch_001:batch_000");
    previous.optimization_journal.push(JournalEntry {
        epoch: 1,
        batch_index: 0,
        reason: REASON_SELECTED.to_owned(),
        score: Some(0.7),
        promoted: true,
    });
    previous.evaluation_protocol_signature = "v1".to_owned();

    let resumed = SingleHarnessState::resume_from(previous.clone(), "fp", "task", 3, "v1")
        .expect("state resumes");
    assert_eq!(resumed.status, SingleHarnessStatus::Running);
    assert!(resumed.stop_reason.is_none());
    assert_eq!(resumed.best_score, Some(0.7));
    assert_eq!(resumed.completed_batches, previous.completed_batches);
    assert_eq!(resumed.optimization_journal, previous.optimization_journal);
}

#[test]
fn resume_from_names_every_mismatched_field() {
    let previous = SingleHarnessState::new(
        "task",
        "fp",
        ids(&["a"]),
        3,
        PromotionPolicyMode::ContinuousTwoStageV2,
    );

    let cases: [(&str, Result<SingleHarnessState, _>); 4] = [
        (
            "fingerprint",
            SingleHarnessState::resume_from(previous.clone(), "other", "task", 3, ""),
        ),
        (
            "task_id",
            SingleHarnessState::resume_from(previous.clone(), "fp", "other", 3, ""),
        ),
        (
            "max_iteration",
            SingleHarnessState::resume_from(previous.clone(), "fp", "task", 9, ""),
        ),
        (
            "evaluation_protocol_signature",
            SingleHarnessState::resume_from(previous, "fp", "task", 3, "v9"),
        ),
    ];
    for (field, result) in cases {
        let err = result.expect_err("resume must reject a mismatch");
        assert_eq!(err.kind(), ErrorKind::Validation);
        assert!(
            err.to_string().contains(field),
            "error `{err}` should name `{field}`"
        );
    }
}
