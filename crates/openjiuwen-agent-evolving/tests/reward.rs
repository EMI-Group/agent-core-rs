//! Integration tests for the `reward` module.
//!
//! Exercises the reward records, the normalisation variants, the
//! missing-dimension policies, the weighted aggregation, the score statistics,
//! and the pack helpers end to end. Everything is offline and deterministic:
//! no model calls, no clock, no environment.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;

use openjiuwen_agent_evolving::dataset::{Case, EvaluatedCase};
use openjiuwen_agent_evolving::reward::{
    aggregate_scores, default_pack, reward_from_score, score_evaluated_cases, DimensionScore,
    MissingDimensionPolicy, RewardAggregator, RewardDimension, RewardPack, RewardSignal,
    RewardSummary, ScoreNormalisation,
};
use openjiuwen_common::error::Error;
use serde_json::Value;

/// A dimension with an explicit weight and the default scale.
fn dimension(name: &str, weight: f64) -> RewardDimension {
    RewardDimension::new(name).with_weight(weight)
}

/// A one-entry map used for the `Case` fixtures.
fn map(key: &str, value: &str) -> BTreeMap<String, Value> {
    let mut map = BTreeMap::new();
    map.insert(key.to_owned(), Value::String(value.to_owned()));
    map
}

/// An evaluated case with a fixed identifier and score.
fn evaluated(id: &str, score: f64) -> EvaluatedCase {
    let case = Case::new(map("question", "hi"), map("answer", "hello"))
        .expect("fixture case is valid")
        .with_case_id(id);
    EvaluatedCase::new(case, None, score, "fixture").expect("fixture score is finite")
}

#[test]
fn dimension_defaults_and_builders() {
    let dimension = RewardDimension::new("accuracy");
    assert_eq!(dimension.name, "accuracy");
    assert!((dimension.weight - 1.0).abs() < f64::EPSILON);
    assert!((dimension.max_score - 1.0).abs() < f64::EPSILON);
    assert!(dimension.higher_is_better);
    assert!(dimension.validate().is_ok());

    let dimension = dimension
        .with_weight(0.4)
        .with_max_score(5.0)
        .with_higher_is_better(false);
    assert!((dimension.weight - 0.4).abs() < f64::EPSILON);
    assert!((dimension.max_score - 5.0).abs() < f64::EPSILON);
    assert!(!dimension.higher_is_better);
    assert!(dimension.validate().is_ok());
}

#[test]
fn dimension_validation_rejects_each_bad_field() {
    let cases = [
        ("blank name", RewardDimension::new("  ")),
        ("zero weight", RewardDimension::new("x").with_weight(0.0)),
        (
            "negative weight",
            RewardDimension::new("x").with_weight(-0.5),
        ),
        (
            "non-finite weight",
            RewardDimension::new("x").with_weight(f64::INFINITY),
        ),
        (
            "NaN weight",
            RewardDimension::new("x").with_weight(f64::NAN),
        ),
        (
            "zero max_score",
            RewardDimension::new("x").with_max_score(0.0),
        ),
        (
            "negative max_score",
            RewardDimension::new("x").with_max_score(-1.0),
        ),
        (
            "NaN max_score",
            RewardDimension::new("x").with_max_score(f64::NAN),
        ),
    ];
    for (label, dimension) in cases {
        let error = dimension
            .validate()
            .expect_err(&format!("{label} must be rejected"));
        assert!(
            matches!(&error, Error::Validation(_)),
            "{label}: unexpected error {error}"
        );
    }
}

#[test]
fn signal_defaults_builders_and_validation() {
    let signal = RewardSignal::new("correctness", 0.5);
    assert_eq!(signal.dimension, "correctness");
    assert!((signal.value - 0.5).abs() < f64::EPSILON);
    assert!((signal.confidence - 1.0).abs() < f64::EPSILON);
    assert!(signal.source.is_none());
    assert!(signal.evidence.is_none());
    assert!(signal.validate().is_ok());

    // Builders accept anything; `validate` is where the constraint lands.
    let built = signal
        .clone()
        .with_source("metric")
        .with_evidence("case-1")
        .with_confidence(1.5);
    assert_eq!(built.source.as_deref(), Some("metric"));
    assert_eq!(built.evidence.as_deref(), Some("case-1"));
    assert!(built.validate().is_err());

    let rejected = [
        ("blank dimension", RewardSignal::new(" ", 0.5)),
        ("NaN value", RewardSignal::new("x", f64::NAN)),
        ("infinite value", RewardSignal::new("x", f64::NEG_INFINITY)),
        (
            "negative confidence",
            RewardSignal::new("x", 0.5).with_confidence(-0.1),
        ),
        (
            "confidence above one",
            RewardSignal::new("x", 0.5).with_confidence(1.1),
        ),
        (
            "NaN confidence",
            RewardSignal::new("x", 0.5).with_confidence(f64::NAN),
        ),
    ];
    for (label, signal) in rejected {
        assert!(
            signal.validate().is_err(),
            "{label} must be rejected by validate"
        );
    }
    assert!(RewardSignal::new("x", 0.5)
        .with_confidence(0.0)
        .validate()
        .is_ok());
    assert!(RewardSignal::new("x", 0.5)
        .with_confidence(1.0)
        .validate()
        .is_ok());
}

#[test]
fn records_round_trip_through_json_with_defaults() {
    let dimension: RewardDimension = serde_json::from_str(r#"{"name": "correctness"}"#).unwrap();
    assert!((dimension.weight - 1.0).abs() < f64::EPSILON);
    assert!((dimension.max_score - 1.0).abs() < f64::EPSILON);
    assert!(dimension.higher_is_better);
    let value = serde_json::to_value(&dimension).unwrap();
    assert_eq!(value["name"], "correctness");
    let restored: RewardDimension = serde_json::from_value(value).unwrap();
    assert_eq!(restored, dimension);

    let signal: RewardSignal =
        serde_json::from_str(r#"{"dimension": "clarity", "value": 0.25}"#).unwrap();
    assert!((signal.confidence - 1.0).abs() < f64::EPSILON);
    assert!(signal.source.is_none());
    assert!(signal.evidence.is_none());
    let value = serde_json::to_value(&signal).unwrap();
    assert_eq!(value["confidence"], 1.0);
    assert_eq!(
        serde_json::from_value::<RewardSignal>(value).unwrap(),
        signal
    );

    // Unknown fields are tolerated on every record.
    let permissive: RewardDimension =
        serde_json::from_str(r#"{"name": "x", "bogus": true}"#).unwrap();
    assert_eq!(permissive.name, "x");

    let row = DimensionScore::new("correctness", 0.6, 0.6, 0.5);
    assert_eq!(row.name(), "correctness");
    assert_eq!(row.raw(), 0.6);
    assert_eq!(row.normalised(), 0.6);
    assert_eq!(row.weight(), 0.5);
    let restored: DimensionScore =
        serde_json::from_value(serde_json::to_value(&row).unwrap()).unwrap();
    assert_eq!(restored, row);

    let summary = RewardSummary::from_signals(
        &[RewardSignal::new("correctness", 0.5)],
        BTreeMap::from([("correctness".to_owned(), row)]),
        vec!["clarity".to_owned()],
        0.4,
    );
    let value = serde_json::to_value(&summary).unwrap();
    for key in ["total", "per_dimension", "missing", "num_signals", "passed"] {
        assert!(value.get(key).is_some(), "missing field {key}");
    }
    let restored: RewardSummary = serde_json::from_value(value).unwrap();
    assert_eq!(restored, summary);

    let zero: RewardSummary = serde_json::from_str("{}").unwrap();
    assert_eq!(zero, RewardSummary::zero());
}

#[test]
fn normalisation_variants_rescale_and_clamp() {
    let dimension = RewardDimension::new("quality").with_max_score(2.0);
    assert!((ScoreNormalisation::Raw.apply(&dimension, 1.5) - 0.75).abs() < 1e-9);
    assert!((ScoreNormalisation::Raw.apply(&dimension, 3.0) - 1.5).abs() < 1e-9);
    assert!((ScoreNormalisation::Clamp.apply(&dimension, 3.0) - 1.0).abs() < 1e-9);
    assert!(ScoreNormalisation::Clamp.apply(&dimension, -1.0).abs() < 1e-9);

    let min_max = ScoreNormalisation::MinMax {
        min: 10.0,
        max: 30.0,
    };
    assert!((min_max.apply(&dimension, 20.0) - 0.5).abs() < 1e-9);
    assert!((min_max.apply(&dimension, 40.0) - 1.0).abs() < 1e-9);
    assert!(min_max.apply(&dimension, 0.0).abs() < 1e-9);

    assert_eq!(ScoreNormalisation::default(), ScoreNormalisation::Clamp);
}

#[test]
fn normalisation_inverts_lower_is_better_dimensions() {
    let dimension = RewardDimension::new("latency")
        .with_max_score(2.0)
        .with_higher_is_better(false);
    assert!((ScoreNormalisation::Clamp.apply(&dimension, 0.5) - 0.75).abs() < 1e-9);
    assert!((ScoreNormalisation::Raw.apply(&dimension, 0.5) - 0.75).abs() < 1e-9);
    let min_max = ScoreNormalisation::MinMax { min: 0.0, max: 2.0 };
    assert!((min_max.apply(&dimension, 0.5) - 0.75).abs() < 1e-9);
    assert!((min_max.apply(&dimension, 2.0)).abs() < 1e-9);
    // A raw value worse than the scale inverts below zero and is clamped to 0.
    assert!(ScoreNormalisation::Clamp.apply(&dimension, 4.0).abs() < f64::EPSILON);
}

#[test]
fn normalisation_never_yields_nan_or_infinity() {
    let dimension = RewardDimension::new("quality");
    for normalisation in [
        ScoreNormalisation::Raw,
        ScoreNormalisation::Clamp,
        ScoreNormalisation::MinMax { min: 0.0, max: 1.0 },
    ] {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let normalised = normalisation.apply(&dimension, value);
            assert!(
                normalised.is_finite(),
                "{normalisation:?} leaked {normalised}"
            );
            assert!(normalised.abs() < f64::EPSILON);
        }
        let zero_scale = RewardDimension::new("quality").with_max_score(0.0);
        assert!(normalisation.apply(&zero_scale, 0.5).is_finite());
        assert!(normalisation.apply(&zero_scale, 0.5).abs() < f64::EPSILON);
        let nan_scale = RewardDimension::new("quality").with_max_score(f64::NAN);
        assert!(normalisation.apply(&nan_scale, 0.5).abs() < f64::EPSILON);
    }

    let tiny_scale = RewardDimension::new("quality").with_max_score(f64::MIN_POSITIVE);
    assert!(ScoreNormalisation::Raw
        .apply(&tiny_scale, f64::MAX)
        .is_finite());

    let degenerate = ScoreNormalisation::MinMax { min: 5.0, max: 5.0 };
    assert!(degenerate.apply(&dimension, 5.0).abs() < f64::EPSILON);
    let non_finite = ScoreNormalisation::MinMax {
        min: f64::NAN,
        max: 1.0,
    };
    assert!(non_finite.apply(&dimension, 0.5).abs() < f64::EPSILON);
    let nan_scale = RewardDimension::new("quality").with_max_score(f64::NAN);
    assert!(ScoreNormalisation::Clamp
        .apply(&nan_scale, f64::NAN)
        .is_finite());
}

#[test]
fn aggregator_defaults_and_builders() {
    let aggregator = RewardAggregator::new([dimension("correctness", 0.6)]);
    assert_eq!(aggregator.dimensions().len(), 1);
    assert_eq!(aggregator.normalisation(), ScoreNormalisation::Clamp);
    assert_eq!(aggregator.policy(), MissingDimensionPolicy::Skip);
    assert!((aggregator.threshold() - 0.6).abs() < f64::EPSILON);

    // Duplicate names keep the last definition.
    let duplicated =
        RewardAggregator::new([dimension("correctness", 0.2), dimension("correctness", 0.9)]);
    assert_eq!(duplicated.dimensions().len(), 1);
    assert!((duplicated.dimensions()["correctness"].weight - 0.9).abs() < f64::EPSILON);

    let aggregator = aggregator
        .with_normalisation(ScoreNormalisation::Raw)
        .with_policy(MissingDimensionPolicy::Zero)
        .with_threshold(0.25);
    assert_eq!(aggregator.normalisation(), ScoreNormalisation::Raw);
    assert_eq!(aggregator.policy(), MissingDimensionPolicy::Zero);
    assert!((aggregator.threshold() - 0.25).abs() < f64::EPSILON);
}

#[test]
fn weighted_total_is_hand_computed() {
    let aggregator = RewardAggregator::new([
        dimension("a", 0.5),
        dimension("b", 0.3),
        dimension("c", 0.2),
    ])
    .with_threshold(0.6);
    let signals = [
        RewardSignal::new("a", 0.9),
        RewardSignal::new("b", 0.5),
        RewardSignal::new("c", 0.1),
    ];
    let summary = aggregator.aggregate(&signals).unwrap();
    // (0.5*0.9 + 0.3*0.5 + 0.2*0.1) / (0.5+0.3+0.2) = 0.62
    assert!((summary.total() - 0.62).abs() < 1e-9);
    assert!(summary.passed());
    assert!(
        summary.missing().is_empty(),
        "expected no missing dimensions"
    );
    assert_eq!(summary.num_signals(), 3);
    assert!((summary.dimension("b").unwrap().normalised() - 0.5).abs() < 1e-9);

    let strict = aggregator.with_threshold(0.7);
    assert!(!strict.aggregate(&signals).unwrap().passed());

    // `total >= threshold` is inclusive.
    let exact = RewardAggregator::new([dimension("a", 1.0)]).with_threshold(0.5);
    assert!(exact
        .aggregate(&[RewardSignal::new("a", 0.5)])
        .unwrap()
        .passed());
}

#[test]
fn duplicate_signals_combine_by_confidence_and_fall_back() {
    let aggregator = RewardAggregator::new([dimension("correctness", 1.0)]);
    let signals = [
        RewardSignal::new("correctness", 1.0).with_confidence(0.2),
        RewardSignal::new("correctness", 0.5).with_confidence(0.6),
    ];
    let summary = aggregator.aggregate(&signals).unwrap();
    // (1.0*0.2 + 0.5*0.6) / 0.8 = 0.625
    assert!((summary.dimension("correctness").unwrap().raw() - 0.625).abs() < 1e-9);
    assert!((summary.total() - 0.625).abs() < 1e-9);

    let zero_confidence = [
        RewardSignal::new("correctness", 0.9).with_confidence(0.0),
        RewardSignal::new("correctness", 0.1).with_confidence(0.0),
    ];
    let summary = aggregator.aggregate(&zero_confidence).unwrap();
    assert!((summary.dimension("correctness").unwrap().raw() - 0.9).abs() < 1e-9);

    // A single signal passes through unchanged regardless of its confidence.
    let single = aggregator
        .aggregate(&[RewardSignal::new("correctness", 0.3).with_confidence(0.1)])
        .unwrap();
    assert!((single.total() - 0.3).abs() < 1e-9);
}

#[test]
fn undeclared_dimensions_are_counted_but_ignored() {
    let aggregator = RewardAggregator::new([dimension("correctness", 1.0)]);
    let signals = [
        RewardSignal::new("correctness", 0.5),
        RewardSignal::new("mystery", 1.0),
    ];
    let summary = aggregator.aggregate(&signals).unwrap();
    assert!((summary.total() - 0.5).abs() < 1e-9);
    assert_eq!(summary.num_signals(), 2);
    assert!(summary.dimension("mystery").is_none());
    assert!(
        summary.missing().is_empty(),
        "expected no missing dimensions"
    );
}

#[test]
fn missing_dimension_policy_fail_names_every_missing_dimension() {
    let aggregator = RewardAggregator::new([
        dimension("correctness", 0.6),
        dimension("efficiency", 0.2),
        dimension("clarity", 0.2),
    ])
    .with_policy(MissingDimensionPolicy::Fail);
    let error = aggregator.aggregate(&[]).unwrap_err();
    assert!(matches!(&error, Error::Validation(_)));
    assert!(
        error
            .to_string()
            .contains("missing reward dimensions: clarity, correctness, efficiency"),
        "unexpected message: {error}"
    );

    // A fully measured pack is accepted by the same policy.
    let measured = [
        RewardSignal::new("correctness", 0.5),
        RewardSignal::new("efficiency", 0.5),
        RewardSignal::new("clarity", 0.5),
    ];
    assert!((aggregator.aggregate(&measured).unwrap().total() - 0.5).abs() < 1e-9);
}

#[test]
fn missing_dimension_policy_zero_drags_the_total_down() {
    let aggregator =
        RewardAggregator::new([dimension("correctness", 0.6), dimension("clarity", 0.4)])
            .with_policy(MissingDimensionPolicy::Zero);
    let summary = aggregator
        .aggregate(&[RewardSignal::new("correctness", 0.8)])
        .unwrap();
    // (0.6*0.8 + 0.4*0.0) / (0.6+0.4) = 0.48
    assert!((summary.total() - 0.48).abs() < 1e-9);
    assert_eq!(summary.missing(), ["clarity".to_owned()]);
    let clarity = summary.dimension("clarity").unwrap();
    assert!(clarity.raw().abs() < f64::EPSILON);
    assert!(clarity.normalised().abs() < f64::EPSILON);
    assert!((clarity.weight() - 0.4).abs() < f64::EPSILON);
}

#[test]
fn missing_dimension_policy_skip_excludes_and_lists() {
    let aggregator =
        RewardAggregator::new([dimension("correctness", 0.6), dimension("clarity", 0.4)]);
    let summary = aggregator
        .aggregate(&[RewardSignal::new("correctness", 0.8)])
        .unwrap();
    assert!((summary.total() - 0.8).abs() < 1e-9);
    assert_eq!(summary.missing(), ["clarity".to_owned()]);
    assert!(summary.dimension("clarity").is_none());
}

#[test]
fn invalid_signals_are_reported_with_index_and_dimension() {
    let aggregator = RewardAggregator::new([dimension("correctness", 1.0)]);
    let signals = [
        RewardSignal::new("correctness", 0.5),
        RewardSignal::new("correctness", f64::NAN),
        RewardSignal::new("correctness", 0.5).with_confidence(2.0),
    ];
    let error = aggregator.aggregate(&signals).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("index 1"), "unexpected message: {message}");
    assert!(
        message.contains("correctness"),
        "unexpected message: {message}"
    );
    assert!(message.contains("value"), "unexpected message: {message}");

    let error = aggregator
        .aggregate(&[RewardSignal::new("correctness", 0.5).with_confidence(2.0)])
        .unwrap_err();
    assert!(
        error.to_string().contains("confidence"),
        "unexpected message: {error}"
    );
}

#[test]
fn aggregate_scores_statistics() {
    let stats = aggregate_scores(&[0.2, 0.4, 0.6]);
    assert_eq!(stats.count, 3);
    assert!((stats.mean - 0.4).abs() < 1e-9);
    assert!((stats.min - 0.2).abs() < 1e-9);
    assert!((stats.max - 0.6).abs() < 1e-9);
    // Population standard deviation: sqrt(0.08 / 3).
    assert!((stats.std_dev - 0.163_299_316_185_545_2).abs() < 1e-9);

    let empty = aggregate_scores(&[]);
    assert_eq!(empty.count, 0);
    assert!(empty.mean.abs() < f64::EPSILON);
    assert!(empty.min.abs() < f64::EPSILON);
    assert!(empty.max.abs() < f64::EPSILON);
    assert!(empty.std_dev.abs() < f64::EPSILON);

    let constant = aggregate_scores(&[0.5, 0.5, 0.5]);
    assert!((constant.mean - 0.5).abs() < 1e-9);
    assert!(constant.std_dev.abs() < 1e-9);

    let sanitised = aggregate_scores(&[f64::NAN, 1.0, f64::NEG_INFINITY]);
    assert_eq!(sanitised.count, 3);
    assert!((sanitised.mean - 1.0 / 3.0).abs() < 1e-9);
    assert!(sanitised.min.abs() < f64::EPSILON);
    assert!((sanitised.max - 1.0).abs() < f64::EPSILON);
    assert!(sanitised.std_dev.is_finite());
}

#[test]
fn default_pack_weights_and_validation() {
    let pack = default_pack();
    assert_eq!(pack, RewardPack::default_pack());
    assert_eq!(pack.name, "default");
    assert_eq!(pack.version, "1");
    assert!((pack.passing_score - 0.6).abs() < f64::EPSILON);
    let weights: Vec<(&str, f64)> = pack
        .dimensions
        .iter()
        .map(|dimension| (dimension.name.as_str(), dimension.weight))
        .collect();
    assert_eq!(
        weights,
        [("correctness", 0.6), ("efficiency", 0.2), ("clarity", 0.2)]
    );
    let total_weight: f64 = pack
        .dimensions
        .iter()
        .map(|dimension| dimension.weight)
        .sum();
    assert!((total_weight - 1.0).abs() < 1e-9);
    assert!(pack.validate().is_ok());

    let aggregator = pack.aggregator();
    assert_eq!(aggregator.normalisation(), ScoreNormalisation::Clamp);
    assert_eq!(aggregator.policy(), MissingDimensionPolicy::Skip);
    assert!((aggregator.threshold() - pack.passing_score).abs() < f64::EPSILON);
}

#[test]
fn pack_validation_rejects_each_bad_field() {
    let mut pack = RewardPack::default_pack();
    pack.name = "  ".to_owned();
    assert!(pack.validate().is_err());

    let mut pack = RewardPack::default_pack();
    pack.version = String::new();
    assert!(pack.validate().is_err());

    let mut pack = RewardPack::default_pack();
    pack.dimensions.clear();
    assert!(pack.validate().is_err());

    let mut pack = RewardPack::default_pack();
    pack.dimensions = vec![dimension("correctness", 1.0), dimension("correctness", 1.0)];
    let error = pack.validate().unwrap_err();
    assert!(
        error.to_string().contains("duplicate"),
        "unexpected message: {error}"
    );

    let mut pack = RewardPack::default_pack();
    pack.dimensions = vec![RewardDimension::new(" ")];
    assert!(pack.validate().is_err());

    for passing_score in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
        let mut pack = RewardPack::default_pack();
        pack.passing_score = passing_score;
        assert!(
            pack.validate().is_err(),
            "passing_score {passing_score} must be rejected"
        );
    }
    for passing_score in [0.0, 1.0] {
        let mut pack = RewardPack::default_pack();
        pack.passing_score = passing_score;
        assert!(
            pack.validate().is_ok(),
            "passing_score {passing_score} must be accepted"
        );
    }
}

#[test]
fn pack_json_round_trip_and_aggregation() {
    let pack = RewardPack::default_pack();
    let value = serde_json::to_value(&pack).unwrap();
    for key in ["name", "version", "dimensions", "passing_score"] {
        assert!(value.get(key).is_some(), "missing field {key}");
    }
    let restored: RewardPack = serde_json::from_value(value).unwrap();
    assert_eq!(restored, pack);
    // `passing_score` has a serde default, so a pack without it still parses.
    let partial: RewardPack = serde_json::from_str(
        r#"{"name": "custom", "version": "2", "dimensions": [{"name": "correctness"}]}"#,
    )
    .unwrap();
    assert!((partial.passing_score - 0.6).abs() < f64::EPSILON);

    let summary = pack
        .aggregate(&[
            RewardSignal::new("correctness", 1.0),
            RewardSignal::new("efficiency", 1.0),
            RewardSignal::new("clarity", 1.0),
        ])
        .unwrap();
    assert!((summary.total() - 1.0).abs() < 1e-9);
    assert!(summary.passed());
}

#[test]
fn score_evaluated_cases_over_a_batch() {
    let cases = [evaluated("a", 0.8), evaluated("b", 0.4)];
    let pack = RewardPack::default_pack();
    let summary = score_evaluated_cases(&cases, &pack);
    // correctness = (0.8 + 0.4) / 2 = 0.6; efficiency and clarity are skipped.
    assert!((summary.total() - 0.6).abs() < 1e-9);
    assert!(summary.passed());
    assert_eq!(
        summary.num_signals(),
        3,
        "two per-case signals plus pass_rate"
    );
    assert_eq!(
        summary.missing(),
        ["clarity".to_owned(), "efficiency".to_owned()]
    );
    let correctness = summary.dimension("correctness").unwrap();
    assert!((correctness.raw() - 0.6).abs() < 1e-9);
    assert!((correctness.weight() - 0.6).abs() < 1e-9);
}

#[test]
fn score_evaluated_cases_uses_the_pack_threshold_for_passing() {
    let pack = RewardPack::default_pack();
    let passing = score_evaluated_cases(&[evaluated("a", 1.0), evaluated("b", 1.0)], &pack);
    assert!((passing.total() - 1.0).abs() < 1e-9);
    assert!(passing.passed());

    let failing = score_evaluated_cases(&[evaluated("a", 0.1), evaluated("b", 0.1)], &pack);
    assert!((failing.total() - 0.1).abs() < 1e-9);
    assert!(!failing.passed());

    // A pack without `correctness` scores the first declared dimension instead.
    let mut custom = RewardPack::default_pack();
    custom.dimensions = vec![dimension("efficiency", 1.0)];
    custom.passing_score = 0.5;
    let summary = score_evaluated_cases(&[evaluated("a", 0.9), evaluated("b", 0.5)], &custom);
    assert!((summary.dimension("efficiency").unwrap().raw() - 0.7).abs() < 1e-9);
    assert!((summary.total() - 0.7).abs() < 1e-9);
    assert!(summary.passed());
}

#[test]
fn score_evaluated_cases_empty_input_is_zero() {
    let summary = score_evaluated_cases(&[], &RewardPack::default_pack());
    assert!(summary.total().abs() < f64::EPSILON);
    assert!(!summary.passed());
    assert_eq!(summary.num_signals(), 0);
    assert!(summary.per_dimension().is_empty(), "expected no dimensions");
    assert_eq!(
        summary.missing(),
        [
            "clarity".to_owned(),
            "correctness".to_owned(),
            "efficiency".to_owned()
        ]
    );
}

#[test]
fn reward_from_score_selects_dimension_and_sanitises() {
    let pack = RewardPack::default_pack();
    let signal = reward_from_score(0.75, &pack);
    assert_eq!(signal.dimension, "correctness");
    assert_eq!(signal.source.as_deref(), Some("evaluated_case"));
    assert!(signal.evidence.is_none());
    assert!((signal.value - 0.75).abs() < 1e-9);
    assert!((signal.confidence - 1.0).abs() < f64::EPSILON);
    assert!(signal.validate().is_ok());

    let non_finite = reward_from_score(f64::NAN, &pack);
    assert!(non_finite.value.abs() < f64::EPSILON);
    assert!(non_finite.validate().is_ok());

    let mut without_correctness = pack.clone();
    without_correctness.dimensions = vec![dimension("clarity", 0.5), dimension("efficiency", 0.5)];
    assert_eq!(
        reward_from_score(0.5, &without_correctness).dimension,
        "clarity"
    );

    let mut empty = pack;
    empty.dimensions.clear();
    assert_eq!(reward_from_score(0.5, &empty).dimension, "correctness");
}
