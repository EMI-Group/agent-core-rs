//! Offline integration tests for the `ttse` module.
//!
//! Covers the config, the tip record schema, the bank (dedupe, ranking, pruning),
//! the keyword classifier, success detection, the tip parser, induction,
//! dreaming, consultation rendering and the end-to-end engine.
//!
//! Everything here is deterministic: no clock, model or network. The one model
//! path is driven by `ScriptedModel`, which never performs I/O.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::assert_is_empty
)]

use std::sync::Arc;

use openjiuwen_agent_evolving::signal::{EvolutionSignal, EvolutionTarget};
use openjiuwen_agent_evolving::trajectory::{
    StepKind, StepStatus, Trajectory, TrajectoryOutcome, TrajectoryStep,
};
use openjiuwen_agent_evolving::ttse::{
    consult, render_tips, tip_id_for, AddOutcome, ClassificationRule, ConsultResult, DreamReport,
    Dreamer, SuccessAssessment, SuccessDetector, TipBank, TipBankStats, TipCategory, TipClassifier,
    TipInducer, TipParser, TipRecord, TtseConfig, TtseEngine, DEFAULT_MAX_RULE_CHARS,
};
use openjiuwen_core::foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_test_support::ScriptedModel;
use serde_json::json;

// ---- helpers ---------------------------------------------------------------

/// Floating-point equality with the epsilon the module's scores are computed at.
fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

fn step(kind: StepKind, content: &str) -> TrajectoryStep {
    TrajectoryStep::new(kind, content)
}

fn tool_call(name: &str) -> TrajectoryStep {
    TrajectoryStep::new(StepKind::ToolCall, format!("call {name}")).with_tool_name(name)
}

fn error_step(kind: StepKind, content: &str) -> TrajectoryStep {
    step(kind, content).with_status(StepStatus::Error)
}

fn trajectory(id: &str, outcome: TrajectoryOutcome, steps: Vec<TrajectoryStep>) -> Trajectory {
    let mut trajectory = Trajectory::new(id)
        .expect("valid trajectory id")
        .with_outcome(outcome);
    for step in steps {
        trajectory.push(step);
    }
    trajectory
}

fn signal(excerpt: &str, confidence: f64) -> EvolutionSignal {
    EvolutionSignal::new("execution_failure", EvolutionTarget::Other, excerpt)
        .expect("valid signal")
        .with_confidence(confidence)
        .expect("valid confidence")
}

fn weighted_signal(excerpt: &str, confidence: f64, weight: f64) -> EvolutionSignal {
    signal(excerpt, confidence)
        .with_weight(weight)
        .expect("valid weight")
}

fn tip(rule: &str, category: TipCategory, weight: f64) -> TipRecord {
    TipRecord::new(rule, category)
        .expect("valid rule")
        .with_weight(weight)
        .expect("valid weight")
}

fn config() -> TtseConfig {
    TtseConfig::default()
}

/// The default config with `f` applied, for the one-field tweaks below.
fn config_with(f: impl FnOnce(&mut TtseConfig)) -> TtseConfig {
    let mut config = TtseConfig::default();
    f(&mut config);
    config
}

/// A failed run with one tool error (`shell`) and one plain error step.
fn failed_trajectory() -> Trajectory {
    trajectory(
        "t-fail",
        TrajectoryOutcome::Failure,
        vec![
            tool_call("cargo"),
            error_step(StepKind::ToolResult, "command not found").with_tool_name("shell"),
            error_step(StepKind::Message, "permission denied"),
        ],
    )
}

/// The one signal the engine tests feed in; it makes a `signal`-sourced tip.
fn learning_signals() -> [EvolutionSignal; 1] {
    [weighted_signal("confirm before deleting files", 1.0, 0.5)]
}

/// An engine that already learned from one failed run and one signal.
async fn observed_engine() -> TtseEngine {
    let mut engine = TtseEngine::new(config()).expect("valid config");
    engine
        .observe(&[failed_trajectory()], &learning_signals())
        .await
        .expect("observe succeeds");
    engine
}

/// The rules of `tips`, in order.
fn rules(tips: &[TipRecord]) -> Vec<&str> {
    tips.iter().map(|tip| tip.rule.as_str()).collect()
}

/// A bank of `Tool` tips, one per rule, with the default threshold.
fn bank_of(rules: &[&str]) -> TipBank {
    let mut bank = TipBank::with_capacity(16);
    for rule in rules {
        bank.add(tip(rule, TipCategory::Tool, 1.0));
    }
    bank
}

// ---- config ----------------------------------------------------------------

#[test]
fn config_defaults_match_the_spec() {
    let config = config();
    assert_eq!(config.store_path, ".ttse/bank.json");
    assert_eq!(config.capacity, 512);
    assert!(approx(config.dedup_threshold, 0.88));
    assert!(config.inject_enabled);
    assert!(config.evolve_enabled);
    assert!(approx(config.success_threshold, 0.999));
    assert_eq!(config.detect_min_tool_calls, 5);
    assert_eq!(config.consult_max_chars, 8_000);
    assert_eq!(config.consult_max_rules, 40);
    assert_eq!(config.consult_top_k, 8);
    assert!(config.dream_enabled);
    assert_eq!(config.dream_interval, 20);
    assert_eq!(config.dream_min_rules, 8);
    assert_eq!(config.dream_ttl_ticks, 90);
    assert_eq!(config.dream_cluster_min_size, 2);
    assert!(approx(config.dream_merge_threshold, 0.72));
    assert!(config.dream_prune_enabled);
    assert_eq!(config.max_tips_per_observation, 8);
    assert_eq!(config.consult_limit(), 8);
    assert!(config.validate().is_ok());
}

#[test]
fn config_validate_rejects_each_bad_class() {
    let invalid = [
        (
            "dedup_threshold",
            config_with(|config| config.dedup_threshold = 1.5),
        ),
        (
            "success_threshold",
            config_with(|config| config.success_threshold = -0.1),
        ),
        (
            "dream_merge_threshold",
            config_with(|config| config.dream_merge_threshold = f64::NAN),
        ),
        ("capacity", config_with(|config| config.capacity = 0)),
        (
            "dream_interval",
            config_with(|config| config.dream_interval = 0),
        ),
        (
            "store_path",
            config_with(|config| config.store_path = "   ".to_owned()),
        ),
    ];
    for (field, bad) in invalid {
        let error = bad.validate().expect_err(field);
        assert!(
            error.to_string().contains(field),
            "`{error}` must name `{field}`"
        );
    }

    // A zero TTL is legal: it disables TTL pruning instead of being invalid.
    assert!(config_with(|config| config.dream_ttl_ticks = 0)
        .validate()
        .is_ok());
}

#[test]
fn config_serde_fills_defaults_and_round_trips() {
    let partial: TtseConfig = serde_json::from_value(json!({
        "capacity": 10,
        "unknown_field": 1,
    }))
    .expect("a partial config loads");
    assert_eq!(partial.capacity, 10);
    assert!(approx(partial.dedup_threshold, 0.88));
    assert_eq!(partial.consult_limit(), 8);

    let original = config_with(|config| {
        config.capacity = 7;
        config.dream_ttl_ticks = 0;
        config.dedup_threshold = 0.5;
    });
    let encoded = serde_json::to_string(&original).expect("serializable");
    let restored: TtseConfig = serde_json::from_str(&encoded).expect("deserializable");
    assert_eq!(restored, original);
}

// ---- models ----------------------------------------------------------------

#[test]
fn tip_record_trims_and_derives_a_deterministic_id() {
    let first = TipRecord::new("  use rg  ", TipCategory::Tool).expect("valid rule");
    assert_eq!(first.rule, "use rg");
    assert_eq!(first.tip_id, tip_id_for(TipCategory::Tool, "use rg"));
    assert_eq!(first.tip_id.len(), 16);

    let again = TipRecord::new("use rg", TipCategory::Tool).expect("valid rule");
    assert_eq!(first, again);

    let other_category = TipRecord::new("use rg", TipCategory::Pitfall).expect("valid rule");
    assert_ne!(other_category.tip_id, first.tip_id);
    assert_eq!(
        other_category.tip_id,
        tip_id_for(TipCategory::Pitfall, "use rg")
    );

    assert!(TipRecord::new("   ", TipCategory::Tool).is_err());
}

#[test]
fn tip_record_score_and_success_rate() {
    let fresh = tip("use rg instead of grep", TipCategory::Tool, 1.0);
    assert_eq!(fresh.total_observations(), 0);
    assert!(approx(fresh.success_rate(), 0.0));
    assert!(approx(fresh.score(), 0.5));

    let observed = fresh.clone().with_observation(3, 2, 1);
    assert_eq!(observed.total_observations(), 3);
    assert!(approx(observed.success_rate(), 2.0 / 3.0));
    // 1.0 * (0.5 + 2/3) = 7/6
    assert!(approx(observed.score(), 7.0 / 6.0));

    let doubled = tip("use rg instead of grep", TipCategory::Tool, 2.0).with_observation(3, 2, 1);
    // 2.0 * (0.5 + 2/3) = 7/3
    assert!(approx(doubled.score(), 7.0 / 3.0));

    assert!(fresh.clone().with_weight(0.0).is_err());
    assert!(fresh.with_weight(f64::NAN).is_err());
}

#[test]
fn tip_record_normalizes_keywords() {
    let tip = TipRecord::new("use rg", TipCategory::Tool)
        .expect("valid rule")
        .with_keywords(vec![" Tool ".to_owned(), "tool".to_owned(), String::new()]);
    assert_eq!(tip.keywords, vec!["tool".to_owned()]);

    let sorted = TipRecord::new("use rg", TipCategory::Tool)
        .expect("valid rule")
        .with_keywords(vec![
            "zeta".to_owned(),
            "Alpha".to_owned(),
            "beta".to_owned(),
        ]);
    assert_eq!(sorted.keywords, vec!["alpha", "beta", "zeta"]);
}

// ---- bank ------------------------------------------------------------------

#[test]
fn bank_merges_similar_tips_above_the_threshold() {
    // Token Jaccard is 4/5 = 0.8, so the threshold sits below that.
    let mut bank = TipBank::new(512, 0.75);
    let first = tip("check the working directory", TipCategory::Tool, 1.0)
        .with_keywords(vec!["Directory".to_owned(), "check".to_owned()])
        .with_observation(1, 1, 0);
    let second = tip("check the working directory first", TipCategory::Tool, 1.0)
        .with_keywords(vec!["first".to_owned(), " check ".to_owned()])
        .with_observation(2, 1, 1);

    assert_eq!(
        bank.add(first.clone()),
        AddOutcome::Inserted(first.tip_id.clone())
    );
    match bank.add(second) {
        AddOutcome::Merged { into, similarity } => {
            assert_eq!(into, first.tip_id);
            // `{check, the, working, directory}` against its five-token variant.
            assert!(approx(similarity, 0.8), "similarity {similarity}");
        }
        other @ AddOutcome::Inserted(_) => panic!("expected a merge, got {other:?}"),
    }

    assert_eq!(bank.len(), 1);
    let merged = bank.get(&first.tip_id).expect("representative kept");
    assert_eq!(merged.rule, "check the working directory first");
    assert_eq!(merged.keywords, vec!["check", "directory", "first"]);
    assert_eq!((merged.hits, merged.successes, merged.failures), (3, 2, 1));
}

#[test]
fn bank_inserts_dissimilar_tips() {
    let mut bank = TipBank::new(512, 0.88);
    let first = tip("use rg for searching", TipCategory::Tool, 1.0);
    let second = tip("confirm before deleting files", TipCategory::Workflow, 1.0);

    assert_eq!(
        bank.add(first.clone()),
        AddOutcome::Inserted(first.tip_id.clone())
    );
    match bank.add(second.clone()) {
        AddOutcome::Inserted(tip_id) => assert_eq!(tip_id, second.tip_id),
        other @ AddOutcome::Merged { .. } => panic!("expected an insert, got {other:?}"),
    }
    assert_eq!(bank.len(), 2);
}

#[test]
fn bank_threshold_boundary_merges_only_identical_token_sets() {
    let mut identical = TipBank::new(16, 1.0);
    let rule = tip("run the tests", TipCategory::Workflow, 1.0);
    assert!(matches!(
        identical.add(rule.clone()),
        AddOutcome::Inserted(_)
    ));
    match identical.add(rule.clone()) {
        AddOutcome::Merged { similarity, .. } => assert!(approx(similarity, 1.0)),
        other @ AddOutcome::Inserted(_) => panic!("expected a merge, got {other:?}"),
    }
    assert_eq!(identical.len(), 1);

    // `{run, the, tests}` against `{run, all, tests}` shares 2 of 4 tokens (0.5),
    // so a 1.0 threshold keeps both.
    let mut distinct = TipBank::new(16, 1.0);
    distinct.add(tip("run the tests", TipCategory::Workflow, 1.0));
    let outcome = distinct.add(tip("run all tests", TipCategory::Workflow, 1.0));
    assert!(matches!(outcome, AddOutcome::Inserted(_)));
    assert_eq!(distinct.len(), 2);
}

#[test]
fn bank_ticks_and_observations() {
    let mut bank = TipBank::new(16, 0.99);
    assert_eq!(bank.tick(), 0);
    assert_eq!(bank.increment_tick(), 1);

    let stored = tip("use rg instead of grep", TipCategory::Tool, 1.0);
    bank.add(stored.clone());
    let record = bank.get(&stored.tip_id).expect("stored");
    assert_eq!(record.created_tick, 1);
    assert_eq!(record.updated_tick, 1);

    assert_eq!(bank.increment_tick(), 2);
    assert!(bank.observe(&stored.tip_id, true));
    assert!(bank.observe(&stored.tip_id, false));
    assert!(!bank.observe("no-such-tip", true));

    let record = bank.get(&stored.tip_id).expect("stored");
    assert_eq!((record.hits, record.successes, record.failures), (2, 1, 1));
    assert_eq!(record.updated_tick, 2);
    assert!(approx(record.success_rate(), 0.5));
}

#[test]
fn bank_top_orders_by_score_then_id() {
    let mut bank = TipBank::with_capacity(8);
    let heavy = tip("use rg instead of grep", TipCategory::Tool, 2.0);
    let light_a = tip("confirm before deleting files", TipCategory::Workflow, 1.0);
    let light_b = tip(
        "warm the cache before running",
        TipCategory::Environment,
        1.0,
    );
    for record in [&heavy, &light_a, &light_b] {
        bank.add(record.clone());
    }

    let ranked = bank.top(3, None);
    assert_eq!(ranked.len(), 3);
    // 2.0 * 0.5 = 1.0, the two others 1.0 * 0.5 = 0.5.
    assert_eq!(ranked[0].tip_id, heavy.tip_id);
    let (second, third) = if light_a.tip_id < light_b.tip_id {
        (&light_a, &light_b)
    } else {
        (&light_b, &light_a)
    };
    assert_eq!(ranked[1].tip_id, second.tip_id);
    assert_eq!(ranked[2].tip_id, third.tip_id);
    assert_eq!(bank.top(2, None).len(), 2);
}

#[test]
fn bank_top_prefers_the_query_match() {
    let mut bank = TipBank::with_capacity(8);
    let matching = tip("check the working directory", TipCategory::Tool, 1.0);
    // A heavier tip without any query overlap: 1.2 * 0.5 = 0.6.
    let heavier = tip("use rg instead of grep", TipCategory::Tool, 1.2);
    bank.add(matching.clone());
    bank.add(heavier.clone());

    // Without a query the heavier tip wins (0.6 > 0.5).
    let plain = bank.top(2, None);
    assert_eq!(plain[0].tip_id, heavier.tip_id);
    assert_eq!(plain[1].tip_id, matching.tip_id);

    // With the query the match wins: 0.5 * (1 + 2/4) = 0.75 > 0.6.
    let queried = bank.top(2, Some("working directory"));
    assert_eq!(queried[0].tip_id, matching.tip_id);
    assert_eq!(queried[1].tip_id, heavier.tip_id);
}

#[test]
fn bank_category_lookup_len_and_remove() {
    let mut bank = TipBank::with_capacity(8);
    assert!(bank.is_empty());
    let tool = tip("use rg instead of grep", TipCategory::Tool, 1.0);
    let pitfall = tip("never delete the lock file", TipCategory::Pitfall, 1.0);
    bank.add(tool.clone());
    bank.add(pitfall.clone());
    assert_eq!(bank.len(), 2);
    assert!(!bank.is_empty());

    assert_eq!(bank.by_category(TipCategory::Tool).len(), 1);
    assert_eq!(bank.by_category(TipCategory::Tool)[0].tip_id, tool.tip_id);
    assert!(bank.by_category(TipCategory::Other).is_empty());

    assert_eq!(bank.get(&pitfall.tip_id), Some(&pitfall));
    let removed = bank.remove(&pitfall.tip_id).expect("removed");
    assert_eq!(removed, pitfall);
    assert_eq!(bank.len(), 1);
    assert!(bank.remove(&pitfall.tip_id).is_none());
}

#[test]
fn bank_prune_decay_and_stats() {
    let mut bank = TipBank::with_capacity(8);
    // `add` stamps `updated_tick` with the current tick, so the stale record is
    // pushed directly and the other two are stored at tick 10.
    bank.tick = 10;
    let stale = tip("stale rule one", TipCategory::Tool, 1.0).with_ticks(0, 0);
    let light = tip("light rule two", TipCategory::Tool, 0.01);
    let healthy = tip("healthy rule three", TipCategory::Tool, 1.0);
    bank.add(light.clone());
    bank.add(healthy.clone());
    bank.tips.push(stale.clone());
    assert_eq!(bank.len(), 3);

    let pruned = bank.prune(&config_with(|config| config.dream_ttl_ticks = 5));
    // `stale` is 10 ticks old (> 5) and `light` weighs 0.01 (< the 0.05 floor).
    assert_eq!(pruned, 2);
    assert_eq!(bank.len(), 1);
    assert_eq!(bank.tips()[0].tip_id, healthy.tip_id);

    let mut decaying = TipBank::with_capacity(8);
    decaying.add(tip("first rule", TipCategory::Tool, 1.0));
    decaying.add(tip("second rule", TipCategory::Tool, 2.0));
    decaying.decay(0.5);
    let weights: Vec<f64> = decaying.tips().iter().map(|tip| tip.weight).collect();
    assert_eq!(weights.len(), 2);
    assert!(approx(weights[0], 0.5));
    assert!(approx(weights[1], 1.0));

    let mut mixed = TipBank::with_capacity(8);
    mixed.add(tip("first rule", TipCategory::Tool, 1.0));
    mixed.add(tip("second rule", TipCategory::Tool, 3.0));
    mixed.add(tip("third rule", TipCategory::Pitfall, 2.0));
    let stats: TipBankStats = mixed.stats();
    assert_eq!(stats.count, 3);
    assert!(approx(stats.mean_weight, 2.0));
    assert_eq!(stats.categories.get("tool"), Some(&2));
    assert_eq!(stats.categories.get("pitfall"), Some(&1));
    assert_eq!(stats.categories.len(), 2);
}

#[test]
fn bank_serde_shape_and_round_trip() {
    let mut bank = TipBank::new(4, 0.9);
    bank.increment_tick();
    bank.add(tip("use rg instead of grep", TipCategory::Tool, 1.0));

    let value = serde_json::to_value(&bank).expect("serializable");
    let mut keys: Vec<&str> = value
        .as_object()
        .expect("a bank is a JSON object")
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    assert_eq!(keys, vec!["capacity", "dedup_threshold", "tick", "tips"]);

    let restored: TipBank = serde_json::from_value(value).expect("deserializable");
    assert_eq!(restored, bank);
    assert_eq!(restored.tick, 1);
    assert_eq!(restored.len(), 1);
}

// ---- classifier ------------------------------------------------------------

#[test]
fn classifier_default_rules() {
    let classifier = TipClassifier::default();
    assert_eq!(
        classifier.classify("use the shell command to list files"),
        TipCategory::Tool
    );
    assert_eq!(
        classifier.classify("the command failed with a timeout error"),
        TipCategory::Pitfall
    );
    assert_eq!(
        classifier.classify("first run the tests, then deploy the pipeline"),
        TipCategory::Workflow
    );
    assert_eq!(
        classifier.classify("please explain and confirm before proceeding"),
        TipCategory::Communication
    );
    assert_eq!(
        classifier.classify("install the package into the environment path"),
        TipCategory::Environment
    );
    assert_eq!(classifier.classify(""), TipCategory::Other);
    assert!(approx(classifier.classify_scored("").1, 0.0));
}

#[test]
fn classifier_handles_cjk_keywords() {
    let classifier = TipClassifier::default();
    // `command` votes Tool and `path` votes Environment; the tie goes to the
    // earlier rule in the sorted table (Tool).
    let (category, confidence) = classifier.classify_scored("运行 command 之前先检查 path");
    assert_ne!(category, TipCategory::Other);
    assert_eq!(category, TipCategory::Tool);
    assert!(confidence > 0.0);

    let custom = TipClassifier::new(vec![ClassificationRule {
        keywords: vec!["环境".to_owned(), "路径".to_owned()],
        category: TipCategory::Environment,
        weight: 1.0,
    }]);
    let (category, confidence) = custom.classify_scored("先检查环境变量");
    assert_eq!(category, TipCategory::Environment);
    // One of the two keywords matched.
    assert!(approx(confidence, 0.5));
}

#[test]
fn classifier_rules_are_sorted_and_repeatable() {
    let classifier = TipClassifier::new(vec![
        ClassificationRule {
            keywords: vec!["shell".to_owned()],
            category: TipCategory::Tool,
            weight: 1.0,
        },
        ClassificationRule {
            keywords: vec!["timeout".to_owned()],
            category: TipCategory::Pitfall,
            weight: 2.0,
        },
    ]);
    let weights: Vec<f64> = classifier.rules.iter().map(|rule| rule.weight).collect();
    assert_eq!(weights, vec![2.0, 1.0]);
    assert_eq!(classifier.rules[0].category, TipCategory::Pitfall);

    let text = "the shell command hit a timeout";
    assert_eq!(classifier.classify(text), classifier.classify(text));
    // Pitfall: 1 * 2.0 beats Tool: 1 * 1.0.
    assert_eq!(classifier.classify(text), TipCategory::Pitfall);
}

// ---- success detector ------------------------------------------------------

#[test]
fn detector_success_outcome_ignores_errors() {
    let detector = SuccessDetector::new(5, 0.999);
    let recorded = trajectory(
        "t-ok",
        TrajectoryOutcome::Success,
        vec![
            tool_call("cargo"),
            error_step(StepKind::ToolResult, "a retried command failed"),
        ],
    );
    let assessment: SuccessAssessment = detector.detect(&recorded);
    assert!(assessment.successful);
    assert!(approx(assessment.score, 1.0));
    assert!(assessment.evidence.starts_with("outcome=success"));
    assert!(assessment.evidence.contains("errors=1"));
}

#[test]
fn detector_failure_outcome_reports_first_error() {
    let detector = SuccessDetector::new(0, 0.999);
    let recorded = trajectory(
        "t-bad",
        TrajectoryOutcome::Failure,
        vec![
            tool_call("cargo"),
            error_step(StepKind::ToolResult, "command not found: catt"),
            error_step(StepKind::Message, "second error"),
        ],
    );
    let assessment = detector.detect(&recorded);
    assert!(!assessment.successful);
    assert!(approx(assessment.score, 0.0));
    assert!(assessment.evidence.contains("outcome=failure"));
    assert!(assessment
        .evidence
        .contains("first_error=command not found: catt"));
    assert!(!assessment.evidence.contains("second error"));
}

#[test]
fn detector_unknown_requires_tool_calls_without_errors() {
    let detector = SuccessDetector::new(5, 0.999);
    let five_calls = trajectory(
        "t-unknown-5",
        TrajectoryOutcome::Unknown,
        (0..5)
            .map(|index| tool_call(&format!("tool{index}")))
            .collect(),
    );
    let assessment = detector.detect(&five_calls);
    assert!(assessment.successful);
    assert!(approx(assessment.score, 1.0));

    let four_calls = trajectory(
        "t-unknown-4",
        TrajectoryOutcome::Unknown,
        (0..4)
            .map(|index| tool_call(&format!("tool{index}")))
            .collect(),
    );
    let assessment = detector.detect(&four_calls);
    assert!(!assessment.successful);
    assert!(approx(assessment.score, 0.0));

    let mut with_error: Vec<TrajectoryStep> = (0..5)
        .map(|index| tool_call(&format!("tool{index}")))
        .collect();
    with_error.push(error_step(StepKind::Message, "boom"));
    let assessment = detector.detect(&trajectory(
        "t-unknown-error",
        TrajectoryOutcome::Unknown,
        with_error,
    ));
    assert!(!assessment.successful);
}

#[test]
fn detector_evidence_is_deterministic_and_config_backed() {
    let detector = SuccessDetector::from_config(&config());
    assert_eq!(detector.min_tool_calls(), 5);
    assert!(approx(detector.success_threshold(), 0.999));

    let recorded = trajectory(
        "t-det",
        TrajectoryOutcome::Failure,
        vec![tool_call("shell"), error_step(StepKind::ToolResult, "boom")],
    );
    let first = detector.detect(&recorded);
    let second = detector.detect(&recorded);
    assert_eq!(first, second);
    assert!(first.evidence.contains("tool_calls=1"));
    assert!(first.evidence.contains("errors=1"));
}

// ---- parser ----------------------------------------------------------------

#[test]
fn parser_json_array_honours_explicit_fields() {
    let tips = TipParser::default().parse(
        r#"[
            "use rg to search",
            {"rule": "Call the shell", "category": "tool", "keywords": ["rg"]}
        ]"#,
    );
    assert_eq!(tips.len(), 2);
    assert_eq!(tips[0].rule, "use rg to search");
    assert_eq!(tips[0].category, TipCategory::Other);
    assert_eq!(tips[1].rule, "Call the shell");
    assert_eq!(tips[1].category, TipCategory::Tool);
    assert_eq!(tips[1].keywords, vec!["rg".to_owned()]);
}

#[test]
fn parser_accepts_tips_wrapper_and_single_object() {
    let parser = TipParser::default();
    let wrapped = parser.parse(r#"{"tips": ["run cargo fmt before committing"]}"#);
    assert_eq!(wrapped.len(), 1);
    assert_eq!(wrapped[0].rule, "run cargo fmt before committing");

    let single = parser.parse(r#"{"rule": "ask the user to confirm", "type": "comm"}"#);
    assert_eq!(single.len(), 1);
    assert_eq!(single[0].category, TipCategory::Communication);
}

#[test]
fn parser_reads_a_fenced_json_block() {
    let text = "Here is what we learned:\n\n```json\n[\n  {\"rule\": \"first run the tests\", \"category\": \"workflow\"}\n]\n```\n\n- this bullet is outside the fence\n";
    let tips = TipParser::default().parse(text);
    assert_eq!(tips.len(), 1);
    assert_eq!(tips[0].rule, "first run the tests");
    assert_eq!(tips[0].category, TipCategory::Workflow);
}

#[test]
fn parser_reads_markdown_bullets_and_tags() {
    let text = "\
Tips:
- use the shell command to list files
* [pitfall] never delete the lock file
• explain the change to the user
1. first run the tests
2) then deploy the pipeline
TIP: confirm before deleting files
FACT: the workspace path comes from config
";
    let tips = TipParser::default().parse(text);
    assert_eq!(
        rules(&tips),
        vec![
            "use the shell command to list files",
            "never delete the lock file",
            "explain the change to the user",
            "first run the tests",
            "then deploy the pipeline",
            "confirm before deleting files",
            "the workspace path comes from config",
        ]
    );
    let categories: Vec<TipCategory> = tips.iter().map(|tip| tip.category).collect();
    assert_eq!(
        categories,
        vec![
            TipCategory::Tool,
            TipCategory::Pitfall,
            TipCategory::Communication,
            TipCategory::Workflow,
            TipCategory::Workflow,
            TipCategory::Workflow,
            TipCategory::Environment,
        ]
    );
}

#[test]
fn parser_falls_back_to_lines_on_truncated_json() {
    let text = "{\"tips\": [\n- use rg to search the workspace\n- confirm before deleting files\n";
    let tips = TipParser::default().parse(text);
    assert_eq!(tips.len(), 3);
    let rules: Vec<&str> = tips.iter().map(|tip| tip.rule.as_str()).collect();
    assert_eq!(rules[1], "use rg to search the workspace");
    assert_eq!(rules[2], "confirm before deleting files");
}

#[test]
fn parser_dedupes_caps_and_truncates_safely() {
    let duplicated =
        TipParser::default().parse("- use rg instead of grep\n- use rg instead of grep\n");
    assert_eq!(duplicated.len(), 1);

    let capped = TipParser::new(2, 240).parse(
        "- use rg instead of grep\n- confirm before deleting files\n- warm the cache before running\n",
    );
    assert_eq!(capped.len(), 2);

    let long_cjk = "检查工作目录并确认环境变量然后再运行测试命令并报告结果给用户以及记录日志";
    assert!(long_cjk.chars().count() > 20);
    let truncated = TipParser::new(8, 20).parse(long_cjk);
    assert_eq!(truncated.len(), 1);
    assert_eq!(truncated[0].rule.chars().count(), 20);
    assert!(truncated[0].rule.ends_with('…'));

    assert_eq!(DEFAULT_MAX_RULE_CHARS, 240);
    let from_config = TipParser::from_config(&config_with(|config| {
        config.max_tips_per_observation = 3;
    }));
    assert_eq!(from_config.max_tips(), 3);
    assert_eq!(from_config.max_rule_chars(), DEFAULT_MAX_RULE_CHARS);
}

// ---- inducer ---------------------------------------------------------------

#[tokio::test]
async fn inducer_mines_failures_and_signals() {
    let inducer = TipInducer::new(config());
    let tips = inducer
        .induce(
            &[failed_trajectory()],
            &[weighted_signal("confirm before deleting files", 1.0, 0.5)],
        )
        .await
        .expect("induction succeeds");

    assert_eq!(tips.len(), 3);
    assert_eq!(
        tips.iter().filter(|tip| tip.source == "trajectory").count(),
        2
    );
    assert_eq!(tips.iter().filter(|tip| tip.source == "signal").count(), 1);

    let shell = tips
        .iter()
        .find(|tip| tip.rule.contains("shell"))
        .expect("the shell tip");
    assert!(shell.rule.contains("Check"));
    assert!(shell.rule.contains("command not found"));

    let plain = tips
        .iter()
        .find(|tip| tip.rule.starts_with("Avoid repeating"))
        .expect("the plain error tip");
    assert!(plain.rule.contains("permission denied"));

    let from_signal = tips
        .iter()
        .find(|tip| tip.source == "signal")
        .expect("the signal tip");
    assert_eq!(from_signal.rule, "confirm before deleting files");
    assert!(approx(from_signal.weight, 1.5));

    // Categories come from the classifier and keywords from the rule tokens.
    let classifier = TipClassifier::default();
    for record in &tips {
        assert!(
            !record.keywords.is_empty(),
            "{} has no keywords",
            record.rule
        );
        assert_eq!(record.category, classifier.classify(&record.rule));
    }
}

#[tokio::test]
async fn inducer_skips_successful_and_clean_trajectories() {
    let inducer = TipInducer::new(config());

    let succeeded = trajectory(
        "t-ok",
        TrajectoryOutcome::Success,
        vec![error_step(StepKind::ToolResult, "boom")],
    );
    assert!(inducer.induce(&[succeeded], &[]).await.unwrap().is_empty());

    let clean_unknown = trajectory(
        "t-clean",
        TrajectoryOutcome::Unknown,
        vec![tool_call("cargo"), step(StepKind::Message, "done")],
    );
    assert!(inducer
        .induce(&[clean_unknown], &[])
        .await
        .unwrap()
        .is_empty());

    let failed_unknown = trajectory(
        "t-dirty",
        TrajectoryOutcome::Unknown,
        vec![error_step(StepKind::Message, "build broke")],
    );
    let tips = inducer.induce(&[failed_unknown], &[]).await.unwrap();
    assert_eq!(tips.len(), 1);
    assert_eq!(tips[0].rule, "Avoid repeating: build broke");
}

#[tokio::test]
async fn inducer_drops_low_confidence_and_duplicate_signals() {
    let inducer = TipInducer::new(config());
    assert!(inducer
        .induce(&[], &[signal("confirm before deleting files", 0.4)])
        .await
        .unwrap()
        .is_empty());

    let tips = inducer
        .induce(
            &[],
            &[
                signal("confirm before deleting files", 1.0),
                signal("confirm before deleting files", 1.0),
            ],
        )
        .await
        .unwrap();
    assert_eq!(tips.len(), 1);
    // The default signal weight is 1.0, so the tip weighs 2.0.
    assert!(approx(tips[0].weight, 2.0));
}

#[tokio::test]
async fn inducer_respects_the_tip_cap_and_is_deterministic() {
    let capped = TipInducer::new(config_with(|config| {
        config.max_tips_per_observation = 1;
    }));
    let signals = [weighted_signal("confirm before deleting files", 1.0, 0.5)];
    let tips = capped
        .induce(&[failed_trajectory()], &signals)
        .await
        .unwrap();
    assert_eq!(tips.len(), 1);

    let inducer = TipInducer::new(config());
    let first = inducer
        .induce(&[failed_trajectory()], &signals)
        .await
        .unwrap();
    let second = inducer
        .induce(&[failed_trajectory()], &signals)
        .await
        .unwrap();
    assert_eq!(first, second);
    assert_eq!(first.len(), 3);
}

#[tokio::test]
async fn inducer_uses_the_model_answer() {
    let inducer = TipInducer::new(config()).with_model(
        Arc::new(ScriptedModel::text(
            r#"{"tips":[{"rule":"use rg instead of grep","category":"tool"}]}"#,
        )) as Arc<dyn Model>,
        ModelRequestConfig::default(),
    );
    assert!(inducer.has_model());

    let tips = inducer
        .induce(&[failed_trajectory()], &[])
        .await
        .expect("induction succeeds");
    assert_eq!(tips.len(), 1);
    assert_eq!(tips[0].rule, "use rg instead of grep");
    assert_eq!(tips[0].category, TipCategory::Tool);
    assert_eq!(tips[0].source, "model");
}

#[tokio::test]
async fn inducer_falls_back_when_the_model_answer_is_junk() {
    let signals = [weighted_signal("confirm before deleting files", 1.0, 0.5)];
    let junk = TipInducer::new(config()).with_model(
        Arc::new(ScriptedModel::text("Default mock response")) as Arc<dyn Model>,
        ModelRequestConfig::default(),
    );
    let model_rules: Vec<String> = junk
        .induce(&[failed_trajectory()], &signals)
        .await
        .unwrap()
        .into_iter()
        .map(|tip| tip.rule)
        .collect();

    let offline_rules: Vec<String> = TipInducer::new(config())
        .induce(&[failed_trajectory()], &signals)
        .await
        .unwrap()
        .into_iter()
        .map(|tip| tip.rule)
        .collect();

    assert_eq!(model_rules, offline_rules);
    assert_eq!(model_rules.len(), 3);
}

// ---- dreamer ---------------------------------------------------------------

/// Two tips at 4/5 token overlap and one unrelated tip, pushed directly so the
/// bank does not dedupe the pair at its own 0.88 threshold.
fn dream_bank() -> TipBank {
    let mut bank = TipBank::with_capacity(16);
    bank.tick = 20;
    for record in [
        tip("check the working directory", TipCategory::Tool, 1.0),
        tip("check the working directory first", TipCategory::Tool, 1.0),
        tip(
            "confirm with the user before deleting",
            TipCategory::Workflow,
            1.0,
        ),
    ] {
        bank.tips.push(record);
    }
    bank
}

fn dream_config() -> TtseConfig {
    dream_config_with(|_| {})
}

/// The dream-pass tunables with `f` applied on top.
fn dream_config_with(f: impl FnOnce(&mut TtseConfig)) -> TtseConfig {
    let mut config = config_with(|config| {
        config.dream_merge_threshold = 0.72;
        config.dream_cluster_min_size = 2;
        config.dream_prune_enabled = false;
        config.dream_ttl_ticks = 90;
    });
    f(&mut config);
    config
}

#[test]
fn dreamer_merges_near_duplicates_and_is_idempotent() {
    let mut dreamer = Dreamer::new(dream_config());
    let mut bank = dream_bank();

    let report = dreamer.dream(&mut bank);
    assert_eq!(
        report,
        DreamReport {
            merged: 1,
            pruned: 0,
            promoted: 1,
            examined: 3,
        }
    );
    assert_eq!(bank.len(), 2);

    let second = dreamer.dream(&mut bank);
    assert_eq!(second.merged, 0);
    assert_eq!(second.examined, 2);
    assert_eq!(bank.len(), 2);
}

#[test]
fn dreamer_cluster_min_size_gates_the_merge() {
    let mut dreamer = Dreamer::new(dream_config_with(|config| {
        config.dream_cluster_min_size = 3;
    }));
    let mut bank = dream_bank();

    let report = dreamer.dream(&mut bank);
    assert_eq!(report.merged, 0);
    assert_eq!(report.promoted, 0);
    assert_eq!(report.examined, 3);
    assert_eq!(bank.len(), 3);
}

#[test]
fn dreamer_should_dream_requires_every_gate() {
    let dreamer = Dreamer::new(config_with(|config| {
        config.dream_interval = 20;
        config.dream_min_rules = 8;
    }));
    let mut bank = TipBank::with_capacity(16);
    bank.tick = 20;
    for index in 0..8 {
        bank.tips.push(tip(
            &format!("distinct rule {index}"),
            TipCategory::Other,
            1.0,
        ));
    }
    assert!(dreamer.should_dream(&bank));

    bank.tick = 19;
    assert!(!dreamer.should_dream(&bank), "tick 19 is not a multiple");
    bank.tick = 20;

    let removed = bank.tips.pop().expect("a tip to drop");
    assert!(!dreamer.should_dream(&bank), "seven tips are too few");
    bank.tips.push(removed);

    let disabled = Dreamer::new(config_with(|config| {
        config.dream_enabled = false;
        config.dream_min_rules = 8;
    }));
    assert!(!disabled.should_dream(&bank));
}

#[test]
fn dreamer_prunes_only_when_enabled() {
    let mut bank = TipBank::with_capacity(8);
    bank.tick = 10;
    bank.tips
        .push(tip("check the working directory", TipCategory::Tool, 1.0).with_ticks(0, 0));

    let mut keeping = Dreamer::new(config_with(|config| {
        config.dream_ttl_ticks = 5;
        config.dream_prune_enabled = false;
    }));
    let report = keeping.dream(&mut bank);
    assert_eq!(report.pruned, 0);
    assert_eq!(bank.len(), 1);

    let mut pruning = Dreamer::new(config_with(|config| {
        config.dream_ttl_ticks = 5;
        config.dream_prune_enabled = true;
    }));
    let report = pruning.dream(&mut bank);
    assert_eq!(report.pruned, 1);
    assert!(bank.is_empty());
}

#[test]
fn dreamer_keeps_the_higher_scoring_representative() {
    let mut bank = TipBank::with_capacity(8);
    bank.tick = 20;
    let strong =
        tip("check the working directory", TipCategory::Tool, 1.0).with_observation(2, 2, 0);
    let weak = tip("check the working directory first", TipCategory::Tool, 1.0);
    bank.tips.push(strong.clone());
    bank.tips.push(weak.clone());
    assert!(strong.rule.chars().count() < weak.rule.chars().count());

    let mut dreamer = Dreamer::new(dream_config());
    let report = dreamer.dream(&mut bank);

    assert_eq!((report.merged, report.promoted), (1, 1));
    assert_eq!(bank.len(), 1);
    assert_eq!(bank.tips()[0].tip_id, strong.tip_id);
    assert_eq!(bank.tips()[0].rule, "check the working directory");
    // The absorbed tip was a full observation of the rule.
    assert_eq!(
        (
            bank.tips()[0].hits,
            bank.tips()[0].successes,
            bank.tips()[0].failures
        ),
        (2, 2, 0)
    );
}

// ---- consultation ----------------------------------------------------------

fn render_bank() -> Vec<TipRecord> {
    vec![
        tip("quote every path", TipCategory::Tool, 1.0),
        tip("use rg instead of grep", TipCategory::Tool, 1.0).with_observation(4, 4, 0),
        tip("first run the tests", TipCategory::Workflow, 1.0),
        tip("never delete the lock file", TipCategory::Pitfall, 1.0),
        tip("warm the cache first", TipCategory::Environment, 1.0),
        tip("ask before deleting", TipCategory::Communication, 1.0),
        tip("note anything unusual", TipCategory::Other, 1.0),
    ]
}

#[test]
fn render_groups_by_category_and_orders_by_score() {
    let rendered = render_tips(&render_bank(), 10_000);
    assert_eq!(
        rendered,
        [
            "- [tool] use rg instead of grep",
            "- [tool] quote every path",
            "- [workflow] first run the tests",
            "- [pitfall] never delete the lock file",
            "- [environment] warm the cache first",
            "- [communication] ask before deleting",
            "- [other] note anything unusual",
        ]
        .join("\n")
    );
}

#[test]
fn render_handles_empty_input_and_zero_budget() {
    assert_eq!(render_tips(&[], 100), "");
    assert_eq!(
        render_tips(&[tip("use rg instead of grep", TipCategory::Tool, 1.0)], 0),
        ""
    );
}

#[test]
fn render_truncates_to_the_budget() {
    // The first line is 31 characters, so a 40 character budget keeps it plus
    // the newline plus seven characters of the second line, then the ellipsis.
    let rendered = render_tips(&render_bank(), 40);
    assert_eq!(rendered.chars().count(), 40);
    assert!(rendered.ends_with('…'));
    assert_eq!(
        rendered,
        format!("- [tool] use rg instead of grep\n- [tool{}", '…')
    );
}

#[test]
fn consult_respects_the_limit_without_mutating_the_bank() {
    let bank = bank_of(&[
        "use rg instead of grep",
        "confirm before deleting files",
        "warm the cache before running",
        "first run the tests",
        "note anything unusual",
    ]);
    assert_eq!(bank.len(), 5);
    let before = bank.tips().to_vec();

    let tight_k = config_with(|config| {
        config.consult_top_k = 2;
        config.consult_max_rules = 40;
    });
    let result = consult(&bank, "shell command", &tight_k);
    assert_eq!(result.tips.len(), 2);

    let tight_rules = config_with(|config| {
        config.consult_top_k = 40;
        config.consult_max_rules = 3;
    });
    let result = consult(&bank, "shell command", &tight_rules);
    assert_eq!(result.tips.len(), 3);
    assert!(!result.truncated, "three short tips fit in 8000 chars");
    assert!(result.rendered.contains("- [tool] "));

    let wide = consult(&bank, "shell command", &config());
    assert_eq!(wide.tips.len(), 5);
    assert!(!wide.truncated);
    assert_eq!(bank.tips(), before.as_slice());

    let empty: ConsultResult = consult(&TipBank::with_capacity(4), "anything", &config());
    assert!(empty.tips.is_empty());
    assert!(empty.rendered.is_empty());
}

// ---- engine ----------------------------------------------------------------

#[tokio::test]
async fn engine_new_validates_the_config() {
    let engine = TtseEngine::new(config()).expect("valid config");
    assert!(engine.bank().is_empty());
    assert_eq!(engine.consult_count(), 0);
    assert_eq!(engine.bank().tick(), 0);
    assert!(!engine.inducer().has_model());

    let error = TtseEngine::new(config_with(|config| config.capacity = 0))
        .expect_err("capacity 0 is invalid");
    assert!(error.to_string().contains("capacity"), "got {error}");
}

#[tokio::test]
async fn engine_observe_stores_and_dedupes() {
    let mut engine = TtseEngine::new(config()).expect("valid config");
    let signals = [weighted_signal("confirm before deleting files", 1.0, 0.5)];

    let added = engine
        .observe(&[failed_trajectory()], &signals)
        .await
        .expect("observe succeeds");
    assert_eq!(added.len(), 3);
    let mut ids: Vec<&str> = added.iter().map(|tip| tip.tip_id.as_str()).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), 3, "each tip is returned once");
    for record in &added {
        assert_eq!(engine.bank().get(&record.tip_id), Some(record));
    }
    assert_eq!(engine.bank().len(), 3);
    assert_eq!(engine.bank().tick(), 1);

    let again = engine
        .observe(&[failed_trajectory()], &signals)
        .await
        .expect("observe succeeds");
    assert_eq!(again.len(), 3, "merged records are returned once each");
    assert_eq!(engine.bank().len(), 3, "no rule is duplicated");
    assert_eq!(engine.bank().tick(), 2);
}

#[tokio::test]
async fn engine_flags_disable_evolve_and_inject() {
    let mut frozen =
        TtseEngine::new(config_with(|config| config.evolve_enabled = false)).expect("valid config");
    let added = frozen
        .observe(&[failed_trajectory()], &learning_signals())
        .await
        .expect("observe succeeds");
    assert!(added.is_empty());
    assert!(frozen.bank().is_empty());
    assert_eq!(frozen.bank().tick(), 0);

    let mut silent =
        TtseEngine::new(config_with(|config| config.inject_enabled = false)).expect("valid config");
    silent
        .bank_mut()
        .add(tip("use rg instead of grep", TipCategory::Tool, 1.0));

    let result = silent.consult("shell command");
    assert_eq!(result, ConsultResult::default());
    assert_eq!(silent.consult_count(), 0);
    assert_eq!(silent.bank().tips()[0].hits, 0);
}

#[tokio::test]
async fn engine_consult_records_hits() {
    let mut engine = observed_engine().await;

    let result = engine.consult("shell command");
    assert!(!result.tips.is_empty());
    assert!(result.rendered.contains("- ["));
    assert_eq!(engine.consult_count(), 1);

    for served in &result.tips {
        let stored = engine.bank().get(&served.tip_id).expect("tip stored");
        assert_eq!(stored.hits, 1);
        assert_eq!(stored.successes, 1);
    }
    assert_eq!(result.tips.len(), engine.bank().len());
}

#[tokio::test]
async fn engine_auto_dreams_on_the_interval() {
    let mut engine = TtseEngine::new(config_with(|config| {
        config.dream_interval = 2;
        config.dream_min_rules = 2;
        config.dream_enabled = true;
        config.dream_prune_enabled = false;
    }))
    .expect("valid config");
    for rule in [
        "check the working directory",
        "check the working directory first",
    ] {
        let record = tip(rule, TipCategory::Tool, 1.0);
        engine.bank_mut().tips.push(record);
    }

    let failed = trajectory(
        "t-fail",
        TrajectoryOutcome::Failure,
        vec![error_step(StepKind::Message, "build broke")],
    );
    engine.observe(&[failed], &[]).await.expect("observe");
    assert_eq!(engine.bank().tick(), 1);
    let after_first = engine.bank().len();
    assert_eq!(after_first, 3, "two seeded tips plus the induced one");

    engine
        .observe(&[], &learning_signals())
        .await
        .expect("observe");
    assert_eq!(engine.bank().tick(), 2);
    let after_second = engine.bank().len();
    assert!(
        after_second <= after_first,
        "the auto dream pass merged the pair: {after_second} > {after_first}"
    );
    assert_eq!(after_second, 3, "one tip added, one absorbed");
    let survivors: Vec<&str> = engine
        .bank()
        .tips()
        .iter()
        .map(|tip| tip.rule.as_str())
        .filter(|rule| rule.contains("working directory"))
        .collect();
    assert_eq!(survivors.len(), 1, "the pair collapsed onto one rule");

    let before = engine.bank().len();
    let report = engine.dream();
    assert_eq!(report.examined, before);
    assert!(report.examined >= 1);
    assert_eq!(report.merged, 0, "nothing is left to merge");
}

#[tokio::test]
async fn engine_persistence_round_trip() {
    let dir = tempfile::tempdir().expect("temporary directory");
    let path = dir.path().join("bank.json");

    let engine = observed_engine().await;
    let saved = engine.bank().tips().to_vec();
    assert_eq!(saved.len(), 3);

    engine.save(&path).await.expect("save succeeds");
    assert!(path.exists());
    assert!(
        !dir.path().join("bank.json.tmp").exists(),
        "the temporary file is renamed away"
    );

    let loaded = TtseEngine::load(&path).await.expect("load succeeds");
    assert_eq!(loaded.bank().tips(), saved.as_slice());
    assert_eq!(loaded.bank().tick(), engine.bank().tick());
    assert_eq!(loaded.consult_count(), 0);

    let second_path = dir.path().join("second.json");
    loaded.save(&second_path).await.expect("re-save succeeds");
    let reloaded = TtseEngine::load(&second_path).await.expect("reload");
    assert_eq!(reloaded.bank().tips(), saved.as_slice());

    assert!(TtseEngine::load(dir.path().join("missing.json"))
        .await
        .is_err());
}

#[tokio::test]
async fn engine_is_repeatable() {
    let mut first = observed_engine().await;
    let mut second = observed_engine().await;

    assert_eq!(first.bank().tips(), second.bank().tips());
    assert_eq!(first.stats(), second.stats());
    assert_eq!(
        first.consult("shell command").rendered,
        second.consult("shell command").rendered
    );
}
