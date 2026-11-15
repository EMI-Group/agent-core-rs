//! Unit tests for the reward components and the composite reward.
use super::*;
use crate::tune::prompt_search::models::{PromptCandidate, PromptTaskCase};
use crate::tune::tool::RunTelemetry;

fn execution(candidate_id: &str, visible: &[f64], hidden: &[f64]) -> Execution {
    let candidate = PromptCandidate::new("prompt", "").with_id(candidate_id);
    let mut execution = Execution::new(candidate);
    for (index, score) in visible.iter().enumerate() {
        execution.push_result(CaseResult::new(format!("v{index}"), "out").with_score(*score));
    }
    for (index, score) in hidden.iter().enumerate() {
        execution.push_result(
            CaseResult::new(format!("h{index}"), "held")
                .with_score(*score)
                .with_hidden(true),
        );
    }
    execution
}

fn task_with_hidden() -> PromptTaskSpec {
    PromptTaskSpec::new("objective").with_cases(vec![
        PromptTaskCase::from_text("v", "a").unwrap(),
        PromptTaskCase::from_text("h", "b")
            .unwrap()
            .with_hidden(true),
    ])
}

#[test]
fn clamp01_is_total() {
    assert_eq!(clamp01(-1.0), 0.0);
    assert!((clamp01(0.4) - 0.4).abs() < 1e-12);
    assert_eq!(clamp01(2.0), 1.0);
    assert_eq!(clamp01(f64::NAN), 0.0);
    assert_eq!(clamp01(f64::INFINITY), 0.0);
    assert_eq!(clamp01(f64::NEG_INFINITY), 0.0);
}

#[test]
fn min_max_normalization_boundaries() {
    assert_eq!(min_max_normalize(&[]).len(), 0);
    assert_eq!(min_max_normalize(&[0.3, 0.3, 0.3]), vec![1.0, 1.0, 1.0]);
    let scaled = min_max_normalize(&[0.2, 0.6]);
    assert!((scaled[0] - 0.0).abs() < 1e-12);
    assert!((scaled[1] - 1.0).abs() < 1e-12);
    let middle = min_max_normalize(&[0.0, 1.0, 2.0]);
    assert!((middle[1] - 0.5).abs() < 1e-12);
    // A non-finite value is treated as 0.0 rather than poisoning the range.
    let with_nan = min_max_normalize(&[f64::NAN, 1.0]);
    assert_eq!(with_nan, vec![0.0, 1.0]);
}

#[test]
fn completeness_counts_only_complete_visible_results() {
    let mut execution = Execution::new(PromptCandidate::new("p", ""));
    execution.push_result(CaseResult::new("a", "  word  "));
    execution.push_result(CaseResult::new("b", " "));
    execution.push_result(CaseResult {
        error: Some("boom".to_owned()),
        ..CaseResult::new("c", "word")
    });
    execution.push_result(CaseResult::new("h", "held").with_hidden(true));

    let component = CompletenessReward::new();
    assert!((component.score(&execution, &PromptTaskSpec::default()) - 1.0 / 3.0).abs() < 1e-12);
    let strict = CompletenessReward::new().with_min_chars(5);
    assert!((strict.score(&execution, &PromptTaskSpec::default()) - 0.0).abs() < 1e-12);
    assert_eq!(strict.min_chars(), 5);
    assert_eq!(CompletenessReward::new().with_min_chars(0).min_chars(), 1);
}

#[test]
fn latency_token_and_cost_rewards_hit_their_half_lives() {
    let mut zero = Execution::new(PromptCandidate::new("p", ""));
    assert!(
        (LatencyReward::default().score(&zero, &PromptTaskSpec::default()) - 1.0).abs() < 1e-12
    );
    assert!(
        (TokenUsageReward::default().score(&zero, &PromptTaskSpec::default()) - 1.0).abs() < 1e-12
    );
    assert!((CostReward::default().score(&zero, &PromptTaskSpec::default()) - 1.0).abs() < 1e-12);

    zero.latency_ms = 5000.0;
    zero.total_tokens = 2000;
    let task = PromptTaskSpec::default();
    assert!((LatencyReward::default().score(&zero, &task) - 0.5).abs() < 1e-12);
    assert!((TokenUsageReward::default().score(&zero, &task) - 0.5).abs() < 1e-12);

    // 1M tokens at 0.01 per Mtok costs 0.01, the default half-life.
    zero.total_tokens = 1_000_000;
    let cost = CostReward::new(0.01, 0.01);
    assert!((cost.score(&zero, &task) - 0.5).abs() < 1e-12);
    assert!((cost.price_per_mtok() - 0.01).abs() < 1e-12);
    assert!((cost.half_life_cost() - 0.01).abs() < 1e-12);

    // Clamps keep the half-lives usable.
    assert!((LatencyReward::new(-1.0).half_life_s() - 1e-3).abs() < 1e-12);
    assert!((TokenUsageReward::new(0.0).half_life_tokens() - 1.0).abs() < 1e-12);
    assert!((CostReward::new(-5.0, 0.0).price_per_mtok() - 0.0).abs() < 1e-12);
    assert!((CostReward::new(0.0, 0.0).half_life_cost() - 1e-9).abs() < 1e-12);
}

#[test]
fn measured_telemetry_drives_latency_token_and_cost_rewards() {
    let task = PromptTaskSpec::default();

    let mut measured = Execution::new(PromptCandidate::new("p", ""));
    measured.telemetry = RunTelemetry::default().with_latency_ms(1000.0);
    assert!((LatencyReward::new(1.0).score(&measured, &task) - 0.5).abs() < 1e-12);

    let mut measured = Execution::new(PromptCandidate::new("p", ""));
    measured.telemetry = RunTelemetry::default()
        .with_prompt_tokens(1000)
        .with_completion_tokens(1000);
    assert!((TokenUsageReward::new(2000.0).score(&measured, &task) - 0.5).abs() < 1e-12);

    let mut measured = Execution::new(PromptCandidate::new("p", ""));
    measured.telemetry = RunTelemetry::default().with_cost(1.0);
    let cost = CostReward::new(1_000_000.0, 1.0);
    assert!((cost.score(&measured, &task) - 0.5).abs() < 1e-12);
}

#[test]
fn a_prompt_only_token_measurement_counts_the_missing_side_as_zero() {
    let mut execution = Execution::new(PromptCandidate::new("p", ""));
    execution.telemetry = RunTelemetry::default().with_prompt_tokens(2000);
    let task = PromptTaskSpec::default();

    // total = 2000 prompt + 0 unreported completion, the 2000 half-life.
    assert!((TokenUsageReward::new(2000.0).score(&execution, &task) - 0.5).abs() < 1e-12);
    // The derived cost also uses the measured total: 2000 tokens at 5/Mtok.
    assert!((CostReward::new(5.0, 0.01).score(&execution, &task) - 0.5).abs() < 1e-12);

    let mut completion_only = Execution::new(PromptCandidate::new("p", ""));
    completion_only.telemetry = RunTelemetry::default().with_completion_tokens(500);
    assert!((TokenUsageReward::new(500.0).score(&completion_only, &task) - 0.5).abs() < 1e-12);
}

#[test]
fn telemetry_overrides_the_legacy_caller_supplied_fields() {
    let task = PromptTaskSpec::default();
    let mut execution = Execution::new(PromptCandidate::new("p", ""));
    execution.latency_ms = 60_000.0;
    execution.total_tokens = 1_000_000;

    // Without telemetry the legacy caller-supplied fields still score as before.
    assert!((LatencyReward::new(60.0).score(&execution, &task) - 0.5).abs() < 1e-12);
    assert!((TokenUsageReward::new(1_000_000.0).score(&execution, &task) - 0.5).abs() < 1e-12);
    assert!((CostReward::new(1.0, 1.0).score(&execution, &task) - 0.5).abs() < 1e-12);

    execution.telemetry = RunTelemetry::default()
        .with_latency_ms(1000.0)
        .with_prompt_tokens(1000)
        .with_completion_tokens(1000);
    // The measured values now drive every component, not the legacy fields.
    assert!((LatencyReward::new(1.0).score(&execution, &task) - 0.5).abs() < 1e-12);
    assert!((TokenUsageReward::new(2000.0).score(&execution, &task) - 0.5).abs() < 1e-12);
    // A token-derived cost uses the measured total: 2000 tokens at 500/Mtok.
    assert!((CostReward::new(500.0, 1.0).score(&execution, &task) - 0.5).abs() < 1e-12);
}

#[test]
fn absent_telemetry_keeps_the_run_cost_rewards_constant() {
    let task = PromptTaskSpec::default();
    let mut empty = Execution::new(PromptCandidate::new("p", ""));
    empty.telemetry = RunTelemetry::default();
    assert!(empty.telemetry.is_empty());

    assert_eq!(LatencyReward::default().score(&empty, &task), 1.0);
    assert_eq!(TokenUsageReward::default().score(&empty, &task), 1.0);
    // The default price is 0.0, so an unreported run costs nothing.
    assert_eq!(CostReward::default().score(&empty, &task), 1.0);
    // A non-zero price with no measured tokens still derives a zero cost.
    assert_eq!(CostReward::new(5.0, 0.01).score(&empty, &task), 1.0);
}

#[test]
fn a_measured_zero_scores_like_an_absent_measurement() {
    let task = PromptTaskSpec::default();
    let mut zero = Execution::new(PromptCandidate::new("p", ""));
    zero.telemetry = RunTelemetry::default()
        .with_latency_ms(0.0)
        .with_prompt_tokens(0)
        .with_cost(0.0);
    assert!(!zero.telemetry.is_empty());

    // Zero and absent are indistinguishable here: both score the full 1.0.
    assert_eq!(LatencyReward::new(1.0).score(&zero, &task), 1.0);
    assert_eq!(TokenUsageReward::new(2000.0).score(&zero, &task), 1.0);
    assert_eq!(CostReward::new(5.0, 0.01).score(&zero, &task), 1.0);

    // An explicit zero cost wins over a token-derived cost.
    let mut priced = Execution::new(PromptCandidate::new("p", ""));
    priced.telemetry = RunTelemetry::default()
        .with_prompt_tokens(1_000_000)
        .with_cost(0.0);
    assert_eq!(CostReward::new(0.01, 0.01).score(&priced, &task), 1.0);
}

#[test]
fn correctness_ignores_failed_results_and_empty_lists() {
    let component = CorrectnessReward::new();
    assert_eq!(component.correctness_on(&[]), 0.0);
    let results = vec![
        CaseResult::new("a", "x").with_score(0.4),
        CaseResult {
            error: Some("boom".to_owned()),
            ..CaseResult::new("b", "y").with_score(1.0)
        },
        CaseResult::new("c", "z").with_score(0.8),
    ];
    assert!((component.correctness_on(&results) - 0.6).abs() < 1e-12);

    let mut execution = Execution::new(PromptCandidate::new("p", ""));
    execution.push_result(CaseResult::new("a", "x").with_score(0.5));
    execution.push_result(CaseResult::new("h", "y").with_score(1.0).with_hidden(true));
    assert!((component.score(&execution, &PromptTaskSpec::default()) - 0.5).abs() < 1e-12);
    assert!(component.is_correctness());
    assert_eq!(component.name(), "correctness");
}

#[test]
fn structured_validation_and_custom_rewards() {
    let mut execution = Execution::new(PromptCandidate::new("p", ""));
    execution.push_result(CaseResult::new("a", "{\"ok\":true}"));
    execution.push_result(CaseResult::new("b", "not json"));
    let task = PromptTaskSpec::default();

    let json_reward = StructuredValidationReward::json();
    assert!((json_reward.score(&execution, &task) - 0.5).abs() < 1e-12);
    assert_eq!(json_reward.name(), "structured_validation");
    assert!(format!("{json_reward:?}").contains("StructuredValidationReward"));

    let always = StructuredValidationReward::new(|_| true);
    assert!((always.score(&execution, &task) - 1.0).abs() < 1e-12);
    let empty = Execution::new(PromptCandidate::new("p", ""));
    assert_eq!(always.score(&empty, &task), 0.0);

    let custom = CustomReward::new("speed", |execution, _| 10.0 - execution.latency_ms);
    execution.latency_ms = 9.5;
    assert!((custom.score(&execution, &task) - 0.5).abs() < 1e-12);
    assert_eq!(custom.name(), "speed");
    assert!(format!("{custom:?}").contains("speed"));
}

#[test]
fn default_composite_filters_zero_weight_components() {
    let composite = CompositeReward::default();
    assert_eq!(
        composite.component_names(),
        vec!["correctness", "completeness", "latency", "token_usage"]
    );
    assert_eq!(composite.weights()["correctness"], 1.0);
    assert!(!composite.normalize());
    assert!((composite.min_correctness() - 0.5).abs() < 1e-12);
    assert!((composite.drift_penalty() - 0.5).abs() < 1e-12);
    assert_eq!(composite.correctness_name(), "correctness");
    assert!(format!("{composite:?}").contains("CompositeReward"));
}

#[test]
fn weighted_mean_divides_by_kept_weights() {
    // correctness 1.0 and completeness 1.0 with weights 1.0 / 0.3.
    let composite = CompositeReward::new(
        vec![
            Box::new(CorrectnessReward::new()),
            Box::new(CompletenessReward::new()),
        ],
        BTreeMap::from([
            ("correctness".to_owned(), 1.0),
            ("completeness".to_owned(), 0.3),
        ]),
    );
    let breakdowns = composite.evaluate(
        &[execution("a", &[1.0], &[])],
        &PromptTaskSpec::default(),
        &BTreeMap::new(),
    );
    assert_eq!(breakdowns.len(), 1);
    assert!((breakdowns[0].score - 1.0).abs() < 1e-12);
    assert_eq!(breakdowns[0].components["completeness"], 1.0);
}

#[test]
fn normalize_flag_rescales_unbounded_components() {
    let make = |normalize: bool| {
        CompositeReward::new(
            vec![Box::new(CustomReward::new("raw", |execution, _| {
                execution.latency_ms
            }))],
            BTreeMap::from([("raw".to_owned(), 1.0)]),
        )
        .with_normalize(normalize)
    };
    let executions = [execution("a", &[], &[]), execution("b", &[], &[])];
    let mut first = executions[0].clone();
    first.latency_ms = 0.2;
    let mut second = executions[1].clone();
    second.latency_ms = 0.6;
    let batch = [first, second];
    let task = PromptTaskSpec::default();

    let raw = make(false).evaluate(&batch, &task, &BTreeMap::new());
    assert!((raw[0].score - 0.2).abs() < 1e-12);
    assert!((raw[1].score - 0.6).abs() < 1e-12);

    let normalized = make(true).evaluate(&batch, &task, &BTreeMap::new());
    assert!((normalized[0].score - 0.0).abs() < 1e-12);
    assert!((normalized[1].score - 1.0).abs() < 1e-12);
    assert!((normalized[0].raw_components["raw"] - 0.2).abs() < 1e-12);
}

#[test]
fn gate_and_overfit_penalty_apply_in_order() {
    let composite = CompositeReward::new(
        vec![Box::new(CorrectnessReward::new())],
        BTreeMap::from([("correctness".to_owned(), 1.0)]),
    )
    .with_min_correctness(0.95);
    let task = task_with_hidden();
    let breakdowns = composite.evaluate(&[execution("a", &[0.9], &[0.0])], &task, &BTreeMap::new());
    let breakdown = &breakdowns[0];

    // Weighted mean 0.9, capped at correctness 0.9, then 0.5 * 0.9 = 0.45.
    assert!((breakdown.score - 0.45).abs() < 1e-9, "{breakdown:?}");
    assert!(breakdown.gated);
    assert_eq!(breakdown.notes.len(), 2);
    assert_eq!(
        breakdown.notes[0],
        "correctness 0.90 < gate 0.95: reward capped"
    );
    assert_eq!(
        breakdown.notes[1],
        "overfitting: visible 0.90 vs hidden 0.00 (-0.45)"
    );
    assert!((breakdown.correctness - 0.9).abs() < 1e-12);
}

#[test]
fn overfit_needs_hidden_cases_and_a_correctness_component() {
    let composite = CompositeReward::new(
        vec![Box::new(CorrectnessReward::new())],
        BTreeMap::from([("correctness".to_owned(), 1.0)]),
    );
    let without_hidden = PromptTaskSpec::new("objective");
    let breakdown = composite.evaluate(
        &[execution("a", &[1.0], &[])],
        &without_hidden,
        &BTreeMap::new(),
    );
    assert_eq!(breakdown[0].notes.len(), 0);

    let no_correctness = CompositeReward::new(
        vec![Box::new(CompletenessReward::new())],
        BTreeMap::from([("completeness".to_owned(), 1.0)]),
    );
    let task = task_with_hidden();
    let breakdown =
        no_correctness.evaluate(&[execution("a", &[1.0], &[0.0])], &task, &BTreeMap::new());
    assert_eq!(breakdown[0].notes.len(), 0);
    // No correctness component: the reported correctness defaults to 1.0.
    assert!((breakdown[0].correctness - 1.0).abs() < 1e-12);
}

#[test]
fn drift_penalty_is_subtracted_and_reported() {
    let composite = CompositeReward::new(
        vec![Box::new(CorrectnessReward::new())],
        BTreeMap::from([("correctness".to_owned(), 1.0)]),
    );
    let task = PromptTaskSpec::default();
    let drift = BTreeMap::from([("a".to_owned(), 0.4), ("b".to_owned(), 2.0)]);
    let breakdowns = composite.evaluate(
        &[execution("a", &[1.0], &[]), execution("b", &[1.0], &[])],
        &task,
        &drift,
    );
    assert!((breakdowns[0].score - 0.8).abs() < 1e-12);
    assert!((breakdowns[0].drift - 0.4).abs() < 1e-12);
    assert!((breakdowns[0].drift_penalty_applied - 0.2).abs() < 1e-12);
    assert_eq!(breakdowns[0].notes, vec!["drift 0.40 penalty -0.20"]);

    // Drift above 1.0 is clamped, so the penalty bottoms out at 0.5.
    assert!((breakdowns[1].drift - 1.0).abs() < 1e-12);
    assert!((breakdowns[1].drift_penalty_applied - 0.5).abs() < 1e-12);
    assert!((breakdowns[1].score - 0.5).abs() < 1e-12);
}

#[test]
fn execution_error_forces_zero_and_notes_it() {
    let composite = CompositeReward::new(
        vec![Box::new(CorrectnessReward::new())],
        BTreeMap::from([("correctness".to_owned(), 1.0)]),
    );
    let mut failed = execution("a", &[1.0], &[]);
    failed.error = Some("provider down".to_owned());
    let breakdown = composite.evaluate(&[failed], &PromptTaskSpec::default(), &BTreeMap::new());
    assert_eq!(breakdown[0].score, 0.0);
    assert_eq!(
        breakdown[0].notes.last().unwrap(),
        "execution error: provider down"
    );
}

#[test]
fn empty_batch_and_deterministic_repeat() {
    let composite = CompositeReward::default();
    assert_eq!(
        composite
            .evaluate(&[], &PromptTaskSpec::default(), &BTreeMap::new())
            .len(),
        0
    );

    let task = task_with_hidden();
    let batch = [execution("a", &[0.8], &[0.7])];
    let first = composite.evaluate(&batch, &task, &BTreeMap::new());
    let second = composite.evaluate(&batch, &task, &BTreeMap::new());
    assert_eq!(first, second);
}

#[test]
fn breakdown_round_trips_with_python_field_names() {
    let composite = CompositeReward::default();
    let breakdown = &composite.evaluate(
        &[execution("a", &[0.4], &[])],
        &PromptTaskSpec::default(),
        &BTreeMap::new(),
    )[0];
    let value = serde_json::to_value(breakdown).unwrap();
    for key in [
        "score",
        "components",
        "raw_components",
        "correctness",
        "drift",
        "drift_penalty_applied",
        "gated",
        "notes",
    ] {
        assert!(value.get(key).is_some(), "missing {key}: {value}");
    }
    let parsed: RewardBreakdown = serde_json::from_value(value).unwrap();
    assert_eq!(&parsed, breakdown);
    assert!(parsed.gated);
}
