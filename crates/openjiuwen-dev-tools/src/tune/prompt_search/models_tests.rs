//! Unit tests for the prompt-search domain models and their Python wire shapes.
use serde_json::json;

use super::*;

fn case(inputs: &[(&str, Value)], label: &[(&str, Value)]) -> Case {
    Case::new(
        inputs
            .iter()
            .map(|(key, value)| ((*key).to_owned(), value.clone()))
            .collect(),
        label
            .iter()
            .map(|(key, value)| ((*key).to_owned(), value.clone()))
            .collect(),
    )
    .unwrap()
}

#[test]
fn from_text_round_trips_through_case() {
    let case = PromptTaskCase::from_text("2 + 2?", "4").unwrap();
    assert_eq!(case.input_text, "2 + 2?");
    assert_eq!(case.expected_text, "4");
    assert!(!case.hidden);

    let shared = case.to_case().unwrap();
    assert_eq!(shared.inputs["query"], json!("2 + 2?"));
    assert_eq!(shared.label["output"], json!("4"));
}

#[test]
fn from_text_rejects_blank_text() {
    let error = PromptTaskCase::from_text("   ", "4").unwrap_err();
    assert_eq!(
        error.kind(),
        openjiuwen_common::error::ErrorKind::Validation
    );
}

#[test]
fn from_case_prefers_the_python_keys() {
    let shared = case(
        &[("input", json!("fallback")), ("query", json!("preferred"))],
        &[("expected", json!("b")), ("output", json!("a"))],
    );
    let task_case = PromptTaskCase::from_case(&shared, true);
    assert_eq!(task_case.input_text, "preferred");
    assert_eq!(task_case.expected_text, "a");
    assert!(task_case.hidden);
}

#[test]
fn from_case_falls_back_to_the_first_value_and_renders_json() {
    let shared = case(&[("prompt", json!({"nested": 1}))], &[("score", json!(3))]);
    let task_case = PromptTaskCase::from_case(&shared, false);
    assert_eq!(task_case.input_text, "{\"nested\":1}");
    assert_eq!(task_case.expected_text, "3");
}

#[test]
fn to_case_requires_a_non_blank_input() {
    let task_case = PromptTaskCase {
        input_text: "  ".to_owned(),
        expected_text: "a".to_owned(),
        hidden: false,
    };
    assert!(task_case.to_case().is_err());
}

#[test]
fn spec_builders_and_characteristics() {
    let spec = PromptTaskSpec::new("write a haiku")
        .with_cases(vec![
            PromptTaskCase::from_text("a", "b").unwrap(),
            PromptTaskCase::from_text("c", "d")
                .unwrap()
                .with_hidden(true),
        ])
        .with_constraints(vec!["short".to_owned(), "no rhyme".to_owned()])
        .with_base_prompt("base")
        .with_metadata(BTreeMap::from([("k".to_owned(), json!(1))]));

    assert_eq!(
        spec.characteristics(),
        "write a haiku\nconstraints: short; no rhyme"
    );
    assert_eq!(spec.base_prompt, "base");
    assert_eq!(spec.metadata["k"], json!(1));
    assert_eq!(spec.visible_cases().len(), 1);
    assert_eq!(spec.hidden_cases().len(), 1);

    let plain = PromptTaskSpec::new("only objective");
    assert_eq!(plain.characteristics(), "only objective");
}

#[test]
fn candidate_ids_are_deterministic_and_index_sensitive() {
    let first = candidate_id("prompt", 0);
    assert_eq!(first.len(), 8);
    assert!(first
        .chars()
        .all(|ch| ch.is_ascii_hexdigit() && !ch.is_ascii_uppercase()));
    assert_eq!(first, candidate_id("prompt", 0));
    assert_ne!(first, candidate_id("prompt", 1));
    assert_ne!(first, candidate_id("prompt ", 0));

    let candidate = PromptCandidate::new("prompt", "because");
    assert_eq!(candidate.candidate_id, first);
    assert_eq!(candidate.prompt(), "prompt");
    assert_eq!(candidate.rationale(), "because");
    assert_eq!(candidate.clone().with_id("abc").candidate_id(), "abc");
}

#[test]
fn case_result_flags_errors_and_blank_placeholders() {
    let result = CaseResult::new("c1", "out").with_score(0.5);
    assert!(result.ok());
    assert!(!result.hidden);
    assert_eq!(result.score, 0.5);

    let failed = CaseResult {
        error: Some("boom".to_owned()),
        ..CaseResult::new("c2", "")
    };
    assert!(!failed.ok());

    let blank = CaseResult::blank();
    assert_eq!(blank.case_id, "");
    assert_eq!(blank.output, "");
    assert!(blank.ok());
    assert!(CaseResult::new("c3", "x").with_hidden(true).hidden);
}

#[test]
fn execution_combines_outputs_and_reports_latency() {
    let mut execution = Execution::new(PromptCandidate::new("p", ""));
    execution.latency_ms = 1500.0;
    execution.total_tokens = 42;
    execution.push_result(CaseResult::new("v1", "one"));
    execution.push_result(CaseResult::new("v2", ""));
    execution.push_result(CaseResult::new("v3", "three"));
    execution.push_result(CaseResult::new("h1", "held").with_hidden(true));

    assert!((execution.latency_s() - 1.5).abs() < 1e-12);
    assert_eq!(execution.visible_results().len(), 3);
    assert_eq!(execution.hidden_results().len(), 1);
    assert_eq!(execution.combined_output(), "one\n\n---\n\nthree");
    assert_eq!(execution.combined_output_hidden(), "held");
}

#[test]
fn iteration_record_tracks_the_best_candidate() {
    let mut record = IterationRecord::new(2);
    assert_eq!(record.index, 2);
    assert!(record.best_candidate().is_none());

    record.push(PromptCandidate::new("weak", "").with_id("a"), 0.2);
    record.push(PromptCandidate::new("strong", "").with_id("b"), 0.8);
    record.push(PromptCandidate::new("tie", "").with_id("c"), 0.8);

    assert_eq!(record.best_score, 0.8);
    assert_eq!(record.scores.len(), 3);
    assert_eq!(record.best_candidate().unwrap().candidate_id, "b");
}

#[test]
fn prompt_record_gain_and_pending() {
    let record = PromptRecord::new("r1", "prompt", 0.8, "objective")
        .with_characteristics("chars")
        .with_created_at(123.0)
        .with_baseline_reward(0.5)
        .with_observations(vec!["kept".to_owned()])
        .with_metrics(BTreeMap::from([("m".to_owned(), json!(1))]))
        .with_metadata(BTreeMap::from([("k".to_owned(), json!("v"))]));

    assert!((record.gain() - 0.3).abs() < 1e-12);
    assert!(record.is_pending(0.1));
    assert!(!record.is_pending(0.31));
    assert!(!record.with_applied(true, Some(200.0)).is_pending(0.0));
    assert_eq!(PromptRecord::new("r2", "p", 0.2, "o").gain(), 0.2);
}

#[test]
fn reward_breakdown_default_and_gate_flag() {
    let breakdown = RewardBreakdown::default();
    assert!(!breakdown.is_gated());
    assert_eq!(breakdown.score, 0.0);
    assert_eq!(breakdown.notes.len(), 0);

    let gated = RewardBreakdown {
        gated: true,
        notes: vec!["correctness 0.10 < gate 0.50: reward capped".to_owned()],
        ..RewardBreakdown::default()
    };
    assert!(gated.is_gated());
}

#[test]
fn serde_uses_the_python_field_names() {
    let record = PromptRecord::new("r1", "prompt", 0.5, "objective").with_baseline_reward(0.2);
    let value = serde_json::to_value(&record).unwrap();
    for key in [
        "record_id",
        "prompt",
        "reward",
        "objective",
        "task_characteristics",
        "metrics",
        "observations",
        "metadata",
        "created_at",
        "baseline_reward",
        "applied",
        "applied_at",
    ] {
        assert!(value.get(key).is_some(), "missing {key}: {value}");
    }
    assert_eq!(value["baseline_reward"], json!(0.2));

    let case = PromptTaskCase::from_text("q", "a")
        .unwrap()
        .with_hidden(true);
    let value = serde_json::to_value(&case).unwrap();
    assert_eq!(value["input_text"], json!("q"));
    assert_eq!(value["expected_text"], json!("a"));
    assert_eq!(value["hidden"], json!(true));

    let execution = Execution {
        candidate: PromptCandidate::new("p", "r"),
        latency_ms: 10.0,
        total_tokens: 5,
        error: None,
        results: vec![CaseResult::new("c", "o").with_score(0.5)],
        telemetry: RunTelemetry::default(),
    };
    let value = serde_json::to_value(&execution).unwrap();
    assert_eq!(value["latency_ms"], json!(10.0));
    assert_eq!(value["total_tokens"], json!(5));
    assert_eq!(value["telemetry"]["latency_ms"], Value::Null);
    assert_eq!(value["results"][0]["case_id"], json!("c"));
    assert_eq!(
        value["candidate"]["candidate_id"],
        json!(execution.candidate.candidate_id)
    );
}

#[test]
fn measured_accessors_prefer_telemetry_over_legacy_fields() {
    let mut execution = Execution::new(PromptCandidate::new("p", ""));
    execution.latency_ms = 4000.0;
    execution.total_tokens = 4000;
    // Nothing measured: the legacy caller-supplied fields win.
    assert!((execution.measured_latency_s() - 4.0).abs() < 1e-12);
    assert_eq!(execution.measured_total_tokens(), 4000);
    assert_eq!(execution.measured_cost(), None);

    execution.telemetry = RunTelemetry::default()
        .with_latency_ms(2000.0)
        .with_prompt_tokens(300)
        .with_completion_tokens(700)
        .with_cost(0.25);
    assert!((execution.measured_latency_s() - 2.0).abs() < 1e-12);
    assert_eq!(execution.measured_total_tokens(), 1000);
    assert_eq!(execution.measured_cost(), Some(0.25));
    // The legacy view still reports the legacy field.
    assert!((execution.latency_s() - 4.0).abs() < 1e-12);
    assert_eq!(execution.total_tokens, 4000);
}

#[test]
fn measured_zero_wins_over_a_non_zero_legacy_value() {
    let mut execution = Execution::new(PromptCandidate::new("p", ""));
    execution.latency_ms = 9000.0;
    execution.total_tokens = 9000;
    execution.telemetry = RunTelemetry::default()
        .with_latency_ms(0.0)
        .with_prompt_tokens(0)
        .with_cost(0.0);

    assert_eq!(execution.measured_latency_s(), 0.0);
    assert_eq!(execution.measured_total_tokens(), 0);
    assert_eq!(execution.measured_cost(), Some(0.0));
}

#[test]
fn telemetry_round_trips_beside_the_legacy_keys() {
    let mut execution = Execution::new(PromptCandidate::new("p", ""));
    execution.latency_ms = 10.0;
    execution.total_tokens = 5;
    execution.telemetry = RunTelemetry::default()
        .with_latency_ms(12.5)
        .with_prompt_tokens(4)
        .with_completion_tokens(6)
        .with_cost(0.5);
    let value = serde_json::to_value(&execution).unwrap();
    assert_eq!(value["latency_ms"], json!(10.0));
    assert_eq!(value["total_tokens"], json!(5));
    assert_eq!(value["telemetry"]["latency_ms"], json!(12.5));
    assert_eq!(value["telemetry"]["prompt_tokens"], json!(4));
    assert_eq!(value["telemetry"]["completion_tokens"], json!(6));
    assert_eq!(value["telemetry"]["cost"], json!(0.5));

    let parsed: Execution = serde_json::from_value(value).unwrap();
    assert_eq!(parsed, execution);
    assert_eq!(parsed.measured_total_tokens(), 10);
}

#[test]
fn old_payloads_without_telemetry_leave_it_absent() {
    let payload = json!({
        "candidate": {"prompt": "p", "candidate_id": "abc", "rationale": ""},
        "latency_ms": 2500.0,
        "total_tokens": 800,
        "results": [{"case_id": "c", "output": "o", "score": 0.5}],
    });
    let execution: Execution = serde_json::from_value(payload).unwrap();
    assert_eq!(execution.telemetry, RunTelemetry::default());
    assert!(execution.telemetry.is_empty());
    assert!((execution.measured_latency_s() - 2.5).abs() < 1e-12);
    assert_eq!(execution.measured_total_tokens(), 800);
    assert_eq!(execution.measured_cost(), None);
    assert_eq!(
        execution.results[0].telemetry,
        RunTelemetry::default(),
        "a case result without telemetry keeps the default"
    );
}

#[test]
fn case_result_telemetry_defaults_and_round_trips() {
    let result = CaseResult::new("c", "out");
    assert!(result.telemetry.is_empty());
    assert_eq!(
        result,
        result.clone().with_telemetry(RunTelemetry::default())
    );
    assert_eq!(CaseResult::blank().telemetry, RunTelemetry::default());

    let measured = result.with_telemetry(
        RunTelemetry::default()
            .with_latency_ms(20.0)
            .with_completion_tokens(9),
    );
    assert_eq!(measured.telemetry.latency_ms, Some(20.0));
    assert_eq!(measured.telemetry.total_tokens(), Some(9));
    assert_eq!(measured.output, "out");

    let value = serde_json::to_value(&measured).unwrap();
    assert_eq!(value["telemetry"]["latency_ms"], json!(20.0));
    assert_eq!(value["telemetry"]["completion_tokens"], json!(9));
    let parsed: CaseResult = serde_json::from_value(value).unwrap();
    assert_eq!(parsed, measured);

    let legacy: CaseResult =
        serde_json::from_value(json!({"case_id": "c", "output": "o", "score": 0.2})).unwrap();
    assert!(legacy.telemetry.is_empty());
}

#[test]
fn deserialization_tolerates_missing_and_unknown_fields() {
    let value = json!({
        "index": 3,
        "best_score": 0.7,
        "candidates": [{"prompt": "p"}],
        "scores": [0.7],
        "unexpected": "ignored",
    });
    let record: IterationRecord = serde_json::from_value(value).unwrap();
    assert_eq!(record.index, 3);
    assert_eq!(record.best_score, 0.7);
    assert_eq!(record.candidates[0].prompt, "p");
    assert_eq!(record.candidates[0].candidate_id, "");

    let breakdown: RewardBreakdown =
        serde_json::from_value(json!({"score": 0.4, "gated": true})).unwrap();
    assert_eq!(breakdown.score, 0.4);
    assert!(breakdown.gated);
    assert_eq!(breakdown.correctness, 0.0);
}

#[test]
fn evaluation_exposes_its_candidate_and_score() {
    let candidate = PromptCandidate::new("p", "");
    let evaluation = Evaluation {
        execution: Execution::new(candidate.clone()),
        reward: RewardBreakdown {
            score: 0.6,
            ..RewardBreakdown::default()
        },
    };
    assert_eq!(evaluation.candidate(), &candidate);
    assert!((evaluation.score() - 0.6).abs() < 1e-12);
}
