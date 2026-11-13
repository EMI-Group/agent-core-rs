#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Integration tests for the model-backed LLM-as-judge evaluator.
//!
//! Covers the whole judge surface that lives in
//! `openjiuwen_agent_evolving::evaluator`: `JudgeRubric`/`JudgeCriterion`
//! validation, the tolerant `JudgeVerdict::parse` reader (plain, fenced, and
//! prose-embedded JSON; prose score forms; score aliases; scale normalisation
//! and clamping), the deterministic `build_judge_messages` prompt (rubric
//! rendering, prediction truncation), `LlmJudge` retry/error policy over a
//! `ScriptedModel`, `LlmJudgeEvaluator`'s explicit fallback, and
//! `DefaultEvaluator`'s optional judge.
//!
//! Everything here is offline and deterministic: the only model is
//! `openjiuwen_test_support::ScriptedModel`, which serves a fixed turn queue and
//! records its calls. Nothing reads the clock, the environment, or the network,
//! and no test depends on execution order.
use std::collections::BTreeMap;
use std::sync::Arc;

use openjiuwen_agent_evolving::dataset::Case;
use openjiuwen_agent_evolving::evaluator::{
    build_judge_messages, DefaultEvaluator, Evaluator, ExactMatchMetric, JudgeCriterion,
    JudgeRubric, JudgeVerdict, LlmJudge, LlmJudgeEvaluator,
};
use openjiuwen_common::error::Error;
use openjiuwen_core::foundation::llm::ModelRequestConfig;
use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};
use serde_json::{json, Value};

/// Compares scores without relying on exact float equality.
fn close(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() < 1e-9
}

/// A string prediction.
fn text(value: &str) -> Value {
    Value::String(value.to_owned())
}

/// A one-entry map holding a string, the common label shape.
fn text_map(key: &str, value: &str) -> BTreeMap<String, Value> {
    BTreeMap::from([(key.to_owned(), text(value))])
}

/// A valid case with fixed inputs and the given label.
fn case(id: &str, label: BTreeMap<String, Value>) -> Case {
    Case::new(
        BTreeMap::from([("question".to_owned(), json!("2 + 2?"))]),
        label,
    )
    .unwrap()
    .with_case_id(id)
}

/// A valid case whose label is `{"answer": answer}`.
fn answer_case(id: &str, answer: &str) -> Case {
    case(id, text_map("answer", answer))
}

/// Asserts an error is a validation failure mentioning `needle`.
fn assert_validation(error: &Error, needle: &str) {
    assert!(
        matches!(error, Error::Validation(message) if message.contains(needle)),
        "unexpected error: {error}"
    );
}

/// The request configuration every judge call uses here.
fn request() -> ModelRequestConfig {
    ModelRequestConfig::default()
}

/// A model that never answers with a parseable verdict, however often it is called.
fn junk_model() -> ScriptedModel {
    ScriptedModel::builder()
        .fallback(ScriptedTurn::text("I cannot decide."))
        .build()
}

/// Builds a judge over a clone of `model`, sharing its script and call history.
fn judge_over(model: &ScriptedModel, rubric: JudgeRubric, attempts: usize) -> LlmJudge {
    LlmJudge::new(Arc::new(model.clone()), rubric, request())
        .unwrap()
        .with_attempts(attempts)
}

#[test]
fn rubric_validation_accepts_valid_rubrics() {
    assert!(JudgeRubric::default_rubric().validate().is_ok());
    assert!(JudgeRubric::single("accuracy", "is it right")
        .validate()
        .is_ok());
    assert!(JudgeCriterion::new("accuracy", "").validate().is_ok());

    let explicit = JudgeRubric::new(
        vec![JudgeCriterion::new("accuracy", "is it right").with_weight(2.0)],
        5.0,
        0.5,
    );
    assert!(explicit.validate().is_ok());
    // The scale boundaries themselves are valid.
    let boundaries = JudgeRubric::new(
        vec![JudgeCriterion::new("accuracy", "is it right")],
        1.0,
        1.0,
    );
    assert!(boundaries.validate().is_ok());
    let no_pass = JudgeRubric::new(
        vec![JudgeCriterion::new("accuracy", "is it right")],
        10.0,
        0.0,
    );
    assert!(no_pass.validate().is_ok());
}

#[test]
fn rubric_validation_rejects_bad_criteria() {
    let empty = JudgeRubric::new(Vec::new(), 10.0, 0.6)
        .validate()
        .unwrap_err();
    assert_validation(&empty, "at least one criterion");

    let unnamed = JudgeRubric::single("   ", "blank name")
        .validate()
        .unwrap_err();
    assert_validation(&unnamed, "name");

    let default_criterion = JudgeCriterion::default().validate().unwrap_err();
    assert_validation(&default_criterion, "name");

    for weight in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let rubric = JudgeRubric::new(
            vec![JudgeCriterion::new("accuracy", "right").with_weight(weight)],
            10.0,
            0.6,
        );
        let error = rubric.validate().unwrap_err();
        assert_validation(&error, "weight");
    }
}

#[test]
fn rubric_validation_rejects_bad_scales() {
    for max_score in [0.0, -5.0, f64::NAN, f64::INFINITY] {
        let rubric = JudgeRubric::new(
            vec![JudgeCriterion::new("accuracy", "right")],
            max_score,
            0.6,
        );
        let error = rubric.validate().unwrap_err();
        assert_validation(&error, "max_score");
    }
    for passing_score in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
        let rubric = JudgeRubric::new(
            vec![JudgeCriterion::new("accuracy", "right")],
            10.0,
            passing_score,
        );
        let error = rubric.validate().unwrap_err();
        assert_validation(&error, "passing_score");
    }
}

#[test]
fn verdict_parse_reads_plain_json() {
    let verdict = JudgeVerdict::parse(
        r#"{"score": 8, "reasoning": "close enough",
            "per_criterion": {"correctness": 8, "clarity": 10}}"#,
        &JudgeRubric::default_rubric(),
    )
    .unwrap();

    assert!(close(verdict.score, 0.8));
    assert!(close(verdict.raw_score, 8.0));
    assert!(verdict.passed);
    assert_eq!(verdict.reasoning, "close enough");
    assert_eq!(verdict.per_criterion.len(), 2);
    assert!(close(verdict.per_criterion["correctness"], 0.8));
    assert!(close(verdict.per_criterion["clarity"], 1.0));
}

#[test]
fn verdict_parse_reads_fenced_and_prose_embedded_json() {
    let rubric = JudgeRubric::default_rubric();

    let fenced = JudgeVerdict::parse(
        "```json\n{\"score\": 6, \"reasoning\": \"ok\"}\n```",
        &rubric,
    )
    .unwrap();
    assert!(close(fenced.score, 0.6));
    assert_eq!(fenced.reasoning, "ok");

    let embedded = JudgeVerdict::parse(
        "Here is my verdict: {\"score\": 9, \"reasoning\": \"great\"} — done.",
        &rubric,
    )
    .unwrap();
    assert!(close(embedded.score, 0.9));
    assert_eq!(embedded.reasoning, "great");
}

#[test]
fn verdict_parse_recovers_scores_from_prose() {
    let rubric = JudgeRubric::default_rubric();
    for (response, expected) in [
        ("Score: 7", 0.7),
        ("score = 9", 0.9),
        ("7/10", 0.7),
        ("9 out of 10", 0.9),
    ] {
        let verdict = JudgeVerdict::parse(response, &rubric).unwrap();
        assert!(
            close(verdict.score, expected),
            "{response} parsed as {}",
            verdict.score
        );
    }

    let embedded = JudgeVerdict::parse("My final answer is 8/10.", &rubric).unwrap();
    assert!(close(embedded.score, 0.8));
}

#[test]
fn verdict_parse_accepts_alias_keys() {
    let total = JudgeVerdict::parse(
        r#"{"total_score": 3, "reasoning": "partial"}"#,
        &JudgeRubric::default_rubric(),
    )
    .unwrap();
    assert!(close(total.score, 0.3));
    assert!(close(total.raw_score, 3.0));
    assert_eq!(total.reasoning, "partial");

    let five_point = JudgeRubric::new(
        vec![JudgeCriterion::new("accuracy", "is it right")],
        5.0,
        0.6,
    );
    let rating = JudgeVerdict::parse(r#"{"rating": "4/5"}"#, &five_point).unwrap();
    assert!(close(rating.score, 0.8));
    assert!(close(rating.raw_score, 4.0));
}

#[test]
fn verdict_parse_scales_and_clamps_raw_scores() {
    let rubric = JudgeRubric::default_rubric();

    let above_scale = JudgeVerdict::parse(r#"{"score": 99}"#, &rubric).unwrap();
    assert!(close(above_scale.score, 1.0));
    assert!(close(above_scale.raw_score, 99.0));

    let negative = JudgeVerdict::parse(r#"{"score": -2}"#, &rubric).unwrap();
    assert!(close(negative.score, 0.0));
    assert!(close(negative.raw_score, -2.0));

    let five_point = JudgeRubric::new(
        vec![JudgeCriterion::new("accuracy", "is it right")],
        5.0,
        0.6,
    );
    let on_scale = JudgeVerdict::parse(r#"{"score": 5}"#, &five_point).unwrap();
    assert!(close(on_scale.score, 1.0));
    assert!(close(on_scale.raw_score, 5.0));
}

#[test]
fn verdict_passed_follows_the_rubric_threshold() {
    let strict = JudgeRubric::new(
        vec![JudgeCriterion::new("accuracy", "is it right")],
        10.0,
        0.9,
    );
    let verdict = JudgeVerdict::parse(r#"{"score": 8}"#, &strict).unwrap();
    assert!(close(verdict.score, 0.8));
    assert!(!verdict.passed);

    let lenient = JudgeRubric::new(
        vec![JudgeCriterion::new("accuracy", "is it right")],
        10.0,
        0.5,
    );
    assert!(
        JudgeVerdict::parse(r#"{"score": 8}"#, &lenient)
            .unwrap()
            .passed
    );
}

#[test]
fn verdict_parse_rejects_unparseable_responses() {
    let rubric = JudgeRubric::default_rubric();
    let error = JudgeVerdict::parse("I have no idea", &rubric).unwrap_err();
    assert_validation(&error, "could not parse a judge verdict");
    assert_validation(&error, "I have no idea");

    // The quoted excerpt is bounded, so a long response cannot bloat the error.
    let long = "no verdict ".repeat(100);
    let error = JudgeVerdict::parse(&long, &rubric).unwrap_err();
    assert!(
        error.to_string().chars().count() < 300,
        "error is not bounded: {error}"
    );
}

#[test]
fn judge_messages_render_rubric_and_case() {
    let rubric = JudgeRubric::default_rubric();
    let messages = build_judge_messages(&answer_case("case-1", "4"), &text("4"), &rubric, 0);

    assert_eq!(messages.len(), 2);
    assert!(messages[0].role.is_system());
    assert!(messages[1].role.is_user());

    let system = messages[0].text().into_owned();
    let user = messages[1].text().into_owned();

    for name in ["correctness", "completeness", "clarity"] {
        assert!(system.contains(name), "system prompt is missing {name}");
    }
    assert!(system.contains("correctness (weight 1)"));
    assert!(system.contains("completeness (weight 0.8)"));
    assert!(system.contains("clarity (weight 0.5)"));
    assert!(system.contains("0..10"));
    assert!(system.contains("strict JSON"));
    assert!(system.contains("per_criterion"));

    assert!(user.contains("2 + 2?"), "user prompt is missing the inputs");
    assert!(
        user.contains(r#"{"answer":"4"}"#),
        "user prompt is missing the label: {user}"
    );
    assert!(
        user.ends_with(r#""4""#),
        "user prompt is missing the prediction"
    );
}

#[test]
fn judge_messages_are_deterministic() {
    let case = answer_case("case-1", "4");
    let rubric = JudgeRubric::single("accuracy", "is it right");
    let first = build_judge_messages(&case, &text("4"), &rubric, 0);
    let second = build_judge_messages(&case, &text("4"), &rubric, 0);

    assert_eq!(first.len(), second.len());
    for (left, right) in first.iter().zip(second.iter()) {
        assert_eq!(left.role, right.role);
        assert_eq!(left.text(), right.text());
    }
}

#[test]
fn judge_messages_truncate_long_predictions() {
    let case = answer_case("case-1", "4");
    let rubric = JudgeRubric::default_rubric();
    let prediction = text(&"x".repeat(200));
    let rendered = prediction.to_string();

    let limit = 20;
    let messages = build_judge_messages(&case, &prediction, &rubric, limit);
    let user = messages[1].text().into_owned();
    let prefix: String = rendered.chars().take(limit).collect();
    let marker = format!("... [truncated {} chars]", rendered.chars().count() - limit);
    assert!(
        user.contains(&format!("{prefix}{marker}")),
        "prediction was not truncated at {limit} chars: {user}"
    );
    assert!(
        !user.contains(&rendered),
        "the full prediction leaked through"
    );

    // `0` disables truncation.
    let untruncated = build_judge_messages(&case, &prediction, &rubric, 0);
    let user = untruncated[1].text().into_owned();
    assert!(user.contains(&rendered));
    assert!(!user.contains("[truncated"));
}

#[tokio::test]
async fn judge_returns_a_parsed_verdict() {
    let model = ScriptedModel::new([ScriptedTurn::text(
        r#"{"score": 8, "reasoning": "close enough", "per_criterion": {"correctness": 8}}"#,
    )]);
    let judge = judge_over(&model, JudgeRubric::default_rubric(), 1);
    assert_eq!(judge.attempts(), 1);
    assert_eq!(judge.max_prediction_chars(), 8000);
    assert_eq!(judge.rubric().criteria().len(), 3);
    assert_eq!(judge.model().model_name(), "scripted-model");

    let verdict = judge
        .judge(&answer_case("case-1", "4"), &text("4"))
        .await
        .unwrap();
    assert!(close(verdict.score, 0.8));
    assert!(verdict.passed);
    assert_eq!(verdict.reasoning, "close enough");

    assert_eq!(model.call_count(), 1);
    let calls = model.calls();
    assert_eq!(calls[0].messages.len(), 2);
    assert!(calls[0].messages[0].role.is_system());
    assert!(calls[0].messages[1].role.is_user());
    assert!(calls[0].last_user_text().unwrap().contains("\"4\""));
}

#[tokio::test]
async fn judge_with_one_attempt_does_not_retry() {
    let model = ScriptedModel::new([ScriptedTurn::text("no verdict at all")]);
    let judge = judge_over(&model, JudgeRubric::default_rubric(), 1);

    let error = judge
        .judge(&answer_case("case-1", "4"), &text("4"))
        .await
        .unwrap_err();
    assert_validation(&error, "could not parse a judge verdict");
    assert_eq!(model.call_count(), 1);
}

#[tokio::test]
async fn judge_retries_an_unparseable_response() {
    let model = ScriptedModel::new([
        ScriptedTurn::text("not json at all"),
        ScriptedTurn::text(r#"{"score": 6}"#),
    ]);
    let judge = judge_over(&model, JudgeRubric::default_rubric(), 2);

    let verdict = judge
        .judge(&answer_case("case-1", "4"), &text("4"))
        .await
        .unwrap();
    assert!(close(verdict.score, 0.6));
    assert_eq!(model.call_count(), 2);
    assert_eq!(model.remaining(), 0, "a third call would have been made");
}

#[tokio::test]
async fn judge_propagates_model_errors_without_retrying() {
    let model = ScriptedModel::new([
        ScriptedTurn::fail("boom"),
        ScriptedTurn::text(r#"{"score": 9}"#),
    ]);
    let judge = judge_over(&model, JudgeRubric::default_rubric(), 3);

    let error = judge
        .judge(&answer_case("case-1", "4"), &text("4"))
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("boom"),
        "unexpected error: {error}"
    );
    assert!(
        !matches!(&error, Error::Validation(_)),
        "a model error must not be reported as a parse failure"
    );
    assert_eq!(model.call_count(), 1, "model errors are never retried");
    assert_eq!(model.remaining(), 1, "the retry turn must not be consumed");
}

#[tokio::test]
async fn evaluator_uses_the_verdict_score_and_reasoning() {
    let model = ScriptedModel::new([ScriptedTurn::text(
        r#"{"score": 8, "reasoning": "close enough", "per_criterion": {"correctness": 8}}"#,
    )]);
    let judge = Arc::new(judge_over(&model, JudgeRubric::default_rubric(), 1));
    let evaluator = LlmJudgeEvaluator::new().with_judge(judge);

    assert_eq!(evaluator.name(), "llm_as_judge");
    assert!(evaluator.judge().is_some());
    assert_eq!(evaluator.rubric().criteria().len(), 3);
    assert!(evaluator.fallback().higher_is_better());

    let evaluated = evaluator
        .evaluate(&answer_case("case-1", "4"), &text("4"))
        .await
        .unwrap();
    assert!(close(evaluated.score, 0.8));
    assert!(
        evaluated.reason.contains("close enough"),
        "reason: {}",
        evaluated.reason
    );
    let per_metric = evaluated.per_metric.unwrap();
    assert!(close(per_metric["llm_as_judge"], 0.8));
    assert!(close(per_metric["correctness"], 0.8));
}

#[tokio::test]
async fn evaluator_falls_back_to_exact_match_when_the_judge_fails() {
    let model = junk_model();
    let judge = Arc::new(judge_over(&model, JudgeRubric::default_rubric(), 1));
    let evaluator = LlmJudgeEvaluator::new()
        .with_judge(judge)
        .with_fallback(Arc::new(ExactMatchMetric::new()));
    assert_eq!(evaluator.fallback().name(), "exact_match");

    let hit = evaluator
        .evaluate(&answer_case("case-1", "4"), &text("4"))
        .await
        .unwrap();
    assert!(close(hit.score, 1.0));
    assert!(
        hit.reason.contains("the judge call failed"),
        "reason: {}",
        hit.reason
    );
    assert!(hit.reason.contains("exact_match"), "reason: {}", hit.reason);
    let per_metric = hit.per_metric.unwrap();
    assert!(close(per_metric["exact_match"], 1.0));

    let miss = evaluator
        .evaluate(&answer_case("case-1", "4"), &text("5"))
        .await
        .unwrap();
    assert!(close(miss.score, 0.0));
    assert!(
        miss.reason.contains("the judge call failed"),
        "reason: {}",
        miss.reason
    );
}

#[tokio::test]
async fn evaluator_without_a_judge_scores_by_exact_match() {
    let evaluator = LlmJudgeEvaluator::default();
    assert!(evaluator.judge().is_none());

    let hit = evaluator
        .evaluate(&answer_case("case-1", "4"), &text("4"))
        .await
        .unwrap();
    assert!(close(hit.score, 1.0));
    assert!(
        hit.reason.contains("no judge model is configured"),
        "reason: {}",
        hit.reason
    );

    let miss = evaluator
        .evaluate(&answer_case("case-1", "4"), &text("5"))
        .await
        .unwrap();
    assert!(close(miss.score, 0.0));
}

#[tokio::test]
async fn default_evaluator_scores_by_exact_match_without_a_model() {
    for name in ["exact_match", "accuracy"] {
        let evaluator = DefaultEvaluator::new(name);
        assert_eq!(evaluator.name(), "default");
        assert_eq!(evaluator.metric(), name);
        assert!(evaluator.judge().is_none());
        assert_eq!(evaluator.fallback().name(), "exact_match");

        let hit = evaluator
            .evaluate(&answer_case("case-1", "4"), &text("4"))
            .await
            .unwrap();
        assert!(close(hit.score, 1.0), "{name} scored {}", hit.score);

        let miss = evaluator
            .evaluate(&answer_case("case-1", "4"), &text("5"))
            .await
            .unwrap();
        assert!(close(miss.score, 0.0), "{name} scored {}", miss.score);
    }
    assert_eq!(DefaultEvaluator::exact_match().metric(), "exact_match");
}

#[tokio::test]
async fn default_evaluator_judges_with_an_installed_model() {
    let model = ScriptedModel::new([ScriptedTurn::text(
        r#"{"score": 7, "reasoning": "mostly right", "per_criterion": {"accuracy": 7}}"#,
    )]);
    let evaluator = DefaultEvaluator::new("exact_match").with_model(Arc::new(model.clone()));

    let evaluated = evaluator
        .evaluate(&answer_case("case-1", "4"), &text("4"))
        .await
        .unwrap();
    assert!(close(evaluated.score, 0.7));
    assert!(
        evaluated.reason.contains("mostly right"),
        "reason: {}",
        evaluated.reason
    );
    let per_metric = evaluated.per_metric.unwrap();
    assert!(close(per_metric["llm_as_judge"], 0.7));
    assert!(close(per_metric["accuracy"], 0.7));
    assert_eq!(model.call_count(), 1);
}

#[tokio::test]
async fn default_evaluator_falls_back_when_the_model_is_unusable() {
    let model = junk_model();
    let evaluator = DefaultEvaluator::new("exact_match").with_model(Arc::new(model.clone()));
    assert_eq!(evaluator.rubric().criteria().len(), 3);

    let evaluated = evaluator
        .evaluate(&answer_case("case-1", "4"), &text("4"))
        .await
        .unwrap();
    assert!(close(evaluated.score, 1.0));
    assert!(
        evaluated.reason.contains("the judge call failed"),
        "reason: {}",
        evaluated.reason
    );
    assert!(
        evaluated.reason.contains("exact_match"),
        "reason: {}",
        evaluated.reason
    );
    assert!(evaluated.per_metric.unwrap().contains_key("exact_match"));
    // The default judge policy is two attempts, and both were spent.
    assert_eq!(model.call_count(), 2);
}
