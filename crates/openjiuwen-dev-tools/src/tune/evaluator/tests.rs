//! Tests for the evaluator seam.

#[path = "telemetry_tests.rs"]
mod telemetry_tests;

use std::collections::BTreeMap;
use std::sync::Arc;

use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_foundation::tool::ToolInfo;
use openjiuwen_test_support::{MockCompletion, MockToolCall, ScriptedModel};
use serde_json::{json, Value};

use crate::tune::base::Case;
use crate::tune::tool::{RunOutcome, ToolCallRecord, ToolStub, ToolStubs, ToolTrace};

use super::runner::answer_to_map;
use super::scorer::score_pair;
use super::*;

/// Builds a case from `(key, value)` pairs.
fn case_of(inputs: &[(&str, &str)], label: &[(&str, &str)], id: &str) -> Case {
    let inputs = inputs
        .iter()
        .map(|(key, value)| ((*key).to_owned(), Value::String((*value).to_owned())))
        .collect();
    let label = label
        .iter()
        .map(|(key, value)| ((*key).to_owned(), Value::String((*value).to_owned())))
        .collect();
    Case::new(inputs, label).unwrap().with_case_id(id)
}

/// A two-case fixture with distinct answers.
fn cases() -> Vec<Case> {
    vec![
        case_of(
            &[("question", "capital of France")],
            &[("answer", "Paris")],
            "c1",
        ),
        case_of(&[("question", "2 + 2")], &[("answer", "4")], "c2"),
    ]
}

fn answer(entries: &[(&str, &str)]) -> BTreeMap<String, Value> {
    entries
        .iter()
        .map(|(key, value)| ((*key).to_owned(), Value::String((*value).to_owned())))
        .collect()
}

#[test]
fn render_case_inputs_uses_one_line_per_input() {
    let case = case_of(
        &[("question", "2 + 2?"), ("language", "english")],
        &[("answer", "4")],
        "c1",
    );
    assert_eq!(
        render_case_inputs(&case),
        "[language]: english\n[question]: 2 + 2?"
    );
}

#[test]
fn answer_to_map_prefers_json_objects() {
    assert_eq!(
        answer_to_map("```json\n{\"answer\": \"4\"}\n```"),
        answer(&[("answer", "4")])
    );
    assert_eq!(
        answer_to_map("{\"answer\": \"4\"}"),
        answer(&[("answer", "4")])
    );
    assert_eq!(
        answer_to_map("  four  "),
        answer(&[("output", "four")]),
        "plain text becomes the output entry"
    );
    assert_eq!(
        answer_to_map("[1, 2]"),
        answer(&[("output", "[1, 2]")]),
        "a JSON array is not an answer map"
    );
}

#[test]
fn label_scorer_bands() {
    let scorer = LabelScorer::new();
    let case = case_of(&[("q", "x")], &[("answer", "Paris")], "c1");
    let bands = [
        ("Paris", 1.0),
        ("  Paris  ", 1.0),
        ("paris", 0.9),
        ("PARIS", 0.9),
        ("the capital is Paris, France", 0.75),
        ("Par", 0.75),
        ("Lyon", 0.0),
    ];
    for (text, expected) in bands {
        let (score, reason) = scorer.score(&case, &answer(&[("answer", text)]));
        assert!(
            (score - expected).abs() < 1e-9,
            "{text:?} should score {expected}, got {score} ({reason})"
        );
    }
}

#[test]
fn label_scorer_maps_jaccard_into_the_band() {
    let scorer = LabelScorer::new();
    // Neither value contains the other, so the Jaccard band decides:
    // |intersection| / |union| = 2 / 4 = 0.5 maps to 0.30.
    let case = case_of(&[("q", "x")], &[("answer", "alpha beta gamma")], "c1");
    let (score, _) = scorer.score(&case, &answer(&[("answer", "alpha gamma delta")]));
    assert!((score - 0.3).abs() < 1e-9, "j=0.5 maps to 0.3: {score}");

    // 3 / 5 = 0.6 maps to 0.3 + 0.3 * (0.1 / 0.5) = 0.36.
    let case = case_of(&[("q", "x")], &[("answer", "alpha beta gamma delta")], "c1");
    let (score, _) = scorer.score(&case, &answer(&[("answer", "alpha beta gamma epsilon")]));
    assert!((score - 0.36).abs() < 1e-9, "j=0.6 maps to 0.36: {score}");

    // Below 0.5 the answer scores nothing.
    let (score, _) = scorer.score(&case, &answer(&[("answer", "alpha omega")]));
    assert!(score.abs() < 1e-9, "j=0.33 maps to 0: {score}");
}

#[test]
fn label_scorer_averages_keys_and_names_the_first_mismatch() {
    let scorer = LabelScorer::new();
    let case = case_of(&[("q", "x")], &[("first", "one"), ("second", "two")], "c1");
    let (score, reason) = scorer.score(&case, &answer(&[("first", "one"), ("second", "2")]));
    assert!(
        (score - 0.5).abs() < 1e-9,
        "one of two keys matched: {score}"
    );
    assert!(
        reason.contains("second"),
        "reason names the mismatch: {reason}"
    );

    let (score, reason) = scorer.score(&case, &answer(&[("first", "one"), ("second", "two")]));
    assert!((score - 1.0).abs() < 1e-9);
    assert!(reason.contains("matched exactly"), "{reason}");
}

#[test]
fn label_scorer_uses_a_single_answer_entry_for_missing_keys() {
    let scorer = LabelScorer::new();
    let case = case_of(&[("q", "x")], &[("answer", "4")], "c1");
    let (score, _) = scorer.score(&case, &answer(&[("output", "4")]));
    assert!((score - 1.0).abs() < 1e-9);
}

#[test]
fn label_scorer_reports_missing_answers_and_scores_are_bounded() {
    let scorer = LabelScorer::new();
    let case = case_of(&[("q", "x")], &[("answer", "4")], "c1");
    let (score, reason) = scorer.score(&case, &BTreeMap::new());
    assert!(score.abs() < 1e-9);
    assert!(reason.contains("missing"), "{reason}");

    let answers = [
        answer(&[("answer", "4")]),
        answer(&[("answer", "four")]),
        answer(&[("other", "4"), ("more", "5")]),
    ];
    for answer in answers {
        let (score, _) = scorer.score(&case, &answer);
        assert!((0.0..=1.0).contains(&score), "score {score} in range");
    }
}

#[test]
fn label_scorer_on_empty_label_is_zero() {
    let scorer = LabelScorer::new();
    let case = Case {
        inputs: answer(&[("q", "x")]),
        label: BTreeMap::new(),
        tools: None,
        case_id: "c1".to_owned(),
    };
    let (score, reason) = scorer.score(&case, &answer(&[("answer", "x")]));
    assert!(score.abs() < 1e-9);
    assert!(reason.contains("no keys"), "{reason}");
}

#[test]
fn score_pair_handles_empty_strings() {
    assert!((score_pair("", "") - 1.0).abs() < 1e-9);
    assert!(score_pair("", "x").abs() < 1e-9);
    assert!(score_pair("x", "").abs() < 1e-9);
}

#[tokio::test]
async fn model_prompt_runner_sends_prompt_and_inputs() {
    let model = ScriptedModel::text("Paris");
    let runner = ModelPromptRunner::new(Arc::new(model.clone()));
    let case = cases().remove(0);
    let answer = runner.run("You answer questions.", &case).await.unwrap();
    assert_eq!(answer, answer_map("output", "Paris"));

    let call = model.last_call().expect("one recorded call");
    assert_eq!(call.messages.len(), 2);
    assert_eq!(call.messages[0].text(), "You answer questions.");
    assert!(call.messages[0].role.is_system());
    assert_eq!(call.messages[1].text(), "[question]: capital of France");
    assert!(call.messages[1].role.is_user());
}

#[tokio::test]
async fn model_prompt_runner_parses_a_fenced_json_answer() {
    let model = ScriptedModel::text("```json\n{\"answer\": \"Paris\"}\n```");
    let runner = ModelPromptRunner::new(Arc::new(model));
    let answer = runner.run("prompt", &cases().remove(0)).await.unwrap();
    assert_eq!(answer, answer_map("answer", "Paris"));
}

#[tokio::test]
async fn model_prompt_runner_propagates_provider_errors() {
    let model = ScriptedModel::new([openjiuwen_test_support::ScriptedTurn::fail("boom")]);
    let runner = ModelPromptRunner::new(Arc::new(model));
    let error = runner.run("prompt", &cases().remove(0)).await.unwrap_err();
    assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Llm);
}

#[tokio::test]
async fn scripted_runner_replays_answers_and_records_calls() {
    let runner = ScriptedPromptRunner::new()
        .with_text_answer("c1", "Paris")
        .with_fallback(answer(&[("output", "fallback")]));
    let cases = cases();
    assert_eq!(
        runner.run("p1", &cases[0]).await.unwrap(),
        answer_map("output", "Paris")
    );
    assert_eq!(
        runner.run("p2", &cases[1]).await.unwrap(),
        answer_map("output", "fallback")
    );
    let calls = runner.calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].case_id, "c1");
    assert_eq!(calls[1].prompt, "p2");
    assert_eq!(runner.call_count(), 2);
}

#[tokio::test]
async fn scripted_runner_reports_unknown_cases() {
    let runner = ScriptedPromptRunner::new();
    let error = runner.run("p", &cases().remove(0)).await.unwrap_err();
    assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::NotFound);
    assert!(error.to_string().contains("c1"));
}

#[tokio::test]
async fn runner_evaluator_scores_through_the_runner() {
    let runner = Arc::new(
        ScriptedPromptRunner::new()
            .with_text_answer("c1", "Paris")
            .with_text_answer("c2", "5"),
    );
    let evaluator = RunnerEvaluator::new(runner);
    let cases = cases();
    let evaluated = evaluator.evaluate("prompt", &cases[0]).await.unwrap();
    assert!((evaluated.score - 1.0).abs() < 1e-9);
    assert_eq!(evaluated.answer, answer_map("output", "Paris"));
    assert_eq!(evaluated.case_id(), "c1");

    let (mean, all) = evaluator.score_cases("prompt", &cases).await.unwrap();
    assert_eq!(all.len(), 2);
    assert!(
        (mean - 0.5).abs() < 1e-9,
        "one of two cases correct: {mean}"
    );

    let (mean, empty) = evaluator.score_cases("prompt", &[]).await.unwrap();
    assert!(mean.abs() < 1e-9);
    assert_eq!(empty.len(), 0);
}

#[test]
fn runner_evaluator_accepts_a_custom_scorer() {
    /// A scorer that always returns a fixed score.
    struct Fixed;

    impl CaseScorer for Fixed {
        fn score(&self, _case: &Case, _answer: &BTreeMap<String, Value>) -> (f64, String) {
            (0.42, "fixed".to_owned())
        }
    }

    let runner = Arc::new(ScriptedPromptRunner::new());
    let evaluator = RunnerEvaluator::new(runner).with_scorer(Arc::new(Fixed));
    assert_eq!(
        evaluator
            .scorer()
            .score(&cases().remove(0), &BTreeMap::new())
            .0,
        0.42
    );
}

#[tokio::test]
async fn model_evaluator_runs_the_model_and_scores_the_answer() {
    let model = ScriptedModel::text("Paris");
    let evaluator = ModelEvaluator::new(Arc::new(model.clone()));
    let evaluated = evaluator
        .evaluate("prompt", &cases().remove(0))
        .await
        .unwrap();
    assert!((evaluated.score - 1.0).abs() < 1e-9);
    assert_eq!(model.call_count(), 1);
    assert!(
        evaluator
            .inner()
            .scorer()
            .score(
                &case_of(&[("q", "x")], &[("answer", "y")], "z"),
                &answer(&[("answer", "y")])
            )
            .0
            > 0.99
    );
}

#[test]
fn deterministic_score_rewards_structure_and_placeholders() {
    let evaluator = DeterministicEvaluator::new("You answer {{question}} briefly.");
    let case = case_of(&[("question", "2 + 2")], &[("answer", "4")], "c1");

    let seed = evaluator.score_parts(evaluator.seed(), &case);
    assert!((seed.placeholder - 1.0).abs() < 1e-9);
    assert!(seed.structure > 0.0);

    let bare = evaluator.score_parts("Short.", &case);
    assert!(
        bare.placeholder.abs() < 1e-9,
        "the placeholder was dropped: {bare:?}"
    );
    assert!(bare.total < seed.total);
}

#[test]
fn deterministic_coverage_tracks_case_terms() {
    let evaluator = DeterministicEvaluator::new("Answer the question.");
    let case = case_of(&[("question", "2 + 2")], &[("answer", "four")], "c1");
    let partial = evaluator.score_parts("Answer the question about 2.", &case);
    let full = evaluator.score_parts("Answer the question: 2 + 2 gives four.", &case);
    assert!(partial.coverage < full.coverage);
    assert!((full.coverage - 1.0).abs() < 1e-9, "{full:?}");
}

#[test]
fn deterministic_scores_are_case_dependent() {
    let evaluator = DeterministicEvaluator::new("Answer the question.");
    // `Paris` is a term of the first case only, so coverage differs.
    let prompt = "Answer the question about Paris in the requested output format.";
    let scores: Vec<f64> = cases()
        .iter()
        .map(|case| evaluator.score_parts(prompt, case).total)
        .collect();
    assert_ne!(
        scores[0], scores[1],
        "coverage differs per case: {scores:?}"
    );
}

#[test]
fn deterministic_blank_and_degenerate_candidates_score_low() {
    let evaluator = DeterministicEvaluator::new("You answer {{question}}.");
    let case = case_of(&[("question", "2 + 2")], &[("answer", "4")], "c1");
    assert!(evaluator.score_parts("", &case).total.abs() < 1e-9);
    assert!(evaluator.score_parts("   \n\t ", &case).total.abs() < 1e-9);

    let degenerate = evaluator.score_parts("!!!", &case);
    assert!(degenerate.hygiene <= 0.1, "{degenerate:?}");
    assert!(degenerate.total < 0.2, "{degenerate:?}");

    let tiny = evaluator.score_parts("ok", &case);
    assert!((tiny.hygiene - 0.5).abs() < 1e-9, "{tiny:?}");
}

#[test]
fn deterministic_hygiene_penalizes_repetition_and_length() {
    let evaluator = DeterministicEvaluator::new("Answer.");
    let case = case_of(&[("q", "x")], &[("answer", "y")], "c1");
    let repeated =
        "Answer the question.\nAnswer the question.\nAnswer the question.\nAnswer the question.";
    let parts = evaluator.score_parts(repeated, &case);
    assert!(parts.hygiene < 1.0, "{parts:?}");

    let long = "Answer the question. ".repeat(600);
    let parts = evaluator.score_parts(&long, &case);
    assert!(parts.hygiene <= 0.7, "{parts:?}");
}

#[test]
fn deterministic_reason_names_every_dimension() {
    let evaluator = DeterministicEvaluator::new("Answer {{question}}.");
    let parts = evaluator.score_parts(
        "Answer {{question}} step by step with the output format requested.",
        &case_of(&[("question", "x")], &[("answer", "y")], "c1"),
    );
    let reason = parts.reason();
    for dimension in ["placeholder", "structure", "coverage", "hygiene"] {
        assert!(reason.contains(dimension), "{reason}");
    }
}

#[tokio::test]
async fn deterministic_evaluator_scores_cases_without_a_runner() {
    let evaluator = DeterministicEvaluator::new("Answer {{question}}.");
    let cases = cases();
    let evaluated = evaluator
        .evaluate("Answer {{question}} in the requested format.", &cases[0])
        .await
        .unwrap();
    assert!(evaluated.answer.is_empty());
    assert!(evaluated.reason.contains("deterministic score"));
    let (mean, all) = evaluator
        .score_cases("Answer the question.", &cases)
        .await
        .unwrap();
    assert_eq!(all.len(), 2);
    assert!(mean > 0.0 && mean <= 1.0);
}

#[test]
fn deterministic_scores_are_repeatable() {
    let evaluator = DeterministicEvaluator::new("Answer {{question}}.");
    let candidate = "Answer {{question}} step by step and verify the result.";
    let case = cases().remove(0);
    let first = evaluator.score_parts(candidate, &case);
    let second = evaluator.score_parts(candidate, &case);
    assert_eq!(first, second);
    assert_eq!(first.total, second.total);
}

/// Builds a one-entry answer map.
fn answer_map(key: &str, value: &str) -> BTreeMap<String, Value> {
    let mut map = BTreeMap::new();
    map.insert(key.to_owned(), Value::String(value.to_owned()));
    map
}

#[test]
fn deterministic_score_serializes_all_dimensions() {
    let score = DeterministicScore {
        total: 0.5,
        placeholder: 1.0,
        structure: 0.25,
        coverage: 0.5,
        hygiene: 1.0,
    };
    let value = serde_json::to_value(score).unwrap();
    for key in ["total", "placeholder", "structure", "coverage", "hygiene"] {
        assert!(value.get(key).is_some(), "missing {key}");
    }
    let restored: DeterministicScore = serde_json::from_value(json!({
        "total": 0.5,
        "extra": true
    }))
    .unwrap();
    assert!((restored.total - 0.5).abs() < 1e-9);
    assert!(
        restored.structure.abs() < 1e-9,
        "missing dimensions default"
    );
}

/// A case declaring the given tools, built from the shared case fixture.
fn tool_case(id: &str, tool_names: &[&str]) -> Case {
    let tools = tool_names
        .iter()
        .map(|name| ToolInfo::new(*name, format!("{name} tool"), json!({"type": "object"})))
        .collect();
    case_of(&[("question", "2 + 2")], &[("answer", "4")], id).with_tools(tools)
}

/// A stub answering every call with the same fixture.
fn stub(name: &str) -> ToolStub {
    ToolStub::constant(name, json!({"ok": true})).unwrap()
}

/// A successful trace record for `name`.
fn called(name: &str) -> ToolCallRecord {
    ToolCallRecord::success(name, json!({}), json!({"ok": true}))
}

/// A runner that answers every case in prose and reports one tool call.
struct TracedRunner {
    /// The tool name reported as used.
    tool: String,
}

#[async_trait::async_trait]
impl PromptRunner for TracedRunner {
    async fn run(&self, _prompt: &str, _case: &Case) -> Result<BTreeMap<String, Value>> {
        Ok(answer_map("output", "Paris"))
    }

    async fn run_traced(&self, prompt: &str, case: &Case) -> Result<RunOutcome> {
        let answer = self.run(prompt, case).await?;
        Ok(RunOutcome::with_trace(
            answer,
            ToolTrace::new([called(&self.tool)]),
        ))
    }
}

#[tokio::test]
async fn tool_declarations_reach_the_model_request() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call("lookup", json!({"q": "x"})))
        .respond_text("final")
        .build();
    let recorder = model.clone();
    let runner = ModelPromptRunner::new(Arc::new(model))
        .with_stubs(ToolStubs::new().with_stub(stub("lookup")));
    let case = Case::from_text("2 + 2", "4")
        .unwrap()
        .with_case_id("c1")
        .with_tools(vec![ToolInfo::new(
            "lookup",
            "looks things up",
            json!({"type": "object"}),
        )]);

    let answer = runner.run("You answer questions.", &case).await.unwrap();
    assert_eq!(answer, answer_map("output", "final"));

    let calls = recorder.calls();
    assert_eq!(calls.len(), 2);
    let expected = json!([{
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "looks things up",
            "parameters": {"type": "object"}
        }
    }]);
    assert_eq!(calls[0].request.tools, Some(expected.clone()));
    assert_eq!(
        calls[1].request.tools,
        Some(expected),
        "the tool-result round repeats the declarations"
    );
}

#[tokio::test]
async fn cases_without_tools_send_no_declarations() {
    let model = ScriptedModel::text("4");
    let recorder = model.clone();
    let runner = ModelPromptRunner::new(Arc::new(model));
    let case = cases().remove(0);
    let outcome = runner.run_traced("prompt", &case).await.unwrap();
    assert_eq!(outcome.answer, answer_map("output", "4"));
    assert!(outcome.trace.is_empty());
    assert_eq!(recorder.call_count(), 1);
    assert!(
        recorder.calls()[0].request.tools.is_none(),
        "no declared tools means no tool declarations"
    );
}

#[tokio::test]
async fn executes_a_tool_call_and_returns_the_final_answer() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call("lookup", json!({"q": "paris"})))
        .respond_text("final")
        .build();
    let recorder = model.clone();
    let runner = ModelPromptRunner::new(Arc::new(model)).with_stubs(
        ToolStubs::new().with_stub(ToolStub::constant("lookup", json!({"city": "Paris"})).unwrap()),
    );
    let outcome = runner
        .run_traced("prompt", &tool_case("c1", &["lookup"]))
        .await
        .unwrap();
    assert_eq!(outcome.answer, answer_map("output", "final"));
    assert_eq!(outcome.trace.len(), 1);
    assert_eq!(outcome.trace.names(), vec!["lookup"]);
    let record = &outcome.trace.calls()[0];
    assert_eq!(record.arguments, json!({"q": "paris"}));
    assert_eq!(record.result, json!({"city": "Paris"}));
    assert!(record.error.is_none(), "{record:?}");

    assert_eq!(recorder.call_count(), 2);
    let calls = recorder.calls();
    let messages = &calls[1].messages;
    assert_eq!(messages.len(), 4);
    assert!(messages[2].role.is_assistant());
    assert_eq!(
        messages[2].tool_calls.len(),
        1,
        "the assistant turn carries the call"
    );
    assert_eq!(messages[2].tool_calls[0].name, "lookup");
    assert!(messages[3].role.is_tool());
    assert_eq!(messages[3].tool_call_id.as_deref(), Some("call_lookup"));
    assert_eq!(
        messages[3].text(),
        "{\"city\":\"Paris\"}",
        "the stub result is rendered compactly"
    );
}

#[tokio::test]
async fn executes_every_call_of_a_parallel_turn_in_order() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_calls(vec![
            MockToolCall::new("alpha", json!({"n": 1})).with_id("call_a"),
            MockToolCall::new("beta", json!({"n": 2})).with_id("call_b"),
        ]))
        .respond_text("done")
        .build();
    let recorder = model.clone();
    let runner = ModelPromptRunner::new(Arc::new(model)).with_stubs(
        ToolStubs::new()
            .with_stub(stub("alpha"))
            .with_stub(stub("beta")),
    );
    let outcome = runner
        .run_traced("prompt", &tool_case("c1", &["alpha", "beta"]))
        .await
        .unwrap();
    assert_eq!(outcome.trace.names(), vec!["alpha", "beta"]);

    let calls = recorder.calls();
    let ids: Vec<Option<&str>> = calls[1]
        .messages
        .iter()
        .filter(|message| message.role.is_tool())
        .map(|message| message.tool_call_id.as_deref())
        .collect();
    assert_eq!(ids, vec![Some("call_a"), Some("call_b")]);
}

#[tokio::test]
async fn records_an_unknown_tool_and_feeds_the_error_back() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call("lookup", json!({"q": "x"})))
        .respond_text("done")
        .build();
    let recorder = model.clone();
    let runner = ModelPromptRunner::new(Arc::new(model));
    let outcome = runner
        .run_traced("prompt", &tool_case("c1", &["lookup"]))
        .await
        .unwrap();
    assert_eq!(outcome.answer, answer_map("output", "done"));
    assert_eq!(outcome.trace.len(), 1);
    let record = &outcome.trace.calls()[0];
    let error = record
        .error
        .as_deref()
        .unwrap_or_else(|| panic!("the unknown tool is recorded: {record:?}"));
    assert!(
        error.contains("lookup"),
        "the error names the tool: {error}"
    );
    assert_eq!(record.result, Value::String(error.to_owned()));

    let calls = recorder.calls();
    let message = calls[1]
        .messages
        .iter()
        .find(|message| message.role.is_tool())
        .expect("the failure is fed back as a tool-result message");
    assert_eq!(message.text(), error);
}

#[tokio::test]
async fn records_a_failing_stub_and_feeds_the_error_back() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call("boom", json!({"q": "x"})))
        .respond_text("recovered")
        .build();
    let recorder = model.clone();
    let runner = ModelPromptRunner::new(Arc::new(model)).with_stubs(
        ToolStubs::new()
            .with_stub(ToolStub::new("boom", |_| Err(Error::validation("boom failed"))).unwrap()),
    );
    let outcome = runner
        .run_traced("prompt", &tool_case("c1", &["boom"]))
        .await
        .unwrap();
    assert_eq!(outcome.answer, answer_map("output", "recovered"));
    let record = &outcome.trace.calls()[0];
    assert!(record.is_error());
    assert_eq!(record.error.as_deref(), Some("validation: boom failed"));

    let calls = recorder.calls();
    let message = calls[1]
        .messages
        .iter()
        .find(|message| message.role.is_tool())
        .expect("the failure is fed back as a tool-result message");
    assert_eq!(message.text(), "validation: boom failed");
}

#[tokio::test]
async fn max_tool_rounds_must_be_positive() {
    let model = ScriptedModel::text("4");
    let recorder = model.clone();
    let runner = ModelPromptRunner::new(Arc::new(model)).with_max_tool_rounds(0);
    let error = runner
        .run("prompt", &cases().remove(0))
        .await
        .expect_err("a zero budget is rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("max_tool_rounds"), "{error}");
    assert_eq!(
        recorder.call_count(),
        0,
        "the budget is checked before any model call"
    );
    assert_eq!(runner.max_tool_rounds(), 0);
}

#[tokio::test]
async fn max_tool_rounds_bounds_a_tool_looping_model() {
    /// The turn a looping model keeps repeating.
    fn looping() -> MockCompletion {
        MockCompletion::tool_call("lookup", json!({"q": "x"}))
    }

    let model = ScriptedModel::builder()
        .completion(looping())
        .completion(looping())
        .completion(looping())
        .build();
    let recorder = model.clone();
    let runner = ModelPromptRunner::new(Arc::new(model))
        .with_stubs(ToolStubs::new().with_stub(stub("lookup")))
        .with_max_tool_rounds(2);
    let outcome = runner
        .run_traced("prompt", &tool_case("c1", &["lookup"]))
        .await
        .unwrap();
    assert_eq!(recorder.call_count(), 3);
    assert_eq!(outcome.trace.len(), 2);
    assert_eq!(
        outcome.answer,
        answer_map("output", ""),
        "the exhausted turn's empty text becomes the answer"
    );
}

#[test]
fn tool_usage_scorer_requires_every_required_tool() {
    let case = case_of(&[("q", "x")], &[("answer", "4")], "c1");
    let answer = answer_map("output", "4");
    let used = ToolTrace::new([called("lookup")]);

    let scorer = ToolUsageScorer::new(vec!["lookup".to_owned()]).unwrap();
    let (score, reason) = scorer.score_with_trace(&case, &answer, &used);
    assert!((score - 1.0).abs() < 1e-9, "{score}: {reason}");
    assert!(!reason.contains("missing"), "{reason}");

    let (score, reason) = scorer.score_with_trace(&case, &answer, &ToolTrace::default());
    assert!(score.abs() < 1e-9, "an unused required tool scores nothing");
    assert!(
        reason.starts_with("missing required tool(s): lookup"),
        "{reason}"
    );
    assert!(
        reason.contains("matched exactly"),
        "the inner reason is kept: {reason}"
    );

    let two = ToolUsageScorer::new(vec!["lookup".to_owned(), "search".to_owned()]).unwrap();
    let (score, reason) = two.score_with_trace(&case, &answer, &used);
    assert!(
        (score - 0.5).abs() < 1e-9,
        "half the required tools: {score}"
    );
    assert!(reason.contains("search"), "{reason}");
    assert_eq!(two.missing(&used), vec!["search".to_owned()]);

    let (score, reason) = scorer.score(&case, &answer);
    assert!(score.abs() < 1e-9, "the untraced score assumes no tool ran");
    assert!(
        reason.starts_with("missing required tool(s): lookup"),
        "{reason}"
    );

    let from_case = ToolUsageScorer::from_case(&tool_case("c2", &["lookup", "search"])).unwrap();
    assert_eq!(
        from_case
            .required()
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        vec!["lookup", "search"]
    );
    assert!(
        ToolUsageScorer::from_case(&case)
            .unwrap()
            .required()
            .is_empty(),
        "a case without tools requires nothing"
    );

    for blank in ["", "   ", "\t"] {
        let error = ToolUsageScorer::new(vec![blank.to_owned()]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("required"), "{error}");
    }
    let empty = ToolUsageScorer::new(Vec::new()).unwrap();
    let (score, _) = empty.score_with_trace(&case, &answer, &ToolTrace::default());
    assert!(
        (score - 1.0).abs() < 1e-9,
        "no requirement leaves the inner score unchanged"
    );

    /// A scorer with a fixed score and a fixed reason.
    struct Fixed;

    impl CaseScorer for Fixed {
        fn score(&self, _case: &Case, _answer: &BTreeMap<String, Value>) -> (f64, String) {
            (0.8, "fixed".to_owned())
        }
    }

    let custom = ToolUsageScorer::new(vec!["lookup".to_owned()])
        .unwrap()
        .with_inner(Box::new(Fixed));
    let (score, reason) = custom.score_with_trace(&case, &answer, &ToolTrace::default());
    assert!(score.abs() < 1e-9);
    assert!(reason.ends_with("; fixed"), "{reason}");
    let (score, reason) = custom.score_with_trace(&case, &answer, &used);
    assert!((score - 0.8).abs() < 1e-9);
    assert_eq!(reason, "fixed");
    assert!(format!("{custom:?}").contains("lookup"), "{custom:?}");
}

#[tokio::test]
async fn runner_evaluator_reports_traces() {
    let plain = RunnerEvaluator::new(Arc::new(
        ScriptedPromptRunner::new()
            .with_text_answer("c1", "Paris")
            .with_text_answer("c2", "4"),
    ));
    let cases = cases();
    let evaluated = plain.evaluate("prompt", &cases[0]).await.unwrap();
    let (traced, trace) = plain.evaluate_traced("prompt", &cases[0]).await.unwrap();
    assert!(trace.is_empty(), "a scripted runner runs no tools");
    assert_eq!(traced, evaluated);

    let (mean, all) = plain.score_cases("prompt", &cases).await.unwrap();
    let (traced_mean, traced_all) = plain.score_cases_traced("prompt", &cases).await.unwrap();
    assert!((traced_mean - mean).abs() < 1e-12);
    assert!((traced_mean - 1.0).abs() < 1e-9);
    assert_eq!(traced_all.len(), all.len());
    for ((traced_case, trace), case) in traced_all.iter().zip(&all) {
        assert!(trace.is_empty());
        assert_eq!(traced_case, case);
    }
    assert!(tool_usage(&traced_all).is_empty());

    let (empty_mean, empty) = plain.score_cases_traced("prompt", &[]).await.unwrap();
    assert!(empty_mean.abs() < 1e-9);
    assert_eq!(empty.len(), 0);

    let traced = RunnerEvaluator::new(Arc::new(TracedRunner {
        tool: "lookup".to_owned(),
    }));
    let (_, results) = traced.score_cases_traced("prompt", &cases).await.unwrap();
    let usage = tool_usage(&results);
    assert_eq!(usage.get("lookup"), Some(&2), "{usage:?}");
    assert_eq!(usage.len(), 1);
}

#[tokio::test]
async fn runner_evaluator_executes_tools_through_the_model_runner() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call("lookup", json!({"q": "paris"})))
        .respond_text("4")
        .build();
    let recorder = model.clone();
    let runner: Arc<dyn PromptRunner> = Arc::new(
        ModelPromptRunner::new(Arc::new(model))
            .with_stubs(ToolStubs::new().with_stub(stub("lookup"))),
    );
    let case = tool_case("c1", &["lookup"]);
    let evaluator = RunnerEvaluator::new(Arc::clone(&runner));
    let (evaluated, trace) = evaluator.evaluate_traced("prompt", &case).await.unwrap();
    assert!(
        (evaluated.score - 1.0).abs() < 1e-9,
        "the tool-backed answer is correct: {}",
        evaluated.reason
    );
    assert_eq!(trace.names(), vec!["lookup"]);
    assert_eq!(recorder.call_count(), 2);

    let strict = RunnerEvaluator::new(runner).with_scorer(Arc::new(
        ToolUsageScorer::new(vec!["search".to_owned()]).unwrap(),
    ));
    let evaluated = strict.evaluate("prompt", &case).await.unwrap();
    assert!(
        evaluated.score.abs() < 1e-9,
        "a never-called required tool drops the score: {evaluated:?}"
    );
    assert!(evaluated.reason.contains("search"), "{}", evaluated.reason);
}

#[tokio::test]
async fn model_evaluator_with_stubs_is_tool_aware() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call("lookup", json!({"q": "paris"})))
        .respond_text("4")
        .build();
    let recorder = model.clone();
    let evaluator =
        ModelEvaluator::with_stubs(Arc::new(model), ToolStubs::new().with_stub(stub("lookup")));
    let (evaluated, trace) = evaluator
        .evaluate_traced("prompt", &tool_case("c1", &["lookup"]))
        .await
        .unwrap();
    assert_eq!(trace.names(), vec!["lookup"]);
    assert!((evaluated.score - 1.0).abs() < 1e-9);
    assert_eq!(recorder.call_count(), 2);
}
