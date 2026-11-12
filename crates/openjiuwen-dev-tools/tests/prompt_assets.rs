#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Crate-level seam tests for the prompt assets.
//!
//! The module unit tests cover each builder in isolation; this suite exercises
//! the seams *between* modules:
//!
//! - [`openjiuwen_dev_tools::tune`] is the data model
//!   [`openjiuwen_dev_tools::prompt_builder`] consumes: `Case`/`EvaluatedCase`
//!   must accept and emit Python's JSON field names.
//! - The three builders must keep the Python model-call sequence and feed the
//!   `tune` values into the rendered templates.
//!
//! Every assertion carries the invariant it protects, and every test is
//! offline and deterministic (`ScriptedModel`, fixed ASCII prompts).

use std::collections::BTreeMap;
use std::sync::Arc;

use openjiuwen_dev_tools::prompt_builder::{
    bank, BadCasePromptBuilder, FeedbackMode, FeedbackOptions, FeedbackPromptBuilder, Language,
    MetaBuildOptions, MetaTemplateBuilder, TemplateType, INSERT_STR,
};
use openjiuwen_dev_tools::tune::{Case, EvaluatedCase};
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::prompt::Role;
use openjiuwen_foundation::tool::ToolInfo;
use openjiuwen_test_support::ScriptedModel;
use serde_json::{json, Value};

/// The OpenAI-shape tool descriptor used by the plan-mode and round-trip tests.
fn tool_value() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {}}
        }
    })
}

/// The Python-field JSON payload of one `tune::Case`.
fn case_value() -> Value {
    json!({
        "inputs": {"question": "what is 2 + 2?"},
        "label": {"answer": "4"},
        "tools": [tool_value()],
        "case_id": "case-1"
    })
}

/// A one-entry JSON map, matching the shape `tune` stores in its maps.
fn json_map(pairs: &[(&str, &str)]) -> BTreeMap<String, Value> {
    pairs
        .iter()
        .map(|(key, value)| ((*key).to_owned(), Value::String((*value).to_owned())))
        .collect()
}

/// An evaluated case whose inputs/answer/reason are recognizable in the
/// rendered analyze message.
fn evaluated_case() -> EvaluatedCase {
    let case = Case::new(
        json_map(&[("question", "capital of France?")]),
        json_map(&[("answer", "Paris")]),
    )
    .expect("a non-empty inputs and label map make a valid case")
    .with_case_id("case-42");
    EvaluatedCase::new(case, json_map(&[("answer", "Lyon")]), 0.0, "wrong city")
        .expect("a 0.0 score is inside the accepted bounds")
}

/// A recording model queued with `answers` and the `Arc<dyn Model>` view the
/// builders take.
fn scripted(answers: &[&str]) -> (ScriptedModel, Arc<dyn Model>) {
    let mut builder = ScriptedModel::builder().model_name("scripted-test");
    for answer in answers {
        builder = builder.respond_text(*answer);
    }
    let recorder = builder.build();
    let model: Arc<dyn Model> = Arc::new(recorder.clone());
    (recorder, model)
}

#[test]
fn evaluated_cases_round_trip_through_python_field_names() {
    let case_json = case_value();
    let case: Case = serde_json::from_value(case_json.clone()).expect(
        "tune::Case must deserialize from Python's field names \
         (inputs/label/tools/case_id)",
    );
    assert_eq!(
        case.case_id, "case-1",
        "the supplied case_id must survive deserialization instead of being regenerated"
    );
    assert_eq!(
        case.inputs,
        json_map(&[("question", "what is 2 + 2?")]),
        "the inputs map must keep its Python field contents"
    );
    assert_eq!(
        case.tools.as_deref(),
        Some(
            [ToolInfo::new(
                "search",
                "Search the web",
                json!({"type": "object", "properties": {}})
            )]
            .as_slice()
        ),
        "the tools list must deserialize into the OpenAI-shaped ToolInfo"
    );

    let case_back = serde_json::to_value(&case)
        .expect("tune::Case must serialize back to the Python field names");
    assert_eq!(
        case_back, case_json,
        "a tune::Case must round-trip byte-for-byte through its Python JSON shape"
    );

    let evaluated_json = json!({
        "case": case_json,
        "answer": {"answer": "Lyon"},
        "score": 0.25,
        "reason": "wrong city"
    });
    let evaluated: EvaluatedCase = serde_json::from_value(evaluated_json.clone()).expect(
        "tune::EvaluatedCase must deserialize from Python's field names \
         (case/answer/score/reason)",
    );
    evaluated
        .validate()
        .expect("a 0.25 score is inside the accepted 0.0..=1.0 bounds");
    assert!(
        (evaluated.score - 0.25).abs() < f64::EPSILON,
        "the score field must map onto EvaluatedCase::score, got {}",
        evaluated.score
    );
    assert_eq!(
        evaluated.case_id(),
        "case-1",
        "the accessor must forward to the nested Python-named case"
    );

    let evaluated_back = serde_json::to_value(&evaluated)
        .expect("tune::EvaluatedCase must serialize back to the Python field names");
    assert_eq!(
        evaluated_back, evaluated_json,
        "an EvaluatedCase must round-trip byte-for-byte through its Python JSON shape"
    );
    let object = evaluated_back
        .as_object()
        .expect("the evaluated-case payload is a JSON object");
    for key in ["case", "answer", "score", "reason"] {
        assert!(
            object.contains_key(key),
            "the serialized EvaluatedCase must keep the Python key {key:?}"
        );
    }
}

#[tokio::test]
async fn bad_case_prompt_builder_makes_exactly_two_calls() {
    // The analyze answer uses the `<intent>`/`<summary>` shape
    // `utils::parse_bad_case_analysis` reads; the optimize answer is the
    // repaired prompt handed back unchanged.
    let analysis = "<intent>true</intent>\n<summary>the answer names the wrong city</summary>";
    let (recorder, model) = scripted(&[analysis, "OPTIMIZED_PROMPT"]);
    let builder = BadCasePromptBuilder::new(model);
    let cases = vec![evaluated_case()];

    let answer = builder
        .build("Answer the question with a city.", &cases, Language::ZhCn)
        .await
        .expect("a non-empty prompt and one valid case make the repair succeed");

    assert_eq!(
        answer, "OPTIMIZED_PROMPT",
        "the returned text must be the second (optimize) model answer verbatim"
    );
    assert_eq!(
        recorder.call_count(),
        2,
        "bad-case repair must make exactly the analyze and optimize calls"
    );

    let calls = recorder.calls();
    let analyze_user = calls[0]
        .last_user_text()
        .expect("the analyze call must carry a user message");
    assert!(
        analyze_user.contains("Answer the question with a city."),
        "the analyze message must embed the original prompt: {analyze_user}"
    );
    assert!(
        analyze_user.contains(r#"{"question":"capital of France?"}"#),
        "the analyze message must embed the tune::Case inputs rendered by \
         render_bad_cases: {analyze_user}"
    );
    assert!(
        analyze_user.contains(r#"{"answer":"Paris"}"#),
        "the analyze message must embed the tune::Case label rendered by \
         render_bad_cases: {analyze_user}"
    );
    assert!(
        analyze_user.contains(r#"{"answer":"Lyon"}"#),
        "the analyze message must embed the EvaluatedCase answer: {analyze_user}"
    );
    assert!(
        analyze_user.contains("wrong city"),
        "the analyze message must embed the EvaluatedCase reason: {analyze_user}"
    );

    let optimize_user = calls[1]
        .last_user_text()
        .expect("the optimize call must carry a user message");
    assert!(
        optimize_user.contains("the answer names the wrong city"),
        "the optimize message must carry the parsed <summary> as feedback: {optimize_user}"
    );
}

#[tokio::test]
async fn meta_template_plan_mode_sends_tools_json() {
    let (recorder, model) = scripted(&["PLAN_PROMPT"]);
    let builder = MetaTemplateBuilder::new(model);
    let tools = vec![ToolInfo::new(
        "search",
        "Search the web",
        json!({"type": "object", "properties": {}}),
    )];
    let options = MetaBuildOptions {
        tools: tools.clone(),
        template_type: TemplateType::Plan,
        ..MetaBuildOptions::default()
    };
    let task = "Book a flight to Paris";

    let answer = builder
        .build(task, &options)
        .await
        .expect("a non-empty task makes plan-mode assembly succeed");

    assert_eq!(
        answer, "PLAN_PROMPT",
        "standardization returns the model answer verbatim"
    );
    assert_eq!(
        recorder.call_count(),
        1,
        "meta-template standardization makes exactly one model call"
    );

    let call = recorder
        .last_call()
        .expect("the single standardization call must be recorded");
    assert_eq!(
        call.messages.len(),
        2,
        "plan mode sends exactly a system and a user message"
    );
    assert_eq!(
        call.messages[0].role,
        Role::System,
        "the first message must be the system prompt"
    );
    assert_eq!(
        call.messages[0].text(),
        bank(Language::ZhCn).meta_plan_system,
        "plan mode must send the plan system template byte for byte"
    );
    assert_eq!(
        call.messages[1].role,
        Role::User,
        "the second message must be the user prompt"
    );

    let user = call.messages[1].text();
    assert!(
        user.contains(task),
        "the user message must interpolate the task instruction: {user}"
    );
    let compact_tools =
        serde_json::to_string(&tools).expect("the OpenAI-shaped tool descriptor serializes");
    assert!(
        user.contains(&compact_tools),
        "the user message must interpolate the compact JSON of the tool descriptors: {user}"
    );
}

#[tokio::test]
async fn feedback_insert_mode_splices_the_marker() {
    let intent =
        "```json\n{\"intent\": true, \"optimized_feedback\": \"mention the weather\"}\n```";
    let (recorder, model) = scripted(&[intent, "REWRITTEN_PROMPT"]);
    let builder = FeedbackPromptBuilder::new(model);
    let prompt = "Summarize the report";
    let start = 9; // between "Summarize" and " the report"
    let options = FeedbackOptions {
        feedback: "add a weather note".to_owned(),
        mode: FeedbackMode::Insert,
        start_pos: Some(start),
        ..FeedbackOptions::default()
    };

    let answer = builder
        .build(prompt, &options)
        .await
        .expect("a non-empty prompt, feedback, and in-range start_pos make the splice succeed");

    assert_eq!(
        answer, "REWRITTEN_PROMPT",
        "feedback rewriting returns the second (optimize) model answer verbatim"
    );
    assert_eq!(
        recorder.call_count(),
        2,
        "feedback rewriting must make exactly the intent and optimize calls"
    );

    let calls = recorder.calls();
    let intent_user = calls[0]
        .last_user_text()
        .expect("the intent call must carry a user message");
    assert!(
        intent_user.contains(prompt),
        "the intent call must see the original, untagged prompt: {intent_user}"
    );
    assert!(
        !intent_user.contains(INSERT_STR),
        "the intent call must not see the insert marker: {intent_user}"
    );

    let optimize_user = calls[1]
        .last_user_text()
        .expect("the optimize call must carry a user message");
    let prefix: String = prompt.chars().take(start).collect();
    let suffix: String = prompt.chars().skip(start).collect();
    let spliced = format!("{prefix}{INSERT_STR}{suffix}");
    assert_eq!(
        spliced, "Summarize[用户要插入的位置] the report",
        "character index 9 must fall between \"Summarize\" and \" the report\""
    );
    assert!(
        optimize_user.contains(&spliced),
        "the optimize message must render the prompt with {INSERT_STR} spliced at \
         character index {start}: {optimize_user}"
    );
    assert!(
        optimize_user.contains("mention the weather"),
        "the optimize message must carry the normalized intent feedback: {optimize_user}"
    );
}
