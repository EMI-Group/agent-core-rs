#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(
    unreachable_pub,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::new_without_default,
    clippy::module_name_repetitions,
    clippy::unnecessary_literal_bound
)]
//! Behavior of the `intent_detection` component and its `IntentRouter`.
//!
//! Ports `tests/unit_tests/core/component/test_intent_detection_comp.py` plus the
//! Rust-only rule path. Both paths are covered offline: the model path drives
//! `openjiuwen_test_support::ScriptedModel`, the rule path needs no model at all.
//! The workflow test proves the routing contract — the classification is written to the
//! graph state as a normal output and `IntentRouter` picks the successor from it.
mod common;

use std::collections::BTreeMap;
use std::sync::Arc;

use openjiuwen_session::Session;
use openjiuwen_test_support::ScriptedModel;
use serde_json::{json, Value};

use openjiuwen_workflow::components::intent_detection::{
    IntentDetectionCompConfig, IntentDetectionComponent, IntentRouter,
};
use openjiuwen_workflow::{
    Component, ComponentContext, ComponentOptions, EndComponent, StartComponent, Workflow,
    WorkflowCard,
};

use common::{card, session};

/// A context over a fresh session.
fn context(session_id: &str) -> ComponentContext {
    ComponentContext::new(session(session_id), "intent", "wf")
}

/// A valid English configuration over `weather` (class 1) and `travel` (class 2).
fn english_config() -> IntentDetectionCompConfig {
    IntentDetectionCompConfig::default()
        .with_category_names(vec!["weather".to_owned(), "travel".to_owned()])
        .with_accept_language("en")
}

/// A valid Chinese configuration over `天气` (class 1) and `旅行` (class 2).
fn chinese_config() -> IntentDetectionCompConfig {
    IntentDetectionCompConfig::default()
        .with_category_names(vec!["天气".to_owned(), "旅行".to_owned()])
}

fn component(config: IntentDetectionCompConfig) -> IntentDetectionComponent {
    IntentDetectionComponent::new(config).expect("the config is valid")
}

// port of tests/unit_tests/core/component/test_intent_detection_comp.py::test_invoke_success
#[tokio::test]
async fn the_rule_path_selects_the_matching_category() {
    let config = english_config().with_keywords(BTreeMap::from([
        ("weather".to_owned(), vec!["weather".to_owned()]),
        ("travel".to_owned(), vec!["flight".to_owned()]),
    ]));
    let intent = component(config);

    let output = intent
        .invoke(
            json!({"query": "I need a flight to Paris"}),
            &context("rule-hit"),
        )
        .await
        .unwrap();

    assert_eq!(
        output,
        json!({
            "classification_id": 2,
            "category_name": "travel",
            "reason": "keyword matched: flight"
        })
    );
}

#[tokio::test]
async fn the_rule_path_falls_back_to_the_default_class() {
    let intent = component(english_config());

    let output = intent
        .invoke(
            json!({"query": "something unrelated"}),
            &context("rule-miss"),
        )
        .await
        .unwrap();

    assert_eq!(output["classification_id"], json!(0));
    assert_eq!(output["category_name"], json!("Default intent"));
    assert_eq!(output["reason"], json!("no keyword matched"));
}

// port of tests/unit_tests/core/component/test_intent_detection_comp.py::test_invoke_success_accept_language_en
#[tokio::test]
async fn the_model_path_maps_an_english_reply_to_the_configured_name() {
    let model =
        ScriptedModel::text(r#"{"class": "Category2", "reason": "User asks about travel"}"#);
    let intent = component(english_config().with_model(Arc::new(model.clone())));

    let output = intent
        .invoke(json!({"query": "I want to travel"}), &context("model-en"))
        .await
        .unwrap();

    assert_eq!(output["classification_id"], json!(2));
    assert_eq!(output["category_name"], json!("travel"));
    assert_eq!(output["reason"], json!("User asks about travel"));
    assert_eq!(model.call_count(), 1, "the model is called exactly once");
}

// port of tests/unit_tests/core/component/test_intent_detection_comp.py::test_invoke_success
#[tokio::test]
async fn the_model_path_maps_a_chinese_reply_to_the_configured_name() {
    let model = ScriptedModel::text(r#"{"class": "分类2"}"#);
    let intent = component(chinese_config().with_model(Arc::new(model.clone())));

    let output = intent
        .invoke(json!({"query": "我打算去旅行"}), &context("model-zh"))
        .await
        .unwrap();

    assert_eq!(output["classification_id"], json!(2));
    assert_eq!(output["category_name"], json!("旅行"));
    assert!(
        output.get("reason").is_none(),
        "an empty reason is omitted: {output}"
    );
}

#[tokio::test]
async fn an_unknown_model_class_returns_the_default_class() {
    let model = ScriptedModel::text(r#"{"class": "Category9", "reason": "nope"}"#);
    let intent = component(english_config().with_model(Arc::new(model)));

    let output = intent
        .invoke(json!({"query": "anything"}), &context("model-unknown"))
        .await
        .unwrap();

    assert_eq!(output["classification_id"], json!(0));
    assert_eq!(output["category_name"], json!("Default intent"));
    let reason = output["reason"].as_str().unwrap();
    assert!(reason.contains("'Category9'"), "{reason}");
    assert!(
        reason.contains("['Category0', 'Category1', 'Category2']"),
        "{reason}"
    );
}

#[tokio::test]
async fn an_unparsable_model_reply_returns_the_default_class() {
    let model = ScriptedModel::text("I cannot answer that");
    let intent = component(english_config().with_model(Arc::new(model)));

    let output = intent
        .invoke(json!({"query": "anything"}), &context("model-garbage"))
        .await
        .unwrap();

    assert_eq!(output["classification_id"], json!(0));
    let reason = output["reason"].as_str().unwrap();
    assert!(reason.contains("I cannot answer that"), "{reason}");
    assert!(
        reason.contains("does not conform to valid JSON format"),
        "{reason}"
    );
}

#[tokio::test]
async fn a_failing_model_is_code_101052() {
    let model = ScriptedModel::builder()
        .fail(openjiuwen_test_support::ScriptedError::llm("boom"))
        .build();
    let intent = component(english_config().with_model(Arc::new(model)));

    let error = intent
        .invoke(json!({"query": "anything"}), &context("model-fail"))
        .await
        .unwrap_err();

    assert!(error.message().contains("[101052]"), "{error}");
    assert!(error.message().contains("boom"), "{error}");
}

// port of tests/unit_tests/core/component/test_intent_detection_comp.py::test_invoke_uses_completed_history_before_writing_current_turn
#[tokio::test]
async fn history_is_read_before_the_call_and_the_turn_is_appended_after() {
    let session: Arc<dyn Session> = session("model-history");
    session
        .update_state(json!({
            "__messages__": [
                {"role": "user", "content": "previous weather query"},
                {"role": "assistant", "content": r#"{"class": "Category1", "reason": "previous"}"#}
            ]
        }))
        .await
        .unwrap();
    let ctx = ComponentContext::new(Arc::clone(&session), "intent", "wf");

    let reply = r#"{"class": "Category2", "reason": "travel intent"}"#;
    let model = ScriptedModel::text(reply);
    let intent = component(
        english_config()
            .with_enable_history(true)
            .with_model(Arc::new(model.clone())),
    );

    let output = intent
        .invoke(json!({"query": "current travel query"}), &ctx)
        .await
        .unwrap();

    assert_eq!(output["classification_id"], json!(2));
    let call = model.last_call().expect("one model call");
    let user_text = call.last_user_text().expect("a user message");
    let history_part = user_text
        .split_once("Current input:")
        .expect("the prompt has a current-input section")
        .0;
    assert!(
        history_part.contains("User: previous weather query"),
        "{history_part}"
    );
    assert!(
        !history_part.contains("current travel query"),
        "the current turn is not part of the history prompt: {history_part}"
    );

    let state = session
        .get_state("__messages__")
        .await
        .unwrap()
        .expect("the history lane exists");
    let messages = state.as_array().expect("the history lane is an array");
    assert_eq!(messages.len(), 4, "{state}");
    assert_eq!(
        messages[2],
        json!({"role": "user", "content": "current travel query"})
    );
    assert_eq!(messages[3], json!({"role": "assistant", "content": reply}));
}

/// A workflow `start → intent → {end_a, end_b}` where `end_id` is the declared exit.
///
/// Only one end may be the workflow's exit, so each branch is asserted by building the
/// workflow twice with the matching exit; the other end stays a registered component.
fn intent_workflow(end_id: &str) -> Workflow {
    let keywords = BTreeMap::from([
        ("weather".to_owned(), vec!["weather".to_owned()]),
        ("travel".to_owned(), vec!["travel".to_owned()]),
    ]);
    let intent = component(english_config().with_keywords(keywords));
    let router = IntentRouter::new(
        "intent",
        vec![(1, vec!["end_a".to_owned()]), (2, vec!["end_b".to_owned()])],
        vec!["end_b".to_owned()],
    );
    let end_schema = json!({
        "id": "${intent.classification_id}",
        "name": "${intent.category_name}",
    });
    let mut workflow = Workflow::new(card("intent"));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"query": "${inputs.query}"})),
        )
        .add_workflow_comp(
            "intent",
            intent,
            ComponentOptions::new().with_inputs_schema(json!({"query": "${start.query}"})),
        );
    if end_id == "end_a" {
        workflow
            .set_end_comp(
                "end_a",
                EndComponent::new(),
                ComponentOptions::new().with_inputs_schema(end_schema.clone()),
            )
            .add_workflow_comp(
                "end_b",
                EndComponent::new(),
                ComponentOptions::new().with_inputs_schema(end_schema),
            );
    } else {
        workflow
            .set_end_comp(
                "end_b",
                EndComponent::new(),
                ComponentOptions::new().with_inputs_schema(end_schema.clone()),
            )
            .add_workflow_comp(
                "end_a",
                EndComponent::new(),
                ComponentOptions::new().with_inputs_schema(end_schema),
            );
    }
    workflow
        .add_connection("start", "intent")
        .add_conditional_connection("intent", "end_a", router.clone().router())
        .add_conditional_connection("intent", "end_b", router.router());
    workflow
}

/// The `id`/`name` payload of a run that ended in the declared exit.
fn payload(output: &Value) -> (Value, Value) {
    (
        output["output"]["id"].clone(),
        output["output"]["name"].clone(),
    )
}

// port of tests/unit_tests/core/workflow/test_workflow.py::test_start_node_with_intent_detection
#[tokio::test]
async fn the_router_selects_the_matching_end_node() {
    let workflow = intent_workflow("end_a");
    let output = workflow
        .invoke(
            json!({"query": "what is the weather"}),
            session("wf-intent-a"),
        )
        .await
        .unwrap();
    assert_eq!(
        payload(&output.result),
        (json!(1), json!("weather")),
        "class 1 routes to end_a"
    );

    let workflow = intent_workflow("end_b");
    let output = workflow
        .invoke(json!({"query": "book travel"}), session("wf-intent-b"))
        .await
        .unwrap();
    assert_eq!(
        payload(&output.result),
        (json!(2), json!("travel")),
        "class 2 routes to end_b"
    );
}

#[tokio::test]
async fn an_empty_fallback_ends_the_run_without_a_successor() {
    let router = IntentRouter::new("intent", vec![(1, vec!["end".to_owned()])], Vec::new());
    let keywords = BTreeMap::from([("travel".to_owned(), vec!["flight".to_owned()])]);
    let intent = component(english_config().with_keywords(keywords));
    let mut workflow = Workflow::new(WorkflowCard::new("empty-intent"));
    workflow
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp("intent", intent, ComponentOptions::new())
        .set_end_comp("end", EndComponent::new(), ComponentOptions::new())
        .add_connection("start", "intent")
        .add_conditional_connection("intent", "end", router.router());

    let output = workflow
        .invoke(
            json!({"query": "nothing matches"}),
            session("wf-intent-empty"),
        )
        .await
        .unwrap();

    assert_eq!(
        output.result,
        Value::Null,
        "an empty route activates no successor, so the exit never runs"
    );
}
