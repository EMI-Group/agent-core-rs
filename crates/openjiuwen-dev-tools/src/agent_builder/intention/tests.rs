//! Tests for the intention detector: the deterministic rules, the model path,
//! its fallbacks, and the serde surface.
//!
//! Every case runs offline on scripted models: no network, no environment reads,
//! and no wall-clock dependence.
use std::sync::Arc;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_foundation::llm::Model;
use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};
use serde_json::json;

use super::*;

/// Coerces a scripted model into the trait object the detector takes.
fn dynamic(model: &Arc<ScriptedModel>) -> Arc<dyn Model> {
    let cloned: Arc<ScriptedModel> = Arc::clone(model);
    cloned
}

/// A history holding one user turn.
fn history_with(turn: &str) -> HistoryManager {
    let mut history = HistoryManager::new(10);
    history.add_user_message(turn);
    history
}

#[tokio::test]
async fn deterministic_initial_rules_recognise_processes() {
    let detector = IntentionDetector::new();
    let empty = HistoryManager::new(10);
    for query in [
        "First read the file, then summarize it, then return the result.",
        "1. Read the file\n2. Summarize it",
        "Build a workflow for invoices",
        "工作流：读取文档然后输出摘要",
    ] {
        let verdict = detector
            .detect_initial(query, &empty)
            .await
            .expect("detection succeeds");
        assert_eq!(verdict.intent, BuildIntent::Build, "{query}");
        assert_eq!(verdict.source, GenerationSource::Deterministic);
        assert_ne!(verdict.detail, "");
    }

    for query in ["do something", "help me", "文档"] {
        let verdict = detector
            .detect_initial(query, &empty)
            .await
            .expect("detection succeeds");
        assert_eq!(verdict.intent, BuildIntent::Clarify, "{query}");
    }
}

#[tokio::test]
async fn retained_user_turns_count_towards_the_process_rule() {
    let detector = IntentionDetector::new();
    let history = history_with("1. read the file\n2. write the report");
    let verdict = detector
        .detect_initial("ok", &history)
        .await
        .expect("detection succeeds");
    assert_eq!(verdict.intent, BuildIntent::Build);
    assert!(verdict.detail.contains("numbers 2 steps"));
}

#[tokio::test]
async fn deterministic_refine_rule_needs_a_diagram_and_a_change_request() {
    let detector = IntentionDetector::new();
    let empty = HistoryManager::new(10);
    let refined = detector
        .detect_refine(
            "modify the workflow to add a validation step",
            &empty,
            "graph TD",
        )
        .await
        .expect("detection succeeds");
    assert_eq!(refined.intent, BuildIntent::Refine);
    assert!(refined.detail.contains("modify"));

    let cjk = detector
        .detect_refine("增加一个步骤", &empty, "graph TD")
        .await
        .expect("detection succeeds");
    assert_eq!(cjk.intent, BuildIntent::Refine);

    let no_diagram = detector
        .detect_refine("modify it", &empty, "   ")
        .await
        .expect("detection succeeds");
    assert_eq!(no_diagram.intent, BuildIntent::Build);

    let confirmation = detector
        .detect_refine("looks good", &empty, "graph TD")
        .await
        .expect("detection succeeds");
    assert_eq!(confirmation.intent, BuildIntent::Build);
}

#[tokio::test]
async fn blank_query_is_rejected_before_any_model_call() {
    let model = Arc::new(ScriptedModel::new([]));
    let detector = IntentionDetector::with_model(dynamic(&model));
    let empty = HistoryManager::new(10);

    for query in ["", "   ", "\n\t"] {
        let initial = detector
            .detect_initial(query, &empty)
            .await
            .expect_err("a blank query is invalid");
        assert_eq!(initial.kind(), ErrorKind::Validation);
        assert!(initial.to_string().contains("query must not be blank"));
        let refine = detector
            .detect_refine(query, &empty, "graph TD")
            .await
            .expect_err("a blank query is invalid");
        assert_eq!(refine.kind(), ErrorKind::Validation);
    }
    assert_eq!(model.call_count(), 0);
}

#[tokio::test]
async fn model_initial_answer_overrides_the_rule_and_records_the_request() {
    let model = Arc::new(ScriptedModel::text(
        "```json\n{\"provide_process\": false}\n```",
    ));
    let detector = IntentionDetector::with_model(dynamic(&model));
    let history = history_with("earlier turn");

    let verdict = detector
        .detect_initial("build a workflow", &history)
        .await
        .expect("detection succeeds");

    assert_eq!(verdict.intent, BuildIntent::Clarify);
    assert_eq!(verdict.source, GenerationSource::Model);
    assert!(verdict.detail.contains("provide_process=false"));

    assert_eq!(model.call_count(), 1);
    let call = model.last_call().expect("one call");
    assert_eq!(call.messages.len(), 2);
    assert_eq!(call.messages[0].text(), INITIAL_SYSTEM_PROMPT);
    let user = call.messages[1].text().into_owned();
    assert!(user.contains("earlier turn"), "{user}");
    assert!(user.contains("build a workflow"), "{user}");
    assert!(!user.contains("Mermaid"), "{user}");
}

#[tokio::test]
async fn model_refine_answer_receives_the_diagram() {
    let model = Arc::new(ScriptedModel::text(
        "```json\n{\"need_refined\": true}\n```",
    ));
    let detector = IntentionDetector::with_model(dynamic(&model));
    let empty = HistoryManager::new(10);

    let verdict = detector
        .detect_refine("looks good", &empty, "graph TD\n  start --> end")
        .await
        .expect("detection succeeds");

    assert_eq!(verdict.intent, BuildIntent::Refine);
    assert_eq!(verdict.source, GenerationSource::Model);
    assert!(verdict.detail.contains("need_refined=true"));

    let call = model.last_call().expect("one call");
    assert_eq!(call.messages[0].text(), REFINE_SYSTEM_PROMPT);
    let user = call.messages[1].text().into_owned();
    assert!(user.contains("graph TD\n  start --> end"), "{user}");
    assert!(user.contains("Current Mermaid diagram"), "{user}");
}

#[tokio::test]
async fn unusable_model_answers_fall_back_to_the_deterministic_rule() {
    for turn in [
        ScriptedTurn::text("I cannot help with that."),
        ScriptedTurn::text("```json\n{\"need_refined\": true}\n```"),
        ScriptedTurn::fail("boom"),
    ] {
        let model = Arc::new(ScriptedModel::new([turn]));
        let detector = IntentionDetector::with_model(dynamic(&model));
        let empty = HistoryManager::new(10);
        let verdict = detector
            .detect_initial("build a workflow", &empty)
            .await
            .expect("a model failure is absorbed");
        assert_eq!(verdict.intent, BuildIntent::Build);
        assert_eq!(verdict.source, GenerationSource::Fallback);
        assert!(verdict.detail.contains("deterministic rule"));
        assert_eq!(model.call_count(), 1);
    }
}

#[test]
fn intent_and_verdict_round_trip_through_their_wire_names() {
    for (intent, wire_name) in [
        (BuildIntent::Build, "build"),
        (BuildIntent::Refine, "refine"),
        (BuildIntent::Clarify, "clarify"),
    ] {
        assert_eq!(intent.as_str(), wire_name);
        assert_eq!(intent.to_string(), wire_name);
        assert_eq!(
            serde_json::to_value(intent).expect("serialize"),
            json!(wire_name)
        );
    }

    let verdict = IntentionVerdict {
        intent: BuildIntent::Refine,
        source: GenerationSource::Fallback,
        detail: "no diagram".to_string(),
    };
    let value = serde_json::to_value(&verdict).expect("serialize");
    assert_eq!(
        value,
        json!({"intent": "refine", "source": "fallback", "detail": "no diagram"})
    );
    let restored: IntentionVerdict = serde_json::from_value(value).expect("deserialize");
    assert_eq!(restored, verdict);

    let empty: IntentionVerdict = serde_json::from_value(json!({})).expect("defaults");
    assert_eq!(empty.intent, BuildIntent::Build);
    assert_eq!(empty.detail, "");
}
