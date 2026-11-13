#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(
    unreachable_pub,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::new_without_default,
    clippy::module_name_repetitions,
    clippy::unnecessary_literal_bound
)]
//! The `questioner` (human-in-the-loop) component inside a real workflow.
//!
//! Ports of `tests/unit_tests/core/component/test_questioner_comp.py`, rewritten against
//! the Rust public API: `start → questioner → end`, a checkpointed session, and
//! `Workflow::resume` for every answered question. Everything is offline and
//! deterministic (`ScriptedModel` answers, `MemoryCheckpointer`, no sleeps).
mod common;

use std::sync::Arc;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_session::{Checkpointer, InteractiveInput, MemoryCheckpointer, Session};
use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};
use serde_json::{json, Value};

use openjiuwen_workflow::components::questioner::{
    FieldInfo, QuestionerComponent, QuestionerConfig,
};
use openjiuwen_workflow::{
    ComponentOptions, EndComponent, StartComponent, Workflow, WorkflowExecutionState,
    WorkflowOutput, MESSAGES_STATE_KEY,
};

/// The questioner node id, which is also the interaction id.
const NODE_ID: &str = "questioner";

/// A required string field.
fn field(name: &str, description: &str) -> FieldInfo {
    FieldInfo {
        field_name: name.to_owned(),
        description: description.to_owned(),
        required: true,
        ..FieldInfo::default()
    }
}

/// A required field with an explicit JSON type.
fn typed_field(name: &str, description: &str, field_type: &str) -> FieldInfo {
    FieldInfo {
        field_type: field_type.to_owned(),
        ..field(name, description)
    }
}

/// A checkpointed session through `checkpointer`.
fn checkpointed(id: &str, checkpointer: &MemoryCheckpointer) -> Arc<dyn Session> {
    common::session_with_checkpointer(id, Arc::new(checkpointer.clone()) as Arc<dyn Checkpointer>)
}

/// Seeds the session's `__messages__` conversation lane with `messages`.
async fn seed_history(session: &Arc<dyn Session>, messages: Value) {
    let mut patch = serde_json::Map::new();
    patch.insert(MESSAGES_STATE_KEY.to_owned(), messages);
    session.update_state(Value::Object(patch)).await.unwrap();
}

/// The conversation lane of `session`, or `None` when nothing was recorded.
async fn recorded_history(session: &Arc<dyn Session>) -> Option<Value> {
    session.get_state(MESSAGES_STATE_KEY).await.unwrap()
}

/// The last extraction prompt sent to the model.
///
/// The `dialogue_history` section sits between the template's `对话历史` heading and the
/// closing instruction, so the assertions look for distinctive turn texts inside the
/// whole prompt.
fn last_prompt(model: &ScriptedModel) -> String {
    model
        .last_call()
        .expect("one extraction call")
        .last_user_text()
        .expect("the extraction prompt has a user message")
}

/// An `InteractiveInput` carrying one answer for the questioner node.
fn answer(text: &str) -> InteractiveInput {
    let mut interactive = InteractiveInput::new();
    interactive.update(NODE_ID, json!(text));
    interactive
}

/// `start → questioner → end`; the caller supplies both input templates.
fn questioner_workflow(
    id: &str,
    config: QuestionerConfig,
    questioner_inputs: Value,
    end_inputs: Value,
) -> Workflow {
    let component = QuestionerComponent::new(config).expect("the questioner config is valid");
    let mut workflow = Workflow::new(common::card(id));
    workflow
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp(
            NODE_ID,
            component,
            ComponentOptions::new().with_inputs_schema(questioner_inputs),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(end_inputs),
        )
        .add_connection("start", NODE_ID)
        .add_connection(NODE_ID, "end");
    workflow
}

/// The question carried by the first `__interaction__` wire of an interrupted run.
fn first_question(output: &WorkflowOutput) -> String {
    output.result[0]["payload"]["question"]
        .as_str()
        .expect("the interaction carries a question string")
        .to_owned()
}

// port of tests/unit_tests/core/component/test_questioner_comp.py::test_invoke_questioner_component_in_workflow_initial_ask
#[tokio::test]
async fn initial_ask_halts_with_the_continue_ask_question() {
    let checkpointer = MemoryCheckpointer::new();
    let session = checkpointed("questioner-initial-ask", &checkpointer);
    let config = QuestionerConfig::default()
        .with_model(Arc::new(ScriptedModel::text("{}")))
        .with_field_names(vec![field("location", "地点")]);
    let workflow = questioner_workflow(
        "questioner-initial-ask",
        config,
        json!({"query": "${start.query}"}),
        json!({"location": "${questioner.location}"}),
    );

    let first = workflow
        .invoke(json!({"query": "查询天气"}), Arc::clone(&session))
        .await
        .unwrap();

    assert_eq!(first.state, WorkflowExecutionState::InputRequired);
    assert_eq!(first.result[0]["type"], json!("__interaction__"));
    assert_eq!(first.result[0]["payload"]["id"], json!(NODE_ID));
    assert_eq!(first_question(&first), "请您提供地点相关的信息");
    assert_eq!(
        checkpointer.len(session.id()),
        1,
        "an interrupted run keeps its snapshot for the resume"
    );
}

// port of tests/unit_tests/core/component/test_questioner_comp.py::test_invoke_questioner_component_in_workflow_repeat_ask
#[tokio::test]
async fn resuming_with_an_answer_completes_with_the_user_response_and_fields() {
    let checkpointer = MemoryCheckpointer::new();
    let session = checkpointed("questioner-resume", &checkpointer);
    let model = ScriptedModel::new([
        ScriptedTurn::text("{}"),
        ScriptedTurn::text(r#"{"location": "hangzhou", "time": "tomorrow"}"#),
    ]);
    let config = QuestionerConfig::default()
        .with_model(Arc::new(model.clone()))
        .with_field_names(vec![field("location", "地点"), field("time", "时间")]);
    let workflow = questioner_workflow(
        "questioner-resume",
        config,
        json!({"query": "${start.query}"}),
        json!({
            "user_response": "${questioner.user_response}",
            "location": "${questioner.location}",
            "time": "${questioner.time}",
        }),
    );

    let first = workflow
        .invoke(json!({"query": "查询天气"}), Arc::clone(&session))
        .await
        .unwrap();
    assert_eq!(first.state, WorkflowExecutionState::InputRequired);

    let resumed = workflow
        .resume(answer("地点是杭州"), Arc::clone(&session))
        .await
        .unwrap();

    assert_eq!(resumed.state, WorkflowExecutionState::Completed);
    assert_eq!(
        resumed.result,
        json!({
            "output": {
                "user_response": "地点是杭州",
                "location": "hangzhou",
                "time": "tomorrow",
            }
        })
    );
    assert_eq!(model.call_count(), 2, "one extraction per round");
    assert_eq!(checkpointer.len(session.id()), 0);
}

// port of tests/unit_tests/core/component/test_questioner_comp.py::test_questioner_ignores_fields_outside_field_names_when_resuming
// and ...::test_questioner_type_conversion_with_multiple_types
#[tokio::test]
async fn extraction_keeps_configured_fields_converts_types_and_fills_defaults() {
    let checkpointer = MemoryCheckpointer::new();
    let session = checkpointed("questioner-types", &checkpointer);
    let model = ScriptedModel::text(
        r#"```json
{"location": "hangzhou", "age": "25", "score": "98.5", "is_vip": "true",
 "nickname": null, "message": "请提供昵称"}
```"#,
    );
    let config = QuestionerConfig::default()
        .with_model(Arc::new(model.clone()))
        .with_field_names(vec![
            typed_field("location", "地点", "string"),
            typed_field("age", "年龄", "integer"),
            typed_field("score", "分数", "number"),
            typed_field("is_vip", "会员", "boolean"),
            FieldInfo {
                field_name: "nickname".to_owned(),
                description: "昵称".to_owned(),
                default_value: json!("anon"),
                ..FieldInfo::default()
            },
        ]);
    let workflow = questioner_workflow(
        "questioner-types",
        config,
        json!({"query": "${start.query}"}),
        json!({"all": "${questioner}"}),
    );

    let output = workflow
        .invoke(json!({"query": "我的信息"}), Arc::clone(&session))
        .await
        .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::Completed);
    let all = &output.result["output"]["all"];
    assert_eq!(all["location"], json!("hangzhou"));
    assert_eq!(all["age"], json!(25), "\"25\" converts to an integer");
    assert_eq!(all["score"], json!(98.5), "\"98.5\" converts to a number");
    assert_eq!(all["is_vip"], json!(true), "\"true\" converts to a boolean");
    assert_eq!(
        all["nickname"],
        json!("anon"),
        "the default fills a field the model did not extract"
    );
    assert!(
        all.get("message").is_none(),
        "fields outside field_names are ignored"
    );
    assert!(
        all.get("user_response").is_none(),
        "an empty user_response is omitted"
    );

    let call = model.last_call().expect("one extraction call");
    assert_eq!(call.messages.len(), 2, "system + user extraction prompt");
    assert!(
        call.messages[0].text().contains("location: 地点"),
        "the prompt lists the configured parameters"
    );
}

// port of tests/unit_tests/core/component/test_questioner_comp.py::test_questioner_type_validation_failure_triggers_continue_ask
// plus Python's `_check_if_continue_ask` max_response branch
#[tokio::test]
async fn a_missing_required_field_asks_again_and_reports_max_response() {
    let checkpointer = MemoryCheckpointer::new();
    let session = checkpointed("questioner-max-response", &checkpointer);
    let response = r#"{"location": "hangzhou"}"#;
    let model = ScriptedModel::new([ScriptedTurn::text(response), ScriptedTurn::text(response)]);
    let config = QuestionerConfig::default()
        .with_model(Arc::new(model))
        .with_max_response(1)
        .with_field_names(vec![field("location", "地点"), field("date", "日期")]);
    let workflow = questioner_workflow(
        "questioner-max-response",
        config,
        json!({"query": "${start.query}"}),
        json!({"location": "${questioner.location}", "date": "${questioner.date}"}),
    );

    let first = workflow
        .invoke(json!({"query": "查询天气"}), Arc::clone(&session))
        .await
        .unwrap();
    assert_eq!(first.state, WorkflowExecutionState::InputRequired);
    assert!(
        first_question(&first).contains("日期"),
        "the continue-ask names the field that is still missing"
    );

    let failure = workflow
        .resume(answer("日期是明天"), Arc::clone(&session))
        .await
        .unwrap_err();

    assert_eq!(failure.kind(), ErrorKind::Workflow);
    assert!(failure.message().contains("[101074]"), "{failure}");
    assert!(
        failure
            .message()
            .contains("max_response reached before all required fields were extracted"),
        "{failure}"
    );
}

// port of tests/unit_tests/core/component/test_questioner_comp.py::test_invoke_questioner_component_in_workflow_repeat_ask
// (the `question_content` branch) and ...::test_questioner_accept_language_zh's topology
#[tokio::test]
async fn question_content_mode_renders_the_question_and_completes() {
    let checkpointer = MemoryCheckpointer::new();
    let session = checkpointed("questioner-content", &checkpointer);
    let config = QuestionerConfig::default()
        .with_question_content("请问{{name}}的{{topic}}是什么？")
        .with_extract_fields_from_response(false);
    let workflow = questioner_workflow(
        "questioner-content",
        config,
        json!({"query": "${start.query}", "name": "${start.name}"}),
        json!({
            "question": "${questioner.question}",
            "user_response": "${questioner.user_response}",
        }),
    );

    let first = workflow
        .invoke(
            json!({"query": "你好", "name": "小王"}),
            Arc::clone(&session),
        )
        .await
        .unwrap();
    assert_eq!(first.state, WorkflowExecutionState::InputRequired);
    assert_eq!(
        first_question(&first),
        "请问小王的None是什么？",
        "the template renders the inputs and an unresolved placeholder as None"
    );

    let resumed = workflow
        .resume(answer("天气"), Arc::clone(&session))
        .await
        .unwrap();
    assert_eq!(resumed.state, WorkflowExecutionState::Completed);
    assert_eq!(
        resumed.result,
        json!({
            "output": {
                "question": "请问小王的None是什么？",
                "user_response": "天气",
            }
        })
    );
}

// port of tests/unit_tests/core/component/test_questioner_comp.py::test_questioner_format_continue_ask_accept_language
#[tokio::test]
async fn continue_ask_uses_the_configured_language() {
    assert_eq!(second_question("zh").await, "请您提供地点相关的信息");
    assert_eq!(
        second_question("en").await,
        "Please provide information related to: Location"
    );
}

/// Runs two rounds whose extraction answers never satisfy the field, and returns the
/// second question of `accept_language`.
async fn second_question(accept_language: &str) -> String {
    let id = format!("questioner-language-{accept_language}");
    let checkpointer = MemoryCheckpointer::new();
    let session = checkpointed(&id, &checkpointer);
    let mut location = field("location", "地点");
    if accept_language == "en" {
        "Location".clone_into(&mut location.description);
    }
    let model = ScriptedModel::new([ScriptedTurn::text("{}"), ScriptedTurn::text("{}")]);
    let config = QuestionerConfig::default()
        .with_model(Arc::new(model))
        .with_field_names(vec![location])
        .with_accept_language(accept_language);
    let workflow = questioner_workflow(
        &id,
        config,
        json!({"query": "${start.query}"}),
        json!({"location": "${questioner.location}"}),
    );

    let first = workflow
        .invoke(json!({"query": "查询天气"}), Arc::clone(&session))
        .await
        .unwrap();
    assert_eq!(first.state, WorkflowExecutionState::InputRequired);

    let resumed = workflow
        .resume(answer("北京"), Arc::clone(&session))
        .await
        .unwrap();
    assert_eq!(resumed.state, WorkflowExecutionState::InputRequired);
    first_question(&resumed)
}

// port of tests/unit_tests/core/component/test_questioner_comp.py::test_questioner_state_reset_on_second_workflow_invocation
// (the configuration half: nothing to ask and nothing to extract)
#[test]
fn question_content_and_extraction_cannot_both_be_empty() {
    let config = QuestionerConfig::default().with_extract_fields_from_response(false);

    let error = QuestionerComponent::new(config).unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Workflow);
    assert!(error.message().starts_with("[101072]"), "{error}");
    assert!(
        error
            .message()
            .contains("question_content is empty and no extractable fields are configured"),
        "{error}"
    );
}

// port of tests/unit_tests/core/component/test_questioner_comp.py::test_questioner_state_reset_on_second_workflow_invocation
#[tokio::test]
async fn a_completed_run_starts_a_fresh_state_on_the_next_invoke() {
    let checkpointer = MemoryCheckpointer::new();
    let session = checkpointed("questioner-reset", &checkpointer);
    let config = QuestionerConfig::default()
        .with_question_content("请问{{name}}的{{topic}}是什么？")
        .with_extract_fields_from_response(false);
    let workflow = questioner_workflow(
        "questioner-reset",
        config,
        json!({"query": "${start.query}", "name": "${start.name}"}),
        json!({"user_response": "${questioner.user_response}"}),
    );

    let first = workflow
        .invoke(
            json!({"query": "你好", "name": "小王"}),
            Arc::clone(&session),
        )
        .await
        .unwrap();
    assert_eq!(first.state, WorkflowExecutionState::InputRequired);
    let resumed = workflow
        .resume(answer("天气"), Arc::clone(&session))
        .await
        .unwrap();
    assert_eq!(resumed.state, WorkflowExecutionState::Completed);

    let again = workflow
        .invoke(
            json!({"query": "你好", "name": "小李"}),
            Arc::clone(&session),
        )
        .await
        .unwrap();

    assert_eq!(
        again.state,
        WorkflowExecutionState::InputRequired,
        "the stored `end` state is not treated as an in-progress interaction"
    );
    assert_eq!(
        first_question(&again),
        "请问小李的None是什么？",
        "the question is rendered from the new inputs, not carried over"
    );
}

// port of Python's `QuestionerUtils.is_valid_value` + `validate_and_convert_type`
#[tokio::test]
async fn fractional_integers_and_ambiguous_booleans_are_dropped() {
    let checkpointer = MemoryCheckpointer::new();
    let session = checkpointed("questioner-conversions", &checkpointer);
    let response = r#"{"age": "25.5", "is_vip": "maybe", "score": 25}"#;
    let model = ScriptedModel::new([ScriptedTurn::text(response), ScriptedTurn::text(response)]);
    let config = QuestionerConfig::default()
        .with_model(Arc::new(model))
        .with_field_names(vec![
            typed_field("age", "年龄", "integer"),
            typed_field("is_vip", "会员", "boolean"),
            typed_field("score", "分数", "number"),
        ]);
    let workflow = questioner_workflow(
        "questioner-conversions",
        config,
        json!({"query": "${start.query}"}),
        json!({"score": "${questioner.score}"}),
    );

    let first = workflow
        .invoke(json!({"query": "我的信息"}), Arc::clone(&session))
        .await
        .unwrap();

    assert_eq!(first.state, WorkflowExecutionState::InputRequired);
    let question = first_question(&first);
    assert!(question.contains("年龄"), "{question}");
    assert!(question.contains("会员"), "{question}");
    assert!(
        !question.contains("分数"),
        "25 converts to a number, so that field is settled: {question}"
    );
}

// port of Python's `QuestionerUtils.parse_response` (non-object payload)
#[tokio::test]
async fn a_non_object_extraction_reports_a_parse_error() {
    let checkpointer = MemoryCheckpointer::new();
    let session = checkpointed("questioner-non-object", &checkpointer);
    let config = QuestionerConfig::default()
        .with_model(Arc::new(ScriptedModel::text("[1, 2, 3]")))
        .with_field_names(vec![field("location", "地点")]);
    let workflow = questioner_workflow(
        "questioner-non-object",
        config,
        json!({"query": "${start.query}"}),
        json!({"location": "${questioner.location}"}),
    );

    let failure = workflow
        .invoke(json!({"query": "查询天气"}), Arc::clone(&session))
        .await
        .unwrap_err();

    assert_eq!(failure.kind(), ErrorKind::Workflow);
    assert!(failure.message().contains("[101076]"), "{failure}");
    assert!(
        failure
            .message()
            .contains("failed to parse json from llm response"),
        "{failure}"
    );
}

// port of Python's `QuestionerUtils.parse_response` (unparseable payload yields no fields)
#[tokio::test]
async fn an_unparseable_extraction_yields_no_fields_and_asks_again() {
    let checkpointer = MemoryCheckpointer::new();
    let session = checkpointed("questioner-unparseable", &checkpointer);
    let config = QuestionerConfig::default()
        .with_model(Arc::new(ScriptedModel::text("抱歉，我无法回答。")))
        .with_field_names(vec![field("location", "地点")]);
    let workflow = questioner_workflow(
        "questioner-unparseable",
        config,
        json!({"query": "${start.query}"}),
        json!({"location": "${questioner.location}"}),
    );

    let first = workflow
        .invoke(json!({"query": "查询天气"}), Arc::clone(&session))
        .await
        .unwrap();

    assert_eq!(first.state, WorkflowExecutionState::InputRequired);
    assert_eq!(first_question(&first), "请您提供地点相关的信息");
}

// port of Python's `QuestionerExecutable._validate_config`
#[test]
fn configuration_errors_use_the_python_codes() {
    let cases = [
        (
            QuestionerConfig::default()
                .with_response_type("reply_later")
                .with_field_names(vec![field("a", "A")]),
            "response type reply_later is invalid",
        ),
        (
            QuestionerConfig::default(),
            "extracted key fields cannot be empty",
        ),
        (
            QuestionerConfig::default().with_field_names(vec![field("", "A")]),
            "extracted key field name cannot be empty",
        ),
        (
            QuestionerConfig::default()
                .with_field_names(vec![field("a", "A")])
                .with_max_response(0),
            "max response must be greater than 0",
        ),
    ];

    for (config, reason) in cases {
        let error = QuestionerComponent::new(config).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Workflow);
        assert!(error.message().starts_with("[101071]"), "{error}");
        assert!(error.message().contains(reason), "{error}");
    }
}

// port of Python's `QuestionerDirectReplyHandler._get_latest_chat_history` /
// `_create_prompt_template_keywords` (chat history injected into `dialogue_history`)
#[tokio::test]
async fn with_chat_history_the_prior_conversation_reaches_the_extraction_prompt() {
    let checkpointer = MemoryCheckpointer::new();
    let session = checkpointed("questioner-history-prompt", &checkpointer);
    seed_history(
        &session,
        json!([
            {"role": "user", "content": "上一轮我问了杭州的天气"},
            {"role": "assistant", "content": "上一轮我回答了晴转多云"},
        ]),
    )
    .await;
    let model = ScriptedModel::text(r#"{"location": "hangzhou"}"#);
    let config = QuestionerConfig::default()
        .with_model(Arc::new(model.clone()))
        .with_field_names(vec![field("location", "地点")])
        .with_chat_history(true);
    let workflow = questioner_workflow(
        "questioner-history-prompt",
        config,
        json!({"query": "${start.query}"}),
        json!({"location": "${questioner.location}"}),
    );

    let output = workflow
        .invoke(json!({"query": "那明天呢"}), Arc::clone(&session))
        .await
        .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::Completed);
    let prompt = last_prompt(&model);
    assert!(prompt.contains("user：上一轮我问了杭州的天气"), "{prompt}");
    assert!(
        prompt.contains("assistant：上一轮我回答了晴转多云"),
        "{prompt}"
    );
    assert!(
        prompt.contains("user：那明天呢"),
        "the current query closes the window: {prompt}"
    );
    assert!(
        prompt.find("上一轮我问了杭州的天气").unwrap() < prompt.find("user：那明天呢").unwrap(),
        "history precedes the current query: {prompt}"
    );
}

// port of Python's `_write_user_message_to_context` + `_write_assistant_message_to_context`
#[tokio::test]
async fn with_chat_history_a_completed_run_records_the_user_and_assistant_turns() {
    let checkpointer = MemoryCheckpointer::new();
    let session = checkpointed("questioner-history-record", &checkpointer);
    let model = ScriptedModel::new([
        ScriptedTurn::text("{}"),
        ScriptedTurn::text(r#"{"location": "hangzhou"}"#),
    ]);
    let config = QuestionerConfig::default()
        .with_model(Arc::new(model))
        .with_field_names(vec![field("location", "地点")])
        .with_chat_history(true);
    let workflow = questioner_workflow(
        "questioner-history-record",
        config,
        json!({"query": "${start.query}"}),
        json!({"location": "${questioner.location}"}),
    );

    let first = workflow
        .invoke(json!({"query": "查询天气"}), Arc::clone(&session))
        .await
        .unwrap();
    assert_eq!(first.state, WorkflowExecutionState::InputRequired);

    let resumed = workflow
        .resume(answer("地点是杭州"), Arc::clone(&session))
        .await
        .unwrap();
    assert_eq!(resumed.state, WorkflowExecutionState::Completed);

    assert_eq!(
        recorded_history(&session).await,
        Some(json!([
            {"role": "user", "content": "查询天气"},
            {"role": "assistant", "content": "请您提供地点相关的信息"},
            {"role": "user", "content": "地点是杭州"},
            {"role": "assistant", "content": "{\"location\":\"hangzhou\"}"},
        ]))
    );
}

// regression guard: the default configuration neither reads nor records history
#[tokio::test]
async fn without_chat_history_nothing_is_prompted_or_recorded() {
    let checkpointer = MemoryCheckpointer::new();
    let session = checkpointed("questioner-no-history", &checkpointer);
    let seeded = json!([
        {"role": "user", "content": "不该出现在提示词里的历史"},
        {"role": "assistant", "content": "也不该出现在提示词里的回答"},
    ]);
    seed_history(&session, seeded.clone()).await;
    let model = ScriptedModel::text(r#"{"location": "hangzhou"}"#);
    let config = QuestionerConfig::default()
        .with_model(Arc::new(model.clone()))
        .with_field_names(vec![field("location", "地点")]);
    let workflow = questioner_workflow(
        "questioner-no-history",
        config,
        json!({"query": "${start.query}"}),
        json!({"location": "${questioner.location}"}),
    );

    let output = workflow
        .invoke(json!({"query": "查询天气"}), Arc::clone(&session))
        .await
        .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::Completed);
    let prompt = last_prompt(&model);
    assert!(
        !prompt.contains("不该出现在提示词里的历史"),
        "the seeded lane must not reach the prompt: {prompt}"
    );
    assert!(
        prompt.contains("user：查询天气"),
        "only the current query is rendered: {prompt}"
    );
    assert_eq!(
        recorded_history(&session).await,
        Some(seeded),
        "a history-disabled component leaves the lane untouched"
    );
}

// port of Python's `get_context_window(dialogue_round=chat_history_max_rounds)`
#[tokio::test]
async fn chat_history_max_rounds_keeps_only_the_recent_rounds() {
    let checkpointer = MemoryCheckpointer::new();
    let session = checkpointed("questioner-history-trim", &checkpointer);
    seed_history(
        &session,
        json!([
            {"role": "user", "content": "第一轮问题"},
            {"role": "assistant", "content": "第一轮回答"},
            {"role": "user", "content": "第二轮问题"},
            {"role": "assistant", "content": "第二轮回答"},
            {"role": "user", "content": "第三轮问题"},
            {"role": "assistant", "content": "第三轮回答"},
        ]),
    )
    .await;
    let model = ScriptedModel::text(r#"{"location": "hangzhou"}"#);
    let config = QuestionerConfig::default()
        .with_model(Arc::new(model.clone()))
        .with_field_names(vec![field("location", "地点")])
        .with_chat_history(true)
        .with_chat_history_max_rounds(2);
    let workflow = questioner_workflow(
        "questioner-history-trim",
        config,
        json!({"query": "${start.query}"}),
        json!({"location": "${questioner.location}"}),
    );

    let output = workflow
        .invoke(json!({"query": "当前问题"}), Arc::clone(&session))
        .await
        .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::Completed);
    let prompt = last_prompt(&model);
    assert!(prompt.contains("user：第三轮问题"), "{prompt}");
    assert!(prompt.contains("assistant：第三轮回答"), "{prompt}");
    assert!(prompt.contains("user：当前问题"), "{prompt}");
    assert!(
        !prompt.contains("第二轮问题") && !prompt.contains("第一轮问题"),
        "rounds beyond the limit are trimmed: {prompt}"
    );
}

// port of Python's trailing-turn rule in `_get_latest_chat_history`
#[tokio::test]
async fn an_empty_query_still_closes_the_window_with_a_user_turn() {
    let checkpointer = MemoryCheckpointer::new();
    let session = checkpointed("questioner-history-trailing", &checkpointer);
    seed_history(
        &session,
        json!([
            {"role": "user", "content": "先前的问题"},
            {"role": "assistant", "content": "先前的回答"},
        ]),
    )
    .await;
    let model = ScriptedModel::text("{}");
    let config = QuestionerConfig::default()
        .with_model(Arc::new(model.clone()))
        .with_field_names(vec![field("location", "地点")])
        .with_chat_history(true);
    let workflow = questioner_workflow(
        "questioner-history-trailing",
        config,
        json!({"query": "${start.query}"}),
        json!({"location": "${questioner.location}"}),
    );

    let first = workflow
        .invoke(json!({}), Arc::clone(&session))
        .await
        .unwrap();

    assert_eq!(first.state, WorkflowExecutionState::InputRequired);
    let prompt = last_prompt(&model);
    assert!(
        prompt.contains("assistant：先前的回答\nuser："),
        "a window ending with an assistant turn gains a user turn: {prompt}"
    );
    assert_eq!(
        recorded_history(&session).await,
        Some(json!([
            {"role": "user", "content": "先前的问题"},
            {"role": "assistant", "content": "先前的回答"},
            {"role": "assistant", "content": "请您提供地点相关的信息"},
        ])),
        "a blank query records no user turn, only the asked question"
    );
}

// a structured (non-string) query degrades to its plain JSON text
#[tokio::test]
async fn a_structured_query_is_recorded_as_json_text() {
    let checkpointer = MemoryCheckpointer::new();
    let session = checkpointed("questioner-history-structured", &checkpointer);
    let model = ScriptedModel::text(r#"{"location": "hangzhou"}"#);
    let config = QuestionerConfig::default()
        .with_model(Arc::new(model.clone()))
        .with_field_names(vec![field("location", "地点")])
        .with_chat_history(true);
    let workflow = questioner_workflow(
        "questioner-history-structured",
        config,
        json!({"query": "${start.query}"}),
        json!({"location": "${questioner.location}"}),
    );

    let output = workflow
        .invoke(json!({"query": {"city": "杭州"}}), Arc::clone(&session))
        .await
        .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert!(
        last_prompt(&model).contains(r#"user：{"city":"杭州"}"#),
        "the object is rendered as JSON text"
    );
    assert_eq!(
        recorded_history(&session).await,
        Some(json!([
            {"role": "user", "content": r#"{"city":"杭州"}"#},
            {"role": "assistant", "content": "{\"location\":\"hangzhou\"}"},
        ]))
    );
}
