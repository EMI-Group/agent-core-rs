#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(
    dead_code,
    unreachable_pub,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::new_without_default,
    clippy::module_name_repetitions,
    clippy::unnecessary_literal_bound,
    clippy::assert_is_empty
)]
//! LLM component behavior through the public workflow API.
//!
//! These tests pin prompt assembly (`{{var}}` placeholders rendered from the node's
//! resolved inputs), the output key taken from `output_config`, streaming deltas written
//! as buffered frames, the `${start.query}` reference feeding the user prompt, and two
//! LLM nodes composing in one workflow.
//!
//! Ports of `tests/unit_tests/core/component/test_llm_comp.py`, rewritten against
//! `Workflow`/`ComponentOptions`. Every test is offline and deterministic: the only model
//! is `ScriptedModel`, which records every call for assertions and never sleeps.
mod common;

use std::sync::Arc;

use futures::StreamExt;
use openjiuwen_common::error::Error;
use openjiuwen_foundation::{Model, Role, StreamChunk};
use openjiuwen_session::Session;
use openjiuwen_test_support::{CallMode, ModelCall, ScriptedModel};
use serde_json::{json, Value};

use openjiuwen_workflow::{
    ComponentOptions, EndComponent, LlmCompConfig, LlmComponent, LlmResponseFormat, StartComponent,
    Workflow, WorkflowChunk, WorkflowExecutionState, WorkflowOutput, END_NODE_STREAM,
    WORKFLOW_FINAL,
};

use common::{assistant_text, card, end_with_template, llm_component, session};

/// One `ScriptedModel` plus the type-erased handle the components receive.
fn scripted(turns: ScriptedModel) -> (ScriptedModel, Arc<dyn Model>) {
    let model: Arc<dyn Model> = Arc::new(turns.clone());
    (turns, model)
}

/// The text of every message in `call`.
fn message_texts(call: &ModelCall) -> Vec<String> {
    call.messages
        .iter()
        .map(|message| message.text().into_owned())
        .collect()
}

/// Collects every frame of a run, panicking on the first failure.
async fn run_stream(
    workflow: &Workflow,
    inputs: Value,
    session: Arc<dyn Session>,
) -> Vec<WorkflowChunk> {
    workflow
        .stream(inputs, session)
        .map(|frame| frame.unwrap())
        .collect()
        .await
}

// port of tests/unit_tests/core/component/test_llm_comp.py::test_prepare_model_inputs_system_and_user_keeps_both
#[tokio::test]
async fn system_and_user_templates_are_assembled_from_the_resolved_inputs() {
    let (turns, model) = scripted(ScriptedModel::builder().respond_text("answer").build());
    let llm = LlmComponent::new(LlmCompConfig {
        model: Some(Arc::clone(&model)),
        system_prompt_template: Some("system {{name}}".to_owned()),
        user_prompt_template: Some("hello {{name}}, inputs={{inputs}}".to_owned()),
        ..LlmCompConfig::default()
    })
    .unwrap();

    let mut workflow = Workflow::new(card("llm-prompts"));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({
                "name": "${inputs.name}",
                "extra": "${inputs.extra}",
            })),
        )
        .add_workflow_comp(
            "llm",
            llm,
            ComponentOptions::new().with_inputs_schema(json!({
                "name": "${start.name}",
                "extra": "${start.extra}",
            })),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!("${llm}")),
        )
        .add_connection("start", "llm")
        .add_connection("llm", "end");

    workflow
        .invoke(
            json!({"name": "bob", "extra": {"deep": 1}}),
            session("llm-prompts"),
        )
        .await
        .unwrap();

    assert_eq!(turns.call_count(), 1, "the model is called exactly once");
    let call = turns.last_call().unwrap();
    assert_eq!(call.mode, CallMode::Invoke);
    assert_eq!(
        call.messages
            .iter()
            .map(|message| message.role.clone())
            .collect::<Vec<_>>(),
        vec![Role::System, Role::User]
    );
    let texts = message_texts(&call);
    assert_eq!(texts[0], "system bob");
    // `inputs` renders the node's whole resolved input object, including the nested
    // `extra` object, exactly as the component's documented render map describes.
    assert_eq!(
        texts[1],
        format!(
            "hello bob, inputs={}",
            json!({"name": "bob", "extra": {"deep": 1}})
        )
    );
}

// port of tests/unit_tests/core/component/test_llm_comp.py::test_invoke_success
#[tokio::test]
async fn output_config_names_the_answer_field() {
    let (_turns, model) = scripted(ScriptedModel::builder().respond_text("the answer").build());
    let llm = LlmComponent::new(LlmCompConfig {
        model: Some(model),
        user_prompt_template: Some("{{query}}".to_owned()),
        output_config: Some(json!({"answer": {"type": "string"}})),
        ..LlmCompConfig::default()
    })
    .unwrap();

    let mut workflow = Workflow::new(card("llm-output-config"));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"query": "${inputs.query}"})),
        )
        .add_workflow_comp(
            "llm",
            llm,
            ComponentOptions::new().with_inputs_schema(json!({"query": "${start.query}"})),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"answer": "${llm.answer}"})),
        )
        .add_connection("start", "llm")
        .add_connection("llm", "end");

    let output = workflow
        .invoke(json!({"query": "hi"}), session("llm-output-config"))
        .await
        .unwrap();

    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert_eq!(output.result, json!({"output": {"answer": "the answer"}}));
}

// port of tests/unit_tests/core/component/test_llm_comp.py::test_invoke_success
#[tokio::test]
async fn the_default_output_key_is_result() {
    let (_turns, model) = scripted(ScriptedModel::builder().respond_text("plain").build());
    let llm = llm_component(model, "{{query}}");

    let mut workflow = Workflow::new(card("llm-default-key"));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"query": "${inputs.query}"})),
        )
        .add_workflow_comp(
            "llm",
            llm,
            ComponentOptions::new().with_inputs_schema(json!({"query": "${start.query}"})),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"answer": "${llm.result}"})),
        )
        .add_connection("start", "llm")
        .add_connection("llm", "end");

    let output = workflow
        .invoke(json!({"query": "hi"}), session("llm-default-key"))
        .await
        .unwrap();

    assert_eq!(output.result, json!({"output": {"answer": "plain"}}));
}

// port of tests/unit_tests/core/component/test_llm_comp.py::test_real_workflow_stream_start_llm_end_with_component_streaming
#[tokio::test]
async fn a_streaming_llm_yields_one_buffered_frame_per_delta() {
    let (turns, model) = scripted(
        ScriptedModel::builder()
            .chunks(vec![
                StreamChunk::Delta {
                    index: 0,
                    role: Some(Role::Assistant),
                    content: Some("he".to_owned()),
                    reasoning: None,
                    tool_call: None,
                    finish_reason: None,
                },
                StreamChunk::Delta {
                    index: 0,
                    role: None,
                    content: Some("llo".to_owned()),
                    reasoning: None,
                    tool_call: None,
                    finish_reason: None,
                },
                StreamChunk::Done,
            ])
            .build(),
    );
    let llm = LlmComponent::new(LlmCompConfig {
        model: Some(Arc::clone(&model)),
        user_prompt_template: Some("{{query}}".to_owned()),
        output_config: Some(json!({"answer": {}})),
        ..LlmCompConfig::default()
    })
    .unwrap();

    let mut workflow = Workflow::new(card("llm-stream"));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"query": "${inputs.query}"})),
        )
        .add_workflow_comp(
            "llm",
            llm,
            ComponentOptions::new()
                .with_inputs_schema(json!({"query": "${start.query}"}))
                .streaming(),
        )
        .set_end_comp(
            "end",
            end_with_template("Answer: {{llm.answer}}"),
            ComponentOptions::new()
                .with_inputs_schema(json!({"llm": "${llm}"}))
                .streaming(),
        )
        .add_connection("start", "llm")
        .add_connection("llm", "end");

    let chunks = run_stream(&workflow, json!({"query": "hi"}), session("llm-stream")).await;

    let final_chunk = chunks.last().unwrap();
    assert_eq!(final_chunk.item_type, WORKFLOW_FINAL);

    let frames = &chunks[..chunks.len() - 1];
    // One template literal frame, then one frame per delta in provider order; the
    // deltas only resolve because the LLM wrote them under the `answer` output key.
    let payloads: Vec<Value> = frames.iter().map(|chunk| chunk.payload.clone()).collect();
    assert_eq!(
        payloads,
        vec![
            json!({"response": "Answer: "}),
            json!({"response": "he"}),
            json!({"response": "llo"}),
        ]
    );
    for (index, chunk) in frames.iter().enumerate() {
        assert_eq!(chunk.item_type, END_NODE_STREAM);
        assert_eq!(chunk.index, Some(i64::try_from(index).unwrap()));
    }
    assert_eq!(assistant_text(&payloads), "Answer: hello");
    assert_eq!(
        final_chunk.payload,
        json!([
            {"type": END_NODE_STREAM, "index": 0, "payload": {"response": "Answer: "}},
            {"type": END_NODE_STREAM, "index": 1, "payload": {"response": "he"}},
            {"type": END_NODE_STREAM, "index": 2, "payload": {"response": "llo"}},
        ])
    );

    assert_eq!(turns.call_count(), 1);
    assert_eq!(turns.last_call().unwrap().mode, CallMode::Stream);
}

// port of tests/unit_tests/core/component/test_llm_comp.py::test_start_llm_end_in_workflow
#[tokio::test]
async fn the_user_prompt_sees_the_resolved_start_reference() {
    let (turns, model) = scripted(ScriptedModel::builder().respond_text("done").build());
    let llm = llm_component(model, "{{query}}");

    let mut workflow = Workflow::new(card("llm-start-ref"));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"query": "${inputs.query}"})),
        )
        .add_workflow_comp(
            "llm",
            llm,
            ComponentOptions::new().with_inputs_schema(json!({"query": "${start.query}"})),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!("${llm}")),
        )
        .add_connection("start", "llm")
        .add_connection("llm", "end");

    workflow
        .invoke(json!({"query": "hello"}), session("llm-start-ref"))
        .await
        .unwrap();

    assert_eq!(
        turns.last_call().unwrap().last_user_text(),
        Some("hello".to_owned())
    );
}

// port of tests/unit_tests/core/component/test_llm_comp.py::test_llm_in_workflow
#[tokio::test]
async fn two_llm_nodes_each_get_their_own_turn_and_compose() {
    let (turns, model) = scripted(
        ScriptedModel::builder()
            .respond_text("first answer")
            .respond_text("second answer")
            .build(),
    );

    let mut workflow = Workflow::new(card("llm-chain"));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"query": "${inputs.query}"})),
        )
        .add_workflow_comp(
            "llm1",
            llm_component(Arc::clone(&model), "{{query}}"),
            ComponentOptions::new().with_inputs_schema(json!({"query": "${start.query}"})),
        )
        .add_workflow_comp(
            "llm2",
            llm_component(Arc::clone(&model), "previous: {{previous}}"),
            ComponentOptions::new().with_inputs_schema(json!({"previous": "${llm1.result}"})),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!("${llm2}")),
        )
        .add_connection("start", "llm1")
        .add_connection("llm1", "llm2")
        .add_connection("llm2", "end");

    let output = workflow
        .invoke(json!({"query": "hello"}), session("llm-chain"))
        .await
        .unwrap();

    assert_eq!(turns.call_count(), 2);
    let calls = turns.calls();
    assert_eq!(calls[0].last_user_text(), Some("hello".to_owned()));
    assert_eq!(
        calls[1].last_user_text(),
        Some("previous: first answer".to_owned()),
        "the second node must see the first node's answer"
    );
    assert_eq!(
        output.result,
        json!({"output": {"result": "second answer"}})
    );
}

/// Runs `start -> llm -> end`, with the end node reading `${llm.<answer_key>}`.
async fn invoke_llm(
    id: &str,
    llm: LlmComponent,
    answer_key: &str,
    query: &str,
) -> Result<WorkflowOutput, Error> {
    let mut workflow = Workflow::new(card(id));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"query": "${inputs.query}"})),
        )
        .add_workflow_comp(
            "llm",
            llm,
            ComponentOptions::new().with_inputs_schema(json!({"query": "${start.query}"})),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new()
                .with_inputs_schema(json!({"answer": format!("${{llm.{answer_key}}}")})),
        )
        .add_connection("start", "llm")
        .add_connection("llm", "end");

    workflow
        .invoke(json!({ "query": query }), session(id))
        .await
}

// port of openjiuwen/core/workflow/components/llm/llm_comp.py::LLMPromptFormatter.format_prompt
#[tokio::test]
async fn markdown_format_rewrites_the_prompt_and_keeps_the_answer_key() {
    let (turns, model) = scripted(ScriptedModel::builder().respond_text("**bold**").build());
    let llm = LlmComponent::new(LlmCompConfig {
        model: Some(Arc::clone(&model)),
        user_prompt_template: Some("{{query}}".to_owned()),
        response_format: LlmResponseFormat::Markdown,
        output_config: Some(json!({"answer": {}})),
        ..LlmCompConfig::default()
    })
    .unwrap();

    let output = invoke_llm("llm-markdown", llm, "answer", "hi")
        .await
        .unwrap();
    assert_eq!(output.result, json!({"output": {"answer": "**bold**"}}));

    let call = turns.last_call().unwrap();
    let prompt = call.last_user_text().unwrap();
    assert!(prompt.contains("Please return the answer in markdown format."));
    assert!(prompt.ends_with("The question is: hi."));
    assert_eq!(
        call.messages
            .iter()
            .filter(|message| message.role == Role::User)
            .count(),
        1,
        "only the last user message is rewritten"
    );
}

#[tokio::test]
async fn a_custom_markdown_instruction_replaces_the_default() {
    let (turns, model) = scripted(ScriptedModel::builder().respond_text("ok").build());
    let llm = LlmComponent::new(LlmCompConfig {
        model: Some(Arc::clone(&model)),
        user_prompt_template: Some("{{query}}".to_owned()),
        response_format: LlmResponseFormat::Markdown,
        markdown_instruction: Some("Answer in markdown. Question: ${query}".to_owned()),
        output_config: Some(json!({"answer": {}})),
        ..LlmCompConfig::default()
    })
    .unwrap();

    invoke_llm("llm-markdown-override", llm, "answer", "hi")
        .await
        .unwrap();

    let prompt = turns.last_call().unwrap().last_user_text().unwrap();
    assert!(
        prompt.starts_with("Answer in markdown. Question: hi"),
        "{prompt}"
    );
    assert!(
        !prompt.contains("Please return the answer in markdown format."),
        "{prompt}"
    );
}

// port of openjiuwen/core/workflow/components/llm/llm_comp.py::SchemaGenerator / OutputFormatter
#[tokio::test]
async fn json_format_embeds_the_schema_and_shapes_a_fenced_reply() {
    let (turns, model) = scripted(
        ScriptedModel::builder()
            .respond_text("```json\n{\"answer\": \"42\", \"extra\": 1}\n```")
            .build(),
    );
    let llm = LlmComponent::new(LlmCompConfig {
        model: Some(Arc::clone(&model)),
        user_prompt_template: Some("{{query}}".to_owned()),
        response_format: LlmResponseFormat::Json,
        output_config: Some(json!({"answer": {"type": "string"}})),
        ..LlmCompConfig::default()
    })
    .unwrap();

    let output = invoke_llm("llm-json", llm, "answer", "hi").await.unwrap();
    // The fenced block is stripped, the unconfigured `extra` key is dropped, and the
    // value lands under the configured output key.
    assert_eq!(output.result, json!({"output": {"answer": "42"}}));

    let prompt = turns.last_call().unwrap().last_user_text().unwrap();
    assert!(prompt.contains("Here is the JSON schema:"), "{prompt}");
    assert!(prompt.contains("\"answer\""), "{prompt}");
    assert!(prompt.ends_with("The question is: hi."), "{prompt}");
}

#[tokio::test]
async fn json_missing_required_field_is_an_execution_error() {
    let (_turns, model) = scripted(
        ScriptedModel::builder()
            .respond_text("{\"a\": \"x\"}")
            .build(),
    );
    let llm = LlmComponent::new(LlmCompConfig {
        model: Some(model),
        user_prompt_template: Some("{{query}}".to_owned()),
        response_format: LlmResponseFormat::Json,
        output_config: Some(json!({"a": {}, "b": {}})),
        ..LlmCompConfig::default()
    })
    .unwrap();

    let error = invoke_llm("llm-json-missing", llm, "a", "hi")
        .await
        .unwrap_err();
    assert!(error.message().contains("[101004]"), "{error}");
    assert!(
        error.message().contains("missing required fields: b"),
        "{error}"
    );
}

#[tokio::test]
async fn text_format_with_two_output_keys_is_a_config_error() {
    let (_turns, model) = scripted(ScriptedModel::builder().respond_text("x").build());
    let llm = LlmComponent::new(LlmCompConfig {
        model: Some(model),
        user_prompt_template: Some("{{query}}".to_owned()),
        output_config: Some(json!({"a": {}, "b": {}})),
        ..LlmCompConfig::default()
    })
    .unwrap();

    let error = invoke_llm("llm-text-two-keys", llm, "a", "hi")
        .await
        .unwrap_err();
    assert!(error.message().contains("[101002]"), "{error}");
}

#[tokio::test]
async fn text_format_leaves_the_rendered_prompt_untouched() {
    let (turns, model) = scripted(ScriptedModel::builder().respond_text("plain").build());
    let llm = LlmComponent::new(LlmCompConfig {
        model: Some(Arc::clone(&model)),
        user_prompt_template: Some("hello {{query}}".to_owned()),
        output_config: Some(json!({"answer": {}})),
        ..LlmCompConfig::default()
    })
    .unwrap();

    let output = invoke_llm("llm-text-untouched", llm, "answer", "hi")
        .await
        .unwrap();
    assert_eq!(output.result, json!({"output": {"answer": "plain"}}));
    assert_eq!(
        turns.last_call().unwrap().last_user_text(),
        Some("hello hi".to_owned()),
        "a text response leaves the rendered conversation byte-identical"
    );
}
