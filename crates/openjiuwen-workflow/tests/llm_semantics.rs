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
//! LLM component semantics through the public workflow API.
//!
//! Two halves, both offline and deterministic (`ScriptedModel` only):
//!
//! - **Conversation history** (`enable_history`), porting Python's
//!   `_insert_history_to_system_and_user_prompt`, `_write_user_message_to_context` and
//!   `_write_assistant_message_to_context`. A second run on the same session must see the
//!   first turn spliced between the system prompt and its own user prompt, the recorded
//!   user text is the one actually sent, the assistant entry is the raw reply, and a
//!   history-disabled component must neither read nor write the session lane.
//! - **Response formatting** (`response_format`, `output_config`), porting
//!   `LLMPromptFormatter.format_prompt` and `OutputFormatter.format_response`.
//!
//! This file is long on purpose: it is the single home of the LLM *semantics* suites, so
//! `tests/llm_component.rs` can stay the prompt-assembly and streaming suite.

mod common;

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use openjiuwen_common::error::Error;
use openjiuwen_foundation::{BaseMessage, Model, Role, StreamChunk};
use openjiuwen_session::Session;
use openjiuwen_test_support::{ModelCall, ScriptedModel};
use parking_lot::Mutex;
use serde_json::{json, Value};

use openjiuwen_workflow::{
    Component, ComponentContext, ComponentOptions, EndComponent, LlmCompConfig, LlmComponent,
    LlmResponseFormat, MessageHistory, StartComponent, Workflow, WorkflowExecutionState,
    WorkflowOutput, END_NODE_STREAM, MESSAGES_STATE_KEY, WORKFLOW_FINAL,
};

use common::{assistant_text, card, end_with_template, session};

/// One `ScriptedModel` plus the type-erased handle the components receive.
fn scripted(turns: ScriptedModel) -> (ScriptedModel, Arc<dyn Model>) {
    let model: Arc<dyn Model> = Arc::new(turns.clone());
    (turns, model)
}

/// The `(role, text)` pair of every message in `call`.
fn call_pairs(call: &ModelCall) -> Vec<(Role, String)> {
    message_pairs(&call.messages)
}

/// The `(role, text)` pair of every message.
fn message_pairs(messages: &[BaseMessage]) -> Vec<(Role, String)> {
    messages
        .iter()
        .map(|message| (message.role.clone(), message.text().into_owned()))
        .collect()
}

fn user(text: &str) -> (Role, String) {
    (Role::User, text.to_owned())
}

fn assistant(text: &str) -> (Role, String) {
    (Role::Assistant, text.to_owned())
}

fn system(text: &str) -> (Role, String) {
    (Role::System, text.to_owned())
}

/// Builds `start -> llm -> end`, feeding `inputs.query` to the LLM's `query` input.
///
/// The end node reads the whole LLM node, so the result always nests the LLM output.
fn single_llm_workflow(id: &str, llm: LlmComponent) -> Workflow {
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
            ComponentOptions::new().with_inputs_schema(json!("${llm}")),
        )
        .add_connection("start", "llm")
        .add_connection("llm", "end");
    workflow
}

/// Runs `workflow` twice on `session`, querying `first` then `second`.
async fn invoke_twice(workflow: &Workflow, session: Arc<dyn Session>, first: &str, second: &str) {
    for query in [first, second] {
        workflow
            .invoke(json!({ "query": query }), Arc::clone(&session))
            .await
            .unwrap();
    }
}

// port of openjiuwen/core/workflow/components/llm/llm_comp.py::LLMExecutable._get_model_input
#[tokio::test]
async fn a_second_run_sees_the_first_turn_before_its_own_prompt() {
    let (turns, model) = scripted(
        ScriptedModel::builder()
            .respond_text("first answer")
            .respond_text("second answer")
            .build(),
    );
    let llm = LlmComponent::new(LlmCompConfig {
        model: Some(model),
        user_prompt_template: Some("{{query}}".to_owned()),
        enable_history: true,
        ..LlmCompConfig::default()
    })
    .unwrap();
    let workflow = single_llm_workflow("llm-history", llm);
    let session = session("llm-history");

    invoke_twice(&workflow, Arc::clone(&session), "first", "second").await;

    let calls = turns.calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(
        call_pairs(&calls[0]),
        vec![user("first")],
        "the first run starts from an empty history"
    );
    assert_eq!(
        call_pairs(&calls[1]),
        vec![user("first"), assistant("first answer"), user("second"),],
        "the recorded turn sits between the past and the current prompt"
    );

    assert_eq!(
        session.get_state(MESSAGES_STATE_KEY).await.unwrap(),
        Some(json!([
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "second answer"}
        ])),
        "one user + one assistant entry per turn, in session order"
    );
}

// port of openjiuwen/core/workflow/components/llm/llm_comp.py::LLMExecutable._insert_history_to_system_and_user_prompt
#[tokio::test]
async fn history_is_spliced_after_the_system_prompt() {
    let (turns, model) = scripted(
        ScriptedModel::builder()
            .respond_text("first answer")
            .respond_text("second answer")
            .build(),
    );
    let llm = LlmComponent::new(LlmCompConfig {
        model: Some(model),
        system_prompt_template: Some("you are a bot".to_owned()),
        user_prompt_template: Some("{{query}}".to_owned()),
        enable_history: true,
        ..LlmCompConfig::default()
    })
    .unwrap();
    let workflow = single_llm_workflow("llm-history-system", llm);
    let session = session("llm-history-system");

    invoke_twice(&workflow, Arc::clone(&session), "first", "second").await;

    let calls = turns.calls();
    assert_eq!(
        call_pairs(&calls[1]),
        vec![
            system("you are a bot"),
            user("first"),
            assistant("first answer"),
            user("second"),
        ],
        "the leading system prefix stays first and the current prompt stays last"
    );
}

#[tokio::test]
async fn a_history_disabled_component_neither_reads_nor_writes() {
    let (turns, model) = scripted(
        ScriptedModel::builder()
            .respond_text("first answer")
            .respond_text("second answer")
            .build(),
    );
    let llm = LlmComponent::new(LlmCompConfig {
        model: Some(model),
        user_prompt_template: Some("{{query}}".to_owned()),
        ..LlmCompConfig::default()
    })
    .unwrap();
    let workflow = single_llm_workflow("llm-no-history", llm);
    let session = session("llm-no-history");

    invoke_twice(&workflow, Arc::clone(&session), "first", "second").await;

    let calls = turns.calls();
    assert_eq!(call_pairs(&calls[0]), vec![user("first")]);
    assert_eq!(
        call_pairs(&calls[1]),
        vec![user("second")],
        "without enable_history the conversation is exactly the rendered prompt"
    );
    assert_eq!(
        session.get_state(MESSAGES_STATE_KEY).await.unwrap(),
        None,
        "a disabled component must not create the history lane"
    );
}

#[tokio::test]
async fn a_streamed_run_records_the_full_answer_once() {
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
            .respond_text("second answer")
            .build(),
    );
    let session = session("llm-history-stream");

    // A streaming run: one buffered frame per delta and an `end node stream` template,
    // as in `tests/llm_component.rs`.
    let mut streaming = Workflow::new(card("llm-history-stream"));
    streaming
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"query": "${inputs.query}"})),
        )
        .add_workflow_comp(
            "llm",
            LlmComponent::new(LlmCompConfig {
                model: Some(Arc::clone(&model)),
                user_prompt_template: Some("{{query}}".to_owned()),
                enable_history: true,
                ..LlmCompConfig::default()
            })
            .unwrap(),
            ComponentOptions::new()
                .with_inputs_schema(json!({"query": "${start.query}"}))
                .streaming(),
        )
        .set_end_comp(
            "end",
            end_with_template("Answer: {{llm.result}}"),
            ComponentOptions::new()
                .with_inputs_schema(json!({"llm": "${llm}"}))
                .streaming(),
        )
        .add_connection("start", "llm")
        .add_connection("llm", "end");

    let chunks: Vec<_> = streaming
        .stream(json!({"query": "hi"}), Arc::clone(&session))
        .map(|chunk| chunk.unwrap())
        .collect()
        .await;
    let payloads: Vec<Value> = chunks[..chunks.len() - 1]
        .iter()
        .map(|chunk| chunk.payload.clone())
        .collect();
    assert_eq!(
        assistant_text(&payloads),
        "Answer: hello",
        "the deltas still stream through the end template"
    );
    assert_eq!(chunks.last().unwrap().item_type, WORKFLOW_FINAL);
    assert_eq!(chunks[..chunks.len() - 1][0].item_type, END_NODE_STREAM);

    assert_eq!(
        session.get_state(MESSAGES_STATE_KEY).await.unwrap(),
        Some(json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"}
        ])),
        "the accumulated answer is recorded once, not one entry per delta"
    );

    // A normal run on the same session sees exactly that one assistant entry.
    let follow_up = single_llm_workflow(
        "llm-history-stream-follow-up",
        LlmComponent::new(LlmCompConfig {
            model: Some(Arc::clone(&model)),
            user_prompt_template: Some("{{query}}".to_owned()),
            enable_history: true,
            ..LlmCompConfig::default()
        })
        .unwrap(),
    );
    follow_up
        .invoke(json!({"query": "second"}), Arc::clone(&session))
        .await
        .unwrap();

    let calls = turns.calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(
        call_pairs(&calls[1]),
        vec![user("hi"), assistant("hello"), user("second")]
    );
}

#[tokio::test]
async fn a_two_node_workflow_keeps_each_prompt_last() {
    let (turns, model) = scripted(
        ScriptedModel::builder()
            .respond_text("first answer")
            .respond_text("second answer")
            .build(),
    );
    let mut workflow = Workflow::new(card("llm-history-chain"));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"query": "${inputs.query}"})),
        )
        .add_workflow_comp(
            "llm1",
            LlmComponent::new(LlmCompConfig {
                model: Some(Arc::clone(&model)),
                user_prompt_template: Some("{{query}}".to_owned()),
                enable_history: true,
                ..LlmCompConfig::default()
            })
            .unwrap(),
            ComponentOptions::new().with_inputs_schema(json!({"query": "${start.query}"})),
        )
        .add_workflow_comp(
            "llm2",
            LlmComponent::new(LlmCompConfig {
                model: Some(Arc::clone(&model)),
                system_prompt_template: Some("second node".to_owned()),
                user_prompt_template: Some("previous: {{previous}}".to_owned()),
                enable_history: true,
                ..LlmCompConfig::default()
            })
            .unwrap(),
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
        .invoke(json!({"query": "hello"}), session("llm-history-chain"))
        .await
        .unwrap();
    assert_eq!(output.state, WorkflowExecutionState::Completed);
    assert_eq!(
        output.result,
        json!({"output": {"result": "second answer"}})
    );

    let calls = turns.calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(
        call_pairs(&calls[0]),
        vec![user("hello")],
        "the first node runs against an empty history"
    );
    let second = call_pairs(&calls[1]);
    assert_eq!(
        second,
        vec![
            system("second node"),
            user("hello"),
            assistant("first answer"),
            user("previous: first answer"),
        ],
        "the second node injects the shared turn after its system prompt"
    );
    assert_eq!(
        second.last(),
        Some(&user("previous: first answer")),
        "the injected prompt still ends with the node's own user message"
    );
}

/// An in-memory [`MessageHistory`] owned by one test.
#[derive(Default)]
struct FakeHistory {
    messages: Mutex<Vec<BaseMessage>>,
    appends: Mutex<Vec<Vec<BaseMessage>>>,
}

impl FakeHistory {
    fn contents(&self) -> Vec<BaseMessage> {
        self.messages.lock().clone()
    }

    fn appends(&self) -> Vec<Vec<BaseMessage>> {
        self.appends.lock().clone()
    }
}

#[async_trait]
impl MessageHistory for FakeHistory {
    async fn messages(&self) -> Vec<BaseMessage> {
        self.messages.lock().clone()
    }

    async fn append(&self, messages: Vec<BaseMessage>) {
        self.messages.lock().extend(messages.iter().cloned());
        self.appends.lock().push(messages);
    }

    async fn clear(&self) {
        self.messages.lock().clear();
    }
}

#[tokio::test]
async fn the_override_history_seam_drives_the_turn() {
    let (turns, model) = scripted(
        ScriptedModel::builder()
            .respond_text("a1")
            .respond_text("a2")
            .build(),
    );
    let llm = LlmComponent::new(LlmCompConfig {
        model: Some(model),
        user_prompt_template: Some("{{query}}".to_owned()),
        enable_history: true,
        ..LlmCompConfig::default()
    })
    .unwrap();
    let history = Arc::new(FakeHistory::default());
    let session: Arc<dyn Session> = session("llm-history-override");

    for (query, reply) in [("q1", "a1"), ("q2", "a2")] {
        let ctx = ComponentContext::new(Arc::clone(&session), "llm", "wf")
            .with_history(Arc::clone(&history) as Arc<dyn MessageHistory>);
        let output = llm.invoke(json!({"query": query}), &ctx).await.unwrap();
        assert_eq!(output, json!({"result": reply}));
    }

    assert_eq!(
        message_pairs(&history.contents()),
        vec![user("q1"), assistant("a1"), user("q2"), assistant("a2")],
        "the fake saw one user + one assistant entry per turn"
    );
    assert_eq!(
        history.appends().len(),
        2,
        "one append per turn, never one per message"
    );

    let calls = turns.calls();
    assert_eq!(
        call_pairs(&calls[1]),
        vec![user("q1"), assistant("a1"), user("q2")],
        "the second call is assembled from the fake's contents"
    );
    assert_eq!(
        session.get_state(MESSAGES_STATE_KEY).await.unwrap(),
        None,
        "the override keeps the session lane untouched"
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
