//! Built-in `llm` workflow component.
//!
//! Ports `openjiuwen/core/workflow/components/llm/llm_comp.py` onto
//! [`openjiuwen_foundation::Model`] and [`openjiuwen_foundation::PromptTemplate`].
//! The component turns its inputs into a conversation, sends it to the configured
//! model, and writes the answer under one output key.
//!
//! Three prompt shapes are supported, matching Python:
//!
//! - a system template only, rendered into `[system, user("")]`;
//! - a user template only, rendered into `[user]`;
//! - both templates, rendered into `[system, user]`;
//! - neither, in which case the stored `template_content` messages are rendered
//!   (their roles are preserved) or, when that list is empty, the conversation is
//!   a single empty user message.
//!
//! Rendering values contain every top-level key of the input object plus the whole
//! input under `"inputs"`. Only the `{{name}}` delimiters of
//! `PromptTemplate` are placeholders; single-brace `{name}` stays literal.
//!
//! # Response formatting
//!
//! `response_format` is applied in two steps, both ported from Python:
//!
//! - `format_prompt` (Python `LLMPromptFormatter.format_prompt`) rewrites the
//!   **last** user message with a format instruction — `DEFAULT_MARKDOWN_INSTRUCTION`
//!   / `DEFAULT_JSON_INSTRUCTION`, or the `markdown_instruction` /
//!   `json_instruction` override. JSON embeds the compact schema produced by
//!   `generate_json_schema` (Python `SchemaGenerator.generate_json_schema`).
//! - `shape_output` (Python `OutputFormatter.format_response` /
//!   `_format_text_response` / `_format_json_response`) turns the answer into the
//!   configured output object. JSON replies are parsed by
//!   `JsonParser.parse_json_content` after `_clean_markdown_blocks`, so a fenced
//!   ` ```json … ``` ` answer is accepted.
//!
//! A JSON-format `stream` is one non-streaming `Model::invoke` call whose whole
//! answer is shaped and yielded as a single frame (Python
//! `_invoke_for_json_format`). Text and markdown streams keep one frame per
//! non-empty content delta.
//!
//! No provider `response_format` parameter is ever sent: Python shapes the answer
//! text instead of asking the provider, and this component does the same.
//! Configuration failures use `[101002]` (text/markdown must name exactly one
//! output field) and JSON parsing/validation failures use `[101004]`. The schema
//! embedded in a prompt is serialized compactly, whereas Python's `json.dumps`
//! inserts spaces after `:` and `,`; the value only ever appears inside a prompt.
//!
//! # Conversation history
//!
//! `enable_history` ports Python's `_insert_history_to_system_and_user_prompt`,
//! `_write_user_message_to_context` and `_write_assistant_message_to_context`, driven by
//! `_get_model_input`:
//!
//! - the built prompt is split into its leading system messages and the rest, and the
//!   recorded history is spliced between them, so the current user prompt stays last;
//! - the format instruction is applied afterwards, rewriting that last user message and
//!   never a history entry;
//! - after a successful model call the sent user text and the raw assistant reply are
//!   appended as one turn, which a later call on the same session reads back.
//!
//! With `enable_history` off nothing is read from or written to the history store, so the
//! conversation stays byte-identical to a history-less component.
//!
//! Deviations from Python:
//!
//! - Python's `ModelContext` message list is reproduced by the session-backed
//!   [`crate::MessageHistory`] on the session global lane `__messages__`, so history is
//!   shared by every history-enabled LLM node running on the same session.
//! - The recorded user message is the conversation text actually sent (post
//!   `format_prompt`, so a markdown/JSON run records the instruction), and the assistant
//!   entry stores the raw reply rather than the shaped output object.
//! - The user and assistant entries are appended after a successful model call (Python
//!   writes the user message before calling the model), so a failed call records nothing.
//! - A streamed run records the accumulated text once at the end, not per delta.
//!
//! # Deferred behavior
//!
//! - Stream frames are content deltas: each frame carries the
//!   [`StreamChunk::Delta`](openjiuwen_foundation::StreamChunk::Delta) text, not the
//!   accumulated answer.

use std::fmt;
use std::sync::Arc;

use futures::stream::BoxStream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::{BaseMessage, ChatResponse, Model, ModelRequestConfig};

use crate::ability::ComponentAbility;
use crate::component::{Component, ComponentCard};
use crate::context::ComponentContext;
use crate::error::llm_invoke_call_failed;

mod format;
mod messages;

/// The requested shape of the model answer.
///
/// Serializes as a `snake_case` string (`"text"`, `"markdown"`, `"json"`), the form
/// Python stores in `response_format["type"]`. The value drives both the prompt
/// instruction and the shaped output object; see the module docs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlmResponseFormat {
    /// Plain text; the default.
    #[default]
    Text,
    /// Markdown text.
    Markdown,
    /// A JSON document.
    Json,
}

impl LlmResponseFormat {
    /// The wire name of this format (`"text"`, `"markdown"`, `"json"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Markdown => "markdown",
            Self::Json => "json",
        }
    }
}

impl fmt::Display for LlmResponseFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Configuration of an [`LlmComponent`].
///
/// Every field is optional except the model itself ([`LlmComponent::new`]
/// rejects a missing one). The struct is `Default`, and every setter consumes and
/// returns `self`.
#[derive(Default)]
pub struct LlmCompConfig {
    /// The model to call; required by [`LlmComponent::new`].
    pub model: Option<Arc<dyn Model>>,
    /// Logical model id carried for configuration parity.
    pub model_id: Option<String>,
    /// Request settings sent with every call.
    pub request: ModelRequestConfig,
    /// Optional system-prompt template (`{{name}}` placeholders).
    pub system_prompt_template: Option<String>,
    /// Optional user-prompt template (`{{name}}` placeholders).
    pub user_prompt_template: Option<String>,
    /// Fallback conversation used when neither template is set.
    pub template_content: Vec<BaseMessage>,
    /// Requested answer shape.
    pub response_format: LlmResponseFormat,
    /// Overrides the default markdown instruction (`response_format["markdownInstruction"]` in Python).
    pub markdown_instruction: Option<String>,
    /// Overrides the default JSON instruction (`response_format["jsonInstruction"]` in Python).
    pub json_instruction: Option<String>,
    /// Python `output_config`; its first key names the output field.
    pub output_config: Option<Value>,
    /// Whether history is inserted before the call and the turn recorded after it;
    /// see the module docs.
    pub enable_history: bool,
}

impl fmt::Debug for LlmCompConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LlmCompConfig")
            .field("model", &self.model.as_deref().map(Model::model_name))
            .field("model_id", &self.model_id)
            .field("request", &self.request)
            .field("system_prompt_template", &self.system_prompt_template)
            .field("user_prompt_template", &self.user_prompt_template)
            .field("template_content", &self.template_content)
            .field("response_format", &self.response_format)
            .field("markdown_instruction", &self.markdown_instruction)
            .field("json_instruction", &self.json_instruction)
            .field("output_config", &self.output_config)
            .field("enable_history", &self.enable_history)
            .finish()
    }
}

impl LlmCompConfig {
    /// Sets the model to call.
    #[must_use]
    pub fn with_model(mut self, model: Arc<dyn Model>) -> Self {
        self.model = Some(model);
        self
    }

    /// Sets the logical model id.
    #[must_use]
    pub fn with_model_id(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = Some(model_id.into());
        self
    }

    /// Replaces the request settings.
    #[must_use]
    pub fn with_request(mut self, request: ModelRequestConfig) -> Self {
        self.request = request;
        self
    }

    /// Sets the system-prompt template.
    #[must_use]
    pub fn with_system_prompt(mut self, template: impl Into<String>) -> Self {
        self.system_prompt_template = Some(template.into());
        self
    }

    /// Sets the user-prompt template.
    #[must_use]
    pub fn with_user_prompt(mut self, template: impl Into<String>) -> Self {
        self.user_prompt_template = Some(template.into());
        self
    }

    /// Replaces the fallback conversation.
    #[must_use]
    pub fn with_template_content(mut self, messages: Vec<BaseMessage>) -> Self {
        self.template_content = messages;
        self
    }

    /// Sets the requested answer shape.
    #[must_use]
    pub fn with_response_format(mut self, response_format: LlmResponseFormat) -> Self {
        self.response_format = response_format;
        self
    }

    /// Overrides the default markdown instruction.
    ///
    /// `${query}` in the instruction is replaced by the rendered user text.
    #[must_use]
    pub fn with_markdown_instruction(mut self, instruction: impl Into<String>) -> Self {
        self.markdown_instruction = Some(instruction.into());
        self
    }

    /// Overrides the default JSON instruction.
    ///
    /// `${query}` and `${json_schema}` are replaced before the call.
    #[must_use]
    pub fn with_json_instruction(mut self, instruction: impl Into<String>) -> Self {
        self.json_instruction = Some(instruction.into());
        self
    }

    /// Sets the Python `output_config`.
    #[must_use]
    pub fn with_output_config(mut self, output_config: Value) -> Self {
        self.output_config = Some(output_config);
        self
    }

    /// Enables (or disables) history insertion and turn recording; see the module docs.
    #[must_use]
    pub fn with_enable_history(mut self, enable_history: bool) -> Self {
        self.enable_history = enable_history;
        self
    }
}

/// Mirrors the Python class name `LLMCompConfig` (`llm_comp.py`).
///
/// A plain alias, not a distinct type: [`LlmCompConfig`] is the canonical Rust name and
/// both spellings name the same struct.
#[allow(clippy::upper_case_acronyms)]
pub type LLMCompConfig = LlmCompConfig;

/// The built-in `llm` component.
///
/// `invoke` returns one field named by [`LlmComponent::output_key`] whose value is
/// the assistant text; `stream` yields that same shape once per content delta.
pub struct LlmComponent {
    model: Arc<dyn Model>,
    config: LlmCompConfig,
}

impl fmt::Debug for LlmComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LlmComponent")
            .field("model", &self.model.model_name())
            .field("config", &self.config)
            .finish()
    }
}

impl LlmComponent {
    /// Builds a component from `config`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `config.model` is `None`: the model is
    /// the one dependency this component cannot synthesize.
    pub fn new(config: LlmCompConfig) -> Result<Self> {
        let Some(model) = config.model.clone() else {
            return Err(Error::validation("llm component requires a model"));
        };
        Ok(Self { model, config })
    }

    /// The configuration this component was built from.
    #[must_use]
    pub fn config(&self) -> &LlmCompConfig {
        &self.config
    }

    /// The model this component calls.
    #[must_use]
    pub fn model(&self) -> &Arc<dyn Model> {
        &self.model
    }

    /// The key the answer is written under.
    ///
    /// This is the first key of the normalized `output_config`, otherwise `"result"`
    /// (Python's `outputs_config` default; see `format::output_config_object`).
    #[must_use]
    pub fn output_key(&self) -> String {
        format::output_config_object(self.config.output_config.as_ref())
            .as_object()
            .and_then(|object| object.keys().next().cloned())
            .unwrap_or_else(|| "result".to_owned())
    }

    /// The wire frame carrying one answer fragment.
    pub(super) fn frame(output_key: &str, text: &str) -> Value {
        let mut output = serde_json::Map::new();
        output.insert(output_key.to_owned(), Value::String(text.to_owned()));
        Value::Object(output)
    }
}

#[async_trait::async_trait]
#[allow(clippy::unnecessary_literal_bound)]
impl Component for LlmComponent {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("llm", "llm")
            .with_description("calls a model with a rendered prompt")
            .with_abilities(vec![ComponentAbility::Invoke, ComponentAbility::Stream])
    }

    fn abilities(&self) -> Vec<ComponentAbility> {
        vec![ComponentAbility::Invoke, ComponentAbility::Stream]
    }

    fn component_type(&self) -> &str {
        "llm"
    }

    async fn invoke(&self, inputs: Value, ctx: &ComponentContext) -> Result<Value> {
        let messages = self.assemble_messages(&inputs, ctx).await;
        let response: ChatResponse = self
            .model
            .invoke(&messages, &self.config.request)
            .await
            .map_err(|error| llm_invoke_call_failed(&format!("{error}")))?;
        let text = response
            .message()
            .map(|message| message.text().into_owned())
            .unwrap_or_default();
        // Python records the turn before `_create_output`, so a shaping failure still
        // leaves it in the history.
        self.record_turn(&messages, &text, ctx).await;
        format::shape_output(
            &text,
            self.config.response_format,
            self.config.output_config.as_ref(),
        )
    }

    fn stream(
        self: Arc<Self>,
        inputs: Value,
        ctx: Arc<ComponentContext>,
    ) -> BoxStream<'static, Result<Value>> {
        let this = Arc::clone(&self);
        let ctx = Arc::clone(&ctx);
        let request = self.config.request.clone();
        let response_format = self.config.response_format;
        let output_config = self.config.output_config.clone();

        // Python's `_invoke_for_json_format`: a JSON answer cannot be shaped per
        // delta, so the stream is one non-streaming call yielding one frame.
        if response_format == LlmResponseFormat::Json {
            return Box::pin(async_stream::stream! {
                let messages = this.assemble_messages(&inputs, &ctx).await;
                match this.model.invoke(&messages, &request).await {
                    Err(error) => yield Err(llm_invoke_call_failed(&format!("{error}"))),
                    Ok(response) => {
                        let text = response
                            .message()
                            .map(|message| message.text().into_owned())
                            .unwrap_or_default();
                        this.record_turn(&messages, &text, &ctx).await;
                        match format::shape_output(&text, response_format, output_config.as_ref()) {
                            Ok(value) => yield Ok(value),
                            Err(error) => yield Err(error),
                        }
                    }
                }
            });
        }

        // The one-field rule is a configuration error, so it fails before any frame
        // is produced rather than on the first delta.
        if let Err(error) = format::single_output_key(output_config.as_ref()) {
            return Box::pin(futures::stream::iter(std::iter::once(Err::<Value, Error>(
                error,
            ))));
        }

        Box::pin(async_stream::stream! {
            let messages = this.assemble_messages(&inputs, &ctx).await;
            let mut source = this.model.stream(messages.clone(), request);
            // Python's `_stream_with_chunks` records the whole answer once the stream
            // ends, so the accumulator is the raw delta text.
            let mut accumulated = String::new();
            while let Some(item) = source.next().await {
                match item {
                    Err(error) => {
                        yield Err(llm_invoke_call_failed(&format!("{error}")));
                        // A provider failure terminates the stream: emit one error
                        // frame and stop instead of forwarding trailing chunks.
                        return;
                    }
                    Ok(chunk) => {
                        if let Some(text) = chunk.content() {
                            if !text.is_empty() {
                                accumulated.push_str(text);
                                match format::shape_output(
                                    text,
                                    response_format,
                                    output_config.as_ref(),
                                ) {
                                    Ok(value) => yield Ok(value),
                                    Err(error) => {
                                        yield Err(error);
                                        return;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            this.record_turn(&messages, &accumulated, &ctx).await;
        })
    }
}

/// Mirrors the Python class name `LLMComponent` (`llm_comp.py`).
///
/// A plain alias, not a distinct type: [`LlmComponent`] is the canonical Rust name and
/// both spellings name the same struct.
#[allow(clippy::upper_case_acronyms)]
pub type LLMComponent = LlmComponent;

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::needless_pass_by_value,
    clippy::unnecessary_literal_bound
)]
mod tests {
    use std::sync::Arc;

    use futures::StreamExt;
    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_foundation::{BaseMessage, Role, StreamChunk};
    use openjiuwen_session::{Session, WorkflowSession};
    use openjiuwen_test_support::{ScriptedError, ScriptedModel};
    use serde_json::json;

    use super::*;

    fn context() -> ComponentContext {
        let session: Arc<dyn Session> = Arc::new(WorkflowSession::new("llm-tests"));
        ComponentContext::new(session, "llm", "wf")
    }

    fn component(config: LlmCompConfig) -> LlmComponent {
        LlmComponent::new(config).expect("model is set")
    }

    fn roles(messages: &[BaseMessage]) -> Vec<Role> {
        messages
            .iter()
            .map(|message| message.role.clone())
            .collect()
    }

    fn texts(messages: &[BaseMessage]) -> Vec<String> {
        messages
            .iter()
            .map(|message| message.text().into_owned())
            .collect()
    }

    #[tokio::test]
    async fn system_and_user_templates_are_rendered() {
        let model = ScriptedModel::text("answer");
        let llm = component(LlmCompConfig {
            model: Some(Arc::new(model.clone())),
            system_prompt_template: Some("system {{name}}".to_owned()),
            user_prompt_template: Some("hello {{name}}, inputs={{inputs}}".to_owned()),
            ..LlmCompConfig::default()
        });

        let output = llm
            .invoke(json!({"name": "bob"}), &context())
            .await
            .unwrap();
        assert_eq!(output, json!({"result": "answer"}));

        let call = model.last_call().expect("one call");
        assert_eq!(roles(&call.messages), vec![Role::System, Role::User]);
        assert_eq!(texts(&call.messages)[0], "system bob");
        assert_eq!(
            texts(&call.messages)[1],
            format!("hello bob, inputs={}", json!({"name": "bob"}))
        );
    }

    #[tokio::test]
    async fn user_only_template_yields_one_user_message() {
        let model = ScriptedModel::text("answer");
        let llm = component(LlmCompConfig {
            model: Some(Arc::new(model.clone())),
            user_prompt_template: Some("ask {{q}}".to_owned()),
            ..LlmCompConfig::default()
        });

        llm.invoke(json!({"q": "why"}), &context()).await.unwrap();

        let call = model.last_call().expect("one call");
        assert_eq!(roles(&call.messages), vec![Role::User]);
        assert_eq!(texts(&call.messages), vec!["ask why".to_owned()]);
    }

    #[tokio::test]
    async fn single_brace_placeholders_stay_literal() {
        let model = ScriptedModel::text("answer");
        let llm = component(LlmCompConfig {
            model: Some(Arc::new(model.clone())),
            user_prompt_template: Some("literal {name}".to_owned()),
            ..LlmCompConfig::default()
        });

        llm.invoke(json!({"name": "bob"}), &context())
            .await
            .unwrap();
        assert_eq!(
            texts(&model.last_call().unwrap().messages),
            vec!["literal {name}".to_owned()]
        );
    }

    #[tokio::test]
    async fn template_content_is_rendered_keeping_roles() {
        let model = ScriptedModel::text("answer");
        let llm = component(LlmCompConfig {
            model: Some(Arc::new(model.clone())),
            template_content: vec![
                BaseMessage::system("system {{x}}"),
                BaseMessage::user("user {{x}}"),
            ],
            ..LlmCompConfig::default()
        });

        llm.invoke(json!({"x": 7}), &context()).await.unwrap();

        let call = model.last_call().expect("one call");
        assert_eq!(roles(&call.messages), vec![Role::System, Role::User]);
        assert_eq!(texts(&call.messages), vec!["system 7", "user 7"]);
    }

    #[tokio::test]
    async fn empty_template_content_falls_back_to_empty_user_message() {
        let model = ScriptedModel::text("answer");
        let llm = component(LlmCompConfig {
            model: Some(Arc::new(model.clone())),
            ..LlmCompConfig::default()
        });

        llm.invoke(json!({}), &context()).await.unwrap();

        let call = model.last_call().expect("one call");
        assert_eq!(roles(&call.messages), vec![Role::User]);
        assert_eq!(texts(&call.messages), vec![String::new()]);
    }

    #[tokio::test]
    async fn non_object_inputs_render_only_the_inputs_key() {
        let model = ScriptedModel::text("answer");
        let llm = component(LlmCompConfig {
            model: Some(Arc::new(model.clone())),
            user_prompt_template: Some("all={{inputs}}".to_owned()),
            ..LlmCompConfig::default()
        });

        llm.invoke(json!("hello"), &context()).await.unwrap();
        assert_eq!(
            texts(&model.last_call().unwrap().messages),
            vec!["all=hello".to_owned()]
        );
    }

    #[tokio::test]
    async fn output_key_comes_from_output_config() {
        let model = ScriptedModel::text("answer");
        let llm = component(LlmCompConfig {
            model: Some(Arc::new(model)),
            output_config: Some(json!({"answer": {"type": "string"}})),
            ..LlmCompConfig::default()
        });

        assert_eq!(llm.output_key(), "answer");
        let output = llm.invoke(json!({}), &context()).await.unwrap();
        assert_eq!(output, json!({"answer": "answer"}));
    }

    #[test]
    fn default_output_key_is_result() {
        let llm = component(LlmCompConfig {
            model: Some(Arc::new(ScriptedModel::text("x"))),
            ..LlmCompConfig::default()
        });
        assert_eq!(llm.output_key(), "result");
    }

    #[test]
    fn output_config_without_object_keys_falls_back_to_result() {
        let llm = component(LlmCompConfig {
            model: Some(Arc::new(ScriptedModel::text("x"))),
            output_config: Some(json!({})),
            ..LlmCompConfig::default()
        });
        assert_eq!(llm.output_key(), "result");
    }

    #[tokio::test]
    async fn stream_yields_content_deltas_in_order() {
        let model = ScriptedModel::builder()
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
                    reasoning: Some("thinking".to_owned()),
                    tool_call: None,
                    finish_reason: None,
                },
                StreamChunk::Done,
            ])
            .build();
        let llm: Arc<dyn Component> = Arc::new(component(LlmCompConfig {
            model: Some(Arc::new(model)),
            ..LlmCompConfig::default()
        }));

        let frames: Vec<Value> = llm
            .stream(json!({}), Arc::new(context()))
            .map(|frame| frame.unwrap())
            .collect()
            .await;
        assert_eq!(
            frames,
            vec![json!({"result": "he"}), json!({"result": "llo"})]
        );
    }

    #[tokio::test]
    async fn stream_error_yields_one_error_frame() {
        let model = ScriptedModel::builder()
            .fail(ScriptedError::llm("boom"))
            .build();
        let llm: Arc<dyn Component> = Arc::new(component(LlmCompConfig {
            model: Some(Arc::new(model)),
            ..LlmCompConfig::default()
        }));

        let frames: Vec<_> = llm.stream(json!({}), Arc::new(context())).collect().await;
        assert_eq!(frames.len(), 1);
        let error = frames.into_iter().next().unwrap().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Workflow);
        assert!(error.message().contains("[101003]"));
        assert!(error.message().contains("boom"));
    }

    #[tokio::test]
    async fn invoke_error_is_wrapped_with_the_python_code() {
        let model = ScriptedModel::builder()
            .fail(ScriptedError::llm("boom"))
            .build();
        let llm = component(LlmCompConfig {
            model: Some(Arc::new(model)),
            ..LlmCompConfig::default()
        });

        let error = llm.invoke(json!({}), &context()).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Workflow);
        assert!(error.message().contains("[101003]"));
        assert!(error.message().contains("boom"));
    }

    #[test]
    fn missing_model_is_a_validation_error() {
        let error = LlmComponent::new(LlmCompConfig::default()).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(error.message(), "llm component requires a model");
    }

    #[test]
    fn response_format_serde_is_snake_case() {
        assert_eq!(
            serde_json::to_value(LlmResponseFormat::Json).unwrap(),
            json!("json")
        );
        assert_eq!(
            serde_json::to_value(LlmResponseFormat::Markdown).unwrap(),
            json!("markdown")
        );
        assert_eq!(
            serde_json::from_value::<LlmResponseFormat>(json!("text")).unwrap(),
            LlmResponseFormat::Text
        );
        assert_eq!(LlmResponseFormat::default(), LlmResponseFormat::Text);
        assert_eq!(LlmResponseFormat::Json.as_str(), "json");
        assert_eq!(LlmResponseFormat::Json.to_string(), "json");
    }

    #[test]
    fn config_builders_are_chainable() {
        let model: Arc<dyn Model> = Arc::new(ScriptedModel::text("x"));
        let config = LlmCompConfig::default()
            .with_model(Arc::clone(&model))
            .with_model_id("m")
            .with_request(ModelRequestConfig::new("gpt"))
            .with_system_prompt("s")
            .with_user_prompt("u")
            .with_template_content(vec![BaseMessage::user("t")])
            .with_response_format(LlmResponseFormat::Json)
            .with_markdown_instruction("md ${query}")
            .with_json_instruction("json ${json_schema} ${query}")
            .with_output_config(json!({"out": {}}))
            .with_enable_history(true);

        assert_eq!(config.model_id.as_deref(), Some("m"));
        assert_eq!(config.request.model_name, "gpt");
        assert_eq!(config.system_prompt_template.as_deref(), Some("s"));
        assert_eq!(config.user_prompt_template.as_deref(), Some("u"));
        assert_eq!(config.template_content.len(), 1);
        assert_eq!(config.response_format, LlmResponseFormat::Json);
        assert_eq!(config.markdown_instruction.as_deref(), Some("md ${query}"));
        assert_eq!(
            config.json_instruction.as_deref(),
            Some("json ${json_schema} ${query}")
        );
        assert_eq!(config.output_config, Some(json!({"out": {}})));
        assert!(config.enable_history);

        let llm = component(config);
        assert_eq!(llm.component_type(), "llm");
        assert_eq!(
            llm.abilities(),
            vec![ComponentAbility::Invoke, ComponentAbility::Stream]
        );
        assert_eq!(llm.card().id, "llm");
        assert_eq!(llm.card().component_type, "llm");
    }
}
