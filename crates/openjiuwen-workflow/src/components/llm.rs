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
//! [`PromptTemplate`] are placeholders; single-brace `{name}` stays literal.
//!
//! # Deferred behavior
//!
//! - `enable_history` is carried for builder/spec parity but history insertion is
//!   deferred: there is no history source on [`ComponentContext`] yet, and
//!   inventing one would change the observable conversation.
//! - `response_format` is carried but does not reshape the answer; Python applies
//!   the format through a post-processing step that needs a JSON-schema
//!   generator.
//! - Stream frames are content deltas: each frame carries the
//!   [`StreamChunk::Delta`] text, not the accumulated answer.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use futures::stream::BoxStream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::{
    BaseMessage, ChatResponse, Model, ModelRequestConfig, PromptTemplate, TemplateContent,
};

use crate::ability::ComponentAbility;
use crate::component::{Component, ComponentCard};
use crate::context::ComponentContext;
use crate::error::llm_invoke_call_failed;

/// The requested shape of the model answer.
///
/// Serializes as a `snake_case` string (`"text"`, `"markdown"`, `"json"`), the form
/// Python stores in `response_format["type"]`. The value is carried for
/// configuration parity; see the module docs for the deferred post-processing.
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
    /// Python `output_config`; its first key names the output field.
    pub output_config: Option<Value>,
    /// Whether history is inserted before the call; deferred (see module docs).
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

    /// Sets the Python `output_config`.
    #[must_use]
    pub fn with_output_config(mut self, output_config: Value) -> Self {
        self.output_config = Some(output_config);
        self
    }

    /// Enables (or disables) history insertion; deferred (see module docs).
    #[must_use]
    pub fn with_enable_history(mut self, enable_history: bool) -> Self {
        self.enable_history = enable_history;
        self
    }
}

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
    /// This is the first key of `output_config` when it is a JSON object with at
    /// least one key, otherwise `"result"` (Python's `outputs_config` default).
    #[must_use]
    pub fn output_key(&self) -> String {
        self.config
            .output_config
            .as_ref()
            .and_then(Value::as_object)
            .and_then(|object| object.keys().next().cloned())
            .unwrap_or_else(|| "result".to_owned())
    }

    /// Builds the conversation for `inputs`.
    ///
    /// See the module docs for the template precedence rules.
    fn build_messages(&self, inputs: &Value) -> Vec<BaseMessage> {
        let values = render_values(inputs);
        match (
            self.config.system_prompt_template.as_deref(),
            self.config.user_prompt_template.as_deref(),
        ) {
            (Some(system), None) => vec![
                BaseMessage::system(render_text(system, &values)),
                BaseMessage::user(""),
            ],
            (None, Some(user)) => vec![BaseMessage::user(render_text(user, &values))],
            (Some(system), Some(user)) => vec![
                BaseMessage::system(render_text(system, &values)),
                BaseMessage::user(render_text(user, &values)),
            ],
            (None, None) => {
                if self.config.template_content.is_empty() {
                    vec![BaseMessage::user("")]
                } else {
                    PromptTemplate::from_messages(self.config.template_content.clone())
                        .format(&values)
                        .to_messages()
                }
            }
        }
    }

    /// The wire frame carrying one answer fragment.
    fn frame(output_key: &str, text: &str) -> Value {
        let mut output = serde_json::Map::new();
        output.insert(output_key.to_owned(), Value::String(text.to_owned()));
        Value::Object(output)
    }
}

/// The rendering values of `inputs`: its top-level entries plus `"inputs"`.
fn render_values(inputs: &Value) -> BTreeMap<String, Value> {
    let mut values = BTreeMap::new();
    if let Value::Object(object) = inputs {
        values.extend(
            object
                .iter()
                .map(|(key, value)| (key.clone(), value.clone())),
        );
    }
    values.insert("inputs".to_owned(), inputs.clone());
    values
}

/// Renders `{{name}}` placeholders in one template string.
///
/// An unresolvable placeholder is kept verbatim by [`PromptTemplate`], matching
/// Python's partial rendering.
fn render_text(template: &str, values: &BTreeMap<String, Value>) -> String {
    match PromptTemplate::new(template).format(values).content {
        TemplateContent::Text(text) => text,
        // `PromptTemplate::new` always builds text content; join defensively so a
        // future change cannot panic here.
        TemplateContent::Messages(messages) => messages
            .iter()
            .map(|message| message.text().into_owned())
            .collect::<String>(),
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

    async fn invoke(&self, inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
        let messages = self.build_messages(&inputs);
        let response: ChatResponse = self
            .model
            .invoke(&messages, &self.config.request)
            .await
            .map_err(|error| llm_invoke_call_failed(&format!("{error}")))?;
        let text = response
            .message()
            .map(|message| message.text().into_owned())
            .unwrap_or_default();
        Ok(Self::frame(&self.output_key(), &text))
    }

    fn stream(
        self: Arc<Self>,
        inputs: Value,
        _ctx: Arc<ComponentContext>,
    ) -> BoxStream<'static, Result<Value>> {
        let messages = self.build_messages(&inputs);
        let request = self.config.request.clone();
        let output_key = self.output_key();
        let mut source = self.model.stream(messages, request);
        Box::pin(async_stream::stream! {
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
                                yield Ok(LlmComponent::frame(&output_key, text));
                            }
                        }
                    }
                }
            }
        })
    }
}

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
            output_config: Some(json!({"answer": {"type": "string"}, "other": {}})),
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
            .with_output_config(json!({"out": {}}))
            .with_enable_history(true);

        assert_eq!(config.model_id.as_deref(), Some("m"));
        assert_eq!(config.request.model_name, "gpt");
        assert_eq!(config.system_prompt_template.as_deref(), Some("s"));
        assert_eq!(config.user_prompt_template.as_deref(), Some("u"));
        assert_eq!(config.template_content.len(), 1);
        assert_eq!(config.response_format, LlmResponseFormat::Json);
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
