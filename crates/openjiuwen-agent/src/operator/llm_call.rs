//! `LLMCallOperator`: prompt parameters of a model call plus the real call
//! path.
//!
//! Ports `openjiuwen/core/operator/llm_call/base.py`. The operator owns the
//! system prompt, the user prompt template (`{{query}}` substitution), the
//! model name, and the sampling temperature. Python's handle does not execute
//! the call; the Rust port additionally exposes [`LLMCallOperator::call`] and
//! [`LLMCallOperator::render_messages`] so an evolving agent can drive the
//! tuned prompt without duplicating the rendering rules.
use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::{BaseMessage, ChatResponse, MessageContent, Model, ModelRequestConfig};
use serde_json::{json, Value};

use super::base::{ApplyResult, Operator, ParameterCallback, TunableKind, TunableSpec};

/// User prompt used when the operator is constructed without one.
pub const DEFAULT_USER_PROMPT: &str = "{{query}}";

/// Placeholder replaced by the caller's query in both prompts.
pub const QUERY_PLACEHOLDER: &str = "{{query}}";

/// Tunable prompt/model parameters of one LLM call.
pub struct LLMCallOperator {
    operator_id: String,
    system_prompt: Vec<BaseMessage>,
    user_prompt: String,
    model: Option<String>,
    temperature: Option<f32>,
    freeze_system_prompt: bool,
    freeze_user_prompt: bool,
    on_parameter_updated: Option<ParameterCallback>,
}

impl LLMCallOperator {
    /// Creates a handle with no system prompt, the default user prompt
    /// (`{{query}}`), no model override, no temperature override, an
    /// evolvable system prompt, and a frozen user prompt (Python defaults).
    #[must_use]
    pub fn new(operator_id: impl Into<String>) -> Self {
        Self {
            operator_id: operator_id.into(),
            system_prompt: Vec::new(),
            user_prompt: DEFAULT_USER_PROMPT.to_owned(),
            model: None,
            temperature: None,
            freeze_system_prompt: false,
            freeze_user_prompt: true,
            on_parameter_updated: None,
        }
    }

    /// Sets the system prompt messages.
    #[must_use]
    pub fn with_system_prompt(mut self, system_prompt: Vec<BaseMessage>) -> Self {
        self.system_prompt = system_prompt;
        self
    }

    /// Sets the user prompt template.
    #[must_use]
    pub fn with_user_prompt(mut self, user_prompt: impl Into<String>) -> Self {
        let user_prompt = user_prompt.into();
        self.user_prompt = if user_prompt.is_empty() {
            DEFAULT_USER_PROMPT.to_owned()
        } else {
            user_prompt
        };
        self
    }

    /// Sets the model name used by [`LLMCallOperator::call`].
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets the sampling temperature used by [`LLMCallOperator::call`].
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Freezes (or unfreezes) the system prompt against evolution updates.
    #[must_use]
    pub const fn with_freeze_system_prompt(mut self, freeze: bool) -> Self {
        self.freeze_system_prompt = freeze;
        self
    }

    /// Freezes (or unfreezes) the user prompt against evolution updates.
    #[must_use]
    pub const fn with_freeze_user_prompt(mut self, freeze: bool) -> Self {
        self.freeze_user_prompt = freeze;
        self
    }

    /// Installs the parameter callback.
    #[must_use]
    pub fn with_callback(mut self, callback: ParameterCallback) -> Self {
        self.on_parameter_updated = Some(callback);
        self
    }

    /// The current system prompt messages.
    #[must_use]
    pub fn system_prompt(&self) -> &[BaseMessage] {
        &self.system_prompt
    }

    /// The current user prompt template.
    #[must_use]
    pub fn user_prompt(&self) -> &str {
        &self.user_prompt
    }

    /// Whether the system prompt is frozen (Python `get_freeze_system_prompt`).
    #[must_use]
    pub const fn freeze_system_prompt(&self) -> bool {
        self.freeze_system_prompt
    }

    /// Whether the user prompt is frozen (Python `get_freeze_user_prompt`).
    #[must_use]
    pub const fn freeze_user_prompt(&self) -> bool {
        self.freeze_user_prompt
    }

    /// The model-name override, when set.
    #[must_use]
    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    /// The temperature override, when set.
    #[must_use]
    pub const fn temperature(&self) -> Option<f32> {
        self.temperature
    }

    /// Renders the operator's prompts for `query`: the system messages with
    /// `{{query}}` substituted, followed by one user message.
    #[must_use]
    pub fn render_messages(&self, query: &str) -> Vec<BaseMessage> {
        let mut messages: Vec<BaseMessage> = self
            .system_prompt
            .iter()
            .map(|message| substitute(message, query))
            .collect();
        messages.push(BaseMessage::user(self.user_prompt_for(query)));
        messages
    }

    /// Calls `model` with the tuned prompts.
    ///
    /// The query is taken from the last user message of `messages`; earlier
    /// non-system messages are preserved as history, while system messages are
    /// replaced by the operator's own. A model/temperature override replaces
    /// the matching field of `request`.
    ///
    /// # Errors
    ///
    /// Returns the provider's error from [`Model::invoke`].
    pub async fn call(
        &self,
        model: &dyn Model,
        messages: &[BaseMessage],
        request: ModelRequestConfig,
    ) -> Result<ChatResponse> {
        let query = messages
            .iter()
            .rev()
            .find(|message| message.role.is_user())
            .map_or_else(String::new, |message| message.text().into_owned());
        let rendered = self.render_conversation(messages, &query);
        let mut request = request;
        if let Some(name) = &self.model {
            request.model_name.clone_from(name);
        }
        if let Some(temperature) = self.temperature {
            request.temperature = Some(temperature);
        }
        model.invoke(&rendered, &request).await
    }

    /// Renders the full conversation for `call`.
    fn render_conversation(&self, messages: &[BaseMessage], query: &str) -> Vec<BaseMessage> {
        let mut rendered: Vec<BaseMessage> = self
            .system_prompt
            .iter()
            .map(|message| substitute(message, query))
            .collect();
        if let Some(index) = messages.iter().rposition(|message| message.role.is_user()) {
            rendered.extend(
                messages[..index]
                    .iter()
                    .filter(|message| !message.role.is_system())
                    .cloned(),
            );
        }
        rendered.push(BaseMessage::user(self.user_prompt_for(query)));
        rendered
    }

    /// Renders the user prompt for `query`.
    fn user_prompt_for(&self, query: &str) -> String {
        let rendered = self.user_prompt.replace(QUERY_PLACEHOLDER, query);
        if rendered.is_empty() {
            query.to_owned()
        } else {
            rendered
        }
    }

    /// Stores one field without the freeze check and returns its canonical
    /// JSON value for the callback.
    fn store(&mut self, target: &str, value: &Value) -> Result<Value> {
        match target {
            "system_prompt" => {
                self.system_prompt = parse_system_prompt(value)?;
                Ok(serde_json::to_value(&self.system_prompt).unwrap_or(Value::Null))
            }
            "user_prompt" => {
                let text = parse_prompt_text(value)?;
                self.user_prompt = if text.is_empty() {
                    DEFAULT_USER_PROMPT.to_owned()
                } else {
                    text
                };
                Ok(Value::String(self.user_prompt.clone()))
            }
            "model" => {
                self.model = match value {
                    Value::Null => None,
                    Value::String(name) if !name.trim().is_empty() => Some(name.clone()),
                    _ => {
                        return Err(Error::validation(
                            "tunable 'model' expects a text value or null",
                        ))
                    }
                };
                Ok(self.model.clone().map_or(Value::Null, Value::String))
            }
            "temperature" => {
                self.temperature = match value {
                    Value::Null => None,
                    other => Some(serde_json::from_value::<f32>(other.clone()).map_err(|_| {
                        Error::validation("tunable 'temperature' expects a number or null")
                    })?),
                };
                Ok(self
                    .temperature
                    .map_or(Value::Null, |value| json!(f64::from(value))))
            }
            other => Err(Error::validation(format!("unknown target '{other}'"))),
        }
    }

    /// Fires the callback for `target`.
    fn fire(&self, target: &str, value: &Value) {
        if let Some(callback) = &self.on_parameter_updated {
            callback(target, value);
        }
    }
}

#[async_trait]
impl Operator for LLMCallOperator {
    fn operator_id(&self) -> &str {
        &self.operator_id
    }

    fn get_tunables(&self) -> Vec<TunableSpec> {
        vec![
            TunableSpec::new("system_prompt", TunableKind::TextList)
                .with_description("system prompt messages")
                .with_frozen(self.freeze_system_prompt),
            TunableSpec::new("user_prompt", TunableKind::Text)
                .with_description("user prompt template with {{query}} substitution")
                .with_frozen(self.freeze_user_prompt),
            TunableSpec::new("model", TunableKind::Text).with_description("model name override"),
            TunableSpec::new("temperature", TunableKind::Number)
                .with_description("sampling temperature override"),
        ]
    }

    fn get_state(&self) -> Value {
        json!({
            "system_prompt": serde_json::to_value(&self.system_prompt).unwrap_or(Value::Null),
            "user_prompt": self.user_prompt,
            "model": self.model,
            "temperature": self.temperature.map(f64::from),
        })
    }

    fn set_parameter(&mut self, target: &str, value: Value) -> Result<ApplyResult> {
        let frozen = match target {
            "system_prompt" => self.freeze_system_prompt,
            "user_prompt" => self.freeze_user_prompt,
            _ => false,
        };
        if frozen {
            return Ok(ApplyResult::skipped(target, "frozen"));
        }
        let stored = match self.store(target, &value) {
            Ok(stored) => stored,
            Err(error) if error.kind() == openjiuwen_common::error::ErrorKind::Validation => {
                return Ok(ApplyResult::skipped(target, error.to_string()));
            }
            Err(error) => return Err(error),
        };
        self.fire(target, &stored);
        Ok(ApplyResult::applied(target))
    }

    fn load_state(&mut self, state: Value) -> Result<()> {
        let object = state
            .as_object()
            .ok_or_else(|| Error::validation("operator state must be a JSON object"))?;
        for target in ["system_prompt", "user_prompt", "model", "temperature"] {
            let Some(value) = object.get(target) else {
                continue;
            };
            let stored = self.store(target, value)?;
            self.fire(target, &stored);
        }
        Ok(())
    }
}

/// Parses a system prompt value: one text message, a list of text messages, or
/// a list of message objects.
fn parse_system_prompt(value: &Value) -> Result<Vec<BaseMessage>> {
    match value {
        Value::String(text) => Ok(vec![BaseMessage::system(text.clone())]),
        Value::Array(items) => items
            .iter()
            .map(|item| match item {
                Value::String(text) => Ok(BaseMessage::system(text.clone())),
                other => serde_json::from_value(other.clone())
                    .map_err(|error| Error::validation(format!("invalid prompt message: {error}"))),
            })
            .collect(),
        other => Err(Error::validation(format!(
            "tunable 'system_prompt' expects text or a list of messages, got {}",
            json_type(other)
        ))),
    }
}

/// Parses a user prompt value: text, or a list whose text parts are joined.
fn parse_prompt_text(value: &Value) -> Result<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::Array(items) => {
            let mut parts = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    Value::String(text) => parts.push(text.clone()),
                    other => {
                        let message: BaseMessage =
                            serde_json::from_value(other.clone()).map_err(|error| {
                                Error::validation(format!("invalid prompt message: {error}"))
                            })?;
                        parts.push(message.text().into_owned());
                    }
                }
            }
            Ok(parts.join("\n"))
        }
        other => Err(Error::validation(format!(
            "tunable 'user_prompt' expects text or a list of messages, got {}",
            json_type(other)
        ))),
    }
}

/// Replaces `{{query}}` in a message's text, keeping its role and metadata.
fn substitute(message: &BaseMessage, query: &str) -> BaseMessage {
    let mut rendered = message.clone();
    rendered.content = MessageContent::Text(message.text().replace(QUERY_PLACEHOLDER, query));
    rendered
}

/// The JSON type name of `value`, used in validation messages.
fn json_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "text",
        Value::Array(_) => "list",
        Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use openjiuwen_test_support::ScriptedModel;
    use std::sync::Arc;

    /// Captures every callback invocation.
    type Recorded = Arc<parking_lot::Mutex<Vec<(String, Value)>>>;

    fn recording_callback() -> (ParameterCallback, Recorded) {
        let log = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let sink = Arc::clone(&log);
        let callback: ParameterCallback = Arc::new(move |target: &str, value: &Value| {
            sink.lock().push((target.to_owned(), value.clone()));
        });
        (callback, log)
    }

    fn operator() -> LLMCallOperator {
        LLMCallOperator::new("agent/llm_call")
            .with_system_prompt(vec![BaseMessage::system("You are {{query}}")])
            .with_user_prompt("Answer {{query}}")
    }

    #[test]
    fn tunables_report_freeze_flags() {
        let frozen = LLMCallOperator::new("op").with_freeze_system_prompt(true);
        let tunables = frozen.get_tunables();
        let names: Vec<&str> = tunables.iter().map(|spec| spec.name.as_str()).collect();
        assert_eq!(
            names,
            ["system_prompt", "user_prompt", "model", "temperature"]
        );
        assert!(tunables[0].frozen);
        assert!(tunables[1].frozen);
        assert!(!tunables[2].frozen);
        assert_eq!(tunables[0].kind, TunableKind::TextList);
        assert_eq!(tunables[3].kind, TunableKind::Number);
    }

    #[test]
    fn set_parameter_updates_and_fires_callback() {
        let (callback, log) = recording_callback();
        let mut operator = operator().with_callback(callback);
        let result = operator
            .set_parameter("system_prompt", json!("be {{query}}"))
            .unwrap();
        assert!(result.applied);
        assert_eq!(operator.system_prompt()[0].text(), "be {{query}}");
        assert_eq!(
            log.lock().as_slice(),
            [(
                "system_prompt".to_owned(),
                json!([{"role": "system", "content": "be {{query}}"}])
            )]
        );
        let result = operator.set_parameter("temperature", json!(0.4)).unwrap();
        assert!(result.applied);
        assert_eq!(operator.temperature(), Some(0.4));
        let result = operator.set_parameter("model", Value::Null).unwrap();
        assert!(result.applied);
        assert_eq!(operator.model(), None);
    }

    #[test]
    fn frozen_targets_are_skipped() {
        let (callback, log) = recording_callback();
        let mut operator = operator().with_callback(callback);
        let result = operator
            .set_parameter("user_prompt", json!("changed"))
            .unwrap();
        assert!(!result.applied);
        assert_eq!(result.reason, "frozen");
        assert_eq!(operator.user_prompt(), "Answer {{query}}");
        assert!(log.lock().is_empty());
    }

    #[test]
    fn render_messages_substitutes_the_query() {
        let operator = operator();
        let messages = operator.render_messages("weather");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].text(), "You are weather");
        assert!(messages[0].role.is_system());
        assert_eq!(messages[1].text(), "Answer weather");
        assert!(messages[1].role.is_user());

        let default = LLMCallOperator::new("op");
        assert_eq!(default.render_messages("hi")[0].text(), "hi");
    }

    #[tokio::test]
    async fn call_renders_prompts_and_applies_overrides() {
        let model = ScriptedModel::text("done");
        let operator = operator().with_model("tuned-model").with_temperature(0.5);
        let response = operator
            .call(
                &model,
                &[BaseMessage::user("hello")],
                ModelRequestConfig::new("base-model"),
            )
            .await
            .unwrap();
        assert_eq!(response.message().unwrap().text(), "done");
        let call = model.last_call().unwrap();
        assert_eq!(call.messages.len(), 2);
        assert_eq!(call.messages[0].text(), "You are hello");
        assert_eq!(call.last_user_text().unwrap(), "Answer hello");
        assert_eq!(call.request.model_name, "tuned-model");
        assert_eq!(call.request.temperature, Some(0.5));
    }

    #[test]
    fn call_keeps_history_but_replaces_system_messages() {
        let operator = operator();
        let history = vec![
            BaseMessage::system("stale system"),
            BaseMessage::user("first"),
            BaseMessage::assistant("reply"),
            BaseMessage::user("second"),
        ];
        let rendered = operator.render_conversation(&history, "second");
        let roles: Vec<&str> = rendered
            .iter()
            .map(|message| message.role.as_str())
            .collect();
        assert_eq!(roles, ["system", "user", "assistant", "user"]);
        assert_eq!(rendered[0].text(), "You are second");
        assert_eq!(rendered[1].text(), "first");
        assert_eq!(rendered[3].text(), "Answer second");
    }

    #[test]
    fn load_state_restores_fields_and_fires_per_field() {
        let (callback, log) = recording_callback();
        let mut operator = LLMCallOperator::new("op").with_callback(callback);
        operator
            .load_state(json!({
                "system_prompt": [{"role": "system", "content": "restored"}],
                "user_prompt": "ask",
                "model": "m1",
                "temperature": 0.1
            }))
            .unwrap();
        assert_eq!(operator.system_prompt()[0].text(), "restored");
        assert_eq!(operator.user_prompt(), "ask");
        assert_eq!(operator.model(), Some("m1"));
        assert_eq!(operator.temperature(), Some(0.1));
        let targets: Vec<String> = log
            .lock()
            .iter()
            .map(|(target, _)| target.clone())
            .collect();
        assert_eq!(
            targets,
            ["system_prompt", "user_prompt", "model", "temperature"]
        );
    }

    #[test]
    fn state_round_trips_through_serde() {
        let operator = operator()
            .with_model("m2")
            .with_temperature(0.25)
            .with_freeze_system_prompt(true);
        let state = operator.get_state();
        let mut restored = LLMCallOperator::new("op");
        restored
            .load_state(serde_json::from_value(state).unwrap())
            .unwrap();
        assert_eq!(restored.get_state(), operator.get_state());
        assert_eq!(restored.system_prompt(), operator.system_prompt());
    }

    #[test]
    fn invalid_values_are_skipped_not_stored() {
        let mut operator = operator().with_freeze_user_prompt(false);
        let result = operator.set_parameter("temperature", json!("hot")).unwrap();
        assert!(!result.applied);
        assert_eq!(operator.temperature(), None);
        let result = operator.set_parameter("unknown", json!(1)).unwrap();
        assert!(!result.applied);
        assert!(result.reason.contains("unknown target"));
    }
}
