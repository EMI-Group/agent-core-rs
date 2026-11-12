//! LLM-agent configuration and composition over [`ReActAgent`].
//!
//! Routing table for this module:
//!
//! - [`ConstrainConfig`] — chat-round and iteration limits.
//! - [`ModelInfo`] / [`ModelConfig`] — provider-facing model settings.
//! - [`LlmAgentConfig`] / [`create_llm_agent_config`] — configuration surface.
//! - [`LlmAgent`] — the [`ReActAgent`] composition implementing [`Agent`].
//!
//! Python parity: `openjiuwen/core/application/llm_agent/`. `LLMAgent` is a
//! thin configured [`ReActAgent`] (there is no second loop); the mapping follows
//! Python's `mock_llm_agent`: `constrain.max_iteration` becomes the `ReAct`
//! iteration budget and `constrain.reserved_max_chat_rounds` becomes the context
//! window's default round count.
use crate::agent::Agent;
use crate::card::AgentCard;
use crate::output::{AgentChunk, AgentOutput};
use crate::react::{ReActAgent, ReActAgentConfig};
use futures::stream::BoxStream;
use openjiuwen_common::error::Result;
use openjiuwen_common::security::SecretString;
use openjiuwen_foundation::{BaseMessage, ModelClientConfig, ModelRequestConfig};
use openjiuwen_memory::AgentMemoryConfig;
use openjiuwen_session::Session;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

/// Budget limits applied to an LLM agent.
///
/// The agent keeps at most `reserved_max_chat_rounds` conversation rounds in its
/// context window and allows up to `max_iteration` `ReAct` iterations per
/// invocation; both defaults match the Python `ConstrainConfig`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ConstrainConfig {
    /// Conversation rounds reserved in the context window; defaults to 10.
    pub reserved_max_chat_rounds: usize,
    /// Maximum `ReAct` iterations per invocation; defaults to 5.
    pub max_iteration: usize,
}
impl Default for ConstrainConfig {
    fn default() -> Self {
        Self {
            reserved_max_chat_rounds: 10,
            max_iteration: 5,
        }
    }
}
impl ConstrainConfig {
    /// Creates the default constraint budget (10 rounds, 5 iterations).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Provider-facing settings of one model.
///
/// Field names match the Python `ModelInfo`; `api_base` uses `""` to mean "use
/// the provider default" and `streaming` selects streamed model calls.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ModelInfo {
    /// Provider base URL; empty means "use the provider default".
    pub api_base: String,
    /// API key; `None` means "resolve from the environment".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<SecretString>,
    /// Provider-facing model name.
    pub model_name: String,
    /// Sampling temperature, when set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus-sampling probability mass, when set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Whether model calls should stream.
    pub streaming: bool,
    /// Request timeout, when set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<std::time::Duration>,
    /// Extra request headers merged into every model call.
    pub custom_headers: BTreeMap<String, String>,
}

/// A provider name paired with its model settings.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ModelConfig {
    /// Provider name; empty falls back to `"openai"`.
    pub model_provider: String,
    /// Settings of the selected model.
    pub model_info: ModelInfo,
}

/// Configuration of one [`LlmAgent`].
///
/// `agent_memory_config` is retained here only: [`ReActAgentConfig`] has no
/// memory-toggle field, so the value is exposed to hosts but does not change the
/// composed `ReAct` configuration.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct LlmAgentConfig {
    /// Unique identifier of the agent card.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Free-form description.
    pub description: String,
    /// Version label.
    pub version: String,
    /// System/user prompt messages.
    pub prompt_template: Vec<BaseMessage>,
    /// Chat-round and iteration limits.
    pub constrain: ConstrainConfig,
    /// Model settings; `None` leaves the agent without a model.
    pub model: Option<ModelConfig>,
    /// Memory scope the agent recalls from and writes to.
    pub memory_scope_id: String,
    /// Per-agent memory toggles (host-visible only, see the type docs).
    pub agent_memory_config: AgentMemoryConfig,
}

impl LlmAgentConfig {
    /// Creates a configuration for the agent `id` with every other field at its
    /// default.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            ..Self::default()
        }
    }

    /// Sets the model configuration.
    #[must_use]
    pub fn with_model(mut self, model: ModelConfig) -> Self {
        self.model = Some(model);
        self
    }

    /// Replaces the prompt messages.
    #[must_use]
    pub fn with_prompt_template(mut self, template: Vec<BaseMessage>) -> Self {
        self.prompt_template = template;
        self
    }

    /// Replaces the budget constraints.
    #[must_use]
    pub fn with_constrain(mut self, constrain: ConstrainConfig) -> Self {
        self.constrain = constrain;
        self
    }
}

/// Builds an [`LlmAgentConfig`] for `id` with `model` and the default budgets.
#[must_use]
pub fn create_llm_agent_config(id: impl Into<String>, model: ModelConfig) -> LlmAgentConfig {
    LlmAgentConfig::new(id).with_model(model)
}

/// An LLM agent: a [`ReActAgent`] configured from an [`LlmAgentConfig`].
///
/// `LLMAgent` owns no loop of its own; it forwards [`Agent::invoke`] and
/// [`Agent::stream`] to the composed `ReAct` agent and filters a completed output
/// down to Python's `{output, result_type}` shape, passing interrupts through
/// unchanged.
pub struct LlmAgent {
    card: AgentCard,
    react: Arc<ReActAgent>,
}

impl LlmAgent {
    /// Builds the composed `ReAct` agent from `config`.
    ///
    /// # Errors
    ///
    /// Fails when the composed `ReAct` configuration cannot be built.
    pub fn new(config: LlmAgentConfig) -> Result<Self> {
        let LlmAgentConfig {
            id,
            name,
            description,
            prompt_template,
            constrain,
            model,
            memory_scope_id,
            ..
        } = config;
        let card = AgentCard::new(name)
            .with_id(id)
            .with_description(description);

        let mut react_config = ReActAgentConfig::new();
        react_config.prompt_template = prompt_template;
        react_config.max_iterations = constrain.max_iteration;
        react_config.mem_scope_id = memory_scope_id;
        react_config.context_engine_config.default_window_rounds =
            Some(constrain.reserved_max_chat_rounds);

        if let Some(ModelConfig {
            model_provider,
            model_info,
        }) = model
        {
            let provider = if model_provider.is_empty() {
                "openai".to_owned()
            } else {
                model_provider
            };
            let ModelInfo {
                api_base,
                api_key,
                model_name,
                temperature,
                top_p,
                streaming,
                timeout,
                custom_headers,
            } = model_info;
            react_config.model_name.clone_from(&model_name);
            react_config.model_provider.clone_from(&provider);
            react_config.api_base = (!api_base.is_empty()).then(|| api_base.clone());
            react_config.api_key.clone_from(&api_key);
            react_config.custom_headers.clone_from(&custom_headers);

            let mut client = ModelClientConfig::new(provider);
            if !api_base.is_empty() {
                client = client.with_api_base(api_base);
            }
            if let Some(key) = api_key {
                client = client.with_secret(key);
            }
            if let Some(timeout) = timeout {
                client = client.with_timeout(timeout);
            }
            client = client.with_headers(custom_headers);
            react_config.model_client_config = Some(client);

            let mut request = ModelRequestConfig::new(model_name).with_stream(streaming);
            request.temperature = temperature;
            request.top_p = top_p;
            react_config.model_config = Some(request);
        }

        let react = Arc::new(ReActAgent::new(card.clone()));
        react.configure(react_config);
        Ok(Self { card, react })
    }

    /// Replaces the prompt template and rebuilds the identity prompt section.
    pub fn set_prompt_template(&self, template: Vec<BaseMessage>) {
        let mut config = self.react.config();
        config.prompt_template = template;
        self.react.configure(config);
    }

    /// Returns the composed `ReAct` agent.
    #[must_use]
    pub fn react(&self) -> &Arc<ReActAgent> {
        &self.react
    }

    /// Returns a copy of the composed `ReAct` configuration.
    #[must_use]
    pub fn config(&self) -> ReActAgentConfig {
        self.react.config()
    }
}

// The composed ReAct agent holds trait objects and lock guards, so the debug
// view summarizes its runtime state instead of printing their internals.
#[allow(clippy::missing_fields_in_debug)]
impl fmt::Debug for LlmAgent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LlmAgent")
            .field("card_id", &self.card.id)
            .field("tools", &self.react.tools().len())
            .finish()
    }
}

#[async_trait::async_trait]
impl Agent for LlmAgent {
    fn card(&self) -> &AgentCard {
        &self.card
    }

    async fn invoke(
        &self,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> Result<AgentOutput> {
        let output = self.react.invoke(inputs, session).await?;
        if output.is_interrupt() {
            return Ok(output);
        }
        Ok(AgentOutput {
            output: output.output,
            result_type: output.result_type,
            ..AgentOutput::default()
        })
    }

    fn stream(
        &self,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> BoxStream<'static, Result<AgentChunk>> {
        self.react.stream(inputs, session)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    fn model_config(provider: &str, api_base: &str) -> ModelConfig {
        ModelConfig {
            model_provider: provider.to_owned(),
            model_info: ModelInfo {
                api_base: api_base.to_owned(),
                model_name: "test-model".to_owned(),
                streaming: true,
                ..ModelInfo::default()
            },
        }
    }

    #[test]
    fn defaults_match_python() {
        let constrain = ConstrainConfig::default();
        assert_eq!(constrain.reserved_max_chat_rounds, 10);
        assert_eq!(constrain.max_iteration, 5);
        assert_eq!(ConstrainConfig::new(), constrain);
    }

    #[test]
    fn json_round_trip() {
        let constrain = ConstrainConfig {
            reserved_max_chat_rounds: 3,
            max_iteration: 2,
        };
        let value = serde_json::to_value(&constrain).unwrap();
        let restored: ConstrainConfig = serde_json::from_value(value).unwrap();
        assert_eq!(restored, constrain);
    }

    #[test]
    fn unknown_keys_and_missing_fields_are_tolerated() {
        let constrain: ConstrainConfig =
            serde_json::from_value(serde_json::json!({"max_iteration": 7, "surprise": true}))
                .unwrap();
        assert_eq!(constrain.max_iteration, 7);
        assert_eq!(constrain.reserved_max_chat_rounds, 10);
    }

    #[test]
    fn create_llm_agent_config_sets_the_model() {
        let config = create_llm_agent_config("agent-1", model_config("openai", ""));
        assert_eq!(config.id, "agent-1");
        assert_eq!(config.model, Some(model_config("openai", "")));
    }

    #[test]
    fn new_maps_config_onto_the_react_agent() {
        let config = LlmAgentConfig::new("agent-1")
            .with_model(model_config("anthropic", ""))
            .with_constrain(ConstrainConfig {
                reserved_max_chat_rounds: 4,
                max_iteration: 3,
            });
        let config = LlmAgentConfig {
            name: "Helper".to_owned(),
            ..config
        };
        let agent = LlmAgent::new(config).unwrap();
        assert_eq!(agent.card().id, "agent-1");
        assert_eq!(agent.card().name, "Helper");
        let react = agent.react().config();
        assert_eq!(react.max_iterations, 3);
        assert_eq!(react.context_engine_config.default_window_rounds, Some(4));
        assert_eq!(react.model_name, "test-model");
        assert_eq!(react.model_provider, "anthropic");
        // An empty api_base maps to "use the provider default".
        assert_eq!(react.api_base, None);
        assert!(react.model_config.unwrap().stream);
    }

    #[test]
    fn empty_provider_falls_back_to_openai() {
        let config = LlmAgentConfig::new("a").with_model(model_config("", ""));
        let agent = LlmAgent::new(config).unwrap();
        assert_eq!(agent.react().config().model_provider, "openai");
    }

    #[test]
    fn non_empty_api_base_is_kept() {
        let config = LlmAgentConfig::new("a").with_model(model_config("openai", "http://x"));
        let agent = LlmAgent::new(config).unwrap();
        assert_eq!(agent.react().config().api_base.as_deref(), Some("http://x"));
    }

    #[test]
    fn set_prompt_template_rebuilds_the_builder() {
        let agent = LlmAgent::new(LlmAgentConfig::new("a")).unwrap();
        agent.set_prompt_template(vec![BaseMessage::system("be terse")]);
        assert_eq!(agent.react().prompt_builder().build(), "be terse");
        assert_eq!(agent.config().prompt_template.len(), 1);
    }

    #[tokio::test]
    async fn invoke_without_a_model_reports_a_config_error() {
        let agent = LlmAgent::new(LlmAgentConfig::new("a")).unwrap();
        let error = agent
            .invoke(Value::String("hi".to_owned()), None)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
    }
}
