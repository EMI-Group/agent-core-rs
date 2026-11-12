//! Configuration of the `ReAct` agent.
//!
//! Ports `ReActAgentConfig` from `openjiuwen/core/single_agent/agents/react_agent.py`.
//! Field names and defaults match the Python dataclass; the `configure_*`
//! mutators mirror Python's chained setters and keep the type a pure data
//! holder (the loop lives in [`super`]).
use openjiuwen_common::security::SecretString;
use openjiuwen_context::ContextConfig;
use openjiuwen_foundation::{BaseMessage, ModelClientConfig, ModelRequestConfig};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// What the loop does when it exhausts [`ReActAgentConfig::max_iterations`].
///
/// Wire names are `error`, `final_answer`, and `return_observations`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaxIterationsPolicy {
    /// Fail with a `result_type: "error"` output (Python default).
    #[default]
    Error,
    /// Make one extra tool-free model call and answer with its text.
    FinalAnswer,
    /// Return an answer output carrying the observations gathered so far.
    ReturnObservations,
}

/// Configuration of one `ReActAgent`.
///
/// Construct it with [`ReActAgentConfig::new`] (identical to
/// [`Default`]) and adjust it through the builder-style `configure_*` methods.
/// Every field is serde-serializable under its Python name; `model_config`
/// corresponds to Python's `model_config_obj` field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ReActAgentConfig {
    /// Provider-facing model name; empty means "not configured yet".
    pub model_name: String,
    /// Provider name; `"openai"` by default (Python parity).
    pub model_provider: String,
    /// API key; `None` means "resolve from the environment".
    pub api_key: Option<SecretString>,
    /// Provider base URL; `None` means "use the provider default".
    pub api_base: Option<String>,
    /// Extra request headers merged into every model call.
    pub custom_headers: BTreeMap<String, String>,
    /// Name of a registered prompt template; the loop ignores it when empty.
    pub prompt_template_name: String,
    /// System/user prompt messages; Python's `List[{"role", "content"}]`.
    ///
    /// Rendered system roles feed the identity prompt section.
    pub prompt_template: Vec<BaseMessage>,
    /// Maximum model/tool iterations in one invocation (Python `5`).
    pub max_iterations: usize,
    /// Host-imposed hard limit that may lower [`max_iterations`](Self::max_iterations).
    pub max_iterations_hard_limit: Option<usize>,
    /// Behaviour when the iteration budget is exhausted.
    pub on_max_iterations: MaxIterationsPolicy,
    /// Transport-level client configuration; `None` means "derive from the
    /// top-level model fields".
    pub model_client_config: Option<ModelClientConfig>,
    /// Request-level model configuration (Python `model_config_obj`).
    pub model_config: Option<ModelRequestConfig>,
    /// Context-window configuration for the per-run [`ContextEngine`].
    ///
    /// [`ContextEngine`]: openjiuwen_context::ContextEngine
    pub context_engine_config: ContextConfig,
    /// Memory scope the agent recalls from and writes to; empty disables it.
    pub mem_scope_id: String,
    /// Whether independent tool calls may run concurrently (Python `true`).
    pub parallel_tool_calls: bool,
    /// Whether incomplete tool facts survive a cancelled run (Python `false`).
    pub retain_tool_facts_on_abort: bool,
    /// Token budget across a whole run; `None` disables the budget.
    ///
    /// Extension over Python: it bounds the total tokens a run may consume.
    pub max_total_tokens: Option<usize>,
}

impl Default for ReActAgentConfig {
    /// Python defaults: provider `"openai"`, `5` iterations,
    /// `on_max_iterations = Error`, parallel tool calls enabled, and no model,
    /// prompt, memory scope, or token budget configured.
    fn default() -> Self {
        Self {
            model_name: String::new(),
            model_provider: "openai".to_owned(),
            api_key: None,
            api_base: None,
            custom_headers: BTreeMap::new(),
            prompt_template_name: String::new(),
            prompt_template: Vec::new(),
            max_iterations: 5,
            max_iterations_hard_limit: None,
            on_max_iterations: MaxIterationsPolicy::Error,
            model_client_config: None,
            model_config: None,
            context_engine_config: ContextConfig::default(),
            mem_scope_id: String::new(),
            parallel_tool_calls: true,
            retain_tool_facts_on_abort: false,
            max_total_tokens: None,
        }
    }
}

impl ReActAgentConfig {
    /// Creates a configuration with the Python defaults; same as [`Default`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the model name and provider.
    pub fn configure_model(
        &mut self,
        model_name: impl Into<String>,
        model_provider: impl Into<String>,
    ) -> &mut Self {
        self.model_name = model_name.into();
        self.model_provider = model_provider.into();
        self
    }

    /// Sets the provider name only.
    pub fn configure_model_provider(&mut self, provider: impl Into<String>) -> &mut Self {
        self.model_provider = provider.into();
        self
    }

    /// Replaces the prompt messages.
    pub fn configure_prompt(&mut self, prompt: Vec<BaseMessage>) -> &mut Self {
        self.prompt_template = prompt;
        self
    }

    /// Replaces the prompt template; same field as
    /// [`configure_prompt`](Self::configure_prompt).
    pub fn configure_prompt_template(&mut self, template: Vec<BaseMessage>) -> &mut Self {
        self.prompt_template = template;
        self
    }

    /// Sets the iteration budget.
    pub fn configure_max_iterations(&mut self, max_iterations: usize) -> &mut Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Sets both the transport and the request model configurations.
    pub fn configure_model_client(
        &mut self,
        client: ModelClientConfig,
        request: ModelRequestConfig,
    ) -> &mut Self {
        self.model_client_config = Some(client);
        self.model_config = Some(request);
        self
    }

    /// Enables or disables concurrent tool calls.
    pub fn configure_parallel_tool_calls(&mut self, parallel: bool) -> &mut Self {
        self.parallel_tool_calls = parallel;
        self
    }

    /// Sets the context-window configuration.
    pub fn configure_context_engine(&mut self, config: ContextConfig) -> &mut Self {
        self.context_engine_config = config;
        self
    }

    /// Sets the memory scope.
    pub fn configure_mem_scope(&mut self, mem_scope_id: impl Into<String>) -> &mut Self {
        self.mem_scope_id = mem_scope_id.into();
        self
    }

    /// Replaces the extra request headers.
    pub fn configure_custom_headers(&mut self, headers: BTreeMap<String, String>) -> &mut Self {
        self.custom_headers = headers;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_python() {
        let config = ReActAgentConfig::new();
        assert_eq!(config.model_name, "");
        assert_eq!(config.model_provider, "openai");
        assert_eq!(config.api_key, None);
        assert_eq!(config.api_base, None);
        assert!(config.custom_headers.is_empty());
        assert_eq!(config.prompt_template_name, "");
        assert_eq!(config.prompt_template, Vec::<BaseMessage>::new());
        assert_eq!(config.max_iterations, 5);
        assert_eq!(config.max_iterations_hard_limit, None);
        assert_eq!(config.on_max_iterations, MaxIterationsPolicy::Error);
        assert_eq!(config.model_client_config, None);
        assert_eq!(config.model_config, None);
        assert_eq!(config.context_engine_config, ContextConfig::default());
        assert_eq!(config.mem_scope_id, "");
        assert!(config.parallel_tool_calls);
        assert!(!config.retain_tool_facts_on_abort);
        assert_eq!(config.max_total_tokens, None);
    }

    #[test]
    fn new_equals_default() {
        assert_eq!(ReActAgentConfig::new(), ReActAgentConfig::default());
    }

    #[test]
    fn max_iterations_policy_uses_python_wire_names() {
        assert_eq!(
            serde_json::to_value(MaxIterationsPolicy::Error).unwrap(),
            serde_json::json!("error")
        );
        assert_eq!(
            serde_json::to_value(MaxIterationsPolicy::FinalAnswer).unwrap(),
            serde_json::json!("final_answer")
        );
        assert_eq!(
            serde_json::to_value(MaxIterationsPolicy::ReturnObservations).unwrap(),
            serde_json::json!("return_observations")
        );
    }

    #[test]
    fn builders_mutate_the_expected_fields() {
        let mut config = ReActAgentConfig::new();
        config
            .configure_model("gpt-4o", "openai")
            .configure_prompt(vec![BaseMessage::system("be terse")])
            .configure_max_iterations(9)
            .configure_parallel_tool_calls(false)
            .configure_mem_scope("scope")
            .configure_custom_headers(BTreeMap::from([("x".to_owned(), "y".to_owned())]));
        config.configure_model_provider("anthropic");
        config.configure_prompt_template(vec![BaseMessage::user("hi")]);
        assert_eq!(config.model_name, "gpt-4o");
        assert_eq!(config.model_provider, "anthropic");
        assert_eq!(config.prompt_template.len(), 1);
        assert_eq!(config.max_iterations, 9);
        assert!(!config.parallel_tool_calls);
        assert_eq!(config.mem_scope_id, "scope");
        assert_eq!(config.custom_headers["x"], "y");
    }

    #[test]
    fn unknown_fields_are_tolerated() {
        let config: ReActAgentConfig =
            serde_json::from_value(serde_json::json!({"max_iterations": 3, "future": true}))
                .unwrap();
        assert_eq!(config.max_iterations, 3);
        assert_eq!(config.model_provider, "openai");
    }
}
