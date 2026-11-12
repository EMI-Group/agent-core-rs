//! The bridge from the platform DSL into `openjiuwen-agent`.
//!
//! Ports the executor stage of `openjiuwen.dev_tools.agent_builder`: a
//! [`LlmAgentDsl`] document becomes an [`AgentCard`], an [`LlmAgentConfig`], and
//! — when a model is supplied — a runnable [`LlmAgent`].
//!
//! The mapping is deliberately total and infallible for the card and the
//! configuration: absent DSL fields fall back to the documented defaults
//! (`"openai"` for the provider, [`ConstrainConfig::default`] for the budgets).
//! Only building the agent itself can fail.
//!
//! Provider and constraint values live in [`LlmAgentDsl::extra`] (written there
//! by [`crate::agent_builder::spec::LlmAgentSpec::to_dsl`]); the bridge reads
//! them back under their Python key names.

use std::sync::Arc;

use openjiuwen_agent::{
    AgentCard, ConstrainConfig, LlmAgent, LlmAgentConfig, ModelConfig, ModelInfo,
};
use openjiuwen_common::error::Result;
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::BaseMessage;
use serde_json::Value;

use super::dsl::LlmAgentDsl;

/// The provider used when neither the typed model block nor the DSL extras name
/// one.
const DEFAULT_PROVIDER: &str = "openai";

impl LlmAgentDsl {
    /// Builds the agent card described by the document.
    ///
    /// The card takes its name from [`LlmAgentDsl::name`], its id from
    /// [`LlmAgentDsl::agent_id`], and its description from
    /// [`LlmAgentDsl::description`].
    #[must_use]
    pub fn to_agent_card(&self) -> AgentCard {
        AgentCard::new(self.name.clone())
            .with_id(self.agent_id.clone())
            .with_description(self.description.clone())
    }

    /// Builds the [`LlmAgentConfig`] described by the document.
    ///
    /// Mapping rules:
    ///
    /// - `id`, `name`, and `description` are copied verbatim;
    /// - `prompt_template` holds one [`BaseMessage::system`] message when
    ///   [`DslConfigs::system_prompt`] is non-empty, and is empty otherwise;
    /// - `model` mirrors the DSL model block; the provider is the
    ///   `extra["model_provider"]` string when present and non-empty, and
    ///   `"openai"` otherwise;
    /// - `constrain` starts from [`ConstrainConfig::default`] and is overridden
    ///   by the numeric `extra["constraints"]["max_iterations"]` and
    ///   `extra["constraints"]["reserved_max_chat_rounds"]` values, when present.
    ///
    /// Non-numeric or out-of-range constraint values are ignored rather than
    /// rejected: a platform document is not allowed to make the conversion fail.
    ///
    /// [`DslConfigs::system_prompt`]: crate::agent_builder::dsl::DslConfigs::system_prompt
    #[must_use]
    pub fn to_llm_agent_config(&self) -> LlmAgentConfig {
        let mut config = LlmAgentConfig::new(self.agent_id.clone());
        config.name.clone_from(&self.name);
        config.description.clone_from(&self.description);
        config.prompt_template = self.system_prompt_messages();
        config.model = self.model_block();
        config.constrain = self.constrain();
        config
    }

    /// Builds a runnable [`LlmAgent`] from the document.
    ///
    /// When `model` is supplied it replaces whatever model the configuration
    /// derived from the DSL — the document's `model_info` only describes the
    /// endpoint, it never constructs a client. Passing `None` leaves the agent
    /// without a model until a host installs one via
    /// [`ReActAgent::set_model`](openjiuwen_agent::ReActAgent::set_model).
    ///
    /// # Errors
    ///
    /// Returns the error of [`LlmAgent::new`] when the composed `ReAct`
    /// configuration cannot be built.
    pub fn build_llm_agent(&self, model: Option<Arc<dyn Model>>) -> Result<LlmAgent> {
        let agent = LlmAgent::new(self.to_llm_agent_config())?;
        if let Some(model) = model {
            agent.react().set_model(model);
        }
        Ok(agent)
    }

    /// The prompt template derived from [`DslConfigs::system_prompt`].
    ///
    /// [`DslConfigs::system_prompt`]: crate::agent_builder::dsl::DslConfigs::system_prompt
    fn system_prompt_messages(&self) -> Vec<BaseMessage> {
        if self.configs.system_prompt.is_empty() {
            Vec::new()
        } else {
            vec![BaseMessage::system(self.configs.system_prompt.clone())]
        }
    }

    /// The model block derived from [`LlmAgentDsl::model`].
    fn model_block(&self) -> Option<ModelConfig> {
        let model = self.model.as_ref()?;
        Some(ModelConfig {
            model_provider: self.provider_name(),
            model_info: ModelInfo {
                api_base: model.model_info.api_base.clone(),
                model_name: model.model_info.model_name.clone(),
                temperature: model.model_info.temperature,
                top_p: model.model_info.top_p,
                ..ModelInfo::default()
            },
        })
    }

    /// The provider name taken from `extra["model_provider"]`, or the fallback.
    fn provider_name(&self) -> String {
        self.extra
            .get("model_provider")
            .and_then(Value::as_str)
            .filter(|provider| !provider.is_empty())
            .unwrap_or(DEFAULT_PROVIDER)
            .to_owned()
    }

    /// The budgets taken from `extra["constraints"]`, or the defaults.
    fn constrain(&self) -> ConstrainConfig {
        let mut constrain = ConstrainConfig::default();
        let Some(Value::Object(limits)) = self.extra.get("constraints") else {
            return constrain;
        };
        if let Some(max_iteration) = numeric_limit(limits.get("max_iterations")) {
            constrain.max_iteration = max_iteration;
        }
        if let Some(reserved_max_chat_rounds) =
            numeric_limit(limits.get("reserved_max_chat_rounds"))
        {
            constrain.reserved_max_chat_rounds = reserved_max_chat_rounds;
        }
        constrain
    }
}

/// Converts a JSON number into a `usize`, ignoring anything that does not fit.
fn numeric_limit(value: Option<&Value>) -> Option<usize> {
    value
        .and_then(Value::as_u64)
        .and_then(|limit| usize::try_from(limit).ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_agent::Agent;
    use openjiuwen_test_support::ScriptedModel;

    /// A document that populates the fields the bridge reads.
    fn sample() -> LlmAgentDsl {
        let mut dsl = LlmAgentDsl::from_json_str(
            r#"{
                "agent_id": "agent-1",
                "agent_type": "react",
                "name": "Demo Agent",
                "description": "a demo",
                "configs": {
                    "system_prompt": "You are helpful.",
                    "opening_remarks": "Hi!"
                },
                "model": {
                    "model_info": {
                        "api_base": "https://example.test/v1",
                        "model_name": "demo-model",
                        "temperature": 0.5,
                        "top_p": 0.9
                    }
                },
                "model_provider": "anthropic",
                "constraints": {
                    "max_iterations": 7,
                    "reserved_max_chat_rounds": 3
                }
            }"#,
        )
        .expect("valid document");
        dsl.create_time = 1_000;
        dsl.update_time = 1_000;
        dsl
    }

    #[test]
    fn to_agent_card_maps_identity_and_description() {
        let card = sample().to_agent_card();
        assert_eq!(card.id, "agent-1");
        assert_eq!(card.name, "Demo Agent");
        assert_eq!(card.description, "a demo");

        let empty = LlmAgentDsl::default().to_agent_card();
        assert_eq!(empty.id, "");
        assert_eq!(empty.name, "");
        assert_eq!(empty.description, "");
    }

    #[test]
    fn to_llm_agent_config_maps_identity_and_description() {
        let config = sample().to_llm_agent_config();
        assert_eq!(config.id, "agent-1");
        assert_eq!(config.name, "Demo Agent");
        assert_eq!(config.description, "a demo");
    }

    #[test]
    fn prompt_template_holds_one_system_message() {
        let config = sample().to_llm_agent_config();
        assert_eq!(config.prompt_template.len(), 1);
        assert_eq!(config.prompt_template[0].text(), "You are helpful.");
    }

    #[test]
    fn empty_system_prompt_yields_empty_template() {
        let config = LlmAgentDsl::default().to_llm_agent_config();
        assert_eq!(config.prompt_template, Vec::<BaseMessage>::new());
    }

    #[test]
    fn model_block_falls_back_to_openai_provider() {
        let dsl = LlmAgentDsl::from_json_str(
            r#"{
                "agent_id": "agent-1",
                "model": {"model_info": {"model_name": "demo-model"}}
            }"#,
        )
        .expect("valid document");
        let config = dsl.to_llm_agent_config();
        let model = config.model.expect("model block is present");
        assert_eq!(model.model_provider, "openai");
        assert_eq!(model.model_info.model_name, "demo-model");
        assert_eq!(model.model_info.api_base, "");
        assert!(model.model_info.temperature.is_none());
        assert!(model.model_info.top_p.is_none());
        assert!(!model.model_info.streaming);
    }

    #[test]
    fn model_block_honors_extra_provider_and_sampling_settings() {
        let config = sample().to_llm_agent_config();
        let model = config.model.expect("model block is present");

        assert_eq!(model.model_provider, "anthropic");
        assert_eq!(model.model_info.api_base, "https://example.test/v1");
        assert_eq!(model.model_info.model_name, "demo-model");
        assert_eq!(model.model_info.temperature, Some(0.5));
        assert_eq!(model.model_info.top_p, Some(0.9));
        assert!(!model.model_info.streaming);
        assert!(model.model_info.api_key.is_none());
        assert!(model.model_info.timeout.is_none());
        assert!(model.model_info.custom_headers.is_empty());
    }

    #[test]
    fn empty_extra_provider_uses_the_fallback() {
        let dsl = LlmAgentDsl::from_json_str(r#"{"agent_id": "a", "model_provider": ""}"#)
            .expect("valid document");
        assert_eq!(dsl.provider_name(), "openai");
    }

    #[test]
    fn absent_model_yields_no_model_config() {
        let config = LlmAgentDsl::from_json_str(r#"{"agent_id": "a"}"#)
            .expect("valid document")
            .to_llm_agent_config();
        assert!(config.model.is_none());
    }

    #[test]
    fn constraints_extra_overrides_the_defaults() {
        let config = sample().to_llm_agent_config();
        assert_eq!(config.constrain.max_iteration, 7);
        assert_eq!(config.constrain.reserved_max_chat_rounds, 3);
    }

    #[test]
    fn constraints_fall_back_to_the_defaults() {
        let config = LlmAgentDsl::default().to_llm_agent_config();
        assert_eq!(config.constrain, ConstrainConfig::default());
        assert_eq!(config.constrain.max_iteration, 5);
        assert_eq!(config.constrain.reserved_max_chat_rounds, 10);
    }

    #[test]
    fn non_numeric_constraints_are_ignored() {
        for constraints in [
            serde_json::json!({"max_iterations": "seven"}),
            serde_json::json!({"max_iterations": -1, "reserved_max_chat_rounds": 1.5}),
            serde_json::json!([1, 2]),
            serde_json::json!(null),
        ] {
            let mut dsl = LlmAgentDsl::default();
            dsl.extra
                .insert("constraints".to_string(), constraints.clone());
            let config = dsl.to_llm_agent_config();
            assert_eq!(
                config.constrain,
                ConstrainConfig::default(),
                "constraints {constraints} must be ignored"
            );
        }
    }

    #[test]
    fn build_llm_agent_without_a_model_leaves_the_agent_model_less() {
        let agent = sample().build_llm_agent(None).expect("agent builds");
        assert!(agent.react().model().is_err());
        assert_eq!(agent.card().id, "agent-1");
    }

    #[test]
    fn build_llm_agent_installs_the_supplied_model() {
        let model: Arc<dyn Model> = Arc::new(ScriptedModel::text("hi"));
        let agent = sample().build_llm_agent(Some(model)).expect("agent builds");

        let installed = agent.react().model().expect("model is installed");
        assert_eq!(installed.provider(), "scripted");
        assert_eq!(installed.model_name(), "scripted-model");
        assert_eq!(agent.card().name, "Demo Agent");
    }

    #[test]
    fn built_agent_card_matches_the_document() {
        let dsl = sample();
        let agent = dsl.build_llm_agent(None).expect("agent builds");
        let card = agent.card();
        assert_eq!(card.id, dsl.agent_id);
        assert_eq!(card.name, dsl.name);
        assert_eq!(card.description, dsl.description);
    }
}
