//! [`ChatConfig`]: the typed settings a chat agent is built from.
//!
//! Ports `openjiuwen/dev_tools/tune/chat_agent/chat_config.py`. Python's
//! `ChatAgentConfig` is an `AgentConfig` (`id`, `version`, `description`) plus a
//! `model: LLMCallConfig`, and the `LLMCallConfig` in turn carries the model
//! name and the system/user prompt pair the tuning loop rewrites.
//! [`ChatConfig`] flattens both into one value: the identity fields, the model
//! name, the two prompts, the optional sampling settings, and the two freeze
//! flags that tell the agent whether a prompt is under optimisation.
//!
//! [`ChatConfig::validate`] enforces the constraints Python leaves to
//! `pydantic` (a blank agent id or model name, an out-of-range sampling setting,
//! a frozen but blank user prompt) and reports them as
//! [`Error::validation`] naming the offending field, so nothing panics on user
//! input.

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Highest sampling temperature Python's model configs accept.
const MAX_TEMPERATURE: f64 = 2.0;

/// Highest nucleus-sampling probability mass a model config accepts.
const MAX_TOP_P: f64 = 1.0;

/// The settings a [`ChatAgent`](super::ChatAgent) is built from.
///
/// Python: `openjiuwen.dev_tools.tune.chat_agent.chat_config.ChatAgentConfig`
/// (an `AgentConfig` plus a `model: LLMCallConfig`). The fields default to the
/// Python defaults — a blank identity, an explicit model name, blank prompts,
/// no sampling overrides, and no frozen prompt — so a partially populated JSON
/// payload still deserializes.
///
/// The fields are public, like [`OptimizerConfig`](crate::tune::OptimizerConfig),
/// so they double as the readers a caller needs.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ChatConfig {
    /// The agent identifier Python's `AgentConfig.id` carries.
    pub agent_id: String,
    /// The agent version Python's `AgentConfig.version` carries.
    pub version: String,
    /// The human-readable description Python's `AgentConfig.description` carries.
    pub description: String,
    /// The provider-facing model name the agent's model requests use.
    pub model_name: String,
    /// The system prompt; frozen verbatim only when
    /// [`freeze_system_prompt`](Self::freeze_system_prompt) is set.
    pub system_prompt: String,
    /// The user prompt; a `{{name}}` template over the case inputs unless it is
    /// empty or frozen.
    pub user_prompt: String,
    /// The sampling temperature, in `0.0..=2.0` when set.
    pub temperature: Option<f64>,
    /// The nucleus-sampling probability mass, in `0.0..=1.0` when set.
    pub top_p: Option<f64>,
    /// Keep [`system_prompt`](Self::system_prompt) in front of the candidate
    /// prompt instead of letting the candidate replace it.
    pub freeze_system_prompt: bool,
    /// Send [`user_prompt`](Self::user_prompt) verbatim, with no placeholder
    /// substitution.
    pub freeze_user_prompt: bool,
}

impl ChatConfig {
    /// Creates a configuration for `agent_id` and `model_name`, with every other
    /// field at its Python default.
    ///
    /// The result is deliberately unvalidated: [`validate`](Self::validate) (and
    /// therefore every `ChatAgent` constructor) rejects a blank identifier, so a
    /// test or a caller assembling a configuration field by field can still
    /// build one it is about to repair.
    #[must_use]
    pub fn new(agent_id: impl Into<String>, model_name: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            model_name: model_name.into(),
            ..Self::default()
        }
    }

    /// Replaces [`version`](Self::version).
    #[must_use]
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Replaces [`description`](Self::description).
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Replaces [`system_prompt`](Self::system_prompt).
    #[must_use]
    pub fn with_system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.system_prompt = system_prompt.into();
        self
    }

    /// Replaces [`user_prompt`](Self::user_prompt).
    #[must_use]
    pub fn with_user_prompt(mut self, user_prompt: impl Into<String>) -> Self {
        self.user_prompt = user_prompt.into();
        self
    }

    /// Sets [`temperature`](Self::temperature).
    #[must_use]
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets [`top_p`](Self::top_p).
    #[must_use]
    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Sets [`freeze_system_prompt`](Self::freeze_system_prompt).
    #[must_use]
    pub fn with_frozen_system_prompt(mut self, freeze: bool) -> Self {
        self.freeze_system_prompt = freeze;
        self
    }

    /// Sets [`freeze_user_prompt`](Self::freeze_user_prompt).
    #[must_use]
    pub fn with_frozen_user_prompt(mut self, freeze: bool) -> Self {
        self.freeze_user_prompt = freeze;
        self
    }

    /// Checks the configuration against Python's field constraints.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when
    /// [`agent_id`](Self::agent_id) or [`model_name`](Self::model_name) is blank
    /// or whitespace, when [`temperature`](Self::temperature) is set but not a
    /// finite value in `0.0..=2.0`, when [`top_p`](Self::top_p) is set but not a
    /// finite value in `0.0..=1.0`, or when
    /// [`freeze_user_prompt`](Self::freeze_user_prompt) is set while
    /// [`user_prompt`](Self::user_prompt) is blank.
    ///
    /// A frozen but blank [`system_prompt`](Self::system_prompt) is accepted: it
    /// simply means there is no frozen block to put in front of the candidate
    /// prompt.
    pub fn validate(&self) -> Result<()> {
        if self.agent_id.trim().is_empty() {
            return Err(Error::validation("ChatConfig.agent_id must not be blank"));
        }
        if self.model_name.trim().is_empty() {
            return Err(Error::validation("ChatConfig.model_name must not be blank"));
        }
        if let Some(temperature) = self.temperature {
            if !temperature.is_finite() || !(0.0..=MAX_TEMPERATURE).contains(&temperature) {
                return Err(Error::validation(format!(
                    "ChatConfig.temperature must be a finite value in 0.0..={MAX_TEMPERATURE}, \
                     got {temperature}"
                )));
            }
        }
        if let Some(top_p) = self.top_p {
            if !top_p.is_finite() || !(0.0..=MAX_TOP_P).contains(&top_p) {
                return Err(Error::validation(format!(
                    "ChatConfig.top_p must be a finite value in 0.0..={MAX_TOP_P}, got {top_p}"
                )));
            }
        }
        if self.freeze_user_prompt && self.user_prompt.trim().is_empty() {
            return Err(Error::validation(
                "ChatConfig.user_prompt must not be blank when freeze_user_prompt is set: a \
                 frozen prompt is sent verbatim, so there would be nothing to send",
            ));
        }
        Ok(())
    }
}

/// Builds a validated [`ChatConfig`], the port of Python's
/// `create_chat_agent_config(agent_id, agent_version, description, model)`.
///
/// The argument order is Python's, with the model name standing in for the
/// `LLMCallConfig` it carries. The prompts and sampling settings are applied
/// afterwards with the [`ChatConfig`] builders, which keeps this factory as
/// small as the Python one.
///
/// # Errors
///
/// Whatever [`ChatConfig::validate`] reports for the built configuration (a
/// blank `agent_id` or `model_name`).
///
/// # Examples
///
/// ```
/// # use openjiuwen_dev_tools::tune::create_chat_config;
/// let config = create_chat_config("agent-1", "1.0.0", "a chat agent", "gpt-4o-mini")?;
/// assert_eq!(config.agent_id, "agent-1");
/// assert_eq!(config.model_name, "gpt-4o-mini");
/// config.validate()?;
/// # Ok::<(), openjiuwen_common::error::Error>(())
/// ```
pub fn create_chat_config(
    agent_id: impl Into<String>,
    version: impl Into<String>,
    description: impl Into<String>,
    model_name: impl Into<String>,
) -> Result<ChatConfig> {
    let config = ChatConfig::new(agent_id, model_name)
        .with_version(version)
        .with_description(description);
    config.validate()?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// A minimal valid configuration.
    fn config() -> ChatConfig {
        ChatConfig::new("agent-1", "gpt-4o-mini")
    }

    #[test]
    fn new_starts_from_the_python_defaults() {
        let config = ChatConfig::new("agent-1", "gpt-4o-mini");
        assert_eq!(config.agent_id, "agent-1");
        assert_eq!(config.model_name, "gpt-4o-mini");
        assert_eq!(config.version, "");
        assert_eq!(config.description, "");
        assert_eq!(config.system_prompt, "");
        assert_eq!(config.user_prompt, "");
        assert_eq!(config.temperature, None);
        assert_eq!(config.top_p, None);
        assert!(!config.freeze_system_prompt);
        assert!(!config.freeze_user_prompt);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn builders_replace_every_field() {
        let config = config()
            .with_version("1.2.3")
            .with_description("answers questions")
            .with_system_prompt("be brief")
            .with_user_prompt("Q: {{question}}")
            .with_temperature(0.2)
            .with_top_p(0.9)
            .with_frozen_system_prompt(true)
            .with_frozen_user_prompt(true);
        assert_eq!(config.version, "1.2.3");
        assert_eq!(config.description, "answers questions");
        assert_eq!(config.system_prompt, "be brief");
        assert_eq!(config.user_prompt, "Q: {{question}}");
        assert_eq!(config.temperature, Some(0.2));
        assert_eq!(config.top_p, Some(0.9));
        assert!(config.freeze_system_prompt);
        assert!(config.freeze_user_prompt);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn blank_identity_fields_are_rejected() {
        let error = ChatConfig::new("", "gpt-4o-mini").validate().unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        assert!(error.to_string().contains("agent_id"), "{error}");

        let error = ChatConfig::new("   ", "gpt-4o-mini")
            .validate()
            .unwrap_err();
        assert!(error.to_string().contains("agent_id"), "{error}");

        let error = ChatConfig::new("agent-1", "").validate().unwrap_err();
        assert!(error.to_string().contains("model_name"), "{error}");

        let error = ChatConfig::new("agent-1", "\t\n").validate().unwrap_err();
        assert!(error.to_string().contains("model_name"), "{error}");
    }

    #[test]
    fn temperature_is_bounded_by_the_python_range() {
        for accepted in [0.0, 0.7, 2.0] {
            let config = config().with_temperature(accepted);
            assert!(config.validate().is_ok(), "temperature {accepted}");
        }
        for rejected in [-0.1, 2.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let error = config().with_temperature(rejected).validate().unwrap_err();
            assert!(
                error.to_string().contains("ChatConfig.temperature"),
                "{rejected}: {error}"
            );
        }
    }

    #[test]
    fn top_p_is_bounded_by_the_python_range() {
        for accepted in [0.0, 0.5, 1.0] {
            let config = config().with_top_p(accepted);
            assert!(config.validate().is_ok(), "top_p {accepted}");
        }
        for rejected in [-0.01, 1.01, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let error = config().with_top_p(rejected).validate().unwrap_err();
            assert!(
                error.to_string().contains("ChatConfig.top_p"),
                "{rejected}: {error}"
            );
        }
    }

    #[test]
    fn a_frozen_user_prompt_must_not_be_blank() {
        let error = config()
            .with_frozen_user_prompt(true)
            .validate()
            .unwrap_err();
        assert!(error.to_string().contains("freeze_user_prompt"), "{error}");

        let error = config()
            .with_user_prompt("  ")
            .with_frozen_user_prompt(true)
            .validate()
            .unwrap_err();
        assert!(error.to_string().contains("freeze_user_prompt"), "{error}");

        assert!(config()
            .with_user_prompt("{{question}}")
            .with_frozen_user_prompt(true)
            .validate()
            .is_ok());
    }

    #[test]
    fn a_frozen_system_prompt_may_stay_blank() {
        assert!(config().with_frozen_system_prompt(true).validate().is_ok());
    }

    #[test]
    fn a_frozen_user_prompt_may_stay_empty_when_not_frozen() {
        let mut config = config();
        config.freeze_user_prompt = false;
        assert_eq!(config.user_prompt, String::new());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn json_uses_the_snake_case_field_names() {
        let config = config()
            .with_version("0.1.0")
            .with_system_prompt("be brief")
            .with_user_prompt("Q: {{question}}")
            .with_temperature(0.5)
            .with_top_p(0.8)
            .with_frozen_system_prompt(true);

        let value = serde_json::to_value(&config).unwrap();
        assert_eq!(value["agent_id"], json!("agent-1"));
        assert_eq!(value["model_name"], json!("gpt-4o-mini"));
        assert_eq!(value["version"], json!("0.1.0"));
        assert_eq!(value["system_prompt"], json!("be brief"));
        assert_eq!(value["user_prompt"], json!("Q: {{question}}"));
        assert_eq!(value["temperature"], json!(0.5));
        assert_eq!(value["top_p"], json!(0.8));
        assert_eq!(value["freeze_system_prompt"], json!(true));

        let reparsed: ChatConfig = serde_json::from_value(value).unwrap();
        assert_eq!(reparsed, config);
    }

    #[test]
    fn missing_fields_default_and_unknown_fields_are_tolerated() {
        let config: ChatConfig = serde_json::from_value(json!({
            "agent_id": "agent-2",
            "model_name": "gpt-4o",
            "temperature": null,
            "future_setting": {"nested": true},
        }))
        .unwrap();
        assert_eq!(config.agent_id, "agent-2");
        assert_eq!(config.model_name, "gpt-4o");
        assert_eq!(config.version, "");
        assert_eq!(config.description, "");
        assert_eq!(config.system_prompt, "");
        assert_eq!(config.user_prompt, "");
        assert_eq!(config.temperature, None);
        assert_eq!(config.top_p, None);
        assert!(!config.freeze_system_prompt);
        assert!(!config.freeze_user_prompt);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn deserialization_stays_permissive_so_validate_reports_the_problem() {
        let config: ChatConfig = serde_json::from_value(json!({"temperature": 9.0})).unwrap();
        assert_eq!(config.temperature, Some(9.0));
        assert!(config.validate().is_err());
    }

    #[test]
    fn the_python_factory_builds_a_validated_config() {
        let config = create_chat_config("agent-1", "1.0.0", "a chat agent", "gpt-4o-mini").unwrap();
        assert_eq!(config.agent_id, "agent-1");
        assert_eq!(config.version, "1.0.0");
        assert_eq!(config.description, "a chat agent");
        assert_eq!(config.model_name, "gpt-4o-mini");
        assert!(config.validate().is_ok());
    }

    #[test]
    fn the_python_factory_rejects_a_blank_identity() {
        let error = create_chat_config("", "1.0.0", "", "gpt-4o-mini").unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
        assert!(error.to_string().contains("agent_id"), "{error}");

        let error = create_chat_config("agent-1", "1.0.0", "", " ").unwrap_err();
        assert!(error.to_string().contains("model_name"), "{error}");
    }
}
