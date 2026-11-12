//! Native harness provider: create-config parsing and model-choice mapping.
//!
//! [`NativeHarnessProvider`] is the factory the host drives: [`HarnessProvider::create`] turns a
//! JSON create mapping into an **unstarted** [`NativeHarness`] over a freshly built
//! [`DeepAgent`](crate::agent::DeepAgent), and [`ModelConfigProvider::model_config`] maps a host
//! [`ModelChoice`] onto the create fragment the host merges before calling `create`.
//!
//! # Create configuration
//!
//! Every key is optional: `model`, `provider`, `api_key`, `api_base`, `temperature` (`0.0..=2.0`),
//! `workspace`, `enable_task_loop`, `max_iterations` (positive), `system_prompt`, `permissions`
//! (a [`PermissionsSection`](crate::permission::PermissionsSection) object) and `language`.
//! Unknown keys, wrong types and blank strings are rejected with
//! [`HarnessError::Validation`](openjiuwen_harness_protocol::HarnessError::Validation).
//!
//! No model handle is supplied: [`DeepAgent::ensure_initialized`](crate::agent::DeepAgent)
//! resolves the model lazily from `model`/`provider`/`api_base`/`api_key` when the harness starts.
//! Creating a harness without a model identity therefore succeeds, while starting one fails with
//! a protocol error — intentional, so a host can create first and pick a model later.
//!
//! # Model choices
//!
//! The card declares `model_selection` and `model_discovery` but not `endpoint_models` or
//! `auth_fallback`, so an endpoint choice and a fallback choice are rejected with
//! [`HarnessError::UnsupportedCapability`](openjiuwen_harness_protocol::HarnessError::UnsupportedCapability);
//! a recognized effort (`low`/`medium`/`high`) becomes a `temperature` in the fragment.

use std::path::PathBuf;
use std::sync::Arc;

use openjiuwen_agent::AgentCard;
use openjiuwen_common::security::SecretString;
use openjiuwen_foundation::llm::ModelRequestConfig;
use openjiuwen_harness_protocol::{
    HarnessCard, HarnessError, HarnessProtocol, HarnessProvider, HarnessResult, JsonObject,
    JsonValue, ModelChoice, ModelConfigProvider,
};

use crate::agent::{DeepAgent, DeepAgentBuilder};
use crate::permission::PermissionsSection;
use crate::protocol::{
    native_card, NativeHarness, NativeHarnessOptions, HARNESS_NAME, PROVIDER_VERSION,
};
use crate::schema::DeepAgentConfig;
use crate::workspace::Workspace;

/// Message shared by every blank string option (Python `_optional_str` parity).
const ERR_BLANK_STRING: &str = "native harness string options must be non-empty strings";

/// The model name used by the card of every agent this provider creates.
const DEFAULT_AGENT_NAME: &str = "deep_agent";

/// The description of the card of every agent this provider creates.
const DEFAULT_AGENT_DESCRIPTION: &str = "DeepAgent instance";

/// Create-mapping keys the native provider understands; everything else is rejected.
const KNOWN_KEYS: [&str; 11] = [
    "api_base",
    "api_key",
    "enable_task_loop",
    "language",
    "max_iterations",
    "model",
    "permissions",
    "provider",
    "system_prompt",
    "temperature",
    "workspace",
];

/// Factory for native harness instances, driven by a JSON create mapping.
///
/// The card is built once and shared by reference with every host call; the options are applied
/// to every harness [`HarnessProvider::create`] returns.
#[derive(Debug, Clone)]
pub struct NativeHarnessProvider {
    /// Card shared by every harness this provider creates.
    card: HarnessCard,
    /// Options applied to every created harness.
    options: NativeHarnessOptions,
}

impl Default for NativeHarnessProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl NativeHarnessProvider {
    /// Builds a provider with the default harness options.
    #[must_use]
    pub fn new() -> Self {
        Self {
            card: native_card(HARNESS_NAME, PROVIDER_VERSION),
            options: NativeHarnessOptions::new(),
        }
    }

    /// Replaces the options applied to every created harness.
    #[must_use]
    pub fn with_options(mut self, options: NativeHarnessOptions) -> Self {
        self.options = options;
        self
    }
}

impl HarnessProvider for NativeHarnessProvider {
    fn card(&self) -> &HarnessCard {
        &self.card
    }

    /// Validates `config` and builds an unstarted harness over a fresh `DeepAgent`.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::Validation`] when `config` is not a valid native-harness
    /// configuration, and [`HarnessError::Protocol`] when the resulting `DeepAgent`
    /// configuration is rejected at construction time.
    fn create(&self, config: JsonObject) -> HarnessResult<Box<dyn HarnessProtocol>> {
        let config = NativeCreateConfig::parse(&config)?;
        let agent = build_agent(config.into_deep_agent_config())?;
        Ok(Box::new(NativeHarness::new(agent, self.options.clone())))
    }
}

impl ModelConfigProvider for NativeHarnessProvider {
    /// Maps `choice` onto the native create fragment.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::UnsupportedCapability`] for an endpoint-served model or an
    /// authentication fallback, and [`HarnessError::Validation`] for a blank model name or an
    /// effort the native runtime cannot map onto a temperature.
    fn model_config(
        &self,
        choice: &ModelChoice,
        fallback: Option<&ModelChoice>,
    ) -> HarnessResult<JsonObject> {
        if choice.endpoint.is_some() {
            return Err(HarnessError::unsupported_capability(
                "openjiuwen-native does not take endpoint models",
            ));
        }
        if fallback.is_some() {
            return Err(HarnessError::unsupported_capability(
                "openjiuwen-native has no authentication fallback",
            ));
        }
        let mut fragment = JsonObject::new();
        if let Some(model) = &choice.model {
            if model.trim().is_empty() {
                return Err(HarnessError::validation(ERR_BLANK_STRING));
            }
            fragment.insert("model".to_string(), JsonValue::String(model.clone()));
        }
        if let Some(effort) = &choice.effort {
            fragment.insert(
                "temperature".to_string(),
                JsonValue::from(effort_temperature(effort)?),
            );
        }
        Ok(fragment)
    }
}

/// The validated native create mapping, before it is folded onto [`DeepAgentConfig`].
///
/// Parsing is separated from application so every key is type-checked and blank-checked before
/// a `DeepAgent` (and its workspace rails) is constructed.
#[derive(Debug)]
struct NativeCreateConfig {
    /// Model name of the inner `ReAct` agent.
    model: Option<String>,
    /// Model provider name (`openai`, `anthropic`, …).
    provider: Option<String>,
    /// Model-provider credential.
    api_key: Option<SecretString>,
    /// Model-provider base URL.
    api_base: Option<String>,
    /// Sampling temperature, already validated into `0.0..=2.0`.
    temperature: Option<f32>,
    /// Workspace root; also becomes the shell working directory.
    workspace: Option<PathBuf>,
    /// Whether the outer task loop runs instead of a single `ReAct` round.
    enable_task_loop: Option<bool>,
    /// Ceiling on inner `ReAct` iterations per invocation.
    max_iterations: Option<u32>,
    /// System prompt of the inner `ReAct` agent.
    system_prompt: Option<String>,
    /// Tool permission policy.
    permissions: Option<PermissionsSection>,
    /// Prompt and workspace language.
    language: Option<String>,
}

impl NativeCreateConfig {
    /// Validates and decodes a create mapping; unknown keys are rejected first.
    fn parse(config: &JsonObject) -> HarnessResult<Self> {
        reject_unknown_keys(config)?;
        Ok(Self {
            model: optional_string(config, "model")?,
            provider: optional_string(config, "provider")?,
            api_key: optional_string(config, "api_key")?.map(SecretString::new),
            api_base: optional_string(config, "api_base")?,
            temperature: optional_temperature(config)?,
            workspace: optional_string(config, "workspace")?.map(PathBuf::from),
            enable_task_loop: optional_bool(config, "enable_task_loop")?,
            max_iterations: optional_positive_integer(config, "max_iterations")?,
            system_prompt: optional_string(config, "system_prompt")?,
            permissions: optional_permissions(config)?,
            language: optional_string(config, "language")?,
        })
    }

    /// Folds the mapping onto a [`DeepAgentConfig`], leaving unset keys at their defaults.
    ///
    /// The temperature is stored on [`ModelRequestConfig::temperature`] — the same field
    /// `set_model` writes — and a request config always carries a model name, so it is only
    /// applied when `model` is set; otherwise the temperature is dropped with a warning.
    fn into_deep_agent_config(self) -> DeepAgentConfig {
        let mut config = DeepAgentConfig::default();
        match (self.model.as_deref(), self.temperature) {
            (Some(model), Some(temperature)) => {
                config.model_config =
                    Some(ModelRequestConfig::new(model).with_temperature(temperature));
            }
            (None, Some(_)) => {
                tracing::warn!(
                    "native harness temperature is ignored: no model name is configured"
                );
            }
            _ => {}
        }
        config.model_name = self.model;
        config.model_provider = self.provider;
        config.api_key = self.api_key;
        config.api_base = self.api_base;
        if let Some(workspace) = self.workspace {
            config.workspace = Workspace::new(workspace.clone());
            config.cwd = Some(workspace);
        }
        if let Some(enable_task_loop) = self.enable_task_loop {
            config.enable_task_loop = enable_task_loop;
        }
        config.max_iterations = self.max_iterations;
        config.system_prompt = self.system_prompt;
        config.permissions = self.permissions;
        if let Some(language) = self.language {
            config.language = language;
        }
        config
    }
}

/// Builds the `DeepAgent` an unstarted harness wraps.
fn build_agent(config: DeepAgentConfig) -> HarnessResult<Arc<DeepAgent>> {
    let card = AgentCard::new(DEFAULT_AGENT_NAME).with_description(DEFAULT_AGENT_DESCRIPTION);
    DeepAgentBuilder::new(card)
        .config(config)
        .build()
        .map_err(|error| HarnessError::protocol(error.to_string()))
}

/// Rejects every create key the native provider does not understand.
///
/// `JsonObject` is a `BTreeMap`, so the offending keys are already in the alphabetical order the
/// message promises.
fn reject_unknown_keys(config: &JsonObject) -> HarnessResult<()> {
    let unknown: Vec<&str> = config
        .keys()
        .map(String::as_str)
        .filter(|key| !KNOWN_KEYS.contains(key))
        .collect();
    if unknown.is_empty() {
        return Ok(());
    }
    Err(HarnessError::validation(format!(
        "unknown native harness configuration fields: {}",
        unknown.join(", ")
    )))
}

/// Decodes an optional non-blank string option.
fn optional_string(config: &JsonObject, key: &str) -> HarnessResult<Option<String>> {
    let Some(value) = config.get(key) else {
        return Ok(None);
    };
    let text = value
        .as_str()
        .ok_or_else(|| HarnessError::validation(format!("{key} must be a string")))?;
    if text.trim().is_empty() {
        return Err(HarnessError::validation(ERR_BLANK_STRING));
    }
    Ok(Some(text.to_string()))
}

/// Decodes an optional boolean option.
fn optional_bool(config: &JsonObject, key: &str) -> HarnessResult<Option<bool>> {
    let Some(value) = config.get(key) else {
        return Ok(None);
    };
    value
        .as_bool()
        .map(Some)
        .ok_or_else(|| HarnessError::validation(format!("{key} must be a boolean")))
}

/// Decodes an optional strictly positive integer option that fits an `u32`.
fn optional_positive_integer(config: &JsonObject, key: &str) -> HarnessResult<Option<u32>> {
    let Some(value) = config.get(key) else {
        return Ok(None);
    };
    let invalid = || HarnessError::validation(format!("{key} must be a positive integer"));
    let number = value.as_u64().ok_or_else(invalid)?;
    if number == 0 {
        return Err(invalid());
    }
    u32::try_from(number).map(Some).map_err(|_| invalid())
}

/// Decodes the optional sampling temperature, which must be a finite number in `0.0..=2.0`.
// The range check runs on `f64` first, so the narrowing cast cannot change the outcome.
#[allow(clippy::cast_possible_truncation)]
fn optional_temperature(config: &JsonObject) -> HarnessResult<Option<f32>> {
    let Some(value) = config.get("temperature") else {
        return Ok(None);
    };
    let invalid = || HarnessError::validation("temperature must be a number between 0 and 2");
    let number = value.as_f64().ok_or_else(invalid)?;
    if !number.is_finite() || !(0.0..=2.0).contains(&number) {
        return Err(invalid());
    }
    Ok(Some(number as f32))
}

/// Decodes the optional permission policy, keeping the serde message on failure.
fn optional_permissions(config: &JsonObject) -> HarnessResult<Option<PermissionsSection>> {
    let Some(value) = config.get("permissions") else {
        return Ok(None);
    };
    serde_json::from_value(value.clone())
        .map(Some)
        .map_err(|error| {
            HarnessError::validation(format!(
                "permissions must be a valid permissions section: {error}"
            ))
        })
}

/// Maps a reasoning effort onto the sampling temperature the native runtime applies.
///
/// Unrecognized efforts are rejected rather than ignored, because the host selected a behavior
/// the runtime cannot reproduce.
fn effort_temperature(effort: &str) -> HarnessResult<f64> {
    match effort {
        "low" => Ok(0.2),
        "medium" => Ok(0.6),
        "high" => Ok(1.0),
        other => Err(HarnessError::validation(format!(
            "unsupported reasoning effort '{other}'"
        ))),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use std::path::PathBuf;

    use openjiuwen_harness_protocol::{
        HarnessErrorKind, HarnessProvider, JsonObject, JsonValue, ModelChoice, ModelConfigProvider,
        ModelEndpoint,
    };

    use super::*;

    /// Decodes a JSON literal into the create-mapping type.
    fn mapping(value: JsonValue) -> JsonObject {
        serde_json::from_value(value).expect("the create mapping must be a JSON object")
    }

    /// Builds a built-in (non-endpoint) model choice.
    fn choice(model: Option<&str>, effort: Option<&str>) -> ModelChoice {
        ModelChoice {
            model: model.map(str::to_string),
            effort: effort.map(str::to_string),
            endpoint: None,
        }
    }

    #[test]
    fn the_create_mapping_is_validated_and_folded() {
        let config = NativeCreateConfig::parse(&mapping(serde_json::json!({
            "model": "gpt-4o",
            "provider": "openai",
            "api_key": "test-key",
            "api_base": "https://example.invalid/v1",
            "temperature": 0.5,
            "workspace": "work",
            "enable_task_loop": true,
            "max_iterations": 3,
            "system_prompt": "You are a test agent.",
            "permissions": {"enabled": true},
            "language": "en",
        })))
        .unwrap()
        .into_deep_agent_config();
        let request = config.model_config.expect("a request config is built");
        assert_eq!(request.model_name, "gpt-4o");
        assert_eq!(request.temperature, Some(0.5));
        assert_eq!(config.model_provider.as_deref(), Some("openai"));
        // The credential is compared without ever being formatted into an assertion message.
        assert!(config
            .api_key
            .as_ref()
            .is_some_and(|key| key.expose_secret() == "test-key"));
        assert_eq!(
            config.api_base.as_deref(),
            Some("https://example.invalid/v1")
        );
        assert_eq!(
            config.system_prompt.as_deref(),
            Some("You are a test agent.")
        );
        assert_eq!(config.workspace.root_path, PathBuf::from("work"));
        assert_eq!(config.cwd, Some(PathBuf::from("work")));
        assert!(config.enable_task_loop);
        assert_eq!(config.max_iterations, Some(3));
        assert!(config.permissions.unwrap().enabled);
        assert_eq!(config.language, "en");

        let error = NativeCreateConfig::parse(&mapping(serde_json::json!({
            "model": 7,
            "zeta": 1,
            "alpha": true,
        })))
        .unwrap_err();
        assert_eq!(error.kind(), HarnessErrorKind::Validation);
        assert_eq!(
            error.message(),
            "unknown native harness configuration fields: alpha, zeta"
        );
        let blank = NativeCreateConfig::parse(&mapping(serde_json::json!({"model": " "})));
        assert_eq!(blank.unwrap_err().message(), ERR_BLANK_STRING);
        let hot = NativeCreateConfig::parse(&mapping(serde_json::json!({"temperature": 2.5})));
        assert_eq!(
            hot.unwrap_err().message(),
            "temperature must be a number between 0 and 2"
        );
        let zero = NativeCreateConfig::parse(&mapping(serde_json::json!({"max_iterations": 0})));
        assert_eq!(
            zero.unwrap_err().message(),
            "max_iterations must be a positive integer"
        );
    }

    #[test]
    fn model_config_maps_models_and_efforts() {
        let provider = NativeHarnessProvider::new();
        assert_eq!(provider.card().name, HARNESS_NAME);
        assert!(provider
            .model_config(&choice(None, None), None)
            .unwrap()
            .is_empty());

        let fragment = provider
            .model_config(&choice(Some("gpt-4o"), Some("medium")), None)
            .unwrap();
        assert_eq!(fragment.get("model"), Some(&serde_json::json!("gpt-4o")));
        for (effort, expected) in [("low", 0.2), ("medium", 0.6), ("high", 1.0)] {
            let fragment = provider
                .model_config(&choice(None, Some(effort)), None)
                .unwrap();
            assert_eq!(
                fragment.get("temperature"),
                Some(&serde_json::json!(expected)),
                "for {effort}"
            );
        }
    }

    #[test]
    fn model_config_rejects_invalid_choices() {
        let provider = NativeHarnessProvider::new();
        let error = provider
            .model_config(&choice(None, Some("extreme")), None)
            .unwrap_err();
        assert_eq!(error.kind(), HarnessErrorKind::Validation);
        assert_eq!(error.message(), "unsupported reasoning effort 'extreme'");
        let error = provider
            .model_config(&choice(Some(" "), None), None)
            .unwrap_err();
        assert_eq!(error.message(), ERR_BLANK_STRING);

        let endpoint = ModelChoice {
            model: None,
            effort: None,
            endpoint: Some(ModelEndpoint {
                api_base: Some("https://example.invalid/v1".to_string()),
                api_key: None,
                provider: Some("openai".to_string()),
            }),
        };
        let error = provider.model_config(&endpoint, None).unwrap_err();
        assert_eq!(error.kind(), HarnessErrorKind::UnsupportedCapability);
        assert_eq!(
            error.message(),
            "openjiuwen-native does not take endpoint models"
        );

        let error = provider
            .model_config(&choice(None, None), Some(&choice(None, None)))
            .unwrap_err();
        assert_eq!(
            error.message(),
            "openjiuwen-native has no authentication fallback"
        );
    }
}
