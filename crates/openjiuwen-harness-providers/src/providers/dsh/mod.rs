//! The `DeepSeek` Harness (DSH) provider: card, configuration data, model mapping and the
//! subprocess harness.
//!
//! Ports Python `openjiuwen.harness_providers.dsh` (v0.1.20): [`DshHarnessConfig`] mirrors
//! `DshHarnessConfig` in `config.py`, [`model_config`] mirrors the provider's `model_config`
//! classmethod, and [`DshDriver`] is the subprocess harness the provider's `create` returns.
//!
//! # Structure
//!
//! - this module — the configuration data, the system-prompt enum, the card, the provider and the
//!   model mapping;
//! - `wire` — the process invocation (argv, environment, `DSH_HOME` resolution);
//! - `frames` — the accumulator that projects DSH session events onto harness observations and
//!   builds the terminal turn result;
//! - `harness` — [`DshDriver`], the JSON-RPC transport that spawns the `dsh` runtime and drives one
//!   session.
//!
//! # Python parity
//!
//! - [`DshSystemPromptMode`] is a typed enum in place of Python's `Literal`, with Python's
//!   `replace` default and error wording.
//! - The plain serde decoders tolerate unknown fields (the crate-wide rule) while
//!   [`DshHarnessConfig::from_json`] — the provider SPI path — rejects them like Python's
//!   `from_mapping`.
//! - [`model_config`] returns the flat `{"model", "base_url", "api_key"}` mapping DSH's SDK takes,
//!   and rejects a reasoning effort or an authentication fallback, which DSH has no setting for.
//! - `api_key` is a [`SecretString`] so `Debug` redacts it, and the harness configuration's manual
//!   `Debug` redacts the `env` map (`repr=False` in Python).
use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::security::SecretString;
use openjiuwen_harness_protocol::{
    EventBufferConfig, EventOverflowPolicy, HarnessCapability, HarnessCard, HarnessError,
    HarnessProtocol, HarnessProvider, HarnessResult, HostCapability, JsonObject, ModelChoice,
    ModelConfigProvider,
};
use serde::{Deserialize, Serialize};

use crate::decode::{
    optional_f64, optional_str, optional_string_list, optional_string_map, optional_u32,
    optional_u64, reject_unknown_keys,
};
use crate::skills::{parse_skills_value, SkillConflict, SkillSource};
use crate::vendor_cli::{resolve_binary, BinarySpec, VendorHarness};

use super::vendor;

mod frames;
mod harness;
mod wire;

pub use harness::DshDriver;

// The shared `vendor` module still owns the Python "SDK required" text and its own unit tests
// assert it; this provider no longer produces that message, so keep the constant referenced in
// non-test builds (removing it is the vendor module's call, not this provider's).
const _: &str = crate::providers::vendor::DSH_SDK_REQUIRED;

/// Registry name of the DSH provider.
pub const PROVIDER_NAME: &str = "dsh";

/// Card name the DSH provider reports (Python `DshHarness.card`).
pub const CARD_NAME: &str = "deepseek-harness";

/// Version of the adapter whose card is reported here (Python `ADAPTER_VERSION`).
pub const IMPLEMENTATION_VERSION: &str = "0.3.0";

/// Configuration keys [`DshHarnessConfig::from_json`] accepts (Python `fields(cls)`).
const CONFIG_FIELDS: [&str; 22] = [
    "provider",
    "model",
    "reasoning_effort",
    "max_tokens",
    "system_prompt_mode",
    "skills",
    "skill_conflict",
    "cwd",
    "runtime_cwd",
    "dsh_bin",
    "dsh_home",
    "profile",
    "patches",
    "env",
    "launch_args_override",
    "initialize_timeout_seconds",
    "request_timeout_seconds",
    "shutdown_timeout_seconds",
    "base_url",
    "api_key",
    "system_prompt_env_var",
    "event_buffer_capacity",
];

/// How the host's system prompt combines with the DSH system prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DshSystemPromptMode {
    /// Append an independent host section to the DSH prompt.
    #[serde(rename = "append")]
    Append,
    /// Replace only the native prompt prefix.
    #[serde(rename = "replace")]
    Replace,
}

impl Default for DshSystemPromptMode {
    /// Returns [`DshSystemPromptMode::Replace`], Python's `system_prompt_mode` default.
    fn default() -> Self {
        Self::Replace
    }
}

/// Configuration of the `DeepSeek` Harness adapter (Python `DshHarnessConfig`).
///
/// Field names mirror `deepseek_harness.DeepSeekHarnessConfig`; `dsh_home` (or a non-empty
/// `DSH_HOME` in `env`) is mandatory for the SDK runtime, which never falls back to `~/.dsh`
/// implicitly. A custom Cordis composition must consume `system_prompt_env_var` when it is
/// explicitly set; otherwise the adapter uses the bundled system-prompt plugin through a temporary
/// launch overlay.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct DshHarnessConfig {
    /// Model provider registered in the DSH runtime.
    #[serde(default = "default_provider")]
    pub provider: String,
    /// Model served by that provider.
    #[serde(default = "default_model")]
    pub model: String,
    /// Reasoning effort of the model; `None` uses the provider default.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    /// Maximum number of generated tokens per request.
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// How the host system prompt combines with the DSH prompt.
    #[serde(default)]
    pub system_prompt_mode: DshSystemPromptMode,
    /// Skill sources installed into the DSH project-local discovery directory.
    #[serde(default)]
    pub skills: Vec<SkillSource>,
    /// Conflict policy applied when a destination skill already exists.
    #[serde(default = "default_skill_conflict")]
    pub skill_conflict: SkillConflict,
    /// Working directory of the DSH process.
    #[serde(default)]
    pub cwd: Option<PathBuf>,
    /// Working directory of the DSH runtime itself.
    #[serde(default)]
    pub runtime_cwd: Option<PathBuf>,
    /// Path of the DSH executable; `None` uses the SDK's lookup.
    #[serde(default)]
    pub dsh_bin: Option<PathBuf>,
    /// DSH home directory; it never defaults to `~/.dsh`.
    #[serde(default)]
    pub dsh_home: Option<PathBuf>,
    /// Cordis profile the runtime starts with.
    #[serde(default)]
    pub profile: Option<String>,
    /// Cordis plugin patches applied to the composition.
    #[serde(default)]
    pub patches: Vec<String>,
    /// Environment variables added to the DSH process; redacted in `Debug`.
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    /// Complete launch-argument vector replacing the adapter's own.
    #[serde(default)]
    pub launch_args_override: Option<Vec<String>>,
    /// Startup timeout of the runtime in seconds.
    #[serde(default)]
    pub initialize_timeout_seconds: Option<f64>,
    /// Timeout of one request in seconds.
    #[serde(default)]
    pub request_timeout_seconds: Option<f64>,
    /// Timeout of a graceful shutdown in seconds.
    #[serde(default = "default_shutdown_timeout_seconds")]
    pub shutdown_timeout_seconds: f64,
    /// Base URL of an external endpoint.
    #[serde(default)]
    pub base_url: Option<String>,
    /// API key of the external endpoint; redacted by [`SecretString`]'s `Debug` and `Display`.
    #[serde(default)]
    pub api_key: Option<SecretString>,
    /// Environment variable a custom composition reads the system prompt from.
    #[serde(default)]
    pub system_prompt_env_var: Option<String>,
    /// Capacity of the event buffer the harness publishes.
    #[serde(default = "default_event_buffer_capacity")]
    pub event_buffer_capacity: u64,
}

impl Default for DshHarnessConfig {
    /// Returns the Python defaults: the official `deepseek-v4-flash` provider, `replace` system
    /// prompt mode, a one-second shutdown timeout and a 1024-event buffer.
    fn default() -> Self {
        Self {
            provider: default_provider(),
            model: default_model(),
            reasoning_effort: None,
            max_tokens: None,
            system_prompt_mode: DshSystemPromptMode::default(),
            skills: Vec::new(),
            skill_conflict: SkillConflict::Skip,
            cwd: None,
            runtime_cwd: None,
            dsh_bin: None,
            dsh_home: None,
            profile: None,
            patches: Vec::new(),
            env: BTreeMap::new(),
            launch_args_override: None,
            initialize_timeout_seconds: None,
            request_timeout_seconds: None,
            shutdown_timeout_seconds: default_shutdown_timeout_seconds(),
            base_url: None,
            api_key: None,
            system_prompt_env_var: None,
            event_buffer_capacity: default_event_buffer_capacity(),
        }
    }
}

impl DshHarnessConfig {
    /// Decodes a provider SPI mapping, validating it like Python's `from_mapping`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an unknown key (`"unknown DSH configuration fields: …"`),
    /// a wrongly typed field, a mode conflict (`append` with `system_prompt_env_var`), an empty
    /// `provider` / `model` / `dsh_home` / `profile` / `system_prompt_env_var`, a non-positive
    /// `max_tokens` / `event_buffer_capacity` / timeout, or a `launch_args_override` that is not a
    /// non-empty array of non-empty strings.
    pub fn from_json(object: &JsonObject) -> Result<Self> {
        reject_unknown_keys(object, &CONFIG_FIELDS, "unknown DSH configuration fields")?;
        let defaults = Self::default();
        let system_prompt_mode = match optional_str(object, "system_prompt_mode")? {
            Some(value) => parse_system_prompt_mode(&value)?,
            None => defaults.system_prompt_mode,
        };
        let system_prompt_env_var = optional_str(object, "system_prompt_env_var")?;
        if system_prompt_mode == DshSystemPromptMode::Append && system_prompt_env_var.is_some() {
            return Err(Error::validation(
                "append mode cannot use system_prompt_env_var",
            ));
        }
        let skill_conflict = match optional_str(object, "skill_conflict")? {
            Some(value) => value.parse::<SkillConflict>()?,
            None => defaults.skill_conflict,
        };
        let launch_args_override = optional_string_list(object, "launch_args_override")?;
        let config = Self {
            provider: optional_str(object, "provider")?.unwrap_or(defaults.provider),
            model: optional_str(object, "model")?.unwrap_or(defaults.model),
            reasoning_effort: optional_str(object, "reasoning_effort")?,
            max_tokens: optional_u32(object, "max_tokens")?,
            system_prompt_mode,
            skills: parse_skills_value(object.get("skills"))?,
            skill_conflict,
            cwd: optional_str(object, "cwd")?.map(PathBuf::from),
            runtime_cwd: optional_str(object, "runtime_cwd")?.map(PathBuf::from),
            dsh_bin: optional_str(object, "dsh_bin")?.map(PathBuf::from),
            dsh_home: optional_str(object, "dsh_home")?.map(PathBuf::from),
            profile: optional_str(object, "profile")?,
            patches: optional_string_list(object, "patches")?.unwrap_or_default(),
            env: optional_string_map(object, "env")?.unwrap_or_default(),
            launch_args_override,
            initialize_timeout_seconds: optional_f64(object, "initialize_timeout_seconds")?,
            request_timeout_seconds: optional_f64(object, "request_timeout_seconds")?,
            shutdown_timeout_seconds: optional_f64(object, "shutdown_timeout_seconds")?
                .unwrap_or(defaults.shutdown_timeout_seconds),
            base_url: optional_str(object, "base_url")?,
            api_key: optional_str(object, "api_key")?.map(SecretString::new),
            system_prompt_env_var,
            event_buffer_capacity: optional_u64(object, "event_buffer_capacity")?
                .unwrap_or(defaults.event_buffer_capacity),
        };
        config.validate()?;
        Ok(config)
    }

    /// Applies the cross-field rules of Python's `__post_init__`.
    fn validate(&self) -> Result<()> {
        if self.provider.is_empty() || self.model.is_empty() {
            return Err(Error::validation(
                "DSH provider and model must not be empty",
            ));
        }
        if self.max_tokens == Some(0) {
            return Err(Error::validation("DSH max_tokens must be positive"));
        }
        if self.event_buffer_capacity == 0 {
            return Err(Error::validation(
                "DSH event_buffer_capacity must be positive",
            ));
        }
        for (name, timeout) in [
            (
                "initialize_timeout_seconds",
                self.initialize_timeout_seconds,
            ),
            ("request_timeout_seconds", self.request_timeout_seconds),
            (
                "shutdown_timeout_seconds",
                Some(self.shutdown_timeout_seconds),
            ),
        ] {
            if timeout.is_some_and(|seconds| seconds <= 0.0) {
                return Err(Error::validation(format!(
                    "DSH {name} must be positive when provided"
                )));
            }
        }
        if self
            .dsh_home
            .as_ref()
            .is_some_and(|home| home.to_string_lossy().trim().is_empty())
        {
            return Err(Error::validation("DSH dsh_home must not be empty"));
        }
        for (name, value) in [
            ("profile", self.profile.as_deref()),
            (
                "system_prompt_env_var",
                self.system_prompt_env_var.as_deref(),
            ),
        ] {
            if value.is_some_and(|text| text.trim().is_empty()) {
                return Err(Error::validation(format!("DSH {name} must not be empty")));
            }
        }
        if let Some(arguments) = &self.launch_args_override {
            if arguments.is_empty() || arguments.iter().any(String::is_empty) {
                return Err(Error::validation(
                    "DSH launch_args_override must contain non-empty strings",
                ));
            }
        }
        Ok(())
    }
}

impl fmt::Debug for DshHarnessConfig {
    /// Formats the configuration with the `env` values redacted (Python's `repr=False`).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DshHarnessConfig")
            .field("provider", &self.provider)
            .field("model", &self.model)
            .field("reasoning_effort", &self.reasoning_effort)
            .field("max_tokens", &self.max_tokens)
            .field("system_prompt_mode", &self.system_prompt_mode)
            .field("skills", &self.skills)
            .field("skill_conflict", &self.skill_conflict)
            .field("cwd", &self.cwd)
            .field("runtime_cwd", &self.runtime_cwd)
            .field("dsh_bin", &self.dsh_bin)
            .field("dsh_home", &self.dsh_home)
            .field("profile", &self.profile)
            .field("patches", &self.patches)
            .field("env", &"<redacted>")
            .field("launch_args_override", &self.launch_args_override)
            .field(
                "initialize_timeout_seconds",
                &self.initialize_timeout_seconds,
            )
            .field("request_timeout_seconds", &self.request_timeout_seconds)
            .field("shutdown_timeout_seconds", &self.shutdown_timeout_seconds)
            .field("base_url", &self.base_url)
            .field("api_key", &self.api_key)
            .field("system_prompt_env_var", &self.system_prompt_env_var)
            .field("event_buffer_capacity", &self.event_buffer_capacity)
            .finish()
    }
}

/// Returns the DSH provider.
#[must_use]
pub fn provider() -> DshHarnessProvider {
    DshHarnessProvider::default()
}

/// Returns the capability card of the DSH adapter.
///
/// The adapter keeps one reusable runtime session across turns, executes its own tools, consumes
/// MCP server configurations and runs endpoint models. The pinned DSH protocol exposes no steering
/// input, no abort of a running turn, no pause/resume and no portable checkpoint, so the card
/// declares none of those: [`abort`](HarnessProtocol::abort),
/// [`pause`](HarnessProtocol::pause) and
/// [`export_checkpoint`](HarnessProtocol::export_checkpoint) report
/// [`HarnessError::UnsupportedCapability`](openjiuwen_harness_protocol::HarnessError) instead of
/// pretending to work.
#[must_use]
pub fn card() -> HarnessCard {
    vendor::vendor_card(
        CARD_NAME,
        IMPLEMENTATION_VERSION,
        [
            HarnessCapability::PersistentSession,
            HarnessCapability::McpTools,
            HarnessCapability::EndpointModels,
        ],
        [HostCapability::McpServers],
        &[],
    )
}

/// Maps a model choice onto the flat `{"model", "base_url", "api_key"}` mapping DSH takes.
///
/// # Errors
///
/// Returns [`HarnessError::Validation`] when the choice carries a reasoning effort — DSH has no
/// reasoning effort setting — and [`HarnessError::UnsupportedCapability`] when a `fallback` is
/// requested, which DSH cannot switch to.
pub fn model_config(
    choice: &ModelChoice,
    fallback: Option<&ModelChoice>,
) -> HarnessResult<JsonObject> {
    if choice.effort.is_some() {
        return Err(HarnessError::validation(
            "DSH has no reasoning effort setting",
        ));
    }
    if fallback.is_some() {
        return Err(HarnessError::unsupported_capability(
            "DSH has no authentication fallback",
        ));
    }
    let endpoint = choice.endpoint.as_ref();
    let mut object = JsonObject::new();
    vendor::insert_opt_string(&mut object, "model", choice.model.as_deref());
    vendor::insert_opt_string(
        &mut object,
        "base_url",
        endpoint.and_then(|endpoint| endpoint.api_base.as_deref()),
    );
    vendor::insert_opt_string(
        &mut object,
        "api_key",
        endpoint.and_then(|endpoint| endpoint.api_key.as_deref()),
    );
    Ok(object)
}

/// The DSH provider: identity card plus the subprocess `create` path.
#[derive(Debug, Clone)]
pub struct DshHarnessProvider {
    /// Card reported by [`HarnessProvider::card`].
    card: HarnessCard,
}

impl Default for DshHarnessProvider {
    /// Builds the provider with the card of [`card`].
    fn default() -> Self {
        Self { card: card() }
    }
}

impl HarnessProvider for DshHarnessProvider {
    /// Returns the DSH adapter card.
    fn card(&self) -> &HarnessCard {
        &self.card
    }

    /// Validates `config`, resolves the `dsh` executable and `DSH_HOME`, and returns an unstarted
    /// harness.
    ///
    /// The returned [`VendorHarness`] wraps a [`DshDriver`], so it reports
    /// [`HarnessState::Terminated`](openjiuwen_harness_protocol::HarnessState::Terminated) until
    /// [`start`](HarnessProtocol::start) succeeds; it never spawns the runtime here.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::Validation`] when `config` is not a valid [`DshHarnessConfig`] (a
    /// bad `patches` entry included), and [`HarnessError::Protocol`] naming `dsh` when the
    /// executable cannot be resolved or naming `DSH_HOME` when neither `dsh_home` nor the
    /// environment supplies the mandatory home directory. A missing runtime or home is an operator
    /// configuration problem, never an unimplemented capability.
    fn create(&self, config: JsonObject) -> HarnessResult<Box<dyn HarnessProtocol>> {
        let config =
            DshHarnessConfig::from_json(&config).map_err(vendor::config_error_to_harness)?;
        let binary = resolve_binary(&BinarySpec {
            name: "dsh",
            explicit: config.dsh_bin.as_deref(),
            label: "DeepSeek Harness",
            install_hint: "install the DeepSeek Harness runtime (npm install -g deepseek-harness)",
        })?;
        let dsh_home = wire::require_dsh_home(&config)?;
        // A patch path is resolved against the project directory before anything is spawned, so an
        // unusable path is a configuration error and not a failed start.
        let patches_base = config.cwd.clone().or_else(|| config.runtime_cwd.clone());
        let launch = wire::build_args(&config, patches_base.as_deref())
            .map_err(vendor::config_error_to_harness)?;
        let buffer = EventBufferConfig {
            capacity: config.event_buffer_capacity,
            overflow: EventOverflowPolicy::CoalesceOrBlock,
        };
        let driver = Arc::new(DshDriver::new(config, binary, dsh_home, launch));
        Ok(Box::new(VendorHarness::new(card(), buffer, driver)))
    }
}

impl ModelConfigProvider for DshHarnessProvider {
    /// Forwards to the module's [`model_config`].
    ///
    /// # Errors
    ///
    /// Returns the effort and fallback rejections of [`model_config`].
    fn model_config(
        &self,
        choice: &ModelChoice,
        fallback: Option<&ModelChoice>,
    ) -> HarnessResult<JsonObject> {
        model_config(choice, fallback)
    }
}

/// Parses a Python `system_prompt_mode` wire value (the error keeps Python's prefix-free wording).
fn parse_system_prompt_mode(value: &str) -> Result<DshSystemPromptMode> {
    match value {
        "append" => Ok(DshSystemPromptMode::Append),
        "replace" => Ok(DshSystemPromptMode::Replace),
        _ => Err(Error::validation(
            "system_prompt_mode must be 'append' or 'replace'",
        )),
    }
}

/// Returns [`SkillConflict::Skip`], Python's `skill_conflict` default.
fn default_skill_conflict() -> SkillConflict {
    SkillConflict::Skip
}

/// Returns the default DSH model provider (`deepseek-official`).
fn default_provider() -> String {
    "deepseek-official".to_owned()
}

/// Returns the default DSH model (`deepseek-v4-flash`).
fn default_model() -> String {
    "deepseek-v4-flash".to_owned()
}

/// Returns the default shutdown timeout (1 second).
const fn default_shutdown_timeout_seconds() -> f64 {
    1.0
}

/// Returns the default event-buffer capacity (1024 events).
const fn default_event_buffer_capacity() -> u64 {
    1024
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use openjiuwen_common::security::SecretString;
    use openjiuwen_harness_protocol::{
        HarnessCapability, HarnessErrorKind, HarnessProvider, HostCapability, ModelChoice,
        ModelEndpoint,
    };
    use serde_json::json;

    use crate::decode::expect_object;

    use super::{
        card, model_config, provider, DshHarnessConfig, DshSystemPromptMode, CARD_NAME,
        IMPLEMENTATION_VERSION, PROVIDER_NAME,
    };

    /// Converts a JSON literal into a provider create mapping.
    // The temporary from `json!` is forwarded into the decoder, so taking it by value keeps the
    // call sites readable.
    #[allow(clippy::needless_pass_by_value)]
    fn object(value: serde_json::Value) -> openjiuwen_harness_protocol::JsonObject {
        expect_object(&value, "configuration").expect("a JSON object")
    }

    /// Choice that runs the built-in endpoint.
    fn builtin_choice() -> ModelChoice {
        ModelChoice {
            model: Some("deepseek-v4-flash".to_owned()),
            effort: None,
            endpoint: None,
        }
    }

    #[test]
    fn identity_constants_match_python() {
        assert_eq!(PROVIDER_NAME, "dsh");
        assert_eq!(CARD_NAME, "deepseek-harness");
        assert_eq!(IMPLEMENTATION_VERSION, "0.3.0");
    }

    #[test]
    fn card_declares_a_persistent_session_mcp_tools_and_endpoint_models() {
        let card = card();
        assert_eq!(card.name, "deepseek-harness");
        assert_eq!(card.implementation_version, "0.3.0");
        assert_eq!(
            card.capabilities,
            std::collections::BTreeSet::from([
                HarnessCapability::PersistentSession,
                HarnessCapability::McpTools,
                HarnessCapability::EndpointModels,
            ])
        );
        assert!(!card
            .capabilities
            .contains(&HarnessCapability::GracefulAbort));
        assert!(!card.capabilities.contains(&HarnessCapability::ForceAbort));
        assert!(!card.capabilities.contains(&HarnessCapability::Steer));
        assert!(!card.capabilities.contains(&HarnessCapability::PauseResume));
        assert!(!card.capabilities.contains(&HarnessCapability::Checkpoint));
        assert_eq!(
            card.optional_host_capabilities,
            std::collections::BTreeSet::from([HostCapability::McpServers])
        );
        assert!(card.required_host_capabilities.is_empty());
        assert!(card.fork_modes.is_empty());
        assert_eq!(provider().card(), &card);
    }

    #[test]
    fn default_configuration_matches_python() {
        let config = DshHarnessConfig::default();
        assert_eq!(config.provider, "deepseek-official");
        assert_eq!(config.model, "deepseek-v4-flash");
        assert_eq!(config.system_prompt_mode, DshSystemPromptMode::Replace);
        assert!((config.shutdown_timeout_seconds - 1.0).abs() < f64::EPSILON);
        assert_eq!(config.event_buffer_capacity, 1024);
        assert_eq!(config.max_tokens, None);
        assert_eq!(config.launch_args_override, None);
    }

    #[test]
    fn from_json_accepts_an_empty_mapping() {
        let config = DshHarnessConfig::from_json(&object(json!({}))).expect("defaults");
        assert_eq!(config, DshHarnessConfig::default());
    }

    #[test]
    fn from_json_rejects_unknown_keys_sorted() {
        let error = DshHarnessConfig::from_json(&object(json!({"zeta": 1, "alpha": 2})))
            .expect_err("unknown keys are rejected");
        assert_eq!(
            error.to_string(),
            "validation: unknown DSH configuration fields: alpha, zeta"
        );
    }

    #[test]
    fn from_json_rejects_append_mode_with_a_prompt_env_var() {
        let error = DshHarnessConfig::from_json(&object(json!({
            "system_prompt_mode": "append",
            "system_prompt_env_var": "DSH_SYSTEM_PROMPT",
        })))
        .expect_err("append mode cannot use the env var");
        assert_eq!(
            error.to_string(),
            "validation: append mode cannot use system_prompt_env_var"
        );
    }

    #[test]
    fn from_json_decodes_launch_arguments_and_endpoints() {
        let config = DshHarnessConfig::from_json(&object(json!({
            "launch_args_override": ["--profile", "prod"],
            "shutdown_timeout_seconds": 2.5,
            "api_key": "sk-test",
            "provider": "custom",
        })))
        .expect("a valid configuration");
        assert_eq!(
            config.launch_args_override,
            Some(vec!["--profile".to_owned(), "prod".to_owned()])
        );
        assert!((config.shutdown_timeout_seconds - 2.5).abs() < f64::EPSILON);
        assert_eq!(config.provider, "custom");
        assert_eq!(
            config.api_key.as_ref().map(SecretString::expose_secret),
            Some("sk-test")
        );
    }

    #[test]
    fn from_json_rejects_blank_provider() {
        let error = DshHarnessConfig::from_json(&object(json!({"provider": ""})))
            .expect_err("an empty provider is invalid");
        assert_eq!(
            error.to_string(),
            "validation: DSH provider and model must not be empty"
        );
    }

    #[test]
    fn from_json_rejects_empty_launch_arguments() {
        for value in [json!([]), json!(["--profile", ""])] {
            let error = DshHarnessConfig::from_json(&object(json!({
                "launch_args_override": value,
            })))
            .expect_err("empty arguments are invalid");
            assert_eq!(
                error.to_string(),
                "validation: DSH launch_args_override must contain non-empty strings"
            );
        }
    }

    #[test]
    fn model_config_returns_the_flat_mapping() {
        let choice = ModelChoice {
            model: Some("deepseek-v4-flash".to_owned()),
            effort: None,
            endpoint: Some(ModelEndpoint {
                api_base: Some("https://api.example.test".to_owned()),
                api_key: Some("sk-test".to_owned()),
                provider: Some("openai".to_owned()),
            }),
        };
        let fragment = model_config(&choice, None).expect("a model fragment");
        assert_eq!(
            fragment,
            expect_object(
                &json!({
                    "model": "deepseek-v4-flash",
                    "base_url": "https://api.example.test",
                    "api_key": "sk-test",
                }),
                "fragment"
            )
            .expect("a JSON object")
        );
    }

    #[test]
    fn model_config_rejects_effort_and_fallback() {
        let effort = ModelChoice {
            effort: Some("high".to_owned()),
            ..builtin_choice()
        };
        let error = model_config(&effort, None).expect_err("DSH has no effort setting");
        assert_eq!(error.kind(), HarnessErrorKind::Validation);
        assert_eq!(error.message(), "DSH has no reasoning effort setting");

        let error = model_config(&builtin_choice(), Some(&builtin_choice()))
            .expect_err("DSH has no authentication fallback");
        assert_eq!(error.kind(), HarnessErrorKind::UnsupportedCapability);
        assert_eq!(error.message(), "DSH has no authentication fallback");
    }

    #[test]
    fn create_reports_the_missing_runtime() {
        let error = provider()
            .create(object(json!({"dsh_bin": "/nonexistent/dsh"})))
            .err()
            .expect("the runtime is required");
        assert_eq!(error.kind(), HarnessErrorKind::Protocol);
        assert!(error.message().contains("dsh"), "{}", error.message());
    }

    #[test]
    fn debug_redacts_the_environment() {
        let config = DshHarnessConfig::from_json(&object(json!({
            "env": {"DEEPSEEK_TOKEN": "secret-value"},
        })))
        .expect("a valid configuration");
        let rendered = format!("{config:?}");
        assert!(!rendered.contains("secret-value"));
        assert!(rendered.contains("<redacted>"));
    }
}
