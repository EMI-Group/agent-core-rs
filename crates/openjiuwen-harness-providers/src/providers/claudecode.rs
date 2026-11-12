//! The deferred Claude Code provider: card, configuration data and model mapping.
//!
//! Ports the data half of Python `openjiuwen.harness_providers.claudecode` (v0.1.20):
//! [`ClaudeCodeHarnessConfig`] mirrors `ClaudeCodeHarnessConfig` in `config.py`,
//! [`ClaudeModelConfig`] mirrors `ClaudeModelConfig`, and [`model_config`] mirrors the provider's
//! `model_config` classmethod. The harness itself is not ported — the adapter needs the Claude
//! Agent SDK and the `claude` CLI, so [`HarnessProvider::create`] validates the mapping and then
//! reports `HarnessError::UnsupportedCapability` with the SDK requirement.
//!
//! # Python parity
//!
//! - [`ClaudePermissionMode`] and [`ClaudeSystemPromptMode`] are typed enums in place of Python's
//!   `Literal`/string fields, so the `permission_mode` / `system_prompt_mode` strings are validated
//!   when a configuration is decoded. The wire values (`default`, `acceptEdits`, `plan`,
//!   `bypassPermissions`, `dontAsk`, `auto`; `append`, `replace`) and the error wording are Python's.
//! - The plain serde decoders tolerate unknown fields (the crate-wide rule) while
//!   [`ClaudeCodeHarnessConfig::from_json`] — the provider SPI path — rejects them like Python's
//!   `from_mapping`.
//! - `api_key` is a [`SecretString`] so `Debug` redacts it, and the harness configuration's manual
//!   `Debug` redacts the `env` / `settings_env` maps (`repr=False` in Python).
use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::security::SecretString;
use openjiuwen_harness_protocol::{
    HarnessCapability, HarnessCard, HarnessProtocol, HarnessProvider, HarnessResult,
    HostCapability, JsonObject, JsonValue, ModelChoice, ModelConfigProvider,
};
use serde::{Deserialize, Serialize};

use crate::decode::{
    expect_object, optional_bool, optional_f64, optional_str, optional_string_list,
    optional_string_map, optional_u32, optional_u64, reject_unknown_keys,
};
use crate::skills::{parse_skills_value, SkillConflict, SkillSource};

use super::vendor;

/// Registry name of the Claude Code provider.
pub const PROVIDER_NAME: &str = "claudecode";

/// Card name the Claude Code provider reports (Python `mapping.PROVIDER_NAME`).
pub const CARD_NAME: &str = "claude-code";

/// Version of the adapter whose card is reported here (Python `ADAPTER_VERSION`).
pub const IMPLEMENTATION_VERSION: &str = "0.1.0";

/// Per-line stdout buffer ceiling (bytes) of the Claude SDK NDJSON transport.
///
/// Python raises the SDK's 1 MiB default because tool results that embed large payloads (a base64
/// image read through the CLI `Read` tool is roughly 4/3 of the file size) otherwise kill the turn
/// with a decode error. The buffer is a transient per-line string, so the memory cost only appears
/// while a large message is in flight.
pub const DEFAULT_CLAUDE_MAX_BUFFER_SIZE: u64 = 32 * 1024 * 1024;

/// Configuration keys [`ClaudeCodeHarnessConfig::from_json`] accepts (Python `fields(cls)`).
const CONFIG_FIELDS: [&str; 20] = [
    "skills",
    "skill_conflict",
    "cwd",
    "add_dirs",
    "env",
    "inherit_process_env",
    "cli_path",
    "model",
    "fallback_model",
    "session_id",
    "permission_mode",
    "system_prompt_mode",
    "include_partial_messages",
    "max_turns",
    "max_buffer_size",
    "settings",
    "settings_env",
    "event_buffer_capacity",
    "request_observation_wait_s",
    "lifecycle_ack_timeout_s",
];

/// Configuration keys [`ClaudeModelConfig::from_json`] accepts (Python `fields(cls)`).
const MODEL_CONFIG_FIELDS: [&str; 4] = ["model", "api_base", "api_key", "effort"];

/// Permission mode of the Claude CLI (`--permission-mode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClaudePermissionMode {
    /// Standard prompting for permissions the CLI does not already allow.
    #[serde(rename = "default")]
    Default,
    /// Accept file edits automatically.
    #[serde(rename = "acceptEdits")]
    AcceptEdits,
    /// Plan mode: inspect and propose without changing anything.
    #[serde(rename = "plan")]
    Plan,
    /// Skip every permission prompt (the CLI default of this adapter).
    #[serde(rename = "bypassPermissions")]
    BypassPermissions,
    /// Deny instead of asking when a tool would need approval.
    #[serde(rename = "dontAsk")]
    DontAsk,
    /// Let the CLI decide between prompting and its own heuristics.
    #[serde(rename = "auto")]
    Auto,
}

impl Default for ClaudePermissionMode {
    /// Returns [`ClaudePermissionMode::BypassPermissions`], Python's `permission_mode` default.
    fn default() -> Self {
        Self::BypassPermissions
    }
}

/// How the host's system prompt combines with the Claude Code preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClaudeSystemPromptMode {
    /// Append the host system prompt to the CLI's preset prompt.
    #[serde(rename = "append")]
    Append,
    /// Replace the CLI's preset prompt with the host system prompt.
    #[serde(rename = "replace")]
    Replace,
}

impl Default for ClaudeSystemPromptMode {
    /// Returns [`ClaudeSystemPromptMode::Append`], Python's `system_prompt_mode` default.
    fn default() -> Self {
        Self::Append
    }
}

/// Model endpoint the Claude CLI runs on.
///
/// `api_base` / `api_key` are injected through the CLI `--settings` flag-settings layer so they win
/// over the user's `settings.json`. Without them the CLI runs on its own login (for example a
/// subscription), where `model` may be a built-in alias such as `"sonnet"` or `"haiku"`. `effort`
/// is the CLI `--effort` level (`low` … `max`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ClaudeModelConfig {
    /// Model identifier or built-in alias.
    #[serde(default)]
    pub model: Option<String>,
    /// API base URL of an external endpoint.
    #[serde(default)]
    pub api_base: Option<String>,
    /// API key of an external endpoint; redacted by [`SecretString`]'s `Debug` and `Display`.
    #[serde(default)]
    pub api_key: Option<SecretString>,
    /// CLI `--effort` level.
    #[serde(default)]
    pub effort: Option<String>,
}

impl ClaudeModelConfig {
    /// Decodes a model configuration, rejecting unknown keys like Python's `from_mapping`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an unknown key (`"unknown Claude model config fields: …"`),
    /// a wrong field type, or a provided string field that is empty
    /// (`"Claude model {field} must be a non-empty string when provided"`).
    pub fn from_json(object: &JsonObject) -> Result<Self> {
        reject_unknown_keys(
            object,
            &MODEL_CONFIG_FIELDS,
            "unknown Claude model config fields",
        )?;
        let config = Self {
            model: optional_str(object, "model")?,
            api_base: optional_str(object, "api_base")?,
            api_key: optional_str(object, "api_key")?.map(SecretString::new),
            effort: optional_str(object, "effort")?,
        };
        for (name, value) in [
            ("model", config.model.as_deref()),
            ("api_base", config.api_base.as_deref()),
            (
                "api_key",
                config.api_key.as_ref().map(SecretString::expose_secret),
            ),
            ("effort", config.effort.as_deref()),
        ] {
            if value == Some("") {
                return Err(Error::validation(format!(
                    "Claude model {name} must be a non-empty string when provided"
                )));
            }
        }
        Ok(config)
    }
}

/// Configuration of the Claude Code harness (Python `ClaudeCodeHarnessConfig`).
///
/// The optional SDK is deliberately not imported by this decoder, so the configuration can be
/// validated offline. `session_id` pins the CLI session id; when omitted the eventual adapter
/// derives a stable UUID from the host session and agent name so persistent sessions survive
/// restarts.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ClaudeCodeHarnessConfig {
    /// Skill sources installed into the CLI's project-local discovery directory.
    #[serde(default)]
    pub skills: Vec<SkillSource>,
    /// Conflict policy applied when a destination skill already exists.
    #[serde(default = "default_skill_conflict")]
    pub skill_conflict: SkillConflict,
    /// Working directory of the CLI process.
    #[serde(default)]
    pub cwd: Option<PathBuf>,
    /// Additional directories the CLI may read and edit (`--add-dir`).
    #[serde(default)]
    pub add_dirs: Vec<PathBuf>,
    /// Environment variables added to the CLI process; redacted in `Debug`.
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    /// Whether the CLI process inherits the host process environment.
    #[serde(default = "default_true")]
    pub inherit_process_env: bool,
    /// Path of the `claude` executable; `None` uses the SDK's lookup.
    #[serde(default)]
    pub cli_path: Option<PathBuf>,
    /// Model endpoint used for the session.
    #[serde(default)]
    pub model: Option<ClaudeModelConfig>,
    /// Model endpoint the CLI switches to after an authentication failure.
    #[serde(default)]
    pub fallback_model: Option<ClaudeModelConfig>,
    /// Pinned CLI session id; `None` lets the adapter derive one.
    #[serde(default)]
    pub session_id: Option<String>,
    /// Permission mode passed to the CLI.
    #[serde(default)]
    pub permission_mode: ClaudePermissionMode,
    /// How the host system prompt combines with the CLI preset.
    #[serde(default)]
    pub system_prompt_mode: ClaudeSystemPromptMode,
    /// Stream partial assistant messages instead of complete ones.
    #[serde(default = "default_true")]
    pub include_partial_messages: bool,
    /// Maximum number of agent turns per query.
    #[serde(default)]
    pub max_turns: Option<u32>,
    /// Stdout buffer ceiling in bytes; see [`DEFAULT_CLAUDE_MAX_BUFFER_SIZE`].
    #[serde(default)]
    pub max_buffer_size: Option<u64>,
    /// Raw JSON passed to the CLI `--settings` flag.
    #[serde(default)]
    pub settings: Option<JsonValue>,
    /// Environment variables merged into the CLI `--settings` layer; redacted in `Debug`.
    #[serde(default)]
    pub settings_env: BTreeMap<String, String>,
    /// Capacity of the event buffer the harness publishes.
    #[serde(default = "default_event_buffer_capacity")]
    pub event_buffer_capacity: u64,
    /// How long a finished reply waits for the CLI's request logs before its model request is
    /// reported from the SDK message alone.
    #[serde(default = "default_request_observation_wait_s")]
    pub request_observation_wait_s: f64,
    /// How long a turn waits, after a result, for a delivery receipt on a message the CLI has not
    /// acknowledged at all.
    #[serde(default = "default_lifecycle_ack_timeout_s")]
    pub lifecycle_ack_timeout_s: f64,
}

impl Default for ClaudeCodeHarnessConfig {
    /// Returns the Python defaults: no skills or endpoints, bypassed permissions, `append` system
    /// prompt mode, a 1024-event buffer and the observation/acknowledgement timeouts.
    fn default() -> Self {
        Self {
            skills: Vec::new(),
            skill_conflict: SkillConflict::Skip,
            cwd: None,
            add_dirs: Vec::new(),
            env: BTreeMap::new(),
            inherit_process_env: true,
            cli_path: None,
            model: None,
            fallback_model: None,
            session_id: None,
            permission_mode: ClaudePermissionMode::default(),
            system_prompt_mode: ClaudeSystemPromptMode::default(),
            include_partial_messages: true,
            max_turns: None,
            max_buffer_size: None,
            settings: None,
            settings_env: BTreeMap::new(),
            event_buffer_capacity: default_event_buffer_capacity(),
            request_observation_wait_s: default_request_observation_wait_s(),
            lifecycle_ack_timeout_s: default_lifecycle_ack_timeout_s(),
        }
    }
}

impl ClaudeCodeHarnessConfig {
    /// Decodes a provider SPI mapping, validating it like Python's `from_mapping`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an unknown key (`"unknown Claude Code configuration
    /// fields: …"`), a wrongly typed or empty field, or a cross-field rule: an empty `session_id`,
    /// a non-positive `max_turns` / `max_buffer_size` / `event_buffer_capacity`, or a
    /// non-positive `lifecycle_ack_timeout_s`.
    pub fn from_json(object: &JsonObject) -> Result<Self> {
        reject_unknown_keys(
            object,
            &CONFIG_FIELDS,
            "unknown Claude Code configuration fields",
        )?;
        let defaults = Self::default();
        let skill_conflict = match optional_str(object, "skill_conflict")? {
            Some(value) => value.parse::<SkillConflict>()?,
            None => defaults.skill_conflict,
        };
        let permission_mode = match optional_str(object, "permission_mode")? {
            Some(value) => parse_permission_mode(&value)?,
            None => defaults.permission_mode,
        };
        let system_prompt_mode = match optional_str(object, "system_prompt_mode")? {
            Some(value) => parse_system_prompt_mode(&value)?,
            None => defaults.system_prompt_mode,
        };
        let config = Self {
            skills: parse_skills_value(object.get("skills"))?,
            skill_conflict,
            cwd: optional_str(object, "cwd")?.map(PathBuf::from),
            add_dirs: optional_string_list(object, "add_dirs")?
                .map(|dirs| dirs.into_iter().map(PathBuf::from).collect())
                .unwrap_or_default(),
            env: optional_string_map(object, "env")?.unwrap_or_default(),
            inherit_process_env: optional_bool(object, "inherit_process_env")?
                .unwrap_or(defaults.inherit_process_env),
            cli_path: optional_str(object, "cli_path")?.map(PathBuf::from),
            model: parse_model_config(object, "model")?,
            fallback_model: parse_model_config(object, "fallback_model")?,
            session_id: optional_str(object, "session_id")?,
            permission_mode,
            system_prompt_mode,
            include_partial_messages: optional_bool(object, "include_partial_messages")?
                .unwrap_or(defaults.include_partial_messages),
            max_turns: optional_u32(object, "max_turns")?,
            max_buffer_size: optional_u64(object, "max_buffer_size")?,
            settings: object
                .get("settings")
                .cloned()
                .filter(|value| !value.is_null()),
            settings_env: optional_string_map(object, "settings_env")?.unwrap_or_default(),
            event_buffer_capacity: optional_u64(object, "event_buffer_capacity")?
                .unwrap_or(defaults.event_buffer_capacity),
            request_observation_wait_s: optional_f64(object, "request_observation_wait_s")?
                .unwrap_or(defaults.request_observation_wait_s),
            lifecycle_ack_timeout_s: optional_f64(object, "lifecycle_ack_timeout_s")?
                .unwrap_or(defaults.lifecycle_ack_timeout_s),
        };
        config.validate()?;
        Ok(config)
    }

    /// Applies the cross-field rules of Python's `__post_init__`.
    fn validate(&self) -> Result<()> {
        if self.session_id.as_deref() == Some("") {
            return Err(Error::validation("Claude session_id must not be empty"));
        }
        if self.max_turns == Some(0) {
            return Err(Error::validation(
                "Claude max_turns must be a positive integer when provided",
            ));
        }
        if self.max_buffer_size == Some(0) {
            return Err(Error::validation(
                "Claude max_buffer_size must be a positive integer when provided",
            ));
        }
        if self.event_buffer_capacity == 0 {
            return Err(Error::validation(
                "Claude event_buffer_capacity must be positive",
            ));
        }
        if self.lifecycle_ack_timeout_s <= 0.0 {
            return Err(Error::validation(
                "Claude lifecycle_ack_timeout_s must be positive",
            ));
        }
        Ok(())
    }
}

impl fmt::Debug for ClaudeCodeHarnessConfig {
    /// Formats the configuration with the `env` and `settings_env` values redacted (Python's
    /// `repr=False`).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClaudeCodeHarnessConfig")
            .field("skills", &self.skills)
            .field("skill_conflict", &self.skill_conflict)
            .field("cwd", &self.cwd)
            .field("add_dirs", &self.add_dirs)
            .field("env", &"<redacted>")
            .field("inherit_process_env", &self.inherit_process_env)
            .field("cli_path", &self.cli_path)
            .field("model", &self.model)
            .field("fallback_model", &self.fallback_model)
            .field("session_id", &self.session_id)
            .field("permission_mode", &self.permission_mode)
            .field("system_prompt_mode", &self.system_prompt_mode)
            .field("include_partial_messages", &self.include_partial_messages)
            .field("max_turns", &self.max_turns)
            .field("max_buffer_size", &self.max_buffer_size)
            .field("settings", &self.settings)
            .field("settings_env", &"<redacted>")
            .field("event_buffer_capacity", &self.event_buffer_capacity)
            .field(
                "request_observation_wait_s",
                &self.request_observation_wait_s,
            )
            .field("lifecycle_ack_timeout_s", &self.lifecycle_ack_timeout_s)
            .finish()
    }
}

/// Returns the deferred Claude Code provider stub.
#[must_use]
pub fn provider() -> ClaudeCodeHarnessProvider {
    ClaudeCodeHarnessProvider::default()
}

/// Returns the capability card of the Claude Code adapter.
///
/// The card describes what the eventual adapter exploits; this build never creates it.
#[must_use]
pub fn card() -> HarnessCard {
    vendor::vendor_card(
        CARD_NAME,
        IMPLEMENTATION_VERSION,
        [
            HarnessCapability::Steer,
            HarnessCapability::GracefulAbort,
            HarnessCapability::PersistentSession,
            HarnessCapability::Checkpoint,
            HarnessCapability::McpTools,
            HarnessCapability::ModelSelection,
            HarnessCapability::ModelDiscovery,
            HarnessCapability::EndpointModels,
            HarnessCapability::AuthFallback,
        ],
        [
            HostCapability::ToolApproval,
            HostCapability::UserInput,
            HostCapability::CheckpointSink,
            HostCapability::McpServers,
            HostCapability::ProviderInteraction,
            HostCapability::ModelRequestObservation,
        ],
        &["full", "before"],
    )
}

/// Maps a model choice onto `{"model": {…}}` (plus `{"fallback_model": {…}}`).
///
/// Claude Code speaks the Anthropic protocol only, so an endpoint's `provider` label is not
/// forwarded (Python's `_claude_model` drops it deliberately).
///
/// # Errors
///
/// This mapping cannot fail; the [`HarnessResult`] signature matches the SPI.
pub fn model_config(
    choice: &ModelChoice,
    fallback: Option<&ModelChoice>,
) -> HarnessResult<JsonObject> {
    Ok(vendor::nested_model_config(choice, fallback, false))
}

/// The deferred Claude Code provider: identity card plus validation-only `create`.
#[derive(Debug, Clone)]
pub struct ClaudeCodeHarnessProvider {
    /// Card reported by [`HarnessProvider::card`].
    card: HarnessCard,
}

impl Default for ClaudeCodeHarnessProvider {
    /// Builds the stub with the card of [`card`].
    fn default() -> Self {
        Self { card: card() }
    }
}

impl HarnessProvider for ClaudeCodeHarnessProvider {
    /// Returns the Claude Code adapter card.
    fn card(&self) -> &HarnessCard {
        &self.card
    }

    /// Validates `config` and then reports the missing Claude Agent SDK.
    ///
    /// # Errors
    ///
    /// Returns `HarnessError::Validation` when `config` is not a valid
    /// [`ClaudeCodeHarnessConfig`], and `HarnessError::UnsupportedCapability` carrying
    /// `vendor::CLAUDE_SDK_REQUIRED` otherwise.
    fn create(&self, config: JsonObject) -> HarnessResult<Box<dyn HarnessProtocol>> {
        vendor::unsupported_create(
            ClaudeCodeHarnessConfig::from_json(&config).map(|_| ()),
            vendor::CLAUDE_SDK_REQUIRED,
        )
    }
}

impl ModelConfigProvider for ClaudeCodeHarnessProvider {
    /// Forwards to the module's [`model_config`].
    ///
    /// # Errors
    ///
    /// This mapping cannot fail; the [`HarnessResult`] signature matches the SPI.
    fn model_config(
        &self,
        choice: &ModelChoice,
        fallback: Option<&ModelChoice>,
    ) -> HarnessResult<JsonObject> {
        model_config(choice, fallback)
    }
}

/// Decodes an optional nested model configuration (`model` / `fallback_model`).
fn parse_model_config(object: &JsonObject, key: &str) -> Result<Option<ClaudeModelConfig>> {
    match object.get(key) {
        None | Some(JsonValue::Null) => Ok(None),
        Some(value) => {
            let nested = expect_object(value, "Claude model config")?;
            Ok(Some(ClaudeModelConfig::from_json(&nested)?))
        }
    }
}

/// Parses a Python `permission_mode` wire value.
fn parse_permission_mode(value: &str) -> Result<ClaudePermissionMode> {
    match value {
        "default" => Ok(ClaudePermissionMode::Default),
        "acceptEdits" => Ok(ClaudePermissionMode::AcceptEdits),
        "plan" => Ok(ClaudePermissionMode::Plan),
        "bypassPermissions" => Ok(ClaudePermissionMode::BypassPermissions),
        "dontAsk" => Ok(ClaudePermissionMode::DontAsk),
        "auto" => Ok(ClaudePermissionMode::Auto),
        _ => Err(Error::validation(
            "Claude permission_mode must be one of default, acceptEdits, plan, bypassPermissions, \
             dontAsk, auto",
        )),
    }
}

/// Parses a Python `system_prompt_mode` wire value.
fn parse_system_prompt_mode(value: &str) -> Result<ClaudeSystemPromptMode> {
    match value {
        "append" => Ok(ClaudeSystemPromptMode::Append),
        "replace" => Ok(ClaudeSystemPromptMode::Replace),
        _ => Err(Error::validation(
            "Claude system_prompt_mode must be 'append' or 'replace'",
        )),
    }
}

/// Returns [`SkillConflict::Skip`], Python's `skill_conflict` default.
fn default_skill_conflict() -> SkillConflict {
    SkillConflict::Skip
}

/// Returns `true`, Python's default for the process-environment flags.
const fn default_true() -> bool {
    true
}

/// Returns the default event-buffer capacity (1024 events).
const fn default_event_buffer_capacity() -> u64 {
    1024
}

/// Returns the default request-observation wait (5 seconds).
const fn default_request_observation_wait_s() -> f64 {
    5.0
}

/// Returns the default lifecycle-acknowledgement timeout (10 seconds).
const fn default_lifecycle_ack_timeout_s() -> f64 {
    10.0
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use openjiuwen_harness_protocol::{
        HarnessCapability, HarnessErrorKind, HarnessProvider, HostCapability, ModelChoice,
        ModelEndpoint,
    };
    use serde_json::json;

    use crate::decode::expect_object;

    use super::{
        card, model_config, provider, ClaudeCodeHarnessConfig, ClaudeModelConfig,
        ClaudePermissionMode, ClaudeSystemPromptMode, CARD_NAME, DEFAULT_CLAUDE_MAX_BUFFER_SIZE,
        IMPLEMENTATION_VERSION, PROVIDER_NAME,
    };

    /// Converts a JSON literal into a provider create mapping.
    // The temporary from `json!` is forwarded into the decoder, so taking it by value keeps the
    // call sites readable.
    #[allow(clippy::needless_pass_by_value)]
    fn object(value: serde_json::Value) -> openjiuwen_harness_protocol::JsonObject {
        expect_object(&value, "configuration").expect("a JSON object")
    }

    #[test]
    fn identity_constants_match_python() {
        assert_eq!(PROVIDER_NAME, "claudecode");
        assert_eq!(CARD_NAME, "claude-code");
        assert_eq!(IMPLEMENTATION_VERSION, "0.1.0");
        assert_eq!(DEFAULT_CLAUDE_MAX_BUFFER_SIZE, 32 * 1024 * 1024);
    }

    #[test]
    fn card_declares_the_claude_capabilities() {
        let card = card();
        assert_eq!(card.name, "claude-code");
        assert_eq!(card.implementation_version, "0.1.0");
        for capability in [
            HarnessCapability::Steer,
            HarnessCapability::GracefulAbort,
            HarnessCapability::PersistentSession,
            HarnessCapability::Checkpoint,
            HarnessCapability::McpTools,
            HarnessCapability::ModelSelection,
            HarnessCapability::ModelDiscovery,
            HarnessCapability::EndpointModels,
            HarnessCapability::AuthFallback,
        ] {
            assert!(card.supports(capability), "{capability:?} must be declared");
        }
        for capability in [
            HostCapability::ToolApproval,
            HostCapability::UserInput,
            HostCapability::CheckpointSink,
            HostCapability::McpServers,
            HostCapability::ProviderInteraction,
            HostCapability::ModelRequestObservation,
        ] {
            assert!(card.optional_host_capabilities.contains(&capability));
        }
        assert!(card.required_host_capabilities.is_empty());
        assert_eq!(provider().card(), &card);
    }

    #[test]
    fn default_configuration_matches_python() {
        let config = ClaudeCodeHarnessConfig::default();
        assert_eq!(config.skills.len(), 0);
        assert_eq!(config.add_dirs.len(), 0);
        assert!(config.inherit_process_env);
        assert_eq!(
            config.permission_mode,
            ClaudePermissionMode::BypassPermissions
        );
        assert_eq!(config.system_prompt_mode, ClaudeSystemPromptMode::Append);
        assert!(config.include_partial_messages);
        assert_eq!(config.event_buffer_capacity, 1024);
        assert!((config.request_observation_wait_s - 5.0).abs() < f64::EPSILON);
        assert!((config.lifecycle_ack_timeout_s - 10.0).abs() < f64::EPSILON);
        assert_eq!(config.max_turns, None);
        assert_eq!(config.max_buffer_size, None);
    }

    #[test]
    fn from_json_accepts_an_empty_mapping() {
        let config = ClaudeCodeHarnessConfig::from_json(&object(json!({}))).expect("defaults");
        assert_eq!(config, ClaudeCodeHarnessConfig::default());
    }

    #[test]
    fn from_json_rejects_unknown_keys_sorted() {
        let error = ClaudeCodeHarnessConfig::from_json(&object(json!({"zeta": 1, "extra": 2})))
            .expect_err("unknown keys are rejected");
        assert_eq!(
            error.to_string(),
            "validation: unknown Claude Code configuration fields: extra, zeta"
        );
    }

    #[test]
    fn from_json_rejects_an_empty_session_id() {
        let error = ClaudeCodeHarnessConfig::from_json(&object(json!({"session_id": ""})))
            .expect_err("an empty session id is invalid");
        assert_eq!(
            error.to_string(),
            "validation: Claude session_id must not be empty"
        );
    }

    #[test]
    fn from_json_decodes_permission_and_model_fields() {
        let config = ClaudeCodeHarnessConfig::from_json(&object(json!({
            "permission_mode": "acceptEdits",
            "system_prompt_mode": "replace",
            "model": {"model": "sonnet", "effort": "high"},
            "max_buffer_size": DEFAULT_CLAUDE_MAX_BUFFER_SIZE,
            "settings": {"sandbox": {"enabled": true}},
        })))
        .expect("a valid configuration");
        assert_eq!(config.permission_mode, ClaudePermissionMode::AcceptEdits);
        assert_eq!(config.system_prompt_mode, ClaudeSystemPromptMode::Replace);
        assert_eq!(
            config.model,
            Some(ClaudeModelConfig {
                model: Some("sonnet".to_owned()),
                effort: Some("high".to_owned()),
                ..ClaudeModelConfig::default()
            })
        );
        assert_eq!(config.max_buffer_size, Some(DEFAULT_CLAUDE_MAX_BUFFER_SIZE));
        assert!(config.settings.is_some());
    }

    #[test]
    fn from_json_rejects_an_unknown_permission_mode() {
        let error = ClaudeCodeHarnessConfig::from_json(&object(json!({"permission_mode": "yolo"})))
            .expect_err("an unknown permission mode is rejected");
        assert_eq!(
            error.to_string(),
            "validation: Claude permission_mode must be one of default, acceptEdits, plan, \
             bypassPermissions, dontAsk, auto"
        );
    }

    #[test]
    fn model_config_rejects_empty_strings() {
        let error = ClaudeModelConfig::from_json(&object(json!({"api_key": ""})))
            .expect_err("an empty key is invalid");
        assert_eq!(
            error.to_string(),
            "validation: Claude model api_key must be a non-empty string when provided"
        );
    }

    #[test]
    fn model_config_maps_model_and_fallback_without_the_endpoint_provider() {
        let choice = ModelChoice {
            model: Some("sonnet".to_owned()),
            effort: None,
            endpoint: Some(ModelEndpoint {
                api_base: Some("https://example.test".to_owned()),
                api_key: Some("sk-test".to_owned()),
                provider: Some("anthropic".to_owned()),
            }),
        };
        let fallback = ModelChoice {
            model: Some("haiku".to_owned()),
            effort: None,
            endpoint: None,
        };
        let fragment = model_config(&choice, Some(&fallback)).expect("a model fragment");
        assert_eq!(
            fragment.get("model"),
            Some(&json!({
                "model": "sonnet",
                "api_base": "https://example.test",
                "api_key": "sk-test",
            }))
        );
        assert_eq!(
            fragment.get("fallback_model"),
            Some(&json!({"model": "haiku"}))
        );
    }

    #[test]
    fn create_reports_the_missing_sdk() {
        let error = provider()
            .create(object(json!({})))
            .err()
            .expect("the SDK is required");
        assert_eq!(error.kind(), HarnessErrorKind::UnsupportedCapability);
        assert!(error.message().contains("claude-agent-sdk"));
    }

    #[test]
    fn create_surfaces_configuration_errors_first() {
        let error = provider()
            .create(object(json!({"unknown": true})))
            .err()
            .expect("unknown keys are rejected");
        assert_eq!(error.kind(), HarnessErrorKind::Validation);
    }

    #[test]
    fn debug_redacts_environment_maps() {
        let config = ClaudeCodeHarnessConfig::from_json(&object(json!({
            "env": {"CLAUDE_TOKEN": "secret-value"},
            "settings_env": {"OTHER_TOKEN": "another-secret"},
        })))
        .expect("a valid configuration");
        let rendered = format!("{config:?}");
        assert!(!rendered.contains("secret-value"));
        assert!(!rendered.contains("another-secret"));
        assert!(rendered.contains("<redacted>"));
    }
}
