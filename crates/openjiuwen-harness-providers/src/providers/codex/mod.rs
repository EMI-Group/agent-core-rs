//! The Codex provider: card, configuration data, model mapping and the app-server harness.
//!
//! Ports Python `openjiuwen.harness_providers.codex` (v0.1.20): [`CodexHarnessConfig`] mirrors
//! `CodexHarnessConfig` in `config.py`, [`CodexModelConfig`] mirrors `CodexModelConfig`,
//! [`model_config`] mirrors the provider's `model_config` classmethod, and [`CodexDriver`] is the
//! subprocess harness the provider's `create` returns.
//!
//! # Structure
//!
//! - this module — the configuration data, the approval/system-prompt enums, the card, the
//!   provider and the model mapping;
//! - `harness` — [`CodexDriver`], the transport that spawns `codex app-server`, performs the
//!   JSON-RPC handshake, opens or resumes the thread and runs the turn loop;
//! - `wire` — the argv (configuration overrides, MCP servers), the process environment and the
//!   thread options;
//! - `frames` — the projection of app-server notifications and server requests onto the harness
//!   SPI.
//!
//! # Transport contract
//!
//! The harness speaks newline-delimited JSON-RPC 2.0 to `codex app-server --listen stdio://` over
//! the shared [`JsonRpcClient`](crate::vendor_cli::JsonRpcClient):
//!
//! - the executable is `codex_bin` when set, otherwise the `codex` on `PATH`; a missing binary is
//!   a typed `Protocol` error naming `codex`, never an unimplemented capability;
//! - one `-c key=value` override per configuration entry: the external endpoint's
//!   `model_provider` / `model_providers.<key>.{name,base_url,env_key}` group, the
//!   `features.default_mode_request_user_input` flag when the host declared user input, and one
//!   `mcp_servers.<key>.{command,args,url}` group per host MCP server;
//! - `initialize` + `initialized`, then `thread/start` (or `thread/resume` with a checkpoint's
//!   `thread_id`) carrying the model, the developer instructions, the approval policy and sandbox
//!   mode, and the thread configuration;
//! - `turn/start` per turn, then notifications until `turn/completed`: message and reasoning
//!   deltas, tool items (`shell`, `dynamicToolCall`, `apply_patch`, `<server>.<tool>`), usage
//!   updates, errors and every other notification as a `codex` provider event;
//! - the app-server's server-to-client requests are answered through the host's tool-approval and
//!   user-input policies; dynamic host tools stay unsupported;
//! - `turn/steer` steers, `turn/interrupt` aborts, and the thread id is the provider session id
//!   carried by every checkpoint as `{"thread_id": …, "resumed": …}`.
//!
//! The full contract is documented on the [`harness`] module.
//!
//! # Python parity
//!
//! - [`CodexSystemPromptMode`] and [`CodexApprovalMode`] are typed enums in place of Python's
//!   `Literal`/string fields. The wire values and error wording are Python's, including the
//!   `system_prompt_mode` error that deliberately omits the `Codex` prefix.
//! - `thread_config` keeps raw JSON values; its default is `{"model_reasoning_summary":
//!   "detailed"}` like Python's `field(default_factory=…)`.
//! - The plain serde decoders tolerate unknown fields (the crate-wide rule) while
//!   [`CodexHarnessConfig::from_json`] — the provider SPI path — rejects them like Python's
//!   `from_mapping`.
//! - `api_key` is a [`SecretString`] so `Debug` redacts it, and the harness configuration's manual
//!   `Debug` redacts the `env` map (`repr=False` in Python).
use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::security::SecretString;
use openjiuwen_harness_protocol::{
    json_object_from_value, EventBufferConfig, EventOverflowPolicy, HarnessCapability, HarnessCard,
    HarnessProtocol, HarnessProvider, HarnessResult, HostCapability, JsonObject, JsonValue,
    ModelChoice, ModelConfigProvider,
};
use serde::{Deserialize, Serialize};

use crate::decode::{
    expect_object, optional_bool, optional_f64, optional_str, optional_string_list,
    optional_string_map, optional_u32, optional_u64, reject_unknown_keys,
};
use crate::skills::{parse_skills_value, SkillConflict, SkillSource};
use crate::vendor_cli::{resolve_binary, BinarySpec, VendorHarness};

use super::vendor;

mod frames;
mod harness;
mod wire;

pub use harness::CodexDriver;

// The shared `vendor` module still owns the Python "SDK required" text and its own unit tests
// assert it; this provider no longer produces that message, so keep the constant referenced in
// non-test builds (removing it is the vendor module's call, not this provider's).
const _: &str = crate::providers::vendor::CODEX_SDK_REQUIRED;

/// Registry name of the Codex provider.
pub const PROVIDER_NAME: &str = "codex";

/// Card name the Codex provider reports (Python `mapping.PROVIDER_NAME`).
pub const CARD_NAME: &str = "codex";

/// Version of the adapter whose card is reported here (Python `ADAPTER_VERSION`).
pub const IMPLEMENTATION_VERSION: &str = "0.1.0";

/// Configuration keys [`CodexHarnessConfig::from_json`] accepts (Python `fields(cls)`).
const CONFIG_FIELDS: [&str; 24] = [
    "system_prompt_mode",
    "skills",
    "skill_conflict",
    "cwd",
    "env",
    "inherit_process_env",
    "codex_bin",
    "model",
    "fallback_model",
    "config_overrides",
    "thread_config",
    "bypass_approvals_and_sandbox",
    "turn_idle_timeout_s",
    "turn_idle_retries",
    "max_will_retry_count",
    "mcp_env_passthrough",
    "mcp_startup_timeout_s",
    "mcp_required",
    "mcp_default_tools_approval_mode",
    "client_name",
    "client_title",
    "experimental_raw_events",
    "event_buffer_capacity",
    "request_observation_wait_s",
];

/// Configuration keys [`CodexModelConfig::from_json`] accepts (Python `fields(cls)`).
const MODEL_CONFIG_FIELDS: [&str; 5] = ["model", "provider", "api_base", "api_key", "effort"];

/// Approval mode of one Codex tool class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodexApprovalMode {
    /// Approve without prompting.
    #[serde(rename = "auto")]
    Auto,
    /// Prompt the host for every call.
    #[serde(rename = "prompt")]
    Prompt,
    /// Prompt only for calls that write.
    #[serde(rename = "writes")]
    Writes,
    /// Route the call through the host's approval flow.
    #[serde(rename = "approve")]
    Approve,
}

impl Default for CodexApprovalMode {
    /// Returns [`CodexApprovalMode::Auto`]; the value is only used when explicitly configured.
    fn default() -> Self {
        Self::Auto
    }
}

/// How the host's system prompt combines with the Codex developer instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodexSystemPromptMode {
    /// Append the host prompt to what the CLI was configured with.
    #[serde(rename = "append")]
    Append,
    /// Replace the effective developer instructions field.
    #[serde(rename = "replace")]
    Replace,
}

impl Default for CodexSystemPromptMode {
    /// Returns [`CodexSystemPromptMode::Append`], Python's `system_prompt_mode` default.
    fn default() -> Self {
        Self::Append
    }
}

/// Model selection for the Codex thread, optionally on an external endpoint.
///
/// `provider` / `api_base` / `api_key` render Codex `model_provider` overrides for an external
/// endpoint. Without them Codex runs on its own login (for example a `ChatGPT` subscription) and
/// `model` picks one of its built-in models. `effort` is the Codex `model_reasoning_effort`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct CodexModelConfig {
    /// Model identifier.
    #[serde(default)]
    pub model: Option<String>,
    /// Wire-protocol label of the external endpoint's provider.
    #[serde(default)]
    pub provider: Option<String>,
    /// API base URL of the external endpoint.
    #[serde(default)]
    pub api_base: Option<String>,
    /// API key of the external endpoint; redacted by [`SecretString`]'s `Debug` and `Display`.
    #[serde(default)]
    pub api_key: Option<SecretString>,
    /// Codex `model_reasoning_effort`.
    #[serde(default)]
    pub effort: Option<String>,
}

impl CodexModelConfig {
    /// Decodes a model configuration, rejecting unknown keys like Python's `from_mapping`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an unknown key (`"unknown Codex model config fields: …"`),
    /// a wrong field type, or a provided string field that is empty
    /// (`"Codex model {field} must be a non-empty string when provided"`).
    pub fn from_json(object: &JsonObject) -> Result<Self> {
        reject_unknown_keys(
            object,
            &MODEL_CONFIG_FIELDS,
            "unknown Codex model config fields",
        )?;
        let config = Self {
            model: optional_str(object, "model")?,
            provider: optional_str(object, "provider")?,
            api_base: optional_str(object, "api_base")?,
            api_key: optional_str(object, "api_key")?.map(SecretString::new),
            effort: optional_str(object, "effort")?,
        };
        for (name, value) in [
            ("model", config.model.as_deref()),
            ("provider", config.provider.as_deref()),
            ("api_base", config.api_base.as_deref()),
            (
                "api_key",
                config.api_key.as_ref().map(SecretString::expose_secret),
            ),
            ("effort", config.effort.as_deref()),
        ] {
            if value == Some("") {
                return Err(Error::validation(format!(
                    "Codex model {name} must be a non-empty string when provided"
                )));
            }
        }
        Ok(config)
    }

    /// Returns whether this targets an endpoint other than Codex's own login.
    #[must_use]
    pub fn is_external(&self) -> bool {
        self.provider.is_some() || self.api_base.is_some()
    }
}

/// Configuration of the Codex harness (Python `CodexHarnessConfig`).
///
/// The optional SDK is deliberately not imported by this decoder, so the configuration can be
/// validated offline. `config_overrides` are appended verbatim to the `CodexConfig` overrides after
/// the ones the harness renders itself (model endpoint, MCP servers).
// Python's configuration exposes these independent switches; keeping the parity field set is more
// useful than folding them into one enum.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct CodexHarnessConfig {
    /// How the host system prompt combines with the developer instructions.
    #[serde(default)]
    pub system_prompt_mode: CodexSystemPromptMode,
    /// Skill sources installed into the CLI's project-local discovery directory.
    #[serde(default)]
    pub skills: Vec<SkillSource>,
    /// Conflict policy applied when a destination skill already exists.
    #[serde(default = "default_skill_conflict")]
    pub skill_conflict: SkillConflict,
    /// Working directory of the Codex process.
    #[serde(default)]
    pub cwd: Option<PathBuf>,
    /// Environment variables added to the Codex process; redacted in `Debug`.
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    /// Whether the Codex process inherits the host process environment.
    #[serde(default = "default_true")]
    pub inherit_process_env: bool,
    /// Path of the `codex` executable; `None` uses the SDK's lookup.
    #[serde(default)]
    pub codex_bin: Option<PathBuf>,
    /// Model endpoint used for the thread.
    #[serde(default)]
    pub model: Option<CodexModelConfig>,
    /// Model endpoint Codex switches to after an authentication failure.
    #[serde(default)]
    pub fallback_model: Option<CodexModelConfig>,
    /// Extra `CodexConfig` overrides rendered verbatim.
    #[serde(default)]
    pub config_overrides: Vec<String>,
    /// Per-thread Codex configuration; defaults to `{"model_reasoning_summary": "detailed"}`.
    #[serde(default = "default_thread_config")]
    pub thread_config: BTreeMap<String, JsonValue>,
    /// Skip Codex's approval and sandbox machinery entirely.
    #[serde(default)]
    pub bypass_approvals_and_sandbox: bool,
    /// How long one turn may stay idle before the adapter retries it.
    #[serde(default = "default_turn_idle_timeout_s")]
    pub turn_idle_timeout_s: f64,
    /// How often an idle turn is retried.
    #[serde(default = "default_turn_idle_retries")]
    pub turn_idle_retries: u32,
    /// How often a `will` request is retried.
    #[serde(default = "default_max_will_retry_count")]
    pub max_will_retry_count: u32,
    /// Environment variables forwarded to MCP servers.
    #[serde(default)]
    pub mcp_env_passthrough: Vec<String>,
    /// Startup timeout of the MCP servers in seconds.
    #[serde(default = "default_mcp_startup_timeout_s")]
    pub mcp_startup_timeout_s: u64,
    /// Fail startup when an MCP server cannot be reached.
    #[serde(default = "default_true")]
    pub mcp_required: bool,
    /// Default approval mode of the MCP tools.
    #[serde(default)]
    pub mcp_default_tools_approval_mode: Option<CodexApprovalMode>,
    /// Client name Codex reports in its app-server handshake.
    #[serde(default = "default_client_name")]
    pub client_name: String,
    /// Client title Codex reports in its app-server handshake.
    #[serde(default = "default_client_title")]
    pub client_title: String,
    /// Subscribe to the app-server's raw event stream.
    #[serde(default = "default_true")]
    pub experimental_raw_events: bool,
    /// Capacity of the event buffer the harness publishes.
    #[serde(default = "default_event_buffer_capacity")]
    pub event_buffer_capacity: u64,
    /// How long a tool item or turn end waits for the rollout record of a model request before the
    /// request is reported from raw events alone.
    #[serde(default = "default_request_observation_wait_s")]
    pub request_observation_wait_s: f64,
}

impl Default for CodexHarnessConfig {
    /// Returns the Python defaults: `append` system prompt mode, a detailed reasoning summary, a
    /// 180-second idle timeout with one retry, five `will` retries, required MCP servers, and a
    /// 1024-event buffer.
    fn default() -> Self {
        Self {
            system_prompt_mode: CodexSystemPromptMode::default(),
            skills: Vec::new(),
            skill_conflict: SkillConflict::Skip,
            cwd: None,
            env: BTreeMap::new(),
            inherit_process_env: true,
            codex_bin: None,
            model: None,
            fallback_model: None,
            config_overrides: Vec::new(),
            thread_config: default_thread_config(),
            bypass_approvals_and_sandbox: false,
            turn_idle_timeout_s: default_turn_idle_timeout_s(),
            turn_idle_retries: default_turn_idle_retries(),
            max_will_retry_count: default_max_will_retry_count(),
            mcp_env_passthrough: Vec::new(),
            mcp_startup_timeout_s: default_mcp_startup_timeout_s(),
            mcp_required: true,
            mcp_default_tools_approval_mode: None,
            client_name: default_client_name(),
            client_title: default_client_title(),
            experimental_raw_events: true,
            event_buffer_capacity: default_event_buffer_capacity(),
            request_observation_wait_s: default_request_observation_wait_s(),
        }
    }
}

impl CodexHarnessConfig {
    /// Decodes a provider SPI mapping, validating it like Python's `from_mapping`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an unknown key (`"unknown Codex configuration fields:
    /// …"`), a wrongly typed field, a non-object `thread_config`, an unknown
    /// `mcp_default_tools_approval_mode`, a non-positive `turn_idle_timeout_s`, or a non-positive
    /// `mcp_startup_timeout_s` / `event_buffer_capacity`.
    pub fn from_json(object: &JsonObject) -> Result<Self> {
        reject_unknown_keys(object, &CONFIG_FIELDS, "unknown Codex configuration fields")?;
        let defaults = Self::default();
        let system_prompt_mode = match optional_str(object, "system_prompt_mode")? {
            Some(value) => parse_system_prompt_mode(&value)?,
            None => defaults.system_prompt_mode,
        };
        let skill_conflict = match optional_str(object, "skill_conflict")? {
            Some(value) => value.parse::<SkillConflict>()?,
            None => defaults.skill_conflict,
        };
        let mcp_default_tools_approval_mode =
            match optional_str(object, "mcp_default_tools_approval_mode")? {
                Some(value) => Some(parse_approval_mode(&value)?),
                None => defaults.mcp_default_tools_approval_mode,
            };
        let config = Self {
            system_prompt_mode,
            skills: parse_skills_value(object.get("skills"))?,
            skill_conflict,
            cwd: optional_str(object, "cwd")?.map(PathBuf::from),
            env: optional_string_map(object, "env")?.unwrap_or_default(),
            inherit_process_env: optional_bool(object, "inherit_process_env")?
                .unwrap_or(defaults.inherit_process_env),
            codex_bin: optional_str(object, "codex_bin")?.map(PathBuf::from),
            model: parse_model_config(object, "model")?,
            fallback_model: parse_model_config(object, "fallback_model")?,
            config_overrides: optional_string_list(object, "config_overrides")?.unwrap_or_default(),
            thread_config: parse_thread_config(object, defaults.thread_config)?,
            bypass_approvals_and_sandbox: optional_bool(object, "bypass_approvals_and_sandbox")?
                .unwrap_or(defaults.bypass_approvals_and_sandbox),
            turn_idle_timeout_s: optional_f64(object, "turn_idle_timeout_s")?
                .unwrap_or(defaults.turn_idle_timeout_s),
            turn_idle_retries: optional_u32(object, "turn_idle_retries")?
                .unwrap_or(defaults.turn_idle_retries),
            max_will_retry_count: optional_u32(object, "max_will_retry_count")?
                .unwrap_or(defaults.max_will_retry_count),
            mcp_env_passthrough: optional_string_list(object, "mcp_env_passthrough")?
                .unwrap_or_default(),
            mcp_startup_timeout_s: optional_u64(object, "mcp_startup_timeout_s")?
                .unwrap_or(defaults.mcp_startup_timeout_s),
            mcp_required: optional_bool(object, "mcp_required")?.unwrap_or(defaults.mcp_required),
            mcp_default_tools_approval_mode,
            client_name: optional_str(object, "client_name")?.unwrap_or(defaults.client_name),
            client_title: optional_str(object, "client_title")?.unwrap_or(defaults.client_title),
            experimental_raw_events: optional_bool(object, "experimental_raw_events")?
                .unwrap_or(defaults.experimental_raw_events),
            event_buffer_capacity: optional_u64(object, "event_buffer_capacity")?
                .unwrap_or(defaults.event_buffer_capacity),
            request_observation_wait_s: optional_f64(object, "request_observation_wait_s")?
                .unwrap_or(defaults.request_observation_wait_s),
        };
        config.validate()?;
        Ok(config)
    }

    /// Applies the cross-field rules of Python's `__post_init__`.
    fn validate(&self) -> Result<()> {
        if self.turn_idle_timeout_s <= 0.0 {
            return Err(Error::validation(
                "Codex turn_idle_timeout_s must be greater than zero",
            ));
        }
        if self.mcp_startup_timeout_s == 0 || self.event_buffer_capacity == 0 {
            return Err(Error::validation(
                "Codex mcp_startup_timeout_s and event_buffer_capacity must be positive",
            ));
        }
        Ok(())
    }
}

impl fmt::Debug for CodexHarnessConfig {
    /// Formats the configuration with the `env` values redacted (Python's `repr=False`).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CodexHarnessConfig")
            .field("system_prompt_mode", &self.system_prompt_mode)
            .field("skills", &self.skills)
            .field("skill_conflict", &self.skill_conflict)
            .field("cwd", &self.cwd)
            .field("env", &"<redacted>")
            .field("inherit_process_env", &self.inherit_process_env)
            .field("codex_bin", &self.codex_bin)
            .field("model", &self.model)
            .field("fallback_model", &self.fallback_model)
            .field("config_overrides", &self.config_overrides)
            .field("thread_config", &self.thread_config)
            .field(
                "bypass_approvals_and_sandbox",
                &self.bypass_approvals_and_sandbox,
            )
            .field("turn_idle_timeout_s", &self.turn_idle_timeout_s)
            .field("turn_idle_retries", &self.turn_idle_retries)
            .field("max_will_retry_count", &self.max_will_retry_count)
            .field("mcp_env_passthrough", &self.mcp_env_passthrough)
            .field("mcp_startup_timeout_s", &self.mcp_startup_timeout_s)
            .field("mcp_required", &self.mcp_required)
            .field(
                "mcp_default_tools_approval_mode",
                &self.mcp_default_tools_approval_mode,
            )
            .field("client_name", &self.client_name)
            .field("client_title", &self.client_title)
            .field("experimental_raw_events", &self.experimental_raw_events)
            .field("event_buffer_capacity", &self.event_buffer_capacity)
            .field(
                "request_observation_wait_s",
                &self.request_observation_wait_s,
            )
            .finish()
    }
}

/// Returns the Codex provider.
#[must_use]
pub fn provider() -> CodexHarnessProvider {
    CodexHarnessProvider::default()
}

/// Returns the capability card of the Codex adapter.
///
/// The card declares only what the app-server transport really provides through the generic
/// harness shell: steering, graceful abort (Codex has no harder stop than `turn/interrupt`),
/// persistent threads, checkpoints, MCP tools, endpoint models and the authentication fallback to
/// `fallback_model`. Model control (selection and discovery) and pausing stay deferred: the shell
/// exposes no `HarnessModelControl` seam.
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
            HarnessCapability::EndpointModels,
            HarnessCapability::AuthFallback,
        ],
        [
            HostCapability::ToolApproval,
            HostCapability::UserInput,
            HostCapability::CheckpointSink,
            HostCapability::McpServers,
            HostCapability::NativeToolGateway,
            HostCapability::DynamicToolCall,
            HostCapability::ProviderInteraction,
            HostCapability::ModelRequestObservation,
        ],
        &["full", "before"],
    )
}

/// Maps a model choice onto `{"model": {…}}` (plus `{"fallback_model": {…}}`).
///
/// Codex renders `model_provider` overrides for an external endpoint, so the endpoint's `provider`
/// label is preserved (the difference from the Claude Code mapping).
///
/// # Errors
///
/// This mapping cannot fail; the [`HarnessResult`] signature matches the SPI.
pub fn model_config(
    choice: &ModelChoice,
    fallback: Option<&ModelChoice>,
) -> HarnessResult<JsonObject> {
    Ok(vendor::nested_model_config(choice, fallback, true))
}

/// The Codex provider: identity card plus the app-server harness creation.
#[derive(Debug, Clone)]
pub struct CodexHarnessProvider {
    /// Card reported by [`HarnessProvider::card`].
    card: HarnessCard,
}

impl Default for CodexHarnessProvider {
    /// Builds the provider with the card of [`card`].
    fn default() -> Self {
        Self { card: card() }
    }
}

impl HarnessProvider for CodexHarnessProvider {
    /// Returns the Codex adapter card.
    fn card(&self) -> &HarnessCard {
        &self.card
    }

    /// Validates `config`, resolves the `codex` executable and returns an unstarted harness.
    ///
    /// The returned [`VendorHarness`] wraps a [`CodexDriver`], so it reports
    /// [`HarnessState::Terminated`](openjiuwen_harness_protocol::HarnessState::Terminated) until
    /// [`start`](HarnessProtocol::start) succeeds; it never spawns the app-server here.
    ///
    /// # Errors
    ///
    /// Returns `HarnessError::Validation` when `config` is not a valid [`CodexHarnessConfig`], and
    /// `HarnessError::Protocol` naming `codex` and the fix when the executable cannot be resolved.
    /// A missing CLI is an operator configuration problem, never an unimplemented capability.
    fn create(&self, config: JsonObject) -> HarnessResult<Box<dyn HarnessProtocol>> {
        let config =
            CodexHarnessConfig::from_json(&config).map_err(vendor::config_error_to_harness)?;
        let binary = resolve_binary(&BinarySpec {
            name: "codex",
            explicit: config.codex_bin.as_deref(),
            label: "Codex",
            install_hint: "install the Codex CLI (npm install -g @openai/codex)",
        })?;
        let buffer = EventBufferConfig {
            capacity: config.event_buffer_capacity,
            overflow: EventOverflowPolicy::CoalesceOrBlock,
        };
        let driver = Arc::new(CodexDriver::new(config, binary));
        Ok(Box::new(VendorHarness::new(card(), buffer, driver)))
    }
}

impl ModelConfigProvider for CodexHarnessProvider {
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
fn parse_model_config(object: &JsonObject, key: &str) -> Result<Option<CodexModelConfig>> {
    match object.get(key) {
        None | Some(JsonValue::Null) => Ok(None),
        Some(value) => {
            let nested = expect_object(value, "Codex model config")?;
            Ok(Some(CodexModelConfig::from_json(&nested)?))
        }
    }
}

/// Decodes `thread_config`, falling back to `default` when it is absent or `null`.
fn parse_thread_config(
    object: &JsonObject,
    default: BTreeMap<String, JsonValue>,
) -> Result<BTreeMap<String, JsonValue>> {
    match object.get("thread_config") {
        None | Some(JsonValue::Null) => Ok(default),
        Some(value) => json_object_from_value(value.clone())
            .ok_or_else(|| Error::validation("Codex thread_config must be an object")),
    }
}

/// Parses a Python `system_prompt_mode` wire value (the error keeps Python's prefix-free wording).
fn parse_system_prompt_mode(value: &str) -> Result<CodexSystemPromptMode> {
    match value {
        "append" => Ok(CodexSystemPromptMode::Append),
        "replace" => Ok(CodexSystemPromptMode::Replace),
        _ => Err(Error::validation(
            "system_prompt_mode must be 'append' or 'replace'",
        )),
    }
}

/// Parses a Python `mcp_default_tools_approval_mode` wire value.
fn parse_approval_mode(value: &str) -> Result<CodexApprovalMode> {
    match value {
        "auto" => Ok(CodexApprovalMode::Auto),
        "prompt" => Ok(CodexApprovalMode::Prompt),
        "writes" => Ok(CodexApprovalMode::Writes),
        "approve" => Ok(CodexApprovalMode::Approve),
        _ => Err(Error::validation(
            "Codex mcp_default_tools_approval_mode must be one of auto, prompt, writes, approve",
        )),
    }
}

/// Returns [`SkillConflict::Skip`], Python's `skill_conflict` default.
fn default_skill_conflict() -> SkillConflict {
    SkillConflict::Skip
}

/// Returns `true`, Python's default for the process-environment and raw-event flags.
const fn default_true() -> bool {
    true
}

/// Returns the default thread configuration (`{"model_reasoning_summary": "detailed"}`).
fn default_thread_config() -> BTreeMap<String, JsonValue> {
    let mut thread_config = BTreeMap::new();
    thread_config.insert(
        "model_reasoning_summary".to_owned(),
        JsonValue::String("detailed".to_owned()),
    );
    thread_config
}

/// Returns the default turn idle timeout (180 seconds).
const fn default_turn_idle_timeout_s() -> f64 {
    180.0
}

/// Returns the default number of idle-turn retries (1).
const fn default_turn_idle_retries() -> u32 {
    1
}

/// Returns the default number of `will` retries (5).
const fn default_max_will_retry_count() -> u32 {
    5
}

/// Returns the default MCP startup timeout (120 seconds).
const fn default_mcp_startup_timeout_s() -> u64 {
    120
}

/// Returns the default app-server client name.
fn default_client_name() -> String {
    "openjiuwen_harness".to_owned()
}

/// Returns the default app-server client title.
fn default_client_title() -> String {
    "OpenJiuwen Harness".to_owned()
}

/// Returns the default event-buffer capacity (1024 events).
const fn default_event_buffer_capacity() -> u64 {
    1024
}

/// Returns the default request-observation wait (5 seconds).
const fn default_request_observation_wait_s() -> f64 {
    5.0
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use openjiuwen_harness_protocol::{
        HarnessCapability, HarnessErrorKind, HarnessProvider, ModelChoice, ModelEndpoint,
    };
    use serde_json::json;

    use crate::decode::expect_object;

    use super::{
        card, model_config, provider, CodexApprovalMode, CodexHarnessConfig, CodexModelConfig,
        CodexSystemPromptMode, CARD_NAME, IMPLEMENTATION_VERSION, PROVIDER_NAME,
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
        assert_eq!(PROVIDER_NAME, "codex");
        assert_eq!(CARD_NAME, "codex");
        assert_eq!(IMPLEMENTATION_VERSION, "0.1.0");
    }

    #[test]
    fn card_declares_only_what_the_app_server_transport_provides() {
        let card = card();
        assert_eq!(card.name, "codex");
        for capability in [
            HarnessCapability::Steer,
            HarnessCapability::GracefulAbort,
            HarnessCapability::PersistentSession,
            HarnessCapability::Checkpoint,
            HarnessCapability::McpTools,
            HarnessCapability::EndpointModels,
            HarnessCapability::AuthFallback,
        ] {
            assert!(card.supports(capability), "{capability:?} must be declared");
        }
        for deferred in [
            HarnessCapability::NativeTools,
            HarnessCapability::ForceAbort,
            HarnessCapability::PauseResume,
            HarnessCapability::ModelSelection,
            HarnessCapability::ModelDiscovery,
        ] {
            assert!(
                !card.supports(deferred),
                "{deferred:?} is not provided by the transport"
            );
        }
        assert!(card.required_host_capabilities.is_empty());
        assert_eq!(provider().card(), &card);
    }

    #[test]
    fn default_configuration_matches_python() {
        let config = CodexHarnessConfig::default();
        assert_eq!(config.system_prompt_mode, CodexSystemPromptMode::Append);
        assert!(config.inherit_process_env);
        assert_eq!(
            config.thread_config.get("model_reasoning_summary"),
            Some(&json!("detailed"))
        );
        assert!(config.mcp_required);
        assert!(config.experimental_raw_events);
        assert!((config.turn_idle_timeout_s - 180.0).abs() < f64::EPSILON);
        assert_eq!(config.turn_idle_retries, 1);
        assert_eq!(config.max_will_retry_count, 5);
        assert_eq!(config.mcp_startup_timeout_s, 120);
        assert_eq!(config.event_buffer_capacity, 1024);
        assert_eq!(config.client_name, "openjiuwen_harness");
        assert_eq!(config.client_title, "OpenJiuwen Harness");
    }

    #[test]
    fn from_json_accepts_an_empty_mapping() {
        let config = CodexHarnessConfig::from_json(&object(json!({}))).expect("defaults");
        assert_eq!(config, CodexHarnessConfig::default());
    }

    #[test]
    fn from_json_rejects_unknown_keys_sorted() {
        let error = CodexHarnessConfig::from_json(&object(json!({"zeta": 1, "foo": 2})))
            .expect_err("unknown keys are rejected");
        assert_eq!(
            error.to_string(),
            "validation: unknown Codex configuration fields: foo, zeta"
        );
    }

    #[test]
    fn from_json_rejects_an_unknown_system_prompt_mode_without_the_codex_prefix() {
        let error = CodexHarnessConfig::from_json(&object(json!({"system_prompt_mode": "merge"})))
            .expect_err("an unknown mode is rejected");
        assert_eq!(
            error.to_string(),
            "validation: system_prompt_mode must be 'append' or 'replace'"
        );
    }

    #[test]
    fn from_json_rejects_a_non_positive_turn_idle_timeout() {
        let error = CodexHarnessConfig::from_json(&object(json!({"turn_idle_timeout_s": 0.0})))
            .expect_err("a zero timeout is invalid");
        assert_eq!(
            error.to_string(),
            "validation: Codex turn_idle_timeout_s must be greater than zero"
        );
    }

    #[test]
    fn from_json_decodes_thread_config_and_approval_mode() {
        let config = CodexHarnessConfig::from_json(&object(json!({
            "thread_config": {"model_reasoning_effort": "high"},
            "mcp_default_tools_approval_mode": "writes",
            "config_overrides": ["model_reasoning_summary=\"concise\""],
            "model": {"model": "gpt-5", "provider": "openai", "api_base": "https://example.test"},
        })))
        .expect("a valid configuration");
        assert_eq!(
            config.thread_config.get("model_reasoning_effort"),
            Some(&json!("high"))
        );
        assert_eq!(
            config.mcp_default_tools_approval_mode,
            Some(CodexApprovalMode::Writes)
        );
        assert_eq!(
            config.config_overrides,
            ["model_reasoning_summary=\"concise\""]
        );
        assert!(config
            .model
            .as_ref()
            .is_some_and(CodexModelConfig::is_external));
    }

    #[test]
    fn model_config_preserves_the_endpoint_provider() {
        let choice = ModelChoice {
            model: Some("gpt-5".to_owned()),
            effort: Some("high".to_owned()),
            endpoint: Some(ModelEndpoint {
                api_base: Some("https://example.test/v1".to_owned()),
                api_key: Some("sk-test".to_owned()),
                provider: Some("openai".to_owned()),
            }),
        };
        let fragment = model_config(&choice, None).expect("a model fragment");
        assert_eq!(
            fragment.get("model"),
            Some(&json!({
                "model": "gpt-5",
                "effort": "high",
                "provider": "openai",
                "api_base": "https://example.test/v1",
                "api_key": "sk-test",
            }))
        );
    }

    #[test]
    fn model_config_rejects_empty_strings() {
        let error = CodexModelConfig::from_json(&object(json!({"provider": ""})))
            .expect_err("an empty provider is invalid");
        assert_eq!(
            error.to_string(),
            "validation: Codex model provider must be a non-empty string when provided"
        );
    }

    #[test]
    fn create_rejects_an_invalid_mapping() {
        let error = provider()
            .create(object(json!({"unknown_key": true})))
            .err()
            .expect("an unknown key is invalid");
        assert_eq!(error.kind(), HarnessErrorKind::Validation);
        assert!(error
            .message()
            .contains("unknown Codex configuration fields"));
    }

    #[test]
    fn create_reports_a_missing_binary() {
        let error = provider()
            .create(object(json!({"codex_bin": "/nonexistent/codex"})))
            .err()
            .expect("the CLI is missing");
        assert_eq!(error.kind(), HarnessErrorKind::Protocol);
        assert!(error.message().contains("codex"));
    }

    #[test]
    fn debug_redacts_the_environment() {
        let config = CodexHarnessConfig::from_json(&object(json!({
            "env": {"OPENAI_API_KEY": "secret-value"},
        })))
        .expect("a valid configuration");
        let rendered = format!("{config:?}");
        assert!(!rendered.contains("secret-value"));
        assert!(rendered.contains("<redacted>"));
    }
}
