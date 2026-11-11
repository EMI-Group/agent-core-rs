//! Session configuration: built-in defaults, environment overrides, and
//! per-workflow/per-agent metadata.
//!
//! Python parity: [`SessionConfig`] maps to `Config` in
//! `openjiuwen/core/session/config/base.py` (v0.1.20). Python stores a
//! `dict` of environment entries and *also* exposes the timeout knobs as keys of
//! that dict; the Rust model makes each knob a typed field and keeps
//! [`SessionConfig::envs`] as the raw-string view so callers that need the
//! original environment values can still read them.
//!
//! Deviations from Python (intentional, documented):
//!
//! - Timeouts are integers of **milliseconds**. Python stores seconds for
//!   `WORKFLOW_EXECUTE_TIMEOUT` and float seconds for the stream timeouts; the Rust
//!   runtime uses millisecond deadlines, and the environment variables are read as
//!   milliseconds directly (no conversion).
//! - `FORCE_DEL_WORKFLOW_STATE` accepts only the literal strings `"true"` and
//!   `"false"` (case-sensitive). Python lowercases the value, so `"TRUE"` would
//!   enable the flag there.
//! - There is no ambient `workflow_session_vars` `ContextVar`; only the process
//!   environment is consulted.
//!
//! ```
//! use openjiuwen_session::config::SessionConfig;
//!
//! let mut config = SessionConfig::default();
//! assert_eq!(config.execute_timeout_ms, 60_000);
//! config.set_env("CUSTOM_KEY", serde_json::json!(42));
//! assert_eq!(config.get_env("CUSTOM_KEY"), Some(&serde_json::json!(42)));
//! ```

use std::collections::BTreeMap;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::warn;

use crate::constants::{
    ENV_COMP_STREAM_CALL_TIMEOUT, ENV_FORCE_DEL_WORKFLOW_STATE, ENV_LOOP_NUMBER_MAX_LIMIT,
    ENV_STREAM_INPUT_GEN_TIMEOUT, ENV_WORKFLOW_EXECUTE_TIMEOUT,
    ENV_WORKFLOW_STREAM_FIRST_FRAME_TIMEOUT, ENV_WORKFLOW_STREAM_FRAME_TIMEOUT,
};

/// `tracing` target for configuration diagnostics.
const LOG_TARGET: &str = "openjiuwen_session::config";

/// Session-scoped configuration shared by every workflow, agent, and team run.
///
/// The struct is serializable and tolerant of unknown fields, so a configuration
/// document can grow without breaking older readers. [`SessionConfig::default`]
/// carries the Python built-in values (see the field docs); [`SessionConfig::new`]
/// and [`SessionConfig::from_env`] additionally apply process-environment
/// overrides.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SessionConfig {
    /// Raw environment entries.
    ///
    /// [`SessionConfig::from_env`] seeds this map with the seven canonical keys
    /// from [`crate::constants`] (the raw string when the variable is set, the
    /// effective default otherwise); [`SessionConfig::set_env`] adds or replaces
    /// arbitrary entries.
    pub envs: BTreeMap<String, Value>,
    /// Per-workflow configuration blobs keyed by workflow id (Python
    /// `_workflow_configs`, holding `WorkflowConfig` objects there).
    pub workflow_configs: BTreeMap<String, Value>,
    /// Opaque agent configuration (Python `_agent_config`, holding `AgentConfig`).
    pub agent_config: Option<Value>,
    /// Component stream-call timeout in milliseconds; `-1` waits forever
    /// (Python `COMP_STREAM_CALL_TIMEOUT`).
    pub comp_stream_call_timeout_ms: i64,
    /// Stream-input generator timeout in milliseconds; `-1` waits forever
    /// (Python `STREAM_INPUT_GEN_TIMEOUT`).
    pub stream_input_generator_timeout_ms: i64,
    /// Timeout for the end-component template branch render, in milliseconds
    /// (Python `END_COMP_TEMPLATE_BATCH_READER_TIMEOUT`).
    pub end_comp_template_branch_render_timeout_ms: u64,
    /// Timeout for awaiting the end-component template render position, in
    /// milliseconds (Python `END_COMP_TEMPLATE_RENDER_POSITION_TIMEOUT`).
    pub end_comp_template_render_position_timeout_ms: u64,
    /// Whole-workflow execution timeout in milliseconds; `60_000` = Python's
    /// 60 seconds (Python `WORKFLOW_EXECUTE_TIMEOUT`).
    pub execute_timeout_ms: u64,
    /// Inter-frame stream timeout in milliseconds; `-1` waits forever
    /// (Python `WORKFLOW_STREAM_FRAME_TIMEOUT`).
    pub stream_frame_timeout_ms: i64,
    /// First-frame stream timeout in milliseconds; `-1` waits forever
    /// (Python `WORKFLOW_STREAM_FIRST_FRAME_TIMEOUT`).
    pub stream_first_frame_timeout_ms: i64,
    /// Maximum loop-component iterations (Python `LOOP_NUMBER_MAX_LIMIT`).
    pub loop_number_max_limit: u64,
    /// Whether workflow state is deleted after a run (Python
    /// `FORCE_DEL_WORKFLOW_STATE`).
    pub force_del_workflow_state: bool,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            envs: BTreeMap::new(),
            workflow_configs: BTreeMap::new(),
            agent_config: None,
            comp_stream_call_timeout_ms: -1,
            stream_input_generator_timeout_ms: -1,
            end_comp_template_branch_render_timeout_ms: 5,
            end_comp_template_render_position_timeout_ms: 5,
            execute_timeout_ms: 60_000,
            stream_frame_timeout_ms: -1,
            stream_first_frame_timeout_ms: -1,
            loop_number_max_limit: 1000,
            force_del_workflow_state: false,
        }
    }
}

impl SessionConfig {
    /// Creates a configuration seeded from the process environment.
    ///
    /// Equivalent to [`SessionConfig::from_env`].
    #[must_use]
    pub fn new() -> Self {
        Self::from_env()
    }

    /// Creates the default configuration and applies the seven environment
    /// overrides from [`crate::constants`].
    ///
    /// Each variable is read as a plain integer (milliseconds for the timeout
    /// knobs) or, for [`ENV_FORCE_DEL_WORKFLOW_STATE`], as the literal string
    /// `"true"`/`"false"`. A variable that is unset, unparsable, or holds an
    /// unexpected boolean spelling leaves the default in place and emits a
    /// `tracing::warn!` on target `openjiuwen_session::config`.
    ///
    /// Afterwards [`SessionConfig::envs`] holds the seven canonical keys: the raw
    /// environment string when the variable is present (even when it failed to
    /// parse) and the effective default value otherwise.
    #[must_use]
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Some(value) = env_u64(ENV_WORKFLOW_EXECUTE_TIMEOUT) {
            config.execute_timeout_ms = value;
        }
        if let Some(value) = env_i64(ENV_WORKFLOW_STREAM_FRAME_TIMEOUT) {
            config.stream_frame_timeout_ms = value;
        }
        if let Some(value) = env_i64(ENV_WORKFLOW_STREAM_FIRST_FRAME_TIMEOUT) {
            config.stream_first_frame_timeout_ms = value;
        }
        if let Some(value) = env_i64(ENV_COMP_STREAM_CALL_TIMEOUT) {
            config.comp_stream_call_timeout_ms = value;
        }
        if let Some(value) = env_i64(ENV_STREAM_INPUT_GEN_TIMEOUT) {
            config.stream_input_generator_timeout_ms = value;
        }
        if let Some(value) = env_u64(ENV_LOOP_NUMBER_MAX_LIMIT) {
            config.loop_number_max_limit = value;
        }
        if let Some(value) = env_bool(ENV_FORCE_DEL_WORKFLOW_STATE) {
            config.force_del_workflow_state = value;
        }

        config.sync_envs();
        config
    }

    /// Inserts or replaces an environment entry.
    pub fn set_env(&mut self, key: impl Into<String>, value: Value) {
        self.envs.insert(key.into(), value);
    }

    /// Returns the environment entry stored under `key`, if any.
    #[must_use]
    pub fn get_env(&self, key: &str) -> Option<&Value> {
        self.envs.get(key)
    }

    /// Returns a copy of every environment entry.
    #[must_use]
    pub fn get_envs(&self) -> BTreeMap<String, Value> {
        self.envs.clone()
    }

    /// Sets the opaque agent configuration.
    pub fn set_agent_config(&mut self, value: Value) {
        self.agent_config = Some(value);
    }

    /// Returns the opaque agent configuration, if one was set.
    #[must_use]
    pub fn agent_config(&self) -> Option<&Value> {
        self.agent_config.as_ref()
    }

    /// Adds or replaces the configuration of one workflow.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `workflow_id` is empty or whitespace
    /// only (Python raises `ValueError` for a `None` id).
    pub fn add_workflow_config(
        &mut self,
        workflow_id: impl Into<String>,
        config: Value,
    ) -> Result<()> {
        let workflow_id = workflow_id.into();
        if workflow_id.trim().is_empty() {
            return Err(Error::validation("workflow id must not be empty"));
        }
        self.workflow_configs.insert(workflow_id, config);
        Ok(())
    }

    /// Returns the configuration of `workflow_id`, if one was added.
    #[must_use]
    pub fn workflow_config(&self, workflow_id: &str) -> Option<&Value> {
        self.workflow_configs.get(workflow_id)
    }

    /// Records the seven canonical environment keys in [`SessionConfig::envs`].
    ///
    /// Present variables keep their raw string; absent ones record the effective
    /// (default) value of the corresponding field.
    fn sync_envs(&mut self) {
        let fallbacks = [
            (
                ENV_WORKFLOW_EXECUTE_TIMEOUT,
                Value::from(self.execute_timeout_ms),
            ),
            (
                ENV_WORKFLOW_STREAM_FRAME_TIMEOUT,
                Value::from(self.stream_frame_timeout_ms),
            ),
            (
                ENV_WORKFLOW_STREAM_FIRST_FRAME_TIMEOUT,
                Value::from(self.stream_first_frame_timeout_ms),
            ),
            (
                ENV_COMP_STREAM_CALL_TIMEOUT,
                Value::from(self.comp_stream_call_timeout_ms),
            ),
            (
                ENV_STREAM_INPUT_GEN_TIMEOUT,
                Value::from(self.stream_input_generator_timeout_ms),
            ),
            (
                ENV_LOOP_NUMBER_MAX_LIMIT,
                Value::from(self.loop_number_max_limit),
            ),
            (
                ENV_FORCE_DEL_WORKFLOW_STATE,
                Value::from(self.force_del_workflow_state),
            ),
        ];
        for (key, fallback) in fallbacks {
            let value = env_string(key).map_or(fallback, Value::String);
            self.envs.insert(key.to_string(), value);
        }
    }
}

/// Reads `name` from the process environment, returning `None` when it is unset
/// (or not valid UTF-8).
fn env_string(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// Warns that `name` held an unusable value and the default is kept.
fn warn_invalid(name: &str, value: &str, expected: &str) {
    warn!(
        target: LOG_TARGET,
        env = name,
        value = value,
        expected = expected,
        "invalid session environment value, keeping default"
    );
}

/// Parses `name` as an `i64`, warning and returning `None` when it is present but
/// malformed.
fn env_i64(name: &str) -> Option<i64> {
    let raw = env_string(name)?;
    raw.trim().parse::<i64>().map_or_else(
        |_| {
            warn_invalid(name, &raw, "integer");
            None
        },
        Some,
    )
}

/// Parses `name` as a `u64`, warning and returning `None` when it is present but
/// malformed or negative.
fn env_u64(name: &str) -> Option<u64> {
    let raw = env_string(name)?;
    raw.trim().parse::<u64>().map_or_else(
        |_| {
            warn_invalid(name, &raw, "non-negative integer");
            None
        },
        Some,
    )
}

/// Parses `name` as a boolean, accepting only `"true"` and `"false"`.
fn env_bool(name: &str) -> Option<bool> {
    let raw = env_string(name)?;
    match raw.as_str() {
        "true" => Some(true),
        "false" => Some(false),
        _ => {
            warn_invalid(name, &raw, "\"true\" or \"false\"");
            None
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn default_values_match_python_builtins() {
        let config = SessionConfig::default();
        assert!(config.envs.is_empty());
        assert!(config.workflow_configs.is_empty());
        assert_eq!(config.agent_config, None);
        assert_eq!(config.comp_stream_call_timeout_ms, -1);
        assert_eq!(config.stream_input_generator_timeout_ms, -1);
        assert_eq!(config.end_comp_template_branch_render_timeout_ms, 5);
        assert_eq!(config.end_comp_template_render_position_timeout_ms, 5);
        assert_eq!(config.execute_timeout_ms, 60_000);
        assert_eq!(config.stream_frame_timeout_ms, -1);
        assert_eq!(config.stream_first_frame_timeout_ms, -1);
        assert_eq!(config.loop_number_max_limit, 1000);
        assert!(!config.force_del_workflow_state);
    }

    #[test]
    fn serde_default_tolerates_missing_and_unknown_fields() {
        let config: SessionConfig =
            serde_json::from_value(serde_json::json!({"unknown_key": 1})).expect("tolerant");
        assert_eq!(config, SessionConfig::default());
    }

    #[test]
    fn add_workflow_config_rejects_blank_ids() {
        let mut config = SessionConfig::default();
        let error = config
            .add_workflow_config("  ", serde_json::json!({}))
            .expect_err("blank id");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
    }
}
