//! Sandbox gateway configuration types (Python `core/sys_operation/config.py`).
//!
//! These types are always compiled: they are part of [`crate::card::SysOperationCard`]'s
//! serialized shape, so a card written for the sandbox must round-trip even when the
//! remote backend is not built in. The gateway client itself lives in
//! `crate::sandbox::http` behind the `sandbox` feature.
//!
//! ```
//! use openjiuwen_sysop::sandbox::config::{
//!     ContainerScope, SandboxGatewayConfig, generate_isolation_key_template,
//! };
//!
//! let template = generate_isolation_key_template(
//!     Some("agent1_"), ContainerScope::Session, None, "pre_deploy", "aio").unwrap();
//! assert_eq!(template, "session_pre_deploy_aio_agent1_{session_id}");
//! ```

use std::collections::BTreeMap;

use openjiuwen_common::error::{Error, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::constants::{DEFAULT_LAUNCHER_TYPE, DEFAULT_SANDBOX_TYPE};
use crate::status::isolation_key_message;

/// Who owns a sandbox container.
///
/// Python parity: `ContainerScope` in `core/sys_operation/config.py`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum ContainerScope {
    /// One container for the whole process.
    System,
    /// One container per session (default).
    #[default]
    Session,
    /// One container per caller-supplied id.
    Custom,
}

impl ContainerScope {
    /// The serialized name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::Session => "session",
            Self::Custom => "custom",
        }
    }
}

/// What happens to a sandbox when the caller releases it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum OnStop {
    /// Delete the container (default).
    #[default]
    Delete,
    /// Pause the container for reuse.
    Pause,
    /// Leave the container untouched.
    Keep,
}

/// Isolation-key inputs for a sandbox container.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SandboxIsolationConfig {
    /// Identifier used when [`Self::container_scope`] is [`ContainerScope::Custom`].
    pub custom_id: Option<String>,
    /// Scope of the container.
    pub container_scope: ContainerScope,
    /// Literal prefix inserted before the identity segment.
    pub prefix: Option<String>,
}

/// Launcher configuration of a sandbox deployment.
///
/// Python models this as a union of `PreDeployLauncherConfig` and
/// `SandboxLauncherConfig`; Rust keeps a single tolerant struct tagged by
/// `launcher_type`, so unknown launcher types deserialize instead of failing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SandboxLauncherConfig {
    /// Launcher implementation name (`"pre_deploy"` by default).
    pub launcher_type: String,
    /// Gateway base URL for launchers that talk to a gateway.
    pub gateway_url: String,
    /// Sandbox implementation name (`"aio"` by default).
    pub sandbox_type: String,
    /// Lifecycle action applied when the sandbox is released.
    pub on_stop: OnStop,
    /// Idle time after which a container is evicted, in seconds.
    pub idle_ttl_seconds: Option<u64>,
    /// Launcher-specific parameters passed through verbatim.
    pub extra_params: BTreeMap<String, Value>,
    /// Base URL of an already deployed sandbox (`PreDeployLauncherConfig.base_url`).
    pub base_url: Option<String>,
}

impl Default for SandboxLauncherConfig {
    fn default() -> Self {
        Self {
            launcher_type: DEFAULT_LAUNCHER_TYPE.to_string(),
            gateway_url: String::new(),
            sandbox_type: DEFAULT_SANDBOX_TYPE.to_string(),
            on_stop: OnStop::default(),
            idle_ttl_seconds: None,
            extra_params: BTreeMap::new(),
            base_url: None,
        }
    }
}

/// Sandbox backend configuration stored on a [`crate::card::SysOperationCard`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SandboxGatewayConfig {
    /// Container isolation inputs.
    pub isolation: SandboxIsolationConfig,
    /// Launcher configuration; required for every sandbox card.
    pub launcher_config: Option<SandboxLauncherConfig>,
    /// Gateway request timeout in seconds (Python default `30`).
    pub timeout_seconds: u64,
    /// Extra HTTP headers for the gateway.
    pub auth_headers: BTreeMap<String, String>,
    /// Extra query parameters for the gateway.
    pub auth_query_params: BTreeMap<String, String>,
    /// Whether the backend supports compare-and-swap writes.
    pub supports_conditional_write: bool,
}

impl Default for SandboxGatewayConfig {
    fn default() -> Self {
        Self {
            isolation: SandboxIsolationConfig::default(),
            launcher_config: None,
            timeout_seconds: 30,
            auth_headers: BTreeMap::new(),
            auth_query_params: BTreeMap::new(),
            supports_conditional_write: false,
        }
    }
}

/// Runtime pairing of a gateway configuration and its resolved isolation-key template.
///
/// Python parity: `SandboxRunConfig` (`sandbox/run_config.py`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct SandboxRunConfig {
    /// Gateway configuration copied from the card.
    pub config: SandboxGatewayConfig,
    /// Template with an unresolved `{session_id}` placeholder, if session-scoped.
    pub isolation_key_template: String,
}

/// A request forwarded to a sandbox provider.
///
/// Python parity: `GatewayInvokeRequest`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct GatewayInvokeRequest {
    /// Operation kind (`"fs"`, `"shell"`, `"code"`).
    pub op_type: String,
    /// Operation method name (`"read_file"`, `"execute_cmd"`, ...).
    pub method: String,
    /// Method parameters, serialized from the Rust parameter struct.
    pub params: Value,
    /// Resolved isolation key of the target container.
    pub isolation_key: Option<String>,
}

/// A provider response.
///
/// Python parity: `GatewayResponse`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct GatewayResponse {
    /// `0` on success, `-1` on failure (Python gateway convention).
    pub code: i32,
    /// Human-readable message.
    pub message: String,
    /// The provider payload, if any.
    pub data: Option<Value>,
}

impl GatewayResponse {
    /// Returns `true` when [`Self::code`] is `0`.
    #[must_use]
    pub const fn is_success(&self) -> bool {
        self.code == 0
    }
}

/// A resolved sandbox endpoint.
///
/// Python parity: `SandboxEndpoint`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct SandboxEndpoint {
    /// Base URL of the sandbox container.
    pub base_url: String,
    /// Provider-assigned sandbox id.
    pub sandbox_id: Option<String>,
    /// Resolved isolation key of this endpoint.
    pub isolation_key: Option<String>,
}

/// The placeholder replaced with the session id at call time.
pub const SESSION_ID_PLACEHOLDER: &str = "{session_id}";

/// Session id used when a session-scoped template is resolved without a session.
pub const DEFAULT_SESSION_ID: &str = "default_session";

/// Builds a sandbox isolation-key template.
///
/// Python parity: `generate_isolation_key_template` (`sys_operation.py`). The result is
/// `{scope}_{launcher_type}_{sandbox_type}_{prefix}{identity}` where `identity` is
/// `"system"` for [`ContainerScope::System`], the literal `{session_id}` placeholder for
/// [`ContainerScope::Session`], the custom id for [`ContainerScope::Custom`], and
/// `"default"` otherwise.
///
/// # Errors
///
/// Returns [`openjiuwen_common::error::ErrorKind::SysOp`] with the Python isolation-key
/// message (code 199010) when the scope has no identity: a custom scope without a
/// `custom_id`.
pub fn generate_isolation_key_template(
    isolation_prefix: Option<&str>,
    scope: ContainerScope,
    custom_id: Option<&str>,
    launcher_type: &str,
    sandbox_type: &str,
) -> Result<String> {
    let prefix = isolation_prefix.unwrap_or_default();
    let identity = match scope {
        ContainerScope::System => "system".to_string(),
        ContainerScope::Session => SESSION_ID_PLACEHOLDER.to_string(),
        ContainerScope::Custom => match custom_id {
            Some(id) if !id.trim().is_empty() => id.to_string(),
            _ => {
                return Err(Error::sys_op(isolation_key_message(
                    "custom_id is required when container_scope is 'custom'",
                )))
            }
        },
    };
    Ok(format!(
        "{}_{}_{}_{}{}",
        scope.as_str(),
        launcher_type,
        sandbox_type,
        prefix,
        identity
    ))
}

/// Resolves the `{session_id}` placeholder of an isolation-key template.
///
/// Every occurrence is replaced (Python `str.replace`), matching the reference
/// implementation.
#[must_use]
pub fn resolve_isolation_key(template: &str, session_id: Option<&str>) -> String {
    let session = session_id
        .filter(|id| !id.is_empty())
        .unwrap_or(DEFAULT_SESSION_ID);
    template.replace(SESSION_ID_PLACEHOLDER, session)
}

/// Builds the isolation-key template implied by a gateway configuration.
///
/// # Errors
///
/// Propagates [`generate_isolation_key_template`] failures.
pub fn isolation_key_template_from_config(config: &SandboxGatewayConfig) -> Result<String> {
    let launcher = config.launcher_config.as_ref().ok_or_else(|| {
        crate::card::card_param_error("launcher_config is required when mode is 'sandbox'")
    })?;
    generate_isolation_key_template(
        config.isolation.prefix.as_deref(),
        config.isolation.container_scope,
        config.isolation.custom_id.as_deref(),
        &launcher.launcher_type,
        &launcher.sandbox_type,
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn isolation_key_templates_match_python() {
        let session = generate_isolation_key_template(
            Some("agent1_"),
            ContainerScope::Session,
            None,
            "pre_deploy",
            "aio",
        )
        .unwrap();
        assert_eq!(session, "session_pre_deploy_aio_agent1_{session_id}");

        let system = generate_isolation_key_template(
            None,
            ContainerScope::System,
            None,
            "pre_deploy",
            "aio",
        )
        .unwrap();
        assert_eq!(system, "system_pre_deploy_aio_system");

        let custom = generate_isolation_key_template(
            None,
            ContainerScope::Custom,
            Some("abc"),
            "pre_deploy",
            "aio",
        )
        .unwrap();
        assert_eq!(custom, "custom_pre_deploy_aio_abc");
    }

    #[test]
    fn custom_scope_without_id_fails() {
        let error = generate_isolation_key_template(
            None,
            ContainerScope::Custom,
            None,
            "pre_deploy",
            "aio",
        )
        .unwrap_err();
        assert!(error.message().contains("custom_id is required"));
    }

    #[test]
    fn session_placeholder_resolution_replaces_all_occurrences() {
        assert_eq!(
            resolve_isolation_key("session_x_{session_id}_{session_id}", Some("s1")),
            "session_x_s1_s1"
        );
        assert_eq!(
            resolve_isolation_key("session_x_{session_id}", None),
            "session_x_default_session"
        );
        assert_eq!(
            resolve_isolation_key("static_key", Some("s1")),
            "static_key"
        );
    }

    #[test]
    fn gateway_config_defaults_match_python() {
        let config = SandboxGatewayConfig::default();
        assert_eq!(config.timeout_seconds, 30);
        assert!(config.launcher_config.is_none());
        assert!(!config.supports_conditional_write);
        assert_eq!(config.isolation.container_scope, ContainerScope::Session);
    }

    #[test]
    fn launcher_config_serializes_python_fields() {
        let json = serde_json::to_value(SandboxLauncherConfig::default()).unwrap();
        assert_eq!(json["launcher_type"], "pre_deploy");
        assert_eq!(json["sandbox_type"], "aio");
        assert_eq!(json["on_stop"], "delete");
        assert!(json.get("idle_ttl_seconds").is_some());
    }

    #[test]
    fn gateway_response_success_flag() {
        assert!(GatewayResponse::default().is_success());
        assert!(!GatewayResponse {
            code: -1,
            message: "boom".to_string(),
            data: None
        }
        .is_success());
    }
}
