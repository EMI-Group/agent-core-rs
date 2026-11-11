//! `SysOperationCard`: the registry-safe description of a sys-operation service.
//!
//! Python parity: `SysOperationCard` in `core/sys_operation/sys_operation.py`, which
//! extends `BaseCard` with `mode`, `work_config`, and `gateway_config`. The JSON field
//! names, defaults, and serialized shapes are identical; `description` is an
//! `Option<String>` (omitted when absent) instead of Python's empty string, matching
//! every other card in this workspace.
//!
//! ```
//! use openjiuwen_sysop::card::SysOperationCard;
//! use openjiuwen_sysop::mode::OperationMode;
//!
//! let card = SysOperationCard::new("sys_op").with_mode(OperationMode::Local);
//! assert_eq!(card.id, "sys_op");
//! assert_eq!(card.tool_id(openjiuwen_sysop::mode::OperationKind::Fs, "read_file"),
//!            "sys_op.fs.read_file");
//! ```

use openjiuwen_common::card::{Card, CardBase};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::id::uuid_hex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::config::LocalWorkConfig;
use crate::mode::{OperationKind, OperationMode};
use crate::sandbox::config::SandboxGatewayConfig;
use crate::status::card_param_message;

/// A registered sys-operation service: identity, mode, and backend configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SysOperationCard {
    /// Unique id; also the prefix of every generated tool id (`uuid4().hex` default).
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Optional description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Where operations execute. Defaults to [`OperationMode::Local`].
    pub mode: OperationMode,
    /// Local-backend configuration; `None` means [`LocalWorkConfig::default`].
    pub work_config: Option<LocalWorkConfig>,
    /// Sandbox-backend configuration; required when [`Self::mode`] is
    /// [`OperationMode::Sandbox`].
    pub gateway_config: Option<SandboxGatewayConfig>,
}

impl Default for SysOperationCard {
    fn default() -> Self {
        Self {
            id: uuid_hex(),
            name: String::new(),
            description: None,
            mode: OperationMode::Local,
            work_config: None,
            gateway_config: None,
        }
    }
}

impl SysOperationCard {
    /// Creates a local-mode card with the given id.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            ..Self::default()
        }
    }

    /// Creates a local-mode card with a generated id and the given name.
    #[must_use]
    pub fn unnamed(name: impl Into<String>) -> Self {
        Self::default().with_name(name)
    }

    /// Returns this card with the given id.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Returns this card with the given name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Returns this card with the given description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Returns this card with the given operation mode.
    #[must_use]
    pub fn with_mode(mut self, mode: OperationMode) -> Self {
        self.mode = mode;
        self
    }

    /// Returns this card with the given local work configuration.
    #[must_use]
    pub fn with_work_config(mut self, work_config: LocalWorkConfig) -> Self {
        self.work_config = Some(work_config);
        self
    }

    /// Returns this card with the given sandbox gateway configuration.
    #[must_use]
    pub fn with_gateway_config(mut self, gateway_config: SandboxGatewayConfig) -> Self {
        self.gateway_config = Some(gateway_config);
        self
    }

    /// The effective local work configuration (Python: `card.work_config or LocalWorkConfig()`).
    #[must_use]
    pub fn resolved_work_config(&self) -> LocalWorkConfig {
        self.work_config.clone().unwrap_or_default()
    }

    /// The generated tool id for an operation method (`"<card>.<kind>.<method>"`).
    #[must_use]
    pub fn tool_id(&self, kind: OperationKind, method: &str) -> String {
        generate_tool_id(&self.id, kind, method)
    }

    /// Converts this card into a [`CardBase`], dropping the sys-operation fields.
    #[must_use]
    pub fn to_card_base(&self) -> CardBase {
        CardBase {
            id: self.id.clone(),
            name: self.name.clone(),
            description: self.description.clone(),
            ..CardBase::default()
        }
    }

    /// Validates the mode/backend combination before a backend is built.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::SysOp`] with the Python
    /// `sys operation card param error` text (code 199002) when the card is inconsistent:
    /// an empty id, an invalid local work configuration, a sandbox card without gateway
    /// configuration or launcher configuration, or a launcher configuration missing
    /// `launcher_type`/`sandbox_type`.
    pub fn validate(&self) -> Result<()> {
        if self.id.trim().is_empty() {
            return Err(card_param_error("card id must not be empty"));
        }
        if let Some(work_config) = &self.work_config {
            work_config.validate()?;
        }
        if !self.mode.is_sandbox() {
            return Ok(());
        }
        let gateway = self
            .gateway_config
            .as_ref()
            .ok_or_else(|| card_param_error("gateway_config is required when mode is 'sandbox'"))?;
        let launcher = gateway.launcher_config.as_ref().ok_or_else(|| {
            card_param_error("launcher_config is required when mode is 'sandbox'")
        })?;
        if launcher.launcher_type.trim().is_empty() {
            return Err(card_param_error("launcher_type must not be empty"));
        }
        if launcher.sandbox_type.trim().is_empty() {
            return Err(card_param_error("sandbox_type must not be empty"));
        }
        Ok(())
    }
}

impl Card for SysOperationCard {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }
}

/// Builds a card-parameter error carrying Python's message text (code 199002).
pub fn card_param_error(reason: &str) -> Error {
    Error::sys_op(card_param_message(reason))
}

/// Formats a generated tool id: `"<card>.<kind>.<method>"` (Python `generate_tool_id`).
#[must_use]
pub fn generate_tool_id(card_id: &str, kind: OperationKind, method: &str) -> String {
    format!("{card_id}.{kind}.{method}")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn json_matches_python_shape() {
        let card = SysOperationCard::new("sys_op");
        let json = serde_json::to_value(&card).unwrap();
        assert_eq!(json["id"], "sys_op");
        assert_eq!(json["mode"], "local");
        assert!(json
            .get("work_config")
            .is_none_or(serde_json::Value::is_null));
        assert_eq!(json["gateway_config"], serde_json::Value::Null);
    }

    #[test]
    fn empty_document_gets_generated_id_and_local_mode() {
        let card: SysOperationCard = serde_json::from_str("{}").unwrap();
        assert_eq!(card.id.len(), 32);
        assert_eq!(card.mode, OperationMode::Local);
        assert!(card.work_config.is_none());
    }

    #[test]
    fn mode_string_is_case_insensitive_on_parse() {
        let card: SysOperationCard =
            serde_json::from_str(r#"{"id": "a", "mode": "SANDBOX"}"#).unwrap();
        assert_eq!(card.mode, OperationMode::Sandbox);
    }

    #[test]
    fn validate_rejects_sandbox_without_gateway() {
        let card = SysOperationCard::new("a").with_mode(OperationMode::Sandbox);
        let error = card.validate().unwrap_err();
        assert!(error.message().contains("gateway_config is required"));
    }

    #[test]
    fn validate_accepts_local_card() {
        assert!(SysOperationCard::new("a").validate().is_ok());
        assert!(SysOperationCard::new(" ").validate().is_err());
    }

    #[test]
    fn tool_ids_are_prefixed_by_card_kind_and_method() {
        let card = SysOperationCard::new("sys_op");
        assert_eq!(
            card.tool_id(OperationKind::Fs, "read_file"),
            "sys_op.fs.read_file"
        );
        assert_eq!(
            card.tool_id(OperationKind::Shell, "execute_cmd"),
            "sys_op.shell.execute_cmd"
        );
    }

    #[test]
    fn card_trait_is_implemented() {
        let card = SysOperationCard::new("x").with_description("d");
        assert_eq!(Card::id(&card), "x");
        assert_eq!(Card::description(&card), Some("d"));
    }
}
