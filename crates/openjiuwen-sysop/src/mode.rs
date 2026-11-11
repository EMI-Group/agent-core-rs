//! Operation modes and operation kinds.
//!
//! ```
//! use openjiuwen_sysop::mode::{OperationKind, OperationMode};
//!
//! assert_eq!(OperationMode::default(), OperationMode::Local);
//! assert_eq!(serde_json::to_string(&OperationMode::Sandbox).unwrap(), "\"sandbox\"");
//! assert_eq!(OperationKind::Fs.as_str(), "fs");
//! ```

use std::{fmt, str::FromStr};

use openjiuwen_common::error::Error;
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};

/// Where an operation executes.
///
/// Python parity: `OperationMode` in `core/sys_operation/base.py`. Serialized as
/// `"local"` / `"sandbox"`; deserialization accepts any casing (`"LOCAL"` included), the
/// same tolerance the Python pydantic validator applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum OperationMode {
    /// In-process filesystem, shell, and code execution on the host.
    #[default]
    Local,
    /// Execution delegated to a sandbox provider over the sandbox gateway.
    Sandbox,
}

impl OperationMode {
    /// The serialized name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Sandbox => "sandbox",
        }
    }

    /// Parses a mode case-insensitively, returning `None` for unknown values.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "local" => Some(Self::Local),
            "sandbox" => Some(Self::Sandbox),
            _ => None,
        }
    }

    /// Returns `true` for [`Self::Sandbox`].
    #[must_use]
    pub const fn is_sandbox(self) -> bool {
        matches!(self, Self::Sandbox)
    }
}

impl fmt::Display for OperationMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for OperationMode {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse(value).ok_or_else(|| {
            Error::sys_op(crate::status::card_param_message(&format!(
                "invalid operation mode: {value}"
            )))
        })
    }
}

impl<'de> Deserialize<'de> for OperationMode {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        Self::parse(&raw).ok_or_else(|| {
            serde::de::Error::custom(format!(
                "invalid operation mode '{raw}', expected 'local' or 'sandbox'"
            ))
        })
    }
}

/// The three operation surfaces a [`crate::SysOperation`] exposes.
///
/// Python parity: the `"fs"` / `"shell"` / `"code"` registry names, also used as the
/// middle segment of generated tool ids (`"<card>.<kind>.<method>"`).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum OperationKind {
    /// Filesystem operations.
    Fs,
    /// Shell command execution.
    Shell,
    /// Snippet/script execution.
    Code,
}

impl OperationKind {
    /// The registry name (`"fs"`, `"shell"`, `"code"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fs => "fs",
            Self::Shell => "shell",
            Self::Code => "code",
        }
    }

    /// Every kind, in registry order.
    #[must_use]
    pub const fn all() -> [Self; 3] {
        [Self::Fs, Self::Shell, Self::Code]
    }
}

impl fmt::Display for OperationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn mode_deserializes_case_insensitively() {
        let mode: OperationMode = serde_json::from_str("\"LOCAL\"").unwrap();
        assert_eq!(mode, OperationMode::Local);
        let mode: OperationMode = serde_json::from_str("\" Sandbox \"").unwrap();
        assert_eq!(mode, OperationMode::Sandbox);
        assert!(serde_json::from_str::<OperationMode>("\"remote\"").is_err());
    }

    #[test]
    fn mode_serializes_lowercase() {
        assert_eq!(
            serde_json::to_string(&OperationMode::Local).unwrap(),
            "\"local\""
        );
        assert_eq!(
            serde_json::to_string(&OperationMode::Sandbox).unwrap(),
            "\"sandbox\""
        );
    }

    #[test]
    fn kind_names_match_python_registry() {
        assert_eq!(OperationKind::Fs.to_string(), "fs");
        assert_eq!(OperationKind::Shell.to_string(), "shell");
        assert_eq!(OperationKind::Code.to_string(), "code");
        assert_eq!(OperationKind::all().len(), 3);
    }
}
