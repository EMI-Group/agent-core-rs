//! Python-compatible status codes and message templates.
//!
//! Every operation result carries the numeric code the Python SDK reports
//! (`openjiuwen.core.common.exception.codes`, range `199000..=199999`) and the message
//! built from the same template, so a caller porting from Python can match on either.
//!
//! ```
//! use openjiuwen_sysop::status::{SysOpStatus, fs_execution_message};
//!
//! assert_eq!(SysOpStatus::FsExecutionError.code(), 199_003);
//! assert_eq!(
//!     fs_execution_message("resolve_path", "Access denied"),
//!     "file system operation execution error, execution: resolve_path, reason: Access denied",
//! );
//! ```

use std::fmt;

use serde::{Deserialize, Serialize};

/// The status code of a sys-operation result.
///
/// Python parity: the `SYS_OPERATION_*` members of `StatusCode`
/// (`core/common/exception/codes.py`) plus `SUCCESS` (0) and `ERROR` (-1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SysOpStatus {
    /// The operation completed (`code = 0`).
    Success,
    /// An unspecified failure (`code = -1`), used by the sandbox gateway.
    Error,
    /// Registration/manager failure (`199001`).
    ManagerError,
    /// Invalid card configuration (`199002`).
    CardParamError,
    /// Filesystem operation failure (`199003`).
    FsExecutionError,
    /// Shell operation failure (`199004`).
    ShellExecutionError,
    /// Code operation failure (`199005`).
    CodeExecutionError,
    /// Operation-registry failure (`199006`).
    RegistryError,
    /// Sandbox gateway failure (`199007`).
    SandboxGatewayError,
    /// Sandbox launcher failure (`199008`).
    SandboxLauncherError,
    /// Sandbox provider failure (`199009`).
    SandboxProviderError,
    /// Isolation-key resolution failure (`199010`).
    SandboxIsolationKeyError,
    /// Optimistic-concurrency write rejection (`199011`).
    FileVersionConflict,
}

impl SysOpStatus {
    /// The numeric code reported in [`crate::result::OpResult::code`].
    #[must_use]
    pub const fn code(self) -> i32 {
        match self {
            Self::Success => 0,
            Self::Error => -1,
            Self::ManagerError => 199_001,
            Self::CardParamError => 199_002,
            Self::FsExecutionError => 199_003,
            Self::ShellExecutionError => 199_004,
            Self::CodeExecutionError => 199_005,
            Self::RegistryError => 199_006,
            Self::SandboxGatewayError => 199_007,
            Self::SandboxLauncherError => 199_008,
            Self::SandboxProviderError => 199_009,
            Self::SandboxIsolationKeyError => 199_010,
            Self::FileVersionConflict => 199_011,
        }
    }

    /// The `snake_case` name of the status, matching the serde representation.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Error => "error",
            Self::ManagerError => "manager_error",
            Self::CardParamError => "card_param_error",
            Self::FsExecutionError => "fs_execution_error",
            Self::ShellExecutionError => "shell_execution_error",
            Self::CodeExecutionError => "code_execution_error",
            Self::RegistryError => "registry_error",
            Self::SandboxGatewayError => "sandbox_gateway_error",
            Self::SandboxLauncherError => "sandbox_launcher_error",
            Self::SandboxProviderError => "sandbox_provider_error",
            Self::SandboxIsolationKeyError => "sandbox_isolation_key_error",
            Self::FileVersionConflict => "file_version_conflict",
        }
    }

    /// Returns `true` for [`Self::Success`].
    #[must_use]
    pub const fn is_success(self) -> bool {
        matches!(self, Self::Success)
    }

    /// Maps a numeric code back to its status, if the code is known.
    #[must_use]
    pub const fn from_code(code: i32) -> Option<Self> {
        Some(match code {
            0 => Self::Success,
            -1 => Self::Error,
            199_001 => Self::ManagerError,
            199_002 => Self::CardParamError,
            199_003 => Self::FsExecutionError,
            199_004 => Self::ShellExecutionError,
            199_005 => Self::CodeExecutionError,
            199_006 => Self::RegistryError,
            199_007 => Self::SandboxGatewayError,
            199_008 => Self::SandboxLauncherError,
            199_009 => Self::SandboxProviderError,
            199_010 => Self::SandboxIsolationKeyError,
            199_011 => Self::FileVersionConflict,
            _ => return None,
        })
    }
}

impl fmt::Display for SysOpStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// `sys operation manager process error, process: {process}, reason: {reason}` (199001).
#[must_use]
pub fn manager_message(process: &str, reason: &str) -> String {
    format!("sys operation manager process error, process: {process}, reason: {reason}")
}

/// `sys operation card param error, reason: {reason}` (199002).
#[must_use]
pub fn card_param_message(reason: &str) -> String {
    format!("sys operation card param error, reason: {reason}")
}

/// `file system operation execution error, execution: {execution}, reason: {reason}` (199003).
#[must_use]
pub fn fs_execution_message(execution: &str, reason: &str) -> String {
    format!("file system operation execution error, execution: {execution}, reason: {reason}")
}

/// `shell operation execution error, execution: {execution}, reason: {reason}` (199004).
#[must_use]
pub fn shell_execution_message(execution: &str, reason: &str) -> String {
    format!("shell operation execution error, execution: {execution}, reason: {reason}")
}

/// `code operation execution error, execution: {execution}, reason: {reason}` (199005).
#[must_use]
pub fn code_execution_message(execution: &str, reason: &str) -> String {
    format!("code operation execution error, execution: {execution}, reason: {reason}")
}

/// `sys operation registry error, process: {process}, reason: {reason}` (199006).
#[must_use]
pub fn registry_message(process: &str, reason: &str) -> String {
    format!("sys operation registry error, process: {process}, reason: {reason}")
}

/// `sandbox gateway error, operation: {operation}, reason: {reason}` (199007).
#[must_use]
pub fn sandbox_gateway_message(operation: &str, reason: &str) -> String {
    format!("sandbox gateway error, operation: {operation}, reason: {reason}")
}

/// `sandbox launcher error, process: {process}, reason: {reason}` (199008).
#[must_use]
pub fn sandbox_launcher_message(process: &str, reason: &str) -> String {
    format!("sandbox launcher error, process: {process}, reason: {reason}")
}

/// `sandbox provider error, process: {process}, reason: {reason}` (199009).
#[must_use]
pub fn sandbox_provider_message(process: &str, reason: &str) -> String {
    format!("sandbox provider error, process: {process}, reason: {reason}")
}

/// `sandbox isolation key error, reason: {reason}` (199010).
#[must_use]
pub fn isolation_key_message(reason: &str) -> String {
    format!("sandbox isolation key error, reason: {reason}")
}

/// `file version conflict` (199011).
#[must_use]
pub fn file_version_conflict_message() -> String {
    "file version conflict".to_string()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn codes_match_python_status_codes() {
        assert_eq!(SysOpStatus::Success.code(), 0);
        assert_eq!(SysOpStatus::Error.code(), -1);
        assert_eq!(SysOpStatus::ManagerError.code(), 199_001);
        assert_eq!(SysOpStatus::CardParamError.code(), 199_002);
        assert_eq!(SysOpStatus::FsExecutionError.code(), 199_003);
        assert_eq!(SysOpStatus::ShellExecutionError.code(), 199_004);
        assert_eq!(SysOpStatus::CodeExecutionError.code(), 199_005);
        assert_eq!(SysOpStatus::RegistryError.code(), 199_006);
        assert_eq!(SysOpStatus::SandboxGatewayError.code(), 199_007);
        assert_eq!(SysOpStatus::SandboxLauncherError.code(), 199_008);
        assert_eq!(SysOpStatus::SandboxProviderError.code(), 199_009);
        assert_eq!(SysOpStatus::SandboxIsolationKeyError.code(), 199_010);
        assert_eq!(SysOpStatus::FileVersionConflict.code(), 199_011);
    }

    #[test]
    fn from_code_round_trips() {
        for status in [
            SysOpStatus::Success,
            SysOpStatus::Error,
            SysOpStatus::ManagerError,
            SysOpStatus::CardParamError,
            SysOpStatus::FsExecutionError,
            SysOpStatus::ShellExecutionError,
            SysOpStatus::CodeExecutionError,
            SysOpStatus::RegistryError,
            SysOpStatus::SandboxGatewayError,
            SysOpStatus::SandboxLauncherError,
            SysOpStatus::SandboxProviderError,
            SysOpStatus::SandboxIsolationKeyError,
            SysOpStatus::FileVersionConflict,
        ] {
            assert_eq!(SysOpStatus::from_code(status.code()), Some(status));
        }
        assert_eq!(SysOpStatus::from_code(12_345), None);
    }

    #[test]
    fn messages_match_python_templates() {
        assert_eq!(
            manager_message("add", "duplicated"),
            "sys operation manager process error, process: add, reason: duplicated"
        );
        assert_eq!(
            card_param_message("bad mode"),
            "sys operation card param error, reason: bad mode"
        );
        assert_eq!(
            fs_execution_message("read_file", "boom"),
            "file system operation execution error, execution: read_file, reason: boom"
        );
        assert_eq!(
            shell_execution_message("execute_cmd", "boom"),
            "shell operation execution error, execution: execute_cmd, reason: boom"
        );
        assert_eq!(
            code_execution_message("execute_code", "boom"),
            "code operation execution error, execution: execute_code, reason: boom"
        );
        assert_eq!(
            registry_message("lookup", "missing"),
            "sys operation registry error, process: lookup, reason: missing"
        );
        assert_eq!(
            sandbox_gateway_message("fs", "down"),
            "sandbox gateway error, operation: fs, reason: down"
        );
        assert_eq!(
            sandbox_launcher_message("launch", "down"),
            "sandbox launcher error, process: launch, reason: down"
        );
        assert_eq!(
            sandbox_provider_message("create", "down"),
            "sandbox provider error, process: create, reason: down"
        );
        assert_eq!(
            isolation_key_message("bad"),
            "sandbox isolation key error, reason: bad"
        );
        assert_eq!(file_version_conflict_message(), "file version conflict");
    }

    #[test]
    fn serde_round_trip_is_snake_case() {
        let json = serde_json::to_string(&SysOpStatus::FsExecutionError).unwrap();
        assert_eq!(json, "\"fs_execution_error\"");
        let back: SysOpStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(back, SysOpStatus::FsExecutionError);
    }
}
