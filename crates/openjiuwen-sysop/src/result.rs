//! The operation-result envelope, ported from Python `BaseResult[T]`.
//!
//! Every filesystem, shell, and code operation returns [`OpResult`]: `code` (0 on
//! success), a human-readable `message`, and an optional payload. Python uses this shape
//! so a failure is a *value*, not an exception; the Rust API keeps the same wire format
//! for tool calls while [`OpResult::into_result`] offers the idiomatic `Result` view.
//!
//! ```
//! use openjiuwen_common::error::ErrorKind;
//! use openjiuwen_sysop::result::OpResult;
//! use openjiuwen_sysop::status::{SysOpStatus, fs_execution_message};
//!
//! let ok = OpResult::success(42_u32);
//! assert_eq!(ok.code, 0);
//! assert_eq!(ok.into_result().unwrap(), 42);
//!
//! let failure = OpResult::<u32>::failure(
//!     SysOpStatus::FsExecutionError,
//!     fs_execution_message("read_file", "not found"),
//! );
//! assert_eq!(failure.code, 199_003);
//! assert_eq!(failure.into_result().unwrap_err().kind(), ErrorKind::SysOp);
//! ```

use std::fmt;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

use crate::status::SysOpStatus;

/// A sys-operation result carrying a Python-compatible status code.
///
/// The serialized shape is exactly Python's `BaseResult.model_dump()`:
/// `{"code": 0, "message": "success", "data": ...}` with `data` omitted when absent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct OpResult<T> {
    /// Status code: `0` on success, `-1` or `1990xx` on failure.
    pub code: i32,
    /// Human-readable message; the Python template text on failure.
    pub message: String,
    /// The payload, absent for failures and for `()`-shaped successes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<T>,
}

impl<T> Default for OpResult<T> {
    fn default() -> Self {
        Self {
            code: SysOpStatus::Success.code(),
            message: "success".to_string(),
            data: None,
        }
    }
}

impl<T> OpResult<T> {
    /// Builds a successful result with a payload.
    #[must_use]
    pub fn success(data: T) -> Self {
        Self {
            code: SysOpStatus::Success.code(),
            message: "success".to_string(),
            data: Some(data),
        }
    }

    /// Builds a successful result without a payload.
    #[must_use]
    pub fn success_empty() -> Self {
        Self::default()
    }

    /// Builds a failure result for `status` with the given message.
    ///
    /// Use the message helpers in [`crate::status`] so the text matches Python.
    #[must_use]
    pub fn failure(status: SysOpStatus, message: impl Into<String>) -> Self {
        Self {
            code: status.code(),
            message: message.into(),
            data: None,
        }
    }

    /// Builds an [`SysOpStatus::Error`] failure from a shared [`Error`].
    #[must_use]
    pub fn from_error(error: &Error) -> Self {
        Self::failure(SysOpStatus::Error, error.message())
    }

    /// Returns `true` when the status code is `0`.
    #[must_use]
    pub const fn is_success(&self) -> bool {
        self.code == 0
    }

    /// Returns `true` when the status code is not `0`.
    #[must_use]
    pub const fn is_failure(&self) -> bool {
        self.code != 0
    }

    /// The status matching [`Self::code`], when the code is one this crate defines.
    #[must_use]
    pub fn status(&self) -> Option<SysOpStatus> {
        SysOpStatus::from_code(self.code)
    }

    /// Borrows the payload.
    #[must_use]
    pub const fn data(&self) -> Option<&T> {
        self.data.as_ref()
    }

    /// Maps the payload of a successful result, leaving failures untouched.
    #[must_use]
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> OpResult<U> {
        OpResult {
            code: self.code,
            message: self.message,
            data: self.data.map(f),
        }
    }

    /// Converts into an idiomatic [`Result`], mapping any non-zero code to
    /// [`openjiuwen_common::error::ErrorKind::SysOp`].
    ///
    /// # Errors
    ///
    /// Returns the failure message as [`Error::sys_op`] when the code is non-zero, and
    /// returns a `sys_op` error when the operation succeeded without a payload.
    pub fn into_result(self) -> Result<T> {
        if self.is_failure() {
            return Err(Error::sys_op(self.message));
        }
        self.data
            .ok_or_else(|| Error::sys_op("operation succeeded without a payload"))
    }

    /// Converts into an idiomatic [`Result`] keeping the payload optional.
    ///
    /// # Errors
    ///
    /// Returns the failure message as [`Error::sys_op`] when the code is non-zero.
    pub fn into_optional_result(self) -> Result<Option<T>> {
        if self.is_failure() {
            return Err(Error::sys_op(self.message));
        }
        Ok(self.data)
    }
}

impl<T> fmt::Display for OpResult<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_success() {
            write!(f, "ok: {}", self.message)
        } else {
            write!(f, "code {}: {}", self.code, self.message)
        }
    }
}

/// Convenience conversions used by the backend implementations.
impl<T> From<Result<T>> for OpResult<T> {
    fn from(value: Result<T>) -> Self {
        match value {
            Ok(data) => Self::success(data),
            Err(error) => Self::from_error(&error),
        }
    }
}

/// Builds a filesystem failure result, filling Python's `execution` slot.
#[must_use]
pub fn fs_failure<T>(execution: &str, reason: &str) -> OpResult<T> {
    OpResult::failure(
        SysOpStatus::FsExecutionError,
        crate::status::fs_execution_message(execution, reason),
    )
}

/// Builds a shell failure result, filling Python's `execution` slot.
#[must_use]
pub fn shell_failure<T>(execution: &str, reason: &str) -> OpResult<T> {
    OpResult::failure(
        SysOpStatus::ShellExecutionError,
        crate::status::shell_execution_message(execution, reason),
    )
}

/// Builds a code failure result, filling Python's `execution` slot.
#[must_use]
pub fn code_failure<T>(execution: &str, reason: &str) -> OpResult<T> {
    OpResult::failure(
        SysOpStatus::CodeExecutionError,
        crate::status::code_execution_message(execution, reason),
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn success_serializes_like_python_base_result() {
        let result = OpResult::success(7_u8);
        let json = serde_json::to_value(&result).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"code": 0, "message": "success", "data": 7})
        );
    }

    #[test]
    fn failure_omits_data() {
        let result = fs_failure::<u8>("read_file", "boom");
        let json = serde_json::to_value(&result).unwrap();
        assert_eq!(json["code"], 199_003);
        assert_eq!(
            json["message"],
            "file system operation execution error, execution: read_file, reason: boom"
        );
        assert!(json.get("data").is_none());
    }

    #[test]
    fn empty_success_round_trips() {
        let result = OpResult::<()>::success_empty();
        assert!(result.is_success());
        assert_eq!(result.status(), Some(SysOpStatus::Success));
        assert!(result.into_optional_result().unwrap().is_none());
    }

    #[test]
    fn into_result_maps_failure_to_sysop_error() {
        let failure = shell_failure::<u8>("execute_cmd", "command can not be empty");
        let error = failure.into_result().unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::SysOp);
        assert!(error.message().contains("command can not be empty"));
    }

    #[test]
    fn deserializes_tolerantly() {
        let result: OpResult<u32> = serde_json::from_str(r#"{"code": 0}"#).unwrap();
        assert_eq!(result.code, 0);
        assert_eq!(result.message, "success");
        assert!(result.data.is_none());
    }

    #[test]
    fn map_keeps_failure() {
        let failure = fs_failure::<u8>("read_file", "boom").map(u16::from);
        assert_eq!(failure.code, 199_003);
        assert!(failure.data().is_none());
    }
}
