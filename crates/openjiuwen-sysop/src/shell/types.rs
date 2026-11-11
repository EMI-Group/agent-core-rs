//! Shell parameter and result types (Python `core/sys_operation/result/shell_operation_result.py`).
//!
//! ```
//! use openjiuwen_sysop::shell::types::{ShellExecParams, ShellType};
//!
//! let params = ShellExecParams::new("echo hi");
//! assert_eq!(params.shell_type, ShellType::Auto);
//! assert_eq!(params.effective_timeout_secs(300), 300);
//! ```

use std::collections::BTreeMap;
use std::str::FromStr;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::constants::DEFAULT_TIMEOUT_SECS;

/// The interpreter used to run a command.
///
/// Python parity: `ShellType` in `core/sys_operation/shell.py`. Parsing is
/// case-insensitive and unknown values fall back to [`ShellType::Auto`] (Python
/// `ShellType.from_str`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum ShellType {
    /// Pick an interpreter from the command text and the host OS.
    #[default]
    Auto,
    /// Windows `cmd.exe`.
    Cmd,
    /// PowerShell (`pwsh` or `powershell`).
    Powershell,
    /// POSIX `bash`.
    Bash,
    /// POSIX `sh`.
    Sh,
}

impl ShellType {
    /// The serialized name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cmd => "cmd",
            Self::Powershell => "powershell",
            Self::Bash => "bash",
            Self::Sh => "sh",
        }
    }

    /// Parses a shell type, falling back to [`ShellType::Auto`] for unknown values.
    #[must_use]
    pub fn parse_lenient(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "cmd" => Self::Cmd,
            "powershell" => Self::Powershell,
            "bash" => Self::Bash,
            "sh" => Self::Sh,
            _ => Self::Auto,
        }
    }

    /// Returns `true` for [`ShellType::Auto`].
    #[must_use]
    pub const fn is_auto(self) -> bool {
        matches!(self, Self::Auto)
    }
}

impl FromStr for ShellType {
    type Err = std::convert::Infallible;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Ok(Self::parse_lenient(value))
    }
}

/// Which pipe a streamed chunk came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum StreamKind {
    /// Standard output.
    Stdout,
    /// Standard error.
    Stderr,
}

impl StreamKind {
    /// The serialized name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Stdout => "stdout",
            Self::Stderr => "stderr",
        }
    }
}

/// Parameters of `execute_cmd` / `execute_cmd_stream` / `execute_cmd_background`.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct ShellExecParams {
    /// Command line to run. Interpreted by a shell, so it must pass the allowlist and
    /// deny-pattern checks. Prefer [`ShellArgvParams`] when no shell features are needed.
    pub command: String,
    /// Working directory; `None` uses the workspace root.
    pub cwd: Option<String>,
    /// Timeout in seconds; `None` uses the configured default (Python: `timeout or 300`).
    pub timeout_secs: Option<u64>,
    /// Environment overlay; a `None` value removes the variable.
    pub environment: BTreeMap<String, Option<String>>,
    /// Interpreter selection.
    pub shell_type: ShellType,
    /// Session id used to register the child process for cancellation.
    pub session_id: Option<String>,
}

impl ShellExecParams {
    /// Parameters running `command` with defaults.
    #[must_use]
    pub fn new(command: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            ..Self::default()
        }
    }

    /// Returns these parameters with a working directory.
    #[must_use]
    pub fn with_cwd(mut self, cwd: impl Into<String>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }

    /// Returns these parameters with a timeout.
    #[must_use]
    pub fn with_timeout_secs(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = Some(timeout_secs);
        self
    }

    /// Returns these parameters with an interpreter selection.
    #[must_use]
    pub fn with_shell_type(mut self, shell_type: ShellType) -> Self {
        self.shell_type = shell_type;
        self
    }

    /// Returns these parameters associated with a session.
    #[must_use]
    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// The timeout to apply, treating `None` and `0` as the provided default
    /// (Python `timeout = timeout or 300`).
    #[must_use]
    pub fn effective_timeout_secs(&self, default_secs: u64) -> u64 {
        match self.timeout_secs {
            Some(0) | None => default_secs,
            Some(value) => value,
        }
    }
}

/// Parameters of `execute_argv`: a command executed without a shell.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct ShellArgvParams {
    /// Program and arguments; the first element is the program name.
    pub argv: Vec<String>,
    /// Working directory; `None` uses the workspace root.
    pub cwd: Option<String>,
    /// Timeout in seconds; `None` uses the configured default.
    pub timeout_secs: Option<u64>,
    /// Environment overlay; a `None` value removes the variable.
    pub environment: BTreeMap<String, Option<String>>,
    /// Session id used to register the child process for cancellation.
    pub session_id: Option<String>,
}

impl ShellArgvParams {
    /// Parameters running `argv` with defaults.
    #[must_use]
    pub fn new<I, S>(argv: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            argv: argv.into_iter().map(Into::into).collect(),
            ..Self::default()
        }
    }

    /// Returns these parameters with a working directory.
    #[must_use]
    pub fn with_cwd(mut self, cwd: impl Into<String>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }

    /// Returns these parameters with a timeout.
    #[must_use]
    pub fn with_timeout_secs(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = Some(timeout_secs);
        self
    }

    /// The timeout to apply, treating `None` and `0` as the provided default.
    #[must_use]
    pub fn effective_timeout_secs(&self, default_secs: u64) -> u64 {
        match self.timeout_secs {
            Some(0) | None => default_secs,
            Some(value) => value,
        }
    }
}

/// Result of a completed command (Python `ExecuteCmdData`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct ShellOutput {
    /// The command that ran.
    pub command: String,
    /// Working directory the command ran in.
    pub cwd: String,
    /// Exit status; `-1` when the process was killed or never started.
    pub exit_code: i32,
    /// Captured standard output.
    pub stdout: String,
    /// Captured standard error.
    pub stderr: String,
    /// `true` when a stream hit the configured capture limit.
    pub truncated: bool,
    /// `true` when the timeout expired and the process was killed.
    pub timed_out: bool,
    /// `true` when the caller's cancellation token fired.
    pub cancelled: bool,
    /// Wall-clock duration in milliseconds.
    pub duration_ms: u64,
}

impl Default for ShellOutput {
    fn default() -> Self {
        Self {
            command: String::new(),
            cwd: String::new(),
            exit_code: -1,
            stdout: String::new(),
            stderr: String::new(),
            truncated: false,
            timed_out: false,
            cancelled: false,
            duration_ms: 0,
        }
    }
}

impl ShellOutput {
    /// Returns `true` when the process exited with status `0` and was not interrupted.
    #[must_use]
    pub const fn is_success(&self) -> bool {
        self.exit_code == 0 && !self.timed_out && !self.cancelled
    }
}

/// One chunk of `execute_cmd_stream` (Python `ExecuteCmdChunkData`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct ShellChunk {
    /// Chunk text, decoded as UTF-8 with replacement characters.
    pub text: String,
    /// Which pipe the chunk came from (`None` for the terminal chunk).
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub stream: Option<StreamKind>,
    /// Zero-based chunk index.
    pub chunk_index: u64,
    /// Exit status, carried by the terminal chunk.
    pub exit_code: Option<i32>,
    /// Free-form chunk metadata.
    pub metadata: Option<BTreeMap<String, Value>>,
}

impl ShellChunk {
    /// Builds an output chunk.
    #[must_use]
    pub fn new(text: impl Into<String>, stream: StreamKind, chunk_index: u64) -> Self {
        Self {
            text: text.into(),
            stream: Some(stream),
            chunk_index,
            exit_code: None,
            metadata: None,
        }
    }

    /// Builds the terminal chunk carrying the exit status.
    #[must_use]
    pub fn terminal(chunk_index: u64, exit_code: i32) -> Self {
        Self {
            chunk_index,
            exit_code: Some(exit_code),
            ..Self::default()
        }
    }
}

/// Result of `execute_cmd_background` (Python `ExecuteCmdBackgroundData`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct ShellBackgroundOutput {
    /// The command that was started.
    pub command: String,
    /// Working directory the command runs in.
    pub cwd: String,
    /// Process id, when the child started successfully.
    pub pid: Option<u32>,
    /// Launch or early-exit error, formatted like Python
    /// (`"process exited early with code <n>: <stderr>"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// The configured default timeout used when a caller omits one.
#[must_use]
pub const fn default_timeout_secs() -> u64 {
    DEFAULT_TIMEOUT_SECS
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn shell_type_parsing_is_lenient() {
        assert_eq!(
            ShellType::parse_lenient("POWERSHELL"),
            ShellType::Powershell
        );
        assert_eq!(ShellType::parse_lenient(" bash "), ShellType::Bash);
        assert_eq!(ShellType::parse_lenient("fish"), ShellType::Auto);
        assert_eq!("sh".parse::<ShellType>().unwrap(), ShellType::Sh);
    }

    #[test]
    fn timeout_defaults_like_python() {
        assert_eq!(ShellExecParams::new("x").effective_timeout_secs(300), 300);
        assert_eq!(
            ShellExecParams::new("x")
                .with_timeout_secs(0)
                .effective_timeout_secs(300),
            300
        );
        assert_eq!(
            ShellExecParams::new("x")
                .with_timeout_secs(5)
                .effective_timeout_secs(300),
            5
        );
    }

    #[test]
    fn chunks_serialize_python_type_field() {
        let chunk = ShellChunk::new("hi\n", StreamKind::Stdout, 0);
        let json = serde_json::to_value(&chunk).unwrap();
        assert_eq!(json["type"], "stdout");
        assert_eq!(json["chunk_index"], 0);
        assert!(json.get("exit_code").is_none() || json["exit_code"].is_null());

        let terminal = ShellChunk::terminal(3, 0);
        let json = serde_json::to_value(&terminal).unwrap();
        assert_eq!(json["exit_code"], 0);
        assert!(json.get("type").is_none());
    }

    #[test]
    fn shell_output_success_flag() {
        let mut output = ShellOutput {
            exit_code: 0,
            ..ShellOutput::default()
        };
        assert!(output.is_success());
        output.timed_out = true;
        assert!(!output.is_success());
        assert_eq!(ShellOutput::default().exit_code, -1);
    }

    #[test]
    fn argv_params_defaults() {
        let params = ShellArgvParams::new(["echo", "hi"]).with_timeout_secs(2);
        assert_eq!(params.argv, vec!["echo".to_string(), "hi".to_string()]);
        assert_eq!(params.effective_timeout_secs(300), 2);
        assert!(params.environment.is_empty());
    }

    #[test]
    fn background_output_omits_error_when_absent() {
        let json = serde_json::to_value(ShellBackgroundOutput::default()).unwrap();
        assert!(json.get("error").is_none());
    }
}
