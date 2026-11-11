//! Code-execution parameter and result types (Python `core/sys_operation/result/code_operation_result.py`).
//!
//! ```
//! use openjiuwen_sysop::code::types::{CodeExecParams, Language};
//!
//! let params = CodeExecParams::new("print(1)");
//! assert_eq!(params.language, Language::Python);
//! assert_eq!(Language::Python.executable(), "python3");
//! ```

use std::collections::BTreeMap;
use std::fmt;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::constants::DEFAULT_TIMEOUT_SECS;
use crate::shell::types::StreamKind;

/// Language of a code snippet.
///
/// Python parity: the `language: Literal['python', 'javascript']` argument and
/// `_SUPPORT_LANGUAGE_CONFIG_DICT` (`local/code_operation.py`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    /// Python 3 (default).
    #[default]
    Python,
    /// JavaScript, run with `node`.
    Javascript,
}

impl Language {
    /// The serialized name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Python => "python",
            Self::Javascript => "javascript",
        }
    }

    /// The temporary-file suffix used when a snippet is written to disk.
    #[must_use]
    pub const fn file_suffix(self) -> &'static str {
        match self {
            Self::Python => ".py",
            Self::Javascript => ".js",
        }
    }

    /// The interpreter executable, looked up on `PATH`.
    #[must_use]
    pub const fn executable(self) -> &'static str {
        match self {
            Self::Python => "python3",
            Self::Javascript => "node",
        }
    }

    /// The interpreter flags used when running a snippet inline.
    #[must_use]
    pub const fn inline_flags(self) -> &'static [&'static str] {
        match self {
            Self::Python => &["-u", "-c"],
            Self::Javascript => &["-e"],
        }
    }

    /// Environment variables forced for this language (applied after the caller's
    /// overrides, matching Python).
    #[must_use]
    pub const fn forced_env(self) -> &'static [(&'static str, &'static str)] {
        match self {
            Self::Python => &[("PYTHONIOENCODING", "utf-8"), ("PYTHONUTF8", "1")],
            Self::Javascript => &[("NODE_DISABLE_COLORS", "1")],
        }
    }

    /// Parses a language name case-insensitively, returning `None` when unsupported.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "python" | "python3" => Some(Self::Python),
            "javascript" | "js" | "node" => Some(Self::Javascript),
            _ => None,
        }
    }
}

impl fmt::Display for Language {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Parameters of `execute_code` / `execute_code_stream`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct CodeExecParams {
    /// Snippet or script body.
    pub code: String,
    /// Language of `code`.
    pub language: Language,
    /// Timeout in seconds (Python default `300`; `0` keeps the configured default).
    pub timeout_secs: u64,
    /// Environment overlay; a `None` value removes the variable.
    pub environment: BTreeMap<String, Option<String>>,
    /// Working directory; `None` uses the process working directory.
    pub cwd: Option<String>,
    /// Always write the snippet to a temporary file instead of passing it inline.
    pub force_file: bool,
    /// Output decoding encoding.
    pub encoding: String,
}

impl Default for CodeExecParams {
    fn default() -> Self {
        Self {
            code: String::new(),
            language: Language::Python,
            timeout_secs: DEFAULT_TIMEOUT_SECS,
            environment: BTreeMap::new(),
            cwd: None,
            force_file: false,
            encoding: "utf-8".to_string(),
        }
    }
}

impl CodeExecParams {
    /// Python parameters for `code`.
    #[must_use]
    pub fn new(code: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            ..Self::default()
        }
    }

    /// JavaScript parameters for `code`.
    #[must_use]
    pub fn javascript(code: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            language: Language::Javascript,
            ..Self::default()
        }
    }

    /// Returns these parameters with a timeout.
    #[must_use]
    pub fn with_timeout_secs(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }

    /// Returns these parameters with a working directory.
    #[must_use]
    pub fn with_cwd(mut self, cwd: impl Into<String>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }

    /// Returns these parameters forced through a temporary file.
    #[must_use]
    pub fn with_force_file(mut self, force_file: bool) -> Self {
        self.force_file = force_file;
        self
    }

    /// The timeout to apply, treating `0` as the configured default.
    #[must_use]
    pub fn effective_timeout_secs(&self, default_secs: u64) -> u64 {
        if self.timeout_secs == 0 {
            default_secs
        } else {
            self.timeout_secs
        }
    }
}

/// Result of `execute_code` (Python `ExecuteCodeData`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct CodeOutput {
    /// The snippet that ran.
    pub code_content: String,
    /// Language name.
    pub language: String,
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
    /// Wall-clock duration in milliseconds.
    pub duration_ms: u64,
}

impl Default for CodeOutput {
    fn default() -> Self {
        Self {
            code_content: String::new(),
            language: Language::Python.as_str().to_string(),
            exit_code: -1,
            stdout: String::new(),
            stderr: String::new(),
            truncated: false,
            timed_out: false,
            duration_ms: 0,
        }
    }
}

/// One chunk of `execute_code_stream` (Python `ExecuteCodeChunkData`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct CodeChunk {
    /// Chunk text, decoded as UTF-8 with replacement characters.
    pub text: String,
    /// Which pipe the chunk came from (`None` for the terminal chunk).
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub stream: Option<StreamKind>,
    /// Zero-based chunk index.
    pub chunk_index: u64,
    /// Exit status, carried by the terminal chunk.
    pub exit_code: Option<i32>,
}

impl CodeChunk {
    /// Builds an output chunk.
    #[must_use]
    pub fn new(text: impl Into<String>, stream: StreamKind, chunk_index: u64) -> Self {
        Self {
            text: text.into(),
            stream: Some(stream),
            chunk_index,
            exit_code: None,
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn language_parsing_and_metadata() {
        assert_eq!(Language::parse("JavaScript"), Some(Language::Javascript));
        assert_eq!(Language::parse("py"), None);
        assert_eq!(Language::Python.executable(), "python3");
        assert_eq!(Language::Javascript.file_suffix(), ".js");
        assert_eq!(Language::Python.inline_flags(), &["-u", "-c"]);
        assert!(Language::Python.forced_env().contains(&("PYTHONUTF8", "1")));
        assert_eq!(
            Language::Javascript.forced_env(),
            &[("NODE_DISABLE_COLORS", "1")]
        );
    }

    #[test]
    fn params_defaults_and_timeout() {
        let params = CodeExecParams::new("print(1)");
        assert_eq!(params.language, Language::Python);
        assert_eq!(params.effective_timeout_secs(300), 300);
        assert!(!params.force_file);
        assert_eq!(
            params
                .clone()
                .with_timeout_secs(0)
                .effective_timeout_secs(300),
            300
        );
        assert_eq!(params.with_timeout_secs(7).effective_timeout_secs(300), 7);
    }

    #[test]
    fn output_defaults() {
        let output = CodeOutput::default();
        assert_eq!(output.exit_code, -1);
        assert_eq!(output.language, "python");
    }

    #[test]
    fn terminal_chunk_carries_exit_code() {
        let chunk = CodeChunk::terminal(2, 1);
        assert_eq!(chunk.exit_code, Some(1));
        assert!(chunk.stream.is_none());
        let json = serde_json::to_value(&chunk).unwrap();
        assert_eq!(json["exit_code"], 1);
    }
}
