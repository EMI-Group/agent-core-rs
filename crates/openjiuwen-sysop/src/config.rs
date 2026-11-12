//! Local work configuration: allowlist, sandbox roots, limits, and environment policy.
//!
//! Python parity: `LocalWorkConfig` in `core/sys_operation/config.py` keeps
//! `shell_allowlist`, `sandbox_root`, `restrict_to_sandbox`, and `dangerous_patterns`
//! with identical JSON names and defaults. The remaining fields are documented
//! additions that make the local backend safer and testable in Rust (bounded output
//! capture, max file size, environment scoping).
//!
//! ```
//! use openjiuwen_sysop::config::LocalWorkConfig;
//!
//! let config = LocalWorkConfig::default();
//! assert_eq!(config.shell_allowlist.as_ref().map(Vec::len), Some(30));
//! assert!(!config.restrict_to_sandbox);
//! ```

use openjiuwen_common::error::{Error, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::constants::{DEFAULT_MAX_OUTPUT_BYTES, DEFAULT_SHELL_ALLOWLIST, DEFAULT_TIMEOUT_SECS};

/// Regular expressions that reject a dangerous command when
/// [`LocalWorkConfig::dangerous_patterns`] enables them.
///
/// These are **not** enabled by default: Python's built-in table
/// (`_DANGEROUS_PATTERNS` in `local/shell_operation.py`) is empty, so a command such as
/// `rm -rf /` passes the default configuration. Callers that want defense in depth set
/// `dangerous_patterns = Some(DEFAULT_DANGEROUS_PATTERNS.iter().map(...).collect())`.
pub const DEFAULT_DANGEROUS_PATTERNS: &[&str] = &[
    r"(?i)\brm\s+-[a-z]*[rf]",
    r"(?i)\bmkfs(\.[a-z0-9]+)?\b",
    r"(?i)\bdd\b[^\n]*\bof=/dev/",
    r"(?i)\bshutdown\b",
    r"(?i)\breboot\b",
    r"(?i)\bhalt\b",
    r"(?i)\bdiskpart\b",
    r"(?i)\breg\s+delete\b",
    r"(?i)remove-item[^\n]*-recurse",
    r"(?i):\s*\(\s*\)\s*\{[^\n]*\}\s*;\s*:",
];

/// How bytes are decoded when a file is read in text mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum BinaryPolicy {
    /// Decode with replacement characters (Python `errors="replace"`).
    #[default]
    Lossy,
    /// Reject content that [`crate::encoding::is_binary`] classifies as binary.
    Reject,
}

/// Which environment variables a child process inherits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EnvInheritMode {
    /// Inherit the whole parent environment (Python parity).
    #[default]
    All,
    /// Start from an empty environment; only explicit values are set.
    None,
    /// Inherit only the names listed in [`EnvPolicy::allow`].
    Allowlist,
}

/// Environment scoping applied to shell and code subprocesses.
///
/// Python always inherits the host environment (no scrubbing) and only applies explicit
/// removals. [`EnvPolicy::default`] preserves that behavior; the other modes let a host
/// application keep credentials out of agent-spawned processes.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct EnvPolicy {
    /// How the parent environment is inherited.
    pub mode: EnvInheritMode,
    /// Variable names kept when [`EnvPolicy::mode`] is [`EnvInheritMode::Allowlist`].
    pub allow: Vec<String>,
    /// Variable names always removed (matched case-insensitively on Windows).
    pub scrub: Vec<String>,
}

impl EnvPolicy {
    /// Builds a policy that inherits nothing but `PATH`-like essentials.
    #[must_use]
    pub fn minimal() -> Self {
        Self {
            mode: EnvInheritMode::Allowlist,
            allow: vec![
                "PATH".to_string(),
                "HOME".to_string(),
                "TMPDIR".to_string(),
                "TEMP".to_string(),
                "TMP".to_string(),
                "SYSTEMROOT".to_string(),
                "COMSPEC".to_string(),
                "PATHEXT".to_string(),
                "LANG".to_string(),
                "LC_ALL".to_string(),
                "TERM".to_string(),
            ],
            scrub: Vec::new(),
        }
    }

    /// Returns `true` when `name` must be removed from the inherited environment.
    #[must_use]
    pub fn scrubs(&self, name: &str) -> bool {
        self.scrub.iter().any(|s| s.eq_ignore_ascii_case(name))
    }

    /// Returns `true` when `name` may be inherited.
    #[must_use]
    pub fn inherits(&self, name: &str) -> bool {
        if self.scrubs(name) {
            return false;
        }
        match self.mode {
            EnvInheritMode::All => true,
            EnvInheritMode::None => false,
            EnvInheritMode::Allowlist => self.allow.iter().any(|s| s.eq_ignore_ascii_case(name)),
        }
    }
}

/// Configuration of the local (`OperationMode::Local`) backend.
///
/// The first four fields are Python's `LocalWorkConfig` verbatim; the rest extend it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct LocalWorkConfig {
    /// First-token allowlist for shell commands; `None` or empty allows every command.
    ///
    /// Defaults to [`DEFAULT_SHELL_ALLOWLIST`] (Python parity).
    pub shell_allowlist: Option<Vec<String>>,
    /// Filesystem roots a path must live under when [`Self::restrict_to_sandbox`] is set.
    ///
    /// `None` falls back to `[workspace_root, sandbox_root_defaults]`; see
    /// [`crate::workspace::PathPolicy`].
    pub sandbox_root: Option<Vec<String>>,
    /// Rejects reads and writes outside [`Self::sandbox_root`]. Default `false`.
    pub restrict_to_sandbox: bool,
    /// Regexes (case-insensitive, see [`crate::shell::allowlist::CommandFilter`]) that reject a
    /// shell command before execution. `None` = no deny list (Python default).
    pub dangerous_patterns: Option<Vec<String>>,
    /// Root used to resolve relative paths. `None` = the process working directory at
    /// construction time (Python resolves against its `ContextVar` CWD).
    pub workspace_root: Option<String>,
    /// Maximum size of a file the local backend will read, in bytes.
    /// `None` = unlimited (Python parity).
    pub max_file_size_bytes: Option<u64>,
    /// Bytes retained per captured stdout/stderr stream before truncation.
    /// Defaults to [`DEFAULT_MAX_OUTPUT_BYTES`].
    pub max_output_bytes: u64,
    /// Default timeout for shell/code execution in seconds. Defaults to `300`.
    /// `0` disables the timeout.
    pub default_timeout_secs: u64,
    /// Environment scoping for shell and code subprocesses.
    pub env_policy: EnvPolicy,
}

impl Default for LocalWorkConfig {
    fn default() -> Self {
        Self {
            shell_allowlist: Some(
                DEFAULT_SHELL_ALLOWLIST
                    .iter()
                    .map(|entry| (*entry).to_string())
                    .collect(),
            ),
            sandbox_root: None,
            restrict_to_sandbox: false,
            dangerous_patterns: None,
            workspace_root: None,
            max_file_size_bytes: None,
            max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
            default_timeout_secs: DEFAULT_TIMEOUT_SECS,
            env_policy: EnvPolicy::default(),
        }
    }
}

impl LocalWorkConfig {
    /// Configuration rooted at `workspace` that accepts any shell command.
    ///
    /// Useful for tests and for hosts that apply their own command policy above this
    /// crate.
    #[must_use]
    pub fn unrestricted(workspace: impl Into<String>) -> Self {
        Self {
            shell_allowlist: None,
            workspace_root: Some(workspace.into()),
            env_policy: EnvPolicy::minimal(),
            ..Self::default()
        }
    }

    /// Returns this configuration with `workspace` as the path-resolution root.
    #[must_use]
    pub fn with_workspace(mut self, workspace: impl Into<String>) -> Self {
        self.workspace_root = Some(workspace.into());
        self
    }

    /// Returns this configuration restricted to the given roots.
    #[must_use]
    pub fn with_sandbox_roots<I, S>(mut self, roots: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.sandbox_root = Some(roots.into_iter().map(Into::into).collect());
        self.restrict_to_sandbox = true;
        self
    }

    /// Returns this configuration with an explicit shell allowlist.
    #[must_use]
    pub fn with_shell_allowlist<I, S>(mut self, allow: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.shell_allowlist = Some(allow.into_iter().map(Into::into).collect());
        self
    }

    /// Returns this configuration with the built-in dangerous-pattern list enabled.
    #[must_use]
    pub fn with_default_dangerous_patterns(mut self) -> Self {
        self.dangerous_patterns = Some(
            DEFAULT_DANGEROUS_PATTERNS
                .iter()
                .map(|pattern| (*pattern).to_string())
                .collect(),
        );
        self
    }

    /// The configured workspace root, or the current working directory.
    #[must_use]
    pub fn resolved_workspace_root(&self) -> std::path::PathBuf {
        if let Some(root) = &self.workspace_root {
            return std::path::PathBuf::from(root);
        }
        std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."))
    }

    /// Validates the configuration before a backend is built.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] when a dangerous
    /// pattern does not compile, when an allowlist entry is empty, when a sandbox root is
    /// empty, or when `max_output_bytes` is zero.
    pub fn validate(&self) -> Result<()> {
        if let Some(patterns) = &self.dangerous_patterns {
            for pattern in patterns {
                regex::Regex::new(pattern).map_err(|error| {
                    Error::validation(format!("invalid dangerous pattern '{pattern}': {error}"))
                })?;
            }
        }
        if let Some(allow) = &self.shell_allowlist {
            for entry in allow {
                if entry.trim().is_empty() {
                    return Err(Error::validation(
                        "shell_allowlist entries must not be blank",
                    ));
                }
            }
        }
        if let Some(roots) = &self.sandbox_root {
            for root in roots {
                if root.trim().is_empty() {
                    return Err(Error::validation("sandbox_root entries must not be blank"));
                }
            }
        }
        if self.max_output_bytes == 0 {
            return Err(Error::validation(
                "max_output_bytes must be greater than zero",
            ));
        }
        if self
            .workspace_root
            .as_ref()
            .is_some_and(|root| root.trim().is_empty())
        {
            return Err(Error::validation("workspace_root must not be blank"));
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_python() {
        let config = LocalWorkConfig::default();
        assert_eq!(
            config.shell_allowlist.as_deref().map(<[String]>::len),
            Some(DEFAULT_SHELL_ALLOWLIST.len())
        );
        assert_eq!(config.sandbox_root, None);
        assert!(!config.restrict_to_sandbox);
        assert_eq!(config.dangerous_patterns, None);
        assert_eq!(config.max_output_bytes, DEFAULT_MAX_OUTPUT_BYTES);
        assert_eq!(config.default_timeout_secs, 300);
        assert!(!config.env_policy.scrubs("PATH"));
    }

    #[test]
    fn json_uses_python_field_names() {
        let json = serde_json::to_value(LocalWorkConfig::default()).unwrap();
        for key in [
            "shell_allowlist",
            "sandbox_root",
            "restrict_to_sandbox",
            "dangerous_patterns",
        ] {
            assert!(json.get(key).is_some(), "missing {key}");
        }
        assert_eq!(json["shell_allowlist"].as_array().unwrap().len(), 30);
    }

    #[test]
    fn null_allowlist_means_allow_all() {
        let config: LocalWorkConfig = serde_json::from_str(r#"{"shell_allowlist": null}"#).unwrap();
        assert!(config.shell_allowlist.is_none());
        let config: LocalWorkConfig = serde_json::from_str("{}").unwrap();
        assert!(config.shell_allowlist.is_some());
    }

    #[test]
    fn empty_document_uses_defaults() {
        let config: LocalWorkConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(config, LocalWorkConfig::default());
    }

    #[test]
    fn validate_rejects_bad_patterns() {
        let config = LocalWorkConfig {
            dangerous_patterns: Some(vec!["(unclosed".to_string()]),
            ..LocalWorkConfig::default()
        };
        assert!(config.validate().is_err());
        let config = LocalWorkConfig {
            max_output_bytes: 0,
            ..LocalWorkConfig::default()
        };
        assert!(config.validate().is_err());
        assert!(LocalWorkConfig::default().validate().is_ok());
    }

    #[test]
    fn env_policy_scoping() {
        let policy = EnvPolicy::minimal();
        assert!(policy.inherits("PATH"));
        assert!(!policy.inherits("AWS_SECRET_ACCESS_KEY"));
        let policy = EnvPolicy {
            scrub: vec!["TOKEN".to_string()],
            ..EnvPolicy::default()
        };
        assert!(policy.inherits("PATH"));
        assert!(!policy.inherits("token"));
    }

    #[test]
    fn default_dangerous_patterns_compile() {
        for pattern in DEFAULT_DANGEROUS_PATTERNS {
            assert!(regex::Regex::new(pattern).is_ok(), "bad pattern {pattern}");
        }
    }
}
