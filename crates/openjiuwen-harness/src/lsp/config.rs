//! The LSP server process configuration.
//!
//! Ports the server descriptor of `openjiuwen/harness/lsp/servers/types.py` as serde data:
//! the executable, its arguments, environment, workspace root, language id and startup
//! timeout. Live process handles stay in [`StdioTransport`](crate::lsp::StdioTransport).
use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Default time allowed for the `initialize` handshake (Python's 45 s).
pub(super) const DEFAULT_STARTUP_TIMEOUT: Duration = Duration::from_secs(45);
/// Default timeout for one request (Python's 15 s).
pub(super) const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(15);
/// Default maximum file size the tool will synchronize (Python's 10 MiB).
pub(super) const DEFAULT_MAX_FILE_SIZE: u64 = 10 * 1024 * 1024;

/// Describes how to launch one language server.
///
/// Python parity: `LSPServerConfig` (`name`, `command`, `args`, `extension_to_language`,
/// `workspace_folder`, `initialization_options`). The Rust subset launches a single command
/// per workspace root and keeps the extra Python fields deferred.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct LspServerConfig {
    /// Executable to launch; resolved through `PATH` by the operating system.
    pub command: String,
    /// Arguments passed to the executable.
    pub args: Vec<String>,
    /// Extra environment variables layered over the inherited environment.
    pub env: BTreeMap<String, String>,
    /// Working directory and `rootUri` of the server; defaults to the tool's workspace root.
    pub root: Option<PathBuf>,
    /// LSP `languageId` sent in `textDocument/didOpen`.
    pub language_id: String,
    /// Time allowed for the `initialize` handshake, in milliseconds over the wire.
    #[serde(with = "duration_millis", default = "default_startup_timeout")]
    pub startup_timeout: Duration,
}

/// Returns the default startup timeout for serde.
const fn default_startup_timeout() -> Duration {
    DEFAULT_STARTUP_TIMEOUT
}

impl fmt::Debug for LspServerConfig {
    /// Renders the configuration with environment *keys* only, so secrets in `env` values
    /// never reach a log line.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LspServerConfig")
            .field("command", &self.command)
            .field("args", &self.args)
            .field("env_keys", &self.env.keys().collect::<Vec<_>>())
            .field("root", &self.root)
            .field("language_id", &self.language_id)
            .field("startup_timeout", &self.startup_timeout)
            .finish()
    }
}

impl Default for LspServerConfig {
    fn default() -> Self {
        Self {
            command: String::new(),
            args: Vec::new(),
            env: BTreeMap::new(),
            root: None,
            language_id: "plaintext".to_owned(),
            startup_timeout: DEFAULT_STARTUP_TIMEOUT,
        }
    }
}

impl LspServerConfig {
    /// Builds a configuration for `command` with the remaining fields at their defaults.
    #[must_use]
    pub fn new(command: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            ..Self::default()
        }
    }

    /// Returns this configuration with `args` appended.
    #[must_use]
    pub fn with_args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.args.extend(args.into_iter().map(Into::into));
        self
    }

    /// Returns this configuration with one environment variable added.
    #[must_use]
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.insert(key.into(), value.into());
        self
    }

    /// Returns this configuration rooted at `root`.
    #[must_use]
    pub fn with_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.root = Some(root.into());
        self
    }

    /// Returns this configuration with a different `language_id`.
    #[must_use]
    pub fn with_language_id(mut self, language_id: impl Into<String>) -> Self {
        self.language_id = language_id.into();
        self
    }

    /// Returns this configuration with a different startup timeout.
    #[must_use]
    pub fn with_startup_timeout(mut self, timeout: Duration) -> Self {
        self.startup_timeout = timeout;
        self
    }
}

/// Serializes a duration as whole milliseconds.
mod duration_millis {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serializer};

    /// Serializes `duration` as milliseconds.
    pub(super) fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(duration.as_millis().try_into().unwrap_or(u64::MAX))
    }

    /// Deserializes milliseconds into a duration.
    pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let millis = u64::deserialize(deserializer)?;
        Ok(Duration::from_millis(millis))
    }
}
