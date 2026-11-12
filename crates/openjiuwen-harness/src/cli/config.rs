//! Effective CLI options: flags, environment, settings file, defaults.
//!
//! The Python CLI resolves every knob inline in `openjiuwen/harness/cli/`
//! (`cli.py`, `agent/factory.py`). Rust funnels the same precedence chain through
//! [`CliOptions`] so the binary, the one-shot runner and the REPL all read one
//! value:
//!
//! 1. the command-line flag,
//! 2. an `openJiuwen`-specific environment variable (`OPENJIUWEN_*`),
//! 3. a generic SDK variable (`MODEL_NAME`, `API_KEY`, …),
//! 4. `~/.openjiuwen/settings.json`,
//! 5. the built-in default.
//!
//! [`CliOptions::resolve_with`] is the pure implementation of that chain: it only
//! reads through an injected [`EnvSource`], so precedence is testable without
//! touching the process environment. [`CliOptions::resolve`] is the single place
//! that reads the real environment and the settings file.
//!
//! Credentials never leak: [`CliOptions`] has a hand-written `Debug` that renders
//! the API key as `<redacted>`, and the value itself is held in a
//! [`SecretString`].

use std::collections::HashMap;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use openjiuwen_common::config::env::{self as env_vars, API_BASE, API_KEY, MODEL_NAME};
use openjiuwen_common::config::env::{MODEL_PROVIDER, OPENAI_API_KEY, OPENAI_BASE_URL};
use openjiuwen_common::config::OpenJiuwenHome;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::security::SecretString;
use openjiuwen_foundation::llm::{init_model, Model, ModelClientConfig, ModelRequestConfig};

use crate::permission::{PermissionLevel, PermissionsSection};

use super::session_store::SessionStore;

/// Default model provider when neither a flag nor the environment names one.
pub const DEFAULT_PROVIDER: &str = "openai";

/// Default model when neither a flag nor the environment names one.
pub const DEFAULT_MODEL: &str = "gpt-4o";

/// Default agent iteration budget.
pub const DEFAULT_MAX_ITERATIONS: u32 = 30;

/// Default per-response token budget.
pub const DEFAULT_MAX_TOKENS: u32 = 8192;

/// Default prompt language.
pub const DEFAULT_LANGUAGE: &str = "en";

/// Smallest accepted token budget; smaller values are rejected as unsafe.
pub const MIN_MAX_TOKENS: u32 = 256;

/// Prefix of every session id the CLI creates.
pub const SESSION_PREFIX: &str = "cli-";

/// Base URL of the model provider API: `OPENJIUWEN_API_BASE`.
pub const OPENJIUWEN_API_BASE: &str = "OPENJIUWEN_API_BASE";

/// Model name override: `OPENJIUWEN_MODEL`.
pub const OPENJIUWEN_MODEL: &str = "OPENJIUWEN_MODEL";

/// Model provider override: `OPENJIUWEN_PROVIDER`.
pub const OPENJIUWEN_PROVIDER: &str = "OPENJIUWEN_PROVIDER";

/// API key override: `OPENJIUWEN_API_KEY`.
pub const OPENJIUWEN_API_KEY: &str = "OPENJIUWEN_API_KEY";

/// Workspace directory override: `OPENJIUWEN_WORKSPACE`.
pub const OPENJIUWEN_WORKSPACE: &str = "OPENJIUWEN_WORKSPACE";

/// Iteration budget override: `OPENJIUWEN_MAX_ITERATIONS`.
pub const OPENJIUWEN_MAX_ITERATIONS: &str = "OPENJIUWEN_MAX_ITERATIONS";

/// Token budget override: `OPENJIUWEN_MAX_TOKENS`.
pub const OPENJIUWEN_MAX_TOKENS: &str = "OPENJIUWEN_MAX_TOKENS";

/// Runtime-state root override: `OPENJIUWEN_HOME`.
pub const OPENJIUWEN_HOME: &str = "OPENJIUWEN_HOME";

/// Reads environment variables for option resolution.
///
/// Implementations must treat an unset, empty, or whitespace-only value as
/// absent, matching [`env::var`](openjiuwen_common::config::env::var).
pub trait EnvSource {
    /// Returns the trimmed value of `name`, or `None` when it is unusable.
    fn get(&self, name: &str) -> Option<String>;
}

/// [`EnvSource`] backed by the process environment.
#[derive(Debug, Clone, Copy, Default)]
pub struct ProcessEnv;

impl EnvSource for ProcessEnv {
    fn get(&self, name: &str) -> Option<String> {
        env_vars::var(name)
    }
}

/// [`EnvSource`] backed by an in-memory map, for tests.
#[derive(Debug, Clone, Default)]
pub struct MapEnv {
    values: HashMap<String, String>,
}

impl MapEnv {
    /// Creates an empty map.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a map from `(name, value)` pairs.
    #[must_use]
    pub fn from_pairs(
        iter: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    ) -> Self {
        let mut env = Self::new();
        for (name, value) in iter {
            env.insert(name, value);
        }
        env
    }

    /// Inserts one `(name, value)` pair, replacing any previous value.
    pub fn insert(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.values.insert(name.into(), value.into());
    }
}

impl EnvSource for MapEnv {
    fn get(&self, name: &str) -> Option<String> {
        let value = self.values.get(name)?;
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_owned())
        }
    }
}

/// Contents of `~/.openjiuwen/settings.json`.
///
/// Every field is optional so a partial file only overrides the knobs it names;
/// unknown keys are ignored and the camelCase Python spellings (`apiKey`,
/// `apiBase`, `maxIterations`, `maxTokens`) are accepted as aliases. The API key
/// is stored in a [`SecretString`], so `Debug` redacts it.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct Settings {
    /// Model name override.
    pub model: Option<String>,
    /// Model provider override.
    pub provider: Option<String>,
    /// API key override.
    #[serde(alias = "apiKey")]
    pub api_key: Option<SecretString>,
    /// Base URL of the model provider API.
    #[serde(alias = "apiBase")]
    pub api_base: Option<String>,
    /// Workspace directory.
    pub workspace: Option<PathBuf>,
    /// Agent iteration budget.
    #[serde(alias = "maxIterations")]
    pub max_iterations: Option<u32>,
    /// Per-response token budget.
    #[serde(alias = "maxTokens")]
    pub max_tokens: Option<u32>,
}

impl Settings {
    /// Loads the settings file at `path`.
    ///
    /// A missing file is `Ok(None)`; a file that is not valid JSON is logged at
    /// `warn` and also yields `Ok(None)`, so a broken settings file never blocks
    /// the CLI. Other I/O failures are [`Error::Config`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when the file exists but cannot be read, naming
    /// the path.
    pub fn load(path: &Path) -> Result<Option<Self>> {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(Error::config(format!(
                    "failed to read settings file {}: {error}",
                    path.display()
                )))
            }
        };
        match Self::from_json(&text) {
            Ok(settings) => Ok(Some(settings)),
            Err(error) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %error,
                    "ignoring malformed settings file and using defaults"
                );
                Ok(None)
            }
        }
    }

    /// Parses settings from a JSON document.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when `text` is not a JSON object with the
    /// expected field types.
    pub fn from_json(text: &str) -> Result<Self> {
        serde_json::from_str(text)
            .map_err(|error| Error::config(format!("failed to parse settings file JSON: {error}")))
    }
}

/// Fully resolved options of one CLI invocation.
///
/// Built by [`CliOptions::resolve`] (or [`CliOptions::resolve_with`] in tests),
/// then handed to the runner and the REPL. The `Debug` implementation prints
/// every field but renders the API key as `<redacted>`.
#[derive(Clone)]
pub struct CliOptions {
    /// Model name to request.
    pub model: String,
    /// Model provider to talk to.
    pub provider: String,
    /// Provider credential, if one was supplied.
    pub api_key: Option<SecretString>,
    /// Base URL of the model provider API, if one was supplied.
    pub api_base: Option<String>,
    /// Absolute workspace directory the agent may use.
    pub workspace: PathBuf,
    /// Agent iteration budget.
    pub max_iterations: u32,
    /// Per-response token budget.
    pub max_tokens: u32,
    /// Whether streamed reasoning and diagnostics are shown.
    pub verbose: bool,
    /// Runtime-state root (settings, sessions, logs).
    pub home: OpenJiuwenHome,
}

impl std::fmt::Debug for CliOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let api_key = self.api_key.as_ref().map(|_| "<redacted>");
        f.debug_struct("CliOptions")
            .field("model", &self.model)
            .field("provider", &self.provider)
            .field("api_key", &api_key)
            .field("api_base", &self.api_base)
            .field("workspace", &self.workspace)
            .field("max_iterations", &self.max_iterations)
            .field("max_tokens", &self.max_tokens)
            .field("verbose", &self.verbose)
            .field("home", &self.home)
            .finish()
    }
}

impl CliOptions {
    /// Resolves the effective options from the command line, the process
    /// environment, and `~/.openjiuwen/settings.json`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when the settings file cannot be read or parsed
    /// at the I/O level, and whatever [`CliOptions::resolve_with`] validates.
    pub fn resolve(cli: &super::Cli) -> Result<Self> {
        let env = ProcessEnv;
        let home = home_from(&env);
        let settings = Settings::load(&home.join("settings.json"))?;
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        Self::resolve_with(cli, &env, settings.as_ref(), &cwd)
    }

    /// Resolves the effective options without touching the process environment.
    ///
    /// Precedence per field is flag → `OPENJIUWEN_*` → generic SDK variable →
    /// settings file → default, and every level is skipped when blank. A relative
    /// workspace is resolved against `cwd`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `OPENJIUWEN_MAX_ITERATIONS` or
    /// `OPENJIUWEN_MAX_TOKENS` is not an integer, when `max_tokens` is below
    /// [`MIN_MAX_TOKENS`], or when `max_iterations` is zero.
    pub fn resolve_with(
        cli: &super::Cli,
        env: &dyn EnvSource,
        settings: Option<&Settings>,
        cwd: &Path,
    ) -> Result<Self> {
        let home = home_from(env);

        let model = pick([
            cli.model.clone(),
            env.get(OPENJIUWEN_MODEL),
            env.get(MODEL_NAME),
            settings.and_then(|settings| settings.model.clone()),
        ])
        .unwrap_or_else(|| DEFAULT_MODEL.to_owned());

        let provider = pick([
            cli.provider.clone(),
            env.get(OPENJIUWEN_PROVIDER),
            env.get(MODEL_PROVIDER),
            settings.and_then(|settings| settings.provider.clone()),
        ])
        .unwrap_or_else(|| DEFAULT_PROVIDER.to_owned());

        let api_key = cli
            .api_key
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .map(SecretString::new)
            .or_else(|| env.get(OPENJIUWEN_API_KEY).map(SecretString::new))
            .or_else(|| env.get(API_KEY).map(SecretString::new))
            .or_else(|| env.get(OPENAI_API_KEY).map(SecretString::new))
            .or_else(|| {
                settings
                    .and_then(|settings| settings.api_key.clone())
                    .filter(|secret| !secret.is_empty())
            });

        let api_base = pick([
            cli.api_base.clone(),
            env.get(OPENJIUWEN_API_BASE),
            env.get(API_BASE),
            env.get(OPENAI_BASE_URL),
            settings.and_then(|settings| settings.api_base.clone()),
        ]);

        let workspace = cli
            .workspace
            .clone()
            .filter(|path| !path.as_os_str().is_empty())
            .or_else(|| env.get(OPENJIUWEN_WORKSPACE).map(PathBuf::from))
            .or_else(|| settings.and_then(|settings| settings.workspace.clone()))
            .unwrap_or_else(|| home.join("workspace"));
        let workspace = if workspace.is_relative() {
            cwd.join(workspace)
        } else {
            workspace
        };

        let max_iterations = match env.get(OPENJIUWEN_MAX_ITERATIONS) {
            Some(value) => value.parse::<u32>().map_err(|_| {
                Error::validation(format!(
                    "{OPENJIUWEN_MAX_ITERATIONS} must be an integer, got {value}"
                ))
            })?,
            None => settings
                .and_then(|settings| settings.max_iterations)
                .unwrap_or(DEFAULT_MAX_ITERATIONS),
        };

        let max_tokens = match env.get(OPENJIUWEN_MAX_TOKENS) {
            Some(value) => value.parse::<u32>().map_err(|_| {
                Error::validation(format!(
                    "{OPENJIUWEN_MAX_TOKENS} must be an integer, got {value}"
                ))
            })?,
            None => settings
                .and_then(|settings| settings.max_tokens)
                .unwrap_or(DEFAULT_MAX_TOKENS),
        };

        if max_tokens < MIN_MAX_TOKENS {
            return Err(Error::validation(format!(
                "max_tokens={max_tokens} is dangerously small (min {MIN_MAX_TOKENS}). Check {OPENJIUWEN_MAX_TOKENS}."
            )));
        }
        if max_iterations == 0 {
            return Err(Error::validation(format!(
                "max_iterations={max_iterations} must be >= 1."
            )));
        }

        Ok(Self {
            model,
            provider,
            api_key,
            api_base,
            workspace,
            max_iterations,
            max_tokens,
            verbose: cli.verbose,
            home,
        })
    }

    /// Creates the workspace directory if it does not exist yet.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] naming the workspace when the directory cannot
    /// be created.
    pub fn ensure_workspace(&self) -> Result<()> {
        std::fs::create_dir_all(&self.workspace).map_err(|error| {
            Error::config(format!(
                "failed to create workspace at {}: {error}",
                self.workspace.display()
            ))
        })
    }

    /// Builds the model client described by these options.
    ///
    /// This is the only place the CLI calls
    /// [`init_model`](openjiuwen_foundation::llm::init_model); the runner and the
    /// REPL accept a prebuilt model so tests never touch the network.
    ///
    /// # Errors
    ///
    /// Returns whatever model construction reports: [`Error::Config`] for a
    /// missing credential or endpoint, [`Error::Unsupported`] for an unknown
    /// provider.
    pub async fn build_model(&self) -> Result<Arc<dyn Model>> {
        let mut client = ModelClientConfig::new(self.provider.clone());
        if let Some(api_key) = &self.api_key {
            client = client.with_secret(api_key.clone());
        }
        if let Some(api_base) = &self.api_base {
            client = client.with_api_base(api_base.clone());
        }
        let request = ModelRequestConfig::new(self.model.clone()).with_max_tokens(self.max_tokens);
        init_model(client, request).await
    }

    /// Returns the session store rooted at this home directory.
    #[must_use]
    pub fn sessions(&self) -> SessionStore {
        SessionStore::new(&self.home)
    }

    /// Returns the permission section used by CLI sessions.
    ///
    /// `write_file` and `edit_file` require confirmation (Python
    /// `ConfirmInterruptRail` parity); every other field keeps its default,
    /// fail-closed value.
    #[must_use]
    pub fn permissions(&self) -> PermissionsSection {
        let mut permissions = PermissionsSection::default();
        permissions
            .tools
            .insert("write_file".to_owned(), PermissionLevel::Ask);
        permissions
            .tools
            .insert("edit_file".to_owned(), PermissionLevel::Ask);
        permissions
    }
}

/// Returns the first non-blank value, preserving its original spelling.
fn pick(values: impl IntoIterator<Item = Option<String>>) -> Option<String> {
    values
        .into_iter()
        .flatten()
        .find(|value| !value.trim().is_empty())
}

/// Resolves the runtime-state root from `env` without touching the process.
fn home_from(env: &dyn EnvSource) -> OpenJiuwenHome {
    env.get(OPENJIUWEN_HOME)
        .map_or_else(OpenJiuwenHome::resolve, OpenJiuwenHome::from_root)
}
