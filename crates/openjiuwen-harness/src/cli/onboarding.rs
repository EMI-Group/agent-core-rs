//! First-run onboarding wizard for the CLI.
//!
//! The wizard runs only when the CLI home has no `settings.json` yet **and** the
//! IO handle is interactive; a one-shot `run`, a piped stdin or
//! `OPENJIUWEN_NO_ONBOARDING` never prompts and never writes a file. This keeps
//! scripted and CI use byte-exact and side-effect free.
//!
//! The flow asks five questions and merges the answers into
//! `~/.openjiuwen/settings.json` (see [`settings_path`](super::paths::settings_path)):
//!
//! 1. `Configure a provider now? (y/n) ` — `n`/`no` declines, anything else
//!    (including an empty line) continues;
//! 2. `LLM Provider (openai, anthropic, dashscope, siliconflow) [openai]: `;
//! 3. `API Base URL [<provider default>]: `;
//! 4. `Model name [gpt-4o]: `;
//! 5. `API Key (blank to skip): `.
//!
//! A blank answer keeps the bracketed default. An unknown provider name is
//! accepted verbatim and simply reuses the `OpenAI` base URL as its default. The
//! API key is only ever held in [`SecretString`], never printed and never logged.
//! End of input at any prompt declines the wizard without writing anything.
//!
//! The file is written as pretty JSON plus a trailing newline through a
//! temporary sibling file and a rename, so a crash never leaves a partial
//! settings file behind.

use std::path::Path;

use openjiuwen_common::config::OpenJiuwenHome;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::security::SecretString;

use super::config::{EnvSource, Settings, DEFAULT_MODEL};
use super::io::CliIo;
use super::paths;

/// Provider pre-selected when the user just presses Enter.
pub const DEFAULT_PROVIDER_CHOICE: &str = "openai";

/// Providers offered by the wizard, in prompt order.
pub const PROVIDER_CHOICES: &[&str] = &["openai", "anthropic", "dashscope", "siliconflow"];

/// Model pre-selected when the user just presses Enter.
///
/// This is the CLI default model, kept in one place through
/// [`DEFAULT_MODEL`].
pub const DEFAULT_MODEL_CHOICE: &str = DEFAULT_MODEL;

/// Default OpenAI-compatible base URL, used for `openai` and unknown providers.
const OPENAI_DEFAULT_API_BASE: &str = "https://api.openai.com/v1";

/// Default base URL of the Anthropic API.
const ANTHROPIC_DEFAULT_API_BASE: &str = "https://api.anthropic.com";

/// Default base URL of the `DashScope` `OpenAI`-compatible API.
const DASHSCOPE_DEFAULT_API_BASE: &str = "https://dashscope.aliyuncs.com/compatible-mode/v1";

/// Default base URL of the `SiliconFlow` `OpenAI`-compatible API.
const SILICONFLOW_DEFAULT_API_BASE: &str = "https://api.siliconflow.cn/v1";

/// What the onboarding wizard did.
#[derive(Debug, Clone)]
pub enum OnboardingOutcome {
    /// Setup was not attempted: the terminal is not interactive, a settings file
    /// already exists, or the wizard is disabled by the environment.
    Skipped,
    /// The user declined setup, or the input ended before it finished; nothing
    /// was written.
    Declined,
    /// A settings file was written.
    Configured {
        /// Location of the written settings file.
        path: std::path::PathBuf,
        /// Settings as written (the merge of the wizard answers over any
        /// previously stored values).
        settings: Settings,
    },
}

/// Field-wise equality, because [`Settings`] does not implement `PartialEq`.
///
/// Every field of `Settings` is comparable, so this compares them all instead of
/// leaving the two `Configured` variants unequal.
fn settings_eq(left: &Settings, right: &Settings) -> bool {
    left.model == right.model
        && left.provider == right.provider
        && left.api_key == right.api_key
        && left.api_base == right.api_base
        && left.workspace == right.workspace
        && left.max_iterations == right.max_iterations
        && left.max_tokens == right.max_tokens
}

impl PartialEq for OnboardingOutcome {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Skipped, Self::Skipped) | (Self::Declined, Self::Declined) => true,
            (
                Self::Configured {
                    path: left_path,
                    settings: left_settings,
                },
                Self::Configured {
                    path: right_path,
                    settings: right_settings,
                },
            ) => left_path == right_path && settings_eq(left_settings, right_settings),
            _ => false,
        }
    }
}

impl Eq for OnboardingOutcome {}

/// First-run wizard bound to one IO handle and one CLI home directory.
///
/// Constructing a wizard touches nothing; [`OnboardingWizard::run`] is what
/// prompts and writes.
pub struct OnboardingWizard {
    /// Terminal the prompts are written to and read from.
    io: CliIo,
    /// CLI home directory the settings file lives in.
    home: OpenJiuwenHome,
}

impl OnboardingWizard {
    /// Builds a wizard over `io` writing under `home`.
    #[must_use]
    pub fn new(io: CliIo, home: OpenJiuwenHome) -> Self {
        Self { io, home }
    }

    /// Whether setup is needed: no settings file yet and the terminal is
    /// interactive.
    ///
    /// This is a pure filesystem probe: it never prompts and never writes.
    #[must_use]
    pub fn needed(home: &OpenJiuwenHome, interactive: bool) -> bool {
        interactive && !paths::settings_path(home).exists()
    }

    /// As [`OnboardingWizard::needed`], additionally honouring
    /// [`onboarding_disabled`](super::paths::onboarding_disabled).
    ///
    /// `OPENJIUWEN_NO_ONBOARDING` suppresses the wizard even on a fresh home with
    /// an interactive terminal.
    #[must_use]
    pub fn needed_with(home: &OpenJiuwenHome, env: &dyn EnvSource, interactive: bool) -> bool {
        !paths::onboarding_disabled(env) && Self::needed(home, interactive)
    }

    /// Runs the wizard, prompting on the bound IO handle.
    ///
    /// Returns [`OnboardingOutcome::Skipped`] without printing anything when the
    /// handle is not interactive or a settings file already exists, so calling
    /// this unconditionally is safe. End of input at any prompt yields
    /// [`OnboardingOutcome::Declined`] with nothing written.
    ///
    /// # Errors
    ///
    /// Returns whatever the IO handle reports, and [`Error::Config`] when the
    /// home directory or the settings file cannot be written.
    pub async fn run(&self) -> Result<OnboardingOutcome> {
        if !Self::needed(&self.home, self.io.is_interactive()) {
            return Ok(OnboardingOutcome::Skipped);
        }

        self.io.write_line("Welcome to OpenJiuwen CLI!").await?;
        self.io
            .write_line("No configuration found. Let's set one up.")
            .await?;
        self.io.write_blank().await?;

        match self
            .io
            .prompt_line("Configure a provider now? (y/n) ")
            .await?
        {
            None => return Ok(OnboardingOutcome::Declined),
            Some(answer) if declined(&answer) => {
                self.io.write_line("Skipped setup.").await?;
                return Ok(OnboardingOutcome::Declined);
            }
            Some(_) => {}
        }

        let provider = match self
            .io
            .prompt_line("LLM Provider (openai, anthropic, dashscope, siliconflow) [openai]: ")
            .await?
        {
            None => return Ok(OnboardingOutcome::Declined),
            Some(answer) if answer.is_empty() => DEFAULT_PROVIDER_CHOICE.to_owned(),
            Some(answer) => answer,
        };

        let default_api_base = default_api_base(&provider);
        let api_base = match self
            .io
            .prompt_line(&format!("API Base URL [{default_api_base}]: "))
            .await?
        {
            None => return Ok(OnboardingOutcome::Declined),
            Some(answer) if answer.is_empty() => default_api_base.to_owned(),
            Some(answer) => answer,
        };

        let model = match self
            .io
            .prompt_line(&format!("Model name [{DEFAULT_MODEL_CHOICE}]: "))
            .await?
        {
            None => return Ok(OnboardingOutcome::Declined),
            Some(answer) if answer.is_empty() => DEFAULT_MODEL_CHOICE.to_owned(),
            Some(answer) => answer,
        };

        let api_key = match self.io.prompt_line("API Key (blank to skip): ").await? {
            None => return Ok(OnboardingOutcome::Declined),
            Some(answer) if answer.is_empty() => None,
            Some(answer) => Some(SecretString::new(answer)),
        };

        let path = paths::settings_path(&self.home);
        self.home.ensure()?;
        let mut settings = Settings::load(&path)?.unwrap_or_default();
        settings.provider = Some(provider);
        settings.model = Some(model);
        settings.api_base = Some(api_base);
        if let Some(api_key) = api_key {
            settings.api_key = Some(api_key);
        }
        write_settings(&path, &settings)?;

        self.io
            .write_line(&format!("Config saved to {}", path.display()))
            .await?;
        self.io.write_blank().await?;
        Ok(OnboardingOutcome::Configured { path, settings })
    }
}

/// Entry point for the CLI: no-op unless setup is needed.
///
/// Skips silently when the terminal is not interactive, a settings file already
/// exists, or `OPENJIUWEN_NO_ONBOARDING` is set; otherwise it runs the wizard
/// over `io`.
///
/// # Errors
///
/// See [`OnboardingWizard::run`].
pub async fn maybe_run(
    io: &CliIo,
    home: &OpenJiuwenHome,
    env: &dyn EnvSource,
) -> Result<OnboardingOutcome> {
    if !OnboardingWizard::needed_with(home, env, io.is_interactive()) {
        return Ok(OnboardingOutcome::Skipped);
    }
    OnboardingWizard::new(io.clone(), home.clone()).run().await
}

/// Returns whether an answer declines setup: `n` or `no`, case-insensitive.
fn declined(answer: &str) -> bool {
    matches!(answer.trim().to_ascii_lowercase().as_str(), "n" | "no")
}

/// Returns the default base URL of `provider`.
///
/// Unknown provider names are accepted verbatim and fall back to the `OpenAI`
/// default, matching the wizard's "only the default mapping" rule.
fn default_api_base(provider: &str) -> &'static str {
    match provider {
        "anthropic" => ANTHROPIC_DEFAULT_API_BASE,
        "dashscope" => DASHSCOPE_DEFAULT_API_BASE,
        "siliconflow" => SILICONFLOW_DEFAULT_API_BASE,
        _ => OPENAI_DEFAULT_API_BASE,
    }
}

/// Writes `settings` to `path` as pretty JSON plus a trailing newline.
///
/// The document goes to `<path>.tmp` first and is renamed over `path`, so a
/// crash never exposes a partial settings file.
fn write_settings(path: &Path, settings: &Settings) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|error| {
            Error::config(format!(
                "failed to create settings directory {}: {error}",
                dir.display()
            ))
        })?;
    }
    let mut json = serde_json::to_string_pretty(settings).map_err(|error| {
        Error::config(format!(
            "failed to encode settings file {}: {error}",
            path.display()
        ))
    })?;
    json.push('\n');
    let temp = temp_path(path);
    std::fs::write(&temp, json).map_err(|error| {
        Error::config(format!(
            "failed to write settings file {}: {error}",
            temp.display()
        ))
    })?;
    std::fs::rename(&temp, path).map_err(|error| {
        Error::config(format!(
            "failed to replace settings file {}: {error}",
            path.display()
        ))
    })
}

/// Returns the sibling temporary path used for the atomic write.
fn temp_path(path: &Path) -> std::path::PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".tmp");
    std::path::PathBuf::from(name)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use tempfile::TempDir;

    use super::*;
    use crate::cli::config::MapEnv;
    use crate::cli::io::test_support::{output, SharedWriter};
    use crate::cli::io::LineReader;
    use crate::cli::paths::OPENJIUWEN_NO_ONBOARDING;

    /// [`LineReader`] that answers from a fixed script, then reports EOF.
    struct ScriptedReader {
        /// Lines still to hand out.
        lines: VecDeque<String>,
    }

    #[async_trait]
    impl LineReader for ScriptedReader {
        async fn read_line(&mut self) -> Result<Option<String>> {
            Ok(self.lines.pop_front())
        }
    }

    /// Builds an interactive handle over `lines` and a fresh output buffer.
    fn interactive(lines: &[&str]) -> (CliIo, Arc<Mutex<String>>) {
        let buffer = Arc::new(Mutex::new(String::new()));
        let reader = Box::new(ScriptedReader {
            lines: lines.iter().map(|line| (*line).to_owned()).collect(),
        });
        let writer = Box::new(SharedWriter {
            buffer: Arc::clone(&buffer),
        });
        (CliIo::with_interactive(reader, writer), buffer)
    }

    /// Builds a non-interactive handle over `lines` and a fresh output buffer.
    fn non_interactive(lines: &[&str]) -> (CliIo, Arc<Mutex<String>>) {
        let buffer = Arc::new(Mutex::new(String::new()));
        let reader = Box::new(ScriptedReader {
            lines: lines.iter().map(|line| (*line).to_owned()).collect(),
        });
        let writer = Box::new(SharedWriter {
            buffer: Arc::clone(&buffer),
        });
        (CliIo::new(reader, writer), buffer)
    }

    /// Returns the home rooted at a fresh temporary directory.
    fn home() -> (TempDir, OpenJiuwenHome) {
        let temp = TempDir::new().unwrap();
        let home = OpenJiuwenHome::from_root(temp.path());
        (temp, home)
    }

    #[tokio::test]
    async fn happy_path_writes_the_settings_file() {
        let (_temp, home) = home();
        let (io, buffer) = interactive(&["y", "anthropic", "", "claude-3-5-sonnet", "sk-test"]);
        let wizard = OnboardingWizard::new(io, home.clone());

        let outcome = wizard.run().await.unwrap();
        let path = paths::settings_path(&home);
        match &outcome {
            OnboardingOutcome::Configured {
                path: written,
                settings,
            } => {
                assert_eq!(written, &path);
                assert_eq!(settings.provider.as_deref(), Some("anthropic"));
                assert_eq!(settings.model.as_deref(), Some("claude-3-5-sonnet"));
                assert_eq!(
                    settings.api_base.as_deref(),
                    Some("https://api.anthropic.com")
                );
                assert_eq!(
                    settings.api_key.as_ref().map(SecretString::expose_secret),
                    Some("sk-test")
                );
            }
            other => panic!("unexpected outcome: {other:?}"),
        }

        let loaded = Settings::load(&path).unwrap().unwrap();
        assert_eq!(loaded.provider.as_deref(), Some("anthropic"));
        assert_eq!(loaded.model.as_deref(), Some("claude-3-5-sonnet"));
        assert_eq!(
            loaded.api_base.as_deref(),
            Some("https://api.anthropic.com")
        );
        assert_eq!(
            loaded.api_key.as_ref().map(SecretString::expose_secret),
            Some("sk-test")
        );
        assert!(output(&buffer).contains("Config saved to"));
        assert!(output(&buffer).contains("Welcome to OpenJiuwen CLI!"));
    }

    #[tokio::test]
    async fn blank_answers_keep_the_defaults() {
        let (_temp, home) = home();
        let (io, _buffer) = interactive(&["", "", "", "", ""]);
        let outcome = OnboardingWizard::new(io, home.clone()).run().await.unwrap();

        match outcome {
            OnboardingOutcome::Configured { settings, .. } => {
                assert_eq!(settings.provider.as_deref(), Some(DEFAULT_PROVIDER_CHOICE));
                assert_eq!(settings.model.as_deref(), Some(DEFAULT_MODEL_CHOICE));
                assert_eq!(
                    settings.api_base.as_deref(),
                    Some("https://api.openai.com/v1")
                );
                assert!(settings.api_key.is_none());
            }
            other => panic!("unexpected outcome: {other:?}"),
        }
    }

    #[tokio::test]
    async fn declining_writes_nothing() {
        let (_temp, home) = home();
        let (io, buffer) = interactive(&["n"]);
        let outcome = OnboardingWizard::new(io, home.clone()).run().await.unwrap();

        assert_eq!(outcome, OnboardingOutcome::Declined);
        assert!(!paths::settings_path(&home).exists());
        assert!(output(&buffer).contains("Skipped setup."));
    }

    #[tokio::test]
    async fn end_of_input_at_any_prompt_declines() {
        for script in [&[][..], &["y"][..], &["y", "openai"][..]] {
            let (_temp, home) = home();
            let (io, _buffer) = interactive(script);
            let outcome = OnboardingWizard::new(io, home.clone()).run().await.unwrap();
            assert_eq!(outcome, OnboardingOutcome::Declined, "script: {script:?}");
            assert!(!paths::settings_path(&home).exists());
        }
    }

    #[tokio::test]
    async fn non_interactive_handle_skips_silently() {
        let (_temp, home) = home();
        let (io, buffer) = non_interactive(&["y", "openai"]);
        let outcome = OnboardingWizard::new(io, home.clone()).run().await.unwrap();

        assert_eq!(outcome, OnboardingOutcome::Skipped);
        assert_eq!(output(&buffer), "");
        assert!(!paths::settings_path(&home).exists());
    }

    #[tokio::test]
    async fn blank_api_key_writes_no_key() {
        let (_temp, home) = home();
        let (io, _buffer) = interactive(&["y", "openai", "", "", ""]);
        OnboardingWizard::new(io, home.clone()).run().await.unwrap();

        let text = std::fs::read_to_string(paths::settings_path(&home)).unwrap();
        let value: serde_json::Value = serde_json::from_str(&text).unwrap();
        match value.get("api_key") {
            None => {}
            Some(key) => assert!(key.is_null(), "unexpected api_key value: {key}"),
        }
        match value.get("apiKey") {
            None => {}
            Some(key) => assert!(key.is_null(), "unexpected apiKey value: {key}"),
        }
    }

    #[test]
    fn needed_truth_table() {
        let (temp, home) = home();
        assert!(OnboardingWizard::needed(&home, true));
        assert!(!OnboardingWizard::needed(&home, false));

        std::fs::write(paths::settings_path(&home), "{}").unwrap();
        assert!(!OnboardingWizard::needed(&home, true));
        assert!(!OnboardingWizard::needed(&home, false));
        assert!(temp.path().join("settings.json").exists());
    }

    #[test]
    fn needed_with_honours_the_disable_switch() {
        let (_temp, home) = home();
        let enabled = MapEnv::new();
        let disabled = MapEnv::from_pairs([(OPENJIUWEN_NO_ONBOARDING, "1")]);
        assert!(OnboardingWizard::needed_with(&home, &enabled, true));
        assert!(!OnboardingWizard::needed_with(&home, &disabled, true));

        std::fs::write(paths::settings_path(&home), "{}").unwrap();
        assert!(!OnboardingWizard::needed_with(&home, &enabled, true));
    }

    #[tokio::test]
    async fn maybe_run_skips_when_a_settings_file_exists() {
        let (_temp, home) = home();
        std::fs::write(paths::settings_path(&home), "{}").unwrap();
        let (io, buffer) = interactive(&["y"]);
        let outcome = maybe_run(&io, &home, &MapEnv::new()).await.unwrap();

        assert_eq!(outcome, OnboardingOutcome::Skipped);
        assert_eq!(output(&buffer), "");
    }

    #[tokio::test]
    async fn maybe_run_honours_the_disable_switch() {
        let (_temp, home) = home();
        let (io, buffer) = interactive(&["y"]);
        let env = MapEnv::from_pairs([(OPENJIUWEN_NO_ONBOARDING, "1")]);
        let outcome = maybe_run(&io, &home, &env).await.unwrap();

        assert_eq!(outcome, OnboardingOutcome::Skipped);
        assert_eq!(output(&buffer), "");
        assert!(!paths::settings_path(&home).exists());
    }
}
