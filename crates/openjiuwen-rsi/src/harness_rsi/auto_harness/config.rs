//! Serialisable configuration of an auto-harness optimisation session.
//!
//! Ported from Python `openjiuwen.rsi.harness_rsi.auto_harness.schema.AutoHarnessConfig`,
//! reduced to the data-only subset the Rust port owns: no live models, no clients,
//! no environment lookups. Environment loading happens in binaries, never here.
//!
//! The Python defaults are preserved for every field. Unknown YAML/JSON keys are
//! tolerated and collected into [`AutoHarnessConfig::extra`] so a newer Python
//! config file still loads.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Python `_DEFAULT_REPO_URL`.
const DEFAULT_REPO_URL: &str = "https://gitcode.com/openJiuwen/agent-core.git";

/// Python fallback data directory (`.auto_harness/`) used when `data_dir` is empty.
const FALLBACK_DATA_DIR: &str = ".auto_harness";

/// Which auto-harness pipeline the session should prefer.
///
/// Mirrors Python `pipeline_preference`, whose user-facing values are
/// `auto`, `meta`, and `extended`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PipelinePreference {
    /// Let the pipeline selector choose (Python default).
    #[default]
    Auto,
    /// The meta-evolve pipeline.
    Meta,
    /// The extended-evolve pipeline.
    Extended,
}

/// Configuration of one auto-harness optimisation session.
///
/// Every field that Python reads from YAML is `snake_case` and defaulted, so a
/// partial document yields the Python defaults for the missing keys. Unknown
/// keys are retained in [`Self::extra`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct AutoHarnessConfig {
    /// Root directory for every session artifact (experience, worktrees, runs).
    ///
    /// Python leaves this empty and lets the host CLI supply it.
    pub data_dir: PathBuf,
    /// Optional local checkout of the target repository used to speed up worktrees.
    ///
    /// Python default is the empty string, represented by an empty [`PathBuf`].
    pub local_repo: PathBuf,
    /// Git URL cloned into the session cache when no local repository is configured.
    pub repo_url: String,
    /// Path of the YAML file this configuration was loaded from, when known.
    pub config_path: PathBuf,
    /// Explicit experience directory; when empty, [`Self::resolved_experience_dir`]
    /// derives one from `data_dir`.
    pub experience_dir: PathBuf,
    /// Extra skill directories made available to the agent.
    pub skills_dirs: Vec<PathBuf>,
    /// Prompt language (`"cn"` by default, Python parity).
    pub language: String,
    /// Free-form description of what the session should optimise.
    pub optimization_goal: String,
    /// Preferred optimization pipeline.
    pub pipeline_preference: PipelinePreference,
    /// Wall-clock budget of one session, in seconds (default 900 000 s).
    pub session_budget_secs: f64,
    /// API cost budget of one session, in US dollars (default 10.0).
    pub cost_limit_usd: f64,
    /// Timeout of a single task, in seconds (default 300 000 s).
    pub task_timeout_secs: f64,
    /// Timeout of a single model call, in seconds (default 300 000 s).
    pub model_timeout_secs: f64,
    /// Maximum number of optimisation tasks per session (default 10).
    pub max_tasks_per_session: usize,
    /// Maximum number of optimisation iterations per session (default 10).
    pub max_iterations: u32,
    /// Score at which the session may stop early; `None` disables the target.
    pub target_score: Option<f64>,
    /// Minimum best-score improvement that counts as progress (default 0.05).
    pub min_improvement: f64,
    /// Number of consecutive flat iterations before declaring convergence (default 2).
    pub patience: usize,
    /// Per-stage agent iteration budgets, overlaid on the Python defaults.
    pub agent_iterations: BTreeMap<String, u32>,
    /// Files the agent must never modify.
    pub immutable_files: Vec<String>,
    /// Path prefixes whose modification is considered high impact.
    pub high_impact_prefixes: Vec<String>,
    /// Git branch used as the base of generated work (default `"develop"`).
    pub git_base_branch: String,
    /// Retry budget of the first fix phase (default 10).
    pub fix_phase1_max_retries: u32,
    /// Retry budget of the second fix phase (default 9).
    pub fix_phase2_max_retries: u32,
    /// Whether the config file was synthesised during bootstrap.
    pub config_bootstrapped: bool,
    /// Locally detected repository suggested to the user, when none is configured.
    pub suggested_local_repo: Option<PathBuf>,
    /// Unknown keys from a newer configuration document, preserved verbatim.
    #[serde(flatten)]
    pub extra: Value,
}

impl Default for AutoHarnessConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::new(),
            local_repo: PathBuf::new(),
            repo_url: DEFAULT_REPO_URL.to_owned(),
            config_path: PathBuf::new(),
            experience_dir: PathBuf::new(),
            skills_dirs: Vec::new(),
            language: "cn".to_owned(),
            optimization_goal: String::new(),
            pipeline_preference: PipelinePreference::Auto,
            session_budget_secs: 900_000.0,
            cost_limit_usd: 10.0,
            task_timeout_secs: 300_000.0,
            model_timeout_secs: 300_000.0,
            max_tasks_per_session: 10,
            max_iterations: 10,
            target_score: None,
            min_improvement: 0.05,
            patience: 2,
            agent_iterations: Self::default_agent_iterations(),
            immutable_files: Vec::new(),
            high_impact_prefixes: vec!["openjiuwen/core/".to_owned()],
            git_base_branch: "develop".to_owned(),
            fix_phase1_max_retries: 10,
            fix_phase2_max_retries: 9,
            config_bootstrapped: false,
            suggested_local_repo: None,
            extra: Value::Object(serde_json::Map::new()),
        }
    }
}

impl AutoHarnessConfig {
    /// The Python `agent_iterations` defaults.
    ///
    /// `implement` 30, `assess` 30, `plan` 15, `select_pipeline` 10, `eval` 10,
    /// `pr_draft` 5, `learnings` 5, `explore_subagent` 20,
    /// `browser_subagent` 20, `merge_ext` 8.
    #[must_use]
    pub fn default_agent_iterations() -> BTreeMap<String, u32> {
        [
            ("implement", 30),
            ("assess", 30),
            ("plan", 15),
            ("select_pipeline", 10),
            ("eval", 10),
            ("pr_draft", 5),
            ("learnings", 5),
            ("explore_subagent", 20),
            ("browser_subagent", 20),
            ("merge_ext", 8),
        ]
        .into_iter()
        .map(|(stage, iterations)| (stage.to_owned(), iterations))
        .collect()
    }

    /// Parse a configuration from an already-decoded JSON value.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when the value does not match the expected
    /// shape (wrong scalar type, unknown enum value, …).
    pub fn from_json_value(value: Value) -> Result<Self> {
        serde_json::from_value(value)
            .map_err(|error| Error::validation(format!("invalid auto-harness config: {error}")))
    }

    /// Parse a configuration from a YAML document.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when the YAML cannot be parsed or does not
    /// match the expected shape.
    pub fn from_yaml_str(yaml: &str) -> Result<Self> {
        serde_yaml_ng::from_str(yaml)
            .map_err(|error| Error::validation(format!("invalid auto-harness config: {error}")))
    }

    /// Read and parse the YAML configuration at `path`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::config`] when the file cannot be read and
    /// [`Error::validation`] when its contents are not a valid configuration.
    pub async fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let contents = tokio::fs::read_to_string(path).await.map_err(|error| {
            Error::config(format!(
                "failed to read auto-harness config {}: {error}",
                path.display()
            ))
        })?;
        Self::from_yaml_str(&contents)
    }

    /// Load the configuration at `path`, falling back to [`Self::default`].
    ///
    /// Python bootstrap semantics: a missing, unreadable, or invalid file is not
    /// an error — the session starts from the Python defaults. Unlike Python this
    /// implementation never writes a template, so the returned defaults keep
    /// `config_bootstrapped == false` and an empty `config_path`; a successfully
    /// loaded file keeps whatever `config_bootstrapped` value it contains.
    pub async fn load_or_default(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref();
        match Self::load(path).await {
            Ok(config) => config,
            Err(error) => {
                tracing::debug!(
                    target: "openjiuwen_rsi::auto_harness::config",
                    path = %path.display(),
                    %error,
                    "auto-harness config unavailable, using defaults"
                );
                Self::default()
            }
        }
    }

    /// Reject configurations that cannot drive a session.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when a path or URL is empty, a budget is not
    /// a positive finite number, a count is zero, `min_improvement` is negative or
    /// non-finite, `target_score` is non-finite, or `extra` is not a JSON object.
    pub fn validate(&self) -> Result<()> {
        if self.data_dir.as_os_str().is_empty() {
            return Err(Error::validation("auto-harness data_dir must not be empty"));
        }
        if self.repo_url.trim().is_empty() {
            return Err(Error::validation("auto-harness repo_url must not be empty"));
        }
        if self.git_base_branch.trim().is_empty() {
            return Err(Error::validation(
                "auto-harness git_base_branch must not be empty",
            ));
        }
        require_positive_finite("session_budget_secs", self.session_budget_secs)?;
        require_positive_finite("cost_limit_usd", self.cost_limit_usd)?;
        require_positive_finite("task_timeout_secs", self.task_timeout_secs)?;
        require_positive_finite("model_timeout_secs", self.model_timeout_secs)?;
        if self.max_tasks_per_session == 0 {
            return Err(Error::validation(
                "auto-harness max_tasks_per_session must be greater than zero",
            ));
        }
        if self.max_iterations == 0 {
            return Err(Error::validation(
                "auto-harness max_iterations must be greater than zero",
            ));
        }
        if self.patience == 0 {
            return Err(Error::validation(
                "auto-harness patience must be greater than zero",
            ));
        }
        if !self.min_improvement.is_finite() || self.min_improvement < 0.0 {
            return Err(Error::validation(format!(
                "auto-harness min_improvement must be a non-negative finite number, got {}",
                self.min_improvement
            )));
        }
        if let Some(target) = self.target_score {
            if !target.is_finite() {
                return Err(Error::validation(format!(
                    "auto-harness target_score must be finite, got {target}"
                )));
            }
        }
        if !self.extra.is_object() {
            return Err(Error::validation(
                "auto-harness config extra fields must be a JSON object",
            ));
        }
        Ok(())
    }

    /// Per-stage iteration budgets: the Python defaults overlaid with the
    /// configured entries.
    #[must_use]
    pub fn resolve_agent_iterations(&self) -> BTreeMap<String, u32> {
        let mut resolved = Self::default_agent_iterations();
        for (stage, iterations) in &self.agent_iterations {
            resolved.insert(stage.clone(), *iterations);
        }
        resolved
    }

    /// Experience directory: `experience_dir` when set, else `<data_dir>/experience`.
    ///
    /// When `data_dir` is empty the Python fallback `.auto_harness/experience` is used.
    #[must_use]
    pub fn resolved_experience_dir(&self) -> PathBuf {
        if !self.experience_dir.as_os_str().is_empty() {
            return self.experience_dir.clone();
        }
        self.data_base_dir().join("experience")
    }

    /// Worktree root: `<data_dir>/worktrees`, or `.auto_harness/worktrees`.
    #[must_use]
    pub fn resolved_worktrees_dir(&self) -> PathBuf {
        self.data_base_dir().join("worktrees")
    }

    /// Run-record directory: `<data_dir>/runs`, or `.auto_harness/runs`.
    #[must_use]
    pub fn resolved_runs_dir(&self) -> PathBuf {
        self.data_base_dir().join("runs")
    }

    /// Clone-cache directory: `<data_dir>/cache/repo`, or `.auto_harness/cache/repo`.
    #[must_use]
    pub fn resolved_cache_repo_dir(&self) -> PathBuf {
        self.data_base_dir().join("cache").join("repo")
    }

    /// `data_dir`, falling back to the Python `.auto_harness` default when empty.
    fn data_base_dir(&self) -> PathBuf {
        if self.data_dir.as_os_str().is_empty() {
            PathBuf::from(FALLBACK_DATA_DIR)
        } else {
            self.data_dir.clone()
        }
    }
}

/// Validate that `value` is a strictly positive finite number.
fn require_positive_finite(name: &str, value: f64) -> Result<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(Error::validation(format!(
            "auto-harness {name} must be a positive finite number, got {value}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_config() -> AutoHarnessConfig {
        AutoHarnessConfig {
            data_dir: PathBuf::from("/tmp/rsi-data"),
            ..AutoHarnessConfig::default()
        }
    }

    #[test]
    fn defaults_match_python() {
        let config = AutoHarnessConfig::default();
        assert_eq!(
            config.repo_url,
            "https://gitcode.com/openJiuwen/agent-core.git"
        );
        assert_eq!(config.language, "cn");
        assert_eq!(config.pipeline_preference, PipelinePreference::Auto);
        assert_eq!(config.session_budget_secs, 900_000.0);
        assert_eq!(config.cost_limit_usd, 10.0);
        assert_eq!(config.task_timeout_secs, 300_000.0);
        assert_eq!(config.model_timeout_secs, 300_000.0);
        assert_eq!(config.max_tasks_per_session, 10);
        assert_eq!(config.max_iterations, 10);
        assert_eq!(config.target_score, None);
        assert_eq!(config.min_improvement, 0.05);
        assert_eq!(config.patience, 2);
        assert_eq!(config.git_base_branch, "develop");
        assert_eq!(config.fix_phase1_max_retries, 10);
        assert_eq!(config.fix_phase2_max_retries, 9);
        assert!(!config.config_bootstrapped);
        assert_eq!(config.suggested_local_repo, None);
        assert_eq!(config.high_impact_prefixes, vec!["openjiuwen/core/"]);
        assert_eq!(config.immutable_files, Vec::<String>::new());
        assert!(config.data_dir.as_os_str().is_empty());
        assert!(config.local_repo.as_os_str().is_empty());
        assert!(config.extra.is_object());

        let iterations = AutoHarnessConfig::default_agent_iterations();
        assert_eq!(iterations.len(), 10);
        assert_eq!(iterations["implement"], 30);
        assert_eq!(iterations["assess"], 30);
        assert_eq!(iterations["plan"], 15);
        assert_eq!(iterations["select_pipeline"], 10);
        assert_eq!(iterations["eval"], 10);
        assert_eq!(iterations["pr_draft"], 5);
        assert_eq!(iterations["learnings"], 5);
        assert_eq!(iterations["explore_subagent"], 20);
        assert_eq!(iterations["browser_subagent"], 20);
        assert_eq!(iterations["merge_ext"], 8);
    }

    #[test]
    fn yaml_parsing_overlays_defaults_and_keeps_unknown_keys() {
        let yaml = r"
data_dir: /tmp/rsi
local_repo: /tmp/repo
pipeline_preference: meta
max_iterations: 3
agent_iterations:
  implement: 42
future_python_key: 7
";
        let config = AutoHarnessConfig::from_yaml_str(yaml).unwrap();
        assert_eq!(config.data_dir, PathBuf::from("/tmp/rsi"));
        assert_eq!(config.local_repo, PathBuf::from("/tmp/repo"));
        assert_eq!(config.pipeline_preference, PipelinePreference::Meta);
        assert_eq!(config.max_iterations, 3);
        assert_eq!(config.language, "cn");
        assert_eq!(config.agent_iterations["implement"], 42);
        assert_eq!(config.extra["future_python_key"], 7);
    }

    #[test]
    fn yaml_round_trips_with_unknown_fields() {
        let mut config = valid_config();
        config.pipeline_preference = PipelinePreference::Extended;
        config
            .extra
            .as_object_mut()
            .unwrap()
            .insert("custom".to_owned(), Value::Bool(true));

        let yaml = serde_yaml_ng::to_string(&config).unwrap();
        let reparsed = AutoHarnessConfig::from_yaml_str(&yaml).unwrap();
        assert_eq!(reparsed, config);
    }

    #[test]
    fn json_parsing_rejects_wrong_types() {
        let value = serde_json::json!({ "data_dir": 5, "max_iterations": "many" });
        let error = AutoHarnessConfig::from_json_value(value).unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
    }

    #[test]
    fn yaml_parsing_rejects_invalid_documents() {
        let error = AutoHarnessConfig::from_yaml_str("data_dir: [unclosed").unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
    }

    #[test]
    fn validate_accepts_a_full_config() {
        let mut config = valid_config();
        config.target_score = Some(0.9);
        config.experience_dir = PathBuf::from("/tmp/rsi/experience");
        config.validate().unwrap();
    }

    #[test]
    fn validate_rejects_bad_values() {
        let mut config = valid_config();
        config.data_dir = PathBuf::new();
        assert!(config.validate().is_err());

        let mut config = valid_config();
        config.repo_url = "  ".to_owned();
        assert!(config.validate().is_err());

        let mut config = valid_config();
        config.git_base_branch = String::new();
        assert!(config.validate().is_err());

        let mut config = valid_config();
        config.session_budget_secs = 0.0;
        assert!(config.validate().is_err());

        let mut config = valid_config();
        config.cost_limit_usd = f64::NAN;
        assert!(config.validate().is_err());

        let mut config = valid_config();
        config.task_timeout_secs = -1.0;
        assert!(config.validate().is_err());

        let mut config = valid_config();
        config.model_timeout_secs = f64::INFINITY;
        assert!(config.validate().is_err());

        let mut config = valid_config();
        config.max_tasks_per_session = 0;
        assert!(config.validate().is_err());

        let mut config = valid_config();
        config.max_iterations = 0;
        assert!(config.validate().is_err());

        let mut config = valid_config();
        config.patience = 0;
        assert!(config.validate().is_err());

        let mut config = valid_config();
        config.min_improvement = -0.1;
        assert!(config.validate().is_err());

        let mut config = valid_config();
        config.min_improvement = f64::NAN;
        assert!(config.validate().is_err());

        let mut config = valid_config();
        config.target_score = Some(f64::NAN);
        assert!(config.validate().is_err());

        let mut config = valid_config();
        config.extra = Value::Null;
        assert!(config.validate().is_err());
    }

    #[test]
    fn resolve_agent_iterations_overlays_config() {
        let mut config = AutoHarnessConfig::default();
        config.agent_iterations.insert("plan".to_owned(), 99);
        config.agent_iterations.insert("custom".to_owned(), 4);

        let resolved = config.resolve_agent_iterations();
        assert_eq!(resolved["plan"], 99);
        assert_eq!(resolved["custom"], 4);
        assert_eq!(resolved["implement"], 30);
        assert_eq!(resolved["merge_ext"], 8);
    }

    #[test]
    fn derived_paths_live_under_data_dir() {
        let config = valid_config();
        assert_eq!(
            config.resolved_experience_dir(),
            PathBuf::from("/tmp/rsi-data/experience")
        );
        assert_eq!(
            config.resolved_worktrees_dir(),
            PathBuf::from("/tmp/rsi-data/worktrees")
        );
        assert_eq!(
            config.resolved_runs_dir(),
            PathBuf::from("/tmp/rsi-data/runs")
        );
        assert_eq!(
            config.resolved_cache_repo_dir(),
            PathBuf::from("/tmp/rsi-data/cache/repo")
        );
    }

    #[test]
    fn experience_dir_override_wins() {
        let mut config = valid_config();
        config.experience_dir = PathBuf::from("/tmp/custom-experience");
        assert_eq!(
            config.resolved_experience_dir(),
            PathBuf::from("/tmp/custom-experience")
        );
    }

    #[test]
    fn derived_paths_fall_back_to_auto_harness_dir() {
        let config = AutoHarnessConfig::default();
        assert_eq!(
            config.resolved_experience_dir(),
            PathBuf::from(".auto_harness/experience")
        );
        assert_eq!(
            config.resolved_worktrees_dir(),
            PathBuf::from(".auto_harness/worktrees")
        );
        assert_eq!(
            config.resolved_runs_dir(),
            PathBuf::from(".auto_harness/runs")
        );
        assert_eq!(
            config.resolved_cache_repo_dir(),
            PathBuf::from(".auto_harness/cache/repo")
        );
    }

    #[tokio::test]
    async fn load_reads_yaml_and_reports_io_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auto_harness.yaml");
        tokio::fs::write(
            &path,
            "data_dir: /tmp/rsi-loaded\npipeline_preference: extended\nconfig_bootstrapped: true\n",
        )
        .await
        .unwrap();

        let config = AutoHarnessConfig::load(&path).await.unwrap();
        assert_eq!(config.data_dir, PathBuf::from("/tmp/rsi-loaded"));
        assert_eq!(config.pipeline_preference, PipelinePreference::Extended);
        assert!(config.config_bootstrapped);

        let missing = dir.path().join("absent.yaml");
        let error = AutoHarnessConfig::load(&missing).await.unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Config);
    }

    #[tokio::test]
    async fn load_or_default_falls_back_to_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("absent.yaml");
        let config = AutoHarnessConfig::load_or_default(&missing).await;
        assert_eq!(config, AutoHarnessConfig::default());

        let invalid = dir.path().join("invalid.yaml");
        tokio::fs::write(&invalid, "data_dir: [unclosed")
            .await
            .unwrap();
        let config = AutoHarnessConfig::load_or_default(&invalid).await;
        assert_eq!(config, AutoHarnessConfig::default());
    }

    #[tokio::test]
    async fn load_or_default_preserves_loaded_bootstrap_flag() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bootstrapped.yaml");
        tokio::fs::write(&path, "config_bootstrapped: true\n")
            .await
            .unwrap();
        let config = AutoHarnessConfig::load_or_default(&path).await;
        assert!(config.config_bootstrapped);
    }

    #[test]
    fn pipeline_preference_serde_names() {
        let yaml = "pipeline_preference: extended\n";
        let config = AutoHarnessConfig::from_yaml_str(yaml).unwrap();
        assert_eq!(config.pipeline_preference, PipelinePreference::Extended);
        let rendered = serde_yaml_ng::to_string(&PipelinePreference::Meta).unwrap();
        assert_eq!(rendered.trim(), "meta");
    }
}
