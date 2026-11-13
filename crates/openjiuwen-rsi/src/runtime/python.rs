//! The Python toolchain a harness run needs: a virtualenv, `pip`, and `pytest`.
//!
//! Port of the Python-toolchain half of
//! `harness_rsi/auto_harness/stages/verify.py` (v0.1.20): `_check_pip_available`,
//! `_bootstrap_pip`, `_run_pip_install`, and `_run_pytest_file`. Python drives
//! all of it through `asyncio.create_subprocess_exec(sys.executable, …)`; this
//! module drives the same argv shapes through the [`CommandRunner`] seam, so the
//! whole flow is testable without a real interpreter.
//!
//! Everything here is offline-scriptable: [`PythonToolchain::ensure_venv`] and
//! the pip/pytest helpers record their exact command line in a
//! [`FakeCommandRunner`](super::command::FakeCommandRunner), and
//! [`discover_python`] is the only host-dependent entry point.
//!
//! Test-report files are the one side effect with a temporary directory: a
//! pytest run writes its JSON or `JUnit` report under
//! `std::env::temp_dir()/openjiuwen-rsi-pytest-<uuid>/`, which is removed again
//! once parsed.
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tracing::warn;

use openjiuwen_common::error::{Error, Result};

use super::command::{CommandOutcome, CommandRunner, CommandSpec};
use super::pytest::{parse_report, PytestReport};

/// Maximum number of characters kept from a failing `pip install`.
const INSTALL_ERROR_LIMIT: usize = 500;

/// The report format a pytest run writes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReportFormat {
    /// `pytest-json-report`'s JSON document (`--json-report`).
    #[default]
    Json,
    /// `JUnit` XML (`--junitxml`), available without a plugin.
    JUnit,
}

/// Where a virtualenv ended up and whether this call created it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolchainSetup {
    /// Interpreter used to create the virtualenv (`python -m venv`).
    pub python: String,
    /// Virtualenv root directory.
    pub venv_dir: PathBuf,
    /// The interpreter inside the virtualenv.
    pub venv_python: PathBuf,
    /// `true` when this call created the virtualenv, `false` when it existed.
    pub created: bool,
}

/// Outcome of one `pip install`.
///
/// A failed install is data, matching Python's `(success, error)` tuple: the
/// message is prefixed `pip_install_failed: ` and truncated to 500 characters.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct PipInstallOutcome {
    /// Whether pip exited successfully.
    pub success: bool,
    /// Combined output, or the `pip_install_failed: …` error text.
    pub output: String,
}

/// Outcome of one pytest run.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PytestOutcome {
    /// Whether pytest exited with code `0`.
    pub passed: bool,
    /// The parsed report, when one was written and could be read.
    pub report: Option<PytestReport>,
    /// Report format that was requested.
    pub format: Option<ReportFormat>,
    /// Combined output of the run.
    pub output: String,
    /// Exit code, or `None` when the run was killed.
    pub exit_code: Option<i32>,
}

/// A Python interpreter bound to one workspace, with an optional virtualenv.
///
/// `new` discovers an interpreter on the host; tests and callers that know their
/// interpreter should use [`PythonToolchain::with_python`] so no discovery
/// happens.
#[derive(Clone)]
pub struct PythonToolchain {
    runner: Arc<dyn CommandRunner>,
    workspace: PathBuf,
    python: String,
    timeout: Option<Duration>,
    venv_dir: Option<PathBuf>,
    report_format: ReportFormat,
}

impl PythonToolchain {
    /// Creates a toolchain for `workspace`.
    ///
    /// The interpreter is discovered with [`discover_python`], falling back to
    /// `python3` (Unix) or `python` (Windows) when discovery finds nothing.
    #[must_use]
    pub fn new(runner: Arc<dyn CommandRunner>, workspace: impl Into<PathBuf>) -> Self {
        Self {
            runner,
            workspace: workspace.into(),
            python: discover_python().unwrap_or_else(|| default_python().to_owned()),
            timeout: None,
            venv_dir: None,
            report_format: ReportFormat::Json,
        }
    }

    /// Sets the interpreter used for every command.
    #[must_use]
    pub fn with_python(mut self, python: impl Into<String>) -> Self {
        self.python = python.into();
        self
    }

    /// Sets the virtualenv directory (defaults to `<workspace>/.venv`).
    #[must_use]
    pub fn with_venv(mut self, dir: impl Into<PathBuf>) -> Self {
        self.venv_dir = Some(dir.into());
        self
    }

    /// Sets the per-command deadline.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Sets the pytest report format (JSON by default).
    #[must_use]
    pub fn with_report_format(mut self, format: ReportFormat) -> Self {
        self.report_format = format;
        self
    }

    /// Returns the configured interpreter.
    #[must_use]
    pub fn python(&self) -> &str {
        &self.python
    }

    /// Returns the workspace directory.
    #[must_use]
    pub fn workspace(&self) -> &Path {
        &self.workspace
    }

    /// Returns the virtualenv root (`<workspace>/.venv` unless configured).
    #[must_use]
    pub fn venv_dir(&self) -> PathBuf {
        self.venv_dir
            .clone()
            .unwrap_or_else(|| self.workspace.join(".venv"))
    }

    /// Returns the interpreter inside the virtualenv.
    ///
    /// `<venv>/bin/python` on Unix and `<venv>/Scripts/python.exe` on Windows.
    #[must_use]
    pub fn venv_python(&self) -> PathBuf {
        if cfg!(windows) {
            self.venv_dir().join("Scripts").join("python.exe")
        } else {
            self.venv_dir().join("bin").join("python")
        }
    }

    /// Returns the configured report format.
    #[must_use]
    pub fn report_format(&self) -> ReportFormat {
        self.report_format
    }

    /// Creates the virtualenv unless it already exists.
    ///
    /// Idempotent by `pyvenv.cfg`: an existing marker means the environment was
    /// created before (by this crate or by hand) and is returned with
    /// `created: false`, so a resumed session never rebuilds it. Creating a
    /// virtualenv also records the directory on the toolchain.
    ///
    /// # Errors
    ///
    /// Returns [`Error::tool`] when `python -m venv` fails, and otherwise
    /// propagates [`CommandRunner::run`] errors (a missing interpreter is
    /// [`Error::config`]).
    pub async fn ensure_venv(&mut self) -> Result<ToolchainSetup> {
        let venv_dir = self.venv_dir();
        let venv_python = self.venv_python();
        if tokio::fs::metadata(venv_dir.join("pyvenv.cfg"))
            .await
            .is_ok()
        {
            return Ok(ToolchainSetup {
                python: self.python.clone(),
                venv_dir,
                venv_python,
                created: false,
            });
        }
        let venv_arg = venv_dir.to_string_lossy().into_owned();
        let spec = self.command(&["-m", "venv", &venv_arg]);
        let outcome = self.runner.run(spec).await?;
        outcome.require_success("python -m venv")?;
        self.venv_dir = Some(venv_dir.clone());
        Ok(ToolchainSetup {
            python: self.python.clone(),
            venv_dir,
            venv_python,
            created: true,
        })
    }

    /// Returns the installed pip version (`python -m pip --version`).
    ///
    /// # Errors
    ///
    /// Returns [`Error::tool`] when pip is missing or broken, and propagates
    /// [`CommandRunner::run`] errors.
    pub async fn pip_version(&self) -> Result<String> {
        let outcome = self.run_python(&["-m", "pip", "--version"]).await?;
        outcome.require_success("python -m pip --version")?;
        Ok(outcome.combined_output().trim().to_owned())
    }

    /// Makes pip available, bootstrapping it with `ensurepip` when missing.
    ///
    /// Python's `_check_pip_available` followed by `_bootstrap_pip`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::tool`] when both the pip probe and
    /// `python -m ensurepip --upgrade` fail, and propagates
    /// [`CommandRunner::run`] errors.
    pub async fn ensure_pip(&self) -> Result<()> {
        let probe = self.run_python(&["-m", "pip", "--version"]).await?;
        if probe.success() {
            return Ok(());
        }
        let bootstrap = self.run_python(&["-m", "ensurepip", "--upgrade"]).await?;
        bootstrap.require_success("python -m ensurepip --upgrade")
    }

    /// Runs `python -m pip install <requirements>`.
    ///
    /// A failing install is returned as [`PipInstallOutcome::success`] `false`
    /// with the `pip_install_failed: ` prefix Python uses, not as an error.
    ///
    /// # Errors
    ///
    /// Propagates [`CommandRunner::run`] errors only (missing interpreter,
    /// spawn failure, cancellation).
    pub async fn pip_install(&self, requirements: &[String]) -> Result<PipInstallOutcome> {
        let mut args = vec!["-m".to_owned(), "pip".to_owned(), "install".to_owned()];
        args.extend(requirements.iter().cloned());
        let refs = args.iter().map(String::as_str).collect::<Vec<_>>();
        let outcome = self.run_python(&refs).await?;
        if outcome.success() {
            return Ok(PipInstallOutcome {
                success: true,
                output: outcome.combined_output().trim().to_owned(),
            });
        }
        Ok(PipInstallOutcome {
            success: false,
            output: format!(
                "pip_install_failed: {}",
                truncate(outcome.combined_output().trim(), INSTALL_ERROR_LIMIT)
            ),
        })
    }

    /// Runs `python -m pip install -r <path>`.
    ///
    /// # Errors
    ///
    /// Propagates [`CommandRunner::run`] errors only, like
    /// [`PythonToolchain::pip_install`].
    pub async fn pip_install_requirements_file(&self, path: &Path) -> Result<PipInstallOutcome> {
        let requirements = vec!["-r".to_owned(), path.to_string_lossy().into_owned()];
        self.pip_install(&requirements).await
    }

    /// Runs pytest over `target` and parses the report it writes.
    ///
    /// The argv is Python's `verify.py` shape — `<python> -m pytest <target> -q
    /// -o addopts=` — plus the report flags from [`pytest_args`] and any
    /// `extra_args`. `CI=1` and `VIRTUAL_ENV` (when a virtualenv is configured)
    /// are set, matching `_run_pytest_file`.
    ///
    /// The report file is written under a fresh temporary directory and removed
    /// after parsing. A missing or unreadable report is not an error: the raw
    /// output is still returned, which is what Python reports as the gate
    /// output.
    ///
    /// # Errors
    ///
    /// Returns [`Error::tool`] when the temporary report directory cannot be
    /// created, and propagates [`CommandRunner::run`] errors.
    pub async fn run_pytest(&self, target: &Path, extra_args: &[String]) -> Result<PytestOutcome> {
        let report_dir =
            std::env::temp_dir().join(format!("openjiuwen-rsi-pytest-{}", uuid::Uuid::new_v4()));
        tokio::fs::create_dir_all(&report_dir)
            .await
            .map_err(|error| {
                Error::tool(format!(
                    "could not create pytest report directory {}: {error}",
                    report_dir.display()
                ))
            })?;
        let report_path = report_dir.join(match self.report_format {
            ReportFormat::Json => "report.json",
            ReportFormat::JUnit => "report.xml",
        });
        let mut args = vec![
            "-m".to_owned(),
            "pytest".to_owned(),
            target.to_string_lossy().into_owned(),
            "-q".to_owned(),
            "-o".to_owned(),
            "addopts=".to_owned(),
        ];
        args.extend(pytest_args(self.report_format, &report_path));
        args.extend(extra_args.iter().cloned());
        let refs = args.iter().map(String::as_str).collect::<Vec<_>>();
        let mut spec = self.command(&refs).env("CI", "1");
        if let Some(venv) = &self.venv_dir {
            spec = spec.env("VIRTUAL_ENV", venv.to_string_lossy().into_owned());
        }
        let outcome = self.runner.run(spec).await;
        let report = read_report(&report_path, self.report_format).await;
        if let Err(error) = tokio::fs::remove_dir_all(&report_dir).await {
            warn!(
                "could not remove pytest report directory {}: {error}",
                report_dir.display()
            );
        }
        let outcome = outcome?;
        Ok(PytestOutcome {
            passed: outcome.code == Some(0),
            report,
            format: Some(self.report_format),
            output: outcome.combined_output().to_owned(),
            exit_code: outcome.code,
        })
    }

    /// Builds a command spec for the configured interpreter.
    fn command(&self, args: &[&str]) -> CommandSpec {
        let mut spec = CommandSpec::new(&self.python)
            .args(args.iter().copied())
            .cwd(self.workspace.clone())
            .merge_stderr();
        if let Some(timeout) = self.timeout {
            spec = spec.timeout(timeout);
        }
        spec
    }

    /// Runs the configured interpreter with `args`.
    async fn run_python(&self, args: &[&str]) -> Result<CommandOutcome> {
        self.runner.run(self.command(args)).await
    }
}

/// The report flag list for `format`, writing to `report_path`.
///
/// `Json` selects `pytest-json-report`; `JUnit` uses pytest's built-in
/// `--junitxml`, which needs no plugin.
#[must_use]
pub fn pytest_args(format: ReportFormat, report_path: &Path) -> Vec<String> {
    let path = report_path.to_string_lossy().into_owned();
    match format {
        ReportFormat::Json => vec![
            "--json-report".to_owned(),
            format!("--json-report-file={path}"),
        ],
        ReportFormat::JUnit => vec![format!("--junitxml={path}")],
    }
}

/// Reads and parses a report file, logging and returning `None` when unreadable.
async fn read_report(path: &Path, format: ReportFormat) -> Option<PytestReport> {
    let text = match tokio::fs::read_to_string(path).await {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return None,
        Err(error) => {
            warn!("could not read pytest report {}: {error}", path.display());
            return None;
        }
    };
    match parse_report(&text, format) {
        Ok(report) => Some(report),
        Err(error) => {
            warn!("could not parse pytest report {}: {error}", path.display());
            None
        }
    }
}

/// Discovers a Python interpreter on the host (`python3`, then `python`).
///
/// Returns the resolved path when `which` finds one and `None` otherwise; it
/// never panics or falls back silently to a name that does not exist.
#[must_use]
pub fn discover_python() -> Option<String> {
    ["python3", "python"].iter().find_map(|name| {
        which::which(name)
            .ok()
            .map(|path| path.to_string_lossy().into_owned())
    })
}

/// The platform's conventional interpreter name.
fn default_python() -> &'static str {
    if cfg!(windows) {
        "python"
    } else {
        "python3"
    }
}

/// Truncates `text` to at most `limit` characters.
fn truncate(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_owned();
    }
    text.chars().take(limit).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_counts_characters() {
        assert_eq!(truncate("abcd", 10), "abcd");
        assert_eq!(truncate("abcd", 2), "ab");
        assert_eq!(truncate("😀😀😀", 2), "😀😀");
    }

    #[test]
    fn pytest_args_use_the_requested_report_file() {
        let path = Path::new("/tmp/report.json");
        assert_eq!(
            pytest_args(ReportFormat::Json, path),
            vec![
                "--json-report".to_owned(),
                "--json-report-file=/tmp/report.json".to_owned()
            ]
        );
        assert_eq!(
            pytest_args(ReportFormat::JUnit, Path::new("/tmp/report.xml")),
            vec!["--junitxml=/tmp/report.xml".to_owned()]
        );
    }
}
