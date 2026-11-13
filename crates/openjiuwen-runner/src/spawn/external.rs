//! The external-command backend of the spawn seam: a host-configured program
//! that speaks the NDJSON protocol, resolved on `PATH`.
//!
//! A vendor CLI (or any child that already implements the wire) is not a program
//! this crate should hard-code, so [`ExternalCommandSpawner`] names one and
//! resolves it: a bare name is looked up on a search path, a path is used as
//! given, and the result must be an existing, executable regular file.
//!
//! Resolution then delegates to the subprocess backend — the process
//! handling, framing, timeouts, health task and kill-on-drop are
//! [`crate::spawn::SubprocessSpawner`]'s, not reimplemented here. The only
//! difference a handle sees is [`crate::spawn::SpawnBackend::ExternalCommand`],
//! which the spawner reports so a registry and a host can tell the two apart.
//!
//! Search path: [`ExternalCommandSpawner::with_search_path`] replaces the
//! inherited `PATH` entirely (the hermetic form, for tests and sandboxes); unset,
//! the process `PATH` is used, split with the platform's own separator rules.
use crate::spawn::subprocess::SubprocessSpawner;
use crate::spawn::{AgentSpawner, SpawnBackend, SpawnSpec, SpawnedAgent, SubprocessConfig};
use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

/// A spawner for a named external program.
///
/// Holds configuration only: every spawn resolves the command again and starts an
/// independent child.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExternalCommandSpawner {
    /// Program to resolve: a bare name (looked up on the search path) or a path.
    command: String,
    /// Launch configuration the resolved program is started with.
    ///
    /// Its [`SubprocessConfig::program`] is filled in from
    /// [`ExternalCommandSpawner::resolve`] at spawn time; a program set here is
    /// ignored.
    config: SubprocessConfig,
    /// Directory list searched instead of `PATH`, when set.
    search_path: Option<PathBuf>,
}

impl ExternalCommandSpawner {
    /// A spawner for `command`.
    #[must_use]
    pub fn new(command: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            ..Self::default()
        }
    }

    /// The configured command, as written.
    #[must_use]
    pub fn command(&self) -> &str {
        &self.command
    }

    /// The launch configuration (without the resolved program).
    #[must_use]
    pub fn config(&self) -> &SubprocessConfig {
        &self.config
    }

    /// Replaces the argument list.
    #[must_use]
    pub fn with_args(mut self, args: Vec<String>) -> Self {
        self.config.args = args;
        self
    }

    /// Appends one argument.
    #[must_use]
    pub fn with_arg(mut self, arg: impl Into<String>) -> Self {
        self.config.args.push(arg.into());
        self
    }

    /// Adds one environment override for the child.
    #[must_use]
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.config.env.insert(key.into(), value.into());
        self
    }

    /// Sets the child's working directory.
    #[must_use]
    pub fn with_cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.config.cwd = Some(cwd.into());
        self
    }

    /// Searches `search_path` instead of `PATH`.
    ///
    /// The single directory is used as given (no `PATH` splitting), which is what
    /// a hermetic test or a sandboxed host wants.
    #[must_use]
    pub fn with_search_path(mut self, search_path: impl Into<PathBuf>) -> Self {
        self.search_path = Some(search_path.into());
        self
    }

    /// Sets the deadline for one request/reply exchange.
    #[must_use]
    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.config.request_timeout_secs = timeout.as_secs_f64();
        self
    }

    /// Sets the deadline for a graceful shutdown before a kill.
    #[must_use]
    pub fn with_shutdown_timeout(mut self, timeout: Duration) -> Self {
        self.config.shutdown_timeout_secs = timeout.as_secs_f64();
        self
    }

    /// Sets the child's background health-check policy.
    #[must_use]
    pub fn with_health_check(
        mut self,
        interval: Duration,
        timeout: Duration,
        max_failures: u32,
    ) -> Self {
        self.config.health_check_interval_secs = interval.as_secs_f64();
        self.config.health_check_timeout_secs = timeout.as_secs_f64();
        self.config.max_health_failures = max_failures;
        self
    }

    /// Sets the cap on one frame line read from the child.
    #[must_use]
    pub fn with_max_frame_bytes(mut self, max_frame_bytes: usize) -> Self {
        self.config.max_frame_bytes = max_frame_bytes;
        self
    }

    /// Resolves the configured command to a path this process may execute.
    ///
    /// A command containing a path separator (or an absolute path) is checked as
    /// written; a bare name is looked up on the configured search path, or on `PATH`
    /// when unset.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Config`] when the command is empty, and
    /// [`openjiuwen_common::error::Error::NotFound`] when nothing usable matches
    /// — a missing file, a directory, or a file without an executable bit.
    pub fn resolve(&self) -> Result<PathBuf> {
        let command = self.command.trim();
        if command.is_empty() {
            return Err(Error::config(
                "ExternalCommandSpawner::command cannot be empty or whitespace only",
            ));
        }
        if looks_like_a_path(command) {
            let path = PathBuf::from(command);
            check_executable(&path).map_err(|reason| {
                Error::not_found(format!("external command '{command}' {reason}"))
            })?;
            return Ok(path);
        }
        let (directories, searched) = self.search_directories();
        for directory in &directories {
            let candidate = directory.join(command);
            if check_executable(&candidate).is_ok() {
                return Ok(candidate);
            }
        }
        let scope = if searched {
            format!("{} directories were searched", directories.len())
        } else {
            "no search directory is configured".to_owned()
        };
        Err(Error::not_found(format!(
            "external command '{command}' is not on the search path: {scope}"
        )))
    }

    /// The directories a bare command is looked up in, and whether there were
    /// any (an unset or empty `PATH` searches nothing).
    fn search_directories(&self) -> (Vec<PathBuf>, bool) {
        if let Some(search_path) = &self.search_path {
            return (vec![search_path.clone()], true);
        }
        let from_env = std::env::var_os("PATH")
            .map(|path| std::env::split_paths(&path).collect::<Vec<_>>())
            .unwrap_or_default();
        let searched = !from_env.is_empty();
        (from_env, searched)
    }
}

#[async_trait]
impl AgentSpawner for ExternalCommandSpawner {
    fn backend(&self) -> SpawnBackend {
        SpawnBackend::ExternalCommand
    }

    async fn spawn(&self, spec: SpawnSpec) -> Result<Arc<dyn SpawnedAgent>> {
        let program = self.resolve()?;
        let config = SubprocessConfig {
            program: program.to_string_lossy().into_owned(),
            ..self.config.clone()
        };
        SubprocessSpawner::new(config).spawn(spec).await
    }
}

/// Whether `command` names a path rather than a bare program name.
fn looks_like_a_path(command: &str) -> bool {
    Path::new(command).is_absolute()
        || command.contains('/')
        || command.contains(std::path::MAIN_SEPARATOR)
}

/// Checks one candidate: an existing, regular, executable file.
fn check_executable(path: &Path) -> std::result::Result<(), String> {
    let metadata = std::fs::metadata(path).map_err(|error| format!("cannot be used: {error}"))?;
    if !metadata.is_file() {
        return Err("is not a regular file".to_owned());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o111 == 0 {
            return Err("is not executable".to_owned());
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    /// A program in `dir`: `name`, executable on unix, plus a non-executable
    /// sibling.
    fn program(dir: &Path, name: &str, executable: bool) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, "#!/bin/sh\nexit 0\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = if executable { 0o755 } else { 0o644 };
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
        }
        #[cfg(not(unix))]
        let _ = executable;
        path
    }

    #[test]
    fn a_bare_name_resolves_on_the_search_path() {
        let dir = tempfile::tempdir().unwrap();
        let expected = program(dir.path(), "vendor-cli", true);
        let spawner = ExternalCommandSpawner::new("vendor-cli")
            .with_search_path(dir.path())
            .with_args(vec!["--protocol".to_owned()])
            .with_arg("ndjson")
            .with_env("VENDOR_MODE", "agent");
        assert_eq!(spawner.command(), "vendor-cli");
        assert_eq!(spawner.config().args, vec!["--protocol", "ndjson"]);
        assert_eq!(
            spawner.config().env.get("VENDOR_MODE").map(String::as_str),
            Some("agent")
        );
        assert_eq!(spawner.resolve().unwrap(), expected);
        assert_eq!(spawner.backend(), SpawnBackend::ExternalCommand);
    }

    #[test]
    fn a_missing_name_reports_the_command_and_the_search_scope() {
        let dir = tempfile::tempdir().unwrap();
        let error = ExternalCommandSpawner::new("not-installed")
            .with_search_path(dir.path())
            .resolve()
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert!(error.to_string().contains("not-installed"), "{error}");
        assert!(error.to_string().contains("1 directories"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn a_non_executable_file_is_not_a_program() {
        let dir = tempfile::tempdir().unwrap();
        program(dir.path(), "not-executable", false);
        let error = ExternalCommandSpawner::new("not-executable")
            .with_search_path(dir.path())
            .resolve()
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert!(error.to_string().contains("not-executable"), "{error}");

        // The same file named by path is rejected for the same reason.
        let by_path = ExternalCommandSpawner::new(
            dir.path()
                .join("not-executable")
                .to_string_lossy()
                .into_owned(),
        )
        .resolve()
        .unwrap_err();
        assert_eq!(by_path.kind(), ErrorKind::NotFound);
        assert!(by_path.to_string().contains("not executable"), "{by_path}");
    }

    #[test]
    fn an_absolute_program_resolves_as_written() {
        let current = std::env::current_exe().unwrap();
        let spawner = ExternalCommandSpawner::new(current.to_string_lossy().into_owned());
        assert_eq!(spawner.resolve().unwrap(), current);
    }

    #[test]
    fn an_empty_command_is_a_config_error() {
        for command in ["", "   "] {
            let error = ExternalCommandSpawner::new(command).resolve().unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Config);
            assert!(
                error
                    .to_string()
                    .contains("ExternalCommandSpawner::command"),
                "{error}"
            );
        }
    }

    #[test]
    fn a_directory_is_not_a_program() {
        let error =
            ExternalCommandSpawner::new(std::env::temp_dir().to_string_lossy().into_owned())
                .resolve()
                .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert!(error.to_string().contains("not a regular file"), "{error}");
    }

    #[tokio::test]
    async fn spawning_a_missing_command_launches_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let spawner = ExternalCommandSpawner::new("not-installed").with_search_path(dir.path());
        let Err(error) = spawner.spawn(SpawnSpec::default()).await else {
            panic!("a missing command cannot be launched");
        };
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert!(error.to_string().contains("not-installed"), "{error}");
    }

    #[test]
    fn builders_chain_into_the_subprocess_configuration() {
        let spawner = ExternalCommandSpawner::new("agent")
            .with_args(vec!["serve".to_owned()])
            .with_cwd("/tmp/agent")
            .with_request_timeout(Duration::from_secs(12))
            .with_shutdown_timeout(Duration::from_secs(4))
            .with_health_check(Duration::from_secs(2), Duration::from_millis(500), 7)
            .with_max_frame_bytes(4096);
        let config = spawner.config();
        assert_eq!(config.program, "", "the program is filled in at spawn time");
        assert_eq!(config.cwd.as_deref(), Some(Path::new("/tmp/agent")));
        assert!((config.request_timeout_secs - 12.0).abs() < f64::EPSILON);
        assert!((config.shutdown_timeout_secs - 4.0).abs() < f64::EPSILON);
        assert!((config.health_check_interval_secs - 2.0).abs() < f64::EPSILON);
        assert!((config.health_check_timeout_secs - 0.5).abs() < f64::EPSILON);
        assert_eq!(config.max_health_failures, 7);
        assert_eq!(config.max_frame_bytes, 4096);
        assert_eq!(
            ExternalCommandSpawner::default(),
            ExternalCommandSpawner::new("")
        );
    }
}
