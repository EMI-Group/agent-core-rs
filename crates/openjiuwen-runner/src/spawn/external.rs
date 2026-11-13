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
//! [`crate::spawn::SubprocessSpawner`]'s, not reimplemented here. That includes
//! the shutdown rule: a [`SpawnedAgent::shutdown`] with a zero timeout uses the
//! configured `shutdown_timeout_secs`, a non-zero one wins over it. This backend is
//! therefore the subprocess protocol plus `PATH` resolution, and it labels its
//! children [`crate::spawn::SpawnBackend::ExternalCommand`] through
//! [`crate::spawn::SubprocessSpawner::with_backend`], so the handle, this spawner,
//! and a runner's monitor detail all agree on the backend that produced the child.
//!
//! Search path: [`ExternalCommandSpawner::with_search_path`] replaces the
//! inherited `PATH` entirely (the hermetic form, for tests and sandboxes); unset,
//! the process `PATH` is used, split with the platform's own separator rules.
use crate::spawn::subprocess::{describe_settings, SubprocessSpawner};
use crate::spawn::{AgentSpawner, SpawnBackend, SpawnSpec, SpawnedAgent, SubprocessConfig};
use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use serde_json::{json, Value};
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
    ///
    /// Children are supervised by the subprocess backend, so this is the
    /// deadline a caller gets by passing [`Duration::ZERO`] to
    /// [`SpawnedAgent::shutdown`]; a non-zero caller timeout wins over it.
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

    /// The command, the search path, and the subprocess summary it inherits
    /// (`launch`) — the environment is reported by name only, never by value.
    fn describe(&self) -> Value {
        json!({
            "backend": SpawnBackend::ExternalCommand.as_str(),
            "command": self.command,
            "search_path": self
                .search_path
                .as_ref()
                .map(|path| path.to_string_lossy().into_owned()),
            "launch": Value::Object(describe_settings(&self.config)),
        })
    }

    /// Resolves the command, then spawns it through the subprocess backend
    /// labelled [`SpawnBackend::ExternalCommand`], so the child reports the
    /// backend that resolved it rather than the bare protocol underneath.
    async fn spawn(&self, spec: SpawnSpec) -> Result<Arc<dyn SpawnedAgent>> {
        let program = self.resolve()?;
        let config = SubprocessConfig {
            program: program.to_string_lossy().into_owned(),
            ..self.config.clone()
        };
        SubprocessSpawner::with_backend(config, SpawnBackend::ExternalCommand)
            .spawn(spec)
            .await
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

    /// The libtest arguments that re-execute this binary as the hermetic
    /// protocol child of `subprocess_tests.rs`, so the delegation is exercised
    /// against a real child rather than a stand-in.
    fn child_args() -> Vec<String> {
        [
            "spawn::subprocess::tests::fake_child",
            "--exact",
            "--ignored",
            "--test-threads=1",
            "--nocapture",
        ]
        .map(str::to_owned)
        .to_vec()
    }

    #[tokio::test]
    async fn a_spawned_command_is_labelled_external_command() {
        const BOUND: Duration = Duration::from_secs(10);
        let program = std::env::current_exe()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let spawner = ExternalCommandSpawner::new(program)
            .with_args(child_args())
            .with_request_timeout(BOUND)
            .with_health_check(Duration::ZERO, Duration::from_secs(2), 3);
        assert_eq!(spawner.backend(), SpawnBackend::ExternalCommand);

        let child: Arc<dyn SpawnedAgent> =
            tokio::time::timeout(BOUND, spawner.spawn(SpawnSpec::default()))
                .await
                .expect("spawn must not hang")
                .expect("the resolved program launches through the seam");
        assert_eq!(
            child.backend(),
            SpawnBackend::ExternalCommand,
            "the handle reports the backend that resolved it"
        );
        assert_eq!(
            child.health_snapshot().backend,
            SpawnBackend::ExternalCommand
        );
        assert!(
            tokio::time::timeout(BOUND, child.shutdown(Duration::from_secs(5)))
                .await
                .expect("shutdown must not hang")
                .expect("the child acknowledges the shutdown")
        );
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

#[cfg(test)]
mod describe_tests {
    use super::*;

    #[test]
    fn describe_reports_the_command_search_path_and_launch_settings() {
        let spawner = ExternalCommandSpawner::new("vendor-cli")
            .with_search_path("/opt/vendor/bin")
            .with_args(vec!["--protocol".to_owned(), "ndjson".to_owned()])
            .with_env("VENDOR_TOKEN", "super-secret-value")
            .with_request_timeout(Duration::from_secs(12))
            .with_health_check(Duration::from_secs(2), Duration::from_millis(500), 7);
        let described = spawner.describe();
        assert_eq!(described["backend"], json!("external_command"));
        assert_eq!(described["command"], json!("vendor-cli"));
        assert_eq!(described["search_path"], json!("/opt/vendor/bin"));
        // The program is resolved at spawn time, so the summary reports it empty.
        assert_eq!(described["launch"]["program"], json!(""));
        assert_eq!(described["launch"]["args"], json!(["--protocol", "ndjson"]));
        assert_eq!(described["launch"]["env_names"], json!(["VENDOR_TOKEN"]));
        assert_eq!(described["launch"]["request_timeout_secs"], json!(12.0));
        assert_eq!(described["launch"]["health_check_timeout_secs"], json!(0.5));
        assert_eq!(described["launch"]["max_health_failures"], json!(7));
        let rendered = described.to_string();
        assert!(
            !rendered.contains("super-secret-value"),
            "no environment value leaks: {rendered}"
        );

        // An unset search path is JSON null, never an empty string.
        let plain = ExternalCommandSpawner::default().describe();
        assert_eq!(plain["search_path"], Value::Null);
        assert_eq!(plain["launch"]["cwd"], Value::Null);
    }
}
