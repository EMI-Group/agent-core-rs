//! The Docker environment lifecycle: build, start, exec, copy, stop.
//!
//! Python: `evaluator_pipeline/docker_env.py`. The Rust port keeps the lifecycle
//! and the command-line shape, and drops the parts that only make sense for the
//! Python agent adapters (locale probing, `tail -f /dev/null` keep-alives chosen
//! at runtime, encoding diagnostics): the container is started with
//! `sleep infinity`, the workspace is bind-mounted at the container workdir, and
//! every operation reports a typed error.
//!
//! All side effects go through a [`CommandRunner`], so the whole lifecycle is
//! exercised offline by [`FakeCommandRunner`](super::FakeCommandRunner). The
//! methods take no cancellation token: the pipeline checks cancellation between
//! steps, and a single Docker call is bounded by
//! [`timeout_ms`](DockerEnvironment::timeout_ms) instead.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use openjiuwen_common::error::{Error, Result};
use tokio_util::sync::CancellationToken;

use super::command::CommandRunner;
use super::models::{sanitize_task_id, CommandSpec, ExecResult};

/// The default per-command timeout: Python's `DockerEnvironment(timeout=900)`.
pub const DEFAULT_TIMEOUT_MS: u64 = 900_000;

/// The prefix of every container this module names.
pub const CONTAINER_NAME_PREFIX: &str = "openjiuwen-eval-";

/// The deterministic container name for `task_id`.
///
/// The name is derived from the sanitised task id (see
/// [`sanitize_task_id`]) so two runs of the same task reuse the same name and
/// a leftover container is easy to find.
///
/// # Example
///
/// ```
/// use openjiuwen_agent_evolving::evaluator_pipeline::container_name;
///
/// assert_eq!(container_name("Task 42"), "openjiuwen-eval-task-42");
/// ```
#[must_use]
pub fn container_name(task_id: &str) -> String {
    format!("{CONTAINER_NAME_PREFIX}{}", sanitize_task_id(task_id))
}

/// One Docker container: the image, the bind-mounted workspace, the environment,
/// and the container id once it is running.
///
/// The id is the only mutable state; every other field is configuration.
#[derive(Debug, Clone)]
pub struct DockerEnvironment {
    runner: Arc<dyn CommandRunner>,
    image: String,
    container_name: String,
    workspace: PathBuf,
    container_workdir: String,
    env: BTreeMap<String, String>,
    timeout_ms: u64,
    container_id: Option<String>,
}

impl DockerEnvironment {
    /// Describes a container for `image`, named `container_name`, with
    /// `workspace` bind-mounted at `container_workdir`.
    ///
    /// Nothing is created or verified here: call
    /// [`ensure_available`](Self::ensure_available) before the first command.
    #[must_use]
    pub fn new(
        runner: Arc<dyn CommandRunner>,
        image: impl Into<String>,
        container_name: impl Into<String>,
        workspace: impl Into<PathBuf>,
        container_workdir: impl Into<String>,
    ) -> Self {
        Self {
            runner,
            image: image.into(),
            container_name: container_name.into(),
            workspace: workspace.into(),
            container_workdir: container_workdir.into(),
            env: BTreeMap::new(),
            timeout_ms: DEFAULT_TIMEOUT_MS,
            container_id: None,
        }
    }

    /// Replaces the environment variables passed to every `docker run` and
    /// `docker exec`.
    #[must_use]
    pub fn with_env(mut self, env: BTreeMap<String, String>) -> Self {
        self.env = env;
        self
    }

    /// Sets the per-command timeout in milliseconds.
    #[must_use]
    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Sets the per-command timeout from a [`Duration`].
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout_ms = u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX);
        self
    }

    /// The image tag.
    #[must_use]
    pub fn image(&self) -> &str {
        &self.image
    }

    /// The container name.
    #[must_use]
    pub fn container_name(&self) -> &str {
        &self.container_name
    }

    /// The host workspace bind-mounted into the container.
    #[must_use]
    pub fn workspace(&self) -> &Path {
        &self.workspace
    }

    /// The directory inside the container the workspace is mounted at.
    #[must_use]
    pub fn container_workdir(&self) -> &str {
        &self.container_workdir
    }

    /// The per-command timeout in milliseconds.
    #[must_use]
    pub fn timeout_ms(&self) -> u64 {
        self.timeout_ms
    }

    /// The running container's id, or `None` before [`start`](Self::start) and
    /// after [`stop`](Self::stop).
    #[must_use]
    pub fn container_id(&self) -> Option<&str> {
        self.container_id.as_deref()
    }

    /// `true` when a container id is recorded (that is, `start` succeeded and
    /// `stop` has not run).
    #[must_use]
    pub fn has_container(&self) -> bool {
        self.container_id.is_some()
    }

    /// Verifies that the `docker` binary is on `PATH`.
    ///
    /// Returns [`Error::Unsupported`] naming the missing binary, so the pipeline
    /// reports an explicit deferred/absent-tool failure rather than a generic
    /// spawn error.
    pub async fn ensure_available(&self) -> Result<PathBuf> {
        match self.runner.which("docker").await? {
            Some(path) => Ok(path),
            None => Err(Error::unsupported(
                "docker is not available: install Docker and put the `docker` binary on PATH to run the evaluator pipeline",
            )),
        }
    }

    /// Builds `tag` from `dockerfile` with `context_dir` as the build context.
    ///
    /// The command is `docker build -t <tag> -f <dockerfile> <context>`. Returns
    /// the tag so callers can chain into [`start`](Self::start). A missing
    /// Dockerfile is [`Error::validation`]; a failing build is
    /// [`Error::sys_op`].
    pub async fn build(&self, context_dir: &Path, dockerfile: &Path, tag: &str) -> Result<String> {
        if !dockerfile.is_file() {
            return Err(Error::validation(format!(
                "Dockerfile not found: {}",
                dockerfile.display()
            )));
        }
        let spec = CommandSpec::new("docker")
            .arg("build")
            .arg("-t")
            .arg(tag)
            .arg("-f")
            .arg(dockerfile.display().to_string())
            .arg(context_dir.display().to_string())
            .with_timeout_ms(self.timeout_ms);
        let result = self.run(&spec).await?;
        require_success(&result, "docker build")?;
        Ok(tag.to_owned())
    }

    /// Starts the container, stopping any container this environment already
    /// tracks first.
    ///
    /// The command is
    /// `docker run -d --name <name> -w <workdir> -v <workspace>:<workdir>
    /// [-e KEY=VALUE ...] <image> sleep infinity`; the container id is read from
    /// standard output.
    pub async fn start(&mut self) -> Result<()> {
        if self.container_id.is_some() {
            self.stop().await?;
        }
        let mut spec = CommandSpec::new("docker")
            .arg("run")
            .arg("-d")
            .arg("--name")
            .arg(&self.container_name)
            .arg("-w")
            .arg(&self.container_workdir)
            .arg("-v")
            .arg(format!(
                "{}:{}",
                self.workspace.display(),
                self.container_workdir
            ));
        for (key, value) in &self.env {
            spec = spec.arg("-e").arg(format!("{key}={value}"));
        }
        let spec = spec
            .arg(&self.image)
            .arg("sleep")
            .arg("infinity")
            .with_timeout_ms(self.timeout_ms);
        let result = self.run(&spec).await?;
        require_success(&result, "docker run")?;
        let id = result.stdout.trim();
        if id.is_empty() {
            return Err(Error::validation(
                "docker run produced no container id on standard output",
            ));
        }
        self.container_id = Some(id.to_owned());
        Ok(())
    }

    /// Runs `command` inside the container.
    ///
    /// The command is
    /// `docker exec -w <workdir> [-e KEY=VALUE ...] <container> <argv...>`; the
    /// container's exit code, timeout, and both streams come back in the
    /// [`ExecResult`]. A non-zero exit is not an error — the score parsers
    /// interpret it.
    pub async fn exec(&self, command: &[String]) -> Result<ExecResult> {
        let id = self.require_container()?;
        let mut spec = CommandSpec::new("docker")
            .arg("exec")
            .arg("-w")
            .arg(&self.container_workdir);
        for (key, value) in &self.env {
            spec = spec.arg("-e").arg(format!("{key}={value}"));
        }
        let spec = spec
            .arg(id)
            .args(command.iter().cloned())
            .with_timeout_ms(self.timeout_ms);
        self.run(&spec).await
    }

    /// Copies `src` from the host to `dst` inside the container.
    ///
    /// The command is `docker cp <src> <container>:<dst>`.
    pub async fn copy_to(&self, src: &Path, dst: &str) -> Result<()> {
        let id = self.require_container()?;
        let spec = CommandSpec::new("docker")
            .arg("cp")
            .arg(src.display().to_string())
            .arg(format!("{id}:{dst}"))
            .with_timeout_ms(self.timeout_ms);
        let result = self.run(&spec).await?;
        require_success(&result, "docker cp")
    }

    /// Copies `src` inside the container to `dst` on the host.
    ///
    /// The command is `docker cp <container>:<src> <dst>`.
    pub async fn copy_from(&self, src: &str, dst: &Path) -> Result<()> {
        let id = self.require_container()?;
        let spec = CommandSpec::new("docker")
            .arg("cp")
            .arg(format!("{id}:{src}"))
            .arg(dst.display().to_string())
            .with_timeout_ms(self.timeout_ms);
        let result = self.run(&spec).await?;
        require_success(&result, "docker cp")
    }

    /// Queries the daemon for the container's running state.
    ///
    /// The command is `docker inspect -f {{.State.Running}} <container>`. Returns
    /// `false` without querying when no container is tracked.
    pub async fn is_running(&self) -> Result<bool> {
        let Some(id) = self.container_id.clone() else {
            return Ok(false);
        };
        let spec = CommandSpec::new("docker")
            .arg("inspect")
            .arg("-f")
            .arg("{{.State.Running}}")
            .arg(id)
            .with_timeout_ms(self.timeout_ms);
        let result = self.run(&spec).await?;
        require_success(&result, "docker inspect")?;
        Ok(result.stdout.trim().eq_ignore_ascii_case("true"))
    }

    /// Force-removes the container (`docker rm -f`) and forgets its id.
    ///
    /// A container that is already gone is not an error: Docker reports a
    /// non-zero exit and that is ignored (only a runner-level failure, such as a
    /// vanished `docker` binary, propagates). The id is cleared either way, so a
    /// second `stop` is a no-op.
    pub async fn stop(&mut self) -> Result<()> {
        let Some(id) = self.container_id.take() else {
            return Ok(());
        };
        let spec = CommandSpec::new("docker")
            .arg("rm")
            .arg("-f")
            .arg(&id)
            .with_timeout_ms(self.timeout_ms);
        let result = self.run(&spec).await?;
        if !result.success() {
            tracing::debug!(
                container = %id,
                returncode = result.returncode,
                stderr = result.stderr.trim(),
                "docker rm reported a failure; the container was probably already gone"
            );
        }
        Ok(())
    }

    /// The container id, or [`Error::validation`] when no container is running.
    fn require_container(&self) -> Result<&str> {
        self.container_id.as_deref().ok_or_else(|| {
            Error::validation(format!(
                "container `{}` is not running; call `start` first",
                self.container_name
            ))
        })
    }

    /// Runs one command with a private cancellation token.
    async fn run(&self, spec: &CommandSpec) -> Result<ExecResult> {
        self.runner.run(spec, &CancellationToken::new()).await
    }
}

/// Fails unless `result` is a successful, un-timed-out command.
fn require_success(result: &ExecResult, what: &str) -> Result<()> {
    if result.timed_out {
        return Err(Error::timeout(format!("{what} timed out")));
    }
    if !result.success() {
        return Err(Error::sys_op(format!(
            "{what} failed (rc={}): {}",
            result.returncode,
            result.stderr.trim()
        )));
    }
    Ok(())
}
