//! The pipeline driver: workspace preparation, image build, container run, test
//! execution, scoring, and cleanup.
//!
//! Python: `evaluator_pipeline/pipeline.py` (`EvolutionPipeline`). The Rust port
//! keeps the container lifecycle and the score parsing, and drops the parts that
//! need a live agent or a GPU: agent adapters (`base.py`), the LLM evolution loop
//! (`_run_evolution`), skill management (`skill_manager.py`), and the
//! mirror-rewriting image-build tricks (`_build_base_image`), which cannot be
//! exercised offline. The caller supplies the environment recipe, the test
//! command, and the score policy.
//!
//! One task runs through these steps, each separated by a cancellation check:
//!
//! 1. validate the task and the config;
//! 2. verify `docker` is on `PATH` ([`Error::Unsupported`] when it is not);
//! 3. build the image when [`PipelineConfig::dockerfile`] is set;
//! 4. prepare the workspace with [`WorkspacePreparer`];
//! 5. start the container and copy the workspace into it;
//! 6. execute the test command (up to
//!    [`PipelineConfig::effective_max_attempts`] times while it keeps failing);
//! 7. parse the score with the task's or the config's [`ScoreSpec`];
//! 8. stop the container — always, also when a step failed — unless
//!    [`PipelineConfig::keep_container`] is set.
//!
//! There are no clocks and no randomness: the workspace directory and the
//! container name are derived from the task id, and a report contains only the
//! executed command's result and the attempt count.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use super::command::{check_cancel, CommandRunner};
use super::docker::{container_name, DockerEnvironment, DEFAULT_TIMEOUT_MS};
use super::git::WorkspacePreparer;
use super::models::{sanitize_task_id, EvaluationTask, ExecResult, GitWorkspaceSpec, TaskReport};
use super::scoring::{parse_score, ScoreSpec, ScoredRun};

/// The fewest times a test command is executed.
pub const MIN_ATTEMPTS: usize = 1;

/// The most times a test command is executed, however the config is written.
pub const MAX_ATTEMPTS: usize = 5;

/// Everything the pipeline needs that is not part of the task.
///
/// Python: `evaluator_pipeline/config.py` (`PipelineConfig`) reduced to the
/// fields the Rust pipeline uses; agent/benchmark adapter configuration and the
/// YAML/`${ENV}` loading are out of scope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct PipelineConfig {
    /// The directory task workspaces are created under.
    pub workspace_root: PathBuf,
    /// The image the test command runs in.
    pub docker_image: String,
    /// A Dockerfile to build before running; `None` uses the image as it is.
    pub dockerfile: Option<PathBuf>,
    /// The directory inside the container the workspace is mounted at.
    pub container_workdir: String,
    /// Environment variables for every `docker run`/`docker exec`.
    pub env: BTreeMap<String, String>,
    /// The per-command timeout in milliseconds.
    pub timeout_ms: u64,
    /// Leave the container running after the task instead of removing it.
    pub keep_container: bool,
    /// How often the test command is executed while it keeps failing; clamped
    /// to `1..=5` by [`effective_max_attempts`](Self::effective_max_attempts).
    pub max_attempts: usize,
    /// The score policy used when a task does not select one itself.
    pub score_spec: ScoreSpec,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            workspace_root: PathBuf::from(".openjiuwen/eval_workspaces"),
            docker_image: String::from("openjiuwen-eval:latest"),
            dockerfile: None,
            container_workdir: String::from("/workspace"),
            env: BTreeMap::new(),
            timeout_ms: DEFAULT_TIMEOUT_MS,
            keep_container: false,
            max_attempts: MIN_ATTEMPTS,
            score_spec: ScoreSpec::Auto,
        }
    }
}

impl PipelineConfig {
    /// The default config with `workspace_root` and `docker_image` set.
    #[must_use]
    pub fn new(workspace_root: impl Into<PathBuf>, docker_image: impl Into<String>) -> Self {
        Self {
            workspace_root: workspace_root.into(),
            docker_image: docker_image.into(),
            ..Self::default()
        }
    }

    /// Sets the Dockerfile to build before running.
    #[must_use]
    pub fn with_dockerfile(mut self, dockerfile: impl Into<PathBuf>) -> Self {
        self.dockerfile = Some(dockerfile.into());
        self
    }

    /// Sets the container workdir.
    #[must_use]
    pub fn with_container_workdir(mut self, container_workdir: impl Into<String>) -> Self {
        self.container_workdir = container_workdir.into();
        self
    }

    /// Replaces the environment variables.
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

    /// Leaves the container running after a task.
    #[must_use]
    pub fn with_keep_container(mut self) -> Self {
        self.keep_container = true;
        self
    }

    /// Sets the attempt count; [`effective_max_attempts`](Self::effective_max_attempts)
    /// still clamps it to `1..=5`.
    #[must_use]
    pub fn with_max_attempts(mut self, max_attempts: usize) -> Self {
        self.max_attempts = max_attempts;
        self
    }

    /// Sets the score policy used when a task does not select one itself.
    #[must_use]
    pub fn with_score_spec(mut self, score_spec: ScoreSpec) -> Self {
        self.score_spec = score_spec;
        self
    }

    /// The attempt count clamped to `MIN_ATTEMPTS..=MAX_ATTEMPTS`.
    #[must_use]
    pub fn effective_max_attempts(&self) -> usize {
        self.max_attempts.clamp(MIN_ATTEMPTS, MAX_ATTEMPTS)
    }

    /// Rejects an empty image, workdir, or workspace root.
    pub fn validate(&self) -> Result<()> {
        if self.docker_image.trim().is_empty() {
            return Err(Error::validation("docker_image must not be empty"));
        }
        if self.container_workdir.trim().is_empty() {
            return Err(Error::validation("container_workdir must not be empty"));
        }
        if self.workspace_root.as_os_str().is_empty() {
            return Err(Error::validation("workspace_root must not be empty"));
        }
        Ok(())
    }
}

/// Aggregate counts over a batch of [`TaskReport`]s.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct PipelineSummary {
    /// How many tasks were evaluated.
    pub tasks: usize,
    /// How many of them passed.
    pub passed: usize,
    /// How many of them did not pass.
    pub failed: usize,
    /// The mean score over every task, `0.0` for an empty batch.
    pub mean_score: f64,
}

impl PipelineSummary {
    /// Summarises `reports`, counting a report as passed only when its
    /// [`passed`](TaskReport::passed) flag is set.
    #[must_use]
    pub fn from_reports(reports: &[TaskReport]) -> Self {
        let tasks = reports.len();
        let passed = reports.iter().filter(|report| report.passed).count();
        #[allow(clippy::cast_precision_loss)]
        let mean_score = if tasks == 0 {
            0.0
        } else {
            reports.iter().map(|report| report.score).sum::<f64>() / tasks as f64
        };
        Self {
            tasks,
            passed,
            failed: tasks - passed,
            mean_score,
        }
    }
}

/// Runs tasks through the Docker + git lifecycle.
///
/// Holds the configuration and one [`CommandRunner`]; both are cheap to clone
/// and the pipeline is safe to share behind an `Arc`.
#[derive(Debug, Clone)]
pub struct EvaluationPipeline {
    config: PipelineConfig,
    runner: Arc<dyn CommandRunner>,
}

impl EvaluationPipeline {
    /// Creates a pipeline with `config`, running commands through `runner`.
    #[must_use]
    pub fn new(config: PipelineConfig, runner: Arc<dyn CommandRunner>) -> Self {
        Self { config, runner }
    }

    /// Creates a pipeline with the default config, running commands through
    /// `runner`.
    #[must_use]
    pub fn with_docker(runner: Arc<dyn CommandRunner>) -> Self {
        Self::new(PipelineConfig::default(), runner)
    }

    /// The config this pipeline runs with.
    #[must_use]
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// The command runner this pipeline shells through.
    #[must_use]
    pub fn runner(&self) -> &Arc<dyn CommandRunner> {
        &self.runner
    }

    /// Evaluates one task end to end.
    ///
    /// A failing test is a normal [`TaskReport`], not an error. Errors are
    /// reserved for stages that could not run at all: an invalid task, a missing
    /// `docker` binary ([`Error::Unsupported`]), a failed image build or clone, a
    /// failed copy, or a score that could not be parsed. The container created
    /// for this task is removed before either outcome is returned, unless
    /// [`PipelineConfig::keep_container`] is set.
    #[allow(clippy::too_many_lines)]
    pub async fn run_task(
        &self,
        task: &EvaluationTask,
        cancel: &CancellationToken,
    ) -> Result<TaskReport> {
        task.validate()?;
        self.config.validate()?;
        check_cancel(cancel)?;

        let workspace = self.workspace_for(task);
        let mut environment = self.environment(task, &workspace);
        environment.ensure_available().await?;
        check_cancel(cancel)?;

        if let Some(dockerfile) = &self.config.dockerfile {
            let context = self.build_context(task, dockerfile);
            environment
                .build(&context, dockerfile, &self.config.docker_image)
                .await?;
            check_cancel(cancel)?;
        }

        let spec = GitWorkspaceSpec::from_environment_spec(&task.environment_spec);
        let preparer = WorkspacePreparer::new(Arc::clone(&self.runner));
        let prepared_tree = preparer.prepare(&workspace, &spec, cancel).await?;
        check_cancel(cancel)?;

        environment.start().await?;
        let outcome = self
            .execute(&environment, task, &prepared_tree.workspace, cancel)
            .await;
        let container_id = environment.container_id().map(str::to_owned);
        let stopped = if self.config.keep_container {
            Ok(())
        } else {
            environment.stop().await
        };
        match outcome {
            Ok(mut report) => {
                report.container_id = container_id;
                stopped?;
                Ok(report)
            }
            Err(error) => {
                if let Err(stop_error) = stopped {
                    tracing::warn!(
                        task = %task.task_id,
                        error = %stop_error,
                        "failed to remove the container after a failed task step"
                    );
                }
                Err(error)
            }
        }
    }

    /// Evaluates every task sequentially.
    ///
    /// A per-task error becomes a [`TaskReport::failed`] report and the batch
    /// continues, so one broken environment recipe does not hide the rest of the
    /// results. Cancellation is the exception: it aborts the batch with
    /// [`Error::Cancelled`], because no later task could run either.
    pub async fn run_all(
        &self,
        tasks: &[EvaluationTask],
        cancel: &CancellationToken,
    ) -> Result<Vec<TaskReport>> {
        let mut reports = Vec::with_capacity(tasks.len());
        for task in tasks {
            check_cancel(cancel)?;
            match self.run_task(task, cancel).await {
                Ok(report) => reports.push(report),
                Err(error) if error.is_cancelled() => return Err(error),
                Err(error) => reports.push(TaskReport::failed(&task.task_id, error.to_string())),
            }
        }
        Ok(reports)
    }

    /// Summarises `reports`.
    #[must_use]
    pub fn summary(reports: &[TaskReport]) -> PipelineSummary {
        PipelineSummary::from_reports(reports)
    }

    /// Copies the workspace in, executes the test command, and scores it.
    async fn execute(
        &self,
        environment: &DockerEnvironment,
        task: &EvaluationTask,
        workspace: &Path,
        cancel: &CancellationToken,
    ) -> Result<TaskReport> {
        environment
            .copy_to(workspace, &self.config.container_workdir)
            .await?;
        check_cancel(cancel)?;
        let score_spec = self.effective_score_spec(task);
        let attempts = self.config.effective_max_attempts();
        let mut last: Option<(ExecResult, ScoredRun, usize)> = None;
        for attempt in 1..=attempts {
            check_cancel(cancel)?;
            let exec = environment.exec(&task.test_command).await?;
            let scored = parse_score(&exec, score_spec)?;
            let passed = scored.passed;
            last = Some((exec, scored, attempt));
            if passed {
                break;
            }
        }
        let (exec, scored, attempts) = last.ok_or_else(|| {
            Error::validation("the pipeline executed no attempt; max_attempts is zero")
        })?;
        let failure = (!scored.passed).then_some(scored.detail);
        Ok(TaskReport::new(
            task.task_id.clone(),
            exec,
            scored.score,
            scored.passed,
            attempts,
            failure,
        ))
    }

    /// The score policy for `task`: its own when it selects one, otherwise the
    /// pipeline's default.
    ///
    /// A task cannot distinguish "unset" from "explicitly `ExitCode`" because
    /// [`ScoreSpec::ExitCode`] is its serde default, so `ExitCode` means "use
    /// [`PipelineConfig::score_spec`]".
    fn effective_score_spec<'a>(&'a self, task: &'a EvaluationTask) -> &'a ScoreSpec {
        if task.score_spec == ScoreSpec::default() {
            &self.config.score_spec
        } else {
            &task.score_spec
        }
    }

    /// The absolute workspace directory for `task`.
    fn workspace_for(&self, task: &EvaluationTask) -> PathBuf {
        let relative = self
            .config
            .workspace_root
            .join(sanitize_task_id(&task.task_id));
        std::path::absolute(&relative).unwrap_or(relative)
    }

    /// The build context for `dockerfile`: the task's `build_context`, else the
    /// Dockerfile's directory, else the workspace root.
    fn build_context(&self, task: &EvaluationTask, dockerfile: &Path) -> PathBuf {
        let configured = task
            .environment_spec
            .get("build_context")
            .and_then(Value::as_str)
            .filter(|context| !context.trim().is_empty())
            .map(PathBuf::from);
        configured
            .or_else(|| dockerfile.parent().map(Path::to_path_buf))
            .unwrap_or_else(|| self.config.workspace_root.clone())
    }

    /// The Docker environment for `task`.
    fn environment(&self, task: &EvaluationTask, workspace: &Path) -> DockerEnvironment {
        DockerEnvironment::new(
            Arc::clone(&self.runner),
            self.config.docker_image.clone(),
            container_name(&task.task_id),
            workspace.to_path_buf(),
            self.config.container_workdir.clone(),
        )
        .with_env(self.config.env.clone())
        .with_timeout_ms(self.config.timeout_ms)
    }
}
