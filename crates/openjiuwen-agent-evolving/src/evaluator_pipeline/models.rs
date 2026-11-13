//! The pipeline records: [`ExecResult`], [`CommandSpec`], [`GitWorkspaceSpec`],
//! [`EvaluationTask`], and [`TaskReport`].
//!
//! Python: `evaluator_pipeline/models.py` (`ExecResult`, `Task`) and the
//! environment/workspace inputs the pipeline driver derives from
//! `Task.environment_spec`. The agent-rollout records (`AgentContext`,
//! `AgentRunResult`, `IterationResult`, `PipelineResult`) are deliberately not
//! ported: the Rust pipeline evaluates one task against one test command and
//! reports [`TaskReport`], leaving agent adapters and evolution loops to the
//! caller.
//!
//! Everything here is pure data plus validation: no clocks, no environment
//! reads, no I/O. [`TaskReport`] carries no timestamps so two identical runs
//! serialize identically.
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::scoring::ScoreSpec;

/// The longest slug [`sanitize_task_id`] produces.
///
/// Container names and workspace directory names are both derived from a task
/// id, so the slug is bounded to stay well inside filesystem and Docker name
/// limits.
pub const MAX_TASK_ID_SLUG_CHARS: usize = 64;

/// Maps a task id onto a slug usable in a Docker container name and a directory
/// name.
///
/// ASCII alphanumerics, `-`, `_` and `.` are kept (letters lowercased), every
/// other character becomes `-`, and leading/trailing separators are trimmed.
/// An empty or fully-sanitised id becomes `task`, so the result is never empty.
///
/// # Example
///
/// ```
/// use openjiuwen_agent_evolving::evaluator_pipeline::sanitize_task_id;
///
/// assert_eq!(sanitize_task_id("Task 42/b"), "task-42-b");
/// assert_eq!(sanitize_task_id(""), "task");
/// ```
#[must_use]
pub fn sanitize_task_id(task_id: &str) -> String {
    let mapped: String = task_id
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    let trimmed = mapped.trim_matches(|ch: char| matches!(ch, '-' | '.' | '_'));
    if trimmed.is_empty() {
        return String::from("task");
    }
    trimmed.chars().take(MAX_TASK_ID_SLUG_CHARS).collect()
}

/// The captured result of one external command.
///
/// Python: `evaluator_pipeline.models.ExecResult`. A non-zero `returncode` is a
/// normal result, not an error; [`success`](ExecResult::success) and the score
/// parsers decide what it means. A timed-out command reports
/// [`timed_out`](ExecResult::timed_out) with a `returncode` of `-1`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ExecResult {
    /// Everything the command wrote to standard output.
    #[serde(default)]
    pub stdout: String,
    /// Everything the command wrote to standard error.
    #[serde(default)]
    pub stderr: String,
    /// The process exit code; `-1` when the process was killed or never started.
    #[serde(default = "unknown_returncode")]
    pub returncode: i32,
    /// `true` when the runner killed the process because it exceeded its timeout.
    #[serde(default)]
    pub timed_out: bool,
}

/// The Python default return code for a result that never completed: `-1`.
const fn unknown_returncode() -> i32 {
    -1
}

impl Default for ExecResult {
    fn default() -> Self {
        Self {
            stdout: String::new(),
            stderr: String::new(),
            returncode: unknown_returncode(),
            timed_out: false,
        }
    }
}

impl ExecResult {
    /// Builds a completed result from the captured streams and exit code.
    #[must_use]
    pub fn new(stdout: impl Into<String>, stderr: impl Into<String>, returncode: i32) -> Self {
        Self {
            stdout: stdout.into(),
            stderr: stderr.into(),
            returncode,
            timed_out: false,
        }
    }

    /// A successful result with the given standard output.
    #[must_use]
    pub fn ok(stdout: impl Into<String>) -> Self {
        Self::new(stdout, String::new(), 0)
    }

    /// A failed result with the given standard error and exit code.
    #[must_use]
    pub fn failure(stderr: impl Into<String>, returncode: i32) -> Self {
        Self::new(String::new(), stderr, returncode)
    }

    /// A result for a process the runner killed after its timeout expired.
    #[must_use]
    pub fn timeout(stderr: impl Into<String>) -> Self {
        Self {
            stdout: String::new(),
            stderr: stderr.into(),
            returncode: unknown_returncode(),
            timed_out: true,
        }
    }

    /// `true` only for an un-timed-out exit code of `0`.
    ///
    /// Python's `success` checks the exit code alone; the Rust port also rejects
    /// a timed-out result, because a killed process never counts as a success
    /// even when a runner reports `0`.
    #[must_use]
    pub fn success(&self) -> bool {
        !self.timed_out && self.returncode == 0
    }

    /// The non-empty, trimmed lines of standard output.
    ///
    /// This is the projection the score parsers scan, exposed for consumers that
    /// want to inspect the output the same way.
    #[must_use]
    pub fn stdout_lines(&self) -> Vec<&str> {
        self.stdout
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect()
    }

    /// Standard output followed by standard error, each trimmed of trailing
    /// whitespace, separated by one newline when both are non-empty.
    ///
    /// Useful for diagnostics and for parsers that accept a summary on either
    /// stream.
    #[must_use]
    pub fn combined_output(&self) -> String {
        let stdout = self.stdout.trim_end();
        let stderr = self.stderr.trim_end();
        match (stdout.is_empty(), stderr.is_empty()) {
            (true, true) => String::new(),
            (false, true) => stdout.to_owned(),
            (true, false) => stderr.to_owned(),
            (false, false) => format!("{stdout}\n{stderr}"),
        }
    }
}

/// A fully described external command: the only shape the pipeline executes.
///
/// Python shells out with ad-hoc argument lists; the Rust port makes every
/// invocation a value so the [`CommandRunner`](super::CommandRunner) seam can be
/// scripted and so tests can assert the exact command line.
///
/// The timeout is stored as [`timeout_ms`](CommandSpec::timeout_ms) so the whole
/// record round-trips through JSON without a custom `Duration` adapter;
/// [`timeout`](CommandSpec::timeout) returns it as a [`Duration`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct CommandSpec {
    /// The executable to spawn; looked up on `PATH` when it is not a path.
    pub program: String,
    /// The arguments, passed verbatim after the program name.
    #[serde(default)]
    pub args: Vec<String>,
    /// The working directory, when the command must not inherit the process cwd.
    #[serde(default)]
    pub cwd: Option<PathBuf>,
    /// Additional environment variables layered over the process environment.
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    /// The wall-clock budget in milliseconds; `None` means no timeout.
    #[serde(default)]
    pub timeout_ms: Option<u64>,
}

impl CommandSpec {
    /// Starts a command for `program` with no arguments.
    #[must_use]
    pub fn new(program: impl Into<String>) -> Self {
        Self {
            program: program.into(),
            ..Self::default()
        }
    }

    /// Appends one argument.
    #[must_use]
    pub fn arg(mut self, arg: impl Into<String>) -> Self {
        self.args.push(arg.into());
        self
    }

    /// Appends every argument of `args`, in order.
    #[must_use]
    pub fn args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.args.extend(args.into_iter().map(Into::into));
        self
    }

    /// Sets the working directory.
    #[must_use]
    pub fn with_cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }

    /// Adds one environment variable, replacing any previous value for the key.
    #[must_use]
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.insert(key.into(), value.into());
        self
    }

    /// Sets the timeout in milliseconds.
    #[must_use]
    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }

    /// Sets the timeout from a [`Duration`], truncated to whole milliseconds.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout_ms = Some(u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX));
        self
    }

    /// The timeout as a [`Duration`], when one is configured.
    #[must_use]
    pub fn timeout(&self) -> Option<Duration> {
        self.timeout_ms.map(Duration::from_millis)
    }

    /// The command as one space-separated line, for deterministic logs and
    /// error messages.
    ///
    /// Values are not quoted: this is a human-readable label, never input to a
    /// shell.
    #[must_use]
    pub fn command_line(&self) -> String {
        let mut line = self.program.clone();
        for arg in &self.args {
            line.push(' ');
            line.push_str(arg);
        }
        line
    }
}

/// How to obtain the source tree the test command runs against.
///
/// Python's benchmark adapter decided this imperatively; the Rust port makes it
/// a value so [`WorkspacePreparer`](super::WorkspacePreparer) is a pure function
/// of its inputs and tests can assert the exact `git` invocations.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct GitWorkspaceSpec {
    /// The repository to clone; `None` means "use the workspace as it is".
    #[serde(default)]
    pub url: Option<String>,
    /// The revision (branch, tag, or commit) to check out.
    #[serde(default)]
    pub revision: Option<String>,
    /// A unified diff applied with `git apply` after checkout.
    #[serde(default)]
    pub patch: Option<String>,
    /// Clone with `--depth 1` instead of the full history.
    #[serde(default)]
    pub shallow: bool,
    /// Initialise submodules after the checkout.
    #[serde(default)]
    pub submodules: bool,
}

impl GitWorkspaceSpec {
    /// A spec that clones `url` at the repository default revision.
    #[must_use]
    pub fn from_url(url: impl Into<String>) -> Self {
        Self {
            url: Some(url.into()),
            ..Self::default()
        }
    }

    /// Sets the revision to check out.
    #[must_use]
    pub fn with_revision(mut self, revision: impl Into<String>) -> Self {
        self.revision = Some(revision.into());
        self
    }

    /// Sets the unified diff to apply after checkout.
    #[must_use]
    pub fn with_patch(mut self, patch: impl Into<String>) -> Self {
        self.patch = Some(patch.into());
        self
    }

    /// Enables a shallow (`--depth 1`) clone.
    #[must_use]
    pub fn shallow(mut self) -> Self {
        self.shallow = true;
        self
    }

    /// Enables submodule initialisation.
    #[must_use]
    pub fn with_submodules(mut self) -> Self {
        self.submodules = true;
        self
    }

    /// Derives a spec from a task's `environment_spec` map.
    ///
    /// Recognised keys (first match wins):
    ///
    /// - repository URL: `repo_url`, `git_url`, `url`;
    /// - revision: `revision`, `commit`, `git_ref`, `ref`;
    /// - patch content: `patch`;
    /// - booleans: `shallow`, `submodules`.
    ///
    /// Values of the wrong JSON type are ignored rather than rejected, so a
    /// benchmark-specific environment block can carry unrelated keys.
    #[must_use]
    pub fn from_environment_spec(spec: &BTreeMap<String, Value>) -> Self {
        Self {
            url: string_field(spec, &["repo_url", "git_url", "url"]),
            revision: string_field(spec, &["revision", "commit", "git_ref", "ref"]),
            patch: string_field(spec, &["patch"]),
            shallow: bool_field(spec, "shallow"),
            submodules: bool_field(spec, "submodules"),
        }
    }
}

/// The first key of `keys` whose value is a JSON string.
fn string_field(spec: &BTreeMap<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        spec.get(*key)
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
    })
}

/// The boolean under `key`, defaulting to `false`.
fn bool_field(spec: &BTreeMap<String, Value>, key: &str) -> bool {
    spec.get(key).and_then(Value::as_bool).unwrap_or(false)
}

/// One benchmark task: the instruction, the environment recipe, and the test
/// command that decides the score.
///
/// Python: `evaluator_pipeline.models.Task`, extended with the Rust pipeline's
/// [`test_command`](EvaluationTask::test_command) and
/// [`score_spec`](EvaluationTask::score_spec). The Python adapter's
/// `metadata` map is dropped: nothing in the Rust pipeline reads it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct EvaluationTask {
    /// The task identifier; must be non-empty.
    pub task_id: String,
    /// The instruction given to the agent under evaluation.
    #[serde(default)]
    pub instruction: String,
    /// Benchmark-specific environment keys (repository, revision, build context).
    #[serde(default)]
    pub environment_spec: BTreeMap<String, Value>,
    /// Whether the task ships pre-defined skills.
    #[serde(default)]
    pub has_skills: bool,
    /// The names of the pre-defined skills.
    #[serde(default)]
    pub skills: Vec<String>,
    /// The argv executed inside the container to score the task; must be non-empty.
    pub test_command: Vec<String>,
    /// How the test command's result becomes a score.
    #[serde(default)]
    pub score_spec: ScoreSpec,
}

impl EvaluationTask {
    /// Builds a task and validates the identifier and the test command.
    pub fn new<I, S>(
        task_id: impl Into<String>,
        instruction: impl Into<String>,
        test_command: I,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let task = Self {
            task_id: task_id.into(),
            instruction: instruction.into(),
            environment_spec: BTreeMap::new(),
            has_skills: false,
            skills: Vec::new(),
            test_command: test_command.into_iter().map(Into::into).collect(),
            score_spec: ScoreSpec::default(),
        };
        task.validate()?;
        Ok(task)
    }

    /// Replaces the environment spec.
    #[must_use]
    pub fn with_environment(mut self, environment_spec: BTreeMap<String, Value>) -> Self {
        self.environment_spec = environment_spec;
        self
    }

    /// Declares the task's pre-defined skills.
    #[must_use]
    pub fn with_skills<I, S>(mut self, skills: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.skills = skills.into_iter().map(Into::into).collect();
        self.has_skills = !self.skills.is_empty();
        self
    }

    /// Sets the score policy for this task.
    #[must_use]
    pub fn with_score_spec(mut self, score_spec: ScoreSpec) -> Self {
        self.score_spec = score_spec;
        self
    }

    /// Rejects an empty identifier, an empty test command, or an empty argv
    /// element with [`Error::validation`].
    pub fn validate(&self) -> Result<()> {
        if self.task_id.trim().is_empty() {
            return Err(Error::validation("task_id must not be empty"));
        }
        if self.test_command.is_empty() {
            return Err(Error::validation(
                "test_command must contain at least one element",
            ));
        }
        if self.test_command.iter().any(|arg| arg.trim().is_empty()) {
            return Err(Error::validation(
                "test_command must not contain empty arguments",
            ));
        }
        Ok(())
    }

    /// The repository spec this task's `environment_spec` describes.
    #[must_use]
    pub fn workspace_spec(&self) -> GitWorkspaceSpec {
        GitWorkspaceSpec::from_environment_spec(&self.environment_spec)
    }
}

/// The outcome of evaluating one task.
///
/// A report always exists — failing tests, a missing `docker` binary turned into
/// a batch failure by [`run_all`](super::EvaluationPipeline::run_all), or a
/// malformed score all produce one. There are no wall-clock fields: `attempts`
/// and [`exec`](TaskReport::exec) are the only provenance, so reports are
/// reproducible.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TaskReport {
    /// The evaluated task's identifier.
    pub task_id: String,
    /// `true` only when the score parser reported a pass.
    pub passed: bool,
    /// The score in `0.0..=1.0`.
    pub score: f64,
    /// The last executed test command's result.
    #[serde(default)]
    pub exec: ExecResult,
    /// How many times the test command was executed.
    #[serde(default)]
    pub attempts: usize,
    /// The container the task ran in, when one was started.
    #[serde(default)]
    pub container_id: Option<String>,
    /// Why the task failed, when it did.
    #[serde(default)]
    pub failure: Option<String>,
}

impl TaskReport {
    /// Builds a completed report from a score and the final execution result.
    #[must_use]
    pub fn new(
        task_id: impl Into<String>,
        exec: ExecResult,
        score: f64,
        passed: bool,
        attempts: usize,
        failure: Option<String>,
    ) -> Self {
        Self {
            task_id: task_id.into(),
            passed,
            score,
            exec,
            attempts,
            container_id: None,
            failure,
        }
    }

    /// Builds a failed report for a task that never ran, carrying `message` as
    /// its failure reason and a `0.0` score.
    #[must_use]
    pub fn failed(task_id: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            task_id: task_id.into(),
            passed: false,
            score: 0.0,
            exec: ExecResult::default(),
            attempts: 0,
            container_id: None,
            failure: Some(message.into()),
        }
    }

    /// The task id, for grouping.
    #[must_use]
    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    /// `true` when the task produced a passing score.
    #[must_use]
    pub fn is_passed(&self) -> bool {
        self.passed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_id_sanitising_is_deterministic_and_bounded() {
        assert_eq!(sanitize_task_id("Task 42/b"), "task-42-b");
        assert_eq!(sanitize_task_id("a_b.c-d"), "a_b.c-d");
        assert_eq!(sanitize_task_id("  "), "task");
        assert_eq!(sanitize_task_id("--edge--"), "edge");
        let long = "x".repeat(MAX_TASK_ID_SLUG_CHARS + 10);
        assert_eq!(
            sanitize_task_id(&long).chars().count(),
            MAX_TASK_ID_SLUG_CHARS
        );
    }

    #[test]
    fn exec_result_helpers_project_the_streams() {
        let result = ExecResult::new("a\n\n  b  \n", "boom\n", 3);
        assert_eq!(result.stdout_lines(), vec!["a", "b"]);
        assert_eq!(result.combined_output(), "a\n\n  b\nboom");
        assert!(!result.success());
        assert!(ExecResult::ok("done").success());
        assert!(!ExecResult::timeout("slow").success());
        assert_eq!(ExecResult::default().returncode, -1);
    }

    #[test]
    fn command_spec_renders_a_deterministic_line() {
        let spec = CommandSpec::new("docker")
            .arg("run")
            .args(["-d", "--name"])
            .arg("box")
            .with_timeout(Duration::from_millis(1_500))
            .with_env("A", "1");
        assert_eq!(spec.command_line(), "docker run -d --name box");
        assert_eq!(spec.timeout(), Some(Duration::from_millis(1_500)));
        assert_eq!(spec.env.get("A").map(String::as_str), Some("1"));
    }

    #[test]
    fn workspace_spec_reads_the_known_environment_keys() {
        let mut spec = BTreeMap::new();
        spec.insert(
            "git_url".to_owned(),
            Value::String("https://example/repo".to_owned()),
        );
        spec.insert("commit".to_owned(), Value::String("abc123".to_owned()));
        spec.insert(
            "patch".to_owned(),
            Value::String("diff --git a/x b/x\n".to_owned()),
        );
        spec.insert("shallow".to_owned(), Value::Bool(true));
        spec.insert("unrelated".to_owned(), Value::Number(7.into()));
        let parsed = GitWorkspaceSpec::from_environment_spec(&spec);
        assert_eq!(parsed.url.as_deref(), Some("https://example/repo"));
        assert_eq!(parsed.revision.as_deref(), Some("abc123"));
        assert!(parsed.shallow);
        assert!(!parsed.submodules);
        assert!(parsed.patch.is_some());
    }

    #[test]
    fn tasks_validate_the_identifier_and_test_command() {
        assert!(EvaluationTask::new("id", "do it", ["pytest", "-q"]).is_ok());
        assert!(EvaluationTask::new("  ", "do it", ["pytest"]).is_err());
        assert!(EvaluationTask::new("id", "do it", Vec::<String>::new()).is_err());
        assert!(EvaluationTask::new("id", "do it", ["pytest", ""]).is_err());
        let task = EvaluationTask::new("id", "do it", ["pytest"]).unwrap();
        let task = task.with_skills(["alpha"]);
        assert!(task.has_skills);
        assert_eq!(task.skills, vec!["alpha".to_owned()]);
    }
}
