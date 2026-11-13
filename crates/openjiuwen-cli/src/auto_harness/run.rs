//! `openjiuwen auto-harness run`: configuration, task resolution, dry run.
//!
//! Ported from Python's `_run_auto_harness` (`openjiuwen/harness/cli/cli.py`),
//! restricted to the part the Rust port can honour end to end: resolving the
//! data directory, loading and overriding [`AutoHarnessConfig`], turning
//! `--task` / `--task-file` into [`OptimizationTask`]s, and the `--dry-run` task
//! list. Everything after that point — assess, plan, implement, verify — needs a
//! harness evaluation runner and an LLM-driven pipeline the `openjiuwen-rsi`
//! crate does not ship, so it returns
//! [`Error::Unsupported`](openjiuwen_common::error::Error::Unsupported) instead
//! of pretending to run.
use std::path::{Path, PathBuf};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_harness::cli::{Cli as HarnessCli, CliOptions};
use openjiuwen_rsi::auto_harness::{
    AutoHarnessConfig, OptimizationTask, PipelinePreference, TaskStatus,
};
use serde::Deserialize;
use serde_json::Value;

use super::data_dir;

/// The reason every non-dry auto-harness run fails today.
const RUN_DEFERRED: &str = "openjiuwen-rsi ships no harness evaluation runner and no LLM-driven \
                            assess/plan/implement pipeline";

/// Options of `auto-harness run`.
///
/// Flag names, value names and defaults follow Python's Click decorators; the
/// help texts are English renderings of the same sentences.
#[derive(Debug, Clone, clap::Args)]
pub(crate) struct RunCommand {
    /// Manually specify a single task description.
    #[arg(long, value_name = "TEXT")]
    pub(crate) task: Option<String>,
    /// Load the task list from a JSON file.
    #[arg(long = "task-file", value_name = "PATH")]
    pub(crate) task_file: Option<PathBuf>,
    /// Resolve and print the task list only; implement nothing.
    #[arg(long = "dry-run")]
    pub(crate) dry_run: bool,
    /// Only run the given stage.
    #[arg(long, value_enum)]
    pub(crate) stage: Option<Stage>,
    /// Do not push or open a merge request.
    ///
    /// Accepted for command-line parity only: the Rust port has no push step, so
    /// this flag changes nothing.
    #[arg(long = "no-push")]
    pub(crate) no_push: bool,
    /// Override the session budget (seconds).
    #[arg(long, value_name = "SECS")]
    pub(crate) budget: Option<f64>,
    /// Natural-language optimisation goal driving the whole assess/plan flow.
    #[arg(long, value_name = "TEXT")]
    pub(crate) goal: Option<String>,
    /// Session pipeline to use; defaults to `meta`.
    #[arg(long, value_enum)]
    pub(crate) pipeline: Option<Pipeline>,
}

/// The stage selector of `--stage` (Python `click.Choice`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum Stage {
    /// Assessment stage.
    Assess,
    /// Planning stage.
    Plan,
    /// Implementation stage.
    Implement,
    /// Verification stage.
    Verify,
}

impl Stage {
    /// Returns the wire name of the stage: `assess`, `plan`, `implement`, `verify`.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Assess => "assess",
            Self::Plan => "plan",
            Self::Implement => "implement",
            Self::Verify => "verify",
        }
    }
}

/// The pipeline selector of `--pipeline` (Python `click.Choice`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum Pipeline {
    /// The meta-evolve pipeline (the Python default when the flag is absent).
    Meta,
    /// The extended-evolve pipeline.
    Extended,
    /// Let the pipeline selector choose.
    Auto,
}

impl From<Pipeline> for PipelinePreference {
    fn from(pipeline: Pipeline) -> Self {
        match pipeline {
            Pipeline::Meta => Self::Meta,
            Pipeline::Extended => Self::Extended,
            Pipeline::Auto => Self::Auto,
        }
    }
}

/// One entry of a `--task-file` document.
///
/// Unknown keys are ignored, matching the workspace convention and Python's
/// `item.get(...)` access.
#[derive(Debug, Clone, Deserialize)]
struct TaskEntry {
    /// Required task topic.
    topic: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    files: Vec<String>,
}

impl TaskEntry {
    /// Turns the entry into the session task record Python builds.
    fn into_task(self) -> OptimizationTask {
        OptimizationTask {
            topic: self.topic,
            description: self.description,
            files: self.files,
            status: TaskStatus::Pending,
            ..OptimizationTask::default()
        }
    }
}

/// Runs `auto-harness run`.
///
/// # Errors
///
/// Returns [`Error::Config`] when the read-only inputs cannot be read or written,
/// [`Error::Validation`] when a task file is malformed or the loaded
/// configuration is invalid, and [`Error::Unsupported`] for every non-dry run.
pub(crate) async fn run(args: &RunCommand, global: &HarnessCli) -> Result<()> {
    let options = CliOptions::resolve(global)?;
    let data_dir = data_dir(global, &options);
    let config_path = data_dir.join("config.yaml");
    let mut config = AutoHarnessConfig::load_or_default(&config_path).await;
    config.data_dir = data_dir;
    if let Some(budget) = args.budget {
        config.session_budget_secs = budget;
        config.task_timeout_secs = config.task_timeout_secs.min(budget * 0.95);
    }
    if let Some(goal) = &args.goal {
        config.optimization_goal.clone_from(goal);
    }
    // Python defaults the preference to `meta` even though the schema default is
    // `auto`; the flag's absence therefore means `meta`.
    config.pipeline_preference = args
        .pipeline
        .map_or(PipelinePreference::Meta, PipelinePreference::from);
    config.validate()?;
    // There is no push step in the Rust port, so the flag only needs to be
    // visible to a diagnostic; it deliberately changes no state.
    tracing::debug!(no_push = args.no_push, "auto-harness has no push step");

    let tasks = resolve_tasks(args).await?;
    if args.dry_run {
        return write_dry_run(&config, &tasks).await;
    }
    if !tasks.is_empty() {
        println!("using {} manually specified task(s)", tasks.len());
    }
    let stage_note = args.stage.map_or_else(String::new, |stage| {
        format!("; stage `{}` cannot run", stage.as_str())
    });
    Err(Error::unsupported(format!(
        "auto-harness session execution is not implemented in the Rust port: {RUN_DEFERRED}\
         {stage_note}; use --dry-run to print the resolved task list"
    )))
}

/// Resolves the manual task list exactly as Python does.
///
/// `--task` wins over `--task-file`; with neither, the list is empty and the
/// session would generate its own tasks.
async fn resolve_tasks(args: &RunCommand) -> Result<Vec<OptimizationTask>> {
    if let Some(topic) = &args.task {
        return Ok(vec![OptimizationTask {
            topic: topic.clone(),
            status: TaskStatus::Pending,
            ..OptimizationTask::default()
        }]);
    }
    match &args.task_file {
        Some(path) => load_task_file(path).await,
        None => Ok(Vec::new()),
    }
}

/// Reads and parses a `--task-file` document.
async fn load_task_file(path: &Path) -> Result<Vec<OptimizationTask>> {
    let text = tokio::fs::read_to_string(path)
        .await
        .map_err(|err| Error::config(format!("cannot read task file {}: {err}", path.display())))?;
    parse_task_file(path, &text)
}

/// Parses the text of a `--task-file` document.
///
/// A JSON object is a single task, a JSON array is a list of tasks, and anything
/// else is a validation error naming the file.
fn parse_task_file(path: &Path, text: &str) -> Result<Vec<OptimizationTask>> {
    let value: Value = serde_json::from_str(text).map_err(|err| {
        Error::validation(format!(
            "invalid JSON in task file {}: {err}",
            path.display()
        ))
    })?;
    let entries = match value {
        Value::Object(_) => vec![value],
        Value::Array(items) => items,
        other => {
            return Err(Error::validation(format!(
                "task file {} must contain a JSON object or array, found {}",
                path.display(),
                json_type(&other)
            )));
        }
    };
    entries
        .into_iter()
        .map(|entry| {
            serde_json::from_value::<TaskEntry>(entry)
                .map(TaskEntry::into_task)
                .map_err(|err| {
                    Error::validation(format!("invalid task entry in {}: {err}", path.display()))
                })
        })
        .collect()
}

/// Names the JSON kind of `value` for a diagnostic.
fn json_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

/// Writes `<runs>/tasks.json` and echoes the same array on stdout.
///
/// Python prints the array on stdout and the note on stdout too; the Rust port
/// keeps the note on **stderr** so stdout stays machine-readable (documented
/// difference).
async fn write_dry_run(config: &AutoHarnessConfig, tasks: &[OptimizationTask]) -> Result<()> {
    let runs_dir = config.resolved_runs_dir();
    tokio::fs::create_dir_all(&runs_dir).await.map_err(|err| {
        Error::config(format!(
            "cannot create runs directory {}: {err}",
            runs_dir.display()
        ))
    })?;
    let rows: Vec<Value> = tasks
        .iter()
        .map(|task| {
            serde_json::json!({
                "topic": task.topic,
                "description": task.description,
                "files": task.files,
            })
        })
        .collect();
    let pretty = serde_json::to_string_pretty(&rows)
        .map_err(|err| Error::config(format!("cannot serialise the task list: {err}")))?;
    let path = runs_dir.join("tasks.json");
    tokio::fs::write(&path, &pretty).await.map_err(|err| {
        Error::config(format!("cannot write task list {}: {err}", path.display()))
    })?;
    println!("{pretty}");
    eprintln!("[dry-run] task list -> {}", path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    //! Task resolution tests: `--task`, `--task-file` (object/array/malformed).
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::path::Path;

    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_test_support::TempWorkspace;

    use super::{parse_task_file, resolve_tasks, RunCommand};
    use crate::auto_harness::run::{Pipeline, Stage};

    /// Builds a `run` command with only the field under test set.
    fn command(task: Option<&str>, task_file: Option<&Path>) -> RunCommand {
        RunCommand {
            task: task.map(str::to_owned),
            task_file: task_file.map(Path::to_path_buf),
            dry_run: false,
            stage: None,
            no_push: false,
            budget: None,
            goal: None,
            pipeline: None,
        }
    }

    #[test]
    fn task_file_object_is_one_task() {
        let path = Path::new("tasks.json");
        let tasks = parse_task_file(
            path,
            r#"{"topic": "speed up search", "description": "cache hits", "files": ["a.rs"]}"#,
        )
        .unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].topic, "speed up search");
        assert_eq!(tasks[0].description, "cache hits");
        assert_eq!(tasks[0].files, vec!["a.rs".to_owned()]);
        assert_eq!(
            tasks[0].status,
            openjiuwen_rsi::auto_harness::TaskStatus::Pending
        );
        assert!(tasks[0].issue_ref.is_none());
        assert!(tasks[0].pipeline_name.is_empty(), "no pipeline assigned");
    }

    #[test]
    fn task_file_array_keeps_order_and_tolerates_unknown_keys() {
        let path = Path::new("tasks.json");
        let tasks = parse_task_file(
            path,
            r#"[
                {"topic": "first", "extra": {"nested": true}},
                {"topic": "second", "files": ["x", "y"], "description": "d"}
            ]"#,
        )
        .unwrap();
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].topic, "first");
        assert_eq!(tasks[0].description, "");
        assert!(tasks[0].files.is_empty(), "no files listed");
        assert_eq!(tasks[1].topic, "second");
        assert_eq!(tasks[1].files, vec!["x".to_owned(), "y".to_owned()]);
    }

    #[test]
    fn malformed_task_file_is_a_validation_error_naming_the_path() {
        let path = Path::new("broken.json");
        let error = parse_task_file(path, "{not json").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("broken.json"), "{error}");
    }

    #[test]
    fn task_without_topic_is_a_validation_error() {
        let path = Path::new("tasks.json");
        let error = parse_task_file(path, r#"[{"description": "no topic"}]"#).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("topic"), "{error}");
    }

    #[test]
    fn scalar_document_is_a_validation_error() {
        let path = Path::new("tasks.json");
        let error = parse_task_file(path, "42").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("a number"), "{error}");
    }

    #[test]
    fn task_flag_wins_over_the_task_file() {
        let workspace = TempWorkspace::new().unwrap();
        let file = workspace.path("tasks.json");
        std::fs::write(&file, r#"[{"topic": "from file"}]"#).unwrap();
        let cli = command(Some("from flag"), Some(&file));
        let tasks = futures_lite_block_on(resolve_tasks(&cli)).unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].topic, "from flag");
    }

    #[test]
    fn task_file_is_read_and_resolved() {
        let workspace = TempWorkspace::new().unwrap();
        let file = workspace.path("tasks.json");
        std::fs::write(&file, r#"{"topic": "from file", "files": ["src/lib.rs"]}"#).unwrap();
        let cli = command(None, Some(&file));
        let tasks = futures_lite_block_on(resolve_tasks(&cli)).unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].topic, "from file");
        assert_eq!(tasks[0].files, vec!["src/lib.rs".to_owned()]);
    }

    #[test]
    fn missing_task_file_is_a_config_error() {
        let workspace = TempWorkspace::new().unwrap();
        let cli = command(None, Some(&workspace.path("absent.json")));
        let error = futures_lite_block_on(resolve_tasks(&cli)).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
        assert!(error.to_string().contains("absent.json"), "{error}");
    }

    #[test]
    fn no_task_inputs_resolve_to_an_empty_list() {
        let tasks = futures_lite_block_on(resolve_tasks(&command(None, None))).unwrap();
        assert!(tasks.is_empty(), "no manual tasks");
    }

    #[test]
    fn stage_and_pipeline_wire_names_match_python() {
        let mut cli = command(None, None);
        cli.stage = Some(Stage::Assess);
        cli.pipeline = Some(Pipeline::Extended);
        assert_eq!(cli.stage.unwrap().as_str(), "assess");
        assert_eq!(
            openjiuwen_rsi::auto_harness::PipelinePreference::from(cli.pipeline.unwrap()),
            openjiuwen_rsi::auto_harness::PipelinePreference::Extended
        );
    }

    /// Runs one short future to completion on a throwaway current-thread runtime.
    fn futures_lite_block_on<T>(future: impl std::future::Future<Output = T>) -> T {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(future)
    }
}
