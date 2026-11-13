//! Terminal-bench evaluation: the task command, result parsing, command logs.
//!
//! Ports the portable core of Python `harness_rsi/evaluator/terminal_bench_runtime.py`
//! plus the benchmark-run half the Python runtime leaves to its caller:
//!
//! - [`TerminalBenchEvaluator::run_command`] builds
//!   `<command> run --dataset … --task-id … --output-path …`, merging standard
//!   error and applying the task timeout.
//! - [`parse_results`] reads the three result shapes the harness emits
//!   (`results` list, `tasks` map, `resolved_ids`/`unresolved_ids`) and the
//!   per-command log with Python's field names and a
//!   [`COMMAND_EXCERPT_LIMIT`]-character excerpt cap.
//! - [`TerminalBenchEvaluator::evaluate_case`] runs one case and reads the
//!   results file it wrote; a **missing** results file is an outcome with an
//!   error, not an `Err`, because a harness that wrote nothing is a case-level
//!   failure while a malformed file is a bug in the harness.
//!
//! Docker containers, the solver, and the terminal-bench package itself stay out
//! of scope behind the [`CommandRunner`] seam.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};

use crate::evolution::candidate::Candidate;
use crate::evolution::evaluation::{Case, CaseResult, EvaluationReport, Evaluator};
use crate::runtime::command::{CommandRunner, CommandSpec};

use super::case::EvaluationCase;
use super::sanitize_id;

/// The maximum characters kept from each captured command stream.
///
/// Python's `TerminalBenchCommandRecorder.max_stream_chars` default is 2000;
/// traces here are capped at 1000 characters with a `...` suffix.
pub const COMMAND_EXCERPT_LIMIT: usize = 1000;

/// The case metadata key that names the terminal-bench task.
pub const TASK_ID_KEY: &str = "task_id";

/// The error recorded when the harness wrote no results file.
pub const RESULTS_FILE_MISSING: &str = "results file missing";

/// The terminal-bench configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TerminalBenchConfig {
    /// The harness executable.
    pub command: String,
    /// The dataset name or path handed to the harness.
    pub dataset: String,
    /// Per-task timeout in seconds.
    pub timeout_secs: u64,
    /// Optional task filter (`--task-filter`) restricting the dataset.
    pub task_filter: Option<String>,
    /// Where results files are written.
    pub results_dir: PathBuf,
    /// Extra harness arguments appended verbatim.
    pub extra_args: Vec<String>,
}

impl Default for TerminalBenchConfig {
    /// Returns the Python runtime defaults.
    fn default() -> Self {
        Self {
            command: "terminal-bench".to_owned(),
            dataset: String::new(),
            timeout_secs: 3600,
            task_filter: None,
            results_dir: PathBuf::from("terminal_bench_results"),
            extra_args: Vec::new(),
        }
    }
}

/// One recorded shell command from a task run.
///
/// Field names match Python's `TerminalBenchCommandLogEntry`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TerminalBenchCommandLog {
    /// The command line the solver ran.
    pub command: String,
    /// The working directory the command ran in.
    pub cwd: String,
    /// The exit code (`0` when the entry does not record one).
    pub exit_code: i32,
    /// The bounded tail of standard output.
    pub stdout_excerpt: String,
    /// The bounded tail of standard error.
    pub stderr_excerpt: String,
    /// The per-command timeout in seconds (`0` when unrecorded).
    pub timeout_sec: u64,
    /// Whether the command was started in the background.
    pub background: bool,
}

/// The outcome of evaluating one terminal-bench task.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TerminalBenchOutcome {
    /// The task id.
    pub task_id: String,
    /// Whether the task resolved.
    pub passed: bool,
    /// `Some(1.0)`/`Some(0.0)` derived from `passed`, or the harness score.
    pub score: Option<f64>,
    /// The recorded solver commands.
    pub log: Vec<TerminalBenchCommandLog>,
    /// The failure text, when the harness or the solver reported one.
    pub error: Option<String>,
    /// Diagnostics: exit code, results path, stdout excerpt, …
    pub metadata: BTreeMap<String, Value>,
}

/// Runs the terminal-bench harness for one case.
#[derive(Clone)]
pub struct TerminalBenchEvaluator {
    runner: Arc<dyn CommandRunner>,
    config: TerminalBenchConfig,
}

impl TerminalBenchEvaluator {
    /// Creates an evaluator over `runner` with `config`.
    #[must_use]
    pub fn new(runner: Arc<dyn CommandRunner>, config: TerminalBenchConfig) -> Self {
        Self { runner, config }
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &TerminalBenchConfig {
        &self.config
    }

    /// Builds the harness command for one task.
    ///
    /// `--task-filter` is appended only when [`TerminalBenchConfig::task_filter`]
    /// is set and non-empty; standard error is merged into the captured output
    /// so a harness that only logs to stderr still yields a usable trace.
    #[must_use]
    pub fn run_command(&self, task_id: &str, results_path: &Path) -> CommandSpec {
        let config = &self.config;
        let mut spec = CommandSpec::new(&config.command)
            .args(["run", "--dataset"])
            .arg(&config.dataset)
            .arg("--task-id")
            .arg(task_id)
            .arg("--output-path")
            .arg(results_path.display().to_string())
            .args(config.extra_args.iter().cloned())
            .merge_stderr()
            .timeout(Duration::from_secs(config.timeout_secs));
        if let Some(filter) = config
            .task_filter
            .as_deref()
            .filter(|filter| !filter.trim().is_empty())
        {
            spec = spec.arg("--task-filter").arg(filter);
        }
        spec
    }

    /// Evaluates one case: run the harness, then read and parse its results file.
    ///
    /// # Errors
    ///
    /// Returns any [`CommandRunner`] error (a missing harness is
    /// [`Error::Config`]) and [`Error::Validation`] when the results file exists
    /// but is malformed or describes an unknown shape.
    pub async fn evaluate_case(
        &self,
        case: &EvaluationCase,
        output_dir: &Path,
    ) -> Result<TerminalBenchOutcome> {
        let task_id = task_id_for(case);
        tokio::fs::create_dir_all(output_dir)
            .await
            .map_err(|error| {
                Error::session(format!(
                    "failed to create the terminal-bench output directory `{}`: {error}",
                    output_dir.display()
                ))
            })?;
        let results_path = output_dir.join(format!("{}.json", sanitize_id(&task_id)));
        let spec = self.run_command(&task_id, &results_path);
        let command_line = spec.command_line();
        let outcome = self.runner.run(spec).await?;
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "exit_code".to_owned(),
            outcome
                .code
                .map_or(Value::Null, |code| Value::from(i64::from(code))),
        );
        metadata.insert("timed_out".to_owned(), Value::from(outcome.timed_out));
        metadata.insert("command".to_owned(), Value::String(command_line));
        metadata.insert(
            "results_path".to_owned(),
            Value::String(results_path.display().to_string()),
        );
        metadata.insert(
            "stdout_excerpt".to_owned(),
            Value::String(outcome.truncated_output(COMMAND_EXCERPT_LIMIT)),
        );
        let Some(text) = tokio::fs::read_to_string(&results_path).await.ok() else {
            return Ok(TerminalBenchOutcome {
                task_id,
                passed: false,
                score: None,
                log: Vec::new(),
                error: Some(RESULTS_FILE_MISSING.to_owned()),
                metadata,
            });
        };
        let value: Value = serde_json::from_str(&text).map_err(|error| {
            Error::validation(format!(
                "terminal-bench results are not valid JSON: {error}"
            ))
        })?;
        let mut parsed = parse_results(&value, &task_id)?;
        parsed.metadata.extend(metadata);
        Ok(parsed)
    }
}

/// Resolves the task id: `metadata.task_id`, else the case id.
fn task_id_for(case: &EvaluationCase) -> String {
    case.metadata
        .get(TASK_ID_KEY)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|task_id| !task_id.is_empty())
        .map_or_else(|| case.case_id.clone(), str::to_owned)
}

/// Parses a terminal-bench results document for `task_id`.
///
/// Accepts `{"results": [{task_id, resolved|status, score, commands: [...]}]}`,
/// `{"tasks": {<task_id>: {…}}}`, and
/// `{"resolved_ids": [...], "unresolved_ids": [...]}`.
///
/// # Errors
///
/// Returns [`Error::Validation`] for an unknown document shape, a missing task,
/// or a malformed task entry.
pub fn parse_results(value: &Value, task_id: &str) -> Result<TerminalBenchOutcome> {
    let object = value
        .as_object()
        .ok_or_else(|| Error::validation("terminal-bench results must be a JSON object"))?;
    if let Some(results) = object.get("results").filter(|value| !value.is_null()) {
        let items = results
            .as_array()
            .ok_or_else(|| Error::validation("terminal-bench `results` must be a list"))?;
        let entry = items
            .iter()
            .find(|item| entry_task_id(item).as_deref() == Some(task_id))
            .ok_or_else(|| missing_task(task_id))?;
        return outcome_from_entry(task_id, entry);
    }
    if let Some(tasks) = object.get("tasks").filter(|value| !value.is_null()) {
        let entries = tasks
            .as_object()
            .ok_or_else(|| Error::validation("terminal-bench `tasks` must be an object"))?;
        let entry = entries.get(task_id).ok_or_else(|| missing_task(task_id))?;
        return outcome_from_entry(task_id, entry);
    }
    let resolved = id_list(object.get("resolved_ids"));
    let unresolved = id_list(object.get("unresolved_ids"));
    if resolved.iter().any(|id| id == task_id) {
        return Ok(list_outcome(task_id, true));
    }
    if unresolved.iter().any(|id| id == task_id) {
        return Ok(list_outcome(task_id, false));
    }
    Err(missing_task(task_id))
}

/// Builds the error for a results document that does not mention the task.
fn missing_task(task_id: &str) -> Error {
    Error::validation(format!(
        "terminal-bench results contain no task `{task_id}`"
    ))
}

/// Builds an outcome from an id-only results document.
fn list_outcome(task_id: &str, passed: bool) -> TerminalBenchOutcome {
    TerminalBenchOutcome {
        task_id: task_id.to_owned(),
        passed,
        score: Some(if passed { 1.0 } else { 0.0 }),
        log: Vec::new(),
        error: None,
        metadata: BTreeMap::new(),
    }
}

/// Builds an outcome from one task entry (an object or a bare boolean).
fn outcome_from_entry(task_id: &str, entry: &Value) -> Result<TerminalBenchOutcome> {
    if let Some(passed) = entry.as_bool() {
        return Ok(list_outcome(task_id, passed));
    }
    let object = entry.as_object().ok_or_else(|| {
        Error::validation("terminal-bench task entries must be objects or booleans")
    })?;
    let passed = match object.get("resolved").and_then(Value::as_bool) {
        Some(resolved) => resolved,
        None => match object.get("status").and_then(Value::as_str) {
            Some(status) => status_flag(status)?,
            None => {
                return Err(Error::validation(
                    "terminal-bench task entry must contain `resolved` or `status`",
                ))
            }
        },
    };
    let log = match object.get("commands").filter(|value| !value.is_null()) {
        None => Vec::new(),
        Some(value) => {
            let items = value
                .as_array()
                .ok_or_else(|| Error::validation("terminal-bench `commands` must be a list"))?;
            items
                .iter()
                .map(command_log)
                .collect::<Result<Vec<TerminalBenchCommandLog>>>()?
        }
    };
    Ok(TerminalBenchOutcome {
        task_id: task_id.to_owned(),
        passed,
        score: Some(score_or_default(object.get("score"), passed)),
        log,
        error: object
            .get("error")
            .and_then(Value::as_str)
            .map(str::to_owned),
        metadata: object
            .get("metadata")
            .and_then(Value::as_object)
            .map(|metadata| {
                metadata
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect()
            })
            .unwrap_or_default(),
    })
}

/// Maps a harness status literal to a pass flag.
fn status_flag(status: &str) -> Result<bool> {
    match status.trim().to_ascii_lowercase().as_str() {
        "resolved" | "passed" | "success" | "succeeded" => Ok(true),
        "unresolved" | "failed" | "failure" | "error" => Ok(false),
        other => Err(Error::validation(format!(
            "unknown terminal-bench status `{other}`"
        ))),
    }
}

/// Reads an id list, ignoring non-string entries.
fn id_list(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

/// Reads the `task_id`/`id` of one entry.
fn entry_task_id(value: &Value) -> Option<String> {
    ["task_id", "id"]
        .iter()
        .find_map(|key| value.get(*key).and_then(Value::as_str))
        .map(str::to_owned)
}

/// Reads a score, falling back to the pass flag.
fn score_or_default(value: Option<&Value>, passed: bool) -> f64 {
    match value.and_then(Value::as_f64) {
        Some(score) if score.is_finite() => score.clamp(0.0, 1.0),
        _ => {
            if passed {
                1.0
            } else {
                0.0
            }
        }
    }
}

/// Parses one command log entry.
fn command_log(value: &Value) -> Result<TerminalBenchCommandLog> {
    let object = value
        .as_object()
        .ok_or_else(|| Error::validation("terminal-bench command log entries must be objects"))?;
    Ok(TerminalBenchCommandLog {
        command: text_field(object.get("command")),
        cwd: text_field(object.get("cwd")),
        exit_code: exit_code(object.get("exit_code")),
        stdout_excerpt: excerpt(&text_field(object.get("stdout_excerpt"))),
        stderr_excerpt: excerpt(&text_field(object.get("stderr_excerpt"))),
        timeout_sec: object
            .get("timeout_sec")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        background: object
            .get("background")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    })
}

/// Reads a text field, defaulting to empty.
fn text_field(value: Option<&Value>) -> String {
    value.and_then(Value::as_str).unwrap_or_default().to_owned()
}

/// Reads an exit code, defaulting to `0` and clamping out-of-range values.
fn exit_code(value: Option<&Value>) -> i32 {
    match value.and_then(Value::as_i64) {
        None => 0,
        Some(code) => i32::try_from(code).unwrap_or(if code < 0 { i32::MIN } else { i32::MAX }),
    }
}

/// Caps a captured stream at [`COMMAND_EXCERPT_LIMIT`] characters.
///
/// Ports Python's `_excerpt`: keep the head and append `...` when characters
/// were dropped, so multi-byte text is never split.
fn excerpt(text: &str) -> String {
    if text.chars().count() <= COMMAND_EXCERPT_LIMIT {
        return text.to_owned();
    }
    let mut head: String = text.chars().take(COMMAND_EXCERPT_LIMIT - 3).collect();
    head.push_str("...");
    head
}

/// A [`crate::evolution::Evaluator`] over terminal-bench cases.
///
/// Each [`Case::input`] is an [`EvaluationCase`] document; the task id comes
/// from `metadata.task_id` and falls back to the case id. A malformed case or a
/// runner error becomes an ungraded [`CaseResult`] instead of aborting the batch.
#[derive(Clone)]
pub struct TerminalBenchDatasetEvaluator {
    evaluator: TerminalBenchEvaluator,
}

impl TerminalBenchDatasetEvaluator {
    /// Wraps the harness evaluator.
    #[must_use]
    pub fn new(evaluator: TerminalBenchEvaluator) -> Self {
        Self { evaluator }
    }

    /// Returns the wrapped evaluator.
    #[must_use]
    pub fn evaluator(&self) -> &TerminalBenchEvaluator {
        &self.evaluator
    }

    /// Runs one evolution case and maps the outcome to a case result.
    async fn evaluate_one(&self, index: usize, case: &Case) -> CaseResult {
        let evaluation_case = match EvaluationCase::from_json_value(index, &case.input) {
            Ok(evaluation_case) => evaluation_case,
            Err(error) => return ungraded_result(&case.id, case.weight, error.to_string()),
        };
        let output_dir = self
            .evaluator
            .config()
            .results_dir
            .join(sanitize_id(&case.id));
        match self
            .evaluator
            .evaluate_case(&evaluation_case, &output_dir)
            .await
        {
            Ok(outcome) => CaseResult {
                case_id: case.id.clone(),
                passed: outcome.passed,
                score: outcome.score,
                weight: case.weight,
                error: outcome.error.clone(),
                details: serde_json::json!({
                    "task_id": outcome.task_id,
                    "commands": outcome.log.len(),
                    "error": outcome.error,
                }),
            },
            Err(error) => ungraded_result(&case.id, case.weight, error.to_string()),
        }
    }
}

/// Builds an ungraded case result carrying `error`.
fn ungraded_result(case_id: &str, weight: f64, error: String) -> CaseResult {
    CaseResult {
        case_id: case_id.to_owned(),
        passed: false,
        score: None,
        weight,
        error: Some(error),
        details: Value::Null,
    }
}

#[async_trait]
impl Evaluator for TerminalBenchDatasetEvaluator {
    #[allow(clippy::unnecessary_literal_bound)] // the trait contract is `&str`
    fn name(&self) -> &str {
        "terminal_bench"
    }

    async fn evaluate(&self, _candidate: &Candidate, cases: &[Case]) -> Result<EvaluationReport> {
        let mut results = Vec::with_capacity(cases.len());
        for (index, case) in cases.iter().enumerate() {
            results.push(self.evaluate_one(index, case).await);
        }
        Ok(EvaluationReport::from_results(results))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn results_list_shape_parses_with_logs() {
        let value = json!({
            "results": [
                {"task_id": "other", "resolved": true},
                {
                    "task_id": "t1",
                    "status": "PASSED",
                    "commands": [
                        {"command": "ls", "cwd": "/app", "exit_code": 0, "stdout_excerpt": "ok"}
                    ]
                }
            ]
        });
        let outcome = parse_results(&value, "t1").expect("parses");
        assert!(outcome.passed);
        assert_eq!(outcome.score, Some(1.0));
        assert_eq!(outcome.log.len(), 1);
        assert_eq!(outcome.log[0].command, "ls");
        assert_eq!(outcome.log[0].cwd, "/app");
    }

    #[test]
    fn tasks_and_id_list_shapes_parse() {
        let tasks = json!({"tasks": {"t1": {"resolved": false, "score": 0.25}}});
        let outcome = parse_results(&tasks, "t1").expect("parses");
        assert!(!outcome.passed);
        assert_eq!(outcome.score, Some(0.25));
        let ids = json!({"resolved_ids": ["a"], "unresolved_ids": ["t1"]});
        assert!(!parse_results(&ids, "t1").expect("parses").passed);
        assert!(
            parse_results(&json!({"resolved_ids": ["t1"]}), "t1")
                .expect("parses")
                .passed
        );
        assert!(parse_results(&json!({"resolved_ids": []}), "t1").is_err());
        assert!(parse_results(&json!([]), "t1").is_err());
    }

    #[test]
    fn excerpts_are_capped() {
        let long = "x".repeat(COMMAND_EXCERPT_LIMIT + 10);
        let capped = excerpt(&long);
        assert_eq!(capped.chars().count(), COMMAND_EXCERPT_LIMIT);
        assert!(capped.ends_with("..."));
        assert_eq!(excerpt("short"), "short");
    }

    #[test]
    fn command_is_merged_and_timed() {
        let evaluator = TerminalBenchEvaluator::new(
            Arc::new(crate::runtime::command::FakeCommandRunner::new()),
            TerminalBenchConfig::default(),
        );
        let spec = evaluator.run_command("t1", Path::new("results/t1.json"));
        assert_eq!(spec.program, "terminal-bench");
        assert!(spec.merge_stderr);
        assert_eq!(spec.timeout, Some(Duration::from_secs(3600)));
        assert_eq!(spec.args[0], "run");
    }
}
