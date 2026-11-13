//! SWE-bench evaluation: the official harness command and report classification.
//!
//! Ports the portable core of Python `harness_rsi/evaluator/swebench_runtime.py`:
//!
//! - [`SweBenchEvaluator::run_command`] builds the exact
//!   `python -m swebench.harness.run_evaluation` argument vector (dataset,
//!   split, instance, predictions, `--max_workers 1`, timeout, cache level,
//!   `--clean false`, run id, namespace, instance image tag, extra args).
//! - [`SweBenchEvaluator::evaluate_case`] writes `predictions.jsonl`, runs the
//!   harness through the [`CommandRunner`] seam, locates the aggregate report,
//!   and classifies the instance ([`classification_from_report`]).
//! - [`failure_reason`] turns a classification plus the exit state into
//!   Python's exact infrastructure-failure literals.
//!
//! What stays out of scope: the Docker/WSL instance runtimes, the per-instance
//! container lifecycle, and the solver that produces a patch. Grading therefore
//! requires the official `swebench` package to be installed on the host; tests
//! inject [`crate::runtime::command::FakeCommandRunner`] and write the report
//! file themselves.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use async_trait::async_trait;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use openjiuwen_common::error::{Error, Result};

use crate::evolution::candidate::Candidate;
use crate::evolution::evaluation::{Case, CaseResult, EvaluationReport, Evaluator};
use crate::runtime::command::{CommandRunner, CommandSpec};

use super::case::EvaluationCase;
use super::sanitize_id;

/// Python's default payload for a non-zero exit: `official harness exited with code {n}`.
pub const HARNESS_EXIT_REASON_PREFIX: &str = "official harness exited with code ";

/// Reported when the aggregate report file is absent.
pub const AGGREGATE_REPORT_MISSING: &str = "official aggregate report is missing";

/// Reported when only the per-instance report is absent.
pub const PER_INSTANCE_REPORT_MISSING: &str = "official per-instance report is missing";

/// Reported when the aggregate report lists the instance under `error_ids`.
pub const REPORT_CLASSIFIED_ERROR: &str = "official report classified the instance as an error";

/// Reported when the aggregate report lists the instance under `incomplete_ids`.
pub const REPORT_CLASSIFIED_INCOMPLETE: &str =
    "official report classified the instance as incomplete";

/// Prefix of the caller-pattern infrastructure failure reason.
pub const INFRASTRUCTURE_PATTERN_PREFIX: &str = "configured infrastructure failure matched: ";

/// Python's default infrastructure failure patterns (`swebench_runtime.py`).
pub const DEFAULT_INFRASTRUCTURE_FAILURE_PATTERNS: &[&str] = &[
    "AttributeError: `np.Inf` was removed in the NumPy 2.0 release",
    "numpy.dtype size changed, may indicate binary incompatibility",
    "ImportError: libGL.so.1: cannot open shared object file",
    "ImportError: libXrender.so.1: cannot open shared object file",
];

/// Extra seconds added to the harness timeout for container setup.
const TIMEOUT_GRACE_SECS: u64 = 600;

/// Matches a `.json` report path in harness output.
///
/// Kept as an `Option` so a regex-engine failure degrades to "no path found".
static REPORT_TOKEN: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r#"[^\s"'`()\[\],;=]+\.json"#).ok());

/// The SWE-bench harness configuration.
///
/// Every field has the Python default; `run_id` is a fresh uuid so two runs
/// never share report files.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SweBenchConfig {
    /// The Python interpreter that has the `swebench` package installed.
    pub python: String,
    /// The dataset name or path.
    pub dataset: String,
    /// The dataset split.
    pub split: String,
    /// Docker image namespace.
    pub namespace: String,
    /// Docker instance image tag.
    pub instance_image_tag: String,
    /// The harness run id, used in report file names.
    pub run_id: String,
    /// The model name reported by the harness.
    pub model_name: String,
    /// Per-instance timeout handed to the harness.
    pub timeout_secs: u64,
    /// Docker cache level (`none`, `base`, `env`, `instance`).
    pub cache_level: String,
    /// Extra harness arguments appended verbatim.
    pub extra_args: Vec<String>,
    /// Where predictions and reports are written.
    pub reports_dir: PathBuf,
}

impl Default for SweBenchConfig {
    /// Returns Python's defaults with a fresh run id.
    fn default() -> Self {
        Self {
            python: "python3".to_owned(),
            dataset: "princeton-nlp/SWE-bench_Verified".to_owned(),
            split: "test".to_owned(),
            namespace: "swebench".to_owned(),
            instance_image_tag: "latest".to_owned(),
            run_id: Uuid::new_v4().to_string(),
            model_name: "openjiuwen".to_owned(),
            timeout_secs: 1800,
            cache_level: "env".to_owned(),
            extra_args: Vec::new(),
            reports_dir: PathBuf::from("swebench_reports"),
        }
    }
}

/// How the official harness classified one instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SweBenchClassification {
    /// The instance is in `resolved_ids`.
    Resolved,
    /// The instance ran and did not resolve.
    Unresolved,
    /// The harness or the report reported an infrastructure error.
    Error,
    /// The harness ran out of budget before the instance finished.
    Incomplete,
}

impl SweBenchClassification {
    /// Returns the lowercase literal recorded in outcome metadata.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Resolved => "resolved",
            Self::Unresolved => "unresolved",
            Self::Error => "error",
            Self::Incomplete => "incomplete",
        }
    }
}

/// The outcome of evaluating one SWE-bench instance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct SweBenchOutcome {
    /// The benchmark instance id.
    pub instance_id: String,
    /// The classification.
    pub classification: SweBenchClassification,
    /// `1.0` resolved, `0.0` unresolved, `None` error/incomplete.
    pub score: Option<f64>,
    /// The aggregate report path, when one was located.
    pub report_path: Option<PathBuf>,
    /// The infrastructure failure text, when there was one.
    pub error: Option<String>,
    /// Diagnostics: exit code, predictions path, stdout excerpt, …
    pub metadata: BTreeMap<String, Value>,
}

impl SweBenchOutcome {
    /// Whether the instance resolved.
    #[must_use]
    pub fn resolved(&self) -> bool {
        self.classification == SweBenchClassification::Resolved
    }
}

/// Runs the official SWE-bench harness for one case.
#[derive(Clone)]
pub struct SweBenchEvaluator {
    runner: Arc<dyn CommandRunner>,
    config: SweBenchConfig,
}

impl SweBenchEvaluator {
    /// Creates an evaluator over `runner` with `config`.
    #[must_use]
    pub fn new(runner: Arc<dyn CommandRunner>, config: SweBenchConfig) -> Self {
        Self { runner, config }
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &SweBenchConfig {
        &self.config
    }

    /// Builds the official harness command for one instance.
    ///
    /// The argument vector mirrors `_official_command` in
    /// `swebench_runtime.py`; the deadline is `timeout_secs + 600` so image
    /// setup does not race the harness's own per-instance timeout.
    #[must_use]
    pub fn run_command(&self, instance_id: &str, predictions_path: &Path) -> CommandSpec {
        let config = &self.config;
        CommandSpec::new(&config.python)
            .args(["-m", "swebench.harness.run_evaluation"])
            .arg("-d")
            .arg(&config.dataset)
            .arg("-s")
            .arg(&config.split)
            .arg("-i")
            .arg(instance_id)
            .arg("-p")
            .arg(predictions_path.display().to_string())
            .arg("--max_workers")
            .arg("1")
            .arg("-t")
            .arg(config.timeout_secs.to_string())
            .arg("--cache_level")
            .arg(&config.cache_level)
            .arg("--clean")
            .arg("false")
            .arg("-id")
            .arg(&config.run_id)
            .arg("--namespace")
            .arg(&config.namespace)
            .arg("--instance_image_tag")
            .arg(&config.instance_image_tag)
            .args(config.extra_args.iter().cloned())
            .timeout(Duration::from_secs(
                config.timeout_secs + TIMEOUT_GRACE_SECS,
            ))
    }

    /// Evaluates one case: write predictions, run the harness, classify.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the case carries no
    /// `swebench.instance_id`, [`Error::Session`] when the predictions cannot be
    /// written, and any [`CommandRunner`] error (a missing interpreter is
    /// [`Error::Config`]).
    pub async fn evaluate_case(
        &self,
        case: &EvaluationCase,
        output_dir: &Path,
    ) -> Result<SweBenchOutcome> {
        let instance_id = case
            .swebench
            .as_ref()
            .map(|swebench| swebench.instance_id.clone())
            .filter(|instance_id| !instance_id.is_empty())
            .ok_or_else(|| {
                Error::validation(format!(
                    "case `{}` is not a SWE-bench case: `swebench.instance_id` is missing",
                    case.case_id
                ))
            })?;
        let predictions_path =
            write_predictions(case, output_dir, &self.config, &instance_id).await?;
        let spec = self.run_command(&instance_id, &predictions_path);
        let command_line = spec.command_line();
        let outcome = self.runner.run(spec).await?;
        let report_path = self.resolve_report_path(&outcome.stdout).await;
        let report = read_report(report_path.as_deref()).await;
        let report_present = report.is_some();
        let classification = if outcome.success() && report_present {
            classification_from_report(&report.unwrap_or(Value::Null), &instance_id)
        } else {
            SweBenchClassification::Error
        };
        let error = failure_reason(classification, outcome.code, report_present);
        let score = match classification {
            SweBenchClassification::Resolved => Some(1.0),
            SweBenchClassification::Unresolved => Some(0.0),
            SweBenchClassification::Error | SweBenchClassification::Incomplete => None,
        };
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "run_id".to_owned(),
            Value::String(self.config.run_id.clone()),
        );
        metadata.insert(
            "model_name".to_owned(),
            Value::String(self.config.model_name.clone()),
        );
        metadata.insert(
            "dataset".to_owned(),
            Value::String(self.config.dataset.clone()),
        );
        metadata.insert("split".to_owned(), Value::String(self.config.split.clone()));
        metadata.insert(
            "exit_code".to_owned(),
            outcome
                .code
                .map_or(Value::Null, |code| Value::from(i64::from(code))),
        );
        metadata.insert("timed_out".to_owned(), Value::from(outcome.timed_out));
        metadata.insert("report_present".to_owned(), Value::from(report_present));
        metadata.insert(
            "predictions_path".to_owned(),
            Value::String(predictions_path.display().to_string()),
        );
        metadata.insert("command".to_owned(), Value::String(command_line));
        metadata.insert(
            "stdout_excerpt".to_owned(),
            Value::String(outcome.truncated_output(1000)),
        );
        Ok(SweBenchOutcome {
            instance_id,
            classification,
            score,
            report_path,
            error,
            metadata,
        })
    }

    /// Finds the aggregate report: `<reports_dir>/<model>.<run>.json` first, then
    /// any `.json` report path printed by the harness.
    ///
    /// The path may not exist (the harness can print a report path before
    /// writing it); `evaluate_case` records presence in `metadata.report_present`
    /// and reports a missing report explicitly.
    async fn resolve_report_path(&self, stdout: &str) -> Option<PathBuf> {
        let aggregate = self.config.reports_dir.join(format!(
            "{}.{}.json",
            self.config.model_name, self.config.run_id
        ));
        if tokio::fs::metadata(&aggregate).await.is_ok() {
            return Some(aggregate);
        }
        let token = report_token(stdout)?;
        let candidate = Path::new(&token);
        if candidate.is_absolute() {
            return Some(candidate.to_path_buf());
        }
        let joined = self.config.reports_dir.join(candidate);
        if tokio::fs::metadata(&joined).await.is_ok() {
            return Some(joined);
        }
        if tokio::fs::metadata(candidate).await.is_ok() {
            return Some(candidate.to_path_buf());
        }
        Some(joined)
    }
}

/// Writes `predictions.jsonl` for one instance and returns its path.
async fn write_predictions(
    case: &EvaluationCase,
    output_dir: &Path,
    config: &SweBenchConfig,
    instance_id: &str,
) -> Result<PathBuf> {
    tokio::fs::create_dir_all(output_dir)
        .await
        .map_err(|error| {
            Error::session(format!(
                "failed to create the SWE-bench output directory `{}`: {error}",
                output_dir.display()
            ))
        })?;
    let prediction = serde_json::json!({
        "instance_id": instance_id,
        "model_name_or_path": config.model_name,
        "model_patch": model_patch(case),
    });
    let mut content = serde_json::to_string(&prediction).map_err(|error| {
        Error::session(format!(
            "failed to encode the SWE-bench prediction: {error}"
        ))
    })?;
    content.push('\n');
    let path = output_dir.join("predictions.jsonl");
    tokio::fs::write(&path, content).await.map_err(|error| {
        Error::session(format!(
            "failed to write the SWE-bench predictions `{}`: {error}",
            path.display()
        ))
    })?;
    Ok(path)
}

/// Resolves the model patch: `swebench.patch`, then `input.patch`, then the
/// reference answer.
fn model_patch(case: &EvaluationCase) -> String {
    if let Some(patch) = case
        .swebench
        .as_ref()
        .and_then(|swebench| swebench.patch.as_deref())
        .filter(|patch| !patch.trim().is_empty())
    {
        return patch.to_owned();
    }
    if let Some(patch) = case
        .input
        .get("patch")
        .and_then(Value::as_str)
        .filter(|patch| !patch.trim().is_empty())
    {
        return patch.to_owned();
    }
    match &case.reference.answer {
        Value::Null => String::new(),
        Value::String(answer) => answer.clone(),
        other => other.to_string(),
    }
}

/// Reads a report document, treating an unreadable or non-object file as absent.
async fn read_report(path: Option<&Path>) -> Option<Value> {
    let path = path?;
    let text = tokio::fs::read_to_string(path).await.ok()?;
    let value: Value = serde_json::from_str(&text).ok()?;
    value.is_object().then_some(value)
}

/// Extracts the first `.json` token that names a report from harness output.
fn report_token(stdout: &str) -> Option<String> {
    let pattern = REPORT_TOKEN.as_ref()?;
    pattern
        .find_iter(stdout)
        .map(|found| found.as_str().trim().to_owned())
        .find(|token| token.to_lowercase().contains("report"))
}

/// Classifies one instance from the aggregate report.
///
/// Precedence follows Python: `resolved_ids` wins, then `error_ids`, then
/// `incomplete_ids`, otherwise the instance is unresolved.
#[must_use]
pub fn classification_from_report(report: &Value, instance_id: &str) -> SweBenchClassification {
    if id_list_contains(report, "resolved_ids", instance_id) {
        return SweBenchClassification::Resolved;
    }
    if id_list_contains(report, "error_ids", instance_id) {
        return SweBenchClassification::Error;
    }
    if id_list_contains(report, "incomplete_ids", instance_id) {
        return SweBenchClassification::Incomplete;
    }
    SweBenchClassification::Unresolved
}

/// Whether `report[key]` is a list containing `instance_id`.
fn id_list_contains(report: &Value, key: &str, instance_id: &str) -> bool {
    report
        .get(key)
        .and_then(Value::as_array)
        .is_some_and(|ids| {
            ids.iter()
                .any(|id| id.as_str().is_some_and(|id| id == instance_id))
        })
}

/// Turns a classification and exit state into Python's failure literal.
///
/// `None` means the instance was legitimately graded (resolved or unresolved).
#[must_use]
pub fn failure_reason(
    classification: SweBenchClassification,
    exit_code: Option<i32>,
    report_present: bool,
) -> Option<String> {
    if let Some(code) = exit_code {
        if code != 0 {
            return Some(format!("{HARNESS_EXIT_REASON_PREFIX}{code}"));
        }
    }
    if !report_present {
        return Some(AGGREGATE_REPORT_MISSING.to_owned());
    }
    match classification {
        SweBenchClassification::Error => Some(REPORT_CLASSIFIED_ERROR.to_owned()),
        SweBenchClassification::Incomplete => Some(REPORT_CLASSIFIED_INCOMPLETE.to_owned()),
        SweBenchClassification::Resolved | SweBenchClassification::Unresolved => None,
    }
}

/// Reports a missing per-instance report.
///
/// Python checks the per-instance `report.json` in addition to the aggregate
/// report (except for empty patches, which the harness skips on purpose). The
/// port classifies from the aggregate report alone, so this check is exposed for
/// callers that materialise the per-instance artifact.
#[must_use]
pub fn per_instance_failure_reason(instance_report_present: bool) -> Option<String> {
    if instance_report_present {
        None
    } else {
        Some(PER_INSTANCE_REPORT_MISSING.to_owned())
    }
}

/// Matches caller-supplied (and Python's default) infrastructure failure patterns
/// against the test output.
///
/// Returns `configured infrastructure failure matched: {pattern}` for the first
/// case-insensitive match, trimming the pattern first.
#[must_use]
pub fn infrastructure_failure_reason(patterns: &[String], output: &str) -> Option<String> {
    let folded = output.to_lowercase();
    let defaults = DEFAULT_INFRASTRUCTURE_FAILURE_PATTERNS
        .iter()
        .map(|pattern| (*pattern).to_owned());
    for pattern in defaults.chain(patterns.iter().cloned()) {
        let normalized = pattern.trim();
        if !normalized.is_empty() && folded.contains(&normalized.to_lowercase()) {
            return Some(format!("{INFRASTRUCTURE_PATTERN_PREFIX}{normalized}"));
        }
    }
    None
}

/// A [`crate::evolution::Evaluator`] over SWE-bench cases.
///
/// Each [`Case::input`] is an [`EvaluationCase`] document with a `swebench`
/// object. Grading needs the official harness installed on the host; tests
/// inject a scripted runner. A malformed case or a runner error becomes an
/// ungraded [`CaseResult`] instead of aborting the batch.
#[derive(Clone)]
pub struct SweBenchDatasetEvaluator {
    evaluator: SweBenchEvaluator,
}

impl SweBenchDatasetEvaluator {
    /// Wraps the harness evaluator.
    #[must_use]
    pub fn new(evaluator: SweBenchEvaluator) -> Self {
        Self { evaluator }
    }

    /// Returns the wrapped evaluator.
    #[must_use]
    pub fn evaluator(&self) -> &SweBenchEvaluator {
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
            .reports_dir
            .join(sanitize_id(&case.id));
        match self
            .evaluator
            .evaluate_case(&evaluation_case, &output_dir)
            .await
        {
            Ok(outcome) => CaseResult {
                case_id: case.id.clone(),
                passed: outcome.resolved(),
                score: outcome.score,
                weight: case.weight,
                error: outcome.error.clone(),
                details: serde_json::json!({
                    "classification": outcome.classification.as_str(),
                    "instance_id": outcome.instance_id,
                    "report_path": outcome.report_path.map(|path| path.display().to_string()),
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
impl Evaluator for SweBenchDatasetEvaluator {
    #[allow(clippy::unnecessary_literal_bound)] // the trait contract is `&str`
    fn name(&self) -> &str {
        "swebench"
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
    use super::*;

    #[test]
    fn classification_follows_python_precedence() {
        let report = serde_json::json!({
            "resolved_ids": ["a"],
            "error_ids": ["b"],
            "incomplete_ids": ["c"],
        });
        assert_eq!(
            classification_from_report(&report, "a"),
            SweBenchClassification::Resolved
        );
        assert_eq!(
            classification_from_report(&report, "b"),
            SweBenchClassification::Error
        );
        assert_eq!(
            classification_from_report(&report, "c"),
            SweBenchClassification::Incomplete
        );
        assert_eq!(
            classification_from_report(&report, "d"),
            SweBenchClassification::Unresolved
        );
    }

    #[test]
    fn failure_literals_match_python() {
        assert_eq!(
            failure_reason(SweBenchClassification::Error, Some(2), true).as_deref(),
            Some("official harness exited with code 2")
        );
        assert_eq!(
            failure_reason(SweBenchClassification::Error, Some(0), false).as_deref(),
            Some("official aggregate report is missing")
        );
        assert_eq!(
            failure_reason(SweBenchClassification::Error, Some(0), true).as_deref(),
            Some("official report classified the instance as an error")
        );
        assert_eq!(
            failure_reason(SweBenchClassification::Incomplete, None, true).as_deref(),
            Some("official report classified the instance as incomplete")
        );
        assert!(failure_reason(SweBenchClassification::Unresolved, Some(0), true).is_none());
        assert_eq!(
            per_instance_failure_reason(false).as_deref(),
            Some("official per-instance report is missing")
        );
        assert!(per_instance_failure_reason(true).is_none());
    }

    #[test]
    fn infrastructure_patterns_are_case_insensitive() {
        let reason = infrastructure_failure_reason(
            &["Disk quota exceeded".to_owned()],
            "FAILED: DISK QUOTA EXCEEDED on /testbed",
        );
        assert_eq!(
            reason.as_deref(),
            Some("configured infrastructure failure matched: Disk quota exceeded")
        );
        assert!(infrastructure_failure_reason(&[], "all good").is_none());
        assert!(infrastructure_failure_reason(
            &[],
            "AttributeError: `np.Inf` was removed in the NumPy 2.0 release"
        )
        .is_some());
    }
}
