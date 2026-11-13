//! Case execution backends, the case runner, and the judge-grade bridge.
//!
//! Python's `case_backend.py` owns one `CaseExecutionBackend` per execution
//! strategy (a single harness, a team) and `case_runner.py` grades whatever the
//! backend returns. The Rust port keeps that split and nothing else:
//!
//! - [`CaseExecutionBackend`] runs one [`EvaluationCase`] and returns
//!   [`CaseExecutionResult::response`] — a judge verdict or a plain score. The
//!   `DeepAgent` solver backend Python ships is **out of scope** (it needs the
//!   harness factory, rails and container topologies); it implements this trait
//!   elsewhere. [`CommandCaseBackend`] is the offline, subprocess-backed
//!   strategy: it runs a caller-built [`CommandSpec`] through the crate's
//!   [`CommandRunner`] seam and reads the JSON response from standard output.
//! - [`CaseRunner`] grades a response against the case's reference contract and
//!   produces a [`CaseTrace`] (status + [`CaseResult`] + execution detail).
//!
//! Failure policy, ported from `case_runner.py`:
//!
//! - a backend error, a failed execution, or an unusable verdict yields an
//!   **ungraded** [`CaseResult`] (`score: None`) so a broken judge never looks
//!   like a pass and never pollutes an average;
//! - a transient transport failure is marked so the caller can retry, using the
//!   exact marker list from `evaluator/transport.py` ([`is_transient_error`]).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};

use crate::evolution::candidate::Candidate;
use crate::evolution::evaluation::{Case, CaseResult, EvaluationReport, Evaluator};
use crate::runtime::command::{CommandRunner, CommandSpec};

use super::case::{CaseReference, EvaluationCase};
use super::scoring::{
    classify_judge_failure, judge_verdict_from_value, score_verdict, verdict_from_case,
    JudgeResult, METHOD_RESPONSE_SCORE, VERDICT_COMPLETED,
};

/// The retryable transport markers from Python `evaluator/transport.py`.
///
/// Matching is a lowercase substring test, exactly like
/// `transient_transport_error`.
pub const TRANSIENT_ERROR_MARKERS: &[&str] = &[
    "connection error",
    "connection reset",
    "connection refused",
    "connection aborted",
    "connectionerror",
    "remoteprotocolerror",
    "peer closed connection",
    "incomplete chunked read",
    "server disconnected",
    "remotedisconnected",
    "timed out",
    "timeout",
    "temporarily unavailable",
    "temporary failure in name resolution",
    "could not resolve host",
    "service unavailable",
    "rate limit",
    "too many requests",
    "http 429",
    "http 502",
    "http 503",
    "http 504",
];

/// The metadata key carrying the candidate body into the execution backend.
pub const CANDIDATE_BODY_KEY: &str = "candidate_body";

/// Whether `message` describes a retryable transport failure.
///
/// Ports `transient_transport_error`: a lowercase substring match against
/// [`TRANSIENT_ERROR_MARKERS`].
#[must_use]
pub fn is_transient_error(message: &str) -> bool {
    let lower = message.to_lowercase();
    TRANSIENT_ERROR_MARKERS
        .iter()
        .any(|marker| lower.contains(marker))
}

/// Whether execution reached the end of the case.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionStatus {
    /// The solver ran to completion and produced a gradeable response.
    Passed,
    /// The solver stopped early (non-zero exit, missing artifacts, …).
    #[default]
    Failed,
}

impl ExecutionStatus {
    /// Returns the lowercase literal recorded in traces.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Passed => "passed",
            Self::Failed => "failed",
        }
    }
}

/// What one execution backend produced for one case.
///
/// The [`Default`] (and `serde` default) value is a **failed** execution with no
/// response, so a document that omits `execution_status` can never read as a
/// success.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct CaseExecutionResult {
    /// The judge verdict or plain score the runner grades.
    pub response: Value,
    /// Whether execution reached the end of the case.
    pub execution_status: ExecutionStatus,
    /// The execution failure text; empty when nothing failed.
    pub error: String,
    /// The workspace the run used, when the backend materialises one.
    pub workspace_dir: Option<PathBuf>,
    /// Backend-specific diagnostics (`exit_code`, `command`, …).
    pub metadata: BTreeMap<String, Value>,
}

impl CaseExecutionResult {
    /// Builds a completed execution carrying `response`.
    #[must_use]
    pub fn passed(response: Value) -> Self {
        Self {
            response,
            execution_status: ExecutionStatus::Passed,
            ..Self::default()
        }
    }

    /// Builds a failed execution carrying `error`.
    #[must_use]
    pub fn failed(error: impl Into<String>) -> Self {
        Self {
            error: error.into(),
            execution_status: ExecutionStatus::Failed,
            ..Self::default()
        }
    }
}

/// One case handed to an execution backend.
#[derive(Debug, Clone, PartialEq)]
pub struct CaseExecutionRequest {
    /// The case to run.
    pub case: EvaluationCase,
    /// The per-case output directory (workspace, predictions, reports).
    pub output_dir: PathBuf,
    /// The solver session id; empty when the backend runs without a session.
    pub session_id: String,
}

/// Runs one evaluation case.
///
/// Implementations own the solver (a harness run, a subprocess, a fake) and
/// return the judge response plus an execution status; grading stays in
/// [`CaseRunner`] so every backend is scored the same way.
#[async_trait]
pub trait CaseExecutionBackend: Send + Sync {
    /// Returns the backend's short name, used in diagnostics.
    fn name(&self) -> &str;

    /// Executes `request`.
    ///
    /// # Errors
    ///
    /// Returns an error when the case cannot be executed at all (for example a
    /// missing executable). A solver that ran and failed is a
    /// [`ExecutionStatus::Failed`] result, not an error.
    async fn execute(&self, request: CaseExecutionRequest) -> Result<CaseExecutionResult>;

    /// Releases case-scoped resources after the artifacts are written.
    ///
    /// # Errors
    ///
    /// Returns an error when cleanup cannot complete; the default does nothing.
    async fn cleanup(&self, _case_id: &str) -> Result<()> {
        Ok(())
    }
}

/// The graded trace of one case.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct CaseTrace {
    /// The case id.
    pub case_id: String,
    /// `passed`, `failed`, or `error`.
    pub status: String,
    /// The graded (or ungraded) result for this case.
    pub result: CaseResult,
    /// The backend result, when the backend ran at all.
    pub execution: Option<CaseExecutionResult>,
    /// Caller/materialised artifacts keyed by name.
    pub artifacts: BTreeMap<String, Value>,
}

impl Default for CaseTrace {
    /// Returns an empty trace with a failed, ungraded result.
    fn default() -> Self {
        Self {
            case_id: String::new(),
            status: "error".to_owned(),
            result: CaseResult::default(),
            execution: None,
            artifacts: BTreeMap::new(),
        }
    }
}

impl CaseTrace {
    /// Whether the failure looks like retryable infrastructure, not a verdict.
    ///
    /// Checks the trace's own error first and then the execution's error, using
    /// [`is_transient_error`].
    #[must_use]
    pub fn is_infrastructure_error(&self) -> bool {
        let own = self.result.error.as_deref().is_some_and(is_transient_error);
        let execution = self
            .execution
            .as_ref()
            .is_some_and(|execution| is_transient_error(&execution.error));
        own || execution
    }
}

/// Executes a case and grades the response.
///
/// Holds the execution backend plus the contract knobs that are not part of a
/// case's reference (the pass threshold and binary mode).
#[derive(Clone)]
pub struct CaseRunner {
    backend: Arc<dyn CaseExecutionBackend>,
    threshold: f64,
    binary: bool,
}

impl CaseRunner {
    /// Creates a runner over `backend` with Python's `0.8` threshold and
    /// fractional scores.
    #[must_use]
    pub fn new(backend: Arc<dyn CaseExecutionBackend>) -> Self {
        Self {
            backend,
            threshold: super::scoring::DEFAULT_SUCCESS_THRESHOLD,
            binary: false,
        }
    }

    /// Sets the score at or above which a case passes.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Collapses the reported score to `1.0`/`0.0` while keeping the raw score
    /// in the result metadata.
    #[must_use]
    pub fn with_binary(mut self, binary: bool) -> Self {
        self.binary = binary;
        self
    }

    /// Returns the pass threshold.
    #[must_use]
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Returns whether binary scoring is on.
    #[must_use]
    pub fn binary(&self) -> bool {
        self.binary
    }

    /// Returns the backend name.
    #[must_use]
    pub fn backend_name(&self) -> &str {
        self.backend.name()
    }

    /// Runs and grades one case.
    ///
    /// A backend error becomes a trace with `status: "error"`, an ungraded
    /// result and no execution; a failed execution or an unusable verdict
    /// becomes an ungraded `failed` result. Neither is returned as an `Err`,
    /// because the caller decides whether one case should abort the suite.
    ///
    /// The result weight is `1.0`: an [`EvaluationCase`] carries no per-case
    /// weight, and [`CandidateCaseRunner`] copies the evolution case weight over
    /// it.
    ///
    /// # Errors
    ///
    /// Returns an error only when the case document itself cannot be used
    /// (propagated from the backend).
    pub async fn run(&self, request: CaseExecutionRequest) -> Result<CaseTrace> {
        let case_id = request.case.case_id.clone();
        let reference = request.case.reference.clone();
        let execution = match self.backend.execute(request).await {
            Ok(execution) => execution,
            Err(error) => return Ok(error_trace(&case_id, &error.to_string())),
        };
        let (result, status) = self.grade(&case_id, &reference, &execution);
        Ok(CaseTrace {
            case_id,
            status,
            result,
            execution: Some(execution),
            artifacts: BTreeMap::new(),
        })
    }

    /// Grades one execution result.
    fn grade(
        &self,
        case_id: &str,
        reference: &CaseReference,
        execution: &CaseExecutionResult,
    ) -> (CaseResult, String) {
        if execution.execution_status == ExecutionStatus::Failed {
            let error = if execution.error.is_empty() {
                "case execution failed".to_owned()
            } else {
                execution.error.clone()
            };
            return (
                ungraded(case_id, error, Value::Null),
                ExecutionStatus::Failed.as_str().to_owned(),
            );
        }
        match grade_response(reference, self.threshold, self.binary, &execution.response) {
            Ok(judge) => (
                CaseResult {
                    case_id: case_id.to_owned(),
                    passed: judge.passed,
                    score: Some(judge.score),
                    weight: 1.0,
                    error: None,
                    details: judge_details(&judge),
                },
                status_of(judge.passed),
            ),
            Err(error) => {
                let classification = classify_judge_failure(&execution.response.to_string());
                let details = serde_json::json!({
                    "classification": classification.as_str(),
                    "detail": error.to_string(),
                    "response": execution.response,
                });
                (
                    ungraded(case_id, classification.as_str(), details),
                    "failed".to_owned(),
                )
            }
        }
    }
}

/// Builds a trace for a backend error.
fn error_trace(case_id: &str, message: &str) -> CaseTrace {
    CaseTrace {
        case_id: case_id.to_owned(),
        status: "error".to_owned(),
        result: ungraded(case_id, message, Value::Null),
        execution: None,
        artifacts: BTreeMap::new(),
    }
}

/// Builds an ungraded case result carrying `error`.
fn ungraded(case_id: &str, error: impl Into<String>, details: Value) -> CaseResult {
    CaseResult {
        case_id: case_id.to_owned(),
        passed: false,
        score: None,
        weight: 1.0,
        error: Some(error.into()),
        details,
    }
}

/// Maps a pass/fail decision to a trace status.
fn status_of(passed: bool) -> String {
    if passed {
        "passed".to_owned()
    } else {
        "failed".to_owned()
    }
}

/// Renders a judge result as case-result details.
fn judge_details(judge: &JudgeResult) -> Value {
    let mut object = serde_json::Map::new();
    object.insert("method".to_owned(), Value::String(judge.method.clone()));
    for (key, value) in &judge.metadata {
        object.insert(key.clone(), value.clone());
    }
    Value::Object(object)
}

/// Grades one response document against the reference contract.
///
/// Accepts a full judge verdict (`{status, behaviors, forbidden_hits}`) or a
/// plain `{"score": f64, "passed": bool}` response.
fn grade_response(
    reference: &CaseReference,
    threshold: f64,
    binary: bool,
    response: &Value,
) -> Result<JudgeResult> {
    let contract = verdict_from_case(reference, threshold, binary);
    let Some(object) = response.as_object() else {
        return Err(Error::validation(
            "judge response must be a JSON object with a verdict or a score",
        ));
    };
    if let Some(status) = object.get("status").and_then(Value::as_str) {
        if status != VERDICT_COMPLETED {
            return Err(Error::validation(format!(
                "judge status must be `{VERDICT_COMPLETED}`, got `{status}`"
            )));
        }
    }
    let is_verdict = ["behaviors", "forbidden_hits", "overall_reason"]
        .iter()
        .any(|key| object.contains_key(*key));
    if is_verdict {
        let verdict = judge_verdict_from_value(response)?;
        return Ok(score_verdict(&contract, &verdict));
    }
    let Some(score) = object.get("score").and_then(Value::as_f64) else {
        return Err(Error::validation(
            "judge response must carry `behaviors` or a numeric `score`",
        ));
    };
    if !score.is_finite() {
        return Err(Error::validation("judge response score must be finite"));
    }
    let raw = score.clamp(0.0, 1.0);
    let passed = object
        .get("passed")
        .and_then(Value::as_bool)
        .unwrap_or(raw >= contract.success_threshold);
    let mut metadata = BTreeMap::new();
    metadata.insert("raw_score".to_owned(), Value::from(raw));
    metadata.insert("binary".to_owned(), Value::from(binary));
    metadata.insert(
        "success_threshold".to_owned(),
        Value::from(contract.success_threshold),
    );
    Ok(JudgeResult {
        method: METHOD_RESPONSE_SCORE.to_owned(),
        score: if binary {
            if passed {
                1.0
            } else {
                0.0
            }
        } else {
            raw
        },
        passed,
        reason: object
            .get("reason")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        metadata,
    })
}

/// Builds the [`CommandSpec`] that runs one case.
pub trait CaseCommandBuilder: Send + Sync {
    /// Whether the builder is applicable to `case`.
    ///
    /// # Errors
    ///
    /// Returns an error when the case lacks the fields the command needs.
    fn build(&self, case: &EvaluationCase) -> Result<CommandSpec>;
}

/// A [`CaseExecutionBackend`] that runs a command and reads its JSON output.
///
/// The command is caller-built ([`CaseCommandBuilder`]), so the same backend
/// drives any offline solver shim. Standard output must be the response
/// document, either bare or inside a ```` ```json ```` fence. A non-zero exit is
/// an [`ExecutionStatus::Failed`] execution carrying standard error, never an
/// `Err`: an exit code is data.
///
/// When [`CommandCaseBackend::field`] is non-empty and the parsed document is an
/// object carrying that member, the member becomes the response — the usual
/// `{"response": {...verdict...}}` envelope.
#[derive(Clone)]
pub struct CommandCaseBackend {
    runner: Arc<dyn CommandRunner>,
    builder: Arc<dyn CaseCommandBuilder>,
    field: String,
}

impl CommandCaseBackend {
    /// Creates a backend reading the `response` member of the parsed document
    /// when present.
    #[must_use]
    pub fn new(runner: Arc<dyn CommandRunner>, builder: Arc<dyn CaseCommandBuilder>) -> Self {
        Self {
            runner,
            builder,
            field: "response".to_owned(),
        }
    }

    /// Overrides the response member name; an empty name disables unwrapping.
    #[must_use]
    pub fn with_field(mut self, field: impl Into<String>) -> Self {
        self.field = field.into();
        self
    }

    /// Returns the response member name.
    #[must_use]
    pub fn field(&self) -> &str {
        &self.field
    }
}

#[async_trait]
impl CaseExecutionBackend for CommandCaseBackend {
    #[allow(clippy::unnecessary_literal_bound)] // the trait contract is `&str`
    fn name(&self) -> &str {
        "command"
    }

    async fn execute(&self, request: CaseExecutionRequest) -> Result<CaseExecutionResult> {
        let spec = self.builder.build(&request.case)?;
        let command_line = spec.command_line();
        let outcome = self.runner.run(spec).await?;
        let mut metadata = BTreeMap::new();
        metadata.insert("command".to_owned(), Value::String(command_line));
        metadata.insert("exit_code".to_owned(), Value::from(outcome.code));
        metadata.insert("timed_out".to_owned(), Value::from(outcome.timed_out));
        if !outcome.success() {
            let error = if outcome.stderr.is_empty() {
                outcome.truncated_output(1000)
            } else {
                outcome.stderr.clone()
            };
            return Ok(CaseExecutionResult {
                response: Value::Null,
                execution_status: ExecutionStatus::Failed,
                error,
                workspace_dir: None,
                metadata,
            });
        }
        let response = parse_response(&outcome.stdout).ok_or_else(|| {
            Error::tool(format!(
                "case command produced no JSON response: {}",
                outcome.truncated_output(300)
            ))
        })?;
        let response = self.select_response(response);
        Ok(CaseExecutionResult {
            response,
            execution_status: ExecutionStatus::Passed,
            error: String::new(),
            workspace_dir: None,
            metadata,
        })
    }
}

impl CommandCaseBackend {
    /// Unwraps the configured response member when the document is an envelope.
    fn select_response(&self, response: Value) -> Value {
        if self.field.is_empty() {
            return response;
        }
        response
            .as_object()
            .and_then(|object| object.get(&self.field))
            .cloned()
            .unwrap_or(response)
    }
}

/// Parses a response document from command output: bare JSON first, then a
/// ```` ```json ```` fence.
fn parse_response(stdout: &str) -> Option<Value> {
    let trimmed = stdout.trim();
    if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
        return Some(value);
    }
    let start = trimmed.find("```json")? + "```json".len();
    let rest = &trimmed[start..];
    let end = rest.find("```")?;
    serde_json::from_str::<Value>(rest[..end].trim()).ok()
}

/// A [`CaseRunner`] that grades [`crate::evolution::Case`] inputs.
///
/// The mapping is explicit: every evolution case's [`Case::input`] **is** an
/// [`EvaluationCase`] document (`EvaluationCase::from_json_value(index, input)`),
/// the candidate body is injected into the case metadata under
/// [`CANDIDATE_BODY_KEY`], and the run uses an empty session id. Each input case
/// yields exactly one [`CaseResult`] carrying the evolution case weight; a
/// malformed input becomes an ungraded result instead of aborting the batch.
#[derive(Clone)]
pub struct CandidateCaseRunner {
    inner: CaseRunner,
    output_dir: PathBuf,
}

impl CandidateCaseRunner {
    /// Wraps `inner`, writing case outputs under the system temp directory.
    #[must_use]
    pub fn new(inner: CaseRunner) -> Self {
        Self {
            inner,
            output_dir: std::env::temp_dir().join("openjiuwen-rsi-evaluator"),
        }
    }

    /// Sets the directory each case writes into.
    #[must_use]
    pub fn with_output_dir(mut self, output_dir: impl Into<PathBuf>) -> Self {
        self.output_dir = output_dir.into();
        self
    }

    /// Returns the wrapped runner.
    #[must_use]
    pub fn runner(&self) -> &CaseRunner {
        &self.inner
    }

    /// Returns the per-case output directory.
    #[must_use]
    pub fn output_dir(&self) -> &std::path::Path {
        &self.output_dir
    }

    /// Maps one evolution case, runs it, and copies the evolution weight.
    async fn evaluate_case(&self, index: usize, case: &Case, body: &str) -> CaseResult {
        let mut evaluation_case = match EvaluationCase::from_json_value(index, &case.input) {
            Ok(evaluation_case) => evaluation_case,
            Err(error) => {
                return CaseResult {
                    case_id: case.id.clone(),
                    passed: false,
                    score: None,
                    weight: case.weight,
                    error: Some(error.to_string()),
                    details: Value::Null,
                }
            }
        };
        evaluation_case.metadata.insert(
            CANDIDATE_BODY_KEY.to_owned(),
            Value::String(body.to_owned()),
        );
        let request = CaseExecutionRequest {
            case: evaluation_case,
            output_dir: self.output_dir.join(&case.id),
            session_id: String::new(),
        };
        match self.inner.run(request).await {
            Ok(trace) => CaseResult {
                case_id: case.id.clone(),
                weight: case.weight,
                ..trace.result
            },
            Err(error) => CaseResult {
                case_id: case.id.clone(),
                passed: false,
                score: None,
                weight: case.weight,
                error: Some(error.to_string()),
                details: Value::Null,
            },
        }
    }
}

#[async_trait]
impl Evaluator for CandidateCaseRunner {
    #[allow(clippy::unnecessary_literal_bound)] // the trait contract is `&str`
    fn name(&self) -> &str {
        "case_runner"
    }

    async fn evaluate(&self, candidate: &Candidate, cases: &[Case]) -> Result<EvaluationReport> {
        let mut results = Vec::with_capacity(cases.len());
        for (index, case) in cases.iter().enumerate() {
            results.push(self.evaluate_case(index, case, &candidate.body).await);
        }
        Ok(EvaluationReport::from_results(results))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transient_markers_match_case_insensitively() {
        assert!(is_transient_error("HTTP 503 Service Unavailable"));
        assert!(is_transient_error("Read failed: Incomplete chunked read"));
        assert!(is_transient_error("Connection reset by peer"));
        assert!(!is_transient_error("judge output must contain behaviors"));
    }

    #[test]
    fn responses_parse_bare_or_fenced() {
        assert_eq!(
            parse_response("{\"score\": 1.0}"),
            Some(serde_json::json!({"score": 1.0}))
        );
        assert_eq!(
            parse_response("```json\n{\"score\": 0.5}\n```"),
            Some(serde_json::json!({"score": 0.5}))
        );
        assert!(parse_response("no json here").is_none());
    }

    #[test]
    fn plain_scores_grade_and_binary_collapses() {
        let reference = CaseReference::default();
        let response = serde_json::json!({"score": 0.9, "reason": "fine"});
        let graded = grade_response(&reference, 0.8, false, &response).expect("grades");
        assert!(graded.passed);
        assert!((graded.score - 0.9).abs() < 1e-9);
        assert_eq!(graded.reason, "fine");
        let binary = grade_response(&reference, 0.8, true, &response).expect("grades");
        assert!((binary.score - 1.0).abs() < f64::EPSILON);
        assert_eq!(
            binary.metadata.get("raw_score"),
            Some(&serde_json::json!(0.9))
        );
    }

    #[test]
    fn unusable_responses_are_rejected() {
        let reference = CaseReference::default();
        assert!(grade_response(&reference, 0.8, false, &serde_json::json!(1.0)).is_err());
        assert!(grade_response(&reference, 0.8, false, &serde_json::json!({})).is_err());
        assert!(grade_response(
            &reference,
            0.8,
            false,
            &serde_json::json!({"status": "unavailable"})
        )
        .is_err());
    }

    #[test]
    fn infrastructure_error_checks_both_errors() {
        let mut trace = CaseTrace::default();
        trace.result.error = Some("request timed out".to_owned());
        assert!(trace.is_infrastructure_error());
        trace.result.error = Some("invalid_schema".to_owned());
        assert!(!trace.is_infrastructure_error());
        trace.execution = Some(CaseExecutionResult::failed("rate limit exceeded"));
        assert!(trace.is_infrastructure_error());
    }
}
