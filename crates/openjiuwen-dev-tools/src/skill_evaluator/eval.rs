//! Evaluation harness: case loading, agent execution, run reports, and single-query mode.
//!
//! The executable shell around the report models: it loads cases, runs each
//! prompt through an [`EvalAgent`], checks the expectations with the shared
//! [`Assertion`] engine, and emits artifacts other tools can consume.
//!
//! - **Case loading** ([`load_cases_json`], [`load_cases_file`]) accepts a JSON
//!   array, an object with a `cases` array, a single case object, or JSONL; every
//!   loaded case is validated.
//! - **Agent seam** ([`EvalAgent`]): [`ModelAgent`] calls a `Model` once per
//!   prompt, [`CaseRunnerAgent`] adapts a `CaseRunner` plus skill.
//! - **Harness** ([`EvalHarness`]) runs cases in input order and produces an
//!   [`EvalRun`]: per-case statuses plus the shared `CaseSummary`.
//! - **Reports** ([`EvalRun`]) serialize to run-artifact JSON, whose wire name
//!   for the in-memory `id` is `case_id`, and to JSONL, and can be written out.
//! - **Single-query mode** ([`QueryRequest`], [`EvalHarness::run_query`]) mirrors
//!   the Python `run_eval_query.py` script: one prompt, one answer file, one
//!   JSON report.
//!
//! # Determinism rules
//!
//! Cases run sequentially in input order and a failed agent call aborts the run.
//! Case identity, the query timestamp, and every path are caller-supplied, so
//! nothing reads the clock or the environment, and no path is resolved against
//! the working directory (a relative path is handed to the OS as given).
//! Expected-output comparison normalizes whitespace (trim plus collapse) and
//! evidence is a deterministic prefix excerpt, so identical inputs always
//! produce byte-identical artifacts.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_foundation::prompt::BaseMessage;
use serde::{Deserialize, Serialize};

use crate::skill_creator::SkillDefinition;

use super::assertion::{Assertion, AssertionOutcome};
use super::case::{CaseStatus, EvalCase};
use super::report::{status_from, summarize, CaseResult, CaseSummary};
use super::runner::CaseRunner;

/// The default system prompt of [`ModelAgent`].
const DEFAULT_SYSTEM_PROMPT: &str =
    "You are an intelligent assistant. Answer the user's request directly.";

/// Characters kept in a synthesized evidence excerpt, mirroring `assertion.rs`.
const EVIDENCE_LIMIT: usize = 200;

/// Prefix of the generated answer filename.
const RESULT_FILE_PREFIX: &str = "skill_test_result_";

/// Loads evaluation cases from a JSON document held in memory.
///
/// Three shapes are accepted: a JSON array of cases, an object
/// `{"cases": [...]}`, or a single case object. Elements deserialize into
/// [`EvalCase`] (unknown fields are ignored) and are then checked with
/// [`EvalCase::validate`], so a malformed case fails with its own message. An
/// empty array is a valid, empty case list.
///
/// # Errors
///
/// [`Error::validation`] for a blank or malformed document, a document that is
/// neither an array nor an object, a non-array `cases` member, a case that does
/// not deserialize, and a case that fails validation.
pub fn load_cases_json(text: &str) -> Result<Vec<EvalCase>> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(Error::validation(
            "evaluation case document must not be blank",
        ));
    }
    let document: serde_json::Value = serde_json::from_str(trimmed).map_err(|error| {
        Error::validation(format!("evaluation cases are not valid JSON: {error}"))
    })?;
    let items = match document {
        serde_json::Value::Array(items) => items,
        serde_json::Value::Object(mut object) => match object.remove("cases") {
            Some(serde_json::Value::Array(items)) => items,
            Some(other) => {
                return Err(Error::validation(format!(
                    "evaluation case document: `cases` must be a JSON array, found {}",
                    json_kind(&other)
                )))
            }
            None => vec![serde_json::Value::Object(object)],
        },
        other => {
            return Err(Error::validation(format!(
                "evaluation case document must be a JSON array, an object with a `cases` array, \
                 or a single case object, found {}",
                json_kind(&other)
            )))
        }
    };
    let mut cases = Vec::with_capacity(items.len());
    for (index, item) in items.into_iter().enumerate() {
        let case: EvalCase = serde_json::from_value(item).map_err(|error| {
            Error::validation(format!(
                "evaluation case at index {index} is malformed: {error}"
            ))
        })?;
        case.validate()?;
        cases.push(case);
    }
    Ok(cases)
}

/// Loads evaluation cases from a file.
///
/// A `jsonl` extension (case-insensitive) selects one case per non-blank,
/// trimmed line. Otherwise the content decides: a document starting with `{` or
/// `[` is parsed as JSON and, if that fails, retried as JSONL so a
/// one-object-per-line file with a misleading `.json` extension still loads; a
/// document starting with anything else is parsed as JSONL directly.
///
/// # Errors
///
/// [`Error::validation`] naming the path when the file cannot be read or is
/// blank, plus the parsing/validation error of a bad line (carrying its 1-based
/// line number) or of a bad JSON document.
pub async fn load_cases_file(path: impl AsRef<Path>) -> Result<Vec<EvalCase>> {
    let path = path.as_ref();
    let text = tokio::fs::read_to_string(path).await.map_err(|error| {
        Error::validation(format!(
            "reading evaluation cases from `{}`: {error}",
            path.display()
        ))
    })?;
    if text.trim().is_empty() {
        return Err(Error::validation(format!(
            "evaluation case file `{}` is blank",
            path.display()
        )));
    }
    let is_jsonl = path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("jsonl"));
    let head = text.trim_start();
    if is_jsonl || !(head.starts_with('{') || head.starts_with('[')) {
        return parse_cases_jsonl(&text);
    }
    match load_cases_json(&text) {
        Ok(cases) => Ok(cases),
        // A JSONL file named `.json` starts with `{`, so retry line-wise before
        // reporting the JSON failure.
        Err(json_error) => parse_cases_jsonl(&text).or(Err(json_error)),
    }
}

/// Parses one case per non-blank line, reporting the 1-based line number.
fn parse_cases_jsonl(text: &str) -> Result<Vec<EvalCase>> {
    let mut cases = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let line_number = index + 1;
        let case: EvalCase = serde_json::from_str(trimmed).map_err(|error| {
            Error::validation(format!("line {line_number} is not a case object: {error}"))
        })?;
        case.validate().map_err(|error| {
            error.with_context(format!("evaluation case on line {line_number}"))
        })?;
        cases.push(case);
    }
    Ok(cases)
}

/// Names the JSON kind of a value for diagnostics.
fn json_kind(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "a boolean",
        serde_json::Value::Number(_) => "a number",
        serde_json::Value::String(_) => "a string",
        serde_json::Value::Array(_) => "an array",
        serde_json::Value::Object(_) => "an object",
    }
}

/// A system under test: one prompt in, one answer out.
///
/// The trait keeps the harness independent of how answers are produced: a model
/// call ([`ModelAgent`]), a case runner ([`CaseRunnerAgent`]), or a test double.
#[async_trait]
pub trait EvalAgent: Send + Sync {
    /// Answers `prompt` and returns the answer text.
    ///
    /// # Errors
    ///
    /// [`Error::llm`](openjiuwen_common::error::Error::llm) for provider
    /// failures, [`Error::validation`](openjiuwen_common::error::Error::validation)
    /// for unusable input.
    async fn run(&self, prompt: &str) -> Result<String>;
}

/// An [`EvalAgent`] that asks a [`Model`] for every prompt.
///
/// Each call sends exactly one system message (the configured system prompt) and
/// one user message (the prompt) and returns the first choice's text.
pub struct ModelAgent {
    /// The model consulted for every prompt.
    model: Arc<dyn Model>,
    /// The system message prefixing every conversation.
    system_prompt: String,
}

impl ModelAgent {
    /// Creates an agent over `model` with the default system prompt.
    #[must_use]
    pub fn new(model: Arc<dyn Model>) -> Self {
        Self {
            model,
            system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
        }
    }

    /// Replaces the system prompt.
    #[must_use]
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }
}

#[async_trait]
impl EvalAgent for ModelAgent {
    async fn run(&self, prompt: &str) -> Result<String> {
        let messages = [
            BaseMessage::system(self.system_prompt.clone()),
            BaseMessage::user(prompt.to_string()),
        ];
        let request = ModelRequestConfig::new(self.model.model_name());
        let response = self.model.invoke(&messages, &request).await?;
        response
            .message()
            .map(|message| message.text().into_owned())
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| {
                Error::llm(format!(
                    "model returned no answer text for prompt {:?} ({} characters)",
                    excerpt(prompt),
                    prompt.chars().count()
                ))
            })
    }
}

/// An [`EvalAgent`] backed by a [`CaseRunner`] and the skill under test.
///
/// The prompt becomes a throwaway [`EvalCase`] with the id and name `query`, so
/// scripted runners can key an answer on that id while skill-aware runners
/// receive the full skill document.
pub struct CaseRunnerAgent {
    /// The runner that produces the answer.
    runner: Arc<dyn CaseRunner>,
    /// The skill handed to the runner.
    skill: SkillDefinition,
}

impl CaseRunnerAgent {
    /// Creates an agent over `runner` for `skill`.
    #[must_use]
    pub fn new(runner: Arc<dyn CaseRunner>, skill: SkillDefinition) -> Self {
        Self { runner, skill }
    }

    /// The wrapped runner.
    #[must_use]
    pub fn runner(&self) -> &Arc<dyn CaseRunner> {
        &self.runner
    }

    /// The skill handed to the runner.
    #[must_use]
    pub fn skill(&self) -> &SkillDefinition {
        &self.skill
    }
}

#[async_trait]
impl EvalAgent for CaseRunnerAgent {
    async fn run(&self, prompt: &str) -> Result<String> {
        let case = EvalCase::new("query", "query", prompt);
        self.runner.run(&case, &self.skill).await
    }
}

/// Runs a batch of evaluation cases through one [`EvalAgent`].
///
/// The harness holds the agent, and optionally the skill under test, and turns a
/// case slice into an [`EvalRun`]; it never talks to a model itself.
pub struct EvalHarness {
    /// The system under test.
    agent: Arc<dyn EvalAgent>,
    /// The skill under test, validated before a run when present.
    skill: Option<SkillDefinition>,
}

impl EvalHarness {
    /// Creates a harness over `agent`.
    #[must_use]
    pub fn new(agent: Arc<dyn EvalAgent>) -> Self {
        Self { agent, skill: None }
    }

    /// Attaches the skill under test, validated at the start of every run.
    #[must_use]
    pub fn with_skill(mut self, skill: SkillDefinition) -> Self {
        self.skill = Some(skill);
        self
    }

    /// The system under test.
    #[must_use]
    pub fn agent(&self) -> &Arc<dyn EvalAgent> {
        &self.agent
    }

    /// The attached skill, when one was configured.
    #[must_use]
    pub fn skill(&self) -> Option<&SkillDefinition> {
        self.skill.as_ref()
    }

    /// Runs every case in input order and summarizes the outcomes.
    ///
    /// For each case the harness checks that id, name, and prompt are non-blank
    /// (without requiring expectations, so an expected-output-only case stays
    /// runnable), builds the checks, asks the agent once, and records the
    /// outcomes: every `expectations` entry becomes an [`Assertion`] check, a
    /// case with none falls back to one normalized `equals_normalized:<expected>`
    /// check on `expected_output`, and a case with neither is a validation error
    /// naming the case id. Status and summary come from the shared `status_from`
    /// and `summarize` helpers, so a harness run cannot drift from the evaluator.
    ///
    /// # Errors
    ///
    /// [`Error::validation`](openjiuwen_common::error::Error::validation) when the
    /// attached skill, a case, or an expectation is invalid; the agent's own
    /// error aborts the run.
    pub async fn run(&self, cases: &[EvalCase]) -> Result<EvalRun> {
        if let Some(skill) = &self.skill {
            skill.validate()?;
        }
        let mut results = Vec::with_capacity(cases.len());
        for case in cases {
            validate_case_inputs(case)?;
            let checks = build_checks(case)?;
            let answer = self.agent.run(&case.prompt).await?;
            let mut outcomes = Vec::with_capacity(checks.len());
            for check in &checks {
                outcomes.push(check.evaluate(&answer)?);
            }
            let status = status_from(&outcomes);
            results.push(CaseResult {
                id: case.id.clone(),
                name: case.name.clone(),
                status,
                assertions: outcomes,
            });
        }
        let summary = summarize(&results);
        Ok(EvalRun {
            cases: results,
            summary,
        })
    }

    /// Runs one query in single-query mode.
    ///
    /// The output file is chosen by [`resolve_output_file`]; its directory is
    /// created and becomes the default `files_base_dir`. The agent is called
    /// exactly once with the query prefixed by the context block, the answer text
    /// is written to the answer file, and a JSON report is written beside it
    /// (same file name, `.json` extension) holding `prompt`, `output_file`,
    /// `report_file`, `files_base_dir`, `timestamp`, `max_iterations`, and
    /// `answer`.
    ///
    /// # Errors
    ///
    /// [`Error::validation`](openjiuwen_common::error::Error::validation) for a
    /// blank prompt or timestamp, the agent's own error, and
    /// [`Error::external`](openjiuwen_common::error::Error::external) naming the
    /// path when a directory or file cannot be written.
    pub async fn run_query(&self, request: &QueryRequest) -> Result<QueryOutcome> {
        if request.prompt.trim().is_empty() {
            return Err(Error::validation("query prompt must not be blank"));
        }
        if request.timestamp.trim().is_empty() {
            return Err(Error::validation("query timestamp must not be blank"));
        }
        let output_file = resolve_output_file(&request.output_path, &request.timestamp);
        let output_dir = answer_directory(&output_file).await?;
        let files_base_dir = request
            .files_base_dir
            .clone()
            .unwrap_or_else(|| output_dir.clone());
        let prompt = build_query_prompt(
            &request.prompt,
            &files_base_dir,
            &output_dir,
            request.max_iterations,
        );
        let answer = self.agent.run(&prompt).await?;
        write_artifact(&output_file, &answer).await?;

        let report_file = output_file.with_extension("json");
        let report = serde_json::json!({
            "prompt": prompt,
            "output_file": output_file.to_string_lossy(),
            "report_file": report_file.to_string_lossy(),
            "files_base_dir": files_base_dir.to_string_lossy(),
            "timestamp": request.timestamp,
            "max_iterations": request.max_iterations,
            "answer": &answer,
        });
        let text = serde_json::to_string_pretty(&report).map_err(Error::external)?;
        write_artifact(&report_file, &text).await?;

        Ok(QueryOutcome {
            output_file,
            report_file,
            answer,
        })
    }
}

/// The artifact of one harness run: per-case results plus their summary.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EvalRun {
    /// One entry per executed case, in execution order.
    pub cases: Vec<CaseResult>,
    /// Aggregate counts over [`cases`](Self::cases).
    pub summary: CaseSummary,
}

impl EvalRun {
    /// Serializes the run into `{"cases": [...], "summary": {...}}`.
    ///
    /// Each case object uses the run-artifact wire names: `case_id` (the in-memory
    /// [`CaseResult::id`]), `name`, `status` (the `snake_case` status string),
    /// `passed`, `reason` (empty when the case passed, otherwise the failed
    /// assertion texts joined with `"; "`), and `assertions`.
    ///
    /// # Errors
    ///
    /// [`Error::external`](openjiuwen_common::error::Error::external) when a
    /// value cannot be serialized.
    pub fn to_json(&self) -> Result<serde_json::Value> {
        let mut cases = Vec::with_capacity(self.cases.len());
        for case in &self.cases {
            cases.push(case_json(case)?);
        }
        let summary = serde_json::to_value(&self.summary).map_err(Error::external)?;
        Ok(serde_json::json!({ "cases": cases, "summary": summary }))
    }

    /// Serializes the run into a compact JSON string.
    ///
    /// # Errors
    ///
    /// As [`to_json`](Self::to_json).
    pub fn to_json_string(&self) -> Result<String> {
        serde_json::to_string(&self.to_json()?).map_err(Error::external)
    }

    /// Serializes the run as JSONL: one compact case object per line.
    ///
    /// Every line, including the last, ends with `\n`, so an empty run yields the
    /// empty string and an `n`-case run yields exactly `n` newlines.
    ///
    /// # Errors
    ///
    /// As [`to_json`](Self::to_json).
    pub fn to_jsonl(&self) -> Result<String> {
        let mut text = String::new();
        for case in &self.cases {
            text.push_str(&serde_json::to_string(&case_json(case)?).map_err(Error::external)?);
            text.push('\n');
        }
        Ok(text)
    }

    /// Writes the pretty JSON artifact to `path`, creating missing parents.
    ///
    /// # Errors
    ///
    /// As [`to_json`](Self::to_json), plus
    /// [`Error::external`](openjiuwen_common::error::Error::external) naming the
    /// path when a directory or file cannot be written.
    pub async fn write_json(&self, path: impl AsRef<Path>) -> Result<()> {
        let text = serde_json::to_string_pretty(&self.to_json()?).map_err(Error::external)?;
        write_artifact(path.as_ref(), &text).await
    }

    /// Writes the [`to_jsonl`](Self::to_jsonl) artifact to `path`.
    ///
    /// # Errors
    ///
    /// As [`write_json`](Self::write_json).
    pub async fn write_jsonl(&self, path: impl AsRef<Path>) -> Result<()> {
        let text = self.to_jsonl()?;
        write_artifact(path.as_ref(), &text).await
    }
}

/// One check applied to an answer: a parsed assertion or the expected output.
enum Check {
    /// A parsed expectation from `EvalCase::expectations`.
    Parsed(Assertion),
    /// The normalized expected output, used when no expectation was given.
    ExpectedOutput(String),
}

impl Check {
    /// Applies the check to `answer`, producing one assertion outcome.
    fn evaluate(&self, answer: &str) -> Result<AssertionOutcome> {
        match self {
            Self::Parsed(assertion) => assertion.check(answer),
            Self::ExpectedOutput(expected) => Ok(AssertionOutcome {
                text: format!("equals_normalized:{expected}"),
                passed: normalize_whitespace(answer) == *expected,
                evidence: excerpt(answer),
            }),
        }
    }
}

/// Validates the fields the harness needs; unlike `EvalCase::validate`, an empty
/// expectation list is allowed so expected-output-only cases stay runnable.
fn validate_case_inputs(case: &EvalCase) -> Result<()> {
    if case.id.trim().is_empty() {
        return Err(Error::validation("evaluation case id must not be blank"));
    }
    if case.name.trim().is_empty() {
        return Err(Error::validation(format!(
            "evaluation case {:?}: name must not be blank",
            case.id
        )));
    }
    if case.prompt.trim().is_empty() {
        return Err(Error::validation(format!(
            "evaluation case {:?}: prompt must not be blank",
            case.id
        )));
    }
    Ok(())
}

/// Builds the checks of one case, falling back to its expected output.
fn build_checks(case: &EvalCase) -> Result<Vec<Check>> {
    let mut checks = Vec::with_capacity(case.expectations.len().max(1));
    for expectation in &case.expectations {
        checks.push(Check::Parsed(Assertion::parse(expectation)?));
    }
    if !checks.is_empty() {
        return Ok(checks);
    }
    let expected = normalize_whitespace(&case.expected_output);
    if expected.is_empty() {
        return Err(Error::validation(format!(
            "evaluation case {:?}: at least one expectation or a non-blank expected output is \
             required",
            case.id
        )));
    }
    checks.push(Check::ExpectedOutput(expected));
    Ok(checks)
}

/// Trims `text` and collapses every whitespace run to a single space.
fn normalize_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Returns the first 200 characters of `text`, `…`-marked when truncated.
fn excerpt(text: &str) -> String {
    if text.chars().count() <= EVIDENCE_LIMIT {
        return text.to_string();
    }
    let mut kept: String = text.chars().take(EVIDENCE_LIMIT - 1).collect();
    kept.push('…');
    kept
}

/// Renders one case with the run-artifact wire keys.
fn case_json(case: &CaseResult) -> Result<serde_json::Value> {
    let passed = case.status == CaseStatus::Passed;
    let reason = if passed {
        String::new()
    } else {
        case.assertions
            .iter()
            .filter(|outcome| !outcome.passed)
            .map(|outcome| outcome.text.as_str())
            .collect::<Vec<_>>()
            .join("; ")
    };
    Ok(serde_json::json!({
        "case_id": case.id,
        "name": case.name,
        "status": serde_json::to_value(case.status).map_err(Error::external)?,
        "passed": passed,
        "reason": reason,
        "assertions": serde_json::to_value(&case.assertions).map_err(Error::external)?,
    }))
}

/// Creates the parent directory of `path` and writes `text` to it.
async fn write_artifact(path: &Path, text: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            tokio::fs::create_dir_all(parent).await.map_err(|error| {
                Error::external(error)
                    .with_context(format!("creating output directory `{}`", parent.display()))
            })?;
        }
    }
    tokio::fs::write(path, text.as_bytes())
        .await
        .map_err(|error| {
            Error::external(error).with_context(format!("writing `{}`", path.display()))
        })
}

/// One single-query run, mirroring the Python `run_eval_query.py` arguments.
///
/// Every value is caller-supplied, including the timestamp used in the generated
/// answer filename, so a run is reproducible and never reads the clock.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct QueryRequest {
    /// The query sent to the agent.
    pub prompt: String,
    /// Directory or explicit file the answer is written to.
    pub output_path: PathBuf,
    /// Base directory announced to the agent; defaults to the output directory.
    pub files_base_dir: Option<PathBuf>,
    /// Timestamp string embedded in a generated answer filename.
    pub timestamp: String,
    /// Optional iteration budget announced to the agent.
    pub max_iterations: Option<usize>,
}

/// What a single-query run produced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct QueryOutcome {
    /// The answer file.
    pub output_file: PathBuf,
    /// The JSON report written beside the answer file.
    pub report_file: PathBuf,
    /// The agent's answer, also stored in the answer file.
    pub answer: String,
}

/// Decides the answer filename for an output path.
///
/// The path is a directory when it already is one, has no extension, or its
/// string form ends with `/` or `\`; a directory gets
/// `<dir>/skill_test_result_<timestamp>.txt`. Any other path is an explicit file
/// and is returned unchanged. This function is pure: it inspects the path but
/// never creates a directory, which [`EvalHarness::run_query`] does.
#[must_use]
pub fn resolve_output_file(output_path: &Path, timestamp: &str) -> PathBuf {
    let file_name = format!("{RESULT_FILE_PREFIX}{timestamp}.txt");
    let rendered = output_path.to_string_lossy();
    let looks_like_directory = output_path.is_dir()
        || output_path.extension().is_none()
        || rendered.ends_with('/')
        || rendered.ends_with('\\');
    if looks_like_directory {
        output_path.join(file_name)
    } else {
        output_path.to_path_buf()
    }
}

/// The directory the answer file lives in, created if necessary.
///
/// A resolved file with no parent directory (for example `result.txt`) yields
/// `.`, the current directory, which is always present.
async fn answer_directory(output_file: &Path) -> Result<PathBuf> {
    let directory = output_file
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
    tokio::fs::create_dir_all(&directory)
        .await
        .map_err(|error| {
            Error::external(error).with_context(format!(
                "creating output directory `{}`",
                directory.display()
            ))
        })?;
    Ok(directory)
}

/// Builds the prompt of a single-query run: the Python context block, the
/// optional iteration budget, then the query itself.
fn build_query_prompt(
    query: &str,
    files_base_dir: &Path,
    output_dir: &Path,
    max_iterations: Option<usize>,
) -> String {
    let mut prompt = format!(
        "You are an intelligent assistant.\n\
         All user-provided files are located at '{}'\n\
         Put all generated files into '{}' folder\n\
         You may use tools when necessary.\n",
        files_base_dir.display(),
        output_dir.display()
    );
    if let Some(iterations) = max_iterations {
        prompt.push_str("Maximum agent iterations: ");
        prompt.push_str(&iterations.to_string());
        prompt.push('\n');
    }
    prompt.push_str(query);
    prompt
}

#[cfg(test)]
mod tests;
