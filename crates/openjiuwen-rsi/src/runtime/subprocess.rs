//! Subprocess evaluation of candidate programs.
//!
//! Port of the execution seam in `artifact_rsi/program_opt/execution.py` and the
//! result reading in `artifact_rsi/program_opt/reply_format.py` (v0.1.20), minus
//! the injected `SysOperation` sandbox: this port stages a program in a temporary
//! directory, runs one interpreter through the [`CommandRunner`] seam, and reads
//! the reply back.
//!
//! The shape of one evaluation, in Python's words: *stage a file tree, run one
//! command in it with an explicit environment, read one result file back,
//! tolerantly*. Absence of a result is an ordinary outcome — a candidate that
//! printed nothing scores `-inf` upstream rather than failing the run — so the
//! [`SubprocessResult`] carries `success: false` plus an `error` naming the
//! problem instead of returning an [`Error`].
//!
//! Two reply protocols, matching [`ReplyFormat`]:
//!
//! - [`ReplyFormat::Files`] — the program writes `output.json` (or
//!   `response.json`) into its working directory;
//! - [`ReplyFormat::Tagged`] — the program prints the JSON, either in a
//!   ```` ```json ```` fenced block, inside `<result>…</result>`, or as the whole
//!   of standard output.
//!
//! The working directory is removed after every evaluation unless
//! [`SubprocessEvaluator::keep_workdir`] was called.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::warn;

use openjiuwen_common::error::{Error, Result};

use crate::artifact_rsi::program_opt::ReplyFormat;

use super::command::{CommandOutcome, CommandRunner, CommandSpec};

/// Default wall-clock budget for one candidate run.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

/// How one candidate program is executed and read back.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct SubprocessConfig {
    /// Interpreter used to run the candidate.
    pub python: String,
    /// Directory the per-run working directory is created under.
    pub workdir_root: PathBuf,
    /// Wall-clock budget for one run; `None` means no deadline.
    pub timeout: Option<Duration>,
    /// Program file name written into the working directory.
    pub entrypoint: String,
    /// Reply protocol the candidate is expected to follow.
    pub reply_format: ReplyFormat,
    /// Extra environment variables set for the child.
    pub env: BTreeMap<String, String>,
}

impl Default for SubprocessConfig {
    fn default() -> Self {
        Self {
            python: "python3".to_owned(),
            workdir_root: std::env::temp_dir(),
            timeout: Some(DEFAULT_TIMEOUT),
            entrypoint: "candidate.py".to_owned(),
            reply_format: ReplyFormat::Files,
            env: BTreeMap::new(),
        }
    }
}

/// One candidate evaluation: the process outcome plus the parsed reply.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SubprocessResult {
    /// Whether the process succeeded **and** its reply parsed.
    pub success: bool,
    /// Parsed reply, or [`Value::Null`] when none was read.
    pub response: Value,
    /// Raw process outcome (exit code, output, timing).
    pub outcome: CommandOutcome,
    /// Working directory the candidate ran in.
    pub workdir: PathBuf,
    /// Why the evaluation failed, when it did.
    pub error: Option<String>,
}

impl Default for SubprocessResult {
    fn default() -> Self {
        Self {
            success: false,
            response: Value::Null,
            outcome: CommandOutcome {
                code: None,
                stdout: String::new(),
                stderr: String::new(),
                timed_out: false,
                cancelled: false,
                duration: Duration::ZERO,
            },
            workdir: PathBuf::new(),
            error: None,
        }
    }
}

/// Runs candidate programs through a [`CommandRunner`].
#[derive(Clone)]
pub struct SubprocessEvaluator {
    runner: Arc<dyn CommandRunner>,
    config: SubprocessConfig,
    keep_workdir: bool,
}

impl SubprocessEvaluator {
    /// Creates an evaluator with the default configuration.
    #[must_use]
    pub fn new(runner: Arc<dyn CommandRunner>) -> Self {
        Self {
            runner,
            config: SubprocessConfig::default(),
            keep_workdir: false,
        }
    }

    /// Replaces the configuration.
    #[must_use]
    pub fn with_config(mut self, config: SubprocessConfig) -> Self {
        self.config = config;
        self
    }

    /// Keeps each run's working directory for inspection (debugging only).
    ///
    /// Off by default: a search evaluates hundreds of candidates, so leaving the
    /// staged trees behind would fill the temporary directory.
    #[must_use]
    pub fn keep_workdir(mut self) -> Self {
        self.keep_workdir = true;
        self
    }

    /// Returns the configuration in use.
    #[must_use]
    pub fn config(&self) -> &SubprocessConfig {
        &self.config
    }

    /// Stages `program`, runs it, and parses its reply.
    ///
    /// The working directory is `<workdir_root>/openjiuwen-rsi-program-<uuid>`;
    /// the program is written to [`SubprocessConfig::entrypoint`] and run as
    /// `<python> <entrypoint>` with `cwd` set to that directory.
    ///
    /// A non-zero exit, a missing reply file, or an unparseable reply is
    /// reported through [`SubprocessResult::success`] and
    /// [`SubprocessResult::error`]. The directory is removed afterwards unless
    /// [`SubprocessEvaluator::keep_workdir`] is set.
    ///
    /// # Errors
    ///
    /// Returns [`Error::tool`] when the working directory or entrypoint cannot be
    /// written, and propagates [`CommandRunner::run`] errors (a missing
    /// interpreter is [`Error::config`]).
    pub async fn evaluate(&self, program: &str) -> Result<SubprocessResult> {
        let workdir = self
            .config
            .workdir_root
            .join(format!("openjiuwen-rsi-program-{}", uuid::Uuid::new_v4()));
        tokio::fs::create_dir_all(&workdir).await.map_err(|error| {
            Error::tool(format!(
                "could not create program workdir {}: {error}",
                workdir.display()
            ))
        })?;
        let entrypoint_path = workdir.join(&self.config.entrypoint);
        let staged = self.stage(&entrypoint_path, program).await;
        if let Err(error) = staged {
            self.cleanup(&workdir).await;
            return Err(error);
        }
        let mut spec = CommandSpec::new(&self.config.python)
            .arg(&self.config.entrypoint)
            .cwd(&workdir)
            .merge_stderr();
        for (key, value) in &self.config.env {
            spec = spec.env(key, value);
        }
        if let Some(timeout) = self.config.timeout {
            spec = spec.timeout(timeout);
        }
        let outcome = match self.runner.run(spec).await {
            Ok(outcome) => outcome,
            Err(error) => {
                self.cleanup(&workdir).await;
                return Err(error);
            }
        };
        let (response, error) = self.read_reply(&workdir, &outcome).await;
        let success = outcome.success() && response.is_some();
        let error = match (error, outcome.success()) {
            (Some(error), _) => Some(error),
            (None, true) => None,
            (None, false) => Some(failure_reason(&outcome)),
        };
        self.cleanup(&workdir).await;
        Ok(SubprocessResult {
            success,
            response: response.unwrap_or(Value::Null),
            outcome,
            workdir,
            error,
        })
    }

    /// Writes the program to `path`, creating parent directories.
    async fn stage(&self, path: &Path, program: &str) -> Result<()> {
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|error| {
                Error::tool(format!(
                    "could not create program directory {}: {error}",
                    parent.display()
                ))
            })?;
        }
        tokio::fs::write(path, program).await.map_err(|error| {
            Error::tool(format!(
                "could not write program to {}: {error}",
                path.display()
            ))
        })
    }

    /// Reads the reply according to the configured protocol.
    async fn read_reply(
        &self,
        workdir: &Path,
        outcome: &CommandOutcome,
    ) -> (Option<Value>, Option<String>) {
        match self.config.reply_format {
            ReplyFormat::Files => match parse_files_reply(workdir).await {
                Ok(value) => (Some(value), None),
                Err(error) => (None, Some(error.to_string())),
            },
            ReplyFormat::Tagged => match parse_tagged_reply(outcome.combined_output()) {
                Some(value) => (Some(value), None),
                None => (
                    None,
                    Some(
                        "no JSON reply found in stdout: expected a ```json fenced block, \
                         a <result> block, or a JSON document"
                            .to_owned(),
                    ),
                ),
            },
        }
    }

    /// Removes the working directory unless it is being kept.
    async fn cleanup(&self, workdir: &Path) {
        if self.keep_workdir {
            return;
        }
        if tokio::fs::remove_dir_all(workdir).await.is_err() {
            warn!(
                "could not remove program workdir {}; leaking it",
                workdir.display()
            );
        }
    }
}

/// Describes a non-successful process outcome for [`SubprocessResult::error`].
fn failure_reason(outcome: &CommandOutcome) -> String {
    if outcome.timed_out {
        return "program timed out".to_owned();
    }
    if outcome.cancelled {
        return "program was cancelled".to_owned();
    }
    let code = outcome
        .code
        .map_or_else(|| "none".to_owned(), |code| code.to_string());
    format!(
        "program exited with code {code}: {}",
        outcome.truncated_output(500)
    )
}

/// Reads the JSON reply a `Files`-protocol candidate wrote.
///
/// Looks for `output.json`, then `response.json`, in `dir`. A missing file is a
/// validation error naming both candidates; malformed JSON is a validation error
/// naming the file.
///
/// # Errors
///
/// Returns [`Error::validation`] when neither file exists or the one found is not
/// JSON, and [`Error::tool`] when a file exists but cannot be read.
pub async fn parse_files_reply(dir: &Path) -> Result<Value> {
    for name in ["output.json", "response.json"] {
        let path = dir.join(name);
        let text = match tokio::fs::read_to_string(&path).await {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(Error::tool(format!(
                    "could not read reply file {}: {error}",
                    path.display()
                )));
            }
        };
        return serde_json::from_str(&text).map_err(|error| {
            Error::validation(format!(
                "reply file {} is not valid JSON: {error}",
                path.display()
            ))
        });
    }
    Err(Error::validation(format!(
        "program wrote neither output.json nor response.json in {}",
        dir.display()
    )))
}

/// Parses a `Tagged`-protocol reply out of standard output.
///
/// Tries, in order: the last ```` ```json ```` (or unlabelled) fenced block, the
/// last `<result>…</result>` block, and finally the whole trimmed output as a
/// JSON document. Returns `None` when none of them is valid JSON.
#[must_use]
pub fn parse_tagged_reply(stdout: &str) -> Option<Value> {
    if let Some(block) = last_fenced_block(stdout) {
        if let Ok(value) = serde_json::from_str(block.trim()) {
            return Some(value);
        }
    }
    if let Some(block) = last_element(stdout, "result") {
        if let Ok(value) = serde_json::from_str(block.trim()) {
            return Some(value);
        }
    }
    serde_json::from_str(stdout.trim()).ok()
}

/// Returns the content of the last JSON (or unlabelled) fenced code block.
fn last_fenced_block(text: &str) -> Option<&str> {
    let mut candidates = Vec::new();
    let mut open: Option<(usize, bool)> = None;
    let mut offset = 0usize;
    for line in text.split_inclusive('\n') {
        let line_start = offset;
        offset += line.len();
        let trimmed = line.trim();
        let Some(info) = trimmed.strip_prefix("```") else {
            continue;
        };
        if let Some((content_start, allowed)) = open.take() {
            if allowed {
                candidates.push(&text[content_start..line_start]);
            }
        } else {
            let info = info.trim().trim_end_matches('`').to_ascii_lowercase();
            open = Some((offset, info.is_empty() || info == "json"));
        }
    }
    candidates.pop()
}

/// Returns the content of the last `<name>…</name>` element, case-insensitively.
fn last_element<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    let lowered = text.to_ascii_lowercase();
    let open = format!("<{name}>");
    let close = format!("</{name}>");
    let start = lowered.rfind(&open)? + open.len();
    let end = lowered[start..].find(&close)? + start;
    Some(&text[start..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tagged_reply_prefers_the_last_json_fence() {
        let stdout = "```json\n{\"first\": 1}\n```\nnoise\n```json\n{\"second\": 2}\n```\n";
        assert_eq!(
            parse_tagged_reply(stdout),
            Some(serde_json::json!({"second": 2}))
        );
    }

    #[test]
    fn tagged_reply_accepts_result_block_and_bare_json() {
        assert_eq!(
            parse_tagged_reply("<result>{\"a\": 1}</result>"),
            Some(serde_json::json!({"a": 1}))
        );
        assert_eq!(
            parse_tagged_reply("  {\"b\": 2}  "),
            Some(serde_json::json!({"b": 2}))
        );
        assert_eq!(parse_tagged_reply("not json"), None);
        assert_eq!(parse_tagged_reply("```json\n{nope}\n```"), None);
    }

    #[test]
    fn unlabelled_fence_is_accepted() {
        assert_eq!(
            parse_tagged_reply("```\n{\"c\": 3}\n```"),
            Some(serde_json::json!({"c": 3}))
        );
    }
}
