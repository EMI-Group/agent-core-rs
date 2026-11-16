//! The JSONL run log for prompt-search runs.
//!
//! Ports `openjiuwen/dev_tools/tune/optimizer/prompt_search/run_log.py`: an
//! append-only JSONL stream with one record per iteration (the iteration index,
//! the candidate, its rewards, the convergence state, and the adoption decision)
//! plus a reader that tolerates truncated or foreign lines.
//!
//! Python's logger is untyped — `record(stage, **details)` writes whatever
//! keyword arguments the loop happens to pass, and `read_run_log` returns raw
//! dictionaries filtered on a `stage` key. The port replaces that with
//! [`RunLogRecord`], one typed record per round, so a run log can be read back,
//! compared, and asserted on. The seam itself is [`RunLog`]: [`JsonlRunLog`]
//! appends to a caller-supplied path, [`NullRunLog`] discards (the default when
//! no run directory is configured), and [`InMemoryRunLog`] keeps the records in
//! memory for tests and offline runs.
//!
//! # Surface
//!
//! - [`RunLogRecord`] — one iteration: `iteration` (1-based), the winning
//!   [`PromptCandidate`], the other `candidate_prompts` tried that round, the
//!   per-candidate-id `scores`, the winner's `reward` with its raw
//!   `reward_components` and `reward_notes`, the run's `best_score` after the
//!   round together with the `prior_best_score` the round started from (so
//!   [`RunLogRecord::is_improvement`] can report whether the elite was actually
//!   advanced) — `None` when there was no elite to beat — the `drift` deviation,
//!   the optional [`ConvergenceState`] with the
//!   mirrored `converged`/`convergence_reason` flags, the optional
//!   [`AdoptDecision`](crate::tune::AdoptDecision), whether it was `selected` as
//!   the run's elite, the caller-supplied `timestamp` (seconds), the `run_id`,
//!   and an optional `observation`. [`RunLogRecord::new`] plus the `with_*`
//!   builders fill it, and [`RunLogRecord::to_json_line`] /
//!   [`RunLogRecord::from_json_line`] are the codec.
//! - [`RunLog`] — `record(&mut self, &RunLogRecord)`, with a provided
//!   `record_all`. Writing never fails a run: an I/O or serialization failure is
//!   logged with `tracing::warn!` and reported as `Ok(())`.
//! - [`JsonlRunLog`] — `new(path)` (creates the parent directories and the file,
//!   keeps existing lines), `create(path)` (Python's `reset`: truncates),
//!   `path()`. One compact `serde_json` object per line, newline-terminated, in
//!   fixed struct-field order, so the same record always produces the same
//!   bytes.
//! - [`read_run_log`] / [`read_run_log_limited`] — a missing file reads as an
//!   empty list, blank lines are skipped, and a line that is not a
//!   [`RunLogRecord`] object is skipped rather than failing the read (that
//!   tolerance is deliberate: a truncated tail or a foreign event type must not
//!   hide the good records). `limit` keeps the last `limit` records; `0` means
//!   *no limit*.
//!
//! # Determinism
//!
//! Nothing here reads a clock, the environment, or a random number generator:
//! `timestamp` is a caller-supplied parameter, ids come from the record, and the
//! serialized line is a pure function of the record. The only I/O is the file
//! the caller names.
//!
//! Deviations from Python, all documented above: a typed record instead of
//! `stage` + arbitrary keyword details, a caller-supplied timestamp instead of
//! `datetime.now(timezone.utc)`, `limit == 0` meaning unlimited instead of
//! `max(1, limit)`, and no `logger.debug` rendering of the payload (the discard
//! path logs at `debug` instead).

use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

use super::convergence::ConvergenceState;
use super::models::PromptCandidate;
use crate::tune::AdoptDecision;

/// One iteration of a prompt-search run, as written to the run log.
///
/// `best_score` is the run's best reward *after* the round, so a reader can
/// compare a round against the run it belonged to, and `prior_best_score` is the
/// run's best reward *entering* the round — the two together say whether the
/// round advanced the elite (see [`RunLogRecord::is_improvement`]); a caller with
/// no history leaves them at `0.0` and `None`. `converged` and
/// `convergence_reason` mirror the round's stopping verdict for readers that only
/// need it, and `selected` says whether this candidate became the run's elite.
///
/// Every field defaults on deserialization, so a legacy or hand-written line
/// that carries only `iteration` and `candidate` still loads, and unknown keys
/// are ignored. `prior_best_score` is an added field: a line written before it
/// existed still parses and leaves it `None`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RunLogRecord {
    /// The 1-based round number.
    pub iteration: usize,
    /// The round's winning candidate.
    pub candidate: PromptCandidate,
    /// The other candidate prompts tried in this round, in evaluation order.
    pub candidate_prompts: Vec<String>,
    /// The score of every candidate of the round, keyed by candidate id.
    pub scores: BTreeMap<String, f64>,
    /// The reward the winning candidate earned.
    pub reward: f64,
    /// The winner's raw (pre-normalization) reward components.
    pub reward_components: BTreeMap<String, f64>,
    /// The winner's reward-adjustment notes (gate, overfitting, drift).
    pub reward_notes: Vec<String>,
    /// The run's best reward when this round started.
    ///
    /// `None` when the round had no elite to beat (round 1, which establishes
    /// the first one), and for a line written before this field existed — the
    /// reader leaves it `None` rather than inventing a history. A later round
    /// carries the elite score it started from, so
    /// [`RunLogRecord::is_improvement`] is a plain comparison against
    /// [`RunLogRecord::reward`].
    pub prior_best_score: Option<f64>,
    /// The run's best reward after this round.
    pub best_score: f64,
    /// The winner's semantic deviation from the task objective.
    pub drift: f64,
    /// The convergence verdict of this round, when the loop produced one.
    pub convergence: Option<ConvergenceState>,
    /// Whether the loop stopped after this round (`convergence.converged`).
    pub converged: bool,
    /// Why the loop stopped (`convergence.reason`), when it carried one.
    pub convergence_reason: Option<String>,
    /// The adoption decision for this candidate, when one was taken.
    pub adopt: Option<AdoptDecision>,
    /// Whether this candidate became the run's elite.
    pub selected: bool,
    /// The record's timestamp in seconds, always supplied by the caller.
    pub timestamp: f64,
    /// The identifier of the run, supplied by the caller.
    pub run_id: String,
    /// A free-form observation about the round.
    pub observation: Option<String>,
}

impl RunLogRecord {
    /// Creates the record of `iteration` whose winner is `candidate`.
    ///
    /// Every other field keeps its default (empty maps and lists, `0.0`
    /// rewards, no convergence or adoption verdict, `timestamp` `0.0`).
    #[must_use]
    pub fn new(iteration: usize, candidate: PromptCandidate) -> Self {
        Self {
            iteration,
            candidate,
            ..Self::default()
        }
    }

    /// Replaces the other candidate prompts tried in the round.
    #[must_use]
    pub fn with_candidate_prompts(mut self, candidate_prompts: Vec<String>) -> Self {
        self.candidate_prompts = candidate_prompts;
        self
    }

    /// Replaces the per-candidate scores of the round.
    #[must_use]
    pub fn with_scores(mut self, scores: BTreeMap<String, f64>) -> Self {
        self.scores = scores;
        self
    }

    /// Records the score of one candidate of the round.
    #[must_use]
    pub fn with_score(mut self, candidate_id: impl Into<String>, score: f64) -> Self {
        self.scores.insert(candidate_id.into(), score);
        self
    }

    /// Replaces the winner's reward.
    #[must_use]
    pub fn with_reward(mut self, reward: f64) -> Self {
        self.reward = reward;
        self
    }

    /// Replaces the winner's raw reward components.
    #[must_use]
    pub fn with_reward_components(mut self, components: BTreeMap<String, f64>) -> Self {
        self.reward_components = components;
        self
    }

    /// Replaces the winner's reward notes.
    #[must_use]
    pub fn with_reward_notes(mut self, notes: Vec<String>) -> Self {
        self.reward_notes = notes;
        self
    }

    /// Sets the run's best reward when the round started.
    ///
    /// Pass `None` for a round that had no elite to beat (round 1).
    #[must_use]
    pub fn with_prior_best_score(mut self, prior_best_score: Option<f64>) -> Self {
        self.prior_best_score = prior_best_score;
        self
    }

    /// Sets the run's best reward after the round.
    #[must_use]
    pub fn with_best_score(mut self, best_score: f64) -> Self {
        self.best_score = best_score;
        self
    }

    /// Sets the winner's drift deviation.
    #[must_use]
    pub fn with_drift(mut self, drift: f64) -> Self {
        self.drift = drift;
        self
    }

    /// Sets the convergence verdict and mirrors it into
    /// [`RunLogRecord::converged`] and (when the state carries one)
    /// [`RunLogRecord::convergence_reason`].
    #[must_use]
    pub fn with_convergence(mut self, convergence: ConvergenceState) -> Self {
        self.converged = convergence.converged;
        if !convergence.reason.is_empty() {
            self.convergence_reason = Some(convergence.reason.clone());
        }
        self.convergence = Some(convergence);
        self
    }

    /// Overrides the mirrored [`RunLogRecord::converged`] flag.
    #[must_use]
    pub fn with_converged(mut self, converged: bool) -> Self {
        self.converged = converged;
        self
    }

    /// Overrides the mirrored [`RunLogRecord::convergence_reason`].
    #[must_use]
    pub fn with_convergence_reason(mut self, reason: impl Into<String>) -> Self {
        self.convergence_reason = Some(reason.into());
        self
    }

    /// Sets the adoption decision for this candidate.
    #[must_use]
    pub fn with_adopt(mut self, adopt: AdoptDecision) -> Self {
        self.adopt = Some(adopt);
        self
    }

    /// Sets whether this candidate became the run's elite.
    #[must_use]
    pub fn with_selected(mut self, selected: bool) -> Self {
        self.selected = selected;
        self
    }

    /// Sets the caller-supplied timestamp, in seconds.
    #[must_use]
    pub fn with_timestamp(mut self, timestamp: f64) -> Self {
        self.timestamp = timestamp;
        self
    }

    /// Sets the run identifier.
    #[must_use]
    pub fn with_run_id(mut self, run_id: impl Into<String>) -> Self {
        self.run_id = run_id.into();
        self
    }

    /// Sets the free-form observation of the round.
    #[must_use]
    pub fn with_observation(mut self, observation: impl Into<String>) -> Self {
        self.observation = Some(observation.into());
        self
    }

    /// The 1-based round number.
    #[must_use]
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// The round's winning candidate.
    #[must_use]
    pub fn candidate(&self) -> &PromptCandidate {
        &self.candidate
    }

    /// The winner's reward.
    #[must_use]
    pub fn reward(&self) -> f64 {
        self.reward
    }

    /// Whether the round advanced the run's elite.
    ///
    /// The round's own best score is its winner's [`RunLogRecord::reward`], and
    /// the elite it had to beat is [`RunLogRecord::prior_best_score`], so the
    /// comparison is exactly "the round's best score strictly exceeded the score
    /// of the elite entering the round": a round that matched the elite or fell
    /// short reports `false`.
    ///
    /// A round with no pre-round elite reports `false`. That is round 1, which
    /// *establishes* the first elite rather than advancing one — the
    /// [`RunLogRecord::selected`] flag reports that establishment — and it is
    /// also a hand-built record or a legacy line written before
    /// `prior_best_score` existed, where no comparison is possible.
    ///
    /// [`RunLogRecord::best_score`] is the *post*-round best, so comparing
    /// against it would only ever be able to report `false`; use it to read the
    /// run's level, and this predicate to read the round's effect on it.
    #[must_use]
    pub fn is_improvement(&self) -> bool {
        self.prior_best_score
            .is_some_and(|prior| self.reward > prior)
    }

    /// Whether an adoption decision was recorded and it adopted the candidate.
    #[must_use]
    pub fn adopted(&self) -> bool {
        self.adopt.is_some_and(|decision| decision.adopted)
    }

    /// Renders the record as one compact JSON object without a trailing newline.
    ///
    /// The output is a pure function of the record — fields in declaration order
    /// and maps in key order — so the same record always serializes to the same
    /// bytes.
    ///
    /// # Errors
    ///
    /// The serializer error, wrapped as
    /// [`Error::external`](openjiuwen_common::error::Error::external). The data
    /// types here are all serializable, so this is a defensive path only.
    pub fn to_json_line(&self) -> Result<String> {
        serde_json::to_string(self).map_err(Error::external)
    }

    /// Parses one JSONL line, or `None` when it is not a [`RunLogRecord`].
    ///
    /// Surrounding whitespace is trimmed. A line that is blank, not JSON, not a
    /// JSON object (an array, a scalar, or `null`), or names a field with the
    /// wrong type yields `None` instead of an error, so a reader can skip it.
    /// Because every field defaults, a JSON object that shares no keys with the
    /// record parses as an all-default record.
    #[must_use]
    pub fn from_json_line(line: &str) -> Option<Self> {
        let line = line.trim();
        if line.is_empty() {
            return None;
        }
        let value: serde_json::Value = serde_json::from_str(line).ok()?;
        if !value.is_object() {
            return None;
        }
        serde_json::from_value(value).ok()
    }
}

/// A sink for [`RunLogRecord`]s.
///
/// Implementations are `Send + Sync` so a log can live in a shared optimizer, and
/// `record` takes `&mut self` because a JSONL sink owns the append position. The
/// `Result` is part of the seam for a caller that supplies its own sink; every
/// implementation in this module keeps a run alive — [`JsonlRunLog`] warns and
/// continues, [`NullRunLog`] discards, and [`InMemoryRunLog`] stores.
pub trait RunLog: Send + Sync {
    /// Appends one record.
    ///
    /// # Errors
    ///
    /// The implementation's error. [`JsonlRunLog`] deliberately returns `Ok(())`
    /// after logging a failed write, so a run log can never fail a run.
    fn record(&mut self, record: &RunLogRecord) -> Result<()>;

    /// Appends every record, in order.
    ///
    /// # Errors
    ///
    /// The first error [`RunLog::record`] returns.
    fn record_all(&mut self, records: &[RunLogRecord]) -> Result<()> {
        for record in records {
            self.record(record)?;
        }
        Ok(())
    }
}

/// A JSONL run log at a caller-supplied path.
///
/// [`JsonlRunLog::new`] creates the parent directories and the file when they do
/// not exist and keeps whatever the file already holds, so a resumed run appends
/// to its history; [`JsonlRunLog::create`] truncates instead (Python's `reset`).
/// Each record is written as one compact JSON object followed by `\n`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonlRunLog {
    /// The log file path.
    path: PathBuf,
}

impl JsonlRunLog {
    /// Opens the log at `path`, creating the parent directories and the file.
    ///
    /// An existing file is kept: earlier lines survive and the new records are
    /// appended after them.
    ///
    /// # Errors
    ///
    /// [`Error::validation`](openjiuwen_common::error::Error::validation) for a
    /// blank path, and the filesystem error when a directory cannot be created or
    /// the file cannot be opened for writing.
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let path = checked_path(path.as_ref())?;
        let log = Self { path };
        log.prepare(false)?;
        Ok(log)
    }

    /// Creates (or truncates) the log at `path`, discarding earlier records.
    ///
    /// This is Python's `OptimizerRunLogger::reset`: use it once at the start of a
    /// fresh run.
    ///
    /// # Errors
    ///
    /// [`Error::validation`](openjiuwen_common::error::Error::validation) for a
    /// blank path, and the filesystem error when a directory cannot be created or
    /// the file cannot be truncated.
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        let path = checked_path(path.as_ref())?;
        let log = Self { path };
        log.prepare(true)?;
        Ok(log)
    }

    /// The log file path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Creates the parent directories and opens the file, truncating on demand.
    fn prepare(&self, truncate: bool) -> Result<()> {
        ensure_parent(&self.path)?;
        let mut options = OpenOptions::new();
        options.create(true).write(true);
        if truncate {
            options.truncate(true);
        }
        options.open(&self.path)?;
        Ok(())
    }

    /// Appends one serialized line plus its newline.
    fn append(&self, line: &str) -> Result<()> {
        ensure_parent(&self.path)?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        let mut bytes = String::with_capacity(line.len() + 1);
        bytes.push_str(line);
        bytes.push('\n');
        file.write_all(bytes.as_bytes())?;
        file.flush()?;
        Ok(())
    }
}

impl RunLog for JsonlRunLog {
    /// Appends one record, logging (never propagating) a failed write.
    ///
    /// Python warns and continues, so a full disk, a read-only mount, or a path
    /// replaced by a directory cannot fail an optimization run; the same holds
    /// here, and the reason reaches the logs through `tracing::warn!`.
    fn record(&mut self, record: &RunLogRecord) -> Result<()> {
        let line = match record.to_json_line() {
            Ok(line) => line,
            Err(error) => {
                tracing::warn!(
                    path = %self.path.display(),
                    error = %error,
                    "run log: could not serialize the record"
                );
                return Ok(());
            }
        };
        if let Err(error) = self.append(&line) {
            tracing::warn!(
                path = %self.path.display(),
                error = %error,
                "run log: failed to write the record"
            );
        }
        Ok(())
    }
}

/// A run log that discards every record.
///
/// The default when no run directory is configured. Each discarded record is
/// logged at `debug`, mirroring Python's `NullRunLogger`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NullRunLog;

impl NullRunLog {
    /// Creates the discarding log.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl RunLog for NullRunLog {
    fn record(&mut self, record: &RunLogRecord) -> Result<()> {
        tracing::debug!(
            iteration = record.iteration,
            candidate = %record.candidate.candidate_id,
            "run log: discarding the record"
        );
        Ok(())
    }
}

/// A run log that keeps its records in memory.
///
/// For tests and fully offline runs: [`InMemoryRunLog::records`] exposes what was
/// written, in write order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct InMemoryRunLog {
    /// The recorded rounds, in write order.
    records: Vec<RunLogRecord>,
}

impl InMemoryRunLog {
    /// Creates an empty in-memory log.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The recorded rounds, in write order.
    #[must_use]
    pub fn records(&self) -> &[RunLogRecord] {
        &self.records
    }

    /// Consumes the log and returns its records, in write order.
    #[must_use]
    pub fn into_records(self) -> Vec<RunLogRecord> {
        self.records
    }

    /// The most recently recorded round.
    #[must_use]
    pub fn last(&self) -> Option<&RunLogRecord> {
        self.records.last()
    }

    /// How many rounds were recorded.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether nothing was recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Drops every recorded round.
    pub fn clear(&mut self) {
        self.records.clear();
    }
}

impl RunLog for InMemoryRunLog {
    fn record(&mut self, record: &RunLogRecord) -> Result<()> {
        self.records.push(record.clone());
        Ok(())
    }
}

/// Reads every acceptable record of the log at `path`.
///
/// Equivalent to [`read_run_log_limited`] with no limit.
///
/// # Errors
///
/// [`Error::validation`](openjiuwen_common::error::Error::validation) for a blank
/// path, and the filesystem error when the file exists but cannot be read.
pub fn read_run_log(path: impl AsRef<Path>) -> Result<Vec<RunLogRecord>> {
    read_run_log_limited(path, 0)
}

/// Reads the last `limit` acceptable records of the log at `path`.
///
/// A missing file yields an empty list (Python's behaviour), blank lines are
/// skipped, and a line that is not a [`RunLogRecord`] object is skipped — a
/// truncated tail from a crashed run or a foreign event type never hides the
/// good records, and neither is reported as an error.
///
/// `limit == 0` means *no limit*; any other value keeps the last `limit`
/// records, in file order.
///
/// # Errors
///
/// [`Error::validation`](openjiuwen_common::error::Error::validation) for a blank
/// path, and the filesystem error when the file exists but cannot be read (an
/// unreadable file is a real failure, unlike a corrupt line).
pub fn read_run_log_limited(path: impl AsRef<Path>, limit: usize) -> Result<Vec<RunLogRecord>> {
    let path = checked_path(path.as_ref())?;
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let text = std::fs::read_to_string(&path)?;
    let mut records = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        if let Some(record) = RunLogRecord::from_json_line(line) {
            records.push(record);
        }
    }
    if limit > 0 && records.len() > limit {
        let excess = records.len() - limit;
        records.drain(..excess);
    }
    Ok(records)
}

/// Rejects a blank path, which is always a caller mistake.
fn checked_path(path: &Path) -> Result<PathBuf> {
    if path.as_os_str().is_empty() {
        return Err(Error::validation("run log path must not be empty"));
    }
    Ok(path.to_path_buf())
}

/// Creates the parent directory of `path` when it has one.
fn ensure_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "run_log_tests.rs"]
mod tests;
