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
//!   `reward_components` and `reward_notes`, the `best_score` it had to beat,
//!   the `drift` deviation, the optional [`ConvergenceState`] with the mirrored
//!   `converged`/`convergence_reason` flags, the optional
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

// The whole surface belongs at the `tune` root: until `crate::tune` re-exports
// it, no path outside this crate can name it, so the reachability and dead-code
// lints are structurally inert (the same pattern the `adopt` family used before
// its root re-export landed).
#![allow(dead_code, unreachable_pub)]

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
/// `best_score` is the reward the round's winner had to beat (the run's best
/// before the round); a caller with no history leaves it at `0.0`. `converged`
/// and `convergence_reason` mirror `convergence` for readers that only need the
/// verdict, and `selected` says whether this candidate became the run's elite.
///
/// Every field defaults on deserialization, so a legacy or hand-written line
/// that carries only `iteration` and `candidate` still loads, and unknown keys
/// are ignored.
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
    /// The best reward recorded before this round: the score to beat.
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

    /// Sets the best reward the round had to beat.
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

    /// Whether the round's winner beat [`RunLogRecord::best_score`].
    ///
    /// An equal reward is not an improvement: the reward is a clamped float and
    /// the loop keeps the incumbent on a tie.
    #[must_use]
    pub fn is_improvement(&self) -> bool {
        self.reward > self.best_score
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
mod tests {
    use super::*;
    use crate::tune::AdoptReason;
    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_test_support::TempWorkspace;

    fn candidate(prompt: &str, rationale: &str) -> PromptCandidate {
        PromptCandidate::new(prompt, rationale)
    }

    fn convergence() -> ConvergenceState {
        ConvergenceState {
            converged: true,
            reason: "rewards plateaued".to_owned(),
            best: 0.8,
            moving_average: 0.79,
            variance: 0.0001,
        }
    }

    /// A fully populated round, including an adoption verdict.
    fn full_record(iteration: usize) -> RunLogRecord {
        let winner = candidate("Answer {{question}} briefly.", "kept the placeholder");
        RunLogRecord::new(iteration, winner.clone())
            .with_candidate_prompts(vec![
                "Answer {{question}} with steps.".to_owned(),
                "Answer shortly.".to_owned(),
            ])
            .with_score(winner.candidate_id.clone(), 0.8)
            .with_score("deadbeef", 0.55)
            .with_reward(0.8)
            .with_reward_components(BTreeMap::from([
                ("correctness".to_owned(), 0.9),
                ("completeness".to_owned(), 0.7),
            ]))
            .with_reward_notes(vec!["correctness gate passed".to_owned()])
            .with_best_score(0.6)
            .with_drift(0.12)
            .with_convergence(convergence())
            .with_adopt(AdoptDecision::new(
                true,
                AdoptReason::Improved { gain: 0.2 },
            ))
            .with_selected(true)
            .with_timestamp(1_700_000_000.5)
            .with_run_id("run-1")
            .with_observation("first round")
    }

    /// A round with every optional field absent.
    fn bare_record(iteration: usize) -> RunLogRecord {
        RunLogRecord::new(iteration, candidate("Do the thing.", "seed"))
    }

    fn log_path(workspace: &TempWorkspace) -> PathBuf {
        workspace.path("runs/prompt_search.jsonl")
    }

    #[test]
    fn full_round_trip_preserves_every_field() {
        let workspace = TempWorkspace::new().unwrap();
        let path = log_path(&workspace);
        let mut log = JsonlRunLog::new(&path).unwrap();
        assert_eq!(log.path(), path.as_path());

        let first = full_record(1);
        let second = bare_record(2);
        log.record(&first).unwrap();
        log.record(&second).unwrap();

        let read = read_run_log(&path).unwrap();
        assert_eq!(read, vec![first.clone(), second.clone()]);
        assert_eq!(
            read[0].adopt,
            Some(AdoptDecision::new(
                true,
                AdoptReason::Improved { gain: 0.2 }
            ))
        );
        assert_eq!(read[0].convergence, Some(convergence()));
        assert!(read[0].converged);
        assert_eq!(
            read[0].convergence_reason.as_deref(),
            Some("rewards plateaued")
        );
        assert_eq!(read[0].scores.len(), 2);
        assert_eq!(read[0].candidate_prompts.len(), 2);
        assert_eq!(read[0].reward_components["correctness"], 0.9);
        assert_eq!(
            read[0].reward_notes,
            vec!["correctness gate passed".to_owned()]
        );
        assert!((read[0].timestamp - 1_700_000_000.5).abs() < 1e-9);
        assert_eq!(read[0].run_id, "run-1");
        assert_eq!(read[0].observation.as_deref(), Some("first round"));
        assert!(read[0].selected);
        assert!(read[0].is_improvement());
        assert!(read[0].adopted());
        assert_eq!(read[0].iteration(), 1);
        assert_eq!(read[0].candidate(), &first.candidate);
        assert!((read[0].reward() - 0.8).abs() < 1e-12);

        // The `None` variants survive as `None`.
        assert_eq!(read[1].adopt, None);
        assert_eq!(read[1].convergence, None);
        assert_eq!(read[1].convergence_reason, None);
        assert_eq!(read[1].observation, None);
        assert!(!read[1].converged);
        assert!(!read[1].selected);
        assert!(!read[1].adopted());
        assert_eq!(read[1].scores, BTreeMap::new());
        assert_eq!(read[1].candidate_prompts, Vec::<String>::new());
    }

    #[test]
    fn append_across_instances_keeps_earlier_lines() {
        let workspace = TempWorkspace::new().unwrap();
        let path = log_path(&workspace);

        {
            let mut log = JsonlRunLog::new(&path).unwrap();
            log.record(&bare_record(1)).unwrap();
        }
        {
            let mut log = JsonlRunLog::new(&path).unwrap();
            log.record(&bare_record(2)).unwrap();
            log.record(&bare_record(3)).unwrap();
        }

        let read = read_run_log(&path).unwrap();
        assert_eq!(
            read.iter()
                .map(|record| record.iteration)
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn create_truncates_the_file() {
        let workspace = TempWorkspace::new().unwrap();
        let path = log_path(&workspace);

        let mut log = JsonlRunLog::new(&path).unwrap();
        log.record(&bare_record(1)).unwrap();
        log.record(&bare_record(2)).unwrap();
        assert_eq!(read_run_log(&path).unwrap().len(), 2);

        let mut truncated = JsonlRunLog::create(&path).unwrap();
        assert_eq!(read_run_log(&path).unwrap(), Vec::new());

        truncated.record(&bare_record(7)).unwrap();
        let read = read_run_log(&path).unwrap();
        assert_eq!(read.len(), 1);
        assert_eq!(read[0].iteration, 7);
    }

    #[test]
    fn missing_file_reads_empty() {
        let workspace = TempWorkspace::new().unwrap();
        let path = workspace.path("absent/runs.jsonl");
        assert_eq!(read_run_log(&path).unwrap(), Vec::new());
        assert_eq!(read_run_log_limited(&path, 3).unwrap(), Vec::new());
    }

    #[test]
    fn blank_and_corrupt_lines_are_skipped() {
        let workspace = TempWorkspace::new().unwrap();
        let path = workspace.path("runs/mixed.jsonl");
        let good = bare_record(4);
        let line = good.to_json_line().unwrap();
        let contents = format!(
            "\n{line}\n\n   \n{{\"stage\":\"round\"}}\nnot json at all\n[1, 2, 3]\n{line}\n"
        );
        workspace.write("runs/mixed.jsonl", contents).unwrap();

        let read = read_run_log(&path).unwrap();
        // The `{"stage": …}` object is an *acceptable* record — every field
        // defaults — while the plain scalar, the array, and the garbage line are
        // skipped.
        assert_eq!(read.len(), 3);
        assert_eq!(read[0], good);
        assert_eq!(read[1], RunLogRecord::default());
        assert_eq!(read[2], good);
    }

    #[test]
    fn limit_keeps_the_last_records() {
        let workspace = TempWorkspace::new().unwrap();
        let path = log_path(&workspace);
        let mut log = JsonlRunLog::new(&path).unwrap();
        for iteration in 1..=5 {
            log.record(&bare_record(iteration)).unwrap();
        }

        let last_two = read_run_log_limited(&path, 2).unwrap();
        assert_eq!(
            last_two
                .iter()
                .map(|record| record.iteration)
                .collect::<Vec<_>>(),
            vec![4, 5]
        );

        assert_eq!(read_run_log_limited(&path, 10).unwrap().len(), 5);
        assert_eq!(read_run_log_limited(&path, 1).unwrap()[0].iteration, 5);

        // `0` means no limit, and is not clamped to one.
        assert_eq!(read_run_log_limited(&path, 0).unwrap().len(), 5);
        assert_eq!(read_run_log(&path).unwrap().len(), 5);
    }

    #[test]
    fn multiline_and_unicode_prompts_survive() {
        let workspace = TempWorkspace::new().unwrap();
        let path = log_path(&workspace);
        let prompt = "第一行\n第二行\t\"quoted\" \\ backslash 🙂\r\n";
        let record = RunLogRecord::new(1, candidate(prompt, "中文说明\nsecond line"))
            .with_candidate_prompts(vec![format!("prefix {prompt} suffix")])
            .with_observation("emoji ✅ and a newline\n");

        let mut log = JsonlRunLog::new(&path).unwrap();
        log.record(&record).unwrap();

        // One physical line only.
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text.lines().count(), 1);

        let read = read_run_log(&path).unwrap();
        assert_eq!(read, vec![record]);
        assert_eq!(read[0].candidate.prompt, prompt);
        assert_eq!(read[0].candidate.rationale, "中文说明\nsecond line");
    }

    #[test]
    fn null_run_log_discards_and_in_memory_keeps_order() {
        let mut null = NullRunLog::new();
        assert!(null.record(&bare_record(1)).is_ok());
        assert_eq!(null, NullRunLog);

        let mut memory = InMemoryRunLog::new();
        assert!(memory.is_empty());
        memory.record(&bare_record(1)).unwrap();
        memory.record(&full_record(2)).unwrap();
        assert_eq!(memory.len(), 2);
        assert_eq!(memory.records()[0].iteration, 1);
        assert_eq!(memory.last().unwrap().iteration, 2);
        assert_eq!(memory.into_records().len(), 2);

        let mut memory = InMemoryRunLog::new();
        memory
            .record_all(&[bare_record(1), bare_record(2), bare_record(3)])
            .unwrap();
        let iterations = memory
            .records()
            .iter()
            .map(|record| record.iteration)
            .collect::<Vec<_>>();
        assert_eq!(iterations, vec![1, 2, 3]);
        memory.clear();
        assert!(memory.is_empty());
    }

    #[test]
    fn json_line_is_deterministic() {
        let record = full_record(3);
        let first = record.to_json_line().unwrap();
        let second = record.to_json_line().unwrap();
        assert_eq!(first, second);
        assert!(!first.contains('\n'));
        assert!(first.starts_with('{'));
        assert!(first.ends_with('}'));
        assert_eq!(first, serde_json::to_string(&record).unwrap());

        // Field order follows the struct, not a hash map.
        let keys = [
            "iteration",
            "candidate",
            "candidate_prompts",
            "scores",
            "reward",
            "reward_components",
            "reward_notes",
            "best_score",
            "drift",
            "convergence",
            "converged",
            "convergence_reason",
            "adopt",
            "timestamp",
            "run_id",
            "observation",
        ];
        let mut cursor = 0;
        for key in keys {
            let needle = format!("\"{key}\":");
            let found = first[cursor..]
                .find(&needle)
                .unwrap_or_else(|| panic!("missing {needle} after {cursor}"));
            cursor += found;
        }

        assert_eq!(RunLogRecord::from_json_line(&first), Some(record));
    }

    #[test]
    fn serde_defaults_from_a_minimal_object() {
        let line = "{\"iteration\": 3, \"candidate\": {\"prompt\": \"P\", \
                    \"candidate_id\": \"abc12345\", \"rationale\": \"why\"}, \
                    \"unknown_key\": [1, 2]}";
        let record = RunLogRecord::from_json_line(line).unwrap();
        assert_eq!(
            record,
            RunLogRecord::new(3, PromptCandidate::new("P", "why").with_id("abc12345"))
        );
        assert_eq!(record.candidate.candidate_id, "abc12345");
        assert_eq!(record.reward, 0.0);
        assert_eq!(record.timestamp, 0.0);
        assert!(!record.converged);
        assert_eq!(record.adopt, None);
        assert!(!record.is_improvement());

        // A line that shares no keys is an all-default record; anything that is
        // not a JSON object is rejected.
        assert_eq!(
            RunLogRecord::from_json_line("{}"),
            Some(RunLogRecord::default())
        );
        assert_eq!(
            RunLogRecord::from_json_line("{\"stage\": \"round\"}"),
            Some(RunLogRecord::default())
        );
        assert_eq!(RunLogRecord::from_json_line("[]"), None);
        assert_eq!(RunLogRecord::from_json_line("[{}]"), None);
        assert_eq!(RunLogRecord::from_json_line("null"), None);
        assert_eq!(RunLogRecord::from_json_line("3"), None);
        assert_eq!(RunLogRecord::from_json_line("\"text\""), None);
        assert_eq!(RunLogRecord::from_json_line("   "), None);
        assert_eq!(
            RunLogRecord::from_json_line("{\"iteration\": \"three\"}"),
            None
        );
        assert_eq!(RunLogRecord::from_json_line("{\"iteration\": -1}"), None);
    }

    #[test]
    fn parent_directories_are_created() {
        let workspace = TempWorkspace::new().unwrap();
        let path = workspace.path("a/b/c/runs.jsonl");
        assert!(!path.parent().unwrap().exists());

        let mut log = JsonlRunLog::new(&path).unwrap();
        assert!(path.is_file());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "");

        log.record(&bare_record(1)).unwrap();
        assert_eq!(read_run_log(&path).unwrap().len(), 1);

        // The directory is recreated if it disappeared between records.
        std::fs::remove_dir_all(workspace.path("a")).unwrap();
        log.record(&bare_record(2)).unwrap();
        assert_eq!(read_run_log(&path).unwrap().len(), 1);
    }

    #[test]
    fn blank_paths_are_validation_errors() {
        let error = JsonlRunLog::new("").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        let error = JsonlRunLog::create("").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        let error = read_run_log("").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        let error = read_run_log_limited("", 2).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
    }

    #[test]
    fn builders_and_readers_cover_the_optional_fields() {
        let record = RunLogRecord::default();
        assert_eq!(record.iteration, 0);
        assert!(!record.is_improvement());
        assert!(!record.adopted());

        let record = RunLogRecord::new(2, candidate("bare prompt", ""))
            .with_converged(true)
            .with_convergence_reason("target reward reached")
            .with_convergence(ConvergenceState {
                converged: false,
                ..ConvergenceState::default()
            })
            .with_adopt(AdoptDecision::new(
                false,
                AdoptReason::NoImprovement { gain: -0.1 },
            ));
        // `with_convergence` mirrors the state it is given.
        assert!(!record.converged);
        assert_eq!(
            record.convergence_reason.as_deref(),
            Some("target reward reached")
        );
        assert_eq!(record.convergence.as_ref().unwrap().reason, "");
        assert_eq!(
            record.adopt.unwrap().reason,
            AdoptReason::NoImprovement { gain: -0.1 }
        );
        assert!(!record.adopted());

        let record = full_record(1);
        assert_eq!(
            record.reward_notes,
            vec!["correctness gate passed".to_owned()]
        );
    }

    #[cfg(unix)]
    #[test]
    fn write_failures_do_not_fail_the_run() {
        let workspace = TempWorkspace::new().unwrap();
        let path = log_path(&workspace);
        let mut log = JsonlRunLog::new(&path).unwrap();

        // Replace the file with a directory: appending is now an I/O failure.
        std::fs::remove_file(&path).unwrap();
        std::fs::create_dir(&path).unwrap();

        assert!(log.record(&bare_record(1)).is_ok());
        assert!(log.record(&full_record(2)).is_ok());
        assert!(path.is_dir());
    }
}
