//! Resumable single-harness state and its idempotency keys.
//!
//! Ported from the epoch checkpoint of Python
//! `openjiuwen/rsi/harness_rsi/single_harness/iterative.py` (`state.yaml`).
//! One [`SingleHarnessState`] is the whole checkpoint: it carries the
//! fingerprint of the run configuration, the dataset identity, the incumbent,
//! the completed batches, the candidate gate records, the optimisation journal
//! and the usage counters. Restoring it is a full state restore, so a resumed
//! run never replays a completed batch (see [`batch_key`]) and never loses the
//! journal.
//!
//! Keys are strings rather than tuples because they are persisted: a batch is
//! identified by `epoch_{epoch:03}:batch_{batch_index:03}` and an attempt by
//! `<batch_key>:attempt_{attempt:03}`, which keeps the Python idempotency
//! contract (`batch_key` / `attempt_key`) readable in the YAML file.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::fs;

use openjiuwen_common::error::{Error, Result};

use crate::harness_rsi::auto_harness::StopReason;

/// Version of the persisted state layout (Python `version: 7`).
pub const STATE_VERSION: u32 = 7;

/// File name of the persisted state inside an output directory.
pub const STATE_FILE_NAME: &str = "single_harness_state.yaml";

/// Which promotion policy produced (or should consume) a state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromotionPolicyMode {
    /// Epoch-checkpointed batches with the legacy "promote if it beats the
    /// incumbent" rule (`iterative.py`).
    LegacyEpochV1,
    /// Per-target improvement followed by one independent full evaluation
    /// (`continuous.py`, the Python default).
    #[default]
    ContinuousTwoStageV2,
}

impl PromotionPolicyMode {
    /// Returns the stable token used inside configuration fingerprints.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::LegacyEpochV1 => "legacy_epoch_v1",
            Self::ContinuousTwoStageV2 => "continuous_two_stage_v2",
        }
    }
}

/// Lifecycle status of a single-harness run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SingleHarnessStatus {
    /// The run is in progress (also the status of a resumed run).
    #[default]
    Running,
    /// The run reached a stop reason and finished normally.
    Completed,
    /// The run aborted; the state records why.
    Failed,
    /// The caller cancelled the run.
    Cancelled,
}

impl SingleHarnessStatus {
    /// Returns whether the status is terminal.
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        !matches!(self, Self::Running)
    }
}

/// Usage counters accumulated by the run.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct UsageState {
    /// Number of evaluator invocations.
    pub calls: u64,
    /// Cost reported by the evaluator, in US dollars.
    pub cost_usd: f64,
}

/// One candidate gate outcome, kept for auditability.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct CandidateGateRecord {
    /// Idempotency key of the batch this candidate belongs to.
    pub batch_key: String,
    /// Candidate id assigned by the improver.
    pub candidate_id: String,
    /// Whether the candidate passed the stage-one gate.
    pub passed: bool,
    /// Gate reason literal (see the `REASON_*` constants in
    /// [`super::promotion`]).
    pub reason: String,
    /// 1-based attempt number inside the batch.
    pub round: u32,
}

/// One batch decision, kept so a resumed run can explain its history.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct JournalEntry {
    /// Epoch the batch belongs to (1-based).
    pub epoch: u32,
    /// Batch index inside the epoch (0-based).
    pub batch_index: u32,
    /// Decision reason literal.
    pub reason: String,
    /// Candidate score that justified the decision, when one was graded.
    pub score: Option<f64>,
    /// Whether the candidate replaced the incumbent.
    pub promoted: bool,
}

/// The whole persisted state of one single-harness run.
///
/// Field names follow the Python checkpoint so a state file stays readable by
/// both implementations; unknown keys are tolerated on load.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SingleHarnessState {
    /// Layout version, always [`STATE_VERSION`] for a state written here.
    pub version: u32,
    /// Lifecycle status.
    pub status: SingleHarnessStatus,
    /// Promotion policy the run uses.
    pub mode: PromotionPolicyMode,
    /// Configuration fingerprint the run was started with.
    pub fingerprint: String,
    /// Dataset (case ids) the run optimises, in evaluation order.
    pub dataset: Vec<String>,
    /// Path of the harness references of the incumbent, when it has one.
    pub best_harness_refs_path: Option<String>,
    /// Best overall score found so far.
    pub best_score: Option<f64>,
    /// Body (configuration or source) of the incumbent.
    pub best_body: String,
    /// Score of the untouched seed candidate over the whole case set.
    pub baseline_score: Option<f64>,
    /// Idempotency keys of the batches that were fully processed.
    pub completed_batches: Vec<String>,
    /// One record per evaluated candidate.
    pub candidate_gates: Vec<CandidateGateRecord>,
    /// One record per batch decision.
    pub optimization_journal: Vec<JournalEntry>,
    /// Highest score ever observed, even if it was not promoted.
    pub historical_peak_score: Option<f64>,
    /// Attempts used inside the current batch (continuous policy).
    pub continuous_attempts: u32,
    /// Candidate attempts used across the whole run (continuous policy).
    pub continuous_trials: u32,
    /// Iteration cap the state is valid for; a resume must match it.
    pub max_iteration: u32,
    /// Task identity the state belongs to.
    pub task_id: String,
    /// Evaluation protocol signature the scores were produced with.
    pub evaluation_protocol_signature: String,
    /// Why the run stopped; `None` while it is still running.
    pub stop_reason: Option<StopReason>,
    /// Usage accumulated so far.
    pub usage: UsageState,
    /// Free-form extras (the crate stores the incumbent per-case scores here).
    pub metadata: BTreeMap<String, Value>,
}

impl Default for SingleHarnessState {
    fn default() -> Self {
        Self {
            version: STATE_VERSION,
            status: SingleHarnessStatus::Running,
            mode: PromotionPolicyMode::default(),
            fingerprint: String::new(),
            dataset: Vec::new(),
            best_harness_refs_path: None,
            best_score: None,
            best_body: String::new(),
            baseline_score: None,
            completed_batches: Vec::new(),
            candidate_gates: Vec::new(),
            optimization_journal: Vec::new(),
            historical_peak_score: None,
            continuous_attempts: 0,
            continuous_trials: 0,
            max_iteration: 0,
            task_id: String::new(),
            evaluation_protocol_signature: String::new(),
            stop_reason: None,
            usage: UsageState::default(),
            metadata: BTreeMap::new(),
        }
    }
}

/// Builds the idempotency key of one batch.
///
/// The key is `epoch_{epoch:03}:batch_{batch_index:03}`, so it sorts by epoch
/// first and stays stable across process restarts.
#[must_use]
pub fn batch_key(epoch: u32, batch_index: u32) -> String {
    format!("epoch_{epoch:03}:batch_{batch_index:03}")
}

/// Builds the idempotency key of one improvement attempt inside a batch.
///
/// The key is `"<batch_key>:attempt_{attempt:03}"`.
#[must_use]
pub fn attempt_key(batch_key: &str, attempt: u32) -> String {
    format!("{batch_key}:attempt_{attempt:03}")
}

/// Returns the path of the persisted state inside `output_dir`.
#[must_use]
pub fn state_path(output_dir: impl AsRef<Path>) -> PathBuf {
    output_dir.as_ref().join(STATE_FILE_NAME)
}

impl SingleHarnessState {
    /// Creates the state of a fresh run.
    #[must_use]
    pub fn new(
        task_id: impl Into<String>,
        fingerprint: impl Into<String>,
        dataset: Vec<String>,
        max_iteration: u32,
        mode: PromotionPolicyMode,
    ) -> Self {
        Self {
            version: STATE_VERSION,
            status: SingleHarnessStatus::Running,
            mode,
            fingerprint: fingerprint.into(),
            dataset,
            max_iteration,
            task_id: task_id.into(),
            ..Self::default()
        }
    }

    /// Returns whether `key` names a batch that was already processed.
    #[must_use]
    pub fn is_batch_completed(&self, key: &str) -> bool {
        self.completed_batches.iter().any(|item| item == key)
    }

    /// Records `key` as a processed batch (idempotent).
    pub fn mark_batch_completed(&mut self, key: &str) {
        if !self.is_batch_completed(key) {
            self.completed_batches.push(key.to_owned());
        }
    }

    /// Returns the number of journal entries (one per attempted batch).
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn iterations(&self) -> u32 {
        self.optimization_journal.len() as u32
    }

    /// Returns how many trailing journal entries did not promote.
    ///
    /// This is the convergence signal: a run that keeps rejecting candidates
    /// eventually exhausts its patience.
    #[must_use]
    pub fn consecutive_non_promotions(&self) -> usize {
        self.optimization_journal
            .iter()
            .rev()
            .take_while(|entry| !entry.promoted)
            .count()
    }

    /// Resumes `previous` as the state of a new run.
    ///
    /// The caller must prove it is resuming the *same* run: the configuration
    /// fingerprint, the task id, the iteration cap and the evaluation protocol
    /// signature all have to match, otherwise the persisted scores would be
    /// compared against a different protocol. A mismatch names the offending
    /// field in the error message.
    ///
    /// On success the status becomes [`SingleHarnessStatus::Running`], the stop
    /// reason is cleared, and the journal, completed batches, incumbent and
    /// usage are preserved.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the fingerprint, task id, iteration
    /// cap or protocol signature differ from the current run.
    pub fn resume_from(
        previous: Self,
        fingerprint: &str,
        task_id: &str,
        max_iteration: u32,
        signature: &str,
    ) -> Result<Self> {
        if previous.fingerprint != fingerprint {
            return Err(Error::validation(format!(
                "single-harness state fingerprint mismatch: expected `{fingerprint}`, found `{}`",
                previous.fingerprint
            )));
        }
        if previous.task_id != task_id {
            return Err(Error::validation(format!(
                "single-harness state task_id mismatch: expected `{task_id}`, found `{}`",
                previous.task_id
            )));
        }
        if previous.max_iteration != max_iteration {
            return Err(Error::validation(format!(
                "single-harness state max_iteration mismatch: expected `{max_iteration}`, found `{}`",
                previous.max_iteration
            )));
        }
        if previous.evaluation_protocol_signature != signature {
            return Err(Error::validation(format!(
                "single-harness state evaluation_protocol_signature mismatch: \
                 expected `{signature}`, found `{}`",
                previous.evaluation_protocol_signature
            )));
        }
        let mut state = previous;
        state.status = SingleHarnessStatus::Running;
        state.stop_reason = None;
        Ok(state)
    }

    /// Serialises the state to `path` as YAML, creating the file's parent
    /// directory when it does not exist yet.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Session`] when the parent directory cannot be created,
    /// the state cannot be serialised, or the file cannot be written.
    pub async fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).await.map_err(|err| {
                    Error::session(format!(
                        "cannot create single-harness state directory `{}`: {err}",
                        parent.display()
                    ))
                })?;
            }
        }
        let encoded = serde_yaml_ng::to_string(self).map_err(|err| {
            Error::session(format!("cannot serialise single-harness state: {err}"))
        })?;
        fs::write(path, encoded).await.map_err(|err| {
            Error::session(format!(
                "cannot write single-harness state `{}`: {err}",
                path.display()
            ))
        })
    }

    /// Loads a state from `path`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Session`] when the file cannot be read or does not
    /// contain a valid single-harness state.
    pub async fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let encoded = fs::read_to_string(path).await.map_err(|err| {
            Error::session(format!(
                "cannot read single-harness state `{}`: {err}",
                path.display()
            ))
        })?;
        serde_yaml_ng::from_str(&encoded).map_err(|err| {
            Error::session(format!(
                "cannot parse single-harness state `{}`: {err}",
                path.display()
            ))
        })
    }
}
