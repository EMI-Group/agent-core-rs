//! The training-run state machine: sample claim, execution, activation, stop and
//! crash recovery.
//!
//! Python parity: `agent_rl/online/training_runner.py` (`TrainingRunner`,
//! `TrainingRunRecord`, `RunStatus`, `RunStage`, `PPOExecutor`, `LoRAActivator`).
//! Python drives the run with an `asyncio.Task` plus a Redis-store wrapper and
//! WATCH/MULTI transactions; the port keeps the same durable states and rules on
//! top of [`OnlineStore`] primitives plus the [`TrainingExecutor`] and
//! [`LoraActivator`] seams.
//!
//! # Run lifecycle
//!
//! ```text
//!   claim                         run                      auto_activate_lora
//!  (pending/queued) ──▶ running/training ──▶ running/activating ──▶ succeeded
//!        │                    │                     │
//!        │ stop               │ stop                │ stop
//!        ▼                    ▼                     ▼
//!    canceled            canceled             canceled (samples stay trained)
//!                             │ executor or name-validation failure
//!                             ▼
//!                          failed  (samples return to pending)
//! ```
//!
//! | From | Event | To | Sample disposition | Error on refusal |
//! |---|---|---|---|---|
//! | — | `claim`, no active run, `pending >= min` | `pending/queued` | `pending` → `training` | [`OnlineError::InsufficientSamples`] when `pending < min` |
//! | — | `claim`, active run exists | unchanged | untouched | — (`created = false`) |
//! | `pending/queued` | `run` | `running/training` → `running/activating` → `succeeded` | `training` → `trained` | [`OnlineError::TaskConflict`] for a non-queued stage |
//! | `running/training` | executor error, artifact-name violation | `failed` | `training` → `pending` | — |
//! | `running/activating` | activation error | `failed` | stays `trained` | — |
//! | `pending`/`running` | `stop` (stage queued/training) | `canceled` | `training` → `pending` | executor `cancel` error propagates while training |
//! | `pending`/`running` | `stop` (stage activating) | `canceled` | stays `trained` | — |
//! | `succeeded`/`failed`/`canceled` | `stop`, `run` | unchanged | untouched | — |
//!
//! # Single flight
//!
//! `rl:v1:training_run:active` names the one run the service may execute. `claim`
//! writes the run record first and then wins the marker with
//! [`OnlineStore::set_if_absent`]; the loser deletes its own record, returns the
//! winner's run with `created = false`, and gives its samples back to `pending`.
//! That ordering means a crash between the two writes leaves a harmless orphan
//! record, never an active marker without a run.
//!
//! # Determinism
//!
//! Run ids come from the `rl:v1:training_run:seq` counter (`run-{seq:012x}`),
//! timestamps come from the injected [`Clock`], and sample batches are ordered by
//! `(created_seq, sample_id)`. Nothing here reads the wall clock, a random number
//! or a filesystem mtime.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result as KernelResult};
use serde::{Deserialize, Serialize};

use super::error::{OnlineError, Result};
use super::executor::{TrainingArtifact, TrainingExecutor, TrainingRequest};
use super::json;
use super::sample::{SampleLedger, SampleRecord, SampleStatus};
use super::store::{keys, OnlineStore};
use super::task::{Clock, FixedClock};
use crate::rl::RlTrainingConfig;

/// `failure_reason` of a run the restart rule had to fail.
pub const SERVICE_RESTARTED: &str = "service_restarted";

/// Schema version of a [`TrainingCheckpoint`] file.
pub const TRAINING_CHECKPOINT_VERSION: &str = "training-checkpoint/v1";

/// Public lifecycle state of a training run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    /// Claimed, not started.
    #[default]
    Pending,
    /// Owned by an executor (or between training and activation).
    Running,
    /// Artifact exported (and activated, when auto-activation is on).
    Succeeded,
    /// Training, name validation or activation failed.
    Failed,
    /// Stopped by the service or the user.
    Canceled,
}

impl RunStatus {
    /// The Python wire literal of the status.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Canceled => "canceled",
        }
    }

    /// Whether the status is final (no transition leaves it).
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed | Self::Canceled)
    }
}

/// Durable recovery boundary of a training run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStage {
    /// Claimed; the executor has not been entered.
    #[default]
    Queued,
    /// The executor owns the sample batch.
    Training,
    /// The artifact exists; the run is being activated.
    Activating,
}

impl RunStage {
    /// The Python wire literal of the stage.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Training => "training",
            Self::Activating => "activating",
        }
    }
}

/// Public durable state of one fixed sample batch.
///
/// The first nine fields are the wire shape served over HTTP (Python's
/// `public_dict()`, implemented here with `serde(skip_serializing)` on the three
/// recovery-only fields); the full record — including `sample_ids` and the parent
/// policy — is what `to_json` persists (Python's `to_json`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TrainingRunRecord {
    /// Run id (`run-{seq:012x}`).
    pub training_run_id: String,
    /// Lifecycle state.
    pub status: RunStatus,
    /// Recovery boundary.
    pub stage: RunStage,
    /// Samples the batch holds.
    pub sample_count: usize,
    /// Samples per `policy_version` the batch was rolled out on.
    pub policy_versions: BTreeMap<String, u32>,
    /// Creation stamp from the runner's [`Clock`].
    pub created_at: String,
    /// Execution start stamp.
    pub started_at: Option<String>,
    /// Terminal stamp.
    pub finished_at: Option<String>,
    /// Artifact name of a succeeded run.
    pub lora_name: Option<String>,
    /// Artifact path of a succeeded run.
    pub lora_path: Option<String>,
    /// Why the run failed.
    pub failure_reason: Option<String>,
    /// Sample ids of the batch (recovery only).
    #[serde(skip_serializing)]
    pub sample_ids: Vec<String>,
    /// Parent policy name the batch started from (recovery only).
    #[serde(skip_serializing)]
    pub parent_lora_name: Option<String>,
    /// Parent policy path the batch started from (recovery only).
    #[serde(skip_serializing)]
    pub parent_lora_path: Option<String>,
}

impl TrainingRunRecord {
    /// Whether the run still owns a batch (the Python `_ACTIVE_STATUSES` set).
    #[must_use]
    pub const fn is_active(&self) -> bool {
        matches!(self.status, RunStatus::Pending | RunStatus::Running)
    }

    /// Whether the run reached a final state.
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        self.status.is_terminal()
    }

    /// The parent policy name, `"base"` when the run started from scratch.
    #[must_use]
    pub fn parent_lora_name(&self) -> &str {
        self.parent_lora_name.as_deref().unwrap_or("base")
    }

    /// The parent policy path, empty when the run started from scratch.
    #[must_use]
    pub fn parent_lora_path(&self) -> &str {
        self.parent_lora_path.as_deref().unwrap_or_default()
    }

    /// The public HTTP record: the recovery-only fields stripped.
    #[must_use]
    pub fn public(&self) -> Self {
        Self {
            sample_ids: Vec::new(),
            parent_lora_name: None,
            parent_lora_path: None,
            ..self.clone()
        }
    }

    /// Serializes the public shape (Python `public_dict()`).
    pub fn to_public_json(&self) -> KernelResult<String> {
        serde_json::to_string(self).map_err(Error::external)
    }

    /// Serializes the complete durable record (Python `to_json`).
    pub fn to_json(&self) -> Result<String> {
        json::encode(&TrainingRunWire::from(self))
    }

    /// Restores a record from its durable representation.
    ///
    /// The public shape also parses — its recovery-only fields default to empty —
    /// so a checkpoint file round-trips through the same reader.
    pub fn from_json(payload: &str) -> Result<Self> {
        let wire: TrainingRunWire = json::decode(payload)?;
        Ok(wire.into())
    }
}

/// Durable mirror of [`TrainingRunRecord`] that keeps the recovery-only fields.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
struct TrainingRunWire {
    training_run_id: String,
    status: RunStatus,
    stage: RunStage,
    sample_count: usize,
    policy_versions: BTreeMap<String, u32>,
    created_at: String,
    started_at: Option<String>,
    finished_at: Option<String>,
    lora_name: Option<String>,
    lora_path: Option<String>,
    failure_reason: Option<String>,
    sample_ids: Vec<String>,
    parent_lora_name: Option<String>,
    parent_lora_path: Option<String>,
}

impl From<&TrainingRunRecord> for TrainingRunWire {
    fn from(record: &TrainingRunRecord) -> Self {
        Self {
            training_run_id: record.training_run_id.clone(),
            status: record.status,
            stage: record.stage,
            sample_count: record.sample_count,
            policy_versions: record.policy_versions.clone(),
            created_at: record.created_at.clone(),
            started_at: record.started_at.clone(),
            finished_at: record.finished_at.clone(),
            lora_name: record.lora_name.clone(),
            lora_path: record.lora_path.clone(),
            failure_reason: record.failure_reason.clone(),
            sample_ids: record.sample_ids.clone(),
            parent_lora_name: record.parent_lora_name.clone(),
            parent_lora_path: record.parent_lora_path.clone(),
        }
    }
}

impl From<TrainingRunWire> for TrainingRunRecord {
    fn from(wire: TrainingRunWire) -> Self {
        Self {
            training_run_id: wire.training_run_id,
            status: wire.status,
            stage: wire.stage,
            sample_count: wire.sample_count,
            policy_versions: wire.policy_versions,
            created_at: wire.created_at,
            started_at: wire.started_at,
            finished_at: wire.finished_at,
            lora_name: wire.lora_name,
            lora_path: wire.lora_path,
            failure_reason: wire.failure_reason,
            sample_ids: wire.sample_ids,
            parent_lora_name: wire.parent_lora_name,
            parent_lora_path: wire.parent_lora_path,
        }
    }
}

/// Fixed policy of a training runner.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TrainingRunnerConfig {
    /// Model identity every artifact must be named after.
    pub model_id: String,
    /// Pending samples a claim needs before it starts a run.
    pub min_samples_for_training: usize,
    /// Largest batch one run may claim.
    pub max_samples_per_run: usize,
    /// Activate the artifact through the [`LoraActivator`] when training succeeds.
    pub auto_activate_lora: bool,
    /// Batch size handed to the executor's [`RlTrainingConfig`].
    pub batch_size: usize,
    /// Root the executor writes artifacts under, when it has no root of its own.
    pub output_dir: Option<PathBuf>,
}

impl Default for TrainingRunnerConfig {
    fn default() -> Self {
        Self {
            model_id: String::new(),
            min_samples_for_training: 1,
            max_samples_per_run: 64,
            auto_activate_lora: true,
            batch_size: 8,
            output_dir: None,
        }
    }
}

impl TrainingRunnerConfig {
    /// Creates a configuration for `model_id` with the remaining defaults.
    #[must_use]
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            ..Self::default()
        }
    }

    /// Rejects a blank model, a zero minimum, and an inverted batch range.
    ///
    /// Python raises `AGENT_RL_SERVICE_PARAM_ERROR` from `TrainingRunner.__init__`;
    /// the port validates at the first `claim`, because the constructor stays
    /// infallible for embedders that build the runner before their config is final.
    pub fn validate(&self) -> Result<()> {
        if self.model_id.trim().is_empty() {
            return Err(OnlineError::InvalidRequest(
                "model_id is required".to_owned(),
            ));
        }
        if self.min_samples_for_training == 0 {
            return Err(OnlineError::InvalidRequest(
                "min_samples_for_training must be at least 1".to_owned(),
            ));
        }
        if self.max_samples_per_run < self.min_samples_for_training {
            return Err(OnlineError::InvalidRequest(
                "max_samples_per_run must be >= min_samples_for_training".to_owned(),
            ));
        }
        if self.batch_size == 0 {
            return Err(OnlineError::InvalidRequest(
                "batch_size must be at least 1".to_owned(),
            ));
        }
        Ok(())
    }
}

/// The artifact a training run's executor exported.
#[async_trait]
pub trait LoraActivator: Send + Sync {
    /// Activates (or idempotently confirms) `request`'s artifact.
    async fn activate(&self, request: &LoraActivation) -> KernelResult<()>;
}

/// One activation request handed to a [`LoraActivator`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct LoraActivation {
    /// Model identity the artifact belongs to.
    pub base_model: String,
    /// Artifact name to activate.
    pub lora_name: String,
    /// Artifact path to activate.
    pub lora_path: String,
    /// Policy name that must currently be active (the run's parent policy).
    pub expected_lora_name: String,
    /// Run the activation belongs to.
    pub training_run_id: String,
}

/// A [`LoraActivator`] that records the activation instead of performing it.
///
/// It writes the request under `rl:v1:training_run:{id}:activation`, which makes
/// activation observable and deterministic for tests and local runs without a
/// model server.
#[derive(Debug, Clone)]
pub struct NoopLoraActivator<S: OnlineStore> {
    store: S,
}

impl<S: OnlineStore> NoopLoraActivator<S> {
    /// Creates the activator over `store`.
    pub fn new(store: S) -> Self {
        Self { store }
    }

    /// The backing store.
    #[must_use]
    pub fn store(&self) -> &S {
        &self.store
    }

    /// The activation recorded for `training_run_id`, if any.
    pub async fn recorded(&self, training_run_id: &str) -> KernelResult<Option<LoraActivation>> {
        match self
            .store
            .get(&keys::training_activation(training_run_id))
            .await?
        {
            Some(payload) => Ok(Some(
                serde_json::from_str(&payload).map_err(Error::external)?,
            )),
            None => Ok(None),
        }
    }
}

#[async_trait]
impl<S: OnlineStore> LoraActivator for NoopLoraActivator<S> {
    async fn activate(&self, request: &LoraActivation) -> KernelResult<()> {
        let encoded = serde_json::to_string(request).map_err(Error::external)?;
        self.store
            .set(
                &keys::training_activation(&request.training_run_id),
                &encoded,
            )
            .await
    }
}

/// The vendor activation path, retained as a typed seam.
///
/// Python calls the AI gateway (AIGW) to swap the served LoRA; that transport is
/// vendor HTTP and is not ported, so every activation fails with a reason that
/// names it. The artifact is still exported and recorded — only the switch is
/// missing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedLoraActivator {
    gateway: String,
}

impl UnsupportedLoraActivator {
    /// Creates the activator for the default gateway name.
    #[must_use]
    pub fn new() -> Self {
        Self::with_gateway("vendor AI gateway (AIGW)")
    }

    /// Creates the activator for an explicitly named gateway.
    #[must_use]
    pub fn with_gateway(gateway: impl Into<String>) -> Self {
        Self {
            gateway: gateway.into(),
        }
    }

    /// The gateway this activator would have called.
    #[must_use]
    pub fn gateway(&self) -> &str {
        &self.gateway
    }
}

impl Default for UnsupportedLoraActivator {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LoraActivator for UnsupportedLoraActivator {
    async fn activate(&self, request: &LoraActivation) -> KernelResult<()> {
        Err(Error::unsupported(format!(
            "LoRA activation is not available in this build: switching {} requires the {} HTTP \
             service (run {})",
            request.lora_name, self.gateway, request.training_run_id
        )))
    }
}

/// A portable export of one training run.
///
/// [`crate::rl::RlCheckpoint`] is the epoch checkpoint of the offline trainer
/// (policy blob, best reward, per-epoch metrics); an online run has a fixed
/// batch, a policy version and an artifact instead, so the export gets its own
/// record rather than a lossy mapping into that one.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TrainingCheckpoint {
    /// Schema version; always [`TRAINING_CHECKPOINT_VERSION`] for files written here.
    pub version: String,
    /// Run the checkpoint belongs to.
    pub training_run_id: String,
    /// Model identity of the run.
    pub model_id: String,
    /// Lifecycle state at export time.
    pub status: RunStatus,
    /// Recovery boundary at export time.
    pub stage: RunStage,
    /// Run creation stamp.
    pub created_at: String,
    /// Run terminal stamp.
    pub finished_at: Option<String>,
    /// Samples the batch held.
    pub sample_count: usize,
    /// Sample ids of the batch.
    pub sample_ids: Vec<String>,
    /// Samples per `policy_version`.
    pub policy_versions: BTreeMap<String, u32>,
    /// Artifact name.
    pub lora_name: Option<String>,
    /// Artifact path.
    pub lora_path: Option<String>,
    /// Failure reason, when the run failed.
    pub failure_reason: Option<String>,
    /// The executor's artifact descriptor, when the run exported one.
    pub artifact: Option<TrainingArtifact>,
}

impl TrainingCheckpoint {
    /// Builds the checkpoint of `run`, with `model_id` and its artifact.
    #[must_use]
    pub fn from_run(
        model_id: &str,
        run: &TrainingRunRecord,
        artifact: Option<TrainingArtifact>,
    ) -> Self {
        Self {
            version: TRAINING_CHECKPOINT_VERSION.to_owned(),
            training_run_id: run.training_run_id.clone(),
            model_id: model_id.to_owned(),
            status: run.status,
            stage: run.stage,
            created_at: run.created_at.clone(),
            finished_at: run.finished_at.clone(),
            sample_count: run.sample_count,
            sample_ids: run.sample_ids.clone(),
            policy_versions: run.policy_versions.clone(),
            lora_name: run.lora_name.clone(),
            lora_path: run.lora_path.clone(),
            failure_reason: run.failure_reason.clone(),
            artifact,
        }
    }

    /// File name of the checkpoint in its export directory.
    #[must_use]
    pub fn file_name(&self) -> String {
        format!("training-run-{}.json", self.training_run_id)
    }

    /// Rejects an unknown schema version and a blank run id.
    pub fn validate(&self) -> KernelResult<()> {
        if self.version != TRAINING_CHECKPOINT_VERSION {
            return Err(Error::validation(format!(
                "unsupported TrainingCheckpoint version {:?}, expected {TRAINING_CHECKPOINT_VERSION:?}",
                self.version
            )));
        }
        if self.training_run_id.trim().is_empty() {
            return Err(Error::validation(
                "TrainingCheckpoint training_run_id must not be empty",
            ));
        }
        Ok(())
    }

    /// Writes the checkpoint into `dir` and returns its path.
    ///
    /// The directory is created when missing and the file is written atomically
    /// (temp + rename), so a reader never sees a half-written export.
    pub async fn save_to_dir(&self, dir: &Path) -> KernelResult<PathBuf> {
        self.validate()?;
        tokio::fs::create_dir_all(dir)
            .await
            .map_err(Error::external)?;
        let encoded = serde_json::to_vec_pretty(self).map_err(Error::external)?;
        let path = dir.join(self.file_name());
        let mut temp_name = std::ffi::OsString::from(".");
        temp_name.push(self.file_name());
        temp_name.push(".tmp");
        let temp_path = dir.join(temp_name);
        tokio::fs::write(&temp_path, &encoded)
            .await
            .map_err(Error::external)?;
        tokio::fs::rename(&temp_path, &path)
            .await
            .map_err(Error::external)?;
        Ok(path)
    }

    /// Loads and validates a checkpoint file.
    pub async fn load_from_path(path: &Path) -> KernelResult<Self> {
        let encoded = tokio::fs::read(path).await.map_err(Error::external)?;
        let checkpoint: Self = serde_json::from_slice(&encoded).map_err(Error::external)?;
        checkpoint.validate()?;
        Ok(checkpoint)
    }
}

/// Owns sample claim, execution, activation, cancellation and crash recovery.
///
/// The runner is stateless: all state lives in the [`OnlineStore`] plus the two
/// seams, so two runners over one store agree, and a restarted process resumes
/// through [`TrainingRunner::recover`].
pub struct TrainingRunner<S: OnlineStore> {
    store: S,
    clock: Arc<dyn Clock>,
    executor: Arc<dyn TrainingExecutor>,
    activator: Arc<dyn LoraActivator>,
    config: TrainingRunnerConfig,
    parent_policy: Option<(String, String)>,
}

impl<S: OnlineStore> TrainingRunner<S> {
    /// Creates a runner over `store` with the deterministic [`FixedClock`].
    pub fn new(
        store: S,
        executor: Arc<dyn TrainingExecutor>,
        activator: Arc<dyn LoraActivator>,
        config: TrainingRunnerConfig,
    ) -> Self {
        Self {
            store,
            clock: Arc::new(FixedClock::default()),
            executor,
            activator,
            config,
            parent_policy: None,
        }
    }

    /// Stamps records with `clock` instead of the fixed epoch.
    #[must_use]
    pub fn with_clock(mut self, clock: Arc<dyn Clock>) -> Self {
        self.clock = clock;
        self
    }

    /// Sets the parent policy a new run starts from (`base`/empty by default).
    ///
    /// Python reads it from an `active_policy` callable — in practice the LoRA
    /// repository's [`get_latest_available`](crate::online::LoraRepository::get_latest_available).
    /// The port takes the snapshot explicitly so a run never depends on when the
    /// repository was last written.
    #[must_use]
    pub fn with_parent_policy(
        mut self,
        lora_name: impl Into<String>,
        lora_path: impl Into<String>,
    ) -> Self {
        self.parent_policy = Some((lora_name.into(), lora_path.into()));
        self
    }

    /// The backing store.
    #[must_use]
    pub fn store(&self) -> &S {
        &self.store
    }

    /// The fixed configuration.
    #[must_use]
    pub fn config(&self) -> &TrainingRunnerConfig {
        &self.config
    }

    /// The executor the runner drives.
    #[must_use]
    pub fn executor(&self) -> &Arc<dyn TrainingExecutor> {
        &self.executor
    }

    /// Claims one fixed sample batch, or returns the already-active run.
    ///
    /// `created = false` means another caller owns the single active run and its
    /// batch was returned instead; the samples of this call were given back to
    /// `pending`.
    pub async fn claim(&self) -> Result<(TrainingRunRecord, bool, Vec<SampleRecord>)> {
        self.config.validate()?;
        if let Some(active) = self.active_run().await? {
            return Ok((active, false, Vec::new()));
        }

        let ledger = self.ledger();
        let pending = ledger.count_by_status(SampleStatus::Pending).await?;
        let need = self.config.min_samples_for_training;
        if pending < need {
            return Err(OnlineError::InsufficientSamples {
                have: pending,
                need,
            });
        }
        let limit = pending.min(self.config.max_samples_per_run);
        let samples = ledger.fetch_and_mark_training(limit).await?;
        if samples.len() < need {
            // Another writer drained part of the queue between the count and the
            // fetch; give the batch back rather than start an undersized run.
            let sample_ids = sample_ids(&samples);
            ledger.reset_to_pending(&sample_ids).await?;
            return Err(OnlineError::InsufficientSamples {
                have: samples.len(),
                need,
            });
        }

        let mut claimed = None;
        for _ in 0..8 {
            let candidate = self.build_run(&samples).await?;
            if self.store_run(&candidate).await? {
                claimed = Some(candidate);
                break;
            }
        }
        let Some(run) = claimed else {
            let sample_ids = sample_ids(&samples);
            ledger.reset_to_pending(&sample_ids).await?;
            return Err(OnlineError::TaskConflict(
                "cannot allocate a training run id: run sequence is exhausted".to_owned(),
            ));
        };
        if !self
            .write_if_absent(&keys::active_training_run(), &run.training_run_id)
            .await?
        {
            // Lost the single-flight race: drop the orphan run and requeue.
            self.remove(&keys::training_run(&run.training_run_id))
                .await?;
            self.remove_member(&keys::training_runs(), &run.training_run_id)
                .await?;
            let sample_ids = run.sample_ids.clone();
            ledger.reset_to_pending(&sample_ids).await?;
            let winner = self.active_run().await?.ok_or_else(|| {
                OnlineError::TaskConflict(
                    "training run claim was lost to an invisible active run".to_owned(),
                )
            })?;
            return Ok((winner, false, Vec::new()));
        }
        Ok((run, true, samples))
    }

    /// Returns a training run by id.
    pub async fn get(&self, training_run_id: &str) -> Result<TrainingRunRecord> {
        self.load(training_run_id)
            .await?
            .ok_or_else(|| missing_run(training_run_id))
    }

    /// Returns the single active run, if the service has one.
    ///
    /// A marker that points at a terminal or vanished run is stale (a crash
    /// between finishing a run and clearing the marker); it is cleared here so
    /// the next claim can proceed.
    pub async fn active_run(&self) -> Result<Option<TrainingRunRecord>> {
        let Some(training_run_id) = self.read(&keys::active_training_run()).await? else {
            return Ok(None);
        };
        match self.load(&training_run_id).await? {
            Some(run) if run.is_active() => Ok(Some(run)),
            _ => {
                self.remove(&keys::active_training_run()).await?;
                Ok(None)
            }
        }
    }

    /// Drives one queued run to a terminal state.
    ///
    /// A terminal run is returned unchanged (idempotent); a run that already
    /// left the queue refuses with [`OnlineError::TaskConflict`]. Every step that
    /// awaits re-reads the durable record, so a concurrent [`TrainingRunner::stop`]
    /// always wins over the in-flight run.
    pub async fn run(&self, training_run_id: &str) -> Result<TrainingRunRecord> {
        let run = self.get(training_run_id).await?;
        if run.is_terminal() {
            return Ok(run);
        }
        if run.status != RunStatus::Pending || run.stage != RunStage::Queued {
            return Err(OnlineError::TaskConflict(format!(
                "training run is not queued: {training_run_id}"
            )));
        }

        let sample_ids = run.sample_ids.clone();
        let samples = self.load_samples(&sample_ids).await?;
        let running = TrainingRunRecord {
            status: RunStatus::Running,
            stage: RunStage::Training,
            started_at: Some(self.clock.now()),
            ..run
        };
        self.save(&running).await?;

        let request = self.training_request(&running, samples);
        let artifact = match self.executor.train_batch(&request).await {
            Ok(artifact) => artifact,
            Err(error) => {
                if let Some(stopped) = self.stopped(&running).await? {
                    return Ok(stopped);
                }
                self.restore_samples(&sample_ids).await?;
                return self
                    .finish(&running, RunStatus::Failed, Some(error.to_string()))
                    .await;
            }
        };
        if let Some(stopped) = self.stopped(&running).await? {
            return Ok(stopped);
        }
        if let Err(error) = artifact.validate_name(&self.config.model_id) {
            self.restore_samples(&sample_ids).await?;
            return self
                .finish(&running, RunStatus::Failed, Some(error.to_string()))
                .await;
        }

        let activating = TrainingRunRecord {
            status: RunStatus::Running,
            stage: RunStage::Activating,
            lora_name: Some(artifact.lora_name.clone()),
            lora_path: Some(artifact.lora_path.clone()),
            ..running
        };
        self.ledger().mark_trained(&sample_ids).await?;
        self.save(&activating).await?;
        self.write(
            &keys::training_artifact(training_run_id),
            &json::encode(&artifact)?,
        )
        .await?;

        if let Some(stopped) = self.stopped(&activating).await? {
            return Ok(stopped);
        }
        if !self.config.auto_activate_lora {
            return self.finish(&activating, RunStatus::Succeeded, None).await;
        }
        let activation = self.activation(&activating);
        if let Err(error) = self.activator.activate(&activation).await {
            if let Some(stopped) = self.stopped(&activating).await? {
                return Ok(stopped);
            }
            return self
                .finish(&activating, RunStatus::Failed, Some(error.to_string()))
                .await;
        }
        if let Some(stopped) = self.stopped(&activating).await? {
            return Ok(stopped);
        }
        self.finish(&activating, RunStatus::Succeeded, None).await
    }

    /// Cancels an active run and returns its durable record.
    ///
    /// The executor owns the batch only while the stage is `training`, so `cancel`
    /// is issued exactly then; a queued (not yet started) or activating run is
    /// canceled in the store alone. Samples claimed but not trained return to
    /// `pending`; samples consumed by the executor stay `trained`.
    pub async fn stop(&self, training_run_id: &str) -> Result<TrainingRunRecord> {
        let run = self.get(training_run_id).await?;
        if run.is_terminal() {
            return Ok(run);
        }
        if run.stage == RunStage::Training {
            self.executor
                .cancel(training_run_id)
                .await
                .map_err(OnlineError::from)?;
        }
        let current = self.get(training_run_id).await?;
        if current.is_terminal() {
            return Ok(current);
        }
        if matches!(current.stage, RunStage::Queued | RunStage::Training) {
            self.restore_samples(&current.sample_ids).await?;
        }
        self.finish(&current, RunStatus::Canceled, None).await
    }

    /// Applies the fixed restart rule to every non-terminal run.
    ///
    /// A run that never reached activation loses its batch and fails with
    /// [`SERVICE_RESTARTED`]; a run that exported its artifact is activated again
    /// (or simply succeeded when auto-activation is off) because activation is
    /// idempotent. Returns the runs this call finalized.
    pub async fn recover(&self) -> Result<Vec<TrainingRunRecord>> {
        self.config.validate()?;
        let mut recovered = Vec::new();
        for training_run_id in self.members(&keys::training_runs()).await? {
            let Some(run) = self.load(&training_run_id).await? else {
                continue;
            };
            if run.is_terminal() {
                continue;
            }
            let outcome = if matches!(run.stage, RunStage::Queued | RunStage::Training) {
                self.restore_samples(&run.sample_ids).await?;
                self.finish(&run, RunStatus::Failed, Some(SERVICE_RESTARTED.to_owned()))
                    .await?
            } else if !self.config.auto_activate_lora {
                self.finish(&run, RunStatus::Succeeded, None).await?
            } else if let Err(error) = self.activator.activate(&self.activation(&run)).await {
                self.finish(&run, RunStatus::Failed, Some(error.to_string()))
                    .await?
            } else {
                self.finish(&run, RunStatus::Succeeded, None).await?
            };
            recovered.push(outcome);
        }
        Ok(recovered)
    }

    /// Counters of the runs and the sample ledger.
    pub async fn stats(&self) -> Result<BTreeMap<String, usize>> {
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        let mut total_runs = 0;
        for training_run_id in self.members(&keys::training_runs()).await? {
            if let Some(run) = self.load(&training_run_id).await? {
                total_runs += 1;
                *counts.entry(run.status.as_str()).or_default() += 1;
            }
        }
        let mut stats = BTreeMap::new();
        for status in [
            RunStatus::Pending,
            RunStatus::Running,
            RunStatus::Succeeded,
            RunStatus::Failed,
            RunStatus::Canceled,
        ] {
            stats.insert(
                format!("{}_runs", status.as_str()),
                counts.get(status.as_str()).copied().unwrap_or_default(),
            );
        }
        stats.insert("total_runs".to_owned(), total_runs);
        stats.extend(self.ledger().stats().await?);
        Ok(stats)
    }

    /// Exports the run record, its policy versions and artifact descriptor into `dir`.
    ///
    /// The file is written atomically and its path returned.
    pub async fn export_checkpoint(&self, training_run_id: &str, dir: &Path) -> Result<PathBuf> {
        let run = self.get(training_run_id).await?;
        let artifact = match self.read(&keys::training_artifact(training_run_id)).await? {
            Some(payload) => Some(json::decode::<TrainingArtifact>(&payload)?),
            // A run exported before the artifact record existed still exports the
            // name/path the record carries.
            None => run.lora_name.clone().map(|lora_name| {
                TrainingArtifact::new(lora_name, run.lora_path.clone().unwrap_or_default())
            }),
        };
        let checkpoint = TrainingCheckpoint::from_run(&self.config.model_id, &run, artifact);
        checkpoint
            .save_to_dir(dir)
            .await
            .map_err(OnlineError::store)
    }

    async fn build_run(&self, samples: &[SampleRecord]) -> Result<TrainingRunRecord> {
        let mut policy_versions: BTreeMap<String, u32> = BTreeMap::new();
        for sample in samples {
            *policy_versions.entry(sample.policy_version()).or_default() += 1;
        }
        let (parent_lora_name, parent_lora_path) = self
            .parent_policy
            .clone()
            .unwrap_or_else(|| ("base".to_owned(), String::new()));
        Ok(TrainingRunRecord {
            training_run_id: self.next_run_id().await?,
            status: RunStatus::Pending,
            stage: RunStage::Queued,
            sample_count: samples.len(),
            policy_versions,
            created_at: self.clock.now(),
            started_at: None,
            finished_at: None,
            lora_name: None,
            lora_path: None,
            failure_reason: None,
            sample_ids: sample_ids(samples),
            parent_lora_name: Some(parent_lora_name),
            parent_lora_path: Some(parent_lora_path),
        })
    }

    /// Allocates the next run id from the monotonic counter.
    async fn next_run_id(&self) -> Result<String> {
        let key = keys::training_run_seq();
        let next = match self.read(&key).await? {
            Some(raw) => raw
                .trim()
                .parse::<u64>()
                .map_err(|error| OnlineError::store(Error::external(error)))?
                .saturating_add(1),
            None => 1,
        };
        self.write(&key, &next.to_string()).await?;
        Ok(format!("run-{next:012x}"))
    }

    fn training_request(
        &self,
        run: &TrainingRunRecord,
        samples: Vec<SampleRecord>,
    ) -> TrainingRequest {
        TrainingRequest {
            training_run_id: run.training_run_id.clone(),
            model_id: self.config.model_id.clone(),
            samples,
            init_lora_name: Some(run.parent_lora_name().to_owned()),
            init_lora_path: Some(run.parent_lora_path().to_owned()),
            output_dir: self.config.output_dir.clone(),
            config: RlTrainingConfig {
                train_batch_size: self.config.batch_size,
                model_path: Some(self.config.model_id.clone()),
                save_path: self
                    .config
                    .output_dir
                    .as_ref()
                    .map(|dir| dir.to_string_lossy().into_owned()),
                ..RlTrainingConfig::default()
            },
        }
    }

    fn activation(&self, run: &TrainingRunRecord) -> LoraActivation {
        LoraActivation {
            base_model: self.config.model_id.clone(),
            lora_name: run.lora_name.clone().unwrap_or_default(),
            lora_path: run.lora_path.clone().unwrap_or_default(),
            expected_lora_name: run.parent_lora_name().to_owned(),
            training_run_id: run.training_run_id.clone(),
        }
    }

    async fn load_samples(&self, sample_ids: &[String]) -> Result<Vec<SampleRecord>> {
        let ledger = self.ledger();
        let mut samples = Vec::with_capacity(sample_ids.len());
        for sample_id in sample_ids {
            match ledger.get(sample_id).await? {
                Some(sample) => samples.push(sample),
                None => {
                    return Err(OnlineError::store(Error::workflow(format!(
                        "training sample is missing: {sample_id}"
                    ))))
                }
            }
        }
        Ok(samples)
    }

    /// Returns the durable record when a concurrent caller finalized the run.
    async fn stopped(&self, run: &TrainingRunRecord) -> Result<Option<TrainingRunRecord>> {
        let current = self.get(&run.training_run_id).await?;
        Ok(current.is_terminal().then_some(current))
    }

    async fn restore_samples(&self, sample_ids: &[String]) -> Result<()> {
        self.ledger().reset_to_pending(sample_ids).await
    }

    /// Persists a record and, when final, clears the active marker that names it.
    async fn save(&self, run: &TrainingRunRecord) -> Result<()> {
        self.write(&keys::training_run(&run.training_run_id), &run.to_json()?)
            .await?;
        self.insert_member(&keys::training_runs(), &run.training_run_id)
            .await?;
        if run.is_terminal()
            && self.read(&keys::active_training_run()).await?.as_deref()
                == Some(run.training_run_id.as_str())
        {
            self.remove(&keys::active_training_run()).await?;
        }
        Ok(())
    }

    /// Persists a newly claimed record; `false` means its id was taken.
    async fn store_run(&self, run: &TrainingRunRecord) -> Result<bool> {
        let written = self
            .write_if_absent(&keys::training_run(&run.training_run_id), &run.to_json()?)
            .await?;
        if written {
            self.insert_member(&keys::training_runs(), &run.training_run_id)
                .await?;
        }
        Ok(written)
    }

    async fn finish(
        &self,
        run: &TrainingRunRecord,
        status: RunStatus,
        failure_reason: Option<String>,
    ) -> Result<TrainingRunRecord> {
        let finished = TrainingRunRecord {
            status,
            finished_at: Some(self.clock.now()),
            failure_reason,
            ..run.clone()
        };
        self.save(&finished).await?;
        Ok(finished)
    }

    async fn load(&self, training_run_id: &str) -> Result<Option<TrainingRunRecord>> {
        match self.read(&keys::training_run(training_run_id)).await? {
            Some(payload) => Ok(Some(TrainingRunRecord::from_json(&payload)?)),
            None => Ok(None),
        }
    }

    fn ledger(&self) -> SampleLedger<&S> {
        SampleLedger::new(&self.store)
    }

    async fn read(&self, key: &str) -> Result<Option<String>> {
        self.store.get(key).await.map_err(OnlineError::store)
    }

    async fn write(&self, key: &str, value: &str) -> Result<()> {
        self.store.set(key, value).await.map_err(OnlineError::store)
    }

    async fn write_if_absent(&self, key: &str, value: &str) -> Result<bool> {
        self.store
            .set_if_absent(key, value)
            .await
            .map_err(OnlineError::store)
    }

    async fn remove(&self, key: &str) -> Result<bool> {
        self.store.del(key).await.map_err(OnlineError::store)
    }

    async fn members(&self, key: &str) -> Result<Vec<String>> {
        self.store
            .set_members(key)
            .await
            .map_err(OnlineError::store)
    }

    async fn insert_member(&self, key: &str, member: &str) -> Result<bool> {
        self.store
            .set_add(key, member)
            .await
            .map_err(OnlineError::store)
    }

    async fn remove_member(&self, key: &str, member: &str) -> Result<bool> {
        self.store
            .set_remove(key, member)
            .await
            .map_err(OnlineError::store)
    }
}

fn sample_ids(samples: &[SampleRecord]) -> Vec<String> {
    samples
        .iter()
        .map(|sample| sample.sample_id.clone())
        .collect()
}

fn missing_run(training_run_id: &str) -> OnlineError {
    OnlineError::TrainingRunNotFound(format!("unknown Training Run: {training_run_id}"))
}
