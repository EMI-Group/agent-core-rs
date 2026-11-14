//! The training-executor seam and its two implementations.
//!
//! Python parity: `agent_rl/online/abstract/trainer.py` (`TrainingExecutor` /
//! `BatchTrainingExecutor` protocols) and `agent_rl/online/scheduler/ppo_executor.py`.
//! Python's online executor is a Ray/verl PPO job driver; this crate replaces the
//! transport with a trait so the training runner is executable offline and the
//! real GPU path can be dropped in later without touching the state machine.
//!
//! # The artifact contract
//!
//! Every executor must name its output `<model_id>:vN` with `N >= 1`, the
//! contract Python enforces with `re.fullmatch(rf"{model_id}:v[1-9][0-9]*")`
//! after `train()`. [`TrainingArtifact::validate_name`] is that check, and the
//! training runner calls it before it records a successful run, so a sloppy
//! executor cannot smuggle a foreign policy version into the ledger.
//!
//! # Implementations
//!
//! - [`LocalTrainingExecutor`] — deterministic and offline: it computes the
//!   batch's reward statistics, bumps the policy version, and writes an artifact
//!   descriptor under its root directory. It is what the tests and local runs use.
//! - [`PpoTrainingExecutor`] — the real GPU path, which is **not** available in
//!   this build: every method returns [`Error::Unsupported`] naming `torch`,
//!   `transformers`, `verl`, `ray` and the CUDA device. It never fakes training,
//!   and its `check_health` reports the same typed error so the service's
//!   `/health` answers `503` with that precise reason.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

use super::sample::SampleRecord;
use crate::rl::RlTrainingConfig;

/// Trains one immutable sample batch and exports its versioned `LoRA`.
///
/// Implementations own the training transport only; batching, sample
/// disposition, activation and recovery belong to the training runner.
#[async_trait]
pub trait TrainingExecutor: Send + Sync {
    /// Short executor name (`"local"`, `"ppo"`), used in diagnostics and health.
    fn name(&self) -> &'static str;

    /// Trains `request.samples` and returns the exported artifact.
    async fn train_batch(&self, request: &TrainingRequest) -> Result<TrainingArtifact>;

    /// Requests cancellation of `training_run_id`.
    ///
    /// Returns `true` when the executor settled the worker itself (the runner
    /// then leaves the sample disposition to the executor's own failure path),
    /// `false` when there was nothing to cancel.
    async fn cancel(&self, training_run_id: &str) -> Result<bool>;

    /// Probes whether the executor can train at all.
    async fn check_health(&self) -> Result<()>;
}

/// One fixed batch handed to a [`TrainingExecutor`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TrainingRequest {
    /// Run the batch belongs to.
    pub training_run_id: String,
    /// Model identity the artifact must be named after.
    pub model_id: String,
    /// The claimed sample snapshot.
    pub samples: Vec<SampleRecord>,
    /// `LoRA` name the batch was rolled out on (the run's parent policy).
    pub init_lora_name: Option<String>,
    /// Path of that parent policy's weights.
    pub init_lora_path: Option<String>,
    /// Directory the executor writes its output under, when the runner configured one.
    pub output_dir: Option<PathBuf>,
    /// Training-loop parameters of the batch.
    pub config: RlTrainingConfig,
}

/// The exported policy of one training batch, ready for activation.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TrainingArtifact {
    /// Artifact name, which must satisfy the `<model_id>:vN` contract.
    pub lora_name: String,
    /// Where the artifact lives (a weight directory for a real executor).
    pub lora_path: String,
    /// Scalar training metrics of the batch.
    pub metrics: BTreeMap<String, f64>,
}

impl TrainingArtifact {
    /// Creates an artifact from its name and path.
    #[must_use]
    pub fn new(lora_name: impl Into<String>, lora_path: impl Into<String>) -> Self {
        Self {
            lora_name: lora_name.into(),
            lora_path: lora_path.into(),
            metrics: BTreeMap::new(),
        }
    }

    /// Enforces Python's `^{model_id}:v[1-9][0-9]*$` artifact-name contract.
    pub fn validate_name(&self, model_id: &str) -> Result<()> {
        if !is_artifact_name(model_id, &self.lora_name) {
            return Err(Error::validation(
                "PPO artifact must use '<model_id>:vN' name",
            ));
        }
        Ok(())
    }
}

/// The artifact name of `model_id` at policy `version`.
#[must_use]
pub fn artifact_name(model_id: &str, version: u32) -> String {
    format!("{model_id}:v{version}")
}

/// Whether `name` is a legal artifact name for `model_id`.
///
/// The port compares plain string prefixes instead of compiling Python's
/// `re.escape`d pattern, so a model id containing regex metacharacters behaves
/// exactly like any other id.
#[must_use]
pub fn is_artifact_name(model_id: &str, name: &str) -> bool {
    artifact_version(model_id, name).is_some()
}

/// The policy version of `name`, when it is a legal artifact name of `model_id`.
#[must_use]
pub fn artifact_version(model_id: &str, name: &str) -> Option<u32> {
    if model_id.is_empty() {
        return None;
    }
    let version = name.strip_prefix(model_id)?.strip_prefix(":v")?.as_bytes();
    // `v[1-9][0-9]*`: a leading zero is rejected, as Python's pattern does.
    if !matches!(version.first(), Some(b'1'..=b'9')) || !version[1..].iter().all(u8::is_ascii_digit)
    {
        return None;
    }
    std::str::from_utf8(version).ok()?.parse::<u32>().ok()
}

/// Deterministic, offline training executor.
///
/// It does not learn anything: it turns the batch into reward statistics and a
/// policy-version bump (`init_lora_name`'s version + 1, else `v1`), writes the
/// descriptor as JSON, and returns it. That is enough for the training-run state
/// machine, the `LoRA` repository and the service tests to run end to end with no
/// GPU, no model and no clock.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalTrainingExecutor {
    root: PathBuf,
}

impl LocalTrainingExecutor {
    /// Creates an executor writing its descriptors under `root`.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The configured root directory.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Directory a run writes its artifact into.
    ///
    /// `request.output_dir` replaces the executor root when the runner set one;
    /// the run id always scopes the directory, so two runs never collide.
    #[must_use]
    pub fn run_dir(&self, base: Option<&Path>, training_run_id: &str) -> PathBuf {
        base.map_or_else(|| self.root.clone(), Path::to_path_buf)
            .join(training_run_id)
    }

    /// Path of the descriptor `train_batch` writes for `training_run_id`.
    #[must_use]
    pub fn descriptor_path(
        &self,
        base: Option<&Path>,
        training_run_id: &str,
        version: u32,
    ) -> PathBuf {
        self.run_dir(base, training_run_id)
            .join(format!("v{version}.json"))
    }
}

/// The descriptor [`LocalTrainingExecutor`] writes for one artifact.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct LocalArtifactDescriptor {
    /// Artifact name (`<model_id>:vN`).
    pub lora_name: String,
    /// Model identity the artifact belongs to.
    pub model_id: String,
    /// Policy version the artifact carries.
    pub version: u32,
    /// Samples the artifact was trained on.
    pub sample_count: usize,
    /// Mean reward of the batch.
    pub mean_reward: f64,
    /// Best reward of the batch.
    pub best_reward: f64,
    /// Parent policy name (`base` when the run started from scratch).
    pub base: String,
}

#[async_trait]
impl TrainingExecutor for LocalTrainingExecutor {
    fn name(&self) -> &'static str {
        "local"
    }

    async fn train_batch(&self, request: &TrainingRequest) -> Result<TrainingArtifact> {
        let descriptor = local_descriptor(request)?;
        let dir = self.run_dir(request.output_dir.as_deref(), &request.training_run_id);
        tokio::fs::create_dir_all(&dir)
            .await
            .map_err(Error::external)?;
        let path = dir.join(format!("v{}.json", descriptor.version));
        let encoded = serde_json::to_vec_pretty(&descriptor).map_err(Error::external)?;
        write_atomic(&path, &encoded).await?;

        let mut metrics = BTreeMap::new();
        metrics.insert("sample_count".to_owned(), descriptor.sample_count as f64);
        metrics.insert("mean_reward".to_owned(), descriptor.mean_reward);
        metrics.insert("best_reward".to_owned(), descriptor.best_reward);
        metrics.insert("policy_version".to_owned(), f64::from(descriptor.version));
        Ok(TrainingArtifact {
            lora_name: descriptor.lora_name,
            lora_path: path.to_string_lossy().into_owned(),
            metrics,
        })
    }

    async fn cancel(&self, _training_run_id: &str) -> Result<bool> {
        // `train_batch` runs to completion inside the call, so no worker of this
        // executor can outlive it and there is nothing to settle.
        Ok(false)
    }

    async fn check_health(&self) -> Result<()> {
        Ok(())
    }
}

fn local_descriptor(request: &TrainingRequest) -> Result<LocalArtifactDescriptor> {
    if request.model_id.trim().is_empty() {
        return Err(Error::validation("model_id is required"));
    }
    if request.samples.is_empty() {
        return Err(Error::validation(
            "training batch must contain at least one sample",
        ));
    }
    let mut total = 0.0;
    let mut best = f64::NEG_INFINITY;
    for sample in &request.samples {
        if !sample.reward.is_finite() {
            return Err(Error::validation(format!(
                "sample reward must be finite: {}",
                sample.sample_id
            )));
        }
        total += sample.reward;
        best = best.max(sample.reward);
    }
    let sample_count = request.samples.len();
    // A policy-version bump: `base` or an unparsable parent starts at v1.
    let version = request
        .init_lora_name
        .as_deref()
        .and_then(|name| artifact_version(&request.model_id, name))
        .map_or(1, |version| version + 1);
    Ok(LocalArtifactDescriptor {
        lora_name: artifact_name(&request.model_id, version),
        model_id: request.model_id.clone(),
        version,
        sample_count,
        mean_reward: total / sample_count as f64,
        best_reward: best,
        base: request
            .init_lora_name
            .clone()
            .unwrap_or_else(|| "base".to_owned()),
    })
}

/// The real GPU executor, retained as a typed seam.
///
/// Every operation fails with [`Error::Unsupported`]: the online PPO path runs
/// Python's `torch`/`transformers` training stack through `verl` and `ray` on a
/// CUDA device, none of which this workspace links. No capability probe exists,
/// so `check_health` reports the same typed error — the service's `/health`
/// answers `503` with this reason rather than claiming a training capability it
/// does not have.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PpoTrainingExecutor {
    device: String,
}

impl PpoTrainingExecutor {
    /// Creates the executor for the default device `cuda:0`.
    #[must_use]
    pub fn new() -> Self {
        Self::with_device("cuda:0")
    }

    /// Creates the executor for an explicit CUDA device.
    #[must_use]
    pub fn with_device(device: impl Into<String>) -> Self {
        Self {
            device: device.into(),
        }
    }

    /// The CUDA device this executor would train on.
    #[must_use]
    pub fn device(&self) -> &str {
        &self.device
    }

    fn unavailable(&self, operation: &str) -> Error {
        Error::unsupported(format!(
            "online PPO training is not available in this build: {operation} requires the Python \
             training stack (torch, transformers, verl, ray) on CUDA device {}",
            self.device
        ))
    }
}

impl Default for PpoTrainingExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TrainingExecutor for PpoTrainingExecutor {
    fn name(&self) -> &'static str {
        "ppo"
    }

    async fn train_batch(&self, _request: &TrainingRequest) -> Result<TrainingArtifact> {
        Err(self.unavailable("train_batch"))
    }

    async fn cancel(&self, _training_run_id: &str) -> Result<bool> {
        Err(self.unavailable("cancel"))
    }

    async fn check_health(&self) -> Result<()> {
        Err(self.unavailable("check_health"))
    }
}

async fn write_atomic(path: &Path, encoded: &[u8]) -> Result<()> {
    let file_name = path.file_name().map_or_else(
        || std::ffi::OsString::from("artifact"),
        std::ffi::OsStr::to_os_string,
    );
    let mut temp_name = std::ffi::OsString::from(".");
    temp_name.push(file_name);
    temp_name.push(".tmp");
    let temp_path = path.with_file_name(temp_name);
    tokio::fs::write(&temp_path, encoded)
        .await
        .map_err(Error::external)?;
    tokio::fs::rename(&temp_path, path)
        .await
        .map_err(Error::external)?;
    Ok(())
}
