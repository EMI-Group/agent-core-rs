//! The portable export of one training run.
//!
//! [`crate::rl::RlCheckpoint`] is the epoch checkpoint of the offline trainer
//! (policy blob, best reward, per-epoch metrics); an online run has a fixed
//! batch, a policy version and an artifact instead, so the export gets its own
//! record rather than a lossy mapping into that one.
//!
//! A checkpoint is written atomically (temp + rename) into a caller-supplied
//! directory and reads back into the same record, so an operator can archive a
//! finished run's identity, batch, policy versions and artifact descriptor
//! without a live store.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use openjiuwen_common::error::{Error, Result as KernelResult};
use serde::{Deserialize, Serialize};

use super::executor::TrainingArtifact;
use super::training::{RunStage, RunStatus, TrainingRunRecord};

/// Schema version of a [`TrainingCheckpoint`] file.
pub const TRAINING_CHECKPOINT_VERSION: &str = "training-checkpoint/v1";

/// A portable export of one training run.
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
