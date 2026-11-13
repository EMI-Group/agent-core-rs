//! Checkpoint records for the offline RL loop.
//!
//! A checkpoint is one JSON document holding the epoch, the best reward seen,
//! the opaque policy parameters, and scalar metrics. Files are written
//! atomically (`temp file + rename`) as `epoch-{epoch}.json` plus a
//! `latest.json` alias, matching the Python offline store's layout without its
//! external backends. No clock is consulted, so checkpoint names and contents
//! are reproducible.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};

/// The only checkpoint schema version this implementation reads and writes.
pub const CHECKPOINT_VERSION: &str = "v1";

/// Name of the alias file pointing at the newest checkpoint.
const LATEST_FILE: &str = "latest.json";

/// One RL checkpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RlCheckpoint {
    /// Schema version; always [`CHECKPOINT_VERSION`] for files written here.
    pub version: String,
    /// Identifier of the training run.
    pub run_id: String,
    /// Epoch the checkpoint was written after.
    pub epoch: usize,
    /// Best normalised reward seen up to this epoch.
    pub best_reward: f64,
    /// Opaque policy parameters.
    pub policy: Value,
    /// Scalar metrics of the epoch.
    pub metrics: BTreeMap<String, f64>,
}

impl Default for RlCheckpoint {
    fn default() -> Self {
        Self {
            version: CHECKPOINT_VERSION.to_owned(),
            run_id: String::new(),
            epoch: 0,
            best_reward: 0.0,
            policy: Value::Null,
            metrics: BTreeMap::new(),
        }
    }
}

impl RlCheckpoint {
    /// Creates a checkpoint with the current schema version.
    #[must_use]
    pub fn new(
        run_id: impl Into<String>,
        epoch: usize,
        best_reward: f64,
        policy: Value,
        metrics: BTreeMap<String, f64>,
    ) -> Self {
        Self {
            version: CHECKPOINT_VERSION.to_owned(),
            run_id: run_id.into(),
            epoch,
            best_reward,
            policy,
            metrics,
        }
    }

    /// The per-epoch file name (`epoch-{epoch}.json`).
    #[must_use]
    pub fn file_name(&self) -> String {
        format!("epoch-{}.json", self.epoch)
    }

    /// Validates the schema version, run id, and reward.
    pub fn validate(&self) -> Result<()> {
        if self.version != CHECKPOINT_VERSION {
            return Err(Error::validation(format!(
                "unsupported RlCheckpoint version {:?}, expected {CHECKPOINT_VERSION:?}",
                self.version
            )));
        }
        if self.run_id.trim().is_empty() {
            return Err(Error::validation("RlCheckpoint run_id must not be empty"));
        }
        if !self.best_reward.is_finite() {
            return Err(Error::validation("RlCheckpoint best_reward must be finite"));
        }
        Ok(())
    }

    /// Writes the checkpoint into `dir` and returns the epoch file path.
    ///
    /// The directory is created when missing. Both the epoch file and
    /// `latest.json` are written atomically; the returned path is the epoch file.
    pub async fn save_to_dir(&self, dir: &Path) -> Result<PathBuf> {
        self.validate()?;
        tokio::fs::create_dir_all(dir)
            .await
            .map_err(Error::external)?;
        let encoded = serde_json::to_vec_pretty(self).map_err(Error::external)?;
        let epoch_path = dir.join(self.file_name());
        write_atomic(&epoch_path, &encoded).await?;
        write_atomic(&dir.join(LATEST_FILE), &encoded).await?;
        Ok(epoch_path)
    }

    /// Loads and validates a checkpoint file.
    pub async fn load_from_path(path: &Path) -> Result<Self> {
        let encoded = tokio::fs::read(path).await.map_err(Error::external)?;
        let checkpoint: Self = serde_json::from_slice(&encoded).map_err(Error::external)?;
        checkpoint.validate()?;
        Ok(checkpoint)
    }

    /// Loads the newest checkpoint in `dir`, if any.
    ///
    /// Files named `epoch-{n}.json` win, newest epoch first; ties break on the
    /// file path. When no epoch file exists, `latest.json` is used. A missing
    /// directory yields `Ok(None)`.
    pub async fn latest_in(dir: &Path) -> Result<Option<Self>> {
        let mut entries = match tokio::fs::read_dir(dir).await {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(Error::external(error)),
        };
        let mut candidates: Vec<(usize, PathBuf)> = Vec::new();
        while let Some(entry) = entries.next_entry().await.map_err(Error::external)? {
            let file_name = entry.file_name();
            if let Some(epoch) = parse_epoch_name(&file_name.to_string_lossy()) {
                candidates.push((epoch, entry.path()));
            }
        }
        candidates.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| right.1.cmp(&left.1)));
        if let Some((_, path)) = candidates.first() {
            return Self::load_from_path(path).await.map(Some);
        }
        let latest = dir.join(LATEST_FILE);
        match tokio::fs::metadata(&latest).await {
            Ok(_) => Self::load_from_path(&latest).await.map(Some),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(Error::external(error)),
        }
    }
}

fn parse_epoch_name(file_name: &str) -> Option<usize> {
    file_name
        .strip_prefix("epoch-")?
        .strip_suffix(".json")?
        .parse()
        .ok()
}

async fn write_atomic(path: &Path, encoded: &[u8]) -> Result<()> {
    let file_name = path.file_name().map_or_else(
        || std::ffi::OsString::from("checkpoint"),
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
