//! The filesystem LoRA artifact repository.
//!
//! Python parity: `agent_rl/storage/lora_repo.py` (`LoRARepository`,
//! `LoRAPublishRequest`, `LoRAVersion`). The port keeps the repository's
//! filesystem contract — one directory per model, one version directory per
//! publish, an atomically replaced `latest` pointer, and an availability flag the
//! inference probe flips — without a database or a service.
//!
//! ```text
//! {root}/{model_id}/latest          version literal the pointer names ("v3")
//! {root}/{model_id}/v{N}/version.json   serialized LoraVersion
//! {root}/{model_id}/v{N}/artifact.json  the executor's TrainingArtifact descriptor
//! ```
//!
//! # Version allocation and concurrency
//!
//! `publish` computes the next version as the largest existing `vN` plus one and
//! claims it with an atomic directory creation: if a concurrent publisher took
//! `v{N}` first, the `AlreadyExists` result retries with `v{N+1}`. That makes the
//! allocator safe across processes without a lock. The one caveat is that the
//! repository is filesystem-local — two writers over a *network* filesystem with
//! weak directory-creation atomicity are outside the contract, and `latest` is
//! last-writer-wins between them.
//!
//! # Determinism
//!
//! Nothing reads the clock. `created_seq` is the version number: version numbers
//! are strictly increasing per model, so they already carry the creation order,
//! and an export/import of the tree preserves every ordering decision.

use std::path::{Path, PathBuf};

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

use super::executor::TrainingArtifact;

/// File name of the serialized [`LoraVersion`] inside a version directory.
const VERSION_FILE: &str = "version.json";
/// File name of the executor descriptor inside a version directory.
const ARTIFACT_FILE: &str = "artifact.json";
/// File name of the pointer naming the latest version of a model.
const LATEST_FILE: &str = "latest";
/// Attempts `publish` makes to claim a version number before giving up.
const CLAIM_ATTEMPTS: u32 = 16;

/// Inference availability of a published artifact.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoraAvailability {
    /// Published, not yet confirmed servable by the inference gateway.
    #[default]
    Pending,
    /// The inference gateway serves this version.
    Available,
    /// The inference gateway rejected this version.
    Unavailable,
}

impl LoraAvailability {
    /// The Python wire literal of the state.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Available => "available",
            Self::Unavailable => "unavailable",
        }
    }
}

/// One published version of a model's LoRA artifact.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct LoraVersion {
    /// Model identity the artifact belongs to (Python's `user_id`).
    pub model_id: String,
    /// Version number (`1` is the first publish of the model).
    pub version: u32,
    /// Artifact name (`<model_id>:vN`).
    pub lora_name: String,
    /// Artifact path (a weight directory for a real executor).
    pub lora_path: String,
    /// Inference availability.
    pub availability: LoraAvailability,
    /// Why the availability is what it is.
    pub reason: String,
    /// Creation sequence — the version number, because versions are monotonic.
    pub created_seq: u64,
}

impl LoraVersion {
    /// The Python-style version literal (`"v3"`).
    #[must_use]
    pub fn version_literal(&self) -> String {
        version_literal(self.version)
    }
}

/// Filesystem-backed artifact repository.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoraRepository {
    root: PathBuf,
}

impl LoraRepository {
    /// Creates a repository over `root`; directories are created on the first write.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The repository root.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Directory of one model.
    #[must_use]
    pub fn model_dir(&self, model_id: &str) -> PathBuf {
        self.root.join(model_id)
    }

    /// Directory of one published version.
    #[must_use]
    pub fn version_dir(&self, model_id: &str, version: u32) -> PathBuf {
        self.model_dir(model_id).join(version_literal(version))
    }

    /// Publishes `artifact` as the next version of `model_id`.
    ///
    /// The artifact name must satisfy the executor contract `<model_id>:vN`, so
    /// the repository and the trainer cannot disagree about a version. The new
    /// version starts `pending`; the inference probe promotes it through
    /// [`LoraRepository::set_availability`].
    pub async fn publish(
        &self,
        model_id: &str,
        artifact: &TrainingArtifact,
    ) -> Result<LoraVersion> {
        validate_model_id(model_id)?;
        artifact.validate_name(model_id)?;

        let model_dir = self.model_dir(model_id);
        tokio::fs::create_dir_all(&model_dir)
            .await
            .map_err(Error::external)?;
        let mut next = self.next_version(model_id).await?;
        let mut version_dir = model_dir.join(version_literal(next));
        let mut claimed = false;
        for _ in 0..CLAIM_ATTEMPTS {
            match tokio::fs::create_dir(&version_dir).await {
                Ok(()) => {
                    claimed = true;
                    break;
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    next += 1;
                    version_dir = model_dir.join(version_literal(next));
                }
                Err(error) => return Err(Error::external(error)),
            }
        }
        if !claimed {
            return Err(Error::workflow(format!(
                "cannot claim a LoRA version for {model_id}: {CLAIM_ATTEMPTS} attempts collided"
            )));
        }

        let version = LoraVersion {
            model_id: model_id.to_owned(),
            version: next,
            lora_name: artifact.lora_name.clone(),
            lora_path: artifact.lora_path.clone(),
            availability: LoraAvailability::Pending,
            reason: String::new(),
            created_seq: u64::from(next),
        };
        self.write_json(&version_dir.join(ARTIFACT_FILE), artifact)
            .await?;
        self.write_version(&version_dir, &version).await?;
        write_atomic(
            &model_dir.join(LATEST_FILE),
            version_literal(next).as_bytes(),
        )
        .await?;
        Ok(version)
    }

    /// Every published version of `model_id`, oldest first.
    pub async fn list_versions(&self, model_id: &str) -> Result<Vec<LoraVersion>> {
        let mut versions = Vec::new();
        for version in self.version_numbers(model_id).await? {
            if let Some(record) = self.get_version(model_id, version).await? {
                versions.push(record);
            }
        }
        Ok(versions)
    }

    /// The newest published version of `model_id`.
    pub async fn get_latest(&self, model_id: &str) -> Result<Option<LoraVersion>> {
        if let Some(literal) = self.read_latest(model_id).await? {
            if let Some(version) = parse_version_literal(&literal) {
                if let Some(record) = self.get_version(model_id, version).await? {
                    return Ok(Some(record));
                }
            }
        }
        // A missing or dangling pointer falls back to the newest directory.
        let versions = self.version_numbers(model_id).await?;
        match versions.last() {
            Some(version) => self.get_version(model_id, *version).await,
            None => Ok(None),
        }
    }

    /// The newest version of `model_id` the inference gateway confirmed.
    pub async fn get_latest_available(&self, model_id: &str) -> Result<Option<LoraVersion>> {
        for version in self.version_numbers(model_id).await?.into_iter().rev() {
            if let Some(record) = self.get_version(model_id, version).await? {
                if record.availability == LoraAvailability::Available {
                    return Ok(Some(record));
                }
            }
        }
        Ok(None)
    }

    /// One published version, if it exists.
    pub async fn get_version(&self, model_id: &str, version: u32) -> Result<Option<LoraVersion>> {
        validate_model_id(model_id)?;
        let path = self.version_dir(model_id, version).join(VERSION_FILE);
        match tokio::fs::read(&path).await {
            Ok(encoded) => Ok(Some(
                serde_json::from_slice(&encoded).map_err(Error::external)?,
            )),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(Error::external(error)),
        }
    }

    /// The artifact descriptor stored with a published version.
    pub async fn artifact(&self, model_id: &str, version: u32) -> Result<Option<TrainingArtifact>> {
        validate_model_id(model_id)?;
        let path = self.version_dir(model_id, version).join(ARTIFACT_FILE);
        match tokio::fs::read(&path).await {
            Ok(encoded) => Ok(Some(
                serde_json::from_slice(&encoded).map_err(Error::external)?,
            )),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(Error::external(error)),
        }
    }

    /// Records the inference gateway's verdict on one version.
    ///
    /// `available = true` marks the version servable, `false` marks it rejected;
    /// the reason is kept verbatim so the service can explain a fallback.
    pub async fn set_availability(
        &self,
        model_id: &str,
        version: u32,
        available: bool,
        reason: &str,
    ) -> Result<LoraVersion> {
        let Some(mut record) = self.get_version(model_id, version).await? else {
            return Err(Error::not_found(format!(
                "LoRA version not found: {model_id}/{}",
                version_literal(version)
            )));
        };
        record.availability = if available {
            LoraAvailability::Available
        } else {
            LoraAvailability::Unavailable
        };
        record.reason = reason.to_owned();
        self.write_version(&self.version_dir(model_id, version), &record)
            .await?;
        Ok(record)
    }

    /// Every model with published versions, sorted.
    pub async fn list_models(&self) -> Result<Vec<String>> {
        let mut entries = match tokio::fs::read_dir(&self.root).await {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(Error::external(error)),
        };
        let mut models = Vec::new();
        while let Some(entry) = entries.next_entry().await.map_err(Error::external)? {
            if !entry.file_type().await.map_err(Error::external)?.is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            if !name.starts_with('.') {
                models.push(name);
            }
        }
        models.sort();
        Ok(models)
    }

    async fn next_version(&self, model_id: &str) -> Result<u32> {
        Ok(self
            .version_numbers(model_id)
            .await?
            .last()
            .map_or(1, |version| version + 1))
    }

    /// Published version numbers of `model_id`, ascending.
    async fn version_numbers(&self, model_id: &str) -> Result<Vec<u32>> {
        validate_model_id(model_id)?;
        let mut entries = match tokio::fs::read_dir(self.model_dir(model_id)).await {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(Error::external(error)),
        };
        let mut versions = Vec::new();
        while let Some(entry) = entries.next_entry().await.map_err(Error::external)? {
            if !entry.file_type().await.map_err(Error::external)?.is_dir() {
                continue;
            }
            if let Some(version) = parse_version_literal(&entry.file_name().to_string_lossy()) {
                versions.push(version);
            }
        }
        versions.sort_unstable();
        Ok(versions)
    }

    async fn read_latest(&self, model_id: &str) -> Result<Option<String>> {
        validate_model_id(model_id)?;
        let path = self.model_dir(model_id).join(LATEST_FILE);
        match tokio::fs::read_to_string(&path).await {
            Ok(literal) => Ok(Some(literal)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(Error::external(error)),
        }
    }

    async fn write_version(&self, version_dir: &Path, version: &LoraVersion) -> Result<()> {
        self.write_json(&version_dir.join(VERSION_FILE), version)
            .await
    }

    async fn write_json<T: Serialize>(&self, path: &Path, value: &T) -> Result<()> {
        let encoded = serde_json::to_vec_pretty(value).map_err(Error::external)?;
        write_atomic(path, &encoded).await
    }
}

/// The Python-style version literal (`"v3"`).
#[must_use]
pub fn version_literal(version: u32) -> String {
    format!("v{version}")
}

/// Parses a version literal (`"v3"` → `3`); `v0`, `03` and non-numeric names fail.
#[must_use]
pub fn parse_version_literal(literal: &str) -> Option<u32> {
    let digits = literal.strip_prefix('v')?.as_bytes();
    if !matches!(digits.first(), Some(b'1'..=b'9')) || !digits[1..].iter().all(u8::is_ascii_digit) {
        return None;
    }
    std::str::from_utf8(digits).ok()?.parse().ok()
}

/// Rejects a model id that is not one safe path segment.
fn validate_model_id(model_id: &str) -> Result<()> {
    if model_id.trim().is_empty() {
        return Err(Error::validation("model_id is required"));
    }
    if model_id == "."
        || model_id == ".."
        || model_id.contains(['/', '\\', '\0'])
        || model_id.starts_with('.')
    {
        return Err(Error::validation(format!(
            "model_id must be a single path segment: {model_id}"
        )));
    }
    Ok(())
}

async fn write_atomic(path: &Path, encoded: &[u8]) -> Result<()> {
    let file_name = path.file_name().map_or_else(
        || std::ffi::OsString::from("lora"),
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
