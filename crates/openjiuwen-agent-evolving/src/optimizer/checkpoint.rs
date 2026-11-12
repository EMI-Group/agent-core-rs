//! Evolve checkpoint state, stores, and the checkpoint manager.
//!
//! Ports Python `openjiuwen.agent_evolving.checkpointing`:
//!
//! - [`EvolveCheckpoint`] — the resume payload with Python's exact JSON keys
//!   (`version`, `run_id`, `step`, `best`, `seed`, `operators_state`,
//!   `updater_state`, `searcher_state`, `last_metrics`).
//! - [`CheckpointStore`] — the async persistence seam; [`FileCheckpointStore`]
//!   writes JSON files atomically, [`MemoryCheckpointStore`] is an in-process store
//!   for tests.
//! - [`CheckpointManager`] — save policy and snapshot/restore mapping;
//!   [`DefaultCheckpointManager`] mirrors Python's `EvolveCheckpointManager`.
//!
//! Relative names resolve against the store's base directory; absolute paths are
//! used as-is. A [`FileCheckpointStore`] without a base directory (and a missing
//! file) is a silent no-op returning `Ok(None)`, matching Python's `None` behavior.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result, ResultExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Training-loop position inside an [`EvolveCheckpoint`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct CheckpointStep {
    /// Epoch index (zero-based).
    #[serde(default)]
    pub epoch: usize,
    /// Batch index within the epoch (zero-based).
    #[serde(default)]
    pub batch: usize,
}

/// Best-so-far record inside an [`EvolveCheckpoint`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct CheckpointBest {
    /// Best validation score observed up to the checkpoint.
    #[serde(default)]
    pub best_score: f64,
}

/// Training checkpoint used for resume.
///
/// The field set and JSON names mirror Python's `EvolveCheckpoint` dataclass.
/// Collections default to empty and options to `None` so older files remain
/// readable, while `version`/`run_id`/`step`/`best` stay required.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct EvolveCheckpoint {
    /// Checkpoint format version, e.g. `v1`.
    pub version: String,
    /// Identifier of the training run that produced the checkpoint.
    pub run_id: String,
    /// Training-loop position.
    pub step: CheckpointStep,
    /// Best-so-far scores.
    pub best: CheckpointBest,
    /// Optional RNG seed for reproducibility.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Per-operator parameter state (`operator_id -> state`).
    #[serde(default)]
    pub operators_state: BTreeMap<String, Value>,
    /// Optimizer updater state, if the optimizer has one.
    #[serde(default)]
    pub updater_state: Option<Value>,
    /// Optimizer searcher state, if the optimizer has one.
    #[serde(default)]
    pub searcher_state: Option<Value>,
    /// Metrics recorded at checkpoint time (`current_epoch_score` from the manager).
    #[serde(default)]
    pub last_metrics: BTreeMap<String, f64>,
}

/// Async persistence seam for [`EvolveCheckpoint`].
#[async_trait]
pub trait CheckpointStore: Send + Sync {
    /// Saves `checkpoint` under `filename` and returns the written path, if any.
    ///
    /// Relative names resolve against the store's base directory; absolute paths
    /// are used as-is.
    ///
    /// # Errors
    ///
    /// Returns an error when serialization or writing fails.
    async fn save_checkpoint(
        &self,
        checkpoint: &EvolveCheckpoint,
        filename: &str,
    ) -> Result<Option<String>>;

    /// Loads a checkpoint from `path`; a missing file yields `Ok(None)`.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read or is not a valid checkpoint.
    async fn load_checkpoint(&self, path: &str) -> Result<Option<EvolveCheckpoint>>;

    /// Loads only the `operators_state` object from `path`; `Ok(None)` when the
    /// file is missing, the store is disabled, or the key is absent.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read or is not valid JSON.
    async fn load_state_dict(&self, path: &str) -> Result<Option<Value>>;
}

/// Local JSON-file checkpoint store.
///
/// Writes are atomic (temp file in the same directory, then rename) and pretty
/// printed with non-ASCII characters unescaped (`ensure_ascii=False` in Python).
/// A store built with [`FileCheckpointStore::disabled`] silently does nothing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FileCheckpointStore {
    base_dir: Option<PathBuf>,
}

impl FileCheckpointStore {
    /// Creates an enabled store rooted at `base_dir` (created on first save).
    #[must_use]
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: Some(base_dir.into()),
        }
    }

    /// Creates a disabled store: saves return `Ok(None)` and loads return `Ok(None)`.
    #[must_use]
    pub fn disabled() -> Self {
        Self { base_dir: None }
    }

    /// The configured base directory, if the store is enabled.
    #[must_use]
    pub fn base_dir(&self) -> Option<&Path> {
        self.base_dir.as_deref()
    }

    /// Resolves `path` against the base directory; absolute paths pass through.
    fn resolve(&self, path: &str) -> Option<PathBuf> {
        let base = self.base_dir.as_ref()?;
        let path = Path::new(path);
        Some(if path.is_absolute() {
            path.to_path_buf()
        } else {
            base.join(path)
        })
    }

    /// Returns the resolved path for loading, or `None` when disabled.
    fn load_path(&self, path: &str) -> Option<PathBuf> {
        self.resolve(path)
    }
}

#[async_trait]
impl CheckpointStore for FileCheckpointStore {
    async fn save_checkpoint(
        &self,
        checkpoint: &EvolveCheckpoint,
        filename: &str,
    ) -> Result<Option<String>> {
        let Some(path) = self.resolve(filename) else {
            return Ok(None);
        };
        let directory = path.parent().map(Path::to_path_buf).unwrap_or_default();
        tokio::fs::create_dir_all(&directory)
            .await
            .context(format!(
                "creating checkpoint directory {}",
                directory.display()
            ))?;

        let encoded = serde_json::to_vec_pretty(checkpoint)?;
        let base_name = path.file_name().map_or_else(
            || "checkpoint".to_owned(),
            |name| name.to_string_lossy().into_owned(),
        );
        let temp_name = format!(
            "{base_name}.tmp-{}",
            openjiuwen_common::utils::id::uuid_hex()
        );
        let temp_path = directory.join(temp_name);
        tokio::fs::write(&temp_path, &encoded)
            .await
            .context(format!(
                "writing checkpoint temp file {}",
                temp_path.display()
            ))?;
        tokio::fs::rename(&temp_path, &path)
            .await
            .context(format!("renaming checkpoint to {}", path.display()))?;
        Ok(Some(path.to_string_lossy().into_owned()))
    }

    async fn load_checkpoint(&self, path: &str) -> Result<Option<EvolveCheckpoint>> {
        let Some(path) = self.load_path(path) else {
            return Ok(None);
        };
        let bytes = match tokio::fs::read(&path).await {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(error).context(format!("reading checkpoint {}", path.display()))
            }
        };
        let checkpoint = serde_json::from_slice::<EvolveCheckpoint>(&bytes).map_err(|error| {
            Error::validation(format!("invalid checkpoint {}: {error}", path.display()))
        })?;
        Ok(Some(checkpoint))
    }

    async fn load_state_dict(&self, path: &str) -> Result<Option<Value>> {
        let Some(path) = self.load_path(path) else {
            return Ok(None);
        };
        let bytes = match tokio::fs::read(&path).await {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(error).context(format!("reading checkpoint {}", path.display()))
            }
        };
        let raw = serde_json::from_slice::<Value>(&bytes).map_err(|error| {
            Error::validation(format!("invalid checkpoint {}: {error}", path.display()))
        })?;
        Ok(raw.get("operators_state").cloned())
    }
}

/// In-process checkpoint store for tests and dry runs.
///
/// Entries are keyed by the filename passed to [`CheckpointStore::save_checkpoint`]
/// with fallback to the last path component on load, so both `latest.json` and a
/// full path resolve. The synchronous [`Mutex`] is only held in non-`await` sections.
#[derive(Debug, Default)]
pub struct MemoryCheckpointStore {
    entries: Mutex<BTreeMap<String, EvolveCheckpoint>>,
}

impl MemoryCheckpointStore {
    /// Creates an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Looks up an entry without holding the lock across an `await`.
    fn get(&self, path: &str) -> Option<EvolveCheckpoint> {
        let entries = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        entries.get(path).cloned().or_else(|| {
            Path::new(path)
                .file_name()
                .and_then(|name| entries.get(name.to_string_lossy().as_ref()).cloned())
        })
    }
}

#[async_trait]
impl CheckpointStore for MemoryCheckpointStore {
    async fn save_checkpoint(
        &self,
        checkpoint: &EvolveCheckpoint,
        filename: &str,
    ) -> Result<Option<String>> {
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        entries.insert(filename.to_owned(), checkpoint.clone());
        drop(entries);
        Ok(Some(filename.to_owned()))
    }

    async fn load_checkpoint(&self, path: &str) -> Result<Option<EvolveCheckpoint>> {
        Ok(self.get(path))
    }

    async fn load_state_dict(&self, path: &str) -> Result<Option<Value>> {
        Ok(self
            .get(path)
            .map(|checkpoint| Value::Object(checkpoint.operators_state.into_iter().collect())))
    }
}

/// State handed to [`CheckpointManager::build_checkpoint`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct CheckpointSnapshot {
    /// Current epoch index.
    #[serde(default)]
    pub epoch: usize,
    /// Current batch index within the epoch.
    #[serde(default)]
    pub batch: usize,
    /// Best validation score so far.
    #[serde(default)]
    pub best_score: f64,
    /// Score of the epoch being checkpointed.
    #[serde(default)]
    pub current_epoch_score: f64,
    /// Optional RNG seed.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Per-operator parameter state.
    #[serde(default)]
    pub operators_state: BTreeMap<String, Value>,
    /// Optional optimizer updater state.
    #[serde(default)]
    pub updater_state: Option<Value>,
    /// Optional optimizer searcher state.
    #[serde(default)]
    pub searcher_state: Option<Value>,
}

/// State recovered from an [`EvolveCheckpoint`] by [`CheckpointManager::restore`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct RestoredState {
    /// Epoch the trainer should continue from.
    #[serde(default)]
    pub start_epoch: usize,
    /// Best score recorded in the checkpoint.
    #[serde(default)]
    pub best_score: f64,
    /// Run id recorded in the checkpoint.
    #[serde(default)]
    pub run_id: String,
    /// Per-operator parameter state to restore.
    #[serde(default)]
    pub operators_state: BTreeMap<String, Value>,
}

/// Synchronous save-policy and snapshot/restore mapping.
pub trait CheckpointManager: Send + Sync {
    /// Identifier of the run this manager checkpoints.
    fn run_id(&self) -> &str;

    /// Whether epoch `epoch` should be checkpointed, given improvement status.
    fn should_save(&self, epoch: usize, improved: bool) -> bool;

    /// Builds a checkpoint from the current snapshot.
    fn build_checkpoint(&self, snapshot: &CheckpointSnapshot) -> EvolveCheckpoint;

    /// Maps a loaded checkpoint back to trainer state.
    fn restore(&self, checkpoint: &EvolveCheckpoint) -> RestoredState;
}

/// Python-compatible checkpoint manager (`EvolveCheckpointManager`).
///
/// Epoch `0` always saves because `0 % n == 0`; `save_every_n_epochs` is clamped to
/// at least `1`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DefaultCheckpointManager {
    run_id: String,
    checkpoint_version: String,
    save_every_n_epochs: usize,
    save_on_improve: bool,
}

impl DefaultCheckpointManager {
    /// Creates a manager with a fresh UUID-hex run id and Python's default policy.
    #[must_use]
    pub fn new() -> Self {
        Self {
            run_id: openjiuwen_common::utils::id::uuid_hex(),
            checkpoint_version: "v1".to_owned(),
            save_every_n_epochs: 1,
            save_on_improve: true,
        }
    }

    /// Overrides the run id.
    #[must_use]
    pub fn with_run_id(mut self, run_id: impl Into<String>) -> Self {
        self.run_id = run_id.into();
        self
    }

    /// Sets the periodic save interval (clamped to at least `1`) and the
    /// save-on-improvement flag.
    #[must_use]
    pub fn with_save_policy(mut self, every_n_epochs: usize, save_on_improve: bool) -> Self {
        self.save_every_n_epochs = every_n_epochs.max(1);
        self.save_on_improve = save_on_improve;
        self
    }

    /// Checkpoint format version written into every checkpoint.
    #[must_use]
    pub fn checkpoint_version(&self) -> &str {
        &self.checkpoint_version
    }

    /// Configured save interval.
    #[must_use]
    pub fn save_every_n_epochs(&self) -> usize {
        self.save_every_n_epochs
    }

    /// Whether improvement triggers a save.
    #[must_use]
    pub fn save_on_improve(&self) -> bool {
        self.save_on_improve
    }
}

impl Default for DefaultCheckpointManager {
    fn default() -> Self {
        Self::new()
    }
}

impl CheckpointManager for DefaultCheckpointManager {
    fn run_id(&self) -> &str {
        &self.run_id
    }

    fn should_save(&self, epoch: usize, improved: bool) -> bool {
        (improved && self.save_on_improve) || epoch % self.save_every_n_epochs.max(1) == 0
    }

    fn build_checkpoint(&self, snapshot: &CheckpointSnapshot) -> EvolveCheckpoint {
        let mut last_metrics = BTreeMap::new();
        last_metrics.insert(
            "current_epoch_score".to_owned(),
            snapshot.current_epoch_score,
        );
        EvolveCheckpoint {
            version: self.checkpoint_version.clone(),
            run_id: self.run_id.clone(),
            step: CheckpointStep {
                epoch: snapshot.epoch,
                batch: snapshot.batch,
            },
            best: CheckpointBest {
                best_score: snapshot.best_score,
            },
            seed: snapshot.seed,
            operators_state: snapshot.operators_state.clone(),
            updater_state: snapshot.updater_state.clone(),
            searcher_state: snapshot.searcher_state.clone(),
            last_metrics,
        }
    }

    fn restore(&self, checkpoint: &EvolveCheckpoint) -> RestoredState {
        RestoredState {
            start_epoch: checkpoint.step.epoch,
            best_score: checkpoint.best.best_score,
            run_id: checkpoint.run_id.clone(),
            operators_state: checkpoint.operators_state.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn checkpoint() -> EvolveCheckpoint {
        let mut operators_state = BTreeMap::new();
        operators_state.insert("op".to_owned(), json!({"temperature": 0.3}));
        EvolveCheckpoint {
            version: "v1".to_owned(),
            run_id: "run-1".to_owned(),
            step: CheckpointStep { epoch: 2, batch: 1 },
            best: CheckpointBest { best_score: 0.8 },
            seed: Some(7),
            operators_state,
            updater_state: None,
            searcher_state: Some(json!({"cursor": 3})),
            last_metrics: BTreeMap::from([("current_epoch_score".to_owned(), 0.8)]),
        }
    }

    #[test]
    fn checkpoint_round_trips_through_python_keys() {
        let value = serde_json::to_value(checkpoint()).expect("serialize");
        assert_eq!(value["step"]["epoch"], json!(2));
        assert_eq!(value["best"]["best_score"], json!(0.8));
        assert_eq!(value["operators_state"]["op"]["temperature"], json!(0.3));
        assert_eq!(value["last_metrics"]["current_epoch_score"], json!(0.8));
        let decoded: EvolveCheckpoint = serde_json::from_value(value).expect("deserialize");
        assert_eq!(decoded, checkpoint());
    }

    #[test]
    fn optional_fields_default() {
        let minimal = json!({
            "version": "v1",
            "run_id": "r",
            "step": {"epoch": 0, "batch": 0},
            "best": {"best_score": 0.0}
        });
        let decoded: EvolveCheckpoint = serde_json::from_value(minimal).expect("deserialize");
        assert_eq!(decoded.seed, None);
        assert!(decoded.operators_state.is_empty());
        assert_eq!(decoded.updater_state, None);
        assert!(decoded.last_metrics.is_empty());
    }

    #[test]
    fn manager_should_save_matrix() {
        let manager = DefaultCheckpointManager::new().with_run_id("run-x");
        assert_eq!(manager.run_id(), "run-x");
        assert!(manager.should_save(0, false));
        assert!(manager.should_save(1, false));
        assert!(manager.should_save(3, true));

        let every_two = DefaultCheckpointManager::new().with_save_policy(2, false);
        assert!(every_two.should_save(0, false));
        assert!(!every_two.should_save(1, false));
        assert!(every_two.should_save(2, false));
        assert!(!every_two.should_save(3, true));

        let clamped = DefaultCheckpointManager::new().with_save_policy(0, false);
        assert_eq!(clamped.save_every_n_epochs(), 1);
        assert!(clamped.should_save(5, false));
    }

    #[test]
    fn manager_builds_and_restores() {
        let manager = DefaultCheckpointManager::new().with_run_id("run-9");
        let mut operators_state = BTreeMap::new();
        operators_state.insert("op".to_owned(), json!({"x": 1}));
        let snapshot = CheckpointSnapshot {
            epoch: 4,
            batch: 2,
            best_score: 0.75,
            current_epoch_score: 0.6,
            seed: Some(11),
            operators_state: operators_state.clone(),
            updater_state: Some(json!({"u": 1})),
            searcher_state: None,
        };
        let built = manager.build_checkpoint(&snapshot);
        assert_eq!(built.version, "v1");
        assert_eq!(built.run_id, "run-9");
        assert_eq!(built.step, CheckpointStep { epoch: 4, batch: 2 });
        assert_eq!(built.best.best_score, 0.75);
        assert_eq!(built.last_metrics["current_epoch_score"], 0.6);
        assert_eq!(built.updater_state, Some(json!({"u": 1})));

        let restored = manager.restore(&built);
        assert_eq!(restored.start_epoch, 4);
        assert_eq!(restored.best_score, 0.75);
        assert_eq!(restored.run_id, "run-9");
        assert_eq!(restored.operators_state, operators_state);
    }

    #[tokio::test]
    async fn memory_store_round_trip() {
        let store = MemoryCheckpointStore::new();
        let saved = store
            .save_checkpoint(&checkpoint(), "latest.json")
            .await
            .expect("save");
        assert_eq!(saved.as_deref(), Some("latest.json"));
        assert_eq!(
            store.load_checkpoint("latest.json").await.expect("load"),
            Some(checkpoint())
        );
        assert_eq!(
            store
                .load_state_dict("latest.json")
                .await
                .expect("load_state_dict"),
            Some(json!({"op": {"temperature": 0.3}}))
        );
        assert_eq!(
            store.load_checkpoint("missing.json").await.expect("load"),
            None
        );
    }

    #[tokio::test]
    async fn disabled_file_store_is_noop() {
        let store = FileCheckpointStore::disabled();
        assert!(store.base_dir().is_none());
        assert_eq!(
            store
                .save_checkpoint(&checkpoint(), "latest.json")
                .await
                .expect("save"),
            None
        );
        assert_eq!(
            store.load_checkpoint("latest.json").await.expect("load"),
            None
        );
    }

    #[tokio::test]
    async fn file_store_creates_nested_dirs_and_round_trips() {
        let directory = tempfile::tempdir().expect("tempdir");
        let base = directory.path().join("nested").join("checkpoints");
        let store = FileCheckpointStore::new(base.clone());
        assert_eq!(store.base_dir(), Some(base.as_path()));

        let mut checkpoint = checkpoint();
        checkpoint.run_id = "训练-run-ü".to_owned();
        let path = store
            .save_checkpoint(&checkpoint, "latest.json")
            .await
            .expect("save")
            .expect("path returned");
        assert!(path.starts_with(base.to_string_lossy().as_ref()));
        assert!(base.join("latest.json").is_file());

        let raw = tokio::fs::read_to_string(&path).await.expect("read");
        assert!(raw.contains("训练-run-ü"), "unicode must not be escaped");
        assert!(!raw.contains("\\u"), "unicode must not be escaped");
        assert!(raw.contains('\n'), "checkpoint must be pretty printed");
        let parsed: Value = serde_json::from_str(&raw).expect("parse");
        assert_eq!(parsed, serde_json::to_value(&checkpoint).expect("value"));

        assert_eq!(
            store.load_checkpoint(&path).await.expect("load"),
            Some(checkpoint.clone())
        );
        assert_eq!(
            store
                .load_state_dict("latest.json")
                .await
                .expect("load_state_dict"),
            Some(json!({"op": {"temperature": 0.3}}))
        );
    }

    #[tokio::test]
    async fn file_store_rejects_malformed_json() {
        let directory = tempfile::tempdir().expect("tempdir");
        let store = FileCheckpointStore::new(directory.path());
        tokio::fs::write(directory.path().join("bad.json"), b"{not json")
            .await
            .expect("write");
        let error = store
            .load_checkpoint("bad.json")
            .await
            .expect_err("malformed");
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
    }
}
