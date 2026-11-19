//! Rollout persistence (Python `store/`).
//!
//! Three pieces are ported: the [`RolloutPersistence`] seam
//! (`store/base.py`), a JSONL [`FileRolloutStore`] (`store/file_store.py`) and a
//! no-op [`NullRolloutStore`] (`store/null_store.py`).
//!
//! # File layout
//!
//! ```text
//! <root>/train/rollouts/steps_000000_000099.jsonl
//! <root>/val/rollouts/steps_000000_000099.jsonl
//! <root>/step_summaries/steps_000000_000099.jsonl
//! ```
//!
//! Files are append-only and split into `flush_interval`-sized step ranges, so
//! `lo = step / flush_interval * flush_interval` and
//! `hi = lo + flush_interval - 1`. Every line is one JSON object with Python's
//! field set. Python's wall-clock `timestamp` field is deliberately omitted: it
//! would make otherwise identical runs produce different files.
//!
//! The store uses `tokio::fs`, so the blocking `open`/`write`/`read` syscalls
//! run on tokio's blocking pool instead of a runtime worker thread.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use openjiuwen_common::error::{Error, Result};

use crate::rl::RolloutMessage;

/// Which rollout stream a record belongs to.
///
/// Python passes `"train"`/`"val"` strings; the enum keeps the choice typed and
/// maps anything that is not `val` to the training directory, as Python does.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RolloutPhase {
    /// Training rollouts.
    #[default]
    Train,
    /// Validation rollouts.
    Val,
}

impl RolloutPhase {
    /// The stable directory/JSON name of the phase.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Val => "val",
        }
    }
}

/// The persistence seam of the offline runtime.
///
/// Every method is `async` because a real backend may reach a database or a
/// remote store; the file and null implementations below are the offline ones.
/// Implementations must be safe to share across rollout workers
/// (`Send + Sync`).
#[async_trait]
pub trait RolloutPersistence: Send + Sync {
    /// Persists one rollout with its complete trajectory.
    ///
    /// `phase` selects the output stream, `step` the file within it.
    async fn save_rollout(
        &self,
        step: usize,
        task_id: &str,
        rollout: &RolloutMessage,
        phase: RolloutPhase,
    ) -> Result<()>;

    /// Persists the metrics of one training or validation step.
    async fn save_step_summary(&self, step: usize, metrics: &BTreeMap<String, Value>)
        -> Result<()>;

    /// Queries historical rollouts by exact top-level field matches.
    ///
    /// A `limit` of `0` means unlimited; any other value stops the scan as soon
    /// as that many matches were collected.
    async fn query_rollouts(
        &self,
        filters: &BTreeMap<String, Value>,
        limit: usize,
    ) -> Result<Vec<Value>>;

    /// Releases resources. The offline implementations hold none, but a remote
    /// backend may need to flush or disconnect here.
    async fn close(&self) -> Result<()>;
}

/// The JSONL document written for one rollout.
fn rollout_document(step: usize, task_id: &str, rollout: &RolloutMessage) -> Value {
    let mut document = serde_json::Map::new();
    document.insert("step".to_owned(), json!(step));
    document.insert("task_id".to_owned(), json!(task_id));
    document.insert("origin_task_id".to_owned(), json!(rollout.origin_task_id));
    document.insert("rollout_id".to_owned(), json!(rollout.rollout_id));
    document.insert("turns".to_owned(), rollout.rollout_info.clone());
    document.insert("reward_list".to_owned(), json!(rollout.reward_list));
    document.insert("global_reward".to_owned(), json!(rollout.global_reward));
    document.insert("turn_count".to_owned(), json!(rollout.turn_count));
    document.insert("round_num".to_owned(), json!(rollout.round_num));
    document.insert("start_time".to_owned(), json!(rollout.start_time));
    document.insert("end_time".to_owned(), json!(rollout.end_time));
    Value::Object(document)
}

/// The JSONL document written for one step summary.
fn summary_document(step: usize, metrics: &BTreeMap<String, Value>) -> Value {
    let mut document = serde_json::Map::new();
    document.insert("step".to_owned(), json!(step));
    document.insert("metrics".to_owned(), json!(metrics));
    Value::Object(document)
}

/// Appends one JSON line to `path`, creating parents and the file as needed.
async fn append_line(path: &Path, document: &Value) -> Result<()> {
    let mut line = serde_json::to_string(document).map_err(Error::external)?;
    line.push('\n');
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(Error::external)?;
    }
    use tokio::io::AsyncWriteExt;
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .await
        .map_err(Error::external)?;
    file.write_all(line.as_bytes())
        .await
        .map_err(Error::external)?;
    file.flush().await.map_err(Error::external)?;
    Ok(())
}

/// The step range covered by one JSONL file.
fn step_range(step: usize, flush_interval: usize) -> (usize, usize) {
    let flush_interval = flush_interval.max(1);
    let low = step / flush_interval * flush_interval;
    (low, low + flush_interval - 1)
}

/// A filesystem [`RolloutPersistence`] writing append-only JSONL files.
///
/// The store is a plain value: it can be cloned, shared through an `Arc`, and
/// constructed before the output directory exists (directories are created on
/// the first write, or eagerly by [`FileRolloutStore::open`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileRolloutStore {
    root: PathBuf,
    flush_interval: usize,
}

impl FileRolloutStore {
    /// Creates a store rooted at `root` with `flush_interval` steps per file.
    ///
    /// A `flush_interval` below one is clamped to one, matching Python's
    /// `max(1, flush_interval)`. No I/O happens here; see [`Self::open`] to
    /// create the directory tree up front.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>, flush_interval: usize) -> Self {
        Self {
            root: root.into(),
            flush_interval: flush_interval.max(1),
        }
    }

    /// Creates a store and its directory tree.
    ///
    /// # Errors
    ///
    /// The underlying filesystem error when the directories cannot be created.
    pub async fn open(root: impl Into<PathBuf>, flush_interval: usize) -> Result<Self> {
        let store = Self::new(root, flush_interval);
        for directory in [
            store.rollout_dir(RolloutPhase::Train),
            store.rollout_dir(RolloutPhase::Val),
            store.summary_dir(),
        ] {
            tokio::fs::create_dir_all(&directory)
                .await
                .map_err(Error::external)?;
        }
        Ok(store)
    }

    /// The root directory of the store.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The number of steps per JSONL file.
    #[must_use]
    pub fn flush_interval(&self) -> usize {
        self.flush_interval
    }

    /// The rollout directory of `phase`.
    #[must_use]
    pub fn rollout_dir(&self, phase: RolloutPhase) -> PathBuf {
        self.root.join(phase.as_str()).join("rollouts")
    }

    /// The step-summary directory.
    #[must_use]
    pub fn summary_dir(&self) -> PathBuf {
        self.root.join("step_summaries")
    }

    /// The JSONL file a rollout of `phase` at `step` is appended to.
    #[must_use]
    pub fn rollout_path(&self, phase: RolloutPhase, step: usize) -> PathBuf {
        self.rollout_dir(phase)
            .join(file_name(step, self.flush_interval))
    }

    /// The JSONL file a step summary at `step` is appended to.
    #[must_use]
    pub fn summary_path(&self, step: usize) -> PathBuf {
        self.summary_dir()
            .join(file_name(step, self.flush_interval))
    }

    /// Reads the summaries persisted for `step` (used by analysis tooling).
    ///
    /// # Errors
    ///
    /// The underlying filesystem error when the file cannot be read.
    pub async fn read_summaries(&self, step: usize) -> Result<Vec<Value>> {
        read_jsonl(&self.summary_path(step)).await
    }

    /// Whether no rollout file exists for `phase` yet.
    #[must_use]
    pub fn has_rollouts(&self, phase: RolloutPhase) -> bool {
        self.rollout_dir(phase).is_dir()
    }
}

fn file_name(step: usize, flush_interval: usize) -> String {
    let (low, high) = step_range(step, flush_interval);
    format!("steps_{low:06}_{high:06}.jsonl")
}

async fn read_jsonl(path: &Path) -> Result<Vec<Value>> {
    let text = match tokio::fs::read_to_string(path).await {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(Error::external(error)),
    };
    Ok(text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .collect())
}

#[async_trait]
impl RolloutPersistence for FileRolloutStore {
    async fn save_rollout(
        &self,
        step: usize,
        task_id: &str,
        rollout: &RolloutMessage,
        phase: RolloutPhase,
    ) -> Result<()> {
        let document = rollout_document(step, task_id, rollout);
        append_line(&self.rollout_path(phase, step), &document).await
    }

    async fn save_step_summary(
        &self,
        step: usize,
        metrics: &BTreeMap<String, Value>,
    ) -> Result<()> {
        let document = summary_document(step, metrics);
        append_line(&self.summary_path(step), &document).await
    }

    async fn query_rollouts(
        &self,
        filters: &BTreeMap<String, Value>,
        limit: usize,
    ) -> Result<Vec<Value>> {
        let mut results = Vec::new();
        for phase in [RolloutPhase::Train, RolloutPhase::Val] {
            let directory = self.rollout_dir(phase);
            let mut files = match tokio::fs::read_dir(&directory).await {
                Ok(entries) => entries,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => return Err(Error::external(error)),
            };
            let mut paths: Vec<PathBuf> = Vec::new();
            while let Some(entry) = files.next_entry().await.map_err(Error::external)? {
                let path = entry.path();
                if path.extension().is_some_and(|ext| ext == "jsonl") {
                    paths.push(path);
                }
            }
            // Newest file first: the step range is zero padded, so the file
            // name sorts numerically.
            paths.sort_by(|left, right| right.cmp(left));
            for path in paths {
                for document in read_jsonl(&path).await? {
                    let matches = filters.iter().all(|(key, expected)| {
                        document
                            .get(key.as_str())
                            .is_some_and(|actual| actual == expected)
                    });
                    if matches {
                        results.push(document);
                        if limit > 0 && results.len() >= limit {
                            return Ok(results);
                        }
                    }
                }
            }
        }
        Ok(results)
    }

    async fn close(&self) -> Result<()> {
        Ok(())
    }
}

/// A [`RolloutPersistence`] that performs no I/O.
///
/// Used when persistence is disabled: every write succeeds silently and every
/// query is empty. Python's null store swallows writes the same way; here the
/// caller opts in explicitly, so no capability is hidden behind a no-op that
/// looks like a working store.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NullRolloutStore;

impl NullRolloutStore {
    /// Creates the null store.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl RolloutPersistence for NullRolloutStore {
    async fn save_rollout(
        &self,
        _step: usize,
        _task_id: &str,
        _rollout: &RolloutMessage,
        _phase: RolloutPhase,
    ) -> Result<()> {
        Ok(())
    }

    async fn save_step_summary(
        &self,
        _step: usize,
        _metrics: &BTreeMap<String, Value>,
    ) -> Result<()> {
        Ok(())
    }

    async fn query_rollouts(
        &self,
        _filters: &BTreeMap<String, Value>,
        _limit: usize,
    ) -> Result<Vec<Value>> {
        Ok(Vec::new())
    }

    async fn close(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn rollout(rollout_id: &str, task_id: &str, reward: f64) -> RolloutMessage {
        RolloutMessage {
            task_id: task_id.to_owned(),
            origin_task_id: Some(format!("origin-{task_id}")),
            rollout_id: rollout_id.to_owned(),
            start_time: None,
            end_time: None,
            rollout_info: json!([{"turn_id": 0, "output_response_ids": [7]}]),
            reward_list: vec![reward],
            global_reward: reward,
            turn_count: 1,
            round_num: 3,
        }
    }

    #[test]
    fn file_names_follow_the_python_step_ranges() {
        assert_eq!(file_name(0, 100), "steps_000000_000099.jsonl");
        assert_eq!(file_name(99, 100), "steps_000000_000099.jsonl");
        assert_eq!(file_name(100, 100), "steps_000100_000199.jsonl");
        // A flush interval below one is clamped to one, so every step is its own file.
        assert_eq!(file_name(7, 0), "steps_000007_000007.jsonl");
        assert_eq!(file_name(7, 1), "steps_000007_000007.jsonl");
        assert_eq!(step_range(250, 100), (200, 299));
    }

    #[tokio::test]
    async fn save_rollout_round_trips_through_query() {
        let temp = TempDir::new().expect("temp dir");
        let store = FileRolloutStore::open(temp.path(), 100)
            .await
            .expect("open store");
        assert_eq!(store.root(), temp.path());
        assert_eq!(store.flush_interval(), 100);

        store
            .save_rollout(3, "t-1", &rollout("r-1", "t-1", 1.0), RolloutPhase::Train)
            .await
            .expect("save train");
        store
            .save_rollout(3, "t-2", &rollout("r-2", "t-2", 0.0), RolloutPhase::Val)
            .await
            .expect("save val");

        let train_path = store.rollout_path(RolloutPhase::Train, 3);
        assert!(train_path.ends_with("train/rollouts/steps_000000_000099.jsonl"));
        let val_path = store.rollout_path(RolloutPhase::Val, 3);
        assert!(val_path.ends_with("val/rollouts/steps_000000_000099.jsonl"));
        assert!(store.has_rollouts(RolloutPhase::Train));

        let all = store
            .query_rollouts(&BTreeMap::new(), 0)
            .await
            .expect("query all");
        assert_eq!(all.len(), 2);
        // Train is scanned before val, as in Python.
        assert_eq!(all[0]["rollout_id"], json!("r-1"));
        assert_eq!(all[1]["rollout_id"], json!("r-2"));

        let document = &all[0];
        for key in [
            "step",
            "task_id",
            "origin_task_id",
            "rollout_id",
            "turns",
            "reward_list",
            "global_reward",
            "turn_count",
            "round_num",
            "start_time",
            "end_time",
        ] {
            assert!(document.get(key).is_some(), "missing {key}");
        }
        assert!(document.get("timestamp").is_none(), "timestamp is omitted");
        assert_eq!(document["round_num"], json!(3));
        assert_eq!(document["turns"][0]["turn_id"], json!(0));
    }

    #[tokio::test]
    async fn query_filters_exactly_and_stops_at_the_limit() {
        let temp = TempDir::new().expect("temp dir");
        let store = FileRolloutStore::new(temp.path(), 2);
        for step in 0..4 {
            for index in 0..3 {
                store
                    .save_rollout(
                        step,
                        "t-1",
                        &rollout(&format!("r-{step}-{index}"), "t-1", step as f64),
                        RolloutPhase::Train,
                    )
                    .await
                    .expect("save");
            }
        }

        let filters = BTreeMap::from([("global_reward".to_owned(), json!(1.0))]);
        let matched = store
            .query_rollouts(&filters, 0)
            .await
            .expect("query filtered");
        assert_eq!(matched.len(), 3);
        assert!(matched.iter().all(|doc| doc["global_reward"] == json!(1.0)));

        let limited = store
            .query_rollouts(&BTreeMap::new(), 2)
            .await
            .expect("query limited");
        assert_eq!(limited.len(), 2);
        // Newest file first: step 2-3 is written into steps_000002_000003.
        assert_eq!(limited[0]["step"], json!(2));

        let none = store
            .query_rollouts(
                &BTreeMap::from([("rollout_id".to_owned(), json!("missing"))]),
                0,
            )
            .await
            .expect("query none");
        assert!(none.is_empty(), "no rolls in an empty filter");
    }

    #[tokio::test]
    async fn summaries_are_appended_per_step() {
        let temp = TempDir::new().expect("temp dir");
        let store = FileRolloutStore::open(temp.path(), 10).await.expect("open");
        let mut first = BTreeMap::new();
        first.insert("training/global_step".to_owned(), json!(1));
        store.save_step_summary(1, &first).await.expect("summary");
        let mut second = BTreeMap::new();
        second.insert("training/global_step".to_owned(), json!(2));
        store.save_step_summary(2, &second).await.expect("summary");

        let summaries = store.read_summaries(1).await.expect("read");
        assert_eq!(summaries.len(), 2);
        assert_eq!(summaries[0]["step"], json!(1));
        assert_eq!(summaries[0]["metrics"]["training/global_step"], json!(1));
        assert_eq!(summaries[1]["metrics"]["training/global_step"], json!(2));
        assert!(summaries[0].get("timestamp").is_none());

        assert!(
            store.read_summaries(900).await.expect("absent").is_empty(),
            "absent step has no summary"
        );
        store.close().await.expect("close");
    }

    #[tokio::test]
    async fn a_missing_directory_queries_empty() {
        let temp = TempDir::new().expect("temp dir");
        let store = FileRolloutStore::new(temp.path().join("absent"), 10);
        assert!(!store.has_rollouts(RolloutPhase::Val));
        assert!(
            store
                .query_rollouts(&BTreeMap::new(), 0)
                .await
                .expect("query")
                .is_empty(),
            "a missing directory holds no rollout"
        );
    }

    #[tokio::test]
    async fn malformed_lines_are_skipped() {
        let temp = TempDir::new().expect("temp dir");
        let store = FileRolloutStore::open(temp.path(), 10).await.expect("open");
        store
            .save_rollout(0, "t-1", &rollout("r-1", "t-1", 1.0), RolloutPhase::Train)
            .await
            .expect("save");
        let path = store.rollout_path(RolloutPhase::Train, 0);
        let mut text = tokio::fs::read_to_string(&path).await.expect("read");
        text.push_str("not json\n\n");
        tokio::fs::write(&path, text).await.expect("write");

        let results = store
            .query_rollouts(&BTreeMap::new(), 0)
            .await
            .expect("query");
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn the_null_store_is_a_pure_no_op() {
        let store = NullRolloutStore::new();
        store
            .save_rollout(0, "t", &rollout("r", "t", 1.0), RolloutPhase::Val)
            .await
            .expect("no-op");
        store
            .save_step_summary(0, &BTreeMap::new())
            .await
            .expect("no-op");
        assert!(
            store
                .query_rollouts(&BTreeMap::new(), 10)
                .await
                .expect("no-op")
                .is_empty(),
            "the null store drops every rollout"
        );
        store.close().await.expect("no-op");
        assert_eq!(NullRolloutStore, NullRolloutStore);
    }
}
