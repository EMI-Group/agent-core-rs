//! The append-only evidence log below an artifact root.
//!
//! Python stores the rows in `SQLite`; this port appends one canonical JSON object per line,
//! which keeps the log greppable and lets two processes tail it. See [`super`] for the disk
//! layout and the concurrency rules.

use std::collections::{BTreeMap, BTreeSet};
use std::io::ErrorKind as IoErrorKind;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::fs::{self, OpenOptions};
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;
use tracing::warn;

use super::files::{path_error, EVOLUTION_DIR};
use crate::error::{validation, Error, Result};
use crate::llm::canonical_json;
use crate::observation::identity::stable_hash;
/// The schema version Python stamps into the evidence table (`PRAGMA user_version = 1`).
pub const EVIDENCE_SCHEMA: &str = "symphony.graph-evidence.v2";

/// Name of the append-only evidence log.
const EVIDENCE_FILE: &str = "evidence.jsonl";

/// One row of the append-only evidence log.
///
/// The row is the log line itself: it carries the caller's qualification verdict
/// (`eligible` plus `reason`) next to the *normalized* payload, so replaying the log never
/// depends on re-running the qualification rules.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct EvidenceRow {
    /// The row's position in the log, 1-based and monotonic per store.
    pub sequence: u64,
    /// The graph scope the evidence belongs to.
    pub graph_scope_id: String,
    /// The caller's evidence id, unique per scope.
    pub evidence_id: String,
    /// The caller's `observed_at` timestamp, kept verbatim.
    pub observed_at: String,
    /// Whether the row may move an edge weight (`false` for an `audit_only` row).
    pub eligible: bool,
    /// The refusal reason, empty for an eligible row.
    pub reason: String,
    /// The SHA-256 of the canonical payload JSON.
    pub payload_digest: String,
    /// The payload as canonical JSON *text*: object keys sorted, no whitespace.
    ///
    /// This mirrors Python's `payload_json` column, which stores
    /// `json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))`.
    /// Keeping it as text means one row parses without a schema, and two payloads that
    /// differ only in key order store identically.
    pub payload_json: String,
}

impl EvidenceRow {
    /// Parses `payload_json` back into a JSON object.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the stored text is not valid JSON, and
    /// [`Error::Validation`] naming `payload_json` when it is valid JSON but not an object.
    pub fn payload(&self) -> Result<Value> {
        let value: Value = serde_json::from_str(&self.payload_json).map_err(|error| {
            Error::external(error).with_context(format!(
                "decoding the payload of evidence `{}`",
                self.evidence_id
            ))
        })?;
        if value.is_object() {
            Ok(value)
        } else {
            Err(validation("payload_json", "must be a JSON object"))
        }
    }
}

/// One append request: the caller's evidence plus its qualification verdict.
#[derive(Debug, Clone, Copy)]
pub struct EvidenceAppendInput<'a> {
    /// The graph scope the evidence belongs to.
    pub graph_scope_id: &'a str,
    /// The caller's evidence id, unique per scope.
    pub evidence_id: &'a str,
    /// The caller's `observed_at` timestamp, kept verbatim.
    pub observed_at: &'a str,
    /// Whether the row may move an edge weight.
    pub eligible: bool,
    /// The refusal reason, empty for an eligible row.
    pub reason: &'a str,
    /// The normalized payload to persist.
    pub payload: &'a Value,
}

/// The outcome of one append: the row's sequence and whether it was a duplicate.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct EvidenceAppendResult {
    /// The sequence of the stored row (a duplicate reports the original sequence).
    pub sequence: u64,
    /// Whether the row already existed with the same payload digest.
    pub duplicate: bool,
    /// Whether the stored row is eligible for aggregation.
    pub eligible: bool,
    /// The stored row's refusal reason, empty for an eligible row.
    pub reason: String,
}

/// The parsed evidence log: rows in file order plus a `(scope, evidence id)` index.
#[derive(Debug, Default)]
struct EvidenceIndex {
    /// Every valid row, ascending by sequence.
    rows: Vec<EvidenceRow>,
    /// The position of a row inside [`EvidenceIndex::rows`], keyed by scope and evidence id.
    positions: BTreeMap<(String, String), usize>,
}

impl EvidenceIndex {
    /// Reads the whole log, skipping blank and malformed lines.
    async fn read(path: &Path) -> Result<Self> {
        let text = match fs::read_to_string(path).await {
            Ok(text) => text,
            Err(error) if error.kind() == IoErrorKind::NotFound => return Ok(Self::default()),
            Err(error) => return Err(path_error("reading", path, error)),
        };
        let mut index = Self::default();
        for (number, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<EvidenceRow>(line) {
                Ok(row) => index.insert(row),
                Err(error) => warn!(
                    path = %path.display(),
                    line = number + 1,
                    %error,
                    "skipping a malformed Symphony evidence row"
                ),
            }
        }
        Ok(index)
    }

    /// Appends `row` to the in-memory index and registers its key.
    fn insert(&mut self, row: EvidenceRow) {
        self.positions.insert(
            (row.graph_scope_id.clone(), row.evidence_id.clone()),
            self.rows.len(),
        );
        self.rows.push(row);
    }

    /// Returns the row registered for `key`.
    fn get(&self, key: &(String, String)) -> Option<&EvidenceRow> {
        self.positions
            .get(key)
            .and_then(|position| self.rows.get(*position))
    }

    /// Returns the sequence a newly appended row takes: one past the last row.
    fn next_sequence(&self) -> u64 {
        self.rows
            .last()
            .map_or(1, |row| row.sequence.saturating_add(1))
    }
}

/// The lazily loaded evidence index.
#[derive(Debug, Default)]
struct EvidenceState {
    /// Whether [`EvidenceState::index`] reflects the log on disk.
    loaded: bool,
    /// The parsed log.
    index: EvidenceIndex,
}

/// The append-only evidence log of one artifact root.
///
/// Python stores the rows in `SQLite`; this port appends one canonical JSON object per
/// line, which keeps the log greppable and lets two processes tail it. Deduplication is on
/// `(graph_scope_id, evidence_id)`: the same payload digest re-reports the original row,
/// while a *different* payload for a used id is a validation error rather than a silent
/// overwrite.
#[derive(Debug, Clone)]
pub struct EvidenceStore {
    /// `<root>/evolution`.
    root: PathBuf,
    /// `<root>/evolution/evidence.jsonl`.
    path: PathBuf,
    /// The lazily loaded index, shared by every clone of this store.
    state: Arc<Mutex<EvidenceState>>,
}

impl EvidenceStore {
    /// Builds the store for an artifact root; the log lives at `<root>/evolution/evidence.jsonl`.
    ///
    /// Nothing is read or created here: a read of a missing root answers an empty log, and
    /// the `evolution` directory is created by the first append.
    #[must_use]
    pub fn new(root: impl AsRef<Path>) -> Self {
        let root = root.as_ref().join(EVOLUTION_DIR);
        let path = root.join(EVIDENCE_FILE);
        Self {
            root,
            path,
            state: Arc::new(Mutex::new(EvidenceState::default())),
        }
    }

    /// The log file this store appends to.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Appends one evidence row, or reports the existing row on a replay.
    ///
    /// The sequence is `last_sequence + 1`, seeded from the log itself, so reopening a root
    /// continues the numbering and nothing depends on a clock or a global counter.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] naming `evidence_id` when the id was already used for a
    /// different payload, [`Error::External`] when the log cannot be read or appended to,
    /// and [`Error::External`] when a row cannot be encoded.
    pub async fn append(&self, input: EvidenceAppendInput<'_>) -> Result<EvidenceAppendResult> {
        let key = (
            input.graph_scope_id.to_string(),
            input.evidence_id.to_string(),
        );
        let payload_digest = stable_hash(input.payload);

        let mut guard = self.state.lock().await;
        if !guard.loaded {
            guard.index = EvidenceIndex::read(&self.path).await?;
            guard.loaded = true;
        }
        let state = &mut *guard;
        if let Some(row) = state.index.get(&key) {
            if row.payload_digest != payload_digest {
                return Err(validation(
                    "evidence_id",
                    "already refers to different evidence",
                ));
            }
            return Ok(EvidenceAppendResult {
                sequence: row.sequence,
                duplicate: true,
                eligible: row.eligible,
                reason: row.reason.clone(),
            });
        }

        let row = EvidenceRow {
            sequence: state.index.next_sequence(),
            graph_scope_id: key.0,
            evidence_id: key.1,
            observed_at: input.observed_at.to_string(),
            eligible: input.eligible,
            reason: input.reason.to_string(),
            payload_digest,
            payload_json: canonical_json(input.payload),
        };
        let mut line = serde_json::to_string(&row).map_err(|error| {
            Error::external(error).with_context("encoding a Symphony evidence row")
        })?;
        line.push('\n');

        fs::create_dir_all(&self.root)
            .await
            .map_err(|error| path_error("creating", &self.root, error))?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .await
            .map_err(|error| path_error("opening", &self.path, error))?;
        file.write_all(line.as_bytes())
            .await
            .map_err(|error| path_error("writing", &self.path, error))?;
        file.flush()
            .await
            .map_err(|error| path_error("flushing", &self.path, error))?;

        let result = EvidenceAppendResult {
            sequence: row.sequence,
            duplicate: false,
            eligible: row.eligible,
            reason: row.reason.clone(),
        };
        state.index.insert(row);
        Ok(result)
    }

    /// Returns the rows of one scope with `sequence > after`, ascending.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the log cannot be read.
    pub async fn iter_after(&self, graph_scope_id: &str, after: u64) -> Result<Vec<EvidenceRow>> {
        let rows = self.snapshot().await?;
        Ok(rows
            .into_iter()
            .filter(|row| row.graph_scope_id == graph_scope_id && row.sequence > after)
            .collect())
    }

    /// Returns the distinct graph scope ids of the log, sorted.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the log cannot be read.
    pub async fn scope_ids(&self) -> Result<Vec<String>> {
        let unique: BTreeSet<String> = self
            .snapshot()
            .await?
            .into_iter()
            .map(|row| row.graph_scope_id)
            .collect();
        Ok(unique.into_iter().collect())
    }

    /// Returns every row, ascending by sequence.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the log cannot be read.
    pub async fn rows(&self) -> Result<Vec<EvidenceRow>> {
        self.snapshot().await
    }

    /// Returns the number of rows in the log.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the log cannot be read.
    pub async fn len(&self) -> Result<usize> {
        Ok(self.snapshot().await?.len())
    }

    /// Reports whether the log holds no rows.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the log cannot be read.
    pub async fn is_empty(&self) -> Result<bool> {
        self.len().await.map(|len| len == 0)
    }

    /// Returns the row of one scope and evidence id, when the log holds it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the log cannot be read.
    pub async fn lookup(
        &self,
        graph_scope_id: &str,
        evidence_id: &str,
    ) -> Result<Option<EvidenceRow>> {
        let key = (graph_scope_id.to_string(), evidence_id.to_string());
        let mut guard = self.state.lock().await;
        if !guard.loaded {
            guard.index = EvidenceIndex::read(&self.path).await?;
            guard.loaded = true;
        }
        Ok(guard.index.get(&key).cloned())
    }

    /// Drops the in-memory index so the next access re-reads the log.
    ///
    /// # Errors
    ///
    /// This implementation never fails; the signature keeps the reload fallible for the
    /// backends that have to reopen a connection.
    pub async fn reload(&self) -> Result<()> {
        let mut guard = self.state.lock().await;
        *guard = EvidenceState::default();
        Ok(())
    }

    /// Clones every row, loading the index first when it is still empty.
    async fn snapshot(&self) -> Result<Vec<EvidenceRow>> {
        let mut guard = self.state.lock().await;
        if !guard.loaded {
            guard.index = EvidenceIndex::read(&self.path).await?;
            guard.loaded = true;
        }
        Ok(guard.index.rows.clone())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    /// The payload of one test evidence row.
    fn sample_payload() -> Value {
        json!({"edge": {"source_id": "a", "target_id": "b"}, "success": true})
    }

    /// Builds an append request for a valid row.
    fn append_input<'a>(
        scope: &'a str,
        evidence_id: &'a str,
        payload: &'a Value,
    ) -> EvidenceAppendInput<'a> {
        EvidenceAppendInput {
            graph_scope_id: scope,
            evidence_id,
            observed_at: "2024-03-01T00:30:00Z",
            eligible: true,
            reason: "",
            payload,
        }
    }

    #[tokio::test]
    async fn sequences_start_at_one_and_increase() {
        let temp = TempDir::new().unwrap();
        let store = EvidenceStore::new(temp.path());
        let payload = sample_payload();

        let first = store
            .append(append_input("scope-a", "e1", &payload))
            .await
            .unwrap();
        let second = store
            .append(append_input("scope-a", "e2", &payload))
            .await
            .unwrap();

        assert_eq!(first.sequence, 1);
        assert!(!first.duplicate);
        assert_eq!(second.sequence, 2);
        assert_eq!(store.len().await.unwrap(), 2);
        assert!(!store.is_empty().await.unwrap());
        assert_eq!(store.path(), temp.path().join("evolution/evidence.jsonl"));
    }

    #[tokio::test]
    async fn reopen_reads_the_same_rows_and_continues_numbering() {
        let temp = TempDir::new().unwrap();
        let payload = sample_payload();
        let first = {
            let store = EvidenceStore::new(temp.path());
            store
                .append(append_input("scope-a", "e1", &payload))
                .await
                .unwrap();
            store
                .append(append_input("scope-a", "e2", &payload))
                .await
                .unwrap();
            store.rows().await.unwrap()
        };

        let reopened = EvidenceStore::new(temp.path());
        assert_eq!(reopened.rows().await.unwrap(), first);
        let third = reopened
            .append(append_input("scope-a", "e3", &payload))
            .await
            .unwrap();
        assert_eq!(
            third.sequence, 3,
            "numbering resumes from the log, not from zero"
        );
    }

    #[tokio::test]
    async fn duplicate_append_is_reported_and_appends_nothing() {
        let temp = TempDir::new().unwrap();
        let store = EvidenceStore::new(temp.path());
        let payload = sample_payload();

        let first = store
            .append(append_input("scope-a", "e1", &payload))
            .await
            .unwrap();
        let replay = store
            .append(append_input("scope-a", "e1", &payload))
            .await
            .unwrap();

        assert!(replay.duplicate);
        assert_eq!(replay.sequence, first.sequence);
        assert_eq!(replay.eligible, first.eligible);
        assert_eq!(replay.reason, first.reason);
        assert_eq!(store.len().await.unwrap(), 1);
        let text = std::fs::read_to_string(store.path()).unwrap();
        assert_eq!(
            text.lines().count(),
            1,
            "a duplicate must not append a line"
        );
    }

    #[tokio::test]
    async fn reused_evidence_id_with_other_payload_is_rejected() {
        let temp = TempDir::new().unwrap();
        let store = EvidenceStore::new(temp.path());
        store
            .append(append_input("scope-a", "e1", &sample_payload()))
            .await
            .unwrap();

        let other = json!({"edge": {"source_id": "a", "target_id": "c"}});
        let error = store
            .append(append_input("scope-a", "e1", &other))
            .await
            .expect_err("a used id may not change payload");

        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
        assert!(error.message().contains("evidence_id"));
        assert_eq!(store.len().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn scopes_are_isolated_and_iter_after_is_exclusive() {
        let temp = TempDir::new().unwrap();
        let store = EvidenceStore::new(temp.path());
        let payload = sample_payload();
        for id in ["e1", "e2", "e3"] {
            store
                .append(append_input("scope-a", id, &payload))
                .await
                .unwrap();
            store
                .append(append_input("scope-b", id, &payload))
                .await
                .unwrap();
        }

        let all = store.iter_after("scope-a", 0).await.unwrap();
        assert_eq!(
            all.iter().map(|row| row.sequence).collect::<Vec<_>>(),
            vec![1, 3, 5]
        );
        assert!(all.iter().all(|row| row.graph_scope_id == "scope-a"));

        let later = store.iter_after("scope-a", 3).await.unwrap();
        assert_eq!(
            later.iter().map(|row| row.sequence).collect::<Vec<_>>(),
            vec![5]
        );
        assert_eq!(
            store.iter_after("scope-a", 5).await.unwrap(),
            Vec::<EvidenceRow>::new()
        );
        assert_eq!(store.iter_after("scope-b", 0).await.unwrap().len(), 3);
    }

    #[tokio::test]
    async fn scope_ids_are_distinct_and_sorted() {
        let temp = TempDir::new().unwrap();
        let store = EvidenceStore::new(temp.path());
        let payload = sample_payload();
        for (scope, id) in [("scope-b", "e1"), ("scope-a", "e1"), ("scope-b", "e2")] {
            store
                .append(append_input(scope, id, &payload))
                .await
                .unwrap();
        }

        assert_eq!(store.scope_ids().await.unwrap(), vec!["scope-a", "scope-b"]);
    }

    #[tokio::test]
    async fn lookup_reload_and_missing_root_behave() {
        let temp = TempDir::new().unwrap();
        let store = EvidenceStore::new(temp.path());
        assert!(store.is_empty().await.unwrap());
        assert_eq!(store.len().await.unwrap(), 0);
        assert_eq!(store.rows().await.unwrap(), Vec::<EvidenceRow>::new());
        assert_eq!(store.scope_ids().await.unwrap(), Vec::<String>::new());
        assert!(store.lookup("scope-a", "e1").await.unwrap().is_none());
        assert!(
            !store.path().exists(),
            "reading a missing root creates nothing"
        );

        let payload = sample_payload();
        store
            .append(append_input("scope-a", "e1", &payload))
            .await
            .unwrap();
        let row = store.lookup("scope-a", "e1").await.unwrap().unwrap();
        assert_eq!(row.sequence, 1);
        assert!(store.lookup("scope-b", "e1").await.unwrap().is_none());

        // A second store sees the row once its own index is dropped and re-read.
        let other = EvidenceStore::new(temp.path());
        assert_eq!(other.len().await.unwrap(), 1);
        other.reload().await.unwrap();
        assert_eq!(other.len().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn malformed_lines_are_skipped() {
        let temp = TempDir::new().unwrap();
        let store = EvidenceStore::new(temp.path());
        store
            .append(append_input("scope-a", "e1", &sample_payload()))
            .await
            .unwrap();
        let text = std::fs::read_to_string(store.path()).unwrap();
        let mut broken = text.clone();
        broken.push_str("{\"sequence\": 2,\n");
        broken.push('\n');
        std::fs::write(store.path(), broken).unwrap();

        let reopened = EvidenceStore::new(temp.path());
        let rows = reopened.rows().await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].evidence_id, "e1");
    }

    #[tokio::test]
    async fn payload_round_trips_through_the_row() {
        let temp = TempDir::new().unwrap();
        let store = EvidenceStore::new(temp.path());
        let payload = json!({"z": [1, 2], "a": {"nested": true}});
        store
            .append(append_input("scope-a", "e1", &payload))
            .await
            .unwrap();

        let row = store.lookup("scope-a", "e1").await.unwrap().unwrap();
        assert_eq!(row.payload().unwrap(), payload);
        assert_eq!(row.payload_digest, stable_hash(&payload));
        assert_eq!(row.payload_json, canonical_json(&payload));
        assert!(
            !row.payload_json.contains(' '),
            "the stored text is compact"
        );

        let mut broken = row;
        broken.payload_json = "[]".to_string();
        let error = broken.payload().expect_err("a payload must be an object");
        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
    }
}
