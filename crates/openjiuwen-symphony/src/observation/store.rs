//! Durable evidence and immutable observation/merged revision storage.
//!
//! Ports Python `openjiuwen.symphony.observation.store` (v0.1.20). Python keeps the
//! evidence log in `SQLite` behind a `filelock`, and the derived revisions as immutably
//! written JSON documents. This port keeps the same observable contract with no database
//! and no external lock: the evidence log is an append-only `JSONL` file, and every
//! revision document is written atomically (a `<name>.tmp` sibling plus a rename).
//!
//! # On-disk layout
//!
//! ```text
//! <root>/evolution/evidence.jsonl                      # one row per line, append-only
//! <root>/evolution/scopes/<sha256(scope)[:20]>/
//!   current.json                                       # pointer to the active revisions
//!   observations/<observation-revision>/overlay.json   # immutable
//!   merged/<merged-revision>/manifest.json             # immutable
//! ```
//!
//! The scope directory name is the first 20 hex characters of
//! `stable_hash({"graph_scope_id": <scope>})` — the hash is over the JSON *object*, exactly
//! as Python computes it.
//!
//! # Revision ids are content hashes
//!
//! `observation-<sha256(canonical overlay)[:20]>` and
//! `merged-<sha256(schema, scope, static_revision, observation_revision)[:20]>`. Both
//! deliberately exclude `generated_at` (and the merged id also excludes
//! `high_water_sequence`), so re-aggregating identical state reproduces the same ids and a
//! retried publish is idempotent rather than an error: a revision document whose stored
//! content equals the requested content is left untouched, and only a *different* document
//! at the same path is refused.
//!
//! # Concurrency
//!
//! Python serializes appends with `FileLock`. This port serializes them with a
//! [`tokio::sync::Mutex`] that is held across the append's own file I/O, which is the
//! in-process equivalent; a `std::sync` guard is never held across an `.await`. The
//! revision store serializes [`RevisionStore::publish`] the same way, so the overlay, the
//! manifest and the `current.json` pointer can never interleave with a second publisher.
//! The evidence log keeps its parsed rows in memory: the first access reads the whole file
//! (a log is bounded by one row per submitted evidence record), and mutating calls update
//! the index under the same lock.

use std::collections::{BTreeMap, BTreeSet};
use std::io::ErrorKind as IoErrorKind;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use tokio::fs::{self, OpenOptions};
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;
use tracing::{debug, warn};

use crate::error::{validation, Error, Result};
use crate::llm::canonical_json;
use crate::observation::contracts::GraphSnapshot;
use crate::observation::identity::stable_hash;

/// The schema version Python stamps into the evidence table (`PRAGMA user_version = 1`).
pub const EVIDENCE_SCHEMA: &str = "symphony.graph-evidence.v2";

/// The `schema_version` of a stored observation overlay document.
pub const OBSERVATION_SCHEMA: &str = "symphony.graph-observation.v2";

/// The `schema_version` of a stored merged manifest and of the `current.json` pointer.
pub const MERGED_SCHEMA: &str = "symphony.graph-merged.v2";

/// The artifact sub-directory holding the log and every derived revision.
const EVOLUTION_DIR: &str = "evolution";

/// Name of the append-only evidence log.
const EVIDENCE_FILE: &str = "evidence.jsonl";

/// The sub-directory holding one directory per graph scope.
const SCOPES_DIR: &str = "scopes";

/// The sub-directory holding the immutable observation overlays of one scope.
const OBSERVATIONS_DIR: &str = "observations";

/// The sub-directory holding the immutable merged manifests of one scope.
const MERGED_DIR: &str = "merged";

/// File name of one observation revision's overlay document.
const OVERLAY_FILE: &str = "overlay.json";

/// File name of one merged revision's manifest document.
const MANIFEST_FILE: &str = "manifest.json";

/// File name of the scope's snapshot pointer document.
const CURRENT_FILE: &str = "current.json";

/// The number of hash characters a scope directory name keeps.
const SCOPE_HASH_LEN: usize = 20;

/// The number of hash characters a revision id keeps.
const REVISION_HASH_LEN: usize = 20;

/// The id prefix of an observation revision.
const OBSERVATION_PREFIX: &str = "observation-";

/// The id prefix of a merged revision.
const MERGED_PREFIX: &str = "merged-";

/// The `observation_revision` JSON key shared by the overlay, the manifest and the pointer.
const OBSERVATION_REVISION_FIELD: &str = "observation_revision";

/// The `merged_revision` JSON key shared by the manifest and the pointer.
const MERGED_REVISION_FIELD: &str = "merged_revision";

/// The `generated_at` JSON key, which never participates in a revision id.
const GENERATED_AT_FIELD: &str = "generated_at";

/// The `high_water_sequence` JSON key of an overlay, manifest or pointer.
const HIGH_WATER_SEQUENCE_FIELD: &str = "high_water_sequence";

/// The `static_revision` JSON key of an overlay, manifest or pointer.
const STATIC_REVISION_FIELD: &str = "static_revision";

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

/// One overlay publication: the caller's aggregated edge state.
#[derive(Debug, Clone, Copy)]
pub struct OverlayInput<'a> {
    /// The graph scope the overlay belongs to.
    pub graph_scope_id: &'a str,
    /// The static `can_feed` graph revision the overlay was built on.
    pub static_revision: &'a str,
    /// The highest evidence sequence the overlay has consumed.
    pub high_water_sequence: u64,
    /// The per-edge overlay, keyed by edge identity (hence written id-sorted).
    pub edges: &'a BTreeMap<String, Value>,
    /// When the overlay was aggregated; metadata only, never part of the revision id.
    pub generated_at: &'a str,
    /// `ready`, or [`crate::observation::contracts::SAFE_FALLBACK`] for a neutral overlay.
    pub projection_status: &'a str,
    /// Diagnostics to record; empty entries are dropped, the rest are sorted and deduped.
    pub diagnostics: &'a [String],
}

/// One merged-publication request: the pair of input revisions to point at.
#[derive(Debug, Clone, Copy)]
pub struct MergedInput<'a> {
    /// The graph scope the manifest belongs to.
    pub graph_scope_id: &'a str,
    /// The static `can_feed` graph revision the overlay was built on.
    pub static_revision: &'a str,
    /// The observation revision the manifest merges.
    pub observation_revision: &'a str,
    /// The highest evidence sequence the snapshot has consumed.
    pub high_water_sequence: u64,
    /// When the manifest was published; metadata only, never part of the revision id.
    pub generated_at: &'a str,
}

/// The scope's `current.json` pointer document.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct CurrentPointer {
    /// The schema version, always [`MERGED_SCHEMA`].
    pub schema_version: String,
    /// The graph scope this pointer belongs to.
    pub graph_scope_id: String,
    /// The static `can_feed` graph revision the snapshot was built on.
    pub static_revision: String,
    /// The observation revision the snapshot's overlay lives in.
    pub observation_revision: String,
    /// The merged revision the snapshot is pinned to.
    pub merged_revision: String,
    /// The highest evidence sequence the snapshot has consumed.
    pub high_water_sequence: u64,
    /// When the snapshot was published; metadata only, never part of a revision id.
    pub generated_at: String,
}

impl CurrentPointer {
    /// Builds the pointer document describing `snapshot`.
    #[must_use]
    pub fn from_snapshot(snapshot: &GraphSnapshot, generated_at: impl Into<String>) -> Self {
        Self {
            schema_version: MERGED_SCHEMA.to_string(),
            graph_scope_id: snapshot.graph_scope_id.clone(),
            static_revision: snapshot.static_revision.clone(),
            observation_revision: snapshot.observation_revision.clone(),
            merged_revision: snapshot.merged_revision.clone(),
            high_water_sequence: snapshot.high_water_sequence,
            generated_at: generated_at.into(),
        }
    }
}

/// Immutable observation and merged revision documents of one artifact root.
///
/// Publishing writes the overlay, then the manifest, then the scope's `current.json`
/// pointer, all atomically, and returns the [`GraphSnapshot`] a planner reads. A revision
/// that already exists with identical content is reused; one that exists with *different*
/// content is a validation error, so a published revision is never rewritten.
#[derive(Debug, Clone)]
pub struct RevisionStore {
    /// `<root>/evolution`.
    root: PathBuf,
    /// Serializes the multi-file publication of one scope.
    publish: Arc<Mutex<()>>,
}

impl RevisionStore {
    /// Builds the store for an artifact root; revisions live under `<root>/evolution/scopes/`.
    #[must_use]
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().join(EVOLUTION_DIR),
            publish: Arc::new(Mutex::new(())),
        }
    }

    /// The scope directory holding `graph_scope_id`'s revisions.
    ///
    /// The directory name is the first 20 hex characters of the scope's JSON object hash,
    /// so two spellings of one scope always map to the same directory and two scopes never
    /// collide in practice.
    #[must_use]
    pub fn scope_root(&self, graph_scope_id: &str) -> PathBuf {
        let scope = json!({ "graph_scope_id": graph_scope_id });
        let digest = short_digest(&stable_hash(&scope), SCOPE_HASH_LEN);
        self.root.join(SCOPES_DIR).join(digest)
    }

    /// Publishes overlay -> merged -> `current.json` and returns the snapshot.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when a target revision already exists with different
    /// content, [`Error::External`] on any I/O or encoding failure, and
    /// [`Error::External`] naming the path when the overlay it just wrote cannot be read
    /// back.
    pub async fn publish(&self, input: &OverlayInput<'_>) -> Result<GraphSnapshot> {
        let _guard = self.publish.lock().await;
        let observation_revision = self.publish_overlay(input).await?;
        let merged_revision = self
            .publish_merged(&MergedInput {
                graph_scope_id: input.graph_scope_id,
                static_revision: input.static_revision,
                observation_revision: &observation_revision,
                high_water_sequence: input.high_water_sequence,
                generated_at: input.generated_at,
            })
            .await?;
        self.write_current(&CurrentPointer {
            schema_version: MERGED_SCHEMA.to_string(),
            graph_scope_id: input.graph_scope_id.to_string(),
            static_revision: input.static_revision.to_string(),
            observation_revision: observation_revision.clone(),
            merged_revision: merged_revision.clone(),
            high_water_sequence: input.high_water_sequence,
            generated_at: input.generated_at.to_string(),
        })
        .await?;

        let overlay_path = self.overlay_path(input.graph_scope_id, &observation_revision);
        let Some(overlay) = read_object(&overlay_path).await? else {
            return Err(missing_document_error(&overlay_path));
        };
        Ok(GraphSnapshot {
            graph_scope_id: input.graph_scope_id.to_string(),
            static_revision: input.static_revision.to_string(),
            observation_revision,
            merged_revision,
            high_water_sequence: input.high_water_sequence,
            overlay: overlay.into_iter().collect(),
        })
    }

    /// Writes `observations/<observation-revision>/overlay.json` immutably.
    ///
    /// Returns the content-derived revision id, so a caller can publish the matching merged
    /// manifest itself.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the revision already exists with different
    /// content, and [`Error::External`] on any I/O or encoding failure.
    pub async fn publish_overlay(&self, input: &OverlayInput<'_>) -> Result<String> {
        let (revision, document) = build_overlay(input);
        let path = self.overlay_path(input.graph_scope_id, &revision);
        write_immutable_json(&path, &document).await?;
        Ok(revision)
    }

    /// Writes `merged/<merged-revision>/manifest.json` immutably and returns the revision id.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the revision already exists with different
    /// content, and [`Error::External`] on any I/O or encoding failure.
    pub async fn publish_merged(&self, input: &MergedInput<'_>) -> Result<String> {
        let (revision, manifest) = build_merged(input);
        let path = self.manifest_path(input.graph_scope_id, &revision);
        write_immutable_json(&path, &manifest).await?;
        Ok(revision)
    }

    /// Writes the scope's `current.json` pointer atomically.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the pointer cannot be encoded or written.
    pub async fn write_current(&self, pointer: &CurrentPointer) -> Result<()> {
        let document = serde_json::to_value(pointer)
            .map_err(|error| Error::external(error).with_context("encoding a Symphony pointer"))?;
        let path = self.scope_root(&pointer.graph_scope_id).join(CURRENT_FILE);
        write_json_atomic(&path, &document).await
    }

    /// Reads the scope's `current.json` pointer.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the document is not a JSON object,
    /// [`Error::External`] when it cannot be read or decoded.
    pub async fn current_pointer(&self, graph_scope_id: &str) -> Result<Option<CurrentPointer>> {
        let path = self.scope_root(graph_scope_id).join(CURRENT_FILE);
        let Some(document) = read_object(&path).await? else {
            return Ok(None);
        };
        let pointer = serde_json::from_value(Value::Object(document)).map_err(|error| {
            Error::external(error).with_context(format!("decoding `{}`", path.display()))
        })?;
        Ok(Some(pointer))
    }

    /// Reads the snapshot `current.json` describes, including its observation overlay.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for a malformed pointer or revision name, and
    /// [`Error::External`] when a document is unreadable, undecodable, or missing while the
    /// pointer names it.
    pub async fn current(&self, graph_scope_id: &str) -> Result<Option<GraphSnapshot>> {
        self.read_snapshot(graph_scope_id, None).await
    }

    /// Reads the overlay document of one observation revision.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] naming `observation_revision` for an illegal revision
    /// name, [`Error::Validation`] when the document is not a JSON object, and
    /// [`Error::External`] when it cannot be read or decoded.
    pub async fn read_overlay(
        &self,
        graph_scope_id: &str,
        observation_revision: &str,
    ) -> Result<Option<Value>> {
        validate_revision(OBSERVATION_REVISION_FIELD, observation_revision)?;
        let path = self.overlay_path(graph_scope_id, observation_revision);
        Ok(read_object(&path).await?.map(Value::Object))
    }

    /// Reads the overlay document `current.json` points at.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for a malformed pointer or revision name, and
    /// [`Error::External`] when a document cannot be read or decoded.
    pub async fn current_overlay(&self, graph_scope_id: &str) -> Result<Option<Value>> {
        let Some(pointer) = self.current_pointer(graph_scope_id).await? else {
            return Ok(None);
        };
        self.read_overlay(graph_scope_id, &pointer.observation_revision)
            .await
    }

    /// Reads one snapshot: `current.json` when `merged_revision` is [`None`], else that
    /// pinned manifest.
    ///
    /// A pinned revision that does not exist is [`None`] (the observation service turns
    /// that into a `NotFound`), while a missing *overlay* below an existing manifest is an
    /// error: the artifact tree is then inconsistent.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an illegal revision name or a document that is not
    /// a JSON object, and [`Error::External`] when a document cannot be read or decoded or a
    /// named overlay is missing.
    pub async fn read_snapshot(
        &self,
        graph_scope_id: &str,
        pinned_revision: Option<&str>,
    ) -> Result<Option<GraphSnapshot>> {
        let Some((pointer_path, pointer)) = self
            .snapshot_pointer(graph_scope_id, pinned_revision)
            .await?
        else {
            return Ok(None);
        };
        let observation_revision = match pointer
            .get(OBSERVATION_REVISION_FIELD)
            .and_then(Value::as_str)
        {
            Some(revision) if !revision.is_empty() => revision.to_string(),
            _ => {
                return Err(validation(
                    OBSERVATION_REVISION_FIELD,
                    format!("is missing from `{}`", pointer_path.display()),
                ))
            }
        };
        let overlay_path = self.overlay_path(graph_scope_id, &observation_revision);
        let Some(overlay) = read_object(&overlay_path).await? else {
            return Err(missing_document_error(&overlay_path));
        };
        let mut merged_revision = text_field(&pointer, MERGED_REVISION_FIELD);
        if merged_revision.is_empty() {
            merged_revision = pinned_revision.unwrap_or_default().to_string();
        }
        Ok(Some(GraphSnapshot {
            graph_scope_id: graph_scope_id.to_string(),
            static_revision: text_field(&pointer, STATIC_REVISION_FIELD),
            observation_revision,
            merged_revision,
            high_water_sequence: pointer
                .get(HIGH_WATER_SEQUENCE_FIELD)
                .and_then(Value::as_u64)
                .unwrap_or_default(),
            overlay: overlay.into_iter().collect(),
        }))
    }

    /// Returns the scope directory names (the SHA-256 scope hashes), sorted.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the `scopes` directory cannot be listed.
    pub async fn scopes(&self) -> Result<Vec<String>> {
        let directory = self.root.join(SCOPES_DIR);
        let mut entries = match fs::read_dir(&directory).await {
            Ok(entries) => entries,
            Err(error) if error.kind() == IoErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(path_error("reading", &directory, error)),
        };
        let mut names = Vec::new();
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|error| path_error("reading", &directory, error))?
        {
            let path = entry.path();
            let is_dir = entry
                .file_type()
                .await
                .map_err(|error| path_error("reading", &path, error))?
                .is_dir();
            if is_dir {
                if let Some(name) = entry.file_name().to_str() {
                    names.push(name.to_string());
                }
            }
        }
        names.sort_unstable();
        Ok(names)
    }

    /// Reads the document a snapshot is described by, with the path it came from.
    async fn snapshot_pointer(
        &self,
        graph_scope_id: &str,
        merged_revision: Option<&str>,
    ) -> Result<Option<(PathBuf, Map<String, Value>)>> {
        let path = match merged_revision {
            Some(revision) => {
                validate_revision(MERGED_REVISION_FIELD, revision)?;
                self.manifest_path(graph_scope_id, revision)
            }
            None => self.scope_root(graph_scope_id).join(CURRENT_FILE),
        };
        Ok(read_object(&path).await?.map(|document| (path, document)))
    }

    /// The overlay document path of one observation revision.
    fn overlay_path(&self, graph_scope_id: &str, observation_revision: &str) -> PathBuf {
        self.scope_root(graph_scope_id)
            .join(OBSERVATIONS_DIR)
            .join(observation_revision)
            .join(OVERLAY_FILE)
    }

    /// The manifest document path of one merged revision.
    fn manifest_path(&self, graph_scope_id: &str, merged_revision: &str) -> PathBuf {
        self.scope_root(graph_scope_id)
            .join(MERGED_DIR)
            .join(merged_revision)
            .join(MANIFEST_FILE)
    }
}

/// Builds the canonical overlay object and its content-derived revision id.
///
/// The canonical object is exactly the hashed document: `schema_version`, the scope, the
/// static revision, the high-water sequence, the id-sorted edges, the projection status and
/// the sorted diagnostics. `generated_at` is added to the stored document only, so
/// re-aggregating identical state reproduces the same id.
fn build_overlay(input: &OverlayInput<'_>) -> (String, Value) {
    let mut document = json!({
        "schema_version": OBSERVATION_SCHEMA,
        "graph_scope_id": input.graph_scope_id,
        "static_revision": input.static_revision,
        "high_water_sequence": input.high_water_sequence,
        "edges": input.edges,
        "projection_status": input.projection_status,
        "diagnostics": sorted_diagnostics(input.diagnostics),
    });
    let digest = short_digest(&stable_hash(&document), REVISION_HASH_LEN);
    let revision = format!("{OBSERVATION_PREFIX}{digest}");
    document[OBSERVATION_REVISION_FIELD] = Value::from(revision.clone());
    document[GENERATED_AT_FIELD] = Value::from(input.generated_at);
    (revision, document)
}

/// Builds the canonical merged manifest and its content-derived revision id.
///
/// The hashed document is `{schema_version, graph_scope_id, static_revision,
/// observation_revision}`; `merged_revision`, `high_water_sequence` and `generated_at` are
/// stored next to it but never hashed, matching Python.
fn build_merged(input: &MergedInput<'_>) -> (String, Value) {
    let mut manifest = json!({
        "schema_version": MERGED_SCHEMA,
        "graph_scope_id": input.graph_scope_id,
        "static_revision": input.static_revision,
        "observation_revision": input.observation_revision,
    });
    let digest = short_digest(&stable_hash(&manifest), REVISION_HASH_LEN);
    let revision = format!("{MERGED_PREFIX}{digest}");
    manifest[MERGED_REVISION_FIELD] = Value::from(revision.clone());
    manifest[HIGH_WATER_SEQUENCE_FIELD] = Value::from(input.high_water_sequence);
    manifest[GENERATED_AT_FIELD] = Value::from(input.generated_at);
    (revision, manifest)
}

/// Returns the first `length` characters of `digest`.
fn short_digest(digest: &str, length: usize) -> String {
    digest.chars().take(length).collect()
}

/// Returns the non-empty diagnostics, sorted and deduplicated.
fn sorted_diagnostics(items: &[String]) -> Vec<String> {
    let mut sorted: Vec<String> = items
        .iter()
        .filter(|item| !item.is_empty())
        .cloned()
        .collect();
    sorted.sort_unstable();
    sorted.dedup();
    sorted
}

/// Returns the string field `field` of `document`, or `""` when it is absent or not a string.
fn text_field(document: &Map<String, Value>, field: &str) -> String {
    document
        .get(field)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// Rejects a revision name that could escape its directory.
///
/// Caller-supplied revisions are only ever used as one path segment, so anything beyond
/// `[A-Za-z0-9._-]` (and the `.`/`..` entries) is a validation error naming the field.
fn validate_revision(field: &str, revision: &str) -> Result<()> {
    let legal = !revision.is_empty()
        && revision != "."
        && revision != ".."
        && revision.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-')
        });
    if legal {
        Ok(())
    } else {
        Err(validation(
            field,
            format!("`{revision}` is not a legal Symphony revision name"),
        ))
    }
}

/// Reads a JSON object document, or [`None`] when the path does not exist.
async fn read_object(path: &Path) -> Result<Option<Map<String, Value>>> {
    let bytes = match fs::read(path).await {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == IoErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(path_error("reading", path, error)),
    };
    let value: Value = serde_json::from_slice(&bytes).map_err(|error| {
        Error::external(error).with_context(format!("decoding `{}`", path.display()))
    })?;
    match value {
        Value::Object(document) => Ok(Some(document)),
        _ => Err(Error::validation(format!(
            "Symphony revision payload must be an object: {}",
            path.display()
        ))),
    }
}

/// Writes an immutable revision document, reusing an identical existing one.
///
/// `generated_at` is metadata, so it is dropped from both sides before the comparison: a
/// republish of the same state succeeds without touching the file, while a *different*
/// document at the same content-derived path is refused.
async fn write_immutable_json(path: &Path, document: &Value) -> Result<()> {
    if let Some(existing) = read_object(path).await? {
        if strip_generated_at(&Value::Object(existing)) != strip_generated_at(document) {
            return Err(Error::validation(format!(
                "immutable Symphony revision already exists with different content: {}",
                path.display()
            )));
        }
        return Ok(());
    }
    write_json_atomic(path, document).await
}

/// Returns `value` without its `generated_at` entry.
///
/// `generated_at` is metadata, so two publications of the same state compare equal even
/// though their timestamps differ.
fn strip_generated_at(value: &Value) -> Value {
    let mut value = value.clone();
    if let Value::Object(document) = &mut value {
        document.remove(GENERATED_AT_FIELD);
    }
    value
}

/// Writes `value` to `path` atomically: a `<name>.tmp` sibling, then a rename.
///
/// A reader therefore observes either the previous document or the complete new one. A
/// failed rename removes the temporary file so a retry starts from a clean directory.
async fn write_json_atomic(path: &Path, value: &Value) -> Result<()> {
    let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
        return Err(Error::config(format!(
            "Symphony revision path `{}` has no file name",
            path.display()
        )));
    };
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .await
            .map_err(|error| path_error("creating", parent, error))?;
    }
    let temporary = path.with_file_name(format!("{file_name}.tmp"));
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| {
        Error::external(error).with_context(format!("encoding `{}`", path.display()))
    })?;
    fs::write(&temporary, &bytes)
        .await
        .map_err(|error| path_error("writing", &temporary, error))?;
    if let Err(error) = fs::rename(&temporary, path).await {
        if let Err(cleanup) = fs::remove_file(&temporary).await {
            debug!(
                path = %temporary.display(),
                error = %cleanup,
                "failed to remove a temporary Symphony revision"
            );
        }
        return Err(path_error("committing", path, error));
    }
    Ok(())
}

/// Wraps an I/O failure as [`Error::External`], naming the action and the path.
fn path_error(action: &str, path: &Path, error: std::io::Error) -> Error {
    Error::external(error).with_context(format!("{action} `{}`", path.display()))
}

/// Builds the error for a pointer that names a document which is not on disk.
fn missing_document_error(path: &Path) -> Error {
    let error = std::io::Error::new(
        IoErrorKind::NotFound,
        "Symphony revision document is missing",
    );
    Error::external(error).with_context(format!("reading `{}`", path.display()))
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

    /// Builds two id-sorted overlay edges.
    fn sample_edges() -> BTreeMap<String, Value> {
        let mut edges = BTreeMap::new();
        edges.insert(
            "skill:a->skill:b".to_string(),
            json!({"runtime_weight": 1.1}),
        );
        edges.insert(
            "skill:b->skill:c".to_string(),
            json!({"runtime_weight": 0.9}),
        );
        edges
    }

    /// Builds an overlay publication request.
    fn overlay_input<'a>(
        scope: &'a str,
        static_revision: &'a str,
        edges: &'a BTreeMap<String, Value>,
        diagnostics: &'a [String],
        generated_at: &'a str,
    ) -> OverlayInput<'a> {
        OverlayInput {
            graph_scope_id: scope,
            static_revision,
            high_water_sequence: 7,
            edges,
            generated_at,
            projection_status: "ready",
            diagnostics,
        }
    }

    /// Reads a document the store wrote.
    fn read_document(path: &Path) -> Value {
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
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

    #[tokio::test]
    async fn overlay_revision_ignores_generated_at_and_tracks_edges() {
        let temp = TempDir::new().unwrap();
        let first_store = RevisionStore::new(temp.path());
        let second_store = RevisionStore::new(temp.path());
        let edges = sample_edges();
        let diagnostics = Vec::new();

        let first = first_store
            .publish_overlay(&overlay_input("scope-a", "v1", &edges, &diagnostics, "t1"))
            .await
            .unwrap();
        let again = second_store
            .publish_overlay(&overlay_input("scope-a", "v1", &edges, &diagnostics, "t2"))
            .await
            .unwrap();
        assert_eq!(first, again);
        assert!(first.starts_with("observation-"));

        let mut changed = edges.clone();
        changed.insert(
            "skill:a->skill:c".to_string(),
            json!({"runtime_weight": 2.0}),
        );
        let other = first_store
            .publish_overlay(&overlay_input(
                "scope-a",
                "v1",
                &changed,
                &diagnostics,
                "t1",
            ))
            .await
            .unwrap();
        assert_ne!(first, other);
    }

    #[tokio::test]
    async fn overlay_document_sorts_edges_and_diagnostics() {
        let temp = TempDir::new().unwrap();
        let store = RevisionStore::new(temp.path());
        let edges = sample_edges();
        let diagnostics = vec![
            "second".to_string(),
            "first".to_string(),
            "second".to_string(),
            String::new(),
        ];
        let mut input = overlay_input("scope-a", "v1", &edges, &diagnostics, "t1");
        input.projection_status = crate::observation::contracts::SAFE_FALLBACK;
        let revision = store.publish_overlay(&input).await.unwrap();

        let overlay = store
            .read_overlay("scope-a", &revision)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(overlay["schema_version"], Value::from(OBSERVATION_SCHEMA));
        assert_eq!(
            overlay["observation_revision"],
            Value::from(revision.clone())
        );
        assert_eq!(
            overlay["projection_status"],
            Value::from(crate::observation::contracts::SAFE_FALLBACK)
        );
        assert_eq!(overlay["diagnostics"], json!(["first", "second"]));
        assert_eq!(overlay["high_water_sequence"], Value::from(7));
        assert_eq!(overlay["edges"], Value::Object(edges.into_iter().collect()));
    }

    #[tokio::test]
    async fn publish_writes_the_three_documents() {
        let temp = TempDir::new().unwrap();
        let store = RevisionStore::new(temp.path());
        let edges = sample_edges();
        let diagnostics = vec!["no_qualified_edge_outcome".to_string()];
        let snapshot = store
            .publish(&overlay_input("scope-a", "v1", &edges, &diagnostics, "t1"))
            .await
            .unwrap();

        let scope_root = store.scope_root("scope-a");
        let current = read_document(&scope_root.join(CURRENT_FILE));
        assert_eq!(
            current["merged_revision"],
            Value::from(snapshot.merged_revision.clone())
        );
        assert_eq!(current["schema_version"], Value::from(MERGED_SCHEMA));
        let overlay = read_document(
            &scope_root
                .join(OBSERVATIONS_DIR)
                .join(&snapshot.observation_revision)
                .join(OVERLAY_FILE),
        );
        assert_eq!(overlay["graph_scope_id"], Value::from("scope-a"));
        let manifest = read_document(
            &scope_root
                .join(MERGED_DIR)
                .join(&snapshot.merged_revision)
                .join(MANIFEST_FILE),
        );
        assert_eq!(
            manifest["observation_revision"],
            Value::from(snapshot.observation_revision.clone())
        );
        assert_eq!(snapshot.high_water_sequence, 7);
        assert_eq!(snapshot.overlay.len(), overlay.as_object().unwrap().len());
        assert_eq!(
            store.scopes().await.unwrap(),
            vec![scope_root.file_name().unwrap().to_str().unwrap()]
        );
    }

    #[tokio::test]
    async fn snapshot_reads_agree_for_current_and_pinned() {
        let temp = TempDir::new().unwrap();
        let store = RevisionStore::new(temp.path());
        let edges = sample_edges();
        let diagnostics = Vec::new();
        let published = store
            .publish(&overlay_input("scope-a", "v1", &edges, &diagnostics, "t1"))
            .await
            .unwrap();

        let current = store.current("scope-a").await.unwrap().unwrap();
        let by_pointer = store.read_snapshot("scope-a", None).await.unwrap().unwrap();
        let pinned = store
            .read_snapshot("scope-a", Some(&published.merged_revision))
            .await
            .unwrap()
            .unwrap();

        assert_eq!(current, published);
        assert_eq!(by_pointer, published);
        assert_eq!(pinned, published);
        let pointer = store.current_pointer("scope-a").await.unwrap().unwrap();
        assert_eq!(pointer.observation_revision, published.observation_revision);
        let overlay = store.current_overlay("scope-a").await.unwrap().unwrap();
        assert_eq!(overlay["edges"], json!(sample_edges()));
        assert_eq!(
            overlay["observation_revision"],
            json!(published.observation_revision)
        );
        assert!(store.current_overlay("scope-b").await.unwrap().is_none());
        assert!(store.current("scope-b").await.unwrap().is_none());
        assert!(store.current_pointer("scope-b").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn unknown_pinned_revision_is_none() {
        let temp = TempDir::new().unwrap();
        let store = RevisionStore::new(temp.path());
        assert!(store
            .read_snapshot("scope-a", Some("merged-00000000000000000000"))
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn republishing_identical_state_is_idempotent() {
        let temp = TempDir::new().unwrap();
        let store = RevisionStore::new(temp.path());
        let edges = sample_edges();
        let diagnostics = Vec::new();

        let first = store
            .publish(&overlay_input("scope-a", "v1", &edges, &diagnostics, "t1"))
            .await
            .unwrap();
        let second = store
            .publish(&overlay_input("scope-a", "v1", &edges, &diagnostics, "t2"))
            .await
            .unwrap();

        assert_eq!(first.observation_revision, second.observation_revision);
        assert_eq!(first.merged_revision, second.merged_revision);
        let pointer = store.current_pointer("scope-a").await.unwrap().unwrap();
        assert_eq!(
            pointer.generated_at, "t2",
            "the pointer is rewritten, the revisions are not"
        );
    }

    #[tokio::test]
    async fn rewriting_a_published_document_is_rejected() {
        let temp = TempDir::new().unwrap();
        let store = RevisionStore::new(temp.path());
        let edges = sample_edges();
        let diagnostics = Vec::new();
        let input = overlay_input("scope-a", "v1", &edges, &diagnostics, "t1");
        let revision = store.publish_overlay(&input).await.unwrap();

        let path = store
            .scope_root("scope-a")
            .join(OBSERVATIONS_DIR)
            .join(&revision)
            .join(OVERLAY_FILE);
        let mut document = read_document(&path);
        document["projection_status"] = Value::from("safe_fallback");
        std::fs::write(&path, serde_json::to_vec_pretty(&document).unwrap()).unwrap();

        let error = store.publish_overlay(&input).await.expect_err("immutable");
        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
        assert!(error.message().contains("immutable Symphony revision"));
        assert!(error.message().contains(&revision));
    }

    #[tokio::test]
    async fn scope_root_is_a_stable_sha_prefix() {
        let temp = TempDir::new().unwrap();
        let store = RevisionStore::new(temp.path());
        let other = RevisionStore::new(temp.path().join("elsewhere"));

        let first = store.scope_root("scope-a");
        let name = first.file_name().unwrap().to_str().unwrap();
        assert_eq!(name.len(), SCOPE_HASH_LEN);
        assert!(name.chars().all(|character| character.is_ascii_hexdigit()));
        assert_eq!(first, store.scope_root("scope-a"));
        assert_eq!(
            name,
            other
                .scope_root("scope-a")
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
        );
        assert_ne!(
            name,
            store
                .scope_root("scope-b")
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
        );
        let expected = stable_hash(&json!({"graph_scope_id": "scope-a"}));
        assert_eq!(name, &expected[..SCOPE_HASH_LEN]);
    }

    #[tokio::test]
    async fn current_pointer_round_trips_and_rejects_non_objects() {
        let temp = TempDir::new().unwrap();
        let store = RevisionStore::new(temp.path());
        let edges = sample_edges();
        let diagnostics = Vec::new();
        let snapshot = store
            .publish(&overlay_input("scope-a", "v1", &edges, &diagnostics, "t1"))
            .await
            .unwrap();

        let pointer = CurrentPointer::from_snapshot(&snapshot, "t9");
        assert_eq!(pointer.schema_version, MERGED_SCHEMA);
        assert_eq!(pointer.merged_revision, snapshot.merged_revision);
        store.write_current(&pointer).await.unwrap();
        assert_eq!(
            store.current_pointer("scope-a").await.unwrap().unwrap(),
            pointer
        );

        let path = store.scope_root("scope-a").join(CURRENT_FILE);
        std::fs::write(&path, b"[]").unwrap();
        let error = store
            .current_pointer("scope-a")
            .await
            .expect_err("a pointer must be an object");
        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
        assert!(error.message().contains("must be an object"));
    }

    #[tokio::test]
    async fn traversal_revision_names_are_rejected() {
        let temp = TempDir::new().unwrap();
        let store = RevisionStore::new(temp.path());
        for revision in ["", ".", "..", "../evil", "with space", "a/b"] {
            let error = store
                .read_overlay("scope-a", revision)
                .await
                .expect_err("illegal revision");
            assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
            assert!(error.message().contains("observation_revision"));
            assert!(store
                .read_snapshot("scope-a", Some(revision))
                .await
                .is_err());
        }
    }
}
