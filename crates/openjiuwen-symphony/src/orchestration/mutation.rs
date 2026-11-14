//! The graph-mutation request contract: normalization, request digests, change-set
//! validation, and the durable idempotency journal.
//!
//! Ports the pure half of Python `openjiuwen/symphony/orchestration/mutations.py`:
//! [`MutationRequest`] plus its normalization (`prepare_mutation_request` /
//! `validate_mutation_batch`), the order-independent request digest
//! (`mutation_request_digest`), the change-set check
//! (`validate_mutation_against_inventory`), and the idempotency journal
//! (`MutationJournal`) that lets a restarted engine replay a batch instead of applying it
//! twice.
//!
//! # Mutation requests
//!
//! A [`MutationRequest`] is the normalized, validated description of one batch: the
//! operation, the caller's idempotency key (`request_id`), the optional graph revision the
//! caller built its batch against (`source_revision`), and the batch itself
//! ([`MutationRequest::changed_skill_ids`], sorted and unique qualified capability ids).
//! [`MutationRequest::prepare`] trims, sorts, and rejects: a blank or oversized request id,
//! a blank or oversized source revision, an empty or oversized batch, a blank id, and
//! duplicate ids after trimming.
//!
//! # Request digests
//!
//! [`request_digest`] is the SHA-256 hex digest of the canonical (sorted-key, compact)
//! JSON of exactly Python's field set:
//!
//! ```json
//! {"schema_version": "Symphony-skill-graph-mutation-request-v2",
//!  "request_id": "...", "operation": "add" | "update" | "delete",
//!  "source_revision": "..." | null, "changed_skill_ids": ["...", "..."]}
//! ```
//!
//! The ids are sorted before hashing, so two batches that differ only in id order share a
//! digest — reordering never invalidates an idempotency key — while every other field
//! change does. The digest *is* the replay identity: a journal hit whose recorded digest
//! differs from the incoming one is rejected instead of answered with a stale result.
//!
//! # The idempotency journal
//!
//! [`MutationJournal`] owns one JSON document:
//!
//! ```json
//! {"schema_version": "Symphony-skill-graph-mutation-requests-v1",
//!  "records": {"<request_id>": {"digest": "...", "operation": "add",
//!                               "revision": "v3", "inserted_at": "...",
//!                               "result": {"...": "..."}}}}
//! ```
//!
//! Records live in a [`BTreeMap`], so the file is byte-identical for the same records
//! regardless of insertion order, and every write goes through a temporary sibling plus a
//! rename. `inserted_at` and the optional `result` are metadata: the first never
//! participates in a lookup or a digest, and the second is an additive Rust extension
//! (Python stores the full result inside the graph artifact) that lets a reopened runtime
//! answer a replay with the exact recorded [`GraphMutationResult`](crate::runtime::GraphMutationResult).
//!
//! # Errors
//!
//! Every rejected request is [`Error::Validation`] naming
//! the offending field (`request_id`, `source_revision`, `changed_skill_ids`, or
//! `mutation_journal_path`). Validation never degrades into a silent no-op.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use openjiuwen_common::utils::id::now_rfc3339;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::error::{validation, Error, Result};
use crate::llm::identity::stable_metadata_sha256;

/// The `schema_version` covered by a [`request_digest`].
pub const REQUEST_SCHEMA_VERSION: &str = "Symphony-skill-graph-mutation-request-v2";

/// The `schema_version` written to (and required from) a [`MutationJournal`] file.
pub const JOURNAL_SCHEMA_VERSION: &str = "Symphony-skill-graph-mutation-requests-v1";

/// The longest accepted [`MutationRequest::request_id`], in characters.
pub const MAX_REQUEST_ID_CHARS: usize = 256;

/// The longest accepted [`MutationRequest::source_revision`], in characters.
pub const MAX_SOURCE_REVISION_CHARS: usize = 512;

/// The largest accepted batch.
///
/// Python bounds only the two strings above and the batch's non-emptiness; this count cap
/// is a Rust-only guard so one call cannot rewrite an unbounded catalog slice.
pub const MAX_CHANGED_IDS: usize = 1024;

/// The kind of change a capability mutation applies.
///
/// The JSON names (`add`, `update`, `delete`) are Python's `MutationOperation` literals and
/// are part of both the request digest and the journal record.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum MutationOperation {
    /// Descriptors are inserted or replaced.
    #[default]
    Add,
    /// Already-registered descriptors are replaced.
    Update,
    /// Keys are deleted.
    Delete,
}

impl MutationOperation {
    /// The lowercase name used in digests, journal records, and error messages.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Update => "update",
            Self::Delete => "delete",
        }
    }
}

impl std::fmt::Display for MutationOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// One normalized, validated graph-mutation batch.
///
/// Build one with [`MutationRequest::prepare`], which enforces the contract; the fields are
/// public so a deserialized request can be inspected, and [`request_digest`] normalizes the
/// id order again before hashing.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct MutationRequest {
    /// The caller's idempotency key, trimmed; never blank.
    pub request_id: String,
    /// What the batch does.
    pub operation: MutationOperation,
    /// The graph revision the caller built the batch against, when it declared one.
    ///
    /// Python requires this field; here it is optional so the three positional Rust
    /// mutation methods keep working. When it is set, the runtime compares it with the
    /// current artifact revision before publishing (a compare-and-swap).
    pub source_revision: Option<String>,
    /// The batch: sorted, de-duplicated qualified capability ids (`type:id`).
    pub changed_skill_ids: Vec<String>,
}

impl MutationRequest {
    /// Normalizes and validates one mutation batch.
    ///
    /// The ids are trimmed, rejected when blank, rejected when duplicated after trimming,
    /// and returned sorted, so the digest is order-independent. `source_revision` is
    /// optional; when supplied it is trimmed and must hold 1..=[`MAX_SOURCE_REVISION_CHARS`]
    /// characters.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] naming `request_id`, `source_revision`, or
    /// `changed_skill_ids` for a blank or oversized request id or source revision, an empty
    /// batch (including a `delete` without ids), an oversized batch, a blank id, or
    /// duplicate ids.
    pub fn prepare(
        operation: MutationOperation,
        changed_skill_ids: &[String],
        request_id: &str,
        source_revision: Option<&str>,
    ) -> Result<Self> {
        let request_id = request_id.trim();
        if request_id.is_empty() || request_id.chars().count() > MAX_REQUEST_ID_CHARS {
            return Err(validation(
                "request_id",
                format!("must contain between 1 and {MAX_REQUEST_ID_CHARS} characters"),
            ));
        }
        let source_revision = match source_revision {
            None => None,
            Some(raw) => {
                let trimmed = raw.trim();
                if trimmed.is_empty() || trimmed.chars().count() > MAX_SOURCE_REVISION_CHARS {
                    return Err(validation(
                        "source_revision",
                        format!(
                            "must contain between 1 and {MAX_SOURCE_REVISION_CHARS} characters"
                        ),
                    ));
                }
                Some(trimmed.to_string())
            }
        };
        if changed_skill_ids.is_empty() {
            return Err(validation(
                "changed_skill_ids",
                format!("a {operation} batch must not be empty"),
            ));
        }
        if changed_skill_ids.len() > MAX_CHANGED_IDS {
            return Err(validation(
                "changed_skill_ids",
                format!(
                    "a {operation} batch must not hold more than {MAX_CHANGED_IDS} capability ids"
                ),
            ));
        }
        let mut ids = Vec::with_capacity(changed_skill_ids.len());
        for raw in changed_skill_ids {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                return Err(validation(
                    "changed_skill_ids",
                    "ids must be non-empty strings",
                ));
            }
            ids.push(trimmed.to_string());
        }
        ids.sort();
        let duplicates: Vec<&str> = ids
            .windows(2)
            .filter(|pair| pair[0] == pair[1])
            .map(|pair| pair[0].as_str())
            .collect();
        if !duplicates.is_empty() {
            return Err(validation(
                "changed_skill_ids",
                format!(
                    "a mutation batch must not contain duplicate capability ids ({})",
                    duplicates.join(", ")
                ),
            ));
        }
        Ok(Self {
            request_id: request_id.to_string(),
            operation,
            source_revision,
            changed_skill_ids: ids,
        })
    }

    /// The replay identity of this request; see [`request_digest`].
    #[must_use]
    pub fn digest(&self) -> String {
        request_digest(self)
    }

    /// The declared batch as a sorted, de-duplicated id set.
    #[must_use]
    pub fn declared_ids(&self) -> std::collections::BTreeSet<&str> {
        self.changed_skill_ids
            .iter()
            .map(String::as_str)
            .filter(|id| !id.trim().is_empty())
            .collect()
    }
}

/// Returns the order-independent digest of `request`.
///
/// The digest is the SHA-256 hex of the canonical (sorted-key, compact) JSON of
/// [`REQUEST_SCHEMA_VERSION`], `request_id`, `operation`, `source_revision` (JSON `null`
/// when undeclared), and the sorted `changed_skill_ids`. Every field is covered, so a
/// changed operation, revision, or id set is a different request, while reordering the ids
/// is not.
#[must_use]
pub fn request_digest(request: &MutationRequest) -> String {
    stable_metadata_sha256(&digest_document(request))
}

/// Builds the JSON document [`request_digest`] hashes.
fn digest_document(request: &MutationRequest) -> Value {
    let mut ids: Vec<&str> = request
        .changed_skill_ids
        .iter()
        .map(String::as_str)
        .collect();
    ids.sort_unstable();
    let mut document = Map::new();
    document.insert(
        "schema_version".to_string(),
        Value::String(REQUEST_SCHEMA_VERSION.to_string()),
    );
    document.insert(
        "request_id".to_string(),
        Value::String(request.request_id.clone()),
    );
    document.insert(
        "operation".to_string(),
        Value::String(request.operation.as_str().to_string()),
    );
    document.insert(
        "source_revision".to_string(),
        request
            .source_revision
            .as_ref()
            .map_or(Value::Null, |revision| Value::String(revision.clone())),
    );
    document.insert(
        "changed_skill_ids".to_string(),
        Value::Array(
            ids.into_iter()
                .map(str::to_string)
                .map(Value::String)
                .collect(),
        ),
    );
    Value::Object(document)
}

/// Requires the applied change set to match what `request` declared.
///
/// This is the Rust mapping of Python's `validate_mutation_against_inventory`: the caller
/// supplies the ids the mutation actually touched (`changed`, normally
/// [`GraphMutationResult::changed`](crate::runtime::GraphMutationResult::changed)) and the
/// inventory the mutation left behind (`inventory_ids`, the catalog's qualified ids after
/// the batch). The check is:
///
/// - the declared batch must equal `changed` exactly — an id that was declared but not
///   changed, and an id that changed without being declared, are both rejected;
/// - `add` and `update` require every declared id to be present in `inventory_ids`;
/// - `delete` requires every declared id to be absent from `inventory_ids`.
///
/// An empty `changed` list is therefore only valid for an empty declared batch, which
/// [`MutationRequest::prepare`] already rejects.
///
/// # Errors
///
/// Returns [`Error::Validation`] naming `changed_skill_ids` and the offending ids.
pub fn validate_mutation_against_inventory(
    request: &MutationRequest,
    changed: &[String],
    inventory_ids: &[String],
) -> Result<()> {
    let declared = request.declared_ids();
    let actual: std::collections::BTreeSet<&str> = changed
        .iter()
        .map(String::as_str)
        .filter(|id| !id.trim().is_empty())
        .collect();
    if declared != actual {
        let missing: Vec<&str> = declared.difference(&actual).copied().collect();
        let extra: Vec<&str> = actual.difference(&declared).copied().collect();
        return Err(validation(
            "changed_skill_ids",
            format!(
                "the applied change set does not match the request: declared but not changed [{}], changed but not declared [{}]",
                missing.join(", "),
                extra.join(", ")
            ),
        ));
    }
    let inventory: std::collections::BTreeSet<&str> =
        inventory_ids.iter().map(String::as_str).collect();
    match request.operation {
        MutationOperation::Add | MutationOperation::Update => {
            if let Some(absent) = declared.iter().find(|id| !inventory.contains(*id)) {
                return Err(validation(
                    "changed_skill_ids",
                    format!("`{absent}` is not present in the capability inventory"),
                ));
            }
        }
        MutationOperation::Delete => {
            if let Some(present) = declared.iter().find(|id| inventory.contains(*id)) {
                return Err(validation(
                    "changed_skill_ids",
                    format!("`{present}` is still present in the capability inventory"),
                ));
            }
        }
    }
    Ok(())
}

/// One recorded idempotency entry.
///
/// `digest` is the [`request_digest`] of the request that produced `revision`, so a reuse
/// of the same `request_id` for a different batch is detectable. `inserted_at` is metadata
/// only and never participates in a lookup. `result` is the recorded public result as JSON
/// (a Rust-only extension: it lets a reopened runtime answer a replay with the exact
/// recorded result instead of a reconstruction).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct JournalRecord {
    /// The request digest this record answers.
    pub digest: String,
    /// The operation the recorded request performed.
    pub operation: MutationOperation,
    /// The graph revision the recorded request published.
    pub revision: String,
    /// When the record was written (RFC 3339); metadata only.
    pub inserted_at: String,
    /// The recorded public result, serialized; omitted when the caller did not keep one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
}

impl JournalRecord {
    /// Builds a record for `operation`/`revision` under `digest`, stamped with the current
    /// time.
    #[must_use]
    pub fn new(
        operation: MutationOperation,
        revision: impl Into<String>,
        digest: impl Into<String>,
    ) -> Self {
        Self {
            digest: digest.into(),
            operation,
            revision: revision.into(),
            inserted_at: now_rfc3339(),
            result: None,
        }
    }

    /// Replaces the recorded timestamp (a caller-supplied value).
    #[must_use]
    pub fn with_inserted_at(mut self, inserted_at: impl Into<String>) -> Self {
        self.inserted_at = inserted_at.into();
        self
    }

    /// Attaches the recorded public result.
    #[must_use]
    pub fn with_result(mut self, result: Value) -> Self {
        self.result = Some(result);
        self
    }
}

/// A durable, JSON-file-backed idempotency journal keyed by caller `request_id`.
///
/// The journal is authoritative in memory after [`MutationJournal::open`]: lookups never
/// touch the filesystem, and every mutation writes the whole record map through
/// atomically. The struct is not synchronized; the runtime owns it behind its mutation
/// lock.
#[derive(Debug, Clone)]
pub struct MutationJournal {
    /// The JSON file the records live in.
    path: PathBuf,
    /// The records by request id, ordered so the file is byte-stable.
    records: BTreeMap<String, JournalRecord>,
}

impl MutationJournal {
    /// Builds an empty journal over `path` without touching the filesystem.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            records: BTreeMap::new(),
        }
    }

    /// Opens the journal at `path`, loading the records already there.
    ///
    /// An absent or empty file opens as an empty journal. Malformed JSON, an envelope that
    /// is not an object, and a missing or unexpected `schema_version` are
    /// [`Error::Validation`] naming the file: a journal is never silently reset. A
    /// malformed *record* is warned about and skipped, so one bad entry cannot cost the
    /// rest of the journal its idempotency.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the file cannot be read and
    /// [`Error::Validation`] when its contents are not a journal this crate can honour.
    pub async fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let text = match tokio::fs::read_to_string(&path).await {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Self::new(path));
            }
            Err(error) => {
                return Err(Error::external(std::io::Error::new(
                    error.kind(),
                    format!("mutation journal read {}: {error}", path.display()),
                )));
            }
        };
        if text.trim().is_empty() {
            return Ok(Self::new(path));
        }
        let payload: Value = serde_json::from_str(&text).map_err(|error| {
            validation(
                "mutation_journal_path",
                format!("{} is not valid JSON: {error}", path.display()),
            )
        })?;
        let Some(object) = payload.as_object() else {
            return Err(validation(
                "mutation_journal_path",
                format!("{} does not hold a JSON object", path.display()),
            ));
        };
        match object.get("schema_version").and_then(Value::as_str) {
            Some(JOURNAL_SCHEMA_VERSION) => {}
            Some(other) => {
                return Err(validation(
                    "mutation_journal_path",
                    format!(
                        "{} uses schema_version {other:?}, expected {JOURNAL_SCHEMA_VERSION:?}",
                        path.display()
                    ),
                ));
            }
            None => {
                return Err(validation(
                    "mutation_journal_path",
                    format!("{} carries no schema_version", path.display()),
                ));
            }
        }
        let mut records = BTreeMap::new();
        match object.get("records") {
            None | Some(Value::Null) => {}
            Some(Value::Object(entries)) => {
                for (key, entry) in entries {
                    match serde_json::from_value::<JournalRecord>(entry.clone()) {
                        Ok(record) => {
                            records.insert(key.clone(), record);
                        }
                        Err(error) => {
                            tracing::warn!(
                                request_id = key,
                                filename = %path.display(),
                                %error,
                                "mutation journal: skipping a malformed record"
                            );
                        }
                    }
                }
            }
            Some(_) => {
                return Err(validation(
                    "mutation_journal_path",
                    format!("{} holds a non-object `records`", path.display()),
                ));
            }
        }
        Ok(Self { path, records })
    }

    /// The file this journal is backed by.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The number of recorded requests.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the journal holds no records.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Whether `request_id` is recorded.
    #[must_use]
    pub fn contains(&self, request_id: &str) -> bool {
        self.records.contains_key(request_id)
    }

    /// The recorded request ids, in file order.
    pub fn request_ids(&self) -> impl Iterator<Item = &str> {
        self.records.keys().map(String::as_str)
    }

    /// Looks `request_id` up, answering a clone of the recorded entry.
    ///
    /// This is an in-memory read: it performs no I/O and cannot fail. The `async`
    /// signature keeps callers uniform with the write path, which does touch the
    /// filesystem.
    // Kept async so a lookup and a recording compose alike, even though this read is
    // purely in memory.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn lookup(&self, request_id: &str) -> Option<JournalRecord> {
        self.records.get(request_id).cloned()
    }

    /// Records `record` under `request_id` and writes the journal file through
    /// immediately.
    ///
    /// Existing records are preserved and the file is rewritten in full (the map is a
    /// [`BTreeMap`], so records are sorted by request id and the layout is stable). The
    /// write goes through a temporary sibling plus a rename, so a reader never observes a
    /// partial file.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for a blank `request_id`, and [`Error::External`] when
    /// the file cannot be written.
    pub async fn record(
        &mut self,
        request_id: impl Into<String>,
        record: JournalRecord,
    ) -> Result<()> {
        let request_id = request_id.into();
        if request_id.trim().is_empty() {
            return Err(validation(
                "request_id",
                "must be non-blank to be recorded in the mutation journal",
            ));
        }
        self.records.insert(request_id, record);
        self.persist().await
    }

    /// Drops every record and writes the empty journal through.
    ///
    /// The file is kept (holding an empty `records` object) rather than deleted.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the file cannot be written.
    pub async fn clear(&mut self) -> Result<()> {
        self.records.clear();
        self.persist().await
    }

    /// Serializes the records and publishes them atomically.
    async fn persist(&self) -> Result<()> {
        let mut envelope = Map::new();
        envelope.insert(
            "schema_version".to_string(),
            Value::String(JOURNAL_SCHEMA_VERSION.to_string()),
        );
        envelope.insert(
            "records".to_string(),
            serde_json::to_value(&self.records).map_err(|error| {
                Error::external(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("mutation journal serialize: {error}"),
                ))
            })?,
        );
        let text = serde_json::to_string_pretty(&Value::Object(envelope)).map_err(|error| {
            Error::external(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("mutation journal serialize: {error}"),
            ))
        })?;
        crate::flow::store::atomic_write(&self.path, format!("{text}\n").as_bytes()).await
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// A prepared `add` request over two ids.
    fn add_request() -> MutationRequest {
        MutationRequest::prepare(
            MutationOperation::Add,
            &["tool:fetch".to_string(), "tool:sum".to_string()],
            "req-1",
            None,
        )
        .expect("prepare")
    }

    #[test]
    fn prepare_sorts_and_trims_ids() {
        let request = MutationRequest::prepare(
            MutationOperation::Update,
            &["  tool:sum ".to_string(), "tool:fetch".to_string()],
            "  req-1  ",
            Some("  v2 "),
        )
        .expect("prepare");
        assert_eq!(request.request_id, "req-1");
        assert_eq!(request.changed_skill_ids, vec!["tool:fetch", "tool:sum"]);
        assert_eq!(request.source_revision.as_deref(), Some("v2"));
    }

    #[test]
    fn prepare_rejects_blank_and_duplicate_input() {
        let blank = MutationRequest::prepare(MutationOperation::Add, &[], "  ", None)
            .expect_err("a blank request id must be rejected");
        assert!(blank.message().contains("request_id"));

        let empty = MutationRequest::prepare(MutationOperation::Delete, &[], "req", None)
            .expect_err("an empty batch must be rejected");
        assert!(empty.message().contains("changed_skill_ids"));

        let duplicate = MutationRequest::prepare(
            MutationOperation::Add,
            &["tool:fetch".to_string(), " tool:fetch".to_string()],
            "req",
            None,
        )
        .expect_err("duplicate ids must be rejected");
        assert!(duplicate.message().contains("tool:fetch"));
    }

    #[test]
    fn digest_ignores_id_order_but_covers_every_field() {
        let request = add_request();
        let mut reordered = request.clone();
        reordered.changed_skill_ids.reverse();
        assert_eq!(request.digest(), reordered.digest());
        assert_eq!(request.digest().len(), 64);
        assert!(request
            .digest()
            .chars()
            .all(|character| character.is_ascii_hexdigit() && !character.is_ascii_uppercase()));

        let mut other_id = request.clone();
        other_id.request_id = "req-2".to_string();
        let mut other_operation = request.clone();
        other_operation.operation = MutationOperation::Update;
        let mut other_revision = request.clone();
        other_revision.source_revision = Some("v3".to_string());
        let mut other_ids = request.clone();
        other_ids.changed_skill_ids = vec!["tool:fetch".to_string()];
        let digests = [
            request.digest(),
            other_id.digest(),
            other_operation.digest(),
            other_revision.digest(),
            other_ids.digest(),
        ];
        for (index, left) in digests.iter().enumerate() {
            assert!(
                digests[index + 1..].iter().all(|right| left != right),
                "digests must differ per field"
            );
        }
    }

    #[test]
    fn validate_accepts_an_exact_change_set_and_names_mismatches() {
        let request = add_request();
        let inventory = vec!["tool:fetch".to_string(), "tool:sum".to_string()];
        validate_mutation_against_inventory(
            &request,
            &["tool:sum".to_string(), "tool:fetch".to_string()],
            &inventory,
        )
        .expect("an exact change set must validate");

        let missing =
            validate_mutation_against_inventory(&request, &["tool:fetch".to_string()], &inventory)
                .expect_err("a missing id must be rejected");
        assert!(missing.message().contains("tool:sum"));

        let extra = validate_mutation_against_inventory(
            &request,
            &[
                "tool:fetch".to_string(),
                "tool:sum".to_string(),
                "tool:extra".to_string(),
            ],
            &inventory,
        )
        .expect_err("an undeclared id must be rejected");
        assert!(extra.message().contains("tool:extra"));

        let absent = validate_mutation_against_inventory(
            &request,
            &["tool:fetch".to_string(), "tool:sum".to_string()],
            &["tool:fetch".to_string()],
        )
        .expect_err("an added id must exist in the inventory");
        assert!(absent.message().contains("tool:sum"));
    }

    #[test]
    fn delete_requires_an_absent_inventory_and_a_matching_change_set() {
        let request = MutationRequest::prepare(
            MutationOperation::Delete,
            &["tool:fetch".to_string()],
            "req-del",
            Some("v1"),
        )
        .expect("prepare");
        validate_mutation_against_inventory(&request, &["tool:fetch".to_string()], &[])
            .expect("a deleted id must be absent from the inventory");
        let present = validate_mutation_against_inventory(
            &request,
            &["tool:fetch".to_string()],
            &["tool:fetch".to_string()],
        )
        .expect_err("a deleted id must not remain in the inventory");
        assert!(present.message().contains("tool:fetch"));
    }
}
