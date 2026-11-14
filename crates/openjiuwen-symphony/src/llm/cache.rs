//! The persistent relation-match cache.
//!
//! Python's `RelationMatchCache` memoizes ontology-match answers in a JSON file keyed by
//! the candidate's evidence, the two endpoint fingerprints, and the matcher identity, so
//! a re-run of the graph builder only asks the model about candidates it has never seen.
//! This module ports that cache onto a single owned file:
//!
//! ```json
//! {"schema_version": "Symphony-relation-match-cache-v1",
//!  "records": {"<sha256 key>": {"candidate_id": "...", "matches": [...],
//!                               "diagnostics": [...], "updated_at": "..."}}}
//! ```
//!
//! # Rules
//!
//! - [`cache_key`] is the SHA-256 of the canonical JSON of the *cache-key document*
//!   (see its documentation), so a changed candidate, fingerprint, or matcher signature
//!   never reuses a stale answer.
//! - Records live in a `BTreeMap`, so a file written twice from the same records is
//!   byte-identical regardless of insertion order.
//! - Every [`MatchCache::record`] and [`MatchCache::clear`] writes the whole file
//!   through immediately, via a temporary sibling and a rename, so a reader never sees
//!   a partial file.
//! - `updated_at` is metadata only; it never participates in a lookup or a key. Use
//!   [`CachedMatch::new`] for the current time or [`CachedMatch::with_updated_at`] for a
//!   caller-supplied value.
//! - There is no TTL, no eviction, and no size bound: the cache is a memo, not a policy.
//!   [`MatchCache::clear`] is the only way to forget entries.
//!
//! # Failures
//!
//! An absent or empty file opens as an empty cache. Malformed JSON, a non-object
//! envelope, or an unexpected `schema_version` is
//! [`Error::Validation`](crate::error::Error::Validation) naming the file —
//! a cache is never silently reset. A single malformed record is warned about and
//! skipped, so one bad entry cannot cost the rest of the catalog its answers.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::error::{validation, Error, Result};
use crate::graph::CAN_FEED;
use crate::llm::identity::{canonical_json, stable_metadata_sha256};
use crate::llm::ontology::OntologyMatch;
use crate::orchestration::RelationCandidate;

use openjiuwen_common::utils::id::now_rfc3339;

/// The `schema_version` written to (and required from) a cache file.
pub const CACHE_SCHEMA_VERSION: &str = "Symphony-relation-match-cache-v1";

/// One cached ontology-match answer.
///
/// `matches` and `diagnostics` hold the stage's JSON payloads verbatim
/// (`matches` is the serialized match list, `diagnostics` the accompanying records).
/// `updated_at` is informational and never affects a lookup.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct CachedMatch {
    /// The qualified candidate identity the answer belongs to.
    pub candidate_id: String,
    /// The serialized match list.
    pub matches: Value,
    /// The serialized diagnostics accompanying the matches.
    pub diagnostics: Value,
    /// When the entry was recorded (RFC 3339); metadata only.
    pub updated_at: String,
}

impl Default for CachedMatch {
    fn default() -> Self {
        Self {
            candidate_id: String::new(),
            matches: Value::Array(Vec::new()),
            diagnostics: Value::Array(Vec::new()),
            updated_at: String::new(),
        }
    }
}

impl CachedMatch {
    /// Builds an entry for `candidate_id` holding `matches`, stamped with the current
    /// time.
    ///
    /// The timestamp is metadata: it never participates in a key or a lookup, so tests
    /// can assert on entry contents without freezing the clock.
    #[must_use]
    pub fn new(candidate_id: impl Into<String>, matches: Value) -> Self {
        Self {
            candidate_id: candidate_id.into(),
            matches,
            diagnostics: Value::Array(Vec::new()),
            updated_at: now_rfc3339(),
        }
    }

    /// Replaces the diagnostics payload.
    #[must_use]
    pub fn with_diagnostics(mut self, diagnostics: Value) -> Self {
        self.diagnostics = diagnostics;
        self
    }

    /// Replaces the recorded timestamp (a caller-supplied string).
    #[must_use]
    pub fn with_updated_at(mut self, updated_at: impl Into<String>) -> Self {
        self.updated_at = updated_at.into();
        self
    }
}

/// A JSON-file-backed relation-match cache.
///
/// The cache is authoritative in memory after [`MatchCache::open`]: lookups never touch
/// the filesystem, and every mutation writes the whole record map through atomically.
/// Wrap it in a [`tokio::sync::Mutex`] to share it between matchers;
/// the struct itself is not synchronized.
#[derive(Debug, Clone)]
pub struct MatchCache {
    /// The JSON file the records live in.
    path: PathBuf,
    /// The records by cache key, ordered so the file is byte-stable.
    records: BTreeMap<String, CachedMatch>,
}

impl MatchCache {
    /// Builds an empty cache over `path` without touching the filesystem.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            records: BTreeMap::new(),
        }
    }

    /// Opens the cache at `path`, loading any records already there.
    ///
    /// An absent or empty file opens as an empty cache. Malformed JSON, an envelope that
    /// is not an object, or a `schema_version` other than [`CACHE_SCHEMA_VERSION`] is
    /// [`Error::Validation`] naming the file: a cache is never silently reset. A
    /// malformed *record* is warned about and skipped.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the file cannot be read and
    /// [`Error::Validation`] when its contents are not a cache this crate can honour.
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
                    format!("match cache read {}: {error}", path.display()),
                )));
            }
        };
        if text.trim().is_empty() {
            return Ok(Self::new(path));
        }
        let payload: Value = serde_json::from_str(&text).map_err(|error| {
            validation(
                "match_cache_path",
                format!("{} is not valid JSON: {error}", path.display()),
            )
        })?;
        let Some(object) = payload.as_object() else {
            return Err(validation(
                "match_cache_path",
                format!("{} does not hold a JSON object", path.display()),
            ));
        };
        match object.get("schema_version").and_then(Value::as_str) {
            Some(CACHE_SCHEMA_VERSION) => {}
            Some(other) => {
                return Err(validation(
                    "match_cache_path",
                    format!(
                        "{} uses schema_version {other:?}, expected {CACHE_SCHEMA_VERSION:?}",
                        path.display()
                    ),
                ));
            }
            None => {
                return Err(validation(
                    "match_cache_path",
                    format!("{} carries no schema_version", path.display()),
                ));
            }
        }
        let mut records = BTreeMap::new();
        match object.get("records") {
            None | Some(Value::Null) => {}
            Some(Value::Object(entries)) => {
                for (key, entry) in entries {
                    match serde_json::from_value::<CachedMatch>(entry.clone()) {
                        Ok(record) => {
                            records.insert(key.clone(), record);
                        }
                        Err(error) => {
                            tracing::warn!(
                                key,
                                filename = %path.display(),
                                %error,
                                "match cache: skipping a malformed record"
                            );
                        }
                    }
                }
            }
            Some(_) => {
                return Err(validation(
                    "match_cache_path",
                    format!("{} holds a non-object `records`", path.display()),
                ));
            }
        }
        Ok(Self { path, records })
    }

    /// The file this cache is backed by.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The number of cached records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the cache holds no records.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Whether `key` is cached.
    #[must_use]
    pub fn contains(&self, key: &str) -> bool {
        self.records.contains_key(key)
    }

    /// The cache keys currently held, in file order.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.records.keys().map(String::as_str)
    }

    /// Looks `key` up, answering a clone of the recorded entry.
    ///
    /// This is an in-memory read: it performs no I/O and cannot fail. The `async`
    /// signature keeps callers uniform with the write path (which does touch the
    /// filesystem).
    // Kept async so a lookup and a recording compose alike, even though this read is
    // purely in memory.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn lookup(&self, key: &str) -> Option<CachedMatch> {
        self.records.get(key).cloned()
    }

    /// Records `entry` under `key` and writes the cache file through immediately.
    ///
    /// Existing records are preserved and the file is rewritten in full (the map is a
    /// `BTreeMap`, so the layout is stable). On a write failure the in-memory entry is
    /// kept — the next successful write will persist it — and the error is returned so
    /// the caller can log and carry on with the model answer.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the file cannot be written.
    pub async fn record(&mut self, key: String, entry: CachedMatch) -> Result<()> {
        self.records.insert(key, entry);
        self.persist().await
    }

    /// Drops every record and writes the empty cache through.
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
            Value::String(CACHE_SCHEMA_VERSION.to_string()),
        );
        envelope.insert(
            "records".to_string(),
            serde_json::to_value(&self.records).map_err(|error| {
                Error::external(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("match cache serialize: {error}"),
                ))
            })?,
        );
        let text = serde_json::to_string_pretty(&Value::Object(envelope)).map_err(|error| {
            Error::external(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("match cache serialize: {error}"),
            ))
        })?;
        crate::flow::store::atomic_write(&self.path, format!("{text}\n").as_bytes()).await
    }
}

/// Returns the cache key for one candidate.
///
/// The key is the SHA-256 hex digest of the canonical JSON of this document:
///
/// ```json
/// {"candidate": <the candidate's JSON>,
///  "source_fingerprint_hash": "<graph identity hash of the source>",
///  "target_fingerprint_hash": "<graph identity hash of the target>",
///  "matcher_signature": "<the matcher's identity document>"}
/// ```
///
/// Object-key order never matters (the document is canonicalized first), while changing
/// the candidate, either fingerprint hash, or any part of the signature produces a
/// different key. The signature is compared as a whole, so a caller that embeds a model
/// identity in it invalidates every entry when the model changes.
#[must_use]
pub fn cache_key(
    candidate: &Value,
    source_fingerprint_hash: &str,
    target_fingerprint_hash: &str,
    signature: &str,
) -> String {
    stable_metadata_sha256(&key_document(
        candidate,
        source_fingerprint_hash,
        target_fingerprint_hash,
        signature,
    ))
}

/// Renders the cache-key document for `candidate` (without hashing it).
///
/// Exposed for tests and diagnostics: it is exactly what [`cache_key`] hashes.
#[must_use]
pub fn cache_key_document(
    candidate: &Value,
    source_fingerprint_hash: &str,
    target_fingerprint_hash: &str,
    signature: &str,
) -> String {
    canonical_json(&key_document(
        candidate,
        source_fingerprint_hash,
        target_fingerprint_hash,
        signature,
    ))
}

/// The documented candidate identity recorded with a cached answer
/// (Python's `RelationCandidate.key`: `source->target:relation`).
pub(crate) fn candidate_identity(candidate: &RelationCandidate) -> String {
    let relation = if candidate.relation.trim().is_empty() {
        CAN_FEED
    } else {
        candidate.relation.as_str()
    };
    format!("{}->{}:{relation}", candidate.source, candidate.target)
}

/// Reports whether `item` answers the `candidate` pair, in either orientation.
pub(crate) fn covers(item: &OntologyMatch, candidate: &RelationCandidate) -> bool {
    let forward = item.source_id == candidate.source && item.target_id == candidate.target;
    let reverse = item.source_id == candidate.target && item.target_id == candidate.source;
    forward || reverse
}

/// Decodes the match list of a cached record, answering `None` when it is unusable.
pub(crate) fn decode_cached_matches(record: &CachedMatch) -> Option<Vec<OntologyMatch>> {
    match serde_json::from_value::<Vec<OntologyMatch>>(record.matches.clone()) {
        Ok(matches) => Some(matches),
        Err(error) => {
            tracing::warn!(%error, "ontology matcher: cached matches could not be decoded");
            None
        }
    }
}

/// Builds the cache-key document as a JSON value.
fn key_document(
    candidate: &Value,
    source_fingerprint_hash: &str,
    target_fingerprint_hash: &str,
    signature: &str,
) -> Value {
    let mut document = Map::new();
    document.insert("candidate".to_string(), candidate.clone());
    document.insert(
        "source_fingerprint_hash".to_string(),
        Value::String(source_fingerprint_hash.to_string()),
    );
    document.insert(
        "target_fingerprint_hash".to_string(),
        Value::String(target_fingerprint_hash.to_string()),
    );
    document.insert(
        "matcher_signature".to_string(),
        Value::String(signature.to_string()),
    );
    Value::Object(document)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample(key_label: &str) -> CachedMatch {
        CachedMatch::new(
            "tool:a->tool:b",
            json!([{"source_id": "tool:a", "target_id": "tool:b", "confidence": 0.9,
                    "reason": key_label}]),
        )
        .with_diagnostics(json!([]))
        .with_updated_at("2024-01-02T03:04:05.000Z")
    }

    #[test]
    fn cache_key_is_order_independent_and_field_sensitive() {
        let candidate = json!({"source": "tool:a", "target": "tool:b", "confidence": 0.75});
        let reordered = json!({"confidence": 0.75, "target": "tool:b", "source": "tool:a"});
        let base = cache_key(&candidate, "src", "dst", "sig");
        assert_eq!(base.len(), 64);
        assert_eq!(cache_key(&reordered, "src", "dst", "sig"), base);
        assert_ne!(cache_key(&candidate, "src2", "dst", "sig"), base);
        assert_ne!(cache_key(&candidate, "src", "dst2", "sig"), base);
        assert_ne!(cache_key(&candidate, "src", "dst", "sig2"), base);
        assert_ne!(
            cache_key(&json!({"source": "tool:a"}), "src", "dst", "sig"),
            base
        );
    }

    #[test]
    fn cache_key_document_is_canonical_json() {
        let candidate = json!({"b": 1, "a": 2});
        let document = cache_key_document(&candidate, "src", "dst", "sig");
        assert_eq!(
            document,
            r#"{"candidate":{"a":2,"b":1},"matcher_signature":"sig","source_fingerprint_hash":"src","target_fingerprint_hash":"dst"}"#
        );
    }

    #[tokio::test]
    async fn an_absent_file_opens_empty() {
        let directory = tempfile::tempdir().unwrap();
        let cache = MatchCache::open(directory.path().join("cache.json"))
            .await
            .unwrap();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert!(cache.lookup("missing").await.is_none());
        assert!(!cache.path().exists());
    }

    #[tokio::test]
    async fn records_round_trip_through_the_file() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("nested/cache.json");
        let mut cache = MatchCache::open(&path).await.unwrap();
        cache
            .record("k1".to_string(), sample("first"))
            .await
            .unwrap();
        cache
            .record("k2".to_string(), sample("second"))
            .await
            .unwrap();
        assert_eq!(cache.len(), 2);
        assert!(cache.contains("k1"));
        assert_eq!(cache.keys().collect::<Vec<&str>>(), ["k1", "k2"]);

        let reopened = MatchCache::open(&path).await.unwrap();
        assert_eq!(reopened.len(), 2);
        let entry = reopened.lookup("k2").await.expect("recorded");
        assert_eq!(entry.candidate_id, "tool:a->tool:b");
        assert_eq!(entry.updated_at, "2024-01-02T03:04:05.000Z");
        assert_eq!(entry.matches[0]["reason"], "second");
        assert_eq!(entry.diagnostics, json!([]));
    }

    #[tokio::test]
    async fn the_file_layout_is_byte_stable() {
        let directory = tempfile::tempdir().unwrap();
        let first_path = directory.path().join("first.json");
        let second_path = directory.path().join("second.json");
        let mut first = MatchCache::open(&first_path).await.unwrap();
        first
            .record("b".to_string(), sample("second"))
            .await
            .unwrap();
        first
            .record("a".to_string(), sample("first"))
            .await
            .unwrap();
        let mut second = MatchCache::open(&second_path).await.unwrap();
        second
            .record("a".to_string(), sample("first"))
            .await
            .unwrap();
        second
            .record("b".to_string(), sample("second"))
            .await
            .unwrap();
        let first_bytes = tokio::fs::read(&first_path).await.unwrap();
        let second_bytes = tokio::fs::read(&second_path).await.unwrap();
        assert_eq!(first_bytes, second_bytes);
        let text = String::from_utf8(first_bytes).unwrap();
        assert!(text.starts_with("{\n  \"schema_version\": \"Symphony-relation-match-cache-v1\","));
        // `a` sorts before `b` in the records object.
        let a_index = text.find("\"a\": {").expect("a record");
        let b_index = text.find("\"b\": {").expect("b record");
        assert!(a_index < b_index);
    }

    #[tokio::test]
    async fn clear_empties_the_file_without_deleting_it() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("cache.json");
        let mut cache = MatchCache::open(&path).await.unwrap();
        cache.record("k".to_string(), sample("x")).await.unwrap();
        cache.clear().await.unwrap();
        assert!(cache.is_empty());
        assert!(path.exists());
        let reopened = MatchCache::open(&path).await.unwrap();
        assert!(reopened.is_empty());
    }

    #[tokio::test]
    async fn a_malformed_file_is_a_validation_error_naming_it() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("broken.json");
        tokio::fs::write(&path, b"{not json").await.unwrap();
        let error = MatchCache::open(&path).await.expect_err("malformed JSON");
        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
        assert!(error.message().contains("broken.json"));

        for contents in [
            "[]",
            r#"{"records": {"k": {}}}"#,
            r#"{"schema_version": "other", "records": {}}"#,
            r#"{"schema_version": "Symphony-relation-match-cache-v1", "records": []}"#,
        ] {
            tokio::fs::write(&path, contents.as_bytes()).await.unwrap();
            let error = MatchCache::open(&path).await.expect_err(contents);
            assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
            assert!(error.message().contains("broken.json"));
        }
    }

    #[tokio::test]
    async fn malformed_records_are_skipped_but_valid_ones_survive() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("cache.json");
        let mut cache = MatchCache::open(&path).await.unwrap();
        cache.record("good".to_string(), sample("x")).await.unwrap();
        let text = tokio::fs::read_to_string(&path).await.unwrap();
        let broken = text.replace("\"records\": {", "\"records\": {\"bad\": 7, ");
        tokio::fs::write(&path, broken.as_bytes()).await.unwrap();
        let reopened = MatchCache::open(&path).await.unwrap();
        assert_eq!(reopened.len(), 1);
        assert!(reopened.contains("good"));
        assert!(!reopened.contains("bad"));
    }

    #[tokio::test]
    async fn an_empty_file_opens_empty() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("empty.json");
        tokio::fs::write(&path, b"  \n").await.unwrap();
        assert!(MatchCache::open(&path).await.unwrap().is_empty());
    }

    #[test]
    fn cached_match_defaults_to_empty_arrays_and_tolerates_unknown_fields() {
        let record: CachedMatch =
            serde_json::from_value(json!({"unknown": true})).expect("tolerant");
        assert_eq!(record.matches, json!([]));
        assert_eq!(record.diagnostics, json!([]));
        assert_eq!(record.candidate_id, "");
        assert_eq!(record.updated_at, "");
    }
}
