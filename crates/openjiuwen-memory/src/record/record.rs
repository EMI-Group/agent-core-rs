//! [`MemoryRecord`], Python-compatible record ids, and text normalization.
//!
//! The JSON shape is the Python `MemoryDoc`/`MemResult` shape: `content` is written
//! as `"text"` (and accepts `"content"`/`"mem"`), `kind` as `"type"` (and accepts
//! `"kind"`/`"mem_type"`), `metadata` as `"fields"` (and accepts `"metadata"`), and
//! the scope fields are flattened onto the top level.
use super::kind::MemoryType;
use super::scope::MemoryScope;
use chrono::{DateTime, Utc};
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use uuid::Uuid;
/// The default importance of a record when the caller does not set one.
const DEFAULT_IMPORTANCE: f32 = 0.5;
/// The 64-bit FNV-1a offset basis.
const FNV1A_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// The 64-bit FNV-1a prime.
const FNV1A_PRIME: u64 = 0x0000_0100_0000_01b3;
/// The lowercase hex digits used to render memory ids.
const HEX_DIGITS: &[u8; 16] = b"0123456789abcdef";
/// Returns the current UTC time; the serde default for record timestamps.
fn utc_now() -> DateTime<Utc> {
    Utc::now()
}
/// Returns [`DEFAULT_IMPORTANCE`]; the serde default for [`MemoryRecord::importance`].
fn default_importance() -> f32 {
    DEFAULT_IMPORTANCE
}
/// A stored memory record.
///
/// Every record carries its scope, so a record can be persisted, moved, and rendered
/// without consulting the engine. Timestamps are RFC 3339; `importance` lives in
/// `0.0..=1.0` and defaults to `0.5`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryRecord {
    /// Unique record id (24 lowercase hex characters, see [`new_memory_id`]).
    pub id: String,
    /// The memory text; JSON field `"text"` (accepts `"content"` and `"mem"`).
    #[serde(rename = "text", alias = "content", alias = "mem")]
    pub content: String,
    /// The memory kind; JSON field `"type"` (accepts `"kind"` and `"mem_type"`).
    #[serde(rename = "type", alias = "kind", alias = "mem_type")]
    pub kind: MemoryType,
    /// Relative importance in `0.0..=1.0`; defaults to `0.5`.
    #[serde(default = "default_importance")]
    pub importance: f32,
    /// Creation time; defaults to "now" when absent from the input.
    #[serde(default = "utc_now")]
    pub timestamp: DateTime<Utc>,
    /// Last modification time; defaults to "now" when absent from the input.
    #[serde(default = "utc_now")]
    pub updated_at: DateTime<Utc>,
    /// Time of the most recent recall that returned this record.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_access: Option<DateTime<Utc>>,
    /// Number of recalls that returned this record.
    #[serde(default)]
    pub access_count: u64,
    /// Free-form structured fields; JSON field `"fields"` (accepts `"metadata"`).
    #[serde(rename = "fields", alias = "metadata", default)]
    pub metadata: BTreeMap<String, Value>,
    /// Cached embedding of `content`, when the engine computed one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
    /// Optional provenance marker (conversation, workflow node, import).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Ownership scope, flattened onto the top level of the JSON object.
    #[serde(flatten)]
    pub scope: MemoryScope,
}
impl MemoryRecord {
    /// Builds a record with a generated id, the current time, and importance `0.5`.
    #[must_use]
    pub fn new(scope: MemoryScope, content: impl Into<String>, kind: MemoryType) -> Self {
        let now = utc_now();
        let id = new_memory_id(&scope.user_id, now);
        Self {
            id,
            content: content.into(),
            kind,
            importance: DEFAULT_IMPORTANCE,
            timestamp: now,
            updated_at: now,
            last_access: None,
            access_count: 0,
            metadata: BTreeMap::new(),
            embedding: None,
            source: None,
            scope,
        }
    }
    /// Returns the record with `id` replaced.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }
    /// Returns the record with `importance` replaced.
    #[must_use]
    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance;
        self
    }
    /// Returns the record with one metadata field inserted (incoming value wins).
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
    /// Returns the record with `source` replaced.
    #[must_use]
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }
    /// Returns the record with `timestamp` replaced.
    #[must_use]
    pub fn with_timestamp(mut self, timestamp: DateTime<Utc>) -> Self {
        self.timestamp = timestamp;
        self
    }
    /// Returns the record with `embedding` replaced.
    #[must_use]
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }
    /// Validates the record's invariants.
    ///
    /// The id must be non-empty, the content must contain a non-whitespace character,
    /// `importance` must be finite and inside `0.0..=1.0`, and the scope must be valid.
    pub fn validate(&self) -> Result<()> {
        if self.id.is_empty() {
            return Err(Error::validation("memory record `id` must not be empty"));
        }
        if self.content.trim().is_empty() {
            return Err(Error::validation(
                "memory record `content` must not be empty or whitespace",
            ));
        }
        if !self.importance.is_finite() || !(0.0..=1.0).contains(&self.importance) {
            return Err(Error::validation(format!(
                "memory record `importance` must be within 0.0..=1.0 (got {})",
                self.importance
            )));
        }
        self.scope.validate()
    }
    /// Returns the string value of a metadata field, if present and a string.
    #[must_use]
    pub fn metadata_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(Value::as_str)
    }
    /// Records one recall access: sets `last_access`/`updated_at` to `now` and bumps
    /// `access_count`.
    pub fn touch(&mut self, now: DateTime<Utc>) {
        self.last_access = Some(now);
        self.updated_at = now;
        self.access_count = self.access_count.saturating_add(1);
    }
}
/// Generates a 24-character lowercase hex memory id.
///
/// Ports the Python `MemoryDoc` id layout: 6 bytes of big-endian millisecond
/// timestamp, 3 random bytes from `Uuid::new_v4()`, and the low 3 bytes of the
/// FNV-1a 64-bit hash of `user_id`. Ids are time-ordered and unique per user without
/// a shared counter.
#[must_use]
pub fn new_memory_id(user_id: &str, now: DateTime<Utc>) -> String {
    let millis = now.timestamp_millis() as u64;
    let hash = fnv1a_64(user_id.as_bytes());
    let uuid = Uuid::new_v4();
    let mut bytes = [0_u8; 12];
    bytes[..6].copy_from_slice(&millis.to_be_bytes()[2..]);
    bytes[6..9].copy_from_slice(&uuid.as_bytes()[..3]);
    bytes[9..].copy_from_slice(&hash.to_le_bytes()[..3]);
    let mut id = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        id.push(char::from(HEX_DIGITS[usize::from(byte >> 4)]));
        id.push(char::from(HEX_DIGITS[usize::from(byte & 0x0f)]));
    }
    id
}
/// Computes the 64-bit FNV-1a hash of `bytes` (the same constant family as Python's
/// `hashlib`-free helper used for short ids).
fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash = FNV1A_OFFSET_BASIS;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV1A_PRIME);
    }
    hash
}
/// Normalizes memory text for duplicate detection.
///
/// The text is lowercased, every whitespace run collapses to one space, the result is
/// trimmed, and any surrounding non-alphanumeric characters (punctuation, quotes,
/// brackets) are stripped. Two contents that differ only in case, spacing, or edge
/// punctuation therefore normalize to the same string.
#[must_use]
pub fn normalize_text(text: &str) -> String {
    let lowered = text.to_lowercase();
    let collapsed = lowered.split_whitespace().collect::<Vec<_>>().join(" ");
    collapsed
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_owned()
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    fn fixed_time() -> DateTime<Utc> {
        DateTime::from_timestamp(1_700_000_000, 0).unwrap()
    }
    fn sample_record() -> MemoryRecord {
        MemoryRecord::new(
            MemoryScope::new("alice", "ws"),
            "Alice prefers dark mode",
            MemoryType::SemanticMemory,
        )
        .with_id("000000000000000000000001")
        .with_importance(0.7)
        .with_timestamp(fixed_time())
        .with_metadata("key", serde_json::json!("value"))
        .with_source("conversation")
    }
    #[test]
    fn new_sets_defaults_and_generated_id() {
        let record = MemoryRecord::new(
            MemoryScope::new("alice", "ws"),
            "hello",
            MemoryType::SemanticMemory,
        );
        assert_eq!(record.id.len(), 24);
        assert!(record
            .id
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
        assert!((record.importance - 0.5).abs() < f32::EPSILON);
        assert_eq!(record.access_count, 0);
        assert!(record.last_access.is_none());
        assert!(record.embedding.is_none());
        assert!(record.source.is_none());
        assert_eq!(record.scope.user_id, "alice");
    }
    #[test]
    fn json_round_trip_uses_python_field_names() {
        let record = sample_record();
        let json = serde_json::to_value(&record).unwrap();
        assert_eq!(json["text"], serde_json::json!("Alice prefers dark mode"));
        assert_eq!(json["type"], serde_json::json!("semantic_memory"));
        assert_eq!(json["fields"]["key"], serde_json::json!("value"));
        assert_eq!(json["user_id"], serde_json::json!("alice"));
        assert_eq!(json["scope_id"], serde_json::json!("ws"));
        assert!(json.get("content").is_none());
        assert!(json.get("kind").is_none());
        assert!(json.get("metadata").is_none());
        let restored: MemoryRecord = serde_json::from_value(json).unwrap();
        assert_eq!(restored, record);
    }
    #[test]
    fn deserializes_python_doc_shape() {
        let json = serde_json::json!({
            "id": "000000000000000000000002",
            "text": "likes tea",
            "type": "user_profile",
            "timestamp": "2023-11-14T22:13:20Z",
            "fields": {"lang": "en"},
            "user_id": "alice",
            "scope_id": "ws"
        });
        let record: MemoryRecord = serde_json::from_value(json).unwrap();
        assert_eq!(record.content, "likes tea");
        assert_eq!(record.kind, MemoryType::UserProfile);
        assert_eq!(record.metadata_str("lang"), Some("en"));
        assert!((record.importance - 0.5).abs() < f32::EPSILON);
        assert_eq!(record.timestamp, fixed_time());
    }
    #[test]
    fn deserializes_mem_result_aliases() {
        let json = serde_json::json!({
            "id": "000000000000000000000003",
            "mem": "went hiking",
            "mem_type": "episodic_memory",
            "metadata": {"where": "alps"},
            "user_id": "alice",
            "scope_id": "ws"
        });
        let record: MemoryRecord = serde_json::from_value(json).unwrap();
        assert_eq!(record.content, "went hiking");
        assert_eq!(record.kind, MemoryType::EpisodicMemory);
        assert_eq!(record.metadata_str("where"), Some("alps"));
    }
    #[test]
    fn deserializes_content_and_kind_aliases() {
        let json = serde_json::json!({
            "id": "000000000000000000000004",
            "content": "note",
            "kind": "summary",
            "user_id": "alice",
            "scope_id": "ws"
        });
        let record: MemoryRecord = serde_json::from_value(json).unwrap();
        assert_eq!(record.content, "note");
        assert_eq!(record.kind, MemoryType::Summary);
    }
    #[test]
    fn missing_scope_fields_default_to_default_id() {
        let json = serde_json::json!({
            "id": "000000000000000000000005",
            "text": "orphan",
            "type": "semantic_memory"
        });
        let record: MemoryRecord = serde_json::from_value(json).unwrap();
        assert_eq!(record.scope, MemoryScope::default());
    }
    #[test]
    fn validate_rejects_bad_records() {
        let mut record = sample_record();
        record.id = String::new();
        assert!(record.validate().unwrap_err().to_string().contains("id"));
        let mut record = sample_record();
        record.content = "   ".to_owned();
        assert!(record
            .validate()
            .unwrap_err()
            .to_string()
            .contains("content"));
        let mut record = sample_record();
        record.importance = 1.5;
        assert!(record
            .validate()
            .unwrap_err()
            .to_string()
            .contains("importance"));
        let mut record = sample_record();
        record.importance = f32::NAN;
        assert!(record
            .validate()
            .unwrap_err()
            .to_string()
            .contains("importance"));
        let mut record = sample_record();
        record.scope = MemoryScope::new("", "ws");
        assert!(record
            .validate()
            .unwrap_err()
            .to_string()
            .contains("user_id"));
        assert!(sample_record().validate().is_ok());
    }
    #[test]
    fn builders_are_chainable() {
        let record = sample_record().with_embedding(vec![0.1, 0.2]);
        assert_eq!(record.embedding.as_deref(), Some(&[0.1_f32, 0.2][..]));
        assert_eq!(record.source.as_deref(), Some("conversation"));
        assert_eq!(record.metadata_str("key"), Some("value"));
        assert_eq!(record.metadata_str("absent"), None);
    }
    #[test]
    fn touch_updates_access_fields() {
        let mut record = sample_record();
        let later = DateTime::from_timestamp(1_700_000_100, 0).unwrap();
        record.touch(later);
        assert_eq!(record.last_access, Some(later));
        assert_eq!(record.updated_at, later);
        assert_eq!(record.access_count, 1);
        record.touch(later);
        assert_eq!(record.access_count, 2);
        assert_eq!(record.timestamp, fixed_time());
    }
    #[test]
    fn new_memory_id_is_24_lowercase_hex_and_time_ordered() {
        let earlier = DateTime::from_timestamp(1_700_000_000, 0).unwrap();
        let later = DateTime::from_timestamp(1_700_000_001, 0).unwrap();
        let first = new_memory_id("alice", earlier);
        let second = new_memory_id("alice", later);
        assert_eq!(first.len(), 24);
        assert_eq!(second.len(), 24);
        assert!(first
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
        assert!(first < second);
        let ids: std::collections::HashSet<String> =
            (0..16).map(|_| new_memory_id("alice", earlier)).collect();
        assert_eq!(ids.len(), 16);
    }
    #[test]
    fn new_memory_id_is_user_specific_at_fixed_time() {
        let now = fixed_time();
        let alice = new_memory_id("alice", now);
        let bob = new_memory_id("bob", now);
        assert_eq!(alice[..12], bob[..12]);
        assert_eq!(alice[18..], expected_user_suffix("alice"));
        assert_eq!(bob[18..], expected_user_suffix("bob"));
        assert_ne!(expected_user_suffix("alice"), expected_user_suffix("bob"));
    }
    /// Renders the deterministic 3-byte FNV-1a user suffix the id builder must use.
    fn expected_user_suffix(user_id: &str) -> String {
        use std::fmt::Write as _;
        fnv1a_64(user_id.as_bytes()).to_le_bytes()[..3].iter().fold(
            String::new(),
            |mut out, byte| {
                write!(out, "{byte:02x}").unwrap();
                out
            },
        )
    }
    #[test]
    fn normalize_text_lowercases_collapses_and_strips() {
        assert_eq!(normalize_text("  Hello   World  "), "hello world");
        assert_eq!(normalize_text("\"Hello, World!\""), "hello, world");
        assert_eq!(
            normalize_text("Alice prefers dark mode."),
            "alice prefers dark mode"
        );
        assert_eq!(normalize_text("---"), "");
        assert_eq!(normalize_text(""), "");
        assert_eq!(
            normalize_text("Alice PREFERS\tdark\nmode"),
            "alice prefers dark mode"
        );
    }
}
