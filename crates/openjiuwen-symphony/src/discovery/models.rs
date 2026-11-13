//! The skill catalog models: one skill record and the fingerprinted inventory.
//!
//! Ports Python `openjiuwen.symphony.discovery.models` (`SkillRecord`,
//! `SkillInventory`). A [`SkillRecord`] is the host-visible identity of one installed
//! skill folder; a [`SkillInventory`] is the whole catalog plus a deterministic
//! fingerprint of it.
//!
//! The fingerprint is the SHA-256 hex digest of the canonical JSON of the records
//! sorted by `worker_id`. Records are sorted *before* hashing (a plain sorted `Vec`, no
//! map iteration) and object keys are emitted in sorted order, so the
//! digest does not depend on directory iteration order, `serde_json` map ordering, or
//! the order a caller happens to build the vector in.

use std::collections::BTreeSet;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// The default skill source when a record does not name one.
pub const DEFAULT_SOURCE: &str = "local";

/// Returns [`DEFAULT_SOURCE`] for the `serde` field default.
fn default_source() -> String {
    DEFAULT_SOURCE.to_string()
}

/// The host-visible identity of one installed skill folder.
///
/// Mirrors Python `openjiuwen.symphony.discovery.models.SkillRecord`: the folder's
/// worker id (`worker_id`), its display fields, the `SKILL.md` path relative to the
/// scan root (`skill_file`), provenance (`source`, `source_root`, `author`,
/// `version`), the content hash of the skill file, and optional aliases.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SkillRecord {
    /// The stable worker id of the skill (the capability id of its descriptor).
    pub worker_id: String,
    /// Display name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// The `SKILL.md` path relative to the scan root, with `/` separators.
    pub skill_file: String,
    /// Where the skill came from; [`DEFAULT_SOURCE`] unless the scan overrides it.
    #[serde(default = "default_source")]
    pub source: String,
    /// The skill version from its front matter; empty when unset.
    pub version: String,
    /// The skill author from its front matter; empty when unset.
    pub author: String,
    /// The SHA-256 hex digest of the raw `SKILL.md` bytes.
    pub content_hash: String,
    /// The absolute root the skill was discovered under.
    pub source_root: String,
    /// Alternate names the skill answers to.
    pub aliases: Vec<String>,
}

impl SkillRecord {
    /// Builds a record with `worker_id` and `name`; everything else is defaulted and
    /// `source` is [`DEFAULT_SOURCE`].
    #[must_use]
    pub fn new(worker_id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            worker_id: worker_id.into(),
            name: name.into(),
            source: DEFAULT_SOURCE.to_string(),
            ..Self::default()
        }
    }

    /// Sets the description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Sets the relative skill-file path.
    #[must_use]
    pub fn with_skill_file(mut self, skill_file: impl Into<String>) -> Self {
        self.skill_file = skill_file.into();
        self
    }

    /// Sets the source.
    #[must_use]
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = source.into();
        self
    }

    /// Sets the version.
    #[must_use]
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Sets the author.
    #[must_use]
    pub fn with_author(mut self, author: impl Into<String>) -> Self {
        self.author = author.into();
        self
    }

    /// Sets the content hash.
    #[must_use]
    pub fn with_content_hash(mut self, content_hash: impl Into<String>) -> Self {
        self.content_hash = content_hash.into();
        self
    }

    /// Sets the absolute scan root.
    #[must_use]
    pub fn with_source_root(mut self, source_root: impl Into<String>) -> Self {
        self.source_root = source_root.into();
        self
    }

    /// Replaces the aliases.
    #[must_use]
    pub fn with_aliases(mut self, aliases: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.aliases = aliases.into_iter().map(Into::into).collect();
        self
    }
}

/// The fingerprinted catalog of discovered skills.
///
/// `items` is always sorted by `worker_id` (then `skill_file`), `count` is its length,
/// and `fingerprint` is the SHA-256 hex digest of the canonical JSON of `items`; see
/// [`inventory_fingerprint`].
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SkillInventory {
    /// The records, sorted by `worker_id`.
    pub items: Vec<SkillRecord>,
    /// The SHA-256 hex digest of the canonical JSON of `items`.
    pub fingerprint: String,
    /// The number of records in `items`.
    pub count: usize,
}

impl SkillInventory {
    /// Builds an inventory, sorting `items` and fingerprinting them.
    #[must_use]
    pub fn new(mut items: Vec<SkillRecord>) -> Self {
        sort_records(&mut items);
        let fingerprint = inventory_fingerprint(&items);
        let count = items.len();
        Self {
            items,
            fingerprint,
            count,
        }
    }

    /// Whether the inventory holds no records.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Looks a record up by its worker id.
    #[must_use]
    pub fn get(&self, worker_id: &str) -> Option<&SkillRecord> {
        self.items
            .iter()
            .find(|record| record.worker_id == worker_id)
    }

    /// The worker ids of the inventory, in sorted order.
    #[must_use]
    pub fn worker_ids(&self) -> Vec<&str> {
        self.items
            .iter()
            .map(|record| record.worker_id.as_str())
            .collect()
    }
}

/// Sorts records by `worker_id`, then by `skill_file` to keep the order total.
pub fn sort_records(records: &mut [SkillRecord]) {
    records.sort_by(|left, right| {
        left.worker_id
            .cmp(&right.worker_id)
            .then_with(|| left.skill_file.cmp(&right.skill_file))
    });
}

/// The SHA-256 hex digest of the canonical JSON of `records`, sorted by `worker_id`.
///
/// Sorting happens here as well as in [`SkillInventory::new`], so the fingerprint is
/// order-independent even when a caller hashes an unsorted slice. Every record
/// contributes its canonical JSON object (keys sorted); the records form a JSON array.
#[must_use]
pub fn inventory_fingerprint(records: &[SkillRecord]) -> String {
    let mut sorted: Vec<&SkillRecord> = records.iter().collect();
    sorted.sort_by(|left, right| {
        left.worker_id
            .cmp(&right.worker_id)
            .then_with(|| left.skill_file.cmp(&right.skill_file))
    });

    let mut hasher = Sha256::new();
    hasher.update(b"[");
    for (index, record) in sorted.iter().enumerate() {
        if index > 0 {
            hasher.update(b",");
        }
        hasher.update(canonical_json(record).as_bytes());
    }
    hasher.update(b"]");
    format!("{:x}", hasher.finalize())
}

/// Renders one record as canonical JSON: object keys sorted, no whitespace.
fn canonical_json(record: &SkillRecord) -> String {
    let value = serde_json::to_value(record).unwrap_or(Value::Null);
    let mut rendered = String::new();
    render_canonical(&value, &mut rendered);
    rendered
}

/// Writes `value` to `rendered` with object keys sorted, tolerating every value kind.
fn render_canonical(value: &Value, rendered: &mut String) {
    match value {
        Value::Object(mapping) => {
            rendered.push('{');
            let mut keys: Vec<&String> = mapping.keys().collect();
            keys.sort_unstable();
            for (index, key) in keys.into_iter().enumerate() {
                if index > 0 {
                    rendered.push(',');
                }
                render_canonical(&Value::String(key.clone()), rendered);
                rendered.push(':');
                if let Some(entry) = mapping.get(key) {
                    render_canonical(entry, rendered);
                }
            }
            rendered.push('}');
        }
        Value::Array(items) => {
            rendered.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    rendered.push(',');
                }
                render_canonical(item, rendered);
            }
            rendered.push(']');
        }
        scalar => rendered.push_str(&serde_json::to_string(scalar).unwrap_or_default()),
    }
}

/// Case-insensitive, order-preserving de-duplication of `values`.
///
/// Shared with the front-matter parser so tag and alias lists normalize the same way.
pub(crate) fn dedupe_ignore_case(values: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut deduped = Vec::new();
    for value in values {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            continue;
        }
        if seen.insert(trimmed.to_lowercase()) {
            deduped.push(trimmed.to_string());
        }
    }
    deduped
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn record(id: &str, description: &str) -> SkillRecord {
        SkillRecord::new(id, id.to_uppercase()).with_description(description)
    }

    #[test]
    fn inventory_sorts_and_counts() {
        let inventory = SkillInventory::new(vec![record("b", "second"), record("a", "first")]);
        assert_eq!(inventory.count, 2);
        assert_eq!(inventory.worker_ids(), ["a", "b"]);
        assert!(!inventory.is_empty());
        assert!(inventory.get("a").is_some());
        assert!(inventory.get("missing").is_none());
        assert_eq!(inventory.fingerprint.len(), 64);
    }

    #[test]
    fn fingerprint_is_order_independent_and_content_sensitive() {
        let first = record("a", "first");
        let second = record("b", "second");
        let forwards = inventory_fingerprint(&[first.clone(), second.clone()]);
        let backwards = inventory_fingerprint(&[second.clone(), first.clone()]);
        assert_eq!(forwards, backwards);
        assert_ne!(
            forwards,
            inventory_fingerprint(&[first, second.with_description("changed")])
        );
    }

    #[test]
    fn fingerprint_ignores_whitespace_insensitive_fields() {
        // Two inventories that differ only in the order they were built share a digest.
        let items = vec![record("b", "x"), record("a", "y")];
        let reversed: Vec<SkillRecord> = items.iter().rev().cloned().collect();
        assert_eq!(
            SkillInventory::new(items).fingerprint,
            SkillInventory::new(reversed).fingerprint
        );
    }

    #[test]
    fn default_source_is_local_and_survives_deserialization() {
        let record = SkillRecord::new("a", "A");
        assert_eq!(record.source, DEFAULT_SOURCE);
        let decoded: SkillRecord = serde_json::from_str(r#"{"worker_id":"a"}"#).unwrap();
        assert_eq!(decoded.source, DEFAULT_SOURCE);
    }

    #[test]
    fn records_tolerate_unknown_fields() {
        let decoded: SkillRecord =
            serde_json::from_str(r#"{"worker_id":"a","name":"A","future_field":42}"#).unwrap();
        assert_eq!(decoded.worker_id, "a");
    }

    #[test]
    fn dedupe_folds_case_and_keeps_first_spelling() {
        let deduped = dedupe_ignore_case(vec![
            "Pdf".to_string(),
            " pdf ".to_string(),
            String::new(),
            "Docs".to_string(),
        ]);
        assert_eq!(deduped, ["Pdf", "Docs"]);
    }
}
