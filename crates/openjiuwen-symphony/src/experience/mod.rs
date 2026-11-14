//! The experience bank: distilled capability-combination knowledge with ranking.
//!
//! Python's `openjiuwen.symphony.experience` package turns observed traces into a
//! small knowledge base of "queries like this are served by these skills" entries and
//! ranks it with FAISS over an embedding model. This module ports the durable, offline
//! core of that idea:
//!
//! - [`ExperienceItem`](crate::experience::ExperienceItem) — one entry: an id, a query
//!   pattern, example queries, the skill ids that serve the pattern, a success counter,
//!   an optional embedding, and two metadata timestamps.
//! - [`ExperienceStore`](crate::experience::ExperienceStore) — the persistence seam
//!   (in-memory, one JSON document file, or `SQLite` behind the `sqlite` feature).
//! - [`ExperienceBank`](crate::experience::ExperienceBank) — the authoritative in-memory
//!   catalog with write-through persistence, generated `exp_NNNN` ids, deterministic
//!   dedupe, and search.
//! - [`ExperienceIndex`](crate::experience::ExperienceIndex) — the persisted,
//!   embedding-backed catalog: one vector per item, a JSONL scalar file, and an integrity
//!   manifest under an index directory (Python's `ExperienceBank(index_dir, ...)` layout
//!   without faiss).
//! - [`ExperienceHit`](crate::experience::ExperienceHit),
//!   [`ExperienceRanking`](crate::experience::ExperienceRanking), and
//!   [`SearchOptions`](crate::experience::SearchOptions) — the ranking configuration and
//!   result types.
//!
//! # Two catalogs, one rule set
//!
//! [`ExperienceBank`](crate::experience::ExperienceBank) is the store-agnostic catalog: it
//! keeps items in memory, writes through an
//! [`ExperienceStore`](crate::experience::ExperienceStore), and re-derives embeddings at
//! search time. [`ExperienceIndex`](crate::experience::ExperienceIndex) is the durable
//! embedding catalog: it persists the vectors next to the items, rehydrates them on open
//! instead of re-embedding, and keeps an integrity manifest of both files. Both share the
//! generated-id counter, the dedupe merge (the crate-private `merge_item`), and the ranking
//! helpers, so an item behaves identically in either.
//!
//! # Ranking
//!
//! With an [`openjiuwen_retrieval::Embedding`] configured, hits are ranked by cosine
//! similarity against the query embedding (the same quantity Python's FAISS index
//! computes after L2 normalization). Without one, hits are ranked by a deterministic
//! lexical score over the normalized pattern, the example queries, and the skill ids,
//! built on [`crate::search::tokenize`]. Both paths filter on the requested skill ids,
//! truncate to `top_k`, and only then apply `min_score` — Python's
//! `search_by_embedding` applies its `threshold` after `faiss.search` truncation, and
//! the port keeps that order so a low `top_k` behaves identically.
//!
//! # Intentional difference: deterministic dedupe
//!
//! Python's bank appends blindly and treats `id` as the only key, so two entries with
//! the same pattern and skills coexist. The Rust port dedupes on
//! [`ExperienceItem::dedupe_key`](crate::experience::ExperienceItem::dedupe_key)
//! (normalized pattern plus the sorted skill ids) and merges duplicates instead: the
//! success counters add up, the example lists union in first-seen order, the later
//! `last_hit_at` wins, and an existing embedding survives. The merge keeps the id of the
//! entry that was stored first.
//!
//! `bank.rs` holds the store-backed catalog, `index/` the persisted embedding catalog,
//! `store.rs` the persistence seam and its backends, and `ranking.rs` the hit type, ranking
//! configuration, and scoring helpers.
//!
//! # Ingestion: trace → cluster → collector
//!
//! `trace/` parses agent-runtime session exports into [`TraceRecord`]s and caches them
//! next to the sessions; `cluster.rs` groups those records by skill set and query topic;
//! `collector.rs` turns each cluster into an [`ExperienceItem`] and writes it into an
//! [`ExperienceBank`] or an [`ExperienceIndex`]. Python's LLM distillation is replaced by
//! the cluster's centroid query and a deterministic token-similarity merge, so the whole
//! input half works offline.
#![allow(clippy::module_name_repetitions)]

mod bank;
pub mod cluster;
pub mod collector;
mod index;
mod ranking;
mod store;
pub mod trace;

use std::collections::BTreeSet;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub use bank::{AddOutcome, ExperienceBank};
pub use cluster::{cluster_traces, ClusteredQuery, CLUSTER_SIMILARITY_THRESHOLD};
pub use collector::{
    CollectorStats, ExperienceCollector, ExperienceSink, PATTERN_MERGE_THRESHOLD,
    PENDING_FLUSH_THRESHOLD, QUERY_EXAMPLES_COUNT,
};
pub use index::{ExperienceIndex, ExperienceIndexMeta};
pub use index::{FLAT_VECTOR_ALGORITHM, MANIFEST_VERSION};
pub use ranking::{
    cosine_similarity, ExperienceHit, ExperienceRanking, SearchOptions, DEFAULT_TOP_K,
};
#[cfg(feature = "sqlite")]
pub use store::SqliteExperienceStore;
pub use store::{
    ExperienceStore, JsonFileExperienceStore, MemoryExperienceStore, DEFAULT_KEY_PREFIX,
};
pub use trace::{SessionPaths, TraceRecord, TraceStore, TraceToolCall};

/// The prefix of every generated experience id (`exp_0001`, `exp_0002`, ...).
pub const ID_PREFIX: &str = "exp_";

/// The minimum number of digits a generated id is zero-padded to.
pub const ID_DIGITS: usize = 4;

/// The default [`ExperienceItem::success_count`] (Python's `success_count: int = 1`).
pub const DEFAULT_SUCCESS_COUNT: u32 = 1;

/// One entry in the experience bank.
///
/// An item says "queries that look like `query_pattern` are served by `skill_ids`" and
/// carries the evidence gathered so far: the example queries that matched the pattern,
/// how often the combination succeeded, and when it was created and last hit.
///
/// `embedding` is accepted on load but never written out ([`serde::Serialize`] skips
/// it), matching Python's `to_dict`, which omits it from `metadata.jsonl` while
/// `from_dict` still reads it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct ExperienceItem {
    /// The item id; generated as `exp_NNNN` when an added item leaves it blank.
    pub id: String,
    /// The distilled query pattern this item stands for.
    pub query_pattern: String,
    /// Example queries that matched the pattern, in first-seen order.
    pub query_examples: Vec<String>,
    /// The ids of the skills that serve the pattern.
    pub skill_ids: Vec<String>,
    /// How many times this pattern/skill combination succeeded; defaults to `1`.
    pub success_count: u32,
    /// The embedding of the pattern, when one was computed; never serialized.
    #[serde(default, skip_serializing)]
    pub embedding: Vec<f32>,
    /// Creation time in epoch seconds (metadata only, never used for ordering).
    pub created_at: f64,
    /// Last-hit time in epoch seconds (metadata only, never used for ordering).
    pub last_hit_at: f64,
}

impl Default for ExperienceItem {
    /// Returns an item with the Python defaults: a success count of
    /// [`DEFAULT_SUCCESS_COUNT`] and zeroed timestamps, so a bank built without an
    /// explicit clock stays deterministic.
    fn default() -> Self {
        Self {
            id: String::new(),
            query_pattern: String::new(),
            query_examples: Vec::new(),
            skill_ids: Vec::new(),
            success_count: DEFAULT_SUCCESS_COUNT,
            embedding: Vec::new(),
            created_at: 0.0,
            last_hit_at: 0.0,
        }
    }
}

impl ExperienceItem {
    /// Creates an item for `query_pattern` with a blank id and the default counters.
    ///
    /// [`ExperienceBank::add`] assigns an `exp_NNNN` id; call [`Self::with_id`] when
    /// the id has to be chosen by the caller.
    #[must_use]
    pub fn new(query_pattern: impl Into<String>) -> Self {
        Self {
            query_pattern: query_pattern.into(),
            ..Self::default()
        }
    }

    /// Sets the item id, overriding the generated one.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Replaces the example queries.
    #[must_use]
    pub fn with_query_examples(
        mut self,
        query_examples: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.query_examples = query_examples.into_iter().map(Into::into).collect();
        self
    }

    /// Replaces the skill ids.
    #[must_use]
    pub fn with_skill_ids(
        mut self,
        skill_ids: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.skill_ids = skill_ids.into_iter().map(Into::into).collect();
        self
    }

    /// Sets the success counter.
    #[must_use]
    pub fn with_success_count(mut self, success_count: u32) -> Self {
        self.success_count = success_count;
        self
    }

    /// Sets the embedding vector.
    #[must_use]
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = embedding;
        self
    }

    /// Sets the creation timestamp (epoch seconds).
    #[must_use]
    pub fn with_created_at(mut self, created_at: f64) -> Self {
        self.created_at = created_at;
        self
    }

    /// Sets the last-hit timestamp (epoch seconds).
    #[must_use]
    pub fn with_last_hit_at(mut self, last_hit_at: f64) -> Self {
        self.last_hit_at = last_hit_at;
        self
    }

    /// The query pattern lowercased with whitespace runs collapsed to single spaces.
    ///
    /// This is the form used for dedupe and lexical ranking, so
    /// `"  Summarize   the Report "` and `"summarize the report"` are the same pattern.
    #[must_use]
    pub fn normalized_pattern(&self) -> String {
        self.query_pattern
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase()
    }

    /// A deterministic dedupe key: the SHA-256 hex digest of the normalized pattern
    /// plus the sorted, deduped skill ids.
    ///
    /// Skill id *order* does not matter, so two items that list the same skills in a
    /// different order collide — which is what the merge in [`ExperienceBank::add`]
    /// expects. Example queries, counts, and timestamps are not part of the key.
    #[must_use]
    pub fn dedupe_key(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.normalized_pattern().as_bytes());
        let mut skill_ids: Vec<&str> = self.skill_ids.iter().map(String::as_str).collect();
        skill_ids.sort_unstable();
        skill_ids.dedup();
        for skill_id in skill_ids {
            hasher.update([0x1f]);
            hasher.update(skill_id.as_bytes());
        }
        format!("{:x}", hasher.finalize())
    }

    /// The text indexed for embedding and lexical ranking: the pattern, the example
    /// queries, and the skill ids, one per line.
    #[must_use]
    pub fn search_text(&self) -> String {
        let mut parts = Vec::with_capacity(2 + self.query_examples.len() + self.skill_ids.len());
        parts.push(self.normalized_pattern());
        parts.extend(self.query_examples.iter().cloned());
        parts.extend(self.skill_ids.iter().cloned());
        let joined = parts.join("\n");
        joined.trim().to_string()
    }

    /// The distinct skill ids of this item, in sorted order.
    #[must_use]
    pub fn skill_set(&self) -> BTreeSet<&str> {
        self.skill_ids.iter().map(String::as_str).collect()
    }

    /// Reports whether this item lists `skill_id`.
    #[must_use]
    pub fn has_skill(&self, skill_id: &str) -> bool {
        self.skill_ids.iter().any(|known| known == skill_id)
    }
}

/// Merges `incoming` into `existing`, keeping `existing`'s id, pattern, and embedding.
///
/// The success counters add up, the example lists union in first-seen order, the later
/// `last_hit_at` wins, and `existing`'s embedding survives unless it was empty. The
/// incoming id is discarded: the entry stored first keeps its identity, so repeated
/// observations reinforce one record instead of fragmenting the catalog.
///
/// This is the single merge rule shared by
/// [`ExperienceBank::add`](crate::experience::ExperienceBank::add) and
/// [`ExperienceIndex::add`](crate::experience::ExperienceIndex::add).
pub(crate) fn merge_item(existing: &mut ExperienceItem, incoming: ExperienceItem) {
    existing.success_count = existing
        .success_count
        .saturating_add(incoming.success_count);
    for example in incoming.query_examples {
        if !existing.query_examples.contains(&example) {
            existing.query_examples.push(example);
        }
    }
    existing.last_hit_at = existing.last_hit_at.max(incoming.last_hit_at);
    if existing.embedding.is_empty() {
        existing.embedding = incoming.embedding;
    }
}

/// Parses the ordinal of a generated `exp_NNNN` id.
///
/// Returns `None` for an id without the [`ID_PREFIX`], for an all-digit suffix that does
/// not fit a `u64`, and for any other suffix, so a malformed id never aborts a catalog
/// scan or pushes the generated-id counter around.
pub(crate) fn ordinal_of(id: &str) -> Option<u64> {
    let digits = id.strip_prefix(ID_PREFIX)?;
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_python() {
        let item = ExperienceItem::default();
        assert_eq!(item.success_count, DEFAULT_SUCCESS_COUNT);
        assert!(item.id.is_empty());
        assert!(item.embedding.is_empty());
        assert!((item.created_at - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn normalization_lowercases_and_collapses_whitespace() {
        let item = ExperienceItem::new("  Summarize \n the\tReport  ");
        assert_eq!(item.normalized_pattern(), "summarize the report");
        assert_eq!(ExperienceItem::new("   ").normalized_pattern(), "");
    }

    #[test]
    fn dedupe_key_ignores_skill_order_but_not_skill_membership() {
        let first = ExperienceItem::new("Translate Text").with_skill_ids(["translate", "format"]);
        let reordered = ExperienceItem::new(" translate   text ").with_skill_ids([
            "format",
            "translate",
            "format",
        ]);
        let different = ExperienceItem::new("Translate Text").with_skill_ids(["translate"]);

        assert_eq!(first.dedupe_key(), reordered.dedupe_key());
        assert_ne!(first.dedupe_key(), different.dedupe_key());
        assert_eq!(first.dedupe_key().len(), 64);
    }

    #[test]
    fn embedding_is_accepted_on_load_but_skipped_on_save() {
        let item = ExperienceItem::new("Summarize the report")
            .with_id("exp_0001")
            .with_embedding(vec![0.25, -0.5]);
        let document = serde_json::to_value(&item).expect("serialize");
        assert!(document.get("embedding").is_none());

        let mut loaded = document;
        loaded["embedding"] = serde_json::json!([0.25, -0.5]);
        loaded["unknown_field"] = serde_json::json!("tolerated");
        let parsed: ExperienceItem = serde_json::from_value(loaded).expect("deserialize");
        assert_eq!(parsed.embedding, vec![0.25, -0.5]);
        assert_eq!(parsed.id, "exp_0001");
    }

    #[test]
    fn missing_fields_fall_back_to_the_defaults() {
        let parsed: ExperienceItem =
            serde_json::from_str(r#"{"id":"exp_0007","query_pattern":"do a thing"}"#)
                .expect("deserialize");
        assert_eq!(parsed.success_count, DEFAULT_SUCCESS_COUNT);
        assert!(parsed.query_examples.is_empty());
        assert!(parsed.skill_ids.is_empty());
    }

    #[test]
    fn search_text_includes_pattern_examples_and_skills() {
        let item = ExperienceItem::new("Summarize")
            .with_query_examples(["Summarize this"])
            .with_skill_ids(["summarizer"]);
        assert_eq!(item.search_text(), "summarize\nSummarize this\nsummarizer");
        assert!(item.has_skill("summarizer"));
        assert!(!item.has_skill("translator"));
        assert_eq!(item.skill_set().len(), 1);
    }

    #[test]
    fn ordinals_are_parsed_from_generated_ids_only() {
        assert_eq!(ordinal_of("exp_0001"), Some(1));
        assert_eq!(ordinal_of("exp_42"), Some(42));
        assert_eq!(ordinal_of("exp_"), None);
        assert_eq!(ordinal_of("exp_x"), None);
        assert_eq!(ordinal_of("custom_7"), None);
        assert_eq!(ordinal_of("exp_999999999999999999999999"), None);
    }

    #[test]
    fn merge_unions_examples_and_counts() {
        let mut existing = ExperienceItem::new("Summarize")
            .with_id("exp_0001")
            .with_query_examples(["one", "two"])
            .with_success_count(2)
            .with_last_hit_at(5.0);
        let incoming = ExperienceItem::new("summarize")
            .with_query_examples(["two", "three"])
            .with_success_count(3)
            .with_last_hit_at(9.0)
            .with_embedding(vec![1.0]);
        merge_item(&mut existing, incoming);
        assert_eq!(existing.id, "exp_0001");
        assert_eq!(existing.success_count, 5);
        assert_eq!(existing.query_examples, vec!["one", "two", "three"]);
        assert!((existing.last_hit_at - 9.0).abs() < f64::EPSILON);
        assert_eq!(existing.embedding, vec![1.0]);
    }

    #[test]
    fn merge_keeps_the_existing_embedding_and_the_latest_hit() {
        let mut existing = ExperienceItem::new("Summarize")
            .with_id("exp_0001")
            .with_embedding(vec![1.0, 0.0])
            .with_last_hit_at(9.0);
        let incoming = ExperienceItem::new("summarize")
            .with_id("exp_0009")
            .with_embedding(vec![0.0, 1.0])
            .with_last_hit_at(3.0);
        merge_item(&mut existing, incoming);
        assert_eq!(existing.id, "exp_0001");
        assert_eq!(existing.embedding, vec![1.0, 0.0]);
        assert!((existing.last_hit_at - 9.0).abs() < f64::EPSILON);
    }
}
