//! Team memory storage: the [`TeamMemoryStore`] seam, its in-memory implementation,
//! the [`TeamMemory`] facade, and the shared search ranking.
//!
//! [`model`](super::model) is pure data; this module is the persistence boundary. A
//! [`TeamMemoryStore`] is the *only* authority for team memory records: the facade never
//! caches, and the unified-recall projection in [`super::projection`] is derived from
//! these rows, not stored beside them.
//!
//! # Layout
//!
//! - this file — [`TeamSearchOptions`], the [`TeamMemoryStore`] trait, and
//!   `rank_candidates`.
//! - `memory` — [`InMemoryTeamMemoryStore`], the reference implementation.
//! - `facade` — [`TeamMemory`], the ergonomic entry point (create/read/update/delete,
//!   search, expiry, statistics, and JSONL import/export).
//!
//! # Search ranking
//!
//! Team memory has no embedding column, so recall is keyword plus importance plus
//! recency. The formula is fixed (and shared by every store implementation, so an
//! in-memory search and a SQLite search cannot drift):
//!
//! ```text
//! document   = title + " " + content + " " + tags.join(" ")
//! keyword    = normalized_bm25_scores(query, documents)  // 0 when no overlap
//! recency    = 0.5 ^ (age_secs(updated_at) / 30 days)
//! ranking    = RankingConfig { similarity_weight: 0.0, ..Default::default() }  // no embeddings
//! score      = blend(0.0, keyword, importance, recency, &ranking)
//! ```
//!
//! With `similarity_weight = 0.0` the semantic group collapses to the keyword term, so
//! the score is `(0.85 * keyword + 0.10 * importance + 0.05 * recency) / 1.0`. A
//! candidate whose normalized keyword score is zero shares no term with the query and
//! is dropped before ranking, so a search never returns an unrelated record just
//! because it is recent or important.
use super::model::{
    TeamMemoryFilter, TeamMemoryKind, TeamMemoryRecord, TeamMemoryScope, TeamMemoryStats,
    TeamSearchHit,
};
use crate::config::RankingConfig;
use crate::recall::{blend, normalized_bm25_scores, recency_score};
use crate::record::CallOptions;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use openjiuwen_common::error::{Error, Result};
use tokio_util::sync::CancellationToken;
pub mod facade;
pub mod memory;
pub use facade::*;
pub use memory::*;
/// Default number of hits [`TeamMemoryStore::search`] returns.
pub const DEFAULT_SEARCH_TOP_K: usize = 5;
/// Default minimum blended score a [`TeamMemoryStore::search`] hit must reach.
pub const DEFAULT_SEARCH_THRESHOLD: f32 = 0.3;
/// Half-life of the recency term, in seconds (30 days).
pub const SEARCH_RECENCY_HALF_LIFE_SECS: u64 = 30 * 24 * 60 * 60;
/// Options for one team memory search.
///
/// Not `serde`-serializable because [`CallOptions`] carries a cancellation token; build
/// it with the builders and hand it to [`TeamMemoryStore::search`]. The default is
/// [`DEFAULT_SEARCH_TOP_K`], [`DEFAULT_SEARCH_THRESHOLD`], no kind or tag narrowing, and
/// expired records hidden.
#[derive(Debug, Clone)]
pub struct TeamSearchOptions {
    /// Maximum number of hits returned; must be greater than zero.
    pub top_k: usize,
    /// Minimum blended score in `0.0..=1.0`; must be finite.
    pub threshold: f32,
    /// Accepted kinds; empty means every kind.
    pub kinds: Vec<TeamMemoryKind>,
    /// Tags that must all be present (case-insensitive); empty means no constraint.
    pub tags: Vec<String>,
    /// Whether expired records may be returned; `false` hides them.
    pub include_expired: bool,
    /// Cancellation and deadline for the call.
    pub call: CallOptions,
}
impl Default for TeamSearchOptions {
    /// Returns the documented defaults (`top_k = 5`, `threshold = 0.3`, nothing
    /// narrowed, expired records hidden, no cancellation).
    fn default() -> Self {
        Self {
            top_k: DEFAULT_SEARCH_TOP_K,
            threshold: DEFAULT_SEARCH_THRESHOLD,
            kinds: Vec::new(),
            tags: Vec::new(),
            include_expired: false,
            call: CallOptions::new(),
        }
    }
}
impl TeamSearchOptions {
    /// Builds the default options.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets the maximum number of hits.
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }
    /// Sets the minimum blended score.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }
    /// Restricts the search to the given kinds (an empty list means every kind).
    #[must_use]
    pub fn with_kinds<I>(mut self, kinds: I) -> Self
    where
        I: IntoIterator<Item = TeamMemoryKind>,
    {
        self.kinds = kinds.into_iter().collect();
        self
    }
    /// Requires every given tag, matched case-insensitively.
    ///
    /// Entries are trimmed and blank ones dropped; matching folds case, so
    /// `" Storage "` matches a stored `storage` tag.
    #[must_use]
    pub fn with_tags<I, S>(mut self, tags: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.tags = tags
            .into_iter()
            .map(Into::into)
            .map(|tag| tag.trim().to_owned())
            .filter(|tag| !tag.is_empty())
            .collect();
        self
    }
    /// Includes expired records in the results.
    #[must_use]
    pub fn include_expired(mut self) -> Self {
        self.include_expired = true;
        self
    }
    /// Binds the search to a cancellation token.
    #[must_use]
    pub fn with_cancellation(mut self, cancellation: CancellationToken) -> Self {
        self.call = self.call.with_cancellation(cancellation);
        self
    }
    /// Validates the options: `top_k` must be greater than zero and `threshold` must be
    /// finite and inside `0.0..=1.0`.
    pub fn validate(&self) -> Result<()> {
        if self.top_k == 0 {
            return Err(Error::validation(
                "team memory search `top_k` must be greater than zero",
            ));
        }
        if !self.threshold.is_finite() || !(0.0..=1.0).contains(&self.threshold) {
            return Err(Error::validation(format!(
                "team memory search `threshold` must be finite and within 0.0..=1.0 (got {})",
                self.threshold
            )));
        }
        Ok(())
    }
}
/// Persistence seam for team memory records.
///
/// Implementations own the rows of every team; a caller cannot address another team's
/// record because `team_id` is part of every lookup. The contract of each method is
/// documented on the method; the shared rules are:
///
/// - Errors are [`openjiuwen_common::error::Error`]; invalid caller input is
///   [`Error::Validation`], and a missing record is never an error (it is `None` or
///   `false`).
/// - Implementations are `Send + Sync` and object-safe so a facade can hold
///   `Arc<dyn TeamMemoryStore>`.
/// - Cancellation is the caller's concern: the facade wraps each call in
///   [`CallOptions::run`], and a store must leave previously committed rows intact when
///   a call is dropped mid-flight.
#[async_trait]
pub trait TeamMemoryStore: Send + Sync {
    /// Inserts or replaces the row with `record.id`.
    ///
    /// The record is validated first ([`Error::Validation`] on failure). An existing
    /// row with the same id is overwritten but keeps its original `created_at`;
    /// `updated_at` always comes from the input. Returns the stored record (which is the
    /// input, apart from the preserved `created_at`).
    async fn upsert(&self, record: &TeamMemoryRecord) -> Result<TeamMemoryRecord>;
    /// Looks one record up by team and id.
    ///
    /// Returns `None` for an unknown id or a record owned by another team. Expired
    /// records are returned: expiry filtering is a read-view concern (see
    /// [`TeamMemoryFilter::include_expired`]), so `get` can be used to inspect a record
    /// that is about to be pruned.
    async fn get(&self, team_id: &str, id: &str) -> Result<Option<TeamMemoryRecord>>;
    /// Lists the records matching `filter`.
    ///
    /// The filter is validated ([`Error::Validation`] on failure) and matched with
    /// `now = Utc::now()`, so expired records are hidden unless
    /// [`TeamMemoryFilter::include_expired`] is set. Results are ordered by
    /// `updated_at` descending, then `id` ascending, and then paginated with the
    /// filter's `offset`/`limit`.
    async fn list(&self, filter: &TeamMemoryFilter) -> Result<Vec<TeamMemoryRecord>>;
    /// Scores the records `scope` may read against `query`.
    ///
    /// The scope and the options are validated ([`Error::Validation`] on failure).
    /// Candidates are the records for which [`TeamMemoryScope::allows`] returns `true`,
    /// narrowed by `options.kinds`/`options.tags` and by expiry (`options.include_expired`
    /// may keep expired rows). Candidates sharing no term with the query are dropped, the
    /// survivors are scored with the module ranking formula, and the best `top_k` are
    /// returned.
    async fn search(
        &self,
        scope: &TeamMemoryScope,
        query: &str,
        options: &TeamSearchOptions,
    ) -> Result<Vec<TeamSearchHit>>;
    /// Deletes one record.
    ///
    /// Returns `false` when the id is unknown or belongs to another team; deletion is
    /// idempotent and never an error.
    async fn delete(&self, team_id: &str, id: &str) -> Result<bool>;
    /// Deletes every record matching `filter` and returns how many were removed.
    ///
    /// The filter is validated ([`Error::Validation`] on failure) but its
    /// `offset`/`limit` do not apply: `clear` removes the whole matching set.
    async fn clear(&self, filter: &TeamMemoryFilter) -> Result<usize>;
    /// Aggregates the records matching `filter`.
    ///
    /// The filter is validated ([`Error::Validation`] on failure) and matched with
    /// `now = Utc::now()`; expired rows are counted only when
    /// [`TeamMemoryFilter::include_expired`] is set, so `total` is the size of the
    /// returned view. Team-wide records are bucketed under
    /// [`TEAM_WIDE_MEMBER_KEY`](super::model::TEAM_WIDE_MEMBER_KEY).
    async fn stats(&self, filter: &TeamMemoryFilter) -> Result<TeamMemoryStats>;
    /// Sets or clears the expiry of one record without touching `updated_at`.
    ///
    /// Returns `false` when the record is absent or belongs to another team.
    async fn set_expiry(
        &self,
        team_id: &str,
        id: &str,
        expires_at: Option<DateTime<Utc>>,
    ) -> Result<bool>;
    /// Deletes every record whose `expires_at` is at or before `now`, across all teams.
    ///
    /// Returns how many rows were removed. Records without an expiry are untouched.
    async fn prune_expired(&self, now: DateTime<Utc>) -> Result<usize>;
}
/// Ranks `records` against `query`.
///
/// Shared by every [`TeamMemoryStore`] implementation so the search semantics are
/// identical everywhere. Filters out expired records (unless
/// [`TeamSearchOptions::include_expired`]), the option kinds, and records missing one of
/// the option tags; then scores the survivors with the module ranking formula, keeps only
/// candidates with `keyword > 0.0` and `score >= options.threshold`, sorts by `score`
/// descending, `updated_at` descending, and `id` ascending, and truncates to
/// `options.top_k`. `now` is passed in so tests (and future batch callers) are
/// deterministic.
pub(crate) fn rank_candidates(
    records: Vec<TeamMemoryRecord>,
    query: &str,
    options: &TeamSearchOptions,
    now: DateTime<Utc>,
) -> Vec<TeamSearchHit> {
    let candidates: Vec<TeamMemoryRecord> = records
        .into_iter()
        .filter(|record| options.include_expired || !record.is_expired(now))
        .filter(|record| options.kinds.is_empty() || options.kinds.contains(&record.kind))
        .filter(|record| has_all_tags(&record.tags, &options.tags))
        .collect();
    if candidates.is_empty() {
        return Vec::new();
    }
    let documents: Vec<String> = candidates.iter().map(document_text).collect();
    let document_refs: Vec<&str> = documents.iter().map(String::as_str).collect();
    let keywords = normalized_bm25_scores(query, &document_refs);
    let ranking = RankingConfig {
        similarity_weight: 0.0,
        ..RankingConfig::default()
    };
    let mut hits: Vec<TeamSearchHit> = Vec::with_capacity(candidates.len());
    for (record, keyword) in candidates.into_iter().zip(keywords) {
        if keyword <= 0.0 {
            continue;
        }
        let importance = record.importance;
        let recency = recency_score(record.updated_at, now, SEARCH_RECENCY_HALF_LIFE_SECS);
        let score = blend(0.0, keyword, importance, recency, &ranking);
        if score < options.threshold {
            continue;
        }
        hits.push(TeamSearchHit {
            record,
            score,
            keyword,
            importance,
            recency,
        });
    }
    hits.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| right.record.updated_at.cmp(&left.record.updated_at))
            .then_with(|| left.record.id.cmp(&right.record.id))
    });
    hits.truncate(options.top_k);
    hits
}
/// The searchable text of one record: title, content, and tags.
fn document_text(record: &TeamMemoryRecord) -> String {
    format!(
        "{} {} {}",
        record.title.as_deref().unwrap_or_default(),
        record.content,
        record.tags.join(" ")
    )
}
/// Returns `true` when `tags` contains every entry of `wanted`, ignoring case.
fn has_all_tags(tags: &[String], wanted: &[String]) -> bool {
    wanted.iter().all(|needle| {
        let needle = needle.to_lowercase();
        tags.iter().any(|tag| tag.to_lowercase() == needle)
    })
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;
    use crate::team::model::{TeamMemoryScope, TeamVisibility};
    fn at(seconds: i64) -> DateTime<Utc> {
        DateTime::from_timestamp(seconds, 0).unwrap()
    }
    fn record(id: &str, content: &str, updated: i64, importance: f32) -> TeamMemoryRecord {
        let mut record = TeamMemoryRecord::new(
            &TeamMemoryScope::member("review-team", "alice"),
            content,
            TeamMemoryKind::Decision,
        )
        .with_id(id)
        .with_importance(importance);
        record.created_at = at(updated);
        record.updated_at = at(updated);
        record
    }
    #[test]
    fn defaults_and_builders() {
        let options = TeamSearchOptions::default();
        assert_eq!(options.top_k, DEFAULT_SEARCH_TOP_K);
        assert!((options.threshold - DEFAULT_SEARCH_THRESHOLD).abs() < f32::EPSILON);
        assert!(options.kinds.is_empty());
        assert!(options.tags.is_empty());
        assert!(!options.include_expired);
        assert!(options.validate().is_ok());
        let options = TeamSearchOptions::new()
            .with_top_k(2)
            .with_threshold(0.5)
            .with_kinds([TeamMemoryKind::Task])
            .with_tags([" Storage ", "", "sqlite"])
            .include_expired();
        assert_eq!(options.top_k, 2);
        assert!((options.threshold - 0.5).abs() < f32::EPSILON);
        assert_eq!(options.kinds, vec![TeamMemoryKind::Task]);
        assert_eq!(
            options.tags,
            vec!["Storage".to_owned(), "sqlite".to_owned()]
        );
        assert!(options.include_expired);
    }
    #[test]
    fn validate_rejects_zero_top_k_and_bad_thresholds() {
        let err = TeamSearchOptions::default()
            .with_top_k(0)
            .validate()
            .unwrap_err()
            .to_string();
        assert!(err.contains("top_k"), "{err}");
        for threshold in [-0.1, 1.1, f32::NAN, f32::INFINITY] {
            let err = TeamSearchOptions::default()
                .with_threshold(threshold)
                .validate()
                .unwrap_err()
                .to_string();
            assert!(err.contains("threshold"), "{err}");
        }
        assert!(TeamSearchOptions::default()
            .with_threshold(0.0)
            .validate()
            .is_ok());
        assert!(TeamSearchOptions::default()
            .with_threshold(1.0)
            .validate()
            .is_ok());
    }
    #[test]
    fn rank_orders_by_keyword_with_identical_timestamps() {
        let now = at(1_700_000_000);
        let records = vec![
            record(
                "id-a",
                "unrelated prose about gardening",
                now.timestamp(),
                0.5,
            ),
            record(
                "id-b",
                "sqlite storage decision for the shared store",
                now.timestamp(),
                0.5,
            ),
            record(
                "id-c",
                "sqlite sqlite sqlite storage storage decision",
                now.timestamp(),
                0.5,
            ),
        ];
        let hits = rank_candidates(
            records,
            "sqlite storage decision",
            &TeamSearchOptions::default(),
            now,
        );
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].record.id, "id-c");
        assert_eq!(hits[1].record.id, "id-b");
        assert!(hits[0].score >= hits[1].score);
        assert_eq!(hits[0].keyword, 1.0);
        assert_eq!(hits[0].importance, 0.5);
        assert!((hits[0].recency - 1.0).abs() < f32::EPSILON);
    }
    #[test]
    fn rank_excludes_zero_keyword_candidates() {
        let now = at(1_700_000_000);
        let records = vec![record("id-a", "gardening notes", now.timestamp(), 1.0)];
        let hits = rank_candidates(records, "sqlite", &TeamSearchOptions::default(), now);
        assert!(hits.is_empty());
        let hits = rank_candidates(Vec::new(), "sqlite", &TeamSearchOptions::default(), now);
        assert!(hits.is_empty());
    }
    #[test]
    fn rank_applies_the_threshold() {
        let now = at(1_700_000_000);
        let record = record("id-a", "sqlite storage", now.timestamp(), 0.0);
        let lenient = rank_candidates(
            vec![record.clone()],
            "sqlite storage",
            &TeamSearchOptions::default(),
            now,
        );
        assert_eq!(lenient.len(), 1);
        // 0.85 * keyword(1.0) + 0.10 * importance(0.0) + 0.05 * recency(1.0).
        assert!(
            (lenient[0].score - 0.90).abs() < 1e-6,
            "{}",
            lenient[0].score
        );
        let strict = TeamSearchOptions::default().with_threshold(0.95);
        assert!(rank_candidates(vec![record], "sqlite storage", &strict, now).is_empty());
    }
    #[test]
    fn rank_truncates_to_top_k() {
        let now = at(1_700_000_000);
        let records: Vec<TeamMemoryRecord> = (0..4)
            .map(|index| {
                record(
                    &format!("id-{index}"),
                    "sqlite storage decision",
                    now.timestamp(),
                    0.5,
                )
            })
            .collect();
        let options = TeamSearchOptions::default().with_top_k(2);
        let hits = rank_candidates(records, "sqlite storage decision", &options, now);
        assert_eq!(hits.len(), 2);
        // Equal scores and timestamps fall back to ascending id order.
        assert_eq!(hits[0].record.id, "id-0");
        assert_eq!(hits[1].record.id, "id-1");
    }
    #[test]
    fn recency_breaks_score_ties() {
        let now = at(1_700_000_000);
        let records = vec![
            record(
                "id-old",
                "sqlite storage decision",
                now.timestamp() - 90 * 86_400,
                0.5,
            ),
            record("id-new", "sqlite storage decision", now.timestamp(), 0.5),
        ];
        let hits = rank_candidates(
            records,
            "sqlite storage decision",
            &TeamSearchOptions::default(),
            now,
        );
        assert_eq!(hits[0].record.id, "id-new");
        assert!(hits[0].recency > hits[1].recency);
    }
    #[test]
    fn rank_filters_kinds_tags_and_expiry() {
        let now = at(1_700_000_000);
        let mut tagged = record("id-tagged", "sqlite storage", now.timestamp(), 0.5);
        tagged.tags = vec!["storage".to_owned()];
        let expired =
            record("id-expired", "sqlite storage", now.timestamp(), 0.5).with_expires_at(now);
        let task = {
            let mut record = record("id-task", "sqlite storage", now.timestamp(), 0.5);
            record.kind = TeamMemoryKind::Task;
            record
        };
        let records = vec![tagged, expired, task];
        let hits = rank_candidates(
            records.clone(),
            "sqlite storage",
            &TeamSearchOptions::default(),
            now,
        );
        assert_eq!(hits.len(), 2);
        let by_tag = TeamSearchOptions::default().with_tags(["STORAGE"]);
        let hits = rank_candidates(records.clone(), "sqlite storage", &by_tag, now);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].record.id, "id-tagged");
        let by_kind = TeamSearchOptions::default().with_kinds([TeamMemoryKind::Task]);
        let hits = rank_candidates(records.clone(), "sqlite storage", &by_kind, now);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].record.id, "id-task");
        let with_expired = TeamSearchOptions::default().include_expired();
        let hits = rank_candidates(records, "sqlite storage", &with_expired, now);
        assert_eq!(hits.len(), 3);
    }
    #[test]
    fn visibility_is_not_ranked_but_tags_and_titles_are_searchable() {
        let now = at(1_700_000_000);
        let mut record = record("id-a", "body text", now.timestamp(), 0.5);
        record.title = Some("Storage choice".to_owned());
        record.visibility = TeamVisibility::Private;
        let hits = rank_candidates(
            vec![record],
            "storage choice",
            &TeamSearchOptions::default(),
            now,
        );
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].record.id, "id-a");
    }
}
