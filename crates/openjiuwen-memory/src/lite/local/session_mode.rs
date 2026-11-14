//! JSONL session recall for the local coding-memory store: appending session logs,
//! reading every logged record, and scoring those records against a query.
//!
//! Ports the "session index" half of Python `core/memory/lite/`: the logs themselves live
//! under `sessions/` (see [`crate::lite::session`]), and this module makes them recallable
//! next to the coding facts without copying them into `MEMORY.md`.
//!
//! # Layout
//!
//! ```text
//! <root>/sessions/YYYY-MM-DD/<session-id>.jsonl
//! ```
//!
//! [`LocalCodingMemory::sessions_dir`] reports `<root>/sessions`;
//! [`LocalCodingMemory::append_session`] delegates the write to
//! [`SessionStore::append`](crate::lite::session::SessionStore::append), so the day
//! directory, the one-JSON-object-per-line format, and the session-id sanitizing are the
//! session module's documented behaviour. [`LocalCodingMemory::session_records`] decodes
//! every `*.jsonl` file under the directory, tolerantly, into
//! [`SessionChunk`](crate::lite::session::SessionChunk)s whose paths are relative to the
//! store root and whose line numbers are 1-based, so a hit can be cited as
//! `sessions/2026-11-14/abc.jsonl#L7`.
//!
//! # Scoring
//!
//! [`LocalCodingMemory::search_sessions`] scores one record per [`SessionRecord::text`]
//! (the `role: content` projection), like the session module's own keyword scorer:
//!
//! - The **keyword** component is
//!   [`keyword_score`](crate::lite::session::keyword_score)`(query, record.text())`: the
//!   covered fraction of the query tokens, in `0.0..=1.0`.
//! - The **semantic** component is the cosine similarity of the record text to the query,
//!   normalized to `0.0..=1.0` the same way the fact index normalizes it
//!   ([`similarity_from_cosine`](crate::lite::embed::similarity_from_cosine)). It is served
//!   by a transient [`EmbeddingIndex`](crate::lite::embed::EmbeddingIndex) built from the
//!   `(path:line, text)` pairs of the decoded logs, reusing the store's own
//!   [`EmbeddingCache`](crate::lite::embed::EmbeddingCache), so an unchanged line is never
//!   embedded twice and a repeated search costs one query embedding.
//! - [`SearchMode::Hybrid`](crate::record::SearchMode::Hybrid) (the default) blends the
//!   two `0.7`/`0.3`
//!   ([`blend`](crate::lite::embed::blend)`(semantic, keyword, 0.7, 0.3)`).
//!   [`SearchMode::Keyword`](crate::record::SearchMode::Keyword) scores the keyword
//!   component alone. [`SearchMode::Semantic`](crate::record::SearchMode::Semantic) scores
//!   the semantic component alone.
//!
//! The session index is keyword-only without an embedding model: `Hybrid` then degrades to
//! the keyword score (the documented keyword-only behaviour of the store) and `Semantic`
//! reports [`Error::Unsupported`](openjiuwen_common::error::Error::Unsupported) naming the
//! fix (`LocalCodingMemory::builder(..).embedding(..)`). `Keyword` mode makes **no** model
//! call, neither for the query nor for the logs.
//!
//! # Filtering and ordering
//!
//! The final score is filtered by
//! [`SearchOptions::threshold`](crate::record::SearchOptions::threshold), the hits are
//! ordered by score descending, then `path` ascending, then `line` ascending, and the
//! result is truncated to [`SearchOptions::top_k`](crate::record::SearchOptions::top_k)
//! (raised to at least one). Session logs have no memory kind of their own, so
//! [`SearchOptions::kinds`](crate::record::SearchOptions::kinds) is read as a coarse gate:
//! an empty list recalls sessions, and a non-empty list recalls sessions only when it
//! contains [`MemoryType::EpisodicMemory`] (a session line is a time-stamped event).
//! `token_budget` is ignored, and the call's cancellation state is honoured.
//!
//! Nothing here panics on user input: a blank query and unchecked options are typed errors,
//! a malformed log line is tolerated by the session decoder, and filesystem failures carry
//! the failing path.
use std::collections::BTreeMap;
use std::path::PathBuf;

use chrono::NaiveDate;

use openjiuwen_common::error::{Error, Result};

use crate::lite::embed::{blend, EmbeddingIndex};
use crate::lite::session::{
    keyword_score, SessionChunk, SessionHit, SessionRecord, SessionStore, SESSIONS_DIR,
};
use crate::record::{MemoryType, SearchMode, SearchOptions};

use super::state::LocalCodingMemory;

/// Weight of the semantic component of a hybrid session score (`0.7`).
const VECTOR_WEIGHT: f32 = 0.7;

/// Weight of the keyword component of a hybrid session score (`0.3`).
const TEXT_WEIGHT: f32 = 0.3;

/// The reason reported for semantic session search without an embedding model.
const NO_EMBEDDING_REASON: &str = "the local coding-memory store has no embedding index; \
     configure one with `LocalCodingMemory::builder(..).embedding(..)` or use \
     keyword/hybrid session search";

impl LocalCodingMemory {
    /// The directory holding the JSONL session logs (`<root>/sessions`).
    ///
    /// The path is reported whether or not the directory exists yet;
    /// [`LocalCodingMemory::append_session`] creates it on demand.
    #[must_use]
    pub fn sessions_dir(&self) -> PathBuf {
        self.inner.root.join(SESSIONS_DIR)
    }

    /// Appends `records` as one JSON line each to `session_id`'s log for `date`.
    ///
    /// The log is `<root>/sessions/<YYYY-MM-DD>/<sanitized id>.jsonl`; the day directory
    /// and the file are created when missing and an existing file is appended to, never
    /// rewritten. Empty (whitespace-only) records are skipped, so a caller may pass a whole
    /// turn list. Returns the log path.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the session id sanitizes to nothing or every
    /// record is empty, and [`Error::External`] carrying the path when the directory or the
    /// file cannot be created, written, or flushed.
    pub async fn append_session(
        &self,
        session_id: &str,
        date: NaiveDate,
        records: &[SessionRecord],
    ) -> Result<PathBuf> {
        SessionStore::new(&self.inner.root)
            .append(session_id, date, records)
            .await
    }

    /// Every decoded record of every session log, in `(path, line)` order.
    ///
    /// The path of a chunk is relative to the store root with `/` separators
    /// (`sessions/2026-11-14/abc.jsonl`), and the line number is 1-based. Decoding is
    /// tolerant: a blank line is skipped and a malformed line becomes a plain-text record,
    /// so one bad append never hides a session. A missing `sessions/` directory is an empty
    /// list, not an error.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] carrying the path when a file cannot be read as UTF-8
    /// text or a directory entry cannot be inspected.
    pub async fn session_records(&self) -> Result<Vec<SessionChunk>> {
        SessionStore::new(&self.inner.root).chunks().await
    }

    /// Recalls the session lines matching `query`.
    ///
    /// See the module docs for the per-mode scoring, the keyword-only degradation, and the
    /// `kinds` gate. Each hit reports the record together with its source location and the
    /// score components (`similarity`, `keyword`, `score`).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for a blank query, the call's cancellation error,
    /// [`Error::Unsupported`] for semantic mode on a store without an embedding model, and
    /// the filesystem or embedding error when a log or the query cannot be read or
    /// embedded.
    pub async fn search_sessions(
        &self,
        query: &str,
        options: &SearchOptions,
    ) -> Result<Vec<SessionHit>> {
        let cleaned = query.trim();
        if cleaned.is_empty() {
            return Err(Error::validation(
                "session search query must not be empty (got \"\")",
            ));
        }
        options.call.check()?;
        let use_semantic = match options.mode {
            SearchMode::Keyword => false,
            SearchMode::Semantic => {
                if !self.has_embedding() {
                    return Err(Error::unsupported(NO_EMBEDDING_REASON));
                }
                true
            }
            SearchMode::Hybrid => self.has_embedding(),
        };
        if !options.kinds.is_empty() && !options.kinds.contains(&MemoryType::EpisodicMemory) {
            return Ok(Vec::new());
        }
        let limit = options.top_k.max(1);
        let chunks = self.session_records().await?;
        let semantic = if use_semantic {
            self.session_semantic_scores(&chunks, cleaned, limit.saturating_mul(4))
                .await?
        } else {
            BTreeMap::new()
        };
        let mut hits: Vec<SessionHit> = chunks
            .iter()
            .filter_map(|chunk| {
                let keyword = keyword_score(cleaned, &chunk.record.text());
                let similarity = semantic.get(&session_key(chunk)).copied().unwrap_or(0.0);
                let score = match (use_semantic, options.mode) {
                    (_, SearchMode::Keyword) | (false, _) => keyword,
                    (true, SearchMode::Semantic) => similarity,
                    (true, _) => blend(similarity, keyword, VECTOR_WEIGHT, TEXT_WEIGHT),
                };
                (score >= options.threshold).then(|| SessionHit {
                    path: chunk.path.clone(),
                    line: chunk.line,
                    score,
                    similarity,
                    keyword,
                    record: chunk.record.clone(),
                })
            })
            .collect();
        hits.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.path.cmp(&right.path))
                .then_with(|| left.line.cmp(&right.line))
        });
        hits.truncate(limit);
        Ok(hits)
    }

    /// Scores every session record against `query`, keyed by its `path:line` location.
    ///
    /// The index is transient: it is built from the decoded logs on every call, but its
    /// vectors come from the store's content-hash cache, so only the chunks whose text is
    /// new are embedded. With no embedding model, or no session line at all, the map is
    /// empty and no model call is made.
    async fn session_semantic_scores(
        &self,
        chunks: &[SessionChunk],
        query: &str,
        limit: usize,
    ) -> Result<BTreeMap<String, f32>> {
        let Some(embedding) = self.inner.embedding.clone() else {
            return Ok(BTreeMap::new());
        };
        let documents: Vec<(String, String)> = chunks
            .iter()
            .map(|chunk| (session_key(chunk), chunk.record.text()))
            .collect();
        if documents.is_empty() {
            return Ok(BTreeMap::new());
        }
        let mut index = EmbeddingIndex::new(embedding.model_name(), self.inner.cache.clone());
        let _stats = index
            .reindex(embedding.as_ref(), &documents, &self.inner.chunking)
            .await?;
        let hits = index.search(embedding.as_ref(), query, limit).await?;
        let mut scores: BTreeMap<String, f32> = BTreeMap::new();
        for hit in hits {
            let best = scores.entry(hit.document).or_insert(0.0);
            if hit.similarity > *best {
                *best = hit.similarity;
            }
        }
        Ok(scores)
    }
}

/// The embedding-index document key of a session line: `<path>:<line>`.
fn session_key(chunk: &SessionChunk) -> String {
    format!("{}:{}", chunk.path, chunk.line)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use std::sync::Arc;

    use chrono::{TimeZone, Utc};
    use tempfile::TempDir;

    use openjiuwen_common::error::Error as MemoryError;

    use super::super::embedding::test_support::{
        open_embedded, open_keyword, scope, CountingEmbedding,
    };
    use super::*;

    /// The test day used by the layout and rollup assertions.
    fn day() -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 11, 14).unwrap()
    }

    /// 09:00 UTC on [`day`].
    fn at() -> chrono::DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 11, 14, 9, 0, 0).unwrap()
    }

    /// Keyword options that keep every matching candidate.
    fn keyword_options() -> SearchOptions {
        SearchOptions::default()
            .with_mode(SearchMode::Keyword)
            .with_threshold(0.1)
            .with_top_k(10)
    }

    #[tokio::test]
    async fn append_writes_a_day_directory_log_and_records_number_its_lines() {
        let temp = TempDir::new().unwrap();
        let store = open_keyword(&temp).await;
        let path = store
            .append_session(
                "abc",
                day(),
                &[
                    SessionRecord::new("alpha").with_timestamp(at()),
                    SessionRecord::new(""),
                    SessionRecord::new("beta").with_role("user"),
                ],
            )
            .await
            .unwrap();
        assert_eq!(
            path,
            temp.path()
                .join("sessions")
                .join("2026-11-14")
                .join("abc.jsonl")
        );
        assert_eq!(store.sessions_dir(), temp.path().join("sessions"));
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            text.lines().count(),
            2,
            "the empty record is skipped: {text}"
        );

        let records = store.session_records().await.unwrap();
        assert_eq!(
            records
                .iter()
                .map(|chunk| (chunk.path.as_str(), chunk.line, chunk.record.text()))
                .collect::<Vec<_>>(),
            vec![
                ("sessions/2026-11-14/abc.jsonl", 1, "alpha".to_owned()),
                ("sessions/2026-11-14/abc.jsonl", 2, "user: beta".to_owned()),
            ]
        );
        assert_eq!(records[0].record.timestamp, Some(at()));
    }

    #[tokio::test]
    async fn a_hand_written_log_with_malformed_and_blank_lines_stays_searchable() {
        let temp = TempDir::new().unwrap();
        let store = open_keyword(&temp).await;
        let directory = store.sessions_dir().join("2026-11-14");
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(
            directory.join("hand.jsonl"),
            "{\"content\":\"deploy the workspace\"}\n\n{\"content\":\"truncated\"\nplain text turn\n",
        )
        .unwrap();

        let records = store.session_records().await.unwrap();
        assert_eq!(
            records
                .iter()
                .map(|chunk| (chunk.line, chunk.record.content.as_str()))
                .collect::<Vec<_>>(),
            vec![
                (1, "deploy the workspace"),
                (3, "{\"content\":\"truncated\""),
                (4, "plain text turn"),
            ]
        );
        let hits = store
            .search_sessions("plain text", &keyword_options())
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].line, 4);
        assert_eq!(hits[0].record.content, "plain text turn");
        assert_eq!(hits[0].path, "sessions/2026-11-14/hand.jsonl");
    }

    #[tokio::test]
    async fn keyword_search_ranks_filters_and_truncates() {
        let temp = TempDir::new().unwrap();
        let store = open_keyword(&temp).await;
        store
            .append_session(
                "one",
                day(),
                &[
                    SessionRecord::new("alpha beta gamma"),
                    SessionRecord::new("alpha only"),
                    SessionRecord::new("unrelated"),
                ],
            )
            .await
            .unwrap();
        store
            .append_session("two", day(), &[SessionRecord::new("alpha beta gamma")])
            .await
            .unwrap();

        let options = keyword_options();
        let hits = store
            .search_sessions("alpha beta gamma", &options)
            .await
            .unwrap();
        assert_eq!(
            hits.iter()
                .map(|hit| (hit.path.as_str(), hit.line))
                .collect::<Vec<_>>(),
            vec![
                ("sessions/2026-11-14/one.jsonl", 1),
                ("sessions/2026-11-14/two.jsonl", 1),
                ("sessions/2026-11-14/one.jsonl", 2),
            ]
        );
        assert_eq!(hits[0].score, 1.0);
        assert_eq!(hits[0].similarity, 0.0, "keyword mode has no semantic term");
        assert_eq!(hits[0].keyword, hits[0].score);
        assert!((hits[2].score - (1.0 / 3.0)).abs() < 1e-6, "{:?}", hits[2]);

        let filtered = store
            .search_sessions("alpha beta", &options.clone().with_threshold(0.9))
            .await
            .unwrap();
        assert_eq!(
            filtered.len(),
            2,
            "only the fully covered lines survive the threshold"
        );
        assert!(filtered.iter().all(|hit| hit.score == 1.0));
        assert_eq!(
            store
                .search_sessions("alpha beta", &options.clone().with_top_k(1))
                .await
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            store
                .search_sessions("nothing-matches", &options)
                .await
                .unwrap(),
            Vec::new()
        );
    }

    #[tokio::test]
    async fn blank_queries_and_semantic_mode_without_an_embedding_are_typed() {
        let temp = TempDir::new().unwrap();
        let store = open_keyword(&temp).await;
        let error = store
            .search_sessions("   ", &keyword_options())
            .await
            .unwrap_err();
        assert!(matches!(error, MemoryError::Validation(_)), "{error:?}");
        let error = store
            .search_sessions(
                "alpha",
                &SearchOptions::default().with_mode(SearchMode::Semantic),
            )
            .await
            .unwrap_err();
        assert!(matches!(error, MemoryError::Unsupported(_)), "{error:?}");
        assert!(
            error.to_string().contains("embedding"),
            "the reason names the fix: {error}"
        );
        assert_eq!(
            store
                .search_sessions(
                    "alpha",
                    &keyword_options().with_kinds(vec![MemoryType::Summary]),
                )
                .await
                .unwrap(),
            Vec::new(),
            "a non-episodic kind filter recalls no session"
        );
        assert_eq!(
            store
                .search_sessions(
                    "alpha",
                    &keyword_options().with_kinds(vec![MemoryType::EpisodicMemory]),
                )
                .await
                .unwrap()
                .len(),
            0,
            "an episodic filter recalls the (empty) session set"
        );
    }

    #[tokio::test]
    async fn semantic_and_hybrid_recall_the_exact_line_first_and_keyword_never_embeds() {
        let temp = TempDir::new().unwrap();
        let counter = Arc::new(CountingEmbedding::new());
        let store = open_embedded(&temp, &counter).await;
        assert_eq!(counter.document_calls(), 0);
        store
            .append_session(
                "one",
                day(),
                &[
                    SessionRecord::new("cargo test --workspace"),
                    SessionRecord::new("the workspace is offline"),
                ],
            )
            .await
            .unwrap();
        assert_eq!(
            counter.document_calls(),
            0,
            "appending a log embeds nothing"
        );

        let keyword = store
            .search_sessions("cargo test --workspace", &keyword_options())
            .await
            .unwrap();
        assert_eq!(keyword[0].line, 1);
        assert_eq!(
            counter.query_calls(),
            0,
            "keyword mode must not embed the query"
        );
        assert_eq!(counter.document_calls(), 0);

        let options = SearchOptions::default().with_threshold(0.0).with_top_k(10);
        for mode in [SearchMode::Semantic, SearchMode::Hybrid] {
            let hits = store
                .search_sessions("cargo test --workspace", &options.clone().with_mode(mode))
                .await
                .unwrap();
            assert_eq!(hits[0].line, 1, "mode {mode:?}");
            assert!(
                hits[0].similarity > 0.99,
                "mode {mode:?} got {}",
                hits[0].similarity
            );
        }
        assert_eq!(
            counter.document_calls(),
            1,
            "the second mode reuses the cached vectors"
        );
        assert_eq!(counter.texts(), 2, "only the two log lines were embedded");
        assert_eq!(counter.query_calls(), 2, "one query embedding per mode");
    }

    #[tokio::test]
    async fn the_scope_of_the_store_does_not_gate_session_recall() {
        let temp = TempDir::new().unwrap();
        let store = open_keyword(&temp).await;
        assert_eq!(store.scope(), &scope());
        assert_eq!(store.session_records().await.unwrap(), Vec::new());
    }
}
