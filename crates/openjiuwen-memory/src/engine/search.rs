//! The read path: candidate generation, ranking, threshold/top-k/token-budget
//! application, and best-effort access tracking.
//!
//! Candidate generation has two sources: per-kind vector search when an embedding
//! model is attached (semantic/hybrid mode), and a keyword scope scan otherwise (or in
//! [`SearchMode::Keyword`]).
//! Every score comes from [`crate::recall::rank`]; this module only chooses candidates,
//! applies the caller's filters, and records access statistics.

use chrono::{DateTime, Utc};
use openjiuwen_common::error::{Error, Result};

use crate::recall::{rank, trim_to_token_budget, ScoredCandidate};
use crate::record::{MemoryFilter, MemoryScope, MemoryType, RecallHit, SearchMode, SearchOptions};

/// The tracing target of engine diagnostics.
const TARGET: &str = "openjiuwen_memory::engine";

impl super::MemoryEngine {
    /// Searches the stored memories of `scope`, returning the best hits first.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid scope, a blank query, `top_k == 0`,
    /// an out-of-range threshold, or a kind list without any searchable kind;
    /// [`Error::Unsupported`] when semantic mode is requested without an embedding
    /// model; [`Error::Cancelled`]/[`Error::Timeout`] when the call is cancelled or
    /// exceeds its deadline; and the storage/embedding error when a read fails.
    pub async fn search(
        &self,
        scope: &MemoryScope,
        query: &str,
        options: &SearchOptions,
    ) -> Result<Vec<RecallHit>> {
        self.recall(scope, query, None, options, true).await
    }

    /// The shared recall pipeline behind [`MemoryEngine::search`] and the dedupe
    /// candidate scan.
    ///
    /// `vector` supplies the query embedding when the caller already computed it;
    /// `touch` controls whether the returned records' access counters are updated (the
    /// internal dedupe scan must not count as a recall).
    pub(crate) async fn recall(
        &self,
        scope: &MemoryScope,
        query: &str,
        vector: Option<&[f32]>,
        options: &SearchOptions,
        touch: bool,
    ) -> Result<Vec<RecallHit>> {
        scope.validate()?;
        if query.trim().is_empty() {
            return Err(Error::validation("memory search query must not be empty"));
        }
        validate_search_options(options)?;
        options.call.check()?;
        if options.mode == SearchMode::Semantic && !self.has_embedding() {
            return Err(Error::unsupported(
                "semantic search requires an embedding model",
            ));
        }

        let kinds = self.search_kinds(&options.kinds)?;
        let candidate_k = self.candidate_limit(options.top_k);
        let now = Utc::now();
        let mut candidates = Vec::new();

        if options.mode != SearchMode::Keyword && self.has_embedding() {
            let query_vector = match vector {
                Some(vector) => vector.to_vec(),
                None => self.embed_text(query, &options.call).await?,
            };
            for kind in &kinds {
                options.call.check()?;
                let hits = self
                    .storage()
                    .search_embeddings(scope, *kind, &query_vector, candidate_k)
                    .await?;
                for (id, score) in hits {
                    let Some(record) = self.storage().get_record(scope, &id).await? else {
                        tracing::warn!(
                            target: TARGET,
                            %id,
                            "skipping a vector hit whose memory record is missing (stale index)"
                        );
                        continue;
                    };
                    if &record.kind != kind || !scope.matches(&record.scope) {
                        tracing::warn!(
                            target: TARGET,
                            %id,
                            "skipping a vector hit that does not belong to the searched scope/kind"
                        );
                        continue;
                    }
                    candidates.push(ScoredCandidate::new(record, Some(score)));
                }
            }
        } else {
            if options.mode == SearchMode::Hybrid {
                tracing::warn!(
                    target: TARGET,
                    "no embedding model is configured; hybrid recall degrades to keyword scoring"
                );
            }
            let mut filter = MemoryFilter::for_scope(scope);
            filter.kinds.clone_from(&kinds);
            let records = self.storage().list_records(&filter).await?;
            for record in records.into_iter().take(self.config.ranking.max_candidates) {
                options.call.check()?;
                candidates.push(ScoredCandidate::keyword_only(record));
            }
        }

        let mut hits = rank(candidates, query, now, &self.config.ranking);
        hits.retain(|hit| hit.score >= options.threshold);
        hits.truncate(options.top_k);
        if let Some(budget) = options.token_budget.or(self.config.token_budget) {
            hits = trim_to_token_budget(hits, budget, self.token_counter());
        }
        if touch {
            self.touch_hits(&hits, now).await;
        }
        Ok(hits)
    }

    /// Resolves the kinds a recall searches: the caller's list, or the engine's
    /// searchable kinds, filtered to searchable kinds and deduplicated in order.
    fn search_kinds(&self, requested: &[MemoryType]) -> Result<Vec<MemoryType>> {
        let source = if requested.is_empty() {
            self.config.searchable_kinds.as_slice()
        } else {
            requested
        };
        let mut kinds = Vec::with_capacity(source.len());
        for kind in source {
            if kind.is_searchable() && !kinds.contains(kind) {
                kinds.push(*kind);
            }
        }
        if kinds.is_empty() {
            return Err(Error::validation(
                "memory search requires at least one searchable kind",
            ));
        }
        Ok(kinds)
    }

    /// The size of the vector candidate pool for `top_k`.
    fn candidate_limit(&self, top_k: usize) -> usize {
        let ranking = &self.config.ranking;
        let scaled = (top_k as f32 * ranking.candidate_multiplier) as usize;
        scaled.max(top_k).min(ranking.max_candidates)
    }

    /// Updates `last_access`/`access_count` of the returned records, best-effort.
    ///
    /// `updated_at` is deliberately untouched: a recall is not a modification, so a
    /// search never changes a record's recency ranking. A failed write is logged and
    /// never fails the search.
    async fn touch_hits(&self, hits: &[RecallHit], now: DateTime<Utc>) {
        for hit in hits {
            let mut record = hit.record.clone();
            record.last_access = Some(now);
            record.access_count = record.access_count.saturating_add(1);
            if let Err(error) = self.storage().put_record(&record).await {
                tracing::warn!(
                    target: TARGET,
                    %error,
                    id = %record.id,
                    "failed to record a memory access; the recall result is unaffected"
                );
            }
        }
    }
}

/// Validates the caller-supplied search knobs.
fn validate_search_options(options: &SearchOptions) -> Result<()> {
    if options.top_k == 0 {
        return Err(Error::validation(
            "memory search `top_k` must be greater than zero (got 0)",
        ));
    }
    if !options.threshold.is_finite() || !(0.0..=1.0).contains(&options.threshold) {
        return Err(Error::validation(format!(
            "memory search `threshold` must be between 0.0 and 1.0 (got {})",
            options.threshold
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use chrono::{DateTime, Utc};
    use openjiuwen_common::error::Error;
    use openjiuwen_foundation::prompt::tokens::{ApproxTokenCounter, TokenCounter};
    use openjiuwen_retrieval::embedding::HashEmbedding;

    use super::super::{MemoryEngine, MemoryEngineBuilder};
    use crate::config::MemoryEngineConfig;
    use crate::record::{
        AddOptions, CallOptions, MemoryFilter, MemoryRecord, MemoryScope, MemoryType, NewMemory,
        SearchMode, SearchOptions,
    };
    use crate::store::{keys, MemoryStores};

    fn engine() -> Arc<MemoryEngine> {
        MemoryEngine::new(MemoryEngineConfig::default(), MemoryStores::in_memory()).expect("engine")
    }

    fn semantic_engine() -> Arc<MemoryEngine> {
        MemoryEngineBuilder::new(MemoryEngineConfig::default())
            .embedding(Arc::new(HashEmbedding::new(32)))
            .build()
            .expect("engine")
    }

    fn scope() -> MemoryScope {
        MemoryScope::new("alice", "ws")
    }

    fn ts(offset: i64) -> DateTime<Utc> {
        DateTime::from_timestamp(1_700_000_000 + offset, 0).expect("fixed time")
    }

    async fn add(engine: &MemoryEngine, scope: &MemoryScope, content: &str) -> String {
        add_kind(engine, scope, content, MemoryType::SemanticMemory).await
    }

    async fn add_kind(
        engine: &MemoryEngine,
        scope: &MemoryScope,
        content: &str,
        kind: MemoryType,
    ) -> String {
        let mut outcome = engine
            .add(scope, NewMemory::new(content, kind), &AddOptions::default())
            .await
            .expect("add");
        outcome.added.remove(0).id
    }

    #[tokio::test]
    async fn semantic_search_orders_the_nearest_memory_first() {
        let engine = semantic_engine();
        let scope = scope();
        add(&engine, &scope, "banana bread recipe with walnuts").await;
        let target = add(&engine, &scope, "tokio async runtime task scheduling").await;
        let hits = engine
            .search(
                &scope,
                "async runtime tokio",
                &SearchOptions::default().with_threshold(0.0).with_top_k(3),
            )
            .await
            .expect("search");
        assert_ne!(hits.len(), 0);
        assert_eq!(hits[0].record.id, target);
        assert!(hits.iter().all(|hit| (0.0..=1.0).contains(&hit.score)));
    }

    #[tokio::test]
    async fn keyword_fallback_orders_by_relevance_without_an_embedding() {
        let engine = engine();
        let scope = scope();
        let bread = add(&engine, &scope, "banana bread recipe").await;
        add(&engine, &scope, "rust ownership and borrowing").await;
        let hits = engine
            .search(
                &scope,
                "banana bread",
                &SearchOptions::default().with_threshold(0.0),
            )
            .await
            .expect("search");
        assert_eq!(hits[0].record.id, bread);
    }

    #[tokio::test]
    async fn semantic_mode_without_an_embedding_is_unsupported() {
        let engine = engine();
        let scope = scope();
        add(&engine, &scope, "any content").await;
        let error = engine
            .search(
                &scope,
                "any",
                &SearchOptions::default().with_mode(SearchMode::Semantic),
            )
            .await
            .expect_err("unsupported");
        assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
    }

    #[tokio::test]
    async fn blank_queries_and_invalid_options_are_rejected() {
        let engine = engine();
        let scope = scope();
        let error = engine
            .search(&scope, "   ", &SearchOptions::default())
            .await
            .expect_err("blank query");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
        let error = engine
            .search(&scope, "query", &SearchOptions::default().with_top_k(0))
            .await
            .expect_err("zero top_k");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
        let error = engine
            .search(
                &scope,
                "query",
                &SearchOptions::default().with_threshold(1.5),
            )
            .await
            .expect_err("bad threshold");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
    }

    #[tokio::test]
    async fn threshold_and_top_k_filter_the_hits() {
        let engine = engine();
        let scope = scope();
        add(&engine, &scope, "alpha content").await;
        add(&engine, &scope, "beta content").await;
        add(&engine, &scope, "gamma content").await;

        let all = engine
            .search(
                &scope,
                "content",
                &SearchOptions::default().with_threshold(0.0),
            )
            .await
            .expect("search");
        assert_eq!(all.len(), 3);

        let top_two = engine
            .search(
                &scope,
                "content",
                &SearchOptions::default().with_threshold(0.0).with_top_k(2),
            )
            .await
            .expect("search");
        assert_eq!(top_two.len(), 2);

        let none = engine
            .search(
                &scope,
                "quantum chromodynamics",
                &SearchOptions::default().with_threshold(0.99),
            )
            .await
            .expect("search");
        assert_eq!(none.len(), 0);
    }

    #[tokio::test]
    async fn token_budget_truncates_the_hits() {
        let engine = engine();
        let scope = scope();
        add(
            &engine,
            &scope,
            "alpha beta gamma delta epsilon matching content",
        )
        .await;
        add(
            &engine,
            &scope,
            "zeta eta theta iota kappa unrelated content",
        )
        .await;
        let hits = engine
            .search(
                &scope,
                "alpha beta gamma delta epsilon",
                &SearchOptions::default().with_threshold(0.0),
            )
            .await
            .expect("search");
        assert_eq!(hits.len(), 2);
        let counter = ApproxTokenCounter::new();
        let budget = counter.count_text(&hits[0].record.content);
        let trimmed = engine
            .search(
                &scope,
                "alpha beta gamma delta epsilon",
                &SearchOptions::default()
                    .with_threshold(0.0)
                    .with_token_budget(budget),
            )
            .await
            .expect("search");
        assert_eq!(trimmed.len(), 1);
        assert_eq!(trimmed[0].record.id, hits[0].record.id);

        let empty = engine
            .search(
                &scope,
                "alpha beta gamma delta epsilon",
                &SearchOptions::default()
                    .with_threshold(0.0)
                    .with_token_budget(0),
            )
            .await
            .expect("search");
        assert_eq!(empty.len(), 0);
    }

    #[tokio::test]
    async fn kind_restriction_and_empty_kinds_use_searchable_kinds() {
        let engine = engine();
        let scope = scope();
        let coding = add_kind(&engine, &scope, "cargo clippy notes", MemoryType::Coding).await;
        let semantic = add(&engine, &scope, "cargo clippy habits from review").await;
        let all = engine
            .search(
                &scope,
                "cargo clippy",
                &SearchOptions::default().with_threshold(0.0),
            )
            .await
            .expect("search");
        assert!(all.iter().any(|hit| hit.record.id == coding));
        assert!(all.iter().any(|hit| hit.record.id == semantic));
        let only_coding = engine
            .search(
                &scope,
                "cargo clippy",
                &SearchOptions::default()
                    .with_threshold(0.0)
                    .with_kinds(vec![MemoryType::Coding]),
            )
            .await
            .expect("search");
        assert_eq!(only_coding.len(), 1);
        assert_eq!(only_coding[0].record.id, coding);
        let error = engine
            .search(
                &scope,
                "cargo clippy",
                &SearchOptions::default().with_kinds(vec![MemoryType::Variable]),
            )
            .await
            .expect_err("no searchable kind");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
    }

    #[tokio::test]
    async fn scopes_do_not_leak_between_users() {
        let engine = engine();
        let alice = MemoryScope::new("alice", "ws");
        let bob = MemoryScope::new("bob", "ws");
        add(&engine, &alice, "alice secret preference").await;
        let hits = engine
            .search(
                &bob,
                "alice secret",
                &SearchOptions::default().with_threshold(0.0),
            )
            .await
            .expect("search");
        assert_eq!(hits.len(), 0);
    }

    #[tokio::test]
    async fn stale_vector_hits_are_skipped() {
        let engine = semantic_engine();
        let scope = scope();
        let ghost = MemoryRecord::new(scope.clone(), "ghost memory", MemoryType::SemanticMemory)
            .with_id("ghost-id");
        let vector = engine
            .embed_text("ghost memory", &CallOptions::new())
            .await
            .expect("embed");
        engine
            .storage()
            .index_embedding(&ghost, &vector)
            .await
            .expect("index ghost");
        let hits = engine
            .search(
                &scope,
                "ghost memory",
                &SearchOptions::default().with_threshold(0.0),
            )
            .await
            .expect("search");
        assert_eq!(hits.len(), 0);
        // The KV record really is absent, only the vector row exists.
        assert!(engine.get(&scope, "ghost-id").await.expect("get").is_none());
    }

    #[tokio::test]
    async fn a_recall_touches_access_without_changing_updated_at() {
        let engine = engine();
        let scope = scope();
        let mut added = engine
            .add(
                &scope,
                NewMemory::semantic("durable access test").with_timestamp(ts(0)),
                &AddOptions::default(),
            )
            .await
            .expect("add");
        let id = added.added.remove(0).id;
        let options = SearchOptions::default().with_threshold(0.0);
        let hits = engine
            .search(&scope, "durable access", &options)
            .await
            .expect("search");
        assert_eq!(hits.len(), 1);
        let stored = engine
            .get(&scope, &id)
            .await
            .expect("get")
            .expect("present");
        assert_eq!(stored.access_count, 1);
        assert_eq!(stored.updated_at, ts(0));
        assert!(stored.last_access.is_some());

        engine
            .search(&scope, "durable access", &options)
            .await
            .expect("search");
        let stored = engine
            .get(&scope, &id)
            .await
            .expect("get")
            .expect("present");
        assert_eq!(stored.access_count, 2);
        assert_eq!(stored.updated_at, ts(0));
    }

    #[tokio::test]
    async fn a_record_without_a_kv_row_is_skipped_even_without_other_hits() {
        let engine = semantic_engine();
        let scope = scope();
        let id = add(&engine, &scope, "vector row without kv row").await;
        engine
            .stores()
            .kv
            .delete(&keys::record_key(&scope, &id))
            .await
            .expect("delete kv only");
        let hits = engine
            .search(
                &scope,
                "vector row without kv row",
                &SearchOptions::default().with_threshold(0.0),
            )
            .await
            .expect("search");
        assert_eq!(hits.len(), 0);
        assert_eq!(
            engine
                .list(&MemoryFilter::for_scope(&scope))
                .await
                .expect("list")
                .len(),
            0
        );
    }
}
