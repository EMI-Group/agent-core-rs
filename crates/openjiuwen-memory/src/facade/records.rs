//! Record CRUD, listing, statistics, and the Python parity aliases.
//!
//! Every method delegates to [`crate::engine::MemoryEngine`]; the facade only builds
//! the scope filters and adapts the positional Python signatures. Ranking, dedupe, and
//! persistence are never reimplemented here.

use openjiuwen_common::error::{Error, Result};

use crate::record::{
    AddOptions, AddOutcome, CallOptions, MemoryFilter, MemoryRecord, MemoryScope, MemoryStats,
    MemoryType, MemoryUpdate, NewMemory, RecallHit, SearchOptions,
};

use super::variables::validate_scope_id;
use super::LongTermMemory;

impl LongTermMemory {
    /// Stores one memory, running duplicate detection according to `options`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid scope or memory,
    /// [`Error::Cancelled`] when the call is cancelled, and the storage/embedding error
    /// when a write fails.
    pub async fn add(
        &self,
        scope: &MemoryScope,
        memory: NewMemory,
        options: &AddOptions,
    ) -> Result<AddOutcome> {
        self.engine.add(scope, memory, options).await
    }

    /// Stores a batch of memories under one scope lock, deduping across the batch.
    ///
    /// # Errors
    ///
    /// See [`LongTermMemory::add`]; the first failing item's error is returned and the
    /// items written before it stay committed.
    pub async fn add_many(
        &self,
        scope: &MemoryScope,
        memories: Vec<NewMemory>,
        options: &AddOptions,
    ) -> Result<AddOutcome> {
        self.engine.add_many(scope, memories, options).await
    }

    /// Reads one record of `scope`, or `None` when it does not exist.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid scope or corrupt stored JSON.
    pub async fn get(&self, scope: &MemoryScope, id: &str) -> Result<Option<MemoryRecord>> {
        self.engine.get(scope, id).await
    }

    /// Applies `patch` to the stored record `id` and returns the updated record.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when `id` does not exist, [`Error::Validation`] for
    /// an invalid scope or patch, and the storage/embedding error when the write fails.
    pub async fn update(
        &self,
        scope: &MemoryScope,
        id: &str,
        patch: MemoryUpdate,
        options: &CallOptions,
    ) -> Result<MemoryRecord> {
        self.engine.update(scope, id, patch, options).await
    }

    /// Deletes one record and its embedding, returning whether it existed.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid scope and the storage error when
    /// the delete fails.
    pub async fn delete(&self, scope: &MemoryScope, id: &str) -> Result<bool> {
        self.engine.delete(scope, id).await
    }

    /// Deletes every record matching `filter`, returning how many were removed.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Cancelled`] when the call is cancelled and the storage error
    /// when a delete fails.
    pub async fn clear(&self, filter: &MemoryFilter, options: &CallOptions) -> Result<usize> {
        self.engine.clear(filter, options).await
    }

    /// Lists the records of `scope`, newest first, restricted to `kinds`.
    ///
    /// An empty `kinds` slice lists every kind.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid scope and the storage error when
    /// the listing fails.
    pub async fn list(
        &self,
        scope: &MemoryScope,
        kinds: &[MemoryType],
    ) -> Result<Vec<MemoryRecord>> {
        scope.validate()?;
        let mut filter = MemoryFilter::for_scope(scope);
        filter.kinds = kinds.to_vec();
        self.engine.list(&filter).await
    }

    /// Computes statistics over the records of `scope`, restricted to `kinds`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid scope and the storage error when
    /// the listing fails.
    pub async fn stats(&self, scope: &MemoryScope, kinds: &[MemoryType]) -> Result<MemoryStats> {
        scope.validate()?;
        let mut filter = MemoryFilter::for_scope(scope);
        filter.kinds = kinds.to_vec();
        self.engine.stats(&filter).await
    }

    /// Python `search_user_mem`: recall from the scope named by `user_id`/`scope_id`.
    ///
    /// # Errors
    ///
    /// See [`LongTermMemory::search`].
    pub async fn search_user_mem(
        &self,
        query: &str,
        num: usize,
        user_id: &str,
        scope_id: &str,
        threshold: f32,
    ) -> Result<Vec<RecallHit>> {
        let scope = MemoryScope::new(user_id, scope_id);
        let options = SearchOptions {
            top_k: num,
            threshold,
            ..SearchOptions::default()
        };
        self.search(&scope, query, &options).await
    }

    /// Python `search_user_history_summary`: recall summaries of the named scope.
    ///
    /// # Errors
    ///
    /// See [`LongTermMemory::search`].
    pub async fn search_user_history_summary(
        &self,
        query: &str,
        num: usize,
        user_id: &str,
        scope_id: &str,
        threshold: f32,
    ) -> Result<Vec<RecallHit>> {
        let scope = MemoryScope::new(user_id, scope_id);
        let options = SearchOptions {
            top_k: num,
            threshold,
            kinds: vec![MemoryType::Summary],
            ..SearchOptions::default()
        };
        self.search(&scope, query, &options).await
    }

    /// Python `update_mem_by_id`: replaces the content of one record, discarding it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] when `id` does not exist and [`Error::Validation`]
    /// for an invalid scope or empty content.
    pub async fn update_mem_by_id(
        &self,
        id: &str,
        content: &str,
        user_id: &str,
        scope_id: &str,
    ) -> Result<()> {
        let scope = MemoryScope::new(user_id, scope_id);
        self.engine
            .update(
                &scope,
                id,
                MemoryUpdate::content(content),
                &CallOptions::default(),
            )
            .await?;
        Ok(())
    }

    /// Python `delete_mem_by_id`: deletes one record and its embedding.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid scope and the storage error when
    /// the delete fails.
    pub async fn delete_mem_by_id(&self, id: &str, user_id: &str, scope_id: &str) -> Result<bool> {
        let scope = MemoryScope::new(user_id, scope_id);
        self.engine.delete(&scope, id).await
    }

    /// Python `delete_mem_by_user_id`: clears every record of the user in the scope.
    ///
    /// Messages and variables are not touched (documented deviation from Python).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid scope and the storage error when a
    /// delete fails.
    pub async fn delete_mem_by_user_id(&self, user_id: &str, scope_id: &str) -> Result<usize> {
        let scope = MemoryScope::new(user_id, scope_id);
        scope.validate()?;
        self.engine
            .clear(&MemoryFilter::for_scope(&scope), &CallOptions::default())
            .await
    }

    /// Python `delete_mem_by_scope`: clears every user's records in `scope_id`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid `scope_id` and the storage error
    /// when a delete fails.
    pub async fn delete_mem_by_scope(&self, scope_id: &str) -> Result<usize> {
        validate_scope_id(scope_id)?;
        let filter = MemoryFilter {
            scope_id: Some(scope_id.to_owned()),
            ..MemoryFilter::default()
        };
        self.engine.clear(&filter, &CallOptions::default()).await
    }

    /// Python `user_mem_total_num`: counts every record of the named scope.
    ///
    /// # Errors
    ///
    /// See [`LongTermMemory::stats`].
    pub async fn user_mem_total_num(&self, user_id: &str, scope_id: &str) -> Result<usize> {
        let scope = MemoryScope::new(user_id, scope_id);
        Ok(self.stats(&scope, &[]).await?.total)
    }

    /// Python `get_user_mem_by_page`: returns one 1-based page of the scoped listing.
    ///
    /// `page_idx` is 1-based; `0` is treated as page 1. `kind` narrows the listing
    /// unless it is [`MemoryType::Unknown`], which selects every kind. A page past the
    /// end returns an empty vector.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `page_size` is zero or the scope is invalid
    /// and the storage error when the listing fails.
    pub async fn get_user_mem_by_page(
        &self,
        user_id: &str,
        scope_id: &str,
        page_size: usize,
        page_idx: usize,
        kind: MemoryType,
    ) -> Result<Vec<MemoryRecord>> {
        if page_size == 0 {
            return Err(Error::validation(
                "memory page size must be greater than zero (got 0)",
            ));
        }
        let kinds: &[MemoryType] = if kind == MemoryType::Unknown {
            &[]
        } else {
            std::slice::from_ref(&kind)
        };
        let scope = MemoryScope::new(user_id, scope_id);
        let records = self.list(&scope, kinds).await?;
        let page = page_idx.max(1);
        let start = (page - 1).saturating_mul(page_size);
        if start >= records.len() {
            return Ok(Vec::new());
        }
        let end = start.saturating_add(page_size).min(records.len());
        Ok(records[start..end].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::config::MemoryEngineConfig;

    async fn facade() -> std::sync::Arc<LongTermMemory> {
        LongTermMemory::in_memory(MemoryEngineConfig::default())
            .await
            .expect("facade")
    }

    fn scope() -> MemoryScope {
        MemoryScope::new("alice", "ws")
    }

    async fn seed(ltm: &LongTermMemory, count: usize, kind: MemoryType) {
        for index in 0..count {
            ltm.add(
                &scope(),
                NewMemory::new(format!("record number {index}"), kind),
                &AddOptions::default().with_dedupe(crate::record::DedupePolicy::Add),
            )
            .await
            .expect("add");
        }
    }

    #[tokio::test]
    async fn crud_round_trip_through_the_facade() {
        let ltm = facade().await;
        let added = ltm
            .add(
                &scope(),
                NewMemory::semantic("alice prefers dark mode"),
                &AddOptions::default(),
            )
            .await
            .expect("add");
        let id = added.added[0].id.clone();
        assert_eq!(
            ltm.get(&scope(), &id)
                .await
                .expect("get")
                .expect("present")
                .content,
            "alice prefers dark mode"
        );
        let updated = ltm
            .update(
                &scope(),
                &id,
                MemoryUpdate::content("alice prefers light mode"),
                &CallOptions::default(),
            )
            .await
            .expect("update");
        assert_eq!(updated.content, "alice prefers light mode");
        assert_eq!(ltm.list(&scope(), &[]).await.expect("list").len(), 1);
        assert_eq!(ltm.stats(&scope(), &[]).await.expect("stats").total, 1);
        assert!(ltm.delete(&scope(), &id).await.expect("delete"));
        assert!(!ltm.delete(&scope(), &id).await.expect("delete again"));
        assert_eq!(
            ltm.clear(&MemoryFilter::for_scope(&scope()), &CallOptions::default())
                .await
                .expect("clear"),
            0
        );
    }

    #[tokio::test]
    async fn listing_and_stats_filter_by_kind() {
        let ltm = facade().await;
        seed(&ltm, 2, MemoryType::SemanticMemory).await;
        seed(&ltm, 1, MemoryType::Summary).await;
        assert_eq!(ltm.list(&scope(), &[]).await.expect("list all").len(), 3);
        assert_eq!(
            ltm.list(&scope(), &[MemoryType::Summary])
                .await
                .expect("list summaries")
                .len(),
            1
        );
        let stats = ltm
            .stats(&scope(), &[MemoryType::SemanticMemory])
            .await
            .expect("stats");
        assert_eq!(stats.total, 2);
        assert_eq!(stats.by_kind.get("semantic_memory"), Some(&2));
    }

    #[tokio::test]
    async fn listing_rejects_an_invalid_scope() {
        let ltm = facade().await;
        let error = ltm
            .list(&MemoryScope::new("alice", "bad/scope"), &[])
            .await
            .expect_err("invalid scope");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
    }

    #[tokio::test]
    async fn search_user_mem_maps_the_positional_signature() {
        let ltm = facade().await;
        ltm.add(
            &scope(),
            NewMemory::semantic("alice prefers dark mode"),
            &AddOptions::default(),
        )
        .await
        .expect("add");
        let hits = ltm
            .search_user_mem("dark mode", 3, "alice", "ws", 0.0)
            .await
            .expect("search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].record.content, "alice prefers dark mode");
        // Another user sees nothing.
        assert_eq!(
            ltm.search_user_mem("dark mode", 3, "bob", "ws", 0.0)
                .await
                .expect("search bob")
                .len(),
            0
        );
    }

    #[tokio::test]
    async fn search_user_history_summary_is_restricted_to_summaries() {
        let ltm = facade().await;
        ltm.add(
            &scope(),
            NewMemory::semantic("summarised weather fact"),
            &AddOptions::default(),
        )
        .await
        .expect("add semantic");
        ltm.add(
            &scope(),
            NewMemory::summary("summary of the weather talk"),
            &AddOptions::default(),
        )
        .await
        .expect("add summary");
        let hits = ltm
            .search_user_history_summary("weather", 5, "alice", "ws", 0.0)
            .await
            .expect("summary search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].record.kind, MemoryType::Summary);
    }

    #[tokio::test]
    async fn update_mem_by_id_replaces_the_content() {
        let ltm = facade().await;
        let added = ltm
            .add(
                &scope(),
                NewMemory::semantic("old content"),
                &AddOptions::default(),
            )
            .await
            .expect("add");
        let id = added.added[0].id.clone();
        ltm.update_mem_by_id(&id, "new content", "alice", "ws")
            .await
            .expect("update");
        let stored = ltm.get(&scope(), &id).await.expect("get").expect("present");
        assert_eq!(stored.content, "new content");
        let error = ltm
            .update_mem_by_id("missing-id", "new content", "alice", "ws")
            .await
            .expect_err("missing record");
        assert!(matches!(error, Error::NotFound(_)), "{error:?}");
    }

    #[tokio::test]
    async fn delete_aliases_clear_the_expected_records() {
        let ltm = facade().await;
        let added = ltm
            .add(
                &scope(),
                NewMemory::semantic("first record"),
                &AddOptions::default(),
            )
            .await
            .expect("add");
        let id = added.added[0].id.clone();
        assert!(ltm
            .delete_mem_by_id(&id, "alice", "ws")
            .await
            .expect("delete by id"));
        assert!(!ltm
            .delete_mem_by_id(&id, "alice", "ws")
            .await
            .expect("delete by id again"));

        seed(&ltm, 2, MemoryType::SemanticMemory).await;
        assert_eq!(
            ltm.user_mem_total_num("alice", "ws").await.expect("total"),
            2
        );
        assert_eq!(
            ltm.delete_mem_by_user_id("alice", "ws")
                .await
                .expect("clear user"),
            2
        );
        assert_eq!(
            ltm.user_mem_total_num("alice", "ws").await.expect("total"),
            0
        );

        let other = MemoryScope::new("bob", "ws");
        ltm.add(
            &other,
            NewMemory::semantic("bob record"),
            &AddOptions::default(),
        )
        .await
        .expect("add bob");
        ltm.add(
            &MemoryScope::new("carol", "ws"),
            NewMemory::semantic("carol record"),
            &AddOptions::default(),
        )
        .await
        .expect("add carol");
        assert_eq!(ltm.delete_mem_by_scope("ws").await.expect("clear scope"), 2);
        assert_eq!(ltm.user_mem_total_num("bob", "ws").await.expect("total"), 0);
        let error = ltm
            .delete_mem_by_scope("bad/scope")
            .await
            .expect_err("invalid scope id");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
    }

    #[tokio::test]
    async fn paging_is_one_based_and_out_of_range_pages_are_empty() {
        let ltm = facade().await;
        seed(&ltm, 5, MemoryType::SemanticMemory).await;

        let page = ltm
            .get_user_mem_by_page("alice", "ws", 2, 1, MemoryType::Unknown)
            .await
            .expect("page 1");
        assert_eq!(page.len(), 2);
        let first_ids: Vec<&str> = page.iter().map(|record| record.id.as_str()).collect();

        let page0 = ltm
            .get_user_mem_by_page("alice", "ws", 2, 0, MemoryType::Unknown)
            .await
            .expect("page 0");
        let ids0: Vec<&str> = page0.iter().map(|record| record.id.as_str()).collect();
        assert_eq!(ids0, first_ids);

        let page3 = ltm
            .get_user_mem_by_page("alice", "ws", 2, 3, MemoryType::Unknown)
            .await
            .expect("page 3");
        assert_eq!(page3.len(), 1);

        let page4 = ltm
            .get_user_mem_by_page("alice", "ws", 2, 4, MemoryType::Unknown)
            .await
            .expect("page 4");
        assert_eq!(page4.len(), 0);
        assert_eq!(
            ltm.get_user_mem_by_page("alice", "ws", 5, 99, MemoryType::Unknown)
                .await
                .expect("huge page")
                .len(),
            0
        );

        let error = ltm
            .get_user_mem_by_page("alice", "ws", 0, 1, MemoryType::Unknown)
            .await
            .expect_err("zero page size");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
    }

    #[tokio::test]
    async fn paging_with_a_kind_filters_the_listing() {
        let ltm = facade().await;
        seed(&ltm, 3, MemoryType::SemanticMemory).await;
        seed(&ltm, 2, MemoryType::Summary).await;
        let page = ltm
            .get_user_mem_by_page("alice", "ws", 10, 1, MemoryType::Summary)
            .await
            .expect("summary page");
        assert_eq!(page.len(), 2);
        assert!(page.iter().all(|record| record.kind == MemoryType::Summary));
        let page = ltm
            .get_user_mem_by_page("alice", "ws", 10, 1, MemoryType::Unknown)
            .await
            .expect("all kinds");
        assert_eq!(page.len(), 5);
    }
}
