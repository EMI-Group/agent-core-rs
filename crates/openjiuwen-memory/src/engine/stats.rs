//! Statistics aggregation over a filtered record set.
//!
//! The statistics are a read-only projection: matching records are listed through
//! [`MemoryStorage::list_records`] and aggregated into a [`MemoryStats`] value.
//! Token counts come from the engine's injected [`TokenCounter`], so a harness that
//! counts with a real tokenizer sees the same numbers here as in the context engine.

use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use openjiuwen_common::error::Result;
use openjiuwen_foundation::prompt::tokens::TokenCounter;

use crate::record::{MemoryFilter, MemoryStats};
use crate::store::MemoryStorage;

impl super::MemoryEngine {
    /// Computes aggregate statistics of every record matching `filter`.
    ///
    /// # Errors
    ///
    /// Returns the storage error when the listing fails.
    pub async fn stats(&self, filter: &MemoryFilter) -> Result<MemoryStats> {
        compute_stats(self.storage(), filter, self.token_counter()).await
    }
}

/// Aggregates `filter`'s matching records into [`MemoryStats`].
///
/// Kinds are keyed by [`crate::record::MemoryType::as_str`] so callers can forward the
/// counts without a second mapping; timestamps are the record creation times, and
/// `total_tokens` is the summed token count of every matching content.
pub(crate) async fn compute_stats(
    storage: &MemoryStorage,
    filter: &MemoryFilter,
    counter: &(dyn TokenCounter + Send + Sync),
) -> Result<MemoryStats> {
    let records = storage.list_records(filter).await?;
    let mut by_kind: BTreeMap<String, usize> = BTreeMap::new();
    let mut total_tokens = 0usize;
    let mut oldest: Option<DateTime<Utc>> = None;
    let mut newest: Option<DateTime<Utc>> = None;
    for record in &records {
        *by_kind.entry(record.kind.as_str().to_owned()).or_insert(0) += 1;
        total_tokens = total_tokens.saturating_add(counter.count_text(&record.content));
        oldest = Some(oldest.map_or(record.timestamp, |current| current.min(record.timestamp)));
        newest = Some(newest.map_or(record.timestamp, |current| current.max(record.timestamp)));
    }
    Ok(MemoryStats {
        total: records.len(),
        by_kind,
        oldest,
        newest,
        total_tokens,
    })
}

#[cfg(test)]
mod tests {
    use chrono::{DateTime, Utc};
    use openjiuwen_foundation::prompt::tokens::{ApproxTokenCounter, TokenCounter};

    use super::super::MemoryEngine;
    use crate::config::MemoryEngineConfig;
    use crate::record::{AddOptions, MemoryFilter, MemoryScope, MemoryType, NewMemory};
    use crate::store::MemoryStores;

    fn engine() -> std::sync::Arc<MemoryEngine> {
        MemoryEngine::new(MemoryEngineConfig::default(), MemoryStores::in_memory()).expect("engine")
    }

    fn scope() -> MemoryScope {
        MemoryScope::new("alice", "ws")
    }

    fn ts(offset: i64) -> DateTime<Utc> {
        DateTime::from_timestamp(1_700_000_000 + offset, 0).expect("fixed time")
    }

    #[tokio::test]
    async fn stats_count_kinds_timestamps_and_tokens() {
        let engine = engine();
        let scope = scope();
        for (content, kind, offset) in [
            ("first memory", MemoryType::SemanticMemory, 0),
            ("coding fact", MemoryType::Coding, 10),
            ("profile note", MemoryType::UserProfile, 20),
        ] {
            engine
                .add(
                    &scope,
                    NewMemory::new(content, kind).with_timestamp(ts(offset)),
                    &AddOptions::default(),
                )
                .await
                .expect("add");
        }
        let stats = engine
            .stats(&MemoryFilter::for_scope(&scope))
            .await
            .expect("stats");
        assert_eq!(stats.total, 3);
        assert_eq!(stats.by_kind.get("semantic_memory"), Some(&1));
        assert_eq!(stats.by_kind.get("coding"), Some(&1));
        assert_eq!(stats.by_kind.get("user_profile"), Some(&1));
        assert_eq!(stats.oldest, Some(ts(0)));
        assert_eq!(stats.newest, Some(ts(20)));
        let counter = ApproxTokenCounter::new();
        let expected = counter.count_text("first memory")
            + counter.count_text("coding fact")
            + counter.count_text("profile note");
        assert_eq!(stats.total_tokens, expected);
    }

    #[tokio::test]
    async fn stats_of_an_empty_scope_are_zero() {
        let engine = engine();
        let stats = engine
            .stats(&MemoryFilter::for_scope(&scope()))
            .await
            .expect("stats");
        assert_eq!(stats.total, 0);
        assert_eq!(stats.by_kind.len(), 0);
        assert_eq!(stats.oldest, None);
        assert_eq!(stats.newest, None);
        assert_eq!(stats.total_tokens, 0);
    }

    #[tokio::test]
    async fn stats_respect_the_filter() {
        let engine = engine();
        let scope = scope();
        engine
            .add(
                &scope,
                NewMemory::semantic("kept").with_importance(0.9),
                &AddOptions::default(),
            )
            .await
            .expect("add");
        engine
            .add(
                &scope,
                NewMemory::semantic("dropped").with_importance(0.1),
                &AddOptions::default(),
            )
            .await
            .expect("add");
        let mut filter = MemoryFilter::for_scope(&scope);
        filter.min_importance = Some(0.5);
        let stats = engine.stats(&filter).await.expect("stats");
        assert_eq!(stats.total, 1);
        assert_eq!(stats.by_kind.get("semantic_memory"), Some(&1));
    }
}
