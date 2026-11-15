//! Recall: `search`, the `recall` alias, and the combined memory/graph `search_all`.
//!
//! Ranking is the engine's job; this module only decides whether graph recall runs and
//! keeps memory recall working when it does not. A graph failure never fails a recall:
//! memory degrades to records only, the failure is logged, and it is reported on
//! [`CombinedRecall::graph_error`] so callers can tell "the graph had no match" from
//! "the graph could not be asked".

use openjiuwen_common::error::{ErrorKind, Result};

use crate::graph::{CombinedRecall, GraphRecallHit, GraphSearchOptions};
use crate::record::{MemoryScope, RecallHit, SearchOptions};

use super::{LongTermMemory, TARGET};

impl LongTermMemory {
    /// Searches the records of `scope`, returning the best hits first.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::Validation`] for an invalid scope or
    /// blank query, [`openjiuwen_common::error::Error::Unsupported`] when semantic mode
    /// is requested without an embedding model or when the scope's override document
    /// names an `embedding` this facade has not registered,
    /// [`openjiuwen_common::error::Error::Cancelled`] when the call is cancelled, and the
    /// storage/embedding error otherwise.
    ///
    /// The scope's resolved override embedding (when it names one) embeds the query, so a
    /// recall searches its own scope's vectors with the same embedder that wrote them.
    pub async fn search(
        &self,
        scope: &MemoryScope,
        query: &str,
        options: &SearchOptions,
    ) -> Result<Vec<RecallHit>> {
        let overrides = self.resolve_scope_overrides(scope).await?;
        self.engine
            .search_with_embedding(scope, query, options, overrides.embedding())
            .await
    }

    /// Python `recall`: an alias of [`LongTermMemory::search`].
    ///
    /// # Errors
    ///
    /// See [`LongTermMemory::search`].
    pub async fn recall(
        &self,
        scope: &MemoryScope,
        query: &str,
        options: &SearchOptions,
    ) -> Result<Vec<RecallHit>> {
        self.search(scope, query, options).await
    }

    /// Recalls memory records and, when graph memory is enabled, graph entities.
    ///
    /// The memory hits are returned first and their errors propagate; graph recall is
    /// best-effort, so a graph failure is logged, reported on
    /// [`CombinedRecall::graph_error`], and leaves
    /// [`CombinedRecall::entities`] empty (the rail parity rule: memory degrades, it
    /// never breaks the turn). A partial result is therefore visible instead of looking
    /// like a graph with no matches; check [`CombinedRecall::is_partial`].
    ///
    /// Cancellation is the one exception: a graph recall cancelled through
    /// [`SearchOptions::call`] propagates `Error::Cancelled` rather than being reported
    /// as a degradation.
    ///
    /// # Errors
    ///
    /// See [`LongTermMemory::search`], plus `Error::Cancelled` when the call's token is
    /// triggered during graph recall.
    pub async fn search_all(
        &self,
        scope: &MemoryScope,
        query: &str,
        options: &SearchOptions,
    ) -> Result<CombinedRecall> {
        let memories = self.search(scope, query, options).await?;
        let (entities, graph_error) = self.graph_entities(scope, query, options).await?;
        Ok(CombinedRecall {
            memories,
            entities,
            graph_error,
        })
    }

    /// Runs graph recall when graph memory is wired, reporting a failure without
    /// failing the call.
    ///
    /// Returns the entity hits plus the degradation reason (`None` when graph memory is
    /// not wired, when it has no match, or when it answered). Cancellation is
    /// propagated instead of being reported as a degradation.
    async fn graph_entities(
        &self,
        scope: &MemoryScope,
        query: &str,
        options: &SearchOptions,
    ) -> Result<(Vec<GraphRecallHit>, Option<String>)> {
        let Some(graph) = &self.graph else {
            return Ok((Vec::new(), None));
        };
        let config = self.config();
        let graph_options = GraphSearchOptions {
            top_k: options.top_k,
            threshold: options.threshold,
            max_depth: config.graph.max_depth,
            max_neighbors: config.graph.max_neighbors,
            include_relations: true,
            call: options.call.clone(),
        };
        match graph.search(scope, query, &graph_options).await {
            Ok(entities) => Ok((entities, None)),
            Err(error) if error.kind() == ErrorKind::Cancelled => Err(error),
            Err(error) => {
                let reason = error.to_string();
                tracing::warn!(
                    target: TARGET,
                    %error,
                    "graph recall failed; returning memory hits only"
                );
                Ok((Vec::new(), Some(reason)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::config::MemoryEngineConfig;
    use crate::record::{AddOptions, NewMemory};

    fn scope() -> MemoryScope {
        MemoryScope::new("alice", "ws")
    }

    fn graph_enabled() -> MemoryEngineConfig {
        let mut config = MemoryEngineConfig::default();
        config.graph.enabled = true;
        config
    }

    async fn seeded(config: MemoryEngineConfig) -> std::sync::Arc<LongTermMemory> {
        let ltm = LongTermMemory::in_memory(config).await.expect("facade");
        ltm.add(
            &scope(),
            NewMemory::semantic("alice prefers dark mode"),
            &AddOptions::default(),
        )
        .await
        .expect("add");
        ltm
    }

    #[tokio::test]
    async fn search_all_without_graph_returns_memories_only() {
        let ltm = seeded(MemoryEngineConfig::default()).await;
        assert!(ltm.graph().is_none());
        let combined = ltm
            .search_all(&scope(), "dark mode", &SearchOptions::default())
            .await
            .expect("search all");
        assert_eq!(combined.memories.len(), 1);
        assert_eq!(combined.entities.len(), 0);
        assert_eq!(combined.graph_error, None, "no graph, no graph failure");
        assert!(!combined.is_partial());
    }

    #[tokio::test]
    async fn recall_is_an_alias_of_search() {
        let ltm = seeded(MemoryEngineConfig::default()).await;
        let options = SearchOptions::default().with_threshold(0.0);
        let hits = ltm
            .search(&scope(), "dark mode", &options)
            .await
            .expect("search");
        let recalled = ltm
            .recall(&scope(), "dark mode", &options)
            .await
            .expect("recall");
        assert_eq!(recalled.len(), hits.len());
        assert_eq!(recalled[0].record.id, hits[0].record.id);
        assert_eq!(recalled[0].record.content, hits[0].record.content);
        assert_eq!(recalled.len(), 1);
    }

    #[tokio::test]
    async fn search_all_with_an_empty_graph_returns_memories_only() {
        let ltm = seeded(graph_enabled()).await;
        assert!(ltm.graph().is_some());
        let combined = ltm
            .search_all(&scope(), "dark mode", &SearchOptions::default())
            .await
            .expect("search all");
        assert_eq!(combined.memories.len(), 1);
        assert_eq!(combined.entities.len(), 0);
        assert_eq!(
            combined.graph_error, None,
            "an empty graph has no matches, which is not a failure"
        );
        assert!(!combined.is_partial());
    }

    #[tokio::test]
    async fn search_all_propagates_memory_errors() {
        let ltm = seeded(graph_enabled()).await;
        let error = ltm
            .search_all(&scope(), "   ", &SearchOptions::default())
            .await
            .expect_err("blank query");
        assert!(matches!(
            error,
            openjiuwen_common::error::Error::Validation(_)
        ));
    }

    #[tokio::test]
    async fn search_all_keeps_memories_when_graph_recall_fails() {
        struct FailingEmbedding;

        #[async_trait::async_trait]
        impl openjiuwen_retrieval::embedding::Embedding for FailingEmbedding {
            fn model_name(&self) -> &'static str {
                "failing"
            }

            async fn embed_query(&self, _text: &str) -> openjiuwen_common::error::Result<Vec<f32>> {
                Err(openjiuwen_common::error::Error::retrieval(
                    "embedding backend unavailable",
                ))
            }

            async fn embed_documents(
                &self,
                _texts: &[String],
                _batch_size: Option<usize>,
            ) -> openjiuwen_common::error::Result<Vec<Vec<f32>>> {
                Err(openjiuwen_common::error::Error::retrieval(
                    "embedding backend unavailable",
                ))
            }
        }

        // Graph recall embeds the query, memory recall here is keyword-only, so the
        // graph failure degrades to "memories only" instead of failing the call.
        let ltm = LongTermMemory::builder(graph_enabled())
            .embedding(std::sync::Arc::new(FailingEmbedding))
            .build()
            .expect("facade");
        ltm.add(
            &scope(),
            NewMemory::semantic("alice prefers dark mode"),
            &AddOptions::default().skip_embedding(),
        )
        .await
        .expect("add");
        let options = SearchOptions::default()
            .with_threshold(0.0)
            .with_mode(crate::record::SearchMode::Keyword);
        let combined = ltm
            .search_all(&scope(), "dark mode", &options)
            .await
            .expect("search all");
        assert_eq!(combined.memories.len(), 1);
        assert_eq!(combined.entities.len(), 0);
        let reason = combined
            .graph_error
            .as_deref()
            .expect("the graph failure must be reported");
        assert!(
            reason.contains("embedding backend unavailable"),
            "the reason must carry the provider error: {reason}"
        );
        assert!(combined.is_partial());
    }

    #[tokio::test]
    async fn a_cancelled_graph_recall_propagates_cancelled() {
        struct CancellingEmbedding;

        #[async_trait::async_trait]
        impl openjiuwen_retrieval::embedding::Embedding for CancellingEmbedding {
            fn model_name(&self) -> &'static str {
                "cancelling"
            }

            async fn embed_query(&self, _text: &str) -> openjiuwen_common::error::Result<Vec<f32>> {
                Err(openjiuwen_common::error::Error::cancelled())
            }

            async fn embed_documents(
                &self,
                _texts: &[String],
                _batch_size: Option<usize>,
            ) -> openjiuwen_common::error::Result<Vec<Vec<f32>>> {
                Err(openjiuwen_common::error::Error::cancelled())
            }
        }

        // Only graph recall touches the embedder (memory recall below is keyword-only),
        // so the cancellation surfaces there and must not be reported as a degradation.
        let ltm = LongTermMemory::builder(graph_enabled())
            .embedding(std::sync::Arc::new(CancellingEmbedding))
            .build()
            .expect("facade");
        ltm.add(
            &scope(),
            NewMemory::semantic("alice prefers dark mode"),
            &AddOptions::default().skip_embedding(),
        )
        .await
        .expect("add");
        let options = SearchOptions::default()
            .with_threshold(0.0)
            .with_mode(crate::record::SearchMode::Keyword);
        let error = ltm
            .search_all(&scope(), "dark mode", &options)
            .await
            .expect_err("a cancelled graph recall must not look like a partial result");
        assert!(
            matches!(error, openjiuwen_common::error::Error::Cancelled),
            "got {error:?}"
        );
    }

    #[tokio::test]
    async fn search_all_merges_graph_entities_when_enabled() {
        let ltm = seeded(graph_enabled()).await;
        let scope = MemoryScope::new("alice", "ws");
        let extraction = crate::graph::Extraction {
            entities: vec![crate::graph::ExtractedEntity {
                name: "dark mode preferences".to_owned(),
                entity_type: "Concept".to_owned(),
                importance: None,
            }],
            relations: Vec::new(),
            degraded: None,
        };
        ltm.graph()
            .expect("graph")
            .add_extracted(&scope, &extraction)
            .await
            .expect("graph add");
        let combined = ltm
            .search_all(&scope, "dark mode", &SearchOptions::default())
            .await
            .expect("search all");
        assert_eq!(combined.memories.len(), 1);
        assert_eq!(combined.entities.len(), 1);
        assert_eq!(combined.graph_error, None);
    }
}
