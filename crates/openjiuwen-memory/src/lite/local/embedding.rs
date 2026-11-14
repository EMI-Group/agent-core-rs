//! The embedding index of the local store: the optional model handle, the chunked vector
//! index over the fact bodies, and the refresh hooks the mutation path calls.
//!
//! # What is indexed
//!
//! One document per fact, keyed by the fact's name, with the fact body ([`CodingFact::content`])
//! as its content, so a chunk hit maps straight back to the fact it came from
//! (`ScoredChunk::document`) and a fact's semantic score is the maximum similarity over its
//! own chunks. The body is the only indexed text: names, descriptions, and tags stay the
//! keyword signal's business, which keeps the two signals independent.
//!
//! # Consistency
//!
//! The index is derived state — it can always be rebuilt from `MEMORY.md` — and it is
//! rebuilt from the current facts by [`LocalCodingMemory::refresh_embeddings`], which every
//! mutation calls after the Markdown file and the FTS index are updated. Chunk vectors are
//! read from the content-hash cache, so a refresh only embeds the chunks whose text changed.
//! A fact that is edited gets new chunk text and is re-embedded; a deleted fact disappears
//! because the rebuild drops documents that are no longer in the store.
//!
//! Nothing here holds the document guard across a model call: the facts are snapshotted
//! first, then the embedding index (under its own lock) is rebuilt, so a slow model never
//! blocks a concurrent read of the Markdown document.
use std::collections::BTreeMap;
use std::sync::Arc;

use openjiuwen_retrieval::embedding::Embedding;

use openjiuwen_common::error::Result;

use crate::lite::embed::{ChunkingConfig, EmbeddingIndex, EmbeddingStats};
use crate::lite::fact::CodingFact;
use crate::lite::markdown::MarkdownMemory;

use super::state::LocalCodingMemory;

impl LocalCodingMemory {
    /// The embedding model backing semantic recall, or `None` for a keyword-only store.
    #[must_use]
    pub fn embedding(&self) -> Option<&Arc<dyn Embedding>> {
        self.inner.embedding.as_ref()
    }

    /// Whether the store has an embedding model, so semantic and hybrid recall may use it.
    #[must_use]
    pub fn has_embedding(&self) -> bool {
        self.inner.embedding.is_some()
    }

    /// The chunk shape used by the embedding index.
    #[must_use]
    pub fn chunking(&self) -> &ChunkingConfig {
        &self.inner.chunking
    }

    /// The number of chunks in the embedding index (`0` without an embedding model).
    ///
    /// This is the observable size of the derived vector index: one fact usually
    /// contributes one chunk, and a long body contributes several.
    pub async fn embedding_chunks(&self) -> usize {
        self.inner
            .embeddings
            .lock()
            .await
            .as_ref()
            .map_or(0, EmbeddingIndex::len)
    }

    /// Rebuilds the chunk index from the current facts and returns what the pass did.
    ///
    /// Only chunks whose text is missing from the content-hash cache are embedded, so a
    /// refresh after an unchanged document makes no model call. Without an embedding model
    /// there is nothing to index and the returned stats are zeroed.
    ///
    /// # Errors
    ///
    /// Returns the reload, cache, or model error when the index cannot be rebuilt; a
    /// failed pass keeps the previous chunks, so the store stays searchable.
    pub async fn refresh_embeddings(&self) -> Result<EmbeddingStats> {
        let Some(embedding) = self.inner.embedding.clone() else {
            return Ok(EmbeddingStats::default());
        };
        let documents = {
            let mut state = self.inner.state.lock().await;
            self.ensure_fresh(&mut state).await?;
            document_pairs(&state.document)
        };
        let mut guard = self.inner.embeddings.lock().await;
        let index = match guard.as_mut() {
            Some(index) => index,
            None => guard.insert(EmbeddingIndex::new(
                embedding.model_name(),
                self.inner.cache.clone(),
            )),
        };
        index
            .reindex(embedding.as_ref(), &documents, &self.inner.chunking)
            .await
    }

    /// The number of vectors in the content-hash embedding cache.
    ///
    /// The cache is shared by every rebuild and survives a reopen when the store is built
    /// with the `sqlite` feature and an embedding model.
    ///
    /// # Errors
    ///
    /// Returns the cache's storage error when the count cannot be read.
    pub async fn embedding_cache_len(&self) -> Result<usize> {
        self.inner.cache.len().await
    }

    /// The highest similarity of every fact to `query`, keyed by fact name.
    ///
    /// A fact with no indexed chunk (or with a chunk whose vector is unknown) is absent
    /// from the map, which the caller reads as `0.0`. The index is built from the current
    /// facts when it has not been built yet.
    pub(super) async fn semantic_scores(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<BTreeMap<String, f32>> {
        let Some(embedding) = self.inner.embedding.clone() else {
            return Ok(BTreeMap::new());
        };
        let needs_build = { self.inner.embeddings.lock().await.is_none() };
        if needs_build {
            let _stats = self.refresh_embeddings().await?;
        }
        let guard = self.inner.embeddings.lock().await;
        let Some(index) = guard.as_ref() else {
            return Ok(BTreeMap::new());
        };
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

/// The `(document key, content)` pairs of the document: one per fact, keyed by name.
fn document_pairs(document: &MarkdownMemory) -> Vec<(String, String)> {
    document
        .facts()
        .into_iter()
        .map(|fact: CodingFact| (fact.name, fact.content))
        .collect()
}

/// Shared test doubles of the local-store suites (the embedding and recall unit tests).
#[cfg(test)]
pub(super) mod test_support {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use async_trait::async_trait;
    use openjiuwen_retrieval::embedding::{Embedding, HashEmbedding};
    use tempfile::TempDir;

    use openjiuwen_common::error::Result;

    use crate::config::CodingMemoryConfig;
    use crate::lite::fact::CodingFact;
    use crate::record::{MemoryScope, RecallHit};

    use super::LocalCodingMemory;

    /// The vector dimension of the test embedder.
    ///
    /// Deliberately larger than the fake's default: unrelated texts then land near a
    /// cosine of `0.0` (a normalized similarity of about `0.5`), which keeps the
    /// semantic-ranking assertions stable instead of depending on 64-dimension noise.
    pub(crate) const DIMENSION: usize = 256;

    /// A [`HashEmbedding`] that counts model calls, embedded texts, and query calls.
    pub(crate) struct CountingEmbedding {
        /// The real embedder producing deterministic vectors.
        inner: HashEmbedding,
        /// Number of `embed_query` calls.
        query_calls: AtomicUsize,
        /// Number of `embed_documents` calls.
        document_calls: AtomicUsize,
        /// Number of texts passed to `embed_documents`.
        texts: AtomicUsize,
    }

    impl CountingEmbedding {
        /// Creates the counting embedder around a [`DIMENSION`]-sized hash embedder.
        pub(crate) fn new() -> Self {
            Self {
                inner: HashEmbedding::new(DIMENSION),
                query_calls: AtomicUsize::new(0),
                document_calls: AtomicUsize::new(0),
                texts: AtomicUsize::new(0),
            }
        }

        /// The `embed_query` call count.
        pub(crate) fn query_calls(&self) -> usize {
            self.query_calls.load(Ordering::SeqCst)
        }

        /// The `embed_documents` call count.
        pub(crate) fn document_calls(&self) -> usize {
            self.document_calls.load(Ordering::SeqCst)
        }

        /// The number of texts handed to `embed_documents`.
        pub(crate) fn texts(&self) -> usize {
            self.texts.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl Embedding for CountingEmbedding {
        fn model_name(&self) -> &str {
            self.inner.model_name()
        }

        fn dimension_hint(&self) -> Option<usize> {
            self.inner.dimension_hint()
        }

        async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
            self.query_calls.fetch_add(1, Ordering::SeqCst);
            self.inner.embed_query(text).await
        }

        async fn embed_documents(
            &self,
            texts: &[String],
            batch_size: Option<usize>,
        ) -> Result<Vec<Vec<f32>>> {
            self.document_calls.fetch_add(1, Ordering::SeqCst);
            self.texts.fetch_add(texts.len(), Ordering::SeqCst);
            self.inner.embed_documents(texts, batch_size).await
        }
    }

    /// The scope every test store and test fact uses.
    pub(crate) fn scope() -> MemoryScope {
        MemoryScope::new("alice", "ws")
    }

    /// A fact of [`scope`] named `name` with `content` as its body.
    pub(crate) fn fact(name: &str, content: &str) -> CodingFact {
        CodingFact::new(&scope(), name, content)
    }

    /// The fact name a recall hit carries in its record metadata.
    pub(crate) fn hit_name(hit: &RecallHit) -> &str {
        hit.record
            .metadata
            .get("name")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
    }

    /// Opens a keyword-only (no embedding) store at `temp`.
    pub(crate) async fn open_keyword(temp: &TempDir) -> LocalCodingMemory {
        LocalCodingMemory::open_in_scope(temp.path(), CodingMemoryConfig::default(), scope())
            .await
            .expect("the keyword-only store opens")
    }

    /// Opens a store at `temp` configured with `counter` as its embedding model.
    pub(crate) async fn open_embedded(
        temp: &TempDir,
        counter: &Arc<CountingEmbedding>,
    ) -> LocalCodingMemory {
        let shared = Arc::clone(counter);
        let embedding: Arc<dyn Embedding> = shared;
        LocalCodingMemory::builder(temp.path().to_path_buf(), CodingMemoryConfig::default())
            .scope(scope())
            .embedding(embedding)
            .open()
            .await
            .expect("the embedded store opens")
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use std::sync::Arc;

    use tempfile::TempDir;

    use crate::record::{SearchMode, SearchOptions};

    use super::test_support::{fact, hit_name, open_embedded, open_keyword, CountingEmbedding};
    use super::*;

    #[tokio::test]
    async fn open_builds_the_index_from_the_current_facts() {
        let temp = TempDir::new().unwrap();
        let counter = Arc::new(CountingEmbedding::new());
        let store = open_embedded(&temp, &counter).await;
        assert!(store.has_embedding());
        assert!(store.embedding().is_some());
        assert_eq!(store.embedding_chunks().await, 0);
        assert_eq!(counter.document_calls(), 0, "an empty store embeds nothing");

        store.write(fact("build", "cargo build")).await.unwrap();
        store.write(fact("test", "cargo test")).await.unwrap();
        assert_eq!(store.embedding_chunks().await, 2);
        assert_eq!(store.embedding_cache_len().await.unwrap(), 2);
        assert_eq!(
            counter.document_calls(),
            2,
            "each write embeds only the new fact"
        );
        assert_eq!(store.chunking(), &ChunkingConfig::default());
    }

    #[tokio::test]
    async fn a_refresh_after_an_unchanged_document_serves_every_chunk_from_the_cache() {
        let temp = TempDir::new().unwrap();
        let counter = Arc::new(CountingEmbedding::new());
        let store = open_embedded(&temp, &counter).await;
        store.write(fact("build", "cargo build")).await.unwrap();
        store
            .write(fact("test", "cargo test --workspace"))
            .await
            .unwrap();
        let chunks = store.embedding_chunks().await;
        assert_eq!(chunks, 2);
        let calls = counter.document_calls();
        let texts = counter.texts();

        let stats = store.refresh_embeddings().await.unwrap();
        assert_eq!(stats.chunks, chunks);
        assert_eq!(stats.embedded, 0, "a cache hit must not embed anything");
        assert_eq!(stats.cached, chunks);
        assert_eq!(stats.dimension, Some(256));
        assert_eq!(counter.document_calls(), calls);
        assert_eq!(counter.texts(), texts);
        assert_eq!(store.embedding_cache_len().await.unwrap(), chunks);
    }

    #[tokio::test]
    async fn edit_and_delete_keep_the_chunk_index_consistent() {
        let temp = TempDir::new().unwrap();
        let counter = Arc::new(CountingEmbedding::new());
        let store = open_embedded(&temp, &counter).await;
        store.write(fact("build", "cargo build")).await.unwrap();
        store.write(fact("test", "cargo test")).await.unwrap();
        assert_eq!(store.embedding_chunks().await, 2);

        assert!(store.delete("test").await.unwrap());
        assert_eq!(
            store.embedding_chunks().await,
            1,
            "the deleted fact is dropped"
        );
        let options = SearchOptions::default().with_mode(SearchMode::Semantic);
        let hits = store.search("cargo test", &options).await.unwrap();
        assert!(
            hits.iter().all(|hit| hit_name(hit) != "test"),
            "a deleted fact must not be recalled"
        );

        store.edit("build", "cargo build --release").await.unwrap();
        let hits = store
            .search("cargo build --release", &options)
            .await
            .unwrap();
        let hit = hits
            .iter()
            .find(|hit| hit_name(hit) == "build")
            .expect("the edited fact is recalled");
        assert!(
            hit.similarity > 0.99,
            "the edited body must be re-embedded, got {}",
            hit.similarity
        );
    }

    #[tokio::test]
    async fn a_keyword_only_store_reports_no_embedding_state() {
        let temp = TempDir::new().unwrap();
        let store = open_keyword(&temp).await;
        assert!(!store.has_embedding());
        assert!(store.embedding().is_none());
        assert_eq!(store.embedding_chunks().await, 0);
        assert_eq!(
            store.refresh_embeddings().await.unwrap(),
            EmbeddingStats::default()
        );
        assert_eq!(store.embedding_cache_len().await.unwrap(), 0);
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn a_reopen_serves_every_chunk_from_the_persisted_cache() {
        let temp = TempDir::new().unwrap();
        let first = Arc::new(CountingEmbedding::new());
        let store = open_embedded(&temp, &first).await;
        store.write(fact("build", "cargo build")).await.unwrap();
        store.write(fact("test", "cargo test")).await.unwrap();
        assert_eq!(store.embedding_chunks().await, 2);
        assert_eq!(first.texts(), 2);
        drop(store);

        let second = Arc::new(CountingEmbedding::new());
        let reopened = open_embedded(&temp, &second).await;
        assert_eq!(reopened.embedding_chunks().await, 2);
        assert_eq!(
            second.document_calls(),
            0,
            "the persisted cache feeds the reopened index"
        );
        assert_eq!(second.texts(), 0);
        let stats = reopened.refresh_embeddings().await.unwrap();
        assert_eq!(stats.embedded, 0);
        assert_eq!(stats.cached, 2);
    }
}
