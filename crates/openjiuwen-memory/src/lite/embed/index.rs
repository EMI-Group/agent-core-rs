//! Chunked vector index over Markdown documents, plus the scoring helpers.
//!
//! Ports the per-chunk vector half of Python `core/memory/lite/`: documents are chunked
//! with `chunk_markdown`, every chunk is embedded once, and recall scores chunk vectors
//! against the query vector. The index is derived state — it can be dropped and rebuilt
//! from the Markdown documents at any time — and it keeps no reference to those
//! documents, so `EmbeddingIndex::reindex` always receives the current text.
//!
//! # Identity and incremental work
//!
//! A chunk id is `"<document>:<start_line>:<end_line>"` (for example
//! `"MEMORY.md:1:12"`), so re-chunking unchanged text produces identical ids and a
//! caller can relate a hit back to a citable line range. The originating document key is
//! kept beside the id ([`IndexedChunk::document`], [`ScoredChunk::document`]) so a caller
//! can map a chunk back to the document it came from without parsing the id.
//!
//! `EmbeddingIndex::reindex` rebuilds the chunk list from scratch but embeds only cache
//! misses: each chunk's vector is looked up in the `EmbeddingCache` under
//! `(embedding.model_name(), content_hash(chunk.text))`, and the misses of the whole pass
//! are sent to `Embedding::embed_documents` in ONE batch. Re-indexing unchanged
//! documents therefore makes zero model calls, and editing one document re-embeds only
//! that document's changed chunks. Query vectors are never cached: a query is embedded on
//! demand by `EmbeddingIndex::search` and never reaches the cache.
//!
//! # Scoring
//!
//! Similarity is the cosine similarity between the query vector and the chunk vector,
//! mapped to `0.0..=1.0` by `similarity_from_cosine` — the same normalization the
//! crate's record recall uses. A chunk without a stored vector (or with one of another
//! dimension) scores `0.0` rather than being dropped, so a partially embedded index still
//! answers every query. `blend` combines that semantic score with a text score under
//! per-group renormalized weights:
//!
//! ```text
//! blend = (w_vector * semantic + w_text * keyword) / (w_vector + w_text)
//! ```
//!
//! Both weights zero falls back to `max(semantic, keyword)`, so no combination can
//! produce a `NaN`.
//!
//! `search` orders by similarity descending and then by id ascending (a deterministic
//! tie-break) and truncates to the caller's limit. A blank query is
//! `Error::Validation`, and an empty index answers without a model call.

use std::fmt;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_retrieval::embedding::Embedding;
use serde::{Deserialize, Serialize};

use super::cache::{content_hash, EmbeddingCache};
use super::chunk::{chunk_markdown, ChunkingConfig, MemChunk};
use super::TARGET;

/// One chunk of a document together with its vector.
///
/// `document` is the key of the document the chunk was cut from (the first element of the
/// `(document, content)` pair passed to [`EmbeddingIndex::reindex`]); `id` is
/// `"<document>:<start_line>:<end_line>"`; `vector` is the chunk's embedding, and a chunk
/// without one is kept and scores `0.0` instead of being dropped.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexedChunk {
    /// The chunk identity: document id plus line range.
    pub id: String,
    /// The document key the chunk was cut from.
    pub document: String,
    /// The chunk text and its line range.
    pub chunk: MemChunk,
    /// The chunk's embedding, when one is known.
    pub vector: Option<Vec<f32>>,
}

/// What one [`EmbeddingIndex::reindex`] pass did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct EmbeddingStats {
    /// The number of chunks the index now holds.
    pub chunks: usize,
    /// The number of chunks embedded by this pass (its cache misses).
    pub embedded: usize,
    /// The number of chunks whose vector came from the cache.
    pub cached: usize,
    /// The vector dimension of the index, when one is known.
    pub dimension: Option<usize>,
}

/// One recall hit: a chunk and its normalized similarity to the query.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoredChunk {
    /// The chunk identity (see [`IndexedChunk::id`]).
    pub id: String,
    /// The document key the chunk was cut from (see [`IndexedChunk::document`]).
    pub document: String,
    /// The chunk text and its line range.
    pub chunk: MemChunk,
    /// The `0.0..=1.0` similarity to the query (`0.0` without a comparable vector).
    pub similarity: f32,
}

/// A chunked vector index over Markdown documents.
///
/// Clone it freely: clones share the same [`EmbeddingCache`] but keep their own chunk
/// list, so re-indexing one clone does not disturb another.
#[derive(Clone)]
pub struct EmbeddingIndex {
    /// The embedding model the index was created for.
    model: String,
    /// The vector cache shared by every re-index pass.
    cache: EmbeddingCache,
    /// The current chunks, in document order.
    chunks: Vec<IndexedChunk>,
}

impl fmt::Debug for EmbeddingIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EmbeddingIndex")
            .field("model", &self.model)
            .field("cache", &self.cache)
            .field("chunks", &self.chunks.len())
            .finish()
    }
}

impl EmbeddingIndex {
    /// Creates an empty index for `model` that reuses `cache`.
    #[must_use]
    pub fn new(model: impl Into<String>, cache: EmbeddingCache) -> Self {
        Self {
            model: model.into(),
            cache,
            chunks: Vec::new(),
        }
    }

    /// The embedding model the index was created for.
    ///
    /// Cache entries are keyed by the *embedder's* reported name (which may differ from
    /// this label), so a relabelled index still reuses its vectors.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    /// The number of indexed chunks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// Whether the index holds no chunk.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// The indexed chunks, in document order.
    #[must_use]
    pub fn chunks(&self) -> &[IndexedChunk] {
        &self.chunks
    }

    /// Drops every chunk; cached vectors are kept and may still be reused.
    pub fn clear(&mut self) {
        self.chunks.clear();
    }

    /// Re-chunks every `(document id, content)` pair, embedding only cache misses.
    ///
    /// The chunk list is replaced wholesale, so a document that disappeared from
    /// `documents` disappears from the index. A blank document (empty or
    /// whitespace-only content) is skipped entirely. Chunk vectors are read from the
    /// cache under `(embedding.model_name(), content_hash(chunk.text))`; all misses are
    /// embedded in one [`Embedding::embed_documents`] call whose results are written back
    /// to the cache. The previous chunk list is kept when the model call fails, so a
    /// failed re-index leaves the index usable.
    ///
    /// `config` bounds the chunk shape; it is validated first.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid chunking config, the cache's errors
    /// when a lookup or write fails, [`Error::Memory`] when the model returns a different
    /// number of vectors than it was given, and the model's own error when embedding
    /// fails.
    pub async fn reindex(
        &mut self,
        embedding: &dyn Embedding,
        documents: &[(String, String)],
        config: &ChunkingConfig,
    ) -> Result<EmbeddingStats> {
        config.validate()?;
        let model = embedding.model_name().to_owned();
        let mut next: Vec<IndexedChunk> = Vec::new();
        let mut misses: Vec<(usize, String)> = Vec::new();
        let mut cached = 0_usize;
        for (document, content) in documents {
            if content.trim().is_empty() {
                tracing::debug!(
                    target: TARGET,
                    document = %document,
                    "skipping a blank document while indexing"
                );
                continue;
            }
            for chunk in chunk_markdown(content, config) {
                let hash = content_hash(&chunk.text);
                let vector = self.cache.get(&model, &hash).await?;
                if vector.is_some() {
                    cached += 1;
                } else {
                    misses.push((next.len(), hash));
                }
                next.push(IndexedChunk {
                    id: format!("{document}:{}:{}", chunk.start_line, chunk.end_line),
                    document: document.clone(),
                    chunk,
                    vector,
                });
            }
        }
        let embedded = misses.len();
        if !misses.is_empty() {
            let texts: Vec<String> = misses
                .iter()
                .map(|(slot, _)| next[*slot].chunk.text.clone())
                .collect();
            let vectors = embedding.embed_documents(&texts, None).await?;
            if vectors.len() != texts.len() {
                return Err(Error::memory(format!(
                    "embedding model `{model}` returned {} vectors for {} chunks",
                    vectors.len(),
                    texts.len()
                )));
            }
            for ((slot, hash), vector) in misses.iter().zip(vectors) {
                self.cache.put(&model, hash, &vector).await?;
                if let Some(indexed) = next.get_mut(*slot) {
                    indexed.vector = Some(vector);
                }
            }
        }
        let dimension = next
            .iter()
            .find_map(|indexed| indexed.vector.as_ref().filter(|vector| !vector.is_empty()))
            .map(Vec::len)
            .or_else(|| {
                embedding
                    .dimension_hint()
                    .filter(|dimension| *dimension > 0)
            });
        let stats = EmbeddingStats {
            chunks: next.len(),
            embedded,
            cached,
            dimension,
        };
        tracing::debug!(
            target: TARGET,
            model = %model,
            chunks = stats.chunks,
            embedded = stats.embedded,
            cached = stats.cached,
            "reindexed the lite embedding index"
        );
        self.chunks = next;
        Ok(stats)
    }

    /// Scores every chunk against `query`, best first.
    ///
    /// The query is embedded with [`Embedding::embed_query`] (never cached). Every chunk
    /// is scored — a chunk without a vector, or with one of another dimension, contributes
    /// `similarity = 0.0` — and the result is sorted by similarity descending, then by id
    /// ascending, and truncated to `limit.max(1)`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for a blank query and the model's own error when the
    /// query cannot be embedded. An empty index returns no hits without calling the model.
    pub async fn search(
        &self,
        embedding: &dyn Embedding,
        query: &str,
        limit: usize,
    ) -> Result<Vec<ScoredChunk>> {
        if query.trim().is_empty() {
            return Err(Error::validation(
                "embedding search: query must not be blank",
            ));
        }
        if self.chunks.is_empty() {
            tracing::debug!(
                target: TARGET,
                "the lite embedding index is empty; answering without a model call"
            );
            return Ok(Vec::new());
        }
        let query_vector = embedding.embed_query(query).await?;
        let mut scored: Vec<ScoredChunk> = self
            .chunks
            .iter()
            .map(|indexed| {
                let similarity = match indexed.vector.as_deref() {
                    Some(vector) if !vector.is_empty() && vector.len() == query_vector.len() => {
                        similarity_from_cosine(cosine_similarity(&query_vector, vector))
                    }
                    _ => 0.0,
                };
                ScoredChunk {
                    id: indexed.id.clone(),
                    document: indexed.document.clone(),
                    chunk: indexed.chunk.clone(),
                    similarity,
                }
            })
            .collect();
        scored.sort_by(|left, right| {
            right
                .similarity
                .total_cmp(&left.similarity)
                .then_with(|| left.id.cmp(&right.id))
        });
        scored.truncate(limit.max(1));
        Ok(scored)
    }
}

/// The cosine similarity of two vectors, in `-1.0..=1.0`.
///
/// Returns `0.0` when the vectors have different lengths (or are empty), when either has
/// a zero norm, or when the result is not finite — a non-finite score would poison the
/// ordering, and a mismatched dimension means the vectors are not comparable at all. The
/// result is clamped to `-1.0..=1.0` against floating-point drift.
#[must_use]
pub fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0_f32;
    let mut left_squares = 0.0_f32;
    let mut right_squares = 0.0_f32;
    for (left, right) in left.iter().zip(right) {
        dot += left * right;
        left_squares += left * left;
        right_squares += right * right;
    }
    if left_squares <= 0.0 || right_squares <= 0.0 {
        return 0.0;
    }
    let cosine = dot / (left_squares.sqrt() * right_squares.sqrt());
    if cosine.is_finite() {
        cosine.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

/// Maps a cosine similarity into the `0.0..=1.0` range: `(cosine + 1) / 2`.
///
/// A non-finite input maps to `0.0`, and the result is clamped so an out-of-range cosine
/// cannot escape the range.
#[must_use]
pub fn similarity_from_cosine(cosine: f32) -> f32 {
    if !cosine.is_finite() {
        return 0.0;
    }
    cosine.midpoint(1.0).clamp(0.0, 1.0)
}

/// Blends a semantic and a text score under renormalized weights.
///
/// `semantic` and `keyword` are clamped to `0.0..=1.0`; negative or non-finite weights
/// count as `0.0`. The weights are renormalized by their sum, so `(0.7, 0.3)` and
/// `(7.0, 3.0)` produce the same blend, and either weight alone yields that signal
/// unchanged. When both weights are zero the larger of the two scores wins, so the
/// function is never a `NaN` and never silently returns zero.
#[must_use]
pub fn blend(semantic: f32, keyword: f32, vector_weight: f32, text_weight: f32) -> f32 {
    let semantic = unit_score(semantic);
    let keyword = unit_score(keyword);
    let vector_weight = non_negative_weight(vector_weight);
    let text_weight = non_negative_weight(text_weight);
    let total = vector_weight + text_weight;
    if total <= 0.0 {
        return semantic.max(keyword);
    }
    ((vector_weight * semantic + text_weight * keyword) / total).clamp(0.0, 1.0)
}

/// Clamps a score to `0.0..=1.0`, mapping a non-finite value to `0.0`.
fn unit_score(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Maps a weight to a finite non-negative value.
fn non_negative_weight(value: f32) -> f32 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Mutex;

    use async_trait::async_trait;
    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_retrieval::embedding::HashEmbedding;

    use super::*;

    /// The vector dimension used by the test embedder.
    const DIMENSION: usize = 16;

    /// A [`HashEmbedding`] that counts model calls and can be made to fail.
    struct CountingEmbedding {
        inner: HashEmbedding,
        query_calls: AtomicUsize,
        document_calls: AtomicUsize,
        batch_sizes: Mutex<Vec<usize>>,
        failing: AtomicBool,
    }

    impl CountingEmbedding {
        fn new() -> Self {
            Self {
                inner: HashEmbedding::new(DIMENSION),
                query_calls: AtomicUsize::new(0),
                document_calls: AtomicUsize::new(0),
                batch_sizes: Mutex::new(Vec::new()),
                failing: AtomicBool::new(false),
            }
        }

        fn fail(&self) {
            self.failing.store(true, Ordering::SeqCst);
        }

        fn query_calls(&self) -> usize {
            self.query_calls.load(Ordering::SeqCst)
        }

        fn document_calls(&self) -> usize {
            self.document_calls.load(Ordering::SeqCst)
        }

        fn batch_sizes(&self) -> Vec<usize> {
            self.batch_sizes.lock().unwrap().clone()
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
            self.batch_sizes.lock().unwrap().push(texts.len());
            if self.failing.load(Ordering::SeqCst) {
                return Err(Error::llm("scripted embedding failure"));
            }
            self.inner.embed_documents(texts, batch_size).await
        }
    }

    fn index() -> EmbeddingIndex {
        EmbeddingIndex::new("hash-embedding", EmbeddingCache::in_memory())
    }

    fn wide_config() -> ChunkingConfig {
        ChunkingConfig {
            target_tokens: 1_000,
            overlap: 0,
        }
    }

    fn documents(entries: &[(&str, &str)]) -> Vec<(String, String)> {
        entries
            .iter()
            .map(|(id, content)| ((*id).to_owned(), (*content).to_owned()))
            .collect()
    }

    fn ids(hits: &[ScoredChunk]) -> Vec<&str> {
        hits.iter().map(|hit| hit.id.as_str()).collect()
    }

    #[test]
    fn cosine_similarity_of_identical_vectors_is_one() {
        let vector = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&vector, &vector) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_of_opposite_vectors_is_minus_one() {
        let left = vec![1.0, 0.0];
        let right = vec![-1.0, 0.0];
        assert!((cosine_similarity(&left, &right) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_of_orthogonal_vectors_is_zero() {
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_rejects_incomparable_vectors() {
        // Dimension mismatch, empty slices, zero norms, and NaN components.
        assert_eq!(cosine_similarity(&[1.0, 2.0], &[1.0, 2.0, 3.0]), 0.0);
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
        assert_eq!(cosine_similarity(&[1.0, 1.0], &[0.0, 0.0]), 0.0);
        assert_eq!(cosine_similarity(&[f32::NAN, 1.0], &[1.0, 1.0]), 0.0);
    }

    #[test]
    fn similarity_from_cosine_maps_the_range() {
        assert_eq!(similarity_from_cosine(1.0), 1.0);
        assert_eq!(similarity_from_cosine(0.0), 0.5);
        assert_eq!(similarity_from_cosine(-1.0), 0.0);
        assert_eq!(similarity_from_cosine(2.0), 1.0);
        assert_eq!(similarity_from_cosine(-2.0), 0.0);
        assert_eq!(similarity_from_cosine(f32::NAN), 0.0);
        assert_eq!(similarity_from_cosine(f32::INFINITY), 0.0);
    }

    #[test]
    fn blend_renormalizes_the_weights() {
        assert!((blend(0.8, 0.2, 0.7, 0.3) - 0.62).abs() < 1e-6);
        assert!((blend(0.8, 0.2, 7.0, 3.0) - 0.62).abs() < 1e-6);
        assert!((blend(0.4, 0.9, 1.0, 0.0) - 0.4).abs() < 1e-6);
        assert!((blend(0.4, 0.9, 0.0, 1.0) - 0.9).abs() < 1e-6);
    }

    #[test]
    fn blend_without_weights_takes_the_strongest_signal() {
        assert_eq!(blend(0.3, 0.8, 0.0, 0.0), 0.8);
        assert_eq!(blend(0.9, 0.2, 0.0, 0.0), 0.9);
        assert_eq!(blend(0.0, 0.0, 0.0, 0.0), 0.0);
    }

    #[test]
    fn blend_clamps_scores_and_ignores_invalid_weights() {
        assert_eq!(blend(1.5, -2.0, 0.5, 0.5), 0.5);
        assert_eq!(blend(0.6, 0.6, -1.0, -1.0), 0.6);
        assert_eq!(blend(f32::NAN, 0.4, 1.0, 1.0), 0.2);
        let blended = blend(0.5, 0.5, f32::NAN, 1.0);
        assert!(blended.is_finite(), "blend produced {blended}");
        assert!((blended - 0.5).abs() < 1e-6);
    }

    #[tokio::test]
    async fn reindex_embeds_misses_in_one_batch_and_chunks_whole_lines() {
        let embedding = CountingEmbedding::new();
        let mut index = index();
        let documents = documents(&[("MEMORY.md", "one\ntwo\nthree")]);
        let stats = index
            .reindex(&embedding, &documents, &wide_config())
            .await
            .unwrap();
        assert_eq!(stats.chunks, 1);
        assert_eq!(stats.embedded, 1);
        assert_eq!(stats.cached, 0);
        assert_eq!(stats.dimension, Some(DIMENSION));
        assert_eq!(embedding.document_calls(), 1);
        assert_eq!(embedding.batch_sizes(), vec![1]);
        assert_eq!(embedding.query_calls(), 0);
        assert_eq!(index.len(), 1);
        assert!(!index.is_empty());
        assert_eq!(index.model(), "hash-embedding");
        let chunk = &index.chunks()[0];
        assert_eq!(chunk.id, "MEMORY.md:1:3");
        assert_eq!(chunk.document, "MEMORY.md");
        assert_eq!(chunk.chunk.text, "one\ntwo\nthree");
        assert_eq!((chunk.chunk.start_line, chunk.chunk.end_line), (1, 3));
        assert_eq!(chunk.vector.as_ref().map(Vec::len), Some(DIMENSION));
    }

    #[tokio::test]
    async fn reindexing_unchanged_documents_makes_no_model_call() {
        let embedding = CountingEmbedding::new();
        let mut index = index();
        let documents = documents(&[
            ("MEMORY.md", "first line\nsecond line"),
            ("PROJECT.md", "build with cargo"),
        ]);
        let first = index
            .reindex(&embedding, &documents, &wide_config())
            .await
            .unwrap();
        assert_eq!(first.chunks, 2);
        assert_eq!(first.embedded, 2);
        assert_eq!(first.cached, 0);
        assert_eq!(embedding.document_calls(), 1);

        let second = index
            .reindex(&embedding, &documents, &wide_config())
            .await
            .unwrap();
        assert_eq!(second.chunks, 2);
        assert_eq!(second.embedded, 0);
        assert_eq!(second.cached, 2);
        assert_eq!(
            embedding.document_calls(),
            1,
            "a cache hit must not call the model"
        );
        assert_eq!(embedding.batch_sizes(), vec![2]);
    }

    #[tokio::test]
    async fn reindex_embeds_only_the_changed_document() {
        let embedding = CountingEmbedding::new();
        let mut index = index();
        let config = wide_config();
        index
            .reindex(
                &embedding,
                &documents(&[("a.md", "unchanged text"), ("b.md", "old text")]),
                &config,
            )
            .await
            .unwrap();
        let stats = index
            .reindex(
                &embedding,
                &documents(&[("a.md", "unchanged text"), ("b.md", "new text")]),
                &config,
            )
            .await
            .unwrap();
        assert_eq!(stats.chunks, 2);
        assert_eq!(stats.embedded, 1);
        assert_eq!(stats.cached, 1);
        assert_eq!(embedding.document_calls(), 2);
        assert_eq!(embedding.batch_sizes(), vec![2, 1]);
    }

    #[tokio::test]
    async fn reindex_replaces_the_chunk_list_and_skips_blank_documents() {
        let embedding = CountingEmbedding::new();
        let mut index = index();
        let config = wide_config();
        index
            .reindex(
                &embedding,
                &documents(&[("a.md", "alpha"), ("b.md", "beta")]),
                &config,
            )
            .await
            .unwrap();
        let stats = index
            .reindex(
                &embedding,
                &documents(&[("a.md", "alpha"), ("b.md", "   \n  ")]),
                &config,
            )
            .await
            .unwrap();
        assert_eq!(stats.chunks, 1);
        assert_eq!(stats.cached, 1);
        assert_eq!(stats.embedded, 0);
        assert_eq!(index.len(), 1);
        assert_eq!(index.chunks()[0].id, "a.md:1:1");

        let empty = index
            .reindex(&embedding, &documents(&[]), &config)
            .await
            .unwrap();
        assert_eq!(empty.chunks, 0);
        assert_eq!(index.len(), 0);
        assert_eq!(empty.dimension, Some(DIMENSION));
    }

    #[tokio::test]
    async fn reindex_numbers_chunks_per_document() {
        let embedding = CountingEmbedding::new();
        let mut index = index();
        let config = ChunkingConfig {
            target_tokens: 2,
            overlap: 0,
        };
        // Lines of 2 tokens each: every line becomes its own chunk.
        let content = ["aaaa bbbb", "cccc dddd", "eeee ffff"].join("\n");
        let stats = index
            .reindex(&embedding, &documents(&[("MEMORY.md", &content)]), &config)
            .await
            .unwrap();
        assert_eq!(stats.chunks, 3);
        let ids: Vec<&str> = index
            .chunks()
            .iter()
            .map(|chunk| chunk.id.as_str())
            .collect();
        assert_eq!(ids, vec!["MEMORY.md:1:1", "MEMORY.md:2:2", "MEMORY.md:3:3"]);
    }

    #[tokio::test]
    async fn reindex_rejects_an_invalid_config() {
        let embedding = CountingEmbedding::new();
        let mut index = index();
        let config = ChunkingConfig {
            target_tokens: 0,
            overlap: 0,
        };
        let error = index
            .reindex(&embedding, &documents(&[("a.md", "text")]), &config)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(embedding.document_calls(), 0);
    }

    #[tokio::test]
    async fn a_failed_reindex_keeps_the_previous_chunks() {
        let embedding = CountingEmbedding::new();
        let mut index = index();
        index
            .reindex(&embedding, &documents(&[("a.md", "kept")]), &wide_config())
            .await
            .unwrap();
        embedding.fail();
        let error = index
            .reindex(
                &embedding,
                &documents(&[("a.md", "kept"), ("b.md", "changed")]),
                &wide_config(),
            )
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Llm);
        assert_eq!(index.len(), 1, "the failed pass must not commit its chunks");
        assert_eq!(index.chunks()[0].id, "a.md:1:1");
    }

    #[tokio::test]
    async fn search_orders_by_similarity_then_id_and_truncates() {
        let embedding = CountingEmbedding::new();
        let mut index = index();
        let config = wide_config();
        index
            .reindex(
                &embedding,
                &documents(&[
                    ("d1", "alpha beta gamma"),
                    ("d2", "delta epsilon zeta"),
                    ("d3", "alpha beta gamma"),
                ]),
                &config,
            )
            .await
            .unwrap();
        let hits = index
            .search(&embedding, "alpha beta gamma", 10)
            .await
            .unwrap();
        assert_eq!(ids(&hits), vec!["d1:1:1", "d3:1:1", "d2:1:1"]);
        assert!((hits[0].similarity - 1.0).abs() < 1e-5);
        assert!(hits[1].similarity >= hits[2].similarity);
        assert_eq!(hits[0].chunk.text, "alpha beta gamma");
        assert_eq!(hits[0].document, "d1", "the document key maps a hit back");
        assert_eq!(embedding.query_calls(), 1);

        let limited = index
            .search(&embedding, "alpha beta gamma", 1)
            .await
            .unwrap();
        assert_eq!(ids(&limited), vec!["d1:1:1"]);
        assert_eq!(embedding.query_calls(), 2);

        let zero = index
            .search(&embedding, "alpha beta gamma", 0)
            .await
            .unwrap();
        assert_eq!(zero.len(), 1, "a zero limit is raised to one");
    }

    #[tokio::test]
    async fn search_on_an_empty_index_calls_no_model() {
        let embedding = CountingEmbedding::new();
        let index = index();
        assert_eq!(
            index.search(&embedding, "anything", 5).await.unwrap(),
            Vec::<ScoredChunk>::new()
        );
        assert_eq!(embedding.query_calls(), 0);
        assert_eq!(embedding.document_calls(), 0);
    }

    #[tokio::test]
    async fn search_rejects_a_blank_query() {
        let embedding = CountingEmbedding::new();
        let mut index = index();
        index
            .reindex(&embedding, &documents(&[("a.md", "text")]), &wide_config())
            .await
            .unwrap();
        for query in ["", "   ", "\n\t"] {
            let error = index.search(&embedding, query, 5).await.unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation);
        }
        assert_eq!(embedding.query_calls(), 0);
    }

    #[tokio::test]
    async fn a_chunk_without_a_vector_scores_zero_but_is_still_returned() {
        let embedding = CountingEmbedding::new();
        let mut index = index();
        index
            .reindex(
                &embedding,
                &documents(&[("a.md", "alpha"), ("b.md", "beta")]),
                &wide_config(),
            )
            .await
            .unwrap();
        // Simulate a chunk whose vector is missing (for example an evicted cache row).
        index.chunks[1].vector = None;
        let hits = index.search(&embedding, "alpha", 10).await.unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[1].id, "b.md:1:1");
        assert_eq!(
            hits[1].similarity, 0.0,
            "a missing vector must not look orthogonal"
        );

        // A vector of another dimension is not comparable either.
        index.chunks[1].vector = Some(vec![1.0, 0.0]);
        let hits = index.search(&embedding, "alpha", 10).await.unwrap();
        assert_eq!(hits[1].similarity, 0.0);
    }

    #[tokio::test]
    async fn clear_drops_chunks_but_keeps_cached_vectors() {
        let embedding = CountingEmbedding::new();
        let mut index = index();
        index
            .reindex(&embedding, &documents(&[("a.md", "text")]), &wide_config())
            .await
            .unwrap();
        index.clear();
        assert_eq!(index.len(), 0);
        let stats = index
            .reindex(&embedding, &documents(&[("a.md", "text")]), &wide_config())
            .await
            .unwrap();
        assert_eq!(stats.cached, 1);
        assert_eq!(stats.embedded, 0);
        assert_eq!(embedding.document_calls(), 1);
    }

    #[tokio::test]
    async fn a_shared_cache_is_reused_by_another_index() {
        let embedding = CountingEmbedding::new();
        let cache = EmbeddingCache::in_memory();
        let documents = documents(&[("a.md", "shared text")]);
        let mut first = EmbeddingIndex::new("first", cache.clone());
        first
            .reindex(&embedding, &documents, &wide_config())
            .await
            .unwrap();
        let mut second = EmbeddingIndex::new("second", cache);
        let stats = second
            .reindex(&embedding, &documents, &wide_config())
            .await
            .unwrap();
        assert_eq!(stats.cached, 1);
        assert_eq!(stats.embedded, 0);
        assert_eq!(embedding.document_calls(), 1);
        assert_eq!(second.model(), "second");
    }
}
