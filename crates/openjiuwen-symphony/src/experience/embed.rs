//! A batching, caching embedding client over [`openjiuwen_retrieval::Embedding`].
//!
//! Python's `experience/embed.py` shell is one thin wrapper around an embeddings endpoint:
//! it splits a document list into API-sized batches, normalizes the vectors, and counts
//! tokens. The Rust port keeps the two behaviours that matter for an offline,
//! deterministic crate — **batching** and **caching** — and delegates the transport to the
//! [`Embedding`] seam this workspace already uses.
//!
//! # Batching
//!
//! [`EmbeddingClient::embed_documents`](crate::experience::embed::EmbeddingClient::embed_documents)
//! splits the request into chunks of at most
//! [`batch_size`](crate::experience::embed::EmbeddingClient::batch_size) texts (default
//! [`DEFAULT_BATCH_SIZE`], clamped to `1..=MAX_BATCH_SIZE`) and issues exactly one
//! provider call per chunk. Input order is preserved, and a text repeated inside one call
//! is embedded once: the duplicates share the vector.
//!
//! # Caching
//!
//! Every vector is kept under
//! `"<model_name>:<sha256(text)>"` in a bounded FIFO cache (default
//! [`DEFAULT_CACHE_CAPACITY`] entries; `0` disables caching). Eviction is by insertion
//! sequence, never by time, so identical call sequences always produce identical cache
//! contents. The cache mutex is only held for the read or the write — never across an
//! `.await`.
//!
//! # Errors
//!
//! Provider failures propagate unchanged, so a caller still sees the typed
//! [`crate::error::Error`] the embedder produced; nothing is swallowed and no fallback
//! vector is invented.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use openjiuwen_retrieval::{Embedding, HashEmbedding};

use crate::error::{Error, Result};

/// The number of texts sent to the provider per call unless overridden.
pub const DEFAULT_BATCH_SIZE: usize = 32;

/// The largest accepted batch size; [`EmbeddingClient::with_batch_size`] clamps to it.
pub const MAX_BATCH_SIZE: usize = 256;

/// The number of cached vectors kept by default.
pub const DEFAULT_CACHE_CAPACITY: usize = 1024;

/// The counters of one [`EmbeddingClient`].
///
/// `calls` counts public entry-point invocations, `batches` counts the provider calls they
/// produced, `embedded` counts the texts actually handed to the provider, and
/// `cache_hits`/`cache_misses` count the cache lookups (one per distinct text).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct EmbeddingClientStats {
    /// The number of [`EmbeddingClient::embed_query`]/[`EmbeddingClient::embed_documents`]
    /// calls.
    pub calls: u64,
    /// The number of provider calls those made.
    pub batches: u64,
    /// The number of texts handed to the provider.
    pub embedded: u64,
    /// The number of distinct texts served from the cache.
    pub cache_hits: u64,
    /// The number of distinct texts the cache did not hold.
    pub cache_misses: u64,
}

/// The live counters behind [`EmbeddingClientStats`].
#[derive(Debug, Default)]
struct Counters {
    /// Public entry-point invocations.
    calls: AtomicU64,
    /// Provider invocations.
    batches: AtomicU64,
    /// Texts handed to the provider.
    embedded: AtomicU64,
    /// Cache hits (per distinct text lookup).
    cache_hits: AtomicU64,
    /// Cache misses (per distinct text lookup).
    cache_misses: AtomicU64,
}

impl Counters {
    /// Reads a counter with relaxed ordering: the statistics are telemetry, never a gate.
    fn read(counter: &AtomicU64) -> u64 {
        counter.load(Ordering::Relaxed)
    }
}

/// A bounded FIFO cache of embedding vectors, keyed by model name and text digest.
#[derive(Debug)]
struct VectorCache {
    /// The maximum number of entries; `0` disables the cache.
    capacity: usize,
    /// The cached vectors.
    entries: HashMap<String, Vec<f32>>,
    /// The keys in insertion order, for FIFO eviction.
    order: VecDeque<String>,
}

impl VectorCache {
    /// An empty cache bounded by `capacity`.
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    /// Returns the cached vector without changing the eviction order.
    fn get(&self, key: &str) -> Option<Vec<f32>> {
        if self.capacity == 0 {
            return None;
        }
        self.entries.get(key).cloned()
    }

    /// Inserts `vector` under `key`, evicting the oldest entries to stay within capacity.
    fn insert(&mut self, key: String, vector: Vec<f32>) {
        if self.capacity == 0 {
            return;
        }
        if let Some(existing) = self.entries.get_mut(&key) {
            *existing = vector;
            return;
        }
        while self.entries.len() >= self.capacity {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            self.entries.remove(&oldest);
        }
        self.order.push_back(key.clone());
        self.entries.insert(key, vector);
    }

    /// Drops the oldest entries until the cache fits `capacity`.
    fn resize(&mut self, capacity: usize) {
        self.capacity = capacity;
        while self.entries.len() > self.capacity {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            self.entries.remove(&oldest);
        }
        if self.capacity == 0 {
            self.entries.clear();
            self.order.clear();
        }
    }

    /// The number of cached vectors.
    fn len(&self) -> usize {
        self.entries.len()
    }
}

/// A batching, caching [`Embedding`] wrapper.
///
/// The client owns the embedding model, so it is cheap to clone behind an [`Arc`]:
/// construct one per model and share it with the experience layers that need vectors.
pub struct EmbeddingClient {
    /// The wrapped embedding model.
    embedding: Arc<dyn Embedding>,
    /// The number of texts sent to the provider per call.
    batch_size: usize,
    /// The bounded vector cache.
    cache: Mutex<VectorCache>,
    /// The telemetry counters.
    counters: Counters,
}

impl fmt::Debug for EmbeddingClient {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EmbeddingClient")
            .field("model", &self.embedding.model_name())
            .field("batch_size", &self.batch_size)
            .field("cache_len", &self.cache_len())
            .field("cache_capacity", &self.cache_capacity())
            .finish_non_exhaustive()
    }
}

impl EmbeddingClient {
    /// Wraps `embedding` with [`DEFAULT_BATCH_SIZE`] and [`DEFAULT_CACHE_CAPACITY`].
    #[must_use]
    pub fn new(embedding: Arc<dyn Embedding>) -> Self {
        Self {
            embedding,
            batch_size: DEFAULT_BATCH_SIZE,
            cache: Mutex::new(VectorCache::new(DEFAULT_CACHE_CAPACITY)),
            counters: Counters::default(),
        }
    }

    /// A deterministic offline client over
    /// [`HashEmbedding::default`](openjiuwen_retrieval::HashEmbedding).
    #[must_use]
    pub fn hash() -> Self {
        Self::new(Arc::new(HashEmbedding::default()))
    }

    /// An alias of [`Self::hash`], named for the offline/test intent.
    #[must_use]
    pub fn offline() -> Self {
        Self::hash()
    }

    /// Sets the batch size, clamped to `1..=MAX_BATCH_SIZE`.
    ///
    /// The clamp keeps `range`-style chunking valid: a batch size of `0` would otherwise
    /// mean "never call the provider" instead of "one text at a time".
    #[must_use]
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.clamp(1, MAX_BATCH_SIZE);
        self
    }

    /// Sets the cache capacity; shrinking drops the oldest entries, `0` disables caching.
    #[must_use]
    pub fn with_cache_capacity(mut self, capacity: usize) -> Self {
        self.cache_mut().resize(capacity);
        self
    }

    /// The wrapped embedding model.
    #[must_use]
    pub fn embedding(&self) -> &Arc<dyn Embedding> {
        &self.embedding
    }

    /// The model name reported by the wrapped embedder.
    #[must_use]
    pub fn model_name(&self) -> &str {
        self.embedding.model_name()
    }

    /// The number of texts sent to the provider per call.
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// The number of cached vectors.
    #[must_use]
    pub fn cache_len(&self) -> usize {
        self.lock_cache().len()
    }

    /// The configured cache capacity (`0` when caching is disabled).
    #[must_use]
    pub fn cache_capacity(&self) -> usize {
        self.lock_cache().capacity
    }

    /// A snapshot of the telemetry counters.
    #[must_use]
    pub fn stats(&self) -> EmbeddingClientStats {
        EmbeddingClientStats {
            calls: Counters::read(&self.counters.calls),
            batches: Counters::read(&self.counters.batches),
            embedded: Counters::read(&self.counters.embedded),
            cache_hits: Counters::read(&self.counters.cache_hits),
            cache_misses: Counters::read(&self.counters.cache_misses),
        }
    }

    /// Embeds one query string, served from the cache when it is already known.
    ///
    /// # Errors
    ///
    /// Propagates the wrapped embedder's error unchanged.
    pub async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.counters.calls.fetch_add(1, Ordering::Relaxed);
        let key = self.cache_key(text);
        if let Some(vector) = self.cache_get(&key) {
            self.counters.cache_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(vector);
        }
        self.counters.cache_misses.fetch_add(1, Ordering::Relaxed);
        let vector = self.embedding.embed_query(text).await?;
        self.counters.batches.fetch_add(1, Ordering::Relaxed);
        self.counters.embedded.fetch_add(1, Ordering::Relaxed);
        self.cache_put(key, &vector);
        Ok(vector)
    }

    /// Embeds `texts`, preserving input order.
    ///
    /// Each distinct text is embedded at most once: a repeat inside this call and a text
    /// already in the cache both reuse their vector. An empty request is answered with an
    /// empty vector list and makes no provider call.
    ///
    /// # Errors
    ///
    /// Propagates the wrapped embedder's error unchanged, including its validation errors
    /// for blank texts. A provider that returns a different number of vectors than it was
    /// asked for is reported as [`Error::Retrieval`], because silently re-aligning the
    /// rows would attach the wrong vector to a document.
    pub async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.counters.calls.fetch_add(1, Ordering::Relaxed);
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let (unique_texts, assignment) = distinct_texts(texts);
        let mut resolved: Vec<Option<Vec<f32>>> = vec![None; unique_texts.len()];
        let mut missing: Vec<(usize, &str)> = Vec::new();
        for (slot, text) in unique_texts.iter().enumerate() {
            let key = self.cache_key(text);
            if let Some(vector) = self.cache_get(&key) {
                self.counters.cache_hits.fetch_add(1, Ordering::Relaxed);
                resolved[slot] = Some(vector);
            } else {
                self.counters.cache_misses.fetch_add(1, Ordering::Relaxed);
                missing.push((slot, text));
            }
        }
        for chunk in missing.chunks(self.batch_size) {
            let batch: Vec<String> = chunk.iter().map(|(_, text)| (*text).to_owned()).collect();
            let vectors = self
                .embedding
                .embed_documents(&batch, Some(self.batch_size))
                .await?;
            self.counters.batches.fetch_add(1, Ordering::Relaxed);
            self.counters
                .embedded
                .fetch_add(batch.len() as u64, Ordering::Relaxed);
            if vectors.len() != batch.len() {
                return Err(Error::retrieval(format!(
                    "embedding model `{}` returned {} vectors for {} texts",
                    self.embedding.model_name(),
                    vectors.len(),
                    batch.len()
                )));
            }
            for ((slot, text), vector) in chunk.iter().zip(vectors) {
                self.cache_put(self.cache_key(text), &vector);
                resolved[*slot] = Some(vector);
            }
        }
        let mut output = Vec::with_capacity(texts.len());
        for slot in assignment {
            let Some(vector) = resolved[slot].as_ref() else {
                return Err(Error::retrieval(format!(
                    "embedding model `{}` produced no vector for one of the requested texts",
                    self.embedding.model_name()
                )));
            };
            output.push(vector.clone());
        }
        Ok(output)
    }

    /// The cache key of `text`: the model name plus the text's SHA-256 hex digest.
    fn cache_key(&self, text: &str) -> String {
        format!(
            "{}:{:x}",
            self.embedding.model_name(),
            Sha256::digest(text.as_bytes())
        )
    }

    /// Locks the cache, recovering from a poisoned lock.
    ///
    /// A panic while the mutex was held cannot corrupt a vector map, so recovering keeps
    /// the client usable instead of propagating an unrelated panic.
    fn lock_cache(&self) -> std::sync::MutexGuard<'_, VectorCache> {
        self.cache.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Reads the cache without holding the lock beyond the clone.
    fn cache_get(&self, key: &str) -> Option<Vec<f32>> {
        self.lock_cache().get(key)
    }

    /// Writes the cache without holding the lock beyond the insert.
    fn cache_put(&self, key: String, vector: &[f32]) {
        self.lock_cache().insert(key, vector.to_vec());
    }

    /// Mutable access to the cache for construction-time resizing.
    fn cache_mut(&mut self) -> &mut VectorCache {
        self.cache.get_mut().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Splits `texts` into its distinct texts and the slot of each input.
///
/// The distinct list keeps first-seen order, so the provider sees the request in caller
/// order and the cache warms deterministically.
fn distinct_texts(texts: &[String]) -> (Vec<&str>, Vec<usize>) {
    let mut unique: Vec<&str> = Vec::new();
    let mut slots: HashMap<&str, usize> = HashMap::new();
    let mut assignment = Vec::with_capacity(texts.len());
    for text in texts {
        let slot = *slots.entry(text.as_str()).or_insert_with(|| {
            unique.push(text.as_str());
            unique.len() - 1
        });
        assignment.push(slot);
    }
    (unique, assignment)
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::assert_is_empty,
    clippy::cast_precision_loss
)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::Mutex as StdMutex;

    /// An embedder that records the length of every provider call.
    #[derive(Default)]
    struct CountingEmbedding {
        /// The batch length of each provider call, in call order.
        calls: StdMutex<Vec<usize>>,
        /// Whether every call fails.
        fail: bool,
        /// When set, the provider returns this many vectors regardless of the request.
        force_len: Option<usize>,
    }

    impl CountingEmbedding {
        /// The recorded call lengths.
        fn calls(&self) -> Vec<usize> {
            self.calls.lock().unwrap().clone()
        }

        /// The number of provider calls.
        fn call_count(&self) -> usize {
            self.calls.lock().unwrap().len()
        }
    }

    #[async_trait]
    impl Embedding for CountingEmbedding {
        fn model_name(&self) -> &'static str {
            "counting"
        }

        fn dimension_hint(&self) -> Option<usize> {
            Some(2)
        }

        async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
            if self.fail {
                return Err(Error::retrieval("provider is down"));
            }
            self.calls.lock().unwrap().push(1);
            Ok(vec![text.len() as f32, 1.0])
        }

        async fn embed_documents(
            &self,
            texts: &[String],
            _batch_size: Option<usize>,
        ) -> Result<Vec<Vec<f32>>> {
            if self.fail {
                return Err(Error::retrieval("provider is down"));
            }
            self.calls.lock().unwrap().push(texts.len());
            let rows = self.force_len.unwrap_or(texts.len());
            Ok(texts
                .iter()
                .take(rows)
                .map(|text| vec![text.len() as f32, 1.0])
                .collect())
        }
    }

    /// A client over a fresh [`CountingEmbedding`], keeping the fake reachable.
    fn counting_client() -> (EmbeddingClient, Arc<CountingEmbedding>) {
        let embedding = Arc::new(CountingEmbedding::default());
        (EmbeddingClient::new(embedding.clone()), embedding)
    }

    /// `count` distinct documents.
    fn documents(count: usize) -> Vec<String> {
        (0..count)
            .map(|index| format!("document {index}"))
            .collect()
    }

    #[tokio::test]
    async fn documents_are_batched_by_the_batch_size() {
        let (client, embedding) = counting_client();
        let client = client.with_batch_size(2);
        let vectors = client
            .embed_documents(&documents(5))
            .await
            .expect("embed documents");
        assert_eq!(vectors.len(), 5);
        assert_eq!(embedding.calls(), vec![2, 2, 1]);
        let stats = client.stats();
        assert_eq!(stats.calls, 1);
        assert_eq!(stats.batches, 3);
        assert_eq!(stats.embedded, 5);
        assert_eq!(stats.cache_misses, 5);
        assert_eq!(stats.cache_hits, 0);
    }

    #[tokio::test]
    async fn batch_size_is_clamped_to_the_supported_range() {
        assert_eq!(EmbeddingClient::hash().with_batch_size(0).batch_size(), 1);
        assert_eq!(
            EmbeddingClient::hash().with_batch_size(10_000).batch_size(),
            MAX_BATCH_SIZE
        );
        assert_eq!(EmbeddingClient::hash().batch_size(), DEFAULT_BATCH_SIZE);

        let (client, embedding) = counting_client();
        let client = client.with_batch_size(usize::MAX);
        assert_eq!(client.batch_size(), MAX_BATCH_SIZE);
        let texts = documents(300);
        assert_eq!(
            client.embed_documents(&texts).await.expect("embed").len(),
            300
        );
        assert_eq!(
            embedding.calls(),
            vec![MAX_BATCH_SIZE, 300 - MAX_BATCH_SIZE]
        );
    }

    #[tokio::test]
    async fn duplicate_texts_are_embedded_once() {
        let (client, embedding) = counting_client();
        let texts = vec!["same".to_string(), "other".to_string(), "same".to_string()];
        let vectors = client.embed_documents(&texts).await.expect("embed");
        assert_eq!(vectors.len(), 3);
        assert_eq!(vectors[0], vectors[2]);
        assert_eq!(embedding.calls(), vec![2]);
        let stats = client.stats();
        assert_eq!(stats.embedded, 2);
        assert_eq!(stats.cache_misses, 2);
    }

    #[tokio::test]
    async fn cache_reuses_vectors_across_calls() {
        let (client, embedding) = counting_client();
        let first = client
            .embed_query("summarize the report")
            .await
            .expect("first");
        let second = client
            .embed_query("summarize the report")
            .await
            .expect("second");
        assert_eq!(first, second);
        assert_eq!(embedding.call_count(), 1);
        assert_eq!(client.cache_len(), 1);
        let stats = client.stats();
        assert_eq!(stats.calls, 2);
        assert_eq!(stats.batches, 1);
        assert_eq!(stats.embedded, 1);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 1);

        // A document batch served entirely from the cache makes no provider call.
        let again = client
            .embed_documents(&["summarize the report".to_string()])
            .await
            .expect("documents");
        assert_eq!(again, vec![first]);
        assert_eq!(embedding.call_count(), 1);
        assert_eq!(client.stats().cache_hits, 2);
    }

    #[tokio::test]
    async fn fifo_eviction_bounds_the_cache() {
        let (client, embedding) = counting_client();
        let client = client.with_cache_capacity(2);
        assert_eq!(client.cache_capacity(), 2);
        client.embed_query("alpha").await.expect("alpha");
        client.embed_query("beta").await.expect("beta");
        client.embed_query("gamma").await.expect("gamma");
        assert_eq!(client.cache_len(), 2);
        // `alpha` was evicted first, so asking again reaches the provider and, being a
        // new insertion, evicts the then-oldest entry (`beta`).
        client.embed_query("alpha").await.expect("alpha again");
        assert_eq!(embedding.call_count(), 4);
        assert_eq!(client.stats().cache_hits, 0);
        // `gamma` is the surviving older entry and is still cached.
        client.embed_query("gamma").await.expect("gamma again");
        assert_eq!(embedding.call_count(), 4);
        assert_eq!(client.stats().cache_hits, 1);
        // `beta` was evicted by `alpha`, so it reaches the provider once more.
        client.embed_query("beta").await.expect("beta again");
        assert_eq!(embedding.call_count(), 5);
        assert_eq!(client.stats().cache_hits, 1);
    }

    #[tokio::test]
    async fn shrinking_the_cache_drops_the_oldest_entries() {
        let (client, embedding) = counting_client();
        for text in ["one", "two", "three"] {
            client.embed_query(text).await.expect("embed");
        }
        assert_eq!(client.cache_len(), 3);
        let client = client.with_cache_capacity(1);
        assert_eq!(client.cache_len(), 1);
        // The survivor is the newest entry, so the two oldest are re-embedded.
        client.embed_query("three").await.expect("cached");
        let before = embedding.call_count();
        client.embed_query("one").await.expect("re-embed");
        assert_eq!(embedding.call_count(), before + 1);
    }

    #[tokio::test]
    async fn zero_capacity_disables_caching() {
        let (client, embedding) = counting_client();
        let client = client.with_cache_capacity(0);
        assert_eq!(client.cache_capacity(), 0);
        client.embed_query("same text").await.expect("first");
        client.embed_query("same text").await.expect("second");
        assert_eq!(client.cache_len(), 0);
        assert_eq!(embedding.call_count(), 2);
        let stats = client.stats();
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 2);
    }

    #[tokio::test]
    async fn embedding_errors_propagate_unchanged() {
        let embedding = Arc::new(CountingEmbedding {
            fail: true,
            ..CountingEmbedding::default()
        });
        let client = EmbeddingClient::new(embedding);
        let error = client
            .embed_query("anything")
            .await
            .expect_err("query must fail");
        assert_eq!(error.kind(), crate::error::ErrorKind::Retrieval);
        assert!(error.message().contains("provider is down"));
        let error = client
            .embed_documents(&documents(2))
            .await
            .expect_err("documents must fail");
        assert_eq!(error.kind(), crate::error::ErrorKind::Retrieval);
        assert_eq!(client.cache_len(), 0);
    }

    #[tokio::test]
    async fn a_row_count_mismatch_is_a_typed_error() {
        let embedding = Arc::new(CountingEmbedding {
            force_len: Some(1),
            ..CountingEmbedding::default()
        });
        let client = EmbeddingClient::new(embedding);
        let error = client
            .embed_documents(&documents(3))
            .await
            .expect_err("mismatch must fail");
        assert_eq!(error.kind(), crate::error::ErrorKind::Retrieval);
        assert!(error.message().contains("returned 1 vectors for 3 texts"));
    }

    #[tokio::test]
    async fn an_empty_request_makes_no_provider_call() {
        let (client, embedding) = counting_client();
        assert!(client.embed_documents(&[]).await.expect("empty").is_empty());
        assert_eq!(embedding.call_count(), 0);
        assert_eq!(client.stats().calls, 1);
        assert_eq!(client.stats().batches, 0);
    }

    #[tokio::test]
    async fn the_hash_backed_client_is_byte_stable() {
        let client = EmbeddingClient::hash();
        assert_eq!(client.model_name(), EmbeddingClient::offline().model_name());
        assert_eq!(client.batch_size(), DEFAULT_BATCH_SIZE);
        assert_eq!(client.cache_capacity(), DEFAULT_CACHE_CAPACITY);
        let first = client
            .embed_query("bake a chocolate cake")
            .await
            .expect("first");
        let second = EmbeddingClient::offline()
            .embed_query("bake a chocolate cake")
            .await
            .expect("second");
        assert_eq!(first, second);
        assert_eq!(
            first.len(),
            openjiuwen_retrieval::HashEmbedding::default().dimension_size()
        );
        assert_eq!(client.stats().cache_hits, 0);
        client
            .embed_query("bake a chocolate cake")
            .await
            .expect("cached");
        assert_eq!(client.stats().cache_hits, 1);
    }

    #[test]
    fn cache_keys_carry_the_model_name_and_a_digest() {
        let client = EmbeddingClient::hash();
        let key = client.cache_key("some text");
        let (model, digest) = key.split_once(':').expect("model:digest");
        assert_eq!(model, client.model_name());
        assert_eq!(digest.len(), 64);
        assert_ne!(key, client.cache_key("some other text"));
        assert_eq!(key, client.cache_key("some text"));
    }

    #[test]
    fn distinct_texts_keeps_first_seen_order() {
        let texts = vec![
            "b".to_string(),
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "a".to_string(),
        ];
        let (unique, assignment) = distinct_texts(&texts);
        assert_eq!(unique, vec!["b", "a", "c"]);
        assert_eq!(assignment, vec![0, 1, 0, 2, 1]);
    }

    #[test]
    fn stats_are_serde_round_trippable() {
        let stats = EmbeddingClientStats {
            calls: 1,
            batches: 2,
            embedded: 3,
            cache_hits: 4,
            cache_misses: 5,
        };
        let json = serde_json::to_string(&stats).expect("serialize");
        let parsed: EmbeddingClientStats = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, stats);
        let parsed: EmbeddingClientStats =
            serde_json::from_str(r#"{"unknown":true}"#).expect("defaults");
        assert_eq!(parsed, EmbeddingClientStats::default());
    }
}
