#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Deterministic dedupe/merge integration tests for
//! [`openjiuwen_memory::facade::LongTermMemory`].
//!
//! These tests pin the real write-path behaviour behind the "Deterministic
//! dedupe/merge" design decision (`crates/openjiuwen-memory/CONTEXT.md`):
//! exact normalized-text equality, embedding-cosine similarity (default threshold
//! `0.92`), token Jaccard (default `0.75`), and the `Merge`/`Skip`/`Replace`/`Add`
//! policies including the per-call `AddOptions::with_dedupe` override.
//! Everything runs offline against in-memory stores with fixed timestamps, so no
//! assertion depends on the wall clock.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::store::{
    CollectionSchema, InMemoryVectorStore, VectorQuery, VectorSearchResult, VectorStore,
};
use openjiuwen_memory::analyzer::ExtractionInput;
use openjiuwen_memory::config::MemoryEngineConfig;
use openjiuwen_memory::facade::LongTermMemory;
use openjiuwen_memory::record::{
    AddOptions, AddOutcome, CallOptions, DedupePolicy, MemoryScope, MemoryType, MemoryUpdate,
    NewMemory, SearchOptions,
};
use openjiuwen_memory::store::{keys, MemoryStores};
use openjiuwen_retrieval::embedding::{Embedding, HashEmbedding};
use serde_json::{json, Map, Value};

/// Builds an empty in-memory facade over `config`.
async fn facade(config: MemoryEngineConfig) -> Arc<LongTermMemory> {
    LongTermMemory::in_memory(config)
        .await
        .expect("in-memory facade")
}

/// Builds an in-memory facade with an injected embedding model.
fn facade_with_embedding(
    config: MemoryEngineConfig,
    embedding: Arc<dyn Embedding>,
) -> Arc<LongTermMemory> {
    LongTermMemory::builder(config)
        .embedding(embedding)
        .build()
        .expect("facade with embedding")
}

/// The scope every test owns its records under.
fn scope() -> MemoryScope {
    MemoryScope::new("alice", "ws")
}

/// A deterministic timestamp `offset` seconds after a fixed epoch.
fn ts(offset: i64) -> DateTime<Utc> {
    DateTime::from_timestamp(1_700_000_000 + offset, 0).expect("fixed timestamp")
}

/// `AddOptions` carrying a per-call dedupe policy override.
fn add_opts(policy: DedupePolicy) -> AddOptions {
    AddOptions::default().with_dedupe(policy)
}

/// An embedding model that returns the same non-zero vector for every text.
///
/// The engine never re-reads a stored embedding during dedupe (the record JSON keeps
/// `embedding: None`; only the vector column holds it), so `find_duplicate` compares the
/// incoming embedding against the stored record's recall similarity. Returning one
/// constant vector makes that recall similarity exactly `1.0` for every pair, which
/// lets the cosine rule be exercised deterministically. `HashEmbedding` cannot do this
/// for unrelated texts: its mean-pooled signed hash vectors are nearly orthogonal
/// (normalized similarity around `0.5`), well below the configured `0.92` threshold.
struct ConstantEmbedding;

#[async_trait]
impl Embedding for ConstantEmbedding {
    fn model_name(&self) -> &'static str {
        "constant"
    }

    fn dimension_hint(&self) -> Option<usize> {
        Some(2)
    }

    async fn embed_query(&self, _text: &str) -> Result<Vec<f32>> {
        Ok(vec![1.0, 0.5])
    }

    async fn embed_documents(
        &self,
        texts: &[String],
        _batch_size: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![1.0, 0.5]).collect())
    }
}

#[tokio::test]
async fn add_outcome_distinguishes_added_from_merged() {
    let ltm = facade(MemoryEngineConfig::default()).await;
    let scope = scope();

    let first = ltm
        .add(
            &scope,
            NewMemory::semantic("Alice prefers dark mode"),
            &AddOptions::default(),
        )
        .await
        .expect("first add");
    assert_eq!(first.added.len(), 1);
    assert_eq!(first.len(), 1);
    assert_eq!(first.merged.len(), 0);
    assert_eq!(first.skipped.len(), 0);
    assert_eq!(first.deleted.len(), 0);
    let id = first.added[0].id.clone();

    let second = ltm
        .add(
            &scope,
            NewMemory::semantic("Alice prefers dark mode"),
            &AddOptions::default(),
        )
        .await
        .expect("duplicate add");
    assert_eq!(second.added.len(), 0, "the duplicate must not be added");
    assert_eq!(second.merged.len(), 1);
    assert_eq!(second.skipped.len(), 0);
    assert_eq!(second.len(), 1);
    // `merged` reports the stored record, whose id is the one created by the first add.
    assert_eq!(second.merged[0].id, id);
    assert_eq!(ltm.list(&scope, &[]).await.expect("list").len(), 1);
}

#[tokio::test]
async fn exact_normalized_duplicates_merge_into_the_existing_record() {
    let ltm = facade(MemoryEngineConfig::default()).await;
    let scope = scope();

    let first = ltm
        .add(
            &scope,
            NewMemory::semantic("Alice prefers dark mode"),
            &AddOptions::default(),
        )
        .await
        .expect("add");
    let id = first.added[0].id.clone();

    // `normalize_text` lowercases, collapses whitespace, and strips edge punctuation,
    // so this input normalizes to "alice prefers dark mode" (exact rule).
    let incoming_text = "  Alice   prefers dark mode. ";
    assert_eq!(
        openjiuwen_memory::record::normalize_text(incoming_text),
        openjiuwen_memory::record::normalize_text("Alice prefers dark mode")
    );
    let second = ltm
        .add(
            &scope,
            NewMemory::semantic(incoming_text),
            &AddOptions::default(),
        )
        .await
        .expect("duplicate add");
    assert_eq!(second.added.len(), 0);
    assert_eq!(second.merged.len(), 1);
    assert_eq!(second.merged[0].id, id);

    // Merge keeps the existing id and takes the longer raw content (the incoming text
    // has 30 characters, the existing one 23).
    let stored = ltm
        .get(&scope, &id)
        .await
        .expect("get")
        .expect("stored record");
    assert_eq!(stored.id, id);
    assert_eq!(stored.content, incoming_text);
    assert_eq!(ltm.list(&scope, &[]).await.expect("list").len(), 1);
}

#[tokio::test]
async fn near_duplicates_merge_by_embedding_cosine() {
    let first_text = "alice keeps a spare umbrella";
    let second_text = "quantum chromodynamics lecture notes";

    // Document why the test does not use `HashEmbedding` for this case: measured on the
    // actual texts, the hash vectors' normalized similarity is far below the default
    // `similarity_threshold` of 0.92.
    let hash = HashEmbedding::new(16);
    let left = hash.embed_query(first_text).await.expect("hash left");
    let right = hash.embed_query(second_text).await.expect("hash right");
    let hash_similarity = openjiuwen_memory::recall::normalized_similarity(&left, &right);
    assert!(
        hash_similarity < 0.92,
        "HashEmbedding unexpectedly crossed the cosine threshold: {hash_similarity}"
    );

    // A local test-only embedding returns one constant vector, so the dedupe scan sees
    // cosine 1.0 (normalized similarity 1.0) and merges deterministically.
    let ltm = facade_with_embedding(MemoryEngineConfig::default(), Arc::new(ConstantEmbedding));
    let scope = scope();
    assert!(
        (openjiuwen_memory::recall::normalized_similarity(&[1.0, 0.5], &[1.0, 0.5]) - 1.0).abs()
            < f32::EPSILON
    );
    // The texts share no token, so the Jaccard rule cannot be responsible for the match.
    assert_eq!(
        openjiuwen_memory::recall::jaccard(first_text, second_text),
        0.0
    );

    let first = ltm
        .add(
            &scope,
            NewMemory::semantic(first_text),
            &AddOptions::default(),
        )
        .await
        .expect("add");
    let id = first.added[0].id.clone();

    let second = ltm
        .add(
            &scope,
            NewMemory::semantic(second_text),
            &AddOptions::default(),
        )
        .await
        .expect("similar add");
    assert_eq!(second.added.len(), 0);
    assert_eq!(second.merged.len(), 1);
    assert_eq!(second.merged[0].id, id);
    assert_eq!(ltm.list(&scope, &[]).await.expect("list").len(), 1);
}

#[tokio::test]
async fn merge_takes_metadata_union_max_importance_and_stable_identity() {
    let ltm = facade(MemoryEngineConfig::default()).await;
    let scope = scope();

    let shorter = "alice prefers dark mode for the editor";
    let longer = "alice prefers dark mode for the editor window";
    // Token Jaccard is 7/8 = 0.875 >= the default 0.75, so the pair is `Similar`.
    assert!(openjiuwen_memory::recall::jaccard(shorter, longer) >= 0.75);

    let a = ltm
        .add(
            &scope,
            NewMemory::semantic(shorter)
                .with_timestamp(ts(0))
                .with_importance(0.3)
                .with_metadata("source", json!("chat"))
                .with_metadata("tag", json!("a")),
            &AddOptions::default(),
        )
        .await
        .expect("add A");
    let id = a.added[0].id.clone();

    let b = ltm
        .add(
            &scope,
            NewMemory::semantic(longer)
                .with_timestamp(ts(60))
                .with_importance(0.9)
                .with_metadata("tag", json!("b"))
                .with_metadata("extra", json!("x")),
            &AddOptions::default(),
        )
        .await
        .expect("add B");
    assert_eq!(b.added.len(), 0);
    assert_eq!(b.merged.len(), 1);
    let merged = &b.merged[0];

    // Identity and creation time come from the existing record (A).
    assert_eq!(merged.id, id);
    assert_eq!(merged.timestamp, ts(0));
    // Content becomes the longer text (B).
    assert_eq!(merged.content, longer);
    // Importance is the maximum of both sides.
    assert!((merged.importance - 0.9).abs() < f32::EPSILON);
    // Metadata is the union with incoming values winning conflicts.
    assert_eq!(merged.metadata.get("source"), Some(&json!("chat")));
    assert_eq!(merged.metadata.get("tag"), Some(&json!("b")));
    assert_eq!(merged.metadata.get("extra"), Some(&json!("x")));
    assert_eq!(merged.metadata.len(), 3);

    let stored = ltm
        .get(&scope, &id)
        .await
        .expect("get")
        .expect("stored record");
    assert_eq!(&stored, merged);
    assert_eq!(ltm.list(&scope, &[]).await.expect("list").len(), 1);
}

#[tokio::test]
async fn skip_policy_drops_the_duplicate_and_leaves_the_record_unchanged() {
    let ltm = facade(MemoryEngineConfig::default()).await;
    let scope = scope();

    let first = ltm
        .add(
            &scope,
            NewMemory::semantic("alice prefers dark mode"),
            &AddOptions::default(),
        )
        .await
        .expect("add");
    let id = first.added[0].id.clone();
    let before = ltm
        .get(&scope, &id)
        .await
        .expect("get")
        .expect("stored record");

    let second = ltm
        .add(
            &scope,
            NewMemory::semantic("alice prefers dark mode"),
            &add_opts(DedupePolicy::Skip),
        )
        .await
        .expect("skip duplicate");
    assert_eq!(second.added.len(), 0);
    assert_eq!(second.merged.len(), 0);
    assert_eq!(second.skipped.len(), 1);
    // `skipped` reports the incoming (never stored) record id, not the existing one.
    assert_ne!(second.skipped[0], id);
    assert_eq!(second.skipped[0].len(), 24);

    let after = ltm
        .get(&scope, &id)
        .await
        .expect("get")
        .expect("stored record");
    assert_eq!(before, after);
    assert_eq!(ltm.list(&scope, &[]).await.expect("list").len(), 1);
}

#[tokio::test]
async fn replace_policy_keeps_the_id_but_takes_the_incoming_content() {
    let ltm = facade(MemoryEngineConfig::default()).await;
    let scope = scope();

    let a = ltm
        .add(
            &scope,
            NewMemory::semantic("alice prefers dark mode")
                .with_timestamp(ts(0))
                .with_importance(0.3)
                .with_metadata("source", json!("chat"))
                .with_metadata("tag", json!("a")),
            &AddOptions::default(),
        )
        .await
        .expect("add A");
    let id = a.added[0].id.clone();

    // Exact normalized duplicate, so the policy is exercised without relying on
    // similarity thresholds.
    let incoming_text = "  Alice   prefers DARK mode. ";
    let replaced = ltm
        .add(
            &scope,
            NewMemory::semantic(incoming_text)
                .with_timestamp(ts(30))
                .with_importance(0.7)
                .with_metadata("tag", json!("b")),
            &add_opts(DedupePolicy::Replace),
        )
        .await
        .expect("replace duplicate");
    assert_eq!(replaced.added.len(), 0);
    assert_eq!(replaced.merged.len(), 1);
    // The superseded row was really removed, so the outcome reports the deletion even
    // though the replacement is stored under the same id.
    assert_eq!(replaced.deleted, vec![id.clone()]);
    assert_eq!(replaced.skipped.len(), 0);
    let record = &replaced.merged[0];

    // Actual `Replace` semantics: the existing id survives, and the whole incoming
    // record (content/timestamp/importance/metadata) takes its place. Metadata is not
    // unioned and importance is not maximized, unlike `Merge`.
    assert_eq!(record.id, id);
    assert_eq!(record.content, incoming_text);
    assert_eq!(record.timestamp, ts(30));
    assert!((record.importance - 0.7).abs() < f32::EPSILON);
    assert_eq!(record.metadata.get("tag"), Some(&json!("b")));
    assert!(!record.metadata.contains_key("source"));

    let stored = ltm
        .get(&scope, &id)
        .await
        .expect("get")
        .expect("stored record");
    assert_eq!(&stored, record);
    assert_eq!(ltm.list(&scope, &[]).await.expect("list").len(), 1);
}

#[tokio::test]
async fn add_policy_stores_the_duplicate_alongside_the_existing_record() {
    let ltm = facade(MemoryEngineConfig::default()).await;
    let scope = scope();

    let first = ltm
        .add(
            &scope,
            NewMemory::semantic("alice prefers dark mode"),
            &AddOptions::default(),
        )
        .await
        .expect("add");
    let first_id = first.added[0].id.clone();

    let second = ltm
        .add(
            &scope,
            NewMemory::semantic("alice prefers dark mode"),
            &add_opts(DedupePolicy::Add),
        )
        .await
        .expect("add duplicate");
    assert_eq!(second.added.len(), 1);
    assert_eq!(second.merged.len(), 0);
    assert_eq!(second.skipped.len(), 0);
    let second_id = second.added[0].id.clone();
    assert_ne!(second_id, first_id);

    let listed = ltm.list(&scope, &[]).await.expect("list");
    assert_eq!(listed.len(), 2);
    let ids: BTreeSet<&str> = listed.iter().map(|record| record.id.as_str()).collect();
    assert_eq!(ids, BTreeSet::from([first_id.as_str(), second_id.as_str()]));
}

#[tokio::test]
async fn a_batch_dedupes_against_stored_and_earlier_batch_records() {
    let ltm = facade(MemoryEngineConfig::default()).await;
    let scope = scope();

    let existing_text = "alice lives in berlin";
    let existing = ltm
        .add(
            &scope,
            NewMemory::semantic(existing_text),
            &AddOptions::default(),
        )
        .await
        .expect("seed");
    let existing_id = existing.added[0].id.clone();
    let unique_text = "bob writes rust code";

    let batch = ltm
        .add_many(
            &scope,
            vec![
                NewMemory::semantic(existing_text),
                NewMemory::semantic(unique_text),
                NewMemory::semantic(existing_text),
            ],
            &AddOptions::default(),
        )
        .await
        .expect("add many");

    // Default `Merge`: dup1 and dup2 fold into the seed, only `unique_text` is new.
    assert_eq!(batch.added.len(), 1);
    assert_eq!(batch.merged.len(), 2);
    assert_eq!(batch.skipped.len(), 0);
    assert_eq!(batch.len(), 3);
    assert_eq!(batch.added[0].content, unique_text);
    assert!(batch.merged.iter().all(|record| record.id == existing_id));

    let listed = ltm.list(&scope, &[]).await.expect("list");
    assert_eq!(listed.len(), 2);
    // The outcome agrees with what is actually stored.
    let outcome_ids: BTreeSet<&str> = batch
        .added
        .iter()
        .chain(batch.merged.iter())
        .map(|record| record.id.as_str())
        .collect();
    let listed_ids: BTreeSet<&str> = listed.iter().map(|record| record.id.as_str()).collect();
    assert_eq!(outcome_ids, listed_ids);
}

#[tokio::test]
async fn blank_content_is_rejected_and_forbidden_variables_do_not_gate_content() {
    let config = MemoryEngineConfig {
        forbidden_variables: vec!["api_key".to_owned()],
        ..Default::default()
    };
    let ltm = facade(config).await;
    let scope = scope();

    for blank in ["", "   ", "\n\t "] {
        let error = ltm
            .add(&scope, NewMemory::semantic(blank), &AddOptions::default())
            .await
            .expect_err("blank content must be rejected");
        assert!(
            matches!(error, Error::Validation(_)),
            "{blank:?}: {error:?}"
        );
    }
    assert_eq!(ltm.list(&scope, &[]).await.expect("list").len(), 0);

    let error = ltm
        .add_many(
            &scope,
            vec![
                NewMemory::semantic(""),
                NewMemory::semantic("valid content"),
            ],
            &AddOptions::default(),
        )
        .await
        .expect_err("a blank batch item must be rejected");
    assert!(matches!(error, Error::Validation(_)), "{error:?}");
    // The blank item failed validation before anything was written.
    assert_eq!(ltm.list(&scope, &[]).await.expect("list").len(), 0);

    // `forbidden_variables` only gates extracted variable names, never record content.
    let outcome = ltm
        .add(
            &scope,
            NewMemory::semantic("alice stores the api_key in the vault"),
            &AddOptions::default(),
        )
        .await
        .expect("forbidden variables must not gate record content");
    assert_eq!(outcome.added.len(), 1);

    // A blank incoming text after a stored record neither merges nor panics.
    let error = ltm
        .add(&scope, NewMemory::semantic("  "), &AddOptions::default())
        .await
        .expect_err("blank duplicate add");
    assert!(matches!(error, Error::Validation(_)), "{error:?}");
    assert_eq!(ltm.list(&scope, &[]).await.expect("list").len(), 1);
}

#[tokio::test]
async fn texts_below_the_similarity_thresholds_stay_separate() {
    let ltm = facade(MemoryEngineConfig::default()).await;
    let scope = scope();

    let first_text = "alice prefers dark mode";
    let second_text = "the deployment failed yesterday";
    // Without an embedding model only the exact/Jaccard rules apply; these texts share
    // no token, so Jaccard is 0.0, far below the default 0.75 threshold.
    assert_eq!(
        openjiuwen_memory::recall::jaccard(first_text, second_text),
        0.0
    );

    let first = ltm
        .add(
            &scope,
            NewMemory::semantic(first_text),
            &AddOptions::default(),
        )
        .await
        .expect("add first");
    let first_id = first.added[0].id.clone();

    let second = ltm
        .add(
            &scope,
            NewMemory::semantic(second_text),
            &AddOptions::default(),
        )
        .await
        .expect("add second");
    assert_eq!(second.added.len(), 1);
    assert_eq!(second.merged.len(), 0);
    assert_ne!(second.added[0].id, first_id);

    let listed = ltm.list(&scope, &[]).await.expect("list");
    assert_eq!(listed.len(), 2);
    assert!(listed.iter().any(|record| record.content == first_text));
    assert!(listed.iter().any(|record| record.content == second_text));
}

// --- degraded dedupe and partial-update findings -------------------------------
//
// Two write-path failures used to be swallowed: a failed duplicate candidate scan
// looked exactly like "no duplicates", and a failed re-index after a kind change left
// the new content persisted but unsearchable. Both are now caller-visible, and these
// tests pin that with stores whose one broken half is armed on demand.

/// Builds an in-memory store bundle whose vector backend is `vector`.
fn stores_with(vector: Arc<dyn VectorStore>) -> MemoryStores {
    let mut stores = MemoryStores::in_memory();
    stores.vector = vector;
    stores
}

/// A vector store whose `search` can be armed to fail, so the dedupe candidate scan —
/// the only recall that runs inside a write — fails deterministically.
///
/// Indexing (`add_docs`) keeps working, which is what a store with a broken *read* path
/// looks like: writes land, duplicate detection cannot look.
#[derive(Clone)]
struct ArmedSearchStore {
    inner: Arc<InMemoryVectorStore>,
    mode: Arc<AtomicU8>,
}

impl ArmedSearchStore {
    /// Creates an unarmed store (`search` delegates to the in-memory backend).
    fn new() -> Self {
        Self {
            inner: Arc::new(InMemoryVectorStore::new()),
            mode: Arc::new(AtomicU8::new(0)),
        }
    }

    /// Makes `search` fail with a backend error.
    fn fail_search(&self) {
        self.mode.store(1, Ordering::SeqCst);
    }

    /// Makes `search` fail the way a cancelled call does.
    fn cancel_search(&self) {
        self.mode.store(2, Ordering::SeqCst);
    }
}

#[async_trait]
impl VectorStore for ArmedSearchStore {
    async fn create_collection(&self, collection: &str, schema: CollectionSchema) -> Result<()> {
        self.inner.create_collection(collection, schema).await
    }

    async fn delete_collection(&self, collection: &str) -> Result<()> {
        self.inner.delete_collection(collection).await
    }

    async fn collection_exists(&self, collection: &str) -> Result<bool> {
        self.inner.collection_exists(collection).await
    }

    async fn get_schema(&self, collection: &str) -> Result<Option<CollectionSchema>> {
        self.inner.get_schema(collection).await
    }

    async fn add_docs(&self, collection: &str, docs: Vec<Map<String, Value>>) -> Result<usize> {
        self.inner.add_docs(collection, docs).await
    }

    async fn search(
        &self,
        collection: &str,
        query: &VectorQuery,
    ) -> Result<Vec<VectorSearchResult>> {
        match self.mode.load(Ordering::SeqCst) {
            1 => Err(Error::llm("vector index is unavailable")),
            2 => Err(Error::Cancelled),
            _ => self.inner.search(collection, query).await,
        }
    }

    async fn delete_docs_by_ids(&self, collection: &str, ids: &[String]) -> Result<usize> {
        self.inner.delete_docs_by_ids(collection, ids).await
    }

    async fn delete_docs_by_filters(&self, collection: &str, filters: &Value) -> Result<usize> {
        self.inner.delete_docs_by_filters(collection, filters).await
    }

    async fn list_collection_names(&self) -> Result<Vec<String>> {
        self.inner.list_collection_names().await
    }
}

/// A vector store that fails `add_docs` for one armed collection only, so a single
/// kind's re-index fails while every other collection keeps working.
#[derive(Clone)]
struct FailingCollectionStore {
    inner: Arc<InMemoryVectorStore>,
    failing: Arc<Mutex<Option<String>>>,
}

impl FailingCollectionStore {
    /// Creates a store that accepts every write.
    fn new() -> Self {
        Self {
            inner: Arc::new(InMemoryVectorStore::new()),
            failing: Arc::new(Mutex::new(None)),
        }
    }

    /// Makes every `add_docs` call for `collection` fail with a backend error.
    fn fail(&self, collection: String) {
        *self.failing.lock().expect("lock") = Some(collection);
    }
}

#[async_trait]
impl VectorStore for FailingCollectionStore {
    async fn create_collection(&self, collection: &str, schema: CollectionSchema) -> Result<()> {
        self.inner.create_collection(collection, schema).await
    }

    async fn delete_collection(&self, collection: &str) -> Result<()> {
        self.inner.delete_collection(collection).await
    }

    async fn collection_exists(&self, collection: &str) -> Result<bool> {
        self.inner.collection_exists(collection).await
    }

    async fn get_schema(&self, collection: &str) -> Result<Option<CollectionSchema>> {
        self.inner.get_schema(collection).await
    }

    async fn add_docs(&self, collection: &str, docs: Vec<Map<String, Value>>) -> Result<usize> {
        let armed = self
            .failing
            .lock()
            .expect("lock")
            .as_deref()
            .is_some_and(|failing| failing == collection);
        if armed {
            return Err(Error::llm(format!(
                "vector collection `{collection}` rejected the write"
            )));
        }
        self.inner.add_docs(collection, docs).await
    }

    async fn search(
        &self,
        collection: &str,
        query: &VectorQuery,
    ) -> Result<Vec<VectorSearchResult>> {
        self.inner.search(collection, query).await
    }

    async fn delete_docs_by_ids(&self, collection: &str, ids: &[String]) -> Result<usize> {
        self.inner.delete_docs_by_ids(collection, ids).await
    }

    async fn delete_docs_by_filters(&self, collection: &str, filters: &Value) -> Result<usize> {
        self.inner.delete_docs_by_filters(collection, filters).await
    }

    async fn list_collection_names(&self) -> Result<Vec<String>> {
        self.inner.list_collection_names().await
    }
}

/// Asserts `outcome` is complete (nothing degraded) and returns its single added id.
fn one_added(outcome: &AddOutcome) -> String {
    assert_eq!(
        outcome.degraded, None,
        "a write over a working store must not degrade"
    );
    assert_eq!(outcome.added.len(), 1);
    outcome.added[0].id.clone()
}

#[tokio::test]
async fn a_failed_candidate_scan_is_reported_and_the_record_is_still_stored() {
    let vector = ArmedSearchStore::new();
    let ltm = LongTermMemory::builder(MemoryEngineConfig::default())
        .stores(stores_with(Arc::new(vector.clone())))
        .embedding(Arc::new(HashEmbedding::new(16)))
        .build()
        .expect("facade");
    let scope = scope();
    let first = ltm
        .add(
            &scope,
            NewMemory::semantic("Alice prefers dark mode"),
            &AddOptions::default(),
        )
        .await
        .expect("add");
    let first_id = one_added(&first);

    // The read half of the vector store breaks: the duplicate candidate scan cannot
    // look, but writes still land.
    vector.fail_search();
    let outcome = ltm
        .add(
            &scope,
            NewMemory::semantic("Bob prefers light mode"),
            &AddOptions::default(),
        )
        .await
        .expect("degraded add");
    let reason = outcome
        .degraded
        .clone()
        .expect("a failed candidate scan must be caller-visible, not silently empty");
    assert!(
        reason.contains("duplicate candidate scan failed"),
        "{reason}"
    );
    assert!(reason.contains("vector index is unavailable"), "{reason}");
    assert_eq!(
        reason.matches(&outcome.added[0].id).count(),
        1,
        "the reason must name the unchecked record: {reason}"
    );
    assert_eq!(outcome.added.len(), 1);
    assert_eq!(outcome.merged, Vec::new());

    let stored = ltm
        .get(&scope, &outcome.added[0].id)
        .await
        .expect("get")
        .expect("stored");
    assert_eq!(stored.content, "Bob prefers light mode");
    let listed = ltm.list(&scope, &[]).await.expect("list");
    assert_eq!(
        listed.len(),
        2,
        "the unchecked record is stored beside the existing one"
    );
    assert!(listed.iter().any(|record| record.id == first_id));
}

#[tokio::test]
async fn remember_reports_a_degraded_write_and_still_stores_the_fact() {
    let vector = ArmedSearchStore::new();
    let ltm = LongTermMemory::builder(MemoryEngineConfig::default())
        .stores(stores_with(Arc::new(vector.clone())))
        .embedding(Arc::new(HashEmbedding::new(16)))
        .build()
        .expect("facade");
    let scope = scope();

    // One stored user-profile record creates the vector collection the scan reads, so the
    // armed failure is reached instead of the scan short-circuiting on "no collection".
    ltm.add(
        &scope,
        NewMemory::new("Alice prefers tea", MemoryType::UserProfile),
        &AddOptions::default(),
    )
    .await
    .expect("add");

    // The read half of the vector store is broken before the `remember` write, so the
    // analyzer's fact is stored without any duplicate check.
    vector.fail_search();
    let conversation = ExtractionInput::from_text("I prefer dark mode.");
    let report = ltm
        .remember(&scope, &conversation, &AddOptions::default())
        .await
        .expect("remember");
    assert_eq!(report.extracted, 1, "{report:?}");
    assert_eq!(report.added, 1, "{report:?}");
    assert_eq!(report.merged, 0, "{report:?}");
    assert_eq!(report.skipped, 0, "{report:?}");
    assert_eq!(report.gated, None, "{report:?}");

    // The write's degradation is the report's: an unchecked fact write must not look
    // like a clean one just because the analysis pass was clean.
    let reason = report
        .degraded
        .as_deref()
        .expect("a fact stored without a duplicate check must be caller-visible");
    assert!(
        reason.contains("duplicate candidate scan failed"),
        "{reason}"
    );
    assert!(reason.contains("vector index is unavailable"), "{reason}");

    // The fact is still stored: the failed scan degrades the check, never the write.
    let stored = ltm.list(&scope, &[]).await.expect("list");
    assert_eq!(stored.len(), 2, "the unchecked fact is still stored");
    let unchecked = stored
        .iter()
        .find(|record| record.content.contains("dark mode"))
        .expect("the unchecked fact");
    assert_eq!(unchecked.kind, MemoryType::UserProfile);
}

#[tokio::test]
async fn a_cancelled_candidate_scan_aborts_instead_of_degrading() {
    let vector = ArmedSearchStore::new();
    let ltm = LongTermMemory::builder(MemoryEngineConfig::default())
        .stores(stores_with(Arc::new(vector.clone())))
        .embedding(Arc::new(HashEmbedding::new(16)))
        .build()
        .expect("facade");
    let scope = scope();
    let first = ltm
        .add(
            &scope,
            NewMemory::semantic("Alice prefers dark mode"),
            &AddOptions::default(),
        )
        .await
        .expect("add");
    let first_id = one_added(&first);

    // A cancelled scan is not a degraded write: cancellation must abort.
    vector.cancel_search();
    let error = ltm
        .add(
            &scope,
            NewMemory::semantic("Bob prefers light mode"),
            &AddOptions::default(),
        )
        .await
        .expect_err("cancelled scan");
    assert!(matches!(error, Error::Cancelled), "{error:?}");
    let listed = ltm.list(&scope, &[]).await.expect("list");
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].id, first_id);
}

#[tokio::test]
async fn a_failed_reindex_after_a_kind_change_restores_the_previous_record() {
    let vector = FailingCollectionStore::new();
    let ltm = LongTermMemory::builder(MemoryEngineConfig::default())
        .stores(stores_with(Arc::new(vector.clone())))
        .embedding(Arc::new(HashEmbedding::new(16)))
        .build()
        .expect("facade");
    let scope = scope();
    let added = ltm
        .add(
            &scope,
            NewMemory::semantic("Alice prefers dark mode"),
            &AddOptions::default(),
        )
        .await
        .expect("add");
    let id = one_added(&added);

    // Only the new kind's collection rejects writes, so the kind change deletes the old
    // vector and then fails to index the new one.
    vector.fail(keys::vector_collection(&scope, MemoryType::UserProfile));
    let error = ltm
        .update(
            &scope,
            &id,
            MemoryUpdate::content("Alice prefers light mode").with_kind(MemoryType::UserProfile),
            &CallOptions::new(),
        )
        .await
        .expect_err("re-index failure");
    assert!(matches!(error, Error::Llm(_)), "{error:?}");
    let message = error.to_string();
    assert!(
        message.contains("restored"),
        "the error must name the restored state: {message}"
    );
    assert!(message.contains(&id), "{message}");

    let restored = ltm
        .get(&scope, &id)
        .await
        .expect("get")
        .expect("the previous record is restored, not deleted");
    assert_eq!(restored.content, "Alice prefers dark mode");
    assert_eq!(restored.kind, MemoryType::SemanticMemory);

    // The previous vector was re-indexed, so the record is searchable again.
    let hits = ltm
        .search(
            &scope,
            "Alice prefers dark mode",
            &SearchOptions::default().with_threshold(0.0),
        )
        .await
        .expect("search");
    assert!(
        hits.iter().any(|hit| hit.record.id == id),
        "the restored record must be searchable, got {hits:?}"
    );
}
