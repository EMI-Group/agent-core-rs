//! Behaviour tests for the persisted, embedding-backed experience index.
//!
//! Everything is offline and deterministic: a `tempfile` directory for persistence,
//! `HashEmbedding` for the real retrieval model, a three-axis keyword embedder for exact
//! cosine assertions, explicit timestamps instead of the wall clock, and no assertion on
//! hash-map ordering.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::float_cmp)]
#![allow(clippy::assert_is_empty)]

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_retrieval::{Embedding, HashEmbedding};
use openjiuwen_symphony::error::{Error, ErrorKind, Result};
use openjiuwen_symphony::experience::{
    AddOutcome, ExperienceHit, ExperienceIndex, ExperienceItem, ExperienceRanking,
};

/// Builds an item with the given pattern and skills.
fn item(pattern: &str, skills: &[&str]) -> ExperienceItem {
    ExperienceItem::new(pattern).with_skill_ids(skills.iter().copied())
}

/// The ids of `hits`, in order.
fn hit_ids(hits: &[ExperienceHit]) -> Vec<String> {
    hits.iter().map(|hit| hit.id().to_string()).collect()
}

/// The ids of the index's items, in order.
fn index_ids(index: &ExperienceIndex) -> Vec<String> {
    index.items().into_iter().map(|item| item.id).collect()
}

/// A tiny deterministic embedder: three fixed keyword axes plus a small floor.
///
/// `HashEmbedding` proves a real `openjiuwen-retrieval` model plugs in, but its mean-pooled
/// token hashes make exact score assertions fragile; this fake gives the ranking tests a
/// known geometry (identical text, cosine `1.0`; disjoint text, near zero).
struct KeywordEmbedding;

impl KeywordEmbedding {
    /// Encodes the keyword axes of `text`.
    fn encode(text: &str) -> Vec<f32> {
        let lowered = text.to_lowercase();
        let mut vector = vec![0.1_f32; 3];
        if lowered.contains("deploy") || lowered.contains("kubernetes") {
            vector[0] += 1.0;
        }
        if lowered.contains("summar") || lowered.contains("report") {
            vector[1] += 1.0;
        }
        if lowered.contains("cake") || lowered.contains("chocolate") {
            vector[2] += 1.0;
        }
        vector
    }
}

#[async_trait]
impl Embedding for KeywordEmbedding {
    fn model_name(&self) -> &'static str {
        "keyword-embedding"
    }

    fn dimension_hint(&self) -> Option<usize> {
        Some(3)
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        Ok(Self::encode(text))
    }

    async fn embed_documents(
        &self,
        texts: &[String],
        _batch_size: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|text| Self::encode(text)).collect())
    }
}

/// The keyword embedder plus call counters, proving `add_batch` embeds once.
#[derive(Default)]
struct CountingEmbedding {
    /// How many `embed_documents` calls were made.
    document_calls: AtomicU64,
    /// How many `embed_query` calls were made.
    query_calls: AtomicU64,
}

#[async_trait]
impl Embedding for CountingEmbedding {
    fn model_name(&self) -> &'static str {
        "counting-embedding"
    }

    fn dimension_hint(&self) -> Option<usize> {
        Some(3)
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.query_calls.fetch_add(1, Ordering::SeqCst);
        Ok(KeywordEmbedding::encode(text))
    }

    async fn embed_documents(
        &self,
        texts: &[String],
        _batch_size: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        self.document_calls.fetch_add(1, Ordering::SeqCst);
        Ok(texts
            .iter()
            .map(|text| KeywordEmbedding::encode(text))
            .collect())
    }
}

/// An in-memory index over the keyword embedder.
fn keyword_index() -> ExperienceIndex {
    ExperienceIndex::new(Arc::new(KeywordEmbedding))
}

/// An in-memory index over the counting embedder.
fn counting_index() -> (ExperienceIndex, Arc<CountingEmbedding>) {
    let embedding = Arc::new(CountingEmbedding::default());
    (ExperienceIndex::new(embedding.clone()), embedding)
}

/// Reads one JSON artifact of a persistent index.
async fn read_json(dir: &Path, relative: &str) -> serde_json::Value {
    let text = tokio::fs::read_to_string(dir.join(relative))
        .await
        .expect("read artifact");
    serde_json::from_str(&text).expect("parse artifact")
}

// ---------------------------------------------------------------------------
// CRUD, generated ids, and dedupe
// ---------------------------------------------------------------------------

#[tokio::test]
async fn add_generates_sequential_ids_and_keeps_explicit_ones() {
    let index = keyword_index();
    assert!(index.is_empty());

    let first = index
        .add(item("Summarize the report", &["summarizer"]))
        .await
        .expect("add first");
    assert_eq!(first, AddOutcome::inserted("exp_0001"));
    let second = index
        .add(item("Deploy the service", &["deployer"]))
        .await
        .expect("add second");
    assert_eq!(second, AddOutcome::inserted("exp_0002"));
    let explicit = index
        .add(item("Bake a cake", &["baking"]).with_id("  custom-1  "))
        .await
        .expect("add explicit");
    assert_eq!(explicit, AddOutcome::inserted("custom-1"));

    assert_eq!(index.count(), 3);
    assert_eq!(index_ids(&index), vec!["custom-1", "exp_0001", "exp_0002"]);
    let stored = index
        .items()
        .into_iter()
        .find(|stored| stored.id == "exp_0001")
        .expect("stored first");
    assert_eq!(stored.query_pattern, "Summarize the report");
    assert_eq!(stored.skill_ids, vec!["summarizer".to_string()]);
    assert_eq!(stored.success_count, 1);
    assert_eq!(
        stored.embedding,
        KeywordEmbedding::encode(&stored.search_text())
    );
}

#[tokio::test]
async fn a_blank_pattern_is_a_validation_error() {
    let index = keyword_index();
    let error = index
        .add(item("   \n ", &["skill"]))
        .await
        .expect_err("blank pattern");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("query_pattern"));
    assert!(index.is_empty());

    let batch = index
        .add_batch(vec![item("fine", &["a"]), item("", &["b"])])
        .await
        .expect_err("blank pattern in a batch");
    assert_eq!(batch.kind(), ErrorKind::Validation);
}

#[tokio::test]
async fn duplicates_merge_across_add_calls() {
    let (index, embedding) = counting_index();
    let first = index
        .add(
            item("Summarize the report", &["summarizer", "formatter"])
                .with_query_examples(["summarize this"])
                .with_last_hit_at(5.0),
        )
        .await
        .expect("add first");
    assert_eq!(first, AddOutcome::inserted("exp_0001"));
    assert_eq!(embedding.query_calls.load(Ordering::SeqCst), 1);

    let duplicate = index
        .add(
            item("  summarize   the REPORT ", &["formatter", "summarizer"])
                .with_query_examples(["summarize this", "summarize that"])
                .with_last_hit_at(9.0),
        )
        .await
        .expect("add duplicate");
    assert_eq!(duplicate, AddOutcome::merged_into("exp_0001"));
    assert_eq!(index.count(), 1);
    let stored = index.items().remove(0);
    assert_eq!(stored.success_count, 2);
    assert_eq!(
        stored.query_examples,
        vec!["summarize this".to_string(), "summarize that".to_string()]
    );
    assert!((stored.last_hit_at - 9.0).abs() < f64::EPSILON);
    // A merge into an already embedded record reuses its vector instead of re-embedding.
    assert_eq!(embedding.query_calls.load(Ordering::SeqCst), 1);

    // The same pattern with a different skill set is a different record.
    let other = index
        .add(item("Summarize the report", &["summarizer"]))
        .await
        .expect("add other skill set");
    assert_eq!(other, AddOutcome::inserted("exp_0002"));
    assert_eq!(index.count(), 2);
}

#[tokio::test]
async fn add_batch_embeds_and_persists_once() {
    let (index, embedding) = counting_index();
    let outcomes = index
        .add_batch(vec![
            item("Summarize the report", &["summarizer"]),
            item("Deploy the service", &["deployer"]),
            item("Summarize the report", &["summarizer"]),
            item("Bake a cake", &["baking"]),
        ])
        .await
        .expect("add batch");

    assert_eq!(embedding.document_calls.load(Ordering::SeqCst), 1);
    assert_eq!(embedding.query_calls.load(Ordering::SeqCst), 0);
    assert_eq!(outcomes.len(), 4);
    assert!(!outcomes[0].merged);
    assert!(outcomes[2].merged);
    assert_eq!(index.count(), 3);
    let merged = index
        .items()
        .into_iter()
        .find(|stored| stored.id == outcomes[2].id)
        .expect("merged item");
    assert_eq!(merged.success_count, 2);

    assert!(index
        .add_batch(Vec::new())
        .await
        .expect("empty batch")
        .is_empty());
}

#[tokio::test]
async fn exist_matches_whole_skill_sets() {
    let index = keyword_index();
    index
        .add(item("Summarize the report", &["a", "b"]))
        .await
        .expect("add a/b");
    index.add(item("Bake a cake", &[])).await.expect("add none");

    assert!(index.exist(&["b", "a"]));
    assert!(index.exist(&["a", "b", "a"]));
    assert!(!index.exist(&["a"]));
    // The skill-less item makes the empty set present, exactly like Python's frozenset.
    assert!(index.exist(&[]));
}

// ---------------------------------------------------------------------------
// Ranking
// ---------------------------------------------------------------------------

#[tokio::test]
async fn search_orders_by_cosine_and_truncates() {
    let index = keyword_index();
    index
        .add(item("Summarize the report", &["summarizer"]))
        .await
        .expect("add summarize");
    index
        .add(item("Deploy the service", &["deployer"]))
        .await
        .expect("add deploy");
    index
        .add(item("Bake a cake", &["baking"]))
        .await
        .expect("add cake");

    let best = index
        .search("summarize the report", &ExperienceRanking::new(1))
        .await
        .expect("search top one");
    assert_eq!(hit_ids(&best), vec!["exp_0001"]);
    assert!((best[0].score - 1.0).abs() < 1e-9);

    let all = index
        .search("summarize the report", &ExperienceRanking::new(0))
        .await
        .expect("search unlimited");
    assert_eq!(hit_ids(&all), vec!["exp_0001", "exp_0002", "exp_0003"]);
    assert!(all[1].score < all[0].score);

    let strict = index
        .search(
            "summarize the report",
            &ExperienceRanking::new(0).with_min_score(0.5),
        )
        .await
        .expect("search with threshold");
    assert_eq!(hit_ids(&strict), vec!["exp_0001"]);
}

#[tokio::test]
async fn top_k_zero_is_unlimited_and_ties_break_on_id() {
    let index = keyword_index();
    for (ordinal, pattern) in ["zeta", "alpha", "beta"].iter().enumerate() {
        index
            .add(item(pattern, &["filler"]).with_id(format!("exp_{:04}", ordinal + 1)))
            .await
            .expect("add");
    }
    let hits = index
        .search("completely unrelated query", &ExperienceRanking::new(0))
        .await
        .expect("search");
    assert_eq!(hits.len(), 3);
    // Every item encodes the same three-axis floor, so the scores tie and the id decides.
    assert_eq!(hit_ids(&hits), vec!["exp_0001", "exp_0002", "exp_0003"]);
    assert!(hits
        .windows(2)
        .all(|pair| (pair[0].score - pair[1].score).abs() < 1e-9));
}

#[tokio::test]
async fn the_threshold_applies_after_truncation() {
    let index = keyword_index();
    index
        .add(item("Summarize the report", &["summarizer"]))
        .await
        .expect("add");
    let truncated = index
        .search(
            "deploy kubernetes",
            &ExperienceRanking::new(1).with_min_score(0.9),
        )
        .await
        .expect("search");
    assert!(truncated.is_empty());
}

#[tokio::test]
async fn search_with_skill_ids_flattens_and_dedupes() {
    let index = keyword_index();
    index
        .add(item("Summarize the report", &["a", "summarizer"]))
        .await
        .expect("add first");
    index
        .add(item("Summarize the text", &["summarizer", "c"]))
        .await
        .expect("add second");

    let skills = index
        .search_with_skill_ids("summarize the report", &ExperienceRanking::new(0))
        .await
        .expect("search skills");
    assert_eq!(skills, vec!["a", "summarizer", "c"]);
}

#[tokio::test]
async fn a_blank_query_falls_back_to_lexical_ranking() {
    let index = keyword_index();
    index
        .add(item("Summarize the report", &["summarizer"]))
        .await
        .expect("add");
    let hits = index
        .search("   ", &ExperienceRanking::new(0))
        .await
        .expect("blank query");
    assert_eq!(hit_ids(&hits), vec!["exp_0001"]);
    assert!(hits[0].score.abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

#[tokio::test]
async fn an_empty_directory_opens_empty_and_then_persists() {
    let dir = tempfile::tempdir().expect("tempdir");
    let index = ExperienceIndex::open(dir.path(), Arc::new(KeywordEmbedding))
        .await
        .expect("open empty");
    assert!(index.is_empty());
    assert_eq!(index.dir(), Some(dir.path()));
    assert_eq!(index.algorithm(), "flat");
    assert_eq!(index.meta().vector_count, 0);
    assert_eq!(index.meta().version, 1);
    assert!(index.meta().embeddings_sha256.is_empty());

    index
        .add(item("Summarize the report", &["summarizer"]))
        .await
        .expect("add");
    assert!(tokio::fs::try_exists(dir.path().join("meta.json"))
        .await
        .expect("stat"));
    assert!(
        tokio::fs::try_exists(dir.path().join("scalar/metadata.jsonl"))
            .await
            .expect("stat")
    );
    assert!(
        tokio::fs::try_exists(dir.path().join("vector/embeddings.json"))
            .await
            .expect("stat")
    );
}

#[tokio::test]
async fn a_missing_directory_is_created_by_the_first_persist() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().join("nested/knowledge");
    let index = ExperienceIndex::open(&root, Arc::new(KeywordEmbedding))
        .await
        .expect("open absent");
    assert!(index.is_empty());
    index
        .add(item("Summarize the report", &["summarizer"]))
        .await
        .expect("add");
    assert!(tokio::fs::try_exists(root.join("vector/embeddings.json"))
        .await
        .expect("stat"));
}

#[tokio::test]
async fn a_full_round_trip_preserves_items_vectors_meta_and_hits() {
    let dir = tempfile::tempdir().expect("tempdir");
    let index = ExperienceIndex::open(dir.path(), Arc::new(KeywordEmbedding))
        .await
        .expect("open");
    for (pattern, skills) in [
        ("Summarize the report", &["summarizer"][..]),
        ("Deploy the service", &["deployer"][..]),
        ("Bake a cake", &["baking"][..]),
    ] {
        index.add(item(pattern, skills)).await.expect("add");
    }
    let query = "summarize the report";
    let before_hits = index
        .search(query, &ExperienceRanking::new(0))
        .await
        .expect("search before");
    let before_meta = index.meta();
    let before_items = index.items();
    assert_eq!(before_meta.vector_count, 3);
    assert_eq!(before_meta.vector_dimension, 3);
    assert_eq!(before_meta.vector_algorithm, "flat");
    assert_eq!(before_meta.embeddings_sha256.len(), 64);
    assert_eq!(before_meta.scalar_sha256.len(), 64);

    let reopened = ExperienceIndex::open(dir.path(), Arc::new(KeywordEmbedding))
        .await
        .expect("reopen");
    assert_eq!(reopened.items(), before_items);
    assert_eq!(reopened.meta(), before_meta);
    assert_eq!(reopened.count(), 3);
    assert_eq!(reopened.algorithm(), "flat");
    let after_hits = reopened
        .search(query, &ExperienceRanking::new(0))
        .await
        .expect("search after");
    assert_eq!(hit_ids(&after_hits), hit_ids(&before_hits));
    assert_eq!(after_hits, before_hits);

    let manifest = read_json(dir.path(), "meta.json").await;
    assert_eq!(manifest["vector_count"], 3);
    assert_eq!(manifest["vector_dimension"], 3);
    assert_eq!(manifest["vector_algorithm"], "flat");
    let vectors = read_json(dir.path(), "vector/embeddings.json").await;
    assert_eq!(vectors["ids"].as_array().expect("ids").len(), 3);
    assert_eq!(vectors["dimension"], 3);
}

#[tokio::test]
async fn generated_ids_resume_past_the_loaded_ids() {
    let dir = tempfile::tempdir().expect("tempdir");
    let index = ExperienceIndex::open(dir.path(), Arc::new(KeywordEmbedding))
        .await
        .expect("open");
    index.add(item("first", &["a"])).await.expect("add first");
    index
        .add(item("second", &["b"]).with_id("exp_0007"))
        .await
        .expect("add second");

    let reopened = ExperienceIndex::open(dir.path(), Arc::new(KeywordEmbedding))
        .await
        .expect("reopen");
    let outcome = reopened
        .add(item("third", &["c"]))
        .await
        .expect("add third");
    assert_eq!(outcome, AddOutcome::inserted("exp_0008"));
}

#[tokio::test]
async fn persist_is_idempotent() {
    let dir = tempfile::tempdir().expect("tempdir");
    let index = ExperienceIndex::open(dir.path(), Arc::new(KeywordEmbedding))
        .await
        .expect("open");
    index
        .add(item("Summarize the report", &["summarizer"]))
        .await
        .expect("add");
    let meta = index.meta();

    let scalar_before = tokio::fs::read(dir.path().join("scalar/metadata.jsonl"))
        .await
        .expect("read scalar");
    let vectors_before = tokio::fs::read(dir.path().join("vector/embeddings.json"))
        .await
        .expect("read vectors");
    index.persist().await.expect("persist again");
    assert_eq!(
        tokio::fs::read(dir.path().join("scalar/metadata.jsonl"))
            .await
            .expect("read scalar"),
        scalar_before
    );
    assert_eq!(
        tokio::fs::read(dir.path().join("vector/embeddings.json"))
            .await
            .expect("read vectors"),
        vectors_before
    );
    assert_eq!(index.meta(), meta);
}

#[tokio::test]
async fn an_in_memory_index_persists_a_manifest_but_no_files() {
    let index = keyword_index();
    index
        .add(item("Summarize the report", &["summarizer"]))
        .await
        .expect("add");
    index.persist().await.expect("persist");
    let meta = index.meta();
    assert_eq!(meta.vector_count, 1);
    assert_eq!(meta.vector_dimension, 3);
    assert_eq!(meta.embeddings_sha256.len(), 64);
    assert!(index.dir().is_none());
}

#[tokio::test]
async fn remove_persists_and_reports_whether_the_id_existed() {
    let dir = tempfile::tempdir().expect("tempdir");
    let index = ExperienceIndex::open(dir.path(), Arc::new(KeywordEmbedding))
        .await
        .expect("open");
    index.add(item("first", &["a"])).await.expect("add first");
    index.add(item("second", &["b"])).await.expect("add second");
    assert!(index.remove("exp_0001").await.expect("remove"));
    assert!(!index.remove("exp_0001").await.expect("remove again"));
    assert_eq!(index.count(), 1);

    let reopened = ExperienceIndex::open(dir.path(), Arc::new(KeywordEmbedding))
        .await
        .expect("reopen");
    assert_eq!(index_ids(&reopened), vec!["exp_0002"]);
    assert_eq!(reopened.meta().vector_count, 1);
}

// ---------------------------------------------------------------------------
// Corrupt directories are rejected, not repaired
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_missing_artifact_is_rejected() {
    let dir = tempfile::tempdir().expect("tempdir");
    let index = ExperienceIndex::open(dir.path(), Arc::new(KeywordEmbedding))
        .await
        .expect("open");
    index.add(item("first", &["a"])).await.expect("add");
    tokio::fs::remove_file(dir.path().join("vector/embeddings.json"))
        .await
        .expect("remove artifact");

    let error = ExperienceIndex::open(dir.path(), Arc::new(KeywordEmbedding))
        .await
        .expect_err("missing artifact");
    assert_eq!(error.kind(), ErrorKind::External);
    assert!(error.message().contains("embeddings.json"));
}

#[tokio::test]
async fn a_digest_mismatch_is_rejected_and_names_the_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    let index = ExperienceIndex::open(dir.path(), Arc::new(KeywordEmbedding))
        .await
        .expect("open");
    index.add(item("first", &["a"])).await.expect("add");
    let path = dir.path().join("scalar/metadata.jsonl");
    let mut text = tokio::fs::read_to_string(&path).await.expect("read");
    text.push_str("{\"id\":\"exp_0009\",\"query_pattern\":\"tampered\"}\n");
    tokio::fs::write(&path, text).await.expect("write");

    let error = ExperienceIndex::open(dir.path(), Arc::new(KeywordEmbedding))
        .await
        .expect_err("digest mismatch");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("metadata.jsonl"));
    assert!(error.message().contains("sha256 mismatch"));
}

#[tokio::test]
async fn a_vector_count_mismatch_is_rejected_and_names_the_manifest() {
    let dir = tempfile::tempdir().expect("tempdir");
    let index = ExperienceIndex::open(dir.path(), Arc::new(KeywordEmbedding))
        .await
        .expect("open");
    index.add(item("first", &["a"])).await.expect("add");
    let path = dir.path().join("meta.json");
    let mut manifest = read_json(dir.path(), "meta.json").await;
    manifest["vector_count"] = serde_json::json!(9);
    tokio::fs::write(&path, serde_json::to_vec_pretty(&manifest).expect("encode"))
        .await
        .expect("write");

    let error = ExperienceIndex::open(dir.path(), Arc::new(KeywordEmbedding))
        .await
        .expect_err("count mismatch");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("meta.json"));
    assert!(error.message().contains("vector_count 9"));
}

#[tokio::test]
async fn a_dimension_mismatch_is_rejected_when_the_manifest_disagrees() {
    let dir = tempfile::tempdir().expect("tempdir");
    let index = ExperienceIndex::open(dir.path(), Arc::new(KeywordEmbedding))
        .await
        .expect("open");
    index.add(item("first", &["a"])).await.expect("add");
    let path = dir.path().join("meta.json");
    let mut manifest = read_json(dir.path(), "meta.json").await;
    manifest["vector_dimension"] = serde_json::json!(2);
    tokio::fs::write(&path, serde_json::to_vec_pretty(&manifest).expect("encode"))
        .await
        .expect("write");

    let error = ExperienceIndex::open(dir.path(), Arc::new(KeywordEmbedding))
        .await
        .expect_err("dimension mismatch");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("embeddings.json"));
    assert!(error.message().contains("dimension"));
}

#[tokio::test]
async fn adding_a_vector_from_another_model_is_rejected() {
    let dir = tempfile::tempdir().expect("tempdir");
    let index = ExperienceIndex::open(dir.path(), Arc::new(HashEmbedding::new(8)))
        .await
        .expect("open");
    index.add(item("first", &["a"])).await.expect("add");

    // The stored vectors keep their dimension, so loading with another model still works…
    let reopened = ExperienceIndex::open(dir.path(), Arc::new(HashEmbedding::new(4)))
        .await
        .expect("reopen");
    assert_eq!(reopened.count(), 1);
    // …but a new item from the other vector space cannot be stored next to them.
    let error = reopened
        .add(item("second", &["b"]))
        .await
        .expect_err("dimension mismatch");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("embedding"));
    assert_eq!(reopened.count(), 1);
}

// ---------------------------------------------------------------------------
// Determinism and algorithms
// ---------------------------------------------------------------------------

#[tokio::test]
async fn hash_embedding_search_is_deterministic() {
    let build = || {
        let index = ExperienceIndex::new(Arc::new(HashEmbedding::new(8)));
        (index, item("Summarize the report", &["summarizer"]))
    };
    let (first, first_item) = build();
    let (second, second_item) = build();
    first.add(first_item).await.expect("add first");
    second.add(second_item).await.expect("add second");
    first
        .add(item("Deploy the service", &["deployer"]))
        .await
        .expect("add deploy one");
    second
        .add(item("Deploy the service", &["deployer"]))
        .await
        .expect("add deploy two");

    let ranking = ExperienceRanking::new(0);
    let first_hits = first
        .search("summarize the report", &ranking)
        .await
        .expect("search first");
    let repeated = first
        .search("summarize the report", &ranking)
        .await
        .expect("search again");
    let second_hits = second
        .search("summarize the report", &ranking)
        .await
        .expect("search second");
    assert_eq!(first_hits, repeated);
    assert_eq!(first_hits, second_hits);
    assert_eq!(hit_ids(&first_hits), vec!["exp_0001", "exp_0002"]);
}

#[tokio::test]
async fn the_flat_algorithm_is_case_insensitive() {
    let dir = tempfile::tempdir().expect("tempdir");
    for raw in ["flat", "Flat", " FLAT "] {
        let index =
            ExperienceIndex::open_with_algorithm(dir.path(), Arc::new(KeywordEmbedding), raw)
                .await
                .expect("flat");
        assert_eq!(index.algorithm(), "flat");
    }
}

#[tokio::test]
async fn ivf_algorithms_are_unsupported_and_unknown_ones_are_validation_errors() {
    let dir = tempfile::tempdir().expect("tempdir");
    for raw in ["ivf", "IVFPQ", "IVF4096,PQ16"] {
        let error =
            ExperienceIndex::open_with_algorithm(dir.path(), Arc::new(HashEmbedding::new(8)), raw)
                .await
                .expect_err("ivf");
        assert_eq!(error.kind(), ErrorKind::Unsupported, "{raw}");
        assert!(error.message().contains("faiss IVF"), "{raw}");
    }

    let unknown =
        ExperienceIndex::open_with_algorithm(dir.path(), Arc::new(HashEmbedding::new(8)), "hnsw")
            .await
            .expect_err("hnsw");
    assert_eq!(unknown.kind(), ErrorKind::Validation);
    assert!(unknown.message().contains("vector_algorithm"));

    let blank =
        ExperienceIndex::open_with_algorithm(dir.path(), Arc::new(HashEmbedding::new(8)), "  ")
            .await
            .expect_err("blank");
    assert_eq!(blank.kind(), ErrorKind::Validation);
    assert!(blank.message().contains("vector_algorithm"));

    // None of the rejected openings wrote an artifact.
    assert!(!tokio::fs::try_exists(dir.path().join("meta.json"))
        .await
        .expect("stat"));
}

#[tokio::test]
async fn a_failed_persist_leaves_the_catalog_unchanged() {
    let dir = tempfile::tempdir().expect("tempdir");
    let index = ExperienceIndex::open(dir.path(), Arc::new(KeywordEmbedding))
        .await
        .expect("open");
    index.add(item("first", &["a"])).await.expect("add");
    assert_eq!(index.count(), 1);

    // Tear the directory down so the next publish cannot write, then add again.
    tokio::fs::remove_dir_all(dir.path())
        .await
        .expect("remove dir");
    tokio::fs::write(dir.path(), b"not a directory")
        .await
        .expect("occupy the path");
    let error: Error = index
        .add(item("second", &["b"]))
        .await
        .expect_err("persist must fail");
    assert_eq!(error.kind(), ErrorKind::External);
    assert_eq!(index.count(), 1);
    assert_eq!(index_ids(&index), vec!["exp_0001"]);
}
