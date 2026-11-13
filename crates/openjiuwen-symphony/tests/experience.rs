//! Behaviour tests for the experience bank: generated ids, deterministic dedupe,
//! lexical and embedding ranking, skill filters, and JSON-file persistence.
//!
//! Everything is offline and deterministic: the bank's in-memory store or a
//! `tempfile`-backed document, `HashEmbedding` for the semantic path, explicit
//! timestamps instead of the wall clock, and no assertion on hash-map ordering.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::float_cmp)]
#![allow(clippy::assert_is_empty)]

use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_retrieval::{Embedding, HashEmbedding};
use openjiuwen_symphony::error::ErrorKind;
use openjiuwen_symphony::experience::{
    AddOutcome, ExperienceBank, ExperienceHit, ExperienceItem, ExperienceRanking, SearchOptions,
};

/// Builds an item with the given pattern and skills.
fn item(pattern: &str, skills: &[&str]) -> ExperienceItem {
    ExperienceItem::new(pattern).with_skill_ids(skills.iter().copied())
}

/// The ids of `hits`, in order.
fn hit_ids(hits: &[ExperienceHit]) -> Vec<String> {
    hits.iter().map(|hit| hit.id().to_string()).collect()
}

/// The ids of the bank's items, in order.
fn bank_ids(bank: &ExperienceBank) -> Vec<String> {
    bank.list().into_iter().map(|item| item.id).collect()
}

/// A tiny deterministic embedder: three fixed keyword axes plus a small floor.
///
/// `HashEmbedding` proves a real `openjiuwen-retrieval` model plugs in, but its
/// mean-pooled token hashes make exact score assertions fragile; this fake gives the
/// ranking test a known geometry (identical text, cosine `1.0`; disjoint text, near
/// zero).
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

    async fn embed_query(&self, text: &str) -> openjiuwen_symphony::Result<Vec<f32>> {
        Ok(Self::encode(text))
    }

    async fn embed_documents(
        &self,
        texts: &[String],
        _batch_size: Option<usize>,
    ) -> openjiuwen_symphony::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|text| Self::encode(text)).collect())
    }
}

// ---------------------------------------------------------------------------
// CRUD and generated ids
// ---------------------------------------------------------------------------

#[tokio::test]
async fn add_generates_sequential_ids_and_keeps_explicit_ones() {
    let bank = ExperienceBank::new();
    assert!(bank.is_empty());

    let first = bank
        .add(item("Summarize the report", &["summarizer"]))
        .await
        .expect("add first");
    assert_eq!(first, AddOutcome::inserted("exp_0001"));
    let second = bank
        .add(item("Translate text to French", &["translator"]))
        .await
        .expect("add second");
    assert_eq!(second, AddOutcome::inserted("exp_0002"));
    let explicit = bank
        .add(item("Bake a cake", &["baking"]).with_id("custom-1"))
        .await
        .expect("add explicit");
    assert_eq!(explicit, AddOutcome::inserted("custom-1"));

    assert_eq!(bank.len(), 3);
    assert_eq!(bank_ids(&bank), vec!["custom-1", "exp_0001", "exp_0002"]);

    let stored = bank.get("exp_0001").expect("stored first");
    assert_eq!(stored.query_pattern, "Summarize the report");
    assert_eq!(stored.skill_ids, vec!["summarizer".to_string()]);
    assert_eq!(stored.success_count, 1);
    assert!(bank.get("absent").is_none());

    // An explicit id is trimmed, and a generated id skips an id that is already taken.
    let pinned = bank
        .add(item("Pinned pattern", &["pinned"]).with_id("  exp_0003  "))
        .await
        .expect("add pinned");
    assert_eq!(pinned.id, "exp_0003");
    let next = bank
        .add(item("Another pattern", &["other"]))
        .await
        .expect("add next");
    assert_eq!(next.id, "exp_0004");
}

#[tokio::test]
async fn add_rejects_a_blank_pattern() {
    let bank = ExperienceBank::new();
    let error = bank
        .add(ExperienceItem::new("   \n\t "))
        .await
        .expect_err("blank pattern must be rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("query_pattern"));
    assert!(bank.is_empty());
}

#[tokio::test]
async fn add_batch_returns_one_outcome_per_item() {
    let bank = ExperienceBank::new();
    let outcomes = bank
        .add_batch(vec![
            item("Summarize the report", &["summarizer"]),
            item("Translate text", &["translator"]),
            item("summarize   the report", &["summarizer"]),
        ])
        .await
        .expect("add batch");

    assert_eq!(outcomes.len(), 3);
    assert_eq!(outcomes[0], AddOutcome::inserted("exp_0001"));
    assert_eq!(outcomes[1], AddOutcome::inserted("exp_0002"));
    assert_eq!(outcomes[2], AddOutcome::merged_into("exp_0001"));
    assert_eq!(bank.len(), 2);
    let merged = bank.get("exp_0001").expect("merged item");
    assert_eq!(merged.success_count, 2);
}

#[tokio::test]
async fn remove_and_clear_handle_the_catalog_and_the_store() {
    let bank = ExperienceBank::new();
    bank.add(item("First pattern", &["a"])).await.expect("add");
    bank.add(item("Second pattern", &["b"])).await.expect("add");

    assert!(bank.remove("exp_0001").await.expect("remove"));
    assert!(!bank.remove("exp_0001").await.expect("remove again"));
    assert!(!bank.remove("absent").await.expect("remove absent"));
    assert_eq!(bank_ids(&bank), vec!["exp_0002"]);

    bank.clear().await.expect("clear");
    assert!(bank.is_empty());
    assert_eq!(bank.list().len(), 0);
    assert!(bank.store().list().await.expect("store list").is_empty());

    // The generated-id counter is not reset, so a cleared id is never reused.
    let after_clear = bank
        .add(item("Third pattern", &["c"]))
        .await
        .expect("add after clear");
    assert_eq!(after_clear.id, "exp_0003");
}

#[tokio::test]
async fn exists_for_skills_compares_skill_sets() {
    let bank = ExperienceBank::new();
    bank.add(item("Summarize the report", &["summarizer", "report"]))
        .await
        .expect("add");
    bank.add(item("Just a pattern", &[]))
        .await
        .expect("add empty");

    assert!(bank.exists_for_skills(&["summarizer".to_string(), "report".to_string()]));
    assert!(bank.exists_for_skills(&["report".to_string(), "summarizer".to_string()]));
    assert!(bank.exists_for_skills(&[]));
    assert!(!bank.exists_for_skills(&["summarizer".to_string()]));
    assert!(!bank.exists_for_skills(&["unknown".to_string()]));
}

// ---------------------------------------------------------------------------
// Dedupe
// ---------------------------------------------------------------------------

#[tokio::test]
async fn dedupe_merges_the_same_pattern_and_skill_set() {
    let bank = ExperienceBank::new();
    let first = bank
        .add(
            ExperienceItem::new("Summarize   the Report")
                .with_query_examples(["summarize a report", "summarize the doc"])
                .with_skill_ids(["summarizer", "report"])
                .with_success_count(1)
                .with_last_hit_at(10.0),
        )
        .await
        .expect("add first");
    assert_eq!(first, AddOutcome::inserted("exp_0001"));

    let merged = bank
        .add(
            ExperienceItem::new("summarize the report")
                .with_query_examples(["summarize the doc", "summarise it"])
                .with_skill_ids(["report", "summarizer"])
                .with_success_count(3)
                .with_last_hit_at(25.0),
        )
        .await
        .expect("add duplicate");

    assert_eq!(merged, AddOutcome::merged_into("exp_0001"));
    assert_eq!(bank.len(), 1);
    let stored = bank.get("exp_0001").expect("merged item");
    assert_eq!(stored.success_count, 4);
    assert_eq!(
        stored.query_examples,
        vec!["summarize a report", "summarize the doc", "summarise it"]
    );
    assert_eq!(stored.last_hit_at, 25.0);
    assert_eq!(stored.query_pattern, "Summarize   the Report");
}

#[tokio::test]
async fn dedupe_keeps_different_skill_sets_separate_and_fills_embeddings() {
    let bank = ExperienceBank::new();
    bank.add(
        ExperienceItem::new("Summarize the report")
            .with_skill_ids(["summarizer"])
            .with_last_hit_at(5.0),
    )
    .await
    .expect("add first");
    let separate = bank
        .add(item("summarize the report", &["summarizer", "report"]))
        .await
        .expect("add different skills");
    assert!(!separate.merged);
    assert_eq!(separate.id, "exp_0002");
    assert_eq!(bank.len(), 2);

    // A duplicate arriving with the only embedding fills the stored one.
    let merged = bank
        .add(
            ExperienceItem::new("summarize the report")
                .with_skill_ids(["summarizer"])
                .with_embedding(vec![0.5, -0.5]),
        )
        .await
        .expect("add duplicate with embedding");
    assert!(merged.merged);
    assert_eq!(
        bank.get("exp_0001").expect("stored").embedding,
        vec![0.5, -0.5]
    );
    // The later timestamp wins, and an earlier one never moves it backwards.
    assert_eq!(bank.get("exp_0001").expect("stored").last_hit_at, 5.0);
}

// ---------------------------------------------------------------------------
// Lexical ranking and filters
// ---------------------------------------------------------------------------

#[tokio::test]
async fn lexical_search_ranks_the_best_pattern_first() {
    let bank = ExperienceBank::new();
    bank.add(item("summarize a document", &["summarizer"]))
        .await
        .expect("add");
    bank.add(item("translate text to french", &["translator"]))
        .await
        .expect("add");
    bank.add(item("document indexing", &["indexer"]))
        .await
        .expect("add");

    let hits = bank.search("summarize document", 0).await.expect("search");
    assert_eq!(hit_ids(&hits), vec!["exp_0001", "exp_0003", "exp_0002"]);
    assert_eq!(hits[0].score, 1.0);
    // `document` is a pattern term of exp_0003, so half of the query is covered.
    assert_eq!(hits[1].score, 0.5);
    assert_eq!(hits[2].score, 0.0);

    let limited = bank.search("summarize document", 1).await.expect("search");
    assert_eq!(hit_ids(&limited), vec!["exp_0001"]);

    // A blank query scores everything at zero and falls back to id order.
    let empty_query = bank.search("   ", 0).await.expect("search blank");
    assert_eq!(
        hit_ids(&empty_query),
        vec!["exp_0001", "exp_0002", "exp_0003"]
    );
    assert!(empty_query.iter().all(|hit| hit.score == 0.0));

    let missing = bank
        .search("nothing matches this", 5)
        .await
        .expect("search");
    assert_eq!(missing.len(), 3);
    assert!(missing.iter().all(|hit| hit.score == 0.0));
}

#[tokio::test]
async fn lexical_search_ties_break_on_id_and_are_repeatable() {
    let bank = ExperienceBank::new();
    bank.add(item("export report", &["a"]).with_id("exp_0002"))
        .await
        .expect("add");
    bank.add(item("export report", &["b"]).with_id("exp_0001"))
        .await
        .expect("add");

    let first = bank.search("export report", 0).await.expect("search");
    let second = bank.search("export report", 0).await.expect("search again");
    assert_eq!(hit_ids(&first), vec!["exp_0001", "exp_0002"]);
    assert_eq!(first, second);
    assert!(first.iter().all(|hit| hit.score == 1.0));
}

#[tokio::test]
async fn search_options_filter_on_skills() {
    let bank = ExperienceBank::new();
    bank.add(item("summarize a document", &["summarizer", "report"]))
        .await
        .expect("add");
    bank.add(item("summarize a document", &["summarizer"]).with_id("exp_0002"))
        .await
        .expect("add");

    let both = SearchOptions::new(0).with_required_skills(["summarizer", "report"]);
    let hits = bank
        .search_with_options("summarize document", &both)
        .await
        .expect("search");
    assert_eq!(hit_ids(&hits), vec!["exp_0001"]);

    let either = both.clone().with_require_all_skills(false);
    let hits = bank
        .search_with_options("summarize document", &either)
        .await
        .expect("search");
    assert_eq!(hits.len(), 2);

    let excluded = SearchOptions::new(0).with_excluded_skills(["report"]);
    let hits = bank
        .search_with_options("summarize document", &excluded)
        .await
        .expect("search");
    assert_eq!(hit_ids(&hits), vec!["exp_0002"]);

    let ranking = ExperienceRanking::new(1).with_min_score(0.9);
    let hits = bank
        .search_with_ranking("summarize document", &ranking)
        .await
        .expect("search");
    assert_eq!(hits.len(), 1);
    assert!(hits[0].score >= 0.9);

    let strict = SearchOptions::new(0).with_min_score(2.0);
    assert!(bank
        .search_with_options("summarize document", &strict)
        .await
        .expect("search")
        .is_empty());
}

#[tokio::test]
async fn search_with_skill_ids_flattens_and_dedupes() {
    let bank = ExperienceBank::new();
    bank.add(item("summarize the report", &["sum", "report"]))
        .await
        .expect("add");
    bank.add(item("summarize the report", &["sum", "brief"]))
        .await
        .expect("add");

    let skills = bank
        .search_with_skill_ids("summarize the report", 2)
        .await
        .expect("search");
    assert_eq!(skills, vec!["sum", "report", "brief"]);

    let single = bank
        .search_with_skill_ids("summarize the report", 1)
        .await
        .expect("search");
    assert_eq!(single, vec!["sum", "report"]);
}

// ---------------------------------------------------------------------------
// Embedding-backed ranking
// ---------------------------------------------------------------------------

#[tokio::test]
async fn embedding_search_ranks_by_cosine_and_applies_the_threshold() {
    let bank = ExperienceBank::new().with_embedding(Arc::new(KeywordEmbedding));
    bank.add(item("deploy kubernetes cluster", &["k8s"]))
        .await
        .expect("add");
    bank.add(item("bake chocolate cake", &["baking"]))
        .await
        .expect("add");

    let hits = bank
        .search("deploy kubernetes cluster", 0)
        .await
        .expect("search");
    assert_eq!(hit_ids(&hits), vec!["exp_0001", "exp_0002"]);
    let identical = hits[0].score;
    assert!((identical - 1.0).abs() < 1e-9, "score was {identical}");
    assert!(hits[1].score < 0.3, "score was {}", hits[1].score);
    // Items without a stored embedding are embedded on demand for the search.
    assert_eq!(hits[0].item.embedding.len(), 3);

    let threshold = SearchOptions::new(0).with_min_score(0.5);
    let filtered = bank
        .search_with_options("deploy kubernetes cluster", &threshold)
        .await
        .expect("search");
    assert_eq!(hit_ids(&filtered), vec!["exp_0001"]);

    // A blank query falls back to lexical ranking instead of calling the provider.
    let empty_query = bank.search("   ", 0).await.expect("search blank");
    assert_eq!(hit_ids(&empty_query), vec!["exp_0001", "exp_0002"]);
    assert!(empty_query.iter().all(|hit| hit.score == 0.0));
}

#[tokio::test]
async fn embedding_search_reembeds_items_loaded_from_a_store() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("experience.jsonl");
    {
        let bank = ExperienceBank::open_json_file(&path).await.expect("open");
        bank.add(item("deploy kubernetes cluster", &["k8s"]))
            .await
            .expect("add");
        bank.add(item("bake chocolate cake", &["baking"]))
            .await
            .expect("add");
    }

    // Embeddings are not persisted, so a reopened bank recomputes them on search.
    let bank = ExperienceBank::open_json_file(&path)
        .await
        .expect("reopen")
        .with_embedding(Arc::new(HashEmbedding::new(32)));
    assert!(bank.embedding().is_some());
    let hits = bank
        .search("deploy kubernetes cluster", 1)
        .await
        .expect("search");
    assert_eq!(hit_ids(&hits), vec!["exp_0001"]);
    assert_eq!(hits[0].item.embedding.len(), 32);
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

#[tokio::test]
async fn json_file_persistence_keeps_records_and_the_id_counter() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("nested").join("experience.jsonl");
    {
        let bank = ExperienceBank::open_json_file(&path).await.expect("open");
        bank.add(
            item("summarize the report", &["summarizer"]).with_query_examples(["summarize this"]),
        )
        .await
        .expect("add");
        bank.add(item("translate text", &["translator"]).with_embedding(vec![0.5, 0.5]))
            .await
            .expect("add");
        bank.add(item("bake a cake", &["baking"]).with_id("custom-1"))
            .await
            .expect("add");
    }

    let document = tokio::fs::read_to_string(&path).await.expect("read");
    assert_eq!(document.lines().count(), 3);
    assert!(!document.contains("embedding"));

    let bank = ExperienceBank::open_json_file(&path).await.expect("reopen");
    assert_eq!(bank_ids(&bank), vec!["custom-1", "exp_0001", "exp_0002"]);
    let restored = bank.get("exp_0001").expect("restored");
    assert_eq!(restored.query_examples, vec!["summarize this"]);
    assert_eq!(restored.success_count, 1);
    // Vectors are not persisted, so the restored item has none.
    assert!(bank.get("exp_0002").expect("restored").embedding.is_empty());

    let next = bank
        .add(item("a new pattern", &["new"]))
        .await
        .expect("add after reopen");
    assert_eq!(next.id, "exp_0003");
    assert_eq!(bank.store().list().await.expect("store list").len(), 4);
}

#[tokio::test]
async fn corrupt_lines_are_skipped_on_load() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("experience.jsonl");
    let valid = serde_json::json!({
        "id": "exp_0001",
        "query_pattern": "summarize the report",
        "skill_ids": ["summarizer"],
        "success_count": 2,
    });
    let document = format!(
        "{}\n{{this is not json\n\n{{\"no_id\": true}}\n",
        serde_json::to_string(&valid).expect("encode")
    );
    tokio::fs::write(&path, document).await.expect("write");

    let bank = ExperienceBank::open_json_file(&path).await.expect("open");
    assert_eq!(bank.len(), 1);
    assert_eq!(bank.get("exp_0001").expect("loaded").success_count, 2);
    let next = bank
        .add(item("a new pattern", &["new"]))
        .await
        .expect("add");
    assert_eq!(next.id, "exp_0002");
}

#[tokio::test]
async fn clear_persists_and_leaves_an_empty_document() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("experience.jsonl");
    let bank = ExperienceBank::open_json_file(&path).await.expect("open");
    bank.add(item("summarize the report", &["summarizer"]))
        .await
        .expect("add");
    bank.add(item("translate text", &["translator"]))
        .await
        .expect("add");
    bank.remove("exp_0001").await.expect("remove");
    assert_eq!(bank.store().get("exp_0001").await.expect("store get"), None);

    bank.clear().await.expect("clear");
    let document = tokio::fs::read_to_string(&path).await.expect("read");
    assert!(document.is_empty());
    let reopened = ExperienceBank::open_json_file(&path).await.expect("reopen");
    assert!(reopened.is_empty());
}

// ---------------------------------------------------------------------------
// SQLite backend
// ---------------------------------------------------------------------------

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn sqlite_backend_round_trips_through_the_bank() {
    let dir = tempfile::tempdir().expect("tempdir");
    let url = format!("sqlite:{}", dir.path().join("experience.db").display());
    {
        let bank = ExperienceBank::open_sqlite(&url).await.expect("open");
        assert_eq!(bank.store().list().await.expect("list").len(), 0);
        bank.add(item("summarize the report", &["summarizer"]))
            .await
            .expect("add");
        bank.add(item("translate text", &["translator"]))
            .await
            .expect("add");
    }

    let bank = ExperienceBank::open_sqlite(&url).await.expect("reopen");
    assert_eq!(bank_ids(&bank), vec!["exp_0001", "exp_0002"]);
    let next = bank
        .add(item("bake a cake", &["baking"]))
        .await
        .expect("add");
    assert_eq!(next.id, "exp_0003");

    assert!(bank.remove("exp_0002").await.expect("remove"));
    bank.clear().await.expect("clear");
    assert!(bank.is_empty());
    assert!(bank.store().list().await.expect("list").is_empty());
}

#[cfg(feature = "sqlite")]
#[test]
fn sqlite_keys_live_under_the_experience_prefix() {
    assert_eq!(
        openjiuwen_symphony::experience::DEFAULT_KEY_PREFIX,
        "experience/"
    );
}

#[test]
fn item_defaults_and_builders_are_exposed() {
    let item = ExperienceItem::new("pattern")
        .with_id("exp_0042")
        .with_query_examples(["a", "b"])
        .with_skill_ids(["skill"])
        .with_success_count(7)
        .with_created_at(1.0)
        .with_last_hit_at(2.0);
    assert_eq!(item.id, "exp_0042");
    assert_eq!(item.normalized_pattern(), "pattern");
    assert_eq!(item.dedupe_key().len(), 64);
    assert_eq!(item.skill_set().len(), 1);
    assert_eq!(item.search_text(), "pattern\na\nb\nskill");
    let json = serde_json::to_value(&item).expect("serialize");
    assert_eq!(json["success_count"], serde_json::json!(7));
    assert!(json.get("embedding").is_none());
}
