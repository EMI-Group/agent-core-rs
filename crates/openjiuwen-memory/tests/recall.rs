#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Integration coverage of the recall pipeline through the public facade.
//!
//! The unit tests beside `recall::scoring`, `recall::pipeline`, and `engine::search`
//! pin the individual formulas; this suite exercises the wired path: semantic candidate
//! generation with `HashEmbedding`, the keyword fallback without an embedding, the
//! threshold/top-k/token-budget knobs, the importance and recency terms, prompt
//! rendering, and the memory+graph `search_all` merge.
//!
//! Every assertion is deterministic and offline. Scores are only compared with each
//! other or against the public [`openjiuwen_memory::recall::blend`] helper, never
//! against absolute values derived from the wall clock.

use std::sync::Arc;

use chrono::{DateTime, Duration, Utc};
use openjiuwen_common::error::Error;
use openjiuwen_foundation::prompt::{ApproxTokenCounter, TokenCounter};
use openjiuwen_memory::config::{MemoryEngineConfig, RankingConfig};
use openjiuwen_memory::facade::LongTermMemory;
use openjiuwen_memory::graph::GraphAddOptions;
use openjiuwen_memory::prompt::render_snippets;
use openjiuwen_memory::recall::blend;
use openjiuwen_memory::record::{AddOptions, MemoryScope, NewMemory, SearchMode, SearchOptions};
use openjiuwen_memory::store::MemoryStores;
use openjiuwen_retrieval::embedding::HashEmbedding;

/// The scope shared by every test.
fn scope() -> MemoryScope {
    MemoryScope::new("alice", "ws")
}

/// A fixed instant used where recency must not differ between two records.
fn fixed_timestamp() -> DateTime<Utc> {
    DateTime::from_timestamp(1_700_000_000, 0).expect("fixed time")
}

/// A facade with semantic recall enabled.
fn semantic_facade() -> Arc<LongTermMemory> {
    LongTermMemory::builder(MemoryEngineConfig::default())
        .embedding(Arc::new(HashEmbedding::new(16)))
        .build()
        .expect("semantic facade")
}

/// A facade without an embedding model, so recall degrades to keyword scoring.
async fn keyword_facade() -> Arc<LongTermMemory> {
    LongTermMemory::in_memory(MemoryEngineConfig::default())
        .await
        .expect("keyword facade")
}

/// Search options that keep every candidate regardless of its score.
fn permissive(top_k: usize) -> SearchOptions {
    SearchOptions::default()
        .with_threshold(0.0)
        .with_top_k(top_k)
}

/// Stores `memory` in [`scope`] and returns the id of the new record.
async fn add_record(ltm: &LongTermMemory, memory: NewMemory) -> String {
    ltm.add(&scope(), memory, &AddOptions::default())
        .await
        .expect("add")
        .added
        .first()
        .expect("the write added one record")
        .id
        .clone()
}

/// Stores a semantic-memory record with `content`.
async fn add(ltm: &LongTermMemory, content: &str) -> String {
    add_record(ltm, NewMemory::semantic(content)).await
}

#[tokio::test]
async fn semantic_search_orders_the_vocabulary_match_first() {
    let ltm = semantic_facade();
    add(&ltm, "banana bread recipe with walnuts").await;
    let target = add(&ltm, "tokio async runtime task scheduling").await;
    add(&ltm, "history of ancient rome").await;

    let hits = ltm
        .search(&scope(), "tokio async runtime", &permissive(5))
        .await
        .expect("search");
    assert_eq!(hits.len(), 3);
    assert_eq!(hits[0].record.id, target);
    assert_eq!(
        hits[0].record.content,
        "tokio async runtime task scheduling"
    );
    assert!(
        hits.windows(2).all(|pair| pair[0].score >= pair[1].score),
        "hits must be ordered by descending score: {hits:?}"
    );
    // The target shares the whole query vocabulary, so it wins both components.
    assert!((hits[0].keyword - 1.0).abs() < f32::EPSILON);
    assert!(hits[0].similarity > hits[1].similarity);

    // `RecallHit` exposes every component that produced the score.
    for hit in &hits {
        for (name, value) in [
            ("score", hit.score),
            ("similarity", hit.similarity),
            ("keyword", hit.keyword),
            ("importance", hit.importance),
            ("recency", hit.recency),
        ] {
            assert!(value.is_finite(), "{name} must be finite (got {value})");
            assert!(
                (0.0..=1.0).contains(&value),
                "{name} must be within 0..=1 (got {value})"
            );
        }
    }
}

#[tokio::test]
async fn top_k_and_threshold_bound_the_result_set() {
    let ltm = semantic_facade();
    for content in [
        "tokio async runtime task scheduling",
        "banana bread recipe with walnuts",
        "history of ancient rome",
    ] {
        add(&ltm, content).await;
    }

    let one = ltm
        .search(
            &scope(),
            "tokio async runtime",
            &SearchOptions::default().with_top_k(1).with_threshold(0.0),
        )
        .await
        .expect("top one");
    assert_eq!(one.len(), 1);

    let none = ltm
        .search(
            &scope(),
            "tokio async runtime",
            &SearchOptions::default().with_threshold(1.0),
        )
        .await
        .expect("high threshold");
    assert!(none.is_empty(), "no blended score can reach 1.0 here");

    let all = ltm
        .search(
            &scope(),
            "tokio async runtime",
            &SearchOptions::default().with_threshold(0.0),
        )
        .await
        .expect("low threshold");
    assert_eq!(all.len(), 3);
}

#[tokio::test]
async fn higher_importance_ranks_first_at_equal_similarity_and_recency() {
    let ltm = keyword_facade().await;
    // Both records match exactly one query term with the same document length, so the
    // BM25 keyword component and the (absent) similarity component are identical.
    let important = add_record(
        &ltm,
        NewMemory::semantic("alpha beta gamma")
            .with_importance(0.9)
            .with_timestamp(fixed_timestamp()),
    )
    .await;
    let minor = add_record(
        &ltm,
        NewMemory::semantic("delta epsilon zeta")
            .with_importance(0.1)
            .with_timestamp(fixed_timestamp()),
    )
    .await;

    let hits = ltm
        .search(
            &scope(),
            "alpha delta",
            &permissive(5).with_mode(SearchMode::Keyword),
        )
        .await
        .expect("search");
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].record.id, important);
    assert_eq!(hits[1].record.id, minor);
    assert!(hits[0].importance > hits[1].importance);
    assert!(hits[0].score > hits[1].score);
    // The confounders are equal, so only the importance term breaks the tie.
    assert!((hits[0].keyword - hits[1].keyword).abs() < f32::EPSILON);
    assert!((hits[0].similarity - hits[1].similarity).abs() < f32::EPSILON);
    assert!((hits[0].recency - hits[1].recency).abs() < f32::EPSILON);
}

#[tokio::test]
async fn newer_records_rank_first_at_equal_similarity_and_importance() {
    let ltm = keyword_facade().await;
    // The recency term is weighted at 5%, so absolute timestamps from years ago decay
    // below f32 resolution and would tie. Anchoring the pair to "now" keeps the 90-day
    // gap meaningful without asserting any wall-clock value.
    let now = Utc::now();
    let older = add_record(
        &ltm,
        NewMemory::semantic("alpha beta gamma").with_timestamp(now - Duration::days(90)),
    )
    .await;
    let newer = add_record(
        &ltm,
        NewMemory::semantic("delta epsilon zeta").with_timestamp(now),
    )
    .await;

    let hits = ltm
        .search(
            &scope(),
            "alpha delta",
            &permissive(5).with_mode(SearchMode::Keyword),
        )
        .await
        .expect("search");
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].record.id, newer);
    assert_eq!(hits[1].record.id, older);
    assert!(hits[0].recency > hits[1].recency);
    assert!(hits[0].score > hits[1].score);
    // Equal keyword and importance, so recency alone breaks the tie.
    assert!((hits[0].keyword - hits[1].keyword).abs() < f32::EPSILON);
    assert!((hits[0].importance - hits[1].importance).abs() < f32::EPSILON);
}

#[tokio::test]
async fn hybrid_components_are_unit_range_and_match_the_documented_blend() {
    let ltm = semantic_facade();
    for content in [
        "tokio async runtime task scheduling",
        "banana bread recipe with walnuts",
        "history of ancient rome",
    ] {
        add(&ltm, content).await;
    }

    let hits = ltm
        .search(&scope(), "tokio async runtime", &permissive(5))
        .await
        .expect("search");
    assert_eq!(hits.len(), 3);

    let config = RankingConfig::default();
    assert_eq!(ltm.config().ranking, config);
    for hit in &hits {
        for value in [hit.similarity, hit.keyword, hit.importance, hit.recency] {
            assert!(
                (0.0..=1.0).contains(&value),
                "component out of range: {value}"
            );
        }
        let expected = blend(
            hit.similarity,
            hit.keyword,
            hit.importance,
            hit.recency,
            &config,
        );
        assert!(
            (hit.score - expected).abs() < 1e-6,
            "score {} does not match blend {expected}",
            hit.score
        );
        // The weighted average can never leave the range of its inputs.
        let lowest = hit
            .similarity
            .min(hit.keyword)
            .min(hit.importance)
            .min(hit.recency);
        let highest = hit
            .similarity
            .max(hit.keyword)
            .max(hit.importance)
            .max(hit.recency);
        assert!(
            hit.score >= lowest - 1e-6 && hit.score <= highest + 1e-6,
            "score {} must lie within [{lowest}, {highest}]",
            hit.score
        );
    }
}

#[tokio::test]
async fn keyword_fallback_orders_by_term_overlap_and_rejects_semantic_mode() {
    let ltm = keyword_facade().await;
    let both_terms = add(&ltm, "rust ownership and borrowing rules").await;
    add(&ltm, "rust compiler toolchain internals").await;

    let hits = ltm
        .search(&scope(), "rust ownership", &permissive(5))
        .await
        .expect("hybrid degrades to keyword");
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].record.id, both_terms);
    assert!(hits[0].keyword >= hits[1].keyword);
    assert!(hits[0].score >= hits[1].score);
    // No embedder means no semantic signal at all.
    assert!(hits.iter().all(|hit| hit.similarity == 0.0));

    let error = ltm
        .search(
            &scope(),
            "rust",
            &SearchOptions::default().with_mode(SearchMode::Semantic),
        )
        .await
        .expect_err("semantic mode needs an embedder");
    assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
}

#[tokio::test]
async fn a_small_token_budget_trims_the_tail_but_keeps_the_best_hit() {
    let ltm = keyword_facade().await;
    for content in [
        "rust ownership and borrowing rules",
        "rust compiler toolchain internals",
        "banana bread recipe with walnuts",
    ] {
        add(&ltm, content).await;
    }

    let options = permissive(5);
    let unlimited = ltm
        .search(&scope(), "rust", &options)
        .await
        .expect("unlimited");
    assert!(unlimited.len() > 1, "the fixture must seed several hits");

    // Exactly the cost of the first hit's content: only that hit fits.
    let counter = ApproxTokenCounter::new();
    let budget = counter.count_text(&unlimited[0].record.content);
    assert!(budget > 0);
    let trimmed = ltm
        .search(&scope(), "rust", &options.with_token_budget(budget))
        .await
        .expect("trimmed");
    assert_eq!(trimmed.len(), 1);
    assert_eq!(trimmed[0].record.id, unlimited[0].record.id);
}

#[tokio::test]
async fn render_snippets_preserves_hit_order_and_the_budget_shape() {
    let ltm = semantic_facade();
    for content in [
        "tokio async runtime task scheduling",
        "banana bread recipe with walnuts",
        "history of ancient rome",
    ] {
        add(&ltm, content).await;
    }
    let hits = ltm
        .search(&scope(), "tokio async runtime", &permissive(5))
        .await
        .expect("search");
    assert_eq!(hits.len(), 3);

    let rendered = render_snippets(&hits, 512);
    assert!(rendered.starts_with("## Long-term memory"));
    let first = rendered
        .find(&hits[0].record.content)
        .expect("the first hit is rendered");
    let second = rendered
        .find(&hits[1].record.content)
        .expect("the second hit is rendered");
    assert!(first < second, "rendering must preserve hit order");

    let lines: Vec<&str> = rendered.lines().collect();
    assert_eq!(
        lines.len(),
        hits.len() + 1,
        "header plus one bullet per hit"
    );

    // The budget is charged per whole line: header plus the first bullet fits, the
    // second bullet does not.
    let counter = ApproxTokenCounter::new();
    let budget = counter.count_text(lines[0]) + counter.count_text(lines[1]);
    assert_eq!(
        render_snippets(&hits, budget),
        format!("{}\n{}", lines[0], lines[1])
    );
    assert_eq!(render_snippets(&hits, 0), "");
}

#[tokio::test]
async fn search_all_returns_memories_and_graph_entities() {
    let mut config = MemoryEngineConfig::default();
    config.graph.enabled = true;
    let ltm = LongTermMemory::builder(config)
        .stores(MemoryStores::in_memory())
        .build()
        .expect("graph facade");
    let scope = scope();

    ltm.add(
        &scope,
        NewMemory::semantic("alice uses RustLang for systems work"),
        &AddOptions::default(),
    )
    .await
    .expect("add memory");

    let extraction = ltm
        .graph()
        .expect("graph memory is enabled")
        .add(&scope, "Alice uses RustLang", &GraphAddOptions::default())
        .await
        .expect("graph add");
    assert!(
        extraction
            .entities
            .iter()
            .any(|entity| entity.name == "RustLang"),
        "the heuristic extractor must find RustLang: {extraction:?}"
    );

    let combined = ltm
        .search_all(&scope, "RustLang", &SearchOptions::default())
        .await
        .expect("search all");
    assert_eq!(combined.memories.len(), 1);
    assert_eq!(
        combined.memories[0].record.content,
        "alice uses RustLang for systems work"
    );
    assert_eq!(combined.entities.len(), 1);
    assert_eq!(combined.entities[0].entity.name, "RustLang");
    assert_eq!(combined.entities[0].depth, 0);
    assert!(combined.entities[0].score >= 0.3);
}

#[tokio::test]
async fn search_all_without_graph_returns_memories_only() {
    let ltm = keyword_facade().await;
    add(&ltm, "alice prefers dark mode").await;
    assert!(ltm.graph().is_none());

    let combined = ltm
        .search_all(&scope(), "dark mode", &SearchOptions::default())
        .await
        .expect("search all");
    assert_eq!(combined.memories.len(), 1);
    assert_eq!(combined.entities.len(), 0);
}

#[tokio::test]
async fn blank_queries_are_rejected_with_validation_errors() {
    let ltm = keyword_facade().await;
    add(&ltm, "alice prefers dark mode").await;

    for query in ["   ", "\t\n"] {
        let error = ltm
            .search(&scope(), query, &SearchOptions::default())
            .await
            .expect_err("blank query");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
    }
    let error = ltm
        .recall(&scope(), " ", &SearchOptions::default())
        .await
        .expect_err("blank recall");
    assert!(matches!(error, Error::Validation(_)), "{error:?}");
}
