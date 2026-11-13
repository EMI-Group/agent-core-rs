//! Memory + retrieval scenario: remember facts, then recall them by keyword, by
//! vector similarity, and through the degraded path when no embedder is wired.
//!
//! This is the cross-crate seam between `openjiuwen::core::memory` (scoped
//! long-term records, the rule-based analyzer, the recall pipeline) and
//! `openjiuwen::core::retrieval` (documents, chunking, the BM25 index, the
//! `Embedding` trait). The only public bridge between the two crates is the
//! shared `Embedding` implementation, so the integration scenario indexes a
//! small corpus for retrieval, stores the best retrieved chunk as a memory, and
//! reads it back through `get` and through recall.
//!
//! Every scenario is offline and deterministic: the facade runs over in-memory
//! stores, the embedder is the hash-based `HashEmbedding` test double, the
//! analyzer is the deterministic rule extractor, and nothing asserts the wall
//! clock, a generated id it did not read back, or an absolute score. Scores are
//! only compared against each other or between two queries over the same
//! records.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod support;

use openjiuwen::core::common::error::Error;
use openjiuwen::core::memory::analyzer::ExtractionInput;
use openjiuwen::core::memory::{
    AddOptions, LongTermMemory, MemoryEngineConfig, MemoryScope, MemoryType, NewMemory, SearchMode,
    SearchOptions,
};
use openjiuwen::core::retrieval::{
    Bm25Index, Bm25Query, CharChunker, Chunker, Document, Embedding, HashEmbedding, Metadata,
};
use std::sync::Arc;
use support::within;
/// The fixed scope every scenario remembers into and recalls from.
fn scope() -> MemoryScope {
    MemoryScope::new("alice", "workspace")
}
/// Search options that keep every candidate regardless of its score.
fn permissive(top_k: usize) -> SearchOptions {
    SearchOptions::default()
        .with_threshold(0.0)
        .with_top_k(top_k)
}
/// A facade with vector recall enabled through the deterministic hash embedder.
fn semantic_facade() -> Arc<LongTermMemory> {
    LongTermMemory::builder(MemoryEngineConfig::default())
        .embedding(Arc::new(HashEmbedding::new(16)))
        .build()
        .expect("the facade with an embedding model builds")
}
/// A facade without an embedding model, so recall is keyword-only.
async fn keyword_facade() -> Arc<LongTermMemory> {
    within(
        "build the keyword-only facade",
        LongTermMemory::in_memory(MemoryEngineConfig::default()),
    )
    .await
    .expect("the keyword-only facade builds")
}
/// Stores one semantic-memory fact and returns the id the engine assigned it.
async fn add_fact(ltm: &LongTermMemory, content: &str) -> String {
    let outcome = within(
        "add a memory fact",
        ltm.add(
            &scope(),
            NewMemory::semantic(content),
            &AddOptions::default(),
        ),
    )
    .await
    .expect("the fact is stored");
    assert_eq!(
        outcome.added.len(),
        1,
        "a distinct fact is added: {outcome:?}"
    );
    outcome
        .added
        .first()
        .expect("the write reported its new record")
        .id
        .clone()
}
/// The three short documents the retrieval half of the scenario indexes.
fn corpus() -> Vec<Document> {
    vec![
        Document::with_id(
            "runbook",
            "The service uses Postgres for durable storage. Backups run nightly.",
        ),
        Document::with_id("style", "I prefer dark mode for late review sessions."),
        Document::with_id(
            "deploy",
            "Deployments land through the blue green pipeline every friday.",
        ),
    ]
}
/// Chunks [`corpus()`] and indexes every chunk in a BM25 lexical index.
fn retrieval_index() -> Bm25Index {
    let chunker = CharChunker::new(160, 20).expect("valid chunker parameters");
    let chunks = chunker
        .chunk_documents(&corpus())
        .expect("the short documents chunk");
    assert_eq!(chunks.len(), 3, "each short document is exactly one chunk");
    let mut index = Bm25Index::default();
    for chunk in &chunks {
        index.add_document(chunk.id.clone(), chunk.text.clone(), Metadata::new());
    }
    index
}
/// The best chunk for `query`, as the retrieval half would hand it to the caller.
fn top_chunk(index: &Bm25Index, query: &str) -> String {
    let hits = index.search(&Bm25Query::new(query).with_top_k(3));
    hits.first()
        .expect("the query matches at least one indexed chunk")
        .text
        .clone()
}
#[tokio::test]
async fn keyword_recall_returns_the_remembered_fact_first() {
    let ltm = keyword_facade().await;
    let target = add_fact(&ltm, "rust ownership and borrowing rules").await;
    let unrelated = [
        add_fact(&ltm, "banana bread recipe with walnuts").await,
        add_fact(&ltm, "history of ancient rome").await,
    ];
    let hits = within(
        "keyword recall",
        ltm.search(
            &scope(),
            "ownership borrowing",
            &permissive(5).with_mode(SearchMode::Keyword),
        ),
    )
    .await
    .expect("keyword recall");
    assert_eq!(hits.len(), 3, "{hits:?}");
    assert_eq!(hits[0].record.id, target);
    assert_eq!(hits[0].record.content, "rust ownership and borrowing rules");
    // The query vocabulary covers the whole target document, so BM25 saturates.
    assert!(hits[0].keyword > 0.99, "{hits:?}");
    // Keyword mode reports no semantic signal at all.
    assert!(hits[0].similarity.abs() < f32::EPSILON, "{hits:?}");
    // The two unrelated facts are still candidates but carry no keyword evidence.
    assert!(hits[1].keyword.abs() < f32::EPSILON, "{hits:?}");
    for hit in &hits[1..] {
        assert!(
            unrelated.contains(&hit.record.id),
            "unexpected record: {hit:?}"
        );
        assert!(hit.keyword.abs() < f32::EPSILON, "{hit:?}");
        assert!(hit.score < hits[0].score, "{hit:?}");
    }
    assert!(
        hits.windows(2).all(|pair| pair[0].score >= pair[1].score),
        "hits must be ordered by descending score: {hits:?}"
    );
    // Every remembered fact is listed back under its kind.
    let listed = within(
        "list the remembered facts",
        ltm.list(&scope(), &[MemoryType::SemanticMemory]),
    )
    .await
    .expect("list");
    assert_eq!(listed.len(), 3, "{listed:?}");
}
#[tokio::test]
async fn exact_token_queries_outrank_partial_token_queries() {
    let ltm = keyword_facade().await;
    let target = add_fact(&ltm, "kubernetes autoscaling policy for payments").await;
    add_fact(&ltm, "kubernetes dashboard widgets for release reviews").await;
    add_fact(&ltm, "banana bread recipe with walnuts").await;
    let options = permissive(5).with_mode(SearchMode::Keyword);
    let exact = within(
        "exact-token recall",
        ltm.search(&scope(), "autoscaling", &options),
    )
    .await
    .expect("exact-token recall");
    assert_eq!(exact.len(), 3, "{exact:?}");
    assert_eq!(exact[0].record.id, target);
    assert!(exact[0].keyword > 0.99, "{exact:?}");
    // A partial token is a different token: BM25 matches whole terms only.
    let partial = within(
        "partial-token recall",
        ltm.search(&scope(), "autoscal", &options),
    )
    .await
    .expect("partial-token recall");
    assert!(
        partial.iter().all(|hit| hit.keyword.abs() < f32::EPSILON),
        "{partial:?}"
    );
    let partial_target = partial
        .iter()
        .find(|hit| hit.record.id == target)
        .expect("the target is still a candidate");
    assert!(partial_target.score < exact[0].score, "{partial:?}");
    // A query sharing no term with any fact cannot clear the default 0.3 threshold,
    // because only the equal importance and recency terms remain.
    let no_overlap = within(
        "no-overlap recall",
        ltm.search(
            &scope(),
            "xylophone marmalade",
            &SearchOptions::default()
                .with_top_k(5)
                .with_mode(SearchMode::Keyword),
        ),
    )
    .await
    .expect("no-overlap recall");
    assert!(no_overlap.is_empty(), "{no_overlap:?}");
    // At threshold 0.0 the same records are returned but rank by tie-breakers only.
    let permissive_hits = within(
        "permissive no-overlap recall",
        ltm.search(&scope(), "xylophone marmalade", &options),
    )
    .await
    .expect("permissive no-overlap recall");
    assert_eq!(permissive_hits.len(), 3, "{permissive_hits:?}");
    assert!(
        permissive_hits
            .iter()
            .all(|hit| hit.keyword.abs() < f32::EPSILON),
        "{permissive_hits:?}"
    );
}
#[tokio::test]
async fn semantic_mode_recalls_the_vocabulary_match_first() {
    let ltm = semantic_facade();
    add_fact(&ltm, "banana bread recipe with walnuts").await;
    let target = add_fact(&ltm, "tokio async runtime task scheduling").await;
    add_fact(&ltm, "history of ancient rome").await;
    let hits = within(
        "semantic recall",
        ltm.search(
            &scope(),
            "tokio async runtime",
            &permissive(5).with_mode(SearchMode::Semantic),
        ),
    )
    .await
    .expect("semantic recall");
    assert_eq!(hits.len(), 3, "{hits:?}");
    assert_eq!(hits[0].record.id, target);
    assert_eq!(
        hits[0].record.content,
        "tokio async runtime task scheduling"
    );
    // The embedder produced a real similarity, unlike the keyword-only fallback.
    assert!(hits[0].similarity > 0.0, "{hits:?}");
    assert!(hits[0].similarity >= hits[1].similarity, "{hits:?}");
    assert!(hits[0].score > hits[1].score, "{hits:?}");
    // Blending the same records in hybrid mode agrees on the top hit.
    let hybrid = within(
        "hybrid recall",
        ltm.search(
            &scope(),
            "tokio async runtime",
            &SearchOptions::default()
                .with_top_k(5)
                .with_mode(SearchMode::Hybrid),
        ),
    )
    .await
    .expect("hybrid recall");
    assert_eq!(hybrid[0].record.id, target);
    assert!(hybrid[0].similarity > 0.0, "{hybrid:?}");
}
#[tokio::test]
async fn a_missing_embedding_rejects_semantics_and_degrades_hybrid() {
    let ltm = keyword_facade().await;
    let target = add_fact(&ltm, "rust ownership and borrowing rules").await;
    add_fact(&ltm, "rust compiler toolchain internals").await;
    // (a) Semantic mode is a typed capability gap, not an empty result.
    let error = within(
        "rejected semantic recall",
        ltm.search(
            &scope(),
            "rust",
            &SearchOptions::default().with_mode(SearchMode::Semantic),
        ),
    )
    .await
    .expect_err("semantic mode needs an embedding model");
    match error {
        Error::Unsupported(reason) => {
            assert!(reason.contains("embedding"), "unexpected reason: {reason}");
        }
        other => panic!("expected Error::Unsupported, got {other:?}"),
    }
    // (b) Hybrid mode degrades to the keyword path instead of failing.
    let hits = within(
        "degraded hybrid recall",
        ltm.search(
            &scope(),
            "ownership borrowing",
            &permissive(5).with_mode(SearchMode::Hybrid),
        ),
    )
    .await
    .expect("hybrid recall degrades to keywords");
    assert_eq!(hits.len(), 2, "{hits:?}");
    assert_eq!(hits[0].record.id, target);
    assert!(hits[0].keyword > 0.99, "{hits:?}");
    // `RecallHit::similarity` is an `f32` component whose documented "no semantic
    // signal" value is 0.0, so every degraded hit reports zero similarity.
    assert!(
        hits.iter().all(|hit| hit.similarity.abs() < f32::EPSILON),
        "{hits:?}"
    );
    // Keyword mode over the same records produces the same order.
    let keyword = within(
        "keyword recall",
        ltm.search(
            &scope(),
            "ownership borrowing",
            &permissive(5).with_mode(SearchMode::Keyword),
        ),
    )
    .await
    .expect("keyword recall");
    let ids = |hits: &[openjiuwen::core::memory::RecallHit]| {
        hits.iter()
            .map(|hit| hit.record.id.clone())
            .collect::<Vec<String>>()
    };
    assert_eq!(ids(&hits), ids(&keyword));
}
#[tokio::test]
async fn retrieved_chunks_bridge_into_recallable_memories() {
    let embedding = Arc::new(HashEmbedding::new(16));
    // The bridge: retrieval and memory share one `Embedding` implementation.
    assert_eq!(embedding.dimension_hint(), Some(16));
    let query_vector = within(
        "embed the retrieval query",
        embedding.embed_query("Postgres durable storage"),
    )
    .await
    .expect("the shared embedder encodes the query");
    assert_eq!(query_vector.len(), 16);
    assert!(query_vector.iter().all(|value| value.is_finite()));
    let index = retrieval_index();
    let retrieved = index.search(&Bm25Query::new("Postgres durable storage").with_top_k(3));
    assert!(!retrieved.is_empty(), "the corpus has a matching chunk");
    let top = top_chunk(&index, "Postgres durable storage");
    assert!(top.contains("Postgres"), "the runbook chunk wins: {top}");
    // Store the retrieved text as a memory in an embedding-backed facade.
    let ltm = LongTermMemory::builder(MemoryEngineConfig::default())
        .embedding(embedding)
        .build()
        .expect("the facade with an embedding model builds");
    let outcome = within(
        "store the retrieved chunk",
        ltm.add(
            &scope(),
            NewMemory::semantic(top.clone()),
            &AddOptions::default(),
        ),
    )
    .await
    .expect("the retrieved chunk is stored");
    let id = outcome
        .added
        .first()
        .expect("the retrieval bridge added one record")
        .id
        .clone();
    // The retrieved text and the stored record round-trip through `get`.
    let stored = within("read the stored chunk", ltm.get(&scope(), &id))
        .await
        .expect("get")
        .expect("the stored record exists");
    assert_eq!(stored.content, top);
    assert_eq!(retrieved[0].text, stored.content);
    // The same text is recallable through the keyword index ...
    let keyword = within(
        "keyword recall of the retrieved chunk",
        ltm.search(
            &scope(),
            "Postgres durable storage",
            &permissive(5).with_mode(SearchMode::Keyword),
        ),
    )
    .await
    .expect("keyword recall");
    assert_eq!(keyword[0].record.id, id);
    assert!(keyword[0].keyword > 0.99, "{keyword:?}");
    // ... and through the vectors produced by the shared embedder.
    let semantic = within(
        "semantic recall of the retrieved chunk",
        ltm.search(
            &scope(),
            "Postgres durable storage",
            &permissive(5).with_mode(SearchMode::Semantic),
        ),
    )
    .await
    .expect("semantic recall");
    assert_eq!(semantic[0].record.id, id);
    assert!(semantic[0].similarity > 0.0, "{semantic:?}");
}
#[tokio::test]
async fn the_retrieved_chunk_is_remembered_as_a_recallable_fact() {
    let index = retrieval_index();
    let top = top_chunk(&index, "Postgres durable storage");
    // `remember` runs the deterministic rule analyzer: offline, no model, no prompt.
    let ltm = semantic_facade();
    let report = within(
        "remember the retrieved chunk",
        ltm.remember(
            &scope(),
            &ExtractionInput::from_text(top),
            &AddOptions::default(),
        ),
    )
    .await
    .expect("the rule analyzer extracts facts");
    assert_eq!(
        report.extracted,
        report.added + report.merged + report.skipped,
        "{report:?}"
    );
    assert!(
        report.added >= 1,
        "the project statement becomes a new record: {report:?}"
    );
    assert_eq!(report.degraded, None, "the rule extractor needs no model");
    let records = within("list the remembered facts", ltm.list(&scope(), &[]))
        .await
        .expect("list");
    assert_eq!(records.len(), report.added, "{records:?}");
    let fact = records
        .iter()
        .find(|record| record.content.contains("Postgres"))
        .expect("the project statement is stored");
    assert_eq!(fact.kind, MemoryType::SemanticMemory);
    // The rule analyzer tags the statement it extracted rather than guessing more.
    let tags = fact
        .metadata
        .get("tags")
        .and_then(|value| value.as_array())
        .expect("the analyzer records its tags on the fact");
    assert!(
        tags.iter().any(|tag| tag.as_str() == Some("project")),
        "{fact:?}"
    );
    // The remembered fact is recallable, not just listable.
    let hits = within(
        "recall the remembered fact",
        ltm.search(&scope(), "Postgres", &permissive(5)),
    )
    .await
    .expect("recall");
    assert_eq!(hits[0].record.id, fact.id);
    assert_eq!(hits[0].record.content, fact.content);
}
