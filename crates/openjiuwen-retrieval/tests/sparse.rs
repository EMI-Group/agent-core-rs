#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::assert_is_empty
)]
//! Integration tests for sparse (BM25) retrieval and rank fusion.
//!
//! Python has no BM25 implementation of its own (its sparse retriever delegates
//! to Milvus/Chroma/PGVector), so the expected scores here are hand-computed from
//! the standard Okapi BM25 formula documented in `sparse::bm25` rather than ported
//! from a Python test file. Everything runs offline and deterministically.

use std::sync::Arc;

use openjiuwen_retrieval::retriever::{
    deduplicate, rrf_fusion, rrf_fusion_default, search_result_to_retrieval, RetrieveMode,
    Retriever, SparseRetriever, DEFAULT_RRF_K,
};
use openjiuwen_retrieval::sparse::{tokenize, Bm25Config, Bm25Index, Bm25Query, Bm25Statistics};
use openjiuwen_retrieval::{Document, Metadata, MetadataFilter, SearchResult, TextChunk};

/// Absolute tolerance for hand-computed score comparisons.
const TOLERANCE: f32 = 1e-4;

fn hit(id: &str, text: &str, score: f32) -> SearchResult {
    SearchResult {
        id: id.to_owned(),
        text: text.to_owned(),
        score,
        metadata: Metadata::new(),
    }
}

fn ids(hits: &[SearchResult]) -> Vec<String> {
    hits.iter().map(|result| result.id.clone()).collect()
}

fn texts(results: &[SearchResult]) -> Vec<&str> {
    results.iter().map(|result| result.text.as_str()).collect()
}

fn metadata(entries: &[(&str, &str)]) -> Metadata {
    entries
        .iter()
        .map(|(key, value)| ((*key).to_owned(), (*value).into()))
        .collect()
}

/// Three documents: `d2` repeats `quick`, `d3` shares no term with the query.
///
/// Tokens: `d1 = [quick, brown, fox]`, `d2 = [quick, quick, fox]`,
/// `d3 = [lazy, dog]`; `N = 3`, `total_tokens = 8`, `avgdl = 8/3`.
fn quick_corpus() -> Bm25Index {
    let mut index = Bm25Index::default();
    index.add_document("d1", "quick brown fox", Metadata::new());
    index.add_document("d2", "quick quick fox", Metadata::new());
    index.add_document("d3", "lazy dog", Metadata::new());
    index
}

#[test]
fn bm25_ranks_by_hand_computed_scores() {
    let hits = quick_corpus().search(&Bm25Query::new("quick"));
    assert_eq!(ids(&hits), vec!["d2", "d1"]);
    // idf(quick) = ln(1 + (3 - 2 + 0.5) / 2.5) = ln(1.6)
    // d2: tf = 2 -> 0.4700036 * 4.4 / 3.3125
    // d1: tf = 1 -> 0.4700036 * 2.2 / 2.3125
    assert!((hits[0].score - 0.624_307).abs() < TOLERANCE);
    assert!((hits[1].score - 0.447_139).abs() < TOLERANCE);
    assert_eq!(hits[0].text, "quick quick fox");
    // Documents sharing no query term are excluded rather than scored zero.
    assert!(!ids(&hits).contains(&"d3".to_owned()));
}

#[test]
fn rare_terms_outrank_common_terms_through_idf() {
    let mut index = Bm25Index::default();
    index.add_document("rare", "rare", Metadata::new());
    index.add_document("common-1", "common", Metadata::new());
    index.add_document("common-2", "common", Metadata::new());
    index.add_document("common-3", "common", Metadata::new());

    // Every document has length 1 = avgdl, so tf saturation and length
    // normalisation reduce to 1.0 and the score equals the idf.
    assert!((index.idf("rare") - 1.203_972_8).abs() < TOLERANCE);
    assert!((index.idf("common") - 0.356_674_94).abs() < TOLERANCE);
    assert!(index.idf("rare") > index.idf("common"));

    let hits = index.search(&Bm25Query::new("rare common"));
    assert_eq!(ids(&hits), vec!["rare", "common-1", "common-2", "common-3"]);
    assert!((hits[0].score - 1.203_972_8).abs() < TOLERANCE);
    assert!((hits[1].score - 0.356_674_94).abs() < TOLERANCE);
}

#[test]
fn k1_saturates_term_frequency() {
    let default = quick_corpus().search(&Bm25Query::new("quick"));
    assert!(default[0].score > default[1].score);

    let mut index = Bm25Index::new(Bm25Config::new(0.0, 0.75)).unwrap();
    index.add_document("d1", "quick brown fox", Metadata::new());
    index.add_document("d2", "quick quick fox", Metadata::new());
    index.add_document("d3", "lazy dog", Metadata::new());
    let saturated = index.search(&Bm25Query::new("quick"));
    // k1 = 0 removes tf saturation: both documents score exactly idf(quick).
    assert_eq!(ids(&saturated), vec!["d1", "d2"]);
    assert!((saturated[0].score - saturated[1].score).abs() < TOLERANCE);
    assert!((saturated[0].score - 0.470_003_63).abs() < TOLERANCE);
}

#[test]
fn b_zero_disables_length_normalisation() {
    let mut long_and_short = Bm25Index::default();
    long_and_short.add_document("short", "apple", Metadata::new());
    long_and_short.add_document("long", "apple banana cherry date", Metadata::new());
    let normalised = long_and_short.search(&Bm25Query::new("apple"));
    assert_eq!(ids(&normalised), vec!["short", "long"]);
    assert!(normalised[0].score > normalised[1].score);

    let mut unnormalised = Bm25Index::new(Bm25Config::new(1.2, 0.0)).unwrap();
    unnormalised.add_document("short", "apple", Metadata::new());
    unnormalised.add_document("long", "apple banana cherry date", Metadata::new());
    let flat = unnormalised.search(&Bm25Query::new("apple"));
    // Same tf and idf, different lengths: b = 0 makes the scores identical.
    assert!((flat[0].score - flat[1].score).abs() < TOLERANCE);
    assert_eq!(ids(&flat), vec!["short", "long"]);

    let invalid = Bm25Index::new(Bm25Config::new(1.2, 1.5));
    assert!(invalid.is_err());
}

#[test]
fn top_k_truncates_and_zero_returns_nothing() {
    let index = quick_corpus();
    let top_one = index.search(&Bm25Query::new("quick").with_top_k(1));
    assert_eq!(ids(&top_one), vec!["d2"]);
    let default_k = index.search(&Bm25Query::new("quick"));
    assert_eq!(default_k.len(), 2);
    assert!(index
        .search(&Bm25Query::new("quick").with_top_k(0))
        .is_empty());
}

#[test]
fn metadata_filters_select_documents_before_scoring() {
    let mut index = Bm25Index::default();
    index.add_document("en", "quick fox", metadata(&[("lang", "en")]));
    index.add_document("de", "quick fuchs", metadata(&[("lang", "de")]));
    index.add_document("fr", "quick renard", metadata(&[("lang", "fr")]));

    let filter = MetadataFilter::eq("lang", "de");
    let hits = index.search(&Bm25Query::new("quick").with_filter(filter));
    assert_eq!(ids(&hits), vec!["de"]);

    let conjunction = MetadataFilter::and([
        MetadataFilter::ne("lang", "en"),
        MetadataFilter::exists("lang"),
    ]);
    let hits = index.search(&Bm25Query::new("quick").with_filter(conjunction));
    assert_eq!(ids(&hits), vec!["de", "fr"]);

    let unmatched =
        index.search(&Bm25Query::new("quick").with_filter(MetadataFilter::eq("lang", "jp")));
    assert!(unmatched.is_empty());
}

#[test]
fn chunk_doc_id_is_filterable_and_reported() {
    let document = Document::with_id("doc-1", "source text");
    let chunk = TextChunk::from_document(&document, "quick fox", Some("chunk-1".into()));
    let mut index = Bm25Index::default();
    index.add_chunks(std::slice::from_ref(&chunk));

    let hits =
        index.search(&Bm25Query::new("quick").with_filter(MetadataFilter::eq("doc_id", "doc-1")));
    assert_eq!(ids(&hits), vec!["chunk-1"]);
    let projected = search_result_to_retrieval(hits[0].clone());
    assert_eq!(projected.doc_id.as_deref(), Some("doc-1"));
    assert_eq!(projected.chunk_id.as_deref(), Some("chunk-1"));
    assert_eq!(
        hits[0].metadata.get("doc_id"),
        Some(&serde_json::json!("doc-1"))
    );
}

#[test]
fn score_threshold_is_inclusive() {
    let index = quick_corpus();
    let all = index.search(&Bm25Query::new("quick"));
    let top_score = all[0].score;

    let above = index.search(&Bm25Query::new("quick").with_score_threshold(0.5));
    assert_eq!(ids(&above), vec!["d2"]);
    let inclusive = index.search(&Bm25Query::new("quick").with_score_threshold(top_score));
    assert_eq!(ids(&inclusive), vec!["d2"]);
    let impossible = index.search(&Bm25Query::new("quick").with_score_threshold(top_score + 0.5));
    assert!(impossible.is_empty());
}

#[test]
fn equal_scores_break_ties_by_insertion_order() {
    let mut index = Bm25Index::default();
    index.add_document("second", "identical text", Metadata::new());
    index.add_document("first", "identical text", Metadata::new());

    let hits = index.search(&Bm25Query::new("identical"));
    assert_eq!(ids(&hits), vec!["second", "first"]);
    assert!((hits[0].score - hits[1].score).abs() < TOLERANCE);
}

#[test]
fn remove_and_upsert_keep_the_index_consistent() {
    let mut index = quick_corpus();
    assert!(index.remove("d2"));
    assert!(!index.remove("d2"));
    assert_eq!(index.len(), 2);
    assert_eq!(ids(&index.search(&Bm25Query::new("quick"))), vec!["d1"]);
    assert_eq!(index.stats().document_count, 2);

    // Replacing an existing id keeps its position and updates its terms.
    assert!(!index.upsert("d3", "quick dog", Metadata::new()));
    assert_eq!(index.len(), 2);
    let hits = index.search(&Bm25Query::new("quick"));
    assert!(ids(&hits).contains(&"d3".to_owned()));
    assert!(index.search(&Bm25Query::new("lazy")).is_empty());

    // A new id grows the index.
    assert!(index.upsert("d4", "quick", Metadata::new()));
    assert_eq!(index.len(), 3);
    assert!(index.contains("d4"));
    let stats = index.stats();
    assert_eq!(stats.document_count, 3);
    assert_eq!(stats.total_tokens, 3 + 2 + 1);
    assert!((stats.average_document_length - 2.0).abs() < TOLERANCE);
}

#[test]
fn empty_index_and_unknown_terms_yield_no_hits() {
    let mut empty = Bm25Index::default();
    assert!(empty.is_empty());
    assert_eq!(empty.stats(), Bm25Statistics::default());
    assert_eq!(empty.idf("anything"), 0.0);
    assert!(empty.search(&Bm25Query::new("anything")).is_empty());
    assert!(!empty.remove("missing"));

    let index = quick_corpus();
    assert!(index.search(&Bm25Query::new("unknown terms")).is_empty());
    assert!(index.search(&Bm25Query::new("")).is_empty());
    // Stopwords alone cannot match anything.
    assert!(index.search(&Bm25Query::new("the and of")).is_empty());
    assert!(index.search(&Bm25Query::new("zzz")).is_empty());
}

#[test]
fn tokenizer_handles_punctuation_case_and_cjk() {
    assert_eq!(tokenize("Hello, world!"), vec!["hello", "world"]);
    assert_eq!(tokenize("BM25 is FAST"), vec!["bm25", "is", "fast"]);
    assert_eq!(tokenize("a-b_c"), vec!["a", "b", "c"]);
    assert!(tokenize(" \t\n ").is_empty());
    assert_eq!(tokenize("你好世界"), vec!["你", "好", "世", "界"]);
    assert_eq!(tokenize("Rust检索"), vec!["rust", "检", "索"]);

    // The index scores CJK queries per character.
    let mut index = Bm25Index::default();
    index.add_document("cn", "你好世界", Metadata::new());
    let hits = index.search(&Bm25Query::new("世界"));
    assert_eq!(ids(&hits), vec!["cn"]);
    assert!(index.search(&Bm25Query::new("再见")).is_empty());
}

#[test]
fn configurable_stopwords_change_the_analyzer() {
    let mut custom = Bm25Index::default().with_stopwords(["quick"]);
    custom.add_document("d1", "quick fox", Metadata::new());
    assert!(custom.search(&Bm25Query::new("quick")).is_empty());
    assert_eq!(ids(&custom.search(&Bm25Query::new("fox"))), vec!["d1"]);

    let mut unfiltered = Bm25Index::default().without_stopwords();
    unfiltered.add_document("d1", "the fox", Metadata::new());
    assert_eq!(unfiltered.stats().total_tokens, 2);
    assert_eq!(ids(&unfiltered.search(&Bm25Query::new("the"))), vec!["d1"]);
}

#[test]
fn rrf_fusion_orders_by_reciprocal_rank_with_hand_computed_scores() {
    let vector_hits = vec![hit("v1", "alpha", 0.9), hit("v2", "beta", 0.4)];
    let sparse_hits = vec![hit("s1", "beta", 12.0), hit("s2", "gamma", 3.0)];
    let fused = rrf_fusion(vec![vector_hits, sparse_hits], 60);

    assert_eq!(texts(&fused), vec!["beta", "alpha", "gamma"]);
    assert!((fused[0].score - (1.0 / 62.0 + 1.0 / 61.0)).abs() < 1e-7);
    assert!((fused[1].score - 1.0 / 61.0).abs() < 1e-7);
    assert!((fused[2].score - 1.0 / 62.0).abs() < 1e-7);
    // The first-seen result object wins, so vector ids/metadata survive.
    assert_eq!(fused[1].id, "v1");
    assert_eq!(fused[0].id, "v2");

    assert_eq!(
        fused,
        rrf_fusion_default(vec![
            vec![hit("v1", "alpha", 0.9), hit("v2", "beta", 0.4)],
            vec![hit("s1", "beta", 12.0), hit("s2", "gamma", 3.0)],
        ])
    );
    assert_eq!(DEFAULT_RRF_K, 60);
}

#[test]
fn rrf_fusion_handles_ties_empty_input_and_duplicates() {
    let tied = rrf_fusion(
        vec![vec![hit("a", "alpha", 5.0)], vec![hit("b", "beta", 5.0)]],
        60,
    );
    assert_eq!(texts(&tied), vec!["alpha", "beta"]);
    assert!((tied[0].score - tied[1].score).abs() < 1e-7);

    assert!(rrf_fusion(Vec::new(), 60).is_empty());
    assert!(rrf_fusion_default(vec![Vec::new(), Vec::new()]).is_empty());

    let duplicated = rrf_fusion(
        vec![vec![hit("a", "alpha", 1.0), hit("b", "alpha", 1.0)]],
        60,
    );
    assert_eq!(duplicated.len(), 1);
    assert!((duplicated[0].score - (1.0 / 61.0 + 1.0 / 62.0)).abs() < 1e-7);
}

#[test]
fn deduplicate_keeps_first_seen_results_with_the_maximum_score() {
    let results = vec![
        hit("a", "alpha", 0.2),
        hit("b", "beta", 0.5),
        hit("c", "alpha", 0.7),
    ];
    let unique = deduplicate(results);
    assert_eq!(texts(&unique), vec!["alpha", "beta"]);
    assert_eq!(unique[0].id, "a");
    assert_eq!(unique[0].score, 0.7);
    assert_eq!(unique[1].score, 0.5);
    assert!(deduplicate(Vec::new()).is_empty());
}

#[tokio::test]
async fn sparse_retriever_projects_hits_and_rejects_other_modes() {
    let mut index = Bm25Index::default();
    let document = Document::with_id("doc-1", "source").with_metadata(metadata(&[("lang", "en")]));
    index.add_documents(std::slice::from_ref(&document));
    let chunk = TextChunk::from_document(&document, "quick brown fox", Some("chunk-1".into()));
    index.add_chunks(std::slice::from_ref(&chunk));
    let retriever = SparseRetriever::new(Arc::new(index));

    let results = retriever
        .retrieve("quick", 5, None, RetrieveMode::Sparse, None)
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].chunk_id.as_deref(), Some("chunk-1"));
    assert_eq!(results[0].doc_id.as_deref(), Some("doc-1"));
    assert_eq!(
        results[0].metadata.get("lang"),
        Some(&serde_json::json!("en"))
    );

    let filtered = retriever
        .retrieve(
            "quick",
            5,
            None,
            RetrieveMode::Sparse,
            Some(&MetadataFilter::eq("doc_id", "doc-1")),
        )
        .await
        .unwrap();
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].chunk_id.as_deref(), Some("chunk-1"));

    let error = retriever
        .retrieve("quick", 5, None, RetrieveMode::Hybrid, None)
        .await
        .unwrap_err();
    assert_eq!(
        error.kind(),
        openjiuwen_common::error::ErrorKind::Validation
    );
    assert!(error.to_string().contains("got hybrid"));

    let batch = retriever
        .batch_retrieve(
            &["quick".to_owned(), "brown".to_owned()],
            1,
            RetrieveMode::Sparse,
        )
        .await
        .unwrap();
    assert_eq!(batch.len(), 2);
    assert_eq!(batch[0].len(), 1);
    retriever.close().await;
}
