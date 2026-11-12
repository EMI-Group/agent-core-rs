//! Behaviour tests for the search layer: keyword ranking, catalog filters, and the
//! embedding-backed semantic index.
//!
//! Everything runs in-process and offline: BM25 is deterministic by construction and
//! the semantic tests use `openjiuwen_retrieval::HashEmbedding`, which derives vectors
//! purely from the input text. No network, no clock-dependent assertions.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::float_cmp)]
#![allow(clippy::assert_is_empty)]

use std::sync::Arc;

use openjiuwen_retrieval::{Embedding, HashEmbedding};
use openjiuwen_symphony::model::{CapabilityDescriptor, CapabilityIo};
use openjiuwen_symphony::search::{
    search_descriptors, tokenize, LexicalIndex, SearchHit, SearchRequest, SemanticIndex,
    DEFAULT_TOP_K,
};
use serde_json::json;

/// Builds a descriptor with a minimal, fully specified card.
fn descriptor(
    capability_type: &str,
    id: &str,
    name: &str,
    description: &str,
) -> CapabilityDescriptor {
    CapabilityDescriptor::new(capability_type, id, name)
        .with_description(description)
        .with_source("builtin")
        .with_available(true)
        .with_inputs(vec![CapabilityIo::new("raw", "string")])
}

/// The catalog used by the keyword ranking tests.
fn catalog() -> Vec<CapabilityDescriptor> {
    vec![
        descriptor("tool", "sum", "Sum Numbers", "adds two numbers together"),
        descriptor(
            "tool",
            "translate",
            "Translate Text",
            "translates text between languages",
        ),
        descriptor(
            "workflow",
            "weather",
            "Weather Forecast",
            "fetches the forecast for a city",
        ),
    ]
}

fn qualified_ids(hits: &[SearchHit]) -> Vec<String> {
    hits.iter()
        .map(|hit| hit.descriptor.qualified_id())
        .collect()
}

// ---------------------------------------------------------------------------
// tokenize
// ---------------------------------------------------------------------------

#[test]
fn tokenize_folds_case_and_splits_on_punctuation() {
    assert_eq!(tokenize("Adds, Two\nnumbers!"), ["adds", "two", "numbers"]);
    assert_eq!(tokenize("A-B_C"), ["a", "b", "c"]);
}

#[test]
fn tokenize_drops_empties_and_handles_blanks() {
    assert!(tokenize("").is_empty());
    assert!(tokenize("   \t\n ").is_empty());
    assert!(tokenize("!!! ???").is_empty());
}

#[test]
fn tokenize_keeps_unicode_alphanumerics() {
    assert_eq!(tokenize("Grüße 世界 42"), ["grüße", "世界", "42"]);
}

// ---------------------------------------------------------------------------
// LexicalIndex
// ---------------------------------------------------------------------------

#[test]
fn best_keyword_match_ranks_first_with_matched_terms() {
    let index = LexicalIndex::build(catalog());
    assert_eq!(index.len(), 3);
    assert!(!index.is_empty());

    let hits = index.search("adds numbers", 10);
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].qualified_id, "tool:sum");
    assert_eq!(hits[0].matched_terms, ["adds", "numbers"]);
    assert!(hits[0].score > 0.0);
}

#[test]
fn more_query_terms_outrank_fewer_matches() {
    let descriptors = vec![
        descriptor("tool", "partial", "Partial", "alpha"),
        descriptor("tool", "full", "Full", "alpha beta gamma"),
        descriptor("tool", "unrelated", "Unrelated", "delta"),
    ];
    let index = LexicalIndex::build(descriptors);
    let hits = index.search("alpha beta gamma", 10);
    let ids: Vec<&str> = hits.iter().map(|hit| hit.qualified_id.as_str()).collect();
    assert_eq!(ids, ["tool:full", "tool:partial"]);
    assert_eq!(hits[0].matched_terms, ["alpha", "beta", "gamma"]);
    assert_eq!(hits[1].matched_terms, ["alpha"]);
    assert!(hits[0].score > hits[1].score);
}

#[test]
fn rankings_are_deterministic_across_searches() {
    let index = LexicalIndex::build(catalog());
    let first = index.search("numbers text forecast", 10);
    let second = index.search("numbers text forecast", 10);
    assert_eq!(first, second);
    assert_eq!(first.len(), 3);
}

#[test]
fn equal_scores_are_ordered_by_qualified_id() {
    let descriptors = vec![
        descriptor("tool", "zebra", "Zebra", "shared words"),
        descriptor("tool", "alpha", "Alpha", "shared words"),
    ];
    let index = LexicalIndex::build(descriptors);
    let ids: Vec<String> = index
        .search("shared words", 10)
        .into_iter()
        .map(|hit| hit.qualified_id)
        .collect();
    assert_eq!(ids, ["tool:alpha", "tool:zebra"]);
}

#[test]
fn top_k_truncates_and_zero_means_unlimited() {
    let descriptors = vec![
        descriptor("tool", "a", "Alpha", "shared words"),
        descriptor("tool", "b", "Beta", "shared words"),
        descriptor("tool", "c", "Gamma", "shared words"),
    ];
    let index = LexicalIndex::build(descriptors);
    assert_eq!(index.search("shared", 1).len(), 1);
    assert_eq!(index.search("shared", 2).len(), 2);
    assert_eq!(index.search("shared", 0).len(), 3);
    assert_eq!(index.search("shared", 99).len(), 3);
}

#[test]
fn blank_and_unknown_queries_return_no_hits() {
    let index = LexicalIndex::build(catalog());
    assert!(index.search("", 10).is_empty());
    assert!(index.search("   ", 10).is_empty());
    assert!(index.search("nonexistentterm", 10).is_empty());
}

#[test]
fn empty_index_has_no_hits() {
    let index = LexicalIndex::default();
    assert!(index.is_empty());
    assert_eq!(index.len(), 0);
    assert!(index.search("anything", 10).is_empty());
}

// ---------------------------------------------------------------------------
// search_descriptors — filters
// ---------------------------------------------------------------------------

#[test]
fn for_text_defaults_to_ten_hits_and_no_filters() {
    let request = SearchRequest::for_text("numbers");
    assert_eq!(request.text.as_deref(), Some("numbers"));
    assert_eq!(request.top_k, DEFAULT_TOP_K);
    assert_eq!(request.top_k, 10);
    assert!(request.capability_types.is_empty());
    assert!(request.tags.is_empty());
    assert!(request.exclude_tags.is_empty());
    assert!(request.available.is_none());
    assert!(request.metadata_filter.is_none());
    assert!(request.min_score.is_none());
    assert_eq!(SearchRequest::default().top_k, 10);
}

#[test]
fn filter_only_requests_return_every_descriptor_by_qualified_id() {
    let descriptors = vec![
        descriptor("tool", "b", "Beta", "second"),
        descriptor("workflow", "a", "Alpha", "first"),
    ];
    let hits = search_descriptors(&descriptors, &SearchRequest::default()).unwrap();
    assert_eq!(qualified_ids(&hits), ["tool:b", "workflow:a"]);
    assert!(hits.iter().all(|hit| hit.score == 0.0));
    assert!(hits.iter().all(|hit| hit.matched_terms.is_empty()));
}

#[test]
fn capability_types_filter_is_exact() {
    let descriptors = catalog();
    let request = SearchRequest {
        capability_types: vec!["tool".to_string()],
        ..SearchRequest::default()
    };
    let hits = search_descriptors(&descriptors, &request).unwrap();
    assert_eq!(qualified_ids(&hits), ["tool:sum", "tool:translate"]);

    let unknown = SearchRequest {
        capability_types: vec!["skill".to_string()],
        ..SearchRequest::default()
    };
    assert!(search_descriptors(&descriptors, &unknown)
        .unwrap()
        .is_empty());
}

#[test]
fn tags_filter_requires_all_tags_ignoring_case() {
    let descriptors = vec![
        descriptor("tool", "sum", "Sum", "adds numbers").with_tags(["Math", "Numbers"]),
        descriptor("tool", "translate", "Translate", "translates").with_tags(["text"]),
    ];
    let request = SearchRequest {
        tags: vec!["math".to_string(), "NUMBERS".to_string()],
        ..SearchRequest::default()
    };
    let hits = search_descriptors(&descriptors, &request).unwrap();
    assert_eq!(qualified_ids(&hits), ["tool:sum"]);

    let too_specific = SearchRequest {
        tags: vec!["math".to_string(), "text".to_string()],
        ..SearchRequest::default()
    };
    assert!(search_descriptors(&descriptors, &too_specific)
        .unwrap()
        .is_empty());
}

#[test]
fn exclude_tags_filter_drops_matching_descriptors() {
    let descriptors = vec![
        descriptor("tool", "sum", "Sum", "adds numbers").with_tags(["math"]),
        descriptor("tool", "translate", "Translate", "translates").with_tags(["text"]),
    ];
    let request = SearchRequest {
        exclude_tags: vec!["Math".to_string()],
        ..SearchRequest::default()
    };
    let hits = search_descriptors(&descriptors, &request).unwrap();
    assert_eq!(qualified_ids(&hits), ["tool:translate"]);
}

#[test]
fn available_filter_keeps_the_requested_flag() {
    let descriptors = vec![
        descriptor("tool", "sum", "Sum", "adds numbers").with_available(true),
        descriptor("tool", "translate", "Translate", "translates").with_available(false),
    ];
    let request = SearchRequest {
        available: Some(false),
        ..SearchRequest::default()
    };
    let hits = search_descriptors(&descriptors, &request).unwrap();
    assert_eq!(qualified_ids(&hits), ["tool:translate"]);
}

#[test]
fn metadata_filter_uses_the_shared_json_predicate() {
    let descriptors = vec![
        descriptor("tool", "sum", "Sum", "adds numbers").with_source("builtin"),
        descriptor("tool", "remote", "Remote", "remote work").with_source("remote"),
    ];
    let request = SearchRequest {
        metadata_filter: Some(json!({"source": "builtin"})),
        ..SearchRequest::default()
    };
    let hits = search_descriptors(&descriptors, &request).unwrap();
    assert_eq!(qualified_ids(&hits), ["tool:sum"]);

    let operator = SearchRequest {
        metadata_filter: Some(json!({"source": {"$ne": "builtin"}})),
        ..SearchRequest::default()
    };
    let hits = search_descriptors(&descriptors, &operator).unwrap();
    assert_eq!(qualified_ids(&hits), ["tool:remote"]);
}

#[test]
fn filters_and_text_combine() {
    let descriptors = vec![
        descriptor("tool", "sum", "Sum Numbers", "adds two numbers").with_tags(["math"]),
        descriptor("tool", "mean", "Mean Numbers", "averages numbers").with_tags(["math"]),
        descriptor("workflow", "report", "Numbers Report", "reports numbers"),
    ];
    let request = SearchRequest {
        text: Some("numbers".to_string()),
        capability_types: vec!["tool".to_string()],
        tags: vec!["math".to_string()],
        ..SearchRequest::default()
    };
    let hits = search_descriptors(&descriptors, &request).unwrap();
    assert_eq!(hits.len(), 2);
    assert!(hits
        .iter()
        .all(|hit| hit.descriptor.capability_type == "tool"));
    assert!(hits.iter().all(|hit| !hit.matched_terms.is_empty()));
}

// ---------------------------------------------------------------------------
// search_descriptors — scoring and paging
// ---------------------------------------------------------------------------

#[test]
fn text_requests_rank_by_bm25() {
    let hits = search_descriptors(&catalog(), &SearchRequest::for_text("adds numbers")).unwrap();
    assert_eq!(qualified_ids(&hits), ["tool:sum"]);
    assert_eq!(hits[0].matched_terms, ["adds", "numbers"]);
}

#[test]
fn min_score_filters_low_scoring_hits() {
    let descriptors = vec![
        descriptor("tool", "full", "Full", "alpha beta"),
        descriptor("tool", "half", "Half", "alpha"),
        descriptor("tool", "none", "None", "gamma"),
    ];
    let request = SearchRequest {
        text: Some("alpha beta".to_string()),
        ..SearchRequest::default()
    };
    let all = search_descriptors(&descriptors, &request).unwrap();
    assert_eq!(qualified_ids(&all), ["tool:full", "tool:half"]);
    let top_score = all[0].score;
    assert!(top_score > all[1].score);

    let explicit = SearchRequest {
        min_score: Some(top_score),
        ..request.clone()
    };
    let best = search_descriptors(&descriptors, &explicit).unwrap();
    assert_eq!(qualified_ids(&best), ["tool:full"]);

    let impossible = SearchRequest {
        min_score: Some(f32::MAX),
        ..request
    };
    assert!(search_descriptors(&descriptors, &impossible)
        .unwrap()
        .is_empty());
}

#[test]
fn top_k_truncates_and_zero_is_unlimited() {
    let descriptors = catalog();
    let limited = SearchRequest {
        text: Some("numbers text forecast".to_string()),
        top_k: 2,
        ..SearchRequest::default()
    };
    assert_eq!(search_descriptors(&descriptors, &limited).unwrap().len(), 2);

    let unlimited = SearchRequest {
        top_k: 0,
        ..limited.clone()
    };
    assert_eq!(
        search_descriptors(&descriptors, &unlimited).unwrap().len(),
        3
    );

    let filter_only_limited = SearchRequest {
        top_k: 1,
        ..SearchRequest::default()
    };
    assert_eq!(
        search_descriptors(&descriptors, &filter_only_limited)
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn blank_text_falls_back_to_filter_only_ordering() {
    let descriptors = vec![
        descriptor("tool", "b", "Beta", "second"),
        descriptor("tool", "a", "Alpha", "first"),
    ];
    let hits = search_descriptors(&descriptors, &SearchRequest::for_text("  ")).unwrap();
    assert_eq!(qualified_ids(&hits), ["tool:a", "tool:b"]);
    assert!(hits.iter().all(|hit| hit.score == 0.0));
}

#[test]
fn empty_catalog_and_blank_query_are_harmless() {
    assert!(search_descriptors(&[], &SearchRequest::default())
        .unwrap()
        .is_empty());
    assert!(search_descriptors(&[], &SearchRequest::for_text("sum"))
        .unwrap()
        .is_empty());
}

// ---------------------------------------------------------------------------
// SemanticIndex
// ---------------------------------------------------------------------------

/// A deterministic embedder shared by the semantic tests.
fn embedder() -> Arc<dyn Embedding> {
    Arc::new(HashEmbedding::new(64))
}

#[tokio::test]
async fn semantic_index_reports_the_indexed_count_and_finds_exact_text() {
    let index = SemanticIndex::new(embedder());
    let descriptors = catalog();
    assert_eq!(index.index(&descriptors).await.unwrap(), 3);

    let hits = index.search("adds two numbers", 10).await.unwrap();
    assert_eq!(hits.len(), 3);
    assert_eq!(hits[0].descriptor.qualified_id(), "tool:sum");
    assert!(hits[0].matched_terms.is_empty());
    // Scores are ordered descending.
    for window in hits.windows(2) {
        assert!(window[0].score >= window[1].score);
    }
}

#[tokio::test]
async fn semantic_hits_reconstruct_the_original_descriptors() {
    let index = SemanticIndex::new(embedder());
    let descriptors = catalog();
    index.index(&descriptors).await.unwrap();

    let mut hits = index.search("weather forecast city", 0).await.unwrap();
    assert_eq!(hits.len(), 3);
    let mut actual: Vec<CapabilityDescriptor> = hits.drain(..).map(|hit| hit.descriptor).collect();
    actual.sort_by_key(CapabilityDescriptor::qualified_id);
    let mut expected = descriptors;
    expected.sort_by_key(CapabilityDescriptor::qualified_id);
    assert_eq!(actual, expected);
}

#[tokio::test]
async fn semantic_upsert_replaces_the_existing_record() {
    let index = SemanticIndex::new(embedder());
    let descriptors = catalog();
    index.index(&descriptors).await.unwrap();

    let updated = descriptor(
        "tool",
        "sum",
        "Sum Numbers",
        "adds two numbers and multiplies them",
    );
    index.upsert(&updated).await.unwrap();

    let hits = index.search("adds two numbers", 0).await.unwrap();
    assert_eq!(hits.len(), 3);
    let sum = hits
        .iter()
        .find(|hit| hit.descriptor.capability_id == "sum")
        .unwrap();
    assert_eq!(sum.descriptor.description, updated.description);
}

#[tokio::test]
async fn semantic_remove_reports_presence_and_drops_the_record() {
    let index = SemanticIndex::new(embedder());
    let descriptors = catalog();
    index.index(&descriptors).await.unwrap();

    assert!(index.remove(&descriptors[0]).await.unwrap());
    assert!(!index.remove(&descriptors[0]).await.unwrap());

    let hits = index.search("adds two numbers", 0).await.unwrap();
    assert_eq!(hits.len(), 2);
    assert!(hits
        .iter()
        .all(|hit| hit.descriptor.qualified_id() != "tool:sum"));
}

#[tokio::test]
async fn semantic_clear_empties_the_store() {
    let index = SemanticIndex::new(embedder());
    index.index(&catalog()).await.unwrap();
    index.clear().await.unwrap();
    assert!(index
        .search("adds two numbers", 0)
        .await
        .unwrap()
        .is_empty());
    assert!(index
        .search("adds two numbers", 5)
        .await
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn semantic_top_k_truncates_and_zero_is_unlimited() {
    let index = SemanticIndex::new(embedder());
    index.index(&catalog()).await.unwrap();
    assert_eq!(
        index
            .search("numbers text forecast", 1)
            .await
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        index
            .search("numbers text forecast", 2)
            .await
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        index
            .search("numbers text forecast", 0)
            .await
            .unwrap()
            .len(),
        3
    );
}

#[tokio::test]
async fn semantic_searches_are_deterministic() {
    let index = SemanticIndex::new(embedder());
    index.index(&catalog()).await.unwrap();
    let first = index.search("numbers text forecast", 0).await.unwrap();
    let second = index.search("numbers text forecast", 0).await.unwrap();
    assert_eq!(first, second);
}

#[tokio::test]
async fn semantic_blank_query_and_empty_index_return_no_hits() {
    let index = SemanticIndex::new(embedder());
    index.index(&catalog()).await.unwrap();
    assert!(index.search("   ", 5).await.unwrap().is_empty());

    let empty = SemanticIndex::new(embedder());
    assert!(empty.search("numbers", 5).await.unwrap().is_empty());
    assert_eq!(empty.index(&[]).await.unwrap(), 0);
}

#[tokio::test]
async fn semantic_rejects_blank_retrieval_text() {
    let index = SemanticIndex::new(embedder());
    let blank = CapabilityDescriptor {
        name: "  ".to_string(),
        ..descriptor("tool", "blank", "Blank", "")
    };
    // A card with no name, description, classification, or tags has no text to embed.
    assert!(index.index(&[blank]).await.is_err());
}
