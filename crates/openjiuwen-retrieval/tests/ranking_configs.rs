#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::assert_is_empty
)]
//! Integration tests for the result-ranking configs and the fusion formulas.
//!
//! The Python `result_ranking.py` module is a re-export shim over Milvus ranker
//! configs, and the weighted/RRF formulas run server-side in Milvus, so these
//! expectations are derived from the documented Python defaults and the
//! hand-checkable formulas, not ported from Python tests.
//!
//! Everything runs offline: the ordering test drives two fixed channels, and the
//! end-to-end regression test uses [`HashEmbedding`], [`InMemoryVectorStore`],
//! and [`Bm25Index`] exactly like the crate's other retriever tests.

use std::sync::Arc;

use async_trait::async_trait;

use openjiuwen_common::error::{ErrorKind, Result};

use openjiuwen_retrieval::common::{
    RankConfig, RankMethod, RankerArgs, RrfRankConfig, WeightedRankConfig, CONTENT_DENSE,
    CONTENT_SPARSE, DEFAULT_RRF_RANK_K, NAME_DENSE, RANK_CHANNEL_COUNT,
};
use openjiuwen_retrieval::embedding::{Embedding, HashEmbedding};
use openjiuwen_retrieval::retriever::{
    rrf_fusion, rrf_fusion_channels, rrf_fusion_weighted, search_result_to_retrieval,
    weighted_fusion, HybridRetriever, RetrieveMode, Retriever, DEFAULT_RRF_K,
};
use openjiuwen_retrieval::sparse::{Bm25Config, Bm25Index};
use openjiuwen_retrieval::vector_store::{InMemoryVectorStore, VectorRecord, VectorStore};
use openjiuwen_retrieval::{Metadata, MetadataFilter, RetrievalResult, SearchResult};

const TOLERANCE: f32 = 1e-6;
const DIMENSION: usize = 64;

fn hit(id: &str, text: &str, score: f32) -> SearchResult {
    SearchResult {
        id: id.to_owned(),
        text: text.to_owned(),
        score,
        metadata: Metadata::new(),
    }
}

fn texts(results: &[SearchResult]) -> Vec<&str> {
    results.iter().map(|result| result.text.as_str()).collect()
}

fn assert_close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() < TOLERANCE,
        "expected {expected}, got {actual}"
    );
}

/// Channel returning a fixed list of hits, whatever the query.
struct FixedChannel {
    hits: Vec<SearchResult>,
}

#[async_trait]
impl Retriever for FixedChannel {
    async fn retrieve(
        &self,
        query: &str,
        top_k: usize,
        score_threshold: Option<f32>,
        mode: RetrieveMode,
        filters: Option<&MetadataFilter>,
    ) -> Result<Vec<RetrievalResult>> {
        let hits = self
            .retrieve_search_results(query, top_k, mode, filters)
            .await?;
        let mut results: Vec<RetrievalResult> =
            hits.into_iter().map(search_result_to_retrieval).collect();
        if let Some(threshold) = score_threshold {
            results.retain(|result| result.score >= threshold);
        }
        Ok(results)
    }

    async fn retrieve_search_results(
        &self,
        _query: &str,
        top_k: usize,
        _mode: RetrieveMode,
        _filters: Option<&MetadataFilter>,
    ) -> Result<Vec<SearchResult>> {
        Ok(self.hits.iter().take(top_k).cloned().collect())
    }
}

fn hybrid(dense: Vec<SearchResult>, sparse: Vec<SearchResult>) -> HybridRetriever {
    HybridRetriever::new(
        Arc::new(FixedChannel { hits: dense }),
        Arc::new(FixedChannel { hits: sparse }),
    )
}

// --- config defaults, validation, serde -----------------------------------

#[test]
fn weighted_defaults_match_python_and_serialize() {
    let config = WeightedRankConfig::default();
    assert_close(config.name_dense, 0.15);
    assert_close(config.content_dense, 0.6);
    assert_close(config.content_sparse, 0.25);
    assert_eq!(config.method_name(), "weighted");
    assert!(!config.higher_is_better());

    let value = serde_json::to_value(&config).unwrap();
    assert_eq!(value["name_dense"], serde_json::json!(0.15_f32));
    assert_eq!(value["content_dense"], serde_json::json!(0.6_f32));
    assert_eq!(value["content_sparse"], serde_json::json!(0.25_f32));
    assert_eq!(
        serde_json::from_value::<WeightedRankConfig>(value).unwrap(),
        config
    );
    // Missing fields fall back to the Python defaults and unknown fields are ignored.
    let parsed: WeightedRankConfig =
        serde_json::from_value(serde_json::json!({"unknown": 1})).unwrap();
    assert_eq!(parsed, config);
}

#[test]
fn weighted_validation_names_the_offending_field() {
    let error = WeightedRankConfig::new(0.15, 1.5, 0.25).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains("content_dense"), "{message}");
    assert!(message.contains("1.5"), "{message}");
    assert!(WeightedRankConfig::new(-0.1, 0.6, 0.25).is_err());
    assert!(WeightedRankConfig::new(0.0, 0.0, 0.0).is_ok());
}

#[test]
fn weighted_normalisation_drops_zeros_and_renormalises() {
    let defaults = WeightedRankConfig::default();
    let weights = defaults.normalized_weights();
    assert_eq!(weights.len(), RANK_CHANNEL_COUNT);
    assert_close(weights.iter().sum::<f32>(), 1.0);

    let no_name = WeightedRankConfig::new(0.0, 0.6, 0.25).unwrap();
    let weights = no_name.normalized_weights();
    assert_eq!(weights.len(), 2);
    assert_close(weights[0], 0.6 / 0.85);
    assert_close(weights[1], 0.25 / 0.85);
    assert_eq!(no_name.is_active(), [false, true, true]);

    let all_zero = WeightedRankConfig::new(0.0, 0.0, 0.0).unwrap();
    assert!(all_zero.normalized_weights().is_empty());
    assert_eq!(all_zero.is_active(), [false, false, false]);
}

#[test]
fn rrf_config_defaults_to_k_40_and_masks_channels() {
    assert_eq!(DEFAULT_RRF_RANK_K, 40);
    // The crate's own fusion default deliberately stays at 60.
    assert_eq!(DEFAULT_RRF_K, 60);
    assert_ne!(DEFAULT_RRF_RANK_K, DEFAULT_RRF_K);

    let config = RrfRankConfig::default();
    assert_eq!(config.k, 40);
    assert_eq!(config.args(), vec![40]);
    assert_eq!(config.is_active(), [true, true, true]);
    assert!(config.higher_is_better());

    let masked = RrfRankConfig {
        k: 10,
        name_dense: false,
        content_dense: true,
        content_sparse: false,
    };
    assert_eq!(masked.is_active(), [false, true, false]);
    assert_eq!(masked.method_name(), "rrf");
}

#[test]
fn rank_config_is_tagged_and_defaults_to_rrf() {
    let config = RankConfig::default();
    assert_eq!(config.method(), RankMethod::Rrf);
    assert_eq!(config.name(), "rrf");
    assert_eq!(config.rrf_k(), Some(40));
    assert_eq!(config.ranker_args(), RankerArgs::Rrf(40));
    assert_eq!(config.is_active(), [true, true, true]);

    let value = serde_json::to_value(&config).unwrap();
    assert_eq!(value["method"], "rrf");
    assert_eq!(value["k"], 40);
    assert_eq!(serde_json::from_value::<RankConfig>(value).unwrap(), config);

    let weighted = RankConfig::Weighted(WeightedRankConfig::default());
    let value = serde_json::to_value(&weighted).unwrap();
    assert_eq!(value["method"], "weighted");
    assert_eq!(
        serde_json::from_value::<RankConfig>(value).unwrap(),
        weighted
    );
    assert_eq!(weighted.method(), RankMethod::Weighted);
    assert_eq!(weighted.rrf_k(), None);
    assert!(!weighted.higher_is_better());
}

#[test]
fn channel_constants_follow_the_milvus_order() {
    assert_eq!(NAME_DENSE, 0);
    assert_eq!(CONTENT_DENSE, 1);
    assert_eq!(CONTENT_SPARSE, 2);
    assert_eq!(RANK_CHANNEL_COUNT, 3);
}

// --- weighted fusion -------------------------------------------------------

#[test]
fn weighted_fusion_normalises_dedupes_and_sorts() {
    let dense = vec![
        SearchResult {
            metadata: Metadata::from_iter([("source".to_owned(), serde_json::json!("vector"))]),
            ..hit("d1", "alpha", 0.9)
        },
        hit("d2", "beta", 0.5),
        hit("d3", "gamma", 0.1),
    ];
    let sparse = vec![hit("s1", "beta", 10.0), hit("s2", "delta", 2.0)];
    let fused = weighted_fusion(vec![dense, sparse], &[0.6, 0.25]).unwrap();

    assert_eq!(texts(&fused), vec!["alpha", "beta", "gamma", "delta"]);
    assert_close(fused[0].score, 0.6);
    assert_close(fused[1].score, 0.55);
    assert_close(fused[2].score, 0.0);
    assert_close(fused[3].score, 0.0);
    // First-seen metadata and ids survive.
    assert_eq!(fused[0].id, "d1");
    assert_eq!(
        fused[0].metadata.get("source"),
        Some(&serde_json::json!("vector"))
    );
}

#[test]
fn weighted_fusion_normalises_a_constant_channel_to_one() {
    let fused = weighted_fusion(
        vec![vec![hit("a", "alpha", 5.0), hit("b", "beta", 5.0)]],
        &[1.0],
    )
    .unwrap();
    assert_close(fused[0].score, 1.0);
    assert_close(fused[1].score, 1.0);
}

#[test]
fn weighted_fusion_rejects_mismatched_lengths() {
    let error = weighted_fusion(vec![Vec::new(), Vec::new()], &[1.0]).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("one weight per channel"));
}

// --- RRF with channel weights and masks -----------------------------------

#[test]
fn rrf_weighted_multiplies_channel_contributions() {
    let first = vec![hit("a", "alpha", 1.0), hit("b", "beta", 1.0)];
    let second = vec![hit("c", "beta", 1.0)];
    let fused = rrf_fusion_weighted(vec![first, second], 40, &[1.0, 0.5]).unwrap();
    // beta: 1/42 + 0.5/41 beats alpha: 1/41.
    assert_eq!(texts(&fused), vec!["beta", "alpha"]);
    assert_close(fused[0].score, 1.0 / 42.0 + 0.5 / 41.0);
    assert_close(fused[1].score, 1.0 / 41.0);
}

#[test]
fn rrf_channels_drop_inactive_channels() {
    let channels = vec![
        vec![hit("n", "name", 1.0)],
        vec![hit("d", "dense", 3.0)],
        vec![hit("s", "sparse", 5.0)],
    ];
    let config = RrfRankConfig {
        k: 40,
        name_dense: false,
        content_dense: true,
        content_sparse: false,
    };
    let fused = rrf_fusion_channels(channels, &config);
    assert_eq!(texts(&fused), vec!["dense"]);
    assert_close(fused[0].score, 1.0 / 41.0);

    // A zero-weight channel is dropped for exactly the same reason.
    let masked = rrf_fusion_weighted(
        vec![vec![hit("d", "dense", 1.0)], vec![hit("s", "sparse", 1.0)]],
        40,
        &[1.0, 0.0],
    )
    .unwrap();
    assert_eq!(texts(&masked), vec!["dense"]);
}

// --- hybrid retriever wiring ----------------------------------------------

#[tokio::test]
async fn weighted_config_changes_hybrid_ordering_predictably() {
    let dense = vec![
        hit("d1", "alpha", 0.9),
        hit("d2", "beta", 0.8),
        hit("d3", "gamma", 0.7),
    ];
    let sparse = vec![
        hit("s1", "gamma", 10.0),
        hit("s2", "beta", 9.0),
        hit("s3", "alpha", 8.0),
    ];

    let default = hybrid(dense.clone(), sparse.clone());
    assert_eq!(default.rrf_k(), DEFAULT_RRF_K);
    assert_eq!(default.rank_config().name(), "rrf");
    let default_results = default
        .retrieve("query", 3, None, RetrieveMode::Hybrid, None)
        .await
        .unwrap();
    // alpha and gamma tie on 1/61 + 1/63; alpha is seen first, beta is last.
    assert_eq!(texts_of(&default_results), vec!["alpha", "gamma", "beta"]);

    let weighted =
        hybrid(dense, sparse).with_rank_config(RankConfig::Weighted(WeightedRankConfig::default()));
    assert_eq!(weighted.rank_config().name(), "weighted");
    assert_eq!(weighted.rrf_k(), DEFAULT_RRF_K);
    let weighted_results = weighted
        .retrieve("query", 3, None, RetrieveMode::Hybrid, None)
        .await
        .unwrap();
    // Weighted ranker: alpha 0.6, beta 0.425, gamma 0.25.
    assert_eq!(texts_of(&weighted_results), vec!["alpha", "beta", "gamma"]);
}

fn texts_of(results: &[openjiuwen_retrieval::RetrievalResult]) -> Vec<&str> {
    results.iter().map(|result| result.text.as_str()).collect()
}

#[tokio::test]
async fn default_hybrid_still_matches_rrf_fusion_on_real_channels() {
    let embedder = HashEmbedding::new(DIMENSION);
    let store = InMemoryVectorStore::with_collection("chunks");
    let corpus = [
        "rust async runtime tokio",
        "python data science pandas",
        "vector search similarity index",
        "bm25 lexical scoring retrieval",
    ];
    let mut records = Vec::new();
    let mut index = Bm25Index::new(Bm25Config::default()).unwrap();
    for (position, text) in corpus.iter().enumerate() {
        let id = format!("chunk-{position}");
        records.push(VectorRecord {
            id: id.clone(),
            text: (*text).to_owned(),
            vector: embedder.embed_query(text).await.unwrap(),
            metadata: Metadata::new(),
            doc_id: Some(format!("doc-{position}")),
            chunk_id: Some(id.clone()),
        });
        index.add_document(id, *text, Metadata::new());
    }
    store.upsert(records).await.unwrap();

    let retriever = HybridRetriever::from_parts(
        Arc::new(store),
        Arc::new(HashEmbedding::new(DIMENSION)),
        Arc::new(index),
    );

    let query = "vector search retrieval";
    let top_k = 3;
    let candidates = top_k * retriever.candidate_factor();
    let dense = retriever
        .dense()
        .retrieve_search_results(query, candidates, RetrieveMode::Vector, None)
        .await
        .unwrap();
    let sparse = retriever
        .sparse()
        .retrieve_search_results(query, candidates, RetrieveMode::Sparse, None)
        .await
        .unwrap();

    let mut expected = rrf_fusion(vec![dense, sparse], DEFAULT_RRF_K);
    expected.truncate(top_k);
    let actual = retriever
        .retrieve_search_results(query, top_k, RetrieveMode::Hybrid, None)
        .await
        .unwrap();
    assert_eq!(texts(&actual), texts(&expected));
    for (actual, expected) in actual.iter().zip(&expected) {
        assert_close(actual.score, expected.score);
    }
}
