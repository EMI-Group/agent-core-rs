#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::assert_is_empty
)]
//! Python-parity contract for the sparse, rewrite, and graph-retriever surfaces.
//!
//! The reference Python SDK is the contract: `QueryRewriter.rewrite(query)`
//! returns `{before, intention, standalone_query, references, missing,
//! typo[{original, corrected, reason}], gibberish, from_history}`,
//! `SparseRetriever` only answers `mode="sparse"`, `rrf_fusion(results, k=60)` is
//! keyed by result text, `GraphRetriever` expands `graph_hops=2` by default, and
//! `AgenticRetriever(retriever, llm_client, max_iter=2)` needs an LLM.
//!
//! Only assertions that the rest of the suite does not already make live here;
//! the existing coverage is:
//!
//! - `tests/sparse.rs` — hand-computed BM25 scores, `idf`/`k1`/`b`, filters,
//!   `DEFAULT_RRF_K == 60`, text-keyed RRF fusion at `k = 60`, `deduplicate`,
//!   `SparseRetriever` projecting hits and rejecting `Hybrid`, batch retrieval.
//! - `tests/rewrite.rs` — the full `RewriteOutput` JSON key set and round trip,
//!   partial-object defaults for `intention`/`references`/`missing`, heuristic
//!   and model rewriting, JSON repair, and the fallback paths.
//! - `tests/retrievers.rs` — `VectorRetriever` rejecting `Sparse`/`Hybrid`,
//!   hybrid fusion, `GraphRetriever` defaults (`max_hops == 2`,
//!   `expansion_limit == 5`, triple mode `Sparse`), `AgenticConfig::max_iter == 2`
//!   and `AgenticRetriever::model().is_none()` (the offline default).
//! - `tests/graph.rs` — triple/entity serialization, extraction families, and
//!   graph-store traversal.
//!
//! So this file adds: `RetrieveMode`'s wire names and serde contract, the
//! `SparseRetriever` rejection of `Vector` (and through
//! `retrieve_search_results`), `AgenticConfig` defaults and serde defaults for
//! every field, `TypoFix`'s own JSON shape and field defaults,
//! `RewriteOutput`'s required-versus-defaulted fields, and RRF with a
//! non-default rank constant.
//!
//! Intentional Rust deviations are called out where they encode an assertion:
//! Rust owns the BM25 index, the graph retriever is store-backed and LLM-free,
//! the rewriter adds `keywords`/`expand`, and agentic retrieval has a
//! deterministic fallback so it works with no model.

use std::sync::Arc;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_retrieval::retriever::agentic::{
    DEFAULT_MAX_ITER, DEFAULT_PASSAGES_PER_ITERATION, DEFAULT_SUFFICIENCY_THRESHOLD,
};
use openjiuwen_retrieval::sparse::Bm25Index;
use openjiuwen_retrieval::{
    rrf_fusion, AgenticConfig, Metadata, RetrievalResult, RetrieveMode, Retriever, RewriteOutput,
    SearchResult, SparseRetriever, TypoFix,
};
use serde_json::json;

/// A store-level hit; RRF only looks at the text, id, and score.
fn hit(id: &str, text: &str, score: f32) -> SearchResult {
    SearchResult {
        id: id.to_owned(),
        text: text.to_owned(),
        score,
        metadata: Metadata::new(),
    }
}

#[test]
fn retrieve_mode_keeps_the_python_wire_names() {
    // Python spells the mode as `Literal["vector", "sparse", "hybrid"]`.
    for (mode, name) in [
        (RetrieveMode::Vector, "vector"),
        (RetrieveMode::Sparse, "sparse"),
        (RetrieveMode::Hybrid, "hybrid"),
    ] {
        assert_eq!(mode.as_str(), name);
        assert_eq!(serde_json::to_value(mode).unwrap(), json!(name));
        let restored: RetrieveMode =
            serde_json::from_value(serde_json::Value::String(name.to_owned())).unwrap();
        assert_eq!(restored, mode);
    }

    // The serde default is the Python default (`"hybrid"`).
    assert_eq!(RetrieveMode::default(), RetrieveMode::Hybrid);
    assert_eq!(
        serde_json::from_str::<RetrieveMode>(r#""hybrid""#).unwrap(),
        RetrieveMode::default()
    );

    // Anything outside the Python literal is rejected rather than guessed at.
    let error = serde_json::from_str::<RetrieveMode>(r#""dense""#).unwrap_err();
    assert!(
        error.to_string().contains("unknown variant"),
        "unexpected message: {error}"
    );
}

#[tokio::test]
async fn sparse_retriever_rejects_every_mode_but_sparse() {
    // Deviation: Python's `SparseRetriever` delegates to the store's BM25
    // function; Rust owns the index, so the retriever wraps an `Arc<Bm25Index>`.
    let mut index = Bm25Index::default();
    index.add_document("d1", "quick brown fox", Metadata::new());
    let retriever = SparseRetriever::new(Arc::new(index));

    // `tests/sparse.rs` covers `Hybrid`; Python's mode guard applies to `Vector`
    // too, and the error names the only supported mode.
    for mode in [RetrieveMode::Vector, RetrieveMode::Hybrid] {
        let error = retriever
            .retrieve("quick", 5, None, mode, None)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.to_string().contains("only supports 'sparse' mode"),
            "unexpected message: {error}"
        );
        assert!(
            error
                .to_string()
                .contains(&format!("got {}", mode.as_str())),
            "unexpected message: {error}"
        );

        // The raw-results API enforces the same contract.
        let error = retriever
            .retrieve_search_results("quick", 5, mode, None)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("only supports 'sparse' mode"));
    }

    let results: Vec<RetrievalResult> = retriever
        .retrieve("quick", 5, None, RetrieveMode::Sparse, None)
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].text, "quick brown fox");
}

#[test]
fn agentic_config_defaults_match_python_and_deserialise_from_an_empty_object() {
    // Python: `AgenticRetriever(retriever, llm_client, max_iter=2)`.
    let config = AgenticConfig::default();
    assert_eq!(config.max_iter, DEFAULT_MAX_ITER);
    assert_eq!(DEFAULT_MAX_ITER, 2);

    // Deviation: Python drives the sufficiency test and passage window from the
    // LLM prompt; the Rust loop keeps them as explicit knobs with the documented
    // defaults so it can stop deterministically without a model.
    assert_eq!(config.sufficiency_threshold, DEFAULT_SUFFICIENCY_THRESHOLD);
    assert_eq!(DEFAULT_SUFFICIENCY_THRESHOLD, 0.6);
    assert_eq!(
        config.passages_per_iteration,
        DEFAULT_PASSAGES_PER_ITERATION
    );
    assert_eq!(DEFAULT_PASSAGES_PER_ITERATION, 5);

    // Every field defaults, so an empty JSON object is the default config.
    let from_json: AgenticConfig = serde_json::from_str("{}").unwrap();
    assert_eq!(from_json, config);

    // A partial config keeps the defaults for the fields it omits.
    let partial: AgenticConfig = serde_json::from_str(r#"{"max_iter": 4}"#).unwrap();
    assert_eq!(partial.max_iter, 4);
    assert_eq!(partial.sufficiency_threshold, 0.6);
    assert_eq!(partial.passages_per_iteration, 5);

    let serialized = serde_json::to_value(config).unwrap();
    assert_eq!(serialized["max_iter"], json!(2));
    assert_eq!(serialized["passages_per_iteration"], json!(5));
    let round_tripped: AgenticConfig = serde_json::from_value(serialized).unwrap();
    assert_eq!(round_tripped, config);
}

#[test]
fn typo_fix_round_trips_with_the_python_field_names() {
    // Python's `typo` entries are `{original, corrected, reason}`; the existing
    // rewrite test only checks the key inside a whole `RewriteOutput`.
    let typo = TypoFix::new("rustt", "rust", "spelling");
    assert_eq!(
        serde_json::to_value(&typo).unwrap(),
        json!({"original": "rustt", "corrected": "rust", "reason": "spelling"})
    );
    let restored: TypoFix = serde_json::from_value(serde_json::to_value(&typo).unwrap()).unwrap();
    assert_eq!(restored, typo);

    // Every field defaults, so a partial correction still parses.
    let partial: TypoFix = serde_json::from_value(json!({"original": "rustt"})).unwrap();
    assert_eq!(partial.original, "rustt");
    assert_eq!(partial.corrected, "");
    assert_eq!(partial.reason, "");

    // The same shape inside a rewrite result.
    let output: RewriteOutput = serde_json::from_value(json!({
        "before": "rustt ownership",
        "standalone_query": "rust ownership",
        "typo": [serde_json::to_value(&typo).unwrap()],
    }))
    .unwrap();
    assert_eq!(output.typo, vec![typo]);
}

#[test]
fn rewrite_output_requires_before_and_standalone_query() {
    // Python always fills these two from the original query; they are the only
    // required keys, and everything else defaults.
    let error = serde_json::from_str::<RewriteOutput>("{}").unwrap_err();
    assert!(
        error.to_string().contains("missing field"),
        "unexpected message: {error}"
    );

    let error = serde_json::from_str::<RewriteOutput>(r#"{"before": "q"}"#).unwrap_err();
    assert!(
        error.to_string().contains("standalone_query"),
        "unexpected message: {error}"
    );

    let partial: RewriteOutput =
        serde_json::from_str(r#"{"before": "q", "standalone_query": "q"}"#).unwrap();
    // `tests/rewrite.rs` asserts the `intention`/`references`/`missing` defaults;
    // the remaining Python keys default the same way.
    assert!(partial.typo.is_empty());
    assert!(partial.gibberish.is_empty());
    assert_eq!(partial.from_history, "");

    // Unknown keys are tolerated, matching a Python dict round trip.
    let tolerant: RewriteOutput =
        serde_json::from_str(r#"{"before": "q", "standalone_query": "q", "extra": 1}"#).unwrap();
    assert_eq!(tolerant, partial);
}

#[test]
fn rrf_fusion_honours_a_custom_rank_constant() {
    // Python: `rrf_fusion(results, k=60)`; `k` is a parameter, and the fusion is
    // keyed by result text. `tests/sparse.rs` fixes k = 60, so this pins the
    // parameter with a different constant.
    let fused = rrf_fusion(
        vec![
            vec![hit("a", "alpha", 0.1)],
            vec![hit("b", "alpha", 0.2), hit("c", "beta", 0.3)],
        ],
        10,
    );

    assert_eq!(fused.len(), 2);
    // `alpha` appears first in both lists, so its two 1-based ranks are 1 and 1.
    assert_eq!(fused[0].text, "alpha");
    assert!((fused[0].score - (1.0 / 11.0 + 1.0 / 11.0)).abs() < 1e-7);
    // The first-seen object wins, so `alpha` keeps the id from the first list.
    assert_eq!(fused[0].id, "a");
    // `beta` is only in the second list, at rank 2.
    assert_eq!(fused[1].text, "beta");
    assert!((fused[1].score - 1.0 / 12.0).abs() < 1e-7);
}
