#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::assert_is_empty
)]
//! Integration tests for the four retrievers: vector, hybrid, graph, agentic.
//!
//! Everything runs offline and deterministically: [`HashEmbedding`] for the
//! dense channel, [`Bm25Index`] for the sparse one, [`InMemoryVectorStore`] and
//! [`InMemoryGraphStore`] for storage, and scripted rewriters/models instead of
//! live LLM calls.
//!
//! Python has no offline story for these retrievers (its vector/hybrid/graph
//! retrievers need Milvus/Chroma and its agentic retriever needs an LLM), so the
//! expectations here are derived from the documented Rust behaviour and from the
//! hand-checkable RRF formula in `retriever::fusion`, not ported from Python tests.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use openjiuwen_common::error::{ErrorKind, Result};
use openjiuwen_retrieval::embedding::{Embedding, HashEmbedding};
use openjiuwen_retrieval::graph::{GraphStore, InMemoryGraphStore, Triple};
use openjiuwen_retrieval::model::StaticCompletionModel;
use openjiuwen_retrieval::retriever::{
    AgenticConfig, AgenticRetriever, AgenticStopReason, GraphRetriever, HybridRetriever,
    RetrieveMode, Retriever, SparseRetriever, VectorRetriever,
};
use openjiuwen_retrieval::rewrite::{
    HeuristicQueryRewriter, QueryRewriter, RewriteOutput, RewriteRequest,
};
use openjiuwen_retrieval::sparse::{Bm25Config, Bm25Index};
use openjiuwen_retrieval::vector_store::{InMemoryVectorStore, VectorRecord, VectorStore};
use openjiuwen_retrieval::{Document, Metadata};

/// Embedding dimension used by every dense channel in this file.
const DIMENSION: usize = 64;

// --- shared helpers -------------------------------------------------------

/// BM25 index over `(id, text)` pairs.
fn bm25_index(documents: &[(&str, &str)]) -> Arc<Bm25Index> {
    let mut index = Bm25Index::new(Bm25Config::default()).unwrap();
    let documents: Vec<Document> = documents
        .iter()
        .map(|(id, text)| Document::with_id(*id, *text))
        .collect();
    index.add_documents(&documents);
    Arc::new(index)
}

/// Vector store whose records carry the embedding of their own text.
async fn dense_store(texts: &[&str]) -> Arc<dyn VectorStore> {
    let embedder = HashEmbedding::new(DIMENSION);
    let store = InMemoryVectorStore::with_collection("chunks");
    let mut records = Vec::new();
    for (position, text) in texts.iter().enumerate() {
        records.push(VectorRecord {
            id: format!("chunk-{position}"),
            text: (*text).to_owned(),
            vector: embedder.embed_query(text).await.unwrap(),
            metadata: Metadata::new(),
            doc_id: Some(format!("doc-{position}")),
            chunk_id: Some(format!("chunk-{position}")),
        });
    }
    store.upsert(records).await.unwrap();
    Arc::new(store)
}

fn texts_of(results: &[openjiuwen_retrieval::RetrievalResult]) -> Vec<&str> {
    results.iter().map(|result| result.text.as_str()).collect()
}

/// Rewriter replaying scripted standalone queries and recording its requests.
#[derive(Debug, Default)]
struct ScriptedRewriter {
    outputs: Vec<String>,
    calls: Mutex<usize>,
    requests: Mutex<Vec<RewriteRequest>>,
}

impl ScriptedRewriter {
    fn new(outputs: &[&str]) -> Arc<Self> {
        Arc::new(Self {
            outputs: outputs.iter().map(|output| (*output).to_owned()).collect(),
            calls: Mutex::new(0),
            requests: Mutex::new(Vec::new()),
        })
    }

    fn calls(&self) -> usize {
        *self.calls.lock().unwrap()
    }

    fn requests(&self) -> Vec<RewriteRequest> {
        self.requests.lock().unwrap().clone()
    }
}

#[async_trait]
impl QueryRewriter for ScriptedRewriter {
    async fn rewrite(&self, request: &RewriteRequest) -> Result<RewriteOutput> {
        self.requests.lock().unwrap().push(request.clone());
        let index = {
            let mut calls = self.calls.lock().unwrap();
            let index = *calls;
            *calls += 1;
            index
        };
        let last = self.outputs.len().saturating_sub(1);
        let output = self
            .outputs
            .get(index.min(last))
            .cloned()
            .unwrap_or_default();
        Ok(RewriteOutput::from_query(output))
    }

    async fn keywords(&self, query: &str) -> Result<Vec<String>> {
        Ok(HeuristicQueryRewriter::default().extract_keywords(query))
    }

    async fn expand(&self, query: &str) -> Result<Vec<String>> {
        Ok(vec![query.to_owned()])
    }
}

/// Corpus shared by the agentic tests: `chunk-a` answers "ownership", `chunk-b`
/// answers "borrow checker", and only `chunk-a` yields a rule-based triple.
fn agentic_index() -> Arc<Bm25Index> {
    bm25_index(&[
        ("chunk-a", "Acme uses Rust. Ownership rules apply."),
        ("chunk-b", "The borrow checker notes"),
    ])
}

fn agentic_retriever(rewriter: Arc<ScriptedRewriter>) -> AgenticRetriever {
    AgenticRetriever::new(Arc::new(SparseRetriever::new(agentic_index())), rewriter)
}

// --- vector retriever -----------------------------------------------------

#[tokio::test]
async fn vector_retriever_orders_hits_and_rejects_other_modes() {
    let embedder = HashEmbedding::new(DIMENSION);
    let retriever = VectorRetriever::new(
        dense_store(&["alpha", "beta", "gamma"]).await,
        Arc::new(embedder),
    );

    let results = retriever
        .retrieve("alpha", 3, None, RetrieveMode::Vector, None)
        .await
        .unwrap();
    assert_eq!(results.len(), 3);
    // The record whose text *is* the query embeds to the same vector: cosine 1.
    assert_eq!(results[0].text, "alpha");
    assert_eq!(results[0].score, 1.0);
    assert_eq!(results[0].doc_id.as_deref(), Some("doc-0"));
    assert_eq!(results[0].chunk_id.as_deref(), Some("chunk-0"));
    assert!(results[0].score > results[1].score);
    assert!(results[1].score >= results[2].score);

    let top_one = retriever
        .retrieve("alpha", 1, None, RetrieveMode::Vector, None)
        .await
        .unwrap();
    assert_eq!(texts_of(&top_one), vec!["alpha"]);

    for mode in [RetrieveMode::Sparse, RetrieveMode::Hybrid] {
        let error = retriever
            .retrieve("alpha", 3, None, mode, None)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("only supports 'vector' mode"));
    }
}

// --- hybrid retriever -----------------------------------------------------

/// Dense channel: `alpha beta` (the query vector) and `alpha gamma` (unrelated).
/// Sparse channel: `alpha beta` and `alpha alpha delta` (higher BM25 thanks to tf).
async fn hybrid_retriever() -> HybridRetriever {
    let embedder = Arc::new(HashEmbedding::new(DIMENSION));
    let store = InMemoryVectorStore::with_collection("chunks");
    let alpha_vector = embedder.embed_query("alpha").await.unwrap();
    let other_vector = embedder.embed_query("gamma delta epsilon").await.unwrap();
    store
        .upsert(vec![
            VectorRecord {
                id: "dense-1".to_owned(),
                text: "alpha beta".to_owned(),
                vector: alpha_vector,
                metadata: Metadata::new(),
                doc_id: Some("doc-shared".to_owned()),
                chunk_id: Some("dense-1".to_owned()),
            },
            VectorRecord {
                id: "dense-2".to_owned(),
                text: "alpha gamma".to_owned(),
                vector: other_vector,
                metadata: Metadata::new(),
                doc_id: Some("doc-dense".to_owned()),
                chunk_id: Some("dense-2".to_owned()),
            },
        ])
        .await
        .unwrap();
    let index = bm25_index(&[
        ("sparse-1", "alpha beta"),
        ("sparse-2", "alpha alpha delta"),
    ]);
    HybridRetriever::from_parts(Arc::new(store), embedder, index)
}

#[tokio::test]
async fn hybrid_retriever_fuses_with_rrf_and_over_fetches() {
    let retriever = hybrid_retriever().await;
    let results = retriever
        .retrieve("alpha", 3, None, RetrieveMode::Hybrid, None)
        .await
        .unwrap();
    // Dense: [alpha beta, alpha gamma]; sparse: [alpha alpha delta, alpha beta].
    // RRF with k = 60 gives `alpha beta` 1/61 + 1/62, `alpha alpha delta` 1/61,
    // and `alpha gamma` 1/62.
    assert_eq!(
        texts_of(&results),
        vec!["alpha beta", "alpha alpha delta", "alpha gamma"]
    );
    let expected = 1.0 / 61.0 + 1.0 / 62.0;
    assert!((results[0].score - expected).abs() < 1e-6);
    assert!((results[1].score - 1.0 / 61.0).abs() < 1e-6);
    assert!((results[2].score - 1.0 / 62.0).abs() < 1e-6);
    assert_eq!(results[0].chunk_id.as_deref(), Some("dense-1"));
    assert_eq!(results[0].doc_id.as_deref(), Some("doc-shared"));
}

#[tokio::test]
async fn hybrid_retriever_applies_the_threshold_to_the_dense_channel_only() {
    let retriever = hybrid_retriever().await;
    let dense_only = retriever
        .retrieve("alpha", 3, None, RetrieveMode::Hybrid, None)
        .await
        .unwrap();
    let dense_other = dense_only
        .iter()
        .find(|result| result.text == "alpha gamma")
        .unwrap();
    assert!(dense_other.score < 1.0);

    let thresholded = retriever
        .retrieve("alpha", 3, Some(0.9), RetrieveMode::Hybrid, None)
        .await
        .unwrap();
    // `alpha gamma` drops out of the dense channel; BM25 hits are never filtered.
    assert_eq!(
        texts_of(&thresholded),
        vec!["alpha beta", "alpha alpha delta"]
    );
}

#[tokio::test]
async fn hybrid_retriever_falls_back_to_sparse_when_dense_is_empty() {
    let empty_store: Arc<dyn VectorStore> = Arc::new(InMemoryVectorStore::new());
    let index = bm25_index(&[
        ("sparse-1", "alpha beta"),
        ("sparse-2", "alpha alpha delta"),
    ]);
    let retriever =
        HybridRetriever::from_parts(empty_store, Arc::new(HashEmbedding::new(DIMENSION)), index);
    let results = retriever
        .retrieve("alpha", 2, None, RetrieveMode::Vector, None)
        .await
        .unwrap();
    assert_eq!(texts_of(&results), vec!["alpha alpha delta", "alpha beta"]);
    assert_eq!(retriever.candidate_factor(), 2);
    assert_eq!(retriever.rrf_k(), 60);
}

// --- graph retriever ------------------------------------------------------

/// `chunk-a` answers the query, `chunk-b` shares no query term and is only
/// reachable through the triple `(Rust, has, Borrow Checker)`.
fn graph_corpus() -> Arc<Bm25Index> {
    bm25_index(&[
        ("chunk-a", "the ownership model prevents data races"),
        ("chunk-b", "the borrow checker guide"),
    ])
}

#[tokio::test]
async fn graph_retriever_finds_a_chunk_that_shares_no_query_terms() {
    let plain = SparseRetriever::new(graph_corpus());
    let direct = plain
        .retrieve("Rust ownership", 2, None, RetrieveMode::Sparse, None)
        .await
        .unwrap();
    assert_eq!(
        texts_of(&direct),
        vec!["the ownership model prevents data races"]
    );

    let store: Arc<dyn GraphStore> = Arc::new(
        InMemoryGraphStore::with_triples(vec![Triple::new("Rust", "has", "Borrow Checker")])
            .unwrap(),
    );
    let retriever = GraphRetriever::new(Arc::new(plain), store);
    let results = retriever
        .retrieve("Rust ownership", 2, None, RetrieveMode::Sparse, None)
        .await
        .unwrap();
    // Both lists rank their chunks first, so the stable RRF tie keeps the
    // expanded list (the first argument of `rrf_fusion`) ahead.
    assert_eq!(
        texts_of(&results),
        vec![
            "the borrow checker guide",
            "the ownership model prevents data races"
        ]
    );
    assert_eq!(results[0].chunk_id.as_deref(), Some("chunk-b"));
    assert_eq!(retriever.max_hops(), 2);
    assert_eq!(retriever.expansion_limit(), 5);
    assert_eq!(retriever.triple_mode(), RetrieveMode::Sparse);
    assert_eq!(retriever.graph_store().len().await.unwrap(), 1);
}

#[tokio::test]
async fn graph_retriever_degrades_to_the_chunk_results() {
    let store: Arc<dyn GraphStore> = Arc::new(InMemoryGraphStore::new());
    let retriever = GraphRetriever::new(Arc::new(SparseRetriever::new(graph_corpus())), store);
    let results = retriever
        .retrieve("Rust ownership", 2, None, RetrieveMode::Sparse, None)
        .await
        .unwrap();
    assert_eq!(
        texts_of(&results),
        vec!["the ownership model prevents data races"]
    );
    assert!(results[0].score > 0.0);
}

// --- agentic retriever ----------------------------------------------------

#[tokio::test]
async fn agentic_retriever_improves_recall_over_the_plain_channel() {
    let plain = SparseRetriever::new(agentic_index());
    let direct = plain
        .retrieve("ownership safety", 3, None, RetrieveMode::Sparse, None)
        .await
        .unwrap();
    assert_eq!(
        texts_of(&direct),
        vec!["Acme uses Rust. Ownership rules apply."]
    );

    let rewriter = ScriptedRewriter::new(&["borrow checker"]);
    let retriever = AgenticRetriever::new(
        Arc::new(SparseRetriever::new(agentic_index())),
        rewriter.clone(),
    );
    let (results, trace) = retriever
        .retrieve_with_trace("ownership safety", 3, None, RetrieveMode::Sparse, None)
        .await
        .unwrap();
    // Iteration 1 finds `chunk-a`, iteration 2 finds `chunk-b`; both rank first in
    // their own list, so the stable RRF tie keeps iteration 1 first.
    assert_eq!(
        texts_of(&results),
        vec![
            "Acme uses Rust. Ownership rules apply.",
            "The borrow checker notes"
        ]
    );
    assert_eq!(trace.iterations, 2);
    assert_eq!(
        trace.queries,
        vec!["ownership safety".to_owned(), "borrow checker".to_owned()]
    );
    assert_eq!(trace.triples.len(), 1);
    assert_eq!(trace.triples[0].subject, "Acme");
    assert_eq!(trace.triples[0].predicate, "uses");
    assert_eq!(trace.triples[0].object, "Rust");
    assert!(trace.triples[0].metadata.contains_key("doc_id"));
    assert_eq!(trace.stopped_reason, AgenticStopReason::MaxIterations);
    let requests = rewriter.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].query, "ownership safety");
    assert_eq!(
        requests[0].history,
        vec!["ownership safety".to_owned(), "(acme uses rust)".to_owned()]
    );
    assert!(retriever.model().is_none());
    assert_eq!(retriever.config().max_iter, 2);
}

#[tokio::test]
async fn agentic_retriever_stops_when_the_passages_are_sufficient() {
    let rewriter = ScriptedRewriter::new(&["never used"]);
    let retriever = agentic_retriever(rewriter.clone());
    let (results, trace) = retriever
        .retrieve_with_trace("ownership rules", 3, None, RetrieveMode::Sparse, None)
        .await
        .unwrap();
    // Both keywords appear in the first iteration, so the loop never rewrites.
    assert_eq!(trace.stopped_reason, AgenticStopReason::Sufficient);
    assert_eq!(trace.iterations, 1);
    assert_eq!(trace.queries, vec!["ownership rules".to_owned()]);
    assert_eq!(results.len(), 1);
    assert_eq!(rewriter.calls(), 0);
}

#[tokio::test]
async fn agentic_retriever_stops_when_the_rewrite_repeats() {
    let rewriter = ScriptedRewriter::new(&["ownership safety"]);
    let retriever = agentic_retriever(rewriter.clone());
    let (results, trace) = retriever
        .retrieve_with_trace("ownership safety", 3, None, RetrieveMode::Sparse, None)
        .await
        .unwrap();
    assert_eq!(trace.stopped_reason, AgenticStopReason::NoNewResults);
    assert_eq!(trace.iterations, 1);
    assert_eq!(trace.queries, vec!["ownership safety".to_owned()]);
    assert_eq!(results.len(), 1);
    assert_eq!(rewriter.calls(), 1);
}

#[tokio::test]
async fn agentic_retriever_honours_max_iter_and_a_single_pass() {
    let single = agentic_retriever(ScriptedRewriter::new(&["unused"]))
        .with_config(AgenticConfig::new().with_max_iter(1))
        .unwrap();
    let (results, trace) = single
        .retrieve_with_trace("ownership safety", 3, None, RetrieveMode::Sparse, None)
        .await
        .unwrap();
    assert_eq!(trace.iterations, 1);
    assert_eq!(trace.stopped_reason, AgenticStopReason::MaxIterations);
    assert_eq!(trace.queries, vec!["ownership safety".to_owned()]);
    assert_eq!(results.len(), 1);

    let rewriter = ScriptedRewriter::new(&["alpha one", "beta two", "gamma three"]);
    let bounded = AgenticRetriever::new(
        Arc::new(SparseRetriever::new(bm25_index(&[]))),
        rewriter.clone(),
    )
    .with_config(AgenticConfig::new().with_max_iter(3))
    .unwrap();
    let (results, trace) = bounded
        .retrieve_with_trace("ownership safety", 3, None, RetrieveMode::Sparse, None)
        .await
        .unwrap();
    assert_eq!(trace.iterations, 3);
    assert_eq!(trace.queries.len(), 3);
    assert_eq!(trace.stopped_reason, AgenticStopReason::MaxIterations);
    assert_eq!(rewriter.calls(), 2);
    assert!(results.is_empty());

    let error = AgenticRetriever::new(Arc::new(SparseRetriever::new(agentic_index())), rewriter)
        .with_config(AgenticConfig::new().with_max_iter(0))
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("max_iter"));

    let retriever = agentic_retriever(ScriptedRewriter::new(&["unused"]));
    let error = retriever
        .retrieve_with_trace("ownership safety", 0, None, RetrieveMode::Sparse, None)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("top_k"));
}

#[tokio::test]
async fn agentic_retriever_uses_a_supplied_model_for_reads_and_rewrites() {
    let model = Arc::new(StaticCompletionModel::new(vec![
        "[[\"Acme\", \"uses\", \"Rust\"]]".to_owned(),
        "{\"sufficient\": false, \"next_question\": \"borrow checker\"}".to_owned(),
        "[[\"Rust\", \"has\", \"Borrow Checker\"]]".to_owned(),
    ]));
    let rewriter = ScriptedRewriter::new(&["unused"]);
    let retriever = agentic_retriever(rewriter.clone()).with_model(model.clone());
    let (results, trace) = retriever
        .retrieve_with_trace("ownership safety", 3, None, RetrieveMode::Sparse, None)
        .await
        .unwrap();
    assert_eq!(trace.iterations, 2);
    assert_eq!(
        trace.queries,
        vec!["ownership safety".to_owned(), "borrow checker".to_owned()]
    );
    assert_eq!(
        trace.triples,
        vec![
            Triple::new("Acme", "uses", "Rust"),
            Triple::new("Rust", "has", "Borrow Checker")
        ]
    );
    assert_eq!(trace.stopped_reason, AgenticStopReason::MaxIterations);
    assert_eq!(results.len(), 2);
    // The scripted rewriter is only a fallback and was never needed.
    assert_eq!(rewriter.calls(), 0);
    assert_eq!(model.call_count(), 3);
    let prompts = model.prompts();
    assert!(prompts[0].contains("Ownership rules apply"));
    assert!(prompts[1].contains("Knowledge triples"));
    assert!(prompts[1].contains("(acme uses rust)"));
}

#[tokio::test]
async fn agentic_retriever_falls_back_when_the_model_reply_is_malformed() {
    let model = Arc::new(StaticCompletionModel::single("I do not know."));
    let rewriter = ScriptedRewriter::new(&["ownership safety"]);
    let retriever = agentic_retriever(rewriter.clone()).with_model(model.clone());
    let (results, trace) = retriever
        .retrieve_with_trace("ownership safety", 3, None, RetrieveMode::Sparse, None)
        .await
        .unwrap();
    // Both model steps failed, so the rule extractor supplied the triples and the
    // rewriter repeated the query.
    assert_eq!(trace.triples.len(), 1);
    assert_eq!(trace.triples[0].normalized_key(), "acme|uses|rust");
    assert_eq!(trace.stopped_reason, AgenticStopReason::ModelFailure);
    assert_eq!(trace.iterations, 1);
    assert_eq!(rewriter.calls(), 1);
    assert_eq!(model.call_count(), 2);
    assert_eq!(results.len(), 1);
}
