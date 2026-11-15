#![allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
//! Cross-cutting strictness tests: three former silent-loss paths, now explicit.
//!
//! Site 1 — `InMemoryVectorStore::search` rejects a query whose width disagrees
//! with the stored records instead of dropping them and returning a short list.
//! Site 2 — `GraphRetriever` reports every failed graph step and flags the hits
//! it degraded to instead of only logging the failure.
//! Site 3 — multi-KB retrieval has reporting and strict variants, so per-KB
//! failures are observable (or fatal) instead of vanishing into an empty list.
//!
//! Everything runs offline and deterministically: in-memory stores, the
//! deterministic `HashEmbedding`, a BM25 index, and scripted failures.
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_retrieval::graph::{
    GraphStatistics, GraphStore, InMemoryGraphStore, Triple, TripleQuery,
};
use openjiuwen_retrieval::knowledge_base::{
    retrieve_multi_kb_reporting, retrieve_multi_kb_strict, retrieve_multi_kb_with_source_reporting,
    retrieve_multi_kb_with_source_strict, MultiKbFailure,
};
use openjiuwen_retrieval::retriever::graph::{
    GraphExpansionReport, GraphStage, GRAPH_DEGRADED_FIELD,
};
use openjiuwen_retrieval::retriever::{GraphRetriever, RetrieveMode, Retriever, SparseRetriever};
use openjiuwen_retrieval::sparse::{Bm25Config, Bm25Index};
use openjiuwen_retrieval::{
    Document, Embedding, HashEmbedding, InMemoryVectorStore, KnowledgeBaseConfig, RetrievalResult,
    SimpleKnowledgeBase, VectorQuery, VectorRecord, VectorStore,
};

/// The result texts, in order (shared by the graph and multi-KB sections).
fn texts(results: &[RetrievalResult]) -> Vec<String> {
    results.iter().map(|result| result.text.clone()).collect()
}

// ---------------------------------------------------------------------------
// Site 1 — vector store dimension validation
//
// `search` used to score only the records whose width matched the query and
// silently discard the rest, so a caller could not tell "nothing is relevant"
// from "your query vector is the wrong shape". It is now a typed error that
// reaches the knowledge-base caller as well.
// ---------------------------------------------------------------------------

/// A record with `vector` and a text derived from its id.
fn record(id: &str, vector: Vec<f32>) -> VectorRecord {
    VectorRecord {
        id: id.to_owned(),
        text: format!("text of {id}"),
        vector,
        ..VectorRecord::default()
    }
}

#[tokio::test]
async fn dimension_mismatch_is_an_error_not_a_short_result_list() {
    let store = InMemoryVectorStore::new();
    store
        .upsert(vec![
            record("four-d", vec![1.0, 0.0, 0.0, 0.0]),
            record("also-four-d", vec![0.0, 1.0, 0.0, 0.0]),
        ])
        .await
        .unwrap();

    // Before the fix this returned a shorter (typically empty) hit list, because
    // both 4-dimensional records were skipped for not matching a 2-dimensional
    // query. A caller who mixes embedders now gets a diagnosable error instead.
    let error = store
        .search(VectorQuery::new(vec![1.0, 0.0]))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains("vector dimension mismatch"), "{message}");
    assert!(message.contains("4-dimensional"), "{message}");
    assert!(message.contains("2 dimensions"), "{message}");
}

#[tokio::test]
async fn an_internally_inconsistent_store_reports_the_conflict() {
    // A store whose records disagree on their width has no single query width
    // that scores everything; the old code filtered per record and hid the
    // conflict, the new code names the two offending records.
    let mixed = InMemoryVectorStore::new();
    mixed
        .upsert(vec![
            record("two-d", vec![1.0, 0.0]),
            record("three-d", vec![1.0, 0.0, 0.0]),
        ])
        .await
        .unwrap();
    let error = mixed
        .search(VectorQuery::new(vec![1.0, 0.0]))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains("internally inconsistent"), "{message}");
    assert!(
        message.contains("`two-d`") && message.contains("`three-d`"),
        "{message}"
    );

    // An empty store knows no width, so any query width is a legitimate miss,
    // not an error — this is the `Ok` side of the same validation.
    let empty = InMemoryVectorStore::new();
    let hits = empty
        .search(VectorQuery::new(vec![1.0, 0.0, 0.0]))
        .await
        .unwrap();
    assert_eq!(hits.len(), 0);
}

#[tokio::test]
async fn a_query_width_mismatch_surfaces_through_the_knowledge_base() {
    // Seed the store with 64-dimensional records, then query it through a
    // 32-dimensional embedder: the store error must propagate through the
    // pipeline with the retrieval-stage context instead of yielding a partial
    // (here: empty) result set.
    let store = Arc::new(InMemoryVectorStore::new());
    store
        .upsert(vec![VectorRecord {
            id: "chunk-1".to_owned(),
            text: "seeded at 64 dimensions".to_owned(),
            vector: vec![0.5; 64],
            ..VectorRecord::default()
        }])
        .await
        .unwrap();

    let vector_store: Arc<dyn VectorStore> = store;
    let kb = SimpleKnowledgeBase::from_config(
        KnowledgeBaseConfig::new("kb-dim"),
        vector_store,
        Arc::new(HashEmbedding::new(32)),
    )
    .unwrap();

    let error = kb.retrieve("anything").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(
        message.contains("retrieving from the knowledge base"),
        "{message}"
    );
    assert!(message.contains("vector dimension mismatch"), "{message}");
}

// ---------------------------------------------------------------------------
// Site 2 — graph expansion report and metadata flag
//
// Degrading to the plain chunk hits is intended (Python does the same), but it
// used to be indistinguishable from a clean run: the failure was only logged.
// `retrieve_with_report` now returns what happened and marks the degraded hits.
// ---------------------------------------------------------------------------

/// Message every call of [`OfflineGraphStore`] fails with.
const GRAPH_OFFLINE: &str = "graph offline";

/// The query shared by the graph tests; its capitalised token `Rust` is the
/// entity candidate the retriever looks up in the store.
const QUERY: &str = "Rust ownership";

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

/// Chunk corpus: an ownership chunk and a borrow-checker chunk that only the
/// graph expansion can surface for [`QUERY`].
fn corpus() -> Arc<Bm25Index> {
    bm25_index(&[
        ("chunk-a", "the ownership model prevents data races"),
        ("chunk-b", "the borrow checker guide"),
    ])
}

/// A [`GraphStore`] double whose `neighbors` lookup fails, standing in for an
/// unreachable graph backend. Every other method is a harmless no-op so only
/// the expansion path under test can fail.
struct OfflineGraphStore;

#[async_trait]
impl GraphStore for OfflineGraphStore {
    async fn add_triples(&self, _triples: Vec<Triple>) -> Result<usize> {
        Ok(0)
    }

    async fn triples_for_entity(&self, _entity: &str) -> Result<Vec<Triple>> {
        Ok(Vec::new())
    }

    async fn neighbors(&self, _entity: &str, _hops: usize) -> Result<Vec<Triple>> {
        Err(Error::retrieval(GRAPH_OFFLINE))
    }

    async fn search(&self, _query: &TripleQuery) -> Result<Vec<Triple>> {
        Ok(Vec::new())
    }

    async fn all_triples(&self) -> Result<Vec<Triple>> {
        Ok(Vec::new())
    }

    async fn len(&self) -> Result<usize> {
        Ok(0)
    }

    async fn is_empty(&self) -> Result<bool> {
        Ok(true)
    }

    async fn clear(&self) -> Result<()> {
        Ok(())
    }

    async fn statistics(&self) -> Result<GraphStatistics> {
        Ok(GraphStatistics::default())
    }
}

/// A clean run reports no degradation and leaves every hit unmarked.
fn assert_clean(report: &GraphExpansionReport, hits: &[RetrievalResult]) {
    assert_eq!(
        report.degradations.len(),
        0,
        "a clean run records no failed step"
    );
    assert!(!report.is_degraded());
    assert!(
        hits.iter()
            .all(|hit| !hit.metadata.contains_key(GRAPH_DEGRADED_FIELD)),
        "a clean expansion must not mark its hits"
    );
}

#[tokio::test]
async fn a_broken_graph_is_reported_and_flagged_while_the_chunk_hits_survive() {
    let index = corpus();
    let plain = SparseRetriever::new(index.clone())
        .retrieve(QUERY, 2, None, RetrieveMode::Sparse, None)
        .await
        .unwrap();

    let retriever = GraphRetriever::new(
        Arc::new(SparseRetriever::new(index)),
        Arc::new(OfflineGraphStore),
    );
    let (hits, report) = retriever
        .retrieve_with_report(QUERY, 2, None, RetrieveMode::Sparse, None)
        .await
        .unwrap();

    // The chunk hits still arrive, unchanged: a broken graph never costs a hit.
    assert_ne!(hits.len(), 0);
    assert_eq!(texts(&hits), texts(&plain));
    // ...but they are marked, so a caller can tell this run from a clean one.
    assert!(hits
        .iter()
        .all(|hit| hit.metadata.get(GRAPH_DEGRADED_FIELD) == Some(&json!(true))));

    assert!(report.is_degraded());
    assert_eq!(report.degradations.len(), 1);
    let degradation = &report.degradations[0];
    assert_eq!(degradation.stage, GraphStage::NeighborExpansion);
    assert!(
        degradation.error.contains(GRAPH_OFFLINE),
        "{}",
        degradation.error
    );
    assert!(degradation.subject.is_some());
    assert_ne!(report.entities.len(), 0);
    // No triples could be collected, so nothing was expanded.
    assert_eq!(report.triple_count, 0);
    assert_eq!(report.expanded_queries, 0);
}

#[tokio::test]
async fn an_empty_graph_is_not_a_degradation() {
    let index = corpus();
    let plain = SparseRetriever::new(index.clone())
        .retrieve(QUERY, 2, None, RetrieveMode::Sparse, None)
        .await
        .unwrap();

    let retriever = GraphRetriever::new(
        Arc::new(SparseRetriever::new(index)),
        Arc::new(InMemoryGraphStore::new()),
    );
    let (hits, report) = retriever
        .retrieve_with_report(QUERY, 2, None, RetrieveMode::Sparse, None)
        .await
        .unwrap();

    // "The graph holds nothing for this query" is a genuine no-op, not a
    // failure: no degradation entry and no flag on the hits that came back.
    assert_clean(&report, &hits);
    assert_eq!(report.triple_count, 0);
    assert_eq!(report.expanded_queries, 0);
    assert_eq!(texts(&hits), texts(&plain));
}

#[tokio::test]
async fn a_clean_expansion_reports_what_it_did() {
    let graph =
        InMemoryGraphStore::with_triples(vec![Triple::new("Rust", "has", "Borrow Checker")])
            .unwrap();
    let retriever = GraphRetriever::new(Arc::new(SparseRetriever::new(corpus())), Arc::new(graph));
    let (hits, report) = retriever
        .retrieve_with_report(QUERY, 2, None, RetrieveMode::Sparse, None)
        .await
        .unwrap();

    // One triple was collected and one chunk query was issued for its text; the
    // report states exactly that instead of leaving the expansion invisible.
    assert_eq!(report.triple_count, 1);
    assert_eq!(report.expanded_queries, 1);
    assert_clean(&report, &hits);
    // Whether the borrow-checker chunk makes the final cut depends on the RRF
    // scores, so only the non-emptiness and the absence of the flag are asserted.
    assert_ne!(hits.len(), 0);
}

// ---------------------------------------------------------------------------
// Site 3 — multi-KB reporting and strict variants
//
// The compatible multi-KB functions log a failing knowledge base and drop its
// results, so "every KB failed" looked like "no KB had a match". The reporting
// variants return the failures, the strict variants turn an all-failed fan-out
// into an error.
// ---------------------------------------------------------------------------

/// Embedder that always fails, so every retrieval of its knowledge base errors.
struct FailingEmbedding;

#[async_trait]
impl Embedding for FailingEmbedding {
    fn model_name(&self) -> &'static str {
        "failing-embedding"
    }

    async fn embed_query(&self, _text: &str) -> Result<Vec<f32>> {
        Err(Error::retrieval("boom"))
    }

    async fn embed_documents(
        &self,
        _texts: &[String],
        _batch_size: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        Err(Error::retrieval("boom"))
    }
}

/// A knowledge base whose embedder always fails, so every retrieval errors.
fn failing_kb(kb_id: &str) -> Arc<SimpleKnowledgeBase> {
    let vector_store: Arc<dyn VectorStore> = Arc::new(InMemoryVectorStore::new());
    Arc::new(
        SimpleKnowledgeBase::from_config(
            KnowledgeBaseConfig::new(kb_id),
            vector_store,
            Arc::new(FailingEmbedding),
        )
        .unwrap(),
    )
}

/// A working knowledge base holding one document.
async fn healthy_kb() -> Arc<SimpleKnowledgeBase> {
    let vector_store: Arc<dyn VectorStore> = Arc::new(InMemoryVectorStore::new());
    let kb = SimpleKnowledgeBase::from_config(
        KnowledgeBaseConfig::new("kb-healthy"),
        vector_store,
        Arc::new(HashEmbedding::new(64)),
    )
    .unwrap();
    kb.add_documents(vec![Document::new("shared text")])
        .await
        .unwrap();
    Arc::new(kb)
}

/// The failed knowledge base ids, in report order.
fn failure_ids(failures: &[MultiKbFailure]) -> Vec<&str> {
    failures
        .iter()
        .map(|failure| failure.kb_id.as_str())
        .collect()
}

#[tokio::test]
async fn partial_failures_are_listed_and_the_surviving_results_returned() {
    let kbs = vec![
        healthy_kb().await,
        failing_kb("kb-fail-1"),
        failing_kb("kb-fail-2"),
    ];

    // The surviving hits are returned and the failures are named, in input
    // order — previously both failures were only logged.
    let report = retrieve_multi_kb_reporting(&kbs, "shared", None).await;
    assert_ne!(report.results.len(), 0);
    assert!(report
        .results
        .iter()
        .any(|result| result.text == "shared text"));
    assert_eq!(
        failure_ids(&report.failures),
        vec!["kb-fail-1", "kb-fail-2"]
    );
    assert!(report
        .failures
        .iter()
        .all(|failure| failure.error.contains("boom")));

    // A partial failure is not fatal: the strict variant returns the same hits.
    let strict = retrieve_multi_kb_strict(&kbs, "shared", None)
        .await
        .unwrap();
    assert_eq!(texts(&strict), texts(&report.results));

    let source_report = retrieve_multi_kb_with_source_reporting(&kbs, "shared", None).await;
    assert_eq!(failure_ids(&source_report.failures).len(), 2);
    assert_ne!(source_report.results.len(), 0);
    let source_strict = retrieve_multi_kb_with_source_strict(&kbs, "shared", None)
        .await
        .unwrap();
    assert_eq!(source_strict.len(), source_report.results.len());
}

/// Asserts the all-failed error names both knowledge bases.
fn assert_all_failed(error: &Error) {
    assert_eq!(error.kind(), ErrorKind::Retrieval);
    let message = error.to_string();
    assert!(message.contains("kb-fail-1"), "{message}");
    assert!(message.contains("kb-fail-2"), "{message}");
}

#[tokio::test]
async fn every_failure_is_an_error_not_an_empty_result() {
    let kbs = vec![failing_kb("kb-fail-1"), failing_kb("kb-fail-2")];

    // "Every KB failed" must not look like "no KB had a match".
    assert_all_failed(
        &retrieve_multi_kb_strict(&kbs, "shared", None)
            .await
            .unwrap_err(),
    );
    assert_all_failed(
        &retrieve_multi_kb_with_source_strict(&kbs, "shared", None)
            .await
            .unwrap_err(),
    );

    // The reporting variants still hand back a value, so a caller that would
    // rather inspect than throw can see both failures.
    let report = retrieve_multi_kb_reporting(&kbs, "shared", None).await;
    assert_eq!(report.results.len(), 0);
    assert_eq!(
        failure_ids(&report.failures),
        vec!["kb-fail-1", "kb-fail-2"]
    );

    let source_report = retrieve_multi_kb_with_source_reporting(&kbs, "shared", None).await;
    assert_eq!(source_report.results.len(), 0);
    assert_eq!(source_report.failures.len(), 2);
}

#[tokio::test]
async fn an_empty_kb_list_is_not_an_error() {
    let empty: Vec<Arc<SimpleKnowledgeBase>> = Vec::new();

    // No knowledge bases means no failures, not "everything failed".
    assert_eq!(
        retrieve_multi_kb_strict(&empty, "q", None)
            .await
            .unwrap()
            .len(),
        0
    );
    assert_eq!(
        retrieve_multi_kb_with_source_strict(&empty, "q", None)
            .await
            .unwrap()
            .len(),
        0
    );

    let report = retrieve_multi_kb_reporting(&empty, "q", None).await;
    assert_eq!(report.results.len(), 0);
    assert_eq!(report.failures.len(), 0);

    let source_report = retrieve_multi_kb_with_source_reporting(&empty, "q", None).await;
    assert_eq!(source_report.results.len(), 0);
    assert_eq!(source_report.failures.len(), 0);
}
