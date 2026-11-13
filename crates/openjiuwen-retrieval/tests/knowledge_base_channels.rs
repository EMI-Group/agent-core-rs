#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Integration tests for the knowledge base retrieval channels: BM25,
//! hybrid fusion, the agentic loop, graph expansion, statistics, and the
//! filter/threshold/reranker plumbing through each channel.
//!
//! Everything runs offline. The dense channel either uses [`HashEmbedding`] or
//! the deterministic [`TableEmbedding`] below, whose hand-assigned 2-dimensional
//! vectors make the dense and lexical rankings independently checkable.
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use openjiuwen_common::error::{ErrorKind, Result};
use openjiuwen_retrieval::graph::{GraphStore, InMemoryGraphStore};
use openjiuwen_retrieval::knowledge_base::{DEFAULT_AGENTIC_ITERATIONS, DEFAULT_GRAPH_HOPS};
use openjiuwen_retrieval::rewrite::{QueryRewriter, RewriteOutput, RewriteRequest};
use openjiuwen_retrieval::{
    Document, Embedding, HashEmbedding, InMemoryVectorStore, KnowledgeBaseConfig,
    KnowledgeBaseStatistics, Metadata, MetadataFilter, Reranker, RetrievalConfig, RetrievalResult,
    SearchResult, SimpleKnowledgeBase, VectorStore, INDEX_TYPE_BM25, INDEX_TYPE_HYBRID,
    INDEX_TYPE_VECTOR,
};

/// Dimension used by [`HashEmbedding`] fixtures.
const EMBEDDING_DIMENSION: usize = 64;

/// Builds a knowledge base from `config` with `embedder` and a fresh store.
fn knowledge_base(
    config: KnowledgeBaseConfig,
    embedder: Arc<dyn Embedding>,
) -> SimpleKnowledgeBase {
    let store: Arc<dyn VectorStore> = Arc::new(InMemoryVectorStore::new());
    SimpleKnowledgeBase::from_config(config, store, embedder).unwrap()
}

/// A knowledge base using the default [`HashEmbedding`] encoder.
fn hash_kb(config: KnowledgeBaseConfig) -> SimpleKnowledgeBase {
    knowledge_base(config, Arc::new(HashEmbedding::new(EMBEDDING_DIMENSION)))
}

/// The index-type override used by most fixtures.
fn config(index_type: &str) -> KnowledgeBaseConfig {
    let mut config = KnowledgeBaseConfig::new("kb");
    index_type.clone_into(&mut config.index_type);
    config
}

/// The texts of `results`, in order.
fn texts(results: &[RetrievalResult]) -> Vec<String> {
    results.iter().map(|result| result.text.clone()).collect()
}

/// A deterministic embedder: exact texts map to hand-assigned 2-d vectors.
///
/// Unknown texts map to `[0, 0]`, which scores below every assigned vector.
#[derive(Debug, Clone)]
struct TableEmbedding {
    /// `(exact text, vector)` pairs consulted in order.
    table: Vec<(&'static str, [f32; 2])>,
}

impl TableEmbedding {
    /// Builds the embedder from its table.
    fn new(table: Vec<(&'static str, [f32; 2])>) -> Self {
        Self { table }
    }

    /// The vector assigned to `text`.
    fn vector(&self, text: &str) -> Vec<f32> {
        self.table
            .iter()
            .find(|(key, _)| *key == text)
            .map_or_else(|| vec![0.0, 0.0], |(_, vector)| vector.to_vec())
    }
}

#[async_trait]
impl Embedding for TableEmbedding {
    fn model_name(&self) -> &'static str {
        "table-embedding"
    }

    fn dimension_hint(&self) -> Option<usize> {
        Some(2)
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.vector(text))
    }

    async fn embed_documents(
        &self,
        texts: &[String],
        _batch_size: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|text| self.vector(text)).collect())
    }
}

/// A rewriter that always proposes `next` and records how often it was called.
#[derive(Debug)]
struct ScriptedRewriter {
    /// Query proposed by every rewrite.
    next: String,
    /// Keywords reported to the agentic loop (drive its sufficiency check).
    keywords: Vec<String>,
    /// Number of `rewrite` calls.
    rewrites: AtomicUsize,
    /// Number of `keywords` calls.
    keyword_calls: AtomicUsize,
}

impl ScriptedRewriter {
    /// Builds the rewriter with a fixed next query and keyword list.
    fn new(next: &str, keywords: &[&str]) -> Self {
        Self {
            next: next.to_owned(),
            keywords: keywords
                .iter()
                .map(|keyword| (*keyword).to_owned())
                .collect(),
            rewrites: AtomicUsize::new(0),
            keyword_calls: AtomicUsize::new(0),
        }
    }

    /// Number of rewrites so far.
    fn rewrites(&self) -> usize {
        self.rewrites.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl QueryRewriter for ScriptedRewriter {
    async fn rewrite(&self, _request: &RewriteRequest) -> Result<RewriteOutput> {
        self.rewrites.fetch_add(1, Ordering::SeqCst);
        Ok(RewriteOutput::from_query(self.next.clone()))
    }

    async fn keywords(&self, _query: &str) -> Result<Vec<String>> {
        self.keyword_calls.fetch_add(1, Ordering::SeqCst);
        Ok(self.keywords.clone())
    }

    async fn expand(&self, query: &str) -> Result<Vec<String>> {
        Ok(vec![query.to_owned()])
    }
}

/// A reranker that reverses the hits, recording the `top_n` it received.
#[derive(Default)]
struct ReverseReranker {
    /// Last `top_n` supplied by the knowledge base.
    seen_top_n: Mutex<Option<usize>>,
}

#[async_trait]
impl Reranker for ReverseReranker {
    async fn rerank(
        &self,
        _query: &str,
        mut documents: Vec<SearchResult>,
        top_n: Option<usize>,
    ) -> Result<Vec<SearchResult>> {
        *self.seen_top_n.lock().unwrap() = top_n;
        documents.reverse();
        Ok(documents)
    }
}

#[tokio::test]
async fn bm25_index_type_retrieves_lexically_and_tracks_the_index() {
    let kb = hash_kb(config(INDEX_TYPE_BM25));
    assert!(kb.has_sparse_index());
    assert_eq!(kb.sparse_index_len(), 0);
    let ids = kb
        .add_texts(vec![
            "quarterly report summary".to_owned(),
            "revenue totals report".to_owned(),
            "quokka sightings".to_owned(),
        ])
        .await
        .unwrap();
    assert_eq!(ids.len(), 3);
    assert_eq!(kb.sparse_index_len(), 3);

    let results = kb
        .retrieve_with_config("quokka", &RetrievalConfig::default().with_top_k(1))
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].text, "quokka sightings");
    assert!(results[0].chunk_id.is_some());
    assert_eq!(results[0].doc_id.as_deref(), Some(ids[2].as_str()));

    // Deletes drop the chunk from the index, updates re-index the new content.
    assert_eq!(kb.delete_documents(&[ids[0].clone()]).await.unwrap(), 1);
    assert_eq!(kb.sparse_index_len(), 2);
    assert_eq!(
        kb.retrieve_with_config("quarterly", &RetrievalConfig::default())
            .await
            .unwrap()
            .len(),
        0
    );
    let updated = kb
        .update_documents(vec![Document::with_id(ids[2].clone(), "quokka colony")])
        .await
        .unwrap();
    assert_eq!(updated, vec![ids[2].clone()]);
    assert_eq!(kb.sparse_index_len(), 2);
    let colony = kb
        .retrieve_with_config("colony", &RetrievalConfig::default().with_top_k(1))
        .await
        .unwrap();
    assert_eq!(colony.len(), 1);
    assert_eq!(colony[0].text, "quokka colony");
}

#[tokio::test]
async fn hybrid_with_sparse_index_fuses_dense_and_bm25() {
    let embedder = || {
        Arc::new(TableEmbedding::new(vec![
            ("quokka notes", [1.0, 0.0]),
            ("quartz mines", [1.0, 0.0]),
            ("orphan ledger", [0.5, 0.5]),
            ("quokka", [0.0, 1.0]),
        ]))
    };
    let corpus = vec![
        "quartz mines".to_owned(),
        "orphan ledger".to_owned(),
        "quokka".to_owned(),
    ];

    let dense = knowledge_base(config(INDEX_TYPE_HYBRID), embedder());
    assert!(!dense.has_sparse_index());
    dense.add_texts(corpus.clone()).await.unwrap();

    let hybrid = knowledge_base(config(INDEX_TYPE_HYBRID), embedder()).with_sparse_index(true);
    assert!(hybrid.has_sparse_index());
    hybrid.add_texts(corpus).await.unwrap();
    assert_eq!(hybrid.sparse_index_len(), 3);

    let top_two = RetrievalConfig::default().with_top_k(2);
    let dense_results = dense
        .retrieve_with_config("quokka notes", &top_two)
        .await
        .unwrap();
    assert_eq!(dense_results.len(), 2);
    assert_eq!(texts(&dense_results), vec!["quartz mines", "orphan ledger"]);

    // The lexical channel lifts the only chunk containing "quokka" into the
    // fused top two, where the dense-only channel never ranks it.
    let hybrid_results = hybrid
        .retrieve_with_config("quokka notes", &top_two)
        .await
        .unwrap();
    assert_eq!(hybrid_results.len(), 2);
    assert!(texts(&hybrid_results).iter().any(|text| text == "quokka"));

    // Disabling the sparse index returns a `bm25` knowledge base to dense.
    let dense_only = hash_kb(config(INDEX_TYPE_BM25)).with_sparse_index(false);
    assert!(!dense_only.has_sparse_index());
    assert_eq!(dense_only.sparse_index_len(), 0);
}

/// The agentic corpus embedder: dense ranks "report" docs first, "zephyrine" second.
fn agentic_embedder() -> Arc<TableEmbedding> {
    Arc::new(TableEmbedding::new(vec![
        ("report", [1.0, 0.0]),
        ("zephyrine", [0.0, 1.0]),
        ("report summary", [1.0, 0.0]),
        ("budget review", [0.9, 0.1]),
        ("acquisition notes", [0.1, 0.9]),
    ]))
}

/// The four-document corpus the agentic test iterates over.
fn agentic_corpus() -> Vec<String> {
    vec![
        "report summary".to_owned(),
        "budget review".to_owned(),
        "acquisition notes".to_owned(),
        "zephyrine".to_owned(),
    ]
}

#[tokio::test]
async fn agentic_retrieval_improves_recall_and_stays_bounded() {
    let top_two = RetrievalConfig::default().with_top_k(2);
    let mut agentic_config = top_two.clone();
    agentic_config.agentic = true;

    let plain = knowledge_base(config(INDEX_TYPE_VECTOR), agentic_embedder());
    plain.add_texts(agentic_corpus()).await.unwrap();
    let plain_results = plain
        .retrieve_with_config("report", &top_two)
        .await
        .unwrap();
    assert_eq!(
        texts(&plain_results),
        vec!["report summary", "budget review"]
    );
    assert!(!texts(&plain_results).iter().any(|text| text == "zephyrine"));

    let scripted = Arc::new(ScriptedRewriter::new("zephyrine", &["zephyrine"]));
    let rewriter: Arc<dyn QueryRewriter> = scripted.clone();
    let agentic =
        knowledge_base(config(INDEX_TYPE_VECTOR), agentic_embedder()).with_rewriter(rewriter);
    agentic.add_texts(agentic_corpus()).await.unwrap();

    let agentic_results = agentic
        .retrieve_with_config("report", &agentic_config)
        .await
        .unwrap();
    assert_eq!(agentic_results.len(), 2);
    assert!(texts(&agentic_results)
        .iter()
        .any(|text| text == "zephyrine"));
    // Two iterations: exactly one rewrite between them, then the loop stops.
    assert_eq!(agentic.agentic_iterations(), DEFAULT_AGENTIC_ITERATIONS);
    assert_eq!(DEFAULT_AGENTIC_ITERATIONS, 2);
    assert_eq!(scripted.rewrites(), 1);

    // A budget of one iteration never rewrites and matches the plain channel.
    let single = Arc::new(ScriptedRewriter::new("zephyrine", &["zephyrine"]));
    let single_rewriter: Arc<dyn QueryRewriter> = single.clone();
    let bounded = knowledge_base(config(INDEX_TYPE_VECTOR), agentic_embedder())
        .with_agentic_iterations(1)
        .with_rewriter(single_rewriter);
    bounded.add_texts(agentic_corpus()).await.unwrap();
    let bounded_results = bounded
        .retrieve_with_config("report", &agentic_config)
        .await
        .unwrap();
    assert_eq!(
        texts(&bounded_results),
        vec!["report summary", "budget review"]
    );
    assert_eq!(single.rewrites(), 0);

    // A zero budget is a validation error, matching Python's `max_iter` check.
    let invalid = knowledge_base(config(INDEX_TYPE_VECTOR), agentic_embedder())
        .with_agentic_iterations(0)
        .with_rewriter(Arc::new(ScriptedRewriter::new("zephyrine", &["zephyrine"])));
    invalid
        .add_texts(vec!["report summary".to_owned()])
        .await
        .unwrap();
    let error = invalid
        .retrieve_with_config("report", &agentic_config)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[tokio::test]
async fn graph_store_indexes_triples_and_expands_retrieval() {
    let graph = Arc::new(InMemoryGraphStore::new());
    let mut graph_config = config(INDEX_TYPE_HYBRID);
    graph_config.use_graph = true;
    let kb = hash_kb(graph_config.clone())
        .with_sparse_index(true)
        .with_graph_store(graph.clone());
    assert_eq!(DEFAULT_GRAPH_HOPS, 2);

    let ids = kb
        .add_texts(vec![
            "Acme is a company. It builds rockets.".to_owned(),
            "acme is_a company".to_owned(),
        ])
        .await
        .unwrap();
    assert_eq!(ids.len(), 2);
    let triples = graph.all_triples().await.unwrap();
    assert!(
        triples
            .iter()
            .any(|triple| triple.subject.eq_ignore_ascii_case("acme")
                && triple.predicate == "is_a"
                && triple.object.eq_ignore_ascii_case("company")),
        "triples were not indexed: {triples:?}"
    );

    // Without a graph store, `use_graph` falls back to the plain channel.
    let fallback = hash_kb(graph_config).with_sparse_index(true);
    fallback
        .add_texts(vec![
            "Acme is a company. It builds rockets.".to_owned(),
            "acme is_a company".to_owned(),
        ])
        .await
        .unwrap();
    let top_one = RetrievalConfig::default().with_top_k(1);
    let plain_results = fallback
        .retrieve_with_config("rockets", &top_one)
        .await
        .unwrap();
    assert_eq!(plain_results.len(), 1);
    assert!(plain_results[0].text.contains("rockets"));

    // With the graph store, the connected chunk arrives through the triple.
    let graph_results = kb.retrieve_with_config("rockets", &top_one).await.unwrap();
    assert_eq!(graph_results.len(), 1);
    assert_eq!(graph_results[0].text, "acme is_a company");

    // Deletes drop the chunk from the sparse index but keep the triples.
    let triple_count = graph.len().await.unwrap();
    kb.delete_documents(&[ids[0].clone()]).await.unwrap();
    assert_eq!(kb.sparse_index_len(), 1);
    assert_eq!(graph.len().await.unwrap(), triple_count);
}

#[tokio::test]
async fn per_query_graph_expansion_uses_the_indexed_triples() {
    let graph = Arc::new(InMemoryGraphStore::new());
    let kb = hash_kb(config(INDEX_TYPE_HYBRID))
        .with_sparse_index(true)
        .with_graph_store(graph)
        .with_graph_hops(1);
    kb.add_texts(vec![
        "Acme is a company. It builds rockets.".to_owned(),
        "acme is_a company".to_owned(),
    ])
    .await
    .unwrap();

    let mut expansion = RetrievalConfig::default().with_top_k(1);
    expansion.graph_expansion = true;
    let results = kb
        .retrieve_with_config("rockets", &expansion)
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].text, "acme is_a company");

    // The same query without expansion stays on the chunk channel.
    let plain = kb
        .retrieve_with_config("rockets", &RetrievalConfig::default().with_top_k(1))
        .await
        .unwrap();
    assert_eq!(plain.len(), 1);
    assert!(plain[0].text.contains("rockets"));
}

#[tokio::test]
async fn filters_reach_every_channel() {
    let documents = vec![
        Document::with_id("d1", "quick brown fox")
            .with_metadata(Metadata::from_iter([("kind".to_owned(), "pet".into())])),
        Document::with_id("d2", "quick compiler notes")
            .with_metadata(Metadata::from_iter([("kind".to_owned(), "tech".into())])),
    ];
    let filtered = || RetrievalConfig::default().with_filter(MetadataFilter::eq("kind", "pet"));

    let dense = hash_kb(config(INDEX_TYPE_VECTOR));
    let bm25 = hash_kb(config(INDEX_TYPE_BM25));
    let hybrid = hash_kb(config(INDEX_TYPE_HYBRID)).with_sparse_index(true);
    for kb in [&dense, &bm25, &hybrid] {
        kb.add_documents(documents.clone()).await.unwrap();
        let results = kb.retrieve_with_config("quick", &filtered()).await.unwrap();
        assert_eq!(results.len(), 1, "channel {}", kb.config().index_type);
        assert_eq!(results[0].doc_id.as_deref(), Some("d1"));
        assert_eq!(
            results[0].metadata.get("kind"),
            Some(&serde_json::json!("pet"))
        );
    }
}

#[tokio::test]
async fn score_thresholds_reach_every_channel() {
    let corpus = vec![
        "quokka habitats".to_owned(),
        "cooking pasta".to_owned(),
        "rust ownership".to_owned(),
    ];
    let dense = hash_kb(config(INDEX_TYPE_VECTOR));
    let bm25 = hash_kb(config(INDEX_TYPE_BM25));
    let hybrid = hash_kb(config(INDEX_TYPE_HYBRID)).with_sparse_index(true);
    for kb in [&dense, &bm25, &hybrid] {
        kb.add_texts(corpus.clone()).await.unwrap();
    }

    // Dense scores live in `[0, 1]` and BM25 scores are far below 1e6.
    let impossible = RetrievalConfig::default().with_score_threshold(2.0);
    assert_eq!(
        dense
            .retrieve_with_config("quokka", &impossible)
            .await
            .unwrap()
            .len(),
        0
    );
    assert_eq!(
        bm25.retrieve_with_config(
            "quokka",
            &RetrievalConfig::default().with_score_threshold(1_000_000.0)
        )
        .await
        .unwrap()
        .len(),
        0
    );

    // The hybrid threshold filters the dense channel only (documented), so the
    // lexical match still comes through.
    let hybrid_results = hybrid
        .retrieve_with_config("quokka", &impossible)
        .await
        .unwrap();
    assert_eq!(texts(&hybrid_results), vec!["quokka habitats"]);
}

#[tokio::test]
async fn reranker_applies_after_the_bm25_and_hybrid_channels() {
    let corpus = vec![
        "quick brown fox".to_owned(),
        "quick quick fox".to_owned(),
        "lazy dog".to_owned(),
    ];
    for (index_type, sparse) in [(INDEX_TYPE_BM25, false), (INDEX_TYPE_HYBRID, true)] {
        let plain = hash_kb(config(index_type));
        let plain = if sparse {
            plain.with_sparse_index(true)
        } else {
            plain
        };
        plain.add_texts(corpus.clone()).await.unwrap();
        let top_two = RetrievalConfig::default().with_top_k(2);
        let expected = plain.retrieve_with_config("quick", &top_two).await.unwrap();
        assert_eq!(expected.len(), 2);

        let reranker = Arc::new(ReverseReranker::default());
        let target = hash_kb(config(index_type));
        let target = if sparse {
            target.with_sparse_index(true)
        } else {
            target
        };
        let target = target.with_reranker(reranker.clone());
        target.add_texts(corpus.clone()).await.unwrap();
        let actual = target
            .retrieve_with_config("quick", &top_two)
            .await
            .unwrap();
        let mut reversed = texts(&expected);
        reversed.reverse();
        assert_eq!(texts(&actual), reversed);
        assert_eq!(*reranker.seen_top_n.lock().unwrap(), Some(2));
    }
}

#[tokio::test]
async fn statistics_report_the_new_channels() {
    let graph = Arc::new(InMemoryGraphStore::new());
    let kb = hash_kb(config(INDEX_TYPE_HYBRID))
        .with_sparse_index(true)
        .with_graph_store(graph)
        .with_rewriter(Arc::new(ScriptedRewriter::new("next", &["next"])))
        .with_agentic(true);
    kb.add_texts(vec!["alpha one".to_owned(), "beta two".to_owned()])
        .await
        .unwrap();

    let statistics = kb.get_statistics().await.unwrap();
    assert_eq!(statistics.sparse_index_size, 2);
    assert!(statistics.has_graph);
    assert!(statistics.has_rewriter);
    assert!(statistics.agentic_enabled);
    assert!(!statistics.has_reranker);

    let value = serde_json::to_value(&statistics).unwrap();
    assert_eq!(value["sparse_index_size"], 2);
    assert_eq!(value["has_graph"], true);
    assert_eq!(value["has_rewriter"], true);
    assert_eq!(value["agentic_enabled"], true);

    // Statistics written before the channel fields existed still deserialize.
    let old = serde_json::json!({
        "kb_id": "kb-old",
        "index_type": "hybrid",
        "chunk_count": 4,
        "document_count": 2,
        "has_reranker": false,
        "has_loader": true,
        "unknown_future_field": 7
    });
    let restored: KnowledgeBaseStatistics = serde_json::from_value(old).unwrap();
    assert_eq!(restored.kb_id, "kb-old");
    assert_eq!(restored.sparse_index_size, 0);
    assert!(!restored.has_graph);
    assert!(!restored.has_rewriter);
    assert!(!restored.agentic_enabled);

    // The default heuristic rewriter needs no injection and keeps working.
    let default_rewriter = hash_kb(config(INDEX_TYPE_VECTOR)).with_agentic(true);
    default_rewriter
        .add_texts(vec!["alpha one".to_owned()])
        .await
        .unwrap();
    let agentic_config = RetrievalConfig {
        agentic: true,
        ..RetrievalConfig::default()
    };
    let results = default_rewriter
        .retrieve_with_config("alpha", &agentic_config)
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert!(
        !default_rewriter
            .get_statistics()
            .await
            .unwrap()
            .has_rewriter
    );
}

#[tokio::test]
async fn empty_queries_and_zero_top_k_are_validation_errors_on_every_channel() {
    let kb = hash_kb(config(INDEX_TYPE_BM25));
    kb.add_texts(vec!["alpha".to_owned()]).await.unwrap();
    let error = kb
        .retrieve_with_config("   ", &RetrievalConfig::default())
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    let error = kb
        .retrieve_with_config("alpha", &RetrievalConfig::default().with_top_k(0))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
}
