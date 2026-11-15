#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Integration tests for the knowledge base pipeline: add, retrieve, filter,
//! rerank, delete, update, parsing helpers, statistics, and multi-KB retrieval.
//!
//! Everything runs offline with [`InMemoryVectorStore`] and [`HashEmbedding`],
//! so chunk counts, scores, and orderings are deterministic. The behavior ports
//! mirror `tests/unit_tests/core/retrieval` in the Python reference; the
//! in-memory store and hash embedder are Rust additions, so their score
//! convention (`(cos + 1) / 2`) is asserted as documented.
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_retrieval::knowledge_base::{
    retrieve_multi_kb_reporting, retrieve_multi_kb_strict, retrieve_multi_kb_with_source_reporting,
    retrieve_multi_kb_with_source_strict,
};
use openjiuwen_retrieval::{
    retrieve_multi_kb, retrieve_multi_kb_with_source, Document, Embedding, HashEmbedding,
    InMemoryVectorStore, IndexConfig, KnowledgeBaseConfig, KnowledgeBaseStatistics, Metadata,
    MetadataFilter, Reranker, RetrievalConfig, RetrievalResult, ScoreReranker, SearchResult,
    SimpleKnowledgeBase, VectorStore, INDEX_TYPE_BM25,
};
use serde_json::json;

/// Dimension used by every test embedder.
const EMBEDDING_DIMENSION: usize = 64;
/// Absolute tolerance for score comparisons.
const TOLERANCE: f32 = 1e-5;
/// `CharChunker` default size used by the pipeline fixtures.
const CHUNK_SIZE: usize = 512;
/// `CharChunker` default overlap used by the pipeline fixtures.
const CHUNK_OVERLAP: usize = 50;

/// Builds an empty in-memory store.
fn store() -> Arc<InMemoryVectorStore> {
    Arc::new(InMemoryVectorStore::new())
}

/// Builds a knowledge base from `config` with a [`HashEmbedding`] encoder.
fn knowledge_base(
    config: KnowledgeBaseConfig,
    store: Arc<InMemoryVectorStore>,
) -> SimpleKnowledgeBase {
    let embedder: Arc<dyn Embedding> = Arc::new(HashEmbedding::new(EMBEDDING_DIMENSION));
    let vector_store: Arc<dyn VectorStore> = store;
    SimpleKnowledgeBase::from_config(config, vector_store, embedder).unwrap()
}

/// Builds a shared knowledge base with a fresh store.
fn kb(config: KnowledgeBaseConfig) -> Arc<SimpleKnowledgeBase> {
    Arc::new(knowledge_base(config, store()))
}

/// A reranker that reverses the hits, then repeats them, recording each `top_n`.
///
/// The repetition makes the knowledge base's final truncation observable even
/// though the store is already queried with `top_k`.
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
        let repeated = documents.clone();
        documents.extend(repeated);
        Ok(documents)
    }
}

/// An embedder that always fails, used to prove multi-KB error tolerance.
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

/// Builds a knowledge base whose embedder always fails, so every retrieval errors.
fn failing_kb(kb_id: &str) -> Arc<SimpleKnowledgeBase> {
    let embedder: Arc<dyn Embedding> = Arc::new(FailingEmbedding);
    let vector_store: Arc<dyn VectorStore> = store();
    Arc::new(
        SimpleKnowledgeBase::from_config(KnowledgeBaseConfig::new(kb_id), vector_store, embedder)
            .unwrap(),
    )
}

/// Extracts result texts in order.
fn texts(results: &[RetrievalResult]) -> Vec<String> {
    results.iter().map(|result| result.text.clone()).collect()
}

#[tokio::test]
async fn add_documents_returns_ids_in_input_order() {
    let kb = kb(KnowledgeBaseConfig::new("kb-add"));
    let ids = kb
        .add_documents(vec![
            Document::with_id("doc-a", "alpha text"),
            Document::with_id("doc-b", "beta text"),
            Document::with_id("doc-c", "gamma text"),
        ])
        .await
        .unwrap();
    assert_eq!(ids, vec!["doc-a", "doc-b", "doc-c"]);
    // The default 512/50 chunker keeps each short document in one chunk.
    assert_eq!(kb.count().await.unwrap(), 3);
}

#[tokio::test]
async fn add_documents_handles_empty_batches_and_blank_ids() {
    let kb = kb(KnowledgeBaseConfig::new("kb-blank"));
    let ids = kb.add_documents(Vec::<Document>::new()).await.unwrap();
    assert_eq!(ids.len(), 0);

    let ids = kb
        .add_documents(vec![Document::with_id("   ", "padded id")])
        .await
        .unwrap();
    assert_eq!(ids.len(), 1);
    assert_eq!(ids[0].len(), 36, "blank ids are replaced by a uuid4");
    assert_eq!(kb.count().await.unwrap(), 1);
}

#[tokio::test]
async fn add_documents_records_documents_that_produce_no_chunks() {
    let kb = kb(KnowledgeBaseConfig::new("kb-empty-doc"));
    let ids = kb
        .add_documents(vec![Document::with_id("doc-empty", "")])
        .await
        .unwrap();
    assert_eq!(ids, vec!["doc-empty"]);
    assert_eq!(kb.count().await.unwrap(), 0);
}

#[tokio::test]
async fn add_texts_wraps_texts_in_documents() {
    let kb = kb(KnowledgeBaseConfig::new("kb-texts"));
    let ids = kb
        .add_texts(vec!["first text".to_owned(), "second text".to_owned()])
        .await
        .unwrap();
    assert_eq!(ids.len(), 2);
    assert!(ids.iter().all(|id| id.len() == 36));
    assert_eq!(kb.count().await.unwrap(), 2);
}

#[tokio::test]
async fn retrieve_returns_the_exact_match_first_with_ids() {
    let store = store();
    let kb = Arc::new(knowledge_base(
        KnowledgeBaseConfig::new("kb-retrieve"),
        store.clone(),
    ));
    kb.add_documents(vec![
        Document::with_id("doc-a", "alpha text"),
        Document::with_id("doc-b", "totally unrelated words"),
    ])
    .await
    .unwrap();

    let results = kb.retrieve("alpha text").await.unwrap();
    assert_eq!(results[0].text, "alpha text");
    assert_eq!(results[0].doc_id.as_deref(), Some("doc-a"));
    assert!(
        (results[0].score - 1.0).abs() < TOLERANCE,
        "an identical query scores 1.0, got {}",
        results[0].score
    );

    // The chunk id is the store record id, and metadata carries `doc_id`.
    let chunk_id = results[0].chunk_id.clone().unwrap();
    assert!(store.get(&chunk_id).await.unwrap().is_some());
    assert_eq!(
        results[0].metadata.get("doc_id"),
        Some(&serde_json::Value::String("doc-a".to_owned()))
    );
    assert_eq!(results.len(), 2);
}

#[tokio::test]
async fn retrieve_rejects_blank_queries_and_zero_top_k() {
    let kb = kb(KnowledgeBaseConfig::new("kb-validation"));
    let error = kb.retrieve("   ").await.unwrap_err();
    assert!(
        error.to_string().contains("query must not be empty"),
        "{error}"
    );

    let config = RetrievalConfig::default().with_top_k(0);
    let error = kb.retrieve_with_config("alpha", &config).await.unwrap_err();
    assert!(
        error.to_string().contains("top_k must be greater than 0"),
        "{error}"
    );
}

#[tokio::test]
async fn metadata_filters_restrict_the_results() {
    let kb = kb(KnowledgeBaseConfig::new("kb-filter"));
    kb.add_documents(vec![
        Document::with_id("doc-a", "alpha text").metadata_entry("category", "a"),
        Document::with_id("doc-b", "alpha text").metadata_entry("category", "b"),
    ])
    .await
    .unwrap();

    let config = RetrievalConfig::default().with_filter(MetadataFilter::eq("category", "a"));
    let results = kb
        .retrieve_with_config("alpha text", &config)
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].doc_id.as_deref(), Some("doc-a"));
    assert_eq!(
        results[0].metadata.get("category"),
        Some(&serde_json::Value::String("a".to_owned()))
    );
}

#[tokio::test]
async fn score_threshold_drops_unrelated_documents() {
    let kb = kb(KnowledgeBaseConfig::new("kb-threshold"));
    kb.add_documents(vec![
        Document::with_id("doc-a", "alpha text"),
        Document::with_id("doc-b", "totally unrelated words"),
    ])
    .await
    .unwrap();

    let config = RetrievalConfig::default().with_score_threshold(0.9);
    let results = kb
        .retrieve_with_config("alpha text", &config)
        .await
        .unwrap();
    assert_eq!(texts(&results), vec!["alpha text".to_owned()]);

    // Without the threshold both chunks are returned.
    assert_eq!(kb.retrieve("alpha text").await.unwrap().len(), 2);
}

#[tokio::test]
async fn a_reranker_receives_top_k_and_its_order_is_truncated() {
    let documents = || {
        vec![
            Document::with_id("doc-a", "first text"),
            Document::with_id("doc-b", "second text"),
            Document::with_id("doc-c", "third text"),
        ]
    };

    let baseline_kb = Arc::new(knowledge_base(
        KnowledgeBaseConfig::new("kb-rerank"),
        store(),
    ));
    baseline_kb.add_documents(documents()).await.unwrap();
    let top_k = 2;
    let baseline = baseline_kb
        .retrieve_with_config("first text", &RetrievalConfig::default().with_top_k(top_k))
        .await
        .unwrap();
    assert_eq!(baseline.len(), top_k);

    let reranker = Arc::new(ReverseReranker::default());
    let reranked_kb = Arc::new(
        knowledge_base(KnowledgeBaseConfig::new("kb-rerank"), store())
            .with_reranker(reranker.clone()),
    );
    reranked_kb.add_documents(documents()).await.unwrap();

    let results = reranked_kb
        .retrieve_with_config("first text", &RetrievalConfig::default().with_top_k(top_k))
        .await
        .unwrap();
    // The mandated signature passes the configured `top_k` to the reranker.
    assert_eq!(*reranker.seen_top_n.lock().unwrap(), Some(top_k));
    // The reranker's reversed order wins and its repeated hits are truncated.
    assert_eq!(results.len(), top_k);
    assert_eq!(results[0].text, baseline[1].text);
    assert_eq!(results[1].text, baseline[0].text);
}

#[tokio::test]
async fn delete_documents_removes_chunks_and_reports_the_count() {
    let kb = kb(KnowledgeBaseConfig::new("kb-delete"));
    kb.add_documents(vec![
        Document::with_id("doc-a", "alpha text"),
        Document::with_id("doc-b", "beta text"),
    ])
    .await
    .unwrap();

    let removed = kb.delete_documents(&["doc-a".to_owned()]).await.unwrap();
    assert_eq!(removed, 1);
    assert_eq!(kb.count().await.unwrap(), 1);
    let results = kb.retrieve("alpha text").await.unwrap();
    assert!(
        results
            .iter()
            .all(|result| result.doc_id.as_deref() != Some("doc-a")),
        "deleted documents must not be retrievable"
    );
    assert_eq!(kb.get_statistics().await.unwrap().document_count, 1);

    // Deleting nothing (or an unknown id) is a no-op, not an error.
    assert_eq!(kb.delete_documents(&[]).await.unwrap(), 0);
    assert_eq!(
        kb.delete_documents(&["missing".to_owned()]).await.unwrap(),
        0
    );
}

#[tokio::test]
async fn update_documents_replaces_content_and_keeps_the_chunk_count() {
    let kb = kb(KnowledgeBaseConfig::new("kb-update"));
    kb.add_documents(vec![Document::with_id("doc-a", "alpha text")])
        .await
        .unwrap();

    let ids = kb
        .update_documents(vec![Document::with_id("doc-a", "omega text")])
        .await
        .unwrap();
    assert_eq!(ids, vec!["doc-a"]);
    assert_eq!(kb.count().await.unwrap(), 1);

    let results = kb.retrieve("omega text").await.unwrap();
    assert_eq!(results[0].text, "omega text");
    assert_eq!(results[0].doc_id.as_deref(), Some("doc-a"));
    assert!(
        texts(&kb.retrieve("alpha text").await.unwrap())
            .iter()
            .all(|text| text != "alpha text"),
        "the replaced text must not survive"
    );
}

#[tokio::test]
async fn from_config_honours_the_configured_chunk_window() {
    let mut config = KnowledgeBaseConfig::new("kb-chunking");
    config.chunk_size = 10;
    config.chunk_overlap = 0;
    let kb = kb(config);
    kb.add_documents(vec![Document::with_id(
        "doc-long",
        "abcdefghijklmnopqrstuvwxyz",
    )])
    .await
    .unwrap();
    // 26 characters split into windows of 10 with no overlap: 10 + 10 + 6.
    assert_eq!(kb.count().await.unwrap(), 3);
}

#[tokio::test]
async fn from_config_rejects_an_invalid_chunk_window() {
    let mut config = KnowledgeBaseConfig::new("kb-invalid");
    config.chunk_size = 10;
    config.chunk_overlap = 10;
    let embedder: Arc<dyn Embedding> = Arc::new(HashEmbedding::new(EMBEDDING_DIMENSION));
    let vector_store: Arc<dyn VectorStore> = store();
    let error = SimpleKnowledgeBase::from_config(config, vector_store, embedder)
        .err()
        .expect("an overlap equal to the chunk size is invalid");
    assert!(error.to_string().contains("building chunker"), "{error}");
}

#[tokio::test]
async fn parse_files_skips_missing_sources() {
    let dir = tempfile::tempdir().unwrap();
    let existing = dir.path().join("notes.txt");
    std::fs::write(&existing, "loaded text").unwrap();
    let missing = dir.path().join("missing.txt");

    let kb = kb(KnowledgeBaseConfig::new("kb-parse"));
    let documents = kb
        .parse_files(&[
            existing.to_str().unwrap().to_owned(),
            missing.to_str().unwrap().to_owned(),
        ])
        .await
        .unwrap();
    assert_eq!(documents.len(), 1);
    assert_eq!(documents[0].text, "loaded text");
    assert_eq!(
        documents[0].metadata.get("file_path"),
        Some(&serde_json::Value::String(
            existing.to_str().unwrap().to_owned()
        ))
    );
}

#[tokio::test]
async fn parse_urls_skips_sources_that_are_not_urls() {
    let kb = kb(KnowledgeBaseConfig::new("kb-urls"));
    let documents = kb.parse_urls(&["not a url".to_owned()]).await.unwrap();
    assert_eq!(documents, Vec::new());
}

#[tokio::test]
async fn statistics_report_the_configuration_and_counts() {
    let reranked = Arc::new(
        knowledge_base(KnowledgeBaseConfig::new("kb-stats"), store())
            .with_reranker(Arc::new(ScoreReranker::new(None))),
    );
    reranked
        .add_documents(vec![
            Document::with_id("doc-a", "alpha text"),
            Document::with_id("doc-b", "beta text"),
        ])
        .await
        .unwrap();

    let statistics: KnowledgeBaseStatistics = reranked.get_statistics().await.unwrap();
    assert_eq!(statistics.kb_id, "kb-stats");
    assert_eq!(statistics.index_type, "hybrid");
    assert_eq!(statistics.chunk_count, 2);
    assert_eq!(statistics.document_count, 2);
    assert!(statistics.has_reranker);
    assert!(statistics.has_loader);

    // A knowledge base without a reranker reports `has_reranker = false`.
    let plain = kb(KnowledgeBaseConfig::new("kb-stats-plain"));
    let statistics = plain.get_statistics().await.unwrap();
    assert!(!statistics.has_reranker);
    assert!(statistics.has_loader);
    assert_eq!(statistics.chunk_count, 0);
    assert_eq!(statistics.document_count, 0);
}

#[tokio::test]
async fn multi_kb_merges_identical_text_and_sorts_the_sources() {
    let first = kb(KnowledgeBaseConfig::new("kb-1"));
    let second = kb(KnowledgeBaseConfig::new("kb-2"));
    for target in [&first, &second] {
        target
            .add_documents(vec![
                Document::with_id("doc-shared", "shared text"),
                Document::with_id("doc-unique", format!("unique {}", target.id())),
            ])
            .await
            .unwrap();
    }

    let merged =
        retrieve_multi_kb_with_source(&[first.clone(), second.clone()], "shared text", Some(5))
            .await;
    let shared: Vec<&openjiuwen_retrieval::MultiKBRetrievalResult> = merged
        .iter()
        .filter(|result| result.text == "shared text")
        .collect();
    assert_eq!(shared.len(), 1, "identical text is merged into one result");
    assert_eq!(shared[0].kb_ids, vec!["kb-1".to_owned(), "kb-2".to_owned()]);
    assert!((shared[0].score - 1.0).abs() < TOLERANCE);
    assert!((shared[0].raw_score - shared[0].score).abs() < TOLERANCE);
    assert!(merged.len() <= 5);

    let plain = retrieve_multi_kb(&[first, second], "shared text", Some(1)).await;
    assert_eq!(plain.len(), 1);
    assert_eq!(plain[0].text, "shared text");
}

#[tokio::test]
async fn multi_kb_tolerates_a_failing_knowledge_base() {
    let failing_embedder: Arc<dyn Embedding> = Arc::new(FailingEmbedding);
    let vector_store: Arc<dyn VectorStore> = store();
    let failing = Arc::new(
        SimpleKnowledgeBase::from_config(
            KnowledgeBaseConfig::new("kb-failing"),
            vector_store,
            failing_embedder,
        )
        .unwrap(),
    );
    let healthy = kb(KnowledgeBaseConfig::new("kb-healthy"));
    healthy
        .add_documents(vec![Document::with_id("doc-a", "shared text")])
        .await
        .unwrap();

    let merged =
        retrieve_multi_kb_with_source(&[failing.clone(), healthy.clone()], "shared text", None)
            .await;
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].text, "shared text");
    assert_eq!(merged[0].kb_ids, vec!["kb-healthy".to_owned()]);

    let plain = retrieve_multi_kb(&[failing, healthy], "shared text", None).await;
    assert_eq!(plain.len(), 1);
    assert_eq!(plain[0].text, "shared text");
}

#[tokio::test]
async fn multi_kb_without_knowledge_bases_returns_nothing() {
    assert_eq!(retrieve_multi_kb(&[], "query", None).await, Vec::new());
    assert_eq!(
        retrieve_multi_kb_with_source(&[], "query", Some(3)).await,
        Vec::new()
    );
}

#[test]
fn knowledge_base_config_is_exposed() {
    let kb = kb(KnowledgeBaseConfig::new("kb-config"));
    assert_eq!(kb.id(), "kb-config");
    assert_eq!(kb.config().kb_id, "kb-config");
    assert_eq!(kb.config().chunk_size, CHUNK_SIZE);
    assert_eq!(kb.config().chunk_overlap, CHUNK_OVERLAP);
}

#[tokio::test]
async fn from_config_rejects_an_unknown_index_type() {
    let mut config = KnowledgeBaseConfig::new("kb-unknown");
    config.index_type = "faiss".to_owned();
    let embedder: Arc<dyn Embedding> = Arc::new(HashEmbedding::new(EMBEDDING_DIMENSION));
    let error = SimpleKnowledgeBase::from_config(config, store(), embedder)
        .err()
        .expect("an unknown index type must be rejected");
    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(
        error
            .to_string()
            .contains("validating knowledge base configuration"),
        "{error}"
    );
}

/// An embedder that advertises the image+text capability no crate type has.
struct MultimodalEmbedding {
    inner: HashEmbedding,
}

#[async_trait]
impl Embedding for MultimodalEmbedding {
    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    fn supports_multimodal(&self) -> bool {
        true
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.inner.embed_query(text).await
    }

    async fn embed_documents(
        &self,
        texts: &[String],
        batch_size: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        self.inner.embed_documents(texts, batch_size).await
    }
}

/// An image document as [`openjiuwen_retrieval::ImageLoader`] emits it.
fn image_document() -> Document {
    Document::with_id("img-1", "[image] cat.png").with_metadata(Metadata::from_iter([(
        "modality".to_owned(),
        json!("image"),
    )]))
}

#[tokio::test]
async fn use_caption_for_images_gates_the_image_embedding_path() {
    // Text-only embedders always take the caption branch, which is Python's
    // `not callable(embed_multimodal)` case, no matter what the flag says.
    for flag in [false, true] {
        let mut config = KnowledgeBaseConfig::new("kb-image");
        config.use_caption_for_images = flag;
        let kb = knowledge_base(config, store());
        kb.add_documents(vec![image_document()]).await.unwrap();
        assert_eq!(kb.count().await.unwrap(), 1, "flag = {flag}");
    }

    // A multimodal embedder with the flag off asks for image+text embedding,
    // which this crate does not implement: a typed error, not a silent caption.
    let mut config = KnowledgeBaseConfig::new("kb-image");
    config.use_caption_for_images = false;
    let embedder: Arc<dyn Embedding> = Arc::new(MultimodalEmbedding {
        inner: HashEmbedding::new(EMBEDDING_DIMENSION),
    });
    let kb = SimpleKnowledgeBase::from_config(config, store(), embedder).unwrap();
    let error = kb
        .add_documents(vec![image_document()])
        .await
        .expect_err("the multimodal branch is deferred");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(
        error.to_string().contains("use_caption_for_images = true"),
        "{error}"
    );
    assert!(kb.count().await.unwrap() == 0, "nothing may be indexed");

    // Turning the flag on selects the caption branch for the same embedder.
    let mut config = KnowledgeBaseConfig::new("kb-image");
    config.use_caption_for_images = true;
    let embedder: Arc<dyn Embedding> = Arc::new(MultimodalEmbedding {
        inner: HashEmbedding::new(EMBEDDING_DIMENSION),
    });
    let kb = SimpleKnowledgeBase::from_config(config, store(), embedder).unwrap();
    kb.add_documents(vec![image_document()]).await.unwrap();
    assert_eq!(kb.count().await.unwrap(), 1);
}

#[tokio::test]
async fn from_index_config_honours_every_index_field() {
    let index = IndexConfig {
        index_name: "idx-docs".to_owned(),
        index_type: INDEX_TYPE_BM25.to_owned(),
        use_caption_for_images: true,
    };
    let embedder: Arc<dyn Embedding> = Arc::new(HashEmbedding::new(EMBEDDING_DIMENSION));
    let kb = SimpleKnowledgeBase::from_index_config(&index, store(), embedder).unwrap();
    assert_eq!(kb.id(), "idx-docs");
    assert_eq!(kb.config().index_type, INDEX_TYPE_BM25);
    assert!(kb.config().use_caption_for_images);
    // A `bm25` index type enables the sparse index, so retrieval is lexical.
    kb.add_texts(vec!["the quokka lives here".to_owned()])
        .await
        .unwrap();
    assert_eq!(kb.sparse_index_len(), 1);
    let results = kb.retrieve("quokka").await.unwrap();
    assert_eq!(results.len(), 1);
}

#[tokio::test]
async fn from_index_config_rejects_a_blank_name_and_unknown_type() {
    let embedder: Arc<dyn Embedding> = Arc::new(HashEmbedding::new(EMBEDDING_DIMENSION));
    let blank = IndexConfig {
        index_name: "  ".to_owned(),
        index_type: INDEX_TYPE_BM25.to_owned(),
        use_caption_for_images: false,
    };
    let error = SimpleKnowledgeBase::from_index_config(&blank, store(), Arc::clone(&embedder))
        .err()
        .expect("a blank index name must be rejected");
    assert_eq!(error.kind(), ErrorKind::Config);

    let unknown = IndexConfig {
        index_name: "idx".to_owned(),
        index_type: "annoy".to_owned(),
        use_caption_for_images: false,
    };
    let error = SimpleKnowledgeBase::from_index_config(&unknown, store(), embedder)
        .err()
        .expect("an unknown index type must be rejected");
    assert_eq!(error.kind(), ErrorKind::Config);
}

#[tokio::test]
async fn multi_kb_reporting_lists_every_failed_knowledge_base() {
    let healthy = kb(KnowledgeBaseConfig::new("kb-healthy"));
    healthy
        .add_documents(vec![Document::with_id("doc-a", "shared text")])
        .await
        .unwrap();
    let kbs = vec![failing_kb("kb-fail-1"), healthy, failing_kb("kb-fail-2")];

    let report = retrieve_multi_kb_reporting(&kbs, "shared text", None).await;
    assert_eq!(texts(&report.results), vec!["shared text".to_owned()]);
    assert_eq!(
        report
            .failures
            .iter()
            .map(|failure| failure.kb_id.as_str())
            .collect::<Vec<_>>(),
        vec!["kb-fail-1", "kb-fail-2"]
    );
    for failure in &report.failures {
        assert!(!failure.error.is_empty(), "{failure:?}");
        assert!(failure.error.contains("boom"), "{failure:?}");
    }

    // The compatible function keeps returning only the surviving results.
    let compatible = retrieve_multi_kb(&kbs, "shared text", None).await;
    assert_eq!(compatible, report.results);
}

#[tokio::test]
async fn multi_kb_with_source_reporting_lists_every_failed_knowledge_base() {
    let healthy = kb(KnowledgeBaseConfig::new("kb-healthy"));
    healthy
        .add_documents(vec![Document::with_id("doc-a", "shared text")])
        .await
        .unwrap();
    let kbs = vec![failing_kb("kb-fail-1"), healthy, failing_kb("kb-fail-2")];

    let report = retrieve_multi_kb_with_source_reporting(&kbs, "shared text", None).await;
    assert_eq!(report.results.len(), 1);
    assert_eq!(report.results[0].text, "shared text");
    assert_eq!(report.results[0].kb_ids, vec!["kb-healthy".to_owned()]);
    assert_eq!(
        report
            .failures
            .iter()
            .map(|failure| failure.kb_id.as_str())
            .collect::<Vec<_>>(),
        vec!["kb-fail-1", "kb-fail-2"]
    );
    for failure in &report.failures {
        assert!(failure.error.contains("boom"), "{failure:?}");
    }

    let compatible = retrieve_multi_kb_with_source(&kbs, "shared text", None).await;
    assert_eq!(compatible, report.results);
}

#[tokio::test]
async fn multi_kb_strict_keeps_partial_results() {
    let healthy = kb(KnowledgeBaseConfig::new("kb-healthy"));
    healthy
        .add_documents(vec![Document::with_id("doc-a", "shared text")])
        .await
        .unwrap();
    let kbs = vec![failing_kb("kb-fail-1"), healthy, failing_kb("kb-fail-2")];

    let plain = retrieve_multi_kb_strict(&kbs, "shared text", None)
        .await
        .expect("a partial failure is not an error");
    assert_eq!(plain, retrieve_multi_kb(&kbs, "shared text", None).await);

    let with_source = retrieve_multi_kb_with_source_strict(&kbs, "shared text", None)
        .await
        .expect("a partial failure is not an error");
    assert_eq!(
        with_source,
        retrieve_multi_kb_with_source(&kbs, "shared text", None).await
    );
    assert_eq!(with_source.len(), 1);
    assert_eq!(with_source[0].kb_ids, vec!["kb-healthy".to_owned()]);
}

#[tokio::test]
async fn multi_kb_strict_fails_when_every_knowledge_base_fails() {
    let kbs = vec![failing_kb("kb-fail-1"), failing_kb("kb-fail-2")];

    let error = retrieve_multi_kb_strict(&kbs, "shared text", None)
        .await
        .expect_err("every knowledge base failed");
    assert_eq!(error.kind(), ErrorKind::Retrieval);
    let message = error.to_string();
    assert!(message.contains("kb-fail-1"), "{message}");
    assert!(message.contains("kb-fail-2"), "{message}");

    let error = retrieve_multi_kb_with_source_strict(&kbs, "shared text", None)
        .await
        .expect_err("every knowledge base failed");
    assert_eq!(error.kind(), ErrorKind::Retrieval);
    let message = error.to_string();
    assert!(message.contains("kb-fail-1"), "{message}");
    assert!(message.contains("kb-fail-2"), "{message}");
}

#[tokio::test]
async fn multi_kb_without_knowledge_bases_reports_nothing() {
    // An empty slice is a successful no-op for both strict variants.
    let plain = retrieve_multi_kb_strict(&[], "query", None)
        .await
        .expect("an empty slice is a successful no-op");
    assert!(plain.is_empty(), "{plain:?}");
    let with_source = retrieve_multi_kb_with_source_strict(&[], "query", None)
        .await
        .expect("an empty slice is a successful no-op");
    assert!(with_source.is_empty(), "{with_source:?}");

    let report = retrieve_multi_kb_reporting(&[], "query", None).await;
    assert_eq!(report.results, Vec::new());
    assert_eq!(report.failures, Vec::new());

    let report = retrieve_multi_kb_with_source_reporting(&[], "query", None).await;
    assert_eq!(report.results, Vec::new());
    assert_eq!(report.failures, Vec::new());
}
