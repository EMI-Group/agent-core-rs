#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
//! Per-scope model and embedding swaps through the facade's handle registry.
//!
//! These tests pin the contract of [`LongTermMemoryBuilder::model_named`] /
//! [`LongTermMemoryBuilder::embedding_named`] and of
//! [`ScopeAnalyzerOverrideSpec::resolve`]:
//!
//! - a name registered on the builder resolves, whatever the handle's own `model_name()`;
//! - an unregistered name is a typed `Error::Unsupported` listing the registered names —
//!   never a silent fallback to the engine handle and never a silent drop;
//! - a resolved per-scope *model* is the model the analyzer actually calls;
//! - a resolved per-scope *embedding* is the embedder of the scope's writes, dedupe scan
//!   and recall, so two embedders of different dimensions coexist in one facade.
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::store::{
    InMemoryDocumentStore, InMemoryKvStore, InMemoryVectorStore, VectorStore,
};
use openjiuwen_memory::analyzer::{ExtractorKind, StagedAnalyzer};
use openjiuwen_memory::config::MemoryEngineConfig;
use openjiuwen_memory::facade::{LongTermMemory, ScopeAnalyzerOverrideSpec};
use openjiuwen_memory::record::{
    AddOptions, MemoryScope, MemoryType, NewMemory, RecallHit, SearchOptions,
};
use openjiuwen_memory::store::{keys, MemoryStores};
use openjiuwen_retrieval::embedding::{Embedding, HashEmbedding};
use openjiuwen_test_support::ScriptedModel;

/// The embedding the facade is built with (eight dimensions).
const BASE_DIMENSIONS: usize = 8;
/// The embedding one scope swaps in (twenty-four dimensions).
const SCOPED_DIMENSIONS: usize = 24;
/// The name the scoped embedding is registered under.
const SCOPED_EMBEDDING: &str = "wide";
/// The name the scoped model is registered under.
const SCOPED_MODEL: &str = "scoped-model";
/// A model answer the LLM extractor parses into exactly one fact.
const FACTS_RESPONSE: &str = "{\"facts\":[{\"content\":\"Alice prefers dark mode\",\
    \"type\":\"user_profile\",\"importance\":0.9}]}";

fn scope(scope_id: &str) -> MemoryScope {
    MemoryScope::new("alice", scope_id)
}

fn conversation() -> openjiuwen_memory::analyzer::ExtractionInput {
    let mut input = openjiuwen_memory::analyzer::ExtractionInput::default();
    input.push_user("I prefer dark mode.");
    input
}

/// A hash embedder that counts the queries it served.
struct CountingEmbedding {
    /// The vector source.
    inner: HashEmbedding,
    /// Number of `embed_query` calls.
    queries: Arc<AtomicUsize>,
}

impl CountingEmbedding {
    /// Builds a `dimensions`-sized embedder reporting `name`, plus its query counter.
    fn new(name: &str, dimensions: usize) -> (Arc<Self>, Arc<AtomicUsize>) {
        let queries = Arc::new(AtomicUsize::new(0));
        let embedding = Arc::new(Self {
            inner: HashEmbedding::new(dimensions).with_model_name(name),
            queries: Arc::clone(&queries),
        });
        (embedding, queries)
    }
}

#[async_trait]
impl Embedding for CountingEmbedding {
    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    fn dimension_hint(&self) -> Option<usize> {
        self.inner.dimension_hint()
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.queries.fetch_add(1, Ordering::SeqCst);
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

/// The facade, its vector store, and the query counters of both embedders.
struct Fixture {
    /// The facade under test.
    ltm: Arc<LongTermMemory>,
    /// The vector backend, inspected for collection dimensions.
    vectors: Arc<InMemoryVectorStore>,
    /// Query count of the base (engine) embedder.
    base_queries: Arc<AtomicUsize>,
    /// Query count of the scoped embedder.
    scoped_queries: Arc<AtomicUsize>,
}

/// Builds a facade with an eight-dimension engine embedder and a named 24-dimension one.
fn embedding_fixture() -> Fixture {
    let vectors = Arc::new(InMemoryVectorStore::new());
    let stores = MemoryStores::from_parts(
        Arc::new(InMemoryKvStore::new()),
        vectors.clone(),
        Arc::new(InMemoryDocumentStore::new()),
    );
    let (base, base_queries) = CountingEmbedding::new("hash-embedding", BASE_DIMENSIONS);
    let (scoped, scoped_queries) = CountingEmbedding::new(SCOPED_EMBEDDING, SCOPED_DIMENSIONS);
    let ltm = LongTermMemory::builder(MemoryEngineConfig::default())
        .stores(stores)
        .embedding(base)
        .embedding_named(SCOPED_EMBEDDING, scoped)
        .build()
        .expect("facade");
    Fixture {
        ltm,
        vectors,
        base_queries,
        scoped_queries,
    }
}

/// The stored vector dimension of `scope`'s semantic collection.
async fn stored_dimension(vectors: &InMemoryVectorStore, scope: &MemoryScope) -> Option<usize> {
    let collection = keys::vector_collection(scope, MemoryType::SemanticMemory);
    let schema = vectors
        .get_schema(&collection)
        .await
        .expect("schema read")
        .expect("the scope's collection exists");
    schema
        .fields
        .iter()
        .find(|field| field.is_vector())
        .and_then(|field| field.dim)
}

async fn search(ltm: &LongTermMemory, scope: &MemoryScope, query: &str) -> Vec<RecallHit> {
    ltm.search(
        scope,
        query,
        &SearchOptions::default().with_threshold(0.0).with_top_k(5),
    )
    .await
    .expect("search")
}

/// A registry-resolved model really is the model the analyzer calls, and the base model is
/// not reached; naming the engine's own handle resolves to it as well.
#[tokio::test]
async fn a_registered_per_scope_model_is_the_one_the_analyzer_calls() {
    let base = ScriptedModel::builder()
        .model_name("scripted-model")
        .respond_text(FACTS_RESPONSE)
        .build();
    let scoped = ScriptedModel::builder()
        .model_name("scoped-model")
        .respond_text(FACTS_RESPONSE)
        .build();
    let base_handle = base.clone();
    let scoped_handle = scoped.clone();
    let ltm = LongTermMemory::builder(MemoryEngineConfig::default())
        .model(Arc::new(base))
        .model_named(SCOPED_MODEL, Arc::new(scoped))
        .build()
        .expect("facade");
    let registry = ltm.scope_override_handles();
    assert_eq!(
        registry.model_names(),
        vec!["scoped-model", "scripted-model"]
    );
    assert_eq!(registry.embedding_names(), Vec::<&str>::new());

    // A scope naming the registered model analyzes with it.
    ltm.set_scope_overrides(
        "ws",
        &ScopeAnalyzerOverrideSpec::new()
            .with_extractor(ExtractorKind::Llm)
            .with_model(SCOPED_MODEL),
    )
    .await
    .expect("store the override");
    let report = ltm
        .remember(&scope("ws"), &conversation(), &AddOptions::default())
        .await
        .expect("remember");
    assert_eq!(report.extracted, 1, "{report:?}");
    assert_eq!(scoped_handle.call_count(), 1);
    assert_eq!(base_handle.call_count(), 0, "the base model is not called");
    assert_eq!(ltm.list(&scope("ws"), &[]).await.expect("list").len(), 1);

    // A scope naming the engine's own handle (`model_name()`) resolves to it.
    ltm.set_scope_overrides(
        "other",
        &ScopeAnalyzerOverrideSpec::new()
            .with_extractor(ExtractorKind::Llm)
            .with_model("scripted-model"),
    )
    .await
    .expect("store the override");
    ltm.remember(&scope("other"), &conversation(), &AddOptions::default())
        .await
        .expect("remember");
    assert_eq!(base_handle.call_count(), 1);
    assert_eq!(
        scoped_handle.call_count(),
        1,
        "the scoped model is not called for the other scope"
    );
}

/// An unregistered name fails loudly and lists what is registered.
#[tokio::test]
async fn an_unregistered_name_is_unsupported_and_lists_the_registered_names() {
    let (base, _queries) = CountingEmbedding::new("hash-embedding", BASE_DIMENSIONS);
    let ltm = LongTermMemory::builder(MemoryEngineConfig::default())
        .model(Arc::new(ScriptedModel::builder().build()))
        .embedding(base)
        .build()
        .expect("facade");

    let error = ScopeAnalyzerOverrideSpec::new()
        .with_model("gpt-4o")
        .resolve(&ltm.scope_override_handles())
        .expect_err("unregistered model");
    match error {
        Error::Unsupported(reason) => {
            assert!(reason.contains("gpt-4o"), "{reason}");
            assert!(reason.contains("scripted-model"), "{reason}");
            assert!(reason.contains("model_named"), "{reason}");
        }
        other => panic!("expected Error::Unsupported, got {other:?}"),
    }

    // Through the facade the same document is stored (only its shape is checked) and
    // fails every call that resolves it — never a silent fallback.
    ltm.set_scope_overrides(
        "ws",
        &ScopeAnalyzerOverrideSpec::new().with_embedding("nope"),
    )
    .await
    .expect("stored, not resolved");
    let error = ltm
        .resolve_scope_overrides(&scope("ws"))
        .await
        .expect_err("unregistered embedding");
    match error {
        Error::Unsupported(reason) => {
            assert!(reason.contains("nope"), "{reason}");
            assert!(
                reason.contains("registered embeddings: hash-embedding"),
                "{reason}"
            );
        }
        other => panic!("expected Error::Unsupported, got {other:?}"),
    }
    for error in [
        ltm.add(
            &scope("ws"),
            NewMemory::semantic("never stored"),
            &AddOptions::default(),
        )
        .await
        .expect_err("write refuses"),
        ltm.search(&scope("ws"), "never stored", &SearchOptions::default())
            .await
            .expect_err("recall refuses"),
    ] {
        assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
    }
    assert_eq!(ltm.list(&scope("ws"), &[]).await.expect("list").len(), 0);
}

/// A named per-scope embedding is the embedder of the scope's writes and recalls, and the
/// two dimensions coexist.
#[tokio::test]
async fn a_registered_per_scope_embedding_embeds_the_scope_writes_and_recalls() {
    let fixture = embedding_fixture();
    let ltm = &fixture.ltm;
    assert_eq!(
        fixture.ltm.scope_override_handles().embedding_names().len(),
        2
    );
    ltm.set_scope_overrides(
        "ws",
        &ScopeAnalyzerOverrideSpec::new().with_embedding(SCOPED_EMBEDDING),
    )
    .await
    .expect("store the override");

    // The write lands in the scope's own collection at the scoped dimension.
    ltm.add(
        &scope("ws"),
        NewMemory::semantic("tokio runtime task scheduling"),
        &AddOptions::default(),
    )
    .await
    .expect("add");
    assert_eq!(
        stored_dimension(&fixture.vectors, &scope("ws")).await,
        Some(SCOPED_DIMENSIONS),
        "the scope's vectors are written with its own embedder"
    );
    assert_eq!(fixture.scoped_queries.load(Ordering::SeqCst), 1);
    assert_eq!(
        fixture.base_queries.load(Ordering::SeqCst),
        0,
        "the engine embedder is not used for the overridden scope"
    );

    // The recall embeds the query with the same handle, so the vectors match.
    let hits = search(ltm, &scope("ws"), "tokio runtime").await;
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].record.content, "tokio runtime task scheduling");
    assert_eq!(fixture.scoped_queries.load(Ordering::SeqCst), 2);
    assert_eq!(fixture.base_queries.load(Ordering::SeqCst), 0);

    // A scope without an override keeps the engine embedder and its own dimension.
    ltm.add(
        &scope("plain"),
        NewMemory::semantic("tokio runtime task scheduling"),
        &AddOptions::default(),
    )
    .await
    .expect("add");
    assert_eq!(
        stored_dimension(&fixture.vectors, &scope("plain")).await,
        Some(BASE_DIMENSIONS)
    );
    assert_eq!(fixture.base_queries.load(Ordering::SeqCst), 1);
    let hits = search(ltm, &scope("plain"), "tokio runtime").await;
    assert_eq!(hits.len(), 1);
    assert_eq!(fixture.base_queries.load(Ordering::SeqCst), 2);
    assert_eq!(
        fixture.scoped_queries.load(Ordering::SeqCst),
        2,
        "the scoped embedder serves only its own scope"
    );
}

/// A per-scope embedding also drives the dedupe scan, the update re-embedding and the
/// variable projection of that scope.
#[tokio::test]
async fn a_per_scope_embedding_covers_dedupe_update_and_variables() {
    let fixture = embedding_fixture();
    let ltm = &fixture.ltm;
    ltm.set_scope_overrides(
        "ws",
        &ScopeAnalyzerOverrideSpec::new().with_embedding(SCOPED_EMBEDDING),
    )
    .await
    .expect("store the override");
    let first = ltm
        .add(
            &scope("ws"),
            NewMemory::semantic("Alice prefers dark mode"),
            &AddOptions::default(),
        )
        .await
        .expect("add");
    let id = first.added[0].id.clone();
    // The duplicate is found by the vector scan that embedded with the scoped handle.
    let scoped_before = fixture.scoped_queries.load(Ordering::SeqCst);
    let second = ltm
        .add(
            &scope("ws"),
            NewMemory::semantic("Alice prefers dark mode"),
            &AddOptions::default(),
        )
        .await
        .expect("add duplicate");
    assert_eq!(second.merged.len(), 1, "{second:?}");
    assert!(
        fixture.scoped_queries.load(Ordering::SeqCst) > scoped_before,
        "the dedupe scan embeds with the scoped handle"
    );
    assert_eq!(fixture.base_queries.load(Ordering::SeqCst), 0);

    // A content update re-embeds into the same collection.
    ltm.update(
        &scope("ws"),
        &id,
        openjiuwen_memory::record::MemoryUpdate::content("Alice prefers light mode"),
        &openjiuwen_memory::record::CallOptions::new(),
    )
    .await
    .expect("update");
    assert!(fixture.scoped_queries.load(Ordering::SeqCst) >= 2);
    assert_eq!(
        stored_dimension(&fixture.vectors, &scope("ws")).await,
        Some(SCOPED_DIMENSIONS)
    );

    // The variable projection of the scope is embedded with the same handle.
    ltm.set_variable(&scope("ws"), "theme", "dark")
        .await
        .expect("variable");
    let projection = keys::vector_collection(&scope("ws"), MemoryType::Variable);
    let schema = fixture
        .vectors
        .get_schema(&projection)
        .await
        .expect("schema")
        .expect("the variable projection is indexed");
    assert_eq!(
        schema
            .fields
            .iter()
            .find(|field| field.is_vector())
            .and_then(|field| field.dim),
        Some(SCOPED_DIMENSIONS)
    );
    assert_eq!(fixture.base_queries.load(Ordering::SeqCst), 0);
}

/// The analyzer pipeline a facade builds honors the resolved override model, not the base.
#[tokio::test]
async fn the_facade_pipeline_uses_the_resolved_override_model() {
    let base = ScriptedModel::builder()
        .model_name("scripted-model")
        .respond_text(FACTS_RESPONSE)
        .build();
    let scoped = ScriptedModel::builder()
        .model_name("scoped-model")
        .respond_text(FACTS_RESPONSE)
        .build();
    let base_handle = base.clone();
    let scoped_handle = scoped.clone();
    let ltm = LongTermMemory::builder(MemoryEngineConfig::default())
        .model(Arc::new(base))
        .model_named(SCOPED_MODEL, Arc::new(scoped))
        .build()
        .expect("facade");
    ltm.set_scope_overrides(
        "ws",
        &ScopeAnalyzerOverrideSpec::new()
            .with_extractor(ExtractorKind::Llm)
            .with_model(SCOPED_MODEL),
    )
    .await
    .expect("store the override");
    let overrides = ltm
        .resolve_scope_overrides(&scope("ws"))
        .await
        .expect("resolve");
    assert_eq!(
        overrides
            .model()
            .map(|model| openjiuwen_foundation::llm::Model::model_name(model.as_ref())),
        Some("scoped-model")
    );
    let staged = StagedAnalyzer::from_engine(ltm.config(), ltm.engine().model().cloned())
        .with_overrides(overrides);
    let result = staged
        .analyze(
            &conversation(),
            &openjiuwen_memory::record::CallOptions::new(),
        )
        .await
        .expect("analyze");
    assert_eq!(result.facts().len(), 1);
    assert_eq!(scoped_handle.call_count(), 1);
    assert_eq!(base_handle.call_count(), 0);
}

/// A blank registered name is a configuration error, not a silent no-op.
#[tokio::test]
async fn a_blank_registered_name_is_rejected_at_build_time() {
    let (base, _queries) = CountingEmbedding::new("hash-embedding", BASE_DIMENSIONS);
    for (name, error) in [
        (
            "  ",
            LongTermMemory::builder(MemoryEngineConfig::default())
                .model_named("  ", Arc::new(ScriptedModel::builder().build()))
                .build()
                .expect_err("blank model name"),
        ),
        (
            "",
            LongTermMemory::builder(MemoryEngineConfig::default())
                .embedding_named("", base)
                .build()
                .expect_err("blank embedding name"),
        ),
    ] {
        match error {
            Error::Config(reason) => assert!(reason.contains("non-blank name"), "{reason}"),
            other => panic!("expected Error::Config for `{name}`, got {other:?}"),
        }
    }
}

/// A scope with no override document resolves to empty overrides (the engine's handles).
#[tokio::test]
async fn an_absent_document_changes_nothing() {
    let fixture = embedding_fixture();
    let overrides = fixture
        .ltm
        .resolve_scope_overrides(&scope("ws"))
        .await
        .expect("resolve");
    assert!(overrides.is_empty());
    assert!(overrides.model().is_none());
    assert!(overrides.embedding().is_none());
    fixture
        .ltm
        .add(
            &scope("ws"),
            NewMemory::semantic("no override here"),
            &AddOptions::default(),
        )
        .await
        .expect("add");
    assert_eq!(
        stored_dimension(&fixture.vectors, &scope("ws")).await,
        Some(BASE_DIMENSIONS)
    );
    let hits = search(&fixture.ltm, &scope("ws"), "override").await;
    assert_eq!(hits.len(), 1);
    assert_eq!(fixture.base_queries.load(Ordering::SeqCst), 2);
    assert_eq!(fixture.scoped_queries.load(Ordering::SeqCst), 0);
}
