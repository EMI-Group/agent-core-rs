//! The [`LongTermMemory`] public facade.
//!
//! [`LongTermMemory`] assembles one [`MemoryEngine`], the optional graph memory, and
//! the optional lite coding memory, and preserves the Python call-site vocabulary
//! (`add_messages`, `search_user_mem`, `update_mem_by_id`, variables, scope configs).
//! Every method here is wiring and argument adaptation: ranking, dedupe, and
//! persistence live in `crate::engine`, `crate::store`, and `crate::recall`.
//!
//! Construction goes through [`LongTermMemoryBuilder`], so the stores, the embedding
//! model, and the LLM are injected rather than read from process-global state:
//!
//! ```
//! use openjiuwen_memory::facade::LongTermMemory;
//! use openjiuwen_memory::config::MemoryEngineConfig;
//! use openjiuwen_memory::record::MemoryScope;
//!
//! # async fn example() -> openjiuwen_common::error::Result<()> {
//! let ltm = LongTermMemory::in_memory(MemoryEngineConfig::default()).await?;
//! let scope = MemoryScope::new("alice", "workspace");
//! ltm.add(&scope, openjiuwen_memory::record::NewMemory::semantic("likes tea"),
//!         &Default::default()).await?;
//! assert_eq!(ltm.list(&scope, &[]).await?.len(), 1);
//! # Ok(())
//! # }
//! ```
//!
//! - `records` — CRUD, listing, statistics, and the Python parity aliases.
//! - `recall` — `search`, `recall`, and the combined memory/graph `search_all`.
//! - `messages` — conversation history, extraction, and the heuristic fallback.
//! - `variables` — variables and per-scope configuration.
//! - `overrides` — per-scope analyzer/embedding overrides, persisted beside the scope
//!   configuration.

// The constructors mirror the Python `async` call sites even when they never await.
#![allow(clippy::unused_async_trait_impl)]

mod handles;
mod integrations;
mod messages;
mod overrides;
mod recall;
mod records;
mod variables;

pub use handles::ScopeOverrideHandles;
pub use overrides::ScopeAnalyzerOverrideSpec;
pub use variables::VariableAssignmentReport;

use std::path::Path;
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::Model;
use openjiuwen_retrieval::embedding::Embedding;

use crate::analyzer::{ExtractedFact, MemoryAnalyzer};
use crate::config::MemoryEngineConfig;
use crate::engine::MemoryEngine;
use crate::graph::GraphMemory;
use crate::lite::CodingMemory;
use crate::record::NewMemory;
use crate::store::MemoryStores;

/// The tracing target of facade diagnostics.
const TARGET: &str = "openjiuwen_memory::facade";

/// The crate-wide metadata key carrying a record's free-form tags (the `lite` convention).
pub(super) const TAGS_KEY: &str = "tags";

/// Converts an extracted fact into an engine write input, keeping its tags.
///
/// This is the facade's only `ExtractedFact` → [`NewMemory`] conversion, shared by
/// [`LongTermMemory::remember`] and the variable-assignment write paths.
/// [`ExtractedFact::to_new_memory`] already carries content, kind, importance, source,
/// and metadata. Tags are not a [`NewMemory`] field, so they go under the crate's
/// [`TAGS_KEY`] metadata key unless the fact supplies that key itself.
pub(super) fn record_input(fact: &ExtractedFact) -> NewMemory {
    let mut memory = fact.to_new_memory();
    if !fact.tags.is_empty() && !memory.metadata.contains_key(TAGS_KEY) {
        let tags = fact.tags.iter().cloned().map(serde_json::Value::String);
        memory.metadata.insert(
            TAGS_KEY.to_owned(),
            serde_json::Value::Array(tags.collect()),
        );
    }
    memory
}

/// The public entry point of long-term memory.
///
/// One instance per application (shared as an `Arc<LongTermMemory>`) owns the engine,
/// the optional graph memory, and the optional lite coding memory. The struct holds no
/// interior mutability of its own: every method takes `&self` and the injected stores
/// provide the synchronization, so clones of the `Arc` are cheap and safe to share
/// across tasks.
pub struct LongTermMemory {
    /// The write/recall engine shared by every record-level method.
    engine: Arc<MemoryEngine>,
    /// Entity/relation memory, present when graph memory is enabled and a graph store
    /// is injected.
    graph: Option<Arc<GraphMemory>>,
    /// Lite coding-memory facade, present when coding memory is enabled.
    coding: Option<CodingMemory>,
    /// Fact extraction, classification, and consolidation planner.
    analyzer: Arc<MemoryAnalyzer>,
    /// The named model/embedding handles a per-scope override can name.
    handles: ScopeOverrideHandles,
}

/// Builds a [`LongTermMemory`].
///
/// The configuration is mandatory; the stores default to
/// [`MemoryStores::in_memory`], and there is no embedding model and no LLM. Attach
/// them with [`LongTermMemoryBuilder::embedding`] and [`LongTermMemoryBuilder::model`];
/// the same instances are shared with graph memory.
///
/// The attached model and embedding register under their own `model_name()`, so a
/// scope override document can name them. Register *additional* handles with
/// [`LongTermMemoryBuilder::model_named`] and
/// [`LongTermMemoryBuilder::embedding_named`]; those are what make a per-scope model or
/// embedding swap possible.
pub struct LongTermMemoryBuilder {
    /// Engine configuration, validated by [`LongTermMemoryBuilder::build`].
    config: MemoryEngineConfig,
    /// Persistence backends (default: in-memory).
    stores: Option<MemoryStores>,
    /// Optional embedding model for semantic recall and dedupe.
    embedding: Option<Arc<dyn Embedding>>,
    /// Optional LLM for message-history extraction.
    model: Option<Arc<dyn Model>>,
    /// Optional analyzer override; built from the validated config when absent.
    analyzer: Option<Arc<MemoryAnalyzer>>,
    /// Extra model handles by registered name (per-scope overrides).
    named_models: Vec<(String, Arc<dyn Model>)>,
    /// Extra embedding handles by registered name (per-scope overrides).
    named_embeddings: Vec<(String, Arc<dyn Embedding>)>,
}

impl LongTermMemoryBuilder {
    /// Uses `stores` for every persistence operation (default: in-memory).
    #[must_use]
    pub fn stores(mut self, stores: MemoryStores) -> Self {
        self.stores = Some(stores);
        self
    }

    /// Attaches the embedding model used for semantic recall, dedupe, and graph
    /// entity vectors.
    #[must_use]
    pub fn embedding(mut self, embedding: Arc<dyn Embedding>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Attaches the LLM used for message-history extraction and graph extraction.
    #[must_use]
    pub fn model(mut self, model: Arc<dyn Model>) -> Self {
        self.model = Some(model);
        self
    }

    /// Registers `model` under `name` so a per-scope override document can name it.
    ///
    /// This is what makes a per-scope *model* swap possible: a
    /// [`ScopeAnalyzerOverrideSpec`] with `model = Some(name)` resolves to this handle
    /// instead of the engine's. The attached [`LongTermMemoryBuilder::model`] is
    /// registered under its own `model_name()`, so naming it keeps working; a duplicate
    /// name replaces the earlier registration (last wins).
    ///
    /// [`LongTermMemoryBuilder::build`] rejects a blank `name` with
    /// [`openjiuwen_common::error::Error::Config`]: a name no document can spell is not a
    /// resolution target.
    #[must_use]
    pub fn model_named(mut self, name: impl Into<String>, model: Arc<dyn Model>) -> Self {
        self.named_models.push((name.into(), model));
        self
    }

    /// Registers `embedding` under `name` so a per-scope override document can name it.
    ///
    /// This is what makes a per-scope *embedding* swap possible: a
    /// [`ScopeAnalyzerOverrideSpec`] with `embedding = Some(name)` resolves to this
    /// handle, and the write and recall paths of that scope then embed with it (each
    /// scope owns its own vector collections, so embedders of different dimensions
    /// coexist). The attached [`LongTermMemoryBuilder::embedding`] is registered under
    /// its own `model_name()`; a duplicate name replaces the earlier registration.
    ///
    /// [`LongTermMemoryBuilder::build`] rejects a blank `name` with
    /// [`openjiuwen_common::error::Error::Config`].
    #[must_use]
    pub fn embedding_named(
        mut self,
        name: impl Into<String>,
        embedding: Arc<dyn Embedding>,
    ) -> Self {
        self.named_embeddings.push((name.into(), embedding));
        self
    }

    /// Replaces the analyzer used by [`LongTermMemory::remember`](crate::facade::LongTermMemory::remember).
    ///
    /// The default is [`MemoryAnalyzer::from_engine`] over the validated configuration
    /// and the attached model, so an override only needs to be supplied when a caller
    /// wants a different extractor, classifier, or analyzer knobs.
    #[must_use]
    pub fn analyzer(mut self, analyzer: Arc<MemoryAnalyzer>) -> Self {
        self.analyzer = Some(analyzer);
        self
    }

    /// Validates the configuration and builds the facade.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::Config`] when the configuration (or
    /// one of its nested sections) is invalid, or when a name passed to
    /// [`LongTermMemoryBuilder::model_named`]/[`LongTermMemoryBuilder::embedding_named`]
    /// is blank.
    pub fn build(self) -> Result<Arc<LongTermMemory>> {
        let Self {
            config,
            stores,
            embedding,
            model,
            analyzer,
            named_models,
            named_embeddings,
        } = self;
        for (kind, name) in named_models
            .iter()
            .map(|(name, _)| ("model", name))
            .chain(named_embeddings.iter().map(|(name, _)| ("embedding", name)))
        {
            if name.trim().is_empty() {
                return Err(Error::config(format!(
                    "a scope override `{kind}` handle registered through LongTermMemoryBuilder \
                     must have a non-blank name"
                )));
            }
        }
        let handles = scope_handles(
            model.as_ref(),
            embedding.as_ref(),
            named_models,
            named_embeddings,
        );
        let stores = stores.unwrap_or_else(MemoryStores::in_memory);
        let mut builder = MemoryEngine::builder(config).stores(stores.clone());
        if let Some(embedding) = &embedding {
            builder = builder.embedding(Arc::clone(embedding));
        }
        if let Some(model) = &model {
            builder = builder.model(Arc::clone(model));
        }
        let engine = builder.build()?;
        let analyzer = analyzer.unwrap_or_else(|| {
            Arc::new(MemoryAnalyzer::from_engine(engine.config(), model.clone()))
        });
        let graph = build_graph(&engine, &stores, embedding.as_ref(), model.as_ref());
        let coding = engine
            .config()
            .coding
            .enabled
            .then(|| CodingMemory::new(Arc::clone(&engine)));
        Ok(Arc::new(LongTermMemory {
            engine,
            graph,
            coding,
            analyzer,
            handles,
        }))
    }
}

/// Builds the handle registry: the engine's own handles first, then the named ones.
///
/// The attachment order matters for [`ScopeOverrideHandles::model`]: the first handle
/// registered is the default a scope inherits when it names nothing.
fn scope_handles(
    model: Option<&Arc<dyn Model>>,
    embedding: Option<&Arc<dyn Embedding>>,
    named_models: Vec<(String, Arc<dyn Model>)>,
    named_embeddings: Vec<(String, Arc<dyn Embedding>)>,
) -> ScopeOverrideHandles {
    let mut handles = ScopeOverrideHandles::new();
    if let Some(model) = model {
        handles = handles.with_model(Arc::clone(model));
    }
    if let Some(embedding) = embedding {
        handles = handles.with_embedding(Arc::clone(embedding));
    }
    for (name, model) in named_models {
        handles = handles.with_named_model(name, model);
    }
    for (name, embedding) in named_embeddings {
        handles = handles.with_named_embedding(name, embedding);
    }
    handles
}

impl LongTermMemory {
    /// Starts a builder from `config`.
    #[must_use]
    pub fn builder(config: MemoryEngineConfig) -> LongTermMemoryBuilder {
        LongTermMemoryBuilder {
            config,
            stores: None,
            embedding: None,
            model: None,
            analyzer: None,
            named_models: Vec::new(),
            named_embeddings: Vec::new(),
        }
    }

    /// Builds a facade over ephemeral in-memory stores.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::Config`] when `config` is invalid.
    pub async fn in_memory(config: MemoryEngineConfig) -> Result<Arc<Self>> {
        Self::builder(config).build()
    }

    /// Builds a facade over file-backed stores rooted at `root`.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::Config`] when `config` is invalid and
    /// the storage error when the local-file stores cannot be created.
    pub async fn local_file(
        config: MemoryEngineConfig,
        root: impl AsRef<Path>,
    ) -> Result<Arc<Self>> {
        let stores = MemoryStores::local_file(root).await?;
        Self::builder(config).stores(stores).build()
    }

    /// Builds a facade over SQLite-backed stores at `path`.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::Config`] when `config` is invalid and
    /// the storage error when the database cannot be opened.
    #[cfg(feature = "sqlite")]
    pub async fn sqlite(config: MemoryEngineConfig, path: impl AsRef<Path>) -> Result<Arc<Self>> {
        let stores = MemoryStores::sqlite(path).await?;
        Self::builder(config).stores(stores).build()
    }

    /// Builds a facade over explicit `stores`.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::Config`] when `config` is invalid.
    pub async fn with_stores(
        config: MemoryEngineConfig,
        stores: MemoryStores,
    ) -> Result<Arc<Self>> {
        Self::builder(config).stores(stores).build()
    }

    /// The shared write/recall engine.
    #[must_use]
    pub fn engine(&self) -> &Arc<MemoryEngine> {
        &self.engine
    }

    /// The validated engine configuration.
    #[must_use]
    pub fn config(&self) -> &MemoryEngineConfig {
        self.engine.config()
    }

    /// Graph memory, when it is enabled and a graph store is injected.
    #[must_use]
    pub fn graph(&self) -> Option<&Arc<GraphMemory>> {
        self.graph.as_ref()
    }

    /// Lite coding memory, when it is enabled.
    #[must_use]
    pub fn coding(&self) -> Option<&CodingMemory> {
        self.coding.as_ref()
    }

    /// The named model/embedding handles a per-scope override can name.
    ///
    /// A registry is immutable once built, so this is a snapshot of what the facade was
    /// constructed with: the engine's own handles under their `model_name()` plus every
    /// [`LongTermMemoryBuilder::model_named`]/[`LongTermMemoryBuilder::embedding_named`]
    /// registration.
    #[must_use]
    pub fn scope_override_handles(&self) -> ScopeOverrideHandles {
        self.handles.clone()
    }
}

impl std::fmt::Debug for LongTermMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LongTermMemory")
            .field("graph", &self.graph.is_some())
            .field("coding", &self.coding.is_some())
            .field("embedding", &self.engine.has_embedding())
            .field("model", &self.engine.model().is_some())
            .field("analyzer", &self.analyzer.extractor_name())
            .field("handles", &self.handles)
            .finish()
    }
}

/// Builds graph memory when the config enables it and the stores carry a graph backend.
fn build_graph(
    engine: &MemoryEngine,
    stores: &MemoryStores,
    embedding: Option<&Arc<dyn Embedding>>,
    model: Option<&Arc<dyn Model>>,
) -> Option<Arc<GraphMemory>> {
    let config = &engine.config().graph;
    if !config.enabled {
        return None;
    }
    let store = stores.graph.clone()?;
    let mut graph = GraphMemory::new(store, config.clone());
    if let Some(embedding) = embedding {
        graph = graph.with_embedding(Arc::clone(embedding));
    }
    if let Some(model) = model {
        graph = graph.with_model(Arc::clone(model));
    }
    Some(Arc::new(graph))
}

#[cfg(test)]
mod tests {
    use super::*;

    use openjiuwen_common::error::Error;
    use openjiuwen_foundation::store::{
        InMemoryDocumentStore, InMemoryKvStore, InMemoryVectorStore,
    };
    use openjiuwen_retrieval::embedding::HashEmbedding;
    use openjiuwen_test_support::ScriptedModel;

    fn assert_send_sync_static<T: Send + Sync + 'static>() {}

    fn graph_enabled() -> MemoryEngineConfig {
        let mut config = MemoryEngineConfig::default();
        config.graph.enabled = true;
        config
    }

    #[test]
    fn the_facade_is_send_sync_and_static() {
        assert_send_sync_static::<LongTermMemory>();
        assert_send_sync_static::<Arc<LongTermMemory>>();
    }

    #[test]
    fn the_default_builder_has_no_graph_and_enables_coding() {
        let ltm = LongTermMemory::builder(MemoryEngineConfig::default())
            .build()
            .expect("build");
        assert!(ltm.graph().is_none());
        assert!(ltm.coding().is_some());
        assert!(!ltm.engine().has_embedding());
        assert!(ltm.engine().model().is_none());
        assert_eq!(ltm.config().default_top_k, 5);
        assert!(std::ptr::eq(ltm.config(), ltm.engine().config()));
    }

    #[tokio::test]
    async fn enabling_graph_wires_graph_memory() {
        let ltm = LongTermMemory::builder(graph_enabled())
            .build()
            .expect("build");
        let graph = ltm.graph().expect("graph enabled");
        assert_eq!(graph.config(), &ltm.config().graph);
    }

    #[tokio::test]
    async fn missing_graph_store_leaves_graph_memory_absent() {
        let stores = MemoryStores::from_parts(
            Arc::new(InMemoryKvStore::new()),
            Arc::new(InMemoryVectorStore::new()),
            Arc::new(InMemoryDocumentStore::new()),
        );
        let ltm = LongTermMemory::builder(graph_enabled())
            .stores(stores)
            .build()
            .expect("build");
        assert!(ltm.graph().is_none());
    }

    #[tokio::test]
    async fn builder_injects_the_embedding_and_the_model() {
        let scripted = Arc::new(ScriptedModel::text("{}"));
        let ltm = LongTermMemory::builder(graph_enabled())
            .embedding(Arc::new(HashEmbedding::new(8)))
            .model(scripted)
            .build()
            .expect("build");
        assert!(ltm.engine().has_embedding());
        assert!(ltm.engine().model().is_some());
        // Graph memory shares the engine's embedding and model.
        assert!(ltm.graph().is_some());
    }

    #[tokio::test]
    async fn constructors_validate_the_config() {
        let config = MemoryEngineConfig {
            default_top_k: 0,
            ..Default::default()
        };
        let error = LongTermMemory::in_memory(config)
            .await
            .expect_err("invalid config");
        assert!(matches!(error, Error::Config(_)), "{error:?}");
    }

    #[tokio::test]
    async fn local_file_and_with_stores_construct_a_facade() {
        let directory = tempfile::tempdir().expect("tempdir");
        let ltm = LongTermMemory::local_file(MemoryEngineConfig::default(), directory.path())
            .await
            .expect("local file");
        assert!(ltm.graph().is_none());
        let ltm =
            LongTermMemory::with_stores(MemoryEngineConfig::default(), MemoryStores::in_memory())
                .await
                .expect("with stores");
        assert!(ltm.coding().is_some());
    }
}
