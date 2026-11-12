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

// The constructors mirror the Python `async` call sites even when they never await.
#![allow(clippy::unused_async_trait_impl)]

mod messages;
mod recall;
mod records;
mod variables;

use std::path::Path;
use std::sync::Arc;

use openjiuwen_common::error::Result;
use openjiuwen_foundation::llm::Model;
use openjiuwen_retrieval::embedding::Embedding;

use crate::config::MemoryEngineConfig;
use crate::engine::MemoryEngine;
use crate::graph::GraphMemory;
use crate::lite::CodingMemory;
use crate::store::MemoryStores;

/// The tracing target of facade diagnostics.
const TARGET: &str = "openjiuwen_memory::facade";

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
}

/// Builds a [`LongTermMemory`].
///
/// The configuration is mandatory; the stores default to
/// [`MemoryStores::in_memory`], and there is no embedding model and no LLM. Attach
/// them with [`LongTermMemoryBuilder::embedding`] and [`LongTermMemoryBuilder::model`];
/// the same instances are shared with graph memory.
pub struct LongTermMemoryBuilder {
    /// Engine configuration, validated by [`LongTermMemoryBuilder::build`].
    config: MemoryEngineConfig,
    /// Persistence backends (default: in-memory).
    stores: Option<MemoryStores>,
    /// Optional embedding model for semantic recall and dedupe.
    embedding: Option<Arc<dyn Embedding>>,
    /// Optional LLM for message-history extraction.
    model: Option<Arc<dyn Model>>,
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

    /// Validates the configuration and builds the facade.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::Config`] when the configuration (or
    /// one of its nested sections) is invalid.
    pub fn build(self) -> Result<Arc<LongTermMemory>> {
        let Self {
            config,
            stores,
            embedding,
            model,
        } = self;
        let stores = stores.unwrap_or_else(MemoryStores::in_memory);
        let mut builder = MemoryEngine::builder(config).stores(stores.clone());
        if let Some(embedding) = &embedding {
            builder = builder.embedding(Arc::clone(embedding));
        }
        if let Some(model) = &model {
            builder = builder.model(Arc::clone(model));
        }
        let engine = builder.build()?;
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
        }))
    }
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
}

impl std::fmt::Debug for LongTermMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LongTermMemory")
            .field("graph", &self.graph.is_some())
            .field("coding", &self.coding.is_some())
            .field("embedding", &self.engine.has_embedding())
            .field("model", &self.engine.model().is_some())
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
