//! The async write and recall orchestration of long-term memory.
//!
//! [`MemoryEngine`] owns the engine state: the configuration, the persistence seam,
//! the optional embedding model, the optional LLM (kept for message-history
//! extraction wiring), the token counter, and the per-scope write locks.
//! The write path lives in `write`, the variable projection in `variables`, the read
//! path in `search`, statistics in `stats`, the lock registry in `locks`, and the
//! embedding helpers in `embedding`.
//!
//! Construction always goes through [`MemoryEngineBuilder`] (or the
//! [`MemoryEngine::new`] shorthand), because one engine instance is shared as an
//! `Arc<MemoryEngine>`: it is `Send + Sync + 'static` and every method takes `&self`.
//! Ranking is never implemented here — the engine assembles candidates and delegates
//! every formula to the `recall` module.

mod embedding;
mod locks;
mod search;
mod stats;
mod variables;
mod write;

use std::sync::Arc;

use openjiuwen_common::error::Result;
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::prompt::tokens::{ApproxTokenCounter, TokenCounter};
use openjiuwen_retrieval::embedding::Embedding;

use crate::config::MemoryEngineConfig;
use crate::store::{MemoryStorage, MemoryStores, MessageStore};

use locks::WriteLocks;
pub use variables::VariableWrite;

/// The write and recall orchestrator of long-term memory.
///
/// The engine is created through [`MemoryEngineBuilder`], shared as an
/// `Arc<MemoryEngine>`, and injected wherever memory is used (agent rails, workflow
/// memory components, the harness, and the [`crate::facade`] facade).
/// It owns no global state: every instance is isolated by its configuration and its
/// injected stores.
pub struct MemoryEngine {
    /// Engine configuration (validated at construction).
    config: MemoryEngineConfig,
    /// The injected store bundle, exposed to graph and message concerns.
    stores: MemoryStores,
    /// Typed record/embedding persistence over `stores`.
    storage: MemoryStorage,
    /// Optional embedding model; absent means keyword-only recall.
    embedding: Option<Arc<dyn Embedding>>,
    /// Optional LLM kept for message-history extraction.
    model: Option<Arc<dyn Model>>,
    /// Token counter used by statistics and the recall token budget.
    token_counter: Arc<dyn TokenCounter + Send + Sync>,
    /// Per-scope write serialization.
    write_locks: WriteLocks,
    /// Cached embedding dimension, initialized on first use.
    dimensions: tokio::sync::OnceCell<usize>,
}

/// Builds a [`MemoryEngine`].
///
/// Every component is optional except the configuration: stores default to
/// [`MemoryStores::in_memory`], there is no embedding model and no LLM, and the token
/// counter defaults to [`ApproxTokenCounter`].
pub struct MemoryEngineBuilder {
    /// Engine configuration, validated by [`MemoryEngineBuilder::build`].
    config: MemoryEngineConfig,
    /// Persistence backends (default: in-memory).
    stores: MemoryStores,
    /// Optional embedding model.
    embedding: Option<Arc<dyn Embedding>>,
    /// Optional LLM.
    model: Option<Arc<dyn Model>>,
    /// Token counter (default: [`ApproxTokenCounter`]).
    token_counter: Arc<dyn TokenCounter + Send + Sync>,
}

impl MemoryEngineBuilder {
    /// Starts a builder from `config` with in-memory stores and no embedding/LLM.
    #[must_use]
    pub fn new(config: MemoryEngineConfig) -> Self {
        Self {
            config,
            stores: MemoryStores::in_memory(),
            embedding: None,
            model: None,
            token_counter: Arc::new(ApproxTokenCounter::new()),
        }
    }

    /// Uses `stores` for every persistence operation (default: in-memory).
    #[must_use]
    pub fn stores(mut self, stores: MemoryStores) -> Self {
        self.stores = stores;
        self
    }

    /// Attaches the embedding model used for semantic recall and dedupe.
    #[must_use]
    pub fn embedding(mut self, embedding: Arc<dyn Embedding>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Attaches the LLM kept for message-history extraction.
    #[must_use]
    pub fn model(mut self, model: Arc<dyn Model>) -> Self {
        self.model = Some(model);
        self
    }

    /// Overrides the token counter (default: [`ApproxTokenCounter`]).
    #[must_use]
    pub fn token_counter(mut self, counter: Arc<dyn TokenCounter + Send + Sync>) -> Self {
        self.token_counter = counter;
        self
    }

    /// Validates the configuration and builds the engine.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::Config`] when the configuration or
    /// one of its nested sections is invalid.
    pub fn build(self) -> Result<Arc<MemoryEngine>> {
        self.config.validate()?;
        let storage = MemoryStorage::new(self.stores.clone());
        Ok(Arc::new(MemoryEngine {
            config: self.config,
            stores: self.stores,
            storage,
            embedding: self.embedding,
            model: self.model,
            token_counter: self.token_counter,
            write_locks: WriteLocks::default(),
            dimensions: tokio::sync::OnceCell::new(),
        }))
    }
}

impl MemoryEngine {
    /// Starts a builder from `config`.
    #[must_use]
    pub fn builder(config: MemoryEngineConfig) -> MemoryEngineBuilder {
        MemoryEngineBuilder::new(config)
    }

    /// Builds an engine over `stores` without an embedding model or an LLM.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::Config`] when `config` is invalid.
    pub fn new(config: MemoryEngineConfig, stores: MemoryStores) -> Result<Arc<Self>> {
        Self::builder(config).stores(stores).build()
    }

    /// The validated engine configuration.
    #[must_use]
    pub fn config(&self) -> &MemoryEngineConfig {
        &self.config
    }

    /// The injected store bundle.
    #[must_use]
    pub fn stores(&self) -> &MemoryStores {
        &self.stores
    }

    /// The embedding model, when one is attached.
    #[must_use]
    pub fn embedding(&self) -> Option<&Arc<dyn Embedding>> {
        self.embedding.as_ref()
    }

    /// The LLM attached for message extraction, when present.
    #[must_use]
    pub fn model(&self) -> Option<&Arc<dyn Model>> {
        self.model.as_ref()
    }

    /// Returns `true` when an embedding model is attached.
    #[must_use]
    pub fn has_embedding(&self) -> bool {
        self.embedding.is_some()
    }

    /// The typed record/embedding persistence seam, for the facade and graph memory.
    pub(crate) fn storage(&self) -> &MemoryStorage {
        &self.storage
    }

    /// A message store over the engine's document backend, for the facade.
    pub(crate) fn message_store(&self) -> MessageStore {
        MessageStore::new(self.stores.document.clone())
    }

    /// The token counter used by statistics and the recall token budget.
    pub(crate) fn token_counter(&self) -> &(dyn TokenCounter + Send + Sync) {
        self.token_counter.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send_sync_static<T: Send + Sync + 'static>() {}

    #[test]
    fn the_engine_is_send_sync_and_exposes_its_components() {
        assert_send_sync_static::<MemoryEngine>();
        let engine = MemoryEngine::new(MemoryEngineConfig::default(), MemoryStores::in_memory())
            .expect("engine");
        assert_eq!(engine.config().default_top_k, 5);
        assert!(!engine.has_embedding());
        assert!(engine.embedding().is_none());
        assert!(engine.model().is_none());
        assert!(engine.stores().graph.is_some());
        assert!(engine.token_counter().count_text("hello world") > 0);
    }

    #[test]
    fn an_invalid_config_is_rejected_at_build_time() {
        let config = MemoryEngineConfig {
            default_top_k: 0,
            ..Default::default()
        };
        let result = MemoryEngine::new(config, MemoryStores::in_memory());
        assert!(
            matches!(result, Err(openjiuwen_common::error::Error::Config(_))),
            "invalid config must be rejected"
        );
    }
}
