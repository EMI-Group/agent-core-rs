//! [`EngineProvider`]: the [`MemoryProvider`] adapter over the in-process engine.
//!
//! The adapter is the reference implementation of the SPI: it proves the trait against
//! real storage, ranking, and dedupe code instead of a stub, and it is what
//! [`ProviderRegistry::with_local`](crate::external::ProviderRegistry::with_local)
//! registers under the name `local` (the Rust analogue of Python's in-process
//! `OpenJiuwenMemoryProvider`).
//!
//! Every call forwards to [`crate::engine::MemoryEngine`] unchanged, so scope validation, dedupe,
//! ranking, and cancellation keep exactly the semantics documented by the engine.
//! `get`/`delete`/`list`/`clear` take no per-call options in the SPI; they use the
//! engine's default (never-cancelled, no deadline) call options and are cancel-safe
//! through future dropping.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;

use openjiuwen_common::error::{Error, Result};

use crate::engine::MemoryEngine;
use crate::record::{
    AddOptions, AddOutcome, CallOptions, MemoryFilter, MemoryRecord, MemoryScope, MemoryUpdate,
    NewMemory, RecallHit, SearchOptions,
};

use super::provider::{MemoryProvider, ProviderCapabilities, ProviderHealth};
use super::registry::{ProviderConfig, ProviderFactory, ProviderKind};

/// The tracing target of local-provider diagnostics.
const TARGET: &str = "openjiuwen_memory::external";

/// The registry name of the local provider.
const DEFAULT_NAME: &str = "local";

/// The sentinel user id of the health probe.
const HEALTH_USER: &str = "__health__";

/// The sentinel scope id of the health probe.
const HEALTH_SCOPE: &str = "__health__";

/// The [`MemoryProvider`] implementation over [`MemoryEngine`].
///
/// The provider shares the engine with the rest of the application (the facade, agent
/// rails, workflow nodes), so a record written through the provider is visible through
/// `LongTermMemory` and vice versa.
///
/// ```
/// use std::sync::Arc;
///
/// use openjiuwen_memory::config::MemoryEngineConfig;
/// use openjiuwen_memory::engine::MemoryEngine;
/// use openjiuwen_memory::external::{EngineProvider, MemoryProvider};
/// use openjiuwen_memory::store::MemoryStores;
///
/// # async fn example() -> openjiuwen_common::error::Result<()> {
/// let engine = MemoryEngine::new(MemoryEngineConfig::default(), MemoryStores::in_memory())?;
/// let provider = EngineProvider::new(Arc::clone(&engine));
/// provider.initialize().await?;
/// assert_eq!(provider.name(), "local");
/// # Ok(())
/// # }
/// ```
pub struct EngineProvider {
    /// Registry name of this instance.
    name: String,
    /// The shared write/recall engine.
    engine: Arc<MemoryEngine>,
    /// Capabilities computed from the engine's configuration.
    capabilities: ProviderCapabilities,
    /// Whether [`MemoryProvider::initialize`] has run; the engine needs no setup.
    initialized: AtomicBool,
}

impl EngineProvider {
    /// Builds the provider named `local` over `engine`.
    #[must_use]
    pub fn new(engine: Arc<MemoryEngine>) -> Self {
        Self::with_name(engine, DEFAULT_NAME)
    }

    /// Builds the provider named `name` (a blank name falls back to `local`).
    ///
    /// The name is what [`ProviderRegistry::register`](crate::external::ProviderRegistry::register)
    /// keys the instance by, so two engines can be served under distinct names.
    #[must_use]
    pub fn with_name(engine: Arc<MemoryEngine>, name: impl Into<String>) -> Self {
        let name = name.into();
        let name = if name.trim().is_empty() {
            DEFAULT_NAME.to_owned()
        } else {
            name
        };
        let capabilities = ProviderCapabilities::local(engine.has_embedding());
        Self {
            name,
            engine,
            capabilities,
            initialized: AtomicBool::new(false),
        }
    }

    /// The shared engine behind this provider.
    #[must_use]
    pub fn engine(&self) -> &Arc<MemoryEngine> {
        &self.engine
    }

    /// Returns `true` once [`MemoryProvider::initialize`] has run.
    #[must_use]
    pub fn is_initialized(&self) -> bool {
        self.initialized.load(Ordering::Relaxed)
    }
}

impl fmt::Debug for EngineProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EngineProvider")
            .field("name", &self.name)
            .field("initialized", &self.is_initialized())
            .field("capabilities", &self.capabilities)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl MemoryProvider for EngineProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.capabilities.clone()
    }

    async fn initialize(&self) -> Result<()> {
        // The engine is fully constructed; initialization only records the state, so a
        // repeated call is a no-op instead of re-running backend setup.
        self.initialized.store(true, Ordering::Relaxed);
        Ok(())
    }

    async fn add(
        &self,
        scope: &MemoryScope,
        memories: Vec<NewMemory>,
        options: &AddOptions,
    ) -> Result<AddOutcome> {
        self.engine.add_many(scope, memories, options).await
    }

    async fn search(
        &self,
        scope: &MemoryScope,
        query: &str,
        options: &SearchOptions,
    ) -> Result<Vec<RecallHit>> {
        self.engine.search(scope, query, options).await
    }

    async fn get(&self, scope: &MemoryScope, id: &str) -> Result<Option<MemoryRecord>> {
        self.engine.get(scope, id).await
    }

    async fn update(
        &self,
        scope: &MemoryScope,
        id: &str,
        update: MemoryUpdate,
        options: &CallOptions,
    ) -> Result<MemoryRecord> {
        self.engine.update(scope, id, update, options).await
    }

    async fn delete(&self, scope: &MemoryScope, id: &str) -> Result<bool> {
        self.engine.delete(scope, id).await
    }

    async fn list(&self, filter: &MemoryFilter) -> Result<Vec<MemoryRecord>> {
        self.engine.list(filter).await
    }

    async fn clear(&self, filter: &MemoryFilter) -> Result<usize> {
        self.engine.clear(filter, &CallOptions::default()).await
    }

    async fn health(&self) -> Result<ProviderHealth> {
        // A read on a sentinel scope: no writes, no embedding call, and an I/O error from
        // the backend surfaces as a degraded report instead of failing the caller.
        let probe = MemoryFilter::for_scope(&MemoryScope::new(HEALTH_USER, HEALTH_SCOPE));
        match self.engine.list(&probe).await {
            Ok(_) => Ok(ProviderHealth::healthy().with_detail("in-process engine reachable")),
            Err(error) => {
                tracing::warn!(
                    target: TARGET,
                    %error,
                    provider = %self.name,
                    "local memory health probe failed"
                );
                Ok(ProviderHealth::degraded(error.to_string()))
            }
        }
    }
}

/// Builds [`EngineProvider`] instances over one shared engine.
///
/// The factory makes the local engine reachable through
/// [`ProviderRegistry::build`](crate::external::ProviderRegistry::build) and
/// [`ProviderRegistry::select`](crate::external::ProviderRegistry::select) exactly like a
/// vendor backend, so switching a deployment from `local` to a remote provider changes
/// only the configuration.
pub struct LocalProviderFactory {
    /// The engine every built provider shares.
    engine: Arc<MemoryEngine>,
    /// The fallback name when the configuration does not set one.
    name: String,
}

impl LocalProviderFactory {
    /// Builds the factory described by `engine`, using the `local` name.
    #[must_use]
    pub fn new(engine: Arc<MemoryEngine>) -> Self {
        Self {
            engine,
            name: DEFAULT_NAME.to_owned(),
        }
    }

    /// Overrides the fallback name of the providers this factory builds.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        let name = name.into();
        if !name.trim().is_empty() {
            self.name = name;
        }
        self
    }

    /// The engine every built provider shares.
    #[must_use]
    pub fn engine(&self) -> &Arc<MemoryEngine> {
        &self.engine
    }
}

impl fmt::Debug for LocalProviderFactory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalProviderFactory")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl ProviderFactory for LocalProviderFactory {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Local
    }

    fn build(&self, config: &ProviderConfig) -> Result<Arc<dyn MemoryProvider>> {
        if config.kind != ProviderKind::Local {
            return Err(Error::validation(format!(
                "the local provider factory cannot build kind `{}`",
                config.kind
            )));
        }
        let name = if config.name.trim().is_empty() {
            self.name.clone()
        } else {
            config.name.clone()
        };
        Ok(Arc::new(EngineProvider::with_name(
            Arc::clone(&self.engine),
            name,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::config::MemoryEngineConfig;
    use crate::facade::LongTermMemory;
    use crate::record::{MemoryType, SearchMode};

    use openjiuwen_retrieval::embedding::HashEmbedding;

    fn scope() -> MemoryScope {
        MemoryScope::new("alice", "ws")
    }

    async fn local_provider() -> EngineProvider {
        let memory = LongTermMemory::in_memory(MemoryEngineConfig::default())
            .await
            .expect("facade");
        EngineProvider::new(Arc::clone(memory.engine()))
    }

    #[tokio::test]
    async fn round_trip_through_the_local_engine() {
        let provider = local_provider().await;
        provider.initialize().await.expect("initialize");
        provider.initialize().await.expect("idempotent initialize");

        let outcome = provider
            .add(
                &scope(),
                vec![NewMemory::semantic("alice prefers dark mode")],
                &AddOptions::default(),
            )
            .await
            .expect("add");
        assert_eq!(outcome.added.len(), 1);
        let id = outcome.added[0].id.clone();

        let hits = provider
            .search(
                &scope(),
                "dark mode",
                &SearchOptions::default().with_threshold(0.0),
            )
            .await
            .expect("search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].record.id, id);
        assert_eq!(hits[0].record.content, "alice prefers dark mode");

        let read = provider.get(&scope(), &id).await.expect("get");
        assert_eq!(read.expect("record").id, id);

        let updated = provider
            .update(
                &scope(),
                &id,
                MemoryUpdate::content("alice prefers light mode"),
                &CallOptions::default(),
            )
            .await
            .expect("update");
        assert_eq!(updated.content, "alice prefers light mode");

        let listed = provider
            .list(&MemoryFilter::for_scope(&scope()))
            .await
            .expect("list");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].content, "alice prefers light mode");

        assert!(provider.delete(&scope(), &id).await.expect("delete"));
        assert!(!provider.delete(&scope(), &id).await.expect("delete again"));
        assert!(provider.get(&scope(), &id).await.expect("get").is_none());

        provider
            .add(
                &scope(),
                vec![NewMemory::semantic("first"), NewMemory::episodic("second")],
                &AddOptions::default(),
            )
            .await
            .expect("add batch");
        let cleared = provider
            .clear(&MemoryFilter::for_scope(&scope()))
            .await
            .expect("clear");
        assert_eq!(cleared, 2);
        let remaining = provider
            .list(&MemoryFilter::for_scope(&scope()))
            .await
            .expect("list after clear");
        assert!(
            remaining.is_empty(),
            "records left after clear: {remaining:?}"
        );

        let health = provider.health().await.expect("health");
        assert!(health.healthy);
        assert!(health.detail.is_some());
        assert!(provider.is_initialized());
    }

    #[tokio::test]
    async fn the_provider_shares_state_with_the_facade() {
        let memory = LongTermMemory::in_memory(MemoryEngineConfig::default())
            .await
            .expect("facade");
        let provider = EngineProvider::new(Arc::clone(memory.engine()));
        provider
            .add(
                &scope(),
                vec![NewMemory::coding("uses rustfmt")],
                &AddOptions::default(),
            )
            .await
            .expect("add through the provider");
        let listed = memory.list(&scope(), &[]).await.expect("facade list");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].kind, MemoryType::Coding);
    }

    #[tokio::test]
    async fn capabilities_track_the_embedding_model() {
        let keyword_only = local_provider().await;
        assert!(!keyword_only.capabilities().semantic_search);
        assert!(keyword_only.capabilities().keyword_search);

        let memory = LongTermMemory::builder(MemoryEngineConfig::default())
            .embedding(Arc::new(HashEmbedding::new(8)))
            .build()
            .expect("facade");
        let semantic = EngineProvider::new(Arc::clone(memory.engine()));
        assert!(semantic.capabilities().semantic_search);

        // A keyword-only backend reports no semantic capability and refuses that mode.
        let error = keyword_only
            .search(
                &scope(),
                "dark mode",
                &SearchOptions::default().with_mode(SearchMode::Semantic),
            )
            .await
            .expect_err("semantic search without an embedding model");
        assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
    }

    #[tokio::test]
    async fn the_name_defaults_and_can_be_overridden() {
        let memory = LongTermMemory::in_memory(MemoryEngineConfig::default())
            .await
            .expect("facade");
        let default = EngineProvider::new(Arc::clone(memory.engine()));
        assert_eq!(default.name(), DEFAULT_NAME);
        let named = EngineProvider::with_name(Arc::clone(memory.engine()), "team-a");
        assert_eq!(named.name(), "team-a");
        let blank = EngineProvider::with_name(Arc::clone(memory.engine()), "   ");
        assert_eq!(blank.name(), DEFAULT_NAME);
        assert!(Arc::ptr_eq(default.engine(), memory.engine()));
    }

    #[tokio::test]
    async fn the_factory_builds_named_providers_over_the_shared_engine() {
        let memory = LongTermMemory::in_memory(MemoryEngineConfig::default())
            .await
            .expect("facade");
        let factory = LocalProviderFactory::new(Arc::clone(memory.engine()));
        assert_eq!(factory.kind(), ProviderKind::Local);
        assert!(Arc::ptr_eq(factory.engine(), memory.engine()));

        let provider = factory
            .build(&ProviderConfig::new(ProviderKind::Local))
            .expect("build");
        assert_eq!(provider.name(), DEFAULT_NAME);
        provider
            .add(
                &scope(),
                vec![NewMemory::semantic("from the factory")],
                &AddOptions::default(),
            )
            .await
            .expect("add");
        assert_eq!(memory.list(&scope(), &[]).await.expect("list").len(), 1);

        let named = factory
            .build(&ProviderConfig::new(ProviderKind::Local).with_name("secondary"))
            .expect("named build");
        assert_eq!(named.name(), "secondary");

        let error = factory
            .build(&ProviderConfig::new(ProviderKind::Mem0))
            .err()
            .expect("kind mismatch");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
    }

    #[tokio::test]
    async fn the_provider_never_leaks_the_engine_in_debug() {
        let provider = local_provider().await;
        let debug = format!("{provider:?}");
        assert!(debug.contains("EngineProvider"));
        assert!(debug.contains(DEFAULT_NAME));
        assert!(!debug.contains("MemoryEngine {"));
    }
}
