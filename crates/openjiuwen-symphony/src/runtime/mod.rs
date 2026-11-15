//! The symphony composition root: configuration, lifecycle, graph build, plan, invoke,
//! mutations, and evolution feedback.
//!
//! [`SymphonyRuntime`](crate::runtime::SymphonyRuntime) wires the crate's parts into one
//! engine: a [`CapabilityRegistry`](crate::registry::CapabilityRegistry) catalog, a
//! [`GraphArtifactStore`](crate::artifact::GraphArtifactStore) of immutable
//! [`CapabilityGraph`](crate::graph::CapabilityGraph) revisions, a
//! [`Planner`](crate::orchestration::planner::Planner), a
//! [`CapabilityRunner`](crate::orchestration::executor::CapabilityRunner), and an optional
//! [`SemanticIndex`](crate::search::semantic::SemanticIndex). Python's
//! engine is host-driven and owns no composition root, so this type is the Rust port's.
//!
//! # Lifecycle
//!
//! ```text
//! Created --start--> Running --stop/close--> Stopped --start--> Running
//! ```
//!
//! [`SymphonyRuntime::start`](crate::runtime::SymphonyRuntime::start) publishes the first
//! revision when the store holds none,
//! [`SymphonyRuntime::stop`](crate::runtime::SymphonyRuntime::stop) stops the engine, and
//! [`SymphonyRuntime::close`](crate::runtime::SymphonyRuntime::close) stops it and
//! re-reads the current revision as a consistency check (the flush a caller awaits before
//! dropping the runtime). `start` and `stop` are idempotent.
//!
//! # State discipline
//!
//! [`SymphonyRuntime::invoke`](crate::runtime::SymphonyRuntime::invoke) and the capability
//! mutations require
//! [`RuntimeState::Running`](crate::runtime::RuntimeState::Running); in
//! [`RuntimeState::Created`](crate::runtime::RuntimeState::Created) or
//! [`RuntimeState::Stopped`](crate::runtime::RuntimeState::Stopped)
//! they fail with an [`Error::Validation`] naming `runtime.state`. Everything else —
//! `plan`, `graph`, `graph_status`, `read_graph`, `search`, `semantic_search`, `build`,
//! `submit_evolution`, `capture_graph_snapshot`, and the read-only accessors — works in
//! every state, so a stopped engine can still be inspected, planned against, and rebuilt.
//!
//! # Graph mutations
//!
//! [`SymphonyRuntime::mutate`](crate::runtime::SymphonyRuntime::mutate) takes a
//! [`MutationRequest`](crate::orchestration::mutation::MutationRequest) — the Rust port of
//! Python's mutation contract — plus the payload to apply, and the three positional
//! methods ([`add_capabilities`](crate::runtime::SymphonyRuntime::add_capabilities),
//! [`update_capabilities`](crate::runtime::SymphonyRuntime::update_capabilities),
//! [`remove_capabilities`](crate::runtime::SymphonyRuntime::remove_capabilities)) are thin
//! delegations that build such a request without a `source_revision`.
//!
//! - **Idempotency.** Every request carries a caller `request_id` and a
//!   [digest](crate::orchestration::mutation::request_digest) covering its operation, id,
//!   optional source revision, and sorted id batch. A repeated `request_id` answers the
//!   recorded result with [`MutationStatus::Replayed`](crate::runtime::MutationStatus::Replayed) instead of applying the batch twice,
//!   and reusing it for a *different* batch is [`Error::Validation`] naming `request_id`.
//! - **Durability.** With a journal configured — [`SymphonyRuntimeBuilder::with_mutation_journal`](crate::runtime::SymphonyRuntimeBuilder::with_mutation_journal)
//!   or [`SymphonyConfig::mutation_journal_path`](crate::runtime::SymphonyConfig::mutation_journal_path) — every recorded result is written
//!   through atomically, so a rebuilt runtime replays it instead of re-applying it. The
//!   default is an in-memory ledger only.
//! - **Compare-and-swap.** A request that declares
//!   [`source_revision`](crate::orchestration::mutation::MutationRequest::source_revision)
//!   must name the current artifact revision; otherwise the batch is stale and rejected
//!   with [`Error::Validation`] naming `source_revision` before anything is published.
//!
//! # Concurrency
//!
//! The lifecycle state and the cached graph pointer live behind a [`std::sync::RwLock`]
//! held only for a single copy or clone — never across an `.await` — with poisoned locks
//! recovered through [`std::sync::PoisonError::into_inner`]. The derive-and-publish
//! critical section (`build`, the mutations, `submit_evolution`) is serialized by a
//! [`tokio::sync::Mutex`], an async-aware lock that legitimately spans the `.await`s
//! persisting a revision; holding it is what makes mutation replay and the edge diff
//! meaningful. The mutation ledger (in-memory records plus the optional journal) lives
//! inside that mutex, so its journal writes are serialized with the publication they record.
//!
//! # Graph observation
//!
//! When [`SymphonyConfig::graph_artifact_root`] names a directory, the runtime also owns a
//! [`GraphObservationService`](crate::observation::service::GraphObservationService) rooted
//! beside the graph revisions: [`SymphonyRuntime::submit_observation`] appends caller-supplied
//! evidence and derives the merged overlay, [`SymphonyRuntime::observation_snapshot`] reads it,
//! and [`SymphonyRuntime::flush_observations`] reports a deferred projection failure. The
//! append-only evidence log needs a directory, so an in-memory runtime has no observation
//! service and all three methods return
//! [`Error::Unsupported`](crate::error::Error::Unsupported).
//!
//! A second [`tokio::sync::Mutex`] (`semantic_sync`) serializes semantic (re)indexing. It
//! may span the registry snapshot read plus the embedding calls — that serialization is
//! intentional, because a concurrent query must not clear the index another query is
//! building — and it is released before the vector search and every unrelated `.await`.
//! The mutex records the
//! [`CapabilityRegistry::revision`](crate::registry::CapabilityRegistry::revision) the
//! index reflects, so an
//! unchanged catalog embeds only the query; a stale record just costs one extra reindex,
//! never wrong results.

mod mutations;

pub use crate::orchestration::mutation::{
    request_digest, validate_mutation_against_inventory, JournalRecord, MutationJournal,
    MutationOperation, MutationRequest,
};
use mutations::MutationLedger;
pub use mutations::{GraphMutationResult, MutationStatus};

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::PathBuf;
use std::sync::{Arc, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::Mutex as AsyncMutex;
use tracing::{debug, info, warn};

use openjiuwen_session::Session;

use crate::artifact::{
    FileGraphArtifactStore, GraphArtifactStatus, GraphArtifactStore, GraphRevision,
    MemoryGraphArtifactStore,
};
use crate::error::{unsupported, validation, Error, Result};
use crate::graph::CapabilityGraph;
use crate::model::{CapabilityDescriptor, CapabilityKey};
use crate::observation::contracts::{GraphEvolutionInput, GraphSnapshot, ObservationReceipt};
use crate::observation::service::GraphObservationService;
use crate::orchestration::executor::StepOutcome;
use crate::orchestration::{
    CandidateConfig, CandidateGenerator, CapabilityRunner, ExecutorConfig, OrchestrationPlan,
    PlanConfig, PlanExecution, PlanExecutor, PlanRequest, PlanStatus, Planner, UnsupportedRunner,
};
use crate::registry::CapabilityRegistry;
use crate::search::{SearchHit, SearchRequest, SemanticIndex};
use crate::store::AssetStore;

/// The graph scope a runtime publishes into when its configuration does not name one.
const DEFAULT_GRAPH_SCOPE_ID: &str = "default";

/// The lifecycle state of a [`SymphonyRuntime`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeState {
    /// Built by [`SymphonyRuntime::open`], started by nothing yet.
    #[default]
    Created,
    /// Serving `invoke` and capability mutations.
    Running,
    /// Stopped by [`SymphonyRuntime::stop`] or [`SymphonyRuntime::close`].
    Stopped,
}

/// Everything a [`SymphonyRuntime`] is opened with; every field has a usable default, so
/// `SymphonyConfig::default()` opens an in-memory engine on the `"default"` graph scope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SymphonyConfig {
    /// Root directory for file-backed graph revisions; `None` keeps them in memory.
    pub graph_artifact_root: Option<PathBuf>,
    /// The `graph_id` every published revision carries.
    pub graph_scope_id: String,
    /// The bounded planner's tuning.
    pub plan: PlanConfig,
    /// The plan executor's limits and failure handling.
    pub executor: ExecutorConfig,
    /// The deterministic `can_feed` candidate rules.
    pub candidates: CandidateConfig,
    /// Path of the durable mutation-idempotency journal; `None` keeps it in memory.
    ///
    /// When set, every recorded mutation result is written through to this JSON file, so a
    /// rebuilt runtime replays a repeated `request_id` instead of applying the batch again.
    /// [`SymphonyRuntimeBuilder::with_mutation_journal`] takes precedence over this path.
    pub mutation_journal_path: Option<PathBuf>,
}

impl Default for SymphonyConfig {
    /// Returns in-memory artifacts, the `"default"` graph scope, the default planner,
    /// executor, and candidate configurations, and an in-memory mutation journal.
    fn default() -> Self {
        Self {
            graph_artifact_root: None,
            graph_scope_id: DEFAULT_GRAPH_SCOPE_ID.to_string(),
            plan: PlanConfig::default(),
            executor: ExecutorConfig::default(),
            candidates: CandidateConfig::default(),
            mutation_journal_path: None,
        }
    }
}

/// Assembles a [`SymphonyRuntime`] from explicit parts; every setter is optional and an
/// unset part falls back to its documented default (an empty registry, in-memory
/// artifacts, a lexical planner, [`UnsupportedRunner`], no semantic index, an in-memory
/// mutation journal).
pub struct SymphonyRuntimeBuilder {
    /// The engine configuration.
    config: SymphonyConfig,
    /// The supplied catalog, if any.
    registry: Option<Arc<CapabilityRegistry>>,
    /// The supplied asset store, used to hydrate a catalog when none is given.
    asset_store: Option<Arc<dyn AssetStore>>,
    /// The supplied revision store, if any.
    artifacts: Option<Arc<dyn GraphArtifactStore>>,
    /// The supplied planner, if any.
    planner: Option<Planner>,
    /// The supplied capability runner, if any.
    runner: Option<Arc<dyn CapabilityRunner>>,
    /// The supplied semantic index, if any.
    semantic_index: Option<Arc<SemanticIndex>>,
    /// The supplied mutation journal, if any.
    journal: Option<MutationJournal>,
}

impl SymphonyRuntimeBuilder {
    /// Starts a builder from `config` with every optional part unset.
    #[must_use]
    pub fn new(config: SymphonyConfig) -> Self {
        Self {
            config,
            registry: None,
            asset_store: None,
            artifacts: None,
            planner: None,
            runner: None,
            semantic_index: None,
            journal: None,
        }
    }

    /// Uses `registry` as the authoritative catalog, ignoring any asset store.
    #[must_use]
    pub fn with_registry(mut self, registry: Arc<CapabilityRegistry>) -> Self {
        self.registry = Some(registry);
        self
    }

    /// Hydrates the catalog from `store` when no registry is supplied.
    ///
    /// The runtime becomes the store's single writer, so every later registration is
    /// persisted through it. Ignored when [`Self::with_registry`] is also called.
    #[must_use]
    pub fn with_asset_store(mut self, store: Arc<dyn AssetStore>) -> Self {
        self.asset_store = Some(store);
        self
    }

    /// Publishes revisions into `artifacts` instead of the configured backend.
    #[must_use]
    pub fn with_artifacts(mut self, artifacts: Arc<dyn GraphArtifactStore>) -> Self {
        self.artifacts = Some(artifacts);
        self
    }

    /// Invokes capabilities through `runner` instead of [`UnsupportedRunner`].
    #[must_use]
    pub fn with_runner(mut self, runner: Arc<dyn CapabilityRunner>) -> Self {
        self.runner = Some(runner);
        self
    }

    /// Plans with `planner` instead of `Planner::new(config.plan)`.
    #[must_use]
    pub fn with_planner(mut self, planner: Planner) -> Self {
        self.planner = Some(planner);
        self
    }

    /// Enables [`SymphonyRuntime::semantic_search`] with `index`.
    #[must_use]
    pub fn with_semantic_index(mut self, index: Arc<SemanticIndex>) -> Self {
        self.semantic_index = Some(index);
        self
    }

    /// Records mutation results in `journal` instead of an in-memory ledger.
    ///
    /// The journal is durable: a runtime rebuilt over the same file replays a repeated
    /// `request_id` with [`MutationStatus::Replayed`] instead of applying the batch again.
    /// Takes precedence over [`SymphonyConfig::mutation_journal_path`]; open the journal
    /// with [`MutationJournal::open`] to hydrate it from an existing file.
    #[must_use]
    pub fn with_mutation_journal(mut self, journal: MutationJournal) -> Self {
        self.journal = Some(journal);
        self
    }

    /// Assembles the runtime in [`RuntimeState::Created`], seeding the graph pointer from
    /// the store's current revision so a reopened file-backed root is reported as-is.
    ///
    /// # Errors
    ///
    /// Returns the asset store's hydration error, the artifact backend's open/status
    /// error, the journal's open error, or [`Error::Config`](crate::error::Error::Config) when
    /// [`SymphonyConfig::graph_artifact_root`] is unusable.
    pub async fn open(self) -> Result<SymphonyRuntime> {
        let artifacts: Arc<dyn GraphArtifactStore> = match self.artifacts {
            Some(artifacts) => artifacts,
            None => match self.config.graph_artifact_root.clone() {
                Some(root) => Arc::new(FileGraphArtifactStore::new(root).await?),
                None => Arc::new(MemoryGraphArtifactStore::new()),
            },
        };
        let registry = match self.registry {
            Some(registry) => registry,
            None => match self.asset_store {
                Some(store) => Arc::new(CapabilityRegistry::load(store).await?),
                None => Arc::new(CapabilityRegistry::new()),
            },
        };
        let planner = self
            .planner
            .unwrap_or_else(|| Planner::new(self.config.plan.clone()));
        let runner = self
            .runner
            .unwrap_or_else(|| Arc::new(UnsupportedRunner::new()) as Arc<dyn CapabilityRunner>);
        let journal = match self.journal {
            Some(journal) => Some(journal),
            None => match self.config.mutation_journal_path.clone() {
                Some(path) => Some(MutationJournal::open(path).await?),
                None => None,
            },
        };
        let status = artifacts.status().await?;
        // Borrow the root before `self.config` moves into the runtime. An in-memory artifact
        // store leaves the observation service unset: its evidence log is append-only on disk.
        let observation = self
            .config
            .graph_artifact_root
            .as_ref()
            .map(|root| GraphObservationService::new(Arc::clone(&artifacts), root));
        let runtime = SymphonyRuntime {
            config: self.config,
            state: RwLock::new(RuntimeState::Created),
            pointer: RwLock::new(GraphPointer::from_status(&status)),
            registry,
            artifacts,
            planner,
            runner,
            semantic: self.semantic_index,
            observation,
            // `None`: a `with_registry` catalog may already hold descriptors this index
            // has never seen, so the first search must synchronize.
            semantic_sync: AsyncMutex::new(None),
            ledger: AsyncMutex::new(MutationLedger::with_journal(journal)),
        };
        debug!(
            graph_scope_id = %runtime.config.graph_scope_id,
            revision = ?status.current_version,
            capabilities = runtime.registry.len(),
            "opened the symphony runtime"
        );
        Ok(runtime)
    }
}

impl fmt::Debug for SymphonyRuntimeBuilder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SymphonyRuntimeBuilder")
            .field("config", &self.config)
            .field(
                "supplied",
                &(
                    self.registry.is_some(),
                    self.asset_store.is_some(),
                    self.artifacts.is_some(),
                    self.planner.is_some(),
                    self.runner.is_some(),
                    self.semantic_index.is_some(),
                    self.journal.is_some(),
                ),
            )
            .finish()
    }
}

/// The engine: catalog, graph revisions, planner, runner, and lifecycle in one place.
///
/// See the module docs for the lifecycle, the state discipline, and the locking rules.
/// A runtime is shared as `Arc<SymphonyRuntime>`; every method takes `&self`.
pub struct SymphonyRuntime {
    /// The configuration captured at `open` time.
    config: SymphonyConfig,
    /// The lifecycle state; guarded by a `std::sync` lock never held across an `.await`.
    state: RwLock<RuntimeState>,
    /// The cached pointer to the current revision (`capture_graph_snapshot` is synchronous).
    pointer: RwLock<GraphPointer>,
    /// The authoritative capability catalog.
    registry: Arc<CapabilityRegistry>,
    /// The published graph revisions.
    artifacts: Arc<dyn GraphArtifactStore>,
    /// The bounded planner, configured from [`SymphonyConfig::plan`].
    planner: Planner,
    /// The capability invoker used by [`SymphonyRuntime::invoke`].
    runner: Arc<dyn CapabilityRunner>,
    /// The optional embedding-backed index used by [`SymphonyRuntime::semantic_search`].
    semantic: Option<Arc<SemanticIndex>>,
    /// The graph-observation service, present only when
    /// [`SymphonyConfig::graph_artifact_root`] names a directory: the append-only evidence log
    /// and the derived revision documents live on disk, so an in-memory runtime has nowhere
    /// to keep them and [`SymphonyRuntime::submit_observation`] reports
    /// [`Error::Unsupported`](crate::error::Error::Unsupported) instead.
    observation: Option<GraphObservationService>,
    /// The catalog revision the semantic index currently reflects; `None` means the index
    /// has not been synchronized yet. Serializes semantic (re)indexing.
    semantic_sync: AsyncMutex<Option<u64>>,
    /// Serializes derive-and-publish and records mutation results per `request_id`
    /// (in memory, and through the optional journal).
    ledger: AsyncMutex<MutationLedger>,
}

impl SymphonyRuntime {
    /// Opens a runtime with every default part and no preliminary wiring.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`SymphonyRuntimeBuilder::open`].
    pub async fn open(config: SymphonyConfig) -> Result<Self> {
        Self::builder(config).open().await
    }

    /// Starts a builder for a runtime configured by `config`.
    #[must_use]
    pub fn builder(config: SymphonyConfig) -> SymphonyRuntimeBuilder {
        SymphonyRuntimeBuilder::new(config)
    }

    /// The configuration this runtime was opened with.
    #[must_use]
    pub fn config(&self) -> &SymphonyConfig {
        &self.config
    }

    /// The current lifecycle state.
    #[must_use]
    pub fn state(&self) -> RuntimeState {
        *self.read_state()
    }

    /// The authoritative capability catalog.
    ///
    /// Handy for direct catalog management: writing through this handle does not publish
    /// a graph revision, so call [`Self::build`] afterwards to reflect the change.
    #[must_use]
    pub fn registry(&self) -> &Arc<CapabilityRegistry> {
        &self.registry
    }

    /// Starts the engine: [`RuntimeState::Created`] (or [`RuntimeState::Stopped`]) becomes
    /// [`RuntimeState::Running`], publishing a first revision through [`Self::build`] when
    /// the store holds none. The state changes only after a successful build, and calling
    /// `start` while running is a no-op.
    ///
    /// # Errors
    ///
    /// Returns the artifact backend's status error or the [`Self::build`] errors.
    pub async fn start(&self) -> Result<()> {
        if self.state() == RuntimeState::Running {
            return Ok(());
        }
        if !self.artifacts.status().await?.is_ready() {
            self.build().await?;
        }
        *self.write_state() = RuntimeState::Running;
        info!("symphony runtime started");
        Ok(())
    }

    /// Stops the engine; idempotent.
    ///
    /// # Errors
    ///
    /// This implementation never fails; the [`Result`] keeps the lifecycle API uniform
    /// with [`Self::start`] and [`Self::close`].
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn stop(&self) -> Result<()> {
        if self.state() != RuntimeState::Stopped {
            *self.write_state() = RuntimeState::Stopped;
            info!("symphony runtime stopped");
        }
        Ok(())
    }

    /// Stops the engine and flushes the artifact store by re-reading the current revision,
    /// so a backend that cannot reproduce what it published surfaces here. Idempotent like
    /// [`Self::stop`].
    ///
    /// When the runtime owns an observation service, the observation log is flushed too (see
    /// [`Self::flush_observations`]), because that flush must happen before the runtime is
    /// dropped and it is the only way a stored projection error is surfaced. An in-memory
    /// runtime has no observation service, so its `close` behaviour is unchanged.
    ///
    /// # Errors
    ///
    /// Returns the artifact backend's status and read errors, plus the
    /// [`Self::flush_observations`] error — a projection failure recorded by an earlier
    /// [`Self::submit_observation`].
    pub async fn close(&self) -> Result<()> {
        self.stop().await?;
        let status = self.graph_status().await?;
        let flushed = self.artifacts.read(None).await?;
        debug!(
            revision = ?status.current_version,
            nodes = flushed.as_ref().map_or(0, CapabilityGraph::node_count),
            edges = flushed.as_ref().map_or(0, CapabilityGraph::edge_count),
            "closed the symphony runtime and flushed its graph artifacts"
        );
        // Only a runtime with an observation service has anything to flush; an in-memory
        // runtime keeps its current behaviour.
        if let Some(observation) = &self.observation {
            observation.flush().await?;
        }
        Ok(())
    }

    /// Re-derives the capability graph from the catalog plus the [`CandidateGenerator`]'s
    /// `can_feed` edges and publishes a new immutable revision (`v1`, `v2`, ...).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](crate::error::Error::Validation) for a rejected derived edge, or the artifact
    /// backend's publish error.
    pub async fn build(&self) -> Result<GraphRevision> {
        let _ledger = self.ledger.lock().await;
        self.rebuild().await
    }

    /// The current graph revision, or a node-only graph derived from the catalog.
    ///
    /// A backend read failure is logged and answered with the derived graph: this accessor
    /// reports a best-effort view instead of failing a caller that only wants to inspect.
    pub async fn graph(&self) -> CapabilityGraph {
        match self.artifacts.read(None).await {
            Ok(Some(graph)) => graph,
            Ok(None) => self.derived_graph().await,
            Err(error) => {
                warn!(
                    error = %error,
                    "failed to read the current capability graph; deriving one from the catalog"
                );
                self.derived_graph().await
            }
        }
    }

    /// Describes the current revision, or reports `"missing"` when none is published.
    ///
    /// # Errors
    ///
    /// Returns the artifact backend's error.
    pub async fn graph_status(&self) -> Result<GraphArtifactStatus> {
        let status = self.artifacts.status().await?;
        *self.write_pointer() = GraphPointer::from_status(&status);
        Ok(status)
    }

    /// Reads a published revision; `None` means the current one.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](crate::error::Error::Validation) for an illegal version, or the backend's error.
    pub async fn read_graph(&self, version: Option<&str>) -> Result<Option<CapabilityGraph>> {
        self.artifacts.read(version).await
    }

    /// Plans a request over the current graph and catalog, building a revision first when
    /// the store holds none. Planning never mutates the catalog or the graph.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](crate::error::Error::Validation) for a blank query, the [`Self::build`] errors, or the
    /// configured planner's scorer error.
    pub async fn plan(&self, request: &PlanRequest) -> Result<OrchestrationPlan> {
        if !self.artifacts.status().await?.is_ready() {
            self.build().await?;
        }
        let graph = self.graph().await;
        let capabilities = self.registry.list().await;
        self.planner.plan(request, &graph, &capabilities).await
    }

    /// Plans the request, and optionally executes the plan.
    ///
    /// Requires [`RuntimeState::Running`]. `query`, `candidate_ids`, and `disabled_ids` map
    /// onto a [`PlanRequest`] whose other fields keep their defaults (`"cn"` language, no
    /// `provided_inputs`, no overrides). With [`InvokeRequest::execute`] set and a plan
    /// that is not [`PlanStatus::NoPlan`], the plan runs through the configured
    /// [`CapabilityRunner`] with the request's `inputs` and `session`; a
    /// [`PlanStatus::NoPlan`] result has nothing to execute, so `execution` is `None`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](crate::error::Error::Validation) naming `runtime.state` unless running, the
    /// [`Self::plan`] errors, and the [`PlanExecutor::execute_with_inputs`] errors —
    /// notably [`Error::Unsupported`](crate::error::Error::Unsupported) without a runner.
    pub async fn invoke(&self, request: InvokeRequest) -> Result<InvokeResponse> {
        self.require_running()?;
        let plan_request = PlanRequest {
            query: request.query.clone(),
            candidate_ids: request.candidate_ids.clone(),
            disabled_ids: request.disabled_ids.clone(),
            ..PlanRequest::default()
        };
        let plan = self.plan(&plan_request).await?;
        let execution = if request.execute && plan.status != PlanStatus::NoPlan {
            let capabilities = self.registry.list().await;
            let executor =
                PlanExecutor::with_config(self.config.executor.clone(), Arc::clone(&self.runner));
            let execution = executor
                .execute_with_inputs(
                    &plan,
                    &capabilities,
                    request.inputs.clone(),
                    request.session.clone(),
                )
                .await?;
            Some(execution)
        } else {
            None
        };
        let graph_status = self.graph_status().await?;
        Ok(InvokeResponse {
            plan,
            execution,
            graph_status,
        })
    }

    /// Searches the catalog with keyword and metadata filters.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](crate::error::Error::Validation) for an invalid `metadata_filter` predicate.
    pub async fn search(&self, request: &SearchRequest) -> Result<Vec<SearchHit>> {
        self.registry.search(request).await
    }

    /// Searches the catalog semantically, embedding only `query` when the index is
    /// already synchronized.
    ///
    /// The index is lazily synchronized: the catalog is cleared and embedded again only
    /// when [`CapabilityRegistry::revision`] changed since the last synchronization, so an
    /// unchanged catalog embeds just the query, while every mutation (including removals)
    /// is reflected before the next search. `start` does not seed the index. `top_k == 0`
    /// means unlimited.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`](crate::error::Error::Unsupported) without a [`SemanticIndex`], or the embedding
    /// model's and vector store's errors.
    pub async fn semantic_search(&self, query: &str, top_k: usize) -> Result<Vec<SearchHit>> {
        let Some(index) = &self.semantic else {
            return Err(unsupported(
                "semantic capability search without a semantic index",
            ));
        };
        // The revision is read *before* the snapshot: a mutation in between records an
        // older revision against a newer catalog (one harmless extra reindex) instead of
        // recording a newer revision against a stale catalog.
        let mut synced = self.semantic_sync.lock().await;
        let revision = self.registry.revision();
        if *synced != Some(revision) {
            let capabilities = self.registry.list().await;
            index.clear().await?;
            index.index(&capabilities).await?;
            *synced = Some(revision);
        }
        // The search is not part of the reindex critical section.
        drop(synced);
        index.search(query, top_k).await
    }

    /// Clears and re-embeds the whole catalog, recording the current revision, and returns
    /// how many descriptors were indexed.
    ///
    /// This is the manual escape hatch for a changed embedding model, which the catalog
    /// revision cannot observe by itself. It serializes with [`SymphonyRuntime::semantic_search`]
    /// through the same mutex.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`](crate::error::Error::Unsupported) without a [`SemanticIndex`], or the embedding
    /// model's and vector store's errors.
    pub async fn reindex_semantics(&self) -> Result<usize> {
        let Some(index) = &self.semantic else {
            return Err(unsupported("semantic reindex without a semantic index"));
        };
        let mut synced = self.semantic_sync.lock().await;
        let revision = self.registry.revision();
        let capabilities = self.registry.list().await;
        index.clear().await?;
        let indexed = index.index(&capabilities).await?;
        *synced = Some(revision);
        Ok(indexed)
    }

    /// Upserts `descriptors` into the catalog and publishes a new revision.
    ///
    /// A thin delegation to [`Self::mutate`] with an [`MutationOperation::Add`] request
    /// built from the descriptors' qualified ids, no `source_revision`, and `request_id` as
    /// the idempotency key: a replay returns the recorded result with
    /// [`GraphMutationResult::replayed`] set and no side effect, while reusing the id for a
    /// different batch is rejected. Every descriptor is validated first, so a bad batch
    /// changes nothing. Requires [`RuntimeState::Running`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](crate::error::Error::Validation) naming `runtime.state`, `request_id`,
    /// `changed_skill_ids`, or a descriptor, plus the store's and artifact backend's errors.
    pub async fn add_capabilities(
        &self,
        descriptors: Vec<CapabilityDescriptor>,
        request_id: &str,
    ) -> Result<GraphMutationResult> {
        let request = MutationRequest::prepare(
            MutationOperation::Add,
            &qualified_ids(&descriptors),
            request_id,
            None,
        )?;
        self.mutate(request, descriptors).await
    }

    /// Upserts only already-registered `descriptors` and publishes a new revision.
    ///
    /// A thin delegation to [`Self::mutate`] with an [`MutationOperation::Update`] request,
    /// governed by the same replay and digest rules as [`Self::add_capabilities`]; an
    /// unknown key rejects the whole batch.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](crate::error::Error::Validation) naming `runtime.state`/`request_id`/`changed_skill_ids`
    /// or a descriptor, [`Error::NotFound`](crate::error::Error::NotFound) naming the first unregistered id, and backend
    /// errors.
    pub async fn update_capabilities(
        &self,
        descriptors: Vec<CapabilityDescriptor>,
        request_id: &str,
    ) -> Result<GraphMutationResult> {
        let request = MutationRequest::prepare(
            MutationOperation::Update,
            &qualified_ids(&descriptors),
            request_id,
            None,
        )?;
        self.mutate(request, descriptors).await
    }

    /// Removes the listed keys and publishes a new revision.
    ///
    /// A thin delegation to [`Self::mutate`] with a [`MutationOperation::Delete`] request
    /// whose batch is the keys' qualified ids; an unregistered key is skipped, not reported,
    /// which is why such a batch reports [`MutationStatus::Noop`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](crate::error::Error::Validation) naming `runtime.state`, `request_id`, or
    /// `changed_skill_ids` (an empty batch cannot be prepared), and backend errors.
    pub async fn remove_capabilities(
        &self,
        keys: &[CapabilityKey],
        request_id: &str,
    ) -> Result<GraphMutationResult> {
        let ids: Vec<String> = keys.iter().map(CapabilityKey::qualified).collect();
        let request = MutationRequest::prepare(MutationOperation::Delete, &ids, request_id, None)?;
        self.mutate(request, Vec::new()).await
    }

    /// Applies observed step transitions to the graph as an exponentially weighted moving
    /// average: each consecutive pair of successful steps (sorted by id) whose pair has a
    /// graph edge sets that weight to `(w + 1.0) / 2` and is accepted, the rest are
    /// rejected. A revision is published only when at least one edge was updated —
    /// otherwise `revision` is the current version (empty when none exists). Evolution
    /// works in every state.
    ///
    /// # Errors
    ///
    /// Returns the artifact backend's read or publish errors.
    pub async fn submit_evolution(&self, execution: &PlanExecution) -> Result<EvolutionResult> {
        let _ledger = self.ledger.lock().await;
        let transitions = transitions(execution);
        let Some(mut graph) = self.artifacts.read(None).await? else {
            debug!(
                transitions = transitions.len(),
                "no graph revision exists; every observed transition is rejected"
            );
            return Ok(EvolutionResult {
                accepted_edges: 0,
                rejected_edges: transitions.len(),
                revision: String::new(),
            });
        };
        let mut accepted_edges = 0;
        let mut rejected_edges = 0;
        for (source, target) in &transitions {
            // Weights stay inside `[0, 1]` because `[0, 1]` maps onto `[0.5, 1]`, so the
            // updated edge is always valid and the in-place update cannot corrupt the
            // document ordering.
            let edge = graph
                .edges
                .iter_mut()
                .find(|edge| edge.source == *source && edge.target == *target);
            match edge {
                Some(edge) => {
                    edge.weight = f64::midpoint(edge.weight, 1.0);
                    accepted_edges += 1;
                }
                None => rejected_edges += 1,
            }
        }
        let revision = if accepted_edges > 0 {
            let revision = self.artifacts.publish(&graph, None).await?;
            self.observe(&revision);
            revision.version
        } else {
            self.artifacts
                .status()
                .await?
                .current_version
                .unwrap_or_default()
        };
        debug!(
            accepted = accepted_edges,
            rejected = rejected_edges,
            revision = %revision,
            "applied evolution feedback to the capability graph"
        );
        Ok(EvolutionResult {
            accepted_edges,
            rejected_edges,
            revision,
        })
    }

    /// Submits one caller-supplied evidence row and returns its receipt.
    ///
    /// The service qualifies the payload against the static revision it names, appends one row
    /// to the evidence log below [`SymphonyConfig::graph_artifact_root`], and rebinds the
    /// scope's merged overlay, so a later [`Self::observation_snapshot`] sees the observed
    /// weights. Works in every lifecycle state: evidence describes what a task did, not what
    /// the engine is doing.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`](crate::error::Error::Unsupported) when no
    /// [`SymphonyConfig::graph_artifact_root`] is configured, and the service's errors —
    /// [`Error::Validation`](crate::error::Error::Validation) for a contract violation or a
    /// reused `evidence_id`, [`Error::NotFound`](crate::error::Error::NotFound) when no static
    /// revision has been published, and [`Error::External`](crate::error::Error::External)
    /// when the log cannot be appended to.
    pub async fn submit_observation(
        &self,
        input: &GraphEvolutionInput,
    ) -> Result<ObservationReceipt> {
        self.require_observation()?.submit(input).await
    }

    /// Reads one merged snapshot of `graph_scope_id`: the static `can_feed` graph plus the
    /// observation overlay a planner consumes.
    ///
    /// `merged_revision` pins a published revision; `None` answers the current one, which is
    /// reprojected first when it is missing or bound to a stale static revision.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`](crate::error::Error::Unsupported) when no
    /// [`SymphonyConfig::graph_artifact_root`] is configured, and the service's errors —
    /// notably [`Error::NotFound`](crate::error::Error::NotFound) for an unknown static or
    /// merged revision, and [`Error::External`](crate::error::Error::External) for unreadable
    /// documents.
    pub async fn observation_snapshot(
        &self,
        graph_scope_id: &str,
        merged_revision: Option<&str>,
    ) -> Result<GraphSnapshot> {
        self.require_observation()?
            .get_snapshot(graph_scope_id, merged_revision)
            .await
    }

    /// Reports the first projection failure recorded by an earlier
    /// [`Self::submit_observation`], if any.
    ///
    /// A submit appends its row and aggregates it; an aggregation failure neither loses the
    /// row nor fails the submit, so this flush is where the caller learns about it. The error
    /// is taken, so two consecutive flushes report each failure at most once.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`](crate::error::Error::Unsupported) when no
    /// [`SymphonyConfig::graph_artifact_root`] is configured, otherwise the stored projection
    /// error.
    pub async fn flush_observations(&self) -> Result<()> {
        self.require_observation()?.flush().await
    }

    /// Returns the observation service, or the documented `Unsupported` error when the
    /// runtime keeps its graph revisions in memory.
    fn require_observation(&self) -> Result<&GraphObservationService> {
        self.observation
            .as_ref()
            .ok_or_else(observation_unavailable)
    }

    /// A synchronous snapshot of the current revision pointer:
    /// `{"current_version", "graph_id", "graph_scope_id"}`, with the first two `null` when
    /// no revision is known yet. The pointer is cached on open, publish, and status, so
    /// this accessor never awaits.
    #[must_use]
    pub fn capture_graph_snapshot(&self) -> BTreeMap<String, Value> {
        let pointer = self.read_pointer();
        BTreeMap::from([
            (
                "current_version".to_string(),
                pointer
                    .current_version
                    .clone()
                    .map_or(Value::Null, Value::String),
            ),
            (
                "graph_id".to_string(),
                pointer.graph_id.clone().map_or(Value::Null, Value::String),
            ),
            (
                "graph_scope_id".to_string(),
                Value::String(self.config.graph_scope_id.clone()),
            ),
        ])
    }

    /// Re-derives the graph from the catalog and publishes a revision; the caller holds the
    /// ledger lock.
    async fn rebuild(&self) -> Result<GraphRevision> {
        let capabilities = self.registry.list().await;
        self.publish_graph(&capabilities).await
    }

    /// Derives the graph of `capabilities` and publishes it as a new revision; the caller
    /// holds the ledger lock.
    async fn publish_graph(&self, capabilities: &[CapabilityDescriptor]) -> Result<GraphRevision> {
        let mut graph =
            CapabilityGraph::from_descriptors(self.config.graph_scope_id.clone(), capabilities);
        let generator = CandidateGenerator::new(self.config.candidates.clone());
        let candidates = generator.generate(capabilities);
        for edge in generator.edges(&candidates) {
            graph.add_edge(edge)?;
        }
        let revision = self.artifacts.publish(&graph, None).await?;
        self.observe(&revision);
        debug!(
            version = %revision.version,
            nodes = revision.node_count,
            edges = revision.edge_count,
            "published a capability graph revision"
        );
        Ok(revision)
    }

    /// Builds the node-only graph the catalog implies (no derived edges).
    async fn derived_graph(&self) -> CapabilityGraph {
        CapabilityGraph::from_descriptors(
            self.config.graph_scope_id.clone(),
            &self.registry.list().await,
        )
    }

    /// The `(source, target)` pairs of the current revision, or an empty set.
    async fn edge_set(&self) -> Result<BTreeSet<(String, String)>> {
        let Some(graph) = self.artifacts.read(None).await? else {
            return Ok(BTreeSet::new());
        };
        Ok(graph
            .edges
            .iter()
            .map(|edge| (edge.source.clone(), edge.target.clone()))
            .collect())
    }

    /// Fails unless the runtime is [`RuntimeState::Running`].
    fn require_running(&self) -> Result<()> {
        let state = self.state();
        if state == RuntimeState::Running {
            return Ok(());
        }
        Err(validation(
            "runtime.state",
            format!(
                "must be `running` for this operation, got `{}`",
                state_name(state)
            ),
        ))
    }

    /// Records a published revision in the cached pointer.
    fn observe(&self, revision: &GraphRevision) {
        *self.write_pointer() = GraphPointer {
            current_version: Some(revision.version.clone()),
            graph_id: Some(revision.graph_id.clone()),
        };
    }

    /// Locks the lifecycle state for reading, recovering from a poisoned lock.
    fn read_state(&self) -> RwLockReadGuard<'_, RuntimeState> {
        self.state.read().unwrap_or_else(PoisonError::into_inner)
    }

    /// Locks the lifecycle state for writing, recovering from a poisoned lock.
    fn write_state(&self) -> RwLockWriteGuard<'_, RuntimeState> {
        self.state.write().unwrap_or_else(PoisonError::into_inner)
    }

    /// Locks the cached graph pointer for reading, recovering from a poisoned lock.
    fn read_pointer(&self) -> RwLockReadGuard<'_, GraphPointer> {
        self.pointer.read().unwrap_or_else(PoisonError::into_inner)
    }

    /// Locks the cached graph pointer for writing, recovering from a poisoned lock.
    fn write_pointer(&self) -> RwLockWriteGuard<'_, GraphPointer> {
        self.pointer.write().unwrap_or_else(PoisonError::into_inner)
    }
}

impl fmt::Debug for SymphonyRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SymphonyRuntime")
            .field("config", &self.config)
            .field("state", &self.state())
            .field("pointer", &*self.read_pointer())
            .field("capabilities", &self.registry.len())
            .field("artifacts", &"dyn GraphArtifactStore")
            .field("planner", &"Planner")
            .field("runner", &"dyn CapabilityRunner")
            .field("semantic_index", &self.semantic.is_some())
            .field("observation", &self.observation.is_some())
            .finish_non_exhaustive()
    }
}

/// The cached pointer to the current graph revision.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct GraphPointer {
    /// The version of the current revision, when one is known.
    current_version: Option<String>,
    /// The `graph_id` of the current revision, when one is known.
    graph_id: Option<String>,
}

impl GraphPointer {
    /// Builds the pointer an artifact status reports.
    fn from_status(status: &GraphArtifactStatus) -> Self {
        Self {
            current_version: status.current_version.clone(),
            graph_id: status.graph_id.clone(),
        }
    }
}

/// The sorted, de-duplicated qualified ids of `descriptors`.
fn qualified_ids(descriptors: &[CapabilityDescriptor]) -> Vec<String> {
    descriptors
        .iter()
        .map(CapabilityDescriptor::qualified_id)
        .collect::<BTreeSet<String>>()
        .into_iter()
        .collect()
}

/// The error an observation call reports when the runtime has no filesystem artifact root.
fn observation_unavailable() -> Error {
    unsupported("graph observation requires a configured `graph_artifact_root`")
}

/// The consecutive successful step pairs of `execution`, sorted by step id.
fn transitions(execution: &PlanExecution) -> Vec<(String, String)> {
    let steps: Vec<&StepOutcome> = execution.steps.values().collect();
    steps
        .windows(2)
        .filter(|pair| pair[0].success && pair[1].success)
        .map(|pair| (pair[0].step_id.clone(), pair[1].step_id.clone()))
        .collect()
}

/// The lowercase name of `state`, used in error messages and logs.
fn state_name(state: RuntimeState) -> &'static str {
    match state {
        RuntimeState::Created => "created",
        RuntimeState::Running => "running",
        RuntimeState::Stopped => "stopped",
    }
}

/// One call to [`SymphonyRuntime::invoke`]: a request, and whether to execute the plan.
///
/// `inputs` and `session` are the execution seeds; a plan-only call ignores them.
pub struct InvokeRequest {
    /// The natural-language request the plan must answer.
    pub query: String,
    /// When set, only these qualified ids may appear in the plan.
    pub candidate_ids: Option<Vec<String>>,
    /// Qualified ids excluded from the plan entirely.
    pub disabled_ids: Vec<String>,
    /// Whether to execute the plan after planning it.
    pub execute: bool,
    /// The caller-supplied inputs seeded into the plan's root steps.
    pub inputs: Value,
    /// The session the plan executes in; `None` uses a fresh workflow session.
    pub session: Option<Arc<dyn Session>>,
}

impl InvokeRequest {
    /// Builds a plan-only `"cn"` request for `query` with no filters and no inputs.
    #[must_use]
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            candidate_ids: None,
            disabled_ids: Vec::new(),
            execute: false,
            inputs: Value::Null,
            session: None,
        }
    }

    /// Requests execution of the resulting plan.
    #[must_use]
    pub fn execute(mut self) -> Self {
        self.execute = true;
        self
    }

    /// Seeds the plan's root steps with `inputs`.
    #[must_use]
    pub fn with_inputs(mut self, inputs: Value) -> Self {
        self.inputs = inputs;
        self
    }

    /// Runs the plan in `session`.
    #[must_use]
    pub fn with_session(mut self, session: Arc<dyn Session>) -> Self {
        self.session = Some(session);
        self
    }
}

/// What [`SymphonyRuntime::invoke`] produced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct InvokeResponse {
    /// The plan for the request.
    pub plan: OrchestrationPlan,
    /// The execution, when the request asked for one and the plan has steps.
    pub execution: Option<PlanExecution>,
    /// The graph revision status observed after handling the request.
    pub graph_status: GraphArtifactStatus,
}

/// What [`SymphonyRuntime::submit_evolution`] applied.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct EvolutionResult {
    /// The number of observed transitions whose graph edge weight was updated.
    pub accepted_edges: usize,
    /// The number of observed transitions with no matching graph edge.
    pub rejected_edges: usize,
    /// The version of the published revision, or the unchanged current version.
    pub revision: String,
}
