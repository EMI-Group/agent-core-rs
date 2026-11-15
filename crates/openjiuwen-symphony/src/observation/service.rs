//! The composition layer: qualification, aggregation, rebinding and snapshot services.
//!
//! Ports Python `openjiuwen.symphony.observation.service` (v0.1.20). The service owns the
//! append-only evidence log and the derived observation/merged revisions: it qualifies a
//! caller-supplied [`GraphEvolutionInput`], appends one row, replays the eligible rows into
//! a per-edge overlay, rebinds that overlay to the current static revision, and publishes an
//! immutable snapshot a planner can pin.
//!
//! # Substitutions for Python machinery
//!
//! - **No worker thread.** Python appends on the caller's thread and schedules aggregation
//!   on a single-worker `ThreadPoolExecutor`. Here both steps run *synchronously inside the
//!   caller's task*: [`GraphObservationService::submit`] appends, then reprojects, and only
//!   returns once the overlay is published. A failed reprojection does not lose the row — it
//!   is reported by [`GraphObservationService::flush`], mirroring Python's deferred
//!   worker-error semantics — so a caller can still treat the append as durable.
//! - **No `FileLock`.** Mutual exclusion is one [`tokio::sync::Mutex`] covering the whole
//!   aggregation of one scope, which is the in-process equivalent of Python's per-scope
//!   projection lock. No `std::sync` guard is ever held across an `.await`.
//! - **The evidence log is `JSONL`.** Python keeps rows in `SQLite`; this port uses
//!   [`EvidenceStore`]'s append-only `JSONL` file. Deduplication, sequences and the
//!   high-water cursor behave identically.
//!
//! # The overlay edge record
//!
//! The overlay is a deliberately untyped cross-module contract: planners read
//! `binding_status`, `runtime_weight`, `edge_identity`, `attempt_count`, `success_count` and
//! `runtime_only` straight out of the JSON, so those keys must never be renamed. Each record
//! also carries the identity fields (`identity_schema`, `source_id`, `target_id`,
//! `relation_type`, `source_content_hash`, `target_content_hash`), the counters
//! (`failure_count`, `attempt_count`), the session bookkeeping
//! (`success_session_hashes`, `success_sessions_by_revision`, `failure_session_hashes`), the
//! task clusters (`task_cluster_ids`), `ever_static`, and the observation timestamps
//! (`first_observed_at`, `last_observed_at`, `last_evidence_static_revision`).
//!
//! # Fail-soft reads
//!
//! [`GraphObservationService::get_snapshot`] never fails because a projection failed: it
//! reprojects, and on failure publishes a [`SAFE_FALLBACK`] overlay whose zero high-water
//! cursor makes the next successful reprojection replay the whole scope log.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

use serde_json::{Map, Value};
use tokio::sync::{Mutex as AsyncMutex, RwLock as AsyncRwLock};
use tracing::warn;

use crate::artifact::GraphArtifactStore;
use crate::error::{Error, Result};
use crate::observation::contracts::{
    EvidenceStrength, EvolutionEdgeMetadata, FailureDomain, GraphEvolutionInput, GraphSnapshot,
    ObservationReceipt, ObservationStatus, TaskOutcomeLabel, CAPABILITY_IDENTITY_MISMATCH,
    GRAPH_EVOLUTION_INPUT_SCHEMA, MERGED_OBSERVATION_REVISION_MISMATCH,
    MERGED_STATIC_REVISION_MISMATCH, NO_QUALIFIED_EDGE_OUTCOME, OUTCOME_NOT_STRONG, SAFE_FALLBACK,
    TRACE_QUALITY_FLAGS, TRUNCATED_TRACE, UNKNOWN_MERGED_REVISION, UNKNOWN_STATIC_REVISION,
};
use crate::observation::identity::{
    edge_identity_from_observation, StaticGraphIndex, POINT_EDGE_IDENTITY_SCHEMA,
};
use crate::observation::payload::{
    self, EdgeOutcome, QualifiedEdge, EDGES_FIELD, EDGE_IDENTITY_FIELD, IDENTITY_SCHEMA_FIELD,
    OBSERVED_AS_STATIC_FIELD, OBSERVED_AT_FIELD, OUTCOME_FAILURE, OUTCOME_FIELD, OUTCOME_SUCCESS,
    RELATION_TYPE_FIELD, SCHEMA_VERSION_FIELD, SESSION_ID_HASH_FIELD, SOURCE_CONTENT_HASH_FIELD,
    SOURCE_ID_FIELD, STATIC_REVISION_FIELD, TARGET_CONTENT_HASH_FIELD, TARGET_ID_FIELD,
    TASK_CLUSTER_ID_FIELD,
};
use crate::observation::store::{EvidenceAppendInput, EvidenceStore, OverlayInput, RevisionStore};
use crate::observation::timestamp;

/// How many distinct successful sessions a runtime-only edge needs on the current static
/// revision before it becomes plannable.
pub const MIN_RUNTIME_ONLY_SUCCESS_SESSIONS: usize = 2;

/// The weight step applied per net success.
pub const RUNTIME_ADJUSTMENT_STEP: f64 = 0.05;

/// The lower bound of the runtime adjustment (a weight of `0.0`).
pub const MIN_RUNTIME_ADJUSTMENT: f64 = -1.0;

/// The upper bound of the runtime adjustment (a weight of `2.0`).
pub const MAX_RUNTIME_ADJUSTMENT: f64 = 1.0;

/// The scope name used when the evidence log is empty.
const DEFAULT_SCOPE: &str = "default";

/// The `high_water_sequence` key of an overlay document.
const HIGH_WATER_SEQUENCE_FIELD: &str = "high_water_sequence";

/// The `projection_status` key of an overlay document.
const PROJECTION_STATUS_FIELD: &str = "projection_status";

/// The projection status of an overlay derived from the evidence log.
const PROJECTION_READY: &str = "ready";

/// `binding_status` — read by planners; never rename.
const BINDING_STATUS_FIELD: &str = "binding_status";
/// The runtime weight `1.0 + runtime_adjustment` — read by planners; never rename.
const RUNTIME_WEIGHT_FIELD: &str = "runtime_weight";
/// The clamped weight delta — derived, never persisted by a caller.
const RUNTIME_ADJUSTMENT_FIELD: &str = "runtime_adjustment";
/// Whether the edge is plannable purely from runtime evidence — read by planners.
const RUNTIME_ONLY_FIELD: &str = "runtime_only";
/// The number of distinct sessions that touched the edge — read by planners.
const ATTEMPT_COUNT_FIELD: &str = "attempt_count";
/// The number of distinct successful sessions — read by planners.
const SUCCESS_COUNT_FIELD: &str = "success_count";
/// The number of distinct failing sessions.
const FAILURE_COUNT_FIELD: &str = "failure_count";
/// The sorted session hashes an edge succeeded for.
const SUCCESS_SESSION_HASHES_FIELD: &str = "success_session_hashes";
/// Successful session hashes per static revision.
const SUCCESS_SESSIONS_BY_REVISION_FIELD: &str = "success_sessions_by_revision";
/// The sorted session hashes an edge failed for.
const FAILURE_SESSION_HASHES_FIELD: &str = "failure_session_hashes";
/// The sorted task cluster ids that reached the edge.
const TASK_CLUSTER_IDS_FIELD: &str = "task_cluster_ids";
/// Whether the edge ever belonged to a static graph.
const EVER_STATIC_FIELD: &str = "ever_static";
/// When the edge was first observed.
const FIRST_OBSERVED_AT_FIELD: &str = "first_observed_at";
/// When the edge was last observed.
const LAST_OBSERVED_AT_FIELD: &str = "last_observed_at";
/// The static revision of the last applied evidence.
const LAST_EVIDENCE_STATIC_REVISION_FIELD: &str = "last_evidence_static_revision";

/// The identity schema does not match [`POINT_EDGE_IDENTITY_SCHEMA`].
const BINDING_LEGACY_IGNORED: &str = "legacy_ignored";
/// An endpoint no longer exists in the static graph.
const BINDING_ORPHANED: &str = "orphaned";
/// An endpoint's content changed, so the recorded evidence is stale.
const BINDING_INVALIDATED: &str = "invalidated";
/// The edge exists in the current static graph.
const BINDING_ACTIVE_STATIC: &str = "active_static";
/// The edge was static once but its evidence belongs to an older static revision.
const BINDING_QUARANTINED: &str = "quarantined";
/// The edge is absent from the static graph but has enough runtime-only successes.
const BINDING_ACTIVE_RUNTIME_ONLY: &str = "active_runtime_only";
/// The edge is absent from the static graph and lacks runtime-only evidence.
const BINDING_INSUFFICIENT_EVIDENCE: &str = "insufficient_evidence";

/// Qualification, aggregation, rebinding and snapshot services over one artifact root.
///
/// Build one with [`GraphObservationService::new`] from the same
/// [`GraphArtifactStore`] the runtime publishes static revisions to, then hand evidence to
/// [`GraphObservationService::submit`] and read the derived overlay through
/// [`GraphObservationService::get_snapshot`].
pub struct GraphObservationService {
    /// The static `can_feed` artifact source.
    artifacts: Arc<dyn GraphArtifactStore>,
    /// The append-only evidence log.
    evidence: EvidenceStore,
    /// The immutable observation and merged revision documents.
    revisions: RevisionStore,
    /// Serializes aggregation; Python's per-scope projection lock.
    aggregate_lock: AsyncMutex<()>,
    /// One [`StaticGraphIndex`] per static revision, never invalidated.
    static_indexes: AsyncRwLock<BTreeMap<String, Arc<StaticGraphIndex>>>,
    /// The first failed reprojection, reported (and cleared) by `flush`.
    worker_error: AsyncMutex<Option<Error>>,
}

impl GraphObservationService {
    /// Builds a service over `artifacts`, storing evidence and revisions below `root`.
    #[must_use]
    pub fn new(artifacts: Arc<dyn GraphArtifactStore>, root: impl AsRef<Path>) -> Self {
        let root = root.as_ref();
        Self {
            artifacts,
            evidence: EvidenceStore::new(root),
            revisions: RevisionStore::new(root),
            aggregate_lock: AsyncMutex::new(()),
            static_indexes: AsyncRwLock::new(BTreeMap::new()),
            worker_error: AsyncMutex::new(None),
        }
    }

    /// Qualifies `input`, appends one evidence row, and aggregates the scope.
    ///
    /// The input is normalized first, so the row records the canonical UTC `observed_at`
    /// and canonical capability ports. Evidence against a static revision the artifact store
    /// does not know is still recorded, but as an `audit_only` row whose reason starts with
    /// [`UNKNOWN_STATIC_REVISION`]. A reprojection failure after a successful append is kept
    /// for [`GraphObservationService::flush`] and does not fail this call: the row is
    /// durable either way.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] naming the offending field when `input` violates the
    /// contract or reuses an `evidence_id` with different content, [`Error::NotFound`] when
    /// no static revision has been published, and [`Error::External`] when the log cannot be
    /// appended to.
    pub async fn submit(&self, input: &GraphEvolutionInput) -> Result<ObservationReceipt> {
        let input = input.clone().normalized()?;
        let requested = input.graph_snapshot.static_revision.clone();
        let (eligible, reason, edges) = if let Some(index) = self.static_index(&requested).await? {
            let (reasons, edges) = self.qualify(&input, &index).await?;
            (reasons.is_empty(), reasons.join(","), edges)
        } else {
            // An unknown static revision is audited against the current index so the
            // payload still carries per-edge identities, then forced ineligible.
            let index = self.current_static_index().await?;
            let (mut reasons, edges) = self.qualify(&input, &index).await?;
            reasons.insert(0, UNKNOWN_STATIC_REVISION.to_string());
            (false, join_reasons(&reasons), edges)
        };

        let payload = payload::evidence_payload(&input, &edges);
        let appended = self
            .evidence
            .append(EvidenceAppendInput {
                graph_scope_id: &input.graph_scope_id,
                evidence_id: &input.evidence_id,
                observed_at: &input.observed_at,
                eligible,
                reason: &reason,
                payload: &payload,
            })
            .await?;

        if !appended.duplicate {
            if let Err(error) = self.reproject(&input.graph_scope_id).await {
                warn!(
                    scope = %input.graph_scope_id,
                    %error,
                    "aggregating Symphony graph observations failed; the evidence row is recorded and the failure is reported by flush"
                );
                let mut pending = self.worker_error.lock().await;
                if pending.is_none() {
                    *pending = Some(error);
                }
            }
        }

        let status = if appended.duplicate {
            ObservationStatus::Duplicate
        } else if eligible {
            ObservationStatus::Accepted
        } else {
            ObservationStatus::AuditOnly
        };
        Ok(ObservationReceipt {
            evidence_id: input.evidence_id,
            graph_scope_id: input.graph_scope_id,
            sequence: appended.sequence,
            status,
            reason: appended.reason,
        })
    }

    /// Returns the snapshot for `merged_revision`, or the current one rebound to the static
    /// graph.
    ///
    /// A pinned revision that does not resolve is [`Error::NotFound`] rather than a silent
    /// fallback; the current snapshot is reprojected when it is missing, bound to a stale
    /// static revision, or carrying a [`SAFE_FALLBACK`] overlay. A failed reprojection
    /// publishes a neutral fallback snapshot instead of failing the read.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] naming `static_revision` when no static graph has been
    /// published, [`Error::NotFound`] naming `merged_revision` when a pinned revision is
    /// unknown, and [`Error::External`] when a document cannot be read.
    pub async fn get_snapshot(
        &self,
        graph_scope_id: &str,
        merged_revision: Option<&str>,
    ) -> Result<GraphSnapshot> {
        let static_index = self.current_static_index().await?;
        let snapshot = self
            .revisions
            .read_snapshot(graph_scope_id, merged_revision)
            .await?;
        if let Some(revision) = merged_revision {
            return snapshot.ok_or_else(|| {
                Error::not_found(format!(
                    "merged_revision: unknown Symphony merged revision: {revision}"
                ))
            });
        }
        let static_revision = static_index.revision.clone();
        let needs_projection = snapshot.as_ref().is_none_or(|current| {
            current.static_revision != static_revision
                || current
                    .overlay
                    .get(PROJECTION_STATUS_FIELD)
                    .and_then(Value::as_str)
                    == Some(SAFE_FALLBACK)
        });
        if let Some(snapshot) = snapshot {
            if !needs_projection {
                return Ok(snapshot);
            }
        }
        match self.reproject(graph_scope_id).await {
            Ok(snapshot) => Ok(snapshot),
            Err(error) => {
                warn!(
                    scope = graph_scope_id,
                    %error,
                    "Symphony observation projection failed; publishing a neutral fallback overlay"
                );
                self.publish_safe_fallback(graph_scope_id, error.kind().as_str())
                    .await
            }
        }
    }

    /// Reports the first failed reprojection since the last flush, if any.
    ///
    /// Submitting deliberately does not surface an aggregation failure (the append is
    /// durable), so this is the place a caller learns about it; the error is taken, so two
    /// consecutive flushes report each failure at most once.
    ///
    /// # Errors
    ///
    /// Returns the stored reprojection error, or `Ok(())` when every aggregation succeeded.
    pub async fn flush(&self) -> Result<()> {
        match self.worker_error.lock().await.take() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    /// Rebinds the accumulated observations of one scope to the current static revision.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] naming `static_revision` when no static graph has been
    /// published, and [`Error::External`] when the evidence log or a revision document
    /// cannot be read or written.
    pub async fn reproject(&self, graph_scope_id: &str) -> Result<GraphSnapshot> {
        let _guard = self.aggregate_lock.lock().await;
        self.aggregate(graph_scope_id).await
    }

    /// Rebinds every scope in the evidence log, flushing first.
    ///
    /// # Errors
    ///
    /// Returns whatever [`GraphObservationService::flush`] or the per-scope reprojection
    /// returns; use [`GraphObservationService::reproject_all_safely`] for the fail-soft
    /// variant.
    pub async fn reproject_all(&self) -> Result<Vec<GraphSnapshot>> {
        self.flush().await?;
        let mut snapshots = Vec::new();
        for scope in self.scopes().await? {
            snapshots.push(self.reproject(&scope).await?);
        }
        Ok(snapshots)
    }

    /// Rebinds every scope, falling back to a neutral overlay per scope on failure.
    ///
    /// # Errors
    ///
    /// Returns only errors that also defeat the fallback, i.e. a missing static revision or
    /// an unreadable scope list.
    pub async fn reproject_all_safely(&self) -> Result<Vec<GraphSnapshot>> {
        if let Err(error) = self.flush().await {
            warn!(
                %error,
                "Symphony observation worker failed before static rebinding; continuing"
            );
        }
        let mut snapshots = Vec::new();
        for scope in self.scopes().await? {
            match self.reproject(&scope).await {
                Ok(snapshot) => snapshots.push(snapshot),
                Err(error) => {
                    warn!(
                        scope = %scope,
                        %error,
                        "rebinding failed for a Symphony graph scope; publishing a neutral fallback"
                    );
                    snapshots.push(
                        self.publish_safe_fallback(&scope, error.kind().as_str())
                            .await?,
                    );
                }
            }
        }
        Ok(snapshots)
    }

    /// Publishes an empty, explicitly retryable overlay bound to the current static revision.
    ///
    /// The zero high-water cursor is deliberate: the next successful reprojection replays the
    /// whole scope log, so a fallback never hides evidence permanently.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] naming `static_revision` when no static graph has been
    /// published, and [`Error::External`] when the revision documents cannot be written.
    pub async fn publish_safe_fallback(
        &self,
        graph_scope_id: &str,
        reason: &str,
    ) -> Result<GraphSnapshot> {
        let index = self.current_static_index().await?;
        let edges = BTreeMap::new();
        self.revisions
            .publish(&OverlayInput {
                graph_scope_id,
                static_revision: &index.revision,
                high_water_sequence: 0,
                edges: &edges,
                generated_at: &timestamp::utc_now(),
                projection_status: SAFE_FALLBACK,
                diagnostics: &[reason.to_string()],
            })
            .await
    }

    /// Releases the service.
    ///
    /// Documented no-op: this port has no worker thread to drain, and every append and
    /// projection has already been written by the time the call that triggered it returned.
    /// The method exists because Python's `close()` shuts the executor down.
    ///
    /// # Errors
    ///
    /// Never fails; the `Result` keeps the caller's teardown path uniform.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)] // the Python teardown contract is asynchronous
    pub async fn close(&self) -> Result<()> {
        Ok(())
    }

    /// Qualifies `input` against `index`, returning the refusal reasons and accepted edges.
    async fn qualify(
        &self,
        input: &GraphEvolutionInput,
        index: &StaticGraphIndex,
    ) -> Result<(Vec<String>, Vec<QualifiedEdge>)> {
        let mut reasons = Vec::new();
        if input.trace.truncated {
            reasons.push(TRUNCATED_TRACE.to_string());
        }
        if !input.trace.quality_flags.is_empty() {
            reasons.push(TRACE_QUALITY_FLAGS.to_string());
        }
        if input.task.outcome.evidence_strength != EvidenceStrength::Strong {
            reasons.push(OUTCOME_NOT_STRONG.to_string());
        }
        if !index.validates_nodes(input.execution_graph.nodes.keys().map(String::as_str)) {
            reasons.push(CAPABILITY_IDENTITY_MISMATCH.to_string());
        }
        if let Some(reason) = self.snapshot_reference_reason(input).await? {
            reasons.push(reason.to_string());
        }

        let mut edges = Vec::new();
        for edge in &input.execution_graph.edges {
            let Some(identity) = edge_identity_from_observation(edge, index) else {
                continue;
            };
            let Some(outcome) = edge_outcome(input, &edge.metadata) else {
                continue;
            };
            let observed_as_static = index
                .edges_by_identity
                .contains_key(&identity.identity_hash());
            edges.push(QualifiedEdge {
                identity,
                outcome,
                observed_as_static,
            });
        }
        if edges.is_empty() {
            reasons.push(NO_QUALIFIED_EDGE_OUTCOME.to_string());
        }
        Ok((reasons, edges))
    }

    /// Reports why a pinned merged revision disqualifies the evidence, if it does.
    async fn snapshot_reference_reason(
        &self,
        input: &GraphEvolutionInput,
    ) -> Result<Option<&'static str>> {
        let Some(merged_revision) = input.graph_snapshot.merged_revision.as_deref() else {
            return Ok(None);
        };
        let snapshot = self
            .revisions
            .read_snapshot(&input.graph_scope_id, Some(merged_revision))
            .await?;
        let Some(snapshot) = snapshot else {
            return Ok(Some(UNKNOWN_MERGED_REVISION));
        };
        if snapshot.static_revision != input.graph_snapshot.static_revision {
            return Ok(Some(MERGED_STATIC_REVISION_MISMATCH));
        }
        if snapshot.observation_revision != input.graph_snapshot.observation_revision {
            return Ok(Some(MERGED_OBSERVATION_REVISION_MISMATCH));
        }
        Ok(None)
    }

    /// Returns the cached index for `revision`, or [`None`] when no such revision exists.
    async fn static_index(&self, revision: &str) -> Result<Option<Arc<StaticGraphIndex>>> {
        let cached = self.static_indexes.read().await.get(revision).cloned();
        if let Some(cached) = cached {
            return Ok(Some(cached));
        }
        let Some(graph) = self.artifacts.read(Some(revision)).await? else {
            return Ok(None);
        };
        let index = Arc::new(StaticGraphIndex::from_graph(revision, &graph));
        self.static_indexes
            .write()
            .await
            .insert(revision.to_string(), Arc::clone(&index));
        Ok(Some(index))
    }

    /// Returns the index of the artifact store's current static revision.
    async fn current_static_index(&self) -> Result<Arc<StaticGraphIndex>> {
        let status = self.artifacts.status().await?;
        let Some(revision) = status.current_version else {
            return Err(Error::not_found(
                "static_revision: no Symphony static graph revision has been published",
            ));
        };
        self.static_index(&revision).await?.ok_or_else(|| {
            Error::not_found(format!(
                "static_revision: advertised revision `{revision}` cannot be read"
            ))
        })
    }

    /// Replays the eligible evidence of one scope into a freshly published overlay.
    ///
    /// The caller must hold [`Self::aggregate_lock`].
    async fn aggregate(&self, graph_scope_id: &str) -> Result<GraphSnapshot> {
        let index = self.current_static_index().await?;
        let current = self
            .revisions
            .current_overlay(graph_scope_id)
            .await?
            .unwrap_or(Value::Null);
        let mut high_water_sequence = current
            .get(HIGH_WATER_SEQUENCE_FIELD)
            .and_then(Value::as_u64)
            .unwrap_or_default();
        let mut edges = overlay_edges(&current);
        for row in self
            .evidence
            .iter_after(graph_scope_id, high_water_sequence)
            .await?
        {
            // The cursor advances past ineligible rows too, so an audit-only row is never
            // re-evaluated even if the qualification rules would now accept it.
            high_water_sequence = row.sequence;
            if row.eligible {
                apply_evidence(&mut edges, &row.payload()?, &index);
            }
        }
        for record in edges.values_mut() {
            rebind(record, &index);
        }
        self.revisions
            .publish(&OverlayInput {
                graph_scope_id,
                static_revision: &index.revision,
                high_water_sequence,
                edges: &edges,
                generated_at: &timestamp::utc_now(),
                projection_status: PROJECTION_READY,
                diagnostics: &[],
            })
            .await
    }

    /// The scopes the evidence log knows, defaulting to [`DEFAULT_SCOPE`] when it is empty.
    async fn scopes(&self) -> Result<Vec<String>> {
        let mut scopes = self.evidence.scope_ids().await?;
        if scopes.is_empty() {
            scopes.push(DEFAULT_SCOPE.to_string());
        }
        Ok(scopes)
    }
}

/// Joins non-empty reasons with `,`.
fn join_reasons(reasons: &[String]) -> String {
    reasons
        .iter()
        .filter(|reason| !reason.is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join(",")
}

/// Resolves the qualified outcome of one observed transition.
///
/// A verified success credits only an edge the rail marked successful; a verified failure
/// blames only an edge whose failure *and* whose task carry the orchestration domain. A
/// partial, cancelled or unknown task credits nothing, so a transport or permission failure
/// can never poison routing.
fn edge_outcome(
    input: &GraphEvolutionInput,
    metadata: &EvolutionEdgeMetadata,
) -> Option<EdgeOutcome> {
    match input.task.outcome.label {
        TaskOutcomeLabel::VerifiedSuccess => {
            matches!(metadata.success, Some(true)).then_some(EdgeOutcome::Success)
        }
        TaskOutcomeLabel::VerifiedFailure => {
            if metadata.success != Some(false) {
                return None;
            }
            let orchestration = metadata.failure_domain == Some(FailureDomain::Orchestration)
                && input.task.outcome.failure_domain == Some(FailureDomain::Orchestration);
            orchestration.then_some(EdgeOutcome::Failure)
        }
        TaskOutcomeLabel::Partial | TaskOutcomeLabel::Cancelled | TaskOutcomeLabel::Unknown => None,
    }
}

/// Copies the object entries of an overlay's `edges` document.
fn overlay_edges(overlay: &Value) -> BTreeMap<String, Value> {
    overlay
        .get(EDGES_FIELD)
        .and_then(Value::as_object)
        .map(|entries| {
            entries
                .iter()
                .filter(|(_, record)| record.is_object())
                .map(|(identity, record)| (identity.clone(), record.clone()))
                .collect()
        })
        .unwrap_or_default()
}

/// Folds one eligible payload into the per-edge overlay records.
///
/// A payload from another schema version is ignored, and an edge object without an
/// `edge_identity` is skipped, so a hand-written or future log row cannot corrupt the
/// overlay. Counters are session-deduplicated, exactly as Python's `_apply_evidence`.
fn apply_evidence(edges: &mut BTreeMap<String, Value>, payload: &Value, index: &StaticGraphIndex) {
    if payload.get(SCHEMA_VERSION_FIELD).and_then(Value::as_str)
        != Some(GRAPH_EVOLUTION_INPUT_SCHEMA)
    {
        return;
    }
    let session_hash = value_text(payload, SESSION_ID_HASH_FIELD);
    let static_revision = value_text(payload, STATIC_REVISION_FIELD);
    let task_cluster_id = value_text(payload, TASK_CLUSTER_ID_FIELD)
        .trim()
        .to_string();
    let observed_at = value_text(payload, OBSERVED_AT_FIELD);
    let Some(items) = payload.get(EDGES_FIELD).and_then(Value::as_array) else {
        return;
    };
    for item in items {
        let Some(item) = item.as_object() else {
            continue;
        };
        let edge_identity = item
            .get(EDGE_IDENTITY_FIELD)
            .and_then(Value::as_str)
            .unwrap_or_default();
        if edge_identity.is_empty() {
            continue;
        }
        let record = edges
            .entry(edge_identity.to_string())
            .or_insert_with(|| new_record(edge_identity, item, &observed_at));
        let Some(record) = record.as_object_mut() else {
            continue;
        };
        let outcome = item.get(OUTCOME_FIELD).and_then(Value::as_str);
        if outcome == Some(OUTCOME_SUCCESS) {
            let success_sessions = string_list(record.get(SUCCESS_SESSION_HASHES_FIELD));
            if !session_hash.is_empty() && !success_sessions.contains(&session_hash) {
                increment(record, SUCCESS_COUNT_FIELD);
            }
            append_unique(record, SUCCESS_SESSION_HASHES_FIELD, &session_hash);
            record_revision_success(record, &static_revision, &session_hash);
        } else if outcome == Some(OUTCOME_FAILURE) {
            let failure_sessions = string_list(record.get(FAILURE_SESSION_HASHES_FIELD));
            if !session_hash.is_empty() && !failure_sessions.contains(&session_hash) {
                increment(record, FAILURE_COUNT_FIELD);
            }
            append_unique(record, FAILURE_SESSION_HASHES_FIELD, &session_hash);
        }
        let attempt_count = session_union_size(record);
        record.insert(ATTEMPT_COUNT_FIELD.to_string(), Value::from(attempt_count));
        if !task_cluster_id.is_empty() {
            append_unique(record, TASK_CLUSTER_IDS_FIELD, &task_cluster_id);
        }
        record.insert(
            LAST_OBSERVED_AT_FIELD.to_string(),
            Value::from(observed_at.clone()),
        );
        record.insert(
            LAST_EVIDENCE_STATIC_REVISION_FIELD.to_string(),
            Value::from(static_revision.clone()),
        );
        let ever_static = bool_at(record, EVER_STATIC_FIELD)
            || item
                .get(OBSERVED_AS_STATIC_FIELD)
                .and_then(Value::as_bool)
                .unwrap_or(false)
            || index.edges_by_identity.contains_key(edge_identity);
        record.insert(EVER_STATIC_FIELD.to_string(), Value::Bool(ever_static));
    }
}

/// Creates the counter record of an edge the overlay has not seen before.
fn new_record(edge_identity: &str, item: &Map<String, Value>, observed_at: &str) -> Value {
    let mut record = Map::new();
    record.insert(
        IDENTITY_SCHEMA_FIELD.to_string(),
        Value::from(POINT_EDGE_IDENTITY_SCHEMA),
    );
    record.insert(EDGE_IDENTITY_FIELD.to_string(), Value::from(edge_identity));
    for field in [
        SOURCE_ID_FIELD,
        TARGET_ID_FIELD,
        RELATION_TYPE_FIELD,
        SOURCE_CONTENT_HASH_FIELD,
        TARGET_CONTENT_HASH_FIELD,
    ] {
        record.insert(
            field.to_string(),
            item.get(field).cloned().unwrap_or(Value::Null),
        );
    }
    record.insert(SUCCESS_COUNT_FIELD.to_string(), Value::from(0));
    record.insert(FAILURE_COUNT_FIELD.to_string(), Value::from(0));
    record.insert(ATTEMPT_COUNT_FIELD.to_string(), Value::from(0));
    record.insert(
        SUCCESS_SESSION_HASHES_FIELD.to_string(),
        Value::Array(Vec::new()),
    );
    record.insert(
        SUCCESS_SESSIONS_BY_REVISION_FIELD.to_string(),
        Value::Object(Map::new()),
    );
    record.insert(
        FAILURE_SESSION_HASHES_FIELD.to_string(),
        Value::Array(Vec::new()),
    );
    record.insert(TASK_CLUSTER_IDS_FIELD.to_string(), Value::Array(Vec::new()));
    record.insert(EVER_STATIC_FIELD.to_string(), Value::Bool(false));
    record.insert(
        FIRST_OBSERVED_AT_FIELD.to_string(),
        Value::from(observed_at),
    );
    Value::Object(record)
}

/// Records one successful session under the static revision it was observed on.
fn record_revision_success(
    record: &mut Map<String, Value>,
    static_revision: &str,
    session_hash: &str,
) {
    let mut by_revision = record
        .get(SUCCESS_SESSIONS_BY_REVISION_FIELD)
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let mut sessions = string_list(by_revision.get(static_revision));
    if !session_hash.is_empty() && !sessions.iter().any(|hash| hash == session_hash) {
        sessions.push(session_hash.to_string());
    }
    sessions.sort_unstable();
    by_revision.insert(static_revision.to_string(), string_array(sessions));
    record.insert(
        SUCCESS_SESSIONS_BY_REVISION_FIELD.to_string(),
        Value::Object(by_revision),
    );
}

/// Appends `value` to the sorted string list at `key`, ignoring duplicates and empty values.
fn append_unique(record: &mut Map<String, Value>, key: &str, value: &str) {
    if value.is_empty() {
        return;
    }
    let mut values = string_list(record.get(key));
    if !values.iter().any(|existing| existing == value) {
        values.push(value.to_string());
    }
    values.sort_unstable();
    record.insert(key.to_string(), string_array(values));
}

/// Increments the counter at `key`, saturating instead of overflowing.
fn increment(record: &mut Map<String, Value>, key: &str) {
    let next = u64_at(record, key).saturating_add(1);
    record.insert(key.to_string(), Value::from(next));
}

/// Returns the size of the union of the success and failure session hashes.
fn session_union_size(record: &Map<String, Value>) -> u64 {
    let union: BTreeSet<String> = string_list(record.get(SUCCESS_SESSION_HASHES_FIELD))
        .into_iter()
        .chain(string_list(record.get(FAILURE_SESSION_HASHES_FIELD)))
        .collect();
    u64::try_from(union.len()).unwrap_or(u64::MAX)
}

/// Recomputes an overlay record's `binding_status`, `runtime_weight` and `runtime_only`.
///
/// The precedence is Python's: a legacy identity is ignored, a vanished endpoint is
/// orphaned, changed endpoint content invalidates, a current static edge is active, an edge
/// that was static on an older revision is quarantined, a runtime-only edge with enough
/// distinct successes is plannable, and everything else is insufficient evidence.
fn rebind(record: &mut Value, index: &StaticGraphIndex) {
    let Some(record) = record.as_object_mut() else {
        return;
    };
    let identity_schema = map_text(record, IDENTITY_SCHEMA_FIELD);
    let source_id = map_text(record, SOURCE_ID_FIELD);
    let target_id = map_text(record, TARGET_ID_FIELD);
    let edge_identity = map_text(record, EDGE_IDENTITY_FIELD);
    let ever_static = bool_at(record, EVER_STATIC_FIELD);
    let on_current_revision = record
        .get(LAST_EVIDENCE_STATIC_REVISION_FIELD)
        .and_then(Value::as_str)
        .is_some_and(|revision| revision == index.revision);
    let status = if identity_schema != POINT_EDGE_IDENTITY_SCHEMA {
        BINDING_LEGACY_IGNORED
    } else if !index.capability_ids.contains(source_id.as_str())
        || !index.capability_ids.contains(target_id.as_str())
    {
        BINDING_ORPHANED
    } else if content_hash_changed(record, &source_id, &target_id, index) {
        BINDING_INVALIDATED
    } else if index.edges_by_identity.contains_key(&edge_identity) {
        BINDING_ACTIVE_STATIC
    } else if ever_static && !on_current_revision {
        BINDING_QUARANTINED
    } else if has_runtime_only_evidence(record, &index.revision) {
        BINDING_ACTIVE_RUNTIME_ONLY
    } else {
        BINDING_INSUFFICIENT_EVIDENCE
    };
    if status == BINDING_ACTIVE_STATIC {
        record.insert(EVER_STATIC_FIELD.to_string(), Value::Bool(true));
    }
    let adjustment = runtime_adjustment(
        u64_at(record, SUCCESS_COUNT_FIELD),
        u64_at(record, FAILURE_COUNT_FIELD),
    );
    record.insert(BINDING_STATUS_FIELD.to_string(), Value::from(status));
    record.insert(
        RUNTIME_ADJUSTMENT_FIELD.to_string(),
        Value::from(adjustment),
    );
    record.insert(
        RUNTIME_WEIGHT_FIELD.to_string(),
        Value::from(1.0 + adjustment),
    );
    record.insert(
        RUNTIME_ONLY_FIELD.to_string(),
        Value::Bool(status == BINDING_ACTIVE_RUNTIME_ONLY),
    );
}

/// Reports whether either endpoint's content hash moved away from the record's.
fn content_hash_changed(
    record: &Map<String, Value>,
    source_id: &str,
    target_id: &str,
    index: &StaticGraphIndex,
) -> bool {
    index.content_hash_by_id.get(source_id).map(String::as_str)
        != record
            .get(SOURCE_CONTENT_HASH_FIELD)
            .and_then(Value::as_str)
        || index.content_hash_by_id.get(target_id).map(String::as_str)
            != record
                .get(TARGET_CONTENT_HASH_FIELD)
                .and_then(Value::as_str)
}

/// Reports whether the edge has enough failure-free successes on `static_revision`.
fn has_runtime_only_evidence(record: &Map<String, Value>, static_revision: &str) -> bool {
    if u64_at(record, FAILURE_COUNT_FIELD) != 0 {
        return false;
    }
    let successes = record
        .get(SUCCESS_SESSIONS_BY_REVISION_FIELD)
        .and_then(Value::as_object)
        .map_or_else(Vec::new, |by_revision| {
            string_list(by_revision.get(static_revision))
        });
    successes.len() >= MIN_RUNTIME_ONLY_SUCCESS_SESSIONS
}

/// Clamps `0.05 * (success - failure)` into `[-1.0, 1.0]`.
#[allow(clippy::cast_precision_loss)] // session counters stay far below 2^53
fn runtime_adjustment(success_count: u64, failure_count: u64) -> f64 {
    let raw = RUNTIME_ADJUSTMENT_STEP * (success_count as f64 - failure_count as f64);
    raw.clamp(MIN_RUNTIME_ADJUSTMENT, MAX_RUNTIME_ADJUSTMENT)
}

/// Returns the string at `key` of a JSON object, or `""`.
fn value_text(document: &Value, key: &str) -> String {
    document
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// Returns the string at `key` of a JSON object map, or `""`.
fn map_text(record: &Map<String, Value>, key: &str) -> String {
    record
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// Returns the boolean at `key` of a JSON object map, or `false`.
fn bool_at(record: &Map<String, Value>, key: &str) -> bool {
    record.get(key).and_then(Value::as_bool).unwrap_or(false)
}

/// Returns the unsigned integer at `key` of a JSON object map, or `0`.
fn u64_at(record: &Map<String, Value>, key: &str) -> u64 {
    record.get(key).and_then(Value::as_u64).unwrap_or_default()
}

/// Returns the string entries of a JSON array, ignoring other values.
fn string_list(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// Renders a string list as a JSON array.
fn string_array(values: Vec<String>) -> Value {
    Value::Array(values.into_iter().map(Value::from).collect())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::json;
    use tempfile::TempDir;

    use super::*;
    use crate::artifact::MemoryGraphArtifactStore;
    use crate::error::ErrorKind;
    use crate::graph::{CapabilityEdge, CapabilityGraph, CapabilityNode, CAN_FEED};
    use crate::observation::contracts::{
        EvolutionGraph, EvolutionGraphEdge, EvolutionGraphNode, GraphSnapshotRef, TaskEvidence,
        TaskOutcome, TraceEvidence,
    };

    /// A service over a fresh artifact root whose static graph is published as `v1`.
    struct Harness {
        /// Keeps the artifact root alive for as long as the service lives.
        _root: TempDir,
        /// The static revision source, kept typed so a test can republish.
        artifacts: Arc<MemoryGraphArtifactStore>,
        /// The service under test.
        service: GraphObservationService,
    }

    /// Builds a harness with `v1` published.
    async fn harness() -> Harness {
        let root = tempfile::tempdir().unwrap();
        let artifacts = Arc::new(MemoryGraphArtifactStore::new());
        artifacts
            .publish(&static_graph(), Some("v1"))
            .await
            .unwrap();
        let service = GraphObservationService::new(
            Arc::clone(&artifacts) as Arc<dyn GraphArtifactStore>,
            root.path(),
        );
        Harness {
            _root: root,
            artifacts,
            service,
        }
    }

    /// A two-capability static graph with one `can_feed` edge.
    fn static_graph() -> CapabilityGraph {
        let mut graph = CapabilityGraph::new("observation-test");
        graph.upsert_node(node("skill:a"));
        graph.upsert_node(node("skill:b"));
        graph
            .add_edge(CapabilityEdge::new("skill:a", "skill:b", CAN_FEED, 1.0))
            .unwrap();
        graph
    }

    /// The same nodes without the edge, for the quarantine scenario.
    fn static_graph_without_edges() -> CapabilityGraph {
        let mut graph = CapabilityGraph::new("observation-test");
        graph.upsert_node(node("skill:a"));
        graph.upsert_node(node("skill:b"));
        graph
    }

    /// One static node carrying a capability type and a content hash.
    fn node(id: &str) -> CapabilityNode {
        let mut node = CapabilityNode::new(id, id, "skill");
        node.metadata
            .insert("capability_type".to_string(), json!("skill"));
        node.metadata
            .insert("content_hash".to_string(), json!(format!("hash-{id}")));
        node
    }

    /// One execution node carrying the metadata the contract allow-lists.
    fn execution_node() -> EvolutionGraphNode {
        let mut metadata = BTreeMap::new();
        metadata.insert("capability_type".to_string(), json!("skill"));
        metadata.insert("version".to_string(), json!("1.0.0"));
        EvolutionGraphNode {
            label: "skill".to_string(),
            metadata,
        }
    }

    /// One observed transition.
    fn execution_edge(
        source: &str,
        target: &str,
        success: Option<bool>,
        failure_domain: Option<FailureDomain>,
    ) -> EvolutionGraphEdge {
        EvolutionGraphEdge {
            source_id: source.to_string(),
            target_id: target.to_string(),
            relation_type: CAN_FEED.to_string(),
            metadata: EvolutionEdgeMetadata {
                success,
                failure_domain,
            },
        }
    }

    /// An execution graph over both capabilities with `edges`.
    fn execution_graph_with(edges: Vec<EvolutionGraphEdge>) -> EvolutionGraph {
        let nodes = ["skill:a", "skill:b"]
            .into_iter()
            .map(|id| (id.to_string(), execution_node()))
            .collect();
        EvolutionGraph {
            graph_type: "execution_graph".to_string(),
            nodes,
            edges,
            ..EvolutionGraph::default()
        }
    }

    /// A successful transition along the static edge.
    fn execution_graph() -> EvolutionGraph {
        execution_graph_with(vec![execution_edge("skill:a", "skill:b", Some(true), None)])
    }

    /// A valid input that credits the static edge as a strong success.
    fn input(evidence_id: &str, scope: &str, session_id: &str) -> GraphEvolutionInput {
        GraphEvolutionInput {
            evidence_id: evidence_id.to_string(),
            graph_scope_id: scope.to_string(),
            observed_at: "2026-03-01T00:00:00Z".to_string(),
            graph_snapshot: GraphSnapshotRef {
                static_revision: "v1".to_string(),
                observation_revision: "observation-seed".to_string(),
                merged_revision: None,
            },
            trace: TraceEvidence {
                trajectory_id: format!("trajectory-{evidence_id}"),
                session_id: session_id.to_string(),
                capture_mode: "agent".to_string(),
                ..TraceEvidence::default()
            },
            task: TaskEvidence {
                outcome: TaskOutcome {
                    label: TaskOutcomeLabel::VerifiedSuccess,
                    evidence_strength: EvidenceStrength::Strong,
                    failure_domain: None,
                },
                ..TaskEvidence::default()
            },
            execution_graph: execution_graph(),
            ..GraphEvolutionInput::default()
        }
    }

    /// A valid input whose edge and task both carry a verified failure in `domain`.
    fn failure_input(
        evidence_id: &str,
        scope: &str,
        session_id: &str,
        domain: FailureDomain,
    ) -> GraphEvolutionInput {
        let mut input = input(evidence_id, scope, session_id);
        input.task.outcome = TaskOutcome {
            label: TaskOutcomeLabel::VerifiedFailure,
            evidence_strength: EvidenceStrength::Strong,
            failure_domain: Some(domain),
        };
        input.execution_graph = execution_graph_with(vec![execution_edge(
            "skill:a",
            "skill:b",
            Some(false),
            Some(domain),
        )]);
        input
    }

    /// An input crediting the reverse transition, which the static graph does not contain.
    fn runtime_only_input(evidence_id: &str, scope: &str, session_id: &str) -> GraphEvolutionInput {
        let mut input = input(evidence_id, scope, session_id);
        input.execution_graph =
            execution_graph_with(vec![execution_edge("skill:b", "skill:a", Some(true), None)]);
        input
    }

    /// The overlay edges of a snapshot.
    fn edges_of(snapshot: &GraphSnapshot) -> &Map<String, Value> {
        snapshot.overlay[EDGES_FIELD].as_object().unwrap()
    }

    /// The single overlay edge record of a snapshot.
    fn only_edge(snapshot: &GraphSnapshot) -> &Map<String, Value> {
        let edges = edges_of(snapshot);
        assert_eq!(edges.len(), 1, "expected exactly one overlay edge");
        edges.values().next().unwrap().as_object().unwrap()
    }

    /// Asserts a weight to within floating-point noise.
    fn assert_weight(record: &Map<String, Value>, expected: f64) {
        let weight = record[RUNTIME_WEIGHT_FIELD].as_f64().unwrap();
        assert!(
            (weight - expected).abs() < 1e-9,
            "expected weight {expected}, got {weight}"
        );
    }

    #[tokio::test]
    async fn strong_success_on_a_static_edge_is_accepted() {
        let harness = harness().await;
        let receipt = harness
            .service
            .submit(&input("e1", "default", "session-1"))
            .await
            .unwrap();
        assert_eq!(receipt.status, ObservationStatus::Accepted);
        assert_eq!(receipt.reason, "");
        assert_eq!(receipt.sequence, 1);

        let snapshot = harness.service.get_snapshot("default", None).await.unwrap();
        assert_eq!(snapshot.high_water_sequence, 1);
        assert_eq!(snapshot.static_revision, "v1");
        let edge = only_edge(&snapshot);
        assert_eq!(edge[BINDING_STATUS_FIELD], json!("active_static"));
        assert_eq!(edge[SUCCESS_COUNT_FIELD], json!(1));
        assert_eq!(edge[ATTEMPT_COUNT_FIELD], json!(1));
        assert_eq!(edge[RUNTIME_ONLY_FIELD], json!(false));
        assert_eq!(edge[EVER_STATIC_FIELD], json!(true));
        assert_weight(edge, 1.05);
    }

    #[tokio::test]
    async fn identical_resubmission_is_a_duplicate() {
        let harness = harness().await;
        let first = harness
            .service
            .submit(&input("e1", "default", "session-1"))
            .await
            .unwrap();
        let second = harness
            .service
            .submit(&input("e1", "default", "session-1"))
            .await
            .unwrap();
        assert_eq!(second.status, ObservationStatus::Duplicate);
        assert_eq!(second.sequence, first.sequence);
        assert_eq!(harness.service.evidence.len().await.unwrap(), 1);
        assert_eq!(
            edges_of(&harness.service.get_snapshot("default", None).await.unwrap()).len(),
            1
        );
    }

    #[tokio::test]
    async fn reused_evidence_id_with_different_content_is_rejected() {
        let harness = harness().await;
        harness
            .service
            .submit(&input("e1", "default", "session-1"))
            .await
            .unwrap();
        let mut changed = input("e1", "default", "session-2");
        changed.observed_at = "2026-03-02T00:00:00Z".to_string();
        let error = harness.service.submit(&changed).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.message().contains("evidence_id"));
    }

    #[tokio::test]
    async fn truncated_trace_is_audit_only_and_advances_the_cursor() {
        let harness = harness().await;
        let mut truncated = input("e1", "default", "session-1");
        truncated.trace.truncated = true;
        let receipt = harness.service.submit(&truncated).await.unwrap();
        assert_eq!(receipt.status, ObservationStatus::AuditOnly);
        assert_eq!(receipt.reason, TRUNCATED_TRACE);

        let snapshot = harness.service.get_snapshot("default", None).await.unwrap();
        assert_eq!(snapshot.high_water_sequence, 1);
        assert!(edges_of(&snapshot).is_empty());

        // The cursor has already passed the row, so a resubmission can never become
        // eligible later.
        let again = harness.service.submit(&truncated).await.unwrap();
        assert_eq!(again.status, ObservationStatus::Duplicate);
        assert_eq!(again.reason, TRUNCATED_TRACE);
        let snapshot = harness.service.get_snapshot("default", None).await.unwrap();
        assert_eq!(snapshot.high_water_sequence, 1);
        assert!(edges_of(&snapshot).is_empty());
    }

    #[tokio::test]
    async fn unknown_static_revision_is_audited_against_the_current_index() {
        let harness = harness().await;
        let mut unknown = input("e1", "default", "session-1");
        unknown.graph_snapshot.static_revision = "v9".to_string();
        let receipt = harness.service.submit(&unknown).await.unwrap();
        assert_eq!(receipt.status, ObservationStatus::AuditOnly);
        assert!(receipt.reason.starts_with(UNKNOWN_STATIC_REVISION));

        let rows = harness.service.evidence.rows().await.unwrap();
        assert_eq!(rows.len(), 1);
        let payload = rows[0].payload().unwrap();
        assert_eq!(payload[STATIC_REVISION_FIELD], json!("v9"));
        assert_eq!(payload[EDGES_FIELD].as_array().unwrap().len(), 1);
        assert!(edges_of(&harness.service.get_snapshot("default", None).await.unwrap()).is_empty());
    }

    #[tokio::test]
    async fn orchestration_failure_decrements_the_weight() {
        let harness = harness().await;
        let receipt = harness
            .service
            .submit(&failure_input(
                "e1",
                "default",
                "session-1",
                FailureDomain::Orchestration,
            ))
            .await
            .unwrap();
        assert_eq!(receipt.status, ObservationStatus::Accepted);
        let snapshot = harness.service.get_snapshot("default", None).await.unwrap();
        let edge = only_edge(&snapshot);
        assert_eq!(edge[FAILURE_COUNT_FIELD], json!(1));
        assert_weight(edge, 0.95);
    }

    #[tokio::test]
    async fn network_failure_counts_nothing() {
        let harness = harness().await;
        harness
            .service
            .submit(&input("e1", "default", "session-1"))
            .await
            .unwrap();
        harness
            .service
            .submit(&failure_input(
                "e2",
                "default",
                "session-2",
                FailureDomain::Orchestration,
            ))
            .await
            .unwrap();
        // One success and one orchestration failure cancel out exactly.
        assert_weight(
            only_edge(&harness.service.get_snapshot("default", None).await.unwrap()),
            1.0,
        );

        let receipt = harness
            .service
            .submit(&failure_input(
                "e3",
                "default",
                "session-3",
                FailureDomain::Network,
            ))
            .await
            .unwrap();
        assert_eq!(receipt.status, ObservationStatus::AuditOnly);
        assert_eq!(receipt.reason, NO_QUALIFIED_EDGE_OUTCOME);
        let snapshot = harness.service.get_snapshot("default", None).await.unwrap();
        let edge = only_edge(&snapshot);
        assert_eq!(edge[FAILURE_COUNT_FIELD], json!(1));
        assert_weight(edge, 1.0);
    }

    #[tokio::test]
    async fn runtime_only_edges_need_two_successful_sessions() {
        let harness = harness().await;
        harness
            .service
            .submit(&runtime_only_input("e1", "default", "session-1"))
            .await
            .unwrap();
        let snapshot = harness.service.get_snapshot("default", None).await.unwrap();
        let edge = only_edge(&snapshot);
        assert_eq!(
            edge[BINDING_STATUS_FIELD],
            json!(BINDING_INSUFFICIENT_EVIDENCE)
        );
        assert_eq!(edge[RUNTIME_ONLY_FIELD], json!(false));
        assert_weight(edge, 1.05);

        harness
            .service
            .submit(&runtime_only_input("e2", "default", "session-2"))
            .await
            .unwrap();
        let snapshot = harness.service.get_snapshot("default", None).await.unwrap();
        let edge = only_edge(&snapshot);
        assert_eq!(
            edge[BINDING_STATUS_FIELD],
            json!(BINDING_ACTIVE_RUNTIME_ONLY)
        );
        assert_eq!(edge[RUNTIME_ONLY_FIELD], json!(true));
        assert_eq!(edge[SUCCESS_COUNT_FIELD], json!(2));
        assert_eq!(edge[ATTEMPT_COUNT_FIELD], json!(2));
        assert_weight(edge, 1.1);
    }

    #[tokio::test]
    async fn a_dropped_static_edge_is_quarantined() {
        let harness = harness().await;
        harness
            .service
            .submit(&input("e1", "default", "session-1"))
            .await
            .unwrap();
        harness
            .artifacts
            .publish(&static_graph_without_edges(), Some("v2"))
            .await
            .unwrap();
        let snapshot = harness.service.reproject("default").await.unwrap();
        assert_eq!(snapshot.static_revision, "v2");
        let edge = only_edge(&snapshot);
        assert_eq!(edge[BINDING_STATUS_FIELD], json!(BINDING_QUARANTINED));
        assert_eq!(edge[RUNTIME_ONLY_FIELD], json!(false));
        assert_eq!(edge[LAST_EVIDENCE_STATIC_REVISION_FIELD], json!("v1"));
    }

    #[tokio::test]
    async fn pinned_snapshots_are_returned_or_not_found() {
        let harness = harness().await;
        harness
            .service
            .submit(&input("e1", "default", "session-1"))
            .await
            .unwrap();
        let snapshot = harness.service.get_snapshot("default", None).await.unwrap();
        let pinned = harness
            .service
            .get_snapshot("default", Some(&snapshot.merged_revision))
            .await
            .unwrap();
        assert_eq!(pinned.merged_revision, snapshot.merged_revision);
        assert_eq!(pinned.observation_revision, snapshot.observation_revision);
        assert_eq!(pinned.high_water_sequence, snapshot.high_water_sequence);
        assert_eq!(pinned.overlay, snapshot.overlay);

        let error = harness
            .service
            .get_snapshot("default", Some("merged-missing"))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert!(error.message().contains("merged_revision"));
        assert!(error.message().contains("merged-missing"));
    }

    #[tokio::test]
    async fn safe_fallback_replays_the_whole_log_on_the_next_read() {
        let harness = harness().await;
        harness
            .service
            .submit(&input("e1", "default", "session-1"))
            .await
            .unwrap();
        let fallback = harness
            .service
            .publish_safe_fallback("default", "graph")
            .await
            .unwrap();
        assert_eq!(fallback.high_water_sequence, 0);
        assert!(edges_of(&fallback).is_empty());
        assert_eq!(
            fallback.overlay[PROJECTION_STATUS_FIELD],
            json!(SAFE_FALLBACK)
        );
        assert_eq!(fallback.overlay["diagnostics"], json!(["graph"]));

        let replayed = harness.service.get_snapshot("default", None).await.unwrap();
        assert_eq!(replayed.high_water_sequence, 1);
        assert_eq!(
            replayed.overlay[PROJECTION_STATUS_FIELD],
            json!(PROJECTION_READY)
        );
        assert_eq!(only_edge(&replayed)[SUCCESS_COUNT_FIELD], json!(1));
    }

    #[tokio::test]
    async fn reproject_all_covers_every_scope_and_flush_reports_nothing() {
        let harness = harness().await;
        harness
            .service
            .submit(&input("e1", "alpha", "session-1"))
            .await
            .unwrap();
        harness
            .service
            .submit(&input("e2", "beta", "session-2"))
            .await
            .unwrap();
        harness.service.flush().await.unwrap();
        let snapshots = harness.service.reproject_all().await.unwrap();
        let scopes: Vec<&str> = snapshots
            .iter()
            .map(|snapshot| snapshot.graph_scope_id.as_str())
            .collect();
        assert_eq!(scopes, vec!["alpha", "beta"]);
        assert!(snapshots
            .iter()
            .all(|snapshot| snapshot.high_water_sequence >= 1));
        assert_eq!(
            harness.service.reproject_all_safely().await.unwrap().len(),
            2
        );
        harness.service.close().await.unwrap();
    }

    #[tokio::test]
    async fn submit_without_a_static_revision_is_not_found() {
        let root = tempfile::tempdir().unwrap();
        let artifacts = Arc::new(MemoryGraphArtifactStore::new());
        let service =
            GraphObservationService::new(artifacts as Arc<dyn GraphArtifactStore>, root.path());
        let error = service
            .submit(&input("e1", "default", "session-1"))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert!(error.message().contains("static_revision"));
        let error = service.get_snapshot("default", None).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);
    }
}
