//! The in-process sub-agent runtime.
//!
//! Ports the control subset of `openjiuwen/harness/subagent_runtime/`: one parent session
//! owns a bounded set of live child instances, each running its turns serially on a
//! background `tokio` task. Children are built lazily by a [`SubAgentFactory`], so this
//! module never depends on [`crate::agent`]. Capacity is reserved before the child is
//! created and rolled back on failure; a [`tokio::sync::Semaphore`] bounds concurrent turns.
//! Python defaults (`config.py`): `max_subagents = 10`, `max_concurrent_running = 5`,
//! turn timeout 2400 s.
//!
//! `SubAgentQuota` carries the remaining, opt-in limits (cumulative spawns, per-parent
//! fan-out, nesting depth, LRU eviction) and every timestamp comes from an injected `Clock`,
//! so a `ManualClock` makes the lifecycle fully deterministic in tests.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::sync::Arc;
use std::time::Duration;

use openjiuwen_agent::Agent;
use openjiuwen_common::error::{Error, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::{Notify, Semaphore};

use self::clock::Clock;
use self::clock::SystemClock;
use self::instance::Instance;
use self::lifecycle::{LifecycleHooks, LifecycleStage, SubagentLifecycleEvent};
use self::quota::{SpawnBudget, SubAgentQuota, SubagentActivity};
use self::transcript::{TranscriptEntry, TranscriptSummary};
use self::turn::TurnRunner;
use super::{SubAgentConfig, SubAgentRegistry};

pub mod activity;
mod activity_api;
pub mod clock;
mod evict;
mod instance;
pub mod lifecycle;
mod lifecycle_api;
mod persist_api;
pub mod persistence;
pub mod quota;
pub mod transcript;
mod transcript_api;
mod turn;

pub use self::evict::{SubagentEviction, EVICTION_REASON};
pub use self::persist_api::LoadReport;

/// Default per-turn timeout (Python `TURN_TIMEOUT_S_DEFAULT`).
const DEFAULT_TURN_TIMEOUT: Duration = Duration::from_secs(2400);
/// The sub-agent type whose id is sticky (no random suffix) across spawns.
const STICKY_SUBAGENT_TYPE: &str = "verification_agent";
/// Close reason recorded when the parent ends every instance (Python `parent_ended`).
const CLOSE_REASON_PARENT_ENDED: &str = "parent_ended";
/// Close reason recorded when a caller closes one instance (Python `manual`).
const CLOSE_REASON_MANUAL: &str = "manual";

/// Identifier of one sub-agent instance; it doubles as the child session id.
pub type SubAgentId = String;

/// Lifecycle status of one sub-agent instance (Python `SubagentStatusKind`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubagentStatus {
    /// Registered but the first turn has not started.
    #[default]
    PendingInit,
    /// A turn is currently running.
    Running,
    /// The last turn produced a final answer.
    Completed,
    /// The last turn was interrupted on the parent's request.
    Interrupted,
    /// The last turn failed (model or invocation error).
    Errored,
    /// Closed; only an in-memory tombstone remains.
    Closed,
    /// The requested id was never registered.
    NotFound,
}

impl SubagentStatus {
    /// Whether no further turn will complete without new input.
    #[must_use]
    pub fn is_final(&self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Errored | Self::Closed | Self::NotFound
        )
    }
    /// Returns the Python wire string of this status.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::PendingInit => "pending_init",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Interrupted => "interrupted",
            Self::Errored => "errored",
            Self::Closed => "closed",
            Self::NotFound => "not_found",
        }
    }
}

/// One request to spawn a sub-agent instance (Python `SubagentControl.spawn` arguments).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SubAgentSpawnRequest {
    /// Registered sub-agent type (`agent_card.name` in the registry).
    pub subagent_type: String,
    /// The task the child agent runs in its first turn.
    pub task_description: String,
    /// Optional human-readable name overriding the preset's display name.
    pub display_name: Option<String>,
    /// Optional role description overriding the preset's role.
    pub role: Option<String>,
    /// Optional explicit instance id; generated when absent.
    pub subagent_id: Option<String>,
    /// The instance spawning this one, when nested; `None` for a top-level spawn.
    ///
    /// A nested spawn must name a live instance: it inherits `depth(parent) + 1`, and a
    /// missing or closed parent is rejected with [`Error::Validation`].
    pub parent_subagent_id: Option<String>,
}

impl SubAgentSpawnRequest {
    /// Creates a spawn request for `subagent_type` running `task_description`.
    #[must_use]
    pub fn new(subagent_type: impl Into<String>, task_description: impl Into<String>) -> Self {
        Self {
            subagent_type: subagent_type.into(),
            task_description: task_description.into(),
            ..Self::default()
        }
    }
}

/// The collected result of one sub-agent instance (Python `wait` entry).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SubAgentOutcome {
    /// Instance id, equal to the child session id.
    pub subagent_id: SubAgentId,
    /// Child session id.
    pub sub_session_id: String,
    /// Registered sub-agent type.
    pub subagent_type: String,
    /// Identity card id of the child agent once created.
    pub agent_id: String,
    /// Current lifecycle status.
    pub status: SubagentStatus,
    /// Answer of the last completed turn, when it succeeded.
    pub output: Option<String>,
    /// Error message of the last failed turn, when it failed.
    pub error: Option<String>,
    /// Digest of the instance's transcript at collection time.
    #[serde(default)]
    pub transcript: TranscriptSummary,
}

impl SubAgentOutcome {
    /// Whether this outcome is final (see [`SubagentStatus::is_final`]).
    #[must_use]
    pub fn is_final(&self) -> bool {
        self.status.is_final()
    }
}

/// Registry metadata of one sub-agent instance; timestamps are RFC 3339 strings.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SubagentMetadata {
    /// Instance id, equal to the child session id.
    pub subagent_id: SubAgentId,
    /// Child session id.
    pub sub_session_id: String,
    /// Registered sub-agent type.
    pub subagent_type: String,
    /// Human-readable name from the spawn request.
    pub display_name: Option<String>,
    /// Role description from the spawn request.
    pub role: Option<String>,
    /// The task the instance was spawned with.
    pub task_description: String,
    /// Current lifecycle status.
    pub status: SubagentStatus,
    /// Creation time (RFC 3339).
    pub created_at: String,
    /// Last status transition time (RFC 3339).
    pub updated_at: String,
    /// Derived activity phase driving LRU eviction.
    pub activity: SubagentActivity,
    /// Nesting depth (top-level = 1).
    pub depth: usize,
    /// The instance that spawned this one, when nested.
    pub parent_subagent_id: Option<String>,
    /// Last time the instance was handed work (RFC 3339).
    pub last_used_at: String,
    /// Why the instance was closed, when it was.
    pub closed_reason: Option<String>,
    /// Digest of the instance's transcript (size, ring evictions, newest entry).
    #[serde(default)]
    pub transcript: TranscriptSummary,
}

/// The response of [`SubAgentRuntime::resume`] (Python `ResumeResult`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ResumeReport {
    /// Status of the instance after the call.
    pub status: SubagentStatus,
    /// Whether a closed instance was revived.
    pub restored: bool,
    /// Informational message; set when the instance was already live.
    pub message: Option<String>,
}

/// Builds the child agent of one spawned instance.
///
/// The harness implements this over [`crate::agent::DeepAgent`]; tests use fakes. The
/// factory owns the child's live resources and re-roots the workspace at
/// `sub-agents/<session_id>`.
#[async_trait::async_trait]
pub trait SubAgentFactory: Send + Sync {
    /// Creates the child agent configured by `config` for child session `session_id`.
    ///
    /// # Errors
    ///
    /// Returns an error when the child cannot be built; the runtime rolls the reservation
    /// back and propagates the error unchanged.
    async fn create(&self, config: &SubAgentConfig, session_id: &str) -> Result<Arc<dyn Agent>>;
}

/// State shared by the runtime and its turn tasks.
struct Shared {
    registry: Arc<SubAgentRegistry>,
    factory: Arc<dyn SubAgentFactory>,
    parent_session_id: String,
    notify: Notify,
    /// Hooks awaited at the lifecycle transitions (see [`lifecycle`]).
    lifecycle: LifecycleHooks,
    state: Mutex<RuntimeState>,
}

/// Mutable runtime state, guarded by one [`Mutex`].
struct RuntimeState {
    instances: BTreeMap<SubAgentId, Instance>,
    next_seq: u64,
    /// Cumulative reservations that were not rolled back; never decreases on close.
    spawned: usize,
    /// Revision of the persisted bucket; every snapshot advances it.
    revision: u64,
    /// Ids claimed by an in-flight two-phase eviction, so concurrent evictions of one instance
    /// cannot both write a tombstone.
    evicting: BTreeSet<SubAgentId>,
}

/// The sub-agent instances of one parent session; share it behind an `Arc`.
pub struct SubAgentRuntime {
    shared: Arc<Shared>,
    quota: SubAgentQuota,
    clock: Arc<dyn Clock>,
    turn_timeout: Duration,
    semaphore: Arc<Semaphore>,
}

impl SubAgentRuntime {
    /// Creates a runtime for `parent_session_id` over `registry` and `factory`.
    ///
    /// Defaults match Python: 10 live instances, 5 concurrent turns, 2400 s per turn, no
    /// cumulative/per-parent/depth cap, LRU eviction policy enabled.
    #[must_use]
    pub fn new(
        registry: Arc<SubAgentRegistry>,
        factory: Arc<dyn SubAgentFactory>,
        parent_session_id: impl Into<String>,
    ) -> Self {
        let quota = SubAgentQuota::default();
        Self {
            shared: Arc::new(Shared {
                registry,
                factory,
                parent_session_id: parent_session_id.into(),
                notify: Notify::new(),
                lifecycle: LifecycleHooks::new(),
                state: Mutex::new(RuntimeState {
                    instances: BTreeMap::new(),
                    next_seq: 0,
                    spawned: 0,
                    revision: 0,
                    evicting: BTreeSet::new(),
                }),
            }),
            quota,
            clock: Arc::new(SystemClock),
            turn_timeout: DEFAULT_TURN_TIMEOUT,
            semaphore: Arc::new(Semaphore::new(quota.max_concurrent_running)),
        }
    }

    /// Replaces every limit and rebuilds the turn semaphore; call before sharing.
    #[must_use]
    pub fn with_quota(mut self, quota: SubAgentQuota) -> Self {
        self.semaphore = Arc::new(Semaphore::new(quota.max_concurrent_running));
        self.quota = quota;
        self
    }

    /// Injects the clock every activity timestamp is read from.
    #[must_use]
    pub fn with_clock(mut self, clock: Arc<dyn Clock>) -> Self {
        self.clock = clock;
        self
    }

    /// Overrides the live-instance and concurrent-turn caps; call before sharing.
    ///
    /// Pre-eviction convenience constructor: it keeps the rest of the quota and forces
    /// `enable_lru_eviction = false`, so a full runtime refuses a spawn instead of recycling
    /// an idle instance.
    #[must_use]
    pub fn with_limits(mut self, max_subagents: usize, max_concurrent_running: usize) -> Self {
        self.quota = self
            .quota
            .with_max_subagents(max_subagents)
            .with_concurrent_running(max_concurrent_running)
            .with_lru_eviction(false);
        self.semaphore = Arc::new(Semaphore::new(max_concurrent_running));
        self
    }

    /// Overrides the per-turn timeout (default 2400 s).
    #[must_use]
    pub fn with_turn_timeout(mut self, timeout: Duration) -> Self {
        self.turn_timeout = timeout;
        self
    }

    /// Returns the limits this runtime enforces.
    #[must_use]
    pub fn quota(&self) -> &SubAgentQuota {
        &self.quota
    }

    /// Spawns a sub-agent instance and returns its id immediately.
    ///
    /// The first turn runs on a background task; use [`SubAgentRuntime::wait`] to collect
    /// its result. Capacity is reserved as a `PendingInit` instance before the child is
    /// created and rolled back when construction fails.
    ///
    /// # Errors
    ///
    /// - [`Error::Validation`] when the type is unknown, the id is already live, or
    ///   [`SubAgentSpawnRequest::parent_subagent_id`] names a missing or closed instance;
    /// - [`Error::Config`] when a quota limit (`max_subagents`, `max_total_spawned`,
    ///   `max_children_per_parent`, `max_depth`) is reached;
    /// - the factory error when the child agent cannot be created.
    pub async fn spawn(&self, request: SubAgentSpawnRequest) -> Result<SubAgentId> {
        let spec = self
            .shared
            .registry
            .get_spec(&request.subagent_type)
            .ok_or_else(|| {
                Error::validation(format!(
                    "subagent type not found: {}",
                    request.subagent_type
                ))
            })?;
        let id = request.subagent_id.clone().unwrap_or_else(|| {
            build_subagent_id(
                &self.shared.parent_session_id,
                &request.subagent_type,
                is_sticky_type(&request.subagent_type),
            )
        });
        let mut config = spec.config;
        if request.display_name.is_some() {
            config.display_name = request.display_name.clone();
        }
        if request.role.is_some() {
            config.role = request.role.clone();
        }
        config.subagent_type = Some(request.subagent_type.clone());
        // The instance keeps the patched configuration, so a restored instance can be rebuilt.
        // The capacity-pressure eviction inside `reserve` cannot await `BeforeEvict` (the slot
        // must be freed synchronously), so it only captures the `Close` snapshots; they are
        // awaited here, right after the state lock is released.
        let mut closed_events = Vec::new();
        let seq = match self.reserve(&id, &request, config.clone(), &mut closed_events) {
            Ok(seq) => seq,
            Err(error) => {
                lifecycle_api::run_stage(&self.shared, LifecycleStage::Close, closed_events).await;
                return Err(error);
            }
        };
        lifecycle_api::run_stage(&self.shared, LifecycleStage::Close, closed_events).await;
        let agent = match self.shared.factory.create(&config, &id).await {
            Ok(agent) => agent,
            Err(error) => {
                self.rollback(&id, seq);
                return Err(error);
            }
        };
        let Some((generation, sub_session_id)) = self.attach_agent(&id, seq, &agent) else {
            self.shared.notify.notify_waiters();
            return Err(Error::validation(format!(
                "subagent closed or not found: {id}"
            )));
        };
        tracing::debug!(
            target: "openjiuwen_harness::subagents",
            subagent_id = %id,
            subagent_type = %request.subagent_type,
            depth = self.depth_of(&id).unwrap_or(1),
            "subagent spawned"
        );
        if let Some(event) = self.lifecycle_event_for(&id, LifecycleStage::Spawn, None) {
            lifecycle_api::run_stage(&self.shared, LifecycleStage::Spawn, [event]).await;
        }
        self.start_turn(
            &id,
            generation,
            request.task_description,
            agent,
            sub_session_id,
        );
        Ok(id)
    }

    /// Waits until every requested instance is final or `timeout` elapses.
    ///
    /// Input order is preserved and duplicate ids are collapsed. Unknown ids map to
    /// [`SubagentStatus::NotFound`]; on timeout the current snapshot is returned.
    pub async fn wait(&self, ids: &[SubAgentId], timeout: Duration) -> Vec<SubAgentOutcome> {
        let deadline = tokio::time::Instant::now() + timeout;
        let targets = dedupe(ids);
        loop {
            // Register before snapshotting so a transition between the two is not lost.
            let notified = self.shared.notify.notified();
            let outcomes: Vec<SubAgentOutcome> =
                targets.iter().map(|id| self.outcome_for(id)).collect();
            if outcomes.iter().all(SubAgentOutcome::is_final) {
                return outcomes;
            }
            if tokio::time::timeout_at(deadline, notified).await.is_err() {
                return outcomes;
            }
        }
    }

    /// Returns the current status of `id`; unknown ids report `NotFound`.
    #[must_use]
    pub fn get_status(&self, id: &str) -> SubagentStatus {
        let state = self.shared.state.lock();
        state
            .instances
            .get(id)
            .map_or(SubagentStatus::NotFound, |instance| instance.status)
    }

    /// Returns the current outcome of `id`, or `None` when the id is unknown.
    #[must_use]
    pub fn get_outcome(&self, id: &str) -> Option<SubAgentOutcome> {
        let state = self.shared.state.lock();
        state.instances.get(id).map(Instance::outcome)
    }

    /// Returns the metadata of `id`, or `None` when the id is unknown.
    ///
    /// Closed instances still have metadata (the in-memory tombstone), so this reports them
    /// too; use [`SubAgentRuntime::list`] for live instances only.
    #[must_use]
    pub fn get_metadata(&self, id: &str) -> Option<SubagentMetadata> {
        let state = self.shared.state.lock();
        state.instances.get(id).map(Instance::metadata)
    }

    /// Returns the metadata of every live (non-`Closed`) instance in spawn order.
    ///
    /// Instances are ordered by creation time and then by spawn sequence, which is exactly
    /// spawn order under a monotonic clock and stays deterministic under a frozen one.
    #[must_use]
    pub fn list(&self) -> Vec<SubagentMetadata> {
        let state = self.shared.state.lock();
        let mut live: Vec<&Instance> = state.instances.values().filter(|i| i.is_live()).collect();
        live.sort_by_key(|instance| (instance.created_at_ms, instance.seq));
        live.into_iter().map(Instance::metadata).collect()
    }

    /// Returns `(used, max)` where `used` counts live (non-`Closed`) instances.
    #[must_use]
    pub fn capacity(&self) -> (usize, usize) {
        let state = self.shared.state.lock();
        let used = state.instances.values().filter(|i| i.is_live()).count();
        (used, self.quota.max_subagents)
    }

    /// Returns the cumulative number of spawns (successful reservations, not rolled back).
    ///
    /// Closing an instance does not lower it: it is the session's spawn budget ledger, not a
    /// live count.
    #[must_use]
    pub fn spawned_total(&self) -> usize {
        let state = self.shared.state.lock();
        state.spawned
    }

    /// Queues follow-up input for `id` and returns once it is queued.
    ///
    /// When no turn is running the queued query starts immediately; otherwise the running
    /// turn picks it up (FIFO) after its current invocation. `interrupt` aborts the
    /// running turn first and drops any earlier queued input.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the id is unknown or already closed
    /// (`subagent closed or not found: <id>; call subagent_resume first`), and
    /// `subagent not resumed: <id>; call subagent_resume first` when the instance has no child
    /// agent — a restored instance, which [`SubAgentRuntime::resume`] rebuilds.
    // Kept async for parity with the Python control surface; the tools await it.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn send_input(&self, id: &str, query: &str, interrupt: bool) -> Result<()> {
        let now_ms = self.clock.now_ms();
        let mut start: Option<(u64, String, Arc<dyn Agent>, String)> = None;
        {
            let mut state = self.shared.state.lock();
            let Some(instance) = state.instances.get_mut(id) else {
                return Err(closed_or_missing(id));
            };
            if instance.status == SubagentStatus::Closed {
                return Err(closed_or_missing(id));
            }
            // A restored or tombstoned instance has no child agent, so queueing would never run:
            // the caller has to resume first. A spawn still inside the factory keeps queueing —
            // it is `PendingInit` and its first turn drains what was queued.
            if instance.agent.is_none() && instance.status != SubagentStatus::PendingInit {
                return Err(not_resumed(id));
            }
            let mut interrupted = false;
            if interrupt && instance.running {
                if let Some(handle) = instance.handle.take() {
                    handle.abort();
                }
                instance.running = false;
                instance.generation = instance.generation.wrapping_add(1);
                instance.status = SubagentStatus::Interrupted;
                interrupted = true;
            }
            if interrupt {
                instance.pending.clear();
            }
            if instance.running {
                // The query stays queued behind the running turn (without an interrupt).
                instance.activity_stream.queued(now_ms);
            }
            instance.pending.push_back(query.to_owned());
            instance.record(TranscriptEntry::input(id, query, now_ms));
            if interrupted {
                instance.record(TranscriptEntry::status(
                    id,
                    SubagentStatus::Interrupted.as_str(),
                    now_ms,
                ));
            }
            if !instance.running {
                if let Some(agent) = instance.agent.clone() {
                    if let Some(next) = instance.pending.pop_front() {
                        instance.running = true;
                        instance.generation = instance.generation.wrapping_add(1);
                        if !interrupted {
                            instance.status = SubagentStatus::PendingInit;
                        }
                        start = Some((
                            instance.generation,
                            next,
                            agent,
                            instance.sub_session_id.clone(),
                        ));
                    }
                }
            }
            let activity = if instance.running {
                SubagentActivity::Running
            } else {
                SubagentActivity::Queued
            };
            instance.touch(now_ms, activity);
        }
        self.shared.notify.notify_waiters();
        if let Some((generation, query, agent, sub_session_id)) = start {
            self.start_turn(id, generation, query, agent, sub_session_id);
        }
        Ok(())
    }

    /// Closes `id` and returns its previous status.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the id is unknown or its turn is running.
    // Kept async for parity with the Python control surface; the tools await it.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn close(&self, id: &str) -> Result<SubagentStatus> {
        let now_ms = self.clock.now_ms();
        let (previous, close_event) = {
            let mut state = self.shared.state.lock();
            let Some(instance) = state.instances.get_mut(id) else {
                return Err(not_found(id));
            };
            if instance.status == SubagentStatus::Running {
                return Err(Error::validation(format!(
                    "cannot close running subagent: {id}"
                )));
            }
            let previous = instance.status;
            instance.close(now_ms, CLOSE_REASON_MANUAL);
            instance.record(TranscriptEntry::closed(id, CLOSE_REASON_MANUAL, now_ms));
            instance.activity_stream.closed(CLOSE_REASON_MANUAL, now_ms);
            let event =
                instance.lifecycle_event(LifecycleStage::Close, Some(CLOSE_REASON_MANUAL), now_ms);
            (previous, event)
        };
        self.shared.notify.notify_waiters();
        lifecycle_api::run_stage(&self.shared, LifecycleStage::Close, [close_event]).await;
        tracing::debug!(
            target: "openjiuwen_harness::subagents",
            subagent_id = %id,
            previous = previous.as_str(),
            "subagent closed"
        );
        Ok(previous)
    }

    /// Force-closes every live instance (running ones included) in spawn order.
    // Kept async for parity with the Python control surface; the tools await it.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn cancel_all(&self) -> Vec<SubAgentId> {
        let now_ms = self.clock.now_ms();
        let (ids, events) = {
            let mut state = self.shared.state.lock();
            let mut closed: Vec<(u64, SubAgentId, SubagentLifecycleEvent)> = Vec::new();
            for (id, instance) in &mut state.instances {
                if !instance.is_live() {
                    continue;
                }
                instance.close(now_ms, CLOSE_REASON_PARENT_ENDED);
                instance.record(TranscriptEntry::closed(
                    id,
                    CLOSE_REASON_PARENT_ENDED,
                    now_ms,
                ));
                instance
                    .activity_stream
                    .closed(CLOSE_REASON_PARENT_ENDED, now_ms);
                let event = instance.lifecycle_event(
                    LifecycleStage::Close,
                    Some(CLOSE_REASON_PARENT_ENDED),
                    now_ms,
                );
                closed.push((instance.seq, id.clone(), event));
            }
            closed.sort_by_key(|(seq, _, _)| *seq);
            let events: Vec<SubagentLifecycleEvent> =
                closed.iter().map(|(_, _, event)| event.clone()).collect();
            let ids = closed.into_iter().map(|(_, id, _)| id).collect();
            (ids, events)
        };
        self.shared.notify.notify_waiters();
        lifecycle_api::run_stage(&self.shared, LifecycleStage::Close, events).await;
        ids
    }

    /// Returns the Python `describe_list` payload; `subagents` aliases `live_subagents`.
    ///
    /// Every row carries the metadata fields (including `activity`, `depth`,
    /// `parent_subagent_id`, `last_used_at`, `closed_reason` and the `transcript` digest);
    /// `capacity` reports the live usage, the cumulative spawn ledger, the running-turn count
    /// against their limits, and `transcript_entries`, the number of buffered transcript
    /// entries across the live instances (`capacity.transcript_entries`).
    /// `summary` reports `live_count`, `closed_count`, `evicted_count` (closed instances whose
    /// `closed_reason` is `"evicted"`) and `eviction`, the policy in force:
    /// `{"enabled": <bool>, "candidates": <evictable live instances>}`.
    #[must_use]
    pub fn describe_list(&self) -> Value {
        let state = self.shared.state.lock();
        let mut instances: Vec<&Instance> = state.instances.values().collect();
        instances.sort_by_key(|instance| instance.seq);
        let running = instances
            .iter()
            .filter(|instance| instance.activity == SubagentActivity::Running)
            .count();
        let transcript_entries = instances
            .iter()
            .filter(|instance| instance.is_live())
            .map(|instance| instance.transcript.len())
            .sum::<usize>();
        let evicted_count = instances
            .iter()
            .filter(|instance| {
                !instance.is_live() && instance.closed_reason.as_deref() == Some(EVICTION_REASON)
            })
            .count();
        let candidates = instances
            .iter()
            .filter(|instance| instance.is_live() && instance.activity.is_evictable())
            .count();
        let (mut live, mut closed) = (Vec::new(), Vec::new());
        for instance in instances {
            let value = serde_json::to_value(instance.metadata()).unwrap_or(Value::Null);
            if instance.is_live() {
                live.push(value);
            } else {
                closed.push(value);
            }
        }
        let (live_count, closed_count) = (live.len(), closed.len());
        let live_alias = live.clone();
        json!({
            "capacity": {
                "used": live_count,
                "max": self.quota.max_subagents,
                "spawned": state.spawned,
                "max_spawned": self.quota.max_total_spawned,
                "running": running,
                "max_running": self.quota.max_concurrent_running,
                "transcript_entries": transcript_entries,
            },
            "subagents": live,
            "live_subagents": live_alias,
            "closed_subagents": closed,
            "summary": {
                "live_count": live_count,
                "closed_count": closed_count,
                "evicted_count": evicted_count,
                "eviction": {
                    "enabled": self.quota.enable_lru_eviction,
                    "candidates": candidates,
                },
            },
        })
    }

    /// Returns the parent session id this runtime belongs to.
    #[must_use]
    pub fn parent_session_id(&self) -> &str {
        &self.shared.parent_session_id
    }

    /// Returns the registry the runtime resolves sub-agent types from.
    ///
    /// The model-facing tools use it to reject an unavailable type before spawning.
    #[must_use]
    pub fn registry(&self) -> &Arc<SubAgentRegistry> {
        &self.shared.registry
    }

    fn outcome_for(&self, id: &str) -> SubAgentOutcome {
        let state = self.shared.state.lock();
        state.instances.get(id).map_or_else(
            || SubAgentOutcome {
                subagent_id: id.to_owned(),
                status: SubagentStatus::NotFound,
                ..SubAgentOutcome::default()
            },
            Instance::outcome,
        )
    }

    /// Builds the lifecycle snapshot of a known instance, stamped from the runtime clock.
    fn lifecycle_event_for(
        &self,
        id: &str,
        stage: LifecycleStage,
        reason: Option<&str>,
    ) -> Option<SubagentLifecycleEvent> {
        let now_ms = self.clock.now_ms();
        let state = self.shared.state.lock();
        state
            .instances
            .get(id)
            .map(|instance| instance.lifecycle_event(stage, reason, now_ms))
    }

    /// Returns the depth of a known instance.
    fn depth_of(&self, id: &str) -> Option<usize> {
        let state = self.shared.state.lock();
        state.instances.get(id).map(|instance| instance.depth)
    }

    /// Reserves the id and one capacity slot by inserting a `Queued` instance.
    ///
    /// `config` is the patched child configuration the instance keeps for persistence and
    /// resume. When the live cap is reached with [`SubAgentQuota::enable_lru_eviction`] set, the
    /// least-recently-used evictable instance is closed first (Python
    /// `SubagentControl._acquire_slot`); the requesting parent is never recycled. The
    /// cumulative, per-parent and depth limits come from [`SubAgentQuota`], and the live cap
    /// keeps its legacy error message when nothing can be evicted.
    #[allow(clippy::too_many_lines)]
    fn reserve(
        &self,
        id: &SubAgentId,
        request: &SubAgentSpawnRequest,
        config: SubAgentConfig,
        closed_events: &mut Vec<SubagentLifecycleEvent>,
    ) -> Result<u64> {
        let now_ms = self.clock.now_ms();
        let seq = {
            let mut state = self.shared.state.lock();
            if let Some(existing) = state.instances.get(id) {
                if existing.status != SubagentStatus::Closed {
                    return Err(Error::validation(format!(
                        "subagent already live: {id}; use subagent_wait to collect its result"
                    )));
                }
            }
            let parent = request.parent_subagent_id.as_deref();
            let depth = if let Some(parent) = parent {
                let Some(parent_instance) = state
                    .instances
                    .get(parent)
                    .filter(|instance| instance.is_live())
                else {
                    return Err(Error::validation(format!(
                        "subagent parent not found: {parent}"
                    )));
                };
                parent_instance.depth.saturating_add(1)
            } else {
                1
            };
            let (evicted, evicted_events) =
                evict::make_room(&mut state, &self.quota, now_ms, parent);
            closed_events.extend(evicted_events);
            let budget = SpawnBudget {
                live: evict::live_count(&state),
                spawned: state.spawned,
                parent_live: state
                    .instances
                    .values()
                    .filter(|instance| {
                        instance.is_live() && instance.parent_subagent_id.as_deref() == parent
                    })
                    .count(),
                depth,
            };
            let parent_label = parent.unwrap_or(self.shared.parent_session_id.as_str());
            if let Err(error) = self.quota.check_spawn(&budget, parent_label) {
                tracing::debug!(
                    target: "openjiuwen_harness::subagents",
                    subagent_id = %id,
                    live = budget.live,
                    spawned = budget.spawned,
                    parent_live = budget.parent_live,
                    depth = budget.depth,
                    %error,
                    "subagent spawn refused by quota"
                );
                if evicted > 0 {
                    // The early return skips the post-reservation notify; waking here keeps a
                    // `wait` registered on an evicted id from blocking on a stale snapshot.
                    self.shared.notify.notify_waiters();
                }
                return Err(error);
            }
            state.spawned = state.spawned.saturating_add(1);
            state.instances.remove(id);
            state.evicting.remove(id);
            let seq = state.next_seq;
            state.next_seq = state.next_seq.wrapping_add(1);
            state.instances.insert(
                id.clone(),
                Instance::new(seq, request, id.clone(), now_ms, depth, config, &self.quota),
            );
            seq
        };
        self.shared.notify.notify_waiters();
        Ok(seq)
    }

    /// Removes a reservation that could not be completed and frees its budget entry.
    fn rollback(&self, id: &str, seq: u64) {
        {
            let mut state = self.shared.state.lock();
            state.evicting.remove(id);
            if state
                .instances
                .get(id)
                .is_some_and(|instance| instance.seq == seq)
            {
                state.instances.remove(id);
                state.spawned = state.spawned.saturating_sub(1);
            }
        }
        self.shared.notify.notify_waiters();
    }

    /// Attaches the created agent to the reserved instance, if it is still pending.
    fn attach_agent(&self, id: &str, seq: u64, agent: &Arc<dyn Agent>) -> Option<(u64, String)> {
        let now_ms = self.clock.now_ms();
        let mut state = self.shared.state.lock();
        let instance = state.instances.get_mut(id)?;
        if instance.seq != seq || instance.status != SubagentStatus::PendingInit {
            return None;
        }
        instance.agent_id.clone_from(&agent.card().id);
        instance.agent = Some(Arc::clone(agent));
        instance.generation = instance.generation.wrapping_add(1);
        instance.running = true;
        instance.touch(now_ms, SubagentActivity::Queued);
        Some((instance.generation, instance.sub_session_id.clone()))
    }

    fn start_turn(
        &self,
        id: &str,
        generation: u64,
        query: String,
        agent: Arc<dyn Agent>,
        sub_session_id: String,
    ) {
        let runner = TurnRunner {
            shared: Arc::clone(&self.shared),
            semaphore: Arc::clone(&self.semaphore),
            clock: Arc::clone(&self.clock),
            turn_timeout: self.turn_timeout,
            subagent_id: id.to_owned(),
            generation,
            agent,
            sub_session_id,
        };
        let handle = tokio::spawn(runner.run(query));
        let mut state = self.shared.state.lock();
        if let Some(instance) = state.instances.get_mut(id) {
            if instance.generation == generation && instance.status != SubagentStatus::Closed {
                instance.handle = Some(handle);
            }
        }
    }
}

/// Builds the child session id (Python `build_subagent_id`).
fn build_subagent_id(parent_session_id: &str, subagent_type: &str, sticky: bool) -> SubAgentId {
    let normalized_type = subagent_type.trim();
    if sticky {
        return format!("{parent_session_id}_sub_{normalized_type}");
    }
    let hex = uuid::Uuid::new_v4().simple().to_string();
    let suffix = hex.get(..8).unwrap_or(hex.as_str());
    format!("{parent_session_id}_sub_{normalized_type}_{suffix}")
}

/// Whether `subagent_type` uses a sticky (unsuffixed) id.
fn is_sticky_type(subagent_type: &str) -> bool {
    subagent_type.trim() == STICKY_SUBAGENT_TYPE
}

/// Collapses duplicate ids while preserving input order.
fn dedupe(ids: &[SubAgentId]) -> Vec<SubAgentId> {
    let mut seen = HashSet::new();
    let mut targets = Vec::new();
    for id in ids {
        if seen.insert(id.clone()) {
            targets.push(id.clone());
        }
    }
    targets
}

/// Error for input sent to a closed or unknown instance.
fn closed_or_missing(id: &str) -> Error {
    Error::validation(format!(
        "subagent closed or not found: {id}; call subagent_resume first"
    ))
}

/// Error for input sent to an instance that has no child agent (a restored one).
fn not_resumed(id: &str) -> Error {
    Error::validation(format!(
        "subagent not resumed: {id}; call subagent_resume first"
    ))
}

/// Error for an unknown instance id.
fn not_found(id: &str) -> Error {
    Error::validation(format!("subagent not found: {id}"))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
