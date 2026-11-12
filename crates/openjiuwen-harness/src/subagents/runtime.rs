//! The in-process sub-agent runtime.
//!
//! Ports the control subset of `openjiuwen/harness/subagent_runtime/`: one parent session
//! owns a bounded set of live child instances, each running its turns serially on a
//! background `tokio` task. Children are built lazily by a [`SubAgentFactory`], so this
//! module never depends on [`crate::agent`]. Capacity is reserved before the child is
//! created and rolled back on failure; a [`tokio::sync::Semaphore`] bounds concurrent turns.
//! Python defaults (`config.py`): `max_subagents = 10`, `max_concurrent_running = 5`,
//! turn timeout 2400 s.

use std::collections::{BTreeMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use openjiuwen_agent::{Agent, AgentOutput, ResultType};
use openjiuwen_common::error::{Error, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::{Notify, Semaphore};
use tokio::task::JoinHandle;

use super::{SubAgentConfig, SubAgentRegistry};

/// Default live-instance cap (Python `max_subagents`).
const DEFAULT_MAX_SUBAGENTS: usize = 10;
/// Default concurrent-turn cap (Python `max_concurrent_running`).
const DEFAULT_MAX_CONCURRENT_RUNNING: usize = 5;
/// Default per-turn timeout (Python `TURN_TIMEOUT_S_DEFAULT`).
const DEFAULT_TURN_TIMEOUT: Duration = Duration::from_secs(2400);
/// The sub-agent type whose id is sticky (no random suffix) across spawns.
const STICKY_SUBAGENT_TYPE: &str = "verification_agent";

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
    state: Mutex<RuntimeState>,
}

/// Mutable runtime state, guarded by one [`Mutex`].
struct RuntimeState {
    instances: BTreeMap<SubAgentId, Instance>,
    next_seq: u64,
}

/// One live (or closed-tombstone) sub-agent instance.
struct Instance {
    seq: u64,
    subagent_id: SubAgentId,
    sub_session_id: String,
    subagent_type: String,
    display_name: Option<String>,
    role: Option<String>,
    task_description: String,
    status: SubagentStatus,
    created_at: String,
    updated_at: String,
    agent_id: String,
    output: Option<String>,
    error: Option<String>,
    agent: Option<Arc<dyn Agent>>,
    pending: VecDeque<String>,
    running: bool,
    generation: u64,
    handle: Option<JoinHandle<()>>,
}

impl Instance {
    fn new(seq: u64, request: &SubAgentSpawnRequest, id: SubAgentId, now: String) -> Self {
        Self {
            seq,
            sub_session_id: id.clone(),
            subagent_id: id,
            subagent_type: request.subagent_type.clone(),
            display_name: request.display_name.clone(),
            role: request.role.clone(),
            task_description: request.task_description.clone(),
            status: SubagentStatus::PendingInit,
            created_at: now.clone(),
            updated_at: now,
            agent_id: String::new(),
            output: None,
            error: None,
            agent: None,
            pending: VecDeque::new(),
            running: false,
            generation: 0,
            handle: None,
        }
    }

    fn outcome(&self) -> SubAgentOutcome {
        SubAgentOutcome {
            subagent_id: self.subagent_id.clone(),
            sub_session_id: self.sub_session_id.clone(),
            subagent_type: self.subagent_type.clone(),
            agent_id: self.agent_id.clone(),
            status: self.status,
            output: self.output.clone(),
            error: self.error.clone(),
        }
    }

    fn metadata(&self) -> SubagentMetadata {
        SubagentMetadata {
            subagent_id: self.subagent_id.clone(),
            sub_session_id: self.sub_session_id.clone(),
            subagent_type: self.subagent_type.clone(),
            display_name: self.display_name.clone(),
            role: self.role.clone(),
            task_description: self.task_description.clone(),
            status: self.status,
            created_at: self.created_at.clone(),
            updated_at: self.updated_at.clone(),
        }
    }
}

/// The sub-agent instances of one parent session; share it behind an `Arc`.
pub struct SubAgentRuntime {
    shared: Arc<Shared>,
    max_subagents: usize,
    turn_timeout: Duration,
    semaphore: Arc<Semaphore>,
}

impl SubAgentRuntime {
    /// Creates a runtime for `parent_session_id` over `registry` and `factory`.
    ///
    /// Defaults match Python: 10 live instances, 5 concurrent turns, 2400 s per turn.
    #[must_use]
    pub fn new(
        registry: Arc<SubAgentRegistry>,
        factory: Arc<dyn SubAgentFactory>,
        parent_session_id: impl Into<String>,
    ) -> Self {
        Self {
            shared: Arc::new(Shared {
                registry,
                factory,
                parent_session_id: parent_session_id.into(),
                notify: Notify::new(),
                state: Mutex::new(RuntimeState {
                    instances: BTreeMap::new(),
                    next_seq: 0,
                }),
            }),
            max_subagents: DEFAULT_MAX_SUBAGENTS,
            turn_timeout: DEFAULT_TURN_TIMEOUT,
            semaphore: Arc::new(Semaphore::new(DEFAULT_MAX_CONCURRENT_RUNNING)),
        }
    }

    /// Overrides the live-instance and concurrent-turn caps; call before sharing.
    #[must_use]
    pub fn with_limits(mut self, max_subagents: usize, max_concurrent_running: usize) -> Self {
        self.max_subagents = max_subagents;
        self.semaphore = Arc::new(Semaphore::new(max_concurrent_running));
        self
    }

    /// Overrides the per-turn timeout (default 2400 s).
    #[must_use]
    pub fn with_turn_timeout(mut self, timeout: Duration) -> Self {
        self.turn_timeout = timeout;
        self
    }

    /// Spawns a sub-agent instance and returns its id immediately.
    ///
    /// The first turn runs on a background task; use [`SubAgentRuntime::wait`] to collect
    /// its result. Capacity is reserved as a `PendingInit` instance before the child is
    /// created and rolled back when construction fails.
    ///
    /// # Errors
    ///
    /// - [`Error::Validation`] when the type is unknown or the id is already live;
    /// - [`Error::Config`] when the live-instance cap is reached;
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
        let seq = self.reserve(&id, &request)?;

        let mut config = spec.config;
        if request.display_name.is_some() {
            config.display_name = request.display_name.clone();
        }
        if request.role.is_some() {
            config.role = request.role.clone();
        }
        config.subagent_type = Some(request.subagent_type.clone());
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
            "subagent spawned"
        );
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

    /// Returns the metadata of every live (non-`Closed`) instance in spawn order.
    #[must_use]
    pub fn list(&self) -> Vec<SubagentMetadata> {
        let state = self.shared.state.lock();
        let mut live: Vec<&Instance> = state
            .instances
            .values()
            .filter(|instance| instance.status != SubagentStatus::Closed)
            .collect();
        live.sort_by_key(|instance| instance.seq);
        live.into_iter().map(Instance::metadata).collect()
    }

    /// Returns `(used, max)` where `used` counts live (non-`Closed`) instances.
    #[must_use]
    pub fn capacity(&self) -> (usize, usize) {
        let state = self.shared.state.lock();
        let used = state
            .instances
            .values()
            .filter(|instance| instance.status != SubagentStatus::Closed)
            .count();
        (used, self.max_subagents)
    }

    /// Queues follow-up input for `id` and returns once it is queued.
    ///
    /// When no turn is running the queued query starts immediately; otherwise the running
    /// turn picks it up (FIFO) after its current invocation. `interrupt` aborts the
    /// running turn first and drops any earlier queued input.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the id is unknown or already closed.
    // Kept async for parity with the Python control surface; the tools await it.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn send_input(&self, id: &str, query: &str, interrupt: bool) -> Result<()> {
        let mut start: Option<(u64, String, Arc<dyn Agent>, String)> = None;
        {
            let mut state = self.shared.state.lock();
            let Some(instance) = state.instances.get_mut(id) else {
                return Err(closed_or_missing(id));
            };
            if instance.status == SubagentStatus::Closed {
                return Err(closed_or_missing(id));
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
            instance.pending.push_back(query.to_owned());
            instance.updated_at = now_rfc3339();
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
        }
        self.shared.notify.notify_waiters();
        if let Some((generation, query, agent, sub_session_id)) = start {
            self.start_turn(id, generation, query, agent, sub_session_id);
        }
        Ok(())
    }

    /// Revives a closed instance so new input can be sent to it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the id was never registered.
    // Kept async for parity with the Python control surface; the tools await it.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn resume(&self, id: &str) -> Result<ResumeReport> {
        let report = {
            let mut state = self.shared.state.lock();
            let Some(instance) = state.instances.get_mut(id) else {
                return Err(not_found(id));
            };
            if instance.status != SubagentStatus::Closed {
                return Ok(ResumeReport {
                    status: instance.status,
                    restored: false,
                    message: Some(
                        "Instance is already live; use subagent_send_input directly.".to_owned(),
                    ),
                });
            }
            instance.status = SubagentStatus::Completed;
            instance.running = false;
            instance.generation = instance.generation.wrapping_add(1);
            instance.handle = None;
            instance.updated_at = now_rfc3339();
            ResumeReport {
                status: SubagentStatus::Completed,
                restored: true,
                message: None,
            }
        };
        self.shared.notify.notify_waiters();
        tracing::debug!(
            target: "openjiuwen_harness::subagents",
            subagent_id = %id,
            status = "completed",
            "subagent resumed"
        );
        Ok(report)
    }

    /// Closes `id` and returns its previous status.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the id is unknown or its turn is running.
    // Kept async for parity with the Python control surface; the tools await it.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn close(&self, id: &str) -> Result<SubagentStatus> {
        let previous = {
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
            if let Some(handle) = instance.handle.take() {
                handle.abort();
            }
            instance.running = false;
            instance.generation = instance.generation.wrapping_add(1);
            instance.status = SubagentStatus::Closed;
            instance.updated_at = now_rfc3339();
            previous
        };
        self.shared.notify.notify_waiters();
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
        let ids = {
            let mut state = self.shared.state.lock();
            let mut closed: Vec<(u64, SubAgentId)> = Vec::new();
            for (id, instance) in &mut state.instances {
                if instance.status == SubagentStatus::Closed {
                    continue;
                }
                if let Some(handle) = instance.handle.take() {
                    handle.abort();
                }
                instance.running = false;
                instance.generation = instance.generation.wrapping_add(1);
                instance.status = SubagentStatus::Closed;
                instance.updated_at = now_rfc3339();
                closed.push((instance.seq, id.clone()));
            }
            closed.sort_by_key(|(seq, _)| *seq);
            closed.into_iter().map(|(_, id)| id).collect()
        };
        self.shared.notify.notify_waiters();
        ids
    }

    /// Returns the Python `describe_list` payload; `subagents` aliases `live_subagents`.
    #[must_use]
    pub fn describe_list(&self) -> Value {
        let state = self.shared.state.lock();
        let mut instances: Vec<&Instance> = state.instances.values().collect();
        instances.sort_by_key(|instance| instance.seq);
        let (mut live, mut closed) = (Vec::new(), Vec::new());
        for instance in instances {
            let value = serde_json::to_value(instance.metadata()).unwrap_or(Value::Null);
            if instance.status == SubagentStatus::Closed {
                closed.push(value);
            } else {
                live.push(value);
            }
        }
        let (live_count, closed_count) = (live.len(), closed.len());
        let live_alias = live.clone();
        json!({
            "capacity": {"used": live_count, "max": self.max_subagents},
            "subagents": live,
            "live_subagents": live_alias,
            "closed_subagents": closed,
            "summary": {"live_count": live_count, "closed_count": closed_count},
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

    /// Reserves the id and one capacity slot by inserting a `PendingInit` instance.
    fn reserve(&self, id: &SubAgentId, request: &SubAgentSpawnRequest) -> Result<u64> {
        let seq = {
            let mut state = self.shared.state.lock();
            if let Some(existing) = state.instances.get(id) {
                if existing.status != SubagentStatus::Closed {
                    return Err(Error::validation(format!(
                        "subagent already live: {id}; use subagent_wait to collect its result"
                    )));
                }
            }
            let used = state
                .instances
                .values()
                .filter(|instance| instance.status != SubagentStatus::Closed)
                .count();
            if used >= self.max_subagents {
                return Err(Error::config(format!(
                    "subagent capacity exceeded: used={used}, limit={}",
                    self.max_subagents
                )));
            }
            state.instances.remove(id);
            let seq = state.next_seq;
            state.next_seq = state.next_seq.wrapping_add(1);
            state.instances.insert(
                id.clone(),
                Instance::new(seq, request, id.clone(), now_rfc3339()),
            );
            seq
        };
        self.shared.notify.notify_waiters();
        Ok(seq)
    }

    /// Removes a reservation that could not be completed.
    fn rollback(&self, id: &str, seq: u64) {
        {
            let mut state = self.shared.state.lock();
            if state
                .instances
                .get(id)
                .is_some_and(|instance| instance.seq == seq)
            {
                state.instances.remove(id);
            }
        }
        self.shared.notify.notify_waiters();
    }

    /// Attaches the created agent to the reserved instance, if it is still pending.
    fn attach_agent(&self, id: &str, seq: u64, agent: &Arc<dyn Agent>) -> Option<(u64, String)> {
        let mut state = self.shared.state.lock();
        let instance = state.instances.get_mut(id)?;
        if instance.seq != seq || instance.status != SubagentStatus::PendingInit {
            return None;
        }
        instance.agent_id.clone_from(&agent.card().id);
        instance.agent = Some(Arc::clone(agent));
        instance.generation = instance.generation.wrapping_add(1);
        instance.running = true;
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

/// One background turn worker; it runs queued queries of one instance serially.
struct TurnRunner {
    shared: Arc<Shared>,
    semaphore: Arc<Semaphore>,
    turn_timeout: Duration,
    subagent_id: String,
    generation: u64,
    agent: Arc<dyn Agent>,
    sub_session_id: String,
}

impl TurnRunner {
    async fn run(self, first_query: String) {
        let mut next_query = Some(first_query);
        while let Some(query) = next_query.take() {
            let permit = match self.semaphore.acquire().await {
                Ok(permit) => permit,
                Err(error) => {
                    tracing::debug!(%error, "subagent turn semaphore closed");
                    return;
                }
            };
            if !begin_turn(&self.shared, &self.subagent_id, self.generation) {
                return;
            }
            let inputs = json!({"query": query, "conversation_id": self.sub_session_id});
            let result =
                tokio::time::timeout(self.turn_timeout, self.agent.invoke(inputs, None)).await;
            drop(permit);
            let (status, output, error) = map_turn_result(result, self.turn_timeout);
            next_query = finish_turn(
                &self.shared,
                &self.subagent_id,
                self.generation,
                status,
                output,
                error,
            );
        }
    }
}

/// Marks the instance `Running` if the turn still owns it.
fn begin_turn(shared: &Shared, id: &str, generation: u64) -> bool {
    let started = {
        let mut state = shared.state.lock();
        match state.instances.get_mut(id) {
            Some(instance)
                if instance.generation == generation
                    && instance.status != SubagentStatus::Closed =>
            {
                instance.status = SubagentStatus::Running;
                instance.updated_at = now_rfc3339();
                true
            }
            _ => false,
        }
    };
    if started {
        shared.notify.notify_waiters();
        tracing::debug!(target: "openjiuwen_harness::subagents", subagent_id = %id, status = "running", "subagent turn started");
    }
    started
}

/// Stores the turn result and returns the next queued query, if any.
fn finish_turn(
    shared: &Shared,
    id: &str,
    generation: u64,
    status: SubagentStatus,
    output: Option<String>,
    error: Option<String>,
) -> Option<String> {
    let next = {
        let mut state = shared.state.lock();
        let instance = match state.instances.get_mut(id) {
            Some(instance)
                if instance.generation == generation
                    && instance.status != SubagentStatus::Closed =>
            {
                instance
            }
            _ => return None,
        };
        instance.output = output;
        instance.error = error;
        instance.updated_at = now_rfc3339();
        if let Some(query) = instance.pending.pop_front() {
            instance.status = SubagentStatus::PendingInit;
            Some(query)
        } else {
            instance.status = status;
            instance.running = false;
            tracing::debug!(target: "openjiuwen_harness::subagents", subagent_id = %id, status = status.as_str(), "subagent turn finished");
            None
        }
    };
    shared.notify.notify_waiters();
    next
}

/// Maps one invocation result to the next status and its payload.
fn map_turn_result(
    result: std::result::Result<Result<AgentOutput>, tokio::time::error::Elapsed>,
    turn_timeout: Duration,
) -> (SubagentStatus, Option<String>, Option<String>) {
    match result {
        Ok(Ok(output)) => match output.result_type {
            ResultType::Answer => (SubagentStatus::Completed, Some(output.output), None),
            ResultType::Error => (SubagentStatus::Errored, None, Some(output.output)),
            ResultType::Interrupt => (SubagentStatus::Interrupted, None, None),
        },
        Ok(Err(error)) => (SubagentStatus::Errored, None, Some(error.to_string())),
        Err(_elapsed) => (
            SubagentStatus::Errored,
            None,
            Some(format!(
                "subagent turn timed out after {}s",
                turn_timeout.as_secs()
            )),
        ),
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

/// Returns the current UTC time as an RFC 3339 string with millisecond precision.
fn now_rfc3339() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
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

/// Error for an unknown instance id.
fn not_found(id: &str) -> Error {
    Error::validation(format!("subagent not found: {id}"))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use futures::stream::BoxStream;
    use openjiuwen_agent::{AgentCard, AgentChunk};
    use openjiuwen_session::Session;
    use tokio::sync::watch;

    use super::*;
    use crate::subagents::SubAgentSpec;

    type Recorded = Arc<Mutex<Vec<(SubAgentConfig, String)>>>;
    type Calls = Arc<Mutex<Vec<String>>>;

    enum Mode {
        Echo,
        ErrorResult,
        Blocked(watch::Sender<bool>),
    }

    struct FakeAgent {
        card: AgentCard,
        calls: Calls,
        mode: Mode,
    }

    #[async_trait::async_trait]
    impl Agent for FakeAgent {
        fn card(&self) -> &AgentCard {
            &self.card
        }

        async fn invoke(
            &self,
            inputs: Value,
            _session: Option<Arc<dyn Session>>,
        ) -> Result<AgentOutput> {
            let query = inputs
                .get("query")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            self.calls.lock().push(query.clone());
            match &self.mode {
                Mode::Echo => Ok(AgentOutput::answer(format!("echo:{query}"))),
                Mode::ErrorResult => Ok(AgentOutput::error("model failed")),
                Mode::Blocked(gate) => {
                    let mut gate = gate.subscribe();
                    if !*gate.borrow_and_update() {
                        let _ = gate.changed().await;
                    }
                    Ok(AgentOutput::answer(format!("echo:{query}")))
                }
            }
        }

        fn stream(
            &self,
            _inputs: Value,
            _session: Option<Arc<dyn Session>>,
        ) -> BoxStream<'static, Result<AgentChunk>> {
            Box::pin(futures::stream::empty())
        }
    }

    enum Behavior {
        Echo,
        Fail,
        ErrorResult,
        Blocked(watch::Sender<bool>),
    }

    struct FakeFactory {
        recorded: Recorded,
        calls: Calls,
        behavior: Behavior,
    }

    impl FakeFactory {
        fn new(behavior: Behavior) -> Arc<Self> {
            Arc::new(Self {
                recorded: Arc::default(),
                calls: Arc::default(),
                behavior,
            })
        }
    }

    #[async_trait::async_trait]
    impl SubAgentFactory for FakeFactory {
        async fn create(
            &self,
            config: &SubAgentConfig,
            session_id: &str,
        ) -> Result<Arc<dyn Agent>> {
            self.recorded
                .lock()
                .push((config.clone(), session_id.to_owned()));
            let mode = match &self.behavior {
                Behavior::Echo => Mode::Echo,
                Behavior::Fail => return Err(Error::config("boom")),
                Behavior::ErrorResult => Mode::ErrorResult,
                Behavior::Blocked(gate) => Mode::Blocked(gate.clone()),
            };
            Ok(Arc::new(FakeAgent {
                card: config.agent_card.clone(),
                calls: Arc::clone(&self.calls),
                mode,
            }))
        }
    }

    fn build_runtime(names: &[&str], factory: Arc<FakeFactory>) -> SubAgentRuntime {
        let registry = SubAgentRegistry::new();
        for name in names {
            registry.register(SubAgentSpec::new(SubAgentConfig::new(AgentCard::new(
                *name,
            ))));
        }
        SubAgentRuntime::new(Arc::new(registry), factory, "parent-session")
    }

    async fn spawn(runtime: &SubAgentRuntime, subagent_type: &str, task: &str) -> SubAgentId {
        runtime
            .spawn(SubAgentSpawnRequest::new(subagent_type, task))
            .await
            .unwrap()
    }

    async fn wait_one(runtime: &SubAgentRuntime, id: &str) -> SubAgentOutcome {
        runtime
            .wait(&[id.to_owned()], Duration::from_secs(5))
            .await
            .pop()
            .unwrap()
    }

    async fn wait_running(runtime: &SubAgentRuntime, id: &str) {
        for _ in 0..200 {
            if runtime.get_status(id) == SubagentStatus::Running {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!("subagent did not start: {id}");
    }

    fn blocked() -> (watch::Sender<bool>, Arc<FakeFactory>) {
        let (gate, _receiver) = watch::channel(false);
        (gate.clone(), FakeFactory::new(Behavior::Blocked(gate)))
    }

    #[tokio::test]
    async fn spawn_runs_a_turn_to_completion() {
        let factory = FakeFactory::new(Behavior::Echo);
        let runtime = build_runtime(&["worker"], Arc::clone(&factory));
        let id = spawn(&runtime, "worker", "hello").await;
        assert!(matches!(
            runtime.get_status(&id),
            SubagentStatus::PendingInit | SubagentStatus::Running
        ));
        let outcome = wait_one(&runtime, &id).await;
        assert_eq!(outcome.status, SubagentStatus::Completed);
        assert_eq!(outcome.output.as_deref(), Some("echo:hello"));
        assert_eq!(outcome.sub_session_id, id);
        assert_eq!(outcome.subagent_type, "worker");
        let recorded = factory.recorded.lock().clone();
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].1, id);
        assert_eq!(recorded[0].0.subagent_type.as_deref(), Some("worker"));
        let stored = runtime.get_outcome(&id).unwrap();
        assert_eq!(stored.agent_id, recorded[0].0.agent_card.id);
        assert_eq!(stored.status, SubagentStatus::Completed);
        let metadata = runtime.list();
        assert_eq!(metadata.len(), 1);
        assert_eq!(metadata[0].subagent_id, id);
        assert_eq!(metadata[0].task_description, "hello");
        assert_eq!(metadata[0].status, SubagentStatus::Completed);
        assert_ne!(metadata[0].created_at, "");
        assert_eq!(runtime.capacity(), (1, 10));
    }

    #[tokio::test]
    async fn child_session_id_format_and_overrides() {
        let factory = FakeFactory::new(Behavior::Echo);
        let runtime = build_runtime(&["worker", "verification_agent"], Arc::clone(&factory));
        let generated = spawn(&runtime, "worker", "task").await;
        let suffix = generated
            .strip_prefix("parent-session_sub_worker_")
            .unwrap();
        assert_eq!(suffix.len(), 8);
        assert!(suffix
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
        wait_one(&runtime, &generated).await;
        let sticky = spawn(&runtime, "verification_agent", "verify").await;
        assert_eq!(sticky, "parent-session_sub_verification_agent");
        let mut request = SubAgentSpawnRequest::new("worker", "custom");
        request.subagent_id = Some("explicit-id".to_owned());
        assert_eq!(runtime.spawn(request).await.unwrap(), "explicit-id");
        let recorded = factory.recorded.lock().clone();
        let sessions: Vec<&str> = recorded
            .iter()
            .map(|(_, session)| session.as_str())
            .collect();
        assert_eq!(
            sessions,
            vec![generated.as_str(), sticky.as_str(), "explicit-id"]
        );
    }

    #[tokio::test]
    async fn spawn_validation_and_rollback() {
        let factory = FakeFactory::new(Behavior::Echo);
        let runtime = build_runtime(&["worker"], Arc::clone(&factory));
        let error = runtime
            .spawn(SubAgentSpawnRequest::new("ghost", "task"))
            .await
            .unwrap_err();
        assert!(
            matches!(&error, Error::Validation(message) if message.contains("subagent type not found"))
        );
        let mut request = SubAgentSpawnRequest::new("worker", "first");
        request.subagent_id = Some("dup".to_owned());
        let id = runtime.spawn(request.clone()).await.unwrap();
        wait_one(&runtime, &id).await;
        let error = runtime.spawn(request).await.unwrap_err();
        assert!(
            matches!(&error, Error::Validation(message) if message.contains("subagent already live"))
        );

        let failing = FakeFactory::new(Behavior::Fail);
        let runtime = build_runtime(&["worker"], failing);
        let mut request = SubAgentSpawnRequest::new("worker", "task");
        request.subagent_id = Some("will-fail".to_owned());
        let error = runtime.spawn(request).await.unwrap_err();
        assert!(matches!(&error, Error::Config(message) if message == "boom"));
        assert_eq!(runtime.capacity(), (0, 10));
        assert_eq!(runtime.get_status("will-fail"), SubagentStatus::NotFound);
        assert_eq!(runtime.list().len(), 0);
    }

    #[tokio::test]
    async fn capacity_limit_and_slot_reuse() {
        let factory = FakeFactory::new(Behavior::Echo);
        let runtime = build_runtime(&["worker"], factory).with_limits(1, 1);
        let first = spawn(&runtime, "worker", "one").await;
        wait_one(&runtime, &first).await;
        let error = runtime
            .spawn(SubAgentSpawnRequest::new("worker", "two"))
            .await
            .unwrap_err();
        assert!(
            matches!(&error, Error::Config(message) if message == "subagent capacity exceeded: used=1, limit=1")
        );
        assert_eq!(runtime.capacity(), (1, 1));
        assert_eq!(
            runtime.close(&first).await.unwrap(),
            SubagentStatus::Completed
        );
        assert_eq!(runtime.capacity(), (0, 1));
        assert_ne!(spawn(&runtime, "worker", "two").await, first);
    }

    #[tokio::test]
    async fn wait_times_out_while_the_agent_is_blocked() {
        let (gate, factory) = blocked();
        let runtime = build_runtime(&["worker"], factory);
        let id = spawn(&runtime, "worker", "task").await;
        let outcomes = runtime
            .wait(std::slice::from_ref(&id), Duration::from_millis(50))
            .await;
        assert_eq!(outcomes.len(), 1);
        assert!(!outcomes[0].is_final());
        gate.send_replace(true);
        let outcome = wait_one(&runtime, &id).await;
        assert_eq!(outcome.status, SubagentStatus::Completed);
        assert_eq!(outcome.output.as_deref(), Some("echo:task"));
    }

    #[tokio::test]
    async fn error_output_maps_to_errored() {
        let runtime = build_runtime(&["worker"], FakeFactory::new(Behavior::ErrorResult));
        let id = spawn(&runtime, "worker", "task").await;
        let outcome = wait_one(&runtime, &id).await;
        assert_eq!(outcome.status, SubagentStatus::Errored);
        assert_eq!(outcome.error.as_deref(), Some("model failed"));
        assert!(outcome.output.is_none());
    }

    #[tokio::test]
    async fn close_refuses_running_then_resume_revives() {
        let (gate, factory) = blocked();
        let runtime = build_runtime(&["worker"], factory);
        let id = spawn(&runtime, "worker", "task").await;
        wait_running(&runtime, &id).await;
        let error = runtime.close(&id).await.unwrap_err();
        assert!(
            matches!(&error, Error::Validation(message) if message.starts_with("cannot close running subagent:"))
        );
        gate.send_replace(true);
        wait_one(&runtime, &id).await;
        assert_eq!(runtime.close(&id).await.unwrap(), SubagentStatus::Completed);
        assert_eq!(runtime.close(&id).await.unwrap(), SubagentStatus::Closed);
        let report = runtime.resume(&id).await.unwrap();
        assert!(report.restored);
        assert_eq!(report.status, SubagentStatus::Completed);
        assert!(report.message.is_none());
        let live = runtime.resume(&id).await.unwrap();
        assert!(!live.restored);
        assert_eq!(
            live.message.as_deref(),
            Some("Instance is already live; use subagent_send_input directly.")
        );
        let error = runtime.resume("ghost").await.unwrap_err();
        assert!(
            matches!(&error, Error::Validation(message) if message.contains("subagent not found"))
        );
    }

    #[tokio::test]
    async fn send_input_validates_closed_and_unknown_ids() {
        let runtime = build_runtime(&["worker"], FakeFactory::new(Behavior::Echo));
        let error = runtime
            .send_input("ghost", "query", false)
            .await
            .unwrap_err();
        assert!(
            matches!(&error, Error::Validation(message) if message.contains("subagent closed or not found"))
        );
        let id = spawn(&runtime, "worker", "task").await;
        wait_one(&runtime, &id).await;
        runtime.close(&id).await.unwrap();
        let error = runtime.send_input(&id, "query", false).await.unwrap_err();
        assert!(
            matches!(&error, Error::Validation(message) if message.contains("subagent closed or not found"))
        );
    }

    #[tokio::test]
    async fn send_input_after_completion_starts_a_new_turn() {
        let factory = FakeFactory::new(Behavior::Echo);
        let calls = Arc::clone(&factory.calls);
        let runtime = build_runtime(&["worker"], factory);
        let id = spawn(&runtime, "worker", "first").await;
        wait_one(&runtime, &id).await;
        runtime.send_input(&id, "second", false).await.unwrap();
        assert_eq!(runtime.get_status(&id), SubagentStatus::PendingInit);
        assert_eq!(
            wait_one(&runtime, &id).await.output.as_deref(),
            Some("echo:second")
        );
        assert_eq!(*calls.lock(), vec!["first".to_owned(), "second".to_owned()]);
    }

    #[tokio::test]
    async fn queued_inputs_run_in_fifo_order() {
        let (gate, factory) = blocked();
        let calls = Arc::clone(&factory.calls);
        let runtime = build_runtime(&["worker"], factory);
        let id = spawn(&runtime, "worker", "task").await;
        wait_running(&runtime, &id).await;
        runtime.send_input(&id, "one", false).await.unwrap();
        runtime.send_input(&id, "two", false).await.unwrap();
        gate.send_replace(true);
        assert_eq!(
            wait_one(&runtime, &id).await.output.as_deref(),
            Some("echo:two")
        );
        assert_eq!(
            *calls.lock(),
            vec!["task".to_owned(), "one".to_owned(), "two".to_owned()]
        );
    }

    #[tokio::test]
    async fn interrupt_aborts_the_running_turn_and_runs_the_queued_query() {
        let (gate, factory) = blocked();
        let calls = Arc::clone(&factory.calls);
        let runtime = build_runtime(&["worker"], factory);
        let id = spawn(&runtime, "worker", "original").await;
        wait_running(&runtime, &id).await;
        runtime.send_input(&id, "next", true).await.unwrap();
        assert_eq!(runtime.get_status(&id), SubagentStatus::Interrupted);
        gate.send_replace(true);
        let outcome = wait_one(&runtime, &id).await;
        assert_eq!(outcome.status, SubagentStatus::Completed);
        assert_eq!(outcome.output.as_deref(), Some("echo:next"));
        assert_eq!(
            *calls.lock(),
            vec!["original".to_owned(), "next".to_owned()]
        );
    }

    #[tokio::test]
    async fn cancel_all_closes_live_instances_and_describe_list_separates_them() {
        let (_gate, factory) = blocked();
        let runtime = build_runtime(&["worker"], factory);
        let first = spawn(&runtime, "worker", "one").await;
        let second = spawn(&runtime, "worker", "two").await;
        wait_running(&runtime, &first).await;
        wait_running(&runtime, &second).await;
        let live = runtime.describe_list();
        assert_eq!(live["summary"]["live_count"], json!(2));
        assert_eq!(live["subagents"].as_array().map(Vec::len), Some(2));
        assert_eq!(live["live_subagents"].as_array().map(Vec::len), Some(2));
        assert_eq!(live["closed_subagents"].as_array().map(Vec::len), Some(0));
        assert_eq!(live["capacity"]["max"], json!(10));
        assert_eq!(
            runtime.cancel_all().await,
            vec![first.clone(), second.clone()]
        );
        assert_eq!(runtime.get_status(&first), SubagentStatus::Closed);
        assert_eq!(runtime.get_status(&second), SubagentStatus::Closed);
        let list = runtime.describe_list();
        assert_eq!(list["capacity"]["used"], json!(0));
        assert_eq!(list["subagents"].as_array().map(Vec::len), Some(0));
        assert_eq!(list["live_subagents"].as_array().map(Vec::len), Some(0));
        assert_eq!(list["closed_subagents"].as_array().map(Vec::len), Some(2));
        assert_eq!(list["summary"], json!({"live_count": 0, "closed_count": 2}));
    }

    #[tokio::test]
    async fn unknown_ids_report_not_found() {
        let runtime = build_runtime(&["worker"], FakeFactory::new(Behavior::Echo));
        assert_eq!(runtime.get_status("ghost"), SubagentStatus::NotFound);
        assert!(runtime.get_outcome("ghost").is_none());
        let outcomes = runtime
            .wait(&["ghost".to_owned()], Duration::from_millis(50))
            .await;
        assert_eq!(outcomes.len(), 1);
        assert_eq!(outcomes[0].subagent_id, "ghost");
        assert_eq!(outcomes[0].status, SubagentStatus::NotFound);
        assert!(outcomes[0].is_final());
        assert_eq!(runtime.parent_session_id(), "parent-session");
    }

    #[tokio::test]
    async fn duplicate_wait_ids_are_collapsed() {
        let runtime = build_runtime(&["worker"], FakeFactory::new(Behavior::Echo));
        let id = spawn(&runtime, "worker", "task").await;
        let outcomes = runtime
            .wait(&[id.clone(), id.clone()], Duration::from_secs(5))
            .await;
        assert_eq!(outcomes.len(), 1);
        assert_eq!(outcomes[0].status, SubagentStatus::Completed);
    }

    #[test]
    fn status_wire_strings_and_finality() {
        let cases = [
            ("pending_init", SubagentStatus::PendingInit, false),
            ("running", SubagentStatus::Running, false),
            ("completed", SubagentStatus::Completed, true),
            ("interrupted", SubagentStatus::Interrupted, false),
            ("errored", SubagentStatus::Errored, true),
            ("closed", SubagentStatus::Closed, true),
            ("not_found", SubagentStatus::NotFound, true),
        ];
        for (wire, status, final_) in cases {
            assert_eq!(status.as_str(), wire);
            assert_eq!(status.is_final(), final_);
        }
        let outcome = SubAgentOutcome {
            status: SubagentStatus::Completed,
            ..SubAgentOutcome::default()
        };
        assert!(outcome.is_final());
    }
}
