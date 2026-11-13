//! One sub-agent instance: its identity, quota bookkeeping, and lifecycle timestamps.
//!
//! The runtime owns instances inside its state lock and mutates them only through the helpers
//! here, so every state transition also bumps the activity phase and the timestamps together.
//! Timestamps come from the injected [`super::clock::Clock`]: `created_at`/`updated_at` keep
//! the RFC 3339 strings of the original API, while the numeric fields drive activity and LRU.
use std::collections::VecDeque;
use std::sync::Arc;

use openjiuwen_agent::Agent;
use tokio::task::JoinHandle;

use super::clock::rfc3339_from_ms;
use super::quota::SubagentActivity;
use super::{SubAgentId, SubAgentOutcome, SubAgentSpawnRequest, SubagentMetadata, SubagentStatus};

/// One live (or closed-tombstone) sub-agent instance.
pub(super) struct Instance {
    /// Monotonic spawn sequence; orders live instances.
    pub(super) seq: u64,
    /// Instance id, equal to the child session id.
    pub(super) subagent_id: SubAgentId,
    /// Child session id handed to the child agent.
    pub(super) sub_session_id: String,
    /// Registered sub-agent type.
    pub(super) subagent_type: String,
    /// Human-readable name of the instance.
    pub(super) display_name: Option<String>,
    /// Role description of the instance.
    pub(super) role: Option<String>,
    /// The task the instance was spawned with.
    pub(super) task_description: String,
    /// Lifecycle status (`SubagentStatus`).
    pub(super) status: SubagentStatus,
    /// Derived activity phase used by LRU eviction.
    pub(super) activity: SubagentActivity,
    /// Nesting depth (top-level = 1).
    pub(super) depth: usize,
    /// The instance that spawned this one, when nested.
    pub(super) parent_subagent_id: Option<String>,
    /// Creation time (RFC 3339), derived from `created_at_ms`.
    pub(super) created_at: String,
    /// Last transition time (RFC 3339), derived from `updated_at_ms`.
    pub(super) updated_at: String,
    /// Creation time (milliseconds since the Unix epoch).
    pub(super) created_at_ms: i64,
    /// Last transition time (milliseconds since the Unix epoch).
    pub(super) updated_at_ms: i64,
    /// Last time the instance was handed work (milliseconds since the Unix epoch).
    pub(super) last_used_at_ms: i64,
    /// Last activity time (milliseconds since the Unix epoch); drives LRU order.
    pub(super) last_activity_at_ms: i64,
    /// Why the instance was closed, when it was.
    pub(super) closed_reason: Option<String>,
    /// Identity card id of the child agent once created.
    pub(super) agent_id: String,
    /// Answer of the last finished turn.
    pub(super) output: Option<String>,
    /// Error of the last failed turn.
    pub(super) error: Option<String>,
    /// The child agent, once the factory created it.
    pub(super) agent: Option<Arc<dyn Agent>>,
    /// FIFO of queued follow-up queries.
    pub(super) pending: VecDeque<String>,
    /// Whether a turn currently owns the instance.
    pub(super) running: bool,
    /// Turn generation; a bump invalidates the outputs of an older turn.
    pub(super) generation: u64,
    /// Handle of the running turn task, when any.
    pub(super) handle: Option<JoinHandle<()>>,
}

impl Instance {
    /// Reserves a `Queued` instance at `depth` for `request`, stamped at `now_ms`.
    pub(super) fn new(
        seq: u64,
        request: &SubAgentSpawnRequest,
        id: SubAgentId,
        now_ms: i64,
        depth: usize,
    ) -> Self {
        let now = rfc3339_from_ms(now_ms);
        Self {
            seq,
            sub_session_id: id.clone(),
            subagent_id: id,
            subagent_type: request.subagent_type.clone(),
            display_name: request.display_name.clone(),
            role: request.role.clone(),
            task_description: request.task_description.clone(),
            status: SubagentStatus::PendingInit,
            activity: SubagentActivity::Queued,
            depth,
            parent_subagent_id: request.parent_subagent_id.clone(),
            created_at: now.clone(),
            updated_at: now,
            created_at_ms: now_ms,
            updated_at_ms: now_ms,
            last_used_at_ms: now_ms,
            last_activity_at_ms: now_ms,
            closed_reason: None,
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

    /// Whether the instance still occupies a capacity slot.
    pub(super) fn is_live(&self) -> bool {
        self.status != SubagentStatus::Closed
    }

    /// Stamps a transition to `activity` at `now_ms`.
    ///
    /// `updated_at`, `last_used_at_ms` and `last_activity_at_ms` all follow the clock, so a
    /// [`super::clock::ManualClock`] makes the whole lifecycle deterministic in tests.
    pub(super) fn touch(&mut self, now_ms: i64, activity: SubagentActivity) {
        self.activity = activity;
        self.updated_at_ms = now_ms;
        self.updated_at = rfc3339_from_ms(self.updated_at_ms);
        self.last_used_at_ms = now_ms;
        self.last_activity_at_ms = self.updated_at_ms;
    }

    /// Marks the instance closed for `reason`, aborting its turn task.
    pub(super) fn close(&mut self, now_ms: i64, reason: &str) {
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
        self.status = SubagentStatus::Closed;
        self.closed_reason = Some(reason.to_owned());
        self.running = false;
        self.generation = self.generation.wrapping_add(1);
        self.touch(now_ms, SubagentActivity::Closed);
    }

    /// Returns the collected result of this instance.
    pub(super) fn outcome(&self) -> SubAgentOutcome {
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

    /// Returns the registry metadata of this instance.
    pub(super) fn metadata(&self) -> SubagentMetadata {
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
            activity: self.activity,
            depth: self.depth,
            parent_subagent_id: self.parent_subagent_id.clone(),
            last_used_at: rfc3339_from_ms(self.last_used_at_ms),
            closed_reason: self.closed_reason.clone(),
        }
    }
}
