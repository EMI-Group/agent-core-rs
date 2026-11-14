//! One sub-agent instance: its identity, quota bookkeeping, and lifecycle timestamps.
//!
//! The runtime owns instances inside its state lock and mutates them only through the helpers
//! here, so every state transition also bumps the activity phase and the timestamps together.
//! Timestamps come from the injected [`super::clock::Clock`]: `created_at`/`updated_at` keep
//! the RFC 3339 strings of the original API, while the numeric fields drive activity and LRU.
//! `config` keeps the configuration the child was built from, so [`Instance::persisted`] can
//! write a restorable record and [`Instance::restore`] can rebuild one without a child agent.
use std::collections::VecDeque;
use std::sync::Arc;

use openjiuwen_agent::Agent;
use tokio::task::JoinHandle;

use super::activity::{ActivityStream, ActivityThrottle};
use super::clock::rfc3339_from_ms;
use super::lifecycle::{LifecycleStage, SubagentLifecycleEvent};
use super::persistence::PersistedSubAgentRecord;
use super::quota::{SubAgentQuota, SubagentActivity};
use super::transcript::{TranscriptBuffer, TranscriptChannel, TranscriptEntry};
use super::{SubAgentId, SubAgentOutcome, SubAgentSpawnRequest, SubagentMetadata, SubagentStatus};
use crate::schema::SubAgentConfig;

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
    /// The configuration the child agent was built from; persistence and resume use it.
    pub(super) config: SubAgentConfig,
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
    /// Bounded transcript of everything that happened to this instance.
    pub(super) transcript: TranscriptBuffer,
    /// Live feed of the entries appended to `transcript`.
    pub(super) channel: TranscriptChannel,
    /// Bounded activity recorder and feed of this instance (disabled by
    /// [`SubAgentQuota::enable_activity_stream`]).
    pub(super) activity_stream: ActivityStream,
}

impl Instance {
    /// Reserves a `Queued` instance at `depth` for `request` and `config`, stamped at `now_ms`.
    ///
    /// The transcript ring, its live feed and the activity stream are sized and enabled from
    /// `quota`. The reservation records the first transcript entry,
    /// [`TranscriptEntry::spawn`] with the delegated task description, and then the matching
    /// [`ActivityStream::spawn`] frame.
    #[allow(clippy::too_many_arguments)] // one reservation record, straight from the runtime
    pub(super) fn new(
        seq: u64,
        request: &SubAgentSpawnRequest,
        id: SubAgentId,
        now_ms: i64,
        depth: usize,
        config: SubAgentConfig,
        quota: &SubAgentQuota,
    ) -> Self {
        let now = rfc3339_from_ms(now_ms);
        let spawn_entry =
            TranscriptEntry::spawn(id.as_str(), request.task_description.clone(), now_ms);
        let stream_id = id.clone();
        let mut instance = Self {
            seq,
            sub_session_id: id.clone(),
            subagent_id: id,
            subagent_type: request.subagent_type.clone(),
            display_name: request.display_name.clone(),
            role: request.role.clone(),
            task_description: request.task_description.clone(),
            config,
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
            transcript: TranscriptBuffer::new(quota.transcript_capacity),
            channel: TranscriptChannel::new(quota.transcript_channel_capacity),
            activity_stream: recorder(&stream_id, quota),
        };
        instance.record(spawn_entry);
        instance
            .activity_stream
            .spawn(&request.task_description, now_ms);
        instance
    }

    /// Rebuilds an instance from a persisted `record` under the fresh sequence `seq`.
    ///
    /// `id` is the bucket key the record was stored under and becomes the instance id; a record
    /// without a `sub_session_id` falls back to it. Everything else is the persisted identity,
    /// configuration, status/activity, transcript and timestamps, with two rebuild-safe folds,
    /// because no turn survives reconstruction: a `Running`/`PendingInit` status becomes
    /// `Interrupted` and a `Running`/`Queued` activity becomes `Idle`. `closed_reason` is kept
    /// only on a `Closed` record. There is no child agent, no running turn, no queued input and
    /// no turn generation; the persisted transcript is replayed into a fresh ring of
    /// `quota.transcript_capacity` entries, so it keeps its order and the sequence numbers the
    /// ring assigns from the oldest retained entry. Activity frames are not persisted.
    pub(super) fn restore(
        seq: u64,
        id: SubAgentId,
        record: &PersistedSubAgentRecord,
        quota: &SubAgentQuota,
    ) -> Self {
        let status = match record.status {
            SubagentStatus::Running | SubagentStatus::PendingInit => SubagentStatus::Interrupted,
            other => other,
        };
        let activity = match record.activity {
            SubagentActivity::Running | SubagentActivity::Queued => SubagentActivity::Idle,
            other => other,
        };
        let closed_reason = if status == SubagentStatus::Closed {
            record.closed_reason.clone()
        } else {
            None
        };
        let sub_session_id = if record.sub_session_id.is_empty() {
            id.clone()
        } else {
            record.sub_session_id.clone()
        };
        let mut transcript = TranscriptBuffer::new(quota.transcript_capacity);
        transcript.extend(record.transcript.iter().cloned());
        let stream_id = id.clone();
        Self {
            seq,
            subagent_id: id,
            sub_session_id,
            subagent_type: record.subagent_type.clone(),
            display_name: record.display_name.clone(),
            role: record.role.clone(),
            task_description: record.task_description.clone(),
            config: record.config.clone(),
            status,
            activity,
            depth: record.depth,
            parent_subagent_id: record.parent_subagent_id.clone(),
            created_at: rfc3339_from_ms(record.created_at_ms),
            updated_at: rfc3339_from_ms(record.updated_at_ms),
            created_at_ms: record.created_at_ms,
            updated_at_ms: record.updated_at_ms,
            last_used_at_ms: record.last_used_at_ms,
            last_activity_at_ms: record.last_activity_at_ms,
            closed_reason,
            agent_id: record.agent_id.clone(),
            output: record.output.clone(),
            error: record.error.clone(),
            agent: None,
            pending: VecDeque::new(),
            running: false,
            generation: 0,
            handle: None,
            transcript,
            channel: TranscriptChannel::new(quota.transcript_channel_capacity),
            // Activity frames are deliberately not persisted: a restored instance starts with an
            // empty activity history (and the configured enabled/disabled mode) instead of
            // replaying the frames of the process that wrote the record.
            activity_stream: recorder(&stream_id, quota),
        }
    }

    /// Serializes this instance into the persisted record shape.
    ///
    /// `spawn_seq` is the instance's sequence number, `closed_at_ms` is the close timestamp of a
    /// tombstone (`None` while live), and the transcript is the retained ring in order.
    pub(super) fn persisted(&self) -> PersistedSubAgentRecord {
        PersistedSubAgentRecord {
            subagent_id: self.subagent_id.clone(),
            sub_session_id: self.sub_session_id.clone(),
            subagent_type: self.subagent_type.clone(),
            display_name: self.display_name.clone(),
            role: self.role.clone(),
            task_description: self.task_description.clone(),
            status: self.status,
            activity: self.activity,
            depth: self.depth,
            parent_subagent_id: self.parent_subagent_id.clone(),
            spawn_seq: self.seq,
            agent_id: self.agent_id.clone(),
            output: self.output.clone(),
            error: self.error.clone(),
            created_at_ms: self.created_at_ms,
            updated_at_ms: self.updated_at_ms,
            last_used_at_ms: self.last_used_at_ms,
            last_activity_at_ms: self.last_activity_at_ms,
            closed_at_ms: if self.is_live() {
                None
            } else {
                Some(self.updated_at_ms)
            },
            closed_reason: self.closed_reason.clone(),
            config: self.config.clone(),
            transcript: self.transcript.entries(),
        }
    }

    /// Appends `entry` to the transcript and publishes it on the live feed.
    ///
    /// Returns the assigned sequence number. [`TranscriptBuffer::push`] assigns it and reports
    /// ring evictions through [`super::transcript::TranscriptSummary::dropped`]; publishing is a
    /// non-blocking fan-out, so this is safe to call while the runtime holds its state lock.
    pub(super) fn record(&mut self, entry: TranscriptEntry) -> u64 {
        let stored = self.transcript.push(entry);
        self.channel.publish(&stored);
        stored.seq
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
            transcript: self.transcript.summary(),
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
            transcript: self.transcript.summary(),
        }
    }

    /// Builds the owned snapshot a lifecycle hook receives for `stage`.
    ///
    /// The snapshot carries the identity, nesting and transcript digest of this instance at the
    /// time of the transition, plus `reason` (the close/evict reason, or the turn error) and
    /// `now_ms` from the runtime clock. The runtime builds it under its state lock and awaits the
    /// hooks after releasing it, so a hook never observes a half-updated instance.
    pub(super) fn lifecycle_event(
        &self,
        stage: LifecycleStage,
        reason: Option<&str>,
        now_ms: i64,
    ) -> SubagentLifecycleEvent {
        SubagentLifecycleEvent {
            subagent_id: self.subagent_id.clone(),
            sub_session_id: self.sub_session_id.clone(),
            subagent_type: self.subagent_type.clone(),
            depth: self.depth,
            parent_subagent_id: self.parent_subagent_id.clone(),
            stage,
            status: self.status,
            activity: self.activity,
            reason: reason.map(str::to_owned),
            at_ms: now_ms,
            transcript: self.transcript.summary(),
        }
    }
}

/// Builds the activity recorder of `id`: enabled or disabled by
/// [`SubAgentQuota::enable_activity_stream`], with the throttle of the quota.
fn recorder(id: &str, quota: &SubAgentQuota) -> ActivityStream {
    let throttle = ActivityThrottle::new(
        quota.activity_capacity,
        quota.activity_text_max_len,
        quota.activity_throttle_ms,
    );
    if quota.enable_activity_stream {
        ActivityStream::new(id, throttle)
    } else {
        ActivityStream::disabled(id, throttle)
    }
}
