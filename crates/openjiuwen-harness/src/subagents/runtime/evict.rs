//! Deterministic LRU eviction of idle sub-agent instances.
//!
//! Ports the eviction step of Python `openjiuwen/harness/subagent_runtime/eviction.py`: when
//! the live cap is reached with [`SubAgentQuota::enable_lru_eviction`] set,
//! `SubagentControl._acquire_slot` closes the least-recently-used idle instance and retries the
//! reservation. [`SubAgentRuntime::evict`], [`SubAgentRuntime::evict_lru`] and
//! [`SubAgentRuntime::eviction_candidates`] expose the same policy to callers, and
//! [`EVICTION_REASON`] is the `closed_reason` every eviction records.
//!
//! Two eviction paths exist:
//! - the explicit [`SubAgentRuntime::evict`]/[`SubAgentRuntime::evict_lru`] calls run a two-phase
//!   protocol (`evict_with_hooks`): they claim the id under the state lock, await the
//!   [`LifecycleStage::BeforeEvict`] hooks without it, then re-check, write the tombstone and
//!   await the [`LifecycleStage::Close`] hooks. The private `RuntimeState::evicting` claim keeps two
//!   concurrent calls from writing two tombstones;
//! - the capacity-pressure path inside the reservation ([`make_room`]) must free the slot
//!   synchronously, so it closes the instance under the lock, captures the `Close` snapshot and
//!   lets the spawning caller await it right after the lock is released. It fires `Close` only.
//!
//! A `Running` instance is never evicted, so a runtime saturated with running turns keeps
//! refusing spawns instead of closing them.

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

use super::lifecycle::{LifecycleStage, SubagentLifecycleEvent};
use super::quota::{
    compare_lru, select_lru_candidate, LruCandidate, SubAgentQuota, SubagentActivity,
};
use super::transcript::{TranscriptEntry, TranscriptSummary};
use super::{lifecycle_api, not_found, RuntimeState, SubAgentId, SubAgentRuntime, SubagentStatus};

/// `closed_reason` recorded when an instance is evicted to free a capacity slot.
///
/// The other reasons are `"manual"` ([`SubAgentRuntime::close`]) and `"parent_ended"`
/// ([`SubAgentRuntime::cancel_all`]); a resumed instance clears the field again.
pub const EVICTION_REASON: &str = "evicted";

/// The record one eviction returns (see the runtime's `evict_with_hooks`).
///
/// It reports what was evicted, its status before the tombstone, the reason stamped on the
/// instance, and the transcript digest at eviction time (after the `evicted` entry).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SubagentEviction {
    /// Instance id, equal to the child session id.
    pub subagent_id: SubAgentId,
    /// Status of the instance right before the tombstone was written.
    pub previous_status: SubagentStatus,
    /// Why the instance was closed ([`EVICTION_REASON`] for every eviction).
    pub reason: String,
    /// Digest of the instance's transcript at eviction time.
    pub transcript: TranscriptSummary,
}

impl SubAgentRuntime {
    /// Evicts the instance `id` on request and returns its previous status.
    ///
    /// The instance is closed with [`EVICTION_REASON`]: its status and activity become
    /// `Closed`, its timestamps are stamped from the runtime clock, a running turn task would
    /// be aborted, and a [`SubAgentRuntime::wait`] on the id observes the close.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `id` is unknown (`subagent not found: <id>`), when
    /// its turn is running (`cannot evict running subagent: <id>`), when it is already
    /// closed (`subagent already closed: <id>`), or when another eviction of the same id is in
    /// flight (`subagent eviction in progress: <id>`).
    pub async fn evict(&self, id: &str) -> Result<SubagentStatus> {
        Ok(self.evict_with_hooks(id).await?.previous_status)
    }

    /// Evicts one instance in two phases: claim and `BeforeEvict`, then tombstone and `Close`.
    ///
    /// The id is validated and claimed under the state lock, the [`LifecycleStage::BeforeEvict`]
    /// hooks are awaited without it, and the instance is re-checked before the tombstone is
    /// written (`cannot evict running subagent: <id>` if it started a turn meanwhile, releasing
    /// the claim). After [`evict_locked`] records the `evicted` transcript entry and activity
    /// frame, the lock is dropped, waiters are notified, and the [`LifecycleStage::Close`] hooks
    /// run with reason [`EVICTION_REASON`].
    ///
    /// Two concurrent calls on one id produce exactly one tombstone, one transcript entry, one
    /// activity frame and one hook pass: the loser is refused with
    /// `subagent eviction in progress: <id>` (or `subagent already closed: <id>` once the winner
    /// finished). Hook failures are contained and never abort the eviction.
    ///
    /// # Errors
    ///
    /// Same validation errors as [`SubAgentRuntime::evict`]; the instance is left untouched on
    /// every error.
    pub(super) async fn evict_with_hooks(&self, id: &str) -> Result<SubagentEviction> {
        let now_ms = self.clock.now_ms();
        let before = {
            let mut state = self.shared.state.lock();
            let Some(instance) = state.instances.get(id) else {
                return Err(not_found(id));
            };
            if instance.status == SubagentStatus::Closed {
                return Err(Error::validation(format!("subagent already closed: {id}")));
            }
            if instance.status == SubagentStatus::Running
                || instance.activity == SubagentActivity::Running
            {
                return Err(cannot_evict(id));
            }
            if state.evicting.contains(id) {
                return Err(Error::validation(format!(
                    "subagent eviction in progress: {id}"
                )));
            }
            let event = instance.lifecycle_event(LifecycleStage::BeforeEvict, None, now_ms);
            state.evicting.insert(id.to_owned());
            event
        };
        lifecycle_api::run_stage(&self.shared, LifecycleStage::BeforeEvict, [before]).await;
        let (eviction, close) = {
            let mut state = self.shared.state.lock();
            let still_evictable = state
                .instances
                .get(id)
                .is_some_and(|instance| instance.is_live() && instance.activity.is_evictable());
            if !still_evictable {
                state.evicting.remove(id);
                let running = state.instances.get(id).is_some_and(|instance| {
                    instance.status == SubagentStatus::Running
                        || instance.activity == SubagentActivity::Running
                });
                return Err(if running {
                    cannot_evict(id)
                } else {
                    not_found(id)
                });
            }
            let Some((previous, _live)) = evict_locked(&mut state, id, now_ms) else {
                state.evicting.remove(id);
                return Err(cannot_evict(id));
            };
            let instance = state.instances.get(id);
            let transcript = instance.map_or_else(TranscriptSummary::default, |instance| {
                instance.transcript.summary()
            });
            let close = instance.map(|instance| {
                instance.lifecycle_event(LifecycleStage::Close, Some(EVICTION_REASON), now_ms)
            });
            state.evicting.remove(id);
            (
                SubagentEviction {
                    subagent_id: id.to_owned(),
                    previous_status: previous,
                    reason: EVICTION_REASON.to_owned(),
                    transcript,
                },
                close,
            )
        };
        self.shared.notify.notify_waiters();
        if let Some(event) = close {
            lifecycle_api::run_stage(&self.shared, LifecycleStage::Close, [event]).await;
        }
        Ok(eviction)
    }

    /// Evicts the least-recently-used evictable instance and returns its id.
    ///
    /// The pick is [`select_lru_candidate`]'s: an `Idle` instance before a `Queued` one, then
    /// the oldest activity, then the smallest id. `Ok(None)` means nothing is evictable — no
    /// live instance at all, or every live instance is running. The picked instance goes through
    /// [`SubAgentRuntime::evict`]'s two-phase protocol, hooks included.
    ///
    /// # Errors
    ///
    /// A picked instance that a concurrent call claimed, closed or started a turn for surfaces
    /// the matching validation error instead of silently skipping it.
    pub async fn evict_lru(&self) -> Result<Option<SubAgentId>> {
        let id = {
            let state = self.shared.state.lock();
            candidates(&state, None)
                .first()
                .map(|candidate| candidate.subagent_id.clone())
        };
        let Some(id) = id else {
            return Ok(None);
        };
        Ok(Some(self.evict_with_hooks(&id).await?.subagent_id))
    }

    /// Returns the ids of the evictable live instances in LRU order.
    ///
    /// Only `Idle` and `Queued` instances are listed, best eviction candidate first, which is
    /// the order [`SubAgentRuntime::evict_lru`] consumes; running and closed instances never
    /// appear.
    #[must_use]
    pub fn eviction_candidates(&self) -> Vec<SubAgentId> {
        let state = self.shared.state.lock();
        candidates(&state, None)
            .into_iter()
            .map(|candidate| candidate.subagent_id)
            .collect()
    }
}

/// The error of evicting an instance that is running a turn.
fn cannot_evict(id: &str) -> Error {
    Error::validation(format!("cannot evict running subagent: {id}"))
}

/// Returns the number of live (non-`Closed`) instances.
pub(super) fn live_count(state: &RuntimeState) -> usize {
    state
        .instances
        .values()
        .filter(|instance| instance.is_live())
        .count()
}

/// Collects the evictable live instances as LRU candidates, in LRU order.
///
/// `protect` names one instance to leave alone: a nested spawn never recycles the parent that
/// issued it, so an idle parent cannot be closed out from under its own child.
pub(super) fn candidates(state: &RuntimeState, protect: Option<&str>) -> Vec<LruCandidate> {
    let mut candidates: Vec<LruCandidate> = state
        .instances
        .values()
        .filter(|instance| instance.is_live() && instance.activity.is_evictable())
        .filter(|instance| protect != Some(instance.subagent_id.as_str()))
        .map(|instance| LruCandidate {
            subagent_id: instance.subagent_id.clone(),
            activity: instance.activity,
            last_activity_ms: instance.last_activity_at_ms,
            spawn_seq: instance.seq,
        })
        .collect();
    candidates.sort_by(compare_lru);
    candidates
}

/// Closes one evictable instance and reports its previous status plus the live count after.
///
/// Returns `None` when `id` is unknown or not evictable (`Running`/already `Closed`), leaving
/// the state untouched. The caller holds the state lock and must notify waiters afterwards.
pub(super) fn evict_locked(
    state: &mut RuntimeState,
    id: &str,
    now_ms: i64,
) -> Option<(SubagentStatus, usize)> {
    let instance = state.instances.get_mut(id)?;
    if !instance.is_live() || !instance.activity.is_evictable() {
        return None;
    }
    let previous = instance.status;
    instance.close(now_ms, EVICTION_REASON);
    instance.record(TranscriptEntry::evicted(id, EVICTION_REASON, now_ms));
    instance.activity_stream.evicted(EVICTION_REASON, now_ms);
    let live = live_count(state);
    tracing::debug!(
        target: "openjiuwen_harness::subagents",
        subagent_id = %id,
        reason = EVICTION_REASON,
        live,
        "subagent evicted"
    );
    Some((previous, live))
}

/// Evicts idle instances until the live cap admits one more spawn; returns how many went and the
/// `Close` snapshots the caller must await.
///
/// Mirrors Python `SubagentControl._acquire_slot`: eviction happens before the reservation is
/// retried, one instance per iteration, and stops as soon as nothing is evictable. Every
/// eviction frees exactly one slot, so the loop is bounded by the number of live instances.
/// This path runs under the state lock, so it cannot await `BeforeEvict`; each evicted instance
/// yields one `Close`-stage snapshot (reason [`EVICTION_REASON`]) taken after its tombstone, for
/// the spawning caller to await once the lock is released.
pub(super) fn make_room(
    state: &mut RuntimeState,
    quota: &SubAgentQuota,
    now_ms: i64,
    protect: Option<&str>,
) -> (usize, Vec<SubagentLifecycleEvent>) {
    let mut evicted = 0;
    let mut events = Vec::new();
    while quota.enable_lru_eviction && live_count(state) >= quota.max_subagents {
        let Some(id) = select_lru_candidate(&candidates(state, protect))
            .map(|candidate| candidate.subagent_id.clone())
        else {
            break;
        };
        if evict_locked(state, &id, now_ms).is_none() {
            break;
        }
        evicted += 1;
        if let Some(event) = state.instances.get(&id).map(|instance| {
            instance.lifecycle_event(LifecycleStage::Close, Some(EVICTION_REASON), now_ms)
        }) {
            events.push(event);
        }
    }
    (evicted, events)
}
