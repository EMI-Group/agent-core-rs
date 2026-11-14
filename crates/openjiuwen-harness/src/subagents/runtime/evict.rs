//! Deterministic LRU eviction of idle sub-agent instances.
//!
//! Ports the eviction step of Python `openjiuwen/harness/subagent_runtime/eviction.py`: when
//! the live cap is reached with [`SubAgentQuota::enable_lru_eviction`] set,
//! `SubagentControl._acquire_slot` closes the least-recently-used idle instance and retries the
//! reservation. [`SubAgentRuntime::evict`], [`SubAgentRuntime::evict_lru`] and
//! [`SubAgentRuntime::eviction_candidates`] expose the same policy to callers, and
//! [`EVICTION_REASON`] is the `closed_reason` every eviction records.
//!
//! Eviction is synchronous: the runtime holds its state lock, closes one instance, and notifies
//! waiters once the lock is released. A `Running` instance is never evicted, so a runtime
//! saturated with running turns keeps refusing spawns instead of closing them.

use openjiuwen_common::error::{Error, Result};

use super::quota::{
    compare_lru, select_lru_candidate, LruCandidate, SubAgentQuota, SubagentActivity,
};
use super::transcript::TranscriptEntry;
use super::{not_found, RuntimeState, SubAgentId, SubAgentRuntime, SubagentStatus};

/// `closed_reason` recorded when an instance is evicted to free a capacity slot.
///
/// The other reasons are `"manual"` ([`SubAgentRuntime::close`]) and `"parent_ended"`
/// ([`SubAgentRuntime::cancel_all`]); a resumed instance clears the field again.
pub const EVICTION_REASON: &str = "evicted";

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
    /// its turn is running (`cannot evict running subagent: <id>`), or when it is already
    /// closed (`subagent already closed: <id>`).
    // Kept async for parity with the Python control surface; the tools await it.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn evict(&self, id: &str) -> Result<SubagentStatus> {
        let now_ms = self.clock.now_ms();
        let previous = {
            let mut state = self.shared.state.lock();
            let Some(instance) = state.instances.get(id) else {
                return Err(not_found(id));
            };
            if instance.status == SubagentStatus::Running
                || instance.activity == SubagentActivity::Running
            {
                return Err(Error::validation(format!(
                    "cannot evict running subagent: {id}"
                )));
            }
            if instance.status == SubagentStatus::Closed {
                return Err(Error::validation(format!("subagent already closed: {id}")));
            }
            match evict_locked(&mut state, id, now_ms) {
                Some((previous, _live)) => previous,
                None => return Err(not_found(id)),
            }
        };
        self.shared.notify.notify_waiters();
        Ok(previous)
    }

    /// Evicts the least-recently-used evictable instance and returns its id.
    ///
    /// The pick is [`select_lru_candidate`]'s: an `Idle` instance before a `Queued` one, then
    /// the oldest activity, then the smallest id. `Ok(None)` means nothing is evictable — no
    /// live instance at all, or every live instance is running.
    ///
    /// # Errors
    ///
    /// Reserved for parity with the Python control surface; the in-process runtime never fails
    /// this call.
    #[allow(
        clippy::unused_async,
        clippy::unused_async_trait_impl,
        clippy::unnecessary_wraps
    )]
    pub async fn evict_lru(&self) -> Result<Option<SubAgentId>> {
        let now_ms = self.clock.now_ms();
        let id = {
            let mut state = self.shared.state.lock();
            let Some(id) = candidates(&state, None)
                .first()
                .map(|candidate| candidate.subagent_id.clone())
            else {
                return Ok(None);
            };
            if evict_locked(&mut state, &id, now_ms).is_none() {
                return Ok(None);
            }
            id
        };
        self.shared.notify.notify_waiters();
        Ok(Some(id))
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

/// Evicts idle instances until the live cap admits one more spawn; returns how many went.
///
/// Mirrors Python `SubagentControl._acquire_slot`: eviction happens before the reservation is
/// retried, one instance per iteration, and stops as soon as nothing is evictable. Every
/// eviction frees exactly one slot, so the loop is bounded by the number of live instances.
pub(super) fn make_room(
    state: &mut RuntimeState,
    quota: &SubAgentQuota,
    now_ms: i64,
    protect: Option<&str>,
) -> usize {
    let mut evicted = 0;
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
    }
    evicted
}
