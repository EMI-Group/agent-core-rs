//! Idle eviction of pooled team runs.
//!
//! One background task per pool — started lazily by the first admission and
//! guarded by a single slot in [`TeamPoolInner`] — scans the entry map on
//! `TeamPoolConfig::reaper_interval`. An entry whose last lease was released
//! more than `TeamPoolConfig::idle_timeout` ago is evicted: a finished one is
//! simply dropped from the map, a still-live one is cancelled, finalised as
//! [`RunPhase::Cancelled`](crate::monitor::RunPhase::Cancelled), unregistered
//! from the hub, and then dropped. That is Python's orphan reclaim
//! (`TeamRuntimeManager.finalize` over a stale pool entry) without a disk hop.
//!
//! The task uses only [`tokio::time`] (`Instant`, `sleep`), never
//! `std::time::Instant`, so it runs on the tokio clock: a process that enables
//! tokio's `test-util` feature can pause and advance it deterministically.

use std::sync::{Arc, Weak};
use std::time::Duration;

use tokio::task::JoinHandle;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use tracing::debug;

use super::key::PoolKey;
use super::{TeamPoolInner, LOG_TARGET};

/// The reason an idle-evicted live run records.
pub(crate) const IDLE_REASON: &str = "idle timeout";

/// The pool's single reaper task.
pub(crate) struct ReaperHandle {
    /// Cancels the scan loop.
    pub(crate) token: CancellationToken,
    /// The scan loop, joined (bounded) by `stop_all`.
    pub(crate) handle: JoinHandle<()>,
}

impl ReaperHandle {
    /// Builds a handle over an already-spawned task (tests only).
    #[cfg(test)]
    pub(crate) fn detached(token: CancellationToken) -> Self {
        Self {
            token,
            handle: tokio::spawn(async {}),
        }
    }
}

/// The instant an entry idle since `idle_since` falls due.
pub(crate) fn due_at(idle_since: Instant, timeout: Duration) -> Instant {
    idle_since + timeout
}

/// Whether an entry idle since `idle_since` is due for eviction at `now`.
///
/// An entry that was never released (`None`) is never due.
pub(crate) fn is_due(idle_since: Option<Instant>, timeout: Duration, now: Instant) -> bool {
    idle_since.is_some_and(|since| due_at(since, timeout) <= now)
}

/// Starts the pool's reaper, unless one is already running.
///
/// A pool that disables eviction (`idle_timeout() == None`) never starts one,
/// and a call outside a tokio runtime is a no-op rather than a panic.
pub(crate) fn ensure_started(pool: &Arc<TeamPoolInner>) {
    let Some(timeout) = pool.config().idle_timeout() else {
        return;
    };
    let Ok(runtime) = tokio::runtime::Handle::try_current() else {
        return;
    };
    let token = CancellationToken::new();
    let handle = runtime.spawn(run(
        Arc::downgrade(pool),
        token.clone(),
        pool.config().reaper_interval(),
        timeout,
    ));
    if !pool.install_reaper(ReaperHandle {
        token: token.clone(),
        handle,
    }) {
        // Another reaper won the slot; this one exits on its first select.
        token.cancel();
    }
}

/// The scan loop: sleep, reap, and stop when the pool goes away or closes.
async fn run(
    pool: Weak<TeamPoolInner>,
    token: CancellationToken,
    interval: Duration,
    timeout: Duration,
) {
    loop {
        tokio::select! {
            biased;
            () = token.cancelled() => break,
            () = tokio::time::sleep(interval) => {}
        }
        let Some(inner) = pool.upgrade() else {
            break;
        };
        let evicted = reap(&inner, timeout);
        if evicted > 0 {
            debug!(
                target: LOG_TARGET,
                evicted,
                "evicted idle pooled team runs"
            );
        }
        if inner.is_closed() {
            break;
        }
    }
}

/// Evicts every entry idle past `timeout`, returning how many were evicted.
///
/// An entry with a live lease is never evicted, and a still-live entry is
/// cancelled and finalised before it is dropped. The worker task is not awaited
/// here: it observes the entry's token and finishes on its own (bounded by the
/// team strategy), so the scan never blocks.
pub(crate) fn reap(pool: &TeamPoolInner, timeout: Duration) -> usize {
    let now = Instant::now();
    let due: Vec<PoolKey> = pool
        .keys()
        .into_iter()
        .filter(|key| {
            pool.entry(key).is_some_and(|entry| {
                entry.leases() == 0 && is_due(entry.idle_since(), timeout, now)
            })
        })
        .collect();

    let mut evicted = 0;
    for key in due {
        let Some(entry) = pool.remove(&key) else {
            continue;
        };
        if entry.is_terminal() {
            debug!(target: LOG_TARGET, key = %key, "evicted a finished pooled team run");
        } else {
            entry.cancel_and_finalise(pool.next_seq(), IDLE_REASON);
            debug!(target: LOG_TARGET, key = %key, "stopped an idle pooled team run");
        }
        evicted += 1;
    }
    evicted
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::config::TeamPoolConfig;
    use crate::pool::entry::{Entry, EntryInit, FinalOutcome};
    use crate::pool::TeamPool;

    #[test]
    fn the_idle_deadline_is_inclusive_and_never_due_without_a_stamp() {
        let now = Instant::now();
        let timeout = Duration::from_secs(300);
        assert_eq!(due_at(now, timeout), now + timeout);
        assert!(!is_due(None, timeout, now));
        assert!(!is_due(Some(now), timeout, now));
        assert!(!is_due(
            Some(now),
            timeout,
            now + timeout - Duration::from_millis(1)
        ));
        assert!(is_due(Some(now - timeout), timeout, now));
        assert!(is_due(Some(now), Duration::ZERO, now));
    }

    #[test]
    fn reaping_evicts_a_finished_idle_entry_and_keeps_a_leased_one() {
        let pool = TeamPool::new();
        let idle = PoolKey::new("crew", "s-1");
        let leased = PoolKey::new("crew", "s-2");
        let mut entries = Vec::new();
        for key in [&idle, &leased] {
            let entry = Arc::new(Entry::new(EntryInit {
                run_id: format!("run-{}", key.session_id),
                key: key.clone(),
                label: "crew".to_owned(),
                deadline: None,
                payload_capacity: 2,
                frame_capacity: 2,
                hub: None,
                monitor: None,
            }));
            pool.inner.entries().insert(key.clone(), Arc::clone(&entry));
            entries.push(entry);
        }
        let seq = pool.inner.next_seq();
        entries[0].finalise(&FinalOutcome::Completed(serde_json::json!(1)), seq);
        let seq = pool.inner.next_seq();
        entries[1].finalise(&FinalOutcome::Completed(serde_json::json!(2)), seq);
        entries[1].add_lease();

        // A deadline of zero is due as soon as the entry is stamped idle, which
        // keeps this deterministic without a paused clock.
        let timeout = Duration::ZERO;
        assert_eq!(reap(&pool.inner, timeout), 1);
        assert!(pool.get(&idle).is_none(), "the idle entry was evicted");
        assert!(pool.get(&leased).is_some(), "a leased entry survives");
    }

    #[test]
    fn reaping_cancels_a_live_orphan() {
        let pool = TeamPool::new();
        let key = PoolKey::new("crew", "s-1");
        let entry = Arc::new(Entry::new(EntryInit {
            run_id: "run-1".to_owned(),
            key: key.clone(),
            label: "crew".to_owned(),
            deadline: None,
            payload_capacity: 2,
            frame_capacity: 2,
            hub: None,
            monitor: None,
        }));
        pool.inner.entries().insert(key.clone(), Arc::clone(&entry));
        entry.mark_running();
        entry.add_lease();
        assert_eq!(reap(&pool.inner, Duration::ZERO), 0, "a leased run stays");
        entry.release_lease();
        assert_eq!(reap(&pool.inner, Duration::ZERO), 1);
        assert!(pool.get(&key).is_none());
        assert_eq!(entry.phase(), crate::monitor::RunPhase::Cancelled);
        assert!(entry.token().is_cancelled(), "the orphan was cancelled");
    }

    #[tokio::test]
    async fn the_reaper_is_started_at_most_once_and_stops_with_the_pool() {
        let pool = TeamPool::with_config(TeamPoolConfig::default().with_idle_timeout_secs(1.0));
        ensure_started(&pool.inner);
        ensure_started(&pool.inner);
        assert!(pool.inner.take_reaper().is_some());
        assert!(pool.inner.take_reaper().is_none(), "only one reaper ran");
    }

    #[tokio::test]
    async fn a_pool_without_an_idle_timeout_never_starts_a_reaper() {
        let pool = TeamPool::with_config(TeamPoolConfig::default().with_idle_timeout_secs(0.0));
        ensure_started(&pool.inner);
        assert!(pool.inner.take_reaper().is_none());
    }
}
