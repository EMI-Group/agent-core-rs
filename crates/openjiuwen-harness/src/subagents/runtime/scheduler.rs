//! Opt-in background eviction scheduler: periodic TTL/idle sweeps of sub-agent instances.
//!
//! Every other eviction path of this crate is caller-driven: [`SubAgentRuntime::evict`],
//! [`SubAgentRuntime::evict_lru`] and the capacity-pressure pass inside `spawn` only run when
//! someone asks for them. A long-lived host (a `DeepAgent` REPL, a team leader) additionally
//! wants instances that have been unused for a while to be reclaimed, so this module adds two
//! layers:
//!
//! - [`SubAgentRuntime::sweep_expired_idle`] performs exactly one idle-sweep: it selects the
//!   live, non-running instances whose last activity is at least `idle_ttl_ms` old, ordered by
//!   the same LRU preference as [`SubAgentRuntime::eviction_candidates`], and evicts them
//!   through the two-phase [`SubAgentRuntime::evict`] protocol (lifecycle hooks included).
//! - [`SubAgentRuntime::start_eviction_scheduler`] drives that sweep from a periodic tick and
//!   hands back an [`EvictionSchedulerHandle`], while
//!   [`SubAgentRuntime::start_eviction_scheduler_with_ticks`] lets a host own the timer.
//!
//! # Disabled by default
//!
//! **Nothing runs unless a caller explicitly starts a scheduler.** Constructing a
//! [`SubAgentRuntime`] spawns no task, opens no timer and installs no global state, so an
//! existing embedding keeps its completely caller-driven behaviour. This is a documented
//! deviation from Python `openjiuwen/harness/subagent_runtime/`, where `SubagentControl` owns
//! a background sweeper; here the equivalent is an explicit handle, so the host stays in
//! control of task lifetime (and a one-shot CLI process can simply never start one).
//!
//! The task lives exactly as long as the handle: dropping it cancels the loop, and
//! [`EvictionSchedulerHandle::shutdown`] awaits the driver so an in-flight sweep is never
//! aborted mid-way. A sweep never fails — the instances it picks may of course be gone or busy
//! by the time the eviction attempt runs, and those typed errors are logged at `debug` and
//! treated as skips, which makes repeated sweeps idempotent.
//!
//! # Determinism
//!
//! The sweep reads `now_ms` from the runtime's injected [`Clock`](super::clock::Clock), so a
//! [`ManualClock`](super::clock::ManualClock) makes the TTL comparison exact, and
//! [`SubAgentRuntime::start_eviction_scheduler_with_ticks`] accepts a caller-supplied tick
//! stream, so a test drives the loop without sleeping.

use std::sync::Arc;
use std::time::Duration;

use futures::{FutureExt, StreamExt};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::sync::Notify;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use super::evict;
use super::quota::SubagentActivity;
use super::{SubAgentId, SubAgentRuntime, SubagentEviction, SubagentStatus};

/// Default period between two sweeps of [`SubAgentRuntime::start_eviction_scheduler`].
pub const DEFAULT_EVICTION_INTERVAL: Duration = Duration::from_secs(30);

/// Default idle time after which a sweep evicts an instance (15 minutes, in milliseconds).
pub const DEFAULT_IDLE_TTL_MS: i64 = 900_000;

/// Settings of one eviction scheduler (Python `SubagentRuntimeConfig` has no counterpart).
///
/// The default is "sweep every 30 s, reclaim instances idle for 15 min, reclaim every candidate
/// of the sweep"; the scheduler itself is still opt-in, see the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct EvictionSchedulerConfig {
    /// Period between two sweeps.
    pub interval: Duration,
    /// Idle time after which an instance becomes a sweep candidate, in milliseconds.
    ///
    /// Never negative: [`EvictionSchedulerConfig::new`] and
    /// [`EvictionSchedulerConfig::with_idle_ttl_ms`] clamp it to `0`, which makes every idle
    /// instance a candidate of the next sweep.
    pub idle_ttl_ms: i64,
    /// Maximum instances one sweep evicts; `None` = no limit, `Some(0)` = evict nothing.
    pub max_per_sweep: Option<usize>,
}

impl Default for EvictionSchedulerConfig {
    fn default() -> Self {
        Self {
            interval: DEFAULT_EVICTION_INTERVAL,
            idle_ttl_ms: DEFAULT_IDLE_TTL_MS,
            max_per_sweep: None,
        }
    }
}

impl EvictionSchedulerConfig {
    /// Creates a configuration sweeping every `interval` for instances idle for `idle_ttl_ms`.
    ///
    /// A negative `idle_ttl_ms` is clamped to `0`.
    #[must_use]
    pub fn new(interval: Duration, idle_ttl_ms: i64) -> Self {
        Self {
            interval,
            idle_ttl_ms: idle_ttl_ms.max(0),
            max_per_sweep: None,
        }
    }

    /// Replaces the period between two sweeps.
    #[must_use]
    pub fn with_interval(mut self, interval: Duration) -> Self {
        self.interval = interval;
        self
    }

    /// Replaces the idle TTL; a negative value is clamped to `0`.
    #[must_use]
    pub fn with_idle_ttl_ms(mut self, idle_ttl_ms: i64) -> Self {
        self.idle_ttl_ms = idle_ttl_ms.max(0);
        self
    }

    /// Replaces the per-sweep eviction cap.
    #[must_use]
    pub fn with_max_per_sweep(mut self, max_per_sweep: Option<usize>) -> Self {
        self.max_per_sweep = max_per_sweep;
        self
    }

    /// Returns the period between two sweeps.
    #[must_use]
    pub fn interval(&self) -> Duration {
        self.interval
    }

    /// Returns the idle TTL in milliseconds.
    #[must_use]
    pub fn idle_ttl_ms(&self) -> i64 {
        self.idle_ttl_ms
    }

    /// Returns the per-sweep eviction cap.
    #[must_use]
    pub fn max_per_sweep(&self) -> Option<usize> {
        self.max_per_sweep
    }
}

/// What one idle sweep did.
///
/// `considered` counts the instances that passed the TTL filter (before
/// [`EvictionSchedulerConfig::max_per_sweep`] truncation), `evicted` holds the record of every
/// instance that was actually closed, and `skipped_running` counts the live instances the sweep
/// left alone only because a turn was running on them.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SweepReport {
    /// Instances that passed the TTL filter.
    pub considered: usize,
    /// Instances the sweep evicted, in eviction order (LRU order).
    pub evicted: Vec<SubagentEviction>,
    /// Live instances skipped because a turn was running on them.
    pub skipped_running: usize,
}

impl SweepReport {
    /// Whether the sweep evicted nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.evicted.is_empty()
    }

    /// Number of instances the sweep evicted.
    #[must_use]
    pub fn len(&self) -> usize {
        self.evicted.len()
    }
}

/// Cumulative counters of one eviction scheduler.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct EvictionSchedulerStats {
    /// Completed sweeps.
    pub sweeps: u64,
    /// Instances evicted by those sweeps.
    pub evicted: u64,
    /// Live instances the sweeps skipped because their turn was running.
    pub skipped_running: u64,
    /// Sweeps that could not complete (a panic inside the sweep task).
    pub failures: u64,
    /// Id of the most recently evicted instance, when any.
    pub last_evicted: Option<SubAgentId>,
}

impl EvictionSchedulerStats {
    /// Number of completed sweeps.
    #[must_use]
    pub fn sweeps(&self) -> u64 {
        self.sweeps
    }

    /// Number of instances evicted by the sweeps.
    #[must_use]
    pub fn evicted(&self) -> u64 {
        self.evicted
    }

    /// Number of live instances skipped because their turn was running.
    #[must_use]
    pub fn skipped_running(&self) -> u64 {
        self.skipped_running
    }

    /// Number of sweeps that could not complete.
    #[must_use]
    pub fn failures(&self) -> u64 {
        self.failures
    }

    /// Id of the most recently evicted instance, when any.
    #[must_use]
    pub fn last_evicted(&self) -> Option<&str> {
        self.last_evicted.as_deref()
    }
}

impl SubAgentRuntime {
    /// Evicts every live, non-running instance idle for at least `idle_ttl_ms`.
    ///
    /// `now_ms` is read once from the runtime clock; an instance is a candidate when
    /// `now_ms - last_activity_at_ms >= idle_ttl_ms`, so a TTL of `0` sweeps everything that is
    /// not running right now (`0` never evicts a running instance, which is not a candidate at
    /// all). A negative TTL is clamped to `0`.
    ///
    /// Candidates are ordered exactly like [`SubAgentRuntime::eviction_candidates`] — `Idle`
    /// before `Queued`, then the oldest activity, then the smallest id — and the ids already
    /// claimed by a concurrent eviction are skipped, so two overlapping sweeps never attempt the
    /// same instance. `max` truncates the list before anything is evicted (`Some(0)` evicts
    /// nothing), and `considered` reports the untruncated candidate count.
    ///
    /// The selection happens under one brief state lock which is dropped before any `.await`;
    /// each candidate then goes through the two-phase eviction protocol, so its lifecycle hooks
    /// run and its `BeforeEvict`/`Close` stages are awaited. An eviction that lost the race
    /// (the instance started a turn, was closed, or was claimed by another sweep) is a skip,
    /// logged at `debug`; the sweep itself cannot fail.
    pub async fn sweep_expired_idle(&self, idle_ttl_ms: i64, max: Option<usize>) -> SweepReport {
        let now_ms = self.clock.now_ms();
        let ttl = idle_ttl_ms.max(0);
        let (candidates, considered, skipped_running) = {
            let state = self.shared.state.lock();
            let mut candidates: Vec<SubAgentId> = evict::candidates(&state, None)
                .into_iter()
                .filter(|candidate| !state.evicting.contains(candidate.subagent_id.as_str()))
                .filter(|candidate| now_ms.saturating_sub(candidate.last_activity_ms) >= ttl)
                .map(|candidate| candidate.subagent_id)
                .collect();
            let considered = candidates.len();
            if let Some(max) = max {
                candidates.truncate(max);
            }
            let skipped_running = state
                .instances
                .values()
                .filter(|instance| {
                    instance.is_live()
                        && (instance.status == SubagentStatus::Running
                            || instance.activity == SubagentActivity::Running)
                })
                .count();
            (candidates, considered, skipped_running)
        };
        let mut report = SweepReport {
            considered,
            skipped_running,
            ..SweepReport::default()
        };
        for id in candidates {
            match self.evict_with_hooks(&id).await {
                Ok(eviction) => report.evicted.push(eviction),
                Err(error) => tracing::debug!(
                    target: "openjiuwen_harness::subagents",
                    subagent_id = %id,
                    %error,
                    "subagent sweep skipped instance"
                ),
            }
        }
        report
    }

    /// Starts the background eviction scheduler, sweeping once every `config.interval`.
    ///
    /// The first sweep happens after one full interval, never immediately; missed ticks are
    /// delayed instead of bursting, so a sweep that outlasts the period does not trigger a
    /// catch-up storm. The returned [`EvictionSchedulerHandle`] owns the task: dropping it
    /// cancels the loop, [`EvictionSchedulerHandle::shutdown`] also awaits it.
    ///
    /// The scheduler is opt-in and off by default: no code in this crate starts one, so a
    /// runtime only sweeps in the background when a host asks it to. A zero interval is treated
    /// as one millisecond so the timer cannot panic.
    #[must_use]
    pub fn start_eviction_scheduler(
        self: &Arc<Self>,
        config: EvictionSchedulerConfig,
    ) -> EvictionSchedulerHandle {
        let period = if config.interval.is_zero() {
            Duration::from_millis(1)
        } else {
            config.interval
        };
        let mut interval = tokio::time::interval_at(tokio::time::Instant::now() + period, period);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let ticks = futures::stream::unfold(interval, |mut interval| async move {
            interval.tick().await;
            Some(((), interval))
        });
        self.start_eviction_scheduler_with_ticks(config, ticks)
    }

    /// Starts the background eviction scheduler over a caller-supplied tick stream.
    ///
    /// This is the deterministic driver seam: every item the stream yields triggers one
    /// [`SubAgentRuntime::sweep_expired_idle`] call, so a test (or a host that owns its timer,
    /// event loop or clock) drives the sweeps exactly. The stream ending stops the scheduler
    /// like a cancellation does. The task ends after the in-flight sweep completes, so no sweep
    /// is ever aborted mid-way.
    #[must_use]
    pub fn start_eviction_scheduler_with_ticks(
        self: &Arc<Self>,
        config: EvictionSchedulerConfig,
        ticks: impl futures::Stream<Item = ()> + Send + 'static,
    ) -> EvictionSchedulerHandle {
        let cancel = CancellationToken::new();
        let stats = Arc::new(Mutex::new(EvictionSchedulerStats::default()));
        let swept = Arc::new(Notify::new());
        let join = tokio::spawn(run_eviction_loop(
            Arc::clone(self),
            config,
            ticks,
            cancel.clone(),
            Arc::clone(&stats),
            Arc::clone(&swept),
        ));
        EvictionSchedulerHandle {
            config,
            cancel,
            join: Some(join),
            stats,
            swept,
        }
    }
}

/// The driver task behind one [`EvictionSchedulerHandle`].
///
/// It waits for either the next tick or a cancellation, runs one sweep and records its
/// counters before publishing the completed-sweep signal. Because the `select!` only runs
/// *between* sweeps, a cancellation is honoured after the current sweep finished, which is what
/// makes [`EvictionSchedulerHandle::shutdown`] drain instead of abort. A panicking sweep is
/// contained (counted in [`EvictionSchedulerStats::failures`]) so the loop keeps running.
pub(crate) async fn run_eviction_loop(
    runtime: Arc<SubAgentRuntime>,
    config: EvictionSchedulerConfig,
    ticks: impl futures::Stream<Item = ()> + Send + 'static,
    cancel: CancellationToken,
    stats: Arc<Mutex<EvictionSchedulerStats>>,
    swept: Arc<Notify>,
) {
    let mut ticks = Box::pin(ticks);
    loop {
        tokio::select! {
            biased;
            () = cancel.cancelled() => break,
            tick = ticks.next() => {
                if tick.is_none() {
                    tracing::debug!(
                        target: "openjiuwen_harness::subagents",
                        "subagent eviction scheduler tick stream ended"
                    );
                    break;
                }
            }
        }
        let sweep = std::panic::AssertUnwindSafe(
            runtime.sweep_expired_idle(config.idle_ttl_ms, config.max_per_sweep),
        )
        .catch_unwind()
        .await;
        match sweep {
            Ok(report) => record_sweep(&stats, &report),
            Err(payload) => {
                stats.lock().failures += 1;
                tracing::error!(
                    target: "openjiuwen_harness::subagents",
                    panic = panic_message(payload.as_ref()),
                    "subagent eviction sweep panicked"
                );
            }
        }
        swept.notify_waiters();
    }
    tracing::debug!(
        target: "openjiuwen_harness::subagents",
        "subagent eviction scheduler stopped"
    );
}

/// Adds one finished sweep to `stats` and stamps the last evicted id.
fn record_sweep(stats: &Mutex<EvictionSchedulerStats>, report: &SweepReport) {
    let mut stats = stats.lock();
    stats.sweeps += 1;
    stats.evicted += u64::try_from(report.evicted.len()).unwrap_or(u64::MAX);
    stats.skipped_running += u64::try_from(report.skipped_running).unwrap_or(u64::MAX);
    if let Some(last) = report.evicted.last() {
        stats.last_evicted = Some(last.subagent_id.clone());
    }
}

/// Renders the payload of a caught panic, `"unknown panic"` for a non-string payload.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> &str {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        return message;
    }
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.as_str();
    }
    "unknown panic"
}

/// Handle of one running eviction scheduler: the task, its cancellation token and its counters.
///
/// The handle owns the driver task. Dropping it cancels the loop, which then ends at the next
/// tick/cancellation boundary — that is enough to stop a leak but does not wait, so a host that
/// needs the task to be finished (a test asserting counters, a graceful shutdown) calls
/// [`EvictionSchedulerHandle::shutdown`] instead.
pub struct EvictionSchedulerHandle {
    /// Configuration the scheduler was started with.
    config: EvictionSchedulerConfig,
    /// Cancellation token of the driver task.
    cancel: CancellationToken,
    /// The driver task, until [`EvictionSchedulerHandle::shutdown`] takes it.
    join: Option<JoinHandle<()>>,
    /// Counters the driver task updates after every sweep.
    stats: Arc<Mutex<EvictionSchedulerStats>>,
    /// Signal published after every completed sweep.
    swept: Arc<Notify>,
}

impl EvictionSchedulerHandle {
    /// Returns the configuration this scheduler was started with.
    #[must_use]
    pub fn config(&self) -> EvictionSchedulerConfig {
        self.config
    }

    /// Returns a snapshot copy of the scheduler counters.
    #[must_use]
    pub fn stats(&self) -> EvictionSchedulerStats {
        self.stats.lock().clone()
    }

    /// Whether the driver task is still alive.
    ///
    /// A finished task means the scheduler was cancelled, shut down, or its tick stream ended;
    /// the counters then stay frozen at their final values.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.join.as_ref().is_some_and(|join| !join.is_finished())
    }

    /// Requests a graceful stop; synchronous, idempotent and safe to call repeatedly.
    ///
    /// The driver task ends after the sweep it is currently running, so this may return before
    /// the task is finished; await [`EvictionSchedulerHandle::shutdown`] to be sure.
    pub fn cancel(&self) {
        self.cancel.cancel();
    }

    /// Awaits the next completed sweep and returns the new [`EvictionSchedulerStats::sweeps`].
    ///
    /// Deterministic test hook: a caller that feeds the tick stream can await the sweep that
    /// tick triggered instead of sleeping. Awaiting a scheduler whose task already ended
    /// returns the (frozen) current count instead of hanging; callers should still bound the
    /// wait with a timeout.
    #[must_use]
    pub async fn wait_for_sweep(&self) -> u64 {
        let start = self.stats().sweeps;
        loop {
            // Register before re-reading, so a sweep that completes between the two is seen.
            let notified = self.swept.notified();
            let sweeps = self.stats().sweeps;
            if sweeps > start || !self.is_running() {
                return sweeps;
            }
            notified.await;
        }
    }

    /// Cancels the scheduler, awaits the driver task and returns the final counters.
    ///
    /// The in-flight sweep is awaited to completion (nothing is aborted mid-way) and a panicking
    /// driver task is reported with `tracing::warn!` instead of propagating. Both
    /// [`EvictionSchedulerHandle::cancel`] and the [`Drop`] implementation already cancelled the
    /// token, so the task exits as soon as the current sweep returns.
    #[must_use]
    pub async fn shutdown(mut self) -> EvictionSchedulerStats {
        self.cancel.cancel();
        if let Some(join) = self.join.take() {
            if let Err(error) = join.await {
                tracing::warn!(
                    target: "openjiuwen_harness::subagents",
                    %error,
                    "subagent eviction scheduler task ended unexpectedly"
                );
            }
        }
        self.stats()
    }
}

impl Drop for EvictionSchedulerHandle {
    /// Cancels the token so a dropped handle never leaves a running loop behind.
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use serde_json::json;
    use tokio::sync::{mpsc, Notify};
    use tokio::time::timeout;

    use super::*;
    use crate::subagents::test_support::{build_registry, Behavior, FakeFactory};
    use crate::subagents::{
        ActivityKind, Clock, ManualClock, SubAgentSpawnRequest, TranscriptKind, TranscriptSummary,
        EVICTION_REASON,
    };

    /// Frozen instant every test clock starts at.
    const T0: i64 = 1_700_000_000_000;
    /// Upper bound of every wait in this module.
    const BOUND: Duration = Duration::from_secs(5);
    /// Number of `evicted` activity frames one eviction records.
    const EVICTED_FRAMES: usize = 1;

    /// Builds a runtime over the built-in presets sharing one injected manual clock.
    fn manual_runtime(behavior: Behavior) -> (Arc<SubAgentRuntime>, Arc<ManualClock>) {
        let factory = FakeFactory::new(behavior);
        let clock = Arc::new(ManualClock::new(T0));
        let injected: Arc<dyn Clock> = clock.clone();
        let runtime =
            SubAgentRuntime::new(build_registry(), factory, "parent-session").with_clock(injected);
        (Arc::new(runtime), clock)
    }

    /// Spawns one `code_agent` instance running `task`.
    async fn spawn(runtime: &SubAgentRuntime, task: &str) -> SubAgentId {
        runtime
            .spawn(SubAgentSpawnRequest::new("code_agent", task))
            .await
            .unwrap()
    }

    /// Waits until `id` finished its turn and asserts it completed.
    async fn wait_completed(runtime: &SubAgentRuntime, id: &str) {
        let outcomes = runtime.wait(&[id.to_owned()], BOUND).await;
        assert_eq!(outcomes[0].status, SubagentStatus::Completed);
    }

    /// Yields until the instance reports a running turn.
    async fn wait_running(runtime: &SubAgentRuntime, id: &str) {
        for _ in 0..200 {
            if runtime.get_status(id) == SubagentStatus::Running {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!("subagent did not start: {id}");
    }

    /// Counts the activity frames of `kind` recorded for `id`.
    fn count_frames(runtime: &SubAgentRuntime, id: &str, kind: ActivityKind) -> usize {
        runtime
            .activity(id)
            .unwrap_or_default()
            .iter()
            .filter(|frame| frame.kind == kind)
            .count()
    }

    /// Counts the transcript entries of `kind` recorded for `id`.
    fn count_entries(runtime: &SubAgentRuntime, id: &str, kind: TranscriptKind) -> usize {
        runtime
            .transcript(id)
            .unwrap_or_default()
            .iter()
            .filter(|entry| entry.kind == kind)
            .count()
    }

    #[tokio::test]
    async fn sweep_skips_running_instances() {
        let gate = Arc::new(Notify::new());
        let (runtime, _clock) = manual_runtime(Behavior::Blocked(Arc::clone(&gate)));
        let id = spawn(&runtime, "held").await;
        wait_running(&runtime, &id).await;

        // A TTL of 0 makes every non-running instance a candidate, so the only reason to leave
        // this one alone is its running turn.
        let report = runtime.sweep_expired_idle(0, None).await;
        assert!(report.is_empty());
        assert_eq!(report.considered, 0);
        assert_eq!(report.skipped_running, 1);
        assert_eq!(runtime.get_status(&id), SubagentStatus::Running);
        assert_eq!(count_entries(&runtime, &id, TranscriptKind::Evicted), 0);
        gate.notify_waiters();
    }

    #[tokio::test]
    async fn sweep_evicts_idle_instances_past_the_ttl() {
        let (runtime, clock) = manual_runtime(Behavior::Echo);
        let id = spawn(&runtime, "one").await;
        wait_completed(&runtime, &id).await;

        let ttl = 1_000;
        clock.advance(ttl - 1);
        let report = runtime.sweep_expired_idle(ttl, None).await;
        assert!(report.is_empty());
        assert_eq!(report.considered, 0);
        assert_eq!(runtime.get_status(&id), SubagentStatus::Completed);

        clock.advance(1);
        let report = runtime.sweep_expired_idle(ttl, None).await;
        assert_eq!(report.considered, 1);
        assert_eq!(report.len(), 1);
        let record = &report.evicted[0];
        assert_eq!(record.subagent_id, id);
        assert_eq!(record.previous_status, SubagentStatus::Completed);
        assert_eq!(record.reason, EVICTION_REASON);
        assert_eq!(count_entries(&runtime, &id, TranscriptKind::Evicted), 1);
        assert_eq!(
            count_frames(&runtime, &id, ActivityKind::Evicted),
            EVICTED_FRAMES
        );
        let summary = runtime.activity_summary(&id).unwrap();
        assert_eq!(summary.last_kind, Some(ActivityKind::Evicted));
        assert!(summary.terminal);
        assert_eq!(runtime.get_status(&id), SubagentStatus::Closed);
    }

    #[tokio::test]
    async fn sweep_honours_max_per_sweep_in_lru_order() {
        let (runtime, clock) = manual_runtime(Behavior::Echo);
        let mut ids = Vec::new();
        for task in ["one", "two", "three"] {
            let id = spawn(&runtime, task).await;
            wait_completed(&runtime, &id).await;
            ids.push(id);
            // Stagger the last-activity stamps by 10 ms so the LRU order is unambiguous.
            clock.advance(10);
        }

        let report = runtime.sweep_expired_idle(0, Some(1)).await;
        assert_eq!(report.considered, 3);
        assert_eq!(report.len(), 1);
        assert_eq!(report.evicted[0].subagent_id, ids[0]);
        assert_eq!(runtime.get_status(&ids[0]), SubagentStatus::Closed);
        assert_eq!(runtime.get_status(&ids[1]), SubagentStatus::Completed);
        assert_eq!(runtime.get_status(&ids[2]), SubagentStatus::Completed);

        let report = runtime.sweep_expired_idle(0, None).await;
        let evicted: Vec<SubAgentId> = report
            .evicted
            .iter()
            .map(|record| record.subagent_id.clone())
            .collect();
        assert_eq!(evicted, vec![ids[1].clone(), ids[2].clone()]);
        assert_eq!(runtime.get_status(&ids[1]), SubagentStatus::Closed);
        assert_eq!(runtime.get_status(&ids[2]), SubagentStatus::Closed);

        // A `Some(0)` cap evicts nothing.
        assert!(runtime.sweep_expired_idle(0, Some(0)).await.is_empty());
    }

    #[tokio::test]
    async fn concurrent_sweeps_write_exactly_one_tombstone() {
        let (runtime, _clock) = manual_runtime(Behavior::Echo);
        let id = spawn(&runtime, "one").await;
        wait_completed(&runtime, &id).await;

        let (first, second) = tokio::join!(
            runtime.sweep_expired_idle(0, None),
            runtime.sweep_expired_idle(0, None)
        );
        assert_eq!(first.len() + second.len(), 1);
        assert_eq!(runtime.get_status(&id), SubagentStatus::Closed);
        assert_eq!(count_entries(&runtime, &id, TranscriptKind::Evicted), 1);
        assert_eq!(
            count_frames(&runtime, &id, ActivityKind::Evicted),
            EVICTED_FRAMES
        );
        assert_eq!(runtime.capacity().0, 0);
    }

    #[tokio::test]
    async fn scheduler_evicts_on_injected_ticks_and_shuts_down() {
        let (runtime, clock) = manual_runtime(Behavior::Echo);
        let id = spawn(&runtime, "one").await;
        wait_completed(&runtime, &id).await;

        let (sender, receiver) = mpsc::channel::<()>(4);
        let ticks = tokio_stream::wrappers::ReceiverStream::new(receiver);
        let config = EvictionSchedulerConfig::new(Duration::from_secs(60), 1_000);
        let handle = runtime.start_eviction_scheduler_with_ticks(config, ticks);
        assert!(handle.is_running());
        assert_eq!(handle.config(), config);
        assert_eq!(handle.stats().sweeps, 0);

        clock.advance(1_000);
        sender.send(()).await.unwrap();
        let sweeps = timeout(BOUND, handle.wait_for_sweep()).await.unwrap();
        assert_eq!(sweeps, 1);
        let stats = handle.stats();
        assert_eq!(stats.evicted, 1);
        assert_eq!(stats.skipped_running, 0);
        assert_eq!(stats.last_evicted.as_deref(), Some(id.as_str()));
        assert_eq!(runtime.get_status(&id), SubagentStatus::Closed);

        let before_shutdown = handle.stats();
        let final_stats = timeout(BOUND, handle.shutdown()).await.unwrap();
        assert_eq!(final_stats, before_shutdown);
        assert_eq!(final_stats.sweeps, 1);
    }

    #[tokio::test]
    async fn interval_scheduler_waits_one_interval_before_the_first_sweep() {
        let (runtime, clock) = manual_runtime(Behavior::Echo);
        let id = spawn(&runtime, "one").await;
        wait_completed(&runtime, &id).await;

        // Short period so the test stays bounded, but the first tick is still one period away.
        let config = EvictionSchedulerConfig::new(Duration::from_millis(20), 1_000);
        let handle = runtime.start_eviction_scheduler(config);
        assert_eq!(handle.stats().sweeps, 0);

        clock.advance(1_000);
        let sweeps = timeout(BOUND, handle.wait_for_sweep()).await.unwrap();
        assert!(sweeps >= 1);
        assert_eq!(runtime.get_status(&id), SubagentStatus::Closed);
        let stats = timeout(BOUND, handle.shutdown()).await.unwrap();
        assert!(stats.sweeps >= 1);
        assert_eq!(stats.evicted, 1);
        assert_eq!(stats.failures, 0);
    }

    #[tokio::test]
    async fn shutdown_drains_without_leaking() {
        let (runtime, _clock) = manual_runtime(Behavior::Echo);
        let (sender, receiver) = mpsc::channel::<()>(1);
        let ticks = tokio_stream::wrappers::ReceiverStream::new(receiver);
        let handle =
            runtime.start_eviction_scheduler_with_ticks(EvictionSchedulerConfig::default(), ticks);
        drop(sender);
        // Cancelling twice is a no-op, and the ended tick stream also stops the loop.
        handle.cancel();
        handle.cancel();
        let stats = timeout(BOUND, handle.shutdown()).await.unwrap();
        assert_eq!(stats.sweeps, 0);
        assert_eq!(stats.failures, 0);
        assert_eq!(stats.last_evicted, None);

        // The runtime keeps working after a scheduler stopped.
        let id = spawn(&runtime, "after").await;
        wait_completed(&runtime, &id).await;
        assert_eq!(runtime.get_status(&id), SubagentStatus::Completed);
    }

    #[tokio::test]
    async fn dropping_a_handle_cancels_the_loop() {
        let (runtime, _clock) = manual_runtime(Behavior::Echo);
        let (_sender, receiver) = mpsc::channel::<()>(1);
        let ticks = tokio_stream::wrappers::ReceiverStream::new(receiver);
        let handle =
            runtime.start_eviction_scheduler_with_ticks(EvictionSchedulerConfig::default(), ticks);
        assert!(handle.is_running());
        drop(handle);
        // The token is cancelled synchronously; the runtime is left untouched.
        assert_eq!(runtime.capacity().0, 0);
    }

    #[test]
    fn config_defaults_and_clamping() {
        assert_eq!(DEFAULT_EVICTION_INTERVAL, Duration::from_secs(30));
        assert_eq!(DEFAULT_IDLE_TTL_MS, 900_000);

        let config = EvictionSchedulerConfig::default();
        assert_eq!(config.interval(), DEFAULT_EVICTION_INTERVAL);
        assert_eq!(config.idle_ttl_ms(), DEFAULT_IDLE_TTL_MS);
        assert_eq!(config.max_per_sweep(), None);

        let clamped = EvictionSchedulerConfig::new(Duration::from_millis(5), -1);
        assert_eq!(clamped.interval(), Duration::from_millis(5));
        assert_eq!(clamped.idle_ttl_ms(), 0);

        let custom = EvictionSchedulerConfig::default()
            .with_interval(Duration::from_secs(2))
            .with_idle_ttl_ms(-10)
            .with_max_per_sweep(Some(3));
        assert_eq!(custom.interval(), Duration::from_secs(2));
        assert_eq!(custom.idle_ttl_ms(), 0);
        assert_eq!(custom.max_per_sweep(), Some(3));

        let value = serde_json::to_value(custom).unwrap();
        assert_eq!(value["idle_ttl_ms"], json!(0));
        assert_eq!(value["max_per_sweep"], json!(3));
        assert_eq!(
            serde_json::from_value::<EvictionSchedulerConfig>(value).unwrap(),
            custom
        );

        let report = SweepReport::default();
        assert!(report.is_empty());
        assert_eq!(report.len(), 0);
        let report = SweepReport {
            considered: 2,
            evicted: vec![SubagentEviction {
                subagent_id: "parent_sub_code_agent_0000".to_owned(),
                previous_status: SubagentStatus::Completed,
                reason: EVICTION_REASON.to_owned(),
                transcript: TranscriptSummary::default(),
            }],
            skipped_running: 1,
        };
        let value = serde_json::to_value(&report).unwrap();
        assert_eq!(value["considered"], json!(2));
        assert_eq!(value["skipped_running"], json!(1));
        assert_eq!(value["evicted"][0]["reason"], json!(EVICTION_REASON));
        assert_eq!(report.len(), 1);
        assert!(!report.is_empty());
        assert_eq!(
            serde_json::from_value::<SweepReport>(value).unwrap(),
            report
        );
    }

    #[test]
    fn stats_accessors_and_snapshot_copy() {
        let stats = EvictionSchedulerStats {
            sweeps: 2,
            evicted: 3,
            skipped_running: 1,
            failures: 4,
            last_evicted: Some("id".to_owned()),
        };
        assert_eq!(stats.sweeps(), 2);
        assert_eq!(stats.evicted(), 3);
        assert_eq!(stats.skipped_running(), 1);
        assert_eq!(stats.failures(), 4);
        assert_eq!(stats.last_evicted(), Some("id"));
        let value = serde_json::to_value(stats.clone()).unwrap();
        assert_eq!(value["last_evicted"], json!("id"));
        assert_eq!(
            serde_json::from_value::<EvictionSchedulerStats>(value).unwrap(),
            stats
        );
        assert_eq!(
            EvictionSchedulerStats::default().last_evicted(),
            None,
            "an untouched scheduler reports no eviction"
        );
    }
}
