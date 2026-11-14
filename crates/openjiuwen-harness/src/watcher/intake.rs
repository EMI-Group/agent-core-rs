//! The watcher's bounded queues and the drop policy that keeps the loss visible.
//!
//! Two hops of the pipeline are bounded, so no producer can grow memory without
//! limit behind a slow consumer:
//!
//! - **intake** — the queue between the `notify` callback and the debounce loop
//!   ([`WorkspaceWatcher`](crate::watcher::WorkspaceWatcher)'s `intake_capacity`);
//! - **dispatch** — the queue between the debounce loop and the reload
//!   dispatcher (`dispatch_capacity`).
//!
//! Both use a **drop-newest** policy: an item that arrives while the queue is
//! full is discarded, so the queue keeps FIFO order for everything it admitted.
//! Drop-newest is the only policy a synchronous OS callback can honour without
//! blocking the notification thread (a drop-oldest queue would need a lock
//! inside that callback).
//!
//! A drop is never silent:
//!
//! - [`DropCounters`] counts every drop per stage and is shared with
//!   [`WatcherHandle`](crate::watcher::WatcherHandle), which reports the totals
//!   through `dropped_notifications()` / `dropped_dispatches()`;
//! - each drop is logged at `debug` with its reason;
//! - [`DropReporter`] publishes a typed
//!   [`WatchUpdate::Dropped`](crate::watcher::WatchUpdate::Dropped) notice and
//!   logs one `warn`, at most once per notice interval. A sustained overflow
//!   therefore produces one notice per interval instead of an error storm, and
//!   the notice carries both the number of notifications it covers and the
//!   running total.
//!
//! Because a dropped notification may have been the only one for a path, a
//! consumer that receives a drop notice must reload from disk instead of
//! applying the deltas it did see.
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::broadcast;

use crate::watcher::event::{DropStage, DroppedUpdates, WatchUpdate};

/// Shared counters of the notifications each stage dropped.
///
/// Cloning shares the counters, so the OS callback thread, the intake task and
/// the handle all observe the same totals.
#[derive(Debug, Clone, Default)]
pub(crate) struct DropCounters {
    intake: Arc<AtomicU64>,
    dispatch: Arc<AtomicU64>,
}

impl DropCounters {
    /// Creates a pair of zeroed counters.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Returns how many raw notifications the intake queue dropped.
    pub(crate) fn intake(&self) -> u64 {
        self.intake.load(Ordering::Relaxed)
    }

    /// Returns how many debounced events the dispatch queue dropped.
    pub(crate) fn dispatch(&self) -> u64 {
        self.dispatch.load(Ordering::Relaxed)
    }

    /// Returns the total of `stage`.
    pub(crate) fn total(&self, stage: DropStage) -> u64 {
        match stage {
            DropStage::Intake => self.intake(),
            DropStage::Dispatch => self.dispatch(),
        }
    }

    /// Counts one intake drop and returns the new total.
    pub(crate) fn record_intake(&self) -> u64 {
        self.intake
            .fetch_add(1, Ordering::Relaxed)
            .saturating_add(1)
    }

    /// Counts one dispatch drop and returns the new total.
    pub(crate) fn record_dispatch(&self) -> u64 {
        self.dispatch
            .fetch_add(1, Ordering::Relaxed)
            .saturating_add(1)
    }
}

/// Throttles the drop notices of one watcher.
///
/// The reporter remembers the totals it already announced per stage and only
/// publishes once `interval` has passed since the previous notice, so a queue
/// that stays full produces a notice per interval rather than one per drop.
/// [`DropReporter::flush`] ignores the throttle and is what shutdown uses.
pub(crate) struct DropReporter {
    interval: Duration,
    last: Option<Instant>,
    reported_intake: u64,
    reported_dispatch: u64,
}

impl DropReporter {
    /// Creates a reporter that announces at most once per `interval`.
    pub(crate) fn new(interval: Duration) -> Self {
        Self {
            interval,
            last: None,
            reported_intake: 0,
            reported_dispatch: 0,
        }
    }

    /// Returns whether a stage dropped something that was not announced yet.
    pub(crate) fn pending(&self, counters: &DropCounters) -> bool {
        counters.intake() > self.reported_intake || counters.dispatch() > self.reported_dispatch
    }

    /// Returns when [`Self::report`] next has something to say.
    ///
    /// `None` means nothing is pending, so the caller has no notice deadline to
    /// wait for; a returned instant is never in the past, so the deadline can be
    /// handed straight to a timer.
    pub(crate) fn next_deadline(&self, counters: &DropCounters, now: Instant) -> Option<Instant> {
        if !self.pending(counters) {
            return None;
        }
        Some(match self.last {
            Some(last) => last
                .checked_add(self.interval)
                .map_or(now, |deadline| deadline.max(now)),
            None => now,
        })
    }

    /// Publishes the pending notices unless the interval has not elapsed yet.
    pub(crate) fn report(
        &mut self,
        events: &broadcast::Sender<WatchUpdate>,
        counters: &DropCounters,
        now: Instant,
    ) -> bool {
        if !self.pending(counters) {
            return false;
        }
        if let Some(last) = self.last {
            let elapsed = now.checked_duration_since(last).unwrap_or_default();
            if elapsed < self.interval {
                return false;
            }
        }
        self.last = Some(now);
        self.publish(events, counters)
    }

    /// Publishes every pending notice, ignoring the throttle.
    ///
    /// Used on shutdown, so the last window of drops is still reported.
    pub(crate) fn flush(
        &mut self,
        events: &broadcast::Sender<WatchUpdate>,
        counters: &DropCounters,
        now: Instant,
    ) -> bool {
        if !self.pending(counters) {
            return false;
        }
        self.last = Some(now);
        self.publish(events, counters)
    }

    /// Publishes one notice per stage that dropped since the previous notice.
    fn publish(
        &mut self,
        events: &broadcast::Sender<WatchUpdate>,
        counters: &DropCounters,
    ) -> bool {
        let mut published = false;
        for stage in [DropStage::Intake, DropStage::Dispatch] {
            let total = counters.total(stage);
            let reported = self.reported(stage);
            if total <= reported {
                continue;
            }
            let dropped = total.saturating_sub(reported);
            self.set_reported(stage, total);
            let reason = match stage {
                DropStage::Intake => "the OS notification intake queue was full",
                DropStage::Dispatch => "the reload dispatch queue was full",
            };
            tracing::warn!(
                stage = stage.as_str(),
                dropped,
                total,
                "workspace watcher dropped notifications because {reason}"
            );
            let notice = WatchUpdate::Dropped(DroppedUpdates {
                stage,
                dropped,
                total,
            });
            // A watcher without subscribers is not an error; the counters on the
            // handle still carry the loss.
            published |= events.send(notice).is_ok();
        }
        published
    }

    /// Returns the total this reporter already announced for `stage`.
    fn reported(&self, stage: DropStage) -> u64 {
        match stage {
            DropStage::Intake => self.reported_intake,
            DropStage::Dispatch => self.reported_dispatch,
        }
    }

    /// Records the total announced for `stage`.
    fn set_reported(&mut self, stage: DropStage, total: u64) {
        match stage {
            DropStage::Intake => self.reported_intake = total,
            DropStage::Dispatch => self.reported_dispatch = total,
        }
    }
}

impl std::fmt::Debug for DropReporter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DropReporter")
            .field("interval", &self.interval)
            .field("reported_intake", &self.reported_intake)
            .field("reported_dispatch", &self.reported_dispatch)
            .finish_non_exhaustive()
    }
}

/// Offers one raw notification to the bounded intake queue.
///
/// Runs on the `notify` backend thread, so it must neither block nor panic:
/// [`try_send`](tokio::sync::mpsc::Sender::try_send) returns immediately, and a
/// full queue drops the **newest** notification while counting it in
/// `counters`. A closed queue means the watcher task is already gone; the
/// notification is discarded without counting, because that is shutdown and not
/// a drop.
///
/// Returns whether the notification was queued.
#[cfg(feature = "notify")]
pub(crate) fn push_raw(
    sender: &tokio::sync::mpsc::Sender<notify::Result<notify::Event>>,
    counters: &DropCounters,
    incoming: notify::Result<notify::Event>,
) -> bool {
    use tokio::sync::mpsc::error::TrySendError;

    match sender.try_send(incoming) {
        Ok(()) => true,
        Err(error) => {
            if matches!(error, TrySendError::Full(_)) {
                let total = counters.record_intake();
                tracing::debug!(
                    total,
                    "workspace watcher intake queue is full; dropping the newest notification"
                );
            }
            false
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    /// Builds a counter pair with the given totals.
    fn counters(intake: u64, dispatch: u64) -> DropCounters {
        let counters = DropCounters::new();
        for _ in 0..intake {
            counters.record_intake();
        }
        for _ in 0..dispatch {
            counters.record_dispatch();
        }
        counters
    }

    /// Reads everything a receiver has, without waiting.
    fn drain(receiver: &mut broadcast::Receiver<WatchUpdate>) -> Vec<WatchUpdate> {
        let mut updates = Vec::new();
        while let Ok(update) = receiver.try_recv() {
            updates.push(update);
        }
        updates
    }

    #[test]
    fn counters_are_shared_between_clones() {
        let counters = DropCounters::new();
        let clone = counters.clone();
        assert_eq!(clone.record_intake(), 1);
        assert_eq!(counters.record_intake(), 2);
        assert_eq!(counters.intake(), 2);
        assert_eq!(counters.dispatch(), 0);
        assert_eq!(counters.total(DropStage::Intake), 2);
        assert_eq!(counters.total(DropStage::Dispatch), 0);
    }

    #[tokio::test]
    async fn a_first_drop_is_announced_immediately() {
        let counters = counters(2, 1);
        let (events, mut receiver) = broadcast::channel::<WatchUpdate>(8);
        let mut reporter = DropReporter::new(Duration::from_millis(200));
        let now = Instant::now();

        assert!(reporter.pending(&counters));
        assert_eq!(reporter.next_deadline(&counters, now), Some(now));
        assert!(reporter.report(&events, &counters, now));

        let updates = drain(&mut receiver);
        assert_eq!(updates.len(), 2, "{updates:?}");
        assert_eq!(
            updates[0],
            WatchUpdate::Dropped(DroppedUpdates {
                stage: DropStage::Intake,
                dropped: 2,
                total: 2,
            })
        );
        assert_eq!(
            updates[1],
            WatchUpdate::Dropped(DroppedUpdates {
                stage: DropStage::Dispatch,
                dropped: 1,
                total: 1,
            })
        );

        // Nothing new is pending, so there is nothing to schedule or announce.
        assert!(!reporter.pending(&counters));
        assert_eq!(reporter.next_deadline(&counters, now), None);
        assert!(!reporter.report(&events, &counters, now));
        assert_eq!(drain(&mut receiver), Vec::<WatchUpdate>::new());
    }

    #[tokio::test]
    async fn drops_inside_one_interval_coalesce_into_one_notice() {
        let counters = counters(1, 0);
        let (events, mut receiver) = broadcast::channel::<WatchUpdate>(8);
        let mut reporter = DropReporter::new(Duration::from_millis(200));
        let start = Instant::now();

        assert!(reporter.report(&events, &counters, start));
        assert_eq!(drain(&mut receiver).len(), 1);

        for _ in 0..4 {
            counters.record_intake();
        }
        // Still inside the interval: the notice is withheld ...
        let inside = start + Duration::from_millis(50);
        assert!(!reporter.report(&events, &counters, inside));
        assert_eq!(drain(&mut receiver), Vec::<WatchUpdate>::new());
        // ... and the deadline says when it becomes due.
        assert_eq!(
            reporter.next_deadline(&counters, inside),
            Some(start + Duration::from_millis(200))
        );

        let due = start + Duration::from_millis(200);
        assert!(reporter.report(&events, &counters, due));
        let updates = drain(&mut receiver);
        assert_eq!(
            updates,
            vec![WatchUpdate::Dropped(DroppedUpdates {
                stage: DropStage::Intake,
                dropped: 4,
                total: 5,
            })]
        );

        // A deadline for an overdue notice is never in the past.
        counters.record_dispatch();
        let late = due + Duration::from_secs(1);
        assert_eq!(reporter.next_deadline(&counters, late), Some(late));
    }

    #[tokio::test]
    async fn the_shutdown_flush_ignores_the_throttle() {
        let counters = counters(1, 0);
        let (events, mut receiver) = broadcast::channel::<WatchUpdate>(8);
        let mut reporter = DropReporter::new(Duration::from_secs(60));
        let start = Instant::now();

        assert!(reporter.report(&events, &counters, start));
        assert_eq!(drain(&mut receiver).len(), 1);

        counters.record_intake();
        assert!(!reporter.report(&events, &counters, start));
        assert!(reporter.flush(&events, &counters, start));
        assert_eq!(
            drain(&mut receiver),
            vec![WatchUpdate::Dropped(DroppedUpdates {
                stage: DropStage::Intake,
                dropped: 1,
                total: 2,
            })]
        );
        // A flush with nothing pending publishes nothing.
        assert!(!reporter.flush(&events, &counters, start));
    }

    #[tokio::test]
    async fn an_unsubscribed_watcher_still_updates_its_counters() {
        let counters = counters(3, 0);
        let (events, receiver) = broadcast::channel::<WatchUpdate>(8);
        drop(receiver);
        let mut reporter = DropReporter::new(Duration::from_millis(1));

        assert!(!reporter.report(&events, &counters, Instant::now()));
        assert_eq!(counters.intake(), 3);
        assert!(!reporter.pending(&counters));
    }

    #[cfg(feature = "notify")]
    #[test]
    fn a_full_intake_queue_drops_and_counts_the_newest() {
        let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
        let counters = DropCounters::new();
        let event = || Ok(notify::Event::new(notify::EventKind::Any));

        assert!(push_raw(&sender, &counters, event()));
        assert!(!push_raw(&sender, &counters, event()));
        assert!(!push_raw(&sender, &counters, event()));
        assert_eq!(counters.intake(), 2);
        // The queue kept the first notification, in order.
        assert!(receiver.try_recv().is_ok());
        assert!(receiver.try_recv().is_err());
    }

    #[cfg(feature = "notify")]
    #[test]
    fn a_closed_intake_queue_is_not_counted_as_a_drop() {
        let (sender, receiver) = tokio::sync::mpsc::channel(1);
        drop(receiver);
        let counters = DropCounters::new();

        assert!(!push_raw(
            &sender,
            &counters,
            Ok(notify::Event::new(notify::EventKind::Any))
        ));
        assert_eq!(counters.intake(), 0);
    }
}
