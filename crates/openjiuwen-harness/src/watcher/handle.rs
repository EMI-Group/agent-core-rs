//! Bounded event distribution and clean shutdown for the workspace watcher.
//!
//! A running watcher may fan its updates out to several consumers, and a
//! consumer that stops reading must never stall the producer. Distribution is
//! therefore a `tokio::sync::broadcast` ring buffer: [`Subscription`] wraps one
//! receiver, and [`WatcherHandle`] owns the sender, the cancellation token and
//! the background task of the OS watcher.
//!
//! # Drop-oldest semantics
//!
//! The ring buffer holds at most `capacity` updates (the value rounded up to a
//! power of two, as `tokio` does). Publishing never blocks: when a stalled
//! consumer is still behind and the buffer overflows, the **oldest** updates
//! are discarded for that consumer only. The loss is never silent — the
//! receiver reports it as a lag, this module counts it in
//! [`Subscription::lagged_updates`] and logs one `tracing::warn!` per lag
//! (never one per dropped update, so a stalled consumer cannot flood the log).
//! Consumers that need every event should either read promptly or treat a
//! non-zero [`Subscription::lagged_updates`] as "reload from disk" rather than
//! "apply these deltas".
//!
//! # Shutdown
//!
//! [`WatcherHandle::shutdown`] cancels the token, drains the pending debounce
//! window through the watcher task, awaits the reload dispatcher and then awaits
//! both tasks, so no update is produced after `shutdown()` returns. The call is
//! idempotent, and dropping the handle cancels cooperatively as well.
//!
//! # Dropped notifications
//!
//! The watcher's two internal queues are bounded (see `intake.rs`), so a
//! producer that outruns its consumer loses notifications instead of growing
//! memory. The loss is visible three ways: a
//! [`WatchUpdate::Dropped`](crate::watcher::WatchUpdate::Dropped) notice on the
//! subscription, a `tracing::warn!` per notice, and the totals this handle
//! reports through [`WatcherHandle::dropped_notifications`] and
//! [`WatcherHandle::dropped_dispatches`].
use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;
use openjiuwen_common::error::{Error, Result};
use tokio::sync::{broadcast, Mutex};
use tokio::task::JoinHandle;
use tokio_util::sync::{CancellationToken, ReusableBoxFuture};

use crate::watcher::event::WatchUpdate;
use crate::watcher::intake::DropCounters;

/// One in-flight receive: the outcome plus the receiver to poll next.
///
/// `tokio`'s broadcast receiver only exposes `recv().await`, so a manual
/// [`Stream`] implementation parks that future in a reusable box (the same
/// technique `tokio_stream::wrappers::BroadcastStream` uses) instead of keeping
/// a `&mut self` borrow alive in a `'static` future.
type RecvState = (
    Result<WatchUpdate, broadcast::error::RecvError>,
    broadcast::Receiver<WatchUpdate>,
);

/// Waits for the next broadcast item and hands the receiver back.
async fn receive(mut receiver: broadcast::Receiver<WatchUpdate>) -> RecvState {
    let result = receiver.recv().await;
    (result, receiver)
}

/// One consumer's view of the watcher's update stream.
///
/// Yields [`WatchUpdate`] values in publication order and skips the updates a
/// stalled consumer lost, which are counted in [`Self::lagged_updates`]. The
/// stream ends (`None`) once every [`WatcherHandle`] sender has been dropped.
///
/// The type is `Send + 'static`, so it can be moved into a spawned task or
/// boxed into a [`futures::stream::BoxStream`].
pub struct Subscription {
    inner: ReusableBoxFuture<'static, RecvState>,
    lagged: u64,
    ended: bool,
}

impl Subscription {
    /// Builds a subscription over an existing receiver.
    #[must_use]
    pub(crate) fn new(receiver: broadcast::Receiver<WatchUpdate>) -> Self {
        Self {
            inner: ReusableBoxFuture::new(receive(receiver)),
            lagged: 0,
            ended: false,
        }
    }

    /// Receives the next update, resynchronising across a lag.
    ///
    /// A [`broadcast::error::RecvError::Lagged`] is skipped rather than
    /// returned: the number of lost updates is added to
    /// [`Self::lagged_updates`] and one warning is logged. Returns `None` once
    /// the channel is closed, i.e. every handle was dropped or shut down.
    pub async fn recv(&mut self) -> Option<WatchUpdate> {
        futures::StreamExt::next(self).await
    }

    /// Returns how many updates this consumer missed because it fell behind.
    ///
    /// `0` means it observed every update published since it subscribed.
    #[must_use]
    pub fn lagged_updates(&self) -> u64 {
        self.lagged
    }

    /// Counts a lag and logs it once.
    fn record_lag(&mut self, skipped: u64) {
        self.lagged = self.lagged.saturating_add(skipped);
        tracing::warn!(
            skipped,
            total = self.lagged,
            "workspace watcher subscriber lagged behind; the oldest updates were dropped"
        );
    }
}

impl Stream for Subscription {
    type Item = WatchUpdate;

    /// Polls the next update; a lag is folded into the counter, as in
    /// [`Subscription::recv`]. The stream is fused: once it returned `None`
    /// every later poll returns `None` without touching the closed channel.
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.ended {
            return Poll::Ready(None);
        }
        loop {
            let Poll::Ready((result, receiver)) = this.inner.poll(cx) else {
                return Poll::Pending;
            };
            match result {
                Ok(update) => {
                    this.inner.set(receive(receiver));
                    return Poll::Ready(Some(update));
                }
                Err(broadcast::error::RecvError::Lagged(skipped)) => {
                    // The oldest updates are gone for this consumer; resubscribe
                    // to the ring buffer and keep reading from the oldest survivor.
                    this.inner.set(receive(receiver));
                    this.record_lag(skipped);
                }
                Err(broadcast::error::RecvError::Closed) => {
                    this.ended = true;
                    return Poll::Ready(None);
                }
            }
        }
    }
}

impl std::fmt::Debug for Subscription {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Subscription")
            .field("lagged", &self.lagged)
            .finish_non_exhaustive()
    }
}

/// Control handle of a running workspace watcher.
///
/// Owns the update sender, so subscriptions keep working as long as the handle
/// lives, plus the cancellation token, the tasks that drive the OS watcher and
/// its reload dispatcher, and the shared drop counters. See the module
/// documentation for the drop-oldest and shutdown contracts.
pub struct WatcherHandle {
    events: broadcast::Sender<WatchUpdate>,
    cancel: CancellationToken,
    tasks: Mutex<Vec<JoinHandle<()>>>,
    counters: DropCounters,
    target_count: usize,
}

impl WatcherHandle {
    /// Builds a handle from its parts.
    ///
    /// The watcher tasks must publish through `events` and stop once `cancel` is
    /// triggered; [`Self::shutdown`] awaits every task in `tasks` after
    /// cancelling, so a panic in any of them is reported through its result.
    // The OS-watcher backend and the tests of this module are the only callers.
    #[allow(dead_code)]
    pub(crate) fn new(
        events: broadcast::Sender<WatchUpdate>,
        cancel: CancellationToken,
        tasks: Vec<JoinHandle<()>>,
        counters: DropCounters,
        target_count: usize,
    ) -> Self {
        Self {
            events,
            cancel,
            tasks: Mutex::new(tasks),
            counters,
            target_count,
        }
    }

    /// Creates a new subscription for one consumer.
    ///
    /// Every subscriber sees the updates published after it subscribed; the
    /// ring buffer keeps at most the configured capacity for a slow one.
    #[must_use]
    pub fn subscribe(&self) -> Subscription {
        Subscription::new(self.events.subscribe())
    }

    /// Returns how many watch targets the watcher covers.
    #[must_use]
    pub fn target_count(&self) -> usize {
        self.target_count
    }

    /// Returns how many raw OS notifications the intake queue dropped.
    ///
    /// The intake queue sits between the `notify` callback and the debounce
    /// loop and drops the newest notification when it is full, so a burst the
    /// debounce loop cannot drain fast enough loses notifications instead of
    /// growing memory. A non-zero value means the deltas in the received events
    /// are incomplete: reload from disk.
    #[must_use]
    pub fn dropped_notifications(&self) -> u64 {
        self.counters.intake()
    }

    /// Returns how many debounced events the reload dispatch queue dropped.
    ///
    /// The dispatch queue sits between the debounce loop and the reload
    /// dispatcher and drops the newest event when it is full, which happens when
    /// a reload handler is slower than the events arrive. Subscribers still
    /// receive those events; only the matching reload bodies did not run.
    #[must_use]
    pub fn dropped_dispatches(&self) -> u64 {
        self.counters.dispatch()
    }

    /// Returns whether the watcher was cancelled or fully shut down.
    #[must_use]
    pub fn is_shutdown(&self) -> bool {
        self.cancel.is_cancelled()
    }

    /// Stops the watcher and waits for its tasks to finish.
    ///
    /// Cancelling lets the intake task flush its debounce window, so subscribers
    /// still receive the last pending events and the dispatcher still runs their
    /// reload handlers. Calling this again, or after the tasks already finished,
    /// returns `Ok(())`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::SysOp`](openjiuwen_common::error::Error::SysOp) when a
    /// watcher task panicked or was cancelled; the message carries the join
    /// failure.
    pub async fn shutdown(&self) -> Result<()> {
        self.cancel.cancel();
        let tasks = std::mem::take(&mut *self.tasks.lock().await);
        for task in tasks {
            task.await.map_err(|error| {
                Error::sys_op(format!("workspace watcher task failed: {error}"))
            })?;
        }
        Ok(())
    }
}

impl std::fmt::Debug for WatcherHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WatcherHandle")
            .field("target_count", &self.target_count)
            .field("shutdown", &self.is_shutdown())
            .field("dropped_notifications", &self.dropped_notifications())
            .field("dropped_dispatches", &self.dropped_dispatches())
            .finish_non_exhaustive()
    }
}

impl Drop for WatcherHandle {
    /// Cancels the watcher cooperatively when the handle is dropped without an
    /// explicit [`WatcherHandle::shutdown`].
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::watcher::event::{ChangeKind, WatchEvent, WatchTargetKind};
    use futures::StreamExt;
    use std::path::PathBuf;

    /// Builds an update with the given sequence number.
    fn update(sequence: u64) -> WatchUpdate {
        WatchUpdate::Event(WatchEvent {
            sequence,
            kind: WatchTargetKind::Skills,
            path: PathBuf::from("/w/skills/a.md"),
            change: ChangeKind::Modified,
            previous_path: None,
            label: None,
        })
    }

    /// Builds a handle whose task is a no-op.
    fn handle_with(
        capacity: usize,
    ) -> (
        broadcast::Sender<WatchUpdate>,
        WatcherHandle,
        CancellationToken,
    ) {
        let (events, _) = broadcast::channel(capacity);
        let cancel = CancellationToken::new();
        let handle = WatcherHandle::new(
            events.clone(),
            cancel.clone(),
            vec![tokio::spawn(async {})],
            DropCounters::new(),
            3,
        );
        (events, handle, cancel)
    }

    #[tokio::test]
    async fn two_updates_are_received_in_order() {
        let (events, handle, _cancel) = handle_with(8);
        let mut subscription = handle.subscribe();
        assert_eq!(handle.target_count(), 3);
        assert_eq!(subscription.lagged_updates(), 0);

        events.send(update(1)).unwrap();
        events.send(update(2)).unwrap();

        assert_eq!(subscription.recv().await.unwrap().sequence(), Some(1));
        assert_eq!(subscription.recv().await.unwrap().sequence(), Some(2));
        assert_eq!(subscription.lagged_updates(), 0);
    }

    #[tokio::test]
    async fn a_lagging_subscription_skips_the_oldest_updates() {
        let (events, handle, _cancel) = handle_with(2);
        let mut subscription = handle.subscribe();
        for sequence in 1..=5 {
            events.send(update(sequence)).unwrap();
        }

        // The ring buffer kept the two newest updates; the three older ones are
        // reported as a lag once, and reading continues from the oldest survivor.
        let first = subscription.recv().await.unwrap();
        assert_eq!(first.sequence(), Some(4));
        assert_eq!(subscription.lagged_updates(), 3);

        let newest = subscription.recv().await.unwrap();
        assert_eq!(newest.sequence(), Some(5));
        // The lag is not counted twice.
        assert_eq!(subscription.lagged_updates(), 3);
    }

    #[tokio::test]
    async fn the_stream_impl_matches_recv() {
        let (events, handle, _cancel) = handle_with(2);
        let subscription = handle.subscribe();
        for sequence in 1..=4 {
            events.send(update(sequence)).unwrap();
        }

        let collected: Vec<Option<u64>> = subscription
            .take(2)
            .map(|update| update.sequence())
            .collect()
            .await;
        assert_eq!(collected, vec![Some(3), Some(4)]);
    }

    #[tokio::test]
    async fn the_stream_ends_when_the_sender_is_dropped() {
        let (events, handle, _cancel) = handle_with(4);
        let mut subscription = handle.subscribe();
        drop(handle);
        drop(events);
        assert!(subscription.recv().await.is_none());
        assert!(subscription.next().await.is_none());
    }

    #[tokio::test]
    async fn shutdown_is_idempotent_and_flips_is_shutdown() {
        let (_events, handle, cancel) = handle_with(4);
        assert!(!handle.is_shutdown());
        handle.shutdown().await.unwrap();
        assert!(handle.is_shutdown());
        assert!(cancel.is_cancelled());

        // A second call finds no task left and still succeeds.
        handle.shutdown().await.unwrap();
        assert!(handle.is_shutdown());
    }

    #[tokio::test]
    async fn shutdown_awaits_the_watcher_task() {
        let (events, _) = broadcast::channel(4);
        let cancel = CancellationToken::new();
        let (finished_tx, finished_rx) = tokio::sync::oneshot::channel::<()>();
        let task_cancel = cancel.clone();
        let task = tokio::spawn(async move {
            task_cancel.cancelled().await;
            let _ = finished_tx.send(());
        });
        let handle = WatcherHandle::new(events, cancel, vec![task], DropCounters::new(), 0);

        handle.shutdown().await.unwrap();
        // The task completed, so its completion signal is already there.
        finished_rx.await.unwrap();
    }

    #[tokio::test]
    async fn shutdown_reports_a_failed_task() {
        let (events, _) = broadcast::channel(4);
        let cancel = CancellationToken::new();
        let task = tokio::spawn(async {
            panic!("watcher task exploded");
        });
        let handle = WatcherHandle::new(events, cancel, vec![task], DropCounters::new(), 0);

        let error = handle.shutdown().await.unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::SysOp);
        assert!(error.to_string().contains("workspace watcher task failed"));
    }

    #[tokio::test]
    async fn shutdown_reports_a_failed_dispatch_task_and_keeps_both_finished() {
        let (events, _) = broadcast::channel(4);
        let cancel = CancellationToken::new();
        let intake = tokio::spawn(async {});
        let dispatch = tokio::spawn(async {
            panic!("reload dispatcher exploded");
        });
        let handle = WatcherHandle::new(
            events,
            cancel,
            vec![intake, dispatch],
            DropCounters::new(),
            0,
        );

        let error = handle.shutdown().await.unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::SysOp);
        assert!(
            error.to_string().contains("reload dispatcher exploded"),
            "{error}"
        );
        // Both tasks were consumed, so a second shutdown has nothing left to await.
        assert!(handle.shutdown().await.is_ok());
        assert!(handle.is_shutdown());
    }

    #[tokio::test]
    async fn the_handle_reports_the_shared_drop_counters() {
        let (events, _) = broadcast::channel(4);
        let cancel = CancellationToken::new();
        let counters = DropCounters::new();
        let handle = WatcherHandle::new(events, cancel, vec![], counters.clone(), 0);
        assert_eq!(handle.dropped_notifications(), 0);
        assert_eq!(handle.dropped_dispatches(), 0);

        counters.record_intake();
        counters.record_intake();
        counters.record_dispatch();
        assert_eq!(handle.dropped_notifications(), 2);
        assert_eq!(handle.dropped_dispatches(), 1);
        assert!(format!("{handle:?}").contains("dropped_notifications"));
    }

    #[tokio::test]
    async fn dropping_the_handle_cancels_the_token() {
        let (_events, handle, cancel) = handle_with(4);
        assert!(!cancel.is_cancelled());
        drop(handle);
        assert!(cancel.is_cancelled());
    }
}
