//! Bounded async status stream for the team monitor.
//!
//! [`StatusStream`] connects the monitor's renderer to an async sink: producers push
//! [`MonitorEvent`]s through a [`MonitorEmitter`], and a spawned writer task drains them in FIFO
//! order, renders one line per event with [`TeamMonitor::render_event`] and appends it to a
//! `tokio::io::AsyncWrite` destination. The stream is the only stateful part of the monitor; the
//! event model ([`crate::monitor::event`]) and the renderer ([`crate::monitor::render`]) stay pure.
//!
//! # Queue
//!
//! Emitter and receiver share one bounded `VecDeque` behind a mutex plus two
//! [`tokio::sync::Notify`] handles: `item_ready` wakes a waiting receiver, `space_ready` wakes an
//! emitter blocked by [`BackpressurePolicy::Block`]. The queue closes explicitly
//! ([`StatusStreamHandle::shutdown`], [`StatusStreamHandle::abort`]) or implicitly, when the last
//! [`MonitorEmitter`] clone is dropped — so a writer task never hangs once every producer is gone.
//! Push, pop and close are race-free: the `closed` flag is read **and** written while the `items`
//! lock is held, and the receiver re-checks it under the same lock after an empty pop, so no event
//! enters the buffer after the receiver decided the stream ended and no buffered event is lost.
//!
//! # Deviation: the receiver is not an `mpsc::Receiver`
//!
//! [`StatusStream::channel`] returns a [`MonitorReceiver`] instead of a
//! `tokio::sync::mpsc::Receiver<StampedEvent>`: drop-oldest eviction is impossible from the sender
//! side of an mpsc channel (a sender cannot remove the oldest element of a full channel), while the
//! required semantics are "when full, discard the oldest buffered event and enqueue the new one".
//!
//! # Backpressure and best-effort delivery
//!
//! [`BackpressurePolicy::DropOldest`] (the default) never blocks a producer: a monitoring sink must
//! not be able to stall the team it observes. [`BackpressurePolicy::Block`] is for sinks that must
//! not lose a record and is the only policy that makes [`MonitorEmitter::emit`] wait. Every lossy
//! path is explicit and counted by [`MonitorEmitter::dropped`]: drop-oldest eviction, a refusal
//! from the non-blocking [`MonitorEmitter::try_emit`], the documented eviction race described
//! there, and cancelling [`MonitorEmitter::emit`] before it completes (which drops its event).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncWrite, AsyncWriteExt};
use tokio::sync::Notify;
use tokio::task::JoinHandle;

use crate::monitor::event::{MonitorEvent, StampedEvent};
use crate::monitor::render::TeamMonitor;

/// What an emitter does when the bounded queue is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackpressurePolicy {
    /// Wait for a free slot: [`MonitorEmitter::emit`] suspends, and
    /// [`MonitorEmitter::try_emit`] refuses the event instead of waiting.
    Block,
    /// Evict the oldest buffered event and keep the newest one; never blocks a producer.
    #[default]
    DropOldest,
}

/// What happened to one event handed to an emitter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmitOutcome {
    /// The event entered the buffer.
    Sent,
    /// An older buffered event was evicted to make room (drop-oldest policy).
    DroppedOldest,
    /// The event itself was refused because the buffer was full.
    DroppedNew,
    /// The queue was already closed; nothing was buffered.
    Closed,
}

impl EmitOutcome {
    /// Returns the wire name of this outcome (`sent`, `dropped_oldest`, `dropped_new`, `closed`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Sent => "sent",
            Self::DroppedOldest => "dropped_oldest",
            Self::DroppedNew => "dropped_new",
            Self::Closed => "closed",
        }
    }
}

/// What a writer task reports when it finishes.
///
/// `written` counts the lines fully written to the sink; `dropped` counts the events the queue lost
/// (see [`MonitorEmitter::dropped`]) — a `written + dropped` total below the number of emit attempts
/// means an [`MonitorEmitter::emit`] future was cancelled before it completed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct StreamSummary {
    /// Lines written and flushed to the sink.
    pub written: u64,
    /// Events never rendered because the queue dropped them.
    pub dropped: u64,
}

/// The shared bounded queue behind one emitter/receiver pair.
#[derive(Debug)]
struct StreamQueue {
    /// Maximum number of buffered events; never zero.
    capacity: usize,
    /// How a full buffer is treated.
    policy: BackpressurePolicy,
    /// The buffered events, oldest first.
    items: Mutex<VecDeque<StampedEvent>>,
    /// Set once the stream accepts no more events; guarded by `items` when written.
    closed: AtomicBool,
    /// Live [`MonitorEmitter`] clones; the last drop closes the queue.
    publishers: AtomicUsize,
    /// Events that entered the buffer (sent plus drop-oldest evictions).
    emitted: AtomicU64,
    /// Events that never entered the buffer, plus events evicted by drop-oldest.
    dropped: AtomicU64,
    /// Wakes a waiting [`MonitorReceiver::recv`].
    item_ready: Notify,
    /// Wakes an emitter blocked by [`BackpressurePolicy::Block`].
    space_ready: Notify,
}

/// Result of one attempt to append an event to a [`StreamQueue`].
enum PushOutcome {
    /// The event entered the buffer.
    Pushed,
    /// The buffer was full; the event is handed back unchanged.
    Full(StampedEvent),
    /// The queue was already closed; nothing changed.
    Closed,
}

impl StreamQueue {
    /// Creates a queue, clamping `capacity` to at least one slot.
    fn new(capacity: usize, policy: BackpressurePolicy) -> Self {
        Self {
            capacity: capacity.max(1),
            policy,
            items: Mutex::new(VecDeque::with_capacity(capacity.max(1))),
            closed: AtomicBool::new(false),
            publishers: AtomicUsize::new(0),
            emitted: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            item_ready: Notify::new(),
            space_ready: Notify::new(),
        }
    }

    /// Appends `event` when the buffer has room and the queue is open.
    ///
    /// The `closed` flag is read under the `items` lock, so a close racing this push is ordered:
    /// either the push observes `closed` and refuses, or the close observes the buffered event.
    fn push(&self, event: StampedEvent) -> PushOutcome {
        let mut items = self.items.lock();
        if self.closed.load(Ordering::Acquire) {
            return PushOutcome::Closed;
        }
        if items.len() < self.capacity {
            items.push_back(event);
            PushOutcome::Pushed
        } else {
            PushOutcome::Full(event)
        }
    }

    /// Removes and returns the oldest buffered event, announcing the freed slot.
    fn pop(&self) -> Option<StampedEvent> {
        let popped = {
            let mut items = self.items.lock();
            items.pop_front()
        };
        if popped.is_some() {
            self.space_ready.notify_waiters();
        }
        popped
    }

    /// Removes the oldest buffered event and counts it as dropped.
    ///
    /// Returns whether anything was evicted. The slot is immediately reused by the caller, so no
    /// `space_ready` wake-up is announced.
    fn evict_oldest(&self) -> bool {
        let evicted = {
            let mut items = self.items.lock();
            items.pop_front().is_some()
        };
        if evicted {
            self.dropped.fetch_add(1, Ordering::AcqRel);
        }
        evicted
    }

    /// Closes the queue and wakes every waiter, so no emitter or receiver can hang afterwards.
    ///
    /// The flag is set under the `items` lock: a concurrent [`Self::push`] either completed before
    /// the close (its event is drained) or observes the close and is refused.
    fn close(&self) {
        {
            let _items = self.items.lock();
            self.closed.store(true, Ordering::Release);
        }
        self.item_ready.notify_waiters();
        self.space_ready.notify_waiters();
    }

    /// Whether the queue has been closed.
    fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    /// Number of buffered events.
    fn len(&self) -> usize {
        self.items.lock().len()
    }

    /// Number of events that entered the buffer.
    fn emitted(&self) -> u64 {
        self.emitted.load(Ordering::Acquire)
    }

    /// Number of events the queue lost.
    fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Acquire)
    }
}

/// Producer side of a [`StatusStream`].
///
/// The emitter owns a reference-counted handle to the shared queue: clones are cheap and all of
/// them feed the same buffer, so [`MonitorEmitter::emitted`] and [`MonitorEmitter::dropped`] count
/// the stream, not one clone. Dropping the last clone closes the queue, which is what lets a writer
/// task finish when the host forgets to shut the stream down.
#[derive(Debug)]
pub struct MonitorEmitter {
    /// The shared queue.
    queue: Arc<StreamQueue>,
}

impl MonitorEmitter {
    /// Attaches a new emitter to `queue`, counting it as a live publisher.
    fn attach(queue: Arc<StreamQueue>) -> Self {
        queue.publishers.fetch_add(1, Ordering::AcqRel);
        Self { queue }
    }

    /// Buffers `event` stamped with the current wall-clock time, without waiting for space.
    ///
    /// # Losses under drop-oldest
    ///
    /// Eviction is a two-step operation — remove the oldest event, then append the new one — so a
    /// competing producer can take the freed slot first. The new event is then refused
    /// ([`EmitOutcome::DroppedNew`]) and counted in [`MonitorEmitter::dropped`]; the stream keeps
    /// whatever the competitor pushed. This race is accepted because monitoring is observational.
    #[must_use]
    pub fn try_emit(&self, event: MonitorEvent) -> EmitOutcome {
        self.enqueue(StampedEvent::now(event))
    }

    /// Buffers `event` stamped with an explicit timestamp in milliseconds since the UNIX epoch.
    ///
    /// Behaviour is [`MonitorEmitter::try_emit`] without the clock read; `timestamp` is rendered
    /// as-is, which is what makes golden-line tests and stream replay possible.
    #[must_use]
    pub fn emit_at(&self, timestamp: u64, event: MonitorEvent) -> EmitOutcome {
        self.enqueue(StampedEvent::new(timestamp, event))
    }

    /// Buffers `event`, waiting for a free slot under [`BackpressurePolicy::Block`].
    ///
    /// Under [`BackpressurePolicy::DropOldest`] this is [`MonitorEmitter::try_emit`] with an
    /// `.await`. Under [`BackpressurePolicy::Block`] the call suspends until the receiver drains a
    /// slot, the queue is closed ([`EmitOutcome::Closed`]), or the future is cancelled.
    ///
    /// Cancelling this future before it completes **drops the event it was carrying** and leaves no
    /// trace in the counters; monitoring is best-effort and must never be a reason to keep a task
    /// alive.
    pub async fn emit(&self, event: MonitorEvent) -> EmitOutcome {
        let mut pending = StampedEvent::now(event);
        if self.queue.policy == BackpressurePolicy::DropOldest {
            return self.enqueue(pending);
        }
        loop {
            // Register the waiter before the attempt, so a slot freed between the attempt and the
            // await cannot be missed.
            let notified = self.queue.space_ready.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            match self.queue.push(pending) {
                PushOutcome::Closed => return EmitOutcome::Closed,
                PushOutcome::Pushed => {
                    self.queue.emitted.fetch_add(1, Ordering::AcqRel);
                    self.queue.item_ready.notify_waiters();
                    return EmitOutcome::Sent;
                }
                PushOutcome::Full(event) => {
                    pending = event;
                    notified.await;
                }
            }
        }
    }

    /// Number of events that entered the buffer (sent plus drop-oldest evictions).
    #[must_use]
    pub fn emitted(&self) -> u64 {
        self.queue.emitted()
    }

    /// Number of events the queue lost: refusals plus drop-oldest evictions.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.queue.dropped()
    }

    /// Whether the queue is closed and refuses further events.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.queue.is_closed()
    }

    /// The backpressure policy this stream uses.
    #[must_use]
    pub fn policy(&self) -> BackpressurePolicy {
        self.queue.policy
    }

    /// The effective buffer capacity (clamped to at least one).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.queue.capacity
    }

    /// Buffers `stamped`, applying the backpressure policy when the buffer is full.
    fn enqueue(&self, stamped: StampedEvent) -> EmitOutcome {
        match self.queue.push(stamped) {
            PushOutcome::Closed => EmitOutcome::Closed,
            PushOutcome::Pushed => {
                self.queue.emitted.fetch_add(1, Ordering::AcqRel);
                self.queue.item_ready.notify_waiters();
                EmitOutcome::Sent
            }
            PushOutcome::Full(stamped) => {
                if self.queue.policy == BackpressurePolicy::Block {
                    self.queue.dropped.fetch_add(1, Ordering::AcqRel);
                    return EmitOutcome::DroppedNew;
                }
                self.queue.evict_oldest();
                match self.queue.push(stamped) {
                    PushOutcome::Pushed => {
                        self.queue.emitted.fetch_add(1, Ordering::AcqRel);
                        self.queue.item_ready.notify_waiters();
                        EmitOutcome::DroppedOldest
                    }
                    PushOutcome::Full(_) => {
                        self.queue.dropped.fetch_add(1, Ordering::AcqRel);
                        EmitOutcome::DroppedNew
                    }
                    PushOutcome::Closed => EmitOutcome::Closed,
                }
            }
        }
    }
}

impl Clone for MonitorEmitter {
    /// Attaches another producer to the same queue (the queue counts it as a live publisher).
    fn clone(&self) -> Self {
        Self::attach(Arc::clone(&self.queue))
    }
}

impl Drop for MonitorEmitter {
    /// Closes the queue when the last producer goes away, so a writer task can finish.
    fn drop(&mut self) {
        if self.queue.publishers.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.queue.close();
        }
    }
}

/// Consumer side of a [`StatusStream`].
///
/// Yields [`StampedEvent`]s in the order they entered the buffer. [`MonitorReceiver::recv`] returns
/// [`None`] once the queue is closed **and** drained, so a writer task ends after the last buffered
/// event rather than truncating the tail.
#[derive(Debug)]
pub struct MonitorReceiver {
    /// The shared queue.
    queue: Arc<StreamQueue>,
}

impl MonitorReceiver {
    /// Awaits the oldest buffered event, or [`None`] once the queue is closed and drained.
    ///
    /// The wait is registered before the buffer is inspected, and the `closed` flag is re-read under
    /// the `items` lock after an empty pop, so the return value is exact: either a buffered event,
    /// or a stream that can never produce another one.
    pub async fn recv(&mut self) -> Option<StampedEvent> {
        loop {
            let notified = self.queue.item_ready.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            {
                let mut items = self.queue.items.lock();
                if let Some(item) = items.pop_front() {
                    drop(items);
                    self.queue.space_ready.notify_waiters();
                    return Some(item);
                }
                if self.queue.closed.load(Ordering::Acquire) {
                    return None;
                }
            }
            notified.await;
        }
    }

    /// Takes the oldest buffered event without waiting.
    ///
    /// Returns [`None`] when the buffer is empty, whether or not the stream is still open; a caller
    /// that must distinguish the two checks [`MonitorEmitter::is_closed`] on its emitter.
    pub fn try_recv(&mut self) -> Option<StampedEvent> {
        self.queue.pop()
    }

    /// Number of buffered events.
    #[must_use]
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Whether the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Handle over a spawned writer task.
///
/// Produced by [`StatusStream::spawn`]; either awaited with [`StatusStreamHandle::shutdown`] for a
/// summary, or discarded after [`StatusStreamHandle::abort`] on host teardown.
#[derive(Debug)]
pub struct StatusStreamHandle {
    /// The queue the writer drains; closed before joining so the task can reach its tail.
    queue: Arc<StreamQueue>,
    /// The writer task.
    task: JoinHandle<Result<StreamSummary>>,
}

impl StatusStreamHandle {
    /// Closes the queue and awaits the writer task, returning its summary.
    ///
    /// Events emitted before the call are still in the buffer and are written in FIFO order before
    /// the task exits, so `shutdown()` flushes everything already emitted even while the caller
    /// still holds an emitter. Afterwards the emitter reports
    /// [`MonitorEmitter::is_closed`] and [`MonitorEmitter::try_emit`] returns [`EmitOutcome::Closed`].
    ///
    /// # Errors
    ///
    /// Returns the writer task's error when the sink failed to write or flush, and
    /// [`Error::external`] wrapping the join error when the task panicked or was cancelled.
    pub async fn shutdown(self) -> Result<StreamSummary> {
        self.queue.close();
        match self.task.await {
            Ok(result) => result,
            Err(error) => Err(Error::external(error)),
        }
    }

    /// Closes the queue and cancels the writer task without flushing.
    ///
    /// This is the host teardown path: no summary is produced and buffered events are discarded
    /// rather than written. Emitters observe the stream as closed immediately, because the queue is
    /// closed before the task is aborted.
    pub fn abort(&self) {
        self.queue.close();
        self.task.abort();
    }
}

/// Factory for monitor streams.
///
/// [`StatusStream::channel`] builds the in-memory pair a host drains itself;
/// [`StatusStream::spawn`] wraps it in the writer task that renders lines into an async sink.
pub struct StatusStream;

impl StatusStream {
    /// Creates a queue of at most `capacity` events under `policy`.
    ///
    /// `capacity` is clamped to at least one slot (a zero-length buffer could hold nothing, which
    /// no policy can express), which is why this returns no `Result`. Dropping the emitter closes
    /// the queue; the receiver then drains what is buffered and reports [`None`].
    #[must_use]
    pub fn channel(
        capacity: usize,
        policy: BackpressurePolicy,
    ) -> (MonitorEmitter, MonitorReceiver) {
        let queue = Arc::new(StreamQueue::new(capacity, policy));
        let emitter = MonitorEmitter::attach(Arc::clone(&queue));
        (emitter, MonitorReceiver { queue })
    }

    /// Spawns a writer task that renders every emitted event into `writer`.
    ///
    /// The task drains the bounded queue, renders one line per event with
    /// [`TeamMonitor::render_event`] and appends it followed by a single `'\n'`, counts a line as
    /// written only after both writes succeed, and flushes once when the queue closes and empties.
    /// A write or flush error closes the queue and is reported as [`Error::external`] instead of
    /// panicking on a broken or closed sink.
    ///
    /// The returned handle either flushes with [`StatusStreamHandle::shutdown`] or discards with
    /// [`StatusStreamHandle::abort`]; dropping the last emitter also ends the task.
    ///
    /// # Panics
    ///
    /// Panics when called outside a Tokio runtime, because the writer runs on
    /// `tokio::spawn`.
    pub fn spawn<W>(
        writer: W,
        monitor: TeamMonitor,
        capacity: usize,
        policy: BackpressurePolicy,
    ) -> (MonitorEmitter, StatusStreamHandle)
    where
        W: AsyncWrite + Unpin + Send + 'static,
    {
        let (emitter, receiver) = Self::channel(capacity, policy);
        let queue = Arc::clone(&emitter.queue);
        let task_queue = Arc::clone(&queue);
        let task =
            tokio::spawn(async move { run_writer(writer, &monitor, receiver, &task_queue).await });
        (emitter, StatusStreamHandle { queue, task })
    }
}

/// Drains `receiver` into `writer` until the queue is closed and empty, then flushes once.
async fn run_writer<W>(
    mut writer: W,
    monitor: &TeamMonitor,
    mut receiver: MonitorReceiver,
    queue: &StreamQueue,
) -> Result<StreamSummary>
where
    W: AsyncWrite + Unpin,
{
    let mut written: u64 = 0;
    while let Some(stamped) = receiver.recv().await {
        let line = monitor.render_event(&stamped);
        if let Err(error) = writer.write_all(line.as_bytes()).await {
            queue.close();
            return Err(Error::external(error));
        }
        if let Err(error) = writer.write_all(b"\n").await {
            queue.close();
            return Err(Error::external(error));
        }
        written += 1;
    }
    if let Err(error) = writer.flush().await {
        queue.close();
        return Err(Error::external(error));
    }
    Ok(StreamSummary {
        written,
        dropped: queue.dropped(),
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::io;
    use std::pin::Pin;
    use std::task::{Context, Poll};
    use std::time::Duration;

    use serde_json::json;

    use super::*;
    use crate::monitor::render::{TeamMonitorConfig, TimestampPolicy};
    use crate::schema::{HarnessState, TeamRole};

    /// 12:00:00 UTC in milliseconds.
    const NOON_MS: u64 = 43_200_000; // a fixed stamp keeps rendered lines golden

    /// An in-memory `AsyncWrite` double recording bytes and flush calls.
    #[derive(Clone, Default)]
    struct SharedSink {
        state: Arc<Mutex<SinkState>>,
    }

    /// What a [`SharedSink`] recorded.
    #[derive(Default)]
    struct SinkState {
        bytes: Vec<u8>,
        flushes: usize,
    }

    impl SharedSink {
        /// The recorded bytes as text.
        fn text(&self) -> String {
            String::from_utf8(self.state.lock().bytes.clone()).unwrap()
        }

        /// Number of flushes observed.
        fn flushes(&self) -> usize {
            self.state.lock().flushes
        }
    }

    impl AsyncWrite for SharedSink {
        fn poll_write(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &[u8],
        ) -> Poll<io::Result<usize>> {
            self.state.lock().bytes.extend_from_slice(buf);
            Poll::Ready(Ok(buf.len()))
        }

        fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            self.state.lock().flushes += 1;
            Poll::Ready(Ok(()))
        }
        fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    /// An `AsyncWrite` double whose every write fails.
    struct FailingSink;

    impl AsyncWrite for FailingSink {
        fn poll_write(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            _buf: &[u8],
        ) -> Poll<io::Result<usize>> {
            Poll::Ready(Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "sink closed",
            )))
        }
        fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }
        fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    /// Emits `TeamStopped` at `timestamp`, keeping the assertions short.
    fn stopped(emitter: &MonitorEmitter, timestamp: u64) -> EmitOutcome {
        emitter.emit_at(timestamp, MonitorEvent::TeamStopped)
    }

    /// Spawns a writer into `sink` with the default monitor and a four-event buffer.
    fn spawn_sink(sink: &SharedSink) -> (MonitorEmitter, StatusStreamHandle) {
        StatusStream::spawn(
            sink.clone(),
            TeamMonitor::new(),
            4,
            BackpressurePolicy::DropOldest,
        )
    }

    /// A bounded await so a hung writer task fails its test instead of the suite.
    async fn shutdown<F>(future: F) -> Result<StreamSummary>
    where
        F: std::future::Future<Output = Result<StreamSummary>>,
    {
        tokio::time::timeout(Duration::from_secs(1), future)
            .await
            .unwrap()
    }

    /// Asserts a shutdown result equals the expected summary.
    fn assert_summary(summary: Result<StreamSummary>, written: u64, dropped: u64) {
        let expected = StreamSummary { written, dropped };
        assert_eq!(summary.ok(), Some(expected));
    }

    #[test]
    fn emit_outcome_as_str_covers_every_outcome() {
        let pairs = [
            (EmitOutcome::Sent, "sent"),
            (EmitOutcome::DroppedOldest, "dropped_oldest"),
            (EmitOutcome::DroppedNew, "dropped_new"),
            (EmitOutcome::Closed, "closed"),
        ];
        for (outcome, name) in pairs {
            assert_eq!(outcome.as_str(), name);
            assert_eq!(serde_json::to_value(outcome).unwrap(), json!(name));
        }
    }

    #[test]
    fn backpressure_policy_serde_names_and_default() {
        assert_eq!(
            BackpressurePolicy::default(),
            BackpressurePolicy::DropOldest
        );
        let oldest = serde_json::to_value(BackpressurePolicy::DropOldest).unwrap();
        assert_eq!(oldest, json!("drop_oldest"));
        let block: BackpressurePolicy = serde_json::from_value(json!("block")).unwrap();
        assert_eq!(block, BackpressurePolicy::Block);
        assert_eq!(
            serde_json::to_value(block).unwrap(),
            json!("block"),
            "the serde names are the wire contract"
        );
    }

    #[test]
    fn stream_summary_serde_round_trip_and_default() {
        let empty = StreamSummary::default();
        assert_eq!((empty.written, empty.dropped), (0, 0));
        let summary = StreamSummary {
            written: 3,
            dropped: 1,
        };
        let value = serde_json::to_value(summary).unwrap();
        assert_eq!(value, json!({"written": 3, "dropped": 1}));
        assert_eq!(
            serde_json::from_value::<StreamSummary>(value).unwrap(),
            summary
        );
        let sparse: StreamSummary = serde_json::from_value(json!({"future": true})).unwrap();
        assert_eq!(sparse, StreamSummary::default());
    }

    #[test]
    fn drop_oldest_keeps_the_newest_event() {
        let (emitter, mut receiver) = StatusStream::channel(1, BackpressurePolicy::DropOldest);
        assert_eq!(stopped(&emitter, 1), EmitOutcome::Sent);
        assert_eq!(stopped(&emitter, 2), EmitOutcome::DroppedOldest);
        assert_eq!(stopped(&emitter, 3), EmitOutcome::DroppedOldest);
        assert_eq!(emitter.dropped(), 2);
        assert_eq!(emitter.emitted(), 3);
        assert_eq!(emitter.policy(), BackpressurePolicy::DropOldest);
        assert_eq!(receiver.len(), 1);
        assert_eq!(receiver.try_recv().unwrap().timestamp, 3);
        assert!(receiver.try_recv().is_none());
    }

    #[test]
    fn capacity_zero_is_clamped_to_one() {
        let (emitter, mut receiver) = StatusStream::channel(0, BackpressurePolicy::DropOldest);
        assert_eq!(emitter.capacity(), 1);
        assert_eq!(stopped(&emitter, 1), EmitOutcome::Sent);
        assert_eq!(stopped(&emitter, 2), EmitOutcome::DroppedOldest);
        assert_eq!(receiver.try_recv().unwrap().timestamp, 2);
        assert!(receiver.try_recv().is_none());
    }

    #[tokio::test]
    async fn block_policy_refuses_then_waits_for_a_free_slot() {
        let (emitter, mut receiver) = StatusStream::channel(1, BackpressurePolicy::Block);
        assert_eq!(emitter.policy(), BackpressurePolicy::Block);
        assert_eq!(stopped(&emitter, 1), EmitOutcome::Sent);
        assert_eq!(stopped(&emitter, 2), EmitOutcome::DroppedNew);
        assert_eq!(emitter.dropped(), 1);

        let running = MonitorEvent::TeamStatus {
            state: HarnessState::Running,
        };
        let next = emitter.emit(running.clone());
        tokio::pin!(next);
        assert!(
            futures::poll!(next.as_mut()).is_pending(),
            "a full block-policy queue must suspend the emitter"
        );
        assert_eq!(
            receiver.try_recv().unwrap().event,
            MonitorEvent::TeamStopped
        );
        let outcome = tokio::time::timeout(Duration::from_secs(1), next.as_mut())
            .await
            .unwrap();
        assert_eq!(outcome, EmitOutcome::Sent);
        assert_eq!(
            receiver.try_recv().unwrap().event,
            running,
            "the waiting event lands behind the one it queued behind"
        );
        assert!(receiver.try_recv().is_none());
        assert_eq!(emitter.emitted(), 2);
    }

    #[tokio::test]
    async fn emit_at_keeps_explicit_timestamps_and_fifo_order() {
        let sink = SharedSink::default();
        let (emitter, handle) = spawn_sink(&sink);
        assert_eq!(stopped(&emitter, NOON_MS), EmitOutcome::Sent);
        let spawned = MonitorEvent::MemberSpawned {
            member_name: "dev-1".to_owned(),
            role: TeamRole::Teammate,
        };
        assert_eq!(emitter.emit_at(NOON_MS + 1_000, spawned), EmitOutcome::Sent);
        let summary = shutdown(handle.shutdown()).await;
        assert_summary(summary, 2, 0);
        assert_eq!(
            sink.text(),
            "[12:00:00] team stopped\n[12:00:01] member dev-1 spawned role=teammate\n"
        );
    }

    #[tokio::test]
    async fn shutdown_closes_the_stream_for_later_emits() {
        let sink = SharedSink::default();
        let (emitter, handle) = spawn_sink(&sink);
        assert!(!emitter.is_closed());
        let running = MonitorEvent::TeamStatus {
            state: HarnessState::Running,
        };
        assert_eq!(emitter.emit_at(0, running), EmitOutcome::Sent);
        let summary = shutdown(handle.shutdown()).await.unwrap();
        assert_eq!(summary.written, 1);
        assert!(emitter.is_closed());
        assert_eq!(
            emitter.try_emit(MonitorEvent::TeamStopped),
            EmitOutcome::Closed
        );
        assert_eq!(sink.text(), "[00:00:00] team running\n");
    }

    #[tokio::test]
    async fn dropping_the_last_emitter_closes_the_queue() {
        let sink = SharedSink::default();
        let (emitter, handle) = spawn_sink(&sink);
        let clone = emitter.clone();
        drop(emitter);
        assert!(
            !clone.is_closed(),
            "one live publisher keeps the queue open"
        );
        assert_eq!(stopped(&clone, 0), EmitOutcome::Sent);
        drop(clone);
        let summary = shutdown(handle.shutdown()).await;
        assert_summary(summary, 1, 0);
        assert_eq!(sink.text(), "[00:00:00] team stopped\n");
    }

    #[test]
    fn emitter_clone_shares_one_queue() {
        let (emitter, mut receiver) = StatusStream::channel(4, BackpressurePolicy::DropOldest);
        let clone = emitter.clone();
        assert_eq!(stopped(&emitter, 1), EmitOutcome::Sent);
        assert_eq!(stopped(&clone, 2), EmitOutcome::Sent);
        assert_eq!(emitter.emitted(), 2);
        assert_eq!(clone.emitted(), 2);
        assert_eq!(emitter.capacity(), 4);
        assert_eq!(clone.capacity(), 4);
        assert_eq!(receiver.len(), 2);
        drop(clone);
        assert!(!emitter.is_closed());
        assert_eq!(receiver.try_recv().unwrap().timestamp, 1);
        assert_eq!(receiver.try_recv().unwrap().timestamp, 2);
    }

    #[tokio::test]
    async fn spawn_writes_lines_to_a_temp_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("status.log");
        let file = tokio::fs::File::create(&path).await.unwrap();
        let (emitter, handle) =
            StatusStream::spawn(file, TeamMonitor::new(), 4, BackpressurePolicy::DropOldest);
        assert_eq!(stopped(&emitter, NOON_MS), EmitOutcome::Sent);
        let quit = MonitorEvent::MemberStopped {
            member_name: "dev-1".to_owned(),
            reason: None,
        };
        assert_eq!(emitter.emit_at(NOON_MS + 1_000, quit), EmitOutcome::Sent);
        let summary = shutdown(handle.shutdown()).await;
        assert_summary(summary, 2, 0);
        let text = tokio::fs::read_to_string(&path).await.unwrap();
        assert_eq!(
            text,
            "[12:00:00] team stopped\n[12:00:01] member dev-1 stopped\n"
        );
        assert!(text.ends_with('\n'));
    }

    #[tokio::test]
    async fn emits_after_shutdown_leave_the_file_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("status.log");
        let file = tokio::fs::File::create(&path).await.unwrap();
        let (emitter, handle) =
            StatusStream::spawn(file, TeamMonitor::new(), 4, BackpressurePolicy::DropOldest);
        let idle = MonitorEvent::TeamStatus {
            state: HarnessState::Idle,
        };
        assert_eq!(emitter.emit_at(NOON_MS, idle), EmitOutcome::Sent);
        let summary = shutdown(handle.shutdown()).await.unwrap();
        assert_eq!(summary.written, 1);
        let before = tokio::fs::read_to_string(&path).await.unwrap();
        assert_eq!(
            emitter.try_emit(MonitorEvent::TeamStopped),
            EmitOutcome::Closed
        );
        assert!(emitter.is_closed());
        assert_eq!(tokio::fs::read_to_string(&path).await.unwrap(), before);
    }

    #[tokio::test]
    async fn in_memory_sink_receives_bytes_and_one_flush() {
        let sink = SharedSink::default();
        let (emitter, handle) = spawn_sink(&sink);
        assert_eq!(stopped(&emitter, NOON_MS), EmitOutcome::Sent);
        let created = MonitorEvent::TaskCreated {
            task_id: "t-1".to_owned(),
            title: "write docs".to_owned(),
        };
        assert_eq!(emitter.emit_at(NOON_MS, created), EmitOutcome::Sent);
        let summary = shutdown(handle.shutdown()).await.unwrap();
        assert_eq!(summary.written, 2);
        assert_eq!(
            sink.text(),
            "[12:00:00] team stopped\n[12:00:00] task t-1 created title=write docs\n"
        );
        assert_eq!(sink.flushes(), 1, "the writer flushes exactly once");
    }

    #[tokio::test]
    async fn write_errors_surface_as_external_errors() {
        let (emitter, handle) = StatusStream::spawn(
            FailingSink,
            TeamMonitor::new(),
            4,
            BackpressurePolicy::DropOldest,
        );
        assert_eq!(stopped(&emitter, 0), EmitOutcome::Sent);
        let error = shutdown(handle.shutdown())
            .await
            .expect_err("a failing sink must surface an error");
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::External);
        assert!(error.message().contains("sink closed"), "{error}");
        assert!(emitter.is_closed(), "a failed writer closes the queue");
    }

    #[test]
    fn receiver_len_and_try_recv_are_fifo() {
        let (emitter, mut receiver) = StatusStream::channel(4, BackpressurePolicy::DropOldest);
        assert!(receiver.is_empty());
        assert_eq!(receiver.len(), 0);
        assert!(receiver.try_recv().is_none());
        for timestamp in [1, 2, 3] {
            assert_eq!(stopped(&emitter, timestamp), EmitOutcome::Sent);
        }
        assert_eq!(receiver.len(), 3);
        assert!(!receiver.is_empty());
        let stamps: Vec<u64> = (0..3)
            .map(|_| receiver.try_recv().unwrap().timestamp)
            .collect();
        assert_eq!(stamps, vec![1, 2, 3]);
        assert!(receiver.try_recv().is_none());
        assert!(receiver.is_empty());
    }

    #[tokio::test]
    async fn abort_closes_the_queue_without_flushing() {
        let sink = SharedSink::default();
        let (emitter, handle) = spawn_sink(&sink);
        assert_eq!(stopped(&emitter, 0), EmitOutcome::Sent);
        handle.abort();
        tokio::task::yield_now().await;
        assert!(emitter.is_closed());
        assert_eq!(
            emitter.try_emit(MonitorEvent::TeamStopped),
            EmitOutcome::Closed
        );
        handle.abort();
        assert_eq!(sink.flushes(), 0, "abort is not a flush path");
        drop(handle);
    }

    #[tokio::test]
    async fn hidden_timestamps_render_without_prefix() {
        let sink = SharedSink::default();
        let monitor = TeamMonitor::with_config(TeamMonitorConfig {
            timestamp: TimestampPolicy::Hidden,
            ..TeamMonitorConfig::default()
        });
        let (emitter, handle) =
            StatusStream::spawn(sink.clone(), monitor, 2, BackpressurePolicy::Block);
        assert_eq!(stopped(&emitter, NOON_MS), EmitOutcome::Sent);
        let summary = shutdown(handle.shutdown()).await.unwrap();
        assert_eq!(summary.written, 1);
        assert_eq!(sink.text(), "team stopped\n");
    }
}
