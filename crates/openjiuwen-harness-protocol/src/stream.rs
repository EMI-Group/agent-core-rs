//! Closable single-consumer cursor for external harness observations.
//!
//! Mirrors the Python `openjiuwen.harness_protocol.stream` module (v0.1.20): one ordered,
//! single-consumer view over a harness's observation channel plus an explicit
//! release-the-lease operation. The Rust layer adds the ownership-safe adapters
//! [`EventCursorStream`] and [`event_cursor_stream`] that turn a cursor into a
//! [`HarnessEventStream`].
//!
//! # Contract inherited from Python
//!
//! - A *turn view* (`HarnessProtocol::turn_events`) is finite: it yields the turn start and the
//!   following ordered events through the matching terminal event. `TurnEventKind::Paused` and
//!   `TurnEventKind::Resumed` do **not** end the view; `TurnEventKind::Finished`,
//!   `TurnEventKind::Aborted`, and `TurnEventKind::Failed` (that is,
//!   [`TERMINAL_TURN_EVENT_KINDS`](crate::events::TERMINAL_TURN_EVENT_KINDS)) do.
//! - A *cycle view* (`HarnessProtocol::events`) is cycle-long: it stays open across turn
//!   boundaries and ends only when the current `start`/`stop` cycle closes.
//! - Both views read the same logical single-consumer channel. A provider must raise
//!   [`HarnessError::State`](crate::HarnessError::State) when a second observation cursor is
//!   already active instead of racing it. That rule is an **implementation obligation**: this
//!   module cannot enforce it, because it neither owns the channel nor the per-provider cursor
//!   registry.
//! - [`HarnessEventCursor::close`] is idempotent and never rewinds or discards events that were
//!   already delivered; a later `next_event` returns `None`.
//! - Dropping a cursor without calling `close` is the *abandon* path: `Drop` releases the lease.
//!   Rust has no `aclose()`-less garbage-collection hook, so this is expressed as ordinary drop
//!   semantics and documented on the trait and on [`EventCursorStream`].
use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};

use async_trait::async_trait;
use futures::stream::{BoxStream, Stream};

use crate::errors::HarnessResult;
use crate::events::HarnessEvent;

/// Async observation cursor whose consumer lease can be released early.
///
/// The cursor is the Rust translation of the Python `HarnessEventCursor` protocol: an ordered,
/// single-consumer view over a harness's observation channel. Implementations are providers'
/// responsibility; [`VecEventCursor`] is the reference implementation used by fakes and tests.
///
/// # Contract
///
/// - Events arrive in `sequence` order, one per [`next_event`](Self::next_event) call.
/// - A turn view ends after the matching terminal event; `PAUSED`/`RESUMED` do not end it. A cycle
///   view stays open across turn boundaries and ends only when the start/stop cycle closes.
/// - Only one observation cursor may be active per harness at a time. An implementation must raise
///   [`HarnessError::State`](crate::HarnessError::State) for a second concurrently active cursor rather than racing it. This is
///   an implementation obligation and is not enforced here.
/// - [`close`](Self::close) is idempotent; it releases the consumer lease without rewinding, so
///   events already delivered are never repeated or discarded retroactively.
/// - Dropping the cursor without calling [`close`](Self::close) is the abandon path: `Drop`
///   releases the lease. Prefer an explicit `close().await` when the provider failure must be
///   observed.
#[async_trait]
pub trait HarnessEventCursor: Send {
    /// Returns the next ordered event, or `None` once the view is exhausted or closed.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::State`](crate::HarnessError::State) when the view can no longer be observed (for example the
    /// cycle was stopped while the caller was still awaiting an event), or the provider's own
    /// transport error. An error ends the view: callers must treat the cursor as finished.
    async fn next_event(&mut self) -> HarnessResult<Option<HarnessEvent>>;

    /// Idempotently releases the observation consumer lease.
    ///
    /// Closing never rewinds: already delivered events are not replayed, and a subsequent
    /// [`next_event`](Self::next_event) returns `None`. Calling `close` again is always safe.
    ///
    /// # Errors
    ///
    /// Returns the provider-side failure while releasing the lease. The cursor must still be
    /// treated as closed; retrying is safe because the operation is idempotent.
    async fn close(&mut self) -> HarnessResult<()>;
}

/// Boxed stream of ordered harness observations, each item fallible.
///
/// This is the concrete shape returned by [`event_cursor_stream`] and by every stream-returning
/// helper in the harness layer. Items are yielded in cursor order; the stream ends after the
/// cursor is exhausted (and closed) or after its final error item.
pub type HarnessEventStream = BoxStream<'static, HarnessResult<HarnessEvent>>;

/// Ownership-safe adapter that exposes a [`HarnessEventCursor`] as a [`Stream`].
///
/// The adapter stores a boxed, `'static` stream built by [`EventCursorStream::new`]; it owns the
/// cursor through that stream, so no `unsafe` code and no self-referential struct is required.
///
/// # Behaviour
///
/// - Each [`Stream::poll_next`] yields the next cursor event in order.
/// - When the cursor returns `Ok(None)` the adapter calls [`HarnessEventCursor::close`] and ends
///   the stream. If `close` fails, that error is the stream's **final item** and the stream ends
///   on the following poll; the failure is never swallowed.
/// - When the cursor returns `Err(e)` the adapter yields `Err(e)` as an item, closes the cursor on
///   the following poll, and then ends. A failing `close` there is surfaced as one additional final
///   error item, so a lease-release failure is still observable.
/// - Dropping the adapter drops the boxed stream together with the cursor, which is the abandon
///   path: the lease is released by `Drop` even when [`HarnessEventCursor::close`] was never
///   awaited.
/// - The stream is fused: polling it after it returned `None` keeps returning `None`.
///
/// Consumers that need `Unpin` ergonomics can use [`event_cursor_stream`], which returns the same
/// machine as a pinned [`HarnessEventStream`].
pub struct EventCursorStream {
    inner: HarnessEventStream,
    ended: bool,
}

impl EventCursorStream {
    /// Wrap `cursor` into a stream adapter; the adapter takes ownership of the cursor.
    ///
    /// Nothing is polled until the returned stream is polled for the first time, so building an
    /// adapter cannot fail.
    #[must_use]
    pub fn new(cursor: Box<dyn HarnessEventCursor>) -> Self {
        Self {
            inner: Box::pin(futures::stream::unfold(
                CursorState::Active(cursor),
                advance,
            )),
            ended: false,
        }
    }
}

impl Stream for EventCursorStream {
    type Item = HarnessResult<HarnessEvent>;

    /// Polls the next event; once the stream has ended, every later poll returns
    /// [`Poll::Ready`]`(None)` instead of polling the exhausted cursor machine again.
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.ended {
            return Poll::Ready(None);
        }
        let polled = this.inner.as_mut().poll_next(cx);
        if matches!(polled, Poll::Ready(None)) {
            this.ended = true;
        }
        polled
    }
}

/// Adapts `cursor` into a [`HarnessEventStream`].
///
/// The returned stream preserves cursor order, closes the cursor when it is exhausted, and emits
/// the cursor's error as its last item (see [`EventCursorStream`] for the exact state machine).
/// Dropping the stream drops the cursor and therefore releases the lease.
pub fn event_cursor_stream(cursor: Box<dyn HarnessEventCursor>) -> HarnessEventStream {
    Box::pin(EventCursorStream::new(cursor))
}

/// State machine behind [`EventCursorStream`]; the cursor is owned by exactly one arm at a time.
enum CursorState {
    /// Polling events from the cursor.
    Active(Box<dyn HarnessEventCursor>),
    /// An error item was already yielded; the cursor still has to be closed.
    Closing(Box<dyn HarnessEventCursor>),
    /// Nothing left to do; the stream ends on the next poll.
    Done,
}

/// Produces the next stream item and the state to poll afterwards.
async fn advance(state: CursorState) -> Option<(HarnessResult<HarnessEvent>, CursorState)> {
    match state {
        CursorState::Active(mut cursor) => match cursor.next_event().await {
            Ok(Some(event)) => Some((Ok(event), CursorState::Active(cursor))),
            Ok(None) => close_after_exhaustion(cursor).await,
            Err(error) => Some((Err(error), CursorState::Closing(cursor))),
        },
        CursorState::Closing(mut cursor) => match cursor.close().await {
            Ok(()) => None,
            Err(error) => Some((Err(error), CursorState::Done)),
        },
        CursorState::Done => None,
    }
}

/// Closes an exhausted cursor, surfacing a release failure as the stream's final item.
async fn close_after_exhaustion(
    mut cursor: Box<dyn HarnessEventCursor>,
) -> Option<(HarnessResult<HarnessEvent>, CursorState)> {
    match cursor.close().await {
        Ok(()) => None,
        Err(error) => Some((Err(error), CursorState::Done)),
    }
}

/// Reference [`HarnessEventCursor`] over an in-memory event list.
///
/// This is the implementation fakes and tests use: it hands out the buffered events in order,
/// reports how many are left, and models the lease with a `closed` flag. Closing is idempotent and
/// does not rewind — the remaining events stay buffered (and [`remaining`](Self::remaining) keeps
/// reporting them) but are no longer delivered.
#[derive(Debug)]
pub struct VecEventCursor {
    events: VecDeque<HarnessEvent>,
    closed: bool,
}

impl VecEventCursor {
    /// Builds a cursor that will yield `events` in the given order.
    #[must_use]
    pub fn new(events: Vec<HarnessEvent>) -> Self {
        Self {
            events: events.into(),
            closed: false,
        }
    }

    /// Returns how many events have not been delivered yet.
    ///
    /// Closing does not rewind, so the count is unchanged by [`HarnessEventCursor::close`].
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.events.len()
    }

    /// Returns whether the cursor lease was released.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.closed
    }
}

#[async_trait]
impl HarnessEventCursor for VecEventCursor {
    async fn next_event(&mut self) -> HarnessResult<Option<HarnessEvent>> {
        if self.closed {
            return Ok(None);
        }
        Ok(self.events.pop_front())
    }

    async fn close(&mut self) -> HarnessResult<()> {
        self.closed = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    use async_trait::async_trait;
    use futures::executor::block_on;
    use futures::StreamExt;

    use super::{event_cursor_stream, EventCursorStream, HarnessEventCursor, VecEventCursor};
    use crate::errors::{HarnessError, HarnessErrorKind, HarnessResult};
    use crate::events::{HarnessEvent, StateChangedEvent};
    use crate::HarnessState;

    /// Builds a valid, ordered observation with the given `sequence`.
    fn event(sequence: u64) -> HarnessEvent {
        HarnessEvent::new(
            sequence,
            1.5,
            StateChangedEvent {
                old: HarnessState::Idle,
                new: HarnessState::Running,
            },
            "host-session",
            "agent-a",
        )
    }

    /// Cursor that records its lease release, so the adapter's close-on-exhaustion is observable.
    struct RecordingCursor {
        events: VecDeque<HarnessEvent>,
        closed: Arc<AtomicBool>,
    }

    #[async_trait]
    impl HarnessEventCursor for RecordingCursor {
        async fn next_event(&mut self) -> HarnessResult<Option<HarnessEvent>> {
            Ok(self.events.pop_front())
        }

        async fn close(&mut self) -> HarnessResult<()> {
            self.closed.store(true, Ordering::SeqCst);
            Ok(())
        }
    }

    /// Cursor whose only observation is an error.
    struct FailingCursor {
        closed: Arc<AtomicBool>,
    }

    #[async_trait]
    impl HarnessEventCursor for FailingCursor {
        async fn next_event(&mut self) -> HarnessResult<Option<HarnessEvent>> {
            Err(HarnessError::state("cursor is no longer observable"))
        }

        async fn close(&mut self) -> HarnessResult<()> {
            self.closed.store(true, Ordering::SeqCst);
            Ok(())
        }
    }

    /// Cursor that is exhausted immediately but fails to release its lease.
    struct CloseFailingCursor;

    #[async_trait]
    impl HarnessEventCursor for CloseFailingCursor {
        async fn next_event(&mut self) -> HarnessResult<Option<HarnessEvent>> {
            Ok(None)
        }

        async fn close(&mut self) -> HarnessResult<()> {
            Err(HarnessError::state("lease release failed"))
        }
    }

    #[test]
    fn vec_event_cursor_yields_events_in_order() {
        let mut cursor = VecEventCursor::new(vec![event(1), event(2), event(3)]);
        assert_eq!(cursor.remaining(), 3);
        assert!(!cursor.is_closed());

        let first = block_on(cursor.next_event())
            .expect("first event")
            .expect("event");
        assert_eq!(first.sequence, 1);
        assert_eq!(cursor.remaining(), 2);

        let second = block_on(cursor.next_event())
            .expect("second event")
            .expect("event");
        assert_eq!(second.sequence, 2);
        assert_eq!(cursor.remaining(), 1);

        let third = block_on(cursor.next_event())
            .expect("third event")
            .expect("event");
        assert_eq!(third.sequence, 3);
        assert_eq!(cursor.remaining(), 0);
        assert!(block_on(cursor.next_event()).expect("exhausted").is_none());
    }

    #[test]
    fn vec_event_cursor_close_is_idempotent_and_does_not_rewind() {
        let mut cursor = VecEventCursor::new(vec![event(1), event(2)]);
        assert_eq!(
            block_on(cursor.next_event()).expect("event"),
            Some(event(1))
        );

        block_on(cursor.close()).expect("first close");
        assert!(cursor.is_closed());
        assert_eq!(cursor.remaining(), 1);

        block_on(cursor.close()).expect("second close");
        assert!(cursor.is_closed());
        assert_eq!(cursor.remaining(), 1);

        assert!(block_on(cursor.next_event()).expect("closed").is_none());
        assert_eq!(cursor.remaining(), 1);
    }

    #[test]
    fn empty_vec_event_cursor_yields_none() {
        let mut cursor = VecEventCursor::new(Vec::new());
        assert_eq!(cursor.remaining(), 0);
        assert!(block_on(cursor.next_event()).expect("empty").is_none());
        block_on(cursor.close()).expect("close");
        assert!(cursor.is_closed());
    }

    #[test]
    fn event_cursor_stream_preserves_order_and_closes_on_exhaustion() {
        let closed = Arc::new(AtomicBool::new(false));
        let cursor = RecordingCursor {
            events: VecDeque::from(vec![event(2), event(3), event(4)]),
            closed: Arc::clone(&closed),
        };
        assert!(!closed.load(Ordering::SeqCst));

        let stream = event_cursor_stream(Box::new(cursor));
        let sequences = block_on(
            stream
                .map(|item| item.expect("event").sequence)
                .collect::<Vec<_>>(),
        );

        assert_eq!(sequences, vec![2, 3, 4]);
        assert!(
            closed.load(Ordering::SeqCst),
            "exhaustion must close the cursor"
        );
    }

    #[test]
    fn event_cursor_stream_over_vec_cursor_yields_then_ends() {
        let stream = EventCursorStream::new(Box::new(VecEventCursor::new(vec![event(7)])));
        let items = block_on(stream.collect::<Vec<_>>());

        assert_eq!(items.len(), 1);
        assert_eq!(items[0].as_ref().expect("event").sequence, 7);
    }

    #[test]
    fn event_cursor_stream_surfaces_a_cursor_error_as_the_last_item() {
        let closed = Arc::new(AtomicBool::new(false));
        let stream = event_cursor_stream(Box::new(FailingCursor {
            closed: Arc::clone(&closed),
        }));
        let items = block_on(stream.collect::<Vec<_>>());

        assert_eq!(items.len(), 1, "the error is the only item");
        let error = items[0].as_ref().expect_err("state error");
        assert_eq!(error.kind(), HarnessErrorKind::State);
        assert_eq!(error.message(), "cursor is no longer observable");
        assert!(
            closed.load(Ordering::SeqCst),
            "the cursor is closed after an error item"
        );
    }

    #[test]
    fn stream_error_is_followed_by_none() {
        let mut stream = event_cursor_stream(Box::new(FailingCursor {
            closed: Arc::new(AtomicBool::new(false)),
        }));

        let first = block_on(stream.next());
        assert!(matches!(first, Some(Err(_))), "first item is the error");
        assert!(
            block_on(stream.next()).is_none(),
            "the stream ends after the error"
        );
        assert!(
            block_on(stream.next()).is_none(),
            "an ended stream stays ended"
        );
    }

    #[test]
    fn close_error_after_exhaustion_is_the_final_item() {
        let mut stream = event_cursor_stream(Box::new(CloseFailingCursor));

        let first = block_on(stream.next());
        let error = first.expect("close error item").expect_err("state error");
        assert_eq!(error.kind(), HarnessErrorKind::State);
        assert_eq!(error.message(), "lease release failed");
        assert!(block_on(stream.next()).is_none());
    }
}
