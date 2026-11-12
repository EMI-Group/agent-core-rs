//! Single-consumer observation channel and cursors for the native harness.
//!
//! The native harness publishes its [`HarnessEvent`]s through one bounded
//! [`EventChannel`]: a `tokio::sync::mpsc` queue whose capacity and overflow
//! behavior come from the [`EventBufferConfig`] the harness advertises.
//! Consumers observe the queue through exactly one
//! [`HarnessEventCursor`] at a time — either the cycle-long view
//! (`events()`, which survives turn boundaries) or a turn view
//! (`turn_events()`, which is finite and ends with the matching terminal
//! `TurnLifecycle` event).
//!
//! The channel is deliberately destructive: a cursor consumes events as it
//! reads them, so a later cursor continues where the previous one stopped and
//! never replays. Reading is a plain `recv` on the shared receiver — no
//! per-cursor buffering and no rewinding — which keeps the single-consumer
//! lease meaningful and the memory footprint bounded by
//! [`EventBufferConfig::capacity`].
//!
//! Lifecycle rules mirrored from `openjiuwen.harness_protocol.stream`:
//!
//! - only one cursor may hold the receiver; a second request is rejected with
//!   [`HarnessError::State`] instead of racing the active consumer;
//! - [`HarnessEventCursor::close`] is idempotent, releases the lease, and
//!   returns the receiver to the channel so a later cursor can continue;
//! - dropping a cursor without closing is the abandon path and releases the
//!   lease too;
//! - [`EventChannel::close`] marks the cycle closed and drops the sender, so a
//!   cursor drains what is still buffered and then ends with `Ok(None)`.

use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_harness_protocol::{
    event_retention, EventBufferConfig, EventOverflowPolicy, EventRetention, HarnessError,
    HarnessEvent, HarnessEventCursor, HarnessEventPayload, HarnessResult, TurnEventKind,
};
use parking_lot::Mutex;
use tokio::sync::mpsc::{self, error::TrySendError, Receiver, Sender};

/// Largest channel capacity the channel will allocate.
///
/// `tokio` panics when a semaphore is created with more permits than it
/// supports, so a wire-supplied capacity beyond this bound saturates here
/// instead of aborting the process. It is far above any realistic observation
/// buffer (the protocol default is `1024`).
const MAX_CHANNEL_CAPACITY: usize = 1 << 20;

/// `true` while a consumer holds the shared receiver.
#[derive(Debug)]
struct ChannelState {
    /// Producer side; `None` once the cycle was closed (or before it was opened).
    sender: Option<Sender<HarnessEvent>>,
    /// Consumer side; `None` while a cursor owns it.
    receiver: Option<Receiver<HarnessEvent>>,
    /// Whether a cursor currently holds the receiver lease.
    leased: bool,
    /// Whether [`EventChannel::close`] already ran.
    closed: bool,
}

/// Bounded observation queue shared by the harness producer and its one consumer.
///
/// The channel applies the advertised [`EventOverflowPolicy`] when the queue is
/// full: required events block (await room), coalescible/best-effort events may
/// be dropped and are logged at `debug`. Sending into a closed cycle is a
/// no-op, so a shutdown race never fails a turn.
#[derive(Debug)]
pub(crate) struct EventChannel {
    /// Sender, receiver, and lease flags; never locked across an `await`.
    state: Mutex<ChannelState>,
    /// Effective queue capacity in elements (at least `1`).
    capacity: usize,
    /// Overflow behavior advertised by the harness.
    policy: EventOverflowPolicy,
}

impl EventChannel {
    /// Opens a fresh channel for `config`.
    ///
    /// `config.capacity` is clamped to at least `1` and to `usize` (saturating
    /// at [`MAX_CHANNEL_CAPACITY`]); `config.overflow` is stored verbatim and
    /// applied by [`emit`](Self::emit).
    pub(crate) fn open(config: EventBufferConfig) -> Arc<Self> {
        let capacity = usize::try_from(config.capacity)
            .unwrap_or(usize::MAX)
            .clamp(1, MAX_CHANNEL_CAPACITY);
        let (sender, receiver) = mpsc::channel(capacity);
        Arc::new(Self {
            state: Mutex::new(ChannelState {
                sender: Some(sender),
                receiver: Some(receiver),
                leased: false,
                closed: false,
            }),
            capacity,
            policy: config.overflow,
        })
    }

    /// Returns the effective buffer contract of this channel.
    ///
    /// `capacity` reflects the clamped value used to build the queue.
    pub(crate) fn config(&self) -> EventBufferConfig {
        EventBufferConfig {
            capacity: u64::try_from(self.capacity).unwrap_or(u64::MAX),
            overflow: self.policy,
        }
    }

    /// Publishes `event`, applying the advertised overflow policy.
    ///
    /// Never fails: a closed channel drops the event (the cycle is over), and a
    /// full channel either awaits room (required retention) or drops the event
    /// (coalescible/best-effort retention under a coalescing policy).
    pub(crate) async fn emit(&self, event: HarnessEvent) {
        let sender = {
            let state = self.state.lock();
            if state.closed {
                None
            } else {
                state.sender.clone()
            }
        };
        let Some(sender) = sender else {
            return;
        };
        match sender.try_send(event) {
            Ok(()) => {}
            Err(TrySendError::Closed(_)) => {
                tracing::debug!("native harness event channel is closed; event dropped");
            }
            Err(TrySendError::Full(event)) => self.handle_full(sender, event).await,
        }
    }

    /// Applies the overflow policy to `event` after `try_send` reported a full queue.
    async fn handle_full(&self, sender: Sender<HarnessEvent>, event: HarnessEvent) {
        let retention = event_retention(&event.event);
        let drop_when_full = match self.policy {
            EventOverflowPolicy::Block => false,
            EventOverflowPolicy::CoalesceOrBlock => retention != EventRetention::Required,
            EventOverflowPolicy::DropBestEffortOrBlock => retention == EventRetention::BestEffort,
        };
        if drop_when_full {
            tracing::debug!(
                sequence = event.sequence,
                retention = retention.as_str(),
                "dropping native harness event because the observation buffer is full"
            );
            return;
        }
        if sender.send(event).await.is_err() {
            tracing::debug!("native harness event channel closed while waiting for room");
        }
    }

    /// Opens the cycle-long observation view.
    ///
    /// The view stays open across turn boundaries and ends when the cycle
    /// closes. When no cycle is active, the cycle is already closed, or another
    /// cursor holds the lease, the returned cursor surfaces the matching
    /// [`HarnessError::State`] on its first [`HarnessEventCursor::next_event`].
    pub(crate) fn events(self: &Arc<Self>) -> Box<dyn HarnessEventCursor> {
        self.cursor(View::Cycle)
    }

    /// Opens a turn-scoped observation view.
    ///
    /// Events are discarded until the next `TurnLifecycle::Started`. When
    /// `turn_id` is `Some`, that start must belong to the requested turn or the
    /// cursor fails with [`HarnessError::State`]; the view then yields every
    /// event of the turn, including its terminal one, and ends.
    pub(crate) fn turn_events(
        self: &Arc<Self>,
        turn_id: Option<&str>,
    ) -> Box<dyn HarnessEventCursor> {
        self.cursor(View::Turn {
            requested: turn_id.map(str::to_owned),
        })
    }

    /// Builds a cursor for `view`, taking the single-consumer lease.
    fn cursor(self: &Arc<Self>, view: View) -> Box<dyn HarnessEventCursor> {
        let mut state = self.state.lock();
        if state.sender.is_none() || state.closed {
            return Box::new(ErrorCursor::new(HarnessError::state(
                "native harness has no active event cycle",
            )));
        }
        if state.leased {
            return Box::new(ErrorCursor::new(HarnessError::state(
                "the harness observation stream already has an active consumer",
            )));
        }
        let Some(receiver) = state.receiver.take() else {
            return Box::new(ErrorCursor::new(HarnessError::state(
                "native harness has no active event cycle",
            )));
        };
        state.leased = true;
        drop(state);
        Box::new(ChannelEventCursor {
            channel: Arc::clone(self),
            receiver: Some(receiver),
            view,
            selected: false,
            selected_turn: None,
            closed: false,
        })
    }

    /// Closes the cycle: idempotent, drops the sender, and marks the channel closed.
    ///
    /// A cursor that currently holds the receiver drains the remaining buffered
    /// events and then ends; every later cursor reports that no cycle is active.
    pub(crate) fn close(&self) {
        let mut state = self.state.lock();
        state.closed = true;
        state.sender = None;
    }

    /// Returns whether [`close`](Self::close) already ran.
    pub(crate) fn is_closed(&self) -> bool {
        self.state.lock().closed
    }
}

/// Which slice of the ordered observations a [`ChannelEventCursor`] exposes.
#[derive(Debug)]
enum View {
    /// Cycle-long: emits every event until the cycle closes.
    Cycle,
    /// Turn-scoped: selects the next unconsumed turn and ends at its terminal event.
    Turn {
        /// Turn the caller asked for, when it pinned the view to one turn.
        requested: Option<String>,
    },
}

/// Cursor over one [`EventChannel`] view; owns the single-consumer lease.
#[derive(Debug)]
struct ChannelEventCursor {
    /// Channel the cursor reads from and returns the lease to.
    channel: Arc<EventChannel>,
    /// Receiver while the lease is held; `None` once released.
    receiver: Option<Receiver<HarnessEvent>>,
    /// View this cursor exposes.
    view: View,
    /// Whether the turn view already selected its turn.
    selected: bool,
    /// Turn id selected by the turn view, when known.
    selected_turn: Option<String>,
    /// Whether [`HarnessEventCursor::close`] already ran.
    closed: bool,
}

impl ChannelEventCursor {
    /// Receives the next event while the lease is held, or `None` once released.
    async fn recv_next(&mut self) -> Option<HarnessEvent> {
        match self.receiver.as_mut() {
            Some(receiver) => receiver.recv().await,
            None => None,
        }
    }

    /// Reads the next event of the cycle view.
    async fn next_cycle_event(&mut self) -> HarnessResult<Option<HarnessEvent>> {
        let Some(event) = self.recv_next().await else {
            self.release();
            return Ok(None);
        };
        Ok(Some(event))
    }

    /// Reads the next event of a turn view, discarding events before the turn starts.
    async fn next_turn_event(&mut self) -> HarnessResult<Option<HarnessEvent>> {
        loop {
            let Some(event) = self.recv_next().await else {
                self.release();
                return Ok(None);
            };
            if !self.selected {
                if !is_turn_start(&event) {
                    continue;
                }
                if let Some(expected) = self.requested_turn().map(str::to_owned) {
                    let found = event.turn_id.clone().unwrap_or_default();
                    if found != expected {
                        self.release();
                        return Err(HarnessError::state(format!(
                            "requested turn is not the next unconsumed turn: expected {expected}, found {found}"
                        )));
                    }
                }
                self.selected = true;
                self.selected_turn.clone_from(&event.turn_id);
                return Ok(Some(event));
            }
            let terminal = is_terminal_for(&event, self.selected_turn.as_deref());
            if terminal {
                self.release();
            }
            return Ok(Some(event));
        }
    }

    /// Returns the turn id the view was pinned to, if any.
    fn requested_turn(&self) -> Option<&str> {
        match &self.view {
            View::Cycle => None,
            View::Turn { requested } => requested.as_deref(),
        }
    }

    /// Releases the lease and returns the receiver to the channel, exactly once.
    fn release(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;
        let receiver = self.receiver.take();
        let mut state = self.channel.state.lock();
        if state.receiver.is_none() {
            state.receiver = receiver;
        }
        state.leased = false;
    }
}

/// Returns whether `event` starts a turn.
fn is_turn_start(event: &HarnessEvent) -> bool {
    matches!(
        &event.event,
        HarnessEventPayload::TurnLifecycle(payload) if payload.kind == TurnEventKind::Started
    )
}

/// Returns whether `event` is the terminal event of `turn`.
fn is_terminal_for(event: &HarnessEvent, turn: Option<&str>) -> bool {
    matches!(
        &event.event,
        HarnessEventPayload::TurnLifecycle(payload) if payload.kind.is_terminal()
    ) && event.turn_id.as_deref() == turn
}

#[async_trait]
impl HarnessEventCursor for ChannelEventCursor {
    /// Returns the next event of the selected view, or `None` once it is exhausted.
    ///
    /// # Errors
    ///
    /// [`HarnessError::State`] when a turn view was pinned to a turn that is not
    /// the next unconsumed turn. The cursor is released before the error is
    /// returned, so a later cursor can continue from the same position.
    async fn next_event(&mut self) -> HarnessResult<Option<HarnessEvent>> {
        if self.closed {
            return Ok(None);
        }
        match self.view {
            View::Cycle => self.next_cycle_event().await,
            View::Turn { .. } => self.next_turn_event().await,
        }
    }

    /// Idempotently releases the lease, handing the receiver back to the channel.
    ///
    /// # Errors
    ///
    /// Never fails; the lease is entirely in-process.
    async fn close(&mut self) -> HarnessResult<()> {
        self.release();
        Ok(())
    }
}

impl Drop for ChannelEventCursor {
    /// Abandon path: releases the lease when the cursor was dropped unclosed.
    fn drop(&mut self) {
        self.release();
    }
}

/// Cursor that reports the same [`HarnessError::State`] once, then ends.
#[derive(Debug)]
struct ErrorCursor {
    /// Error to surface on the first read; taken so it is reported only once.
    error: Option<HarnessError>,
}

impl ErrorCursor {
    /// Builds a cursor that reports `error` on its first read.
    fn new(error: HarnessError) -> Self {
        Self { error: Some(error) }
    }
}

#[async_trait]
impl HarnessEventCursor for ErrorCursor {
    /// Reports the stored state error once, then ends.
    ///
    /// # Errors
    ///
    /// The [`HarnessError::State`] supplied to [`ErrorCursor::new`], on the
    /// first call only.
    async fn next_event(&mut self) -> HarnessResult<Option<HarnessEvent>> {
        match self.error.take() {
            Some(error) => Err(error),
            None => Ok(None),
        }
    }

    /// Closes the cursor; releasing nothing because it never held a lease.
    ///
    /// # Errors
    ///
    /// Never fails.
    async fn close(&mut self) -> HarnessResult<()> {
        self.error = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::EventChannel;
    use openjiuwen_harness_protocol::{
        DiagnosticEvent, DiagnosticLevel, EventBufferConfig, EventOverflowPolicy, HarnessErrorKind,
        HarnessEvent, HarnessEventPayload, HarnessState, JsonObject, OutputChannel, OutputEvent,
        OutputKind, OutputOperation, StateChangedEvent, TurnEventKind, TurnLifecycleEvent,
    };
    use serde_json::Value;

    /// Builds a `StateChanged` payload.
    fn state_payload() -> HarnessEventPayload {
        StateChangedEvent {
            old: HarnessState::Idle,
            new: HarnessState::Running,
        }
        .into()
    }

    /// Builds a turn lifecycle payload.
    fn turn_payload(kind: TurnEventKind) -> HarnessEventPayload {
        TurnLifecycleEvent { kind, result: None }.into()
    }

    /// Builds a delta output payload.
    fn output_payload(text: &str) -> HarnessEventPayload {
        OutputEvent {
            output_id: "out".to_owned(),
            kind: OutputKind::Text,
            content: Value::String(text.to_owned()),
            operation: OutputOperation::Delta,
            channel: OutputChannel::Answer,
            content_index: 0,
            content_type: Some("text/plain".to_owned()),
            data: JsonObject::new(),
        }
        .into()
    }

    /// Builds one event, optionally scoped to `turn`.
    fn event(sequence: u64, payload: HarnessEventPayload, turn: Option<&str>) -> HarnessEvent {
        let mut event = HarnessEvent::new(sequence, 1.0, payload, "session-1", "agent-1");
        if let Some(turn) = turn {
            event = event.with_turn_id(turn);
        }
        event
    }

    /// Builds a channel with the default contract.
    fn channel() -> std::sync::Arc<EventChannel> {
        EventChannel::open(EventBufferConfig::default())
    }

    #[tokio::test]
    async fn cycle_view_preserves_order_and_ends_when_closed() {
        let channel = channel();
        channel.emit(event(1, state_payload(), None)).await;
        channel.emit(event(2, output_payload("hi"), None)).await;
        let mut cursor = channel.events();
        channel.close();
        assert!(channel.is_closed());
        assert_eq!(cursor.next_event().await.unwrap().unwrap().sequence, 1);
        assert_eq!(cursor.next_event().await.unwrap().unwrap().sequence, 2);
        assert!(cursor.next_event().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn cursor_after_close_reports_no_active_cycle() {
        let channel = channel();
        channel.close();
        let mut cursor = channel.events();
        let error = cursor.next_event().await.unwrap_err();
        assert_eq!(error.kind(), HarnessErrorKind::State);
        assert_eq!(error.message(), "native harness has no active event cycle");
        assert!(cursor.next_event().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn capacity_is_clamped_to_at_least_one() {
        let channel = EventChannel::open(EventBufferConfig {
            capacity: 0,
            overflow: EventOverflowPolicy::Block,
        });
        assert_eq!(channel.config().capacity, 1);
        assert_eq!(channel.config().overflow, EventOverflowPolicy::Block);
    }

    #[tokio::test]
    async fn second_cursor_is_rejected_while_the_lease_is_held() {
        let channel = channel();
        channel.emit(event(1, state_payload(), None)).await;
        let mut first = channel.events();
        let mut second = channel.events();
        let error = second.next_event().await.unwrap_err();
        assert_eq!(error.kind(), HarnessErrorKind::State);
        assert_eq!(
            error.message(),
            "the harness observation stream already has an active consumer"
        );
        assert_eq!(first.next_event().await.unwrap().unwrap().sequence, 1);
    }

    #[tokio::test]
    async fn close_is_idempotent_and_a_later_cursor_continues_without_replay() {
        let channel = channel();
        for sequence in 1..=3 {
            channel.emit(event(sequence, state_payload(), None)).await;
        }
        let mut first = channel.events();
        assert_eq!(first.next_event().await.unwrap().unwrap().sequence, 1);
        first.close().await.unwrap();
        first.close().await.unwrap();
        assert!(first.next_event().await.unwrap().is_none());

        let mut second = channel.events();
        assert_eq!(second.next_event().await.unwrap().unwrap().sequence, 2);
        assert_eq!(second.next_event().await.unwrap().unwrap().sequence, 3);
        channel.close();
        assert!(second.next_event().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn dropping_a_cursor_releases_the_lease() {
        let channel = channel();
        for sequence in 1..=3 {
            channel.emit(event(sequence, state_payload(), None)).await;
        }
        let mut first = channel.events();
        assert_eq!(first.next_event().await.unwrap().unwrap().sequence, 1);
        drop(first);

        let mut second = channel.events();
        assert_eq!(second.next_event().await.unwrap().unwrap().sequence, 2);
        assert_eq!(second.next_event().await.unwrap().unwrap().sequence, 3);
    }

    #[tokio::test]
    async fn turn_view_skips_unrelated_events_and_ends_at_the_terminal_event() {
        let channel = channel();
        channel.emit(event(1, state_payload(), None)).await;
        channel
            .emit(event(
                2,
                turn_payload(TurnEventKind::Started),
                Some("turn-1"),
            ))
            .await;
        channel
            .emit(event(3, output_payload("one"), Some("turn-1")))
            .await;
        channel
            .emit(event(
                4,
                turn_payload(TurnEventKind::Finished),
                Some("turn-1"),
            ))
            .await;
        channel.emit(event(5, state_payload(), None)).await;
        channel
            .emit(event(
                6,
                turn_payload(TurnEventKind::Started),
                Some("turn-2"),
            ))
            .await;
        channel
            .emit(event(7, output_payload("two"), Some("turn-2")))
            .await;
        channel
            .emit(event(
                8,
                turn_payload(TurnEventKind::Finished),
                Some("turn-2"),
            ))
            .await;

        let mut first = channel.turn_events(None);
        for expected in [2, 3, 4] {
            assert_eq!(
                first.next_event().await.unwrap().unwrap().sequence,
                expected
            );
        }
        assert!(
            first.next_event().await.unwrap().is_none(),
            "the terminal event ends the turn view"
        );

        let mut second = channel.turn_events(Some("turn-2"));
        for expected in [6, 7, 8] {
            assert_eq!(
                second.next_event().await.unwrap().unwrap().sequence,
                expected
            );
        }
        assert!(
            second.next_event().await.unwrap().is_none(),
            "the unrelated event between turns is discarded and the terminal ends the view"
        );
    }

    #[tokio::test]
    async fn turn_view_selection_is_destructive() {
        let channel = channel();
        channel
            .emit(event(
                1,
                turn_payload(TurnEventKind::Started),
                Some("turn-1"),
            ))
            .await;
        channel
            .emit(event(
                2,
                turn_payload(TurnEventKind::Finished),
                Some("turn-1"),
            ))
            .await;
        channel
            .emit(event(
                3,
                turn_payload(TurnEventKind::Started),
                Some("turn-2"),
            ))
            .await;
        channel
            .emit(event(
                4,
                turn_payload(TurnEventKind::Finished),
                Some("turn-2"),
            ))
            .await;

        let mut cursor = channel.turn_events(None);
        assert_eq!(cursor.next_event().await.unwrap().unwrap().sequence, 1);
        assert_eq!(cursor.next_event().await.unwrap().unwrap().sequence, 2);
        assert!(cursor.next_event().await.unwrap().is_none());

        let mut next = channel.turn_events(None);
        assert_eq!(next.next_event().await.unwrap().unwrap().sequence, 3);
    }

    #[tokio::test]
    async fn requested_turn_mismatch_is_a_state_error_and_releases_the_lease() {
        let channel = channel();
        channel
            .emit(event(
                1,
                turn_payload(TurnEventKind::Started),
                Some("turn-1"),
            ))
            .await;
        channel
            .emit(event(
                2,
                turn_payload(TurnEventKind::Finished),
                Some("turn-1"),
            ))
            .await;
        channel
            .emit(event(
                3,
                turn_payload(TurnEventKind::Started),
                Some("turn-2"),
            ))
            .await;

        let mut cursor = channel.turn_events(Some("turn-2"));
        let error = cursor.next_event().await.unwrap_err();
        assert_eq!(error.kind(), HarnessErrorKind::State);
        assert_eq!(
            error.message(),
            "requested turn is not the next unconsumed turn: expected turn-2, found turn-1"
        );
        assert!(cursor.next_event().await.unwrap().is_none());

        let mut next = channel.turn_events(None);
        assert_eq!(
            next.next_event().await.unwrap().unwrap().sequence,
            3,
            "the mismatched cursor consumed only the turn-1 start, so the next \
             view skips the turn-1 terminal and selects turn-2"
        );
    }

    #[tokio::test]
    async fn paused_and_resumed_events_do_not_end_the_turn_view() {
        let channel = channel();
        for (sequence, kind) in [
            (1, TurnEventKind::Started),
            (2, TurnEventKind::Paused),
            (3, TurnEventKind::Resumed),
            (4, TurnEventKind::Finished),
        ] {
            channel
                .emit(event(sequence, turn_payload(kind), Some("turn-1")))
                .await;
        }
        let mut cursor = channel.turn_events(Some("turn-1"));
        for expected in 1..=4 {
            let next = cursor.next_event().await.unwrap().unwrap();
            assert_eq!(next.sequence, expected);
        }
        assert!(cursor.next_event().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn coalescing_policy_drops_best_effort_events_when_full() {
        let channel = EventChannel::open(EventBufferConfig {
            capacity: 1,
            overflow: EventOverflowPolicy::CoalesceOrBlock,
        });
        let mut cursor = channel.events();
        channel.emit(event(1, state_payload(), None)).await;
        channel
            .emit(event(
                2,
                DiagnosticEvent {
                    level: DiagnosticLevel::Debug,
                    message: "noise".to_owned(),
                    data: JsonObject::new(),
                }
                .into(),
                None,
            ))
            .await;
        channel.close();
        assert_eq!(cursor.next_event().await.unwrap().unwrap().sequence, 1);
        assert!(cursor.next_event().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn drop_best_effort_policy_keeps_coalescible_events() {
        let channel = EventChannel::open(EventBufferConfig {
            capacity: 1,
            overflow: EventOverflowPolicy::DropBestEffortOrBlock,
        });
        let mut cursor = channel.events();
        channel.emit(event(1, state_payload(), None)).await;
        // `state_payload` is required retention; with `DropBestEffortOrBlock`
        // it would block, so only the dropped best-effort event is checked here.
        channel
            .emit(event(
                2,
                DiagnosticEvent {
                    level: DiagnosticLevel::Debug,
                    message: "noise".to_owned(),
                    data: JsonObject::new(),
                }
                .into(),
                None,
            ))
            .await;
        channel.close();
        assert_eq!(cursor.next_event().await.unwrap().unwrap().sequence, 1);
        assert!(cursor.next_event().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn emit_after_close_is_a_noop() {
        let channel = channel();
        channel.close();
        channel.emit(event(1, state_payload(), None)).await;
        assert!(channel.events().next_event().await.is_err());
    }
}
