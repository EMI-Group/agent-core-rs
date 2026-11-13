//! The observation bus every vendor CLI harness publishes its events through.
//!
//! A vendor harness observes its CLI asynchronously (a pump task reads NDJSON lines) while the host
//! consumes the harness SPI event cursor ([`HarnessEventCursor`]). [`EventBus`] is the seam between
//! the two:
//!
//! - [`EventBus::emit`] appends an event to a bounded queue and wakes the consumer; on overflow the
//!   *oldest* observation is dropped, so the consumer always sees the most recent turn rather than
//!   a stale prefix (matching the SPI's `drop_best_effort_or_block` overflow contract);
//! - [`EventBus::events`] / [`EventBus::turn_events`] hand out the single allowed consumer cursor
//!   for the current start/stop cycle, so a second concurrent consumer is a [`HarnessError::State`]
//!   instead of a race;
//! - a cursor starts at the first event still retained (events emitted before the cursor opened are
//!   visible), yields in `sequence` order, and returns `Ok(None)` only once the bus is closed and
//!   drained;
//! - [`EventBus::close`] is idempotent and ends every cursor after its remaining events, while a
//!   cursor's own `close` releases its lease early without discarding anything.
//!
//! [`EnvelopeBuilder`] fills the envelope fields a vendor harness never has to invent: the
//! session/agent identity, the provider session id, a strictly increasing `sequence` and a finite
//! epoch-second `timestamp`.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use openjiuwen_harness_protocol::{
    HarnessError, HarnessEvent, HarnessEventCursor, HarnessEventPayload, HarnessResult,
};

use super::lock;

/// Capacity used when [`EventBus::new`] is given zero.
pub const DEFAULT_CAPACITY: usize = 1024;

/// Bounded, single-consumer observation queue for one harness start/stop cycle.
pub struct EventBus {
    /// Maximum number of retained events.
    capacity: usize,
    /// Retained events, oldest first.
    events: Mutex<VecDeque<HarnessEvent>>,
    /// Wakes cursors waiting for the next event or for the cycle to close.
    notify: tokio::sync::Notify,
    /// Whether the cycle closed; cursors end after draining.
    closed: AtomicBool,
    /// Whether a cursor currently holds the single-consumer lease.
    leased: AtomicBool,
    /// Whether the current overflow episode was already reported.
    overflow_warned: AtomicBool,
}

impl EventBus {
    /// Creates a shared bus retaining at most `capacity` events (`0` selects
    /// [`DEFAULT_CAPACITY`]).
    #[must_use]
    pub fn new(capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            capacity: if capacity == 0 {
                DEFAULT_CAPACITY
            } else {
                capacity
            },
            events: Mutex::new(VecDeque::new()),
            notify: tokio::sync::Notify::new(),
            closed: AtomicBool::new(false),
            leased: AtomicBool::new(false),
            overflow_warned: AtomicBool::new(false),
        })
    }

    /// Appends `event`, dropping the oldest retained event when the queue is full.
    ///
    /// Overflow is not an error: an observation tail is more valuable than a stale head, and the
    /// harness SPI expects the buffer to degrade instead of blocking a live turn. The first drop of
    /// each overflow episode is logged once at `warn`.
    #[allow(clippy::unused_async)] // async keeps the bus composable with a future backpressure path
    pub async fn emit(&self, event: HarnessEvent) {
        let dropped = {
            let mut events = lock(&self.events);
            let dropped = events.len() >= self.capacity;
            if dropped {
                events.pop_front();
            }
            events.push_back(event);
            dropped
        };
        if dropped {
            if !self.overflow_warned.swap(true, Ordering::SeqCst) {
                tracing::warn!(
                    capacity = self.capacity,
                    "the harness observation buffer is full; dropping the oldest event"
                );
            }
        } else {
            self.overflow_warned.store(false, Ordering::SeqCst);
        }
        self.notify.notify_waiters();
    }

    /// Opens the single-consumer cursor over the current cycle.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::State`] when another cursor still holds the lease.
    pub fn events(self: &Arc<Self>) -> HarnessResult<Box<dyn HarnessEventCursor>> {
        self.open(None)
            .map(|cursor| Box::new(cursor) as Box<dyn HarnessEventCursor>)
    }

    /// Opens the single-consumer cursor restricted to one turn.
    ///
    /// Only events whose envelope `turn_id` equals `turn_id` are yielded (events without a turn id
    /// are skipped), and the view ends after the turn's terminal lifecycle event. A [`None`] turn id
    /// yields the whole cycle.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::State`] when another cursor still holds the lease.
    pub fn turn_events(
        self: &Arc<Self>,
        turn_id: Option<&str>,
    ) -> HarnessResult<Box<dyn HarnessEventCursor>> {
        let filter = turn_id.map(|turn_id| TurnFilter {
            turn_id: turn_id.to_owned(),
            finished: false,
        });
        self.open(filter)
            .map(|cursor| Box::new(cursor) as Box<dyn HarnessEventCursor>)
    }

    /// Closes the cycle: no further events are accepted and cursors end after draining.
    ///
    /// Idempotent, and safe to call while a cursor is waiting.
    pub fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
        self.notify.notify_waiters();
    }

    /// Returns whether the cycle is closed.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    /// Returns the number of retained events.
    #[must_use]
    pub fn len(&self) -> usize {
        lock(&self.events).len()
    }

    /// Returns whether no event is currently retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Takes the consumer lease and builds the cursor.
    fn open(self: &Arc<Self>, filter: Option<TurnFilter>) -> HarnessResult<EventCursor> {
        if self.leased.swap(true, Ordering::SeqCst) {
            return Err(HarnessError::state(
                "the harness observation stream already has an active consumer",
            ));
        }
        Ok(EventCursor {
            bus: Arc::clone(self),
            filter,
            last_sequence: None,
            lease_held: true,
            closed: false,
        })
    }
}

/// Turn restriction of a [`EventCursor`].
struct TurnFilter {
    /// Turn the cursor observes.
    turn_id: String,
    /// Whether the terminal lifecycle event was already yielded.
    finished: bool,
}

/// Single-consumer view over an [`EventBus`].
struct EventCursor {
    /// The observed bus.
    bus: Arc<EventBus>,
    /// Optional turn restriction.
    filter: Option<TurnFilter>,
    /// Sequence of the last event handed to the consumer.
    last_sequence: Option<u64>,
    /// Whether this cursor still holds the consumer lease.
    lease_held: bool,
    /// Whether the view is finished.
    closed: bool,
}

impl EventCursor {
    /// Marks the view finished and releases the lease.
    fn finish(&mut self) {
        self.closed = true;
        self.release();
    }

    /// Releases the single-consumer lease, if this cursor holds it.
    fn release(&mut self) {
        if self.lease_held {
            self.lease_held = false;
            self.bus.leased.store(false, Ordering::SeqCst);
        }
    }
}

/// Returns the next event at or after the cursor position, applying the turn filter.
///
/// Events that do not belong to the observed turn advance the cursor position (so they are not
/// rescanned) without being yielded; the first terminal turn-lifecycle event finishes the view.
fn take_next(
    filter: &mut Option<TurnFilter>,
    last_sequence: &mut Option<u64>,
    events: &VecDeque<HarnessEvent>,
) -> Option<HarnessEvent> {
    for event in events {
        if last_sequence.is_some_and(|last| event.sequence <= last) {
            continue;
        }
        *last_sequence = Some(event.sequence);
        let matches = match filter.as_ref() {
            None => true,
            Some(state) => event.turn_id.as_deref() == Some(state.turn_id.as_str()),
        };
        if !matches {
            continue;
        }
        let terminal = matches!(
            event.event,
            HarnessEventPayload::TurnLifecycle(ref lifecycle) if lifecycle.kind.is_terminal()
        );
        if terminal {
            if let Some(state) = filter.as_mut() {
                state.finished = true;
            }
        }
        return Some(event.clone());
    }
    None
}

#[async_trait]
impl HarnessEventCursor for EventCursor {
    async fn next_event(&mut self) -> HarnessResult<Option<HarnessEvent>> {
        loop {
            if self.closed {
                return Ok(None);
            }
            if self.filter.as_ref().is_some_and(|filter| filter.finished) {
                self.finish();
                return Ok(None);
            }
            // Register interest before inspecting the queue, so an emit racing this check wakes us.
            // The bus is cloned so the borrow of the local does not overlap the cursor's own
            // mutable borrow of its filter state below.
            let bus = Arc::clone(&self.bus);
            let notified = bus.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let next = {
                let events = lock(&bus.events);
                take_next(&mut self.filter, &mut self.last_sequence, &events)
            };
            if let Some(event) = next {
                return Ok(Some(event));
            }
            if self.bus.is_closed() {
                self.finish();
                return Ok(None);
            }
            notified.await;
        }
    }

    async fn close(&mut self) -> HarnessResult<()> {
        self.finish();
        Ok(())
    }
}

impl Drop for EventCursor {
    fn drop(&mut self) {
        self.release();
    }
}

/// Fills the shared envelope fields of every event a vendor harness emits.
pub struct EnvelopeBuilder {
    /// Host session every event belongs to.
    host_session_id: String,
    /// Agent that produces the observations.
    agent_id: String,
    /// Provider-side session id, learned once the CLI reveals it.
    provider_session_id: Mutex<Option<String>>,
    /// Last allocated event sequence.
    sequence: AtomicU64,
}

impl EnvelopeBuilder {
    /// Creates a builder for `host_session_id` / `agent_id` with no provider session and sequence
    /// starting at 1.
    #[must_use]
    pub fn new(host_session_id: impl Into<String>, agent_id: impl Into<String>) -> Self {
        Self {
            host_session_id: host_session_id.into(),
            agent_id: agent_id.into(),
            provider_session_id: Mutex::new(None),
            sequence: AtomicU64::new(0),
        }
    }

    /// Sets (or clears) the provider session id stamped on later events.
    pub fn set_provider_session_id(&self, id: Option<String>) {
        *lock(&self.provider_session_id) = id;
    }

    /// Builds the next event envelope for `payload`.
    #[must_use]
    pub fn next(&self, payload: impl Into<HarnessEventPayload>) -> HarnessEvent {
        let sequence = self.sequence.fetch_add(1, Ordering::SeqCst) + 1;
        let mut event = HarnessEvent::new(
            sequence,
            now_epoch_seconds(),
            payload,
            self.host_session_id.clone(),
            self.agent_id.clone(),
        );
        if let Some(provider_session_id) = lock(&self.provider_session_id).clone() {
            event.provider_session_id = Some(provider_session_id);
        }
        event
    }

    /// Builds the next envelope for `payload`, tagged with `turn_id`.
    #[must_use]
    pub fn next_for_turn(
        &self,
        payload: impl Into<HarnessEventPayload>,
        turn_id: &str,
    ) -> HarnessEvent {
        self.next(payload).with_turn_id(turn_id)
    }

    /// Builds the next envelope for `payload`, tagged with `turn_id` and `item_id`.
    #[must_use]
    pub fn next_for_item(
        &self,
        payload: impl Into<HarnessEventPayload>,
        turn_id: &str,
        item_id: &str,
    ) -> HarnessEvent {
        self.next(payload)
            .with_turn_id(turn_id)
            .with_item_id(item_id)
    }
}

/// Returns the current Unix epoch second as a finite `f64`, or `0.0` when the clock is unreadable.
#[allow(clippy::cast_precision_loss)] // the SPI timestamp is `f64` epoch seconds
fn now_epoch_seconds() -> f64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(elapsed) => {
            elapsed.as_secs() as f64 + f64::from(elapsed.subsec_nanos()) / 1_000_000_000.0
        }
        Err(error) => {
            tracing::warn!(%error, "system clock is before the Unix epoch; using timestamp 0.0");
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use std::time::Duration;

    use openjiuwen_harness_protocol::{
        HarnessErrorKind, HarnessEvent, HarnessEventPayload, JsonObject, JsonValue, OutputChannel,
        OutputEvent, OutputKind, OutputOperation, TurnEventKind, TurnLifecycleEvent,
    };

    use super::{EnvelopeBuilder, EventBus};

    /// Bounded await budget for every bus interaction in these tests.
    const BOUND: Duration = Duration::from_secs(10);

    /// An output payload, so an event carries no terminal semantics.
    fn output(text: &str) -> OutputEvent {
        OutputEvent {
            output_id: "output-1".to_owned(),
            kind: OutputKind::Text,
            content: JsonValue::String(text.to_owned()),
            operation: OutputOperation::default(),
            channel: OutputChannel::default(),
            content_index: 0,
            content_type: None,
            data: JsonObject::new(),
        }
    }

    /// A turn lifecycle payload.
    fn lifecycle(kind: TurnEventKind) -> HarnessEventPayload {
        HarnessEventPayload::TurnLifecycle(TurnLifecycleEvent { kind, result: None })
    }

    /// Builder over a fixed identity.
    fn builder() -> EnvelopeBuilder {
        EnvelopeBuilder::new("host-session", "agent-a")
    }

    /// Pulls the next event with a bounded await.
    async fn next(
        cursor: &mut Box<dyn openjiuwen_harness_protocol::HarnessEventCursor>,
    ) -> Option<HarnessEvent> {
        tokio::time::timeout(BOUND, cursor.next_event())
            .await
            .expect("the cursor must not block")
            .expect("the cursor must not fail")
    }

    #[tokio::test]
    async fn yields_events_emitted_before_the_cursor_opened() {
        let bus = EventBus::new(8);
        let envelopes = builder();
        bus.emit(envelopes.next(output("one"))).await;
        bus.emit(envelopes.next(output("two"))).await;
        bus.close();
        let mut cursor = bus.events().expect("no other consumer");
        assert_eq!(next(&mut cursor).await.map(|event| event.sequence), Some(1));
        assert_eq!(next(&mut cursor).await.map(|event| event.sequence), Some(2));
        assert!(next(&mut cursor).await.is_none());
    }

    #[tokio::test]
    async fn waits_for_an_event_emitted_after_the_cursor_opened() {
        let bus = EventBus::new(8);
        let mut cursor = bus.events().expect("no other consumer");
        let producer = {
            let bus = bus.clone();
            let envelopes = builder();
            tokio::spawn(async move {
                bus.emit(envelopes.next(output("late"))).await;
            })
        };
        let event = next(&mut cursor).await.expect("the emitted event");
        assert_eq!(event.sequence, 1);
        producer.await.expect("the producer task");
    }

    #[tokio::test]
    async fn a_second_cursor_is_rejected_until_the_lease_is_released() {
        let bus = EventBus::new(4);
        let mut first = bus.events().expect("the first consumer");
        let error = bus.events().err().expect("the lease is held");
        assert_eq!(error.kind(), HarnessErrorKind::State);
        assert_eq!(
            error.message(),
            "the harness observation stream already has an active consumer"
        );
        first.close().await.expect("closing releases the lease");
        let second = bus.events().expect("the lease is free again");
        drop(second);
    }

    #[tokio::test]
    async fn dropping_a_cursor_releases_the_lease() {
        let bus = EventBus::new(4);
        let cursor = bus.events().expect("the first consumer");
        drop(cursor);
        assert!(bus.events().is_ok());
    }

    #[tokio::test]
    async fn close_then_drain_returns_none() {
        let bus = EventBus::new(4);
        let envelopes = builder();
        bus.emit(envelopes.next(output("one"))).await;
        bus.close();
        assert!(bus.is_closed());
        let mut cursor = bus.events().expect("no other consumer");
        assert!(next(&mut cursor).await.is_some());
        assert!(next(&mut cursor).await.is_none());
        assert!(next(&mut cursor).await.is_none());
    }

    #[tokio::test]
    async fn an_open_bus_does_not_end_the_cursor() {
        let bus = EventBus::new(4);
        let mut cursor = bus.events().expect("no other consumer");
        tokio::time::timeout(Duration::from_millis(50), cursor.next_event())
            .await
            .expect_err("an open, empty bus keeps the cursor waiting");
    }

    #[tokio::test]
    async fn overflow_drops_the_oldest_event() {
        let bus = EventBus::new(2);
        let envelopes = builder();
        for index in 0..3 {
            bus.emit(envelopes.next(output(&format!("event {index}"))))
                .await;
        }
        assert_eq!(bus.len(), 2);
        bus.close();
        let mut cursor = bus.events().expect("no other consumer");
        assert_eq!(next(&mut cursor).await.map(|event| event.sequence), Some(2));
        assert_eq!(next(&mut cursor).await.map(|event| event.sequence), Some(3));
        assert!(next(&mut cursor).await.is_none());
    }

    #[tokio::test]
    async fn a_turn_cursor_skips_other_turns_and_ends_at_the_terminal_event() {
        let bus = EventBus::new(16);
        let envelopes = builder();
        bus.emit(envelopes.next_for_turn(lifecycle(TurnEventKind::Started), "t1"))
            .await;
        bus.emit(envelopes.next(output("unowned"))).await;
        bus.emit(envelopes.next_for_turn(output("other turn"), "t2"))
            .await;
        bus.emit(envelopes.next_for_turn(output("mine"), "t1"))
            .await;
        bus.emit(envelopes.next_for_turn(lifecycle(TurnEventKind::Finished), "t1"))
            .await;
        bus.emit(envelopes.next_for_turn(output("after terminal"), "t1"))
            .await;
        let mut cursor = bus.turn_events(Some("t1")).expect("no other consumer");
        let mut seen = Vec::new();
        while let Some(event) = next(&mut cursor).await {
            seen.push((event.sequence, event.turn_id.clone()));
        }
        assert_eq!(
            seen,
            vec![
                (1, Some("t1".to_owned())),
                (4, Some("t1".to_owned())),
                (5, Some("t1".to_owned())),
            ]
        );
    }

    #[tokio::test]
    async fn a_paused_turn_cursor_keeps_waiting() {
        let bus = EventBus::new(8);
        let envelopes = builder();
        bus.emit(envelopes.next_for_turn(lifecycle(TurnEventKind::Paused), "t1"))
            .await;
        let mut cursor = bus.turn_events(Some("t1")).expect("no other consumer");
        assert_eq!(next(&mut cursor).await.map(|event| event.sequence), Some(1));
        tokio::time::timeout(Duration::from_millis(50), cursor.next_event())
            .await
            .expect_err("a paused turn is not terminal");
    }

    #[tokio::test]
    async fn turn_events_without_a_turn_id_observes_the_cycle() {
        let bus = EventBus::new(8);
        let envelopes = builder();
        bus.emit(envelopes.next_for_turn(output("one"), "t1")).await;
        bus.emit(envelopes.next_for_turn(output("two"), "t2")).await;
        bus.close();
        let mut cursor = bus.turn_events(None).expect("no other consumer");
        assert_eq!(next(&mut cursor).await.map(|event| event.sequence), Some(1));
        assert_eq!(next(&mut cursor).await.map(|event| event.sequence), Some(2));
        assert!(next(&mut cursor).await.is_none());
    }

    #[test]
    fn envelope_builder_stamps_identity_sequence_and_timestamp() {
        let envelopes = EnvelopeBuilder::new("host-session", "agent-a");
        envelopes.set_provider_session_id(Some("provider-session".to_owned()));
        let first = envelopes.next(output("one"));
        let second = envelopes.next_for_turn(output("two"), "t1");
        let third = envelopes.next_for_item(output("three"), "t1", "item-1");
        assert_eq!(first.sequence, 1);
        assert_eq!(second.sequence, 2);
        assert_eq!(third.sequence, 3);
        assert_eq!(first.host_session_id, "host-session");
        assert_eq!(first.agent_id, "agent-a");
        assert_eq!(
            first.provider_session_id.as_deref(),
            Some("provider-session")
        );
        assert_eq!(second.turn_id.as_deref(), Some("t1"));
        assert_eq!(third.item_id.as_deref(), Some("item-1"));
        assert!(first.timestamp.is_finite());
        assert!(first.timestamp > 0.0);
    }

    #[test]
    fn envelope_builder_can_clear_the_provider_session() {
        let envelopes = EnvelopeBuilder::new("host-session", "agent-a");
        envelopes.set_provider_session_id(Some("provider-session".to_owned()));
        envelopes.set_provider_session_id(None);
        assert!(envelopes.next(output("one")).provider_session_id.is_none());
    }

    #[test]
    fn zero_capacity_falls_back_to_the_default() {
        let bus = EventBus::new(0);
        assert_eq!(bus.capacity, super::DEFAULT_CAPACITY);
        assert!(bus.is_empty());
        assert_eq!(bus.len(), 0);
    }
}
