//! Round C protocol/cursor integration suite: the `HarnessProtocol` SPI through trait objects, the
//! shared single-consumer cursor fakes, `event_cursor_stream`, `VecEventCursor` semantics, and the
//! compile-level public-surface checklist.
//!
//! Python counterpart: `tests/unit_tests/harness_protocol/test_protocol.py` (v0.1.20). Where a
//! ported case exists it is named in a `// Python:` comment. The Python `_Harness` fixture is
//! re-created here as [`FakeHarness`], and the cursor fakes mirror the Python fake cursor's
//! single-consumer, close-without-rewind behaviour.
//!
//! No runtime is used: async methods run on `futures::executor::block_on`, there are no sleeps and
//! no network calls, so every test is deterministic.

// Tests assert by panicking, which is the intent of a test suite.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::collections::{BTreeSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use async_trait::async_trait;
use futures::executor::block_on;
use futures::StreamExt;
use serde_json::json;

use openjiuwen_harness_protocol::*;

/// Timestamp shared by every fixture event; only the order matters to these tests.
const EVENT_TIMESTAMP: f64 = 1_700_000_000.0;

/// Builds a validated observation with the crate's shared fixture identity.
fn harness_event(sequence: u64, payload: impl Into<HarnessEventPayload>) -> HarnessEvent {
    HarnessEvent::new(
        sequence,
        EVENT_TIMESTAMP,
        payload,
        "host-session",
        "agent-a",
    )
}

/// Builds a validated state transition observation.
fn state_change(sequence: u64, old: HarnessState, new: HarnessState) -> HarnessEvent {
    harness_event(sequence, StateChangedEvent { old, new })
}

/// A completed terminal result whose `final_output` is `"done"`.
fn completed_result() -> TurnResult {
    TurnResult {
        status: TurnStatus::Completed,
        messages: Vec::new(),
        final_output: json!("done"),
        structured_output: JsonValue::Null,
        stop_reason: None,
        termination: None,
        error: None,
        usage: None,
        cost: None,
        started_at: None,
        completed_at: None,
        duration_ms: None,
        provider_data: empty_json_object(),
    }
}

/// The seven-event Python `_Harness` observation cycle.
///
/// Sequence 1 is a cycle-level state change with no turn id, sequences 2–6 belong to `turn-1`
/// (`Started`, `Paused`, `Resumed`, a final `Output`, `Finished`), and sequence 7 is a
/// cycle-level diagnostic emitted after the turn.
fn python_harness_events() -> Vec<HarnessEvent> {
    let events = vec![
        state_change(1, HarnessState::Idle, HarnessState::Running),
        harness_event(
            2,
            TurnLifecycleEvent {
                kind: TurnEventKind::Started,
                result: None,
            },
        )
        .with_turn_id("turn-1"),
        harness_event(
            3,
            TurnLifecycleEvent {
                kind: TurnEventKind::Paused,
                result: None,
            },
        )
        .with_turn_id("turn-1"),
        harness_event(
            4,
            TurnLifecycleEvent {
                kind: TurnEventKind::Resumed,
                result: None,
            },
        )
        .with_turn_id("turn-1"),
        harness_event(
            5,
            OutputEvent {
                output_id: "output-1".to_owned(),
                kind: OutputKind::Text,
                content: json!("done"),
                operation: OutputOperation::Final,
                channel: OutputChannel::Answer,
                content_index: 0,
                content_type: None,
                data: empty_json_object(),
            },
        )
        .with_turn_id("turn-1"),
        harness_event(
            6,
            TurnLifecycleEvent {
                kind: TurnEventKind::Finished,
                result: Some(completed_result()),
            },
        )
        .with_turn_id("turn-1"),
        harness_event(
            7,
            DiagnosticEvent {
                level: DiagnosticLevel::Info,
                message: "turn complete".to_owned(),
                data: empty_json_object(),
            },
        ),
    ];
    for event in &events {
        event.validate().expect("fixture events are valid");
    }
    events
}

/// A configured checkpoint echoed by [`FakeHarness::export_checkpoint`].
fn checkpoint() -> HarnessCheckpoint {
    HarnessCheckpoint {
        provider: "fake-provider".to_owned(),
        schema_version: "1".to_owned(),
        agent_id: "agent-1".to_owned(),
        host_session_id: "host-1".to_owned(),
        checkpoint_id: "cp-1".to_owned(),
        sequence: 1,
        data: empty_json_object(),
        provider_session_id: Some("ps-1".to_owned()),
        revision: None,
    }
}

/// Shared single-consumer cursor over the remaining events of one queue.
///
/// Implements the Python cursor contract used by the fake harness: `next_event` pops the front
/// (returning `None` when closed or empty), and `close` only flips the lease flag, so it is
/// idempotent and never rewinds or discards already-delivered events.
struct QueueCursor {
    queue: Arc<Mutex<VecDeque<HarnessEvent>>>,
    closed: bool,
}

impl QueueCursor {
    fn new(queue: Arc<Mutex<VecDeque<HarnessEvent>>>) -> Self {
        Self {
            queue,
            closed: false,
        }
    }
}

#[async_trait]
impl HarnessEventCursor for QueueCursor {
    async fn next_event(&mut self) -> HarnessResult<Option<HarnessEvent>> {
        if self.closed {
            return Ok(None);
        }
        Ok(self.queue.lock().expect("event queue poisoned").pop_front())
    }

    async fn close(&mut self) -> HarnessResult<()> {
        self.closed = true;
        Ok(())
    }
}

/// Phase of a finite turn view: before its `Started` event or inside the turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TurnPhase {
    BeforeStart,
    InTurn,
}

/// Finite turn view over the shared queue: skips leading events that do not belong to the turn,
/// yields the turn's ordered events through the matching terminal event inclusive, and stops.
struct TurnCursor {
    queue: Arc<Mutex<VecDeque<HarnessEvent>>>,
    turn_id: String,
    phase: TurnPhase,
    finished: bool,
    closed: bool,
}

impl TurnCursor {
    fn new(queue: Arc<Mutex<VecDeque<HarnessEvent>>>, turn_id: String) -> Self {
        Self {
            queue,
            turn_id,
            phase: TurnPhase::BeforeStart,
            finished: false,
            closed: false,
        }
    }
}

#[async_trait]
impl HarnessEventCursor for TurnCursor {
    async fn next_event(&mut self) -> HarnessResult<Option<HarnessEvent>> {
        if self.closed || self.finished {
            return Ok(None);
        }
        loop {
            let next = self.queue.lock().expect("event queue poisoned").pop_front();
            let Some(event) = next else {
                return Ok(None);
            };
            let belongs_to_turn = event.turn_id.as_deref() == Some(self.turn_id.as_str());
            if self.phase == TurnPhase::BeforeStart {
                let is_start = belongs_to_turn
                    && matches!(
                        &event.event,
                        HarnessEventPayload::TurnLifecycle(payload)
                            if payload.kind == TurnEventKind::Started
                    );
                if !is_start {
                    continue;
                }
                self.phase = TurnPhase::InTurn;
                return Ok(Some(event));
            }
            if !belongs_to_turn {
                // The view is finite: an event from another turn ends it.
                return Ok(None);
            }
            if matches!(
                &event.event,
                HarnessEventPayload::TurnLifecycle(payload) if payload.kind.is_terminal()
            ) {
                self.finished = true;
            }
            return Ok(Some(event));
        }
    }

    async fn close(&mut self) -> HarnessResult<()> {
        self.closed = true;
        Ok(())
    }
}

/// Cursor whose only observation is a provider error (also used for the wrong-turn-id case).
struct ErrorCursor {
    error: Option<HarnessError>,
    closed: bool,
}

impl ErrorCursor {
    fn new(error: HarnessError) -> Self {
        Self {
            error: Some(error),
            closed: false,
        }
    }
}

#[async_trait]
impl HarnessEventCursor for ErrorCursor {
    async fn next_event(&mut self) -> HarnessResult<Option<HarnessEvent>> {
        match self.error.take() {
            Some(error) => Err(error),
            None => Ok(None),
        }
    }

    async fn close(&mut self) -> HarnessResult<()> {
        self.closed = true;
        Ok(())
    }
}

/// Cursor that yields `events` and records its lease release in a shared flag.
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

/// Exhausted cursor whose lease release fails; the adapter surfaces the error as its final item.
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

/// Drains a cursor to exhaustion, asserting each step succeeds.
fn drain_cursor(cursor: &mut dyn HarnessEventCursor) -> Vec<HarnessEvent> {
    let mut events = Vec::new();
    while let Some(event) = block_on(cursor.next_event()).expect("cursor event") {
        events.push(event);
    }
    events
}

/// Returns the id of the next unconsumed turn, without consuming anything.
fn next_unconsumed_turn(queue: &Arc<Mutex<VecDeque<HarnessEvent>>>) -> Option<String> {
    let queue = queue.lock().expect("event queue poisoned");
    queue.iter().find_map(|event| match &event.event {
        HarnessEventPayload::TurnLifecycle(payload) if payload.kind == TurnEventKind::Started => {
            event.turn_id.clone()
        }
        _ => None,
    })
}

/// Test double for the Python `_Harness` fixture.
///
/// All trait methods take `&self`; mutable state lives behind `std::sync::Mutex`, matching the
/// production contract that implementations synchronize internally. The event queue is shared by
/// every cursor, which is what makes `turn_events` and `events` two alternative views over one
/// single-consumer channel.
struct FakeHarness {
    card: HarnessCard,
    state: Mutex<HarnessState>,
    provider_session_id: OnceLock<String>,
    event_buffer: EventBufferConfig,
    queue: Arc<Mutex<VecDeque<HarnessEvent>>>,
    last_receipt: Mutex<Option<SendReceipt>>,
    last_mode: Mutex<Option<DeliveryMode>>,
    checkpoint: Mutex<Option<HarnessCheckpoint>>,
}

impl FakeHarness {
    fn new() -> Self {
        let card = HarnessCard {
            capabilities: BTreeSet::from([
                HarnessCapability::Steer,
                HarnessCapability::GracefulAbort,
                HarnessCapability::ForceAbort,
                HarnessCapability::PauseResume,
                HarnessCapability::Checkpoint,
                HarnessCapability::ModelDiscovery,
                HarnessCapability::ModelSelection,
            ]),
            ..HarnessCard::new("fake-harness", "1.0.0")
        };
        Self {
            card,
            state: Mutex::new(HarnessState::Idle),
            provider_session_id: OnceLock::new(),
            event_buffer: EventBufferConfig::default(),
            queue: Arc::new(Mutex::new(VecDeque::new())),
            last_receipt: Mutex::new(None),
            last_mode: Mutex::new(None),
            checkpoint: Mutex::new(None),
        }
    }

    fn with_checkpoint(self, checkpoint: HarnessCheckpoint) -> Self {
        *self.checkpoint.lock().expect("checkpoint poisoned") = Some(checkpoint);
        self
    }

    fn enqueue(&self, events: Vec<HarnessEvent>) {
        self.queue
            .lock()
            .expect("event queue poisoned")
            .extend(events);
    }

    fn last_receipt(&self) -> Option<SendReceipt> {
        self.last_receipt.lock().expect("receipt poisoned").clone()
    }

    fn last_delivery_mode(&self) -> Option<DeliveryMode> {
        *self.last_mode.lock().expect("mode poisoned")
    }
}

#[async_trait]
impl HarnessProtocol for FakeHarness {
    fn card(&self) -> &HarnessCard {
        &self.card
    }

    fn state(&self) -> HarnessState {
        *self.state.lock().expect("state poisoned")
    }

    fn provider_session_id(&self) -> Option<&str> {
        self.provider_session_id.get().map(String::as_str)
    }

    fn event_buffer_config(&self) -> &EventBufferConfig {
        &self.event_buffer
    }

    async fn start(&self, context: HarnessContext) -> HarnessResult<()> {
        // Python parity: context identity is validated first, then host negotiation.
        context.validate()?;
        self.card
            .validate_host(&context.protocol_version, &context.host_capabilities)?;
        let _ = self.provider_session_id.set("ps-1".to_owned());
        *self.state.lock().expect("state poisoned") = HarnessState::Idle;
        Ok(())
    }

    async fn stop(&self) -> HarnessResult<()> {
        // Idempotent: stopping a terminated harness is a no-op.
        *self.state.lock().expect("state poisoned") = HarnessState::Terminated;
        Ok(())
    }

    fn events(&self) -> Box<dyn HarnessEventCursor> {
        Box::new(QueueCursor::new(Arc::clone(&self.queue)))
    }

    fn turn_events(&self, turn_id: Option<&str>) -> Box<dyn HarnessEventCursor> {
        // `turn_id` validates the next unconsumed turn; it is not an out-of-order selector.
        let next = next_unconsumed_turn(&self.queue);
        let target = match turn_id {
            Some(requested) if next.as_deref() != Some(requested) => {
                return Box::new(ErrorCursor::new(HarnessError::state(format!(
                    "turn '{requested}' is not the next unconsumed turn"
                ))));
            }
            Some(requested) => Some(requested.to_owned()),
            None => next,
        };
        match target {
            Some(target) => Box::new(TurnCursor::new(Arc::clone(&self.queue), target)),
            None => Box::new(QueueCursor::new(Arc::clone(&self.queue))),
        }
    }

    async fn send(&self, _input: HarnessInput, mode: DeliveryMode) -> HarnessResult<SendReceipt> {
        let receipt = SendReceipt {
            message_id: "msg-1".to_owned(),
            turn_id: "turn-1".to_owned(),
            accepted_mode: mode,
        };
        receipt.validate()?;
        *self.last_mode.lock().expect("mode poisoned") = Some(mode);
        *self.last_receipt.lock().expect("receipt poisoned") = Some(receipt.clone());
        Ok(receipt)
    }

    async fn abort(&self, _mode: AbortMode) -> HarnessResult<()> {
        let mut state = self.state.lock().expect("state poisoned");
        if *state == HarnessState::Terminated {
            return Err(HarnessError::state("harness is terminated"));
        }
        *state = HarnessState::Idle;
        Ok(())
    }

    async fn pause(&self) -> HarnessResult<()> {
        let mut state = self.state.lock().expect("state poisoned");
        if *state == HarnessState::Terminated {
            return Err(HarnessError::state("harness is terminated"));
        }
        *state = HarnessState::Paused;
        Ok(())
    }

    async fn resume(&self, _query: Option<HarnessInput>) -> HarnessResult<()> {
        let mut state = self.state.lock().expect("state poisoned");
        if *state == HarnessState::Terminated {
            return Err(HarnessError::state("harness is terminated"));
        }
        *state = HarnessState::Running;
        Ok(())
    }

    async fn export_checkpoint(&self) -> HarnessResult<Option<HarnessCheckpoint>> {
        Ok(self.checkpoint.lock().expect("checkpoint poisoned").clone())
    }
}

/// A fresh fake harness preloaded with the seven-event Python `_Harness` cycle.
fn fake_harness_with_python_events() -> FakeHarness {
    let harness = FakeHarness::new();
    harness.enqueue(python_harness_events());
    harness
}

#[test]
fn python_harness_turn_view_is_terminal_event_inclusive() {
    // Python: tests/unit_tests/harness_protocol/test_protocol.py (turn-view `[2, 3, 4, 5, 6]`)
    let harness = fake_harness_with_python_events();
    let events = drain_cursor(harness.turn_events(Some("turn-1")).as_mut());

    let sequences: Vec<u64> = events.iter().map(|event| event.sequence).collect();
    assert_eq!(
        sequences,
        vec![2, 3, 4, 5, 6],
        "the turn view starts at Started and includes every event through the terminal one"
    );
    assert_eq!(
        events.len(),
        5,
        "Paused and Resumed do not end the view, and Finished is included"
    );

    assert!(
        matches!(
            &events.last().expect("terminal event").event,
            HarnessEventPayload::TurnLifecycle(TurnLifecycleEvent {
                kind: TurnEventKind::Finished,
                ..
            })
        ),
        "the view ends on the Finished payload"
    );

    let outputs: Vec<&OutputEvent> = events
        .iter()
        .filter_map(|event| match &event.event {
            HarnessEventPayload::Output(output) => Some(output),
            _ => None,
        })
        .collect();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].operation, OutputOperation::Final);
    assert_eq!(outputs[0].content, json!("done"));
}

#[test]
fn python_harness_cycle_view_continues_after_the_turn_view() {
    let harness = fake_harness_with_python_events();
    let mut cursor = harness.turn_events(Some("turn-1"));

    let turn_events = drain_cursor(cursor.as_mut());
    let sequences: Vec<u64> = turn_events.iter().map(|event| event.sequence).collect();
    assert_eq!(sequences, vec![2, 3, 4, 5, 6]);

    // A fresh cycle view reads the same single-consumer channel from where the turn view stopped.
    let remaining = drain_cursor(harness.events().as_mut());
    let remaining_sequences: Vec<u64> = remaining.iter().map(|event| event.sequence).collect();
    assert_eq!(
        remaining_sequences,
        vec![7],
        "events() must continue at sequence 7 and never rewind to 1"
    );
    assert!(matches!(
        &remaining[0].event,
        HarnessEventPayload::Diagnostic(DiagnosticEvent {
            level: DiagnosticLevel::Info,
            ..
        })
    ));

    // The view is already exhausted; close is idempotent.
    block_on(cursor.close()).expect("first close");
    block_on(cursor.close()).expect("second close");
}

#[test]
fn python_harness_turn_view_rejects_a_non_next_turn() {
    // Python: `turn_id` identifies the next unconsumed turn instead of selecting out of order.
    let harness = fake_harness_with_python_events();
    let mut cursor = harness.turn_events(Some("other"));
    let error = block_on(cursor.next_event()).expect_err("turn id must match the next turn");
    assert_eq!(error.kind(), HarnessErrorKind::State);
    assert_eq!(
        error.message(),
        "turn 'other' is not the next unconsumed turn"
    );
    assert!(block_on(cursor.next_event())
        .expect("error ends the view")
        .is_none());
}

#[test]
fn send_auto_records_auto_delivery() {
    // Python: `send(..., mode=DeliveryMode.AUTO)` default.
    let harness = FakeHarness::new();
    let receipt = block_on(harness.send_auto(HarnessInput::text("hi"))).expect("accepted");
    assert_eq!(receipt.accepted_mode, DeliveryMode::Auto);
    assert_eq!(receipt.message_id, "msg-1");
    assert_eq!(receipt.turn_id, "turn-1");
    assert_eq!(harness.last_delivery_mode(), Some(DeliveryMode::Auto));
    assert_eq!(harness.last_receipt(), Some(receipt));
}

#[test]
fn fake_harness_lifecycle_through_a_trait_object() {
    // Python: tests/unit_tests/harness_protocol/test_protocol.py (fake harness start/stop/send)
    let expected_checkpoint = checkpoint();
    let harness: DynHarness =
        Arc::new(FakeHarness::new().with_checkpoint(expected_checkpoint.clone()));

    assert_eq!(harness.card().name, "fake-harness");
    assert_eq!(harness.state(), HarnessState::Idle);
    assert_eq!(harness.provider_session_id(), None);

    let context = HarnessContext::new("agent-a", "agent-1", "host-1", "be helpful");
    block_on(harness.start(context)).expect("compatible host starts");
    assert_eq!(harness.state(), HarnessState::Idle);
    assert_eq!(harness.provider_session_id(), Some("ps-1"));
    assert_eq!(harness.event_buffer_config().capacity, 1024);

    // Pause/resume/abort adjust the lifecycle state.
    block_on(harness.pause()).expect("paused");
    assert_eq!(harness.state(), HarnessState::Paused);
    block_on(harness.resume(None)).expect("resumed");
    assert_eq!(harness.state(), HarnessState::Running);
    block_on(harness.abort(AbortMode::Graceful)).expect("aborted");
    assert_eq!(harness.state(), HarnessState::Idle);

    let receipt =
        block_on(harness.send(HarnessInput::text("hello"), DeliveryMode::Steer)).expect("accepted");
    assert_eq!(receipt.accepted_mode, DeliveryMode::Steer);
    let receipt = block_on(harness.send_auto(HarnessInput::text("again"))).expect("accepted");
    assert_eq!(receipt.accepted_mode, DeliveryMode::Auto);

    let exported = block_on(harness.export_checkpoint())
        .expect("checkpoint exported")
        .expect("configured checkpoint");
    assert_eq!(exported, expected_checkpoint);

    block_on(harness.stop()).expect("first stop");
    assert_eq!(harness.state(), HarnessState::Terminated);
    block_on(harness.stop()).expect("second stop is idempotent");
    assert_eq!(harness.state(), HarnessState::Terminated);
}

#[test]
fn start_rejects_an_incompatible_host_protocol_version() {
    let harness = FakeHarness::new();
    let context = HarnessContext::new("agent-a", "agent-1", "host-1", "be helpful")
        .with_protocol_version("9.9");
    let error = block_on(harness.start(context)).expect_err("version is not supported");
    assert_eq!(error.kind(), HarnessErrorKind::Protocol);
    assert!(
        error.message().contains("is not supported"),
        "unexpected message: {}",
        error.message()
    );
    assert_eq!(harness.provider_session_id(), None);
}

#[test]
fn vec_event_cursor_is_fifo_and_reports_remaining() {
    let events = vec![
        state_change(1, HarnessState::Idle, HarnessState::Running),
        state_change(2, HarnessState::Running, HarnessState::Idle),
    ];
    let mut cursor = VecEventCursor::new(events);
    assert_eq!(cursor.remaining(), 2);
    assert!(!cursor.is_closed());

    let first = block_on(cursor.next_event())
        .expect("first")
        .expect("event");
    assert_eq!(first.sequence, 1);
    assert_eq!(cursor.remaining(), 1);

    let second = block_on(cursor.next_event())
        .expect("second")
        .expect("event");
    assert_eq!(second.sequence, 2);
    assert_eq!(cursor.remaining(), 0);
    assert!(block_on(cursor.next_event()).expect("exhausted").is_none());
}

#[test]
fn vec_event_cursor_close_is_idempotent_and_never_rewinds() {
    let mut cursor = VecEventCursor::new(vec![
        state_change(1, HarnessState::Idle, HarnessState::Running),
        state_change(2, HarnessState::Running, HarnessState::Idle),
    ]);
    block_on(cursor.next_event())
        .expect("first")
        .expect("event");
    assert_eq!(cursor.remaining(), 1);

    block_on(cursor.close()).expect("first close");
    assert!(cursor.is_closed());
    assert_eq!(cursor.remaining(), 1, "closing does not rewind");

    block_on(cursor.close()).expect("second close");
    assert!(cursor.is_closed());
    assert_eq!(cursor.remaining(), 1);
    assert!(block_on(cursor.next_event()).expect("closed").is_none());
    assert_eq!(cursor.remaining(), 1);
}

#[test]
fn vec_event_cursor_is_empty_when_built_without_events() {
    let mut cursor = VecEventCursor::new(Vec::new());
    assert_eq!(cursor.remaining(), 0);
    assert!(!cursor.is_closed());
    assert!(block_on(cursor.next_event()).expect("empty").is_none());
    block_on(cursor.close()).expect("close");
    assert!(cursor.is_closed());
}

#[test]
fn event_cursor_stream_preserves_order_and_closes_on_exhaustion() {
    let closed = Arc::new(AtomicBool::new(false));
    let events = vec![
        state_change(1, HarnessState::Idle, HarnessState::Running),
        state_change(2, HarnessState::Running, HarnessState::Idle),
        state_change(3, HarnessState::Idle, HarnessState::Running),
    ];
    let cursor = RecordingCursor {
        events: VecDeque::from(events),
        closed: Arc::clone(&closed),
    };
    assert!(!closed.load(Ordering::SeqCst));

    let stream = event_cursor_stream(Box::new(cursor));
    let items = block_on(stream.collect::<Vec<_>>());
    let sequences: Vec<u64> = items
        .into_iter()
        .map(|item| item.expect("event").sequence)
        .collect();
    assert_eq!(sequences, vec![1, 2, 3]);
    assert!(
        closed.load(Ordering::SeqCst),
        "exhaustion closes the cursor"
    );
}

#[test]
fn event_cursor_stream_over_vec_cursor_yields_all_then_none() {
    let mut stream = event_cursor_stream(Box::new(VecEventCursor::new(vec![
        state_change(1, HarnessState::Idle, HarnessState::Running),
        state_change(2, HarnessState::Running, HarnessState::Idle),
    ])));

    assert_eq!(
        block_on(stream.next())
            .expect("event")
            .expect("event")
            .sequence,
        1
    );
    assert_eq!(
        block_on(stream.next())
            .expect("event")
            .expect("event")
            .sequence,
        2
    );
    assert!(
        block_on(stream.next()).is_none(),
        "stream ends after the cursor"
    );
    assert!(
        block_on(stream.next()).is_none(),
        "an ended stream stays ended"
    );
}

#[test]
fn event_cursor_stream_surfaces_a_cursor_error_as_the_last_item() {
    let mut stream = event_cursor_stream(Box::new(ErrorCursor::new(HarnessError::state("boom"))));

    let first = block_on(stream.next()).expect("error item");
    let error = first.expect_err("the error is the only item");
    assert_eq!(error.kind(), HarnessErrorKind::State);
    assert_eq!(error.message(), "boom");
    assert!(
        block_on(stream.next()).is_none(),
        "no item follows the error"
    );
    assert!(
        block_on(stream.next()).is_none(),
        "subsequent polls stay None"
    );
}

#[test]
fn event_cursor_stream_surfaces_a_close_error_after_exhaustion() {
    let mut stream = event_cursor_stream(Box::new(CloseFailingCursor));

    let first = block_on(stream.next()).expect("close error item");
    let error = first.expect_err("the lease release failure is surfaced");
    assert_eq!(error.kind(), HarnessErrorKind::State);
    assert_eq!(error.message(), "lease release failed");
    assert!(block_on(stream.next()).is_none(), "the error is final");
}

/// Fake model control returning one catalog entry and recording the applied selection.
struct FakeModelControl {
    models: Vec<ModelOption>,
    last_selection: Mutex<Option<ModelSelection>>,
}

impl FakeModelControl {
    fn new() -> Self {
        Self {
            models: vec![ModelOption {
                model_id: "fake-model".to_owned(),
                display_name: "Fake Model".to_owned(),
                description: String::new(),
                efforts: vec!["low".to_owned(), "high".to_owned()],
                default_effort: Some("low".to_owned()),
                is_default: true,
                extensions: empty_json_object(),
            }],
            last_selection: Mutex::new(None),
        }
    }

    fn last_selection(&self) -> Option<ModelSelection> {
        self.last_selection
            .lock()
            .expect("selection poisoned")
            .clone()
    }
}

#[async_trait]
impl HarnessModelControl for FakeModelControl {
    async fn list_models(&self) -> HarnessResult<Vec<ModelOption>> {
        Ok(self.models.clone())
    }

    async fn set_model(&self, selection: ModelSelection) -> HarnessResult<()> {
        selection.validate()?;
        *self.last_selection.lock().expect("selection poisoned") = Some(selection);
        Ok(())
    }
}

/// Fake provider that constructs an unstarted [`FakeHarness`].
struct FakeProvider {
    card: HarnessCard,
}

impl FakeProvider {
    fn new() -> Self {
        Self {
            card: HarnessCard::new("fake-provider", "1.0.0"),
        }
    }
}

impl HarnessProvider for FakeProvider {
    fn card(&self) -> &HarnessCard {
        &self.card
    }

    fn create(&self, _config: JsonObject) -> HarnessResult<Box<dyn HarnessProtocol>> {
        // Unstarted by contract: no provider session id and no observations yet; `start` is a
        // separate call.
        Ok(Box::new(FakeHarness::new()))
    }
}

/// Fake model-config mapper that records whether a fallback choice arrived.
struct FakeModelConfigProvider {
    fallback_seen: Mutex<Option<String>>,
}

impl FakeModelConfigProvider {
    fn new() -> Self {
        Self {
            fallback_seen: Mutex::new(None),
        }
    }

    fn fallback_seen(&self) -> Option<String> {
        self.fallback_seen
            .lock()
            .expect("fallback poisoned")
            .clone()
    }
}

impl ModelConfigProvider for FakeModelConfigProvider {
    fn model_config(
        &self,
        choice: &ModelChoice,
        fallback: Option<&ModelChoice>,
    ) -> HarnessResult<JsonObject> {
        *self.fallback_seen.lock().expect("fallback poisoned") =
            fallback.and_then(|choice| choice.model.clone());
        let mut config = empty_json_object();
        if let Some(model) = &choice.model {
            config.insert("model".to_owned(), json!(model));
        }
        Ok(config)
    }
}

#[test]
fn trait_objects_cover_the_protocol_spi() {
    // Compile-time `isinstance` equivalents: every SPI handle must work as a trait object.
    let harness: DynHarness = Arc::new(FakeHarness::new());
    assert_eq!(harness.card().name, "fake-harness");

    let fake_control = Arc::new(FakeModelControl::new());
    let control: Arc<dyn HarnessModelControl> = fake_control.clone();
    let models = block_on(control.list_models()).expect("catalog");
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].model_id, "fake-model");
    let selection = ModelSelection {
        model: Some("fake-model".to_owned()),
        effort: Some("high".to_owned()),
    };
    block_on(control.set_model(selection.clone())).expect("selection applied");
    assert_eq!(fake_control.last_selection(), Some(selection));

    let provider: Box<dyn HarnessProvider> = Box::new(FakeProvider::new());
    assert_eq!(provider.card().name, "fake-provider");
    let created = provider
        .create(empty_json_object())
        .expect("provider creates a harness");
    assert_eq!(
        created.state(),
        HarnessState::Idle,
        "a created harness is unstarted"
    );
    assert_eq!(created.provider_session_id(), None);
    assert_eq!(created.card().name, "fake-harness");

    let fake_config = Arc::new(FakeModelConfigProvider::new());
    let config_provider: Arc<dyn ModelConfigProvider> = fake_config.clone();
    let choice = ModelChoice {
        model: Some("fake-model".to_owned()),
        effort: None,
        endpoint: None,
    };
    let fallback = ModelChoice {
        model: Some("fake-fallback".to_owned()),
        effort: None,
        endpoint: None,
    };
    let config = config_provider
        .model_config(&choice, Some(&fallback))
        .expect("configuration fragment");
    assert_eq!(config.get("model"), Some(&json!("fake-model")));
    assert_eq!(
        fake_config.fallback_seen(),
        Some("fake-fallback".to_owned()),
        "the fallback choice reaches the provider"
    );

    let config = config_provider
        .model_config(&choice, None)
        .expect("no fallback is valid");
    assert_eq!(config.get("model"), Some(&json!("fake-model")));
    assert_eq!(fake_config.fallback_seen(), None);

    // The alias forms remain usable as shared pointers.
    let _: DynModelControl = Arc::new(FakeModelControl::new());
    let _: DynHarnessProvider = Arc::new(FakeProvider::new());
}

/// Compile-time public-surface check: every path listed here must resolve at the crate root.
///
/// Semicolon separators keep the list compact; rustfmt leaves the macro body untouched while the
/// compiler still resolves every type path.
macro_rules! public_types {
    ($($ty:ty);* $(;)?) => {
        $(let _: Option<$ty> = None;)*
    };
}

#[test]
#[allow(clippy::too_many_lines)]
fn public_surface_matches_design_section_eight() {
    // DESIGN §8: a missing root re-export must break this test's build.
    // Constants.
    assert_eq!(PROTOCOL_VERSION, "1.0");
    assert_eq!(EVENT_WIRE_SCHEMA_VERSION, "1");
    assert_eq!(MAX_CHECKPOINT_BYTES, 4 * 1024 * 1024);
    assert_eq!(AUTH_FALLBACK_REQUEST_TYPE, "auth_fallback");
    assert_eq!(AUTH_FALLBACK_EVENT_TYPE, "auth_fallback_activated");
    assert_eq!(TERMINAL_TURN_EVENT_KINDS.len(), 3);

    // Functions, as function-item references with their DESIGN signatures.
    let to_dict: fn(&HarnessEvent) -> HarnessResult<JsonObject> = harness_event_to_dict;
    let from_dict: fn(JsonObject) -> HarnessResult<HarnessEvent> = harness_event_from_dict;
    let to_json: fn(&HarnessEvent) -> HarnessResult<String> = harness_event_to_json_string;
    let from_json: fn(&str) -> HarnessResult<HarnessEvent> = harness_event_from_json_str;
    let retention: fn(&HarnessEventPayload) -> EventRetention = event_retention;
    let validate_response: fn(
        &HarnessInteractionRequest,
        HarnessInteractionResponse,
    ) -> HarnessResult<HarnessInteractionResponse> = validate_interaction_response;
    let object_from: fn(JsonValue) -> Option<JsonObject> = json_object_from_value;
    let object_to: fn(&JsonObject) -> JsonValue = json_object_to_value;
    let versions: fn() -> BTreeSet<String> = default_compatible_protocol_versions;
    let cursor_stream: fn(Box<dyn HarnessEventCursor>) -> HarnessEventStream = event_cursor_stream;

    assert_eq!(versions(), default_compatible_protocol_versions());
    let event = state_change(1, HarnessState::Idle, HarnessState::Running);
    let wire = to_dict(&event).expect("encodes");
    assert_eq!(from_dict(wire).expect("decodes"), event);
    let text = to_json(&event).expect("encodes text");
    assert_eq!(from_json(&text).expect("decodes text"), event);
    assert_eq!(retention(&event.event), EventRetention::Required);
    let object = object_from(json!({"model": "m"})).expect("object");
    assert_eq!(object_to(&object), json!({"model": "m"}));
    let _ = validate_response;
    let mut stream = cursor_stream(Box::new(VecEventCursor::new(vec![event])));
    assert_eq!(
        block_on(stream.next())
            .expect("event")
            .expect("event")
            .sequence,
        1
    );

    // Aliases.
    public_types!(
        JsonValue; JsonObject; HarnessResult<()>; HarnessErrorKind; HarnessEventStream;
        DynHarness; DynHarnessProvider; DynModelControl;
    );

    // Traits (object-safe, so a boxed handle proves the export).
    public_types!(
        Box<dyn Validate>; Box<dyn HarnessTelemetry>; Box<dyn HarnessCheckpointSink>;
        Box<dyn HarnessHookDispatcher>; Box<dyn HarnessInteractionHandler>; Box<dyn ToolGateway>;
        Box<dyn HarnessEventCursor>; Box<dyn HarnessProtocol>; Box<dyn HarnessModelControl>;
        Box<dyn HarnessProvider>; Box<dyn ModelConfigProvider>;
    );

    // Every §5 type.
    public_types!(
        HarnessError; HarnessState; TurnStatus; MessageRole; TurnTerminationKind; ContentBlock;
        TurnMessage; TurnTermination; MonetaryAmount; TurnUsage; TurnError; TurnResult;
        HarnessCapability; HostCapability; HarnessCard; ResumePolicy; HarnessContext;
        DeliveryMode; AbortMode; HarnessInput; SendReceipt; ModelSelection; ModelEndpoint;
        ModelChoice; ModelOption; TurnEventKind; OutputKind; OutputChannel; OutputOperation;
        UsageUpdateMode; EventRetention; EventOverflowPolicy; ItemEventKind; HookEventPhase;
        DiagnosticLevel; ModelRequestStatus; OutputEvent; ItemLifecycleEvent; UsageUpdatedEvent;
        ModelRequestEvent; StateChangedEvent; TurnLifecycleEvent; HookObservedEvent;
        DiagnosticEvent; ProviderEvent; UnknownEvent; HarnessEventPayload; HarnessEvent;
        EventBufferConfig; CheckpointReason; HarnessCheckpoint; CheckpointSaveReceipt;
        ToolDecisionKind; BeforePromptContext; BeforePromptResult; BeforeToolContext; ToolDecision;
        AfterToolContext; AfterToolResult; StopHookContext; InteractionResponseStatus;
        ToolApprovalDecision; InteractionCancelReason; ToolApprovalRequest; UserInputRequest;
        McpElicitationRequest; DynamicToolCallRequest; ProviderInteractionRequest;
        ToolApprovalResponse; UserInputResponse; McpElicitationResponse; DynamicToolCallResponse;
        ProviderInteractionResponse; HarnessInteractionRequest; HarnessInteractionResponse;
        ToolDefinition; ToolInvocation; ToolExecutionResult; McpTransport; McpInstance;
        McpServerConfig; EventCursorStream; VecEventCursor;
    );
}
