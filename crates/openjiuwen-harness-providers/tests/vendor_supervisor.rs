#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::needless_pass_by_value,
    clippy::similar_names
)]
//! Behavioural suite of the generic vendor harness shell.
//!
//! Every case drives a [`VendorHarness`] over the in-file [`FakeDriver`], which implements
//! [`VendorDriver`] with a mode switch (`Answer`, `Fail`, `Error`, `Hang`) and records every
//! command the shell forwards to it. The shell is exercised end to end: capability gating, the
//! per-turn event sequence, the serialized turn queue, steering, abort, pause/resume, checkpoint
//! export, driver failure mapping, the observation lease and teardown.
//!
//! No network, no vendor CLI, no `tokio::time::sleep`: ordering that matters is observed through
//! the cycle cursor, and every await that could hang is wrapped in a five-second timeout as a pure
//! safety net. Python has no supervisor: this shell replaces the per-vendor lifecycle code the three
//! adapters each duplicate.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_harness_protocol::{
    AbortMode, DeliveryMode, EventBufferConfig, HarnessCapability, HarnessCard, HarnessCheckpoint,
    HarnessContext, HarnessError, HarnessErrorKind, HarnessEvent, HarnessEventCursor,
    HarnessEventPayload, HarnessInput, HarnessProtocol, HarnessResult, HarnessState,
    HostCapability, ItemEventKind, JsonObject, JsonValue, OutputChannel, OutputEvent, OutputKind,
    OutputOperation, TurnEventKind, TurnResult, TurnStatus, TurnTerminationKind, TurnUsage,
};
use openjiuwen_harness_providers::vendor_cli::{
    completed_turn, failed_turn, interrupted_turn, turn_error, TurnContext, TurnRequest,
    VendorDriver, VendorHarness,
};
use serde_json::json;

/// Timeout applied to every await that must not hang the suite.
const BOUND: Duration = Duration::from_secs(5);

/// Timeout used only to prove that the harness publishes nothing further.
const QUIET: Duration = Duration::from_millis(100);

/// Locks `mutex`, recovering the guard when a previous holder panicked.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Behaviour the fake driver reports for the next turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Publish one output, one tool-call pair and one usage update, then complete.
    Answer,
    /// Return a normalized failed result.
    Fail,
    /// Return a transport error, which the shell must map to a failed turn.
    Error,
    /// Park until an interrupt or a close releases the turn, then report an interruption.
    Hang,
}

/// Everything the fake driver observed across the suite.
#[derive(Default)]
struct Captures {
    /// Steering payloads, keyed by the turn the shell named.
    steers: Vec<(String, JsonValue)>,
    /// Interrupts, keyed by the turn the driver was running when the shell called it.
    interrupts: Vec<(Option<String>, AbortMode)>,
    /// Number of pause calls.
    pauses: usize,
    /// Resume queries in call order.
    resumes: Vec<Option<HarnessInput>>,
    /// Number of `open` calls.
    opens: usize,
    /// Number of `close` calls.
    closes: usize,
    /// Number of `export_checkpoint` calls.
    checkpoints: usize,
}

/// Deterministic [`VendorDriver`] test double.
struct FakeDriver {
    /// Behaviour of the next turn.
    mode: Mutex<Mode>,
    /// Recorded commands.
    captures: Mutex<Captures>,
    /// Releases a parked [`Mode::Hang`] turn.
    gate: tokio::sync::Notify,
    /// Releases a parked [`open`](VendorDriver::open) call.
    open_gate: tokio::sync::Notify,
    /// Whether `open` parks until [`open_gate`](Self::open_gate) is notified.
    hang_open: std::sync::atomic::AtomicBool,
    /// Provider session id `open` reports.
    open_session: Mutex<String>,
    /// Turn the driver believes it is running.
    active_turn: Mutex<Option<String>>,
    /// Number of turns started.
    turns: AtomicUsize,
    /// Number of turns currently inside `run_turn`.
    running: AtomicUsize,
    /// High-water mark of [`running`](Self::running).
    max_running: AtomicUsize,
}

impl FakeDriver {
    /// Builds a driver in `mode`.
    fn new(mode: Mode) -> Arc<Self> {
        Arc::new(Self {
            mode: Mutex::new(mode),
            captures: Mutex::new(Captures::default()),
            gate: tokio::sync::Notify::new(),
            open_gate: tokio::sync::Notify::new(),
            hang_open: std::sync::atomic::AtomicBool::new(false),
            open_session: Mutex::new("fake-session".to_owned()),
            active_turn: Mutex::new(None),
            turns: AtomicUsize::new(0),
            running: AtomicUsize::new(0),
            max_running: AtomicUsize::new(0),
        })
    }

    /// Makes `open` park (`true`) until the flag is cleared again, or answer immediately (`false`).
    fn hang_open(&self, hang: bool) {
        self.hang_open.store(hang, Ordering::SeqCst);
        if !hang {
            self.open_gate.notify_waiters();
        }
    }

    /// Sets the provider session id `open` reports.
    fn set_open_session(&self, session: &str) {
        let mut guard = lock(&self.open_session);
        session.clone_into(&mut guard);
    }

    /// Replaces the behaviour of the next turn.
    fn set_mode(&self, mode: Mode) {
        *lock(&self.mode) = mode;
    }

    /// Returns the recorded commands.
    fn captures(&self) -> MutexGuard<'_, Captures> {
        lock(&self.captures)
    }

    /// Returns the number of turns that reached `run_turn`.
    fn turns(&self) -> usize {
        self.turns.load(Ordering::SeqCst)
    }

    /// Returns the highest number of turns that ever ran concurrently.
    fn max_running(&self) -> usize {
        self.max_running.load(Ordering::SeqCst)
    }

    /// Returns the recorded interrupts.
    fn interrupts(&self) -> Vec<(Option<String>, AbortMode)> {
        self.captures().interrupts.clone()
    }

    /// Runs `mode`'s body and records the concurrency window.
    async fn run_body(
        &self,
        turn: &TurnRequest,
        session: &TurnContext,
    ) -> HarnessResult<TurnResult> {
        let mode = *lock(&self.mode);
        match mode {
            Mode::Answer => {
                session
                    .output(
                        &turn.turn_id,
                        OutputEvent {
                            output_id: "out-1".to_owned(),
                            kind: OutputKind::Text,
                            content: JsonValue::String(turn.text()),
                            operation: OutputOperation::Snapshot,
                            channel: OutputChannel::Answer,
                            content_index: 0,
                            content_type: None,
                            data: JsonObject::new(),
                        },
                    )
                    .await;
                session
                    .tool_started(
                        &turn.turn_id,
                        "item-1",
                        Some("call-1"),
                        "Read",
                        json!({"path": "a.rs"}),
                    )
                    .await;
                session
                    .tool_completed(
                        &turn.turn_id,
                        "item-1",
                        Some("call-1"),
                        "Read",
                        json!("contents"),
                        Some("contents".to_owned()),
                    )
                    .await;
                let usage = TurnUsage {
                    input_tokens: Some(3),
                    output_tokens: Some(2),
                    ..TurnUsage::default()
                };
                session.usage(&turn.turn_id, usage.clone()).await;
                Ok(completed_turn(
                    JsonValue::String(turn.text()),
                    Vec::new(),
                    Some(usage),
                ))
            }
            Mode::Fail => Ok(failed_turn(turn_error(
                "the model refused the turn",
                "refused",
                "provider",
                false,
            ))),
            Mode::Error => Err(HarnessError::protocol("the fake CLI crashed")),
            Mode::Hang => {
                self.gate.notified().await;
                Ok(interrupted_turn(
                    TurnTerminationKind::HarnessStop,
                    "the fake turn was interrupted",
                ))
            }
        }
    }
}

#[async_trait]
impl VendorDriver for FakeDriver {
    fn label(&self) -> &'static str {
        "Fake"
    }

    async fn open(
        &self,
        _context: &HarnessContext,
        _session: &TurnContext,
    ) -> HarnessResult<Option<String>> {
        self.captures().opens += 1;
        while self.hang_open.load(Ordering::SeqCst) {
            self.open_gate.notified().await;
        }
        Ok(Some(lock(&self.open_session).clone()))
    }

    async fn close(&self) -> HarnessResult<()> {
        self.captures().closes += 1;
        self.gate.notify_one();
        Ok(())
    }

    async fn run_turn(
        &self,
        turn: &TurnRequest,
        session: &TurnContext,
    ) -> HarnessResult<TurnResult> {
        self.turns.fetch_add(1, Ordering::SeqCst);
        let running = self.running.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_running.fetch_max(running, Ordering::SeqCst);
        *lock(&self.active_turn) = Some(turn.turn_id.clone());
        let result = self.run_body(turn, session).await;
        self.running.fetch_sub(1, Ordering::SeqCst);
        result
    }

    async fn steer(
        &self,
        turn: &TurnRequest,
        content: &JsonValue,
        _session: &TurnContext,
    ) -> HarnessResult<()> {
        self.captures()
            .steers
            .push((turn.turn_id.clone(), content.clone()));
        Ok(())
    }

    async fn interrupt(&self, mode: AbortMode, _session: &TurnContext) -> HarnessResult<()> {
        let active = lock(&self.active_turn).clone();
        self.captures().interrupts.push((active, mode));
        self.gate.notify_one();
        Ok(())
    }

    async fn pause(&self, _session: &TurnContext) -> HarnessResult<()> {
        self.captures().pauses += 1;
        Ok(())
    }

    async fn resume(
        &self,
        query: Option<&HarnessInput>,
        _session: &TurnContext,
    ) -> HarnessResult<()> {
        self.captures().resumes.push(query.cloned());
        Ok(())
    }

    async fn export_checkpoint(&self) -> HarnessResult<Option<HarnessCheckpoint>> {
        self.captures().checkpoints += 1;
        Ok(Some(fake_checkpoint()))
    }
}

/// Returns the checkpoint the fake driver exports.
fn fake_checkpoint() -> HarnessCheckpoint {
    HarnessCheckpoint {
        provider: "fake".to_owned(),
        schema_version: "1".to_owned(),
        agent_id: "agent-1".to_owned(),
        host_session_id: "host-session".to_owned(),
        checkpoint_id: "cp-1".to_owned(),
        sequence: 1,
        data: JsonObject::new(),
        provider_session_id: Some("fake-session".to_owned()),
        revision: None,
    }
}

/// Builds a card declaring exactly `capabilities`.
fn card(capabilities: &[HarnessCapability]) -> HarnessCard {
    let mut card = HarnessCard::new("fake-harness", "0.1.0");
    card.capabilities = capabilities.iter().copied().collect();
    card
}

/// Builds a card declaring every capability the shell gates on.
fn full_card() -> HarnessCard {
    card(&[
        HarnessCapability::Steer,
        HarnessCapability::GracefulAbort,
        HarnessCapability::ForceAbort,
        HarnessCapability::PauseResume,
        HarnessCapability::Checkpoint,
    ])
}

/// Wraps `driver` in an unstarted shell with `card`.
fn harness(card: HarnessCard, driver: &Arc<FakeDriver>) -> VendorHarness {
    VendorHarness::new(
        card,
        EventBufferConfig::default(),
        Arc::clone(driver) as Arc<dyn VendorDriver>,
    )
}

/// Returns the host context every case starts from.
fn context() -> HarnessContext {
    HarnessContext::new("agent", "agent-1", "host-session", "system prompt")
}

/// Reads exactly `count` events from `cursor`, failing when they do not arrive in time.
async fn take(cursor: &mut dyn HarnessEventCursor, count: usize) -> Vec<HarnessEvent> {
    tokio::time::timeout(BOUND, async {
        let mut events = Vec::new();
        while events.len() < count {
            match cursor
                .next_event()
                .await
                .expect("the cycle cursor is healthy")
            {
                Some(event) => events.push(event),
                None => break,
            }
        }
        events
    })
    .await
    .expect("the harness must publish the expected events in time")
}

/// Starts `harness` and consumes its `Terminated → Idle` transition.
async fn started(harness: &VendorHarness) -> Box<dyn HarnessEventCursor> {
    harness
        .start(context())
        .await
        .expect("the host is compatible");
    let mut cursor = harness.events();
    let _ = take(cursor.as_mut(), 1).await;
    cursor
}

/// Returns the turn lifecycle kinds of `events`, in order.
fn turn_kinds(events: &[HarnessEvent]) -> Vec<TurnEventKind> {
    events
        .iter()
        .filter_map(|event| match &event.event {
            HarnessEventPayload::TurnLifecycle(lifecycle) => Some(lifecycle.kind),
            _ => None,
        })
        .collect()
}

/// Returns the state transitions of `events`, in order.
fn transitions(events: &[HarnessEvent]) -> Vec<(HarnessState, HarnessState)> {
    events
        .iter()
        .filter_map(|event| match &event.event {
            HarnessEventPayload::StateChanged(change) => Some((change.old, change.new)),
            _ => None,
        })
        .collect()
}

/// Returns the terminal turn result carried by `events`, when one is present.
fn terminal_result(events: &[HarnessEvent]) -> Option<&TurnResult> {
    events.iter().find_map(|event| match &event.event {
        HarnessEventPayload::TurnLifecycle(lifecycle) if lifecycle.kind.is_terminal() => {
            lifecycle.result.as_ref()
        }
        _ => None,
    })
}

/// A card that requires a host capability the context lacks must refuse to start.
#[tokio::test]
async fn start_rejects_an_incompatible_host() {
    let driver = FakeDriver::new(Mode::Answer);
    let mut card = full_card();
    card.required_host_capabilities = BTreeSet::from([HostCapability::Telemetry]);
    let harness = harness(card, &driver);

    let error = harness
        .start(context())
        .await
        .expect_err("the host does not offer telemetry");
    assert_eq!(error.kind(), HarnessErrorKind::Protocol);
    assert_eq!(
        error.message(),
        "host is missing required capabilities: telemetry"
    );
    assert_eq!(harness.state(), HarnessState::Terminated);
    assert_eq!(driver.captures().opens, 0, "the driver must not be opened");
}

/// A driver that spawns but never answers `open` must fail the start with a typed timeout.
#[tokio::test]
async fn start_times_out_when_the_driver_never_opens() {
    let driver = FakeDriver::new(Mode::Answer);
    driver.hang_open(true);
    let harness = harness(full_card(), &driver).with_open_timeout(Duration::from_millis(50));

    let error = tokio::time::timeout(BOUND, harness.start(context()))
        .await
        .expect("start must be bounded by the open timeout")
        .expect_err("a silent driver cannot be started");
    assert_eq!(error.kind(), HarnessErrorKind::Protocol);
    assert_eq!(
        error.message(),
        "the Fake harness did not open within 0.05s"
    );
    assert_eq!(harness.state(), HarnessState::Terminated);
    assert_eq!(harness.provider_session_id(), None);
    assert_eq!(driver.captures().opens, 1);
    assert_eq!(
        driver.captures().closes,
        1,
        "the failed open is closed best-effort"
    );

    // The start lock was released: once the driver answers, the same handle starts and stops.
    driver.hang_open(false);
    tokio::time::timeout(BOUND, harness.start(context()))
        .await
        .expect("the start lock was released")
        .expect("the driver answers now");
    assert_eq!(harness.state(), HarnessState::Idle);
    tokio::time::timeout(BOUND, harness.stop())
        .await
        .expect("stop must not wait on the start lock")
        .expect("stop");
    assert_eq!(harness.state(), HarnessState::Terminated);
}

/// A restart that would switch the provider session id is rejected and closes the driver.
#[tokio::test]
async fn a_refused_session_change_closes_the_driver() {
    let driver = FakeDriver::new(Mode::Answer);
    let harness = harness(full_card(), &driver);
    harness.start(context()).await.expect("first start");
    harness.stop().await.expect("first stop");
    driver.set_open_session("another-session");

    let error = tokio::time::timeout(BOUND, harness.start(context()))
        .await
        .expect("start must not hang")
        .expect_err("the provider session id is the handle identity");
    assert_eq!(error.kind(), HarnessErrorKind::State);
    assert_eq!(
        error.message(),
        "the Fake harness cannot be restarted with a different provider session"
    );
    assert_eq!(harness.state(), HarnessState::Terminated);
    assert_eq!(
        driver.captures().closes,
        2,
        "the rejected restart closes the driver it just opened"
    );
}

/// A compatible host starts the cycle and publishes exactly one transition.
#[tokio::test]
async fn start_publishes_one_transition_and_settles_idle() {
    let driver = FakeDriver::new(Mode::Answer);
    let harness = harness(full_card(), &driver);
    harness
        .start(context())
        .await
        .expect("the host is compatible");

    assert_eq!(harness.state(), HarnessState::Idle);
    assert_eq!(harness.provider_session_id(), Some("fake-session"));
    assert_eq!(harness.event_buffer_config().capacity, 1024);
    assert_eq!(harness.card().name, "fake-harness");
    assert_eq!(harness.label(), "Fake");

    let mut cursor = harness.events();
    let events = take(cursor.as_mut(), 1).await;
    assert_eq!(
        transitions(&events),
        vec![(HarnessState::Terminated, HarnessState::Idle)]
    );
    assert_eq!(events[0].sequence, 1);
    assert_eq!(events[0].host_session_id, "host-session");
    assert_eq!(events[0].agent_id, "agent-1");
    assert_eq!(
        events[0].provider_session_id.as_deref(),
        Some("fake-session")
    );
    let quiet = tokio::time::timeout(QUIET, cursor.next_event()).await;
    assert!(quiet.is_err(), "start publishes exactly one event");

    let error = harness.start(context()).await.expect_err("already started");
    assert_eq!(error.kind(), HarnessErrorKind::State);
    assert_eq!(error.message(), "Fake harness is already started");

    harness.stop().await.expect("stop");
}

/// One accepted input produces the documented event sequence.
#[tokio::test]
async fn one_turn_publishes_the_documented_sequence() {
    let driver = FakeDriver::new(Mode::Answer);
    let harness = harness(full_card(), &driver);
    let mut cursor = started(&harness).await;

    let receipt = harness
        .send(HarnessInput::text("hello"), DeliveryMode::Auto)
        .await
        .expect("the input is accepted");
    assert_eq!(receipt.accepted_mode, DeliveryMode::FollowUp);
    assert!(receipt.turn_id.starts_with("turn-"), "{}", receipt.turn_id);
    assert!(
        receipt.message_id.starts_with("message-"),
        "{}",
        receipt.message_id
    );

    let events = take(cursor.as_mut(), 8).await;
    assert_eq!(
        transitions(&events),
        vec![
            (HarnessState::Idle, HarnessState::Running),
            (HarnessState::Running, HarnessState::Idle),
        ]
    );
    assert_eq!(
        turn_kinds(&events),
        vec![TurnEventKind::Started, TurnEventKind::Finished]
    );

    let started = &events[1];
    assert!(matches!(
        &started.event,
        HarnessEventPayload::TurnLifecycle(_)
    ));
    assert_eq!(started.turn_id.as_deref(), Some(receipt.turn_id.as_str()));
    assert_eq!(
        started.correlation_id.as_deref(),
        Some(receipt.message_id.as_str())
    );

    assert!(matches!(&events[2].event, HarnessEventPayload::Output(_)));
    let HarnessEventPayload::ItemLifecycle(tool_started) = &events[3].event else {
        panic!("expected the tool-call item lifecycle");
    };
    assert_eq!(tool_started.kind, ItemEventKind::Started);
    assert_eq!(tool_started.item_type, "tool_call");
    assert_eq!(events[3].item_id.as_deref(), Some("item-1"));
    let HarnessEventPayload::ItemLifecycle(tool_completed) = &events[4].event else {
        panic!("expected the tool-result item lifecycle");
    };
    assert_eq!(tool_completed.kind, ItemEventKind::Completed);
    assert_eq!(tool_completed.data["result"], json!("contents"));
    assert!(matches!(
        &events[5].event,
        HarnessEventPayload::UsageUpdated(_)
    ));

    let finished = &events[6];
    assert_eq!(finished.turn_id.as_deref(), Some(receipt.turn_id.as_str()));
    assert_eq!(
        finished.correlation_id.as_deref(),
        Some(receipt.message_id.as_str())
    );
    let result = terminal_result(&events).expect("a terminal result");
    assert_eq!(result.status, TurnStatus::Completed);
    assert_eq!(result.final_output, json!("hello"));
    assert!(result.error.is_none());
    assert_eq!(harness.state(), HarnessState::Idle);

    harness.stop().await.expect("stop");
}

/// Two accepted inputs run strictly one after the other.
#[tokio::test]
async fn queued_turns_run_one_at_a_time_in_order() {
    let driver = FakeDriver::new(Mode::Answer);
    let harness = harness(full_card(), &driver);
    let mut cursor = started(&harness).await;

    let first = harness
        .send(HarnessInput::text("one"), DeliveryMode::Auto)
        .await
        .expect("first input");
    let second = harness
        .send(HarnessInput::text("two"), DeliveryMode::FollowUp)
        .await
        .expect("second input");
    assert_ne!(first.turn_id, second.turn_id);

    let events = take(cursor.as_mut(), 16).await;
    let started: Vec<&str> = events
        .iter()
        .filter(|event| {
            matches!(
                &event.event,
                HarnessEventPayload::TurnLifecycle(lifecycle)
                    if lifecycle.kind == TurnEventKind::Started
            )
        })
        .filter_map(|event| event.turn_id.as_deref())
        .collect();
    let finished: Vec<&str> = events
        .iter()
        .filter(|event| {
            matches!(
                &event.event,
                HarnessEventPayload::TurnLifecycle(lifecycle)
                    if lifecycle.kind == TurnEventKind::Finished
            )
        })
        .filter_map(|event| event.turn_id.as_deref())
        .collect();
    assert_eq!(
        started,
        vec![first.turn_id.as_str(), second.turn_id.as_str()]
    );
    assert_eq!(
        finished,
        vec![first.turn_id.as_str(), second.turn_id.as_str()]
    );
    assert_eq!(
        turn_kinds(&events),
        vec![
            TurnEventKind::Started,
            TurnEventKind::Finished,
            TurnEventKind::Started,
            TurnEventKind::Finished,
        ]
    );
    assert_eq!(driver.turns(), 2);
    assert_eq!(driver.max_running(), 1, "turns never overlap");
    assert_eq!(harness.state(), HarnessState::Idle);

    harness.stop().await.expect("stop");
}

/// A driver error becomes a `Failed` terminal event and the harness stays usable.
#[tokio::test]
async fn a_driver_error_fails_the_turn_and_keeps_the_harness_usable() {
    let driver = FakeDriver::new(Mode::Error);
    let harness = harness(full_card(), &driver);
    let mut cursor = started(&harness).await;

    harness
        .send(HarnessInput::text("boom"), DeliveryMode::Auto)
        .await
        .expect("accepted");
    let events = take(cursor.as_mut(), 4).await;
    assert_eq!(
        turn_kinds(&events),
        vec![TurnEventKind::Started, TurnEventKind::Failed]
    );
    let result = terminal_result(&events).expect("a terminal result");
    assert_eq!(result.status, TurnStatus::Failed);
    let error = result.error.as_ref().expect("a normalized error");
    assert_eq!(error.message, "the fake CLI crashed");
    assert_eq!(error.code.as_deref(), Some("protocol"));
    assert_eq!(error.category.as_deref(), Some("protocol"));
    assert_eq!(error.retryable, Some(false));
    assert_eq!(harness.state(), HarnessState::Idle);

    driver.set_mode(Mode::Answer);
    harness
        .send(HarnessInput::text("again"), DeliveryMode::Auto)
        .await
        .expect("accepted");
    let events = take(cursor.as_mut(), 8).await;
    let result = terminal_result(&events).expect("a terminal result");
    assert_eq!(result.status, TurnStatus::Completed);
    assert_eq!(result.final_output, json!("again"));

    harness.stop().await.expect("stop");
}

/// A provider-level failure result keeps its own normalized error.
#[tokio::test]
async fn a_provider_failure_result_is_reported_as_a_failed_turn() {
    let driver = FakeDriver::new(Mode::Fail);
    let harness = harness(full_card(), &driver);
    let mut cursor = started(&harness).await;

    harness
        .send(HarnessInput::text("ask"), DeliveryMode::Auto)
        .await
        .expect("accepted");
    let events = take(cursor.as_mut(), 4).await;
    assert_eq!(
        turn_kinds(&events),
        vec![TurnEventKind::Started, TurnEventKind::Failed]
    );
    let result = terminal_result(&events).expect("a terminal result");
    let error = result.error.as_ref().expect("a normalized error");
    assert_eq!(error.message, "the model refused the turn");
    assert_eq!(error.code.as_deref(), Some("refused"));
    assert_eq!(error.category.as_deref(), Some("provider"));

    harness.stop().await.expect("stop");
}

/// Steering needs an active turn and reaches the driver once one runs.
#[tokio::test]
async fn steer_needs_an_active_turn_and_reaches_the_driver() {
    let driver = FakeDriver::new(Mode::Hang);
    let harness = harness(full_card(), &driver);
    let mut cursor = started(&harness).await;

    let error = harness
        .send(HarnessInput::text("early"), DeliveryMode::Steer)
        .await
        .expect_err("no turn runs yet");
    assert_eq!(error.kind(), HarnessErrorKind::State);
    assert_eq!(error.message(), "there is no active turn to steer");

    let turn = harness
        .send(HarnessInput::text("hello"), DeliveryMode::Auto)
        .await
        .expect("accepted");
    let events = take(cursor.as_mut(), 2).await;
    assert_eq!(
        turn_kinds(&events),
        vec![TurnEventKind::Started],
        "the turn is running"
    );

    let receipt = harness
        .send(HarnessInput::text("hurry"), DeliveryMode::Steer)
        .await
        .expect("the running turn accepts a steer");
    assert_eq!(receipt.accepted_mode, DeliveryMode::Steer);
    assert_eq!(receipt.turn_id, turn.turn_id);
    assert_ne!(receipt.message_id, turn.message_id);
    assert_eq!(
        driver.captures().steers,
        vec![(turn.turn_id.clone(), json!("hurry"))]
    );

    harness.abort(AbortMode::Graceful).await.expect("abort");
    let events = take(cursor.as_mut(), 2).await;
    assert_eq!(turn_kinds(&events), vec![TurnEventKind::Aborted]);
    let result = terminal_result(&events).expect("a terminal result");
    assert_eq!(result.status, TurnStatus::Interrupted);
    assert_eq!(
        result.termination.as_ref().expect("a termination").kind,
        TurnTerminationKind::HarnessStop
    );

    harness.stop().await.expect("stop");
}

/// A graceful abort reaches the driver and ends the turn as interrupted.
#[tokio::test]
async fn abort_reaches_a_hanging_turn() {
    let driver = FakeDriver::new(Mode::Hang);
    let harness = harness(full_card(), &driver);
    let mut cursor = started(&harness).await;

    let turn = harness
        .send(HarnessInput::text("hello"), DeliveryMode::Auto)
        .await
        .expect("accepted");
    let _ = take(cursor.as_mut(), 2).await;

    harness
        .abort(AbortMode::Graceful)
        .await
        .expect("the driver accepts the interrupt");
    assert_eq!(
        driver.interrupts(),
        vec![(Some(turn.turn_id.clone()), AbortMode::Graceful)]
    );
    let events = take(cursor.as_mut(), 2).await;
    assert_eq!(
        transitions(&events),
        vec![(HarnessState::Running, HarnessState::Idle)]
    );
    let result = terminal_result(&events).expect("a terminal result");
    assert_eq!(result.status, TurnStatus::Interrupted);

    harness.stop().await.expect("stop");
}

/// Aborting with no active turn drains every queued turn.
#[tokio::test]
async fn abort_without_an_active_turn_drains_the_queue() {
    let driver = FakeDriver::new(Mode::Hang);
    let harness = harness(full_card(), &driver);
    harness.start(context()).await.expect("start");
    let mut cursor = harness.events();

    // The supervisor only runs when this task yields, so both inputs are still queued when the
    // abort arrives: exactly the "queued turns, no active turn" state.
    let first = harness
        .send(HarnessInput::text("one"), DeliveryMode::Auto)
        .await
        .expect("accepted");
    let second = harness
        .send(HarnessInput::text("two"), DeliveryMode::Auto)
        .await
        .expect("accepted");
    harness.abort(AbortMode::Graceful).await.expect("abort");

    assert_eq!(driver.turns(), 0, "no turn ran before the abort");
    let events = take(cursor.as_mut(), 5).await;
    assert_eq!(
        transitions(&events),
        vec![(HarnessState::Terminated, HarnessState::Idle)]
    );
    assert_eq!(
        turn_kinds(&events),
        vec![
            TurnEventKind::Started,
            TurnEventKind::Aborted,
            TurnEventKind::Started,
            TurnEventKind::Aborted,
        ]
    );
    let drained: Vec<&str> = events
        .iter()
        .filter_map(|event| match &event.event {
            HarnessEventPayload::TurnLifecycle(lifecycle)
                if lifecycle.kind == TurnEventKind::Aborted =>
            {
                event.turn_id.as_deref()
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        drained,
        vec![first.turn_id.as_str(), second.turn_id.as_str()]
    );
    for event in &events[1..] {
        let HarnessEventPayload::TurnLifecycle(lifecycle) = &event.event else {
            assert!(matches!(&event.event, HarnessEventPayload::StateChanged(_)));
            continue;
        };
        if lifecycle.kind == TurnEventKind::Aborted {
            let termination = lifecycle
                .result
                .as_ref()
                .and_then(|result| result.termination.as_ref())
                .expect("a drained turn carries a termination");
            assert_eq!(termination.kind, TurnTerminationKind::HarnessStop);
        }
    }

    harness.stop().await.expect("stop");
}

/// Every gated command reports `UnsupportedCapability` on a bare card.
#[tokio::test]
async fn a_bare_card_rejects_the_gated_commands() {
    let driver = FakeDriver::new(Mode::Hang);
    let harness = harness(card(&[HarnessCapability::GracefulAbort]), &driver);
    let mut cursor = started(&harness).await;

    harness
        .send(HarnessInput::text("hello"), DeliveryMode::Auto)
        .await
        .expect("accepted");
    let _ = take(cursor.as_mut(), 2).await;

    let cases = [
        harness.abort(AbortMode::Force).await.err(),
        harness
            .send(HarnessInput::text("steer"), DeliveryMode::Steer)
            .await
            .err(),
        harness.pause().await.err(),
        harness.export_checkpoint().await.err(),
    ];
    let messages = [
        "the Fake harness does not support force abort",
        "the Fake harness does not support steering",
        "the Fake harness cannot pause",
        "the Fake harness does not support checkpoints",
    ];
    for (error, message) in cases.into_iter().zip(messages) {
        let error = error.expect("the command is gated");
        assert_eq!(error.kind(), HarnessErrorKind::UnsupportedCapability);
        assert_eq!(error.message(), message);
    }
    assert!(driver.interrupts().is_empty(), "the driver is never asked");
    assert_eq!(driver.captures().steers.len(), 0);

    // A resume of a harness that never paused is a state error, not a capability error.
    let error = harness
        .resume(None)
        .await
        .expect_err("this card cannot resume either");
    assert_eq!(error.kind(), HarnessErrorKind::UnsupportedCapability);

    harness.stop().await.expect("stop");
}

/// Pause and resume reach the driver and are published with the active turn id.
#[tokio::test]
async fn pause_and_resume_reach_the_driver() {
    let driver = FakeDriver::new(Mode::Hang);
    let harness = harness(full_card(), &driver);
    let mut cursor = started(&harness).await;

    let turn = harness
        .send(HarnessInput::text("hello"), DeliveryMode::Auto)
        .await
        .expect("accepted");
    let _ = take(cursor.as_mut(), 2).await;

    harness.pause().await.expect("the driver pauses");
    assert_eq!(driver.captures().pauses, 1);
    assert_eq!(harness.state(), HarnessState::Paused);
    let events = take(cursor.as_mut(), 2).await;
    assert_eq!(turn_kinds(&events), vec![TurnEventKind::Paused]);
    assert_eq!(events[0].turn_id.as_deref(), Some(turn.turn_id.as_str()));
    assert_eq!(
        transitions(&events),
        vec![(HarnessState::Running, HarnessState::Paused)]
    );

    harness.pause().await.expect("pausing twice is a no-op");
    assert_eq!(driver.captures().pauses, 1);

    harness.resume(None).await.expect("the driver resumes");
    assert_eq!(driver.captures().resumes, vec![None]);
    assert_eq!(harness.state(), HarnessState::Running);
    let events = take(cursor.as_mut(), 2).await;
    assert_eq!(turn_kinds(&events), vec![TurnEventKind::Resumed]);
    assert_eq!(events[0].turn_id.as_deref(), Some(turn.turn_id.as_str()));
    assert_eq!(
        transitions(&events),
        vec![(HarnessState::Paused, HarnessState::Running)]
    );

    harness.abort(AbortMode::Graceful).await.expect("abort");
    harness.stop().await.expect("stop");
}

/// Checkpoint export delegates to the driver.
#[tokio::test]
async fn export_checkpoint_delegates_and_returns_the_driver_snapshot() {
    let driver = FakeDriver::new(Mode::Answer);
    let harness = harness(full_card(), &driver);
    harness.start(context()).await.expect("start");

    let checkpoint = harness
        .export_checkpoint()
        .await
        .expect("the driver exports")
        .expect("a checkpoint");
    assert_eq!(checkpoint, fake_checkpoint());
    assert_eq!(driver.captures().checkpoints, 1);

    harness.stop().await.expect("stop");
}

/// The cycle cursor is single-consumer.
#[tokio::test]
async fn the_observation_lease_is_single_consumer() {
    let driver = FakeDriver::new(Mode::Answer);
    let harness = harness(full_card(), &driver);
    harness.start(context()).await.expect("start");

    let _first = harness.events();
    let mut second = harness.events();
    let error = second
        .next_event()
        .await
        .expect_err("a second consumer is refused");
    assert_eq!(error.kind(), HarnessErrorKind::State);
    assert_eq!(
        error.message(),
        "the harness observation stream already has an active consumer"
    );

    harness.stop().await.expect("stop");
}

/// `stop` is idempotent, drains queued turns and settles terminated.
#[tokio::test]
async fn stop_aborts_queued_turns_and_is_idempotent() {
    let driver = FakeDriver::new(Mode::Hang);
    let harness = harness(full_card(), &driver);
    let mut cursor = started(&harness).await;

    harness
        .send(HarnessInput::text("running"), DeliveryMode::Auto)
        .await
        .expect("accepted");
    let _ = take(cursor.as_mut(), 2).await;
    let queued = harness
        .send(HarnessInput::text("queued"), DeliveryMode::Auto)
        .await
        .expect("accepted");

    harness.stop().await.expect("stop closes the driver");
    assert_eq!(driver.captures().closes, 1);
    assert_eq!(harness.state(), HarnessState::Terminated);
    assert!(harness.provider_session_id().is_none());

    let events = take(cursor.as_mut(), 4).await;
    assert!(
        tokio::time::timeout(QUIET, cursor.next_event())
            .await
            .expect("the cursor ends after the cycle")
            .expect("the cursor is healthy")
            .is_none(),
        "the closed bus ends the cursor"
    );
    let drained: Vec<&str> = events
        .iter()
        .filter_map(|event| match &event.event {
            HarnessEventPayload::TurnLifecycle(lifecycle)
                if lifecycle.kind == TurnEventKind::Aborted =>
            {
                event.turn_id.as_deref()
            }
            _ => None,
        })
        .collect();
    assert!(
        drained.contains(&queued.turn_id.as_str()),
        "the queued turn is drained: {drained:?}"
    );
    let last = transitions(&events).last().copied().expect("a transition");
    assert_eq!(last.1, HarnessState::Terminated, "the cycle terminates");

    harness.stop().await.expect("stopping twice is a no-op");
    assert_eq!(driver.captures().closes, 1);

    let error = harness
        .send(HarnessInput::text("late"), DeliveryMode::Auto)
        .await
        .expect_err("a stopped harness refuses input");
    assert_eq!(error.kind(), HarnessErrorKind::State);
    assert_eq!(error.message(), "cannot send to a stopped Fake harness");
    assert!(harness.stop().await.is_ok());

    assert_eq!(harness.state(), HarnessState::Terminated);
}
