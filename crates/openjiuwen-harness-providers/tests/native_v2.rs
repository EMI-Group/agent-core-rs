//! The `native_v2` provider driven end to end through the harness SPI.
//!
//! Every assertion goes through the public surface: the provider's `create`, the
//! [`VendorHarness`] shell it returns, and the `openjiuwen-agent-teams` fakes injected through
//! [`NativeV2Driver::with_deps`]. The fake harness answers every member input with one scripted
//! observation round, records the commands it receives, and can hang a turn so abort, pause and
//! steering are observable while a run is in flight.
//!
//! The `FakeHarnessFactory` / `ScriptedHarness` pair is a local copy of the pattern
//! `openjiuwen-agent-teams` uses in its own suite: only the public API parts are restated, because
//! another crate's `tests/` module is not importable.
#![cfg(feature = "native-v2")]
// Tests assert by panicking on failure, which is the intent of a test.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_agent_teams::harness::{MemberHarnessContext, MemberHarnessFactory};
use openjiuwen_agent_teams::runtime::TeamBuildDeps;
use openjiuwen_agent_teams::schema::{DeepAgentSpec, TeamAgentSpec};
use openjiuwen_common::error::Result;
use openjiuwen_harness_protocol::{
    AbortMode, DeliveryMode, EventBufferConfig, HarnessCapability, HarnessCard, HarnessCheckpoint,
    HarnessContext, HarnessErrorKind, HarnessEvent, HarnessEventCursor, HarnessEventPayload,
    HarnessInput, HarnessProtocol, HarnessProvider, HarnessResult, HarnessState, ItemEventKind,
    ItemLifecycleEvent, JsonObject, OutputChannel, OutputEvent, OutputKind, OutputOperation,
    SendReceipt, StateChangedEvent, TurnEventKind, TurnLifecycleEvent, TurnResult, TurnStatus,
    TurnTerminationKind,
};
use openjiuwen_harness_providers::providers::native_v2::{
    card, provider, NativeV2Config, NativeV2Driver, CARD_NAME,
};
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio::time::timeout;

/// Bound every await in this suite so a hang fails the test instead of blocking the run.
const BOUND: Duration = Duration::from_secs(10);

/// Agent id the scripted harnesses stamp on their observations.
const AGENT_ID: &str = "team_leader";

/// One command a [`ScriptedHarness`] received.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Call {
    /// The cycle was started.
    Start,
    /// The cycle was stopped.
    Stop,
    /// The cycle was paused.
    Pause,
    /// A warm round was resumed.
    Resume,
    /// An input was delivered; the payload is its text.
    Send(String),
    /// An abort was requested; the mode is recorded verbatim.
    Abort(AbortMode),
}

/// Channel-backed cursor that yields the scripted observations in order.
struct ScriptCursor {
    /// Event source; closed when the harness drops its sender.
    receiver: mpsc::UnboundedReceiver<HarnessEvent>,
}

#[async_trait]
impl HarnessEventCursor for ScriptCursor {
    async fn next_event(&mut self) -> HarnessResult<Option<HarnessEvent>> {
        Ok(self.receiver.recv().await)
    }

    async fn close(&mut self) -> HarnessResult<()> {
        Ok(())
    }
}

/// Harness protocol handle with scripted answers: records every command it receives.
///
/// A plain handle emits one `started`/`answer`/`finished` round per input and returns to `Idle`; a
/// [`hanging`](Self::hanging) one leaves the state `Running` and emits nothing, so a suite can
/// observe an in-flight turn.
struct ScriptedHarness {
    /// State reported by [`HarnessProtocol::state`].
    state: Mutex<HarnessState>,
    /// Commands received, in order.
    calls: Mutex<Vec<Call>>,
    /// Static capability card.
    card: HarnessCard,
    /// Observation buffer policy.
    buffer: EventBufferConfig,
    /// The single observation receiver handed to the first `events()` caller.
    events: Mutex<Option<mpsc::UnboundedReceiver<HarnessEvent>>>,
    /// Sender that scripts the observation stream.
    sender: mpsc::UnboundedSender<HarnessEvent>,
    /// Answer text emitted by every `send`.
    answer: String,
    /// When set, `send` leaves the harness `Running` and emits nothing.
    hang: bool,
    /// Envelope sequence counter.
    sequence: AtomicU64,
}

impl ScriptedHarness {
    /// Builds a harness that answers every input with `answer` and returns to `Idle`.
    fn new(answer: impl Into<String>) -> Arc<Self> {
        Self::build(answer.into(), false)
    }

    /// Builds a harness whose every `send` leaves the state `Running` and emits nothing.
    fn hanging(answer: impl Into<String>) -> Arc<Self> {
        Self::build(answer.into(), true)
    }

    /// Shared constructor of the two shapes.
    fn build(answer: String, hang: bool) -> Arc<Self> {
        let (sender, receiver) = mpsc::unbounded_channel();
        Arc::new(Self {
            state: Mutex::new(HarnessState::Idle),
            calls: Mutex::new(Vec::new()),
            card: HarnessCard::new("scripted", "0.0.0"),
            buffer: EventBufferConfig::default(),
            events: Mutex::new(Some(receiver)),
            sender,
            answer,
            hang,
            sequence: AtomicU64::new(0),
        })
    }

    /// Returns the recorded commands, in order.
    fn calls(&self) -> Vec<Call> {
        self.calls.lock().expect("calls lock").clone()
    }

    /// Returns the text of every delivered input, in order.
    fn sends(&self) -> Vec<String> {
        self.calls()
            .into_iter()
            .filter_map(|call| match call {
                Call::Send(text) => Some(text),
                _ => None,
            })
            .collect()
    }

    /// Returns the abort modes requested, in order.
    fn abort_modes(&self) -> Vec<AbortMode> {
        self.calls()
            .into_iter()
            .filter_map(|call| match call {
                Call::Abort(mode) => Some(mode),
                _ => None,
            })
            .collect()
    }

    /// Pushes one observation onto the stream.
    fn emit(&self, event: HarnessEvent) {
        let _ = self.sender.send(event);
    }

    /// Pushes one `turn_lifecycle` observation carrying `result`.
    fn emit_turn(&self, kind: TurnEventKind, result: Option<TurnResult>) {
        self.emit(self.envelope(TurnLifecycleEvent { kind, result }.into()));
    }

    /// Pushes one answer/`text`/`delta` output observation for `text`.
    fn emit_answer(&self, text: &str) {
        self.emit(
            self.envelope(
                OutputEvent {
                    output_id: "out-1".to_owned(),
                    kind: OutputKind::Text,
                    content: json!(text),
                    operation: OutputOperation::Delta,
                    channel: OutputChannel::Answer,
                    content_index: 0,
                    content_type: None,
                    data: BTreeMap::new(),
                }
                .into(),
            ),
        );
    }

    /// Pushes one tool item observation (`Started` or `Completed`).
    fn emit_tool(&self, kind: ItemEventKind, data: JsonObject) {
        let event = self
            .envelope(
                ItemLifecycleEvent {
                    kind,
                    item_type: "tool".to_owned(),
                    data,
                }
                .into(),
            )
            .with_item_id("item-1");
        self.emit(event);
    }

    /// Pushes one state transition observation.
    fn emit_state(&self, old: HarnessState, new: HarnessState) {
        *self.state.lock().expect("state lock") = new;
        self.emit(self.envelope(StateChangedEvent { old, new }.into()));
    }

    /// Builds one envelope for `payload`, advancing the sequence counter.
    fn envelope(&self, payload: HarnessEventPayload) -> HarnessEvent {
        let sequence = self.sequence.fetch_add(1, Ordering::SeqCst);
        HarnessEvent::new(
            sequence + 1,
            1.0,
            payload,
            format!("session:{AGENT_ID}"),
            AGENT_ID,
        )
    }

    /// Pushes the one answer triple a settled round produces: started, output, finished.
    fn emit_round(&self) {
        self.emit_turn(TurnEventKind::Started, None);
        self.emit_answer(&self.answer);
        self.emit_turn(TurnEventKind::Finished, Some(completed_result()));
    }
}

#[async_trait]
impl HarnessProtocol for ScriptedHarness {
    fn card(&self) -> &HarnessCard {
        &self.card
    }

    fn state(&self) -> HarnessState {
        *self.state.lock().expect("state lock")
    }

    fn provider_session_id(&self) -> Option<&str> {
        None
    }

    fn event_buffer_config(&self) -> &EventBufferConfig {
        &self.buffer
    }

    async fn start(&self, _context: HarnessContext) -> HarnessResult<()> {
        self.calls.lock().expect("calls lock").push(Call::Start);
        *self.state.lock().expect("state lock") = HarnessState::Idle;
        Ok(())
    }

    async fn stop(&self) -> HarnessResult<()> {
        self.calls.lock().expect("calls lock").push(Call::Stop);
        *self.state.lock().expect("state lock") = HarnessState::Terminated;
        Ok(())
    }

    fn events(&self) -> Box<dyn HarnessEventCursor> {
        let receiver = self.events.lock().expect("events lock").take();
        match receiver {
            Some(receiver) => Box::new(ScriptCursor { receiver }),
            None => Box::new(EmptyCursor),
        }
    }

    fn turn_events(&self, _turn_id: Option<&str>) -> Box<dyn HarnessEventCursor> {
        Box::new(EmptyCursor)
    }

    async fn send(&self, input: HarnessInput, _mode: DeliveryMode) -> HarnessResult<SendReceipt> {
        let text = input.content.as_str().unwrap_or_default().to_owned();
        self.calls
            .lock()
            .expect("calls lock")
            .push(Call::Send(text));
        *self.state.lock().expect("state lock") = HarnessState::Running;
        if !self.hang {
            self.emit_round();
            *self.state.lock().expect("state lock") = HarnessState::Idle;
        }
        Ok(SendReceipt {
            message_id: "message-1".to_owned(),
            turn_id: "turn-1".to_owned(),
            accepted_mode: DeliveryMode::FollowUp,
        })
    }

    async fn abort(&self, mode: AbortMode) -> HarnessResult<()> {
        self.calls
            .lock()
            .expect("calls lock")
            .push(Call::Abort(mode));
        Ok(())
    }

    async fn pause(&self) -> HarnessResult<()> {
        self.calls.lock().expect("calls lock").push(Call::Pause);
        *self.state.lock().expect("state lock") = HarnessState::Paused;
        Ok(())
    }

    async fn resume(&self, _query: Option<HarnessInput>) -> HarnessResult<()> {
        self.calls.lock().expect("calls lock").push(Call::Resume);
        *self.state.lock().expect("state lock") = HarnessState::Running;
        Ok(())
    }

    async fn export_checkpoint(&self) -> HarnessResult<Option<HarnessCheckpoint>> {
        Ok(None)
    }
}

/// Cursor over an empty stream, for the second observation consumer.
struct EmptyCursor;

#[async_trait]
impl HarnessEventCursor for EmptyCursor {
    async fn next_event(&mut self) -> HarnessResult<Option<HarnessEvent>> {
        Ok(None)
    }

    async fn close(&mut self) -> HarnessResult<()> {
        Ok(())
    }
}

/// Returns the result a `Finished` turn carries.
fn completed_result() -> TurnResult {
    TurnResult {
        status: TurnStatus::Completed,
        messages: Vec::new(),
        final_output: json!(null),
        structured_output: json!(null),
        stop_reason: Some("stop".to_owned()),
        termination: None,
        error: None,
        usage: None,
        cost: None,
        started_at: None,
        completed_at: None,
        duration_ms: None,
        provider_data: JsonObject::new(),
    }
}

/// Harness factory that builds [`ScriptedHarness`]es and records the contexts it was given.
#[derive(Default)]
struct FakeHarnessFactory {
    /// Contexts received by [`MemberHarnessFactory::create`], in call order.
    calls: Mutex<Vec<MemberHarnessContext>>,
    /// Members whose harness never settles.
    hanging: Mutex<BTreeSet<String>>,
    /// Per-member answer scripted before the first `create`.
    answers: Mutex<BTreeMap<String, String>>,
    /// Harness built per member, so a suite can inspect what the member received.
    harnesses: Mutex<BTreeMap<String, Arc<ScriptedHarness>>>,
}

impl FakeHarnessFactory {
    /// Builds a factory whose members answer `done: <member>`.
    fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Makes the harness of `member` hang on every input.
    fn hanging_for(&self, member: &str) {
        self.hanging
            .lock()
            .expect("hanging lock")
            .insert(member.to_owned());
    }

    /// Scripts the answer of `member`.
    fn answers_with(&self, member: &str, answer: &str) {
        self.answers
            .lock()
            .expect("answers lock")
            .insert(member.to_owned(), answer.to_owned());
    }

    /// Returns the members a harness was created for, in creation order.
    fn created_members(&self) -> Vec<String> {
        self.calls
            .lock()
            .expect("calls lock")
            .iter()
            .map(|ctx| ctx.member_name.clone())
            .collect()
    }

    /// Returns the harness built for `member`, panicking when it was never built.
    fn harness(&self, member: &str) -> Arc<ScriptedHarness> {
        self.harnesses
            .lock()
            .expect("harnesses lock")
            .get(member)
            .cloned()
            .unwrap_or_else(|| panic!("no harness was built for '{member}'"))
    }
}

#[async_trait]
impl MemberHarnessFactory for FakeHarnessFactory {
    async fn create(
        &self,
        _spec: &DeepAgentSpec,
        ctx: &MemberHarnessContext,
    ) -> Result<Arc<dyn HarnessProtocol>> {
        self.calls.lock().expect("calls lock").push(ctx.clone());
        let answer = self
            .answers
            .lock()
            .expect("answers lock")
            .get(&ctx.member_name)
            .cloned()
            .unwrap_or_else(|| format!("done: {}", ctx.member_name));
        let harness = if self
            .hanging
            .lock()
            .expect("hanging lock")
            .contains(&ctx.member_name)
        {
            ScriptedHarness::hanging(answer)
        } else {
            ScriptedHarness::new(answer)
        };
        self.harnesses
            .lock()
            .expect("harnesses lock")
            .insert(ctx.member_name.clone(), Arc::clone(&harness));
        Ok(harness)
    }

    fn name(&self) -> &'static str {
        "fake"
    }
}

/// Returns the dependencies of one test: the fake factory over the in-process defaults.
fn deps(factory: &Arc<FakeHarnessFactory>) -> TeamBuildDeps {
    TeamBuildDeps::default()
        .with_harness_factory(Arc::clone(factory) as Arc<dyn MemberHarnessFactory>)
}

/// Builds a single-leader team blueprint as JSON.
fn team_json(team_name: &str) -> Value {
    json!({
        "team_name": team_name,
        "agents": {
            "leader": {"card": {"name": "Leader"}, "system_prompt": "lead the team"},
        },
    })
}

/// Builds a create mapping for `team_name`.
fn create_mapping(team_name: &str) -> JsonObject {
    let mut object = JsonObject::new();
    object.insert("team".to_owned(), team_json(team_name));
    object
}

/// Decodes `create_mapping` into the configuration the driver takes.
fn config(team_name: &str) -> NativeV2Config {
    NativeV2Config::from_json(&create_mapping(team_name)).expect("the configuration is valid")
}

/// Builds the provider card of a `native_v2` harness.
fn v2_card() -> HarnessCard {
    card()
}

/// Builds an unstarted harness over the fake factory.
fn harness_for(team_name: &str, factory: &Arc<FakeHarnessFactory>) -> Box<dyn HarnessProtocol> {
    let driver = NativeV2Driver::with_deps(config(team_name), deps(factory));
    Box::new(
        openjiuwen_harness_providers::vendor_cli::VendorHarness::new(
            v2_card(),
            EventBufferConfig::default(),
            Arc::new(driver),
        ),
    )
}

/// Reads observations until `stop` matches, panicking when the bound expires.
async fn read_until(
    cursor: &mut Box<dyn HarnessEventCursor>,
    mut stop: impl FnMut(&HarnessEvent) -> bool,
) -> Vec<HarnessEvent> {
    timeout(BOUND, async {
        let mut events = Vec::new();
        while let Some(event) = cursor.next_event().await.expect("the stream is healthy") {
            let matched = stop(&event);
            events.push(event);
            if matched {
                break;
            }
        }
        events
    })
    .await
    .expect("the expected observation arrived within the bound")
}

/// Returns whether `event` is a terminal turn lifecycle event.
fn is_terminal(event: &HarnessEvent) -> bool {
    matches!(&event.event, HarnessEventPayload::TurnLifecycle(turn) if turn.kind.is_terminal())
}

/// Returns the terminal result of the last turn lifecycle event in `events`.
fn terminal_result(events: &[HarnessEvent]) -> TurnResult {
    events
        .iter()
        .rev()
        .find_map(|event| match &event.event {
            HarnessEventPayload::TurnLifecycle(turn) => turn.result.clone(),
            _ => None,
        })
        .expect("the sequence ends with a terminal turn event")
}

/// Returns the answer text of the output observations in `events`.
fn answer_text(events: &[HarnessEvent]) -> String {
    events
        .iter()
        .filter_map(|event| match &event.event {
            HarnessEventPayload::Output(output) if output.channel == OutputChannel::Answer => {
                output.content.as_str().map(str::to_owned)
            }
            _ => None,
        })
        .collect()
}

/// Starts `harness` and returns its observation cursor.
async fn started(harness: &dyn HarnessProtocol, session: &str) -> Box<dyn HarnessEventCursor> {
    let context = HarnessContext::new("native-v2", AGENT_ID, session, "lead the team");
    timeout(BOUND, harness.start(context))
        .await
        .expect("start finished within the bound")
        .expect("the harness started");
    let mut cursor = harness.events();
    read_until(&mut cursor, |event| {
        matches!(
            &event.event,
            HarnessEventPayload::StateChanged(change)
                if change.new == HarnessState::Idle
        )
    })
    .await;
    cursor
}

#[test]
fn create_builds_an_unstarted_harness_with_the_v2_card() {
    let factory = FakeHarnessFactory::new();
    let harness = provider()
        .create(create_mapping("create-alpha"))
        .expect("the configuration is valid");
    assert_eq!(harness.state(), HarnessState::Terminated);
    assert!(harness.provider_session_id().is_none());
    assert_eq!(harness.card().name, CARD_NAME);
    assert_eq!(harness.card().implementation_version, "0.2.0");
    assert!(harness
        .card()
        .supports(HarnessCapability::PersistentSession));
    assert!(
        factory.created_members().is_empty(),
        "create never runs a team"
    );
    assert_eq!(
        harness_for("create-beta", &factory).state(),
        HarnessState::Terminated
    );
}

#[test]
fn create_requires_a_team_specification() {
    let error = provider()
        .create(JsonObject::new())
        .err()
        .expect("a team is mandatory");
    assert_eq!(error.kind(), HarnessErrorKind::Validation);
    assert_eq!(error.message(), "native_v2 requires a team specification");
}

#[test]
fn create_rejects_unknown_keys() {
    let mut object = create_mapping("unknown-alpha");
    object.insert("deep_agent".to_owned(), json!({}));
    let error = provider()
        .create(object)
        .err()
        .expect("unknown keys are rejected");
    assert_eq!(error.kind(), HarnessErrorKind::Validation);
    assert_eq!(
        error.message(),
        "unknown native_v2 configuration fields: deep_agent"
    );
}

#[tokio::test]
async fn a_turn_reports_started_answer_and_finished_with_the_team_session() {
    let factory = FakeHarnessFactory::new();
    factory.answers_with(AGENT_ID, "the team answered");
    let harness = harness_for("turn-alpha", &factory);
    let mut cursor = started(harness.as_ref(), "session-host").await;
    let session_id = harness
        .provider_session_id()
        .expect("the team session is the provider session")
        .to_owned();
    assert!(
        session_id.starts_with("session_"),
        "unexpected team session id: {session_id}"
    );
    assert_eq!(factory.created_members(), vec![AGENT_ID.to_owned()]);

    let receipt = timeout(
        BOUND,
        harness.send(HarnessInput::text("ship it"), DeliveryMode::FollowUp),
    )
    .await
    .expect("send finished within the bound")
    .expect("the input was accepted");
    assert_eq!(receipt.accepted_mode, DeliveryMode::FollowUp);

    let events = read_until(&mut cursor, is_terminal).await;
    assert!(matches!(
        events[0].event,
        HarnessEventPayload::StateChanged(StateChangedEvent {
            new: HarnessState::Running,
            ..
        })
    ));
    assert!(matches!(
        events[1].event,
        HarnessEventPayload::TurnLifecycle(TurnLifecycleEvent {
            kind: TurnEventKind::Started,
            ..
        })
    ));
    assert_eq!(answer_text(&events), "the team answered");
    let result = terminal_result(&events);
    assert_eq!(result.status, TurnStatus::Completed);
    assert_eq!(result.final_output, json!("the team answered"));
    assert_eq!(result.messages.len(), 1, "one assistant message");
    assert_eq!(
        result.messages[0].content[0].content,
        json!("the team answered")
    );
    assert_eq!(
        factory.harness(AGENT_ID).sends(),
        vec!["ship it".to_owned()]
    );
    timeout(BOUND, harness.stop())
        .await
        .expect("stop bounded")
        .expect("the harness stopped");
}

#[tokio::test]
async fn a_second_turn_keeps_the_same_team_session() {
    let factory = FakeHarnessFactory::new();
    factory.answers_with(AGENT_ID, "first answer");
    let harness = harness_for("turn-two", &factory);
    let mut cursor = started(harness.as_ref(), "session-host").await;
    let session_id = harness.provider_session_id().map(str::to_owned);

    for query in ["one", "two"] {
        timeout(
            BOUND,
            harness.send(HarnessInput::text(query), DeliveryMode::FollowUp),
        )
        .await
        .expect("send finished within the bound")
        .expect("the input was accepted");
        let events = read_until(&mut cursor, is_terminal).await;
        assert_eq!(terminal_result(&events).status, TurnStatus::Completed);
        assert_eq!(
            harness.provider_session_id().map(str::to_owned),
            session_id,
            "the team session survives every turn"
        );
    }
    assert_eq!(
        factory.harness(AGENT_ID).sends(),
        vec!["one".to_owned(), "two".to_owned()]
    );
    assert_eq!(
        factory.created_members().len(),
        1,
        "one team, one leader harness"
    );
    timeout(BOUND, harness.stop())
        .await
        .expect("stop bounded")
        .expect("the harness stopped");
}

#[tokio::test]
async fn export_checkpoint_is_unsupported_by_the_card() {
    let factory = FakeHarnessFactory::new();
    let harness = harness_for("checkpoint-alpha", &factory);
    let _cursor = started(harness.as_ref(), "session-host").await;
    let error = harness
        .export_checkpoint()
        .await
        .expect_err("the card declares no checkpoint");
    assert_eq!(error.kind(), HarnessErrorKind::UnsupportedCapability);
    assert!(
        error.message().contains("does not support checkpoints"),
        "unexpected message: {}",
        error.message()
    );
    timeout(BOUND, harness.stop())
        .await
        .expect("stop bounded")
        .expect("the harness stopped");
}

#[tokio::test]
async fn a_graceful_abort_reaches_the_member_runtime_and_ends_the_turn() {
    abort_ends_the_turn(AbortMode::Graceful).await;
}

#[tokio::test]
async fn a_forced_abort_reaches_the_member_runtime_and_ends_the_turn() {
    abort_ends_the_turn(AbortMode::Force).await;
}

/// Drives one hanging turn, aborts it in `mode`, and asserts the whole chain.
async fn abort_ends_the_turn(mode: AbortMode) {
    let factory = FakeHarnessFactory::new();
    factory.hanging_for(AGENT_ID);
    let team_name = match mode {
        AbortMode::Graceful => "abort-graceful",
        AbortMode::Force => "abort-force",
    };
    let harness = harness_for(team_name, &factory);
    let mut cursor = started(harness.as_ref(), "session-host").await;
    timeout(
        BOUND,
        harness.send(HarnessInput::text("work"), DeliveryMode::FollowUp),
    )
    .await
    .expect("send finished within the bound")
    .expect("the input was accepted");
    read_until(&mut cursor, |event| {
        matches!(
            &event.event,
            HarnessEventPayload::TurnLifecycle(TurnLifecycleEvent {
                kind: TurnEventKind::Started,
                ..
            })
        )
    })
    .await;

    timeout(BOUND, harness.abort(mode))
        .await
        .expect("abort finished within the bound")
        .expect("the harness accepted the abort");
    assert_eq!(factory.harness(AGENT_ID).abort_modes(), vec![mode]);

    let events = read_until(&mut cursor, is_terminal).await;
    let result = terminal_result(&events);
    assert_eq!(result.status, TurnStatus::Interrupted);
    let termination = result
        .termination
        .expect("an interrupted turn carries a termination");
    assert_eq!(termination.kind, TurnTerminationKind::UserAbort);
    assert_eq!(
        termination.message.as_deref(),
        Some("native_v2 interrupted the turn")
    );
    assert!(matches!(
        events.last().map(|event| &event.event),
        Some(HarnessEventPayload::TurnLifecycle(TurnLifecycleEvent {
            kind: TurnEventKind::Aborted,
            ..
        }))
    ));
    timeout(BOUND, harness.stop())
        .await
        .expect("stop bounded")
        .expect("the harness stopped");
}

#[tokio::test]
async fn pause_and_resume_reach_the_member_runtime() {
    let factory = FakeHarnessFactory::new();
    let harness = harness_for("pause-alpha", &factory);
    let _cursor = started(harness.as_ref(), "session-host").await;

    timeout(BOUND, harness.pause())
        .await
        .expect("pause finished within the bound")
        .expect("the card declares pause/resume");
    assert_eq!(harness.state(), HarnessState::Paused);

    timeout(BOUND, harness.resume(None))
        .await
        .expect("resume finished within the bound")
        .expect("the card declares pause/resume");
    assert_eq!(harness.state(), HarnessState::Idle);

    let calls = factory.harness(AGENT_ID).calls();
    assert!(calls.contains(&Call::Pause), "{calls:?}");
    assert!(calls.contains(&Call::Resume), "{calls:?}");
    timeout(BOUND, harness.stop())
        .await
        .expect("stop bounded")
        .expect("the harness stopped");
    assert!(
        factory.harness(AGENT_ID).calls().contains(&Call::Stop),
        "close stops the member runtime"
    );
}

#[tokio::test]
async fn steering_needs_an_active_turn_and_then_reaches_the_member_runtime() {
    let factory = FakeHarnessFactory::new();
    factory.hanging_for(AGENT_ID);
    let harness = harness_for("steer-alpha", &factory);
    let mut cursor = started(harness.as_ref(), "session-host").await;

    let error = harness
        .send(HarnessInput::text("early"), DeliveryMode::Steer)
        .await
        .expect_err("there is no turn to steer");
    assert_eq!(error.kind(), HarnessErrorKind::State);
    assert_eq!(error.message(), "there is no active turn to steer");

    timeout(
        BOUND,
        harness.send(HarnessInput::text("work"), DeliveryMode::FollowUp),
    )
    .await
    .expect("send finished within the bound")
    .expect("the input was accepted");
    read_until(&mut cursor, |event| {
        matches!(
            &event.event,
            HarnessEventPayload::TurnLifecycle(TurnLifecycleEvent {
                kind: TurnEventKind::Started,
                ..
            })
        )
    })
    .await;

    timeout(
        BOUND,
        harness.send(HarnessInput::text("steered"), DeliveryMode::Steer),
    )
    .await
    .expect("steer finished within the bound")
    .expect("the harness accepted the steer");
    assert_eq!(
        factory.harness(AGENT_ID).sends(),
        vec!["work".to_owned(), "steered".to_owned()]
    );

    timeout(BOUND, harness.abort(AbortMode::Force))
        .await
        .expect("abort finished within the bound")
        .expect("the harness accepted the abort");
    read_until(&mut cursor, is_terminal).await;
    timeout(BOUND, harness.stop())
        .await
        .expect("stop bounded")
        .expect("the harness stopped");
}

#[tokio::test]
async fn tool_chunks_become_the_io_adapter_item_lifecycle() {
    let factory = FakeHarnessFactory::new();
    factory.hanging_for(AGENT_ID);
    let harness = harness_for("tools-alpha", &factory);
    let mut cursor = started(harness.as_ref(), "session-host").await;
    timeout(
        BOUND,
        harness.send(HarnessInput::text("run a tool"), DeliveryMode::FollowUp),
    )
    .await
    .expect("send finished within the bound")
    .expect("the input was accepted");
    read_until(&mut cursor, |event| {
        matches!(
            &event.event,
            HarnessEventPayload::TurnLifecycle(TurnLifecycleEvent {
                kind: TurnEventKind::Started,
                ..
            })
        )
    })
    .await;

    let fake = factory.harness(AGENT_ID);
    let mut started_data = JsonObject::new();
    started_data.insert("name".to_owned(), json!("Bash"));
    started_data.insert("arguments".to_owned(), json!({"cmd": "ls"}));
    started_data.insert("tool_call_id".to_owned(), json!("call-1"));
    fake.emit_tool(ItemEventKind::Started, started_data);
    let mut completed_data = JsonObject::new();
    completed_data.insert("tool_name".to_owned(), json!("Bash"));
    completed_data.insert("result".to_owned(), json!("ok"));
    completed_data.insert("tool_call_id".to_owned(), json!("call-1"));
    fake.emit_tool(ItemEventKind::Completed, completed_data);
    fake.emit_answer("tool done");
    fake.emit_turn(TurnEventKind::Finished, Some(completed_result()));
    fake.emit_state(HarnessState::Running, HarnessState::Idle);

    let events = read_until(&mut cursor, is_terminal).await;
    let items: Vec<&ItemLifecycleEvent> = events
        .iter()
        .filter_map(|event| match &event.event {
            HarnessEventPayload::ItemLifecycle(item) => Some(item),
            _ => None,
        })
        .collect();
    assert_eq!(
        items.len(),
        2,
        "one started and one completed item: {events:?}"
    );
    assert_eq!(items[0].kind, ItemEventKind::Started);
    assert_eq!(items[0].item_type, "tool_call");
    assert_eq!(items[0].data["name"], json!("Bash"));
    assert_eq!(items[0].data["arguments"], json!({"cmd": "ls"}));
    assert_eq!(items[0].data["tool_call_id"], json!("call-1"));
    assert_eq!(items[1].kind, ItemEventKind::Completed);
    assert_eq!(items[1].item_type, "tool_call");
    assert_eq!(items[1].data["tool_name"], json!("Bash"));
    assert_eq!(items[1].data["result"], json!("ok"));
    assert_eq!(terminal_result(&events).final_output, json!("tool done"));
    timeout(BOUND, harness.stop())
        .await
        .expect("stop bounded")
        .expect("the harness stopped");
}

#[test]
fn the_create_mapping_accepts_a_validated_blueprint() {
    let config: TeamAgentSpec =
        serde_json::from_value(team_json("mapping-alpha")).expect("the blueprint parses");
    config.validate().expect("the blueprint is valid");
    assert_eq!(config.team_name, "mapping-alpha");
    let decoded =
        NativeV2Config::from_json(&create_mapping("mapping-beta")).expect("the mapping decodes");
    assert_eq!(decoded.team.team_name, "mapping-beta");
    assert_eq!(decoded.event_buffer_capacity, 1024);
    assert!(decoded.session_id.is_none() && decoded.language.is_none());
}
