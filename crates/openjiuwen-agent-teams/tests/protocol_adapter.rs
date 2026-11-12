//! Harness-protocol adapter behaviour, through the public API.
//!
//! The suite pins the five contracts the team runtime relies on when a member's brain is an
//! `openjiuwen_harness_protocol::HarnessProtocol`:
//!
//! 1. [`HarnessMemberRuntime`] forwards every command to the protocol handle with the documented
//!    argument mapping (steer/follow-up, force/graceful, pause/resume, the settled state on
//!    `start`, and a `stop` that leaves no pump behind);
//! 2. the observation pump dispatches `on_state` / `on_round` / `on_chunk` to a registered
//!    [`MemberObserver`] in event order, with team-stream chunk frames;
//! 3. the two pure state tables and the event projections in `harness::mapping` match the rows
//!    documented in `src/harness/CONTEXT.md`;
//! 4. [`NativeCheckpoint`] round-trips through JSON, tolerates unknown fields, rejects a missing
//!    identity field and refuses to be restored into another session or card;
//! 5. `TeamAgent::state()` mirrors the leader harness verbatim.
//!
//! Everything is offline: the shared `support::ScriptedHarness` answers every input with one
//! scripted observation triple, and no model, network or clock is involved. Every wait is bounded
//! by [`support::BOUND`].
//!
//! Delivery modes are not observable through the shared fake: `ScriptedHarness::send` records
//! `Call::Send(<text>)` and drops the mode, and `HarnessMemberRuntime::send` discards the
//! `SendReceipt` whose `accepted_mode` carries it. The [`ModeRecorder`] decorator below is
//! therefore a local, private wrapper around the scripted harness; it is the only thing this file
//! adds to the fakes.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![allow(clippy::missing_panics_doc)]

mod support;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_agent_teams::harness::{
    chunk_from_event, is_terminal_event, round_from_event, state_from_event,
    state_to_execution_status, state_to_member_status, HarnessMemberRuntime, NativeCheckpoint,
    CHUNK_LLM_OUTPUT, CHUNK_LLM_REASONING, CHUNK_TOOL_CALL, CHUNK_TOOL_RESULT,
};
use openjiuwen_agent_teams::member::{MemberObserver, MemberRuntime};
use openjiuwen_agent_teams::schema::{ExecutionStatus, MemberStatus, RoundOutcome};
use openjiuwen_common::error::ErrorKind;
use openjiuwen_harness_protocol::{
    AbortMode, DeliveryMode, EventBufferConfig, HarnessCard, HarnessCheckpoint, HarnessContext,
    HarnessEvent, HarnessEventCursor, HarnessInput, HarnessProtocol, HarnessResult, HarnessState,
    ItemEventKind, ItemLifecycleEvent, OutputChannel, OutputEvent, OutputKind, OutputOperation,
    SendReceipt, StateChangedEvent, TurnEventKind, TurnLifecycleEvent, TurnUsage, UsageUpdateMode,
    UsageUpdatedEvent,
};
use parking_lot::Mutex;
use serde_json::{json, Value};

use support::{sample_spec, wait_until, Call, Fixture, ScriptedHarness};

/// Host session id every hand-built event and the simulated runtime run under.
const SESSION: &str = "session-1";

/// Agent id every hand-built event names.
const AGENT: &str = "team_leader";

/// Bound of the "nothing more arrives" probes; far inside [`support::BOUND`].
const QUIET: Duration = Duration::from_millis(100);

// ---------------------------------------------------------------------------------------------
// Recording observer
// ---------------------------------------------------------------------------------------------

/// One callback a [`RecordingObserver`] received.
#[derive(Debug, Clone, PartialEq)]
enum Callback {
    /// `MemberObserver::on_state`.
    State(HarnessState),
    /// `MemberObserver::on_round`.
    Round(RoundOutcome),
    /// `MemberObserver::on_chunk`.
    Chunk(Value),
}

/// [`MemberObserver`] that records every callback, in arrival order.
#[derive(Default)]
struct RecordingObserver {
    /// Callbacks received so far, in arrival order.
    callbacks: Mutex<Vec<Callback>>,
}

impl RecordingObserver {
    /// Builds an empty observer behind the `Arc` the runtime seam wants.
    fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Returns the recorded callbacks, in arrival order.
    fn callbacks(&self) -> Vec<Callback> {
        self.callbacks.lock().clone()
    }

    /// Returns how many callbacks have arrived.
    fn len(&self) -> usize {
        self.callbacks.lock().len()
    }

    /// Returns the recorded states, in arrival order.
    fn states(&self) -> Vec<HarnessState> {
        self.callbacks()
            .into_iter()
            .filter_map(|callback| match callback {
                Callback::State(state) => Some(state),
                Callback::Round(_) | Callback::Chunk(_) => None,
            })
            .collect()
    }

    /// Returns the recorded round outcomes, in arrival order.
    fn rounds(&self) -> Vec<RoundOutcome> {
        self.callbacks()
            .into_iter()
            .filter_map(|callback| match callback {
                Callback::Round(outcome) => Some(outcome),
                Callback::State(_) | Callback::Chunk(_) => None,
            })
            .collect()
    }

    /// Returns the recorded chunks, in arrival order.
    fn chunks(&self) -> Vec<Value> {
        self.callbacks()
            .into_iter()
            .filter_map(|callback| match callback {
                Callback::Chunk(chunk) => Some(chunk),
                Callback::State(_) | Callback::Round(_) => None,
            })
            .collect()
    }

    /// Renders every callback as `<kind>:<label>`, so one whole sequence is one assertion.
    fn labels(&self) -> Vec<String> {
        self.callbacks()
            .iter()
            .map(|callback| match callback {
                Callback::State(state) => format!("state:{}", state_label(*state)),
                Callback::Round(outcome) => format!("round:{}", round_label(*outcome)),
                Callback::Chunk(chunk) => format!(
                    "chunk:{}",
                    chunk
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or("<untyped>")
                ),
            })
            .collect()
    }
}

impl MemberObserver for RecordingObserver {
    fn on_state(&self, state: HarnessState) {
        self.callbacks.lock().push(Callback::State(state));
    }

    fn on_round(&self, outcome: RoundOutcome) {
        self.callbacks.lock().push(Callback::Round(outcome));
    }

    fn on_chunk(&self, chunk: Value) {
        self.callbacks.lock().push(Callback::Chunk(chunk));
    }
}

/// Returns the wire label of `state`, for sequence assertions.
fn state_label(state: HarnessState) -> &'static str {
    match state {
        HarnessState::Idle => "idle",
        HarnessState::Running => "running",
        HarnessState::Pausing => "pausing",
        HarnessState::Paused => "paused",
        HarnessState::Terminated => "terminated",
    }
}

/// Returns the wire label of `outcome`, for sequence assertions.
fn round_label(outcome: RoundOutcome) -> &'static str {
    match outcome {
        RoundOutcome::Started => "started",
        RoundOutcome::Finished => "finished",
        RoundOutcome::Aborted => "aborted",
        RoundOutcome::Failed => "failed",
    }
}

// ---------------------------------------------------------------------------------------------
// Mode-recording decorator over the shared fake
// ---------------------------------------------------------------------------------------------

/// [`ScriptedHarness`] decorator that also records the delivery mode and resume query of every
/// input.
///
/// The shared fake only logs the *text* of a delivered input and discards the `SendReceipt`
/// `accepted_mode`, so steer-vs-follow-up cannot be asserted through `support`; this wrapper adds
/// exactly that log and delegates everything else (including the observation cursor) to the
/// scripted harness.
struct ModeRecorder {
    /// The scripted harness everything is delegated to.
    inner: Arc<ScriptedHarness>,
    /// Delivery modes of the inputs received, in call order.
    deliveries: Mutex<Vec<DeliveryMode>>,
    /// Resume queries received, in call order (`None` = no query).
    resumes: Mutex<Vec<Option<String>>>,
}

impl ModeRecorder {
    /// Wraps `inner` in a mode-recording decorator.
    fn new(inner: Arc<ScriptedHarness>) -> Arc<Self> {
        Arc::new(Self {
            inner,
            deliveries: Mutex::new(Vec::new()),
            resumes: Mutex::new(Vec::new()),
        })
    }

    /// Returns the delivery modes received, in call order.
    fn deliveries(&self) -> Vec<DeliveryMode> {
        self.deliveries.lock().clone()
    }

    /// Returns the resume queries received, in call order.
    fn resumes(&self) -> Vec<Option<String>> {
        self.resumes.lock().clone()
    }
}

#[async_trait]
impl HarnessProtocol for ModeRecorder {
    fn card(&self) -> &HarnessCard {
        self.inner.card()
    }

    fn state(&self) -> HarnessState {
        self.inner.state()
    }

    fn provider_session_id(&self) -> Option<&str> {
        None
    }

    fn event_buffer_config(&self) -> &EventBufferConfig {
        self.inner.event_buffer_config()
    }

    async fn start(&self, context: HarnessContext) -> HarnessResult<()> {
        self.inner.start(context).await
    }

    async fn stop(&self) -> HarnessResult<()> {
        self.inner.stop().await
    }

    fn events(&self) -> Box<dyn HarnessEventCursor> {
        self.inner.events()
    }

    fn turn_events(&self, turn_id: Option<&str>) -> Box<dyn HarnessEventCursor> {
        self.inner.turn_events(turn_id)
    }

    async fn send(&self, input: HarnessInput, mode: DeliveryMode) -> HarnessResult<SendReceipt> {
        self.deliveries.lock().push(mode);
        self.inner.send(input, mode).await
    }

    async fn abort(&self, mode: AbortMode) -> HarnessResult<()> {
        self.inner.abort(mode).await
    }

    async fn pause(&self) -> HarnessResult<()> {
        self.inner.pause().await
    }

    async fn resume(&self, query: Option<HarnessInput>) -> HarnessResult<()> {
        self.resumes
            .lock()
            .push(query.and_then(|input| input.content.as_str().map(ToOwned::to_owned)));
        // The recording above consumes the query, so the scripted harness is resumed without it;
        // the scripted harness ignores the payload and only settles its state.
        self.inner.resume(None).await
    }

    async fn export_checkpoint(&self) -> HarnessResult<Option<HarnessCheckpoint>> {
        self.inner.export_checkpoint().await
    }
}

// ---------------------------------------------------------------------------------------------
// Rig
// ---------------------------------------------------------------------------------------------

/// One scripted harness, the adapter driving it, and an observer registered before `start`.
struct Rig {
    /// The scripted fake, for command assertions.
    harness: Arc<ScriptedHarness>,
    /// The mode-recording decorator the adapter drives.
    recorder: Arc<ModeRecorder>,
    /// The adapter under test.
    runtime: HarnessMemberRuntime,
    /// Observer registered through the seam before `start`.
    observer: Arc<RecordingObserver>,
}

impl Rig {
    /// Builds a rig over a harness that answers every input with one settled round.
    fn new(answer: &str) -> Self {
        Self::over(ScriptedHarness::new(answer))
    }

    /// Builds a rig over a harness whose inputs stay `Running` and emit nothing.
    fn hanging(answer: &str) -> Self {
        Self::over(ScriptedHarness::hanging(answer))
    }

    /// Wraps `harness` in the decorator, the adapter and a recording observer.
    fn over(harness: Arc<ScriptedHarness>) -> Self {
        let recorder = ModeRecorder::new(Arc::clone(&harness));
        let runtime =
            HarnessMemberRuntime::new(Arc::clone(&recorder) as Arc<dyn HarnessProtocol>, SESSION);
        let observer = RecordingObserver::new();
        runtime.subscribe(Arc::clone(&observer) as Arc<dyn MemberObserver>);
        Self {
            harness,
            recorder,
            runtime,
            observer,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Event builders (pure projections)
// ---------------------------------------------------------------------------------------------

/// Builds a `state_changed` observation from `old` to `new`.
fn state_event(old: HarnessState, new: HarnessState) -> HarnessEvent {
    HarnessEvent::new(1, 1.0, StateChangedEvent { old, new }, SESSION, AGENT)
}

/// Builds a `turn_lifecycle` observation of `kind` on turn `turn-1`.
///
/// Non-terminal kinds carry no result, matching the protocol's own rule.
fn turn_event(kind: TurnEventKind) -> HarnessEvent {
    HarnessEvent::new(
        2,
        2.0,
        TurnLifecycleEvent { kind, result: None },
        SESSION,
        AGENT,
    )
    .with_turn_id("turn-1")
}

/// Builds an `output` observation on `channel`.
fn output_event(channel: OutputChannel) -> HarnessEvent {
    HarnessEvent::new(
        3,
        3.0,
        OutputEvent {
            output_id: "out-1".to_owned(),
            kind: OutputKind::Text,
            content: json!("hello"),
            operation: OutputOperation::Delta,
            channel,
            content_index: 0,
            content_type: None,
            data: BTreeMap::new(),
        },
        SESSION,
        AGENT,
    )
    .with_turn_id("turn-1")
}

/// Builds an `item_lifecycle` observation of `kind` for an item of `item_type`.
fn item_event(kind: ItemEventKind, item_type: &str) -> HarnessEvent {
    HarnessEvent::new(
        4,
        4.0,
        ItemLifecycleEvent {
            kind,
            item_type: item_type.to_owned(),
            data: BTreeMap::new(),
        },
        SESSION,
        AGENT,
    )
    .with_item_id("call-1")
}

/// Builds a `usage_updated` observation.
fn usage_event() -> HarnessEvent {
    HarnessEvent::new(
        5,
        5.0,
        UsageUpdatedEvent {
            usage: TurnUsage::default(),
            mode: UsageUpdateMode::Cumulative,
        },
        SESSION,
        AGENT,
    )
    .with_turn_id("turn-1")
}

// ---------------------------------------------------------------------------------------------
// 1. Command forwarding
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn start_opens_the_cycle_and_publishes_the_settled_state() {
    let rig = Rig::new("done");
    assert_eq!(rig.runtime.session_id(), SESSION);
    rig.runtime.start(None).await.unwrap();
    assert_eq!(rig.harness.calls(), vec![Call::Start]);
    assert_eq!(rig.runtime.state(), HarnessState::Idle);
    // `start` publishes the settled state before the pump exists, so the observer sees it without
    // waiting for an observation.
    assert_eq!(rig.observer.states(), vec![HarnessState::Idle]);
    assert_eq!(rig.observer.len(), 1);
    rig.runtime.stop().await.unwrap();
}

#[tokio::test]
async fn send_forwards_the_text_and_the_delivery_mode() {
    let rig = Rig::new("ack");
    rig.runtime.start(None).await.unwrap();
    rig.runtime.send("hi".to_owned(), true).await.unwrap();
    rig.runtime.send("later".to_owned(), false).await.unwrap();
    assert_eq!(
        rig.harness.sends(),
        vec!["hi".to_owned(), "later".to_owned()]
    );
    assert_eq!(
        rig.recorder.deliveries(),
        vec![DeliveryMode::Steer, DeliveryMode::FollowUp],
        "immediate steers, queued follows up"
    );
    // Each scripted input produces one settled round (started, output, finished).
    wait_until("two settled scripted rounds", || {
        rig.observer.rounds().len() >= 4
    })
    .await;
    assert_eq!(
        rig.observer.rounds(),
        vec![
            RoundOutcome::Started,
            RoundOutcome::Finished,
            RoundOutcome::Started,
            RoundOutcome::Finished,
        ]
    );
    rig.runtime.stop().await.unwrap();
}

#[tokio::test]
async fn abort_forwards_force_and_graceful() {
    let rig = Rig::new("ack");
    rig.runtime.start(None).await.unwrap();
    rig.runtime.abort(true).await.unwrap();
    rig.runtime.abort(false).await.unwrap();
    assert_eq!(
        rig.harness.abort_modes(),
        vec![AbortMode::Force, AbortMode::Graceful]
    );
    rig.runtime.stop().await.unwrap();
}

#[tokio::test]
async fn pause_and_resume_forward_and_settle_the_state() {
    let rig = Rig::new("ack");
    rig.runtime.start(None).await.unwrap();
    rig.runtime.pause().await.unwrap();
    assert_eq!(rig.runtime.state(), HarnessState::Paused);
    rig.runtime.resume(Some("q".to_owned())).await.unwrap();
    assert_eq!(rig.runtime.state(), HarnessState::Running);
    rig.runtime.resume(None).await.unwrap();
    assert_eq!(rig.runtime.state(), HarnessState::Running);
    assert_eq!(
        rig.harness.calls(),
        vec![Call::Start, Call::Pause, Call::Resume, Call::Resume]
    );
    assert_eq!(
        rig.recorder.resumes(),
        vec![Some("q".to_owned()), None],
        "the continuation query is forwarded verbatim, `None` stays `None`"
    );
    rig.runtime.stop().await.unwrap();
}

#[tokio::test]
async fn stop_terminates_the_cycle_and_leaves_no_pump() {
    let rig = Rig::new("ack");
    rig.runtime.start(None).await.unwrap();
    rig.runtime.stop().await.unwrap();
    assert_eq!(rig.harness.calls(), vec![Call::Start, Call::Stop]);
    assert_eq!(rig.harness.state_now(), HarnessState::Terminated);
    assert_eq!(rig.runtime.state(), HarnessState::Terminated);
    assert_eq!(
        rig.observer.states(),
        vec![HarnessState::Idle, HarnessState::Terminated]
    );
    // A surviving pump would deliver this answer as a chunk; a cancelled one dropped its cursor,
    // so nothing may arrive inside the quiet bound.
    rig.harness.emit_answer("x");
    let delivered = tokio::time::timeout(
        QUIET,
        wait_until("a chunk after stop", || !rig.observer.chunks().is_empty()),
    )
    .await;
    assert!(
        delivered.is_err(),
        "the pump delivered a chunk after stop: {:?}",
        rig.observer.chunks()
    );
    assert_eq!(rig.observer.chunks(), Vec::<Value>::new());
    assert_eq!(
        rig.observer.labels(),
        vec!["state:idle".to_owned(), "state:terminated".to_owned()]
    );
}

// ---------------------------------------------------------------------------------------------
// 2. Observer callbacks
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn observer_receives_states_rounds_and_chunks_in_order() {
    let rig = Rig::hanging("hi");
    rig.runtime.start(None).await.unwrap();
    // The hanging scripted harness emits nothing on `send`, so the sequence below is exactly what
    // this test publishes, in publication order.
    rig.harness
        .emit(state_event(HarnessState::Idle, HarnessState::Running));
    rig.harness.emit_turn(TurnEventKind::Started);
    rig.harness.emit_answer("hi");
    rig.harness.emit_turn(TurnEventKind::Finished);
    rig.harness
        .emit(state_event(HarnessState::Running, HarnessState::Idle));
    rig.harness
        .emit(state_event(HarnessState::Idle, HarnessState::Paused));
    wait_until("the six scripted observations", || rig.observer.len() >= 7).await;
    assert_eq!(
        rig.observer.labels(),
        vec![
            "state:idle".to_owned(),
            "state:running".to_owned(),
            "round:started".to_owned(),
            "chunk:llm_output".to_owned(),
            "round:finished".to_owned(),
            "state:idle".to_owned(),
            "state:paused".to_owned(),
        ],
        "one event must project to at most one callback, in event order"
    );
    rig.runtime.stop().await.unwrap();
    assert_eq!(
        rig.observer.labels().last().map(String::as_str),
        Some("state:terminated")
    );
    assert_eq!(
        rig.observer.states(),
        vec![
            HarnessState::Idle,
            HarnessState::Running,
            HarnessState::Idle,
            HarnessState::Paused,
            HarnessState::Terminated,
        ]
    );
    // The chunk keeps the team-stream frame shape `{"type", "payload"}` with the output model as
    // the payload.
    let chunks = rig.observer.chunks();
    assert_eq!(chunks.len(), 1);
    let chunk = &chunks[0];
    assert_eq!(chunk["type"], json!(CHUNK_LLM_OUTPUT));
    assert_eq!(chunk["payload"]["content"], json!("hi"));
    assert_eq!(chunk["payload"]["channel"], json!("answer"));
    assert_eq!(chunk.as_object().unwrap().len(), 2);
    assert_eq!(
        rig.observer.rounds(),
        vec![RoundOutcome::Started, RoundOutcome::Finished]
    );
}

// ---------------------------------------------------------------------------------------------
// 3. State tables
// ---------------------------------------------------------------------------------------------

#[test]
fn state_tables_match_the_documented_rows() {
    // Every `HarnessState` maps to a member status; the execution status is `None` only for a gone
    // runtime.
    let table = [
        (
            HarnessState::Idle,
            MemberStatus::Ready,
            Some(ExecutionStatus::Idle),
        ),
        (
            HarnessState::Running,
            MemberStatus::Busy,
            Some(ExecutionStatus::Running),
        ),
        (
            HarnessState::Pausing,
            MemberStatus::Paused,
            Some(ExecutionStatus::CancelRequested),
        ),
        (
            HarnessState::Paused,
            MemberStatus::Paused,
            Some(ExecutionStatus::Idle),
        ),
        (HarnessState::Terminated, MemberStatus::Stopped, None),
    ];
    for (state, member, execution) in table {
        assert_eq!(
            state_to_member_status(state),
            Some(member),
            "member status of {}",
            state_label(state)
        );
        assert_eq!(
            state_to_execution_status(state),
            execution,
            "execution status of {}",
            state_label(state)
        );
    }
    // `Paused` is suspended, not cancelled: a resumed round must not have to leave a terminal
    // status first.
    assert_ne!(
        state_to_execution_status(HarnessState::Paused),
        Some(ExecutionStatus::Cancelled)
    );
    // `Terminated` keeps the row revivable as `Stopped`, never terminal `ShutDown`.
    assert_ne!(
        state_to_member_status(HarnessState::Terminated),
        Some(MemberStatus::ShutDown)
    );
}

// ---------------------------------------------------------------------------------------------
// 4. Event projections
// ---------------------------------------------------------------------------------------------

#[test]
fn state_from_event_projects_the_entered_state() {
    let running = state_event(HarnessState::Idle, HarnessState::Running);
    assert_eq!(state_from_event(&running), Some(HarnessState::Running));
    let paused = state_event(HarnessState::Running, HarnessState::Paused);
    assert_eq!(state_from_event(&paused), Some(HarnessState::Paused));
    let terminated = state_event(HarnessState::Idle, HarnessState::Terminated);
    assert_eq!(
        state_from_event(&terminated),
        Some(HarnessState::Terminated)
    );
    assert_eq!(state_from_event(&turn_event(TurnEventKind::Started)), None);
    assert_eq!(state_from_event(&output_event(OutputChannel::Answer)), None);
    assert_eq!(state_from_event(&usage_event()), None);
}

#[test]
fn round_from_event_covers_every_turn_kind() {
    assert_eq!(
        round_from_event(&turn_event(TurnEventKind::Started)),
        Some(RoundOutcome::Started)
    );
    assert_eq!(
        round_from_event(&turn_event(TurnEventKind::Finished)),
        Some(RoundOutcome::Finished)
    );
    assert_eq!(
        round_from_event(&turn_event(TurnEventKind::Aborted)),
        Some(RoundOutcome::Aborted)
    );
    assert_eq!(
        round_from_event(&turn_event(TurnEventKind::Failed)),
        Some(RoundOutcome::Failed)
    );
    // `paused`/`resumed` do not close a round.
    assert_eq!(round_from_event(&turn_event(TurnEventKind::Paused)), None);
    assert_eq!(round_from_event(&turn_event(TurnEventKind::Resumed)), None);
    // Neither does anything that is not a turn lifecycle.
    assert_eq!(
        round_from_event(&state_event(HarnessState::Idle, HarnessState::Running)),
        None
    );
    assert_eq!(round_from_event(&output_event(OutputChannel::Answer)), None);
    assert_eq!(round_from_event(&usage_event()), None);
    assert_eq!(
        round_from_event(&item_event(ItemEventKind::Started, "tool")),
        None
    );
}

#[test]
fn terminal_events_are_turn_lifecycle_only() {
    for kind in [
        TurnEventKind::Finished,
        TurnEventKind::Aborted,
        TurnEventKind::Failed,
    ] {
        assert!(
            is_terminal_event(&turn_event(kind)),
            "{kind:?} closes a turn"
        );
    }
    for kind in [
        TurnEventKind::Started,
        TurnEventKind::Paused,
        TurnEventKind::Resumed,
    ] {
        assert!(
            !is_terminal_event(&turn_event(kind)),
            "{kind:?} keeps a turn open"
        );
    }
    // Only turn lifecycle counts: a `terminated` state change and any output are not turn ends.
    assert!(!is_terminal_event(&state_event(
        HarnessState::Idle,
        HarnessState::Terminated
    )));
    assert!(!is_terminal_event(&output_event(OutputChannel::Answer)));
    assert!(!is_terminal_event(&usage_event()));
}

#[test]
fn chunk_from_event_projects_output_and_tool_lifecycles() {
    let answer = chunk_from_event(&output_event(OutputChannel::Answer)).unwrap();
    assert_eq!(answer["type"], json!(CHUNK_LLM_OUTPUT));
    assert_eq!(answer["payload"]["output_id"], json!("out-1"));
    assert_eq!(answer["payload"]["content"], json!("hello"));
    assert_eq!(answer.as_object().unwrap().len(), 2);

    let system = chunk_from_event(&output_event(OutputChannel::System)).unwrap();
    assert_eq!(system["type"], json!(CHUNK_LLM_OUTPUT));
    assert_eq!(system["payload"]["channel"], json!("system"));

    let reasoning = chunk_from_event(&output_event(OutputChannel::Reasoning)).unwrap();
    assert_eq!(reasoning["type"], json!(CHUNK_LLM_REASONING));
    assert_eq!(reasoning["payload"]["channel"], json!("reasoning"));

    let called = chunk_from_event(&item_event(ItemEventKind::Started, "tool")).unwrap();
    assert_eq!(called["type"], json!(CHUNK_TOOL_CALL));
    assert_eq!(called["payload"]["item_id"], json!("call-1"));
    assert_eq!(called["payload"]["item_type"], json!("tool"));
    assert_eq!(called["payload"]["kind"], json!("started"));

    let result = chunk_from_event(&item_event(ItemEventKind::Completed, "tool")).unwrap();
    assert_eq!(result["type"], json!(CHUNK_TOOL_RESULT));
    assert_eq!(result["payload"]["item_id"], json!("call-1"));
    assert_eq!(result["payload"]["kind"], json!("completed"));

    // An updated tool item, a non-tool item and everything else carry no team chunk.
    assert_eq!(
        chunk_from_event(&item_event(ItemEventKind::Updated, "tool")),
        None
    );
    assert_eq!(
        chunk_from_event(&item_event(ItemEventKind::Started, "message")),
        None
    );
    assert_eq!(chunk_from_event(&turn_event(TurnEventKind::Started)), None);
    assert_eq!(chunk_from_event(&usage_event()), None);
    assert_eq!(
        chunk_from_event(&state_event(HarnessState::Idle, HarnessState::Running)),
        None
    );
}

// ---------------------------------------------------------------------------------------------
// 5. NativeCheckpoint
// ---------------------------------------------------------------------------------------------

#[test]
fn checkpoint_round_trips_validates_and_tolerates_unknown_fields() {
    let checkpoint = NativeCheckpoint::new(SESSION, "alpha_team_leader")
        .with_cwd("/work/team_leader")
        .with_state(HarnessState::Paused)
        .with_turn_id("turn-7")
        .with_outputs(json!({"answer_parts": ["hi"], "tool_blocks": []}));
    let encoded = checkpoint.to_json().unwrap();
    // The wire contract is snake_case with exactly these keys.
    let mut keys: Vec<&str> = encoded
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        vec![
            "card_id",
            "cwd",
            "outputs",
            "session_id",
            "state",
            "turn_id"
        ]
    );
    assert_eq!(encoded["session_id"], json!(SESSION));
    assert_eq!(encoded["card_id"], json!("alpha_team_leader"));
    assert_eq!(encoded["cwd"], json!("/work/team_leader"));
    assert_eq!(encoded["state"], json!("paused"));
    assert_eq!(encoded["turn_id"], json!("turn-7"));
    assert_eq!(encoded["outputs"]["answer_parts"], json!(["hi"]));
    assert_eq!(NativeCheckpoint::from_json(encoded).unwrap(), checkpoint);

    // Unknown fields are tolerated and the optional fields fall back to their defaults.
    let tolerant = NativeCheckpoint::from_json(json!({
        "session_id": SESSION,
        "card_id": "alpha_team_leader",
        "legacy_field": {"any": "thing"},
        "unknown_state_hint": 7,
    }))
    .unwrap();
    assert_eq!(tolerant.session_id, SESSION);
    assert_eq!(tolerant.card_id, "alpha_team_leader");
    assert_eq!(tolerant.cwd, None);
    assert_eq!(tolerant.state, HarnessState::Idle);
    assert_eq!(tolerant.turn_id, None);
    assert_eq!(tolerant.outputs, json!({}));

    // A missing identity field is a validation error naming the document.
    for document in [
        json!({"card_id": "alpha_team_leader"}),
        json!({"session_id": SESSION}),
    ] {
        let error = NativeCheckpoint::from_json(document).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.to_string().contains("invalid native checkpoint"),
            "unexpected message: {error}"
        );
    }

    // Restoring into the same session and card is allowed. Either mismatch is a validation error
    // whose Python-compatible wording names the id the checkpoint was taken under, so an operator
    // can tell a session change from a card change.
    checkpoint
        .validate_compatible(SESSION, "alpha_team_leader")
        .unwrap();
    let session_error = checkpoint
        .validate_compatible("session-2", "alpha_team_leader")
        .unwrap_err();
    assert_eq!(session_error.kind(), ErrorKind::Validation);
    let message = session_error.to_string();
    assert!(
        message.contains("checkpoint belongs to session 'session-1'"),
        "{message}"
    );
    assert!(!message.contains("card"), "{message}");

    let card_error = checkpoint
        .validate_compatible(SESSION, "beta_team_leader")
        .unwrap_err();
    assert_eq!(card_error.kind(), ErrorKind::Validation);
    let message = card_error.to_string();
    assert!(
        message.contains("checkpoint belongs to card 'alpha_team_leader'"),
        "{message}"
    );
    assert!(!message.contains("session"), "{message}");
}

// ---------------------------------------------------------------------------------------------
// 6. TeamAgent state follows the leader harness
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn team_state_follows_the_leader_harness() {
    let fixture = Fixture::new();
    let agent = fixture.activate(&sample_spec(), SESSION).await;
    assert_eq!(agent.state(), HarnessState::Idle, "before the first start");
    let leader = fixture.harness("team_leader");

    agent.start().await.unwrap();
    assert_eq!(
        agent.state(),
        HarnessState::Idle,
        "a started harness settles in idle"
    );

    leader.set_state(HarnessState::Running);
    assert_eq!(agent.state(), HarnessState::Running);
    leader.set_state(HarnessState::Paused);
    assert_eq!(agent.state(), HarnessState::Paused);

    agent.stop().await.unwrap();
    assert_eq!(agent.state(), HarnessState::Terminated);
    assert_eq!(leader.state_now(), HarnessState::Terminated);
    assert!(leader.calls().contains(&Call::Stop));
}
