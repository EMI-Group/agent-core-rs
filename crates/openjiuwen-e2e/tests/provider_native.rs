#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::needless_pass_by_value,
    clippy::similar_names
)]
//! Offline, deterministic cross-crate scenario for the provider-neutral harness SPI.
//!
//! The binary drives the three seams that connect a host to a harness provider:
//!
//! 1. **The registry** ([`available_providers`], [`describe_provider`], [`provider_descriptors`])
//!    answers the declared provider names and describes the implemented one.
//! 2. **The factory** ([`create_harness`]) refuses an unknown name with [`Error::NotFound`] and a
//!    declared-but-deferred vendor name with [`Error::Unsupported`], and builds an unstarted
//!    native harness from a JSON configuration.
//! 3. **The lifecycle and the IO adapter** are exercised end to end: a `NativeHarness` over a
//!    scripted model (direct model injection) and over a loopback [`MockLlmServer`] (the factory
//!    resolves the model lazily from the create config), plus a hand-built harness double pushed
//!    through [`HarnessIOAdapter`] to check the event → chunk projection.
//!
//! Every scenario is offline and deterministic: a `ScriptedModel` or a loopback `MockLlmServer`
//! bound to an ephemeral port, temp workspaces, and no credentials. Nothing sleeps to order
//! events: a turn is drained until both the terminal `TurnLifecycle{Finished}` and the
//! `Running → Idle` state change are observed, and every await that could hang is bounded by
//! [`support::BOUND`]. No assertion depends on a timestamp, `duration_ms`, or a generated id
//! beyond the documented `message-1`/`turn-1` counters.

mod support;

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen::core::common::error::Error;
use openjiuwen::core::foundation::llm::Model;
use openjiuwen::core::single_agent::output::{LLM_OUTPUT, LLM_REASONING};
use openjiuwen::core::single_agent::{AgentCard, AgentChunk};
use openjiuwen::harness::{
    DeepAgentBuilder, DeepAgentConfig, NativeHarness, NativeHarnessOptions, Workspace,
};
use openjiuwen::harness_protocol::{
    json_object_from_value, AbortMode, DeliveryMode, DynHarness, EventBufferConfig, HarnessCard,
    HarnessCheckpoint, HarnessContext, HarnessErrorKind, HarnessEvent, HarnessEventCursor,
    HarnessEventPayload, HarnessInput, HarnessProtocol, HarnessResult, HarnessState, ItemEventKind,
    ItemLifecycleEvent, JsonObject, JsonValue, OutputChannel, OutputEvent, OutputKind,
    OutputOperation, SendReceipt, TurnEventKind, TurnLifecycleEvent, TurnResult, TurnStatus,
    VecEventCursor,
};
use openjiuwen::harness_providers::io_adapter::{HarnessIOAdapter, HarnessIOAdapterConfig};
use openjiuwen::harness_providers::{
    available_providers, create_harness, describe_provider, provider_descriptors, PROVIDER_NAMES,
    TURN_LIFECYCLE_KIND,
};
use openjiuwen_test_support::{MockLlmServer, ScriptedModel, ScriptedResponse, TEST_API_KEY};
use serde_json::json;
use support::within;
use tempfile::TempDir;
use tokio::sync::mpsc;

/// Upper bound on the events one turn may produce before the drain gives up.
///
/// A hang is already caught by [`support::BOUND`]; this bound turns a runaway producer into a
/// failure with a readable message instead of an ever-growing vector.
const MAX_TURN_EVENTS: usize = 256;

/// Number of events the projection scenario pushes (five projected plus the finished turn).
const PROJECTION_CHUNKS: usize = 5;

// ---------------------------------------------------------------------------
// Shared fixtures
// ---------------------------------------------------------------------------

/// The standard host context of the scenario.
fn context() -> HarnessContext {
    HarnessContext::new(
        "deep_agent",
        "deep_agent",
        "host-session-1",
        "You are a test agent.",
    )
}

/// Converts a JSON object value into the protocol's ordered object type.
fn object(value: JsonValue) -> JsonObject {
    json_object_from_value(value).expect("the test value is a JSON object")
}

/// Returns whether `event` is the `kind` lifecycle event of `turn_id`.
fn is_turn_event(event: &HarnessEvent, turn_id: &str, kind: TurnEventKind) -> bool {
    event.turn_id.as_deref() == Some(turn_id)
        && matches!(
            &event.event,
            HarnessEventPayload::TurnLifecycle(payload) if payload.kind == kind
        )
}

/// Returns whether `event` is the `old → new` lifecycle-state transition.
fn is_state_change(event: &HarnessEvent, old: HarnessState, new: HarnessState) -> bool {
    matches!(
        &event.event,
        HarnessEventPayload::StateChanged(change) if change.old == old && change.new == new
    )
}

/// Returns the text a projected output event carries, if it is one.
fn output_text(event: &HarnessEvent) -> Option<String> {
    let HarnessEventPayload::Output(output) = &event.event else {
        return None;
    };
    Some(match &output.content {
        JsonValue::String(text) => text.clone(),
        other => other.to_string(),
    })
}

/// Drains `cursor` until `turn_id` finished *and* the harness settled back in `Idle`.
///
/// The two conditions are observed, never timed: an event stream that ends (or a producer that
/// runs away) fails the test instead of hanging it. The collected events are returned so a
/// scenario can assert on the observed payload sequence.
async fn drain_turn(cursor: &mut Box<dyn HarnessEventCursor>, turn_id: &str) -> Vec<HarnessEvent> {
    let mut seen = Vec::new();
    let mut finished = false;
    for _ in 0..MAX_TURN_EVENTS {
        let event = within("the next protocol event", cursor.next_event())
            .await
            .expect("the observation cursor reported an error")
            .expect("the observation cursor ended before the turn settled");
        finished |= is_turn_event(&event, turn_id, TurnEventKind::Finished);
        let settled = is_state_change(&event, HarnessState::Running, HarnessState::Idle);
        seen.push(event);
        if finished && settled {
            return seen;
        }
    }
    panic!("the turn produced more than {MAX_TURN_EVENTS} events without settling");
}

// ---------------------------------------------------------------------------
// 1. Registry
// ---------------------------------------------------------------------------

/// The registry lists every declared provider and describes the native one.
#[test]
fn registry_lists_and_describes_the_native_provider() {
    let available = available_providers();
    for name in PROVIDER_NAMES {
        assert!(
            available.iter().any(|listed| listed == name),
            "declared provider `{name}` is missing from {available:?}"
        );
    }
    assert!(
        available.iter().any(|listed| listed == "native"),
        "the native provider is listed: {available:?}"
    );
    assert!(
        available.len() >= PROVIDER_NAMES.len(),
        "every declared name is listed: {available:?}"
    );

    let descriptor = describe_provider("native").expect("native is a declared provider");
    assert_eq!(descriptor.name, "native");
    assert_eq!(descriptor.card_name, "openjiuwen-native");
    assert!(
        !descriptor.implementation_version.is_empty(),
        "the descriptor reports a version"
    );
    assert!(descriptor.builtin, "native is a built-in provider");
    assert!(
        descriptor.implemented,
        "native is implemented in this build"
    );
    assert!(
        descriptor.configures_models,
        "native implements the model-configuration hook"
    );
    assert_eq!(descriptor.requires, None, "native needs no prerequisite");

    let descriptors = provider_descriptors();
    for name in PROVIDER_NAMES {
        assert!(
            descriptors.iter().any(|listed| listed.name == name),
            "declared provider `{name}` has no descriptor"
        );
    }
    let listed = descriptors
        .iter()
        .find(|listed| listed.name == "native")
        .expect("native is described");
    assert_eq!(listed.card_name, descriptor.card_name);
    assert_eq!(
        listed.implementation_version,
        descriptor.implementation_version
    );
    assert!(listed.builtin && listed.implemented && listed.configures_models);
}

// ---------------------------------------------------------------------------
// 2. Rejections
// ---------------------------------------------------------------------------

/// An unknown provider name is `NotFound`; a deferred vendor name is `Unsupported`.
#[test]
fn unknown_and_deferred_providers_are_rejected() {
    let error = create_harness("does-not-exist", object(json!({})))
        .err()
        .expect("an unknown provider name is rejected");
    assert!(
        matches!(error, Error::NotFound(_)),
        "an unknown provider is NotFound, got: {error}"
    );

    let error = create_harness("claudecode", object(json!({})))
        .err()
        .expect("a deferred vendor provider is rejected");
    assert!(
        matches!(error, Error::Unsupported(_)),
        "a deferred provider is Unsupported, got: {error}"
    );
}

// ---------------------------------------------------------------------------
// 3. Native harness lifecycle over an injected scripted model
// ---------------------------------------------------------------------------

/// Builds a `NativeHarness` over a `DeepAgent` whose model is injected directly.
///
/// The agent is built through the public builder, so the model never goes through lazy
/// resolution: the turn is answered by [`ScriptedModel`] and nothing touches the network. The
/// returned `TempDir` keeps the workspace alive for the test body.
fn scripted_harness(scripted: &ScriptedModel) -> (TempDir, NativeHarness) {
    let tmp = tempfile::tempdir().expect("a temporary workspace");
    let agent = DeepAgentBuilder::new(AgentCard::new("deep_agent"))
        .config(DeepAgentConfig {
            model_name: Some("scripted-model".to_owned()),
            ..DeepAgentConfig::default()
        })
        .model(Arc::new(scripted.clone()) as Arc<dyn Model>)
        .workspace(Workspace::new(tmp.path()))
        .build()
        .expect("the deep agent builds");
    (tmp, NativeHarness::new(agent, NativeHarnessOptions::new()))
}

/// Drives the native harness through one turn and asserts the whole lifecycle.
async fn assert_one_turn(harness: &dyn HarnessProtocol) {
    assert_eq!(
        harness.state(),
        HarnessState::Terminated,
        "a harness is created unstarted"
    );
    assert_eq!(
        harness.provider_session_id(),
        None,
        "an unstarted harness has no provider session"
    );

    within("harness start", harness.start(context()))
        .await
        .expect("the harness starts");
    assert_eq!(harness.state(), HarnessState::Idle);
    assert_eq!(
        harness.provider_session_id(),
        Some("host-session-1"),
        "the provider session mirrors the host session"
    );

    let mut cursor = harness.events();
    let receipt = within(
        "the first input",
        harness.send_auto(HarnessInput::text("hi")),
    )
    .await
    .expect("the harness accepts the first input");
    assert_eq!(receipt.message_id, "message-1");
    assert_eq!(receipt.turn_id, "turn-1");
    assert_eq!(receipt.accepted_mode, DeliveryMode::Auto);

    let events = drain_turn(&mut cursor, &receipt.turn_id).await;
    drop(cursor);

    let running = events
        .iter()
        .any(|event| is_state_change(event, HarnessState::Idle, HarnessState::Running));
    assert!(running, "the turn enters Running: {events:#?}");
    let started = events
        .iter()
        .any(|event| is_turn_event(event, &receipt.turn_id, TurnEventKind::Started));
    assert!(started, "the turn starts: {events:#?}");
    let finished = events
        .iter()
        .any(|event| is_turn_event(event, &receipt.turn_id, TurnEventKind::Finished));
    assert!(finished, "the turn finishes: {events:#?}");
    let settled = events
        .iter()
        .any(|event| is_state_change(event, HarnessState::Running, HarnessState::Idle));
    assert!(settled, "the harness settles back in Idle: {events:#?}");

    let answer: String = events.iter().filter_map(output_text).collect();
    assert!(
        answer.contains("hello-world"),
        "the model answer is projected, observed: {answer:?}"
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(&event.event, HarnessEventPayload::Output(_))),
        "at least one output event is observed"
    );

    assert_eq!(harness.state(), HarnessState::Idle, "the turn settled");

    within("harness stop", harness.stop())
        .await
        .expect("the harness stops");
    assert_eq!(
        harness.state(),
        HarnessState::Terminated,
        "a stopped harness is terminated"
    );
    assert_eq!(
        harness.provider_session_id(),
        None,
        "a terminated harness reports no provider session"
    );

    let error = within(
        "the rejected input",
        harness.send_auto(HarnessInput::text("again")),
    )
    .await
    .expect_err("a stopped harness accepts no input");
    assert_eq!(
        error.kind(),
        HarnessErrorKind::State,
        "a send after stop is a state error"
    );
}

/// Recipe A: a `NativeHarness` over an injected [`ScriptedModel`], start → send → drain → stop.
#[tokio::test]
async fn native_harness_start_send_events_stop() {
    let scripted = ScriptedModel::text("hello-world");
    let (_tmp, harness) = scripted_harness(&scripted);

    assert_one_turn(&harness).await;

    assert_eq!(
        scripted.call_count(),
        1,
        "the single turn made exactly one model call"
    );
}

// ---------------------------------------------------------------------------
// 4. Factory path over a loopback model server
// ---------------------------------------------------------------------------

/// Recipe B: `create_harness("native", …)` resolves its model lazily against a loopback server.
#[tokio::test]
async fn create_harness_native_runs_a_turn_over_loopback() {
    let server = MockLlmServer::with_script([ScriptedResponse::text("hello-world")]).await;
    let workspace = tempfile::tempdir().expect("a temporary workspace");
    let config = object(json!({
        "model": "mock-model",
        "provider": "openai",
        "api_base": server.base_url(),
        "api_key": TEST_API_KEY,
        "workspace": workspace.path().to_string_lossy(),
    }));
    let harness = create_harness("native", config).expect("the native provider creates a harness");
    assert_eq!(
        harness.state(),
        HarnessState::Terminated,
        "the factory returns an unstarted harness"
    );

    assert_one_turn(harness.as_ref()).await;

    assert!(
        server.request_count() >= 1,
        "the harness reached the loopback model server"
    );
    server.shutdown().await;
}

// ---------------------------------------------------------------------------
// 5. IO adapter projection over a hand-built harness double
// ---------------------------------------------------------------------------

/// Locks `mutex`, recovering from a poisoned lock instead of panicking.
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Cursor over the harness double's event queue.
struct FakeCursor {
    /// Receiver side of the harness double event queue.
    rx: mpsc::UnboundedReceiver<HarnessEvent>,
}

#[async_trait]
impl HarnessEventCursor for FakeCursor {
    async fn next_event(&mut self) -> HarnessResult<Option<HarnessEvent>> {
        Ok(self.rx.recv().await)
    }

    async fn close(&mut self) -> HarnessResult<()> {
        Ok(())
    }
}

/// Harness double: the test owns the event queue, so projection order is fully deterministic.
struct FakeHarness {
    /// Card reported to the adapter.
    card: HarnessCard,
    /// Current lifecycle state.
    state: Mutex<HarnessState>,
    /// Observation buffer contract reported to the adapter.
    buffer: EventBufferConfig,
    /// Producer side of the event queue the test pushes into.
    events_tx: mpsc::UnboundedSender<HarnessEvent>,
    /// Consumer side, taken exactly once by `events`.
    events_rx: Mutex<Option<mpsc::UnboundedReceiver<HarnessEvent>>>,
}

impl FakeHarness {
    /// Builds a harness in `Terminated`, reporting the default card and buffer.
    fn new() -> Self {
        let (events_tx, events_rx) = mpsc::unbounded_channel();
        Self {
            card: HarnessCard::new("fake", "0.1.0"),
            state: Mutex::new(HarnessState::Terminated),
            buffer: EventBufferConfig::default(),
            events_tx,
            events_rx: Mutex::new(Some(events_rx)),
        }
    }

    /// Pushes `event` into the observation queue the adapter consumes.
    fn push(&self, event: HarnessEvent) {
        self.events_tx
            .send(event)
            .expect("the fake event queue stays open");
    }
}

#[async_trait]
impl HarnessProtocol for FakeHarness {
    fn card(&self) -> &HarnessCard {
        &self.card
    }

    fn state(&self) -> HarnessState {
        *lock(&self.state)
    }

    fn provider_session_id(&self) -> Option<&str> {
        None
    }

    fn event_buffer_config(&self) -> &EventBufferConfig {
        &self.buffer
    }

    async fn start(&self, _context: HarnessContext) -> HarnessResult<()> {
        *lock(&self.state) = HarnessState::Idle;
        Ok(())
    }

    async fn stop(&self) -> HarnessResult<()> {
        *lock(&self.state) = HarnessState::Terminated;
        Ok(())
    }

    fn events(&self) -> Box<dyn HarnessEventCursor> {
        let rx = lock(&self.events_rx)
            .take()
            .expect("the fake event receiver is taken once");
        Box::new(FakeCursor { rx })
    }

    fn turn_events(&self, _turn_id: Option<&str>) -> Box<dyn HarnessEventCursor> {
        Box::new(VecEventCursor::new(Vec::new()))
    }

    async fn send(&self, _input: HarnessInput, mode: DeliveryMode) -> HarnessResult<SendReceipt> {
        Ok(SendReceipt {
            message_id: "message-1".to_owned(),
            turn_id: "turn-1".to_owned(),
            accepted_mode: mode,
        })
    }

    async fn abort(&self, _mode: AbortMode) -> HarnessResult<()> {
        Ok(())
    }

    async fn pause(&self) -> HarnessResult<()> {
        Ok(())
    }

    async fn resume(&self, _query: Option<HarnessInput>) -> HarnessResult<()> {
        Ok(())
    }

    async fn export_checkpoint(&self) -> HarnessResult<Option<HarnessCheckpoint>> {
        Ok(None)
    }
}

/// Builds an event envelope correlated to `turn-1`.
fn event(sequence: u64, payload: impl Into<HarnessEventPayload>) -> HarnessEvent {
    HarnessEvent::new(sequence, 1.0, payload, "host-session-1", "agent").with_turn_id("turn-1")
}

/// Builds a text output event on `channel`.
fn text_output(
    sequence: u64,
    output_id: &str,
    text: &str,
    operation: OutputOperation,
    channel: OutputChannel,
) -> HarnessEvent {
    event(
        sequence,
        OutputEvent {
            output_id: output_id.to_owned(),
            kind: OutputKind::Text,
            content: json!(text),
            operation,
            channel,
            content_index: 0,
            content_type: None,
            data: JsonObject::new(),
        },
    )
}

/// Builds a tool-call item lifecycle event identified by `item_id`.
fn item_event(sequence: u64, kind: ItemEventKind, item_id: &str, data: JsonValue) -> HarnessEvent {
    event(
        sequence,
        ItemLifecycleEvent {
            kind,
            item_type: "tool_call".to_owned(),
            data: object(data),
        },
    )
    .with_item_id(item_id)
}

/// Builds the terminal completed turn result of the projection scenario.
fn completed_turn() -> TurnResult {
    TurnResult {
        status: TurnStatus::Completed,
        messages: Vec::new(),
        final_output: json!("hello-world"),
        structured_output: JsonValue::Null,
        stop_reason: None,
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

/// Pushes the six events the projection scenario asserts on, in order.
fn push_projection_events(fake: &FakeHarness) {
    fake.push(text_output(
        1,
        "out-1",
        "hel",
        OutputOperation::Delta,
        OutputChannel::Answer,
    ));
    fake.push(text_output(
        2,
        "out-1",
        "hello-world",
        OutputOperation::Final,
        OutputChannel::Answer,
    ));
    fake.push(text_output(
        3,
        "out-2",
        "thinking",
        OutputOperation::Delta,
        OutputChannel::Reasoning,
    ));
    fake.push(item_event(
        4,
        ItemEventKind::Started,
        "call-1",
        json!({"name": "shell", "arguments": {"cmd": "ls"}}),
    ));
    fake.push(item_event(
        5,
        ItemEventKind::Completed,
        "call-1",
        json!({"tool_name": "shell", "result": "ok", "rendered_result": "ok"}),
    ));
    fake.push(event(
        6,
        TurnLifecycleEvent {
            kind: TurnEventKind::Finished,
            result: Some(completed_turn()),
        },
    ));
}

/// Wraps an adapter around `fake` with `config`.
fn adapter_with(fake: &Arc<FakeHarness>, config: HarnessIOAdapterConfig) -> Arc<HarnessIOAdapter> {
    let harness: DynHarness = fake.clone();
    HarnessIOAdapter::new(harness, config)
}

/// Reads the next projected chunk, bounded by [`support::BOUND`].
async fn next_chunk(stream: &mut BoxStream<'static, AgentChunk>) -> AgentChunk {
    within("the next adapter chunk", stream.next())
        .await
        .expect("the adapter chunk stream stayed open")
}

/// Reads exactly `count` projected chunks (the stream stays open, so it is never collected).
async fn read_chunks(stream: &mut BoxStream<'static, AgentChunk>, count: usize) -> Vec<AgentChunk> {
    let mut chunks = Vec::with_capacity(count);
    for _ in 0..count {
        chunks.push(next_chunk(stream).await);
    }
    chunks
}

/// Assertions shared by both projection configurations: the five projected chunks.
fn assert_projected_chunks(chunks: &[AgentChunk]) {
    let kinds: Vec<&str> = chunks.iter().map(|chunk| chunk.kind.as_str()).collect();
    assert_eq!(
        kinds,
        vec![
            LLM_OUTPUT,
            LLM_OUTPUT,
            LLM_REASONING,
            "tool_call",
            "tool_result"
        ],
        "the default projection emits no turn-lifecycle chunk"
    );
    let indices: Vec<i64> = chunks.iter().map(|chunk| chunk.index).collect();
    assert_eq!(indices, vec![0, 1, 2, 3, 4], "chunk indices are monotonic");

    let delta = &chunks[0];
    assert_eq!(delta.payload["content"], json!("hel"));
    assert_eq!(delta.payload["operation"], json!("delta"));
    assert_eq!(delta.payload["channel"], json!("answer"));
    assert_eq!(delta.payload["output_id"], json!("out-1"));

    let final_output = &chunks[1];
    assert_eq!(final_output.payload["content"], json!("hello-world"));
    assert_eq!(final_output.payload["operation"], json!("final"));
    assert_eq!(final_output.payload["channel"], json!("answer"));

    let reasoning = &chunks[2];
    assert_eq!(reasoning.payload["content"], json!("thinking"));
    assert_eq!(reasoning.payload["operation"], json!("delta"));
    assert_eq!(reasoning.payload["channel"], json!("reasoning"));

    let call = &chunks[3];
    assert_eq!(call.payload["name"], json!("shell"));
    assert_eq!(
        call.payload["arguments"],
        json!(r#"{"cmd":"ls"}"#),
        "tool arguments travel as a JSON string"
    );
    assert_eq!(call.payload["tool_call_id"], json!("call-1"));

    let result = &chunks[4];
    assert_eq!(result.payload["tool_name"], json!("shell"));
    assert_eq!(result.payload["result"], json!("ok"));
    assert_eq!(result.payload["rendered_result"], json!("ok"));
    assert_eq!(result.payload["tool_call_id"], json!("call-1"));
}

/// The adapter projects harness events onto kernel `AgentChunk`s, with and without turn events.
#[tokio::test]
async fn io_adapter_projects_events_to_agent_chunks() {
    // Default configuration: the finished turn projects to nothing, so exactly five chunks
    // arrive. Reading a fixed count (never collecting) is what proves the absence.
    let fake = Arc::new(FakeHarness::new());
    let adapter = adapter_with(&fake, HarnessIOAdapterConfig::default());
    within("adapter start", adapter.start(context()))
        .await
        .expect("the adapter starts");
    let mut stream = adapter
        .outputs()
        .expect("the chunk stream is handed out once");
    push_projection_events(&fake);
    let chunks = read_chunks(&mut stream, PROJECTION_CHUNKS).await;
    assert_projected_chunks(&chunks);
    within("adapter stop", adapter.stop())
        .await
        .expect("the adapter stops");
    assert_eq!(fake.state(), HarnessState::Terminated);

    // Turn-lifecycle configuration: the same six events yield a sixth `turn_lifecycle` chunk.
    let fake = Arc::new(FakeHarness::new());
    let adapter = adapter_with(
        &fake,
        HarnessIOAdapterConfig {
            emit_turn_lifecycle: true,
            ..HarnessIOAdapterConfig::default()
        },
    );
    within("adapter start", adapter.start(context()))
        .await
        .expect("the adapter starts");
    let mut stream = adapter
        .outputs()
        .expect("the chunk stream is handed out once");
    push_projection_events(&fake);
    let chunks = read_chunks(&mut stream, PROJECTION_CHUNKS + 1).await;
    assert_projected_chunks(&chunks[..PROJECTION_CHUNKS]);

    let lifecycle = &chunks[PROJECTION_CHUNKS];
    assert_eq!(lifecycle.kind, TURN_LIFECYCLE_KIND);
    assert_eq!(lifecycle.index, 5, "the chunk counter keeps advancing");
    assert_eq!(lifecycle.payload["kind"], json!("finished"));
    assert_eq!(lifecycle.payload["turn_id"], json!("turn-1"));
    assert_eq!(lifecycle.payload["final_output"], json!("hello-world"));

    within("adapter stop", adapter.stop())
        .await
        .expect("the adapter stops");

    // The harness double is never started twice: each configuration owns its own instance, so
    // the terminal state of the first one cannot leak into the second assertion.
    assert_eq!(fake.state(), HarnessState::Terminated);
}
