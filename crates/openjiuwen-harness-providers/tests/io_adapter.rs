#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::needless_pass_by_value,
    clippy::similar_names
)]
//! Offline, deterministic integration suite of the harness IO adapter.
//!
//! Every test drives a [`HarnessIOAdapter`] over the in-file [`FakeHarness`] test double, which
//! records the lifecycle commands it receives and lets the test push harness events at any time.
//! The adapter is exercised end to end: event projection, interaction routing, the pending
//! interaction map, host-answer parsing, delivery modes, aborts and failed-start cleanup.
//!
//! No network, no real provider, no `tokio::time::sleep`: ordering that matters is observed
//! through the chunk stream, and every await that could hang is wrapped in a five-second timeout
//! as a pure safety net.
//!
//! Python counterpart: `tests/unit_tests/harness_providers/test_io_adapter.py`.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_agent::output::LLM_OUTPUT;
use openjiuwen_agent::AgentChunk;
use openjiuwen_harness_protocol::{
    json_object_from_value, AbortMode, DeliveryMode, DynHarness, EventBufferConfig,
    HarnessCapability, HarnessCard, HarnessContext, HarnessError, HarnessErrorKind, HarnessEvent,
    HarnessEventCursor, HarnessEventPayload, HarnessInput, HarnessInteractionHandler,
    HarnessInteractionRequest, HarnessInteractionResponse, HarnessModelControl, HarnessProtocol,
    HarnessResult, HarnessState, HostCapability, InteractionCancelReason,
    InteractionResponseStatus, ItemEventKind, ItemLifecycleEvent, JsonObject, JsonValue,
    McpElicitationRequest, ModelSelection, OutputChannel, OutputEvent, OutputKind, OutputOperation,
    ProviderInteractionRequest, ProviderInteractionResponse, SendReceipt, ToolApprovalDecision,
    ToolApprovalRequest, ToolDefinition, ToolExecutionResult, ToolGateway, ToolInvocation,
    TurnError, TurnEventKind, TurnLifecycleEvent, TurnResult, TurnStatus, TurnTermination,
    TurnTerminationKind,
};
use openjiuwen_harness_providers::{
    HarnessIOAdapter, HarnessIOAdapterConfig, ProviderInteractionHandler, INTERACTION_KIND,
    INTERACTIVE_INPUT_KIND, TASK_FAILED_KIND, TURN_LIFECYCLE_KIND,
};
use openjiuwen_session::interaction::InteractiveInput;
use serde_json::json;
use tokio::sync::mpsc;

/// Timeout applied to every await that must not hang the suite.
const WAIT: Duration = Duration::from_secs(5);

/// Timeout used only to prove that the adapter emits nothing.
const QUIET: Duration = Duration::from_millis(50);

// ---------------------------------------------------------------------------
// Test doubles
// ---------------------------------------------------------------------------

/// Locks `mutex`, recovering from a poisoned lock instead of panicking.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Cursor over the fake harness event queue.
struct FakeCursor {
    /// Receiver side of the fake harness event queue.
    rx: mpsc::UnboundedReceiver<HarnessEvent>,
    /// Number of times this cursor was closed.
    closes: Arc<AtomicUsize>,
}

#[async_trait]
impl HarnessEventCursor for FakeCursor {
    async fn next_event(&mut self) -> HarnessResult<Option<HarnessEvent>> {
        Ok(self.rx.recv().await)
    }

    async fn close(&mut self) -> HarnessResult<()> {
        self.closes.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

/// Cursor that never yields an event; returned by `turn_events`, which the tests never use.
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

/// Harness test double: records every command and lets the test push events.
struct FakeHarness {
    /// Card reported to the adapter; capabilities are set per test.
    card: HarnessCard,
    /// Current lifecycle state.
    state: Mutex<HarnessState>,
    /// Provider-native session id reported to the adapter.
    session: Option<String>,
    /// Observation buffer contract reported to the adapter.
    buffer: EventBufferConfig,
    /// Producer side of the event queue the tests push into.
    events_tx: mpsc::UnboundedSender<HarnessEvent>,
    /// Consumer side, taken exactly once by `events`.
    events_rx: Mutex<Option<mpsc::UnboundedReceiver<HarnessEvent>>>,
    /// Number of cursor closes observed.
    closes: Arc<AtomicUsize>,
    /// Names of the commands received, in order.
    calls: Mutex<Vec<String>>,
    /// Contexts received by `start`.
    starts: Mutex<Vec<HarnessContext>>,
    /// Number of `stop` calls.
    stops: AtomicUsize,
    /// Inputs received by `send`, with their delivery mode.
    sends: Mutex<Vec<(HarnessInput, DeliveryMode)>>,
    /// Abort modes received by `abort`.
    aborts: Mutex<Vec<AbortMode>>,
    /// Number of `pause` calls.
    pauses: AtomicUsize,
    /// Number of `resume` calls.
    resumes: AtomicUsize,
    /// Last context received by `start`.
    last_context: Mutex<Option<HarnessContext>>,
    /// Fail switch of `start`, when set.
    start_error: Option<Box<dyn Fn() -> HarnessError + Send + Sync>>,
}

impl FakeHarness {
    /// Builds a harness with the default card, state and buffer.
    fn new() -> Self {
        let (events_tx, events_rx) = mpsc::unbounded_channel();
        Self {
            card: HarnessCard::new("fake", "0.1.0"),
            state: Mutex::new(HarnessState::Idle),
            session: None,
            buffer: EventBufferConfig::default(),
            events_tx,
            events_rx: Mutex::new(Some(events_rx)),
            closes: Arc::new(AtomicUsize::new(0)),
            calls: Mutex::new(Vec::new()),
            starts: Mutex::new(Vec::new()),
            stops: AtomicUsize::new(0),
            sends: Mutex::new(Vec::new()),
            aborts: Mutex::new(Vec::new()),
            pauses: AtomicUsize::new(0),
            resumes: AtomicUsize::new(0),
            last_context: Mutex::new(None),
            start_error: None,
        }
    }

    /// Declares `capabilities` on the card.
    fn with_capabilities(mut self, capabilities: &[HarnessCapability]) -> Self {
        self.card.capabilities.extend(capabilities.iter().copied());
        self
    }

    /// Sets the provider-native session id.
    fn with_session(mut self, session: &str) -> Self {
        self.session = Some(session.to_owned());
        self
    }

    /// Makes `start` fail with `kind`/`message`.
    fn fail_start(mut self, kind: HarnessErrorKind, message: &str) -> Self {
        let message = message.to_owned();
        self.start_error = Some(Box::new(move || error_of(kind, &message)));
        self
    }

    /// Pushes `event` into the observation queue the adapter consumes.
    fn push(&self, event: HarnessEvent) {
        self.events_tx
            .send(event)
            .expect("the fake event queue stays open");
    }

    /// Sets the lifecycle state the adapter observes.
    fn set_state(&self, state: HarnessState) {
        *lock(&self.state) = state;
    }

    /// Returns the last context received by `start`.
    fn last_context(&self) -> HarnessContext {
        lock(&self.last_context)
            .clone()
            .expect("the harness was started")
    }

    /// Returns the recorded inputs with their delivery modes.
    fn sends(&self) -> Vec<(HarnessInput, DeliveryMode)> {
        lock(&self.sends).clone()
    }

    /// Returns the recorded abort modes.
    fn aborts(&self) -> Vec<AbortMode> {
        lock(&self.aborts).clone()
    }

    /// Returns the number of `stop` calls.
    fn stop_count(&self) -> usize {
        self.stops.load(Ordering::SeqCst)
    }

    /// Returns the names of the commands received, in order.
    fn call_names(&self) -> Vec<String> {
        lock(&self.calls).clone()
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
        self.session.as_deref()
    }

    fn event_buffer_config(&self) -> &EventBufferConfig {
        &self.buffer
    }

    async fn start(&self, context: HarnessContext) -> HarnessResult<()> {
        if let Some(failure) = self.start_error.as_ref() {
            return Err(failure());
        }
        lock(&self.calls).push("start".to_owned());
        lock(&self.starts).push(context.clone());
        *lock(&self.last_context) = Some(context);
        self.set_state(HarnessState::Idle);
        Ok(())
    }

    async fn stop(&self) -> HarnessResult<()> {
        lock(&self.calls).push("stop".to_owned());
        self.stops.fetch_add(1, Ordering::SeqCst);
        self.set_state(HarnessState::Terminated);
        Ok(())
    }

    fn events(&self) -> Box<dyn HarnessEventCursor> {
        let rx = lock(&self.events_rx)
            .take()
            .expect("the fake event receiver is taken once");
        Box::new(FakeCursor {
            rx,
            closes: self.closes.clone(),
        })
    }

    fn turn_events(&self, _turn_id: Option<&str>) -> Box<dyn HarnessEventCursor> {
        Box::new(EmptyCursor)
    }

    async fn send(&self, input: HarnessInput, mode: DeliveryMode) -> HarnessResult<SendReceipt> {
        lock(&self.calls).push("send".to_owned());
        lock(&self.sends).push((input, mode));
        let count = lock(&self.sends).len();
        Ok(SendReceipt {
            message_id: format!("msg-{count}"),
            turn_id: "turn-1".to_owned(),
            accepted_mode: mode,
        })
    }

    async fn abort(&self, mode: AbortMode) -> HarnessResult<()> {
        lock(&self.calls).push("abort".to_owned());
        lock(&self.aborts).push(mode);
        Ok(())
    }

    async fn pause(&self) -> HarnessResult<()> {
        lock(&self.calls).push("pause".to_owned());
        self.pauses.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn resume(&self, _query: Option<HarnessInput>) -> HarnessResult<()> {
        lock(&self.calls).push("resume".to_owned());
        self.resumes.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn export_checkpoint(
        &self,
    ) -> HarnessResult<Option<openjiuwen_harness_protocol::HarnessCheckpoint>> {
        Ok(None)
    }
}

#[async_trait]
impl HarnessModelControl for FakeHarness {
    async fn list_models(&self) -> HarnessResult<Vec<openjiuwen_harness_protocol::ModelOption>> {
        Ok(Vec::new())
    }

    async fn set_model(&self, _selection: ModelSelection) -> HarnessResult<()> {
        Ok(())
    }
}

/// Tool gateway double: records invocations and answers with a fixed result.
#[derive(Default)]
struct FakeToolGateway {
    /// Invocations received, in order.
    invocations: Mutex<Vec<ToolInvocation>>,
}

#[async_trait]
impl ToolGateway for FakeToolGateway {
    async fn definitions(&self) -> HarnessResult<Vec<ToolDefinition>> {
        Ok(Vec::new())
    }

    async fn invoke(&self, invocation: &ToolInvocation) -> HarnessResult<ToolExecutionResult> {
        lock(&self.invocations).push(invocation.clone());
        Ok(ToolExecutionResult {
            content: json!({"ok": true}),
            is_error: false,
        })
    }
}

/// Provider interaction handler double: returns one fixed response.
struct FakeProviderHandler {
    /// Response returned for every request.
    response: ProviderInteractionResponse,
}

#[async_trait]
impl ProviderInteractionHandler for FakeProviderHandler {
    async fn handle(
        &self,
        _request: &ProviderInteractionRequest,
    ) -> HarnessResult<ProviderInteractionResponse> {
        Ok(self.response.clone())
    }
}

/// Interaction handler used only to prove an existing context handler is preserved.
struct ExistingHandler;

#[async_trait]
impl HarnessInteractionHandler for ExistingHandler {
    async fn handle(
        &self,
        _request: &HarnessInteractionRequest,
    ) -> HarnessResult<HarnessInteractionResponse> {
        Err(HarnessError::state(
            "the existing handler is not driven by this test",
        ))
    }

    async fn cancel(
        &self,
        _request_id: &str,
        _reason: InteractionCancelReason,
    ) -> HarnessResult<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Builders and helpers
// ---------------------------------------------------------------------------

/// Rebuilds a protocol error from its kind and message.
fn error_of(kind: HarnessErrorKind, message: &str) -> HarnessError {
    match kind {
        HarnessErrorKind::Protocol => HarnessError::protocol(message),
        HarnessErrorKind::UnsupportedCapability => HarnessError::unsupported_capability(message),
        HarnessErrorKind::CheckpointConflict => HarnessError::checkpoint_conflict(message),
        HarnessErrorKind::Validation => HarnessError::validation(message),
        _ => HarnessError::state(message),
    }
}

/// Builds a harness context with the default identity fields.
fn context() -> HarnessContext {
    HarnessContext::new("agent", "agent-1", "host-session", "system prompt")
}

/// Wraps an adapter around `fake`.
fn adapter_with(fake: &Arc<FakeHarness>, config: HarnessIOAdapterConfig) -> Arc<HarnessIOAdapter> {
    let harness: DynHarness = fake.clone();
    HarnessIOAdapter::new(harness, config)
}

/// Converts a JSON object value into the protocol's ordered object type.
fn object(value: JsonValue) -> JsonObject {
    json_object_from_value(value).expect("the test value is a JSON object")
}

/// Builds an event envelope correlated to `turn-1`.
fn event(sequence: u64, payload: impl Into<HarnessEventPayload>) -> HarnessEvent {
    HarnessEvent::new(sequence, 1.0, payload, "host-session", "agent").with_turn_id("turn-1")
}

/// Builds an answer-channel text output event.
fn text_output(sequence: u64, text: &str, operation: OutputOperation) -> HarnessEvent {
    event(
        sequence,
        OutputEvent {
            output_id: "out-1".to_owned(),
            kind: OutputKind::Text,
            content: json!(text),
            operation,
            channel: OutputChannel::Answer,
            content_index: 0,
            content_type: None,
            data: JsonObject::new(),
        },
    )
}

/// Builds a reasoning-channel text output event.
fn reasoning_output(sequence: u64, text: &str) -> HarnessEvent {
    event(
        sequence,
        OutputEvent {
            output_id: "out-2".to_owned(),
            kind: OutputKind::Text,
            content: json!(text),
            operation: OutputOperation::Delta,
            channel: OutputChannel::Reasoning,
            content_index: 0,
            content_type: None,
            data: JsonObject::new(),
        },
    )
}

/// Builds a tool-call item lifecycle event.
fn item_event(
    sequence: u64,
    kind: ItemEventKind,
    item_id: Option<&str>,
    data: JsonValue,
) -> HarnessEvent {
    let mut event = event(
        sequence,
        ItemLifecycleEvent {
            kind,
            item_type: "tool_call".to_owned(),
            data: object(data),
        },
    );
    event.item_id = item_id.map(str::to_owned);
    event
}

/// Builds a turn lifecycle event.
fn turn_event(sequence: u64, kind: TurnEventKind, result: Option<TurnResult>) -> HarnessEvent {
    event(sequence, TurnLifecycleEvent { kind, result })
}

/// Builds a terminal turn result.
fn turn_result(
    status: TurnStatus,
    final_output: JsonValue,
    error: Option<TurnError>,
    termination: Option<TurnTermination>,
) -> TurnResult {
    TurnResult {
        status,
        messages: Vec::new(),
        final_output,
        structured_output: JsonValue::Null,
        stop_reason: None,
        termination,
        error,
        usage: None,
        cost: None,
        started_at: None,
        completed_at: None,
        duration_ms: None,
        provider_data: JsonObject::new(),
    }
}

/// Builds a completed turn result carrying `final_output`.
fn completed(final_output: &str) -> TurnResult {
    turn_result(TurnStatus::Completed, json!(final_output), None, None)
}

/// Builds a failed turn result.
fn failed(message: &str, code: Option<&str>) -> TurnResult {
    turn_result(
        TurnStatus::Failed,
        JsonValue::Null,
        Some(TurnError {
            message: message.to_owned(),
            code: code.map(str::to_owned),
            category: None,
            retryable: None,
            provider_data: JsonObject::new(),
        }),
        None,
    )
}

/// Builds an interrupted turn result.
fn aborted(kind: TurnTerminationKind) -> TurnResult {
    turn_result(
        TurnStatus::Interrupted,
        JsonValue::Null,
        None,
        Some(TurnTermination {
            kind,
            message: None,
            code: None,
            provider_data: JsonObject::new(),
        }),
    )
}

/// Builds a tool approval request with `provider_data`.
fn approval_request(request_id: &str, provider_data: JsonObject) -> HarnessInteractionRequest {
    HarnessInteractionRequest::ToolApproval(ToolApprovalRequest {
        request_id: request_id.to_owned(),
        call_id: "call-1".to_owned(),
        tool_name: "shell".to_owned(),
        arguments: JsonObject::new(),
        provider_session_id: None,
        turn_id: Some("turn-1".to_owned()),
        title: None,
        description: None,
        reason: None,
        deadline_at: None,
        provider_data,
    })
}

/// Builds a user-input request.
fn user_request(request_id: &str) -> HarnessInteractionRequest {
    HarnessInteractionRequest::UserInput(openjiuwen_harness_protocol::UserInputRequest {
        request_id: request_id.to_owned(),
        prompt: "Pick one".to_owned(),
        provider_session_id: None,
        turn_id: Some("turn-1".to_owned()),
        choices: vec!["a".to_owned(), "b".to_owned()],
        deadline_at: None,
        provider_data: JsonObject::new(),
    })
}

/// Builds a provider interaction request.
fn provider_request(request_id: &str) -> HarnessInteractionRequest {
    HarnessInteractionRequest::Provider(ProviderInteractionRequest {
        request_id: request_id.to_owned(),
        provider: "claude_code".to_owned(),
        request_type: "permission".to_owned(),
        schema_version: "1".to_owned(),
        payload: JsonObject::new(),
        provider_session_id: None,
        turn_id: None,
        deadline_at: None,
    })
}

/// Builds a `user_inputs` host-answer payload.
fn user_inputs(entries: &[(&str, JsonValue)]) -> JsonValue {
    let answers: BTreeMap<String, JsonValue> = entries
        .iter()
        .map(|(id, value)| ((*id).to_owned(), value.clone()))
        .collect();
    json!({"user_inputs": answers})
}

/// Builds an `interrupt_answers` host-answer payload.
fn interrupt_answers(entries: &[(&str, JsonValue)]) -> JsonValue {
    let answers: BTreeMap<String, JsonValue> = entries
        .iter()
        .map(|(id, value)| ((*id).to_owned(), value.clone()))
        .collect();
    json!({"interrupt_answers": answers})
}

/// Reads one chunk, failing the test when the adapter stays quiet past the deadline.
async fn next_chunk(stream: &mut BoxStream<'static, AgentChunk>) -> AgentChunk {
    tokio::time::timeout(WAIT, stream.next())
        .await
        .expect("the adapter emits the chunk before the deadline")
        .expect("the chunk stream is still open")
}

/// Reads `count` chunks in order.
async fn read_chunks(stream: &mut BoxStream<'static, AgentChunk>, count: usize) -> Vec<AgentChunk> {
    let mut chunks = Vec::with_capacity(count);
    for _ in 0..count {
        chunks.push(next_chunk(stream).await);
    }
    chunks
}

/// Asserts the adapter emits no chunk within a short bounded wait.
async fn assert_no_chunk(stream: &mut BoxStream<'static, AgentChunk>) {
    let quiet = tokio::time::timeout(QUIET, stream.next()).await;
    assert!(quiet.is_err(), "expected no chunk, but one was emitted");
}

/// Waits until the adapter has a pending interaction, without sleeping.
async fn wait_until_pending(adapter: &HarnessIOAdapter) {
    tokio::time::timeout(WAIT, async {
        while !adapter.has_pending_interrupt() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the interaction becomes pending before the deadline");
}

/// Awaits a spawned interaction handler and returns its result.
async fn join_handler(
    waiting: tokio::task::JoinHandle<HarnessResult<HarnessInteractionResponse>>,
) -> HarnessInteractionResponse {
    tokio::time::timeout(WAIT, waiting)
        .await
        .expect("the interaction handler finishes before the deadline")
        .expect("the interaction handler does not panic")
        .expect("the interaction handler answered the request")
}

/// Asserts `response` is a tool approval and returns it.
fn approval_of(
    response: HarnessInteractionResponse,
) -> openjiuwen_harness_protocol::ToolApprovalResponse {
    match response {
        HarnessInteractionResponse::ToolApproval(approval) => approval,
        other => panic!("expected a tool approval response, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Port of `test_io_adapter.py::test_dynamic_tool_call_routes_to_context_tool_gateway`.
#[tokio::test]
async fn dynamic_tool_call_routes_to_context_tool_gateway() {
    let fake = Arc::new(FakeHarness::new());
    let gateway = Arc::new(FakeToolGateway::default());
    let adapter = adapter_with(&fake, HarnessIOAdapterConfig::default());

    let mut prepared = context();
    prepared.tools = Some(gateway.clone() as Arc<dyn ToolGateway>);
    let prepared = adapter.prepare_context(prepared);
    assert!(prepared
        .host_capabilities
        .contains(&HostCapability::NativeToolGateway));

    let request = HarnessInteractionRequest::DynamicToolCall(
        openjiuwen_harness_protocol::DynamicToolCallRequest {
            request_id: "intr_1".to_owned(),
            call_id: "call-1".to_owned(),
            tool_name: "shell".to_owned(),
            arguments: object(json!({"cmd": "ls"})),
            namespace: None,
            provider_session_id: None,
            turn_id: Some("turn-1".to_owned()),
            deadline_at: None,
            provider_data: JsonObject::new(),
        },
    );
    let response = adapter.handle(&request).await.expect("the gateway ran");

    let call = match response {
        HarnessInteractionResponse::DynamicToolCall(call) => call,
        other => panic!("expected a dynamic tool call response, got {other:?}"),
    };
    assert_eq!(call.request_id, "intr_1");
    assert_eq!(call.status, InteractionResponseStatus::Completed);
    assert_eq!(call.result, json!({"ok": true}));
    assert!(!call.is_error);
    assert!(call.error_message.is_none());

    let invocations = lock(&gateway.invocations).clone();
    assert_eq!(invocations.len(), 1);
    assert_eq!(invocations[0].call_id, "call-1");
    assert_eq!(invocations[0].name, "shell");
    assert_eq!(invocations[0].arguments, object(json!({"cmd": "ls"})));
}

/// Port of
/// `test_io_adapter.py::test_outputs_and_tool_items_project_to_deepagent_chunks`.
#[tokio::test]
async fn outputs_and_tool_items_project_to_deepagent_chunks() {
    let fake = Arc::new(FakeHarness::new());
    let adapter = adapter_with(&fake, HarnessIOAdapterConfig::default());
    adapter.start(context()).await.expect("the harness starts");
    let mut stream = adapter.outputs().expect("the stream is handed out once");

    fake.push(text_output(1, "Hel", OutputOperation::Delta));
    fake.push(text_output(2, "Hello", OutputOperation::Final));
    fake.push(reasoning_output(3, "thinking"));
    fake.push(item_event(
        4,
        ItemEventKind::Started,
        Some("call-1"),
        json!({"name": "shell", "arguments": {"cmd": "ls"}}),
    ));
    fake.push(item_event(
        5,
        ItemEventKind::Completed,
        Some("call-1"),
        json!({"tool_name": "shell", "result": "ok", "rendered_result": "ok"}),
    ));

    let chunks = read_chunks(&mut stream, 5).await;
    let kinds: Vec<&str> = chunks.iter().map(|chunk| chunk.kind.as_str()).collect();
    assert_eq!(
        kinds,
        [
            "llm_output",
            "llm_output",
            "llm_reasoning",
            "tool_call",
            "tool_result"
        ]
    );
    let indexes: Vec<i64> = chunks.iter().map(|chunk| chunk.index).collect();
    assert_eq!(indexes, vec![0, 1, 2, 3, 4], "chunk indexes are monotonic");

    assert_eq!(chunks[0].payload["content"], json!("Hel"));
    assert_eq!(chunks[0].payload["operation"], json!("delta"));
    assert_eq!(chunks[0].payload["channel"], json!("answer"));
    assert_eq!(chunks[0].payload["output_id"], json!("out-1"));
    assert_eq!(chunks[1].payload["content"], json!("Hello"));
    assert_eq!(chunks[1].payload["operation"], json!("final"));
    assert_eq!(chunks[2].payload["content"], json!("thinking"));
    assert_eq!(chunks[2].payload["channel"], json!("reasoning"));

    assert_eq!(chunks[3].payload["name"], json!("shell"));
    assert_eq!(chunks[3].payload["arguments"], json!(r#"{"cmd":"ls"}"#));
    assert_eq!(chunks[3].payload["tool_call_id"], json!("call-1"));
    assert_eq!(chunks[4].payload["tool_name"], json!("shell"));
    assert_eq!(chunks[4].payload["result"], json!("ok"));
    assert_eq!(chunks[4].payload["rendered_result"], json!("ok"));
    assert_eq!(chunks[4].payload["tool_call_id"], json!("call-1"));
}

/// Port of `test_io_adapter.py::test_turn_lifecycle_projection_is_disabled_by_default`.
#[tokio::test]
async fn turn_lifecycle_projection_is_disabled_by_default() {
    let fake = Arc::new(FakeHarness::new());
    let adapter = adapter_with(&fake, HarnessIOAdapterConfig::default());
    adapter.start(context()).await.expect("the harness starts");
    let mut stream = adapter.outputs().expect("the stream is handed out once");

    fake.push(turn_event(1, TurnEventKind::Started, None));
    fake.push(turn_event(
        2,
        TurnEventKind::Finished,
        Some(completed("done")),
    ));
    fake.push(turn_event(
        3,
        TurnEventKind::Failed,
        Some(failed("boom", None)),
    ));
    fake.push(turn_event(
        4,
        TurnEventKind::Aborted,
        Some(aborted(TurnTerminationKind::UserAbort)),
    ));
    fake.push(turn_event(5, TurnEventKind::Paused, None));
    fake.push(turn_event(6, TurnEventKind::Resumed, None));
    fake.push(text_output(7, "after", OutputOperation::Final));

    // The trailing output is the first chunk and carries index 0, so no lifecycle event above
    // produced a chunk.
    let chunk = next_chunk(&mut stream).await;
    assert_eq!(chunk.kind, LLM_OUTPUT);
    assert_eq!(chunk.index, 0);
    assert_eq!(chunk.payload["content"], json!("after"));
    assert_no_chunk(&mut stream).await;
}

/// Port of
/// `test_io_adapter.py::test_turn_lifecycle_projection_preserves_output_order_and_ignores_pause_resume`.
#[tokio::test]
async fn turn_lifecycle_projection_preserves_output_order_and_ignores_pause_resume() {
    let fake = Arc::new(FakeHarness::new());
    let adapter = adapter_with(
        &fake,
        HarnessIOAdapterConfig {
            emit_turn_lifecycle: true,
            ..HarnessIOAdapterConfig::default()
        },
    );
    adapter.start(context()).await.expect("the harness starts");
    let mut stream = adapter.outputs().expect("the stream is handed out once");

    fake.push(turn_event(1, TurnEventKind::Started, None));
    fake.push(text_output(2, "partial", OutputOperation::Delta));
    fake.push(turn_event(3, TurnEventKind::Paused, None));
    fake.push(turn_event(4, TurnEventKind::Resumed, None));
    fake.push(turn_event(
        5,
        TurnEventKind::Finished,
        Some(completed("complete")),
    ));
    fake.push(turn_event(
        6,
        TurnEventKind::Failed,
        Some(failed("boom", Some("E1"))),
    ));
    fake.push(turn_event(
        7,
        TurnEventKind::Aborted,
        Some(aborted(TurnTerminationKind::UserAbort)),
    ));

    let chunks = read_chunks(&mut stream, 5).await;
    let kinds: Vec<&str> = chunks.iter().map(|chunk| chunk.kind.as_str()).collect();
    assert_eq!(
        kinds,
        [
            TURN_LIFECYCLE_KIND,
            LLM_OUTPUT,
            TURN_LIFECYCLE_KIND,
            TURN_LIFECYCLE_KIND,
            TURN_LIFECYCLE_KIND
        ]
    );
    let indexes: Vec<i64> = chunks.iter().map(|chunk| chunk.index).collect();
    assert_eq!(indexes, vec![0, 1, 2, 3, 4]);
    assert_eq!(chunks[0].payload["kind"], json!("started"));
    assert_eq!(chunks[1].payload["content"], json!("partial"));
    assert_eq!(chunks[2].payload["kind"], json!("finished"));
    assert_eq!(chunks[2].payload["final_output"], json!("complete"));
    assert_eq!(chunks[3].payload["kind"], json!("failed"));
    assert_eq!(chunks[3].payload["error_message"], json!("boom"));
    assert_eq!(chunks[3].payload["error_code"], json!("E1"));
    assert_eq!(chunks[4].payload["kind"], json!("aborted"));
    assert_eq!(chunks[4].payload["termination_kind"], json!("user_abort"));
    // Python only stamps `turn_id` on the turn-lifecycle payloads; output chunks carry the
    // provider output fields only.
    for chunk in [&chunks[0], &chunks[2], &chunks[3], &chunks[4]] {
        assert_eq!(chunk.payload["turn_id"], json!("turn-1"));
    }
    assert_no_chunk(&mut stream).await;
}

/// Port of `test_io_adapter.py::test_turn_failure_marker_is_emitted_only_with_emit_turn_failure`.
#[tokio::test]
async fn turn_failure_marker_is_emitted_only_with_emit_turn_failure() {
    let failing = Arc::new(FakeHarness::new());
    let adapter = adapter_with(
        &failing,
        HarnessIOAdapterConfig {
            emit_turn_failure: true,
            ..HarnessIOAdapterConfig::default()
        },
    );
    adapter.start(context()).await.expect("the harness starts");
    let mut stream = adapter.outputs().expect("the stream is handed out once");
    failing.push(turn_event(
        1,
        TurnEventKind::Failed,
        Some(failed("boom", Some("E1"))),
    ));

    let chunk = next_chunk(&mut stream).await;
    assert_eq!(chunk.kind, TASK_FAILED_KIND);
    assert_eq!(chunk.payload["message"], json!("boom"));
    assert_eq!(chunk.payload["code"], json!("E1"));
    assert_no_chunk(&mut stream).await;

    // Turn-lifecycle chunks take precedence over the failure marker.
    let both = Arc::new(FakeHarness::new());
    let adapter = adapter_with(
        &both,
        HarnessIOAdapterConfig {
            emit_turn_lifecycle: true,
            emit_turn_failure: true,
            ..HarnessIOAdapterConfig::default()
        },
    );
    adapter.start(context()).await.expect("the harness starts");
    let mut stream = adapter.outputs().expect("the stream is handed out once");
    both.push(turn_event(
        1,
        TurnEventKind::Failed,
        Some(failed("boom", Some("E1"))),
    ));

    let chunk = next_chunk(&mut stream).await;
    assert_eq!(chunk.kind, TURN_LIFECYCLE_KIND);
    assert_eq!(chunk.payload["kind"], json!("failed"));
    assert_no_chunk(&mut stream).await;
}

/// Port of
/// `test_io_adapter.py::test_user_input_request_round_trips_through_interactive_input`.
#[tokio::test]
async fn user_input_request_round_trips_through_interactive_input() {
    let fake = Arc::new(FakeHarness::new().with_session("provider-session"));
    let adapter = adapter_with(&fake, HarnessIOAdapterConfig::default());
    adapter.start(context()).await.expect("the harness starts");
    assert_eq!(adapter.state(), HarnessState::Idle, "state is forwarded");
    assert_eq!(
        adapter.session_id().as_deref(),
        Some("provider-session"),
        "the session id is forwarded"
    );
    let mut stream = adapter.outputs().expect("the stream is handed out once");

    let request = user_request("intr_1");
    let waiting = tokio::spawn({
        let adapter = adapter.clone();
        let request = request.clone();
        async move { adapter.handle(&request).await }
    });

    let chunk = next_chunk(&mut stream).await;
    assert_eq!(chunk.kind, INTERACTION_KIND);
    assert_eq!(chunk.payload["id"], json!("intr_1"));
    assert_eq!(
        chunk.payload["value"]["interaction_type"],
        json!("user_input")
    );
    assert_eq!(chunk.payload["value"]["prompt"], json!("Pick one"));
    assert_eq!(chunk.payload["value"]["choices"], json!(["a", "b"]));

    wait_until_pending(&adapter).await;
    assert_eq!(adapter.pending_interrupt_ids(), vec!["intr_1".to_owned()]);
    let mut resume = InteractiveInput::new();
    resume.update("intr_1", json!("answer"));
    assert!(adapter.is_pending_interrupt_resume_valid(&resume));

    let receipt = adapter
        .send(user_inputs(&[("intr_1", json!("answer"))]), false)
        .await
        .expect("the answer resolves the pending interaction");
    assert!(receipt.is_none(), "a resolving answer is not forwarded");
    assert!(
        fake.sends().is_empty(),
        "the fake harness recorded no forward"
    );

    let response = join_handler(waiting).await;
    match response {
        HarnessInteractionResponse::UserInput(answer) => {
            assert_eq!(answer.request_id, "intr_1");
            assert_eq!(answer.status, InteractionResponseStatus::Completed);
            assert_eq!(answer.content, json!("answer"));
        }
        other => panic!("expected a user input response, got {other:?}"),
    }
}

/// Port of `test_io_adapter.py::test_pending_interaction_can_be_answered_by_its_interrupt_id`.
#[tokio::test]
async fn pending_interaction_can_be_answered_by_its_interrupt_id() {
    let fake = Arc::new(FakeHarness::new());
    let adapter = adapter_with(&fake, HarnessIOAdapterConfig::default());
    adapter.start(context()).await.expect("the harness starts");
    let mut stream = adapter.outputs().expect("the stream is handed out once");

    let waiting = tokio::spawn({
        let adapter = adapter.clone();
        let request = user_request("intr_1");
        async move { adapter.handle(&request).await }
    });
    assert_eq!(next_chunk(&mut stream).await.kind, INTERACTION_KIND);
    wait_until_pending(&adapter).await;

    let receipt = adapter
        .send(interrupt_answers(&[("intr_1", json!("answer"))]), false)
        .await
        .expect("the interrupt answer resolves the interaction");
    assert!(receipt.is_none());
    let response = join_handler(waiting).await;
    match response {
        HarnessInteractionResponse::UserInput(answer) => {
            assert_eq!(answer.content, json!("answer"));
        }
        other => panic!("expected a user input response, got {other:?}"),
    }

    // The same content no longer matches a pending interaction, so it is forwarded.
    let forwarded = adapter
        .send(interrupt_answers(&[("intr_1", json!("answer"))]), false)
        .await
        .expect("the unmatched answer is forwarded");
    assert!(forwarded.is_some(), "the forwarded input returns a receipt");
    let sends = fake.sends();
    assert_eq!(sends.len(), 1);
    assert_eq!(
        sends[0].0.content,
        interrupt_answers(&[("intr_1", json!("answer"))])
    );
    assert_eq!(sends[0].0.metadata["kind"], json!(INTERACTIVE_INPUT_KIND));
}

/// Port of `test_io_adapter.py::test_unmatched_interactive_input_is_forwarded_to_the_provider`.
#[tokio::test]
async fn unmatched_interactive_input_is_forwarded_to_the_provider() {
    let fake = Arc::new(FakeHarness::new());
    let adapter = adapter_with(&fake, HarnessIOAdapterConfig::default());

    let payload = user_inputs(&[("unknown", json!("value"))]);
    let receipt = adapter
        .send(payload.clone(), false)
        .await
        .expect("the unmatched answer is forwarded")
        .expect("the harness accepted the input");
    assert_eq!(receipt.message_id, "msg-1");
    assert_eq!(receipt.turn_id, "turn-1");

    let sends = fake.sends();
    assert_eq!(sends.len(), 1);
    assert_eq!(sends[0].0.content, payload);
    assert_eq!(sends[0].0.metadata["kind"], json!(INTERACTIVE_INPUT_KIND));
    assert_eq!(
        sends[0].1,
        DeliveryMode::Auto,
        "an idle harness lets the provider choose"
    );
}

/// Port of
/// `test_io_adapter.py::test_tool_approval_is_auto_allowed_by_default_and_asked_when_disabled`.
#[tokio::test]
async fn tool_approval_is_auto_allowed_by_default_and_asked_when_disabled() {
    let fake = Arc::new(FakeHarness::new());
    let adapter = adapter_with(&fake, HarnessIOAdapterConfig::default());
    let mut stream = adapter.outputs().expect("the stream is handed out once");

    let response = adapter
        .handle(&approval_request("intr_1", JsonObject::new()))
        .await
        .expect("the approval is answered automatically");
    let approval = approval_of(response);
    assert_eq!(approval.decision, ToolApprovalDecision::Allow);
    assert!(approval.reason.is_none());
    assert_no_chunk(&mut stream).await;

    let strict_fake = Arc::new(FakeHarness::new());
    let strict = adapter_with(
        &strict_fake,
        HarnessIOAdapterConfig {
            auto_approve_tools: false,
            ..HarnessIOAdapterConfig::default()
        },
    );
    let mut stream = strict.outputs().expect("the stream is handed out once");
    let waiting = tokio::spawn({
        let strict = strict.clone();
        let request = approval_request("intr_1", JsonObject::new());
        async move { strict.handle(&request).await }
    });
    let chunk = next_chunk(&mut stream).await;
    assert_eq!(chunk.kind, INTERACTION_KIND);
    assert_eq!(chunk.payload["id"], json!("intr_1"));
    wait_until_pending(&strict).await;

    let receipt = strict
        .send(
            interrupt_answers(&[("intr_1", json!({"approved": false, "feedback": "not now"}))]),
            false,
        )
        .await
        .expect("the host answer resolves the approval");
    assert!(receipt.is_none());
    let approval = approval_of(join_handler(waiting).await);
    assert_eq!(approval.decision, ToolApprovalDecision::Deny);
    assert_eq!(approval.reason.as_deref(), Some("not now"));
}

/// Port of
/// `test_io_adapter.py::test_an_agent_approval_is_asked_even_when_tools_are_auto_approved`.
#[tokio::test]
async fn an_agent_approval_is_asked_even_when_tools_are_auto_approved() {
    let fake = Arc::new(FakeHarness::new());
    let adapter = adapter_with(&fake, HarnessIOAdapterConfig::default());
    let mut stream = adapter.outputs().expect("the stream is handed out once");

    let provider_data = object(json!({"interrupt_id": "intr_1"}));
    let waiting = tokio::spawn({
        let adapter = adapter.clone();
        let request = approval_request("intr_1", provider_data);
        async move { adapter.handle(&request).await }
    });
    let chunk = next_chunk(&mut stream).await;
    assert_eq!(chunk.kind, INTERACTION_KIND);
    assert_eq!(chunk.payload["id"], json!("intr_1"));
    wait_until_pending(&adapter).await;

    let receipt = adapter
        .send(
            interrupt_answers(&[("intr_1", json!({"approved": true, "remember": true}))]),
            false,
        )
        .await
        .expect("the host answer resolves the approval");
    assert!(receipt.is_none());
    let approval = approval_of(join_handler(waiting).await);
    assert_eq!(approval.decision, ToolApprovalDecision::AllowForSession);
}

/// Port of `test_io_adapter.py::test_approval_answers_keep_their_decision`.
#[tokio::test]
async fn approval_answers_keep_their_decision() {
    let fake = Arc::new(FakeHarness::new());
    let adapter = adapter_with(
        &fake,
        HarnessIOAdapterConfig {
            auto_approve_tools: false,
            ..HarnessIOAdapterConfig::default()
        },
    );
    let mut stream = adapter.outputs().expect("the stream is handed out once");

    let cases: Vec<(JsonValue, ToolApprovalDecision, Option<&str>)> = vec![
        (json!(true), ToolApprovalDecision::Allow, None),
        (json!(false), ToolApprovalDecision::Deny, None),
        (json!("yes"), ToolApprovalDecision::Allow, None),
        (json!("always"), ToolApprovalDecision::AllowForSession, None),
        (json!("no"), ToolApprovalDecision::Deny, Some("no")),
        (
            json!("not now"),
            ToolApprovalDecision::Deny,
            Some("not now"),
        ),
        (
            json!({"action": "allow_once"}),
            ToolApprovalDecision::Allow,
            None,
        ),
        (
            json!({"action": "allow_always"}),
            ToolApprovalDecision::AllowForSession,
            None,
        ),
        (json!({"action": "deny"}), ToolApprovalDecision::Deny, None),
        (
            json!({"updated_arguments": {"x": "SAFE"}}),
            ToolApprovalDecision::Allow,
            None,
        ),
        (
            json!({}),
            ToolApprovalDecision::Deny,
            Some("Denied by the host."),
        ),
        (
            json!(1),
            ToolApprovalDecision::Deny,
            Some("Denied by the host."),
        ),
        (
            JsonValue::Null,
            ToolApprovalDecision::Deny,
            Some("Denied by the host."),
        ),
    ];

    for (position, (answer, expected, reason)) in cases.into_iter().enumerate() {
        let request_id = format!("intr_{position}");
        let waiting = tokio::spawn({
            let adapter = adapter.clone();
            let request = approval_request(&request_id, JsonObject::new());
            async move { adapter.handle(&request).await }
        });
        let chunk = next_chunk(&mut stream).await;
        assert_eq!(chunk.kind, INTERACTION_KIND);
        assert_eq!(chunk.payload["id"], json!(request_id));
        wait_until_pending(&adapter).await;

        let receipt = adapter
            .send(interrupt_answers(&[(&request_id, answer.clone())]), false)
            .await
            .expect("the answer resolves the approval");
        assert!(receipt.is_none(), "case {position} must resolve");

        let approval = approval_of(join_handler(waiting).await);
        assert_eq!(approval.decision, expected, "case {position}: {answer:?}");
        assert_eq!(
            approval.reason.as_deref(),
            reason,
            "case {position}: {answer:?}"
        );
        if answer == json!({"updated_arguments": {"x": "SAFE"}}) {
            assert_eq!(
                approval.updated_arguments,
                Some(object(json!({"x": "SAFE"}))),
                "updated arguments are carried over"
            );
        }
    }
}

/// Port of `test_io_adapter.py::test_delivery_mode_follows_state_and_capabilities`.
#[tokio::test]
async fn delivery_mode_follows_state_and_capabilities() {
    let fake = Arc::new(FakeHarness::new().with_capabilities(&[HarnessCapability::Steer]));
    let adapter = adapter_with(&fake, HarnessIOAdapterConfig::default());

    assert_eq!(adapter.state(), HarnessState::Idle);
    assert_eq!(adapter.delivery_mode(true), DeliveryMode::Auto);

    fake.set_state(HarnessState::Running);
    assert_eq!(adapter.state(), HarnessState::Running, "state is forwarded");
    assert_eq!(adapter.delivery_mode(true), DeliveryMode::Steer);
    assert_eq!(adapter.delivery_mode(false), DeliveryMode::FollowUp);

    let receipt = adapter
        .send(json!("steer me"), true)
        .await
        .expect("the input is delivered")
        .expect("the input is accepted");
    assert_eq!(receipt.accepted_mode, DeliveryMode::Steer);
    let sends = fake.sends();
    assert_eq!(sends.len(), 1);
    assert_eq!(sends[0].0.content, json!("steer me"));
    assert_eq!(sends[0].1, DeliveryMode::Steer);

    let plain = Arc::new(FakeHarness::new());
    let adapter = adapter_with(&plain, HarnessIOAdapterConfig::default());
    plain.set_state(HarnessState::Running);
    assert_eq!(adapter.delivery_mode(true), DeliveryMode::FollowUp);
    assert_eq!(adapter.delivery_mode(false), DeliveryMode::FollowUp);
}

/// Port of `test_io_adapter.py::test_abort_uses_the_declared_capability_or_stops`.
#[tokio::test]
async fn abort_uses_the_declared_capability_or_stops() {
    let graceful =
        Arc::new(FakeHarness::new().with_capabilities(&[HarnessCapability::GracefulAbort]));
    let adapter = adapter_with(&graceful, HarnessIOAdapterConfig::default());
    adapter.abort(false).await.expect("the harness aborts");
    assert_eq!(graceful.aborts(), vec![AbortMode::Graceful]);

    let force = Arc::new(FakeHarness::new().with_capabilities(&[HarnessCapability::ForceAbort]));
    let adapter = adapter_with(&force, HarnessIOAdapterConfig::default());
    adapter.abort(true).await.expect("the harness aborts");
    assert_eq!(force.aborts(), vec![AbortMode::Force]);

    let stopped = Arc::new(FakeHarness::new());
    let adapter = adapter_with(
        &stopped,
        HarnessIOAdapterConfig {
            stop_on_unsupported_force_abort: true,
            ..HarnessIOAdapterConfig::default()
        },
    );
    adapter
        .abort(true)
        .await
        .expect("the harness stops instead");
    assert_eq!(stopped.stop_count(), 1);

    let unsupported = Arc::new(FakeHarness::new());
    let adapter = adapter_with(&unsupported, HarnessIOAdapterConfig::default());
    let error = adapter.abort(true).await.expect_err("the abort is refused");
    assert_eq!(error.kind(), HarnessErrorKind::UnsupportedCapability);
    assert!(
        error.message().contains("does not support force_abort"),
        "unexpected message: {}",
        error.message()
    );
    assert_eq!(unsupported.aborts(), Vec::<AbortMode>::new());
}

/// Port of `test_io_adapter.py::test_failed_start_stops_the_harness_and_propagates`.
#[tokio::test]
async fn failed_start_stops_the_harness_and_propagates() {
    let fake = Arc::new(FakeHarness::new().fail_start(HarnessErrorKind::State, "cannot start"));
    let adapter = adapter_with(&fake, HarnessIOAdapterConfig::default());

    let error = adapter.start(context()).await.expect_err("start fails");
    assert_eq!(error.kind(), HarnessErrorKind::State);
    assert_eq!(error.message(), "cannot start");
    assert_eq!(fake.stop_count(), 1, "the failed start stops the harness");
    assert_eq!(fake.call_names(), vec!["stop".to_owned()]);
}

/// Port of `test_io_adapter.py::test_second_outputs_call_is_rejected`.
#[tokio::test]
async fn second_outputs_call_is_rejected() {
    let fake = Arc::new(FakeHarness::new());
    let adapter = adapter_with(&fake, HarnessIOAdapterConfig::default());

    assert!(adapter.outputs().is_ok());
    let Err(error) = adapter.outputs() else {
        panic!("the stream is single-consumer");
    };
    assert_eq!(error.kind(), HarnessErrorKind::State);
    assert!(
        error.message().contains("single-consumer"),
        "unexpected message: {}",
        error.message()
    );
}

/// Port of `test_io_adapter.py::test_second_start_is_rejected`.
#[tokio::test]
async fn second_start_is_rejected() {
    let fake = Arc::new(FakeHarness::new());
    let adapter = adapter_with(&fake, HarnessIOAdapterConfig::default());

    adapter
        .start(context())
        .await
        .expect("the first start succeeds");
    let error = adapter
        .start(context())
        .await
        .expect_err("the second start is rejected");
    assert_eq!(error.kind(), HarnessErrorKind::State);
    assert_eq!(error.message(), "harness IO adapter is already started");
    assert_eq!(fake.call_names(), vec!["start".to_owned()]);
}

/// Port of `test_io_adapter.py::test_stop_cancels_pending_interactions`.
#[tokio::test]
async fn stop_cancels_pending_interactions() {
    let fake = Arc::new(FakeHarness::new());
    let adapter = adapter_with(&fake, HarnessIOAdapterConfig::default());
    adapter.start(context()).await.expect("the harness starts");
    let mut stream = adapter.outputs().expect("the stream is handed out once");

    let waiting = tokio::spawn({
        let adapter = adapter.clone();
        let request = user_request("intr_1");
        async move { adapter.handle(&request).await }
    });
    assert_eq!(next_chunk(&mut stream).await.kind, INTERACTION_KIND);
    wait_until_pending(&adapter).await;

    adapter.stop().await.expect("the harness stops");
    let error = tokio::time::timeout(WAIT, waiting)
        .await
        .expect("the pending interaction settles")
        .expect("the handler does not panic")
        .expect_err("the pending interaction is cancelled");
    assert!(
        error.message().contains("was cancelled"),
        "unexpected message: {}",
        error.message()
    );
    assert!(!adapter.has_pending_interrupt());
    assert_eq!(adapter.pending_interrupt_ids(), Vec::<String>::new());
}

/// Port of
/// `test_io_adapter.py::test_mcp_elicitation_is_declined_and_provider_interactions_route`.
#[tokio::test]
async fn mcp_elicitation_is_declined_and_provider_interactions_route() {
    let fake = Arc::new(FakeHarness::new());
    let handler = Arc::new(FakeProviderHandler {
        response: ProviderInteractionResponse {
            request_id: "intr_1".to_owned(),
            status: InteractionResponseStatus::Completed,
            payload: object(json!({"answer": 42})),
        },
    });
    let adapter = adapter_with(
        &fake,
        HarnessIOAdapterConfig {
            provider_interaction_handler: Some(handler as Arc<dyn ProviderInteractionHandler>),
            ..HarnessIOAdapterConfig::default()
        },
    );
    let prepared = adapter.prepare_context(context());
    assert!(prepared
        .host_capabilities
        .contains(&HostCapability::ProviderInteraction));
    let mut stream = adapter.outputs().expect("the stream is handed out once");

    let elicitation = HarnessInteractionRequest::McpElicitation(McpElicitationRequest {
        request_id: "elic_1".to_owned(),
        server_name: "files".to_owned(),
        prompt: "Name?".to_owned(),
        schema: None,
        provider_session_id: None,
        turn_id: None,
        deadline_at: None,
        provider_data: JsonObject::new(),
    });
    let response = adapter.handle(&elicitation).await.expect("declined");
    match response {
        HarnessInteractionResponse::McpElicitation(declined) => {
            assert_eq!(declined.request_id, "elic_1");
            assert_eq!(declined.status, InteractionResponseStatus::Declined);
            assert!(declined.values.is_empty());
        }
        other => panic!("expected an MCP elicitation response, got {other:?}"),
    }
    assert_no_chunk(&mut stream).await;

    let response = adapter
        .handle(&provider_request("intr_1"))
        .await
        .expect("the configured handler answers");
    match response {
        HarnessInteractionResponse::Provider(routed) => {
            assert_eq!(routed.request_id, "intr_1");
            assert_eq!(routed.status, InteractionResponseStatus::Completed);
            assert_eq!(routed.payload, object(json!({"answer": 42})));
        }
        other => panic!("expected a provider response, got {other:?}"),
    }

    // Without a configured handler the request is declined and the capability is not declared.
    let bare = adapter_with(&fake, HarnessIOAdapterConfig::default());
    assert!(!bare
        .prepare_context(context())
        .host_capabilities
        .contains(&HostCapability::ProviderInteraction));
    let response = bare
        .handle(&provider_request("intr_2"))
        .await
        .expect("declined");
    match response {
        HarnessInteractionResponse::Provider(declined) => {
            assert_eq!(declined.request_id, "intr_2");
            assert_eq!(declined.status, InteractionResponseStatus::Declined);
            assert!(declined.payload.is_empty());
        }
        other => panic!("expected a provider response, got {other:?}"),
    }
}

/// Port of `test_io_adapter.py::test_prepare_context_declares_adapter_capabilities`.
#[tokio::test]
async fn prepare_context_declares_adapter_capabilities() {
    let fake = Arc::new(FakeHarness::new());

    let adapter = adapter_with(&fake, HarnessIOAdapterConfig::default());
    let prepared = adapter.prepare_context(context());
    assert!(prepared
        .host_capabilities
        .contains(&HostCapability::UserInput));
    assert!(!prepared
        .host_capabilities
        .contains(&HostCapability::ToolApproval));
    assert!(
        prepared.interactions.is_some(),
        "the adapter installs itself"
    );

    let strict = adapter_with(
        &fake,
        HarnessIOAdapterConfig {
            auto_approve_tools: false,
            ..HarnessIOAdapterConfig::default()
        },
    );
    let prepared = strict.prepare_context(context());
    assert!(prepared
        .host_capabilities
        .contains(&HostCapability::ToolApproval));

    let with_tools = adapter_with(&fake, HarnessIOAdapterConfig::default());
    let mut supplied = context();
    supplied.tools = Some(Arc::new(FakeToolGateway::default()) as Arc<dyn ToolGateway>);
    let prepared = with_tools.prepare_context(supplied);
    assert!(prepared
        .host_capabilities
        .contains(&HostCapability::NativeToolGateway));
    assert!(prepared
        .host_capabilities
        .contains(&HostCapability::DynamicToolCall));

    // An existing handler is preserved instead of being replaced.
    let existing: Arc<dyn HarnessInteractionHandler> = Arc::new(ExistingHandler);
    let mut supplied = context();
    supplied.interactions = Some(existing.clone());
    let prepared = adapter.prepare_context(supplied);
    let installed = prepared
        .interactions
        .clone()
        .expect("a handler is installed");
    assert!(Arc::ptr_eq(&existing, &installed));

    // The context the harness receives through `start` is the prepared one.
    let started = adapter_with(&fake, HarnessIOAdapterConfig::default());
    let prepared = started.prepare_context(context());
    started.start(prepared).await.expect("the harness starts");
    let received = fake.last_context();
    assert!(received.interactions.is_some());
    assert!(received
        .host_capabilities
        .contains(&HostCapability::UserInput));
    assert_eq!(received.host_session_id, "host-session");
}

/// Port of `test_io_adapter.py::test_structured_output_projects_to_llm_output`.
#[tokio::test]
async fn structured_output_projects_to_llm_output() {
    let fake = Arc::new(FakeHarness::new());
    let adapter = adapter_with(&fake, HarnessIOAdapterConfig::default());
    adapter.start(context()).await.expect("the harness starts");
    let mut stream = adapter.outputs().expect("the stream is handed out once");

    fake.push(event(
        1,
        OutputEvent {
            output_id: "out-1".to_owned(),
            kind: OutputKind::Structured,
            content: json!({"answer": 42, "tags": ["a", "b"]}),
            operation: OutputOperation::Final,
            channel: OutputChannel::Answer,
            content_index: 0,
            content_type: None,
            data: JsonObject::new(),
        },
    ));

    let chunk = next_chunk(&mut stream).await;
    assert_eq!(chunk.kind, LLM_OUTPUT);
    assert_eq!(
        chunk.payload["content"],
        json!({"answer": 42, "tags": ["a", "b"]})
    );
    assert_eq!(chunk.payload["operation"], json!("final"));
    assert_no_chunk(&mut stream).await;

    // A structured array keeps its JSON shape as well; the reasoning text kind is unaffected.
    fake.push(event(
        2,
        OutputEvent {
            output_id: "out-2".to_owned(),
            kind: OutputKind::Structured,
            content: json!([1, 2, 3]),
            operation: OutputOperation::Snapshot,
            channel: OutputChannel::Answer,
            content_index: 0,
            content_type: None,
            data: JsonObject::new(),
        },
    ));
    let chunk = next_chunk(&mut stream).await;
    assert_eq!(chunk.kind, LLM_OUTPUT);
    assert_eq!(chunk.index, 1);
    assert_eq!(chunk.payload["content"], json!([1, 2, 3]));
}
