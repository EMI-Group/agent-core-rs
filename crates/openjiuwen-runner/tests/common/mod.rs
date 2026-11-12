//! Shared fixtures for the runner's execution-path suites.
//!
//! Every suite is its own crate, so helpers used by only one of them still
//! trigger dead-code warnings here; that is expected and silenced below.
//!
//! Nothing in this module touches the network or the wall clock: agents are
//! in-process [`ScriptedAgent`]s (or a real `ReActAgent` over
//! `openjiuwen_test_support::ScriptedModel`, which is equally offline),
//! workflows are built from the crate's own components plus two local ones, and
//! teams are real `BaseTeam`/handoff teams whose members are closures.
#![allow(dead_code, unreachable_pub, clippy::unwrap_used, clippy::expect_used)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_agent::{Agent, AgentCard, AgentChunk, AgentOutput, Observation};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_multiagent::{
    handoff_team, AgentProvider, HandoffTeamConfig, Team, TeamAgent, TeamCard, HANDOFF_MESSAGE_KEY,
    HANDOFF_TARGET_KEY,
};
use openjiuwen_runner::callback::{
    CallbackContext, CallbackEvent, CallbackFramework, CallbackVerdict,
};
use openjiuwen_runner::config::RunnerConfig;
use openjiuwen_runner::resources::ResourceMgr;
use openjiuwen_runner::run::RunServices;
use openjiuwen_session::{
    AgentSession, AgentSessionOptions, Checkpoint, CheckpointMeta, Checkpointer,
    InteractionRequest, InteractionResponse, MemoryCheckpointer, NoopTraceSink, Session,
    SessionType, StreamItem, TraceSink,
};
use openjiuwen_workflow::{
    Component, ComponentCard, ComponentContext, ComponentOptions, EndComponent, StartComponent,
    Workflow, WorkflowCard,
};
use parking_lot::Mutex;
use serde_json::{json, Value};
use tokio::sync::Notify;

// ---------------------------------------------------------------------------
// Execution context
// ---------------------------------------------------------------------------

/// A `RunServices` over fresh, empty registries and default configuration.
pub fn services() -> RunServices {
    RunServices::new(
        ResourceMgr::new(),
        CallbackFramework::new(),
        RunnerConfig::default(),
        None,
    )
}

/// A `RunServices` whose created sessions get `checkpointer`.
pub fn services_with_checkpointer(checkpointer: Arc<dyn Checkpointer>) -> RunServices {
    RunServices::new(
        ResourceMgr::new(),
        CallbackFramework::new(),
        RunnerConfig::default(),
        Some(checkpointer),
    )
}

/// A `RunServices` whose configuration names `session_id` as the default.
pub fn services_with_default_session(session_id: &str) -> RunServices {
    RunServices::new(
        ResourceMgr::new(),
        CallbackFramework::new(),
        RunnerConfig::default().with_default_session_id(session_id),
        None,
    )
}

/// A `RunServices` with both a default session id and a checkpointer.
pub fn services_with_checkpointer_and_default_session(
    checkpointer: Arc<dyn Checkpointer>,
    session_id: &str,
) -> RunServices {
    RunServices::new(
        ResourceMgr::new(),
        CallbackFramework::new(),
        RunnerConfig::default().with_default_session_id(session_id),
        Some(checkpointer),
    )
}

// ---------------------------------------------------------------------------
// Callback recording
// ---------------------------------------------------------------------------

/// One dispatched callback: the event, its component id, and its frame index.
#[derive(Debug, Clone, PartialEq)]
pub struct Recorded {
    /// The dispatched event.
    pub event: CallbackEvent,
    /// `CallbackContext::component`.
    pub component: String,
    /// `CallbackContext::index` (streaming only).
    pub index: Option<i64>,
    /// `CallbackContext::error` (error-shaped events only).
    pub error: Option<String>,
    /// `CallbackContext::session_id` when the event carried a session.
    pub session_id: Option<String>,
}

/// The shared log a recording handler appends to.
pub type Recorder = Arc<Mutex<Vec<Recorded>>>;

/// Creates an empty recording log.
pub fn recorder() -> Recorder {
    Arc::new(Mutex::new(Vec::new()))
}

/// Registers one recording handler for `event` at `priority`.
pub fn record_at(
    callbacks: &CallbackFramework,
    event: CallbackEvent,
    priority: i32,
    recorder: &Recorder,
) {
    let recorder = Arc::clone(recorder);
    callbacks.register_sync(event, priority, move |ctx: &CallbackContext| {
        recorder.lock().push(Recorded {
            event: ctx.event.clone(),
            component: ctx.component.clone(),
            index: ctx.index,
            error: ctx.error.clone(),
            session_id: ctx.session_id().map(ToOwned::to_owned),
        });
        Ok(CallbackVerdict::Continue)
    });
}

/// Registers recording handlers for every event in `events` at priority 0.
pub fn record_all(callbacks: &CallbackFramework, events: &[CallbackEvent], recorder: &Recorder) {
    for event in events {
        record_at(callbacks, event.clone(), 0, recorder);
    }
}

/// The recorded events in dispatch order.
pub fn recorded_events(recorder: &Recorder) -> Vec<CallbackEvent> {
    recorder
        .lock()
        .iter()
        .map(|record| record.event.clone())
        .collect()
}

/// The recorded entries in dispatch order.
pub fn recorded(recorder: &Recorder) -> Vec<Recorded> {
    recorder.lock().clone()
}

/// The session ids of every recorded event that carried one.
pub fn recorded_session_ids(recorder: &Recorder) -> Vec<String> {
    recorder
        .lock()
        .iter()
        .filter_map(|record| record.session_id.clone())
        .collect()
}

/// Every agent event a suite may care about, in lifecycle order.
pub fn agent_events() -> Vec<CallbackEvent> {
    vec![
        CallbackEvent::AgentInvokeInput,
        CallbackEvent::AgentSessionCreated,
        CallbackEvent::AgentInvokeOutput,
        CallbackEvent::AgentStreamInput,
        CallbackEvent::AgentStreamOutput,
    ]
}

/// Every workflow event a suite may care about.
pub fn workflow_events() -> Vec<CallbackEvent> {
    vec![
        CallbackEvent::WorkflowInvokeInput,
        CallbackEvent::WorkflowInvokeOutput,
        CallbackEvent::WorkflowError,
        CallbackEvent::WorkflowStreamInput,
        CallbackEvent::WorkflowStreamOutput,
    ]
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

/// An in-process `AgentSession` named `id`.
pub fn agent_session(id: &str) -> Arc<AgentSession> {
    Arc::new(AgentSession::new(id))
}

/// An `AgentSession` named `id` that checkpoints into `checkpointer`.
pub fn checkpointed_agent_session(
    id: &str,
    checkpointer: Arc<dyn Checkpointer>,
) -> Arc<AgentSession> {
    Arc::new(AgentSession::with_options(AgentSessionOptions {
        session_id: Some(id.to_owned()),
        checkpointer: Some(checkpointer),
        ..AgentSessionOptions::default()
    }))
}

/// A memory checkpointer shared with a test.
pub fn memory_checkpointer() -> Arc<MemoryCheckpointer> {
    Arc::new(MemoryCheckpointer::new())
}

/// A `Session` that records every call made through the trait.
///
/// Public `Session` methods are the only seam a suite has to observe what a
/// lower layer did with a caller-provided session, so this double records the
/// calls instead of keeping state.
#[derive(Debug)]
pub struct RecordingSession {
    id: String,
    trace: NoopTraceSink,
    calls: Mutex<Vec<&'static str>>,
    state: Mutex<Value>,
}

impl RecordingSession {
    /// Builds a recording session named `id`.
    pub fn new(id: &str) -> Arc<Self> {
        Arc::new(Self {
            id: id.to_owned(),
            trace: NoopTraceSink,
            calls: Mutex::new(Vec::new()),
            state: Mutex::new(Value::Object(serde_json::Map::new())),
        })
    }

    /// The recorded call names, in call order.
    pub fn calls(&self) -> Vec<&'static str> {
        self.calls.lock().clone()
    }

    /// Whether `call` was recorded at least once.
    pub fn called(&self, call: &str) -> bool {
        self.calls.lock().contains(&call)
    }

    fn record(&self, call: &'static str) {
        self.calls.lock().push(call);
    }
}

#[async_trait]
impl Session for RecordingSession {
    fn id(&self) -> &str {
        &self.id
    }

    fn parent_id(&self) -> Option<&str> {
        None
    }

    fn session_type(&self) -> SessionType {
        SessionType::Workflow
    }

    async fn get_state(&self, _key: &str) -> Result<Option<Value>> {
        self.record("get_state");
        Ok(Some(self.state.lock().clone()))
    }

    async fn update_state(&self, patch: Value) -> Result<()> {
        self.record("update_state");
        *self.state.lock() = patch;
        Ok(())
    }

    async fn get_component_state(&self, _component_id: &str) -> Result<Option<Value>> {
        self.record("get_component_state");
        Ok(None)
    }

    async fn update_component_state(&self, _component_id: &str, _patch: Value) -> Result<()> {
        self.record("update_component_state");
        Ok(())
    }

    async fn write_stream(&self, _item: StreamItem) -> Result<()> {
        self.record("write_stream");
        Ok(())
    }

    async fn request_interaction(
        &self,
        _request: InteractionRequest,
    ) -> Result<InteractionResponse> {
        self.record("request_interaction");
        Err(Error::unsupported(
            "the recording session has no interactions",
        ))
    }

    fn trace(&self) -> &dyn TraceSink {
        &self.trace
    }

    async fn end_stream(&self) -> Result<()> {
        self.record("end_stream");
        Ok(())
    }

    async fn close(&self) -> Result<()> {
        self.record("close");
        Ok(())
    }
}

/// A `Checkpointer` that records every call and delegates to
/// [`MemoryCheckpointer`].
///
/// A session the runner creates is not visible to a test, so the checkpointer it
/// was built with is the seam that proves the created session's id and backend.
#[derive(Debug, Default)]
pub struct RecordingCheckpointer {
    inner: MemoryCheckpointer,
    calls: Mutex<Vec<(String, &'static str)>>,
}

impl RecordingCheckpointer {
    /// Builds an empty recording checkpointer.
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// The `(session_id, call)` pairs recorded so far.
    pub fn calls(&self) -> Vec<(String, &'static str)> {
        self.calls.lock().clone()
    }

    /// The call names recorded for `session_id`.
    pub fn calls_for(&self, session_id: &str) -> Vec<&'static str> {
        self.calls
            .lock()
            .iter()
            .filter(|(id, _)| id == session_id)
            .map(|(_, call)| *call)
            .collect()
    }

    /// The ids every call was made for, deduplicated in first-seen order.
    pub fn session_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = Vec::new();
        for (id, _) in self.calls.lock().iter() {
            if !ids.contains(id) {
                ids.push(id.clone());
            }
        }
        ids
    }

    fn record(&self, session_id: &str, call: &'static str) {
        self.calls.lock().push((session_id.to_owned(), call));
    }
}

#[async_trait]
impl Checkpointer for RecordingCheckpointer {
    async fn save(&self, checkpoint: Checkpoint) -> Result<String> {
        self.record(&checkpoint.session_id, "save");
        self.inner.save(checkpoint).await
    }

    async fn load(&self, session_id: &str, checkpoint_id: &str) -> Result<Option<Checkpoint>> {
        self.record(session_id, "load");
        self.inner.load(session_id, checkpoint_id).await
    }

    async fn list(&self, session_id: &str) -> Result<Vec<CheckpointMeta>> {
        self.record(session_id, "list");
        self.inner.list(session_id).await
    }

    async fn delete(&self, session_id: &str, checkpoint_id: &str) -> Result<()> {
        self.record(session_id, "delete");
        self.inner.delete(session_id, checkpoint_id).await
    }

    async fn release(&self, session_id: &str) -> Result<()> {
        self.record(session_id, "release");
        self.inner.release(session_id).await
    }

    async fn session_exists(&self, session_id: &str) -> Result<bool> {
        self.record(session_id, "session_exists");
        self.inner.session_exists(session_id).await
    }
}

// ---------------------------------------------------------------------------
// Scripted agent
// ---------------------------------------------------------------------------

/// The card of an agent whose id and name are both `id`.
pub fn agent_card(id: &str) -> AgentCard {
    AgentCard::new(id).with_id(id)
}

/// An [`Agent`] whose invoke and stream behaviour a test scripts.
///
/// It records how often each entry point ran and which session it was handed, so
/// a suite can assert the runner's session ownership and its error paths without
/// a model.
pub struct ScriptedAgent {
    card: AgentCard,
    output: Mutex<AgentOutput>,
    chunks: Mutex<Vec<AgentChunk>>,
    delay: Mutex<Option<Duration>>,
    failure: Mutex<Option<String>>,
    block: Mutex<Option<Arc<Notify>>>,
    started: Mutex<Option<Arc<Notify>>>,
    stream_gate: Mutex<Option<Arc<Notify>>>,
    invocations: AtomicUsize,
    streams: AtomicUsize,
    session_ids: Mutex<Vec<String>>,
}

impl ScriptedAgent {
    /// Builds an agent named and identified by `id`, answering `answer`.
    pub fn new(id: &str) -> Self {
        Self {
            card: agent_card(id),
            output: Mutex::new(AgentOutput::answer(format!("{id}:answer"))),
            chunks: Mutex::new(Vec::new()),
            delay: Mutex::new(None),
            failure: Mutex::new(None),
            block: Mutex::new(None),
            started: Mutex::new(None),
            stream_gate: Mutex::new(None),
            invocations: AtomicUsize::new(0),
            streams: AtomicUsize::new(0),
            session_ids: Mutex::new(Vec::new()),
        }
    }

    /// Replaces the invoked output.
    pub fn with_output(mut self, output: AgentOutput) -> Self {
        self.output = Mutex::new(output);
        self
    }

    /// Answers `answer` from `invoke`.
    pub fn with_answer(self, answer: &str) -> Self {
        self.with_output(AgentOutput::answer(answer))
    }

    /// Yields `chunks` from `stream`.
    pub fn with_chunks(mut self, chunks: Vec<AgentChunk>) -> Self {
        self.chunks = Mutex::new(chunks);
        self
    }

    /// Sleeps `delay` before answering and before every stream frame.
    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = Mutex::new(Some(delay));
        self
    }

    /// Fails both entry points with [`Error::tool`] carrying `message`.
    pub fn with_failure(mut self, message: &str) -> Self {
        self.failure = Mutex::new(Some(message.to_owned()));
        self
    }

    /// Blocks `invoke` until `gate` is notified (after `started` fired).
    pub fn blocking_on(mut self, gate: Arc<Notify>) -> Self {
        self.block = Mutex::new(Some(gate));
        self
    }

    /// Notifies `gate` as soon as `invoke` begins.
    pub fn notifying_start(mut self, gate: Arc<Notify>) -> Self {
        self.started = Mutex::new(Some(gate));
        self
    }

    /// Blocks the stream after its first frame until `gate` is notified.
    pub fn gating_stream_on(mut self, gate: Arc<Notify>) -> Self {
        self.stream_gate = Mutex::new(Some(gate));
        self
    }

    /// The number of `invoke` calls.
    pub fn invocations(&self) -> usize {
        self.invocations.load(Ordering::SeqCst)
    }

    /// The number of `stream` calls.
    pub fn streams(&self) -> usize {
        self.streams.load(Ordering::SeqCst)
    }

    /// The ids of the sessions handed to `invoke`/`stream`, in call order.
    pub fn session_ids(&self) -> Vec<String> {
        self.session_ids.lock().clone()
    }

    fn note_session(&self, session: Option<Arc<dyn Session>>) {
        if let Some(session) = session {
            self.session_ids.lock().push(session.id().to_owned());
        }
    }

    fn fail(&self) -> Option<Error> {
        self.failure.lock().clone().map(Error::tool)
    }
}

#[async_trait]
impl Agent for ScriptedAgent {
    fn card(&self) -> &AgentCard {
        &self.card
    }

    async fn invoke(
        &self,
        _inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> Result<AgentOutput> {
        self.invocations.fetch_add(1, Ordering::SeqCst);
        self.note_session(session);
        let started = self.started.lock().clone();
        let gate = self.block.lock().clone();
        let delay = *self.delay.lock();
        let failure = self.fail();
        if let Some(started) = started {
            started.notify_one();
        }
        if let Some(delay) = delay {
            tokio::time::sleep(delay).await;
        }
        if let Some(gate) = gate {
            gate.notified().await;
        }
        if let Some(error) = failure {
            return Err(error);
        }
        Ok(self.output.lock().clone())
    }

    fn stream(
        &self,
        _inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> BoxStream<'static, Result<AgentChunk>> {
        self.streams.fetch_add(1, Ordering::SeqCst);
        self.note_session(session);
        let chunks = self.chunks.lock().clone();
        let delay = *self.delay.lock();
        let gate = self.stream_gate.lock().clone();
        let failure = self.fail();
        async_stream::stream! {
            for (index, chunk) in chunks.into_iter().enumerate() {
                if let Some(delay) = delay {
                    tokio::time::sleep(delay).await;
                }
                if index > 0 {
                    if let Some(gate) = gate.clone() {
                        gate.notified().await;
                    }
                }
                yield Ok(chunk);
            }
            if let Some(error) = failure {
                yield Err(error);
            }
        }
        .boxed()
    }
}

/// A `ScriptedAgent` answering `answer`, ready to hand to the runner.
pub fn scripted_agent(id: &str, answer: &str) -> Arc<ScriptedAgent> {
    Arc::new(ScriptedAgent::new(id).with_answer(answer))
}

/// The three frames the streaming fixtures use, with deterministic payloads.
pub fn answer_frames() -> Vec<AgentChunk> {
    vec![
        AgentChunk::llm_output(0, "one"),
        AgentChunk::llm_output(1, "two"),
        AgentChunk::new(
            openjiuwen_agent::output::ANSWER,
            2,
            json!({"output": "one two", "result_type": "answer"}),
        ),
    ]
}

// ---------------------------------------------------------------------------
// Workflows
// ---------------------------------------------------------------------------

/// A component that returns its inputs unchanged (a `${node.field}` passthrough).
#[derive(Debug, Clone, Default)]
pub struct EchoComponent;

#[async_trait]
impl Component for EchoComponent {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("echo", "echo")
    }

    fn component_type(&self) -> &'static str {
        "echo"
    }

    async fn invoke(&self, inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
        Ok(inputs)
    }
}

/// A component that always fails with [`Error::workflow`] carrying
/// [`FAILING_COMPONENT_MESSAGE`].
#[derive(Debug, Clone, Default)]
pub struct FailingComponent;

/// The message [`FailingComponent`] fails with.
pub const FAILING_COMPONENT_MESSAGE: &str = "boom component";

#[async_trait]
impl Component for FailingComponent {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("failing", "failing")
    }

    fn component_type(&self) -> &'static str {
        "failing"
    }

    async fn invoke(&self, _inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
        Err(Error::workflow(FAILING_COMPONENT_MESSAGE))
    }
}

/// `start → echo → end`, where the end node reads `${echo.query}`.
///
/// The result is `{"output": {"answer": <inputs.query>}}`, so a suite can see
/// the reference-based passthrough in the final output.
pub fn passthrough_workflow(id: &str) -> Workflow {
    let mut workflow = Workflow::new(WorkflowCard::new(id));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"query": "${inputs.query}"})),
        )
        .add_workflow_comp(
            "echo",
            EchoComponent,
            ComponentOptions::new().with_inputs_schema(json!({"query": "${start.query}"})),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"answer": "${echo.query}"})),
        )
        .add_connection("start", "echo")
        .add_connection("echo", "end");
    workflow
}

/// `start → failing → end`: the run always fails in the middle component.
pub fn failing_workflow(id: &str) -> Workflow {
    let mut workflow = Workflow::new(WorkflowCard::new(id));
    workflow
        .set_start_comp(
            "start",
            StartComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"query": "${inputs.query}"})),
        )
        .add_workflow_comp(
            "failing",
            FailingComponent,
            ComponentOptions::new().with_inputs_schema(json!({"query": "${start.query}"})),
        )
        .set_end_comp(
            "end",
            EndComponent::new(),
            ComponentOptions::new().with_inputs_schema(json!({"answer": "${failing.query}"})),
        )
        .add_connection("start", "failing")
        .add_connection("failing", "end");
    workflow
}

// ---------------------------------------------------------------------------
// Teams
// ---------------------------------------------------------------------------

/// The member that starts a [`handoff_team`] chain.
pub const HANDOFF_FIRST: &str = "first";
/// The member the chain hands off to.
pub const HANDOFF_SECOND: &str = "second";
/// What the first member says before handing off.
pub const HANDOFF_FIRST_ANSWER: &str = "first is done";
/// What the second member answers (the team's final value).
pub const HANDOFF_SECOND_ANSWER: &str = "second is done";
/// The message the first member forwards to the second.
pub const HANDOFF_MESSAGE: &str = "context for second";

/// What the members of a [`handoff_team`] fixture saw while running.
#[derive(Debug, Default)]
pub struct TeamCalls {
    /// The parent session id of every member invocation, in call order.
    pub parent_session_ids: Mutex<Vec<Option<String>>>,
    /// The inputs of every member invocation, in call order.
    pub inputs: Mutex<Vec<Value>>,
}

impl TeamCalls {
    /// The recorded parent session ids.
    pub fn parent_session_ids(&self) -> Vec<Option<String>> {
        self.parent_session_ids.lock().clone()
    }

    /// The recorded member inputs.
    pub fn inputs(&self) -> Vec<Value> {
        self.inputs.lock().clone()
    }
}

/// Creates the shared call log of a [`handoff_team`] fixture.
pub fn team_calls() -> Arc<TeamCalls> {
    Arc::new(TeamCalls::default())
}

/// One member of a [`handoff_team`] fixture: a fixed output plus a call log.
struct TeamMember {
    card: AgentCard,
    output: AgentOutput,
    calls: Arc<TeamCalls>,
}

#[async_trait]
impl Agent for TeamMember {
    fn card(&self) -> &AgentCard {
        &self.card
    }

    async fn invoke(
        &self,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> Result<AgentOutput> {
        self.calls.inputs.lock().push(inputs);
        self.calls
            .parent_session_ids
            .lock()
            .push(session.and_then(|session| session.parent_id().map(ToOwned::to_owned)));
        Ok(self.output.clone())
    }

    fn stream(
        &self,
        _inputs: Value,
        _session: Option<Arc<dyn Session>>,
    ) -> BoxStream<'static, Result<AgentChunk>> {
        futures::stream::empty().boxed()
    }
}

/// Builds an [`AgentOutput`] carrying a handoff directive for `target`.
///
/// The directive travels as a tool [`Observation`] because a plain answer has no
/// top-level signal key (the multiagent contract).
pub fn handoff_directive(
    text: &str,
    target: &str,
    message: Option<&str>,
    reason: Option<&str>,
) -> AgentOutput {
    let mut signal = serde_json::Map::new();
    signal.insert(
        HANDOFF_TARGET_KEY.to_owned(),
        Value::String(target.to_owned()),
    );
    if let Some(message) = message {
        signal.insert(
            HANDOFF_MESSAGE_KEY.to_owned(),
            Value::String(message.to_owned()),
        );
    }
    if let Some(reason) = reason {
        signal.insert("reason".to_owned(), Value::String(reason.to_owned()));
    }
    AgentOutput {
        output: text.to_owned(),
        observations: vec![Observation {
            content: Value::Object(signal).to_string(),
            ..Observation::default()
        }],
        ..AgentOutput::default()
    }
}

/// Wraps `agent` as a team member provider (fresh wrapper per lookup).
pub fn provider(agent: Arc<dyn Agent>) -> AgentProvider {
    Arc::new(move || TeamAgent::new(Arc::clone(&agent)))
}

/// A real handoff team of two scripted members: `first` hands off to `second`.
///
/// `first` answers [`HANDOFF_FIRST_ANSWER`] with a directive to `second`, and
/// `second` answers [`HANDOFF_SECOND_ANSWER`]. Both members log the parent id of
/// the session they ran in, which is the team session the runner leased.
pub fn handoff_team_fixture(
    team_id: &str,
    calls: &Arc<TeamCalls>,
) -> Arc<openjiuwen_multiagent::BaseTeam> {
    let card = TeamCard::new(team_id).with_id(team_id);
    let team = handoff_team(card, HandoffTeamConfig::default());

    let first: Arc<dyn Agent> = Arc::new(TeamMember {
        card: agent_card(HANDOFF_FIRST),
        output: handoff_directive(
            HANDOFF_FIRST_ANSWER,
            HANDOFF_SECOND,
            Some(HANDOFF_MESSAGE),
            Some("route"),
        ),
        calls: Arc::clone(calls),
    });
    let second: Arc<dyn Agent> = Arc::new(TeamMember {
        card: agent_card(HANDOFF_SECOND),
        output: AgentOutput::answer(HANDOFF_SECOND_ANSWER),
        calls: Arc::clone(calls),
    });

    team.add_agent(agent_card(HANDOFF_FIRST), provider(first))
        .expect("the first member registers");
    team.add_agent(agent_card(HANDOFF_SECOND), provider(second))
        .expect("the second member registers");
    Arc::new(team)
}

/// The team id a [`handoff_team_fixture`] card carries.
pub fn team_card(team_id: &str) -> TeamCard {
    TeamCard::new(team_id).with_id(team_id)
}

/// Returns the runtime of `team` (a convenience for teardown assertions).
pub fn team_runtime(
    team: &Arc<openjiuwen_multiagent::BaseTeam>,
) -> Arc<openjiuwen_multiagent::TeamRuntime> {
    team.runtime()
}
