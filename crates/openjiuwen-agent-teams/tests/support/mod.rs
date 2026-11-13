//! Public-API-only fakes shared by the `openjiuwen-agent-teams` integration suites.
//!
//! Every suite under `tests/` is its own binary and only sees the crate's *public* surface, so the
//! fakes here are deliberately a re-expression of the in-crate ones
//! (`src/runtime/team_agent/test_support.rs` and `src/runtime/tests/support.rs`) built from
//! `openjiuwen_agent_teams::…` imports only. Nothing here touches the network, the clock or a
//! model: a [`ScriptedHarness`] answers every `send` with one scripted observation triple, a
//! [`FakeHarnessFactory`] hands out one scripted harness per member and a
//! [`FakeWorkspaceProvider`] allocates a real directory under one `TempDir`.
//!
//! The suites share this module, so each of them uses a different subset. The module inner
//! attributes below keep unused helpers and unused imports from failing the suites that do not
//! exercise them, and relax the lints that only matter in production code.
#![allow(dead_code, unused_imports, missing_docs)]
#![allow(unreachable_pub)]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::missing_panics_doc
)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};
use openjiuwen_agent::AgentChunk;
use openjiuwen_agent_teams::harness::{MemberHarnessContext, MemberHarnessFactory};
use openjiuwen_agent_teams::messager::{EventSink, Messager};
use openjiuwen_agent_teams::runtime::{SpecBuild, TeamAgent, TeamBuildDeps, TeamRuntimeManager};
use openjiuwen_agent_teams::schema::{
    is_team_event_marker, DeepAgentSpec, EventMessage, MemberSpecBase, PredefinedMemberSpec,
    TeamAgentSpec, TeamTopic,
};
use openjiuwen_agent_teams::storage::{InMemoryTeamStore, TeamStore};
use openjiuwen_agent_teams::workspace::{MemberWorkspace, WorkspaceProvider, WorkspaceRequest};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_harness_protocol::{
    AbortMode, DeliveryMode, EventBufferConfig, HarnessCard, HarnessContext, HarnessEvent,
    HarnessEventCursor, HarnessEventPayload, HarnessInput, HarnessProtocol, HarnessResult,
    HarnessState, OutputChannel, OutputEvent, OutputKind, OutputOperation, SendReceipt, TurnError,
    TurnEventKind, TurnLifecycleEvent, TurnResult, TurnStatus, TurnTermination,
    TurnTerminationKind, VecEventCursor,
};
use parking_lot::Mutex;
use serde_json::json;
use tempfile::TempDir;
use tokio::sync::mpsc;
use tokio::time::{timeout, Instant};

/// Upper bound for any single wait in these suites.
pub const BOUND: Duration = Duration::from_secs(5);

/// Poll interval of [`wait_until`].
const POLL: Duration = Duration::from_millis(5);

/// Turn id every scripted observation is correlated with.
const TURN_ID: &str = "turn-1";

/// Agent id of the scripted harness on the wire.
const AGENT_ID: &str = "scripted";

/// Polls `predicate` until it holds, panicking after [`BOUND`].
pub async fn wait_until(what: &str, mut predicate: impl FnMut() -> bool) {
    let deadline = Instant::now() + BOUND;
    while Instant::now() < deadline {
        if predicate() {
            return;
        }
        tokio::time::sleep(POLL).await;
    }
    assert!(predicate(), "timed out waiting for {what}");
}

/// Drains `stream` to its end, panicking when it does not finish within [`BOUND`].
pub async fn collect(
    mut stream: BoxStream<'static, Result<AgentChunk>>,
) -> Vec<Result<AgentChunk>> {
    timeout(BOUND, async move {
        let mut items = Vec::new();
        while let Some(item) = stream.next().await {
            items.push(item);
        }
        items
    })
    .await
    .expect("the run did not end within the bound")
}

/// Returns how many frames of `frames` are team event markers.
pub fn marker_count(frames: &[Result<AgentChunk>]) -> usize {
    frames
        .iter()
        .filter(|item| {
            item.as_ref()
                .is_ok_and(|chunk| is_team_event_marker(&chunk.payload))
        })
        .count()
}

/// One command a [`ScriptedHarness`] received.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Call {
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
    /// An abort was requested; the mode is recorded verbatim so force-vs-graceful is assertable.
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
/// `state` is what [`HarnessProtocol::state`] reports; a plain harness returns to [`Idle`] after
/// one answer triple, a [`hanging`](Self::hanging) one stays [`Running`] and emits nothing so a
/// suite can observe an in-flight turn.
///
/// [`Idle`]: HarnessState::Idle
/// [`Running`]: HarnessState::Running
pub struct ScriptedHarness {
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
    pub fn new(answer: impl Into<String>) -> Arc<Self> {
        Self::build(answer.into(), false)
    }

    /// Builds a harness whose every `send` leaves the state `Running` and emits nothing.
    pub fn hanging(answer: impl Into<String>) -> Arc<Self> {
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
    pub fn calls(&self) -> Vec<Call> {
        self.calls.lock().clone()
    }

    /// Returns the text of every delivered input, in order.
    pub fn sends(&self) -> Vec<String> {
        self.calls
            .lock()
            .iter()
            .filter_map(|call| match call {
                Call::Send(text) => Some(text.clone()),
                _ => None,
            })
            .collect()
    }

    /// Returns the abort modes requested, in order.
    pub fn abort_modes(&self) -> Vec<AbortMode> {
        self.calls
            .lock()
            .iter()
            .filter_map(|call| match call {
                Call::Abort(mode) => Some(*mode),
                _ => None,
            })
            .collect()
    }

    /// Returns the state the harness currently reports.
    pub fn state_now(&self) -> HarnessState {
        *self.state.lock()
    }

    /// Overrides the reported state, for suites that need a settled or terminated member.
    pub fn set_state(&self, state: HarnessState) {
        *self.state.lock() = state;
    }

    /// Pushes one observation onto the stream.
    pub fn emit(&self, event: HarnessEvent) {
        let _ = self.sender.send(event);
    }

    /// Pushes one `turn_lifecycle` observation for [`TURN_ID`].
    ///
    /// Terminal kinds carry the result their status requires, so the events stay valid.
    pub fn emit_turn(&self, kind: TurnEventKind) {
        self.emit(
            self.envelope(
                TurnLifecycleEvent {
                    kind,
                    result: result_for(kind),
                }
                .into(),
            ),
        );
    }

    /// Pushes one answer/`text`/`delta` output observation for [`TURN_ID`].
    pub fn emit_answer(&self, text: &str) {
        self.emit(self.answer_event(text));
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
        .with_turn_id(TURN_ID)
    }

    /// Builds one answer observation carrying `text`.
    fn answer_event(&self, text: &str) -> HarnessEvent {
        let sequence = self.sequence.fetch_add(1, Ordering::SeqCst);
        self.envelope(
            OutputEvent {
                output_id: format!("out-{sequence}"),
                kind: OutputKind::Text,
                content: json!(text),
                operation: OutputOperation::Delta,
                channel: OutputChannel::Answer,
                content_index: 0,
                content_type: None,
                data: BTreeMap::new(),
            }
            .into(),
        )
    }

    /// Pushes the one answer triple a settled round produces: started, output, finished.
    fn emit_round(&self) {
        self.emit(
            self.envelope(
                TurnLifecycleEvent {
                    kind: TurnEventKind::Started,
                    result: None,
                }
                .into(),
            ),
        );
        self.emit(self.answer_event(&self.answer));
        self.emit(
            self.envelope(
                TurnLifecycleEvent {
                    kind: TurnEventKind::Finished,
                    result: result_for(TurnEventKind::Finished),
                }
                .into(),
            ),
        );
    }
}

#[async_trait]
#[allow(clippy::needless_pass_by_value)]
impl HarnessProtocol for ScriptedHarness {
    fn card(&self) -> &HarnessCard {
        &self.card
    }

    fn state(&self) -> HarnessState {
        self.state_now()
    }

    fn provider_session_id(&self) -> Option<&str> {
        None
    }

    fn event_buffer_config(&self) -> &EventBufferConfig {
        &self.buffer
    }

    async fn start(&self, _context: HarnessContext) -> HarnessResult<()> {
        self.calls.lock().push(Call::Start);
        *self.state.lock() = HarnessState::Idle;
        Ok(())
    }

    async fn stop(&self) -> HarnessResult<()> {
        self.calls.lock().push(Call::Stop);
        *self.state.lock() = HarnessState::Terminated;
        Ok(())
    }

    fn events(&self) -> Box<dyn HarnessEventCursor> {
        match self.events.lock().take() {
            Some(receiver) => Box::new(ScriptCursor { receiver }),
            None => Box::new(VecEventCursor::new(Vec::new())),
        }
    }

    fn turn_events(&self, _turn_id: Option<&str>) -> Box<dyn HarnessEventCursor> {
        Box::new(VecEventCursor::new(Vec::new()))
    }

    async fn send(&self, input: HarnessInput, mode: DeliveryMode) -> HarnessResult<SendReceipt> {
        let text = input.content.as_str().unwrap_or_default().to_owned();
        self.calls.lock().push(Call::Send(text));
        *self.state.lock() = HarnessState::Running;
        if !self.hang {
            self.emit_round();
            *self.state.lock() = HarnessState::Idle;
        }
        Ok(SendReceipt {
            message_id: "message-1".to_owned(),
            turn_id: TURN_ID.to_owned(),
            accepted_mode: mode,
        })
    }

    async fn abort(&self, mode: AbortMode) -> HarnessResult<()> {
        self.calls.lock().push(Call::Abort(mode));
        Ok(())
    }

    async fn pause(&self) -> HarnessResult<()> {
        self.calls.lock().push(Call::Pause);
        *self.state.lock() = HarnessState::Paused;
        Ok(())
    }

    async fn resume(&self, _query: Option<HarnessInput>) -> HarnessResult<()> {
        self.calls.lock().push(Call::Resume);
        *self.state.lock() = HarnessState::Running;
        Ok(())
    }

    async fn export_checkpoint(
        &self,
    ) -> HarnessResult<Option<openjiuwen_harness_protocol::HarnessCheckpoint>> {
        Ok(None)
    }
}

/// Returns the result a turn event of `kind` carries, or `None` for a non-terminal kind.
fn result_for(kind: TurnEventKind) -> Option<TurnResult> {
    let (status, termination, error) = match kind {
        TurnEventKind::Started | TurnEventKind::Paused | TurnEventKind::Resumed => return None,
        TurnEventKind::Finished => (TurnStatus::Completed, None, None),
        TurnEventKind::Aborted => (
            TurnStatus::Interrupted,
            Some(TurnTermination {
                kind: TurnTerminationKind::UserAbort,
                message: Some("aborted by the host".to_owned()),
                code: None,
                provider_data: BTreeMap::new(),
            }),
            None,
        ),
        TurnEventKind::Failed => (
            TurnStatus::Failed,
            None,
            Some(TurnError {
                message: "the scripted turn failed".to_owned(),
                code: None,
                category: None,
                retryable: Some(false),
                provider_data: BTreeMap::new(),
            }),
        ),
    };
    Some(TurnResult {
        status,
        messages: Vec::new(),
        final_output: json!(null),
        structured_output: json!(null),
        stop_reason: Some("stop".to_owned()),
        termination,
        error,
        usage: None,
        cost: None,
        started_at: None,
        completed_at: None,
        duration_ms: None,
        provider_data: BTreeMap::new(),
    })
}

/// Harness factory that builds [`ScriptedHarness`]es, records contexts and can fail or hang.
#[derive(Default)]
pub struct FakeHarnessFactory {
    /// Contexts received by [`MemberHarnessFactory::create`], in call order.
    calls: Mutex<Vec<MemberHarnessContext>>,
    /// Members whose build must fail with `Error::config`.
    failing: Mutex<BTreeSet<String>>,
    /// Members whose harness never settles.
    hanging: Mutex<BTreeSet<String>>,
    /// Per-member answer scripted before the first `create`.
    answers: Mutex<BTreeMap<String, String>>,
    /// Harness built per member, so a suite can inspect what the member received.
    harnesses: Mutex<BTreeMap<String, Arc<ScriptedHarness>>>,
}

impl FakeHarnessFactory {
    /// Builds an absent-minded factory: every member gets `done: <member>` as its answer.
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Makes `create` fail for `member` with `Error::config`.
    pub fn failing_for(&self, member: &str) {
        self.failing.lock().insert(member.to_owned());
    }

    /// Makes the harness of `member` hang on every input.
    pub fn hanging_for(&self, member: &str) {
        self.hanging.lock().insert(member.to_owned());
    }

    /// Scripts the answer of `member`; recorded before the first `create` of that member.
    pub fn answers_with(&self, member: &str, answer: &str) {
        self.answers
            .lock()
            .insert(member.to_owned(), answer.to_owned());
    }

    /// Returns the recorded factory contexts, in creation order.
    pub fn calls(&self) -> Vec<MemberHarnessContext> {
        self.calls.lock().clone()
    }

    /// Returns the members a harness was created for, in creation order.
    pub fn created_members(&self) -> Vec<String> {
        self.calls
            .lock()
            .iter()
            .map(|ctx| ctx.member_name.clone())
            .collect()
    }

    /// Returns the workspace root `member`'s harness was created with.
    pub fn workspace_root(&self, member: &str) -> Option<PathBuf> {
        self.calls
            .lock()
            .iter()
            .find(|ctx| ctx.member_name == member)
            .map(|ctx| ctx.workspace_root.clone())
    }

    /// Returns the harness built for `member`, when it was built.
    pub fn harness(&self, member: &str) -> Option<Arc<ScriptedHarness>> {
        self.harnesses.lock().get(member).cloned()
    }

    /// Returns the static factory identity.
    #[allow(clippy::unused_self)]
    pub fn name(&self) -> &'static str {
        "fake"
    }
}

#[async_trait]
impl MemberHarnessFactory for FakeHarnessFactory {
    async fn create(
        &self,
        _spec: &DeepAgentSpec,
        ctx: &MemberHarnessContext,
    ) -> Result<Arc<dyn HarnessProtocol>> {
        self.calls.lock().push(ctx.clone());
        if self.failing.lock().contains(&ctx.member_name) {
            return Err(Error::config(format!(
                "fake factory failure for '{}'",
                ctx.member_name
            )));
        }
        let answer = self
            .answers
            .lock()
            .get(&ctx.member_name)
            .cloned()
            .unwrap_or_else(|| format!("done: {}", ctx.member_name));
        let harness = if self.hanging.lock().contains(&ctx.member_name) {
            ScriptedHarness::hanging(answer)
        } else {
            ScriptedHarness::new(answer)
        };
        self.harnesses
            .lock()
            .insert(ctx.member_name.clone(), Arc::clone(&harness));
        Ok(harness)
    }

    fn name(&self) -> &'static str {
        "fake"
    }
}

/// Workspace provider that hands out one distinct directory per member and records requests.
pub struct FakeWorkspaceProvider {
    /// Root every allocation is placed under; dropped with the provider.
    temp: TempDir,
    /// Allocation requests, in call order.
    requests: Mutex<Vec<WorkspaceRequest>>,
    /// Member names, in allocation order, with the root handed out.
    allocations: Mutex<Vec<(String, PathBuf)>>,
    /// Member names, in release order.
    releases: Mutex<Vec<String>>,
}

impl FakeWorkspaceProvider {
    /// Builds an empty provider rooted at a fresh temporary directory.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            temp: TempDir::new().expect("a member-workspace temp dir"),
            requests: Mutex::new(Vec::new()),
            allocations: Mutex::new(Vec::new()),
            releases: Mutex::new(Vec::new()),
        })
    }

    /// Returns the recorded allocation requests, in call order.
    pub fn requests(&self) -> Vec<WorkspaceRequest> {
        self.requests.lock().clone()
    }

    /// Returns the allocated member names, in allocation order.
    pub fn allocations(&self) -> Vec<String> {
        self.allocations
            .lock()
            .iter()
            .map(|(member, _)| member.clone())
            .collect()
    }

    /// Returns the released member names, in release order.
    pub fn releases(&self) -> Vec<String> {
        self.releases.lock().clone()
    }

    /// Returns the temporary directory every allocation is rooted under.
    pub fn root(&self) -> &Path {
        self.temp.path()
    }

    /// Returns the directory allocated to `member`, when it was allocated.
    pub fn root_for(&self, member: &str) -> Option<PathBuf> {
        self.allocations
            .lock()
            .iter()
            .find(|(name, _)| name == member)
            .map(|(_, root)| root.clone())
    }
}

#[async_trait]
impl WorkspaceProvider for FakeWorkspaceProvider {
    async fn allocate(&self, request: &WorkspaceRequest) -> Result<MemberWorkspace> {
        self.requests.lock().push(request.clone());
        let root = self
            .temp
            .path()
            .join(&request.team_name)
            .join(&request.session_id)
            .join(&request.member_name);
        std::fs::create_dir_all(&root).map_err(|error| {
            Error::config(format!(
                "failed to create the fake workspace '{}': {error}",
                root.display()
            ))
        })?;
        self.allocations
            .lock()
            .push((request.member_name.clone(), root.clone()));
        Ok(MemberWorkspace::new(
            request.team_name.clone(),
            request.session_id.clone(),
            request.member_name.clone(),
            root,
        ))
    }

    async fn release(&self, workspace: &MemberWorkspace) -> Result<()> {
        self.releases.lock().push(workspace.member_name.clone());
        Ok(())
    }
}

/// One runtime manager over fresh fakes, its own in-memory store and the two fakes.
///
/// The manager owns the dependencies; the fakes and the store are kept here as well because every
/// suite asserts against what the harnesses received, which workspace roots were handed out and
/// which rows the store holds.
pub struct Fixture {
    /// The manager under test.
    pub manager: Arc<TeamRuntimeManager>,
    /// Harness factory: records one scripted harness per member.
    pub factory: Arc<FakeHarnessFactory>,
    /// Workspace provider: records one allocation per member.
    pub workspace: Arc<FakeWorkspaceProvider>,
    /// The store injected through [`TeamBuildDeps::with_store`].
    pub store: Arc<InMemoryTeamStore>,
}

impl Fixture {
    /// Builds a fixture: one fake factory, one fake workspace provider, one in-memory store.
    pub fn new() -> Self {
        let factory = FakeHarnessFactory::new();
        let workspace = FakeWorkspaceProvider::new();
        let store: Arc<InMemoryTeamStore> = Arc::new(InMemoryTeamStore::new());
        let deps = build_deps(&factory, &workspace, Some(Arc::clone(&store)));
        Self {
            manager: Arc::new(TeamRuntimeManager::new(deps)),
            factory,
            workspace,
            store,
        }
    }

    /// Activates `spec` for `session` through the manager and returns the live team handle.
    pub async fn activate(&self, spec: &TeamAgentSpec, session: &str) -> Arc<TeamAgent> {
        self.manager
            .activate(spec, session, &serde_json::Value::Null)
            .await
            .expect("the team did not activate")
    }

    /// Returns the harness the factory built for `member`.
    pub fn harness(&self, member: &str) -> Arc<ScriptedHarness> {
        self.factory
            .harness(member)
            .unwrap_or_else(|| panic!("no harness was built for '{member}'"))
    }

    /// Returns the texts `member`'s harness received, in order.
    pub fn sends(&self, member: &str) -> Vec<String> {
        self.harness(member).sends()
    }
}

/// Builds the injected dependencies over the two fakes and, optionally, `store`.
pub fn build_deps(
    factory: &Arc<FakeHarnessFactory>,
    workspace: &Arc<FakeWorkspaceProvider>,
    store: Option<Arc<InMemoryTeamStore>>,
) -> TeamBuildDeps {
    let deps = TeamBuildDeps::default()
        .with_harness_factory(Arc::clone(factory) as Arc<dyn MemberHarnessFactory>)
        .with_workspace(Arc::clone(workspace) as Arc<dyn WorkspaceProvider>);
    match store {
        Some(store) => deps.with_store(Arc::clone(&store) as Arc<dyn TeamStore>),
        None => deps,
    }
}

/// Builds the sample blueprint: team `alpha`, leader `team_leader`, one predefined `dev-1`.
pub fn sample_spec() -> TeamAgentSpec {
    let mut spec = TeamAgentSpec {
        team_name: "alpha".to_owned(),
        ..TeamAgentSpec::default()
    };
    spec.agents
        .insert("leader".to_owned(), deep_spec("Leader", "lead the team"));
    spec.agents
        .insert("teammate".to_owned(), deep_spec("Teammate", "do the work"));
    spec.predefined_members
        .push(PredefinedMemberSpec::Teammate(MemberSpecBase {
            member_name: "dev-1".to_owned(),
            display_name: "Dev One".to_owned(),
            desc: "the developer".to_owned(),
            prompt: "you are dev-1".to_owned(),
            model_name: None,
        }));
    spec
}

/// Builds the sample blueprint with task verification enabled.
pub fn sample_spec_verified() -> TeamAgentSpec {
    TeamAgentSpec {
        enable_task_verification: true,
        ..sample_spec()
    }
}

/// Builds one member blueprint with a card and a prompt.
fn deep_spec(name: &str, prompt: &str) -> DeepAgentSpec {
    DeepAgentSpec {
        card: Some(
            openjiuwen_agent::AgentCard::new(name).with_description(format!("{name} description")),
        ),
        system_prompt: Some(prompt.to_owned()),
        ..DeepAgentSpec::default()
    }
}

/// Events recorded by one observing bus sink.
pub type Recorder = Arc<Mutex<Vec<EventMessage>>>;

/// Subscribes one recording observer to `topics` on the team's own transport.
pub fn subscribe_observer(messager: &Arc<dyn Messager>, topics: &[TeamTopic]) -> Recorder {
    let recorder: Recorder = Arc::new(Mutex::new(Vec::new()));
    let sink: EventSink = {
        let recorder = Arc::clone(&recorder);
        Arc::new(move |event: EventMessage| recorder.lock().push(event))
    };
    messager
        .subscribe("observer", topics, sink)
        .expect("the observer could not subscribe");
    recorder
}

/// Returns a snapshot of the recorded events.
pub fn recorded(recorder: &Recorder) -> Vec<EventMessage> {
    recorder.lock().clone()
}

/// Returns the recorded event types, in order.
pub fn event_types(recorder: &Recorder) -> Vec<String> {
    recorded(recorder)
        .iter()
        .map(|event| event.event_type.clone())
        .collect()
}
