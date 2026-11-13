//! Offline fakes shared by the `team_agent` and `leader` unit tests.
//!
//! The scripted harness answers every `send` with one answer chunk and immediately returns to
//! `Idle` unless it is configured to hang (state stays `Running`, no output), which is what the
//! settle and cancellation tests need. No model, socket or clock is involved.
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_agent::AgentCard;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_harness_protocol::{
    AbortMode, DeliveryMode, EventBufferConfig, HarnessCard, HarnessContext, HarnessEvent,
    HarnessEventCursor, HarnessInput, HarnessProtocol, HarnessResult, HarnessState, OutputChannel,
    OutputEvent, OutputKind, OutputOperation, SendReceipt, VecEventCursor,
};
use parking_lot::Mutex;
use serde_json::json;
use tokio::sync::mpsc;

use crate::harness::{MemberHarnessContext, MemberHarnessFactory};
use crate::runtime::deps::TeamBuildDeps;
use crate::schema::{DeepAgentSpec, TeamAgentSpec};
use crate::workspace::{MemberWorkspace, WorkspaceProvider, WorkspaceRequest};

use super::SpecBuild;

/// One command a [`ScriptedHarness`] received.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Call {
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

/// Harness protocol handle with scripted answers: records commands, scripts one answer chunk.
pub(crate) struct ScriptedHarness {
    /// State reported by [`HarnessProtocol::state`].
    state: Mutex<HarnessState>,
    /// Commands received, in order.
    calls: Arc<Mutex<Vec<Call>>>,
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
    /// Builds a harness that answers every input with `answer`.
    fn new(answer: String, hang: bool) -> Arc<Self> {
        let (sender, receiver) = mpsc::unbounded_channel();
        Arc::new(Self {
            state: Mutex::new(HarnessState::Idle),
            calls: Arc::new(Mutex::new(Vec::new())),
            card: HarnessCard::new("scripted", "0.0.0"),
            buffer: EventBufferConfig::default(),
            events: Mutex::new(Some(receiver)),
            sender,
            answer,
            hang,
            sequence: AtomicU64::new(0),
        })
    }

    /// Returns the recorded commands.
    pub(crate) fn calls(&self) -> Vec<Call> {
        self.calls.lock().clone()
    }

    /// Returns the text of every delivered input, in order.
    pub(crate) fn sends(&self) -> Vec<String> {
        self.calls
            .lock()
            .iter()
            .filter_map(|call| match call {
                Call::Send(text) => Some(text.clone()),
                _ => None,
            })
            .collect()
    }

    /// Builds one answer observation.
    fn output_event(&self) -> HarnessEvent {
        let sequence = self.sequence.fetch_add(1, Ordering::SeqCst);
        HarnessEvent::new(
            sequence + 1,
            1.0,
            OutputEvent {
                output_id: format!("out-{sequence}"),
                kind: OutputKind::Text,
                content: json!(self.answer),
                operation: OutputOperation::Delta,
                channel: OutputChannel::Answer,
                content_index: 0,
                content_type: None,
                data: BTreeMap::new(),
            },
            "session:scripted",
            "scripted",
        )
        .with_turn_id("turn-1")
    }
}

#[async_trait]
#[allow(clippy::needless_pass_by_value)]
impl HarnessProtocol for ScriptedHarness {
    fn card(&self) -> &HarnessCard {
        &self.card
    }

    fn state(&self) -> HarnessState {
        *self.state.lock()
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
            let _ = self.sender.send(self.output_event());
            *self.state.lock() = HarnessState::Idle;
        }
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

/// Harness factory that builds [`ScriptedHarness`]es, records contexts and can fail or hang.
#[derive(Default)]
pub(crate) struct ScriptedFactory {
    /// Contexts received by [`MemberHarnessFactory::create`], in call order.
    calls: Mutex<Vec<MemberHarnessContext>>,
    /// Members whose build must fail with `Error::config`.
    failing: Mutex<BTreeSet<String>>,
    /// Members whose harness never settles.
    hanging: Mutex<BTreeSet<String>>,
    /// Harness built per member, so a test can inspect what the leader received.
    harnesses: Mutex<BTreeMap<String, Arc<ScriptedHarness>>>,
}

impl ScriptedFactory {
    /// Builds an absent-minded factory: every member gets its scripted answer.
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Makes `create` fail for `member`.
    pub(crate) fn failing_for(&self, member: &str) {
        self.failing.lock().insert(member.to_owned());
    }

    /// Makes the harness of `member` hang on every input.
    pub(crate) fn hanging_for(&self, member: &str) {
        self.hanging.lock().insert(member.to_owned());
    }

    /// Returns the recorded factory contexts, in call order.
    pub(crate) fn calls(&self) -> Vec<MemberHarnessContext> {
        self.calls.lock().clone()
    }

    /// Returns the harness built for `member`, when it was built.
    pub(crate) fn harness(&self, member: &str) -> Option<Arc<ScriptedHarness>> {
        self.harnesses.lock().get(member).cloned()
    }
}

#[async_trait]
impl MemberHarnessFactory for ScriptedFactory {
    async fn create(
        &self,
        _spec: &DeepAgentSpec,
        ctx: &MemberHarnessContext,
    ) -> Result<Arc<dyn HarnessProtocol>> {
        self.calls.lock().push(ctx.clone());
        if self.failing.lock().contains(&ctx.member_name) {
            return Err(Error::config(format!(
                "scripted factory failure for '{}'",
                ctx.member_name
            )));
        }
        let harness = ScriptedHarness::new(
            format!("done: {}", ctx.member_name),
            self.hanging.lock().contains(&ctx.member_name),
        );
        self.harnesses
            .lock()
            .insert(ctx.member_name.clone(), Arc::clone(&harness));
        Ok(harness)
    }

    fn name(&self) -> &'static str {
        "scripted"
    }
}

/// Workspace provider that hands out one distinct directory per member, recording every
/// allocation and release.
#[derive(Default)]
pub(crate) struct RecordingWorkspaceProvider {
    /// Allocation requests, in call order.
    requests: Mutex<Vec<WorkspaceRequest>>,
    /// Workspaces handed to [`WorkspaceProvider::release`], in call order.
    releases: Mutex<Vec<MemberWorkspace>>,
    /// When set, every release fails (the teardown-logs-and-continues cases).
    failing: AtomicBool,
}

impl RecordingWorkspaceProvider {
    /// Builds an empty recorder.
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Returns the recorded requests.
    pub(crate) fn requests(&self) -> Vec<WorkspaceRequest> {
        self.requests.lock().clone()
    }

    /// Returns the released workspaces, in release order.
    pub(crate) fn releases(&self) -> Vec<MemberWorkspace> {
        self.releases.lock().clone()
    }

    /// Returns the member name of every released workspace, in release order.
    pub(crate) fn released_names(&self) -> Vec<String> {
        self.releases
            .lock()
            .iter()
            .map(|workspace| workspace.member_name.clone())
            .collect()
    }

    /// Makes every subsequent release fail with [`Error::config`].
    pub(crate) fn fail_releases(&self) {
        self.failing.store(true, Ordering::SeqCst);
    }
}

#[async_trait]
impl WorkspaceProvider for RecordingWorkspaceProvider {
    async fn allocate(&self, request: &WorkspaceRequest) -> Result<MemberWorkspace> {
        self.requests.lock().push(request.clone());
        let root = std::env::temp_dir()
            .join("openjiuwen-agent-teams-tests")
            .join(&request.team_name)
            .join(&request.session_id)
            .join(&request.member_name);
        Ok(MemberWorkspace::new(
            request.team_name.clone(),
            request.session_id.clone(),
            request.member_name.clone(),
            root,
        ))
    }

    async fn release(&self, workspace: &MemberWorkspace) -> Result<()> {
        if self.failing.load(Ordering::SeqCst) {
            return Err(Error::config(format!(
                "scripted release failure for '{}'",
                workspace.member_name
            )));
        }
        self.releases.lock().push(workspace.clone());
        Ok(())
    }
}

/// Builds one member blueprint with a card and a prompt.
pub(crate) fn deep_spec(name: &str) -> DeepAgentSpec {
    DeepAgentSpec {
        card: Some(AgentCard::new(name)),
        system_prompt: Some(format!("prompt for {name}")),
        ..DeepAgentSpec::default()
    }
}

/// Builds a blueprint for `team` led by `leader` with one agent entry per teammate.
pub(crate) fn blueprint(team: &str, leader: &str, teammates: &[&str]) -> TeamAgentSpec {
    let mut spec = TeamAgentSpec {
        team_name: team.to_owned(),
        ..TeamAgentSpec::default()
    };
    spec.leader.member_name = leader.to_owned();
    spec.agents.insert("leader".to_owned(), deep_spec("Leader"));
    for name in teammates {
        spec.agents.insert((*name).to_owned(), deep_spec(name));
    }
    spec
}

/// Builds the injected dependencies over the two fakes.
pub(crate) fn deps(
    factory: &Arc<ScriptedFactory>,
    workspace: &Arc<RecordingWorkspaceProvider>,
) -> TeamBuildDeps {
    TeamBuildDeps::default()
        .with_harness_factory(Arc::clone(factory) as Arc<dyn MemberHarnessFactory>)
        .with_workspace(Arc::clone(workspace) as Arc<dyn WorkspaceProvider>)
}

/// Builds a leader agent for `blueprint` with fresh fakes and returns all three.
pub(crate) async fn built(
    blueprint: &TeamAgentSpec,
) -> (
    Arc<super::TeamAgent>,
    Arc<ScriptedFactory>,
    Arc<RecordingWorkspaceProvider>,
) {
    let factory = ScriptedFactory::new();
    let workspace = RecordingWorkspaceProvider::new();
    let agent = blueprint
        .build_with(deps(&factory, &workspace))
        .await
        .unwrap();
    (agent, factory, workspace)
}
