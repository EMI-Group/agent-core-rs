//! Agent-teams cross-crate scenario: blueprint → built team → run → task board.
//!
//! The suite drives `openjiuwen::agent_teams` only through its public surface, the way the
//! harness/CLI layers consume it:
//!
//! 1. a blueprint with a leader and one predefined teammate builds over an injected
//!    [`MemberHarnessFactory`], materializing the leader harness and an `unstarted` teammate row;
//! 2. a started team with a spawned teammate streams exactly one `team.completed` marker, last,
//!    after the leader's scripted answer;
//! 3. a task created with no reviewers moves `pending → in_progress → completed` through the
//!    board, publishes its mutation events and settles the board.
//!
//! Everything is offline and deterministic: every member is a hand-built scripted
//! [`HarnessProtocol`], workspaces are real directories under one test-owned [`TempDir`], and no
//! assertion depends on a timestamp or a generated id.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod support;

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openjiuwen::agent_teams::harness::{MemberHarnessContext, MemberHarnessFactory};
use openjiuwen::agent_teams::runtime::{SpecBuild, TeamAgent, TeamBuildDeps};
use openjiuwen::agent_teams::schema::{
    is_team_event_marker, DeepAgentSpec, EventPayload, MemberSpecBase, MemberStatus,
    PredefinedMemberSpec, TaskGraphSpec, TaskStatus, TeamAgentSpec, TeamRole, TEAM_COMPLETED,
};
use openjiuwen::agent_teams::storage::{InMemoryTeamStore, TeamStore};
use openjiuwen::agent_teams::taskboard::TaskMutation;
use openjiuwen::agent_teams::workspace::{
    LocalWorkspaceProvider, MemberWorkspaceLayout, WorkspaceProvider,
};
use openjiuwen::core::common::error::Result;
use openjiuwen::core::single_agent::{AgentCard, AgentChunk};
use openjiuwen::harness_protocol::{
    AbortMode, DeliveryMode, EventBufferConfig, HarnessCard, HarnessCheckpoint, HarnessContext,
    HarnessEvent, HarnessEventCursor, HarnessEventPayload, HarnessInput, HarnessProtocol,
    HarnessResult, HarnessState, OutputChannel, OutputEvent, OutputKind, OutputOperation,
    SendReceipt, TurnEventKind, TurnLifecycleEvent, TurnResult, TurnStatus, VecEventCursor,
};
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::sync::mpsc;

use support::{collect_bounded, within};

/// Session fragment every scripted observation is correlated with.
const TURN_ID: &str = "turn-1";

/// Agent id the scripted harness reports on the wire.
const AGENT_ID: &str = "scripted";

/// Upper bound on the frames one run may produce before [`collect_bounded`] gives up.
const MAX_FRAMES: usize = 512;

/// A command a [`ScriptedHarness`] received.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Call {
    /// The cycle was started.
    Start,
    /// An input was delivered; the payload is its text.
    Send(String),
}

/// Cursor over the scripted observation channel.
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

/// Harness protocol handle that answers every `send` with one scripted observation triple.
///
/// `start` settles the harness to [`HarnessState::Idle`]; `send` emits `Started`, one
/// answer/`text`/`delta` output, then `Finished`, and returns to `Idle`. Every command is
/// recorded so a scenario can assert what the member actually received.
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
    /// Envelope sequence counter.
    sequence: AtomicU64,
}

impl ScriptedHarness {
    /// Builds a harness that answers every input with `answer` and returns to `Idle`.
    fn scripted(answer: impl Into<String>) -> Arc<Self> {
        let (sender, receiver) = mpsc::unbounded_channel();
        Arc::new(Self {
            state: Mutex::new(HarnessState::Idle),
            calls: Mutex::new(Vec::new()),
            card: HarnessCard::new("scripted", "0.0.0"),
            buffer: EventBufferConfig::default(),
            events: Mutex::new(Some(receiver)),
            sender,
            answer: answer.into(),
            sequence: AtomicU64::new(0),
        })
    }

    /// Returns the recorded commands, in order.
    fn calls(&self) -> Vec<Call> {
        self.calls.lock().unwrap().clone()
    }

    /// Returns the text of every delivered input, in order.
    fn sends(&self) -> Vec<String> {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .filter_map(|call| match call {
                Call::Send(text) => Some(text.clone()),
                Call::Start => None,
            })
            .collect()
    }

    /// Pushes one observation onto the stream.
    fn emit(&self, event: HarnessEvent) {
        let _ = self.sender.send(event);
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

    /// Builds one answer/`text`/`delta` output observation carrying `text`.
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

    /// Pushes the observation triple a settled round produces: started, output, finished.
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
                    result: Some(completed_result()),
                }
                .into(),
            ),
        );
    }
}

/// Returns the terminal result of a normally finished turn.
fn completed_result() -> TurnResult {
    TurnResult {
        status: TurnStatus::Completed,
        messages: Vec::new(),
        final_output: Value::Null,
        structured_output: Value::Null,
        stop_reason: Some("stop".to_owned()),
        termination: None,
        error: None,
        usage: None,
        cost: None,
        started_at: None,
        completed_at: None,
        duration_ms: None,
        provider_data: BTreeMap::new(),
    }
}

#[async_trait]
#[allow(clippy::needless_pass_by_value)]
impl HarnessProtocol for ScriptedHarness {
    fn card(&self) -> &HarnessCard {
        &self.card
    }

    fn state(&self) -> HarnessState {
        *self.state.lock().unwrap()
    }

    fn provider_session_id(&self) -> Option<&str> {
        None
    }

    fn event_buffer_config(&self) -> &EventBufferConfig {
        &self.buffer
    }

    async fn start(&self, _context: HarnessContext) -> HarnessResult<()> {
        self.calls.lock().unwrap().push(Call::Start);
        *self.state.lock().unwrap() = HarnessState::Idle;
        Ok(())
    }

    async fn stop(&self) -> HarnessResult<()> {
        *self.state.lock().unwrap() = HarnessState::Terminated;
        Ok(())
    }

    fn events(&self) -> Box<dyn HarnessEventCursor> {
        match self.events.lock().unwrap().take() {
            Some(receiver) => Box::new(ScriptCursor { receiver }),
            None => Box::new(VecEventCursor::new(Vec::new())),
        }
    }

    fn turn_events(&self, _turn_id: Option<&str>) -> Box<dyn HarnessEventCursor> {
        Box::new(VecEventCursor::new(Vec::new()))
    }

    async fn send(&self, input: HarnessInput, mode: DeliveryMode) -> HarnessResult<SendReceipt> {
        let text = input.content.as_str().unwrap_or_default().to_owned();
        self.calls.lock().unwrap().push(Call::Send(text));
        *self.state.lock().unwrap() = HarnessState::Running;
        self.emit_round();
        *self.state.lock().unwrap() = HarnessState::Idle;
        Ok(SendReceipt {
            message_id: "message-1".to_owned(),
            turn_id: TURN_ID.to_owned(),
            accepted_mode: mode,
        })
    }

    async fn abort(&self, _mode: AbortMode) -> HarnessResult<()> {
        Ok(())
    }

    async fn pause(&self) -> HarnessResult<()> {
        *self.state.lock().unwrap() = HarnessState::Paused;
        Ok(())
    }

    async fn resume(&self, _query: Option<HarnessInput>) -> HarnessResult<()> {
        *self.state.lock().unwrap() = HarnessState::Running;
        Ok(())
    }

    async fn export_checkpoint(&self) -> HarnessResult<Option<HarnessCheckpoint>> {
        Ok(None)
    }
}

/// Harness factory that builds [`ScriptedHarness`]es and records the contexts it was handed.
#[derive(Default)]
struct FakeHarnessFactory {
    /// Contexts received by [`MemberHarnessFactory::create`], in call order.
    calls: Mutex<Vec<MemberHarnessContext>>,
    /// Per-member answer scripted before the first `create`.
    answers: Mutex<BTreeMap<String, String>>,
    /// Harness built per member, so a scenario can assert what the member received.
    harnesses: Mutex<BTreeMap<String, Arc<ScriptedHarness>>>,
}

impl FakeHarnessFactory {
    /// Builds an absent-minded factory: every member answers `done: <member>`.
    fn scripted() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Scripts the answer of `member`; recorded before its first `create`.
    fn answers_with(&self, member: &str, answer: &str) {
        self.answers
            .lock()
            .unwrap()
            .insert(member.to_owned(), answer.to_owned());
    }

    /// Returns the members a harness was created for, in creation order.
    fn created_members(&self) -> Vec<String> {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .map(|ctx| ctx.member_name.clone())
            .collect()
    }

    /// Returns the harness built for `member`.
    fn harness(&self, member: &str) -> Arc<ScriptedHarness> {
        self.harnesses
            .lock()
            .unwrap()
            .get(member)
            .cloned()
            .expect("a harness was built for the member")
    }

    /// Returns the texts `member`'s harness received, in order.
    fn sends(&self, member: &str) -> Vec<String> {
        self.harness(member).sends()
    }
}

#[async_trait]
impl MemberHarnessFactory for FakeHarnessFactory {
    async fn create(
        &self,
        _spec: &DeepAgentSpec,
        ctx: &MemberHarnessContext,
    ) -> Result<Arc<dyn HarnessProtocol>> {
        self.calls.lock().unwrap().push(ctx.clone());
        let answer = self
            .answers
            .lock()
            .unwrap()
            .get(&ctx.member_name)
            .cloned()
            .unwrap_or_else(|| format!("done: {}", ctx.member_name));
        let harness = ScriptedHarness::scripted(answer);
        self.harnesses
            .lock()
            .unwrap()
            .insert(ctx.member_name.clone(), Arc::clone(&harness));
        Ok(harness)
    }

    fn name(&self) -> &'static str {
        "scripted"
    }
}

/// One scenario fixture: the scripted factory plus the temp directory member workspaces live in.
struct Fixture {
    /// Factory injected through [`TeamBuildDeps`].
    factory: Arc<FakeHarnessFactory>,
    /// Root of every member workspace; dropped with the fixture.
    workspace: TempDir,
}

impl Fixture {
    /// Builds a fixture over a fresh in-memory store and temp workspace root.
    fn new() -> Self {
        Self {
            factory: FakeHarnessFactory::scripted(),
            workspace: TempDir::new().expect("a temp workspace root"),
        }
    }

    /// Builds the injected dependencies: scripted factory, memory store, local workspaces.
    fn deps(&self) -> TeamBuildDeps {
        let workspace =
            LocalWorkspaceProvider::new(MemberWorkspaceLayout::new(self.workspace.path()));
        TeamBuildDeps::default()
            .with_harness_factory(Arc::clone(&self.factory) as Arc<dyn MemberHarnessFactory>)
            .with_store(Arc::new(InMemoryTeamStore::new()) as Arc<dyn TeamStore>)
            .with_workspace(Arc::new(workspace) as Arc<dyn WorkspaceProvider>)
            .with_language("en")
    }
}

/// Builds one member blueprint with a card and a prompt.
fn deep_agent(name: &str, prompt: &str) -> DeepAgentSpec {
    DeepAgentSpec {
        card: Some(AgentCard::new(name).with_description(format!("{name} description"))),
        system_prompt: Some(prompt.to_owned()),
        ..DeepAgentSpec::default()
    }
}

/// Builds the scenario blueprint: team `demo_team`, leader `team_leader`, one teammate `dev`.
fn spec() -> TeamAgentSpec {
    let mut spec = TeamAgentSpec {
        team_name: "demo_team".to_owned(),
        team_desc: Some("a demo team".to_owned()),
        ..TeamAgentSpec::default()
    };
    spec.agents
        .insert("leader".to_owned(), deep_agent("Leader", "lead the team"));
    spec.agents
        .insert("dev".to_owned(), deep_agent("Dev", "do the work"));
    spec.predefined_members
        .push(PredefinedMemberSpec::Teammate(MemberSpecBase {
            member_name: "dev".to_owned(),
            display_name: "Dev".to_owned(),
            desc: "the developer".to_owned(),
            prompt: "you are dev".to_owned(),
            model_name: None,
        }));
    spec
}

/// Builds the scenario team over `fixture`'s injected dependencies.
async fn build(fixture: &Fixture) -> Arc<TeamAgent> {
    let spec = spec();
    within("team build", spec.build_with(fixture.deps()))
        .await
        .expect("the team builds")
}

/// Returns how many frames are team event markers.
fn marker_count(frames: &[Result<AgentChunk>]) -> usize {
    frames
        .iter()
        .filter(|item| {
            item.as_ref()
                .is_ok_and(|chunk| is_team_event_marker(&chunk.payload))
        })
        .count()
}

/// Returns the wire types of a mutation's events, in publication order.
fn event_types(mutation: &TaskMutation) -> Vec<&str> {
    mutation
        .events
        .iter()
        .map(EventPayload::event_type)
        .collect()
}

/// Finds one frame carrying `member`'s scripted answer text.
fn carries_answer(frames: &[Result<AgentChunk>], member: &str, answer: &str) -> bool {
    frames.iter().any(|item| {
        let payload = &item.as_ref().expect("the frame is an answer").payload;
        payload["source_member"] == json!(member)
            && payload["chunk"]["type"] == json!("llm_output")
            && payload["chunk"]["payload"]["content"] == json!(answer)
    })
}

#[tokio::test]
async fn spec_builds_a_team_with_a_scripted_member_factory() {
    let fixture = Fixture::new();
    let agent = build(&fixture).await;

    assert_eq!(agent.team_name(), "demo_team");
    assert_eq!(agent.role(), TeamRole::Leader);
    assert_eq!(agent.member_name(), "team_leader");
    assert_eq!(agent.leader_member_name(), "team_leader");
    assert!(!agent.is_started(), "a built team is not started");
    assert_eq!(
        agent.roster(),
        vec!["dev".to_owned(), "team_leader".to_owned()],
        "the roster is the predefined teammate plus the leader"
    );

    assert_eq!(agent.card().name, "demo_team_team_leader");
    let card = agent.team_card();
    assert_eq!(card.id, "demo_team");
    assert_eq!(card.name, "demo_team");
    let member_cards: Vec<&str> = card
        .agent_cards
        .iter()
        .map(|member| member.name.as_str())
        .collect();
    assert_eq!(
        member_cards,
        ["demo_team_dev", "demo_team_team_leader"],
        "the team card carries one card per roster member, sorted"
    );

    assert_eq!(
        fixture.factory.created_members(),
        vec!["team_leader".to_owned()],
        "the build materializes the leader harness only"
    );

    let snapshot = within("team snapshot", agent.snapshot()).await;
    assert_eq!(snapshot.team_name, "demo_team");
    assert_eq!(snapshot.state, HarnessState::Idle);
    assert!(snapshot.tasks.is_empty(), "the board starts empty");

    let dev = snapshot
        .members
        .iter()
        .find(|member| member.member_name == "dev")
        .expect("dev is on the roster");
    assert_eq!(dev.role, TeamRole::Teammate);
    assert_eq!(dev.display_name, "Dev");
    assert_eq!(
        dev.status,
        MemberStatus::Unstarted,
        "a predefined teammate is materialized unstarted"
    );
    let leader = snapshot
        .members
        .iter()
        .find(|member| member.member_name == "team_leader")
        .expect("the leader is on the roster");
    assert_eq!(leader.role, TeamRole::Leader);
}

#[tokio::test]
async fn leader_dispatch_emits_exactly_one_completion_marker() {
    let fixture = Fixture::new();
    fixture
        .factory
        .answers_with("team_leader", "the leader handled it");
    let agent = build(&fixture).await;

    within("team start", agent.start())
        .await
        .expect("the team starts");
    within("member spawn", agent.spawn_member("dev"))
        .await
        .expect("dev spawns");
    assert!(agent.is_started());
    assert_eq!(
        fixture.factory.created_members(),
        vec!["team_leader".to_owned(), "dev".to_owned()],
        "spawning builds the teammate harness through the same factory"
    );
    assert!(
        fixture
            .factory
            .harness("dev")
            .calls()
            .contains(&Call::Start),
        "spawning starts the teammate's cycle"
    );

    let frames = collect_bounded(agent.stream(json!("ship the release"), None), MAX_FRAMES).await;
    assert!(frames.iter().all(Result::is_ok), "{frames:?}");
    assert!(
        frames.len() >= 2,
        "at least the answer frame and the marker: {frames:?}"
    );

    let last = frames.last().unwrap().as_ref().unwrap();
    assert_eq!(last.kind, TEAM_COMPLETED);
    assert!(is_team_event_marker(&last.payload), "{:?}", last.payload);
    assert_eq!(
        last.payload["chunk"]["payload"]["event_type"],
        json!(TEAM_COMPLETED),
        "the last frame is the completion marker"
    );
    assert_eq!(marker_count(&frames), 1, "exactly one marker: {frames:?}");

    let before_marker = &frames[..frames.len() - 1];
    assert_eq!(
        marker_count(before_marker),
        0,
        "no earlier frame is a team marker: {frames:?}"
    );
    assert!(
        carries_answer(before_marker, "team_leader", "the leader handled it"),
        "one frame carries the leader answer: {frames:?}"
    );
    let leader_sends = fixture.factory.sends("team_leader");
    assert!(
        leader_sends.iter().any(|text| text == "ship the release"),
        "the caller's query reaches the leader verbatim: {leader_sends:?}"
    );
    assert!(
        agent.gate().is_closed(),
        "the run closes the interaction gate"
    );
}

#[tokio::test]
async fn task_board_create_claim_complete_settles_the_board() {
    let fixture = Fixture::new();
    let agent = build(&fixture).await;
    let backend = agent.backend();
    let board = backend.tasks();

    let created = within(
        "task create",
        board.create_task(TaskGraphSpec {
            title: "ship the release".to_owned(),
            content: "cut the tag and publish".to_owned(),
            ..TaskGraphSpec::default()
        }),
    )
    .await
    .expect("the task is created");
    let task_id = created.task.task_id.clone();
    assert_ne!(task_id, "");
    assert_eq!(created.task.status, TaskStatus::Pending);
    assert_eq!(created.task.assignee, None);
    assert!(
        created.task.reviewers.is_empty(),
        "no reviewers means no review gate"
    );
    assert_eq!(
        event_types(&created),
        ["task_created", "task_status_changed"],
        "creation publishes the created event and the initial status row"
    );
    assert_eq!(created.unblocked, Vec::<String>::new());

    let claimed = within("task claim", board.claim(&task_id, "dev"))
        .await
        .expect("the claim is applied");
    assert!(claimed.ok, "{claimed:?}");
    let in_progress = within("task detail", board.detail(&task_id))
        .await
        .expect("the board answers")
        .expect("the task is on the board");
    assert_eq!(in_progress.status, TaskStatus::InProgress);
    assert_eq!(in_progress.assignee.as_deref(), Some("dev"));

    let completed = within("task complete", board.complete(&task_id, "dev", "shipped"))
        .await
        .expect("the completion is applied");
    assert_eq!(completed.task.status, TaskStatus::Completed);
    assert_eq!(completed.task.assignee.as_deref(), Some("dev"));
    assert_eq!(completed.task.summary.as_deref(), Some("shipped"));
    assert_eq!(event_types(&completed), ["task_status_changed"]);
    assert_eq!(
        completed.task.reviewers,
        Vec::<String>::new(),
        "the task completed without a review round"
    );

    assert!(
        within("board settled", board.is_settled())
            .await
            .expect("the board answers"),
        "the only task is terminal, so the board is settled"
    );
    let detail = within("task detail", board.detail(&task_id))
        .await
        .expect("the board answers")
        .expect("the task is still on the board");
    assert_eq!(detail.status, TaskStatus::Completed);
    assert_eq!(detail.summary.as_deref(), Some("shipped"));
    assert_eq!(detail.reviewers, Vec::<String>::new());
}
