//! Agent-teams backend seams: event bus, mailbox/task board, per-member checkpoints, store, attach.
//!
//! The runtime writes every coordination fact through one seam; this suite drives each from outside
//! the crate, the way a host process would:
//!
//! 1. **Event bus** — a subscriber installed before `start()` sees the spawned teammate's
//!    `member_status_changed`, the leader's direct `message` row and a board task's
//!    `task_created`/`task_status_changed` pair, in publication order. A settled run additionally
//!    announces itself once on the bus (`team_completed`) beside the `team.completed` *stream*
//!    marker; the suite pins both halves.
//! 2. **Mailbox and task board** — `send_message`, `unread`, `mark_read` and a board task round-trip
//!    over the injected in-memory store.
//! 3. **Per-member checkpoints** — a member harness exports a protocol-level [`HarnessCheckpoint`]
//!    wrapping a [`NativeCheckpoint`] payload; the payload round-trips through JSON and validates
//!    against its own session/card only. The runtime does not wire this seam itself (the payload is
//!    the durable half a host persists), so the scenario drives the payload API directly.
//! 4. **SQLite store** — the same runtime over `StorageSpec { kind: "sqlite", … }`, then a *second*
//!    store instance over the same file proves the board and mailbox rows are on disk; plus the
//!    cross-process attach host, driven through [`AttachClient`] like the `team-member` CLI.
//!
//! Everything is offline and deterministic: scripted [`HarnessProtocol`] members, one test-owned
//! [`TempDir`] per resource, every await bounded by [`support::BOUND`], no timestamp/id assertions.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod support;

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openjiuwen::agent_teams::harness::{
    MemberHarnessContext, MemberHarnessFactory, NativeCheckpoint,
};
use openjiuwen::agent_teams::messager::EventSink;
use openjiuwen::agent_teams::runtime::{SpecBuild, TeamAgent, TeamBuildDeps};
#[cfg(feature = "sqlite")]
use openjiuwen::agent_teams::schema::StorageSpec;
use openjiuwen::agent_teams::schema::{
    is_team_event_marker, DeepAgentSpec, EventMessage, MemberSpecBase, MemberStatus,
    PredefinedMemberSpec, TaskFilter, TaskGraphSpec, TaskStatus, TeamAgentSpec, TeamRole,
    TeamTopic, TEAM_COMPLETED,
};
#[cfg(feature = "sqlite")]
use openjiuwen::agent_teams::storage::open_store;
use openjiuwen::agent_teams::storage::{InMemoryTeamStore, TeamStore};
use openjiuwen::agent_teams::workspace::{
    LocalWorkspaceProvider, MemberWorkspaceLayout, WorkspaceProvider,
};
use openjiuwen::core::common::error::Result;
use openjiuwen::core::single_agent::{AgentCard, AgentChunk};
use openjiuwen::harness_protocol::{
    AbortMode, DeliveryMode, EventBufferConfig, HarnessCard, HarnessCheckpoint, HarnessContext,
    HarnessEvent, HarnessEventCursor, HarnessEventPayload, HarnessInput, HarnessProtocol,
    HarnessResult, HarnessState, OutputChannel, OutputEvent, OutputKind, OutputOperation,
    SendReceipt, VecEventCursor,
};
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::sync::mpsc;

use support::{collect_bounded, within};

/// Team name every blueprint in this suite builds.
const TEAM: &str = "demo_team";

/// Leader member name the runtime defaults to for [`TEAM`].
const LEADER: &str = "team_leader";

/// The predefined teammate the roster starts with.
const DEV: &str = "dev";

/// Card id each scripted member checkpoints under (the runtime does not hand one to the factory).
const CARD_ID: &str = "card-dev";

/// Session fragment every scripted observation is correlated with.
const TURN_ID: &str = "turn-1";

/// Agent id the scripted harness reports on the wire.
const AGENT_ID: &str = "scripted";

/// Provider name the scripted checkpoint envelope is stamped with.
const CHECKPOINT_PROVIDER: &str = "native";

/// Upper bound on the frames one run may produce before [`collect_bounded`] gives up.
const MAX_FRAMES: usize = 512;

/// Cursor over the scripted observation channel.
struct ScriptCursor {
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

/// Harness protocol handle that answers every `send` with one scripted answer observation.
///
/// `start` settles the harness to [`HarnessState::Idle`]; `send` queues the answer and returns to
/// `Idle`, which is what lets the leader's run settle on member state. Every delivered input is
/// recorded, so a scenario can assert what the member received, and the optional checkpoint is what
/// [`HarnessProtocol::export_checkpoint`] hands back.
struct ScriptedHarness {
    state: Mutex<HarnessState>,
    /// Text of every `send`, in delivery order.
    sends: Mutex<Vec<String>>,
    card: HarnessCard,
    buffer: EventBufferConfig,
    /// The single observation receiver handed to the first `events()` caller.
    events: Mutex<Option<mpsc::UnboundedReceiver<HarnessEvent>>>,
    sender: mpsc::UnboundedSender<HarnessEvent>,
    answer: String,
    sequence: AtomicU64,
    /// Checkpoint the member exports, scripted by the factory from the member's context.
    checkpoint: Mutex<Option<HarnessCheckpoint>>,
}

impl ScriptedHarness {
    /// Builds a harness that answers every input with `answer` and returns to `Idle`.
    fn scripted(answer: impl Into<String>) -> Arc<Self> {
        let (sender, receiver) = mpsc::unbounded_channel();
        Arc::new(Self {
            state: Mutex::new(HarnessState::Idle),
            sends: Mutex::new(Vec::new()),
            card: HarnessCard::new("scripted", "0.0.0"),
            buffer: EventBufferConfig::default(),
            events: Mutex::new(Some(receiver)),
            sender,
            answer: answer.into(),
            sequence: AtomicU64::new(0),
            checkpoint: Mutex::new(None),
        })
    }

    /// Scripts the checkpoint [`HarnessProtocol::export_checkpoint`] returns.
    fn set_checkpoint(&self, checkpoint: HarnessCheckpoint) {
        *self.checkpoint.lock().unwrap() = Some(checkpoint);
    }

    /// Returns the text of every delivered input, in order.
    fn sends(&self) -> Vec<String> {
        self.sends.lock().unwrap().clone()
    }

    /// Queues one observation, advancing the envelope sequence.
    fn emit(&self, payload: HarnessEventPayload) {
        let sequence = self.sequence.fetch_add(1, Ordering::SeqCst);
        let event = HarnessEvent::new(
            sequence + 1,
            1.0,
            payload,
            format!("session:{AGENT_ID}"),
            AGENT_ID,
        )
        .with_turn_id(TURN_ID);
        let _ = self.sender.send(event);
    }

    /// Queues the one answer/`text`/`delta` observation a round produces.
    ///
    /// The run settles on the harness state and the board, not on a lifecycle envelope, so a single
    /// answer observation is the whole script.
    fn emit_answer(&self) {
        self.emit(
            OutputEvent {
                output_id: "out-1".to_owned(),
                kind: OutputKind::Text,
                content: json!(self.answer),
                operation: OutputOperation::Delta,
                channel: OutputChannel::Answer,
                content_index: 0,
                content_type: None,
                data: BTreeMap::new(),
            }
            .into(),
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
        *self.state.lock().unwrap()
    }

    fn provider_session_id(&self) -> Option<&str> {
        None
    }

    fn event_buffer_config(&self) -> &EventBufferConfig {
        &self.buffer
    }

    async fn start(&self, _context: HarnessContext) -> HarnessResult<()> {
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
        self.sends.lock().unwrap().push(text);
        *self.state.lock().unwrap() = HarnessState::Running;
        self.emit_answer();
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
        Ok(self.checkpoint.lock().unwrap().clone())
    }
}

/// Builds the checkpoint envelope a member exports: a [`NativeCheckpoint`] payload inside the
/// protocol-level [`HarnessCheckpoint`] the host persists.
///
/// The payload's `session_id` follows the documented `"<team_session>:<member_name>"` convention: the
/// context's `session_id` is the *team* host session, shared by every member, so the member name is
/// what makes a persisted payload restorable for one member only. `card_id` is the scripted constant
/// [`CARD_ID`] — [`MemberHarnessContext`] carries no card id, and a real factory would resolve the
/// card it built the member from.
fn checkpoint_envelope(ctx: &MemberHarnessContext, answer: &str) -> Result<HarnessCheckpoint> {
    let session_id = format!("{}:{}", ctx.session_id, ctx.member_name);
    let native = NativeCheckpoint::new(session_id, CARD_ID)
        .with_state(HarnessState::Paused)
        .with_turn_id(TURN_ID)
        .with_outputs(json!({"answer_parts": [answer]}));
    // The envelope carries the payload as a `BTreeMap`, so mirror the payload's own keys.
    let data = match native.to_json()? {
        Value::Object(object) => object.into_iter().collect(),
        _ => BTreeMap::new(),
    };
    Ok(HarnessCheckpoint {
        provider: CHECKPOINT_PROVIDER.to_owned(),
        schema_version: "1".to_owned(),
        agent_id: AGENT_ID.to_owned(),
        host_session_id: ctx.session_id.clone(),
        checkpoint_id: "ckpt-1".to_owned(),
        sequence: 0,
        data,
        provider_session_id: None,
        revision: None,
    })
}

/// Harness factory that builds [`ScriptedHarness`]es and records the contexts it was handed.
#[derive(Default)]
struct FakeHarnessFactory {
    /// Contexts received by [`MemberHarnessFactory::create`], in call order.
    calls: Mutex<Vec<MemberHarnessContext>>,
    harnesses: Mutex<BTreeMap<String, Arc<ScriptedHarness>>>,
}

impl FakeHarnessFactory {
    /// Builds an absent-minded factory: every member answers `done: <member>`.
    fn scripted() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Returns the context `member`'s harness was built from.
    fn context(&self, member: &str) -> MemberHarnessContext {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .find(|ctx| ctx.member_name == member)
            .cloned()
            .unwrap_or_else(|| panic!("no harness was built for '{member}'"))
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
        let answer = format!("done: {}", ctx.member_name);
        let harness = ScriptedHarness::scripted(answer.clone());
        harness.set_checkpoint(checkpoint_envelope(ctx, &answer)?);
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
    /// Builds a fixture over a fresh temp workspace root.
    fn new() -> Self {
        Self {
            factory: FakeHarnessFactory::scripted(),
            workspace: TempDir::new().expect("a temp workspace root"),
        }
    }

    /// Builds the injected dependencies over a fresh in-memory store.
    fn deps(&self) -> TeamBuildDeps {
        self.deps_with_store(Arc::new(InMemoryTeamStore::new()))
    }

    /// Builds the injected dependencies over `store`: scripted factory, local workspaces.
    fn deps_with_store(&self, store: Arc<dyn TeamStore>) -> TeamBuildDeps {
        let workspace =
            LocalWorkspaceProvider::new(MemberWorkspaceLayout::new(self.workspace.path()));
        TeamBuildDeps::default()
            .with_harness_factory(Arc::clone(&self.factory) as Arc<dyn MemberHarnessFactory>)
            .with_store(store)
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

/// Builds the scenario blueprint: team [`TEAM`], leader [`LEADER`], one teammate [`DEV`].
fn spec() -> TeamAgentSpec {
    let mut spec = TeamAgentSpec {
        team_name: TEAM.to_owned(),
        team_desc: Some("a demo team".to_owned()),
        ..TeamAgentSpec::default()
    };
    spec.agents
        .insert("leader".to_owned(), deep_agent("Leader", "lead the team"));
    spec.agents
        .insert(DEV.to_owned(), deep_agent("Dev", "do the work"));
    spec.predefined_members
        .push(PredefinedMemberSpec::Teammate(MemberSpecBase {
            member_name: DEV.to_owned(),
            display_name: "Dev".to_owned(),
            desc: "the developer".to_owned(),
            prompt: "you are dev".to_owned(),
            model_name: None,
        }));
    spec
}

/// Builds the scenario team over `fixture`'s default injected dependencies.
async fn build(fixture: &Fixture) -> Arc<TeamAgent> {
    within("team build", spec().build_with(fixture.deps()))
        .await
        .expect("the team builds")
}

/// Returns the wire types of `events`, in publication order.
fn event_types(events: &[EventMessage]) -> Vec<&str> {
    events
        .iter()
        .map(|event| event.event_type.as_str())
        .collect()
}

/// Returns the index of the first event whose type and payload match `predicate`.
fn find_event(
    events: &[EventMessage],
    event_type: &str,
    predicate: impl Fn(&Value) -> bool,
) -> Option<usize> {
    events
        .iter()
        .position(|event| event.event_type == event_type && predicate(&event.payload))
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

/// Asserts the four coordination publications plus the run's completion, in publication order.
///
/// A settled run reports its completion twice: the `team.completed` stream frame (asserted by the
/// caller) and exactly one `team_completed` bus publication, so an observer holding no stream still
/// learns the team completed. Both are part of the contract, so the completion is required here and
/// pinned between the mail the run follows and the task the caller creates afterwards.
fn assert_bus_publications(events: &[EventMessage], mail_id: &str) {
    assert_eq!(
        event_types(events),
        [
            "member_status_changed",
            "message",
            "team_completed",
            "task_created",
            "task_status_changed"
        ],
        "the bus carries the coordination publications and one completion, in order: {events:?}"
    );

    let status = find_event(events, "member_status_changed", |payload| {
        payload["member_name"] == json!(DEV)
    })
    .expect("the spawn publishes the teammate's lifecycle");
    assert_eq!(events[status].payload["team_name"], json!(TEAM));
    assert_eq!(events[status].payload["status"], json!("ready"));
    assert_eq!(
        events[status].payload["previous"],
        json!("unstarted"),
        "the publication names the transition, not just the new state"
    );

    let created = find_event(events, "task_created", |payload| {
        payload["task_id"] == json!("task-1")
    })
    .expect("creating the task publishes it");
    assert_eq!(events[created].payload["title"], json!("ship the release"));
    assert_eq!(events[created].payload["team_name"], json!(TEAM));

    let moved = find_event(events, "task_status_changed", |payload| {
        payload["task_id"] == json!("task-1")
    })
    .expect("the initial status row is published with the creation");
    assert_eq!(events[moved].payload["status"], json!("pending"));

    let message = find_event(events, "message", |payload| {
        payload["message_id"] == json!(mail_id)
    })
    .expect("the direct mail is published");
    assert_eq!(events[message].payload["from_member_name"], json!(LEADER));
    assert_eq!(events[message].payload["to_member_name"], json!(DEV));

    assert!(
        status < message && message < created && created < moved,
        "publication order follows the calls: {events:?}"
    );

    let completed = find_event(events, "team_completed", |payload| {
        payload["team_name"] == json!(TEAM)
    })
    .expect("the settled run publishes its completion on the bus");
    assert_eq!(
        events[completed].payload["reason"],
        json!("settled"),
        "the settlement names why the team completed: {events:?}"
    );
    assert!(
        message < completed && completed < created,
        "the settled run publishes its completion after the mail and before the caller's task: \
         {events:?}"
    );
}

#[tokio::test]
async fn event_bus_records_member_task_and_mail_publications_in_order() {
    let fixture = Fixture::new();
    let agent = build(&fixture).await;

    // Subscribe *before* the team starts so nothing the lifecycle publishes is missed.
    let events: Arc<Mutex<Vec<EventMessage>>> = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&events);
    let sink: EventSink = Arc::new(move |event: EventMessage| recorded.lock().unwrap().push(event));
    agent
        .backend()
        .messager()
        .subscribe(
            "observer",
            &[TeamTopic::Team, TeamTopic::Task, TeamTopic::Message],
            sink,
        )
        .expect("the observer subscribes");

    within("team start", agent.start())
        .await
        .expect("the team starts");
    assert!(
        events.lock().unwrap().is_empty(),
        "start alone publishes no bus event"
    );

    within("member spawn", agent.spawn_member(DEV))
        .await
        .expect("dev spawns");
    let mail = within("direct mail", agent.send_message(Some(DEV), "ping"))
        .await
        .expect("the mail is persisted");

    // The run's completion reaches the caller as a *stream* marker; observers holding no stream see
    // the same settlement as one `team_completed` bus publication, asserted below.
    let frames = collect_bounded(agent.stream(json!("ship the release"), None), MAX_FRAMES).await;
    assert!(frames.iter().all(Result::is_ok), "{frames:?}");
    let last = frames.last().unwrap().as_ref().unwrap();
    assert_eq!(last.kind, TEAM_COMPLETED);
    assert!(is_team_event_marker(&last.payload), "{:?}", last.payload);
    assert_eq!(marker_count(&frames), 1, "exactly one marker: {frames:?}");
    assert!(
        fixture
            .factory
            .sends(LEADER)
            .iter()
            .any(|text| text == "ship the release"),
        "the caller's query reaches the leader verbatim"
    );

    // An empty board is what let the run settle, so the task is created last; the exact type list
    // in `assert_bus_publications` covers everything the settled run itself published.
    let task = within(
        "task create",
        agent.create_task(TaskGraphSpec {
            title: "ship the release".to_owned(),
            content: "cut the tag and publish".to_owned(),
            task_id: Some("task-1".to_owned()),
            ..TaskGraphSpec::default()
        }),
    )
    .await
    .expect("the task is created");
    assert_eq!(task.task_id, "task-1");

    let snapshot = events.lock().unwrap().clone();
    assert_bus_publications(&snapshot, &mail.message_id);

    // A blocking `invoke` is the same run without a stream: there is no frame to observe, so the
    // bus publication is the *only* completion its caller-side observers can see, and it must
    // still arrive exactly once.
    let blocking = Fixture::new();
    let blocking_agent = build(&blocking).await;
    let blocking_events: Arc<Mutex<Vec<EventMessage>>> = Arc::new(Mutex::new(Vec::new()));
    let blocking_recorded = Arc::clone(&blocking_events);
    let blocking_sink: EventSink =
        Arc::new(move |event: EventMessage| blocking_recorded.lock().unwrap().push(event));
    blocking_agent
        .backend()
        .messager()
        .subscribe("observer", &[TeamTopic::Team], blocking_sink)
        .expect("the observer subscribes");
    within(
        "blocking team run",
        blocking_agent.invoke(json!("ship the release"), None),
    )
    .await
    .expect("the blocking run answers");
    let blocking_snapshot = blocking_events.lock().unwrap().clone();
    assert_eq!(
        event_types(&blocking_snapshot),
        ["team_completed"],
        "a blocking run has no stream, so the bus carries the whole completion: {blocking_snapshot:?}"
    );
    assert_eq!(blocking_snapshot[0].payload["team_name"], json!(TEAM));
    assert_eq!(blocking_snapshot[0].payload["reason"], json!("settled"));
    assert!(
        blocking
            .factory
            .sends(LEADER)
            .iter()
            .any(|text| text == "ship the release"),
        "the blocking run's query reaches the leader verbatim"
    );
}

#[tokio::test]
async fn mailbox_and_task_board_read_back_through_the_backend() {
    let fixture = Fixture::new();
    let agent = build(&fixture).await;
    within("team start", agent.start())
        .await
        .expect("the team starts");
    let backend = agent.backend();

    let row = within("direct mail", agent.send_message(Some(DEV), "hello dev"))
        .await
        .expect("the mail is persisted");
    assert_eq!(row.team_name, TEAM);
    assert_eq!(row.from_member_name, LEADER);
    assert_eq!(row.to_member_name.as_deref(), Some(DEV));
    assert_eq!(row.content, "hello dev");
    assert!(!row.is_broadcast());

    let unread = within("dev inbox", backend.unread(DEV))
        .await
        .expect("the mailbox answers");
    assert_eq!(unread.len(), 1, "the row is in the recipient's inbox");
    assert_eq!(unread[0].message_id, row.message_id);
    assert_eq!(unread[0].content, "hello dev");
    assert!(!unread[0].is_read);
    assert!(
        within("leader inbox", backend.unread(LEADER))
            .await
            .expect("the mailbox answers")
            .is_empty(),
        "a direct row is not delivered to the sender"
    );

    within(
        "mark read",
        backend.mark_read(DEV, std::slice::from_ref(&row.message_id)),
    )
    .await
    .expect("the read flag is applied");
    assert!(
        within("dev inbox", backend.unread(DEV))
            .await
            .expect("the mailbox answers")
            .is_empty(),
        "a read row leaves the inbox"
    );

    let created = within(
        "task create",
        backend.tasks().create_task(TaskGraphSpec {
            title: "persist the board".to_owned(),
            content: "write it down".to_owned(),
            task_id: Some("task-board".to_owned()),
            assignee: Some(DEV.to_owned()),
            ..TaskGraphSpec::default()
        }),
    )
    .await
    .expect("the task is created");
    assert_eq!(created.task.status, TaskStatus::Pending);

    let board = within("board view", backend.tasks().view(&TaskFilter::default()))
        .await
        .expect("the board answers");
    assert_eq!(board.len(), 1, "one row on the board: {board:?}");
    assert_eq!(board[0].task_id, "task-board");
    assert_eq!(board[0].status, TaskStatus::Pending);
    assert_eq!(board[0].assignee.as_deref(), Some(DEV));

    let detail = within("task detail", backend.tasks().detail("task-board"))
        .await
        .expect("the board answers")
        .expect("the task is on the board");
    assert_eq!(detail.title, "persist the board");
    assert_eq!(detail.content, "write it down");
}

#[tokio::test]
async fn member_checkpoint_round_trips_and_validates_its_identity() {
    let fixture = Fixture::new();
    let agent = build(&fixture).await;
    within("team start", agent.start())
        .await
        .expect("the team starts");
    within("member spawn", agent.spawn_member(DEV))
        .await
        .expect("dev spawns");

    let ctx = fixture.factory.context(DEV);
    // The context carries the *team* host session, shared by every member; the payload's own
    // session id is the documented `"<host>:<member>"` pair that makes it member-specific.
    let checkpoint_session = format!("{}:{DEV}", ctx.session_id);

    let harness = fixture.factory.harness(DEV);
    let envelope = within("checkpoint export", harness.export_checkpoint())
        .await
        .expect("the export answers")
        .expect("the member exports a checkpoint");
    assert_eq!(envelope.provider, CHECKPOINT_PROVIDER);
    assert_eq!(envelope.schema_version, "1");
    assert_eq!(envelope.agent_id, AGENT_ID);
    assert_eq!(envelope.host_session_id, ctx.session_id);
    assert_eq!(envelope.sequence, 0);
    assert!(envelope.provider_session_id.is_none());
    assert!(envelope.revision.is_none());

    // The opaque payload is the durable half: decode it exactly as a host would after a restart.
    let payload = Value::Object(envelope.data.clone().into_iter().collect());
    let native = NativeCheckpoint::from_json(payload.clone()).expect("the payload decodes");
    assert_eq!(native.session_id, checkpoint_session);
    assert_eq!(native.card_id, CARD_ID);
    assert_eq!(native.state, HarnessState::Paused);
    assert_eq!(native.turn_id.as_deref(), Some(TURN_ID));
    assert_eq!(native.cwd, None);
    assert_eq!(
        native.outputs,
        json!({"answer_parts": [format!("done: {DEV}")]}),
        "the accumulated output survives the envelope"
    );
    assert_eq!(
        NativeCheckpoint::from_json(native.to_json().expect("re-encoded"))
            .expect("the re-encoded payload decodes"),
        native,
        "the payload round-trips through JSON byte-for-byte"
    );

    native
        .validate_compatible(&checkpoint_session, CARD_ID)
        .expect("the payload is restorable into its own session and card");
    let session_error = native
        .validate_compatible("someone-else:dev", CARD_ID)
        .expect_err("a foreign session is rejected");
    assert!(
        session_error.message().contains(&format!(
            "checkpoint belongs to session '{checkpoint_session}'"
        )),
        "{session_error}"
    );
    let card_error = native
        .validate_compatible(&checkpoint_session, "other-card")
        .expect_err("a foreign card is rejected");
    assert!(
        card_error
            .message()
            .contains(&format!("checkpoint belongs to card '{CARD_ID}'")),
        "{card_error}"
    );
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn sqlite_store_persists_board_and_mail_across_instances() {
    let dir = TempDir::new().expect("a team-database temp dir");
    let path = dir.path().join("team.db");
    let storage = StorageSpec {
        kind: "sqlite".to_owned(),
        params: BTreeMap::from([(
            "path".to_owned(),
            json!(path.to_string_lossy().into_owned()),
        )]),
    };
    let store = open_store(&storage).expect("the sqlite store opens");
    assert_eq!(store.name(), "sqlite", "a `sqlite` spec selects the store");

    let fixture = Fixture::new();
    let agent = within(
        "team build",
        spec().build_with(fixture.deps_with_store(Arc::clone(&store))),
    )
    .await
    .expect("the team builds over the sqlite store");
    within("team start", agent.start())
        .await
        .expect("the team starts");

    let task = within(
        "task create",
        agent.create_task(TaskGraphSpec {
            title: "durable task".to_owned(),
            content: "write it to disk".to_owned(),
            task_id: Some("task-sqlite".to_owned()),
            assignee: Some(DEV.to_owned()),
            ..TaskGraphSpec::default()
        }),
    )
    .await
    .expect("the task is created");
    assert_eq!(task.task_id, "task-sqlite");
    let mail = within("direct mail", agent.send_message(Some(DEV), "durable mail"))
        .await
        .expect("the mail is persisted");
    assert_eq!(mail.content, "durable mail");

    // Stop the team and drop every handle on the pool, so the reopen is a real reopen.
    within("team stop", agent.stop())
        .await
        .expect("the team stops");
    drop(agent);
    drop(store);

    let reopened = open_store(&storage).expect("the same database reopens");
    assert_eq!(reopened.name(), "sqlite");
    let tasks = within(
        "task list",
        reopened.list_tasks(TEAM, &TaskFilter::default()),
    )
    .await
    .expect("the reopened store answers");
    assert_eq!(tasks.len(), 1, "the board survived the reopen: {tasks:?}");
    assert_eq!(tasks[0].task_id, "task-sqlite");
    assert_eq!(tasks[0].status, TaskStatus::Pending);
    assert_eq!(tasks[0].assignee.as_deref(), Some(DEV));
    assert_eq!(tasks[0].title, "durable task");

    let unread = within("dev inbox", reopened.unread_messages(TEAM, DEV))
        .await
        .expect("the reopened store answers");
    assert_eq!(
        unread.len(),
        1,
        "the mailbox survived the reopen: {unread:?}"
    );
    assert_eq!(unread[0].message_id, mail.message_id);
    assert_eq!(unread[0].content, "durable mail");
    assert_eq!(unread[0].from_member_name, LEADER);
    assert!(!unread[0].is_read);

    // The reopened database is writable, not just readable.
    within(
        "task claim",
        reopened.update_task(
            TEAM,
            "task-sqlite",
            &openjiuwen::agent_teams::schema::TaskPatch::new().with_note("reopened"),
        ),
    )
    .await
    .expect("the reopened store accepts a write");
    let patched = within("task read", reopened.task(TEAM, "task-sqlite"))
        .await
        .expect("the reopened store answers")
        .expect("the task row is still there");
    assert_eq!(patched.note.as_deref(), Some("reopened"));
    assert_eq!(
        patched.status,
        TaskStatus::Pending,
        "the patch leaves the status untouched"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn attach_socket_drives_the_live_team_from_outside() {
    use openjiuwen::agent_teams::AttachClient;
    use openjiuwen::core::common::error::ErrorKind;

    let fixture = Fixture::new();
    // Short `sun_path`: the socket lives in its own tempdir, not the workspace root.
    let dir = tempfile::Builder::new()
        .prefix("ojb")
        .tempdir()
        .expect("a temp socket dir");
    let socket = dir.path().join("team.sock");
    let socket_text = socket.to_str().expect("a UTF-8 socket path").to_owned();

    let mut blueprint = spec();
    blueprint
        .metadata
        .insert("attach".to_owned(), json!({"socket": socket_text}));
    let agent = within("team build", blueprint.build_with(fixture.deps()))
        .await
        .expect("the team builds");
    assert_eq!(agent.attach_socket(), None, "a build publishes no host");

    within("team start", agent.start())
        .await
        .expect("the team starts, which binds the socket");
    assert_eq!(
        agent.attach_socket(),
        Some(socket.clone()),
        "start publishes the bound path"
    );

    let mut client = within("attach connect", AttachClient::connect(&socket))
        .await
        .expect("the socket client connects");

    let roster = within("attach roster", client.list_members())
        .await
        .expect("the host lists the roster");
    let roster_names: Vec<&str> = roster
        .iter()
        .map(|member| member.member_name.as_str())
        .collect();
    assert_eq!(
        roster_names,
        [DEV, LEADER],
        "the hosting backend reports the whole roster"
    );
    let leader = roster
        .iter()
        .find(|member| member.member_name == LEADER)
        .expect("the leader is on the roster");
    assert_eq!(leader.role, TeamRole::Leader);
    assert_eq!(
        leader.status,
        MemberStatus::Ready,
        "start readied the leader"
    );

    within("member spawn", agent.spawn_member(DEV))
        .await
        .expect("dev spawns");
    let roster = within("attach roster", client.list_members())
        .await
        .expect("the host lists the roster");
    let dev = roster
        .iter()
        .find(|member| member.member_name == DEV)
        .expect("the teammate is on the roster");
    assert_eq!(dev.role, TeamRole::Teammate);
    assert_eq!(dev.status, MemberStatus::Ready, "the spawn settled dev");

    let snapshot = within("attach snapshot", client.snapshot())
        .await
        .expect("the host snapshots the team");
    assert_eq!(snapshot.team_name, TEAM);

    within("attach mail", client.send_member(DEV, "hello from outside"))
        .await
        .expect("the host accepts mail");
    let unread = within("dev inbox", agent.backend().unread(DEV))
        .await
        .expect("the mailbox answers");
    assert!(
        unread
            .iter()
            .any(|row| row.content == "hello from outside" && row.from_member_name == LEADER),
        "the socket-written row lands in the recipient's stored mailbox: {unread:?}"
    );

    let reply = within("attach stop", client.stop_team())
        .await
        .expect("the host answers the stop request");
    assert_eq!(
        reply,
        json!({"stopping": true}),
        "the op asks for a stop; the host owns the teardown"
    );

    within("team stop", agent.stop())
        .await
        .expect("the team stops");
    assert_eq!(agent.attach_socket(), None, "stop drops the host");
    assert!(!socket.exists(), "stop removes the socket file");

    let error = within("attach after stop", client.list_members())
        .await
        .expect_err("the hosted team is gone");
    assert_eq!(error.kind, ErrorKind::Transport, "{}", error.message);
}
