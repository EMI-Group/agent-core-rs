//! Team memory cross-crate scenario: the runtime-owned wiring, the member toolkit, and
//! `TEAM_MEMORY.md`.
//!
//! `openjiuwen::core::memory::team` owns the team-memory engine (facade, lifecycle hooks,
//! toolkit, document), and `openjiuwen::agent_teams` now *drives* it: the build path resolves one
//! [`TeamMemoryWiring`] per team from `metadata["memory"]` (or uses the one a host injects), and
//! the runtime then runs every lifecycle hook itself. This suite drives that composition from
//! outside both crates, the way a team host does:
//!
//! 1. **Runtime-driven lifecycle** — a team configured through `metadata["memory"]` with a file
//!    store opens one wiring; `start()` installs the coordination session, `spawn_member` writes
//!    one `MemberProfile`, a delivered mailbox message becomes exactly one buffered turn,
//!    `shutdown_member` writes the member's handoff note, and `stop()` flushes the buffered turn
//!    into a `Decision` and exports `TEAM_MEMORY.md` at the wiring's document path.
//! 2. **Injection precedence** — an injected wiring wins over a conflicting `metadata["memory"]`
//!    block: the team reports the injected wiring by pointer and the ignored block's store
//!    directory is never touched.
//! 3. **Member toolkit** — the runtime installs the team's [`MemberMemoryToolkit`] on every
//!    member's `MemberHarnessContext`, so a member mounts four `team_memory_*` tools; a plain team
//!    mounts none. Invoking `team_memory_remember` through a mounted tool writes a record the
//!    team's own facade reads back.
//! 4. **Document** — a wiring's `export` writes `TEAM_MEMORY.md`, and both the open-time import
//!    and an explicit `import_from` bring it back into a fresh wiring with the records intact and
//!    no duplicates.
//! 5. **Read-only refusal** — a `read_only()` member toolkit refuses `remember`/`forget` with a
//!    typed [`Error::Unsupported`] plus the documented envelope, while search keeps working.
//! 6. **Task board composition** — the runtime still has no task-board → team-memory bridge
//!    ([`TeamMemoryHooks`] has no public way to buffer a task note), so the completed board task's
//!    own status event is turned into the member turn the hooks extract, at the caller — see the
//!    note on [`a_completed_board_task_becomes_a_flushed_member_memory`].
//!
//! Everything is offline and deterministic: hand-written scripted harnesses over a test-owned
//! workspace root, the process-local team memory store (plus one [`TempDir`] for a file store),
//! and every await bounded by [`support::BOUND`]. No assertion depends on a timestamp, a generated
//! id, or the wall clock.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod support;

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openjiuwen::agent_teams::harness::{MemberHarnessContext, MemberHarnessFactory};
use openjiuwen::agent_teams::runtime::{SpecBuild, TeamAgent, TeamBuildDeps};
use openjiuwen::agent_teams::schema::{
    DeepAgentSpec, MemberLifetime, TaskGraphSpec, TaskStatus, TeamAgentSpec,
};
use openjiuwen::agent_teams::storage::{InMemoryTeamStore, TeamStore};
use openjiuwen::agent_teams::taskboard::TaskBoard;
use openjiuwen::agent_teams::workspace::{
    LocalWorkspaceProvider, MemberWorkspaceLayout, WorkspaceProvider,
};
use openjiuwen::agent_teams::{
    MemoryConfig, MemoryStoreConfig, TeamMemoryWiring, MEMORY_METADATA_KEY,
};
use openjiuwen::core::common::error::{Error, Result};
use openjiuwen::core::foundation::tool::{Tool, ToolContext};
use openjiuwen::core::memory::team::{
    document_team_id, truncate_for_prompt, MemberMemoryContext, MemberMemoryToolkit, TeamMemory,
    TeamMemoryDocument, TeamMemoryFilter, TeamMemoryHookConfig, TeamMemoryHooks, TeamMemoryKind,
    TeamMemoryRecord, TeamMemoryScope, TeamSearchOptions, TeamTurn, TeamVisibility, HOOK_SOURCE,
    RULE_SOURCE, TEAM_MEMORY_FILE, TEAM_MEMORY_FORMAT, TOOL_FORGET, TOOL_REMEMBER, TOOL_SEARCH,
    TOOL_STATS,
};
use openjiuwen::core::memory::CallOptions;
use openjiuwen::core::single_agent::AgentCard;
use openjiuwen::harness_protocol::{
    AbortMode, DeliveryMode, EventBufferConfig, HarnessCard, HarnessCheckpoint, HarnessContext,
    HarnessEvent, HarnessEventCursor, HarnessInput, HarnessProtocol, HarnessResult, HarnessState,
    OutputChannel, OutputEvent, OutputKind, OutputOperation, SendReceipt, VecEventCursor,
};
use serde_json::json;
use tempfile::TempDir;
use tokio::sync::mpsc;

use support::within;

/// Team every fixture reads and writes.
const TEAM: &str = "review_team";

/// The team's leader member; the one member the build path materializes itself.
const LEADER: &str = "team_leader";

/// The teammate the runtime spawns in the lifecycle scenarios.
const ALICE: &str = "alice";

/// The second member the engine-level scenarios attribute records to.
const BOB: &str = "bob";

/// The coordination session the engine-level scenarios bind directly.
const SESSION: &str = "session-1";

/// A turn the rule extractor classifies as a `Decision`.
const DECISION: &str = "We decided to use SQLite for the shared store.";

/// A turn the rule extractor classifies as a `HandoffNote` (the `lesson` section).
const LESSON: &str = "Next time we should run the migration first";

/// A turn the rule extractor classifies as a `Task` (`action item` marker).
const ACTION_ITEM: &str = "Action item: ship the release";

/// Agent id the scripted harness reports on the wire.
const AGENT_ID: &str = "scripted";

/// One buffered team turn.
///
/// The timestamp is left to [`TeamTurn`]'s serde default (the Unix epoch), so no scenario depends
/// on the wall clock and no time type is named in this crate.
fn turn(from_member: &str, content: &str) -> TeamTurn {
    serde_json::from_value(json!({"from_member": from_member, "content": content}))
        .expect("a team turn")
}

// ---------------------------------------------------------------------------
// Scripted harness: the member protocol every scenario's factory hands out.
// ---------------------------------------------------------------------------

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

/// Harness protocol handle that answers every `send` with one scripted answer observation.
///
/// None of these scenarios drives a run through `stream`/`invoke` (the lifecycle hooks fire on
/// membership and delivery, not on turns), so the harness only has to be a lawful protocol handle:
/// `start` settles to [`HarnessState::Idle`], `send` emits one answer/`text`/`delta` output and
/// returns to `Idle`, and `stop` terminates it.
struct ScriptedHarness {
    /// State reported by [`HarnessProtocol::state`].
    state: Mutex<HarnessState>,
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
            card: HarnessCard::new("scripted", "0.0.0"),
            buffer: EventBufferConfig::default(),
            events: Mutex::new(Some(receiver)),
            sender,
            answer: answer.into(),
            sequence: AtomicU64::new(0),
        })
    }

    /// Builds one answer/`text`/`delta` output observation.
    fn answer_event(&self) -> HarnessEvent {
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
            format!("session:{AGENT_ID}"),
            AGENT_ID,
        )
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

    async fn send(&self, _input: HarnessInput, mode: DeliveryMode) -> HarnessResult<SendReceipt> {
        *self.state.lock().unwrap() = HarnessState::Running;
        let _ = self.sender.send(self.answer_event());
        *self.state.lock().unwrap() = HarnessState::Idle;
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

/// Harness factory that builds [`ScriptedHarness`]es, records every context, and keeps the
/// team-memory tools the runtime mounted on each member.
#[derive(Default)]
struct FakeHarnessFactory {
    /// Contexts received by [`MemberHarnessFactory::create`], in call order.
    calls: Mutex<Vec<MemberHarnessContext>>,
    /// The tools [`MemberHarnessContext::memory_tools`] reported per member, in mount order.
    tools: Mutex<BTreeMap<String, Vec<Arc<dyn Tool>>>>,
}

impl FakeHarnessFactory {
    /// Builds an absent-minded factory: every member answers `done: <member>`.
    fn scripted() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Returns the context the runtime handed the factory for `member`.
    fn context(&self, member: &str) -> MemberHarnessContext {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .find(|ctx| ctx.member_name == member)
            .cloned()
            .unwrap_or_else(|| panic!("no harness was built for `{member}`"))
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

    /// Returns the tools mounted on `member`, as the factory saw them.
    fn mounted_tools(&self, member: &str) -> Vec<Arc<dyn Tool>> {
        self.tools
            .lock()
            .unwrap()
            .get(member)
            .cloned()
            .unwrap_or_default()
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
        self.tools
            .lock()
            .unwrap()
            .insert(ctx.member_name.clone(), ctx.memory_tools());
        Ok(ScriptedHarness::scripted(format!(
            "done: {}",
            ctx.member_name
        )))
    }

    fn name(&self) -> &'static str {
        "scripted"
    }
}

// ---------------------------------------------------------------------------
// Shared fixtures: one blueprint, one dependency set, one temp workspace root.
// ---------------------------------------------------------------------------

/// One scenario fixture: the scripted factory plus the temp root member workspaces live in.
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

/// Builds the scenario blueprint: team [`TEAM`], leader [`LEADER`], one teammate [`ALICE`].
fn blueprint() -> TeamAgentSpec {
    let mut spec = TeamAgentSpec {
        team_name: TEAM.to_owned(),
        team_desc: Some("a review team".to_owned()),
        ..TeamAgentSpec::default()
    };
    LEADER.clone_into(&mut spec.leader.member_name);
    spec.agents
        .insert("leader".to_owned(), deep_agent("Leader", "lead the team"));
    spec.agents
        .insert(ALICE.to_owned(), deep_agent(ALICE, "review the change"));
    spec
}

/// Returns `spec` with a `metadata["memory"]` block; `store` selects a file-backed store.
fn with_memory_block(mut spec: TeamAgentSpec, store: Option<&Path>) -> TeamAgentSpec {
    let mut block = json!({"export_on_stop": true});
    if let Some(dir) = store {
        block["store"] = json!({"kind": "file", "dir": dir.to_string_lossy()});
    }
    spec.metadata.insert(MEMORY_METADATA_KEY.to_owned(), block);
    spec
}

/// Builds `spec` over `fixture`'s injected dependencies.
async fn build(fixture: &Fixture, spec: &TeamAgentSpec) -> Arc<TeamAgent> {
    within("team build", spec.build_with(fixture.deps()))
        .await
        .expect("the team builds")
}

/// Every record of `wiring`'s team, as the shared facade holds them.
async fn records(wiring: &Arc<TeamMemoryWiring>) -> Vec<TeamMemoryRecord> {
    within(
        "team list",
        wiring
            .memory()
            .list(&TeamMemoryFilter::for_team(wiring.team_id())),
    )
    .await
    .expect("the team memory store lists")
}

/// Returns the single record of `kind` written for `member`.
fn only_of_kind<'a>(
    records: &'a [TeamMemoryRecord],
    kind: &TeamMemoryKind,
    member: &str,
) -> &'a TeamMemoryRecord {
    let matching: Vec<&TeamMemoryRecord> = records
        .iter()
        .filter(|record| &record.kind == kind && record.member_id.as_deref() == Some(member))
        .collect();
    assert_eq!(
        matching.len(),
        1,
        "expected exactly one {kind:?} record for {member}: {records:?}"
    );
    matching[0]
}

/// Returns how many records of `kind` the store holds.
fn count_of_kind(records: &[TeamMemoryRecord], kind: &TeamMemoryKind) -> usize {
    records.iter().filter(|record| &record.kind == kind).count()
}

/// Indexes records by id so two record sets compare order-insensitively.
fn by_id(records: Vec<TeamMemoryRecord>) -> BTreeMap<String, TeamMemoryRecord> {
    records
        .into_iter()
        .map(|record| (record.id.clone(), record))
        .collect()
}

/// Parses a toolkit success envelope's `result` into the record it carries.
fn record_of(envelope: &serde_json::Value) -> TeamMemoryRecord {
    serde_json::from_value(envelope["result"].clone()).expect("the stored record")
}

// ---------------------------------------------------------------------------
// Scenario 1: the runtime drives the lifecycle of a metadata-configured team.
// ---------------------------------------------------------------------------

/// Builds a `metadata["memory"]` team with a file store, then walks its whole lifecycle.
///
/// Nothing here calls a memory hook by hand: every record and the `TEAM_MEMORY.md` file come out
/// of `start`, `spawn_member`, `send_message`, `shutdown_member` and `stop`.
#[tokio::test]
#[allow(clippy::too_many_lines)]
async fn the_runtime_drives_the_memory_lifecycle_of_a_configured_team() {
    let fixture = Fixture::new();
    let store = TempDir::new().expect("a temp memory store");
    let spec = with_memory_block(blueprint(), Some(store.path()));
    let agent = build(&fixture, &spec).await;

    let wiring = agent.memory().expect("a metadata block opens a wiring");
    assert!(wiring.is_enabled());
    assert_eq!(wiring.team_id(), TEAM);
    assert!(
        wiring.trace_enabled(),
        "the default configuration attaches the built-in bridge"
    );
    assert_eq!(wiring.document_path(), store.path().join(TEAM_MEMORY_FILE));
    assert!(
        !wiring.document_path().exists(),
        "the build imports, it never writes"
    );
    assert!(records(&wiring).await.is_empty(), "the store starts empty");
    assert_eq!(
        fixture.factory.created_members(),
        vec![LEADER.to_owned()],
        "the build materializes the leader harness only"
    );

    within("team start", agent.start())
        .await
        .expect("the team starts");
    assert_eq!(
        wiring.hooks().session_id().as_deref(),
        Some(agent.session_id()),
        "start installs the coordination session"
    );

    within("member spawn", agent.spawn_member(ALICE))
        .await
        .expect("alice spawns");
    assert_eq!(wiring.hooks().members(), vec![ALICE.to_owned()]);
    let after_join = records(&wiring).await;
    assert_eq!(after_join.len(), 1, "the join wrote exactly one record");
    let profile = only_of_kind(&after_join, &TeamMemoryKind::MemberProfile, ALICE);
    assert_eq!(profile.team_id, TEAM);
    assert_eq!(profile.session_id.as_deref(), Some(agent.session_id()));
    assert_eq!(
        profile.content,
        format!("Member {ALICE} joined team {TEAM}.")
    );
    assert_eq!(
        profile.title.as_deref(),
        Some(format!("Member {ALICE} joined").as_str())
    );
    assert_eq!(profile.source.as_deref(), Some(HOOK_SOURCE));
    assert_eq!(profile.visibility, TeamVisibility::Team);
    assert_eq!(
        profile.tags,
        vec!["member".to_owned(), "presence".to_owned()]
    );

    within("member message", agent.send_message(Some(ALICE), DECISION))
        .await
        .expect("the message is delivered");
    assert_eq!(
        wiring.hooks().pending_turns(),
        1,
        "one delivered message buffers exactly one turn"
    );
    assert_eq!(
        records(&wiring).await.len(),
        1,
        "buffering writes nothing by itself"
    );

    within(
        "member shutdown",
        agent.shutdown_member(ALICE, MemberLifetime::Shutdown, false),
    )
    .await
    .expect("alice retires");
    assert!(
        wiring.hooks().members().is_empty(),
        "a retired member is no longer registered"
    );
    let after_leave = records(&wiring).await;
    assert_eq!(after_leave.len(), 2, "profile plus one handoff note");
    let handoff = only_of_kind(&after_leave, &TeamMemoryKind::HandoffNote, ALICE);
    assert_eq!(handoff.content, format!("Member {ALICE} left team {TEAM}."));
    assert_eq!(handoff.session_id.as_deref(), Some(agent.session_id()));
    assert_eq!(
        handoff.tags,
        vec!["handoff".to_owned(), "member".to_owned()]
    );
    assert_eq!(
        wiring.hooks().pending_turns(),
        1,
        "a leave does not flush the buffer"
    );

    within("team stop", agent.stop())
        .await
        .expect("the team stops");
    assert_eq!(
        wiring.hooks().pending_turns(),
        0,
        "stop flushed the buffered turn"
    );
    let stored = records(&wiring).await;
    // The handle that sent the message is the leader, so the extracted turn is attributed to it.
    let decision = only_of_kind(&stored, &TeamMemoryKind::Decision, LEADER);
    assert_eq!(decision.content, DECISION);
    assert_eq!(decision.session_id.as_deref(), Some(agent.session_id()));
    assert_eq!(decision.source.as_deref(), Some(RULE_SOURCE));
    assert_eq!(decision.tags, vec!["decision".to_owned()]);
    assert!(
        decision.id.starts_with("hook-"),
        "the flush writes a deterministic id: {}",
        decision.id
    );
    assert_eq!(
        count_of_kind(&stored, &TeamMemoryKind::MemberProfile),
        1,
        "the join is recorded once"
    );
    assert_eq!(
        count_of_kind(&stored, &TeamMemoryKind::HandoffNote),
        2,
        "alice retired and the leader stopped: {stored:?}"
    );

    let document = wiring.document_path();
    assert!(document.is_file(), "stop wrote {document:?}");
    let markdown = fs::read_to_string(document).expect("the document reads back");
    assert!(
        markdown.starts_with(&format!(
            "---\nteam_memory: {TEAM_MEMORY_FORMAT}\nteam_id: {TEAM}\n"
        )),
        "{markdown}"
    );
    assert!(markdown.contains(&format!("entries: {}", stored.len())));
    assert!(markdown.contains(&format!("# Team memory — {TEAM}")));
    for section in ["decision", "lesson", "member"] {
        assert!(
            markdown.contains(&format!("\n## {section}\n")),
            "the `{section}` section is missing: {markdown}"
        );
    }
    assert!(markdown.contains(&format!("### [decision] {DECISION}")));
    // An entry heading is the record's title when it has one, its content otherwise.
    assert!(markdown.contains(&format!("### [member] Member {ALICE} joined")));
    for member in [ALICE, LEADER] {
        assert!(
            markdown.contains(&format!("### [lesson] Member {member} left")),
            "the `{member}` handoff heading is missing: {markdown}"
        );
    }
    for record in &stored {
        assert!(
            markdown.contains(&format!("*id: {}", record.id)),
            "{} is missing from the document",
            record.id
        );
    }
    let parsed = TeamMemoryDocument::parse(&markdown, None).expect("the document parses");
    assert_eq!(parsed.team_id, TEAM);
    assert_eq!(
        by_id(parsed.entries),
        by_id(stored.clone()),
        "the exported document is the store"
    );
    assert_eq!(document_team_id(&markdown).as_deref(), Some(TEAM));
}

// ---------------------------------------------------------------------------
// Scenario 2: an injected wiring wins over the metadata block.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn an_injected_wiring_wins_over_a_conflicting_metadata_block() {
    let fixture = Fixture::new();
    let injected_store = TempDir::new().expect("a temp injected store");
    let ignored_store = TempDir::new().expect("a temp metadata store");
    let injected = TeamMemoryWiring::open(
        "injected_team",
        MemoryConfig {
            store: MemoryStoreConfig::File {
                dir: Some(injected_store.path().to_path_buf()),
            },
            ..MemoryConfig::default()
        },
        fixture.workspace.path(),
    )
    .await
    .expect("the injected wiring opens");

    let spec = with_memory_block(blueprint(), Some(ignored_store.path()));
    let deps = fixture.deps().with_memory(Arc::clone(&injected));
    let agent = within("team build", spec.build_with(deps))
        .await
        .expect("the team builds");

    let wiring = agent.memory().expect("the injected wiring");
    assert!(
        Arc::ptr_eq(&wiring, &injected),
        "the injected wiring is used verbatim"
    );
    assert_eq!(
        wiring.team_id(),
        "injected_team",
        "the wiring keeps the team id it was opened with"
    );
    assert_eq!(
        wiring.document_path(),
        injected_store.path().join(TEAM_MEMORY_FILE),
        "the wiring keeps the document root it was opened with"
    );

    within("team start", agent.start())
        .await
        .expect("the team starts");
    within("member spawn", agent.spawn_member(ALICE))
        .await
        .expect("alice spawns");
    within("member message", agent.send_message(None, DECISION))
        .await
        .expect("the broadcast is delivered");
    assert_eq!(
        wiring.hooks().pending_turns(),
        1,
        "a broadcast buffers one turn as well"
    );
    within("team stop", agent.stop())
        .await
        .expect("the team stops");

    assert!(
        injected_store.path().join(TEAM_MEMORY_FILE).is_file(),
        "the injected wiring exported its own document"
    );
    assert!(
        !ignored_store.path().join(TEAM_MEMORY_FILE).exists(),
        "the ignored metadata block never wrote a document"
    );
    assert_eq!(
        fs::read_dir(ignored_store.path())
            .expect("the ignored store reads")
            .count(),
        0,
        "the ignored store directory stays empty"
    );
    assert!(
        !records(&wiring).await.is_empty(),
        "the records live under the injected wiring's team id"
    );
    let under_runtime_name = within(
        "runtime-named list",
        wiring.memory().list(&TeamMemoryFilter::for_team(TEAM)),
    )
    .await
    .expect("the facade lists");
    assert!(
        under_runtime_name.is_empty(),
        "nothing was stored under the runtime's own team name: {under_runtime_name:?}"
    );
}

// ---------------------------------------------------------------------------
// Scenario 3: the member context carries the team toolkit (and a plain team carries none).
// ---------------------------------------------------------------------------

#[tokio::test]
#[allow(clippy::too_many_lines)]
async fn every_member_context_carries_the_team_toolkit() {
    let fixture = Fixture::new();
    let store = TempDir::new().expect("a temp memory store");
    let agent = build(
        &fixture,
        &with_memory_block(blueprint(), Some(store.path())),
    )
    .await;
    let wiring = agent.memory().expect("a wiring");

    within("team start", agent.start())
        .await
        .expect("the team starts");
    within("member spawn", agent.spawn_member(ALICE))
        .await
        .expect("alice spawns");
    assert_eq!(
        fixture.factory.created_members(),
        vec![LEADER.to_owned(), ALICE.to_owned()],
        "the leader is built at build time, the teammate on its spawn"
    );

    for member in [LEADER, ALICE] {
        let ctx = fixture.factory.context(member);
        let mounted = ctx
            .memory_toolkit()
            .unwrap_or_else(|| panic!("{member} mounts the team toolkit"));
        let expected = wiring.toolkit(member, None).expect("the team's toolkit");
        assert!(
            Arc::ptr_eq(mounted, &expected),
            "{member} mounts the team's own toolkit"
        );
        let tools = ctx.memory_tools();
        let names: Vec<&str> = tools.iter().map(|tool| tool.card().name.as_str()).collect();
        assert_eq!(
            names,
            vec![TOOL_REMEMBER, TOOL_SEARCH, TOOL_FORGET, TOOL_STATS]
        );
        assert_eq!(
            tools[0].card().id,
            format!("team_memory.{TEAM}.{member}.{TOOL_REMEMBER}"),
            "the mounted card names the calling member"
        );
    }

    let remember = fixture
        .factory
        .mounted_tools(ALICE)
        .into_iter()
        .find(|tool| tool.card().name == TOOL_REMEMBER)
        .expect("alice mounts the remember tool");
    let envelope = within(
        "tool remember",
        remember.invoke(
            json!({
                "content": "Use SQLite for the shared deployment store.",
                "title": "Storage choice",
                "kind": "decision",
                "tags": ["storage"],
            }),
            &ToolContext::new(),
        ),
    )
    .await
    .expect("the mounted tool writes");
    assert_eq!(envelope["ok"], json!(true), "{envelope}");
    let record = record_of(&envelope);
    assert_eq!(record.member_id.as_deref(), Some(ALICE));
    assert_eq!(record.session_id.as_deref(), Some(agent.session_id()));
    assert_eq!(record.kind, TeamMemoryKind::Decision);
    assert_eq!(record.tags, vec!["storage".to_owned()]);
    let stored = records(&wiring).await;
    assert!(
        stored
            .iter()
            .any(|held| held.id == record.id && held.content == record.content),
        "the team's own facade reads the tool's write back: {stored:?}"
    );

    // The same blueprint without a `memory` block allocates nothing at all.
    let plain_fixture = Fixture::new();
    let plain = build(&plain_fixture, &blueprint()).await;
    assert!(
        plain.memory().is_none(),
        "an unconfigured team runs without memory"
    );
    within("plain start", plain.start())
        .await
        .expect("the plain team starts");
    within("plain spawn", plain.spawn_member(ALICE))
        .await
        .expect("alice spawns there too");
    for member in [LEADER, ALICE] {
        let ctx = plain_fixture.factory.context(member);
        assert!(ctx.memory_toolkit().is_none(), "{member} mounts no toolkit");
        assert!(ctx.memory_tools().is_empty(), "{member} mounts no tools");
    }
}

// ---------------------------------------------------------------------------
// Scenario 4: the `TEAM_MEMORY.md` file round trip through one wiring.
// ---------------------------------------------------------------------------

#[tokio::test]
#[allow(clippy::too_many_lines)]
async fn the_wiring_exports_and_reimports_the_team_document_without_duplicates() {
    let dir = TempDir::new().expect("a temp memory store");
    let config = MemoryConfig {
        store: MemoryStoreConfig::File {
            dir: Some(dir.path().to_path_buf()),
        },
        ..MemoryConfig::default()
    };
    let wiring = TeamMemoryWiring::open(TEAM, config.clone(), dir.path())
        .await
        .expect("the wiring opens");

    within("session start", wiring.on_session_start(SESSION))
        .await
        .expect("the session starts");
    assert_eq!(wiring.hooks().session_id().as_deref(), Some(SESSION));
    let profile = within("alice joins", wiring.on_member_join(ALICE))
        .await
        .expect("the join is recorded")
        .expect("a profile record");
    within("turn a", wiring.on_turn(turn(ALICE, DECISION)))
        .await
        .expect("the decision is buffered");
    let first = within("flush a", wiring.flush())
        .await
        .expect("the decision is flushed");
    within("turn b", wiring.on_turn(turn(BOB, LESSON)))
        .await
        .expect("the lesson is buffered");
    let second = within("flush b", wiring.flush())
        .await
        .expect("the lesson is flushed");
    assert_eq!(first.turns, 1);
    assert_eq!(first.len(), 1);
    assert_eq!(second.turns, 1);
    assert_eq!(second.len(), 1);
    let mut expected = vec![profile];
    expected.extend(first.records.iter().cloned());
    expected.extend(second.records.iter().cloned());
    assert_eq!(expected.len(), 3, "profile + decision + lesson");

    let markdown = within("export", wiring.export())
        .await
        .expect("the export runs")
        .expect("a non-empty document");
    let path = wiring.document_path();
    assert_eq!(
        fs::read_to_string(path).expect("the document reads back"),
        markdown,
        "the file carries the export verbatim"
    );
    assert!(markdown.starts_with(&format!(
        "---\nteam_memory: {TEAM_MEMORY_FORMAT}\nteam_id: {TEAM}\n"
    )));
    assert!(markdown.contains("entries: 3"), "{markdown}");
    assert!(markdown.contains(&format!("# Team memory — {TEAM}")));
    for section in ["decision", "member", "lesson"] {
        assert!(
            markdown.contains(&format!("\n## {section}\n")),
            "the `{section}` section is missing: {markdown}"
        );
    }
    assert!(markdown.contains(&format!("### [decision] {DECISION}")));
    assert!(markdown.contains(&format!("### [member] Member {ALICE} joined")));
    assert!(markdown.contains(&format!("### [lesson] {LESSON}")));
    for record in &expected {
        assert!(
            markdown.contains(&format!("*id: {}", record.id)),
            "{} is missing from the document",
            record.id
        );
    }
    let decision = expected
        .iter()
        .find(|record| record.kind == TeamMemoryKind::Decision)
        .expect("the flushed decision");
    assert!(markdown.contains(&format!(
        "*id: {} · member: {ALICE} · importance: {:.2}",
        decision.id, decision.importance
    )));
    assert_eq!(document_team_id(&markdown).as_deref(), Some(TEAM));
    assert_eq!(
        truncate_for_prompt(&markdown),
        markdown.trim(),
        "a short document is the whole prompt view"
    );
    let parsed = TeamMemoryDocument::parse(&markdown, None).expect("the document parses");
    assert_eq!(parsed.team_id, TEAM);
    assert_eq!(parsed.len(), 3);
    assert_eq!(
        by_id(parsed.entries),
        by_id(expected.clone()),
        "the render/parse round trip is lossless"
    );

    // A fresh wiring over the same directory imports the document while it opens.
    let reopened = TeamMemoryWiring::open(TEAM, config, dir.path())
        .await
        .expect("the wiring reopens");
    assert_eq!(
        by_id(records(&reopened).await),
        by_id(expected.clone()),
        "the open-time import restored every record"
    );
    let again = within("re-import", reopened.import_from(&markdown))
        .await
        .expect("the same document imports again");
    assert_eq!(again, 3);
    let after_second = records(&reopened).await;
    assert_eq!(
        after_second.len(),
        3,
        "an import upserts instead of duplicating: {after_second:?}"
    );
    assert_eq!(by_id(after_second), by_id(expected));
}

// ---------------------------------------------------------------------------
// Scenario 5: a read-only member toolkit refuses writes.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_read_only_member_toolkit_refuses_writes_but_keeps_searching() {
    let dir = TempDir::new().expect("a temp memory store");
    let wiring = TeamMemoryWiring::open(TEAM, MemoryConfig::default(), dir.path())
        .await
        .expect("the wiring opens");
    let alice = wiring.toolkit(ALICE, Some(SESSION)).expect("a toolkit");
    assert!(
        Arc::ptr_eq(&alice, &wiring.toolkit(ALICE, Some(SESSION)).unwrap()),
        "the same member always gets the same cached toolkit"
    );
    let shared = within(
        "alice remember",
        alice.handle_tool_call(
            TOOL_REMEMBER,
            json!({
                "content": "Use SQLite for the shared deployment store.",
                "kind": "decision",
                "tags": ["storage"],
            }),
        ),
    )
    .await
    .expect("the toolkit writes");
    assert_eq!(shared["ok"], json!(true));
    let shared_record = record_of(&shared);
    assert_eq!(shared_record.visibility, TeamVisibility::Team);
    assert_eq!(shared_record.session_id.as_deref(), Some(SESSION));

    let reader = MemberMemoryToolkit::new(
        Arc::clone(wiring.memory()),
        MemberMemoryContext::new(TEAM)
            .with_member(BOB)
            .with_session(SESSION)
            .read_only(),
    );
    assert!(reader.context().read_only);
    for (tool, args) in [
        (
            TOOL_REMEMBER,
            json!({"content": "A read-only member may not write this."}),
        ),
        (TOOL_FORGET, json!({"id": shared_record.id})),
    ] {
        let error = within("read-only refusal", reader.handle_tool_call(tool, args))
            .await
            .expect_err("a read-only context refuses writes");
        assert!(matches!(error, Error::Unsupported(_)), "{tool}: {error}");
        let envelope = MemberMemoryToolkit::error_envelope(tool, &error);
        assert_eq!(envelope["ok"], json!(false));
        assert_eq!(envelope["tool"], json!(tool));
        assert_eq!(envelope["error"]["kind"], json!("unsupported"));
        assert!(
            envelope["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("read-only")),
            "{envelope}"
        );
    }
    let still_reads = within(
        "read-only search",
        reader.handle_tool_call(
            TOOL_SEARCH,
            json!({"query": "sqlite shared deployment store"}),
        ),
    )
    .await
    .expect("a read-only context still searches");
    assert_eq!(still_reads["result"]["count"], json!(1));
    assert_eq!(
        records(&wiring).await.len(),
        1,
        "the refusals wrote nothing beyond the one record"
    );
}

// ---------------------------------------------------------------------------
// Scenario 6: a board task's completion is still a caller-composed member turn.
// ---------------------------------------------------------------------------

/// Drives a `TaskBoard` task to `completed`, then flushes the member turn the caller derives.
///
/// Known gap: the runtime wires memory into the *session, membership and mailbox* lifecycle, but
/// there is still no task-board → team-memory bridge — `TeamMemoryHooks` exposes no way to buffer
/// a task note (`HookState::task_notes` is only ever written empty), and the board never reaches
/// the wiring. So a host that wants a completed task remembered composes the two surfaces itself:
/// the task's own status event and summary become the [`TeamTurn`] the hooks extract. Nothing here
/// may be "fixed" inside the sibling crate; the scenario pins the caller-side composition.
#[tokio::test]
#[allow(clippy::too_many_lines)]
async fn a_completed_board_task_becomes_a_flushed_member_memory() {
    let store = Arc::new(InMemoryTeamStore::new());
    let board = TaskBoard::with_defaults(store, TEAM);

    let created = within(
        "task create",
        board.create_task(TaskGraphSpec {
            title: "ship the release".to_owned(),
            content: "cut the tag and publish the shared store".to_owned(),
            ..TaskGraphSpec::default()
        }),
    )
    .await
    .expect("the board creates the task");
    let task_id = created.task.task_id.clone();
    assert_eq!(
        created
            .events
            .iter()
            .map(openjiuwen::agent_teams::schema::EventPayload::event_type)
            .collect::<Vec<_>>(),
        ["task_created", "task_status_changed"]
    );

    let claimed = within("task claim", board.claim(&task_id, BOB))
        .await
        .expect("the claim is applied");
    assert!(claimed.ok, "{claimed:?}");
    let completed = within(
        "task complete",
        board.complete(&task_id, BOB, "shipped the shared store"),
    )
    .await
    .expect("the completion is applied");
    assert_eq!(completed.task.status, TaskStatus::Completed);
    assert_eq!(completed.task.assignee.as_deref(), Some(BOB));
    assert!(within("board settled", board.is_settled()).await.unwrap());

    let status_event = completed
        .events
        .iter()
        .find(|event| event.event_type() == "task_status_changed")
        .expect("the completion publishes the status transition");
    let status_payload = status_event
        .payload_value()
        .expect("the status payload serializes");
    assert_eq!(status_payload["status"], json!("completed"));
    assert_eq!(status_payload["assignee"], json!(BOB));
    assert_eq!(status_payload["task_id"], json!(task_id));

    // Everything below is the caller's composition, not the runtime's.
    let memory = Arc::new(TeamMemory::in_memory());
    let hooks = TeamMemoryHooks::with_rule_extractor(
        TEAM,
        Arc::clone(&memory),
        TeamMemoryHookConfig::default(),
    );
    let call = CallOptions::new();
    within("session start", hooks.on_session_start(SESSION, &call))
        .await
        .expect("the session starts");
    within("member joins", hooks.on_member_join(BOB, &call))
        .await
        .expect("the join is recorded")
        .expect("a profile record");
    let summary = completed
        .task
        .summary
        .clone()
        .expect("the completion summary");
    let turn_content = format!("{ACTION_ITEM} — {summary}");
    within("board turn", hooks.on_turn(turn(BOB, &turn_content), &call))
        .await
        .expect("the board's completion becomes a member turn");
    let flush = within("flush", hooks.on_session_end(&call))
        .await
        .expect("the board turn is extracted");
    assert_eq!(flush.turns, 1);
    assert_eq!(flush.len(), 1);
    assert!(flush.degraded.is_none());
    assert_eq!(hooks.pending_turns(), 0);

    let record = &flush.records[0];
    assert_eq!(record.team_id, TEAM);
    assert_eq!(record.member_id.as_deref(), Some(BOB));
    assert_eq!(record.session_id.as_deref(), Some(SESSION));
    assert_eq!(record.kind, TeamMemoryKind::Task);
    assert_eq!(record.content, turn_content);
    assert!(record.content.contains("ship the release"));
    assert!(record.content.contains("shipped the shared store"));
    assert_eq!(record.source.as_deref(), Some(RULE_SOURCE));
    assert_eq!(
        record.tags,
        vec!["task".to_owned(), "context".to_owned()],
        "a rule match earns its marker category plus the kind's `TEAM_MEMORY.md` tag"
    );
    let stored = within("task list", memory.list(&TeamMemoryFilter::for_team(TEAM)))
        .await
        .expect("the team lists");
    assert_eq!(stored.len(), 2, "profile plus the task");

    let hits = within(
        "task recall",
        memory.search(
            &TeamMemoryScope::new(TEAM),
            "shipped shared store",
            &TeamSearchOptions::default(),
        ),
    )
    .await
    .expect("the task memory is recallable");
    assert_eq!(hits.len(), 1, "{hits:?}");
    assert_eq!(hits[0].record.id, record.id);
    assert_eq!(hits[0].record.content, turn_content);
}
