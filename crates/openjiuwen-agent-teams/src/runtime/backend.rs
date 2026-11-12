//! [`TeamBackend`]: the organisation facade of one running team.
//!
//! A running team is more than its leader: it owns a roster of live member handles, a persisted
//! mailbox, an event bus and a task board. [`TeamBackend`] is the single object that ties those
//! four together, so the leader loop, the tool wrappers, the CLI and the MCP server all reach team
//! state through one seam instead of each rebuilding it from the store.
//!
//! It owns no policy: rows live in [`crate::storage`], task decisions in [`crate::taskboard`],
//! member status in [`crate::member`] and transports in [`crate::messager`]. What this module does
//! own is the *routing* of one message (broadcast vs. direct vs. multicast), the envelope each
//! event is published with, and the canonical ordering of every roster listing.
//!
//! Python parity sources: `openjiuwen/agent_teams/tools/team.py` (`TeamBackend` — the roster,
//! task-board and snapshot facade) and `openjiuwen/agent_teams/tools/message_manager.py`
//! (`send_message` / `broadcast_message` / `mark_messages_read`).
//! This port keeps one facade for both, because the runtime, the tool wrappers, the CLI and the MCP
//! server all need the same four seams in hand and splitting them would only duplicate the wiring.
//!
//! The facade is shared, not copied: it lives behind an [`Arc`], every dependency it holds is
//! already shared, and the roster is one [`RwLock`] over name → handle. That lock is never held
//! across an `.await`.
//!
//! ```text
//! send_message(from, None, body)       -> one broadcast row + one `broadcast` event
//! send_message(from, Some("*"), body)  -> one broadcast row + one `broadcast` event
//! send_message(from, Some("a,b"), body)-> one direct row  + one `message` event per target
//! ```

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use parking_lot::RwLock;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::id::uuid_hex;

use crate::constants::{BROADCAST_TARGET, DEFAULT_LEADER_MEMBER_NAME, MAX_MESSAGE_CONTENT_CHARS};
use crate::member::{MemberActivityRegistry, MemberSpawner, TeamMember};
use crate::messager::Messager;
use crate::schema::{
    BroadcastEvent, EventPayload, HarnessState, MessageEvent, TaskFilter, TaskSummary,
    TeamSnapshot, TeamSpec, TeamTopic,
};
use crate::storage::{MessageRecord, TeamInfoRecord, TeamStore};
use crate::taskboard::{TaskBoard, TaskMutation};
use crate::workspace::WorkspaceProvider;

/// The organisation facade of one running team.
///
/// Holds the runtime mirror of the team blueprint ([`TeamSpec`]) and the shared seams a member
/// needs: the store, the task board, the event bus, the member spawner, the workspace provider and
/// the activity projection. Roster handles are kept here because they are *live* state — a row in
/// the store says a member exists, an entry here says the team can reach its brain.
///
/// Every member listing is returned in `member_name` order, so callers (and tests) see a stable
/// roster regardless of insertion order.
pub struct TeamBackend {
    /// Runtime mirror of the team blueprint.
    spec: TeamSpec,
    /// Persisted state: team row, roster, task rows, votes and mailbox.
    store: Arc<dyn TeamStore>,
    /// Task-board policy over [`Self::store`].
    taskboard: Arc<TaskBoard>,
    /// The team's event bus; shared with every member.
    messager: Arc<dyn Messager>,
    /// Builds and tears down member runtimes.
    spawner: Arc<dyn MemberSpawner>,
    /// Allocates each member's workspace directory.
    workspace: Arc<dyn WorkspaceProvider>,
    /// Leader-side projection of who is working.
    activity: Arc<MemberActivityRegistry>,
    /// Live roster: `member_name` → handle.
    members: RwLock<BTreeMap<String, Arc<TeamMember>>>,
}

impl TeamBackend {
    /// Builds the facade over already-shared seams.
    ///
    /// The caller keeps ownership of every dependency and is expected to have created exactly one
    /// store and one transport per team; the backend never opens a second instance of either. The
    /// roster starts empty — the builder registers the leader and the teammates afterwards with
    /// [`TeamBackend::add_member`].
    ///
    /// `team_spec` is the runtime mirror ([`TeamSpec`]), not the blueprint: the running team reads
    /// its identity, dispatch mode and language from here.
    #[must_use]
    pub fn new(
        team_spec: TeamSpec,
        store: Arc<dyn TeamStore>,
        taskboard: Arc<TaskBoard>,
        messager: Arc<dyn Messager>,
        spawner: Arc<dyn MemberSpawner>,
        workspace: Arc<dyn WorkspaceProvider>,
        activity: Arc<MemberActivityRegistry>,
    ) -> Self {
        Self {
            spec: team_spec,
            store,
            taskboard,
            messager,
            spawner,
            workspace,
            activity,
            members: RwLock::new(BTreeMap::new()),
        }
    }

    /// Returns the team name every row and event of this backend is scoped to.
    #[must_use]
    pub fn team_name(&self) -> &str {
        &self.spec.team_name
    }

    /// Returns the runtime mirror of the team blueprint.
    #[must_use]
    pub fn spec(&self) -> &TeamSpec {
        &self.spec
    }

    /// Returns the shared store.
    #[must_use]
    pub fn store(&self) -> Arc<dyn TeamStore> {
        Arc::clone(&self.store)
    }

    /// Returns the task board.
    #[must_use]
    pub fn tasks(&self) -> &Arc<TaskBoard> {
        &self.taskboard
    }

    /// Returns the team's event bus.
    #[must_use]
    pub fn messager(&self) -> &Arc<dyn Messager> {
        &self.messager
    }

    /// Returns the member spawner.
    #[must_use]
    pub fn spawner(&self) -> &Arc<dyn MemberSpawner> {
        &self.spawner
    }

    /// Returns the workspace provider.
    #[must_use]
    pub fn workspace(&self) -> &Arc<dyn WorkspaceProvider> {
        &self.workspace
    }

    /// Returns the activity projection.
    #[must_use]
    pub fn activity(&self) -> &Arc<MemberActivityRegistry> {
        &self.activity
    }

    /// Returns the leader's member name.
    ///
    /// This is [`TeamSpec::leader_member_name`] when it is set and non-empty, else the crate
    /// default [`DEFAULT_LEADER_MEMBER_NAME`] (`team_leader`), matching the Python blueprint.
    #[must_use]
    pub fn leader_member_name(&self) -> &str {
        match self.spec.leader_member_name.as_deref() {
            Some(name) if !name.is_empty() => name,
            _ => DEFAULT_LEADER_MEMBER_NAME,
        }
    }

    /// Returns every live member handle, sorted by `member_name`.
    #[must_use]
    pub fn members(&self) -> Vec<Arc<TeamMember>> {
        self.members.read().values().cloned().collect()
    }

    /// Returns the live handle of `name`, or `None` when the team has no such member.
    #[must_use]
    pub fn member(&self, name: &str) -> Option<Arc<TeamMember>> {
        self.members.read().get(name).cloned()
    }

    /// Returns every live member name, sorted.
    #[must_use]
    pub fn member_names(&self) -> Vec<String> {
        self.members.read().keys().cloned().collect()
    }

    /// Returns the number of live members.
    #[must_use]
    pub fn len(&self) -> usize {
        self.members.read().len()
    }

    /// Returns `true` when the team has no live member.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.members.read().is_empty()
    }

    /// Registers `member` in the roster and in the activity projection.
    ///
    /// A member with the same name is replaced, which is how a restarted member is re-homed; the
    /// activity registry then keeps the member's existing idle clock instead of restarting it.
    pub fn add_member(&self, member: Arc<TeamMember>) {
        let name = member.member_name().to_owned();
        self.members.write().insert(name, Arc::clone(&member));
        self.activity.register(member);
    }

    /// Removes the live handle of `name` from the roster and the activity projection.
    ///
    /// Returns the handle that was registered, or `None` when the member was not live. The roster
    /// row in the store is left untouched: retirement, not removal, is what settles a member
    /// lifecycle (see [`crate::member::TeamMember::shutdown`]).
    pub fn remove_member(&self, name: &str) -> Option<Arc<TeamMember>> {
        self.activity.remove(name);
        self.members.write().remove(name)
    }

    /// Persists `info` as the team row.
    ///
    /// The store keeps the creation stamp of an existing row and stamps `updated_at`, so this is a
    /// safe upsert on every run.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `info.team_name` is empty, and whatever the store fails
    /// with.
    pub async fn upsert_team_info(&self, info: TeamInfoRecord) -> Result<()> {
        self.store.upsert_team(info).await
    }

    /// Sends `content` from `from` to one broadcast or direct route and returns the first
    /// persisted row.
    ///
    /// Routing:
    ///
    /// * `to == None`, `Some("")` or `Some("*")` — one broadcast row for the whole team.
    /// * `to == Some("a,b")` — one direct row per comma-separated target, trimmed, de-duplicated
    ///   and kept in the order written.
    ///
    /// Every persisted row is published on [`TeamTopic::Message`] as a `broadcast` or `message`
    /// event stamped with `from` as sender, so bus subscribers see the same mail the mailbox holds
    /// and can attribute it to the member that wrote it. The caller — not the store — stamps the
    /// timestamp, and these rows carry `0`: ordering inside a run comes from the bus and from the
    /// mailbox sort, not from a clock read here.
    ///
    /// # Errors
    ///
    /// * [`Error::Validation`] when `content` is longer than
    ///   [`MAX_MESSAGE_CONTENT_CHARS`](crate::constants::MAX_MESSAGE_CONTENT_CHARS) characters, or
    ///   when a direct target list is empty after trimming and filtering.
    /// * Whatever the store or the transport fails with; nothing after the failing row is written
    ///   or published.
    pub async fn send_message(
        &self,
        from: &str,
        to: Option<&str>,
        content: &str,
    ) -> Result<MessageRecord> {
        if content.chars().count() > MAX_MESSAGE_CONTENT_CHARS {
            return Err(Error::validation(format!(
                "message content exceeds {MAX_MESSAGE_CONTENT_CHARS} characters"
            )));
        }
        match to {
            Some(targets) if !is_broadcast_target(targets) => {
                self.multicast_row(from, targets, content).await
            }
            _ => self.broadcast_row(from, content).await,
        }
    }

    /// Returns the unread mailbox rows of `name`.
    ///
    /// # Errors
    ///
    /// Returns whatever the store fails with.
    pub async fn unread(&self, name: &str) -> Result<Vec<MessageRecord>> {
        self.store.unread_messages(self.team_name(), name).await
    }

    /// Flags the direct messages `ids` as read for `name`.
    ///
    /// Broadcast read state is a per-member watermark, not a row flag, so broadcast ids are
    /// skipped silently; advance it through the store's `set_read_watermark`.
    ///
    /// # Errors
    ///
    /// Returns whatever the store fails with.
    pub async fn mark_read(&self, name: &str, ids: &[String]) -> Result<()> {
        self.store.mark_read(self.team_name(), name, ids).await
    }

    /// Publishes `payload` on `topic`, stamped with the team name as sender, and returns the
    /// event type that was published.
    ///
    /// The team name is the sender for organisation-level events (task board changes, team
    /// completion) because they are emitted by the team, not by one member.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the payload cannot be serialized, and whatever the
    /// transport fails with.
    pub async fn post_event(&self, topic: TeamTopic, payload: EventPayload) -> Result<String> {
        self.publish_event(topic, &payload, self.team_name())
            .await?;
        Ok(payload.event_type().to_owned())
    }

    /// Publishes every event of `mutation` on [`TeamTopic::Task`], in mutation order.
    ///
    /// This is the write side of the task board: the board returns the events a mutation implies
    /// and the caller publishes them, so the board never depends on a transport.
    ///
    /// # Errors
    ///
    /// Stops at the first event that cannot be published and returns that error; earlier events
    /// stay published.
    pub async fn publish_mutation(&self, mutation: &TaskMutation) -> Result<()> {
        for event in &mutation.events {
            self.post_event(TeamTopic::Task, event.clone()).await?;
        }
        Ok(())
    }

    /// Returns the task board projection of `filter`.
    ///
    /// # Errors
    ///
    /// Returns whatever the store fails with.
    pub async fn list_tasks(&self, filter: &TaskFilter) -> Result<Vec<TaskSummary>> {
        self.taskboard.view(filter).await
    }

    /// Returns the roster projection of the team, with no task rows.
    ///
    /// The roster comes from the activity registry — the leader-side projection — so the answer
    /// reflects the status the team last observed, not a fresh store read. The state is
    /// [`HarnessState::Idle`]: the runtime decides a live state, the backend does not track one.
    #[must_use]
    pub fn snapshot(&self) -> TeamSnapshot {
        let mut snapshot = TeamSnapshot::new(self.team_name(), HarnessState::Idle);
        snapshot.members = self.activity.snapshot();
        snapshot.sort();
        snapshot
    }

    /// Returns the roster projection of the team together with its task board.
    ///
    /// # Errors
    ///
    /// Returns whatever the store fails with while reading the task board.
    pub async fn snapshot_with_tasks(&self) -> Result<TeamSnapshot> {
        let mut snapshot = self.snapshot();
        snapshot.tasks = self.taskboard.view(&TaskFilter::default()).await?;
        snapshot.sort();
        Ok(snapshot)
    }

    /// Publishes `payload` on `topic` with `sender` as the envelope sender.
    async fn publish_event(
        &self,
        topic: TeamTopic,
        payload: &EventPayload,
        sender: &str,
    ) -> Result<()> {
        let message = payload.to_message(sender)?;
        self.messager.publish(topic, message).await
    }

    /// Persists and publishes one broadcast row sent by `from`.
    async fn broadcast_row(&self, from: &str, content: &str) -> Result<MessageRecord> {
        let message_id = mint_message_id();
        let record =
            MessageRecord::broadcast(self.team_name(), message_id.clone(), from, content, 0);
        let event = EventPayload::Broadcast(BroadcastEvent {
            message_id,
            from_member_name: from.to_owned(),
        });
        self.store.insert_message(record.clone()).await?;
        self.publish_event(TeamTopic::Message, &event, from).await?;
        Ok(record)
    }

    /// Persists and publishes one direct row sent by `from` and addressed to `to`.
    async fn direct_row(&self, from: &str, to: &str, content: &str) -> Result<MessageRecord> {
        let message_id = mint_message_id();
        let record =
            MessageRecord::direct(self.team_name(), message_id.clone(), from, to, content, 0);
        let event = EventPayload::Message(MessageEvent {
            message_id,
            from_member_name: from.to_owned(),
            to_member_name: Some(to.to_owned()),
        });
        self.store.insert_message(record.clone()).await?;
        self.publish_event(TeamTopic::Message, &event, from).await?;
        Ok(record)
    }

    /// Persists and publishes one direct row per comma-separated target.
    async fn multicast_row(
        &self,
        from: &str,
        targets: &str,
        content: &str,
    ) -> Result<MessageRecord> {
        let mut pending = split_targets(targets);
        let Some(first_target) = pending.next() else {
            return Err(Error::validation("message target must not be empty"));
        };
        let first_row = self.direct_row(from, &first_target, content).await?;
        for target in pending {
            self.direct_row(from, &target, content).await?;
        }
        Ok(first_row)
    }
}

impl fmt::Debug for TeamBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TeamBackend")
            .field("team_name", &self.team_name())
            .field("dispatch_mode", &self.spec.dispatch_mode)
            .field("members", &self.member_names())
            .finish_non_exhaustive()
    }
}

/// Returns `true` when `targets` addresses the whole team rather than specific members.
///
/// An empty target and the `*` wildcard both mean "everyone", mirroring
/// [`BROADCAST_TARGET`](crate::constants::BROADCAST_TARGET); whitespace-only input is treated the
/// same way so `Some(" ")` cannot fail as an empty target list.
fn is_broadcast_target(targets: &str) -> bool {
    let trimmed = targets.trim();
    trimmed.is_empty() || trimmed == BROADCAST_TARGET
}

/// Mints a mailbox id in the Python `msg_<uuid hex>` form.
fn mint_message_id() -> String {
    format!("msg_{}", uuid_hex())
}

/// Splits `targets` into trimmed, non-empty, de-duplicated member names, order preserved.
fn split_targets(targets: &str) -> std::vec::IntoIter<String> {
    let mut names: Vec<String> = Vec::new();
    for segment in targets.split(',') {
        let name = segment.trim();
        if name.is_empty() || names.iter().any(|seen| seen == name) {
            continue;
        }
        names.push(name.to_owned());
    }
    names.into_iter()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use async_trait::async_trait;
    use parking_lot::Mutex;

    use crate::member::{MemberRuntime, MemberSpawnRequest};
    use crate::messager::{EventSink, InProcessMessager};
    use crate::schema::{
        EventMessage, MessageEvent, TaskCreatedEvent, TaskGraphSpec, TaskStatus, TeamRole,
    };
    use crate::storage::InMemoryTeamStore;
    use crate::workspace::{MemberWorkspace, WorkspaceRequest};

    use super::*;

    /// A spawner that never builds a runtime: the backend stores the seam but this suite never
    /// spawns.
    struct NoopSpawner;

    #[async_trait]
    impl MemberSpawner for NoopSpawner {
        async fn spawn(&self, _request: MemberSpawnRequest) -> Result<Arc<dyn MemberRuntime>> {
            Err(Error::unsupported("test spawner does not spawn members"))
        }

        async fn despawn(&self, _member_name: &str) -> Result<()> {
            Ok(())
        }
    }

    /// A workspace provider that never touches the filesystem.
    struct NoopWorkspace;

    #[async_trait]
    impl WorkspaceProvider for NoopWorkspace {
        async fn allocate(&self, _request: &WorkspaceRequest) -> Result<MemberWorkspace> {
            Err(Error::unsupported(
                "test workspace provider does not allocate",
            ))
        }

        async fn release(&self, _workspace: &MemberWorkspace) -> Result<()> {
            Ok(())
        }
    }

    /// One backend plus the seams a test asserts against.
    struct Fixture {
        backend: TeamBackend,
        store: Arc<InMemoryTeamStore>,
        events: Arc<Mutex<Vec<EventMessage>>>,
    }

    /// Builds a team `alpha` whose bus mirrors every `team`/`task`/`message` event into `events`.
    fn fixture() -> Fixture {
        fixture_for("alpha", None)
    }

    /// Builds team `team` with an optional leader member name.
    fn fixture_for(team: &str, leader: Option<&str>) -> Fixture {
        let store: Arc<InMemoryTeamStore> = Arc::new(InMemoryTeamStore::new());
        let messager: Arc<InProcessMessager> = Arc::new(InProcessMessager::new());
        let events: Arc<Mutex<Vec<EventMessage>>> = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&events);
        let sink: EventSink = Arc::new(move |event: EventMessage| recorded.lock().push(event));
        messager
            .subscribe(
                "observer",
                &[TeamTopic::Team, TeamTopic::Task, TeamTopic::Message],
                sink,
            )
            .unwrap();

        let spec = TeamSpec {
            team_name: team.to_owned(),
            leader_member_name: leader.map(str::to_owned),
            ..TeamSpec::default()
        };
        let taskboard = Arc::new(TaskBoard::with_defaults(
            Arc::clone(&store) as Arc<dyn TeamStore>,
            team,
        ));
        let backend = TeamBackend::new(
            spec,
            Arc::clone(&store) as Arc<dyn TeamStore>,
            taskboard,
            Arc::clone(&messager) as Arc<dyn Messager>,
            Arc::new(NoopSpawner),
            Arc::new(NoopWorkspace),
            Arc::new(MemberActivityRegistry::new()),
        );
        Fixture {
            backend,
            store,
            events,
        }
    }

    /// Builds a live member handle for the fixture's team.
    fn member(fixture: &Fixture, name: &str) -> Arc<TeamMember> {
        Arc::new(TeamMember::new(
            Arc::clone(&fixture.store) as Arc<dyn TeamStore>,
            crate::storage::MemberRecord::new(
                fixture.backend.team_name(),
                name,
                TeamRole::Teammate,
            ),
        ))
    }

    /// Drains the recorded bus events.
    fn take_events(events: &Arc<Mutex<Vec<EventMessage>>>) -> Vec<EventMessage> {
        std::mem::take(&mut *events.lock())
    }

    #[test]
    fn debug_names_the_team_its_members_and_dispatch_mode() {
        let fixture = fixture();
        fixture.backend.add_member(member(&fixture, "dev-1"));
        let rendered = format!("{:?}", fixture.backend);
        assert!(rendered.contains("alpha"), "{rendered}");
        assert!(rendered.contains("dev-1"), "{rendered}");
        assert!(rendered.contains("Autonomous"), "{rendered}");
    }

    #[tokio::test]
    async fn direct_message_persists_one_row_and_publishes_one_event() {
        let fixture = fixture();
        let row = fixture
            .backend
            .send_message("team_leader", Some("dev-1"), "hello")
            .await
            .unwrap();

        assert_eq!(row.team_name, "alpha");
        assert_eq!(row.from_member_name, "team_leader");
        assert_eq!(row.to_member_name.as_deref(), Some("dev-1"));
        assert!(row.message_id.starts_with("msg_"));
        assert!(!row.is_broadcast());

        let unread = fixture
            .store
            .unread_messages("alpha", "dev-1")
            .await
            .unwrap();
        assert_eq!(unread.len(), 1);
        assert_eq!(unread[0].content, "hello");
        assert!(
            fixture
                .store
                .unread_messages("alpha", "dev-2")
                .await
                .unwrap()
                .is_empty(),
            "dev-2 receives no direct mail"
        );

        let events = take_events(&fixture.events);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "message");
        assert_eq!(events[0].sender_id, "team_leader");
        let payload: MessageEvent = serde_json::from_value(events[0].payload.clone()).unwrap();
        assert_eq!(payload.message_id, row.message_id);
        assert_eq!(payload.from_member_name, "team_leader");
        assert_eq!(payload.to_member_name.as_deref(), Some("dev-1"));
    }

    #[tokio::test]
    async fn unread_and_mark_read_delegate_to_the_store() {
        let fixture = fixture();
        fixture
            .backend
            .send_message("team_leader", Some("dev-1"), "hello")
            .await
            .unwrap();

        let unread = fixture.backend.unread("dev-1").await.unwrap();
        assert_eq!(unread.len(), 1);
        let ids = vec![unread[0].message_id.clone()];
        fixture.backend.mark_read("dev-1", &ids).await.unwrap();
        assert!(
            fixture.backend.unread("dev-1").await.unwrap().is_empty(),
            "a read message stays out of the mailbox"
        );
    }

    #[tokio::test]
    async fn broadcast_routes_persist_one_row_and_publish_one_event() {
        for target in [None, Some(""), Some("*")] {
            let fixture = fixture();
            let row = fixture
                .backend
                .send_message("team_leader", target, "all hands")
                .await
                .unwrap();
            assert!(row.is_broadcast(), "target {target:?}");
            assert_eq!(row.to_member_name, None);
            assert!(row.message_id.starts_with("msg_"));

            let unread = fixture
                .store
                .unread_messages("alpha", "dev-1")
                .await
                .unwrap();
            assert_eq!(unread.len(), 1, "target {target:?}");
            assert!(unread[0].is_broadcast());

            let events = take_events(&fixture.events);
            assert_eq!(events.len(), 1, "target {target:?}");
            assert_eq!(events[0].event_type, "broadcast");
            assert_eq!(events[0].sender_id, "team_leader");
            let payload: BroadcastEvent =
                serde_json::from_value(events[0].payload.clone()).unwrap();
            assert_eq!(payload.message_id, row.message_id);
            assert_eq!(payload.from_member_name, "team_leader");
        }
    }

    #[tokio::test]
    async fn multicast_persists_one_row_per_distinct_target() {
        let fixture = fixture();
        let row = fixture
            .backend
            .send_message("team_leader", Some("dev-1, dev-2"), "pair up")
            .await
            .unwrap();
        assert_eq!(row.to_member_name.as_deref(), Some("dev-1"));
        for name in ["dev-1", "dev-2"] {
            let unread = fixture.store.unread_messages("alpha", name).await.unwrap();
            assert_eq!(unread.len(), 1, "{name}");
        }
        let events = take_events(&fixture.events);
        assert_eq!(events.len(), 2);
        assert!(events.iter().all(|event| event.event_type == "message"));

        let duplicate = fixture
            .backend
            .send_message("team_leader", Some("dev-1,dev-1"), "once")
            .await
            .unwrap();
        assert_eq!(duplicate.to_member_name.as_deref(), Some("dev-1"));
        let events = take_events(&fixture.events);
        assert_eq!(events.len(), 1);
        let unread = fixture
            .store
            .unread_messages("alpha", "dev-1")
            .await
            .unwrap();
        assert_eq!(unread.len(), 2);
    }

    #[tokio::test]
    async fn empty_target_list_is_rejected() {
        let fixture = fixture();
        let err = fixture
            .backend
            .send_message("team_leader", Some(" , "), "hi")
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("message target must not be empty"),
            "{err}"
        );
        assert!(
            take_events(&fixture.events).is_empty(),
            "a rejected message is never published"
        );
        assert!(
            fixture
                .store
                .unread_messages("alpha", "dev-1")
                .await
                .unwrap()
                .is_empty(),
            "a rejected message is never persisted"
        );
    }

    #[tokio::test]
    async fn oversized_content_is_rejected_at_the_boundary() {
        let fixture = fixture();
        let accepted = "x".repeat(MAX_MESSAGE_CONTENT_CHARS);
        fixture
            .backend
            .send_message("team_leader", None, &accepted)
            .await
            .unwrap();

        let rejected = "x".repeat(MAX_MESSAGE_CONTENT_CHARS + 1);
        let err = fixture
            .backend
            .send_message("team_leader", None, &rejected)
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains(&format!(
                "message content exceeds {MAX_MESSAGE_CONTENT_CHARS} characters"
            )),
            "{err}"
        );
        assert_eq!(take_events(&fixture.events).len(), 1);
    }

    #[test]
    fn members_are_listed_in_name_order() {
        let fixture = fixture();
        assert!(fixture.backend.is_empty());
        assert_eq!(fixture.backend.len(), 0);

        fixture.backend.add_member(member(&fixture, "dev-2"));
        fixture.backend.add_member(member(&fixture, "dev-1"));
        fixture.backend.add_member(member(&fixture, "team_leader"));

        assert_eq!(fixture.backend.len(), 3);
        assert!(!fixture.backend.is_empty());
        assert_eq!(
            fixture.backend.member_names(),
            vec!["dev-1", "dev-2", "team_leader"]
        );
        assert_eq!(
            fixture
                .backend
                .members()
                .iter()
                .map(|member| member.member_name().to_owned())
                .collect::<Vec<_>>(),
            vec!["dev-1", "dev-2", "team_leader"]
        );
        assert!(fixture.backend.member("dev-1").is_some());
        assert!(fixture.backend.member("dev-9").is_none());

        let removed = fixture.backend.remove_member("dev-1").unwrap();
        assert_eq!(removed.member_name(), "dev-1");
        assert_eq!(fixture.backend.member_names(), vec!["dev-2", "team_leader"]);
        assert_eq!(fixture.backend.activity().len(), 2);
        assert!(fixture.backend.remove_member("dev-1").is_none());
    }

    #[tokio::test]
    async fn upsert_team_info_round_trips_through_the_store() {
        let fixture = fixture();
        let mut info = TeamInfoRecord::new("alpha");
        info.display_name = "Alpha Team".to_owned();
        info.desc = "does things".to_owned();
        fixture.backend.upsert_team_info(info).await.unwrap();

        let stored = fixture.store.team("alpha").await.unwrap().unwrap();
        assert_eq!(stored.display_name, "Alpha Team");
        assert_eq!(stored.desc, "does things");
        assert!(stored.created > 0);
        assert!(stored.updated_at >= stored.created);
    }

    #[tokio::test]
    async fn list_tasks_projects_the_board() {
        let fixture = fixture();
        assert!(
            fixture
                .backend
                .list_tasks(&TaskFilter::default())
                .await
                .unwrap()
                .is_empty(),
            "a fresh team has no tasks"
        );

        let mutation = fixture
            .backend
            .tasks()
            .create_task(TaskGraphSpec {
                title: "ship it".to_owned(),
                content: "do the thing".to_owned(),
                ..TaskGraphSpec::default()
            })
            .await
            .unwrap();

        let tasks = fixture
            .backend
            .list_tasks(&TaskFilter::default())
            .await
            .unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].task_id, mutation.task.task_id);
        assert_eq!(tasks[0].title, "ship it");
        assert_eq!(tasks[0].status, TaskStatus::Pending);

        let filtered = fixture
            .backend
            .list_tasks(&TaskFilter {
                status: Some(TaskStatus::Completed),
                ..TaskFilter::default()
            })
            .await
            .unwrap();
        assert!(filtered.is_empty(), "no task has completed yet");
    }

    #[tokio::test]
    async fn snapshots_are_ordered_and_carry_the_board() {
        let fixture = fixture();
        fixture.backend.add_member(member(&fixture, "dev-2"));
        fixture.backend.add_member(member(&fixture, "dev-1"));

        let snapshot = fixture.backend.snapshot();
        assert_eq!(snapshot.team_name, "alpha");
        assert_eq!(snapshot.state, HarnessState::Idle);
        assert!(snapshot.tasks.is_empty(), "snapshot() carries no task rows");
        assert_eq!(
            snapshot
                .members
                .iter()
                .map(|member| member.member_name.clone())
                .collect::<Vec<_>>(),
            vec!["dev-1", "dev-2"]
        );

        fixture
            .backend
            .tasks()
            .create_task(TaskGraphSpec {
                title: "one".to_owned(),
                content: "c".to_owned(),
                ..TaskGraphSpec::default()
            })
            .await
            .unwrap();
        let with_tasks = fixture.backend.snapshot_with_tasks().await.unwrap();
        assert_eq!(with_tasks.tasks.len(), 1);
        assert_eq!(with_tasks.members.len(), 2);
        assert!(
            fixture.backend.snapshot().tasks.is_empty(),
            "snapshot() never reads the board"
        );
    }

    #[tokio::test]
    async fn post_event_returns_the_type_and_stamps_the_team_as_sender() {
        let fixture = fixture();
        let event_type = fixture
            .backend
            .post_event(
                TeamTopic::Task,
                EventPayload::TaskCreated(TaskCreatedEvent {
                    team_name: "alpha".to_owned(),
                    task_id: "task_1".to_owned(),
                    title: "one".to_owned(),
                    assignee: None,
                }),
            )
            .await
            .unwrap();
        assert_eq!(event_type, "task_created");

        let events = take_events(&fixture.events);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "task_created");
        assert_eq!(events[0].sender_id, "alpha");
        let round_trip = EventPayload::from_message(&events[0]).unwrap();
        assert_eq!(round_trip.event_type(), "task_created");
    }

    #[tokio::test]
    async fn publish_mutation_publishes_every_board_event() {
        let fixture = fixture();
        let mutation = fixture
            .backend
            .tasks()
            .create_task(TaskGraphSpec {
                title: "one".to_owned(),
                content: "c".to_owned(),
                ..TaskGraphSpec::default()
            })
            .await
            .unwrap();

        fixture.backend.publish_mutation(&mutation).await.unwrap();

        let expected: Vec<String> = mutation
            .events
            .iter()
            .map(|event| event.event_type().to_owned())
            .collect();
        assert!(!expected.is_empty(), "create_task publishes events");
        let events = take_events(&fixture.events);
        let published: Vec<String> = events
            .iter()
            .map(|event| event.event_type.clone())
            .collect();
        assert_eq!(published, expected);
        assert!(events.iter().all(|event| event.sender_id == "alpha"));
    }

    #[test]
    fn leader_member_name_falls_back_to_the_default() {
        let fixture = fixture();
        assert_eq!(
            fixture.backend.leader_member_name(),
            DEFAULT_LEADER_MEMBER_NAME
        );
        assert!(fixture.backend.spec().leader_member_name.is_none());
    }

    #[test]
    fn accessors_expose_the_injected_seams() {
        let fixture = fixture_for("beta", Some("boss"));
        let backend = &fixture.backend;
        assert_eq!(backend.team_name(), "beta");
        assert_eq!(backend.leader_member_name(), "boss");
        assert_eq!(backend.spec().team_name, "beta");
        assert_eq!(backend.store().name(), "memory");
        assert_eq!(backend.tasks().team_name(), "beta");
        assert_eq!(backend.tasks().store().name(), backend.store().name());
        assert_eq!(backend.messager().name(), "inprocess");
        let _: &Arc<dyn MemberSpawner> = backend.spawner();
        let _: &Arc<dyn WorkspaceProvider> = backend.workspace();
        let _: &Arc<MemberActivityRegistry> = backend.activity();
    }
}
