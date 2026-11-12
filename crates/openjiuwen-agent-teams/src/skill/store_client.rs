//! The store-backed [`TeamClient`]: one team's store, task board and messager behind the CLI seam.
//!
//! This file is long on purpose (~1200 lines, roughly half of it the `#[cfg(test)]` suite the node
//! contract requires beside the module): the client is one object whose trait impl only reads as a
//! whole — every operation is a store/board call plus the event it publishes — so splitting it would
//! scatter a single contract. The pure helpers (argument parsing, row projection, the clock) are
//! already at the bottom of the file, outside the `impl`s.
//!
//! Python parity source: the operation methods of `openjiuwen/agent_teams/external/client.py`
//! (`TeamMemberClient` and `TeamOperatorClient`) plus the tool dispatch of
//! `openjiuwen/agent_teams/tools/task_manager.py`.
//!
//! The client owns no policy of its own: every operation is a thin adapter over the crate's
//! existing seams, so the same command tree runs against the in-memory backend (tests) and a live
//! team (the runtime bridge) without a transport-specific branch:
//!
//! * mailbox and roster writes go through [`TeamStore`] (insert, unread scan, read cursor, watermark),
//! * task operations go through [`TaskBoard`] (create, claim, complete, verify, view),
//! * every state change is announced on the team event bus through [`Messager`].
//!
//! Errors are reserved for infrastructure failures; a *refused* operation (task not claimable,
//! content over the limit, an edit of a started task) is a successful call carrying `ok = false`
//! and a stable reason, exactly like the board's own refusal shape.
//!
//! Two deliberate deviations from a naive port are documented where they happen:
//! [`StoreTeamClient::update_task`] bypasses the board (there is no board edit operation) and
//! [`StoreTeamClient::invoke_tool`] answers the `view_task` `note` action with `null` because the
//! crate does not persist completion notes as a read view.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use serde_json::{json, Value};
use uuid::Uuid;

use openjiuwen_common::error::{Error, Result};

use crate::constants::{BROADCAST_TARGET, MAX_MESSAGE_CONTENT_CHARS};
use crate::messager::Messager;
use crate::schema::{
    BroadcastEvent, EventPayload, MessageEvent, TaskDetail, TaskFilter, TaskGraphSpec, TaskPatch,
    TaskStatus, TaskSummary, TeamTopic, Verdict,
};
use crate::storage::{MessageRecord, TeamStore};
use crate::taskboard::TaskBoard;

use super::client::{
    member_tools_for, operator_tools, CreateOutcome, MemberSummary, OpOutcome, SendOutcome,
    TeamClient, TeamToolDef,
};
use super::descriptor::{ClientScope, TeamJoinDescriptor};

/// Rendered for a mailbox with nothing left to read.
const INBOX_EMPTY: &str = "(inbox empty)";

/// Action the `view_task` tool performs when the caller does not name one.
const DEFAULT_VIEW_ACTION: &str = "list";

/// The store-backed [`TeamClient`].
///
/// One instance drives one team: the descriptor fixes the team, the session, the member identity
/// and the client scope, and the three seams supply the state. Construct it per CLI invocation —
/// it holds no cache and no lock of its own.
///
/// ```ignore
/// let client = StoreTeamClient::new(descriptor, store, board, messager);
/// assert_eq!(client.member_name(), "dev-1");
/// ```
pub struct StoreTeamClient {
    /// Identity, scope and team settings the client acts under.
    descriptor: TeamJoinDescriptor,
    /// Persistence seam: mailbox, roster and the direct task rows `update_task` patches.
    store: Arc<dyn TeamStore>,
    /// Task board of [`TeamJoinDescriptor::team_name`].
    board: Arc<TaskBoard>,
    /// Event bus each state change is announced on.
    messager: Arc<dyn Messager>,
}

impl StoreTeamClient {
    /// Builds a client for one joined team.
    ///
    /// `store`, `taskboard` and `messager` must already belong to the descriptor's team: the client
    /// does not re-key them, it only stamps the descriptor's member name on writes and events.
    #[must_use]
    pub fn new(
        descriptor: TeamJoinDescriptor,
        store: Arc<dyn TeamStore>,
        taskboard: Arc<TaskBoard>,
        messager: Arc<dyn Messager>,
    ) -> Self {
        Self {
            descriptor,
            store,
            board: taskboard,
            messager,
        }
    }

    /// Returns the descriptor this client was built from.
    #[must_use]
    pub fn descriptor(&self) -> &TeamJoinDescriptor {
        &self.descriptor
    }

    /// Returns the team the client addresses.
    fn team(&self) -> &str {
        &self.descriptor.team_name
    }

    /// Publishes `events` on `topic`, each envelope stamped with this member as its sender.
    ///
    /// A payload that cannot be serialized is a [`Error::Validation`] failure; nothing is published
    /// in that case for the failing payload only, so the earlier ones may already be on the bus.
    async fn publish(&self, topic: TeamTopic, events: Vec<EventPayload>) -> Result<()> {
        for payload in events {
            let message = payload.to_message(self.member_name())?;
            self.messager.publish(topic, message).await?;
        }
        Ok(())
    }

    /// Implements the `view_task` tool (`list`, `claimable`, `get`, `in_review`, `note`).
    async fn view_task(&self, arguments: &Value) -> Result<Value> {
        let action = arguments
            .get("action")
            .and_then(Value::as_str)
            .unwrap_or(DEFAULT_VIEW_ACTION);
        match action {
            "list" => {
                let status = match arguments.get("status").and_then(Value::as_str) {
                    Some(raw) => Some(parse_task_status(raw)?),
                    None => None,
                };
                let tasks = self.list_tasks(status).await?;
                Ok(Value::Array(tasks.iter().map(task_row).collect()))
            }
            "claimable" => {
                let tasks = self.claimable_tasks().await?;
                Ok(Value::Array(
                    tasks
                        .iter()
                        .map(|task| json!({ "task_id": task.task_id, "title": task.title }))
                        .collect(),
                ))
            }
            "in_review" => {
                let tasks = self.list_tasks(Some(TaskStatus::InReview)).await?;
                Ok(Value::Array(tasks.iter().map(task_row).collect()))
            }
            "get" => {
                let task_id = required_str(arguments, "task_id")?;
                match self.get_task(task_id).await? {
                    Some(detail) => serde_json::to_value(&detail).map_err(|err| {
                        Error::validation(format!("failed to serialize task '{task_id}': {err}"))
                    }),
                    None => Ok(Value::Null),
                }
            }
            // Python's `note` view reads a completion note; this crate keeps notes on the task row
            // itself, so there is no separate read view to answer with.
            "note" => Ok(Value::Null),
            other => Err(Error::validation(format!(
                "unknown view_task action '{other}'"
            ))),
        }
    }

    /// Implements the teammate `claim_task` tool (`status = claimed | completed`).
    async fn claim_task_tool(&self, arguments: &Value) -> Result<Value> {
        let task_id = required_str(arguments, "task_id")?;
        let status = required_str(arguments, "status")?;
        let outcome = match status {
            "claimed" => self.claim_task(task_id).await?,
            "completed" => self.complete_task(task_id).await?,
            other => {
                return Err(Error::validation(format!(
                    "unknown claim_task status '{other}'; expected 'claimed' or 'completed'"
                )))
            }
        };
        Ok(json!({ "ok": outcome.ok, "reason": outcome.reason }))
    }

    /// Implements the `verify_task` tool (one reviewer verdict on a task under review).
    async fn verify_task_tool(&self, arguments: &Value) -> Result<Value> {
        let task_id = required_str(arguments, "task_id")?;
        let decision = required_str(arguments, "decision")?;
        let verdict = parse_verdict(decision)?;
        let feedback = arguments.get("feedback").and_then(Value::as_str);
        let outcome = self
            .board
            .verify(task_id, self.member_name(), verdict, feedback.unwrap_or(""))
            .await;
        match outcome {
            Ok(mutation) => {
                let task_id = mutation.task.task_id.clone();
                self.publish(TeamTopic::Task, mutation.events).await?;
                Ok(json!({
                    "ok": true,
                    "task_id": task_id,
                    "decision": verdict.as_str(),
                    "feedback": feedback,
                }))
            }
            Err(err) => Ok(json!({ "ok": false, "reason": err.to_string() })),
        }
    }

    /// Implements the `send_message` tool.
    async fn send_message_tool(&self, arguments: &Value) -> Result<Value> {
        let to = required_str(arguments, "to")?;
        let content = required_str(arguments, "content")?;
        let outcome = self.send_message(to, content).await?;
        Ok(json!({
            "ok": outcome.ok,
            "message_id": outcome.message_id,
            "reason": outcome.reason,
        }))
    }
}

#[async_trait]
impl TeamClient for StoreTeamClient {
    fn scope(&self) -> ClientScope {
        self.descriptor.scope
    }

    fn member_name(&self) -> &str {
        &self.descriptor.member_name
    }

    /// Renders the member's unread messages and then consumes them.
    ///
    /// One line per message, rendered as
    /// `[<timestamp>] <sender> -> <recipient> : <content>`, where `<recipient>` is `*` for a
    /// broadcast row and otherwise the addressed member name. Rows arrive already ordered by
    /// timestamp then id. Reading marks every direct row read and advances the member's broadcast
    /// watermark to the newest broadcast timestamp, so a second read renders `(inbox empty)`.
    async fn read_inbox(&self) -> Result<String> {
        let team = self.team();
        let member = self.member_name();
        let messages = self.store.unread_messages(team, member).await?;
        if messages.is_empty() {
            return Ok(INBOX_EMPTY.to_string());
        }
        let mut lines = Vec::with_capacity(messages.len());
        let mut direct_ids = Vec::new();
        let mut watermark: Option<u64> = None;
        for message in &messages {
            let to = if message.broadcast {
                BROADCAST_TARGET
            } else {
                message
                    .to_member_name
                    .as_deref()
                    .unwrap_or(BROADCAST_TARGET)
            };
            lines.push(format!(
                "[{}] {} -> {} : {}",
                message.timestamp, message.from_member_name, to, message.content
            ));
            if message.broadcast {
                watermark =
                    Some(watermark.map_or(message.timestamp, |seen| seen.max(message.timestamp)));
            } else {
                direct_ids.push(message.message_id.clone());
            }
        }
        if !direct_ids.is_empty() {
            self.store.mark_read(team, member, &direct_ids).await?;
        }
        if let Some(timestamp) = watermark {
            self.store
                .set_read_watermark(team, member, timestamp)
                .await?;
        }
        Ok(lines.join("\n"))
    }

    /// Persists the message and announces it on the `Message` topic.
    ///
    /// `to = "*"` writes a broadcast row (read state is then the per-member watermark, never a row
    /// flag) and publishes [`EventPayload::Broadcast`]; any other target writes a direct row and
    /// publishes [`EventPayload::Message`]. Leading and trailing whitespace around the target is
    /// ignored, an empty target is a validation error, and content longer than
    /// [`MAX_MESSAGE_CONTENT_CHARS`] is refused before anything is persisted or published.
    async fn send_message(&self, to: &str, content: &str) -> Result<SendOutcome> {
        let to = to.trim();
        if to.is_empty() {
            return Err(Error::validation("target member name is required"));
        }
        let length = content.chars().count();
        if length > MAX_MESSAGE_CONTENT_CHARS {
            return Ok(SendOutcome::failed(format!(
                "content too long: {length} characters exceeds the limit of {MAX_MESSAGE_CONTENT_CHARS}"
            )));
        }
        let member = self.member_name();
        let team = self.team();
        let message_id = Uuid::new_v4().to_string();
        let timestamp = now_millis();
        let broadcast = to == BROADCAST_TARGET;
        let row = if broadcast {
            MessageRecord::broadcast(team, &message_id, member, content, timestamp)
        } else {
            MessageRecord::direct(team, &message_id, member, to, content, timestamp)
        };
        self.store.insert_message(row).await?;
        let payload = if broadcast {
            EventPayload::Broadcast(BroadcastEvent {
                message_id: message_id.clone(),
                from_member_name: member.to_owned(),
            })
        } else {
            EventPayload::Message(MessageEvent {
                message_id: message_id.clone(),
                from_member_name: member.to_owned(),
                to_member_name: Some(to.to_owned()),
            })
        };
        self.publish(TeamTopic::Message, vec![payload]).await?;
        Ok(SendOutcome::sent(message_id))
    }

    /// Creates a task through the board and publishes the board's events on the `Task` topic.
    ///
    /// A board refusal (duplicate id, unknown dependency target, cycle) is reported as
    /// `ok = false` with the board's reason; it is a policy answer, not a transport failure.
    async fn create_task(
        &self,
        title: &str,
        content: &str,
        task_id: Option<&str>,
        dependencies: &[String],
    ) -> Result<CreateOutcome> {
        let spec = TaskGraphSpec {
            title: title.to_owned(),
            content: content.to_owned(),
            task_id: task_id.map(str::to_owned),
            depends_on: dependencies.to_vec(),
            created_by: Some(self.member_name().to_owned()),
            ..TaskGraphSpec::default()
        };
        match self.board.create_task(spec).await {
            Ok(mutation) => {
                let task_id = mutation.task.task_id.clone();
                self.publish(TeamTopic::Task, mutation.events).await?;
                Ok(CreateOutcome::created(task_id))
            }
            Err(err) => Ok(CreateOutcome::failed(err.to_string())),
        }
    }

    /// Edits a task that has not started, writing through the store.
    ///
    /// There is no board edit operation, so the patch is applied by [`TeamStore::update_task`]
    /// after the row is checked: only `pending` and `blocked` tasks are editable, everything else
    /// is refused with the task's current status in the reason. **No event is published**, matching
    /// Python's operator edit path, which changes content without a status transition; a reader
    /// picks the change up from the bumped revision.
    async fn update_task(
        &self,
        task_id: &str,
        title: Option<&str>,
        content: Option<&str>,
    ) -> Result<OpOutcome> {
        let team = self.team();
        let Some(task) = self.store.task(team, task_id).await? else {
            return Ok(OpOutcome::failed(format!("task '{task_id}' not found")));
        };
        if !matches!(task.status, TaskStatus::Pending | TaskStatus::Blocked) {
            return Ok(OpOutcome::failed(format!(
                "task '{task_id}' cannot be edited from status '{}' (only pending or blocked tasks can be edited)",
                task.status.as_str()
            )));
        }
        let patch = TaskPatch {
            title: title.map(str::to_owned),
            content: content.map(str::to_owned),
            ..TaskPatch::default()
        };
        if self.store.update_task(team, task_id, &patch).await? {
            Ok(OpOutcome::ok())
        } else {
            Ok(OpOutcome::failed(format!("task '{task_id}' not found")))
        }
    }

    async fn list_tasks(&self, status: Option<TaskStatus>) -> Result<Vec<TaskSummary>> {
        self.board
            .view(&TaskFilter {
                status,
                ..TaskFilter::default()
            })
            .await
    }

    async fn claimable_tasks(&self) -> Result<Vec<TaskSummary>> {
        self.board.claimable().await
    }

    async fn get_task(&self, task_id: &str) -> Result<Option<TaskDetail>> {
        self.board.detail(task_id).await
    }

    async fn claim_task(&self, task_id: &str) -> Result<OpOutcome> {
        Ok(OpOutcome::from(
            self.board.claim(task_id, self.member_name()).await?,
        ))
    }

    /// Completes a task the client's member owns and publishes the board's events.
    ///
    /// A board refusal (not the assignee, stale content, wrong status) is `ok = false` with the
    /// board's reason.
    async fn complete_task(&self, task_id: &str) -> Result<OpOutcome> {
        match self.board.complete(task_id, self.member_name(), "").await {
            Ok(mutation) => {
                self.publish(TeamTopic::Task, mutation.events).await?;
                Ok(OpOutcome::ok())
            }
            Err(err) => Ok(OpOutcome::failed(err.to_string())),
        }
    }

    async fn list_members(&self) -> Result<Vec<MemberSummary>> {
        let records = self.store.list_members(self.team()).await?;
        Ok(records
            .into_iter()
            .map(|record| MemberSummary::new(record.member_name, record.role, record.status))
            .collect())
    }

    /// Dispatches one tool call with its raw JSON arguments.
    ///
    /// The supported names and their argument keys are exactly the ones
    /// [`member_tools_for`]/[`operator_tools`] advertise; the answer shapes are:
    ///
    /// * `view_task` — an array of `{task_id, title, status, assignee}` for `list`/`in_review`
    ///   (optionally filtered by `status`), an array of `{task_id, title}` for `claimable`, the
    ///   serialized task for `get` (`null` when unknown) and `null` for `note`;
    /// * `claim_task` — `{ok, reason}`;
    /// * `verify_task` — `{ok, task_id, decision, feedback}`, or `{ok: false, reason}` when the
    ///   board refuses the verdict;
    /// * `send_message` — `{ok, message_id, reason}`.
    async fn invoke_tool(&self, name: &str, arguments: Value) -> Result<Value> {
        match name {
            "view_task" => self.view_task(&arguments).await,
            "claim_task" => self.claim_task_tool(&arguments).await,
            "verify_task" => self.verify_task_tool(&arguments).await,
            "send_message" => self.send_message_tool(&arguments).await,
            other => Err(Error::not_found(format!("unknown tool '{other}'"))),
        }
    }

    fn tools(&self) -> Vec<TeamToolDef> {
        match self.descriptor.scope {
            ClientScope::Member => member_tools_for(
                self.descriptor.teammate_mode,
                self.descriptor.dispatch_mode,
                self.descriptor.max_subtask_depth,
            ),
            ClientScope::Operator => operator_tools(),
        }
    }
}

/// Projects one list row of the `view_task` tool.
fn task_row(task: &TaskSummary) -> Value {
    json!({
        "task_id": task.task_id,
        "title": task.title,
        "status": task.status.as_str(),
        "assignee": task.assignee,
    })
}

/// Reads a required string argument, naming the missing key exactly as Python does.
fn required_str<'a>(arguments: &'a Value, key: &str) -> Result<&'a str> {
    arguments
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| Error::validation(format!("'{key}' is required")))
}

/// Parses a task status wire string (`pending`, `blocked`, `in_progress`, …).
fn parse_task_status(raw: &str) -> Result<TaskStatus> {
    serde_json::from_str::<TaskStatus>(&format!("\"{raw}\""))
        .map_err(|_| Error::validation(format!("unknown task status '{raw}'")))
}

/// Parses a review verdict wire string (`pass` or `fail`).
fn parse_verdict(raw: &str) -> Result<Verdict> {
    serde_json::from_str::<Verdict>(&format!("\"{raw}\""))
        .map_err(|_| Error::validation(format!("unknown review verdict '{raw}'")))
}

/// Returns the wall-clock time in epoch milliseconds (`0` when the clock predates 1970).
fn now_millis() -> u64 {
    let Ok(elapsed) = SystemTime::now().duration_since(UNIX_EPOCH) else {
        return 0;
    };
    u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;
    use parking_lot::Mutex;
    use serde_json::json;

    use openjiuwen_common::error::Result;

    use crate::messager::{EventSink, InProcessMessager, Messager};
    use crate::schema::{
        DispatchMode, EventMessage, MemberMode, MemberStatus, TaskGraphSpec, TeamRole, TeamTopic,
    };
    use crate::storage::{InMemoryTeamStore, MemberRecord, TeamStore};
    use crate::taskboard::TaskBoard;

    use super::*;

    /// Records the published topic strings while delegating delivery to a real in-process bus.
    ///
    /// The envelope carries no topic, so the only way a test can observe *where* a client
    /// published is to record it at the seam; delivery (and therefore subscriber assertions) still
    /// goes through [`InProcessMessager`].
    struct RecordingMessager {
        session: String,
        team: String,
        inner: InProcessMessager,
        published: Arc<Mutex<Vec<String>>>,
    }

    impl RecordingMessager {
        fn new(session: &str, team: &str) -> Self {
            Self {
                session: session.to_string(),
                team: team.to_string(),
                inner: InProcessMessager::new(),
                published: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    #[async_trait]
    impl Messager for RecordingMessager {
        fn name(&self) -> &'static str {
            self.inner.name()
        }

        fn subscribe(&self, member: &str, topics: &[TeamTopic], sink: EventSink) -> Result<()> {
            self.inner.subscribe(member, topics, sink)
        }

        fn unsubscribe(&self, member: &str) -> Result<()> {
            self.inner.unsubscribe(member)
        }

        async fn publish(&self, topic: TeamTopic, event: EventMessage) -> Result<()> {
            self.published
                .lock()
                .push(topic.topic(&self.session, &self.team));
            self.inner.publish(topic, event).await
        }

        async fn send(&self, target: &str, event: EventMessage) -> Result<()> {
            self.inner.send(target, event).await
        }

        async fn subscribe_transport(&self, team_name: &str) -> Result<()> {
            self.inner.subscribe_transport(team_name).await
        }

        fn subscribers(&self, topic: TeamTopic) -> Vec<String> {
            self.inner.subscribers(topic)
        }

        fn is_started(&self) -> bool {
            self.inner.is_started()
        }

        async fn start(&self) -> Result<()> {
            self.inner.start().await
        }

        async fn stop(&self) -> Result<()> {
            self.inner.stop().await
        }
    }

    /// One team's store, board, bus and client, built the way the CLI builds them.
    struct Fixture {
        store: Arc<dyn TeamStore>,
        board: Arc<TaskBoard>,
        client: StoreTeamClient,
        events: Arc<Mutex<Vec<EventMessage>>>,
        topics: Arc<Mutex<Vec<String>>>,
    }

    impl Fixture {
        fn new(scope: ClientScope) -> Self {
            let store: Arc<dyn TeamStore> = Arc::new(InMemoryTeamStore::new());
            let board = Arc::new(TaskBoard::with_defaults(store.clone(), "alpha"));
            let messager = Arc::new(RecordingMessager::new("s1", "alpha"));
            let topics = Arc::clone(&messager.published);
            let events: Arc<Mutex<Vec<EventMessage>>> = Arc::new(Mutex::new(Vec::new()));
            let sink_events = Arc::clone(&events);
            let sink: EventSink = Arc::new(move |event| sink_events.lock().push(event));
            messager
                .subscribe("observer", &[TeamTopic::Message, TeamTopic::Task], sink)
                .unwrap();
            let descriptor = TeamJoinDescriptor {
                session_id: "s1".to_string(),
                team_name: "alpha".to_string(),
                member_name: "dev-1".to_string(),
                scope,
                ..TeamJoinDescriptor::default()
            };
            let client =
                StoreTeamClient::new(descriptor, store.clone(), Arc::clone(&board), messager);
            Self {
                store,
                board,
                client,
                events,
                topics,
            }
        }

        fn published(&self) -> Vec<EventMessage> {
            self.events.lock().clone()
        }

        fn published_topics(&self) -> Vec<String> {
            self.topics.lock().clone()
        }

        async fn seed_task(&self, task_id: &str) {
            self.seed_task_with(task_id, Vec::new()).await;
        }

        async fn seed_task_with(&self, task_id: &str, reviewers: Vec<String>) {
            let spec = TaskGraphSpec {
                title: "title".to_string(),
                content: "body".to_string(),
                task_id: Some(task_id.to_string()),
                reviewers,
                ..TaskGraphSpec::default()
            };
            self.board.create_task(spec).await.unwrap();
        }

        /// Seeds a task and drives it to `in_review` with `dev-1` as author and reviewer.
        async fn seed_in_review(&self, task_id: &str) {
            self.seed_task_with(task_id, vec!["dev-1".to_string()])
                .await;
            self.client.claim_task(task_id).await.unwrap();
            self.client.complete_task(task_id).await.unwrap();
        }
    }

    #[tokio::test]
    async fn send_direct_message_persists_the_row_and_publishes() {
        let fixture = Fixture::new(ClientScope::Member);
        let outcome = fixture
            .client
            .send_message("team_leader", "starting now")
            .await
            .unwrap();
        assert!(outcome.ok);
        let message_id = outcome.message_id.clone().unwrap();
        assert_eq!(outcome.reason, None);

        let rows = fixture
            .store
            .unread_messages("alpha", "team_leader")
            .await
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].message_id, message_id);
        assert_eq!(rows[0].from_member_name, "dev-1");
        assert_eq!(rows[0].to_member_name.as_deref(), Some("team_leader"));
        assert!(!rows[0].broadcast);
        assert!(!rows[0].is_read);

        assert_eq!(
            fixture.published_topics(),
            vec!["session:s1:team:alpha:message".to_string()]
        );
        let events = fixture.published();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "message");
        assert_eq!(events[0].sender_id, "dev-1");
        assert_eq!(events[0].payload["message_id"], json!(message_id));
        assert_eq!(events[0].payload["to_member_name"], json!("team_leader"));
    }

    #[tokio::test]
    async fn broadcast_message_is_a_row_without_a_recipient() {
        let fixture = Fixture::new(ClientScope::Member);
        let outcome = fixture.client.send_message("*", "all hands").await.unwrap();
        assert!(outcome.ok);

        let rows = fixture
            .store
            .unread_messages("alpha", "team_leader")
            .await
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].broadcast);
        assert_eq!(rows[0].to_member_name, None);
        assert_eq!(rows[0].content, "all hands");

        let events = fixture.published();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "broadcast");
        assert_eq!(events[0].sender_id, "dev-1");
        assert_eq!(events[0].payload["from_member_name"], json!("dev-1"));
    }

    #[tokio::test]
    async fn over_limit_content_is_refused_without_persisting() {
        let fixture = Fixture::new(ClientScope::Member);
        let content = "x".repeat(MAX_MESSAGE_CONTENT_CHARS + 1);
        let outcome = fixture
            .client
            .send_message("team_leader", &content)
            .await
            .unwrap();
        assert!(!outcome.ok);
        assert_eq!(outcome.message_id, None);
        let reason = outcome.reason.unwrap();
        assert!(
            reason.contains("content too long"),
            "unexpected reason: {reason}"
        );
        assert!(reason.contains(&MAX_MESSAGE_CONTENT_CHARS.to_string()));
        let rows = fixture
            .store
            .unread_messages("alpha", "dev-1")
            .await
            .unwrap();
        assert_eq!(rows.len(), 0);
        assert_eq!(fixture.published(), Vec::new());
        assert_eq!(fixture.published_topics(), Vec::<String>::new());
    }

    #[tokio::test]
    async fn empty_target_is_a_validation_error() {
        let fixture = Fixture::new(ClientScope::Member);
        let err = fixture.client.send_message("  ", "hi").await.unwrap_err();
        assert!(err.to_string().contains("target member name is required"));
    }

    #[tokio::test]
    async fn inbox_renders_lines_and_advances_the_cursors() {
        let fixture = Fixture::new(ClientScope::Member);
        fixture
            .store
            .insert_message(MessageRecord::direct(
                "alpha",
                "m1",
                "team_leader",
                "dev-1",
                "hi",
                1_000,
            ))
            .await
            .unwrap();
        fixture
            .store
            .insert_message(MessageRecord::broadcast(
                "alpha", "m2", "lead", "all", 2_000,
            ))
            .await
            .unwrap();

        let rendered = fixture.client.read_inbox().await.unwrap();
        assert_eq!(
            rendered,
            "[1000] team_leader -> dev-1 : hi\n[2000] lead -> * : all"
        );
        assert_eq!(fixture.client.read_inbox().await.unwrap(), "(inbox empty)");
        assert_eq!(
            fixture
                .store
                .read_watermark("alpha", "dev-1")
                .await
                .unwrap(),
            Some(2_000)
        );
        assert_eq!(fixture.client.read_inbox().await.unwrap(), "(inbox empty)");
    }

    #[tokio::test]
    async fn create_task_pins_the_id_and_persists_dependencies() {
        let fixture = Fixture::new(ClientScope::Operator);
        let first = fixture
            .client
            .create_task("build", "do it", Some("t-1"), &[])
            .await
            .unwrap();
        assert_eq!(first.task_id.as_deref(), Some("t-1"));
        let second = fixture
            .client
            .create_task("test", "verify it", Some("t-2"), &["t-1".to_string()])
            .await
            .unwrap();
        assert!(second.ok, "unexpected refusal: {:?}", second.reason);

        let blocked = fixture.store.task("alpha", "t-2").await.unwrap().unwrap();
        assert_eq!(blocked.status, TaskStatus::Blocked);
        assert_eq!(blocked.created_by.as_deref(), Some("dev-1"));
        assert_eq!(
            fixture.store.dependencies("alpha", "t-2").await.unwrap(),
            vec!["t-1".to_string()]
        );

        let events = fixture.published();
        assert!(events.iter().any(|event| {
            event.event_type == "task_created"
                && event.payload["task_id"] == json!("t-2")
                && event.sender_id == "dev-1"
        }));
        assert!(fixture
            .published_topics()
            .iter()
            .all(|topic| topic == "session:s1:team:alpha:task"));
    }

    #[tokio::test]
    async fn claim_and_complete_run_as_the_descriptor_member() {
        let fixture = Fixture::new(ClientScope::Member);
        fixture.seed_task("t-1").await;

        assert_eq!(
            fixture.client.claim_task("t-1").await.unwrap(),
            OpOutcome::ok()
        );
        let claimed = fixture.store.task("alpha", "t-1").await.unwrap().unwrap();
        assert_eq!(claimed.assignee.as_deref(), Some("dev-1"));
        assert_eq!(claimed.status, TaskStatus::InProgress);

        assert_eq!(
            fixture.client.complete_task("t-1").await.unwrap(),
            OpOutcome::ok()
        );
        let completed = fixture.store.task("alpha", "t-1").await.unwrap().unwrap();
        assert_eq!(completed.status, TaskStatus::Completed);
        assert!(fixture
            .published()
            .iter()
            .any(|event| event.event_type == "task_status_changed" && event.sender_id == "dev-1"));
    }

    #[tokio::test]
    async fn update_task_bumps_the_revision_only_while_editable() {
        let fixture = Fixture::new(ClientScope::Operator);
        fixture.seed_task("t-1").await;
        assert_eq!(
            fixture
                .client
                .update_task("t-1", Some("new title"), None)
                .await
                .unwrap(),
            OpOutcome::ok()
        );
        let edited = fixture.store.task("alpha", "t-1").await.unwrap().unwrap();
        assert_eq!(edited.title, "new title");
        assert_eq!(edited.revision, 1);

        fixture.seed_task("t-2").await;
        fixture.client.claim_task("t-2").await.unwrap();
        let refused = fixture
            .client
            .update_task("t-2", Some("nope"), None)
            .await
            .unwrap();
        assert!(!refused.ok);
        assert_eq!(
            refused.reason.as_deref(),
            Some(
                "task 't-2' cannot be edited from status 'in_progress' (only pending or blocked tasks can be edited)"
            )
        );

        let missing = fixture
            .client
            .update_task("missing", None, Some("body"))
            .await
            .unwrap();
        assert_eq!(missing.reason.as_deref(), Some("task 'missing' not found"));
        // Editing never publishes: the only events so far come from the claim above.
        assert!(fixture
            .published()
            .iter()
            .all(|event| event.event_type != "task_created"));
    }

    #[tokio::test]
    async fn roster_is_sorted_and_statuses_are_mapped() {
        let fixture = Fixture::new(ClientScope::Operator);
        let mut zeta = MemberRecord::new("alpha", "zeta", TeamRole::Teammate);
        zeta.status = MemberStatus::Ready;
        fixture.store.insert_member(zeta).await.unwrap();
        fixture
            .store
            .insert_member(MemberRecord::new("alpha", "alpha", TeamRole::Leader))
            .await
            .unwrap();

        let members = fixture.client.list_members().await.unwrap();
        let names: Vec<&str> = members
            .iter()
            .map(|member| member.member_name.as_str())
            .collect();
        assert_eq!(names, ["alpha", "zeta"]);
        assert_eq!(members[0].role, TeamRole::Leader);
        assert_eq!(members[0].status, MemberStatus::Unstarted);
        assert_eq!(members[1].role, TeamRole::Teammate);
        assert_eq!(members[1].status, MemberStatus::Ready);
    }

    #[tokio::test]
    async fn view_task_tool_answers_every_action() {
        let fixture = Fixture::new(ClientScope::Member);
        fixture.seed_task("t-1").await;

        let list = fixture
            .client
            .invoke_tool("view_task", json!({}))
            .await
            .unwrap();
        assert_eq!(
            list,
            json!([{
                "task_id": "t-1",
                "title": "title",
                "status": "pending",
                "assignee": null,
            }])
        );
        let filtered = fixture
            .client
            .invoke_tool("view_task", json!({ "status": "pending" }))
            .await
            .unwrap();
        assert_eq!(filtered, list);
        let none = fixture
            .client
            .invoke_tool("view_task", json!({ "status": "completed" }))
            .await
            .unwrap();
        assert_eq!(none, json!([]));

        let detail = fixture
            .client
            .invoke_tool("view_task", json!({ "action": "get", "task_id": "t-1" }))
            .await
            .unwrap();
        assert_eq!(detail["task_id"], json!("t-1"));
        assert_eq!(detail["status"], json!("pending"));
        assert_eq!(detail["content"], json!("body"));
        assert_eq!(
            fixture
                .client
                .invoke_tool("view_task", json!({ "action": "get", "task_id": "zzz" }))
                .await
                .unwrap(),
            Value::Null
        );

        let claimable = fixture
            .client
            .invoke_tool("view_task", json!({ "action": "claimable" }))
            .await
            .unwrap();
        assert_eq!(claimable, json!([{ "task_id": "t-1", "title": "title" }]));
        assert_eq!(
            fixture
                .client
                .invoke_tool("view_task", json!({ "action": "in_review" }))
                .await
                .unwrap(),
            json!([])
        );
        assert_eq!(
            fixture
                .client
                .invoke_tool("view_task", json!({ "action": "note", "task_id": "t-1" }))
                .await
                .unwrap(),
            Value::Null
        );

        let err = fixture
            .client
            .invoke_tool("view_task", json!({ "action": "explode" }))
            .await
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("unknown view_task action 'explode'"));
    }

    #[tokio::test]
    async fn claim_task_tool_claims_and_completes() {
        let fixture = Fixture::new(ClientScope::Member);
        fixture.seed_task("t-1").await;

        let claimed = fixture
            .client
            .invoke_tool(
                "claim_task",
                json!({ "task_id": "t-1", "status": "claimed" }),
            )
            .await
            .unwrap();
        assert_eq!(claimed, json!({ "ok": true, "reason": null }));
        let completed = fixture
            .client
            .invoke_tool(
                "claim_task",
                json!({ "task_id": "t-1", "status": "completed" }),
            )
            .await
            .unwrap();
        assert_eq!(completed, json!({ "ok": true, "reason": null }));
        assert_eq!(
            fixture
                .store
                .task("alpha", "t-1")
                .await
                .unwrap()
                .unwrap()
                .status,
            TaskStatus::Completed
        );

        let missing_task = fixture
            .client
            .invoke_tool("claim_task", json!({ "status": "claimed" }))
            .await
            .unwrap_err();
        assert!(missing_task.to_string().contains("'task_id' is required"));
        let missing_status = fixture
            .client
            .invoke_tool("claim_task", json!({ "task_id": "t-1" }))
            .await
            .unwrap_err();
        assert!(missing_status.to_string().contains("'status' is required"));
    }

    #[tokio::test]
    async fn verify_task_tool_reports_both_verdicts() {
        let fixture = Fixture::new(ClientScope::Member);
        fixture.seed_in_review("t-1").await;
        let passed = fixture
            .client
            .invoke_tool(
                "verify_task",
                json!({ "task_id": "t-1", "decision": "pass", "feedback": "looks good" }),
            )
            .await
            .unwrap();
        assert_eq!(
            passed,
            json!({
                "ok": true,
                "task_id": "t-1",
                "decision": "pass",
                "feedback": "looks good",
            })
        );
        assert_eq!(
            fixture
                .store
                .task("alpha", "t-1")
                .await
                .unwrap()
                .unwrap()
                .status,
            TaskStatus::Completed
        );
        assert!(fixture
            .published()
            .iter()
            .any(|event| event.event_type == "task_verified"));

        fixture.seed_in_review("t-2").await;
        let failed = fixture
            .client
            .invoke_tool(
                "verify_task",
                json!({ "task_id": "t-2", "decision": "fail" }),
            )
            .await
            .unwrap();
        assert_eq!(
            failed,
            json!({ "ok": true, "task_id": "t-2", "decision": "fail", "feedback": null })
        );
        assert_eq!(
            fixture
                .store
                .task("alpha", "t-2")
                .await
                .unwrap()
                .unwrap()
                .status,
            TaskStatus::InProgress
        );

        let refused = fixture
            .client
            .invoke_tool(
                "verify_task",
                json!({ "task_id": "zzz", "decision": "pass" }),
            )
            .await
            .unwrap();
        assert_eq!(refused["ok"], json!(false));
        assert!(refused["reason"].as_str().unwrap().contains("not found"));

        let bad_verdict = fixture
            .client
            .invoke_tool(
                "verify_task",
                json!({ "task_id": "t-1", "decision": "maybe" }),
            )
            .await
            .unwrap_err();
        assert!(bad_verdict.to_string().contains("'maybe'"));
    }

    #[tokio::test]
    async fn send_message_tool_and_unknown_tools() {
        let fixture = Fixture::new(ClientScope::Member);
        let sent = fixture
            .client
            .invoke_tool("send_message", json!({ "to": "lead", "content": "hi" }))
            .await
            .unwrap();
        assert_eq!(sent["ok"], json!(true));
        assert!(sent["message_id"].is_string());
        assert_eq!(sent["reason"], Value::Null);

        let content = "x".repeat(MAX_MESSAGE_CONTENT_CHARS + 1);
        let refused = fixture
            .client
            .invoke_tool("send_message", json!({ "to": "lead", "content": content }))
            .await
            .unwrap();
        assert_eq!(refused["ok"], json!(false));
        assert!(refused["reason"]
            .as_str()
            .unwrap()
            .contains("content too long"));

        let unknown = fixture
            .client
            .invoke_tool("explode", json!({}))
            .await
            .unwrap_err();
        assert!(matches!(unknown, Error::NotFound(_)));
        assert!(unknown.to_string().contains("unknown tool 'explode'"));
    }

    #[tokio::test]
    async fn tool_catalogue_follows_the_descriptor_scope() {
        let operator = Fixture::new(ClientScope::Operator);
        let names: Vec<String> = operator
            .client
            .tools()
            .into_iter()
            .map(|tool| tool.name)
            .collect();
        assert_eq!(
            names,
            operator_tools()
                .into_iter()
                .map(|tool| tool.name)
                .collect::<Vec<_>>()
        );
        assert_eq!(names.first().map(String::as_str), Some("read_inbox"));

        let member = Fixture::new(ClientScope::Member);
        let member_names: Vec<String> = member
            .client
            .tools()
            .into_iter()
            .map(|tool| tool.name)
            .collect();
        assert_eq!(
            member_names,
            vec!["view_task", "send_message", "claim_task"]
        );
        assert_eq!(
            member_names,
            member_tools_for(MemberMode::BuildMode, DispatchMode::Autonomous, 0)
                .into_iter()
                .map(|tool| tool.name)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn scope_and_member_name_come_from_the_descriptor() {
        let fixture = Fixture::new(ClientScope::Member);
        assert_eq!(fixture.client.scope(), ClientScope::Member);
        assert_eq!(fixture.client.member_name(), "dev-1");
        assert_eq!(fixture.client.descriptor().team_name, "alpha");
        assert_eq!(fixture.client.descriptor().session_id, "s1");
    }
}
