//! The cross-process [`TeamClient`]: drive a team hosted by *another* process over its socket.
//!
//! Python parity source: the operation methods of `openjiuwen/agent_teams/external/client.py`.
//! Python's external client reaches the runtime it was started next to; this port keeps that
//! shape and adds the transport the Rust runtime publishes, the attach bridge in [`crate::attach`]:
//! one JSON object per line over a Unix domain socket, one in-flight request at a time.
//!
//! # Relationship to the other implementations
//!
//! [`SocketTeamClient`] implements the same [`TeamClient`] seam as
//! [`StoreTeamClient`](crate::skill::StoreTeamClient), so the `team-member` command tree and the
//! `openjiuwen-team-mcp` server cannot tell them apart: every operation maps to one
//! [`AttachOp`] and the payload it returns is decoded into the very same
//! outcome type the in-process client builds. The local parity checks live here too — an empty
//! target and over-limit content are answered without touching the socket, exactly as
//! [`StoreTeamClient`](crate::skill::StoreTeamClient) answers them without touching the store.
//!
//! # One connection, lazily opened, dropped on failure
//!
//! [`AttachClient`] takes `&mut self` for every call: one connection
//! carries at most one in-flight request, and the protocol correlates the answer by request id.
//! This client therefore keeps its connection in a [`tokio::sync::Mutex`] and holds the guard for
//! the whole exchange, which is what serialises concurrent callers onto that single connection.
//! The connection is opened on the first operation ([`SocketTeamClient::new`] does no I/O), so a
//! client can be built while the hosting process is still starting; the bounded connect (and every
//! round trip) uses the client's [`timeout`](SocketTeamClient::timeout).
//!
//! A transport-level failure leaves the socket unusable — the attach client never resynchronises
//! the stream — so a failed exchange with a `Transport` or `Timeout` kind drops the connection and
//! the next call connects again. Errors reported *by the remote team* (a refused task, an unknown
//! tool) keep their kind and do **not** drop the connection, because the exchange itself succeeded.
//!
//! # Errors
//!
//! Every fallible operation returns [`openjiuwen_common::error::Error`]: the typed error the attach
//! bridge carried across the wire, or [`Error::validation`] when a payload does not match its
//! documented shape. Nothing here panics, and a missing socket surfaces as
//! [`ErrorKind::Transport`].

use std::path::{Path, PathBuf};
use std::time::Duration;

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde_json::Value;
use tokio::sync::{Mutex, MutexGuard};

use openjiuwen_common::error::{Error, ErrorKind, Result};

use crate::attach::client::DEFAULT_TIMEOUT;
use crate::attach::{AttachClient, AttachOp};
use crate::constants::{BROADCAST_TARGET, MAX_MESSAGE_CONTENT_CHARS};
use crate::schema::{MemberSnapshot, TaskDetail, TaskStatus, TaskSummary};

use super::client::{
    member_tools_for, operator_tools, CreateOutcome, MemberSummary, OpOutcome, SendOutcome,
    TeamClient, TeamToolDef,
};
use super::descriptor::{ClientScope, TeamJoinDescriptor};

/// A [`TeamClient`] that drives a team hosted by another process over its attach socket.
///
/// Build it with [`SocketTeamClient::new`] (or [`SocketTeamClient::with_timeout`]) for a lazily
/// connected client, or with [`SocketTeamClient::connect`] when the socket must be reachable before
/// the first operation. `path` is the socket the hosting process published — the same value the
/// descriptor resolves from its `socket_path` field or the `OPENJIUWEN_TEAM_SOCKET` variable.
///
/// The descriptor this client carries decides the scope ([`ClientScope`]), the identity reported by
/// [`TeamClient::member_name`] and the tool catalogue [`TeamClient::tools`] advertises; it is never
/// rewritten by an operation.
#[derive(Debug)]
pub struct SocketTeamClient {
    /// Join descriptor this client acts under (scope, identity and tool-set settings).
    descriptor: TeamJoinDescriptor,
    /// The attach socket every operation is sent to.
    path: PathBuf,
    /// Deadline for the connect attempt and for one request/response round trip.
    timeout: Duration,
    /// The connection, opened on first use and dropped after a transport failure.
    ///
    /// Taken for the whole exchange, which is what keeps a single in-flight request on this
    /// connection. It is a `tokio` mutex because a caller waits across the await.
    connection: Mutex<Option<AttachClient>>,
}

impl SocketTeamClient {
    /// Builds a client that connects on its first operation.
    ///
    /// No I/O happens here and the socket does not have to exist yet; the connect attempt is made
    /// (and its failure reported) by the first [`TeamClient`] call. The deadline defaults to the
    /// attach bridge's [`DEFAULT_TIMEOUT`].
    #[must_use]
    pub fn new(descriptor: TeamJoinDescriptor, path: impl Into<PathBuf>) -> Self {
        Self::with_timeout(descriptor, path, DEFAULT_TIMEOUT)
    }

    /// Builds a lazily connecting client with an explicit deadline.
    ///
    /// `timeout` bounds both the connect attempt and each request/response exchange, so a wedged
    /// hosting process fails the operation instead of blocking the CLI forever.
    #[must_use]
    pub fn with_timeout(
        descriptor: TeamJoinDescriptor,
        path: impl Into<PathBuf>,
        timeout: Duration,
    ) -> Self {
        Self {
            descriptor,
            path: path.into(),
            timeout,
            connection: Mutex::new(None),
        }
    }

    /// Builds a client and establishes the connection now.
    ///
    /// Use this when an unreachable socket must be reported up front (a host that is about to
    /// spawn a member, or a test that asserts the failure) instead of on the first operation.
    ///
    /// # Errors
    ///
    /// Returns the attach bridge's typed error: [`ErrorKind::Transport`] when the socket cannot be
    /// reached (missing path, refused connection, not a socket), [`ErrorKind::Timeout`] when the
    /// default deadline elapses first, and [`ErrorKind::Unsupported`] on a target without Unix
    /// domain sockets.
    pub async fn connect(descriptor: TeamJoinDescriptor, path: impl AsRef<Path>) -> Result<Self> {
        let client = Self::new(descriptor, path.as_ref().to_path_buf());
        drop(client.connection().await?);
        Ok(client)
    }

    /// Returns the attach socket this client talks to.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the join descriptor this client acts under.
    #[must_use]
    pub fn descriptor(&self) -> &TeamJoinDescriptor {
        &self.descriptor
    }

    /// Returns the deadline applied to the connect attempt and to one round trip.
    #[must_use]
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Returns whether a connection is currently held.
    ///
    /// `false` only means "not connected yet" or "the last exchange failed and dropped the
    /// connection": the next operation connects again. It never probes the socket, so this costs
    /// nothing and cannot hang.
    pub async fn is_connected(&self) -> bool {
        self.connection.lock().await.is_some()
    }

    /// Locks the connection slot, connecting first when it is empty.
    ///
    /// The returned guard is held for the whole exchange by [`Self::call`]; a connect failure is
    /// reported as the attach bridge's typed error and leaves the slot empty, so a later operation
    /// retries instead of caching the failure.
    async fn connection(&self) -> Result<MutexGuard<'_, Option<AttachClient>>> {
        let mut slot = self.connection.lock().await;
        if slot.is_none() {
            let client = AttachClient::connect_with_timeout(&self.path, self.timeout)
                .await
                .map_err(|error| {
                    tracing::debug!(
                        path = %self.path.display(),
                        kind = %error.kind,
                        "attach client could not connect"
                    );
                    error.into_error()
                })?;
            *slot = Some(client);
        }
        Ok(slot)
    }

    /// Sends one op and returns its `data` payload.
    ///
    /// A `Transport` or `Timeout` failure drops the connection, because the attach client cannot
    /// resynchronise a stream once a frame is lost; anything else (a validation error, or a failure
    /// the remote handler reported in-band) leaves the connection usable.
    async fn call(&self, op: AttachOp) -> Result<Value> {
        let op_name = op.name();
        let mut slot = self.connection().await?;
        let Some(client) = slot.as_mut() else {
            // `connection()` either filled the slot or returned an error; this arm is defensive.
            return Err(Error::transport(format!(
                "attach connection to {} is not open",
                self.path.display()
            )));
        };
        match client.call(op).await {
            Ok(value) => Ok(value),
            Err(error) => {
                if matches!(error.kind, ErrorKind::Transport | ErrorKind::Timeout) {
                    tracing::debug!(
                        path = %self.path.display(),
                        op = op_name,
                        kind = %error.kind,
                        "attach exchange failed; dropping the connection so the next call reconnects"
                    );
                    *slot = None;
                }
                Err(error.into_error())
            }
        }
    }

    /// Sends one op and decodes its payload into `T`.
    ///
    /// `label` names the wire op in the error message, so a shape mismatch is diagnosable without
    /// the caller re-reading this module.
    ///
    /// # Errors
    ///
    /// Propagates [`Self::call`] failures, and reports a payload that does not match the
    /// documented shape as [`Error::validation`].
    async fn decode<T: DeserializeOwned>(&self, op: AttachOp, label: &str) -> Result<T> {
        let value = self.call(op).await?;
        serde_json::from_value(value).map_err(|error| {
            Error::validation(format!(
                "attach op '{label}' returned an unexpected payload: {error}"
            ))
        })
    }
}

#[async_trait]
impl TeamClient for SocketTeamClient {
    /// Returns the descriptor's scope, which selects the CLI's command tree.
    fn scope(&self) -> ClientScope {
        self.descriptor.scope
    }

    /// Returns the descriptor's member name, the identity remote claims and sends use.
    fn member_name(&self) -> &str {
        &self.descriptor.member_name
    }

    /// Reads the inbox through `read_inbox`; the hosting team renders and advances the read state,
    /// so the text is byte-identical to the in-process client's.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error`] when the socket cannot be reached, the deadline
    /// elapses, or the remote inbox read fails.
    async fn read_inbox(&self) -> Result<String> {
        self.decode(AttachOp::ReadInbox, "read_inbox").await
    }

    /// Sends `content` to `to`, broadcasting when the trimmed target is [`BROADCAST_TARGET`].
    ///
    /// The local parity checks of the in-process client run first, before anything crosses the
    /// socket: an empty target is [`Error::validation`], and content longer than
    /// [`MAX_MESSAGE_CONTENT_CHARS`] is a refusal ([`SendOutcome::failed`]), not an error.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error`] when the socket cannot be reached, the deadline
    /// elapses, or the remote send fails.
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
        let op = if to == BROADCAST_TARGET {
            AttachOp::Broadcast {
                content: content.to_owned(),
            }
        } else {
            AttachOp::SendMember {
                to: to.to_owned(),
                content: content.to_owned(),
            }
        };
        let label = op.name();
        self.decode(op, label).await
    }

    /// Creates a task through the hosting team's board; a board refusal comes back as
    /// `ok = false` with the board's reason.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error`] when the socket cannot be reached, the deadline
    /// elapses, or the remote create fails.
    async fn create_task(
        &self,
        title: &str,
        content: &str,
        task_id: Option<&str>,
        dependencies: &[String],
    ) -> Result<CreateOutcome> {
        let op = AttachOp::CreateTask {
            title: title.to_owned(),
            content: content.to_owned(),
            task_id: task_id.map(str::to_owned),
            dependencies: dependencies.to_vec(),
        };
        self.decode(op, "create_task").await
    }

    /// Edits a task that has not started yet.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error`] when the socket cannot be reached, the deadline
    /// elapses, or the remote update fails.
    async fn update_task(
        &self,
        task_id: &str,
        title: Option<&str>,
        content: Option<&str>,
    ) -> Result<OpOutcome> {
        let op = AttachOp::UpdateTask {
            task_id: task_id.to_owned(),
            title: title.map(str::to_owned),
            content: content.map(str::to_owned),
        };
        self.decode(op, "update_task").await
    }

    /// Lists the team's board, optionally restricted to one status.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error`] when the socket cannot be reached, the deadline
    /// elapses, or the remote list fails.
    async fn list_tasks(&self, status: Option<TaskStatus>) -> Result<Vec<TaskSummary>> {
        self.decode(AttachOp::ListTasks { status }, "list_tasks")
            .await
    }

    /// Lists the pending, unclaimed tasks the member may claim.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error`] when the socket cannot be reached, the deadline
    /// elapses, or the remote list fails.
    async fn claimable_tasks(&self) -> Result<Vec<TaskSummary>> {
        self.decode(AttachOp::ClaimableTasks, "claimable_tasks")
            .await
    }

    /// Returns one task's full detail, or `None` when the id is unknown (the payload is JSON
    /// `null`).
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error`] when the socket cannot be reached, the deadline
    /// elapses, or the remote read fails.
    async fn get_task(&self, task_id: &str) -> Result<Option<TaskDetail>> {
        let op = AttachOp::TaskDetail {
            task_id: task_id.to_owned(),
        };
        self.decode(op, "task_detail").await
    }

    /// Claims a pending task for the descriptor's member.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error`] when the socket cannot be reached, the deadline
    /// elapses, or the remote claim fails.
    async fn claim_task(&self, task_id: &str) -> Result<OpOutcome> {
        let op = AttachOp::ClaimTask {
            task_id: task_id.to_owned(),
        };
        self.decode(op, "claim_task").await
    }

    /// Completes a task owned by the descriptor's member.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error`] when the socket cannot be reached, the deadline
    /// elapses, or the remote completion fails.
    async fn complete_task(&self, task_id: &str) -> Result<OpOutcome> {
        let op = AttachOp::CompleteTask {
            task_id: task_id.to_owned(),
        };
        self.decode(op, "complete_task").await
    }

    /// Lists the team roster.
    ///
    /// The wire payload is the full [`MemberSnapshot`] the hosting team publishes; this client
    /// projects it onto the CLI's [`MemberSummary`] row, dropping the display name and the
    /// execution status the command tree does not render — the same three fields the in-process
    /// client builds from its store records.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error`] when the socket cannot be reached, the deadline
    /// elapses, or the remote roster read fails.
    async fn list_members(&self) -> Result<Vec<MemberSummary>> {
        let roster: Vec<MemberSnapshot> =
            self.decode(AttachOp::ListMembers, "list_members").await?;
        Ok(roster
            .into_iter()
            .map(|member| MemberSummary::new(member.member_name, member.role, member.status))
            .collect())
    }

    /// Invokes a tool by name with its raw JSON arguments and returns the tool's own result.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error`] when the socket cannot be reached or the
    /// deadline elapses, and otherwise whatever the remote tool reported with its kind preserved —
    /// [`ErrorKind::NotFound`] for an unknown tool, [`ErrorKind::Validation`] for a bad argument
    /// set.
    async fn invoke_tool(&self, name: &str, arguments: Value) -> Result<Value> {
        let op = AttachOp::InvokeTool {
            name: name.to_owned(),
            arguments,
        };
        self.call(op).await
    }

    /// Returns the tool set the descriptor's scope advertises.
    ///
    /// The rule is the in-process client's, verbatim: a member scope gets
    /// [`member_tools_for`] under the descriptor's teammate mode, dispatch mode and subtask depth,
    /// an operator scope gets [`operator_tools`].
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
