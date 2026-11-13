//! Unix-socket attach hosting for a running team.
//!
//! When the spec metadata (`attach.socket`) or `OPENJIUWEN_TEAM_SOCKET` names a path, the team
//! binds an [`AttachServer`] on `start` and serves it with [`BackendAttachHandler`], which maps
//! every [`AttachOp`] onto this team's [`TeamBackend`] (roster, mail, board, snapshot) and the
//! caller's cancellation token (`stop_team`).
//!
//! Policy:
//!
//! * bind is attempted once per `start`; a second `start` is a no-op;
//! * a bind failure is logged and the run continues unless the metadata sets
//!   `attach.required = true`, in which case `start` fails;
//! * `stop`/`stop_team` cancel the server, await its task with a bounded timeout and let the
//!   server drop, which removes the socket file.
//!
//! The handler is deliberately stateless apart from the backend handle and the team's
//! cancellation token, so the accept loop can serve concurrent clients.
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, ErrorKind, Result};
use serde_json::{json, Value};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use crate::attach::{AttachError, AttachHandler, AttachOp, AttachResult, AttachServer};
use crate::interaction::DeliverResult;
use crate::schema::{MemberStatus, TaskFilter, TaskGraphSpec, TaskStatus};

use super::backend::TeamBackend;
use super::spec_meta::AttachConfig;

/// How long shutdown waits for the accept loop before aborting it.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(2);

/// A bound attach server plus the task serving it.
///
/// Owned by the [`TeamAgent`](super::team_agent::TeamAgent); dropping it (or calling
/// [`AttachHost::shutdown`]) cancels the loop and removes the socket.
pub struct AttachHost {
    path: PathBuf,
    cancel: CancellationToken,
    task: JoinHandle<()>,
}
impl AttachHost {
    /// Binds `config.socket` and starts serving `handler`.
    ///
    /// # Errors
    ///
    /// Propagates [`AttachServer::bind`] failures (missing directory, permission denied, an
    /// existing non-socket file).
    pub async fn start(config: &AttachConfig, handler: Arc<dyn AttachHandler>) -> Result<Self> {
        let server = AttachServer::bind(&config.socket).await?;
        let path = server.path().to_path_buf();
        let cancel = server.cancellation_token();
        let task_path = path.clone();
        let task = tokio::spawn(async move {
            if let Err(error) = server.serve(handler).await {
                warn!(%error, path = %task_path.display(), "attach server stopped with an error");
            }
            debug!(path = %task_path.display(), "attach server task finished");
        });
        Ok(Self { path, cancel, task })
    }
    /// Returns the bound socket path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
    /// Stops serving and removes the socket, waiting at most [`SHUTDOWN_GRACE`].
    pub async fn shutdown(mut self) {
        self.cancel.cancel();
        if tokio::time::timeout(SHUTDOWN_GRACE, &mut self.task)
            .await
            .is_err()
        {
            warn!(path = %self.path.display(), "attach server did not stop in time; aborting");
        }
        // The task owns the server; the join above has dropped it, which unlinks the socket.
        if tokio::fs::remove_file(&self.path).await.is_err() && self.path.exists() {
            warn!(path = %self.path.display(), "socket file survived shutdown");
        }
    }
}
impl std::fmt::Debug for AttachHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AttachHost")
            .field("path", &self.path)
            .finish()
    }
}

/// [`AttachHandler`] over one team's backend.
///
/// The handler acts as the team's leader: it sends mail as the leader, creates tasks as the leader
/// and completes/claims tasks as the leader. `InvokeTool` is rejected — in-process tool invocation
/// is not part of the attach contract yet.
pub struct BackendAttachHandler {
    backend: Arc<TeamBackend>,
    leader: String,
    cancel: CancellationToken,
}
impl BackendAttachHandler {
    /// Builds a handler for `backend` acting as `leader`, cancelling `cancel` on `stop_team`.
    #[must_use]
    pub fn new(
        backend: Arc<TeamBackend>,
        leader: impl Into<String>,
        cancel: CancellationToken,
    ) -> Self {
        Self {
            backend,
            leader: leader.into(),
            cancel,
        }
    }
    /// Returns the leader member name this handler acts as.
    #[must_use]
    pub fn leader(&self) -> &str {
        &self.leader
    }
    /// Maps a backend error onto the attach error vocabulary.
    fn map(error: Error) -> AttachError {
        AttachError::new(error.kind(), error.to_string())
    }
}
#[async_trait]
impl AttachHandler for BackendAttachHandler {
    async fn handle(&self, op: AttachOp) -> AttachResult<Value> {
        match op {
            AttachOp::ListMembers => {
                let members: Vec<Value> = self
                    .backend
                    .members()
                    .into_iter()
                    .map(|member| {
                        json!({
                            "member_name": member.member_name(),
                            "display_name": member.record().display_name,
                            "role": member.role(),
                            "status": member.status(),
                            "execution": member.execution(),
                        })
                    })
                    .collect();
                Ok(Value::Array(members))
            }
            AttachOp::MemberStatus { member_name } => {
                let member = self.backend.member(&member_name).ok_or_else(|| {
                    AttachError::new(
                        ErrorKind::NotFound,
                        format!("member '{member_name}' not found"),
                    )
                })?;
                Ok(json!({
                    "member_name": member.member_name(),
                    "display_name": member.record().display_name,
                    "role": member.role(),
                    "status": member.status(),
                    "execution": member.execution(),
                }))
            }
            AttachOp::ReadInbox => {
                let rows = self.backend.unread(&self.leader).await.map_err(Self::map)?;
                let lines: Vec<String> = rows
                    .iter()
                    .map(|row| format!("{}: {}", row.from_member_name, row.content))
                    .collect();
                Ok(Value::String(lines.join("\n")))
            }
            AttachOp::SendMember { to, content } => {
                let row = self
                    .backend
                    .send_message(&self.leader, Some(&to), &content)
                    .await
                    .map_err(Self::map)?;
                to_data(&row)
            }
            AttachOp::Broadcast { content } => {
                let row = self
                    .backend
                    .send_message(&self.leader, None, &content)
                    .await
                    .map_err(Self::map)?;
                to_data(&row)
            }
            AttachOp::ListTasks { status } => {
                let filter = TaskFilter {
                    status,
                    ..TaskFilter::default()
                };
                Ok(Value::Array(
                    self.backend
                        .list_tasks(&filter)
                        .await
                        .map_err(Self::map)?
                        .iter()
                        .map(|task| serde_json::to_value(task).unwrap_or(Value::Null))
                        .collect(),
                ))
            }
            AttachOp::ClaimableTasks => {
                let rows = self.backend.tasks().claimable().await.map_err(Self::map)?;
                Ok(Value::Array(
                    rows.iter()
                        .map(|task| serde_json::to_value(task).unwrap_or(Value::Null))
                        .collect(),
                ))
            }
            AttachOp::TaskDetail { task_id } => {
                let detail = self
                    .backend
                    .tasks()
                    .detail(&task_id)
                    .await
                    .map_err(Self::map)?
                    .ok_or_else(|| {
                        AttachError::new(ErrorKind::NotFound, format!("task '{task_id}' not found"))
                    })?;
                to_data(&detail)
            }
            AttachOp::CreateTask {
                title,
                content,
                task_id,
                dependencies,
            } => {
                let spec = TaskGraphSpec {
                    title,
                    content,
                    task_id,
                    depends_on: dependencies,
                    assignee: None,
                    created_by: Some(self.leader.clone()),
                    ..TaskGraphSpec::default()
                };
                let mutation = self
                    .backend
                    .tasks()
                    .create_task(spec)
                    .await
                    .map_err(Self::map)?;
                self.backend
                    .publish_mutation(&mutation)
                    .await
                    .map_err(Self::map)?;
                to_data(&mutation.task)
            }
            AttachOp::UpdateTask {
                task_id,
                title,
                content,
            } => {
                let mut patch = crate::schema::TaskPatch::new();
                if let Some(title) = title {
                    patch = patch.with_title(title);
                }
                if let Some(content) = content {
                    patch = patch.with_content(content);
                }
                let updated = self
                    .backend
                    .store()
                    .update_task(self.backend.team_name(), &task_id, &patch)
                    .await
                    .map_err(Self::map)?;
                if !updated {
                    return Err(AttachError::new(
                        ErrorKind::NotFound,
                        format!("task '{task_id}' not found"),
                    ));
                }
                match self
                    .backend
                    .tasks()
                    .detail(&task_id)
                    .await
                    .map_err(Self::map)?
                {
                    Some(detail) => to_data(&detail),
                    None => Ok(Value::Null),
                }
            }
            AttachOp::ClaimTask { task_id } => {
                let result = self
                    .backend
                    .tasks()
                    .claim(&task_id, &self.leader)
                    .await
                    .map_err(Self::map)?;
                to_data(&result)
            }
            AttachOp::CompleteTask { task_id } => {
                let mutation = self
                    .backend
                    .tasks()
                    .complete(&task_id, &self.leader, "completed via attach")
                    .await
                    .map_err(Self::map)?;
                self.backend
                    .publish_mutation(&mutation)
                    .await
                    .map_err(Self::map)?;
                to_data(&mutation.task)
            }
            AttachOp::InvokeTool { name, .. } => Err(AttachError::new(
                ErrorKind::Unsupported,
                format!("invoke_tool '{name}' is not supported by the in-process attach handler"),
            )),
            AttachOp::Snapshot => {
                let snapshot = self
                    .backend
                    .snapshot_with_tasks()
                    .await
                    .map_err(Self::map)?;
                to_data(&snapshot)
            }
            AttachOp::StopTeam => {
                self.cancel.cancel();
                Ok(json!({"stopping": true}))
            }
        }
    }
}
impl std::fmt::Debug for BackendAttachHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackendAttachHandler")
            .field("team", &self.backend.team_name())
            .field("leader", &self.leader)
            .finish()
    }
}
/// Serializes `value` into a JSON reply, mapping a serde failure onto an attach protocol error.
fn to_data<T: serde::Serialize>(value: &T) -> AttachResult<Value> {
    serde_json::to_value(value)
        .map_err(|error| AttachError::protocol(format!("cannot serialize reply: {error}")))
}
/// Renders a member status as its wire string (host-side helpers and tests).
#[must_use]
pub fn status_text(status: MemberStatus) -> &'static str {
    status.as_str()
}
/// Renders a task status as its wire string (host-side helpers and tests).
#[must_use]
pub fn task_status_text(status: TaskStatus) -> &'static str {
    status.as_str()
}
/// Maps a [`DeliverResult`] onto an attach reply (used by hosts that forward HITL).
///
/// # Errors
///
/// Returns [`AttachError`] for `Unsupported`/`Refused`, mirroring the CLI's exit contract.
pub fn delivery_value(result: &DeliverResult) -> AttachResult<Value> {
    if result.ok {
        return Ok(json!({"delivered": true, "message_id": result.message_id}));
    }
    Err(AttachError::new(
        ErrorKind::Validation,
        format!(
            "interaction not delivered: {}",
            result.reason.as_deref().unwrap_or("unknown")
        ),
    ))
}
