//! The serializable value objects of the organisation task pool: the row projection
//! [`OrgTask`] with its [`OrgTaskSpec`] and [`OrgTaskFailure`], the [`OrgTaskSidecar`] JSON
//! carried in the store's `note` column, the [`OrgTaskFilter`] and [`OrgTaskOpResult`] read and
//! write shapes, and the [`OrgTaskEvent`] / [`OrgTaskMutation`] event vocabulary — data only, no
//! policy, so the pool in [`task_pool`](crate::org::task_pool) is the single writer of these
//! values.

use std::collections::BTreeMap;
use std::sync::Arc;

use openjiuwen_common::error::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::split_qualified;
use crate::schema::TaskStatus;

/// Why an organisation task was abandoned.
///
/// Python parity: `OrgTaskFailureCode` plus the failure message on `OrgTask`. Persisted in the
/// row's [`OrgTaskSidecar`], because [`TaskStatus`] has no failed state.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct OrgTaskFailure {
    /// Stable machine token (`timeout`, `dependency_missing`, …).
    pub code: String,
    /// Human sentence explaining the failure.
    pub message: String,
}

impl OrgTaskFailure {
    /// Builds a failure from its code and message.
    #[must_use]
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
}

/// The JSON sidecar the pool stores in [`TaskRecord::note`](crate::schema::TaskRecord::note).
///
/// It carries the fields [`TaskRecord`](crate::schema::TaskRecord) has no column for. A row whose
/// note does not parse is read with the defaults, so a row written by another release stays
/// loadable.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct OrgTaskSidecar {
    /// Team that owns the task.
    pub owner_team: String,
    /// Team the task is addressed to; equal to [`Self::owner_team`] until delegated.
    pub target_team: String,
    /// Decomposition depth: `0` for a root task, `parent.depth + 1` for a child.
    pub depth: usize,
    /// Why the task was abandoned, once
    /// [`OrgTaskPool::fail`](crate::org::task_pool::OrgTaskPool::fail) recorded it.
    pub failure: Option<OrgTaskFailure>,
    /// Free-form metadata copied from [`OrgTaskSpec::metadata`].
    pub metadata: BTreeMap<String, Value>,
}

/// The caller-supplied description of a new organisation task.
///
/// `target_team` falls back to `owner_team`, `created_by` to the owner team's leader, and an empty
/// `reviewers` list means no review gate.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct OrgTaskSpec {
    /// Short title rendered in list views; must not be blank.
    pub title: String,
    /// Full task body handed to the assignee.
    pub content: String,
    /// Team that owns the task; must be a declared chart team.
    pub owner_team: String,
    /// Team the work runs in; `None` means [`Self::owner_team`].
    pub target_team: Option<String>,
    /// Parent task this one decomposes; `None` for a root task.
    pub parent_task_id: Option<String>,
    /// Reviewer member ids, resolved inside the target team, or team-qualified ids.
    pub reviewers: Vec<String>,
    /// Creator member id; `None` means the owner team's leader.
    pub created_by: Option<String>,
    /// Free-form metadata carried into the [`OrgTaskSidecar`].
    pub metadata: BTreeMap<String, Value>,
}

impl OrgTaskSpec {
    /// Builds a root task of `owner_team` targeted at that same team.
    #[must_use]
    pub fn new(
        owner_team: impl Into<String>,
        title: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            title: title.into(),
            content: content.into(),
            owner_team: owner_team.into(),
            ..Self::default()
        }
    }

    /// Sets the team the work is addressed to.
    #[must_use]
    pub fn for_team(mut self, target_team: impl Into<String>) -> Self {
        self.target_team = Some(target_team.into());
        self
    }

    /// Makes this task a decomposition child of `parent_task_id`.
    #[must_use]
    pub fn under(mut self, parent_task_id: impl Into<String>) -> Self {
        self.parent_task_id = Some(parent_task_id.into());
        self
    }

    /// Replaces the reviewer set.
    #[must_use]
    pub fn reviewed_by<I: IntoIterator<Item = String>>(mut self, reviewers: I) -> Self {
        self.reviewers = reviewers.into_iter().collect();
        self
    }

    /// Sets the creator member id.
    #[must_use]
    pub fn by(mut self, member: impl Into<String>) -> Self {
        self.created_by = Some(member.into());
        self
    }
}

/// One organisation task, decoded from its stored row plus sidecar.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct OrgTask {
    /// Task identity (`task_<32 hex chars>`).
    pub task_id: String,
    /// Organisation the task belongs to.
    pub organization_id: String,
    /// Short title.
    pub title: String,
    /// Full task body.
    pub content: String,
    /// Lifecycle status; `cancelled` with a [`OrgTask::failure`] is the failed state.
    pub status: TaskStatus,
    /// Team that owns the task.
    pub owner_team: String,
    /// Team the task is addressed to.
    pub target_team: String,
    /// Qualified (`<team>::<member>`) assignee.
    pub assignee: Option<String>,
    /// Qualified reviewer member ids; empty means no review gate.
    pub reviewers: Vec<String>,
    /// Review round the task is currently in.
    pub review_round: u32,
    /// Per-task review-round ceiling.
    pub max_review_rounds: u32,
    /// Parent task of a decomposition child.
    pub parent_task_id: Option<String>,
    /// Decomposition depth (`0` for a root task).
    pub depth: usize,
    /// Qualified creator.
    pub created_by: String,
    /// The assignee's closing summary.
    pub summary: Option<String>,
    /// Why the task was abandoned, when it failed.
    pub failure: Option<OrgTaskFailure>,
    /// Free-form metadata carried from the specification.
    pub metadata: BTreeMap<String, Value>,
    /// Millisecond timestamp of the last write, as stamped by the store.
    pub updated_at: u64,
}

impl OrgTask {
    /// Returns `true` for the terminal statuses `completed` / `cancelled`.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        self.status.is_terminal()
    }

    /// Returns `true` while the task is not terminal.
    #[must_use]
    pub fn is_open(&self) -> bool {
        !self.is_terminal()
    }

    /// Splits the qualified assignee into its team and member parts.
    #[must_use]
    pub fn assignee_parts(&self) -> Option<(&str, &str)> {
        split_qualified(self.assignee.as_deref()?)
    }
}

/// Filter applied by [`OrgTaskPool::list`](crate::org::task_pool::OrgTaskPool::list).
///
/// An unset field does not constrain the query. `team` matches the task's `target_team`;
/// `assignee` accepts a qualified id or a bare member id (qualified with `team` when set).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct OrgTaskFilter {
    /// Restrict to one status.
    pub status: Option<TaskStatus>,
    /// Restrict to tasks addressed to one team.
    pub team: Option<String>,
    /// Restrict to one assignee.
    pub assignee: Option<String>,
    /// Restrict to the children of one parent task.
    pub parent_task_id: Option<String>,
    /// Cap on the number of returned tasks.
    pub limit: Option<usize>,
}

/// The outcome of an operation that may legitimately refuse without being an error.
///
/// A refusal is model-visible, so it is `ok = false` plus a stable reason rather than an `Err`;
/// [`Self::task`] carries the row as persisted when the operation was applied.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct OrgTaskOpResult {
    /// Whether the operation was applied.
    pub ok: bool,
    /// Why it was refused, when [`Self::ok`] is `false`.
    pub reason: Option<String>,
    /// The task row after the mutation, when known.
    pub task: Option<OrgTask>,
}

impl OrgTaskOpResult {
    /// Returns a successful result carrying the mutated `task`.
    #[must_use]
    pub const fn ok(task: OrgTask) -> Self {
        Self {
            ok: true,
            reason: None,
            task: Some(task),
        }
    }

    /// Returns a refused result carrying `reason` and no task.
    #[must_use]
    pub fn failed(reason: impl Into<String>) -> Self {
        Self {
            ok: false,
            reason: Some(reason.into()),
            task: None,
        }
    }

    /// Returns `true` when the operation was applied.
    #[must_use]
    pub const fn is_ok(&self) -> bool {
        self.ok
    }

    /// Returns the refusal reason, or `None` when the operation was applied.
    #[must_use]
    pub fn reason(&self) -> Option<&str> {
        self.reason.as_deref()
    }
}

impl From<Result<()>> for OrgTaskOpResult {
    /// Maps an internal failure to a refusal; the success arm carries no task.
    fn from(result: Result<()>) -> Self {
        match result {
            Ok(()) => Self {
                ok: true,
                reason: None,
                task: None,
            },
            Err(err) => Self::failed(err.to_string()),
        }
    }
}

/// Something that happened to an organisation task.
///
/// Serialized with the `type` tag (`{"type": "claimed", …}`), so a host can forward the values
/// verbatim onto the team bus.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OrgTaskEvent {
    /// A task was created.
    Created {
        /// The new task id.
        task_id: String,
        /// Team that owns the task.
        owner_team: String,
        /// Team the task is addressed to.
        target_team: String,
    },
    /// A task changed target team (and possibly assignee).
    Delegated {
        /// The delegated task id.
        task_id: String,
        /// Team that delegated the work.
        from_team: String,
        /// Team that received it.
        to_team: String,
        /// Qualified member the work was handed to, when one was named.
        to_member: Option<String>,
    },
    /// A member claimed the task.
    Claimed {
        /// The claimed task id.
        task_id: String,
        /// Qualified member that won the claim.
        assignee: String,
    },
    /// A pre-assigned member started the task.
    Started {
        /// The started task id.
        task_id: String,
        /// Qualified member that started it.
        assignee: String,
    },
    /// The assignee finished the work (before any review gate).
    Completed {
        /// The completed task id.
        task_id: String,
        /// Qualified member that finished it.
        assignee: String,
    },
    /// The assignee abandoned the task.
    Failed {
        /// The abandoned task id.
        task_id: String,
        /// Stable failure code.
        code: String,
    },
    /// A reviewer cast a verdict.
    Reviewed {
        /// The reviewed task id.
        task_id: String,
        /// Qualified reviewer.
        reviewer: String,
        /// `true` for a pass, `false` for a rework request.
        passed: bool,
    },
    /// A task moved `blocked -> pending` because a dependency settled.
    Unblocked {
        /// The unblocked task id.
        task_id: String,
    },
}

/// Everything one mutating call changed.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct OrgTaskMutation {
    /// The task row as persisted after the mutation.
    pub task: OrgTask,
    /// Events describing the mutation, in publication order.
    pub events: Vec<OrgTaskEvent>,
    /// Ids of tasks that moved `blocked -> pending` because of this mutation, sorted.
    pub unblocked: Vec<String>,
}

impl OrgTaskMutation {
    /// Returns the id of the mutated task.
    #[must_use]
    pub fn task_id(&self) -> &str {
        &self.task.task_id
    }
}

/// Receives every [`OrgTaskEvent`] the pool emits, so a runtime can publish them onto the bus.
pub type OrgTaskEventSink = Arc<dyn Fn(OrgTaskEvent) + Send + Sync>;
