//! The task board: dependency graph, claims, review gate and subtask tree of one team.
//!
//! Python parity source: `openjiuwen/agent_teams/tools/task_manager.py` (`TeamTaskManager`, the
//! tool-facing task operations) and `tools/database/task_dao.py` (the atomic task mutations).
//! The Rust split is by responsibility rather than by Python class:
//!
//! | File | Contents |
//! | --- | --- |
//! | [`board`] | [`TaskBoard`] — every board operation over one team's [`TeamStore`](crate::storage::TeamStore) |
//! | [`graph`] | [`detect_cycle`], [`depth_of`] — the pure graph decisions |
//! | [`review`] | [`ReviewTally`], [`tally`], [`tally_with`] — the pure review-round decision |
//!
//! Every mutating call returns a [`TaskMutation`]: the persisted task row, the [`EventPayload`]s the
//! runtime publishes on the team task topic, and the ids of tasks that unblocked as a side effect
//! (the wake list for idle members). Every read returns the crate's plain view types
//! ([`crate::schema::TaskSummary`], [`crate::schema::TaskDetail`]) with their graph-derived fields filled.
//!
//! # Lifecycle
//!
//! `pending` is the only claimable status; a task with an unresolved dependency is created `blocked`
//! and is moved to `pending` by the board once every dependency is terminal. A task carrying
//! reviewers goes `in_progress -> in_review` on completion and only a settled review reaches
//! `completed`; a task without reviewers completes directly. `cancelled` and `completed` are
//! terminal and never unblock anything but their dependents.
//!
//! Two writes are board-level repairs rather than member-driven transitions, and they write the row
//! directly because [`crate::schema::TASK_TRANSITIONS`] deliberately has no such edge:
//!
//! * [`TaskBoard::reset`] — `completed` / `cancelled` / `in_review` back to `pending`, releasing the
//!   owner so the task is claimable again.
//! * [`TaskBoard::create_subtask`] — parks the parent at `blocked` while a subtask is open, even when
//!   the parent was `in_progress` (the member is suspended, not transitioned).
//!
//! # Refusals
//!
//! A policy refusal is not a failure of the board, so the two call shapes differ:
//!
//! * [`TaskOpResult`]-returning calls ([`TaskBoard::claim`], [`TaskBoard::assign`],
//!   [`TaskBoard::reassign`], [`TaskBoard::set_plan`]) report it as `ok = false` plus a stable
//!   reason string, so a tool can hand the reason to the model.
//! * [`TaskMutation`]-returning calls report it as `Err(Error::Validation(reason))`, because a
//!   mutation either produces events or nothing at all. The reason strings are the same ones.
//!
//! A missing member or task is always `Err(Error::NotFound)`; a name the caller supplied badly
//! (empty, unknown dependency target) is `Err(Error::Validation)`.

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

use crate::schema::{EventPayload, TaskRecord};

pub mod board;
pub mod graph;
pub mod review;

#[cfg(test)]
mod board_tests;

pub use board::{TaskBoard, REASON_STALE_CONTENT};
pub use graph::{depth_of, detect_cycle};
pub use review::{tally, tally_with, ReviewTally};

/// The board-wide limits a [`TaskBoard`] applies when a task does not pin its own.
///
/// These mirror `TeamAgentSpec`, which is where the runtime gets them:
/// [`default_max_review_rounds`](Self::default_max_review_rounds) is
/// `TeamAgentSpec::default_max_review_rounds` and
/// [`max_subtask_depth`](Self::max_subtask_depth) is `TeamAgentSpec::max_subtask_depth`.
///
/// The default `max_subtask_depth = 0` means "no subtasks": a member may not split its task at all.
/// The runtime raises it to enable the split, exactly as the Python tool boundary gates
/// `create_subtask` on `max_subtask_depth > 0` (`F_128`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TaskBoardLimits {
    /// Review-round ceiling for a task that does not carry its own; default `3`.
    pub default_max_review_rounds: u32,
    /// How deep a subtask chain may nest; default `0` (subtasks disabled).
    pub max_subtask_depth: usize,
}

impl Default for TaskBoardLimits {
    fn default() -> Self {
        Self {
            default_max_review_rounds: 3,
            max_subtask_depth: 0,
        }
    }
}

impl TaskBoardLimits {
    /// Builds limits with the Python defaults (`3` review rounds, no subtasks).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            default_max_review_rounds: 3,
            max_subtask_depth: 0,
        }
    }

    /// Returns these limits with the review-round ceiling replaced.
    #[must_use]
    pub const fn with_default_max_review_rounds(mut self, rounds: u32) -> Self {
        self.default_max_review_rounds = rounds;
        self
    }

    /// Returns these limits with the subtask depth ceiling replaced.
    #[must_use]
    pub const fn with_max_subtask_depth(mut self, depth: usize) -> Self {
        self.max_subtask_depth = depth;
        self
    }
}

/// Everything one mutating board call changed.
///
/// The runtime publishes [`events`](Self::events) on the team task topic and wakes the members named
/// by [`unblocked`](Self::unblocked); the board itself never touches the messager. `task` is the
/// authoritative row after the write (the store's timestamp included).
#[derive(Debug, Clone, PartialEq)]
pub struct TaskMutation {
    /// The task row as persisted after the mutation.
    pub task: TaskRecord,
    /// Events describing the mutation, in publication order.
    pub events: Vec<EventPayload>,
    /// Ids of tasks that moved `blocked -> pending` because of this mutation, sorted.
    pub unblocked: Vec<String>,
}

impl TaskMutation {
    /// Builds a mutation that changed `task` and emitted `events`.
    #[must_use]
    pub fn new(task: TaskRecord, events: Vec<EventPayload>) -> Self {
        Self {
            task,
            events,
            unblocked: Vec::new(),
        }
    }

    /// Returns the id of the mutated task.
    #[must_use]
    pub fn task_id(&self) -> &str {
        &self.task.task_id
    }
}

/// The outcome of a board call that may legitimately refuse without being an error.
///
/// Python parity: `TaskOpResult` of `tools/task_manager.py` (`ok` plus a human-readable `reason`),
/// minus its untyped `data` bag: the Rust call that has something to return returns it in its own
/// type (`TaskMutation`, [`ReviewTally`]).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TaskOpResult {
    /// Whether the operation was applied.
    pub ok: bool,
    /// Why it was refused, when [`ok`](Self::ok) is `false`.
    pub reason: Option<String>,
}

impl TaskOpResult {
    /// Returns a successful result.
    #[must_use]
    pub const fn ok() -> Self {
        Self {
            ok: true,
            reason: None,
        }
    }

    /// Returns a refused result carrying `reason`.
    ///
    /// The reason strings are part of the tool contract (they are shown to the model), so they are
    /// stable — see the table in `src/taskboard/CONTEXT.md`.
    #[must_use]
    pub fn failed(reason: impl Into<String>) -> Self {
        Self {
            ok: false,
            reason: Some(reason.into()),
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

impl From<Result<()>> for TaskOpResult {
    /// Maps a board error to a refusal, so an internal failure surfaces like any other refusal.
    fn from(result: Result<()>) -> Self {
        match result {
            Ok(()) => Self::ok(),
            Err(err) => Self::failed(err.to_string()),
        }
    }
}

/// Builds the `Error::NotFound` used when `task_id` is not on the board.
pub(crate) fn missing_task(task_id: &str) -> Error {
    Error::not_found(format!("task '{task_id}' not found"))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn limits_default_to_the_python_values() {
        let limits = TaskBoardLimits::default();
        assert_eq!(limits.default_max_review_rounds, 3);
        assert_eq!(limits.max_subtask_depth, 0);
        assert_eq!(TaskBoardLimits::new(), limits);
        assert_eq!(
            limits
                .with_default_max_review_rounds(5)
                .with_max_subtask_depth(2),
            TaskBoardLimits {
                default_max_review_rounds: 5,
                max_subtask_depth: 2
            }
        );
    }

    #[test]
    fn limits_default_a_partial_payload() {
        let limits: TaskBoardLimits = serde_json::from_str(r#"{"max_subtask_depth": 2}"#).unwrap();
        assert_eq!(limits.default_max_review_rounds, 3);
        assert_eq!(limits.max_subtask_depth, 2);
        assert_eq!(
            serde_json::from_str::<TaskBoardLimits>("{}").unwrap(),
            TaskBoardLimits::default()
        );
    }

    #[test]
    fn op_result_success_and_refusal_shapes() {
        let ok = TaskOpResult::ok();
        assert!(ok.is_ok());
        assert_eq!(ok.reason(), None);
        assert_eq!(
            serde_json::to_value(&ok).unwrap(),
            serde_json::json!({"ok": true, "reason": null})
        );

        let refused = TaskOpResult::failed("nope");
        assert!(!refused.is_ok());
        assert_eq!(refused.reason(), Some("nope"));
        assert_eq!(
            serde_json::to_value(&refused).unwrap(),
            serde_json::json!({"ok": false, "reason": "nope"})
        );

        let parsed: TaskOpResult = serde_json::from_str(r#"{"ok": true}"#).unwrap();
        assert_eq!(parsed, TaskOpResult::ok());
    }

    #[test]
    fn mutation_defaults_to_an_empty_wake_list() {
        let mutation = TaskMutation::new(TaskRecord::new("team", "task-1", "t", "c"), Vec::new());
        assert_eq!(mutation.task_id(), "task-1");
        assert_eq!(mutation.events, Vec::new());
        assert_eq!(mutation.unblocked, Vec::<String>::new());
    }

    #[test]
    fn missing_task_is_a_not_found_error() {
        let err = missing_task("task-1");
        assert!(
            matches!(err, Error::NotFound(_)),
            "expected not-found, got {err:?}"
        );
    }
}
