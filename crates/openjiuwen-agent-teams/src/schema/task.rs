//! Task-board records: the persisted task row, its list/detail views and the mutation DTOs.
//!
//! Python parity source: `openjiuwen/agent_teams/schema/task.py` (`TaskSummary`, `TaskDetail`,
//! `TaskGraphSpec`) plus the task row of `openjiuwen/agent_teams/tools/models.py` and the review
//! vote record of the task board.
//!
//! Everything here is cold, serializable data plus pure helpers: the task board owns the
//! behaviour (dependency graph, claims, review gate) and codes against these shapes. Fields are
//! named exactly as they are on the wire (`snake_case`), unknown keys are tolerated so a record
//! persisted by another release still loads, and missing keys fall back to the Python defaults —
//! notably [`TaskRecord::max_review_rounds`], which defaults to `3` rather than `0`.
//!
//! Rust deviations, both documented in `src/schema/CONTEXT.md`: [`TaskPatch`] expresses an
//! assignee clear with the explicit [`TaskPatch::clear_assignee`] flag (an `Option<String>` alone
//! cannot distinguish "leave unchanged" from "set to null"), and `change_notes` holds rendered
//! note strings ([`TaskRevision::to_note`]) rather than a JSON-encoded list.

use serde::{Deserialize, Serialize};

use super::status::{MemberStatus, TaskStatus};

/// Per-task review-round ceiling used when a record does not carry one (Python team default `3`).
const DEFAULT_MAX_REVIEW_ROUNDS: u32 = 3;

/// A reviewer's verdict on a task.
///
/// Wire values are the Python strings `pass` / `fail`; the Python default (and the only verdict a
/// missing value maps to) is [`Verdict::Pass`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// The reviewer accepted the task.
    Pass,
    /// The reviewer rejected the task; it returns to `in_progress`.
    Fail,
}

impl Default for Verdict {
    /// Python default: `pass`.
    fn default() -> Self {
        Self::Pass
    }
}

impl Verdict {
    /// Returns the Python wire string of this verdict.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Fail => "fail",
        }
    }
}

/// One task on the team's task board (the persisted row).
///
/// Python parity: the task row of `tools/models.py`. `revision` is bumped on every content edit,
/// `observed_revision` records the revision the current assignee was actually shown and
/// `parent_revision` the parent's revision a subtask was derived from — a difference between them
/// is the standing "not yet seen" / "stale" signal, there is no separate notification flag.
///
/// The hand-written [`Default`] mirrors the Python defaults: `status = pending`,
/// `max_review_rounds = 3`, every counter `0`, every collection empty and every optional field
/// `None`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TaskRecord {
    /// Team the task belongs to.
    pub team_name: String,
    /// Task identity, unique within the team.
    pub task_id: String,
    /// Short title rendered in list views.
    pub title: String,
    /// Full task body handed to the assignee.
    pub content: String,
    /// Lifecycle status; default `pending`.
    pub status: TaskStatus,
    /// Member that owns the task, if it has been claimed or pre-assigned.
    pub assignee: Option<String>,
    /// Members that must vote when the author completes the task; empty means no review gate.
    pub reviewers: Vec<String>,
    /// Review round the task is currently in; `0` before the first verdict.
    pub review_round: u32,
    /// Per-task review-round ceiling; default `3`. Meaningful only with reviewers present.
    pub max_review_rounds: u32,
    /// Assignee's closing summary, when it reported one.
    pub summary: Option<String>,
    /// Free-form note attached to the task (last review verdict, blocking reason, …).
    pub note: Option<String>,
    /// Parent task for member subtasks; `None` for top-level tasks.
    pub parent_task_id: Option<String>,
    /// The parent's revision this subtask was derived from (`None` for top-level tasks).
    pub parent_revision: Option<u32>,
    /// Member (or leader) that created the task.
    pub created_by: Option<String>,
    /// Content revision, bumped on every edit; `0` for a freshly created task.
    pub revision: u32,
    /// Revision the current assignee was actually shown; a lag marks the "not yet seen" signal.
    pub observed_revision: u32,
    /// Rendered edit notes ([`TaskRevision::to_note`]), kept as audit history.
    pub change_notes: Vec<String>,
    /// Millisecond wall-clock timestamp of the last status transition; `0` when never stamped.
    pub updated_at: u64,
}

impl Default for TaskRecord {
    fn default() -> Self {
        Self {
            team_name: String::new(),
            task_id: String::new(),
            title: String::new(),
            content: String::new(),
            status: TaskStatus::default(),
            assignee: None,
            reviewers: Vec::new(),
            review_round: 0,
            max_review_rounds: DEFAULT_MAX_REVIEW_ROUNDS,
            summary: None,
            note: None,
            parent_task_id: None,
            parent_revision: None,
            created_by: None,
            revision: 0,
            observed_revision: 0,
            change_notes: Vec::new(),
            updated_at: 0,
        }
    }
}

impl TaskRecord {
    /// Creates a `pending`, unassigned task with the Python record defaults.
    #[must_use]
    pub fn new(
        team_name: impl Into<String>,
        task_id: impl Into<String>,
        title: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            team_name: team_name.into(),
            task_id: task_id.into(),
            title: title.into(),
            content: content.into(),
            ..Self::default()
        }
    }

    /// Returns `true` for the terminal statuses `completed` / `cancelled`.
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        self.status.is_terminal()
    }

    /// Appends a rendered edit note to [`TaskRecord::change_notes`].
    pub fn push_change_note(&mut self, note: impl Into<String>) {
        self.change_notes.push(note.into());
    }
}

/// Lightweight task summary returned by the list and claimable actions.
///
/// Python parity: `TaskSummary`. It carries the routing/identity fields plus
/// [`TaskSummary::updated_at`] (millisecond timestamp of the last status transition) so a list
/// view can render a relative age, and [`TaskSummary::assignee_status`] so a reader can tell a
/// stalled task with a working owner from one whose owner is gone.
///
/// [`TaskSummary::from_record`] fills the graph-derived fields (`blocked_by`, `assignee_status`,
/// `stale_against`) with their empty values; the task board sets them from the dependency graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct TaskSummary {
    /// Task identity.
    pub task_id: String,
    /// Short title.
    pub title: String,
    /// Lifecycle status.
    pub status: TaskStatus,
    /// Owner, if any.
    pub assignee: Option<String>,
    /// The owner's current member status, while the task is open and the owner is on the roster.
    pub assignee_status: Option<MemberStatus>,
    /// Ids of the tasks that must complete before this one unblocks.
    pub blocked_by: Vec<String>,
    /// Parent task id for subtasks.
    pub parent_task_id: Option<String>,
    /// Rendered `(parent revision now, revision it was based on)` marker when the subtask is stale.
    pub stale_against: Option<String>,
    /// Millisecond wall-clock timestamp of the last status transition.
    pub updated_at: u64,
}

impl TaskSummary {
    /// Projects a persisted record into its list view, leaving graph-derived fields empty.
    #[must_use]
    pub fn from_record(record: &TaskRecord) -> Self {
        Self {
            task_id: record.task_id.clone(),
            title: record.title.clone(),
            status: record.status,
            assignee: record.assignee.clone(),
            assignee_status: None,
            blocked_by: Vec::new(),
            parent_task_id: record.parent_task_id.clone(),
            stale_against: None,
            updated_at: record.updated_at,
        }
    }
}

/// Full task detail returned by the get action.
///
/// Python parity: `TaskDetail`. Rust carries `reviewers` (Python's `reviewer` column) and
/// `review_round` / `max_review_rounds` as pinned by this crate's contract.
///
/// [`TaskDetail::from_record`] fills the graph-derived fields (`assignee_status`, `blocked_by`,
/// `blocks`, `subtasks`) with their empty values.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct TaskDetail {
    /// Task identity.
    pub task_id: String,
    /// Short title.
    pub title: String,
    /// Full task body.
    pub content: String,
    /// Lifecycle status.
    pub status: TaskStatus,
    /// Owner, if any.
    pub assignee: Option<String>,
    /// The owner's current member status.
    pub assignee_status: Option<MemberStatus>,
    /// Members that vote on completion.
    pub reviewers: Vec<String>,
    /// Current review round.
    pub review_round: u32,
    /// Review-round ceiling.
    pub max_review_rounds: u32,
    /// Assignee's closing summary.
    pub summary: Option<String>,
    /// Free-form note.
    pub note: Option<String>,
    /// Ids of the tasks blocking this one.
    pub blocked_by: Vec<String>,
    /// Ids of the tasks this one blocks.
    pub blocks: Vec<String>,
    /// Parent task id for subtasks.
    pub parent_task_id: Option<String>,
    /// Ids of this task's subtasks.
    pub subtasks: Vec<String>,
    /// Content revision.
    pub revision: u32,
    /// Millisecond wall-clock timestamp of the last status transition.
    pub updated_at: u64,
}

impl TaskDetail {
    /// Projects a persisted record into its detail view, leaving graph-derived fields empty.
    #[must_use]
    pub fn from_record(record: &TaskRecord) -> Self {
        Self {
            task_id: record.task_id.clone(),
            title: record.title.clone(),
            content: record.content.clone(),
            status: record.status,
            assignee: record.assignee.clone(),
            assignee_status: None,
            reviewers: record.reviewers.clone(),
            review_round: record.review_round,
            max_review_rounds: record.max_review_rounds,
            summary: record.summary.clone(),
            note: record.note.clone(),
            blocked_by: Vec::new(),
            blocks: Vec::new(),
            parent_task_id: record.parent_task_id.clone(),
            subtasks: Vec::new(),
            revision: record.revision,
            updated_at: record.updated_at,
        }
    }
}

/// One task in a `TeamTaskManager.add_graph` batch.
///
/// Python parity: `TaskGraphSpec`. `depends_on` edges may reference tasks created in the same
/// batch (forward references) or already-existing tasks; `depended_by` edges must reference
/// already-existing tasks only, because an in-batch edge has exactly one representation.
///
/// `assignee` pre-assigns the task as part of the same atomic mutation. `reviewers` names the
/// verify-gate reviewers — a task carrying reviewers routes through `in_review` on completion,
/// an empty list completes directly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct TaskGraphSpec {
    /// Short title.
    pub title: String,
    /// Full task body.
    pub content: String,
    /// Caller-supplied task id; `None` lets the graph assign one.
    pub task_id: Option<String>,
    /// Ids this task depends on (in-batch forward references allowed).
    pub depends_on: Vec<String>,
    /// Ids that depend on this task (already-existing tasks only).
    pub depended_by: Vec<String>,
    /// Member to pre-assign the task to.
    pub assignee: Option<String>,
    /// Members that vote on completion; empty means no review gate.
    pub reviewers: Vec<String>,
    /// Per-task review-round ceiling; `None` uses the team default.
    pub max_review_rounds: Option<u32>,
    /// Parent task id for member subtasks; `None` for leader tasks.
    pub parent_task_id: Option<String>,
    /// Member (or leader) that created the task.
    pub created_by: Option<String>,
    /// The parent's revision this subtask was derived from.
    pub parent_revision: Option<u32>,
}

/// A recorded review vote.
///
/// Python parity: one `(round, reviewer, verdict)` entry of the review gate; the task board tallies
/// the votes of a round to decide whether a task passes (`scheduled` dispatch) or takes the first
/// verdict (`autonomous` dispatch).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct ReviewVote {
    /// Team the task belongs to.
    pub team_name: String,
    /// Task the vote applies to.
    pub task_id: String,
    /// Review round the vote was cast in.
    pub round: u32,
    /// Member that cast the vote.
    pub reviewer: String,
    /// The verdict; default `pass`.
    pub verdict: Verdict,
}

/// A structured edit note, the element of the task's revision history.
///
/// [`TaskRevision::to_note`] renders it into the string form stored in
/// [`TaskRecord::change_notes`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct TaskRevision {
    /// Revision the note describes.
    pub revision: u32,
    /// What changed.
    pub note: String,
    /// Member that pushed the revision, when known.
    pub by: Option<String>,
}

impl TaskRevision {
    /// Renders the note for [`TaskRecord::change_notes`]: `r{revision}: {note}`, with the author
    /// appended as ` ({by})` when known.
    #[must_use]
    pub fn to_note(&self) -> String {
        let Self { revision, note, by } = self;
        match by {
            Some(by) => format!("r{revision}: {note} ({by})"),
            None => format!("r{revision}: {note}"),
        }
    }
}

/// Filter applied by `TeamStore::list_tasks`.
///
/// Every field is optional; an unset field does not constrain the query and an unset
/// [`TaskFilter::limit`] returns every match.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct TaskFilter {
    /// Restrict to one status.
    pub status: Option<TaskStatus>,
    /// Restrict to one assignee.
    pub assignee: Option<String>,
    /// Restrict to the subtasks of one parent task.
    pub parent: Option<String>,
    /// Cap on the number of returned records.
    pub limit: Option<usize>,
}

/// A partial update of a [`TaskRecord`].
///
/// Every field is optional and `None` means "leave unchanged". An assignee cannot be expressed by
/// the `Option<String>` alone, so releasing an assignee takes the explicit
/// [`TaskPatch::clear_assignee`] flag; the two are mutually exclusive in practice — setting the
/// flag wins over [`TaskPatch::assignee`] because it is the deliberate action.
///
/// Builders consume and return the patch so a call site reads as one expression:
/// `TaskPatch::new().with_status(TaskStatus::InProgress).with_assignee("bob")`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct TaskPatch {
    /// New status.
    pub status: Option<TaskStatus>,
    /// New title.
    pub title: Option<String>,
    /// New content (bumps the revision when applied).
    pub content: Option<String>,
    /// New assignee; use [`TaskPatch::clear_assignee`] to release it instead.
    pub assignee: Option<String>,
    /// When `true`, the stored assignee is set to `null`.
    pub clear_assignee: bool,
    /// Replacement reviewer set.
    pub reviewers: Option<Vec<String>>,
    /// New review round.
    pub review_round: Option<u32>,
    /// New review-round ceiling.
    pub max_review_rounds: Option<u32>,
    /// New assignee summary.
    pub summary: Option<String>,
    /// New note.
    pub note: Option<String>,
    /// New parent task id.
    pub parent_task_id: Option<String>,
    /// New content revision.
    pub revision: Option<u32>,
    /// New observed revision.
    pub observed_revision: Option<u32>,
    /// New `updated_at` timestamp.
    pub updated_at: Option<u64>,
}

impl TaskPatch {
    /// Returns an empty patch: nothing is changed.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            status: None,
            title: None,
            content: None,
            assignee: None,
            clear_assignee: false,
            reviewers: None,
            review_round: None,
            max_review_rounds: None,
            summary: None,
            note: None,
            parent_task_id: None,
            revision: None,
            observed_revision: None,
            updated_at: None,
        }
    }

    /// Returns `true` when the patch sets nothing, so it can be skipped entirely.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.status.is_none()
            && self.title.is_none()
            && self.content.is_none()
            && self.assignee.is_none()
            && !self.clear_assignee
            && self.reviewers.is_none()
            && self.review_round.is_none()
            && self.max_review_rounds.is_none()
            && self.summary.is_none()
            && self.note.is_none()
            && self.parent_task_id.is_none()
            && self.revision.is_none()
            && self.observed_revision.is_none()
            && self.updated_at.is_none()
    }

    /// Sets the new status.
    #[must_use]
    pub const fn with_status(mut self, status: TaskStatus) -> Self {
        self.status = Some(status);
        self
    }

    /// Sets the new title.
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Sets the new content.
    #[must_use]
    pub fn with_content(mut self, content: impl Into<String>) -> Self {
        self.content = Some(content.into());
        self
    }

    /// Sets the new assignee.
    #[must_use]
    pub fn with_assignee(mut self, assignee: impl Into<String>) -> Self {
        self.assignee = Some(assignee.into());
        self
    }

    /// Releases the assignee (`clear_assignee = true`).
    #[must_use]
    pub const fn clearing_assignee(mut self) -> Self {
        self.clear_assignee = true;
        self
    }

    /// Replaces the reviewer set.
    #[must_use]
    pub fn with_reviewers(mut self, reviewers: Vec<String>) -> Self {
        self.reviewers = Some(reviewers);
        self
    }

    /// Sets the review round.
    #[must_use]
    pub const fn with_review_round(mut self, round: u32) -> Self {
        self.review_round = Some(round);
        self
    }

    /// Sets the review-round ceiling.
    #[must_use]
    pub const fn with_max_review_rounds(mut self, rounds: u32) -> Self {
        self.max_review_rounds = Some(rounds);
        self
    }

    /// Sets the assignee summary.
    #[must_use]
    pub fn with_summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = Some(summary.into());
        self
    }

    /// Sets the note.
    #[must_use]
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }

    /// Sets the parent task id.
    #[must_use]
    pub fn with_parent_task_id(mut self, parent_task_id: impl Into<String>) -> Self {
        self.parent_task_id = Some(parent_task_id.into());
        self
    }

    /// Sets the content revision.
    #[must_use]
    pub const fn with_revision(mut self, revision: u32) -> Self {
        self.revision = Some(revision);
        self
    }

    /// Sets the observed revision.
    #[must_use]
    pub const fn with_observed_revision(mut self, revision: u32) -> Self {
        self.observed_revision = Some(revision);
        self
    }

    /// Sets the `updated_at` timestamp.
    #[must_use]
    pub const fn with_updated_at(mut self, updated_at: u64) -> Self {
        self.updated_at = Some(updated_at);
        self
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use serde_json::json;

    use super::*;

    fn populated_record() -> TaskRecord {
        TaskRecord {
            team_name: "team".to_owned(),
            task_id: "task-1".to_owned(),
            title: "Port the task board".to_owned(),
            content: "Every record and builder".to_owned(),
            status: TaskStatus::InReview,
            assignee: Some("bob".to_owned()),
            reviewers: vec!["alice".to_owned(), "carol".to_owned()],
            review_round: 2,
            max_review_rounds: 4,
            summary: Some("done".to_owned()),
            note: Some("needs a second vote".to_owned()),
            parent_task_id: Some("task-0".to_owned()),
            parent_revision: Some(3),
            created_by: Some("team_leader".to_owned()),
            revision: 3,
            observed_revision: 2,
            change_notes: vec!["r1: first".to_owned()],
            updated_at: 1_700_000_000_000,
        }
    }

    #[test]
    fn verdict_wire_strings() {
        assert_eq!(Verdict::Pass.as_str(), "pass");
        assert_eq!(Verdict::Fail.as_str(), "fail");
        assert_eq!(serde_json::to_value(Verdict::Pass).unwrap(), json!("pass"));
        assert_eq!(serde_json::to_value(Verdict::Fail).unwrap(), json!("fail"));
        assert_eq!(
            serde_json::from_value::<Verdict>(json!("fail")).unwrap(),
            Verdict::Fail
        );
        assert_eq!(Verdict::default(), Verdict::Pass);
    }

    #[test]
    fn task_record_new_has_python_defaults() {
        let record = TaskRecord::new("team", "task-1", "Do it", "Details");
        assert_eq!(
            serde_json::to_value(&record).unwrap(),
            json!({
                "team_name": "team",
                "task_id": "task-1",
                "title": "Do it",
                "content": "Details",
                "status": "pending",
                "assignee": null,
                "reviewers": [],
                "review_round": 0,
                "max_review_rounds": 3,
                "summary": null,
                "note": null,
                "parent_task_id": null,
                "parent_revision": null,
                "created_by": null,
                "revision": 0,
                "observed_revision": 0,
                "change_notes": [],
                "updated_at": 0
            })
        );
        assert!(!record.is_terminal());
    }

    #[test]
    fn task_record_missing_fields_fall_back_to_defaults() {
        let record: TaskRecord =
            serde_json::from_value(json!({"task_id": "task-1", "title": "Do it"})).unwrap();
        assert_eq!(record.task_id, "task-1");
        assert_eq!(record.status, TaskStatus::Pending);
        assert_eq!(record.max_review_rounds, DEFAULT_MAX_REVIEW_ROUNDS);
        assert_eq!(record.review_round, 0);
        assert_eq!(record.reviewers, Vec::<String>::new());
        assert_eq!(record.change_notes, Vec::<String>::new());
        assert_eq!(record.updated_at, 0);
    }

    #[test]
    fn task_record_round_trips() {
        let record = populated_record();
        let encoded = serde_json::to_string(&record).unwrap();
        let decoded: TaskRecord = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, record);
    }

    #[test]
    fn task_record_tolerates_unknown_fields() {
        let decoded: TaskRecord = serde_json::from_value(json!({
            "team_name": "team",
            "task_id": "task-1",
            "title": "Do it",
            "content": "Details",
            "future_column": {"nested": [1, 2, 3]},
            "stale_subtasks": ["task-2"]
        }))
        .unwrap();
        assert_eq!(decoded.task_id, "task-1");
        assert_eq!(decoded.status, TaskStatus::Pending);
    }

    #[test]
    fn task_record_terminal_status_and_change_notes() {
        let mut record = TaskRecord::new("team", "task-1", "Do it", "Details");
        record.push_change_note("r1: first");
        record.push_change_note(String::from("r2: second"));
        assert_eq!(record.change_notes, ["r1: first", "r2: second"]);

        record.status = TaskStatus::Completed;
        assert!(record.is_terminal());
        record.status = TaskStatus::Cancelled;
        assert!(record.is_terminal());
        record.status = TaskStatus::InReview;
        assert!(!record.is_terminal());
    }

    #[test]
    fn task_summary_from_record_maps_fields() {
        let record = populated_record();
        let summary = TaskSummary::from_record(&record);
        assert_eq!(
            summary,
            TaskSummary {
                task_id: "task-1".to_owned(),
                title: "Port the task board".to_owned(),
                status: TaskStatus::InReview,
                assignee: Some("bob".to_owned()),
                assignee_status: None,
                blocked_by: Vec::new(),
                parent_task_id: Some("task-0".to_owned()),
                stale_against: None,
                updated_at: 1_700_000_000_000,
            }
        );
        let decoded: TaskSummary =
            serde_json::from_str(&serde_json::to_string(&summary).unwrap()).unwrap();
        assert_eq!(decoded, summary);
    }

    #[test]
    fn task_detail_from_record_maps_fields() {
        let record = populated_record();
        let detail = TaskDetail::from_record(&record);
        assert_eq!(
            detail,
            TaskDetail {
                task_id: "task-1".to_owned(),
                title: "Port the task board".to_owned(),
                content: "Every record and builder".to_owned(),
                status: TaskStatus::InReview,
                assignee: Some("bob".to_owned()),
                assignee_status: None,
                reviewers: vec!["alice".to_owned(), "carol".to_owned()],
                review_round: 2,
                max_review_rounds: 4,
                summary: Some("done".to_owned()),
                note: Some("needs a second vote".to_owned()),
                blocked_by: Vec::new(),
                blocks: Vec::new(),
                parent_task_id: Some("task-0".to_owned()),
                subtasks: Vec::new(),
                revision: 3,
                updated_at: 1_700_000_000_000,
            }
        );
        let decoded: TaskDetail =
            serde_json::from_str(&serde_json::to_string(&detail).unwrap()).unwrap();
        assert_eq!(decoded, detail);
    }

    #[test]
    fn task_graph_spec_round_trips_and_defaults_to_empty() {
        let empty = TaskGraphSpec::default();
        assert_eq!(empty.title, "");
        assert_eq!(empty.depends_on, Vec::<String>::new());
        assert_eq!(empty.reviewers, Vec::<String>::new());
        assert!(empty.task_id.is_none());
        assert!(empty.max_review_rounds.is_none());
        assert!(empty.parent_revision.is_none());

        let spec = TaskGraphSpec {
            title: "Child".to_owned(),
            content: "Body".to_owned(),
            task_id: Some("task-2".to_owned()),
            depends_on: vec!["task-1".to_owned()],
            depended_by: vec!["task-3".to_owned()],
            assignee: Some("bob".to_owned()),
            reviewers: vec!["alice".to_owned()],
            max_review_rounds: Some(5),
            parent_task_id: Some("task-1".to_owned()),
            created_by: Some("bob".to_owned()),
            parent_revision: Some(1),
        };
        let decoded: TaskGraphSpec =
            serde_json::from_str(&serde_json::to_string(&spec).unwrap()).unwrap();
        assert_eq!(decoded, spec);
    }

    #[test]
    fn review_vote_round_trips_and_defaults_verdict() {
        let vote = ReviewVote {
            team_name: "team".to_owned(),
            task_id: "task-1".to_owned(),
            round: 1,
            reviewer: "alice".to_owned(),
            verdict: Verdict::Fail,
        };
        let value = serde_json::to_value(&vote).unwrap();
        assert_eq!(value["verdict"], json!("fail"));
        assert_eq!(serde_json::from_value::<ReviewVote>(value).unwrap(), vote);

        let decoded: ReviewVote = serde_json::from_value(json!({"task_id": "task-1"})).unwrap();
        assert_eq!(decoded.round, 0);
        assert_eq!(decoded.verdict, Verdict::Pass);
        assert_eq!(decoded.reviewer, "");
    }

    #[test]
    fn task_revision_to_note_renders_both_shapes() {
        let anonymous = TaskRevision {
            revision: 3,
            note: "trimmed the intro".to_owned(),
            by: None,
        };
        assert_eq!(anonymous.to_note(), "r3: trimmed the intro");

        let attributed = TaskRevision {
            revision: 4,
            note: "added tests".to_owned(),
            by: Some("bob".to_owned()),
        };
        assert_eq!(attributed.to_note(), "r4: added tests (bob)");

        let decoded: TaskRevision =
            serde_json::from_str(&serde_json::to_string(&attributed).unwrap()).unwrap();
        assert_eq!(decoded, attributed);
        assert_eq!(TaskRevision::default().to_note(), "r0: ");
    }

    #[test]
    fn task_filter_accepts_a_subset_of_keys() {
        let filter: TaskFilter =
            serde_json::from_value(json!({"status": "in_progress", "assignee": "bob"})).unwrap();
        assert_eq!(filter.status, Some(TaskStatus::InProgress));
        assert_eq!(filter.assignee.as_deref(), Some("bob"));
        assert_eq!(filter.parent, None);
        assert_eq!(filter.limit, None);

        let empty: TaskFilter = serde_json::from_value(json!({})).unwrap();
        assert_eq!(empty, TaskFilter::default());
        assert_eq!(
            serde_json::from_value::<TaskFilter>(json!({"limit": 10}))
                .unwrap()
                .limit,
            Some(10)
        );
    }

    #[test]
    fn task_patch_is_empty_only_before_any_setter() {
        assert!(TaskPatch::new().is_empty());
        assert_eq!(TaskPatch::new(), TaskPatch::default());
        assert!(!TaskPatch::new()
            .with_status(TaskStatus::Completed)
            .is_empty());
        assert!(!TaskPatch::new().clearing_assignee().is_empty());
        assert!(!TaskPatch::new().with_updated_at(1).is_empty());
    }

    /// Asserts a builder touched exactly its own field and produced a non-empty patch.
    macro_rules! assert_single_change {
        ($actual:expr, $expected:expr $(,)?) => {{
            let actual = $actual;
            let expected = $expected;
            assert_eq!(actual, expected);
            assert!(!actual.is_empty());
        }};
    }

    #[test]
    fn task_patch_builders_touch_only_their_own_field() {
        assert_single_change!(
            TaskPatch::new().with_status(TaskStatus::InProgress),
            TaskPatch {
                status: Some(TaskStatus::InProgress),
                ..TaskPatch::default()
            },
        );
        assert_single_change!(
            TaskPatch::new().with_title("new title"),
            TaskPatch {
                title: Some("new title".to_owned()),
                ..TaskPatch::default()
            },
        );
        assert_single_change!(
            TaskPatch::new().with_content("new body"),
            TaskPatch {
                content: Some("new body".to_owned()),
                ..TaskPatch::default()
            },
        );
        assert_single_change!(
            TaskPatch::new().with_assignee("bob"),
            TaskPatch {
                assignee: Some("bob".to_owned()),
                ..TaskPatch::default()
            },
        );
        assert_single_change!(
            TaskPatch::new().clearing_assignee(),
            TaskPatch {
                clear_assignee: true,
                ..TaskPatch::default()
            },
        );
    }

    #[test]
    fn task_patch_review_and_bookkeeping_builders_touch_only_their_own_field() {
        assert_single_change!(
            TaskPatch::new().with_reviewers(vec!["alice".to_owned()]),
            TaskPatch {
                reviewers: Some(vec!["alice".to_owned()]),
                ..TaskPatch::default()
            },
        );
        assert_single_change!(
            TaskPatch::new().with_review_round(2),
            TaskPatch {
                review_round: Some(2),
                ..TaskPatch::default()
            },
        );
        assert_single_change!(
            TaskPatch::new().with_max_review_rounds(5),
            TaskPatch {
                max_review_rounds: Some(5),
                ..TaskPatch::default()
            },
        );
        assert_single_change!(
            TaskPatch::new().with_summary("done"),
            TaskPatch {
                summary: Some("done".to_owned()),
                ..TaskPatch::default()
            },
        );
        assert_single_change!(
            TaskPatch::new().with_note("stalled"),
            TaskPatch {
                note: Some("stalled".to_owned()),
                ..TaskPatch::default()
            },
        );
        assert_single_change!(
            TaskPatch::new().with_parent_task_id("task-0"),
            TaskPatch {
                parent_task_id: Some("task-0".to_owned()),
                ..TaskPatch::default()
            },
        );
        assert_single_change!(
            TaskPatch::new().with_revision(3),
            TaskPatch {
                revision: Some(3),
                ..TaskPatch::default()
            },
        );
        assert_single_change!(
            TaskPatch::new().with_observed_revision(2),
            TaskPatch {
                observed_revision: Some(2),
                ..TaskPatch::default()
            },
        );
        assert_single_change!(
            TaskPatch::new().with_updated_at(1_700_000_000_000),
            TaskPatch {
                updated_at: Some(1_700_000_000_000),
                ..TaskPatch::default()
            },
        );
    }

    #[test]
    fn task_patch_clearing_assignee_keeps_assignee_unset() {
        let patch = TaskPatch::new().clearing_assignee();
        assert_eq!(patch.assignee, None);
        assert!(patch.clear_assignee);
        let value = serde_json::to_value(&patch).unwrap();
        assert_eq!(value["clear_assignee"], json!(true));
        assert_eq!(value["assignee"], json!(null));
        assert_eq!(serde_json::from_value::<TaskPatch>(value).unwrap(), patch);
    }

    #[test]
    fn task_patch_deserializes_a_partial_payload_and_tolerates_unknown_fields() {
        let patch: TaskPatch = serde_json::from_value(json!({
            "status": "completed",
            "summary": "shipped",
            "unknown_field": 42
        }))
        .unwrap();
        assert_eq!(
            patch,
            TaskPatch {
                status: Some(TaskStatus::Completed),
                summary: Some("shipped".to_owned()),
                ..TaskPatch::default()
            }
        );
        assert!(!patch.is_empty());
    }
}
