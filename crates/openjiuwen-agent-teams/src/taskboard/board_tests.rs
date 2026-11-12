//! Board suite: creation and the dependency graph, claims and ownership, completion and the review
//! gate, the subtask tree and the read shapes.
//!
//! Every test drives a real [`TaskBoard`] over the in-memory store through the `dyn TeamStore` seam,
//! so the assertions are the ones a future backend has to keep as well. No sleeps: timestamps are
//! either caller-supplied or compared against an explicit clock.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use openjiuwen_common::error::Error;

use crate::schema::{
    EventPayload, MemberMode, TaskFilter, TaskGraphSpec, TaskPatch, TaskRecord, TaskStatus,
    TaskStatusChangedEvent, TeamRole, Verdict,
};
use crate::storage::{InMemoryTeamStore, MemberRecord, TeamStore};

use super::*;

/// Team name every test works under.
const TEAM: &str = "team";

// -----------------------------------------------------------------------------------------------
// Fixtures
// -----------------------------------------------------------------------------------------------

/// Builds an empty board whose subtask ceiling is `max_subtask_depth`.
fn board_with_depth(max_subtask_depth: usize) -> (Arc<dyn TeamStore>, TaskBoard) {
    let store: Arc<dyn TeamStore> = Arc::new(InMemoryTeamStore::new());
    let board = TaskBoard::new(
        store.clone(),
        TEAM,
        TaskBoardLimits::default().with_max_subtask_depth(max_subtask_depth),
    );
    (store, board)
}

/// Builds an empty board with the default limits (subtasks disabled).
fn fixture() -> (Arc<dyn TeamStore>, TaskBoard) {
    board_with_depth(0)
}

/// Owns a list of string literals, the shape the board's `Vec<String>` fields take.
fn owned(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

/// Builds a plain task spec whose id the caller supplies.
fn spec(task_id: &str) -> TaskGraphSpec {
    TaskGraphSpec {
        task_id: Some(task_id.to_owned()),
        title: format!("title {task_id}"),
        content: format!("body of {task_id}"),
        ..TaskGraphSpec::default()
    }
}

/// Builds a task spec that depends on `dependencies`.
fn spec_depending_on(task_id: &str, dependencies: &[&str]) -> TaskGraphSpec {
    TaskGraphSpec {
        depends_on: owned(dependencies),
        ..spec(task_id)
    }
}

/// Builds a task spec carrying `reviewers`.
fn spec_reviewed_by(task_id: &str, reviewers: &[&str]) -> TaskGraphSpec {
    TaskGraphSpec {
        reviewers: owned(reviewers),
        ..spec(task_id)
    }
}

/// Puts one roster row on the board so `assign` can resolve its mode.
async fn add_member(store: &Arc<dyn TeamStore>, name: &str, mode: MemberMode) {
    let record = MemberRecord {
        mode,
        ..MemberRecord::new(TEAM, name, TeamRole::Teammate)
    };
    store.insert_member(record).await.unwrap();
}

/// Puts the named `build_mode` members on the roster.
async fn add_builders(store: &Arc<dyn TeamStore>, names: &[&str]) {
    for name in names {
        add_member(store, name, MemberMode::BuildMode).await;
    }
}

/// Inserts a task row directly, bypassing the board (used by the stale sweep).
async fn insert_row(
    store: &Arc<dyn TeamStore>,
    task_id: &str,
    status: TaskStatus,
    assignee: Option<&str>,
    updated_at: u64,
) {
    let record = TaskRecord {
        status,
        assignee: assignee.map(str::to_owned),
        updated_at,
        ..TaskRecord::new(TEAM, task_id, task_id, "body")
    };
    store.insert_task(record).await.unwrap();
}

/// Reads one persisted row's status.
async fn status_of(store: &Arc<dyn TeamStore>, task_id: &str) -> TaskStatus {
    store.task(TEAM, task_id).await.unwrap().unwrap().status
}

/// Returns the wire type of every event of `mutation`, in publication order.
fn event_types(mutation: &TaskMutation) -> Vec<&str> {
    mutation
        .events
        .iter()
        .map(EventPayload::event_type)
        .collect()
}

/// Counts the events of one wire type.
fn count_events(mutation: &TaskMutation, event_type: &str) -> usize {
    mutation
        .events
        .iter()
        .filter(|event| event.event_type() == event_type)
        .count()
}

/// Returns the verdict of every `task_verified` event of `mutation`.
fn verified_verdicts(mutation: &TaskMutation) -> Vec<Verdict> {
    mutation
        .events
        .iter()
        .filter_map(|event| match event {
            EventPayload::TaskVerified(payload) => Some(payload.verdict),
            _ => None,
        })
        .collect()
}

/// Returns the first `task_status_changed` payload of `mutation`.
fn status_event(mutation: &TaskMutation) -> &TaskStatusChangedEvent {
    mutation
        .events
        .iter()
        .find_map(|event| match event {
            EventPayload::TaskStatusChanged(payload) => Some(payload),
            _ => None,
        })
        .expect("a task_status_changed event")
}

/// Returns the message of a validation error, panicking for any other error kind.
fn validation_message(err: &Error) -> &str {
    match err {
        Error::Validation(message) => message,
        other => panic!("expected a validation error, got {other:?}"),
    }
}

// -----------------------------------------------------------------------------------------------
// Creation
// -----------------------------------------------------------------------------------------------

#[tokio::test]
async fn creation_links_dependencies_and_emits_created_then_status() {
    let (store, board) = fixture();
    let mutation = board
        .create_task(spec_depending_on("downstream", &["left", "right"]))
        .await
        .unwrap();

    assert_eq!(mutation.task.task_id, "downstream");
    assert_eq!(mutation.task.status, TaskStatus::Blocked);
    assert_eq!(mutation.unblocked, Vec::<String>::new());
    assert_eq!(
        store.dependencies(TEAM, "downstream").await.unwrap(),
        owned(&["left", "right"])
    );
    assert_eq!(
        event_types(&mutation),
        ["task_created", "task_status_changed"]
    );
    assert_eq!(status_event(&mutation).status, TaskStatus::Blocked);
    assert_eq!(status_event(&mutation).assignee, None);
}

#[tokio::test]
async fn a_minted_id_is_task_prefixed_lowercase_hex() {
    let (_store, board) = fixture();
    let mutation = board
        .create_task(TaskGraphSpec {
            title: "minted".to_owned(),
            content: "no id supplied".to_owned(),
            ..TaskGraphSpec::default()
        })
        .await
        .unwrap();

    let hex = mutation
        .task
        .task_id
        .strip_prefix("task_")
        .expect("a minted id starts with task_");
    assert_eq!(hex.len(), 32, "id was {}", mutation.task.task_id);
    assert!(
        hex.chars().all(|ch| matches!(ch, '0'..='9' | 'a'..='f')),
        "id was {}",
        mutation.task.task_id
    );
    assert_eq!(mutation.task.status, TaskStatus::Pending);
}

#[tokio::test]
async fn creation_without_an_open_dependency_starts_pending() {
    let (_store, board) = fixture();
    let free = board.create_task(spec("free")).await.unwrap();
    assert_eq!(free.task.status, TaskStatus::Pending);

    board.create_task(spec("upstream")).await.unwrap();
    board.cancel("upstream").await.unwrap();
    let later = board
        .create_task(spec_depending_on("later", &["upstream"]))
        .await
        .unwrap();
    assert_eq!(later.task.status, TaskStatus::Pending);
    assert!(
        later.task.status.can_transition_to(TaskStatus::InProgress),
        "a task created past its dependencies is immediately claimable"
    );
}

#[tokio::test]
async fn a_cycle_is_rejected_before_anything_is_written() {
    let (store, board) = fixture();
    board
        .create_task(spec_depending_on("a", &["b"]))
        .await
        .unwrap();

    let err = board
        .create_task(spec_depending_on("b", &["a"]))
        .await
        .unwrap_err();
    let message = validation_message(&err);
    assert!(
        message.starts_with("Circular dependency: "),
        "message was {message}"
    );
    assert!(message.contains("a -> b -> a"), "message was {message}");

    let rows = board.view(&TaskFilter::default()).await.unwrap();
    assert_eq!(rows.len(), 1, "rows were {rows:?}");
    assert_eq!(rows[0].task_id, "a");
    assert!(store.task(TEAM, "b").await.unwrap().is_none());
    assert_eq!(
        store.dependencies(TEAM, "b").await.unwrap(),
        Vec::<String>::new()
    );
}

// -----------------------------------------------------------------------------------------------
// Unblocking
// -----------------------------------------------------------------------------------------------

#[tokio::test]
async fn completing_an_upstream_task_unblocks_its_dependents() {
    let (store, board) = fixture();
    board.create_task(spec("upstream")).await.unwrap();
    let downstream = board
        .create_task(spec_depending_on("downstream", &["upstream"]))
        .await
        .unwrap();
    assert_eq!(downstream.task.status, TaskStatus::Blocked);

    board.claim("upstream", "bob").await.unwrap();
    let mutation = board.complete("upstream", "bob", "done").await.unwrap();

    assert_eq!(mutation.task.status, TaskStatus::Completed);
    assert_eq!(mutation.unblocked, owned(&["downstream"]));
    assert_eq!(status_of(&store, "downstream").await, TaskStatus::Pending);
    let rows = board.view(&TaskFilter::default()).await.unwrap();
    let row = rows
        .iter()
        .find(|row| row.task_id == "downstream")
        .expect("the downstream row");
    assert!(row.blocked_by.is_empty(), "row was {row:?}");
}

#[tokio::test]
async fn cancelling_an_upstream_task_unblocks_its_dependents() {
    let (store, board) = fixture();
    board.create_task(spec("upstream")).await.unwrap();
    board
        .create_task(spec_depending_on("downstream", &["upstream"]))
        .await
        .unwrap();

    let mutation = board.cancel("upstream").await.unwrap();

    assert_eq!(mutation.task.status, TaskStatus::Cancelled);
    assert_eq!(mutation.unblocked, owned(&["downstream"]));
    assert_eq!(status_of(&store, "downstream").await, TaskStatus::Pending);
}

// -----------------------------------------------------------------------------------------------
// Claims
// -----------------------------------------------------------------------------------------------

#[tokio::test]
async fn claim_moves_a_pending_task_to_in_progress_and_refuses_other_owners() {
    let (store, board) = fixture();
    board.create_task(spec("task")).await.unwrap();

    let claimed = board.claim("task", "bob").await.unwrap();
    assert!(claimed.is_ok(), "claim was {claimed:?}");
    let row = store.task(TEAM, "task").await.unwrap().unwrap();
    assert_eq!(row.status, TaskStatus::InProgress);
    assert_eq!(row.assignee.as_deref(), Some("bob"));

    let refused = board.claim("task", "alice").await.unwrap();
    assert!(!refused.is_ok());
    assert_eq!(
        refused.reason(),
        Some("task 'task' is already claimed by 'bob'")
    );

    let again = board.claim("task", "bob").await.unwrap();
    assert!(again.is_ok());
    assert_eq!(again.reason(), None);
    assert_eq!(status_of(&store, "task").await, TaskStatus::InProgress);
}

#[tokio::test]
async fn claim_refuses_a_task_that_is_not_pending() {
    let (store, board) = fixture();
    board
        .create_task(spec_depending_on("blocked", &["missing"]))
        .await
        .unwrap();

    let refused = board.claim("blocked", "bob").await.unwrap();

    assert!(!refused.is_ok());
    assert_eq!(
        refused.reason(),
        Some("task 'blocked' cannot be claimed from status 'blocked' (only pending tasks are claimable)")
    );
    let row = store.task(TEAM, "blocked").await.unwrap().unwrap();
    assert_eq!(row.status, TaskStatus::Blocked);
    assert_eq!(row.assignee, None);
}

// -----------------------------------------------------------------------------------------------
// Completion
// -----------------------------------------------------------------------------------------------

#[tokio::test]
async fn completing_without_reviewers_closes_the_task() {
    let (store, board) = fixture();
    board.create_task(spec("task")).await.unwrap();
    board.claim("task", "bob").await.unwrap();

    let mutation = board.complete("task", "bob", "shipped").await.unwrap();

    assert_eq!(mutation.task.status, TaskStatus::Completed);
    assert_eq!(mutation.task.summary.as_deref(), Some("shipped"));
    assert_eq!(event_types(&mutation), ["task_status_changed"]);
    assert_eq!(count_events(&mutation, "task_status_changed"), 1);
    assert_eq!(mutation.unblocked, Vec::<String>::new());
    let stored = store.task(TEAM, "task").await.unwrap().unwrap();
    assert_eq!(stored.summary.as_deref(), Some("shipped"));
}

#[tokio::test]
async fn completing_someone_elses_task_is_a_validation_error() {
    let (_store, board) = fixture();
    board.create_task(spec("task")).await.unwrap();
    board.claim("task", "bob").await.unwrap();

    let err = board.complete("task", "alice", "").await.unwrap_err();

    assert_eq!(validation_message(&err), "task is assigned to 'bob'");
}

#[tokio::test]
async fn completing_with_a_reviewer_enters_review_without_unblocking() {
    let (store, board) = fixture();
    board
        .create_task(spec_reviewed_by("upstream", &["alice"]))
        .await
        .unwrap();
    board
        .create_task(spec_depending_on("downstream", &["upstream"]))
        .await
        .unwrap();
    board.claim("upstream", "bob").await.unwrap();

    let mutation = board
        .complete("upstream", "bob", "please check")
        .await
        .unwrap();

    assert_eq!(mutation.task.status, TaskStatus::InReview);
    assert_eq!(mutation.task.review_round, 1);
    assert_eq!(mutation.task.reviewers, owned(&["alice"]));
    assert_eq!(mutation.unblocked, Vec::<String>::new());
    assert_eq!(status_of(&store, "downstream").await, TaskStatus::Blocked);
    assert_eq!(event_types(&mutation), ["task_status_changed"]);
}

// -----------------------------------------------------------------------------------------------
// Autonomous review (`verify`)
// -----------------------------------------------------------------------------------------------

/// Builds the reviewed task plus its dependent, already `in_review`.
async fn reviewed_task(board: &TaskBoard, reviewers: &[&str]) -> TaskMutation {
    board
        .create_task(spec_reviewed_by("reviewed", reviewers))
        .await
        .unwrap();
    board
        .create_task(spec_depending_on("downstream", &["reviewed"]))
        .await
        .unwrap();
    board.claim("reviewed", "author").await.unwrap();
    board.complete("reviewed", "author", "").await.unwrap()
}

#[tokio::test]
async fn a_passing_verdict_completes_the_task_and_unblocks_dependents() {
    let (store, board) = fixture();
    reviewed_task(&board, &["alice"]).await;

    let mutation = board
        .verify("reviewed", "alice", Verdict::Pass, "looks good")
        .await
        .unwrap();

    assert_eq!(mutation.task.status, TaskStatus::Completed);
    assert_eq!(mutation.unblocked, owned(&["downstream"]));
    assert_eq!(status_of(&store, "downstream").await, TaskStatus::Pending);
    assert_eq!(count_events(&mutation, "task_verified"), 1);
    assert_eq!(count_events(&mutation, "task_review_vote"), 1);
    assert_eq!(count_events(&mutation, "task_status_changed"), 1);
    assert_eq!(verified_verdicts(&mutation), [Verdict::Pass]);
}

#[tokio::test]
async fn a_failing_verdict_returns_the_task_to_progress_and_releases_the_author() {
    let (store, board) = fixture();
    reviewed_task(&board, &["alice"]).await;

    let mutation = board
        .verify("reviewed", "alice", Verdict::Fail, "add tests")
        .await
        .unwrap();

    assert_eq!(mutation.task.status, TaskStatus::InProgress);
    assert_eq!(mutation.task.assignee, None);
    assert_eq!(mutation.task.note.as_deref(), Some("add tests"));
    assert_eq!(mutation.unblocked, Vec::<String>::new());
    assert_eq!(count_events(&mutation, "task_verified"), 1);
    assert_eq!(verified_verdicts(&mutation), [Verdict::Fail]);
    assert_eq!(status_of(&store, "downstream").await, TaskStatus::Blocked);
}

#[tokio::test]
async fn verify_refuses_a_task_that_is_not_in_review_or_an_unknown_reviewer() {
    let (_store, board) = fixture();
    board
        .create_task(spec_reviewed_by("reviewed", &["alice"]))
        .await
        .unwrap();

    let err = board
        .verify("reviewed", "alice", Verdict::Pass, "")
        .await
        .unwrap_err();
    assert!(
        validation_message(&err).contains("is not under review"),
        "message was {}",
        validation_message(&err)
    );

    board.claim("reviewed", "author").await.unwrap();
    board.complete("reviewed", "author", "").await.unwrap();
    let err = board
        .verify("reviewed", "mallory", Verdict::Pass, "")
        .await
        .unwrap_err();
    assert!(
        validation_message(&err).contains("is not a reviewer"),
        "message was {}",
        validation_message(&err)
    );
}

// -----------------------------------------------------------------------------------------------
// Scheduled review (vote tally + settle)
// -----------------------------------------------------------------------------------------------

#[tokio::test]
async fn a_scheduled_round_needs_every_reviewer_before_it_settles() {
    let (store, board) = fixture();
    reviewed_task(&board, &["alice", "bob", "carol"]).await;

    let vote = board
        .record_vote("reviewed", "alice", Verdict::Pass)
        .await
        .unwrap();
    assert_eq!(vote.task.status, TaskStatus::InReview);
    assert_eq!(count_events(&vote, "task_verified"), 0);
    assert_eq!(count_events(&vote, "task_review_vote"), 1);

    let tally = board.review_tally("reviewed").await.unwrap();
    assert_eq!(tally.pass_count, 1);
    assert_eq!(tally.fail_count, 0);
    assert_eq!(tally.reviewer_count, 3);
    assert_eq!(tally.voted(), 1);
    assert!(!tally.passed);
    assert!(!tally.decided());

    let err = board.settle_review("reviewed").await.unwrap_err();
    assert!(
        validation_message(&err).contains("is not decided yet"),
        "message was {}",
        validation_message(&err)
    );
    assert_eq!(status_of(&store, "reviewed").await, TaskStatus::InReview);

    board
        .record_vote("reviewed", "bob", Verdict::Pass)
        .await
        .unwrap();
    board
        .record_vote("reviewed", "carol", Verdict::Pass)
        .await
        .unwrap();
    let settled = board.settle_review("reviewed").await.unwrap();

    assert_eq!(settled.task.status, TaskStatus::Completed);
    assert_eq!(settled.unblocked, owned(&["downstream"]));
    assert_eq!(count_events(&settled, "task_verified"), 1);
    assert_eq!(verified_verdicts(&settled), [Verdict::Pass]);
}

#[tokio::test]
async fn a_failing_round_returns_the_task_to_progress_with_the_author_assigned() {
    let (store, board) = fixture();
    reviewed_task(&board, &["alice", "bob", "carol"]).await;
    board
        .record_vote("reviewed", "alice", Verdict::Fail)
        .await
        .unwrap();
    board
        .record_vote("reviewed", "bob", Verdict::Fail)
        .await
        .unwrap();
    board
        .record_vote("reviewed", "carol", Verdict::Pass)
        .await
        .unwrap();

    let settled = board.settle_review("reviewed").await.unwrap();

    assert_eq!(settled.task.status, TaskStatus::InProgress);
    assert_eq!(settled.task.assignee.as_deref(), Some("author"));
    assert_eq!(verified_verdicts(&settled), [Verdict::Fail]);
    assert_eq!(settled.unblocked, Vec::<String>::new());
    let note = settled.task.note.expect("a failed-round note");
    assert!(note.contains("review round 1 failed"), "note was {note}");
    assert_eq!(status_of(&store, "downstream").await, TaskStatus::Blocked);
}

// -----------------------------------------------------------------------------------------------
// Revision gate
// -----------------------------------------------------------------------------------------------

#[tokio::test]
async fn editing_the_content_mid_flight_forces_a_re_read_before_completion() {
    let (store, board) = fixture();
    board.create_task(spec("task")).await.unwrap();
    board.claim("task", "bob").await.unwrap();

    assert!(store
        .update_task(TEAM, "task", &TaskPatch::new().with_content("new body"))
        .await
        .unwrap());
    let edited = store.task(TEAM, "task").await.unwrap().unwrap();
    assert_eq!(edited.revision, 1);

    let err = board.complete("task", "bob", "").await.unwrap_err();
    assert_eq!(
        validation_message(&err),
        "content changed; re-read the task before completing"
    );
    assert_eq!(validation_message(&err), REASON_STALE_CONTENT);

    assert!(board.claim("task", "bob").await.unwrap().is_ok());
    let mutation = board.complete("task", "bob", "done").await.unwrap();
    assert_eq!(mutation.task.status, TaskStatus::Completed);
}

// -----------------------------------------------------------------------------------------------
// Ownership: assign, reassign, set_plan, reset
// -----------------------------------------------------------------------------------------------

#[tokio::test]
async fn assign_moves_a_pending_task_into_the_members_entry_status() {
    let (store, board) = fixture();
    add_builders(&store, &["bob"]).await;
    add_member(&store, "carol", MemberMode::PlanMode).await;
    board.create_task(spec("built")).await.unwrap();
    board.create_task(spec("planned")).await.unwrap();

    let built = board.assign("built", "bob", Some("take it")).await.unwrap();
    assert!(built.is_ok(), "assign was {built:?}");
    let row = store.task(TEAM, "built").await.unwrap().unwrap();
    assert_eq!(row.status, TaskStatus::InProgress);
    assert_eq!(row.assignee.as_deref(), Some("bob"));
    assert_eq!(row.change_notes, owned(&["r0: assigned to bob: take it"]));

    let planned = board.assign("planned", "carol", None).await.unwrap();
    assert!(planned.is_ok(), "assign was {planned:?}");
    assert_eq!(status_of(&store, "planned").await, TaskStatus::Planning);
}

#[tokio::test]
async fn assign_refuses_an_unknown_member_and_an_already_claimed_task() {
    let (store, board) = fixture();
    add_builders(&store, &["bob", "alice"]).await;
    board.create_task(spec("task")).await.unwrap();

    let unknown = board.assign("task", "ghost", None).await.unwrap();
    assert!(!unknown.is_ok());
    assert_eq!(
        unknown.reason(),
        Some("member 'ghost' not found in team 'team'")
    );

    board.claim("task", "bob").await.unwrap();
    let claimed = board.assign("task", "alice", None).await.unwrap();
    assert!(!claimed.is_ok());
    assert_eq!(
        claimed.reason(),
        Some(
            "task 'task' is already claimed by 'bob'; reset the task before reassigning to 'alice'"
        )
    );
    assert_eq!(
        store
            .task(TEAM, "task")
            .await
            .unwrap()
            .unwrap()
            .assignee
            .as_deref(),
        Some("bob")
    );
}

#[tokio::test]
async fn reassign_swaps_the_owner_in_place() {
    let (store, board) = fixture();
    add_builders(&store, &["bob", "alice"]).await;
    board.create_task(spec("task")).await.unwrap();
    board.claim("task", "bob").await.unwrap();

    let moved = board.reassign("task", "bob", "alice").await.unwrap();
    assert!(moved.is_ok(), "reassign was {moved:?}");
    let row = store.task(TEAM, "task").await.unwrap().unwrap();
    assert_eq!(row.assignee.as_deref(), Some("alice"));
    assert_eq!(row.status, TaskStatus::InProgress);
    assert_eq!(
        row.change_notes,
        owned(&["r0: reassigned from bob to alice (bob)"])
    );

    let refused = board.reassign("task", "bob", "alice").await.unwrap();
    assert!(!refused.is_ok());
    assert_eq!(
        refused.reason(),
        Some("task 'task' is not assigned to 'bob'")
    );
}

#[tokio::test]
async fn set_plan_records_the_plan_and_moves_into_planning() {
    let (store, board) = fixture();
    board.create_task(spec("task")).await.unwrap();
    board.claim("task", "bob").await.unwrap();

    let refused = board.set_plan("task", "alice", "step 1").await.unwrap();
    assert!(!refused.is_ok());
    assert_eq!(refused.reason(), Some("task is assigned to 'bob'"));

    let planned = board.set_plan("task", "bob", "1. port it").await.unwrap();
    assert!(planned.is_ok(), "set_plan was {planned:?}");
    let row = store.task(TEAM, "task").await.unwrap().unwrap();
    assert_eq!(row.status, TaskStatus::Planning);
    assert_eq!(row.note.as_deref(), Some("1. port it"));
}

#[tokio::test]
async fn reset_releases_a_closed_or_reviewed_task_back_to_pending() {
    let (store, board) = fixture();
    board.create_task(spec("closed")).await.unwrap();
    board.claim("closed", "bob").await.unwrap();
    board.complete("closed", "bob", "shipped").await.unwrap();

    let reset = board.reset("closed").await.unwrap();
    assert_eq!(reset.task.status, TaskStatus::Pending);
    assert_eq!(reset.task.assignee, None);
    assert_eq!(reset.task.summary, None);
    assert_eq!(event_types(&reset), ["task_status_changed"]);
    assert_eq!(status_of(&store, "closed").await, TaskStatus::Pending);

    reviewed_task(&board, &["alice"]).await;
    let reset = board.reset("reviewed").await.unwrap();
    assert_eq!(reset.task.status, TaskStatus::Pending);
    assert_eq!(reset.task.assignee, None);

    let err = board.reset("reviewed").await.unwrap_err();
    assert!(
        validation_message(&err).contains("cannot be reset"),
        "message was {}",
        validation_message(&err)
    );
}

#[tokio::test]
async fn cancel_refuses_an_already_terminal_task() {
    let (store, board) = fixture();
    board.create_task(spec("task")).await.unwrap();
    board.claim("task", "bob").await.unwrap();
    board.complete("task", "bob", "done").await.unwrap();

    let err = board.cancel("task").await.unwrap_err();

    assert!(
        validation_message(&err).contains("already terminal"),
        "message was {}",
        validation_message(&err)
    );
    assert_eq!(status_of(&store, "task").await, TaskStatus::Completed);
}

// -----------------------------------------------------------------------------------------------
// Subtasks
// -----------------------------------------------------------------------------------------------

#[tokio::test]
async fn subtask_depth_zero_refuses_a_first_level_subtask() {
    let (store, board) = fixture();
    board.create_task(spec("top")).await.unwrap();

    let err = board
        .create_subtask("top", spec("child"))
        .await
        .unwrap_err();

    assert_eq!(
        validation_message(&err),
        "subtask depth 1 exceeds max_subtask_depth 0"
    );
    assert!(store.task(TEAM, "child").await.unwrap().is_none());
    assert_eq!(status_of(&store, "top").await, TaskStatus::Pending);
}

#[tokio::test]
async fn a_subtask_parks_its_parent_and_a_nested_one_is_refused() {
    let (store, board) = board_with_depth(1);
    board.create_task(spec("top")).await.unwrap();

    let created = board.create_subtask("top", spec("child")).await.unwrap();
    assert_eq!(created.task.status, TaskStatus::Pending);
    assert_eq!(created.task.parent_task_id.as_deref(), Some("top"));
    assert_eq!(created.task.parent_revision, Some(0));
    assert_eq!(
        created.unblocked,
        Vec::<String>::new(),
        "creating a subtask wakes nobody"
    );
    assert_eq!(count_events(&created, "task_status_changed"), 2);

    let parent = store.task(TEAM, "top").await.unwrap().unwrap();
    assert_eq!(parent.status, TaskStatus::Blocked);
    assert_eq!(
        store.dependencies(TEAM, "top").await.unwrap(),
        owned(&["child"])
    );

    let err = board
        .create_subtask("child", spec("grandchild"))
        .await
        .unwrap_err();
    assert_eq!(
        validation_message(&err),
        "subtask depth 2 exceeds max_subtask_depth 1"
    );
    assert!(store.task(TEAM, "grandchild").await.unwrap().is_none());
}

// -----------------------------------------------------------------------------------------------
// Board-level reads
// -----------------------------------------------------------------------------------------------

#[tokio::test]
async fn settled_and_completed_track_the_open_tasks() {
    let (_store, board) = fixture();
    assert!(board.is_settled().await.unwrap());
    assert!(!board.is_completed().await.unwrap());

    board.create_task(spec("task")).await.unwrap();
    assert!(!board.is_settled().await.unwrap());
    assert!(!board.is_completed().await.unwrap());

    board.claim("task", "bob").await.unwrap();
    board.complete("task", "bob", "done").await.unwrap();
    assert!(board.is_settled().await.unwrap());
    assert!(board.is_completed().await.unwrap());
}

#[tokio::test]
async fn stale_sweep_uses_inclusive_boundaries_and_skips_parked_rows() {
    const NOW: u64 = 1_700_000_000_000;
    const CLAIM_IDLE: u64 = 600_000;
    const PENDING_IDLE: u64 = 300_000;
    let (store, board) = fixture();

    insert_row(
        &store,
        "working-exact",
        TaskStatus::InProgress,
        Some("bob"),
        NOW - CLAIM_IDLE,
    )
    .await;
    insert_row(
        &store,
        "working-fresh",
        TaskStatus::InProgress,
        Some("bob"),
        NOW - CLAIM_IDLE + 1,
    )
    .await;
    insert_row(
        &store,
        "pending-exact",
        TaskStatus::Pending,
        None,
        NOW - PENDING_IDLE,
    )
    .await;
    insert_row(
        &store,
        "pending-fresh",
        TaskStatus::Pending,
        None,
        NOW - PENDING_IDLE + 1,
    )
    .await;
    insert_row(
        &store,
        "pending-owned",
        TaskStatus::Pending,
        Some("bob"),
        NOW - PENDING_IDLE,
    )
    .await;
    for (task_id, status, assignee) in [
        ("blocked-old", TaskStatus::Blocked, None),
        ("reviewing-old", TaskStatus::InReview, Some("bob")),
        ("completed-old", TaskStatus::Completed, Some("bob")),
        ("cancelled-old", TaskStatus::Cancelled, Some("bob")),
    ] {
        insert_row(&store, task_id, status, assignee, 1).await;
    }

    let stale: Vec<String> = board
        .stale_tasks(NOW, CLAIM_IDLE, PENDING_IDLE)
        .await
        .unwrap()
        .into_iter()
        .map(|summary| summary.task_id)
        .collect();

    assert_eq!(stale, owned(&["pending-exact", "working-exact"]));
}

#[tokio::test]
async fn view_fills_blocked_by_and_claimable_lists_only_free_pending_tasks() {
    let (_store, board) = fixture();
    board.create_task(spec("upstream")).await.unwrap();
    board
        .create_task(spec_depending_on("downstream", &["upstream"]))
        .await
        .unwrap();
    board.create_task(spec("free")).await.unwrap();
    board.create_task(spec("taken")).await.unwrap();
    board.claim("taken", "bob").await.unwrap();

    let rows = board.view(&TaskFilter::default()).await.unwrap();
    let downstream = rows
        .iter()
        .find(|row| row.task_id == "downstream")
        .expect("the downstream row");
    assert_eq!(downstream.blocked_by, owned(&["upstream"]));
    assert_eq!(
        status_of(&board.store().clone(), "taken").await,
        TaskStatus::InProgress
    );

    let mut claimable: Vec<String> = board
        .claimable()
        .await
        .unwrap()
        .into_iter()
        .map(|summary| summary.task_id)
        .collect();
    claimable.sort();
    assert_eq!(claimable, owned(&["free", "upstream"]));
}

#[tokio::test]
async fn detail_fills_the_graph_fields_and_subtasks_lists_direct_children() {
    let (_store, board) = fixture();
    board.create_task(spec("top")).await.unwrap();
    board
        .create_task(spec_depending_on("downstream", &["top"]))
        .await
        .unwrap();
    for (task_id, parent) in [("child", "top"), ("grandchild", "child")] {
        let mut child = spec(task_id);
        child.parent_task_id = Some(parent.to_owned());
        board.create_task(child).await.unwrap();
    }

    let top = board.detail("top").await.unwrap().expect("the top detail");
    assert_eq!(top.blocks, owned(&["downstream"]));
    assert_eq!(top.subtasks, owned(&["child"]));
    assert!(top.blocked_by.is_empty(), "top was {top:?}");

    let downstream = board
        .detail("downstream")
        .await
        .unwrap()
        .expect("the downstream detail");
    assert_eq!(downstream.blocked_by, owned(&["top"]));
    assert!(
        downstream.blocks.is_empty(),
        "downstream was {downstream:?}"
    );

    let child = board
        .detail("child")
        .await
        .unwrap()
        .expect("the child detail");
    assert_eq!(child.subtasks, owned(&["grandchild"]));
    assert_eq!(child.parent_task_id.as_deref(), Some("top"));

    let subtasks: Vec<String> = board
        .subtasks("top")
        .await
        .unwrap()
        .into_iter()
        .map(|summary| summary.task_id)
        .collect();
    assert_eq!(subtasks, owned(&["child"]));
    assert!(board.detail("ghost").await.unwrap().is_none());
}
