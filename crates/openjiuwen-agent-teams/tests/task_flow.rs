//! Task-board flow: the dependency graph, ownership, the review gate, subtasks and cancel/reset,
//! driven through the public API over an in-memory store.
//!
//! The suite pins the observable contract a caller of `openjiuwen_agent_teams::taskboard` relies on:
//!
//! 1. dependency edges: an unresolved dependency parks a task at `blocked`, a terminal dependency
//!    promotes it back to `pending` and the promotion is reported through
//!    [`TaskMutation::unblocked`] (never as an event, so the runtime owns the wake-up);
//! 2. rejected graphs and rejected ids: cycles, duplicate ids, unknown `depended_by` targets and
//!    blank dependency ids are `Error::Validation` with the crate's stable reason text, and a
//!    forward reference parks the task instead of failing;
//! 3. ownership: `claim`/`assign`/`reassign` follow the status rules and report refusals as
//!    `TaskOpResult` with the stable reason strings;
//! 4. completion without reviewers closes the task directly; with reviewers it enters `in_review`
//!    and only a settled review reaches `completed`;
//! 5. the autonomous gate (`verify`) and the scheduled gate (`record_vote` / `review_tally` /
//!    `settle_review`) reach the same two outcomes, with the rounds monotonic per task;
//! 6. subtasks park their parent and honour `max_subtask_depth`;
//! 7. `cancel` is terminal and unblocks dependents, `reset` returns a closed task to `pending`
//!    without rewinding the review round, and an unknown task id is `NotFound` (or an empty read
//!    for `detail`).
//!
//! Everything is offline and deterministic: the only I/O is `InMemoryTeamStore`, and no test
//! sleeps — the board has no clock of its own, so there is nothing to wait for.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;

use openjiuwen_agent_teams::schema::{
    EventPayload, MemberMode, TaskFilter, TaskGraphSpec, TaskStatus, TeamRole, Verdict,
};
use openjiuwen_agent_teams::storage::{InMemoryTeamStore, MemberRecord, TeamStore};
use openjiuwen_agent_teams::taskboard::{ReviewTally, TaskBoard, TaskBoardLimits, TaskMutation};
use openjiuwen_common::error::{Error, ErrorKind};

/// Team every test works under.
const TEAM: &str = "alpha";

// -----------------------------------------------------------------------------------------------
// Fixtures
// -----------------------------------------------------------------------------------------------

/// Builds an empty board over a fresh in-memory store, plus the store for direct assertions.
fn board_with(limits: TaskBoardLimits) -> (Arc<dyn TeamStore>, TaskBoard) {
    let store: Arc<dyn TeamStore> = Arc::new(InMemoryTeamStore::new());
    let board = TaskBoard::new(store.clone(), TEAM, limits);
    (store, board)
}

/// Builds an empty board with the Python defaults (`3` review rounds, subtasks disabled).
fn board() -> (Arc<dyn TeamStore>, TaskBoard) {
    board_with(TaskBoardLimits::new())
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

/// Puts one roster row on the board so `assign` can resolve the member's mode.
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

/// Returns the message of a validation error, panicking for any other error kind.
fn validation_message(err: &Error) -> &str {
    assert_eq!(
        err.kind(),
        ErrorKind::Validation,
        "expected a validation error, got {err:?}"
    );
    match err {
        Error::Validation(message) => message,
        other => panic!("expected a validation error, got {other:?}"),
    }
}

/// Returns the message of a not-found error, panicking for any other error kind.
fn not_found_message(err: &Error) -> &str {
    assert_eq!(
        err.kind(),
        ErrorKind::NotFound,
        "expected a not-found error, got {err:?}"
    );
    match err {
        Error::NotFound(message) => message,
        other => panic!("expected a not-found error, got {other:?}"),
    }
}

/// Builds the reviewed task `reviewed` plus its dependent `downstream`, already `in_review` with
/// `author` as the submitter.
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

// -----------------------------------------------------------------------------------------------
// 1. Dependency graph
// -----------------------------------------------------------------------------------------------

#[tokio::test]
async fn a_dependency_parks_the_task_until_its_upstream_is_done() {
    let (store, board) = board();

    let upstream = board.create_task(spec("t1")).await.unwrap();
    assert_eq!(upstream.task.status, TaskStatus::Pending);
    assert_eq!(
        event_types(&upstream),
        ["task_created", "task_status_changed"]
    );

    let downstream = board
        .create_task(spec_depending_on("t2", &["t1"]))
        .await
        .unwrap();
    assert_eq!(downstream.task.status, TaskStatus::Blocked);
    assert_eq!(downstream.unblocked, Vec::<String>::new());
    assert_eq!(
        store.dependencies(TEAM, "t2").await.unwrap(),
        owned(&["t1"])
    );

    let blocked_filter = TaskFilter {
        status: Some(TaskStatus::Blocked),
        ..TaskFilter::default()
    };
    let blocked: Vec<String> = board
        .view(&blocked_filter)
        .await
        .unwrap()
        .into_iter()
        .map(|summary| summary.task_id)
        .collect();
    assert_eq!(blocked, owned(&["t2"]));

    assert!(!board.is_settled().await.unwrap());
}

#[tokio::test]
async fn completing_an_upstream_task_promotes_its_dependent() {
    let (store, board) = board();
    board.create_task(spec("t1")).await.unwrap();
    board
        .create_task(spec_depending_on("t2", &["t1"]))
        .await
        .unwrap();
    board.claim("t1", "bob").await.unwrap();

    let mutation = board.complete("t1", "bob", "done").await.unwrap();

    // The promotion is a side effect of the completion, so it is reported through `unblocked` and
    // the stored row — the block deliberately emits no extra status event for it.
    assert_eq!(mutation.task.status, TaskStatus::Completed);
    assert_eq!(mutation.unblocked, owned(&["t2"]));
    assert_eq!(event_types(&mutation), ["task_status_changed"]);
    assert_eq!(status_of(&store, "t2").await, TaskStatus::Pending);

    let detail = board.detail("t2").await.unwrap().expect("the t2 detail");
    assert_eq!(detail.status, TaskStatus::Pending);
    assert!(detail.blocked_by.is_empty(), "t2 was {detail:?}");
}

#[tokio::test]
async fn settled_and_completed_track_the_open_tasks() {
    let (_store, board) = board();

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

// -----------------------------------------------------------------------------------------------
// 2. Rejected graphs and rejected ids
// -----------------------------------------------------------------------------------------------

#[tokio::test]
async fn a_self_dependency_is_rejected_as_a_cycle() {
    let (store, board) = board();

    let err = board
        .create_task(spec_depending_on("t1", &["t1"]))
        .await
        .unwrap_err();

    assert_eq!(validation_message(&err), "Circular dependency: t1 -> t1");
    assert!(store.task(TEAM, "t1").await.unwrap().is_none());
    assert_eq!(
        store.dependencies(TEAM, "t1").await.unwrap(),
        Vec::<String>::new()
    );
    assert_eq!(
        board.view(&TaskFilter::default()).await.unwrap(),
        Vec::new()
    );
}

#[tokio::test]
async fn a_two_node_cycle_is_rejected_before_anything_is_written() {
    let (store, board) = board();
    board
        .create_task(spec_depending_on("alpha", &["beta"]))
        .await
        .unwrap();

    let err = board
        .create_task(spec_depending_on("beta", &["alpha"]))
        .await
        .unwrap_err();

    assert_eq!(
        validation_message(&err),
        "Circular dependency: alpha -> beta -> alpha"
    );
    assert!(store.task(TEAM, "beta").await.unwrap().is_none());
    assert_eq!(
        store.dependencies(TEAM, "beta").await.unwrap(),
        Vec::<String>::new()
    );
    assert_eq!(status_of(&store, "alpha").await, TaskStatus::Blocked);
}

#[tokio::test]
async fn a_duplicate_id_and_an_unknown_link_are_validation_errors() {
    let (store, board) = board();
    board.create_task(spec("t1")).await.unwrap();

    let duplicate = board.create_task(spec("t1")).await.unwrap_err();
    assert_eq!(
        validation_message(&duplicate),
        "task 't1' already exists in team 'alpha'"
    );

    let unknown_dependent = TaskGraphSpec {
        depended_by: owned(&["ghost"]),
        ..spec("t2")
    };
    let missing_target = board.create_task(unknown_dependent).await.unwrap_err();
    assert_eq!(
        validation_message(&missing_target),
        "depended_by task 'ghost' does not exist"
    );
    assert!(store.task(TEAM, "t2").await.unwrap().is_none());

    let blank_dependency = board
        .create_task(spec_depending_on("t3", &["  "]))
        .await
        .unwrap_err();
    assert_eq!(
        validation_message(&blank_dependency),
        "depends_on must not contain an empty id"
    );

    // A `depends_on` target that does not exist yet is not an error: it is an unresolved
    // dependency, so the task is created parked exactly like one waiting on a real task.
    let forward = board
        .create_task(spec_depending_on("t4", &["later"]))
        .await
        .unwrap();
    assert_eq!(forward.task.status, TaskStatus::Blocked);
    assert_eq!(
        store.dependencies(TEAM, "t4").await.unwrap(),
        owned(&["later"])
    );
}

// -----------------------------------------------------------------------------------------------
// 3. Claim / assign / reassign
// -----------------------------------------------------------------------------------------------

#[tokio::test]
async fn claim_takes_a_pending_task_and_refuses_other_owners() {
    let (store, board) = board();
    board.create_task(spec("task")).await.unwrap();

    let claimed = board.claim("task", "bob").await.unwrap();
    assert!(claimed.is_ok(), "claim was {claimed:?}");
    let row = store.task(TEAM, "task").await.unwrap().unwrap();
    assert_eq!(row.status, TaskStatus::InProgress);
    assert_eq!(row.assignee.as_deref(), Some("bob"));
    assert_eq!(board.claimable().await.unwrap(), Vec::new());

    let taken = board.claim("task", "alice").await.unwrap();
    assert!(!taken.is_ok());
    assert_eq!(
        taken.reason(),
        Some("task 'task' is already claimed by 'bob'")
    );

    let again = board.claim("task", "bob").await.unwrap();
    assert!(again.is_ok(), "a re-claim by the owner is a no-op success");
    assert_eq!(again.reason(), None);
    assert_eq!(status_of(&store, "task").await, TaskStatus::InProgress);
}

#[tokio::test]
async fn claim_refuses_a_blocked_task_and_only_free_pending_tasks_are_claimable() {
    let (store, board) = board();
    board
        .create_task(spec_depending_on("blocked", &["missing"]))
        .await
        .unwrap();
    board.create_task(spec("free")).await.unwrap();
    let preassigned = TaskGraphSpec {
        assignee: Some("bob".to_owned()),
        ..spec("preassigned")
    };
    board.create_task(preassigned).await.unwrap();

    let refused = board.claim("blocked", "bob").await.unwrap();
    assert!(!refused.is_ok());
    assert_eq!(
        refused.reason(),
        Some(
            "task 'blocked' cannot be claimed from status 'blocked' (only pending tasks are claimable)"
        )
    );
    assert_eq!(status_of(&store, "blocked").await, TaskStatus::Blocked);

    // A pending task that already carries an owner is not in the claimable pool.
    let claimable: Vec<String> = board
        .claimable()
        .await
        .unwrap()
        .into_iter()
        .map(|summary| summary.task_id)
        .collect();
    assert_eq!(claimable, owned(&["free"]));
}

#[tokio::test]
async fn assign_moves_a_pending_task_into_the_members_entry_status() {
    let (store, board) = board();
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
async fn assign_keeps_a_blocked_task_parked_and_refuses_bad_targets() {
    let (store, board) = board();
    add_builders(&store, &["bob", "alice"]).await;
    board
        .create_task(spec_depending_on("parked", &["gate"]))
        .await
        .unwrap();
    board.create_task(spec("task")).await.unwrap();

    let parked = board.assign("parked", "bob", None).await.unwrap();
    assert!(parked.is_ok(), "assign was {parked:?}");
    let row = store.task(TEAM, "parked").await.unwrap().unwrap();
    assert_eq!(row.status, TaskStatus::Blocked);
    assert_eq!(row.assignee.as_deref(), Some("bob"));

    let unknown = board.assign("task", "ghost", None).await.unwrap();
    assert!(!unknown.is_ok());
    assert_eq!(
        unknown.reason(),
        Some("member 'ghost' not found in team 'alpha'")
    );

    board.claim("task", "bob").await.unwrap();
    let foreign = board.assign("task", "alice", None).await.unwrap();
    assert_eq!(
        foreign.reason(),
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
async fn reassign_swaps_the_owner_in_place_and_refuses_bad_hand_overs() {
    let (store, board) = board();
    add_builders(&store, &["bob", "alice"]).await;
    board.create_task(spec("task")).await.unwrap();
    board.create_task(spec("free")).await.unwrap();
    board.claim("task", "bob").await.unwrap();

    let moved = board.reassign("task", "bob", "alice").await.unwrap();
    assert!(moved.is_ok(), "reassign was {moved:?}");
    let row = store.task(TEAM, "task").await.unwrap().unwrap();
    assert_eq!(row.assignee.as_deref(), Some("alice"));
    assert_eq!(
        row.status,
        TaskStatus::InProgress,
        "no bounce through pending"
    );
    assert_eq!(
        row.change_notes,
        owned(&["r0: reassigned from bob to alice (bob)"])
    );

    let not_holder = board.reassign("task", "bob", "alice").await.unwrap();
    assert_eq!(
        not_holder.reason(),
        Some("task 'task' is not assigned to 'bob'")
    );

    let unknown = board.reassign("task", "alice", "ghost").await.unwrap();
    assert_eq!(
        unknown.reason(),
        Some("member 'ghost' not found in team 'alpha'")
    );

    let unowned = board.reassign("free", "bob", "alice").await.unwrap();
    assert_eq!(
        unowned.reason(),
        Some("task 'free' has no current assignee to reassign from")
    );
}

#[tokio::test]
async fn reassign_refuses_a_status_that_cannot_hand_over() {
    let (store, board) = board();
    add_builders(&store, &["alice"]).await;
    board
        .create_task(spec_reviewed_by("reviewed", &["alice"]))
        .await
        .unwrap();
    board.claim("reviewed", "bob").await.unwrap();
    board.complete("reviewed", "bob", "").await.unwrap();

    let in_review = board.reassign("reviewed", "bob", "alice").await.unwrap();

    assert!(!in_review.is_ok());
    assert_eq!(
        in_review.reason(),
        Some("task 'reviewed' cannot be reassigned from status 'in_review'")
    );
}

// -----------------------------------------------------------------------------------------------
// 4. Completion without a review gate
// -----------------------------------------------------------------------------------------------

#[tokio::test]
async fn completing_without_reviewers_closes_the_task() {
    let (store, board) = board();
    board.create_task(spec("task")).await.unwrap();

    let unassigned = board.complete("task", "bob", "").await.unwrap_err();
    assert_eq!(
        validation_message(&unassigned),
        "task is assigned to '<unassigned>'"
    );

    board.claim("task", "bob").await.unwrap();
    let stranger = board.complete("task", "alice", "").await.unwrap_err();
    assert_eq!(validation_message(&stranger), "task is assigned to 'bob'");

    let mutation = board.complete("task", "bob", "shipped").await.unwrap();

    assert_eq!(mutation.task.status, TaskStatus::Completed);
    assert_eq!(mutation.task.summary.as_deref(), Some("shipped"));
    assert_eq!(mutation.unblocked, Vec::<String>::new());
    assert_eq!(event_types(&mutation), ["task_status_changed"]);
    let row = store.task(TEAM, "task").await.unwrap().unwrap();
    assert_eq!(row.status, TaskStatus::Completed);
    assert_eq!(row.summary.as_deref(), Some("shipped"));
}

#[tokio::test]
async fn completing_a_task_that_is_not_in_progress_is_a_validation_error() {
    let (_store, board) = board();
    let parked = TaskGraphSpec {
        assignee: Some("bob".to_owned()),
        ..spec_depending_on("blocked", &["gate"])
    };
    board.create_task(parked).await.unwrap();

    let err = board.complete("blocked", "bob", "").await.unwrap_err();

    assert_eq!(
        validation_message(&err),
        "task 'blocked' cannot be completed from status 'blocked'"
    );
}

// -----------------------------------------------------------------------------------------------
// 5. The review gate: policy, autonomous verdicts, scheduled tally
// -----------------------------------------------------------------------------------------------

#[tokio::test]
async fn a_reviewed_completion_waits_in_review_for_the_verdict() {
    let (store, board) = board();
    // Reviewers are trimmed and deduplicated at creation time.
    board
        .create_task(spec_reviewed_by("upstream", &["alice", "alice", " "]))
        .await
        .unwrap();
    board
        .create_task(spec_depending_on("downstream", &["upstream"]))
        .await
        .unwrap();
    board.claim("upstream", "author").await.unwrap();

    let mutation = board
        .complete("upstream", "author", "please check")
        .await
        .unwrap();

    assert_eq!(mutation.task.status, TaskStatus::InReview);
    assert_eq!(mutation.task.review_round, 1);
    assert_eq!(mutation.unblocked, Vec::<String>::new());
    assert_eq!(event_types(&mutation), ["task_status_changed"]);
    assert_eq!(status_of(&store, "downstream").await, TaskStatus::Blocked);

    let detail = board
        .detail("upstream")
        .await
        .unwrap()
        .expect("the upstream detail");
    assert_eq!(detail.reviewers, owned(&["alice"]));
    assert_eq!(detail.max_review_rounds, 3);
    assert_eq!(detail.summary.as_deref(), Some("please check"));

    // No vote yet: nothing is decided, and the empty round reports no round number at all.
    let tally = board.review_tally("upstream").await.unwrap();
    assert_eq!(
        tally,
        ReviewTally {
            reviewer_count: 1,
            ..ReviewTally::default()
        }
    );
    assert!(tally.is_empty());
    assert!(!tally.decided());
}

#[tokio::test]
async fn the_board_default_decides_the_review_round_ceiling() {
    let (_store, board) = board_with(TaskBoardLimits::new().with_default_max_review_rounds(7));
    let strict = TaskGraphSpec {
        max_review_rounds: Some(5),
        ..spec_reviewed_by("inherited", &["alice"])
    };
    board
        .create_task(spec_reviewed_by("default", &["alice"]))
        .await
        .unwrap();
    board.create_task(strict).await.unwrap();

    let inherited = board
        .detail("default")
        .await
        .unwrap()
        .expect("the default detail");
    let pinned = board
        .detail("inherited")
        .await
        .unwrap()
        .expect("the pinned detail");

    assert_eq!(inherited.max_review_rounds, 7);
    assert_eq!(pinned.max_review_rounds, 5);
}

#[tokio::test]
async fn a_passing_verify_completes_the_task_and_unblocks_dependents() {
    let (store, board) = board();
    reviewed_task(&board, &["alice"]).await;

    let mutation = board
        .verify("reviewed", "alice", Verdict::Pass, "looks good")
        .await
        .unwrap();

    assert_eq!(mutation.task.status, TaskStatus::Completed);
    assert_eq!(mutation.unblocked, owned(&["downstream"]));
    assert_eq!(
        event_types(&mutation),
        ["task_review_vote", "task_status_changed", "task_verified"]
    );
    assert_eq!(count_events(&mutation, "task_verified"), 1);
    assert_eq!(verified_verdicts(&mutation), [Verdict::Pass]);
    assert_eq!(status_of(&store, "downstream").await, TaskStatus::Pending);
}

#[tokio::test]
async fn a_failing_verify_returns_the_task_to_progress_with_a_change_note() {
    let (store, board) = board();
    reviewed_task(&board, &["alice"]).await;

    let mutation = board
        .verify("reviewed", "alice", Verdict::Fail, "add tests")
        .await
        .unwrap();

    assert_eq!(mutation.task.status, TaskStatus::InProgress);
    assert_eq!(
        mutation.task.assignee, None,
        "a foreign reviewer releases it"
    );
    assert_eq!(mutation.task.note.as_deref(), Some("add tests"));
    assert_eq!(
        mutation.task.change_notes,
        owned(&["r0: review failed: add tests (alice)"])
    );
    assert_eq!(verified_verdicts(&mutation), [Verdict::Fail]);
    assert_eq!(mutation.unblocked, Vec::<String>::new());
    assert_eq!(status_of(&store, "downstream").await, TaskStatus::Blocked);
}

#[tokio::test]
async fn verify_refuses_a_task_that_is_not_in_review_or_a_stranger() {
    let (_store, board) = board();
    board
        .create_task(spec_reviewed_by("reviewed", &["alice"]))
        .await
        .unwrap();

    let too_early = board
        .verify("reviewed", "alice", Verdict::Pass, "")
        .await
        .unwrap_err();
    assert_eq!(
        validation_message(&too_early),
        "task 'reviewed' is not under review (status 'pending'); nothing to verify"
    );

    board.claim("reviewed", "author").await.unwrap();
    board.complete("reviewed", "author", "").await.unwrap();
    let stranger = board
        .verify("reviewed", "mallory", Verdict::Pass, "")
        .await
        .unwrap_err();
    assert_eq!(
        validation_message(&stranger),
        "'mallory' is not a reviewer of task 'reviewed'; cannot verify it"
    );
}

#[tokio::test]
async fn the_review_round_is_monotonic_and_the_tally_reports_it() {
    let (_store, board) = board();
    // The author is the reviewer, so a failed verdict keeps the task assigned to them.
    reviewed_task(&board, &["author"]).await;

    let failed = board
        .verify("reviewed", "author", Verdict::Fail, "redo it")
        .await
        .unwrap();
    assert_eq!(failed.task.status, TaskStatus::InProgress);
    assert_eq!(failed.task.assignee.as_deref(), Some("author"));

    let second = board
        .complete("reviewed", "author", "second try")
        .await
        .unwrap();
    assert_eq!(second.task.status, TaskStatus::InReview);
    assert_eq!(second.task.review_round, 2, "rounds never rewind");

    board
        .record_vote("reviewed", "author", Verdict::Pass)
        .await
        .unwrap();
    let tally = board.review_tally("reviewed").await.unwrap();
    assert_eq!(tally.round, 2);
    assert_eq!(tally.reviewer_count, 1);
    assert_eq!(tally.pass_count, 1);
    assert_eq!(tally.fail_count, 0);
    assert!(tally.decided());
    assert!(tally.passed);
}

#[tokio::test]
async fn a_scheduled_round_waits_for_every_reviewer_before_it_settles() {
    let (store, board) = board();
    reviewed_task(&board, &["alice", "bob"]).await;

    let vote = board
        .record_vote("reviewed", "alice", Verdict::Pass)
        .await
        .unwrap();
    assert_eq!(
        vote.task.status,
        TaskStatus::InReview,
        "a vote changes nothing"
    );
    assert_eq!(event_types(&vote), ["task_review_vote"]);
    assert_eq!(count_events(&vote, "task_verified"), 0);

    let partial = board.review_tally("reviewed").await.unwrap();
    assert_eq!(partial.round, 1);
    assert_eq!(partial.pass_count, 1);
    assert_eq!(partial.fail_count, 0);
    assert_eq!(partial.reviewer_count, 2);
    assert_eq!(partial.voted(), 1);
    assert!(!partial.decided());
    assert!(!partial.passed);

    let undecided = board.settle_review("reviewed").await.unwrap_err();
    assert_eq!(
        validation_message(&undecided),
        "review tally of task 'reviewed' is not decided yet (1 of 2 votes)"
    );
    assert_eq!(status_of(&store, "reviewed").await, TaskStatus::InReview);
    assert_eq!(status_of(&store, "downstream").await, TaskStatus::Blocked);

    board
        .record_vote("reviewed", "bob", Verdict::Pass)
        .await
        .unwrap();
    let settled = board.settle_review("reviewed").await.unwrap();

    assert_eq!(settled.task.status, TaskStatus::Completed);
    assert_eq!(settled.unblocked, owned(&["downstream"]));
    assert_eq!(
        event_types(&settled),
        ["task_status_changed", "task_verified", "task_review_vote"]
    );
    assert_eq!(verified_verdicts(&settled), [Verdict::Pass]);
    assert_eq!(status_of(&store, "downstream").await, TaskStatus::Pending);
}

#[tokio::test]
async fn a_scheduled_failing_round_returns_the_task_to_progress() {
    let (store, board) = board();
    reviewed_task(&board, &["alice", "bob"]).await;
    board
        .record_vote("reviewed", "alice", Verdict::Fail)
        .await
        .unwrap();
    board
        .record_vote("reviewed", "bob", Verdict::Fail)
        .await
        .unwrap();

    let settled = board.settle_review("reviewed").await.unwrap();

    assert_eq!(settled.task.status, TaskStatus::InProgress);
    assert_eq!(
        settled.task.assignee.as_deref(),
        Some("author"),
        "the author keeps the task to rework it"
    );
    assert_eq!(
        settled.task.note.as_deref(),
        Some("review round 1 failed (0 pass / 2 fail)")
    );
    assert_eq!(verified_verdicts(&settled), [Verdict::Fail]);
    assert_eq!(settled.unblocked, Vec::<String>::new());
    assert_eq!(status_of(&store, "downstream").await, TaskStatus::Blocked);
}

#[tokio::test]
async fn a_tied_round_is_decided_but_does_not_pass() {
    let (_store, board) = board();
    reviewed_task(&board, &["alice", "bob"]).await;
    board
        .record_vote("reviewed", "alice", Verdict::Pass)
        .await
        .unwrap();
    board
        .record_vote("reviewed", "bob", Verdict::Fail)
        .await
        .unwrap();

    let tally = board.review_tally("reviewed").await.unwrap();
    assert!(tally.decided());
    assert!(!tally.passed);

    let settled = board.settle_review("reviewed").await.unwrap();
    assert_eq!(settled.task.status, TaskStatus::InProgress);
    assert_eq!(
        settled.task.note.as_deref(),
        Some("review round 1 failed (1 pass / 1 fail)")
    );
}

#[tokio::test]
async fn voting_refuses_a_task_that_is_not_under_review_or_a_stranger() {
    let (_store, board) = board();
    board
        .create_task(spec_reviewed_by("reviewed", &["alice"]))
        .await
        .unwrap();
    board.create_task(spec("open")).await.unwrap();

    let too_early = board
        .record_vote("reviewed", "alice", Verdict::Pass)
        .await
        .unwrap_err();
    assert_eq!(
        validation_message(&too_early),
        "task 'reviewed' is not under review (status 'pending'); nothing to vote on"
    );

    let nothing_to_settle = board.settle_review("open").await.unwrap_err();
    assert_eq!(
        validation_message(&nothing_to_settle),
        "task 'open' is not under review (status 'pending'); nothing to settle"
    );

    board.claim("reviewed", "author").await.unwrap();
    board.complete("reviewed", "author", "").await.unwrap();
    let stranger = board
        .record_vote("reviewed", "mallory", Verdict::Pass)
        .await
        .unwrap_err();
    assert_eq!(
        validation_message(&stranger),
        "'mallory' is not a reviewer of task 'reviewed'; cannot vote on it"
    );
}

// -----------------------------------------------------------------------------------------------
// 6. Subtasks
// -----------------------------------------------------------------------------------------------

#[tokio::test]
async fn a_depth_one_subtask_is_created_and_parks_its_parent() {
    let (store, board) = board_with(TaskBoardLimits::new().with_max_subtask_depth(1));
    board.create_task(spec("top")).await.unwrap();

    let created = board.create_subtask("top", spec("child")).await.unwrap();

    assert_eq!(created.task.status, TaskStatus::Pending);
    assert_eq!(created.task.parent_task_id.as_deref(), Some("top"));
    assert_eq!(created.task.parent_revision, Some(0));
    assert!(
        created.unblocked.is_empty(),
        "creating a subtask wakes nobody"
    );
    assert_eq!(
        event_types(&created),
        ["task_created", "task_status_changed", "task_status_changed"]
    );

    let parent = store.task(TEAM, "top").await.unwrap().unwrap();
    assert_eq!(parent.status, TaskStatus::Blocked);
    assert_eq!(
        store.dependencies(TEAM, "top").await.unwrap(),
        owned(&["child"])
    );

    let children: Vec<String> = board
        .subtasks("top")
        .await
        .unwrap()
        .into_iter()
        .map(|summary| summary.task_id)
        .collect();
    assert_eq!(children, owned(&["child"]));

    let too_deep = board
        .create_subtask("child", spec("grandchild"))
        .await
        .unwrap_err();
    assert_eq!(
        validation_message(&too_deep),
        "subtask depth 2 exceeds max_subtask_depth 1"
    );
    assert!(store.task(TEAM, "grandchild").await.unwrap().is_none());
}

#[tokio::test]
async fn subtasks_are_disabled_at_depth_zero() {
    let (store, board) = board();
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
async fn a_subtask_of_an_unknown_parent_is_not_found() {
    let (_store, board) = board_with(TaskBoardLimits::new().with_max_subtask_depth(2));

    let err = board
        .create_subtask("ghost", spec("child"))
        .await
        .unwrap_err();

    assert_eq!(not_found_message(&err), "task 'ghost' not found");
}

#[tokio::test]
async fn finishing_every_subtask_releases_the_parent_without_completing_it() {
    let (store, board) = board_with(TaskBoardLimits::new().with_max_subtask_depth(1));
    board.create_task(spec("top")).await.unwrap();
    board.create_subtask("top", spec("child")).await.unwrap();

    board.claim("child", "bob").await.unwrap();
    let mutation = board.complete("child", "bob", "child done").await.unwrap();

    assert_eq!(mutation.unblocked, owned(&["top"]));
    assert_eq!(status_of(&store, "top").await, TaskStatus::Pending);
    assert!(!board.is_settled().await.unwrap());
}

// -----------------------------------------------------------------------------------------------
// 7. Cancel / reset / unknown ids
// -----------------------------------------------------------------------------------------------

#[tokio::test]
async fn cancel_closes_a_pending_task_and_refuses_a_terminal_one() {
    let (_store, board) = board();
    board.create_task(spec("open")).await.unwrap();

    let cancelled = board.cancel("open").await.unwrap();

    assert_eq!(cancelled.task.status, TaskStatus::Cancelled);
    assert!(cancelled.task.status.is_terminal());
    assert_eq!(cancelled.unblocked, Vec::<String>::new());
    assert_eq!(board.claimable().await.unwrap(), Vec::new());

    let claim_after = board.claim("open", "bob").await.unwrap();
    assert_eq!(
        claim_after.reason(),
        Some(
            "task 'open' cannot be claimed from status 'cancelled' (only pending tasks are claimable)"
        )
    );

    let err = board.cancel("open").await.unwrap_err();
    assert_eq!(
        validation_message(&err),
        "task 'open' cannot be cancelled from status 'cancelled' (already terminal)"
    );
}

#[tokio::test]
async fn cancel_keeps_the_owner_and_releases_the_dependents() {
    let (store, board) = board();
    board.create_task(spec("upstream")).await.unwrap();
    board
        .create_task(spec_depending_on("downstream", &["upstream"]))
        .await
        .unwrap();
    board.claim("upstream", "bob").await.unwrap();

    let cancelled = board.cancel("upstream").await.unwrap();

    assert_eq!(cancelled.task.status, TaskStatus::Cancelled);
    assert_eq!(cancelled.task.assignee.as_deref(), Some("bob"));
    assert_eq!(event_types(&cancelled), ["task_status_changed"]);
    assert_eq!(cancelled.unblocked, owned(&["downstream"]));
    assert_eq!(status_of(&store, "downstream").await, TaskStatus::Pending);
    assert!(!board.is_settled().await.unwrap());
}

#[tokio::test]
async fn reset_returns_a_closed_task_to_pending_without_rewinding_the_round() {
    let (store, board) = board();
    board.create_task(spec("closed")).await.unwrap();
    board.claim("closed", "bob").await.unwrap();
    board.complete("closed", "bob", "shipped").await.unwrap();

    let reset = board.reset("closed").await.unwrap();

    assert_eq!(reset.task.status, TaskStatus::Pending);
    assert_eq!(reset.task.assignee, None);
    assert_eq!(reset.task.summary, None);
    assert_eq!(event_types(&reset), ["task_status_changed"]);
    let claimable: Vec<String> = board
        .claimable()
        .await
        .unwrap()
        .into_iter()
        .map(|summary| summary.task_id)
        .collect();
    assert_eq!(claimable, owned(&["closed"]));

    let reviewed = reviewed_task(&board, &["alice"]).await;
    assert_eq!(reviewed.task.review_round, 1);
    let reset = board.reset("reviewed").await.unwrap();
    assert_eq!(reset.task.status, TaskStatus::Pending);
    assert_eq!(
        reset.task.review_round, 1,
        "the round stays monotonic across a reset"
    );
    assert_eq!(status_of(&store, "reviewed").await, TaskStatus::Pending);

    let err = board.reset("reviewed").await.unwrap_err();
    assert_eq!(
        validation_message(&err),
        "task 'reviewed' cannot be reset from status 'pending'; only completed / cancelled / in-review tasks can be reset"
    );
}

#[tokio::test]
async fn an_unknown_task_id_is_not_found_or_an_empty_read() {
    let (_store, board) = board();

    let claim_err = board.claim("ghost", "bob").await.unwrap_err();
    assert_eq!(not_found_message(&claim_err), "task 'ghost' not found");

    let cancel_err = board.cancel("ghost").await.unwrap_err();
    assert_eq!(not_found_message(&cancel_err), "task 'ghost' not found");

    let complete_err = board.complete("ghost", "bob", "").await.unwrap_err();
    assert_eq!(not_found_message(&complete_err), "task 'ghost' not found");

    let reset_err = board.reset("ghost").await.unwrap_err();
    assert_eq!(not_found_message(&reset_err), "task 'ghost' not found");

    // Reads of an unknown id are empty, not errors.
    assert!(board.detail("ghost").await.unwrap().is_none());
    assert_eq!(board.subtasks("ghost").await.unwrap(), Vec::new());
    assert_eq!(
        board.view(&TaskFilter::default()).await.unwrap(),
        Vec::new()
    );
    assert!(board.is_settled().await.unwrap());
}
