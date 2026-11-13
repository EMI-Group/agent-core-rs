//! Pool suite: creation and the decomposition tree, claims and starts, completion and failure, the
//! review gate, delegation through the router, the read filters and the wire shapes.
//!
//! Every test drives a real [`OrgTaskPool`] over the in-memory store through the `dyn TeamStore`
//! seam, so the assertions are the ones a future backend has to keep as well. No sleeps: the store
//! stamps `updated_at`, and every read is ordered by that stamp.
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use openjiuwen_common::error::{Error, Result};

use crate::org::delegation::{DelegationRouter, DelegationRules};
use crate::org::roles::{OrgChart, OrgMember, OrgRole, OrgTeam, OrganizationSpec};
use crate::org::{qualify_member, split_qualified};
use crate::schema::{TaskFilter, TaskGraphSpec, TaskRecord, TaskStatus, Verdict};
use crate::storage::{InMemoryTeamStore, TeamStore};
use crate::taskboard::{TaskBoard, TaskBoardLimits};

use super::*;

// -----------------------------------------------------------------------------------------------
// Fixtures
// -----------------------------------------------------------------------------------------------

fn team(name: &str, reports_to: Option<&str>) -> OrgTeam {
    OrgTeam {
        team_name: name.to_owned(),
        display_name: name.to_owned(),
        reports_to: reports_to.map(str::to_owned),
        ..OrgTeam::default()
    }
}

fn leader(team_name: &str) -> OrgMember {
    OrgMember {
        member_id: "team_leader".to_owned(),
        team_name: team_name.to_owned(),
        role: OrgRole::Leader,
        ..OrgMember::default()
    }
}

fn member(team_name: &str, member_id: &str) -> OrgMember {
    OrgMember {
        member_id: member_id.to_owned(),
        team_name: team_name.to_owned(),
        ..OrgMember::default()
    }
}

/// `alpha` (root) with `team_leader` / `dev-1`, and `beta` reporting to it with `team_leader` /
/// `dev-2`.
fn chart_spec() -> OrganizationSpec {
    OrganizationSpec {
        organization_id: "acme".to_owned(),
        display_name: "Acme".to_owned(),
        teams: vec![team("alpha", None), team("beta", Some("alpha"))],
        members: vec![
            leader("alpha"),
            member("alpha", "dev-1"),
            leader("beta"),
            member("beta", "dev-2"),
        ],
        ..OrganizationSpec::default()
    }
}

/// `alpha` (root) with two sibling teams `beta` and `gamma` below it.
fn sibling_spec() -> OrganizationSpec {
    OrganizationSpec {
        organization_id: "acme".to_owned(),
        teams: vec![
            team("alpha", None),
            team("beta", Some("alpha")),
            team("gamma", Some("alpha")),
        ],
        members: vec![
            leader("alpha"),
            member("alpha", "dev-1"),
            leader("beta"),
            leader("gamma"),
        ],
        ..OrganizationSpec::default()
    }
}

fn store() -> Arc<dyn TeamStore> {
    Arc::new(InMemoryTeamStore::new())
}

fn pool_over(
    spec: OrganizationSpec,
    store: &Arc<dyn TeamStore>,
    rules: DelegationRules,
) -> OrgTaskPool {
    let chart = Arc::new(OrgChart::new(spec).unwrap());
    OrgTaskPool::new(store.clone(), Arc::new(DelegationRouter::new(chart, rules))).unwrap()
}

fn pool(store: &Arc<dyn TeamStore>) -> OrgTaskPool {
    pool_over(chart_spec(), store, DelegationRules::default())
}

fn pool_with_rules(store: &Arc<dyn TeamStore>, rules: DelegationRules) -> OrgTaskPool {
    pool_over(chart_spec(), store, rules)
}

/// A pool whose emitted events land in the returned recorder.
fn recording_pool(
    spec: OrganizationSpec,
    store: &Arc<dyn TeamStore>,
) -> (OrgTaskPool, Arc<Mutex<Vec<OrgTaskEvent>>>) {
    let seen: Arc<Mutex<Vec<OrgTaskEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let recorder = Arc::clone(&seen);
    let pool = pool_over(spec, store, DelegationRules::default())
        .with_event_sink(Arc::new(move |event| recorder.lock().unwrap().push(event)));
    (pool, seen)
}

fn reviewers(names: &[&str]) -> Vec<String> {
    names.iter().map(|name| (*name).to_owned()).collect()
}

/// Creates a root task and returns it.
async fn root(pool: &OrgTaskPool, owner: &str, title: &str) -> OrgTask {
    pool.create(OrgTaskSpec::new(owner, title, "body"))
        .await
        .unwrap()
        .task
}

/// Claims `task_id` for `alpha::dev-1` and returns the claim result.
async fn claim_alpha(pool: &OrgTaskPool, task_id: &str) -> OrgTaskOpResult {
    pool.claim(task_id, "alpha", "dev-1").await.unwrap()
}

/// Creates, claims and completes a task without reviewers.
async fn completed_task(pool: &OrgTaskPool, owner: &str, title: &str) -> OrgTask {
    let task = root(pool, owner, title).await;
    claim_alpha(pool, &task.task_id).await;
    pool.complete(&task.task_id, "alpha", "dev-1", "done")
        .await
        .unwrap()
        .task
        .unwrap()
}

/// Reads the stored row of `task_id`.
async fn row(store: &Arc<dyn TeamStore>, namespace: &str, task_id: &str) -> TaskRecord {
    store.task(namespace, task_id).await.unwrap().unwrap()
}

/// Asserts that `spec` is rejected as a validation error and returns the message.
async fn validation_message(pool: &OrgTaskPool, spec: OrgTaskSpec) -> String {
    match pool.create(spec).await.unwrap_err() {
        Error::Validation(message) => message,
        other => panic!("expected a validation error, got {other:?}"),
    }
}

// -----------------------------------------------------------------------------------------------
// Creation
// -----------------------------------------------------------------------------------------------

#[tokio::test]
async fn create_root_task_defaults() {
    let store = store();
    let pool = pool(&store);
    let mutation = pool
        .create(OrgTaskSpec::new("alpha", "  Ship it ", " the body ").by("alpha::dev-1"))
        .await
        .unwrap();
    let task = &mutation.task;
    assert_eq!(task.organization_id, "acme");
    assert_eq!(task.owner_team, "alpha");
    assert_eq!(task.target_team, "alpha");
    assert_eq!(task.title, "Ship it");
    assert_eq!(task.content, "the body");
    assert_eq!(task.status, TaskStatus::Pending);
    assert_eq!(task.depth, 0);
    assert_eq!(task.created_by, "alpha::dev-1");
    assert!(task.is_open());
    assert!(!task.is_terminal());
    assert_eq!(task.reviewers, Vec::<String>::new());
    let hex = task.task_id.strip_prefix("task_").unwrap();
    assert_eq!(hex.len(), 32);
    assert!(hex
        .chars()
        .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    assert_eq!(mutation.task_id(), task.task_id);
    assert_eq!(mutation.unblocked, Vec::<String>::new());
    assert_eq!(
        mutation.events,
        vec![OrgTaskEvent::Created {
            task_id: task.task_id.clone(),
            owner_team: "alpha".to_owned(),
            target_team: "alpha".to_owned(),
        }]
    );
    assert_eq!(pool.organization_id(), "acme");
    assert_eq!(pool.namespace(), "org::acme");
    assert_eq!(pool.store().name(), "memory");
    assert_eq!(pool.router().chart().organization_id(), "acme");
    assert_eq!(
        pool.get(&task.task_id).await.unwrap().unwrap(),
        mutation.task
    );
}

#[tokio::test]
async fn create_resolves_target_team_and_reviewers() {
    let store = store();
    let pool = pool(&store);
    let task = pool
        .create(
            OrgTaskSpec::new("alpha", "reviewed", "body")
                .for_team(" beta ")
                .reviewed_by(reviewers(&["dev-2", "beta::team_leader", "dev-2", " "]))
                .by("dev-1"),
        )
        .await
        .unwrap()
        .task;
    assert_eq!(task.target_team, "beta");
    assert_eq!(task.owner_team, "alpha");
    assert_eq!(
        task.reviewers,
        reviewers(&["beta::dev-2", "beta::team_leader"])
    );
    assert_eq!(task.created_by, "alpha::dev-1");
}

#[tokio::test]
async fn create_defaults_the_creator_to_the_owner_leader() {
    let pool = pool(&store());
    let task = pool
        .create(OrgTaskSpec::new("beta", "t", "c"))
        .await
        .unwrap()
        .task;
    assert_eq!(task.created_by, "beta::team_leader");
}

#[tokio::test]
async fn create_rejects_unknown_teams_and_blank_titles() {
    let store = store();
    let pool = pool(&store);
    let message = validation_message(&pool, OrgTaskSpec::new("ghost", "t", "c")).await;
    assert!(
        message.contains("unknown organization team 'ghost'"),
        "{message}"
    );
    let message =
        validation_message(&pool, OrgTaskSpec::new("alpha", "t", "c").for_team("ghost")).await;
    assert!(
        message.contains("unknown organization team 'ghost'"),
        "{message}"
    );
    let message = validation_message(&pool, OrgTaskSpec::new("alpha", "  ", "c")).await;
    assert!(
        message.contains("task title must not be empty"),
        "{message}"
    );
}

#[tokio::test]
async fn create_rejects_bad_reviewers_and_creators() {
    let store = store();
    let pool = pool(&store);
    let message = validation_message(
        &pool,
        OrgTaskSpec::new("alpha", "t", "c")
            .for_team("beta")
            .reviewed_by(reviewers(&["dev-1"])),
    )
    .await;
    assert!(
        message.contains("reviewer 'dev-1' is not a member of team 'beta'"),
        "{message}"
    );
    let message = validation_message(
        &pool,
        OrgTaskSpec::new("alpha", "t", "c")
            .for_team("beta")
            .reviewed_by(reviewers(&["alpha::dev-1"])),
    )
    .await;
    assert!(
        message.contains("reviewer 'alpha::dev-1' is not a member of team 'beta'"),
        "{message}"
    );
    let message = validation_message(&pool, OrgTaskSpec::new("alpha", "t", "c").by("ghost")).await;
    assert!(
        message.contains("creator 'ghost' is not a member of team 'alpha'"),
        "{message}"
    );
}

#[tokio::test]
async fn create_child_parks_the_parent_blocked() {
    let store = store();
    let pool = pool(&store);
    let parent = root(&pool, "alpha", "parent").await;
    let child = pool
        .create(OrgTaskSpec::new("alpha", "child", "c").under(parent.task_id.clone()))
        .await
        .unwrap()
        .task;
    assert_eq!(child.depth, 1);
    assert_eq!(
        child.parent_task_id.as_deref(),
        Some(parent.task_id.as_str())
    );
    assert_eq!(child.status, TaskStatus::Pending);
    assert_eq!(
        pool.get(&parent.task_id).await.unwrap().unwrap().status,
        TaskStatus::Blocked
    );
    let children = pool.children(&parent.task_id).await.unwrap();
    assert_eq!(children.len(), 1);
    assert_eq!(children[0].task_id, child.task_id);
    let namespace = pool.namespace().to_owned();
    let dependencies = store
        .dependencies(&namespace, &parent.task_id)
        .await
        .unwrap();
    assert_eq!(dependencies, vec![child.task_id.clone()]);
    let grandchild = pool
        .create(OrgTaskSpec::new("alpha", "grandchild", "c").under(child.task_id.clone()))
        .await
        .unwrap()
        .task;
    assert_eq!(grandchild.depth, 2);
}

#[tokio::test]
async fn create_rejects_unknown_and_terminal_parents() {
    let store = store();
    let pool = pool(&store);
    let missing = pool
        .create(OrgTaskSpec::new("alpha", "child", "c").under("task_missing"))
        .await
        .unwrap_err();
    assert!(matches!(missing, Error::NotFound(_)), "{missing:?}");

    let parent = completed_task(&pool, "alpha", "parent").await;
    assert!(parent.is_terminal());
    let message = validation_message(
        &pool,
        OrgTaskSpec::new("alpha", "child", "c").under(parent.task_id),
    )
    .await;
    assert!(
        message.contains("is terminal and cannot be decomposed"),
        "{message}"
    );
}

// -----------------------------------------------------------------------------------------------
// Claims and starts
// -----------------------------------------------------------------------------------------------

#[tokio::test]
async fn claim_first_wins_and_second_is_refused() {
    let store = store();
    let pool = pool(&store);
    let task = pool
        .create(OrgTaskSpec::new("alpha", "t", "c").for_team("beta"))
        .await
        .unwrap()
        .task;
    let claimed = pool.claim(&task.task_id, "beta", "dev-2").await.unwrap();
    assert!(claimed.is_ok());
    let claimed = claimed.task.unwrap();
    assert_eq!(claimed.status, TaskStatus::InProgress);
    assert_eq!(claimed.assignee.as_deref(), Some("beta::dev-2"));
    assert_eq!(claimed.assignee_parts(), Some(("beta", "dev-2")));

    let second = pool
        .claim(&task.task_id, "beta", "team_leader")
        .await
        .unwrap();
    assert!(!second.is_ok());
    assert_eq!(second.task, None);
    let expected = format!(
        "task '{}' is already claimed by 'beta::dev-2'",
        task.task_id
    );
    assert_eq!(second.reason(), Some(expected.as_str()));
}

#[tokio::test]
async fn claim_outside_the_target_team_is_refused() {
    let store = store();
    let pool = pool(&store);
    let task = pool
        .create(OrgTaskSpec::new("alpha", "t", "c").for_team("beta"))
        .await
        .unwrap()
        .task;
    let refused = pool.claim(&task.task_id, "alpha", "dev-1").await.unwrap();
    assert!(!refused.is_ok());
    assert!(
        refused
            .reason()
            .unwrap()
            .contains("is not a member of team 'beta'"),
        "{refused:?}"
    );
    // An unknown member of the target team is refused the same way.
    let refused = pool.claim(&task.task_id, "beta", "ghost").await.unwrap();
    assert!(!refused.is_ok());
}

#[tokio::test]
async fn claim_reclaim_by_the_assignee_is_a_noop() {
    let store = store();
    let pool = pool(&store);
    let task = root(&pool, "alpha", "t").await;
    assert!(claim_alpha(&pool, &task.task_id).await.is_ok());
    let again = claim_alpha(&pool, &task.task_id).await;
    assert!(again.is_ok());
    assert_eq!(
        again.task.unwrap().assignee.as_deref(),
        Some("alpha::dev-1")
    );
}

#[tokio::test]
async fn claim_and_review_of_unknown_tasks_are_not_found() {
    let store = store();
    let pool = pool(&store);
    assert!(matches!(
        pool.claim("task_missing", "alpha", "dev-1")
            .await
            .unwrap_err(),
        Error::NotFound(_)
    ));
    assert!(matches!(
        pool.review("task_missing", "alpha", "dev-1", Verdict::Pass, "")
            .await
            .unwrap_err(),
        Error::NotFound(_)
    ));
    assert!(matches!(
        pool.delegate("task_missing", "alpha::team_leader", "beta", None)
            .await
            .unwrap_err(),
        Error::NotFound(_)
    ));
    assert_eq!(pool.get("task_missing").await.unwrap(), None);
}

#[tokio::test]
async fn start_is_reserved_to_the_preassigned_member() {
    let store = store();
    let pool = pool(&store);
    let task = root(&pool, "alpha", "t").await;
    let delegated = pool
        .delegate(&task.task_id, "alpha::team_leader", "alpha", Some("dev-1"))
        .await
        .unwrap();
    assert!(delegated.is_ok());
    assert_eq!(delegated.task.unwrap().status, TaskStatus::Pending);

    let stranger = pool
        .start(&task.task_id, "alpha", "team_leader")
        .await
        .unwrap();
    assert!(!stranger.is_ok());
    let expected = format!("task '{}' is assigned to 'alpha::dev-1'", task.task_id);
    assert_eq!(stranger.reason(), Some(expected.as_str()));

    let started = pool.start(&task.task_id, "alpha", "dev-1").await.unwrap();
    assert!(started.is_ok());
    assert_eq!(started.task.unwrap().status, TaskStatus::InProgress);
    // Starting an already running task the caller holds is idempotent.
    assert!(pool
        .start(&task.task_id, "alpha", "dev-1")
        .await
        .unwrap()
        .is_ok());
}

// -----------------------------------------------------------------------------------------------
// Completion and failure
// -----------------------------------------------------------------------------------------------

#[tokio::test]
async fn complete_without_reviewers_unblocks_the_parent() {
    let store = store();
    let (pool, seen) = recording_pool(chart_spec(), &store);
    let parent = root(&pool, "alpha", "parent").await;
    let first = pool
        .create(OrgTaskSpec::new("alpha", "first", "c").under(parent.task_id.clone()))
        .await
        .unwrap()
        .task;
    let second = pool
        .create(OrgTaskSpec::new("alpha", "second", "c").under(parent.task_id.clone()))
        .await
        .unwrap()
        .task;

    for child in [&first, &second] {
        assert!(claim_alpha(&pool, &child.task_id).await.is_ok());
        let done = pool
            .complete(&child.task_id, "alpha", "dev-1", "done")
            .await
            .unwrap();
        assert!(done.is_ok());
        assert_eq!(done.task.unwrap().status, TaskStatus::Completed);
    }
    let parent = pool.get(&parent.task_id).await.unwrap().unwrap();
    assert_eq!(parent.status, TaskStatus::Pending);
    assert_eq!(parent.summary, None);
    let events = seen.lock().unwrap().clone();
    assert!(events.iter().any(|event| matches!(
        event,
        OrgTaskEvent::Unblocked { task_id } if task_id == &parent.task_id
    )));
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, OrgTaskEvent::Completed { .. }))
            .count(),
        2
    );

    // The parent itself is completable once its children settled.
    assert!(claim_alpha(&pool, &parent.task_id).await.is_ok());
    let done = pool
        .complete(&parent.task_id, "alpha", "dev-1", "shipped")
        .await
        .unwrap();
    assert_eq!(done.task.unwrap().status, TaskStatus::Completed);
}

#[tokio::test]
async fn complete_with_reviewers_enters_the_review_round() {
    let store = store();
    let pool = pool(&store);
    let task = pool
        .create(
            OrgTaskSpec::new("alpha", "t", "c").reviewed_by(reviewers(&["dev-1", "team_leader"])),
        )
        .await
        .unwrap()
        .task;
    assert!(claim_alpha(&pool, &task.task_id).await.is_ok());
    let submitted = pool
        .complete(&task.task_id, "alpha", "dev-1", "please review")
        .await
        .unwrap();
    let submitted = submitted.task.unwrap();
    assert_eq!(submitted.status, TaskStatus::InReview);
    assert_eq!(submitted.review_round, 1);
    assert_eq!(submitted.summary.as_deref(), Some("please review"));
}

#[tokio::test]
async fn complete_by_a_stranger_or_from_a_closed_status_is_refused() {
    let store = store();
    let pool = pool(&store);
    let task = root(&pool, "alpha", "t").await;
    let refused = pool
        .complete(&task.task_id, "alpha", "dev-1", "")
        .await
        .unwrap();
    assert!(!refused.is_ok());
    assert!(refused
        .reason()
        .unwrap()
        .contains("is assigned to '<unassigned>'"));

    claim_alpha(&pool, &task.task_id).await;
    let refused = pool
        .complete(&task.task_id, "alpha", "team_leader", "")
        .await
        .unwrap();
    assert!(!refused.is_ok());
    assert!(refused
        .reason()
        .unwrap()
        .contains("is assigned to 'alpha::dev-1'"));

    assert!(pool
        .complete(&task.task_id, "alpha", "dev-1", "")
        .await
        .unwrap()
        .is_ok());
    let refused = pool
        .complete(&task.task_id, "alpha", "dev-1", "")
        .await
        .unwrap();
    assert!(!refused.is_ok());
    assert!(refused
        .reason()
        .unwrap()
        .contains("cannot be completed from status 'completed'"));
}

#[tokio::test]
async fn fail_cancels_the_task_and_keeps_the_reason() {
    let store = store();
    let pool = pool(&store);
    let parent = root(&pool, "alpha", "parent").await;
    let child = pool
        .create(OrgTaskSpec::new("alpha", "child", "c").under(parent.task_id.clone()))
        .await
        .unwrap()
        .task;
    assert!(claim_alpha(&pool, &child.task_id).await.is_ok());
    let failure = OrgTaskFailure::new("timeout", "the tool never answered");
    let failed = pool
        .fail(&child.task_id, "alpha", "dev-1", failure.clone())
        .await
        .unwrap();
    let failed = failed.task.unwrap();
    assert_eq!(failed.status, TaskStatus::Cancelled);
    assert!(failed.is_terminal());
    assert_eq!(failed.failure, Some(failure));
    assert_eq!(
        pool.get(&child.task_id).await.unwrap().unwrap().failure,
        Some(OrgTaskFailure::new("timeout", "the tool never answered"))
    );
    let namespace = pool.namespace().to_owned();
    let stored = row(&store, &namespace, &child.task_id).await;
    assert!(
        stored
            .change_notes
            .iter()
            .any(|note| note.contains("failed: timeout — the tool never answered")),
        "{:?}",
        stored.change_notes
    );
    assert_eq!(
        pool.get(&parent.task_id).await.unwrap().unwrap().status,
        TaskStatus::Pending
    );
}

#[tokio::test]
async fn fail_is_refused_for_a_task_the_caller_does_not_hold() {
    let store = store();
    let pool = pool(&store);
    let task = root(&pool, "alpha", "t").await;
    let refused = pool
        .fail(
            &task.task_id,
            "alpha",
            "dev-1",
            OrgTaskFailure::new("x", "y"),
        )
        .await
        .unwrap();
    assert!(!refused.is_ok());
    assert!(refused
        .reason()
        .unwrap()
        .contains("is assigned to '<unassigned>'"));

    claim_alpha(&pool, &task.task_id).await;
    let refused = pool
        .fail(
            &task.task_id,
            "alpha",
            "team_leader",
            OrgTaskFailure::new("x", "y"),
        )
        .await
        .unwrap();
    assert!(!refused.is_ok());
}

// -----------------------------------------------------------------------------------------------
// Review gate
// -----------------------------------------------------------------------------------------------

/// Creates a task with `reviewers`, claims it for `alpha::dev-1` and submits it for review.
async fn in_review(pool: &OrgTaskPool, gate: &[&str]) -> OrgTask {
    let task = pool
        .create(OrgTaskSpec::new("alpha", "reviewed", "c").reviewed_by(reviewers(gate)))
        .await
        .unwrap()
        .task;
    assert!(claim_alpha(pool, &task.task_id).await.is_ok());
    pool.complete(&task.task_id, "alpha", "dev-1", "ready")
        .await
        .unwrap()
        .task
        .unwrap()
}

#[tokio::test]
async fn review_pass_completes_and_unblocks_dependents() {
    let store = store();
    let pool = pool(&store);
    let parent = root(&pool, "alpha", "parent").await;
    let task = pool
        .create(
            OrgTaskSpec::new("alpha", "reviewed", "c")
                .under(parent.task_id.clone())
                .reviewed_by(reviewers(&["team_leader"])),
        )
        .await
        .unwrap()
        .task;
    assert!(claim_alpha(&pool, &task.task_id).await.is_ok());
    pool.complete(&task.task_id, "alpha", "dev-1", "ready")
        .await
        .unwrap();
    assert_eq!(
        pool.get(&parent.task_id).await.unwrap().unwrap().status,
        TaskStatus::Blocked
    );
    let passed = pool
        .review(&task.task_id, "alpha", "team_leader", Verdict::Pass, "")
        .await
        .unwrap();
    let passed = passed.task.unwrap();
    assert_eq!(passed.status, TaskStatus::Completed);
    assert_eq!(
        pool.get(&parent.task_id).await.unwrap().unwrap().status,
        TaskStatus::Pending
    );
}

#[tokio::test]
async fn review_fail_returns_the_task_for_rework() {
    let store = store();
    let pool = pool(&store);
    let task = in_review(&pool, &["team_leader"]).await;
    let failed = pool
        .review(
            &task.task_id,
            "alpha",
            "team_leader",
            Verdict::Fail,
            "needs work",
        )
        .await
        .unwrap();
    let failed = failed.task.unwrap();
    assert_eq!(failed.status, TaskStatus::InProgress);
    assert_eq!(failed.assignee, None);
    let namespace = pool.namespace().to_owned();
    let stored = row(&store, &namespace, &task.task_id).await;
    assert!(
        stored
            .change_notes
            .iter()
            .any(|note| note.contains("review failed: needs work")),
        "{:?}",
        stored.change_notes
    );
}

#[tokio::test]
async fn review_is_reserved_to_the_listed_reviewers() {
    let store = store();
    let pool = pool(&store);
    let task = in_review(&pool, &["team_leader"]).await;
    let refused = pool
        .review(&task.task_id, "alpha", "dev-1", Verdict::Pass, "")
        .await
        .unwrap();
    assert!(!refused.is_ok());
    assert!(
        refused
            .reason()
            .unwrap()
            .contains("is not a reviewer of task"),
        "{refused:?}"
    );
    // A task outside the review gate cannot be reviewed at all.
    let open = root(&pool, "alpha", "plain").await;
    let refused = pool
        .review(&open.task_id, "alpha", "team_leader", Verdict::Pass, "")
        .await
        .unwrap();
    assert!(!refused.is_ok());
    assert!(refused.reason().unwrap().contains("is not under review"));
}

#[tokio::test]
async fn scheduled_votes_are_undecided_until_every_reviewer_voted() {
    let store = store();
    let pool = pool(&store);
    let task = in_review(&pool, &["team_leader", "dev-1"]).await;
    assert!(pool
        .vote(&task.task_id, "alpha", "team_leader", Verdict::Pass)
        .await
        .unwrap()
        .is_ok());
    // A vote changes nothing but the vote table.
    assert_eq!(
        pool.get(&task.task_id).await.unwrap().unwrap().status,
        TaskStatus::InReview
    );
    let undecided = pool.settle_review(&task.task_id).await.unwrap();
    assert!(!undecided.is_ok());
    let expected = format!(
        "review tally of task '{}' is not decided yet (1 of 2 votes)",
        task.task_id
    );
    assert_eq!(undecided.reason(), Some(expected.as_str()));

    assert!(pool
        .vote(&task.task_id, "alpha", "dev-1", Verdict::Pass)
        .await
        .unwrap()
        .is_ok());
    let settled = pool.settle_review(&task.task_id).await.unwrap();
    assert!(settled.is_ok());
    assert_eq!(settled.task.unwrap().status, TaskStatus::Completed);
}

#[tokio::test]
async fn scheduled_tie_requests_rework() {
    let store = store();
    let (pool, seen) = recording_pool(chart_spec(), &store);
    let task = in_review(&pool, &["team_leader", "dev-1"]).await;
    assert!(pool
        .vote(&task.task_id, "alpha", "team_leader", Verdict::Pass)
        .await
        .unwrap()
        .is_ok());
    assert!(pool
        .vote(&task.task_id, "alpha", "dev-1", Verdict::Fail)
        .await
        .unwrap()
        .is_ok());
    let settled = pool.settle_review(&task.task_id).await.unwrap();
    assert!(settled.is_ok());
    assert_eq!(settled.task.unwrap().status, TaskStatus::InProgress);
    let events = seen.lock().unwrap().clone();
    assert!(events.iter().any(|event| matches!(
        event,
        OrgTaskEvent::Reviewed { passed: false, reviewer, .. } if reviewer == "alpha::dev-1"
    )));
}

#[tokio::test]
async fn a_vote_outside_the_gate_is_refused() {
    let store = store();
    let pool = pool(&store);
    let task = in_review(&pool, &["team_leader"]).await;
    let refused = pool
        .vote(&task.task_id, "alpha", "dev-1", Verdict::Pass)
        .await
        .unwrap();
    assert!(!refused.is_ok());
    assert!(refused.reason().unwrap().contains("cannot vote on it"));
    let plain = root(&pool, "alpha", "plain").await;
    assert!(!pool.settle_review(&plain.task_id).await.unwrap().is_ok());
}

// -----------------------------------------------------------------------------------------------
// Delegation
// -----------------------------------------------------------------------------------------------

#[tokio::test]
async fn delegate_down_the_line_moves_the_target_team() {
    let store = store();
    let (pool, seen) = recording_pool(chart_spec(), &store);
    let task = root(&pool, "alpha", "t").await;
    let delegated = pool
        .delegate(&task.task_id, "alpha::team_leader", "beta", Some("dev-2"))
        .await
        .unwrap();
    assert!(delegated.is_ok());
    let delegated = delegated.task.unwrap();
    assert_eq!(delegated.target_team, "beta");
    assert_eq!(delegated.owner_team, "alpha");
    assert_eq!(delegated.assignee.as_deref(), Some("beta::dev-2"));
    assert_eq!(delegated.status, TaskStatus::Pending);
    let namespace = pool.namespace().to_owned();
    let stored = row(&store, &namespace, &task.task_id).await;
    assert!(
        stored
            .change_notes
            .iter()
            .any(|note| note.contains("delegated to beta/dev-2")),
        "{:?}",
        stored.change_notes
    );
    assert_eq!(
        seen.lock().unwrap().clone(),
        vec![
            OrgTaskEvent::Created {
                task_id: task.task_id.clone(),
                owner_team: "alpha".to_owned(),
                target_team: "alpha".to_owned(),
            },
            OrgTaskEvent::Delegated {
                task_id: task.task_id.clone(),
                from_team: "alpha".to_owned(),
                to_team: "beta".to_owned(),
                to_member: Some("beta::dev-2".to_owned()),
            },
        ]
    );
}

#[tokio::test]
async fn delegate_to_a_team_without_a_member_keeps_the_assignee() {
    let store = store();
    let pool = pool(&store);
    let task = root(&pool, "alpha", "t").await;
    let delegated = pool
        .delegate(&task.task_id, "alpha::team_leader", "beta", None)
        .await
        .unwrap();
    assert!(delegated.is_ok());
    let delegated = delegated.task.unwrap();
    assert_eq!(delegated.target_team, "beta");
    assert_eq!(delegated.assignee, None);
}

#[tokio::test]
async fn delegate_upward_is_refused_by_the_router() {
    let store = store();
    let pool = pool(&store);
    let task = root(&pool, "beta", "t").await;
    let refused = pool
        .delegate(&task.task_id, "beta::team_leader", "alpha", None)
        .await
        .unwrap();
    assert!(!refused.is_ok());
    assert_eq!(
        refused.reason(),
        Some("delegation must follow the reporting line")
    );
}

#[tokio::test]
async fn delegate_laterally_is_refused_by_the_router() {
    let store = store();
    let pool = pool_over(sibling_spec(), &store, DelegationRules::default());
    let task = root(&pool, "gamma", "t").await;
    let refused = pool
        .delegate(&task.task_id, "gamma::team_leader", "beta", None)
        .await
        .unwrap();
    assert!(!refused.is_ok());
    assert_eq!(
        refused.reason(),
        Some("delegation must follow the reporting line")
    );
}

#[tokio::test]
async fn delegate_respects_the_target_team_budget() {
    let store = store();
    let pool = pool_with_rules(
        &store,
        DelegationRules::default().with_team_budget("beta", 1),
    );
    // One open task already targets `beta`.
    pool.create(OrgTaskSpec::new("alpha", "queued", "c").for_team("beta"))
        .await
        .unwrap();
    let task = root(&pool, "alpha", "t").await;
    let refused = pool
        .delegate(&task.task_id, "alpha::team_leader", "beta", None)
        .await
        .unwrap();
    assert!(!refused.is_ok());
    let reason = refused.reason().unwrap();
    assert!(reason.contains("budget of 1"), "{reason}");
}

#[tokio::test]
async fn delegate_refuses_unknown_delegators_and_closed_tasks() {
    let store = store();
    let pool = pool(&store);
    let task = root(&pool, "alpha", "t").await;
    let refused = pool
        .delegate(&task.task_id, "ghost", "beta", None)
        .await
        .unwrap();
    assert!(!refused.is_ok());
    assert_eq!(
        refused.reason(),
        Some("delegator is not part of the organisation")
    );

    assert!(claim_alpha(&pool, &task.task_id).await.is_ok());
    let refused = pool
        .delegate(&task.task_id, "alpha::team_leader", "beta", None)
        .await
        .unwrap();
    assert!(!refused.is_ok());
    assert!(refused
        .reason()
        .unwrap()
        .contains("cannot be delegated from status 'in_progress'"));
}

#[tokio::test]
async fn a_member_of_a_sibling_team_cannot_delegate_the_task() {
    let store = store();
    let pool = pool_over(sibling_spec(), &store, DelegationRules::default());
    let task = root(&pool, "gamma", "t").await;
    let refused = pool
        .delegate(&task.task_id, "beta::team_leader", "gamma", None)
        .await
        .unwrap();
    assert!(!refused.is_ok());
    assert!(
        refused
            .reason()
            .unwrap()
            .contains("is owned by team 'gamma'"),
        "{refused:?}"
    );
}

// -----------------------------------------------------------------------------------------------
// Reads
// -----------------------------------------------------------------------------------------------

#[tokio::test]
async fn open_task_count_counts_open_tasks_of_one_target_team() {
    let store = store();
    let pool = pool(&store);
    assert_eq!(pool.open_task_count("alpha").await.unwrap(), 0);
    let first = root(&pool, "alpha", "first").await;
    let second = root(&pool, "alpha", "second").await;
    pool.create(OrgTaskSpec::new("alpha", "third", "c").for_team("beta"))
        .await
        .unwrap();
    assert_eq!(pool.open_task_count("alpha").await.unwrap(), 2);
    assert_eq!(pool.open_task_count("beta").await.unwrap(), 1);

    assert!(claim_alpha(&pool, &first.task_id).await.is_ok());
    assert!(pool
        .complete(&first.task_id, "alpha", "dev-1", "done")
        .await
        .unwrap()
        .is_ok());
    assert_eq!(pool.open_task_count("alpha").await.unwrap(), 1);
    assert!(claim_alpha(&pool, &second.task_id).await.is_ok());
    assert!(pool
        .fail(
            &second.task_id,
            "alpha",
            "dev-1",
            OrgTaskFailure::new("timeout", "no reply")
        )
        .await
        .unwrap()
        .is_ok());
    assert_eq!(pool.open_task_count("alpha").await.unwrap(), 0);
    assert_eq!(pool.open_task_count("beta").await.unwrap(), 1);
}

#[tokio::test]
async fn list_filters_by_status_team_parent_and_limit() {
    let store = store();
    let pool = pool(&store);
    let parent = root(&pool, "alpha", "parent").await;
    let child = pool
        .create(OrgTaskSpec::new("alpha", "child", "c").under(parent.task_id.clone()))
        .await
        .unwrap()
        .task;
    let other = pool
        .create(OrgTaskSpec::new("alpha", "other", "c").for_team("beta"))
        .await
        .unwrap()
        .task;
    pool.create(OrgTaskSpec::new("alpha", "newest", "c"))
        .await
        .unwrap();

    let everything = pool.list(&OrgTaskFilter::default()).await.unwrap();
    assert_eq!(everything.len(), 4);
    // The store's order: `updated_at` descending, then `task_id` ascending.
    assert!(everything.windows(2).all(|pair| {
        pair[0].updated_at > pair[1].updated_at
            || (pair[0].updated_at == pair[1].updated_at && pair[0].task_id < pair[1].task_id)
    }));

    let limited = pool
        .list(&OrgTaskFilter {
            limit: Some(2),
            ..OrgTaskFilter::default()
        })
        .await
        .unwrap();
    assert_eq!(limited.len(), 2);

    let beta = pool
        .list(&OrgTaskFilter {
            team: Some("beta".to_owned()),
            ..OrgTaskFilter::default()
        })
        .await
        .unwrap();
    assert_eq!(beta.len(), 1);
    assert_eq!(beta[0].task_id, other.task_id);

    let blocked = pool
        .list(&OrgTaskFilter {
            status: Some(TaskStatus::Blocked),
            ..OrgTaskFilter::default()
        })
        .await
        .unwrap();
    assert_eq!(blocked.len(), 1);
    assert_eq!(blocked[0].task_id, parent.task_id);

    let children = pool.children(&parent.task_id).await.unwrap();
    assert_eq!(children.len(), 1);
    assert_eq!(children[0].task_id, child.task_id);
    assert_eq!(
        pool.children("task_missing").await.unwrap(),
        Vec::<OrgTask>::new()
    );

    assert!(claim_alpha(&pool, &child.task_id).await.is_ok());
    let by_member = pool
        .list(&OrgTaskFilter {
            team: Some("alpha".to_owned()),
            assignee: Some("dev-1".to_owned()),
            ..OrgTaskFilter::default()
        })
        .await
        .unwrap();
    assert_eq!(by_member.len(), 1);
    assert_eq!(by_member[0].task_id, child.task_id);
    let by_qualified = pool
        .list(&OrgTaskFilter {
            assignee: Some(qualify_member("alpha", "dev-1")),
            ..OrgTaskFilter::default()
        })
        .await
        .unwrap();
    assert_eq!(by_qualified.len(), 1);
}

#[tokio::test]
async fn list_applies_the_limit_after_post_filters() {
    let store = store();
    let pool = pool(&store);
    for index in 0..4 {
        pool.create(OrgTaskSpec::new("alpha", format!("task {index}"), "c"))
            .await
            .unwrap();
    }
    pool.create(OrgTaskSpec::new("alpha", "beta task", "c").for_team("beta"))
        .await
        .unwrap();
    let alpha = pool
        .list(&OrgTaskFilter {
            team: Some("alpha".to_owned()),
            limit: Some(2),
            ..OrgTaskFilter::default()
        })
        .await
        .unwrap();
    assert_eq!(alpha.len(), 2);
    assert!(alpha.iter().all(|task| task.target_team == "alpha"));
}

#[tokio::test]
async fn the_pool_and_a_team_board_share_a_store_without_interference() {
    let store = store();
    let pool = pool(&store);
    let board = TaskBoard::new(
        store.clone(),
        "alpha",
        TaskBoardLimits::default().with_max_subtask_depth(1),
    );
    let board_task = board
        .create_task(TaskGraphSpec {
            title: "board task".to_owned(),
            content: "c".to_owned(),
            ..TaskGraphSpec::default()
        })
        .await
        .unwrap()
        .task;
    let org_task = root(&pool, "alpha", "org task").await;

    assert_eq!(board.view(&TaskFilter::default()).await.unwrap().len(), 1);
    assert_eq!(pool.list(&OrgTaskFilter::default()).await.unwrap().len(), 1);
    assert!(board_task.task_id.starts_with("task_"));
    assert_ne!(board_task.task_id, org_task.task_id);
    assert!(pool.namespace().starts_with("org::"));
    assert_eq!(board.team_name(), "alpha");
    assert_eq!(
        pool.list(&OrgTaskFilter::default()).await.unwrap()[0].task_id,
        org_task.task_id
    );
}

#[tokio::test]
async fn a_task_exposes_its_assignee_parts() {
    let store = store();
    let pool = pool(&store);
    let task = root(&pool, "alpha", "t").await;
    assert_eq!(task.assignee_parts(), None);
    assert_eq!(split_qualified("alpha::dev-1"), Some(("alpha", "dev-1")));
    let claimed = claim_alpha(&pool, &task.task_id).await.task.unwrap();
    assert_eq!(claimed.assignee_parts(), Some(("alpha", "dev-1")));
    assert!(claimed.is_open());
}

// -----------------------------------------------------------------------------------------------
// Wire shapes
// -----------------------------------------------------------------------------------------------

#[test]
fn specs_round_trip_with_defaults_and_unknown_fields() {
    let spec: OrgTaskSpec = serde_json::from_str(
        r#"{"title":"t","owner_team":"alpha","unknown_field":42,"metadata":{"k":"v"}}"#,
    )
    .unwrap();
    assert_eq!(spec.target_team, None);
    assert_eq!(spec.reviewers, Vec::<String>::new());
    assert_eq!(spec.created_by, None);
    assert_eq!(spec.metadata["k"], serde_json::json!("v"));
    let encoded = serde_json::to_value(&spec).unwrap();
    assert_eq!(encoded["owner_team"], serde_json::json!("alpha"));
    assert_eq!(
        serde_json::from_value::<OrgTaskSpec>(encoded).unwrap(),
        spec
    );
    assert_eq!(
        serde_json::from_str::<OrgTaskSpec>("{}").unwrap(),
        OrgTaskSpec::default()
    );
}

#[tokio::test]
async fn tasks_round_trip_through_json() {
    let store = store();
    let pool = pool(&store);
    let task = pool
        .create(OrgTaskSpec::new("alpha", "t", "c").for_team("beta"))
        .await
        .unwrap()
        .task;
    let encoded = serde_json::to_string(&task).unwrap();
    let decoded: OrgTask = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, task);
    let partial: OrgTask = serde_json::from_str(r#"{"task_id":"task_1"}"#).unwrap();
    assert_eq!(partial.status, TaskStatus::Pending);
    assert_eq!(partial.organization_id, "");
}

#[test]
fn events_serialize_with_a_type_tag() {
    let claimed = OrgTaskEvent::Claimed {
        task_id: "task_1".to_owned(),
        assignee: "alpha::dev-1".to_owned(),
    };
    assert_eq!(
        serde_json::to_value(&claimed).unwrap(),
        serde_json::json!({
            "type": "claimed",
            "task_id": "task_1",
            "assignee": "alpha::dev-1",
        })
    );
    let unblocked = OrgTaskEvent::Unblocked {
        task_id: "task_2".to_owned(),
    };
    assert_eq!(
        serde_json::to_value(&unblocked).unwrap(),
        serde_json::json!({"type": "unblocked", "task_id": "task_2"})
    );
    let delegated = OrgTaskEvent::Delegated {
        task_id: "task_3".to_owned(),
        from_team: "alpha".to_owned(),
        to_team: "beta".to_owned(),
        to_member: None,
    };
    let encoded = serde_json::to_value(&delegated).unwrap();
    assert_eq!(encoded["type"], serde_json::json!("delegated"));
    assert_eq!(encoded["to_member"], serde_json::Value::Null);
    assert_eq!(
        serde_json::from_value::<OrgTaskEvent>(encoded).unwrap(),
        delegated
    );
}

#[test]
fn mutations_round_trip_through_json() {
    let task = OrgTask {
        task_id: "task_1".to_owned(),
        status: TaskStatus::InProgress,
        ..OrgTask::default()
    };
    let mutation = OrgTaskMutation {
        task,
        events: vec![OrgTaskEvent::Unblocked {
            task_id: "task_2".to_owned(),
        }],
        unblocked: vec!["task_2".to_owned()],
    };
    let encoded = serde_json::to_value(&mutation).unwrap();
    assert_eq!(encoded["unblocked"], serde_json::json!(["task_2"]));
    assert_eq!(encoded["events"][0]["type"], serde_json::json!("unblocked"));
    assert_eq!(
        serde_json::from_value::<OrgTaskMutation>(encoded).unwrap(),
        mutation
    );
}

#[test]
fn failures_and_sidecars_round_trip() {
    let failure = OrgTaskFailure::new("timeout", "no reply");
    let encoded = serde_json::to_value(&failure).unwrap();
    assert_eq!(
        encoded,
        serde_json::json!({"code": "timeout", "message": "no reply"})
    );
    assert_eq!(
        serde_json::from_value::<OrgTaskFailure>(encoded).unwrap(),
        failure
    );
    assert_eq!(
        serde_json::from_str::<OrgTaskFailure>("{}").unwrap(),
        OrgTaskFailure::default()
    );

    let sidecar = OrgTaskSidecar {
        owner_team: "alpha".to_owned(),
        target_team: "beta".to_owned(),
        depth: 2,
        failure: Some(failure),
        metadata: BTreeMap::from([("k".to_owned(), serde_json::json!(1))]),
    };
    let encoded = serde_json::to_string(&sidecar).unwrap();
    assert_eq!(
        serde_json::from_str::<OrgTaskSidecar>(&encoded).unwrap(),
        sidecar
    );
    assert_eq!(
        serde_json::from_str::<OrgTaskSidecar>(r#"{"depth":3,"unknown":true}"#).unwrap(),
        OrgTaskSidecar {
            depth: 3,
            ..OrgTaskSidecar::default()
        }
    );
}

#[test]
fn op_results_report_success_and_refusal() {
    let refused = OrgTaskOpResult::failed("nope");
    assert!(!refused.is_ok());
    assert_eq!(refused.reason(), Some("nope"));
    assert_eq!(refused.task, None);
    assert_eq!(
        serde_json::to_value(&refused).unwrap(),
        serde_json::json!({"ok": false, "reason": "nope", "task": null})
    );
    let from_error = OrgTaskOpResult::from(Result::<()>::Err(Error::validation("bad")));
    assert!(!from_error.is_ok());
    assert_eq!(from_error.reason(), Some("validation: bad"));
    let from_ok = OrgTaskOpResult::from(Result::<()>::Ok(()));
    assert!(from_ok.is_ok());
    assert_eq!(from_ok.task, None);
}
