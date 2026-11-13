//! What only a real database can show: every row kind survives a close and a reopen.

use std::path::Path;

use openjiuwen_common::error::ErrorKind;

use crate::schema::{
    DispatchMode, MemberStatus, ReviewVote, TaskFilter, TaskStatus, TeamRole, Verdict,
};

use crate::storage::records::{MemberRecord, MessageRecord, TeamInfoRecord};
use crate::storage::{SqliteTeamStore, TeamStore};

use super::support::{ids, task, task_ids};

/// Writes one of every row kind into the database at `path`, then closes it.
async fn seed_rows(path: &Path) -> (TeamInfoRecord, MemberRecord, MemberRecord) {
    let mut team = TeamInfoRecord::new("alpha");
    team.display_name = "Alpha team".to_owned();
    team.desc = "alpha desc".to_owned();
    team.dispatch_mode = DispatchMode::Scheduled;
    team.enable_task_verification = true;
    team.created = 111;
    team.updated_at = 222;

    let mut lead = MemberRecord::new("alpha", "team-leader", TeamRole::Leader);
    lead.display_name = "Team Leader".to_owned();
    lead.desc = "drives the team".to_owned();
    lead.prompt = "dispatch".to_owned();
    lead.model_ref = Some("model-a".to_owned());
    lead.worktree = Some("/ws/leader".to_owned());
    lead.updated_at = 10;

    let mut dev = MemberRecord::new("alpha", "dev-1", TeamRole::Teammate);
    dev.status = MemberStatus::Ready;
    dev.updated_at = 20;

    let store = SqliteTeamStore::open(path).await.unwrap();
    store.upsert_team(team.clone()).await.unwrap();
    store.insert_member(lead.clone()).await.unwrap();
    store.insert_member(dev.clone()).await.unwrap();
    for row in [
        task("t-1", 100, TaskStatus::Pending, Some("dev-1"), None),
        task("t-2", 300, TaskStatus::Pending, Some("dev-2"), None),
        task("t-3", 200, TaskStatus::InProgress, Some("dev-1"), None),
    ] {
        store.insert_task(row).await.unwrap();
    }
    store.add_dependency("alpha", "t-1", "t-2").await.unwrap();
    store.add_dependency("alpha", "t-1", "t-0").await.unwrap();
    for (round, reviewer, verdict) in [
        (0, "rev-b", Verdict::Fail),
        (0, "rev-a", Verdict::Pass),
        (1, "rev-a", Verdict::Fail),
    ] {
        store
            .insert_vote(ReviewVote {
                team_name: "alpha".to_owned(),
                task_id: "t-1".to_owned(),
                round,
                reviewer: reviewer.to_owned(),
                verdict,
            })
            .await
            .unwrap();
    }
    store
        .insert_message(MessageRecord::direct(
            "alpha",
            "m-1",
            "leader",
            "dev-1",
            "do the thing",
            10,
        ))
        .await
        .unwrap();
    store
        .insert_message(MessageRecord::broadcast(
            "alpha",
            "m-2",
            "leader",
            "team standup",
            12,
        ))
        .await
        .unwrap();
    store
        .set_read_watermark("alpha", "dev-1", 11)
        .await
        .unwrap();
    drop(store);

    (team, lead, dev)
}

#[tokio::test]
async fn every_row_kind_survives_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("team.db");
    let (team, lead, dev) = seed_rows(&path).await;

    let store = SqliteTeamStore::open(&path).await.unwrap();

    // Team row, including the `desc` -> `description` column mapping and the created stamp.
    assert_eq!(store.team("alpha").await.unwrap(), Some(team));
    // Roster, including both optional columns and the direct read of the lifecycle status.
    assert_eq!(
        store.list_members("alpha").await.unwrap(),
        vec![dev.clone(), lead.clone()],
        "roster sorted by member name"
    );
    assert_eq!(
        store.member("alpha", "team-leader").await.unwrap(),
        Some(lead)
    );
    assert_mailbox(&store).await;
    assert_task_board(&store).await;

    // The reopened database is writable, not just readable.
    assert!(store
        .transition_task("alpha", "t-1", TaskStatus::Pending, TaskStatus::InProgress)
        .await
        .unwrap());
    assert_eq!(
        store.task("alpha", "t-1").await.unwrap().unwrap().status,
        TaskStatus::InProgress
    );
}

/// Task rows, ordering, filters and dependency edges after the reopen.
async fn assert_task_board(store: &dyn TeamStore) {
    assert_eq!(
        task_ids(
            &store
                .list_tasks("alpha", &TaskFilter::default())
                .await
                .unwrap()
        ),
        vec!["t-2", "t-3", "t-1"]
    );
    assert_eq!(
        task_ids(
            &store
                .list_tasks(
                    "alpha",
                    &TaskFilter {
                        status: Some(TaskStatus::Pending),
                        ..TaskFilter::default()
                    },
                )
                .await
                .unwrap()
        ),
        vec!["t-2", "t-1"]
    );
    assert_eq!(
        task_ids(
            &store
                .list_tasks(
                    "alpha",
                    &TaskFilter {
                        assignee: Some("dev-1".to_owned()),
                        limit: Some(1),
                        ..TaskFilter::default()
                    },
                )
                .await
                .unwrap()
        ),
        vec!["t-3"]
    );
    let t1 = store.task("alpha", "t-1").await.unwrap().unwrap();
    assert_eq!(t1.updated_at, 100);
    assert_eq!(t1.assignee.as_deref(), Some("dev-1"));
    assert_eq!(
        store.dependencies("alpha", "t-1").await.unwrap(),
        vec!["t-0".to_owned(), "t-2".to_owned()]
    );
    assert_eq!(
        store.dependents("alpha", "t-2").await.unwrap(),
        vec!["t-1".to_owned()]
    );
}

/// Votes and the mailbox, including the per-member watermark, after the reopen.
async fn assert_mailbox(store: &dyn TeamStore) {
    let round0 = store.votes("alpha", "t-1", 0).await.unwrap();
    assert_eq!(round0.len(), 2);
    assert_eq!(round0[0].reviewer, "rev-a");
    assert_eq!(round0[0].verdict, Verdict::Pass);
    assert_eq!(round0[1].verdict, Verdict::Fail);
    let round1 = store.votes("alpha", "t-1", 1).await.unwrap();
    assert_eq!(round1.len(), 1);
    assert_eq!(round1[0].verdict, Verdict::Fail);

    assert_eq!(
        ids(&store.unread_messages("alpha", "dev-1").await.unwrap()),
        vec!["m-1", "m-2"]
    );
    assert_eq!(store.message_count("alpha", "dev-1").await.unwrap(), 2);
    assert_eq!(
        store.read_watermark("alpha", "dev-1").await.unwrap(),
        Some(11)
    );
    // A member with no watermark still sees the broadcast.
    assert_eq!(
        ids(&store.unread_messages("alpha", "dev-2").await.unwrap()),
        vec!["m-2"]
    );
}

#[tokio::test]
async fn an_empty_file_database_reads_as_empty() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fresh.db");
    let store = SqliteTeamStore::open(&path).await.unwrap();
    assert_eq!(store.team("alpha").await.unwrap(), None);
    assert_eq!(store.list_members("alpha").await.unwrap().len(), 0);
    assert_eq!(
        store
            .list_tasks("alpha", &TaskFilter::default())
            .await
            .unwrap()
            .len(),
        0
    );
    assert_eq!(
        store.unread_messages("alpha", "dev-1").await.unwrap().len(),
        0
    );
    assert_eq!(store.read_watermark("alpha", "dev-1").await.unwrap(), None);
    assert_eq!(store.message_count("alpha", "dev-1").await.unwrap(), 0);
    // The empty path is rejected before anything is created on disk.
    let err = SqliteTeamStore::open("").await.expect_err("empty path");
    assert_eq!(err.kind(), ErrorKind::Validation);
    assert!(err.to_string().contains("non-empty"));
}
