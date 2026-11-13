//! Concurrency: the conditional single-statement compare-and-set decides exactly one winner.
//!
//! These tests need a real multi-threaded runtime and a shared file database, because that is the
//! only configuration in which two writers can race. The in-memory backend serializes through its
//! `RwLock`, so its race answer is trivially correct; here the database has to be.

use std::sync::Arc;

use crate::schema::{MemberStatus, TaskRecord, TaskStatus, TeamRole};
use crate::storage::records::MemberRecord;
use crate::storage::TeamStore;

use crate::storage::SqliteTeamStore;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_task_claim_has_exactly_one_winner() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(
        SqliteTeamStore::open(dir.path().join("team.db"))
            .await
            .unwrap(),
    );
    store
        .insert_task(TaskRecord::new("alpha", "t1", "Claim me", "body"))
        .await
        .unwrap();

    let first = Arc::clone(&store);
    let second = Arc::clone(&store);
    let (left, right) = tokio::join!(
        tokio::spawn(async move {
            first
                .transition_task("alpha", "t1", TaskStatus::Pending, TaskStatus::InProgress)
                .await
                .unwrap()
        }),
        tokio::spawn(async move {
            second
                .transition_task("alpha", "t1", TaskStatus::Pending, TaskStatus::InProgress)
                .await
                .unwrap()
        }),
    );
    let left = left.unwrap();
    let right = right.unwrap();
    assert_ne!(
        left, right,
        "exactly one claim wins, got {left} and {right}"
    );
    assert_eq!(usize::from(left) + usize::from(right), 1);
    assert_eq!(
        store.task("alpha", "t1").await.unwrap().unwrap().status,
        TaskStatus::InProgress
    );
    // The loser's replay is a stale `from` too.
    assert!(!store
        .transition_task("alpha", "t1", TaskStatus::Pending, TaskStatus::InProgress)
        .await
        .unwrap());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_member_transition_has_exactly_one_winner() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(
        SqliteTeamStore::open(dir.path().join("team.db"))
            .await
            .unwrap(),
    );
    store
        .insert_member(MemberRecord::new("alpha", "dev-1", TeamRole::Teammate))
        .await
        .unwrap();

    let first = Arc::clone(&store);
    let second = Arc::clone(&store);
    let (left, right) = tokio::join!(
        tokio::spawn(async move {
            first
                .transition_member(
                    "alpha",
                    "dev-1",
                    MemberStatus::Unstarted,
                    MemberStatus::Starting,
                )
                .await
                .unwrap()
        }),
        tokio::spawn(async move {
            second
                .transition_member(
                    "alpha",
                    "dev-1",
                    MemberStatus::Unstarted,
                    MemberStatus::Starting,
                )
                .await
                .unwrap()
        }),
    );
    let left = left.unwrap();
    let right = right.unwrap();
    assert_ne!(
        left, right,
        "exactly one start wins, got {left} and {right}"
    );
    assert_eq!(usize::from(left) + usize::from(right), 1);
    assert_eq!(
        store
            .member("alpha", "dev-1")
            .await
            .unwrap()
            .unwrap()
            .status,
        MemberStatus::Starting
    );
    // The winner's follow-up edge still works; the losing edge stays a stale `from`.
    assert!(!store
        .transition_member(
            "alpha",
            "dev-1",
            MemberStatus::Unstarted,
            MemberStatus::Starting
        )
        .await
        .unwrap());
    assert!(store
        .transition_member(
            "alpha",
            "dev-1",
            MemberStatus::Starting,
            MemberStatus::Ready
        )
        .await
        .unwrap());
}
