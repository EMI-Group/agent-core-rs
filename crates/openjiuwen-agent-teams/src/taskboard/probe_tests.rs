//! Settle-probe suite: the cost of one settle poll, and the settle point of the adaptive tracker.
//!
//! The board is wrapped in a counting [`TeamStore`] so the assertions are about *store traffic*,
//! not wall-clock time: a board of fifty tasks must cost the same three calls as a board of one, and
//! the tracker must still settle on the second poll that observes a settled board.
//! No sleeps: the tracker's schedule is read from [`SettleTracker::interval`] instead of being
//! measured.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_common::error::Result;

use crate::runtime::settle::SettleTracker;
use crate::schema::{MemberStatus, ReviewVote, TaskFilter, TaskPatch, TaskRecord, TaskStatus};
use crate::storage::memory::InMemoryTeamStore;
use crate::storage::records::{MemberRecord, MessageRecord, TeamInfoRecord};
use crate::storage::{MemberPatch, TeamStore};

use super::{TaskBoard, TaskBoardLimits};

/// Team every test works under.
const TEAM: &str = "team";

/// Store wrapper that counts the calls a board read makes.
///
/// Every method bumps `calls`; the four read methods the settle poll can use are also counted
/// separately, so a test can say *which* query is unbounded instead of only that something grew.
struct CountingStore {
    /// The backend every call is delegated to.
    inner: InMemoryTeamStore,
    /// Every trait method call, in total.
    calls: AtomicU64,
    /// `list_tasks` calls.
    list_tasks: AtomicU64,
    /// `dependencies_many` calls.
    dependencies_many: AtomicU64,
    /// per-task `dependencies` calls.
    dependencies: AtomicU64,
    /// `list_members` calls.
    list_members: AtomicU64,
}

impl CountingStore {
    /// Wraps a fresh in-memory backend.
    fn new() -> Self {
        Self {
            inner: InMemoryTeamStore::new(),
            calls: AtomicU64::new(0),
            list_tasks: AtomicU64::new(0),
            dependencies_many: AtomicU64::new(0),
            dependencies: AtomicU64::new(0),
            list_members: AtomicU64::new(0),
        }
    }

    /// Every trait method call so far.
    fn calls(&self) -> u64 {
        self.calls.load(Ordering::Acquire)
    }

    /// `list_tasks` calls so far.
    fn list_tasks(&self) -> u64 {
        self.list_tasks.load(Ordering::Acquire)
    }

    /// `dependencies_many` calls so far.
    fn dependencies_many(&self) -> u64 {
        self.dependencies_many.load(Ordering::Acquire)
    }

    /// per-task `dependencies` calls so far.
    fn dependencies(&self) -> u64 {
        self.dependencies.load(Ordering::Acquire)
    }

    /// `list_members` calls so far.
    fn list_members(&self) -> u64 {
        self.list_members.load(Ordering::Acquire)
    }

    /// Resets every counter, so a test can measure one poll at a time.
    fn reset(&self) {
        self.calls.store(0, Ordering::Release);
        self.list_tasks.store(0, Ordering::Release);
        self.dependencies_many.store(0, Ordering::Release);
        self.dependencies.store(0, Ordering::Release);
        self.list_members.store(0, Ordering::Release);
    }
}

#[async_trait]
impl TeamStore for CountingStore {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    async fn upsert_team(&self, record: TeamInfoRecord) -> Result<()> {
        self.calls.fetch_add(1, Ordering::AcqRel);
        self.inner.upsert_team(record).await
    }

    async fn team(&self, team_name: &str) -> Result<Option<TeamInfoRecord>> {
        self.calls.fetch_add(1, Ordering::AcqRel);
        self.inner.team(team_name).await
    }

    async fn insert_member(&self, record: MemberRecord) -> Result<()> {
        self.calls.fetch_add(1, Ordering::AcqRel);
        self.inner.insert_member(record).await
    }

    async fn member(&self, team_name: &str, member_name: &str) -> Result<Option<MemberRecord>> {
        self.calls.fetch_add(1, Ordering::AcqRel);
        self.inner.member(team_name, member_name).await
    }

    async fn list_members(&self, team_name: &str) -> Result<Vec<MemberRecord>> {
        self.calls.fetch_add(1, Ordering::AcqRel);
        self.list_members.fetch_add(1, Ordering::AcqRel);
        self.inner.list_members(team_name).await
    }

    async fn update_member(
        &self,
        team_name: &str,
        member_name: &str,
        patch: &MemberPatch,
    ) -> Result<bool> {
        self.calls.fetch_add(1, Ordering::AcqRel);
        self.inner
            .update_member(team_name, member_name, patch)
            .await
    }

    async fn transition_member(
        &self,
        team_name: &str,
        member_name: &str,
        from: MemberStatus,
        to: MemberStatus,
    ) -> Result<bool> {
        self.calls.fetch_add(1, Ordering::AcqRel);
        self.inner
            .transition_member(team_name, member_name, from, to)
            .await
    }

    async fn insert_task(&self, record: TaskRecord) -> Result<()> {
        self.calls.fetch_add(1, Ordering::AcqRel);
        self.inner.insert_task(record).await
    }

    async fn task(&self, team_name: &str, task_id: &str) -> Result<Option<TaskRecord>> {
        self.calls.fetch_add(1, Ordering::AcqRel);
        self.inner.task(team_name, task_id).await
    }

    async fn list_tasks(&self, team_name: &str, filter: &TaskFilter) -> Result<Vec<TaskRecord>> {
        self.calls.fetch_add(1, Ordering::AcqRel);
        self.list_tasks.fetch_add(1, Ordering::AcqRel);
        self.inner.list_tasks(team_name, filter).await
    }

    async fn update_task(&self, team_name: &str, task_id: &str, patch: &TaskPatch) -> Result<bool> {
        self.calls.fetch_add(1, Ordering::AcqRel);
        self.inner.update_task(team_name, task_id, patch).await
    }

    async fn transition_task(
        &self,
        team_name: &str,
        task_id: &str,
        from: TaskStatus,
        to: TaskStatus,
    ) -> Result<bool> {
        self.calls.fetch_add(1, Ordering::AcqRel);
        self.inner
            .transition_task(team_name, task_id, from, to)
            .await
    }

    async fn add_dependency(&self, team_name: &str, task_id: &str, depends_on: &str) -> Result<()> {
        self.calls.fetch_add(1, Ordering::AcqRel);
        self.inner
            .add_dependency(team_name, task_id, depends_on)
            .await
    }

    async fn dependencies(&self, team_name: &str, task_id: &str) -> Result<Vec<String>> {
        self.calls.fetch_add(1, Ordering::AcqRel);
        self.dependencies.fetch_add(1, Ordering::AcqRel);
        self.inner.dependencies(team_name, task_id).await
    }

    async fn dependencies_many(
        &self,
        team_name: &str,
        task_ids: &[String],
    ) -> Result<Vec<(String, Vec<String>)>> {
        self.calls.fetch_add(1, Ordering::AcqRel);
        self.dependencies_many.fetch_add(1, Ordering::AcqRel);
        self.inner.dependencies_many(team_name, task_ids).await
    }

    async fn dependents(&self, team_name: &str, task_id: &str) -> Result<Vec<String>> {
        self.calls.fetch_add(1, Ordering::AcqRel);
        self.inner.dependents(team_name, task_id).await
    }

    async fn insert_vote(&self, record: ReviewVote) -> Result<()> {
        self.calls.fetch_add(1, Ordering::AcqRel);
        self.inner.insert_vote(record).await
    }

    async fn votes(&self, team_name: &str, task_id: &str, round: u32) -> Result<Vec<ReviewVote>> {
        self.calls.fetch_add(1, Ordering::AcqRel);
        self.inner.votes(team_name, task_id, round).await
    }

    async fn insert_message(&self, record: MessageRecord) -> Result<()> {
        self.calls.fetch_add(1, Ordering::AcqRel);
        self.inner.insert_message(record).await
    }

    async fn unread_messages(
        &self,
        team_name: &str,
        member_name: &str,
    ) -> Result<Vec<MessageRecord>> {
        self.calls.fetch_add(1, Ordering::AcqRel);
        self.inner.unread_messages(team_name, member_name).await
    }

    async fn mark_read(
        &self,
        team_name: &str,
        member_name: &str,
        message_ids: &[String],
    ) -> Result<()> {
        self.calls.fetch_add(1, Ordering::AcqRel);
        self.inner
            .mark_read(team_name, member_name, message_ids)
            .await
    }

    async fn read_watermark(&self, team_name: &str, member_name: &str) -> Result<Option<u64>> {
        self.calls.fetch_add(1, Ordering::AcqRel);
        self.inner.read_watermark(team_name, member_name).await
    }

    async fn set_read_watermark(
        &self,
        team_name: &str,
        member_name: &str,
        timestamp: u64,
    ) -> Result<()> {
        self.calls.fetch_add(1, Ordering::AcqRel);
        self.inner
            .set_read_watermark(team_name, member_name, timestamp)
            .await
    }

    async fn message_count(&self, team_name: &str, member_name: &str) -> Result<usize> {
        self.calls.fetch_add(1, Ordering::AcqRel);
        self.inner.message_count(team_name, member_name).await
    }
}

/// A board over `counted`, with the default limits.
fn board(counted: &Arc<CountingStore>) -> TaskBoard {
    let store: Arc<dyn TeamStore> = Arc::clone(counted) as Arc<dyn TeamStore>;
    TaskBoard::new(store, TEAM, TaskBoardLimits::default())
}

/// One pending task row.
fn task_row(task_id: &str, status: TaskStatus) -> TaskRecord {
    TaskRecord {
        status,
        ..TaskRecord::new(TEAM, task_id, format!("Task {task_id}"), "body")
    }
}

/// Seeds `count` pending tasks, each depending on the previous one.
async fn seed_chain(store: &dyn TeamStore, count: usize) {
    for index in 0..count {
        let task_id = format!("task-{index:02}");
        store
            .insert_task(task_row(&task_id, TaskStatus::Pending))
            .await
            .unwrap();
        if index > 0 {
            store
                .add_dependency(TEAM, &task_id, &format!("task-{:02}", index - 1))
                .await
                .unwrap();
        }
    }
}

#[tokio::test]
async fn a_settle_probe_costs_three_store_calls_regardless_of_task_count() {
    let counted = Arc::new(CountingStore::new());
    seed_chain(counted.as_ref(), 50).await;
    let board = board(&counted);
    counted.reset();

    let probe = board.settle_probe().await.unwrap();
    assert_eq!(probe.tasks, 50);
    assert!(!probe.settled, "an open task keeps the board unsettled");
    assert_eq!(counted.calls(), 3, "one poll, three store calls");
    assert_eq!(counted.list_tasks(), 1);
    assert_eq!(counted.dependencies_many(), 1);
    assert_eq!(
        counted.dependencies(),
        0,
        "the per-task dependency read must not be used by a whole-board read"
    );
    assert_eq!(counted.list_members(), 1);

    // A second poll costs the same: the per-poll cost is bounded, not amortized.
    counted.reset();
    let again = board.settle_probe().await.unwrap();
    assert_eq!(again, probe, "an unchanged board reads the same probe");
    assert_eq!(counted.calls(), 3);
}

#[tokio::test]
async fn a_changed_row_changes_the_probe_fingerprint() {
    let counted = Arc::new(CountingStore::new());
    seed_chain(counted.as_ref(), 3).await;
    let board = board(&counted);

    let before = board.settle_probe().await.unwrap();
    assert!(!before.settled);
    counted
        .insert_task(task_row("task-01", TaskStatus::Completed))
        .await
        .unwrap();
    let after = board.settle_probe().await.unwrap();
    assert!(!after.settled, "two tasks are still open");
    assert_ne!(
        after.fingerprint, before.fingerprint,
        "a status change must move the digest the backoff watches"
    );
    assert_eq!(after.tasks, before.tasks);
}

#[tokio::test]
async fn the_backoff_settles_two_polls_after_the_board_settles() {
    let counted = Arc::new(CountingStore::new());
    seed_chain(counted.as_ref(), 1).await;
    let board = board(&counted);
    let mut tracker =
        SettleTracker::with_intervals(Duration::from_millis(10), Duration::from_millis(40));

    // The one task stays open: the poll backs off and the run never settles.
    counted.reset();
    let mut polls = 0;
    let mut intervals = Vec::new();
    for _ in 0..4 {
        let probe = board.settle_probe().await.unwrap();
        polls += 1;
        assert_eq!(counted.calls(), 3 * polls, "one bounded read per poll");
        assert!(
            !tracker.poll(true, probe),
            "an open task keeps the run open"
        );
        intervals.push(tracker.interval());
    }
    assert_eq!(
        intervals,
        vec![
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(40),
            Duration::from_millis(40)
        ],
        "a still board backs off up to the ceiling"
    );

    // The work completes: the settle point is the second poll after the change, exactly as with a
    // fixed interval, and the confirming poll runs on the fast interval.
    counted
        .insert_task(task_row("task-00", TaskStatus::Completed))
        .await
        .unwrap();
    counted.reset();
    polls = 0;
    let probe = board.settle_probe().await.unwrap();
    polls += 1;
    assert!(probe.settled, "the board settled");
    assert_eq!(counted.calls(), 3 * polls);
    assert!(!tracker.poll(true, probe), "one observation is not enough");
    assert_eq!(tracker.interval(), Duration::from_millis(10));
    let probe = board.settle_probe().await.unwrap();
    polls += 1;
    assert!(tracker.poll(true, probe), "the second observation settles");
    assert_eq!(
        counted.calls(),
        3 * polls,
        "the settling polls cost the same bounded read"
    );
}
