//! A full in-process team run over the SQLite `TeamStore`, reloaded from disk afterwards.
//!
//! Everything the runtime writes goes through the injected [`TeamStore`] seam, so pointing that
//! seam at `StorageSpec { kind: "sqlite", params: { path: … } }` turns one ordinary run into a
//! persisted one. This suite drives the public API exactly like a host would (build → start →
//! invoke → task board → mail → spawn → stop) and then opens a **second** store over the same
//! database file to prove the rows are on disk, not in a process-local cache.
//!
//! What is pinned:
//!
//! 1. `open_store` selects the persistent backend for a `sqlite` spec (`TeamStore::name()` is
//!    `"sqlite"`) and a freshly opened store answers the team row the build wrote, including the
//!    persisted leader name and the stamped `created`/`updated_at` columns.
//! 2. The roster survives a close: `list_members` returns the leader, the spawned teammate (its
//!    row settled on `ready`) and the never-spawned roster member (`unstarted`), with roles,
//!    display names, descriptions and prompts intact; the leader's own row settles on `shut_down`
//!    after `TeamAgent::stop`.
//! 3. The task board survives: task rows (status, assignee, closing summary), the dependency edge
//!    written when a second task was created against a pending first one (`blocked`), the
//!    `blocked → pending` promotion performed by completing the first task, and the
//!    `status`/`assignee` filters of `list_tasks`.
//! 4. The mailbox survives: a direct row, a broadcast row and a `mark_read` flag are all still
//!    there, and the per-member broadcast watermark is persisted — a member whose watermark covers
//!    the broadcast reloads with an empty inbox, the sender never reads its own broadcast, and
//!    `mark_read` on a direct id never advances the watermark.
//! 5. The schema is reusable: a second (and third) `open_store` over the same file returns
//!    *identical* rows, never a reset board, and the reopened database still accepts writes.
//!
//! Everything is offline and deterministic: the `support` fakes answer every member input with one
//! scripted observation triple, the database lives in a `tempfile::TempDir`, and every wait is
//! bounded by [`support::BOUND`].
#![cfg(feature = "sqlite")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::collections::BTreeMap;
use std::sync::Arc;

use openjiuwen_agent_teams::harness::MemberHarnessFactory;
use openjiuwen_agent_teams::runtime::{SpecBuild, TeamAgent, TeamBuildDeps};
use openjiuwen_agent_teams::schema::{
    DispatchMode, MemberStatus, StorageSpec, TaskFilter, TaskGraphSpec, TaskPatch, TaskRecord,
    TaskStatus, TeamAgentSpec, TeamRole,
};
use openjiuwen_agent_teams::storage::{open_store, TeamStore};
use openjiuwen_agent_teams::workspace::WorkspaceProvider;
use serde_json::json;
use tempfile::TempDir;
use tokio::time::timeout;

use support::{wait_until, FakeHarnessFactory, FakeWorkspaceProvider, BOUND};

/// Team name every blueprint in this suite uses.
const TEAM: &str = "alpha";
/// Leader member name of [`support::sample_spec`].
const LEADER: &str = "team_leader";
/// The predefined teammate the roster starts with.
const DEV: &str = "dev-1";
/// The teammate derived from `agents["teammate"]`.
const TEAMMATE: &str = "teammate";

/// One live team whose only store is a SQLite database file.
///
/// The `TempDir` keeps the database (and its WAL files) alive for the whole test; [`Self::close`]
/// stops the team and drops every handle on the database so the next `open_store` is a real
/// reopen, not a second reader of a still-open pool.
struct SqliteRun {
    /// Directory holding `team.db`; dropped when the run is dropped.
    dir: TempDir,
    /// The same `StorageSpec` a host would put on the blueprint, reused for every reopen.
    storage: StorageSpec,
    /// The store the runtime was built over.
    store: Arc<dyn TeamStore>,
    /// Leader handle built over [`Self::store`].
    agent: Arc<TeamAgent>,
}

impl SqliteRun {
    /// Builds and starts `sample_spec` with its `storage` pointed at a fresh tempdir database.
    async fn start() -> Self {
        let dir = TempDir::new().expect("a team-database temp dir");
        let path = dir.path().join("team.db");
        let storage = StorageSpec {
            kind: "sqlite".to_owned(),
            params: BTreeMap::from([(
                "path".to_owned(),
                json!(path.to_string_lossy().into_owned()),
            )]),
        };
        let store = open_store(&storage).expect("the sqlite store opens");
        assert_eq!(
            store.name(),
            "sqlite",
            "a `sqlite` spec selects the persistent backend"
        );

        let factory = FakeHarnessFactory::new();
        let workspace = FakeWorkspaceProvider::new();
        let deps = TeamBuildDeps::default()
            .with_harness_factory(Arc::clone(&factory) as Arc<dyn MemberHarnessFactory>)
            .with_store(Arc::clone(&store))
            .with_workspace(Arc::clone(&workspace) as Arc<dyn WorkspaceProvider>);
        let spec = sample_sqlite_spec(storage.clone());
        let agent = spec.build_with(deps).await.expect("the team builds");
        agent.start().await.expect("the team starts");
        Self {
            dir,
            storage,
            store,
            agent,
        }
    }

    /// Stops the team and drops every handle on the database.
    ///
    /// Returns the tempdir (so the caller keeps the database on disk) and the `StorageSpec` used
    /// for the reopen.
    async fn close(self) -> (TempDir, StorageSpec) {
        self.agent.stop().await.expect("the team stops");
        let Self {
            dir,
            storage,
            store,
            agent,
        } = self;
        drop(store);
        drop(agent);
        (dir, storage)
    }
}

/// Builds the sample blueprint with its storage pointed at `storage`.
fn sample_sqlite_spec(storage: StorageSpec) -> TeamAgentSpec {
    TeamAgentSpec {
        storage,
        ..support::sample_spec()
    }
}

/// Returns the member names of `records`, in the order the store reports them.
fn member_names(records: &[openjiuwen_agent_teams::storage::MemberRecord]) -> Vec<&str> {
    records
        .iter()
        .map(|record| record.member_name.as_str())
        .collect()
}

/// Returns the task ids of `records`, in the order the store reports them.
fn task_ids(records: &[TaskRecord]) -> Vec<&str> {
    records
        .iter()
        .map(|record| record.task_id.as_str())
        .collect()
}

/// Returns the message contents of `records`, sorted so timestamp ties cannot flake the assertion.
fn contents(records: &[openjiuwen_agent_teams::storage::MessageRecord]) -> Vec<&str> {
    let mut contents: Vec<&str> = records
        .iter()
        .map(|record| record.content.as_str())
        .collect();
    contents.sort_unstable();
    contents
}

#[tokio::test]
async fn a_full_run_persists_the_team_row_and_the_roster() {
    let run = SqliteRun::start().await;

    // One ordinary run: the leader answers through its scripted harness.
    let output = run
        .agent
        .invoke(json!("ship the release"), None)
        .await
        .unwrap();
    assert_eq!(output.output, "done: team_leader");

    // Spawn the predefined teammate so its row moves past `unstarted`.
    timeout(BOUND, run.agent.spawn_member(DEV))
        .await
        .expect("the spawn did not finish")
        .expect("dev-1 did not spawn");
    wait_until("dev-1 to be persisted ready", || {
        run.agent
            .backend()
            .member(DEV)
            .is_some_and(|member| member.status() == MemberStatus::Ready)
    })
    .await;

    let (_dir, storage) = run.close().await;
    let store = open_store(&storage).expect("a reload of the same database");

    let team = store
        .team(TEAM)
        .await
        .unwrap()
        .expect("the team row is persisted");
    assert_eq!(team.team_name, TEAM);
    assert_eq!(team.leader_member_name, LEADER);
    assert_eq!(team.dispatch_mode, DispatchMode::Autonomous);
    assert!(!team.enable_task_verification);
    assert_eq!(
        team.prompt, "lead the team",
        "the leader prompt is persisted"
    );
    assert_eq!(team.display_name_or_default(), TEAM);
    assert!(team.created > 0, "the store stamped the creation: {team:?}");
    assert!(
        team.updated_at >= team.created,
        "the write stamp follows the creation stamp: {team:?}"
    );

    let members = store.list_members(TEAM).await.unwrap();
    assert_eq!(
        member_names(&members),
        [DEV, LEADER, TEAMMATE],
        "the roster is sorted and complete"
    );

    let leader = members
        .iter()
        .find(|record| record.member_name == LEADER)
        .expect("the leader row");
    assert_eq!(leader.role, TeamRole::Leader);
    assert_eq!(
        leader.status,
        MemberStatus::ShutDown,
        "stopping the team settles the leader's lifecycle row"
    );
    assert_eq!(leader.prompt, "lead the team");

    let dev = members
        .iter()
        .find(|record| record.member_name == DEV)
        .expect("the dev-1 row");
    assert_eq!(dev.role, TeamRole::Teammate);
    assert_eq!(dev.status, MemberStatus::Ready, "the spawn settled dev-1");
    assert_eq!(dev.display_name, "Dev One");
    assert_eq!(dev.desc, "the developer");
    assert_eq!(dev.prompt, "you are dev-1");

    let teammate = members
        .iter()
        .find(|record| record.member_name == TEAMMATE)
        .expect("the derived teammate row");
    assert_eq!(teammate.role, TeamRole::Teammate);
    assert_eq!(
        teammate.status,
        MemberStatus::Unstarted,
        "an unspawned roster member keeps its admission row"
    );

    // The single-row accessor agrees with the roster snapshot.
    assert_eq!(
        store.member(TEAM, DEV).await.unwrap().as_ref(),
        Some(dev),
        "the row survives a keyed lookup"
    );
}

#[tokio::test]
async fn task_rows_dependencies_and_filters_survive_a_reopen() {
    let run = SqliteRun::start().await;

    let first = run
        .agent
        .create_task(TaskGraphSpec {
            title: "first".to_owned(),
            content: "do the first thing".to_owned(),
            task_id: Some("task-a".to_owned()),
            assignee: Some(DEV.to_owned()),
            ..TaskGraphSpec::default()
        })
        .await
        .unwrap();
    assert_eq!(first.status, TaskStatus::Pending);
    assert_eq!(first.assignee.as_deref(), Some(DEV));

    let second = run
        .agent
        .backend()
        .tasks()
        .create_task(TaskGraphSpec {
            title: "second".to_owned(),
            content: "after the first".to_owned(),
            task_id: Some("task-b".to_owned()),
            depends_on: vec!["task-a".to_owned()],
            assignee: Some(TEAMMATE.to_owned()),
            ..TaskGraphSpec::default()
        })
        .await
        .unwrap()
        .task;
    assert_eq!(
        second.status,
        TaskStatus::Blocked,
        "a pending dependency parks the dependent"
    );

    let claimed = run
        .agent
        .backend()
        .tasks()
        .claim("task-a", DEV)
        .await
        .unwrap();
    assert!(claimed.ok, "{claimed:?}");
    let completed = run
        .agent
        .backend()
        .tasks()
        .complete("task-a", DEV, "shipped")
        .await
        .unwrap();
    assert_eq!(completed.task.status, TaskStatus::Completed);
    assert_eq!(
        completed.unblocked,
        vec!["task-b".to_owned()],
        "completing the first task promotes its dependent"
    );

    // What the live store holds, before the reload.
    assert_eq!(
        run.store.dependencies(TEAM, "task-b").await.unwrap(),
        vec!["task-a".to_owned()]
    );

    let (_dir, storage) = run.close().await;
    let store = open_store(&storage).expect("a reload of the same database");

    assert_reloaded_task_board(store.as_ref()).await;
}

/// Asserts the task board a reloaded store reports: rows, promotion, filters and edges.
async fn assert_reloaded_task_board(store: &dyn TeamStore) {
    let tasks = store
        .list_tasks(TEAM, &TaskFilter::default())
        .await
        .unwrap();
    // Order is `updated_at DESC, task_id ASC`; two writes inside the same millisecond tie-break by
    // id, so the exact order is pinned by the store's own unit suite (explicit stamps) and here the
    // suite only asserts that both rows survived.
    let mut ids = task_ids(&tasks);
    ids.sort_unstable();
    assert_eq!(ids, ["task-a", "task-b"], "both task rows are on disk");

    let first = store.task(TEAM, "task-a").await.unwrap().expect("task-a");
    assert_eq!(first.status, TaskStatus::Completed);
    assert_eq!(first.assignee.as_deref(), Some(DEV));
    assert_eq!(first.summary.as_deref(), Some("shipped"));
    assert_eq!(first.title, "first");
    assert_eq!(first.content, "do the first thing");

    let second = store.task(TEAM, "task-b").await.unwrap().expect("task-b");
    assert_eq!(
        second.status,
        TaskStatus::Pending,
        "the promotion is persisted, not replayed on open"
    );
    assert_eq!(second.assignee.as_deref(), Some(TEAMMATE));

    let completed_rows = store
        .list_tasks(
            TEAM,
            &TaskFilter {
                status: Some(TaskStatus::Completed),
                ..TaskFilter::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(task_ids(&completed_rows), ["task-a"]);
    let pending_rows = store
        .list_tasks(
            TEAM,
            &TaskFilter {
                status: Some(TaskStatus::Pending),
                ..TaskFilter::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(task_ids(&pending_rows), ["task-b"]);
    let assignee_rows = store
        .list_tasks(
            TEAM,
            &TaskFilter {
                assignee: Some(TEAMMATE.to_owned()),
                ..TaskFilter::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(task_ids(&assignee_rows), ["task-b"]);

    assert_eq!(
        store.dependencies(TEAM, "task-b").await.unwrap(),
        vec!["task-a".to_owned()],
        "the dependency edge is a row of its own"
    );
    assert_eq!(
        store.dependents(TEAM, "task-a").await.unwrap(),
        vec!["task-b".to_owned()]
    );
}

#[tokio::test]
async fn mailbox_rows_and_the_broadcast_watermark_survive_a_reopen() {
    let run = SqliteRun::start().await;

    let direct = run.agent.send_message(Some(DEV), "hello").await.unwrap();
    assert_eq!(direct.from_member_name, LEADER);
    assert_eq!(direct.to_member_name.as_deref(), Some(DEV));
    assert!(!direct.is_broadcast());

    let broadcast = run.agent.send_message(None, "all hands").await.unwrap();
    assert!(broadcast.is_broadcast());
    assert_eq!(broadcast.to_member_name, None);
    assert_eq!(broadcast.from_member_name, LEADER);

    // A second direct row is consumed through the row flag, to prove `mark_read` is persisted too.
    let consumed = run
        .agent
        .send_message(Some(DEV), "already read")
        .await
        .unwrap();
    run.agent
        .backend()
        .mark_read(DEV, std::slice::from_ref(&consumed.message_id))
        .await
        .unwrap();

    // The broadcast read cursor is the per-member watermark; cover it for the second teammate.
    run.store
        .set_read_watermark(TEAM, TEAMMATE, broadcast.timestamp)
        .await
        .unwrap();

    assert_eq!(
        contents(&run.agent.backend().unread(DEV).await.unwrap()),
        ["all hands", "hello"],
        "the consumed direct row is gone before the reload"
    );
    assert!(
        run.agent
            .backend()
            .unread(TEAMMATE)
            .await
            .unwrap()
            .is_empty(),
        "the watermark covers the broadcast"
    );

    let (_dir, storage) = run.close().await;
    let store = open_store(&storage).expect("a reload of the same database");

    let unread = store.unread_messages(TEAM, DEV).await.unwrap();
    assert_eq!(
        contents(&unread),
        ["all hands", "hello"],
        "the direct and broadcast rows are on disk"
    );
    assert_eq!(
        store.message_count(TEAM, DEV).await.unwrap(),
        2,
        "the consumed row is not counted again"
    );
    assert!(
        unread
            .iter()
            .all(|row| row.message_id != consumed.message_id),
        "the persisted read flag hides the consumed row: {unread:?}"
    );
    let direct_row = unread
        .iter()
        .find(|row| row.message_id == direct.message_id)
        .expect("the direct row is still unread");
    assert_eq!(direct_row.content, "hello");
    assert!(!direct_row.is_read);
    let broadcast_row = unread
        .iter()
        .find(|row| row.message_id == broadcast.message_id)
        .expect("the broadcast row is still unread");
    assert!(broadcast_row.is_broadcast());
    assert!(
        !broadcast_row.is_read,
        "a broadcast never carries a row flag"
    );

    assert!(
        store
            .unread_messages(TEAM, TEAMMATE)
            .await
            .unwrap()
            .is_empty(),
        "the persisted watermark still covers the broadcast"
    );
    assert_eq!(
        store.read_watermark(TEAM, TEAMMATE).await.unwrap(),
        Some(broadcast.timestamp),
        "the watermark is a persisted number, not a row flag"
    );
    assert_eq!(
        store.read_watermark(TEAM, DEV).await.unwrap(),
        None,
        "`mark_read` never advances the broadcast watermark"
    );
    assert!(
        store
            .unread_messages(TEAM, LEADER)
            .await
            .unwrap()
            .is_empty(),
        "a sender never reads its own broadcast"
    );
}

#[tokio::test]
async fn reopening_twice_keeps_every_row_and_the_database_stays_writable() {
    let run = SqliteRun::start().await;

    run.agent
        .create_task(TaskGraphSpec {
            title: "persisted".to_owned(),
            content: "write it down".to_owned(),
            task_id: Some("task-a".to_owned()),
            assignee: Some(DEV.to_owned()),
            ..TaskGraphSpec::default()
        })
        .await
        .unwrap();
    run.agent
        .backend()
        .tasks()
        .claim("task-a", DEV)
        .await
        .unwrap();
    run.agent
        .backend()
        .tasks()
        .complete("task-a", DEV, "shipped")
        .await
        .unwrap();
    timeout(BOUND, run.agent.spawn_member(DEV))
        .await
        .expect("the spawn did not finish")
        .expect("dev-1 did not spawn");
    run.agent.send_message(Some(DEV), "hello").await.unwrap();

    let (_dir, storage) = run.close().await;

    let first = open_store(&storage).expect("the first reload");
    let team = first.team(TEAM).await.unwrap();
    let members = first.list_members(TEAM).await.unwrap();
    let tasks = first
        .list_tasks(TEAM, &TaskFilter::default())
        .await
        .unwrap();
    let unread = first.unread_messages(TEAM, DEV).await.unwrap();
    drop(first);

    let second = open_store(&storage).expect("the second reload");
    assert_eq!(
        second.team(TEAM).await.unwrap(),
        team,
        "opening the schema again never resets the team row"
    );
    assert_eq!(
        second.list_members(TEAM).await.unwrap(),
        members,
        "the roster is byte-identical after a second reopen"
    );
    assert_eq!(
        second
            .list_tasks(TEAM, &TaskFilter::default())
            .await
            .unwrap(),
        tasks,
        "the task board is byte-identical after a second reopen"
    );
    assert_eq!(
        second.unread_messages(TEAM, DEV).await.unwrap(),
        unread,
        "the mailbox is byte-identical after a second reopen"
    );

    // The reopened database is writable, not just readable.
    let patched = second
        .update_task(TEAM, "task-a", &TaskPatch::new().with_note("reopened"))
        .await
        .unwrap();
    assert!(patched, "the patch applied to the reloaded row");
    let task = second.task(TEAM, "task-a").await.unwrap().expect("task-a");
    assert_eq!(task.note.as_deref(), Some("reopened"));
    assert_eq!(
        task.status,
        TaskStatus::Completed,
        "the status is untouched"
    );
}
