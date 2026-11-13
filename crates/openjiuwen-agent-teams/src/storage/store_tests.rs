//! Store suite: row stamps, compare-and-set transitions, patches, filters, graph edges, votes and
//! the mailbox watermark.
//!
//! Everything runs against [`InMemoryTeamStore`] through the `dyn TeamStore` seam, so the same
//! assertions apply to a future backend. No sleeps: timestamps are either caller-supplied or
//! checked for monotonic growth.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use openjiuwen_common::error::ErrorKind;

use crate::schema::{
    DispatchMode, MemberMode, MemberStatus, ReviewVote, StorageSpec, TaskFilter, TaskPatch,
    TaskRecord, TaskStatus, TeamRole, Verdict,
};

use super::memory::InMemoryTeamStore;
use super::records::{MemberPatch, MemberRecord, MessageRecord, TeamInfoRecord};
use super::{open_store, TeamStore};

fn store() -> Arc<dyn TeamStore> {
    Arc::new(InMemoryTeamStore::new())
}

fn task(
    task_id: &str,
    updated_at: u64,
    status: TaskStatus,
    assignee: Option<&str>,
    parent: Option<&str>,
) -> TaskRecord {
    TaskRecord {
        status,
        assignee: assignee.map(str::to_owned),
        parent_task_id: parent.map(str::to_owned),
        updated_at,
        ..TaskRecord::new("alpha", task_id, format!("title {task_id}"), "body")
    }
}

fn ids(messages: &[MessageRecord]) -> Vec<&str> {
    messages
        .iter()
        .map(|message| message.message_id.as_str())
        .collect()
}

fn task_ids(tasks: &[TaskRecord]) -> Vec<&str> {
    tasks.iter().map(|task| task.task_id.as_str()).collect()
}

#[tokio::test]
async fn team_row_upsert_preserves_created_and_stamps() {
    let store = store();
    let mut row = TeamInfoRecord::new("alpha");
    row.display_name = "Alpha team".to_owned();
    row.dispatch_mode = DispatchMode::Scheduled;
    row.enable_task_verification = true;
    row.created = 111;
    row.updated_at = 222;
    store.upsert_team(row.clone()).await.unwrap();

    assert_eq!(store.team("alpha").await.unwrap(), Some(row));
    assert_eq!(store.team("missing").await.unwrap(), None);

    let mut replacement = TeamInfoRecord::new("alpha");
    replacement.desc = "second write".to_owned();
    store.upsert_team(replacement).await.unwrap();
    let stored = store.team("alpha").await.unwrap().unwrap();
    assert_eq!(stored.created, 111);
    assert_eq!(stored.desc, "second write");
    assert!(stored.updated_at > 222);
    // An untouched field is replaced, not merged: upsert_team writes the whole row.
    assert_eq!(stored.display_name, "");
}

#[tokio::test]
async fn team_row_without_stamps_gets_store_stamps() {
    let store = store();
    store
        .upsert_team(TeamInfoRecord::new("alpha"))
        .await
        .unwrap();
    let stored = store.team("alpha").await.unwrap().unwrap();
    assert!(stored.created > 0);
    assert!(stored.updated_at > 0);
}

#[tokio::test]
async fn empty_row_keys_are_rejected() {
    let store = store();
    let err = store
        .upsert_team(TeamInfoRecord::default())
        .await
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Validation);
    assert!(err.to_string().contains("team_name must not be empty"));

    let err = store
        .insert_member(MemberRecord::new("alpha", "", TeamRole::Teammate))
        .await
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Validation);
    assert!(err.to_string().contains("member_name must not be empty"));

    let err = store
        .insert_task(TaskRecord::new("alpha", "", "t", "c"))
        .await
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Validation);
    assert!(err.to_string().contains("task_id must not be empty"));

    let err = store
        .insert_message(MessageRecord::broadcast("alpha", "", "leader", "hi", 1))
        .await
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Validation);
    assert!(err.to_string().contains("message_id must not be empty"));
}

#[tokio::test]
async fn member_status_cas_transitions() {
    let store = store();
    let mut row = MemberRecord::new("alpha", "dev-1", TeamRole::Teammate);
    row.updated_at = 10;
    store.insert_member(row).await.unwrap();

    assert!(store
        .transition_member(
            "alpha",
            "dev-1",
            MemberStatus::Unstarted,
            MemberStatus::Starting
        )
        .await
        .unwrap());
    let stamped = store.member("alpha", "dev-1").await.unwrap().unwrap();
    assert!(stamped.updated_at > 10);

    // Replaying the same compare-and-set is a stale `from`: the member already moved on.
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

    // ready -> dormant is not in MEMBER_TRANSITIONS.
    assert!(!store
        .transition_member("alpha", "dev-1", MemberStatus::Ready, MemberStatus::Dormant)
        .await
        .unwrap());
    // A legal edge with a stale `from` also fails.
    assert!(!store
        .transition_member("alpha", "dev-1", MemberStatus::Busy, MemberStatus::Ready)
        .await
        .unwrap());
    // Unknown member.
    assert!(!store
        .transition_member(
            "alpha",
            "ghost",
            MemberStatus::Unstarted,
            MemberStatus::Starting
        )
        .await
        .unwrap());

    let row = store.member("alpha", "dev-1").await.unwrap().unwrap();
    assert_eq!(row.status, MemberStatus::Ready);
}

#[tokio::test]
async fn member_patch_merges_clears_and_stamps() {
    let store = store();
    let mut row = MemberRecord::new("alpha", "dev-1", TeamRole::Teammate);
    row.model_ref = Some("model-a".to_owned());
    row.worktree = Some("/ws/dev-1".to_owned());
    row.updated_at = 50;
    store.insert_member(row).await.unwrap();

    let patch = MemberPatch::new()
        .with_status(MemberStatus::Ready)
        .with_mode(MemberMode::PlanMode)
        .with_prompt("you build things")
        .clearing_worktree();
    assert!(store.update_member("alpha", "dev-1", &patch).await.unwrap());

    let row = store.member("alpha", "dev-1").await.unwrap().unwrap();
    assert_eq!(row.status, MemberStatus::Ready);
    assert_eq!(row.mode, MemberMode::PlanMode);
    assert_eq!(row.prompt, "you build things");
    assert_eq!(row.worktree, None);
    assert_eq!(row.model_ref.as_deref(), Some("model-a"));
    assert!(row.updated_at > 50);

    // An empty patch on a known member is a successful no-op; on an unknown member it is `false`.
    let before = row.updated_at;
    assert!(store
        .update_member("alpha", "dev-1", &MemberPatch::new())
        .await
        .unwrap());
    assert_eq!(
        store
            .member("alpha", "dev-1")
            .await
            .unwrap()
            .unwrap()
            .updated_at,
        before
    );
    assert!(!store.update_member("alpha", "ghost", &patch).await.unwrap());
}

#[tokio::test]
async fn members_are_listed_per_team_and_sorted() {
    let store = store();
    for name in ["zeta", "dev-1", "alpha"] {
        store
            .insert_member(MemberRecord::new("alpha", name, TeamRole::Teammate))
            .await
            .unwrap();
    }
    store
        .insert_member(MemberRecord::new("beta", "other", TeamRole::Teammate))
        .await
        .unwrap();

    let names: Vec<String> = store
        .list_members("alpha")
        .await
        .unwrap()
        .into_iter()
        .map(|row| row.member_name)
        .collect();
    assert_eq!(names, vec!["alpha", "dev-1", "zeta"]);
    assert_eq!(store.list_members("missing").await.unwrap().len(), 0);
}

#[tokio::test]
async fn task_status_cas_and_stamps() {
    let store = store();
    let mut row = TaskRecord::new("alpha", "t1", "Title", "Body");
    row.updated_at = 100;
    store.insert_task(row).await.unwrap();

    assert!(store
        .transition_task("alpha", "t1", TaskStatus::Pending, TaskStatus::InProgress)
        .await
        .unwrap());
    let row = store.task("alpha", "t1").await.unwrap().unwrap();
    assert_eq!(row.status, TaskStatus::InProgress);
    assert!(row.updated_at > 100);

    // Stale `from`.
    assert!(!store
        .transition_task("alpha", "t1", TaskStatus::Pending, TaskStatus::Blocked)
        .await
        .unwrap());
    assert!(store
        .transition_task("alpha", "t1", TaskStatus::InProgress, TaskStatus::Completed)
        .await
        .unwrap());
    // completed is terminal.
    assert!(!store
        .transition_task("alpha", "t1", TaskStatus::Completed, TaskStatus::InProgress)
        .await
        .unwrap());
    // Unknown task.
    assert!(!store
        .transition_task("alpha", "ghost", TaskStatus::Pending, TaskStatus::Planning)
        .await
        .unwrap());
}

#[tokio::test]
async fn task_update_bumps_revision_and_records_change_notes() {
    let store = store();
    let mut row = TaskRecord::new("alpha", "t1", "Title", "Body");
    row.updated_at = 500;
    store.insert_task(row).await.unwrap();

    assert!(store
        .update_task("alpha", "t1", &TaskPatch::new().with_content("Body 2"))
        .await
        .unwrap());
    let row = store.task("alpha", "t1").await.unwrap().unwrap();
    assert_eq!(row.content, "Body 2");
    assert_eq!(row.revision, 1);
    assert_eq!(row.change_notes, vec!["r1: content updated".to_owned()]);
    assert!(row.updated_at > 500);

    assert!(store
        .update_task(
            "alpha",
            "t1",
            &TaskPatch::new()
                .with_title("New title")
                .with_content("Body 3")
        )
        .await
        .unwrap());
    let row = store.task("alpha", "t1").await.unwrap().unwrap();
    assert_eq!(row.revision, 2);
    assert_eq!(
        row.change_notes,
        vec![
            "r1: content updated".to_owned(),
            "r2: title and content updated".to_owned()
        ]
    );

    // Writing the same content again is not an edit.
    assert!(store
        .update_task("alpha", "t1", &TaskPatch::new().with_content("Body 3"))
        .await
        .unwrap());
    assert_eq!(
        store.task("alpha", "t1").await.unwrap().unwrap().revision,
        2
    );

    // A non-edit patch leaves the revision alone.
    assert!(store
        .update_task(
            "alpha",
            "t1",
            &TaskPatch::new()
                .with_note("waiting on review")
                .with_assignee("dev-1")
        )
        .await
        .unwrap());
    let row = store.task("alpha", "t1").await.unwrap().unwrap();
    assert_eq!(row.revision, 2);
    assert_eq!(row.note.as_deref(), Some("waiting on review"));
    assert_eq!(row.assignee.as_deref(), Some("dev-1"));

    // An explicit revision wins over the automatic bump and records no note.
    assert!(store
        .update_task(
            "alpha",
            "t1",
            &TaskPatch::new().with_content("Body 4").with_revision(9)
        )
        .await
        .unwrap());
    let row = store.task("alpha", "t1").await.unwrap().unwrap();
    assert_eq!(row.revision, 9);
    assert_eq!(row.change_notes.len(), 2);

    // clear_assignee wins over with_assignee.
    assert!(store
        .update_task(
            "alpha",
            "t1",
            &TaskPatch::new().with_assignee("dev-2").clearing_assignee()
        )
        .await
        .unwrap());
    assert_eq!(
        store.task("alpha", "t1").await.unwrap().unwrap().assignee,
        None
    );

    // Unknown task.
    assert!(!store
        .update_task("alpha", "ghost", &TaskPatch::new().with_title("x"))
        .await
        .unwrap());
}

#[tokio::test]
async fn list_tasks_filters_and_orders() {
    let store = store();
    for row in [
        task("t-a", 10, TaskStatus::Pending, Some("dev-1"), None),
        task("t-b", 30, TaskStatus::Pending, Some("dev-2"), None),
        task(
            "t-c",
            20,
            TaskStatus::InProgress,
            Some("dev-1"),
            Some("t-a"),
        ),
        task("t-d", 40, TaskStatus::Pending, None, Some("t-a")),
    ] {
        store.insert_task(row).await.unwrap();
    }
    store
        .insert_task(TaskRecord::new("beta", "t-z", "other team", "body"))
        .await
        .unwrap();

    let all = store
        .list_tasks("alpha", &TaskFilter::default())
        .await
        .unwrap();
    assert_eq!(task_ids(&all), vec!["t-d", "t-b", "t-c", "t-a"]);

    let pending = store
        .list_tasks(
            "alpha",
            &TaskFilter {
                status: Some(TaskStatus::Pending),
                ..TaskFilter::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(task_ids(&pending), vec!["t-d", "t-b", "t-a"]);

    let mine = store
        .list_tasks(
            "alpha",
            &TaskFilter {
                assignee: Some("dev-1".to_owned()),
                ..TaskFilter::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(task_ids(&mine), vec!["t-c", "t-a"]);

    let children = store
        .list_tasks(
            "alpha",
            &TaskFilter {
                parent: Some("t-a".to_owned()),
                ..TaskFilter::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(task_ids(&children), vec!["t-d", "t-c"]);

    let limited = store
        .list_tasks(
            "alpha",
            &TaskFilter {
                limit: Some(2),
                ..TaskFilter::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(task_ids(&limited), vec!["t-d", "t-b"]);

    // A filter matching nothing, and a team with no tasks.
    let none = store
        .list_tasks(
            "alpha",
            &TaskFilter {
                status: Some(TaskStatus::Completed),
                ..TaskFilter::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(none, Vec::new());
    assert_eq!(
        store
            .list_tasks("missing", &TaskFilter::default())
            .await
            .unwrap(),
        Vec::new()
    );
}

#[tokio::test]
async fn dependency_edges_are_idempotent_and_sorted() {
    let store = store();
    store
        .insert_task(TaskRecord::new("alpha", "t-c", "C", "c"))
        .await
        .unwrap();

    store.add_dependency("alpha", "t-c", "t-b").await.unwrap();
    store.add_dependency("alpha", "t-c", "t-a").await.unwrap();
    store.add_dependency("alpha", "t-c", "t-a").await.unwrap();

    assert_eq!(
        store.dependencies("alpha", "t-c").await.unwrap(),
        vec!["t-a".to_owned(), "t-b".to_owned()]
    );
    assert_eq!(
        store.dependents("alpha", "t-a").await.unwrap(),
        vec!["t-c".to_owned()]
    );
    assert_eq!(
        store.dependencies("alpha", "t-a").await.unwrap(),
        Vec::<String>::new()
    );
    assert_eq!(
        store.dependents("alpha", "t-z").await.unwrap(),
        Vec::<String>::new()
    );
    // Edges are team-scoped.
    assert_eq!(
        store.dependencies("beta", "t-c").await.unwrap(),
        Vec::<String>::new()
    );
    assert_eq!(
        store.dependents("beta", "t-a").await.unwrap(),
        Vec::<String>::new()
    );

    let err = store
        .add_dependency("alpha", "t-c", "t-c")
        .await
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Validation);
    assert!(err.to_string().contains("cannot depend on itself"));

    let err = store.add_dependency("alpha", "t-c", "").await.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Validation);
    assert!(err.to_string().contains("depends_on must not be empty"));
}

#[tokio::test]
async fn votes_are_isolated_per_round_and_replaced_per_reviewer() {
    let store = store();
    let vote = |round: u32, reviewer: &str, verdict: Verdict| ReviewVote {
        team_name: "alpha".to_owned(),
        task_id: "t1".to_owned(),
        round,
        reviewer: reviewer.to_owned(),
        verdict,
    };

    store
        .insert_vote(vote(0, "rev-b", Verdict::Pass))
        .await
        .unwrap();
    store
        .insert_vote(vote(0, "rev-a", Verdict::Fail))
        .await
        .unwrap();
    store
        .insert_vote(vote(1, "rev-a", Verdict::Pass))
        .await
        .unwrap();

    let round0 = store.votes("alpha", "t1", 0).await.unwrap();
    assert_eq!(round0.len(), 2);
    assert_eq!(round0[0].reviewer, "rev-a");
    assert_eq!(round0[0].verdict, Verdict::Fail);

    let round1 = store.votes("alpha", "t1", 1).await.unwrap();
    assert_eq!(round1.len(), 1);
    assert_eq!(round1[0].verdict, Verdict::Pass);

    // A reviewer changing their mind replaces the vote instead of appending.
    store
        .insert_vote(vote(0, "rev-a", Verdict::Pass))
        .await
        .unwrap();
    let round0 = store.votes("alpha", "t1", 0).await.unwrap();
    assert_eq!(round0.len(), 2);
    assert_eq!(round0[0].verdict, Verdict::Pass);

    assert_eq!(store.votes("alpha", "t1", 2).await.unwrap(), Vec::new());
    assert_eq!(store.votes("alpha", "t2", 0).await.unwrap(), Vec::new());
}

/// Seeds one direct message per member, a leader broadcast and a self-broadcast by `dev-1`.
async fn seeded_mailbox() -> Arc<dyn TeamStore> {
    let store = store();
    store
        .insert_message(MessageRecord::direct(
            "alpha", "m-d20", "leader", "dev-1", "direct", 20,
        ))
        .await
        .unwrap();
    store
        .insert_message(MessageRecord::direct(
            "alpha", "m-d10", "leader", "dev-2", "other", 10,
        ))
        .await
        .unwrap();
    store
        .insert_message(MessageRecord::broadcast(
            "alpha", "m-b15", "leader", "bcast", 15,
        ))
        .await
        .unwrap();
    // A broadcast is never unread for its own sender.
    store
        .insert_message(MessageRecord::broadcast(
            "alpha", "m-b05", "dev-1", "self", 5,
        ))
        .await
        .unwrap();
    store
}

#[tokio::test]
async fn unread_messages_order_by_timestamp() {
    let store = seeded_mailbox().await;
    assert_eq!(
        ids(&store.unread_messages("alpha", "dev-1").await.unwrap()),
        vec!["m-b15", "m-d20"]
    );
    assert_eq!(store.message_count("alpha", "dev-1").await.unwrap(), 2);
    // dev-2 sees its own direct message plus both broadcasts (neither is its own).
    assert_eq!(
        ids(&store.unread_messages("alpha", "dev-2").await.unwrap()),
        vec!["m-b05", "m-d10", "m-b15"]
    );
    // A member with no mailbox of its own still sees every broadcast.
    assert_eq!(
        ids(&store.unread_messages("alpha", "ghost").await.unwrap()),
        vec!["m-b05", "m-b15"]
    );
    // An unknown team yields nothing.
    assert_eq!(
        store.unread_messages("missing", "dev-1").await.unwrap(),
        Vec::new()
    );
    assert_eq!(store.message_count("missing", "dev-1").await.unwrap(), 0);
}

#[tokio::test]
async fn mark_read_flags_direct_rows_only() {
    let store = seeded_mailbox().await;
    store
        .mark_read(
            "alpha",
            "dev-1",
            &[
                "m-b15".to_owned(),
                "m-d20".to_owned(),
                "m-d10".to_owned(),
                "missing".to_owned(),
            ],
        )
        .await
        .unwrap();

    // The broadcast id in the same batch is ignored, so the broadcast stays unread.
    assert_eq!(
        ids(&store.unread_messages("alpha", "dev-1").await.unwrap()),
        vec!["m-b15"]
    );
    // Marking a broadcast never advances the watermark.
    assert_eq!(store.read_watermark("alpha", "dev-1").await.unwrap(), None);
    // Another member's direct row is untouched.
    assert!(store
        .unread_messages("alpha", "dev-2")
        .await
        .unwrap()
        .iter()
        .any(|message| message.message_id == "m-d10"));
}

#[tokio::test]
async fn watermark_covers_broadcasts_and_never_regresses() {
    let store = seeded_mailbox().await;
    store
        .mark_read("alpha", "dev-1", &["m-d20".to_owned()])
        .await
        .unwrap();

    // The watermark covers every broadcast at or below its timestamp.
    store
        .set_read_watermark("alpha", "dev-1", 15)
        .await
        .unwrap();
    assert_eq!(
        store.unread_messages("alpha", "dev-1").await.unwrap(),
        Vec::new()
    );
    assert_eq!(
        store.read_watermark("alpha", "dev-1").await.unwrap(),
        Some(15)
    );

    // The watermark never moves backwards.
    store.set_read_watermark("alpha", "dev-1", 7).await.unwrap();
    assert_eq!(
        store.read_watermark("alpha", "dev-1").await.unwrap(),
        Some(15)
    );

    // Only a strictly newer broadcast outruns the watermark.
    store
        .insert_message(MessageRecord::broadcast(
            "alpha", "m-b16", "leader", "later", 16,
        ))
        .await
        .unwrap();
    store
        .insert_message(MessageRecord::broadcast(
            "alpha", "m-b15b", "leader", "same", 15,
        ))
        .await
        .unwrap();
    assert_eq!(
        ids(&store.unread_messages("alpha", "dev-1").await.unwrap()),
        vec!["m-b16"]
    );
    assert_eq!(store.message_count("alpha", "dev-1").await.unwrap(), 1);
}

#[tokio::test]
async fn message_insert_is_keyed_by_message_id() {
    let store = store();
    store
        .insert_message(MessageRecord::direct(
            "alpha", "m1", "leader", "dev-1", "first", 10,
        ))
        .await
        .unwrap();
    store
        .insert_message(MessageRecord::direct(
            "alpha",
            "m1",
            "leader",
            "dev-1",
            "rewritten",
            11,
        ))
        .await
        .unwrap();
    let unread = store.unread_messages("alpha", "dev-1").await.unwrap();
    assert_eq!(unread.len(), 1);
    assert_eq!(unread[0].content, "rewritten");
}

#[tokio::test]
async fn member_insert_is_keyed_and_replaces_lifecycle_state() {
    let store = store();
    store
        .insert_member(MemberRecord::new("alpha", "dev-1", TeamRole::Teammate))
        .await
        .unwrap();
    assert!(store
        .transition_member(
            "alpha",
            "dev-1",
            MemberStatus::Unstarted,
            MemberStatus::Ready
        )
        .await
        .unwrap());

    let mut replacement = MemberRecord::new("alpha", "dev-1", TeamRole::Teammate);
    replacement.prompt = "new prompt".to_owned();
    replacement.updated_at = 77;
    store.insert_member(replacement).await.unwrap();

    let row = store.member("alpha", "dev-1").await.unwrap().unwrap();
    assert_eq!(row.prompt, "new prompt");
    assert_eq!(row.updated_at, 77);
    assert_eq!(row.status, MemberStatus::Unstarted);
    assert_eq!(store.list_members("alpha").await.unwrap().len(), 1);
}

#[test]
fn open_store_selects_memory_and_rejects_deferred_backends() {
    let store = open_store(&StorageSpec::in_memory()).unwrap();
    assert_eq!(store.name(), "memory");

    // Parameters are accepted silently, and an empty kind falls back to memory.
    let mut with_params = StorageSpec::in_memory();
    with_params
        .params
        .insert("path".to_owned(), serde_json::json!("/tmp/team.db"));
    assert!(open_store(&with_params).is_ok());
    let blank = StorageSpec {
        kind: "  ".to_owned(),
        ..StorageSpec::in_memory()
    };
    assert!(open_store(&blank).is_ok());

    // Without the feature the sqlite backend is a typed deferral; with it, sqlite is a real backend
    // and only the missing-parameter path is exercised by the crate root suite.
    #[cfg(not(feature = "sqlite"))]
    {
        let spec = StorageSpec {
            kind: "sqlite".to_owned(),
            ..StorageSpec::in_memory()
        };
        let err = open_store(&spec)
            .err()
            .expect("sqlite storage requires the feature");
        assert_eq!(err.kind(), ErrorKind::Unsupported);
        assert_eq!(
            err.to_string(),
            "unsupported: storage backend 'sqlite' requires the 'sqlite' feature; \
             enable 'openjiuwen-agent-teams/sqlite'"
        );
    }

    for kind in ["postgresql", "mysql"] {
        let spec = StorageSpec {
            kind: kind.to_owned(),
            ..StorageSpec::in_memory()
        };
        let err = open_store(&spec).err().expect("SQL backends are deferred");
        assert_eq!(err.kind(), ErrorKind::Unsupported);
        assert_eq!(
            err.to_string(),
            format!(
                "unsupported: storage backend '{kind}' is not implemented; \
                 the {kind} driver is not available offline"
            )
        );
    }

    let unknown = StorageSpec {
        kind: "pulsar".to_owned(),
        ..StorageSpec::in_memory()
    };
    let err = open_store(&unknown)
        .err()
        .expect("an unknown backend is rejected");
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    assert!(err.to_string().contains("unknown storage backend 'pulsar'"));
}

#[tokio::test]
async fn open_store_calls_are_independent() {
    let first = open_store(&StorageSpec::in_memory()).unwrap();
    let second = open_store(&StorageSpec::in_memory()).unwrap();
    first
        .insert_member(MemberRecord::new("alpha", "dev-1", TeamRole::Teammate))
        .await
        .unwrap();
    assert_eq!(first.list_members("alpha").await.unwrap().len(), 1);
    assert_eq!(second.list_members("alpha").await.unwrap(), Vec::new());
}
