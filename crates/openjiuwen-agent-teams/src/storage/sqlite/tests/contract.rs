//! The backend-independent `TeamStore` contract.
//!
//! [`exercise_contract`] runs one representative sequence over the `dyn TeamStore` seam: team rows
//! and stamps, the roster and its compare-and-set lifecycle, the task board with its filters,
//! revisions and transitions, dependency edges, review votes and the mailbox watermark.
//!
//! Both backends run it (see `parity.rs`), so an answer that differs between `InMemoryTeamStore`
//! and `SqliteTeamStore` fails the suite instead of only a backend-specific test. Every expected
//! value is spelled out; nothing here asserts just `is_ok`.

use openjiuwen_common::error::ErrorKind;

use crate::schema::{
    DispatchMode, MemberMode, MemberStatus, ReviewVote, TaskFilter, TaskPatch, TaskRecord,
    TaskStatus, TeamRole, Verdict,
};

use crate::storage::records::{MemberPatch, MemberRecord, MessageRecord, TeamInfoRecord};
use crate::storage::TeamStore;

use super::support::{ids, member_names, task, task_ids};

/// Runs the whole contract against `store`.
pub(super) async fn exercise_contract(store: &dyn TeamStore) {
    assert!(!store.name().is_empty(), "a backend reports its kind");
    contract_teams(store).await;
    contract_member_patches(store).await;
    contract_member_transitions(store).await;
    contract_task_rows(store).await;
    contract_task_filters(store).await;
    contract_task_filter_combinations(store).await;
    contract_task_revisions(store).await;
    contract_task_patch_precedence(store).await;
    contract_task_transitions(store).await;
    contract_dependencies(store).await;
    contract_votes(store).await;
    contract_mailbox_unread(store).await;
    contract_mailbox_watermark(store).await;
}

/// Team row: upsert semantics, `created` preservation, whole-row replacement and stamping.
async fn contract_teams(store: &dyn TeamStore) {
    let mut row = TeamInfoRecord::new("alpha");
    row.display_name = "Alpha team".to_owned();
    row.desc = "alpha desc".to_owned();
    row.prompt = "lead the team".to_owned();
    row.dispatch_mode = DispatchMode::Scheduled;
    row.enable_task_verification = true;
    row.created = 111;
    row.updated_at = 222;
    store.upsert_team(row.clone()).await.unwrap();
    assert_eq!(store.team("alpha").await.unwrap(), Some(row));
    assert_eq!(store.team("missing").await.unwrap(), None);

    // A replacement with `created == 0` keeps the stored creation stamp, is a whole-row
    // replacement (untouched fields go back to their defaults) and gets a newer `updated_at`.
    let mut replacement = TeamInfoRecord::new("alpha");
    replacement.desc = "second write".to_owned();
    store.upsert_team(replacement).await.unwrap();
    let stored = store.team("alpha").await.unwrap().unwrap();
    assert_eq!(stored.created, 111);
    assert_eq!(stored.desc, "second write");
    assert!(stored.updated_at > 222);
    assert_eq!(stored.display_name, "");
    assert_eq!(stored.dispatch_mode, DispatchMode::Autonomous);
    assert!(!stored.enable_task_verification);

    // A record written without stamps gets store stamps, and `created` gets `updated_at` moving.
    store
        .upsert_team(TeamInfoRecord::new("fresh"))
        .await
        .unwrap();
    let fresh = store.team("fresh").await.unwrap().unwrap();
    assert!(fresh.created > 0);
    assert!(fresh.updated_at > fresh.created);

    let err = store
        .upsert_team(TeamInfoRecord::default())
        .await
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Validation);
    assert!(err.to_string().contains("team_name must not be empty"));
}

/// Roster: row round-trip, listing order, patch merge/clear/no-op semantics and the key guards.
async fn contract_member_patches(store: &dyn TeamStore) {
    let mut row = MemberRecord::new("alpha", "dev-1", TeamRole::Teammate);
    row.display_name = "Dev One".to_owned();
    row.desc = "builder".to_owned();
    row.prompt = "build things".to_owned();
    row.model_ref = Some("model-a".to_owned());
    row.worktree = Some("/ws/dev-1".to_owned());
    row.updated_at = 10;
    store.insert_member(row.clone()).await.unwrap();
    assert_eq!(store.member("alpha", "dev-1").await.unwrap(), Some(row));
    assert_eq!(store.member("alpha", "ghost").await.unwrap(), None);

    for name in ["zeta", "alpha"] {
        store
            .insert_member(MemberRecord::new("alpha", name, TeamRole::Teammate))
            .await
            .unwrap();
    }
    assert_eq!(
        member_names(&store.list_members("alpha").await.unwrap()),
        vec!["alpha", "dev-1", "zeta"]
    );
    assert_eq!(store.list_members("missing").await.unwrap().len(), 0);

    // A patch merges, clears through its `clear_*` flag and moves the stamp.
    let patch = MemberPatch::new()
        .with_status(MemberStatus::Ready)
        .with_mode(MemberMode::PlanMode)
        .with_prompt("plan things")
        .clearing_worktree();
    assert!(store.update_member("alpha", "dev-1", &patch).await.unwrap());
    let stored = store.member("alpha", "dev-1").await.unwrap().unwrap();
    assert_eq!(stored.status, MemberStatus::Ready);
    assert_eq!(stored.mode, MemberMode::PlanMode);
    assert_eq!(stored.prompt, "plan things");
    assert_eq!(stored.worktree, None);
    assert_eq!(stored.model_ref.as_deref(), Some("model-a"));
    assert!(stored.updated_at > 10);

    assert!(store
        .update_member("alpha", "dev-1", &MemberPatch::new().clearing_model())
        .await
        .unwrap());
    let stored = store.member("alpha", "dev-1").await.unwrap().unwrap();
    assert_eq!(stored.model_ref, None);
    let stamped = stored.updated_at;

    // An empty patch on a known row succeeds and writes nothing; an unknown row is `false`.
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
        stamped
    );
    assert!(!store.update_member("alpha", "ghost", &patch).await.unwrap());

    let err = store
        .insert_member(MemberRecord::new("alpha", "", TeamRole::Teammate))
        .await
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Validation);
    assert!(err.to_string().contains("member_name must not be empty"));

    let err = store
        .insert_member(MemberRecord::new("", "dev-9", TeamRole::Teammate))
        .await
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Validation);
    assert!(err.to_string().contains("team_name must not be empty"));
}

/// Roster lifecycle: the compare-and-set transition matrix.
async fn contract_member_transitions(store: &dyn TeamStore) {
    let mut row = MemberRecord::new("alpha", "m-cas", TeamRole::Teammate);
    row.updated_at = 20;
    store.insert_member(row).await.unwrap();

    assert!(store
        .transition_member(
            "alpha",
            "m-cas",
            MemberStatus::Unstarted,
            MemberStatus::Starting
        )
        .await
        .unwrap());
    assert!(
        store
            .member("alpha", "m-cas")
            .await
            .unwrap()
            .unwrap()
            .updated_at
            > 20
    );
    // Replaying the same compare-and-set is a stale `from`.
    assert!(!store
        .transition_member(
            "alpha",
            "m-cas",
            MemberStatus::Unstarted,
            MemberStatus::Starting
        )
        .await
        .unwrap());
    assert!(store
        .transition_member(
            "alpha",
            "m-cas",
            MemberStatus::Starting,
            MemberStatus::Ready
        )
        .await
        .unwrap());
    // `ready -> dormant` is not a member transition.
    assert!(!store
        .transition_member("alpha", "m-cas", MemberStatus::Ready, MemberStatus::Dormant)
        .await
        .unwrap());
    // A legal edge with a stale `from`.
    assert!(!store
        .transition_member("alpha", "m-cas", MemberStatus::Busy, MemberStatus::Ready)
        .await
        .unwrap());
    assert!(!store
        .transition_member(
            "alpha",
            "ghost",
            MemberStatus::Unstarted,
            MemberStatus::Starting
        )
        .await
        .unwrap());
    assert_eq!(
        store
            .member("alpha", "m-cas")
            .await
            .unwrap()
            .unwrap()
            .status,
        MemberStatus::Ready
    );
}

/// Task rows: full-column round-trip, listing seeds and the empty-key guards.
async fn contract_task_rows(store: &dyn TeamStore) {
    // A row that sets every persisted column must come back identical.
    let mut rich = TaskRecord::new("gamma", "t-rich", "Rich", "body");
    rich.status = TaskStatus::InReview;
    rich.assignee = Some("dev-1".to_owned());
    rich.reviewers = vec!["rev-a".to_owned(), "rev-b".to_owned()];
    rich.review_round = 2;
    rich.max_review_rounds = 5;
    rich.summary = Some("done".to_owned());
    rich.note = Some("looks good".to_owned());
    rich.parent_task_id = Some("t-parent".to_owned());
    rich.parent_revision = Some(3);
    rich.created_by = Some("leader".to_owned());
    rich.revision = 7;
    rich.observed_revision = 6;
    rich.change_notes = vec!["r7: content updated".to_owned()];
    rich.updated_at = 900;
    store.insert_task(rich.clone()).await.unwrap();
    assert_eq!(store.task("gamma", "t-rich").await.unwrap(), Some(rich));
    assert_eq!(store.task("gamma", "ghost").await.unwrap(), None);

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

    let err = store
        .insert_task(TaskRecord::new("alpha", "", "t", "c"))
        .await
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Validation);
    assert!(err.to_string().contains("task_id must not be empty"));

    let err = store
        .insert_task(TaskRecord::new("", "t-empty-team", "t", "c"))
        .await
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Validation);
    assert!(err.to_string().contains("team_name must not be empty"));
}

/// Task listing: `updated_at DESC, task_id ASC` and every single-field filter.
async fn contract_task_filters(store: &dyn TeamStore) {
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

    let filtered_pending = store
        .list_tasks(
            "alpha",
            &TaskFilter {
                status: Some(TaskStatus::Pending),
                limit: Some(1),
                ..TaskFilter::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(task_ids(&filtered_pending), vec!["t-d"]);
}

/// Task listing: filter combinations, empty results and team scoping.
async fn contract_task_filter_combinations(store: &dyn TeamStore) {
    let combined = store
        .list_tasks(
            "alpha",
            &TaskFilter {
                status: Some(TaskStatus::Pending),
                assignee: Some("dev-1".to_owned()),
                ..TaskFilter::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(task_ids(&combined), vec!["t-a"]);
    let combined_parent = store
        .list_tasks(
            "alpha",
            &TaskFilter {
                status: Some(TaskStatus::Pending),
                parent: Some("t-a".to_owned()),
                limit: Some(1),
                ..TaskFilter::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(task_ids(&combined_parent), vec!["t-d"]);
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
    assert_eq!(none.len(), 0);
    assert_eq!(
        store
            .list_tasks("missing", &TaskFilter::default())
            .await
            .unwrap()
            .len(),
        0
    );
    // Team scoping: `gamma` holds only its own rich row.
    assert_eq!(
        task_ids(
            &store
                .list_tasks("gamma", &TaskFilter::default())
                .await
                .unwrap()
        ),
        vec!["t-rich"]
    );
}

/// Task patches: revision bumps, rendered change notes and the "not an edit" rule.
async fn contract_task_revisions(store: &dyn TeamStore) {
    let mut row = TaskRecord::new("alpha", "t-upd", "Title", "Body");
    row.updated_at = 500;
    store.insert_task(row).await.unwrap();

    assert!(store
        .update_task("alpha", "t-upd", &TaskPatch::new().with_content("Body 2"))
        .await
        .unwrap());
    let stored = store.task("alpha", "t-upd").await.unwrap().unwrap();
    assert_eq!(stored.content, "Body 2");
    assert_eq!(stored.revision, 1);
    assert_eq!(stored.change_notes, vec!["r1: content updated".to_owned()]);
    assert!(stored.updated_at > 500);

    assert!(store
        .update_task(
            "alpha",
            "t-upd",
            &TaskPatch::new()
                .with_title("New title")
                .with_content("Body 3")
        )
        .await
        .unwrap());
    let stored = store.task("alpha", "t-upd").await.unwrap().unwrap();
    assert_eq!(stored.title, "New title");
    assert_eq!(stored.revision, 2);
    assert_eq!(
        stored.change_notes,
        vec![
            "r1: content updated".to_owned(),
            "r2: title and content updated".to_owned(),
        ]
    );

    // Writing the same content is not an edit: no bump, no note, no new stamp.
    let before = stored.updated_at;
    assert!(store
        .update_task("alpha", "t-upd", &TaskPatch::new().with_content("Body 3"))
        .await
        .unwrap());
    let stored = store.task("alpha", "t-upd").await.unwrap().unwrap();
    assert_eq!(stored.revision, 2);
    assert_eq!(stored.change_notes.len(), 2);
    assert_eq!(stored.updated_at, before);
}

/// Task patches: non-edit writes, explicit revisions, `clear_assignee` precedence, unknown rows.
async fn contract_task_patch_precedence(store: &dyn TeamStore) {
    let before = store
        .task("alpha", "t-upd")
        .await
        .unwrap()
        .unwrap()
        .updated_at;
    assert!(store
        .update_task(
            "alpha",
            "t-upd",
            &TaskPatch::new()
                .with_note("waiting on review")
                .with_assignee("dev-1")
        )
        .await
        .unwrap());
    let stored = store.task("alpha", "t-upd").await.unwrap().unwrap();
    assert_eq!(stored.revision, 2);
    assert_eq!(stored.note.as_deref(), Some("waiting on review"));
    assert_eq!(stored.assignee.as_deref(), Some("dev-1"));
    assert!(stored.updated_at > before);

    // An explicit revision wins over the automatic bump and records no note.
    assert!(store
        .update_task(
            "alpha",
            "t-upd",
            &TaskPatch::new().with_content("Body 4").with_revision(9)
        )
        .await
        .unwrap());
    let stored = store.task("alpha", "t-upd").await.unwrap().unwrap();
    assert_eq!(stored.revision, 9);
    assert_eq!(stored.change_notes.len(), 2);

    // `clear_assignee` wins over `assignee`.
    assert!(store
        .update_task(
            "alpha",
            "t-upd",
            &TaskPatch::new().with_assignee("dev-2").clearing_assignee()
        )
        .await
        .unwrap());
    assert_eq!(
        store
            .task("alpha", "t-upd")
            .await
            .unwrap()
            .unwrap()
            .assignee,
        None
    );

    assert!(!store
        .update_task("alpha", "ghost", &TaskPatch::new().with_title("x"))
        .await
        .unwrap());
}

/// Task status transitions: compare-and-set, terminal rejection and unknown rows.
async fn contract_task_transitions(store: &dyn TeamStore) {
    let mut row = TaskRecord::new("alpha", "t-cas", "Claim me", "body");
    row.updated_at = 100;
    store.insert_task(row).await.unwrap();

    assert!(store
        .transition_task(
            "alpha",
            "t-cas",
            TaskStatus::Pending,
            TaskStatus::InProgress
        )
        .await
        .unwrap());
    let stored = store.task("alpha", "t-cas").await.unwrap().unwrap();
    assert_eq!(stored.status, TaskStatus::InProgress);
    assert!(stored.updated_at > 100);
    // A replayed claim is a stale `from`.
    assert!(!store
        .transition_task(
            "alpha",
            "t-cas",
            TaskStatus::Pending,
            TaskStatus::InProgress
        )
        .await
        .unwrap());
    assert!(store
        .transition_task(
            "alpha",
            "t-cas",
            TaskStatus::InProgress,
            TaskStatus::Completed
        )
        .await
        .unwrap());
    // `completed` is terminal.
    assert!(!store
        .transition_task(
            "alpha",
            "t-cas",
            TaskStatus::Completed,
            TaskStatus::InProgress
        )
        .await
        .unwrap());
    assert!(!store
        .transition_task("alpha", "ghost", TaskStatus::Pending, TaskStatus::Planning)
        .await
        .unwrap());
}

/// Dependency edges: idempotence, ordering, team scoping and the key guards.
async fn contract_dependencies(store: &dyn TeamStore) {
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
    assert_eq!(store.dependencies("alpha", "t-a").await.unwrap().len(), 0);
    assert_eq!(store.dependents("alpha", "t-zzz").await.unwrap().len(), 0);
    // Edges are team-scoped.
    assert_eq!(store.dependencies("beta", "t-c").await.unwrap().len(), 0);
    assert_eq!(store.dependents("beta", "t-a").await.unwrap().len(), 0);

    let err = store
        .add_dependency("alpha", "t-c", "t-c")
        .await
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Validation);
    assert!(err.to_string().contains("cannot depend on itself"));

    for (team, task_id, depends_on, expected) in [
        ("", "t-c", "t-a", "team_name must not be empty"),
        ("alpha", "", "t-a", "task_id must not be empty"),
        ("alpha", "t-c", "", "depends_on must not be empty"),
    ] {
        let err = store
            .add_dependency(team, task_id, depends_on)
            .await
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Validation);
        assert!(err.to_string().contains(expected), "{err}");
    }
}

/// Review votes: rounds are isolated and a reviewer's re-vote replaces the earlier one.
async fn contract_votes(store: &dyn TeamStore) {
    let vote = |round: u32, reviewer: &str, verdict: Verdict| ReviewVote {
        team_name: "alpha".to_owned(),
        task_id: "t-vote".to_owned(),
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

    let round0 = store.votes("alpha", "t-vote", 0).await.unwrap();
    assert_eq!(round0.len(), 2);
    assert_eq!(round0[0].reviewer, "rev-a");
    assert_eq!(round0[0].verdict, Verdict::Fail);
    assert_eq!(round0[1].reviewer, "rev-b");
    let round1 = store.votes("alpha", "t-vote", 1).await.unwrap();
    assert_eq!(round1.len(), 1);
    assert_eq!(round1[0].verdict, Verdict::Pass);

    // A reviewer changing their mind replaces the vote instead of appending.
    store
        .insert_vote(vote(0, "rev-a", Verdict::Pass))
        .await
        .unwrap();
    let round0 = store.votes("alpha", "t-vote", 0).await.unwrap();
    assert_eq!(round0.len(), 2);
    assert_eq!(round0[0].verdict, Verdict::Pass);
    assert_eq!(store.votes("alpha", "t-vote", 2).await.unwrap().len(), 0);
    assert_eq!(store.votes("alpha", "t-other", 0).await.unwrap().len(), 0);
}

/// Mailbox: direct rows, broadcast visibility and `mark_read` semantics.
async fn contract_mailbox_unread(store: &dyn TeamStore) {
    for message in [
        MessageRecord::direct("alpha", "m-d20", "leader", "dev-1", "direct", 20),
        MessageRecord::direct("alpha", "m-d10", "leader", "dev-2", "other", 10),
        MessageRecord::broadcast("alpha", "m-b15", "leader", "bcast", 15),
        MessageRecord::broadcast("alpha", "m-b15b", "leader", "same stamp", 15),
        // A broadcast is never unread for its own sender.
        MessageRecord::broadcast("alpha", "m-b05", "dev-1", "self", 5),
    ] {
        store.insert_message(message).await.unwrap();
    }

    // Ordered by timestamp, then id; own broadcasts are excluded, every other broadcast is unread
    // while there is no watermark.
    assert_eq!(
        ids(&store.unread_messages("alpha", "dev-1").await.unwrap()),
        vec!["m-b15", "m-b15b", "m-d20"]
    );
    assert_eq!(store.message_count("alpha", "dev-1").await.unwrap(), 3);
    assert_eq!(
        ids(&store.unread_messages("alpha", "dev-2").await.unwrap()),
        vec!["m-b05", "m-d10", "m-b15", "m-b15b"]
    );
    assert_eq!(store.message_count("alpha", "dev-2").await.unwrap(), 4);
    // A member without a mailbox of its own still sees every broadcast.
    assert_eq!(
        ids(&store.unread_messages("alpha", "ghost").await.unwrap()),
        vec!["m-b05", "m-b15", "m-b15b"]
    );
    assert_eq!(
        store
            .unread_messages("missing", "dev-1")
            .await
            .unwrap()
            .len(),
        0
    );
    assert_eq!(store.message_count("missing", "dev-1").await.unwrap(), 0);

    // `mark_read` flags direct rows only and is idempotent; a broadcast id in the same batch is a
    // silent no-op and never advances the watermark.
    let batch = [
        "m-b15".to_owned(),
        "m-d20".to_owned(),
        "m-d10".to_owned(),
        "missing".to_owned(),
    ];
    store.mark_read("alpha", "dev-1", &batch).await.unwrap();
    assert_eq!(
        ids(&store.unread_messages("alpha", "dev-1").await.unwrap()),
        vec!["m-b15", "m-b15b"]
    );
    assert_eq!(store.message_count("alpha", "dev-1").await.unwrap(), 2);
    assert_eq!(store.read_watermark("alpha", "dev-1").await.unwrap(), None);
    store.mark_read("alpha", "dev-1", &batch).await.unwrap();
    assert_eq!(
        ids(&store.unread_messages("alpha", "dev-1").await.unwrap()),
        vec!["m-b15", "m-b15b"]
    );
    // Another member's direct row is untouched by dev-1's read.
    assert!(store
        .unread_messages("alpha", "dev-2")
        .await
        .unwrap()
        .iter()
        .any(|message| message.message_id == "m-d10"));
}

/// Mailbox: the watermark, keyed message replacement and the mailbox key guards.
async fn contract_mailbox_watermark(store: &dyn TeamStore) {
    // The watermark covers every broadcast at or below its timestamp and never moves backwards.
    store
        .set_read_watermark("alpha", "dev-1", 15)
        .await
        .unwrap();
    assert_eq!(
        store.unread_messages("alpha", "dev-1").await.unwrap().len(),
        0
    );
    assert_eq!(store.message_count("alpha", "dev-1").await.unwrap(), 0);
    assert_eq!(
        store.read_watermark("alpha", "dev-1").await.unwrap(),
        Some(15)
    );
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
            "alpha", "m-b15c", "leader", "same", 15,
        ))
        .await
        .unwrap();
    assert_eq!(
        ids(&store.unread_messages("alpha", "dev-1").await.unwrap()),
        vec!["m-b16"]
    );
    assert_eq!(store.message_count("alpha", "dev-1").await.unwrap(), 1);

    // The mailbox is keyed by message id: a rewrite replaces the row.
    store
        .insert_message(MessageRecord::direct(
            "delta", "m-r1", "leader", "dev-1", "first", 30,
        ))
        .await
        .unwrap();
    store
        .insert_message(MessageRecord::direct(
            "delta",
            "m-r1",
            "leader",
            "dev-1",
            "rewritten",
            31,
        ))
        .await
        .unwrap();
    let unread = store.unread_messages("delta", "dev-1").await.unwrap();
    assert_eq!(unread.len(), 1);
    assert_eq!(unread[0].content, "rewritten");
    assert_eq!(unread[0].timestamp, 31);

    let err = store
        .insert_message(MessageRecord::broadcast("alpha", "", "leader", "hi", 1))
        .await
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Validation);
    assert!(err.to_string().contains("message_id must not be empty"));

    let err = store
        .insert_message(MessageRecord::broadcast("", "m-x", "leader", "hi", 1))
        .await
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Validation);
    assert!(err.to_string().contains("team_name must not be empty"));

    let err = store.set_read_watermark("alpha", "", 1).await.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Validation);
    assert!(err.to_string().contains("member_name must not be empty"));
}
