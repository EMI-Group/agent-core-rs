//! Integration coverage for the team memory subsystem (`openjiuwen_memory::team`).
//!
//! The `team` submodules have their own unit tests for the pure model, the in-memory
//! store, the facade, the projection, and the `SQLite` decoder; this suite goes through
//! the [`TeamMemory`] facade and the [`TeamMemoryStore`](openjiuwen_memory::team::store::TeamMemoryStore)
//! seam the way a team runtime does: create/read/update/delete, filtered listing,
//! expiry, keyword search, aggregate stats, and the projection onto the unified
//! [`MemoryRecord`].
//!
//! Every fixture timestamp is pinned to the shared test epoch (`1_700_000_000`), so no
//! assertion depends on the wall clock; records are seeded through
//! [`TeamMemory::store`] where `created_at`/`updated_at` must be deterministic, because
//! the facade stamps them with "now".
//!
//! The default feature set covers the in-memory backend offline. The
//! `#[cfg(feature = "sqlite")]` tests at the bottom of the file open real databases in a
//! [`tempfile::TempDir`] and prove persistence across a reopen plus team isolation inside
//! one database file.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use chrono::{DateTime, Utc};
use openjiuwen_common::error::Error;
use openjiuwen_memory::config::RankingConfig;
use openjiuwen_memory::record::{CallOptions, MemoryRecord, MemoryScope, MemoryType};
use openjiuwen_memory::team::{
    NewTeamMemory, TeamMemory, TeamMemoryFilter, TeamMemoryKind, TeamMemoryRecord, TeamMemoryScope,
    TeamMemoryUpdate, TeamSearchHit, TeamSearchOptions, TeamVisibility, TEAM_MEMORY_METADATA_KEY,
    TEAM_WIDE_MEMBER_KEY,
};
use openjiuwen_retrieval::embedding::Embedding;
use serde_json::json;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// The team every single-team test writes to.
const TEAM: &str = "review-team";

/// A fixed instant `offset` seconds after the shared test epoch (2023-11-14T22:13:20Z).
fn at(offset: i64) -> DateTime<Utc> {
    DateTime::from_timestamp(1_700_000_000 + offset, 0).expect("a valid fixed timestamp")
}

/// Builds a record with both timestamps pinned to the shared epoch.
///
/// The facade stamps writes with the wall clock, so fixtures that assert ordering or
/// filter bounds are seeded through [`TeamMemory::store`] instead.
fn pinned_record(
    team: &str,
    member: Option<&str>,
    id: &str,
    content: &str,
    kind: TeamMemoryKind,
) -> TeamMemoryRecord {
    let scope = member.map_or_else(
        || TeamMemoryScope::new(team),
        |member| TeamMemoryScope::member(team, member),
    );
    let mut record = TeamMemoryRecord::new(&scope, content, kind)
        .with_id(id)
        .with_importance(0.5);
    record.created_at = at(0);
    record.updated_at = at(0);
    record
}

/// Returns `record` with `created_at` moved to `offset` seconds after the epoch.
fn created(mut record: TeamMemoryRecord, offset: i64) -> TeamMemoryRecord {
    record.created_at = at(offset);
    record
}

/// Returns `record` with `updated_at` moved to `offset` seconds after the epoch.
fn updated(mut record: TeamMemoryRecord, offset: i64) -> TeamMemoryRecord {
    record.updated_at = at(offset);
    record
}

/// The ids of `records`, in the order they were returned.
fn ids(records: Vec<TeamMemoryRecord>) -> Vec<String> {
    records.into_iter().map(|record| record.id).collect()
}

/// Upserts every record of `records` straight into `memory`'s store.
async fn seed(memory: &TeamMemory, records: &[TeamMemoryRecord]) {
    for record in records {
        memory.store().upsert(record).await.expect("seed upsert");
    }
}

/// The ids of `hits`, in ranking order.
fn hit_ids(hits: &[TeamSearchHit]) -> Vec<&str> {
    hits.iter().map(|hit| hit.record.id.as_str()).collect()
}

/// A facade seeded with the filter fixtures: two member records in [`TEAM`] (one private,
/// one tagged), one team-wide record, and one record belonging to `other-team`.
async fn filtered_memory() -> TeamMemory {
    let memory = TeamMemory::in_memory();
    let alice_decision = updated(
        pinned_record(
            TEAM,
            Some("alice"),
            "id-a",
            "shared decision",
            TeamMemoryKind::Decision,
        )
        .with_tags(["storage"]),
        20,
    );
    let bob_task = updated(
        pinned_record(
            TEAM,
            Some("bob"),
            "id-b",
            "task for bob",
            TeamMemoryKind::Task,
        )
        .with_tags(["storage", "ci"])
        .with_visibility(TeamVisibility::Private),
        20,
    );
    let team_fact = updated(
        pinned_record(
            TEAM,
            None,
            "id-c",
            "team context",
            TeamMemoryKind::SharedFact,
        ),
        10,
    );
    let other_team = updated(
        pinned_record(
            "other-team",
            Some("alice"),
            "id-d",
            "decision on other team",
            TeamMemoryKind::Decision,
        ),
        30,
    );
    seed(&memory, &[alice_decision, bob_task, team_fact, other_team]).await;
    memory
}

/// A facade holding three records with identical text, importance, and timestamps, so
/// their blended scores tie and the documented id-ascending tie-break is observable.
async fn identical_memory() -> TeamMemory {
    let memory = TeamMemory::in_memory();
    let records: Vec<TeamMemoryRecord> = ["id-a", "id-b", "id-c"]
        .into_iter()
        .map(|id| {
            pinned_record(
                TEAM,
                Some("alice"),
                id,
                "sqlite storage decision",
                TeamMemoryKind::SharedFact,
            )
        })
        .collect();
    seed(&memory, &records).await;
    memory
}

#[tokio::test]
async fn in_memory_crud_round_trip() {
    let memory = TeamMemory::in_memory();
    let call = CallOptions::new();

    let created = memory
        .remember(
            NewTeamMemory::decision(TEAM, "use sqlite for the shared store")
                .with_member("alice")
                .with_title("Storage choice")
                .with_tag("Storage"),
            &call,
        )
        .await
        .expect("remember");
    assert_eq!(created.team_id, TEAM);
    assert_eq!(created.member_id.as_deref(), Some("alice"));
    assert_eq!(created.kind, TeamMemoryKind::Decision);
    assert_eq!(created.tags, vec!["storage".to_owned()]);
    assert_eq!(created.id.len(), 24);
    assert_eq!(created.created_at, created.updated_at);

    let fetched = memory
        .get(TEAM, &created.id)
        .await
        .expect("get")
        .expect("the remembered record");
    assert_eq!(fetched, created);
    assert_eq!(
        memory
            .get("other-team", &created.id)
            .await
            .expect("get from another team"),
        None,
        "a record is only readable through its own team"
    );

    let update = TeamMemoryUpdate::default()
        .content("use sqlite everywhere")
        .with_importance(0.9)
        .with_tags(["storage", "SQLite"]);
    let updated = memory
        .update(TEAM, &created.id, &update, &call)
        .await
        .expect("update");
    assert_eq!(updated.id, created.id);
    assert_eq!(updated.content, "use sqlite everywhere");
    assert!((updated.importance - 0.9).abs() < f32::EPSILON);
    assert_eq!(
        updated.tags,
        vec!["storage".to_owned(), "sqlite".to_owned()]
    );
    assert_eq!(updated.created_at, created.created_at);
    assert!(
        updated.updated_at >= created.updated_at,
        "an update never moves `updated_at` backwards"
    );

    let error = memory
        .update("other-team", &created.id, &update, &call)
        .await
        .expect_err("updating through another team must fail");
    assert!(matches!(error, Error::NotFound(_)), "{error}");

    assert!(memory.forget(TEAM, &created.id).await.expect("forget"));
    assert_eq!(
        memory
            .get(TEAM, &created.id)
            .await
            .expect("get after forget"),
        None
    );
    assert!(
        !memory
            .forget(TEAM, &created.id)
            .await
            .expect("forget twice"),
        "deleting an absent record is idempotent and returns false"
    );
    assert_eq!(
        ids(memory
            .list(&TeamMemoryFilter::for_team(TEAM))
            .await
            .expect("list")),
        Vec::<String>::new()
    );
}

#[tokio::test]
async fn remember_rejects_blank_input_and_duplicate_tags() {
    let memory = TeamMemory::in_memory();
    let call = CallOptions::new();

    let error = memory
        .remember(NewTeamMemory::decision(TEAM, "   "), &call)
        .await
        .expect_err("blank content must be rejected");
    assert!(matches!(error, Error::Validation(_)), "{error}");
    assert!(error.to_string().contains("content"), "{error}");

    let error = memory
        .remember(NewTeamMemory::decision("a/b", "valid content"), &call)
        .await
        .expect_err("an id with a slash must be rejected");
    assert!(error.to_string().contains("team_id"), "{error}");

    let error = memory
        .remember(
            NewTeamMemory::decision(TEAM, "content").with_importance(f32::NAN),
            &call,
        )
        .await
        .expect_err("a non-finite importance must be rejected");
    assert!(error.to_string().contains("finite"), "{error}");

    // The create input tolerates duplicates (they are normalized away), but a *stored*
    // record must already be normalized, so the store write path rejects them.
    let mut duplicated = pinned_record(
        TEAM,
        Some("alice"),
        "id-duplicate-tags",
        "content",
        TeamMemoryKind::Decision,
    );
    duplicated.tags = vec!["Storage".to_owned(), "storage".to_owned()];
    let error = memory
        .store()
        .upsert(&duplicated)
        .await
        .expect_err("duplicate stored tags must be rejected");
    assert!(error.to_string().contains("duplicate"), "{error}");

    assert_eq!(
        ids(memory
            .list(&TeamMemoryFilter::for_team(TEAM))
            .await
            .expect("list")),
        Vec::<String>::new(),
        "no rejected input may leave a record behind"
    );
}

#[tokio::test]
async fn list_narrows_by_kind_member_tags_visibility_and_text() {
    let memory = filtered_memory().await;

    // `list` orders by `updated_at` descending, then by id ascending.
    assert_eq!(
        ids(memory
            .list(&TeamMemoryFilter::for_team(TEAM))
            .await
            .expect("list")),
        vec!["id-a", "id-b", "id-c"]
    );
    assert_eq!(
        ids(memory
            .list(&TeamMemoryFilter::for_team(TEAM).kind(TeamMemoryKind::Decision))
            .await
            .expect("list by kind")),
        vec!["id-a"]
    );
    assert_eq!(
        ids(memory
            .list(&TeamMemoryFilter::for_team(TEAM).member("alice"))
            .await
            .expect("list by member")),
        vec!["id-a"],
        "team-wide records do not match a member filter"
    );

    let mut tagged = TeamMemoryFilter::for_team(TEAM);
    tagged.tags = vec!["STORAGE".to_owned()];
    assert_eq!(
        ids(memory.list(&tagged).await.expect("list by tag")),
        vec!["id-a", "id-b"],
        "tag matching folds case and requires every tag"
    );

    let mut private = TeamMemoryFilter::for_team(TEAM);
    private.visibility = Some(TeamVisibility::Private);
    assert_eq!(
        ids(memory.list(&private).await.expect("list by visibility")),
        vec!["id-b"]
    );

    let mut containing = TeamMemoryFilter::for_team(TEAM);
    containing.text_contains = Some("TEAM CONTEXT".to_owned());
    assert_eq!(
        ids(memory.list(&containing).await.expect("list by text")),
        vec!["id-c"]
    );
}

#[tokio::test]
async fn list_paginates_and_stays_within_one_team() {
    let memory = filtered_memory().await;

    assert_eq!(
        ids(memory
            .list(&TeamMemoryFilter::for_team(TEAM).with_limit(2))
            .await
            .expect("first page")),
        vec!["id-a", "id-b"]
    );
    assert_eq!(
        ids(memory
            .list(
                &TeamMemoryFilter::for_team(TEAM)
                    .with_offset(1)
                    .with_limit(1)
            )
            .await
            .expect("second page")),
        vec!["id-b"]
    );

    assert_eq!(
        ids(memory
            .list(&TeamMemoryFilter::for_team("other-team"))
            .await
            .expect("list the other team")),
        vec!["id-d"],
        "teams never share their listings"
    );
    assert_eq!(
        memory.get(TEAM, "id-d").await.expect("cross-team get"),
        None
    );
    assert!(memory
        .get("other-team", "id-d")
        .await
        .expect("own-team get")
        .is_some());
    assert!(
        !memory
            .forget(TEAM, "id-d")
            .await
            .expect("cross-team forget"),
        "a delete only reaches the named team"
    );
    assert!(memory
        .forget("other-team", "id-d")
        .await
        .expect("own-team forget"));

    let error = memory
        .list(&TeamMemoryFilter::for_team(TEAM).with_limit(0))
        .await
        .expect_err("a zero limit must be rejected");
    assert!(error.to_string().contains("limit"), "{error}");
}

#[tokio::test]
async fn expired_records_are_hidden_unless_requested() {
    let memory = TeamMemory::in_memory();
    let expired = updated(
        pinned_record(
            TEAM,
            Some("alice"),
            "id-expired",
            "sqlite storage legacy note",
            TeamMemoryKind::SharedFact,
        )
        .with_expires_at(at(0)),
        0,
    );
    seed(&memory, std::slice::from_ref(&expired)).await;

    assert_eq!(
        memory.get(TEAM, &expired.id).await.expect("get"),
        Some(expired.clone()),
        "`get` returns expired records so they stay inspectable"
    );
    assert_eq!(
        ids(memory
            .list(&TeamMemoryFilter::for_team(TEAM))
            .await
            .expect("list")),
        Vec::<String>::new(),
        "`list` hides expired records by default"
    );

    let mut including = TeamMemoryFilter::for_team(TEAM);
    including.include_expired = true;
    assert_eq!(
        ids(memory.list(&including).await.expect("list expired")),
        vec!["id-expired"]
    );

    let stats = memory
        .stats(&TeamMemoryFilter::for_team(TEAM))
        .await
        .expect("default stats");
    assert_eq!(
        stats.total, 0,
        "expired records are outside the default view"
    );
    let stats = memory
        .stats(&including)
        .await
        .expect("stats with expired records");
    assert_eq!(stats.total, 1);
    assert_eq!(stats.expired, 1);

    let scope = TeamMemoryScope::new(TEAM);
    let hits = memory
        .search(&scope, "sqlite storage", &TeamSearchOptions::default())
        .await
        .expect("default search");
    assert_eq!(
        hit_ids(&hits),
        Vec::<&str>::new(),
        "`search` hides expired records by default"
    );
    let hits = memory
        .search(
            &scope,
            "sqlite storage",
            &TeamSearchOptions::default().include_expired(),
        )
        .await
        .expect("search with expired records");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].record.id, "id-expired");

    assert!(
        memory
            .set_expiry(TEAM, "id-expired", None)
            .await
            .expect("clear the expiry"),
        "clearing an expiry reports the row was found"
    );
    assert_eq!(
        ids(memory
            .list(&TeamMemoryFilter::for_team(TEAM))
            .await
            .expect("list after clearing the expiry")),
        vec!["id-expired"]
    );
    assert!(!memory
        .set_expiry("other-team", "id-expired", Some(at(0)))
        .await
        .expect("cross-team expiry"));
    assert!(memory
        .set_expiry(TEAM, "id-expired", Some(at(0)))
        .await
        .expect("expire again"));
    assert_eq!(memory.prune_expired().await.expect("prune"), 1);
    assert_eq!(
        memory.get(TEAM, "id-expired").await.expect("get pruned"),
        None
    );
    assert_eq!(memory.prune_expired().await.expect("prune again"), 0);
}

#[tokio::test]
async fn search_ranks_keyword_matches() {
    let memory = TeamMemory::in_memory();
    let strong = pinned_record(
        TEAM,
        Some("alice"),
        "id-strong",
        "sqlite storage decision",
        TeamMemoryKind::SharedFact,
    );
    let weak = pinned_record(
        TEAM,
        Some("alice"),
        "id-weak",
        "sqlite rollout",
        TeamMemoryKind::SharedFact,
    );
    let unrelated = pinned_record(
        TEAM,
        Some("alice"),
        "id-unrelated",
        "ship on friday",
        TeamMemoryKind::SharedFact,
    );
    seed(&memory, &[strong, weak, unrelated]).await;

    let scope = TeamMemoryScope::new(TEAM);
    let hits = memory
        .search(&scope, "sqlite storage", &TeamSearchOptions::default())
        .await
        .expect("search");
    assert_eq!(
        hit_ids(&hits),
        vec!["id-strong", "id-weak"],
        "the record sharing both query terms must rank first"
    );
    assert!((hits[0].keyword - 1.0).abs() < f32::EPSILON);
    assert!(hits[0].keyword > hits[1].keyword);
    assert!(hits[0].score > hits[1].score);

    let hits = memory
        .search(
            &scope,
            "kubernetes manifests",
            &TeamSearchOptions::default().with_threshold(0.0),
        )
        .await
        .expect("unrelated search");
    assert_eq!(
        hit_ids(&hits),
        Vec::<&str>::new(),
        "a record sharing no term with the query is never returned"
    );
}

#[tokio::test]
async fn search_applies_top_k_and_threshold() {
    let memory = identical_memory().await;
    let scope = TeamMemoryScope::new(TEAM);

    // The three records have identical text, importance, and timestamps, so their scores
    // tie and the documented id-ascending tie-break decides the order.
    let hits = memory
        .search(
            &scope,
            "sqlite storage decision",
            &TeamSearchOptions::default().with_top_k(2),
        )
        .await
        .expect("top-k search");
    assert_eq!(
        hit_ids(&hits),
        vec!["id-a", "id-b"],
        "top_k truncates the ranking"
    );

    let hits = memory
        .search(
            &scope,
            "sqlite storage decision",
            &TeamSearchOptions::default().with_threshold(0.99),
        )
        .await
        .expect("strict search");
    assert_eq!(
        hit_ids(&hits),
        Vec::<&str>::new(),
        "a score with importance 0.5 peaks at 0.95, so threshold 0.99 rejects everything: {hits:?}"
    );

    let hits = memory
        .search(
            &scope,
            "sqlite storage decision",
            &TeamSearchOptions::default().with_threshold(0.0),
        )
        .await
        .expect("loose search");
    assert_eq!(hits.len(), 3, "threshold 0.0 keeps every keyword match");

    let error = memory
        .search(
            &scope,
            "sqlite storage",
            &TeamSearchOptions::default().with_top_k(0),
        )
        .await
        .expect_err("top_k must be greater than zero");
    assert!(error.to_string().contains("top_k"), "{error}");
}

#[tokio::test]
async fn search_applies_the_scope_read_rule() {
    let memory = TeamMemory::in_memory();
    let alice_private = pinned_record(
        TEAM,
        Some("alice"),
        "id-alice-private",
        "sqlite storage private note",
        TeamMemoryKind::SharedFact,
    )
    .with_visibility(TeamVisibility::Private);
    let bob_team = pinned_record(
        TEAM,
        Some("bob"),
        "id-bob-team",
        "sqlite storage shared note",
        TeamMemoryKind::SharedFact,
    );
    let bob_members = pinned_record(
        TEAM,
        Some("bob"),
        "id-bob-members",
        "sqlite storage for alice",
        TeamMemoryKind::SharedFact,
    )
    .with_visibility(TeamVisibility::members(["alice"]));
    let other_team = pinned_record(
        "other-team",
        None,
        "id-other",
        "sqlite storage other team",
        TeamMemoryKind::SharedFact,
    );
    seed(&memory, &[alice_private, bob_team, bob_members, other_team]).await;

    let mut seen: Vec<String> = memory
        .search(
            &TeamMemoryScope::member(TEAM, "alice"),
            "sqlite storage",
            &TeamSearchOptions::default(),
        )
        .await
        .expect("alice search")
        .iter()
        .map(|hit| hit.record.id.clone())
        .collect();
    seen.sort();
    assert_eq!(
        seen,
        vec!["id-alice-private", "id-bob-members", "id-bob-team"],
        "alice reads her private record, team records, and records naming her"
    );

    let mut seen: Vec<String> = memory
        .search(
            &TeamMemoryScope::member(TEAM, "bob"),
            "sqlite storage",
            &TeamSearchOptions::default(),
        )
        .await
        .expect("bob search")
        .iter()
        .map(|hit| hit.record.id.clone())
        .collect();
    seen.sort();
    assert_eq!(
        seen,
        vec!["id-bob-members", "id-bob-team"],
        "bob cannot read alice's private record"
    );

    let mut seen: Vec<String> = memory
        .search(
            &TeamMemoryScope::new(TEAM),
            "sqlite storage",
            &TeamSearchOptions::default(),
        )
        .await
        .expect("team-wide search")
        .iter()
        .map(|hit| hit.record.id.clone())
        .collect();
    seen.sort();
    assert_eq!(
        seen,
        vec!["id-alice-private", "id-bob-members", "id-bob-team"],
        "the team-wide (leader) view reads every visibility"
    );

    assert!(
        memory
            .search(
                &TeamMemoryScope::new("other-team"),
                "sqlite storage",
                &TeamSearchOptions::default(),
            )
            .await
            .expect("other-team search")
            .iter()
            .all(|hit| hit.record.team_id == "other-team"),
        "a search never crosses the team boundary"
    );
}

#[tokio::test]
async fn cancelled_calls_return_cancelled() {
    let memory = TeamMemory::in_memory();
    seed(
        &memory,
        &[pinned_record(
            TEAM,
            Some("alice"),
            "id-a",
            "sqlite storage",
            TeamMemoryKind::SharedFact,
        )],
    )
    .await;

    let token = CancellationToken::new();
    token.cancel();

    let options = TeamSearchOptions::default().with_cancellation(token.clone());
    let error = memory
        .search(&TeamMemoryScope::new(TEAM), "sqlite storage", &options)
        .await
        .expect_err("a cancelled search must fail");
    assert!(matches!(error, Error::Cancelled), "{error}");

    let error = memory
        .remember(
            NewTeamMemory::decision(TEAM, "sqlite storage"),
            &CallOptions::new().with_cancellation(token),
        )
        .await
        .expect_err("a cancelled write must fail");
    assert!(matches!(error, Error::Cancelled), "{error}");

    assert_eq!(
        memory
            .list(&TeamMemoryFilter::for_team(TEAM))
            .await
            .expect("list")
            .len(),
        1,
        "the cancelled write must not add a record"
    );
}

#[tokio::test]
async fn stats_aggregate_kinds_members_and_track_deletes() {
    let memory = TeamMemory::in_memory();
    let decision = updated(
        created(
            pinned_record(
                TEAM,
                Some("alice"),
                "id-a",
                "sqlite decision",
                TeamMemoryKind::Decision,
            ),
            20,
        ),
        20,
    );
    let task = updated(
        created(
            pinned_record(
                TEAM,
                Some("alice"),
                "id-b",
                "release task",
                TeamMemoryKind::Task,
            ),
            10,
        ),
        10,
    );
    let fact = pinned_record(
        TEAM,
        None,
        "id-c",
        "team context",
        TeamMemoryKind::SharedFact,
    );
    seed(&memory, &[decision, task, fact]).await;

    let stats = memory
        .stats(&TeamMemoryFilter::for_team(TEAM))
        .await
        .expect("stats");
    assert_eq!(stats.total, 3);
    assert_eq!(stats.by_kind.get("decision"), Some(&1));
    assert_eq!(stats.by_kind.get("task"), Some(&1));
    assert_eq!(stats.by_kind.get("shared_fact"), Some(&1));
    assert_eq!(stats.by_member.get("alice"), Some(&2));
    assert_eq!(
        stats.by_member.get(TEAM_WIDE_MEMBER_KEY),
        Some(&1),
        "member-less records are bucketed under the team-wide key"
    );
    assert_eq!(stats.expired, 0);
    assert_eq!(stats.oldest, Some(at(0)));
    assert_eq!(stats.newest, Some(at(20)));

    assert!(memory.forget(TEAM, "id-a").await.expect("forget"));
    let stats = memory
        .stats(&TeamMemoryFilter::for_team(TEAM))
        .await
        .expect("stats after delete");
    assert_eq!(stats.total, 2);
    assert_eq!(stats.by_kind.get("decision"), None);
    assert_eq!(stats.by_member.get("alice"), Some(&1));
}

#[tokio::test]
async fn projection_round_trips_every_field() {
    let kinds = [
        TeamMemoryKind::SharedFact,
        TeamMemoryKind::MemberProfile,
        TeamMemoryKind::Decision,
        TeamMemoryKind::Task,
        TeamMemoryKind::Artifact,
        TeamMemoryKind::HandoffNote,
        TeamMemoryKind::Custom("release_check".to_owned()),
    ];
    for kind in kinds {
        let mut record = TeamMemoryRecord::new(
            &TeamMemoryScope::member(TEAM, "alice").with_session("s-1"),
            "use sqlite for the shared store",
            kind.clone(),
        )
        .with_id("0123456789abcdef01234567")
        .with_title("Storage choice")
        .with_importance(0.8)
        .with_tags(["Storage", "sqlite"])
        .with_metadata("issue", json!(42))
        .with_source("leader-extraction")
        .with_visibility(TeamVisibility::members(["alice", "bob"]))
        .with_expires_at(at(86_400));
        record.created_at = at(0);
        record.updated_at = at(60);

        let projected = record.to_memory_record();
        assert_eq!(projected.id, record.id);
        assert_eq!(projected.content, record.content);
        assert_eq!(projected.scope.scope_id, TEAM);
        assert_eq!(projected.scope.user_id, "alice");
        assert_eq!(projected.scope.session_id.as_deref(), Some("s-1"));
        assert_eq!(projected.scope.agent_id, None);
        assert_eq!(
            projected.kind,
            if kind == TeamMemoryKind::MemberProfile {
                MemoryType::UserProfile
            } else {
                MemoryType::SemanticMemory
            },
            "only `member_profile` maps onto `user_profile`"
        );
        assert!((projected.importance - 0.8).abs() < f32::EPSILON);
        assert_eq!(projected.timestamp, record.created_at);
        assert_eq!(projected.updated_at, record.updated_at);
        assert_eq!(projected.last_access, None);
        assert_eq!(projected.access_count, 0);
        assert_eq!(projected.embedding, None);
        assert_eq!(projected.source.as_deref(), Some("leader-extraction"));
        assert_eq!(projected.metadata.get("issue"), Some(&json!(42)));

        let reserved = projected
            .metadata
            .get(TEAM_MEMORY_METADATA_KEY)
            .expect("the reserved team marker")
            .as_object()
            .expect("the reserved marker is an object");
        assert_eq!(reserved["team_id"], json!(TEAM));
        assert_eq!(reserved["member_id"], json!("alice"));
        assert_eq!(reserved["session_id"], json!("s-1"));
        assert_eq!(reserved["kind"], json!(kind.as_str()));
        assert_eq!(reserved["title"], json!("Storage choice"));
        assert_eq!(reserved["tags"], json!(["storage", "sqlite"]));
        assert_eq!(reserved["visibility"], json!({"members": ["alice", "bob"]}));
        assert_eq!(reserved["expires_at"], json!(at(86_400).to_rfc3339()));

        // The projection is documented as lossless for every kind and visibility; this
        // asserts exactly that, field for field.
        let restored =
            TeamMemoryRecord::from_memory_record(&projected).expect("projection round trip");
        assert_eq!(restored, record);
    }

    // A team-wide record has no member, so the projection falls back to the team id as
    // the `MemoryRecord` user id.
    let team_wide = pinned_record(
        TEAM,
        None,
        "0123456789abcdef01234568",
        "team context",
        TeamMemoryKind::SharedFact,
    );
    let projected = team_wide.to_memory_record();
    assert_eq!(projected.scope.user_id, TEAM);
    assert_eq!(projected.scope.scope_id, TEAM);
    assert_eq!(projected.scope.session_id, None);
    assert_eq!(
        TeamMemoryRecord::from_memory_record(&projected).expect("team-wide round trip"),
        team_wide
    );

    // A plain long-term memory record is rejected instead of being mistaken for a team
    // memory.
    let plain = MemoryRecord::new(
        MemoryScope::new("alice", TEAM),
        "plain memory",
        MemoryType::SemanticMemory,
    );
    let error = TeamMemoryRecord::from_memory_record(&plain)
        .expect_err("a record without the reserved marker is not a team memory");
    assert!(matches!(error, Error::Validation(_)), "{error}");
    assert!(
        error.to_string().contains(TEAM_MEMORY_METADATA_KEY),
        "{error}"
    );
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn sqlite_persists_records_and_deletes_across_a_reopen() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("team-memory.db");
    let kept = pinned_record(
        TEAM,
        Some("alice"),
        "id-kept",
        "sqlite storage decision",
        TeamMemoryKind::Decision,
    )
    .with_tags(["storage"]);
    let removed = pinned_record(
        TEAM,
        Some("bob"),
        "id-removed",
        "rollout task",
        TeamMemoryKind::Task,
    );

    {
        let memory = TeamMemory::sqlite(&path).await.expect("open the database");
        seed(&memory, std::slice::from_ref(&kept)).await;
        seed(&memory, std::slice::from_ref(&removed)).await;
        assert!(memory.forget(TEAM, &removed.id).await.expect("forget"));
        assert_eq!(
            memory
                .list(&TeamMemoryFilter::for_team(TEAM))
                .await
                .expect("list before the reopen")
                .len(),
            1
        );
    }

    let reopened = TeamMemory::sqlite(&path)
        .await
        .expect("reopen the database");
    assert_eq!(
        reopened
            .get(TEAM, &kept.id)
            .await
            .expect("get the kept record"),
        Some(kept.clone()),
        "a reopened database returns the record exactly as it was persisted"
    );
    assert_eq!(
        reopened
            .get(TEAM, &removed.id)
            .await
            .expect("get the deleted record"),
        None,
        "a delete survives the reopen"
    );
    assert_eq!(
        ids(reopened
            .list(&TeamMemoryFilter::for_team(TEAM))
            .await
            .expect("list after the reopen")),
        vec!["id-kept"]
    );
    let hits = reopened
        .search(
            &TeamMemoryScope::new(TEAM),
            "sqlite storage",
            &TeamSearchOptions::default(),
        )
        .await
        .expect("search after the reopen");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].record.id, "id-kept");
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn sqlite_keeps_two_teams_sharing_one_database_isolated() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("shared.db");
    let alpha = TeamMemory::sqlite(&path)
        .await
        .expect("open for the first team");
    let beta = TeamMemory::sqlite(&path)
        .await
        .expect("open for the second team");
    let call = CallOptions::new();

    let alpha_record = alpha
        .remember(
            NewTeamMemory::decision("alpha-team", "sqlite decision for alpha").with_member("alice"),
            &call,
        )
        .await
        .expect("remember for alpha");
    let beta_record = beta
        .remember(
            NewTeamMemory::decision("beta-team", "sqlite decision for beta").with_member("bob"),
            &call,
        )
        .await
        .expect("remember for beta");

    assert_eq!(
        ids(alpha
            .list(&TeamMemoryFilter::for_team("alpha-team"))
            .await
            .expect("alpha list")),
        vec![alpha_record.id.clone()]
    );
    assert_eq!(
        alpha.get("alpha-team", &beta_record.id).await.expect("get"),
        None,
        "a team cannot read another team's row by id"
    );
    assert_eq!(
        alpha.get("beta-team", &alpha_record.id).await.expect("get"),
        None,
        "a row is only addressable through its own team id"
    );
    assert!(
        !alpha
            .forget("alpha-team", &beta_record.id)
            .await
            .expect("forget"),
        "a delete only reaches the named team"
    );

    let hits = alpha
        .search(
            &TeamMemoryScope::new("alpha-team"),
            "sqlite decision beta",
            &TeamSearchOptions::default(),
        )
        .await
        .expect("alpha search");
    assert_eq!(hits.len(), 1);
    assert!(hits.iter().all(|hit| hit.record.team_id == "alpha-team"));

    let stats = alpha
        .stats(&TeamMemoryFilter::for_team("alpha-team"))
        .await
        .expect("alpha stats");
    assert_eq!(stats.total, 1);
    assert_eq!(stats.by_member.get("alice"), Some(&1));
    assert_eq!(stats.by_member.get("bob"), None);

    assert_eq!(
        beta.get("beta-team", &beta_record.id)
            .await
            .expect("beta get"),
        Some(beta_record),
        "the other team's row is untouched by the failed cross-team calls"
    );
    assert_eq!(
        alpha
            .clear(&TeamMemoryFilter::for_team("alpha-team"))
            .await
            .expect("alpha clear"),
        1
    );
    assert_eq!(
        beta.list(&TeamMemoryFilter::for_team("beta-team"))
            .await
            .expect("beta list")
            .len(),
        1,
        "clearing one team leaves the other team's rows in place"
    );
}

// ---------------------------------------------------------------------------
// Embedding-based recall
// ---------------------------------------------------------------------------

/// A deterministic two-axis "topic" embedder.
///
/// Every text mentioning `postgres`/`schema`/`database` maps to the first axis and every
/// text mentioning `pizza`/`lunch` to the second, so two texts about the same topic are
/// similar even with no shared word, two texts about different topics are orthogonal
/// (`similarity = 0.5`), and any other text maps to the zero vector, the documented "no
/// semantic signal" value. Nothing here touches the network.
#[derive(Debug)]
struct TopicEmbedding;

/// The vector [`TopicEmbedding`] returns for `text`.
fn topic_vector(text: &str) -> Vec<f32> {
    let lowered = text.to_lowercase();
    if ["postgres", "schema", "database"]
        .iter()
        .any(|term| lowered.contains(term))
    {
        vec![1.0, 0.0]
    } else if ["pizza", "lunch"].iter().any(|term| lowered.contains(term)) {
        vec![0.0, 1.0]
    } else {
        vec![0.0, 0.0]
    }
}

#[async_trait::async_trait]
impl Embedding for TopicEmbedding {
    fn model_name(&self) -> &'static str {
        "topic-embedding"
    }

    fn dimension_hint(&self) -> Option<usize> {
        Some(2)
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, Error> {
        Ok(topic_vector(text))
    }

    async fn embed_documents(
        &self,
        texts: &[String],
        _batch_size: Option<usize>,
    ) -> Result<Vec<Vec<f32>>, Error> {
        Ok(texts.iter().map(|text| topic_vector(text)).collect())
    }
}

/// An embedder whose every call fails, to prove failures are surfaced and not swallowed.
#[derive(Debug)]
struct FailingEmbedding;

#[async_trait::async_trait]
impl Embedding for FailingEmbedding {
    fn model_name(&self) -> &'static str {
        "failing-embedding"
    }

    async fn embed_query(&self, _text: &str) -> Result<Vec<f32>, Error> {
        Err(Error::external(std::io::Error::other(
            "embedding backend is down",
        )))
    }

    async fn embed_documents(
        &self,
        _texts: &[String],
        _batch_size: Option<usize>,
    ) -> Result<Vec<Vec<f32>>, Error> {
        Err(Error::external(std::io::Error::other(
            "embedding backend is down",
        )))
    }
}

/// A facade over the in-memory store with [`TopicEmbedding`] attached.
fn semantic_memory() -> TeamMemory {
    TeamMemory::in_memory().with_embedding(Arc::new(TopicEmbedding))
}

/// Seeds one record per topic through the facade: a semantically related decision that is
/// old and unimportant, and an unrelated one that is brand new and important.
async fn seeded_semantic_memory(memory: &TeamMemory) {
    let call = CallOptions::new();
    let related = memory
        .remember(
            NewTeamMemory::decision(TEAM, "postgres schema upgrade")
                .with_member("alice")
                .with_importance(0.1),
            &call,
        )
        .await
        .expect("remember the related decision");
    let mut aged = related.clone();
    aged.updated_at = Utc::now() - chrono::Duration::days(120);
    memory
        .store()
        .upsert(&aged)
        .await
        .expect("age the related decision");
    memory
        .remember(
            NewTeamMemory::decision(TEAM, "pizza for the sprint lunch")
                .with_member("bob")
                .with_importance(1.0),
            &call,
        )
        .await
        .expect("remember the unrelated decision");
}

#[tokio::test]
async fn an_embedder_ranks_a_related_record_over_a_newer_important_one() {
    let memory = semantic_memory();
    seeded_semantic_memory(&memory).await;
    let records = memory
        .list(&TeamMemoryFilter::for_team(TEAM))
        .await
        .expect("list the seeded records");
    let related = records
        .iter()
        .find(|record| record.content.contains("postgres"))
        .expect("the related record");
    assert_eq!(
        related.embedding.as_deref(),
        Some([1.0, 0.0].as_slice()),
        "the facade embeds the record it writes"
    );
    assert!(
        records.iter().all(|record| record
            .embedding
            .as_ref()
            .is_some_and(|vector| vector.len() == 2)),
        "every record written through the facade carries a vector"
    );
    // The query shares no term with either record, so only the semantic group can rank.
    let hits = memory
        .search(
            &TeamMemoryScope::new(TEAM),
            "database migration",
            &TeamSearchOptions::default(),
        )
        .await
        .expect("semantic search");
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].record.id, related.id);
    assert_eq!(hits[0].keyword, 0.0);
    assert!((hits[0].similarity - 1.0).abs() < 1e-6);
    assert!((hits[1].similarity - 0.5).abs() < 1e-6);
    assert!(hits[0].score > hits[1].score);
    // The score is the documented blend with the facade's default weights.
    let expected = 0.85 * (0.70 * hits[0].similarity + 0.30 * hits[0].keyword)
        + 0.10 * hits[0].importance
        + 0.05 * hits[0].recency;
    assert!((hits[0].score - expected).abs() < 1e-5, "{}", hits[0].score);
}

#[tokio::test]
async fn the_similarity_weight_is_what_drives_the_semantic_ranking() {
    let memory = semantic_memory().with_ranking(RankingConfig {
        similarity_weight: 0.0,
        ..RankingConfig::default()
    });
    seeded_semantic_memory(&memory).await;
    let hits = memory
        .search(
            &TeamMemoryScope::new(TEAM),
            "database migration",
            &TeamSearchOptions::default(),
        )
        .await
        .expect("semantic search");
    assert!(
        hits.is_empty(),
        "with no similarity weight a lexically disjoint record has no score left: {hits:?}"
    );
}

#[tokio::test]
async fn without_an_embedder_scores_and_order_match_the_keyword_formula() {
    let memory = TeamMemory::in_memory();
    assert!(memory.embedding().is_none());
    let call = CallOptions::new();
    let decision = memory
        .remember(NewTeamMemory::decision(TEAM, "sqlite storage"), &call)
        .await
        .expect("remember");
    assert_eq!(
        decision.embedding, None,
        "a keyword-only facade stores no vector"
    );
    memory
        .remember(
            NewTeamMemory::decision(TEAM, "sqlite storage decision for the shared store"),
            &call,
        )
        .await
        .expect("remember the second record");
    memory
        .remember(NewTeamMemory::decision(TEAM, "gardening notes"), &call)
        .await
        .expect("remember the unrelated record");
    let hits = memory
        .search(
            &TeamMemoryScope::new(TEAM),
            "sqlite storage",
            &TeamSearchOptions::default(),
        )
        .await
        .expect("keyword search");
    assert_eq!(hits.len(), 2, "the unrelated record is still dropped");
    for hit in &hits {
        assert_eq!(hit.similarity, 0.0);
        // score = 0.85 * keyword + 0.10 * importance + 0.05 * recency, exactly as before
        // the embedding seam existed.
        let expected = 0.85 * hit.keyword + 0.10 * hit.importance + 0.05 * hit.recency;
        assert!(
            (hit.score - expected).abs() < 1e-6,
            "{} != {expected}",
            hit.score
        );
    }
    // A single record whose text is the query: keyword 1.0, importance 0.5, recency 1.0.
    let single = TeamMemory::in_memory();
    single
        .remember(
            NewTeamMemory::decision(TEAM, "sqlite storage").with_importance(0.5),
            &call,
        )
        .await
        .expect("remember the only record");
    let hits = single
        .search(
            &TeamMemoryScope::new(TEAM),
            "sqlite storage",
            &TeamSearchOptions::default(),
        )
        .await
        .expect("keyword search");
    assert_eq!(hits.len(), 1);
    assert!((hits[0].keyword - 1.0).abs() < 1e-6);
    assert!(
        (hits[0].score - 0.95).abs() < 1e-4,
        "expected the old 0.95 score, got {}",
        hits[0].score
    );
}

#[tokio::test]
async fn a_failing_embedder_fails_the_write_and_the_search() {
    let memory = TeamMemory::in_memory().with_embedding(Arc::new(FailingEmbedding));
    let call = CallOptions::new();
    let error = memory
        .remember(NewTeamMemory::decision(TEAM, "sqlite storage"), &call)
        .await
        .expect_err("the write must fail");
    assert!(
        error.to_string().contains("embedding backend is down"),
        "{error}"
    );
    assert!(
        memory
            .list(&TeamMemoryFilter::for_team(TEAM))
            .await
            .expect("list")
            .is_empty(),
        "a failed embedding stores nothing"
    );
    // Seed a record through the store so the search has a candidate.
    seed(
        &memory,
        &[pinned_record(
            TEAM,
            None,
            "id-a",
            "sqlite storage",
            TeamMemoryKind::Decision,
        )],
    )
    .await;
    let error = memory
        .search(
            &TeamMemoryScope::new(TEAM),
            "sqlite storage",
            &TeamSearchOptions::default(),
        )
        .await
        .expect_err("the search must fail");
    assert!(
        error.to_string().contains("embedding backend is down"),
        "{error}"
    );
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn sqlite_round_trips_the_embedding_and_ranks_with_it() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("semantic.db");
    let memory = TeamMemory::sqlite(&path)
        .await
        .expect("open the database")
        .with_embedding(Arc::new(TopicEmbedding));
    seeded_semantic_memory(&memory).await;
    let reopened = TeamMemory::sqlite(&path)
        .await
        .expect("reopen the database")
        .with_embedding(Arc::new(TopicEmbedding));
    let records = reopened
        .list(&TeamMemoryFilter::for_team(TEAM))
        .await
        .expect("list after the reopen");
    let related = records
        .iter()
        .find(|record| record.content.contains("postgres"))
        .expect("the related record");
    assert_eq!(
        related.embedding.as_deref(),
        Some([1.0, 0.0].as_slice()),
        "the vector survives the reopen"
    );
    let hits = reopened
        .search(
            &TeamMemoryScope::new(TEAM),
            "database migration",
            &TeamSearchOptions::default(),
        )
        .await
        .expect("semantic search after the reopen");
    assert_eq!(hits[0].record.id, related.id);
    assert!((hits[0].similarity - 1.0).abs() < 1e-6);
    // `get` returns the vector too, not just the listing.
    let fetched = reopened
        .get(TEAM, &related.id)
        .await
        .expect("get")
        .expect("the related record");
    assert_eq!(fetched.embedding, related.embedding);
}
