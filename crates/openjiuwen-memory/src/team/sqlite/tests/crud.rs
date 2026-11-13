//! Round-trip fidelity, reopening, the upsert contract, team isolation, expiry
//! mutations, and pool sharing.
use super::*;
use crate::team::model::{TeamMemoryKind, TeamVisibility, DEFAULT_IMPORTANCE};
use chrono::Duration;
use tempfile::TempDir;

#[tokio::test]
async fn round_trip_preserves_every_field() {
    let store = store().await;
    let mut full = record(
        "review-team",
        Some("alice"),
        "id-a",
        "Use SQLite for the shared store",
        TeamMemoryKind::Decision,
    )
    .with_title("Storage choice")
    .with_importance(0.8)
    .with_tags(["Storage", "sqlite"])
    .with_metadata("issue", serde_json::json!(42))
    .with_source("leader-extraction")
    .with_visibility(TeamVisibility::members(["alice", "bob"]));
    full.session_id = Some("s1".to_owned());
    full.expires_at = Some(at(1_800_000_000));
    let stored = store.upsert(&full).await.unwrap();
    assert_eq!(stored, full);
    let read = store.get("review-team", "id-a").await.unwrap().unwrap();
    assert_eq!(read, full);
    // Team-wide variant: no member, no session, no title, no expiry.
    let team_wide = record(
        "review-team",
        None,
        "id-b",
        "team-wide note",
        TeamMemoryKind::SharedFact,
    );
    store.upsert(&team_wide).await.unwrap();
    assert_eq!(
        store.get("review-team", "id-b").await.unwrap().unwrap(),
        team_wide
    );
}

#[tokio::test]
async fn every_kind_and_visibility_round_trips() {
    let store = store().await;
    let mut expected = Vec::new();
    for (index, kind) in TeamMemoryKind::BUILTIN.into_iter().enumerate() {
        for (offset, visibility) in [
            TeamVisibility::Team,
            TeamVisibility::Private,
            TeamVisibility::members(["alice"]),
        ]
        .into_iter()
        .enumerate()
        {
            let id = format!("id-{index}-{offset}");
            let record = record("review-team", Some("alice"), &id, "content", kind.clone())
                .with_visibility(visibility);
            expected.push(record.clone());
            store.upsert(&record).await.unwrap();
        }
    }
    let listed = store
        .list(&TeamMemoryFilter::for_team("review-team"))
        .await
        .unwrap();
    assert_eq!(listed.len(), expected.len());
    for record in &expected {
        assert_eq!(
            store.get("review-team", &record.id).await.unwrap(),
            Some(record.clone())
        );
    }
    // A caller-defined kind survives verbatim.
    let custom = record(
        "review-team",
        Some("alice"),
        "id-custom",
        "content",
        TeamMemoryKind::Custom("review_rule".to_owned()),
    );
    store.upsert(&custom).await.unwrap();
    let read = store
        .get("review-team", "id-custom")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(read.kind, TeamMemoryKind::Custom("review_rule".to_owned()));
}

#[tokio::test]
async fn reopen_observes_committed_records() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("nested").join("team-memory.db");
    let record = record(
        "review-team",
        Some("alice"),
        "id-a",
        "use sqlite",
        TeamMemoryKind::Decision,
    );
    {
        let store = SqliteTeamMemoryStore::open(&path).await.unwrap();
        store.upsert(&record).await.unwrap();
    }
    let reopened = SqliteTeamMemoryStore::open(&path).await.unwrap();
    assert_eq!(
        reopened.get("review-team", "id-a").await.unwrap(),
        Some(record)
    );
    assert_eq!(reopened.pool().size(), reopened.pool().size());
}

#[tokio::test]
async fn upsert_overwrites_content_and_keeps_created_at() {
    let store = store().await;
    let original = record(
        "review-team",
        Some("alice"),
        "id-a",
        "first version",
        TeamMemoryKind::Task,
    );
    store.upsert(&original).await.unwrap();
    let mut updated = original.clone();
    updated.content = "second version".to_owned();
    updated.updated_at = at(1_700_000_500);
    updated.importance = 0.9;
    let stored = store.upsert(&updated).await.unwrap();
    assert_eq!(stored.created_at, original.created_at);
    assert_eq!(stored.updated_at, at(1_700_000_500));
    assert_eq!(stored.content, "second version");
    let read = store.get("review-team", "id-a").await.unwrap().unwrap();
    assert_eq!(read, stored);
    assert_eq!(read.importance, 0.9);
}

#[tokio::test]
async fn upsert_rejects_an_invalid_record() {
    let store = store().await;
    let mut invalid = record(
        "review-team",
        Some("alice"),
        "id-a",
        "content",
        TeamMemoryKind::Task,
    );
    invalid.content = "  ".to_owned();
    let error = store.upsert(&invalid).await.unwrap_err();
    assert_eq!(
        error.kind(),
        openjiuwen_common::error::ErrorKind::Validation
    );
    assert!(store.get("review-team", "id-a").await.unwrap().is_none());
}

#[tokio::test]
async fn teams_are_fully_isolated() {
    let store = store().await;
    let alice = record(
        "review-team",
        Some("alice"),
        "id-review",
        "sqlite storage",
        TeamMemoryKind::Decision,
    );
    let mut bob = alice.clone();
    bob.id = "id-other".to_owned();
    bob.team_id = "other-team".to_owned();
    seed(&store, &[alice.clone(), bob.clone()]).await;
    assert_eq!(
        store.get("review-team", "id-review").await.unwrap(),
        Some(alice.clone())
    );
    assert_eq!(
        store.get("other-team", "id-other").await.unwrap(),
        Some(bob.clone())
    );
    // An id of one team never leaks into another team's view.
    assert!(store
        .get("other-team", "id-review")
        .await
        .unwrap()
        .is_none());
    assert!(store
        .get("unknown-team", "id-review")
        .await
        .unwrap()
        .is_none());
    assert_eq!(
        store
            .list(&TeamMemoryFilter::for_team("other-team"))
            .await
            .unwrap(),
        vec![bob.clone()]
    );
    assert!(!store.delete("other-team", "missing").await.unwrap());
    assert!(!store.delete("other-team", "id-review").await.unwrap());
    assert!(store.delete("other-team", "id-other").await.unwrap());
    assert!(store
        .get("review-team", "id-review")
        .await
        .unwrap()
        .is_some());
    assert_eq!(
        store
            .clear(&TeamMemoryFilter::for_team("review-team"))
            .await
            .unwrap(),
        1
    );
    assert!(store
        .get("review-team", "id-review")
        .await
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn delete_and_clear_report_what_changed() {
    let store = store().await;
    seed(
        &store,
        &[
            record(
                "review-team",
                Some("alice"),
                "id-a",
                "content a",
                TeamMemoryKind::Task,
            ),
            record(
                "review-team",
                Some("bob"),
                "id-b",
                "content b",
                TeamMemoryKind::Task,
            ),
        ],
    )
    .await;
    assert!(store.delete("review-team", "id-a").await.unwrap());
    assert!(!store.delete("review-team", "id-a").await.unwrap());
    assert!(!store.delete("other-team", "id-b").await.unwrap());
    assert_eq!(
        store
            .clear(&TeamMemoryFilter::for_team("review-team").member("alice"))
            .await
            .unwrap(),
        0
    );
    assert_eq!(
        store
            .clear(&TeamMemoryFilter::for_team("review-team"))
            .await
            .unwrap(),
        1
    );
    assert_eq!(
        store
            .clear(&TeamMemoryFilter::for_team("review-team"))
            .await
            .unwrap(),
        0
    );
    assert!(store.clear(&TeamMemoryFilter::default()).await.is_err());
}

#[tokio::test]
async fn set_expiry_sets_and_clears_without_touching_updated_at() {
    let store = store().await;
    let record = updated_at(
        record(
            "review-team",
            Some("alice"),
            "id-a",
            "content",
            TeamMemoryKind::Task,
        ),
        1_700_000_500,
    );
    store.upsert(&record).await.unwrap();
    let expiry = at(1_800_000_000);
    assert!(store
        .set_expiry("review-team", "id-a", Some(expiry))
        .await
        .unwrap());
    let stored = store.get("review-team", "id-a").await.unwrap().unwrap();
    assert_eq!(stored.expires_at, Some(expiry));
    assert_eq!(stored.updated_at, record.updated_at);
    assert!(!store
        .set_expiry("other-team", "id-a", Some(expiry))
        .await
        .unwrap());
    assert!(!store
        .set_expiry("review-team", "missing", None)
        .await
        .unwrap());
    assert!(store.set_expiry("review-team", "id-a", None).await.unwrap());
    let stored = store.get("review-team", "id-a").await.unwrap().unwrap();
    assert_eq!(stored.expires_at, None);
}

#[tokio::test]
async fn from_pool_reuses_an_existing_database() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("team-memory.db");
    let first = SqliteTeamMemoryStore::open(&path).await.unwrap();
    let record = updated_at(
        record(
            "review-team",
            Some("alice"),
            "id-a",
            "content",
            TeamMemoryKind::Task,
        ),
        1_700_000_100,
    );
    first.upsert(&record).await.unwrap();
    let second = SqliteTeamMemoryStore::from_pool(first.pool().clone())
        .await
        .unwrap();
    assert_eq!(
        second.get("review-team", "id-a").await.unwrap(),
        Some(record)
    );
    // Clones share the pool, so a write through one clone is visible through the other.
    let clone = second.clone();
    clone.delete("review-team", "id-a").await.unwrap();
    assert!(second.get("review-team", "id-a").await.unwrap().is_none());
}

#[tokio::test]
async fn default_importance_survives_a_round_trip() {
    let store = store().await;
    let record = record(
        "review-team",
        Some("alice"),
        "id-a",
        "content",
        TeamMemoryKind::Task,
    );
    let stored = store.upsert(&record).await.unwrap();
    assert!((stored.importance - DEFAULT_IMPORTANCE).abs() < f32::EPSILON);
    assert_eq!(
        store.get("review-team", "id-a").await.unwrap().unwrap(),
        record
    );
    assert!(record.updated_at - record.created_at < Duration::seconds(1));
}
