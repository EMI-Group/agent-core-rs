//! Malformed rows: the decoder degrades or skips them instead of failing the listing.
use super::*;
use crate::team::model::{TeamMemoryKind, TeamVisibility};

#[tokio::test]
async fn malformed_json_columns_degrade_instead_of_failing_the_list() {
    let store = store().await;
    seed(
        &store,
        &[record(
            "review-team",
            Some("alice"),
            "id-good",
            "content",
            TeamMemoryKind::Task,
        )],
    )
    .await;
    insert_raw(
        &store,
        "id-malformed",
        "{not-json",
        "not-json",
        "\"text\"",
        4.0,
        "2023-11-14T22:13:20.000Z",
        "2023-11-14T22:13:20.000Z",
    )
    .await;
    let listed = store
        .list(&TeamMemoryFilter::for_team("review-team"))
        .await
        .unwrap();
    assert_eq!(listed.len(), 2);
    let malformed = listed
        .iter()
        .find(|record| record.id == "id-malformed")
        .unwrap();
    assert_eq!(malformed.visibility, TeamVisibility::Private);
    assert!(malformed.tags.is_empty());
    assert_eq!(malformed.metadata, serde_json::json!({}));
    assert!((malformed.importance - 1.0).abs() < f32::EPSILON);
    assert_eq!(malformed.kind, TeamMemoryKind::Task);
    // A fail-closed visibility keeps the row out of another member's view.
    assert_eq!(malformed.created_at, at(1_700_000_000));
}

#[tokio::test]
async fn row_with_unparseable_timestamps_is_skipped() {
    let store = store().await;
    seed(
        &store,
        &[record(
            "review-team",
            Some("alice"),
            "id-good",
            "content",
            TeamMemoryKind::Task,
        )],
    )
    .await;
    insert_raw(
        &store,
        "id-broken-created",
        "\"team\"",
        "[]",
        "{}",
        0.5,
        "not-a-timestamp",
        "2023-11-14T22:13:20.000Z",
    )
    .await;
    insert_raw(
        &store,
        "id-broken-updated",
        "\"team\"",
        "[]",
        "{}",
        0.5,
        "2023-11-14T22:13:20.000Z",
        "whenever",
    )
    .await;
    let listed = store
        .list(&TeamMemoryFilter::for_team("review-team"))
        .await
        .unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].id, "id-good");
    assert!(store
        .get("review-team", "id-broken-created")
        .await
        .unwrap()
        .is_none());
    // Statistics survive the same rows and count every matching row from SQL.
    let stats = store
        .stats(&TeamMemoryFilter::for_team("review-team"))
        .await
        .unwrap();
    assert_eq!(stats.total, 3);
    assert_eq!(stats.oldest, Some(at(1_700_000_000)));
}

#[tokio::test]
async fn malformed_expiry_is_treated_as_no_expiry() {
    let store = store().await;
    sqlx::query(
        "INSERT INTO team_memories (id, team_id, member_id, session_id, kind, visibility_json, \
         title, content, importance, tags_json, metadata_json, source, created_at, updated_at, \
         expires_at) VALUES ('id-a', 'review-team', NULL, NULL, 'task', '\"team\"', NULL, \
         'content', 0.5, '[]', '{}', NULL, '2023-11-14T22:13:20.000Z', \
         '2023-11-14T22:13:20.000Z', 'soon')",
    )
    .execute(store.pool())
    .await
    .unwrap();
    let record = store.get("review-team", "id-a").await.unwrap().unwrap();
    assert_eq!(record.expires_at, None);
    assert_eq!(
        store
            .list(&TeamMemoryFilter::for_team("review-team"))
            .await
            .unwrap()
            .len(),
        1
    );
}

/// Inserts one row with handwritten column values, bypassing the record model.
async fn insert_raw(
    store: &SqliteTeamMemoryStore,
    id: &str,
    visibility_json: &str,
    tags_json: &str,
    metadata_json: &str,
    importance: f64,
    created_at: &str,
    updated_at: &str,
) {
    sqlx::query(
        "INSERT INTO team_memories (id, team_id, member_id, session_id, kind, visibility_json, \
         title, content, importance, tags_json, metadata_json, source, created_at, updated_at, \
         expires_at) VALUES (?1, 'review-team', NULL, NULL, 'task', ?2, NULL, 'raw row', ?3, \
         ?4, ?5, NULL, ?6, ?7, NULL)",
    )
    .bind(id)
    .bind(visibility_json)
    .bind(importance)
    .bind(tags_json)
    .bind(metadata_json)
    .bind(created_at)
    .bind(updated_at)
    .execute(store.pool())
    .await
    .unwrap();
}
