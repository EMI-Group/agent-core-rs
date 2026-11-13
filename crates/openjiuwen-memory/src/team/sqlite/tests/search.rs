//! Search visibility and keyword ranking through the shared ranker.
use super::*;
use crate::team::model::{TeamMemoryKind, TeamVisibility};

#[tokio::test]
async fn search_honours_visibility_and_keyword_ranking() {
    let store = store().await;
    let mut private = record(
        "review-team",
        Some("alice"),
        "id-private",
        "sqlite storage decision",
        TeamMemoryKind::Decision,
    );
    private.visibility = TeamVisibility::Private;
    let mut shared = record(
        "review-team",
        Some("bob"),
        "id-shared",
        "sqlite storage decision",
        TeamMemoryKind::Decision,
    );
    shared.visibility = TeamVisibility::Team;
    let mut stronger = record(
        "review-team",
        Some("bob"),
        "id-stronger",
        "sqlite sqlite storage storage decision",
        TeamMemoryKind::Decision,
    );
    stronger.visibility = TeamVisibility::Team;
    let unrelated = record(
        "review-team",
        Some("bob"),
        "id-unrelated",
        "gardening notes",
        TeamMemoryKind::Decision,
    );
    seed(&store, &[private, shared, stronger, unrelated]).await;
    let options = TeamSearchOptions::default();
    let alice = TeamMemoryScope::member("review-team", "alice");
    let hits = store
        .search(&alice, "sqlite storage", &options)
        .await
        .unwrap();
    let ids: Vec<&str> = hits.iter().map(|hit| hit.record.id.as_str()).collect();
    assert_eq!(ids, ["id-stronger", "id-private", "id-shared"]);
    assert!(hits[0].score >= hits[1].score);
    let bob = TeamMemoryScope::member("review-team", "bob");
    let hits = store
        .search(&bob, "sqlite storage", &options)
        .await
        .unwrap();
    let ids: Vec<&str> = hits.iter().map(|hit| hit.record.id.as_str()).collect();
    assert_eq!(ids, ["id-stronger", "id-shared"]);
    let leader = TeamMemoryScope::new("review-team");
    let hits = store
        .search(&leader, "sqlite storage", &options)
        .await
        .unwrap();
    assert_eq!(hits.len(), 3);
    let other_team = TeamMemoryScope::member("other-team", "alice");
    assert!(store
        .search(&other_team, "sqlite storage", &options)
        .await
        .unwrap()
        .is_empty());
    let capped = TeamSearchOptions::default().with_top_k(1);
    assert_eq!(
        store
            .search(&leader, "sqlite storage", &capped)
            .await
            .unwrap()
            .len(),
        1
    );
    let mut scoped = TeamSearchOptions::default().with_kinds([TeamMemoryKind::Task]);
    scoped = scoped.include_expired();
    assert!(store
        .search(&leader, "sqlite storage", &scoped)
        .await
        .unwrap()
        .is_empty());
    assert!(store
        .search(
            &leader,
            "sqlite storage",
            &TeamSearchOptions::default().with_top_k(0)
        )
        .await
        .is_err());
}

#[tokio::test]
async fn search_hides_expired_records_unless_requested() {
    let store = store().await;
    let expired = record(
        "review-team",
        Some("alice"),
        "id-expired",
        "sqlite storage",
        TeamMemoryKind::Decision,
    )
    .with_expires_at(at(1_700_000_000));
    store.upsert(&expired).await.unwrap();
    let scope = TeamMemoryScope::new("review-team");
    assert!(store
        .search(&scope, "sqlite storage", &TeamSearchOptions::default())
        .await
        .unwrap()
        .is_empty());
    let hits = store
        .search(
            &scope,
            "sqlite storage",
            &TeamSearchOptions::default().include_expired(),
        )
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
}
