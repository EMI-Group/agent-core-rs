//! `list` filtering, ordering, pagination, `stats`, and expiry visibility.
use super::*;
use crate::team::model::{TeamMemoryKind, TeamVisibility, TEAM_WIDE_MEMBER_KEY};

#[tokio::test]
async fn list_filters_combine() {
    let store = store().await;
    let mut expired = record(
        "review-team",
        Some("alice"),
        "id-expired",
        "expired note",
        TeamMemoryKind::Task,
    )
    .with_tags(["storage"]);
    expired.expires_at = Some(at(1_700_000_100));
    let mut tagged = record(
        "review-team",
        Some("bob"),
        "id-tagged",
        "tagged note",
        TeamMemoryKind::Decision,
    )
    .with_tags(["storage", "sqlite"])
    .with_title("Storage choice");
    tagged.session_id = Some("s1".to_owned());
    tagged.visibility = TeamVisibility::Private;
    tagged.importance = 0.9;
    let plain = updated_at(
        record(
            "review-team",
            None,
            "id-plain",
            "plain note",
            TeamMemoryKind::SharedFact,
        ),
        1_700_000_400,
    );
    seed(&store, &[expired, tagged, plain]).await;
    let team = TeamMemoryFilter::for_team("review-team");
    let listed = store.list(&team).await.unwrap();
    assert_eq!(listed.len(), 2, "the expired record is hidden");
    assert!(!listed.iter().any(|record| record.id == "id-expired"));
    let mut include_expired = team.clone();
    include_expired.include_expired = true;
    assert_eq!(store.list(&include_expired).await.unwrap().len(), 3);
    let by_member = store
        .list(&TeamMemoryFilter::for_team("review-team").member("bob"))
        .await
        .unwrap();
    assert_eq!(by_member.len(), 1);
    assert_eq!(by_member[0].id, "id-tagged");
    let by_kind = store
        .list(&TeamMemoryFilter::for_team("review-team").kind(TeamMemoryKind::Decision))
        .await
        .unwrap();
    assert_eq!(by_kind.len(), 1);
    assert_eq!(by_kind[0].id, "id-tagged");
    let mut by_tag = team.clone();
    by_tag.tags = vec!["STORAGE".to_owned()];
    assert_eq!(store.list(&by_tag).await.unwrap().len(), 1);
    let mut by_visibility = team.clone();
    by_visibility.visibility = Some(TeamVisibility::Private);
    assert_eq!(store.list(&by_visibility).await.unwrap().len(), 1);
    let mut by_text = team.clone();
    by_text.text_contains = Some("STORAGE CHOICE".to_owned());
    let listed = store.list(&by_text).await.unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].id, "id-tagged");
    let mut by_title_fragment = team.clone();
    by_title_fragment.text_contains = Some("choice".to_owned());
    assert_eq!(store.list(&by_title_fragment).await.unwrap().len(), 1);
    let mut by_importance = team.clone();
    by_importance.min_importance = Some(0.9);
    let listed = store.list(&by_importance).await.unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].id, "id-tagged");
    let mut min_inclusive = team.clone();
    min_inclusive.min_importance = Some(0.5);
    assert_eq!(store.list(&min_inclusive).await.unwrap().len(), 2);
    let mut by_time = team.clone();
    by_time.updated_after = Some(at(1_700_000_100));
    let listed = store.list(&by_time).await.unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].id, "id-plain");
    let mut by_created = team.clone();
    by_created.created_before = Some(at(1_700_000_001));
    assert_eq!(store.list(&by_created).await.unwrap().len(), 2);
    let mut by_session = team;
    by_session.session_id = Some("s1".to_owned());
    assert_eq!(store.list(&by_session).await.unwrap().len(), 1);
}

#[tokio::test]
async fn list_orders_by_updated_at_desc_then_id_asc() {
    let store = store().await;
    let records = vec![
        updated_at(
            record(
                "review-team",
                None,
                "id-b",
                "same time",
                TeamMemoryKind::Task,
            ),
            1_700_000_300,
        ),
        updated_at(
            record(
                "review-team",
                None,
                "id-a",
                "same time",
                TeamMemoryKind::Task,
            ),
            1_700_000_300,
        ),
        updated_at(
            record(
                "review-team",
                None,
                "id-newest",
                "newest",
                TeamMemoryKind::Task,
            ),
            1_700_000_900,
        ),
        updated_at(
            record(
                "review-team",
                None,
                "id-oldest",
                "oldest",
                TeamMemoryKind::Task,
            ),
            1_700_000_100,
        ),
    ];
    seed(&store, &records).await;
    let listed = store
        .list(&TeamMemoryFilter::for_team("review-team"))
        .await
        .unwrap();
    let ids: Vec<&str> = listed.iter().map(|record| record.id.as_str()).collect();
    assert_eq!(ids, ["id-newest", "id-a", "id-b", "id-oldest"]);
}

#[tokio::test]
async fn list_paginates_with_limit_and_offset() {
    let store = store().await;
    let records: Vec<TeamMemoryRecord> = (0..5)
        .map(|index| {
            updated_at(
                record(
                    "review-team",
                    None,
                    &format!("id-{index}"),
                    "content",
                    TeamMemoryKind::Task,
                ),
                1_700_000_000 + index,
            )
        })
        .collect();
    seed(&store, &records).await;
    let page = |limit: usize, offset: usize| {
        TeamMemoryFilter::for_team("review-team")
            .with_limit(limit)
            .with_offset(offset)
    };
    let ids = |records: Vec<TeamMemoryRecord>| -> Vec<String> {
        records.into_iter().map(|record| record.id).collect()
    };
    assert_eq!(
        ids(store.list(&page(2, 0)).await.unwrap()),
        ["id-4", "id-3"]
    );
    assert_eq!(
        ids(store.list(&page(2, 2)).await.unwrap()),
        ["id-2", "id-1"]
    );
    assert_eq!(ids(store.list(&page(2, 4)).await.unwrap()), ["id-0"]);
    assert!(store.list(&page(2, 9)).await.unwrap().is_empty());
    let offset_only = TeamMemoryFilter::for_team("review-team").with_offset(3);
    assert_eq!(
        ids(store.list(&offset_only).await.unwrap()),
        ["id-1", "id-0"]
    );
    assert!(store
        .list(&TeamMemoryFilter::for_team("review-team").with_limit(0))
        .await
        .is_err());
}

#[tokio::test]
async fn stats_aggregate_the_filtered_view() {
    let store = store().await;
    let mut expired = record(
        "review-team",
        Some("alice"),
        "id-expired",
        "expired",
        TeamMemoryKind::Decision,
    );
    expired.expires_at = Some(at(1_700_000_050));
    let decision = updated_at(
        record(
            "review-team",
            Some("alice"),
            "id-a",
            "decision",
            TeamMemoryKind::Decision,
        ),
        1_700_000_100,
    );
    let task = updated_at(
        record("review-team", None, "id-b", "task", TeamMemoryKind::Task),
        1_700_000_200,
    );
    seed(&store, &[expired, decision, task]).await;
    let stats = store
        .stats(&TeamMemoryFilter::for_team("review-team"))
        .await
        .unwrap();
    assert_eq!(stats.total, 2);
    assert_eq!(stats.by_kind.get("decision"), Some(&1));
    assert_eq!(stats.by_kind.get("task"), Some(&1));
    assert_eq!(stats.by_member.get("alice"), Some(&1));
    assert_eq!(stats.by_member.get(TEAM_WIDE_MEMBER_KEY), Some(&1));
    assert_eq!(stats.expired, 0);
    assert_eq!(stats.oldest, Some(at(1_700_000_000)));
    assert_eq!(stats.newest, Some(at(1_700_000_000)));
    let mut with_expired = TeamMemoryFilter::for_team("review-team");
    with_expired.include_expired = true;
    let stats = store.stats(&with_expired).await.unwrap();
    assert_eq!(stats.total, 3);
    assert_eq!(stats.expired, 1);
    assert_eq!(stats.by_kind.get("decision"), Some(&2));
    assert_eq!(
        store
            .stats(&TeamMemoryFilter::for_team("other-team"))
            .await
            .unwrap(),
        crate::team::model::TeamMemoryStats::default()
    );
}

#[tokio::test]
async fn stats_respects_member_and_kind_filters() {
    let store = store().await;
    seed(
        &store,
        &[
            record(
                "review-team",
                Some("alice"),
                "id-a",
                "decision",
                TeamMemoryKind::Decision,
            ),
            record(
                "review-team",
                Some("bob"),
                "id-b",
                "task",
                TeamMemoryKind::Task,
            ),
        ],
    )
    .await;
    let stats = store
        .stats(&TeamMemoryFilter::for_team("review-team").member("bob"))
        .await
        .unwrap();
    assert_eq!(stats.total, 1);
    assert_eq!(stats.by_member.get("bob"), Some(&1));
    assert_eq!(stats.by_kind.get("task"), Some(&1));
    assert_eq!(stats.by_kind.get("decision"), None);
}

#[tokio::test]
async fn expiry_boundary_is_inclusive_and_prunable() {
    let store = store().await;
    let boundary = record(
        "review-team",
        Some("alice"),
        "id-boundary",
        "expires exactly now",
        TeamMemoryKind::Task,
    )
    .with_expires_at(at(1_700_000_100));
    let live = record(
        "review-team",
        Some("alice"),
        "id-live",
        "still live",
        TeamMemoryKind::Task,
    )
    .with_expires_at(at(1_700_000_200));
    let forever = record(
        "review-team",
        Some("alice"),
        "id-forever",
        "never expires",
        TeamMemoryKind::Task,
    );
    seed(&store, &[boundary, live, forever]).await;
    assert_eq!(store.prune_expired(at(1_700_000_100)).await.unwrap(), 1);
    assert!(store
        .get("review-team", "id-boundary")
        .await
        .unwrap()
        .is_none());
    assert_eq!(store.prune_expired(at(1_700_000_100)).await.unwrap(), 0);
    assert_eq!(store.prune_expired(at(1_700_000_200)).await.unwrap(), 1);
    assert_eq!(store.prune_expired(at(1_800_000_000)).await.unwrap(), 0);
    assert!(store
        .get("review-team", "id-forever")
        .await
        .unwrap()
        .is_some());
}

#[tokio::test]
async fn expired_records_stay_readable_until_requested() {
    let store = store().await;
    let expired = record(
        "review-team",
        Some("alice"),
        "id-a",
        "expired",
        TeamMemoryKind::Task,
    )
    .with_expires_at(at(1_700_000_000));
    store.upsert(&expired).await.unwrap();
    // `get` returns expired rows; the listing hides them unless asked.
    assert_eq!(
        store.get("review-team", "id-a").await.unwrap(),
        Some(expired)
    );
    assert!(store
        .list(&TeamMemoryFilter::for_team("review-team"))
        .await
        .unwrap()
        .is_empty());
    let mut include_expired = TeamMemoryFilter::for_team("review-team");
    include_expired.include_expired = true;
    assert_eq!(store.list(&include_expired).await.unwrap().len(), 1);
}
