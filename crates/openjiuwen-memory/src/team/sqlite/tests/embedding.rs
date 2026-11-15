//! The `embedding_json` column: round trip, the additive schema upgrade, and degradation.
//!
//! A vector is derived state, so the column is additive: a database written before team
//! memory carried embeddings keeps every row and simply gains NULL vectors, and a
//! malformed vector degrades to "no embedding" instead of failing a listing.
use super::*;
use crate::config::RankingConfig;
use crate::team::model::TeamMemoryKind;
use sqlx::sqlite::SqlitePoolOptions;

#[tokio::test]
async fn upsert_get_and_list_round_trip_the_vector() {
    let store = store().await;
    let embedded = record(
        "review-team",
        Some("alice"),
        "id-embedded",
        "sqlite storage",
        TeamMemoryKind::Decision,
    )
    .with_embedding(vec![0.25, -0.5, 1.0]);
    let plain = record(
        "review-team",
        None,
        "id-plain",
        "sqlite storage",
        TeamMemoryKind::Task,
    );
    seed(&store, &[embedded, plain]).await;
    let fetched = store
        .get("review-team", "id-embedded")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        fetched.embedding.as_deref(),
        Some([0.25, -0.5, 1.0].as_slice())
    );
    let listed = store
        .list(&TeamMemoryFilter::for_team("review-team"))
        .await
        .unwrap();
    let by_id =
        |id: &str| -> &TeamMemoryRecord { listed.iter().find(|record| record.id == id).unwrap() };
    assert_eq!(by_id("id-embedded").embedding, fetched.embedding);
    assert_eq!(by_id("id-plain").embedding, None);
    // Clearing the vector on an update clears the column.
    let mut cleared = fetched.clone();
    cleared.embedding = None;
    store.upsert(&cleared).await.unwrap();
    assert_eq!(
        store
            .get("review-team", "id-embedded")
            .await
            .unwrap()
            .unwrap()
            .embedding,
        None
    );
}

#[tokio::test]
async fn search_ranks_lexically_disjoint_rows_by_the_stored_vector() {
    let store = store().await;
    let related = updated_at(
        record(
            "review-team",
            None,
            "id-related",
            "postgres schema upgrade",
            TeamMemoryKind::Decision,
        ),
        1_700_000_000 - 90 * 86_400,
    )
    .with_importance(0.1)
    .with_embedding(vec![1.0, 0.0]);
    let unrelated = record(
        "review-team",
        None,
        "id-unrelated",
        "pizza for the launch lunch",
        TeamMemoryKind::Decision,
    )
    .with_importance(1.0)
    .with_embedding(vec![0.0, 1.0]);
    seed(&store, &[related, unrelated]).await;
    let options = TeamSearchOptions::default()
        .with_query_embedding(vec![1.0, 0.0])
        .with_ranking(RankingConfig::default());
    let hits = store
        .search(
            &TeamMemoryScope::new("review-team"),
            "database migration",
            &options,
        )
        .await
        .unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].record.id, "id-related");
    assert!((hits[0].similarity - 1.0).abs() < 1e-6);
    assert_eq!(hits[0].keyword, 0.0);
    assert!(hits[0].score > hits[1].score);
}

#[tokio::test]
async fn a_table_without_the_column_is_upgraded_and_keeps_its_rows() {
    // The pre-embedding schema, verbatim, plus one row written by that release.
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .connect("sqlite::memory:")
        .await
        .unwrap();
    sqlx::query(
        "CREATE TABLE team_memories (
            id TEXT PRIMARY KEY,
            team_id TEXT NOT NULL,
            member_id TEXT,
            session_id TEXT,
            kind TEXT NOT NULL,
            visibility_json TEXT NOT NULL,
            title TEXT,
            content TEXT NOT NULL,
            importance REAL NOT NULL,
            tags_json TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            source TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            expires_at TEXT
        )",
    )
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query(
        "INSERT INTO team_memories (id, team_id, member_id, session_id, kind, visibility_json, \
         title, content, importance, tags_json, metadata_json, source, created_at, updated_at, \
         expires_at) VALUES ('id-legacy', 'review-team', NULL, NULL, 'decision', '\"team\"', \
         NULL, 'sqlite storage', 0.5, '[]', '{}', NULL, '2023-11-14T22:13:20.000Z', \
         '2023-11-14T22:13:20.000Z', NULL)",
    )
    .execute(&pool)
    .await
    .unwrap();
    let store = SqliteTeamMemoryStore::from_pool(pool.clone())
        .await
        .unwrap();
    let legacy = store
        .get("review-team", "id-legacy")
        .await
        .unwrap()
        .expect("the legacy row survives the upgrade");
    assert_eq!(legacy.content, "sqlite storage");
    assert_eq!(legacy.embedding, None, "a legacy row has no vector");
    // The upgrade is idempotent: applying the schema again is a no-op.
    SqliteTeamMemoryStore::from_pool(pool).await.unwrap();
    // And a fresh write stores a vector in the added column.
    let embedded = record(
        "review-team",
        None,
        "id-new",
        "postgres schema upgrade",
        TeamMemoryKind::Decision,
    )
    .with_embedding(vec![1.0, 0.0]);
    store.upsert(&embedded).await.unwrap();
    assert_eq!(
        store.get("review-team", "id-new").await.unwrap().unwrap(),
        embedded
    );
}

#[tokio::test]
async fn a_malformed_vector_degrades_to_no_embedding() {
    let store = store().await;
    for (id, raw) in [
        ("id-not-json", "{not-json"),
        ("id-not-numbers", "\"text\""),
        ("id-empty", "[]"),
        ("id-non-finite", "[1.0, \"NaN\"]"),
    ] {
        sqlx::query(
            "INSERT INTO team_memories (id, team_id, member_id, session_id, kind, \
             visibility_json, title, content, importance, tags_json, metadata_json, source, \
             created_at, updated_at, expires_at, embedding_json) VALUES (?1, 'review-team', \
             NULL, NULL, 'decision', '\"team\"', NULL, 'sqlite storage', 0.5, '[]', '{}', \
             NULL, '2023-11-14T22:13:20.000Z', '2023-11-14T22:13:20.000Z', NULL, ?2)",
        )
        .bind(id)
        .bind(raw)
        .execute(store.pool())
        .await
        .unwrap();
        let record = store.get("review-team", id).await.unwrap().unwrap();
        assert_eq!(record.embedding, None, "{id}");
    }
    assert_eq!(
        store
            .list(&TeamMemoryFilter::for_team("review-team"))
            .await
            .unwrap()
            .len(),
        4,
        "a malformed vector never drops the row"
    );
}
