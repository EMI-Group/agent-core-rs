//! The schema guard: idempotent DDL, directory creation and rejection of a newer database.

use openjiuwen_common::error::ErrorKind;
use sqlx::sqlite::SqliteConnectOptions;
use sqlx::Connection;

use crate::schema::TaskRecord;
use crate::storage::TeamStore;

use crate::storage::SqliteTeamStore;

#[tokio::test]
async fn opening_the_same_file_twice_is_idempotent() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("team.db");
    {
        let store = SqliteTeamStore::open(&path).await.unwrap();
        store
            .insert_task(TaskRecord::new("alpha", "t1", "kept", "body"))
            .await
            .unwrap();
    }
    // The second open re-applies the whole `CREATE ... IF NOT EXISTS` batch and keeps the row.
    let store = SqliteTeamStore::open(&path).await.unwrap();
    let stored = store.task("alpha", "t1").await.unwrap().unwrap();
    assert_eq!(stored.title, "kept");
    // A third open proves the same again, and the database still accepts writes.
    let store = SqliteTeamStore::open(&path).await.unwrap();
    assert!(store
        .transition_task(
            "alpha",
            "t1",
            crate::schema::TaskStatus::Pending,
            crate::schema::TaskStatus::InProgress
        )
        .await
        .unwrap());
}

#[tokio::test]
async fn a_newer_schema_version_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("team.db");
    {
        let store = SqliteTeamStore::open(&path).await.unwrap();
        store
            .insert_task(TaskRecord::new("alpha", "t1", "kept", "body"))
            .await
            .unwrap();
    }

    let mut connection =
        sqlx::SqliteConnection::connect_with(&SqliteConnectOptions::new().filename(&path))
            .await
            .unwrap();
    let version: i64 = sqlx::query_scalar("PRAGMA user_version")
        .fetch_one(&mut connection)
        .await
        .unwrap();
    assert_eq!(version, 1, "a fresh database is stamped with version 1");
    sqlx::query("PRAGMA user_version = 99")
        .execute(&mut connection)
        .await
        .unwrap();
    connection.close().await.unwrap();

    let err = SqliteTeamStore::open(&path)
        .await
        .expect_err("a newer schema version is refused");
    assert_eq!(err.kind(), ErrorKind::Unsupported);
    let message = err.to_string();
    assert!(message.contains("newer than this build"), "{message}");
    assert!(message.contains("99"), "{message}");
}

#[tokio::test]
async fn open_creates_a_missing_parent_directory() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("nested").join("deeper").join("team.db");
    assert!(!path.parent().unwrap().exists());
    let store = SqliteTeamStore::open(&path).await.unwrap();
    assert!(path.parent().unwrap().exists());
    assert!(path.exists());
    store
        .insert_task(TaskRecord::new("alpha", "t1", "nested", "body"))
        .await
        .unwrap();
    assert_eq!(
        store.task("alpha", "t1").await.unwrap().unwrap().title,
        "nested"
    );
}
