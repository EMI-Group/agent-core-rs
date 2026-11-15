//! Cross-process lock suite for the auth-store file.
//!
//! `flock(2)` locks belong to the open file description, so two handles inside one
//! process conflict exactly like two processes do: every case here is a genuine
//! two-writer test, not a simulation.

#![cfg(unix)]

use std::path::Path;
use std::time::Duration;

use openjiuwen_common::error::ErrorKind;
use serde_json::{json, Value};

use super::lock::{lock_path, TokenFileLock};
use super::store::{FileTokenStore, OAuthToken, TokenStore};

/// Reads the auth-store document at `path`, treating an absent file as `{}`.
fn read_document(path: &Path) -> Value {
    std::fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_else(|| json!({}))
}

#[test]
fn lock_path_is_a_sibling_of_the_auth_store() {
    assert_eq!(
        lock_path(Path::new("/home/tester/.openjiuwen/auth.json")),
        Path::new("/home/tester/.openjiuwen/auth.json.lock")
    );
    assert_eq!(
        FileTokenStore::new("/tmp/openjiuwen/auth.json").lock_path(),
        Path::new("/tmp/openjiuwen/auth.json.lock")
    );
    assert_eq!(
        lock_path(Path::new("auth.json")),
        Path::new("auth.json.lock"),
        "a relative path keeps its directory"
    );
}

#[tokio::test]
async fn try_lock_reports_contention_and_succeeds_after_the_release() {
    let directory = tempfile::tempdir().expect("a temp directory");
    let auth_path = directory.path().join("auth.json");
    let first = FileTokenStore::new(&auth_path);
    let second = FileTokenStore::new(&auth_path);

    let held = first.try_lock().await.expect("the lock is free");
    assert_eq!(held.auth_path(), auth_path.as_path());
    assert_eq!(held.lock_path(), lock_path(&auth_path));

    let error = second
        .try_lock()
        .await
        .expect_err("another handle holds the lock");
    assert_eq!(error.kind(), ErrorKind::External);
    let message = error.to_string();
    assert!(
        message.contains("held by another process"),
        "message: {message}"
    );
    assert!(message.contains("auth.json.lock"), "message: {message}");

    drop(held);

    let reacquired = second
        .try_lock()
        .await
        .expect("the released lock is free again");
    assert_eq!(reacquired.auth_path(), auth_path.as_path());
}

#[tokio::test]
async fn the_blocking_lock_times_out_with_a_typed_error() {
    let directory = tempfile::tempdir().expect("a temp directory");
    let auth_path = directory.path().join("auth.json");
    let first = FileTokenStore::new(&auth_path);
    let second = FileTokenStore::new(&auth_path);

    let held = first.try_lock().await.expect("the lock is free");
    let error = second
        .lock_with_timeout(Duration::from_millis(60))
        .await
        .expect_err("the holder keeps the lock");
    assert_eq!(error.kind(), ErrorKind::Timeout);
    assert!(
        error.to_string().contains("auth.json.lock"),
        "message: {error}"
    );

    drop(held);

    let reacquired = second
        .lock_with_timeout(Duration::from_millis(60))
        .await
        .expect("the released lock is acquired");
    assert_eq!(reacquired.auth_path(), auth_path.as_path());
}

#[tokio::test]
async fn the_guard_reads_writes_and_clears_under_the_lock() {
    let directory = tempfile::tempdir().expect("a temp directory");
    let auth_path = directory.path().join("auth.json");
    let store = FileTokenStore::new(&auth_path);

    let guard = store.lock().await.expect("the lock is free");
    assert!(guard.load().await.expect("the load").is_none());
    assert!(
        !guard.clear().await.expect("the clear"),
        "clearing an absent credential reports nothing removed"
    );
    guard
        .save(&OAuthToken::new("access-1").with_refresh_token("refresh-1"))
        .await
        .expect("the saved token");
    let loaded = guard.load().await.expect("the load").expect("the token");
    assert_eq!(loaded.access_token.expose_secret(), "access-1");
    assert!(guard.clear().await.expect("the clear"));
    assert!(guard.load().await.expect("the load").is_none());
}

#[tokio::test]
async fn the_guard_serializes_the_whole_cycle_when_acquired_directly() {
    let directory = tempfile::tempdir().expect("a temp directory");
    let auth_path = directory.path().join("auth.json");
    let store = FileTokenStore::new(&auth_path);

    let guard = TokenFileLock::try_acquire(&auth_path, "https://example.test").expect("the lock");
    guard
        .save(&OAuthToken::new("access-1"))
        .await
        .expect("the saved token");
    assert_eq!(
        store
            .load()
            .await
            .expect("the read")
            .expect("the token")
            .access_token
            .expose_secret(),
        "access-1"
    );
    assert_eq!(guard.base_url(), "https://example.test");
}

#[tokio::test]
async fn concurrent_read_modify_write_cycles_lose_no_entry() {
    let directory = tempfile::tempdir().expect("a temp directory");
    let auth_path = directory.path().join("auth.json");
    let account = FileTokenStore::new(&auth_path).with_base_url("https://account.test");

    for round in 0..8_u32 {
        let token = OAuthToken::new(format!("access-{round}")).with_refresh_token("refresh");
        let foreign = format!("Foreign{round}");
        let path = auth_path.clone();
        let store = account.clone();
        // Two writers race: the account store merges its own entry, the "other
        // process" adds a provider entry while following the same lock protocol.
        let (saved, ()) = tokio::join!(store.save(&token), async move {
            let other = FileTokenStore::new(&path);
            let _guard = other.lock().await.expect("the lock");
            let mut document = read_document(&path);
            document["providers"][&foreign] = json!({ "tokens": { "access_token": "foreign" } });
            std::fs::write(&path, serde_json::to_vec(&document).expect("the document"))
                .expect("the rewritten store");
        });
        saved.expect("the account save");
    }

    let document = read_document(&auth_path);
    assert!(
        document["providers"]["OpenAIAccount"]["tokens"]["access_token"]
            .as_str()
            .is_some_and(|value| value.starts_with("access-")),
        "the account entry survived every round: {document}"
    );
    for round in 0..8_u32 {
        assert_eq!(
            document["providers"][format!("Foreign{round}")]["tokens"]["access_token"],
            "foreign",
            "round {round} lost the other provider's entry: {document}"
        );
    }
}

#[tokio::test]
async fn concurrent_saves_of_one_entry_leave_a_complete_document() {
    let directory = tempfile::tempdir().expect("a temp directory");
    let auth_path = directory.path().join("auth.json");
    let store = FileTokenStore::new(&auth_path);
    store
        .save(&OAuthToken::new("seed"))
        .await
        .expect("the seeded token");

    let writers: Vec<_> = (0..8_u32)
        .map(|round| {
            let store = store.clone();
            tokio::spawn(async move {
                store
                    .save(&OAuthToken::new(format!("access-{round}")).with_refresh_token("r"))
                    .await
            })
        })
        .collect();
    for writer in writers {
        writer
            .await
            .expect("the writer task joins")
            .expect("the concurrent save");
    }

    let document = read_document(&auth_path);
    let access = document["providers"]["OpenAIAccount"]["tokens"]["access_token"]
        .as_str()
        .expect("the merged entry kept a token");
    assert!(access.starts_with("access-"), "access: {access}");
    assert_eq!(
        document["providers"]["OpenAIAccount"]["tokens"]["refresh_token"],
        "r"
    );
}
