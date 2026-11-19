//! Tests of the repository lock: exclusivity, bounded wait, stale reclamation and release.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::Duration;

use tempfile::TempDir;

use super::lock::{acquire, LockOptions, DEFAULT_LOCK_WAIT};
use openjiuwen_common::error::Error;

fn lock_path(temp: &TempDir) -> std::path::PathBuf {
    temp.path().join(".git").join(super::lock::LOCK_FILE_NAME)
}

#[tokio::test]
async fn lock_is_exclusive_until_released() {
    let temp = TempDir::new().unwrap();
    let path = lock_path(&temp);
    let guard = acquire(&path, &LockOptions::default()).await.unwrap();
    assert!(path.is_file(), "the lock file records the holder");

    let err = acquire(&path, &LockOptions::new(Duration::from_millis(75)))
        .await
        .expect_err("a held lock is not granted twice");
    assert!(matches!(err, Error::Timeout(_)), "{err:?}");
    assert!(
        err.to_string().contains("waiting for the worktree lock"),
        "{err}"
    );

    guard.release().await.unwrap();
    assert!(!path.exists(), "releasing removes the lock file");
    let guard = acquire(&path, &LockOptions::new(Duration::from_millis(75)))
        .await
        .expect("a released lock can be re-acquired");
    drop(guard);
    assert!(!path.exists(), "dropping the guard releases the lock");
}

#[tokio::test]
async fn lock_waits_for_the_current_holder() {
    let temp = TempDir::new().unwrap();
    let path = lock_path(&temp);
    let options = LockOptions::new(Duration::from_secs(10)).with_poll(Duration::from_millis(5));

    let first = acquire(&path, &options).await.unwrap();
    let waiter = tokio::spawn({
        let path = path.clone();
        async move { acquire(&path, &options).await }
    });
    // Give the waiter a chance to start polling, then release.
    tokio::time::sleep(Duration::from_millis(40)).await;
    first.release().await.unwrap();

    let guard = tokio::time::timeout(DEFAULT_LOCK_WAIT, waiter)
        .await
        .expect("the waiter acquires the lock once it is released")
        .expect("the waiting task did not panic")
        .expect("the lock was released");
    guard.release().await.unwrap();
}

#[tokio::test]
async fn a_stale_lock_file_is_reclaimed() {
    let temp = TempDir::new().unwrap();
    let path = lock_path(&temp);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, "pid=999999\n").unwrap();

    // A zero stale bound treats the file as abandoned immediately.
    let options = LockOptions::new(Duration::from_secs(1)).with_stale_after(Duration::ZERO);
    let guard = acquire(&path, &options)
        .await
        .expect("stale lock reclaimed");
    assert!(path.is_file(), "the reclaimed file belongs to this guard");
    guard.release().await.unwrap();
}

#[tokio::test]
async fn a_fresh_lock_file_is_not_stolen() {
    let temp = TempDir::new().unwrap();
    let path = lock_path(&temp);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, "pid=999999\n").unwrap();

    let options =
        LockOptions::new(Duration::from_millis(50)).with_stale_after(Duration::from_secs(3600));
    let err = acquire(&path, &options)
        .await
        .expect_err("a fresh lock file is respected");
    assert!(matches!(err, Error::Timeout(_)), "{err:?}");
}

#[tokio::test]
async fn acquiring_creates_the_lock_directory() {
    let temp = TempDir::new().unwrap();
    let path = temp
        .path()
        .join("nested/deeper")
        .join(super::lock::LOCK_FILE_NAME);
    let guard = acquire(&path, &LockOptions::default()).await.unwrap();
    assert_eq!(guard.path(), path.as_path());
    assert!(path.parent().unwrap().is_dir());
    guard.release().await.unwrap();
}

/// Options whose stale window is far too short for a heartbeat, so a guard can be declared stale.
fn stealable() -> LockOptions {
    LockOptions::new(Duration::from_millis(50))
        .with_stale_after(Duration::from_millis(10))
        .with_poll(Duration::from_millis(1))
}

/// Options that never reclaim a fresh file, however long the caller polls.
fn possessive() -> LockOptions {
    LockOptions::new(Duration::from_millis(50)).with_stale_after(Duration::from_secs(3600))
}

#[tokio::test]
async fn a_stolen_lock_is_not_deleted_by_its_previous_holder() {
    let temp = TempDir::new().unwrap();
    let path = lock_path(&temp);

    // A takes the lock, then loses its heartbeat so a waiter may declare it stale.
    let mut first = acquire(&path, &stealable()).await.unwrap();
    first.suspend_heartbeat();
    tokio::time::sleep(Duration::from_millis(80)).await;

    // B reclaims the abandoned file and stamps its own token into a new one.
    let second = acquire(&path, &stealable())
        .await
        .expect("the stale lock is reclaimed");
    assert_ne!(first.token(), second.token(), "every holder has its own token");
    let holder = std::fs::read_to_string(&path).unwrap();
    assert!(
        holder.contains(second.token()),
        "the file belongs to the new holder: {holder}"
    );

    // A finishes late: its release must not take B's file with it.
    first.release().await.unwrap();
    assert!(
        path.is_file(),
        "the current holder's lock file survives a stale guard's release"
    );
    let holder = std::fs::read_to_string(&path).unwrap();
    assert!(
        holder.contains(second.token()),
        "the survivor is still B's file: {holder}"
    );

    // Mutual exclusion holds: while B owns the lock, C is not admitted.
    let err = acquire(&path, &possessive())
        .await
        .expect_err("B still holds the lock");
    assert!(matches!(err, Error::Timeout(_)), "{err:?}");

    // B's own release does remove its file, and only then is C admitted.
    second.release().await.unwrap();
    assert!(!path.exists(), "the owner removes its own lock file");
    let third = acquire(&path, &possessive())
        .await
        .expect("a released lock can be re-acquired");
    third.release().await.unwrap();
}

#[tokio::test]
async fn a_live_holder_is_never_declared_stale() {
    let temp = TempDir::new().unwrap();
    let path = lock_path(&temp);
    // The heartbeat refreshes the stamp every ~66ms, so the 200ms window never expires while the
    // guard lives — even though the critical section outlasts it twice over.
    let options = LockOptions::new(Duration::from_millis(80))
        .with_stale_after(Duration::from_millis(200))
        .with_poll(Duration::from_millis(5));

    let guard = acquire(&path, &options).await.unwrap();
    tokio::time::sleep(Duration::from_millis(700)).await;
    let err = acquire(&path, &options)
        .await
        .expect_err("a provably alive holder keeps its lock");
    assert!(matches!(err, Error::Timeout(_)), "{err:?}");

    // It still owns the file, so releasing removes it.
    guard.release().await.unwrap();
    assert!(!path.exists(), "the live guard removed its own file");
}

#[tokio::test]
async fn the_lock_file_records_the_holder_pid_and_a_unique_token() {
    let temp = TempDir::new().unwrap();
    let path = lock_path(&temp);
    let guard = acquire(&path, &LockOptions::default()).await.unwrap();
    let content = std::fs::read_to_string(&path).unwrap();
    assert!(
        content.contains(&format!("pid={}", std::process::id())),
        "{content}"
    );
    assert!(
        guard.token().starts_with(&format!("{}-", std::process::id())),
        "the token names the holding process: {}",
        guard.token()
    );
    assert!(content.contains(guard.token()), "{content}");
    guard.release().await.unwrap();
    assert!(!path.exists());
}
