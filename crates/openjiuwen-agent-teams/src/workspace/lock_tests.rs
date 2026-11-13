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
