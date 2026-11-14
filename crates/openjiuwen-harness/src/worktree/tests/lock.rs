//! `WorktreeLock` scenarios: lock-file layout, the disabled handle, contention
//! between two handles in one process, the bounded wait, and the recorded
//! metadata/staleness reporting.
//!
//! Everything is offline and deterministic: temp directories, no wall-clock
//! ordering, and every scenario that needs a real `flock` checks
//! [`WorktreeLock::is_enabled`] first and returns early where the platform (or
//! the configuration) cannot lock. Cross-process exclusion is asserted with two
//! independent handles in one process because the underlying `flock` excludes
//! separate open file descriptions even inside one process; no child process is
//! spawned, so the suite stays offline and deterministic.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use std::path::Path;
use std::time::Duration;

use chrono::Utc;
use openjiuwen_common::error::{Error, ErrorKind};

use super::temp_root;
use crate::worktree::{
    is_lock_contention, is_lock_timeout, WorktreeLock, WorktreeLockConfig, DEFAULT_LOCK_TIMEOUT,
    LOCK_CONTENTION_MARKER, LOCK_TIMEOUT_MARKER, STALE_LOCK_AGE,
};

/// Creates `repo` with a `.git` directory, returning the temp dir and the root.
fn git_repo() -> (tempfile::TempDir, std::path::PathBuf) {
    let (dir, repo) = temp_root();
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    (dir, repo)
}

/// A lock over `repo` (workspace = repo) with `config`.
fn lock_for(repo: &Path, config: WorktreeLockConfig) -> WorktreeLock {
    WorktreeLock::new(repo, repo, config)
}

#[test]
fn lock_path_prefers_the_git_directory() {
    let (_repo_dir, repo) = git_repo();
    let (_ws_dir, workspace) = temp_root();
    let lock = WorktreeLock::new(&repo, &workspace, WorktreeLockConfig::default());
    assert_eq!(lock.root(), repo.join(".git").as_path());
    assert_eq!(lock.repo_root(), repo.as_path());
    let path = lock.lock_path();
    assert_eq!(path.parent().unwrap(), repo.join(".git").as_path());
    assert_eq!(path.extension().and_then(|ext| ext.to_str()), Some("lock"));
    assert!(!path.exists(), "construction must not create the lock file");
}

#[test]
fn lock_path_falls_back_to_the_workspace_registry_directory() {
    let (_repo_dir, repo) = temp_root();
    let (_ws_dir, workspace) = temp_root();
    let lock = WorktreeLock::new(&repo, &workspace, WorktreeLockConfig::default());
    assert_eq!(lock.root(), workspace.join(".worktree").as_path());
    assert_eq!(
        lock.lock_path().parent().unwrap(),
        workspace.join(".worktree").as_path()
    );
}

#[test]
fn two_repositories_never_share_a_lock_file() {
    let (_first_dir, first) = git_repo();
    let (_second_dir, second) = git_repo();
    let left = lock_for(&first, WorktreeLockConfig::default());
    let right = lock_for(&second, WorktreeLockConfig::default());
    assert_ne!(left.lock_path(), right.lock_path());
}

#[test]
fn default_configuration_matches_python_parity() {
    let config = WorktreeLockConfig::default();
    assert!(config.enabled);
    assert_eq!(config.timeout, DEFAULT_LOCK_TIMEOUT);
    assert_eq!(DEFAULT_LOCK_TIMEOUT, Duration::from_secs(120));
    assert_eq!(STALE_LOCK_AGE, Duration::from_secs(30 * 60));
    assert!(!WorktreeLockConfig::disabled().enabled);
    assert_eq!(
        config.with_timeout(Duration::from_millis(5)).timeout,
        Duration::from_millis(5)
    );
}

#[tokio::test]
async fn disabled_configuration_yields_an_inactive_guard() {
    let (_repo_dir, repo) = git_repo();
    let lock = lock_for(&repo, WorktreeLockConfig::disabled());
    assert!(!lock.is_enabled());
    assert!(!lock.config().enabled);

    let guard = lock.acquire().await.unwrap();
    assert!(!guard.is_active());
    assert!(guard.metadata().is_none());
    assert_eq!(guard.lock_path(), lock.lock_path());
    drop(guard);

    let guard = lock.try_acquire().await.unwrap();
    assert!(!guard.is_active());
    assert!(guard.metadata().is_none());
    // No lock file (and no metadata) is created for a disabled lock.
    assert!(!lock.lock_path().exists());
}

#[test]
fn malformed_metadata_is_reported_as_absent() {
    let (_repo_dir, repo) = git_repo();
    let lock = lock_for(&repo, WorktreeLockConfig::default());
    assert!(lock.read_metadata().is_none(), "no lock file yet");
    let path = lock.lock_path();
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, "garbage\npid=not-a-number\n").unwrap();
    assert!(lock.read_metadata().is_none());
    assert!(lock.stale_metadata(Utc::now()).is_none());
}

#[cfg(unix)]
#[tokio::test]
async fn two_handles_contend_within_one_process() {
    let (_repo_dir, repo) = git_repo();
    let first = lock_for(&repo, WorktreeLockConfig::default());
    let second = lock_for(&repo, WorktreeLockConfig::default());
    if !first.is_enabled() || !second.is_enabled() {
        // No `flock` on this platform: nothing excludes, so nothing contends.
        return;
    }
    let held = first.try_acquire().await.unwrap();
    assert!(held.is_active());

    let error = second.try_acquire().await.expect_err("contended");
    assert!(error.is_contention());
    assert!(!error.is_timeout());
    assert_eq!(error.waited(), None);
    assert_eq!(error.target(), repo.as_path());
    assert_eq!(error.lock_path(), first.lock_path());

    drop(held);
    let again = second.try_acquire().await.unwrap();
    assert!(
        again.is_active(),
        "dropping the guard must release the lock"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn contention_converts_to_a_marked_tool_error() {
    let (_repo_dir, repo) = git_repo();
    let first = lock_for(&repo, WorktreeLockConfig::default());
    let second = lock_for(&repo, WorktreeLockConfig::default());
    if !first.is_enabled() {
        return;
    }
    let held = first.try_acquire().await.unwrap();
    let error = second.try_acquire().await.expect_err("contended");

    let lock_path = first.lock_path();
    assert!(
        error.to_string().contains(&lock_path.display().to_string()),
        "display must name the lock file: {error}"
    );

    let converted: Error = error.into();
    assert_eq!(converted.kind(), ErrorKind::Tool);
    assert!(is_lock_contention(&converted));
    assert!(!is_lock_timeout(&converted));
    assert!(converted.to_string().contains(LOCK_CONTENTION_MARKER));
    drop(held);
}

#[cfg(unix)]
#[tokio::test]
async fn waiting_acquisition_times_out_while_the_lock_is_held() {
    let (_repo_dir, repo) = git_repo();
    let holder = lock_for(&repo, WorktreeLockConfig::default());
    let waiter = lock_for(
        &repo,
        WorktreeLockConfig::default().with_timeout(Duration::from_millis(50)),
    );
    if !holder.is_enabled() || !waiter.is_enabled() {
        return;
    }
    let held = holder.acquire().await.unwrap();
    assert!(held.is_active());

    let error = waiter.acquire().await.expect_err("timed out");
    assert!(error.is_timeout(), "{error}");
    let waited_for = error.waited().expect("a timed-out acquisition waited");
    assert!(waited_for > Duration::ZERO, "waited {waited_for:?}");
    assert_eq!(error.target(), repo.as_path());

    let converted: Error = error.into();
    assert!(is_lock_timeout(&converted));
    assert!(!is_lock_contention(&converted));
    let message = converted.to_string();
    assert!(message.contains(LOCK_TIMEOUT_MARKER), "{message}");
    assert!(message.contains(&repo.display().to_string()), "{message}");
    assert!(
        message.contains(&holder.lock_path().display().to_string()),
        "{message}"
    );
    drop(held);
}

#[cfg(unix)]
#[tokio::test]
async fn acquisition_records_metadata_and_stale_metadata_ages_out() {
    let (_repo_dir, repo) = git_repo();
    let lock = lock_for(&repo, WorktreeLockConfig::default());
    if !lock.is_enabled() {
        return;
    }
    let guard = lock.acquire().await.unwrap();
    assert!(guard.is_active());

    let metadata = guard.metadata().expect("metadata recorded");
    assert_eq!(metadata.pid, std::process::id());
    assert_ne!(metadata.acquired_at, "");
    assert_eq!(lock.read_metadata().as_ref(), Some(metadata));

    let now = Utc::now();
    assert!(
        lock.stale_metadata(now).is_none(),
        "a fresh acquisition is never stale"
    );

    // Rewrite the sidecar with an old timestamp: the lock is still held, so only
    // the diagnostic content changes.
    let path = lock.lock_path();
    std::fs::write(&path, "pid=1\nacquired_at=2020-01-01T00:00:00+00:00\n").unwrap();
    let stale = lock.stale_metadata(now).expect("stale metadata");
    assert_eq!(stale.pid, 1);
    assert_eq!(stale.acquired_at, "2020-01-01T00:00:00+00:00");
}

#[cfg(unix)]
#[tokio::test]
async fn release_is_deterministic_on_drop() {
    let (_repo_dir, repo) = git_repo();
    let lock = lock_for(&repo, WorktreeLockConfig::default());
    if !lock.is_enabled() {
        return;
    }
    {
        let guard = lock.acquire().await.unwrap();
        assert!(guard.is_active());
        // A second handle is refused while the guard lives …
        let contender = lock_for(&repo, WorktreeLockConfig::default());
        assert!(contender.try_acquire().await.is_err());
    }
    // … and accepted as soon as it is dropped.
    let guard = lock.try_acquire().await.unwrap();
    assert!(guard.is_active());
}

#[cfg(not(unix))]
#[tokio::test]
async fn platforms_without_flock_report_an_inactive_guard() {
    let (_repo_dir, repo) = git_repo();
    let lock = lock_for(&repo, WorktreeLockConfig::default());
    assert!(!lock.is_enabled());
    let guard = lock.acquire().await.unwrap();
    assert!(!guard.is_active());
    assert!(guard.metadata().is_none());
    assert!(!lock.lock_path().exists());
}
