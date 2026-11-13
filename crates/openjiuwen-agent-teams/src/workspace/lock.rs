//! Filesystem lock guarding the worktree critical section of one repository.
//!
//! Python parity source: `openjiuwen/harness/tools/worktree/lock.py` (which uses `filelock` on
//! `<repo_root>/.git/openjiuwen-worktree.lock`). This port cannot take an OS advisory lock without
//! an extra crate, so it uses `create_new` file creation as the exclusive primitive:
//!
//! 1. Creating `openjiuwen-worktree.lock` with `create_new(true)` succeeds for exactly one caller —
//!    in this process or any other — because the filesystem arbitrates the create atomically;
//! 2. a loser polls every [`LockOptions::poll`] until it either wins or its [`LockOptions::wait`]
//!    budget expires;
//! 3. the holder writes its pid and a start stamp into the file and deletes it on drop.
//!
//! A crash leaves the file behind, so a file whose modification time is older than
//! [`LockOptions::stale_after`] is treated as stale: it is removed (with a `warn!`) and the caller
//! retries. Timeout raises [`Error::Timeout`] rather than degrading to lock-free, which would
//! reintroduce the race the lock prevents.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use tokio::io::AsyncWriteExt;
use tracing::{debug, warn};

use openjiuwen_common::error::{Error, Result};

/// File name of the repository-level worktree lock.
pub const LOCK_FILE_NAME: &str = "openjiuwen-worktree.lock";

/// Default time a caller waits for the lock before failing.
pub const DEFAULT_LOCK_WAIT: Duration = Duration::from_secs(10);

/// Default age after which an abandoned lock file is reclaimed.
pub const DEFAULT_STALE_AFTER: Duration = Duration::from_secs(60);

/// Default interval between two acquisition attempts.
pub const DEFAULT_POLL_INTERVAL: Duration = Duration::from_millis(25);

/// Bounds of one lock acquisition attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LockOptions {
    /// Total time to wait for the lock (`Error::Timeout` afterwards).
    pub wait: Duration,
    /// Age at which an existing lock file is considered abandoned and removed.
    pub stale_after: Duration,
    /// Interval between two attempts.
    pub poll: Duration,
}

impl Default for LockOptions {
    fn default() -> Self {
        Self {
            wait: DEFAULT_LOCK_WAIT,
            stale_after: DEFAULT_STALE_AFTER,
            poll: DEFAULT_POLL_INTERVAL,
        }
    }
}

impl LockOptions {
    /// Creates options that wait at most `wait` and keep the default stale/poll bounds.
    #[must_use]
    pub fn new(wait: Duration) -> Self {
        Self {
            wait,
            ..Self::default()
        }
    }

    /// Sets the age at which an abandoned lock file is reclaimed.
    #[must_use]
    pub fn with_stale_after(mut self, stale_after: Duration) -> Self {
        self.stale_after = stale_after;
        self
    }

    /// Sets the interval between two acquisition attempts.
    #[must_use]
    pub fn with_poll(mut self, poll: Duration) -> Self {
        self.poll = poll;
        self
    }
}

/// The acquired repository lock; removing the file releases it.
///
/// The guard removes its file on drop, so an early return (including an error path) never leaks the
/// lock. [`Self::release`] awaits the removal and reports failures.
#[derive(Debug)]
pub struct WorktreeLockGuard {
    /// Lock file held by this guard.
    path: PathBuf,
    /// Set once the file has been removed, so `Drop` does not remove a file it no longer owns.
    released: bool,
}

impl WorktreeLockGuard {
    /// Returns the lock file held by this guard.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Removes the lock file and gives up the lock.
    ///
    /// # Errors
    ///
    /// Returns [`Error::SysOp`] when the file exists but cannot be removed. A file that is already
    /// gone is treated as a successful release.
    pub async fn release(mut self) -> Result<()> {
        self.released = true;
        match tokio::fs::remove_file(&self.path).await {
            Ok(()) => {
                debug!(lock = %self.path.display(), "released worktree lock");
                Ok(())
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(Error::sys_op(format!(
                "releasing worktree lock {}: {error}",
                self.path.display()
            ))),
        }
    }
}

impl Drop for WorktreeLockGuard {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        // Best effort: the lock file is a hint about a critical section, and a leftover file is
        // reclaimed by `stale_after`. A failing removal cannot be reported from `Drop`, so it is
        // logged instead of panicking.
        if let Err(error) = std::fs::remove_file(&self.path) {
            if error.kind() != std::io::ErrorKind::NotFound {
                warn!(
                    lock = %self.path.display(),
                    %error,
                    "failed to release worktree lock; it will be reclaimed once stale"
                );
            }
        }
    }
}

/// Acquires the exclusive lock stored in `path`.
///
/// Creates the parent directory when missing, then polls [`LockOptions::poll`] until the file can
/// be created with `create_new`, reclaiming a file older than [`LockOptions::stale_after`].
///
/// # Errors
///
/// Returns [`Error::Timeout`] when the lock is not acquired within [`LockOptions::wait`], and
/// [`Error::SysOp`] when the parent directory or the lock file cannot be written.
pub async fn acquire(path: &Path, options: &LockOptions) -> Result<WorktreeLockGuard> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await.map_err(|error| {
            Error::sys_op(format!(
                "creating worktree lock directory {}: {error}",
                parent.display()
            ))
        })?;
    }
    let deadline = tokio::time::Instant::now() + options.wait;
    loop {
        match create_lock_file(path).await {
            Ok(()) => {
                debug!(lock = %path.display(), "acquired worktree lock");
                return Ok(WorktreeLockGuard {
                    path: path.to_path_buf(),
                    released: false,
                });
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                if reclaim_if_stale(path, options.stale_after).await? {
                    continue;
                }
                if tokio::time::Instant::now() >= deadline {
                    return Err(Error::timeout(format!(
                        "timed out after {}ms waiting for the worktree lock {}",
                        options.wait.as_millis(),
                        path.display()
                    )));
                }
                tokio::time::sleep(options.poll).await;
            }
            Err(error) => {
                return Err(Error::sys_op(format!(
                    "creating worktree lock {}: {error}",
                    path.display()
                )));
            }
        }
    }
}

/// Creates the lock file exclusively and stamps it with this process id.
async fn create_lock_file(path: &Path) -> std::io::Result<()> {
    let mut file = tokio::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .await?;
    file.write_all(format!("pid={}\n", std::process::id()).as_bytes())
        .await?;
    file.flush().await?;
    Ok(())
}

/// Removes `path` when it is older than `stale_after`; returns whether it was reclaimed.
///
/// A lock file with an unreadable modification time is left alone (fail closed) — the caller then
/// times out instead of stealing a lock it cannot reason about.
async fn reclaim_if_stale(path: &Path, stale_after: Duration) -> Result<bool> {
    let metadata = match tokio::fs::metadata(path).await {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(true),
        Err(error) => {
            return Err(Error::sys_op(format!(
                "inspecting worktree lock {}: {error}",
                path.display()
            )));
        }
    };
    let age = metadata
        .modified()
        .ok()
        .and_then(|modified| SystemTime::now().duration_since(modified).ok());
    let Some(age) = age else {
        return Ok(false);
    };
    if age < stale_after {
        return Ok(false);
    }
    warn!(
        lock = %path.display(),
        age_secs = age.as_secs(),
        "reclaiming a stale worktree lock file"
    );
    match tokio::fs::remove_file(path).await {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(true),
        Err(error) => Err(Error::sys_op(format!(
            "reclaiming stale worktree lock {}: {error}",
            path.display()
        ))),
    }
}
