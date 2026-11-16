//! Advisory file lock guarding one tokenizer-artefact download.
//!
//! No locking crate is available in the workspace, so the lock is an advisory
//! `create_new` marker file: creating it is atomic on every supported
//! filesystem, and a waiter polls with bounded exponential backoff via
//! [`tokio::time::sleep`] instead of busy-spinning. A marker whose mtime is
//! older than [`STALE_LOCK_AFTER`] is treated as abandoned (a crashed
//! downloader) and removed; a marker dated in the future is clock skew and stays
//! live, logged with both mtimes.
//!
//! The guard removes the marker on drop, so success, failure and cancellation
//! all release the lock.
use std::time::Duration;

use openjiuwen_common::error::{Error, Result};
use tokio::fs::OpenOptions;
use tokio::time::Instant;

/// How long a lock marker may stay untouched before waiters treat it as stale.
pub const STALE_LOCK_AFTER: Duration = Duration::from_secs(120);

/// How long a download waits for the lock before giving up.
pub const LOCK_WAIT_LIMIT: Duration = Duration::from_secs(10);

/// The first backoff between two attempts.
const INITIAL_BACKOFF: Duration = Duration::from_millis(10);

/// The upper bound on the backoff.
const MAX_BACKOFF: Duration = Duration::from_millis(250);

/// An acquired advisory lock; dropping it removes the marker file.
#[derive(Debug)]
pub struct ArtifactLock {
    /// The marker file holding the lock.
    path: std::path::PathBuf,
}

impl ArtifactLock {
    /// Acquires the lock at `path`, waiting at most [`LOCK_WAIT_LIMIT`].
    ///
    /// # Errors
    ///
    /// - [`Error::Timeout`] when the lock is still held when the wait budget
    ///   expires;
    /// - [`Error::External`] when the marker cannot be created (for example a
    ///   permission error), with the OS error preserved as the source.
    pub async fn acquire(path: std::path::PathBuf) -> Result<Self> {
        Self::acquire_within(path, LOCK_WAIT_LIMIT).await
    }

    /// Acquires the lock at `path` with an explicit wait budget.
    ///
    /// # Errors
    ///
    /// Same as [`ArtifactLock::acquire`], with the caller's wait budget in
    /// place of [`LOCK_WAIT_LIMIT`].
    pub async fn acquire_within(path: std::path::PathBuf, wait_limit: Duration) -> Result<Self> {
        let deadline = Instant::now() + wait_limit;
        let mut backoff = INITIAL_BACKOFF;
        loop {
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .await
            {
                Ok(_marker) => return Ok(Self { path }),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    if is_stale(&path).await {
                        tracing::debug!(
                            lock = %path.display(),
                            "removing a stale tokenizer download lock"
                        );
                        if let Err(error) = tokio::fs::remove_file(&path).await {
                            tracing::warn!(
                                lock = %path.display(),
                                %error,
                                "the stale tokenizer download lock could not be removed"
                            );
                        }
                        continue;
                    }
                    if Instant::now() >= deadline {
                        return Err(Error::Timeout(format!(
                            "tokenizer download lock {} was not released within {wait_limit:?}",
                            path.display()
                        )));
                    }
                    tokio::time::sleep(backoff).await;
                    backoff = (backoff * 2).min(MAX_BACKOFF);
                }
                Err(error) => return Err(Error::external(error)),
            }
        }
    }

    /// The marker path this guard holds.
    #[must_use]
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }
}

impl Drop for ArtifactLock {
    /// Removes the marker.
    ///
    /// The removal is one metadata syscall performed in `Drop` so that every
    /// early return and every cancellation releases the lock; a failure to
    /// remove only leaves a marker that later waiters expire as stale.
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_file(&self.path) {
            tracing::debug!(
                lock = %self.path.display(),
                %error,
                "the tokenizer download lock marker could not be removed; waiters expire it as stale"
            );
        }
    }
}

/// Whether the marker at `path` has not been touched for [`STALE_LOCK_AFTER`].
///
/// A marker whose mtime lies in the future is clock skew, not staleness: the two
/// processes disagree about "now", so the elapsed time cannot be trusted either
/// way. It is therefore treated as live — removing a marker a downloader really
/// holds would let two writers race for one artefact — and logged with both
/// mtimes so the skew is visible instead of silently extending the lock.
async fn is_stale(path: &std::path::Path) -> bool {
    let metadata = match tokio::fs::metadata(path).await {
        Ok(metadata) => metadata,
        Err(error) => {
            tracing::debug!(
                lock = %path.display(),
                %error,
                "the tokenizer download lock could not be inspected; it is treated as held"
            );
            return false;
        }
    };
    let modified = match metadata.modified() {
        Ok(modified) => modified,
        Err(error) => {
            tracing::debug!(
                lock = %path.display(),
                %error,
                "the tokenizer download lock has no mtime; it is treated as held"
            );
            return false;
        }
    };
    if let Ok(elapsed) = modified.elapsed() {
        return elapsed > STALE_LOCK_AFTER;
    }
    tracing::warn!(
        lock = %path.display(),
        mtime_unix_ms = unix_millis(modified),
        now_unix_ms = unix_millis(std::time::SystemTime::now()),
        "the tokenizer download lock marker has an mtime in the future (clock skew); it is treated as live"
    );
    false
}

/// A system time in Unix milliseconds, for logging.
///
/// A time before the epoch has no representable value here, so it logs as `0`;
/// only the skew diagnostic reads it.
fn unix_millis(time: std::time::SystemTime) -> u64 {
    time.duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| {
            u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lock_constants_are_ordered() {
        assert!(INITIAL_BACKOFF < MAX_BACKOFF);
        assert!(MAX_BACKOFF < LOCK_WAIT_LIMIT);
        assert!(LOCK_WAIT_LIMIT < STALE_LOCK_AFTER);
    }

    #[tokio::test]
    async fn acquire_creates_and_drop_removes_the_marker() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("one.lock");
        {
            let lock = ArtifactLock::acquire(path.clone()).await.expect("lock");
            assert_eq!(lock.path(), path);
            assert!(path.exists());
            assert!(
                ArtifactLock::acquire_within(path.clone(), Duration::from_millis(30))
                    .await
                    .is_err()
            );
        }
        assert!(!path.exists(), "dropping the guard removes the marker");
        let again = ArtifactLock::acquire(path.clone()).await.expect("lock");
        drop(again);
    }
}
