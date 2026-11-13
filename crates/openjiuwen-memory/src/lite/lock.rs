//! Cross-process advisory locking for the lite memory files.
//!
//! The lite memory keeps a Markdown source of truth and a derived index on the local
//! filesystem, so two processes can race on the same files.
//! [`FileLock`](crate::lite::lock::FileLock) is the
//! cooperative mutex over that pair: a writer takes the lock, rewrites the files, and
//! drops the lock.
//!
//! # Semantics
//!
//! - The Unix implementation is `flock(2)` on the lock file's open file description, so
//!   the lock belongs to the description and not to the process: two
//!   [`FileLock`](crate::lite::lock::FileLock)s for
//!   the same path inside one process conflict exactly like two processes do (which is
//!   what makes single-process tests meaningful). Closing or dropping the handle
//!   releases the lock.
//! - The lock is advisory: only processes that call
//!   [`FileLock`](crate::lite::lock::FileLock) take part. Nothing
//!   stops an unrelated program from writing the lock file or the files it guards.
//! - It is released when the [`FileLock`](crate::lite::lock::FileLock) drops (including when an owning task unwinds)
//!   and, at the latest, when the process exits; the kernel drops `flock` locks with the
//!   file, so there is no stale lock to recover from.
//! - [`FileLock::try_acquire`](crate::lite::lock::FileLock::try_acquire) never waits: when another holder owns the lock it returns
//!   `Ok(None)` rather than an error, so a caller can back off, do other work, or report
//!   contention its own way.
//!   [`FileLock::acquire`](crate::lite::lock::FileLock::acquire) waits for as long as the holder keeps
//!   the lock.
//! - The blocking `flock` call always runs on tokio's blocking thread pool, including
//!   the file open, so waiting for a lock never stalls the async reactor.
//! - The lock file's parent directory and the lock file itself are created when missing;
//!   the lock file's content is never read or written.
//!
//! # Platform support
//!
//! Only Unix is implemented. On other platforms
//! [`FileLock`](crate::lite::lock::FileLock) still exists — so callers
//! compile — but [`FileLock::acquire`](crate::lite::lock::FileLock::acquire) and
//! [`FileLock::try_acquire`](crate::lite::lock::FileLock::try_acquire) return
//! [`Error::Unsupported`] naming the
//! platform instead of degrading silently.
use std::path::{Path, PathBuf};

use openjiuwen_common::error::{Error, Result};

#[cfg(unix)]
use nix::errno::Errno;
#[cfg(unix)]
use nix::fcntl::{Flock, FlockArg};
#[cfg(unix)]
use std::fs::File;

/// An exclusive advisory lock held on one lock file, released on drop.
///
/// Created by [`FileLock::acquire`] (waiting) or [`FileLock::try_acquire`]
/// (non-blocking); see the [module docs](self) for the platform and `flock(2)`
/// semantics.
#[derive(Debug)]
pub struct FileLock {
    /// The lock-file path, as requested and as reported by [`FileLock::path`].
    path: PathBuf,
    /// The open, exclusively-locked handle; dropping it releases the lock.
    /// It is held for its `Drop` impl and never inspected, hence the allow.
    #[cfg(unix)]
    #[allow(dead_code)]
    locked: Flock<File>,
}

#[cfg(unix)]
impl FileLock {
    /// Blocks until the exclusive lock on `path` is acquired.
    ///
    /// The lock file is created when missing, as is its parent directory. The blocking
    /// `flock` call runs on tokio's blocking thread pool, so this future never blocks
    /// the async reactor.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the
    /// parent directory cannot be created, the lock file cannot be opened, or the
    /// `flock` call fails for a reason other than contention.
    pub async fn acquire(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let target = path.clone();
        let locked = tokio::task::spawn_blocking(move || acquire_blocking(&target))
            .await
            .map_err(Error::external)??;
        Ok(Self { path, locked })
    }

    /// Takes the exclusive lock on `path` only when it is free right now.
    ///
    /// `Ok(None)` means another holder owns the lock (`EWOULDBLOCK`/`EAGAIN` from the
    /// non-blocking `flock`); it is not an error, so callers decide whether to wait,
    /// retry later, or give up. The lock file is created when missing, as is its parent
    /// directory. The `flock` call runs on tokio's blocking thread pool.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the
    /// parent directory cannot be created, the lock file cannot be opened, or the
    /// `flock` call fails for a reason other than contention.
    pub async fn try_acquire(path: impl AsRef<Path>) -> Result<Option<Self>> {
        let path = path.as_ref().to_path_buf();
        let target = path.clone();
        let locked = tokio::task::spawn_blocking(move || try_acquire_blocking(&target))
            .await
            .map_err(Error::external)??;
        Ok(locked.map(|locked| Self { path, locked }))
    }

    /// The lock-file path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(not(unix))]
impl FileLock {
    /// Always fails: cross-process file locks are not implemented on this platform.
    ///
    /// # Errors
    ///
    /// Always returns
    /// [`Error::Unsupported`] naming the
    /// current platform; nothing is created on disk.
    pub async fn acquire(path: impl AsRef<Path>) -> Result<Self> {
        Err(unsupported(path.as_ref()))
    }

    /// Always fails: cross-process file locks are not implemented on this platform.
    ///
    /// # Errors
    ///
    /// Always returns
    /// [`Error::Unsupported`] naming the
    /// current platform; nothing is created on disk.
    pub async fn try_acquire(path: impl AsRef<Path>) -> Result<Option<Self>> {
        Err(unsupported(path.as_ref()))
    }

    /// The lock-file path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// The error returned when locking is not implemented on this platform.
#[cfg(not(unix))]
fn unsupported(path: &Path) -> Error {
    Error::unsupported(format!(
        "cross-process file locks are not implemented on this platform ({}) for lock file `{}`",
        std::env::consts::OS,
        path.display()
    ))
}

/// A failed `flock` call, keeping the OS error available as the source.
#[cfg(unix)]
#[derive(Debug)]
struct LockError {
    /// The lock file whose `flock` call failed.
    path: PathBuf,
    /// The OS error of the failed call, exposed as the error source.
    source: std::io::Error,
}

#[cfg(unix)]
impl std::fmt::Display for LockError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "failed to lock `{}`: {}",
            self.path.display(),
            self.source
        )
    }
}

#[cfg(unix)]
impl std::error::Error for LockError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

/// Wraps a failed `flock` call, naming the lock file.
#[cfg(unix)]
fn lock_error(path: &Path, errno: Errno) -> Error {
    Error::external(LockError {
        path: path.to_path_buf(),
        source: errno.into(),
    })
}

/// Whether `errno` means "another holder owns the lock".
///
/// `EWOULDBLOCK` and `EAGAIN` are the same value on most Unix systems, hence the
/// comparison instead of a match arm per name.
#[cfg(unix)]
fn is_held_elsewhere(errno: Errno) -> bool {
    errno == Errno::EWOULDBLOCK || errno == Errno::EAGAIN
}

/// Creates the parent directory (when it has one) and opens the lock file.
///
/// The file is opened read-write because `flock(2)` needs a valid descriptor, not
/// because its content matters; `create` makes a missing lock file and `truncate(false)`
/// keeps an existing lock file untouched.
#[cfg(unix)]
fn open_lock_file(path: &Path) -> Result<File> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent).map_err(Error::external)?;
    }
    std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(path)
        .map_err(Error::external)
}

/// Runs the blocking exclusive `flock`, carrying the returned lock out of the closure.
#[cfg(unix)]
fn acquire_blocking(path: &Path) -> Result<Flock<File>> {
    let file = open_lock_file(path)?;
    Flock::lock(file, FlockArg::LockExclusive).map_err(|(_, errno)| lock_error(path, errno))
}

/// Runs the blocking non-blocking `flock`, mapping contention to `Ok(None)`.
#[cfg(unix)]
fn try_acquire_blocking(path: &Path) -> Result<Option<Flock<File>>> {
    let file = open_lock_file(path)?;
    match Flock::lock(file, FlockArg::LockExclusiveNonblock) {
        Ok(locked) => Ok(Some(locked)),
        Err((_, errno)) if is_held_elsewhere(errno) => Ok(None),
        Err((_, errno)) => Err(lock_error(path, errno)),
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    /// The lock file path used by tests (one per temp directory).
    fn lock_path(dir: &std::path::Path) -> PathBuf {
        dir.join("memory.lock")
    }

    #[tokio::test]
    async fn acquire_creates_the_lock_file_and_a_second_holder_is_refused() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = lock_path(dir.path());
        let held = FileLock::acquire(&path).await.expect("lock");
        assert!(path.exists(), "acquire must create the lock file");
        let contender = FileLock::try_acquire(&path).await.expect("try acquire");
        assert!(
            contender.is_none(),
            "the holder is alive, so the lock must not be granted twice"
        );
        drop(held);
    }

    #[tokio::test]
    async fn a_dropped_lock_can_be_taken_again() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = lock_path(dir.path());
        let held = FileLock::acquire(&path).await.expect("lock");
        drop(held);
        let retaken = FileLock::try_acquire(&path).await.expect("try acquire");
        assert!(
            retaken.is_some(),
            "dropping the holder must release the lock"
        );
    }

    #[tokio::test]
    async fn a_second_task_cannot_take_the_held_lock() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = lock_path(dir.path());
        let held = FileLock::acquire(&path).await.expect("lock");
        let contender_path = path.clone();
        let attempt = tokio::spawn(async move { FileLock::try_acquire(&contender_path).await })
            .await
            .expect("task did not panic")
            .expect("try acquire");
        assert!(
            attempt.is_none(),
            "the holder stays alive across the other task's await"
        );
        drop(held);
    }

    #[tokio::test]
    async fn distinct_paths_do_not_interfere() {
        let dir = tempfile::tempdir().expect("temp dir");
        let first = FileLock::try_acquire(lock_path(dir.path()))
            .await
            .expect("try acquire")
            .expect("first lock is free");
        let second = FileLock::try_acquire(dir.path().join("other.lock"))
            .await
            .expect("try acquire")
            .expect("second lock is free");
        assert_ne!(first.path(), second.path());
    }

    #[tokio::test]
    async fn path_reports_the_requested_path_and_creates_missing_parents() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("nested").join("memory.lock");
        let held = FileLock::acquire(&path).await.expect("lock");
        assert_eq!(held.path(), path.as_path());
        assert!(path.exists(), "acquire must create the missing parent");
        drop(held);
    }
}
