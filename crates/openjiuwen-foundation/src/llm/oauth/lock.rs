//! Cross-process advisory lock around the auth-store read-modify-write cycle.
//!
//! Two openJiuwen processes that both refresh an `OpenAI` account token would
//! otherwise race on the same document: each reads `auth.json`, merges its own
//! provider entry, and renames the result into place, so the last writer silently
//! drops the other's entry — and, worse, can persist a refresh token the issuer
//! has already rotated away.
//!
//! [`TokenFileLock`] serializes those cycles with an exclusive `flock(2)` on a
//! sidecar `<auth_file>.lock` file:
//!
//! - [`TokenFileLock::acquire`] waits for the holder, bounded by [`LOCK_TIMEOUT`];
//! - [`TokenFileLock::acquire_with_timeout`] waits for an explicit budget;
//! - [`TokenFileLock::try_acquire`] never waits and reports contention as the
//!   typed [`Error::External`] wrapping `WouldBlock`.
//!
//! Every read and write performed through the guard ([`TokenFileLock::load`],
//! [`TokenFileLock::save`], [`TokenFileLock::clear`]) happens while the lock is
//! held, so a whole read-modify-write cycle is atomic across processes.
//!
//! # Semantics
//!
//! - The lock belongs to the open file description, not to the process: two
//!   [`TokenFileLock`]s for the same auth file inside one process conflict exactly
//!   like two processes do, which is what makes single-process tests meaningful.
//!   Dropping (or closing) the handle releases the lock, and the kernel releases
//!   it when the process exits, so there is no stale lock to recover from.
//! - The lock file is created when missing (as is its parent directory) and is
//!   never read or written; only its inode matters. It is deliberately not removed
//!   after use, because removing it would let two processes lock two inodes.
//! - The lock is advisory: only writers that take it participate, exactly like
//!   Python's `filelock` around the auth store.
//!
//! # Platform support
//!
//! Only unix advisory locks are implemented. On other platforms the type still
//! exists so callers compile, but every acquisition returns [`Error::Unsupported`]
//! naming the missing support; the auth store's own writes keep working
//! atomically (temporary file plus rename) without cross-process serialization.

use std::path::{Path, PathBuf};
use std::time::Duration;

use openjiuwen_common::error::{Error, Result};

use super::store::OAuthToken;

#[cfg(unix)]
use nix::errno::Errno;
#[cfg(unix)]
use nix::fcntl::{Flock, FlockArg};
#[cfg(unix)]
use std::fs::File;

/// How long [`TokenFileLock::acquire`] waits for another holder.
pub const LOCK_TIMEOUT: Duration = Duration::from_secs(10);

/// Suffix appended to the auth-store name to form the sidecar lock file.
const LOCK_SUFFIX: &str = ".lock";

/// Delay between two attempts of a bounded acquisition.
#[cfg(unix)]
const RETRY_INTERVAL: Duration = Duration::from_millis(20);

/// Fallback lock-file name when the auth path has no file name.
const FALLBACK_NAME: &str = "auth.json";

/// Returns the sidecar lock-file path guarding `auth_file`.
///
/// `auth.json` becomes `auth.json.lock` in the same directory.
#[must_use]
pub fn lock_path(auth_file: &Path) -> PathBuf {
    let name = auth_file.file_name().map_or_else(
        || FALLBACK_NAME.to_owned(),
        |name| name.to_string_lossy().into_owned(),
    );
    auth_file.with_file_name(format!("{name}{LOCK_SUFFIX}"))
}

/// An exclusive advisory lock held on an auth store's lock file.
///
/// Created by [`TokenFileLock::acquire`], [`TokenFileLock::acquire_with_timeout`],
/// or [`TokenFileLock::try_acquire`]; see the [module docs](self) for the writer
/// protocol, the `flock(2)` semantics, and the platform support.
///
/// Drop the handle to release the lock. All token access goes through the guard,
/// so a caller can never accidentally read or write outside the critical section.
pub struct TokenFileLock {
    /// The auth-store path the lock guards (reported by [`Self::auth_path`]).
    auth_path: PathBuf,
    /// The `base_url` recorded next to the tokens (mirrors the owning store).
    base_url: String,
    /// The open, exclusively-locked lock-file handle; dropping it releases the
    /// lock, so it is held for its `Drop` impl and never inspected.
    #[cfg(unix)]
    #[allow(dead_code)]
    locked: Flock<File>,
}

impl TokenFileLock {
    /// Returns the auth-store path this lock guards.
    #[must_use]
    pub fn auth_path(&self) -> &Path {
        &self.auth_path
    }

    /// Returns the sidecar lock-file path.
    #[must_use]
    pub fn lock_path(&self) -> PathBuf {
        lock_path(&self.auth_path)
    }

    /// Returns the `base_url` recorded next to the stored tokens.
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Loads the stored token while the lock is held.
    ///
    /// # Errors
    ///
    /// Returns the auth-store I/O or parse error, and the error of the blocking
    /// worker when it cannot be joined.
    pub async fn load(&self) -> Result<Option<OAuthToken>> {
        let path = self.auth_path.clone();
        tokio::task::spawn_blocking(move || super::store::load_token(&path))
            .await
            .map_err(Error::external)?
    }

    /// Merges `token` into the auth store while the lock is held.
    ///
    /// # Errors
    ///
    /// Returns the auth-store I/O or parse error, and the error of the blocking
    /// worker when it cannot be joined.
    pub async fn save(&self, token: &OAuthToken) -> Result<()> {
        let path = self.auth_path.clone();
        let base_url = self.base_url.clone();
        let token = token.clone();
        tokio::task::spawn_blocking(move || super::store::save_token(&path, &base_url, &token))
            .await
            .map_err(Error::external)?
    }

    /// Removes the stored token while the lock is held, reporting whether one was
    /// present.
    ///
    /// # Errors
    ///
    /// Returns the auth-store I/O or parse error, and the error of the blocking
    /// worker when it cannot be joined.
    pub async fn clear(&self) -> Result<bool> {
        let path = self.auth_path.clone();
        tokio::task::spawn_blocking(move || super::store::clear_token(&path))
            .await
            .map_err(Error::external)?
    }
}

impl std::fmt::Debug for TokenFileLock {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The open handle is an implementation detail; naming the paths keeps the
        // output useful without exposing anything about the credentials.
        formatter
            .debug_struct("TokenFileLock")
            .field("auth_path", &self.auth_path)
            .field("lock_path", &self.lock_path())
            .finish_non_exhaustive()
    }
}

#[cfg(unix)]
impl TokenFileLock {
    /// Takes the exclusive lock on `auth_path`'s lock file without waiting.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] wrapping `WouldBlock` when another holder owns
    /// the lock (retry once it is released), and [`Error::External`] for any other
    /// open or `flock` failure.
    pub fn try_acquire(auth_path: &Path, base_url: &str) -> Result<Self> {
        let path = lock_path(auth_path);
        let file = open_lock_file(&path)?;
        match Flock::lock(file, FlockArg::LockExclusiveNonblock) {
            Ok(locked) => Ok(Self::new(auth_path, base_url, locked)),
            Err((_, errno)) if is_held_elsewhere(errno) => Err(lock_held(&path)),
            Err((_, errno)) => Err(lock_error(&path, errno)),
        }
    }

    /// Takes the exclusive lock on `auth_path`'s lock file, waiting at most
    /// [`LOCK_TIMEOUT`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Timeout`] when the holder keeps the lock for the whole
    /// budget, and [`Error::External`] for any other open or `flock` failure.
    pub fn acquire(auth_path: &Path, base_url: &str) -> Result<Self> {
        Self::acquire_with_timeout(auth_path, base_url, LOCK_TIMEOUT)
    }

    /// Takes the exclusive lock on `auth_path`'s lock file, waiting at most
    /// `timeout`.
    ///
    /// The wait is a non-blocking `flock` retried every 20 ms rather than a
    /// blocking call, so the budget is always honoured.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Timeout`] when the holder keeps the lock for the whole
    /// budget, and [`Error::External`] for any other open or `flock` failure.
    pub fn acquire_with_timeout(
        auth_path: &Path,
        base_url: &str,
        timeout: Duration,
    ) -> Result<Self> {
        let path = lock_path(auth_path);
        let mut file = open_lock_file(&path)?;
        let started = std::time::Instant::now();
        loop {
            match Flock::lock(file, FlockArg::LockExclusiveNonblock) {
                Ok(locked) => return Ok(Self::new(auth_path, base_url, locked)),
                Err((returned, errno)) if is_held_elsewhere(errno) => file = returned,
                Err((_, errno)) => return Err(lock_error(&path, errno)),
            }
            let elapsed = started.elapsed();
            if elapsed >= timeout {
                return Err(lock_timeout(&path, timeout));
            }
            std::thread::sleep(RETRY_INTERVAL.min(timeout.saturating_sub(elapsed)));
        }
    }

    /// Builds the guard around an already-locked handle.
    fn new(auth_path: &Path, base_url: &str, locked: Flock<File>) -> Self {
        Self {
            auth_path: auth_path.to_path_buf(),
            base_url: base_url.to_owned(),
            locked,
        }
    }
}

#[cfg(not(unix))]
impl TokenFileLock {
    /// Always fails: unix advisory file locks are the only implementation.
    ///
    /// # Errors
    ///
    /// Always returns [`Error::Unsupported`] naming the missing platform support.
    pub fn try_acquire(auth_path: &Path, _base_url: &str) -> Result<Self> {
        Err(unsupported(auth_path))
    }

    /// Always fails: unix advisory file locks are the only implementation.
    ///
    /// # Errors
    ///
    /// Always returns [`Error::Unsupported`] naming the missing platform support.
    pub fn acquire(auth_path: &Path, _base_url: &str) -> Result<Self> {
        Err(unsupported(auth_path))
    }

    /// Always fails: unix advisory file locks are the only implementation.
    ///
    /// # Errors
    ///
    /// Always returns [`Error::Unsupported`] naming the missing platform support.
    pub fn acquire_with_timeout(
        auth_path: &Path,
        _base_url: &str,
        _timeout: Duration,
    ) -> Result<Self> {
        Err(unsupported(auth_path))
    }
}

/// The error returned when unix advisory locking is unavailable.
#[cfg(not(unix))]
fn unsupported(auth_path: &Path) -> Error {
    Error::unsupported(format!(
        "cross-process locking of the auth store `{}` requires unix advisory file locks \
         (flock), which are not implemented on this platform ({})",
        lock_path(auth_path).display(),
        std::env::consts::OS
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
            "failed to lock the auth store lock file `{}`: {}",
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

/// Reports that another holder owns the lock (the `try_lock` outcome).
#[cfg(unix)]
fn lock_held(path: &Path) -> Error {
    Error::external(std::io::Error::new(
        std::io::ErrorKind::WouldBlock,
        format!(
            "the auth store lock file `{}` is held by another process; retry once it is released",
            path.display()
        ),
    ))
}

/// Reports that the bounded wait for the lock ran out.
#[cfg(unix)]
fn lock_timeout(path: &Path, timeout: Duration) -> Error {
    Error::timeout(format!(
        "timed out after {timeout:?} waiting for the auth store lock file `{}`, which another \
         process holds",
        path.display()
    ))
}

/// Whether `errno` means "another holder owns the lock".
///
/// `EWOULDBLOCK` and `EAGAIN` share a value on most unix systems, hence the
/// comparison instead of one match arm per name.
#[cfg(unix)]
fn is_held_elsewhere(errno: Errno) -> bool {
    errno == Errno::EWOULDBLOCK || errno == Errno::EAGAIN
}

/// Creates the lock file's parent directory (when it has one) and opens the file.
///
/// The file is opened read-write because `flock(2)` needs a valid descriptor, not
/// because its content matters; `create` makes a missing lock file and
/// `truncate(false)` keeps an existing one untouched.
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
        .map_err(|error| {
            Error::external(std::io::Error::new(
                error.kind(),
                format!(
                    "opening the auth store lock file `{}`: {error}",
                    path.display()
                ),
            ))
        })
}
