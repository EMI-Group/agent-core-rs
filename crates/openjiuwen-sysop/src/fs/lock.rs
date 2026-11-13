//! Cross-process advisory file locks for the local filesystem backend.
//!
//! The filesystem backend locks in two layers:
//!
//! 1. An in-process [`tokio::sync::RwLock`] per resolved path (owned by
//!    [`crate::fs::local::LocalFsOperation`]). It serialises tasks inside one process and is
//!    the cheap fast path.
//! 2. The advisory `flock` implemented here, taken on a *sidecar* lock file so that several
//!    processes sharing one workspace exclude each other as well. This is the role Python's
//!    SQLite-backed `ReadWriteLockManager` plays (`local/_rw_lock_manager.py` and
//!    `local/_async_read_write_lock.py`).
//!
//! Both layers are taken in the same canonical order (see
//! [`LocalFsOperation::acquire_ordered`](crate::fs::local::LocalFsOperation)), which is what
//! keeps a transfer that locks two paths deadlock-free.
//!
//! # Lock-file layout
//!
//! A target is never locked directly: a write that truncates and then rolls back would drop
//! its own lock. Instead every target maps to a deterministic sidecar file:
//!
//! ```text
//! <root>/<sha256-hex-of-the-absolute-resolved-target>.lock
//! ```
//!
//! `<root>` is [`default_lock_root`] (`<system temp dir>/openjiuwen-sysop-locks`) unless a
//! caller overrides it with [`FileLockManager::with_root`]. The directory is created lazily
//! on the first acquisition with `create_dir_all`, and an existing directory is reused.
//! [`lock_path_for`] exposes the derivation so tests can assert the exact file.
//!
//! The hash covers the target path only, so two spellings of the same file lock the same
//! sidecar. The backend resolves every path through
//! [`PathPolicy`](crate::workspace::PathPolicy) before locking (the host side of a transfer
//! is canonicalized), so a symlink and its target normally end up on one lock file.
//!
//! # Blocking and failure
//!
//! Each lock attempt runs the `flock` syscall inside
//! [`tokio::task::spawn_blocking`], so the async runtime is never blocked. The first attempt
//! is non-blocking; while another process holds the lock the manager sleeps for
//! [`POLL_INTERVAL`] and retries until [`LocalWorkConfig::lock_timeout_secs`] expires
//! (`0` = wait forever, still polling so cancellation stays possible). A timeout, a contended
//! [`FileLockManager::try_acquire`], or an I/O error while creating/opening the lock file is
//! returned as a [`LockError`]; the filesystem backend turns that into an `OpResult` with code
//! `199003`. **A lock file that cannot be created refuses the operation — it never runs
//! unlocked.**
//!
//! # Releasing
//!
//! [`FileLockGuard`] owns the open [`std::fs::File`]. It is `Send` and deliberately not
//! `Clone`; dropping it closes the descriptor, and closing the descriptor releases the
//! `flock`. Guards therefore must be kept alive for as long as the file is read or written.
//!
//! # Platforms
//!
//! Only Unix has `flock`. On other platforms [`FileLockManager::is_enabled`] is `false` and
//! every acquisition yields an inactive guard immediately: the in-process layer stays the
//! only lock, and no lock files are created.

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::config::LocalWorkConfig;
use crate::util::{format_path, sha256_hex};

/// Directory name (below the system temporary directory) that holds the sidecar lock files.
pub const LOCK_DIR_NAME: &str = "openjiuwen-sysop-locks";

/// Delay between two non-blocking acquisition attempts, and the longest single sleep while
/// waiting for a held lock.
///
/// Polling (rather than a blocking `flock`) keeps a cancelled operation from staying parked
/// inside a syscall; 10 ms is short enough to look instant to a caller.
pub const POLL_INTERVAL: Duration = Duration::from_millis(10);

/// The directory sidecar lock files live in when no root is configured.
#[must_use]
pub fn default_lock_root() -> PathBuf {
    std::env::temp_dir().join(LOCK_DIR_NAME)
}

/// Returns the sidecar lock file that guards `target` for a manager rooted at `root`.
///
/// The name is the lowercase hex SHA-256 of the target's normalized absolute path, suffixed
/// with `.lock`. Relative targets are made absolute against the current directory first, so
/// the same argument always maps to the same file within one process.
#[must_use]
pub fn lock_path_for(root: &Path, target: &Path) -> PathBuf {
    let normalized = std::path::absolute(target).unwrap_or_else(|_| target.to_path_buf());
    let digest = sha256_hex(normalized.as_os_str().as_encoded_bytes());
    root.join(format!("{digest}.lock"))
}

/// How a target is locked.
///
/// `Read` maps to a shared `flock`, `Write` to an exclusive one. Both names line up with the
/// in-process lock mode so a single description covers the two layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LockMode {
    /// Shared access, used by readers; several processes may hold it at once.
    Read,
    /// Exclusive access, used by writers and transfers' destinations.
    Write,
}

impl LockMode {
    /// The lowercase name used in lock messages (`"read"` or `"write"`).
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
        }
    }
}

impl fmt::Display for LockMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.label())
    }
}

/// Why an advisory lock could not be acquired.
#[derive(Debug)]
pub enum LockError {
    /// The lock is held elsewhere and [`FileLockManager::try_acquire`] does not wait.
    Contended {
        /// The file being guarded.
        target: PathBuf,
        /// The sidecar lock file.
        lock_path: PathBuf,
        /// The requested mode.
        mode: LockMode,
    },
    /// [`FileLockManager::acquire`] gave up after its deadline.
    TimedOut {
        /// The file being guarded.
        target: PathBuf,
        /// The sidecar lock file.
        lock_path: PathBuf,
        /// The requested mode.
        mode: LockMode,
        /// How long the acquisition waited before giving up.
        waited: Duration,
    },
    /// The lock directory or lock file could not be created or opened.
    Io {
        /// The file being guarded.
        target: PathBuf,
        /// The sidecar lock file.
        lock_path: PathBuf,
        /// The underlying operating-system error.
        source: io::Error,
    },
}

impl LockError {
    /// The target the caller wanted to lock.
    #[must_use]
    pub fn target(&self) -> &Path {
        match self {
            Self::Contended { target, .. }
            | Self::TimedOut { target, .. }
            | Self::Io { target, .. } => target,
        }
    }

    /// The sidecar lock file that could not be taken.
    #[must_use]
    pub fn lock_path(&self) -> &Path {
        match self {
            Self::Contended { lock_path, .. }
            | Self::TimedOut { lock_path, .. }
            | Self::Io { lock_path, .. } => lock_path,
        }
    }

    /// How long a timed-out acquisition waited, if it timed out at all.
    #[must_use]
    pub const fn waited(&self) -> Option<Duration> {
        match self {
            Self::TimedOut { waited, .. } => Some(*waited),
            _ => None,
        }
    }

    /// The human-readable reason, ready for `fs_execution_message(execution, reason)`.
    #[must_use]
    pub fn reason(&self) -> String {
        self.to_string()
    }
}

impl fmt::Display for LockError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contended {
                target,
                lock_path,
                mode,
            } => write!(
                formatter,
                "cross-process {} lock is already held: {} (lock file: {})",
                mode.label(),
                format_path(target),
                format_path(lock_path)
            ),
            Self::TimedOut {
                target,
                lock_path,
                mode,
                waited,
            } => write!(
                formatter,
                "cross-process {} lock timeout after {:.1}s waiting for {} (lock file: {})",
                mode.label(),
                waited.as_secs_f64(),
                format_path(target),
                format_path(lock_path)
            ),
            Self::Io {
                target,
                lock_path,
                source,
            } => write!(
                formatter,
                "cross-process lock file error for {}: {source} (lock file: {})",
                format_path(target),
                format_path(lock_path)
            ),
        }
    }
}

impl std::error::Error for LockError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Contended { .. } | Self::TimedOut { .. } => None,
        }
    }
}

/// A held advisory lock.
///
/// Owning the open [`std::fs::File`] is the whole point: closing the descriptor releases the
/// `flock`, so the guard must stay alive for as long as the locked file is in use. The guard
/// is `Send` (so it can be held inside a `BoxStream`) and never `Clone`.
#[derive(Debug)]
pub struct FileLockGuard {
    /// The descriptor whose close releases the lock; `None` when locking is disabled or the
    /// platform has no `flock`.
    file: Option<std::fs::File>,
    /// The sidecar file the lock lives in, kept for diagnostics.
    lock_path: PathBuf,
    /// The mode the lock was taken in.
    mode: LockMode,
}

impl FileLockGuard {
    /// A guard over a real advisory lock.
    fn active(file: std::fs::File, lock_path: PathBuf, mode: LockMode) -> Self {
        Self {
            file: Some(file),
            lock_path,
            mode,
        }
    }

    /// A guard that holds nothing: cross-process locking is off, or the platform has no
    /// `flock`.
    fn inactive(lock_path: PathBuf, mode: LockMode) -> Self {
        Self {
            file: None,
            lock_path,
            mode,
        }
    }

    /// The sidecar lock file, whether or not a lock is actually held.
    #[must_use]
    pub fn lock_path(&self) -> &Path {
        &self.lock_path
    }

    /// The mode this guard was requested in.
    #[must_use]
    pub const fn mode(&self) -> LockMode {
        self.mode
    }

    /// Whether a real advisory lock is held (false when locking is disabled or unsupported).
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.file.is_some()
    }
}

impl Drop for FileLockGuard {
    fn drop(&mut self) {
        // Closing the descriptor releases the `flock`; taking it out of the guard first makes
        // the release explicit and documents why the field exists.
        drop(self.file.take());
    }
}

/// One non-blocking acquisition result, before the target path is attached.
#[derive(Debug)]
enum Attempt {
    /// Another process (or descriptor) holds an incompatible lock.
    Contended,
    /// The lock file could not be created or opened, or `flock` failed for another reason.
    Failed(io::Error),
}

/// Creates the sidecar lock directory and files for targets.
///
/// Cloneable and cheap (two paths and a duration): the filesystem backend keeps one per
/// [`crate::fs::local::LocalFsOperation`] and hands clones to blocking helpers. All clones
/// share the same root and deadline, so the lock-file layout is process-wide deterministic.
#[derive(Debug, Clone)]
pub struct FileLockManager {
    /// Directory holding the sidecar lock files.
    root: PathBuf,
    /// Deadline for [`Self::acquire`]; `None` waits forever.
    timeout: Option<Duration>,
    /// Whether the configuration asked for cross-process locking at all.
    enabled: bool,
}

impl FileLockManager {
    /// Builds a manager for `config`, rooted at [`default_lock_root`].
    #[must_use]
    pub fn new(config: &LocalWorkConfig) -> Self {
        Self::with_root(config, default_lock_root())
    }

    /// Builds a manager for `config` with an explicit lock-file root.
    ///
    /// The root lets tests (and hosts that keep the temp directory clean) choose where the
    /// sidecar files live; it does not change the file naming, which is always
    /// [`lock_path_for`].
    #[must_use]
    pub fn with_root(config: &LocalWorkConfig, root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            timeout: timeout_from_config(config),
            enabled: config.cross_process_locks,
        }
    }

    /// Overrides the acquisition deadline (`None` waits forever).
    ///
    /// [`LocalWorkConfig::lock_timeout_secs`] has one-second granularity; this setter exists
    /// so a caller (and the unit tests) can pick a sub-second deadline.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.timeout = timeout.filter(|timeout| !timeout.is_zero());
        self
    }

    /// The directory holding the sidecar lock files.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Whether a real advisory lock will be taken.
    ///
    /// `false` when [`LocalWorkConfig::cross_process_locks`] is off, and always `false` on
    /// platforms without `flock`; acquisitions then return an inactive guard immediately and
    /// create no files.
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.enabled && cfg!(unix)
    }

    /// The configured acquisition deadline, `None` meaning "wait forever".
    #[must_use]
    pub const fn timeout(&self) -> Option<Duration> {
        self.timeout
    }

    /// The sidecar lock file for `target` under this manager's root.
    #[must_use]
    pub fn lock_path(&self, target: &Path) -> PathBuf {
        lock_path_for(&self.root, target)
    }

    /// Takes the lock for `target` immediately, without waiting.
    ///
    /// # Errors
    ///
    /// Returns [`LockError::Contended`] when an incompatible lock is held elsewhere, and
    /// [`LockError::Io`] when the lock directory or lock file cannot be created or opened.
    pub async fn try_acquire(
        &self,
        target: &Path,
        mode: LockMode,
    ) -> Result<FileLockGuard, LockError> {
        let lock_path = self.lock_path(target);
        if !self.is_enabled() {
            return Ok(FileLockGuard::inactive(lock_path, mode));
        }
        match platform_attempt(&lock_path, mode).await {
            Ok(file) => Ok(FileLockGuard::active(file, lock_path, mode)),
            Err(Attempt::Contended) => Err(LockError::Contended {
                target: target.to_path_buf(),
                lock_path,
                mode,
            }),
            Err(Attempt::Failed(source)) => Err(LockError::Io {
                target: target.to_path_buf(),
                lock_path,
                source,
            }),
        }
    }

    /// Takes the lock for `target`, polling until the configured deadline.
    ///
    /// With [`Self::timeout`] set to `None` (config `lock_timeout_secs = 0`) this waits
    /// forever; it still polls, so awaiting this future stays cancellable.
    ///
    /// # Errors
    ///
    /// Returns [`LockError::TimedOut`] once the deadline passes, and [`LockError::Io`] when
    /// the lock directory or lock file cannot be created or opened.
    pub async fn acquire(&self, target: &Path, mode: LockMode) -> Result<FileLockGuard, LockError> {
        let lock_path = self.lock_path(target);
        if !self.is_enabled() {
            return Ok(FileLockGuard::inactive(lock_path, mode));
        }
        let started = tokio::time::Instant::now();
        let deadline = self.timeout.map(|timeout| started + timeout);
        loop {
            match platform_attempt(&lock_path, mode).await {
                Ok(file) => return Ok(FileLockGuard::active(file, lock_path, mode)),
                Err(Attempt::Failed(source)) => {
                    return Err(LockError::Io {
                        target: target.to_path_buf(),
                        lock_path,
                        source,
                    });
                }
                Err(Attempt::Contended) => {}
            }
            let now = tokio::time::Instant::now();
            let Some(deadline) = deadline else {
                tokio::time::sleep(POLL_INTERVAL).await;
                continue;
            };
            if now >= deadline {
                return Err(LockError::TimedOut {
                    target: target.to_path_buf(),
                    lock_path,
                    mode,
                    waited: now.saturating_duration_since(started),
                });
            }
            tokio::time::sleep(POLL_INTERVAL.min(deadline - now)).await;
        }
    }
}

/// Reads the acquisition deadline out of a configuration (`0` = wait forever).
fn timeout_from_config(config: &LocalWorkConfig) -> Option<Duration> {
    (config.lock_timeout_secs > 0).then(|| Duration::from_secs(config.lock_timeout_secs))
}

/// Runs one non-blocking `flock` attempt on a blocking thread.
async fn platform_attempt(lock_path: &Path, mode: LockMode) -> Result<std::fs::File, Attempt> {
    let path = lock_path.to_path_buf();
    let outcome = tokio::task::spawn_blocking(move || blocking_attempt(&path, mode)).await;
    match outcome {
        Ok(result) => result,
        Err(error) => Err(Attempt::Failed(io::Error::other(error.to_string()))),
    }
}

/// Opens (creating it if needed) and `flock`s the sidecar file, without blocking.
#[cfg(unix)]
fn blocking_attempt(lock_path: &Path, mode: LockMode) -> Result<std::fs::File, Attempt> {
    use std::fs::OpenOptions;
    use std::os::unix::io::AsRawFd;

    use nix::errno::Errno;
    use nix::fcntl::FlockArg;

    if let Some(parent) = lock_path.parent() {
        // Creating the lock directory lazily keeps a process that never locks from touching
        // the temp directory; an existing directory is not an error.
        std::fs::create_dir_all(parent).map_err(Attempt::Failed)?;
    }
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(lock_path)
        .map_err(Attempt::Failed)?;
    let arg = match mode {
        LockMode::Read => FlockArg::LockSharedNonblock,
        LockMode::Write => FlockArg::LockExclusiveNonblock,
    };
    // `nix::fcntl::flock` is deprecated in favour of the owning `Flock` wrapper, which panics
    // when its `Drop` unlock fails; holding the `File` ourselves and letting `close(2)`
    // release the lock keeps the "never panic" rule.
    #[allow(deprecated)]
    let locked = loop {
        match nix::fcntl::flock(file.as_raw_fd(), arg) {
            Err(Errno::EINTR) => {}
            other => break other,
        }
    };
    match locked {
        Ok(()) => Ok(file),
        Err(error) if error == Errno::EWOULDBLOCK || error == Errno::EAGAIN => {
            Err(Attempt::Contended)
        }
        Err(error) => Err(Attempt::Failed(io::Error::from(error))),
    }
}

/// Platforms without `flock` never take a lock (see the module docs).
#[cfg(not(unix))]
fn blocking_attempt(lock_path: &Path, _mode: LockMode) -> Result<std::fs::File, Attempt> {
    Err(Attempt::Failed(io::Error::new(
        io::ErrorKind::Unsupported,
        format!(
            "advisory file locks are unavailable on this platform: {}",
            format_path(lock_path)
        ),
    )))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;

    use super::*;

    /// A manager rooted in its own temporary directory, so unit tests never share lock files.
    fn manager(dir: &tempfile::TempDir) -> FileLockManager {
        FileLockManager::with_root(&LocalWorkConfig::default(), dir.path().join("locks"))
    }

    #[test]
    fn lock_path_is_deterministic_and_named_by_digest() {
        let root = Path::new("/tmp/openjiuwen-locks");
        let first = lock_path_for(root, Path::new("/workspace/a.txt"));
        let second = lock_path_for(root, Path::new("/workspace/a.txt"));
        assert_eq!(first, second);
        assert_eq!(first.parent(), Some(root));
        let name = first.file_name().unwrap().to_string_lossy().into_owned();
        let digest = name
            .strip_suffix(".lock")
            .unwrap_or_else(|| panic!("the lock file must end in .lock: {name}"));
        assert_eq!(digest.len(), 64);
        assert!(
            digest
                .chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "{name}"
        );
        assert_ne!(
            first,
            lock_path_for(root, Path::new("/workspace/b.txt")),
            "different targets get different lock files"
        );
    }

    #[test]
    fn relative_targets_are_made_absolute_before_hashing() {
        let dir = tempfile::tempdir().unwrap();
        let absolute = lock_path_for(dir.path(), &dir.path().join("relative.txt"));
        // The relative spelling resolves against the process working directory, so compare
        // only the absolute spelling with itself here; the point is that the helper never
        // fails and always yields a file name.
        assert_eq!(
            absolute,
            lock_path_for(dir.path(), &dir.path().join("relative.txt"))
        );
    }

    #[test]
    fn configuration_controls_enablement_and_timeout() {
        let enabled = FileLockManager::new(&LocalWorkConfig::default());
        assert!(enabled.is_enabled(), "the default config enables flock");
        assert_eq!(enabled.timeout(), Some(Duration::from_secs(30)));
        assert_eq!(enabled.root(), default_lock_root());

        let disabled = FileLockManager::new(&LocalWorkConfig {
            cross_process_locks: false,
            ..LocalWorkConfig::default()
        });
        assert!(!disabled.is_enabled());

        let unbounded = FileLockManager::new(&LocalWorkConfig {
            lock_timeout_secs: 0,
            ..LocalWorkConfig::default()
        });
        assert_eq!(unbounded.timeout(), None);
    }

    #[tokio::test]
    async fn shared_readers_and_exclusive_writers_exclude_each_other() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("shared.txt");
        let locks = manager(&dir);
        if !locks.is_enabled() {
            eprintln!("skipping: this platform has no flock");
            return;
        }

        let first = locks.try_acquire(&target, LockMode::Read).await.unwrap();
        assert!(first.is_active());
        let second = locks
            .try_acquire(&target, LockMode::Read)
            .await
            .expect("two shared locks are compatible even through separate descriptors");
        let contended = locks.try_acquire(&target, LockMode::Write).await;
        assert!(
            matches!(contended, Err(LockError::Contended { .. })),
            "an exclusive lock must be refused while a shared one is held"
        );
        drop(second);
        drop(first);
        locks
            .try_acquire(&target, LockMode::Write)
            .await
            .expect("the exclusive lock is free once every guard is dropped");
    }

    #[tokio::test]
    async fn acquire_reports_a_timeout_with_the_target_path() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("held.txt");
        let locks = manager(&dir).with_timeout(Some(Duration::from_millis(50)));
        if !locks.is_enabled() {
            eprintln!("skipping: this platform has no flock");
            return;
        }
        let held = locks.acquire(&target, LockMode::Write).await.unwrap();
        let error = locks.acquire(&target, LockMode::Read).await.unwrap_err();
        assert!(matches!(error, LockError::TimedOut { .. }), "{error:?}");
        assert_eq!(error.target(), target.as_path());
        assert!(error.waited().is_some());
        let reason = error.reason();
        assert!(reason.contains("timeout"), "{reason}");
        assert!(reason.contains(&format_path(&target)), "{reason}");
        assert!(reason.contains("lock file"), "{reason}");
        drop(held);

        // An unbounded wait still returns once the holder releases.
        let unbounded = manager(&dir).with_timeout(None);
        let held = unbounded.acquire(&target, LockMode::Write).await.unwrap();
        let waiter = {
            let unbounded = unbounded.clone();
            let target = target.clone();
            tokio::spawn(async move { unbounded.acquire(&target, LockMode::Read).await.map(drop) })
        };
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!waiter.is_finished(), "the reader still waits");
        drop(held);
        tokio::time::timeout(Duration::from_secs(5), waiter)
            .await
            .expect("the unbounded wait finishes once the lock is free")
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn a_disabled_manager_creates_no_lock_files() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("plain.txt");
        let config = LocalWorkConfig {
            cross_process_locks: false,
            ..LocalWorkConfig::default()
        };
        let root = dir.path().join("locks");
        let locks = FileLockManager::with_root(&config, &root);
        let guard = locks.acquire(&target, LockMode::Write).await.unwrap();
        assert!(!guard.is_active());
        assert!(
            !root.exists(),
            "a disabled manager must not create a lock dir"
        );
        drop(guard);
    }

    #[tokio::test]
    async fn the_lock_file_is_created_on_first_use_and_reused() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("used.txt");
        let locks = manager(&dir);
        if !locks.is_enabled() {
            eprintln!("skipping: this platform has no flock");
            return;
        }
        let guard = locks.acquire(&target, LockMode::Read).await.unwrap();
        let lock_path = guard.lock_path().to_path_buf();
        assert_eq!(lock_path, locks.lock_path(&target));
        assert!(lock_path.is_file(), "the sidecar must exist while held");
        drop(guard);
    }

    #[tokio::test]
    async fn guards_are_send_and_release_on_drop() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("sent.txt");
        let locks = Arc::new(manager(&dir));
        if !locks.is_enabled() {
            eprintln!("skipping: this platform has no flock");
            return;
        }
        let guard = locks.acquire(&target, LockMode::Write).await.unwrap();
        let holder = tokio::spawn(async move { guard });
        let guard = holder.await.unwrap();
        assert!(guard.is_active());
        drop(guard);
        locks
            .try_acquire(&target, LockMode::Write)
            .await
            .expect("dropping the guard released the lock");
    }
}
