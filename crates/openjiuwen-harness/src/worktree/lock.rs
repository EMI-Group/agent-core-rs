//! Repo-bucketed cross-process advisory lock for worktree lifecycle sections.
//!
//! Port of `openjiuwen/harness/tools/worktree/lock.py`: Python serialises
//! worktree create/remove against one `repo_root` with an in-process
//! `asyncio.Lock` plus a `filelock.AsyncFileLock` on
//! `<repo_root>/.git/openjiuwen-worktree.lock`. This port keeps the observable
//! behaviour — one exclusive lock per repository root, a bounded wait, a typed
//! timeout — and delegates the operating-system part to the advisory `flock`
//! seam of [`openjiuwen_sysop::fs::lock`].
//!
//! # Layout
//!
//! The lock file is a sidecar derived from the repository root: the manager
//! hashes the absolute target path, so the file itself is named
//! `<root>/<sha256-of-repo-root>.lock`.
//!
//! - `<root>` is `<repo_root>/.git` when that is an existing directory, and
//!   `<workspace_root>/.worktree` otherwise — the directory
//!   [`crate::workspace::Workspace`] already uses for worktree state, which
//!   keeps a non-git directory lockable.
//! - [`WorktreeLock::lock_path`] exposes the derivation so a host (and the
//!   tests) can assert the exact file.
//!
//! # Policy
//!
//! - One lock per repository root: two worktrees of one repository share the
//!   `.git` state, so they share the lock; different roots never contend.
//! - The lock is exclusive (write), never read/write: every worktree critical
//!   section mutates shared git state. Read-only fast paths stay outside it by
//!   caller convention, exactly as in Python.
//! - The deadline is [`DEFAULT_LOCK_TIMEOUT`] (Python's `timeout=120.0`).
//!   Timing out raises instead of degrading to lock-free, which would
//!   reintroduce the race the lock prevents.
//! - A zero timeout means "do not wait at all": it is mapped to a 1 ms budget
//!   so an immediate attempt still happens (the seam reads `None` as "wait
//!   forever", which is the opposite of what a caller asking for zero wants).
//! - `enabled = false` opts out (Python's `repo_lock=False`), for a backend
//!   that provides its own isolation.
//!
//! # Stale locks
//!
//! `flock` is released by the kernel when the holding process exits, so a
//! crashed holder can never leave a *blocking* stale lock: the next acquisition
//! simply succeeds once the dead process is reaped.
//! The `pid`/`acquired_at` lines written into the lock file on acquisition are
//! therefore diagnostics, not arbitration.
//! They let a host tell a long wait (holder alive, metadata fresh) apart from a
//! crash that left only metadata behind ([`WorktreeLock::stale_metadata`]).
//! The lock file is never "broken" or unlinked while another process may hold
//! it: unlinking a `flock` sidecar would let two processes lock different
//! inodes of the same path and exclude nobody.
//!
//! # Holding the guard
//!
//! [`WorktreeLockGuard`] owns an OS advisory lock on a `Send` file descriptor,
//! not a blocking `std`/`parking_lot` mutex, so holding it across the manager's
//! git `.await` points is safe and intended; one acquisition per operation keeps
//! it to the critical section.
//! A `parking_lot`/`std` lock must never be held across `.await` — this guard is
//! the sanctioned alternative.
//! The descriptor is closed (and the lock released) on every `Drop` path,
//! including unwind, so a panicking critical section cannot leak the lock.
//!
//! # Platforms and failures
//!
//! Only Unix has `flock`. Where it is missing, or when the lock is disabled, no
//! cross-process exclusion is possible: `acquire` returns an *inactive* guard
//! immediately and writes no metadata. That is best-effort by construction and
//! callers that need isolation must check [`WorktreeLock::is_enabled`].
//! A lock file that cannot be created is an error rather than an unlocked
//! section.
//!
//! [`WorktreeLockError`] distinguishes contention (somebody else holds it right
//! now) from a timeout (we gave up waiting) and an I/O failure; the manager
//! boundary converts it into [`openjiuwen_common::error::Error`] with a stable
//! marker, and [`is_lock_timeout`]/[`is_lock_contention`] recognise those
//! markers on the converted error.
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use chrono::{DateTime, Utc};
use openjiuwen_common::error::Error;
use openjiuwen_sysop::config::LocalWorkConfig;
use openjiuwen_sysop::fs::lock::{FileLockGuard, FileLockManager, LockError, LockMode};

use super::store::INDEX_DIR;

/// Deadline for acquiring the repository lock (Python parity: `timeout=120.0`).
pub const DEFAULT_LOCK_TIMEOUT: Duration = Duration::from_secs(120);

/// Age after which recorded lock metadata is reported as stale (diagnostics only).
pub const STALE_LOCK_AGE: Duration = Duration::from_secs(30 * 60);

/// Marker present in the [`Error`] produced by a timed-out acquisition.
pub const LOCK_TIMEOUT_MARKER: &str = "worktree lock timeout";

/// Marker present in the [`Error`] produced by a contended acquisition.
pub const LOCK_CONTENTION_MARKER: &str = "worktree lock is held";

/// Directory (inside the repository) the lock file lives in when `.git` exists.
const GIT_DIR: &str = ".git";

/// Budget used for a configured timeout of zero, so "no wait" still attempts once.
const IMMEDIATE_TIMEOUT: Duration = Duration::from_millis(1);

/// Configuration of the repository lock.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorktreeLockConfig {
    /// Whether cross-process locking is attempted (Python's `repo_lock`).
    pub enabled: bool,
    /// How long [`WorktreeLock::acquire`] waits before reporting a timeout.
    pub timeout: Duration,
}

impl Default for WorktreeLockConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            timeout: DEFAULT_LOCK_TIMEOUT,
        }
    }
}

impl WorktreeLockConfig {
    /// A configuration that never takes the lock (Python's `repo_lock=False`).
    #[must_use]
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            timeout: DEFAULT_LOCK_TIMEOUT,
        }
    }

    /// Overrides the acquisition deadline, including sub-second budgets.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

/// Diagnostic metadata recorded in the lock file by the acquiring process.
///
/// Informational only: arbitration is the `flock`, so neither field is trusted
/// for correctness. `acquired_at` is an RFC 3339 timestamp.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WorktreeLockMetadata {
    /// Process id of the lock holder at acquisition time.
    pub pid: u32,
    /// Wall-clock acquisition time, RFC 3339.
    pub acquired_at: String,
}

/// Why a [`WorktreeLock`] acquisition did not produce a guard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorktreeLockError {
    /// Another process (or handle) holds the lock and no waiting was requested.
    Contended {
        /// The locked target: the repository root.
        target: PathBuf,
        /// The sidecar lock file.
        lock_path: PathBuf,
    },
    /// The lock stayed held for the whole configured budget.
    TimedOut {
        /// The locked target: the repository root.
        target: PathBuf,
        /// The sidecar lock file.
        lock_path: PathBuf,
        /// How long the acquisition waited before giving up.
        waited: Duration,
    },
    /// The lock file could not be created or opened.
    Io {
        /// The locked target: the repository root.
        target: PathBuf,
        /// The sidecar lock file.
        lock_path: PathBuf,
        /// The underlying operating-system message.
        message: String,
    },
}

impl WorktreeLockError {
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

    /// How long acquisition waited, when it timed out.
    #[must_use]
    pub const fn waited(&self) -> Option<Duration> {
        match self {
            Self::TimedOut { waited, .. } => Some(*waited),
            Self::Contended { .. } | Self::Io { .. } => None,
        }
    }

    /// Whether the acquisition gave up after its deadline.
    #[must_use]
    pub const fn is_timeout(&self) -> bool {
        matches!(self, Self::TimedOut { .. })
    }

    /// Whether the lock is held elsewhere and no wait was requested.
    #[must_use]
    pub const fn is_contention(&self) -> bool {
        matches!(self, Self::Contended { .. })
    }

    /// The message used for the converted [`Error`], with its stable marker.
    fn message(&self) -> String {
        match self {
            Self::TimedOut {
                target,
                lock_path,
                waited,
            } => format!(
                "{LOCK_TIMEOUT_MARKER}: waited {:.1}s for {} (lock file: {})",
                waited.as_secs_f64(),
                target.display(),
                lock_path.display()
            ),
            Self::Contended { target, lock_path } => format!(
                "{LOCK_CONTENTION_MARKER}: {} is already locked (lock file: {})",
                target.display(),
                lock_path.display()
            ),
            Self::Io {
                target,
                lock_path,
                message,
            } => format!(
                "worktree lock failed for {}: {message} (lock file: {})",
                target.display(),
                lock_path.display()
            ),
        }
    }
}

impl fmt::Display for WorktreeLockError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message())
    }
}

impl std::error::Error for WorktreeLockError {}

impl From<WorktreeLockError> for Error {
    fn from(error: WorktreeLockError) -> Self {
        Self::tool(error.message())
    }
}

impl From<LockError> for WorktreeLockError {
    fn from(error: LockError) -> Self {
        match error {
            LockError::Contended {
                target, lock_path, ..
            } => Self::Contended { target, lock_path },
            LockError::TimedOut {
                target,
                lock_path,
                waited,
                ..
            } => Self::TimedOut {
                target,
                lock_path,
                waited,
            },
            LockError::Io {
                target,
                lock_path,
                source,
            } => Self::Io {
                target,
                lock_path,
                message: source.to_string(),
            },
        }
    }
}

/// Whether `error` is a converted acquisition timeout.
#[must_use]
pub fn is_lock_timeout(error: &Error) -> bool {
    error.to_string().contains(LOCK_TIMEOUT_MARKER)
}

/// Whether `error` is a converted acquisition contention.
#[must_use]
pub fn is_lock_contention(error: &Error) -> bool {
    error.to_string().contains(LOCK_CONTENTION_MARKER)
}

/// A repository-scoped cross-process advisory lock.
///
/// Cheap to clone-free (it owns one [`FileLockManager`]) and safe to build per
/// operation: the lock file and therefore the exclusion are derived from the
/// repository root, not from the handle.
#[derive(Debug, Clone)]
pub struct WorktreeLock {
    /// Repository root the lock is scoped to, and the hashed target.
    repo_root: PathBuf,
    /// Directory holding the sidecar lock file.
    root: PathBuf,
    /// The configured policy.
    config: WorktreeLockConfig,
    /// The advisory-lock seam (sidecar naming, deadline, platform gating).
    manager: FileLockManager,
}

impl WorktreeLock {
    /// Builds a lock for the repository at `repo_root`.
    ///
    /// The lock file lives in `<repo_root>/.git` when that is an existing
    /// directory, and in `<workspace_root>/.worktree` otherwise. The hashed
    /// target is `repo_root`, so every handle over one repository shares one
    /// lock file.
    #[must_use]
    pub fn new(repo_root: &Path, workspace_root: &Path, config: WorktreeLockConfig) -> Self {
        let repo_root = repo_root.to_path_buf();
        let git_dir = repo_root.join(GIT_DIR);
        let root = if git_dir.is_dir() {
            git_dir
        } else {
            workspace_root.join(INDEX_DIR)
        };
        let local = LocalWorkConfig {
            cross_process_locks: config.enabled,
            ..LocalWorkConfig::default()
        };
        // The seam reads a `None` deadline as "wait forever"; a configured zero
        // therefore becomes a minimal budget so one attempt is made.
        let timeout = if config.timeout.is_zero() {
            IMMEDIATE_TIMEOUT
        } else {
            config.timeout
        };
        let manager = FileLockManager::with_root(&local, root.clone()).with_timeout(Some(timeout));
        Self {
            repo_root,
            root,
            config,
            manager,
        }
    }

    /// The configured policy.
    #[must_use]
    pub fn config(&self) -> &WorktreeLockConfig {
        &self.config
    }

    /// The directory holding the sidecar lock file.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The repository root this lock is scoped to.
    #[must_use]
    pub fn repo_root(&self) -> &Path {
        &self.repo_root
    }

    /// The sidecar lock file for this repository.
    #[must_use]
    pub fn lock_path(&self) -> PathBuf {
        self.manager.lock_path(&self.repo_root)
    }

    /// Whether a real cross-process lock is possible (Unix and enabled).
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.manager.is_enabled()
    }

    /// Takes the lock, waiting up to the configured timeout.
    ///
    /// On success the acquisition time and process id are recorded in the lock
    /// file, best-effort.
    ///
    /// # Errors
    ///
    /// Returns [`WorktreeLockError::TimedOut`] once the deadline passes and
    /// [`WorktreeLockError::Io`] when the lock file cannot be created or opened.
    pub async fn acquire(&self) -> Result<WorktreeLockGuard, WorktreeLockError> {
        let guard = self
            .manager
            .acquire(&self.repo_root, LockMode::Write)
            .await?;
        Ok(self.record(guard).await)
    }

    /// Takes the lock without waiting.
    ///
    /// # Errors
    ///
    /// Returns [`WorktreeLockError::Contended`] when the lock is held elsewhere
    /// and [`WorktreeLockError::Io`] when the lock file cannot be created or
    /// opened.
    pub async fn try_acquire(&self) -> Result<WorktreeLockGuard, WorktreeLockError> {
        let guard = self
            .manager
            .try_acquire(&self.repo_root, LockMode::Write)
            .await?;
        Ok(self.record(guard).await)
    }

    /// Reads the metadata an earlier holder left in the lock file.
    ///
    /// Diagnostic only: missing, unreadable or malformed content yields `None`
    /// and never fails. This is a small synchronous read, so callers on the
    /// async paths should treat it as a hint rather than a hot path.
    #[must_use]
    pub fn read_metadata(&self) -> Option<WorktreeLockMetadata> {
        let text = std::fs::read_to_string(self.lock_path()).ok()?;
        parse_metadata(&text)
    }

    /// Reports metadata older than [`STALE_LOCK_AGE`] relative to `now`.
    ///
    /// A stale timestamp means the last recorded acquisition is old — the
    /// holder crashed, or the wait is genuinely long. It never authorises
    /// breaking the lock: `flock` already released whatever the kernel holds.
    #[must_use]
    pub fn stale_metadata(&self, now: DateTime<Utc>) -> Option<WorktreeLockMetadata> {
        let metadata = self.read_metadata()?;
        let acquired = DateTime::parse_from_rfc3339(&metadata.acquired_at).ok()?;
        let age = now.signed_duration_since(acquired.with_timezone(&Utc));
        let age = age.to_std().ok()?;
        (age >= STALE_LOCK_AGE).then_some(metadata)
    }

    /// Wraps a seam guard, recording best-effort metadata while the lock is held.
    async fn record(&self, guard: FileLockGuard) -> WorktreeLockGuard {
        let metadata = self.write_metadata(&guard).await;
        WorktreeLockGuard { guard, metadata }
    }

    /// Writes `pid`/`acquired_at` into the lock file; never fails the acquisition.
    async fn write_metadata(&self, guard: &FileLockGuard) -> Option<WorktreeLockMetadata> {
        // An inactive guard (disabled, or no `flock`) holds nothing, so writing
        // would create state that no lock backs.
        if !guard.is_active() {
            return None;
        }
        let metadata = WorktreeLockMetadata {
            pid: std::process::id(),
            acquired_at: Utc::now().to_rfc3339(),
        };
        let body = format!(
            "pid={}\nacquired_at={}\n",
            metadata.pid, metadata.acquired_at
        );
        if let Err(error) = tokio::fs::write(guard.lock_path(), body.as_bytes()).await {
            tracing::debug!(
                path = %guard.lock_path().display(),
                %error,
                "failed to record worktree lock metadata"
            );
            return None;
        }
        Some(metadata)
    }
}

/// A held repository lock.
///
/// Dropping the guard closes the locked descriptor, which releases the `flock`
/// on every path including unwind. The lock is an OS advisory lock, not a
/// blocking in-process mutex, so the guard may be held across `.await` points.
#[derive(Debug)]
pub struct WorktreeLockGuard {
    /// The seam guard whose `Drop` releases the advisory lock.
    guard: FileLockGuard,
    /// Metadata written at acquisition, when it could be recorded.
    metadata: Option<WorktreeLockMetadata>,
}

impl WorktreeLockGuard {
    /// The sidecar lock file, whether or not a lock is actually held.
    #[must_use]
    pub fn lock_path(&self) -> &Path {
        self.guard.lock_path()
    }

    /// Whether a real advisory lock is held (false when disabled or unsupported).
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.guard.is_active()
    }

    /// The metadata recorded at acquisition, when it could be written.
    #[must_use]
    pub fn metadata(&self) -> Option<&WorktreeLockMetadata> {
        self.metadata.as_ref()
    }
}

/// Parses `pid=<u32>` and `acquired_at=<rfc3339>` lines, ignoring everything else.
///
/// Either key missing, a malformed pid, or a timestamp that is not RFC 3339
/// yields `None`: the pair is only useful together.
fn parse_metadata(text: &str) -> Option<WorktreeLockMetadata> {
    let mut pid = None;
    let mut acquired_at = None;
    for line in text.lines() {
        let Some((key, value)) = line.trim().split_once('=') else {
            continue;
        };
        let value = value.trim();
        match key.trim() {
            "pid" if pid.is_none() => pid = value.parse::<u32>().ok(),
            "acquired_at"
                if acquired_at.is_none() && DateTime::parse_from_rfc3339(value).is_ok() =>
            {
                acquired_at = Some(value.to_owned());
            }
            _ => {}
        }
    }
    Some(WorktreeLockMetadata {
        pid: pid?,
        acquired_at: acquired_at?,
    })
}

#[cfg(test)]
mod tests {
    //! Metadata parsing edge cases: the sidecar content is attacker-adjacent
    //! free-form text, so every malformed variant must yield `None`.
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::parse_metadata;

    #[test]
    fn parses_the_two_recorded_keys() {
        let metadata = parse_metadata("pid=42\nacquired_at=2020-01-01T00:00:00+00:00\n")
            .expect("well-formed metadata");
        assert_eq!(metadata.pid, 42);
        assert_eq!(metadata.acquired_at, "2020-01-01T00:00:00+00:00");
    }

    #[test]
    fn ignores_unknown_lines_and_whitespace() {
        let metadata = parse_metadata(
            "# comment\n\n  pid = 7 \nextra=value\nacquired_at = 2021-02-03T04:05:06Z\n",
        )
        .expect("tolerant parsing");
        assert_eq!(metadata.pid, 7);
        assert_eq!(metadata.acquired_at, "2021-02-03T04:05:06Z");
    }

    #[test]
    fn rejects_partial_or_malformed_content() {
        assert!(parse_metadata("").is_none());
        assert!(parse_metadata("pid=1\n").is_none());
        assert!(parse_metadata("acquired_at=2020-01-01T00:00:00+00:00\n").is_none());
        assert!(parse_metadata("pid=abc\nacquired_at=2020-01-01T00:00:00+00:00\n").is_none());
        assert!(parse_metadata("pid=-1\nacquired_at=2020-01-01T00:00:00+00:00\n").is_none());
        assert!(parse_metadata("pid=1\nacquired_at=yesterday\n").is_none());
        assert!(parse_metadata("no separator at all").is_none());
    }
}
