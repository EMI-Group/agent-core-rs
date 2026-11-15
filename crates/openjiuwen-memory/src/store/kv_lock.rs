//! Distributed KV lock seam: in-process, cross-process file locks, and typed `Unsupported` remote brokers.
//!
//! Python `openjiuwen/core/memory/common/distributed_lock.py` (v0.1.20) guards the memory
//! layouts that are touched by more than one writer: it stores an owner token under the
//! `_lock/{name}` key with `exclusive_set` plus a TTL, deletes it on release (get-then-delete),
//! and polls until the key is free. This module keeps that API shape as a seam with two real
//! local implementations and a typed `Unsupported` broker:
//!
//! | Backend | Excludes | Backing mechanism |
//! | --- | --- | --- |
//! | [`InProcessKvLock`] | tasks inside one process | one `tokio::sync::Mutex` per lock name |
//! | [`FileKvLock`] | other processes sharing a directory | advisory `flock` on a sidecar file |
//! | [`UnsupportedKvLock`] | nothing | documented `Unsupported` seam (multi-node brokers) |
//!
//! # Where the seam is used
//!
//! A migration run and the dream worker each take one lock per store root, so two processes
//! (or two tasks) never apply the same schema step or sweep the same dream jobs
//! concurrently. The lock name is the unit of exclusion (Python's `_lock/{name}` key):
//! `migrate`, `dream`, or a caller-chosen name such as `team_memory`.
//!
//! # Timeout semantics
//!
//! Python's `DistributedLock` waits forever. [`LockOptions::timeout`] keeps that as the
//! default (`None` waits forever, as does [`FileKvLock::with_timeout`]'s `None`) and adds an
//! optional bound: a held lock then yields [`Error::timeout`](openjiuwen_common::error::Error::timeout)
//! instead of parking the caller indefinitely.
//!
//! # TTL
//!
//! [`LockOptions::ttl`] is recorded as [`KvLockGuard::expires_at`] for lease-capable backends.
//! Neither local backend evicts a holder on expiry: an in-process guard is held until it is
//! dropped, and a file lock lives until its descriptor closes (or the process exits). A TTL
//! is therefore advisory here and becomes load-bearing only for a future lease broker (Redis
//! `SET NX PX`, which stays [`UnsupportedKvLock`]).
//!
//! # Lock-file layout
//!
//! [`FileKvLock::lock_path`] derives one logical sidecar per lock name under the caller's
//! root (`<root>/<name>.lock`). The advisory lock itself is taken by
//! [`openjiuwen_sysop::fs::lock::FileLockManager`], which hashes its *target* into
//! `<root>/<sha256(target)>.lock`; [`FileKvLock::sidecar_path`] exposes that real file. Only
//! processes that go through this seam take part: `flock` is advisory, and both processes must
//! share one root for the exclusion to work.
//!
//! # Deferred
//!
//! Multi-node brokers (`Redis`, `ZooKeeper`, `etcd`, a database row lock) are not implemented:
//! [`UnsupportedKvLock`] carries the documented `Unsupported` seam, and no process-global lock
//! registry exists (the backend is injected, like every other store seam).

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_sysop::config::LocalWorkConfig;
use openjiuwen_sysop::fs::lock::{FileLockGuard, FileLockManager, LockError, LockMode};
use parking_lot::Mutex;
use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};

/// Default lease duration requested from lease-capable backends.
pub const DEFAULT_LOCK_TTL: Duration = Duration::from_secs(30);

/// Default delay between acquisition attempts while a lock is held elsewhere.
///
/// This mirrors `openjiuwen-sysop`'s polling interval: the local backends poll (or park on a
/// mutex) at this granularity, so a released lock is visible almost immediately.
pub const DEFAULT_POLL_INTERVAL: Duration = Duration::from_millis(10);

/// Longest accepted lock name or owner, matching Python's `_lock/{name}` key budget.
const MAX_IDENTIFIER_LEN: usize = 128;

/// How to acquire a lock.
///
/// Built with [`LockOptions::new`] (which records the owner) and tuned with the `with_*`
/// builders. [`LockOptions::validate`] is called by every backend before it touches a lock, so
/// a bad owner or a zero poll interval fails with [`Error::Validation`] rather than silently.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LockOptions {
    /// Caller identity recorded on the guard and in the acquisition log; never empty.
    pub owner: String,
    /// Lease duration requested from lease-capable backends; `None` means
    /// [`DEFAULT_LOCK_TTL`].
    pub ttl: Option<Duration>,
    /// How long to wait for a lock held elsewhere; `None` waits forever.
    pub timeout: Option<Duration>,
    /// Delay between acquisition attempts while a lock is held elsewhere.
    pub poll_interval: Duration,
}

impl LockOptions {
    /// Options for `owner` with the default TTL, an unbounded wait, and
    /// [`DEFAULT_POLL_INTERVAL`].
    #[must_use]
    pub fn new(owner: impl Into<String>) -> Self {
        Self {
            owner: owner.into(),
            ttl: None,
            timeout: None,
            poll_interval: DEFAULT_POLL_INTERVAL,
        }
    }

    /// Sets the requested lease duration.
    #[must_use]
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }

    /// Bounds the wait for a lock held elsewhere.
    ///
    /// A zero duration is rejected by [`Self::validate`] rather than reinterpreted as "wait
    /// forever" by the file backend.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Sets the delay between acquisition attempts.
    #[must_use]
    pub fn with_poll_interval(mut self, interval: Duration) -> Self {
        self.poll_interval = interval;
        self
    }

    /// The resolved TTL: [`DEFAULT_LOCK_TTL`] unless the caller chose one.
    #[must_use]
    pub fn ttl(&self) -> Duration {
        self.ttl.unwrap_or(DEFAULT_LOCK_TTL)
    }

    /// Checks the owner, the optional timeout, and the poll interval.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the owner is empty, blank, longer than 128
    /// characters, or contains a NUL, when a configured timeout is zero, or when the poll
    /// interval is zero.
    pub fn validate(&self) -> Result<()> {
        validate_identifier("lock owner", &self.owner)?;
        if matches!(self.timeout, Some(timeout) if timeout.is_zero()) {
            return Err(Error::validation(format!(
                "lock {owner:?} timeout must be greater than zero; omit it to wait forever",
                owner = self.owner
            )));
        }
        if self.poll_interval.is_zero() {
            return Err(Error::validation(format!(
                "lock {owner:?} poll interval must be greater than zero",
                owner = self.owner
            )));
        }
        Ok(())
    }
}

/// Rejects an empty, blank, over-long, or NUL-bearing identifier.
fn validate_identifier(label: &str, value: &str) -> Result<()> {
    if value.is_empty() {
        return Err(Error::validation(format!("{label} must not be empty")));
    }
    if value.trim().is_empty() {
        return Err(Error::validation(format!(
            "{label} {value:?} must not be blank"
        )));
    }
    if value.contains('\0') {
        return Err(Error::validation(format!(
            "{label} {value:?} must not contain a NUL character"
        )));
    }
    let length = value.chars().count();
    if length > MAX_IDENTIFIER_LEN {
        return Err(Error::validation(format!(
            "{label} {value:?} must be at most {MAX_IDENTIFIER_LEN} characters, got {length}"
        )));
    }
    Ok(())
}

/// Rejects a lock name that would also be unusable as one path segment.
fn validate_lock_name(name: &str) -> Result<()> {
    validate_identifier("lock name", name)?;
    if name.contains('/') || name.contains('\\') {
        return Err(Error::validation(format!(
            "lock name {name:?} must not contain a path separator"
        )));
    }
    Ok(())
}

/// The lease expiry recorded on a guard, or `None` when the TTL does not fit a timestamp.
fn lease_expiry(ttl: Duration) -> Option<DateTime<Utc>> {
    let delta = chrono::TimeDelta::from_std(ttl).ok()?;
    Utc::now().checked_add_signed(delta)
}

/// The handle whose ownership keeps a lock acquired.
///
/// The payloads are never inspected: owning one *is* holding the lock, and dropping it is the
/// release, so the fields are read only by [`Drop`].
enum GuardInner {
    /// Holding the guard is holding the [`tokio::sync::Mutex`].
    InProcess(#[allow(dead_code)] OwnedMutexGuard<()>),
    /// Owning the descriptor keeps the advisory `flock`.
    File(#[allow(dead_code)] FileLockGuard),
}

/// A held lock; dropping it releases (immediately and synchronously).
///
/// The guard is `Send`, so it can be moved into a task or held across `.await`. Release is
/// purely drop-based: there is no daemon and no cleanup task, and an in-process guard frees
/// its name the moment it is dropped.
pub struct KvLockGuard {
    /// The lock name (Python's `_lock/{name}` key).
    name: String,
    /// The caller identity that acquired the lock.
    owner: String,
    /// The backend label (`"in-process"`, `"file"`, or a vendor name).
    backend: &'static str,
    /// The requested lease expiry, advisory for the local backends.
    expires_at: Option<DateTime<Utc>>,
    /// The held handle; `None` once released.
    inner: Option<GuardInner>,
}

impl KvLockGuard {
    /// Builds a held guard. Private: guards come from a [`DistributedKvLock`] backend.
    fn new(name: &str, options: &LockOptions, backend: &'static str, inner: GuardInner) -> Self {
        Self {
            name: name.to_owned(),
            owner: options.owner.clone(),
            backend,
            expires_at: lease_expiry(options.ttl()),
            inner: Some(inner),
        }
    }

    /// The locked name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The owner recorded at acquisition.
    #[must_use]
    pub fn owner(&self) -> &str {
        &self.owner
    }

    /// The backend that holds the lock.
    #[must_use]
    pub fn backend(&self) -> &'static str {
        self.backend
    }

    /// Whether the lock is still held (`false` only after [`Self::release`]).
    ///
    /// Every guard a backend returns is a real one: [`FileKvLock`] refuses an environment
    /// without a working advisory lock, so a held guard always rests on an active lock.
    #[must_use]
    pub fn is_held(&self) -> bool {
        self.inner.is_some()
    }

    /// The requested lease expiry, or `None` when the TTL does not fit a timestamp.
    #[must_use]
    pub fn expires_at(&self) -> Option<DateTime<Utc>> {
        self.expires_at
    }

    /// Releases the lock explicitly; the [`Drop`] implementation stays the fallback.
    ///
    /// Dropping the guard runs the same path, so this exists for call sites that want the
    /// release to be visible at the point it happens.
    pub fn release(self) {
        drop(self);
    }
}

impl fmt::Debug for KvLockGuard {
    /// Reports the lock identity and state; the held handle is opaque on purpose.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("KvLockGuard")
            .field("name", &self.name)
            .field("owner", &self.owner)
            .field("backend", &self.backend)
            .field("held", &self.is_held())
            .field("expires_at", &self.expires_at)
            .finish_non_exhaustive()
    }
}

impl Drop for KvLockGuard {
    /// Releases the in-process mutex or closes the lock descriptor.
    fn drop(&mut self) {
        if let Some(inner) = self.inner.take() {
            tracing::debug!(
                target: "openjiuwen_memory::store",
                lock = %self.name,
                owner = %self.owner,
                backend = self.backend,
                "released kv lock"
            );
            drop(inner);
        }
    }
}

/// A lock backend.
///
/// The trait is the whole seam: a caller (a migration runner, the dream worker) takes a lock
/// by name and holds the guard for the critical section. Implementations must be `Send + Sync`
/// because one instance is shared by every task of an application.
#[async_trait]
pub trait DistributedKvLock: Send + Sync {
    /// The backend label recorded on guards (`"in-process"`, `"file"`, ...).
    fn backend(&self) -> &'static str;

    /// Acquires `name` according to `options`, waiting or failing per
    /// [`LockOptions::timeout`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an unusable name or options,
    /// [`Error::timeout`] when a bounded wait expires, [`Error::external`] when the file
    /// backend cannot create or open its lock file, and [`Error::Unsupported`] from a
    /// documented-but-unimplemented broker.
    async fn acquire(&self, name: &str, options: &LockOptions) -> Result<KvLockGuard>;
}

/// `tokio::sync::Mutex` per lock name: serialises tasks in one process.
///
/// A clone shares the lock table (the usual way to hand the seam to another task); two
/// separate [`InProcessKvLock::new`] instances do not exclude each other, so an application
/// builds one per store root and injects it. The table only ever grows: memory keys are
/// bounded, and pruning an entry would let two tasks hold "the same" name.
#[derive(Debug, Clone, Default)]
pub struct InProcessKvLock {
    /// One mutex per lock name, created on first use.
    locks: Arc<Mutex<HashMap<String, Arc<AsyncMutex<()>>>>>,
}

impl InProcessKvLock {
    /// Creates an empty lock table.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The names currently held, sorted, for diagnostics and tests.
    #[must_use]
    pub fn held_names(&self) -> Vec<String> {
        let locks = self.locks.lock();
        let mut names: Vec<String> = locks
            .iter()
            .filter(|(_, mutex)| mutex.try_lock().is_err())
            .map(|(name, _)| name.clone())
            .collect();
        names.sort();
        names
    }

    /// The per-name mutex, created on first use. The table lock is dropped before returning.
    fn mutex_for(&self, name: &str) -> Arc<AsyncMutex<()>> {
        let mut locks = self.locks.lock();
        Arc::clone(
            locks
                .entry(name.to_owned())
                .or_insert_with(|| Arc::new(AsyncMutex::new(()))),
        )
    }
}

#[async_trait]
impl DistributedKvLock for InProcessKvLock {
    fn backend(&self) -> &'static str {
        "in-process"
    }

    async fn acquire(&self, name: &str, options: &LockOptions) -> Result<KvLockGuard> {
        validate_lock_name(name)?;
        options.validate()?;
        let mutex = self.mutex_for(name);
        let inner = match options.timeout {
            Some(timeout) => match tokio::time::timeout(timeout, mutex.lock_owned()).await {
                Ok(inner) => inner,
                Err(_elapsed) => {
                    return Err(Error::timeout(format!(
                        "timed out after {timeout:?} waiting for the in-process lock {name:?}"
                    )));
                }
            },
            None => mutex.lock_owned().await,
        };
        tracing::debug!(
            target: "openjiuwen_memory::store",
            lock = name,
            owner = %options.owner,
            backend = "in-process",
            ttl = ?options.ttl,
            "acquired kv lock"
        );
        Ok(KvLockGuard::new(
            name,
            options,
            "in-process",
            GuardInner::InProcess(inner),
        ))
    }
}

/// Advisory `flock` on a sidecar file under a caller-provided root: excludes other processes.
///
/// The root is the shared rendezvous point: every process that must exclude the others opens
/// its [`FileKvLock`] over the same directory. The lock file is created on first use and is
/// never read or written.
///
/// Where the platform has no `flock` (or the sysop configuration disables cross-process
/// locks), [`Self::is_enabled`] is `false` and every acquisition is a typed
/// [`Error::Unsupported`]: an inactive handle would let two processes both "hold" the same
/// lock, which is worse than refusing. Use [`InProcessKvLock`] on such a platform.
#[derive(Debug, Clone)]
pub struct FileKvLock {
    /// The sysop manager that creates the sidecar files and polls the deadline.
    manager: FileLockManager,
    /// The lock root shared by every participating process.
    root: PathBuf,
}

impl FileKvLock {
    /// Builds a file backend rooted at `root`, creating the directory when missing.
    ///
    /// The constructor is synchronous (one `create_dir_all`) so a caller can build the seam
    /// before entering the async runtime; the acquisition itself is fully async.
    ///
    /// # Errors
    ///
    /// Returns [`Error::external`] when the root cannot be created.
    pub fn new(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        std::fs::create_dir_all(&root).map_err(|error| {
            Error::external(error)
                .with_context(format!("creating the kv lock root {}", root.display()))
        })?;
        let manager = FileLockManager::with_root(&LocalWorkConfig::default(), root.clone())
            .with_timeout(None);
        Ok(Self { manager, root })
    }

    /// Sets the fallback deadline used when [`LockOptions::timeout`] is `None`
    /// (`None` waits forever).
    ///
    /// A zero duration is normalised to `None` by the sysop manager.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.manager = self.manager.with_timeout(timeout);
        self
    }

    /// The lock root.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The logical sidecar path for `name` (`<root>/<name>.lock`).
    ///
    /// This is the target handed to the sysop manager; see [`Self::sidecar_path`] for the file
    /// whose `flock` the manager actually holds. The name is not validated here, so callers
    /// with untrusted input must go through [`DistributedKvLock::acquire`].
    #[must_use]
    pub fn lock_path(&self, name: &str) -> PathBuf {
        self.root.join(format!("{name}.lock"))
    }

    /// The file whose advisory lock [`DistributedKvLock::acquire`] takes for `name`.
    ///
    /// The sysop manager hashes its target into `<root>/<sha256>.lock`, so this differs from
    /// [`Self::lock_path`]; it exists for diagnostics and for tests that assert the sidecar
    /// was created.
    #[must_use]
    pub fn sidecar_path(&self, name: &str) -> PathBuf {
        self.manager.lock_path(&self.lock_path(name))
    }

    /// Whether a real advisory lock will be taken (false on platforms without `flock`).
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.manager.is_enabled()
    }
}

/// Maps a sysop lock failure onto the crate's error kind.
fn map_lock_error(name: &str, error: LockError) -> Error {
    if matches!(&error, LockError::Io { .. }) {
        return Error::external(error).with_context(format!("acquiring the kv lock {name:?}"));
    }
    Error::timeout(format!("could not acquire the kv lock {name:?}: {error}"))
}

#[async_trait]
impl DistributedKvLock for FileKvLock {
    fn backend(&self) -> &'static str {
        "file"
    }

    async fn acquire(&self, name: &str, options: &LockOptions) -> Result<KvLockGuard> {
        validate_lock_name(name)?;
        options.validate()?;
        if !self.manager.is_enabled() {
            // Without `flock` the sysop manager hands back an inactive handle; returning it
            // would report "held" for a lock nobody holds, so the exclusion is refused.
            return Err(Error::unsupported(format!(
                "the file kv lock backend cannot exclude processes for {name:?}: advisory \
                 cross-process locks are disabled on this platform/configuration; use the \
                 in-process backend instead"
            )));
        }
        let target = self.lock_path(name);
        let timeout = options.timeout.or_else(|| self.manager.timeout());
        let manager = self.manager.clone().with_timeout(timeout);
        let guard = manager
            .acquire(&target, LockMode::Write)
            .await
            .map_err(|error| map_lock_error(name, error))?;
        tracing::debug!(
            target: "openjiuwen_memory::store",
            lock = name,
            owner = %options.owner,
            backend = "file",
            path = %target.display(),
            ttl = ?options.ttl,
            "acquired kv lock"
        );
        Ok(KvLockGuard::new(
            name,
            options,
            "file",
            GuardInner::File(guard),
        ))
    }
}

/// A backend that is documented but not implemented (multi-node brokers).
///
/// Construction never touches the network; every acquisition returns
/// [`Error::Unsupported`] naming the backend and pointing at the local alternatives, so a
/// deployment that assumes a distributed lock fails loudly instead of running unlocked.
#[derive(Debug, Clone)]
pub struct UnsupportedKvLock {
    /// The broker label (`"redis"`, `"zookeeper"`, ...).
    backend: &'static str,
}

impl UnsupportedKvLock {
    /// A stub for `backend`.
    #[must_use]
    pub fn new(backend: &'static str) -> Self {
        Self { backend }
    }

    /// A stub for a Redis-backed lock.
    #[must_use]
    pub fn redis() -> Self {
        Self::new("redis")
    }
}

#[async_trait]
impl DistributedKvLock for UnsupportedKvLock {
    fn backend(&self) -> &'static str {
        self.backend
    }

    async fn acquire(&self, name: &str, options: &LockOptions) -> Result<KvLockGuard> {
        let _ = (name, options);
        Err(Error::unsupported(format!(
            "the {} distributed KV lock is not implemented: multi-node lock brokers are deferred; \
             use InProcessKvLock or FileKvLock",
            self.backend
        )))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use openjiuwen_common::error::ErrorKind;

    use super::*;

    /// Guards are moved into tasks, so they must be `Send`.
    fn assert_send<T: Send>() {}

    #[test]
    fn guards_are_send_and_options_default_to_the_documented_values() {
        assert_send::<KvLockGuard>();
        assert_send::<InProcessKvLock>();
        assert_send::<FileKvLock>();
        assert_send::<UnsupportedKvLock>();

        let options = LockOptions::new("alice");
        assert_eq!(options.owner, "alice");
        assert_eq!(options.ttl(), DEFAULT_LOCK_TTL);
        assert_eq!(options.poll_interval, DEFAULT_POLL_INTERVAL);
        assert_eq!(options.timeout, None);
        options.validate().unwrap();

        let tuned = options
            .clone()
            .with_ttl(Duration::from_secs(5))
            .with_timeout(Duration::from_millis(20))
            .with_poll_interval(Duration::from_millis(2));
        assert_eq!(tuned.ttl(), Duration::from_secs(5));
        assert_eq!(tuned.timeout, Some(Duration::from_millis(20)));
        assert_eq!(tuned.poll_interval, Duration::from_millis(2));
        tuned.validate().unwrap();
    }

    #[tokio::test]
    async fn bad_names_and_options_are_validation_errors() {
        let locks = InProcessKvLock::new();
        let long_name = "x".repeat(MAX_IDENTIFIER_LEN + 1);
        for name in [
            "",
            "  ",
            "a/b",
            "back\\slash",
            "null\0name",
            long_name.as_str(),
        ] {
            let error = locks
                .acquire(name, &LockOptions::new("alice"))
                .await
                .unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation, "{name:?}: {error}");
        }

        let long_owner = "o".repeat(MAX_IDENTIFIER_LEN + 1);
        for owner in ["", "  ", "nul\0owner", long_owner.as_str()] {
            let error = LockOptions::new(owner).validate().unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation, "{owner:?}: {error}");
        }

        let error = LockOptions::new("alice")
            .with_ttl(Duration::from_secs(1))
            .with_timeout(Duration::ZERO)
            .validate()
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation, "{error}");

        let error = LockOptions::new("alice")
            .with_poll_interval(Duration::ZERO)
            .validate()
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation, "{error}");
    }

    #[tokio::test]
    async fn in_process_acquire_and_release() {
        let locks = InProcessKvLock::new();
        let guard = locks
            .acquire("migrate", &LockOptions::new("alice"))
            .await
            .unwrap();
        assert!(guard.is_held());
        assert_eq!(guard.name(), "migrate");
        assert_eq!(guard.owner(), "alice");
        assert_eq!(guard.backend(), "in-process");
        assert_eq!(locks.held_names(), vec!["migrate".to_owned()]);

        drop(guard);
        assert_eq!(locks.held_names(), Vec::<String>::new());

        let again = locks
            .acquire("migrate", &LockOptions::new("alice"))
            .await
            .unwrap();
        assert!(again.is_held());
    }

    #[tokio::test]
    async fn in_process_locks_serialise_tasks() {
        let locks = Arc::new(InProcessKvLock::new());
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let mut tasks = Vec::new();
        for _ in 0..4 {
            let locks = Arc::clone(&locks);
            let active = Arc::clone(&active);
            let max_active = Arc::clone(&max_active);
            tasks.push(tokio::spawn(async move {
                let guard = locks
                    .acquire("serial", &LockOptions::new("task"))
                    .await
                    .unwrap();
                let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                max_active.fetch_max(current, Ordering::SeqCst);
                for _ in 0..4 {
                    tokio::task::yield_now().await;
                }
                active.fetch_sub(1, Ordering::SeqCst);
                drop(guard);
            }));
        }
        for task in tasks {
            task.await.unwrap();
        }
        assert_eq!(
            max_active.load(Ordering::SeqCst),
            1,
            "only one task holds the in-process lock at a time"
        );
        assert_eq!(locks.held_names(), Vec::<String>::new());
    }

    #[tokio::test]
    async fn in_process_contention_times_out_and_then_succeeds() {
        let locks = Arc::new(InProcessKvLock::new());
        let (acquired_tx, acquired_rx) = tokio::sync::oneshot::channel();
        let release = Arc::new(tokio::sync::Notify::new());
        let holder = {
            let locks = Arc::clone(&locks);
            let release = Arc::clone(&release);
            tokio::spawn(async move {
                let guard = locks
                    .acquire("migrate", &LockOptions::new("holder"))
                    .await
                    .unwrap();
                let _ = acquired_tx.send(());
                release.notified().await;
                drop(guard);
            })
        };
        acquired_rx.await.unwrap();

        let error = locks
            .acquire(
                "migrate",
                &LockOptions::new("waiter").with_timeout(Duration::from_millis(50)),
            )
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Timeout, "{error}");
        assert!(error.message().contains("migrate"), "{error}");
        assert!(error.message().contains("in-process"), "{error}");

        release.notify_one();
        holder.await.unwrap();

        let guard = locks
            .acquire(
                "migrate",
                &LockOptions::new("waiter").with_timeout(Duration::from_millis(50)),
            )
            .await
            .unwrap();
        assert!(guard.is_held());
    }

    #[tokio::test]
    async fn different_names_do_not_contend() {
        let locks = InProcessKvLock::new();
        let alpha = locks
            .acquire("alpha", &LockOptions::new("alice"))
            .await
            .unwrap();
        let beta = locks
            .acquire("beta", &LockOptions::new("bob"))
            .await
            .unwrap();
        assert!(alpha.is_held());
        assert!(beta.is_held());
        assert_eq!(
            locks.held_names(),
            vec!["alpha".to_owned(), "beta".to_owned()]
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn file_locks_exclude_other_handles() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("locks");
        let locks = FileKvLock::new(&root).unwrap();
        assert!(root.is_dir(), "the constructor creates the lock root");
        assert_eq!(locks.root(), root.as_path());
        assert!(locks.is_enabled(), "unix has flock");
        assert_eq!(locks.lock_path("migrate"), root.join("migrate.lock"));

        let guard = locks
            .acquire("migrate", &LockOptions::new("alice"))
            .await
            .unwrap();
        assert!(guard.is_held());
        assert_eq!(guard.backend(), "file");
        assert!(
            locks.sidecar_path("migrate").is_file(),
            "the sidecar must exist while the lock is held"
        );

        let contender = FileKvLock::new(&root)
            .unwrap()
            .with_timeout(Some(Duration::from_millis(100)));
        let error = contender
            .acquire("migrate", &LockOptions::new("bob"))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Timeout, "{error}");
        assert!(error.message().contains("migrate"), "{error}");

        drop(guard);
        let second = contender
            .acquire("migrate", &LockOptions::new("bob"))
            .await
            .unwrap();
        assert!(second.is_held());

        let other_dir = tempfile::tempdir().unwrap();
        let elsewhere = FileKvLock::new(other_dir.path().join("locks"))
            .unwrap()
            .with_timeout(Some(Duration::from_millis(100)));
        let third = elsewhere
            .acquire("migrate", &LockOptions::new("carol"))
            .await
            .unwrap();
        assert!(third.is_held(), "a different root does not contend");
    }

    #[tokio::test]
    async fn a_disabled_file_backend_refuses_instead_of_reporting_a_held_lock() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("locks");
        std::fs::create_dir_all(&root).unwrap();
        let config = LocalWorkConfig {
            cross_process_locks: false,
            ..LocalWorkConfig::default()
        };
        let locks = FileKvLock {
            manager: FileLockManager::with_root(&config, root.clone()),
            root,
        };
        assert!(
            !locks.is_enabled(),
            "cross-process locks are disabled by configuration"
        );
        let error = locks
            .acquire("migrate", &LockOptions::new("alice"))
            .await
            .expect_err("a disabled backend must not hand out a guard");
        assert_eq!(error.kind(), ErrorKind::Unsupported, "{error}");
        assert!(error.message().contains("migrate"), "{error}");
        assert!(error.message().contains("in-process"), "{error}");
        // Two acquisitions therefore cannot both "hold" the same lock name.
        assert!(locks
            .acquire("migrate", &LockOptions::new("bob"))
            .await
            .is_err());
    }
    #[tokio::test]
    async fn remote_brokers_are_typed_unsupported() {
        let lock = UnsupportedKvLock::redis();
        assert_eq!(lock.backend(), "redis");
        let error = lock
            .acquire("migrate", &LockOptions::new("alice"))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported, "{error}");
        assert!(error.message().contains("redis"), "{error}");
        assert!(error.message().contains("multi-node"), "{error}");
    }

    #[tokio::test]
    async fn owner_ttl_release_and_debug() {
        let locks = InProcessKvLock::new();
        let guard = locks
            .acquire(
                "debug-lock",
                &LockOptions::new("alice").with_ttl(Duration::from_secs(60)),
            )
            .await
            .unwrap();
        assert_eq!(guard.owner(), "alice");
        let expires_at = guard.expires_at().expect("a 60s TTL fits a timestamp");
        assert!(expires_at > Utc::now());

        let debug = format!("{guard:?}");
        assert!(debug.contains("debug-lock"), "{debug}");
        assert!(debug.contains("alice"), "{debug}");
        assert!(debug.contains("in-process"), "{debug}");

        guard.release();
        assert!(
            locks.held_names().is_empty(),
            "release frees the name immediately"
        );

        let again = locks
            .acquire("debug-lock", &LockOptions::new("alice"))
            .await
            .unwrap();
        assert!(again.is_held());
    }
}
