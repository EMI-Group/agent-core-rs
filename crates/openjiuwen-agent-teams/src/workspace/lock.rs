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
//! 3. the holder writes its pid **and a unique owner token** into the file, refreshes the file's
//!    modification time while the guard lives, and deletes the file on drop **only while it still
//!    carries that token**.
//!
//! # Why the token matters
//!
//! A guard that is declared stale is reclaimed by the next caller: that caller removes the file and
//! creates its own. Without an ownership check the *original* guard would then delete the new
//! holder's file when it finished, and a third caller would be admitted next to the second — two
//! concurrent `git worktree add/remove` on one repository, which corrupts `.git/worktrees`.
//! So every release path (explicit [`WorktreeLockGuard::release`] and `Drop`) reads the file and
//! removes it only when the content still names this guard's token.
//!
//! # Why the heartbeat matters
//!
//! Reclamation is decided by age alone, so a critical section that legitimately outlives
//! [`LockOptions::stale_after`] — five ten-second `git` calls in a row — would be declared abandoned
//! while it is provably alive. A guard therefore spawns one small task that keeps the file's
//! modification time fresh while it still owns the token, and reclamation re-checks the stamp and
//! the content after reading them, so a file that was refreshed (or replaced) while it was being
//! inspected is never stolen.
//!
//! A crash leaves the file behind: no heartbeat runs, its age grows past
//! [`LockOptions::stale_after`], and the next caller reclaims it with a `warn!`.
//! Timeout raises [`Error::Timeout`] rather than degrading to lock-free, which would reintroduce
//! the race the lock prevents.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use tokio::io::AsyncWriteExt;
use tokio::task::JoinHandle;
use tracing::{debug, warn};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::id::uuid_hex;

/// File name of the repository-level worktree lock.
pub const LOCK_FILE_NAME: &str = "openjiuwen-worktree.lock";

/// Default time a caller waits for the lock before failing.
pub const DEFAULT_LOCK_WAIT: Duration = Duration::from_secs(10);

/// Default age after which an abandoned lock file is reclaimed.
pub const DEFAULT_STALE_AFTER: Duration = Duration::from_secs(60);

/// Default interval between two acquisition attempts.
pub const DEFAULT_POLL_INTERVAL: Duration = Duration::from_millis(25);

/// Shortest heartbeat interval a guard schedules; a shorter stale window gets no heartbeat.
///
/// [`LockOptions::stale_after`] is always at least three heartbeat intervals, so a live holder has
/// two chances to refresh its stamp before a waiter could consider it abandoned.
const MIN_HEARTBEAT_INTERVAL: Duration = Duration::from_millis(10);

/// Prefix of the line that identifies the lock holder inside the lock file.
const TOKEN_PREFIX: &str = "token=";

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
/// Both paths refuse to remove a file that no longer carries this guard's token: a reclaimed-and-
/// re-taken lock belongs to the new holder.
#[derive(Debug)]
pub struct WorktreeLockGuard {
    /// Lock file held by this guard.
    path: PathBuf,
    /// Unique ownership token stamped into the file, `pid-hex`.
    token: String,
    /// Shared with the heartbeat task: the token plus its stop flag.
    ownership: Arc<Ownership>,
    /// Task that keeps the stamp fresh while the guard lives.
    heartbeat: Option<JoinHandle<()>>,
    /// Set once the file has been removed, so `Drop` does not remove a file it no longer owns.
    released: bool,
}

impl WorktreeLockGuard {
    /// Returns the lock file held by this guard.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the ownership token this guard stamped into its lock file.
    #[must_use]
    pub fn token(&self) -> &str {
        &self.token
    }

    /// Removes the lock file and gives up the lock.
    ///
    /// A file that already vanished is a successful release, because the guard holds nothing any
    /// more. A file that now carries another guard's token is left alone (and logged): the lock was
    /// reclaimed while this guard was running, so the new holder — not this one — owns it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::SysOp`] when the file cannot be read or removed.
    pub async fn release(mut self) -> Result<()> {
        self.stop_heartbeat();
        match tokio::fs::read(&self.path).await {
            Ok(content) => {
                if !self.owns(&content) {
                    warn!(
                        lock = %self.path.display(),
                        token = %self.token,
                        "worktree lock was reclaimed by another holder; leaving its file in place"
                    );
                    self.released = true;
                    return Ok(());
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                // Someone else already reclaimed the file; nothing is held any more.
                self.released = true;
                return Ok(());
            }
            Err(error) => {
                return Err(Error::sys_op(format!(
                    "reading worktree lock {}: {error}",
                    self.path.display()
                )));
            }
        }
        match tokio::fs::remove_file(&self.path).await {
            Ok(()) => {
                self.released = true;
                debug!(lock = %self.path.display(), "released worktree lock");
                Ok(())
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                self.released = true;
                Ok(())
            }
            Err(error) => Err(Error::sys_op(format!(
                "releasing worktree lock {}: {error}",
                self.path.display()
            ))),
        }
    }

    /// Returns `true` when `content` names this guard's token.
    fn owns(&self, content: &[u8]) -> bool {
        let content = String::from_utf8_lossy(content);
        content
            .lines()
            .any(|line| line.trim() == format!("{TOKEN_PREFIX}{}", self.token))
    }

    /// Stops the heartbeat task, if it is still running.
    fn stop_heartbeat(&mut self) {
        self.ownership.stopped.store(true, Ordering::SeqCst);
        if let Some(handle) = self.heartbeat.take() {
            handle.abort();
        }
    }

    /// Stops refreshing the stamp, so a test can let this guard be declared stale.
    #[cfg(test)]
    pub(crate) fn suspend_heartbeat(&mut self) {
        self.stop_heartbeat();
    }
}

impl Drop for WorktreeLockGuard {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        self.stop_heartbeat();
        // Best effort: `Drop` cannot await and cannot report a failure, so problems are logged.
        // The ownership check is what keeps a stale-reclaimed guard from deleting the new holder's
        // file; a file that is already gone needs nothing.
        match std::fs::read(&self.path) {
            Ok(content) => {
                if !self.owns(&content) {
                    warn!(
                        lock = %self.path.display(),
                        token = %self.token,
                        "worktree lock was reclaimed by another holder; leaving its file in place"
                    );
                    return;
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return,
            Err(error) => {
                warn!(
                    lock = %self.path.display(),
                    %error,
                    "failed to read worktree lock on release; it will be reclaimed once stale"
                );
                return;
            }
        }
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

/// State shared between a guard and its heartbeat task.
#[derive(Debug)]
struct Ownership {
    /// Token stamped into the lock file.
    token: String,
    /// Set by the guard when it stops owning the file.
    stopped: AtomicBool,
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
        let token = new_token();
        match create_lock_file(path, &token).await {
            Ok(()) => {
                debug!(lock = %path.display(), "acquired worktree lock");
                let ownership = Arc::new(Ownership {
                    token: token.clone(),
                    stopped: AtomicBool::new(false),
                });
                let heartbeat =
                    spawn_heartbeat(path.to_path_buf(), Arc::clone(&ownership), options.stale_after);
                return Ok(WorktreeLockGuard {
                    path: path.to_path_buf(),
                    token,
                    ownership,
                    heartbeat,
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

/// Mints a fresh ownership token: `<pid>-<32 lowercase hex>`.
fn new_token() -> String {
    format!("{}-{}", std::process::id(), uuid_hex())
}

/// Renders the lock file content for `token`: the holder's pid and its ownership token.
fn lock_file_content(token: &str) -> String {
    format!("pid={}\n{TOKEN_PREFIX}{token}\n", std::process::id())
}

/// Creates the lock file exclusively and stamps it with this process id and `token`.
async fn create_lock_file(path: &Path, token: &str) -> std::io::Result<()> {
    let mut file = tokio::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .await?;
    file.write_all(lock_file_content(token).as_bytes()).await?;
    file.flush().await?;
    Ok(())
}

/// Removes `path` when it is older than `stale_after`; returns whether it was reclaimed.
///
/// The decision is taken three times over the same content: the file must be stale, its content
/// must be byte-identical across two reads, and it must still be stale after the content was read.
/// A holder whose heartbeat refreshed the stamp in between is therefore never stolen, and neither
/// is a file another waiter replaced. A lock file with an unreadable modification time is left
/// alone (fail closed) — the caller then times out instead of stealing a lock it cannot reason
/// about.
async fn reclaim_if_stale(path: &Path, stale_after: Duration) -> Result<bool> {
    let before = match tokio::fs::metadata(path).await {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(true),
        Err(error) => {
            return Err(Error::sys_op(format!(
                "inspecting worktree lock {}: {error}",
                path.display()
            )));
        }
    };
    if !is_stale(&before, stale_after) {
        return Ok(false);
    }
    let content = match tokio::fs::read(path).await {
        Ok(content) => content,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(true),
        Err(error) => {
            return Err(Error::sys_op(format!(
                "reading worktree lock {}: {error}",
                path.display()
            )));
        }
    };
    // Re-check after reading: a live holder refreshes the stamp, so a file that got younger while
    // it was being inspected belongs to somebody who is still working.
    let after = match tokio::fs::metadata(path).await {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(true),
        Err(error) => {
            return Err(Error::sys_op(format!(
                "inspecting worktree lock {}: {error}",
                path.display()
            )));
        }
    };
    if after.modified().ok() != before.modified().ok() || !is_stale(&after, stale_after) {
        return Ok(false);
    }
    match tokio::fs::read(path).await {
        Ok(again) if again == content => {}
        Ok(_) => return Ok(false),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(true),
        Err(error) => {
            return Err(Error::sys_op(format!(
                "reading worktree lock {}: {error}",
                path.display()
            )));
        }
    }
    let age = age_of(&after);
    warn!(
        lock = %path.display(),
        age_secs = age.map_or(0, |age| age.as_secs()),
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

/// Returns the age of a lock file, or `None` when its modification time is unreadable.
fn age_of(metadata: &std::fs::Metadata) -> Option<Duration> {
    metadata
        .modified()
        .ok()
        .and_then(|modified| SystemTime::now().duration_since(modified).ok())
}

/// Returns `true` when `metadata` describes a file that is at least `stale_after` old.
fn is_stale(metadata: &std::fs::Metadata, stale_after: Duration) -> bool {
    age_of(metadata).is_some_and(|age| age >= stale_after)
}

/// Interval at which a guard refreshes its stamp, or `None` when the stale window is too short.
fn heartbeat_interval(stale_after: Duration) -> Option<Duration> {
    let interval = stale_after / 3;
    (interval >= MIN_HEARTBEAT_INTERVAL).then_some(interval)
}

/// Keeps the lock file's modification time fresh while `ownership` still names its holder.
///
/// The task reads the file before every refresh, so it never touches a file that was reclaimed and
/// re-taken by someone else; it stops as soon as the guard releases or dies.
fn spawn_heartbeat(
    path: PathBuf,
    ownership: Arc<Ownership>,
    stale_after: Duration,
) -> Option<JoinHandle<()>> {
    let interval = heartbeat_interval(stale_after)?;
    Some(tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        // The first tick fires immediately; the stamp is fresh, so skip it.
        ticker.tick().await;
        loop {
            ticker.tick().await;
            if ownership.stopped.load(Ordering::SeqCst) {
                return;
            }
            match std::fs::read(&path) {
                Ok(content) => {
                    let content = String::from_utf8_lossy(&content);
                    let ours = content
                        .lines()
                        .any(|line| line.trim() == format!("{TOKEN_PREFIX}{}", ownership.token));
                    if !ours {
                        debug!(
                            lock = %path.display(),
                            "worktree lock changed hands; stopping the heartbeat"
                        );
                        return;
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => return,
                Err(error) => {
                    debug!(
                        lock = %path.display(),
                        %error,
                        "worktree lock cannot be read; stopping the heartbeat"
                    );
                    return;
                }
            }
            if let Err(error) = refresh_stamp(&path) {
                debug!(
                    lock = %path.display(),
                    %error,
                    "failed to refresh the worktree lock stamp"
                );
            }
        }
    }))
}

/// Refreshes the modification time of `path` without changing its content.
fn refresh_stamp(path: &Path) -> std::io::Result<()> {
    let file = std::fs::OpenOptions::new().write(true).open(path)?;
    file.set_modified(SystemTime::now())
}
