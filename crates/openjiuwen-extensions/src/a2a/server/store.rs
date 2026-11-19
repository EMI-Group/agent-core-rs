//! The task-store SPI and its bounded, process-local in-memory implementation.

use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use crate::a2a::types::Task;

/// Default cap on retained tasks ([`RetentionPolicy::max_tasks`]).
///
/// A long-lived server would otherwise leak one stored [`Task`] per request.
pub const DEFAULT_MAX_TASKS: usize = 10_000;

/// Default retention of a terminal task, in milliseconds
/// ([`RetentionPolicy::terminal_ttl`] when used as `Some(..)`).
pub const DEFAULT_TERMINAL_TASK_TTL_MS: u64 = 60 * 60 * 1_000;

/// The stored tasks of one server, keyed by task id.
///
/// Python parity: `a2a.server.tasks.inmemory_task_store.InMemoryTaskStore`'s
/// interface. The server reads through this trait in `GetTask`/`CancelTask` and
/// the [`super::EventSink`] writes through it, so a deployment can swap in a
/// durable store without touching the protocol layer.
pub trait TaskStore: Send + Sync {
    /// The task with this id, if it is known.
    fn get(&self, id: &str) -> Option<Task>;

    /// Inserts or replaces a task; the task id is the key.
    fn put(&self, task: Task);

    /// Every known task.
    fn all(&self) -> Vec<Task>;
}

/// Bounds an [`InMemoryTaskStore`] applies on every write and read.
///
/// The policy is deliberately soft: a task that is not in a terminal state
/// (`WORKING`, `SUBMITTED`, `INPUT_REQUIRED`, `AUTH_REQUIRED`) is **never**
/// evicted, so a burst of concurrent running tasks can exceed
/// [`Self::max_tasks`] until those tasks finish. Awaited tasks
/// (`INPUT_REQUIRED`) therefore stay retrievable no matter how much traffic the
/// server takes in the meantime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetentionPolicy {
    /// Maximum number of retained tasks; `0` disables the cap.
    ///
    /// When the store is over the cap, terminal tasks are evicted oldest
    /// first (by the moment they became terminal).
    pub max_tasks: usize,
    /// How long a terminal task stays retrievable after it reached its terminal
    /// state; `None` keeps terminal tasks forever.
    ///
    /// `REJECTED`, `COMPLETED`, `FAILED` and `CANCELED` are treated as
    /// terminal; everything else is retained.
    pub terminal_ttl: Option<Duration>,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            max_tasks: DEFAULT_MAX_TASKS,
            terminal_ttl: Some(Duration::from_millis(DEFAULT_TERMINAL_TASK_TTL_MS)),
        }
    }
}

impl RetentionPolicy {
    /// A policy with an explicit cap and terminal-task TTL.
    ///
    /// `max_tasks = 0` disables the cap and `terminal_ttl = None` disables
    /// expiry.
    #[must_use]
    pub const fn new(max_tasks: usize, terminal_ttl: Option<Duration>) -> Self {
        Self {
            max_tasks,
            terminal_ttl,
        }
    }
}

/// One stored task plus the moment it became terminal, when it has.
#[derive(Debug, Clone)]
struct Entry {
    /// The task snapshot.
    task: Task,
    /// When the task first reached a terminal state; `None` while it is still
    /// running or awaited.
    terminal_at: Option<Instant>,
}

impl Entry {
    /// Whether this entry can be evicted (it is terminal).
    fn is_evictable(&self) -> bool {
        self.terminal_at.is_some()
    }
}

/// A process-local [`TaskStore`] with a bounded retention policy.
///
/// Tasks are kept in a [`BTreeMap`], so [`TaskStore::all`] returns them in
/// ascending id order. Python's dict yields insertion order instead; the ids are
/// opaque, so the sorted order is a documented, deterministic substitute.
///
/// Retention (see [`RetentionPolicy`]) runs on every [`TaskStore::put`] and
/// lazily on [`TaskStore::get`]/[`TaskStore::all`]: expired terminal tasks are
/// dropped, then terminal tasks are evicted oldest-first until the store fits
/// the cap. A running or awaited task is never evicted, so it is always
/// retrievable while the executor still drives it.
#[derive(Debug)]
pub struct InMemoryTaskStore {
    /// Retained tasks.
    tasks: Mutex<BTreeMap<String, Entry>>,
    /// The bounds applied on every access.
    policy: RetentionPolicy,
}

impl Default for InMemoryTaskStore {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryTaskStore {
    /// An empty store with [`RetentionPolicy::default`].
    #[must_use]
    pub fn new() -> Self {
        Self::with_policy(RetentionPolicy::default())
    }

    /// An empty store with an explicit retention policy.
    #[must_use]
    pub fn with_policy(policy: RetentionPolicy) -> Self {
        Self {
            tasks: Mutex::new(BTreeMap::new()),
            policy,
        }
    }

    /// The retention policy this store applies.
    #[must_use]
    pub fn policy(&self) -> RetentionPolicy {
        self.policy
    }

    /// Locks the task map, recovering the data when a previous holder panicked.
    fn lock(&self) -> MutexGuard<'_, BTreeMap<String, Entry>> {
        self.tasks.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Drops expired terminal tasks, then evicts terminal tasks until the cap
    /// holds. Running and awaited tasks are never touched.
    fn prune(&self, tasks: &mut BTreeMap<String, Entry>, now: Instant) {
        if let Some(ttl) = self.policy.terminal_ttl {
            tasks.retain(|_, entry| match entry.terminal_at {
                Some(at) => now.saturating_duration_since(at) < ttl,
                None => true,
            });
        }
        if self.policy.max_tasks == 0 {
            return;
        }
        while tasks.len() > self.policy.max_tasks {
            let oldest_terminal = tasks
                .iter()
                .filter(|(_, entry)| entry.is_evictable())
                .filter_map(|(id, entry)| entry.terminal_at.map(|at| (at, id.clone())))
                .min();
            let Some((_, id)) = oldest_terminal else {
                // Only running or awaited tasks are left: the cap yields to
                // them rather than dropping in-flight work.
                break;
            };
            tasks.remove(&id);
        }
    }
}

impl TaskStore for InMemoryTaskStore {
    fn get(&self, id: &str) -> Option<Task> {
        let now = Instant::now();
        let mut tasks = self.lock();
        self.prune(&mut tasks, now);
        tasks.get(id).map(|entry| entry.task.clone())
    }

    fn put(&self, task: Task) {
        let now = Instant::now();
        let mut tasks = self.lock();
        let terminal_at = if task.status.state.is_terminal() {
            // Keep the first terminal timestamp so repeated updates of the same
            // finished task do not extend its lifetime forever.
            tasks
                .get(&task.id)
                .and_then(|entry| entry.terminal_at)
                .or(Some(now))
        } else {
            None
        };
        tasks.insert(task.id.clone(), Entry { task, terminal_at });
        self.prune(&mut tasks, now);
    }

    fn all(&self) -> Vec<Task> {
        let now = Instant::now();
        let mut tasks = self.lock();
        self.prune(&mut tasks, now);
        tasks.values().map(|entry| entry.task.clone()).collect()
    }
}
