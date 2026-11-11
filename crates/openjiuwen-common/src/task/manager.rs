//! The structured task manager: registration, cancellation, and shutdown.
//!
//! Every task spawned through [`TaskManager`] is registered in the manager's
//! registry before its future starts, carries an abort handle and a
//! [`tokio_util::sync::CancellationToken`], and publishes a [`TaskInfo`] snapshot
//! on every status transition. That is the tracked-task guarantee: there are no
//! untracked detached tasks, and dropping the manager aborts whatever is still
//! registered.
//!
//! The manager is a thin `Arc` handle and is cheap to clone; a single instance is
//! shared by all clones. [`TaskManager::global()`] exposes the documented
//! process-global singleton (Python `get_task_manager()` parity); tests and
//! applications that need isolation use [`TaskManager::new`].
#![allow(clippy::module_name_repetitions)]

use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::pin::pin;
use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use futures::FutureExt;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use tokio::sync::{broadcast, Notify};
use tokio::task::{AbortHandle, JoinHandle};
use tokio_util::sync::CancellationToken;

use crate::error::{Error, Result};
use crate::task::status::{TaskId, TaskInfo, TaskOptions, TaskStatus};

/// Capacity of the status-event broadcast channel.
const EVENT_BUFFER: usize = 256;

/// Default cancellation reason, matching Python `Task.cancel`.
const DEFAULT_CANCEL_REASON: &str = "manual_cancel";

tokio::task_local! {
    /// Id of the task whose future is currently being polled (Python
    /// `_current_task_id` context variable).
    static CURRENT_TASK_ID: TaskId;
}

/// Error type used to surface an isolated panic through [`TaskHandle::join`].
#[derive(Debug)]
struct TaskPanicError(String);

impl fmt::Display for TaskPanicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "task panicked: {}", self.0)
    }
}

impl std::error::Error for TaskPanicError {}

/// Converts a `catch_unwind` payload into a human-readable message.
fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        return (*message).to_string();
    }
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }
    "task panicked".to_string()
}

/// Outcome of polling the user future, before it is mapped to a status.
enum TaskOutcome<T> {
    /// The future returned a value.
    Completed(T),
    /// The future panicked with the given message.
    Panicked(String),
    /// The future exceeded its timeout.
    TimedOut(Duration),
}

/// Shared mutable state for one tracked task.
struct TaskEntry {
    info: Mutex<TaskInfo>,
    cancellation: CancellationToken,
    notify: Notify,
    abort: Mutex<Option<AbortHandle>>,
    children: Mutex<HashSet<TaskId>>,
    events: broadcast::Sender<TaskInfo>,
}

impl TaskEntry {
    fn new(info: TaskInfo, events: broadcast::Sender<TaskInfo>) -> Self {
        Self {
            info: Mutex::new(info),
            cancellation: CancellationToken::new(),
            notify: Notify::new(),
            abort: Mutex::new(None),
            children: Mutex::new(HashSet::new()),
            events,
        }
    }

    fn id(&self) -> TaskId {
        self.info.lock().id.clone()
    }

    fn snapshot(&self) -> TaskInfo {
        self.info.lock().clone()
    }

    fn is_terminal(&self) -> bool {
        self.info.lock().status.is_terminal()
    }

    fn set_abort(&self, handle: AbortHandle) {
        *self.abort.lock() = Some(handle);
    }

    /// Hard-aborts the backing tokio task; the future is dropped at its next poll.
    fn abort(&self) {
        if let Some(handle) = self.abort.lock().as_ref() {
            handle.abort();
        }
    }

    fn add_child(&self, child: TaskId) {
        self.children.lock().insert(child);
    }

    fn children(&self) -> Vec<TaskId> {
        self.children.lock().iter().cloned().collect()
    }

    /// Moves a pending task to `Running` and records the start timestamp.
    fn mark_running(&self) -> bool {
        let snapshot = {
            let mut info = self.info.lock();
            if info.status != TaskStatus::Pending {
                return false;
            }
            info.status = TaskStatus::Running;
            info.started_at = Some(Utc::now());
            info.clone()
        };
        self.publish(snapshot);
        true
    }

    /// Records a terminal status; a task that is already terminal never changes.
    fn finish(
        &self,
        status: TaskStatus,
        error: Option<String>,
        reason: Option<String>,
        cancelled_by: Option<String>,
    ) -> bool {
        let snapshot = {
            let mut info = self.info.lock();
            if info.status.is_terminal() {
                return false;
            }
            info.status = status;
            info.finished_at = Some(Utc::now());
            if error.is_some() {
                info.error = error;
            }
            if reason.is_some() {
                info.cancel_reason = reason;
            }
            if cancelled_by.is_some() {
                info.cancelled_by = cancelled_by;
            }
            info.clone()
        };
        self.publish(snapshot);
        true
    }

    fn mark_completed(&self) -> bool {
        self.finish(TaskStatus::Completed, None, None, None)
    }

    fn mark_failed(&self, message: impl Into<String>) -> bool {
        self.finish(TaskStatus::Failed, Some(message.into()), None, None)
    }

    fn mark_timeout(&self, message: impl Into<String>) -> bool {
        self.finish(TaskStatus::Timeout, Some(message.into()), None, None)
    }

    fn mark_cancelled(&self, reason: Option<String>, cancelled_by: Option<String>) -> bool {
        self.finish(
            TaskStatus::Cancelled,
            None,
            Some(reason.unwrap_or_else(|| DEFAULT_CANCEL_REASON.to_string())),
            cancelled_by,
        )
    }

    /// Cancels the token and records a cancellation; returns `false` when the
    /// task is already terminal.
    fn cancel(&self, reason: impl Into<String>, cancelled_by: Option<String>) -> bool {
        if self.is_terminal() {
            return false;
        }
        self.cancellation.cancel();
        self.finish(
            TaskStatus::Cancelled,
            None,
            Some(reason.into()),
            cancelled_by,
        )
    }

    /// Completes once the task reaches a terminal status.
    async fn wait_terminal(&self) {
        loop {
            let mut notified = pin!(self.notify.notified());
            notified.as_mut().enable();
            if self.is_terminal() {
                return;
            }
            notified.await;
        }
    }

    fn publish(&self, snapshot: TaskInfo) {
        self.notify.notify_waiters();
        // No subscribers is the common case and not an error.
        let _ = self.events.send(snapshot);
    }
}

/// Shared manager state.
struct ManagerInner {
    tasks: Mutex<HashMap<TaskId, Arc<TaskEntry>>>,
    events: broadcast::Sender<TaskInfo>,
    shutdown: Notify,
}

impl Drop for ManagerInner {
    fn drop(&mut self) {
        let tasks = self.tasks.get_mut();
        for entry in tasks.values() {
            entry.cancellation.cancel();
            entry.abort();
        }
    }
}

/// Structured async task manager.
///
/// Cloning a [`TaskManager`] shares the same registry and event stream.
#[derive(Clone)]
pub struct TaskManager {
    inner: Arc<ManagerInner>,
}

impl TaskManager {
    /// Creates an isolated manager.
    ///
    /// Tests should use this instead of [`TaskManager::global`] so they do not
    /// share state.
    #[must_use]
    pub fn new() -> Self {
        let (events, _receiver) = broadcast::channel(EVENT_BUFFER);
        Self {
            inner: Arc::new(ManagerInner {
                tasks: Mutex::new(HashMap::new()),
                events,
                shutdown: Notify::new(),
            }),
        }
    }

    /// Returns the documented process-global manager (Python singleton parity).
    #[allow(clippy::non_std_lazy_statics)]
    #[must_use]
    pub fn global() -> Self {
        static GLOBAL: Lazy<TaskManager> = Lazy::new(TaskManager::new);
        GLOBAL.clone()
    }

    /// Spawns a named task and returns a handle to it.
    #[must_use]
    pub fn spawn<F>(&self, name: impl Into<String>, future: F) -> TaskHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.spawn_with(TaskOptions::new(name), future)
    }

    /// Spawns a named task inside a cancellation group.
    #[must_use]
    pub fn spawn_in_group<F>(
        &self,
        group: impl Into<String>,
        name: impl Into<String>,
        future: F,
    ) -> TaskHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.spawn_with(TaskOptions::new(name).with_group(group), future)
    }

    /// Spawns a task with full options.
    ///
    /// The task is registered before it is spawned, so it is visible to
    /// [`TaskManager::list`] as soon as this returns. A task spawned from inside
    /// another tracked task records that task as its `parent_id` and becomes
    /// eligible for [`TaskManager::cascade_cancel`].
    #[must_use]
    pub fn spawn_with<F>(&self, options: TaskOptions, future: F) -> TaskHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        let TaskOptions {
            name,
            group,
            timeout,
            metadata,
        } = options;
        let parent_id = CURRENT_TASK_ID.try_with(TaskId::clone).ok();
        let info = TaskInfo {
            id: TaskId::generate(),
            name,
            group,
            status: TaskStatus::Pending,
            parent_id,
            created_at: Utc::now(),
            started_at: None,
            finished_at: None,
            timeout,
            error: None,
            cancel_reason: None,
            cancelled_by: None,
            metadata,
        };
        let id = info.id.clone();
        let handle_name = info.name.clone();
        let handle_group = info.group.clone();
        let entry = Arc::new(TaskEntry::new(info.clone(), self.inner.events.clone()));
        {
            // Register before spawning: no task ever runs untracked.
            let mut tasks = self.inner.tasks.lock();
            tasks.insert(id.clone(), Arc::clone(&entry));
            if let Some(parent_id) = &info.parent_id {
                if let Some(parent) = tasks.get(parent_id) {
                    parent.add_child(id.clone());
                }
            }
        }
        let task_entry = Arc::clone(&entry);
        let join = tokio::spawn(async move { run_task(task_entry, future, timeout).await });
        entry.set_abort(join.abort_handle());
        TaskHandle {
            id,
            name: handle_name,
            group: handle_group,
            entry,
            join,
        }
    }

    /// Returns a snapshot of every tracked task, oldest first.
    #[must_use]
    pub fn list(&self) -> Vec<TaskInfo> {
        let mut tasks: Vec<TaskInfo> = {
            self.inner
                .tasks
                .lock()
                .values()
                .map(|entry| entry.snapshot())
                .collect()
        };
        tasks.sort_by(|left, right| {
            left.created_at
                .cmp(&right.created_at)
                .then_with(|| left.id.cmp(&right.id))
        });
        tasks
    }

    /// Returns the snapshot of one task, if it is still tracked.
    #[must_use]
    pub fn get(&self, id: &TaskId) -> Option<TaskInfo> {
        self.inner
            .tasks
            .lock()
            .get(id)
            .map(|entry| entry.snapshot())
    }

    /// Returns the number of tracked tasks, including terminal ones.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.tasks.lock().len()
    }

    /// Returns `true` when no task is tracked.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.tasks.lock().is_empty()
    }

    /// Cancels one task; returns `false` when it is unknown or already terminal.
    pub fn cancel(&self, id: &TaskId, reason: impl Into<String>) -> bool {
        let entry = self.entry(id);
        match entry {
            Some(entry) => entry.cancel(reason, None),
            None => false,
        }
    }

    /// Cancels a task and, recursively, every task spawned underneath it.
    ///
    /// Returns the number of tasks that transitioned to `Cancelled`.
    #[must_use]
    pub fn cascade_cancel(&self, id: &TaskId, reason: impl Into<String>) -> usize {
        let reason = reason.into();
        let Some(root) = self.entry(id) else {
            return 0;
        };
        let mut cancelled = 0;
        let mut seen: HashSet<TaskId> = HashSet::new();
        let mut stack: Vec<(TaskId, Option<TaskId>)> = vec![(id.clone(), None)];
        while let Some((current, parent)) = stack.pop() {
            if !seen.insert(current.clone()) {
                continue;
            }
            let entry = if current == *id {
                Some(Arc::clone(&root))
            } else {
                self.entry(&current)
            };
            let Some(entry) = entry else {
                continue;
            };
            if entry.cancel(reason.clone(), parent.map(|parent| parent.to_string())) {
                cancelled += 1;
            }
            for child in entry.children() {
                stack.push((child, Some(current.clone())));
            }
        }
        cancelled
    }

    /// Cancels every non-terminal task in a group.
    ///
    /// Returns the number of tasks that transitioned to `Cancelled`.
    #[must_use]
    pub fn cancel_group(&self, group: &str, reason: impl Into<String>) -> usize {
        let reason = reason.into();
        let entries = self.entries();
        let mut cancelled = 0;
        for entry in entries {
            if entry.snapshot().group.as_deref() == Some(group)
                && entry.cancel(reason.clone(), None)
            {
                cancelled += 1;
            }
        }
        cancelled
    }

    /// Cancels every non-terminal task.
    ///
    /// Returns the number of tasks that transitioned to `Cancelled`.
    #[must_use]
    pub fn cancel_all(&self, reason: impl Into<String>) -> usize {
        let reason = reason.into();
        self.entries()
            .into_iter()
            .filter(|entry| entry.cancel(reason.clone(), None))
            .count()
    }

    /// Waits for every currently tracked task to become terminal.
    ///
    /// Tasks spawned after this call starts are not awaited.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::ErrorKind::Timeout`] when `timeout` expires first.
    pub async fn wait_all(&self, timeout: Option<Duration>) -> Result<()> {
        let entries = self.entries();
        let wait = async {
            for entry in entries {
                entry.wait_terminal().await;
            }
        };
        if let Some(timeout) = timeout {
            tokio::time::timeout(timeout, wait)
                .await
                .map_err(|_| Error::timeout("wait_all timed out"))
        } else {
            wait.await;
            Ok(())
        }
    }

    /// Removes terminal tasks from the registry and returns how many were removed.
    #[must_use]
    pub fn remove_finished(&self) -> usize {
        let mut tasks = self.inner.tasks.lock();
        let before = tasks.len();
        tasks.retain(|_, entry| !entry.is_terminal());
        before - tasks.len()
    }

    /// Cancels and aborts every tracked task, then clears the registry.
    ///
    /// Waits briefly (at most two seconds) for cancellations to take effect and
    /// never hangs, even when a task ignores its cancellation token.
    pub async fn shutdown(&self) {
        let _ = self.cancel_all("manager_shutdown");
        let entries = {
            let mut tasks = self.inner.tasks.lock();
            std::mem::take(&mut *tasks)
                .into_values()
                .collect::<Vec<_>>()
        };
        for entry in &entries {
            entry.abort();
        }
        self.inner.shutdown.notify_waiters();
        let _ = tokio::time::timeout(Duration::from_secs(2), async {
            for entry in &entries {
                entry.wait_terminal().await;
            }
        })
        .await;
    }

    /// Subscribes to status transitions of every tracked task.
    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<TaskInfo> {
        self.inner.events.subscribe()
    }

    fn entry(&self, id: &TaskId) -> Option<Arc<TaskEntry>> {
        self.inner.tasks.lock().get(id).cloned()
    }

    fn entries(&self) -> Vec<Arc<TaskEntry>> {
        self.inner.tasks.lock().values().cloned().collect()
    }
}

impl Default for TaskManager {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for TaskManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TaskManager")
            .field("tasks", &self.len())
            .finish_non_exhaustive()
    }
}

/// Polls the user future with timeout and panic isolation, then records the
/// resulting status.
async fn run_task<F>(
    entry: Arc<TaskEntry>,
    future: F,
    timeout: Option<Duration>,
) -> Result<F::Output>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    if !entry.mark_running() {
        // Cancelled before the first poll: never run the user future.
        return Err(Error::cancelled());
    }
    let task_id = entry.id();
    let cancellation = entry.cancellation.clone();
    let outcome = CURRENT_TASK_ID
        .scope(task_id, async move {
            let poll = async {
                match AssertUnwindSafe(future).catch_unwind().await {
                    Ok(value) => TaskOutcome::Completed(value),
                    Err(payload) => TaskOutcome::Panicked(panic_message(payload.as_ref())),
                }
            };
            match timeout {
                Some(duration) => match tokio::time::timeout(duration, poll).await {
                    Ok(outcome) => outcome,
                    Err(_) => TaskOutcome::TimedOut(duration),
                },
                None => poll.await,
            }
        })
        .await;
    if cancellation.is_cancelled() {
        entry.mark_cancelled(None, None);
        return Err(Error::cancelled());
    }
    match outcome {
        TaskOutcome::Completed(value) => {
            entry.mark_completed();
            Ok(value)
        }
        TaskOutcome::Panicked(message) => {
            entry.mark_failed(message.clone());
            Err(Error::external(TaskPanicError(message)))
        }
        TaskOutcome::TimedOut(duration) => {
            let message = format!("task timed out after {duration:?}");
            entry.mark_timeout(message.clone());
            Err(Error::timeout(message))
        }
    }
}

/// Handle to a task spawned through [`TaskManager`].
///
/// Dropping the handle leaves the task running and tracked; use
/// [`TaskHandle::join`] to await it, [`TaskHandle::cancel`] to request
/// cooperative cancellation, or [`TaskHandle::detach`] for a tracked detach.
pub struct TaskHandle<T> {
    id: TaskId,
    name: String,
    group: Option<String>,
    entry: Arc<TaskEntry>,
    join: JoinHandle<Result<T>>,
}

impl<T> TaskHandle<T> {
    /// Returns the task id.
    #[must_use]
    pub fn id(&self) -> &TaskId {
        &self.id
    }

    /// Returns the task name (may be empty).
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the task group, if one was configured.
    #[must_use]
    pub fn group(&self) -> Option<&str> {
        self.group.as_deref()
    }

    /// Returns the current status.
    #[must_use]
    pub fn status(&self) -> TaskStatus {
        self.entry.snapshot().status
    }

    /// Returns a snapshot of the task.
    #[must_use]
    pub fn info(&self) -> TaskInfo {
        self.entry.snapshot()
    }

    /// Returns the cancellation token the task can cooperatively observe.
    #[must_use]
    pub fn cancellation(&self) -> CancellationToken {
        self.entry.cancellation.clone()
    }

    /// Requests cooperative cancellation; returns `false` when already terminal.
    pub fn cancel(&self, reason: impl Into<String>) -> bool {
        self.entry.cancel(reason, None)
    }

    /// Hard-aborts the task and records it as cancelled with reason `"aborted"`.
    pub fn abort(&self) {
        self.join.abort();
        self.entry.cancel("aborted", None);
    }

    /// Awaits the task and returns its output.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::ErrorKind::Cancelled`] when the task was
    /// cancelled, [`crate::error::ErrorKind::Timeout`] when it timed out, and an
    /// external error when the task panicked or the runtime aborted it. When the
    /// task output is itself a `Result`, its error is returned unchanged.
    pub async fn join(self) -> Result<T> {
        let Self { join, .. } = self;
        match join.await {
            Ok(result) => result,
            Err(error) if error.is_cancelled() => Err(Error::cancelled()),
            Err(error) => Err(Error::external(error)),
        }
    }

    /// Drops the join handle while the manager keeps tracking the task.
    pub fn detach(self) {
        let Self { join, .. } = self;
        drop(join);
    }
}

impl<T> fmt::Debug for TaskHandle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TaskHandle")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("group", &self.group)
            .field("status", &self.status())
            .finish_non_exhaustive()
    }
}

// port of tests/unit_tests/core/common/test_task_manager.py
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::error::ErrorKind;

    #[tokio::test]
    async fn status_moves_from_running_to_completed() {
        let manager = TaskManager::new();
        let handle = manager.spawn("worker", async { 7 });
        let id = handle.id().clone();
        assert_eq!(handle.join().await.unwrap(), 7);
        let info = manager.get(&id).unwrap();
        assert_eq!(info.status, TaskStatus::Completed);
        assert!(info.started_at.is_some());
        assert!(info.finished_at.is_some());
        assert!(info.duration_ms().is_some());
        assert_eq!(manager.len(), 1);
    }

    #[tokio::test]
    async fn timeout_marks_timeout_and_join_reports_it() {
        let manager = TaskManager::new();
        let options = TaskOptions::new("slow").with_timeout(Duration::from_millis(10));
        let handle = manager.spawn_with(options, async {
            tokio::time::sleep(Duration::from_secs(5)).await;
            1
        });
        let id = handle.id().clone();
        let error = handle.join().await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Timeout);
        let info = manager.get(&id).unwrap();
        assert_eq!(info.status, TaskStatus::Timeout);
        assert!(info.error.unwrap().contains("timed out"));
    }

    #[tokio::test]
    async fn panic_is_isolated_and_siblings_complete() {
        let manager = TaskManager::new();
        let panicked = manager.spawn("boom", async { panic!("boom") });
        let sibling = manager.spawn("ok", async { 3 });
        let error = panicked.join().await.unwrap_err();
        assert!(error.to_string().contains("boom"));
        assert_eq!(sibling.join().await.unwrap(), 3);
        let list = manager.list();
        assert_eq!(list.len(), 2);
        assert!(list.iter().any(|info| info.status == TaskStatus::Failed));
        assert!(list.iter().any(|info| info.status == TaskStatus::Completed));
    }

    #[tokio::test]
    async fn cooperative_cancellation_observes_the_token() {
        let manager = TaskManager::new();
        let (tx, rx) = tokio::sync::oneshot::channel::<CancellationToken>();
        let handle = manager.spawn("coop", async move {
            let token = rx.await.unwrap();
            token.cancelled().await;
            0
        });
        let id = handle.id().clone();
        tx.send(handle.cancellation()).unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(manager.cancel(&id, "test_cancel"));
        let error = handle.join().await.unwrap_err();
        assert!(error.is_cancelled());
        let info = manager.get(&id).unwrap();
        assert_eq!(info.status, TaskStatus::Cancelled);
        assert_eq!(info.cancel_reason.as_deref(), Some("test_cancel"));
    }

    #[tokio::test]
    async fn abort_stops_a_long_running_task() {
        let manager = TaskManager::new();
        let handle = manager.spawn("sleepy", async {
            tokio::time::sleep(Duration::from_secs(30)).await;
        });
        let id = handle.id().clone();
        handle.abort();
        let error = handle.join().await.unwrap_err();
        assert!(error.is_cancelled());
        assert_eq!(manager.get(&id).unwrap().status, TaskStatus::Cancelled);
    }

    #[tokio::test]
    async fn group_cancellation_counts_and_is_idempotent() {
        let manager = TaskManager::new();
        for index in 0..3 {
            manager
                .spawn_in_group("workers", format!("worker-{index}"), async {
                    tokio::time::sleep(Duration::from_secs(30)).await;
                })
                .detach();
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert_eq!(manager.cancel_group("workers", "stop"), 3);
        assert_eq!(manager.cancel_group("workers", "stop"), 0);
        assert!(manager
            .list()
            .iter()
            .all(|info| info.status == TaskStatus::Cancelled));
    }

    #[tokio::test]
    async fn cascade_cancel_cancels_descendants() {
        let manager = TaskManager::new();
        let manager_for_child = manager.clone();
        let parent = manager.spawn("parent", async move {
            manager_for_child
                .spawn("child", async {
                    tokio::time::sleep(Duration::from_secs(30)).await;
                })
                .detach();
            tokio::time::sleep(Duration::from_secs(30)).await;
        });
        let parent_id = parent.id().clone();
        loop {
            let child = manager
                .list()
                .into_iter()
                .find(|info| info.parent_id.as_ref() == Some(&parent_id));
            if let Some(child) = child {
                assert_eq!(child.name, "child");
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(manager.cascade_cancel(&parent_id, "parent_cancelled"), 2);
        assert!(manager
            .list()
            .iter()
            .all(|info| info.status == TaskStatus::Cancelled));
        // Cancellation is cooperative, so stop the sleeping parent explicitly
        // before awaiting it; the status is already terminal.
        parent.abort();
        assert!(parent.join().await.unwrap_err().is_cancelled());
    }

    #[tokio::test]
    async fn remove_finished_prunes_terminal_tasks() {
        let manager = TaskManager::new();
        let handle = manager.spawn("quick", async {});
        let id = handle.id().clone();
        handle.join().await.unwrap();
        manager
            .spawn("slow", async {
                tokio::time::sleep(Duration::from_secs(30)).await;
            })
            .detach();
        assert_eq!(manager.remove_finished(), 1);
        assert!(manager.get(&id).is_none());
        assert_eq!(manager.len(), 1);
        let _ = manager.cancel_all("cleanup");
    }

    #[tokio::test]
    async fn wait_all_finishes_and_times_out() {
        let manager = TaskManager::new();
        for _ in 0..4 {
            manager
                .spawn("quick", async {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                })
                .detach();
        }
        manager
            .wait_all(Some(Duration::from_secs(5)))
            .await
            .unwrap();
        assert!(manager
            .list()
            .iter()
            .all(|info| info.status == TaskStatus::Completed));
        let _ = manager.remove_finished();

        manager
            .spawn("slow", async {
                tokio::time::sleep(Duration::from_secs(5)).await;
            })
            .detach();
        let error = manager
            .wait_all(Some(Duration::from_millis(10)))
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Timeout);
        let _ = manager.cancel_all("cleanup");
    }

    #[tokio::test]
    async fn subscribe_receives_terminal_snapshots() {
        let manager = TaskManager::new();
        let mut receiver = manager.subscribe();
        let handle = manager.spawn("watched", async {});
        handle.join().await.unwrap();
        let info = tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if let Ok(info) = receiver.recv().await {
                    if info.status == TaskStatus::Completed {
                        return info;
                    }
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(info.name, "watched");
    }

    #[tokio::test]
    async fn detach_keeps_the_task_tracked() {
        let manager = TaskManager::new();
        let handle = manager.spawn("detached", async {
            tokio::time::sleep(Duration::from_millis(20)).await;
            1
        });
        let id = handle.id().clone();
        handle.detach();
        assert!(manager.get(&id).is_some());
        tokio::time::sleep(Duration::from_millis(80)).await;
        let info = manager.get(&id).unwrap();
        assert_eq!(info.status, TaskStatus::Completed);
    }

    #[tokio::test]
    async fn global_manager_is_shared_between_handles() {
        let first = TaskManager::global();
        let second = TaskManager::global();
        let handle = first.spawn("global-task", async {});
        let id = handle.id().clone();
        handle.join().await.unwrap();
        assert_eq!(
            second.get(&id).map(|info| info.status),
            Some(TaskStatus::Completed)
        );
    }

    #[tokio::test]
    async fn shutdown_clears_the_registry_and_stops_tasks() {
        let manager = TaskManager::new();
        manager
            .spawn("stubborn", async {
                tokio::time::sleep(Duration::from_secs(30)).await;
            })
            .detach();
        manager.shutdown().await;
        assert!(manager.is_empty());
    }
}
