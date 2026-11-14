//! Per-conversation task queue: at most one running task per conversation.
//!
//! Ports `TaskQueue`/`RunningTaskInfo` from Python's
//! `openjiuwen/core/controller/legacy/intent_detection_controller.py`. A
//! controller keeps the in-flight turn of every conversation it is driving, so
//! that a newer turn for the same conversation can cancel the older one before
//! it starts, and so that a host can inspect what is running without touching
//! the controller itself.
//!
//! The queue is keyed by conversation id ([`TaskQueue::register`] replaces an
//! existing entry, matching Python's "a newer turn owns the conversation"), and
//! the tracked handle is deliberately left in place by
//! [`TaskQueue::cancel_running`]: the owner of the turn unregisters it once it
//! has observed the cancellation, so a cancelled conversation still reports
//! [`TaskQueue::has_running`] until then.
//!
//! Timings are monotonic: [`RunningTask::started_at`] is the offset from the
//! process-wide base instant returned by [`process_base_instant`], never a wall
//! clock reading, so two registrations are always comparable.
//!
//! # Differences from Python
//!
//! - Python stores the `asyncio.Task` and guards its map with an `asyncio.Lock`;
//!   here the turn is a `tokio::task::JoinHandle` plus the
//!   [`CancellationToken`] its owner watches, behind a `parking_lot::Mutex`
//!   that is never held across an `await`.
//! - Python's start time is `asyncio.get_event_loop().time()`; here it is the
//!   offset from [`process_base_instant`], which stays monotonic within the
//!   process (and needs no event loop).
//! - Python's queue is drained by the legacy controller's per-message
//!   cancellation; this queue is the standalone seam a controller (or a host)
//!   drives, so it holds no background task of its own.
use super::schema::Task;
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::debug;

/// Returns the process-wide monotonic instant every [`RunningTask`] timing is
/// measured from.
///
/// The base instant is captured once per process, so
/// [`RunningTask::started_at`] offsets stay comparable across queues and never
/// move backwards.
#[must_use]
pub fn process_base_instant() -> Instant {
    static BASE: OnceLock<Instant> = OnceLock::new();
    *BASE.get_or_init(Instant::now)
}

/// One in-flight conversation turn.
///
/// A `RunningTask` snapshots the task it runs, its routing target, its start
/// offset, and the [`CancellationToken`] its owner watches, and it owns the
/// [`JoinHandle`] of the spawned turn. Cloning yields a detachable snapshot
/// (the handle stays with the [`TaskQueue`]) — see
/// [`is_finished`](Self::is_finished).
#[derive(Debug)]
pub struct RunningTask {
    conversation_id: String,
    task: Task,
    target_id: String,
    started_at: Duration,
    token: CancellationToken,
    handle: Option<JoinHandle<()>>,
}

impl RunningTask {
    /// Builds a tracked turn; the start offset is taken from the process base
    /// instant.
    fn new(
        conversation_id: String,
        task: Task,
        handle: JoinHandle<()>,
        token: CancellationToken,
        target_id: String,
    ) -> Self {
        Self {
            conversation_id,
            task,
            target_id,
            started_at: process_base_instant().elapsed(),
            token,
            handle: Some(handle),
        }
    }

    /// Returns the conversation this turn belongs to.
    #[must_use]
    pub fn conversation_id(&self) -> &str {
        &self.conversation_id
    }

    /// Returns the snapshot of the task this turn is running.
    #[must_use]
    pub fn task(&self) -> &Task {
        &self.task
    }

    /// Returns the id of the agent or workflow the turn was routed to.
    #[must_use]
    pub fn target_id(&self) -> &str {
        &self.target_id
    }

    /// Returns the monotonic start offset from [`process_base_instant`].
    #[must_use]
    pub fn started_at(&self) -> Duration {
        self.started_at
    }

    /// Returns a clone of the cancellation token the owner watches.
    #[must_use]
    pub fn token(&self) -> CancellationToken {
        self.token.clone()
    }

    /// Returns whether the turn is no longer running.
    ///
    /// A turn is finished once its handle's task has completed (or was
    /// aborted), and once [`TaskQueue::cancel_running`] has taken the handle to
    /// await it. A clone of a running turn reports `true`: clones are
    /// detachable snapshots and never own the join handle.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.handle.as_ref().is_none_or(JoinHandle::is_finished)
    }
}

impl Clone for RunningTask {
    /// Clones the snapshot, leaving the join handle with the queue.
    fn clone(&self) -> Self {
        Self {
            conversation_id: self.conversation_id.clone(),
            task: self.task.clone(),
            target_id: self.target_id.clone(),
            started_at: self.started_at,
            token: self.token.clone(),
            handle: None,
        }
    }
}

/// Per-conversation registry of in-flight turns.
///
/// The queue is a cheap handle: cloning shares the same registry, so a host can
/// hand a copy to a supervisor task while the controller keeps its own.
#[derive(Debug, Clone, Default)]
pub struct TaskQueue {
    running: Arc<Mutex<BTreeMap<String, RunningTask>>>,
}

impl TaskQueue {
    /// Builds an empty queue.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `handle` as the running turn of `conversation_id`.
    ///
    /// Any turn already registered for the conversation is replaced: a newer
    /// turn owns the conversation. The returned [`RunningTask`] is a snapshot of
    /// the registered turn (without the join handle, which stays with the
    /// queue).
    pub fn register(
        &self,
        conversation_id: impl Into<String>,
        task: Task,
        handle: JoinHandle<()>,
        token: CancellationToken,
        target_id: impl Into<String>,
    ) -> RunningTask {
        let conversation_id = conversation_id.into();
        let target_id = target_id.into();
        let entry = RunningTask::new(
            conversation_id.clone(),
            task,
            handle,
            token,
            target_id.clone(),
        );
        let snapshot = entry.clone();
        let replaced = self.running.lock().insert(conversation_id.clone(), entry);
        if replaced.is_some() {
            debug!(
                conversation_id,
                target_id, "TaskQueue: replaced running task"
            );
        } else {
            debug!(
                conversation_id,
                target_id, "TaskQueue: registered running task"
            );
        }
        snapshot
    }

    /// Cancels the running turn of `conversation_id` and waits for it to stop.
    ///
    /// The tracked [`CancellationToken`] is cancelled first, then the tracked
    /// join handle is awaited. The entry itself is left in the queue so its
    /// owner can [`unregister`](Self::unregister) it once it has observed the
    /// cancellation; a second call for the same conversation therefore reports
    /// `false`.
    ///
    /// Returns `true` when a live turn was cancelled (the task ended, or the
    /// join reported a cancellation), and `false` when the conversation has no
    /// turn, its turn already finished, or the turn's join failed for another
    /// reason (logged at debug level).
    pub async fn cancel_running(&self, conversation_id: &str) -> bool {
        // Never hold the lock across the await: copy out the token and take the
        // handle while the lock is held, then await it unlocked.
        let prepared = {
            let mut running = self.running.lock();
            let Some(entry) = running.get_mut(conversation_id) else {
                return false;
            };
            if entry.is_finished() {
                return false;
            }
            (
                entry.token.clone(),
                entry.target_id.clone(),
                entry.handle.take(),
            )
        };
        let (token, target_id, Some(handle)) = prepared else {
            return false;
        };
        debug!(
            conversation_id,
            target_id, "TaskQueue: cancelling running task"
        );
        token.cancel();
        match handle.await {
            Ok(()) => true,
            Err(error) if error.is_cancelled() => true,
            Err(error) => {
                debug!(
                    conversation_id,
                    target_id,
                    error = %error,
                    "TaskQueue: cancel error"
                );
                false
            }
        }
    }

    /// Forgets the turn of `conversation_id`, returning it when it was tracked.
    ///
    /// Dropping the returned value drops the join handle, which detaches (never
    /// aborts) a still-running task.
    pub fn unregister(&self, conversation_id: &str) -> Option<RunningTask> {
        let removed = self.running.lock().remove(conversation_id);
        if removed.is_some() {
            debug!(conversation_id, "TaskQueue: unregistered running task");
        }
        removed
    }

    /// Returns a snapshot of the turn of `conversation_id`, when tracked.
    #[must_use]
    pub fn find(&self, conversation_id: &str) -> Option<RunningTask> {
        self.running.lock().get(conversation_id).cloned()
    }

    /// Returns whether a turn is tracked for `conversation_id`.
    ///
    /// A tracked turn may already have finished or been cancelled; the entry
    /// stays until its owner unregisters it.
    #[must_use]
    pub fn has_running(&self, conversation_id: &str) -> bool {
        self.running.lock().contains_key(conversation_id)
    }

    /// Returns the conversation ids with a tracked turn, in sorted order.
    #[must_use]
    pub fn running_conversations(&self) -> Vec<String> {
        self.running.lock().keys().cloned().collect()
    }

    /// Returns the number of tracked conversations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.running.lock().len()
    }

    /// Returns whether no conversation is tracked.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.running.lock().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration as StdDuration;

    /// Upper bound for the awaits of a test, so a stuck task fails instead of
    /// hanging the suite.
    const BOUND: StdDuration = StdDuration::from_secs(5);

    fn task(id: &str) -> Task {
        Task::submitted("s1", id, "agent", "do it")
    }

    /// Spawns a turn that loops until its token is cancelled.
    fn looping_task(token: &CancellationToken) -> (JoinHandle<()>, Arc<AtomicUsize>) {
        let ticks = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&ticks);
        let watched = token.clone();
        let handle = tokio::spawn(async move {
            loop {
                tokio::select! {
                    () = watched.cancelled() => break,
                    () = tokio::time::sleep(StdDuration::from_millis(5)) => {
                        counter.fetch_add(1, Ordering::SeqCst);
                    }
                }
            }
        });
        (handle, ticks)
    }

    #[tokio::test]
    async fn register_find_has_running_len_and_unregister() {
        let queue = TaskQueue::new();
        assert!(queue.is_empty());
        let token = CancellationToken::new();
        let handle = tokio::spawn(async move { std::future::pending::<()>().await });
        let registered = queue.register("c1", task("t1"), handle, token.clone(), "agent-1");
        assert_eq!(registered.conversation_id(), "c1");
        assert_eq!(registered.task().task_id, "t1");
        assert_eq!(registered.target_id(), "agent-1");
        assert!(!registered.token().is_cancelled());

        let found = queue.find("c1").expect("registered turn is tracked");
        assert_eq!(found.target_id(), "agent-1");
        assert_eq!(found.conversation_id(), "c1");
        assert!(queue.has_running("c1"));
        assert!(!queue.has_running("c2"));
        assert_eq!(queue.len(), 1);
        assert!(!queue.is_empty());
        assert_eq!(queue.running_conversations(), vec!["c1".to_owned()]);

        let removed = queue.unregister("c1").expect("tracked turn is returned");
        assert_eq!(removed.task().task_id, "t1");
        assert!(queue.unregister("c1").is_none());
        assert!(queue.find("c1").is_none());
        assert!(queue.is_empty());
    }

    #[tokio::test]
    async fn register_replaces_the_existing_turn_of_a_conversation() {
        let queue = TaskQueue::new();
        let first_handle = tokio::spawn(async move { std::future::pending::<()>().await });
        queue.register(
            "c1",
            task("t1"),
            first_handle,
            CancellationToken::new(),
            "agent-1",
        );
        let second_handle = tokio::spawn(async move { std::future::pending::<()>().await });
        queue.register(
            "c1",
            task("t2"),
            second_handle,
            CancellationToken::new(),
            "agent-2",
        );
        assert_eq!(queue.len(), 1);
        let found = queue.find("c1").expect("turn is tracked");
        assert_eq!(found.task().task_id, "t2");
        assert_eq!(found.target_id(), "agent-2");
    }

    #[tokio::test]
    async fn cancel_running_on_a_missing_conversation_is_false() {
        let queue = TaskQueue::new();
        assert!(!queue.cancel_running("missing").await);
    }

    #[tokio::test]
    async fn cancel_running_awaits_the_cancelled_task() {
        let queue = TaskQueue::new();
        let token = CancellationToken::new();
        let (handle, ticks) = looping_task(&token);
        queue.register("c1", task("t1"), handle, token.clone(), "agent-1");

        // Let the loop tick at least once so cancellation has something to stop.
        tokio::time::sleep(StdDuration::from_millis(15)).await;
        let before = ticks.load(Ordering::SeqCst);

        let cancelled = tokio::time::timeout(BOUND, queue.cancel_running("c1"))
            .await
            .expect("cancel completes within the bound");
        assert!(cancelled, "a live turn is cancelled");
        assert!(token.is_cancelled());
        let after = ticks.load(Ordering::SeqCst);
        tokio::time::sleep(StdDuration::from_millis(20)).await;
        assert_eq!(
            ticks.load(Ordering::SeqCst),
            after,
            "the cancelled task stops ticking"
        );
        assert!(after >= before);

        // The entry survives cancellation until its owner unregisters it, but
        // the handle is gone, so a second cancellation reports `false`.
        assert!(queue.has_running("c1"));
        assert!(!queue.cancel_running("c1").await);
        let removed = queue.unregister("c1").expect("tracked turn is returned");
        assert_eq!(removed.conversation_id(), "c1");
        assert!(queue.is_empty());
    }

    #[tokio::test]
    async fn cancel_running_on_a_finished_turn_is_false() {
        let queue = TaskQueue::new();
        let token = CancellationToken::new();
        let handle = tokio::spawn(async {});
        // Give the spawned task a chance to complete.
        for _ in 0..50 {
            if handle.is_finished() {
                break;
            }
            tokio::time::sleep(StdDuration::from_millis(1)).await;
        }
        assert!(handle.is_finished(), "the spawned task completes");
        queue.register("c1", task("t1"), handle, token, "agent-1");
        assert!(!queue.cancel_running("c1").await);
        assert!(queue.has_running("c1"));
    }

    #[tokio::test]
    async fn started_at_is_non_decreasing_across_registrations() {
        let queue = TaskQueue::new();
        let first = queue.register(
            "c1",
            task("t1"),
            tokio::spawn(async move { std::future::pending::<()>().await }),
            CancellationToken::new(),
            "agent-1",
        );
        let second = queue.register(
            "c2",
            task("t2"),
            tokio::spawn(async move { std::future::pending::<()>().await }),
            CancellationToken::new(),
            "agent-2",
        );
        assert!(second.started_at() >= first.started_at());
        assert!(first.started_at() <= process_base_instant().elapsed());
    }
}
