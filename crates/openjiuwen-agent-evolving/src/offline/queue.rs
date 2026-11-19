//! The rollout task queue (Python `coordinator/task_queue.py`).
//!
//! One queue serves a whole round: tasks are pushed in, workers pull them and
//! record the in-processing set, and finished rollouts are buffered under their
//! rollout id until the coordinator drains them.
//!
//! Python uses `asyncio.Queue` plus two `asyncio.Lock`s. This port keeps one
//! `tokio::sync::Mutex` over the whole state, which makes every operation
//! atomic and therefore safe to call from several worker tasks. Because the
//! lock is asynchronous, the methods Python declares synchronous
//! ([`TaskQueue::is_finished`], [`TaskQueue::clear`]) are `async` here.

use std::collections::{BTreeMap, VecDeque};

use tokio::sync::Mutex;

use crate::rl::{RlTask, RolloutMessage};

/// The mutable queue state guarded by the queue's lock.
#[derive(Debug, Default)]
struct QueueState {
    tasks: VecDeque<RlTask>,
    in_processing: BTreeMap<String, RlTask>,
    rollouts: BTreeMap<String, RolloutMessage>,
}

/// The async task queue and rollout result buffer of the offline runtime.
///
/// Ordering is FIFO by insertion, and the rollout buffer is a `BTreeMap` so
/// draining it is deterministic regardless of which worker finished first.
#[derive(Debug, Default)]
pub struct TaskQueue {
    state: Mutex<QueueState>,
}

impl TaskQueue {
    /// Creates an empty queue.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enqueues `task` and returns its task id.
    pub async fn queue_task(&self, task: RlTask) -> String {
        let task_id = task.task_id.clone();
        let mut state = self.state.lock().await;
        state.tasks.push_back(task);
        task_id
    }

    /// Pops the next queued task and records it as in processing.
    ///
    /// Returns `None` when the queue is empty; an in-processing task is never
    /// handed out twice.
    pub async fn get_task(&self) -> Option<RlTask> {
        let mut state = self.state.lock().await;
        let task = state.tasks.pop_front()?;
        state
            .in_processing
            .insert(task.task_id.clone(), task.clone());
        Some(task)
    }

    /// Drops a task from the in-processing set.
    ///
    /// Returns whether the task was in processing. Python's version returns
    /// nothing; the flag lets a worker tell "deleted" apart from "already
    /// accounted for" without a second lookup.
    pub async fn delete_task(&self, task: &RlTask) -> bool {
        self.state
            .lock()
            .await
            .in_processing
            .remove(&task.task_id)
            .is_some()
    }

    /// Buffers a finished rollout, keyed by its rollout id.
    ///
    /// The rollout's task id leaves the in-processing set first, which is what
    /// makes [`Self::is_finished`] true once every task reported back. An empty
    /// task id never matched an in-processing entry in Python either.
    ///
    /// Returns the rollout id.
    pub async fn add_rollout(&self, rollout: RolloutMessage) -> String {
        let rollout_id = rollout.rollout_id.clone();
        let mut state = self.state.lock().await;
        if !rollout.task_id.is_empty() {
            state.in_processing.remove(&rollout.task_id);
        }
        state.rollouts.insert(rollout_id.clone(), rollout);
        rollout_id
    }

    /// Returns a copy of the buffered rollouts and clears the buffer.
    pub async fn get_rollouts(&self) -> BTreeMap<String, RolloutMessage> {
        let mut state = self.state.lock().await;
        std::mem::take(&mut state.rollouts)
    }

    /// Whether the queue is empty and no task is in processing.
    ///
    /// A task that was pulled but never returned keeps this `false`, which is
    /// how Python detects a worker that died mid-task.
    pub async fn is_finished(&self) -> bool {
        let state = self.state.lock().await;
        state.tasks.is_empty() && state.in_processing.is_empty()
    }

    /// Drops queued tasks, the in-processing set and the rollout buffer.
    ///
    /// The coordinator calls this between rounds so state never leaks from one
    /// training step into the next.
    pub async fn clear(&self) {
        let mut state = self.state.lock().await;
        state.tasks.clear();
        state.in_processing.clear();
        state.rollouts.clear();
    }

    /// Number of tasks waiting to be pulled.
    pub async fn queued_len(&self) -> usize {
        self.state.lock().await.tasks.len()
    }

    /// Number of tasks that were pulled but have not reported a rollout.
    pub async fn in_processing_len(&self) -> usize {
        self.state.lock().await.in_processing.len()
    }

    /// Number of buffered rollouts waiting to be drained.
    pub async fn rollout_len(&self) -> usize {
        self.state.lock().await.rollouts.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn task(id: &str) -> RlTask {
        RlTask {
            task_id: id.to_owned(),
            origin_task_id: Some("origin".to_owned()),
            task_sample: json!({"query": id}),
            round_num: 0,
        }
    }

    fn rollout(rollout_id: &str, task_id: &str, reward: f64) -> RolloutMessage {
        RolloutMessage {
            task_id: task_id.to_owned(),
            origin_task_id: Some("origin".to_owned()),
            rollout_id: rollout_id.to_owned(),
            start_time: None,
            end_time: None,
            rollout_info: json!([]),
            reward_list: vec![reward],
            global_reward: reward,
            turn_count: 0,
            round_num: 0,
        }
    }

    #[tokio::test]
    async fn tasks_are_handed_out_once_in_fifo_order() {
        let queue = TaskQueue::new();
        assert!(queue.is_finished().await);
        assert_eq!(queue.queue_task(task("a")).await, "a");
        queue.queue_task(task("b")).await;
        assert_eq!(queue.queued_len().await, 2);
        assert!(!queue.is_finished().await);

        let first = queue.get_task().await.expect("first task");
        assert_eq!(first.task_id, "a");
        assert_eq!(queue.queued_len().await, 1);
        assert_eq!(queue.in_processing_len().await, 1);
        assert!(!queue.is_finished().await);

        let second = queue.get_task().await.expect("second task");
        assert_eq!(second.task_id, "b");
        assert!(queue.get_task().await.is_none());
        assert_eq!(queue.in_processing_len().await, 2);
        assert!(!queue.is_finished().await);
    }

    #[tokio::test]
    async fn rollouts_clear_in_processing_and_drain_once() {
        let queue = TaskQueue::new();
        queue.queue_task(task("a")).await;
        let task = queue.get_task().await.expect("task");

        let id = queue.add_rollout(rollout("r-a", &task.task_id, 0.5)).await;
        assert_eq!(id, "r-a");
        assert_eq!(queue.in_processing_len().await, 0);
        assert!(queue.is_finished().await);
        assert_eq!(queue.rollout_len().await, 1);

        let drained = queue.get_rollouts().await;
        assert_eq!(drained.len(), 1);
        assert_eq!(drained["r-a"].global_reward, 0.5);
        assert!(queue.get_rollouts().await.is_empty());

        // A second rollout with the same id replaces the first.
        queue.add_rollout(rollout("r-a", "a", 0.1)).await;
        queue.add_rollout(rollout("r-a", "a", 0.9)).await;
        let drained = queue.get_rollouts().await;
        assert_eq!(drained.len(), 1);
        assert_eq!(drained["r-a"].global_reward, 0.9);
    }

    #[tokio::test]
    async fn delete_task_reports_whether_it_was_processing() {
        let queue = TaskQueue::new();
        queue.queue_task(task("a")).await;
        let taken = queue.get_task().await.expect("task");
        assert!(queue.delete_task(&taken).await);
        assert!(!queue.delete_task(&taken).await);
        assert!(queue.is_finished().await);

        // A task that was never pulled is not in processing.
        assert!(!queue.delete_task(&task("ghost")).await);
    }

    #[tokio::test]
    async fn add_rollout_with_an_empty_task_id_leaves_processing_entries_alone() {
        let queue = TaskQueue::new();
        queue.queue_task(task("a")).await;
        let taken = queue.get_task().await.expect("task");
        queue.add_rollout(rollout("r-x", "", 1.0)).await;
        assert_eq!(queue.in_processing_len().await, 1);
        assert!(!queue.is_finished().await);
        queue.add_rollout(rollout("r-a", &taken.task_id, 1.0)).await;
        assert!(queue.is_finished().await);
    }

    #[tokio::test]
    async fn clear_resets_every_container() {
        let queue = TaskQueue::new();
        queue.queue_task(task("a")).await;
        queue.queue_task(task("b")).await;
        queue.get_task().await;
        queue.add_rollout(rollout("r-1", "a", 1.0)).await;
        queue.clear().await;
        assert!(queue.is_finished().await);
        assert_eq!(queue.queued_len().await, 0);
        assert_eq!(queue.in_processing_len().await, 0);
        assert!(queue.get_rollouts().await.is_empty());
    }
}
