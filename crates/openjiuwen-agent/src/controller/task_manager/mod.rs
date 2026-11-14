//! Shared task manager: tasks plus their priority and hierarchy indexes.
//!
//! Ports `openjiuwen/core/controller/modules/task_manager.py`: one store holds
//! every session's [`Task`]s and keeps three indexes beside the id map — the
//! priority index (`priority -> task ids`), the parent/child links in both
//! directions, and the root set — so a query, a removal, or a status update
//! touches only the tasks it selects.
//!
//! The surface is synchronous: the store is a [`parking_lot::Mutex`] that is
//! never held across an `await`, so a scheduler may call it from any task. Every
//! returned [`Task`] is an owned deep copy, which means a caller can mutate what
//! it read without touching the store.
//!
//! Selection is [`TaskFilter`]'s job: its primary selectors are unioned, then
//! `status`/`user_id` narrow the result, and `with_children` finally appends
//! every transitive child. Results are ordered by the Rust-only
//! [`Task::created_seq`] with `task_id` as the tiebreak, so the order is
//! deterministic (Python iterates a set there).
//!
//! Persistence is per session: [`TaskManager::snapshot_session`] produces the
//! wire shape the controller already stores (`{"tasks": [...], "next_seq": n}`)
//! and [`TaskManager::restore_session`] atomically replaces one session's slice
//! while leaving every other session untouched.
#![allow(clippy::module_name_repetitions)]

mod filter;
mod index;
mod state;

#[cfg(test)]
mod tests;

pub use filter::{PrioritySelector, TaskFilter, TaskIdSelector};
pub use state::TaskManagerState;

use crate::controller::config::ControllerConfig;
use crate::controller::schema::{Task, TaskStatus};
use index::Indexes;
use openjiuwen_common::error::{Error, Result};
use parking_lot::{Mutex, RwLock};
use std::collections::BTreeSet;
use std::sync::Arc;

/// The callback fired when a task enters [`TaskStatus::Submitted`].
type SubmittedCallback = Arc<dyn Fn() + Send + Sync>;

/// The shared task store of the controller stack.
///
/// A manager is a cheap handle (`Arc` inside), so every component shares one
/// instance: the planner inserts tasks, the scheduler claims them, the
/// controller reports their status, and the turn loop snapshots and restores
/// them per session.
#[derive(Clone, Default)]
pub struct TaskManager {
    /// The task map and its indexes.
    inner: Arc<Mutex<Indexes>>,
    /// The controller configuration this manager was built with.
    config: Arc<RwLock<ControllerConfig>>,
    /// The optional task-submitted listener.
    on_submitted: Arc<RwLock<Option<SubmittedCallback>>>,
}

impl TaskManager {
    /// Builds an empty manager with [`ControllerConfig::default`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds an empty manager carrying `config`.
    #[must_use]
    pub fn with_config(config: ControllerConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            ..Self::default()
        }
    }

    /// Returns the configuration the manager was built with.
    #[must_use]
    pub fn config(&self) -> ControllerConfig {
        self.config.read().clone()
    }

    /// Replaces the configuration.
    ///
    /// The manager only exposes the configuration; scheduling defaults are read
    /// by the callers that create tasks.
    pub fn set_config(&self, config: ControllerConfig) {
        *self.config.write() = config;
    }

    /// Registers the listener fired whenever a task enters
    /// [`TaskStatus::Submitted`].
    ///
    /// The listener is invoked after the store lock is released, so it may call
    /// back into the manager.
    pub fn set_on_task_submitted(&self, callback: Arc<dyn Fn() + Send + Sync>) {
        *self.on_submitted.write() = Some(callback);
    }

    /// Clears the task-submitted listener.
    pub fn clear_on_task_submitted(&self) {
        *self.on_submitted.write() = None;
    }

    /// Adds one task.
    ///
    /// The stored task is a copy; the manager assigns it a creation sequence
    /// unless it already carries one.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] with `"<task_id> already exists!"` when the
    /// id is already stored (in this or another session).
    pub fn add_task(&self, task: Task) -> Result<()> {
        let stored = {
            let mut inner = self.inner.lock();
            if inner.contains(&task.task_id) {
                return Err(Error::validation(format!(
                    "{} already exists!",
                    task.task_id
                )));
            }
            inner.insert(task)
        };
        self.notify_submitted(&[stored]);
        Ok(())
    }

    /// Adds several tasks.
    ///
    /// The batch is all-or-nothing: a duplicate id inside the batch or in the
    /// store rejects the whole call and stores nothing.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] with `"<task_id> already exists!"` for the
    /// first duplicated id.
    pub fn add_tasks(&self, tasks: Vec<Task>) -> Result<()> {
        let mut has_submitted = false;
        {
            let mut inner = self.inner.lock();
            let mut seen = BTreeSet::new();
            for task in &tasks {
                if inner.contains(&task.task_id) || !seen.insert(task.task_id.clone()) {
                    return Err(Error::validation(format!(
                        "{} already exists!",
                        task.task_id
                    )));
                }
            }
            for task in tasks {
                has_submitted |= task.status == TaskStatus::Submitted;
                inner.insert(task);
            }
        }
        if has_submitted {
            self.fire_submitted();
        }
        Ok(())
    }

    /// Returns the tasks matching `filter`, or every task when it is `None`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when the filter names no criterion, and
    /// when it asks for [`PrioritySelector::Highest`] (only
    /// [`pop_task`](Self::pop_task) resolves that marker).
    pub fn get_task(&self, filter: Option<&TaskFilter>) -> Result<Vec<Task>> {
        let inner = self.inner.lock();
        let Some(filter) = filter else {
            return Ok(inner.ordered());
        };
        filter.validate()?;
        let level = match filter.priority {
            Some(PrioritySelector::Highest) => {
                return Err(Error::validation(
                    "Priority 'highest' is not supported in get_task, use pop_task instead",
                ));
            }
            Some(PrioritySelector::Level(level)) => Some(level),
            None => None,
        };
        Ok(inner.select(filter, level))
    }

    /// Returns a copy of one task.
    #[must_use]
    pub fn task(&self, task_id: &str) -> Option<Task> {
        self.inner.lock().get(task_id)
    }

    /// Returns every task in creation order.
    #[must_use]
    pub fn tasks(&self) -> Vec<Task> {
        self.inner.lock().ordered()
    }

    /// Returns one session's tasks in creation order.
    #[must_use]
    pub fn session_tasks(&self, session_id: &str) -> Vec<Task> {
        self.inner.lock().session(session_id)
    }

    /// Replaces one stored task.
    ///
    /// The priority and hierarchy indexes move with the task, and a task
    /// carrying [`Task::created_seq`] `0` keeps the stored sequence.
    ///
    /// Returns `false` when the id is unknown; the store is unchanged then.
    #[must_use]
    pub fn update_task(&self, task: Task) -> bool {
        self.inner.lock().update(task)
    }

    /// Replaces several stored tasks.
    ///
    /// Every update is attempted, so a known id is still applied next to an
    /// unknown one; the return value is `true` only when every update landed.
    #[must_use]
    pub fn update_tasks(&self, tasks: Vec<Task>) -> bool {
        let mut inner = self.inner.lock();
        let mut all_updated = true;
        for task in tasks {
            all_updated &= inner.update(task);
        }
        all_updated
    }

    /// Removes the tasks matching `filter`, including their children when the
    /// filter asks for them, and returns the removed copies.
    ///
    /// Every child of a removed task that survives becomes a root.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when `filter` is `None`
    /// (`"task_filter cannot be None in remove_task"`), when it names no
    /// criterion, or when it asks for [`PrioritySelector::Highest`].
    pub fn remove_task(&self, filter: Option<&TaskFilter>) -> Result<Vec<Task>> {
        self.remove_matching(filter, "remove_task", false)
    }

    /// Removes and returns the tasks matching `filter`, resolving
    /// [`PrioritySelector::Highest`] to the largest priority level present.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when `filter` is `None`
    /// (`"task_filter cannot be None in pop_task"`) or when it names no
    /// criterion.
    pub fn pop_task(&self, filter: Option<&TaskFilter>) -> Result<Vec<Task>> {
        self.remove_matching(filter, "pop_task", true)
    }

    /// Returns the direct children of `task_id`, or every transitive child when
    /// `recursive` is set.
    #[must_use]
    pub fn get_child_task(&self, task_id: &str, recursive: bool) -> Vec<Task> {
        let inner = self.inner.lock();
        let task_ids = if recursive {
            inner.descendants(task_id)
        } else {
            inner.children(task_id)
        };
        inner.tasks_for(&task_ids)
    }

    /// Atomically claims a submitted task for execution.
    ///
    /// This is the scheduler's double-execution guard: only the first call for a
    /// [`TaskStatus::Submitted`] task flips it to [`TaskStatus::Working`] and
    /// returns `true`.
    #[must_use]
    pub fn claim_task(&self, task_id: &str) -> bool {
        self.inner.lock().claim(task_id)
    }

    /// Sets the status of `task_id`, optionally of its children.
    ///
    /// A [`TaskStatus::Failed`] transition stores `error_message`, falling back
    /// to `"Task execution failed"`. An update that leaves at least one task in
    /// [`TaskStatus::Submitted`] fires the submitted listener.
    pub fn update_task_status(
        &self,
        task_id: &str,
        status: TaskStatus,
        with_children: bool,
        recursive: bool,
        error_message: Option<String>,
    ) {
        let updated = {
            let mut inner = self.inner.lock();
            let task_ids = inner.targets(task_id, with_children, recursive);
            inner.set_status(&task_ids, status, error_message)
        };
        if status == TaskStatus::Submitted && updated > 0 {
            self.fire_submitted();
        }
    }

    /// Sets the priority of `task_id`, optionally of its children.
    ///
    /// The priority index is rewritten only for the tasks whose level changes.
    pub fn set_priority(&self, task_id: &str, priority: i32, with_children: bool, recursive: bool) {
        let mut inner = self.inner.lock();
        let task_ids = inner.targets(task_id, with_children, recursive);
        inner.set_priority(&task_ids, priority);
    }

    /// Number of stored tasks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.lock().len()
    }

    /// Whether no task is stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.lock().is_empty()
    }

    /// Whether any task is submitted or working.
    #[must_use]
    pub fn has_active_tasks(&self) -> bool {
        self.inner.lock().has_active(None)
    }

    /// Whether any task of `session_id` is submitted or working.
    #[must_use]
    pub fn session_has_active_tasks(&self, session_id: &str) -> bool {
        self.inner.lock().has_active(Some(session_id))
    }

    /// Drops every task and index.
    ///
    /// The creation-sequence counter keeps advancing, so ordering stays
    /// monotonic across a clear.
    pub fn clear(&self) {
        self.inner.lock().clear();
    }

    /// The creation sequence the next inserted task receives.
    #[must_use]
    pub fn next_created_seq(&self) -> u64 {
        self.inner.lock().next_created_seq()
    }

    /// Snapshots one session's tasks plus the ordering sequence for persistence.
    #[must_use]
    pub fn snapshot_session(&self, session_id: &str) -> TaskManagerState {
        self.inner.lock().snapshot(session_id)
    }

    /// Replaces one session's tasks with `state`, leaving other sessions alone.
    ///
    /// Restored tasks are re-indexed and get fresh creation sequences in stored
    /// order, because `created_seq` is not on the wire. `None` clears the
    /// session's slice.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] with
    /// `"Task ID already belongs to another session: <id>"` when a restored id
    /// is owned by a different session; nothing changes in that case.
    pub fn restore_session(
        &self,
        session_id: &str,
        state: Option<&TaskManagerState>,
    ) -> Result<()> {
        self.inner.lock().restore(session_id, state)
    }

    /// The shared removal path behind [`remove_task`](Self::remove_task) and
    /// [`pop_task`](Self::pop_task).
    fn remove_matching(
        &self,
        filter: Option<&TaskFilter>,
        method: &str,
        allow_highest: bool,
    ) -> Result<Vec<Task>> {
        let Some(filter) = filter else {
            return Err(Error::validation(format!(
                "task_filter cannot be None in {method}"
            )));
        };
        filter.validate()?;
        let mut inner = self.inner.lock();
        let level = match filter.priority {
            Some(PrioritySelector::Highest) if allow_highest => {
                let Some(level) = inner.max_priority() else {
                    return Ok(Vec::new());
                };
                Some(level)
            }
            Some(PrioritySelector::Highest) => {
                return Err(Error::validation(format!(
                    "Priority 'highest' is not supported in {method}"
                )));
            }
            Some(PrioritySelector::Level(level)) => Some(level),
            None => None,
        };
        let task_ids: BTreeSet<String> = inner.select_ids(filter, level).into_iter().collect();
        if task_ids.is_empty() {
            return Ok(Vec::new());
        }
        Ok(inner.remove_ids(&task_ids))
    }

    /// Fires the submitted listener when `tasks` holds a submitted task.
    fn notify_submitted(&self, tasks: &[Task]) {
        if tasks
            .iter()
            .any(|task| task.status == TaskStatus::Submitted)
        {
            self.fire_submitted();
        }
    }

    /// Invokes the submitted listener, if any, outside the store lock.
    fn fire_submitted(&self) {
        let callback = self.on_submitted.read().clone();
        if let Some(callback) = callback {
            callback();
        }
    }
}
