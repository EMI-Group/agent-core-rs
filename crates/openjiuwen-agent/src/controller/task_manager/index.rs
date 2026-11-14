//! The task store and the priority/hierarchy indexes kept in sync with it.
//!
//! A [`Indexes`] value owns the task map (`task_id -> task`), the priority
//! index (`priority -> task ids` in insertion order), the parent/child links in
//! both directions, the root set, and the ordering sequence. Every mutation goes
//! through one of the methods here so the indexes never drift from the map.
use super::filter::TaskFilter;
use super::state::TaskManagerState;
use crate::controller::schema::{Task, TaskStatus};
use openjiuwen_common::error::{Error, Result};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

/// Message stored on a task that failed without one of its own.
pub(super) const DEFAULT_FAILURE_MESSAGE: &str = "Task execution failed";

/// The task map plus the indexes that mirror it.
#[derive(Debug, Default)]
pub(super) struct Indexes {
    /// Tasks by id.
    tasks: BTreeMap<String, Task>,
    /// Priority level to its task ids, in insertion order.
    priority: BTreeMap<i32, Vec<String>>,
    /// Parent id to its direct children.
    parent_to_children: BTreeMap<String, BTreeSet<String>>,
    /// Child id to its parent id.
    child_to_parent: BTreeMap<String, String>,
    /// Ids of the tasks that have no parent.
    roots: BTreeSet<String>,
    /// Sequence number assigned to the next inserted task.
    next_seq: u64,
}

/// Orders tasks by creation sequence, then by id so the order is total.
fn sort_tasks(tasks: &mut [Task]) {
    tasks.sort_by(|left, right| {
        left.created_seq
            .cmp(&right.created_seq)
            .then_with(|| left.task_id.cmp(&right.task_id))
    });
}

impl Indexes {
    /// The sequence number [`insert`](Self::insert) assigns to the next task.
    ///
    /// The counter starts at one, so a zero `created_seq` always means
    /// "unassigned".
    pub(super) fn next_created_seq(&self) -> u64 {
        self.next_seq.max(1)
    }

    /// Number of stored tasks.
    pub(super) fn len(&self) -> usize {
        self.tasks.len()
    }

    /// Whether no task is stored.
    pub(super) fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    /// Whether `task_id` is stored.
    pub(super) fn contains(&self, task_id: &str) -> bool {
        self.tasks.contains_key(task_id)
    }

    /// Returns an owned copy of one task.
    pub(super) fn get(&self, task_id: &str) -> Option<Task> {
        self.tasks.get(task_id).cloned()
    }

    /// Drops every task and index; the ordering sequence keeps advancing.
    pub(super) fn clear(&mut self) {
        self.tasks.clear();
        self.priority.clear();
        self.parent_to_children.clear();
        self.child_to_parent.clear();
        self.roots.clear();
    }

    /// Stores `task`, assigning it a creation sequence when it has none, and
    /// indexes it.
    ///
    /// The caller rejects duplicate ids; inserting an existing id overwrites it.
    pub(super) fn insert(&mut self, mut task: Task) -> Task {
        if task.created_seq == 0 {
            task.created_seq = self.next_created_seq();
        }
        self.next_seq = self.next_seq.max(task.created_seq.saturating_add(1));
        self.index(&task);
        self.tasks.insert(task.task_id.clone(), task.clone());
        task
    }

    /// Replaces an existing task and moves its index entries.
    ///
    /// Returns `false` when the id is unknown. A task carrying `created_seq = 0`
    /// keeps the stored task's sequence, so a task that round-tripped through
    /// JSON keeps its place.
    pub(super) fn update(&mut self, mut task: Task) -> bool {
        let Some(old) = self.tasks.get(&task.task_id).cloned() else {
            return false;
        };
        if task.created_seq == 0 {
            task.created_seq = old.created_seq;
        }
        if old.priority != task.priority {
            self.unindex_priority(&old.task_id, old.priority);
            self.priority
                .entry(task.priority)
                .or_default()
                .push(task.task_id.clone());
        }
        if old.parent_task_id != task.parent_task_id {
            if let Some(parent) = old.parent_task_id.as_deref() {
                self.unlink_child(parent, &task.task_id);
            }
            if let Some(parent) = task.parent_task_id.as_deref() {
                self.link_child(parent, &task.task_id);
            } else {
                self.child_to_parent.remove(&task.task_id);
                self.roots.insert(task.task_id.clone());
            }
        }
        self.tasks.insert(task.task_id.clone(), task);
        true
    }

    /// Flips a submitted task to working; `false` when it is missing or already
    /// claimed.
    pub(super) fn claim(&mut self, task_id: &str) -> bool {
        match self.tasks.get_mut(task_id) {
            Some(task) if task.status == TaskStatus::Submitted => {
                task.status = TaskStatus::Working;
                true
            }
            _ => false,
        }
    }

    /// Every task in creation order.
    pub(super) fn ordered(&self) -> Vec<Task> {
        let mut tasks: Vec<Task> = self.tasks.values().cloned().collect();
        sort_tasks(&mut tasks);
        tasks
    }

    /// One session's tasks in creation order.
    pub(super) fn session(&self, session_id: &str) -> Vec<Task> {
        let mut tasks: Vec<Task> = self
            .tasks
            .values()
            .filter(|task| task.session_id == session_id)
            .cloned()
            .collect();
        sort_tasks(&mut tasks);
        tasks
    }

    /// The tasks among `task_ids` that still exist, in creation order.
    pub(super) fn tasks_for(&self, task_ids: &BTreeSet<String>) -> Vec<Task> {
        let mut tasks: Vec<Task> = task_ids
            .iter()
            .filter_map(|task_id| self.tasks.get(task_id).cloned())
            .collect();
        sort_tasks(&mut tasks);
        tasks
    }

    /// The tasks matching `filter` at the resolved priority `level`.
    ///
    /// The caller resolves `"highest"` (and validates the filter) first.
    pub(super) fn select(&self, filter: &TaskFilter, level: Option<i32>) -> Vec<Task> {
        self.select_ids(filter, level)
            .into_iter()
            .filter_map(|task_id| self.tasks.get(&task_id).cloned())
            .collect()
    }

    /// The ids matching `filter`, in creation order.
    pub(super) fn select_ids(&self, filter: &TaskFilter, level: Option<i32>) -> Vec<String> {
        let mut candidates: BTreeSet<String> = BTreeSet::new();
        if let Some(selector) = &filter.task_id {
            candidates.extend(
                selector
                    .to_vec()
                    .into_iter()
                    .filter(|task_id| self.tasks.contains_key(task_id)),
            );
        }
        if let Some(session_id) = &filter.session_id {
            candidates.extend(
                self.tasks
                    .iter()
                    .filter(|(_, task)| &task.session_id == session_id)
                    .map(|(task_id, _)| task_id.clone()),
            );
        }
        if let Some(level) = level {
            if let Some(task_ids) = self.priority.get(&level) {
                candidates.extend(
                    task_ids
                        .iter()
                        .filter(|task_id| self.tasks.contains_key(*task_id))
                        .cloned(),
                );
            }
        }
        if filter.is_root {
            candidates.extend(
                self.roots
                    .iter()
                    .filter(|task_id| self.tasks.contains_key(*task_id))
                    .cloned(),
            );
        }
        if !filter.has_primary_selector() {
            candidates.extend(self.tasks.keys().cloned());
        }

        let mut matched: Vec<String> = candidates
            .into_iter()
            .filter(|task_id| self.matches(task_id, filter))
            .collect();
        if filter.with_children {
            let mut children = BTreeSet::new();
            for task_id in &matched {
                children.extend(self.descendants(task_id));
            }
            matched.extend(children);
            matched.sort_unstable();
            matched.dedup();
        }
        matched.sort_by(|left, right| self.order_key(left).cmp(&self.order_key(right)));
        matched
    }

    /// Removes `task_ids`, promoting every surviving child of a removed task to
    /// a root, and returns the removed copies in creation order.
    pub(super) fn remove_ids(&mut self, task_ids: &BTreeSet<String>) -> Vec<Task> {
        let mut promoted: BTreeSet<String> = BTreeSet::new();
        for task_id in task_ids {
            if let Some(children) = self.parent_to_children.get(task_id) {
                promoted.extend(
                    children
                        .iter()
                        .filter(|child| !task_ids.contains(*child))
                        .cloned(),
                );
            }
        }

        let mut removed = Vec::with_capacity(task_ids.len());
        for task_id in task_ids {
            let Some(task) = self.tasks.remove(task_id) else {
                continue;
            };
            self.unindex_priority(task_id, task.priority);
            if let Some(parent) = task.parent_task_id.as_deref() {
                self.unlink_child(parent, task_id);
            }
            self.parent_to_children.remove(task_id);
            self.child_to_parent.remove(task_id);
            self.roots.remove(task_id);
            removed.push(task);
        }

        for task_id in promoted {
            if let Some(task) = self.tasks.get_mut(&task_id) {
                task.parent_task_id = None;
                self.child_to_parent.remove(&task_id);
                self.roots.insert(task_id);
            }
        }

        sort_tasks(&mut removed);
        removed
    }

    /// Every transitive child of `task_id`.
    pub(super) fn descendants(&self, task_id: &str) -> BTreeSet<String> {
        let mut collected = BTreeSet::new();
        let mut pending = vec![task_id.to_owned()];
        while let Some(parent) = pending.pop() {
            let Some(children) = self.parent_to_children.get(&parent) else {
                continue;
            };
            for child in children {
                if collected.insert(child.clone()) {
                    pending.push(child.clone());
                }
            }
        }
        collected
    }

    /// The direct children of `task_id`.
    pub(super) fn children(&self, task_id: &str) -> BTreeSet<String> {
        self.parent_to_children
            .get(task_id)
            .cloned()
            .unwrap_or_default()
    }

    /// The ids a status or priority update touches: `task_id`, optionally its
    /// direct children, and optionally every descendant.
    pub(super) fn targets(
        &self,
        task_id: &str,
        with_children: bool,
        recursive: bool,
    ) -> BTreeSet<String> {
        let mut ids = BTreeSet::from([task_id.to_owned()]);
        if with_children {
            if recursive {
                ids.extend(self.descendants(task_id));
            } else {
                ids.extend(self.children(task_id));
            }
        }
        ids
    }

    /// The largest priority level that holds a task.
    pub(super) fn max_priority(&self) -> Option<i32> {
        self.priority
            .iter()
            .rev()
            .find(|(_, task_ids)| !task_ids.is_empty())
            .map(|(level, _)| *level)
    }

    /// Sets the status of every existing task in `task_ids`.
    ///
    /// A [`TaskStatus::Failed`] transition stores `error_message`, falling back
    /// to [`DEFAULT_FAILURE_MESSAGE`]. Returns how many tasks changed.
    pub(super) fn set_status(
        &mut self,
        task_ids: &BTreeSet<String>,
        status: TaskStatus,
        error_message: Option<String>,
    ) -> usize {
        let message = if status == TaskStatus::Failed {
            Some(error_message.unwrap_or_else(|| DEFAULT_FAILURE_MESSAGE.to_owned()))
        } else {
            None
        };
        let mut updated = 0;
        for task_id in task_ids {
            if let Some(task) = self.tasks.get_mut(task_id) {
                task.status = status;
                if let Some(message) = &message {
                    task.error_message = Some(message.clone());
                }
                updated += 1;
            }
        }
        updated
    }

    /// Sets the priority of every existing task in `task_ids`, rewriting the
    /// priority index only for the tasks whose level changes.
    pub(super) fn set_priority(&mut self, task_ids: &BTreeSet<String>, priority: i32) {
        for task_id in task_ids {
            let Some(task) = self.tasks.get_mut(task_id) else {
                continue;
            };
            let previous = task.priority;
            task.priority = priority;
            if previous != priority {
                self.unindex_priority(task_id, previous);
                self.priority
                    .entry(priority)
                    .or_default()
                    .push(task_id.clone());
            }
        }
    }

    /// Whether any task is submitted or working, optionally within one session.
    pub(super) fn has_active(&self, session_id: Option<&str>) -> bool {
        self.tasks.values().any(|task| {
            task.status.is_active()
                && session_id.is_none_or(|session_id| task.session_id == session_id)
        })
    }

    /// Snapshots one session's tasks plus the ordering sequence.
    pub(super) fn snapshot(&self, session_id: &str) -> TaskManagerState {
        TaskManagerState::new(self.session(session_id), self.next_created_seq())
    }

    /// Replaces one session's tasks with `state`, rebuilding their index entries.
    ///
    /// Tasks in `state` that belong to another session are ignored, matching
    /// Python's `replace_session_state`. The restored tasks get fresh creation
    /// sequences in stored order because `created_seq` is not on the wire.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when a restored id is already owned by a
    /// different session; nothing is changed in that case.
    pub(super) fn restore(
        &mut self,
        session_id: &str,
        state: Option<&TaskManagerState>,
    ) -> Result<()> {
        let restored: Vec<Task> = state.map_or_else(Vec::new, |state| {
            state
                .tasks
                .iter()
                .filter(|task| task.session_id == session_id)
                .cloned()
                .collect()
        });

        for task in &restored {
            let owned_elsewhere = self
                .tasks
                .get(&task.task_id)
                .is_some_and(|existing| existing.session_id != session_id);
            if owned_elsewhere {
                return Err(Error::validation(format!(
                    "Task ID already belongs to another session: {}",
                    task.task_id
                )));
            }
        }

        let replaced: BTreeSet<String> = self
            .tasks
            .values()
            .filter(|task| task.session_id == session_id)
            .map(|task| task.task_id.clone())
            .collect();
        self.remove_ids(&replaced);

        let mut sequence = self.next_created_seq();
        for mut task in restored {
            task.created_seq = sequence;
            sequence = sequence.saturating_add(1);
            self.index(&task);
            self.tasks.insert(task.task_id.clone(), task);
        }
        let stored_sequence = state.map_or(0, |state| state.next_seq);
        self.next_seq = self.next_seq.max(sequence).max(stored_sequence);
        Ok(())
    }

    /// The `(created_seq, task_id)` sort key of one stored task.
    fn order_key<'key>(&'key self, task_id: &'key str) -> (u64, &'key str) {
        (
            self.tasks.get(task_id).map_or(0, |task| task.created_seq),
            task_id,
        )
    }

    /// Whether one stored task survives `filter`'s status and user narrowing.
    fn matches(&self, task_id: &str, filter: &TaskFilter) -> bool {
        let Some(task) = self.tasks.get(task_id) else {
            return false;
        };
        if let Some(status) = filter.status {
            if task.status != status {
                return false;
            }
        }
        if let Some(user_id) = filter.user_id.as_deref() {
            let owner = task.metadata.get("user_id").and_then(Value::as_str);
            if owner != Some(user_id) {
                return false;
            }
        }
        true
    }

    /// Adds `task` to every index; the caller inserts it into the map.
    fn index(&mut self, task: &Task) {
        self.priority
            .entry(task.priority)
            .or_default()
            .push(task.task_id.clone());
        if let Some(parent) = task.parent_task_id.as_deref() {
            self.link_child(parent, &task.task_id);
        } else {
            self.roots.insert(task.task_id.clone());
        }
    }

    /// Records `child` under `parent` in both directions.
    fn link_child(&mut self, parent: &str, child: &str) {
        self.parent_to_children
            .entry(parent.to_owned())
            .or_default()
            .insert(child.to_owned());
        self.child_to_parent
            .insert(child.to_owned(), parent.to_owned());
        self.roots.remove(child);
    }

    /// Drops the parent/child links of `child` under `parent`, removing empty
    /// child sets so the index holds only live entries.
    fn unlink_child(&mut self, parent: &str, child: &str) {
        let mut empty = false;
        if let Some(children) = self.parent_to_children.get_mut(parent) {
            children.remove(child);
            empty = children.is_empty();
        }
        if empty {
            self.parent_to_children.remove(parent);
        }
        self.child_to_parent.remove(child);
    }

    /// Removes one task id from a priority level, dropping empty levels so
    /// [`max_priority`](Self::max_priority) never reports a stale one.
    fn unindex_priority(&mut self, task_id: &str, priority: i32) {
        let mut empty = false;
        if let Some(task_ids) = self.priority.get_mut(&priority) {
            task_ids.retain(|entry| entry != task_id);
            empty = task_ids.is_empty();
        }
        if empty {
            self.priority.remove(&priority);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controller::schema::TaskStatus;
    use serde_json::json;

    fn task(task_id: &str, session_id: &str) -> Task {
        Task::new(session_id, task_id, "agent")
    }

    /// Asserts that `tasks` holds nothing.
    fn assert_no_tasks(tasks: &[Task]) {
        assert_eq!(tasks, Vec::<Task>::new(), "expected no tasks");
    }

    #[test]
    fn insert_assigns_sequences_and_builds_the_indexes() {
        let mut indexes = Indexes::default();
        assert_eq!(indexes.next_created_seq(), 1);
        let root = indexes.insert(task("t1", "s1").with_priority(2));
        let child = indexes.insert(task("t2", "s1").with_parent("t1"));
        assert_eq!(root.created_seq, 1);
        assert_eq!(child.created_seq, 2);
        assert_eq!(indexes.next_created_seq(), 3);
        assert_eq!(indexes.len(), 2);
        assert!(!indexes.is_empty());
        assert_eq!(indexes.children("t1"), BTreeSet::from(["t2".to_owned()]));
        assert_eq!(indexes.max_priority(), Some(2));
        assert_eq!(indexes.descendants("t1"), BTreeSet::from(["t2".to_owned()]));
        assert!(indexes.get("t2").is_some());
        assert!(indexes.get("missing").is_none());
        assert!(indexes.contains("t1"));
    }

    #[test]
    fn update_moves_priority_and_hierarchy_entries() {
        let mut indexes = Indexes::default();
        indexes.insert(task("root", "s1"));
        indexes.insert(task("other", "s1"));
        let mut child = indexes.insert(task("t1", "s1").with_priority(1));
        assert_eq!(indexes.max_priority(), Some(1));

        child.priority = 5;
        child.parent_task_id = Some("root".to_owned());
        assert!(indexes.update(child.clone()));
        assert_eq!(indexes.max_priority(), Some(5));
        assert_eq!(indexes.children("root"), BTreeSet::from(["t1".to_owned()]));
        assert!(!indexes.update(task("ghost", "s1")));

        // Detaching the parent turns the task back into a root.
        child.parent_task_id = None;
        assert!(indexes.update(child));
        assert!(indexes.children("root").is_empty());
        let roots = indexes.select(&TaskFilter::roots(), None);
        assert_eq!(
            roots
                .iter()
                .map(|task| task.task_id.as_str())
                .collect::<Vec<_>>(),
            vec!["root", "other", "t1"]
        );
    }

    #[test]
    fn update_keeps_the_stored_sequence_when_the_wire_copy_has_none() {
        let mut indexes = Indexes::default();
        indexes.insert(task("t1", "s1"));
        indexes.insert(task("t2", "s1"));
        let mut parsed: Task = serde_json::from_value(json!({
            "session_id": "s1",
            "task_id": "t1",
            "task_type": "agent",
            "status": "working"
        }))
        .unwrap();
        assert_eq!(parsed.created_seq, 0);
        assert!(indexes.update(parsed.clone()));
        assert_eq!(indexes.get("t1").unwrap().created_seq, 1);
        assert_eq!(indexes.get("t2").unwrap().created_seq, 2);

        parsed.created_seq = 42;
        assert!(indexes.update(parsed));
        assert_eq!(indexes.get("t1").unwrap().created_seq, 42);
    }

    #[test]
    fn removal_promotes_only_surviving_orphans() {
        let mut indexes = Indexes::default();
        indexes.insert(task("root", "s1"));
        indexes.insert(task("kept", "s1").with_parent("root"));
        indexes.insert(task("gone", "s1").with_parent("root"));
        indexes.insert(task("grandchild", "s1").with_parent("kept"));

        let removed = indexes.remove_ids(&BTreeSet::from(["root".to_owned(), "gone".to_owned()]));
        assert_eq!(
            removed
                .iter()
                .map(|task| task.task_id.as_str())
                .collect::<Vec<_>>(),
            vec!["root", "gone"]
        );
        assert_eq!(indexes.len(), 2);
        let kept = indexes.get("kept").unwrap();
        assert_eq!(kept.parent_task_id, None);
        assert_eq!(
            indexes.get("grandchild").unwrap().parent_task_id.as_deref(),
            Some("kept")
        );
        assert_eq!(
            indexes
                .select(&TaskFilter::roots(), None)
                .iter()
                .map(|task| task.task_id.as_str())
                .collect::<Vec<_>>(),
            vec!["kept"]
        );
        // Removing an unknown id is a no-op.
        assert_no_tasks(&indexes.remove_ids(&BTreeSet::from(["nope".to_owned()])));
    }

    #[test]
    fn priority_index_drops_empty_levels() {
        let mut indexes = Indexes::default();
        indexes.insert(task("t1", "s1").with_priority(3));
        indexes.insert(task("t2", "s1").with_priority(7));
        assert_eq!(indexes.max_priority(), Some(7));
        indexes.remove_ids(&BTreeSet::from(["t2".to_owned()]));
        assert_eq!(indexes.max_priority(), Some(3));
        indexes.remove_ids(&BTreeSet::from(["t1".to_owned()]));
        assert_eq!(indexes.max_priority(), None);
        assert!(indexes.is_empty());
    }

    #[test]
    fn claim_is_a_one_shot_transition() {
        let mut indexes = Indexes::default();
        indexes.insert(task("t1", "s1"));
        assert!(indexes.claim("t1"));
        assert_eq!(indexes.get("t1").unwrap().status, TaskStatus::Working);
        assert!(!indexes.claim("t1"));
        assert!(!indexes.claim("missing"));
        indexes.set_status(
            &BTreeSet::from(["t1".to_owned()]),
            TaskStatus::Completed,
            None,
        );
        assert!(!indexes.claim("t1"));
    }

    #[test]
    fn set_status_and_priority_report_what_they_touched() {
        let mut indexes = Indexes::default();
        indexes.insert(task("root", "s1"));
        indexes.insert(task("child", "s1").with_parent("root"));
        let targets = indexes.targets("root", true, true);
        assert_eq!(targets.len(), 2);
        let updated = indexes.set_status(&targets, TaskStatus::Failed, None);
        assert_eq!(updated, 2);
        assert_eq!(
            indexes.get("child").unwrap().error_message.as_deref(),
            Some(DEFAULT_FAILURE_MESSAGE)
        );
        let updated = indexes.set_status(
            &BTreeSet::from(["root".to_owned()]),
            TaskStatus::Failed,
            Some("boom".to_owned()),
        );
        assert_eq!(updated, 1);
        assert_eq!(
            indexes.get("root").unwrap().error_message.as_deref(),
            Some("boom")
        );

        indexes.set_priority(&targets, 4);
        assert_eq!(indexes.get("root").unwrap().priority, 4);
        assert_eq!(indexes.get("child").unwrap().priority, 4);
        assert_eq!(indexes.max_priority(), Some(4));
        assert_eq!(
            indexes.select(&TaskFilter::by_priority(4), Some(4)).len(),
            2
        );
        assert_no_tasks(&indexes.select(&TaskFilter::by_priority(1), Some(1)));
    }

    #[test]
    fn selection_orders_by_creation_sequence_and_expands_children() {
        let mut indexes = Indexes::default();
        indexes.insert(task("b", "s1"));
        indexes.insert(task("a", "s1"));
        indexes.insert(task("child", "s1").with_parent("b"));
        indexes.insert(task("grandchild", "s1").with_parent("child"));

        let all = indexes.select(&TaskFilter::by_session("s1"), None);
        assert_eq!(
            all.iter()
                .map(|task| task.task_id.as_str())
                .collect::<Vec<_>>(),
            vec!["b", "a", "child", "grandchild"]
        );

        let with_children = indexes.select(&TaskFilter::by_id("b").include_children(), None);
        assert_eq!(
            with_children
                .iter()
                .map(|task| task.task_id.as_str())
                .collect::<Vec<_>>(),
            vec!["b", "child", "grandchild"]
        );

        let ids = TaskFilter::by_ids(["a", "b"]);
        let many = indexes.select(&ids, None);
        assert_eq!(
            many.iter()
                .map(|task| task.task_id.as_str())
                .collect::<Vec<_>>(),
            vec!["b", "a"]
        );
    }

    #[test]
    fn clear_keeps_the_sequence_monotonic() {
        let mut indexes = Indexes::default();
        indexes.insert(task("t1", "s1"));
        assert_eq!(indexes.next_created_seq(), 2);
        indexes.clear();
        assert!(indexes.is_empty());
        assert_no_tasks(&indexes.ordered());
        assert_no_tasks(&indexes.select(&TaskFilter::roots(), None));
        assert_eq!(indexes.next_created_seq(), 2);
        assert_eq!(indexes.insert(task("t2", "s1")).created_seq, 2);
    }

    #[test]
    fn snapshot_and_restore_are_per_session() {
        let mut indexes = Indexes::default();
        indexes.insert(task("a1", "a").with_priority(2));
        indexes.insert(task("a2", "a").with_parent("a1"));
        let snapshot = indexes.snapshot("a");
        assert_eq!(snapshot.task_ids(), vec!["a1".to_owned(), "a2".to_owned()]);
        assert_eq!(snapshot.tasks[0].created_seq, 1);
        assert_eq!(snapshot.next_seq, 3);

        indexes.insert(task("b1", "b"));
        assert!(indexes.restore("a", Some(&snapshot)).is_ok());
        assert_eq!(indexes.len(), 3);
        assert_eq!(
            indexes.get("a2").unwrap().parent_task_id.as_deref(),
            Some("a1")
        );

        assert!(indexes.restore("a", None).is_ok());
        assert_eq!(
            indexes
                .ordered()
                .iter()
                .map(|task| task.task_id.as_str())
                .collect::<Vec<_>>(),
            vec!["b1"]
        );
    }

    #[test]
    fn restore_rejects_an_id_owned_by_another_session() {
        let mut indexes = Indexes::default();
        indexes.insert(task("t1", "a"));
        let foreign = TaskManagerState::new(vec![task("t1", "b")], 2);
        let error = indexes.restore("b", Some(&foreign)).unwrap_err();
        assert_eq!(
            error.to_string(),
            "validation: Task ID already belongs to another session: t1"
        );
        // The rejected restore left the manager untouched.
        assert_eq!(indexes.len(), 1);
        assert_eq!(indexes.get("t1").unwrap().session_id, "a");
    }

    #[test]
    fn active_helpers_see_submitted_and_working_tasks() {
        let mut indexes = Indexes::default();
        indexes.insert(task("t1", "s1"));
        indexes.insert(task("t2", "s2").with_status(TaskStatus::Completed));
        assert!(indexes.has_active(None));
        assert!(indexes.has_active(Some("s1")));
        assert!(!indexes.has_active(Some("s2")));
    }
}
