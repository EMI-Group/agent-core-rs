//! Persisted task-manager state: one session's tasks plus the ordering sequence.
//!
//! The wire shape is the one the controller already stores in the session
//! (`{"tasks": [<Task>, ...], "next_seq": <n>}`), so a snapshot of the shared
//! manager round-trips through the same session state the turn loop writes.
//! `created_seq` is a Rust-only ordering key and never appears on the wire, so
//! [`TaskManager::restore_session`](super::TaskManager::restore_session)
//! reassigns it in stored order.
use crate::controller::schema::Task;
use serde::{Deserialize, Serialize};

/// A snapshot of the tasks one session owns.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TaskManagerState {
    /// The session's tasks in creation order.
    #[serde(default)]
    pub tasks: Vec<Task>,
    /// The manager's next ordering sequence number.
    #[serde(default)]
    pub next_seq: u64,
}

impl TaskManagerState {
    /// Builds a state from `tasks` and the next ordering sequence.
    #[must_use]
    pub fn new(tasks: Vec<Task>, next_seq: u64) -> Self {
        Self { tasks, next_seq }
    }

    /// Number of tasks in the snapshot.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    /// Whether the snapshot holds no task.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    /// The ids of the snapshot's tasks, in creation order.
    #[must_use]
    pub fn task_ids(&self) -> Vec<String> {
        self.tasks.iter().map(|task| task.task_id.clone()).collect()
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

    #[test]
    fn wire_shape_matches_the_controller_state() {
        let state = TaskManagerState::new(vec![task("t1", "s1")], 2);
        let wire = serde_json::to_value(&state).unwrap();
        assert_eq!(wire["next_seq"], json!(2));
        assert_eq!(wire["tasks"][0]["task_id"], json!("t1"));
        assert_eq!(wire["tasks"][0]["status"], json!("submitted"));
        assert!(wire["tasks"][0].get("created_seq").is_none());
        assert_eq!(wire.as_object().unwrap().len(), 2);
    }

    #[test]
    fn empty_state_uses_both_defaults() {
        let parsed: TaskManagerState = serde_json::from_value(json!({})).unwrap();
        assert_eq!(parsed, TaskManagerState::default());
        assert!(parsed.is_empty());
        assert_eq!(parsed.len(), 0);
        assert_eq!(parsed.task_ids(), Vec::<String>::new());
        assert_eq!(parsed.next_seq, 0);
    }

    #[test]
    fn unknown_fields_are_tolerated_and_order_is_preserved() {
        let parsed: TaskManagerState = serde_json::from_value(json!({
            "tasks": [
                {"session_id": "s1", "task_id": "t2", "task_type": "agent", "status": "working"},
                {"session_id": "s1", "task_id": "t1", "task_type": "agent", "status": "completed"}
            ],
            "next_seq": 7,
            "priority_index": {"1": ["t2", "t1"]}
        }))
        .unwrap();
        assert_eq!(parsed.next_seq, 7);
        assert_eq!(parsed.task_ids(), vec!["t2".to_owned(), "t1".to_owned()]);
        assert_eq!(parsed.tasks[0].status, TaskStatus::Working);
        assert_eq!(parsed.tasks[1].status, TaskStatus::Completed);
        assert_eq!(parsed.tasks[0].created_seq, 0);
    }

    #[test]
    fn state_round_trips_through_serde_json() {
        let state = TaskManagerState::new(
            vec![
                task("t1", "s1")
                    .with_priority(3)
                    .with_metadata("user_id", "u1"),
                task("t2", "s1").with_parent("t1"),
            ],
            9,
        );
        let wire = serde_json::to_string(&state).unwrap();
        let parsed: TaskManagerState = serde_json::from_str(&wire).unwrap();
        assert_eq!(parsed.tasks, state.tasks);
        assert_eq!(parsed.next_seq, 9);
    }
}
