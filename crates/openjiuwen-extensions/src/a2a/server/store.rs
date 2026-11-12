//! The task-store SPI and its process-local, in-memory implementation.

use std::collections::BTreeMap;
use std::sync::{Mutex, PoisonError};

use crate::a2a::types::Task;

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

/// A process-local [`TaskStore`].
///
/// Tasks are kept in a [`BTreeMap`], so [`TaskStore::all`] returns them in
/// ascending id order. Python's dict yields insertion order instead; the ids are
/// opaque, so the sorted order is a documented, deterministic substitute.
#[derive(Debug, Default)]
pub struct InMemoryTaskStore {
    tasks: Mutex<BTreeMap<String, Task>>,
}

impl InMemoryTaskStore {
    /// An empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl TaskStore for InMemoryTaskStore {
    fn get(&self, id: &str) -> Option<Task> {
        self.tasks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(id)
            .cloned()
    }

    fn put(&self, task: Task) {
        self.tasks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(task.id.clone(), task);
    }

    fn all(&self) -> Vec<Task> {
        self.tasks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .values()
            .cloned()
            .collect()
    }
}
