//! Structured async task management.
//!
//! The task manager gives every spawned task an identity, a status, a
//! cancellation token, and an abort handle. Spawning registers the task first,
//! so the manager never loses track of a running future: tasks stay listed until
//! they finish and are pruned by [`TaskManager::remove_finished`], and dropping
//! the manager aborts everything that is still registered. This replaces the
//! Python `openjiuwen.core.common.task_manager` singleton with Rust structured
//! concurrency while keeping the same observable model (statuses, groups, parent
//! chains, cascade cancellation, and status events).
//!
//! Panics inside a task are isolated: they mark the task `Failed` and surface as
//! an error from [`TaskHandle::join`] without poisoning the runtime.
//!
//! [`TaskManager::global`] is the documented process-global instance, matching
//! Python `get_task_manager()`. New code and tests should prefer
//! [`TaskManager::new`] so that shutdown of one manager cannot affect another.
//!
//! ```
//! use openjiuwen_common::task::{TaskManager, TaskStatus};
//!
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() {
//! let manager = TaskManager::new();
//! let handle = manager.spawn("example", async { 41 + 1 });
//! let id = handle.id().clone();
//! assert_eq!(handle.join().await.unwrap(), 42);
//! assert_eq!(manager.get(&id).unwrap().status, TaskStatus::Completed);
//! # }
//! ```

pub mod manager;
pub mod status;

pub use manager::{TaskHandle, TaskManager};
pub use status::{TaskId, TaskInfo, TaskOptions, TaskStatus};
