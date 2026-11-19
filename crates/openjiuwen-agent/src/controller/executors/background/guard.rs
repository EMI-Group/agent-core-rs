//! RAII cleanup for one driven run.
//!
//! [`TaskSchedulerLoop::drive`](super::TaskSchedulerLoop) runs inside a spawned
//! `tokio` task, so an unwind out of the user-supplied
//! [`TaskExecutor::execute`](super::TaskExecutor) — or out of any other step of
//! a driver — would otherwise leave the run in the in-flight map forever: the
//! concurrency gate keeps counting the stale entry and the completion sentinel
//! never fires again. [`RunGuard`] owns that cleanup, so the invariant holds
//! whether the driver returns normally, returns early, or unwinds.

use super::InFlight;
use crate::controller::schema::TaskStatus;
use crate::controller::task_manager::TaskManager;
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};
use tokio::sync::Notify;
use tracing::warn;

/// Reason recorded when a driver unwound without reporting an outcome.
pub(super) const DRIVER_PANIC_MESSAGE: &str = "task driver panicked";

/// Text of a caught panic payload.
///
/// A Rust panic payload is usually the `&'static str` of `panic!("…")` or the
/// `String` of a formatted message; anything else reports a fixed placeholder
/// so the task still carries a reason.
pub(super) fn panic_text(payload: &(dyn std::any::Any + Send)) -> &str {
    if let Some(text) = payload.downcast_ref::<&'static str>() {
        text
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text
    } else {
        "unknown panic payload"
    }
}

/// Cleanup guard of one driven run.
///
/// [`Drop`] always untracks the task from the in-flight map and wakes the scan
/// loop; when the run was not disarmed by the normal exit path it also marks a
/// still-running task [`TaskStatus::Failed`] with a panic reason. The guard
/// holds only [`Weak`] handles to the scheduler's tracking state, so a stopped
/// or dropped scheduler is never kept alive by a driver that is unwinding.
pub(super) struct RunGuard {
    /// Task store, used to record the failure of an unwound run.
    manager: TaskManager,
    /// In-flight map owned by the scheduler.
    in_flight: Weak<Mutex<BTreeMap<String, InFlight>>>,
    /// Wake-up of the scan loop, owned by the scheduler.
    wake: Weak<Notify>,
    /// Task this guard cleans up.
    task_id: String,
    /// Whether the run still owes a terminal state; cleared by the normal exit.
    armed: AtomicBool,
}

impl RunGuard {
    /// Builds the guard of `task_id` over the scheduler's shared handles.
    pub(super) fn new(
        manager: TaskManager,
        in_flight: &Arc<Mutex<BTreeMap<String, InFlight>>>,
        wake: &Arc<Notify>,
        task_id: impl Into<String>,
    ) -> Self {
        Self {
            manager,
            in_flight: Arc::downgrade(in_flight),
            wake: Arc::downgrade(wake),
            task_id: task_id.into(),
            armed: AtomicBool::new(true),
        }
    }

    /// Records that the run reached its own outcome, so dropping the guard only
    /// untracks and wakes.
    pub(super) fn disarm(&self) {
        self.armed.store(false, Ordering::SeqCst);
    }

    /// Cleanup shared by the normal and the unwinding path.
    fn release(&self) {
        if let Some(in_flight) = self.in_flight.upgrade() {
            in_flight.lock().remove(&self.task_id);
        }
        if let Some(wake) = self.wake.upgrade() {
            wake.notify_one();
        }
        if self.armed.swap(false, Ordering::SeqCst) {
            self.fail_unwound_task();
        }
    }

    /// Fails a task that never reached a terminal status of its own.
    fn fail_unwound_task(&self) {
        let Some(task) = self.manager.task(&self.task_id) else {
            return;
        };
        if task.status.is_terminal() {
            // The run already recorded its outcome (or a pause/cancel did).
            return;
        }
        warn!(task_id = %self.task_id, "task driver unwound; failing the task");
        self.manager.update_task_status(
            &self.task_id,
            TaskStatus::Failed,
            false,
            false,
            Some(DRIVER_PANIC_MESSAGE.to_owned()),
        );
    }
}

impl Drop for RunGuard {
    fn drop(&mut self) {
        // A panic that escapes a `Drop` during unwinding aborts the process, so
        // the cleanup itself must never propagate one.
        let _ = std::panic::catch_unwind(AssertUnwindSafe(|| self.release()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controller::schema::Task;
    use openjiuwen_session::{AgentSession, Session};
    use std::time::Duration;
    use tokio_util::sync::CancellationToken;

    /// One tracking entry for `task_id`, as `spawn_driver` inserts it.
    fn in_flight_entry(task_id: &str) -> InFlight {
        InFlight {
            task_id: task_id.to_owned(),
            token: CancellationToken::new(),
            executor: None,
            session: Arc::new(AgentSession::new("s1")) as Arc<dyn Session>,
            handle: None,
        }
    }

    /// Manager holding `task_id` with `status`.
    fn manager_with(task_id: &str, status: TaskStatus) -> TaskManager {
        let manager = TaskManager::new();
        manager
            .add_task(Task::new("s1", task_id, "test").with_status(status))
            .expect("task is added");
        manager
    }

    #[tokio::test]
    async fn drop_untracks_wakes_and_fails_a_running_task() {
        let manager = manager_with("t1", TaskStatus::Working);
        let in_flight = Arc::new(Mutex::new(BTreeMap::new()));
        in_flight
            .lock()
            .insert("t1".to_owned(), in_flight_entry("t1"));
        let wake = Arc::new(Notify::new());

        drop(RunGuard::new(manager.clone(), &in_flight, &wake, "t1"));

        assert!(in_flight.lock().is_empty(), "the run is untracked");
        let task = manager.task("t1").expect("task is stored");
        assert_eq!(task.status, TaskStatus::Failed);
        assert_eq!(task.error_message.as_deref(), Some(DRIVER_PANIC_MESSAGE));
        assert!(
            tokio::time::timeout(Duration::from_secs(1), wake.notified())
                .await
                .is_ok(),
            "the scan loop is woken"
        );
    }

    #[tokio::test]
    async fn unwinding_driver_still_untracks_its_run() {
        let manager = manager_with("t1", TaskStatus::Working);
        let in_flight = Arc::new(Mutex::new(BTreeMap::new()));
        in_flight
            .lock()
            .insert("t1".to_owned(), in_flight_entry("t1"));
        let wake = Arc::new(Notify::new());

        let unwound = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let _guard = RunGuard::new(manager.clone(), &in_flight, &wake, "t1");
            panic!("driver blew up");
        }));

        assert!(unwound.is_err());
        assert!(in_flight.lock().is_empty());
        assert_eq!(
            manager.task("t1").expect("task is stored").status,
            TaskStatus::Failed
        );
    }

    #[tokio::test]
    async fn disarmed_guard_leaves_the_recorded_status_alone() {
        let manager = manager_with("t1", TaskStatus::Completed);
        let in_flight = Arc::new(Mutex::new(BTreeMap::new()));
        in_flight
            .lock()
            .insert("t1".to_owned(), in_flight_entry("t1"));
        let wake = Arc::new(Notify::new());

        let guard = RunGuard::new(manager.clone(), &in_flight, &wake, "t1");
        guard.disarm();
        drop(guard);

        assert!(in_flight.lock().is_empty());
        assert_eq!(
            manager.task("t1").expect("task is stored").status,
            TaskStatus::Completed
        );
    }

    #[tokio::test]
    async fn an_armed_guard_leaves_a_terminal_task_alone() {
        // A run that recorded its outcome and then unwound before `finish` keeps
        // the status it wrote.
        let manager = manager_with("t1", TaskStatus::Completed);
        let in_flight = Arc::new(Mutex::new(BTreeMap::new()));
        let wake = Arc::new(Notify::new());

        drop(RunGuard::new(manager.clone(), &in_flight, &wake, "t1"));

        assert_eq!(
            manager.task("t1").expect("task is stored").status,
            TaskStatus::Completed
        );
    }

    #[test]
    fn panic_text_reads_static_and_formatted_payloads() {
        let static_payload = std::panic::catch_unwind(|| panic!("boom")).expect_err("panics");
        assert_eq!(panic_text(static_payload.as_ref()), "boom");

        let formatted = std::panic::catch_unwind(|| panic!("boom {}", 2)).expect_err("panics");
        assert_eq!(panic_text(formatted.as_ref()), "boom 2");

        let other = std::panic::catch_unwind(|| std::panic::panic_any(7u8)).expect_err("panics");
        assert_eq!(panic_text(other.as_ref()), "unknown panic payload");
    }
}
