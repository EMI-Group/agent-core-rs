//! Background task scheduler: scan, concurrency gate, pause/cancel, sentinel.
//!
//! Ports the scheduling half of
//! `openjiuwen/core/controller/modules/task_scheduler.py`: [`TaskSchedulerLoop`]
//! periodically scans the [`TaskManager`] for [`TaskStatus::Submitted`] tasks,
//! runs each one on the [`TaskExecutor`] registered for its `task_type` in a
//! spawned driver, keeps at most
//! [`max_concurrent_tasks`](ControllerConfig::max_concurrent_tasks) of them
//! alive, and drives Python's two-phase pause/cancel order through the executor
//! hooks plus [`TaskContext::cancellation`]. When a session runs out of active
//! tasks it writes Python's `all_tasks_processed` sentinel to that session's
//! stream.
//!
//! The loop is a plain `tokio` task: [`start`](TaskSchedulerLoop::start) spawns
//! it, [`stop`](TaskSchedulerLoop::stop) cancels it together with every
//! in-flight run and awaits them, and [`scan_once`](TaskSchedulerLoop::scan_once)
//! performs one pass without a loop at all (which is what the tests drive). A
//! new scan happens on three triggers: the periodic
//! [`schedule_interval_ms`](ControllerConfig::schedule_interval_ms) timer, an
//! explicit [`wake`](TaskSchedulerLoop::wake) (a `Notify`, so a wake that
//! arrives before the loop waits is not lost), and the
//! [`TaskManager`] task-submitted callback that `start` installs through a
//! [`Weak`] reference (no reference cycle).
//!
//! # Differences from Python
//!
//! - There is no `asyncio` loop, event queue, or event publication: the loop
//!   owns no event queue, and a task result is recorded on the task itself
//!   (status, error message, and one output chunk) instead of being turned into
//!   a published `TaskCompletionEvent`/`TaskInteractionEvent`/`TaskFailedEvent`.
//! - `max_concurrent_tasks = 0` means unlimited here. Python compares
//!   `len(self._running_tasks) >= self._config.max_concurrent_tasks` without
//!   special-casing `0`, which silently blocks every task.
//! - The scan order is deterministic: submitted tasks are admitted by
//!   `priority` descending, then [`Task::created_seq`] ascending, then
//!   `task_id`; Python iterates the task manager's selection, whose order is
//!   unspecified.
//! - [`TaskStatus::Waiting`] is honoured as a dependency gate: a waiting task
//!   whose `metadata["depends_on"]` array is all [`TaskStatus::Completed`]
//!   flips to [`TaskStatus::Submitted`] at the top of a scan. Python never
//!   promotes a waiting task, so the status was previously unreachable.
//! - The task-submitted wake-up callback is installed by
//!   [`start`](TaskSchedulerLoop::start) and holds a [`Weak`] handle, whereas
//!   Python wires the scheduler into the task manager at construction.
//! - A completed run appends `TaskResult::data` as one
//!   [`ControllerOutputChunk`] on the task (the same way the controller's turn
//!   loop records a signal) instead of streaming the executor's chunk
//!   iterator to the session. When
//!   [`expose_reasoning`](ControllerConfig::expose_reasoning) is on and the
//!   executor captured reasoning, a [`OutputKind::Reasoning`] chunk is appended
//!   before that terminal completion/interaction/failure chunk.
//! - The driver builds its [`TaskContext`] from the live configuration: the
//!   `expose_reasoning` flag is read next to `task_timeout_ms` before any
//!   `.await`, and the history stays empty because a scheduled run has no
//!   dispatching turn.
//! - The loop is single-use: [`stop`](TaskSchedulerLoop::stop) cancels the
//!   loop token, so a stopped scheduler must be rebuilt (not restarted) to run
//!   again.
//! - A panicking executor never unwinds the driver: the panic is recorded as a
//!   failed task (`task executor panicked: <payload>`) and the run's
//!   `RunGuard` still releases its in-flight slot, so a panic can neither leak
//!   concurrency capacity nor silence the completion sentinel.
use super::{TaskContext, TaskExecutor, TaskExecutorRegistry};
use crate::controller::config::ControllerConfig;
use crate::controller::schema::{
    ControllerOutputChunk, ControllerOutputPayload, DataFrame, OutputKind, Task, TaskResult,
    TaskStatus,
};
use crate::controller::task_manager::{TaskFilter, TaskManager};
use futures::FutureExt;
use openjiuwen_session::Session;
use parking_lot::{Mutex, RwLock};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

mod guard;

use guard::{panic_text, RunGuard};

/// Text frame the completion sentinel carries (Python's literal).
pub const COMPLETION_TEXT: &str = "All tasks have been successfully processed";

/// Task metadata key holding the dependency ids of a waiting task.
pub const DEPENDS_ON_KEY: &str = "depends_on";

/// Poll interval of [`TaskSchedulerLoop::wait_until_idle`].
const IDLE_POLL_INTERVAL: Duration = Duration::from_millis(5);

/// Upper bound of [`TaskSchedulerLoop::wait_until_idle`].
const IDLE_TIMEOUT: Duration = Duration::from_secs(5);

/// Resolves the session a task belongs to.
///
/// The scheduler never owns sessions: a controller (or a host) hands it one
/// lookup, and a task whose session cannot be resolved is skipped with a
/// warning instead of being failed.
pub type SessionProvider = Arc<dyn Fn(&str) -> Option<Arc<dyn Session>> + Send + Sync>;

/// One run the scheduler admitted and has not yet cleaned up.
///
/// The driver fills [`executor`](Self::executor) once it has resolved the
/// registry entry, so a pause/cancel request that arrives in the first
/// microseconds of a run still finds the task in flight (and simply skips the
/// executor hooks, like Python does with a `None` executor).
struct InFlight {
    /// Task id, repeated for diagnostics (the map key is the same value).
    task_id: String,
    /// Token the driver selects on; cancelled to stop the run.
    token: CancellationToken,
    /// Executor of this run, `None` until the driver claimed it.
    executor: Option<Arc<dyn TaskExecutor>>,
    /// Session the run belongs to.
    session: Arc<dyn Session>,
    /// Join handle of the driver; taken (never held) across pause/cancel awaits.
    handle: Option<JoinHandle<()>>,
}

/// Snapshot of an in-flight run taken before the executor hooks are awaited.
struct InFlightSnapshot {
    /// Executor of the run, when the driver already resolved it.
    executor: Option<Arc<dyn TaskExecutor>>,
    /// Session the run belongs to.
    session: Arc<dyn Session>,
    /// Token the driver selects on.
    token: CancellationToken,
}

/// The background half of Python's `TaskScheduler`.
///
/// A loop is a cheap handle (`Arc` inside every field), so a controller keeps
/// one while the loop task, the drivers, and the pause/cancel paths share it.
/// Construct it with [`new`](Self::new), run it with
/// [`start`](Self::start) (or one pass with [`scan_once`](Self::scan_once)),
/// and shut it down with [`stop`](Self::stop).
#[derive(Clone)]
pub struct TaskSchedulerLoop {
    /// The shared task store the scan reads and the drivers write.
    manager: TaskManager,
    /// Executors keyed by task type; readable between scans.
    registry: Arc<RwLock<TaskExecutorRegistry>>,
    /// Live configuration (concurrency gate, interval, timeout, sentinel).
    config: Arc<RwLock<ControllerConfig>>,
    /// Id of the controller driving the dispatch, used in diagnostics.
    controller_id: String,
    /// Session lookup handed in by the controller.
    sessions: SessionProvider,
    /// Whether the loop task should keep scanning.
    running: Arc<AtomicBool>,
    /// Wake-up of the scan loop; `notify_one` keeps a permit, so no lost wakeups.
    wake: Arc<tokio::sync::Notify>,
    /// Stops the loop; cancelled by [`stop`](Self::stop).
    cancellation: CancellationToken,
    /// Every admitted run, keyed by task id.
    in_flight: Arc<Mutex<BTreeMap<String, InFlight>>>,
    /// Sessions the completion sentinel was already written to.
    signalled: Arc<Mutex<BTreeSet<String>>>,
    /// Handle of the spawned scan loop, taken by [`stop`](Self::stop).
    loop_handle: Arc<Mutex<Option<JoinHandle<()>>>>,
}

impl TaskSchedulerLoop {
    /// Builds a scheduler over `manager` that dispatches through `registry`
    /// under `config`, for the controller named `controller_id`, resolving
    /// sessions through `sessions`.
    ///
    /// The returned loop is idle: nothing runs until
    /// [`start`](Self::start) or [`scan_once`](Self::scan_once) is called.
    #[must_use]
    pub fn new(
        manager: TaskManager,
        registry: Arc<RwLock<TaskExecutorRegistry>>,
        config: Arc<RwLock<ControllerConfig>>,
        controller_id: impl Into<String>,
        sessions: SessionProvider,
    ) -> Self {
        Self {
            manager,
            registry,
            config,
            controller_id: controller_id.into(),
            sessions,
            running: Arc::new(AtomicBool::new(false)),
            wake: Arc::new(tokio::sync::Notify::new()),
            cancellation: CancellationToken::new(),
            in_flight: Arc::new(Mutex::new(BTreeMap::new())),
            signalled: Arc::new(Mutex::new(BTreeSet::new())),
            loop_handle: Arc::new(Mutex::new(None)),
        }
    }

    /// Returns the task store this scheduler drives.
    #[must_use]
    pub fn manager(&self) -> &TaskManager {
        &self.manager
    }

    /// Returns the live configuration handle.
    #[must_use]
    pub fn config(&self) -> Arc<RwLock<ControllerConfig>> {
        Arc::clone(&self.config)
    }

    /// Returns the executor registry handle.
    #[must_use]
    pub fn registry(&self) -> Arc<RwLock<TaskExecutorRegistry>> {
        Arc::clone(&self.registry)
    }

    /// Starts the scan loop, returning `false` when it is already running.
    ///
    /// The first start also installs the task-submitted wake-up callback on the
    /// [`TaskManager`]; the callback holds a [`Weak`] handle, so it never keeps
    /// the scheduler alive.
    ///
    /// # Panics
    ///
    /// Panics when called outside a `tokio` runtime (the loop is a spawned
    /// task).
    pub fn start(self: &Arc<Self>) -> bool {
        if self.running.swap(true, Ordering::SeqCst) {
            return false;
        }
        let weak: Weak<Self> = Arc::downgrade(self);
        self.manager.set_on_task_submitted(Arc::new(move || {
            if let Some(scheduler) = weak.upgrade() {
                scheduler.wake();
            }
        }));
        let loop_handle = Arc::clone(self);
        let handle = tokio::spawn(async move { loop_handle.run_loop().await });
        *self.loop_handle.lock() = Some(handle);
        debug!(controller_id = %self.controller_id, "task scheduler loop starting");
        true
    }

    /// Whether the scan loop is currently running.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Stops the scan loop and every in-flight run, then clears the tracking.
    ///
    /// The loop is cancelled and joined first, so no new run can be admitted
    /// while the in-flight ones are stopped; then each in-flight token is
    /// cancelled and the drivers are awaited unlocked, so a cooperative executor
    /// stops promptly. Calling `stop` on a scheduler that never started, or
    /// twice, is harmless.
    ///
    /// A stopped scheduler is not restartable ([`start`](Self::start) would
    /// spawn a loop whose token is already cancelled); build a new one.
    pub async fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        self.cancellation.cancel();
        self.wake.notify_one();
        let loop_handle = self.loop_handle.lock().take();
        if let Some(handle) = loop_handle {
            if let Err(error) = handle.await {
                debug!(%error, "task scheduler loop join failed");
            }
        }
        let mut runs = Vec::new();
        {
            let mut in_flight = self.in_flight.lock();
            for entry in in_flight.values_mut() {
                debug!(task_id = %entry.task_id, "task scheduler stopping run");
                entry.token.cancel();
                if let Some(handle) = entry.handle.take() {
                    runs.push(handle);
                }
            }
        }
        for handle in runs {
            if let Err(error) = handle.await {
                debug!(%error, "task scheduler driver join failed");
            }
        }
        self.in_flight.lock().clear();
        info!(controller_id = %self.controller_id, "task scheduler loop stopped");
    }

    /// Wakes the scan loop for an immediate pass.
    ///
    /// [`tokio::sync::Notify`] keeps a permit when no waiter is parked, so a
    /// wake that arrives while the loop is scanning still triggers the next
    /// iteration instead of being dropped.
    pub fn wake(&self) {
        self.wake.notify_one();
    }

    /// Sorted ids of the runs currently in flight.
    #[must_use]
    pub fn in_flight(&self) -> Vec<String> {
        self.in_flight.lock().keys().cloned().collect()
    }

    /// Performs one scheduling pass.
    ///
    /// The pass promotes ready waiting tasks, admits submitted tasks in
    /// `priority`/`created_seq` order until
    /// [`max_concurrent_tasks`](ControllerConfig::max_concurrent_tasks) is
    /// reached (`0` = unlimited), claims each admitted task (the
    /// double-execution guard) and spawns its driver. A task whose session
    /// cannot be resolved, and a task whose claim loses a race, are skipped
    /// with a warning.
    ///
    /// Nothing here returns an error: a failed scan is logged and swallowed so
    /// the loop keeps running.
    ///
    /// # Panics
    ///
    /// Panics when a task is admitted outside a `tokio` runtime (drivers are
    /// spawned tasks).
    pub async fn scan_once(&self) {
        // One yield lets a driver that finished since the previous pass
        // untrack itself before this pass counts the concurrency budget, so a
        // freed slot is reusable without waiting for the next timer tick.
        tokio::task::yield_now().await;
        self.promote_dependents();
        let submitted = match self
            .manager
            .get_task(Some(&TaskFilter::by_status(TaskStatus::Submitted)))
        {
            Ok(tasks) => tasks,
            Err(error) => {
                warn!(controller_id = %self.controller_id, %error, "task scheduler scan failed");
                return;
            }
        };
        if submitted.is_empty() {
            return;
        }
        let mut ordered = submitted;
        ordered.sort_by(|left, right| {
            right
                .priority
                .cmp(&left.priority)
                .then_with(|| left.created_seq.cmp(&right.created_seq))
                .then_with(|| left.task_id.cmp(&right.task_id))
        });
        let config = self.config.read().clone();
        for task in ordered {
            if config.max_concurrent_tasks != 0
                && self.in_flight.lock().len() >= config.max_concurrent_tasks
            {
                debug!(
                    controller_id = %self.controller_id,
                    limit = config.max_concurrent_tasks,
                    "task scheduler reached the concurrency limit"
                );
                break;
            }
            if self.in_flight.lock().contains_key(&task.task_id) {
                continue;
            }
            let Some(session) = (self.sessions)(&task.session_id) else {
                warn!(
                    task_id = %task.task_id,
                    session_id = %task.session_id,
                    "task scheduler: unknown session, skipping task"
                );
                continue;
            };
            if !self.manager.claim_task(&task.task_id) {
                debug!(
                    task_id = %task.task_id,
                    "task scheduler: task changed status before it could be claimed"
                );
                continue;
            }
            self.spawn_driver(task, session);
        }
    }

    /// Waits until no run is in flight and no task is submitted or working.
    ///
    /// The wait is bounded (5 seconds) and polls, so it never hangs a caller
    /// when a task stays un-runnable (for example one whose session is
    /// unknown).
    pub async fn wait_until_idle(&self) {
        let deadline = tokio::time::Instant::now() + IDLE_TIMEOUT;
        loop {
            let busy = !self.in_flight.lock().is_empty() || self.manager.has_active_tasks();
            if !busy {
                return;
            }
            if tokio::time::Instant::now() >= deadline {
                warn!(
                    controller_id = %self.controller_id,
                    "task scheduler did not reach idle within the wait budget"
                );
                return;
            }
            tokio::time::sleep(IDLE_POLL_INTERVAL).await;
        }
    }

    /// Pauses a running task, returning whether the pause was taken.
    ///
    /// Python's order is preserved: the executor's
    /// [`can_pause`](TaskExecutor::can_pause) decides, [`pause`](
    /// TaskExecutor::pause) runs, the run is re-checked (a task that finished in
    /// the meantime is not paused), the token is cancelled, the driver is
    /// awaited, and only then is the status set to [`TaskStatus::Paused`].
    ///
    /// Returns `false` when the task is unknown, not in flight, cannot be
    /// paused (the reason is logged), or finished while it was being paused.
    pub async fn pause_task(&self, task_id: &str) -> bool {
        let Some(task) = self.manager.task(task_id) else {
            warn!(task_id, "task scheduler pause: task not found");
            return false;
        };
        let Some(snapshot) = self.snapshot_in_flight(task_id) else {
            debug!(task_id, "task scheduler pause: task is not running");
            return false;
        };
        let ctx = TaskContext::new(Some(Arc::clone(&snapshot.session)), &self.controller_id)
            .with_cancellation(snapshot.token.clone());
        if let Some(executor) = &snapshot.executor {
            let (can_pause, reason) = executor.can_pause(&task, &ctx).await;
            if !can_pause {
                warn!(task_id, %reason, "task scheduler pause: executor refused");
                return false;
            }
            let _ = executor.pause(&task, &ctx).await;
        }
        let Some((token, handle)) = self.take_run(task_id) else {
            debug!(task_id, "task scheduler pause: task finished while pausing");
            return false;
        };
        token.cancel();
        if let Some(handle) = handle {
            if let Err(error) = handle.await {
                debug!(task_id, %error, "task scheduler pause: driver join failed");
            }
        }
        self.untrack(task_id);
        self.manager
            .update_task_status(task_id, TaskStatus::Paused, false, false, None);
        info!(task_id, "task scheduler paused task");
        true
    }

    /// Cancels a task, returning whether the cancel was taken.
    ///
    /// A [`TaskStatus::Submitted`] task is cancelled without ever running, a
    /// task that already reached a terminal status is reported as cancelled
    /// (idempotent), and a running task goes through the executor's
    /// [`can_cancel`](TaskExecutor::can_cancel)/[`cancel`](
    /// TaskExecutor::cancel) hooks in Python's two-phase order before its token
    /// is cancelled, its driver is awaited, and its status becomes
    /// [`TaskStatus::Canceled`].
    ///
    /// Returns `false` when the task is unknown, cannot be cancelled (the
    /// reason is logged), is neither submitted nor in flight, or finished while
    /// it was being cancelled.
    pub async fn cancel_task(&self, task_id: &str) -> bool {
        let Some(task) = self.manager.task(task_id) else {
            warn!(task_id, "task scheduler cancel: task not found");
            return false;
        };
        if task.status == TaskStatus::Submitted {
            self.manager
                .update_task_status(task_id, TaskStatus::Canceled, false, false, None);
            debug!(task_id, "task scheduler cancelled a submitted task");
            return true;
        }
        if task.status.is_terminal() {
            debug!(
                task_id,
                status = task.status.as_str(),
                "task scheduler cancel is idempotent"
            );
            return true;
        }
        let Some(snapshot) = self.snapshot_in_flight(task_id) else {
            debug!(task_id, "task scheduler cancel: task is not running");
            return false;
        };
        let ctx = TaskContext::new(Some(Arc::clone(&snapshot.session)), &self.controller_id)
            .with_cancellation(snapshot.token.clone());
        if let Some(executor) = &snapshot.executor {
            let (can_cancel, reason) = executor.can_cancel(&task, &ctx).await;
            if !can_cancel {
                warn!(task_id, %reason, "task scheduler cancel: executor refused");
                return false;
            }
            let _ = executor.cancel(&task, &ctx).await;
        }
        let Some((token, handle)) = self.take_run(task_id) else {
            debug!(
                task_id,
                "task scheduler cancel: task finished while cancelling"
            );
            return false;
        };
        token.cancel();
        if let Some(handle) = handle {
            if let Err(error) = handle.await {
                debug!(task_id, %error, "task scheduler cancel: driver join failed");
            }
        }
        self.untrack(task_id);
        self.manager
            .update_task_status(task_id, TaskStatus::Canceled, false, false, None);
        info!(task_id, "task scheduler cancelled task");
        true
    }

    /// Writes the `all_tasks_processed` sentinel to `session_id` when that
    /// session has no active task left.
    ///
    /// The sentinel is written at most once per idle stretch: a session that is
    /// still busy clears the marker (a new idle stretch begins), and admitting a
    /// task clears it too. A suppressed sentinel
    /// ([`suppress_completion_signal`](ControllerConfig::suppress_completion_signal)),
    /// a session without active-task bookkeeping, and a write failure are all
    /// silent in the sense that no error escapes: hosts call this from the
    /// driver's finalizer, where a failure must not kill the loop.
    pub async fn ensure_completion_signal(&self, session_id: &str) {
        if self.config.read().suppress_completion_signal {
            return;
        }
        if self.manager.session_has_active_tasks(session_id) {
            self.signalled.lock().remove(session_id);
            return;
        }
        if !self.signalled.lock().insert(session_id.to_owned()) {
            return;
        }
        let Some(session) = (self.sessions)(session_id) else {
            warn!(
                session_id,
                "task scheduler: unknown session, no completion signal"
            );
            self.signalled.lock().remove(session_id);
            return;
        };
        let chunk = ControllerOutputChunk::last(
            0,
            ControllerOutputPayload::text(OutputKind::AllTasksProcessed, COMPLETION_TEXT),
        );
        if let Err(error) = session.write_stream(chunk.to_stream_item()).await {
            warn!(session_id, %error, "task scheduler: completion signal failed");
        }
    }

    /// Runs the scan loop until [`stop`](Self::stop) cancels it.
    async fn run_loop(self: Arc<Self>) {
        info!(controller_id = %self.controller_id, "task scheduler loop started");
        while self.running.load(Ordering::SeqCst) {
            self.scan_once().await;
            let interval = Duration::from_millis(self.config.read().schedule_interval_ms.max(1));
            tokio::select! {
                () = self.cancellation.cancelled() => break,
                () = self.wake.notified() => {}
                () = tokio::time::sleep(interval) => {}
            }
        }
        self.running.store(false, Ordering::SeqCst);
        info!(controller_id = %self.controller_id, "task scheduler loop exiting");
    }

    /// Promotes waiting tasks whose dependencies all completed.
    fn promote_dependents(&self) {
        let waiting = match self
            .manager
            .get_task(Some(&TaskFilter::by_status(TaskStatus::Waiting)))
        {
            Ok(tasks) => tasks,
            Err(error) => {
                warn!(controller_id = %self.controller_id, %error, "task scheduler dependency scan failed");
                return;
            }
        };
        for task in waiting {
            let Some(Value::Array(dependencies)) = task.metadata.get(DEPENDS_ON_KEY) else {
                continue;
            };
            let ready = dependencies
                .iter()
                .filter_map(Value::as_str)
                .all(|dependency| {
                    self.manager
                        .task(dependency)
                        .is_some_and(|task| task.status == TaskStatus::Completed)
                });
            if ready {
                self.manager.update_task_status(
                    &task.task_id,
                    TaskStatus::Submitted,
                    false,
                    false,
                    None,
                );
                debug!(
                    task_id = %task.task_id,
                    "task scheduler promoted a waiting task"
                );
            }
        }
    }

    /// Spawns the driver of an already claimed task.
    fn spawn_driver(&self, task: Task, session: Arc<dyn Session>) {
        let task_id = task.task_id.clone();
        let session_id = task.session_id.clone();
        let token = CancellationToken::new();
        let driver = self.clone();
        let driver_token = token.clone();
        // A new run opens a new idle stretch for its session.
        self.signalled.lock().remove(&session_id);
        {
            // The handle is registered while the map is locked, so a driver
            // that finishes immediately still finds (and removes) its entry.
            let mut in_flight = self.in_flight.lock();
            if in_flight.contains_key(&task_id) {
                return;
            }
            in_flight.insert(
                task_id.clone(),
                InFlight {
                    task_id: task_id.clone(),
                    token,
                    executor: None,
                    session: Arc::clone(&session),
                    handle: None,
                },
            );
            let handle = tokio::spawn(async move {
                driver.drive(task, session, driver_token).await;
            });
            if let Some(entry) = in_flight.get_mut(&task_id) {
                entry.handle = Some(handle);
            }
        }
        debug!(task_id, "task scheduler started task");
    }

    /// Runs one executor under the run's timeout, containing a panic.
    ///
    /// A user-supplied executor must never unwind the driver: its panic is
    /// turned into a failure message, and the driver's guard keeps the
    /// in-flight bookkeeping consistent either way.
    async fn execute_guarded(
        executor: &Arc<dyn TaskExecutor>,
        task: &Task,
        ctx: &TaskContext<'_>,
        timeout: Option<Duration>,
    ) -> std::result::Result<TaskResult, String> {
        let execution = async {
            match timeout {
                Some(limit) => {
                    match tokio::time::timeout(limit, executor.execute(task, ctx)).await {
                        Ok(outcome) => outcome.map_err(|error| error.message().into_owned()),
                        Err(_) => Err(format!(
                            "Task timeout after {} seconds",
                            limit.as_secs_f64()
                        )),
                    }
                }
                None => executor
                    .execute(task, ctx)
                    .await
                    .map_err(|error| error.message().into_owned()),
            }
        };
        match AssertUnwindSafe(execution).catch_unwind().await {
            Ok(outcome) => outcome,
            Err(payload) => Err(format!(
                "task executor panicked: {}",
                panic_text(payload.as_ref())
            )),
        }
    }

    /// Executes one claimed task and records its outcome on the store.
    async fn drive(&self, task: Task, session: Arc<dyn Session>, token: CancellationToken) {
        let task_id = task.task_id.clone();
        let session_id = task.session_id.clone();
        // Every exit path — returned, early, or unwinding — releases the run
        // through this guard, so a stale entry can never eat concurrency
        // capacity or suppress the completion sentinel forever.
        let guard = RunGuard::new(
            self.manager.clone(),
            &self.in_flight,
            &self.wake,
            task_id.clone(),
        );
        // The two knobs the run needs are read once, before any await: the
        // guard is never held across an `.await`.
        let (timeout, expose_reasoning) = {
            let config = self.config.read();
            (
                config.task_timeout_ms.map(Duration::from_millis),
                config.expose_reasoning,
            )
        };
        let ctx = TaskContext::new(Some(Arc::clone(&session)), &self.controller_id)
            .with_cancellation(token.clone())
            .with_expose_reasoning(expose_reasoning);
        let resolved = self.registry.read().get(&task.task_type);
        let Some(executor) = resolved else {
            let message = format!("task executor not found: {}", task.task_type);
            warn!(task_id, task_type = %task.task_type, "task executor not found");
            if !ctx.cancellation.is_cancelled() {
                self.manager.update_task_status(
                    &task_id,
                    TaskStatus::Failed,
                    false,
                    false,
                    Some(message),
                );
            }
            self.finish(&guard, &task_id, &session_id).await;
            return;
        };
        self.set_executor(&task_id, Arc::clone(&executor));
        // The failure text is Python's: an executor error reports its message
        // (not the `<kind>: <message>` rendering), and a timed-out run reports
        // `Task timeout after <seconds> seconds` verbatim.
        let outcome = Self::execute_guarded(&executor, &task, &ctx, timeout).await;
        if ctx.cancellation.is_cancelled() {
            // A pause/cancel that raced the finish owns the status.
            debug!(
                task_id,
                "task stopped by pause/cancel, leaving the status to the caller"
            );
            self.finish(&guard, &task_id, &session_id).await;
            return;
        }
        if let Some(recorded) = self.recorded_decision(&task_id) {
            // A pause/cancel recorded while the run was in flight owns the
            // status too: an executor that ignores its cancellation token must
            // not overwrite the decision with `Completed`.
            debug!(
                task_id,
                status = recorded.as_str(),
                "task was paused or cancelled during the run; keeping the recorded decision"
            );
            self.finish(&guard, &task_id, &session_id).await;
            return;
        }
        let result = match outcome {
            Ok(result) => result,
            Err(message) => {
                warn!(task_id, %message, "task execution failed");
                self.manager.update_task_status(
                    &task_id,
                    TaskStatus::Failed,
                    false,
                    false,
                    Some(message),
                );
                self.finish(&guard, &task_id, &session_id).await;
                return;
            }
        };
        let status = match result.status {
            TaskStatus::Failed => TaskStatus::Failed,
            TaskStatus::InputRequired => TaskStatus::InputRequired,
            _ => TaskStatus::Completed,
        };
        let error_message = if status == TaskStatus::Failed {
            Some(
                result
                    .error
                    .clone()
                    .unwrap_or_else(|| format!("task {task_id} failed")),
            )
        } else {
            None
        };
        self.manager
            .update_task_status(&task_id, status, false, false, error_message);
        self.record_outputs(
            &task_id,
            status,
            &result.data,
            expose_reasoning,
            result.reasoning.as_deref(),
        );
        self.finish(&guard, &task_id, &session_id).await;
    }

    /// The pause/cancel a caller recorded on the store during a run, if any.
    ///
    /// Only the two statuses a caller can impose on a running task are
    /// reported; every other recorded status is the driver's own to write.
    fn recorded_decision(&self, task_id: &str) -> Option<TaskStatus> {
        let status = self.manager.task(task_id)?.status;
        matches!(status, TaskStatus::Paused | TaskStatus::Canceled).then_some(status)
    }
    /// Cleanup every driver runs: untrack, wake the loop, signal completion.
    ///
    /// Disarming the run's guard first keeps its [`Drop`] from failing a task
    /// whose outcome (or whose pause/cancel) is already recorded, while the
    /// untrack/wake it repeats on drop stay harmless.
    async fn finish(&self, guard: &RunGuard, task_id: &str, session_id: &str) {
        guard.disarm();
        self.untrack(task_id);
        self.wake();
        self.ensure_completion_signal(session_id).await;
    }

    /// Appends the result frames of `task_id` as output chunks.
    ///
    /// Mirrors `base/turn.rs::record_signal`, which folds a task signal into the
    /// persisted task the same way. When `expose_reasoning` is on and the
    /// executor reported non-empty `reasoning`, a [`OutputKind::Reasoning`]
    /// chunk is appended first and the completion/interaction/failure chunk
    /// (always the last one) after it, so the reasoning never displaces the
    /// terminal frame. With the flag off, reasoning is dropped instead of
    /// appearing in a frame.
    fn record_outputs(
        &self,
        task_id: &str,
        status: TaskStatus,
        data: &[DataFrame],
        expose_reasoning: bool,
        reasoning: Option<&str>,
    ) {
        let reasoning = if expose_reasoning {
            reasoning.filter(|text| !text.is_empty())
        } else {
            None
        };
        if data.is_empty() && reasoning.is_none() {
            return;
        }
        let kind = match status {
            TaskStatus::Completed => OutputKind::TaskCompletion,
            TaskStatus::InputRequired => OutputKind::TaskInteraction,
            _ => OutputKind::TaskFailed,
        };
        let mut chunks = Vec::with_capacity(2);
        if let Some(reasoning) = reasoning {
            chunks.push(ControllerOutputChunk::new(
                0,
                ControllerOutputPayload::text(OutputKind::Reasoning, reasoning),
            ));
        }
        if !data.is_empty() {
            let index = i64::try_from(chunks.len()).unwrap_or(i64::MAX);
            chunks.push(ControllerOutputChunk::last(
                index,
                ControllerOutputPayload::new(kind, data.to_vec()),
            ));
        } else if let Some(chunk) = chunks.last_mut() {
            // Reasoning without result frames: the reasoning chunk is the only
            // and therefore terminal chunk of this run.
            chunk.last_chunk = true;
        }
        let Some(mut task) = self.manager.task(task_id) else {
            return;
        };
        if task.status != status {
            // A pause/cancel (or a later run) owns the status now: only append
            // when the stored task is still the one this run recorded.
            return;
        }
        task.outputs.extend(chunks);
        let _ = self.manager.update_task(task);
    }

    /// Records the executor of an in-flight run, once the driver resolved it.
    fn set_executor(&self, task_id: &str, executor: Arc<dyn TaskExecutor>) {
        if let Some(entry) = self.in_flight.lock().get_mut(task_id) {
            entry.executor = Some(executor);
        }
    }

    /// Copies out what a pause/cancel path needs, then releases the lock.
    fn snapshot_in_flight(&self, task_id: &str) -> Option<InFlightSnapshot> {
        let in_flight = self.in_flight.lock();
        let entry = in_flight.get(task_id)?;
        Some(InFlightSnapshot {
            executor: entry.executor.clone(),
            session: Arc::clone(&entry.session),
            token: entry.token.clone(),
        })
    }

    /// Re-checks membership and takes the driver handle out of the map.
    fn take_run(&self, task_id: &str) -> Option<(CancellationToken, Option<JoinHandle<()>>)> {
        let mut in_flight = self.in_flight.lock();
        let entry = in_flight.get_mut(task_id)?;
        Some((entry.token.clone(), entry.handle.take()))
    }

    /// Removes a finished run from the tracking map.
    fn untrack(&self, task_id: &str) {
        self.in_flight.lock().remove(task_id);
    }
}

impl std::fmt::Debug for TaskSchedulerLoop {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TaskSchedulerLoop")
            .field("controller_id", &self.controller_id)
            .field("running", &self.is_running())
            .field("in_flight", &self.in_flight())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests;
