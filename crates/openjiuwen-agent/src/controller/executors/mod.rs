//! Task executors, the executor registry, and the task scheduler.
//!
//! Ports the executor contract of
//! `openjiuwen/core/controller/modules/task_scheduler.py`: the [`TaskExecutor`]
//! seam, the [`TaskContext`] one dispatch receives, the
//! [`TaskExecutorRegistry`] that keys executors by task type, the
//! [`TaskRouter`] that picks the task type of a turn, and the [`TaskScheduler`]
//! that turns a batch of [`Task`]s into [`TaskRun`]s.
//!
//! Unlike Python's `TaskScheduler`, which owns an event queue, a task manager,
//! and a background scheduling loop, [`TaskScheduler`] is a plain value: it
//! dispatches a caller-supplied batch of [`Task`]s to the executor registered
//! for each `task_type` and returns one [`TaskRun`] per task. Missing executors,
//! executor failures, and timeouts become failed results rather than errors, so
//! a batch always yields a run per dispatched task.
//!
//! The background half of Python's `TaskScheduler` (`schedule()`,
//! `execute_task()`, `pause_task()`, `cancel_task()`, and the completion
//! signal) lives in [`background`]: [`TaskSchedulerLoop`] scans the
//! `TaskManager` for submitted tasks, runs them concurrently under
//! `ControllerConfig::max_concurrent_tasks`, and drives pause/cancel through
//! the [`TaskExecutor`] hooks plus [`TaskContext::cancellation`]. Every
//! executor keeps working unchanged: the hooks default to "cannot pause, can
//! cancel, no-op actions".
//!
//! The concrete executors live in [`builtin`] and are re-exported here:
//! [`ControllerAgent`] runs one [`Agent`](crate::agent::Agent),
//! [`WorkflowController`] runs a registered workflow, and [`LLMController`] is
//! the LLM router.

use super::schema::{
    ControllerOutputChunk, ControllerOutputPayload, DataFrame, Intent, OutputKind, Task,
    TaskResult, TaskStatus,
};
use crate::controller::history::HistoryEntry;
use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_session::Session;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tracing::warn;

pub mod background;
pub mod builtin;

pub use background::{SessionProvider, TaskSchedulerLoop};
pub use builtin::{ControllerAgent, ExecutorInfo, LLMController, WorkflowController};

/// Task type of the built-in agent executor ([`ControllerAgent`]).
pub const AGENT_TASK_TYPE: &str = "agent";
/// Task type of the built-in workflow executor ([`WorkflowController`]).
pub const WORKFLOW_TASK_TYPE: &str = "workflow";
/// Task type of the built-in LLM router ([`LLMController`]).
pub const LLM_TASK_TYPE: &str = "llm";

/// Description reported by executors that have none.
const NO_DESCRIPTION: &str = "";
/// Description of [`ControllerAgent`].
const AGENT_DESCRIPTION: &str = "runs a controller agent";
/// Description of [`WorkflowController`].
const WORKFLOW_DESCRIPTION: &str = "runs a workflow";

/// Context handed to a [`TaskExecutor`] for one dispatch.
///
/// The session is optional because a controller may dispatch tasks that do not
/// touch session state; executors that need one call
/// [`require_session`](Self::require_session) to turn its absence into a typed
/// error.
///
/// [`cancellation`](Self::cancellation) is the stop signal of the dispatch: a
/// background scheduler ([`TaskSchedulerLoop`]) cancels it after a pause or a
/// cancel request, and once the scheduler stops, so a cooperative executor
/// selects on it and returns early. A dispatch with no scheduler around it
/// (`TaskScheduler::dispatch`, `BaseController`) installs a fresh,
/// never-cancelled token.
///
/// [`history`](Self::history) is the conversation the turn that dispatched the
/// task has recorded so far, oldest first and already trimmed to
/// [`ControllerConfig::history_limit`](crate::controller::ControllerConfig::history_limit),
/// so an executor can prompt a model with the same context intent recognition
/// saw. A background-scheduler run starts from no turn, so its history is
/// empty.
///
/// [`expose_reasoning`](Self::expose_reasoning) tells the executor whether to
/// capture the model's reasoning into [`TaskResult::reasoning`]; it is `false`
/// by default, which means no executor may capture anything.
#[derive(Clone, Default)]
pub struct TaskContext<'a> {
    /// Session shared by the dispatched tasks, when the caller has one.
    pub session: Option<Arc<dyn Session>>,
    /// Id of the controller driving the dispatch, used in diagnostics.
    pub controller_id: &'a str,
    /// Stop signal of the dispatch; never cancelled unless a scheduler asks.
    pub cancellation: CancellationToken,
    /// Conversation history of the dispatching turn, oldest first.
    ///
    /// Empty for a background-scheduler run: the scheduler has no turn.
    pub history: Vec<HistoryEntry>,
    /// Whether the executor should capture reasoning into
    /// [`TaskResult::reasoning`]; `false` captures nothing.
    pub expose_reasoning: bool,
}

impl<'a> TaskContext<'a> {
    /// Builds a context for `session` owned by `controller_id`.
    ///
    /// The history is empty and reasoning capture is off.
    #[must_use]
    pub fn new(session: Option<Arc<dyn Session>>, controller_id: &'a str) -> Self {
        Self {
            session,
            controller_id,
            cancellation: CancellationToken::new(),
            history: Vec::new(),
            expose_reasoning: false,
        }
    }

    /// Replaces the dispatch stop signal with `token`.
    ///
    /// The scheduler hands the same token to [`TaskExecutor::pause`] and
    /// [`TaskExecutor::cancel`] handling, so an executor that registered a
    /// listener on it observes the request it is being told about.
    #[must_use]
    pub fn with_cancellation(mut self, token: CancellationToken) -> Self {
        self.cancellation = token;
        self
    }

    /// Sets the conversation history of the dispatching turn (oldest first).
    #[must_use]
    pub fn with_history(mut self, history: Vec<HistoryEntry>) -> Self {
        self.history = history;
        self
    }

    /// Sets whether the executor should capture reasoning.
    #[must_use]
    pub fn with_expose_reasoning(mut self, expose: bool) -> Self {
        self.expose_reasoning = expose;
        self
    }

    /// Returns the session id, when a session is present.
    #[must_use]
    pub fn session_id(&self) -> Option<&str> {
        self.session.as_deref().map(Session::id)
    }

    /// Returns the session, or a [`Error::config`] naming the controller.
    ///
    /// # Errors
    ///
    /// Returns [`Error::config`] when no session is attached to the context.
    pub fn require_session(&self) -> Result<Arc<dyn Session>> {
        self.session.clone().ok_or_else(|| {
            let owner = if self.controller_id.is_empty() {
                "task execution"
            } else {
                self.controller_id
            };
            Error::config(format!("{owner} requires a session"))
        })
    }
}

/// Executes one task type.
///
/// Implementations are shared as `Arc<dyn TaskExecutor>` and registered in a
/// [`TaskExecutorRegistry`] under [`task_type`](Self::task_type). A returned
/// [`Err`] is recoverable from the scheduler's point of view: it becomes a
/// failed [`TaskRun`] rather than aborting the batch.
#[async_trait]
pub trait TaskExecutor: Send + Sync {
    /// Task type this executor handles; the registry key.
    fn task_type(&self) -> &str;

    /// Model-visible description used by the [`LLMController`] router.
    fn description(&self) -> &str {
        NO_DESCRIPTION
    }

    /// Executes `task`.
    ///
    /// # Errors
    ///
    /// Returns an error when the task cannot run at all (missing session,
    /// missing workflow, invalid input); the scheduler maps it onto a failed
    /// task result.
    async fn execute(&self, task: &Task, ctx: &TaskContext<'_>) -> Result<TaskResult>;

    /// Reports whether a running `task` can be paused.
    ///
    /// Returns `(true, "")` when the executor supports pausing, and
    /// `(false, reason)` otherwise. The default cannot pause
    /// (`"task executor does not support pausing"`), so an executor becomes
    /// pausable only by overriding this hook and [`pause`](Self::pause).
    async fn can_pause(&self, _task: &Task, _ctx: &TaskContext<'_>) -> (bool, String) {
        (false, "task executor does not support pausing".to_owned())
    }

    /// Pauses `task`, returning whether the executor took the request.
    ///
    /// The background scheduler calls this only after [`can_pause`](
    /// Self::can_pause) said yes, then cancels [`TaskContext::cancellation`]
    /// and records [`TaskStatus::Paused`] itself; the return value reports the
    /// executor's own bookkeeping, not the task status. The default is a no-op
    /// that returns `false`.
    async fn pause(&self, _task: &Task, _ctx: &TaskContext<'_>) -> bool {
        false
    }

    /// Reports whether a running `task` can be cancelled.
    ///
    /// The default can cancel (`(true, "")`), matching Python's permissive
    /// contract: the scheduler cancels [`TaskContext::cancellation`] and marks
    /// the task [`TaskStatus::Canceled`] regardless of what
    /// [`cancel`](Self::cancel) does.
    async fn can_cancel(&self, _task: &Task, _ctx: &TaskContext<'_>) -> (bool, String) {
        (true, String::new())
    }

    /// Cancels `task`, returning whether the executor took the request.
    ///
    /// The return value is informational (the scheduler records the cancelled
    /// status itself); the default is a no-op that returns `false`.
    async fn cancel(&self, _task: &Task, _ctx: &TaskContext<'_>) -> bool {
        false
    }
}

/// Executors keyed by task type.
#[derive(Clone, Default)]
pub struct TaskExecutorRegistry {
    executors: BTreeMap<String, Arc<dyn TaskExecutor>>,
}

impl TaskExecutorRegistry {
    /// Builds an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `executor` under its own [`task_type`](TaskExecutor::task_type).
    ///
    /// Returns the executor previously registered for that type, if any.
    pub fn add(&mut self, executor: Arc<dyn TaskExecutor>) -> Option<Arc<dyn TaskExecutor>> {
        let task_type = executor.task_type().to_owned();
        self.insert(&task_type, executor)
    }

    /// Registers `executor` under `task_type`, replacing any existing entry.
    ///
    /// Returns the executor previously registered for that type, if any.
    pub fn insert(
        &mut self,
        task_type: &str,
        executor: Arc<dyn TaskExecutor>,
    ) -> Option<Arc<dyn TaskExecutor>> {
        self.executors.insert(task_type.to_owned(), executor)
    }

    /// Removes the executor registered for `task_type`.
    pub fn remove(&mut self, task_type: &str) -> Option<Arc<dyn TaskExecutor>> {
        self.executors.remove(task_type)
    }

    /// Returns the executor registered for `task_type`.
    #[must_use]
    pub fn get(&self, task_type: &str) -> Option<Arc<dyn TaskExecutor>> {
        self.executors.get(task_type).map(Arc::clone)
    }

    /// Whether an executor is registered for `task_type`.
    #[must_use]
    pub fn contains(&self, task_type: &str) -> bool {
        self.executors.contains_key(task_type)
    }

    /// Registered task types, in sorted order.
    #[must_use]
    pub fn task_types(&self) -> Vec<String> {
        self.executors.keys().cloned().collect()
    }

    /// Number of registered executors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.executors.len()
    }

    /// Whether no executor is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.executors.is_empty()
    }
}

/// Picks the task type for a turn (the LLM router seam).
///
/// A router is optional: `Ok(None)` means "no opinion", which lets a
/// controller fall back to its configured default task type.
#[async_trait]
pub trait TaskRouter: Send + Sync {
    /// Router name (`"llm"` for [`LLMController`]), used in diagnostics.
    fn name(&self) -> &str;

    /// Selects the task type for `query`, considering `intents` when present.
    ///
    /// # Errors
    ///
    /// Returns an error when the router itself fails (for example a model
    /// transport error); a malformed model answer is not an error.
    async fn select(&self, query: &str, intents: &[Intent]) -> Result<Option<String>>;
}

/// Outcome of running one task.
#[derive(Debug)]
pub struct TaskRun {
    /// Task after the executor updated its status.
    pub task: Task,
    /// Executor result, or the error the scheduler recorded.
    pub result: Result<TaskResult>,
}

impl TaskRun {
    /// Result status; a scheduler-recorded error is [`TaskStatus::Failed`].
    #[must_use]
    pub fn status(&self) -> TaskStatus {
        match &self.result {
            Ok(result) => result.status,
            Err(_) => TaskStatus::Failed,
        }
    }

    /// Result text, falling back to the recorded error message and then `""`.
    #[must_use]
    pub fn text(&self) -> String {
        match &self.result {
            Ok(result) => {
                let text = result.display();
                if text.is_empty() {
                    result.error.clone().unwrap_or_default()
                } else {
                    text
                }
            }
            Err(error) => error.to_string(),
        }
    }

    /// Reasoning text the executor captured, when the run reported a result.
    ///
    /// Returns the executor's [`TaskResult::reasoning`] verbatim (empty text
    /// included) and `None` when the executor captured nothing or the
    /// scheduler recorded an [`Err`] instead of a result.
    #[must_use]
    pub fn reasoning(&self) -> Option<&str> {
        match &self.result {
            Ok(result) => result.reasoning.as_deref(),
            Err(_) => None,
        }
    }

    /// Builds the one controller chunk that reports this run.
    ///
    /// The chunk kind follows the status: `task_completion` for
    /// [`TaskStatus::Completed`], `task_interaction` for
    /// [`TaskStatus::InputRequired`], and `task_failed` otherwise. Its payload
    /// carries the result frames, or one text frame of [`text`](Self::text)
    /// when the result has none. `start_index` is the chunk index; it is always
    /// the last chunk of its own slice.
    #[must_use]
    pub fn frames(&self, start_index: i64) -> Vec<ControllerOutputChunk> {
        let kind = match self.status() {
            TaskStatus::Completed => OutputKind::TaskCompletion,
            TaskStatus::InputRequired => OutputKind::TaskInteraction,
            _ => OutputKind::TaskFailed,
        };
        let data = match &self.result {
            Ok(result) if !result.data.is_empty() => result.data.clone(),
            _ => vec![DataFrame::text(self.text())],
        };
        vec![ControllerOutputChunk::last(
            start_index,
            ControllerOutputPayload::new(kind, data),
        )]
    }
}

/// Dispatches tasks to registered executors.
///
/// Dispatch is sequential in the caller's task order and never returns [`Err`]:
/// a missing executor, an executor failure, or a timeout turns into a failed
/// [`TaskRun`] carrying the reason.
#[derive(Clone, Default)]
pub struct TaskScheduler {
    registry: TaskExecutorRegistry,
    max_tasks: usize,
    timeout: Option<Duration>,
}

impl TaskScheduler {
    /// Builds a scheduler over `registry` that runs at most `max_tasks` tasks
    /// per dispatch (`0` means unlimited) and has no timeout.
    #[must_use]
    pub fn new(registry: TaskExecutorRegistry, max_tasks: usize) -> Self {
        Self {
            registry,
            max_tasks,
            timeout: None,
        }
    }

    /// Sets the per-task timeout; `None` waits indefinitely.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.timeout = timeout;
        self
    }

    /// Sets the per-dispatch task budget (`0` means unlimited).
    #[must_use]
    pub fn with_max_tasks(mut self, max_tasks: usize) -> Self {
        self.max_tasks = max_tasks;
        self
    }

    /// Returns the executor registry.
    #[must_use]
    pub fn registry(&self) -> &TaskExecutorRegistry {
        &self.registry
    }

    /// Returns the executor registry for registration between dispatches.
    pub fn registry_mut(&mut self) -> &mut TaskExecutorRegistry {
        &mut self.registry
    }

    /// Returns the per-dispatch task budget (`0` means unlimited).
    #[must_use]
    pub const fn max_tasks(&self) -> usize {
        self.max_tasks
    }

    /// Returns the per-task timeout, when configured.
    #[must_use]
    pub const fn timeout(&self) -> Option<Duration> {
        self.timeout
    }

    /// Runs `tasks` in order, returning one run per dispatched task.
    ///
    /// Tasks that are already terminal are reported as-is. A task whose
    /// `task_type` has no executor is failed with
    /// `"task executor not found: <type>"`; an executor error or timeout is
    /// recorded on the task and returned as a failed result. No task is
    /// propagated as [`Err`].
    pub async fn dispatch(&self, tasks: Vec<Task>, ctx: &TaskContext<'_>) -> Vec<TaskRun> {
        let mut runs = Vec::with_capacity(tasks.len());
        for task in tasks {
            if self.max_tasks > 0 && runs.len() >= self.max_tasks {
                break;
            }
            runs.push(self.run_one(task, ctx).await);
        }
        runs
    }

    /// Executes one task, folding every failure into a [`TaskRun`].
    async fn run_one(&self, mut task: Task, ctx: &TaskContext<'_>) -> TaskRun {
        if task.is_terminal() {
            let result = TaskResult {
                status: task.status,
                error: task.error_message.clone(),
                ..TaskResult::default()
            };
            return TaskRun {
                task,
                result: Ok(result),
            };
        }
        let Some(executor) = self.registry.get(&task.task_type) else {
            let message = format!("task executor not found: {}", task.task_type);
            warn!(
                task_type = %task.task_type,
                task_id = %task.task_id,
                "task executor not found"
            );
            task.status = TaskStatus::Failed;
            task.error_message = Some(message.clone());
            return TaskRun {
                task,
                result: Ok(TaskResult::failed(message)),
            };
        };
        let outcome = match self.timeout {
            Some(limit) => match tokio::time::timeout(limit, executor.execute(&task, ctx)).await {
                Ok(result) => result,
                Err(_) => Err(Error::timeout(format!(
                    "task {} timed out after {}s",
                    task.task_id,
                    limit.as_secs_f64()
                ))),
            },
            None => executor.execute(&task, ctx).await,
        };
        match outcome {
            Ok(result) => {
                if matches!(
                    result.status,
                    TaskStatus::Completed
                        | TaskStatus::Failed
                        | TaskStatus::InputRequired
                        | TaskStatus::Paused
                        | TaskStatus::Canceled
                ) {
                    task.status = result.status;
                    if result.status == TaskStatus::Failed {
                        task.error_message = result
                            .error
                            .clone()
                            .or_else(|| Some(format!("task {} failed", task.task_id)));
                    }
                }
                TaskRun {
                    task,
                    result: Ok(result),
                }
            }
            Err(error) => {
                let message = error.to_string();
                task.status = TaskStatus::Failed;
                task.error_message = Some(message.clone());
                TaskRun {
                    task,
                    result: Ok(TaskResult::failed(message)),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_session::AgentSession;

    /// Description reported by [`EchoExecutor`].
    const ECHO_DESCRIPTION: &str = "echoes the task text";
    /// Task type of [`FailingExecutor`].
    const FAILING_TASK_TYPE: &str = "failing";
    /// Task type of [`SlowExecutor`].
    const SLOW_TASK_TYPE: &str = "slow";
    /// How long [`SlowExecutor`] sleeps before reporting `"late"`.
    ///
    /// 500 ms against the 10 ms test timeout is a 50x margin, so the timeout
    /// always wins even when the machine is saturated; the timeout drops the
    /// sleeping future, so the wide margin costs the test no runtime.
    const SLOW_EXECUTOR_HOLD: Duration = Duration::from_millis(500);

    /// Executor that echoes the task's input text.
    struct EchoExecutor;

    #[async_trait]
    impl TaskExecutor for EchoExecutor {
        fn task_type(&self) -> &str {
            AGENT_TASK_TYPE
        }

        fn description(&self) -> &str {
            ECHO_DESCRIPTION
        }

        async fn execute(&self, task: &Task, _ctx: &TaskContext<'_>) -> Result<TaskResult> {
            Ok(TaskResult::text(task.input_text().unwrap_or_default()))
        }
    }

    /// Executor that always fails.
    struct FailingExecutor;

    #[async_trait]
    impl TaskExecutor for FailingExecutor {
        fn task_type(&self) -> &str {
            FAILING_TASK_TYPE
        }

        async fn execute(&self, _task: &Task, _ctx: &TaskContext<'_>) -> Result<TaskResult> {
            Err(Error::validation("boom"))
        }
    }

    /// Executor that outlives the test's scheduler timeout by a wide margin.
    ///
    /// It sleeps for [`SLOW_EXECUTOR_HOLD`] against a 10 ms scheduler timeout,
    /// so a loaded machine cannot finish the sleep first; the timeout fires long
    /// before the sleep elapses and cancels it, which keeps the test fast.
    struct SlowExecutor;

    #[async_trait]
    impl TaskExecutor for SlowExecutor {
        fn task_type(&self) -> &str {
            SLOW_TASK_TYPE
        }

        async fn execute(&self, _task: &Task, _ctx: &TaskContext<'_>) -> Result<TaskResult> {
            tokio::time::sleep(SLOW_EXECUTOR_HOLD).await;
            Ok(TaskResult::text("late"))
        }
    }

    /// Executor registered under an arbitrary task type.
    struct NamedExecutor(&'static str);

    #[async_trait]
    impl TaskExecutor for NamedExecutor {
        fn task_type(&self) -> &str {
            self.0
        }

        async fn execute(&self, _task: &Task, _ctx: &TaskContext<'_>) -> Result<TaskResult> {
            Ok(TaskResult::text(self.0))
        }
    }

    fn task(task_id: &str, task_type: &str, description: &str) -> Task {
        Task::submitted("s1", task_id, task_type, description)
    }

    fn ctx() -> TaskContext<'static> {
        TaskContext::new(Some(Arc::new(AgentSession::new("s1"))), "ctl")
    }
    #[test]
    fn task_context_requires_a_session() {
        let empty = TaskContext::default();
        assert!(empty.session_id().is_none());
        let error = empty.require_session().err().expect("missing session");
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Config);
        assert!(error.to_string().contains("requires a session"));

        let with_session = ctx();
        assert_eq!(with_session.session_id(), Some("s1"));
        assert!(with_session.require_session().is_ok());
    }

    #[test]
    fn task_context_defaults_to_no_history_and_no_reasoning() {
        let default = TaskContext::default();
        assert_eq!(default.history.len(), 0);
        assert!(!default.expose_reasoning);
        let built = ctx();
        assert_eq!(built.history.len(), 0);
        assert!(!built.expose_reasoning);
        let configured = ctx()
            .with_history(vec![
                HistoryEntry::user("hi"),
                HistoryEntry::assistant("hello"),
            ])
            .with_expose_reasoning(true);
        assert_eq!(configured.history.len(), 2);
        assert_eq!(configured.history[0].text, "hi");
        assert_eq!(
            configured.history[1].role,
            crate::controller::HistoryRole::Assistant
        );
        assert!(configured.expose_reasoning);
        // The builders are chainable on a cloned context and stay independent.
        let cloned = configured.clone();
        assert!(cloned.expose_reasoning);
        assert_eq!(cloned.history.len(), 2);
    }
    #[test]
    fn task_run_reasoning_reads_only_successful_results() {
        let run = TaskRun {
            task: task("t1", AGENT_TASK_TYPE, "do it"),
            result: Ok(TaskResult::text("done").with_reasoning("why")),
        };
        assert_eq!(run.reasoning(), Some("why"));
        let plain = TaskRun {
            task: task("t2", AGENT_TASK_TYPE, "do it"),
            result: Ok(TaskResult::text("done")),
        };
        assert_eq!(plain.reasoning(), None);
        let failed = TaskRun {
            task: task("t3", AGENT_TASK_TYPE, "do it"),
            result: Err(Error::validation("boom")),
        };
        assert_eq!(failed.reasoning(), None);
    }
    #[test]
    fn registry_adds_replaces_and_removes() {
        let mut registry = TaskExecutorRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);

        assert!(registry.add(Arc::new(NamedExecutor("agent"))).is_none());
        assert_eq!(registry.len(), 1);
        assert!(registry.contains("agent"));
        assert_eq!(
            registry
                .get("agent")
                .map(|executor| executor.task_type().to_owned()),
            Some("agent".to_owned())
        );

        // Re-adding the same task type returns the previous executor.
        assert!(registry.add(Arc::new(NamedExecutor("agent"))).is_some());
        assert_eq!(registry.len(), 1);

        assert!(registry
            .insert("custom", Arc::new(NamedExecutor("custom")))
            .is_none());
        assert_eq!(registry.task_types(), ["agent", "custom"]);

        assert!(registry.remove("agent").is_some());
        assert!(registry.remove("agent").is_none());
        assert!(!registry.contains("agent"));
        assert_eq!(registry.task_types(), ["custom"]);
    }

    #[test]
    fn registry_exposes_executor_descriptions() {
        let mut registry = TaskExecutorRegistry::new();
        registry.add(Arc::new(EchoExecutor));
        let executor = registry.get(AGENT_TASK_TYPE).expect("registered");
        assert_eq!(executor.description(), "echoes the task text");
        assert_eq!(executor.task_type(), AGENT_TASK_TYPE);
        assert!(!registry.is_empty());
    }

    #[tokio::test]
    async fn scheduler_runs_tasks_in_input_order() {
        let mut registry = TaskExecutorRegistry::new();
        registry.add(Arc::new(EchoExecutor));
        let scheduler = TaskScheduler::new(registry, 0);

        let runs = scheduler
            .dispatch(
                vec![
                    task("t1", AGENT_TASK_TYPE, "first"),
                    task("t2", AGENT_TASK_TYPE, "second"),
                ],
                &ctx(),
            )
            .await;

        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].task.task_id, "t1");
        assert_eq!(runs[1].task.task_id, "t2");
        assert_eq!(runs[0].text(), "first");
        assert_eq!(runs[1].text(), "second");
        assert_eq!(runs[0].status(), TaskStatus::Completed);
        assert_eq!(runs[0].task.status, TaskStatus::Completed);
        assert!(runs.iter().all(|run| run.result.is_ok()));
    }

    #[tokio::test]
    async fn scheduler_fails_unknown_task_types_without_erroring() {
        let scheduler = TaskScheduler::new(TaskExecutorRegistry::new(), 0);
        let runs = scheduler
            .dispatch(vec![task("t1", "missing", "do it")], &ctx())
            .await;

        assert_eq!(runs.len(), 1);
        assert!(runs[0].result.is_ok());
        assert_eq!(runs[0].status(), TaskStatus::Failed);
        assert_eq!(runs[0].task.status, TaskStatus::Failed);
        let message = runs[0].task.error_message.clone().unwrap_or_default();
        assert!(message.contains("task executor not found: missing"));
        assert!(runs[0].text().contains("task executor not found: missing"));
    }

    #[tokio::test]
    async fn scheduler_truncates_at_max_tasks() {
        let mut registry = TaskExecutorRegistry::new();
        registry.add(Arc::new(EchoExecutor));
        let scheduler = TaskScheduler::new(registry, 1).with_max_tasks(1);

        let runs = scheduler
            .dispatch(
                vec![
                    task("t1", AGENT_TASK_TYPE, "first"),
                    task("t2", AGENT_TASK_TYPE, "second"),
                ],
                &ctx(),
            )
            .await;

        assert_eq!(scheduler.max_tasks(), 1);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].task.task_id, "t1");
    }

    #[tokio::test]
    async fn scheduler_records_executor_failures() {
        let mut registry = TaskExecutorRegistry::new();
        registry.add(Arc::new(FailingExecutor));
        let scheduler = TaskScheduler::new(registry, 0);

        let runs = scheduler
            .dispatch(vec![task("t1", "failing", "do it")], &ctx())
            .await;

        assert!(runs[0].result.is_ok());
        assert_eq!(runs[0].status(), TaskStatus::Failed);
        assert_eq!(runs[0].task.status, TaskStatus::Failed);
        let message = runs[0].task.error_message.clone().unwrap_or_default();
        assert!(message.contains("boom"));
        assert!(runs[0].text().contains("boom"));
    }

    #[tokio::test]
    async fn scheduler_times_out_slow_executors() {
        let mut registry = TaskExecutorRegistry::new();
        registry.add(Arc::new(SlowExecutor));
        let timeout = Duration::from_millis(10);
        let scheduler = TaskScheduler::new(registry, 0).with_timeout(Some(timeout));

        assert_eq!(scheduler.timeout(), Some(timeout));
        assert!(
            SLOW_EXECUTOR_HOLD >= timeout * 25,
            "the executor must outlive the budget by a wide margin: {SLOW_EXECUTOR_HOLD:?} vs {timeout:?}"
        );
        let runs = scheduler
            .dispatch(vec![task("t1", "slow", "do it")], &ctx())
            .await;

        assert!(runs[0].result.is_ok());
        assert_eq!(runs[0].status(), TaskStatus::Failed);
        assert_eq!(runs[0].task.status, TaskStatus::Failed);
        assert!(runs[0].text().contains("timed out"));
    }

    #[tokio::test]
    async fn scheduler_passes_terminal_tasks_through() {
        let mut registry = TaskExecutorRegistry::new();
        registry.add(Arc::new(FailingExecutor));
        let scheduler = TaskScheduler::new(registry, 0);
        let done = task("t1", "failing", "do it").with_status(TaskStatus::Completed);

        let runs = scheduler.dispatch(vec![done], &ctx()).await;

        assert_eq!(runs[0].status(), TaskStatus::Completed);
        assert_eq!(runs[0].task.status, TaskStatus::Completed);
        assert_eq!(runs[0].text(), "");
    }

    #[test]
    fn task_run_frames_follow_the_status() {
        let run = TaskRun {
            task: task("t1", AGENT_TASK_TYPE, "do it"),
            result: Ok(TaskResult::text("done")),
        };
        let frames = run.frames(3);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].index, 3);
        assert!(frames[0].last_chunk);
        let payload = frames[0].payload.clone().expect("payload");
        assert_eq!(payload.kind, OutputKind::TaskCompletion);
        assert_eq!(payload.data, vec![DataFrame::text("done")]);

        let failed = TaskRun {
            task: task("t2", AGENT_TASK_TYPE, "do it"),
            result: Ok(TaskResult::failed("boom")),
        };
        let payload = failed.frames(0)[0].payload.clone().expect("payload");
        assert_eq!(payload.kind, OutputKind::TaskFailed);
        assert_eq!(
            payload.data,
            vec![DataFrame::text(failed.text())],
            "the text fallback carries the error"
        );

        let interrupted = TaskRun {
            task: task("t3", AGENT_TASK_TYPE, "do it"),
            result: Ok(TaskResult::input_required(vec![DataFrame::text("ask")])),
        };
        let payload = interrupted.frames(1)[0].payload.clone().expect("payload");
        assert_eq!(payload.kind, OutputKind::TaskInteraction);
        assert_eq!(payload.data, vec![DataFrame::text("ask")]);
    }
}
