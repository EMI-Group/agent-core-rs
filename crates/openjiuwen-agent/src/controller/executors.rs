//! Task executors, the executor registry, and the task scheduler.
//!
//! Ports the executor contract of
//! `openjiuwen/core/controller/modules/task_scheduler.py` together with the
//! three concrete executors of the older application lineage:
//!
//! - [`ControllerAgent`] — `core/single_agent/base.py::ControllerAgent`, which
//!   runs one [`Agent`] and maps its output onto a task result.
//! - [`WorkflowController`] — `core/application/workflow_agent/`, which picks a
//!   registered [`Workflow`] and runs it against a child workflow session.
//! - [`LLMController`] — `core/application/llm_agent/llm_controller.py`, the
//!   LLM router that picks the task type of a turn.
//!
//! Unlike Python's `TaskScheduler`, which owns an event queue, a task manager,
//! and a background scheduling loop, [`TaskScheduler`] is a plain value: it
//! dispatches a caller-supplied batch of [`Task`]s to the executor registered
//! for each `task_type` and returns one [`TaskRun`] per task. Missing executors,
//! executor failures, and timeouts become failed results rather than errors, so
//! a batch always yields a run per dispatched task.

use super::schema::{
    ControllerOutputChunk, ControllerOutputPayload, DataFrame, Intent, OutputKind, Task,
    TaskResult, TaskStatus,
};
use crate::agent::Agent;
use crate::card::AgentCard;
use crate::output::{AgentOutput, ResultType};
use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::{BaseMessage, Model, ModelRequestConfig};
use openjiuwen_session::{Session, WorkflowSession, WorkflowSessionOptions};
use openjiuwen_workflow::{Workflow, WorkflowExecutionState, WorkflowOutput};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;
use tracing::warn;

/// Task type of the built-in agent executor ([`ControllerAgent`]).
pub const AGENT_TASK_TYPE: &str = "agent";
/// Task type of the built-in workflow executor ([`WorkflowController`]).
pub const WORKFLOW_TASK_TYPE: &str = "workflow";
/// Task type of the built-in LLM router ([`LLMController`]).
pub const LLM_TASK_TYPE: &str = "llm";

/// System instruction the [`LLMController`] sends when no custom prompt is set.
const DEFAULT_ROUTER_PROMPT: &str =
    "You are a task router. reply with exactly one of these executor names:";

/// Description reported by executors that have none.
const NO_DESCRIPTION: &str = "";
/// Description of [`ControllerAgent`].
const AGENT_DESCRIPTION: &str = "runs a controller agent";
/// Description of [`WorkflowController`].
const WORKFLOW_DESCRIPTION: &str = "runs a workflow";
/// Name of the [`LLMController`] router.
const LLM_ROUTER_NAME: &str = "llm";

/// Context handed to a [`TaskExecutor`] for one dispatch.
///
/// The session is optional because a controller may dispatch tasks that do not
/// touch session state; executors that need one call
/// [`require_session`](Self::require_session) to turn its absence into a typed
/// error.
#[derive(Clone, Default)]
pub struct TaskContext<'a> {
    /// Session shared by the dispatched tasks, when the caller has one.
    pub session: Option<Arc<dyn Session>>,
    /// Id of the controller driving the dispatch, used in diagnostics.
    pub controller_id: &'a str,
}

impl<'a> TaskContext<'a> {
    /// Builds a context for `session` owned by `controller_id`.
    #[must_use]
    pub fn new(session: Option<Arc<dyn Session>>, controller_id: &'a str) -> Self {
        Self {
            session,
            controller_id,
        }
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

/// Executor that runs one [`Agent`] (port of Python's `ControllerAgent`).
pub struct ControllerAgent {
    card: AgentCard,
    agent: Arc<dyn Agent>,
}

impl ControllerAgent {
    /// Wraps `agent` as a task executor identified by `card`.
    #[must_use]
    pub fn new(card: AgentCard, agent: Arc<dyn Agent>) -> Self {
        Self { card, agent }
    }

    /// Returns the agent card.
    #[must_use]
    pub fn card(&self) -> &AgentCard {
        &self.card
    }

    /// Returns the wrapped agent.
    #[must_use]
    pub fn agent(&self) -> &Arc<dyn Agent> {
        &self.agent
    }
}

#[async_trait]
impl TaskExecutor for ControllerAgent {
    fn task_type(&self) -> &str {
        AGENT_TASK_TYPE
    }

    fn description(&self) -> &str {
        AGENT_DESCRIPTION
    }

    async fn execute(&self, task: &Task, ctx: &TaskContext<'_>) -> Result<TaskResult> {
        let query = task.input_text().unwrap_or_default();
        let output = self
            .agent
            .invoke(json!({ "query": query }), ctx.session.clone())
            .await?;
        Ok(map_agent_output(output))
    }
}

/// Maps an [`AgentOutput`] onto a [`TaskResult`].
///
/// Answers become completed text results, errors become failures, and
/// interrupts become [`TaskStatus::InputRequired`] results whose frames are the
/// serialized state frames plus the output text.
fn map_agent_output(output: AgentOutput) -> TaskResult {
    match output.result_type {
        ResultType::Answer => TaskResult::text(output.output),
        ResultType::Error => {
            let message = if output.output.is_empty() {
                "agent returned an error".to_owned()
            } else {
                output.output
            };
            TaskResult::failed(message)
        }
        ResultType::Interrupt => {
            let mut frames: Vec<DataFrame> = output
                .state
                .iter()
                .map(|frame| DataFrame::json(serde_json::to_value(frame).unwrap_or(Value::Null)))
                .collect();
            if !output.output.is_empty() {
                frames.push(DataFrame::text(output.output));
            }
            TaskResult::input_required(frames)
        }
    }
}

/// Executor that runs one of a set of [`Workflow`]s (port of Python's
/// `WorkflowController`).
#[derive(Default)]
pub struct WorkflowController {
    workflows: RwLock<Vec<Arc<Workflow>>>,
}

impl WorkflowController {
    /// Builds a controller with no workflows.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds `workflow`, replacing a workflow with the same card id.
    ///
    /// Returns the replaced workflow, if any.
    pub fn add_workflow(&self, workflow: Arc<Workflow>) -> Option<Arc<Workflow>> {
        let id = workflow.card().id.clone();
        let mut workflows = self.workflows.write();
        if let Some(slot) = workflows
            .iter_mut()
            .find(|existing| existing.card().id == id)
        {
            return Some(std::mem::replace(slot, workflow));
        }
        workflows.push(workflow);
        None
    }

    /// Adds every workflow of `workflows`.
    pub fn add_workflows(&self, workflows: impl IntoIterator<Item = Arc<Workflow>>) {
        for workflow in workflows {
            self.add_workflow(workflow);
        }
    }

    /// Removes the workflow with card id `id`.
    pub fn remove_workflow(&self, id: &str) -> Option<Arc<Workflow>> {
        let mut workflows = self.workflows.write();
        let index = workflows
            .iter()
            .position(|workflow| workflow.card().id == id)?;
        Some(workflows.remove(index))
    }

    /// Returns a snapshot of the registered workflows.
    #[must_use]
    pub fn workflows(&self) -> Vec<Arc<Workflow>> {
        self.workflows.read().clone()
    }

    /// Returns the workflow with card id `id`.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<Arc<Workflow>> {
        self.workflows
            .read()
            .iter()
            .find(|workflow| workflow.card().id == id)
            .map(Arc::clone)
    }

    /// Maps a workflow output onto a task result.
    ///
    /// A completed run carries the workflow result as its output (a JSON string
    /// result displays as raw text) and the workflow id in its metadata; an
    /// interrupted run carries one JSON frame per result element (or one frame
    /// for a non-array result); an error run is a failure.
    #[must_use]
    pub fn map_output(workflow_id: &str, output: WorkflowOutput) -> TaskResult {
        match output.state {
            WorkflowExecutionState::Completed => TaskResult {
                status: TaskStatus::Completed,
                output: output.result,
                metadata: BTreeMap::from([(
                    "workflow_id".to_owned(),
                    Value::String(workflow_id.to_owned()),
                )]),
                ..TaskResult::default()
            },
            WorkflowExecutionState::InputRequired => {
                let frames = match output.result {
                    Value::Array(items) => items.into_iter().map(DataFrame::json).collect(),
                    other => vec![DataFrame::json(other)],
                };
                TaskResult::input_required(frames)
            }
            WorkflowExecutionState::Error => {
                TaskResult::failed(format!("workflow {workflow_id} finished in error state"))
            }
        }
    }

    /// Picks the workflow for `task`.
    ///
    /// An explicit `metadata["workflow_id"]` wins; otherwise a `description`
    /// that names a card id; otherwise the only registered workflow.
    fn select_workflow(&self, task: &Task) -> Option<Arc<Workflow>> {
        let workflows = self.workflows.read();
        if let Some(id) = task.metadata.get("workflow_id").and_then(Value::as_str) {
            return workflows
                .iter()
                .find(|workflow| workflow.card().id == id)
                .map(Arc::clone);
        }
        if let Some(description) = task.description.as_deref() {
            if let Some(workflow) = workflows
                .iter()
                .find(|workflow| workflow.card().id == description)
            {
                return Some(Arc::clone(workflow));
            }
        }
        if workflows.len() == 1 {
            return workflows.first().map(Arc::clone);
        }
        None
    }
}

#[async_trait]
impl TaskExecutor for WorkflowController {
    fn task_type(&self) -> &str {
        WORKFLOW_TASK_TYPE
    }

    fn description(&self) -> &str {
        WORKFLOW_DESCRIPTION
    }

    async fn execute(&self, task: &Task, ctx: &TaskContext<'_>) -> Result<TaskResult> {
        let workflow = self.select_workflow(task).ok_or_else(|| {
            Error::not_found(format!("no workflow registered for task {}", task.task_id))
        })?;
        let parent = ctx.require_session()?;
        let session = workflow_session(&workflow, &parent);
        let query = task.input_text().unwrap_or_default();
        let output = workflow.invoke(json!({ "query": query }), session).await?;
        Ok(Self::map_output(&workflow.card().id, output))
    }
}

/// Builds the child workflow session of `parent`, mirroring
/// `WorkflowAgent`'s session layout.
fn workflow_session(workflow: &Arc<Workflow>, parent: &Arc<dyn Session>) -> Arc<dyn Session> {
    let workflow_id = workflow.card().id.clone();
    let session_id = format!("{}::{workflow_id}", parent.id());
    Arc::new(WorkflowSession::with_options(WorkflowSessionOptions {
        session_id: Some(session_id),
        workflow_id: Some(workflow_id),
        parent: Some(Arc::clone(parent)),
        ..WorkflowSessionOptions::default()
    })) as Arc<dyn Session>
}

/// One executor choice shown to the model.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ExecutorInfo {
    /// Task type the choice maps onto.
    pub task_type: String,
    /// Executor name the model is asked to reply with.
    pub name: String,
    /// Model-visible description of what the executor does.
    pub description: String,
}

impl ExecutorInfo {
    /// Builds a choice from its parts.
    #[must_use]
    pub fn new(
        task_type: impl Into<String>,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            task_type: task_type.into(),
            name: name.into(),
            description: description.into(),
        }
    }
}

/// LLM router that picks the task type of a turn (port of Python's
/// `LLMController`).
///
/// With zero executors it falls back to [`default_task_type`](Self::default_task_type);
/// with exactly one it answers without a model call; with several it asks the
/// model once and parses the reply. A malformed reply falls back to the first
/// executor and never errors.
pub struct LLMController {
    model: Arc<dyn Model>,
    executors: Vec<ExecutorInfo>,
    request: ModelRequestConfig,
    default_task_type: String,
    prompt: String,
}

impl LLMController {
    /// Builds a router over `executors` using `model`.
    ///
    /// The default task type is the first executor's task type, or
    /// [`LLM_TASK_TYPE`] when no executor is configured.
    #[must_use]
    pub fn new(model: Arc<dyn Model>, executors: Vec<ExecutorInfo>) -> Self {
        let default_task_type = executors
            .first()
            .map_or_else(|| LLM_TASK_TYPE.to_owned(), |info| info.task_type.clone());
        let request = ModelRequestConfig::new(model.model_name());
        Self {
            model,
            executors,
            request,
            default_task_type,
            prompt: DEFAULT_ROUTER_PROMPT.to_owned(),
        }
    }

    /// Sets the task type used when no executor is registered.
    #[must_use]
    pub fn with_default_task_type(mut self, task_type: impl Into<String>) -> Self {
        self.default_task_type = task_type.into();
        self
    }

    /// Sets the request sent to the model.
    #[must_use]
    pub fn with_request(mut self, request: ModelRequestConfig) -> Self {
        self.request = request;
        self
    }

    /// Replaces the router instruction sent as the system prompt.
    #[must_use]
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = prompt.into();
        self
    }

    /// Returns the executor choices shown to the model.
    #[must_use]
    pub fn executors(&self) -> &[ExecutorInfo] {
        &self.executors
    }

    /// Returns the task type used when no executor is registered.
    #[must_use]
    pub fn default_task_type(&self) -> &str {
        &self.default_task_type
    }

    /// Appends one executor choice.
    #[must_use]
    pub fn add_executor(mut self, info: ExecutorInfo) -> Self {
        self.executors.push(info);
        self
    }

    /// Chooses the task type for a turn.
    ///
    /// # Errors
    ///
    /// Propagates the model's error; a reply that names no executor is not an
    /// error and falls back to the first executor.
    pub async fn select(&self, query: &str, intents: &[Intent]) -> Result<String> {
        match self.executors.len() {
            0 => return Ok(self.default_task_type.clone()),
            1 => {
                if let Some(info) = self.executors.first() {
                    return Ok(info.task_type.clone());
                }
                return Ok(self.default_task_type.clone());
            }
            _ => {}
        }
        let messages = self.build_messages(query, intents);
        let response = self.model.invoke(&messages, &self.request).await?;
        let reply = response
            .message()
            .map(|message| message.text().into_owned())
            .unwrap_or_default();
        Ok(self.parse_reply(&reply))
    }

    /// Builds the deterministic router prompt: the instruction plus one line
    /// per executor, and a user message with the query and intent labels.
    fn build_messages(&self, query: &str, intents: &[Intent]) -> Vec<BaseMessage> {
        let mut parts = vec![
            self.prompt.clone(),
            "reply with exactly one of these executor names:".to_owned(),
        ];
        parts.extend(
            self.executors
                .iter()
                .map(|info| format!("- {} ({}): {}", info.name, info.task_type, info.description)),
        );
        let system = parts
            .into_iter()
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        let labels = intents
            .iter()
            .map(Intent::label)
            .collect::<Vec<_>>()
            .join(", ");
        let user = if labels.is_empty() {
            format!("Query: {query}")
        } else {
            format!("Query: {query}\nIntents: {labels}")
        };
        vec![BaseMessage::system(system), BaseMessage::user(user)]
    }

    /// Matches the model reply against the executor choices.
    ///
    /// A case-insensitive name or task-type occurrence wins, then a
    /// whitespace-delimited token, then the first executor.
    fn parse_reply(&self, reply: &str) -> String {
        let lower = reply.to_lowercase();
        if let Some(info) = self.executors.iter().find(|info| {
            (!info.name.is_empty() && lower.contains(&info.name.to_lowercase()))
                || (!info.task_type.is_empty() && lower.contains(&info.task_type.to_lowercase()))
        }) {
            return info.task_type.clone();
        }
        for token in lower.split_whitespace() {
            let token = token.trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '-');
            if let Some(info) = self.executors.iter().find(|info| {
                info.name.eq_ignore_ascii_case(token) || info.task_type.eq_ignore_ascii_case(token)
            }) {
                return info.task_type.clone();
            }
        }
        match self.executors.first() {
            Some(info) => {
                warn!(
                    reply = %reply,
                    task_type = %info.task_type,
                    "LLM router reply matched no executor; using the first one"
                );
                info.task_type.clone()
            }
            None => self.default_task_type.clone(),
        }
    }
}

#[async_trait]
impl TaskRouter for LLMController {
    fn name(&self) -> &str {
        LLM_ROUTER_NAME
    }

    async fn select(&self, query: &str, intents: &[Intent]) -> Result<Option<String>> {
        Ok(Some(LLMController::select(self, query, intents).await?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output::AgentChunk;
    use futures::stream::BoxStream;
    use futures::StreamExt;
    use openjiuwen_foundation::Model;
    use openjiuwen_session::AgentSession;
    use openjiuwen_test_support::ScriptedModel;
    use openjiuwen_workflow::{ComponentOptions, EndComponent, StartComponent, WorkflowCard};
    use serde_json::json;

    /// Description reported by [`EchoExecutor`].
    const ECHO_DESCRIPTION: &str = "echoes the task text";
    /// Task type of [`FailingExecutor`].
    const FAILING_TASK_TYPE: &str = "failing";
    /// Task type of [`SlowExecutor`].
    const SLOW_TASK_TYPE: &str = "slow";

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

    /// Executor that outlives the test's scheduler timeout.
    struct SlowExecutor;

    #[async_trait]
    impl TaskExecutor for SlowExecutor {
        fn task_type(&self) -> &str {
            SLOW_TASK_TYPE
        }

        async fn execute(&self, _task: &Task, _ctx: &TaskContext<'_>) -> Result<TaskResult> {
            tokio::time::sleep(Duration::from_millis(50)).await;
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

    /// Agent that returns a canned output.
    struct FakeAgent {
        card: AgentCard,
        output: AgentOutput,
    }

    #[async_trait]
    impl Agent for FakeAgent {
        fn card(&self) -> &AgentCard {
            &self.card
        }

        async fn invoke(
            &self,
            _inputs: Value,
            _session: Option<Arc<dyn Session>>,
        ) -> Result<AgentOutput> {
            Ok(self.output.clone())
        }

        fn stream(
            &self,
            _inputs: Value,
            _session: Option<Arc<dyn Session>>,
        ) -> BoxStream<'static, Result<AgentChunk>> {
            futures::stream::empty().boxed()
        }
    }

    fn task(task_id: &str, task_type: &str, description: &str) -> Task {
        Task::submitted("s1", task_id, task_type, description)
    }

    fn ctx() -> TaskContext<'static> {
        TaskContext::new(Some(Arc::new(AgentSession::new("s1"))), "ctl")
    }

    fn agent_card() -> AgentCard {
        AgentCard::new("a")
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
        let scheduler =
            TaskScheduler::new(registry, 0).with_timeout(Some(Duration::from_millis(10)));

        assert_eq!(scheduler.timeout(), Some(Duration::from_millis(10)));
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

    #[tokio::test]
    async fn controller_agent_maps_answers_and_errors() {
        let answer = ControllerAgent::new(
            agent_card(),
            Arc::new(FakeAgent {
                card: agent_card(),
                output: AgentOutput::answer("done"),
            }),
        );
        assert_eq!(answer.task_type(), AGENT_TASK_TYPE);
        assert_eq!(answer.description(), "runs a controller agent");
        assert_eq!(answer.card().name, "a");
        let result = answer
            .execute(&task("t1", AGENT_TASK_TYPE, "hi"), &ctx())
            .await
            .expect("answer");
        assert_eq!(result.status, TaskStatus::Completed);
        assert_eq!(result.display(), "done");
        assert_eq!(answer.agent().card().name, "a");

        let failure = ControllerAgent::new(
            agent_card(),
            Arc::new(FakeAgent {
                card: agent_card(),
                output: AgentOutput::error(""),
            }),
        );
        let result = failure
            .execute(&task("t2", AGENT_TASK_TYPE, "hi"), &ctx())
            .await
            .expect("error output");
        assert_eq!(result.status, TaskStatus::Failed);
        assert_eq!(result.error.as_deref(), Some("agent returned an error"));
    }

    #[tokio::test]
    async fn controller_agent_maps_interrupts_onto_input_required() {
        let mut output = AgentOutput::interrupt(
            vec![crate::output::OutputFrame::interaction(
                0,
                "i1",
                json!({"kind": "confirm"}),
            )],
            vec!["i1".to_owned()],
        );
        output.output = "please confirm".to_owned();
        let executor = ControllerAgent::new(
            agent_card(),
            Arc::new(FakeAgent {
                card: agent_card(),
                output,
            }),
        );

        let result = executor
            .execute(&task("t1", AGENT_TASK_TYPE, "hi"), &ctx())
            .await
            .expect("interrupt");

        assert_eq!(result.status, TaskStatus::InputRequired);
        assert_eq!(result.data.len(), 2);
        assert_eq!(result.data[0].kind(), "json");
        assert_eq!(result.data[1], DataFrame::text("please confirm"));
    }

    #[test]
    fn workflow_map_output_covers_every_state() {
        let completed = WorkflowController::map_output(
            "wf",
            WorkflowOutput {
                result: json!({"answer": "hi"}),
                state: WorkflowExecutionState::Completed,
            },
        );
        assert_eq!(completed.status, TaskStatus::Completed);
        assert_eq!(completed.output, json!({"answer": "hi"}));
        assert_eq!(completed.metadata["workflow_id"], json!("wf"));

        let raw = WorkflowController::map_output(
            "wf",
            WorkflowOutput {
                result: Value::String("plain".to_owned()),
                state: WorkflowExecutionState::Completed,
            },
        );
        assert_eq!(raw.display(), "plain");

        let interrupted = WorkflowController::map_output(
            "wf",
            WorkflowOutput {
                result: json!([{"id": 1}, {"id": 2}]),
                state: WorkflowExecutionState::InputRequired,
            },
        );
        assert_eq!(interrupted.status, TaskStatus::InputRequired);
        assert_eq!(interrupted.data.len(), 2);
        assert!(interrupted.data.iter().all(|frame| frame.kind() == "json"));

        let single = WorkflowController::map_output(
            "wf",
            WorkflowOutput {
                result: json!({"id": 1}),
                state: WorkflowExecutionState::InputRequired,
            },
        );
        assert_eq!(single.data.len(), 1);

        let failed = WorkflowController::map_output(
            "wf",
            WorkflowOutput {
                result: Value::Null,
                state: WorkflowExecutionState::Error,
            },
        );
        assert_eq!(failed.status, TaskStatus::Failed);
        assert_eq!(
            failed.error.as_deref(),
            Some("workflow wf finished in error state")
        );
    }

    /// `start → end`, projecting the end output to `{"answer": <start.query>}`.
    fn echo_workflow(id: &str) -> Arc<Workflow> {
        let mut workflow = Workflow::new(WorkflowCard::new(id));
        workflow
            .set_start_comp(
                "start",
                StartComponent::new(),
                ComponentOptions::new().with_inputs_schema(json!({"query": "${inputs.query}"})),
            )
            .set_end_comp(
                "end",
                EndComponent::new(),
                ComponentOptions::new()
                    .with_inputs_schema(json!({"answer": "${start.query}"}))
                    .with_outputs_schema(json!({"answer": "${output.output.answer}"})),
            )
            .add_connection("start", "end");
        Arc::new(workflow)
    }

    #[test]
    fn workflow_controller_registry_helpers() {
        let controller = WorkflowController::new();
        assert!(controller.workflows().is_empty());
        assert!(controller.add_workflow(echo_workflow("a")).is_none());
        assert!(controller.add_workflow(echo_workflow("a")).is_some());
        assert_eq!(controller.workflows().len(), 1);
        assert!(controller.get("a").is_some());
        assert!(controller.get("missing").is_none());
        controller.add_workflows([echo_workflow("b"), echo_workflow("c")]);
        assert_eq!(controller.workflows().len(), 3);
        assert!(controller.remove_workflow("b").is_some());
        assert!(controller.remove_workflow("b").is_none());
    }

    #[tokio::test]
    async fn workflow_controller_executes_the_selected_workflow() {
        let controller = WorkflowController::new();
        controller.add_workflows([echo_workflow("first"), echo_workflow("second")]);
        assert_eq!(controller.task_type(), WORKFLOW_TASK_TYPE);
        assert_eq!(controller.description(), "runs a workflow");

        // An explicit selection in the task metadata wins.
        let selected =
            task("t1", WORKFLOW_TASK_TYPE, "ignored").with_metadata("workflow_id", json!("second"));
        let result = controller.execute(&selected, &ctx()).await.expect("run");
        assert_eq!(result.status, TaskStatus::Completed);
        assert_eq!(result.metadata["workflow_id"], json!("second"));
        assert_ne!(result.display(), "");

        // A description naming a card id selects it; otherwise the only
        // workflow is used.
        let single = WorkflowController::new();
        single.add_workflow(echo_workflow("only"));
        let result = single
            .execute(&task("t2", WORKFLOW_TASK_TYPE, "only"), &ctx())
            .await
            .expect("run");
        assert_eq!(result.status, TaskStatus::Completed);
        assert_eq!(result.metadata["workflow_id"], json!("only"));
        assert_ne!(result.display(), "");

        let result = single
            .execute(&task("t3", WORKFLOW_TASK_TYPE, "unrelated"), &ctx())
            .await
            .expect("single workflow fallback");
        assert_eq!(result.metadata["workflow_id"], json!("only"));
    }

    #[tokio::test]
    async fn workflow_controller_reports_missing_workflow_and_session() {
        let empty = WorkflowController::new();
        let error = empty
            .execute(&task("t1", WORKFLOW_TASK_TYPE, "hi"), &ctx())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::NotFound);
        assert!(error
            .to_string()
            .contains("no workflow registered for task t1"));

        let controller = WorkflowController::new();
        controller.add_workflow(echo_workflow("only"));
        let error = controller
            .execute(
                &task("t2", WORKFLOW_TASK_TYPE, "hi"),
                &TaskContext::default(),
            )
            .await
            .unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Config);
        assert!(error.to_string().contains("requires a session"));
    }

    fn info(task_type: &str, name: &str) -> ExecutorInfo {
        ExecutorInfo::new(task_type, name, format!("{name} executor"))
    }

    #[tokio::test]
    async fn llm_controller_skips_the_model_for_one_executor() {
        let scripted = ScriptedModel::builder().build();
        let handle = scripted.clone();
        let controller = LLMController::new(
            Arc::new(scripted) as Arc<dyn Model>,
            vec![info("agent", "agent")],
        );

        assert_eq!(controller.select("hi", &[]).await.expect("select"), "agent");
        assert_eq!(handle.call_count(), 0);
        assert_eq!(controller.default_task_type(), "agent");
        assert_eq!(controller.executors().len(), 1);
    }

    #[tokio::test]
    async fn llm_controller_parses_the_model_reply() {
        let scripted = ScriptedModel::builder().respond_text("workflow").build();
        let handle = scripted.clone();
        let controller = LLMController::new(
            Arc::new(scripted) as Arc<dyn Model>,
            vec![info("agent", "agent"), info("workflow", "workflow")],
        );

        let selected = controller.select("run it", &[Intent::default()]).await;
        assert_eq!(selected.expect("select"), "workflow");
        assert_eq!(handle.call_count(), 1);
    }

    #[tokio::test]
    async fn llm_controller_falls_back_to_the_first_executor() {
        let scripted = ScriptedModel::builder().respond_text("huh?").build();
        let handle = scripted.clone();
        let controller = LLMController::new(
            Arc::new(scripted) as Arc<dyn Model>,
            vec![info("agent", "agent"), info("workflow", "workflow")],
        );

        let selected = controller.select("run it", &[]).await;
        assert_eq!(selected.expect("select"), "agent");
        assert_eq!(handle.call_count(), 1);
    }

    #[tokio::test]
    async fn llm_controller_without_executors_uses_the_default() {
        let scripted = ScriptedModel::builder().build();
        let controller = LLMController::new(Arc::new(scripted) as Arc<dyn Model>, vec![])
            .with_default_task_type("fallback");

        assert_eq!(controller.default_task_type(), "fallback");
        assert_eq!(
            controller.select("hi", &[]).await.expect("select"),
            "fallback"
        );

        // The default is `llm` when nothing is configured.
        let bare = LLMController::new(
            Arc::new(ScriptedModel::builder().build()) as Arc<dyn Model>,
            vec![],
        );
        assert_eq!(bare.default_task_type(), LLM_TASK_TYPE);
    }

    #[tokio::test]
    async fn llm_controller_builders_and_prompt() {
        let scripted = ScriptedModel::builder().respond_text("named").build();
        let handle = scripted.clone();
        let request = ModelRequestConfig::new("custom-model");
        let controller = LLMController::new(
            Arc::new(scripted) as Arc<dyn Model>,
            vec![info("agent", "agent")],
        )
        .with_prompt("pick carefully")
        .with_request(request)
        .add_executor(ExecutorInfo::new("named", "named", "the named one"));

        assert_eq!(controller.executors().len(), 2);
        let selected = controller.select("hi", &[]).await.expect("select");
        assert_eq!(selected, "named");
        assert_eq!(handle.call_count(), 1);

        let kinds = controller
            .build_messages("hi", &[Intent::default()])
            .iter()
            .map(|message| message.role.clone())
            .collect::<Vec<_>>();
        assert_eq!(
            kinds,
            [
                openjiuwen_foundation::Role::System,
                openjiuwen_foundation::Role::User
            ]
        );
    }

    #[tokio::test]
    async fn llm_controller_is_usable_as_a_task_router_object() {
        let scripted = ScriptedModel::builder().respond_text("workflow").build();
        let router: Arc<dyn TaskRouter> = Arc::new(LLMController::new(
            Arc::new(scripted) as Arc<dyn Model>,
            vec![info("agent", "agent"), info("workflow", "workflow")],
        ));

        assert_eq!(router.name(), "llm");
        assert_eq!(
            router.select("hi", &[]).await.expect("select"),
            Some("workflow".to_owned())
        );
    }

    #[test]
    fn executor_info_round_trips() {
        let info = ExecutorInfo::new("agent", "agent", "runs a controller agent");
        let wire = serde_json::to_value(&info).unwrap();
        assert_eq!(
            wire,
            json!({
                "task_type": "agent",
                "name": "agent",
                "description": "runs a controller agent",
            })
        );
        let parsed: ExecutorInfo = serde_json::from_value(wire).unwrap();
        assert_eq!(parsed, info);
        assert_eq!(ExecutorInfo::default(), ExecutorInfo::default());
    }
}
