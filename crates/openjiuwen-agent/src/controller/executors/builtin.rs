//! Concrete task executors shipped with the controller stack.
//!
//! These are the three executors of the older application lineage, expressed on
//! top of the [`TaskExecutor`] seam declared in the parent module:
//!
//! - [`ControllerAgent`] — `core/single_agent/base.py::ControllerAgent`, which
//!   runs one [`Agent`] and maps its output onto a task result.
//! - [`WorkflowController`] — `core/application/workflow_agent/`, which picks a
//!   registered [`Workflow`] and runs it against a child workflow session.
//! - [`LLMController`] — `core/application/llm_agent/llm_controller.py`, the
//!   LLM router that picks the task type of a turn.

use super::{
    TaskContext, TaskExecutor, TaskRouter, AGENT_DESCRIPTION, AGENT_TASK_TYPE, LLM_TASK_TYPE,
    WORKFLOW_DESCRIPTION, WORKFLOW_TASK_TYPE,
};
use crate::agent::Agent;
use crate::card::AgentCard;
use crate::controller::schema::{DataFrame, Intent, Task, TaskResult, TaskStatus};
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
use tracing::warn;

/// System instruction the [`LLMController`] sends when no custom prompt is set.
const DEFAULT_ROUTER_PROMPT: &str =
    "You are a task router. reply with exactly one of these executor names:";

/// Name of the [`LLMController`] router.
const LLM_ROUTER_NAME: &str = "llm";

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
    use openjiuwen_session::AgentSession;
    use openjiuwen_test_support::ScriptedModel;
    use openjiuwen_workflow::{ComponentOptions, EndComponent, StartComponent, WorkflowCard};
    use serde_json::json;

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
