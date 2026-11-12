//! `WorkflowAgent`: run one of a set of workflows, with optional LLM intent
//! detection and human-in-the-loop resume.
//!
//! Python parity: `openjiuwen/core/application/workflow_agent/`, without the
//! legacy `WorkflowController` / `Runner.resource_mgr` machinery.
//!
//! The agent owns an [`AgentCard`] derived from its [`WorkflowAgentConfig`], a
//! set of shared [`Workflow`] values, and an optional [`Model`] used only for
//! intent detection when more than one workflow is configured.
//!
//! ## Invocation
//!
//! - The agent session is resolved as for any agent; the workflow runs in a
//!   child `WorkflowSession` whose `parent` is the agent session, so it shares
//!   that session's stream hub, configuration, checkpointer, and trace id.
//! - A text query is dispatched to the workflow with
//!   [`AgentInput::extra`](crate::input::AgentInput::extra) (filtered to the
//!   workflow's declared input properties) merged over the configured
//!   `global_params`, plus `"query"`. An
//!   [`InteractiveInput`](openjiuwen_session::InteractiveInput) query selects
//!   `Workflow::resume` instead.
//! - A completed run returns `result_type = Answer`, with `output` holding the
//!   JSON text of the workflow result and the structured output in
//!   `workflow_output` / `data`.
//! - An interrupted run returns `result_type = Interrupt`,
//!   `workflow_output.state == InputRequired`, and the `__interaction__` frames
//!   the engine already published on the shared hub (they are not re-written).
//! - When workflow selection fails, the configured [`DefaultResponse`] is
//!   returned with `data["status"] = "default_response"`.
//!
//! ## Streaming
//!
//! [`Agent::stream`] forwards every workflow frame verbatim and then emits
//! exactly one terminal `answer` frame: the interrupt output when the run
//! paused, otherwise the completed (or failed) output.
pub mod config;
mod select;

pub use crate::llm_agent::ConstrainConfig;
pub use config::{DefaultResponse, WorkflowAgentConfig};

use crate::agent::Agent;
use crate::card::AgentCard;
use crate::input::AgentInput;
use crate::output::{AgentChunk, AgentOutput, OutputFrame, INTERACTION, WORKFLOW_FINAL};
use crate::run::resolve_session;
use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_foundation::{Model, ModelRequestConfig};
use openjiuwen_session::{Session, WorkflowSession, WorkflowSessionOptions};
use openjiuwen_workflow::{Workflow, WorkflowExecutionState, WorkflowOutput, WorkflowRunOptions};
use select::Selection;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

/// The error message Python raises when no workflow is registered.
const NO_WORKFLOWS: &str = "No workflows configured for single_agent";

/// An agent that answers by running one of a set of workflows.
///
/// The card comes from [`WorkflowAgentConfig`]; workflows are shared as
/// `Arc<Workflow>` and never mutated by the agent, so they may run
/// concurrently. The optional model is used only to pick a workflow when more
/// than one is configured.
pub struct WorkflowAgent {
    card: AgentCard,
    config: parking_lot::RwLock<WorkflowAgentConfig>,
    workflows: parking_lot::RwLock<Vec<Arc<Workflow>>>,
    model: parking_lot::RwLock<Option<Arc<dyn Model>>>,
}

impl WorkflowAgent {
    /// Creates an agent for `config`, deriving its card from the configured
    /// `id`, `name`, and `description`.
    #[must_use]
    pub fn new(config: WorkflowAgentConfig) -> Self {
        let card = AgentCard::new(config.name.clone())
            .with_id(config.id.clone())
            .with_description(config.description.clone());
        Self {
            card,
            config: parking_lot::RwLock::new(config),
            workflows: parking_lot::RwLock::new(Vec::new()),
            model: parking_lot::RwLock::new(None),
        }
    }

    /// Adds workflows, skipping any whose card id is already registered.
    ///
    /// Duplicates inside the incoming iterator are skipped as well, so the
    /// registration order of the first occurrence is preserved.
    pub fn add_workflows(&self, workflows: impl IntoIterator<Item = Arc<Workflow>>) {
        let mut guard = self.workflows.write();
        for workflow in workflows {
            let id = workflow.card().id.clone();
            if guard.iter().any(|existing| existing.card().id == id) {
                continue;
            }
            guard.push(workflow);
        }
    }

    /// Removes the workflow with card id `id`.
    ///
    /// Returns `true` when a workflow was removed, `false` when no workflow has
    /// that id.
    #[must_use]
    pub fn remove_workflow(&self, id: &str) -> bool {
        let mut guard = self.workflows.write();
        match guard.iter().position(|workflow| workflow.card().id == id) {
            Some(position) => {
                guard.remove(position);
                true
            }
            None => false,
        }
    }

    /// Returns the registered workflows in registration order.
    #[must_use]
    pub fn workflows(&self) -> Vec<Arc<Workflow>> {
        self.workflows.read().clone()
    }

    /// Returns a copy of the agent configuration.
    #[must_use]
    pub fn config(&self) -> WorkflowAgentConfig {
        self.config.read().clone()
    }

    /// Sets the model used for intent detection when several workflows are
    /// registered; it is never used to answer the query itself.
    pub fn set_model(&self, model: Arc<dyn Model>) {
        *self.model.write() = Some(model);
    }

    /// Runs the selected workflow on `session` (the already-resolved agent
    /// session).
    async fn invoke_body(
        &self,
        input: AgentInput,
        session: &Arc<dyn Session>,
    ) -> Result<AgentOutput> {
        let workflows = self.workflows();
        if workflows.is_empty() {
            return Err(Error::workflow(NO_WORKFLOWS));
        }
        let config = self.config();
        let model = self.model.read().clone();
        let request = model
            .as_ref()
            .map_or_else(ModelRequestConfig::default, |model| {
                ModelRequestConfig::new(model.model_name())
            });
        let selection = select::select(
            &workflows,
            input.query_text().unwrap_or_default(),
            model.as_ref(),
            &request,
            config.default_response.text.as_deref(),
        )
        .await?;
        let workflow = match selection {
            Selection::Default(text) => return Ok(default_response_output(&text)),
            Selection::Workflow(index) => workflows
                .get(index)
                .ok_or_else(|| Error::workflow(NO_WORKFLOWS))?,
        };
        let workflow_session = child_session(workflow, session);
        let output = if let Some(interactive) = input.interactive() {
            workflow
                .resume(interactive.clone(), workflow_session)
                .await?
        } else {
            let payload = build_payload(
                &input,
                &config.global_params,
                workflow.card().input_params.as_ref(),
            );
            workflow.invoke(payload, workflow_session).await?
        };
        map_workflow_output(output)
    }
}

/// The child `WorkflowSession` for `workflow`, sharing `parent`'s hub,
/// configuration, checkpointer, and trace id.
fn child_session(workflow: &Arc<Workflow>, parent: &Arc<dyn Session>) -> Arc<dyn Session> {
    let workflow_id = workflow.card().id.clone();
    let session_id = format!("{}::{workflow_id}", parent.id());
    Arc::new(WorkflowSession::with_options(WorkflowSessionOptions {
        session_id: Some(session_id),
        workflow_id: Some(workflow_id),
        parent: Some(Arc::clone(parent)),
        ..WorkflowSessionOptions::default()
    })) as Arc<dyn Session>
}

/// Builds the workflow start inputs: `global_params`, then the caller extras
/// the workflow declares, then `"query"`.
///
/// When the card's `input_params` carries a `properties` object, only extras
/// named in it are forwarded; otherwise every extra is forwarded.
fn build_payload(
    input: &AgentInput,
    global_params: &BTreeMap<String, Value>,
    input_params: Option<&Value>,
) -> Value {
    let mut payload: serde_json::Map<String, Value> = global_params
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    let declared = input_params
        .and_then(|schema| schema.get("properties"))
        .and_then(Value::as_object);
    for (key, value) in &input.extra {
        if declared.is_none_or(|properties| properties.contains_key(key)) {
            payload.insert(key.clone(), value.clone());
        }
    }
    payload.insert(
        "query".to_owned(),
        Value::String(input.query_text().unwrap_or_default().to_owned()),
    );
    Value::Object(payload)
}

/// The default response output for a failed workflow selection.
fn default_response_output(text: &str) -> AgentOutput {
    AgentOutput::answer(text)
        .with_data(json!({"answer": text, "status": "default_response"}))
        .with_extra("status", json!("default_response"))
}

/// Projects a workflow output onto the agent output contract.
fn map_workflow_output(output: WorkflowOutput) -> Result<AgentOutput> {
    match output.state {
        WorkflowExecutionState::Completed => {
            let text = serde_json::to_string(&output.result)?;
            let mut result = AgentOutput::answer(text).with_data(output.result.clone());
            result.workflow_output = Some(output);
            Ok(result)
        }
        WorkflowExecutionState::InputRequired => {
            let frames = interaction_frames(&output.result);
            let ids = frames
                .iter()
                .filter_map(|frame| {
                    frame
                        .payload
                        .get("id")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                })
                .collect();
            let mut result = AgentOutput::interrupt(frames, ids);
            result.workflow_output = Some(output);
            Ok(result)
        }
        WorkflowExecutionState::Error => {
            let message = match &output.result {
                Value::String(text) => text.clone(),
                other => serde_json::to_string(other).unwrap_or_default(),
            };
            let mut result = AgentOutput::error(message).with_termination_reason("workflow_error");
            result.workflow_output = Some(output);
            Ok(result)
        }
    }
}

/// Collects `__interaction__` frames from an interrupted run's wire result.
fn interaction_frames(result: &Value) -> Vec<OutputFrame> {
    let Some(items) = result.as_array() else {
        return Vec::new();
    };
    items.iter().filter_map(frame_from_wire).collect()
}

/// Reads one `{"type", "index", "payload"}` wire frame when it is an
/// interaction frame.
fn frame_from_wire(value: &Value) -> Option<OutputFrame> {
    if value.get("type").and_then(Value::as_str) != Some(INTERACTION) {
        return None;
    }
    Some(OutputFrame {
        kind: INTERACTION.to_owned(),
        index: value.get("index").and_then(Value::as_i64).unwrap_or(0),
        payload: value.get("payload").cloned().unwrap_or(Value::Null),
    })
}

/// Everything [`WorkflowAgent::stream`] snapshots before its task starts.
struct StreamSnapshot {
    agent_id: String,
    agent_name: String,
    config: WorkflowAgentConfig,
    workflows: Vec<Arc<Workflow>>,
    model: Option<Arc<dyn Model>>,
}

/// Runs the streaming body on the already-resolved agent session and finishes
/// it afterwards.
async fn stream_run(
    session: Option<Arc<dyn Session>>,
    input: AgentInput,
    snapshot: StreamSnapshot,
    tx: &tokio::sync::mpsc::UnboundedSender<Result<AgentChunk>>,
) -> Result<()> {
    let run = resolve_session(
        session,
        &snapshot.agent_id,
        Some(&snapshot.agent_name),
        input.conversation_id.as_deref(),
        input.parent_session_id.as_deref(),
    )
    .await?;
    let result = stream_workflow(&input, &snapshot, run.session(), tx).await;
    run.finish().await;
    result
}

/// Selects and streams the workflow, forwarding frames and sending exactly one
/// terminal `answer` frame.
async fn stream_workflow(
    input: &AgentInput,
    snapshot: &StreamSnapshot,
    session: &Arc<dyn Session>,
    tx: &tokio::sync::mpsc::UnboundedSender<Result<AgentChunk>>,
) -> Result<()> {
    let model = snapshot.model.as_ref();
    let request = model.map_or_else(ModelRequestConfig::default, |model| {
        ModelRequestConfig::new(model.model_name())
    });
    let selection = select::select(
        &snapshot.workflows,
        input.query_text().unwrap_or_default(),
        model,
        &request,
        snapshot.config.default_response.text.as_deref(),
    )
    .await?;
    let workflow = match selection {
        Selection::Default(text) => {
            let output = default_response_output(&text);
            let _ = tx.send(Ok(AgentChunk::answer(&output)));
            return Ok(());
        }
        Selection::Workflow(index) => snapshot
            .workflows
            .get(index)
            .ok_or_else(|| Error::workflow(NO_WORKFLOWS))?,
    };
    let workflow_session = child_session(workflow, session);
    let payload = build_payload(
        input,
        &snapshot.config.global_params,
        workflow.card().input_params.as_ref(),
    );
    let options = WorkflowRunOptions {
        interactive: input.interactive().cloned(),
        resume: input.interactive().is_some(),
        ..WorkflowRunOptions::default()
    };
    let mut frames = Vec::new();
    let mut ids = Vec::new();
    let mut saw_interaction = false;
    let mut final_payload = None;
    let mut stream = workflow.stream_with(payload, workflow_session, options);
    while let Some(item) = stream.next().await {
        let item = item?;
        if item.item_type == INTERACTION {
            saw_interaction = true;
            let index = item
                .payload
                .get("index")
                .and_then(Value::as_i64)
                .or(item.index)
                .unwrap_or(0);
            if let Some(id) = item.payload.get("id").and_then(Value::as_str) {
                ids.push(id.to_owned());
            }
            frames.push(OutputFrame {
                kind: INTERACTION.to_owned(),
                index,
                payload: item.payload.clone(),
            });
        } else if item.item_type == WORKFLOW_FINAL {
            final_payload = Some(item.payload.clone());
        }
        let chunk = AgentChunk::new(
            item.item_type.clone(),
            item.index.unwrap_or(0),
            item.payload.clone(),
        );
        let _ = tx.send(Ok(chunk));
    }
    let output = terminal_stream_output(frames, ids, saw_interaction, final_payload);
    let _ = tx.send(Ok(AgentChunk::answer(&output)));
    Ok(())
}

/// The terminal output of a streaming run: an interrupt when the run paused
/// without a final frame, otherwise a completed answer.
fn terminal_stream_output(
    frames: Vec<OutputFrame>,
    ids: Vec<String>,
    saw_interaction: bool,
    final_payload: Option<Value>,
) -> AgentOutput {
    if saw_interaction && final_payload.is_none() {
        let wire = frames.iter().map(OutputFrame::to_value).collect::<Vec<_>>();
        let mut output = AgentOutput::interrupt(frames, ids);
        output.workflow_output = Some(WorkflowOutput {
            result: Value::Array(wire),
            state: WorkflowExecutionState::InputRequired,
        });
        output
    } else {
        let result = final_payload.unwrap_or(Value::Null);
        let text = serde_json::to_string(&result).unwrap_or_default();
        let mut output = AgentOutput::answer(text).with_data(result.clone());
        output.workflow_output = Some(WorkflowOutput {
            result,
            state: WorkflowExecutionState::Completed,
        });
        output
    }
}

#[allow(clippy::missing_fields_in_debug)] // `dyn Model` is not `Debug`; `has_model` reports it
impl fmt::Debug for WorkflowAgent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let workflow_ids = self
            .workflows
            .read()
            .iter()
            .map(|workflow| workflow.card().id.clone())
            .collect::<Vec<_>>();
        f.debug_struct("WorkflowAgent")
            .field("id", &self.card.id)
            .field("workflow_ids", &workflow_ids)
            .field("has_model", &self.model.read().is_some())
            .finish()
    }
}

#[async_trait::async_trait]
impl Agent for WorkflowAgent {
    fn card(&self) -> &AgentCard {
        &self.card
    }

    /// Selects a workflow and runs it to completion or to an interruption.
    ///
    /// # Errors
    ///
    /// Returns an input validation error, a workflow/session error (including a
    /// resume without a checkpointer), or [`Error::workflow`] when no workflow
    /// is configured.
    async fn invoke(
        &self,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> Result<AgentOutput> {
        let input = AgentInput::parse(&inputs)?;
        let run = resolve_session(
            session,
            &self.card.id,
            Some(&self.card.name),
            input.conversation_id.as_deref(),
            input.parent_session_id.as_deref(),
        )
        .await?;
        let result = self.invoke_body(input, run.session()).await;
        run.finish().await;
        result
    }

    /// Streams the selected workflow's frames and one terminal frame.
    ///
    /// Every `Ok` workflow frame (component frames, `__interaction__`, and
    /// `workflow_final`) is forwarded verbatim. Exactly one terminal `answer`
    /// chunk follows: an interrupt output when the run paused, otherwise the
    /// completed answer (or an error output when the run failed). Invalid input
    /// and cancellation surface as a single `Err` item.
    fn stream(
        &self,
        inputs: Value,
        session: Option<Arc<dyn Session>>,
    ) -> BoxStream<'static, Result<AgentChunk>> {
        let input = match AgentInput::parse(&inputs) {
            Ok(input) => input,
            Err(error) => return futures::stream::once(async move { Err(error) }).boxed(),
        };
        let agent_id = self.card.id.clone();
        let agent_name = self.card.name.clone();
        let snapshot = StreamSnapshot {
            agent_id,
            agent_name,
            config: self.config(),
            workflows: self.workflows(),
            model: self.model.read().clone(),
        };
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Result<AgentChunk>>();
        tokio::spawn(async move {
            let result = stream_run(session, input, snapshot, &tx).await;
            match result {
                Ok(()) => {}
                Err(error) if error.kind() == ErrorKind::Cancelled => {
                    let _ = tx.send(Err(error));
                }
                Err(error) => {
                    let _ = tx.send(Ok(AgentChunk::answer(&AgentOutput::error(
                        error.to_string(),
                    ))));
                }
            }
            drop(tx);
        });
        futures::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        })
        .boxed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_workflow::{ComponentOptions, EndComponent, StartComponent, WorkflowCard};
    use serde_json::json;

    fn workflow(id: &str) -> Arc<Workflow> {
        Arc::new(Workflow::new(WorkflowCard::new(id)))
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

    fn agent() -> WorkflowAgent {
        let mut config = WorkflowAgentConfig::new("wf-agent");
        config.name = "Workflow Agent".to_owned();
        config.description = "runs workflows".to_owned();
        WorkflowAgent::new(config)
    }

    #[test]
    fn new_builds_card_from_config() {
        let agent = agent();
        assert_eq!(agent.card().id, "wf-agent");
        assert_eq!(agent.card().name, "Workflow Agent");
        assert_eq!(agent.card().description, "runs workflows");
        assert_eq!(agent.config().id, "wf-agent");
        assert!(agent.workflows().is_empty());
        assert!(format!("{agent:?}").contains("has_model: false"));
    }

    #[test]
    fn add_workflows_deduplicates_by_id() {
        let agent = agent();
        agent.add_workflows([workflow("a"), workflow("a"), workflow("b")]);
        let ids = agent
            .workflows()
            .iter()
            .map(|workflow| workflow.card().id.clone())
            .collect::<Vec<_>>();
        assert_eq!(ids, ["a", "b"]);
        agent.add_workflows([workflow("b"), workflow("c")]);
        let ids = agent
            .workflows()
            .iter()
            .map(|workflow| workflow.card().id.clone())
            .collect::<Vec<_>>();
        assert_eq!(ids, ["a", "b", "c"]);
    }

    #[test]
    fn remove_workflow_only_removes_known_ids() {
        let agent = agent();
        agent.add_workflows([workflow("a"), workflow("b")]);
        assert!(agent.remove_workflow("a"));
        assert!(!agent.remove_workflow("a"));
        assert!(!agent.remove_workflow("missing"));
        let ids = agent
            .workflows()
            .iter()
            .map(|workflow| workflow.card().id.clone())
            .collect::<Vec<_>>();
        assert_eq!(ids, ["b"]);
    }

    #[tokio::test]
    async fn invoke_without_workflows_is_a_workflow_error() {
        let agent = agent();
        let error = agent
            .invoke(json!({"query": "hi"}), None)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Workflow);
        assert!(error.to_string().contains("No workflows configured"));
    }

    #[tokio::test]
    async fn invoke_runs_the_single_workflow() {
        let agent = agent();
        agent.add_workflows([echo_workflow("echo")]);

        let output = agent.invoke(json!({"query": "hi"}), None).await.unwrap();

        assert_eq!(output.result_type, crate::output::ResultType::Answer);
        assert_eq!(output.data.as_ref().unwrap()["answer"], json!("hi"));
        assert_eq!(
            output.workflow_output.as_ref().unwrap().state,
            WorkflowExecutionState::Completed
        );
        let parsed: Value = serde_json::from_str(&output.output).unwrap();
        assert_eq!(parsed["answer"], json!("hi"));
    }

    #[test]
    fn build_payload_merges_globals_extras_and_query() {
        let input = AgentInput::parse(&json!({"query": "hi", "city": "Paris", "keep": 1})).unwrap();
        let mut global = BTreeMap::new();
        global.insert("tenant".to_owned(), json!("acme"));
        global.insert("query".to_owned(), json!("ignored"));

        let declared = json!({"type": "object", "properties": {"city": {}}});
        assert_eq!(
            build_payload(&input, &global, Some(&declared)),
            json!({"tenant": "acme", "city": "Paris", "query": "hi"})
        );
        assert_eq!(
            build_payload(&input, &BTreeMap::new(), None),
            json!({"city": "Paris", "keep": 1, "query": "hi"})
        );
    }

    #[tokio::test]
    async fn stream_without_workflows_yields_one_error_answer() {
        let agent = agent();
        let frames = agent
            .stream(json!({"query": "hi"}), None)
            .collect::<Vec<_>>()
            .await;
        assert_eq!(frames.len(), 1);
        let chunk = frames.into_iter().next().expect("one frame").unwrap();
        assert_eq!(chunk.kind, crate::output::ANSWER);
        assert_eq!(chunk.payload["result_type"], "error");
        assert!(chunk.payload["output"]
            .as_str()
            .unwrap()
            .contains("No workflows configured"));
    }

    #[tokio::test]
    async fn stream_forwards_frames_and_ends_with_one_answer() {
        let agent = agent();
        agent.add_workflows([echo_workflow("echo")]);

        let frames = agent
            .stream(json!({"query": "hi"}), None)
            .collect::<Vec<_>>()
            .await;

        assert!(frames.iter().all(Result::is_ok));
        let chunks = frames
            .into_iter()
            .map(|frame| frame.expect("forwarded frame"))
            .collect::<Vec<_>>();
        assert!(chunks
            .iter()
            .any(|chunk| chunk.kind == crate::output::WORKFLOW_FINAL));
        let terminal = chunks.last().expect("one terminal frame");
        assert_eq!(terminal.kind, crate::output::ANSWER);
        assert_eq!(terminal.payload["result_type"], "answer");
        assert_eq!(terminal.payload["data"]["answer"], json!("hi"));
    }

    #[tokio::test]
    async fn stream_rejects_malformed_input_with_one_error_item() {
        let agent = agent();
        let frames = agent
            .stream(json!({"conversation_id": "c1"}), None)
            .collect::<Vec<_>>()
            .await;
        assert_eq!(frames.len(), 1);
        let error = frames.into_iter().next().expect("one frame").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
    }
}
