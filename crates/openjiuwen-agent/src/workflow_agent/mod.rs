//! `WorkflowAgent`: run one of a set of workflows, with optional LLM intent
//! detection and human-in-the-loop resume.
//!
//! Python parity: `openjiuwen/core/application/workflow_agent/`, without the
//! legacy `WorkflowController` / `Runner.resource_mgr` machinery.
//!
//! The agent owns a [`AgentCard`] derived from its [`WorkflowAgentConfig`], a
//! set of shared [`Workflow`] values, and an optional [`Model`] used only for
//! intent detection when more than one workflow is configured.
//!
//! Intended invocation contract (the loop itself lands in a follow-up round):
//!
//! - the agent session is resolved as for any agent, and the workflow runs in a
//!   child `WorkflowSession` whose `parent_session_id` is the agent session id;
//! - a text query is dispatched to the start component as `{"query": ...}`
//!   merged with [`AgentInput::extra`](crate::input::AgentInput::extra) and the
//!   configured `global_params`; an [`InteractiveInput`](openjiuwen_session::InteractiveInput)
//!   query (or a stored interrupted workflow) selects `Workflow::resume`;
//! - a completed run returns `result_type = Answer`, with `output` holding the
//!   JSON text of the workflow result and the structured output in
//!   `workflow_output` / `data`;
//! - an interrupted run returns `result_type = Interrupt`,
//!   `workflow_output.state == InputRequired`, and the `__interaction__` frames;
//! - when workflow selection fails, the configured
//!   [`DefaultResponse`] is returned with `data["status"] = "default_response"`.
pub mod config;
mod select;

pub use crate::llm_agent::ConstrainConfig;
pub use config::{DefaultResponse, WorkflowAgentConfig};

use crate::agent::Agent;
use crate::card::AgentCard;
use crate::output::{AgentChunk, AgentOutput};
use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::Model;
use openjiuwen_session::Session;
use openjiuwen_workflow::Workflow;
use serde_json::Value;
use std::fmt;
use std::sync::Arc;

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

    /// Not implemented yet: the workflow dispatch/resume loop lands in a
    /// follow-up round.
    ///
    /// # Errors
    ///
    /// Always returns [`Error::unsupported`] until the loop is implemented.
    async fn invoke(
        &self,
        _inputs: Value,
        _session: Option<Arc<dyn Session>>,
    ) -> Result<AgentOutput> {
        // TODO: resolve the agent session, create the child `WorkflowSession`,
        // select a workflow, and run `Workflow::invoke`/`Workflow::resume`.
        Err(Error::unsupported("workflow agent not implemented"))
    }

    /// Not implemented yet: forwards workflow frames once the loop exists.
    fn stream(
        &self,
        _inputs: Value,
        _session: Option<Arc<dyn Session>>,
    ) -> BoxStream<'static, Result<AgentChunk>> {
        // TODO: forward `workflow_final` / `__interaction__` / component frames
        // and mirror them into the session.
        futures::stream::once(async { Err(Error::unsupported("workflow agent not implemented")) })
            .boxed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_workflow::WorkflowCard;
    use serde_json::json;

    fn workflow(id: &str) -> Arc<Workflow> {
        Arc::new(Workflow::new(WorkflowCard::new(id)))
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
    async fn invoke_is_unsupported_for_now() {
        let agent = agent();
        let error = agent
            .invoke(json!({"query": "hi"}), None)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
    }

    #[tokio::test]
    async fn stream_yields_a_single_unsupported_error() {
        let agent = agent();
        let frames = agent
            .stream(json!({"query": "hi"}), None)
            .collect::<Vec<_>>()
            .await;
        assert_eq!(frames.len(), 1);
        let error = frames
            .into_iter()
            .next()
            .expect("one frame")
            .expect_err("unsupported");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
    }
}
