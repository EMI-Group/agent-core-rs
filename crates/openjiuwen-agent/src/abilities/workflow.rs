//! Workflow abilities: one [`Workflow`] exposed to another agent as a [`Tool`].
//!
//! Python parity: the workflow family of `AbilityManager` in
//! `openjiuwen/core/single_agent/ability_manager.py`, without the legacy
//! `Runner.resource_mgr` lookup — a [`WorkflowAbility`] owns its
//! `Arc<Workflow>` and derives its [`ToolCard`] from the workflow's card.
//!
//! # Payload
//!
//! The tool call's arguments are the workflow's start inputs. The input must be
//! a JSON object: a string `query` is forwarded as `query`, and every other key
//! is merged verbatim so a call can supply additional start fields (matching
//! the way an agent's input extras reach a workflow start). A `query` that is
//! present but neither a string nor `null` is a validation error, as is an
//! object that supplies neither a `query` nor any other key.
//!
//! # Session
//!
//! Every call runs in a fresh [`WorkflowSession`] whose id is the tool context's
//! `session_id`, falling back to its `invocation_id`; the workflow id is the
//! card id. The session is not reused across calls, so no workflow state leaks
//! between tool invocations.
//!
//! # Interactive workflows
//!
//! A workflow that stops in `INPUT_REQUIRED` is an [`Error::Tool`]: resuming an
//! interactive workflow from a tool call is out of scope (the tool result cannot
//! carry interrupt frames through the loop's tool-message contract), so the
//! host must run such workflows through [`crate::WorkflowAgent`] instead.
//!
//! # Rendering
//!
//! [`Tool::invoke`] returns the completed workflow result as JSON; the
//! model-visible text is the default [`Tool::render_for_llm`] rendering of that
//! value (compact JSON, or the `content` field when the result is an object with
//! one).
use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::{Tool, ToolCard, ToolContext};
use openjiuwen_session::{Session, WorkflowSession, WorkflowSessionOptions};
use openjiuwen_workflow::{Workflow, WorkflowExecutionState};
use serde_json::{json, Value};
use std::fmt;
use std::sync::Arc;

/// The input schema a workflow ability advertises to the model.
fn default_input_params() -> Value {
    json!({
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "additionalProperties": true
    })
}

/// A [`Workflow`] mounted as a [`Tool`] on an agent.
///
/// Build it with [`WorkflowAbility::new`], optionally override the
/// model-visible name/description, then mount the
/// [`WorkflowAbility::boxed`] tool on an [`crate::AbilityManager`].
pub struct WorkflowAbility {
    workflow: Arc<Workflow>,
    card: ToolCard,
}

impl WorkflowAbility {
    /// Wraps `workflow` as a tool whose card mirrors the workflow's card.
    ///
    /// The tool name is the workflow card's name (or its id when the name is
    /// empty), the description the workflow card's description, and the id the
    /// workflow card's id; the input schema is an open object with an optional
    /// `query` string.
    #[must_use]
    pub fn new(workflow: Arc<Workflow>) -> Self {
        let source = workflow.card();
        let name = if source.name.is_empty() {
            source.id.clone()
        } else {
            source.name.clone()
        };
        let card = ToolCard::new(name, source.description.clone(), default_input_params())
            .with_id(source.id.clone());
        Self { workflow, card }
    }

    /// Returns this ability with the model-visible tool `name`.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.card.name = name.into();
        self
    }

    /// Returns this ability with the model-visible tool `description`.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.card.description = description.into();
        self
    }

    /// Returns the wrapped workflow.
    #[must_use]
    pub fn workflow(&self) -> &Arc<Workflow> {
        &self.workflow
    }

    /// Wraps this ability in the `Arc<dyn Tool>` a registry mounts.
    #[must_use]
    pub fn boxed(self) -> Arc<dyn Tool> {
        Arc::new(self)
    }

    /// Returns the tool name to use in messages, falling back to the id.
    fn label(&self) -> &str {
        if self.card.name.is_empty() {
            &self.card.id
        } else {
            &self.card.name
        }
    }
}

impl fmt::Debug for WorkflowAbility {
    /// Prints the tool card and the wrapped workflow identity; the workflow
    /// itself is printed by its runtime-only `Debug` implementation.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WorkflowAbility")
            .field("card", &self.card)
            .field("workflow_id", &self.workflow.card().id)
            .field("workflow_name", &self.workflow.card().name)
            .finish()
    }
}

#[async_trait]
impl Tool for WorkflowAbility {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    /// Runs the workflow once with the call's arguments as start inputs.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an input that is not a JSON object, a
    /// non-string `query`, or an object with neither a `query` nor other keys;
    /// [`Error::Tool`] when the workflow fails to run, stops for interactive
    /// input, or ends in a state other than `COMPLETED`.
    async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value> {
        let payload = build_payload(self.label(), inputs)?;
        let session_id = ctx
            .session_id
            .clone()
            .unwrap_or_else(|| ctx.invocation_id.clone());
        let session: Arc<dyn Session> =
            Arc::new(WorkflowSession::with_options(WorkflowSessionOptions {
                session_id: Some(session_id),
                workflow_id: Some(self.card.id.clone()),
                ..WorkflowSessionOptions::default()
            }));
        let output = self
            .workflow
            .invoke(payload, session)
            .await
            .map_err(|error| Error::tool(format!("Workflow '{}' failed: {error}", self.label())))?;
        match output.state {
            WorkflowExecutionState::Completed => Ok(output.result),
            WorkflowExecutionState::InputRequired => Err(Error::tool(format!(
                "Workflow '{}' requires interactive input (INPUT_REQUIRED); resuming an \
                 interactive workflow from a tool call is not supported",
                self.label()
            ))),
            other @ WorkflowExecutionState::Error => Err(Error::tool(format!(
                "Workflow '{}' finished in state {} instead of COMPLETED",
                self.label(),
                other.as_str()
            ))),
        }
    }
}

/// Builds the workflow start payload from the tool-call arguments.
///
/// A string `query` is forwarded first, then every other key of the argument
/// object is merged verbatim. A `null` `query` is treated as absent.
fn build_payload(tool: &str, inputs: Value) -> Result<Value> {
    let Value::Object(map) = inputs else {
        return Err(Error::validation(format!(
            "Workflow tool '{tool}' expects a JSON object of workflow inputs"
        )));
    };
    let query = match map.get("query") {
        Some(Value::String(query)) => Some(query.clone()),
        Some(Value::Null) | None => None,
        Some(_) => {
            return Err(Error::validation(format!(
                "Workflow tool '{tool}' requires `query` to be a string"
            )))
        }
    };
    let mut payload = serde_json::Map::new();
    if let Some(query) = query {
        payload.insert("query".to_owned(), Value::String(query));
    }
    for (key, value) in &map {
        if key != "query" {
            payload.insert(key.clone(), value.clone());
        }
    }
    if payload.is_empty() {
        return Err(Error::validation(format!(
            "Workflow tool '{tool}' requires a `query` string or other workflow input keys"
        )));
    }
    Ok(Value::Object(payload))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::Error;
    use openjiuwen_workflow::{ComponentOptions, EndComponent, StartComponent, WorkflowCard};
    use serde_json::json;

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

    /// `start → end`, projecting an extra start input to the end output.
    fn extra_workflow(id: &str) -> Arc<Workflow> {
        let mut workflow = Workflow::new(WorkflowCard::new(id));
        workflow
            .set_start_comp(
                "start",
                StartComponent::new(),
                ComponentOptions::new().with_inputs_schema(json!({"extra": "${inputs.extra}"})),
            )
            .set_end_comp(
                "end",
                EndComponent::new(),
                ComponentOptions::new()
                    .with_inputs_schema(json!({"answer": "${inputs.extra}"}))
                    .with_outputs_schema(json!({"answer": "${output.output.answer}"})),
            )
            .add_connection("start", "end");
        Arc::new(workflow)
    }

    fn ctx() -> ToolContext {
        ToolContext::new()
    }

    #[test]
    fn new_builds_the_card_from_the_workflow_card() {
        let mut source = WorkflowCard::new("wf-id");
        source.name = "Echo".to_owned();
        source.description = "echoes the query".to_owned();
        let ability = WorkflowAbility::new(Arc::new(Workflow::new(source)));
        let card = ability.card();
        assert_eq!(card.id, "wf-id");
        assert_eq!(card.name, "Echo");
        assert_eq!(card.description, "echoes the query");
        assert_eq!(
            card.input_params,
            json!({
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "additionalProperties": true
            })
        );
        assert_eq!(ability.workflow().card().id, "wf-id");
    }

    #[test]
    fn card_name_falls_back_to_the_workflow_id() {
        let mut source = WorkflowCard::new("wf-id");
        source.name = String::new();
        let ability = WorkflowAbility::new(Arc::new(Workflow::new(source)));
        assert_eq!(ability.card().name, "wf-id");
        assert_eq!(ability.card().id, "wf-id");
    }

    #[test]
    fn builders_override_name_and_description() {
        let ability = WorkflowAbility::new(echo_workflow("echo"))
            .with_name("echo_flow")
            .with_description("runs the echo workflow");
        assert_eq!(ability.card().name, "echo_flow");
        assert_eq!(ability.card().description, "runs the echo workflow");
        assert_eq!(ability.card().id, "echo");
        let rendered = format!("{ability:?}");
        assert!(rendered.contains("echo_flow"));
        assert!(rendered.contains("workflow_id: \"echo\""));
    }

    #[tokio::test]
    async fn invoke_returns_the_completed_result() {
        let ability = WorkflowAbility::new(echo_workflow("echo"));
        let value = ability
            .invoke(json!({"query": "hi"}), &ctx())
            .await
            .unwrap();
        assert_eq!(value, json!({"answer": "hi"}));
    }

    #[tokio::test]
    async fn invoke_merges_extra_keys_into_the_payload() {
        let ability = WorkflowAbility::new(extra_workflow("extra"));
        let value = ability
            .invoke(json!({"query": "hi", "extra": "v"}), &ctx())
            .await
            .unwrap();
        assert_eq!(value, json!({"answer": "v"}));
    }

    #[tokio::test]
    async fn invoke_allows_extra_keys_without_a_query() {
        let ability = WorkflowAbility::new(extra_workflow("extra"));
        let value = ability.invoke(json!({"extra": "v"}), &ctx()).await.unwrap();
        assert_eq!(value, json!({"answer": "v"}));
    }

    #[tokio::test]
    async fn empty_object_is_a_validation_error() {
        let ability = WorkflowAbility::new(echo_workflow("echo"));
        let error = ability.invoke(json!({}), &ctx()).await.unwrap_err();
        assert!(matches!(error, Error::Validation(_)), "unexpected: {error}");
        assert!(error.message().contains("requires a `query` string"));
    }

    #[tokio::test]
    async fn null_query_without_other_keys_is_a_validation_error() {
        let ability = WorkflowAbility::new(echo_workflow("echo"));
        let error = ability
            .invoke(json!({"query": null}), &ctx())
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)), "unexpected: {error}");
    }

    #[tokio::test]
    async fn non_string_query_is_a_validation_error() {
        let ability = WorkflowAbility::new(echo_workflow("echo"));
        let error = ability
            .invoke(json!({"query": 3}), &ctx())
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)), "unexpected: {error}");
        assert!(error.message().contains("`query` to be a string"));
    }

    #[tokio::test]
    async fn non_object_input_is_a_validation_error() {
        let ability = WorkflowAbility::new(echo_workflow("echo"));
        let error = ability.invoke(json!(["query"]), &ctx()).await.unwrap_err();
        assert!(matches!(error, Error::Validation(_)), "unexpected: {error}");
        assert!(error.message().contains("expects a JSON object"));
    }

    #[tokio::test]
    async fn failing_workflow_is_a_tool_error() {
        let ability = WorkflowAbility::new(Arc::new(Workflow::new(WorkflowCard::new("broken"))));
        let error = ability
            .invoke(json!({"query": "hi"}), &ctx())
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Tool(_)), "unexpected: {error}");
        assert!(error.message().contains("Workflow 'broken' failed"));
    }

    #[tokio::test]
    async fn boxed_exposes_the_ability_as_a_tool() {
        let tool = WorkflowAbility::new(echo_workflow("echo")).boxed();
        assert_eq!(tool.card().name, "echo");
        let value = tool.invoke(json!({"query": "hi"}), &ctx()).await.unwrap();
        assert_eq!(value, json!({"answer": "hi"}));
    }
}
