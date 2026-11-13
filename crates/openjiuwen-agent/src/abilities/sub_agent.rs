//! Sub-agent abilities: one [`Agent`] exposed to another agent as a [`Tool`].
//!
//! Python parity: the sub-agent family of `AbilityManager` in
//! `openjiuwen/core/single_agent/ability_manager.py`, without the legacy
//! `Runner.resource_mgr` lookup — a [`SubAgentAbility`] owns its
//! `Arc<dyn Agent>` and its [`ToolCard`] is derived from the child's card.
//!
//! # Session handling
//!
//! A [`ToolContext`] carries only a `session_id`, not an
//! `Arc<dyn Session>`, so the child's session is fixed at construction: a
//! caller that wants the child to share a session passes one with
//! [`SubAgentAbility::with_session`]; otherwise the child runs without a
//! session. This is deliberate — the tool call cannot rebuild the parent's
//! session object from its id.
//!
//! # Input and result mapping
//!
//! The tool accepts a bare JSON string or an object with a string `query`, and
//! always forwards `{"query": ...}` to the child agent. An `answer` result
//! becomes the child's answer text ([`Value::String`]); an `error` or
//! `interrupt` result becomes an [`Error::Tool`] naming the child agent and its
//! result type, so the calling model can react to the failure.
use crate::card::AgentCard;
use crate::output::{AgentOutput, ResultType};
use crate::Agent;
use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::{Tool, ToolCard, ToolContext};
use openjiuwen_session::Session;
use serde_json::{json, Value};
use std::fmt;
use std::sync::Arc;

/// The input schema a sub-agent ability advertises to the model.
fn default_input_params() -> Value {
    json!({
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"]
    })
}

/// An [`Agent`] mounted as a [`Tool`] on another agent.
///
/// Build it with [`SubAgentAbility::new`], optionally override the model-visible
/// name/description, then mount the [`SubAgentAbility::boxed`] tool on an
/// [`crate::AbilityManager`].
pub struct SubAgentAbility {
    agent: Arc<dyn Agent>,
    card: ToolCard,
    session: Option<Arc<dyn Session>>,
}

impl SubAgentAbility {
    /// Wraps `agent` as a tool whose card mirrors the child's `AgentCard`.
    ///
    /// The tool name is the child's card name, the description the child's card
    /// description, and the id the child's card id; the input schema is a single
    /// required `query` string.
    #[must_use]
    pub fn new(agent: Arc<dyn Agent>) -> Self {
        let child = agent.card();
        let card = ToolCard::new(
            child.name.clone(),
            child.description.clone(),
            default_input_params(),
        )
        .with_id(child.id.clone());
        Self {
            agent,
            card,
            session: None,
        }
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

    /// Returns this ability with the session every child invocation shares.
    #[must_use]
    pub fn with_session(mut self, session: Arc<dyn Session>) -> Self {
        self.session = Some(session);
        self
    }

    /// Returns the wrapped child agent.
    #[must_use]
    pub fn agent(&self) -> &Arc<dyn Agent> {
        &self.agent
    }

    /// Returns the session passed to the child, if any.
    #[must_use]
    pub fn session(&self) -> Option<&Arc<dyn Session>> {
        self.session.as_ref()
    }

    /// Wraps this ability in the `Arc<dyn Tool>` a registry mounts.
    #[must_use]
    pub fn boxed(self) -> Arc<dyn Tool> {
        Arc::new(self)
    }
}

impl fmt::Debug for SubAgentAbility {
    /// Prints the tool card and the child agent identity; the child itself is
    /// an opaque trait object.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let child = self.agent.card();
        formatter
            .debug_struct("SubAgentAbility")
            .field("card", &self.card)
            .field("agent", &child.name)
            .field("agent_id", &child.id)
            .field("has_session", &self.session.is_some())
            .finish()
    }
}

#[async_trait]
impl Tool for SubAgentAbility {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    /// Runs the child agent once with the call's `query`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the input is neither a JSON string nor
    /// an object with a string `query`, the child's own error when the child
    /// fails to run, and [`Error::Tool`] when the child finishes with an
    /// `error` or `interrupt` result.
    async fn invoke(&self, inputs: Value, _ctx: &ToolContext) -> Result<Value> {
        let query = extract_query(&self.card.name, &inputs)?;
        let output = self
            .agent
            .invoke(json!({"query": query}), self.session.clone())
            .await?;
        if output.result_type == ResultType::Answer {
            return Ok(Value::String(output.output));
        }
        Err(Error::tool(failure_message(self.agent.card(), &output)))
    }
}

/// Extracts the `query` from a bare string or an object argument.
fn extract_query(tool: &str, inputs: &Value) -> Result<String> {
    match inputs {
        Value::String(query) => Ok(query.clone()),
        Value::Object(map) => match map.get("query") {
            Some(Value::String(query)) => Ok(query.clone()),
            Some(_) => Err(Error::validation(format!(
                "Sub-agent tool '{tool}' requires `query` to be a string"
            ))),
            None => Err(Error::validation(format!(
                "Sub-agent tool '{tool}' requires a `query` string"
            ))),
        },
        _ => Err(Error::validation(format!(
            "Sub-agent tool '{tool}' expects a JSON string or an object with a string `query`"
        ))),
    }
}

/// Renders a non-answer child result as the tool error message.
fn failure_message(child: &AgentCard, output: &AgentOutput) -> String {
    let result_type = match output.result_type {
        ResultType::Answer => "answer",
        ResultType::Error => "error",
        ResultType::Interrupt => "interrupt",
    };
    let mut message = format!(
        "Sub-agent '{}' (id '{}') finished with result_type '{result_type}'",
        child.name, child.id
    );
    if !output.output.is_empty() {
        message.push_str(": ");
        message.push_str(&output.output);
    }
    message
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use futures::stream::BoxStream;
    use futures::StreamExt;
    use openjiuwen_common::error::{Error, Result};
    use openjiuwen_session::AgentSession;
    use serde_json::json;
    use std::sync::Mutex;

    struct FakeAgent {
        card: AgentCard,
        calls: Arc<Mutex<Vec<Value>>>,
        output: AgentOutput,
    }

    #[async_trait]
    impl Agent for FakeAgent {
        fn card(&self) -> &AgentCard {
            &self.card
        }

        async fn invoke(
            &self,
            inputs: Value,
            _session: Option<Arc<dyn Session>>,
        ) -> Result<AgentOutput> {
            self.calls.lock().unwrap().push(inputs);
            Ok(self.output.clone())
        }

        fn stream(
            &self,
            _inputs: Value,
            _session: Option<Arc<dyn Session>>,
        ) -> BoxStream<'static, Result<crate::output::AgentChunk>> {
            futures::stream::empty().boxed()
        }
    }

    fn fake(output: AgentOutput) -> (Arc<dyn Agent>, Arc<Mutex<Vec<Value>>>) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let agent = FakeAgent {
            card: AgentCard::new("child")
                .with_id("child-id")
                .with_description("does child work"),
            calls: Arc::clone(&calls),
            output,
        };
        (Arc::new(agent), calls)
    }

    fn ctx() -> ToolContext {
        ToolContext::new()
    }

    #[test]
    fn new_inherits_the_child_card_and_advertises_a_query() {
        let (agent, _) = fake(AgentOutput::answer("done"));
        let ability = SubAgentAbility::new(agent);
        let card = ability.card();
        assert_eq!(card.name, "child");
        assert_eq!(card.id, "child-id");
        assert_eq!(card.description, "does child work");
        assert_eq!(
            card.input_params,
            json!({
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            })
        );
        assert!(ability.session().is_none());
        assert_eq!(ability.agent().card().id, "child-id");
    }

    #[test]
    fn builders_override_name_and_description() {
        let (agent, _) = fake(AgentOutput::answer("done"));
        let ability = SubAgentAbility::new(agent)
            .with_name("helper")
            .with_description("helps with things");
        assert_eq!(ability.card().name, "helper");
        assert_eq!(ability.card().description, "helps with things");
        assert_eq!(ability.card().id, "child-id");
    }

    #[test]
    fn with_session_stores_the_child_session() {
        let (agent, _) = fake(AgentOutput::answer("done"));
        let ability = SubAgentAbility::new(agent).with_session(Arc::new(AgentSession::new("s1")));
        assert_eq!(ability.session().unwrap().id(), "s1");
        let rendered = format!("{ability:?}");
        assert!(rendered.contains("has_session: true"));
        assert!(rendered.contains("child-id"));
    }

    #[tokio::test]
    async fn invoke_returns_the_answer_and_forwards_the_query() {
        let (agent, calls) = fake(AgentOutput::answer("child answer"));
        let ability = SubAgentAbility::new(agent);
        let value = ability
            .invoke(json!({"query": "do it"}), &ctx())
            .await
            .unwrap();
        assert_eq!(value, json!("child answer"));
        let recorded = calls.lock().unwrap().clone();
        assert_eq!(recorded, vec![json!({"query": "do it"})]);
    }

    #[tokio::test]
    async fn bare_string_query_is_accepted() {
        let (agent, calls) = fake(AgentOutput::answer("ok"));
        let ability = SubAgentAbility::new(agent);
        let value = ability.invoke(json!("plain"), &ctx()).await.unwrap();
        assert_eq!(value, json!("ok"));
        assert_eq!(
            calls.lock().unwrap().clone(),
            vec![json!({"query": "plain"})]
        );
    }

    #[tokio::test]
    async fn missing_query_is_a_validation_error() {
        let (agent, calls) = fake(AgentOutput::answer("ok"));
        let ability = SubAgentAbility::new(agent);
        let error = ability
            .invoke(json!({"other": 1}), &ctx())
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)), "unexpected: {error}");
        assert!(error.message().contains("requires a `query` string"));
        assert!(calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn non_string_query_is_a_validation_error() {
        let (agent, calls) = fake(AgentOutput::answer("ok"));
        let ability = SubAgentAbility::new(agent);
        let error = ability
            .invoke(json!({"query": 3}), &ctx())
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)), "unexpected: {error}");
        assert!(error.message().contains("`query` to be a string"));
        assert!(calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn non_object_non_string_input_is_a_validation_error() {
        let (agent, _) = fake(AgentOutput::answer("ok"));
        let ability = SubAgentAbility::new(agent);
        let error = ability.invoke(json!([1, 2]), &ctx()).await.unwrap_err();
        assert!(matches!(error, Error::Validation(_)), "unexpected: {error}");
    }

    #[tokio::test]
    async fn error_result_becomes_a_tool_error_naming_the_child() {
        let (agent, _) = fake(AgentOutput::error("model exploded"));
        let ability = SubAgentAbility::new(agent);
        let error = ability
            .invoke(json!({"query": "go"}), &ctx())
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Tool(_)), "unexpected: {error}");
        let message = error.message().to_string();
        assert!(message.contains("child"));
        assert!(message.contains("child-id"));
        assert!(message.contains("result_type 'error'"));
        assert!(message.contains("model exploded"));
    }

    #[tokio::test]
    async fn interrupt_result_becomes_a_tool_error() {
        let (agent, _) = fake(AgentOutput::interrupt(Vec::new(), vec!["in-1".to_owned()]));
        let ability = SubAgentAbility::new(agent);
        let error = ability
            .invoke(json!({"query": "go"}), &ctx())
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Tool(_)), "unexpected: {error}");
        assert!(error.message().contains("result_type 'interrupt'"));
    }

    #[tokio::test]
    async fn boxed_exposes_the_ability_as_a_tool() {
        let (agent, _) = fake(AgentOutput::answer("ok"));
        let tool = SubAgentAbility::new(agent).boxed();
        assert_eq!(tool.card().name, "child");
        let value = tool.invoke(json!("hi"), &ctx()).await.unwrap();
        assert_eq!(value, json!("ok"));
    }
}
