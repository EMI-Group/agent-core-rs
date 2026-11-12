//! The session-based, model-driven agent builder.
//!
//! Ports the conversational entry point of
//! `openjiuwen/dev_tools/agent_builder`: a caller submits a natural-language
//! request on a session id and receives the artifact it could parse, or a
//! clarifying question.
//!
//! Python runs a multi-stage pipeline here (intention detection, clarifier,
//! designer, DSL generator and reflector, cycle checker, Mermaid rendering);
//! this port performs **one** model round trip and treats any unparseable answer
//! as [`BuildOutput::Response`]. Time comes from [`now_millis`], and the only
//! random value is the fallback DSL agent id.

use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_foundation::BaseMessage;
use serde::{Deserialize, Serialize};

use super::dsl::LlmAgentDsl;
use super::history::{DialogueMessage, HistoryManager, DEFAULT_MAX_HISTORY_SIZE};
use super::outcome::{map_state_to_status, BuildOutput, BuildResult, BuildState, BuildStatus};
use super::workflow_dsl::WorkflowDsl;
use crate::util::extract_json_fence;

/// The instruction sent to the model when an LLM agent is requested.
///
/// It names every key the parser reads, so a compliant answer round-trips
/// through [`LlmAgentDsl`] without repair.
const LLM_AGENT_SYSTEM_PROMPT: &str = r#"You are the openJiuwen agent builder.
Turn the user's request into one complete LLM-agent document.

Answer with exactly one ```json fenced code block and nothing else inside it,
using these keys:
- `agent_id`: the platform id, empty when the agent is new
- `agent_type`: the literal "react"
- `name`, `description`: a display name and a one-sentence purpose
- `configs.system_prompt`, `configs.opening_remarks`: prompt and greeting
- `model.model_info.api_base`, `model.model_info.model_name`,
  `model.model_info.temperature`, `model.model_info.top_p`: the model endpoint
- `plugins`, `workflows`: lists of {"id", "name"} references the agent may use
- `dependencies`: {"plugins", "workflows", "knowledge_bases", "prompt_templates"}
- `create_time`, `update_time`: epoch milliseconds, omitted when unknown

Do not add other keys and do not wrap the document in prose.
When the request is ambiguous, answer in plain text with one clarifying question."#;

/// The instruction sent to the model when a workflow is requested.
///
/// It names every key the parser reads, so a compliant answer round-trips
/// through [`WorkflowDsl`] without repair.
const WORKFLOW_SYSTEM_PROMPT: &str = r#"You are the openJiuwen agent builder.
Turn the user's request into one complete workflow document.

Answer with exactly one ```json fenced code block and nothing else inside it,
using these keys:
- `name`, `description`: a short name and a one-sentence purpose
- `nodes`: a list of {"id", "type", "title", "description", "inputs", "outputs",
  "data"} objects, with unique `id`s and a `type` such as "start", "end", "llm"
  or "code"
- `edges`: a list of {"source", "target", "condition"} objects whose endpoints
  name existing node ids, with `condition` omitted for an unconditional edge

The graph needs at least one node and every edge endpoint must be a node.
Do not add other keys and do not wrap the document in prose.
When the request is ambiguous, answer in plain text with one clarifying question."#;

/// Which artifact the builder is asked to produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentType {
    /// An LLM agent (Python's `"llm_agent"`), the default artifact.
    #[default]
    LlmAgent,
    /// A workflow (Python's `"workflow"`).
    Workflow,
}

impl AgentType {
    /// The wire name of the variant, `"llm_agent"` or `"workflow"`.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::LlmAgent => "llm_agent",
            Self::Workflow => "workflow",
        }
    }
}

impl fmt::Display for AgentType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for AgentType {
    type Err = Error;

    /// Parses the wire name; only `"llm_agent"` and `"workflow"` are accepted.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] naming the accepted values for any other
    /// input.
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "llm_agent" => Ok(Self::LlmAgent),
            "workflow" => Ok(Self::Workflow),
            other => Err(Error::validation(format!(
                "unknown agent type `{other}`; expected `llm_agent` or `workflow`"
            ))),
        }
    }
}

/// Epoch milliseconds read from the system clock, or `0` when it is unreadable.
///
/// This is the crate's only wall-clock read; every other time value is a
/// parameter, so tests stay deterministic.
#[must_use]
pub fn now_millis() -> i64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(elapsed) => i64::try_from(elapsed.as_millis()).unwrap_or(i64::MAX),
        Err(_) => 0,
    }
}

/// The retained state of one builder session.
#[derive(Debug, Clone)]
struct SessionState {
    /// The bounded dialogue history.
    history: HistoryManager,
    /// Where the session is in the build-state machine.
    state: BuildState,
    /// The artifact the session is building.
    agent_type: AgentType,
}

/// A session-based, model-driven builder for LLM agents and workflows.
///
/// Every session keeps its own dialogue history and state; [`Self::build_agent`]
/// answers with the requested artifact, or with a clarifying question when the
/// model's answer could not be parsed. The builder owns a [`Model`] through an
/// [`Arc`] so clones share one client.
#[derive(Clone)]
pub struct AgentBuilder {
    /// The model every session talks to.
    model: Arc<dyn Model>,
    /// The sessions, keyed by caller-supplied id.
    sessions: HashMap<String, SessionState>,
}

impl fmt::Debug for AgentBuilder {
    /// Reports the model identity and the session count.
    ///
    /// [`Model`] is not [`fmt::Debug`], so the provider and model name are
    /// printed instead of the client.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AgentBuilder")
            .field("provider", &self.model.provider())
            .field("model_name", &self.model.model_name())
            .field("sessions", &self.sessions.len())
            .finish()
    }
}

impl AgentBuilder {
    /// Creates a builder talking to `model` with no sessions.
    #[must_use]
    pub fn new(model: Arc<dyn Model>) -> Self {
        Self {
            model,
            sessions: HashMap::new(),
        }
    }

    /// Builds the requested artifact, stamping DSL timestamps with the clock.
    ///
    /// Equivalent to [`Self::build_agent_at`] with [`now_millis`]. Only the DSL
    /// metadata (`create_time`, `update_time`, and a missing `agent_id`) depends
    /// on that clock read; a fresh `UUIDv4` is used solely as the fallback
    /// `agent_id`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for a blank query, the model's own error
    /// when the call fails, and [`Error::Llm`] when the model answers without a
    /// choice.
    pub async fn build_agent(
        &mut self,
        query: &str,
        session_id: &str,
        agent_type: AgentType,
    ) -> Result<BuildResult> {
        self.build_agent_at(query, session_id, agent_type, now_millis())
            .await
    }

    /// Builds the requested artifact, stamping DSL timestamps with
    /// `timestamp_ms`.
    ///
    /// A blank query is rejected before any session is created; otherwise the
    /// session is created on first use, the user turn is appended, and the state
    /// becomes [`BuildState::Processing`]. The model is then called once, with
    /// the artifact-specific system prompt followed by one message per retained
    /// history turn. A fenced JSON answer is parsed into the requested artifact;
    /// any parse or validation failure becomes [`BuildOutput::Response`] instead
    /// of an error. Either way the assistant turn and the final state are
    /// persisted.
    ///
    /// A model failure propagates and leaves the session in
    /// [`BuildState::Processing`] with the user turn recorded, so the caller can
    /// retry the same session.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for a blank query, the model's own error
    /// when the call fails, and [`Error::Llm`] when the model answers without a
    /// choice.
    pub async fn build_agent_at(
        &mut self,
        query: &str,
        session_id: &str,
        agent_type: AgentType,
        timestamp_ms: i64,
    ) -> Result<BuildResult> {
        if query.trim().is_empty() {
            return Err(Error::validation("query must not be blank"));
        }

        let session = self
            .sessions
            .entry(session_id.to_owned())
            .or_insert_with(|| SessionState {
                history: HistoryManager::new(DEFAULT_MAX_HISTORY_SIZE),
                state: BuildState::Initial,
                agent_type,
            });
        session.agent_type = agent_type;
        session.state = BuildState::Processing;
        session.history.add_user_message(query);

        let messages = conversation(&session.history, agent_type);
        let request = ModelRequestConfig::new(self.model.model_name());
        let response = self.model.invoke(&messages, &request).await?;
        let Some(choice) = response.choices.into_iter().next() else {
            return Err(Error::llm("model returned no choices"));
        };
        let text = choice.message.text().into_owned();

        let (output, state) = artifact_from_answer(&text, agent_type, timestamp_ms);
        let status = map_state_to_status(state, agent_type);

        session.history.add_assistant_message(text);
        session.state = state;

        Ok(BuildResult {
            status,
            session_id: session_id.to_owned(),
            agent_type,
            output,
        })
    }

    /// Returns the retained dialogue of a session, oldest first.
    ///
    /// `k` limits the answer to the newest `k` turns ([`HistoryManager`] treats
    /// `k == 0` as "none"); `None` returns the whole history. An unknown session
    /// yields an empty vector.
    #[must_use]
    pub fn history(&self, session_id: &str, k: Option<usize>) -> Vec<DialogueMessage> {
        match self.sessions.get(session_id) {
            Some(session) => match k {
                Some(k) => session.history.get_latest_k(k).to_vec(),
                None => session.history.get_history().to_vec(),
            },
            None => Vec::new(),
        }
    }

    /// Drops a session and its dialogue history; unknown ids are ignored.
    pub fn clear_session(&mut self, session_id: &str) {
        self.sessions.remove(session_id);
    }

    /// Returns a snapshot of a session, or `None` for an unknown id.
    ///
    /// `resource_count` is the number of dialogue turns the session retains.
    #[must_use]
    pub fn status(&self, session_id: &str) -> Option<BuildStatus> {
        self.sessions.get(session_id).map(|session| BuildStatus {
            session_id: session_id.to_owned(),
            agent_type: session.agent_type,
            state: session.state,
            resource_count: session.history.len(),
        })
    }

    /// The model every session talks to.
    #[must_use]
    pub fn model(&self) -> &Arc<dyn Model> {
        &self.model
    }
}

/// Assembles the system prompt plus one message per retained history turn.
fn conversation(history: &HistoryManager, agent_type: AgentType) -> Vec<BaseMessage> {
    let mut messages = Vec::with_capacity(history.len() + 1);
    messages.push(BaseMessage::system(build_system_prompt(agent_type)));
    for message in history.get_history() {
        if message.role == "assistant" {
            messages.push(BaseMessage::assistant(message.content.clone()));
        } else {
            messages.push(BaseMessage::user(message.content.clone()));
        }
    }
    messages
}

/// The instruction sent to the model for `agent_type`.
fn build_system_prompt(agent_type: AgentType) -> &'static str {
    match agent_type {
        AgentType::LlmAgent => LLM_AGENT_SYSTEM_PROMPT,
        AgentType::Workflow => WORKFLOW_SYSTEM_PROMPT,
    }
}

/// The output and build state implied by a model answer.
///
/// A parseable artifact completes the session; anything else is a plain-text
/// answer awaiting clarification.
fn artifact_from_answer(
    text: &str,
    agent_type: AgentType,
    timestamp_ms: i64,
) -> (BuildOutput, BuildState) {
    match parse_artifact(text, agent_type, timestamp_ms) {
        Some(output) => (output, BuildState::Completed),
        None => (
            BuildOutput::Response(text.to_owned()),
            BuildState::Clarifying,
        ),
    }
}

/// Parses a fenced JSON answer into the requested artifact.
///
/// Failures are logged and reported as `None`; the caller falls back to the
/// model's text.
fn parse_artifact(text: &str, agent_type: AgentType, timestamp_ms: i64) -> Option<BuildOutput> {
    let value = extract_json_fence(text)?;
    match agent_type {
        AgentType::LlmAgent => match LlmAgentDsl::from_value(value) {
            Ok(mut dsl) => {
                if dsl.agent_id.is_empty() {
                    dsl.agent_id = uuid::Uuid::new_v4().to_string();
                }
                if dsl.create_time == 0 {
                    dsl.create_time = timestamp_ms;
                }
                if dsl.update_time == 0 {
                    dsl.update_time = timestamp_ms;
                }
                Some(BuildOutput::Dsl(dsl))
            }
            Err(error) => {
                tracing::debug!(%error, "model answer is not an LLM-agent DSL document");
                None
            }
        },
        AgentType::Workflow => match serde_json::from_value::<WorkflowDsl>(value) {
            Ok(workflow) => match workflow.validate() {
                Ok(()) => Some(BuildOutput::Workflow(workflow)),
                Err(error) => {
                    tracing::debug!(%error, "model answer is not a valid workflow");
                    None
                }
            },
            Err(error) => {
                tracing::debug!(%error, "model answer is not a workflow document");
                None
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_test_support::{ScriptedModel, ScriptedTurn};

    /// A model answer holding an LLM-agent document without timestamps.
    const AGENT_FENCE: &str = r#"Here is the agent you asked for.
```json
{"agent_id": "agent-1", "agent_type": "react", "name": "Demo Agent", "description": "a demo agent",
 "configs": {"system_prompt": "You are helpful.", "opening_remarks": "Hi!"},
 "plugins": [{"id": "plugin-1", "name": "search"}], "dependencies": {"plugins": ["plugin-1"]}}
```"#;

    /// A model answer holding a valid two-node workflow document.
    const WORKFLOW_FENCE: &str = r#"```json
{"name": "demo", "description": "a demo workflow",
 "nodes": [{"id": "start", "type": "start", "title": "Start"}, {"id": "end", "type": "end"}],
 "edges": [{"source": "start", "target": "end"}]}
```"#;

    /// Coerces a scripted model into the trait object the builder takes.
    fn dynamic(model: &Arc<ScriptedModel>) -> Arc<dyn Model> {
        let cloned: Arc<ScriptedModel> = Arc::clone(model);
        cloned
    }

    /// The DSL document of a successful build result.
    fn dsl_of(result: &BuildResult) -> &LlmAgentDsl {
        match &result.output {
            BuildOutput::Dsl(dsl) => dsl,
            other => panic!("expected DSL, got {other:?}"),
        }
    }

    /// The workflow document of a successful build result.
    fn workflow_of(result: &BuildResult) -> &WorkflowDsl {
        match &result.output {
            BuildOutput::Workflow(workflow) => workflow,
            other => panic!("expected workflow, got {other:?}"),
        }
    }

    #[test]
    fn agent_type_round_trips_through_display_and_from_str() {
        for agent_type in [AgentType::LlmAgent, AgentType::Workflow] {
            assert_eq!(agent_type.as_str(), agent_type.to_string());
            assert_eq!(
                agent_type
                    .as_str()
                    .parse::<AgentType>()
                    .expect("round trip"),
                agent_type
            );
        }
        assert_eq!(AgentType::LlmAgent.to_string(), "llm_agent");
        assert_eq!(AgentType::Workflow.to_string(), "workflow");
        for value in ["", "LLMAgent", "llmagent", "workflow_dsl"] {
            let error = value.parse::<AgentType>().expect_err("unknown name");
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert!(error.to_string().contains("unknown agent type"));
            assert!(error.to_string().contains("llm_agent"));
            assert!(error.to_string().contains("workflow"));
        }
    }

    #[tokio::test]
    async fn build_agent_returns_the_parsed_dsl() {
        let model = Arc::new(ScriptedModel::text(AGENT_FENCE));
        let mut builder = AgentBuilder::new(dynamic(&model));

        let result = builder
            .build_agent("build me an agent", "s1", AgentType::LlmAgent)
            .await
            .expect("build succeeds");

        assert_eq!(result.status, "completed");
        assert_eq!(result.session_id, "s1");
        assert_eq!(result.agent_type, AgentType::LlmAgent);
        let dsl = dsl_of(&result);
        assert_eq!(dsl.agent_id, "agent-1");
        assert_eq!(dsl.configs.opening_remarks, "Hi!");
        assert!(dsl.create_time > 0, "the clock fills the create time");
        assert_eq!(dsl.create_time, dsl.update_time);

        assert_eq!(model.call_count(), 1);
        let call = model.last_call().expect("one call");
        assert_eq!(call.messages.len(), 2);
        assert_eq!(call.messages[0].text(), LLM_AGENT_SYSTEM_PROMPT);
        assert_eq!(call.messages[1].text(), "build me an agent");
        assert_eq!(call.request.model_name, "scripted-model");

        let status = builder.status("s1").expect("session is known");
        assert_eq!(status.session_id, "s1");
        assert_eq!(status.agent_type, AgentType::LlmAgent);
        assert_eq!(status.state, BuildState::Completed);
        assert_eq!(status.resource_count, 2);
        let history = builder.history("s1", None);
        assert_eq!(history.len(), 2);
        assert_eq!(
            history[0],
            DialogueMessage {
                role: "user".to_string(),
                content: "build me an agent".to_string(),
            }
        );
        assert_eq!(
            history[1],
            DialogueMessage {
                role: "assistant".to_string(),
                content: AGENT_FENCE.to_string(),
            }
        );
    }

    #[tokio::test]
    async fn build_agent_at_stamps_missing_metadata_only() {
        let model = Arc::new(ScriptedModel::text(AGENT_FENCE));
        let mut builder = AgentBuilder::new(dynamic(&model));
        let result = builder
            .build_agent_at("go", "s1", AgentType::LlmAgent, 1_700_000_000_000)
            .await
            .expect("build succeeds");
        let dsl = dsl_of(&result);
        assert_eq!(dsl.agent_id, "agent-1");
        assert_eq!(dsl.create_time, 1_700_000_000_000);
        assert_eq!(dsl.update_time, 1_700_000_000_000);

        let nameless = Arc::new(ScriptedModel::text(
            "```json\n{\"name\": \"nameless\"}\n```",
        ));
        let mut builder = AgentBuilder::new(dynamic(&nameless));
        let result = builder
            .build_agent_at("go", "s2", AgentType::LlmAgent, 42)
            .await
            .expect("build succeeds");
        let dsl = dsl_of(&result);
        assert_eq!(dsl.agent_type, "react");
        assert!(!dsl.agent_id.is_empty(), "a fallback id is generated");
        assert_eq!(dsl.create_time, 42);
        assert_eq!(dsl.update_time, 42);
    }

    #[tokio::test]
    async fn sessions_are_isolated_and_clearable() {
        let model = Arc::new(ScriptedModel::new([
            ScriptedTurn::text(AGENT_FENCE),
            ScriptedTurn::text("which workflow do you want?"),
        ]));
        let mut builder = AgentBuilder::new(dynamic(&model));

        builder
            .build_agent("build me an agent", "s1", AgentType::LlmAgent)
            .await
            .expect("first build succeeds");
        assert_eq!(builder.history("s2", None).len(), 0);
        assert!(builder.status("s2").is_none());
        assert_eq!(builder.history("s1", None).len(), 2);

        let result = builder
            .build_agent("build me a workflow", "s2", AgentType::Workflow)
            .await
            .expect("second build succeeds");
        assert_eq!(result.status, "clarifying");
        assert_eq!(result.session_id, "s2");
        assert_eq!(builder.history("s2", None).len(), 2);
        assert_eq!(builder.history("s1", None).len(), 2);
        assert_eq!(model.calls()[1].messages[0].text(), WORKFLOW_SYSTEM_PROMPT);

        builder.clear_session("s1");
        assert!(builder.status("s1").is_none());
        assert_eq!(builder.history("s1", None).len(), 0);
        builder.clear_session("never-seen");
        assert_eq!(builder.history("s2", None).len(), 2);
    }

    #[tokio::test]
    async fn history_k_limits_the_returned_turns() {
        let model = Arc::new(ScriptedModel::new([
            ScriptedTurn::text(AGENT_FENCE),
            ScriptedTurn::text("and now?"),
        ]));
        let mut builder = AgentBuilder::new(dynamic(&model));

        for query in ["build me an agent", "add a plugin"] {
            builder
                .build_agent(query, "s1", AgentType::LlmAgent)
                .await
                .expect("build succeeds");
        }
        assert_eq!(builder.history("s1", None).len(), 4);
        assert_eq!(builder.history("s1", Some(1))[0].content, "and now?");
        assert_eq!(builder.history("s1", Some(0)).len(), 0);
        assert_eq!(builder.history("s1", Some(99)).len(), 4);
        assert_eq!(model.call_count(), 2);
    }

    #[tokio::test]
    async fn plain_text_answer_is_a_clarifying_response() {
        let model = Arc::new(ScriptedModel::text("Which system prompt should I use?"));
        let mut builder = AgentBuilder::new(dynamic(&model));

        let result = builder
            .build_agent("make an agent", "s1", AgentType::LlmAgent)
            .await
            .expect("build succeeds");

        assert_eq!(result.status, "clarifying");
        assert_eq!(
            result.output,
            BuildOutput::Response("Which system prompt should I use?".to_string())
        );
        let status = builder.status("s1").expect("session is known");
        assert_eq!(status.state, BuildState::Clarifying);
        assert_eq!(status.resource_count, 2);
    }

    #[tokio::test]
    async fn blank_query_is_rejected_before_a_session_exists() {
        let model = Arc::new(ScriptedModel::text(AGENT_FENCE));
        let mut builder = AgentBuilder::new(dynamic(&model));

        for query in ["", "   ", "\n\t "] {
            let error = builder
                .build_agent(query, "s1", AgentType::LlmAgent)
                .await
                .expect_err("blank query is invalid");
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert!(error.to_string().contains("must not be blank"));
        }
        assert!(builder.status("s1").is_none());
        assert_eq!(builder.history("s1", None).len(), 0);
        assert_eq!(model.call_count(), 0);
    }

    #[tokio::test]
    async fn valid_workflow_fence_completes_the_session() {
        let model = Arc::new(ScriptedModel::text(WORKFLOW_FENCE));
        let mut builder = AgentBuilder::new(dynamic(&model));

        let result = builder
            .build_agent("build a workflow", "w1", AgentType::Workflow)
            .await
            .expect("build succeeds");

        assert_eq!(result.status, "completed");
        assert_eq!(result.agent_type, AgentType::Workflow);
        let workflow = workflow_of(&result);
        assert_eq!(workflow.nodes.len(), 2);
        assert_eq!(workflow.edges[0].target, "end");
        assert_eq!(
            builder.status("w1").expect("session is known").state,
            BuildState::Completed
        );
    }

    #[tokio::test]
    async fn invalid_workflow_falls_back_to_a_response() {
        for answer in [
            r#"```json
{"name": "broken", "nodes": [{"id": "start", "type": "start"}], "edges": [{"source": "start", "target": "ghost"}]}
```"#,
            "```json\n{\"name\": \"empty\"}\n```",
        ] {
            let model = Arc::new(ScriptedModel::text(answer));
            let mut builder = AgentBuilder::new(dynamic(&model));

            let result = builder
                .build_agent("build a workflow", "w1", AgentType::Workflow)
                .await
                .expect("build succeeds");

            assert_eq!(result.status, "clarifying");
            assert!(matches!(result.output, BuildOutput::Response(_)));
            assert_eq!(
                builder.status("w1").expect("session is known").state,
                BuildState::Clarifying
            );
        }
    }

    #[tokio::test]
    async fn model_failure_propagates_and_keeps_the_session_processing() {
        let model = Arc::new(ScriptedModel::new([ScriptedTurn::fail("boom")]));
        let mut builder = AgentBuilder::new(dynamic(&model));

        let error = builder
            .build_agent("build me an agent", "s1", AgentType::LlmAgent)
            .await
            .expect_err("the model failed");
        assert_eq!(error.kind(), ErrorKind::Llm);
        assert!(error.to_string().contains("boom"));

        let status = builder.status("s1").expect("the session was created");
        assert_eq!(status.state, BuildState::Processing);
        assert_eq!(status.resource_count, 1);
        let history = builder.history("s1", None);
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].role, "user");
        assert_eq!(history[0].content, "build me an agent");
    }

    #[tokio::test]
    async fn build_agent_uses_the_clock_for_missing_timestamps() {
        let model = Arc::new(ScriptedModel::text(
            "```json\n{\"agent_id\": \"agent-1\", \"name\": \"Demo\"}\n```",
        ));
        let mut builder = AgentBuilder::new(dynamic(&model));

        let result = builder
            .build_agent("go", "s1", AgentType::LlmAgent)
            .await
            .expect("build succeeds");

        let dsl = dsl_of(&result);
        assert!(dsl.create_time > 0);
        assert_eq!(dsl.create_time, dsl.update_time);
    }

    #[test]
    fn model_clock_and_debug_surface() {
        let model = Arc::new(ScriptedModel::text("hi"));
        let builder = AgentBuilder::new(dynamic(&model));
        assert_eq!(builder.model().provider(), "scripted");
        assert_eq!(builder.model().model_name(), "scripted-model");
        assert!(now_millis() > 0);

        let rendered = format!("{builder:?}");
        assert!(rendered.contains("provider: \"scripted\""), "{rendered}");
        assert!(
            rendered.contains("model_name: \"scripted-model\""),
            "{rendered}"
        );
        assert!(rendered.contains("sessions: 0"), "{rendered}");
    }
}
