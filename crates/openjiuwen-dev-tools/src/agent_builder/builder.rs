//! The session-based, model-driven agent builder.
//!
//! Ports the conversational entry point of
//! `openjiuwen/dev_tools/agent_builder`: a caller submits a natural-language
//! request on a session id and receives the artifact it could parse, or a
//! clarifying question.
//!
//! Two entry points share the session machinery:
//!
//! - [`AgentBuilder::build_agent`] / [`AgentBuilder::build_agent_at`] keep the
//!   single-round-trip behavior: one model answer that is either a fenced JSON
//!   artifact or a plain-text clarification.
//! - [`AgentBuilder::build_with_pipeline`] runs the staged pipeline —
//!   clarification, then the workflow designer/generator/reflector chain or the
//!   LLM-agent generation plus deterministic repair — and falls back to the
//!   deterministic stage implementations whenever a model answer is unusable.
//!
//! Time comes from [`now_millis`], and the only random values are the DSL agent
//! ids.

use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_foundation::BaseMessage;
use serde::{Deserialize, Serialize};

use super::clarifier::{Clarification, Clarifier};
use super::designer::WorkflowDesigner;
use super::dl_generator::DlGenerator;
use super::dl_reflector::DlReflector;
use super::dsl::LlmAgentDsl;
use super::history::{DialogueMessage, HistoryManager, DEFAULT_MAX_HISTORY_SIZE};
use super::outcome::{map_state_to_status, BuildOutput, BuildResult, BuildState, BuildStatus};
use super::spec::LlmAgentSpec;
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

/// The clarification question used when the clarifier names no detail at all.
const DEFAULT_CLARIFICATION_QUESTION: &str =
    "Please describe what the agent should accomplish and what output it should produce.";

/// How many whitespace-separated words of a request become the agent name.
const MAX_NAME_WORDS: usize = 6;

/// How many characters of a CJK-only request become the agent name.
const MAX_NAME_CHARS: usize = 24;

/// How many characters of a request become the agent description.
const MAX_DESCRIPTION_CHARS: usize = 160;

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

    /// Builds the requested artifact through the staged pipeline.
    ///
    /// Stage 1 always runs [`Clarifier`]; when the request is incomplete the
    /// session stops with a clarifying response. A ready request then runs the
    /// workflow stages ([`WorkflowDesigner`] -> [`DlGenerator`] -> [`DlReflector`])
    /// or the LLM-agent stage, falling back deterministically whenever a model
    /// answer is unusable. Only a blank query is an error; use this variant when
    /// a full artifact is wanted, and [`Self::build_agent_at`] for the legacy
    /// single round trip.
    ///
    /// Sessions behave exactly like [`Self::build_agent_at`]: a blank query is
    /// rejected before any session exists, the session is created on first use,
    /// its `agent_type` is set, the user turn is appended, and the state becomes
    /// [`BuildState::Processing`]. The clarifier receives the turns retained
    /// *before* this request, so the query is not duplicated.
    ///
    /// A not-ready clarification becomes [`BuildOutput::Response`] holding the
    /// question, or else the summary, or else
    /// `DEFAULT_CLARIFICATION_QUESTION`, with state
    /// [`BuildState::Clarifying`]. A ready workflow request designs, generates,
    /// and reflects a document; an unusable reflection falls back to the
    /// deterministic generation of the same design. A ready LLM-agent request
    /// makes one model call, repairs the parsed document deterministically, and
    /// falls back to `deterministic_agent_dsl` when the answer is unparseable
    /// or the call fails. Both successes complete the session.
    ///
    /// The assistant text (the JSON document, or the clarification) is appended
    /// to the session history and the final state is stored, so
    /// [`Self::status`] reports two more retained turns per call.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for a blank query, which is the only error
    /// a caller can trigger; a failing model answer never propagates. Stage
    /// output that cannot be represented (serialization) or a structurally
    /// impossible design would surface as [`Error::Validation`] as well.
    pub async fn build_with_pipeline(
        &mut self,
        query: &str,
        session_id: &str,
        agent_type: AgentType,
        timestamp_ms: i64,
    ) -> Result<BuildResult> {
        if query.trim().is_empty() {
            return Err(Error::validation("query must not be blank"));
        }

        // The borrow of the session ends here so no async stage holds it.
        let prior = {
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
            let prior = session.history.clone();
            session.history.add_user_message(query);
            prior
        };

        let model = Arc::clone(&self.model);
        let clarification = Clarifier::with_model(Arc::clone(&model))
            .clarify(query, &prior)
            .await?;

        let (output, state) = if clarification.ready {
            match agent_type {
                AgentType::Workflow => {
                    let workflow = workflow_pipeline(&model, query).await?;
                    (BuildOutput::Workflow(workflow), BuildState::Completed)
                }
                AgentType::LlmAgent => {
                    let mut history = prior.clone();
                    history.add_user_message(query);
                    let dsl = llm_agent_pipeline(&model, query, &history, timestamp_ms).await?;
                    (BuildOutput::Dsl(dsl), BuildState::Completed)
                }
            }
        } else {
            let text = clarification_text(clarification);
            (BuildOutput::Response(text), BuildState::Clarifying)
        };

        let assistant_text = match &output {
            BuildOutput::Dsl(dsl) => dsl.to_json_string()?,
            BuildOutput::Workflow(workflow) => workflow.to_json_string()?,
            BuildOutput::Response(text) => text.clone(),
        };

        if let Some(session) = self.sessions.get_mut(session_id) {
            session.history.add_assistant_message(assistant_text);
            session.state = state;
        }

        Ok(BuildResult {
            status: map_state_to_status(state, agent_type),
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

/// The text a not-ready clarification is answered with.
///
/// The clarifier's question wins, then its detected-signal summary, then the
/// generic [`DEFAULT_CLARIFICATION_QUESTION`].
fn clarification_text(clarification: Clarification) -> String {
    if !clarification.question.is_empty() {
        clarification.question
    } else if !clarification.summary.is_empty() {
        clarification.summary
    } else {
        DEFAULT_CLARIFICATION_QUESTION.to_string()
    }
}

/// Designs, generates, and reflects one workflow.
///
/// The reflector repairs the generated document with its deterministic rules; a
/// failing reflection falls back to the generator's deterministic document for
/// the same design rather than failing the request.
async fn workflow_pipeline(model: &Arc<dyn Model>, query: &str) -> Result<WorkflowDsl> {
    let design = WorkflowDesigner::with_model(Arc::clone(model))
        .design(query)
        .await?;
    let generator = DlGenerator::with_model(Arc::clone(model));
    let candidate = generator.generate_with_model(&design).await?;

    match DlReflector::with_model(Arc::clone(model)).reflect(&candidate) {
        Ok(reflection) => Ok(reflection.repaired),
        Err(error) => {
            tracing::debug!(
                %error,
                "workflow reflection failed; using the deterministic document"
            );
            generator.generate(&design)
        }
    }
}

/// Asks the model for one LLM-agent document and repairs it deterministically.
///
/// A failed call, a choice-less answer, or an unparseable answer is logged and
/// replaced by [`deterministic_agent_dsl`], so this entry point never reports a
/// model problem as an error.
async fn llm_agent_pipeline(
    model: &Arc<dyn Model>,
    query: &str,
    history: &HistoryManager,
    timestamp_ms: i64,
) -> Result<LlmAgentDsl> {
    let messages = conversation(history, AgentType::LlmAgent);
    let request = ModelRequestConfig::new(model.model_name());
    let response = match model.invoke(&messages, &request).await {
        Ok(response) => response,
        Err(error) => {
            tracing::debug!(
                %error,
                "LLM-agent model call failed; using the deterministic document"
            );
            return Ok(deterministic_agent_dsl(query, timestamp_ms));
        }
    };
    let Some(choice) = response.choices.into_iter().next() else {
        tracing::debug!("LLM-agent model returned no choices; using the deterministic document");
        return Ok(deterministic_agent_dsl(query, timestamp_ms));
    };

    let text = choice.message.text().into_owned();
    let parsed = extract_json_fence(&text)
        .ok_or_else(|| Error::validation("model answer contained no JSON code fence"))
        .and_then(LlmAgentDsl::from_value);
    match parsed {
        Ok(dsl) => Ok(repair_llm_dsl(dsl, query, timestamp_ms)),
        Err(error) => {
            tracing::debug!(
                %error,
                "LLM-agent answer is not a DSL document; using the deterministic document"
            );
            Ok(deterministic_agent_dsl(query, timestamp_ms))
        }
    }
}

/// Fills the blanks of a parsed LLM-agent document.
///
/// Identity, text fields, and timestamps are each repaired only when blank or
/// zero, so a complete model answer is returned untouched apart from a missing
/// `agent_id`.
fn repair_llm_dsl(mut dsl: LlmAgentDsl, query: &str, timestamp_ms: i64) -> LlmAgentDsl {
    if dsl.agent_id.trim().is_empty() {
        dsl.agent_id = uuid::Uuid::new_v4().to_string();
    }
    if dsl.agent_type.trim().is_empty() {
        dsl.agent_type = "react".to_string();
    }
    if dsl.name.trim().is_empty() {
        dsl.name = derived_agent_name(query);
    }
    if dsl.description.trim().is_empty() {
        dsl.description = describe_query(query).unwrap_or_else(|| dsl.name.clone());
    }
    if dsl.configs.system_prompt.trim().is_empty() {
        dsl.configs.system_prompt = default_system_prompt(query);
    }
    if dsl.create_time == 0 {
        dsl.create_time = timestamp_ms;
    }
    if dsl.update_time == 0 {
        dsl.update_time = timestamp_ms;
    }
    dsl
}

/// Builds a complete LLM-agent document from the request alone.
///
/// The name, description, and system prompt are derived from `query`; the agent
/// id is a fresh `UUIDv4` and the clock is the supplied `timestamp_ms`, so two
/// calls with the same query and timestamp differ only in the id.
fn deterministic_agent_dsl(query: &str, timestamp_ms: i64) -> LlmAgentDsl {
    let name = derived_agent_name(query);
    let description = describe_query(query).unwrap_or_else(|| name.clone());
    LlmAgentSpec::new(name, description, default_system_prompt(query))
        .to_dsl(uuid::Uuid::new_v4().to_string(), timestamp_ms)
}

/// The fallback system prompt for a request that supplied none.
fn default_system_prompt(query: &str) -> String {
    format!("You are an assistant built for the following request: {query}")
}

/// A display name derived from the request: its first few words, or its first
/// few characters when it is written without spaces in a CJK script.
fn derived_agent_name(query: &str) -> String {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return "Agent".to_string();
    }

    let candidate = if is_cjk_only(trimmed) {
        trimmed.chars().take(MAX_NAME_CHARS).collect::<String>()
    } else {
        trimmed
            .split_whitespace()
            .take(MAX_NAME_WORDS)
            .collect::<Vec<_>>()
            .join(" ")
    };
    let candidate = candidate.trim();
    if candidate.is_empty() {
        "Agent".to_string()
    } else {
        candidate.to_string()
    }
}

/// The trimmed request, truncated to [`MAX_DESCRIPTION_CHARS`].
///
/// `None` when nothing usable remains, so callers can fall back to the name.
fn describe_query(query: &str) -> Option<String> {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return None;
    }
    let description: String = trimmed.chars().take(MAX_DESCRIPTION_CHARS).collect();
    let description = description.trim();
    (!description.is_empty()).then(|| description.to_string())
}

/// Whether `text` is written in a CJK script without ASCII letters or digits.
fn is_cjk_only(text: &str) -> bool {
    let mut has_cjk = false;
    for character in text.chars() {
        if is_cjk(character) {
            has_cjk = true;
        } else if character.is_ascii_alphanumeric() {
            return false;
        }
    }
    has_cjk
}

/// Whether `character` belongs to a CJK block (Han, Kana, or Hangul).
fn is_cjk(character: char) -> bool {
    matches!(
        character,
        '\u{3040}'..='\u{30FF}'
            | '\u{3400}'..='\u{4DBF}'
            | '\u{4E00}'..='\u{9FFF}'
            | '\u{AC00}'..='\u{D7AF}'
            | '\u{F900}'..='\u{FAFF}'
    )
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod pipeline_tests;
