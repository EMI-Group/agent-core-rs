//! The builder's session state and result vocabulary.
//!
//! These types are the observable response shape of
//! [`crate::agent_builder::builder::AgentBuilder`]: [`BuildState`] tracks where
//! a session is, [`map_state_to_status`] renders the Python status string,
//! [`BuildOutput`] carries one produced artifact, and [`BuildResult`] flattens
//! the two into the `status`/`session_id`/`agent_type`/output response object.

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::builder::AgentType;
use super::dsl::LlmAgentDsl;
use super::workflow_dsl::WorkflowDsl;

/// Where a session is in Python's build-state machine.
///
/// The port adds [`BuildState::Clarifying`] and [`BuildState::Requesting`],
/// which Python only models as status strings next to its state enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BuildState {
    /// No request has been processed yet.
    #[default]
    Initial,
    /// The builder is waiting for the user to answer a clarification question.
    Clarifying,
    /// The builder is waiting for the user to describe the workflow steps.
    Requesting,
    /// A request is being processed.
    Processing,
    /// The requested artifact was produced.
    Completed,
}

/// The status string reported for `state` and `agent_type`.
///
/// The six reachable strings are `"clarifying"`, `"requesting"`,
/// `"processing"`, and `"completed"`; [`BuildState::Initial`] maps to
/// `"clarifying"` for [`AgentType::LlmAgent`] and to `"requesting"` for
/// [`AgentType::Workflow`], because a workflow session starts by asking for the
/// processing steps.
#[must_use]
pub fn map_state_to_status(state: BuildState, agent_type: AgentType) -> String {
    let status = match state {
        BuildState::Initial => match agent_type {
            AgentType::LlmAgent => "clarifying",
            AgentType::Workflow => "requesting",
        },
        BuildState::Clarifying => "clarifying",
        BuildState::Requesting => "requesting",
        BuildState::Processing => "processing",
        BuildState::Completed => "completed",
    };
    status.to_string()
}

/// A snapshot of one builder session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct BuildStatus {
    /// The session the snapshot belongs to.
    pub session_id: String,
    /// The artifact the session is building.
    pub agent_type: AgentType,
    /// The state the session is in.
    pub state: BuildState,
    /// How many dialogue turns the session retains.
    pub resource_count: usize,
}

impl Default for BuildStatus {
    /// An empty snapshot: no session id, an LLM agent, [`BuildState::Initial`].
    fn default() -> Self {
        Self {
            session_id: String::new(),
            agent_type: AgentType::LlmAgent,
            state: BuildState::Initial,
            resource_count: 0,
        }
    }
}

/// What one build produced.
///
/// The variants serialize externally tagged, matching the Python response
/// shape: `{"dsl": {...}}`, `{"workflow": {...}}`, or `{"response": "..."}`.
// The DSL document is carried inline because it is the payload callers hand to
// the platform; the size difference between the variants is deliberate.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BuildOutput {
    /// A parsed LLM-agent document.
    Dsl(LlmAgentDsl),
    /// A parsed and structurally valid workflow document.
    Workflow(WorkflowDsl),
    /// A plain-text model answer, typically a clarifying question.
    Response(String),
}

impl Default for BuildOutput {
    /// An empty plain-text answer, so [`BuildResult`] can default.
    fn default() -> Self {
        Self::Response(String::new())
    }
}

impl BuildOutput {
    /// Converts the output into a JSON value.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the output cannot be serialized, with
    /// the serde error in the message.
    pub fn as_json(&self) -> Result<Value> {
        serde_json::to_value(self).map_err(|error| {
            Error::validation(format!("failed to serialize build output: {error}"))
        })
    }
}

/// The result of one [`AgentBuilder`](crate::agent_builder::builder::AgentBuilder)
/// call.
///
/// [`Self::to_json`] flattens the output into the Python response shape, so the
/// JSON has `status`, `session_id`, `agent_type`, and exactly one of `dsl`,
/// `workflow`, or `response`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct BuildResult {
    /// The status string produced by [`map_state_to_status`].
    pub status: String,
    /// The session the call belongs to.
    pub session_id: String,
    /// The artifact the session is building.
    pub agent_type: AgentType,
    /// What the call produced.
    pub output: BuildOutput,
}

impl BuildResult {
    /// Flattens the result into the Python response object.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the output cannot be serialized or is
    /// not a JSON object.
    pub fn to_json(&self) -> Result<Value> {
        let mut map = Map::new();
        map.insert("status".to_string(), Value::String(self.status.clone()));
        map.insert(
            "session_id".to_string(),
            Value::String(self.session_id.clone()),
        );
        map.insert(
            "agent_type".to_string(),
            Value::String(self.agent_type.as_str().to_string()),
        );
        match self.output.as_json()? {
            Value::Object(entries) => {
                for (key, value) in entries {
                    map.insert(key, value);
                }
            }
            _ => {
                return Err(Error::validation(
                    "build output did not serialize to a JSON object",
                ));
            }
        }
        Ok(Value::Object(map))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The workflow document used by the output and result round-trip tests.
    fn parsed_workflow() -> WorkflowDsl {
        WorkflowDsl::from_json_str(
            r#"{"name": "demo", "description": "a demo workflow",
 "nodes": [{"id": "start", "type": "start", "title": "Start"}, {"id": "end", "type": "end"}],
 "edges": [{"source": "start", "target": "end"}]}"#,
        )
        .expect("valid workflow document")
    }

    #[test]
    fn status_mapping_covers_every_state_and_agent_type() {
        let cases = [
            (BuildState::Initial, AgentType::LlmAgent, "clarifying"),
            (BuildState::Initial, AgentType::Workflow, "requesting"),
            (BuildState::Clarifying, AgentType::LlmAgent, "clarifying"),
            (BuildState::Clarifying, AgentType::Workflow, "clarifying"),
            (BuildState::Requesting, AgentType::LlmAgent, "requesting"),
            (BuildState::Requesting, AgentType::Workflow, "requesting"),
            (BuildState::Processing, AgentType::LlmAgent, "processing"),
            (BuildState::Processing, AgentType::Workflow, "processing"),
            (BuildState::Completed, AgentType::LlmAgent, "completed"),
            (BuildState::Completed, AgentType::Workflow, "completed"),
        ];
        for (state, agent_type, expected) in cases {
            assert_eq!(map_state_to_status(state, agent_type), expected);
        }
        let distinct: std::collections::BTreeSet<&str> = cases.iter().map(|case| case.2).collect();
        assert_eq!(
            distinct.into_iter().collect::<Vec<_>>(),
            vec!["clarifying", "completed", "processing", "requesting"]
        );
    }

    #[test]
    fn enums_serialize_to_their_wire_names() {
        assert_eq!(
            serde_json::to_value(AgentType::LlmAgent).expect("serialize"),
            json!("llm_agent")
        );
        assert_eq!(
            serde_json::to_value(AgentType::Workflow).expect("serialize"),
            json!("workflow")
        );
        assert!(serde_json::from_value::<AgentType>(json!("other")).is_err());
        assert_eq!(AgentType::default(), AgentType::LlmAgent);
        assert_eq!(BuildState::default(), BuildState::Initial);
        for (state, expected) in [
            (BuildState::Initial, "initial"),
            (BuildState::Clarifying, "clarifying"),
            (BuildState::Requesting, "requesting"),
            (BuildState::Processing, "processing"),
            (BuildState::Completed, "completed"),
        ] {
            assert_eq!(
                serde_json::to_value(state).expect("serialize"),
                json!(expected)
            );
            assert_eq!(
                serde_json::from_value::<BuildState>(json!(expected)).expect("read"),
                state
            );
        }

        let status = BuildStatus::default();
        assert_eq!(status.session_id, "");
        assert_eq!(status.agent_type, AgentType::LlmAgent);
        assert_eq!(status.state, BuildState::Initial);
        assert_eq!(status.resource_count, 0);
        assert_eq!(
            serde_json::from_value::<BuildStatus>(json!({"session_id": "s1"})).expect("read"),
            BuildStatus {
                session_id: "s1".to_string(),
                ..BuildStatus::default()
            }
        );
    }

    #[test]
    fn build_output_serializes_each_variant_and_round_trips() {
        let dsl = LlmAgentDsl::from_json_str(r#"{"agent_id": "agent-1"}"#).expect("parse");
        let dsl_json = BuildOutput::Dsl(dsl.clone()).as_json().expect("serialize");
        assert_eq!(dsl_json, json!({"dsl": dsl.to_value().expect("value")}));

        let workflow_json = BuildOutput::Workflow(parsed_workflow())
            .as_json()
            .expect("serialize");
        assert_eq!(workflow_json["workflow"]["name"], json!("demo"));
        assert_eq!(workflow_json.as_object().expect("object").len(), 1);

        let response = BuildOutput::Response("which agent?".to_string());
        assert_eq!(
            response.as_json().expect("serialize"),
            json!({"response": "which agent?"})
        );
        assert_eq!(BuildOutput::default(), BuildOutput::Response(String::new()));

        let outputs = [
            BuildOutput::Dsl(dsl),
            BuildOutput::Workflow(parsed_workflow()),
            response,
        ];
        for output in outputs {
            let value = output.as_json().expect("serialize");
            let restored: BuildOutput = serde_json::from_value(value.clone()).expect("read");
            assert_eq!(restored, output);
        }
    }

    #[test]
    fn build_result_flattens_the_output() {
        let result = BuildResult {
            status: "completed".to_string(),
            session_id: "s1".to_string(),
            agent_type: AgentType::LlmAgent,
            output: BuildOutput::Response("hi".to_string()),
        };
        let value = result.to_json().expect("serialize");
        let keys: Vec<&String> = value.as_object().expect("object").keys().collect();
        assert_eq!(keys, vec!["status", "session_id", "agent_type", "response"]);
        assert_eq!(value["agent_type"], json!("llm_agent"));
        assert_eq!(value["response"], json!("hi"));
        assert!(value.get("output").is_none());

        let workflow = BuildResult {
            output: BuildOutput::Workflow(parsed_workflow()),
            agent_type: AgentType::Workflow,
            ..BuildResult::default()
        };
        let value = workflow.to_json().expect("serialize");
        assert_eq!(value["workflow"]["name"], json!("demo"));
        assert!(value.get("response").is_none());
    }
}
