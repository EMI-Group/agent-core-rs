//! Typed payloads for each lifecycle event.
//!
//! Every payload is plain, serde-serializable data with the Python field names
//! (`agent_id`, `session_id`, `iteration`, ...), tolerates unknown fields, and
//! has a `Default` value so a partial payload can be built field by field.
//! [`CallbackPayload`] is the union handed to [`super::framework::AgentCallback`]
//! hooks: it selects the event, exposes the shared identity fields, and
//! serializes to the Python-shaped JSON object of its inner payload.
use super::event::AgentCallbackEvent;
use crate::output::AgentOutput;
use openjiuwen_foundation::{BaseMessage, ToolCall, ToolCard, Usage};
use serde::{Deserialize, Serialize};
use serde_json::Value;
/// Payload of [`AgentCallbackEvent::InvokeStart`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct InvokeStartPayload {
    /// Id of the agent starting the invocation.
    pub agent_id: String,
    /// Session of the invocation, when one exists.
    pub session_id: Option<String>,
    /// The caller's query text.
    pub query: String,
    /// Maximum number of `ReAct` steps the run may use.
    pub max_iterations: usize,
}
/// Payload of [`AgentCallbackEvent::StepStart`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct StepStartPayload {
    /// Id of the agent running the step.
    pub agent_id: String,
    /// Session of the invocation, when one exists.
    pub session_id: Option<String>,
    /// The `1`-based loop iteration starting.
    pub iteration: usize,
    /// Messages of the model request.
    pub messages: Vec<BaseMessage>,
    /// Tools exposed to the model in this step.
    pub tools: Vec<ToolCard>,
}
/// Payload of [`AgentCallbackEvent::StepEnd`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct StepEndPayload {
    /// Id of the agent running the step.
    pub agent_id: String,
    /// Session of the invocation, when one exists.
    pub session_id: Option<String>,
    /// The `1`-based loop iteration that ended.
    pub iteration: usize,
    /// Answer text of the model reply.
    pub output_text: String,
    /// Number of tool calls the model requested in this step.
    pub tool_call_count: usize,
    /// Token usage reported for this step's model call.
    pub usage: Usage,
    /// Provider finish reason, when reported.
    pub finish_reason: Option<String>,
}
/// Payload of [`AgentCallbackEvent::ToolCallStart`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ToolCallPayload {
    /// Id of the agent executing the call.
    pub agent_id: String,
    /// Session of the invocation, when one exists.
    pub session_id: Option<String>,
    /// The `1`-based loop iteration the call belongs to.
    pub iteration: usize,
    /// The call about to be executed.
    pub call: ToolCall,
}
/// Payload of [`AgentCallbackEvent::ToolCallEnd`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ToolCallEndPayload {
    /// Id of the agent executing the call.
    pub agent_id: String,
    /// Session of the invocation, when one exists.
    pub session_id: Option<String>,
    /// The `1`-based loop iteration the call belongs to.
    pub iteration: usize,
    /// The call that was executed.
    pub call: ToolCall,
    /// Rendered tool result text.
    pub text: String,
    /// Whether the call succeeded.
    pub success: bool,
    /// Failure description when the call did not succeed.
    pub error: Option<String>,
}
/// Payload of [`AgentCallbackEvent::InvokeEnd`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct InvokeEndPayload {
    /// Id of the agent that finished the invocation.
    pub agent_id: String,
    /// Session of the invocation, when one exists.
    pub session_id: Option<String>,
    /// The terminal output of the invocation.
    pub output: AgentOutput,
}
/// The payload of one callback firing.
///
/// The variant determines [`CallbackPayload::event`]; [`CallbackPayload::to_value`]
/// serializes the inner payload with the Python field names.
// The payloads are carried inline because they are what a callback consumes and
// what the host builds at the firing point; the terminal-output variant is the
// largest only because `AgentOutput` is a full result record.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CallbackPayload {
    /// An invocation starts.
    InvokeStart(InvokeStartPayload),
    /// One `ReAct` step starts.
    StepStart(StepStartPayload),
    /// One `ReAct` step ends.
    StepEnd(StepEndPayload),
    /// One tool call starts.
    ToolCallStart(ToolCallPayload),
    /// One tool call ends.
    ToolCallEnd(ToolCallEndPayload),
    /// The invocation ends.
    InvokeEnd(InvokeEndPayload),
}
/// Serializes one payload, falling back to `null` when serialization fails.
///
/// The payloads are plain data whose serialization cannot fail; the fallback
/// keeps [`CallbackPayload::to_value`] panic-free without hiding a reachable
/// error path.
fn to_json_value<T: Serialize>(payload: &T) -> Value {
    serde_json::to_value(payload).unwrap_or(Value::Null)
}
impl CallbackPayload {
    /// The event this payload belongs to.
    #[must_use]
    pub fn event(&self) -> AgentCallbackEvent {
        match self {
            Self::InvokeStart(_) => AgentCallbackEvent::InvokeStart,
            Self::StepStart(_) => AgentCallbackEvent::StepStart,
            Self::StepEnd(_) => AgentCallbackEvent::StepEnd,
            Self::ToolCallStart(_) => AgentCallbackEvent::ToolCallStart,
            Self::ToolCallEnd(_) => AgentCallbackEvent::ToolCallEnd,
            Self::InvokeEnd(_) => AgentCallbackEvent::InvokeEnd,
        }
    }
    /// Id of the agent the payload belongs to.
    #[must_use]
    pub fn agent_id(&self) -> &str {
        match self {
            Self::InvokeStart(payload) => &payload.agent_id,
            Self::StepStart(payload) => &payload.agent_id,
            Self::StepEnd(payload) => &payload.agent_id,
            Self::ToolCallStart(payload) => &payload.agent_id,
            Self::ToolCallEnd(payload) => &payload.agent_id,
            Self::InvokeEnd(payload) => &payload.agent_id,
        }
    }
    /// Session of the invocation, when the payload carries one.
    #[must_use]
    pub fn session_id(&self) -> Option<&str> {
        match self {
            Self::InvokeStart(payload) => payload.session_id.as_deref(),
            Self::StepStart(payload) => payload.session_id.as_deref(),
            Self::StepEnd(payload) => payload.session_id.as_deref(),
            Self::ToolCallStart(payload) => payload.session_id.as_deref(),
            Self::ToolCallEnd(payload) => payload.session_id.as_deref(),
            Self::InvokeEnd(payload) => payload.session_id.as_deref(),
        }
    }
    /// The Python-shaped JSON object of the inner payload.
    #[must_use]
    pub fn to_value(&self) -> Value {
        match self {
            Self::InvokeStart(payload) => to_json_value(payload),
            Self::StepStart(payload) => to_json_value(payload),
            Self::StepEnd(payload) => to_json_value(payload),
            Self::ToolCallStart(payload) => to_json_value(payload),
            Self::ToolCallEnd(payload) => to_json_value(payload),
            Self::InvokeEnd(payload) => to_json_value(payload),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::{
        CallbackPayload, InvokeEndPayload, InvokeStartPayload, StepEndPayload, StepStartPayload,
        ToolCallEndPayload, ToolCallPayload,
    };
    use crate::callbacks::AgentCallbackEvent;
    use crate::output::AgentOutput;
    use openjiuwen_foundation::{ToolCall, Usage};
    use serde_json::json;
    fn invoke_start() -> CallbackPayload {
        CallbackPayload::InvokeStart(InvokeStartPayload {
            agent_id: "agent-1".to_owned(),
            session_id: Some("session-1".to_owned()),
            query: "hello".to_owned(),
            max_iterations: 5,
        })
    }
    #[test]
    fn event_matches_the_variant() {
        let payloads = vec![
            (
                invoke_start(),
                AgentCallbackEvent::InvokeStart,
                "agent-1",
                Some("session-1"),
            ),
            (
                CallbackPayload::StepStart(StepStartPayload {
                    agent_id: "agent-2".to_owned(),
                    ..StepStartPayload::default()
                }),
                AgentCallbackEvent::StepStart,
                "agent-2",
                None,
            ),
            (
                CallbackPayload::StepEnd(StepEndPayload {
                    agent_id: "agent-3".to_owned(),
                    ..StepEndPayload::default()
                }),
                AgentCallbackEvent::StepEnd,
                "agent-3",
                None,
            ),
            (
                CallbackPayload::ToolCallStart(ToolCallPayload {
                    agent_id: "agent-4".to_owned(),
                    ..ToolCallPayload::default()
                }),
                AgentCallbackEvent::ToolCallStart,
                "agent-4",
                None,
            ),
            (
                CallbackPayload::ToolCallEnd(ToolCallEndPayload {
                    agent_id: "agent-5".to_owned(),
                    ..ToolCallEndPayload::default()
                }),
                AgentCallbackEvent::ToolCallEnd,
                "agent-5",
                None,
            ),
            (
                CallbackPayload::InvokeEnd(InvokeEndPayload {
                    agent_id: "agent-6".to_owned(),
                    ..InvokeEndPayload::default()
                }),
                AgentCallbackEvent::InvokeEnd,
                "agent-6",
                None,
            ),
        ];
        for (payload, event, agent_id, session_id) in payloads {
            assert_eq!(payload.event(), event);
            assert_eq!(payload.agent_id(), agent_id);
            assert_eq!(payload.session_id(), session_id);
        }
    }
    #[test]
    fn to_value_is_the_python_shaped_object() {
        let value = invoke_start().to_value();
        assert_eq!(
            value,
            json!({
                "agent_id": "agent-1",
                "session_id": "session-1",
                "query": "hello",
                "max_iterations": 5,
            })
        );
    }
    #[test]
    fn to_value_keeps_a_null_session_and_the_output_dict() {
        let payload = CallbackPayload::InvokeEnd(InvokeEndPayload {
            agent_id: "agent-1".to_owned(),
            session_id: None,
            output: AgentOutput::answer("done"),
        });
        let value = payload.to_value();
        assert_eq!(value["agent_id"], "agent-1");
        assert!(value["session_id"].is_null());
        assert_eq!(value["output"]["output"], "done");
        assert_eq!(value["output"]["result_type"], "answer");
        assert_eq!(
            value.as_object().map(serde_json::Map::len),
            Some(3),
            "only the payload fields are serialized"
        );
    }
    #[test]
    fn to_value_round_trips_through_serde() {
        let payload = invoke_start();
        let value = payload.to_value();
        let parsed: InvokeStartPayload =
            serde_json::from_value(value).expect("payload deserializes");
        assert_eq!(
            parsed,
            InvokeStartPayload {
                agent_id: "agent-1".to_owned(),
                session_id: Some("session-1".to_owned()),
                query: "hello".to_owned(),
                max_iterations: 5,
            }
        );
    }
    #[test]
    fn payloads_tolerate_unknown_and_missing_fields() {
        let parsed: StepStartPayload = serde_json::from_value(json!({
            "agent_id": "agent-1",
            "iteration": 2,
            "future_field": {"nested": true},
        }))
        .expect("unknown fields are ignored");
        assert_eq!(parsed.agent_id, "agent-1");
        assert_eq!(parsed.iteration, 2);
        assert!(parsed.session_id.is_none());
        assert_eq!(parsed.messages.len(), 0);
        assert_eq!(parsed.tools.len(), 0);
        let parsed: ToolCallEndPayload = serde_json::from_value(json!({
            "call": {"name": "search", "arguments": "{}"},
            "text": "found",
            "success": true,
        }))
        .expect("partial tool-call payload deserializes");
        assert_eq!(parsed.call.name, "search");
        assert!(parsed.success);
        assert!(parsed.error.is_none());
    }
    #[test]
    fn step_end_carries_usage_and_finish_reason() {
        let payload = CallbackPayload::StepEnd(StepEndPayload {
            agent_id: "agent-1".to_owned(),
            session_id: Some("session-1".to_owned()),
            iteration: 1,
            output_text: "answer".to_owned(),
            tool_call_count: 2,
            usage: Usage::new(10, 4),
            finish_reason: Some("tool_calls".to_owned()),
        });
        let value = payload.to_value();
        assert_eq!(value["usage"]["input_tokens"], 10);
        assert_eq!(value["usage"]["output_tokens"], 4);
        assert_eq!(value["usage"]["total_tokens"], 14);
        assert_eq!(value["finish_reason"], "tool_calls");
        assert_eq!(value["tool_call_count"], 2);
    }
    #[test]
    fn callback_payload_round_trips_through_serde() {
        let payload = CallbackPayload::ToolCallStart(ToolCallPayload {
            agent_id: "agent-1".to_owned(),
            session_id: Some("session-1".to_owned()),
            iteration: 3,
            call: ToolCall::new("search").with_id("call-1"),
        });
        let value = serde_json::to_value(&payload).expect("payload serializes");
        assert_eq!(value["tool_call_start"]["call"]["name"], "search");
        let parsed: CallbackPayload = serde_json::from_value(value).expect("payload deserializes");
        assert_eq!(parsed, payload);
        assert_eq!(parsed.event(), AgentCallbackEvent::ToolCallStart);
    }
}
