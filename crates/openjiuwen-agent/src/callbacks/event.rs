//! The lifecycle events a callback can subscribe to.
//!
//! The wire names are the `snake_case` event names used by Python's
//! `AgentCallbackEvent`.
use serde::{Deserialize, Serialize};
/// One firing point of an agent invocation.
///
/// The variants are listed in lifecycle order; every step event of one step
/// precedes the next step's events, and tool-call events nest inside a step.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum AgentCallbackEvent {
    /// A new invocation starts, before any model call.
    #[default]
    InvokeStart,
    /// One `ReAct` step starts: the window and tools about to be sent.
    StepStart,
    /// One `ReAct` step ends, after the model reply and its tool calls.
    StepEnd,
    /// About to execute one tool call.
    ToolCallStart,
    /// One tool call returned a result.
    ToolCallEnd,
    /// The invocation finished and its terminal output is final.
    InvokeEnd,
}
impl AgentCallbackEvent {
    /// Every event, in lifecycle order.
    pub const ALL: [Self; 6] = [
        Self::InvokeStart,
        Self::StepStart,
        Self::StepEnd,
        Self::ToolCallStart,
        Self::ToolCallEnd,
        Self::InvokeEnd,
    ];
    /// The `snake_case` wire name of the event.
    ///
    /// `invoke_start`, `step_start`, `step_end`, `tool_call_start`,
    /// `tool_call_end`, and `invoke_end` match the serde representation.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvokeStart => "invoke_start",
            Self::StepStart => "step_start",
            Self::StepEnd => "step_end",
            Self::ToolCallStart => "tool_call_start",
            Self::ToolCallEnd => "tool_call_end",
            Self::InvokeEnd => "invoke_end",
        }
    }
}
#[cfg(test)]
mod tests {
    use super::AgentCallbackEvent;
    #[test]
    fn as_str_matches_the_wire_names() {
        assert_eq!(AgentCallbackEvent::InvokeStart.as_str(), "invoke_start");
        assert_eq!(AgentCallbackEvent::StepStart.as_str(), "step_start");
        assert_eq!(AgentCallbackEvent::StepEnd.as_str(), "step_end");
        assert_eq!(
            AgentCallbackEvent::ToolCallStart.as_str(),
            "tool_call_start"
        );
        assert_eq!(AgentCallbackEvent::ToolCallEnd.as_str(), "tool_call_end");
        assert_eq!(AgentCallbackEvent::InvokeEnd.as_str(), "invoke_end");
    }
    #[test]
    fn serde_uses_snake_case_and_round_trips() {
        for event in AgentCallbackEvent::ALL {
            let value = serde_json::to_value(event).expect("event serializes");
            assert_eq!(value, serde_json::json!(event.as_str()));
            let parsed: AgentCallbackEvent =
                serde_json::from_value(value).expect("event deserializes");
            assert_eq!(parsed, event);
        }
    }
    #[test]
    fn all_lists_every_event_once_in_lifecycle_order() {
        assert_eq!(AgentCallbackEvent::ALL.len(), 6);
        for (index, event) in AgentCallbackEvent::ALL.iter().enumerate() {
            assert_eq!(
                AgentCallbackEvent::ALL
                    .iter()
                    .filter(|candidate| *candidate == event)
                    .count(),
                1
            );
            assert_eq!(AgentCallbackEvent::ALL[index], *event);
        }
        assert_eq!(
            AgentCallbackEvent::default(),
            AgentCallbackEvent::InvokeStart
        );
        let mut sorted = AgentCallbackEvent::ALL;
        sorted.sort_unstable();
        assert_eq!(sorted, AgentCallbackEvent::ALL);
    }
}
