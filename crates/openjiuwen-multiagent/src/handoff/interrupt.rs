//! Interrupt detection and the post-interrupt session flush.
//!
//! Python parity: `openjiuwen/core/multi_agent/teams/handoff/interrupt.py`.
//! A member agent that pauses for human input returns an interrupt result
//! (`result_type = "interrupt"`); the container agent then persists the chain
//! state and flushes the team session so the interrupt survives a restart.
//!
//! [`flush_team_session`] is deliberately best-effort: an interrupt must reach
//! the caller even when the checkpointer is broken, so failures are logged and
//! never propagated.

use std::sync::Arc;

use openjiuwen_session::{AgentTeamSession, Session};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The `result_type` value that marks an interrupt result.
const INTERRUPT_RESULT_TYPE: &str = "interrupt";

/// An interrupt raised by a member agent in the middle of a handoff chain.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct TeamInterruptSignal {
    /// The agent's full interrupt result, returned to the caller unchanged.
    pub result: Value,
    /// Optional human-readable description of why the chain paused.
    pub message: Option<String>,
}

/// Extracts an interrupt signal from an agent result.
///
/// A result is an interrupt when `result["result_type"] == "interrupt"`, which
/// is how [`openjiuwen_agent::AgentOutput::to_value`] renders a paused run. The
/// message is taken only from a string `message` field.
#[must_use]
pub fn extract_interrupt_signal(result: &Value) -> Option<TeamInterruptSignal> {
    if result.get("result_type").and_then(Value::as_str) != Some(INTERRUPT_RESULT_TYPE) {
        return None;
    }
    Some(TeamInterruptSignal {
        result: result.clone(),
        message: result
            .get("message")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
    })
}

/// Ends the team session stream and commits its checkpoint after an interrupt.
///
/// `None` is a no-op. Both steps are attempted even when the first fails; every
/// failure is logged as a warning and never raised, so the interrupt is always
/// delivered to the caller.
pub async fn flush_team_session(session: Option<&Arc<AgentTeamSession>>) {
    let Some(session) = session else {
        return;
    };
    if let Err(error) = session.end_stream().await {
        tracing::warn!(%error, "handoff: failed to end the team stream after an interrupt");
    }
    if let Err(error) = session.commit().await {
        tracing::warn!(%error, "handoff: failed to commit the team session after an interrupt");
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use futures::StreamExt;
    use openjiuwen_session::StreamEvent;
    use serde_json::json;

    use super::*;

    #[test]
    fn detects_an_interrupt_result() {
        let result = json!({
            "result_type": "interrupt",
            "message": "confirm the write",
            "state": {"step": 2}
        });
        let signal = extract_interrupt_signal(&result).unwrap();
        assert_eq!(signal.message.as_deref(), Some("confirm the write"));
        assert_eq!(signal.result, result);
    }

    #[test]
    fn ignores_other_results() {
        assert!(extract_interrupt_signal(&json!({"result_type": "answer"})).is_none());
        assert!(extract_interrupt_signal(&json!({})).is_none());
        assert!(extract_interrupt_signal(&json!("interrupt")).is_none());
        assert!(extract_interrupt_signal(&Value::Null).is_none());
    }

    #[test]
    fn message_requires_a_string() {
        let signal =
            extract_interrupt_signal(&json!({"result_type": "interrupt", "message": 3})).unwrap();
        assert_eq!(signal.message, None);
        let signal = extract_interrupt_signal(&json!({"result_type": "interrupt"})).unwrap();
        assert_eq!(signal.message, None);
    }

    #[tokio::test]
    async fn flushing_none_is_a_no_op() {
        flush_team_session(None).await;
    }

    #[tokio::test]
    async fn flushing_ends_the_stream() {
        let session = Arc::new(AgentTeamSession::new("team"));
        let mut events = session.subscribe();
        flush_team_session(Some(&session)).await;
        assert!(matches!(events.next().await, Some(Ok(StreamEvent::End))));
    }
}
