//! Tool-approval bridging between a teammate's interrupt and the leader's decision.
//!
//! Python parity source: `openjiuwen/agent_teams/rails/tool_approval_rail.py` and
//! `openjiuwen/harness/rails/interrupt/confirm_rail.py` (v0.1.20). A teammate's tool call is
//! interrupted with [`InteractionKind::TeamToolApproval`]; the leader answers with a
//! [`ConfirmPayload`]; the runtime turns that answer into a
//! [`ToolApprovalResultEvent`] so the waiting teammate
//! can resume or reject the call.
//!
//! The request id is derived from the tool-call id rather than minted randomly, so asking twice
//! about the same interrupted call yields the same interaction id (one decision, one id). This
//! is a documented deviation from Python, which mints `intr_<uuid>` per request.
//!
//! # Example
//!
//! ```
//! use openjiuwen_agent_teams::interaction::{ConfirmPayload, ToolApprovalRequest};
//! use serde_json::json;
//!
//! let request = ToolApprovalRequest {
//!     team_name: "team".to_owned(),
//!     member_name: "dev-1".to_owned(),
//!     tool_call_id: "call_7".to_owned(),
//!     tool_name: "shell".to_owned(),
//!     tool_args: json!({"command": "ls"}),
//! };
//! let interaction = request.to_interaction();
//! assert_eq!(interaction.id, "intr_team_tool_approval_call_7");
//! assert_eq!(
//!     interaction.question,
//!     json!({"tool_name": "shell", "tool_args": {"command": "ls"}, "member_name": "dev-1"})
//! );
//! ```
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_session::{InteractionKind, InteractionRequest};

use crate::schema::{EventPayload, ToolApprovalResultEvent};

/// The answer to a tool-approval request.
///
/// Shape mirrors the harness `ConfirmPayload`: `approved` is required; `feedback` and
/// `auto_confirm` default when absent. Unknown fields are tolerated so a newer client may add
/// fields without breaking the parse.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ConfirmPayload {
    /// Whether the interrupted tool call may proceed.
    pub approved: bool,
    /// Leader feedback shown to the teammate when the call is rejected.
    #[serde(default)]
    pub feedback: String,
    /// Whether future calls to the same tool are approved automatically.
    #[serde(default)]
    pub auto_confirm: bool,
}

/// One interrupted teammate tool call awaiting a leader decision.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ToolApprovalRequest {
    /// Team the member belongs to.
    pub team_name: String,
    /// Member whose tool call is interrupted.
    pub member_name: String,
    /// Identity of the interrupted tool call.
    pub tool_call_id: String,
    /// Name of the tool being called.
    pub tool_name: String,
    /// Arguments passed to the tool.
    pub tool_args: Value,
}

impl ToolApprovalRequest {
    /// Renders the harness interaction request the leader answers.
    ///
    /// The kind is [`InteractionKind::TeamToolApproval`] and the question carries exactly
    /// `tool_name`, `tool_args` and `member_name`. The id is derived from `tool_call_id` (see
    /// the module docs), so it is stable across re-renders of the same request.
    #[must_use]
    pub fn to_interaction(&self) -> InteractionRequest {
        let question = json!({
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "member_name": self.member_name,
        });
        let mut request = InteractionRequest::new(InteractionKind::TeamToolApproval, question);
        request.id = format!("intr_team_tool_approval_{}", self.tool_call_id);
        request
    }
}

/// Parses a leader's raw answer into a [`ConfirmPayload`].
///
/// Python re-interrupts on a malformed answer so the leader can answer again; Rust surfaces the
/// problem as [`Error::Validation`] and leaves re-asking to the caller.
///
/// # Errors
///
/// Returns [`Error::Validation`] when `answer` is not an object, when `approved` is missing or
/// not a boolean, or when `feedback` / `auto_confirm` have the wrong type. The message names
/// `question_id` so the caller can correlate the failure with the pending interaction.
pub fn parse_confirm(question_id: &str, answer: &Value) -> Result<ConfirmPayload> {
    if !answer.is_object() {
        return Err(Error::validation(format!(
            "invalid confirmation answer for interaction {question_id}: expected a JSON object"
        )));
    }
    serde_json::from_value(answer.clone()).map_err(|error| {
        Error::validation(format!(
            "invalid confirmation answer for interaction {question_id}: {error}"
        ))
    })
}

/// Builds the bus event announcing the leader's decision.
///
/// The event carries the team, the requesting member, the tool-call id and the verdict, which is
/// everything the waiting teammate needs to resume the interrupted call.
///
/// Deviation: Python's `ToolApprovalResultEvent` also carries `feedback` and `auto_confirm`;
/// this crate's schema event does not (it is owned by `crate::schema`), so the payload value is
/// limited to the approval bit.
#[must_use]
pub fn approval_result_event(
    request: &ToolApprovalRequest,
    payload: &ConfirmPayload,
) -> EventPayload {
    EventPayload::ToolApprovalResult(ToolApprovalResultEvent {
        team_name: request.team_name.clone(),
        member_name: request.member_name.clone(),
        tool_call_id: request.tool_call_id.clone(),
        approved: payload.approved,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::schema::ToolApprovalResultEvent;
    use openjiuwen_session::InteractionKind;

    fn request() -> ToolApprovalRequest {
        ToolApprovalRequest {
            team_name: "team".to_owned(),
            member_name: "dev-1".to_owned(),
            tool_call_id: "call_7".to_owned(),
            tool_name: "shell".to_owned(),
            tool_args: json!({"command": "rm -rf build"}),
        }
    }

    #[test]
    fn to_interaction_uses_team_tool_approval_kind_and_stable_id() {
        let request = request();
        let first = request.to_interaction();
        let second = request.to_interaction();

        assert_eq!(first.kind, InteractionKind::TeamToolApproval);
        assert_eq!(
            first.id,
            format!("intr_team_tool_approval_{}", request.tool_call_id)
        );
        assert_eq!(first.id, second.id);
        assert_eq!(first.component_id, None);
        assert_eq!(first.options.len(), 0);
        assert_eq!(first.metadata.len(), 0);
    }

    #[test]
    fn to_interaction_question_carries_tool_and_member_identity() {
        let interaction = request().to_interaction();
        assert_eq!(
            interaction.question,
            json!({
                "tool_name": "shell",
                "tool_args": {"command": "rm -rf build"},
                "member_name": "dev-1",
            })
        );
        assert_eq!(
            interaction
                .question
                .get("member_name")
                .and_then(Value::as_str),
            Some("dev-1")
        );
    }

    #[test]
    fn parse_confirm_accepts_approval_and_defaults() {
        let payload = parse_confirm("intr_1", &json!({"approved": true})).unwrap();
        assert_eq!(
            payload,
            ConfirmPayload {
                approved: true,
                feedback: String::new(),
                auto_confirm: false,
            }
        );
    }

    #[test]
    fn parse_confirm_reads_feedback_and_auto_confirm() {
        let payload = parse_confirm(
            "intr_1",
            &json!({"approved": false, "feedback": "use rsync", "auto_confirm": true}),
        )
        .unwrap();
        assert!(!payload.approved);
        assert_eq!(payload.feedback, "use rsync");
        assert!(payload.auto_confirm);
    }

    #[test]
    fn parse_confirm_tolerates_unknown_fields() {
        let payload = parse_confirm(
            "intr_1",
            &json!({"approved": true, "persist_allow": false, "future": [1, 2]}),
        )
        .unwrap();
        assert!(payload.approved);
    }

    #[test]
    fn parse_confirm_rejects_malformed_answers() {
        for answer in [
            json!(null),
            json!(true),
            json!("yes"),
            json!([]),
            json!({}),
            json!({"approved": "yes"}),
            json!({"approved": true, "feedback": 5}),
            json!({"approved": true, "auto_confirm": "no"}),
        ] {
            let error = parse_confirm("intr_1", &answer).unwrap_err();
            let message = error.to_string();
            assert!(
                message.starts_with("validation:"),
                "unexpected error for {answer}: {message}"
            );
            assert!(
                message.contains("intr_1"),
                "error should name the interaction: {message}"
            );
        }
    }

    #[test]
    fn approval_result_event_carries_the_verdict() {
        let accepted = ConfirmPayload {
            approved: true,
            ..ConfirmPayload::default()
        };
        let rejected = ConfirmPayload {
            approved: false,
            feedback: "no".to_owned(),
            ..ConfirmPayload::default()
        };

        assert_eq!(
            approval_result_event(&request(), &accepted),
            EventPayload::ToolApprovalResult(ToolApprovalResultEvent {
                team_name: "team".to_owned(),
                member_name: "dev-1".to_owned(),
                tool_call_id: "call_7".to_owned(),
                approved: true,
            })
        );
        assert_eq!(
            approval_result_event(&request(), &rejected),
            EventPayload::ToolApprovalResult(ToolApprovalResultEvent {
                team_name: "team".to_owned(),
                member_name: "dev-1".to_owned(),
                tool_call_id: "call_7".to_owned(),
                approved: false,
            })
        );
    }

    #[test]
    fn approval_result_event_serializes_with_python_event_type() {
        let payload = approval_result_event(
            &request(),
            &ConfirmPayload {
                approved: true,
                ..ConfirmPayload::default()
            },
        );
        assert_eq!(payload.event_type(), "tool_approval_result");
        assert_eq!(
            payload.payload_value().unwrap(),
            json!({
                "team_name": "team",
                "member_name": "dev-1",
                "tool_call_id": "call_7",
                "approved": true,
            })
        );
    }
}
