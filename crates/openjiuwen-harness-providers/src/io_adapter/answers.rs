//! Parsing host answers into interaction responses.
//!
//! A host answers a pending interaction through the kernel's interactive-input payload, so the
//! adapter has to map an arbitrary JSON value onto the typed response the harness awaits. The
//! accepted shapes mirror Python's adapter:
//!
//! - a boolean, for a plain approve/deny prompt;
//! - a word (`yes`, `allow`, `approve`, `always`, …), for a text prompt;
//! - an object carrying `action`, `approved`, `remember`, `updated_arguments` or `feedback`, for a
//!   structured approval UI.
//!
//! Every unrecognized value denies the request, so an unknown host shape can never silently
//! approve a tool call.

use openjiuwen_harness_protocol::{
    json_object_from_value, InteractionResponseStatus, JsonObject, JsonValue, ToolApprovalDecision,
    ToolApprovalResponse, UserInputResponse,
};
use openjiuwen_session::interaction::InteractiveInput;

/// Keys that mark a JSON object as an interactive host answer.
const INTERACTIVE_KEYS: [&str; 3] = ["user_inputs", "interrupt_answers", "raw_inputs"];

/// Reason attached to a denial whose answer carries no explanation.
const DEFAULT_DENY_REASON: &str = "Denied by the host.";

/// Recognizes an interactive host answer and decodes it.
///
/// Returns [`None`] when `value` is not an object, carries none of `user_inputs`,
/// `interrupt_answers` or `raw_inputs`, or does not decode into [`InteractiveInput`]; the caller
/// then forwards the payload to the harness unchanged.
#[must_use]
pub(super) fn interactive_input(value: &JsonValue) -> Option<InteractiveInput> {
    let object = value.as_object()?;
    if !INTERACTIVE_KEYS.iter().any(|key| object.contains_key(*key)) {
        return None;
    }
    serde_json::from_value(value.clone()).ok()
}

/// Parses one host answer for a tool approval.
///
/// See the module docs for the accepted shapes; every unrecognized shape denies the request with
/// [`DEFAULT_DENY_REASON`]. `updated_arguments` is carried over whenever the answer is an object
/// that allows the call and supplies rewritten arguments.
#[must_use]
pub(super) fn tool_approval_response(request_id: &str, answer: &JsonValue) -> ToolApprovalResponse {
    match answer {
        JsonValue::Bool(true) => approval(request_id, ToolApprovalDecision::Allow, None, None),
        JsonValue::Bool(false) => approval(request_id, ToolApprovalDecision::Deny, None, None),
        JsonValue::String(text) => {
            let normalized = text.trim().to_ascii_lowercase();
            match normalized.as_str() {
                "yes" | "allow" | "approve" => {
                    approval(request_id, ToolApprovalDecision::Allow, None, None)
                }
                "always" | "allow_always" => approval(
                    request_id,
                    ToolApprovalDecision::AllowForSession,
                    None,
                    None,
                ),
                _ => approval(
                    request_id,
                    ToolApprovalDecision::Deny,
                    None,
                    Some(text.clone()),
                ),
            }
        }
        JsonValue::Object(object) => {
            let object: JsonObject = object.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            object_approval_response(request_id, &object)
        }
        _ => deny_by_default(request_id),
    }
}

/// Parses one host answer for a user-input request.
///
/// Every answer is a completed interaction; the raw value becomes the response content so the
/// harness decides what the answer means.
#[must_use]
pub(super) fn user_input_response(request_id: &str, answer: &JsonValue) -> UserInputResponse {
    UserInputResponse {
        request_id: request_id.to_owned(),
        status: InteractionResponseStatus::Completed,
        content: answer.clone(),
        provider_data: JsonObject::new(),
    }
}

/// Parses a structured approval answer.
#[must_use]
fn object_approval_response(request_id: &str, object: &JsonObject) -> ToolApprovalResponse {
    let updated = object
        .get("updated_arguments")
        .and_then(|value| json_object_from_value(value.clone()));

    if let Some(action) = object.get("action").and_then(JsonValue::as_str) {
        match action {
            "allow_once" | "allow" => {
                return approval(request_id, ToolApprovalDecision::Allow, updated, None);
            }
            "allow_always" => {
                return approval(
                    request_id,
                    ToolApprovalDecision::AllowForSession,
                    updated,
                    None,
                );
            }
            "deny" | "reject" => {
                return approval(request_id, ToolApprovalDecision::Deny, None, None);
            }
            _ => {}
        }
    }

    if object.get("approved").and_then(JsonValue::as_bool) == Some(true) {
        let decision = if object.get("remember").and_then(JsonValue::as_bool) == Some(true) {
            ToolApprovalDecision::AllowForSession
        } else {
            ToolApprovalDecision::Allow
        };
        return approval(request_id, decision, updated, None);
    }

    if let Some(feedback) = object.get("feedback").and_then(JsonValue::as_str) {
        return approval(
            request_id,
            ToolApprovalDecision::Deny,
            None,
            Some(feedback.to_owned()),
        );
    }

    if updated.is_some() {
        return approval(request_id, ToolApprovalDecision::Allow, updated, None);
    }

    deny_by_default(request_id)
}

/// Builds an approval response from its parts.
fn approval(
    request_id: &str,
    decision: ToolApprovalDecision,
    updated_arguments: Option<JsonObject>,
    reason: Option<String>,
) -> ToolApprovalResponse {
    ToolApprovalResponse {
        request_id: request_id.to_owned(),
        decision,
        updated_arguments,
        reason,
        interrupt: false,
        provider_data: JsonObject::new(),
    }
}

/// Builds the denial used for unrecognized answers.
fn deny_by_default(request_id: &str) -> ToolApprovalResponse {
    approval(
        request_id,
        ToolApprovalDecision::Deny,
        None,
        Some(DEFAULT_DENY_REASON.to_owned()),
    )
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use openjiuwen_harness_protocol::{
        json_object_from_value, InteractionResponseStatus, ToolApprovalDecision,
    };
    use serde_json::{json, Value};

    use super::{
        interactive_input, tool_approval_response, user_input_response, DEFAULT_DENY_REASON,
    };

    fn decision(answer: &Value) -> ToolApprovalDecision {
        tool_approval_response("req-1", answer).decision
    }

    #[test]
    fn bool_answers_map_to_allow_and_deny() {
        assert_eq!(decision(&json!(true)), ToolApprovalDecision::Allow);
        assert_eq!(decision(&json!(false)), ToolApprovalDecision::Deny);
        assert!(tool_approval_response("req-1", &json!(true))
            .reason
            .is_none());
    }

    #[test]
    fn word_answers_map_to_decisions_keeping_the_original_text() {
        assert_eq!(decision(&json!("yes")), ToolApprovalDecision::Allow);
        assert_eq!(decision(&json!("  ALLOW ")), ToolApprovalDecision::Allow);
        assert_eq!(decision(&json!("approve")), ToolApprovalDecision::Allow);
        assert_eq!(
            decision(&json!("always")),
            ToolApprovalDecision::AllowForSession
        );
        assert_eq!(
            decision(&json!("allow_always")),
            ToolApprovalDecision::AllowForSession
        );
        let denied = tool_approval_response("req-1", &json!("Not now"));
        assert_eq!(denied.decision, ToolApprovalDecision::Deny);
        assert_eq!(denied.reason.as_deref(), Some("Not now"));
    }

    #[test]
    fn structured_answers_map_to_decisions() {
        assert_eq!(
            decision(&json!({"action": "allow_once"})),
            ToolApprovalDecision::Allow
        );
        assert_eq!(
            decision(&json!({"action": "allow_always"})),
            ToolApprovalDecision::AllowForSession
        );
        assert_eq!(
            decision(&json!({"action": "reject"})),
            ToolApprovalDecision::Deny
        );
        assert_eq!(
            decision(&json!({"approved": true, "remember": true})),
            ToolApprovalDecision::AllowForSession
        );
        assert_eq!(
            decision(&json!({"approved": true})),
            ToolApprovalDecision::Allow
        );
        let feedback = tool_approval_response(
            "req-1",
            &json!({"approved": false, "feedback": "too risky"}),
        );
        assert_eq!(feedback.decision, ToolApprovalDecision::Deny);
        assert_eq!(feedback.reason.as_deref(), Some("too risky"));
        let plain_feedback = tool_approval_response("req-1", &json!({"feedback": "not this time"}));
        assert_eq!(plain_feedback.decision, ToolApprovalDecision::Deny);
        assert_eq!(plain_feedback.reason.as_deref(), Some("not this time"));
    }

    #[test]
    fn updated_arguments_are_carried_on_allowing_shapes() {
        let response = tool_approval_response(
            "req-1",
            &json!({"approved": true, "updated_arguments": {"path": "safe.txt"}}),
        );
        assert_eq!(response.decision, ToolApprovalDecision::Allow);
        assert_eq!(
            response.updated_arguments,
            json_object_from_value(json!({"path": "safe.txt"}))
        );

        let standalone =
            tool_approval_response("req-1", &json!({"updated_arguments": {"path": "safe.txt"}}));
        assert_eq!(standalone.decision, ToolApprovalDecision::Allow);
        assert!(standalone.updated_arguments.is_some());

        let denied = tool_approval_response("req-1", &json!({"action": "deny"}));
        assert!(denied.updated_arguments.is_none());
    }

    #[test]
    fn unrecognized_answers_deny_with_the_default_reason() {
        for answer in [
            json!(null),
            json!(3),
            json!([true]),
            json!({}),
            json!({"action": "later"}),
        ] {
            let response = tool_approval_response("req-1", &answer);
            assert_eq!(response.decision, ToolApprovalDecision::Deny);
            assert_eq!(response.reason.as_deref(), Some(DEFAULT_DENY_REASON));
            assert!(!response.interrupt);
        }
    }

    #[test]
    fn user_input_answers_are_completed_with_the_raw_value() {
        let response = user_input_response("req-1", &json!({"answer": 42}));
        assert_eq!(response.request_id, "req-1");
        assert_eq!(response.status, InteractionResponseStatus::Completed);
        assert_eq!(response.content, json!({"answer": 42}));
    }

    #[test]
    fn interactive_input_requires_a_known_key_and_a_valid_payload() {
        assert!(interactive_input(&json!({"other": 1})).is_none());
        assert!(interactive_input(&json!("text")).is_none());
        assert!(interactive_input(&json!({"interrupt_answers": {"req-1": true}})).is_some());
        assert!(interactive_input(&json!({"user_inputs": {"node": "yes"}})).is_some());
        assert!(interactive_input(&json!({"raw_inputs": {"anything": 1}})).is_some());
        // Wrong types inside a known key fail to decode and are not interactive answers.
        assert!(interactive_input(&json!({"user_inputs": 7})).is_none());

        let input = interactive_input(&json!({
            "user_inputs": {"node": "yes"},
            "interrupt_answers": {"req-1": {"approved": true}},
        }))
        .unwrap();
        assert_eq!(input.user_inputs.len(), 1);
        assert_eq!(input.interrupt_answers.len(), 1);
        assert!(input.raw_inputs.is_none());
    }
}
