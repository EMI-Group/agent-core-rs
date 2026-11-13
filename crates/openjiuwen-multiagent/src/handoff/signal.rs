//! Handoff signal extraction from agent results, observations, and sessions.
//!
//! Python parity: `openjiuwen/core/multi_agent/teams/handoff/handoff_signal.py`.
//! A handoff directive is a payload carrying a non-empty string under
//! [`HANDOFF_TARGET_KEY`] plus optional
//! [`HANDOFF_MESSAGE_KEY`]/[`HANDOFF_REASON_KEY`] strings. The payload usually
//! arrives as the direct result of the `transfer_to_*` tool, but a model may
//! wrap it (`output`/`result`/`content`) or generate a follow-up answer that
//! overwrites it, so three locators exist:
//!
//! - [`extract_handoff_signal`] inspects a result value (top level first, then
//!   the wrapped probes);
//! - [`extract_handoff_signal_from_observations`] reads the recorded tool
//!   observations, newest first;
//! - [`extract_handoff_signal_from_session`] falls back to the session context's
//!   `tool` messages, newest first.
//!
//! Python's scan tries `json.loads` first and falls back to `ast.literal_eval`,
//! so a Python-repr tool payload (`"{'__handoff_to__': 'b'}"`) is recovered too;
//! both locators use [`crate::pyrepr::literal_or_json`], which keeps that order
//! and skips content that neither grammar accepts.

use std::sync::Arc;

use openjiuwen_agent::Observation;
use openjiuwen_session::Session;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{HANDOFF_MESSAGE_KEY, HANDOFF_REASON_KEY, HANDOFF_TARGET_KEY};

/// Context-engine key holding the default context (Python `default_context_id`).
///
/// Shared with [`super::container`], which mirrors the same messages.
pub(crate) const DEFAULT_CONTEXT_ID: &str = "default_context_id";

/// Keys probed, in order, when the directive is wrapped one level deep.
const NESTED_PROBE_KEYS: [&str; 3] = ["output", "result", "content"];

/// A handoff directive produced by [`extract_handoff_signal`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct HandoffSignal {
    /// Id of the agent control is transferred to.
    pub target: String,
    /// Optional context message forwarded to the target agent.
    pub message: Option<String>,
    /// Optional human-readable reason for the transfer.
    pub reason: Option<String>,
}

/// Returns `value` as a non-empty string, or `None`.
fn non_empty_string(value: Option<&Value>) -> Option<String> {
    value
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
        .map(ToOwned::to_owned)
}

/// Reads a handoff directive out of one payload object.
///
/// The payload must carry a non-empty string target; the message and reason are
/// taken only when they are non-empty strings.
fn signal_from_payload(payload: &Value) -> Option<HandoffSignal> {
    let object = payload.as_object()?;
    let target = non_empty_string(object.get(HANDOFF_TARGET_KEY))?;
    Some(HandoffSignal {
        target,
        message: non_empty_string(object.get(HANDOFF_MESSAGE_KEY)),
        reason: non_empty_string(object.get(HANDOFF_REASON_KEY)),
    })
}

/// Extracts a handoff directive from an agent result value.
///
/// A non-empty string at the top-level target key wins. Otherwise the
/// `output`, `result`, and `content` fields are probed in that order for an
/// object that carries a non-empty string target; a top-level key holding an
/// empty or non-string value therefore falls through to those probes.
#[must_use]
pub fn extract_handoff_signal(result: &Value) -> Option<HandoffSignal> {
    if let Some(signal) = signal_from_payload(result) {
        return Some(signal);
    }
    NESTED_PROBE_KEYS
        .iter()
        .filter_map(|key| result.get(*key))
        .find_map(signal_from_payload)
}

/// Extracts a handoff directive from recorded tool observations.
///
/// Observations are appended in call order, so they are scanned newest first:
/// the most recent handoff wins when an agent called `transfer_to_*` twice. Each
/// content is read as JSON first and as a Python literal second (Python's
/// `json.loads` → `ast.literal_eval` order), and content that neither grammar
/// accepts is skipped.
#[must_use]
pub fn extract_handoff_signal_from_observations(
    observations: &[Observation],
) -> Option<HandoffSignal> {
    observations.iter().rev().find_map(|observation| {
        let parsed = crate::pyrepr::literal_or_json(&observation.content)?;
        extract_handoff_signal(&parsed)
    })
}

/// Extracts a handoff directive from an agent result or, failing that, from the
/// session context's `tool` messages.
///
/// The result-level extraction runs first. When it finds nothing,
/// `session.get_state("context")["default_context_id"]["messages"]` is scanned
/// newest first: every `tool` message whose string `content` parses as JSON or,
/// failing that, as a Python literal is inspected for a directive payload, which
/// recovers a handoff that a follow-up model answer overwrote. The two grammars
/// are tried in Python's `json.loads` → `ast.literal_eval` order; non-`tool`
/// roles and content that neither grammar accepts are skipped.
pub async fn extract_handoff_signal_from_session(
    result: &Value,
    session: &Arc<dyn Session>,
) -> Option<HandoffSignal> {
    if let Some(signal) = extract_handoff_signal(result) {
        return Some(signal);
    }
    let context = session.get_state("context").await.ok().flatten()?;
    let messages = context
        .get(DEFAULT_CONTEXT_ID)?
        .get("messages")?
        .as_array()?;
    messages.iter().rev().find_map(|message| {
        if message.get("role").and_then(Value::as_str) != Some("tool") {
            return None;
        }
        let text = message.get("content").and_then(Value::as_str)?;
        let parsed = crate::pyrepr::literal_or_json(text)?;
        signal_from_payload(&parsed)
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use openjiuwen_session::AgentSession;
    use serde_json::json;

    use super::*;

    fn observation(content: &Value) -> Observation {
        Observation {
            content: content.to_string(),
            ..Observation::default()
        }
    }

    #[test]
    fn top_level_signal_wins_over_nested_probes() {
        let result = json!({
            HANDOFF_TARGET_KEY: "top",
            "output": {HANDOFF_TARGET_KEY: "nested"}
        });
        let signal = extract_handoff_signal(&result).unwrap();
        assert_eq!(signal.target, "top");
        assert_eq!(signal.message, None);
        assert_eq!(signal.reason, None);
    }

    #[test]
    fn nested_probes_are_ordered_output_result_content() {
        let all = json!({
            "output": {HANDOFF_TARGET_KEY: "from_output"},
            "result": {HANDOFF_TARGET_KEY: "from_result"},
            "content": {HANDOFF_TARGET_KEY: "from_content"}
        });
        assert_eq!(extract_handoff_signal(&all).unwrap().target, "from_output");

        let rest = json!({
            "result": {HANDOFF_TARGET_KEY: "from_result"},
            "content": {HANDOFF_TARGET_KEY: "from_content"}
        });
        assert_eq!(extract_handoff_signal(&rest).unwrap().target, "from_result");

        let content = json!({"content": {HANDOFF_TARGET_KEY: "from_content"}});
        assert_eq!(
            extract_handoff_signal(&content).unwrap().target,
            "from_content"
        );
    }

    #[test]
    fn empty_top_level_target_falls_through() {
        let result = json!({
            HANDOFF_TARGET_KEY: "",
            "output": {HANDOFF_TARGET_KEY: "nested"}
        });
        assert_eq!(extract_handoff_signal(&result).unwrap().target, "nested");

        let non_string = json!({
            HANDOFF_TARGET_KEY: 7,
            "result": {HANDOFF_TARGET_KEY: "nested"}
        });
        assert_eq!(
            extract_handoff_signal(&non_string).unwrap().target,
            "nested"
        );
    }

    #[test]
    fn message_and_reason_require_non_empty_strings() {
        let result = json!({
            HANDOFF_TARGET_KEY: "b",
            HANDOFF_MESSAGE_KEY: "please handle this",
            HANDOFF_REASON_KEY: "busy"
        });
        let signal = extract_handoff_signal(&result).unwrap();
        assert_eq!(signal.message.as_deref(), Some("please handle this"));
        assert_eq!(signal.reason.as_deref(), Some("busy"));

        let empty = json!({
            HANDOFF_TARGET_KEY: "b",
            HANDOFF_MESSAGE_KEY: "",
            HANDOFF_REASON_KEY: 3
        });
        let signal = extract_handoff_signal(&empty).unwrap();
        assert_eq!(signal.message, None);
        assert_eq!(signal.reason, None);
    }

    #[test]
    fn no_signal_without_a_target() {
        assert!(extract_handoff_signal(&json!({"output": "hello"})).is_none());
        assert!(extract_handoff_signal(&json!({HANDOFF_TARGET_KEY: ""})).is_none());
        assert!(extract_handoff_signal(&json!([1, 2, 3])).is_none());
        assert!(extract_handoff_signal(&Value::Null).is_none());
    }

    #[test]
    fn observations_are_scanned_newest_first() {
        let observations = vec![
            observation(&json!({
                HANDOFF_TARGET_KEY: "old",
                HANDOFF_REASON_KEY: "first"
            })),
            observation(&json!({"unrelated": true})),
            observation(&json!({
                HANDOFF_TARGET_KEY: "new",
                HANDOFF_REASON_KEY: "second"
            })),
        ];
        let signal = extract_handoff_signal_from_observations(&observations).unwrap();
        assert_eq!(signal.target, "new");
        assert_eq!(signal.reason.as_deref(), Some("second"));

        let unparseable = vec![
            Observation {
                content: "not json".to_owned(),
                ..Observation::default()
            },
            observation(&json!({HANDOFF_TARGET_KEY: "later"})),
        ];
        assert_eq!(
            extract_handoff_signal_from_observations(&unparseable)
                .unwrap()
                .target,
            "later"
        );
        assert!(extract_handoff_signal_from_observations(&[]).is_none());
    }

    #[test]
    fn observations_accept_python_repr_payloads() {
        let observations = vec![Observation {
            content:
                "{'__handoff_to__': 'b', '__handoff_reason__': 'py', 'flag': True, 'note': None}"
                    .to_owned(),
            ..Observation::default()
        }];
        let signal = extract_handoff_signal_from_observations(&observations).unwrap();
        assert_eq!(signal.target, "b");
        assert_eq!(signal.reason.as_deref(), Some("py"));

        // Garbage is still skipped, and an older directive is still found.
        let mixed = vec![
            Observation {
                content: "{'__handoff_to__': 'older'}".to_owned(),
                ..Observation::default()
            },
            Observation {
                content: "{'unterminated".to_owned(),
                ..Observation::default()
            },
        ];
        assert_eq!(
            extract_handoff_signal_from_observations(&mixed)
                .unwrap()
                .target,
            "older"
        );
    }

    #[test]
    fn json_payloads_keep_their_json_reading() {
        // `true`/`null` are JSON-only spellings, so a recovered signal proves the
        // JSON pass runs first.
        let observations = vec![Observation {
            content: json!({
                HANDOFF_TARGET_KEY: "b",
                HANDOFF_REASON_KEY: "json",
                "flag": true,
                "note": null
            })
            .to_string(),
            ..Observation::default()
        }];
        let signal = extract_handoff_signal_from_observations(&observations).unwrap();
        assert_eq!(signal.target, "b");
        assert_eq!(signal.reason.as_deref(), Some("json"));

        // A payload both grammars accept keeps the JSON reading.
        let shared = vec![Observation {
            content: "{\"__handoff_to__\": \"both\"}".to_owned(),
            ..Observation::default()
        }];
        assert_eq!(
            extract_handoff_signal_from_observations(&shared)
                .unwrap()
                .target,
            "both"
        );
    }

    #[tokio::test]
    async fn session_extraction_reads_tool_messages() {
        let session: Arc<dyn Session> = Arc::new(AgentSession::new("s-1"));
        session
            .update_state(json!({"context": {"default_context_id": {"messages": [
                {"role": "user", "content": "hi"},
                {"role": "tool", "content": "not json"},
                {"role": "tool", "content": "{" },
                {"role": "tool", "content": "{\"__handoff_to__\": \"old\"}"},
                {"role": "tool", "content": "{\"__handoff_to__\": \"new\", \"__handoff_reason__\": \"done\"}"}
            ]}}}))
            .await
            .unwrap();

        let signal = extract_handoff_signal_from_session(&json!({"output": "plain"}), &session)
            .await
            .unwrap();
        assert_eq!(signal.target, "new");
        assert_eq!(signal.reason.as_deref(), Some("done"));

        // A result-level directive wins over the session history.
        let result = json!({HANDOFF_TARGET_KEY: "from_result"});
        let signal = extract_handoff_signal_from_session(&result, &session)
            .await
            .unwrap();
        assert_eq!(signal.target, "from_result");
    }

    #[tokio::test]
    async fn session_scans_python_repr_tool_messages() {
        let session: Arc<dyn Session> = Arc::new(AgentSession::new("s-3"));
        session
            .update_state(json!({"context": {"default_context_id": {"messages": [
                {"role": "tool", "content": "{\"__handoff_to__\": \"json\"}"},
                {"role": "tool", "content": "{'__handoff_to__': 'py', '__handoff_reason__': 'single', 'ok': True, 'note': None}"}
            ]}}}))
            .await
            .unwrap();

        let signal = extract_handoff_signal_from_session(&json!({"output": "plain"}), &session)
            .await
            .unwrap();
        assert_eq!(signal.target, "py", "the newest tool message wins");
        assert_eq!(signal.reason.as_deref(), Some("single"));
    }

    #[tokio::test]
    async fn session_skips_unparseable_content_and_keeps_scanning() {
        let session: Arc<dyn Session> = Arc::new(AgentSession::new("s-4"));
        session
            .update_state(json!({"context": {"default_context_id": {"messages": [
                {"role": "tool", "content": "{'__handoff_to__': 'older', '__handoff_reason__': 'py'}"},
                {"role": "tool", "content": "{'unterminated"},
                {"role": "tool", "content": "not a literal either"}
            ]}}}))
            .await
            .unwrap();

        let signal = extract_handoff_signal_from_session(&json!({}), &session)
            .await
            .unwrap();
        assert_eq!(signal.target, "older");
        assert_eq!(signal.reason.as_deref(), Some("py"));
    }

    #[tokio::test]
    async fn session_extraction_ignores_other_roles_and_missing_state() {
        let session: Arc<dyn Session> = Arc::new(AgentSession::new("s-2"));
        assert!(extract_handoff_signal_from_session(&json!({}), &session)
            .await
            .is_none());

        session
            .update_state(json!({"context": {"default_context_id": {"messages": [
                {"role": "assistant", "content": "{\"__handoff_to__\": \"x\"}"}
            ]}}}))
            .await
            .unwrap();
        assert!(extract_handoff_signal_from_session(&json!({}), &session)
            .await
            .is_none());
    }
}
