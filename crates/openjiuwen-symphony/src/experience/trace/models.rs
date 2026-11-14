//! The parsed-trace record written by the session parser and read by clustering.
//!
//! One [`TraceRecord`] is one `request_id` segment of one agent session: the user query,
//! the skills the run exercised, the tool calls it made, its final text, and whether it
//! failed. It is the data type Python's `experience/models.py` calls `TraceRecord`, with
//! the fields the Rust pipeline needs (session identity, duration, timestamps) added.
//!
//! Every field is `serde`-serializable and tolerant of unknown input keys; `timestamp` is
//! metadata only and never decides behavior on its own (record ordering breaks ties on
//! the trace id).
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One tool call observed in a session.
///
/// `arguments` keeps the payload as received (JSON object, parsed JSON string, or a value
/// recovered from a Python-style single-quoted literal). When the payload cannot be
/// decoded, it is [`Value::Null`] rather than a parser error — a trace is evidence, not
/// input, and one odd tool call must not discard the records around it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct TraceToolCall {
    /// The tool name as the agent reported it; blank when the event carried no name.
    pub name: String,
    /// The decoded arguments payload, or [`Value::Null`] when it could not be decoded.
    pub arguments: Value,
    /// The textual tool result, when a matching `tool_result` event followed the call.
    pub result: Option<String>,
}

/// One parsed session segment: one user query and everything the agent did for it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct TraceRecord {
    /// The session-scoped unique key, `"{session_id}/{request_id}"`.
    ///
    /// Python uses the bare `request_id`, so two sessions that reuse a request id collide
    /// and the later record is dropped by the store's dedupe. Prefixing the session id
    /// keeps every record addressable and makes the on-disk cache honest.
    pub trace_id: String,
    /// The session directory name this record was parsed from.
    pub session_id: String,
    /// The `request_id` that segmented the session history.
    pub request_id: String,
    /// The first non-blank user message text of the segment.
    pub query: String,
    /// The segment's final assistant text: the last `chat.final` payload, or the
    /// accumulated `chat.delta` text when no final event arrived.
    pub final_output: String,
    /// The skill ids the run exercised, sorted and deduped.
    ///
    /// Sourced from `skill_tool` arguments (`skill_name`) and `symphony_compose_graph`
    /// arguments (`candidate_skill_ids`) or its result (`raw_output.plan.steps[]`).
    pub skills: Vec<String>,
    /// Whether the segment is usable evidence: no `error` event and no failure marker in
    /// the session metadata.
    pub success: bool,
    /// The largest `duration_ms` reported by the segment's events (`0` when none did).
    pub duration_ms: u64,
    /// The tool calls of the segment, in event order.
    pub tool_calls: Vec<TraceToolCall>,
    /// The error messages of the segment, in event order.
    pub errors: Vec<String>,
    /// The first event timestamp of the segment in epoch seconds; metadata only.
    ///
    /// Millisecond values (`>= 1e12`) are converted, and an absent or non-numeric
    /// timestamp leaves `0.0`, so a record is comparable without a clock.
    pub timestamp: f64,
}

impl TraceRecord {
    /// Reports whether this record carries the signal the experience collector needs:
    /// it succeeded, lists at least one skill, and has a non-blank query.
    ///
    /// Python's `_filter_valid_traces` and `add` apply the same three conditions; a
    /// failure has no positive pattern to reinforce and a skill-less trace cannot be
    /// partitioned by skill set.
    #[must_use]
    pub fn is_experience_candidate(&self) -> bool {
        self.success && !self.skills.is_empty() && !self.query.trim().is_empty()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;

    #[test]
    fn missing_fields_fall_back_to_the_defaults() {
        let parsed: TraceRecord =
            serde_json::from_str(r#"{"trace_id":"s/1","query":"do a thing"}"#)
                .expect("deserialize");
        assert_eq!(parsed.trace_id, "s/1");
        assert_eq!(parsed.query, "do a thing");
        assert!(parsed.skills.is_empty());
        assert!(!parsed.success);
        assert_eq!(parsed.duration_ms, 0);
        assert_eq!(parsed.timestamp, 0.0);
    }

    #[test]
    fn unknown_fields_are_tolerated() {
        let parsed: TraceRecord = serde_json::from_str(
            r#"{"trace_id":"s/1","future_field":{"nested":true},"skills":["a"]}"#,
        )
        .expect("deserialize");
        assert_eq!(parsed.skills, vec!["a"]);
    }

    #[test]
    fn experience_candidate_requires_success_skills_and_query() {
        let mut record = TraceRecord {
            success: true,
            skills: vec!["summarizer".to_string()],
            query: "summarize".to_string(),
            ..TraceRecord::default()
        };
        assert!(record.is_experience_candidate());
        record.query = "   ".to_string();
        assert!(!record.is_experience_candidate());
        record.query = "summarize".to_string();
        record.skills.clear();
        assert!(!record.is_experience_candidate());
        record.skills.push("x".to_string());
        record.success = false;
        assert!(!record.is_experience_candidate());
    }
}
