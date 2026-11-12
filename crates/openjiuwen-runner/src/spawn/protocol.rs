//! The NDJSON wire a real spawned child speaks.
//!
//! Ports the shape of Python `openjiuwen/core/runner/spawn/protocol.py`
//! (`MessageType`, `Message`, and the newline-delimited serialization): one
//! [`SpawnMessage`] per line of UTF-8 JSON, newline-terminated. Nothing here
//! opens a pipe, reads stdin, or writes stdout — framing is pure data
//! conversion.
//!
//! The timestamp is unix seconds as `f64` (Python used a `datetime`); the
//! message id is a UUID v4 (Python left it `None`).

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::id::uuid_v4;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Longest prefix of a malformed line echoed back in an error message.
const ECHO_PREFIX_CHARS: usize = 64;

/// Message kinds of the stdio protocol (Python `MessageType`).
///
/// Python serialized these as upper-case strings (`"INPUT"`); the Rust port uses
/// the workspace-wide `snake_case` convention.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpawnMessageKind {
    /// Parent to child: one invocation's inputs.
    #[default]
    Input,
    /// Child to parent: a terminal output.
    Output,
    /// Parent to child: liveness probe.
    HealthCheck,
    /// Child to parent: liveness answer.
    HealthCheckResponse,
    /// Parent to child: request a graceful exit.
    Shutdown,
    /// Child to parent: graceful-exit acknowledgement.
    ShutdownAck,
    /// Either direction: the operation failed.
    Error,
    /// Child to parent: one streaming frame.
    StreamChunk,
    /// Child to parent: the stream is complete.
    Done,
}

/// One frame of the newline-delimited protocol.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct SpawnMessage {
    /// Frame kind.
    pub kind: SpawnMessageKind,
    /// Frame body (arbitrary JSON).
    pub payload: Value,
    /// Creation time in unix seconds.
    ///
    /// JSON round-trips this to within one ULP (`serde_json`'s default float
    /// parser is not correctly rounded), so compare timestamps with a tolerance.
    pub timestamp: f64,
    /// Unique frame id (UUID v4), used to correlate replies.
    pub message_id: String,
}

impl SpawnMessage {
    /// Builds a frame with a fresh UUID v4 id and the current unix time.
    #[must_use]
    pub fn new(kind: SpawnMessageKind, payload: Value) -> Self {
        Self {
            kind,
            payload,
            timestamp: unix_secs_now(),
            message_id: uuid_v4(),
        }
    }

    /// Serializes the frame as one newline-terminated JSON line.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the frame cannot be serialized, or
    /// when the serialized text would contain a raw `\n`/`\r` and thus break
    /// the `\n`-delimited framing.
    pub fn to_line(&self) -> Result<String> {
        let mut line = serde_json::to_string(self).map_err(|error| {
            Error::validation(format!("spawn message is not serializable: {error}"))
        })?;
        if line.contains(['\n', '\r']) {
            return Err(Error::validation(
                "spawn message serialization contains a raw newline and would break NDJSON framing",
            ));
        }
        line.push('\n');
        Ok(line)
    }

    /// Parses one line, tolerating the surrounding whitespace a reader may keep.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the line is blank, is not JSON, or is a
    /// JSON object without the required `kind`/`payload`/`timestamp`/`message_id`
    /// fields (or with an unknown `kind`).
    pub fn from_line(line: &str) -> Result<Self> {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Err(Error::validation(
                "spawn message line is empty: expected one NDJSON object",
            ));
        }
        serde_json::from_str(trimmed).map_err(|error| {
            Error::validation(format!(
                "invalid spawn message line '{}': {error}",
                preview(trimmed)
            ))
        })
    }
}

/// Current time in unix seconds, falling back to `0.0` for a clock before the
/// epoch (or a system-time error).
fn unix_secs_now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |elapsed| elapsed.as_secs_f64())
}

/// A short, char-boundary-safe prefix of `text` for error messages.
fn preview(text: &str) -> String {
    let mut shown: String = text.chars().take(ECHO_PREFIX_CHARS).collect();
    if text.chars().count() > ECHO_PREFIX_CHARS {
        shown.push('…');
    }
    shown
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    const ALL_KINDS: [SpawnMessageKind; 9] = [
        SpawnMessageKind::Input,
        SpawnMessageKind::Output,
        SpawnMessageKind::HealthCheck,
        SpawnMessageKind::HealthCheckResponse,
        SpawnMessageKind::Shutdown,
        SpawnMessageKind::ShutdownAck,
        SpawnMessageKind::Error,
        SpawnMessageKind::StreamChunk,
        SpawnMessageKind::Done,
    ];

    /// Compares every field, tolerating `serde_json`'s one-ULP float parsing.
    fn assert_same_frame(left: &SpawnMessage, right: &SpawnMessage) {
        assert_eq!(left.kind, right.kind);
        assert_eq!(left.payload, right.payload);
        assert_eq!(left.message_id, right.message_id);
        assert!(
            (left.timestamp - right.timestamp).abs() < 1e-6,
            "timestamps differ: {} vs {}",
            left.timestamp,
            right.timestamp
        );
    }

    #[test]
    fn kind_defaults_to_input() {
        assert_eq!(SpawnMessageKind::default(), SpawnMessageKind::Input);
    }

    #[test]
    fn kind_wire_names_are_snake_case() {
        let expected = [
            "input",
            "output",
            "health_check",
            "health_check_response",
            "shutdown",
            "shutdown_ack",
            "error",
            "stream_chunk",
            "done",
        ];
        for (kind, name) in ALL_KINDS.iter().zip(expected) {
            assert_eq!(serde_json::to_value(kind).unwrap(), json!(name));
            let parsed: SpawnMessageKind = serde_json::from_value(json!(name)).unwrap();
            assert_eq!(&parsed, kind);
        }
    }

    #[test]
    fn new_stamps_a_unique_id_and_a_timestamp() {
        let first = SpawnMessage::new(SpawnMessageKind::Input, json!({"a": 1}));
        let second = SpawnMessage::new(SpawnMessageKind::Input, json!({"a": 1}));
        assert_ne!(first.message_id, "");
        assert_ne!(first.message_id, second.message_id);
        assert!(
            first.timestamp > 1_704_067_200.0,
            "timestamp before 2024-01-01: {}",
            first.timestamp
        );
        assert!(second.timestamp >= first.timestamp);
    }

    #[test]
    fn message_ids_are_unique_across_many_frames() {
        let ids: std::collections::HashSet<String> = (0..100)
            .map(|_| SpawnMessage::new(SpawnMessageKind::Done, Value::Null).message_id)
            .collect();
        assert_eq!(ids.len(), 100);
    }

    #[test]
    fn every_kind_round_trips_through_a_line() {
        for kind in ALL_KINDS {
            let message = SpawnMessage::new(kind, json!({"kind": kind}));
            let line = message.to_line().unwrap();
            assert!(line.ends_with('\n'), "line is not terminated: {line:?}");
            assert_eq!(
                line.matches('\n').count(),
                1,
                "unexpected raw newline in {line:?}"
            );
            let restored = SpawnMessage::from_line(&line).unwrap();
            assert_same_frame(&restored, &message);
        }
    }

    #[test]
    fn from_line_trims_surrounding_whitespace() {
        let message = SpawnMessage::new(SpawnMessageKind::Output, json!({"answer": "hi"}));
        let line = message.to_line().unwrap();
        let padded = format!("  {line}\r\n");
        assert_same_frame(&SpawnMessage::from_line(&padded).unwrap(), &message);
        assert_same_frame(&SpawnMessage::from_line(line.trim_end()).unwrap(), &message);
    }

    #[test]
    fn payload_with_newlines_stays_on_one_line() {
        let message = SpawnMessage::new(
            SpawnMessageKind::StreamChunk,
            json!({"text": "line one\nline two\r\nline three"}),
        );
        let line = message.to_line().unwrap();
        assert!(line.ends_with('\n'));
        assert_eq!(
            line.matches('\n').count(),
            1,
            "payload newline leaked into the frame: {line:?}"
        );
        assert_eq!(line.matches('\r').count(), 0);
        let restored = SpawnMessage::from_line(&line).unwrap();
        assert_eq!(
            restored.payload,
            json!({"text": "line one\nline two\r\nline three"})
        );
        assert_same_frame(&restored, &message);
    }

    #[test]
    fn line_is_a_single_json_object_with_the_documented_fields() {
        let message = SpawnMessage::new(SpawnMessageKind::HealthCheck, Value::Null);
        let line = message.to_line().unwrap();
        let value: Value = serde_json::from_str(line.trim_end()).unwrap();
        assert_eq!(value["kind"], json!("health_check"));
        assert_eq!(value["payload"], Value::Null);
        assert!(value["timestamp"].is_f64() || value["timestamp"].is_u64());
        assert_eq!(value["message_id"], json!(message.message_id));
    }

    #[test]
    fn blank_lines_are_validation_errors() {
        for line in ["", "   ", "\t", "\r\n", " \n \n "] {
            let error = SpawnMessage::from_line(line).expect_err("blank line must be rejected");
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert!(
                error.to_string().contains("empty"),
                "unexpected message: {error}"
            );
        }
    }

    #[test]
    fn non_json_lines_are_validation_errors() {
        let error = SpawnMessage::from_line("not json at all").expect_err("must be rejected");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.to_string().contains("not json at all"),
            "unexpected message: {error}"
        );
    }

    #[test]
    fn non_object_json_lines_are_validation_errors() {
        for line in ["42", "\"text\"", "[1, 2]", "null"] {
            let error = SpawnMessage::from_line(line).expect_err("must be rejected");
            assert_eq!(error.kind(), ErrorKind::Validation, "for line {line:?}");
        }
    }

    #[test]
    fn missing_required_fields_are_validation_errors() {
        let cases = [
            json!({"kind": "input", "payload": null, "timestamp": 1.0}),
            json!({"payload": null, "timestamp": 1.0, "message_id": "id"}),
            json!({"kind": "input", "timestamp": 1.0, "message_id": "id"}),
            json!({"kind": "input", "payload": null, "message_id": "id"}),
        ];
        for case in cases {
            let error = SpawnMessage::from_line(&case.to_string())
                .expect_err("missing field must be rejected");
            assert_eq!(error.kind(), ErrorKind::Validation);
        }
    }

    #[test]
    fn unknown_kinds_are_validation_errors() {
        let line = json!({
            "kind": "teleport",
            "payload": null,
            "timestamp": 1.0,
            "message_id": "id",
        })
        .to_string();
        let error = SpawnMessage::from_line(&line).expect_err("unknown kind must be rejected");
        assert_eq!(error.kind(), ErrorKind::Validation);
    }

    #[test]
    fn error_messages_echo_at_most_a_short_prefix() {
        let long = "x".repeat(4_096);
        let error = SpawnMessage::from_line(&long).expect_err("must be rejected");
        let message = error.to_string();
        assert!(
            message.len() < 512,
            "error message echoes too much input: {} bytes",
            message.len()
        );
        assert!(message.contains('…'), "unexpected message: {message}");
    }

    #[test]
    fn preview_does_not_split_multibyte_characters() {
        let text = "é".repeat(ECHO_PREFIX_CHARS + 10);
        let shown = preview(&text);
        assert_eq!(shown.chars().count(), ECHO_PREFIX_CHARS + 1);
        assert!(shown.ends_with('…'));
    }
}
