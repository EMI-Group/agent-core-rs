//! One session-log line: the tolerant decoder, the text projection, and the JSONL encoder.
//!
//! Python's `core/memory/lite/manager.py` treats a session file as opaque text and never
//! parses it. This module goes one step further without giving up that tolerance: a line
//! that is a JSON object with a usable string content field is decoded into a
//! [`SessionRecord`](crate::lite::session::record::SessionRecord), and every other
//! non-blank line — plain text, a truncated or non-object JSON payload, a JSON object
//! without a string content — becomes a record whose content is the raw line verbatim.
//! Decoding therefore never fails and never loses a line.
use super::TARGET;
use chrono::{DateTime, SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
/// Keys read, in order, for the message text of a JSON line.
const CONTENT_FIELDS: [&str; 4] = ["content", "text", "message", "output"];
/// Keys read, in order, for the speaker of a JSON line.
const ROLE_FIELDS: [&str; 2] = ["role", "type"];
/// Keys read, in order, for the timestamp of a JSON line.
const TIMESTAMP_FIELDS: [&str; 3] = ["timestamp", "ts", "time"];
/// Numeric epoch values at or above this magnitude are milliseconds, not seconds.
///
/// `1e11` seconds is the year 5138, so no real session log carries a seconds value that
/// large, while `1e11` milliseconds is 1973 — the ambiguous band is empty in practice.
const MILLIS_THRESHOLD: f64 = 1e11;
/// The largest fractional-second value, in nanoseconds.
const NANOS_PER_SECOND: f64 = 1_000_000_000.0;
/// One decoded session-log line.
///
/// The content is always present (it is the raw line when the line does not decode), so a
/// record always renders as a citable chunk; [`SessionRecord::is_empty`] tells a
/// whitespace-only line apart from a real one.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SessionRecord {
    /// When the line was written, when the line records it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<DateTime<Utc>>,
    /// The speaker or event kind (`user`, `assistant`, `tool`, …), when the line records it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// The message text: the decoded content field, or the raw line on a tolerant fallback.
    pub content: String,
}
impl SessionRecord {
    /// Builds a record with `content` and no role or timestamp.
    #[must_use]
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            ..Self::default()
        }
    }
    /// Sets the speaker or event kind.
    #[must_use]
    pub fn with_role(self, role: impl Into<String>) -> Self {
        Self {
            role: Some(role.into()),
            ..self
        }
    }
    /// Sets the write time.
    #[must_use]
    pub fn with_timestamp(self, timestamp: DateTime<Utc>) -> Self {
        Self {
            timestamp: Some(timestamp),
            ..self
        }
    }
    /// Returns `true` when the content is blank (whitespace only).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.content.trim().is_empty()
    }
    /// Returns the prompt-facing text: `role: content` when a role is set, else the content.
    #[must_use]
    pub fn text(&self) -> String {
        match &self.role {
            Some(role) => format!("{role}: {}", self.content),
            None => self.content.clone(),
        }
    }
    /// Decodes one log line, tolerantly; `None` for a blank line.
    ///
    /// A JSON object is decoded when one of `content`, `text`, `message`, or `output` holds a
    /// string; the role comes from `role` or `type` and the timestamp from `timestamp`, `ts`,
    /// or `time` (an RFC 3339 string, or a number of seconds/milliseconds since the epoch).
    /// Optional fields that are missing or of another type are simply absent.
    ///
    /// Every other non-blank line keeps its text: a non-object or malformed JSON payload, a
    /// JSON object with no string content field, and plain text all become a record whose
    /// content is the line verbatim, with no role and no timestamp. Such a line is tolerated
    /// input rather than an error, and one `debug` event records the fallback.
    #[must_use]
    pub fn parse_line(line: &str) -> Option<Self> {
        if line.trim().is_empty() {
            return None;
        }
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            return Some(Self::fallback(line, "line is not a JSON value"));
        };
        let Value::Object(map) = value else {
            return Some(Self::fallback(line, "line is JSON but not an object"));
        };
        let Some(content) = string_field(&map, &CONTENT_FIELDS) else {
            return Some(Self::fallback(line, "line has no string content field"));
        };
        Some(Self {
            timestamp: timestamp_field(&map),
            role: string_field(&map, &ROLE_FIELDS).map(str::to_owned),
            content: content.to_owned(),
        })
    }
    /// Encodes the record as one JSON line, without a trailing newline.
    ///
    /// The object carries `content`, then `role`, then `timestamp` (RFC 3339 in UTC with
    /// nanosecond precision); an absent role or timestamp is omitted. The result is escaped
    /// by the serializer, so embedded quotes and newlines stay on one line, and
    /// [`SessionRecord::parse_line`] decodes it back into an equal record.
    #[must_use]
    pub fn to_json_line(&self) -> String {
        let mut map = Map::new();
        map.insert("content".to_owned(), Value::String(self.content.clone()));
        if let Some(role) = &self.role {
            map.insert("role".to_owned(), Value::String(role.clone()));
        }
        if let Some(timestamp) = self.timestamp {
            let rendered = timestamp.to_rfc3339_opts(SecondsFormat::Nanos, true);
            map.insert("timestamp".to_owned(), Value::String(rendered));
        }
        Value::Object(map).to_string()
    }
    /// Builds the tolerant raw-text fallback for `line`.
    fn fallback(line: &str, reason: &str) -> Self {
        tracing::debug!(
            target: TARGET,
            reason,
            "session line kept as plain text"
        );
        Self::new(line)
    }
}
/// Returns the first string value of `map` among `keys`.
fn string_field<'a>(map: &'a Map<String, Value>, keys: &[&str]) -> Option<&'a str> {
    keys.iter()
        .find_map(|key| map.get(*key).and_then(Value::as_str))
}
/// Returns the first decodable timestamp value of `map`.
fn timestamp_field(map: &Map<String, Value>) -> Option<DateTime<Utc>> {
    TIMESTAMP_FIELDS
        .iter()
        .find_map(|key| map.get(*key).and_then(decode_timestamp))
}
/// Decodes an RFC 3339 string or a numeric epoch value into a UTC timestamp.
fn decode_timestamp(value: &Value) -> Option<DateTime<Utc>> {
    match value {
        Value::String(text) => DateTime::parse_from_rfc3339(text.trim())
            .ok()
            .map(|parsed| parsed.with_timezone(&Utc)),
        Value::Number(number) => match number.as_i64() {
            Some(whole) => from_epoch(whole as f64),
            None => number.as_f64().and_then(from_epoch),
        },
        _ => None,
    }
}
/// Decodes an epoch value as seconds or milliseconds.
fn from_epoch(value: f64) -> Option<DateTime<Utc>> {
    if !value.is_finite() {
        return None;
    }
    if value.abs() >= MILLIS_THRESHOLD {
        return DateTime::from_timestamp_millis(value as i64);
    }
    let seconds = value.trunc() as i64;
    let nanos = (value.fract() * NANOS_PER_SECOND).round();
    let nanos = if nanos.is_finite() {
        nanos.clamp(0.0, NANOS_PER_SECOND - 1.0) as u32
    } else {
        0
    };
    DateTime::from_timestamp(seconds, nanos)
}
#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    /// Builds a UTC timestamp from a date and clock time.
    fn stamp(second: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 11, 14, 9, 5, second).unwrap()
    }
    #[test]
    fn round_trips_through_the_json_line() {
        let records = [
            SessionRecord::new("hello"),
            SessionRecord::new("quoted \"text\"\nand a newline").with_role("user"),
            SessionRecord::new("turn")
                .with_role("assistant")
                .with_timestamp(stamp(0)),
            SessionRecord::new("")
                .with_role("tool")
                .with_timestamp(Utc.timestamp_millis_opt(1_700_000_000_123).unwrap()),
        ];
        for record in records {
            let line = record.to_json_line();
            assert!(!line.contains('\n'), "one line only: {line}");
            assert_eq!(
                SessionRecord::parse_line(&line),
                Some(record.clone()),
                "{line}"
            );
        }
    }
    #[test]
    fn encodes_content_first_and_omits_absent_options() {
        assert_eq!(
            SessionRecord::new("hi").to_json_line(),
            r#"{"content":"hi"}"#
        );
        let full = SessionRecord::new("hi")
            .with_role("user")
            .with_timestamp(stamp(0))
            .to_json_line();
        assert!(
            full.starts_with(r#"{"content":"hi","role":"user","timestamp":"#),
            "{full}"
        );
    }
    #[test]
    fn decodes_the_content_field_aliases() {
        for key in ["content", "text", "message", "output"] {
            let line = format!(r#"{{"{key}":"body"}}"#);
            let record = SessionRecord::parse_line(&line).unwrap();
            assert_eq!(record.content, "body", "{key}");
        }
        let record = SessionRecord::parse_line(r#"{"text":"body","role":"user"}"#).unwrap();
        assert_eq!(record.role.as_deref(), Some("user"));
    }
    #[test]
    fn decodes_the_role_field_aliases() {
        assert_eq!(
            SessionRecord::parse_line(r#"{"content":"x","role":"user"}"#)
                .unwrap()
                .role
                .as_deref(),
            Some("user")
        );
        assert_eq!(
            SessionRecord::parse_line(r#"{"content":"x","type":"assistant"}"#)
                .unwrap()
                .role
                .as_deref(),
            Some("assistant")
        );
    }
    #[test]
    fn decodes_rfc_3339_timestamp_strings() {
        let record =
            SessionRecord::parse_line(r#"{"content":"x","timestamp":"2026-11-14T09:05:00Z"}"#)
                .unwrap();
        assert_eq!(record.timestamp, Some(stamp(0)));
        let offset =
            SessionRecord::parse_line(r#"{"content":"x","time":"2026-11-14T17:05:00+08:00"}"#)
                .unwrap();
        assert_eq!(offset.timestamp, Some(stamp(0)));
        let fractional =
            SessionRecord::parse_line(r#"{"content":"x","ts":"2026-11-14T09:05:00.500Z"}"#)
                .unwrap();
        assert_eq!(
            fractional.timestamp,
            Some(stamp(0) + chrono::Duration::milliseconds(500))
        );
    }
    #[test]
    fn decodes_epoch_seconds_and_milliseconds() {
        let seconds =
            SessionRecord::parse_line(r#"{"content":"x","timestamp":1700000000}"#).unwrap();
        assert_eq!(
            seconds.timestamp,
            Some(Utc.timestamp_opt(1_700_000_000, 0).unwrap())
        );
        let millis =
            SessionRecord::parse_line(r#"{"content":"x","timestamp":1700000000000}"#).unwrap();
        assert_eq!(
            millis.timestamp,
            Some(Utc.timestamp_millis_opt(1_700_000_000_000).unwrap())
        );
        let fractional = SessionRecord::parse_line(r#"{"content":"x","ts":1700000000.5}"#).unwrap();
        assert_eq!(
            fractional.timestamp,
            Some(Utc.timestamp_millis_opt(1_700_000_000_500).unwrap())
        );
    }
    #[test]
    fn ignores_unparsable_timestamps() {
        for line in [
            r#"{"content":"x","timestamp":"not a time"}"#,
            r#"{"content":"x","timestamp":{}}"#,
            r#"{"content":"x","timestamp":true}"#,
            r#"{"content":"x","timestamp":null}"#,
        ] {
            assert_eq!(
                SessionRecord::parse_line(line).unwrap().timestamp,
                None,
                "{line}"
            );
        }
    }
    #[test]
    fn tolerates_malformed_and_non_json_lines() {
        for line in [
            r#"{"content":"truncated""#,
            r#"{"content":"x""#,
            "[1, 2, 3]",
            r#""just a string""#,
            "42",
            "null",
            "plain text turn",
            "  leading and trailing spaces  ",
        ] {
            let record = SessionRecord::parse_line(line).unwrap();
            assert_eq!(record.content, line, "{line}");
            assert_eq!(record.role, None, "{line}");
            assert_eq!(record.timestamp, None, "{line}");
        }
    }
    #[test]
    fn tolerates_wrong_typed_content_fields() {
        for line in [r#"{"content":42}"#, r#"{"text":null,"content":["x"]}"#] {
            let record = SessionRecord::parse_line(line).unwrap();
            assert_eq!(record.content, line, "{line}");
        }
        let record =
            SessionRecord::parse_line(r#"{"content":"x","role":7,"timestamp":[]}"#).unwrap();
        assert_eq!(record.content, "x");
        assert_eq!(record.role, None);
        assert_eq!(record.timestamp, None);
    }
    #[test]
    fn blank_lines_decode_to_nothing() {
        for line in ["", "   ", "\t", "\r"] {
            assert_eq!(SessionRecord::parse_line(line), None, "{line:?}");
        }
    }
    #[test]
    fn text_prefixes_only_a_present_role() {
        assert_eq!(SessionRecord::new("hello").text(), "hello");
        assert_eq!(
            SessionRecord::new("hello").with_role("user").text(),
            "user: hello"
        );
    }
    #[test]
    fn emptiness_is_whitespace_based() {
        assert!(SessionRecord::new("").is_empty());
        assert!(SessionRecord::new("  \n\t").is_empty());
        assert!(!SessionRecord::new("x").is_empty());
        assert!(!SessionRecord::new(" x ").is_empty());
    }
    #[test]
    fn serde_round_trips_through_json() {
        let record = SessionRecord::new("body")
            .with_role("user")
            .with_timestamp(stamp(1));
        let json = serde_json::to_value(&record).unwrap();
        assert_eq!(json["content"], serde_json::json!("body"));
        assert_eq!(json["role"], serde_json::json!("user"));
        let decoded: SessionRecord = serde_json::from_value(json).unwrap();
        assert_eq!(decoded, record);
        let empty: SessionRecord = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(empty, SessionRecord::default());
    }
}
