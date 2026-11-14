//! The session-history parser: one [`TraceRecord`] per `request_id` segment.
//!
//! The input format is an agent-runtime session export:
//!
//! ```text
//! <root>/sessions/<session_id>/metadata.json   // must be a JSON object; validity gate
//! <root>/sessions/<session_id>/history.json    // a JSON array of flat event objects
//! ```
//!
//! Events carry `request_id`, `role`, `event_type`, `content`, and — on tool events —
//! `tool_call`, `tool_name`, `result`, `raw_output`. A change of `request_id` closes the
//! current record and opens the next one, exactly like Python's
//! `experience/trace/parser.py`.
//!
//! # Tolerance rules
//!
//! A trace is evidence, not user input, so the parser is deliberately forgiving and never
//! aborts the scan:
//!
//! - A missing session directory or metadata file yields no records; malformed metadata
//!   is an error the store catches and turns into a warning for that one session.
//! - A missing or malformed `history.json` is treated as an empty history (with a
//!   warning), matching Python's `_read_json` returning `None`.
//! - Non-object history entries, unknown roles, and unknown `event_type` values are
//!   dropped silently.
//! - Tool payloads may be a JSON object, a JSON string, or a Python-style single-quoted
//!   literal (`"{'skill_name': 'translator'}"`). The parser tries `serde_json` first and
//!   then a small lenient literal reader; a payload it cannot decode is dropped
//!   ([`serde_json::Value::Null`]) instead of failing the record. Python reaches for
//!   `ast.literal_eval` here, which has no Rust equivalent.
//!
//! `chat.tool_call` / `chat.tool_result` / `chat.error` are also accepted without the
//! `chat.` prefix, because the Python source only reads the prefixed names while some
//! exports flatten them.
use serde_json::Value;
use tracing::warn;

use crate::error::{Error, Result};
use crate::experience::trace::models::{TraceRecord, TraceToolCall};
use crate::experience::trace::store::SessionPaths;

/// The event-type prefixes recognized on assistant events.
const CHAT_PREFIX: &str = "chat.";

/// Reads `path` as JSON, answering `None` when the file does not exist.
async fn read_json(path: &std::path::Path) -> Result<Option<Value>> {
    match tokio::fs::read_to_string(path).await {
        Ok(text) => serde_json::from_str(&text)
            .map(Some)
            .map_err(Error::external),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(Error::external(error)),
    }
}

/// Returns the session directory names under `<root>/sessions`, sorted.
///
/// Missing session roots answer an empty list; `heartbeat_*` directories are skipped,
/// matching Python's `list_session_ids`.
pub(crate) async fn list_session_ids(paths: &SessionPaths) -> Result<Vec<String>> {
    let sessions_dir = paths.sessions_dir();
    let mut entries = match tokio::fs::read_dir(&sessions_dir).await {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(Error::external(error)),
    };
    let mut ids = Vec::new();
    while let Some(entry) = entries.next_entry().await.map_err(Error::external)? {
        let file_type = entry.file_type().await.map_err(Error::external)?;
        if !file_type.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("heartbeat_") {
            continue;
        }
        ids.push(name);
    }
    ids.sort();
    Ok(ids)
}

/// Parses one session directory into records, one per `request_id` segment.
pub(crate) async fn parse_session(
    session_id: &str,
    paths: &SessionPaths,
) -> Result<Vec<TraceRecord>> {
    let session_dir = paths.session_dir(session_id);
    let Some(metadata) = read_json(&session_dir.join("metadata.json")).await? else {
        return Ok(Vec::new());
    };
    if !metadata.is_object() {
        return Ok(Vec::new());
    }
    let metadata_failed = metadata_marks_failure(&metadata);

    let history_path = session_dir.join("history.json");
    let history = match read_json(&history_path).await {
        Ok(history) => history,
        Err(error) => {
            warn!(
                session_id = %session_id,
                %error,
                "ignoring unreadable session history"
            );
            None
        }
    };
    let events = history
        .as_ref()
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    let mut records = Vec::new();
    let mut segment = Segment::new(String::new());
    for event in &events {
        if !event.is_object() {
            continue;
        }
        let request_id = request_id_of(event);
        if !request_id.is_empty() && request_id != segment.request_id {
            if let Some(record) = segment.finish(session_id, metadata_failed) {
                records.push(record);
            }
            segment = Segment::new(request_id);
        }
        match record_str(event, "role") {
            "user" => {
                let text = extract_text(event.get("content")).trim().to_string();
                if !text.is_empty() && segment.query.is_empty() {
                    segment.query = text;
                }
            }
            "assistant" => fold_assistant_event(event, &mut segment),
            _ => {}
        }
        segment.observe_metrics(event);
    }
    if let Some(record) = segment.finish(session_id, metadata_failed) {
        records.push(record);
    }
    Ok(records)
}

/// Sorts records by `(timestamp, trace_id)` so a scan is deterministic.
pub(crate) fn sort_records(records: &mut [TraceRecord]) {
    records.sort_by(|left, right| {
        left.timestamp
            .total_cmp(&right.timestamp)
            .then_with(|| left.trace_id.cmp(&right.trace_id))
    });
}

/// The accumulation state of one `request_id` segment.
#[derive(Debug)]
struct Segment {
    /// The segment's request id; blank before the first identified event.
    request_id: String,
    /// The first non-blank user message.
    query: String,
    /// The last `chat.final` payload.
    final_text: String,
    /// The concatenated `chat.delta` payloads.
    delta_text: String,
    /// The extracted skill ids, deduped in first-seen order.
    skills: Vec<String>,
    /// The tool calls of the segment.
    tool_calls: Vec<TraceToolCall>,
    /// The error messages of the segment.
    errors: Vec<String>,
    /// Whether an error event was observed.
    saw_error: bool,
    /// The largest reported duration in milliseconds.
    duration_ms: u64,
    /// The first numeric event timestamp in epoch seconds.
    timestamp: Option<f64>,
}

impl Segment {
    /// Starts an empty segment for `request_id`.
    fn new(request_id: String) -> Self {
        Self {
            request_id,
            query: String::new(),
            final_text: String::new(),
            delta_text: String::new(),
            skills: Vec::new(),
            tool_calls: Vec::new(),
            errors: Vec::new(),
            saw_error: false,
            duration_ms: 0,
            timestamp: None,
        }
    }

    /// Observes the first timestamp and the largest duration of an event.
    fn observe_metrics(&mut self, event: &Value) {
        if self.timestamp.is_none() {
            if let Some(timestamp) = numeric(event, "timestamp") {
                self.timestamp = Some(normalize_timestamp(timestamp));
            }
        }
        if let Some(milliseconds) = numeric(event, "duration_ms") {
            self.duration_ms = self.duration_ms.max(milliseconds_of(milliseconds));
        } else if let Some(seconds) = numeric(event, "duration") {
            self.duration_ms = self.duration_ms.max(milliseconds_of(seconds * 1000.0));
        }
    }

    /// Finishes the segment, answering `None` when it has no request id or no query.
    fn finish(self, session_id: &str, metadata_failed: bool) -> Option<TraceRecord> {
        if self.request_id.is_empty() || self.query.trim().is_empty() {
            return None;
        }
        let mut skills = self.skills;
        skills.sort();
        skills.dedup();
        let final_output = if self.final_text.trim().is_empty() {
            self.delta_text.trim().to_string()
        } else {
            self.final_text.trim().to_string()
        };
        Some(TraceRecord {
            trace_id: format!("{session_id}/{}", self.request_id),
            session_id: session_id.to_string(),
            request_id: self.request_id,
            query: self.query.trim().to_string(),
            final_output,
            skills,
            success: !self.saw_error && !metadata_failed,
            duration_ms: self.duration_ms,
            tool_calls: self.tool_calls,
            errors: self.errors,
            timestamp: self.timestamp.unwrap_or(0.0),
        })
    }

    /// Adds each skill that is not already present.
    fn note_skills(&mut self, skills: impl IntoIterator<Item = String>) {
        for skill in skills {
            let skill = skill.trim();
            if !skill.is_empty() && !self.skills.iter().any(|known| known == skill) {
                self.skills.push(skill.to_string());
            }
        }
    }
}

/// Applies one assistant event to the segment.
fn fold_assistant_event(event: &Value, segment: &mut Segment) {
    let event_type = record_str(event, "event_type");
    let normalized = event_type.strip_prefix(CHAT_PREFIX).unwrap_or(event_type);
    match normalized {
        "delta" => segment
            .delta_text
            .push_str(extract_text(event.get("content")).trim()),
        "final" => {
            segment.final_text = extract_text(event.get("content")).trim().to_string();
        }
        "tool_call" => fold_tool_call(event, segment),
        "tool_result" => fold_tool_result(event, segment),
        "error" => fold_error(event, segment),
        _ => {}
    }
}

/// Records a tool call and any skills its arguments select.
fn fold_tool_call(event: &Value, segment: &mut Segment) {
    let call = match event.get("tool_call") {
        Some(Value::String(_)) => event
            .get("tool_call")
            .and_then(parse_payload)
            .unwrap_or(Value::Null),
        Some(Value::Object(_)) => event.get("tool_call").cloned().unwrap_or(Value::Null),
        _ => event.clone(),
    };
    let skills = skills_from_tool_call(&call, event);
    segment.note_skills(skills);

    let name = {
        let name = record_str(&call, "name");
        if name.is_empty() {
            let name = record_str(&call, "tool_name");
            if name.is_empty() {
                record_str(event, "tool_name").to_string()
            } else {
                name.to_string()
            }
        } else {
            name.to_string()
        }
    };
    let arguments = call
        .get("arguments")
        .and_then(parse_payload)
        .unwrap_or(Value::Null);
    let result = call
        .get("result")
        .map(|value| extract_text(Some(value)).trim().to_string())
        .filter(|text| !text.is_empty());
    if name.is_empty() && arguments.is_null() && result.is_none() {
        return;
    }
    segment.tool_calls.push(TraceToolCall {
        name,
        arguments,
        result,
    });
}

/// Attaches a tool result to the open call (or records it as its own call).
fn fold_tool_result(event: &Value, segment: &mut Segment) {
    let tool_name = record_str(event, "tool_name").to_string();
    let text = extract_text(event.get("result")).trim().to_string();
    let target = segment.tool_calls.iter_mut().rev().find(|call| {
        call.result.is_none()
            && (tool_name.is_empty() || call.name.is_empty() || call.name == tool_name)
    });
    match target {
        Some(call) => {
            if call.name.is_empty() {
                call.name = tool_name;
            }
            if !text.is_empty() {
                call.result = Some(text);
            }
        }
        None => {
            if !tool_name.is_empty() || !text.is_empty() {
                segment.tool_calls.push(TraceToolCall {
                    name: tool_name,
                    arguments: Value::Null,
                    result: (!text.is_empty()).then_some(text),
                });
            }
        }
    }
    segment.note_skills(skills_from_tool_result(event));
}

/// Records an error event and marks the segment as failed.
fn fold_error(event: &Value, segment: &mut Segment) {
    segment.saw_error = true;
    let mut message = extract_text(event.get("content")).trim().to_string();
    if message.is_empty() {
        message = extract_text(event.get("error")).trim().to_string();
    }
    if message.is_empty() {
        message = record_str(event, "message").trim().to_string();
    }
    if message.is_empty() {
        message = "unknown error".to_string();
    }
    segment.errors.push(message);
}

/// Extracts skill ids from a tool call.
///
/// Recognized tools are `skill_tool` (`arguments.skill_name`) and
/// `symphony_compose_graph` (`arguments.candidate_skill_ids`, then the plan steps of the
/// call's `raw_output` or decoded `result`).
fn skills_from_tool_call(call: &Value, event: &Value) -> Vec<String> {
    let name = {
        let name = record_str(call, "name");
        if name.is_empty() {
            record_str(event, "tool_name")
        } else {
            name
        }
    };
    let mut skills = Vec::new();
    let arguments = call.get("arguments").and_then(parse_payload);
    match name {
        "skill_tool" => {
            if let Some(skill) = arguments
                .as_ref()
                .and_then(|arguments| arguments.get("skill_name"))
                .and_then(Value::as_str)
            {
                push_skill(&mut skills, skill);
            }
        }
        "symphony_compose_graph" => {
            if let Some(candidate_ids) = arguments
                .as_ref()
                .and_then(|arguments| arguments.get("candidate_skill_ids"))
                .and_then(Value::as_array)
            {
                for candidate in candidate_ids {
                    if let Some(skill) = candidate.as_str() {
                        push_skill(&mut skills, skill);
                    }
                }
            }
            if skills.is_empty() {
                let raw_output = call
                    .get("raw_output")
                    .or_else(|| event.get("raw_output"))
                    .cloned()
                    .or_else(|| call.get("result").and_then(parse_payload));
                if let Some(raw_output) = raw_output {
                    for skill in skills_from_raw_output(&raw_output) {
                        push_skill(&mut skills, &skill);
                    }
                }
            }
        }
        _ => {}
    }
    skills
}

/// Extracts skill ids from a `symphony_compose_graph` tool result.
fn skills_from_tool_result(event: &Value) -> Vec<String> {
    if record_str(event, "tool_name") != "symphony_compose_graph" {
        return Vec::new();
    }
    let raw_output = event
        .get("raw_output")
        .cloned()
        .or_else(|| event.get("result").and_then(parse_payload));
    let mut skills = Vec::new();
    if let Some(raw_output) = raw_output {
        for skill in skills_from_raw_output(&raw_output) {
            push_skill(&mut skills, &skill);
        }
    }
    skills
}

/// Extracts `raw_output.plan.steps[].skill_id` (falling back to `name`).
fn skills_from_raw_output(raw_output: &Value) -> Vec<String> {
    let mut skills = Vec::new();
    let steps = raw_output
        .get("plan")
        .and_then(|plan| plan.get("steps"))
        .and_then(Value::as_array);
    if let Some(steps) = steps {
        for step in steps {
            let skill = step
                .get("skill_id")
                .and_then(Value::as_str)
                .filter(|skill| !skill.trim().is_empty())
                .or_else(|| {
                    step.get("name")
                        .and_then(Value::as_str)
                        .filter(|skill| !skill.trim().is_empty())
                });
            if let Some(skill) = skill {
                push_skill(&mut skills, skill);
            }
        }
    }
    skills
}

/// Pushes `skill` onto `skills` when it is non-blank and not already present.
fn push_skill(skills: &mut Vec<String>, skill: &str) {
    let skill = skill.trim();
    if !skill.is_empty() && !skills.iter().any(|known| known == skill) {
        skills.push(skill.to_string());
    }
}

/// Extracts plain text from a content payload.
///
/// `None`/`null` is blank, a string is itself, and an array concatenates its string
/// entries plus the `text` field of `{"type": "text"}` blocks. Any other JSON value is
/// rendered compactly, which mirrors Python's `str(content)` fallback.
fn extract_text(content: Option<&Value>) -> String {
    match content {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(blocks)) => {
            let mut text = String::new();
            for block in blocks {
                match block {
                    Value::String(part) => text.push_str(part),
                    Value::Object(map)
                        if map.get("type").and_then(Value::as_str) == Some("text") =>
                    {
                        if let Some(part) = map.get("text").and_then(Value::as_str) {
                            text.push_str(part);
                        }
                    }
                    _ => {}
                }
            }
            text
        }
        Some(other) => other.to_string(),
    }
}

/// Returns `record[key]` as a string, or `""` when it is absent or not a string.
fn record_str<'a>(record: &'a Value, key: &str) -> &'a str {
    record.get(key).and_then(Value::as_str).unwrap_or("")
}

/// Returns `record[key]` as a finite number.
fn numeric(record: &Value, key: &str) -> Option<f64> {
    record
        .get(key)
        .and_then(Value::as_f64)
        .filter(|v| v.is_finite())
}

/// Reads the request id of an event, accepting a string or a number.
fn request_id_of(event: &Value) -> String {
    match event.get("request_id") {
        Some(Value::String(id)) => id.trim().to_string(),
        Some(Value::Number(id)) => id.to_string(),
        _ => String::new(),
    }
}

/// Normalizes an epoch timestamp: values that look like milliseconds (`>= 1e12`) become
/// seconds, everything else is kept.
fn normalize_timestamp(timestamp: f64) -> f64 {
    if timestamp >= 1.0e12 {
        timestamp / 1000.0
    } else {
        timestamp
    }
}

/// Rounds a non-negative duration in milliseconds to a `u64`.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn milliseconds_of(value: f64) -> u64 {
    if value.is_finite() && value > 0.0 {
        value.round() as u64
    } else {
        0
    }
}

/// Reports whether the session metadata marks the session as failed.
///
/// Recognized markers are a `success: false` boolean, a failed `status`/`state`/`result`
/// string, and a non-empty `error`, `error_type`, or `error_detail` member.
fn metadata_marks_failure(metadata: &Value) -> bool {
    let Some(map) = metadata.as_object() else {
        return false;
    };
    if map.get("success") == Some(&Value::Bool(false)) {
        return true;
    }
    for key in ["status", "state"] {
        if let Some(Value::String(state)) = map.get(key) {
            let state = state.trim().to_ascii_lowercase();
            if matches!(
                state.as_str(),
                "failed" | "failure" | "error" | "errored" | "cancelled" | "canceled"
            ) {
                return true;
            }
        }
    }
    for key in ["error", "error_type", "error_detail"] {
        match map.get(key) {
            Some(Value::String(text)) if !text.trim().is_empty() => return true,
            Some(Value::Array(entries)) if !entries.is_empty() => return true,
            Some(Value::Object(entries)) if !entries.is_empty() => return true,
            _ => {}
        }
    }
    false
}

/// Decodes a tool payload: an object/array is taken as-is, a string is parsed as JSON and
/// then, failing that, as a Python-style single-quoted literal.
///
/// Returns `None` when the payload cannot be decoded; callers drop it rather than fail
/// the record.
fn parse_payload(value: &Value) -> Option<Value> {
    match value {
        Value::Object(_) | Value::Array(_) => Some(value.clone()),
        Value::String(text) => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                return None;
            }
            serde_json::from_str::<Value>(trimmed)
                .ok()
                .or_else(|| LenientLiteral::parse(trimmed))
        }
        _ => None,
    }
}

/// A tiny recursive-descent reader for Python-style literals
/// (`{'skill_name': 'translator'}`, `[1, True, None]`).
///
/// It supports single- and double-quoted strings, objects, arrays, integers, floats,
/// `true`/`false`/`null`, and Python's `True`/`False`/`None`. Anything else — including
/// bytes literals, sets, and unicode escapes — makes the whole payload decode to `None`,
/// which is the documented drop instead of guessing.
struct LenientLiteral {
    /// The payload as characters, so multi-byte text survives.
    chars: Vec<char>,
    /// The index of the next unconsumed character.
    position: usize,
}

impl LenientLiteral {
    /// Parses `text` as one literal, requiring it to be consumed completely.
    fn parse(text: &str) -> Option<Value> {
        let mut parser = Self {
            chars: text.chars().collect(),
            position: 0,
        };
        let value = parser.value()?;
        parser.skip_whitespace();
        (parser.position == parser.chars.len()).then_some(value)
    }

    /// Returns the next character without consuming it.
    fn peek(&self) -> Option<char> {
        self.chars.get(self.position).copied()
    }

    /// Consumes the next character.
    fn next(&mut self) -> Option<char> {
        let character = self.peek()?;
        self.position += 1;
        Some(character)
    }

    /// Skips whitespace.
    fn skip_whitespace(&mut self) {
        while self.peek().is_some_and(char::is_whitespace) {
            self.position += 1;
        }
    }

    /// Parses one value.
    fn value(&mut self) -> Option<Value> {
        self.skip_whitespace();
        match self.peek()? {
            '{' => self.object(),
            '[' => self.array(),
            '\'' | '"' => self.string().map(Value::String),
            'T' => self.keyword("True", Value::Bool(true)),
            'F' => self.keyword("False", Value::Bool(false)),
            'N' => self.keyword("None", Value::Null),
            't' => self.keyword("true", Value::Bool(true)),
            'f' => self.keyword("false", Value::Bool(false)),
            'n' => self.keyword("null", Value::Null),
            _ => self.number(),
        }
    }

    /// Parses `{key: value, ...}`.
    fn object(&mut self) -> Option<Value> {
        self.next()?;
        let mut map = serde_json::Map::new();
        self.skip_whitespace();
        if self.peek()? == '}' {
            self.next();
            return Some(Value::Object(map));
        }
        loop {
            self.skip_whitespace();
            let key = self.string()?;
            self.skip_whitespace();
            if self.next()? != ':' {
                return None;
            }
            let value = self.value()?;
            map.insert(key, value);
            self.skip_whitespace();
            match self.next()? {
                ',' => {}
                '}' => return Some(Value::Object(map)),
                _ => return None,
            }
        }
    }

    /// Parses `[value, ...]`.
    fn array(&mut self) -> Option<Value> {
        self.next()?;
        let mut values = Vec::new();
        self.skip_whitespace();
        if self.peek()? == ']' {
            self.next();
            return Some(Value::Array(values));
        }
        loop {
            values.push(self.value()?);
            self.skip_whitespace();
            match self.next()? {
                ',' => {}
                ']' => return Some(Value::Array(values)),
                _ => return None,
            }
        }
    }

    /// Parses a single- or double-quoted string with the common backslash escapes.
    fn string(&mut self) -> Option<String> {
        let quote = self.next()?;
        if quote != '\'' && quote != '"' {
            return None;
        }
        let mut text = String::new();
        loop {
            match self.next()? {
                character if character == quote => return Some(text),
                '\\' => match self.next()? {
                    'n' => text.push('\n'),
                    'r' => text.push('\r'),
                    't' => text.push('\t'),
                    'b' => text.push('\u{0008}'),
                    'f' => text.push('\u{000c}'),
                    'u' => return None,
                    other => text.push(other),
                },
                character => text.push(character),
            }
        }
    }

    /// Consumes `literal` (or answers `None`).
    fn keyword(&mut self, literal: &str, value: Value) -> Option<Value> {
        let consumed: String = self.chars[self.position..]
            .iter()
            .take(literal.chars().count())
            .collect();
        if consumed == literal {
            self.position += literal.chars().count();
            Some(value)
        } else {
            None
        }
    }

    /// Parses an integer or float.
    fn number(&mut self) -> Option<Value> {
        let start = self.position;
        while self
            .peek()
            .is_some_and(|character| character.is_ascii_digit() || "+-.eE".contains(character))
        {
            self.position += 1;
        }
        if self.position == start {
            return None;
        }
        let text: String = self.chars[start..self.position].iter().collect();
        if let Ok(integer) = text.parse::<i64>() {
            return Some(Value::Number(integer.into()));
        }
        text.parse::<f64>()
            .ok()
            .and_then(serde_json::Number::from_f64)
            .map(Value::Number)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn lenient_literals_parse_python_style_payloads() {
        let parsed = parse_payload(&json!("{'skill_name': 'translator', 'count': 2}")).unwrap();
        assert_eq!(parsed["skill_name"], "translator");
        assert_eq!(parsed["count"], 2);

        let parsed = parse_payload(&json!("[{'name': \"a\"}, True, None]")).unwrap();
        assert_eq!(parsed[0]["name"], "a");
        assert_eq!(parsed[1], true);
        assert!(parsed[2].is_null());
    }

    #[test]
    fn plain_json_wins_over_the_lenient_reader() {
        let parsed = parse_payload(&json!(r#"{"candidate_skill_ids": ["a", "b"]}"#)).unwrap();
        assert_eq!(parsed["candidate_skill_ids"][1], "b");
    }

    #[test]
    fn undecodable_payloads_are_dropped() {
        assert!(parse_payload(&json!("not a literal")).is_none());
        assert!(parse_payload(&json!("{'a': b'bytes'}")).is_none());
        assert!(parse_payload(&Value::Null).is_none());
    }

    #[test]
    fn text_extraction_matches_the_python_reader() {
        assert_eq!(extract_text(None), "");
        assert_eq!(extract_text(Some(&json!("hello"))), "hello");
        assert_eq!(
            extract_text(Some(
                &json!(["a", {"type": "text", "text": "b"}, {"type": "image"}])
            )),
            "ab"
        );
        assert_eq!(extract_text(Some(&json!(42))), "42");
    }

    #[test]
    fn metadata_failure_markers_are_recognized() {
        assert!(metadata_marks_failure(&json!({"success": false})));
        assert!(metadata_marks_failure(&json!({"status": "FAILED"})));
        assert!(metadata_marks_failure(
            &json!({"error_type": "RuntimeError"})
        ));
        assert!(!metadata_marks_failure(&json!({"success": true})));
        assert!(!metadata_marks_failure(&json!({"error": ""})));
        assert!(!metadata_marks_failure(&json!("not an object")));
    }

    #[test]
    fn millisecond_timestamps_are_converted_to_seconds() {
        assert_eq!(normalize_timestamp(1_700_000_000.0), 1_700_000_000.0);
        assert_eq!(normalize_timestamp(1_700_000_000_000.0), 1_700_000_000.0);
    }
}
