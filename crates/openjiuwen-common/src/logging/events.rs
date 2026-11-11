//! Structured log events.
//!
//! Port of Python `openjiuwen/core/common/logging/events.py` (the documented
//! subset the Rust SDK needs): a [`LogEvent`] is a self-describing record with an
//! event type, module, status, correlation ids, and free-form metadata. It
//! serializes to the same `snake_case` JSON shape as Python's
//! `BaseLogEvent.model_dump()`, drops unset fields, and is redacted before it is
//! handed to a log sink.
//!
//! Unlike Python, the record is not routed through `LogManager`: [`emit`] sends it
//! straight to `tracing`, where the configured subscriber decides what to do with
//! it.
#![allow(clippy::module_name_repetitions)]
use super::scope;
use super::targets;
use crate::security::redact::{redact_json, DEFAULT_SENSITIVE_FIELDS};
use crate::utils::id::{format_iso8601, now_utc, uuid_v4};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::fmt;

/// Severity of a [`LogEvent`] (Python `LogLevel`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum LogLevel {
    /// Unrecoverable failure; the operation cannot continue.
    Critical,
    /// Recoverable failure of the current operation.
    Error,
    /// Degraded behavior that needs attention.
    Warning,
    /// Normal progress information (default).
    #[default]
    Info,
    /// Developer diagnostics.
    Debug,
    /// Level not set; maps to no `tracing` level.
    #[serde(rename = "notset")]
    NotSet,
}

impl LogLevel {
    /// Returns the Python-compatible lowercase name.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Critical => "critical",
            Self::Error => "error",
            Self::Warning => "warning",
            Self::Info => "info",
            Self::Debug => "debug",
            Self::NotSet => "notset",
        }
    }
    /// Maps to a `tracing` level; [`LogLevel::NotSet`] has none.
    #[must_use]
    pub fn to_tracing_level(self) -> Option<tracing::Level> {
        match self {
            Self::Critical | Self::Error => Some(tracing::Level::ERROR),
            Self::Warning => Some(tracing::Level::WARN),
            Self::Info => Some(tracing::Level::INFO),
            Self::Debug => Some(tracing::Level::DEBUG),
            Self::NotSet => None,
        }
    }
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<tracing::Level> for LogLevel {
    /// Maps a `tracing` level onto the closest Python level.
    ///
    /// `TRACE` has no Python counterpart, so it is recorded as `debug`.
    fn from(level: tracing::Level) -> Self {
        if level == tracing::Level::ERROR {
            Self::Error
        } else if level == tracing::Level::WARN {
            Self::Warning
        } else if level == tracing::Level::INFO {
            Self::Info
        } else {
            Self::Debug
        }
    }
}

/// Subsystem that produced a [`LogEvent`] (Python `ModuleType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ModuleType {
    /// Agent runtime.
    Agent,
    /// Workflow engine.
    Workflow,
    /// Single workflow component/node.
    WorkflowComponent,
    /// LLM client.
    Llm,
    /// Tool invocation.
    Tool,
    /// Storage backend.
    Store,
    /// Memory subsystem.
    Memory,
    /// Session subsystem.
    Session,
    /// Context engine.
    Context,
    /// Retrieval subsystem.
    Retrieval,
    /// Internal system event (default).
    #[default]
    System,
    /// User-originated event.
    User,
    /// Filesystem/shell/code operations.
    SysOperation,
}

impl ModuleType {
    /// Returns the Python-compatible `snake_case` name.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Agent => "agent",
            Self::Workflow => "workflow",
            Self::WorkflowComponent => "workflow_component",
            Self::Llm => "llm",
            Self::Tool => "tool",
            Self::Store => "store",
            Self::Memory => "memory",
            Self::Session => "session",
            Self::Context => "context",
            Self::Retrieval => "retrieval",
            Self::System => "system",
            Self::User => "user",
            Self::SysOperation => "sys_operation",
        }
    }
}

impl fmt::Display for ModuleType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Outcome recorded on a [`LogEvent`] (Python `EventStatus`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum EventStatus {
    /// The operation completed successfully (default).
    #[default]
    Success,
    /// The operation failed.
    Failure,
    /// The operation has not finished yet.
    Pending,
    /// The operation exceeded its deadline.
    Timeout,
    /// The operation was cancelled.
    Cancelled,
}

impl EventStatus {
    /// Returns the Python-compatible `snake_case` name.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Failure => "failure",
            Self::Pending => "pending",
            Self::Timeout => "timeout",
            Self::Cancelled => "cancelled",
        }
    }
}

impl fmt::Display for EventStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Well-known event names, expressed as a string newtype.
///
/// A newtype (rather than an enum) keeps the type open: applications and
/// plugins can emit their own names without a breaking change.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EventType(String);

impl EventType {
    /// Creates an event type from any string, without normalization.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
    /// Returns the event name.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for EventType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for EventType {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl From<String> for EventType {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl Default for EventType {
    fn default() -> Self {
        Self(event_types::SYSTEM_START.to_owned())
    }
}

/// Well-known [`EventType`] names.
///
/// Values are lowercase `snake_case`, matching the Python event constants; use
/// [`from_name`] when the name comes from input.
pub mod event_types {
    use super::EventType;
    /// The SDK/agent process started.
    pub const SYSTEM_START: &str = "system_start";
    /// The SDK/agent process is shutting down.
    pub const SYSTEM_END: &str = "system_end";
    /// An agent invocation started.
    pub const AGENT_INVOKE_START: &str = "agent_invoke_start";
    /// An agent invocation finished.
    pub const AGENT_INVOKE_END: &str = "agent_invoke_end";
    /// A workflow run started.
    pub const WORKFLOW_START: &str = "workflow_start";
    /// A workflow run finished.
    pub const WORKFLOW_END: &str = "workflow_end";
    /// A workflow node started.
    pub const NODE_START: &str = "node_start";
    /// A workflow node finished.
    pub const NODE_END: &str = "node_end";
    /// An LLM request was sent.
    pub const LLM_REQUEST: &str = "llm_request";
    /// An LLM response was received.
    pub const LLM_RESPONSE: &str = "llm_response";
    /// A tool call started.
    pub const TOOL_CALL_START: &str = "tool_call_start";
    /// A tool call finished.
    pub const TOOL_CALL_END: &str = "tool_call_end";
    /// A session was created.
    pub const SESSION_CREATE: &str = "session_create";
    /// A session was resumed.
    pub const SESSION_RESUME: &str = "session_resume";
    /// A memory entry was stored.
    pub const MEMORY_STORE: &str = "memory_store";
    /// Memories were retrieved.
    pub const MEMORY_RETRIEVE: &str = "memory_retrieve";
    /// The context window was compressed.
    pub const CONTEXT_COMPRESS: &str = "context_compress";
    /// A retrieval query ran.
    pub const RETRIEVAL_QUERY: &str = "retrieval_query";
    /// A streaming chunk was emitted.
    pub const STREAM_CHUNK: &str = "stream_chunk";
    /// A task's status changed.
    pub const TASK_STATUS_CHANGED: &str = "task_status_changed";

    /// Normalizes `name` to lowercase `snake_case` and wraps it in an [`EventType`].
    ///
    /// Whitespace, hyphens, and other separators become `_`; camelCase word
    /// boundaries are split, so `"AgentInvokeStart"`, `"agent-invoke-start"`, and
    /// `"AGENT INVOKE START"` all produce `"agent_invoke_start"`.
    #[must_use]
    pub fn from_name(name: &str) -> EventType {
        EventType(super::normalize_name(name))
    }
}

/// Normalizes an event name to lowercase `snake_case`.
fn normalize_name(name: &str) -> String {
    let chars: Vec<char> = name.trim().chars().collect();
    let mut out = String::with_capacity(name.len());
    for (index, ch) in chars.iter().copied().enumerate() {
        if ch.is_alphanumeric() {
            if ch.is_uppercase() {
                let previous_is_lower = index
                    .checked_sub(1)
                    .and_then(|previous| chars.get(previous))
                    .is_some_and(|previous| previous.is_lowercase() || previous.is_ascii_digit());
                let next_is_lower = chars.get(index + 1).is_some_and(|next| next.is_lowercase());
                if (previous_is_lower || (next_is_lower && !out.is_empty())) && !out.ends_with('_')
                {
                    out.push('_');
                }
            }
            out.extend(ch.to_lowercase());
        } else if !out.is_empty() && !out.ends_with('_') {
            out.push('_');
        }
    }
    out.trim_end_matches('_').to_owned()
}

/// Normalizes a free-form event name to lowercase `snake_case` and wraps it.
///
/// Shorthand for [`event_types::from_name`].
#[must_use]
pub fn from_name(name: &str) -> EventType {
    event_types::from_name(name)
}

/// A structured log record (Python `BaseLogEvent`).
///
/// Fields describing absent values stay `None` and are dropped by
/// [`LogEvent::to_dict`]; `metadata` is always emitted, even when empty.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct LogEvent {
    /// Unique event id (UUID v4).
    pub event_id: String,
    /// Well-known or application-defined event name.
    pub event_type: EventType,
    /// Event severity.
    pub log_level: LogLevel,
    /// Time the event was created (UTC).
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Subsystem that produced the event.
    pub module_type: ModuleType,
    /// Instance id of the producing module.
    pub module_id: Option<String>,
    /// Human-readable module name.
    pub module_name: Option<String>,
    /// Session the event belongs to.
    pub session_id: Option<String>,
    /// Conversation the event belongs to.
    pub conversation_id: Option<String>,
    /// Trace id shared by one logical request.
    pub trace_id: Option<String>,
    /// Id correlating related events across modules.
    pub correlation_id: Option<String>,
    /// Event that caused this one.
    pub parent_event_id: Option<String>,
    /// Outcome of the operation.
    pub status: EventStatus,
    /// Python-compatible error code (`"-1"` for attached exceptions).
    pub error_code: Option<String>,
    /// Human-readable error message.
    pub error_message: Option<String>,
    /// Free-form message.
    pub message: Option<String>,
    /// Stack trace, when available.
    pub stacktrace: Option<String>,
    /// Stringified exception that caused the event.
    pub exception: Option<String>,
    /// Additional structured fields (redacted before logging).
    pub metadata: BTreeMap<String, Value>,
}

impl LogEvent {
    /// Creates an `info` event of `event_type` in the `system` module.
    #[must_use]
    pub fn new(event_type: impl Into<EventType>) -> Self {
        Self {
            event_id: uuid_v4(),
            event_type: event_type.into(),
            log_level: LogLevel::default(),
            timestamp: now_utc(),
            module_type: ModuleType::default(),
            module_id: None,
            module_name: None,
            session_id: None,
            conversation_id: None,
            trace_id: None,
            correlation_id: None,
            parent_event_id: None,
            status: EventStatus::default(),
            error_code: None,
            error_message: None,
            message: None,
            stacktrace: None,
            exception: None,
            metadata: BTreeMap::new(),
        }
    }
    /// Sets the severity.
    #[must_use]
    pub fn with_log_level(mut self, log_level: LogLevel) -> Self {
        self.log_level = log_level;
        self
    }
    /// Sets the producing subsystem.
    #[must_use]
    pub fn with_module_type(mut self, module_type: ModuleType) -> Self {
        self.module_type = module_type;
        self
    }
    /// Sets the module id and name.
    #[must_use]
    pub fn with_module(mut self, id: impl Into<String>, name: impl Into<String>) -> Self {
        self.module_id = Some(id.into());
        self.module_name = Some(name.into());
        self
    }
    /// Sets the session id.
    #[must_use]
    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }
    /// Sets the conversation id.
    #[must_use]
    pub fn with_conversation_id(mut self, conversation_id: impl Into<String>) -> Self {
        self.conversation_id = Some(conversation_id.into());
        self
    }
    /// Sets the trace id explicitly.
    #[must_use]
    pub fn with_trace_id(mut self, trace_id: impl Into<String>) -> Self {
        self.trace_id = Some(trace_id.into());
        self
    }
    /// Sets the trace id from the active [`LogScope`](super::scope::LogScope).
    #[must_use]
    pub fn with_trace_id_from_scope(mut self) -> Self {
        self.trace_id = Some(scope::current_trace_id());
        self
    }
    /// Sets the correlation id.
    #[must_use]
    pub fn with_correlation_id(mut self, correlation_id: impl Into<String>) -> Self {
        self.correlation_id = Some(correlation_id.into());
        self
    }
    /// Sets the parent event id.
    #[must_use]
    pub fn with_parent_event_id(mut self, parent_event_id: impl Into<String>) -> Self {
        self.parent_event_id = Some(parent_event_id.into());
        self
    }
    /// Sets the outcome status.
    #[must_use]
    pub fn with_status(mut self, status: EventStatus) -> Self {
        self.status = status;
        self
    }
    /// Sets the free-form message.
    #[must_use]
    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.message = Some(message.into());
        self
    }
    /// Sets the error code.
    #[must_use]
    pub fn with_error_code(mut self, error_code: impl Into<String>) -> Self {
        self.error_code = Some(error_code.into());
        self
    }
    /// Sets the error message.
    #[must_use]
    pub fn with_error_message(mut self, error_message: impl Into<String>) -> Self {
        self.error_message = Some(error_message.into());
        self
    }
    /// Sets the stack trace.
    #[must_use]
    pub fn with_stacktrace(mut self, stacktrace: impl Into<String>) -> Self {
        self.stacktrace = Some(stacktrace.into());
        self
    }
    /// Adds one metadata entry.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
    /// Attaches an exception, filling `error_code`/`error_message` when unset.
    ///
    /// Python parity: an attached exception implies `error_code = "-1"` (a
    /// string) and reuses the exception text as the error message.
    #[must_use]
    pub fn with_exception(mut self, exception: impl fmt::Display) -> Self {
        let text = exception.to_string();
        if self.error_code.is_none() {
            self.error_code = Some("-1".to_owned());
        }
        if self.error_message.is_none() {
            self.error_message = Some(text.clone());
        }
        self.exception = Some(text);
        self
    }
    /// Serializes the event to a JSON object, dropping unset fields.
    ///
    /// Enums are written as their Python string values, the timestamp is
    /// ISO-8601, and an attached exception without an explicit error code is
    /// reported as `error_code = "-1"` / `error_message = exception`.
    #[must_use]
    pub fn to_dict(&self) -> Map<String, Value> {
        let mut map = Map::with_capacity(12);
        map.insert("event_id".to_owned(), Value::String(self.event_id.clone()));
        map.insert(
            "event_type".to_owned(),
            Value::String(self.event_type.as_str().to_owned()),
        );
        map.insert(
            "log_level".to_owned(),
            Value::String(self.log_level.as_str().to_owned()),
        );
        map.insert(
            "timestamp".to_owned(),
            Value::String(format_iso8601(&self.timestamp)),
        );
        map.insert(
            "module_type".to_owned(),
            Value::String(self.module_type.as_str().to_owned()),
        );
        let optional = [
            ("module_id", &self.module_id),
            ("module_name", &self.module_name),
            ("session_id", &self.session_id),
            ("conversation_id", &self.conversation_id),
            ("trace_id", &self.trace_id),
            ("correlation_id", &self.correlation_id),
            ("parent_event_id", &self.parent_event_id),
            ("stacktrace", &self.stacktrace),
            ("exception", &self.exception),
        ];
        for (key, value) in optional {
            if let Some(value) = value {
                map.insert(key.to_owned(), Value::String(value.clone()));
            }
        }
        map.insert(
            "status".to_owned(),
            Value::String(self.status.as_str().to_owned()),
        );
        let error_code = self
            .error_code
            .clone()
            .or_else(|| self.exception.as_ref().map(|_| "-1".to_owned()));
        if let Some(error_code) = error_code {
            map.insert("error_code".to_owned(), Value::String(error_code));
        }
        let error_message = self
            .error_message
            .clone()
            .or_else(|| self.exception.clone());
        if let Some(error_message) = error_message {
            map.insert("error_message".to_owned(), Value::String(error_message));
        }
        if let Some(message) = &self.message {
            map.insert("message".to_owned(), Value::String(message.clone()));
        }
        map.insert(
            "metadata".to_owned(),
            Value::Object(self.metadata.clone().into_iter().collect()),
        );
        map
    }
    /// Serializes the event to a JSON object value.
    #[must_use]
    pub fn to_value(&self) -> Value {
        Value::Object(self.to_dict())
    }
    /// Serializes the event and redacts sensitive fields.
    ///
    /// `None` uses [`DEFAULT_SENSITIVE_FIELDS`]; the redaction walk covers nested
    /// metadata objects and arrays.
    #[must_use]
    pub fn redacted(&self, sensitive: Option<&[&str]>) -> Value {
        redact_json(
            &self.to_value(),
            Some(sensitive.unwrap_or(DEFAULT_SENSITIVE_FIELDS)),
        )
    }
}

impl Default for LogEvent {
    /// Creates a default `system_start` event with a fresh id and timestamp.
    fn default() -> Self {
        Self::new(EventType::default())
    }
}

/// Emits `event` to `tracing` under the [`targets::EVENTS`] target.
///
/// The record is redacted with the default sensitive-field list, serialized to
/// JSON, and attached as an `event` field next to the severity, event type,
/// module type, status, and message. Emission never panics: missing optional data
/// degrades to empty strings and a serialization failure drops only the JSON
/// payload.
pub fn emit(event: &LogEvent) {
    let level = event
        .log_level
        .to_tracing_level()
        .unwrap_or(tracing::Level::INFO);
    // Serialization of an owned event cannot fail in practice; if it ever did,
    // the remaining structured fields would still be emitted below.
    let event_json = serde_json::to_string(&event.redacted(None)).unwrap_or_default();
    let event_type = event.event_type.to_string();
    let module_type = event.module_type.to_string();
    let status = event.status.to_string();
    let message = event.message.as_deref().unwrap_or("");
    // `tracing` needs a compile-time level for each callsite, so the dynamic
    // level is dispatched to one of the fixed levels.
    match level {
        tracing::Level::ERROR => tracing::event!(
            target: targets::EVENTS, tracing::Level::ERROR,
            event_type = %event_type, module_type = %module_type, status = %status,
            message = %message, event = %event_json,
        ),
        tracing::Level::WARN => tracing::event!(
            target: targets::EVENTS, tracing::Level::WARN,
            event_type = %event_type, module_type = %module_type, status = %status,
            message = %message, event = %event_json,
        ),
        tracing::Level::DEBUG => tracing::event!(
            target: targets::EVENTS, tracing::Level::DEBUG,
            event_type = %event_type, module_type = %module_type, status = %status,
            message = %message, event = %event_json,
        ),
        tracing::Level::TRACE => tracing::event!(
            target: targets::EVENTS, tracing::Level::TRACE,
            event_type = %event_type, module_type = %module_type, status = %status,
            message = %message, event = %event_json,
        ),
        _ => tracing::event!(
            target: targets::EVENTS, tracing::Level::INFO,
            event_type = %event_type, module_type = %module_type, status = %status,
            message = %message, event = %event_json,
        ),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    // port of tests/unit_tests/core/common/log/test_structured_log.py
    #[test]
    fn new_event_has_python_defaults() {
        let event = LogEvent::new(event_types::AGENT_INVOKE_START);
        assert_eq!(event.event_type.as_str(), "agent_invoke_start");
        assert_eq!(event.log_level, LogLevel::Info);
        assert_eq!(event.module_type, ModuleType::System);
        assert_eq!(event.status, EventStatus::Success);
        assert_eq!(event.event_id.len(), 36, "hyphenated UUID v4");
        assert!(event.session_id.is_none());
    }
    #[test]
    fn to_dict_drops_unset_fields() {
        let event = LogEvent::new(event_types::LLM_REQUEST)
            .with_module("client-1", "LlmClient")
            .with_message("hello")
            .with_session_id("session-1");
        let dict = event.to_dict();
        assert!(dict.contains_key("event_id"));
        assert!(dict.contains_key("event_type"));
        assert!(dict.contains_key("log_level"));
        assert!(dict.contains_key("timestamp"));
        assert!(dict.contains_key("module_type"));
        assert!(dict.contains_key("status"));
        assert!(dict.contains_key("metadata"));
        assert_eq!(dict.get("module_id").unwrap(), "client-1");
        assert_eq!(dict.get("module_name").unwrap(), "LlmClient");
        assert_eq!(dict.get("session_id").unwrap(), "session-1");
        assert_eq!(dict.get("message").unwrap(), "hello");
        for absent in [
            "conversation_id",
            "trace_id",
            "correlation_id",
            "parent_event_id",
            "error_code",
            "error_message",
            "stacktrace",
            "exception",
        ] {
            assert!(!dict.contains_key(absent), "{absent} must be dropped");
        }
        assert_eq!(dict.get("metadata").unwrap(), &serde_json::json!({}));
    }
    #[test]
    fn to_dict_uses_python_enum_values() {
        let event = LogEvent::new(event_types::NODE_END)
            .with_log_level(LogLevel::Critical)
            .with_module_type(ModuleType::WorkflowComponent)
            .with_status(EventStatus::Timeout);
        let dict = event.to_dict();
        assert_eq!(dict.get("log_level").unwrap(), "critical");
        assert_eq!(dict.get("module_type").unwrap(), "workflow_component");
        assert_eq!(dict.get("status").unwrap(), "timeout");
        assert_eq!(dict.get("event_type").unwrap(), "node_end");
    }
    #[test]
    fn exception_sets_string_minus_one_error_code() {
        let event = LogEvent::new(event_types::TOOL_CALL_END).with_exception("boom");
        assert_eq!(event.error_code.as_deref(), Some("-1"));
        assert_eq!(event.error_message.as_deref(), Some("boom"));
        let dict = event.to_dict();
        assert_eq!(dict.get("error_code").unwrap(), "-1");
        assert_eq!(dict.get("exception").unwrap(), "boom");
        assert_eq!(dict.get("error_message").unwrap(), "boom");
    }
    #[test]
    fn explicit_error_code_wins_over_exception() {
        let event = LogEvent::new(event_types::TOOL_CALL_END)
            .with_error_code("E42")
            .with_error_message("tool failed")
            .with_exception("boom");
        let dict = event.to_dict();
        assert_eq!(dict.get("error_code").unwrap(), "E42");
        assert_eq!(dict.get("error_message").unwrap(), "tool failed");
    }
    #[test]
    fn default_redaction_masks_sensitive_metadata() {
        let event = LogEvent::new(event_types::LLM_REQUEST).with_metadata(
            "payload",
            serde_json::json!({
                "messages": [{"role": "user", "content": "secret"}],
                "query": "find me",
                "model": "gpt",
            }),
        );
        let redacted = event.redacted(None);
        assert_eq!(redacted["metadata"]["payload"]["messages"], "<REDACTED>");
        assert_eq!(redacted["metadata"]["payload"]["query"], "<REDACTED>");
        assert_eq!(redacted["metadata"]["payload"]["model"], "gpt");
    }
    #[test]
    fn explicit_redaction_fields_override_the_default_list() {
        let event = LogEvent::new(event_types::LLM_REQUEST)
            .with_metadata("model", Value::String("gpt".to_owned()))
            .with_metadata("query", Value::String("find me".to_owned()));
        let redacted = event.redacted(Some(&["model"]));
        assert_eq!(redacted["metadata"]["model"], "<REDACTED>");
        assert_eq!(redacted["metadata"]["query"], "find me");
        assert_eq!(DEFAULT_SENSITIVE_FIELDS.first(), Some(&"messages"));
    }
    #[test]
    fn event_type_from_name_normalizes_to_snake_case() {
        assert_eq!(
            event_types::from_name("AgentInvokeStart").as_str(),
            event_types::AGENT_INVOKE_START
        );
        assert_eq!(
            event_types::from_name("  LLM_REQUEST ").as_str(),
            event_types::LLM_REQUEST
        );
        assert_eq!(
            event_types::from_name("Tool-Call End").as_str(),
            event_types::TOOL_CALL_END
        );
        assert_eq!(
            event_types::from_name("LLMResponse").as_str(),
            "llm_response"
        );
        assert_eq!(event_types::from_name("").as_str(), "");
        assert_eq!(from_name("node_start").as_str(), "node_start");
    }
    #[test]
    fn event_type_defaults_to_system_start() {
        assert_eq!(EventType::default().as_str(), event_types::SYSTEM_START);
        assert_eq!(EventType::from("custom").as_str(), "custom");
        assert_eq!(EventType::from(String::from("custom2")).as_str(), "custom2");
        assert_eq!(EventType::new("custom3").as_str(), "custom3");
    }
    #[test]
    fn log_event_round_trips_through_json() {
        let event = LogEvent::new(event_types::MEMORY_STORE)
            .with_log_level(LogLevel::Warning)
            .with_module_type(ModuleType::Memory)
            .with_module("mem-1", "Memory")
            .with_session_id("session")
            .with_trace_id("trace")
            .with_status(EventStatus::Failure)
            .with_message("partial")
            .with_metadata("count", serde_json::json!(3));
        let json = serde_json::to_string(&event).unwrap();
        let restored: LogEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.event_id, event.event_id);
        assert_eq!(restored.event_type, event.event_type);
        assert_eq!(restored.log_level, LogLevel::Warning);
        assert_eq!(restored.module_type, ModuleType::Memory);
        assert_eq!(restored.status, EventStatus::Failure);
        assert_eq!(restored.metadata.get("count"), Some(&serde_json::json!(3)));
        assert_eq!(restored.timestamp, event.timestamp);
    }
    #[test]
    fn deserialization_tolerates_missing_fields() {
        let event: LogEvent = serde_json::from_str(r#"{"message":"only"}"#).unwrap();
        assert_eq!(event.message.as_deref(), Some("only"));
        assert_eq!(event.event_type, EventType::default());
        assert_eq!(event.log_level, LogLevel::Info);
        assert!(event.metadata.is_empty());
    }
    #[test]
    fn trace_id_can_be_taken_from_the_scope() {
        let event = LogEvent::new(event_types::SYSTEM_START).with_trace_id_from_scope();
        assert_eq!(
            event.trace_id.as_deref(),
            Some(super::super::scope::DEFAULT_TRACE_ID)
        );
    }
    #[test]
    fn emit_does_not_panic_without_a_subscriber() {
        let event = LogEvent::new(event_types::SYSTEM_START)
            .with_log_level(LogLevel::NotSet)
            .with_metadata("nested", serde_json::json!({"query": "secret"}));
        emit(&event);
    }
    #[test]
    fn log_level_maps_to_tracing_levels() {
        assert_eq!(
            LogLevel::Critical.to_tracing_level(),
            Some(tracing::Level::ERROR)
        );
        assert_eq!(
            LogLevel::Warning.to_tracing_level(),
            Some(tracing::Level::WARN)
        );
        assert_eq!(
            LogLevel::Info.to_tracing_level(),
            Some(tracing::Level::INFO)
        );
        assert_eq!(
            LogLevel::Debug.to_tracing_level(),
            Some(tracing::Level::DEBUG)
        );
        assert_eq!(LogLevel::NotSet.to_tracing_level(), None);
        assert_eq!(LogLevel::from(tracing::Level::TRACE), LogLevel::Debug);
        assert_eq!(LogLevel::from(tracing::Level::ERROR), LogLevel::Error);
        assert_eq!(LogLevel::default().as_str(), "info");
        assert_eq!(ModuleType::default().as_str(), "system");
        assert_eq!(EventStatus::default().as_str(), "success");
        assert_eq!(ModuleType::SysOperation.to_string(), "sys_operation");
        assert_eq!(EventStatus::Cancelled.to_string(), "cancelled");
    }
}
