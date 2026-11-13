//! Typed bodies and events for the Langfuse REST ingestion API.
//!
//! Port of the event vocabulary of the Langfuse Python SDK v2/v3 that
//! `openjiuwen.extensions.observability.exporters.langfuse` targets: the
//! `/api/public/ingestion` endpoint takes a `batch` of events, each carrying an
//! `id`, a `type` (`trace-create`, `span-create`, `generation-create`), an RFC
//! 3339 `timestamp`, and a type-specific `body`.
//!
//! Every body is `camelCase` on the wire (`userId`, `traceId`, `startTime`,
//! `usageDetails`, …), tolerates unknown fields on the way in, and omits absent
//! fields entirely instead of sending `null`.
//!
//! ```
//! use openjiuwen_extensions::otel::langfuse::{IngestionEvent, SpanBody, TraceBody};
//!
//! let trace = IngestionEvent::trace(
//!     &TraceBody::new("checkout").with_user_id("user-1").with_session_id("session-1"),
//! )?;
//! let span = IngestionEvent::span(
//!     &SpanBody::new("trace-1", "retrieve").with_parent_observation_id("obs-1"),
//! )?;
//! assert_eq!(trace.to_json()["type"], "trace-create");
//! assert_eq!(span.to_json()["body"]["parentObservationId"], "obs-1");
//! # Ok::<(), openjiuwen_common::error::Error>(())
//! ```

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::id::now_rfc3339;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use uuid::Uuid;

/// Kind of an ingestion event, serialised as the wire string.
///
/// The variants map to `trace-create`, `span-create`, and `generation-create`
/// — the three types Langfuse v2/v3 accept for observation data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum IngestionEventType {
    /// Creates (or upserts) a trace.
    TraceCreate,
    /// Creates a span observation inside a trace.
    SpanCreate,
    /// Creates a generation observation inside a trace.
    GenerationCreate,
}

impl IngestionEventType {
    /// The exact string Langfuse expects in the event's `type` field.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TraceCreate => "trace-create",
            Self::SpanCreate => "span-create",
            Self::GenerationCreate => "generation-create",
        }
    }
}

impl std::fmt::Display for IngestionEventType {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Body of a `trace-create` event.
///
/// Only [`Self::name`] is set by [`TraceBody::new`]; every other field is
/// omitted from the wire when it stays `None`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct TraceBody {
    /// Trace name shown in the Langfuse UI.
    pub name: String,
    /// End-user identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    /// Session the trace belongs to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Arbitrary input payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
    /// Arbitrary output payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<Value>,
    /// Arbitrary metadata object.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    /// Free-form tags.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,
    /// Release / deployment identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub release: Option<String>,
    /// Application version.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// RFC 3339 timestamp of the trace start.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
}

impl TraceBody {
    /// Builds a trace body naming the trace.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    /// Returns the body with an end-user identifier.
    #[must_use]
    pub fn with_user_id(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    /// Returns the body with a session identifier.
    #[must_use]
    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Returns the body with an input payload.
    #[must_use]
    pub fn with_input(mut self, input: Value) -> Self {
        self.input = Some(input);
        self
    }

    /// Returns the body with an output payload.
    #[must_use]
    pub fn with_output(mut self, output: Value) -> Self {
        self.output = Some(output);
        self
    }

    /// Returns the body with a metadata object.
    #[must_use]
    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Returns the body with tags.
    #[must_use]
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = Some(tags);
        self
    }

    /// Returns the body with a release identifier.
    #[must_use]
    pub fn with_release(mut self, release: impl Into<String>) -> Self {
        self.release = Some(release.into());
        self
    }

    /// Returns the body with an application version.
    #[must_use]
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    /// Returns the body with an explicit RFC 3339 start timestamp.
    #[must_use]
    pub fn with_timestamp(mut self, timestamp: impl Into<String>) -> Self {
        self.timestamp = Some(timestamp.into());
        self
    }
}

/// Body of a `span-create` event.
///
/// [`SpanBody::new`] sets the two required fields (`traceId`, `name`); absent
/// optional fields are omitted from the wire.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct SpanBody {
    /// Id of the trace the span belongs to.
    pub trace_id: String,
    /// Span name shown in the Langfuse UI.
    pub name: String,
    /// RFC 3339 span start.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_time: Option<String>,
    /// RFC 3339 span end.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_time: Option<String>,
    /// Arbitrary input payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
    /// Arbitrary output payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<Value>,
    /// Arbitrary metadata object.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    /// Severity level (`DEBUG`, `DEFAULT`, `WARNING`, `ERROR`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level: Option<String>,
    /// Human-readable status message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status_message: Option<String>,
    /// Id of the parent observation, when the span is nested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_observation_id: Option<String>,
}

impl SpanBody {
    /// Builds a span body for `trace_id` named `name`.
    #[must_use]
    pub fn new(trace_id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            trace_id: trace_id.into(),
            name: name.into(),
            ..Self::default()
        }
    }

    /// Returns the body with a start timestamp.
    #[must_use]
    pub fn with_start_time(mut self, start_time: impl Into<String>) -> Self {
        self.start_time = Some(start_time.into());
        self
    }

    /// Returns the body with an end timestamp.
    #[must_use]
    pub fn with_end_time(mut self, end_time: impl Into<String>) -> Self {
        self.end_time = Some(end_time.into());
        self
    }

    /// Returns the body with an input payload.
    #[must_use]
    pub fn with_input(mut self, input: Value) -> Self {
        self.input = Some(input);
        self
    }

    /// Returns the body with an output payload.
    #[must_use]
    pub fn with_output(mut self, output: Value) -> Self {
        self.output = Some(output);
        self
    }

    /// Returns the body with a metadata object.
    #[must_use]
    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Returns the body with a severity level.
    #[must_use]
    pub fn with_level(mut self, level: impl Into<String>) -> Self {
        self.level = Some(level.into());
        self
    }

    /// Returns the body with a status message.
    #[must_use]
    pub fn with_status_message(mut self, status_message: impl Into<String>) -> Self {
        self.status_message = Some(status_message.into());
        self
    }

    /// Returns the body with a parent observation id.
    #[must_use]
    pub fn with_parent_observation_id(mut self, parent_observation_id: impl Into<String>) -> Self {
        self.parent_observation_id = Some(parent_observation_id.into());
        self
    }
}

/// Body of a `generation-create` event.
///
/// A generation is a span that additionally records the model, its parameters,
/// token usage, and the moment the completion started streaming.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct GenerationBody {
    /// Id of the trace the generation belongs to.
    pub trace_id: String,
    /// Generation name shown in the Langfuse UI.
    pub name: String,
    /// RFC 3339 generation start.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_time: Option<String>,
    /// RFC 3339 generation end.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_time: Option<String>,
    /// RFC 3339 timestamp of the first completion token.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_start_time: Option<String>,
    /// Input payload (prompt or messages).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
    /// Output payload (completion or messages).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<Value>,
    /// Arbitrary metadata object.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    /// Severity level (`DEBUG`, `DEFAULT`, `WARNING`, `ERROR`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level: Option<String>,
    /// Human-readable status message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status_message: Option<String>,
    /// Id of the parent observation, when the generation is nested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_observation_id: Option<String>,
    /// Model identifier (`gpt-4o`, `claude-sonnet-4`, …).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Model parameters (`temperature`, `max_tokens`, …).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_parameters: Option<Value>,
    /// Token usage (`input`, `output`, `total`, `unit`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage_details: Option<Value>,
}

impl GenerationBody {
    /// Builds a generation body for `trace_id` named `name`.
    #[must_use]
    pub fn new(trace_id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            trace_id: trace_id.into(),
            name: name.into(),
            ..Self::default()
        }
    }

    /// Returns the body with a start timestamp.
    #[must_use]
    pub fn with_start_time(mut self, start_time: impl Into<String>) -> Self {
        self.start_time = Some(start_time.into());
        self
    }

    /// Returns the body with an end timestamp.
    #[must_use]
    pub fn with_end_time(mut self, end_time: impl Into<String>) -> Self {
        self.end_time = Some(end_time.into());
        self
    }

    /// Returns the body with the timestamp of the first completion token.
    #[must_use]
    pub fn with_completion_start_time(mut self, completion_start_time: impl Into<String>) -> Self {
        self.completion_start_time = Some(completion_start_time.into());
        self
    }

    /// Returns the body with an input payload.
    #[must_use]
    pub fn with_input(mut self, input: Value) -> Self {
        self.input = Some(input);
        self
    }

    /// Returns the body with an output payload.
    #[must_use]
    pub fn with_output(mut self, output: Value) -> Self {
        self.output = Some(output);
        self
    }

    /// Returns the body with a metadata object.
    #[must_use]
    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Returns the body with a severity level.
    #[must_use]
    pub fn with_level(mut self, level: impl Into<String>) -> Self {
        self.level = Some(level.into());
        self
    }

    /// Returns the body with a status message.
    #[must_use]
    pub fn with_status_message(mut self, status_message: impl Into<String>) -> Self {
        self.status_message = Some(status_message.into());
        self
    }

    /// Returns the body with a parent observation id.
    #[must_use]
    pub fn with_parent_observation_id(mut self, parent_observation_id: impl Into<String>) -> Self {
        self.parent_observation_id = Some(parent_observation_id.into());
        self
    }

    /// Returns the body with a model identifier.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Returns the body with model parameters.
    #[must_use]
    pub fn with_model_parameters(mut self, model_parameters: Value) -> Self {
        self.model_parameters = Some(model_parameters);
        self
    }

    /// Returns the body with token usage details.
    #[must_use]
    pub fn with_usage_details(mut self, usage_details: Value) -> Self {
        self.usage_details = Some(usage_details);
        self
    }
}

/// One event of a Langfuse ingestion batch.
///
/// The `id` and the RFC 3339 `timestamp` are minted by [`Self::new`] (and by the
/// three typed constructors) unless a caller overrides them with
/// [`Self::with_id`] / [`Self::with_timestamp`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IngestionEvent {
    /// Event id; Langfuse deduplicates on it, so it must be stable per event.
    pub id: String,
    /// Event kind, serialised as `type`.
    #[serde(rename = "type")]
    pub event_type: IngestionEventType,
    /// RFC 3339 UTC timestamp of the event.
    pub timestamp: String,
    /// Type-specific body.
    pub body: Value,
}

impl IngestionEvent {
    /// Builds an event of `event_type` with a fresh id and the current UTC
    /// timestamp.
    #[must_use]
    pub fn new(event_type: IngestionEventType, body: Value) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            event_type,
            timestamp: now_rfc3339(),
            body,
        }
    }

    /// Builds a `trace-create` event.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when the body cannot be serialised as JSON.
    pub fn trace(body: &TraceBody) -> Result<Self> {
        Self::from_body(IngestionEventType::TraceCreate, body)
    }

    /// Builds a `span-create` event.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when the body cannot be serialised as JSON.
    pub fn span(body: &SpanBody) -> Result<Self> {
        Self::from_body(IngestionEventType::SpanCreate, body)
    }

    /// Builds a `generation-create` event.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when the body cannot be serialised as JSON.
    pub fn generation(body: &GenerationBody) -> Result<Self> {
        Self::from_body(IngestionEventType::GenerationCreate, body)
    }

    /// Returns the event with a different id.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Returns the event with a different RFC 3339 timestamp.
    #[must_use]
    pub fn with_timestamp(mut self, timestamp: impl Into<String>) -> Self {
        self.timestamp = timestamp.into();
        self
    }

    /// Serialises the event into the wire shape
    /// `{"id": …, "type": …, "timestamp": …, "body": {…}}`.
    #[must_use]
    pub fn to_json(&self) -> Value {
        json!({
            "id": self.id,
            "type": self.event_type,
            "timestamp": self.timestamp,
            "body": self.body,
        })
    }

    /// Serialises `body` and wraps it into an event of `event_type`.
    fn from_body<T: Serialize>(event_type: IngestionEventType, body: &T) -> Result<Self> {
        let body = serde_json::to_value(body).map_err(|error| {
            Error::validation(format!(
                "the Langfuse {event_type} body is not serialisable as JSON: {error}"
            ))
        })?;
        Ok(Self::new(event_type, body))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn event_types_use_the_langfuse_wire_strings() {
        assert_eq!(IngestionEventType::TraceCreate.as_str(), "trace-create");
        assert_eq!(IngestionEventType::SpanCreate.as_str(), "span-create");
        assert_eq!(
            IngestionEventType::GenerationCreate.as_str(),
            "generation-create"
        );
        assert_eq!(
            serde_json::to_value(IngestionEventType::GenerationCreate).unwrap(),
            json!("generation-create"),
        );
        let decoded: IngestionEventType = serde_json::from_value(json!("span-create")).unwrap();
        assert_eq!(decoded, IngestionEventType::SpanCreate);
    }

    #[test]
    fn trace_event_carries_camel_case_body_fields_and_omits_absent_ones() {
        let event = IngestionEvent::trace(
            &TraceBody::new("checkout")
                .with_user_id("user-1")
                .with_session_id("session-1")
                .with_tags(vec!["prod".to_owned()]),
        )
        .unwrap();
        let wire = event.to_json();
        assert_eq!(wire["type"], "trace-create");
        assert_eq!(wire["body"]["name"], "checkout");
        assert_eq!(wire["body"]["userId"], "user-1");
        assert_eq!(wire["body"]["sessionId"], "session-1");
        assert_eq!(wire["body"]["tags"], json!(["prod"]));
        assert!(wire["body"].get("input").is_none(), "{wire}");
        assert!(wire["body"].get("release").is_none(), "{wire}");
        assert!(uuid::Uuid::parse_str(&event.id).is_ok());
        assert!(event.timestamp.ends_with('Z'), "{}", event.timestamp);
    }

    #[test]
    fn span_event_serialises_the_observation_fields() {
        let event = IngestionEvent::span(
            &SpanBody::new("trace-1", "retrieve")
                .with_start_time("2026-01-02T03:04:05.000Z")
                .with_level("WARNING")
                .with_parent_observation_id("obs-1"),
        )
        .unwrap();
        let wire = event.to_json();
        assert_eq!(wire["type"], "span-create");
        assert_eq!(wire["body"]["traceId"], "trace-1");
        assert_eq!(wire["body"]["name"], "retrieve");
        assert_eq!(wire["body"]["startTime"], "2026-01-02T03:04:05.000Z");
        assert_eq!(wire["body"]["level"], "WARNING");
        assert_eq!(wire["body"]["parentObservationId"], "obs-1");
        assert!(wire["body"].get("endTime").is_none(), "{wire}");
    }

    #[test]
    fn generation_event_serialises_model_and_usage() {
        let event = IngestionEvent::generation(
            &GenerationBody::new("trace-1", "chat")
                .with_model("gpt-4o")
                .with_model_parameters(json!({"temperature": 0.2}))
                .with_usage_details(json!({"input": 3, "output": 5}))
                .with_completion_start_time("2026-01-02T03:04:06.000Z"),
        )
        .unwrap();
        let wire = event.to_json();
        assert_eq!(wire["type"], "generation-create");
        assert_eq!(wire["body"]["traceId"], "trace-1");
        assert_eq!(wire["body"]["model"], "gpt-4o");
        assert_eq!(wire["body"]["modelParameters"]["temperature"], 0.2);
        assert_eq!(wire["body"]["usageDetails"]["output"], 5);
        assert_eq!(
            wire["body"]["completionStartTime"],
            "2026-01-02T03:04:06.000Z"
        );
    }

    #[test]
    fn event_ids_and_timestamps_can_be_overridden() {
        let event = IngestionEvent::new(IngestionEventType::TraceCreate, json!({}))
            .with_id("event-1")
            .with_timestamp("2026-01-02T03:04:05.000Z");
        let wire = event.to_json();
        assert_eq!(wire["id"], "event-1");
        assert_eq!(wire["timestamp"], "2026-01-02T03:04:05.000Z");
    }

    #[test]
    fn events_round_trip_through_serde() {
        let event = IngestionEvent::trace(&TraceBody::new("checkout")).unwrap();
        let encoded = serde_json::to_string(&event).unwrap();
        let decoded: IngestionEvent = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, event);
        assert_eq!(decoded.to_json()["type"], "trace-create");
    }
}
