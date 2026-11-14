//! Crate-boundary proof of the **team-memory → OTLP** bridge, in both OTLP/HTTP
//! encodings.
//!
//! Python counterpart: `openjiuwen/extensions/tracer_otel/` plus the team-memory
//! observer seam of `openjiuwen.memory.team`. The producer side of this bridge
//! lives one layer up, in `openjiuwen-agent-teams`
//! (`src/observability/otlp.rs`: `MemorySpanObserver` / `OtlpMemoryObserver`),
//! which turns every `TeamMemoryEvent` into ONE component [`TraceFrame`] and
//! hands it to [`OtelTraceSink`]. That crate cannot be a dependency of this one
//! (it depends on `openjiuwen-extensions`, so the edge would be a cycle), so
//! this suite **reproduces the bridge's frame shape** — the same fields, the
//! same derived ids, the same `meta_data` bag — from the public
//! `openjiuwen_session::tracer` types, and proves the receiving side turns each
//! frame into exactly the span the bridge documents.
//!
//! The mirrored frame shape (`agent-teams/src/observability/otlp.rs`, `span_for`):
//!
//! | Field | Value |
//! | --- | --- |
//! | `name` | `memory.{operation}` |
//! | `kind` | [`TraceSpanKind::Component`], wrapped by [`TraceFrame::workflow`] |
//! | `invoke_id` | `memory.{operation}.{16 hex digits}` derived from team, operation, member |
//! | `trace_id` | 32 lowercase hex characters derived from the team id |
//! | `status` | [`SpanStatus::Finish`] for an `ok` outcome, [`SpanStatus::Error`] otherwise |
//! | `error` | the event's error text when it has one |
//! | `component_id` / `component_name` / `component_type` | the invoke id / `memory.{operation}` / `"MemoryOperation"` |
//! | `meta_data` | `team_id`, `operation`, `outcome`, `count` plus optional `member_id`, `kind`, `error` |
//!
//! What the file pins, per encoding:
//!
//! - a batch of three frames reaches the collector as exactly one
//!   `POST /v1/traces` with the encoding's content type;
//! - the body is a valid OTLP `ExportTraceServiceRequest` carrying three spans,
//!   one per event, each named `component.{invoke_id}` (the workflow handler's
//!   generic component branch, because `"MemoryOperation"` matches none of its
//!   `LLM`/`Tool`/`Retrieval` classifiers);
//! - every span carries the component identity, the team trace id in
//!   `openjiuwen.trace.id`, the `openjiuwen.status` vocabulary, and the whole
//!   team-memory attribute bag inside `openjiuwen.meta_data`;
//! - the failed frame exports `STATUS_CODE_ERROR` with its error text, while the
//!   successful ones export `STATUS_CODE_OK`;
//! - the producer seam stays infallible: a shut-down handle yields a
//!   non-recording sink that drops frames, an exporter-less handle never posts,
//!   and a collector answering `500` leaves `emit` panic-free.
//!
//! Everything is offline and deterministic: the only peer is a loopback
//! `wiremock` collector, ids are pure functions of the fixture events, the
//! configured endpoint/headers/protocol/service name all beat the environment,
//! and every await is bounded.

#![cfg(feature = "otel")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::{json, Map, Value};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

use openjiuwen_extensions::otel::semconv::{
    OJ_ERROR, OJ_META_DATA, OJ_STATUS, OJ_TRACE_ID, OJ_WORKFLOW_COMPONENT_ID,
    OJ_WORKFLOW_COMPONENT_NAME, OJ_WORKFLOW_COMPONENT_TYPE, OJ_WORKFLOW_ERROR_MESSAGE,
    SPAN_STATUS_ERROR, SPAN_STATUS_FINISH,
};
use openjiuwen_extensions::otel::{
    init_otel_tracer, ExporterType, OtelTraceSink, OtelTracerConfig, OtlpProtocol, TracerHandle,
};
use openjiuwen_session::tracer::{SpanStatus, TraceFrame, TraceSink, TraceSpan, TraceSpanKind};

use support::{any_string, bytes, decode, fixed64, key_values, messages, string, varint, Fields};

// ---------------------------------------------------------------------------
// The producer's vocabulary, copied from the agent-teams bridge
// ---------------------------------------------------------------------------

/// Team scope the fixture events ran for.
const TEAM_ID: &str = "team-otlp-1";

/// Member the search event was scoped to.
const MEMBER_ID: &str = "alice";

/// Record kind the write event touched.
const RECORD_KIND: &str = "decision";

/// Error text of the failed event.
const FAILURE_TEXT: &str = "team memory delete failed: storage unavailable";

/// `openjiuwen.workflow.component.type` value of every forwarded span.
///
/// Deliberately free of the extensions component classifiers' substrings
/// (`LLM`, `Tool`, `Retrieval`), which is what selects the generic
/// `component.{invoke_id}` span name.
const MEMORY_COMPONENT_TYPE: &str = "MemoryOperation";

/// `service.name` the tests configure.
const SERVICE_NAME: &str = "openjiuwen-team-memory-tests";

/// The team-memory attribute keys, mirroring
/// `openjiuwen_agent_teams::observability::memory`'s `AT_MEMORY_*` constants.
const AT_MEMORY_TEAM_ID: &str = "team_id";
const AT_MEMORY_OPERATION: &str = "operation";
const AT_MEMORY_OUTCOME: &str = "outcome";
const AT_MEMORY_COUNT: &str = "count";
const AT_MEMORY_MEMBER_ID: &str = "member_id";
const AT_MEMORY_KIND: &str = "kind";
const AT_MEMORY_ERROR: &str = "error";

/// FNV-1a offset basis: the seed of every derived id.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
/// Mixing constant applied between hash parts, so part boundaries stay distinct.
const PART_SEPARATOR: u64 = 0x9e37_79b9_7f4a_7c15;

// ---------------------------------------------------------------------------
// The reproduced bridge
// ---------------------------------------------------------------------------

/// Whether a team-memory operation succeeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Outcome {
    /// The operation completed.
    Ok,
    /// The operation failed.
    Failed,
}

impl Outcome {
    /// The wire value of the `outcome` attribute.
    const fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Failed => "failed",
        }
    }

    /// The frame status the bridge maps the outcome onto.
    const fn status(self) -> SpanStatus {
        match self {
            Self::Ok => SpanStatus::Finish,
            Self::Failed => SpanStatus::Error,
        }
    }
}

/// One team-memory event, i.e. one input of the producer bridge.
#[derive(Debug, Clone)]
struct MemoryEvent {
    /// The operation label (`search`, `write`, `delete`, …).
    operation: &'static str,
    /// The team that ran the operation.
    team_id: &'static str,
    /// The member the operation was scoped to, when member-scoped.
    member_id: Option<&'static str>,
    /// The record kind involved, when the event has one.
    kind: Option<&'static str>,
    /// How many records the operation touched.
    count: u64,
    /// Whether the operation succeeded.
    outcome: Outcome,
    /// The error text of a failed operation.
    error: Option<&'static str>,
}

impl MemoryEvent {
    /// A successful event for `operation` on [`TEAM_ID`].
    fn ok(operation: &'static str) -> Self {
        Self {
            operation,
            team_id: TEAM_ID,
            member_id: None,
            kind: None,
            count: 0,
            outcome: Outcome::Ok,
            error: None,
        }
    }

    /// A failed event for `operation` on [`TEAM_ID`] carrying `error`.
    fn failed(operation: &'static str, error: &'static str) -> Self {
        Self {
            outcome: Outcome::Failed,
            error: Some(error),
            ..Self::ok(operation)
        }
    }

    /// Scopes the event to one member.
    fn with_member(mut self, member_id: &'static str) -> Self {
        self.member_id = Some(member_id);
        self
    }

    /// Records the touched record kind.
    fn with_kind(mut self, kind: &'static str) -> Self {
        self.kind = Some(kind);
        self
    }

    /// Records how many records the operation touched.
    fn with_count(mut self, count: u64) -> Self {
        self.count = count;
        self
    }

    /// The `memory.{operation}` span name of the frame.
    fn name(&self) -> String {
        format!("memory.{}", self.operation)
    }

    /// The derived invoke id: `memory.{operation}.{16 hex digits}`.
    fn invoke_id(&self) -> String {
        let digest = digest(&[self.team_id, self.operation, self.member_id.unwrap_or("")]);
        format!("memory.{}.{digest:016x}", self.operation)
    }

    /// The derived team trace id: 32 lowercase hex characters.
    fn trace_id(&self) -> String {
        let high = digest(&["trace", self.team_id]);
        let low = digest(&["span", self.team_id]);
        format!("{high:016x}{low:016x}")
    }

    /// The attribute bag the bridge stamps into the frame's `meta_data`.
    fn meta_data(&self) -> BTreeMap<String, Value> {
        let mut attributes = BTreeMap::new();
        attributes.insert(AT_MEMORY_TEAM_ID.to_owned(), Value::from(self.team_id));
        attributes.insert(AT_MEMORY_OPERATION.to_owned(), Value::from(self.operation));
        attributes.insert(
            AT_MEMORY_OUTCOME.to_owned(),
            Value::from(self.outcome.as_str()),
        );
        attributes.insert(AT_MEMORY_COUNT.to_owned(), Value::from(self.count));
        if let Some(member_id) = self.member_id.filter(|id| !id.is_empty()) {
            attributes.insert(AT_MEMORY_MEMBER_ID.to_owned(), Value::from(member_id));
        }
        if let Some(kind) = self.kind.filter(|kind| !kind.is_empty()) {
            attributes.insert(AT_MEMORY_KIND.to_owned(), Value::from(kind));
        }
        if let Some(error) = self.error.filter(|text| !text.is_empty()) {
            attributes.insert(AT_MEMORY_ERROR.to_owned(), Value::from(error));
        }
        attributes
    }

    /// The component frame the agent-teams bridge would emit for this event.
    fn frame(&self) -> TraceFrame {
        let invoke_id = self.invoke_id();
        let mut span = TraceSpan::new(
            self.trace_id(),
            "",
            invoke_id.clone(),
            self.name(),
            TraceSpanKind::Component,
        );
        span.status = self.outcome.status();
        // The memory facade reports no timing: the span covers a single instant.
        span.end_time = span.start_time;
        span.elapsed_time = Some(0);
        span.component_id = Some(invoke_id);
        span.component_name = Some(self.name());
        span.component_type = Some(MEMORY_COMPONENT_TYPE.to_owned());
        span.meta_data = self.meta_data();
        span.error = self.error.map(str::to_owned);
        TraceFrame::workflow(span)
    }
}

/// FNV-1a over the parts of one identity, with a mixing round between parts.
fn digest(parts: &[&str]) -> u64 {
    parts.iter().fold(FNV_OFFSET_BASIS, |hash, part| {
        let mixed = part.as_bytes().iter().fold(hash, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME)
        });
        mixed.wrapping_mul(FNV_PRIME) ^ PART_SEPARATOR
    })
}

// ---------------------------------------------------------------------------
// Fixtures and helpers
// ---------------------------------------------------------------------------

/// The three events one batch carries: two successes and one failure.
///
/// A batch (never three separate exports) is what proves the framing survives
/// batching: one POST, one scope, three spans.
fn batch() -> Vec<MemoryEvent> {
    vec![
        MemoryEvent::ok("search")
            .with_member(MEMBER_ID)
            .with_count(3),
        MemoryEvent::ok("write")
            .with_kind(RECORD_KIND)
            .with_count(1),
        MemoryEvent::failed("delete", FAILURE_TEXT),
    ]
}

/// A tokio runtime for the loopback collector and the bounded helper awaits.
fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Runtime::new().expect("a tokio runtime starts")
}

/// Starts a loopback collector whose `/v1/traces` endpoint answers `status`.
fn collector(runtime: &tokio::runtime::Runtime, status: u16) -> MockServer {
    let server = runtime.block_on(MockServer::builder().start());
    runtime.block_on(
        Mock::given(method("POST"))
            .and(path("/v1/traces"))
            .respond_with(ResponseTemplate::new(status))
            .mount(&server),
    );
    server
}

/// The OTLP/HTTP configuration pointing at `server`.
///
/// Every knob is explicit, and each one beats its environment fallback
/// (`resolve_endpoint`, `resolve_headers`, `resolve_protocol`,
/// `resolve_service_name`), so a stray `OTEL_*` variable in the test process
/// cannot redirect, re-encode or rename the export.
fn otlp_config(server: &MockServer, protocol: OtlpProtocol) -> OtelTracerConfig {
    OtelTracerConfig {
        exporter_type: ExporterType::Otlp,
        protocol,
        exporter_endpoint: Some(server.uri()),
        headers: BTreeMap::from([(
            "authorization".to_owned(),
            "Bearer team-memory-token".to_owned(),
        )]),
        service_name: SERVICE_NAME.to_owned(),
        service_version: Some("0.1.20".to_owned()),
        redaction_enabled: false,
        export_timeout_ms: 5_000,
        ..OtelTracerConfig::default()
    }
}

/// Builds the OTLP handle and the sink the bridge feeds.
fn handle_and_sink(config: OtelTracerConfig) -> (TracerHandle, Arc<OtelTraceSink>) {
    let handle = init_otel_tracer(config).expect("the OTLP/HTTP exporter builds offline");
    assert!(handle.is_active(), "the OTLP handle owns a live provider");
    let sink = OtelTraceSink::new(&handle).expect("a live handle builds a sink");
    assert!(
        sink.is_recording(),
        "the sink records against the live handle"
    );
    (handle, sink)
}

/// Emits `events` through `sink` — exactly what the producer bridge does.
fn emit(sink: &OtelTraceSink, events: &[MemoryEvent]) {
    for event in events {
        sink.emit(event.frame());
    }
}

/// The requests the collector recorded so far.
fn requests(runtime: &tokio::runtime::Runtime, server: &MockServer) -> Vec<Request> {
    runtime
        .block_on(server.received_requests())
        .expect("the mock records requests")
}

/// The single POST the batch produced.
fn sole_post(runtime: &tokio::runtime::Runtime, server: &MockServer) -> Request {
    let requests = requests(runtime, server);
    assert_eq!(requests.len(), 1, "one batch, one POST");
    requests.into_iter().next().expect("one request")
}

/// The `content-type` header of a request.
fn content_type(request: &Request) -> String {
    request
        .headers
        .get("content-type")
        .expect("a content type")
        .to_str()
        .expect("an ASCII content type")
        .to_owned()
}

/// Hex-encodes bytes, the OTLP id representation.
fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        // Writing into a `String` cannot fail.
        let _ = write!(encoded, "{byte:02x}");
    }
    encoded
}

/// Whether `text` is a lowercase-hex id of exactly `length` characters.
fn is_lower_hex(text: &str, length: usize) -> bool {
    text.len() == length
        && text
            .chars()
            .all(|character| character.is_ascii_digit() || ('a'..='f').contains(&character))
}

/// The `value` object of one attribute in a bare attribute array.
fn attribute_in<'a>(attributes: &'a Value, key: &str) -> &'a Value {
    attributes
        .as_array()
        .unwrap_or_else(|| panic!("not an attribute array: {attributes}"))
        .iter()
        .find(|pair| pair["key"] == json!(key))
        .map_or_else(
            || panic!("attribute {key} is missing from {attributes}"),
            |pair| &pair["value"],
        )
}

/// The `value` object of one attribute of an OTLP/JSON span.
fn attribute<'a>(span: &'a Value, key: &str) -> &'a Value {
    attribute_in(&span["attributes"], key)
}

/// A string attribute value of an OTLP/JSON span.
fn json_string(span: &Value, key: &str) -> String {
    attribute(span, key)["stringValue"]
        .as_str()
        .unwrap_or_else(|| panic!("attribute {key} is not a string: {span}"))
        .to_owned()
}

/// The `openjiuwen.meta_data` bag of an OTLP/JSON span, parsed back to JSON.
fn json_meta_data(span: &Value) -> Value {
    serde_json::from_str(&json_string(span, OJ_META_DATA))
        .expect("the meta_data attribute holds a JSON object")
}

/// The spans of the single `scopeSpans` entry an OTLP/JSON body carries.
fn json_spans(body: &Value) -> &Vec<Value> {
    assert_eq!(
        body["resourceSpans"]
            .as_array()
            .expect("resourceSpans is an array")
            .len(),
        1,
        "one resource groups the batch",
    );
    assert_eq!(
        attribute_in(
            &body["resourceSpans"][0]["resource"]["attributes"],
            "service.name"
        ),
        &json!({"stringValue": SERVICE_NAME}),
        "the configured service.name is exported",
    );
    let scopes = body["resourceSpans"][0]["scopeSpans"]
        .as_array()
        .expect("scopeSpans is an array");
    assert_eq!(scopes.len(), 1, "one scope groups the batch");
    scopes[0]["spans"]
        .as_array()
        .unwrap_or_else(|| panic!("no spans in the body: {body}"))
}

/// The span named `name` inside an OTLP/JSON batch.
fn json_span_named<'a>(spans: &'a [Value], name: &str) -> &'a Value {
    spans
        .iter()
        .find(|span| span["name"] == json!(name))
        .unwrap_or_else(|| panic!("no span named {name} in {spans:?}"))
}

/// Decodes the spans of an OTLP/protobuf body, flattened across scopes.
fn protobuf_spans(body: &[u8]) -> Vec<Fields> {
    let mut resource_spans = messages(&decode(body), 1);
    assert_eq!(resource_spans.len(), 1, "one resource groups the batch");
    let resource_spans = resource_spans.remove(0);
    let resource_attributes = key_values(&messages(&resource_spans, 1).remove(0), 1);
    assert_eq!(
        any_string(&resource_attributes["service.name"]),
        SERVICE_NAME,
        "the configured service.name is exported",
    );
    messages(&resource_spans, 2)
        .iter()
        .flat_map(|scope_spans| messages(scope_spans, 2))
        .collect()
}

// ---------------------------------------------------------------------------
// Semantic assertions, shared by both encodings
// ---------------------------------------------------------------------------

/// Asserts one OTLP/JSON span carries what the bridge promised for `event`.
fn assert_json_span(span: &Value, event: &MemoryEvent) {
    let invoke_id = event.invoke_id();
    let trace_id = event.trace_id();
    assert!(is_lower_hex(&trace_id, 32), "a team trace id: {trace_id}");
    assert_eq!(
        span["name"],
        json!(format!("component.{invoke_id}")),
        "the generic component branch names the span after its invoke id",
    );
    assert_eq!(span["kind"], json!("SPAN_KIND_INTERNAL"));
    assert!(
        is_lower_hex(span["traceId"].as_str().expect("a trace id"), 32),
        "OTel trace ids are 32 hex characters: {span}",
    );
    assert!(
        is_lower_hex(span["spanId"].as_str().expect("a span id"), 16),
        "OTel span ids are 16 hex characters: {span}",
    );

    assert_eq!(
        json_string(span, OJ_WORKFLOW_COMPONENT_TYPE),
        MEMORY_COMPONENT_TYPE,
    );
    assert_eq!(json_string(span, OJ_WORKFLOW_COMPONENT_NAME), event.name());
    assert_eq!(json_string(span, OJ_WORKFLOW_COMPONENT_ID), invoke_id);
    assert_eq!(json_string(span, OJ_TRACE_ID), trace_id);

    let expected_status = match event.outcome {
        Outcome::Ok => SPAN_STATUS_FINISH,
        Outcome::Failed => SPAN_STATUS_ERROR,
    };
    assert_eq!(json_string(span, OJ_STATUS), expected_status);
    assert_json_status(span, event);

    assert_json_meta_data(span, event);
}

/// Asserts the OTel status of an OTLP/JSON span.
fn assert_json_status(span: &Value, event: &MemoryEvent) {
    match event.outcome {
        Outcome::Ok => assert_eq!(span["status"], json!({"code": "STATUS_CODE_OK"})),
        Outcome::Failed => {
            assert_eq!(
                span["status"],
                json!({"code": "STATUS_CODE_ERROR", "message": FAILURE_TEXT}),
            );
            assert_eq!(json_string(span, OJ_WORKFLOW_ERROR_MESSAGE), FAILURE_TEXT);
            let failure: Value = serde_json::from_str(&json_string(span, OJ_ERROR))
                .expect("the openjiuwen.error attribute holds JSON");
            assert_eq!(failure["message"], json!(FAILURE_TEXT));
        }
    }
}

/// Asserts the `openjiuwen.meta_data` bag of an OTLP/JSON span.
fn assert_json_meta_data(span: &Value, event: &MemoryEvent) {
    let metadata = json_meta_data(span);
    let expected = Value::Object(event.meta_data().into_iter().collect::<Map<_, _>>());
    assert_eq!(
        metadata, expected,
        "the whole team-memory attribute bag travels in openjiuwen.meta_data",
    );
    assert_eq!(metadata[AT_MEMORY_TEAM_ID], json!(TEAM_ID));
    assert_eq!(metadata[AT_MEMORY_OPERATION], json!(event.operation));
    assert_eq!(metadata[AT_MEMORY_OUTCOME], json!(event.outcome.as_str()));
    assert_eq!(metadata[AT_MEMORY_COUNT], json!(event.count));
}

/// Asserts one decoded OTLP/protobuf span carries what the bridge promised.
fn assert_protobuf_span(span: &Fields, event: &MemoryEvent) {
    let invoke_id = event.invoke_id();
    let trace_id = event.trace_id();
    assert_eq!(
        string(span, 5).as_deref(),
        Some(format!("component.{invoke_id}").as_str()),
        "the generic component branch names the span after its invoke id",
    );
    assert_eq!(
        varint(span, 6),
        Some(1),
        "SPAN_KIND_INTERNAL selects the workflow/component path",
    );
    let span_trace_id = bytes(span, 1).expect("a trace id");
    assert_eq!(
        span_trace_id.len(),
        16,
        "trace ids are 16 bytes on the wire"
    );
    assert!(
        is_lower_hex(&hex(span_trace_id), 32),
        "the hex form is 32 characters",
    );
    assert_eq!(bytes(span, 2).expect("a span id").len(), 8);
    assert!(
        fixed64(span, 7) <= fixed64(span, 8),
        "the span must not end before it starts",
    );

    let attributes = key_values(span, 9);
    let text = |key: &str| {
        attributes.get(key).map_or_else(
            || panic!("attribute {key} is missing from {span:?}"),
            any_string,
        )
    };
    assert_eq!(text(OJ_WORKFLOW_COMPONENT_TYPE), MEMORY_COMPONENT_TYPE);
    assert_eq!(text(OJ_WORKFLOW_COMPONENT_NAME), event.name());
    assert_eq!(text(OJ_WORKFLOW_COMPONENT_ID), invoke_id);
    assert_eq!(text(OJ_TRACE_ID), trace_id);
    assert_eq!(
        text(OJ_STATUS),
        match event.outcome {
            Outcome::Ok => SPAN_STATUS_FINISH,
            Outcome::Failed => SPAN_STATUS_ERROR,
        }
    );

    // Status { string message = 2; StatusCode code = 3 }
    let mut statuses = messages(span, 15);
    assert_eq!(statuses.len(), 1, "the status is always present");
    let status = statuses.remove(0);
    match event.outcome {
        Outcome::Ok => {
            assert_eq!(varint(&status, 3), Some(1), "STATUS_CODE_OK is 1");
            assert_eq!(string(&status, 2), None, "a successful span has no message");
        }
        Outcome::Failed => {
            assert_eq!(varint(&status, 3), Some(2), "STATUS_CODE_ERROR is 2");
            assert_eq!(string(&status, 2).as_deref(), Some(FAILURE_TEXT));
            assert_eq!(text(OJ_WORKFLOW_ERROR_MESSAGE), FAILURE_TEXT);
        }
    }

    let metadata: Value = serde_json::from_str(&text(OJ_META_DATA))
        .expect("the meta_data attribute holds a JSON object");
    let expected = Value::Object(event.meta_data().into_iter().collect::<Map<_, _>>());
    assert_eq!(
        metadata, expected,
        "the whole team-memory attribute bag travels in openjiuwen.meta_data",
    );
    assert_eq!(metadata[AT_MEMORY_TEAM_ID], json!(TEAM_ID));
    assert_eq!(metadata[AT_MEMORY_OPERATION], json!(event.operation));
    assert_eq!(metadata[AT_MEMORY_OUTCOME], json!(event.outcome.as_str()));
    assert_eq!(metadata[AT_MEMORY_COUNT], json!(event.count));
}

// ---------------------------------------------------------------------------
// OTLP/HTTP with JSON payloads (the default encoding)
// ---------------------------------------------------------------------------

#[test]
fn the_json_batch_carries_one_component_span_per_team_memory_event() {
    let runtime = runtime();
    let server = collector(&runtime, 200);
    let (handle, sink) = handle_and_sink(otlp_config(&server, OtlpProtocol::Http));

    let events = batch();
    emit(&sink, &events);
    handle.force_flush();

    let request = sole_post(&runtime, &server);
    assert_eq!(request.url.path(), "/v1/traces");
    assert_eq!(request.method.as_str(), "POST");
    assert_eq!(content_type(&request), "application/json");

    let body: Value = serde_json::from_slice(&request.body).expect("the body is JSON");
    let spans = json_spans(&body);
    assert_eq!(spans.len(), events.len(), "one span per team-memory event");
    for event in &events {
        assert_json_span(
            json_span_named(spans, &format!("component.{}", event.invoke_id())),
            event,
        );
    }

    // One team, one trace: the derived id is identical on every span, which is
    // how a backend groups the memory operations of a single team.
    let trace_ids: Vec<String> = events.iter().map(MemoryEvent::trace_id).collect();
    assert_eq!(trace_ids.len(), 3);
    assert_eq!(
        trace_ids[0], trace_ids[1],
        "the team trace id does not depend on the operation",
    );
    assert_eq!(trace_ids[1], trace_ids[2]);
    assert!(is_lower_hex(&trace_ids[0], 32));

    handle.shutdown();
}

// ---------------------------------------------------------------------------
// OTLP/HTTP with protobuf payloads (`http/protobuf`)
// ---------------------------------------------------------------------------

#[test]
fn the_protobuf_batch_carries_the_same_three_component_spans() {
    let runtime = runtime();
    let server = collector(&runtime, 200);
    let (handle, sink) = handle_and_sink(otlp_config(&server, OtlpProtocol::HttpProtobuf));

    let events = batch();
    emit(&sink, &events);
    handle.force_flush();

    let request = sole_post(&runtime, &server);
    assert_eq!(request.url.path(), "/v1/traces");
    assert_eq!(content_type(&request), "application/x-protobuf");

    let spans = protobuf_spans(&request.body);
    assert_eq!(spans.len(), events.len(), "one span per team-memory event");
    for event in &events {
        let name = format!("component.{}", event.invoke_id());
        let span = spans
            .iter()
            .find(|span| string(span, 5).as_deref() == Some(name.as_str()))
            .unwrap_or_else(|| panic!("no span named {name} in the batch"));
        assert_protobuf_span(span, event);
    }

    handle.shutdown();
}

// ---------------------------------------------------------------------------
// Producer-seam robustness
// ---------------------------------------------------------------------------

#[test]
fn a_shut_down_handle_yields_a_sink_that_drops_frames() {
    // The producer bridge asks `is_recording()` first, so a shut-down handle
    // must yield `false` and a sink whose `emit` is a cheap no-op.
    let runtime = runtime();
    let server = collector(&runtime, 200);
    let handle = init_otel_tracer(otlp_config(&server, OtlpProtocol::Http))
        .expect("the OTLP/HTTP exporter builds offline");
    handle.shutdown();
    assert!(!handle.is_active(), "shutdown deactivates the handle");

    let sink = OtelTraceSink::new(&handle).expect("a shut-down handle still builds a sink");
    assert!(!sink.is_recording(), "a shut-down handle records nothing");

    emit(&sink, &batch());
    handle.force_flush();

    assert_eq!(
        requests(&runtime, &server).len(),
        0,
        "a non-recording sink emits nothing, synchronously",
    );
}

#[test]
fn an_exporter_less_handle_records_without_touching_the_network() {
    // `ExporterType::None` builds a provider with no span processor: frames are
    // still mapped (and dropped by the provider), and nothing is ever posted.
    let runtime = runtime();
    let server = collector(&runtime, 200);
    let config = OtelTracerConfig {
        exporter_type: ExporterType::None,
        ..otlp_config(&server, OtlpProtocol::Http)
    };
    let handle = init_otel_tracer(config).expect("a provider without an exporter builds");
    let sink = OtelTraceSink::new(&handle).expect("a live handle builds a sink");

    emit(&sink, &batch());
    handle.force_flush();

    assert_eq!(
        requests(&runtime, &server).len(),
        0,
        "an exporter-less provider posts nothing",
    );
}

#[test]
fn a_failing_collector_leaves_emit_infallible() {
    // `TraceSink::emit` must never fail a traced run: the exporter logs the
    // rejection at `warn` and drops the batch, and the next batch behaves the
    // same way.
    let runtime = runtime();
    let server = collector(&runtime, 500);
    let (handle, sink) = handle_and_sink(otlp_config(&server, OtlpProtocol::Http));

    emit(&sink, &batch());
    handle.force_flush();
    assert_eq!(
        requests(&runtime, &server).len(),
        1,
        "the exporter attempted the batch once",
    );

    emit(&sink, &batch());
    handle.force_flush();
    assert_eq!(
        requests(&runtime, &server).len(),
        2,
        "a rejected batch never poisons the next one",
    );

    // Emitting after the rejection is still infallible, and the handle stays
    // usable for the rest of the run.
    assert!(sink.is_recording());
    handle.shutdown();
}
