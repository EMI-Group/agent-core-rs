// `missing_docs` is allowed before the `cfg` gate: when the `otel` feature is
// off the whole crate (including the `//!` docs) is stripped, and a later
// `allow` would be stripped with it, leaving the denied `missing_docs` lint to
// fire on the empty crate root.
#![allow(missing_docs)]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::too_many_lines // scenarios read as one narrative, like the sibling suites
)]
#![cfg(feature = "otel")]
//! Team-memory observability: lifecycle hook events projected onto OTLP spans.
//!
//! The Python SDK leaves team memory in the dark unless a host wires its own
//! exporter. The Rust port splits the job in two, and this scenario drives both
//! halves through their public seams:
//!
//! 1. **The events** — [`TeamMemory`] carries an optional
//!    [`TeamMemoryObserver`], so a [`TeamMemoryHooks`] lifecycle (member join,
//!    buffered turns, session-end flush) reports itself as
//!    [`TeamMemoryEvent`]s with the operation, team, member, kind, record count,
//!    and outcome. The test-owned [`RecordingObserver`] is that seam's consumer:
//!    it clones every event so the assertions can read them back.
//! 2. **The spans** — the project has no memory→span bridge (no
//!    `openjiuwen-extensions` adapter subscribes to `TeamMemoryEvent`, and the
//!    session tracer has no memory invoke type), so this scenario *hand-builds*
//!    one: each recorded event becomes a session [`Tracer`] span carrying the
//!    event's fields as span metadata and payloads, emitted into an
//!    [`OtelTraceSink`] over an OTLP/HTTP [`TracerHandle`].
//! 3. **The wire** — a loopback `wiremock` collector stands in for the OTLP
//!    backend, and every assertion is made on the decoded OTLP/JSON body
//!    (`resourceSpans → scopeSpans → spans`), never on the SDK's in-process
//!    state.
//!
//! What the file pins:
//!
//! - one flush of the hook pipeline reaches the collector as a single
//!   `POST /v1/traces` with `content-type: application/json`, the configured
//!   `authorization` header, the `service.name`/`service.version` resource
//!   attributes, one instrumentation scope, and one span per memory event whose
//!   name, `openjiuwen.*` identity attributes, and JSON payloads carry the
//!   event's team id, member id, and record count;
//! - a collector that answers `500` is a *failed export*, not a panic and not a
//!   retry loop: the batch is dropped, the pipeline stays usable, and the whole
//!   run stays bounded;
//! - two flushes produce two batches that keep the projection order, carry the
//!   same session trace identity, and never invert their start timestamps.
//!
//! Everything is offline and deterministic: the only endpoint is `127.0.0.1`,
//! the rule extractor is the deterministic Default extractor, turns are built
//! through serde so no time-type crate is named, and every network await is
//! bounded by [`support::within`].
//!
//! [`TeamMemory`]: openjiuwen::core::memory::team::store::TeamMemory
//! [`TeamMemoryObserver`]: openjiuwen::core::memory::team::observe::TeamMemoryObserver
//! [`TeamMemoryHooks`]: openjiuwen::core::memory::team::hooks::TeamMemoryHooks
//! [`TeamMemoryEvent`]: openjiuwen::core::memory::team::observe::TeamMemoryEvent
mod support;

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use openjiuwen::core::memory::record::CallOptions;
use openjiuwen::core::memory::team::extractor::TeamTurn;
use openjiuwen::core::memory::team::hooks::{TeamMemoryHookConfig, TeamMemoryHooks};
use openjiuwen::core::memory::team::observe::{
    TeamMemoryEvent, TeamMemoryObserver, TeamMemoryOperation, TeamMemoryOutcome,
};
use openjiuwen::core::memory::team::store::TeamMemory;
use openjiuwen::core::session::tracer::{InvokeType, TraceSink, Tracer};
use openjiuwen::extensions::otel::{
    init_otel_tracer, ExporterType, OtelTraceSink, OtelTracerConfig, OtlpEncoding,
    OtlpHttpExporter, OtlpProtocol, TracerHandle,
};
use openjiuwen_test_support::wiremock::matchers::{header, method, path};
use openjiuwen_test_support::wiremock::{Mock, MockServer, ResponseTemplate};
use serde_json::{json, Value};

use support::{within, BOUND};

/// The team the hook pipeline runs for.
const TEAM: &str = "review-team";

/// The coordination session bound by `on_session_start`.
const SESSION: &str = "e2e-team-session";

/// The trace id the projected spans share.
const TRACE: &str = "e2e-team-trace";

/// The `service.name` the exporter reports as a resource attribute.
const SERVICE_NAME: &str = "openjiuwen-e2e-team-memory";

/// The `service.version` the exporter reports as a resource attribute.
const SERVICE_VERSION: &str = "0.1.0";

/// The `authorization` header every export carries.
const AUTHORIZATION: &str = "Bearer team-memory-token";

/// The rule extractor's decision record, taken from the extractor's own suite.
const DECISION: &str = "We decided to use SQLite for the shared store";

/// The handoff record the rule extractor derives from the second turn.
const LESSON: &str = "Next time we should run the migration first";

/// The `openjiuwen.agent.name` attribute of the join/leave spans.
const NAME: &str = "openjiuwen.agent.name";

/// The `openjiuwen.agent.invoke_type` attribute.
const INVOKE_TYPE: &str = "openjiuwen.agent.invoke_type";

/// The `openjiuwen.agent.inputs` attribute (a JSON string).
const INPUTS: &str = "openjiuwen.agent.inputs";

/// The `openjiuwen.agent.outputs` attribute (a JSON string).
const OUTPUTS: &str = "openjiuwen.agent.outputs";

/// The `openjiuwen.meta_data` attribute (a JSON string).
const META_DATA: &str = "openjiuwen.meta_data";

/// The `openjiuwen.session_id` attribute.
const SESSION_ID: &str = "openjiuwen.session_id";

/// The `openjiuwen.trace.id` attribute.
const TRACE_ID: &str = "openjiuwen.trace.id";

/// The `openjiuwen.status` attribute.
const STATUS: &str = "openjiuwen.status";

/// A `TeamMemoryObserver` that keeps every event it is handed.
///
/// This is the *only* recording seam the team-memory crate offers: the observer
/// is synchronous and must not block, so it clones the event and does nothing
/// else. A `std::sync::Mutex` is enough — nothing here awaits while holding it.
#[derive(Debug, Default)]
struct RecordingObserver {
    /// The events seen so far, in emission order.
    events: Mutex<Vec<TeamMemoryEvent>>,
}

impl RecordingObserver {
    /// Removes and returns every event seen so far.
    fn take(&self) -> Vec<TeamMemoryEvent> {
        std::mem::take(&mut *lock(&self.events))
    }
}

impl TeamMemoryObserver for RecordingObserver {
    fn observe(&self, event: &TeamMemoryEvent) {
        lock(&self.events).push(event.clone());
    }
}

/// Locks a mutex, recovering the data when a previous holder panicked.
///
/// A poisoned lock must not turn an assertion helper into a second panic.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A turn with the serde-default timestamp, so no time-type crate is named.
fn turn(from_member: &str, content: &str) -> TeamTurn {
    serde_json::from_value(json!({
        "from_member": from_member,
        "content": content,
    }))
    .expect("a team turn deserializes with its default timestamp")
}

/// Runs one deterministic team-memory lifecycle over the observed facade.
///
/// The sequence is the documented hook order: bind the session, join a member,
/// buffer two turns, then flush at session end. The rule extractor is
/// deterministic, so the flush stores exactly one decision record (from the
/// first turn) and one handoff record (from the second).
async fn run_team_session(memory: Arc<TeamMemory>) {
    let hooks = TeamMemoryHooks::with_rule_extractor(TEAM, memory, TeamMemoryHookConfig::default());
    let call = CallOptions::new();

    hooks
        .on_session_start(SESSION, &call)
        .await
        .expect("the team session starts");
    let profile = hooks
        .on_member_join("alice", &call)
        .await
        .expect("the member joins")
        .expect("the join writes a member profile");
    assert_eq!(profile.member_id.as_deref(), Some("alice"));

    hooks
        .on_turn(turn("alice", DECISION), &call)
        .await
        .expect("the decision turn buffers");
    hooks
        .on_turn(turn("bob", LESSON), &call)
        .await
        .expect("the handoff turn buffers");
    assert_eq!(hooks.pending_turns(), 2, "turns are buffered, not written");

    let flush = hooks
        .on_session_end(&call)
        .await
        .expect("the session-end flush extracts and writes");
    assert_eq!(flush.turns, 2, "the flush consumes both buffered turns");
    assert_eq!(flush.len(), 2, "one record per turn");
    assert!(
        flush.degraded.is_none(),
        "the rule extractor never degrades"
    );
}

/// The OTLP/HTTP configuration pointed at the loopback collector.
///
/// Every knob the exporter resolves is set explicitly, so an ambient
/// `OPENJIUWEN_OTEL_*`/`OTEL_*` variable cannot change what the scenario
/// asserts.
fn otlp_config(server: &MockServer) -> OtelTracerConfig {
    OtelTracerConfig {
        exporter_type: ExporterType::Otlp,
        protocol: OtlpProtocol::Http,
        exporter_endpoint: Some(server.uri()),
        headers: BTreeMap::from([("authorization".to_owned(), AUTHORIZATION.to_owned())]),
        service_name: SERVICE_NAME.to_owned(),
        service_version: Some(SERVICE_VERSION.to_owned()),
        redaction_enabled: false,
        export_timeout_ms: 5_000,
        ..OtelTracerConfig::default()
    }
}

/// Builds a session tracer whose sink is the OTLP-backed `OtelTraceSink`.
fn span_tracer(handle: &TracerHandle) -> Tracer {
    let sink: Arc<dyn TraceSink> =
        OtelTraceSink::new(handle).expect("an active handle builds a trace sink");
    Tracer::with_sink(SESSION, TRACE, sink)
}

/// Projects recorded team-memory events onto spans through the session sink.
///
/// This is the bridge the crate does not ship. The session tracer has no memory
/// invoke type, so each event is emitted as a `Chain` span named
/// `team_memory.{operation}` (the sink prefixes it, giving
/// `chain.team_memory.{operation}`); the event's team, member, kind, count, and
/// outcome travel as span metadata, the event as the span inputs, and the
/// outcome plus count as the span outputs.
fn project(tracer: &Tracer, events: &[TeamMemoryEvent]) {
    for event in events {
        let operation = event.operation.as_str();
        let span = tracer.start_span(format!("team_memory.{operation}"), InvokeType::Chain);
        span.set_metadata("team_id", json!(event.team_id.as_str()));
        span.set_metadata("operation", json!(operation));
        span.set_metadata("outcome", json!(event.outcome.as_str()));
        span.set_metadata("count", json!(event.count));
        if let Some(member_id) = &event.member_id {
            span.set_metadata("member_id", json!(member_id.as_str()));
        }
        if let Some(kind) = &event.kind {
            span.set_metadata("kind", json!(kind.as_str()));
        }
        span.set_inputs(json!({
            "team_id": event.team_id.as_str(),
            "operation": operation,
            "member_id": event.member_id.as_deref(),
            "count": event.count,
        }));
        span.finish(json!({
            "count": event.count,
            "outcome": event.outcome.as_str(),
        }));
    }
}

/// The spans of the single `scopeSpans` entry of one exported batch.
fn spans_of(body: &Value) -> &Vec<Value> {
    body["resourceSpans"][0]["scopeSpans"][0]["spans"]
        .as_array()
        .unwrap_or_else(|| panic!("the batch carries no spans: {body}"))
}

/// The `stringValue` of one attribute of a bare attribute array.
fn attribute_in<'a>(attributes: &'a Value, key: &str) -> Option<&'a Value> {
    attributes
        .as_array()?
        .iter()
        .find(|pair| pair["key"] == json!(key))
        .map(|pair| &pair["value"])
}

/// The `stringValue` of one span attribute as text.
fn text_attribute(span: &Value, key: &str) -> Option<String> {
    attribute_in(&span["attributes"], key)
        .and_then(|value| value.get("stringValue"))
        .and_then(Value::as_str)
        .map(str::to_owned)
}

/// The required text of one span attribute, with the span dumped on absence.
fn must_text(span: &Value, key: &str) -> String {
    text_attribute(span, key).unwrap_or_else(|| panic!("span attribute {key} is missing: {span}"))
}

/// One span attribute decoded as JSON (the sink serializes payloads as strings).
fn json_attribute(span: &Value, key: &str) -> Value {
    let text = must_text(span, key);
    serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("attribute {key} is not JSON ({error}): {text}"))
}

/// One span attribute decoded as JSON, or `Null` when the span has no such key.
fn optional_json_attribute(span: &Value, key: &str) -> Value {
    text_attribute(span, key).map_or(Value::Null, |text| {
        serde_json::from_str(&text)
            .unwrap_or_else(|error| panic!("attribute {key} is not JSON ({error}): {text}"))
    })
}

/// The `openjiuwen.meta_data` of one span, decoded.
fn metadata_of(span: &Value) -> Value {
    optional_json_attribute(span, META_DATA)
}

/// The single span matching `predicate`, described by `what` on failure.
fn span_where<'a>(spans: &'a [Value], what: &str, predicate: impl Fn(&Value) -> bool) -> &'a Value {
    let found: Vec<&Value> = spans.iter().filter(|span| predicate(span)).collect();
    assert_eq!(found.len(), 1, "expected one {what}: {spans:?}");
    found[0]
}

/// One span by name; the caller must know the name is unique in the batch.
fn single_span<'a>(spans: &'a [Value], name: &str) -> &'a Value {
    let found: Vec<&Value> = spans
        .iter()
        .filter(|span| span["name"] == json!(name))
        .collect();
    assert_eq!(found.len(), 1, "expected one span named {name}: {spans:?}");
    found[0]
}

/// Parses a decimal-string OTLP timestamp.
fn nanos(value: &Value) -> u128 {
    let text = value
        .as_str()
        .unwrap_or_else(|| panic!("{value} is not a decimal timestamp string"));
    text.parse()
        .unwrap_or_else(|error| panic!("{text} is not a decimal timestamp: {error}"))
}

/// The resource attribute `key` as text.
fn resource_text(body: &Value, key: &str) -> Option<String> {
    attribute_in(&body["resourceSpans"][0]["resource"]["attributes"], key)
        .and_then(|value| value.get("stringValue"))
        .and_then(Value::as_str)
        .map(str::to_owned)
}

/// Asserts the resource and instrumentation scope of one batch.
fn assert_resource_and_scope(body: &Value) {
    assert_eq!(
        resource_text(body, "service.name").as_deref(),
        Some(SERVICE_NAME),
    );
    assert_eq!(
        resource_text(body, "service.version").as_deref(),
        Some(SERVICE_VERSION),
    );
    let scopes = body["resourceSpans"][0]["scopeSpans"]
        .as_array()
        .unwrap_or_else(|| panic!("the batch carries no scopeSpans: {body}"));
    assert_eq!(scopes.len(), 1, "one instrumentation scope per batch");
    assert_eq!(scopes[0]["scope"]["name"], json!("openjiuwen.tracer.otel"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn team_memory_hook_events_reach_the_collector_as_otlp_spans() {
    // Two workers on purpose: `TracerHandle::force_flush` blocks the thread that
    // calls it while the batch is posted, and the loopback collector still has
    // to be driven by the same runtime.
    let server = MockServer::builder().start().await;
    Mock::given(method("POST"))
        .and(path("/v1/traces"))
        .and(header("content-type", "application/json"))
        .and(header("authorization", AUTHORIZATION))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&server)
        .await;

    let handle = init_otel_tracer(otlp_config(&server)).expect("OTLP/HTTP initializes offline");
    assert!(handle.is_active(), "an OTLP/HTTP handle is active");
    assert!(
        handle.capture().is_none(),
        "the OTLP exporter exposes no capture API",
    );
    let tracer = span_tracer(&handle);

    // The observed hook lifecycle: join → two buffered turns → session-end flush.
    let observer = Arc::new(RecordingObserver::default());
    let memory = Arc::new(TeamMemory::in_memory().with_observer(observer.clone()));
    run_team_session(Arc::clone(&memory)).await;
    let events = observer.take();
    assert_eq!(
        events.len(),
        3,
        "join, flush write, and extract: {events:?}"
    );
    assert_eq!(events[0].operation, TeamMemoryOperation::Write);
    assert_eq!(events[0].member_id.as_deref(), Some("alice"));
    assert_eq!(events[0].count, 1);
    assert_eq!(events[1].operation, TeamMemoryOperation::Write);
    assert_eq!(
        events[1].count, 2,
        "the flush writes both records in one batch"
    );
    assert_eq!(events[2].operation, TeamMemoryOperation::Extract);
    assert_eq!(events[2].count, 2);
    for event in &events {
        assert_eq!(event.team_id, TEAM);
        assert_eq!(event.outcome, TeamMemoryOutcome::Ok);
        assert!(event.error.is_none(), "no failure is reported: {event:?}");
    }

    project(&tracer, &events);
    handle.force_flush();

    let requests = within(
        "the collector records the batch",
        server.received_requests(),
    )
    .await
    .expect("the mock records requests");
    assert_eq!(requests.len(), 1, "one flush, one POST");

    let request = &requests[0];
    assert_eq!(request.url.path(), "/v1/traces");
    assert_eq!(
        request
            .headers
            .get("content-type")
            .expect("a content type")
            .to_str()
            .expect("an ASCII content type"),
        "application/json",
    );
    assert_eq!(
        request
            .headers
            .get("authorization")
            .expect("the configured authorization header")
            .to_str()
            .expect("an ASCII authorization"),
        AUTHORIZATION,
    );

    let body: Value = serde_json::from_slice(&request.body).expect("the body is OTLP/JSON");
    assert_resource_and_scope(&body);
    let spans = spans_of(&body);
    assert_eq!(spans.len(), 3, "one span per memory event");

    let names: Vec<&str> = spans
        .iter()
        .filter_map(|span| span["name"].as_str())
        .collect();
    assert_eq!(
        names,
        [
            "chain.team_memory.write",
            "chain.team_memory.write",
            "chain.team_memory.extract",
        ],
        "the projection order is the event order",
    );

    // The sink opens each frame's span without an ambient parent (the documented
    // deviation from Python's context attach/detach), so every projected span is
    // its own OTel trace. What the seam does keep is the session identity: the
    // tracer's session and trace ids travel as `openjiuwen.*` attributes.
    let trace_id = spans[0]["traceId"].as_str().expect("a trace id").to_owned();
    assert_eq!(trace_id.len(), 32, "the trace id is hex-encoded");
    let mut span_ids = Vec::new();
    for span in spans {
        assert_eq!(span["kind"], json!("SPAN_KIND_INTERNAL"));
        assert_eq!(must_text(span, SESSION_ID), SESSION);
        assert_eq!(must_text(span, TRACE_ID), TRACE);
        assert_eq!(must_text(span, STATUS), "finish");
        assert_eq!(must_text(span, INVOKE_TYPE), "chain");
        assert!(
            nanos(&span["startTimeUnixNano"]) <= nanos(&span["endTimeUnixNano"]),
            "the span must not end before it starts: {span}",
        );
        let span_id = span["spanId"].as_str().expect("a span id").to_owned();
        assert_eq!(span_id.len(), 16, "the span id is hex-encoded");
        assert!(
            !span_ids.contains(&span_id),
            "each span id is unique: {span}"
        );
        span_ids.push(span_id);
    }

    // The member-join event: the only write that names a member.
    let join = span_where(spans, "member-join write", |span| {
        metadata_of(span).get("member_id") == Some(&json!("alice"))
    });
    assert_eq!(join["name"], json!("chain.team_memory.write"));
    assert_eq!(must_text(join, NAME), "team_memory.write");
    let meta = metadata_of(join);
    assert_eq!(meta["team_id"], json!(TEAM));
    assert_eq!(meta["operation"], json!("write"));
    assert_eq!(meta["outcome"], json!("ok"));
    assert_eq!(meta["count"], json!(1));
    assert_eq!(meta["member_id"], json!("alice"));
    assert_eq!(
        meta["kind"],
        json!("member_profile"),
        "the join writes a member profile record",
    );
    let inputs = json_attribute(join, INPUTS);
    assert_eq!(inputs["team_id"], json!(TEAM));
    assert_eq!(inputs["count"], json!(1));
    let outputs = json_attribute(join, OUTPUTS);
    assert_eq!(outputs["count"], json!(1));
    assert_eq!(outputs["outcome"], json!("ok"));

    // The flush's batch write: team-scoped, two records, no member.
    let batch = span_where(spans, "team-scoped flush write", |span| {
        let meta = metadata_of(span);
        meta["operation"] == json!("write") && meta.get("member_id").is_none()
    });
    let meta = metadata_of(batch);
    assert_eq!(meta["count"], json!(2));
    assert_eq!(meta["team_id"], json!(TEAM));
    assert!(
        meta.get("kind").is_none(),
        "a multi-record write has no single kind: {meta}",
    );

    // The extraction that produced the batch.
    let extract = single_span(spans, "chain.team_memory.extract");
    let meta = metadata_of(extract);
    assert_eq!(meta["operation"], json!("extract"));
    assert_eq!(meta["count"], json!(2));
    assert_eq!(json_attribute(extract, OUTPUTS)["count"], json!(2));

    handle.shutdown();
    assert!(!handle.is_active(), "shutdown deactivates the handle");
    handle.shutdown(); // idempotent
    handle.force_flush(); // harmless after shutdown
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_rejected_batch_is_a_failed_export_that_never_retries() {
    // The typed failure lives behind `SpanExporter::export`, which returns
    // `OTelSdkError::InternalFailure`; a crate that cannot name
    // `opentelemetry_sdk` types cannot call it. What the public seam *does*
    // guarantee is the outward contract asserted here: a rejected batch is
    // dropped (the handle logs at `warn` and returns `()`), it is posted exactly
    // once, and the exporter keeps accepting later batches.
    let server = MockServer::builder().start().await;
    Mock::given(method("POST"))
        .and(path("/v1/traces"))
        .respond_with(ResponseTemplate::new(500).set_body_raw("collector down", "text/plain"))
        .expect(2)
        .mount(&server)
        .await;

    let handle = init_otel_tracer(otlp_config(&server)).expect("OTLP/HTTP initializes offline");
    let tracer = span_tracer(&handle);

    let observer = Arc::new(RecordingObserver::default());
    let memory = Arc::new(TeamMemory::in_memory().with_observer(observer.clone()));
    run_team_session(Arc::clone(&memory)).await;
    let events = observer.take();
    assert_eq!(
        events.len(),
        3,
        "join, flush write, and extract: {events:?}"
    );

    // First batch: rejected by the collector, dropped by the exporter.
    project(&tracer, &events[..1]);
    handle.force_flush();
    // Second batch: the pipeline is still alive after the rejection.
    project(&tracer, &events[2..3]);
    handle.force_flush();
    handle.shutdown_with_timeout(BOUND);

    let requests = within(
        "the collector records both batches",
        server.received_requests(),
    )
    .await
    .expect("the mock records requests");
    assert_eq!(
        requests.len(),
        2,
        "one POST per flush: a rejection is never retried",
    );
    let mut names = Vec::new();
    for request in &requests {
        assert_eq!(request.url.path(), "/v1/traces");
        assert_eq!(
            request
                .headers
                .get("content-type")
                .expect("a content type")
                .to_str()
                .expect("an ASCII content type"),
            "application/json",
        );
        // The body was encoded correctly; only the collector's status failed.
        let body: Value = serde_json::from_slice(&request.body).expect("the body is OTLP/JSON");
        assert_resource_and_scope(&body);
        let spans = spans_of(&body);
        assert_eq!(spans.len(), 1, "one span per batch");
        names.push(spans[0]["name"].as_str().expect("a span name").to_owned());
    }
    assert_eq!(
        names,
        ["chain.team_memory.write", "chain.team_memory.extract"],
        "both batches were exported, in projection order",
    );
    assert!(
        !handle.is_active(),
        "shutdown is still clean after failures"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_flushes_export_two_ordered_batches_with_the_same_session_trace() {
    let server = MockServer::builder().start().await;
    Mock::given(method("POST"))
        .and(path("/v1/traces"))
        .respond_with(ResponseTemplate::new(200))
        .expect(2)
        .mount(&server)
        .await;

    let handle = init_otel_tracer(otlp_config(&server)).expect("OTLP/HTTP initializes offline");
    let tracer = span_tracer(&handle);

    let observer = Arc::new(RecordingObserver::default());
    let memory = Arc::new(TeamMemory::in_memory().with_observer(observer.clone()));
    run_team_session(Arc::clone(&memory)).await;
    let events = observer.take();

    // One flush per event, so the batch processor drains a single span each time.
    project(&tracer, &events[1..2]);
    handle.force_flush();
    project(&tracer, &events[2..3]);
    handle.force_flush();

    let requests = within(
        "the collector records both batches",
        server.received_requests(),
    )
    .await
    .expect("the mock records requests");
    assert_eq!(requests.len(), 2, "one batch per flush");

    let first: Value =
        serde_json::from_slice(&requests[0].body).expect("the first body is OTLP/JSON");
    let second: Value =
        serde_json::from_slice(&requests[1].body).expect("the second body is OTLP/JSON");
    assert_resource_and_scope(&first);
    assert_resource_and_scope(&second);
    assert_eq!(spans_of(&first).len(), 1, "the first batch holds one span");
    assert_eq!(
        spans_of(&second).len(),
        1,
        "the second batch holds one span"
    );

    let earlier = &spans_of(&first)[0];
    let later = &spans_of(&second)[0];
    assert_eq!(earlier["name"], json!("chain.team_memory.write"));
    assert_eq!(later["name"], json!("chain.team_memory.extract"));
    assert_eq!(
        must_text(earlier, TRACE_ID),
        TRACE,
        "both batches keep the tracer's session trace identity",
    );
    assert_eq!(must_text(later, TRACE_ID), TRACE);
    assert!(
        nanos(&later["startTimeUnixNano"]) >= nanos(&earlier["startTimeUnixNano"]),
        "the projection order is preserved across batches: {earlier} then {later}",
    );

    // The encoding contract the exporter guarantees: the choice fixes both the
    // traces URL and the `Content-Type`, and JSON stays the default.
    let json_exporter =
        OtlpHttpExporter::new(server.uri(), BTreeMap::new(), Duration::from_millis(500))
            .expect("the JSON exporter builds without I/O");
    assert_eq!(
        json_exporter.traces_url(),
        format!("{}/v1/traces", server.uri())
    );
    assert_eq!(
        json_exporter.content_type(),
        OtlpEncoding::Json.content_type()
    );
    assert_eq!(OtlpEncoding::default(), OtlpEncoding::Json);

    let protobuf = OtlpHttpExporter::with_encoding(
        server.uri(),
        BTreeMap::new(),
        Duration::from_millis(500),
        OtlpEncoding::Protobuf,
    )
    .expect("the protobuf exporter builds without I/O");
    assert_eq!(protobuf.content_type(), "application/x-protobuf");

    handle.shutdown();
}
