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
//! Team-memory observability: the shipped `otlp` bridge driven onto the wire.
//!
//! `openjiuwen-agent-teams` owns the bridge this scenario exercises. On the
//! `otlp` feature the crate ships [`OtlpMemoryObserver`], a [`TeamMemoryObserver`]
//! that turns every team-memory event into one `memory.{operation}` component
//! frame and hands it to `openjiuwen-extensions`' [`OtelTraceSink`]. The sink
//! maps the frame onto one `OTel` span and exports it through the provider it
//! was built over — here OTLP/HTTP against a loopback collector.
//!
//! ```text
//! TeamMemoryHooks ──event──> OtlpMemoryObserver ──frame──> OtelTraceSink ──> OTLP/HTTP
//! ```
//!
//! The scenario supplies only the *collector*: it wires the bridge through the
//! crate's own injection points — [`TeamMemoryWiring::open_with_observer`] on the
//! metadata path and `TeamMemory::with_observer` on the bare facade — and
//! asserts the decoded OTLP/JSON body (`resourceSpans → scopeSpans → spans`),
//! never the SDK's in-process state.
//!
//! What the file pins, all of it taken from the bridge's own code:
//!
//! - one flush of the hook lifecycle reaches the collector as a single
//!   `POST /v1/traces` with `content-type: application/json`, the configured
//!   `authorization` header, the `service.name`/`service.version` resource
//!   attributes and one instrumentation scope;
//! - one span per event, named `component.memory.{operation}.{16 hex}` with
//!   `openjiuwen.workflow.component.type = "MemoryOperation"`, the
//!   `openjiuwen.workflow.component.{id,name}` identity, the derived 32-hex
//!   `openjiuwen.trace.id` shared by the team's spans, `openjiuwen.status =
//!   "finish"`, and the `openjiuwen.meta_data` blob carrying the event's team,
//!   member, kind, count and outcome;
//! - a collector that answers `500` is a *failed export*, not a panic and not a
//!   retry loop: one POST per batch, the pipeline stays usable, the shutdown
//!   stays bounded;
//! - two flushes produce two batches in emission order that keep the derived
//!   team trace identity, and `OtlpEncoding` fixes both the URL path and the
//!   content type;
//! - an untraced wiring (`trace: false`) exports nothing even though a bridge is
//!   injected: the bridge is opt-in.
//!
//! Each exported memory span is its own `OTel` trace: the bridge's team trace is
//! plain data, not an `OTel` span, so there is no parent span context to attach
//! to and the derived trace id travels as `openjiuwen.trace.id` instead (see the
//! bridge's own module docs). The observer over an OTLP handle also carries no
//! session id, and the sink was built without a fallback, so no
//! `openjiuwen.session_id` attribute is exported — that absence is pinned below.
//!
//! Everything is offline and deterministic: the only endpoint is `127.0.0.1`,
//! the extractor is the deterministic rule extractor, turns are built through
//! serde so no time-type crate is named, and every network await is bounded.
//!
//! [`TeamMemoryObserver`]: openjiuwen::core::memory::team::observe::TeamMemoryObserver
//! [`TeamMemoryWiring::open_with_observer`]: openjiuwen::agent_teams::TeamMemoryWiring::open_with_observer
mod support;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use openjiuwen::agent_teams::{MemoryConfig, OtlpMemoryObserver, TeamMemoryWiring};
use openjiuwen::core::memory::record::CallOptions;
use openjiuwen::core::memory::team::extractor::TeamTurn;
use openjiuwen::core::memory::team::observe::TeamMemoryObserver;
use openjiuwen::core::memory::team::store::TeamMemory;
use openjiuwen::core::memory::team::{NewTeamMemory, TeamMemoryScope, TeamSearchOptions};
use openjiuwen::extensions::otel::{
    init_otel_tracer, ExporterType, OtelTraceSink, OtelTracerConfig, OtlpEncoding,
    OtlpHttpExporter, OtlpProtocol, TracerHandle,
};
use openjiuwen_test_support::wiremock::matchers::{header, method, path};
use openjiuwen_test_support::wiremock::{Mock, MockServer, ResponseTemplate};
use serde_json::{json, Value};

use support::{within, BOUND};

/// The team the hook lifecycle runs for.
const TEAM: &str = "review-team";

/// The coordination session bound by `on_session_start`.
const SESSION: &str = "e2e-team-session";

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

/// The `openjiuwen.workflow.component.type` the bridge stamps on every span.
const COMPONENT_TYPE: &str = "openjiuwen.workflow.component.type";

/// The `openjiuwen.workflow.component.name` attribute.
const COMPONENT_NAME: &str = "openjiuwen.workflow.component.name";

/// The `openjiuwen.workflow.component.id` attribute.
const COMPONENT_ID: &str = "openjiuwen.workflow.component.id";

/// The component type that keeps the sink on its generic component branch.
const MEMORY_COMPONENT_TYPE: &str = "MemoryOperation";

/// The `openjiuwen.invoke_id` attribute.
const INVOKE_ID: &str = "openjiuwen.invoke_id";

/// The `openjiuwen.parent_invoke_id` attribute.
const PARENT_INVOKE_ID: &str = "openjiuwen.parent_invoke_id";

/// The `openjiuwen.meta_data` attribute (a JSON string).
const META_DATA: &str = "openjiuwen.meta_data";

/// The `openjiuwen.session_id` attribute.
const SESSION_ID: &str = "openjiuwen.session_id";

/// The `openjiuwen.trace.id` attribute.
const TRACE_ID: &str = "openjiuwen.trace.id";

/// The `openjiuwen.status` attribute.
const STATUS: &str = "openjiuwen.status";

/// A turn with the serde-default timestamp, so no time-type crate is named.
fn turn(from_member: &str, content: &str) -> TeamTurn {
    serde_json::from_value(json!({
        "from_member": from_member,
        "content": content,
    }))
    .expect("a team turn deserializes with its default timestamp")
}

/// Builds the shipped bridge over `handle`, plus the sink it forwards to.
///
/// This is exactly the wiring a host writes: the sink comes from the live OTLP
/// handle, the bridge wraps it, and the facade or the wiring is handed the
/// bridge as its observer.
fn otlp_bridge(handle: &TracerHandle) -> (Arc<OtelTraceSink>, Arc<OtlpMemoryObserver>) {
    let sink = OtelTraceSink::new(handle).expect("an active handle builds the OTLP sink");
    assert!(sink.is_recording(), "an active OTLP handle records spans");
    let observer = Arc::new(OtlpMemoryObserver::new(Arc::clone(&sink)));
    assert_eq!(
        observer.session_id(),
        None,
        "the bridge leaves the frame's session id empty for the sink's fallback",
    );
    (sink, observer)
}

/// An in-memory facade whose operations report to `observer`.
fn observed_facade(observer: Arc<OtlpMemoryObserver>) -> Arc<TeamMemory> {
    Arc::new(TeamMemory::in_memory().with_observer(observer))
}

/// Runs the deterministic hook lifecycle over a wired team.
///
/// The sequence is the documented hook order: bind the session, join a member,
/// buffer two turns, then flush at session end. The rule extractor is
/// deterministic, so the flush stores exactly one decision record (from the
/// first turn) and one handoff record (from the second).
async fn run_hook_lifecycle(wiring: &TeamMemoryWiring) {
    let hooks = wiring.hooks();
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
/// asserts. Redaction is off so the bridge's attribute values stay readable.
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

/// A collector that accepts exactly `batches` well-formed exports.
async fn accepting_collector(batches: u64) -> MockServer {
    let server = MockServer::builder().start().await;
    Mock::given(method("POST"))
        .and(path("/v1/traces"))
        .and(header("content-type", "application/json"))
        .and(header("authorization", AUTHORIZATION))
        .respond_with(ResponseTemplate::new(200))
        .expect(batches)
        .mount(&server)
        .await;
    server
}

/// The spans of the single `scopeSpans` entry of one exported batch.
fn spans_of(body: &Value) -> &Vec<Value> {
    body["resourceSpans"][0]["scopeSpans"][0]["spans"]
        .as_array()
        .unwrap_or_else(|| panic!("the batch carries no spans: {body}"))
}

/// The `value` of one entry of a bare OTLP attribute array.
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

/// The `openjiuwen.meta_data` of one span, decoded.
fn metadata_of(span: &Value) -> Value {
    text_attribute(span, META_DATA).map_or(Value::Null, |text| {
        serde_json::from_str(&text)
            .unwrap_or_else(|error| panic!("openjiuwen.meta_data is not JSON ({error}): {text}"))
    })
}

/// The single span matching `predicate`, described by `what` on failure.
fn span_where<'a>(spans: &'a [Value], what: &str, predicate: impl Fn(&Value) -> bool) -> &'a Value {
    let found: Vec<&Value> = spans.iter().filter(|span| predicate(span)).collect();
    assert_eq!(found.len(), 1, "expected one {what}: {spans:?}");
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

/// Asserts that `text` is a lowercase 32-hex-character `OTel` identifier.
fn assert_trace_id_shape(text: &str) {
    assert_eq!(text.len(), 32, "a trace id is 32 hex characters: {text}");
    assert!(
        text.chars()
            .all(|char| char.is_ascii_digit() || ('a'..='f').contains(&char)),
        "a trace id is lowercase hex: {text}",
    );
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

/// Asserts the bridge-stamped identity of one exported memory span.
///
/// Every value comes from the bridge (`openjiuwen-agent-teams`'
/// `observability::otlp`) or from the sink's generic component branch — nothing
/// here is invented by the scenario. Returns the span's derived team trace id.
fn assert_memory_span(span: &Value, operation: &str) -> String {
    let invoke_id = must_text(span, INVOKE_ID);
    let expected_prefix = format!("memory.{operation}.");
    assert!(
        invoke_id.starts_with(&expected_prefix),
        "the invoke id is `memory.{{operation}}.{{16 hex}}`: {invoke_id}",
    );
    assert_eq!(must_text(span, COMPONENT_ID), invoke_id, "the bridge's id");
    assert_eq!(must_text(span, COMPONENT_TYPE), MEMORY_COMPONENT_TYPE);
    assert_eq!(
        must_text(span, COMPONENT_NAME),
        format!("memory.{operation}"),
        "the frame name travels as the component name",
    );
    assert_eq!(
        span["name"],
        json!(format!("component.{invoke_id}")),
        "the sink names a generic component span `component.{{invoke_id}}`",
    );
    assert_eq!(span["kind"], json!("SPAN_KIND_INTERNAL"));
    assert_eq!(must_text(span, STATUS), "finish", "an ok event finishes");

    // The bridge has no parent span: the frame is a trace root and the derived
    // team trace id travels as `openjiuwen.trace.id`.
    assert!(
        text_attribute(span, PARENT_INVOKE_ID).is_none(),
        "a memory span is a trace root: {span}",
    );
    let trace_id = must_text(span, TRACE_ID);
    assert_trace_id_shape(&trace_id);

    // Neither the observer nor this sink carries a session id, so the sink never
    // invents one (the documented no-fallback case).
    assert!(
        text_attribute(span, SESSION_ID).is_none(),
        "no session id is fabricated: {span}",
    );

    let span_id = span["spanId"].as_str().expect("a span id");
    assert_eq!(span_id.len(), 16, "a span id is 16 hex characters");
    assert!(
        nanos(&span["startTimeUnixNano"]) <= nanos(&span["endTimeUnixNano"]),
        "the span must not end before it starts: {span}",
    );
    trace_id
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_otlp_bridge_forwards_the_hook_lifecycle_to_the_collector() {
    // Two workers on purpose: `TracerHandle::force_flush` blocks the thread that
    // calls it while the batch is posted, and the loopback collector still has
    // to be driven by the same runtime.
    let server = accepting_collector(1).await;
    let handle = init_otel_tracer(otlp_config(&server)).expect("OTLP/HTTP initializes offline");
    assert!(handle.is_active(), "an OTLP/HTTP handle is active");
    assert!(
        handle.capture().is_none(),
        "the OTLP exporter exposes no capture API",
    );

    // The shipped bridge, attached through the crate's own injection point.
    let (_sink, observer) = otlp_bridge(&handle);
    let dir = tempfile::tempdir().expect("a temp workspace for the team document");
    let wiring = TeamMemoryWiring::open_with_observer(
        TEAM,
        MemoryConfig::default(),
        dir.path(),
        Arc::clone(&observer) as Arc<dyn TeamMemoryObserver>,
    )
    .await
    .expect("the traced wiring opens");
    assert!(wiring.trace_enabled(), "the default configuration traces");
    assert!(
        wiring.observer().is_some(),
        "the injected bridge replaced the built-in in-memory one",
    );

    // Join → two buffered turns → session-end flush.
    run_hook_lifecycle(&wiring).await;
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
    assert_eq!(spans.len(), 3, "one span per team-memory event: {spans:?}");
    let names: Vec<&str> = spans
        .iter()
        .filter_map(|span| span["name"].as_str())
        .collect();
    let write = |index: usize| names[index].starts_with("component.memory.write.");
    let is_extract = |index: usize| names[index].starts_with("component.memory.extract.");
    assert!(
        write(0) && write(1) && is_extract(2),
        "the spans keep the event order (join write, flush write, extract): {names:?}",
    );

    // The join: the only write that names a member and a kind.
    let join = span_where(spans, "member-join write", |span| {
        metadata_of(span).get("member_id") == Some(&json!("alice"))
    });
    let trace_id = assert_memory_span(join, "write");
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

    // The flush's batch write: team-scoped, two records, no single kind.
    let batch = span_where(spans, "team-scoped flush write", |span| {
        let meta = metadata_of(span);
        meta["operation"] == json!("write") && meta.get("member_id").is_none()
    });
    assert_eq!(assert_memory_span(batch, "write"), trace_id);
    let meta = metadata_of(batch);
    assert_eq!(meta["count"], json!(2));
    assert_eq!(meta["team_id"], json!(TEAM));
    assert_eq!(meta["outcome"], json!("ok"));
    assert!(
        meta.get("kind").is_none(),
        "a multi-record write has no single kind: {meta}",
    );

    // The extraction that produced the batch.
    let extract = spans
        .iter()
        .find(|span| {
            span["name"]
                .as_str()
                .is_some_and(|name| name.starts_with("component.memory.extract."))
        })
        .expect("the extract span");
    assert_eq!(assert_memory_span(extract, "extract"), trace_id);
    let meta = metadata_of(extract);
    assert_eq!(meta["operation"], json!("extract"));
    assert_eq!(meta["count"], json!(2));
    assert_eq!(meta["team_id"], json!(TEAM));

    // Every span shares the derived team trace id, while each is its own OTel
    // trace: the bridge has no parent span context to attach to.
    let mut otel_traces = Vec::new();
    for span in spans {
        assert_eq!(must_text(span, TRACE_ID), trace_id);
        let otel_trace = span["traceId"].as_str().expect("an OTel trace id");
        assert_trace_id_shape(otel_trace);
        assert!(
            !otel_traces.contains(&otel_trace.to_owned()),
            "each parentless frame starts its own OTel trace: {span}",
        );
        otel_traces.push(otel_trace.to_owned());
    }

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
    // dropped, it is posted exactly once, and the exporter keeps accepting later
    // batches.
    let server = MockServer::builder().start().await;
    Mock::given(method("POST"))
        .and(path("/v1/traces"))
        .respond_with(ResponseTemplate::new(500).set_body_raw("collector down", "text/plain"))
        .expect(2)
        .mount(&server)
        .await;

    let handle = init_otel_tracer(otlp_config(&server)).expect("OTLP/HTTP initializes offline");
    let (_sink, observer) = otlp_bridge(&handle);
    let memory = observed_facade(observer);
    let call = CallOptions::new();

    // First batch: one bridge span, rejected by the collector and dropped by the
    // exporter.
    memory
        .remember(
            NewTeamMemory::decision(TEAM, DECISION).with_member("alice"),
            &call,
        )
        .await
        .expect("the decision record is remembered");
    handle.force_flush();

    // Second batch: the pipeline is still alive after the rejection.
    let hits = memory
        .search(
            &TeamMemoryScope::member(TEAM, "alice"),
            "SQLite",
            &TeamSearchOptions::default(),
        )
        .await
        .expect("the search reads the stored record");
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

    let mut operations = Vec::new();
    let mut counts = Vec::new();
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
        assert_eq!(spans.len(), 1, "one span per batch: {spans:?}");
        let name = spans[0]["name"].as_str().expect("a span name");
        operations.push(if name.starts_with("component.memory.write.") {
            "write"
        } else if name.starts_with("component.memory.search.") {
            "search"
        } else {
            panic!("unexpected memory span name: {name}");
        });
        counts.push(metadata_of(&spans[0])["count"].clone());
    }
    assert_eq!(
        operations,
        ["write", "search"],
        "both batches were exported, in emission order",
    );
    assert_eq!(
        counts,
        [json!(1), json!(hits.len())],
        "the spans report the records the facade wrote and the hits it returned",
    );
    assert!(
        !handle.is_active(),
        "shutdown is still clean after failures"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_flushes_export_two_ordered_batches_with_the_same_team_trace() {
    let server = accepting_collector(2).await;
    let handle = init_otel_tracer(otlp_config(&server)).expect("OTLP/HTTP initializes offline");
    let (_sink, observer) = otlp_bridge(&handle);
    let memory = observed_facade(observer);
    let call = CallOptions::new();

    // One flush per operation, so the batch processor drains a single span each
    // time and the two batches keep the emission order.
    memory
        .remember(
            NewTeamMemory::decision(TEAM, DECISION).with_member("alice"),
            &call,
        )
        .await
        .expect("the decision record is remembered");
    handle.force_flush();
    memory
        .search(
            &TeamMemoryScope::member(TEAM, "alice"),
            "migration",
            &TeamSearchOptions::default(),
        )
        .await
        .expect("the search runs");
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
    let team_trace = assert_memory_span(earlier, "write");
    assert_eq!(
        assert_memory_span(later, "search"),
        team_trace,
        "both batches keep the bridge's derived team trace identity",
    );
    assert_eq!(
        metadata_of(earlier)["team_id"],
        metadata_of(later)["team_id"],
    );
    assert!(
        nanos(&later["startTimeUnixNano"]) >= nanos(&earlier["startTimeUnixNano"]),
        "the emission order is preserved across batches: {earlier} then {later}",
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_untraced_wiring_never_reaches_the_collector() {
    // A collector that would fail the test if anything arrived: zero requests is
    // the assertion, so nothing is expected and nothing must be recorded.
    let server = MockServer::builder().start().await;
    let handle = init_otel_tracer(otlp_config(&server)).expect("OTLP/HTTP initializes offline");
    let (_sink, observer) = otlp_bridge(&handle);

    // `trace: false` is the last word: the bridge is supplied but never attached.
    let dir = tempfile::tempdir().expect("a temp workspace for the team document");
    let config = MemoryConfig {
        trace: false,
        ..MemoryConfig::default()
    };
    let wiring = TeamMemoryWiring::open_with_observer(TEAM, config, dir.path(), observer)
        .await
        .expect("the untraced wiring opens");
    assert!(!wiring.trace_enabled(), "tracing is switched off");
    assert!(
        wiring.observer().is_none(),
        "an untraced wiring attaches no observer, not even the supplied bridge",
    );

    // The operations themselves still run (and still store records); only the
    // observation is gone.
    run_hook_lifecycle(&wiring).await;
    let hits = wiring
        .memory()
        .search(
            &TeamMemoryScope::member(TEAM, "alice"),
            "SQLite",
            &TeamSearchOptions::default(),
        )
        .await
        .expect("the untraced facade still serves a search");
    assert_eq!(
        hits.len(),
        1,
        "the flush stored the decision record: {hits:?}"
    );

    handle.force_flush();
    let requests = within("the collector stays silent", server.received_requests())
        .await
        .expect("the mock records requests");
    assert!(
        requests.is_empty(),
        "an observer-less run exports nothing: {} request(s)",
        requests.len(),
    );

    handle.shutdown();
}
