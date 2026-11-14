//! Crate-boundary tests of the Grafana Cloud and Langfuse integration presets.
//!
//! Python counterparts: `tests/unit_tests/extensions/tracer_otel/test_setup.py`
//! for the exporter wiring, plus the vendor presets that the Python SDK leaves
//! to the host (`grafana` / `langfuse` configuration objects). The Python suite
//! stops at "a configuration was produced"; these tests drive the presets
//! through the public `openjiuwen_extensions::otel` surface against a loopback
//! `wiremock` server, so the gateway path, the HTTP Basic credentials, and the
//! ingestion payload are pinned as they go over the wire.
//!
//! Build coverage:
//!
//! - the preset *shapes* (`endpoint`, `authorization`, `otel_config`, `Debug`
//!   redaction, `from_env`, the local Grafana Agent preset) compile and are
//!   asserted in **every** build, and
//! - the Langfuse REST ingestion client is plain `reqwest`, so its suites also
//!   run in a default build;
//! - everything that needs a live provider (`TracerHandle::tracer`,
//!   `GrafanaCloudConfig::exporter`, `LangfuseConfig::exporter`) is gated with
//!   `#[cfg(feature = "otel")]`.
//!
//! Everything is offline and deterministic: the only hosts are `127.0.0.1`,
//! environment mutation is serialized through [`ENV_LOCK`] and undone on drop.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::{Mutex, MutexGuard, PoisonError};

use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use serde_json::{json, Value};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use openjiuwen_common::error::ErrorKind;
use openjiuwen_extensions::otel::grafana::{
    GrafanaAgentConfig, GrafanaCloudConfig, DEFAULT_LOCAL_OTLP_ENDPOINT, GRAFANA_API_TOKEN,
    GRAFANA_INSTANCE_ID, GRAFANA_OTLP_ENDPOINT, GRAFANA_OTLP_ZONE, GRAFANA_ZONE,
};
use openjiuwen_extensions::otel::langfuse::{
    GenerationBody, IngestionBatch, IngestionEvent, IngestionSummary, LangfuseClient,
    LangfuseConfig, SpanBody, TraceBody, LANGFUSE_BASEURL, LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
};
use openjiuwen_extensions::otel::{ExporterType, OtlpProtocol};

/// Serialises the tests that read or write the preset environment.
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Every variable the two presets read.
const ENV_VARS: [&str; 9] = [
    GRAFANA_INSTANCE_ID,
    GRAFANA_API_TOKEN,
    GRAFANA_ZONE,
    GRAFANA_OTLP_ZONE,
    GRAFANA_OTLP_ENDPOINT,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    LANGFUSE_HOST,
    LANGFUSE_BASEURL,
];

/// Holds [`ENV_LOCK`] for one test and keeps the preset variables clean.
struct EnvGuard {
    /// The process-global lock; released when the guard drops.
    _lock: MutexGuard<'static, ()>,
}

impl EnvGuard {
    /// Acquires the lock, recovering it when another test panicked.
    fn new() -> Self {
        let lock = ENV_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        clear_env();
        Self { _lock: lock }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        clear_env();
    }
}

/// Removes every variable the presets read.
fn clear_env() {
    for name in ENV_VARS {
        std::env::remove_var(name);
    }
}

/// Sets one variable for the rest of the test.
fn set_env(name: &str, value: &str) {
    std::env::set_var(name, value);
}

/// The `Authorization` value Langfuse expects for the test keys.
fn langfuse_authorization() -> String {
    format!("Basic {}", STANDARD.encode("pk-lf-test:sk-lf-test"))
}

/// The `Authorization` value Grafana Cloud expects for the test stack.
fn grafana_authorization() -> String {
    format!("Basic {}", STANDARD.encode("123:glc-token"))
}

/// A client pointed at the mock server, built the way a host builds it.
fn client_for(server: &MockServer) -> LangfuseClient {
    LangfuseClient::new(LangfuseConfig::new("pk-lf-test", "sk-lf-test").with_host(server.uri()))
        .expect("the preset addresses the mock host")
}

/// One event of each kind, wired to a single trace.
fn sample_batch() -> IngestionBatch {
    let mut batch = IngestionBatch::new();
    batch.push(
        IngestionEvent::trace(&TraceBody::new("checkout").with_user_id("user-1"))
            .expect("the trace body serializes"),
    );
    batch.push(
        IngestionEvent::span(
            &SpanBody::new("trace-1", "retrieve").with_parent_observation_id("obs-1"),
        )
        .expect("the span body serializes"),
    );
    batch.push(
        IngestionEvent::generation(&GenerationBody::new("trace-1", "chat").with_model("gpt-4o"))
            .expect("the generation body serializes"),
    );
    batch
}

#[test]
fn the_grafana_cloud_preset_derives_the_gateway_endpoint_and_credentials() {
    let preset = GrafanaCloudConfig::new("123", "glc-token", "prod-us-east-0");
    assert_eq!(
        preset.endpoint(),
        "https://otlp-gateway-prod-us-east-0.grafana.net/otlp",
    );
    assert_eq!(preset.authorization(), grafana_authorization());

    let config = preset.otel_config().expect("the preset is complete");
    assert_eq!(config.exporter_type, ExporterType::Otlp);
    assert_eq!(config.protocol, OtlpProtocol::Http);
    assert_eq!(
        config.exporter_endpoint.as_deref(),
        Some(preset.endpoint().as_str()),
    );
    assert_eq!(config.headers.len(), 1);
    assert_eq!(config.headers["authorization"], preset.authorization());
    assert_eq!(config.service_name, "openjiuwen");

    // The access-policy token never leaks through `Debug`.
    let rendered = format!("{preset:?}");
    assert!(rendered.contains("<redacted>"), "{rendered}");
    assert!(!rendered.contains("glc-token"), "{rendered}");
}

#[test]
fn the_grafana_agent_preset_targets_the_local_collector() {
    // A self-hosted Agent/Alloy deployment is an OTLP/HTTP collector on its
    // default port; the preset configures the push, not the collector lifecycle.
    let preset = GrafanaAgentConfig::default().with_service_name("checkout");
    let config = preset.otel_config().expect("the local default is complete");
    assert_eq!(config.exporter_type, ExporterType::Otlp);
    assert_eq!(config.protocol, OtlpProtocol::Http);
    assert_eq!(
        config.exporter_endpoint.as_deref(),
        Some(DEFAULT_LOCAL_OTLP_ENDPOINT),
    );
    assert_eq!(config.service_name, "checkout");
    assert!(config.headers.is_empty());

    // An explicit endpoint and header reach the configuration unchanged.
    let preset = GrafanaAgentConfig::new("http://127.0.0.1:12345").with_header("x-tenant", "acme");
    let config = preset
        .otel_config()
        .expect("an absolute endpoint is complete");
    assert_eq!(
        config.exporter_endpoint.as_deref(),
        Some("http://127.0.0.1:12345"),
    );
    assert_eq!(config.headers["x-tenant"], "acme");
    assert_eq!(config.service_name, "openjiuwen");

    let error = GrafanaAgentConfig::new("/v1/traces")
        .otel_config()
        .expect_err("a relative endpoint cannot be pushed to");
    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[tokio::test]
async fn the_langfuse_client_posts_one_ingestion_request() {
    let server = MockServer::start().await;
    let authorization = langfuse_authorization();
    Mock::given(method("POST"))
        .and(path("/api/public/ingestion"))
        .and(header("content-type", "application/json"))
        .and(header("authorization", authorization.as_str()))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "successes": [{"id": "event-1", "status": 201}],
            "errors": [],
        })))
        .expect(1)
        .mount(&server)
        .await;

    let summary = client_for(&server)
        .ingest(sample_batch())
        .await
        .expect("every event landed");
    assert_eq!(
        summary,
        IngestionSummary {
            events: 3,
            requests: 1,
        }
    );

    let requests = server
        .received_requests()
        .await
        .expect("the mock records requests");
    assert_eq!(requests.len(), 1, "one request per batch");
    let request = &requests[0];
    assert_eq!(request.url.path(), "/api/public/ingestion");
    assert_eq!(
        request
            .headers
            .get("authorization")
            .expect("the Basic header")
            .to_str()
            .expect("an ASCII header"),
        authorization,
    );

    let body: Value = serde_json::from_slice(&request.body).expect("the body is JSON");
    let events = body["batch"].as_array().expect("a batch array");
    assert_eq!(events.len(), 3);
    assert_eq!(events[0]["type"], json!("trace-create"));
    assert_eq!(events[1]["type"], json!("span-create"));
    assert_eq!(events[2]["type"], json!("generation-create"));
    assert_eq!(events[0]["body"]["name"], json!("checkout"));
}

#[tokio::test]
async fn a_partial_ingestion_failure_names_the_failing_event() {
    // Langfuse ingestion is partial: a 207 can still carry per-event failures.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/public/ingestion"))
        .respond_with(ResponseTemplate::new(207).set_body_json(json!({
            "successes": [],
            "errors": [{"id": "e1", "status": 400, "message": "bad event", "error": "invalid"}],
        })))
        .mount(&server)
        .await;

    let error = client_for(&server)
        .ingest(sample_batch())
        .await
        .expect_err("a reported event failure is an error");
    let message = error.message();
    assert!(message.contains("e1"), "the id is named: {message}");
    assert!(message.contains("400"), "the status is named: {message}");
    assert!(
        message.contains("bad event"),
        "the message is named: {message}"
    );
    assert!(
        message.contains("207"),
        "the HTTP status is named: {message}"
    );
}

#[tokio::test]
async fn a_rejected_ingestion_request_is_an_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/public/ingestion"))
        .respond_with(ResponseTemplate::new(500).set_body_string("langfuse exploded"))
        .mount(&server)
        .await;

    let error = client_for(&server)
        .ingest(sample_batch())
        .await
        .expect_err("HTTP 500 is an error");
    let message = error.message();
    assert!(message.contains("500"), "{message}");
    assert!(message.contains("langfuse exploded"), "{message}");
}

#[tokio::test]
async fn the_client_splits_a_batch_at_the_configured_size() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/public/ingestion"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({"successes": [], "errors": []})),
        )
        .expect(3)
        .mount(&server)
        .await;

    let client = client_for(&server).with_max_batch_size(1);
    let summary = client
        .ingest(sample_batch())
        .await
        .expect("every event landed");
    assert_eq!(
        summary,
        IngestionSummary {
            events: 3,
            requests: 3,
        }
    );

    let requests = server
        .received_requests()
        .await
        .expect("the mock records requests");
    assert_eq!(requests.len(), 3, "one request per event");
    for request in &requests {
        let body: Value = serde_json::from_slice(&request.body).expect("the body is JSON");
        assert_eq!(body["batch"].as_array().expect("a batch array").len(), 1);
    }
}

#[test]
fn the_grafana_cloud_preset_reads_the_environment() {
    let _environment = EnvGuard::new();

    let error = GrafanaCloudConfig::from_env().expect_err("no credentials in the environment");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains(GRAFANA_INSTANCE_ID), "{error}");

    set_env(GRAFANA_INSTANCE_ID, "123");
    set_env(GRAFANA_API_TOKEN, "glc-token");
    set_env(GRAFANA_ZONE, "prod-us-east-0");
    let preset = GrafanaCloudConfig::from_env().expect("the credentials are complete");
    assert_eq!(
        preset.endpoint(),
        "https://otlp-gateway-prod-us-east-0.grafana.net/otlp",
    );
    assert_eq!(preset.authorization(), grafana_authorization());

    std::env::remove_var(GRAFANA_API_TOKEN);
    let error = GrafanaCloudConfig::from_env().expect_err("a token is required");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains(GRAFANA_API_TOKEN), "{error}");

    set_env(GRAFANA_API_TOKEN, "glc-token");
    std::env::remove_var(GRAFANA_ZONE);
    let error = GrafanaCloudConfig::from_env().expect_err("a zone or endpoint is required");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains(GRAFANA_ZONE), "{error}");
}

#[test]
fn the_langfuse_preset_reads_the_environment() {
    let _environment = EnvGuard::new();

    let error = LangfuseConfig::from_env().expect_err("no keys in the environment");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains(LANGFUSE_PUBLIC_KEY), "{error}");

    set_env(LANGFUSE_PUBLIC_KEY, "pk-lf-test");
    set_env(LANGFUSE_SECRET_KEY, "sk-lf-test");
    set_env(LANGFUSE_HOST, "https://langfuse.internal/");
    let preset = LangfuseConfig::from_env().expect("the keys are complete");
    assert_eq!(
        preset.endpoint(),
        "https://langfuse.internal/api/public/otel",
    );
    assert_eq!(
        preset.ingestion_url(),
        "https://langfuse.internal/api/public/ingestion",
    );

    // The Python-SDK alias supplies the host when `LANGFUSE_HOST` is unset.
    std::env::remove_var(LANGFUSE_HOST);
    set_env(LANGFUSE_BASEURL, "http://127.0.0.1:8080");
    let preset = LangfuseConfig::from_env().expect("the alias supplies the host");
    assert_eq!(preset.endpoint(), "http://127.0.0.1:8080/api/public/otel",);

    std::env::remove_var(LANGFUSE_SECRET_KEY);
    let error = LangfuseConfig::from_env().expect_err("a secret key is required");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains(LANGFUSE_SECRET_KEY), "{error}");
}

#[cfg(feature = "otel")]
#[test]
fn the_grafana_cloud_preset_posts_to_the_gateway_path() {
    use opentelemetry::trace::{Span as _, Tracer as _};

    let runtime = tokio::runtime::Runtime::new().expect("a tokio runtime starts");
    let server = runtime.block_on(MockServer::builder().start());
    let authorization = grafana_authorization();
    runtime.block_on(
        Mock::given(method("POST"))
            .and(path("/otlp/v1/traces"))
            .and(header("authorization", authorization.as_str()))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server),
    );

    // The documented override points the preset at the loopback gateway; the
    // real preset appends `/v1/traces` to whatever endpoint it is given.
    let preset = GrafanaCloudConfig::new("123", "glc-token", "prod-us-east-0")
        .with_otlp_endpoint(format!("{}/otlp", server.uri()));
    {
        let exporter = preset.exporter().expect("the exporter builds without I/O");
        assert_eq!(
            exporter.traces_url(),
            format!("{}/otlp/v1/traces", server.uri()),
        );
    }

    let handle = openjiuwen_extensions::otel::init_otel_tracer(
        preset.otel_config().expect("the preset is complete"),
    )
    .expect("OTLP/HTTP initializes offline");
    assert!(handle.is_active());
    handle
        .tracer()
        .expect("an active handle exposes a tracer")
        .start("grafana.span")
        .end();
    handle.force_flush();

    let requests = runtime
        .block_on(server.received_requests())
        .expect("the mock records requests");
    assert_eq!(requests.len(), 1, "one batch, one POST");
    assert_eq!(requests[0].url.path(), "/otlp/v1/traces");
    assert_eq!(
        requests[0]
            .headers
            .get("authorization")
            .expect("the Basic header")
            .to_str()
            .expect("an ASCII header"),
        authorization,
    );
    let body: Value = serde_json::from_slice(&requests[0].body).expect("the body is JSON");
    assert_eq!(
        body["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["name"],
        json!("grafana.span"),
    );
    handle.shutdown();
}

#[cfg(feature = "otel")]
#[test]
fn the_langfuse_otlp_preset_posts_to_the_otel_path() {
    use opentelemetry::trace::{Span as _, Tracer as _};

    let runtime = tokio::runtime::Runtime::new().expect("a tokio runtime starts");
    let server = runtime.block_on(MockServer::builder().start());
    let authorization = langfuse_authorization();
    runtime.block_on(
        Mock::given(method("POST"))
            .and(path("/api/public/otel/v1/traces"))
            .and(header("authorization", authorization.as_str()))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server),
    );

    let preset = LangfuseConfig::new("pk-lf-test", "sk-lf-test").with_host(server.uri());
    let otel = preset.otel_config().expect("the preset is complete");
    assert_eq!(
        otel.exporter_endpoint.as_deref(),
        Some(preset.endpoint().as_str()),
    );
    assert_eq!(otel.headers["authorization"], authorization);
    {
        let exporter = preset.exporter().expect("the exporter builds without I/O");
        assert_eq!(
            exporter.traces_url(),
            format!("{}/api/public/otel/v1/traces", server.uri()),
        );
    }

    let handle =
        openjiuwen_extensions::otel::init_otel_tracer(otel).expect("OTLP/HTTP initializes offline");
    assert!(handle.is_active());
    handle
        .tracer()
        .expect("an active handle exposes a tracer")
        .start("langfuse.span")
        .end();
    handle.force_flush();

    let requests = runtime
        .block_on(server.received_requests())
        .expect("the mock records requests");
    assert_eq!(requests.len(), 1, "one batch, one POST");
    assert_eq!(requests[0].url.path(), "/api/public/otel/v1/traces");
    assert_eq!(
        requests[0]
            .headers
            .get("authorization")
            .expect("the Basic header")
            .to_str()
            .expect("an ASCII header"),
        authorization,
    );
    handle.shutdown();
}
