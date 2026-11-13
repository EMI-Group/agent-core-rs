//! Unit tests of the Langfuse REST ingestion client.
//!
//! Langfuse's Python SDK asserts its ingestion payloads against a mocked
//! transport; the Rust port does the same with `wiremock`, so the request URL,
//! headers, and camelCase body — plus the partial-failure contract of the
//! ingestion API — are pinned without a live Langfuse instance.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use openjiuwen_common::error::ErrorKind;
use serde_json::{json, Value};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::{check_response, snippet, LangfuseClient};
use crate::otel::langfuse::{
    GenerationBody, IngestionBatch, IngestionEvent, IngestionSummary, LangfuseConfig, SpanBody,
    TraceBody,
};

/// Builds a client that points at the mock server.
fn client_for(server: &MockServer) -> LangfuseClient {
    LangfuseClient::new(LangfuseConfig::new("pk-lf-public", "sk-lf-secret").with_host(server.uri()))
        .unwrap()
}

/// The response Langfuse sends when every event was accepted.
fn accepted(template: ResponseTemplate) -> ResponseTemplate {
    template.set_body_json(json!({
        "successes": [{"id": "event-1", "status": 201}],
        "errors": [],
    }))
}

/// One event of each kind, wired to a single trace.
fn sample_events() -> Vec<IngestionEvent> {
    vec![
        IngestionEvent::trace(
            &TraceBody::new("checkout")
                .with_user_id("user-1")
                .with_session_id("session-1"),
        )
        .unwrap(),
        IngestionEvent::span(
            &SpanBody::new("trace-1", "retrieve")
                .with_start_time("2026-01-02T03:04:05.000Z")
                .with_end_time("2026-01-02T03:04:06.000Z")
                .with_parent_observation_id("obs-1"),
        )
        .unwrap(),
        IngestionEvent::generation(
            &GenerationBody::new("trace-1", "chat")
                .with_model("gpt-4o")
                .with_model_parameters(json!({"temperature": 0.2}))
                .with_usage_details(json!({"input": 3, "output": 5})),
        )
        .unwrap(),
    ]
}

#[tokio::test]
async fn ingest_posts_the_ingestion_document() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/public/ingestion"))
        .respond_with(accepted(ResponseTemplate::new(200)))
        .mount(&server)
        .await;

    let client = client_for(&server);
    let mut batch = IngestionBatch::new();
    batch.extend(sample_events());
    let summary = client.ingest(batch).await.unwrap();
    assert_eq!(
        summary,
        IngestionSummary {
            events: 3,
            requests: 1,
        }
    );

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1, "one request per batch");
    let request = &requests[0];
    assert_eq!(request.url.path(), "/api/public/ingestion");
    let content_type = request
        .headers
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap();
    assert!(
        content_type.starts_with("application/json"),
        "{content_type}"
    );
    assert_eq!(
        request
            .headers
            .get("authorization")
            .unwrap()
            .to_str()
            .unwrap(),
        format!("Basic {}", STANDARD.encode("pk-lf-public:sk-lf-secret")),
    );

    let body: Value = serde_json::from_slice(&request.body).unwrap();
    let events = body["batch"].as_array().unwrap();
    assert_eq!(events.len(), 3);
    assert_eq!(events[0]["type"], "trace-create");
    assert_eq!(events[1]["type"], "span-create");
    assert_eq!(events[2]["type"], "generation-create");
    for event in events {
        assert!(uuid::Uuid::parse_str(event["id"].as_str().unwrap()).is_ok());
        let timestamp = event["timestamp"].as_str().unwrap();
        assert!(timestamp.ends_with('Z'), "{timestamp}");
        assert!(timestamp.contains('T'), "{timestamp}");
    }
    // Bodies are camelCase and omit absent fields.
    assert_eq!(events[0]["body"]["userId"], "user-1");
    assert_eq!(events[0]["body"]["sessionId"], "session-1");
    assert!(events[0]["body"].get("release").is_none());
    assert_eq!(events[1]["body"]["traceId"], "trace-1");
    assert_eq!(events[1]["body"]["startTime"], "2026-01-02T03:04:05.000Z");
    assert_eq!(events[1]["body"]["parentObservationId"], "obs-1");
    assert_eq!(events[2]["body"]["modelParameters"]["temperature"], 0.2);
    assert_eq!(events[2]["body"]["usageDetails"]["output"], 5);
}

#[tokio::test]
async fn ingest_accepts_a_response_without_any_error_entry() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(accepted(ResponseTemplate::new(207)))
        .mount(&server)
        .await;

    let client = client_for(&server);
    let summary = client.ingest_events(sample_events()).await.unwrap();
    assert_eq!(summary.events, 3);
    assert_eq!(summary.requests, 1);
}

#[tokio::test]
async fn ingest_treats_partial_failures_as_an_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(207).set_body_json(json!({
            "successes": [{"id": "event-1", "status": 201}],
            "errors": [
                {
                    "id": "event-7",
                    "status": 400,
                    "message": "traceId is invalid",
                    "error": "invalid_request",
                },
            ],
        })))
        .mount(&server)
        .await;

    let error = client_for(&server)
        .ingest_events(sample_events())
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Tool);
    let message = error.message();
    assert!(message.contains("event-7"), "{message}");
    assert!(message.contains("400"), "{message}");
    assert!(message.contains("traceId is invalid"), "{message}");
    assert!(message.contains("207"), "{message}");
}

#[tokio::test]
async fn ingest_reports_a_server_error_with_the_status_and_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(500).set_body_string("upstream exploded"))
        .mount(&server)
        .await;

    let error = client_for(&server)
        .ingest_events(sample_events())
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Tool);
    let message = error.message();
    assert!(message.contains("500"), "{message}");
    assert!(message.contains("upstream exploded"), "{message}");
}

#[tokio::test]
async fn ingest_rejects_a_success_status_with_a_non_json_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string("<html>gateway</html>"))
        .mount(&server)
        .await;

    let error = client_for(&server)
        .ingest_events(sample_events())
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Tool);
    assert!(error.message().contains("not JSON"), "{error}");
    assert!(error.message().contains("gateway"), "{error}");
}

#[tokio::test]
async fn ingest_chunks_the_batch_by_max_batch_size() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(accepted(ResponseTemplate::new(200)))
        .mount(&server)
        .await;

    let client = client_for(&server).with_max_batch_size(2);
    // Five events of one kind keep the chunk boundaries easy to assert.
    let events: Vec<IngestionEvent> = (0..5)
        .map(|index| IngestionEvent::trace(&TraceBody::new(format!("trace-{index}"))).unwrap())
        .collect();
    let summary = client.ingest_events(events).await.unwrap();
    assert_eq!(
        summary,
        IngestionSummary {
            events: 5,
            requests: 3,
        }
    );

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 3);
    let mut sizes = Vec::new();
    for request in &requests {
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        sizes.push(body["batch"].as_array().unwrap().len());
    }
    sizes.sort_unstable();
    assert_eq!(sizes, [1, 2, 2]);
}

#[tokio::test]
async fn an_empty_batch_never_reaches_the_network() {
    let server = MockServer::start().await;
    let client = client_for(&server);

    let summary = client.ingest(IngestionBatch::new()).await.unwrap();
    assert_eq!(summary, IngestionSummary::default());
    assert_eq!(client.ingest_events(Vec::new()).await.unwrap().requests, 0);
    assert!(server
        .received_requests()
        .await
        .unwrap_or_default()
        .is_empty());
}

#[tokio::test]
async fn a_zero_batch_size_still_sends_one_event_per_request() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(accepted(ResponseTemplate::new(200)))
        .mount(&server)
        .await;

    let client = client_for(&server).with_max_batch_size(0);
    assert_eq!(client.max_batch_size(), 1);
    let events = vec![
        IngestionEvent::trace(&TraceBody::new("one")).unwrap(),
        IngestionEvent::trace(&TraceBody::new("two")).unwrap(),
    ];
    let summary = client.ingest_events(events).await.unwrap();
    assert_eq!(
        summary,
        IngestionSummary {
            events: 2,
            requests: 2,
        }
    );
}

#[test]
fn a_malformed_errors_field_is_an_error() {
    let error = check_response(
        reqwest::StatusCode::OK,
        r#"{"successes": [], "errors": "nope"}"#,
    )
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Tool);
    assert!(error.message().contains("malformed"), "{error}");
}

#[test]
fn a_missing_errors_field_is_a_success() {
    check_response(reqwest::StatusCode::OK, r#"{"successes": []}"#).unwrap();
    check_response(reqwest::StatusCode::OK, "{}").unwrap();
}

#[test]
fn failure_descriptions_name_ids_statuses_and_messages() {
    let error = check_response(
        reqwest::StatusCode::MULTIPLE_CHOICES,
        r#"{"errors":[{"id":"a","status":400,"message":"bad traceId"},{"id":"b","status":500,"message":"boom"}]}"#,
    )
    .unwrap_err();
    let message = error.message();
    assert!(message.contains("2 failed event(s)"), "{message}");
    assert!(
        message.contains("id=a status=400 message=bad traceId"),
        "{message}"
    );
    assert!(
        message.contains("id=b status=500 message=boom"),
        "{message}"
    );
}

#[test]
fn snippets_are_bounded_on_a_character_boundary() {
    assert_eq!(snippet("  short  "), "short");
    let long = "语".repeat(600);
    let shortened = snippet(&long);
    assert!(shortened.ends_with('…'), "{shortened}");
    assert_eq!(shortened.chars().count(), 513);
}
