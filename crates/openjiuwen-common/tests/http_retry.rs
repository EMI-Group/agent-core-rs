//! Integration tests for the shared HTTP transport.
//!
//! Port of `tests/unit_tests/core/common/clients/test_http_client.py`, extended
//! with the retry, streaming, and cancellation behaviour the Rust transport adds.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::Duration;

use futures::StreamExt;
use openjiuwen_common::error::{Error, ErrorKind};
use openjiuwen_common::http::{HttpClient, HttpClientConfig, HttpRequest, RetryPolicy};
use serde::Deserialize;
use serde_json::json;
use tokio_util::sync::CancellationToken;
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[derive(Debug, Deserialize)]
struct Payload {
    value: u32,
}

fn client_with(config: HttpClientConfig) -> HttpClient {
    HttpClient::new(config).unwrap()
}

#[tokio::test]
async fn get_returns_status_body_and_final_url() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/hello"))
        .respond_with(ResponseTemplate::new(200).set_body_string("hello"))
        .expect(1)
        .mount(&server)
        .await;

    let url = format!("{}/hello", server.uri());
    let response = HttpClient::with_defaults()
        .unwrap()
        .get(&url)
        .await
        .unwrap();

    assert_eq!(response.status, 200);
    assert!(response.is_success());
    assert_eq!(response.text(), "hello");
    assert_eq!(response.url, url);
}

#[tokio::test]
async fn get_json_deserializes_the_body() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/payload"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(r#"{"value": 42}"#, "application/json"),
        )
        .mount(&server)
        .await;

    let payload: Payload = HttpClient::with_defaults()
        .unwrap()
        .get_json(format!("{}/payload", server.uri()))
        .await
        .unwrap();
    assert_eq!(payload.value, 42);
}

#[tokio::test]
async fn non_success_status_is_a_normal_response() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/missing"))
        .respond_with(ResponseTemplate::new(404).set_body_string("missing"))
        .mount(&server)
        .await;

    let response = HttpClient::with_defaults()
        .unwrap()
        .get(format!("{}/missing", server.uri()))
        .await
        .unwrap();

    assert_eq!(response.status, 404);
    assert!(!response.is_success());
    let error = response.error_for_status().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Transport);
}

#[tokio::test]
async fn post_json_sends_the_serialized_body_and_content_type() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/echo"))
        .and(header("content-type", "application/json"))
        .and(body_json(json!({"name": "jiuwen"})))
        .respond_with(ResponseTemplate::new(201).set_body_string("created"))
        .expect(1)
        .mount(&server)
        .await;

    let response = HttpClient::with_defaults()
        .unwrap()
        .post_json(format!("{}/echo", server.uri()), &json!({"name": "jiuwen"}))
        .await
        .unwrap();

    assert_eq!(response.status, 201);
    assert_eq!(response.text(), "created");
}

#[tokio::test]
async fn retries_a_retryable_status_then_succeeds() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/flaky"))
        .respond_with(ResponseTemplate::new(500))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/flaky"))
        .respond_with(ResponseTemplate::new(200).set_body_string("recovered"))
        .mount(&server)
        .await;

    let client = client_with(
        HttpClientConfig::default()
            .with_max_retries(2)
            .with_retry_backoff(Duration::from_millis(1), Duration::from_millis(5)),
    );
    let response = client.get(format!("{}/flaky", server.uri())).await.unwrap();

    assert!(response.is_success());
    assert_eq!(response.text(), "recovered");
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 2);
}

#[tokio::test]
async fn does_not_retry_when_max_retries_is_zero() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/broken"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;

    let client = client_with(HttpClientConfig::default().with_max_retries(0));
    let response = client
        .get(format!("{}/broken", server.uri()))
        .await
        .unwrap();

    assert_eq!(response.status, 500);
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
}

#[test]
fn retry_policy_retries_transport_errors_but_not_validation_errors() {
    let policy = RetryPolicy {
        max_retries: 1,
        ..RetryPolicy::default()
    };

    assert!(policy.should_retry(0, Some(&Error::transport("connection reset")), None));
    assert!(policy.should_retry(0, Some(&Error::timeout("deadline")), None));
    assert!(!policy.should_retry(0, Some(&Error::validation("bad input")), None));
    assert!(!policy.should_retry(1, Some(&Error::transport("connection reset")), None));
}

#[tokio::test]
async fn timeout_maps_to_the_timeout_error_kind() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/slow"))
        .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_millis(500)))
        .mount(&server)
        .await;

    let client = client_with(
        HttpClientConfig::default()
            .with_timeout(Duration::from_millis(100))
            .with_max_retries(0),
    );
    let error = client
        .get(format!("{}/slow", server.uri()))
        .await
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Timeout);
}

#[tokio::test]
async fn stream_collects_the_full_body() {
    let server = MockServer::start().await;
    let body = "streamed response body";
    Mock::given(method("GET"))
        .and(path("/stream"))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
        .mount(&server)
        .await;

    let response = HttpClient::with_defaults()
        .unwrap()
        .stream(HttpRequest::get(format!("{}/stream", server.uri())))
        .await
        .unwrap();

    assert_eq!(response.status, 200);
    let mut stream = response.body;
    let mut collected = Vec::new();
    while let Some(chunk) = stream.next().await {
        collected.extend_from_slice(&chunk.unwrap());
    }
    assert_eq!(collected, body.as_bytes());
}

#[tokio::test]
async fn execute_cancellable_stops_on_cancellation() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/slow"))
        .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_millis(500)))
        .mount(&server)
        .await;

    let token = CancellationToken::new();
    let trigger = token.clone();
    let canceller = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(10)).await;
        trigger.cancel();
    });

    let error = HttpClient::with_defaults()
        .unwrap()
        .execute_cancellable(HttpRequest::get(format!("{}/slow", server.uri())), token)
        .await
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Cancelled);
    canceller.await.unwrap();
}

#[tokio::test]
async fn configuration_headers_drop_authorization_but_send_custom_headers() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/headers"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;

    let client = client_with(
        HttpClientConfig::default()
            .with_header("Authorization", "Bearer must-not-leak")
            .with_header("X-Custom", "custom-value"),
    );
    let response = client
        .get(format!("{}/headers", server.uri()))
        .await
        .unwrap();
    assert!(response.is_success());

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    let headers = &requests[0].headers;
    assert!(headers.get("authorization").is_none());
    assert_eq!(
        headers
            .get("x-custom")
            .and_then(|value| value.to_str().ok()),
        Some("custom-value")
    );
}
