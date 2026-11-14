//! Offline tests for the remote experience-hub backend: the hub protocol v1
//! wire shapes, the local-backend normalisation and the documented error mapping.
//!
//! Every test drives a real loopback `wiremock` server, so the transport, the
//! retry budget and the JSON encoding are exercised rather than stubbed. No test
//! reads a clock for ordering and none reaches the network: the only URLs are
//! ephemeral `127.0.0.1` ports.
//!
//! These live beside the module rather than in `tests/` because the hub backend
//! needs nothing the crate's own unit-test target cannot provide — `wiremock`
//! arrives through the `openjiuwen-test-support` dev-dependency.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::assert_is_empty
)]

use std::time::Duration;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_test_support::wiremock::matchers::{body_json, header, method, path};
use openjiuwen_test_support::wiremock::{Mock, MockServer, Request, ResponseTemplate};
use openjiuwen_test_support::EnvGuard;
use serde_json::json;

use super::*;
use crate::experience::{ExperienceRecord, ExperienceSection};
use crate::sharing::{ShareStager, SharingMeta};

/// The upper bound every hub call must finish within.
const BOUND: Duration = Duration::from_secs(10);
/// The skill every fixture record belongs to.
const SKILL: &str = "pdf-report";
/// A record about the canonical demo skill.
const CONTENT_A: &str = "raise the read timeout before parsing large pdf files";
/// The bearer token the auth tests configure.
const KEY: &str = "hub-secret";

/// A record with a fixed skill, keywords and confidence.
fn record(id: &str, content: &str, confidence: f64) -> ExperienceRecord {
    ExperienceRecord::new(id, SKILL, ExperienceSection::Tool, content)
        .unwrap()
        .with_keywords(["pdf", "timeout"])
        .with_confidence(confidence)
        .unwrap()
}

/// Metadata for the demo skill.
fn meta() -> SharingMeta {
    SharingMeta::new(SKILL, "1.2.0")
}

/// The single-record fixture the publish tests send.
fn records() -> Vec<ExperienceRecord> {
    vec![record("r-1", CONTENT_A, 0.9)]
}

/// A stager that publishes without approval.
fn stager() -> ShareStager {
    ShareStager::default().with_require_approval(false)
}

/// The bundle [`stager`] assembles from [`records`], with its aggregates filled.
fn published_bundle() -> SharedSkillBundle {
    stager().build_bundle(&records(), &meta()).unwrap().bundle
}

/// A bundle carrying `bundle_id` and no experiences.
fn named_bundle(bundle_id: &str, skill_id: &str) -> SharedSkillBundle {
    let mut bundle = SharedSkillBundle::new(skill_id, skill_id, "1.0", vec![]).unwrap();
    bundle.bundle_id = bundle_id.to_owned();
    bundle
}

/// A backend for `server` with the default options.
fn backend(server: &MockServer) -> RemoteHubSharingBackend {
    RemoteHubSharingBackend::new(RemoteHubOptions::new(server.uri())).unwrap()
}

/// A backend for `server` with `configure` applied to the default options.
fn backend_with(
    server: &MockServer,
    configure: impl FnOnce(RemoteHubOptions) -> RemoteHubOptions,
) -> RemoteHubSharingBackend {
    RemoteHubSharingBackend::new(configure(RemoteHubOptions::new(server.uri()))).unwrap()
}

/// Awaits `future` within [`BOUND`].
async fn bounded<T>(future: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(BOUND, future)
        .await
        .expect("the hub call must finish within the bound")
}

/// Every request the mock server received, in arrival order.
async fn received(server: &MockServer) -> Vec<Request> {
    server.received_requests().await.unwrap()
}

/// The error a hub call reports, asserting it is an error at all.
fn error_of<T: std::fmt::Debug>(result: Result<T>) -> Error {
    result.expect_err("the hub call must fail")
}

#[tokio::test]
async fn publish_posts_the_bundle_with_bearer_auth_and_extra_headers() {
    let server = MockServer::start().await;
    let bundle = published_bundle();
    Mock::given(method("POST"))
        .and(path("/v1/hub/bundles"))
        .and(header("authorization", format!("Bearer {KEY}")))
        .and(header("x-org", "team-a"))
        .and(body_json(&bundle))
        .respond_with(ResponseTemplate::new(201).set_body_json(json!({"bundle_id": "hub-9"})))
        .expect(1)
        .mount(&server)
        .await;

    let backend = backend_with(&server, |options| {
        options.with_api_key(KEY).with_header("x-org", "team-a")
    });
    let published = bounded(backend.publish(&bundle)).await.unwrap();

    assert_eq!(published, "hub-9");
    let requests = received(&server).await;
    assert_eq!(requests.len(), 1);
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&requests[0].body).unwrap(),
        serde_json::to_value(&bundle).unwrap()
    );
}

#[tokio::test]
async fn publish_falls_back_to_the_bundle_id_when_the_hub_echoes_a_blank_one() {
    let server = MockServer::start().await;
    let bundle = published_bundle();
    Mock::given(method("POST"))
        .and(path("/v1/hub/bundles"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"bundle_id": "  "})))
        .expect(1)
        .mount(&server)
        .await;

    let published = bounded(backend(&server).publish(&bundle)).await.unwrap();

    assert_eq!(published, bundle.bundle_id);
}

#[tokio::test]
async fn publish_falls_back_to_the_bundle_id_when_the_reply_omits_it() {
    let server = MockServer::start().await;
    let bundle = published_bundle();
    Mock::given(method("POST"))
        .and(path("/v1/hub/bundles"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .expect(1)
        .mount(&server)
        .await;

    let published = bounded(backend(&server).publish(&bundle)).await.unwrap();

    assert_eq!(published, bundle.bundle_id);
}

#[tokio::test]
async fn list_fetches_every_bundle_and_sorts_them_by_id() {
    let server = MockServer::start().await;
    let second = named_bundle("bundle-b", "images");
    let first = named_bundle("bundle-a", "pdf-report");
    Mock::given(method("GET"))
        .and(path("/v1/hub/bundles"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"bundles": [second.clone(), first.clone()]})),
        )
        .expect(1)
        .mount(&server)
        .await;

    let listed = bounded(backend(&server).list()).await.unwrap();

    let ids: Vec<String> = listed
        .iter()
        .map(|bundle| bundle.bundle_id.clone())
        .collect();
    assert_eq!(ids, vec!["bundle-a", "bundle-b"]);
    assert_eq!(listed, vec![first, second]);
}

#[tokio::test]
async fn get_returns_a_bundle_and_maps_404_to_none() {
    let server = MockServer::start().await;
    let bundle = named_bundle("bundle-a", "pdf-report");
    Mock::given(method("GET"))
        .and(path("/v1/hub/bundles/bundle-a"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&bundle))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/hub/bundles/bundle-missing"))
        .respond_with(ResponseTemplate::new(404).set_body_json(
            json!({"error": {"code": "bundle_not_found", "message": "no such bundle"}}),
        ))
        .expect(1)
        .mount(&server)
        .await;

    let client = backend(&server);
    assert_eq!(bounded(client.get("bundle-a")).await.unwrap(), Some(bundle));
    assert_eq!(bounded(client.get("bundle-missing")).await.unwrap(), None);
}

#[tokio::test]
async fn bundle_ids_are_percent_encoded_in_the_path() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/hub/bundles/bundle%20with%20space"))
        .respond_with(ResponseTemplate::new(404))
        .expect(1)
        .mount(&server)
        .await;

    let found = bounded(backend(&server).get("bundle with space"))
        .await
        .unwrap();

    assert_eq!(found, None);
}

#[tokio::test]
async fn remove_reports_whether_the_bundle_existed() {
    let server = MockServer::start().await;
    Mock::given(method("DELETE"))
        .and(path("/v1/hub/bundles/bundle-a"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"removed": true})))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .and(path("/v1/hub/bundles/bundle-b"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"removed": false})))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .and(path("/v1/hub/bundles/bundle-missing"))
        .respond_with(ResponseTemplate::new(404))
        .expect(1)
        .mount(&server)
        .await;

    let client = backend(&server);
    assert!(bounded(client.remove("bundle-a")).await.unwrap());
    assert!(!bounded(client.remove("bundle-b")).await.unwrap());
    assert!(!bounded(client.remove("bundle-missing")).await.unwrap());
}

#[tokio::test]
async fn search_posts_the_query_and_filters_sorts_and_truncates() {
    let server = MockServer::start().await;
    let query = QueryKeywords::new(vec!["pdf".to_owned()], "parse pdf reports", "");
    Mock::given(method("POST"))
        .and(path("/v1/hub/bundles/search"))
        .and(body_json(json!({"query": &query, "limit": 2})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "results": [
                {"bundle_id": "bundle-c", "score": 0.4},
                {"bundle_id": "bundle-a", "score": 0.9},
                {"bundle_id": "bundle-z", "score": 0.0},
                {"bundle_id": "bundle-b", "score": 0.4},
            ]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let results = bounded(backend(&server).search(&query, 2)).await.unwrap();

    let ids: Vec<String> = results
        .iter()
        .map(|result| result.bundle_id.clone())
        .collect();
    // `bundle-z` scores 0.0 and is dropped; the 0.4 tie breaks by id.
    assert_eq!(ids, vec!["bundle-a", "bundle-b"]);
    assert!(results
        .iter()
        .all(|result: &SkillSearchResult| result.score > 0.0));
}

#[tokio::test]
async fn search_limit_zero_is_unlimited() {
    let server = MockServer::start().await;
    let query = QueryKeywords::new(vec!["pdf".to_owned()], "", "");
    Mock::given(method("POST"))
        .and(path("/v1/hub/bundles/search"))
        .and(body_json(json!({"query": &query, "limit": 0})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "results": [
                {"bundle_id": "bundle-b", "score": 0.2},
                {"bundle_id": "bundle-a", "score": 0.8},
            ]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let results = bounded(backend(&server).search(&query, 0)).await.unwrap();

    let ids: Vec<String> = results
        .iter()
        .map(|result| result.bundle_id.clone())
        .collect();
    assert_eq!(ids, vec!["bundle-a", "bundle-b"]);
}

#[tokio::test]
async fn search_with_an_empty_query_makes_no_request() {
    let server = MockServer::start().await;
    let results = bounded(backend(&server).search(&QueryKeywords::default(), 5))
        .await
        .unwrap();

    assert!(results.is_empty());
    assert!(received(&server).await.is_empty());
}

#[tokio::test]
async fn a_client_error_maps_to_validation_quoting_the_hub_code() {
    let server = MockServer::start().await;
    let bundle = published_bundle();
    Mock::given(method("POST"))
        .and(path("/v1/hub/bundles"))
        .respond_with(ResponseTemplate::new(400).set_body_json(
            json!({"error": {"code": "bundle_invalid", "message": "bundle has no experiences"}}),
        ))
        .expect(1)
        .mount(&server)
        .await;

    let error = error_of(bounded(backend(&server).publish(&bundle)).await);

    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.message();
    assert!(
        message.contains("bundle_invalid: bundle has no experiences"),
        "{message}"
    );
    assert!(message.contains("POST /v1/hub/bundles"), "{message}");
    assert!(message.contains("HTTP 400"), "{message}");
}

#[tokio::test]
async fn a_plain_text_error_body_is_quoted_in_the_message() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/hub/bundles"))
        .respond_with(ResponseTemplate::new(403).set_body_string("forbidden\nby policy"))
        .expect(1)
        .mount(&server)
        .await;

    let error = error_of(bounded(backend(&server).list()).await);

    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.message().contains("forbidden by policy"),
        "{}",
        error.message()
    );
}

#[tokio::test]
async fn a_client_error_is_never_retried() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/hub/bundles"))
        .respond_with(ResponseTemplate::new(404))
        .expect(1)
        .mount(&server)
        .await;

    let backend = backend_with(&server, |options| options.with_max_retries(3));
    let error = error_of(bounded(backend.list()).await);

    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(received(&server).await.len(), 1);
}

#[tokio::test]
async fn a_server_error_is_retried_then_reported_as_transport() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/hub/bundles"))
        .respond_with(ResponseTemplate::new(503).set_body_json(
            json!({"error": {"code": "unavailable", "message": "hub is reindexing"}}),
        ))
        .expect(2)
        .mount(&server)
        .await;

    let backend = backend_with(&server, |options| options.with_max_retries(1));
    let error = error_of(bounded(backend.list()).await);

    assert_eq!(error.kind(), ErrorKind::Transport);
    assert!(error.message().contains("HTTP 503"), "{}", error.message());
    assert_eq!(received(&server).await.len(), 2);
}

#[tokio::test]
async fn a_deadline_maps_to_timeout() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/hub/bundles"))
        .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(2)))
        .mount(&server)
        .await;

    let backend = backend_with(&server, |options| {
        options
            .with_timeout(Duration::from_millis(50))
            .with_max_retries(0)
    });
    let error = error_of(bounded(backend.list()).await);

    assert_eq!(error.kind(), ErrorKind::Timeout);
}

#[tokio::test]
async fn a_malformed_body_is_a_validation_error_naming_the_endpoint() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/hub/bundles"))
        .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
        .expect(1)
        .mount(&server)
        .await;

    let error = error_of(bounded(backend(&server).list()).await);

    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.message();
    assert!(message.contains("GET /v1/hub/bundles"), "{message}");
    assert!(message.contains("malformed body"), "{message}");
}

#[tokio::test]
async fn health_checks_the_endpoint_and_normalises_the_base_url() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"status": "ok"})))
        .expect(1)
        .mount(&server)
        .await;

    let backend =
        RemoteHubSharingBackend::new(RemoteHubOptions::new(format!("{}/", server.uri()))).unwrap();
    bounded(backend.health()).await.unwrap();

    assert_eq!(backend.options().base_url, server.uri());
}

#[tokio::test]
async fn health_reports_an_unhealthy_hub_as_an_error() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(503))
        .expect(1)
        .mount(&server)
        .await;

    let backend = backend_with(&server, |options| options.with_max_retries(0));
    let error = error_of(bounded(backend.health()).await);

    assert_eq!(error.kind(), ErrorKind::Transport);
    assert!(
        error.message().contains("GET /health"),
        "{}",
        error.message()
    );
}

#[tokio::test]
async fn share_stager_publishes_through_the_remote_backend() {
    let server = MockServer::start().await;
    let records = records();
    let expected = stager().build_bundle(&records, &meta()).unwrap();
    Mock::given(method("POST"))
        .and(path("/v1/hub/bundles"))
        .and(header("authorization", format!("Bearer {KEY}")))
        .and(body_json(&expected.bundle))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"bundle_id": expected.bundle.bundle_id.clone()})),
        )
        .expect(1)
        .mount(&server)
        .await;

    let backend = backend_with(&server, |options| options.with_api_key(KEY));
    let published = bounded(stager().share(&backend, &records, &meta(), false))
        .await
        .unwrap();

    assert_eq!(published, Some(expected.bundle.bundle_id.clone()));
    let requests = received(&server).await;
    assert_eq!(requests.len(), 1);
    let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(body["bundle_id"], json!(expected.bundle.bundle_id));
    assert_eq!(body["experiences"].as_array().unwrap().len(), 1);
}

#[test]
fn an_invalid_base_url_is_rejected_before_any_request() {
    for raw in [
        "ftp://hub.example.com",
        "file:///tmp/hub",
        "hub.example.com",
        "https://",
        "",
        "   ",
    ] {
        let error = RemoteHubSharingBackend::new(RemoteHubOptions::new(raw))
            .expect_err("the base URL must be rejected");
        assert_eq!(error.kind(), ErrorKind::Validation, "{raw:?}");
    }
    let error = RemoteHubSharingBackend::from_url("ftp://hub.example.com").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("ftp"), "{}", error.message());
}

#[test]
fn base_url_validation_normalises_the_trailing_slash_and_keeps_a_path_prefix() {
    let mut options = RemoteHubOptions::new("  https://hub.example.com/api///  ");
    options.validate().unwrap();
    assert_eq!(options.base_url, "https://hub.example.com/api");

    let mut options = RemoteHubOptions::new("http://127.0.0.1:8080");
    options.validate().unwrap();
    assert_eq!(options.base_url, "http://127.0.0.1:8080");
}

#[test]
fn options_debug_redacts_the_key_and_the_header_values() {
    let options = RemoteHubOptions::new("https://hub.example.com")
        .with_api_key(KEY)
        .with_header("x-org", "team-a");
    let debug = format!("{options:?}");

    assert!(!debug.contains(KEY), "{debug}");
    assert!(!debug.contains("team-a"), "{debug}");
    assert!(debug.contains("x-org"), "{debug}");
    assert!(debug.contains("<REDACTED>"), "{debug}");
    assert!(debug.contains("https://hub.example.com"), "{debug}");
}

#[test]
fn from_env_reads_the_documented_variable_names() {
    let guard = EnvGuard::new(&[HUB_URL_ENV, HUB_API_KEY_ENV]);

    let error = RemoteHubOptions::from_env().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains(HUB_URL_ENV), "{}", error.message());

    guard.set(HUB_URL_ENV, "   ");
    assert!(RemoteHubOptions::from_env().is_err());

    guard.set(HUB_URL_ENV, "https://hub.example.com/");
    let options = RemoteHubOptions::from_env().unwrap();
    assert_eq!(options.base_url, "https://hub.example.com");
    assert_eq!(options.api_key, None);
    assert_eq!(options.timeout, Duration::from_secs(30));
    assert_eq!(options.max_retries, 2);

    guard.set(HUB_API_KEY_ENV, "  hub-secret  ");
    let options = RemoteHubOptions::from_env().unwrap();
    assert_eq!(options.api_key.as_deref(), Some(KEY));

    guard.set(HUB_URL_ENV, "ftp://hub.example.com");
    assert!(RemoteHubOptions::from_env().is_err());
}
