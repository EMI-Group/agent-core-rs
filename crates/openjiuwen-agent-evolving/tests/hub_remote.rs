//! Integration tests for the remote experience-hub client, driven through the
//! crate's public API over a loopback `wiremock` server.
//!
//! Hub protocol v1 is exercised end to end: bearer auth, publish/get/remove/
//! list/search, the error mapping and the retry budget. The stager hand-off is
//! covered too, so the approval-gated path publishes through the same client.
//! Every server is an ephemeral `127.0.0.1` port; no test reaches the network.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::assert_is_empty,
    clippy::match_wildcard_for_single_variants
)]

use std::time::Duration;

use openjiuwen_agent_evolving::{
    ExperienceRecord, ExperienceSection, QueryKeywords, RemoteHubOptions, RemoteHubSharingBackend,
    ShareStager, SharedSkillBundle, SharingBackend, SharingMeta, SkillSearchResult,
};
use openjiuwen_common::error::{Error, ErrorKind};
use openjiuwen_test_support::wiremock::matchers::{body_json, header, method, path};
use openjiuwen_test_support::wiremock::{Mock, MockServer, Request, ResponseTemplate};
use serde_json::json;

/// The upper bound every hub call must finish within.
const BOUND: Duration = Duration::from_secs(10);
/// The skill every fixture record belongs to.
const SKILL: &str = "pdf-report";
/// The bearer token the auth tests configure.
const KEY: &str = "hub-secret";
/// A record about the canonical demo skill.
const CONTENT_A: &str = "raise the read timeout before parsing large pdf files";

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

/// The bundle [`stager`] assembles from [`records`].
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
fn error_of<T: std::fmt::Debug>(result: openjiuwen_common::error::Result<T>) -> Error {
    result.expect_err("the hub call must fail")
}

#[tokio::test]
async fn publish_sends_bearer_auth_and_round_trips_through_get() {
    let server = MockServer::start().await;
    let bundle = published_bundle();
    Mock::given(method("POST"))
        .and(path("/v1/hub/bundles"))
        .and(header("authorization", format!("Bearer {KEY}")))
        .and(body_json(&bundle))
        .respond_with(ResponseTemplate::new(201).set_body_json(json!({"bundle_id": "hub-9"})))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/hub/bundles/hub-9"))
        .and(header("authorization", format!("Bearer {KEY}")))
        .respond_with(ResponseTemplate::new(200).set_body_json(&bundle))
        .expect(1)
        .mount(&server)
        .await;

    let client = backend_with(&server, |options| options.with_api_key(KEY));
    assert_eq!(
        bounded(client.publish(&bundle)).await.unwrap(),
        String::from("hub-9")
    );
    assert_eq!(
        bounded(client.get("hub-9")).await.unwrap(),
        Some(bundle.clone())
    );

    let requests = received(&server).await;
    assert_eq!(requests.len(), 2);
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&requests[0].body).unwrap(),
        serde_json::to_value(&bundle).unwrap()
    );
}

#[tokio::test]
async fn publish_falls_back_to_the_content_id_when_the_hub_echoes_a_blank_id() {
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
async fn get_maps_an_unknown_bundle_to_none() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/hub/bundles/bundle-missing"))
        .respond_with(ResponseTemplate::new(404).set_body_json(
            json!({"error": {"code": "bundle_not_found", "message": "no such bundle"}}),
        ))
        .expect(1)
        .mount(&server)
        .await;

    let found = bounded(backend(&server).get("bundle-missing"))
        .await
        .unwrap();

    assert_eq!(found, None);
}

#[tokio::test]
async fn remove_maps_an_unknown_bundle_to_false() {
    let server = MockServer::start().await;
    Mock::given(method("DELETE"))
        .and(path("/v1/hub/bundles/bundle-a"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"removed": true})))
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
    assert!(!bounded(client.remove("bundle-missing")).await.unwrap());
}

#[tokio::test]
async fn list_returns_bundles_sorted_by_bundle_id() {
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
async fn search_honours_the_limit_and_drops_zero_scores() {
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
async fn search_with_an_empty_query_makes_no_request() {
    let server = MockServer::start().await;

    let results = bounded(backend(&server).search(&QueryKeywords::default(), 5))
        .await
        .unwrap();

    assert!(results.is_empty());
    assert!(received(&server).await.is_empty());
}

#[tokio::test]
async fn a_hub_error_envelope_surfaces_as_a_validation_error_naming_the_code() {
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
async fn a_server_error_then_success_is_retried() {
    let server = MockServer::start().await;
    let first = named_bundle("bundle-a", "pdf-report");
    Mock::given(method("GET"))
        .and(path("/v1/hub/bundles"))
        .respond_with(ResponseTemplate::new(503))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/hub/bundles"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"bundles": [first.clone()]})))
        .expect(1)
        .mount(&server)
        .await;

    let client = backend_with(&server, |options| options.with_max_retries(1));
    let listed = bounded(client.list()).await.unwrap();

    assert_eq!(listed, vec![first]);
    assert_eq!(received(&server).await.len(), 2, "one retry after the 503");
}

#[tokio::test]
async fn share_with_approval_publishes_end_to_end() {
    let server = MockServer::start().await;
    let records = records();
    let expected = ShareStager::default()
        .build_bundle(&records, &meta())
        .unwrap();
    assert!(expected.requires_approval);
    Mock::given(method("POST"))
        .and(path("/v1/hub/bundles"))
        .and(header("authorization", format!("Bearer {KEY}")))
        .and(body_json(&expected.bundle))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "bundle_id": expected.bundle.bundle_id.clone()
        })))
        .expect(1)
        .mount(&server)
        .await;

    let client = backend_with(&server, |options| options.with_api_key(KEY));
    let published = bounded(ShareStager::default().share(&client, &records, &meta(), true))
        .await
        .unwrap();

    assert_eq!(published, Some(expected.bundle.bundle_id));
    assert_eq!(received(&server).await.len(), 1);
}

#[tokio::test]
async fn share_without_approval_publishes_nothing() {
    let server = MockServer::start().await;

    let published =
        bounded(ShareStager::default().share(&backend(&server), &records(), &meta(), false))
            .await
            .unwrap();

    assert_eq!(published, None);
    assert!(received(&server).await.is_empty());
}

#[test]
fn a_non_http_base_url_is_rejected_before_any_request() {
    let error = RemoteHubSharingBackend::new(RemoteHubOptions::new("ftp://hub.example.com"))
        .expect_err("ftp must be rejected");

    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("ftp"), "{}", error.message());
}

#[tokio::test]
async fn health_normalises_a_trailing_slash_on_the_base_url() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(200))
        .expect(1)
        .mount(&server)
        .await;

    let client =
        RemoteHubSharingBackend::new(RemoteHubOptions::new(format!("{}/", server.uri()))).unwrap();
    bounded(client.health()).await.unwrap();

    assert_eq!(client.options().base_url, server.uri());
}

#[tokio::test]
async fn a_malformed_success_body_names_the_endpoint() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/hub/bundles"))
        .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
        .expect(1)
        .mount(&server)
        .await;

    let error = error_of(bounded(backend(&server).list()).await);

    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.message().contains("GET /v1/hub/bundles"),
        "{}",
        error.message()
    );
    assert!(
        error.message().contains("malformed body"),
        "{}",
        error.message()
    );
}
