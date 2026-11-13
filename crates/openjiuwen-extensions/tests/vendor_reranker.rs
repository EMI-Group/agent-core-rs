//! Crate-boundary suite for the deprecated `AliyunReranker` vendor alias.
//!
//! Covers the live `DashScope` text-rerank transport built by
//! `AliyunReranker::from_config` and the delegation path of
//! `AliyunReranker::new`, using only the public `openjiuwen_extensions` API
//! against a loopback `wiremock` server. The in-module unit tests cover the
//! same transport at a lower level; this suite proves the crate boundary.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::time::Duration;

use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_common::security::SecretString;
use openjiuwen_core::retrieval::{Reranker, ScoreReranker, SearchResult};
use openjiuwen_extensions::vendor_specific::{AliyunReranker, AliyunRerankerConfig};
use serde_json::{json, Value};
use tokio::time::timeout;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Service path every request must hit (Python's `DashscopeReranker.end_point`).
const ENDPOINT: &str = "/services/rerank/text-rerank/text-rerank";
/// Upper bound on every await, so a hung transport fails instead of hanging.
const BOUND: Duration = Duration::from_secs(10);
/// Model name the suite configures.
const MODEL: &str = "qwen3-reranker-0.6b";
/// Credential the suite injects; never read from the environment.
const API_KEY: &str = "sk-test";

/// Starts a fresh loopback server per test (no fixed ports, no sockets left open).
async fn mock_server() -> MockServer {
    MockServer::start().await
}

/// A search result whose text and metadata are derived from `id`, so the suite
/// can tell which input document a ranked result came from.
fn hit(id: &str, score: f32) -> SearchResult {
    let mut metadata = serde_json::Map::new();
    metadata.insert("source".to_owned(), Value::String(id.to_owned()));
    SearchResult {
        id: id.to_owned(),
        text: format!("text {id}"),
        score,
        metadata,
    }
}

/// A hermetic configuration: explicit credential, explicit endpoint, no
/// environment fallback.
fn config_for(api_base: &str) -> AliyunRerankerConfig {
    AliyunRerankerConfig {
        api_key: Some(SecretString::new(API_KEY)),
        api_base: api_base.to_owned(),
        model_name: MODEL.to_owned(),
        top_n: None,
    }
}

/// Mounts a `200` response carrying `output.results` as `(index, score)` pairs.
async fn mount_scores(server: &MockServer, scores: &[(usize, f64)]) {
    let results: Vec<Value> = scores
        .iter()
        .map(|(index, score)| json!({"index": index, "relevance_score": score}))
        .collect();
    Mock::given(method("POST"))
        .and(path(ENDPOINT))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({"output": {"results": results}})),
        )
        .mount(server)
        .await;
}

/// Calls `rerank` under [`BOUND`], so a stuck request fails the test.
async fn rerank_bounded<R: Reranker + ?Sized>(
    reranker: &R,
    query: &str,
    documents: Vec<SearchResult>,
    top_n: Option<usize>,
) -> Result<Vec<SearchResult>> {
    timeout(BOUND, reranker.rerank(query, documents, top_n))
        .await
        .expect("the rerank call must finish within the bound")
}

/// Every request the mock server received, in arrival order.
async fn received_requests(server: &MockServer) -> Vec<wiremock::Request> {
    server.received_requests().await.unwrap()
}

/// The request body parsed as JSON.
fn body_json(request: &wiremock::Request) -> Value {
    serde_json::from_slice(&request.body).unwrap()
}

/// A single header value rendered as text.
fn header<'a>(request: &'a wiremock::Request, name: &str) -> &'a str {
    request
        .headers
        .get(name)
        .unwrap_or_else(|| panic!("the request must carry a {name} header"))
        .to_str()
        .unwrap()
}

/// Compares an `f32` score against the `f64` relevance score from the wire.
#[allow(clippy::cast_possible_truncation)] // relevance scores are `[0, 1]` values
fn assert_score(actual: f32, expected: f64) {
    let expected = expected as f32;
    assert!(
        (actual - expected).abs() < 1e-6,
        "expected score {expected}, got {actual}"
    );
}

/// Runs one call against a throwaway server answering with `response` and
/// returns the error the transport maps it onto.
async fn error_for(response: ResponseTemplate) -> Error {
    let server = mock_server().await;
    Mock::given(method("POST"))
        .and(path(ENDPOINT))
        .respond_with(response)
        .mount(&server)
        .await;
    let reranker = AliyunReranker::from_config(config_for(&server.uri()))
        .await
        .unwrap();
    rerank_bounded(&reranker, "q", vec![hit("a", 0.0)], None)
        .await
        .expect_err("the response must map onto an error")
}

// Python: `DashscopeReranker.rerank` -> `_request_params` plus the result mapping
// (`tests/unit_tests/core/retrieval/reranker/test_dashscope_reranker.py`).
#[tokio::test]
async fn success_path_posts_the_document_texts_and_maps_scores() {
    let server = mock_server().await;
    mount_scores(&server, &[(2, 0.9), (0, 0.5), (1, 0.5)]).await;
    let reranker = AliyunReranker::from_config(config_for(&server.uri()))
        .await
        .unwrap();
    let documents = vec![hit("a", 0.0), hit("b", 0.0), hit("c", 0.0)];

    let results = rerank_bounded(&reranker, "what is retrieval?", documents, None)
        .await
        .unwrap();

    // Descending score; the two 0.5 results keep the input order (a before b).
    let ids: Vec<&str> = results.iter().map(|result| result.id.as_str()).collect();
    assert_eq!(
        ids,
        ["c", "a", "b"],
        "results sort by score descending with stable ties"
    );
    assert_score(results[0].score, 0.9);
    assert_score(results[1].score, 0.5);
    assert_score(results[2].score, 0.5);
    // id/text/metadata are carried over from the input document at that index.
    assert_eq!(results[0].id, "c");
    assert_eq!(results[0].text, "text c");
    assert_eq!(results[0].metadata.get("source"), Some(&json!("c")));
    assert_eq!(results[1].text, "text a");
    assert_eq!(results[1].metadata.get("source"), Some(&json!("a")));
    assert_eq!(results[2].text, "text b");

    let requests = received_requests(&server).await;
    assert_eq!(requests.len(), 1, "exactly one request per call");
    let request = &requests[0];
    assert_eq!(request.method.as_str(), "POST");
    assert_eq!(request.url.path(), ENDPOINT);
    assert_eq!(header(request, "authorization"), "Bearer sk-test");
    assert_eq!(header(request, "content-type"), "application/json");
    assert_eq!(
        body_json(request),
        json!({
            "model": MODEL,
            "input": {
                "query": "what is retrieval?",
                "documents": ["text a", "text b", "text c"],
            },
            "parameters": {"return_documents": false, "top_n": 3},
        }),
        "the body is DashScope's fixed text-rerank schema"
    );
}

// Python: `DashscopeReranker._request_params` — a call-level `top_n` wins.
#[tokio::test]
async fn call_top_n_overrides_the_configured_default_and_truncates() {
    let server = mock_server().await;
    mount_scores(&server, &[(0, 0.2), (1, 0.9), (2, 0.5)]).await;
    let config = AliyunRerankerConfig {
        top_n: Some(3),
        ..config_for(&server.uri())
    };
    let reranker = AliyunReranker::from_config(config).await.unwrap();

    let results = rerank_bounded(
        &reranker,
        "q",
        vec![hit("a", 0.0), hit("b", 0.0), hit("c", 0.0)],
        Some(1),
    )
    .await
    .unwrap();

    assert_eq!(results.len(), 1, "the call-level top_n truncates");
    assert_eq!(results[0].id, "b", "the highest score survives truncation");
    let requests = received_requests(&server).await;
    assert_eq!(
        body_json(&requests[0])["parameters"]["top_n"],
        json!(1),
        "the wire top_n is the call-level limit, not the configured one"
    );
}

// Python: `DashscopeReranker._request_params` — `None` falls back to the config.
#[tokio::test]
async fn call_top_n_none_falls_back_to_the_configured_default() {
    let server = mock_server().await;
    mount_scores(&server, &[(0, 0.2), (1, 0.9), (2, 0.5)]).await;
    let config = AliyunRerankerConfig {
        top_n: Some(2),
        ..config_for(&server.uri())
    };
    let reranker = AliyunReranker::from_config(config).await.unwrap();

    let results = rerank_bounded(
        &reranker,
        "q",
        vec![hit("a", 0.0), hit("b", 0.0), hit("c", 0.0)],
        None,
    )
    .await
    .unwrap();

    let ids: Vec<&str> = results.iter().map(|result| result.id.as_str()).collect();
    assert_eq!(ids, ["b", "c"], "the configured top_n applies");
    let requests = received_requests(&server).await;
    assert_eq!(
        body_json(&requests[0])["parameters"]["top_n"],
        json!(2),
        "the wire top_n is the configured limit"
    );
}

// Python: `DashscopeReranker._request_params` — a limit beyond the corpus is
// clamped to the document count (DashScope rejects `top_n = 0`).
#[tokio::test]
async fn an_oversized_top_n_returns_every_document() {
    let server = mock_server().await;
    mount_scores(&server, &[(0, 0.2), (1, 0.9), (2, 0.5)]).await;
    let reranker = AliyunReranker::from_config(config_for(&server.uri()))
        .await
        .unwrap();

    let results = rerank_bounded(
        &reranker,
        "q",
        vec![hit("a", 0.0), hit("b", 0.0), hit("c", 0.0)],
        Some(10),
    )
    .await
    .unwrap();

    let ids: Vec<&str> = results.iter().map(|result| result.id.as_str()).collect();
    assert_eq!(
        ids,
        ["b", "c", "a"],
        "no truncation happens above the count"
    );
    let requests = received_requests(&server).await;
    assert_eq!(
        body_json(&requests[0])["parameters"]["top_n"],
        json!(3),
        "the wire top_n is the effective limit clamped to the document count"
    );
}

// Python: `DashscopeReranker.rerank` returns `{}` for empty input, before any
// request is issued.
#[tokio::test]
async fn empty_documents_return_early_without_a_request() {
    let server = mock_server().await;
    mount_scores(&server, &[(0, 0.5)]).await;
    let reranker = AliyunReranker::from_config(config_for(&server.uri()))
        .await
        .unwrap();

    let results = rerank_bounded(&reranker, "q", Vec::new(), None)
        .await
        .unwrap();

    assert!(results.is_empty(), "empty input yields an empty result");
    assert!(
        received_requests(&server).await.is_empty(),
        "empty input must not reach the service"
    );
}

// Python: `DashscopeReranker._handle_error` — 401/403 mean an auth failure.
#[tokio::test]
async fn unauthorized_statuses_are_validation_errors_without_the_key() {
    for status in [401_u16, 403_u16] {
        // The body echoes the key on purpose: quoting it would leak the
        // credential, so the mapping must not copy the body into the message.
        let error = error_for(ResponseTemplate::new(status).set_body_json(json!({
            "code": "InvalidApiKey",
            "message": format!("{API_KEY} is invalid"),
        })))
        .await;

        assert_eq!(error.kind(), ErrorKind::Validation, "status {status}");
        let message = error.message();
        assert!(
            message.contains("AliyunRerankerConfig::api_key"),
            "expected the credential source, got: {message}"
        );
        assert!(
            message.contains("DASHSCOPE_API_KEY"),
            "expected the env fallback source, got: {message}"
        );
        assert!(
            message.contains(status.to_string().as_str()),
            "expected the status in: {message}"
        );
        assert!(
            !message.contains(API_KEY),
            "the credential value leaked: {message}"
        );
    }
}

// Python: `DashscopeReranker._handle_error` — other statuses surface the
// service's `code`/`message`/`request_id`.
#[tokio::test]
async fn throttled_and_server_errors_are_transport_errors() {
    for (status, detail) in [
        (429_u16, "Requests rate limit exceeded"),
        (500_u16, "internal boom"),
    ] {
        let error = error_for(ResponseTemplate::new(status).set_body_json(json!({
            "code": "Throttling",
            "message": detail,
            "request_id": "req-1",
        })))
        .await;

        assert_eq!(error.kind(), ErrorKind::Transport, "status {status}");
        let message = error.message();
        assert!(
            message.contains(status.to_string().as_str()),
            "expected the status in: {message}"
        );
        assert!(
            message.contains(detail),
            "expected the service message in: {message}"
        );
        assert!(
            message.contains("code=Throttling"),
            "expected the service code in: {message}"
        );
        assert!(
            message.contains("request_id=req-1"),
            "expected the request id in: {message}"
        );
    }
}

// Python: `DashscopeReranker.rerank` — an unparseable success body is an error.
#[tokio::test]
async fn a_malformed_success_body_is_a_transport_error() {
    let error = error_for(ResponseTemplate::new(200).set_body_string("not json")).await;

    assert_eq!(error.kind(), ErrorKind::Transport);
    let message = error.message();
    assert!(
        message.contains("not valid JSON"),
        "expected a JSON parsing failure, got: {message}"
    );
    assert!(
        message.contains("not json"),
        "expected the body snippet in: {message}"
    );
}

// Python: `DashscopeReranker.rerank` — a `200` without `results` is an error.
#[tokio::test]
async fn a_success_body_without_results_is_a_transport_error() {
    let error = error_for(ResponseTemplate::new(200).set_body_json(json!({"output": {}}))).await;

    assert_eq!(error.kind(), ErrorKind::Transport);
    let message = error.message();
    assert!(
        message.contains("carried no results"),
        "expected a missing-results failure, got: {message}"
    );
}

// Python: `AliyunReranker = DashscopeReranker` — the alias adds no behavior, so
// an injected reranker answers verbatim and reports no configuration.
#[tokio::test]
async fn new_forwards_the_inner_reranker_verbatim_and_reports_no_config() {
    let reranker = AliyunReranker::new(Arc::new(ScoreReranker::default()));
    assert!(
        reranker.config().is_none(),
        "an injected reranker has no configuration to report"
    );

    let results = rerank_bounded(
        &reranker,
        "q",
        vec![hit("low", 0.1), hit("high", 0.9), hit("mid", 0.5)],
        Some(2),
    )
    .await
    .unwrap();

    // Exactly `ScoreReranker`'s ordering and truncation, untouched by the alias.
    let ids: Vec<&str> = results.iter().map(|result| result.id.as_str()).collect();
    assert_eq!(ids, ["high", "mid"]);
    assert_score(results[0].score, 0.9);
    assert_score(results[1].score, 0.5);

    let configured = AliyunReranker::from_config(AliyunRerankerConfig {
        api_key: Some(SecretString::new(API_KEY)),
        ..AliyunRerankerConfig::default()
    })
    .await
    .unwrap();
    assert!(
        configured.config().is_some(),
        "from_config reports its configuration"
    );
}

// Python: the alias inherits `DashscopeReranker`, which callers hold as a
// `StandardReranker`; the Rust equivalent must stay object-safe.
#[tokio::test]
async fn from_config_is_usable_as_a_trait_object() {
    let server = mock_server().await;
    mount_scores(&server, &[(0, 0.25), (1, 0.75)]).await;
    let reranker: Arc<dyn Reranker> = Arc::new(
        AliyunReranker::from_config(config_for(&server.uri()))
            .await
            .unwrap(),
    );

    let results = rerank_bounded(
        reranker.as_ref(),
        "q",
        vec![hit("a", 0.0), hit("b", 0.0)],
        None,
    )
    .await
    .unwrap();

    let ids: Vec<&str> = results.iter().map(|result| result.id.as_str()).collect();
    assert_eq!(ids, ["b", "a"]);
    assert_eq!(received_requests(&server).await.len(), 1);
}

// Python: `api_base.removesuffix("/").removesuffix(end_point)` — every base
// variant resolves to the same service path.
#[tokio::test]
async fn api_base_variants_all_hit_the_service_path() {
    let server = mock_server().await;
    mount_scores(&server, &[(0, 0.5)]).await;
    let bases = [
        server.uri(),
        format!("{}/", server.uri()),
        format!("{}{ENDPOINT}", server.uri()),
        format!("{}{ENDPOINT}/", server.uri()),
    ];

    for base in &bases {
        let reranker = AliyunReranker::from_config(config_for(base)).await.unwrap();
        let results = rerank_bounded(&reranker, "q", vec![hit("a", 0.0)], None)
            .await
            .unwrap();
        assert_eq!(results.len(), 1, "api_base `{base}` must reach the service");
    }

    let requests = received_requests(&server).await;
    assert_eq!(
        requests.len(),
        bases.len(),
        "one request per api_base variant"
    );
    for request in &requests {
        assert_eq!(
            request.url.path(),
            ENDPOINT,
            "every api_base variant resolves to the same path"
        );
    }
}
