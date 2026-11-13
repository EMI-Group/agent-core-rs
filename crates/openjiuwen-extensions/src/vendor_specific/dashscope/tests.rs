use std::sync::{Mutex, MutexGuard, PoisonError};

use openjiuwen_common::error::ErrorKind;
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::*;

// Python counterpart: `tests/unit_tests/core/retrieval/` (`DashscopeReranker`).
/// Credential used by every authenticated test.
const API_KEY: &str = "sk-dashscope-test-key";

/// Serialises the tests that mutate [`API_KEY_ENV`].
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Holds [`ENV_LOCK`] and leaves [`API_KEY_ENV`] unset, before and after.
struct EnvGuard {
    /// The lock guard; dropped last.
    _lock: MutexGuard<'static, ()>,
}

impl EnvGuard {
    /// Acquires the lock, recovering it when another test panicked.
    fn new() -> Self {
        let lock = ENV_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        std::env::remove_var(API_KEY_ENV);
        Self { _lock: lock }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        std::env::remove_var(API_KEY_ENV);
    }
}

/// One input document whose text embeds its id.
fn hit(id: &str) -> SearchResult {
    SearchResult {
        id: id.to_owned(),
        text: format!("text {id}"),
        score: 0.0,
        metadata: serde_json::Map::new(),
    }
}

/// A configuration pointed at `server`, with a non-default model.
fn config_for(server: &MockServer, top_n: Option<usize>) -> AliyunRerankerConfig {
    AliyunRerankerConfig {
        api_key: Some(SecretString::new(API_KEY)),
        api_base: server.uri(),
        model_name: "qwen3-reranker-8b".to_owned(),
        top_n,
    }
}

/// A transport aimed at `server` with no configured limit.
fn transport(server: &MockServer) -> DashScopeReranker {
    DashScopeReranker::new(config_for(server, None)).expect("the transport builds")
}

/// Mounts the `Mock` answering every rerank call with `output.results`.
async fn mock_results(server: &MockServer, results: Value) {
    Mock::given(method("POST"))
        .and(path(END_POINT))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({ "output": { "results": results } })),
        )
        .mount(server)
        .await;
}

/// Mounts a `Mock` answering with `status` and a raw string body.
async fn mock_status(server: &MockServer, status: u16, body: &str) {
    Mock::given(method("POST"))
        .and(path(END_POINT))
        .respond_with(ResponseTemplate::new(status).set_body_string(body))
        .mount(server)
        .await;
}

/// The single request `server` received, decoded as JSON.
async fn received_body(server: &MockServer) -> Value {
    let requests = server
        .received_requests()
        .await
        .expect("the mock records requests");
    assert_eq!(requests.len(), 1, "exactly one request was sent");
    serde_json::from_slice(&requests[0].body).expect("the request body is JSON")
}

/// The `parameters.top_n` the single request carried.
async fn sent_top_n(server: &MockServer) -> usize {
    let body = received_body(server).await;
    usize::try_from(body["parameters"]["top_n"].as_u64().unwrap()).unwrap()
}

/// The ids of `results`, in order.
fn ids(results: &[SearchResult]) -> Vec<&str> {
    results.iter().map(|result| result.id.as_str()).collect()
}

#[tokio::test]
async fn request_carries_the_dashscope_body_and_bearer_token() {
    // ports `DashscopeReranker._request_params` plus `StandardReranker._headers`.
    let server = MockServer::start().await;
    mock_results(
        &server,
        json!([{ "index": 0, "relevance_score": 0.9 }, { "index": 1, "relevance_score": 0.5 }]),
    )
    .await;

    transport(&server)
        .rerank("what is retrieval?", vec![hit("a"), hit("b")], None)
        .await
        .expect("the rerank succeeds");

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    let request = &requests[0];
    assert_eq!(request.method.as_str(), "POST");
    assert_eq!(request.url.path(), END_POINT);
    assert_eq!(
        request
            .headers
            .get("authorization")
            .unwrap()
            .to_str()
            .unwrap(),
        format!("Bearer {API_KEY}")
    );
    assert_eq!(
        request
            .headers
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap(),
        "application/json"
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&request.body).unwrap(),
        json!({
            "model": "qwen3-reranker-8b",
            "input": {
                "query": "what is retrieval?",
                "documents": ["text a", "text b"],
            },
            "parameters": { "return_documents": false, "top_n": 2 },
        })
    );
}

#[tokio::test]
async fn results_map_back_to_the_input_documents_and_sort_descending() {
    // ports `StandardReranker._parse_response`, then the kernel's stable order.
    let server = MockServer::start().await;
    mock_results(
        &server,
        json!([
            { "index": 2, "relevance_score": 0.5 },
            { "index": 0, "relevance_score": 0.2 },
            { "index": 1, "relevance_score": 0.5 },
            { "index": 3, "relevance_score": 0.9 },
        ]),
    )
    .await;
    let mut documents = vec![hit("a"), hit("b"), hit("c"), hit("d")];
    documents[3].metadata.insert("k".to_owned(), json!("v"));

    let results = transport(&server)
        .rerank("query", documents, None)
        .await
        .expect("the rerank succeeds");

    // `d` scores highest; `b` and `c` tie at 0.5, so their input order holds.
    assert_eq!(ids(&results), ["d", "b", "c", "a"]);
    assert_eq!(results[0].metadata.get("k"), Some(&json!("v")));
    for result in &results {
        assert_eq!(result.text, format!("text {}", result.id));
    }
    assert!((results[1].score - 0.5).abs() < f32::EPSILON);
    assert!((results[3].score - 0.2).abs() < f32::EPSILON);
}

#[tokio::test]
async fn a_duplicated_index_keeps_the_last_score() {
    // Python assigns into a dict, so the later score overwrites the earlier one.
    let server = MockServer::start().await;
    mock_results(
        &server,
        json!([
            { "index": 0, "relevance_score": 0.1 },
            { "index": 0, "relevance_score": 0.9 },
            { "index": 1, "relevance_score": 0.4 },
        ]),
    )
    .await;

    let results = transport(&server)
        .rerank("query", vec![hit("a"), hit("b")], None)
        .await
        .expect("the rerank succeeds");

    assert_eq!(ids(&results), ["a", "b"]);
    assert!((results[0].score - 0.9).abs() < f32::EPSILON);
}

#[tokio::test]
async fn top_n_precedence_sends_and_truncates_the_effective_limit() {
    // The call argument wins over the configuration, which wins over the count.
    let documents = || vec![hit("a"), hit("b"), hit("c"), hit("d")];
    let all_scores = json!([
        { "index": 0, "relevance_score": 0.1 },
        { "index": 1, "relevance_score": 0.9 },
        { "index": 2, "relevance_score": 0.7 },
        { "index": 3, "relevance_score": 0.3 },
    ]);

    let server = MockServer::start().await;
    mock_results(&server, all_scores.clone()).await;
    let results = DashScopeReranker::new(config_for(&server, Some(3)))
        .unwrap()
        .rerank("query", documents(), Some(2))
        .await
        .unwrap();
    assert_eq!(sent_top_n(&server).await, 2);
    assert_eq!(ids(&results), ["b", "c"]);

    let server = MockServer::start().await;
    mock_results(&server, all_scores.clone()).await;
    let results = DashScopeReranker::new(config_for(&server, Some(3)))
        .unwrap()
        .rerank("query", documents(), None)
        .await
        .unwrap();
    assert_eq!(sent_top_n(&server).await, 3);
    assert_eq!(ids(&results), ["b", "c", "d"]);

    let server = MockServer::start().await;
    mock_results(&server, all_scores).await;
    let results = transport(&server)
        .rerank("query", documents(), None)
        .await
        .unwrap();
    assert_eq!(sent_top_n(&server).await, 4);
    assert_eq!(ids(&results), ["b", "c", "d", "a"]);
}

#[tokio::test]
async fn a_zero_or_oversized_limit_means_no_truncation() {
    // `DashScope` rejects `top_n = 0`, and `Some(0)` keeps everything here.
    for limit in [0, 9] {
        let server = MockServer::start().await;
        mock_results(
            &server,
            json!([
                { "index": 0, "relevance_score": 0.2 },
                { "index": 1, "relevance_score": 0.8 },
            ]),
        )
        .await;

        let results = transport(&server)
            .rerank("query", vec![hit("a"), hit("b")], Some(limit))
            .await
            .expect("the rerank succeeds");

        assert_eq!(sent_top_n(&server).await, 2, "limit {limit}");
        assert_eq!(ids(&results), ["b", "a"], "limit {limit}");
    }
}

#[tokio::test]
async fn documents_absent_from_the_response_score_zero() {
    // Python's `dict.fromkeys(doc_ids, 0.0)` default.
    let server = MockServer::start().await;
    mock_results(&server, json!([{ "index": 1, "relevance_score": 0.8 }])).await;

    let results = transport(&server)
        .rerank("query", vec![hit("a"), hit("b")], None)
        .await
        .expect("the rerank succeeds");

    assert_eq!(ids(&results), ["b", "a"]);
    assert!(results[1].score.abs() < f32::EPSILON);
}

#[tokio::test]
async fn empty_documents_send_no_request() {
    let server = MockServer::start().await;

    let results = transport(&server)
        .rerank("query", Vec::new(), None)
        .await
        .expect("an empty input succeeds");

    assert!(results.is_empty());
    assert!(
        server.received_requests().await.unwrap().is_empty(),
        "no HTTP request was sent"
    );
}

#[tokio::test]
async fn unauthorized_and_forbidden_bodies_are_never_echoed() {
    for status in [401, 403] {
        let server = MockServer::start().await;
        mock_status(
            &server,
            status,
            &json!({
                "code": "InvalidApiKey",
                "message": format!("the key {API_KEY} is invalid"),
                "request_id": "req-auth",
            })
            .to_string(),
        )
        .await;

        let error = transport(&server)
            .rerank("query", vec![hit("a")], None)
            .await
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Validation);
        let message = error.to_string();
        assert!(message.contains(&status.to_string()), "{message}");
        assert!(
            message.contains("AliyunRerankerConfig::api_key"),
            "{message}"
        );
        assert!(message.contains(API_KEY_ENV), "{message}");
        assert!(!message.contains(API_KEY), "leaked the key: {message}");
    }
}

#[tokio::test]
async fn throttling_and_server_errors_keep_the_dashscope_detail() {
    for status in [429, 500] {
        let server = MockServer::start().await;
        mock_status(
            &server,
            status,
            &json!({
                "code": "Throttling",
                "message": "Requests rate limit exceeded",
                "request_id": "req-throttle",
            })
            .to_string(),
        )
        .await;

        let error = transport(&server)
            .rerank("query", vec![hit("a")], None)
            .await
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Transport);
        let message = error.to_string();
        assert!(message.contains(&status.to_string()), "{message}");
        assert!(message.contains("Throttling"), "{message}");
        assert!(
            message.contains("Requests rate limit exceeded"),
            "{message}"
        );
        assert!(message.contains("req-throttle"), "{message}");
        assert!(!message.contains(API_KEY), "leaked the key: {message}");
    }
}

#[tokio::test]
async fn other_client_errors_map_to_transport_with_the_detail() {
    let server = MockServer::start().await;
    mock_status(
        &server,
        400,
        &json!({ "code": "InvalidParameter", "message": "top_n must be positive" }).to_string(),
    )
    .await;

    let error = transport(&server)
        .rerank("query", vec![hit("a")], None)
        .await
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Transport);
    let message = error.to_string();
    assert!(message.contains("400"), "{message}");
    assert!(message.contains("InvalidParameter"), "{message}");
    assert!(message.contains("top_n must be positive"), "{message}");
}

#[tokio::test]
async fn malformed_or_resultless_bodies_map_to_transport() {
    for body in ["not json at all", "{}", r#"{"output": {}}"#] {
        let server = MockServer::start().await;
        mock_status(&server, 200, body).await;

        let error = transport(&server)
            .rerank("query", vec![hit("a")], None)
            .await
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Transport, "body `{body}`");
        let message = error.to_string();
        assert!(
            message.contains("not valid JSON") || message.contains("no results"),
            "body `{body}`: {message}"
        );
    }
}

#[tokio::test]
async fn an_out_of_range_index_maps_to_transport() {
    let server = MockServer::start().await;
    mock_results(&server, json!([{ "index": 5, "relevance_score": 0.5 }])).await;

    let error = transport(&server)
        .rerank("query", vec![hit("a"), hit("b")], None)
        .await
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Transport);
    let message = error.to_string();
    assert!(message.contains("index 5"), "{message}");
    assert!(message.contains("only 2 documents"), "{message}");
}

#[test]
fn a_non_finite_score_maps_to_transport() {
    let error = map_results(
        vec![hit("a")],
        &[RerankResult {
            index: 0,
            relevance_score: f64::NAN,
        }],
        1,
    )
    .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Transport);
    assert!(error.message().contains("non-finite"), "{}", error);
    assert!(error.message().contains("index 0"), "{}", error);
}

#[test]
fn endpoint_url_normalizes_like_python() {
    let expected = "https://dashscope.aliyuncs.com/services/rerank/text-rerank/text-rerank";
    for api_base in [
        "https://dashscope.aliyuncs.com",
        "https://dashscope.aliyuncs.com/",
        "  https://dashscope.aliyuncs.com  ",
        "https://dashscope.aliyuncs.com/services/rerank/text-rerank/text-rerank",
        "https://dashscope.aliyuncs.com/services/rerank/text-rerank/text-rerank/",
    ] {
        assert_eq!(
            endpoint_url(api_base).unwrap().as_str(),
            expected,
            "api_base `{api_base}`"
        );
    }

    // A gateway prefix survives, matching httpx's `base_url` merging.
    assert_eq!(
        endpoint_url("https://gw.internal/proxy").unwrap().as_str(),
        "https://gw.internal/proxy/services/rerank/text-rerank/text-rerank"
    );

    for invalid in [
        "",
        "   ",
        "/",
        "ftp://dashscope.aliyuncs.com",
        "dashscope.aliyuncs.com",
    ] {
        let error = endpoint_url(invalid).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation, "api_base `{invalid}`");
        assert!(error.message().contains("api_base"), "{}", error);
    }
}

#[test]
fn api_key_falls_back_to_the_dashscope_environment_variable() {
    let _guard = EnvGuard::new();
    std::env::set_var(API_KEY_ENV, "sk-from-env");

    for configured in [
        None,
        Some(SecretString::new("")),
        Some(SecretString::new("   ")),
    ] {
        let transport = DashScopeReranker::new(AliyunRerankerConfig {
            api_key: configured,
            ..AliyunRerankerConfig::default()
        })
        .expect("the environment supplies the key");
        assert_eq!(transport.api_key.expose_secret(), "sk-from-env");
    }
}

#[test]
fn missing_credentials_name_both_sources() {
    let _guard = EnvGuard::new();

    let error = DashScopeReranker::new(AliyunRerankerConfig::default()).unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.message();
    assert!(
        message.contains("AliyunRerankerConfig::api_key"),
        "{message}"
    );
    assert!(message.contains(API_KEY_ENV), "{message}");
}

#[test]
fn an_invalid_api_base_is_rejected_before_any_request() {
    for api_base in [String::new(), "   ".to_owned(), "ftp://host".to_owned()] {
        let error = DashScopeReranker::new(AliyunRerankerConfig {
            api_key: Some(SecretString::new(API_KEY)),
            api_base: api_base.clone(),
            ..AliyunRerankerConfig::default()
        })
        .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Validation, "api_base `{api_base}`");
        assert!(error.message().contains("api_base"), "{}", error);
    }
}

#[test]
fn request_serializes_to_the_dashscope_shape() {
    let request = RerankRequest {
        model: "qwen3-reranker-0.6b".to_owned(),
        input: RerankInput {
            query: "q".to_owned(),
            documents: vec!["a".to_owned(), "b".to_owned()],
        },
        parameters: RerankParameters {
            return_documents: false,
            top_n: 2,
        },
    };

    assert_eq!(
        serde_json::to_value(&request).unwrap(),
        json!({
            "model": "qwen3-reranker-0.6b",
            "input": { "query": "q", "documents": ["a", "b"] },
            "parameters": { "return_documents": false, "top_n": 2 },
        })
    );
}

#[test]
fn response_deserializes_nested_top_level_and_tolerates_unknown_fields() {
    let nested: RerankResponse = serde_json::from_str(
        r#"{"output": {"results": [{"index": 0, "relevance_score": 0.25, "extra": true}]}, "usage": 7}"#,
    )
    .unwrap();
    let results = nested.results().unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].index, 0);
    assert!((results[0].relevance_score - 0.25).abs() < 1e-9);

    let top_level: RerankResponse =
        serde_json::from_str(r#"{"results": [{"index": 1, "relevance_score": 0.5}]}"#).unwrap();
    assert_eq!(top_level.results().unwrap()[0].index, 1);

    // An `output` without `results` does not fall back, like Python's lookup.
    let empty_output: RerankResponse = serde_json::from_str(
        r#"{"output": {}, "results": [{"index": 0, "relevance_score": 1.0}]}"#,
    )
    .unwrap();
    assert!(empty_output.results().is_none());

    // Error bodies keep their code/message/request_id.
    let error: RerankResponse = serde_json::from_str(
        r#"{"code": "Throttling", "message": "slow down", "request_id": "r1"}"#,
    )
    .unwrap();
    assert_eq!(
        error.detail().as_deref(),
        Some("code=Throttling, message=slow down, request_id=r1")
    );
}

#[test]
fn error_detail_is_bounded_and_falls_back_to_the_raw_body() {
    let long = "x".repeat(BODY_SNIPPET_LIMIT + 50);
    let detail = error_detail(&long);
    assert!(detail.ends_with('…'), "{detail}");
    assert_eq!(detail.chars().count(), BODY_SNIPPET_LIMIT + 1);

    assert_eq!(error_detail("  \n "), "the response body was empty");
    assert_eq!(error_detail(r#" {"foo": "bar"} "#), r#"{"foo": "bar"}"#);
}
