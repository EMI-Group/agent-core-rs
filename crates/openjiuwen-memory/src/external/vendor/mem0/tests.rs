//! Offline wiremock tests for the mem0 adapter: wire bodies, decoding, retries,
//! failover, cancellation, and the tool rail.
use chrono::Utc;
use openjiuwen_common::error::Error;
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;
use wiremock::matchers::{header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::*;
use crate::external::provider::MemoryProvider;
use crate::record::{
    AddOptions, CallOptions, MemoryFilter, MemoryScope, MemoryType, MemoryUpdate, NewMemory,
    SearchOptions,
};

/// A configuration pointing at `uri` with the test credential.
fn config(uri: &str) -> ProviderConfig {
    ProviderConfig::new(ProviderKind::Mem0)
        .with_endpoint(uri)
        .with_api_key("test-key")
}

/// A configured handle for `uri`.
fn handle(uri: &str) -> Mem0MemoryProvider {
    Mem0MemoryProvider::new(&config(uri))
}

/// The scope every test uses.
fn scope() -> MemoryScope {
    MemoryScope::new("alice", "workspace")
}

/// Reads one request body as JSON.
fn request_json(request: &wiremock::Request) -> Value {
    serde_json::from_slice(&request.body).expect("the request body is JSON")
}

#[tokio::test]
async fn add_posts_one_verbatim_memory_and_keeps_the_vendor_id() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/memories/"))
        .and(header("authorization", "Token test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "results": [{ "id": "m-1", "memory": "hello" }],
        })))
        .expect(1)
        .mount(&server)
        .await;

    let outcome = handle(&server.uri())
        .add(
            &scope(),
            vec![NewMemory::semantic("hello")],
            &AddOptions::default(),
        )
        .await
        .expect("add");

    assert_eq!(outcome.added.len(), 1);
    assert_eq!(outcome.added[0].id, "m-1");
    assert_eq!(outcome.added[0].content, "hello");
    assert_eq!(outcome.added[0].kind, MemoryType::SemanticMemory);
    assert_eq!(outcome.skipped.len(), 0);

    let requests = server.received_requests().await.expect("requests");
    assert_eq!(requests.len(), 1);
    let body = request_json(&requests[0]);
    assert_eq!(body["messages"][0]["role"], json!("user"));
    assert_eq!(body["messages"][0]["content"], json!("hello"));
    assert_eq!(body["user_id"], json!("alice"));
    assert_eq!(body["infer"], json!(false));
    assert!(body.get("agent_id").is_none(), "{body}");
}

#[tokio::test]
async fn add_reports_an_empty_results_array_as_a_vendor_dedupe() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/memories/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "results": [] })))
        .mount(&server)
        .await;

    let outcome = handle(&server.uri())
        .add(
            &scope(),
            vec![NewMemory::semantic("hello")],
            &AddOptions::default(),
        )
        .await
        .expect("add");

    assert!(outcome.added.is_empty(), "{outcome:?}");
    assert_eq!(outcome.skipped.len(), 1);
    assert_eq!(outcome.skipped[0].len(), 24);
}

#[tokio::test]
async fn search_filters_by_threshold_and_sends_the_documented_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/memories/search/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "results": [
                { "id": "a", "memory": "alpha", "score": 0.9, "created_at": "2024-01-01T00:00:00Z" },
                { "id": "b", "memory": "beta", "score": 0.1 },
            ],
        })))
        .expect(1)
        .mount(&server)
        .await;

    let options = SearchOptions::default().with_top_k(5).with_threshold(0.3);
    let hits = handle(&server.uri())
        .search(&scope(), "alpha", &options)
        .await
        .expect("search");

    assert_eq!(hits.len(), 1, "the low-score hit is filtered out");
    assert_eq!(hits[0].record.id, "a");
    assert_eq!(hits[0].record.content, "alpha");
    assert!((hits[0].score - 0.9).abs() < 1e-6);
    assert_eq!(
        hits[0].record.timestamp,
        chrono::DateTime::parse_from_rfc3339("2024-01-01T00:00:00Z")
            .expect("rfc 3339 timestamp")
            .with_timezone(&Utc)
    );

    let requests = server.received_requests().await.expect("requests");
    let body = request_json(&requests[0]);
    assert_eq!(body["query"], json!("alpha"));
    assert_eq!(body["user_id"], json!("alice"));
    assert_eq!(body["limit"], json!(5));
    assert!((body["threshold"].as_f64().expect("threshold") - 0.3).abs() < 1e-6);
}

#[tokio::test]
async fn list_maps_results_and_tolerates_an_empty_body() {
    let populated = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/memories/"))
        .and(query_param("user_id", "alice"))
        .and(query_param("page", "1"))
        .and(query_param("page_size", "100"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "results": [
                { "id": "r1", "memory": "one" },
                { "id": "r2", "memory": "two" },
            ],
        })))
        .mount(&populated)
        .await;

    let records = handle(&populated.uri())
        .list(&MemoryFilter::for_scope(&scope()))
        .await
        .expect("list");
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].id, "r1");
    assert_eq!(records[0].content, "one");
    assert_eq!(records[1].id, "r2");

    let empty = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/memories/"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&empty)
        .await;
    let records = handle(&empty.uri())
        .list(&MemoryFilter::for_scope(&scope()))
        .await
        .expect("empty list");
    assert_eq!(records.len(), 0);
}

#[tokio::test]
async fn get_and_delete_map_ids_and_missing_records() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/memories/m-9/"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({ "id": "m-9", "memory": "stored" })),
        )
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/memories/missing/"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .and(path("/v1/memories/m-9/"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .and(path("/v1/memories/gone/"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;

    let handle = handle(&server.uri());
    let record = handle
        .get(&scope(), "m-9")
        .await
        .expect("get")
        .expect("record");
    assert_eq!(record.id, "m-9");
    assert_eq!(record.content, "stored");
    assert!(handle
        .get(&scope(), "missing")
        .await
        .expect("missing get")
        .is_none());
    assert!(handle.delete(&scope(), "m-9").await.expect("delete"));
    assert!(!handle
        .delete(&scope(), "gone")
        .await
        .expect("delete missing"));
}

#[tokio::test]
async fn update_patches_the_text_and_reads_the_returned_record() {
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .and(path("/v1/memories/m-1/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "m-1",
            "memory": "updated",
            "metadata": { "source": "test" },
        })))
        .expect(1)
        .mount(&server)
        .await;

    let record = handle(&server.uri())
        .update(
            &scope(),
            "m-1",
            MemoryUpdate::content("updated"),
            &CallOptions::default(),
        )
        .await
        .expect("update");
    assert_eq!(record.id, "m-1");
    assert_eq!(record.content, "updated");
    assert_eq!(record.metadata_str("source"), Some("test"));

    let requests = server.received_requests().await.expect("requests");
    assert_eq!(request_json(&requests[0])["text"], json!("updated"));
}

#[tokio::test]
async fn search_retries_a_transient_status_once() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/memories/search/"))
        .respond_with(ResponseTemplate::new(503))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/memories/search/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "results": [{ "id": "a", "memory": "alpha", "score": 0.8 }],
        })))
        .mount(&server)
        .await;

    let config = config(&server.uri())
        .with_option("max_retries", json!(1))
        .with_option("retry_backoff_ms", json!(1));
    let hits = Mem0MemoryProvider::new(&config)
        .search(&scope(), "alpha", &SearchOptions::default())
        .await
        .expect("search after retry");
    assert_eq!(hits.len(), 1);
    assert_eq!(server.received_requests().await.expect("requests").len(), 2);
}

#[tokio::test]
async fn search_fails_over_to_the_second_endpoint() {
    let primary = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/memories/search/"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&primary)
        .await;
    let secondary = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/memories/search/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "results": [{ "id": "a", "memory": "alpha", "score": 0.7 }],
        })))
        .mount(&secondary)
        .await;

    let config = config(&primary.uri())
        .with_option("failover_endpoints", json!([secondary.uri()]))
        .with_option("max_retries", json!(0))
        .with_option("retry_backoff_ms", json!(1));
    let hits = Mem0MemoryProvider::new(&config)
        .search(&scope(), "alpha", &SearchOptions::default())
        .await
        .expect("failover search");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].record.id, "a");
    assert_eq!(
        secondary.received_requests().await.expect("requests").len(),
        1
    );
}

#[tokio::test]
async fn capabilities_follow_a_configured_handle_and_an_unconfigured_one_stays_unsupported() {
    let handle = handle("https://vendor.invalid");
    let capabilities = handle.capabilities();
    assert!(capabilities.semantic_search);
    assert!(!capabilities.keyword_search);
    assert!(!capabilities.graph);
    assert!(!capabilities.variables);
    assert!(!capabilities.message_history);
    assert!(!capabilities.pagination);
    assert_eq!(capabilities.max_batch, None);
    assert!(handle.is_available());

    let unconfigured =
        Mem0MemoryProvider::new(&ProviderConfig::new(ProviderKind::Mem0).with_api_key("test-key"));
    assert!(!unconfigured.capabilities().semantic_search);
    assert!(!unconfigured.is_available());
    let error = unconfigured
        .search(&scope(), "anything", &SearchOptions::default())
        .await
        .expect_err("unconfigured handle");
    assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
    assert!(error.to_string().contains("mem0ai"), "{error}");
}

#[tokio::test]
async fn a_cancelled_search_never_reaches_the_vendor() {
    let token = CancellationToken::new();
    token.cancel();
    let options = SearchOptions::default().with_cancellation(token);
    let error = handle("http://127.0.0.1:9")
        .search(&scope(), "anything", &options)
        .await
        .expect_err("cancelled");
    assert!(matches!(error, Error::Cancelled), "{error:?}");
}

#[tokio::test]
async fn the_rail_advertises_and_dispatches_the_three_mem0_tools() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/memories/search/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "results": [{ "id": "s1", "memory": "likes rust", "score": 0.8 }],
        })))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/memories/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "results": [{ "id": "p1", "memory": "prefers tea" }],
        })))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/memories/"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "results": [{ "id": "c1", "memory": "fact" }],
        })))
        .mount(&server)
        .await;

    let handle =
        Mem0MemoryProvider::new(&config(&server.uri()).with_option("user_id", json!("alice")));

    let schemas = handle.get_tool_schemas();
    let names: Vec<&str> = schemas
        .iter()
        .map(|schema| schema["name"].as_str().expect("tool name"))
        .collect();
    assert_eq!(names, vec!["mem0_profile", "mem0_search", "mem0_conclude"]);
    assert_eq!(
        schemas[1]["parameters"]["properties"]["top_k"]["type"],
        json!("integer")
    );
    assert_eq!(schemas[1]["parameters"]["required"], json!(["query"]));
    assert_eq!(schemas[0]["parameters"]["required"], json!([]));

    let search = handle
        .handle_tool_call(SEARCH_TOOL, &json!({ "query": "q" }))
        .await;
    assert_eq!(search["count"], json!(1));
    assert_eq!(search["results"][0]["memory"], json!("likes rust"));

    let profile = handle.handle_tool_call(PROFILE_TOOL, &json!({})).await;
    assert_eq!(profile["count"], json!(1));
    assert_eq!(profile["result"], json!("prefers tea"));

    let conclusion = handle
        .handle_tool_call(CONCLUDE_TOOL, &json!({ "conclusion": "likes rust" }))
        .await;
    assert_eq!(conclusion["result"], json!("Fact stored."));

    let unknown = handle.handle_tool_call("drop_everything", &json!({})).await;
    assert!(unknown["error"]
        .as_str()
        .expect("error text")
        .contains("drop_everything"));

    let missing = handle.handle_tool_call(SEARCH_TOOL, &json!({})).await;
    assert!(missing["error"]
        .as_str()
        .expect("error text")
        .contains("query"));

    let prompt = handle.system_prompt_block();
    assert!(prompt.starts_with("# Mem0 Memory\n"), "{prompt}");
    assert!(prompt.contains("Active. User: alice."), "{prompt}");
    assert!(prompt.contains("mem0_search"), "{prompt}");

    let rendered = handle.prefetch("q").await;
    assert!(rendered.starts_with("## Mem0 Memory\n- "), "{rendered}");
    assert!(rendered.contains("likes rust"), "{rendered}");
}

#[test]
fn the_prompt_block_names_the_configured_user_or_a_placeholder() {
    let configured = Mem0MemoryProvider::new(
        &ProviderConfig::new(ProviderKind::Mem0).with_option("user_id", json!("alice")),
    );
    assert!(configured
        .system_prompt_block()
        .contains("Active. User: alice."));

    let unset = Mem0MemoryProvider::new(&ProviderConfig::new(ProviderKind::Mem0));
    assert!(unset.system_prompt_block().contains("Active. User: ?."));
}

#[tokio::test]
async fn the_rail_degrades_instead_of_raising_when_the_vendor_fails() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/memories/search/"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;

    let config = config(&server.uri()).with_option("max_retries", json!(0));
    let handle = Mem0MemoryProvider::new(&config);
    let payload = handle
        .handle_tool_call(SEARCH_TOOL, &json!({ "query": "q" }))
        .await;
    assert!(payload["error"].as_str().is_some(), "{payload}");
    assert_eq!(handle.prefetch("q").await, "");
}

#[test]
fn encode_segment_keeps_unreserved_bytes_and_encodes_the_rest() {
    assert_eq!(super::spec::encode_segment("abc-123_~."), "abc-123_~.");
    assert_eq!(super::spec::encode_segment("a/b?c#d"), "a%2Fb%3Fc%23d");
    assert_eq!(super::spec::encode_segment("é"), "%C3%A9");
}
