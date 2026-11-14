//! Offline wiremock tests for the `OpenViking` adapter: wire bodies, response decoding,
//! retries, failover, cancellation, and the tool rail.
use openjiuwen_common::error::Error;
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;
use wiremock::matchers::{header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::*;
use crate::external::provider::MemoryProvider;
use crate::record::{
    AddOptions, CallOptions, MemoryFilter, MemoryScope, MemoryUpdate, NewMemory, SearchOptions,
};

/// The session every configured test handle carries.
const SESSION: &str = "sess-1";
/// The credential every configured test handle carries.
const KEY: &str = "vk-key";
/// The `X-API-Key` header name wiremock matches.
const KEY_HEADER: &str = "x-api-key";

/// A configuration pointing at `uri` with a session id and the test credential.
fn config(uri: &str) -> ProviderConfig {
    ProviderConfig::new(ProviderKind::OpenViking)
        .with_endpoint(uri)
        .with_api_key(KEY)
        .with_option("session_id", json!(SESSION))
}

/// A configured handle for `uri`.
fn handle(uri: &str) -> OpenVikingMemoryProvider {
    OpenVikingMemoryProvider::new(&config(uri))
}

/// A configured handle that names no session id at all.
fn sessionless(uri: &str) -> OpenVikingMemoryProvider {
    OpenVikingMemoryProvider::new(
        &ProviderConfig::new(ProviderKind::OpenViking)
            .with_endpoint(uri)
            .with_api_key(KEY),
    )
}

/// The scope every test uses.
fn scope() -> MemoryScope {
    MemoryScope::new("alice", "workspace")
}

/// Reads one request body as JSON.
fn request_json(request: &wiremock::Request) -> Value {
    serde_json::from_slice(&request.body).expect("the request body is JSON")
}

/// The requests a server received (an empty vector when it received none).
async fn requests(server: &MockServer) -> Vec<wiremock::Request> {
    server.received_requests().await.unwrap_or_default()
}

#[tokio::test]
async fn search_posts_find_and_drops_hits_below_the_threshold() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/search/find"))
        .and(header(KEY_HEADER, KEY))
        .and(header("x-openviking-account", "default"))
        .and(header("x-openviking-agent", "hermes"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "result": {
                "memories": [{ "uri": "viking://m/1", "score": 0.91, "abstract": "alpha" }],
                "resources": [{ "uri": "viking://r/1", "score": 0.2, "abstract": "beta" }],
                "skills": [],
                "total": 2,
            },
        })))
        .expect(1)
        .mount(&server)
        .await;

    let options = SearchOptions::default().with_top_k(5).with_threshold(0.3);
    let hits = handle(&server.uri())
        .search(&scope(), "alpha", &options)
        .await
        .expect("search");
    assert_eq!(hits.len(), 1, "the resource below the threshold is dropped");
    assert_eq!(hits[0].record.id, "viking://m/1");
    assert_eq!(hits[0].record.content, "alpha");
    assert!((hits[0].score - 0.91).abs() < 1e-6);

    let requests = requests(&server).await;
    assert_eq!(requests.len(), 1);
    let body = request_json(&requests[0]);
    assert_eq!(body["query"], json!("alpha"));
    assert_eq!(body["top_k"], json!(5));
}

#[tokio::test]
async fn search_decodes_a_null_result_as_no_hits() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/search/find"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "result": null })))
        .mount(&server)
        .await;

    let hits = handle(&server.uri())
        .search(&scope(), "alpha", &SearchOptions::default())
        .await
        .expect("tolerant search");
    assert!(hits.is_empty(), "{hits:?}");
}

#[tokio::test]
async fn add_posts_one_remember_message_on_the_session() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/sessions/sess-1/messages"))
        .and(header(KEY_HEADER, KEY))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "result": { "ok": true } })))
        .expect(1)
        .mount(&server)
        .await;

    let outcome = handle(&server.uri())
        .add(
            &scope(),
            vec![NewMemory::semantic("alice prefers dark mode")],
            &AddOptions::default(),
        )
        .await
        .expect("add");
    assert_eq!(outcome.added.len(), 1);
    let record = &outcome.added[0];
    assert_eq!(record.content, "alice prefers dark mode");
    assert_eq!(
        record.id.len(),
        24,
        "a client-generated id is 24 hex characters"
    );
    assert_eq!(record.scope, scope());

    let requests = requests(&server).await;
    assert_eq!(requests.len(), 1);
    let body = request_json(&requests[0]);
    assert_eq!(body["role"], json!("user"));
    let text = body["parts"][0]["text"].as_str().expect("text part");
    assert!(text.starts_with("[Remember — "), "{text}");
    assert!(text.ends_with("] alice prefers dark mode"), "{text}");
    assert_eq!(body["parts"][0]["type"], json!("text"));
}

#[tokio::test]
async fn add_uses_the_metadata_category_and_the_scope_session() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/sessions/sess-scope/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .expect(1)
        .mount(&server)
        .await;

    let memory = NewMemory::semantic("deploy on friday").with_metadata("category", json!("event"));
    let outcome = sessionless(&server.uri())
        .add(
            &scope().session("sess-scope"),
            vec![memory],
            &AddOptions::default(),
        )
        .await
        .expect("add through the scope session");
    assert_eq!(outcome.added.len(), 1);

    let requests = requests(&server).await;
    let text = request_json(&requests[0])["parts"][0]["text"]
        .as_str()
        .expect("text part")
        .to_owned();
    assert_eq!(text, "[Remember — event] deploy on friday");
}

#[tokio::test]
async fn add_without_a_session_is_a_validation_error_and_sends_nothing() {
    let server = MockServer::start().await;
    let error = sessionless(&server.uri())
        .add(
            &scope(),
            vec![NewMemory::semantic("nowhere to write")],
            &AddOptions::default(),
        )
        .await
        .expect_err("a session is required");
    assert!(matches!(error, Error::Validation(_)), "{error:?}");
    let message = error.to_string();
    assert!(message.contains("session id"), "{message}");
    assert!(message.contains("MemoryScope::session_id"), "{message}");
    assert!(message.contains("options[\"session_id\"]"), "{message}");
    assert!(requests(&server).await.is_empty(), "no request may be sent");
}

#[tokio::test]
async fn get_reads_the_content_at_a_uri() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/content/read"))
        .and(query_param("uri", "viking://m/1"))
        .and(header(KEY_HEADER, KEY))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "result": "hello" })))
        .expect(1)
        .mount(&server)
        .await;

    let record = handle(&server.uri())
        .get(&scope(), "viking://m/1")
        .await
        .expect("get")
        .expect("record");
    assert_eq!(record.id, "viking://m/1");
    assert_eq!(record.content, "hello");

    let nested = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/content/read"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({ "result": { "content": "hi" } })),
        )
        .mount(&nested)
        .await;
    let record = handle(&nested.uri())
        .get(&scope(), "viking://m/2")
        .await
        .expect("get")
        .expect("record");
    assert_eq!(record.id, "viking://m/2");
    assert_eq!(record.content, "hi");
}

#[tokio::test]
async fn get_truncates_long_content_like_the_python_tool() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/content/read"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({ "result": "x".repeat(9000) })),
        )
        .mount(&server)
        .await;

    let record = handle(&server.uri())
        .get(&scope(), "viking://m/long")
        .await
        .expect("get")
        .expect("record");
    assert_eq!(
        record.content,
        format!(
            "{}\n\n[... truncated, use a more specific URI or abstract level]",
            "x".repeat(8000)
        )
    );
}

#[tokio::test]
async fn get_reports_a_blank_result_as_no_record() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/content/read"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "result": "" })))
        .mount(&server)
        .await;

    assert!(handle(&server.uri())
        .get(&scope(), "viking://m/1")
        .await
        .expect("get")
        .is_none());
}

#[tokio::test]
async fn list_skips_directories_and_maps_the_files() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/fs/ls"))
        .and(query_param("uri", "viking://"))
        .and(header(KEY_HEADER, KEY))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "result": [
                { "rel_path": "notes", "uri": "viking://notes", "isDir": true, "abstract": "a directory" },
                { "rel_path": "notes/today.md", "uri": "viking://notes/today.md", "isDir": false, "abstract": "today's note" },
            ],
        })))
        .expect(1)
        .mount(&server)
        .await;

    let records = handle(&server.uri())
        .list(&MemoryFilter::for_scope(&scope()))
        .await
        .expect("list");
    assert_eq!(records.len(), 1, "the directory is skipped");
    assert_eq!(records[0].id, "viking://notes/today.md");
    assert_eq!(records[0].content, "today's note");

    let custom_root = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/fs/ls"))
        .and(query_param("uri", "viking://notes"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "result": [{ "name": "todo.md", "uri": "viking://notes/todo.md" }],
        })))
        .mount(&custom_root)
        .await;
    let config = config(&custom_root.uri()).with_option("root_uri", json!("viking://notes"));
    let records = OpenVikingMemoryProvider::new(&config)
        .list(&MemoryFilter::for_scope(&scope()))
        .await
        .expect("list a custom root");
    assert_eq!(records.len(), 1);
    assert_eq!(
        records[0].content, "todo.md",
        "the name is the content fallback"
    );
}

#[tokio::test]
async fn list_decodes_a_non_list_result_as_an_empty_listing() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/fs/ls"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({ "result": { "uri": "viking://" } })),
        )
        .mount(&server)
        .await;
    let records = handle(&server.uri())
        .list(&MemoryFilter::for_scope(&scope()))
        .await
        .expect("list");
    assert!(records.is_empty(), "a non-list result is an empty listing");
}

#[tokio::test]
async fn delete_and_update_are_typed_unsupported_without_a_request() {
    let server = MockServer::start().await;
    let handle = handle(&server.uri());

    let error = handle
        .delete(&scope(), "viking://m/1")
        .await
        .expect_err("delete is unsupported");
    assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
    let message = error.to_string();
    assert!(message.contains("has no delete operation"), "{message}");
    assert!(message.contains("no memory-delete endpoint"), "{message}");
    assert!(
        message.contains("POST /api/v1/sessions/{session_id}/commit"),
        "{message}"
    );

    let error = handle
        .update(
            &scope(),
            "viking://m/1",
            MemoryUpdate::content("rewritten"),
            &CallOptions::default(),
        )
        .await
        .expect_err("update is unsupported");
    assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
    let message = error.to_string();
    assert!(message.contains("has no update operation"), "{message}");
    assert!(message.contains("no memory-update endpoint"), "{message}");

    assert!(requests(&server).await.is_empty(), "no request may be sent");
}

#[tokio::test]
async fn clear_removes_nothing_and_reports_the_missing_endpoint() {
    let server = MockServer::start().await;
    // The transport falls back to list-then-delete for a spec without a bulk delete; the
    // listing succeeds and every delete is unsupported.
    Mock::given(method("GET"))
        .and(path("/api/v1/fs/ls"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "result": [{ "rel_path": "a.md", "uri": "viking://a.md", "isDir": false }],
        })))
        .mount(&server)
        .await;

    let error = handle(&server.uri())
        .clear(&MemoryFilter::for_scope(&scope()))
        .await
        .expect_err("clear is unsupported");
    assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
    assert!(
        error.to_string().contains("no memory-delete endpoint"),
        "{error}"
    );
}

#[tokio::test]
async fn health_probes_slash_health_and_reports_a_degraded_answer() {
    let healthy = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/health"))
        .and(header(KEY_HEADER, KEY))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "status": "ok" })))
        .expect(1)
        .mount(&healthy)
        .await;
    let report = handle(&healthy.uri()).health().await.expect("health");
    assert!(report.healthy, "{report:?}");
    assert_eq!(report.detail, None);

    let degraded = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&degraded)
        .await;
    let config = config(&degraded.uri()).with_option("max_retries", json!(0));
    let report = OpenVikingMemoryProvider::new(&config)
        .health()
        .await
        .expect("degraded health is not an error");
    assert!(!report.healthy, "{report:?}");
    assert!(report.detail.is_some(), "{report:?}");
}

#[tokio::test]
async fn search_retries_a_transient_status_once() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/search/find"))
        .respond_with(ResponseTemplate::new(503))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/v1/search/find"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "result": { "memories": [{ "uri": "viking://m/1", "score": 0.8, "abstract": "alpha" }] },
        })))
        .mount(&server)
        .await;

    let config = config(&server.uri())
        .with_option("max_retries", json!(1))
        .with_option("retry_backoff_ms", json!(1));
    let hits = OpenVikingMemoryProvider::new(&config)
        .search(&scope(), "alpha", &SearchOptions::default())
        .await
        .expect("search after retry");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].record.id, "viking://m/1");
    assert_eq!(requests(&server).await.len(), 2);
}

#[tokio::test]
async fn search_fails_over_to_the_second_endpoint() {
    let primary = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/search/find"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&primary)
        .await;
    let secondary = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/search/find"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "result": { "resources": [{ "uri": "viking://r/1", "score": 0.7, "abstract": "beta" }] },
        })))
        .mount(&secondary)
        .await;

    let config = config(&primary.uri())
        .with_option("failover_endpoints", json!([secondary.uri()]))
        .with_option("max_retries", json!(0))
        .with_option("retry_backoff_ms", json!(1));
    let hits = OpenVikingMemoryProvider::new(&config)
        .search(&scope(), "beta", &SearchOptions::default())
        .await
        .expect("failover search");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].record.id, "viking://r/1");
    assert_eq!(requests(&secondary).await.len(), 1);
}

#[tokio::test]
async fn an_unconfigured_handle_names_the_missing_client_and_the_endpoint() {
    let handle = OpenVikingMemoryProvider::new(
        &ProviderConfig::new(ProviderKind::OpenViking).with_api_key(KEY),
    );
    assert_eq!(handle.name(), "openviking");
    assert_eq!(handle.kind(), ProviderKind::OpenViking);
    assert!(!handle.is_available());
    assert!(!handle.capabilities().semantic_search);

    let error = handle
        .search(&scope(), "anything", &SearchOptions::default())
        .await
        .expect_err("unconfigured handle");
    assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
    let message = error.to_string();
    assert!(message.contains("OpenViking API client"), "{message}");
    assert!(message.contains("no `endpoint`"), "{message}");

    // The rail reports the same typed error rather than an empty rendering.
    assert_eq!(handle.prefetch("anything").await, "");
    let payload = handle
        .handle_tool_call(SEARCH_TOOL, &json!({ "query": "anything" }))
        .await;
    assert!(payload["error"]
        .as_str()
        .expect("error text")
        .contains("OpenViking API client"));

    let configured = handle_with_endpoint();
    let capabilities = configured.capabilities();
    assert!(capabilities.semantic_search);
    assert!(!capabilities.keyword_search);
    assert!(!capabilities.pagination);
    assert!(configured.is_available());
}

/// A configured handle for an endpoint no test connects to.
fn handle_with_endpoint() -> OpenVikingMemoryProvider {
    OpenVikingMemoryProvider::new(&config("https://viking.invalid"))
}

#[tokio::test]
async fn a_cancelled_call_never_reaches_the_vendor() {
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
async fn the_rail_advertises_the_five_viking_tools() {
    let schemas = handle("https://viking.invalid").get_tool_schemas();
    let names: Vec<&str> = schemas
        .iter()
        .map(|schema| schema["name"].as_str().expect("tool name"))
        .collect();
    assert_eq!(
        names,
        vec![
            "viking_search",
            "viking_read",
            "viking_browse",
            "viking_remember",
            "viking_add_resource",
        ]
    );
    assert_eq!(
        schemas[0]["parameters"]["properties"]["mode"]["enum"],
        json!(["auto", "fast", "deep"])
    );
    assert_eq!(
        schemas[1]["parameters"]["properties"]["detail"]["enum"],
        json!(["abstract", "overview", "full"])
    );
    assert_eq!(schemas[2]["parameters"]["required"], json!(["action"]));
    assert_eq!(
        schemas[3]["parameters"]["properties"]["category"]["enum"],
        json!(["preference", "entity", "event", "case", "pattern"])
    );
    assert_eq!(schemas[4]["parameters"]["required"], json!(["url"]));

    let prompt = handle("https://viking.invalid").system_prompt_block();
    assert!(prompt.starts_with("# OpenViking Memory\n"), "{prompt}");
    assert!(prompt.contains("viking_add_resource"), "{prompt}");
}

#[tokio::test]
async fn the_rail_search_tool_reports_typed_results_and_a_total() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/search/find"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "result": {
                "memories": [{ "uri": "viking://m/1", "score": 0.91, "abstract": "alpha" }],
                "resources": [{ "uri": "viking://r/1", "score": 0.5, "abstract": "beta" }],
                "skills": [{ "uri": "viking://s/1", "score": 0.2, "abstract": "gamma" }],
                "total": 3,
            },
        })))
        .expect(1)
        .mount(&server)
        .await;

    let payload = handle(&server.uri())
        .handle_tool_call(SEARCH_TOOL, &json!({ "query": "alpha" }))
        .await;
    assert_eq!(payload["total"], json!(3));
    let results = payload["results"].as_array().expect("results");
    assert_eq!(results.len(), 3);
    assert_eq!(results[0]["uri"], json!("viking://m/1"));
    assert_eq!(results[0]["type"], json!("memory"));
    assert_eq!(results[0]["score"], json!(0.91));
    assert_eq!(results[1]["type"], json!("resource"));
    assert_eq!(results[2]["type"], json!("skill"));

    let missing = handle(&server.uri())
        .handle_tool_call(SEARCH_TOOL, &json!({}))
        .await;
    assert!(missing["error"]
        .as_str()
        .expect("error text")
        .contains("query"));

    let unknown = handle(&server.uri())
        .handle_tool_call("viking_teleport", &json!({}))
        .await;
    assert_eq!(unknown["error"], json!("unknown tool: viking_teleport"));
}

#[tokio::test]
async fn the_rail_read_and_browse_tools_use_the_documented_endpoints() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/content/read"))
        .and(query_param("uri", "viking://m/1"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({ "result": { "content": "full text" } })),
        )
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/v1/fs/ls"))
        .and(query_param("uri", "viking://"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "result": [
                { "rel_path": "notes", "uri": "viking://notes", "isDir": true },
                { "rel_path": "notes/today.md", "uri": "viking://notes/today.md", "isDir": false, "abstract": "a note" },
            ],
        })))
        .mount(&server)
        .await;

    let read = handle(&server.uri())
        .handle_tool_call(
            READ_TOOL,
            &json!({ "uri": "viking://m/1", "detail": "full" }),
        )
        .await;
    assert_eq!(read["uri"], json!("viking://m/1"));
    assert_eq!(read["level"], json!("full"));
    assert_eq!(read["content"], json!("full text"));

    let browse = handle(&server.uri())
        .handle_tool_call(BROWSE_TOOL, &json!({ "action": "list" }))
        .await;
    assert_eq!(browse["path"], json!("viking://"));
    let entries = browse["entries"].as_array().expect("entries");
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0]["type"], json!("dir"));
    assert_eq!(entries[1]["type"], json!("file"));
    assert_eq!(entries[1]["abstract"], json!("a note"));
}

#[tokio::test]
async fn the_rail_remember_tool_posts_a_session_message() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/sessions/sess-1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .expect(1)
        .mount(&server)
        .await;

    let payload = handle(&server.uri())
        .handle_tool_call(
            REMEMBER_TOOL,
            &json!({ "content": "alice prefers tea", "category": "preference" }),
        )
        .await;
    assert_eq!(payload["status"], json!("stored"));
    assert_eq!(payload["message"], json!(REMEMBER_MESSAGE));

    let requests = requests(&server).await;
    let body = request_json(&requests[0]);
    assert_eq!(body["role"], json!("user"));
    assert_eq!(
        body["parts"][0]["text"],
        json!("[Remember — preference] alice prefers tea")
    );
}

#[tokio::test]
async fn the_rail_add_resource_tool_indexes_a_url() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/resources"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "result": { "root_uri": "viking://resources/doc" },
        })))
        .expect(1)
        .mount(&server)
        .await;

    let payload = handle(&server.uri())
        .handle_tool_call(
            ADD_RESOURCE_TOOL,
            &json!({ "url": "https://example.com/doc" }),
        )
        .await;
    assert_eq!(payload["status"], json!("added"));
    assert_eq!(payload["root_uri"], json!("viking://resources/doc"));

    let requests = requests(&server).await;
    assert_eq!(
        request_json(&requests[0])["path"],
        json!("https://example.com/doc")
    );
}

#[tokio::test]
async fn prefetch_renders_memory_then_resource_context() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/search/find"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "result": {
                "memories": [
                    { "uri": "viking://m/1", "score": 0.9123, "abstract": "alpha" },
                    { "uri": "viking://m/2", "score": 0.8, "abstract": "" },
                    { "uri": "viking://m/3", "score": 0.7, "abstract": "third" },
                    { "uri": "viking://m/4", "score": 0.6, "abstract": "fourth" },
                ],
                "resources": [{ "uri": "viking://r/1", "score": 0.5, "abstract": "beta" }],
            },
        })))
        .expect(1)
        .mount(&server)
        .await;

    let rendered = handle(&server.uri()).prefetch("alpha").await;
    assert!(
        rendered.starts_with("## OpenViking Context\n"),
        "{rendered}"
    );
    assert!(
        rendered.contains("- [0.912] alpha (viking://m/1)"),
        "{rendered}"
    );
    assert!(!rendered.contains("(viking://m/2)"), "{rendered}");
    assert!(
        rendered.contains("- [0.700] third (viking://m/3)"),
        "{rendered}"
    );
    assert!(!rendered.contains("fourth"), "at most three items per list");
    assert!(
        rendered.contains("- [0.500] beta (viking://r/1)"),
        "{rendered}"
    );
    assert!(rendered.ends_with("(viking://r/1)"), "resources come last");

    let requests = requests(&server).await;
    let body = request_json(&requests[0]);
    assert_eq!(body["query"], json!("alpha"));
    assert_eq!(body["top_k"], json!(5));

    assert_eq!(handle(&server.uri()).prefetch("  ").await, "");
}

#[tokio::test]
async fn sync_turn_posts_the_user_then_the_assistant_message() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/sessions/sess-1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .expect(2)
        .mount(&server)
        .await;

    handle(&server.uri())
        .sync_turn("what is rust?", "a systems language")
        .await
        .expect("sync turn");

    let requests = requests(&server).await;
    assert_eq!(requests.len(), 2);
    assert_eq!(request_json(&requests[0])["role"], json!("user"));
    assert_eq!(
        request_json(&requests[0])["content"],
        json!("what is rust?")
    );
    assert_eq!(request_json(&requests[1])["role"], json!("assistant"));
    assert_eq!(
        request_json(&requests[1])["content"],
        json!("a systems language")
    );
}

#[tokio::test]
async fn sync_turn_skips_a_blank_turn_and_requires_a_session() {
    let server = MockServer::start().await;
    handle(&server.uri())
        .sync_turn("   ", "an answer")
        .await
        .expect("a blank user message is a no-op");
    assert!(requests(&server).await.is_empty());

    let error = sessionless(&server.uri())
        .sync_turn("a question", "an answer")
        .await
        .expect_err("a session is required");
    assert!(matches!(error, Error::Validation(_)), "{error:?}");
    assert!(error.to_string().contains("session id"), "{error}");
    assert!(requests(&server).await.is_empty());
}

#[tokio::test]
async fn on_session_end_commits_the_session() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/sessions/sess-1/commit"))
        .and(header(KEY_HEADER, KEY))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({ "result": { "committed": true } })),
        )
        .expect(1)
        .mount(&server)
        .await;

    handle(&server.uri())
        .on_session_end(&[ConversationTurn::new("user", "hi")])
        .await
        .expect("commit");
    let sent = requests(&server).await;
    assert_eq!(request_json(&sent[0]), json!({}));

    // Without a session id there is nothing to commit and no request is sent.
    let bare = MockServer::start().await;
    sessionless(&bare.uri())
        .on_session_end(&[])
        .await
        .expect("no session is a no-op");
    assert!(requests(&bare).await.is_empty());
}

#[tokio::test]
async fn a_failed_tool_call_becomes_an_error_payload() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/search/find"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;

    let config = config(&server.uri()).with_option("max_retries", json!(0));
    let handle = OpenVikingMemoryProvider::new(&config);
    let payload = handle
        .handle_tool_call(SEARCH_TOOL, &json!({ "query": "alpha" }))
        .await;
    assert!(payload["error"].as_str().is_some(), "{payload}");
    assert_eq!(handle.prefetch("alpha").await, "");
}

#[test]
fn truncate_chars_keeps_char_boundaries() {
    assert_eq!(spec::truncate_chars("abcdef", 3), "abc");
    assert_eq!(spec::truncate_chars("abc", 3), "abc");
    assert_eq!(spec::truncate_chars("abc", 10), "abc");
    assert_eq!(spec::truncate_chars("ééé", 2), "éé");
    assert_eq!(spec::truncate_chars("", 4), "");
}

#[test]
fn render_context_ignores_abstractless_and_excess_items() {
    let answer = json!({
        "result": {
            "memories": [{ "uri": "viking://m/1", "score": 1.0 }],
            "resources": [{ "uri": "viking://r/1", "score": 0.5, "abstract": "beta" }],
        },
    });
    assert_eq!(
        render_context(&answer),
        "## OpenViking Context\n- [0.500] beta (viking://r/1)"
    );
    assert_eq!(render_context(&json!({ "result": null })), "");
    assert_eq!(render_context(&json!({})), "");
}

#[test]
fn directory_detection_accepts_both_spellings() {
    assert!(spec::is_directory(&json!({ "isDir": true })));
    assert!(spec::is_directory(&json!({ "is_dir": true })));
    assert!(!spec::is_directory(&json!({ "isDir": false })));
    assert!(!spec::is_directory(&json!({ "uri": "viking://a" })));
    assert!(!spec::is_directory(&json!({ "isDir": "true" })));
}
