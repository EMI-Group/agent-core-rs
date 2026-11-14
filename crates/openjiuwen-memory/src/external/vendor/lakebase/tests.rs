//! Offline wiremock tests for the `LakeBase` adapter: wire bodies, decoding, retries,
//! failover, status mapping, cancellation, and the tool rail.
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
    ProviderConfig::new(ProviderKind::LakeBase)
        .with_endpoint(uri)
        .with_api_key("lkb-key")
}

/// A configured handle for `uri`.
fn handle(uri: &str) -> LakeBaseMemoryProvider {
    LakeBaseMemoryProvider::new(&config(uri))
}

/// The scope every test uses.
fn scope() -> MemoryScope {
    MemoryScope::new("alice", "workspace")
}

/// Reads one request body as JSON.
fn request_json(request: &wiremock::Request) -> Value {
    serde_json::from_slice(&request.body).expect("the request body is JSON")
}

/// The number of requests `server` has received.
async fn request_count(server: &MockServer) -> usize {
    server.received_requests().await.expect("requests").len()
}

#[tokio::test]
async fn add_ingests_one_memory_and_keeps_the_vendor_id() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/memory/bases/mem_default/ingest"))
        .and(header("authorization", "Bearer lkb-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "memory_id": 42,
            "memory_type": "fact",
        })))
        .expect(1)
        .mount(&server)
        .await;

    let outcome = handle(&server.uri())
        .add(
            &scope(),
            vec![NewMemory::semantic("alice prefers dark mode")
                .with_importance(0.75)
                .with_metadata("origin", json!("api"))],
            &AddOptions::default(),
        )
        .await
        .expect("add");

    assert_eq!(outcome.added.len(), 1);
    let record = &outcome.added[0];
    assert_eq!(record.id, "42");
    assert_eq!(record.content, "alice prefers dark mode");
    assert_eq!(record.kind, MemoryType::SemanticMemory);
    assert!((record.importance - 0.75).abs() < f32::EPSILON);
    assert_eq!(record.metadata_str("memory_type"), Some("fact"));
    assert_eq!(record.metadata_str("origin"), Some("api"));
    assert_eq!(record.scope, scope());

    let requests = server.received_requests().await.expect("requests");
    assert_eq!(requests.len(), 1);
    let body = request_json(&requests[0]);
    assert_eq!(body["content"], json!("alice prefers dark mode"));
    assert_eq!(body["role"], json!("user"));
    assert_eq!(body["memory_type"], json!("fact"));
    assert_eq!(body["metadata"], json!({ "origin": "api" }));
    let importance = body["importance"].as_f64().expect("importance");
    assert!((importance - 0.75).abs() < 1e-6, "{importance}");
}

#[tokio::test]
async fn add_prefers_a_metadata_memory_type_over_the_kind_mapping() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/memory/bases/mem_default/ingest"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({ "memory_id": "m-9", "memory_type": "decision" })),
        )
        .mount(&server)
        .await;

    let outcome = handle(&server.uri())
        .add(
            &scope(),
            vec![NewMemory::semantic("we chose tokio")
                .with_metadata("memory_type", json!("decision"))],
            &AddOptions::default(),
        )
        .await
        .expect("add");

    assert_eq!(outcome.added[0].id, "m-9");
    assert_eq!(
        outcome.added[0].metadata_str("memory_type"),
        Some("decision")
    );
    let body = request_json(&server.received_requests().await.expect("requests")[0]);
    assert_eq!(body["memory_type"], json!("decision"));
    assert_eq!(body["metadata"], json!({ "memory_type": "decision" }));
    assert_eq!(body["importance"].as_f64().expect("importance"), 0.5);
}

#[tokio::test]
async fn search_maps_the_vendor_types_and_drops_hits_under_the_threshold() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/memory/bases/mem_default/recall"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "memories": [
                { "content": "alpha", "memory_type": "fact", "score": 0.8 },
                { "content": "beta", "memory_type": "episode", "score": 0.1 },
            ],
        })))
        .mount(&server)
        .await;

    let handle = handle(&server.uri());
    let hits = handle
        .search(
            &scope(),
            "theme",
            &SearchOptions::default().with_threshold(0.3),
        )
        .await
        .expect("search");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].record.content, "alpha");
    assert!((hits[0].score - 0.8).abs() < 1e-6);
    assert_eq!(hits[0].record.kind, MemoryType::SemanticMemory);
    assert_eq!(hits[0].record.metadata_str("memory_type"), Some("fact"));
    assert_eq!(hits[0].record.scope, scope());

    let requests = server.received_requests().await.expect("requests");
    let body = request_json(&requests[0]);
    assert_eq!(body["query"], json!("theme"));
    assert_eq!(body["top_k"], json!(5));
    assert_eq!(body["memory_types"], Value::Null);

    let hits = handle
        .search(
            &scope(),
            "theme",
            &SearchOptions::default()
                .with_top_k(10)
                .with_kinds(vec![MemoryType::EpisodicMemory, MemoryType::Coding]),
        )
        .await
        .expect("filtered search");
    assert_eq!(hits.len(), 1);
    let requests = server.received_requests().await.expect("requests");
    let body = request_json(&requests[1]);
    assert_eq!(body["memory_types"], json!(["episode", "convention"]));
    assert_eq!(body["top_k"], json!(10));
}

#[tokio::test]
async fn list_pages_with_the_configured_offset_and_limit_and_honours_base_id() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/memory/bases/mem_default/memories"))
        .and(query_param("offset", "0"))
        .and(query_param("limit", "20"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "memories": [
                { "memory_id": 1, "content": "one", "memory_type": "fact" },
                { "memory_id": 2, "content": "two", "memory_type": "episode" },
            ],
            "total": 2,
        })))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/memory/bases/other_base/memories"))
        .and(query_param("offset", "2"))
        .and(query_param("limit", "5"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "memories": [{ "memory_id": 3, "content": "three", "memory_type": "convention" }],
            "total": 3,
        })))
        .mount(&server)
        .await;

    let records = handle(&server.uri())
        .list(&MemoryFilter::for_scope(&scope()))
        .await
        .expect("list");
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].id, "1");
    assert_eq!(records[0].content, "one");
    assert_eq!(records[0].kind, MemoryType::SemanticMemory);
    assert_eq!(records[1].id, "2");
    assert_eq!(records[1].kind, MemoryType::EpisodicMemory);

    let other = LakeBaseMemoryProvider::new(
        &config(&server.uri())
            .with_option("base_id", json!("other_base"))
            .with_option("offset", json!(2))
            .with_option("limit", json!(5)),
    );
    let records = other
        .list(&MemoryFilter::for_scope(&scope()))
        .await
        .expect("other base list");
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].id, "3");
    assert_eq!(records[0].kind, MemoryType::Coding);
}

#[tokio::test]
async fn get_and_delete_map_ids_and_update_is_not_part_of_the_api() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/memory/bases/mem_default/memories/7"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "memory_id": 7,
            "content": "stored",
            "memory_type": "convention",
        })))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/memory/bases/mem_default/memories/99"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .and(path("/memory/bases/mem_default/memories/7"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .and(path("/memory/bases/mem_default/memories/99"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;

    let handle = handle(&server.uri());
    let record = handle
        .get(&scope(), "7")
        .await
        .expect("get")
        .expect("record");
    assert_eq!(record.id, "7");
    assert_eq!(record.content, "stored");
    assert_eq!(record.kind, MemoryType::Coding);
    assert_eq!(record.metadata_str("memory_type"), Some("convention"));

    assert!(handle
        .get(&scope(), "99")
        .await
        .expect("missing get")
        .is_none());
    assert!(handle.delete(&scope(), "7").await.expect("delete"));
    assert!(!handle.delete(&scope(), "99").await.expect("delete missing"));

    let before = request_count(&server).await;
    let error = handle
        .get(&scope(), "not-a-number")
        .await
        .expect_err("non-numeric id");
    assert!(matches!(error, Error::Validation(_)), "{error:?}");
    assert!(error.to_string().contains("not-a-number"), "{error}");
    assert_eq!(
        request_count(&server).await,
        before,
        "the id never reaches the wire"
    );

    let error = handle
        .update(
            &scope(),
            "7",
            MemoryUpdate::content("new text"),
            &CallOptions::default(),
        )
        .await
        .expect_err("update");
    assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
    let message = error.to_string();
    assert!(message.contains("has no update operation"), "{message}");
    assert!(
        message.contains("PUT /memory/bases/{base_id}/memories/{id}"),
        "{message}"
    );
    assert_eq!(request_count(&server).await, before, "update sends nothing");
}

#[tokio::test]
async fn search_retries_a_transient_status_once() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/memory/bases/mem_default/recall"))
        .respond_with(ResponseTemplate::new(503))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/memory/bases/mem_default/recall"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "memories": [{ "content": "alpha", "memory_type": "fact", "score": 0.8 }],
        })))
        .mount(&server)
        .await;

    let handle = LakeBaseMemoryProvider::new(
        &config(&server.uri())
            .with_option("max_retries", json!(1))
            .with_option("retry_backoff_ms", json!(1)),
    );
    let hits = handle
        .search(&scope(), "alpha", &SearchOptions::default())
        .await
        .expect("search after retry");
    assert_eq!(hits.len(), 1);
    assert_eq!(request_count(&server).await, 2);
}

#[tokio::test]
async fn search_fails_over_to_the_second_endpoint() {
    let primary = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/memory/bases/mem_default/recall"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&primary)
        .await;
    let secondary = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/memory/bases/mem_default/recall"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "memories": [{ "content": "alpha", "memory_type": "episode", "score": 0.7 }],
        })))
        .mount(&secondary)
        .await;

    let handle = LakeBaseMemoryProvider::new(
        &config(&primary.uri())
            .with_option("failover_endpoints", json!([secondary.uri()]))
            .with_option("max_retries", json!(0))
            .with_option("retry_backoff_ms", json!(1)),
    );
    let hits = handle
        .search(&scope(), "alpha", &SearchOptions::default())
        .await
        .expect("failover search");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].record.content, "alpha");
    assert_eq!(hits[0].record.kind, MemoryType::EpisodicMemory);
    assert_eq!(request_count(&secondary).await, 1);
}

#[tokio::test]
async fn health_probes_the_stats_endpoint_and_degrades_instead_of_failing() {
    let healthy = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/memory/bases/mem_default/stats"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "count": 1 })))
        .expect(1)
        .mount(&healthy)
        .await;
    let report = handle(&healthy.uri()).health().await.expect("health");
    assert!(report.healthy, "{report:?}");

    let failing = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/memory/bases/mem_default/stats"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&failing)
        .await;
    let handle =
        LakeBaseMemoryProvider::new(&config(&failing.uri()).with_option("max_retries", json!(0)));
    let report = handle.health().await.expect("degraded health");
    assert!(!report.healthy);
    assert!(
        report
            .detail
            .as_deref()
            .is_some_and(|detail| detail.contains("500")),
        "{report:?}"
    );
}

#[tokio::test]
async fn an_unconfigured_handle_names_the_transport_and_capabilities_follow_configuration() {
    let configured = handle("https://lakebase.invalid");
    let capabilities = configured.capabilities();
    assert!(capabilities.semantic_search);
    assert!(capabilities.pagination);
    assert!(!capabilities.keyword_search);
    assert!(!capabilities.graph);
    assert!(!capabilities.variables);
    assert!(!capabilities.message_history);
    assert_eq!(capabilities.max_batch, None);
    assert!(configured.is_available());
    assert_eq!(configured.name(), "lakebase");
    assert_eq!(configured.kind(), ProviderKind::LakeBase);

    let unconfigured = LakeBaseMemoryProvider::new(
        &ProviderConfig::new(ProviderKind::LakeBase).with_api_key("lkb-key"),
    );
    assert!(!unconfigured.capabilities().semantic_search);
    assert!(!unconfigured.is_available());
    let error = unconfigured
        .search(&scope(), "anything", &SearchOptions::default())
        .await
        .expect_err("unconfigured handle");
    assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
    let message = error.to_string();
    assert!(message.contains("httpx.AsyncClient"), "{message}");
    assert!(message.contains("no `endpoint`"), "{message}");
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

/// Mounts the six wire endpoints the rail exercises.
async fn mount_rail_backend(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/memory/bases/mem_default/recall"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "memories": [{ "content": "likes rust", "memory_type": "fact", "score": 0.8 }],
        })))
        .mount(server)
        .await;
    Mock::given(method("POST"))
        .and(path("/memory/bases/mem_default/ingest"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({ "memory_id": 7, "memory_type": "episode" })),
        )
        .mount(server)
        .await;
    Mock::given(method("GET"))
        .and(path("/memory/bases/mem_default/memories"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "memories": [{ "memory_id": 7, "content": "likes rust", "memory_type": "fact" }],
            "total": 1,
        })))
        .mount(server)
        .await;
    Mock::given(method("GET"))
        .and(path("/memory/bases/mem_default/memories/7"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "memory_id": 7,
            "content": "likes rust",
            "memory_type": "fact",
        })))
        .mount(server)
        .await;
    Mock::given(method("DELETE"))
        .and(path("/memory/bases/mem_default/memories/7"))
        .respond_with(ResponseTemplate::new(200))
        .mount(server)
        .await;
    Mock::given(method("GET"))
        .and(path("/memory/bases/mem_default/stats"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "count": 1 })))
        .mount(server)
        .await;
}

#[test]
fn the_rail_advertises_the_six_lakebase_tool_specs() {
    let schemas = tool_schemas();
    let names: Vec<&str> = schemas
        .iter()
        .map(|schema| schema["name"].as_str().expect("tool name"))
        .collect();
    assert_eq!(
        names,
        vec![
            SEARCH_TOOL,
            ADD_TOOL,
            LIST_TOOL,
            GET_TOOL,
            DELETE_TOOL,
            STATS_TOOL
        ]
    );
    assert_eq!(schemas[0]["parameters"]["type"], json!("object"));
    assert_eq!(schemas[0]["parameters"]["required"], json!(["query"]));
    assert_eq!(
        schemas[0]["parameters"]["properties"]["top_k"]["type"],
        json!("integer")
    );
    assert_eq!(
        schemas[0]["parameters"]["properties"]["memory_types"]["items"]["type"],
        json!("string")
    );
    assert_eq!(
        schemas[0]["parameters"]["properties"]["memory_types"]["enum"],
        json!(MEMORY_TYPES)
    );
    assert_eq!(
        schemas[1]["parameters"]["properties"]["memory_type"]["enum"],
        json!(MEMORY_TYPES)
    );
    assert_eq!(
        schemas[1]["parameters"]["properties"]["importance"]["type"],
        json!("number")
    );
    assert_eq!(
        schemas[1]["parameters"]["properties"]["metadata"]["type"],
        json!("object")
    );
    assert_eq!(schemas[1]["parameters"]["required"], json!(["content"]));
    assert_eq!(schemas[2]["parameters"]["required"], json!([]));
    assert_eq!(
        schemas[3]["parameters"]["properties"]["memory_id"]["type"],
        json!("integer")
    );
    assert_eq!(schemas[5]["parameters"]["properties"], json!({}));
}

#[tokio::test]
async fn the_rail_dispatches_the_lakebase_tools_over_the_record_spi() {
    let server = MockServer::start().await;
    mount_rail_backend(&server).await;
    let handle =
        LakeBaseMemoryProvider::new(&config(&server.uri()).with_option("user_id", json!("alice")));

    let search = handle
        .handle_tool_call(SEARCH_TOOL, &json!({ "query": "rust" }))
        .await;
    assert_eq!(search["count"], json!(1));
    assert_eq!(search["memories"][0]["content"], json!("likes rust"));
    assert_eq!(search["memories"][0]["memory_type"], json!("fact"));

    let added = handle
        .handle_tool_call(ADD_TOOL, &json!({ "content": "likes rust" }))
        .await;
    assert_eq!(added["memory_id"], json!("7"));
    assert_eq!(added["memory_type"], json!("episode"));
    assert_eq!(added["base_id"], json!("mem_default"));

    let listed = handle.handle_tool_call(LIST_TOOL, &json!({})).await;
    assert_eq!(listed["count"], json!(1));
    assert_eq!(listed["memories"][0]["memory_id"], json!("7"));
    assert_eq!(listed["base_id"], json!("mem_default"));

    let fetched = handle
        .handle_tool_call(GET_TOOL, &json!({ "memory_id": 7 }))
        .await;
    assert_eq!(fetched["id"], json!("7"));
    assert_eq!(fetched["memory_type"], json!("fact"));
    assert_eq!(fetched["base_id"], json!("mem_default"));

    let deleted = handle
        .handle_tool_call(DELETE_TOOL, &json!({ "memory_id": 7 }))
        .await;
    assert_eq!(deleted["deleted"], json!(true));
    assert_eq!(deleted["memory_id"], json!(7));
    assert_eq!(deleted["base_id"], json!("mem_default"));

    let stats = handle.handle_tool_call(STATS_TOOL, &json!({})).await;
    assert_eq!(stats["healthy"], json!(true));
    assert_eq!(stats["base_id"], json!("mem_default"));

    let unknown = handle
        .handle_tool_call("lkb_memory_digest", &json!({}))
        .await;
    assert!(unknown["error"]
        .as_str()
        .expect("error text")
        .contains("lkb_memory_digest"));

    let missing = handle.handle_tool_call(SEARCH_TOOL, &json!({})).await;
    assert!(missing["error"]
        .as_str()
        .expect("error text")
        .contains("query"));
    assert_eq!(handle.default_scope().user_id, "alice");
}

#[tokio::test]
async fn the_rail_renders_prefetch_and_syncs_a_turn_as_an_episode() {
    let server = MockServer::start().await;
    mount_rail_backend(&server).await;
    let handle = handle(&server.uri());

    let prompt = handle.system_prompt_block();
    assert!(prompt.starts_with("# LakeBase Memory System\n"), "{prompt}");
    assert!(prompt.contains("lkb_memory_search"), "{prompt}");
    assert!(prompt.contains("lkb_memory_stats"), "{prompt}");
    assert!(prompt.contains("`convention`"), "{prompt}");
    assert!(prompt.contains("Search before storing"), "{prompt}");

    let rendered = handle.prefetch("rust").await;
    assert_eq!(
        rendered,
        "## Related Memories\n- [fact] likes rust (score: 0.800)"
    );
    assert_eq!(handle.prefetch("   ").await, "");

    handle.sync_turn("hi", "hello").await.expect("sync turn");
    handle
        .sync_turn("   ", "ignored")
        .await
        .expect("blank turn");
    let requests = server.received_requests().await.expect("requests");
    let ingests: Vec<&wiremock::Request> = requests
        .iter()
        .filter(|request| request.url.path().ends_with("/ingest"))
        .collect();
    assert_eq!(ingests.len(), 1, "one turn, no blank write");
    let body = request_json(ingests[0]);
    assert_eq!(body["memory_type"], json!("episode"));
    assert_eq!(body["content"], json!("User: hi\nAssistant: hello"));
    assert_eq!(body["metadata"], json!({ "source": "conversation" }));
    let importance = body["importance"].as_f64().expect("importance");
    assert!((importance - f64::from(0.4_f32)).abs() < f64::EPSILON);
}

#[tokio::test]
async fn the_rail_degrades_instead_of_raising_when_the_vendor_fails() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/memory/bases/mem_default/recall"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;

    let handle =
        LakeBaseMemoryProvider::new(&config(&server.uri()).with_option("max_retries", json!(0)));
    let payload = handle
        .handle_tool_call(SEARCH_TOOL, &json!({ "query": "rust" }))
        .await;
    assert!(payload["error"].as_str().is_some(), "{payload}");
    assert_eq!(handle.prefetch("rust").await, "");
}

#[test]
fn the_scope_and_base_id_fall_back_to_their_documented_defaults() {
    let config = ProviderConfig::new(ProviderKind::LakeBase);
    assert_eq!(spec::base_id(&config), "mem_default");
    assert_eq!(
        spec::base_id(&config.clone().with_option("base_id", json!(" "))),
        "mem_default"
    );
    assert_eq!(
        spec::base_id(&config.with_option("base_id", json!("mem_other"))),
        "mem_other"
    );
    let handle = LakeBaseMemoryProvider::new(&ProviderConfig::new(ProviderKind::LakeBase));
    assert_eq!(handle.default_scope(), MemoryScope::default());
}

#[test]
fn vendor_types_map_both_ways_and_ids_parse_at_the_boundary() {
    assert_eq!(
        spec::vendor_type_for_kind(MemoryType::EpisodicMemory),
        "episode"
    );
    assert_eq!(spec::vendor_type_for_kind(MemoryType::Coding), "convention");
    for kind in [
        MemoryType::UserProfile,
        MemoryType::SemanticMemory,
        MemoryType::Summary,
        MemoryType::Variable,
        MemoryType::Unknown,
    ] {
        assert_eq!(spec::vendor_type_for_kind(kind), "fact", "{kind:?}");
    }
    assert_eq!(
        spec::kind_for_vendor_type("EPISODE"),
        MemoryType::EpisodicMemory
    );
    assert_eq!(spec::kind_for_vendor_type("convention"), MemoryType::Coding);
    for vendor_type in ["fact", "procedural", "decision", "rejection", "unknown"] {
        assert_eq!(
            spec::kind_for_vendor_type(vendor_type),
            MemoryType::SemanticMemory,
            "{vendor_type}"
        );
    }
    assert_eq!(spec::parse_id(" 42 ").expect("numeric id"), 42);
    let error = spec::parse_id("abc").expect_err("non-numeric id");
    assert!(matches!(error, Error::Validation(_)), "{error:?}");
    assert!(error.to_string().contains('`'), "{error}");
}
