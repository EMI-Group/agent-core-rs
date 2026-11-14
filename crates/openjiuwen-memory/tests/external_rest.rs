//! REST transports of the external memory providers.
//!
//! Drives the four HTTP-speaking vendor adapters (`mem0`, `jiuwen_memory`, `lakebase`,
//! `openviking`) and the SDK-only `agentarts` handle through `wiremock`, covering the
//! transport contract documented in `src/external/CONTEXT.md`: vendor authentication,
//! request paths and bodies, missing-record mapping, retries of idempotent verbs,
//! failover to a second endpoint, the tool rail, cancellation, and status mapping.
//!
//! Everything runs offline against a loopback `MockServer`: no network, no credential, no
//! wall-clock ordering, and every provider call is bounded so a hang fails the test
//! instead of stalling the suite.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::future::Future;
use std::time::Duration;

use openjiuwen_common::error::Error;
use openjiuwen_memory::external::{
    JiuwenMemoryProvider, LakeBaseMemoryProvider, Mem0MemoryProvider, MemoryProvider,
    OpenVikingMemoryProvider, ProviderConfig, ProviderKind, ProviderRegistry,
};
use openjiuwen_memory::record::{
    AddOptions, CallOptions, MemoryScope, MemoryUpdate, NewMemory, SearchOptions,
};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

/// Deadline every provider call is given; a hang fails the test rather than stalling it.
const BOUND: Duration = Duration::from_secs(10);
/// The mem0 credential every configured handle carries.
const MEM0_KEY: &str = "m0-test-key";
/// The Jiuwen bearer token every configured handle carries.
const JIUWEN_JWT: &str = "jwt-test";
/// The `LakeBase` bearer token every configured handle carries.
const LAKEBASE_KEY: &str = "lkb-test-key";
/// The `OpenViking` API key every configured handle carries.
const OPENVIKING_KEY: &str = "vk-test-key";

/// Runs `future` under [`BOUND`].
async fn within<F: Future>(future: F) -> F::Output {
    tokio::time::timeout(BOUND, future)
        .await
        .expect("the provider call finished within the deadline")
}

/// The scope every test reads and writes under.
fn scope() -> MemoryScope {
    MemoryScope::new("alice", "workspace")
}

/// The requests `server` received, in arrival order.
async fn received(server: &MockServer) -> Vec<Request> {
    server
        .received_requests()
        .await
        .expect("the mock server recorded its requests")
}

/// Reads one request body as JSON.
fn body_json(request: &Request) -> Value {
    serde_json::from_slice(&request.body).expect("the request body is JSON")
}

/// The value of the header `name` on `request`, when it carries one.
fn header_value<'a>(request: &'a Request, name: &str) -> Option<&'a str> {
    request
        .headers
        .get(name)
        .and_then(|value| value.to_str().ok())
}

/// A mem0 configuration pointing at `endpoint` with a fast retry backoff.
fn mem0_config(endpoint: &str) -> ProviderConfig {
    ProviderConfig::new(ProviderKind::Mem0)
        .with_endpoint(endpoint)
        .with_api_key(MEM0_KEY)
        .with_option("user_id", json!("alice"))
        .with_option("retry_backoff_ms", json!(1))
}

/// Mounts a mock for `verb path` answering `status` with `body` (when given).
async fn mount(server: &MockServer, verb: &str, route: &str, status: u16, body: Option<Value>) {
    let builder = match verb {
        "GET" => Mock::given(method("GET")),
        "POST" => Mock::given(method("POST")),
        "PUT" => Mock::given(method("PUT")),
        "DELETE" => Mock::given(method("DELETE")),
        other => panic!("unsupported verb {other}"),
    };
    let response = ResponseTemplate::new(status);
    let response = match body {
        Some(body) => response.set_body_json(body),
        None => response,
    };
    builder
        .and(path(route))
        .respond_with(response)
        .mount(server)
        .await;
}

/// Asserts `error` is a typed `Unsupported` naming `needles`.
fn assert_unsupported(error: &Error, needles: &[&str]) {
    assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
    let message = error.to_string();
    for needle in needles {
        assert!(message.contains(needle), "`{message}` must name `{needle}`");
    }
}

/// mem0: token auth on the writes, the search path, and the `404` mapping of reads and
/// deletes, all through the registry's vendor factory.
#[tokio::test]
async fn mem0_configured_over_rest_posts_token_auth_and_maps_missing_records() {
    let server = MockServer::start().await;
    mount(
        &server,
        "POST",
        "/v1/memories/",
        200,
        Some(json!({ "results": [{ "id": "m-1", "memory": "alice prefers dark mode" }] })),
    )
    .await;
    mount(
        &server,
        "POST",
        "/v1/memories/search/",
        200,
        Some(json!({
            "results": [{ "id": "m-1", "memory": "alice prefers dark mode", "score": 0.87 }],
        })),
    )
    .await;
    mount(
        &server,
        "DELETE",
        "/v1/memories/m-1/",
        200,
        Some(json!({ "status": "deleted" })),
    )
    .await;
    mount(&server, "DELETE", "/v1/memories/gone/", 404, None).await;
    mount(&server, "GET", "/v1/memories/gone/", 404, None).await;

    let registry = ProviderRegistry::with_vendor_stubs();
    let provider = registry
        .build(&mem0_config(&server.uri()))
        .expect("the mem0 vendor factory builds a configured handle");
    assert_eq!(provider.name(), "mem0");

    let outcome = within(provider.add(
        &scope(),
        vec![NewMemory::semantic("alice prefers dark mode")],
        &AddOptions::default(),
    ))
    .await
    .expect("add");
    assert_eq!(outcome.added.len(), 1);
    assert_eq!(outcome.added[0].id, "m-1");

    let hits = within(provider.search(
        &scope(),
        "dark mode",
        &SearchOptions::default().with_top_k(3),
    ))
    .await
    .expect("search");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].record.content, "alice prefers dark mode");
    assert!((hits[0].score - 0.87).abs() < 1e-6);

    assert!(within(provider.delete(&scope(), "m-1"))
        .await
        .expect("delete"));
    assert!(!within(provider.delete(&scope(), "gone"))
        .await
        .expect("a 404 delete is `false`"));
    assert_eq!(
        within(provider.get(&scope(), "gone")).await.expect("get"),
        None,
        "a 404 read is `None`"
    );

    let requests = received(&server).await;
    assert_eq!(requests.len(), 5);
    assert_eq!(requests[0].method.as_str(), "POST");
    assert_eq!(requests[0].url.path(), "/v1/memories/");
    assert_eq!(
        header_value(&requests[0], "authorization"),
        Some("Token m0-test-key")
    );
    let add = body_json(&requests[0]);
    assert_eq!(
        add["messages"][0]["content"],
        json!("alice prefers dark mode")
    );
    assert_eq!(add["user_id"], json!("alice"));
    assert_eq!(add["infer"], json!(false));

    assert_eq!(requests[1].url.path(), "/v1/memories/search/");
    assert_eq!(
        header_value(&requests[1], "authorization"),
        Some("Token m0-test-key")
    );
    assert_eq!(body_json(&requests[1])["query"], json!("dark mode"));

    assert_eq!(requests[2].method.as_str(), "DELETE");
    assert_eq!(requests[2].url.path(), "/v1/memories/m-1/");
    assert_eq!(requests[3].url.path(), "/v1/memories/gone/");
    assert_eq!(requests[4].url.path(), "/v1/memories/gone/");
}

/// `jiuwen_memory`: bearer auth on `/v1/add` and `/v1/search`, and a typed `Unsupported`
/// for the point read the documented Mem2 API does not have.
#[tokio::test]
async fn jiuwen_configured_over_rest_posts_bearer_auth_and_reports_missing_verbs() {
    let server = MockServer::start().await;
    mount(
        &server,
        "POST",
        "/v1/add",
        200,
        Some(json!([
            { "id": "u-1", "segments": [{ "content": "theme preference" }] },
        ])),
    )
    .await;
    mount(
        &server,
        "POST",
        "/v1/search",
        200,
        Some(json!({
            "items": [{ "unit_id": "u-1", "content": "theme preference", "score": 0.8 }],
        })),
    )
    .await;

    let provider = JiuwenMemoryProvider::new(
        &ProviderConfig::new(ProviderKind::Jiuwen)
            .with_endpoint(server.uri())
            .with_api_key(JIUWEN_JWT),
    );

    let outcome = within(provider.add(
        &scope(),
        vec![NewMemory::semantic("theme preference")],
        &AddOptions::default(),
    ))
    .await
    .expect("add");
    assert_eq!(outcome.added.len(), 1);
    assert_eq!(outcome.added[0].id, "u-1");
    assert_eq!(outcome.added[0].content, "theme preference");

    let hits = within(provider.search(&scope(), "theme", &SearchOptions::default()))
        .await
        .expect("search");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].record.content, "theme preference");

    let requests = received(&server).await;
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].url.path(), "/v1/add");
    assert_eq!(
        header_value(&requests[0], "authorization"),
        Some("Bearer jwt-test")
    );
    assert_eq!(
        body_json(&requests[0])["content"],
        json!("theme preference")
    );
    assert_eq!(requests[1].url.path(), "/v1/search");
    assert_eq!(
        header_value(&requests[1], "authorization"),
        Some("Bearer jwt-test")
    );
    assert_eq!(body_json(&requests[1])["query"], json!("theme"));

    let error = within(provider.get(&scope(), "u-1"))
        .await
        .expect_err("the Mem2 HTTP API has no point read");
    assert_unsupported(&error, &["has no get operation", "POST /v1/get", "Mem2"]);
    assert_eq!(
        received(&server).await.len(),
        2,
        "a missing verb never reaches the wire"
    );
}

/// `lakebase`: ingest and recall under the configured base id, and the documented
/// `Unsupported` update.
#[tokio::test]
async fn lakebase_configured_over_rest_ingests_recalls_and_rejects_update() {
    let server = MockServer::start().await;
    mount(
        &server,
        "POST",
        "/memory/bases/base-7/ingest",
        200,
        Some(json!({ "memory_id": 7, "memory_type": "fact" })),
    )
    .await;
    mount(
        &server,
        "POST",
        "/memory/bases/base-7/recall",
        200,
        Some(json!({
            "memories": [{ "content": "alpha", "memory_type": "fact", "score": 0.8 }],
        })),
    )
    .await;

    let provider = LakeBaseMemoryProvider::new(
        &ProviderConfig::new(ProviderKind::LakeBase)
            .with_endpoint(server.uri())
            .with_api_key(LAKEBASE_KEY)
            .with_option("base_id", json!("base-7")),
    );

    let outcome = within(provider.add(
        &scope(),
        vec![NewMemory::semantic("alpha")],
        &AddOptions::default(),
    ))
    .await
    .expect("ingest");
    assert_eq!(outcome.added.len(), 1);
    assert_eq!(outcome.added[0].id, "7", "the integer id is stringified");
    assert_eq!(outcome.added[0].content, "alpha");

    let hits = within(provider.search(
        &scope(),
        "alpha",
        &SearchOptions::default().with_threshold(0.3),
    ))
    .await
    .expect("recall");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].record.content, "alpha");
    assert_eq!(hits[0].record.metadata_str("memory_type"), Some("fact"));

    let requests = received(&server).await;
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].url.path(), "/memory/bases/base-7/ingest");
    assert_eq!(
        header_value(&requests[0], "authorization"),
        Some("Bearer lkb-test-key")
    );
    let ingest = body_json(&requests[0]);
    assert_eq!(ingest["content"], json!("alpha"));
    assert_eq!(ingest["role"], json!("user"));
    assert_eq!(requests[1].url.path(), "/memory/bases/base-7/recall");
    assert_eq!(body_json(&requests[1])["query"], json!("alpha"));

    let error = within(provider.update(
        &scope(),
        "7",
        MemoryUpdate::content("beta"),
        &CallOptions::new(),
    ))
    .await
    .expect_err("LakeBase documents no update endpoint");
    assert_unsupported(
        &error,
        &[
            "no update operation",
            "PUT /memory/bases/{base_id}/memories/{id}",
        ],
    );
}

/// `openviking`: the `X-API-Key` (plus routing) headers on the find endpoint, and the
/// `Unsupported` delete/update the vendor API does not document.
#[tokio::test]
async fn openviking_configured_over_rest_sends_the_api_key_and_rejects_writes() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/search/find"))
        .and(header("x-api-key", OPENVIKING_KEY))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "result": {
                "memories": [{ "uri": "viking://m/1", "score": 0.91, "abstract": "alpha" }],
                "resources": [],
                "skills": [],
                "total": 1,
            },
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = OpenVikingMemoryProvider::new(
        &ProviderConfig::new(ProviderKind::OpenViking)
            .with_endpoint(server.uri())
            .with_api_key(OPENVIKING_KEY)
            .with_option("session_id", json!("sess-1")),
    );

    let hits = within(provider.search(&scope(), "alpha", &SearchOptions::default()))
        .await
        .expect("find");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].record.id, "viking://m/1");
    assert_eq!(hits[0].record.content, "alpha");

    let requests = received(&server).await;
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].url.path(), "/api/v1/search/find");
    assert_eq!(
        header_value(&requests[0], "x-api-key"),
        Some(OPENVIKING_KEY)
    );
    assert_eq!(
        header_value(&requests[0], "x-openviking-account"),
        Some("default")
    );
    assert_eq!(
        header_value(&requests[0], "x-openviking-agent"),
        Some("hermes")
    );
    assert_eq!(body_json(&requests[0])["query"], json!("alpha"));

    let delete = within(provider.delete(&scope(), "viking://m/1"))
        .await
        .expect_err("OpenViking documents no memory delete");
    assert_unsupported(
        &delete,
        &["no delete operation", "no memory-delete endpoint"],
    );

    let update = within(provider.update(
        &scope(),
        "viking://m/1",
        MemoryUpdate::content("beta"),
        &CallOptions::new(),
    ))
    .await
    .expect_err("OpenViking documents no memory update");
    assert_unsupported(
        &update,
        &["no update operation", "no memory-update endpoint"],
    );
}

/// `agentarts` stays SDK-only: a fully configured handle is a typed `Unsupported` naming
/// the vendor SDK, because the adapter has no HTTP path.
#[tokio::test]
async fn agentarts_stays_sdk_only_even_when_fully_configured() {
    let registry = ProviderRegistry::with_vendor_stubs();
    let provider = registry
        .build(
            &ProviderConfig::new(ProviderKind::AgentArts)
                .with_endpoint("http://127.0.0.1:9")
                .with_api_key("agentarts-key")
                .with_option("region", json!("cn-north-4")),
        )
        .expect("the agentarts vendor factory builds a handle");
    assert_eq!(provider.name(), "agentarts");

    for error in [
        within(provider.initialize()).await.expect_err("initialize"),
        within(provider.add(
            &scope(),
            vec![NewMemory::semantic("a fact")],
            &AddOptions::default(),
        ))
        .await
        .expect_err("add"),
        within(provider.search(&scope(), "fact", &SearchOptions::default()))
            .await
            .expect_err("search"),
    ] {
        assert_unsupported(&error, &["agentarts-sdk", "MemoryClient"]);
    }
}

/// An idempotent verb retries a transient status exactly once before succeeding.
#[tokio::test]
async fn a_transient_status_is_retried_once_on_an_idempotent_verb() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/memories/m-1/"))
        .respond_with(ResponseTemplate::new(503))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/memories/m-1/"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({ "id": "m-1", "memory": "stored" })),
        )
        .mount(&server)
        .await;

    let provider = Mem0MemoryProvider::new(&mem0_config(&server.uri()));
    let record = within(provider.get(&scope(), "m-1"))
        .await
        .expect("the retried read succeeds")
        .expect("a record");
    assert_eq!(record.content, "stored");

    let requests = received(&server).await;
    assert_eq!(requests.len(), 2, "the 503 is retried exactly once");
    assert_eq!(requests[0].url.path(), "/v1/memories/m-1/");
    assert_eq!(requests[1].url.path(), "/v1/memories/m-1/");
}

/// A dead primary fails over to the configured second endpoint; when every endpoint fails
/// the error names the last URL and the endpoint count.
#[tokio::test]
async fn a_dead_endpoint_fails_over_and_an_exhausted_list_names_the_last_one() {
    let primary = MockServer::start().await;
    mount(&primary, "POST", "/v1/memories/search/", 500, None).await;
    let secondary = MockServer::start().await;
    mount(
        &secondary,
        "POST",
        "/v1/memories/search/",
        200,
        Some(json!({
            "results": [{ "id": "m-1", "memory": "theme preference", "score": 0.9 }],
        })),
    )
    .await;

    let provider = Mem0MemoryProvider::new(
        &mem0_config(&primary.uri())
            .with_option("failover_endpoints", json!([secondary.uri()]))
            .with_option("max_retries", json!(0)),
    );
    let hits = within(provider.search(&scope(), "theme", &SearchOptions::default()))
        .await
        .expect("the second endpoint answers");
    assert_eq!(hits.len(), 1);
    assert_eq!(
        received(&primary).await.len(),
        1,
        "the primary is tried once"
    );
    assert_eq!(received(&secondary).await.len(), 1);

    let dead_primary = MockServer::start().await;
    mount(&dead_primary, "POST", "/v1/memories/search/", 500, None).await;
    let dead_secondary = MockServer::start().await;
    mount(&dead_secondary, "POST", "/v1/memories/search/", 500, None).await;
    let exhausted = Mem0MemoryProvider::new(
        &mem0_config(&dead_primary.uri())
            .with_option("failover_endpoints", json!([dead_secondary.uri()]))
            .with_option("max_retries", json!(0)),
    );
    let error = within(exhausted.search(&scope(), "theme", &SearchOptions::default()))
        .await
        .expect_err("every endpoint answered 500");
    assert!(matches!(error, Error::Transport(_)), "{error:?}");
    let message = error.to_string();
    assert!(
        message.contains(&format!("{}/v1/memories/search/", dead_secondary.uri())),
        "the last URL must be named: {message}"
    );
    assert!(message.contains("2 endpoint(s)"), "{message}");
    assert_eq!(received(&dead_primary).await.len(), 1);
    assert_eq!(received(&dead_secondary).await.len(), 1);
}

/// The configured mem0 rail: schema shape, a `search_memory`-shaped call that posts to the
/// vendor, and the `unknown tool` payload instead of a raised error.
#[tokio::test]
async fn a_configured_mem0_serves_the_tool_rail_round_trip() {
    let server = MockServer::start().await;
    mount(
        &server,
        "POST",
        "/v1/memories/search/",
        200,
        Some(json!({
            "results": [{ "id": "m-1", "memory": "theme preference", "score": 0.9 }],
        })),
    )
    .await;

    let provider = Mem0MemoryProvider::new(&mem0_config(&server.uri()));
    let schemas = provider.get_tool_schemas();
    assert_eq!(
        schemas.len(),
        3,
        "the mem0 rail advertises profile, search, and conclude"
    );
    for schema in &schemas {
        assert!(
            schema
                .get("name")
                .and_then(Value::as_str)
                .is_some_and(|name| !name.is_empty()),
            "{schema}"
        );
        assert!(
            schema.get("parameters").is_some_and(Value::is_object),
            "{schema}"
        );
    }
    assert!(schemas
        .iter()
        .any(|schema| schema["name"] == json!("mem0_search")));

    let payload =
        within(provider.handle_tool_call("mem0_search", &json!({ "query": "theme" }))).await;
    assert!(
        payload.is_object(),
        "the rail answers a JSON object: {payload}"
    );
    assert_eq!(payload["count"], json!(1));
    assert_eq!(payload["results"][0]["memory"], json!("theme preference"));
    let requests = received(&server).await;
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].url.path(), "/v1/memories/search/");
    assert_eq!(body_json(&requests[0])["user_id"], json!("alice"));

    let unknown = within(provider.handle_tool_call("mem0_teleport", &json!({}))).await;
    assert_eq!(unknown, json!({ "error": "unknown tool: mem0_teleport" }));
}

/// A pre-cancelled call short-circuits before the transport and sends nothing.
#[tokio::test]
async fn a_pre_cancelled_call_returns_cancelled_and_sends_nothing() {
    let server = MockServer::start().await;
    mount(
        &server,
        "POST",
        "/v1/memories/",
        200,
        Some(json!({ "results": [{ "id": "m-1", "memory": "never stored" }] })),
    )
    .await;

    let provider = Mem0MemoryProvider::new(&mem0_config(&server.uri()));
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let options = AddOptions::default().with_cancellation(cancellation);
    let error = within(provider.add(
        &scope(),
        vec![NewMemory::semantic("never stored")],
        &options,
    ))
    .await
    .expect_err("a cancelled call does not write");
    assert!(matches!(error, Error::Cancelled), "{error:?}");
    assert!(received(&server).await.is_empty());
}

/// `401` is a security error and `422` a validation error carrying the vendor detail.
#[tokio::test]
async fn status_answers_map_to_security_and_validation_errors() {
    let server = MockServer::start().await;
    mount(&server, "POST", "/v1/memories/search/", 401, None).await;
    mount(
        &server,
        "GET",
        "/v1/memories/bad/",
        422,
        Some(json!({ "detail": "memory id must be a uuid" })),
    )
    .await;

    let provider = Mem0MemoryProvider::new(&mem0_config(&server.uri()));
    let denied = within(provider.search(&scope(), "theme", &SearchOptions::default()))
        .await
        .expect_err("401");
    assert!(matches!(denied, Error::Security(_)), "{denied:?}");
    let message = denied.to_string();
    assert!(message.contains("rejected the credentials"), "{message}");
    assert!(message.contains("HTTP 401"), "{message}");

    let invalid = within(provider.get(&scope(), "bad"))
        .await
        .expect_err("422");
    assert!(matches!(invalid, Error::Validation(_)), "{invalid:?}");
    assert!(
        invalid.to_string().contains("memory id must be a uuid"),
        "the vendor detail must survive: {invalid}"
    );
    assert_eq!(
        received(&server).await.len(),
        2,
        "neither status is retried"
    );
}
