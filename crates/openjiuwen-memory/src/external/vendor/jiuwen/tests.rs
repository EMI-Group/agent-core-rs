use serde_json::json;
use tokio_util::sync::CancellationToken;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::*;
use crate::record::CallOptions;

/// The credential every request must carry.
const JWT: &str = "jwt";

/// The scope every test reads and writes under.
fn scope() -> MemoryScope {
    MemoryScope::new("alice", "ws")
}

/// A configured handle pointing at `server`.
fn handle(server: &MockServer) -> JiuwenMemoryProvider {
    JiuwenMemoryProvider::new(
        &ProviderConfig::new(ProviderKind::Jiuwen)
            .with_endpoint(server.uri())
            .with_api_key(JWT),
    )
}

/// Mounts one `POST route` answering `body` with a `200`.
async fn mount_post(server: &MockServer, route: &str, body: Value) {
    Mock::given(method("POST"))
        .and(path(route))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .mount(server)
        .await;
}

/// The response body of the last received request.
async fn last_body(server: &MockServer) -> Value {
    let requests = server.received_requests().await.expect("received requests");
    let last = requests.last().expect("a request was made");
    serde_json::from_slice(&last.body).expect("a JSON body")
}

/// The sorted object keys of `value`.
fn keys(value: &Value) -> Vec<&str> {
    let object = value.as_object().expect("a JSON object");
    let mut keys: Vec<&str> = object.keys().map(String::as_str).collect();
    keys.sort_unstable();
    keys
}

/// A buffered response carrying `body`.
fn response(body: &str) -> HttpResponse {
    HttpResponse {
        status: 200,
        headers: Default::default(),
        url: "http://jiuwen.test/v1/add".to_owned(),
        body: body.as_bytes().to_vec().into(),
    }
}

#[tokio::test]
async fn search_posts_the_documented_body_and_applies_the_threshold() {
    let server = MockServer::start().await;
    mount_post(
        &server,
        SEARCH_PATH,
        json!({
            "items": [
                { "unit_id": "u1", "content": "alpha", "score": 0.8 },
                { "unit_id": "u2", "content": "beta", "score": 0.2 },
            ]
        }),
    )
    .await;

    let options = SearchOptions::default().with_top_k(120).with_threshold(0.3);
    let hits = handle(&server)
        .search(&scope(), "theme", &options)
        .await
        .expect("search");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].record.id, "u1");
    assert_eq!(hits[0].record.content, "alpha");
    assert!((hits[0].score - 0.8).abs() < f32::EPSILON);

    let received = server.received_requests().await.expect("received requests");
    assert_eq!(received.len(), 1);
    assert_eq!(received[0].method, wiremock::http::Method::POST);
    assert_eq!(received[0].url.path(), SEARCH_PATH);
    assert_eq!(
        received[0]
            .headers
            .get("authorization")
            .expect("auth header"),
        "Bearer jwt"
    );

    let body: Value = serde_json::from_slice(&received[0].body).expect("a JSON body");
    assert_eq!(
        keys(&body),
        vec!["context", "disclosure", "query", "top_k"],
        "the server rejects unknown fields"
    );
    assert_eq!(body["query"], json!("theme"));
    assert_eq!(body["top_k"], json!(50), "clamped to the server maximum");
    assert_eq!(body["disclosure"], json!("l2"));

    assert_eq!(keys(&body["context"]), vec!["scope"]);
    let payload = &body["context"]["scope"];
    assert_eq!(
        keys(payload),
        vec!["agent", "org", "session", "space", "user"]
    );
    assert_eq!(payload["org"], json!("local"));
    assert_eq!(payload["space"], json!(""));
    assert_eq!(payload["user"], json!("developer"));
    assert_eq!(payload["agent"], json!("default"));
    assert_eq!(payload["session"], json!("alice"));
}

#[test]
fn the_scope_payload_reads_the_identity_options_and_the_scope_axes() {
    let config = ProviderConfig::new(ProviderKind::Jiuwen)
        .with_option("identity_org", json!("acme"))
        .with_option("identity_user", json!("svc"))
        .with_option("tenant_id", json!("tenant-7"));
    let scope = MemoryScope::new("alice", "ws")
        .session("session-1")
        .agent("agent-9");
    let payload = scope_payload(&SpecContext::new(&config, &scope));
    assert_eq!(
        payload,
        json!({
            "org": "acme",
            "space": "",
            "user": "svc",
            "agent": "agent-9",
            "session": "session-1",
        })
    );

    let default_agent = MemoryScope::new("alice", "ws").agent(DEFAULT_ID);
    let payload = scope_payload(&SpecContext::new(&config, &default_agent));
    assert_eq!(payload["agent"], json!("tenant-7"));
    assert_eq!(payload["session"], json!("alice"));
}

#[tokio::test]
async fn add_posts_the_documented_body_and_rebuilds_content_from_segments() {
    let server = MockServer::start().await;
    mount_post(
        &server,
        ADD_PATH,
        json!([{
            "id": "unit-1",
            "tier": "l1",
            "segments": [{ "content": "part a" }, { "content": "part b" }],
        }]),
    )
    .await;

    let outcome = handle(&server)
        .add(
            &scope(),
            vec![NewMemory::semantic("part a part b")],
            &AddOptions::default(),
        )
        .await
        .expect("add");
    assert_eq!(outcome.added.len(), 1);
    assert_eq!(outcome.added[0].id, "unit-1");
    assert_eq!(outcome.added[0].content, "part a\npart b");
    assert_eq!(outcome.added[0].metadata.get("tier"), Some(&json!("l1")));

    let received = server.received_requests().await.expect("received requests");
    assert_eq!(received[0].method, wiremock::http::Method::POST);
    assert_eq!(received[0].url.path(), ADD_PATH);
    let body: Value = serde_json::from_slice(&received[0].body).expect("a JSON body");
    assert_eq!(
        keys(&body),
        vec!["content", "scope", "system_metadata", "tags"]
    );
    assert_eq!(body["content"], json!("part a part b"));
    assert_eq!(body["tags"], json!(["conversation", "user"]));
    assert_eq!(body["system_metadata"]["infer"], json!("true"));
    assert_eq!(
        keys(&body["scope"]),
        vec!["agent", "org", "session", "space", "user"]
    );
}

#[tokio::test]
async fn metadata_tags_and_infer_false_ride_the_write_body() {
    let server = MockServer::start().await;
    mount_post(
        &server,
        ADD_PATH,
        json!([{ "id": "unit-2", "segments": [] }]),
    )
    .await;

    let memory = NewMemory::semantic("raw text")
        .with_metadata("infer", json!(false))
        .with_metadata("tags", json!(["note", 7]));
    handle(&server)
        .add(&scope(), vec![memory], &AddOptions::default())
        .await
        .expect("add");

    let body = last_body(&server).await;
    assert_eq!(body["system_metadata"], Value::Null);
    assert_eq!(body["tags"], json!(["note"]), "non-string tags are dropped");
}

#[tokio::test]
async fn an_empty_add_answer_is_a_skip_not_a_failure() {
    let server = MockServer::start().await;
    mount_post(&server, ADD_PATH, json!([])).await;

    let outcome = handle(&server)
        .add(
            &scope(),
            vec![NewMemory::semantic("deduplicated fact")],
            &AddOptions::default(),
        )
        .await
        .expect("add");
    assert!(outcome.added.is_empty());
    assert!(outcome.merged.is_empty());
    assert_eq!(outcome.skipped.len(), 1);
    assert!(!outcome.skipped[0].is_empty(), "a generated id is reported");
}

#[test]
fn a_non_array_add_answer_is_tolerated_as_a_skip() {
    let config = ProviderConfig::new(ProviderKind::Jiuwen).with_endpoint("https://jiuwen.invalid");
    let scope = scope();
    let cx = SpecContext::new(&config, &scope);
    let memory = NewMemory::semantic("fact");
    let outcome = SPEC
        .add_result(
            &cx,
            &response(r#"{"ok":true}"#),
            std::slice::from_ref(&memory),
        )
        .expect("a non-array body is a skip");
    assert!(outcome.added.is_empty());
    assert_eq!(outcome.skipped.len(), 1);

    let outcome = SPEC
        .add_result(&cx, &response("<html>nope</html>"), &[memory])
        .expect("a non-JSON body is a skip");
    assert!(outcome.added.is_empty());
    assert_eq!(outcome.skipped.len(), 1);
}

#[test]
fn search_decodes_tolerantly() {
    let config = ProviderConfig::new(ProviderKind::Jiuwen).with_endpoint("https://jiuwen.invalid");
    let scope = scope();
    let cx = SpecContext::new(&config, &scope);
    let options = SearchOptions::default().with_threshold(0.0);
    for body in [r"[]", r#"{"items": 7}"#, r"null", r#"{"ok":true}"#] {
        let hits = SPEC
            .search_result(&cx, &response(body), &options)
            .expect("a tolerant decode");
        assert!(hits.is_empty(), "{body}");
    }

    let hits = SPEC
        .search_result(
            &cx,
            &response(r#"{"items":[{"unit_id":"u1","content":"a","score":0.4}]}"#),
            &options,
        )
        .expect("a decode");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].record.id, "u1");
}

#[tokio::test]
async fn the_undocumented_record_verbs_fail_before_any_request() {
    let server = MockServer::start().await;
    let provider = handle(&server);
    let scope = scope();

    let error = provider.get(&scope, "unit-1").await.expect_err("get");
    assert_no_verb(&error, "get");
    let error = provider
        .update(
            &scope,
            "unit-1",
            MemoryUpdate::content("changed"),
            &CallOptions::default(),
        )
        .await
        .expect_err("update");
    assert_no_verb(&error, "update");
    let error = provider.delete(&scope, "unit-1").await.expect_err("delete");
    assert_no_verb(&error, "delete");
    let error = provider
        .list(&MemoryFilter::for_scope(&scope))
        .await
        .expect_err("list");
    assert_no_verb(&error, "list");

    assert!(
        server
            .received_requests()
            .await
            .expect("received requests")
            .is_empty(),
        "no endpoint is invented"
    );
}

/// Asserts `error` names `verb` and the undocumented `POST /v1/<verb>` route.
fn assert_no_verb(error: &Error, verb: &str) {
    assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
    let message = error.to_string();
    assert!(message.contains(verb), "{message}");
    assert!(message.contains("POST /v1/"), "{message}");
}

#[tokio::test]
async fn health_probes_healthz_and_reports_degradation_instead_of_failing() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path(HEALTH_PATH))
        .respond_with(ResponseTemplate::new(200))
        .mount(&server)
        .await;
    let health = handle(&server).health().await.expect("health");
    assert!(health.healthy, "{health:?}");
    let received = server.received_requests().await.expect("received requests");
    assert_eq!(received[0].url.path(), HEALTH_PATH);

    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path(HEALTH_PATH))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;
    let provider = JiuwenMemoryProvider::new(
        &ProviderConfig::new(ProviderKind::Jiuwen)
            .with_endpoint(server.uri())
            .with_option("max_retries", json!(0)),
    );
    let health = provider.health().await.expect("a degraded probe");
    assert!(!health.healthy, "{health:?}");
    assert!(
        health.detail.as_deref().unwrap_or_default().contains("500"),
        "{health:?}"
    );

    let provider = JiuwenMemoryProvider::new(
        &ProviderConfig::new(ProviderKind::Jiuwen)
            .with_endpoint("http://127.0.0.1:1")
            .with_option("max_retries", json!(0)),
    );
    let health = provider.health().await.expect("a degraded probe");
    assert!(!health.healthy, "{health:?}");
}

#[tokio::test]
async fn a_transient_search_status_is_retried() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(SEARCH_PATH))
        .respond_with(ResponseTemplate::new(503))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    mount_post(&server, SEARCH_PATH, json!({ "items": [] })).await;

    let provider = JiuwenMemoryProvider::new(
        &ProviderConfig::new(ProviderKind::Jiuwen)
            .with_endpoint(server.uri())
            .with_option("max_retries", json!(1))
            .with_option("retry_backoff_ms", json!(1)),
    );
    let hits = provider
        .search(&scope(), "theme", &SearchOptions::default())
        .await
        .expect("a retried search");
    assert!(hits.is_empty());
    assert_eq!(
        server
            .received_requests()
            .await
            .expect("received requests")
            .len(),
        2
    );
}

#[tokio::test]
async fn a_failing_primary_fails_over_to_the_secondary() {
    let primary = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(SEARCH_PATH))
        .respond_with(ResponseTemplate::new(503))
        .mount(&primary)
        .await;
    let secondary = MockServer::start().await;
    mount_post(
        &secondary,
        SEARCH_PATH,
        json!({ "items": [{ "unit_id": "u1", "content": "alpha", "score": 0.9 }] }),
    )
    .await;

    let provider = JiuwenMemoryProvider::new(
        &ProviderConfig::new(ProviderKind::Jiuwen)
            .with_endpoint(primary.uri())
            .with_option("failover_endpoints", json!([secondary.uri()]))
            .with_option("max_retries", json!(0))
            .with_option("retry_backoff_ms", json!(1)),
    );
    let hits = provider
        .search(&scope(), "theme", &SearchOptions::default())
        .await
        .expect("a failover search");
    assert_eq!(hits.len(), 1);
    assert_eq!(
        secondary
            .received_requests()
            .await
            .expect("received requests")
            .len(),
        1
    );
}

#[tokio::test]
async fn an_unconfigured_handle_names_the_missing_dependency() {
    let provider = JiuwenMemoryProvider::new(&ProviderConfig::new(ProviderKind::Jiuwen));
    let error = provider
        .search(&scope(), "theme", &SearchOptions::default())
        .await
        .expect_err("unconfigured");
    assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
    assert!(error.to_string().contains("jiuwen_memory"), "{error}");

    let server = MockServer::start().await;
    assert_eq!(
        handle(&server).capabilities(),
        ProviderCapabilities {
            semantic_search: true,
            message_history: true,
            ..ProviderCapabilities::none()
        }
    );
}

#[tokio::test]
async fn a_cancelled_search_sends_nothing() {
    let server = MockServer::start().await;
    mount_post(&server, SEARCH_PATH, json!({ "items": [] })).await;

    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let options = SearchOptions::default().with_cancellation(cancellation);
    let error = handle(&server)
        .search(&scope(), "theme", &options)
        .await
        .expect_err("cancelled");
    assert!(matches!(error, Error::Cancelled), "{error:?}");
    assert!(server
        .received_requests()
        .await
        .expect("received requests")
        .is_empty());
}

#[tokio::test]
async fn the_rail_advertises_and_runs_the_two_mem2_tools() {
    let server = MockServer::start().await;
    mount_post(
        &server,
        SEARCH_PATH,
        json!({ "items": [{ "unit_id": "u1", "content": "alpha", "score": 0.8 }] }),
    )
    .await;
    mount_post(
        &server,
        ADD_PATH,
        json!([{ "id": "unit-1", "tier": "l1", "segments": [{ "content": "alpha" }] }]),
    )
    .await;

    let provider = handle(&server);
    let schemas = provider.get_tool_schemas();
    let names: Vec<&str> = schemas
        .iter()
        .filter_map(|schema| schema["name"].as_str())
        .collect();
    assert_eq!(names, vec![MEM2_SEARCH_TOOL, MEM2_ADD_TOOL]);

    let found = provider
        .handle_tool_call(MEM2_SEARCH_TOOL, &json!({ "query": "theme" }))
        .await;
    assert_eq!(found["count"], json!(1));
    assert_eq!(found["results"][0]["item_id"], json!("u1"));
    assert_eq!(found["results"][0]["content"], json!("alpha"));

    let stored = provider
        .handle_tool_call(MEM2_ADD_TOOL, &json!({ "content": "a fact" }))
        .await;
    assert_eq!(stored["result"], json!("stored"));
    assert_eq!(stored["item_id"], json!("unit-1"));
    assert_eq!(stored["tier"], json!("l1"));

    let unknown = provider.handle_tool_call("nope", &json!({})).await;
    assert_eq!(unknown["error"], json!("unknown tool: nope"));

    let prefetch = provider.prefetch("theme").await;
    assert_eq!(prefetch, "## Jiuwen Memory\n- alpha");

    assert!(provider.system_prompt_block().contains("# Jiuwen Memory"));
}
