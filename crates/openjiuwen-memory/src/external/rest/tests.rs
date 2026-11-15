use serde_json::json;
use tokio_util::sync::CancellationToken;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::*;
use crate::external::registry::ProviderKind;
use crate::record::MemoryType;

/// The vendor shape every test speaks: `/items` CRUD, `/search`, `/health`.
struct TestSpec;

impl WireSpec for TestSpec {
    fn id(&self) -> &'static str {
        "test"
    }

    fn display(&self) -> &'static str {
        "Test"
    }

    fn missing(&self) -> &'static str {
        "the `testspec` transport"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            semantic_search: true,
            keyword_search: true,
            ..ProviderCapabilities::none()
        }
    }

    fn default_auth(&self) -> AuthScheme {
        AuthScheme::Bearer
    }

    fn add_call(
        &self,
        _cx: &SpecContext<'_>,
        memories: &[NewMemory],
        _options: &AddOptions,
    ) -> Result<RestCall> {
        let memory = memories
            .first()
            .ok_or_else(|| Error::validation("the test spec needs one memory"))?;
        RestCall::post("/items").json(&json!({"memory": memory.content}))
    }

    fn add_result(
        &self,
        cx: &SpecContext<'_>,
        response: &HttpResponse,
        _memories: &[NewMemory],
    ) -> Result<AddOutcome> {
        let value = body_value(response)?;
        Ok(AddOutcome {
            added: vec![vendor_record(&value, cx.scope, MemoryType::SemanticMemory)],
            ..AddOutcome::default()
        })
    }

    fn search_call(
        &self,
        _cx: &SpecContext<'_>,
        query: &str,
        options: &SearchOptions,
    ) -> Result<RestCall> {
        RestCall::post("/search").json(&json!({"query": query, "top_k": options.top_k}))
    }

    fn search_result(
        &self,
        cx: &SpecContext<'_>,
        response: &HttpResponse,
        _options: &SearchOptions,
    ) -> Result<Vec<RecallHit>> {
        let value = body_value(response)?;
        Ok(results_array(&value)
            .iter()
            .map(|entry| {
                let record = vendor_record(entry, cx.scope, MemoryType::SemanticMemory);
                vendor_hit(record, field_f32(entry, &["score"]).unwrap_or(0.0))
            })
            .collect())
    }

    fn get_call(&self, _cx: &SpecContext<'_>, id: &str) -> Result<RestCall> {
        Ok(RestCall::get(format!("/items/{id}")))
    }

    fn get_result(
        &self,
        cx: &SpecContext<'_>,
        response: &HttpResponse,
        _id: &str,
    ) -> Result<Option<MemoryRecord>> {
        let value = body_value(response)?;
        Ok(Some(vendor_record(
            &value,
            cx.scope,
            MemoryType::SemanticMemory,
        )))
    }

    fn update_call(
        &self,
        _cx: &SpecContext<'_>,
        id: &str,
        update: &MemoryUpdate,
    ) -> Result<RestCall> {
        RestCall::put(format!("/items/{id}")).json(&json!({"memory": update.content}))
    }

    fn update_result(
        &self,
        cx: &SpecContext<'_>,
        response: &HttpResponse,
        _id: &str,
    ) -> Result<MemoryRecord> {
        let value = body_value(response)?;
        Ok(vendor_record(&value, cx.scope, MemoryType::SemanticMemory))
    }

    fn delete_call(&self, _cx: &SpecContext<'_>, id: &str) -> Result<RestCall> {
        Ok(RestCall::delete(format!("/items/{id}")))
    }

    fn delete_result(
        &self,
        _cx: &SpecContext<'_>,
        _response: &HttpResponse,
        _id: &str,
    ) -> Result<bool> {
        Ok(true)
    }

    fn list_call(&self, _cx: &SpecContext<'_>, _filter: &MemoryFilter) -> Result<RestCall> {
        Ok(RestCall::get("/items"))
    }

    fn list_result(
        &self,
        cx: &SpecContext<'_>,
        response: &HttpResponse,
        _filter: &MemoryFilter,
    ) -> Result<Vec<MemoryRecord>> {
        let value = body_value(response)?;
        Ok(results_array(&value)
            .iter()
            .map(|entry| vendor_record(entry, cx.scope, MemoryType::SemanticMemory))
            .collect())
    }

    fn health_call(&self, _cx: &SpecContext<'_>) -> Result<RestCall> {
        Ok(RestCall::get("/health"))
    }
}

/// A spec with only a listing path and no health probe.
struct ListOnlySpec;

impl WireSpec for ListOnlySpec {
    fn id(&self) -> &'static str {
        "listonly"
    }

    fn display(&self) -> &'static str {
        "ListOnly"
    }

    fn missing(&self) -> &'static str {
        "the `listonly` transport"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::none()
    }

    fn default_auth(&self) -> AuthScheme {
        AuthScheme::None
    }

    fn list_call(&self, cx: &SpecContext<'_>, filter: &MemoryFilter) -> Result<RestCall> {
        SPEC.list_call(cx, filter)
    }

    fn list_result(
        &self,
        cx: &SpecContext<'_>,
        response: &HttpResponse,
        filter: &MemoryFilter,
    ) -> Result<Vec<MemoryRecord>> {
        SPEC.list_result(cx, response, filter)
    }
}

/// The one static spec every test shares.
static SPEC: TestSpec = TestSpec;

/// The listing-only spec.
static LIST_ONLY: ListOnlySpec = ListOnlySpec;

/// The static spec as the provider takes it.
fn spec() -> &'static dyn WireSpec {
    &SPEC
}

/// A provider over `server`, with a fast retry backoff.
fn provider(server: &MockServer) -> RestProvider {
    RestProvider::from_config(
        spec(),
        &ProviderConfig::new(ProviderKind::Mem0)
            .with_endpoint(format!("{}/", server.uri()))
            .with_api_key("test-key")
            .with_option("retry_backoff_ms", json!(1)),
    )
}

/// An unconfigured `mem0`-shaped handle.
fn unconfigured() -> RestProvider {
    RestProvider::from_config(spec(), &ProviderConfig::new(ProviderKind::Mem0))
}

/// The scope every test writes under.
fn scope() -> MemoryScope {
    MemoryScope::new("alice", "ws")
}

/// Mounts one mock that answers every request with `status` and `body`.
async fn mount(server: &MockServer, verb: &str, route: &str, response: ResponseTemplate) {
    let matcher = match verb {
        "GET" => Mock::given(method("GET")),
        "POST" => Mock::given(method("POST")),
        "PUT" => Mock::given(method("PUT")),
        "DELETE" => Mock::given(method("DELETE")),
        other => panic!("unsupported verb {other}"),
    };
    matcher
        .and(path(route))
        .respond_with(response)
        .mount(server)
        .await;
}

/// Mounts a `POST /search` mock answering `body`.
async fn mount_search(server: &MockServer, body: serde_json::Value) {
    mount(
        server,
        "POST",
        "/search",
        ResponseTemplate::new(200).set_body_json(body),
    )
    .await;
}

#[test]
fn absolute_url_joins_with_exactly_one_slash() {
    assert_eq!(
        RestCall::absolute_url("http://host/api/", "/items"),
        "http://host/api/items"
    );
    assert_eq!(
        RestCall::absolute_url("http://host/api", "items"),
        "http://host/api/items"
    );
    assert_eq!(
        RestCall::absolute_url("http://host", "items"),
        "http://host/items"
    );
    assert_eq!(RestCall::absolute_url("http://host/", ""), "http://host");
    assert_eq!(
        RestCall::absolute_url("", "/items"),
        "/items",
        "a blank base keeps the path"
    );
    assert_eq!(
        RestCall::absolute_url("http://host", "https://other/items"),
        "https://other/items"
    );
}

#[test]
fn resolve_auth_reads_every_documented_option() {
    let config = ProviderConfig::new(ProviderKind::Mem0);
    assert_eq!(
        resolve_auth(AuthScheme::Bearer, &config),
        AuthScheme::Bearer
    );

    for (raw, expected) in [
        ("none", AuthScheme::None),
        (" None ", AuthScheme::None),
        ("BEARER", AuthScheme::Bearer),
        ("token", AuthScheme::Token),
        (
            "header",
            AuthScheme::Header {
                name: "Authorization".to_owned(),
            },
        ),
        (
            "query",
            AuthScheme::Query {
                name: "api_key".to_owned(),
            },
        ),
        ("nonsense", AuthScheme::Bearer),
        ("", AuthScheme::Bearer),
    ] {
        let config = config.clone().with_option("auth", json!(raw));
        assert_eq!(resolve_auth(AuthScheme::Bearer, &config), expected, "{raw}");
    }

    let named = config
        .clone()
        .with_option("auth", json!("header"))
        .with_option("auth_header", json!("X-Api-Key"));
    assert_eq!(
        resolve_auth(AuthScheme::None, &named),
        AuthScheme::Header {
            name: "X-Api-Key".to_owned()
        }
    );
    let queried = config
        .with_option("auth", json!("query"))
        .with_option("auth_query_param", json!("token"));
    assert_eq!(
        resolve_auth(AuthScheme::None, &queried),
        AuthScheme::Query {
            name: "token".to_owned()
        }
    );
}

#[test]
fn auth_schemes_inject_the_credential_where_documented() {
    assert_eq!(
        AuthScheme::Bearer.header_for("key"),
        Some(("Authorization".to_owned(), "Bearer key".to_owned()))
    );
    assert_eq!(
        AuthScheme::Token.header_for("key"),
        Some(("Authorization".to_owned(), "Token key".to_owned()))
    );
    assert_eq!(
        AuthScheme::Header {
            name: "X-Api-Key".to_owned()
        }
        .header_for("key"),
        Some(("X-Api-Key".to_owned(), "key".to_owned()))
    );
    assert_eq!(
        AuthScheme::Query {
            name: "api_key".to_owned()
        }
        .query_for("key"),
        Some(("api_key".to_owned(), "key".to_owned()))
    );
    assert_eq!(AuthScheme::None.header_for("key"), None);
    assert_eq!(AuthScheme::Bearer.header_for(""), None);
    assert_eq!(
        AuthScheme::Query { name: "q".into() }.header_for("key"),
        None
    );
}

#[test]
fn op_names_and_idempotence() {
    assert_eq!(Op::Add.as_str(), "add");
    assert_eq!(Op::Health.as_str(), "health");
    assert!(!Op::Add.is_idempotent());
    for op in [
        Op::Search,
        Op::Get,
        Op::Update,
        Op::Delete,
        Op::List,
        Op::Clear,
        Op::Health,
    ] {
        assert!(op.is_idempotent(), "{}", op.as_str());
    }
}

#[tokio::test]
async fn an_unconfigured_handle_reports_the_missing_dependency_everywhere() {
    let provider = unconfigured();
    assert!(!provider.is_configured());
    assert_eq!(provider.endpoints(), [] as [String; 0]);
    assert!(provider.client().is_some());
    assert_eq!(provider.name(), "test");
    assert_eq!(provider.capabilities(), ProviderCapabilities::none());

    let scope = scope();
    let operations: Vec<Error> = vec![
        provider.initialize().await.expect_err("initialize"),
        provider
            .add(
                &scope,
                vec![NewMemory::semantic("a write")],
                &AddOptions::default(),
            )
            .await
            .expect_err("add"),
        provider
            .search(&scope, "a query", &SearchOptions::default())
            .await
            .expect_err("search"),
        provider.get(&scope, "id-1").await.expect_err("get"),
        provider
            .update(
                &scope,
                "id-1",
                MemoryUpdate::content("x"),
                &CallOptions::new(),
            )
            .await
            .expect_err("update"),
        provider.delete(&scope, "id-1").await.expect_err("delete"),
        provider
            .list(&MemoryFilter::default())
            .await
            .expect_err("list"),
        provider
            .clear(&MemoryFilter::default())
            .await
            .expect_err("clear"),
        provider.health().await.expect_err("health"),
    ];
    for error in operations {
        assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
        let message = error.to_string();
        assert!(message.contains("testspec"), "{message}");
        assert!(message.contains("does not provide"), "{message}");
        assert!(message.contains("no `endpoint` is configured"), "{message}");
        assert!(message.contains("Test"), "{message}");
    }
    let _ = format!("{provider:?}");
}

#[tokio::test]
async fn a_configured_handle_takes_its_capabilities_from_the_spec() {
    let server = MockServer::start().await;
    let provider = provider(&server);
    assert!(provider.is_configured());
    assert_eq!(provider.endpoints(), vec![server.uri()]);
    assert_eq!(
        provider.capabilities(),
        ProviderCapabilities {
            semantic_search: true,
            keyword_search: true,
            ..ProviderCapabilities::none()
        }
    );
    assert_eq!(provider.spec().id(), "test");
    assert_eq!(provider.auth(), &AuthScheme::Bearer);
    assert_eq!(provider.config().provider_name(), "mem0");
}

#[tokio::test]
async fn add_posts_one_request_per_memory_with_the_bearer_credential() {
    let server = MockServer::start().await;
    mount(
        &server,
        "POST",
        "/items",
        ResponseTemplate::new(200).set_body_json(json!({"id": "mem-1", "memory": "hello"})),
    )
    .await;

    let outcome = provider(&server)
        .add(
            &scope(),
            vec![NewMemory::semantic("hello"), NewMemory::semantic("again")],
            &AddOptions::default(),
        )
        .await
        .expect("add");
    assert_eq!(outcome.added.len(), 2);
    assert_eq!(outcome.added[0].content, "hello");

    let received = server.received_requests().await.expect("requests");
    assert_eq!(received.len(), 2, "one request per memory");
    let first = &received[0];
    assert_eq!(first.method.as_str(), "POST");
    assert_eq!(first.url.path(), "/items");
    let body: serde_json::Value = serde_json::from_slice(&first.body).expect("body");
    assert_eq!(body, json!({"memory": "hello"}));
    assert_eq!(
        first
            .headers
            .get("authorization")
            .and_then(|v| v.to_str().ok()),
        Some("Bearer test-key")
    );
    assert_eq!(
        first
            .headers
            .get("content-type")
            .and_then(|v| v.to_str().ok()),
        Some("application/json")
    );
}

#[tokio::test]
async fn search_maps_results_ignoring_unknown_fields() {
    let server = MockServer::start().await;
    mount_search(
        &server,
        json!({"results": [{"id": "a", "memory": "hello", "score": 0.75, "extra": [1, 2]}]}),
    )
    .await;

    let hits = provider(&server)
        .search(&scope(), "hello", &SearchOptions::default())
        .await
        .expect("search");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].record.id, "a");
    assert_eq!(hits[0].record.content, "hello");
    assert!((hits[0].score - 0.75).abs() < f32::EPSILON);

    let empty = MockServer::start().await;
    mount_search(&empty, json!({"results": []})).await;
    let hits = provider(&empty)
        .search(&scope(), "hello", &SearchOptions::default())
        .await
        .expect("empty search");
    assert_eq!(hits, [] as [RecallHit; 0]);
}

#[tokio::test]
async fn a_missing_record_maps_to_none_false_and_not_found() {
    let server = MockServer::start().await;
    mount(&server, "GET", "/items/missing", ResponseTemplate::new(404)).await;
    mount(&server, "PUT", "/items/missing", ResponseTemplate::new(404)).await;
    mount(
        &server,
        "DELETE",
        "/items/missing",
        ResponseTemplate::new(404),
    )
    .await;
    let provider = provider(&server);

    assert_eq!(provider.get(&scope(), "missing").await.expect("get"), None);
    assert!(!provider.delete(&scope(), "missing").await.expect("delete"));
    let error = provider
        .update(
            &scope(),
            "missing",
            MemoryUpdate::content("changed"),
            &CallOptions::new(),
        )
        .await
        .expect_err("update");
    assert!(matches!(error, Error::NotFound(_)), "{error:?}");
}

#[tokio::test]
async fn a_get_success_decodes_the_record() {
    let server = MockServer::start().await;
    mount(
        &server,
        "GET",
        "/items/a",
        ResponseTemplate::new(200)
            .set_body_json(json!({"id": "a", "memory": "hello", "created_at": 1_700_000_000})),
    )
    .await;
    let record = provider(&server)
        .get(&scope(), "a")
        .await
        .expect("get")
        .expect("some");
    assert_eq!(record.id, "a");
    assert_eq!(record.content, "hello");
    assert_eq!(record.scope, scope());
    assert_eq!(
        record.timestamp,
        chrono::DateTime::from_timestamp(1_700_000_000, 0).expect("timestamp")
    );
}

#[tokio::test]
async fn status_answers_map_to_typed_errors() {
    let server = MockServer::start().await;
    mount(&server, "GET", "/items/denied", ResponseTemplate::new(401)).await;
    mount(
        &server,
        "GET",
        "/items/invalid",
        ResponseTemplate::new(422).set_body_json(json!({"error": "id must be a uuid"})),
    )
    .await;
    let provider = provider(&server);

    let error = provider.get(&scope(), "denied").await.expect_err("401");
    assert!(matches!(error, Error::Security(_)), "{error:?}");
    assert!(
        error.to_string().contains("rejected the credentials"),
        "{error}"
    );

    let error = provider.get(&scope(), "invalid").await.expect_err("422");
    assert!(matches!(error, Error::Validation(_)), "{error:?}");
    assert!(error.to_string().contains("id must be a uuid"), "{error}");
}

#[tokio::test]
async fn a_transient_status_is_retried_once_then_succeeds() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(503))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    mount(&server, "GET", "/health", ResponseTemplate::new(200)).await;

    let health = provider(&server).health().await.expect("health");
    assert!(health.healthy, "{health:?}");
    let received = server.received_requests().await.expect("requests");
    assert_eq!(received.len(), 2, "one retry");
}

#[tokio::test]
async fn exhausted_retries_surface_as_a_transport_error() {
    let server = MockServer::start().await;
    mount(&server, "GET", "/health", ResponseTemplate::new(503)).await;
    mount(&server, "GET", "/items", ResponseTemplate::new(503)).await;
    let provider = RestProvider::from_config(
        spec(),
        &ProviderConfig::new(ProviderKind::Mem0)
            .with_endpoint(server.uri())
            .with_option("max_retries", json!(0))
            .with_option("retry_backoff_ms", json!(1)),
    );

    let health = provider
        .health()
        .await
        .expect("health is degraded, not failing");
    assert!(!health.healthy);
    assert!(
        health.detail.as_deref().unwrap_or_default().contains("503"),
        "{health:?}"
    );
    let list_error = provider
        .list(&MemoryFilter::default())
        .await
        .expect_err("list");
    assert!(matches!(list_error, Error::Transport(_)), "{list_error:?}");
    let received = server.received_requests().await.expect("requests");
    assert_eq!(received.len(), 2, "no retry with max_retries 0");
}

#[tokio::test]
async fn a_failing_endpoint_fails_over_to_the_next() {
    let primary = MockServer::start().await;
    mount(&primary, "POST", "/search", ResponseTemplate::new(503)).await;
    let secondary = MockServer::start().await;
    mount_search(
        &secondary,
        json!({"results": [{"id": "a", "memory": "hello"}]}),
    )
    .await;

    let provider = RestProvider::from_config(
        spec(),
        &ProviderConfig::new(ProviderKind::Mem0)
            .with_endpoint(primary.uri())
            .with_option("failover_endpoints", json!([secondary.uri()]))
            .with_option("max_retries", json!(0))
            .with_option("retry_backoff_ms", json!(1)),
    );
    assert_eq!(provider.endpoints().len(), 2);

    let hits = provider
        .search(&scope(), "hello", &SearchOptions::default())
        .await
        .expect("failover search");
    assert_eq!(hits.len(), 1);
    assert_eq!(
        secondary.received_requests().await.expect("requests").len(),
        1
    );
}

#[tokio::test]
async fn every_endpoint_failing_names_the_last_endpoint() {
    let primary = MockServer::start().await;
    mount(&primary, "GET", "/items", ResponseTemplate::new(503)).await;
    let secondary = MockServer::start().await;
    mount(&secondary, "GET", "/items", ResponseTemplate::new(503)).await;

    let provider = RestProvider::from_config(
        spec(),
        &ProviderConfig::new(ProviderKind::Mem0)
            .with_endpoint(primary.uri())
            .with_option("failover_endpoints", json!([secondary.uri()]))
            .with_option("max_retries", json!(0)),
    );
    let error = provider
        .list(&MemoryFilter::default())
        .await
        .expect_err("list");
    assert!(matches!(error, Error::Transport(_)), "{error:?}");
    let message = error.to_string();
    assert!(message.contains(&secondary.uri()), "{message}");
    assert!(message.contains("2 endpoint(s)"), "{message}");
}

#[tokio::test]
async fn initialize_honours_the_probe_verdict() {
    for status in [401, 403] {
        let rejected = MockServer::start().await;
        mount(&rejected, "GET", "/health", ResponseTemplate::new(status)).await;
        let provider = provider(&rejected);
        let error = provider
            .initialize()
            .await
            .expect_err("rejected credentials");
        assert!(matches!(error, Error::Security(_)), "{status}: {error:?}");
        let message = error.to_string();
        assert!(message.contains("rejected"), "{message}");
        assert!(message.contains("credentials"), "{message}");
        assert!(!provider.is_initialized(), "{status}: stays uninitialized");
        assert_eq!(
            rejected.received_requests().await.expect("requests").len(),
            1,
            "{status}: a definitive answer is never retried"
        );
    }

    let healthy = MockServer::start().await;
    mount(&healthy, "GET", "/health", ResponseTemplate::new(200)).await;
    let provider = provider(&healthy);
    assert!(!provider.is_initialized(), "nothing is initialized yet");
    provider.initialize().await.expect("a healthy probe");
    assert!(provider.is_initialized());
    provider.initialize().await.expect("idempotent initialize");
    assert!(provider.is_initialized());
    assert_eq!(
        healthy.received_requests().await.expect("requests").len(),
        1,
        "a second initialize does no I/O"
    );

    let failing = MockServer::start().await;
    mount(&failing, "GET", "/health", ResponseTemplate::new(500)).await;
    let provider = RestProvider::from_config(
        spec(),
        &ProviderConfig::new(ProviderKind::Mem0)
            .with_endpoint(failing.uri())
            .with_option("max_retries", json!(0)),
    );
    let error = provider.initialize().await.expect_err("a failing probe");
    assert!(matches!(error, Error::Transport(_)), "{error:?}");
    assert!(error.to_string().contains("500"), "{error}");
    assert!(
        !provider.is_initialized(),
        "a failed probe is not initialized"
    );
}

#[tokio::test]
async fn a_failed_probe_leaves_the_handle_uninitialized_for_a_later_retry() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(503).set_body_json(json!({"error": "starting up"})))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    mount(&server, "GET", "/health", ResponseTemplate::new(200)).await;
    let provider = RestProvider::from_config(
        spec(),
        &ProviderConfig::new(ProviderKind::Mem0)
            .with_endpoint(server.uri())
            .with_option("max_retries", json!(0)),
    );

    let error = provider.initialize().await.expect_err("degraded probe");
    assert!(matches!(error, Error::Transport(_)), "{error:?}");
    assert!(error.to_string().contains("starting up"), "{error}");
    assert!(
        !provider.is_initialized(),
        "a degraded probe is not a setup"
    );

    provider.initialize().await.expect("the retried probe");
    assert!(provider.is_initialized());
    assert_eq!(
        server.received_requests().await.expect("requests").len(),
        2,
        "the second initialize really probes again"
    );
}

#[tokio::test]
async fn clear_without_a_bulk_delete_removes_record_by_record() {
    let server = MockServer::start().await;
    mount(
        &server,
        "GET",
        "/items",
        ResponseTemplate::new(200).set_body_json(json!({"items": [{"id": "a"}, {"id": "b"}]})),
    )
    .await;
    mount(&server, "DELETE", "/items/a", ResponseTemplate::new(200)).await;
    mount(&server, "DELETE", "/items/b", ResponseTemplate::new(404)).await;

    let removed = provider(&server)
        .clear(&MemoryFilter::for_scope(&scope()))
        .await
        .expect("clear");
    assert_eq!(removed, 1, "a delete answering 404 is not counted");
}

#[tokio::test]
async fn health_without_a_probe_falls_back_to_a_listing() {
    let server = MockServer::start().await;
    mount(
        &server,
        "GET",
        "/items",
        ResponseTemplate::new(200).set_body_json(json!({"items": []})),
    )
    .await;
    let provider = RestProvider::from_config(
        &LIST_ONLY,
        &ProviderConfig::new(ProviderKind::Custom("listonly".to_owned()))
            .with_endpoint(server.uri()),
    );

    let health = provider.health().await.expect("health");
    assert!(health.healthy, "{health:?}");
    let received = server.received_requests().await.expect("requests");
    assert_eq!(received.len(), 1);
    assert_eq!(received[0].url.path(), "/items");
    assert_eq!(received[0].headers.get("authorization"), None);
}

#[tokio::test]
async fn a_cancelled_call_sends_nothing() {
    let server = MockServer::start().await;
    mount_search(&server, json!({"results": []})).await;

    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let options = SearchOptions::default().with_cancellation(cancellation);
    let error = provider(&server)
        .search(&scope(), "hello", &options)
        .await
        .expect_err("cancelled");
    assert!(matches!(error, Error::Cancelled), "{error:?}");
    assert!(server
        .received_requests()
        .await
        .expect("requests")
        .is_empty());
}

#[tokio::test]
async fn validation_precedes_the_request() {
    let server = MockServer::start().await;
    let provider = provider(&server);

    assert!(matches!(
        provider.get(&scope(), "  ").await.expect_err("blank id"),
        Error::Validation(_)
    ));
    assert!(matches!(
        provider
            .search(&scope(), "   ", &SearchOptions::default())
            .await
            .expect_err("blank query"),
        Error::Validation(_)
    ));
    assert!(matches!(
        provider
            .update(
                &scope(),
                "id-1",
                MemoryUpdate::default(),
                &CallOptions::new()
            )
            .await
            .expect_err("empty update"),
        Error::Validation(_)
    ));
    assert!(server
        .received_requests()
        .await
        .expect("requests")
        .is_empty());
}
