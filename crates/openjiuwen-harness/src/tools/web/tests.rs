// Test-only suite: unwrap/expect are the assertion vocabulary of this module.
#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Cross-module tests of the web tool surface.
//!
//! Every case runs offline: the real `reqwest` path is exercised against a loopback
//! [`TestServer`], and injected [`StubFetcher`]/`FakeSearchBackend` seams cover the failures a
//! loopback server cannot produce deterministically.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::net::TcpListener;

use openjiuwen_foundation::tool::{Tool, ToolContext};

use super::fetch::{FetchRequest, FetchWebpageTool, HttpFetcher, ReqwestFetcher};
use super::search::{
    not_configured_reason, FakeSearchBackend, HttpSearchBackend, SearchBackend, SearchResult,
    WebSearchTool, ENDPOINT_ENV,
};
use super::test_support::{http_response, StubFetcher, TestServer};
use crate::permission::{
    NetGuardSection, PermissionEngine, PermissionEngineConfig, PermissionLevel, PermissionsSection,
};
use crate::tools::support::ToolEnv;
use crate::workspace::Workspace;

/// Builds a fetch tool over a fresh temp workspace, keeping the directory alive.
fn fetch_tool(fetcher: Arc<dyn HttpFetcher>) -> (FetchWebpageTool, TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let tool = FetchWebpageTool::new(
        ToolEnv::new(Workspace::new(dir.path())),
        Some(fetcher),
        "en",
    );
    (tool, dir)
}

/// Invokes a fetch tool and unwraps the payload envelope.
async fn fetch(tool: &FetchWebpageTool, inputs: Value) -> Value {
    tool.invoke(inputs, &ToolContext::new()).await.unwrap()
}

/// The URL of a loopback port that nothing listens on.
async fn closed_port_url() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    format!("http://{address}/nothing")
}

#[tokio::test]
async fn loopback_html_page_is_extracted() {
    let requests = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let recorded = Arc::clone(&requests);
    let server = TestServer::start(move |request| {
        recorded.lock().push(request.to_owned());
        http_response(
            200,
            "text/html; charset=utf-8",
            b"<html><head><title>Doc</title><style>b{color:red}</style></head>\
              <body><h1>Hi</h1><p>World &amp; friends</p><script>x()</script></body></html>",
        )
    })
    .await;

    let (tool, _dir) = fetch_tool(Arc::new(ReqwestFetcher::new()));
    let payload = fetch(&tool, json!({"url": server.url("/doc")})).await;

    assert_eq!(payload["success"], json!(true));
    let data = &payload["data"];
    assert_eq!(data["title"], json!("Doc"));
    assert_eq!(data["content"], json!("Hi\nWorld & friends"));
    assert_eq!(data["status"], json!(200));
    assert_eq!(data["final_url"], json!(server.url("/doc")));
    assert_eq!(data["charset"], json!("utf-8"));
    assert_eq!(data["truncated"], json!(false));
    assert!(data["content_type"]
        .as_str()
        .unwrap()
        .starts_with("text/html"));

    let head = requests.lock()[0].clone();
    assert!(head.starts_with("GET /doc "), "request: {head}");
    assert!(
        head.to_lowercase().contains("user-agent:"),
        "request: {head}"
    );
}

#[tokio::test]
async fn loopback_json_is_pretty_printed() {
    let server = TestServer::start(|_| {
        http_response(200, "application/json", br#"{"name":"value","count":2}"#)
    })
    .await;
    let (tool, _dir) = fetch_tool(Arc::new(ReqwestFetcher::new()));
    let payload = fetch(&tool, json!({"url": server.url("/api")})).await;
    let content = payload["data"]["content"].as_str().unwrap();
    assert!(
        content.contains("\"name\": \"value\""),
        "content: {content}"
    );
    assert!(content.contains("\"count\": 2"), "content: {content}");
    assert!(payload["data"].get("title").is_none());
}

#[tokio::test]
async fn loopback_header_charset_is_decoded() {
    let server =
        TestServer::start(|_| http_response(200, "text/plain; charset=iso-8859-1", b"caf\xe9"))
            .await;
    let (tool, _dir) = fetch_tool(Arc::new(ReqwestFetcher::new()));
    let payload = fetch(&tool, json!({"url": server.url("/latin1")})).await;
    assert_eq!(payload["data"]["charset"], json!("iso-8859-1"));
    assert_eq!(payload["data"]["content"], json!("café"));
}

#[tokio::test]
async fn loopback_meta_charset_is_sniffed() {
    let server = TestServer::start(|_| {
        http_response(
            200,
            "text/html",
            b"<html><head><meta charset=\"iso-8859-1\"><title>Caf\xe9</title></head>\
              <body><p>caf\xe9</p></body></html>",
        )
    })
    .await;
    let (tool, _dir) = fetch_tool(Arc::new(ReqwestFetcher::new()));
    let payload = fetch(&tool, json!({"url": server.url("/meta")})).await;
    assert_eq!(payload["data"]["charset"], json!("iso-8859-1"));
    assert_eq!(payload["data"]["title"], json!("Café"));
    assert_eq!(payload["data"]["content"], json!("café"));
}

#[tokio::test]
async fn loopback_redirects_are_followed() {
    let server = TestServer::start(|request| {
        if request.starts_with("GET /start ") {
            b"HTTP/1.1 302 Found\r\nLocation: /next\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                .to_vec()
        } else {
            http_response(
                200,
                "text/html",
                b"<html><head><title>Next</title></head><body><p>arrived</p></body></html>",
            )
        }
    })
    .await;
    let (tool, _dir) = fetch_tool(Arc::new(ReqwestFetcher::new()));
    let payload = fetch(&tool, json!({"url": server.url("/start")})).await;
    assert_eq!(payload["data"]["title"], json!("Next"));
    assert_eq!(payload["data"]["content"], json!("arrived"));
    assert_eq!(payload["data"]["final_url"], json!(server.url("/next")));
}

#[tokio::test]
async fn oversized_bodies_are_truncated() {
    let body = "x".repeat(5000);
    let server =
        TestServer::start(move |_| http_response(200, "text/plain", body.as_bytes())).await;
    let (tool, _dir) = fetch_tool(Arc::new(ReqwestFetcher::new()));
    let payload = fetch(
        &tool,
        json!({"url": server.url("/big"), "max_bytes": 100, "max_chars": 0}),
    )
    .await;
    assert_eq!(payload["data"]["truncated"], json!(true));
    let content = payload["data"]["content"].as_str().unwrap();
    assert_eq!(content.chars().count(), 100, "content: {content}");
    assert_eq!(content, "x".repeat(100));
}

#[tokio::test]
async fn reqwest_fetcher_reports_a_timeout() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let holding = tokio::spawn(async move {
        if let Ok((_socket, _)) = listener.accept().await {
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
    });

    let fetcher = ReqwestFetcher::new();
    let error = fetcher
        .fetch(FetchRequest {
            url: format!("http://{address}/slow"),
            timeout: Duration::from_millis(100),
            max_bytes: 1024,
            headers: Vec::new(),
        })
        .await
        .unwrap_err();

    assert!(error.to_string().starts_with("timeout:"), "error: {error}");
    holding.abort();
}

#[tokio::test]
async fn connection_refused_becomes_a_failure_payload() {
    let (tool, _dir) = fetch_tool(Arc::new(ReqwestFetcher::new()));
    let payload = fetch(&tool, json!({"url": closed_port_url().await})).await;
    assert_eq!(payload["success"], json!(false));
    assert_eq!(payload["data"], Value::Null);
    assert!(
        payload["error"]
            .as_str()
            .unwrap()
            .starts_with("failed to fetch webpage: "),
        "payload: {payload}"
    );
}

#[tokio::test]
async fn timeout_errors_become_a_failure_payload() {
    let fetcher = StubFetcher::failing(openjiuwen_common::error::Error::timeout(
        "request timed out: slow page",
    ));
    let (tool, _dir) = fetch_tool(fetcher);
    let payload = fetch(&tool, json!({"url": "https://example.com/slow"})).await;
    assert_eq!(payload["success"], json!(false));
    assert_eq!(
        payload["error"],
        json!("failed to fetch webpage: timeout: request timed out: slow page")
    );
}

#[tokio::test]
async fn net_guard_denial_blocks_the_fetch() {
    let fetcher = StubFetcher::ok("text/plain", "should never be read");
    let section = PermissionsSection {
        package_builtin_rules: false,
        defaults: BTreeMap::from([("*".to_owned(), PermissionLevel::Allow)]),
        net_guard: NetGuardSection {
            enabled: true,
            defaults: PermissionLevel::Deny,
            urls: BTreeMap::new(),
        },
        ..PermissionsSection::default()
    };
    let engine = Arc::new(PermissionEngine::new(PermissionEngineConfig::new(section)));
    let dir = tempfile::tempdir().unwrap();
    let env = ToolEnv::new(Workspace::new(dir.path())).with_permission(engine);
    let tool = FetchWebpageTool::new(env, Some(fetcher.clone()), "en");

    let payload = fetch(&tool, json!({"url": "https://blocked.example/page"})).await;

    assert_eq!(payload["success"], json!(false));
    let error = payload["error"].as_str().unwrap();
    assert!(error.starts_with("Denied by rule: "), "error: {error}");
    assert!(error.contains("net_guard"), "error: {error}");
    assert_eq!(fetcher.calls(), 0);
}

#[tokio::test]
async fn fake_search_backend_round_trip() {
    let backend = Arc::new(FakeSearchBackend::with_results(vec![
        SearchResult::new("OpenJiuwen", "https://openjiuwen.example", "agent SDK"),
        SearchResult::new("Rust", "https://rust.example", "language"),
    ]));
    let tool = WebSearchTool::new(Some(backend.clone()), "en");
    let payload = tool
        .invoke(
            json!({"query": "openjiuwen rust", "max_results": 2}),
            &ToolContext::new(),
        )
        .await
        .unwrap();

    assert_eq!(payload["success"], json!(true));
    assert_eq!(payload["data"]["success"], json!(true));
    assert_eq!(
        payload["data"]["results"][0],
        json!({"title": "OpenJiuwen", "url": "https://openjiuwen.example", "snippet": "agent SDK"})
    );
    assert_eq!(backend.calls().len(), 1);
    assert_eq!(backend.calls()[0].query, "openjiuwen rust");
    assert_eq!(backend.calls()[0].max_results, 2);
}

#[tokio::test]
async fn missing_search_backend_fails() {
    let tool = WebSearchTool::new(None, "en");
    let payload = tool
        .invoke(json!({"query": "rust"}), &ToolContext::new())
        .await
        .unwrap();
    assert_eq!(payload["success"], json!(false));
    assert_eq!(payload["error"], json!(not_configured_reason()));
    let error = payload["error"].as_str().unwrap();
    assert!(
        error.starts_with("web search backend is not configured"),
        "error: {error}"
    );
    assert!(error.contains(ENDPOINT_ENV), "error: {error}");
}

/// Starts a loopback search endpoint answering every request with `status`/`body`.
///
/// Returns the server (kept alive by the caller) and the request heads it received, so a test can
/// assert the exact request line and query encoding the backend produced.
async fn search_server(
    status: u16,
    content_type: &'static str,
    body: &'static str,
) -> (TestServer, Arc<parking_lot::Mutex<Vec<String>>>) {
    let requests = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let recorded = Arc::clone(&requests);
    let server = TestServer::start(move |request| {
        recorded.lock().push(request.to_owned());
        http_response(status, content_type, body.as_bytes())
    })
    .await;
    (server, requests)
}

/// Builds a `web_search` tool over `endpoint` with the real `reqwest` fetcher.
fn loopback_search_tool(endpoint: String) -> WebSearchTool {
    let backend = HttpSearchBackend::with_fetcher(endpoint, Arc::new(ReqwestFetcher::new()));
    WebSearchTool::new(Some(Arc::new(backend)), "en")
}

/// Invokes a `web_search` tool and unwraps the payload envelope.
async fn search(tool: &WebSearchTool, inputs: Value) -> Value {
    tool.invoke(inputs, &ToolContext::new()).await.unwrap()
}

#[tokio::test]
async fn loopback_search_encodes_the_query_and_clamps_the_result_count() {
    let (server, requests) = search_server(
        200,
        "application/json",
        r#"{"results":[{"title":"Rust","url":"https://rust.example","snippet":"lang"}]}"#,
    )
    .await;
    let tool = loopback_search_tool(server.url("/search"));

    // `max_results: 0` is clamped up to the minimum of one.
    let payload = search(
        &tool,
        json!({"query": "rust & async/await", "max_results": 0}),
    )
    .await;
    assert_eq!(payload["success"], json!(true));
    assert_eq!(
        payload["data"]["results"],
        json!([{"title": "Rust", "url": "https://rust.example", "snippet": "lang"}])
    );

    // `max_results: 1000` is clamped down to the maximum of twenty, and the absent count uses the
    // documented default of eight.
    let _ = search(&tool, json!({"query": "rust", "max_results": 1000})).await;
    let _ = search(&tool, json!({"query": "rust"})).await;

    let requests = requests.lock();
    assert_eq!(requests.len(), 3);
    assert!(
        requests[0]
            .starts_with("GET /search?q=rust%20%26%20async%2Fawait&max_results=1 HTTP/1.1\r\n"),
        "request: {}",
        requests[0]
    );
    assert!(
        requests[1].starts_with("GET /search?q=rust&max_results=20 HTTP/1.1\r\n"),
        "request: {}",
        requests[1]
    );
    assert!(
        requests[2].starts_with("GET /search?q=rust&max_results=8 HTTP/1.1\r\n"),
        "request: {}",
        requests[2]
    );
    for request in requests.iter() {
        assert!(
            request.to_lowercase().contains("accept: application/json"),
            "request: {request}"
        );
    }
}

#[tokio::test]
async fn loopback_search_uses_the_query_template_when_the_endpoint_has_one() {
    let (server, requests) = search_server(200, "application/json", r#"{"results":[]}"#).await;
    let tool = loopback_search_tool(server.url("/find?text={query}"));
    let payload = search(&tool, json!({"query": "a b&c", "max_results": 5})).await;
    assert_eq!(payload["success"], json!(true));
    assert_eq!(payload["data"]["results"], json!([]));
    let head = requests.lock()[0].clone();
    assert!(
        head.starts_with("GET /find?text=a%20b%26c HTTP/1.1\r\n"),
        "request: {head}"
    );
    assert!(!head.contains("max_results"), "request: {head}");
}

#[tokio::test]
async fn loopback_search_reports_an_http_failure_as_a_runtime_failure() {
    let (server, _requests) =
        search_server(500, "application/json", r#"{"error":"upstream is down"}"#).await;
    let tool = loopback_search_tool(server.url("/search"));
    let payload = search(&tool, json!({"query": "rust"})).await;
    assert_eq!(payload["success"], json!(false));
    let error = payload["error"].as_str().unwrap();
    assert!(error.starts_with("web search failed: "), "error: {error}");
    assert!(error.contains("HTTP 500"), "error: {error}");
    assert!(error.contains("/search?q=rust"), "error: {error}");
    assert!(!error.contains("not configured"), "error: {error}");
}

#[tokio::test]
async fn loopback_search_reports_malformed_json_as_a_runtime_failure() {
    let (server, _requests) = search_server(200, "application/json", "not json").await;
    let tool = loopback_search_tool(server.url("/search"));
    let payload = search(&tool, json!({"query": "rust"})).await;
    assert_eq!(payload["success"], json!(false));
    let error = payload["error"].as_str().unwrap();
    assert!(error.starts_with("web search failed: "), "error: {error}");
    assert!(error.contains("invalid JSON"), "error: {error}");
}

#[tokio::test]
async fn loopback_search_accepts_empty_result_sets() {
    let (object_server, _) = search_server(200, "application/json", r#"{"results":[]}"#).await;
    let payload = search(
        &loopback_search_tool(object_server.url("/search")),
        json!({"query": "nothing"}),
    )
    .await;
    assert_eq!(payload["success"], json!(true));
    assert_eq!(payload["data"]["results"], json!([]));

    let (array_server, _) = search_server(200, "application/json", "[]").await;
    let payload = search(
        &loopback_search_tool(array_server.url("/search")),
        json!({"query": "nothing"}),
    )
    .await;
    assert_eq!(payload["success"], json!(true));
    assert_eq!(payload["data"]["results"], json!([]));
}

#[tokio::test]
async fn endpoint_lookup_configures_the_loopback_backend() {
    let (server, requests) = search_server(
        200,
        "application/json",
        r#"{"results":[{"title":"Env","url":"https://env.example","snippet":"s"}]}"#,
    )
    .await;
    let endpoint = server.url("/env-search");
    let backend = HttpSearchBackend::from_lookup(|key| {
        (key == ENDPOINT_ENV).then(|| format!("  {endpoint}  "))
    })
    .unwrap();
    let tool = WebSearchTool::new(Some(Arc::new(backend)), "en");
    let payload = search(&tool, json!({"query": "rust"})).await;
    assert_eq!(
        payload["data"]["results"],
        json!([{"title": "Env", "url": "https://env.example", "snippet": "s"}])
    );
    let head = requests.lock()[0].clone();
    assert!(
        head.starts_with("GET /env-search?q=rust&max_results=8 HTTP/1.1\r\n"),
        "request: {head}"
    );
}

#[tokio::test]
async fn a_tool_without_a_configured_endpoint_reports_the_typed_gap_without_a_request() {
    let (server, requests) = search_server(200, "application/json", r#"{"results":[]}"#).await;
    // The endpoint is blank, so no connection is attempted even though the URL is reachable.
    let tool = WebSearchTool::with_endpoint("   ", "en");
    let payload = search(&tool, json!({"query": "rust"})).await;
    assert_eq!(payload["success"], json!(false));
    assert_eq!(payload["error"], json!(not_configured_reason()));
    assert!(requests.lock().is_empty());
    drop(server);

    let backend = HttpSearchBackend::from_lookup(|_| None).err().unwrap();
    assert_eq!(&*backend.message(), not_configured_reason().as_str());
}

/// A configured provider that cannot serve this deployment reports `Unsupported`, which the tool
/// renders as its bare reason rather than as a runtime failure.
struct UnsupportedSearchBackend;

#[async_trait::async_trait]
impl SearchBackend for UnsupportedSearchBackend {
    async fn search(
        &self,
        _query: &str,
        _max_results: usize,
    ) -> openjiuwen_common::error::Result<Vec<SearchResult>> {
        Err(openjiuwen_common::error::Error::unsupported(
            "web search vendor API keys are not ported",
        ))
    }
}

#[tokio::test]
async fn an_unsupported_provider_reports_its_bare_reason() {
    let tool = WebSearchTool::new(Some(Arc::new(UnsupportedSearchBackend)), "en");
    let payload = search(&tool, json!({"query": "rust"})).await;
    assert_eq!(payload["success"], json!(false));
    assert_eq!(
        payload["error"],
        json!("web search vendor API keys are not ported")
    );
}

#[tokio::test]
async fn http_search_backend_round_trip_over_loopback() {
    let requests = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let recorded = Arc::clone(&requests);
    let server = TestServer::start(move |request| {
        recorded.lock().push(request.to_owned());
        http_response(
            200,
            "application/json",
            br#"{"results":[{"title":"Rust","url":"https://rust.example","snippet":"lang"}]}"#,
        )
    })
    .await;

    let backend =
        HttpSearchBackend::with_fetcher(server.url("/search"), Arc::new(ReqwestFetcher::new()));
    let tool = WebSearchTool::new(Some(Arc::new(backend)), "en");
    let payload = tool
        .invoke(
            json!({"query": "rust lang", "max_results": 3}),
            &ToolContext::new(),
        )
        .await
        .unwrap();

    assert_eq!(
        payload["data"]["results"],
        json!([{"title": "Rust", "url": "https://rust.example", "snippet": "lang"}])
    );
    let head = requests.lock()[0].clone();
    assert!(
        head.starts_with("GET /search?q=rust%20lang&max_results=3 "),
        "request: {head}"
    );
    assert!(
        head.to_lowercase().contains("accept: application/json"),
        "request: {head}"
    );
}
