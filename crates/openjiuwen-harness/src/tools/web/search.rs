//! `web_search`: a search-results tool over an injected [`SearchBackend`].
//!
//! Ports the tool surface of `openjiuwen/harness/tools/web/free_search.py` and `paid_search.py`
//! without their engine internals: the model-facing contract is one `query`, a clamped result
//! count and a `{title, url, snippet}` result list. The free DuckDuckGo/Bing HTML scrapers, the
//! paid vendor SDKs (Bocha/Perplexity/Serper/Jina), redirect unwrapping, relevance assessment and
//! the LLM quality/budget passes are deferred — a host supplies whatever provider it wants behind
//! [`SearchBackend`], including an HTTP JSON endpoint through [`HttpSearchBackend`].
//!
//! # Configuration, and why "not configured" is typed
//!
//! There are three ways to open the backend gap, in increasing order of effort:
//!
//! 1. [`HttpSearchBackend::from_env`] reads [`ENDPOINT_ENV`] (`OPENJIUWEN_WEB_SEARCH_ENDPOINT`) and
//!    builds the HTTP backend from it; [`HttpSearchBackend::from_lookup`] is the same thing over an
//!    injected environment, so hosts and tests stay deterministic.
//! 2. [`HttpSearchBackend::new`] takes the endpoint (or a `{query}` template) directly, and
//!    [`WebSearchTool::with_endpoint`] wraps that in one call.
//! 3. a host implements [`SearchBackend`] over its vendor of choice and wires it with
//!    `ToolRegistry::with_search_backend`.
//!
//! "No endpoint" is a *configuration* gap, not a provider failure, so it is reported as
//! [`Error::Unsupported`] carrying [`not_configured_reason`] (which names [`ENDPOINT_ENV`] and the
//! two constructors). A provider that is configured but fails keeps its own
//! [`Error::Tool`]/[`Error::Transport`] kind, and the tool renders the two differently: the
//! configuration gap is the bare reason, a runtime failure is `web search failed: <error>`.
//!
//! Every expected failure is a `ToolPayload::fail(...)`; `Err` is reserved for cancellation and
//! protocol violations.

// The registry contract fixes the `Search*` names inside this `search` module.
#![allow(clippy::module_name_repetitions)]

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::tool::{Tool, ToolCard, ToolContext};

use super::decode::{decode_body, detect_charset};
use super::fetch::{FetchRequest, HttpFetcher, ReqwestFetcher};
use crate::tools::descriptions;
use crate::tools::support::{parse_args, schema_for, tool_card, ToolPayload};

/// The model-facing tool name.
pub const TOOL_NAME: &str = "web_search";

/// Default number of results returned when the call does not ask for a count.
pub const DEFAULT_MAX_RESULTS: usize = 8;

/// Lower bound of the requested result count.
pub const MIN_MAX_RESULTS: usize = 1;

/// Upper bound of the requested result count.
pub const MAX_MAX_RESULTS: usize = 20;

/// Default request timeout of [`HttpSearchBackend`], in seconds.
pub const DEFAULT_TIMEOUT_SECONDS: u64 = 20;

/// Default body cap of [`HttpSearchBackend`], in bytes.
pub const DEFAULT_MAX_BYTES: usize = 1_000_000;

/// Environment variable holding the JSON search endpoint: `OPENJIUWEN_WEB_SEARCH_ENDPOINT`.
///
/// Its value may be a plain URL (the query is appended as `?q=…&max_results=…`) or a template
/// containing `{query}`. Blank values count as unset.
pub const ENDPOINT_ENV: &str = "OPENJIUWEN_WEB_SEARCH_ENDPOINT";

/// How to configure a search backend, appended to every "not configured" reason.
pub const CONFIG_GUIDANCE: &str = "set OPENJIUWEN_WEB_SEARCH_ENDPOINT to a JSON search endpoint \
     (it may contain a `{query}` placeholder), construct HttpSearchBackend::new(endpoint) or \
     WebSearchTool::with_endpoint(..), or wire your own provider with \
     ToolRegistry::with_search_backend(..); the Python DuckDuckGo/Bing scrapers and the paid \
     vendor SDKs are not ported, so a vendor needs its own SearchBackend implementation";

/// The typed reason reported when no search endpoint is configured.
///
/// The historical `web search backend is not configured` prefix is kept so existing callers that
/// match on it keep working; the guidance that follows states how to configure it.
#[must_use]
pub fn not_configured_reason() -> String {
    format!("web search backend is not configured: {CONFIG_GUIDANCE}")
}

/// One search result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchResult {
    /// The result title.
    pub title: String,
    /// The absolute result URL.
    pub url: String,
    /// The result snippet or summary, possibly empty.
    pub snippet: String,
}

impl SearchResult {
    /// Builds a result from its three parts.
    #[must_use]
    pub fn new(
        title: impl Into<String>,
        url: impl Into<String>,
        snippet: impl Into<String>,
    ) -> Self {
        Self {
            title: title.into(),
            url: url.into(),
            snippet: snippet.into(),
        }
    }
}

/// The injected search seam of `web_search`.
///
/// Implementations own the provider specifics (endpoint, credentials, ranking) and only have to
/// return at most `max_results`-shaped results in provider order.
#[async_trait]
pub trait SearchBackend: Send + Sync {
    /// Runs one query.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Tool`] for a provider failure (transport error, non-2xx status,
    /// malformed payload) so the tool can render it as a failure payload.
    async fn search(&self, query: &str, max_results: usize) -> Result<Vec<SearchResult>>;
}

/// A [`SearchBackend`] over an HTTP JSON endpoint.
///
/// The endpoint may be either a plain URL — the query is appended as `?q=…&max_results=…` — or a
/// template containing `{query}`, which is replaced with the percent-encoded query and receives
/// no extra parameters. The response must be a JSON array of results or an object with a
/// `results` array; each entry needs a `url` (or `link`/`href`) and may carry `title`/`name` and
/// `snippet`/`description`/`content`.
pub struct HttpSearchBackend {
    /// The endpoint URL or `{query}` template.
    endpoint: String,
    /// The HTTP seam.
    fetcher: Arc<dyn HttpFetcher>,
    /// The request timeout.
    timeout: Duration,
    /// The response byte cap.
    max_bytes: usize,
}

impl HttpSearchBackend {
    /// Builds a backend over `endpoint` with the default [`ReqwestFetcher`].
    #[must_use]
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self::with_fetcher(endpoint, Arc::new(ReqwestFetcher::new()))
    }

    /// Builds a backend over `endpoint` using an injected fetcher.
    #[must_use]
    pub fn with_fetcher(endpoint: impl Into<String>, fetcher: Arc<dyn HttpFetcher>) -> Self {
        Self {
            endpoint: endpoint.into(),
            fetcher,
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECONDS),
            max_bytes: DEFAULT_MAX_BYTES,
        }
    }

    /// Builds a backend from [`ENDPOINT_ENV`], or reports the typed configuration gap.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] carrying [`not_configured_reason`] when the variable is
    /// unset or blank.
    pub fn from_env() -> Result<Self> {
        Self::from_lookup(|key| std::env::var(key).ok())
    }

    /// Builds a backend from an injected environment, which keeps hosts and tests deterministic.
    ///
    /// `lookup` receives [`ENDPOINT_ENV`] and returns its raw value when set.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] carrying [`not_configured_reason`] when `lookup` yields
    /// nothing or only whitespace.
    pub fn from_lookup<F>(lookup: F) -> Result<Self>
    where
        F: Fn(&str) -> Option<String>,
    {
        let endpoint = lookup(ENDPOINT_ENV)
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty());
        match endpoint {
            Some(endpoint) => Ok(Self::new(endpoint)),
            None => Err(Error::unsupported(not_configured_reason())),
        }
    }

    /// Whether this backend has an endpoint to call.
    ///
    /// A backend built with a blank endpoint has none; every call then reports the typed
    /// configuration gap instead of issuing a request to nowhere.
    #[must_use]
    pub fn is_configured(&self) -> bool {
        !self.endpoint.trim().is_empty()
    }

    /// Overrides the request timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Overrides the response byte cap.
    #[must_use]
    pub fn with_max_bytes(mut self, max_bytes: usize) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    /// The configured endpoint URL or template.
    #[must_use]
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Builds the request URL for `query`.
    fn request_url(&self, query: &str, max_results: usize) -> String {
        let encoded = percent_encode(query);
        if self.endpoint.contains("{query}") {
            return self.endpoint.replace("{query}", &encoded);
        }
        let separator = if self.endpoint.contains('?') {
            '&'
        } else {
            '?'
        };
        format!(
            "{}{separator}q={encoded}&max_results={max_results}",
            self.endpoint
        )
    }
}

#[async_trait]
impl SearchBackend for HttpSearchBackend {
    async fn search(&self, query: &str, max_results: usize) -> Result<Vec<SearchResult>> {
        if !self.is_configured() {
            return Err(Error::unsupported(not_configured_reason()));
        }
        let url = self.request_url(query, max_results);
        let request = FetchRequest {
            url: url.clone(),
            timeout: self.timeout,
            max_bytes: self.max_bytes,
            headers: vec![("Accept".to_owned(), "application/json".to_owned())],
        };
        let response = self.fetcher.fetch(request).await?;
        if !(200..300).contains(&response.status) {
            return Err(Error::tool(format!(
                "web search endpoint returned HTTP {} for {url}",
                response.status
            )));
        }
        let charset = detect_charset(response.content_type(), &response.body);
        let text = decode_body(&response.body, &charset);
        let results = parse_search_response(&text)?;
        Ok(results.into_iter().take(max_results).collect())
    }
}

/// Parses the JSON payload of a search endpoint.
///
/// Accepts `{"results": [...]}` and a bare top-level array. Entries without a usable URL are
/// skipped; a document that is not valid JSON, or whose `results` field is not an array, is an
/// [`Error::Tool`].
///
/// # Errors
///
/// Returns [`Error::Tool`] when the payload is not valid JSON or does not have the expected
/// result-array shape.
pub fn parse_search_response(text: &str) -> Result<Vec<SearchResult>> {
    let value: Value = serde_json::from_str(text)
        .map_err(|error| Error::tool(format!("web search returned invalid JSON: {error}")))?;
    let entries = match value {
        Value::Array(entries) => entries,
        Value::Object(mut object) => match object.remove("results") {
            Some(Value::Array(entries)) => entries,
            Some(_) => {
                return Err(Error::tool(
                    "web search response has a non-array 'results' field",
                ))
            }
            None => return Err(Error::tool("web search response has no 'results' array")),
        },
        _ => {
            return Err(Error::tool(
                "web search response is not a JSON object or array",
            ))
        }
    };
    Ok(entries.iter().filter_map(parse_search_entry).collect())
}

/// Parses one result entry, or `None` when it has no usable URL.
fn parse_search_entry(entry: &Value) -> Option<SearchResult> {
    let object = entry.as_object()?;
    let url = string_field(object, &["url", "link", "href"])?;
    Some(SearchResult {
        title: string_field(object, &["title", "name"]).unwrap_or_default(),
        url,
        snippet: string_field(object, &["snippet", "description", "content"]).unwrap_or_default(),
    })
}

/// Returns the first non-blank string among `keys`.
fn string_field(object: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        let value = object.get(*key)?.as_str()?.trim();
        (!value.is_empty()).then(|| value.to_owned())
    })
}

/// Percent-encodes `input` for use in a query string (RFC 3986 unreserved characters survive).
fn percent_encode(input: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut encoded = String::with_capacity(input.len());
    for byte in input.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            encoded.push(char::from(byte));
        } else {
            encoded.push('%');
            encoded.push(char::from(HEX[usize::from(byte >> 4)]));
            encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
        }
    }
    encoded
}

/// One recorded [`FakeSearchBackend`] call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FakeSearchCall {
    /// The query the tool sent.
    pub query: String,
    /// The result count the tool asked for.
    pub max_results: usize,
}

/// A deterministic offline [`SearchBackend`] for tests and examples.
///
/// It returns canned results (truncated to the requested count) or a canned failure, and records
/// every call so a test can assert what the tool asked for. Python parity: the injected fake
/// clients of `openjiuwen_foundation` (for example `FakeMcpClient`).
#[derive(Debug, Default)]
pub struct FakeSearchBackend {
    /// Canned results, failure and recorded calls.
    state: parking_lot::Mutex<FakeSearchState>,
}

/// The mutable state of [`FakeSearchBackend`].
#[derive(Debug, Default)]
struct FakeSearchState {
    /// Results returned while no failure is set.
    results: Vec<SearchResult>,
    /// When set, every call fails with this message.
    failure: Option<String>,
    /// Calls received so far.
    calls: Vec<FakeSearchCall>,
}

impl FakeSearchBackend {
    /// Builds an empty fake.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds a fake returning `results`.
    #[must_use]
    pub fn with_results(results: Vec<SearchResult>) -> Self {
        Self {
            state: parking_lot::Mutex::new(FakeSearchState {
                results,
                ..FakeSearchState::default()
            }),
        }
    }

    /// Builds a fake whose every call fails with `message`.
    #[must_use]
    pub fn failing(message: impl Into<String>) -> Self {
        Self {
            state: parking_lot::Mutex::new(FakeSearchState {
                failure: Some(message.into()),
                ..FakeSearchState::default()
            }),
        }
    }

    /// Replaces the canned results.
    pub fn set_results(&self, results: Vec<SearchResult>) {
        self.state.lock().results = results;
    }

    /// Sets or clears the canned failure message.
    pub fn set_failure(&self, message: Option<String>) {
        self.state.lock().failure = message;
    }

    /// The calls received so far, in order.
    #[must_use]
    pub fn calls(&self) -> Vec<FakeSearchCall> {
        self.state.lock().calls.clone()
    }

    /// The most recent call, if any.
    #[must_use]
    pub fn last_call(&self) -> Option<FakeSearchCall> {
        self.state.lock().calls.last().cloned()
    }
}

#[async_trait]
impl SearchBackend for FakeSearchBackend {
    async fn search(&self, query: &str, max_results: usize) -> Result<Vec<SearchResult>> {
        let mut state = self.state.lock();
        state.calls.push(FakeSearchCall {
            query: query.to_owned(),
            max_results,
        });
        if let Some(message) = state.failure.clone() {
            return Err(Error::tool(message));
        }
        Ok(state.results.iter().take(max_results).cloned().collect())
    }
}

/// Arguments of the `web_search` tool.
///
/// Python parity: the `query` and `max_results` parameters of the free/paid search tools. The
/// Python `provider` and `timeout_seconds` parameters select an engine and are not part of the
/// Rust contract (the host owns the provider behind [`SearchBackend`]).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WebSearchArgs {
    /// The search query, with the entity and intent preserved.
    pub query: String,
    /// Maximum number of results (clamped to 1..=20, default 8).
    #[serde(default)]
    #[schemars(with = "u64")]
    pub max_results: Option<u64>,
}

/// Searches the web and returns a ranked result list.
///
/// The tool holds no workspace state and does not consult the permission pipeline: a search
/// query is not a path, and the network policy of a host applies to fetching, not to the
/// provider call the host injected.
pub struct WebSearchTool {
    /// The injected provider; `None` makes every call fail with a configuration message.
    backend: Option<Arc<dyn SearchBackend>>,
    /// The model-facing card built from the constructor language.
    card: ToolCard,
}

impl WebSearchTool {
    /// Builds the tool over `backend`.
    #[must_use]
    pub fn new(backend: Option<Arc<dyn SearchBackend>>, language: impl Into<String>) -> Self {
        let language = language.into();
        Self {
            backend,
            card: Self::card_for(&language),
        }
    }

    /// Builds the tool over an HTTP JSON endpoint (or a `{query}` template).
    ///
    /// This is the one-call way to open the backend gap: a blank endpoint builds a backend that
    /// reports the typed configuration gap on every call, exactly like a tool with no backend.
    #[must_use]
    pub fn with_endpoint(endpoint: impl Into<String>, language: impl Into<String>) -> Self {
        Self::new(
            Some(Arc::new(HttpSearchBackend::new(endpoint))),
            language,
        )
    }

    /// Builds the model-facing card in `language`.
    ///
    /// The description text lives in [`descriptions`] and is empty until the table gains the
    /// `web_search` row; the card itself is always complete.
    #[must_use]
    pub fn card_for(language: &str) -> ToolCard {
        tool_card(
            TOOL_NAME,
            descriptions::description(TOOL_NAME, language),
            schema_for::<WebSearchArgs>(),
            true,
        )
    }

    /// Runs one search and renders the payload.
    async fn run(&self, inputs: &Value) -> ToolPayload {
        let args: WebSearchArgs = match parse_args(inputs) {
            Ok(args) => args,
            Err(payload) => return payload,
        };
        let query = args.query.trim();
        if query.is_empty() {
            return ToolPayload::fail("query cannot be empty");
        }
        let max_results = resolve_max_results(args.max_results);
        let Some(backend) = self.backend.as_ref() else {
            return ToolPayload::fail(not_configured_reason());
        };
        match backend.search(query, max_results).await {
            Ok(results) => ToolPayload::ok(json!({"success": true, "results": results})),
            // A configuration gap is reported as its own reason; a provider that is configured
            // but fails keeps the `web search failed:` prefix and its error kind.
            Err(Error::Unsupported(message)) => ToolPayload::fail(message),
            Err(error) => ToolPayload::fail(format!("web search failed: {error}")),
        }
    }
}

#[async_trait]
impl Tool for WebSearchTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value> {
        ctx.check_cancelled()?;
        Ok(self.run(&inputs).await.to_json())
    }
}

/// Clamps the requested result count to `MIN_MAX_RESULTS..=MAX_MAX_RESULTS`.
fn resolve_max_results(value: Option<u64>) -> usize {
    value.map_or(DEFAULT_MAX_RESULTS, |value| {
        usize::try_from(value)
            .unwrap_or(DEFAULT_MAX_RESULTS)
            .clamp(MIN_MAX_RESULTS, MAX_MAX_RESULTS)
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::tools::web::test_support::StubFetcher;

    /// Invokes the tool with a fresh context and returns the payload envelope.
    async fn invoke(tool: &WebSearchTool, inputs: Value) -> Value {
        tool.invoke(inputs, &ToolContext::new()).await.unwrap()
    }

    #[test]
    fn card_exposes_the_python_arguments() {
        let card = WebSearchTool::card_for("en");
        assert_eq!(card.name, TOOL_NAME);
        assert!(card.parallel_safe);
        assert_eq!(card.input_params["required"], json!(["query"]));
        let properties = card.input_params["properties"].as_object().unwrap();
        let mut keys: Vec<&str> = properties.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(keys, ["max_results", "query"]);
        assert_eq!(properties["max_results"]["type"], json!("integer"));
    }

    #[test]
    fn clamps_the_requested_result_count() {
        assert_eq!(resolve_max_results(None), DEFAULT_MAX_RESULTS);
        assert_eq!(resolve_max_results(Some(0)), MIN_MAX_RESULTS);
        assert_eq!(resolve_max_results(Some(5)), 5);
        assert_eq!(resolve_max_results(Some(100)), MAX_MAX_RESULTS);
    }

    #[test]
    fn percent_encodes_query_components() {
        assert_eq!(percent_encode("rust async"), "rust%20async");
        assert_eq!(percent_encode("a&b/c"), "a%26b%2Fc");
        assert_eq!(percent_encode("é"), "%C3%A9");
        assert_eq!(percent_encode("a-b_c.d~e"), "a-b_c.d~e");
    }

    #[test]
    fn parses_object_and_array_payloads() {
        let object = r#"{"results":[{"title":"A","url":"https://a.example","snippet":"sa"},
            {"title":"B","url":"https://b.example","snippet":"sb"}]}"#;
        let results = parse_search_response(object).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[1].title, "B");
        assert_eq!(results[0].url, "https://a.example");

        let array = r#"[{"title":"Only","url":"https://c.example","snippet":"sc"}]"#;
        assert_eq!(parse_search_response(array).unwrap().len(), 1);
    }

    #[test]
    fn parses_alternate_field_names_and_skips_entries_without_urls() {
        let payload = r#"{"results":[
            {"name":"Named","link":"https://d.example","description":"desc"},
            {"title":"No URL"},
            {"title":"  ","url":"  ","snippet":"blank"}
        ]}"#;
        let results = parse_search_response(payload).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "Named");
        assert_eq!(results[0].url, "https://d.example");
        assert_eq!(results[0].snippet, "desc");
    }

    #[test]
    fn rejects_malformed_payloads() {
        assert!(parse_search_response("not json")
            .unwrap_err()
            .to_string()
            .contains("invalid JSON"));
        assert!(parse_search_response(r#"{"results":{}}"#)
            .unwrap_err()
            .to_string()
            .contains("non-array"));
        assert!(parse_search_response(r#"{"hits":[]}"#)
            .unwrap_err()
            .to_string()
            .contains("no 'results' array"));
        assert!(parse_search_response("42")
            .unwrap_err()
            .to_string()
            .contains("not a JSON object or array"));
    }

    #[tokio::test]
    async fn http_backend_builds_the_request_and_parses_results() {
        let fetcher = StubFetcher::ok(
            "application/json",
            r#"{"results":[{"title":"A","url":"https://a.example","snippet":"sa"},
                {"title":"B","url":"https://b.example","snippet":"sb"}]}"#,
        );
        let backend =
            HttpSearchBackend::with_fetcher("https://api.example/search", fetcher.clone())
                .with_timeout(Duration::from_secs(3))
                .with_max_bytes(2048);
        let results = backend.search("rust async", 1).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "A");

        let request = fetcher.last_request().unwrap();
        assert_eq!(
            request.url,
            "https://api.example/search?q=rust%20async&max_results=1"
        );
        assert_eq!(request.timeout, Duration::from_secs(3));
        assert_eq!(request.max_bytes, 2048);
        assert!(request
            .headers
            .iter()
            .any(|(name, value)| name == "Accept" && value == "application/json"));
    }

    #[tokio::test]
    async fn http_backend_supports_query_templates_and_existing_parameters() {
        let templated_fetcher = StubFetcher::ok("application/json", "[]");
        let templated = HttpSearchBackend::with_fetcher(
            "https://x.example/s?text={query}",
            templated_fetcher.clone(),
        );
        let results = templated.search("a b", 3).await.unwrap();
        assert_eq!(results, Vec::new());
        assert_eq!(
            templated_fetcher.last_request().unwrap().url,
            "https://x.example/s?text=a%20b"
        );

        let parameterized_fetcher = StubFetcher::ok("application/json", "[]");
        let parameterized = HttpSearchBackend::with_fetcher(
            "https://y.example/s?key=k",
            parameterized_fetcher.clone(),
        );
        let _ = parameterized.search("q", 2).await.unwrap();
        assert_eq!(
            parameterized_fetcher.last_request().unwrap().url,
            "https://y.example/s?key=k&q=q&max_results=2"
        );
    }

    #[tokio::test]
    async fn http_backend_reports_http_and_payload_failures() {
        let failing = StubFetcher::new(Ok(crate::tools::web::test_support::fetch_response(
            500,
            "application/json",
            "{}",
        )));
        let backend = HttpSearchBackend::with_fetcher("https://x.example/s", failing);
        let error = backend.search("q", 2).await.unwrap_err();
        assert!(error.to_string().contains("HTTP 500"), "error: {error}");

        let malformed = StubFetcher::ok("application/json", "not json");
        let backend = HttpSearchBackend::with_fetcher("https://x.example/s", malformed);
        let error = backend.search("q", 2).await.unwrap_err();
        assert!(error.to_string().contains("invalid JSON"), "error: {error}");
    }

    #[tokio::test]
    async fn tool_returns_the_documented_payload() {
        let backend = Arc::new(FakeSearchBackend::with_results(vec![
            SearchResult::new("First", "https://a.example", "one"),
            SearchResult::new("Second", "https://b.example", "two"),
        ]));
        let tool = WebSearchTool::new(Some(backend.clone()), "en");
        let payload = invoke(&tool, json!({"query": "  rust  ", "max_results": 100})).await;
        assert_eq!(payload["success"], json!(true));
        assert_eq!(
            payload["data"],
            json!({
                "success": true,
                "results": [
                    {"title": "First", "url": "https://a.example", "snippet": "one"},
                    {"title": "Second", "url": "https://b.example", "snippet": "two"},
                ],
            })
        );
        assert_eq!(payload["error"], Value::Null);
        assert_eq!(
            backend.last_call(),
            Some(FakeSearchCall {
                query: "rust".to_owned(),
                max_results: MAX_MAX_RESULTS,
            })
        );
    }

    #[tokio::test]
    async fn tool_reports_a_backend_failure() {
        let backend = Arc::new(FakeSearchBackend::failing("provider is down"));
        let tool = WebSearchTool::new(Some(backend), "en");
        let payload = invoke(&tool, json!({"query": "rust"})).await;
        assert_eq!(payload["success"], json!(false));
        assert_eq!(
            payload["error"],
            json!("web search failed: tool: provider is down")
        );
    }

    #[test]
    fn the_configuration_gap_names_the_environment_variable_and_constructors() {
        let reason = not_configured_reason();
        assert!(
            reason.starts_with("web search backend is not configured"),
            "reason: {reason}"
        );
        for needle in [
            ENDPOINT_ENV,
            "{query}",
            "HttpSearchBackend::new",
            "WebSearchTool::with_endpoint",
            "with_search_backend",
            "not ported",
        ] {
            assert!(reason.contains(needle), "reason: {reason} (misses {needle})");
        }
    }

    #[test]
    fn endpoint_lookup_reads_and_trims_the_documented_variable() {
        let backend =
            HttpSearchBackend::from_lookup(|key| {
                (key == ENDPOINT_ENV).then(|| "  https://api.example/s  ".to_owned())
            })
            .unwrap();
        assert!(backend.is_configured());
        assert_eq!(backend.endpoint, "https://api.example/s");

        // Unset, blank and non-matching keys are the same typed configuration gap.
        for lookup in [
            None,
            Some(String::new()),
            Some("   ".to_owned()),
        ] {
            let error = HttpSearchBackend::from_lookup(|_| lookup.clone())
                .err()
                .unwrap();
            assert!(matches!(error, Error::Unsupported(_)), "error: {error:?}");
            assert_eq!(&*error.message(), not_configured_reason().as_str());
        }
        let error = HttpSearchBackend::from_lookup(|_| None).err().unwrap();
        assert_eq!(&*error.message(), not_configured_reason().as_str());
    }

    #[tokio::test]
    async fn a_blank_endpoint_is_a_configuration_gap_not_a_request() {
        let fetcher = StubFetcher::ok("application/json", "[]");
        let backend = HttpSearchBackend::with_fetcher("   ", fetcher.clone());
        assert!(!backend.is_configured());
        let error = backend.search("rust", 2).await.unwrap_err();
        assert!(matches!(error, Error::Unsupported(_)), "error: {error:?}");
        assert_eq!(&*error.message(), not_configured_reason().as_str());
        assert_eq!(fetcher.calls(), 0, "no request is issued for a blank endpoint");
    }

    /// A provider that is configured but cannot serve the deployment (for example a vendor SDK
    /// whose credentials are missing) reports `Unsupported`, which the tool renders bare.
    struct UnsupportedBackend;

    #[async_trait]
    impl SearchBackend for UnsupportedBackend {
        async fn search(&self, _query: &str, _max_results: usize) -> Result<Vec<SearchResult>> {
            Err(Error::unsupported(
                "web search provider requires a vendor API key",
            ))
        }
    }

    #[tokio::test]
    async fn the_tool_renders_configuration_gaps_differently_from_runtime_failures() {
        // No backend at all: the configuration gap, bare.
        let payload = invoke(&WebSearchTool::new(None, "en"), json!({"query": "rust"})).await;
        assert_eq!(payload["error"], json!(not_configured_reason()));

        // A configured endpoint that is blank: the same gap, still bare.
        let tool = WebSearchTool::with_endpoint("   ", "en");
        let payload = invoke(&tool, json!({"query": "rust"})).await;
        assert_eq!(payload["error"], json!(not_configured_reason()));

        // A provider that is configured but unsupported: its own reason, also bare.
        let tool = WebSearchTool::new(Some(Arc::new(UnsupportedBackend)), "en");
        let payload = invoke(&tool, json!({"query": "rust"})).await;
        assert_eq!(
            payload["error"],
            json!("web search provider requires a vendor API key")
        );

        // A runtime failure keeps the `web search failed:` prefix and its error kind.
        let tool = WebSearchTool::new(Some(Arc::new(FakeSearchBackend::failing("boom"))), "en");
        let payload = invoke(&tool, json!({"query": "rust"})).await;
        assert_eq!(payload["error"], json!("web search failed: tool: boom"));
    }

    #[tokio::test]
    async fn tool_without_a_backend_or_query_fails() {
        let tool = WebSearchTool::new(None, "en");
        let payload = invoke(&tool, json!({"query": "rust"})).await;
        assert_eq!(payload["error"], json!(not_configured_reason()));

        let backend = Arc::new(FakeSearchBackend::new());
        let tool = WebSearchTool::new(Some(backend), "en");
        let payload = invoke(&tool, json!({"query": "   "})).await;
        assert_eq!(payload["error"], json!("query cannot be empty"));

        let payload = invoke(&tool, json!([])).await;
        assert_eq!(payload["error"], json!("tool arguments must be an object"));
    }

    #[tokio::test]
    async fn fake_backend_records_calls_and_can_be_reprogrammed() {
        let backend = FakeSearchBackend::new();
        backend.set_results(vec![SearchResult::new("A", "https://a.example", "")]);
        assert_eq!(backend.search("first", 3).await.unwrap().len(), 1);

        backend.set_failure(Some("boom".to_owned()));
        assert!(backend.search("second", 2).await.is_err());
        assert_eq!(backend.calls().len(), 2);
        assert_eq!(backend.calls()[1].query, "second");

        backend.set_failure(None);
        assert_eq!(backend.search("third", 1).await.unwrap().len(), 1);
    }
}
